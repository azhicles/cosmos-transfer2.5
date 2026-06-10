# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI server exposing Cosmos Transfer as an async, job-based HTTP API.

Run with: ``uvicorn cosmos_transfer2.api.server:app --host 0.0.0.0 --port 8000``.
See ``API_GUIDE.md`` at the repository root for the full reference.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen

import pydantic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from cosmos_transfer2.api.config_models import (
    GenerateRequest,
    GenerateResponse,
    JobState,
    JobStatus,
)
from cosmos_transfer2.api.jobs import JobManager
from cosmos_transfer2.api.logging_setup import configure_logging
from cosmos_transfer2.api.styles import load_style_library
from cosmos_transfer2.api.views import load_view_map

log = configure_logging()


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).resolve()


OUTPUT_ROOT = _env_path("COSMOS_API_OUTPUT_DIR", "outputs/api")
WORK_DIR = _env_path("COSMOS_API_WORK_DIR", str(OUTPUT_ROOT / "_engine"))
INPUTS_DIR = _env_path("COSMOS_API_INPUTS_DIR", str(OUTPUT_ROOT / "_inputs"))

manager = JobManager(OUTPUT_ROOT, WORK_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    preload = os.environ.get("COSMOS_API_PRELOAD", "edge")
    preload_controls = (
        None if preload.strip().lower() in ("", "none") else [c.strip() for c in preload.split(",")]
    )
    manager.start(preload_controls=preload_controls)
    log.info("Cosmos Transfer API ready. Output root: %s", OUTPUT_ROOT)
    yield


app = FastAPI(title="Cosmos Transfer API", version="0.1.0", lifespan=lifespan)


# --- input resolution ------------------------------------------------------------


def _save_upload(filename: str, data: bytes) -> Path:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "input.mp4").suffix or ".mp4"
    dest = INPUTS_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    dest.write_bytes(data)
    return dest


def _download_url(url: str) -> Path:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?")[0]).suffix or ".mp4"
    dest = INPUTS_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    with urlopen(url) as resp, open(dest, "wb") as f:  # noqa: S310 (trusted internal use)
        shutil.copyfileobj(resp, f)
    return dest


def _resolve_input(req: GenerateRequest, upload: tuple[str, bytes] | None) -> Path:
    if upload is not None:
        return _save_upload(upload[0], upload[1])
    if req.video_path:
        p = Path(req.video_path).expanduser()
        if not p.is_file():
            raise HTTPException(400, f"video_path not found on server: {p}")
        return p.resolve()
    if req.video_url:
        try:
            return _download_url(req.video_url)
        except Exception as e:
            raise HTTPException(400, f"failed to download video_url: {e}")
    raise HTTPException(400, "provide one of: file upload, video_path, or video_url")


def _job_or_404(job_id: str) -> JobStatus:
    status = manager.get(job_id)
    if status is not None:
        return status
    # Fall back to a persisted job.json (e.g. after a restart).
    persisted = manager.job_dir(job_id) / "job.json"
    if persisted.is_file():
        return JobStatus.model_validate_json(persisted.read_text())
    raise HTTPException(404, f"job not found: {job_id}")


# --- endpoints -------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": manager.engine.is_loaded,
        "loaded_controls": manager.engine.loaded_control_keys,
    }


@app.get("/styles")
def get_styles() -> dict:
    lib = load_style_library(os.environ.get("COSMOS_API_STYLES_FILE"))
    return {name: entry.model_dump(exclude_none=True) for name, entry in lib.items()}


@app.get("/views")
def get_views() -> dict:
    return load_view_map(os.environ.get("COSMOS_API_VIEWS_FILE"))


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request) -> GenerateResponse:
    """Submit a generation job. Accepts JSON (with ``video_path``/``video_url``) or
    multipart/form-data (a ``video`` file plus a ``request`` JSON field)."""
    content_type = request.headers.get("content-type", "")
    upload: tuple[str, bytes] | None = None

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        raw = form.get("request")
        payload = json.loads(raw) if raw else {}
        video = form.get("video")
        if video is not None and hasattr(video, "read"):
            upload = (getattr(video, "filename", "input.mp4"), await video.read())
    else:
        payload = await request.json()

    try:
        req = GenerateRequest.model_validate(payload)
    except pydantic.ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))

    input_video = _resolve_input(req, upload)
    status = manager.submit(req, input_video)
    return GenerateResponse(job_id=status.id, state=status.state, queue_position=status.queue_position)


@app.get("/jobs", response_model=list[JobStatus])
def list_jobs() -> list[JobStatus]:
    return manager.list_jobs()


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    return _job_or_404(job_id)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> JSONResponse:
    _job_or_404(job_id)
    cancelled = manager.cancel(job_id)
    if not cancelled:
        raise HTTPException(409, "job is not queued (already running, finished, or cancelled)")
    return JSONResponse({"id": job_id, "state": JobState.CANCELLED.value})


@app.get("/jobs/{job_id}/results")
def download_results(job_id: str):
    """Download all output videos (plus manifest) for a finished job as a zip."""
    status = _job_or_404(job_id)
    if status.state != JobState.SUCCEEDED:
        raise HTTPException(409, f"job not finished (state={status.state.value})")
    job_dir = manager.job_dir(job_id)
    tmp = Path(tempfile.gettempdir()) / f"{job_id}_results.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for sample in status.samples:
            if sample.output_file:
                f = job_dir / sample.output_file
                if f.is_file():
                    zf.write(f, sample.output_file)
        manifest = job_dir / "manifest.json"
        if manifest.is_file():
            zf.write(manifest, "manifest.json")
    return FileResponse(tmp, media_type="application/zip", filename=f"{job_id}_results.zip")


@app.get("/jobs/{job_id}/results/{index}")
def download_one(job_id: str, index: int):
    """Download a single output video by sample index."""
    status = _job_or_404(job_id)
    if status.state != JobState.SUCCEEDED:
        raise HTTPException(409, f"job not finished (state={status.state.value})")
    match = next((s for s in status.samples if s.index == index), None)
    if match is None or not match.output_file:
        raise HTTPException(404, f"no output for sample index {index}")
    f = manager.job_dir(job_id) / match.output_file
    if not f.is_file():
        raise HTTPException(404, f"output file missing on disk: {match.output_file}")
    return FileResponse(f, media_type="video/mp4", filename=match.output_file)
