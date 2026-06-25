#!/usr/bin/env python3
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

"""ClearML pipeline-step client for the Cosmos Transfer API.

This is a thin, GPU-free client: it submits a generation job to a running Cosmos Transfer API
server, polls until it finishes, downloads the generated videos, and (when ClearML is
available) registers them as artifacts and logs the parameters. It deliberately does NOT import
``cosmos_transfer2`` so it can run on a lightweight ClearML agent without CUDA.

Use as a pipeline step:

    from scripts.clearml_step import run_cosmos_transfer_step
    outputs = run_cosmos_transfer_step(
        api_url="http://cosmos-api:8000",
        video_path="/data/episode_000000.mp4",   # path the *API server* can read
        prompt="A robot arm places a game piece on the board.",
        view="wrist",
        styles=["warm_indoor", "cool_daylight"],   # or None for the 4 defaults
        num_steps=35,
        out_dir="/data/cosmos_out",
    )

Or from the CLI:

    python scripts/clearml_step.py --api-url http://localhost:8000 \
        --video /data/ep0.mp4 --prompt "..." --view wrist --out-dir ./cosmos_out
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

POLL_INTERVAL_S = 5.0
DEFAULT_TIMEOUT_S = 3600.0


def _build_payload(
    prompt: str,
    *,
    video_path: str | None,
    video_url: str | None,
    view: str | None,
    styles: list | None,
    num_samples: int | None,
    num_steps: int | None,
    guidance: int | None,
    seed: int | None,
    controls: dict | None,
    upsample: bool,
    extra: dict | None,
) -> dict:
    payload: dict[str, Any] = {"prompt": prompt}
    if video_path:
        payload["video_path"] = video_path
    if video_url:
        payload["video_url"] = video_url
    if view:
        payload["view"] = view
    if styles is not None:
        payload["styles"] = styles
    if num_samples is not None:
        payload["num_samples"] = num_samples
    if num_steps is not None:
        payload["num_steps"] = num_steps
    if guidance is not None:
        payload["guidance"] = guidance
    if seed is not None:
        payload["seed"] = seed
    if controls is not None:
        payload["controls"] = controls
    if upsample:
        payload["upsample"] = True
    if extra:
        payload.update(extra)
    return payload


def submit_and_wait(
    api_url: str,
    payload: dict,
    *,
    endpoint: str = "/generate",
    uploads: dict[str, str] | None = None,
    poll_interval_s: float = POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Submit a job, poll until terminal, and return the final job status dict.

    ``uploads`` maps multipart field name -> local file path (e.g. {"video": ...} for
    /generate, or {"top": ..., "wrist": ...} for /generate/dual_view). If None, posts JSON.
    """
    api_url = api_url.rstrip("/")
    url = f"{api_url}{endpoint}"
    handles = []
    try:
        if uploads:
            files = {}
            for field, path in uploads.items():
                fh = open(path, "rb")
                handles.append(fh)
                files[field] = (Path(path).name, fh, "video/mp4")
            resp = requests.post(url, files=files, data={"request": json.dumps(payload)})
        else:
            resp = requests.post(url, json=payload)
    finally:
        for fh in handles:
            fh.close()
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"[cosmos] submitted job {job_id}")

    deadline = time.monotonic() + timeout_s
    last_state = None
    while time.monotonic() < deadline:
        status = requests.get(f"{api_url}/jobs/{job_id}").json()
        state = status["state"]
        if state != last_state:
            pos = status.get("queue_position")
            extra = f" (queue position {pos})" if state == "queued" and pos is not None else ""
            print(f"[cosmos] job {job_id}: {state}{extra}")
            last_state = state
        if state in ("succeeded", "failed", "cancelled"):
            if state != "succeeded":
                raise RuntimeError(f"job {job_id} {state}: {status.get('error')}")
            return status
        time.sleep(poll_interval_s)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def download_results(api_url: str, job_id: str, out_dir: str) -> list[Path]:
    """Download the job's result zip and extract the videos into ``out_dir``."""
    api_url = api_url.rstrip("/")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{job_id}_results.zip"
    with requests.get(f"{api_url}/jobs/{job_id}/results", stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            zf.extract(name, out)
            extracted.append(out / name)
    zip_path.unlink(missing_ok=True)
    print(f"[cosmos] downloaded {len(extracted)} file(s) to {out}")
    return extracted


def run_cosmos_transfer_step(
    api_url: str,
    prompt: str,
    out_dir: str,
    *,
    video_path: str | None = None,
    video_url: str | None = None,
    upload_file: str | None = None,
    view: str | None = None,
    styles: list | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    guidance: int | None = None,
    seed: int | None = None,
    controls: dict | None = None,
    upsample: bool = False,
    extra_params: dict | None = None,
    use_clearml: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Run one Cosmos Transfer generation as a (optionally ClearML-logged) pipeline step.

    Returns ``{"job_id", "status", "videos": [paths], "manifest": {...}}``.
    """
    payload = _build_payload(
        prompt,
        video_path=video_path,
        video_url=video_url,
        view=view,
        styles=styles,
        num_samples=num_samples,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        controls=controls,
        upsample=upsample,
        extra=extra_params,
    )

    task = None
    if use_clearml:
        try:
            from clearml import Task

            task = Task.current_task()
            if task is not None:
                task.connect(payload, name="cosmos_transfer_request")
        except Exception as e:  # ClearML optional; never block the step on it
            print(f"[cosmos] ClearML not active ({e}); proceeding without logging.")

    uploads = {"video": upload_file} if upload_file else None
    status = submit_and_wait(api_url, payload, uploads=uploads, timeout_s=timeout_s)
    files = download_results(api_url, status["id"], out_dir)
    videos = [p for p in files if p.suffix == ".mp4"]
    manifest = {}
    manifest_file = next((p for p in files if p.name == "manifest.json"), None)
    if manifest_file is not None:
        manifest = json.loads(manifest_file.read_text())

    if task is not None:
        try:
            for v in videos:
                task.upload_artifact(name=v.stem, artifact_object=str(v))
            task.connect(manifest, name="cosmos_transfer_manifest")
        except Exception as e:
            print(f"[cosmos] artifact logging failed ({e}).")

    return {"job_id": status["id"], "status": status, "videos": videos, "manifest": manifest}


def run_cosmos_transfer_dual_view_step(
    api_url: str,
    prompt: str,
    out_dir: str,
    *,
    top_path: str | None = None,
    wrist_path: str | None = None,
    top_url: str | None = None,
    wrist_url: str | None = None,
    top_upload: str | None = None,
    wrist_upload: str | None = None,
    stack: str = "vstack",
    view: str | None = None,
    styles: list | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    guidance: int | None = None,
    seed: int | None = None,
    control_preset: str | None = None,
    controls: dict | None = None,
    guided_generation: bool = False,
    upsample: bool = False,
    use_clearml: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Restyle top+wrist of one episode together (style-consistent) via /generate/dual_view.

    Provide each view as a server path, a URL, or a local file to upload. Returns
    ``{"job_id", "status", "videos": [paths], "manifest": {...}}`` (videos include per-view files).
    """
    payload: dict = {"prompt": prompt, "stack": stack}
    if top_path:
        payload["top_path"] = top_path
    if wrist_path:
        payload["wrist_path"] = wrist_path
    if top_url:
        payload["top_url"] = top_url
    if wrist_url:
        payload["wrist_url"] = wrist_url
    for k, v in (
        ("view", view), ("styles", styles), ("num_samples", num_samples),
        ("num_steps", num_steps), ("guidance", guidance), ("seed", seed),
        ("control_preset", control_preset), ("controls", controls),
    ):
        if v is not None:
            payload[k] = v
    if guided_generation:
        payload["guided_generation"] = True
    if upsample:
        payload["upsample"] = True

    task = None
    if use_clearml:
        try:
            from clearml import Task

            task = Task.current_task()
            if task is not None:
                task.connect(payload, name="cosmos_transfer_dual_view_request")
        except Exception as e:
            print(f"[cosmos] ClearML not active ({e}); proceeding without logging.")

    uploads = None
    if top_upload and wrist_upload:
        uploads = {"top": top_upload, "wrist": wrist_upload}
    status = submit_and_wait(
        api_url, payload, endpoint="/generate/dual_view", uploads=uploads, timeout_s=timeout_s
    )
    files = download_results(api_url, status["id"], out_dir)
    videos = [p for p in files if p.suffix == ".mp4"]
    manifest = {}
    manifest_file = next((p for p in files if p.name == "manifest.json"), None)
    if manifest_file is not None:
        manifest = json.loads(manifest_file.read_text())

    if task is not None:
        try:
            for v in videos:
                task.upload_artifact(name=v.stem, artifact_object=str(v))
            task.connect(manifest, name="cosmos_transfer_dual_view_manifest")
        except Exception as e:
            print(f"[cosmos] artifact logging failed ({e}).")

    return {"job_id": status["id"], "status": status, "videos": videos, "manifest": manifest}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-url", required=True, help="Base URL of the Cosmos Transfer API.")
    p.add_argument("--prompt", required=True, help="Action prompt for the original video.")
    p.add_argument("--out-dir", required=True, help="Local directory to download results into.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--video-path", help="Single-view: path the API server can read directly.")
    src.add_argument("--video-url", help="Single-view: URL the API server downloads from.")
    src.add_argument("--upload", help="Single-view: local file to upload to the API (multipart).")
    p.add_argument("--dual-view", action="store_true", help="Use /generate/dual_view (top+wrist).")
    p.add_argument("--top-path", default=None)
    p.add_argument("--wrist-path", default=None)
    p.add_argument("--top-url", default=None)
    p.add_argument("--wrist-url", default=None)
    p.add_argument("--stack", default="vstack", choices=["vstack", "hstack"])
    p.add_argument("--control-preset", default=None, choices=["multicontrol_robot", "edge"])
    p.add_argument("--guided-generation", action="store_true")
    p.add_argument("--view", default=None)
    p.add_argument("--styles", default=None, help="Comma-separated style names.")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--guidance", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--controls", default=None, help="JSON control config, e.g. '{\"edge\":{}, \"depth\":{\"control_weight\":0.5}}'.")
    p.add_argument("--upsample", action="store_true")
    p.add_argument("--no-clearml", action="store_true", help="Disable ClearML logging.")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    styles = [s.strip() for s in a.styles.split(",")] if a.styles else None
    controls = json.loads(a.controls) if a.controls else None
    if a.dual_view:
        result = run_cosmos_transfer_dual_view_step(
            api_url=a.api_url, prompt=a.prompt, out_dir=a.out_dir,
            top_path=a.top_path, wrist_path=a.wrist_path,
            top_url=a.top_url, wrist_url=a.wrist_url, stack=a.stack,
            view=a.view, styles=styles, num_samples=a.num_samples,
            num_steps=a.num_steps, guidance=a.guidance, seed=a.seed,
            control_preset=a.control_preset, controls=controls,
            guided_generation=a.guided_generation, upsample=a.upsample,
            use_clearml=not a.no_clearml, timeout_s=a.timeout,
        )
    else:
        if not (a.video_path or a.video_url or a.upload):
            raise SystemExit("single-view requires one of --video-path/--video-url/--upload (or use --dual-view)")
        result = run_cosmos_transfer_step(
            api_url=a.api_url, prompt=a.prompt, out_dir=a.out_dir,
            video_path=a.video_path, video_url=a.video_url, upload_file=a.upload,
            view=a.view, styles=styles, num_samples=a.num_samples,
            num_steps=a.num_steps, guidance=a.guidance, seed=a.seed,
            controls=controls, upsample=a.upsample,
            use_clearml=not a.no_clearml, timeout_s=a.timeout,
        )
    print(json.dumps({"job_id": result["job_id"], "videos": [str(v) for v in result["videos"]]}, indent=2))


if __name__ == "__main__":
    main()
