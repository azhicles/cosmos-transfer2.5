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

"""Job queue and lifecycle management.

A single background worker drains a FIFO queue one job at a time (there is one GPU). Job state
lives in memory; each job's status is also persisted to ``<job_dir>/job.json`` so completed
results survive a server restart and can be re-served from disk.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from cosmos_transfer2.api.config_models import GenerateRequest, JobState, JobStatus
from cosmos_transfer2.api.engine import InferenceEngine
from cosmos_transfer2.api.logging_setup import job_log_file

log = logging.getLogger("cosmos_api")

_LOG_TAIL_CHARS = 4000


class JobManager:
    def __init__(self, output_root: Path, work_dir: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.engine = InferenceEngine(work_dir)
        self._jobs: dict[str, JobStatus] = {}
        # payload: {"req": GenerateRequest, "kind": "single"|"dual", "videos": dict[str, Path]}
        self._payloads: dict[str, dict] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._run_worker, name="cosmos-job-worker", daemon=True)
        self._started = False

    # --- lifecycle ---------------------------------------------------------------

    def start(self, preload_controls: list[str] | None = None) -> None:
        """Start the worker thread and optionally preload a default checkpoint."""
        if self._started:
            return
        self._started = True
        self._worker.start()
        if preload_controls:
            try:
                log.info("Preloading engine for controls %s ...", preload_controls)
                self.engine.ensure_loaded(preload_controls, disable_guardrails=True)
            except Exception as e:  # don't crash startup if preload fails
                log.warning("Engine preload failed (%s); will load on first request.", e)

    def job_dir(self, job_id: str) -> Path:
        return self.output_root / job_id

    # --- submission ---------------------------------------------------------------

    def submit(self, req: GenerateRequest, input_video: Path) -> JobStatus:
        """Queue a single-view job."""
        return self._enqueue(req, {"kind": "single", "videos": {"input": input_video}})

    def submit_dual(self, req: GenerateRequest, top_video: Path, wrist_video: Path) -> JobStatus:
        """Queue a dual-view (top+wrist) job."""
        return self._enqueue(req, {"kind": "dual", "videos": {"top": top_video, "wrist": wrist_video}})

    def _enqueue(self, req: GenerateRequest, payload: dict) -> JobStatus:
        job_id = uuid.uuid4().hex[:12]
        status = JobStatus(id=job_id, state=JobState.QUEUED, created_at=time.time())
        payload["req"] = req
        with self._lock:
            self._jobs[job_id] = status
            self._payloads[job_id] = payload
            status.queue_position = self._queue.qsize()
        self._queue.put(job_id)
        self._persist(job_id)
        log.info("Queued %s job %s (position %s).", payload["kind"], job_id, status.queue_position)
        return self._snapshot(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job (a running job cannot be preempted). Returns True if cancelled."""
        with self._lock:
            status = self._jobs.get(job_id)
            if status is None or status.state != JobState.QUEUED:
                return False
            self._cancelled.add(job_id)
            status.state = JobState.CANCELLED
            status.finished_at = time.time()
        self._persist(job_id)
        return True

    # --- queries ------------------------------------------------------------------

    def get(self, job_id: str) -> JobStatus | None:
        return self._snapshot(job_id)

    def list_jobs(self) -> list[JobStatus]:
        with self._lock:
            ids = list(self._jobs.keys())
        return [s for s in (self._snapshot(i) for i in ids) if s is not None]

    def _snapshot(self, job_id: str) -> JobStatus | None:
        with self._lock:
            status = self._jobs.get(job_id)
            if status is None:
                return None
            snap = status.model_copy(deep=True)
        if snap.state == JobState.QUEUED:
            snap.queue_position = self._queue_position(job_id)
        snap.log_tail = self._read_log_tail(job_id)
        return snap

    def _queue_position(self, job_id: str) -> int | None:
        with self._queue.mutex:
            try:
                return list(self._queue.queue).index(job_id)
            except ValueError:
                return None

    def _read_log_tail(self, job_id: str) -> str | None:
        log_path = self.job_dir(job_id) / "job.log"
        if not log_path.exists():
            return None
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            return None
        return text[-_LOG_TAIL_CHARS:]

    # --- persistence --------------------------------------------------------------

    def _persist(self, job_id: str) -> None:
        with self._lock:
            status = self._jobs.get(job_id)
            snap = status.model_copy(deep=True) if status else None
        if snap is None:
            return
        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(snap.model_dump_json(indent=2))

    # --- worker -------------------------------------------------------------------

    def _run_worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._process(job_id)
            except Exception:  # never let the worker thread die
                log.error("Worker crashed on job %s:\n%s", job_id, traceback.format_exc())
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._cancelled:
                log.info("Skipping cancelled job %s.", job_id)
                return
            status = self._jobs[job_id]
            payload = self._payloads[job_id]
            req = payload["req"]
            kind = payload["kind"]
            videos = payload["videos"]
            status.state = JobState.RUNNING
            status.started_at = time.time()
            status.queue_position = None
        self._persist(job_id)

        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        with job_log_file(d / "job.log"):
            log.info("=== Job %s started (%s) ===", job_id, kind)
            log.info("Request: %s", req.model_dump_json(exclude_none=True))
            try:
                if kind == "dual":
                    result = self.engine.run_dual_view_job(req, videos["top"], videos["wrist"], d)
                else:
                    result = self.engine.run_job(req, videos["input"], d)
                with self._lock:
                    status = self._jobs[job_id]
                    status.state = JobState.SUCCEEDED
                    status.finished_at = time.time()
                    status.base_prompt = result["base_prompt"]
                    status.upsampled_prompt = result["upsampled_prompt"]
                    status.controls = result["controls"]
                    status.samples = result["samples"]
                log.info("=== Job %s succeeded (%d samples) ===", job_id, len(result["samples"]))
            except Exception as e:
                tb = traceback.format_exc()
                log.error("Job %s failed: %s\n%s", job_id, e, tb)
                with self._lock:
                    status = self._jobs[job_id]
                    status.state = JobState.FAILED
                    status.finished_at = time.time()
                    status.error = f"{type(e).__name__}: {e}"
            finally:
                # free the (potentially large) input-video handle references
                self._payloads.pop(job_id, None)
        self._persist(job_id)
