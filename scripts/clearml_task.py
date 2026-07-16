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

"""ClearML Task: Cosmos Transfer domain-randomization via ``InferenceEngine`` (no HTTP server).

This runs Cosmos Transfer *in-process* as a single tracked ClearML Task, instead of talking to a
separate HTTP API. ClearML already provides the queue, job status, remote execution, console
logging and artifact storage that the old ``cosmos_transfer2/api`` HTTP server re-implemented, so
here we call the engine's ``run_job`` directly and let ClearML own the orchestration.

Batch-first: it processes a LIST of episodes in one Task, so the (~1 min) checkpoint load is paid
ONCE and reused across every episode -- the same "warm model" benefit the persistent server gave,
without a server. The smoke test just passes a 1-episode list.

Monitoring: per-sample generation time, batch progress and failures are reported as ClearML
scalars (SCALARS tab), and each output video is attached as an artifact AND shown inline via
``report_media`` (DEBUG SAMPLES tab).

Run locally (creates a fully-tracked Task; no agent needed):

    uv run --no-sync python scripts/clearml_task.py

Requires ClearML credentials once (``clearml-init`` -> ~/clearml.conf). Edit PARAMS below, or
override any field from the ClearML GUI when cloning/enqueuing the Task.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from clearml import Task

# --- Parameters --------------------------------------------------------------------------------
# A plain dict so ClearML captures it (task.connect) and lets you override every field from the
# GUI. When the Task is executed by an agent, GUI edits are written back into this dict.
PARAMS: dict = {
    "clearml_project": "Cosmos Transfer",
    "clearml_task_name": "smoke-tictactoe-wrist-2style-6step",
    # BATCH of inputs. Smoke test = 1 episode; scale by appending more dicts.
    # Single-view episode: {"video_path": "...", "view": "wrist"}
    # Dual-view episode  : {"top_path": "...", "wrist_path": "...", "view": "wrist"}  (dual_view=True)
    "episodes": [
        {
            "video_path": "/home/demouser/robotics/DellRobot/tictactoe_data/videos/chunk-000/observation.images.wrist/episode_000001.mp4",
            "view": "wrist",
        },
    ],
    # Prompt: read from a file, or set "prompt" directly (takes precedence if non-null).
    "prompt_file": "assets/tictactoe/separate/wrist_prompt_v2.txt",
    "prompt": None,
    "styles": ["warm_indoor", "cool_daylight"],
    "num_steps": 6,  # fast for smoke; ~35 for production quality
    "control_preset": "edge",  # fast/lightweight; "balanced" (depth+seg+edge) for production
    "seed": 2025,
    "guidance": 5,
    "sigma_max": "110",
    "resolution": "720",
    "output_root": "outputs/clearml",
    # None -> artifacts go to the ClearML file server. Set to a dir / "s3://..." before scaling
    # to many/large videos (e.g. "/data/clearml_store").
    "output_uri": None,
    "dual_view": False,
    "stack": "vstack",
}


def _resolve_prompt(params: dict) -> str:
    if params.get("prompt"):
        return str(params["prompt"]).strip()
    return Path(params["prompt_file"]).read_text().strip()


def main() -> None:
    params = dict(PARAMS)  # local copy; task.connect will overlay any GUI overrides

    task = Task.init(
        project_name=params["clearml_project"],
        task_name=params["clearml_task_name"],
        output_uri=params["output_uri"],  # None = default ClearML file server
    )
    task.connect(params)
    logger = task.get_logger()

    # Route the engine's logger ("cosmos_api") to stdout so ClearML captures it to the console.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    log = logging.getLogger("clearml_task")

    # Import the engine lazily (heavy, CUDA) so ``Task.init`` / arg capture stay fast.
    from cosmos_transfer2.api.config_models import DualViewRequest, GenerateRequest
    from cosmos_transfer2.api.engine import InferenceEngine

    prompt = _resolve_prompt(params)
    out_root = Path(params["output_root"])
    engine = InferenceEngine(out_root / "_work")  # model loads ONCE, reused across all episodes

    common = dict(
        prompt=prompt,
        styles=params["styles"],
        num_steps=params["num_steps"],
        control_preset=params["control_preset"],
        seed=params["seed"],
        guidance=params["guidance"],
        sigma_max=params["sigma_max"],
        resolution=params["resolution"],
    )

    episodes = params["episodes"]
    log.info("Processing %d episode(s); model loads once and is reused.", len(episodes))
    done = 0
    for i, ep in enumerate(episodes):
        job_dir = out_root / task.id / f"episode_{i:03d}"
        try:
            if params["dual_view"]:
                req = DualViewRequest(view=ep.get("view"), stack=params["stack"], **common)
                result = engine.run_dual_view_job(
                    req, Path(ep["top_path"]), Path(ep["wrist_path"]), job_dir
                )
            else:
                req = GenerateRequest(video_path=ep["video_path"], view=ep.get("view"), **common)
                # control video generated once here and reused across the styles internally
                result = engine.run_job(req, Path(ep["video_path"]), job_dir)
        except Exception as e:  # one bad episode must not abort the whole batch
            log.exception("episode %d failed: %s", i, e)
            logger.report_scalar("failures", "episode", value=1, iteration=i)
            continue

        for s in result["samples"]:
            if s.generation_time_s is not None:
                logger.report_scalar("gen_time_s", s.style, value=s.generation_time_s, iteration=i)
            filenames = [s.output_file, *s.view_outputs.values()]
            for fname in filter(None, filenames):
                mp4 = job_dir / fname
                task.upload_artifact(name=f"ep{i:03d}_{fname}", artifact_object=str(mp4))
                logger.report_media(
                    "outputs", f"ep{i:03d}_{s.style}", local_path=str(mp4), iteration=i
                )
        manifest = job_dir / "manifest.json"
        if manifest.exists():
            task.upload_artifact(name=f"ep{i:03d}_manifest", artifact_object=str(manifest))
        done += 1
        logger.report_scalar("progress", "episodes_done", value=done, iteration=i)
        log.info("episode %d done (%d/%d).", i, done, len(episodes))

    print(f"[cosmos] done: {done}/{len(episodes)} episode(s)")


if __name__ == "__main__":
    main()
