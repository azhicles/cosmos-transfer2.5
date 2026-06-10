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

"""In-process wrapper around ``cosmos_transfer2.inference.Control2WorldInference``.

The engine loads a checkpoint once and keeps it resident. Because there is a single GPU, it
keeps exactly one live inference engine; if a request uses a different control set than what is
loaded, the engine is torn down and the right checkpoint reloaded.

A request's N style variations all share the same input video, so the control video (edge /
depth / etc.) is identical across them. The engine generates the control once on the first
sample and reuses it (via ``control_path``) for the rest — avoiding N expensive control passes,
which matters most for depth/seg multicontrol.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from cosmos_transfer2.api.config_models import (
    CONTROL_KEYS,
    GenerateRequest,
    SampleResult,
    StyleEntry,
)
from cosmos_transfer2.api.prompts import assemble_prompt
from cosmos_transfer2.api.styles import resolve_styles
from cosmos_transfer2.api.upsampler import get_upsampler
from cosmos_transfer2.api.views import resolve_view_hint

log = logging.getLogger("cosmos_api")

# Keys on a ControlSpec that map onto the cosmos ControlConfig fields, per control type.
_CONTROL_PASSTHROUGH = {
    "edge": ["control_weight", "mask_path", "mask_prompt", "preset_edge_threshold"],
    "vis": ["control_weight", "mask_path", "mask_prompt", "preset_blur_strength"],
    "depth": ["control_weight", "mask_path", "mask_prompt"],
    "seg": ["control_weight", "mask_path", "mask_prompt", "control_prompt"],
}


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "style"


def _sorted_control_keys(keys: list[str]) -> list[str]:
    """Sort control keys into the canonical CONTROL_KEYS order (required by the checkpoint
    selection logic in Control2WorldInference)."""
    return sorted(keys, key=lambda k: CONTROL_KEYS.index(k))


class InferenceEngine:
    """Singleton-style holder of the loaded Cosmos Transfer inference pipeline."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._inference = None  # Control2WorldInference | None
        self._loaded_keys: tuple[str, ...] | None = None
        self._loaded_guardrails: bool | None = None
        self._env_ready = False

    # --- lifecycle ---------------------------------------------------------------

    def _ensure_env(self) -> None:
        if self._env_ready:
            return
        # Keep the CUDA allocator from fragmenting across runs (mirrors run_tictactoe_unified).
        os.environ.setdefault(
            "PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512"
        )
        from cosmos_oss.init import init_environment, init_output_dir

        init_environment()
        init_output_dir(self.work_dir, profile=False)
        self._env_ready = True

    def ensure_loaded(self, control_keys: list[str], disable_guardrails: bool) -> None:
        """Make sure the engine for ``control_keys`` is the live one, (re)loading if needed."""
        keys = tuple(_sorted_control_keys(control_keys))
        if (
            self._inference is not None
            and self._loaded_keys == keys
            and self._loaded_guardrails == disable_guardrails
        ):
            return

        if self._inference is not None:
            log.info(
                "Switching control set %s -> %s; reloading checkpoint.",
                list(self._loaded_keys or ()),
                list(keys),
            )
            self._teardown()

        self._ensure_env()
        from cosmos_transfer2.config import SetupArguments
        from cosmos_transfer2.inference import Control2WorldInference

        # model="edge" is a valid base variant; for a single non-edge control the checkpoint is
        # chosen from the hint key, and for multicontrol the multibranch checkpoint is used --
        # in both cases args.model only governs the (always-False here) distilled flag.
        setup_args = SetupArguments(
            output_dir=self.work_dir,
            model="edge",
            disable_guardrails=disable_guardrails,
        )
        t0 = time.perf_counter()
        log.info("Loading inference engine for control set %s ...", list(keys))
        self._inference = Control2WorldInference(setup_args, batch_hint_keys=list(keys))
        self._loaded_keys = keys
        self._loaded_guardrails = disable_guardrails
        log.info("Engine loaded in %.1fs.", time.perf_counter() - t0)

    def _teardown(self) -> None:
        import gc

        import torch

        self._inference = None
        self._loaded_keys = None
        self._loaded_guardrails = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._inference is not None

    @property
    def loaded_control_keys(self) -> list[str]:
        return list(self._loaded_keys or ())

    # --- request -> InferenceArguments -------------------------------------------

    @staticmethod
    def _active_control_keys(req: GenerateRequest) -> list[str]:
        if not req.controls:
            return ["edge"]
        return _sorted_control_keys(list(req.controls.keys()))

    def _build_control_dict(
        self,
        req: GenerateRequest,
        style: StyleEntry,
        control_paths: dict[str, str] | None,
    ) -> dict[str, dict]:
        """Build the per-control sub-dicts for one sample's InferenceArguments."""
        keys = self._active_control_keys(req)
        weight_overrides = (style.overrides.control_weights if style.overrides else None) or {}
        out: dict[str, dict] = {}
        for key in keys:
            spec = req.controls[key] if req.controls and key in req.controls else None
            cfg: dict = {}
            if spec is not None:
                for field in _CONTROL_PASSTHROUGH[key]:
                    val = getattr(spec, field, None)
                    if val is not None:
                        cfg[field] = val
            cfg.setdefault("control_weight", 1.0)
            if key in weight_overrides:
                cfg["control_weight"] = weight_overrides[key]
            if control_paths and key in control_paths:
                cfg["control_path"] = control_paths[key]
            out[key] = cfg
        return out

    def _build_sample_dict(
        self,
        req: GenerateRequest,
        style: StyleEntry,
        index: int,
        video_path: Path,
        resolved_prompt: str,
        seed: int,
        control_paths: dict[str, str] | None,
    ) -> dict:
        sample: dict = {
            "name": f"sample_{index:02d}_{_safe_slug(style.name)}",
            "video_path": str(video_path.resolve()),
            "prompt": resolved_prompt,
            "seed": seed,
            "guidance": (style.overrides.guidance if style.overrides and style.overrides.guidance is not None else req.guidance),
            "num_steps": (style.overrides.num_steps if style.overrides and style.overrides.num_steps is not None else req.num_steps),
            "resolution": req.resolution,
            "num_video_frames_per_chunk": req.num_video_frames_per_chunk,
            "keep_input_resolution": req.keep_input_resolution,
        }
        sigma_max = (style.overrides.sigma_max if style.overrides and style.overrides.sigma_max is not None else req.sigma_max)
        if sigma_max is not None:
            sample["sigma_max"] = sigma_max
        if req.max_frames is not None:
            sample["max_frames"] = req.max_frames

        negative = style.negative_prompt or req.negative_prompt
        if negative is not None:
            sample["negative_prompt"] = negative

        sample.update(self._build_control_dict(req, style, control_paths))
        return sample

    # --- run a job ----------------------------------------------------------------

    def run_job(self, req: GenerateRequest, input_video: Path, job_dir: Path) -> dict:
        """Generate all variations for one request.

        Returns a dict with keys: ``base_prompt``, ``upsampled_prompt``, ``controls``,
        ``samples`` (list[SampleResult]). Writes the videos, control videos, sidecars,
        ``manifest.json`` and ``metrics.log`` into ``job_dir``.
        """
        from cosmos_transfer2.config import InferenceArguments

        job_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve styles / view / prompt.
        styles_path = os.environ.get("COSMOS_API_STYLES_FILE")
        views_path = os.environ.get("COSMOS_API_VIEWS_FILE")
        styles = resolve_styles(req.styles, req.resolved_num_samples(), styles_path)
        view_hint = resolve_view_hint(req.view, req.view_hint, views_path)
        log.info("Resolved %d style(s): %s", len(styles), [s.name for s in styles])
        if view_hint:
            log.info("View hint: %s", view_hint)

        base_prompt = req.prompt
        upsampled_prompt = None
        if req.upsample:
            upsampled = get_upsampler(True)(req.prompt, model=req.upsample_model)
            if upsampled != req.prompt:
                upsampled_prompt = upsampled
        effective_base = upsampled_prompt or base_prompt

        control_keys = self._active_control_keys(req)
        log.info("Active controls: %s", control_keys)
        self.ensure_loaded(control_keys, req.disable_guardrails)
        assert self._inference is not None

        # 2. Generate sample 0 (control generated on-the-fly), then reuse the control for the rest.
        results: list[SampleResult] = []
        control_paths: dict[str, str] = {}

        def _gen_one(index: int, style: StyleEntry, reuse: dict[str, str] | None) -> SampleResult:
            seed = (
                style.overrides.seed
                if style.overrides and style.overrides.seed is not None
                else req.seed + index
            )
            prompt = assemble_prompt(view_hint, effective_base, style.suffix)
            sample_dict = self._build_sample_dict(
                req, style, index, input_video, prompt, seed, reuse
            )
            sample = InferenceArguments.model_validate(sample_dict)
            log.info("[%d/%d] style=%s seed=%d", index + 1, len(styles), style.name, seed)
            t0 = time.perf_counter()
            self._inference.generate([sample], job_dir)
            gen_s = time.perf_counter() - t0

            name = sample_dict["name"]
            out_mp4 = job_dir / f"{name}.mp4"
            ctrl_files: dict[str, str] = {}
            for key in control_keys:
                ctrl = job_dir / f"{name}_control_{key}.mp4"
                if ctrl.exists():
                    ctrl_files[key] = ctrl.name
            return SampleResult(
                index=index,
                style=style.name,
                resolved_prompt=prompt,
                negative_prompt=sample_dict.get("negative_prompt"),
                seed=seed,
                output_file=out_mp4.name if out_mp4.exists() else None,
                control_files=ctrl_files,
                generation_time_s=round(gen_s, 2),
            )

        # Sample 0
        first = _gen_one(0, styles[0], None)
        results.append(first)
        first_name = f"sample_00_{_safe_slug(styles[0].name)}"
        for key in control_keys:
            ctrl = job_dir / f"{first_name}_control_{key}.mp4"
            if ctrl.exists():
                control_paths[key] = str(ctrl.resolve())
        if control_paths:
            log.info("Reusing control video(s) %s for remaining samples.", list(control_paths))

        # Samples 1..N-1, reusing the control video(s).
        for i in range(1, len(styles)):
            results.append(_gen_one(i, styles[i], control_paths or None))

        # 3. Manifest.
        manifest = {
            "request": req.model_dump(exclude_none=True),
            "base_prompt": base_prompt,
            "upsampled_prompt": upsampled_prompt,
            "controls": control_keys,
            "samples": [r.model_dump() for r in results],
        }
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("Wrote manifest.json (%d samples).", len(results))

        return {
            "base_prompt": base_prompt,
            "upsampled_prompt": upsampled_prompt,
            "controls": control_keys,
            "samples": results,
        }
