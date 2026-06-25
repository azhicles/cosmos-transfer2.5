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

"""Request/response schemas for the Cosmos Transfer API.

These mirror a useful subset of ``cosmos_transfer2.config.InferenceArguments`` and add the
API-specific concepts: styles (the appearance variations), views (camera-perspective hints),
and optional prompt upsampling. Every field has a sensible default so the smallest valid
request is just ``{"prompt": "...", "video_path": "..."}``.
"""

from __future__ import annotations

import enum
from typing import Literal, Optional, Union

import pydantic

# Control keys, in the canonical order used by cosmos_transfer2.config.CONTROL_KEYS.
# Re-declared here so the schema module has no import-time dependency on the heavy
# inference stack (lets the API schema be imported/validated without a GPU).
CONTROL_KEYS = ["edge", "vis", "depth", "seg"]
ControlKey = Literal["edge", "vis", "depth", "seg"]
Threshold = Literal["very_low", "low", "medium", "high", "very_high"]

# Named control recipes. `multicontrol_robot` mirrors NVIDIA's
# assets/robot_example/multicontrol/robot_multicontrol_spec.json — depth+seg dominate so the
# 3D geometry and semantic regions of the action are preserved (far fewer hallucinations than
# edge-only), while edge/vis add light structure/colour anchoring.
ControlPreset = Literal["balanced", "multicontrol_robot", "edge"]

CONTROL_PRESETS: dict[str, dict[str, dict]] = {
    # `balanced` is the default — matches the validated production recipe (job 74c): depth+seg+edge
    # with NO vis, for strong restyle while preserving structure/geometry.
    "balanced": {
        "depth": {"control_weight": 1.0},
        "seg": {"control_weight": 1.0},
        "edge": {"control_weight": 0.2},
    },
    "multicontrol_robot": {
        "depth": {"control_weight": 1.0},
        "seg": {"control_weight": 1.0},
        "edge": {"control_weight": 0.2},
        "vis": {"control_weight": 0.5, "preset_blur_strength": "high"},
    },
    "edge": {
        "edge": {"control_weight": 1.0},
    },
}


class JobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ControlSpec(pydantic.BaseModel):
    """Per-control configuration. Maps onto cosmos ControlConfig / EdgeConfig / etc."""

    model_config = pydantic.ConfigDict(extra="forbid")

    control_weight: float = pydantic.Field(1.0, ge=0.0, le=1.0)
    """How strongly the output adheres to this control (0-1). For multicontrol the weights are
    normalised to sum to <= 1.0 by the model."""
    control_path: Optional[str] = None
    """Path to a pre-computed control video. Normally left None - the API generates the control
    once from the input video and reuses it across all samples."""
    mask_path: Optional[str] = None
    """Path to a pre-computed binary spatiotemporal mask."""
    mask_prompt: Optional[str] = None
    """Prompt for on-the-fly mask generation via SAM2 (e.g. 'robot arm gripper')."""
    preset_edge_threshold: Optional[Threshold] = None
    """edge only: Canny sensitivity. Lower detects more edges."""
    preset_blur_strength: Optional[Threshold] = None
    """vis only: bilateral blur strength."""
    control_prompt: Optional[str] = None
    """seg only: what to segment. Defaults to the first 128 words of the prompt."""


class StyleOverrides(pydantic.BaseModel):
    """Optional per-style generation-parameter overrides. Any field left None falls back to the
    request-level value."""

    model_config = pydantic.ConfigDict(extra="forbid")

    guidance: Optional[int] = pydantic.Field(None, ge=0, le=7)
    sigma_max: Optional[str] = None
    seed: Optional[int] = None
    num_steps: Optional[int] = pydantic.Field(None, ge=1)
    control_weights: Optional[dict[ControlKey, float]] = None
    """Override the control_weight of specific controls for this style only."""


class StyleEntry(pydantic.BaseModel):
    """A single appearance variation. The suffix is appended to the (optionally upsampled)
    action prompt; optional fields let a style tune its own negatives and generation params."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    suffix: str
    """Text appended to the prompt, e.g. 'Warm tungsten indoor lighting, soft shadows.'"""
    negative_prompt: Optional[str] = None
    """Per-style negative prompt override."""
    overrides: Optional[StyleOverrides] = None


# A request may reference styles by name (resolved against the bundled styles.yaml) or supply
# a full inline StyleEntry.
StyleRef = Union[str, StyleEntry]


class GenerateRequest(pydantic.BaseModel):
    """The body of POST /generate (JSON), or the 'request' form field (multipart upload)."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # --- Core ---
    prompt: str
    """Action prompt describing what is happening in the video."""
    num_samples: int = pydantic.Field(4, ge=1, le=16)
    """Number of appearance variations to generate."""
    seed: int = 2025
    """Base seed. Per-sample seed = seed + sample_index (a style may override its own seed)."""
    num_steps: int = pydantic.Field(15, ge=1, le=100)
    guidance: int = pydantic.Field(5, ge=0, le=7)
    negative_prompt: Optional[str] = None
    """Request-level negative prompt. If None, the model default is used."""

    # --- Styles & view ---
    styles: Optional[list[StyleRef]] = None
    """Styles to apply. If None, the first ``num_samples`` bundled defaults are used. If given,
    its length defines the number of samples (overriding num_samples)."""
    view: Optional[str] = None
    """View name resolved against views.yaml (e.g. 'wrist', 'top')."""
    view_hint: Optional[str] = None
    """Inline camera-perspective phrase; overrides the resolved view hint."""

    # --- Controls ---
    control_preset: ControlPreset = "balanced"
    """Named control recipe used when ``controls`` is not given. Default 'balanced'
    (depth 1.0 + seg 1.0 + edge 0.2, no vis) — matches the validated production recipe (job 74c)."""
    controls: Optional[dict[ControlKey, ControlSpec]] = None
    """Explicit control configuration. If provided, overrides ``control_preset``."""

    # --- Advanced video / quality ---
    resolution: str = "720"
    sigma_max: Optional[str] = "110"
    """0-200: how much noise is added to the input. Higher = more freedom from the input (more
    restyle, more hallucination); lower = closer to the input structure. This is the main
    fidelity↔freedom knob — prefer tuning it over lowering ``guidance``."""
    num_video_frames_per_chunk: int = pydantic.Field(93, ge=1)
    max_frames: Optional[int] = pydantic.Field(None, ge=1)
    """Read only the first N frames of the input. None = entire video (output matches input length)."""
    keep_input_resolution: bool = True

    # --- Guided generation (foreground anchoring; opt-in) ---
    guided_generation: bool = False
    """When true, anchor a foreground region (e.g. the robot arm) so its geometry/identity is
    preserved while the rest of the scene is restyled. The mask is auto-generated via SAM2 from
    ``guided_foreground_prompt``."""
    guided_foreground_prompt: Optional[str] = None
    """Object(s) to anchor when guided_generation is on. Defaults to a robot-arm prompt."""
    guided_generation_step_threshold: int = 25
    """Diffusion steps over which the foreground anchor is applied (higher = stronger anchoring)."""

    # --- Prompt upsampling (off-GPU, optional) ---
    upsample: bool = False
    upsample_model: Optional[str] = None
    """Override the LLM model used for prompt upsampling."""

    # --- Misc ---
    disable_guardrails: bool = True

    # --- Input source (exactly one is required; upload is handled by the multipart endpoint) ---
    video_path: Optional[str] = None
    """Server-side path to the input video (e.g. on a mounted volume)."""
    video_url: Optional[str] = None
    """URL the server downloads the input video from."""

    def resolved_num_samples(self) -> int:
        if self.styles is not None:
            return len(self.styles)
        return self.num_samples

    def effective_controls(self) -> dict[str, ControlSpec]:
        """Explicit ``controls`` if given, else the selected ``control_preset`` expanded."""
        if self.controls:
            return dict(self.controls)
        return {k: ControlSpec(**v) for k, v in CONTROL_PRESETS[self.control_preset].items()}


class DualViewRequest(GenerateRequest):
    """POST /generate/dual_view: restyle a top and wrist view of the SAME episode together.

    Both views are stacked into one video and restyled in a single diffusion pass per style
    (one seed), then split back — guaranteeing the two views share style/colour/lighting. All the
    GenerateRequest fields apply (styles, control_preset, sigma_max, guided_generation, ...).
    The single-view ``video_path``/``video_url`` are unused here.
    """

    top_path: Optional[str] = None
    top_url: Optional[str] = None
    wrist_path: Optional[str] = None
    wrist_url: Optional[str] = None
    stack: Literal["vstack", "hstack"] = "vstack"
    """vstack = top above wrist (640x960, VRAM-safe). hstack = side by side (more tokens)."""


class SampleResult(pydantic.BaseModel):
    index: int
    style: str
    resolved_prompt: str
    negative_prompt: Optional[str] = None
    seed: int
    output_file: Optional[str] = None
    """Basename of the generated mp4 within the job directory (single-view jobs)."""
    view_outputs: dict[str, str] = pydantic.Field(default_factory=dict)
    """For dual-view jobs: {"top": "<file>.mp4", "wrist": "<file>.mp4"}."""
    control_files: dict[str, str] = pydantic.Field(default_factory=dict)
    generation_time_s: Optional[float] = None


class JobStatus(pydantic.BaseModel):
    id: str
    state: JobState
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    queue_position: Optional[int] = None
    """0 = next to run; only meaningful while queued."""
    base_prompt: Optional[str] = None
    upsampled_prompt: Optional[str] = None
    controls: list[str] = pydantic.Field(default_factory=list)
    samples: list[SampleResult] = pydantic.Field(default_factory=list)
    error: Optional[str] = None
    log_tail: Optional[str] = None


class GenerateResponse(pydantic.BaseModel):
    job_id: str
    state: JobState
    queue_position: Optional[int] = None
