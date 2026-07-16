<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos Transfer 2.5 — Engine Guide

## 1. Overview

The Cosmos Transfer engine restyles an **input video** — guided by structured
**controls** (edge / depth / seg / vis) and an action **prompt** — into **N
domain-randomization style variations** of the same action. Each variation keeps
the scene's geometry and motion (fixed by the controls) but changes lighting,
surfaces, materials and sensor look (driven by a per-style prompt suffix), so the
frames stay realistic and useful as augmented robot-training data.

The engine is called **in-process**: there is no HTTP server, no port, no
container. You construct a `GenerateRequest`, hand it to an `InferenceEngine`
instance, and it writes the outputs to a job directory. The checkpoint loads once
and stays resident on the local GPU.

Production runs are driven by a **ClearML Task** (`scripts/clearml_task.py`),
which owns queueing, console logging, scalar/media monitoring and artifact
storage. ClearML replaces everything the old HTTP layer used to do.

## 2. Quick start

The engine itself is already installed in the repo's `.venv`. Add ClearML and
configure credentials once:

```bash
uv pip install clearml          # into the existing venv
clearml-init                    # writes ~/clearml.conf (credentials, one time)
```

Then edit the `PARAMS` dict at the top of `scripts/clearml_task.py` (episodes,
prompt, styles, steps, preset) and run:

```bash
uv run --no-sync python scripts/clearml_task.py
```

This runs on the **local GPU** and is fully tracked in the ClearML GUI under the
project **"Cosmos Transfer"** — console output, per-sample timing scalars, and
each output video inline (DEBUG SAMPLES) and as an artifact. Check `nvidia-smi`
first; the GPU is shared.

## 3. Programmatic use

To drive the engine directly from Python:

```python
from pathlib import Path
from cosmos_transfer2.api.config_models import GenerateRequest
from cosmos_transfer2.api.engine import InferenceEngine

req = GenerateRequest(
    prompt="A robot arm places a wooden block on the table.",
    video_path="/data/episode_000001.mp4",
    styles=["warm_indoor", "cool_daylight"],
    control_preset="balanced",
)

engine = InferenceEngine(work_dir=Path("outputs/_work"))   # checkpoint loads once
result = engine.run_job(req, Path(req.video_path), job_dir=Path("outputs/job01"))
```

The smallest valid request is just `{"prompt": ..., "video_path": ...}` — every
other field has a default. Reuse one `InferenceEngine` across many jobs so the
model is loaded only once. For top+wrist episodes use
`engine.run_dual_view_job(req, top, wrist, job_dir)` with a `DualViewRequest`.

## 4. GenerateRequest schema

The most-used fields (exact defaults from `cosmos_transfer2/api/config_models.py`):

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `prompt` | str | *(required)* | Action prompt describing what happens in the video. |
| `video_path` | str? | `None` | Path to the input video (single-view). |
| `num_samples` | int | `4` | Number of style variations (ignored if `styles` is given). |
| `seed` | int | `2025` | Base seed; per-sample seed = `seed + index`. |
| `num_steps` | int | `15` | Diffusion steps. ~6 for smoke tests, ~35 for production. |
| `guidance` | int | `5` | CFG strength (0–7); higher = stricter prompt adherence. |
| `styles` | list? | `None` | Style names or inline entries; its length sets the sample count. |
| `view` | str? | `None` | View name resolved against `views.yaml` (e.g. `wrist`, `top`). |
| `view_hint` | str? | `None` | Inline camera phrase; overrides the resolved view. |
| `control_preset` | str | `"balanced"` | Named control recipe when `controls` is not set. |
| `controls` | dict? | `None` | Explicit per-control config; overrides `control_preset`. |
| `resolution` | str | `"720"` | Output resolution. |
| `sigma_max` | str? | `"110"` | Fidelity↔freedom knob (0–200); higher = more restyle/hallucination. |
| `guided_generation` | bool | `False` | Anchor a foreground object (e.g. robot arm) while restyling the rest. |
| `upsample` | bool | `False` | Optional off-GPU LLM prompt upsampling. |
| `disable_guardrails` | bool | `True` | Skip the content guardrail models. |

`sigma_max` is the main fidelity knob — prefer tuning it over lowering
`guidance`. If `styles` is provided, its length overrides `num_samples`.

## 5. Control presets

Selected by `control_preset` and expanded from `CONTROL_PRESETS`:

| Preset | Controls (weights) | Use |
|--------|-------------------|-----|
| `balanced` *(default)* | depth 1.0 + seg 1.0 + edge 0.2 | Strong restyle while preserving structure/geometry (validated production recipe). |
| `multicontrol_robot` | depth 1.0 + seg 1.0 + edge 0.2 + vis 0.5 (blur high) | Adds a colour anchor on top of balanced; mirrors NVIDIA's robot multicontrol spec. |
| `edge` | edge 1.0 | Fast/lightweight, single control; good for smoke tests. |

## 6. Control modalities

- **edge** — Canny edges. Lightweight; preserves motion/structure. `preset_edge_threshold` tunes sensitivity.
- **depth** — monocular depth; preserves 3D geometry.
- **seg** — GroundingDINO + SAM2 segmentation; uses `control_prompt` (defaults to the first 128 words of the prompt) to name what to segment.
- **vis** — blurred input; anchors original colour/appearance. `preset_blur_strength` tunes it.

A single control selects a lighter checkpoint; **two or more controls select the
heavier multicontrol (multi-branch) checkpoint** automatically.

## 7. Styles & views

A **style** appends a text **suffix** to the (optionally upsampled) action prompt
and may carry its own negative prompt and generation overrides. The bundled
library (`cosmos_transfer2/api/styles.yaml`) provides these domain-randomization
styles:

```
warm_indoor      cool_daylight    dim_evening      bright_overhead   wooden_table
white_lab        metallic_surface overcast_window  tungsten_lamp     outdoor_shade
```

A **view** injects a camera-perspective hint at the front of the prompt so the
model renders the right perspective. Names from
`cosmos_transfer2/api/views.yaml`:

```
wrist    top    front    side    default
```

Override either library by mounting your own YAML and pointing
`COSMOS_API_STYLES_FILE` / `COSMOS_API_VIEWS_FILE` at it, or pass inline style
objects / a `view_hint` in the request. An unknown view name produces no hint.

## 8. Outputs

Each job writes into its `job_dir`:

- `sample_NN_<style>.mp4` — one restyled video per style variation.
- `sample_NN_<style>_control_<key>.mp4` — the control video(s) for each modality (edge/depth/seg/vis).
- `manifest.json` — the full request, base/upsampled prompt, active controls, and per-sample metadata (style, resolved prompt, seed, output file, timing).

For **dual-view** jobs each sample is also split into `..._top.mp4` and
`..._wrist.mp4`, which are frame-locked in style/colour/lighting because both
views ride one diffusion trajectory per style.

**Control videos are generated once on the first sample and reused** across all
remaining styles (via `control_path`) — the styles share the same input, so the
control pass is identical. This avoids N expensive control passes, which matters
most for depth/seg multicontrol.

## 9. The ClearML Task

`scripts/clearml_task.py` is the production driver. Highlights:

- **Batch-first**: `PARAMS["episodes"]` is a *list*; the model loads once and is reused across every episode (the "warm model" benefit without a server). The smoke test passes a 1-episode list.
- **Monitoring**: per-sample generation time and batch progress are reported as ClearML **scalars**; each output video is shown inline via `report_media` (DEBUG SAMPLES).
- **Artifacts**: output videos and `manifest.json` are uploaded per episode. `output_uri` (`None` = ClearML file server) can point at a directory or `s3://…` before scaling to many/large videos.
- **Robust batch**: each episode runs in its own `try/except`; one bad episode reports a failure scalar and continues rather than aborting the batch.
- **`dual_view` flag**: when `True`, episodes supply `top_path`/`wrist_path` and go through `run_dual_view_job` with the configured `stack` (`vstack`/`hstack`).

Every `PARAMS` field is captured via `task.connect`, so you can override any of
them from the ClearML GUI when cloning or enqueuing the Task.

## 10. Environment

- **`HF_TOKEN`** — only needed if the checkpoints aren't already in the local HuggingFace cache (they auto-download on first use). Don't re-download or wipe the cache.
- **`disable_guardrails`** — defaults to `True` (guardrail models skipped).
- **`ANTHROPIC_API_KEY`** — only if `upsample=True` (LLM prompt upsampling).
- **GPU**: a single GPU is assumed; the engine keeps exactly one checkpoint resident and reloads only when the control set changes. Run `nvidia-smi` before launching — the GPU is shared.
