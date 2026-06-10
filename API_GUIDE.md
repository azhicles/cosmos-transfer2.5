# Cosmos Transfer API

An HTTP service that wraps **Cosmos Transfer 2.5** (single-view video-to-video restyling) so it
can run as a step in a ClearML pipeline. You call it with an input video, an action prompt, and
metadata (e.g. the camera view); it generates **N appearance variations** (default 4) for
**domain randomization** of robot training data — varied lighting, surface colours, materials,
and time-of-day — by appending per-style prompt suffixes to your action prompt while a control
signal (edge by default) keeps the action/structure intact.

The model is expensive to load, so the service loads it once and stays up. Generation is
**asynchronous**: you submit a job, poll for status, then download the results.

---

## Contents

- [Architecture](#architecture)
- [Build & run](#build--run)
- [Environment variables](#environment-variables)
- [Endpoints](#endpoints)
- [Request schema](#request-schema)
- [Examples](#examples)
- [Styles](#styles)
- [Views](#views)
- [Prompt upsampling](#prompt-upsampling)
- [ClearML integration](#clearml-integration)
- [Outputs on disk](#outputs-on-disk)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
POST /generate ─► job queue (1 at a time, 1 GPU) ─► Control2WorldInference.generate()
   │                                                          │
   └─ returns {job_id}                                        ├─ sample 0: generate control + video
                                                              ├─ samples 1..N: reuse control, new style prompt
GET /jobs/{id} ─► status (queued│running│succeeded│failed)    └─ writes mp4s + manifest.json + job.log
GET /jobs/{id}/results ─► zip of the N videos + manifest
```

- **Per request:** the N style variations share one input video, so the control video (edge/
  depth/…) is generated **once** and reused across all N samples — saving most of the control
  cost, which matters most for depth/seg multicontrol.
- **Per-sample seed:** `seed + sample_index` (each style may override its own seed), so the N
  variations are reproducible and distinct.
- **One GPU:** jobs run strictly one at a time via an internal FIFO queue.

The API server (`cosmos_transfer2/api/`) is ClearML-agnostic. A separate, GPU-free
`scripts/clearml_step.py` calls it from a pipeline.

---

## Build & run

The API image is layered on the existing Cosmos Transfer base image.

**1. Build the base image** (carries the CUDA + model environment):

```bash
# Ampere – Hopper:
docker build -f Dockerfile --build-arg STANDALONE=true -t cosmos-transfer2:base .
# Blackwell (e.g. RTX Pro 6000 Blackwell):
docker build -f docker/nightly.Dockerfile --build-arg STANDALONE=true -t cosmos-transfer2:base .
```

**2. Start the API** (docker-compose builds the API layer on top of the base):

```bash
export HF_TOKEN=hf_...              # required: gated checkpoint download (accept the NVIDIA license on HF first)
export ANTHROPIC_API_KEY=sk-...     # optional: only for prompt upsampling
# If your base image is tagged differently: export COSMOS_BASE_IMAGE=my/cosmos:base
docker compose up --build
```

The service listens on `http://localhost:8000`. First generation downloads checkpoints into the
`hf-cache` volume (one-time, can take a while).

**Run locally without Docker** (inside the repo's environment, on the GPU host):

```bash
uv sync --extra=cu128   # or cu130 on Blackwell
uvicorn cosmos_transfer2.api.server:app --host 0.0.0.0 --port 8000
```

Check it's up: `curl localhost:8000/healthz`.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token for gated checkpoint download (required). |
| `HF_HOME` | `/root/.cache/huggingface` | Checkpoint cache location (mount to persist). |
| `ANTHROPIC_API_KEY` | — | Enables prompt upsampling (optional). |
| `COSMOS_API_UPSAMPLE_MODEL` | `claude-haiku-4-5` | LLM used for prompt upsampling. |
| `COSMOS_API_OUTPUT_DIR` | `outputs/api` | Root for per-job output dirs. |
| `COSMOS_API_WORK_DIR` | `<output>/_engine` | Engine scratch (config dump, etc.). |
| `COSMOS_API_INPUTS_DIR` | `<output>/_inputs` | Uploaded/downloaded input videos. |
| `COSMOS_API_PRELOAD` | `edge` | Control set to preload at startup; `none` to skip; comma-list e.g. `edge,depth`. |
| `COSMOS_API_LOG_LEVEL` | `INFO` | Logger level. |
| `COSMOS_API_STYLES_FILE` | — | Path to a YAML overriding the bundled style library. |
| `COSMOS_API_VIEWS_FILE` | — | Path to a YAML overriding the bundled view map. |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Liveness + whether the model is loaded and which control set. |
| `GET` | `/styles` | The bundled style library. |
| `GET` | `/views` | The bundled view → perspective-phrase map. |
| `POST` | `/generate` | Submit a job (JSON or multipart). Returns `{job_id, state, queue_position}`. |
| `GET` | `/jobs` | List all jobs. |
| `GET` | `/jobs/{id}` | Job status (incl. per-sample results, `log_tail`). |
| `DELETE` | `/jobs/{id}` | Cancel a **queued** job (running jobs can't be preempted). |
| `GET` | `/jobs/{id}/results` | Zip of the output videos + `manifest.json` (only when `succeeded`). |
| `GET` | `/jobs/{id}/results/{index}` | A single output video by sample index. |

`POST /generate` accepts either:
- **JSON** (`application/json`): the [request schema](#request-schema) with `video_path` or `video_url`.
- **multipart/form-data**: a `video` file part + a `request` field containing the JSON body.

---

## Request schema

All fields except `prompt` (and an input source) are optional; defaults shown.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | **Required.** Action prompt describing the video. |
| `video_path` | string | — | Server-side path to the input (one input source required). |
| `video_url` | string | — | URL the server downloads from. |
| *(multipart `video`)* | file | — | Upload instead of `video_path`/`video_url`. |
| `num_samples` | int | `4` | Number of variations. Ignored if `styles` is given (its length wins). |
| `styles` | list | first 4 defaults | Style **names** and/or inline style objects (see [Styles](#styles)). |
| `view` | string | — | View name resolved against `views.yaml`. |
| `view_hint` | string | — | Inline perspective phrase; overrides `view`. |
| `seed` | int | `2025` | Base seed; per-sample seed = `seed + index`. |
| `num_steps` | int | `35` | Diffusion steps. |
| `guidance` | int (0–7) | `3` | Prompt adherence. |
| `negative_prompt` | string | model default | Request-level negative prompt. |
| `controls` | object | `{"edge": {"control_weight": 1.0}}` | Control config (see [below](#controls)). |
| `resolution` | string | `"720"` | e.g. `"720"`, `"480"`. |
| `sigma_max` | string | — | 0–200; how much input noise (higher = more freedom from input). |
| `num_video_frames_per_chunk` | int | `93` | Chunk size for long-video generation. |
| `max_frames` | int | — | Read only first N frames; omit to use the whole video (output matches input length). |
| `keep_input_resolution` | bool | `true` | Resize output back to the input resolution. |
| `upsample` | bool | `false` | Enrich the prompt via an LLM before adding style suffixes. |
| `upsample_model` | string | `claude-haiku-4-5` | Override the upsampling model. |
| `disable_guardrails` | bool | `true` | Disable Cosmos content guardrails (faster, less VRAM). |

### Controls

`controls` maps control type → spec. Default (omit `controls`) is **edge-only** at weight 1.0 —
the lighter single-control checkpoint, which preserves the action's structure while leaving
colour free for the prompt. Provide multiple keys for **multicontrol** (loads the heavier
multibranch checkpoint and triggers a one-time engine reload if a different set was loaded).

Control keys: `edge`, `depth`, `seg`, `vis`. Per-control spec fields:

| Field | Applies to | Notes |
|---|---|---|
| `control_weight` | all | 0.0–1.0. Multicontrol weights are normalised to sum ≤ 1.0. |
| `preset_edge_threshold` | edge | `very_low`…`very_high`. |
| `preset_blur_strength` | vis | `very_low`…`very_high`. |
| `control_prompt` | seg | What to segment (defaults to first 128 prompt words). |
| `mask_prompt` | all | SAM2 mask prompt (e.g. `"robot arm gripper"`). |
| `mask_path` / `control_path` | all | Advanced: pre-computed mask / control video. |

> `vis` transfers the *original* colour/appearance, so it works against generating *new*
> colours — use a low weight (or omit it) for domain randomization.

---

## Examples

**Minimal (JSON, server-side path), then poll + download:**

```bash
# submit
JOB=$(curl -s -X POST localhost:8000/generate -H 'content-type: application/json' -d '{
  "prompt": "A robot arm places a game piece on a tic-tac-toe board.",
  "view": "wrist",
  "video_path": "/workspace/inputs/episode_000000.mp4"
}' | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
echo "job: $JOB"

# poll
curl -s localhost:8000/jobs/$JOB | python -m json.tool

# download all 4 videos once succeeded
curl -s -OJ localhost:8000/jobs/$JOB/results
```

**Upload the video (multipart):**

```bash
curl -X POST localhost:8000/generate \
  -F 'video=@episode_000000.mp4;type=video/mp4' \
  -F 'request={"prompt":"A robot arm places a game piece.","view":"wrist","num_samples":2}'
```

**Multicontrol + custom styles + upsampling:**

```bash
curl -X POST localhost:8000/generate -H 'content-type: application/json' -d '{
  "prompt": "A robot arm places a game piece on the board.",
  "view": "top",
  "video_url": "https://example.internal/ep0.mp4",
  "styles": ["warm_indoor", "white_lab", {"name":"blue_hour","suffix":"Cool blue-hour twilight lighting. Photorealistic."}],
  "controls": {"edge": {"control_weight": 1.0}, "depth": {"control_weight": 0.5}},
  "num_steps": 35,
  "upsample": true
}'
```

---

## Styles

A style is appended to the (optionally upsampled) action prompt to produce one variation. The
bundled library (`cosmos_transfer2/api/styles.yaml`) ships 10 photorealistic
**domain-randomization** styles: `warm_indoor`, `cool_daylight`, `dim_evening`,
`bright_overhead`, `wooden_table`, `white_lab`, `metallic_surface`, `overcast_window`,
`tungsten_lamp`, `outdoor_shade`. List them at runtime with `GET /styles`.

Each entry supports:

```yaml
my_style:
  suffix: "Warm tungsten indoor lighting, soft shadows. Photorealistic."   # required
  negative_prompt: "harsh glare, blown highlights"                          # optional
  overrides:                                                                # optional
    guidance: 4
    sigma_max: "70"
    seed: 7
    num_steps: 40
    control_weights: { edge: 0.9, depth: 0.4 }
```

Select styles per request by **name**, or pass **inline** style objects (same shape, plus a
`name`). Override the whole library by mounting a YAML and setting `COSMOS_API_STYLES_FILE`.

---

## Views

`view` injects a camera-perspective phrase at the front of the prompt so the model renders the
right viewpoint. Bundled map (`cosmos_transfer2/api/views.yaml`): `wrist` (egocentric close-up),
`top` (overhead), `front`, `side`. An unknown view name applies no hint and logs a warning. Pass
`view_hint` to supply an inline phrase instead. Override the map via `COSMOS_API_VIEWS_FILE`.

---

## Prompt upsampling

With `upsample: true` (and `ANTHROPIC_API_KEY` set), the terse action prompt is expanded into a
fuller scene description **once** via the Claude API before the per-style suffixes are appended.
It runs off-GPU so it never competes with the diffusion model for VRAM. The upsampler is
prompted to preserve the action, objects, and viewpoint and to **avoid** dictating colour/style
(those come from the style suffix). If the key or package is missing, or the call fails,
generation proceeds with the original prompt (best-effort — never blocks). The interface
(`cosmos_transfer2/api/upsampler.py`) is pluggable; the in-repo Cosmos-Reason model could be
dropped in later.

---

## ClearML integration

The API stays ClearML-agnostic. Use the GPU-free `scripts/clearml_step.py` from your pipeline:

```python
from scripts.clearml_step import run_cosmos_transfer_step

result = run_cosmos_transfer_step(
    api_url="http://cosmos-api:8000",
    video_path="/data/episode_000000.mp4",   # a path the API server can read
    prompt="A robot arm places a game piece on the board.",
    view="wrist",
    styles=["warm_indoor", "cool_daylight", "dim_evening", "white_lab"],
    num_steps=35,
    out_dir="/data/cosmos_out",
)
# result["videos"] -> downloaded mp4 paths; registered as ClearML artifacts when a Task is active
```

Or as a CLI step:

```bash
python scripts/clearml_step.py --api-url http://localhost:8000 \
  --video-path /data/ep0.mp4 --prompt "A robot arm places a piece." \
  --view wrist --styles warm_indoor,cool_daylight --out-dir ./cosmos_out
```

It submits, polls, downloads the zip, extracts the videos, and (when a ClearML `Task` is active)
uploads them as artifacts and logs the request + manifest. It does **not** import
`cosmos_transfer2`, so it runs on a lightweight agent without CUDA.

---

## Outputs on disk

Each job gets `outputs/api/<job_id>/`:

```
<job_id>/
  sample_00_<style>.mp4            # the generated variations
  sample_01_<style>.mp4
  ...
  sample_00_<style>_control_edge.mp4   # control video(s) used
  sample_NN_<style>.json / .txt        # cosmos sidecars (args + prompt)
  manifest.json                    # request, resolved prompts, seeds, controls, per-sample timings
  metrics.log                      # cosmos per-sample metrics (VRAM, timing)
  job.log                          # full structured log for this job
  job.json                         # persisted JobStatus (survives restart)
```

---

## Troubleshooting

- **`/healthz` shows `model_loaded: false`** — preload was skipped or failed; the engine loads
  lazily on the first request. Check `docker logs` for the load error (usually HF auth).
- **Checkpoint download fails / hangs** — set a valid `HF_TOKEN` and accept the NVIDIA Open Model
  License on the gated Hugging Face repos. The cache lives in the `hf-cache` volume.
- **First multicontrol request is slow** — switching control sets tears down and reloads the
  checkpoint (logged as "Switching control set …"). Keep a request's control set consistent, or
  set `COSMOS_API_PRELOAD=edge,depth` to preload multicontrol.
- **OOM on the GPU** — lower `resolution` to `"480"`, reduce `num_video_frames_per_chunk`, or
  avoid heavy multicontrol. A single RTX Pro 6000 (96 GB) handles edge-only and typical
  multicontrol comfortably.
- **`video_path not found on server`** — the path must be readable *inside the container*; mount
  the directory (see the `./inputs` volume in `docker-compose.yml`) or use upload / `video_url`.
- **Long jobs** — generation runs many minutes for 4 samples × 35 steps; the async job model is
  designed for this. Use the `clearml_step.py` `--timeout` (default 1 h) accordingly.
- **Prompt upsampling did nothing** — confirm `upsample: true` and `ANTHROPIC_API_KEY`; failures
  fall back to the original prompt and log a warning in `job.log`.
