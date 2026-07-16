# CLAUDE.md — Cosmos Transfer 2.5

NVIDIA **Cosmos Transfer 2.5**: control-conditioned video-to-video world generation. You give it an input video plus structured **controls** (edge/depth/seg/vis) and a **prompt**, and it restyles/regenerates the video while preserving the controlled structure. Active work here centers on the **HTTP API** (`cosmos_transfer2/api/`, for the ClearML pipeline) and **general inference** (`examples/`).

The repo is split in two: `cosmos_transfer2/*.py` is the **thin public layer** you work in; `cosmos_transfer2/_src/` is the **vendored NVIDIA engine** — import through the public layer, don't edit `_src/` unless explicitly asked.

## Golden rules

- **When unsure about intent, scope, or an expensive/destructive action, interview the user. Never assume.** A 10-second question beats a wrong 30-minute GPU run.
- **GPU — check, don't clobber.** You're free to launch inference, but run `nvidia-smi` first; don't start work that would OOM or interrupt a job already running. The GPU is shared.
- **Use the existing env. Never `pip install` or call bare `python`.** A `.venv` already exists. Run things with `just run <cmd>` or `uv run --no-sync <cmd>`.
- **Don't re-download or wipe checkpoints.** They auto-download to the HuggingFace cache on first use (multi-GB). Reuse the cache; set `HF_HOME` only if a specific path is needed.

## Environment & commands

Managed by **`uv`** (Python 3.10) and the **`justfile`**. Don't invent flags — check the `justfile` if unsure.

- Install / sync: `just install cu128` (use `cu130` for Blackwell / DGX Spark / aarch64).
- Run in-venv: `just run <cmd>` (syncs first), or `uv run --no-sync <cmd>` (no sync, faster).
- Tests: `just test-cpu`, `just test-gpu`, or `just test` (pyrefly + cpu + gpu). Test files are `*_test.py`.
- Lint / format: `just lint` (pre-commit + ruff). Type-check: `just pyrefly`.

## Architecture (`cosmos_transfer2/`)

Thin public layer — what you edit/import:

- `inference.py` — single-view `Control2WorldInference` (load checkpoint → apply controls → generate).
- `config.py` — Pydantic args (`SetupArguments`, `InferenceArguments`) and `CONTROL_KEYS = ["edge", "vis", "depth", "seg"]`.
- `multiview.py` / `plenoptic.py` / `robot_multiview.py` — 7-camera, novel-view, and Agibot robot variants (+ their `*_config.py`).
- `api/` — in-process restyling engine library (`InferenceEngine`, request schema, styles/views). `gradio/` — UI workers. `experiments/` — project-specific code (e.g. tictactoe).

`cosmos_transfer2/_src/` — vendored engine (`imaginaire/`, `transfer2/`, `transfer2_multiview/`, …). Treat as a dependency.

## Inference entry points (`examples/`)

```bash
just run python examples/inference.py -i <spec.json> -o <out_dir>        # single-view
torchrun --nproc_per_node=N examples/inference.py -i <spec.json> -o <out_dir>  # multi-GPU
```

Other entry points: `examples/multiview.py`, `examples/plenoptic.py`, `examples/robot_multiview_agibot_control.py`.

The `<spec.json>` follows the Pydantic schema in `config.py`: `prompt`, `video_path`, `seed`, `guidance`, `num_steps`, `resolution`, `sigma_max`, plus per-control blocks (`edge`, `depth`, `seg`, `vis`) each with a `control_weight`. Read `config.py` for exact fields rather than guessing.

### Control modalities

- **edge** — Canny edges. Lightweight default; preserves motion/structure.
- **depth** — VideoDepthAnything monocular depth; preserves 3D geometry.
- **seg** — GroundingDINO + SAM2 segmentation; needs a `control_prompt` naming what to segment.
- **vis** — blurred input; anchors original colour/appearance.

Single control → lighter checkpoint. Multiple controls ("multicontrol") → heavier multi-branch checkpoint, auto-selected.

## Restyling engine (`cosmos_transfer2/api/`)

In-process engine for domain-randomized restyling (N style variations from one control pass). No HTTP server — called directly and driven by a ClearML Task.

- `engine.py` — `InferenceEngine(work_dir)`; `run_job(req, input_video, job_dir)` / `run_dual_view_job(...)`. Loads the checkpoint once and reuses it; generates the control video once and reuses it across styles.
- `config_models.py` — `GenerateRequest` / `DualViewRequest` schema + `CONTROL_PRESETS` (balanced/multicontrol_robot/edge). `styles.yaml` / `views.yaml` — style suffixes and camera-view hints.
- Run: `uv run --no-sync python scripts/clearml_task.py` — a batch-first ClearML Task (list of episodes, model loads once) tracked in the ClearML GUI.
- Env: `HF_TOKEN` (only if checkpoints aren't cached), `ANTHROPIC_API_KEY` (optional — prompt upsampling via Claude).
- **See `ENGINE_GUIDE.md` for the full request schema, control presets, and all env vars** — don't duplicate or guess them here.

## Output conventions (`outputs/`)

- Engine/ClearML jobs: `outputs/clearml/<task_id>/episode_NNN/` containing `sample_NN_<style>.mp4`, matching `.json`, per-control videos `..._control_edge.mp4`, and `manifest.json`.
- Experiment scripts: `outputs/tictactoe/<experiment>/` with a per-run `.log`.
- **Control videos are cached and reused across style samples** — don't regenerate them unnecessarily.

## Terminology (so instructions aren't misread)

- **control vs prompt** — controls fix structure; the prompt varies appearance.
- **guidance / CFG** (~0–7) — higher = stricter prompt adherence.
- **`sigma_max`** (~0–200, default 70) — the fidelity↔freedom knob; higher = more restyling and more hallucination.
- **guided generation** — mask-anchor a foreground object (e.g. robot arm) while restyling the rest.
- **multicontrol** — 2+ controls at once (e.g. depth+seg+edge+vis); minimizes hallucination.
- **dual-view consistency** — top+wrist processed as one stacked video, then split, so style is frame-locked.
- **domain randomization** — generating N appearance variants of one action to train downstream vision models.
