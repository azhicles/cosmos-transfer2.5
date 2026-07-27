# ⚠️ This is a modified fork of NVIDIA Cosmos-Transfer2.5

This repository is a **fork** of [`nvidia-cosmos/cosmos-transfer2.5`](https://github.com/nvidia-cosmos/cosmos-transfer2.5), extended for a **domain-randomized robot data-augmentation** workflow (the tic-tac-toe project). The upstream NVIDIA README follows below unchanged. Everything in this section describes what **this fork adds or changes** on top of upstream.

## What this fork adds

1. **In-process restyling engine (`cosmos_transfer2/api/`).** Turns one input video + controls into **N domain-randomization style variations** of the same action. The checkpoint loads once and the control video is generated once and reused across all style samples. It is called **in-process — there is no HTTP server, port, or container.** Production runs are driven by a **ClearML Task** (`scripts/clearml_task.py`). Key modules: `engine.py` (`InferenceEngine`), `config_models.py` (`GenerateRequest` / `DualViewRequest` + `CONTROL_PRESETS`), `styles.yaml`, `views.yaml`, `upsampler.py`, `video_ops.py`.

2. **Tic-tac-toe project.** Assets under `assets/tictactoe/` (control specs, prompts, style references), plus driver scripts:
   - `scripts/run_tictactoe_unified.py` — unified dual-view (top+wrist stacked) restyling runner.
   - `scripts/run_outdoor_180.py` — resumable batch of `outdoor_shade` restyling across all 180 episodes.
   - `scripts/exp_run_batch.py`, `scripts/exp_report.py`, `scripts/exp_pareto.py` — experiment harness for sweeping generation params and reporting time/VRAM/realism trade-offs. Findings are written up in `RESULTS.md`.

3. **Runtime fixes to the vendored engine (`cosmos_transfer2/_src/`).** Small, targeted patches needed to run on this hardware/stack — notably an NVENC bitrate fix for Blackwell, plus tweaks to the depth, SAM2, and edge auxiliary pipelines and the checkpoint DB. The `_src/` tree is otherwise treated as an unmodified upstream dependency.

4. **Project documentation.** [`CLAUDE.md`](CLAUDE.md) (architecture + working conventions) and [`ENGINE_GUIDE.md`](ENGINE_GUIDE.md) (full request schema, control presets, env vars). Environment is managed with **`uv`** + a **`justfile`**.

## Setup

Prerequisites: an NVIDIA GPU with a recent driver, [`uv`](https://docs.astral.sh/uv/), and [`just`](https://github.com/casey/just). The 2B checkpoints auto-download from HuggingFace into your local cache on first use (multi-GB) — don't re-download or wipe them.

```bash
# 1. Install the environment (creates .venv, Python 3.10). Pick the CUDA build:
just install cu128        # x86_64 / most GPUs
just install cu130        # Blackwell / DGX Spark / aarch64

# 2. (Only if checkpoints aren't already cached) export a HuggingFace token:
export HF_TOKEN=hf_...
#    Optionally point the cache at a large disk:
export HF_HOME=/mnt/hf-cache

# 3. (Only for tracked ClearML runs) install ClearML and configure credentials once:
uv pip install clearml
clearml-init              # writes ~/clearml.conf

# 4. (Optional) enable LLM prompt upsampling:
export ANTHROPIC_API_KEY=sk-ant-...
```

Run anything inside the environment with `just run <cmd>` (syncs first) or `uv run --no-sync <cmd>` (faster, no sync). **Never** `pip install` into the system Python or call bare `python`. Check `nvidia-smi` before launching — the GPU is shared.

## Usage

The engine restyles one input video into **N style variations** that share the same geometry/motion (fixed by the controls) but vary lighting, materials, and sensor look (driven by per-style prompt suffixes). Three ways to drive it:

**A. Batch runs via ClearML (production path).** Edit the `PARAMS` dict at the top of `scripts/clearml_task.py` (episode list, prompt, styles, steps, control preset, `dual_view`), then:

```bash
uv run --no-sync python scripts/clearml_task.py
```

The model loads once and is reused across every episode. The run is tracked in the ClearML GUI under project **"Cosmos Transfer"** — console logs, per-sample timing scalars, and each output video inline. Every `PARAMS` field is `task.connect`-ed, so you can override them from the GUI when cloning/enqueuing.

**B. Programmatic use.** Build a `GenerateRequest` and hand it to a reused `InferenceEngine`:

```python
from pathlib import Path
from cosmos_transfer2.api.config_models import GenerateRequest
from cosmos_transfer2.api.engine import InferenceEngine

req = GenerateRequest(
    prompt="A robot arm places a wooden block on the table.",
    video_path="/data/episode_000001.mp4",
    styles=["warm_indoor", "cool_daylight"],   # length sets the number of variations
    control_preset="balanced",                 # depth + seg + edge
)
engine = InferenceEngine(work_dir=Path("outputs/_work"))   # checkpoint loads once
engine.run_job(req, Path(req.video_path), job_dir=Path("outputs/job01"))
```

The smallest valid request is just `{"prompt": ..., "video_path": ...}`; every other field has a default. For top+wrist episodes use `engine.run_dual_view_job(req, top, wrist, job_dir)` with a `DualViewRequest` — both views ride one diffusion trajectory per style, so they stay frame-locked in colour/lighting.

**C. Tic-tac-toe driver scripts.** `scripts/run_tictactoe_unified.py` (dual-view runner) and `scripts/run_outdoor_180.py` (resumable 180-episode batch) show end-to-end usage against the assets in `assets/tictactoe/`.

### Key knobs

| Knob | Default | Effect |
|------|---------|--------|
| `styles` / `num_samples` | `4` | Number of style variations (a `styles` list overrides the count). |
| `num_steps` | `15` | Diffusion steps — ~6 for smoke tests, ~35 for production. |
| `sigma_max` | `110` | **Main fidelity↔freedom knob** (0–200); higher = more restyle + more hallucination. Prefer tuning this over `guidance`. |
| `guidance` | `5` | CFG strength (0–7); higher = stricter prompt adherence. |
| `control_preset` | `balanced` | `balanced` (depth+seg+edge), `multicontrol_robot` (adds a vis colour anchor), or `edge` (fast, single control). |
| `resolution` | `720` | Output resolution. |
| `guided_generation` | `False` | Anchor a foreground object (e.g. the robot arm) while restyling the rest. |

Bundled styles (`cosmos_transfer2/api/styles.yaml`): `warm_indoor`, `cool_daylight`, `dim_evening`, `bright_overhead`, `wooden_table`, `white_lab`, `metallic_surface`, `overcast_window`, `tungsten_lamp`, `outdoor_shade`. Bundled views (`views.yaml`): `wrist`, `top`, `front`, `side`, `default`.

Each job writes `sample_NN_<style>.mp4`, the per-modality control videos `sample_NN_<style>_control_<key>.mp4`, and a `manifest.json` (full request, prompts, controls, per-sample timing) into its `job_dir`. **See [`ENGINE_GUIDE.md`](ENGINE_GUIDE.md) for the full request schema, control presets, and all environment variables.**

---

*Everything below is the original upstream NVIDIA documentation.*

---

<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos/"> Product Website</a>&nbsp | 🤗 <a href="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2511.00062">Paper</a> | <a href="https://research.nvidia.com/labs/dir/cosmos-transfer2.5/">Paper Website</a> | <a href="https://github.com/nvidia-cosmos/cosmos-cookbook">Cosmos Cookbook</a>
</p>

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, Cosmos enables developers to rapidly advance physical AI applications such as autonomous vehicles (AVs), robots, and video analytics AI agents.

Cosmos World Foundation Models come in three model types which can all be customized in post-training: [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2.5), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1).

## News
* [February 23, 2026] Released Transfer2.5 Distilled Edge [model](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/distilled/general/edge) and [inference](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md), enabling low latency (edge deployment) inference. More distilled controlnets coming soon.
* [December 19, 2025] Released Image2Image and ImagePrompt capabilities. See the inference guide [here](docs/inference_image.md).
* [December 12, 2025] Released updated checkpoints for [Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) (blur, depth, segmentation, edge), fixed an issue with autoregressive multiview when num_conditional_frames == 0, optimized control video rendering, refreshed documentation, and added a new post-training example for [single-view](docs/post-training_singleview.md) edge/depth/seg/blur modalities.
* [November 25, 2025] Added Blackwell + ARM inference support, Auto/Multiview code fixes, along with fixes for the help menu and CLI overrides, improved guardrail offloading, and LFS enablement for large assets.
* [November 11, 2025] Refactored the Cosmos-Transfer2.5-2B Auto/Multiview code, and updated the Auto/Multiview checkpoints in Hugging Face.
* [November 7, 2025] We added autoregressive sliding window generation mode for generating longer videos. We also added a new multiview cross-attention module, upgraded dependencies to improve support for Blackwell, and updated inference examples and documentation.
* [November 6, 2025] As part of the Cosmos family, we released the recipe, a reference diffusion model and a tokenizer for [synthetic LiDAR point cloud generation](https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-transfer-lidargen) from RGB image!
* [October 28, 2025] We added [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook), a collection of step-by-step recipes and post-training scripts to quickly build, customize, and deploy NVIDIA’s Cosmos world foundation models for robotics and autonomous systems.
* [October 28, 2025] We added the autogeneration of spatiotemporal masking for control inputs when prompt is given, added cosmos-oss, new pyrefly annotations, introduced multi-storage backend in easyio, reorganized internal packages, and boosted Transfer2 speed with Torch Compile tokenizer optimizations.
* [October 21, 2025] We added on-the-fly computation support for depth and segmentation, and fixed multicontrol experiments in [inference](docs/inference.md). Also, updated Docker base image version, and Gradio related documentation.
* [October 13, 2025] Updated Transfer2.5 Auto Multiview [post-training datasets](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training_auto_multiview.md), and setup dependencies to support NVIDIA Blackwell.
* [October 6, 2025] We released [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) and [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) - the next generation of our world simulation models!
* [June 12, 2025] As part of the Cosmos family, we released [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)

## Cosmos-Transfer2.5

Cosmos-Transfer2.5 is a multi-controlnet designed to accept structured input of multiple video modalities including RGB, depth, segmentation and more. Users can configure generation using JSON-based controlnet_specs, and run inference with just a few commands. It supports both single-video inference, automatic control map generation, and multiple GPU setups.

Physical AI trains upon data generated in two important data augmentation workflows.

### Simulation 2 Real Augmentation

Minimizing the need for achieving high fidelity in 3D simulation.

**Input prompt:**
> A contemporary luxury kitchen with marble tabletops. window with beautiful sunset outside. There is an esspresso coffee maker on the table in front of the white robot arm. Robot arm interacts with a coffee cup and coffee maker on the kitchen table.

<table>
  <tr>
    <th>Input Video</th>
    <th>Computed Control</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/20d63162-0fd5-483a-a306-7b8021df5ed9" width="100%" alt="Input video" controls></video>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/131ffe81-cca0-44cd-8547-7b0e49d5253f" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/e4dd3b80-4696-4930-8b05-6d41e37974c2" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/5a816f4d-fdc3-4939-b2b9-141c6ee64d2b" width="100%" alt="Control map video" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/56f76740-ea36-4916-9e94-c983d6b84d28" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Real 2 Real Augmentation

Leveraging sensor captured RGB augmentation.

**Input prompt:**
> Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow.
<table>
  <tr>
    <th>Input Video</th>
    <th>Computed Control</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/4705c192-b8c6-4ba3-af7f-fd968c4a3eeb" width="100%" alt="Input video" controls></video>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/ba92fa5d-2972-463e-af2e-a637a810a463" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/f8e6c351-78b5-4bd6-949b-e1845aa19f63" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/7edf3f46-c4da-403f-b630-d8853a165602" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/ba59f926-c4c2-4232-bdbf-392c53f29a97" width="100%" alt="Control map video" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/8e62af23-3ca4-4e72-97fe-7a337a31d306" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Scaling World State Diversity Examples

Robotic Matrix Diversity Example
<video src="https://github.com/user-attachments/assets/5daee273-5f49-4238-a67f-d63fdb48a4d9" width="100%" alt="Input video" controls></video>

AV Matrix Diversity Example
<video src="https://github.com/user-attachments/assets/51b18d9b-0cb4-44dc-898c-624e3020dcb1" width="100%" alt="Input video" controls></video>

For an example demonstrating how to augment sythentic data with Cosmos Transfer on robotics navigation tasks to improve Sim2Real performance see [Cosmos Transfer Sim2Real for Robotics Navigation Tasks](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-x-mobility/inference.html) in the [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/).

## Cosmos-Transfer2.5 Model Family

Cosmos-Transfer supports data generation in multiple industry verticals, outlined below. Please check back as we continue to add more specialized models to the Transfer family!

[**Cosmos-Transfer2.5-2B**](docs/inference.md): General [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), Distilled [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/distilled/general), trained from the ground up for Physical AI and robotics.

[**Cosmos-Transfer2.5-2B/auto**](docs/inference_auto_multiview.md): Specialized checkpoints, post-trained for Autonomous Vehicle applications. [Multiview checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/auto). For an example demonstrating how to augment sythentic data with Cosmos Transfer on Autonomous Vehicle see [Cosmos Transfer 2.5 Sim2Real for Simulator Videos](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.html) in the [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/).

[**Cosmos-Transfer2.5-2B/robot-multiview-control**](docs/inference_robot_multiview_control.md): Specialized control-conditioned checkpoints for robot multiview applications. Supports 4 control types (depth, edge, visual blur, segmentation) for precise video generation guided by structural information.

## User Guide

* [Setup Guide](docs/setup.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Inference](docs/inference.md)
  * [Auto Multiview](docs/inference_auto_multiview.md)
  * [Image Inference](docs/inference_image.md)
  * [Robot Multiview Control](docs/inference_robot_multiview_control.md)
* [Post-training](docs/post-training.md)
  * [Single View](docs/post-training_singleview.md)
  * [Auto Multiview](docs/post-training_auto_multiview.md)

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
