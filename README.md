# Dell Synthetic Data Generation

A fork of [NVIDIA Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) used to generate synthetic training data for robot manipulation. Given a recorded episode, it produces several restyled versions of the same motion, changing lighting, materials, and camera appearance while keeping the geometry and action fixed. The variations serve as domain-randomized augmentations for downstream vision models.

The original NVIDIA documentation is preserved further down. The sections below cover how to install the project, how to run a generation job, and what this fork changes relative to upstream.

## Installation

You will need an NVIDIA GPU, [`uv`](https://docs.astral.sh/uv/), and [`just`](https://github.com/casey/just).

Install the project with `just`, which creates a local `.venv` on Python 3.10 and installs the matching CUDA build:

```bash
just install cu128     # x86_64 and most GPUs
just install cu130     # Blackwell, DGX Spark, or aarch64
```

The 2B checkpoints download from Hugging Face on first use and are cached locally. If they are not already present, provide a token, and optionally point the cache at a larger disk:

```bash
export HF_TOKEN=hf_...
export HF_HOME=/mnt/hf-cache     # optional
```

Generation runs are tracked in ClearML. Install it and configure credentials once:

```bash
uv pip install clearml
clearml-init                     # writes ~/clearml.conf
```

Optional prompt upsampling through an LLM is off by default; set `ANTHROPIC_API_KEY` to enable it.

Always run inside the managed environment, using either `just run <cmd>` or `uv run --no-sync <cmd>`. Do not install packages into the system Python or invoke `python` directly. Check `nvidia-smi` before starting a job, since the GPU is shared.

## Running a job

A job takes one input video and produces several style variations of it. The controls (edge, depth, segmentation, and blur) hold the scene geometry and motion in place, while a per-style prompt suffix changes the appearance. The checkpoint is loaded once and the control pass is computed once, then reused across every style.

### With ClearML

`scripts/clearml_task.py` is the main entry point. Edit the `PARAMS` block at the top of the file to set the episode list, prompt, styles, step count, and control preset, then run:

```bash
uv run --no-sync python scripts/clearml_task.py
```

The run appears in the ClearML web UI under the **Cosmos Transfer** project, with console output, per-sample timing, and each output video shown inline. Every parameter is registered with the task, so you can also override it from the UI when cloning or queuing a run. Because a whole episode list is processed in a single task, the checkpoint loads only once for the entire batch.

### From Python

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

engine = InferenceEngine(work_dir=Path("outputs/_work"))
engine.run_job(req, Path(req.video_path), job_dir=Path("outputs/job01"))
```

Only `prompt` and `video_path` are required; every other field has a default. Reuse a single `InferenceEngine` across jobs so the model stays resident in memory. Stacked top-and-wrist episodes go through `run_dual_view_job` with a `DualViewRequest`, which keeps both views locked to the same style.

### Common parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `styles` | *(list)* | Styles to generate; the length of the list sets the number of variations. |
| `num_steps` | `15` | Diffusion steps. Around 6 for quick tests, 35 for final quality. |
| `sigma_max` | `110` | Primary fidelity/freedom control (0–200). Higher values restyle more aggressively. |
| `guidance` | `5` | Classifier-free guidance strength (0–7). Higher values follow the prompt more closely. |
| `control_preset` | `balanced` | `balanced` (depth, segmentation, edge), `multicontrol_robot` (adds a colour anchor), or `edge` (single, fast control). |
| `resolution` | `720` | Output resolution. |
| `guided_generation` | `False` | Holds a foreground object such as the robot arm fixed while the background is restyled. |

The bundled styles are defined in `cosmos_transfer2/api/styles.yaml` (`warm_indoor`, `cool_daylight`, `dim_evening`, `bright_overhead`, `wooden_table`, `white_lab`, `metallic_surface`, `overcast_window`, `tungsten_lamp`, `outdoor_shade`) and the camera views in `views.yaml` (`wrist`, `top`, `front`, `side`, `default`).

### Output

Each job writes into its own directory: one `sample_NN_<style>.mp4` per variation, the control videos used to produce them, and a `manifest.json` recording the request, resolved prompts, and per-sample timing. See [`ENGINE_GUIDE.md`](ENGINE_GUIDE.md) for the full request schema and the complete list of environment variables.

## What this fork changes

Wherever possible the upstream engine is used unmodified. The additions are:

- **Restyling engine** (`cosmos_transfer2/api/`) — a library around the Cosmos inference pipeline that produces several domain-randomized variations from a single episode. It runs in-process, with no server or container, and is driven by ClearML. The main modules are `engine.py` (`InferenceEngine`), `config_models.py` (the request schema and control presets), `styles.yaml`, and `views.yaml`.
- **Tic-tac-toe dataset work** — the assets in `assets/tictactoe/` (control specs, prompts, style references) together with the driver scripts `scripts/run_tictactoe_unified.py` (dual-view runner) and `scripts/run_outdoor_180.py` (resumable 180-episode batch). The experiment harness (`scripts/exp_run_batch.py`, `exp_report.py`, `exp_pareto.py`) sweeps generation parameters and reports the time, memory, and realism trade-offs written up in `RESULTS.md`.
- **Hardware fixes** in the vendored engine (`cosmos_transfer2/_src/`) — small patches needed to run on this stack, notably an NVENC bitrate fix for Blackwell and adjustments to the depth, SAM2, and edge auxiliary pipelines. The rest of `_src/` is treated as an unmodified upstream dependency.
- **Documentation** — [`CLAUDE.md`](CLAUDE.md) for architecture and working conventions, and [`ENGINE_GUIDE.md`](ENGINE_GUIDE.md) for the request schema, control presets, and environment variables. The environment is managed with `uv` and a `justfile`.

---

The original NVIDIA Cosmos-Transfer2.5 README follows.

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
