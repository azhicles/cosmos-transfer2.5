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

"""ffmpeg stacking/splitting for unified dual-view inference.

Ported from scripts/run_tictactoe_unified.py. Stacking top+wrist into one video lets a single
diffusion pass restyle both views with one latent trajectory, so the per-view outputs are
frame-locked in style/colour/lighting. The output is split back into the two views afterwards.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

Stack = Literal["vstack", "hstack"]


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def create_combined_video(top_path: Path, wrist_path: Path, out_path: Path, stack: Stack) -> None:
    """Stack top and wrist into one combined video.

    vstack (default): top above wrist (e.g. 640x960). Detected as 3:4 aspect, processed at the
    same latent token count as single-view 4:3 — VRAM-safe on a 96 GB GPU.
    hstack: top left of wrist (e.g. 1280x480). Detected as 16:9, ~33% more latent tokens.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filter_str = "hstack=inputs=2" if stack == "hstack" else "vstack=inputs=2"
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(top_path),
        "-i", str(wrist_path),
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(out_path),
    ])


def split_combined_video(combined_path: Path, top_out: Path, wrist_out: Path, stack: Stack) -> None:
    """Split a combined output video back into individual top and wrist files."""
    top_out.parent.mkdir(parents=True, exist_ok=True)
    if stack == "hstack":
        # left half = top, right half = wrist
        top_crop, wrist_crop = "crop=iw/2:ih:0:0", "crop=iw/2:ih:iw/2:0"
    else:
        # top half = top, bottom half = wrist
        top_crop, wrist_crop = "crop=iw:ih/2:0:0", "crop=iw:ih/2:0:ih/2"
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(combined_path), "-filter:v", top_crop,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", str(top_out)])
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(combined_path), "-filter:v", wrist_crop,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", str(wrist_out)])
