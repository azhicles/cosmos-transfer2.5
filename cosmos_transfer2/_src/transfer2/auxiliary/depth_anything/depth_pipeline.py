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

"""
Video depth estimation pipeline using Video-Depth-Anything.

Usage:
    python depth_pipeline.py \
        --input_video /path/to/video.mp4 \
        --output_video /path/to/depth.mp4 \
        --encoder vits
"""

import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from cosmos_transfer2._src.transfer2.auxiliary.depth_anything.video_depth_model import VideoDepthAnythingModel


def parse_args():
    parser = argparse.ArgumentParser(description="Video depth estimation using Video-Depth-Anything")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the output depth video")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitl"],
        default="vits",
        help="Model encoder size (vits=small, vitl=large)",
    )
    parser.add_argument("--fps", type=int, default=None, help="FPS for output video (default: same as input)")
    return parser.parse_args()


def _transcode_if_av1(path: str) -> tuple[str, bool]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if probe.stdout.strip().lower() != "av1":
        return path, False
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-vcodec", "libdav1d", "-i", path, "-c:v", "libx264", tmp.name],
        check=True,
    )
    return tmp.name, True


def load_video(video_path: str) -> tuple[np.ndarray, int]:
    """Load video as numpy array [T, H, W, C] and get FPS."""
    work_path, _is_tmp = _transcode_if_av1(video_path)
    cap = cv2.VideoCapture(work_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Keep BGR format (DepthAnything V2 expects BGR input)
        frames.append(frame)

    cap.release()
    if _is_tmp:
        os.unlink(work_path)
    return np.array(frames), fps


def save_depth_video(depth_maps: np.ndarray, output_path: str, fps: int = 30):
    """Save depth maps as video [T, H, W] -> mp4."""
    T, H, W = depth_maps.shape

    # Normalize to 0-255 and convert to uint8
    # Normalize to 0-255 and convert to uint8
    depth_range = depth_maps.max() - depth_maps.min()
    if depth_range == 0:
        depth_normalized = np.zeros_like(depth_maps, dtype=np.uint8)
    else:
        depth_normalized = ((depth_maps - depth_maps.min()) / depth_range * 255).astype(np.uint8)

    # Convert to 3-channel (replicate grayscale for video compatibility)
    depth_3ch = np.stack([depth_normalized] * 3, axis=-1)  # [T, H, W, 3]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write video (OpenCV VideoWriter expects BGR format)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in depth_3ch:
        # Frame is already in BGR format (grayscale replicated)
        out.write(frame)

    out.release()
    print(f"Saved depth video to: {output_path}")


def main():
    args = parse_args()

    print("Initializing Video-Depth-Anything model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    model = VideoDepthAnythingModel(encoder=args.encoder, device=device)
    model.setup()  # Downloads weights if needed
    print(f"Loaded {args.encoder} model")

    # Load video
    print(f"Loading video: {args.input_video}")
    video_frames, fps = load_video(args.input_video)
    print(f"Loaded {len(video_frames)} frames at {fps} FPS, shape: {video_frames.shape}")

    # Generate depth maps
    print("Generating depth maps...")
    depth_maps = model.generate(video_frames)  # [T, H, W]
    print(f"Generated depth maps, shape: {depth_maps.shape}")

    # Save output
    output_fps = args.fps if args.fps else fps
    save_depth_video(depth_maps, args.output_video, output_fps)

    print("✓ Depth estimation complete!")


if __name__ == "__main__":
    main()
