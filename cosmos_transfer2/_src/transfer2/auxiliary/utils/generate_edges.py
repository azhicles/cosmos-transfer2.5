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

import argparse
import os
import subprocess
import tempfile

import cv2


def ranged_float(min_val: float, max_val: float):
    def checker(x: str) -> float:
        x_float = float(x)
        if not (min_val <= x_float <= max_val):
            raise argparse.ArgumentTypeError(f"Value must be between {min_val} and {max_val}")
        return x_float

    return checker


def ranged_int(min_val: int, max_val: int):
    def checker(x: str) -> int:
        x_int = int(x)
        if not (min_val <= x_int <= max_val):
            raise argparse.ArgumentTypeError(f"Value must be between {min_val} and {max_val}")
        return x_int

    return checker


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


def generate_edges(in_path: str, out_path: str, bright: int = 50, contrast: float = 1.0) -> None:
    work_path, _is_tmp = _transcode_if_av1(in_path)
    cap = cv2.VideoCapture(work_path)
    assert cap.isOpened(), "Could not open input video."
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 10, 50)
        out.write(edges)

    cap.release()
    out.release()
    if _is_tmp:
        os.unlink(work_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edge video from input.")

    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_video", help="Path to save generated edge video")

    parser.add_argument(
        "--bright",
        type=ranged_int(-255, 255),
        default=50,
        help="Brightness offset (-255 to 255). Default: 50",
    )
    parser.add_argument(
        "--contrast",
        type=ranged_float(0.0, 5.0),
        default=1.0,
        help="Contrast multiplier (0.0 to 5.0). Default: 1.0",
    )

    args = parser.parse_args()

    generate_edges(args.input_video, args.output_video, bright=args.bright, contrast=args.contrast)


"""
Usage (MP4 output):
  
python cosmos_transfer2/_src/transfer2/auxiliary/utils/generate_edges.py \
input_video.mp4 \
edge.mp4 \
--bright 50 \
--contrast 1
"""
