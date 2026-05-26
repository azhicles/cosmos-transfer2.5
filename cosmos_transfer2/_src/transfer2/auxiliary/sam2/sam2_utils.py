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

import os
import subprocess
import time

import cv2
import imageio
import numpy as np
import pycocotools.mask
import torch
from natsort import natsorted
from PIL import Image
from torchvision import transforms

from cosmos_transfer2._src.transfer2.datasets.augmentors.seg import (
    decode_partial_rle_width1,
    segmentation_color_mask,
)


def write_video(frames, output_path, fps=30):
    """
    expects a sequence of [H, W, 3] or [H, W] frames
    """
    with imageio.get_writer(output_path, fps=fps, macro_block_size=8) as writer:
        for frame in frames:
            if len(frame.shape) == 2:  # single channel
                frame = frame[:, :, None].repeat(3, axis=2)
            writer.append_data(frame)


def capture_fps(input_video_path: str):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    time_start = time.time()

    # Detect video codec so we can choose the right decoder.
    # OpenCV's bundled FFmpeg may fail on AV1 when hardware decoding is unavailable,
    # so we call FFmpeg directly with an explicit software decoder for AV1.
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_loc,
        ],
        capture_output=True,
        text=True,
    )
    codec = probe.stdout.strip().lower()

    cmd = ["ffmpeg", "-y"]
    if codec == "av1":
        cmd += ["-vcodec", "libdav1d"]
    cmd += [
        "-i", input_loc,
        "-q:v", "2",
        "-start_number", "1",
        os.path.join(output_loc, "%05d.jpg"),
    ]

    print("Converting video..\n")
    subprocess.run(cmd, check=True)

    count = len([f for f in os.listdir(output_loc) if f.endswith(".jpg")])
    time_end = time.time()
    print("Done extracting frames.\n%d frames extracted" % count)
    print("It took %d seconds for conversion." % (time_end - time_start))


# Function to generate video
def convert_masks_to_frames(masks: list, num_masks_max: int = 100):
    T, H, W = shape = masks[0]["segmentation_mask_rle"]["mask_shape"]
    frame_start, frame_end = 0, T
    num_masks = min(num_masks_max, len(masks))
    mask_ids_select = np.arange(num_masks).tolist()

    all_masks = np.zeros((num_masks, T, H, W), dtype=np.uint8)
    for idx, mid in enumerate(mask_ids_select):
        mask = masks[mid]
        num_byte_per_mb = 1024 * 1024
        # total number of elements in uint8 (1 byte) / num_byte_per_mb
        if shape[0] * shape[1] * shape[2] / num_byte_per_mb > 256:
            rle = decode_partial_rle_width1(
                mask["segmentation_mask_rle"]["data"],
                frame_start * shape[1] * shape[2],
                frame_end * shape[1] * shape[2],
            )
            partial_shape = (frame_end - frame_start, shape[1], shape[2])
            rle = rle.reshape(partial_shape) * 255
        else:
            rle = pycocotools.mask.decode(mask["segmentation_mask_rle"]["data"])
            rle = rle.reshape(shape) * 255
            # Select the frames that are in the video
            frame_indices = np.arange(frame_start, frame_end).tolist()
            rle = np.stack([rle[i] for i in frame_indices])
        all_masks[idx] = rle
        del rle

    all_masks = segmentation_color_mask(all_masks)  # NTHW -> 3THW
    all_masks = all_masks.transpose(1, 2, 3, 0)
    return all_masks


def generate_video_from_images(masks: list, output_file_path: str, fps, num_masks_max: int = 100):
    all_masks = convert_masks_to_frames(masks, num_masks_max)
    write_video(all_masks, output_file_path, fps)
    print("Video generated successfully!")


def generate_tensor_from_images(
    image_path_str: str, output_file_path: str, fps, search_pattern: str = None, weight_scaler: float = None
):
    images = list()
    image_path = os.path.abspath(image_path_str)
    if search_pattern is None:
        images = [img for img in natsorted(os.listdir(image_path))]
    else:
        for img in natsorted(os.listdir(image_path)):
            if img.__contains__(search_pattern):
                images.append(img)

    transform = transforms.ToTensor()
    image_tensors = list()
    for image in images:
        img_tensor = transform(Image.open(os.path.join(image_path, image)))
        image_tensors.append(img_tensor.squeeze(0))

    tensor = torch.stack(image_tensors)  # [T, H, W], binary values, float

    if weight_scaler is not None:
        print(f"scaling the tensor by the specified scale: {weight_scaler}")
        tensor = tensor * weight_scaler

    print(f"saving tensor shape: {tensor.shape} to {output_file_path}")
    torch.save(tensor, output_file_path)


if __name__ == "__main__":
    input_loc = "cosmos_transfer2/models/sam2/assets/input_video.mp4"
    output_loc = "cosmos_transfer2/models/sam2/assets/outputs/"
    # output_loc = os.path.abspath(tempfile.TemporaryDirectory().name)
    print(f"output_loc --- {output_loc}")
    video_to_frames(input_loc, output_loc)
