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

"""Prompt assembly: combine the view hint, the (optionally upsampled) action prompt, and the
per-style suffix into the final per-sample prompt."""

from __future__ import annotations


def assemble_prompt(view_hint: str, base_prompt: str, style_suffix: str) -> str:
    """Assemble the final prompt sent to the model for one style variation.

    Order: ``<view hint> <base prompt> <style suffix>``. Empty parts are dropped and the
    pieces are joined with single spaces. ``base_prompt`` is the raw action prompt or, when
    upsampling is enabled, the enriched version.
    """
    parts = [p.strip() for p in (view_hint, base_prompt, style_suffix) if p and p.strip()]
    return " ".join(parts)
