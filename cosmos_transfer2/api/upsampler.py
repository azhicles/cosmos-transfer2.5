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

"""Optional prompt upsampling via an external LLM.

The upsampler enriches a terse action prompt into a fuller scene description *before* the
per-style suffixes are appended. It runs off-GPU (an external API call) so it never competes
with the diffusion model for VRAM. The default implementation calls the Claude (Anthropic)
Messages API; swap in another ``Upsampler`` to use a different backend (e.g. the in-repo
Cosmos-Reason model) without touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

log = logging.getLogger("cosmos_api")

# Fast, inexpensive model — appropriate for prompt rewriting. Overridable per-request or via
# the COSMOS_API_UPSAMPLE_MODEL env var.
DEFAULT_UPSAMPLE_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = (
    "You rewrite short robot-manipulation video captions into a single richer scene "
    "description for a video-to-video diffusion model. Keep the described ACTION, objects, "
    "and camera viewpoint exactly the same. Add concrete visual detail about scene geometry, "
    "materials, and spatial layout. Do NOT invent a specific colour scheme, lighting mood, or "
    "art style — those are added separately afterwards. Return ONLY the rewritten description "
    "as a single plain-text paragraph, with no preamble."
)


class Upsampler(Protocol):
    """Turns a base prompt into an enriched one. Implementations must be side-effect free
    apart from the upstream API call and must never raise for ordinary failures (return the
    input unchanged instead)."""

    def __call__(self, prompt: str, *, model: str | None = None) -> str: ...


class NoOpUpsampler:
    """Returns the prompt unchanged. Used when upsampling is disabled."""

    def __call__(self, prompt: str, *, model: str | None = None) -> str:  # noqa: D401
        return prompt


class ClaudeUpsampler:
    """Upsample via the Anthropic Messages API.

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY``. If either is missing, or
    the call fails, the original prompt is returned unchanged and a warning is logged — prompt
    upsampling is best-effort and must never block generation.
    """

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model or os.environ.get(
            "COSMOS_API_UPSAMPLE_MODEL", DEFAULT_UPSAMPLE_MODEL
        )

    def __call__(self, prompt: str, *, model: str | None = None) -> str:
        if not prompt.strip():
            return prompt
        try:
            import anthropic
        except ImportError:
            log.warning("anthropic package not installed; skipping prompt upsampling.")
            return prompt
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY not set; skipping prompt upsampling.")
            return prompt

        used_model = model or self.default_model
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=used_model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in response.content if block.type == "text").strip()
            if not text:
                log.warning("Upsampler returned empty text; using original prompt.")
                return prompt
            log.info("Prompt upsampled with %s (%d -> %d chars).", used_model, len(prompt), len(text))
            return text
        except Exception as e:  # best-effort: never block generation on upsampling
            log.warning("Prompt upsampling failed (%s); using original prompt.", e)
            return prompt


def get_upsampler(enabled: bool) -> Upsampler:
    """Return the configured upsampler, or a no-op when disabled."""
    return ClaudeUpsampler() if enabled else NoOpUpsampler()
