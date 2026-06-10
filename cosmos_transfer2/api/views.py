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

"""Loading and resolution of the view -> camera-perspective-phrase map."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger("cosmos_api")

_BUNDLED_VIEWS = Path(__file__).with_name("views.yaml")


@lru_cache(maxsize=4)
def load_view_map(path: str | None = None) -> dict[str, str]:
    """Load {view_name: perspective_phrase}.

    Resolution order: explicit ``path`` > COSMOS_API_VIEWS_FILE env var > bundled views.yaml.
    """
    resolved = path or os.environ.get("COSMOS_API_VIEWS_FILE") or str(_BUNDLED_VIEWS)
    data = yaml.safe_load(Path(resolved).read_text()) or {}
    views = data.get("views", data)
    return {str(k): str(v) for k, v in views.items()}


def resolve_view_hint(view: str | None, view_hint: str | None, path: str | None = None) -> str:
    """Return the camera-perspective phrase to inject into the prompt.

    Precedence: an explicit inline ``view_hint`` wins; otherwise the named ``view`` is looked up;
    an unknown view name yields an empty hint and a warning.
    """
    if view_hint is not None:
        return view_hint.strip()
    if view is None:
        return ""
    view_map = load_view_map(path)
    if view in view_map:
        return view_map[view].strip()
    log.warning("Unknown view '%s' (known: %s); no view hint applied.", view, sorted(view_map))
    return ""
