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

"""Loading and resolution of the style library."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from cosmos_transfer2.api.config_models import StyleEntry, StyleRef

_BUNDLED_STYLES = Path(__file__).with_name("styles.yaml")


@lru_cache(maxsize=4)
def load_style_library(path: str | None = None) -> dict[str, StyleEntry]:
    """Load the style library from YAML into {name: StyleEntry}.

    Resolution order for the path: explicit ``path`` arg > COSMOS_API_STYLES_FILE env var >
    bundled styles.yaml.
    """
    resolved = path or os.environ.get("COSMOS_API_STYLES_FILE") or str(_BUNDLED_STYLES)
    data = yaml.safe_load(Path(resolved).read_text()) or {}
    raw = data.get("styles", data)
    library: dict[str, StyleEntry] = {}
    for name, entry in raw.items():
        if isinstance(entry, str):
            entry = {"suffix": entry}
        library[name] = StyleEntry(name=name, **entry)
    if not library:
        raise ValueError(f"No styles found in style library: {resolved}")
    return library


def default_style_names(n: int, path: str | None = None) -> list[str]:
    """First ``n`` style names from the library, in file order."""
    names = list(load_style_library(path).keys())
    return names[:n]


def resolve_styles(styles: list[StyleRef] | None, num_samples: int, path: str | None = None) -> list[StyleEntry]:
    """Resolve a request's ``styles`` field into concrete StyleEntry objects.

    - None -> the first ``num_samples`` bundled defaults.
    - a list -> each item is either a style name (looked up) or an inline StyleEntry.
    """
    library = load_style_library(path)
    if styles is None:
        return [library[name] for name in default_style_names(num_samples, path)]

    resolved: list[StyleEntry] = []
    for item in styles:
        if isinstance(item, str):  # a name reference
            if item not in library:
                raise KeyError(
                    f"Unknown style '{item}'. Available: {sorted(library)}. "
                    "Pass an inline style object to use a custom one."
                )
            resolved.append(library[item])
        elif isinstance(item, StyleEntry):
            resolved.append(item)
        else:  # an inline dict
            resolved.append(StyleEntry.model_validate(item))
    return resolved
