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

"""In-process Cosmos Transfer 2.5 engine library for video appearance/style variation.

Provides ``InferenceEngine`` (loads the checkpoint once, generates N domain-randomization
style variations from one control pass) plus the request schema and style/view libraries. It is
called in-process — there is no HTTP server — and driven by the ClearML Task in
``scripts/clearml_task.py``. See ``ENGINE_GUIDE.md`` at the repository root for usage.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
