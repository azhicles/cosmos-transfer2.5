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

"""Logging configuration for the API.

A single ``cosmos_api`` logger writes structured lines to stdout (captured by `docker logs`).
Per-job file handlers are attached/detached around each job so every request also gets a
self-contained ``job.log`` in its output directory.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "cosmos_api"
_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> logging.Logger:
    """Configure and return the root API logger. Idempotent."""
    log = logging.getLogger(LOGGER_NAME)
    if getattr(log, "_cosmos_configured", False):
        return log
    level = os.environ.get("COSMOS_API_LOG_LEVEL", "INFO").upper()
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    log.addHandler(handler)
    log.propagate = False
    log._cosmos_configured = True  # type: ignore[attr-defined]
    return log


@contextmanager
def job_log_file(log_path: Path):
    """Attach a file handler writing into ``log_path`` for the duration of the block.

    Everything logged to the ``cosmos_api`` logger (from any module) while the block is active
    is also captured in the per-job file.
    """
    log = logging.getLogger(LOGGER_NAME)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    log.addHandler(handler)
    try:
        yield
    finally:
        handler.flush()
        handler.close()
        log.removeHandler(handler)
