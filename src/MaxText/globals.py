# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path

# This is the MaxText root: with "max_utils.py"; &etc. TODO: Replace `os.path.basename` with `os.path.abspath`
MAXTEXT_PKG_DIR = os.environ.get("MAXTEXT_PKG_DIR", os.path.basename(os.path.dirname(__file__)))

# This is the maxtext repo root: with ".git" folder; "README.md"; "pyproject.toml"; &etc.
MAXTEXT_REPO_ROOT = os.environ.get(
    "MAXTEXT_REPO_ROOT",
    r
    if os.path.isdir(os.path.join(r := os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".git"))
    else MAXTEXT_PKG_DIR,
)

# This is the assets root: with "tokenizer.gemma3"; &etc.
MAXTEXT_ASSETS_ROOT = os.environ.get("MAXTEXT_ASSETS_ROOT", os.path.join(MAXTEXT_PKG_DIR, "assets"))

# This is the test assets root: with "test_image.jpg"; &etc.
MAXTEXT_TEST_ASSETS_ROOT = os.environ.get("MAXTEXT_TEST_ASSETS_ROOT", os.path.join(MAXTEXT_PKG_DIR, "test_assets"))

EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

__all__ = [
    "DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE",
    "EPS",
    "MAXTEXT_ASSETS_ROOT",
    "MAXTEXT_PKG_DIR",
    "MAXTEXT_REPO_ROOT",
    "MAXTEXT_TEST_ASSETS_ROOT",
]
