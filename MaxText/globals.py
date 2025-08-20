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

PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # MaxText directory path
EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

__all__ = ["DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE", "EPS", "PKG_DIR"]
