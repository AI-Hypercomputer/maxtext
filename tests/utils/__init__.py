# Copyright 2025 Google LLC
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

"""Shim for importing test_helpers that is used for decoupled mode."""

from .test_helpers import (
    get_test_base_output_directory,
    get_test_config_path,
    get_test_dataset_path,
)

__all__ = [
    "get_test_base_output_directory",
    "get_test_config_path",
    "get_test_dataset_path",
]
