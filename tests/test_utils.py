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

"""Test utilities file for helper for test configuration path selection.

Provides a single helper to return the absolute path to a test config. When
running in decoupled mode (DECOUPLE_GCLOUD=TRUE) the decoupled test config is
returned.
"""

import os
from MaxText.gcloud_stub import is_decoupled
from MaxText.globals import MAXTEXT_PKG_DIR


def get_test_config_path():
  """Return absolute path to the chosen test config file.

  Returns `decoupled_base_test.yml` when decoupled, otherwise `base.yml`.
  """
  base_cfg = "base.yml"
  if is_decoupled():
    base_cfg = "decoupled_base_test.yml"
  return os.path.join(MAXTEXT_PKG_DIR, "configs", base_cfg)


__all__ = ["get_test_config_path"]
