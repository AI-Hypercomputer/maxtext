# Copyright 2023-2026 Google LLC
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

"""Test helpers file for helper for test configuration path selection.

Provides helpers to return common test configuration values. When running in
decoupled mode (DECOUPLE_GCLOUD=TRUE), these helpers return local paths instead
of Google Cloud Storage paths.
"""

import os
from maxtext.common.gcloud_stub import is_decoupled
from MaxText.globals import MAXTEXT_CONFIGS_DIR


def get_test_config_path():
  """Return absolute path to the chosen test config file.

  Returns `decoupled_base_test.yml` when decoupled, otherwise `base.yml`.
  """
  base_cfg = "base.yml"
  if is_decoupled():
    base_cfg = "decoupled_base_test.yml"
  return os.path.join(MAXTEXT_CONFIGS_DIR, base_cfg)


def get_test_dataset_path(cloud_path=None):
  """Return the dataset path for tests.

  Args:
    cloud_path: Optional custom GCS path to use in cloud mode.
                Defaults to "gs://maxtext-dataset" if not specified.

  Returns:
    Local minimal dataset path when decoupled, otherwise returns
    the specified cloud path or default GCS maxtext-dataset bucket.
  """
  if is_decoupled():
    return os.path.join("tests", "assets", "local_datasets", "c4_en_dataset_minimal")
  return cloud_path or "gs://maxtext-dataset"


def get_test_base_output_directory(cloud_path=None):
  """Return the base output directory for test logs and checkpoints.

  Args:
    cloud_path: Optional custom GCS path to use in cloud mode.
                Defaults to "gs://runner-maxtext-logs" if not specified.

  Returns:
    Local test logs directory when decoupled, otherwise returns
    the specified cloud path or default GCS runner-maxtext-logs bucket.
  """
  if is_decoupled():
    return os.path.join("maxtext_local_output", "gcloud_decoupled_test_logs")
  return cloud_path or "gs://runner-maxtext-logs"


__all__ = [
    "get_test_base_output_directory",
    "get_test_config_path",
    "get_test_dataset_path",
]
