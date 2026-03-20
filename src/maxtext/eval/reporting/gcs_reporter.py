# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upload eval result files to Google Cloud Storage."""

from __future__ import annotations

import logging
import os

from maxtext.utils.gcs_utils import upload_blob

logger = logging.getLogger(__name__)

def upload_results(local_path: str, gcs_path: str) -> None:
  """Upload local_path to gcs_path.

  Args:
    local_path: Absolute local path to the file to upload.
    gcs_path: Destination GCS path (e.g. gs://<bucket>/eval/).
  """
  if gcs_path.endswith("/"):
    gcs_dest = gcs_path + os.path.basename(local_path)
  else:
    gcs_dest = gcs_path

  upload_blob(gcs_dest, local_path)
  logger.info("Uploaded %s to %s", local_path, gcs_dest)
