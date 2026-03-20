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
import subprocess

logger = logging.getLogger(__name__)


def upload_results(local_path: str, gcs_path: str) -> None:
  """Upload *local_path* to *gcs_path* using ``gsutil cp``.

  If the ``google-cloud-storage`` Python package is available it is used
  directly; otherwise we fall back to shelling out to ``gsutil``.

  Args:
    local_path: Absolute local path to the file to upload.
    gcs_path: Destination GCS path (e.g. ``gs://<gcs_bucket>/eval/``).
              If it ends with ``/``, the basename of *local_path* is appended.

  Raises:
    RuntimeError: If neither ``google-cloud-storage`` nor ``gsutil`` is available.
  """
  if gcs_path.endswith("/"):
    gcs_dest = gcs_path + os.path.basename(local_path)
  else:
    gcs_dest = gcs_path

  # Try the Python client library first.
  try:
    _upload_with_python_client(local_path, gcs_dest)
    return
  except ImportError:
    logger.debug("google-cloud-storage not installed; falling back to gsutil.")
  except Exception as exc:  # pylint: disable=broad-except
    logger.warning("google-cloud-storage upload failed (%s); falling back to gsutil.", exc)

  # Fall back to gsutil subprocess.
  _upload_with_gsutil(local_path, gcs_dest)


def _upload_with_python_client(local_path: str, gcs_dest: str) -> None:
  """Upload using google-cloud-storage Python library."""
  # pylint: disable=import-outside-toplevel
  from google.cloud import storage  # type: ignore

  # gcs_dest format: gs://bucket/path/to/file
  assert gcs_dest.startswith("gs://"), f"Invalid GCS path: {gcs_dest}"
  without_prefix = gcs_dest[len("gs://"):]
  bucket_name, _, blob_name = without_prefix.partition("/")

  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  blob.upload_from_filename(local_path)
  logger.info("Uploaded %s to %s (via Python client)", local_path, gcs_dest)


def _upload_with_gsutil(local_path: str, gcs_dest: str) -> None:
  """Upload using gsutil subprocess."""
  cmd = ["gsutil", "cp", local_path, gcs_dest]
  logger.info("Running: %s", " ".join(cmd))
  result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603, B607
  if result.returncode != 0:
    raise RuntimeError(f"gsutil cp failed:\n{result.stderr}")
  logger.info("Uploaded %s to %s (via gsutil)", local_path, gcs_dest)
