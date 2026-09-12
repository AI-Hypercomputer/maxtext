# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCS poller for detecting Airflow validation failures."""

import logging
import json
from google.cloud import storage

logger = logging.getLogger(__name__)

GCS_BUCKET_NAME = "maxtext-validation-agent-reports"


def check_for_failures(expected_run_name=None):
  """
  Polls GCS for pipeline failures.
  Reads JSON reports from gs://maxtext-validation-agent-reports/
  Returns a tuple (report_data, blob_name) if a failure is found, else (None, None).
  """
  logger.info("Checking for failures in gs://%s/", GCS_BUCKET_NAME)
  report_data_out = None
  blob_name_out = None
  
  try:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = bucket.list_blobs()

    # Filter for unhandled json reports (we want the detailed validator reports, NOT airflow direct triggers)
    valid_blobs = [
        b
        for b in blobs
        if b.name.endswith(".json") and "handled" not in b.name and not b.name.startswith("airflow_direct_failure_")
    ]

    # Sort by creation time descending (newest first)
    valid_blobs.sort(key=lambda b: b.time_created, reverse=True)

    for blob in valid_blobs:
      content = blob.download_as_string()
      try:
        report_data = json.loads(content)
      except json.JSONDecodeError:
        continue

      if expected_run_name and report_data.get("run_name") != expected_run_name:
        continue

      # Check for "failed" (shape check), "FAILURE" (mock tensor), or success == False (forward pass / decode)
      if report_data and (
          report_data.get("status") in ("failed", "FAILED", "FAILURE") or report_data.get("success") is False
      ):
        logger.info("Detected failure report: %s", blob.name)
        report_data_out = report_data
        blob_name_out = blob.name
        break

  except Exception as e:
    logger.error("Error checking GCS for failures: %s", e)

  return report_data_out, blob_name_out


def check_for_direct_airflow_failures():
  """Checks GCS for direct Airflow on_failure_callback trigger blobs (airflow_direct_failure_*.json)."""
  logger.info("Checking for direct Airflow failure triggers in gs://%s/", GCS_BUCKET_NAME)
  try:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="airflow_direct_failure_"))
    valid_blobs = [b for b in blobs if b.name.endswith(".json")]
    valid_blobs.sort(key=lambda b: b.time_created, reverse=True)

    for blob in valid_blobs:
      content = blob.download_as_string()
      data = json.loads(content)
      logger.info("Detected direct Airflow failure trigger blob: %s", blob.name)
      try:
        bucket.rename_blob(blob, "handled_" + blob.name, if_source_generation_match=blob.generation)
        return data
      except Exception as e:
        logger.info("Another container likely claimed %s (Error: %s), skipping to next...", blob.name, e)
        continue
  except Exception as e:
    logger.error("Error checking GCS for direct Airflow failure triggers: %s", e)
  return None


def mark_handled(blob_name):
  """Renames a blob to include 'handled_' so it is ignored in future polls."""
  try:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    if blob.exists():
      # Prepend "handled_" to the original filename
      new_name = "handled_" + blob_name
      bucket.rename_blob(blob, new_name)
      logger.info("Successfully marked %s as handled.", blob_name)
      return True
    return False
  except Exception as e:
    logger.error("Failed to mark blob %s as handled: %s", blob_name, e)
    return False
