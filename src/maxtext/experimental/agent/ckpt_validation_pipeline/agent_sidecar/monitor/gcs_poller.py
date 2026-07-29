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


def check_for_failures():
  """
  Polls GCS for pipeline failures.
  Reads JSON reports from gs://maxtext-validation-agent-reports/
  Returns a dict with run_id, model_name, and log if a failure is found, else None.
  """
  logger.info("Checking for failures in gs://%s/", GCS_BUCKET_NAME)
  try:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = bucket.list_blobs()
    
    for blob in blobs:
      if not blob.name.endswith(".json") or "handled" in blob.name:
        continue

      content = blob.download_as_string()
      report_data = json.loads(content)
      if report_data and report_data.get("status") == "failed":
        logger.info("Detected failure report: %s", blob.name)
        return report_data

  except Exception as e:
    logger.error("Error checking GCS for failures: %s", e)

  return None
