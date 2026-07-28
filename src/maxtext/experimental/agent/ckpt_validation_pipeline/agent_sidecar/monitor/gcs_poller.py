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
from maxtext.utils.gcs_utils import gcs_glob_pattern, read_json_from_gcs

logger = logging.getLogger(__name__)

GCS_REPORTS_DIR = "gs://maxtext-validation-agent-reports/"


def check_for_failures():
  """
  Polls GCS for pipeline failures.
  Reads JSON reports from gs://maxtext-validation-agent-reports/
  Returns a dict with run_id, model_name, and log if a failure is found, else None.
  """
  logger.info("Checking for failures in %s", GCS_REPORTS_DIR)
  try:
    # Glob for all unhandled JSON reports
    report_files = gcs_glob_pattern(GCS_REPORTS_DIR + "*.json")
    for report_file in report_files:
      if "handled" in report_file:
        continue

      report_data = read_json_from_gcs(report_file)
      if report_data and report_data.get("status") == "failed":
        logger.info("Detected failure report: %s", report_file)
        return report_data

  except (ValueError, IOError, KeyError, TypeError, OSError) as e:
    logger.error("Error checking GCS for failures: %s", e)

  return None
