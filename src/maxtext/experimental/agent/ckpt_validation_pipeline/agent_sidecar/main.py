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

"""Overwatch Cloud Run Job Entrypoint."""

import os
import logging
import sys
import json

from monitor.state_manager import MAX_RETRIES, can_attempt, update_run_state
from monitor.alerter import dispatch_email_alert
from monitor.gcs_poller import check_for_failures, mark_handled
from adk_agent import run_agent_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  """Entrypoint for the Cloud Run Job. Executes once and terminates."""
  logger.info("Overwatch Cloud Run Job started. Checking for pipeline failures...")

  try:
    # Check if Airflow passed failure context directly via environment overrides
    airflow_error = os.environ.get("AIRFLOW_ERROR_MESSAGE", "").strip()
    if airflow_error:
      logger.info("Detected direct failure context from Airflow on_failure_callback!")
      context = {
          "remediation_key": os.environ.get("REMEDIATION_KEY", os.environ.get("RUN_NAME", "airflow_run")),
          "run_name": os.environ.get("RUN_NAME", "airflow_run"),
          "maxtext_model_name": os.environ.get("MAXTEXT_MODEL_NAME", "unknown_model"),
          "airflow_dag_id": os.environ.get("TARGET_DAG_ID", ""),
          "airflow_task_id": os.environ.get("AIRFLOW_TASK_ID", ""),
          "airflow_run_id": os.environ.get("AIRFLOW_RUN_ID", ""),
      }
      run_key = context["remediation_key"]
      if not can_attempt(run_key):
        update_run_state(run_key, status="exhausted", max_attempts=MAX_RETRIES)
        logger.error("Run %s exhausted its %s patch attempts", run_key, MAX_RETRIES)
        return
      run_agent_workflow(context, airflow_error)
      return

    # Check if Airflow passed failure context via direct GCS trigger blob (roles/run.invoker compatible)
    from monitor.gcs_poller import check_for_direct_airflow_failures
    direct_trigger = check_for_direct_airflow_failures()
    if direct_trigger:
      logger.info("Detected direct failure trigger blob from GCS!")
      run_key = direct_trigger.get("remediation_key") or direct_trigger.get("run_name", "airflow_run")
      error_msg = direct_trigger.get("airflow_error_message", "")
      if direct_trigger.get("airflow_dag_id"):
        os.environ["TARGET_DAG_ID"] = direct_trigger["airflow_dag_id"]
      if direct_trigger.get("hf_ref_code_url"):
        os.environ["HF_REF_CODE_URL"] = direct_trigger["hf_ref_code_url"]
      if direct_trigger.get("hf_config_url"):
        os.environ["HF_CONFIG_URL"] = direct_trigger["hf_config_url"]
      if direct_trigger.get("alert_recipient"):
        os.environ["ALERT_RECIPIENT"] = direct_trigger["alert_recipient"]
      if direct_trigger.get("maxtext_branch"):
        os.environ["MAXTEXT_BRANCH"] = direct_trigger["maxtext_branch"]
      if not can_attempt(run_key):
        update_run_state(run_key, status="exhausted", max_attempts=MAX_RETRIES)
        logger.error("Run %s exhausted its %s patch attempts", run_key, MAX_RETRIES)
        return
        
      # CRITICAL BUG FIX: Fetch the actual detailed logit/shape divergence report from the bucket
      # because Airflow's exception trace does NOT contain the mathematical failure details.
      validator_report, report_blob = check_for_failures(expected_run_name=direct_trigger.get("run_name"))
      if validator_report:
        error_msg += f"\n\n--- DETAILED VALIDATOR REPORT ---\n{json.dumps(validator_report, indent=2)}"
        mark_handled(report_blob)
        
      run_agent_workflow(direct_trigger, error_msg)
      return

    logger.info("No direct Airflow failure trigger blobs found in GCS. Exiting cleanly.")
    return

  except Exception as e:
    logger.error("Error during job execution: %s", e)
    try:
      from monitor.alerter import dispatch_emergency_alert
      dispatch_emergency_alert(str(e))
    except Exception as alert_err:
      logger.error("Failed to send emergency alert: %s", alert_err)
    sys.exit(1)


if __name__ == "__main__":
  main()
