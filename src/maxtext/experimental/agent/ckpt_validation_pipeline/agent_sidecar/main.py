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

from monitor.state_manager import load_state, save_state, MAX_RETRIES
from monitor.alerter import dispatch_email_alert
from monitor.gcs_poller import check_for_failures, mark_handled
from adk_agent import run_agent_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _check_and_send_victory_laps(state):
  """Checks if any previously failing runs have succeeded and dispatches Victory Lap emails."""
  from monitor.alerter import dispatch_victory_lap_alert
  for run_id, entry in list(state.items()):
    if isinstance(entry, dict) and entry.get("retries", 0) > 0 and not entry.get("victory_lap_sent", False):
      logger.info("Run ID %s succeeded after %s retries! Sending Victory Lap email...", run_id, entry["retries"])
      last_attempt = entry.get("attempts", [{}])[-1] if entry.get("attempts") else {}
      pr_url = last_attempt.get("pr_url", "")
      dispatch_victory_lap_alert(run_id=run_id, model_name=entry.get("model", "unknown"), pr_url=pr_url)
      entry["victory_lap_sent"] = True
  save_state(state)


def main():
  """Entrypoint for the Cloud Run Job. Executes once and terminates."""
  logger.info("Overwatch Cloud Run Job started. Checking for pipeline failures...")

  try:
    # Check if Airflow passed failure context directly via environment overrides
    airflow_error = os.environ.get("AIRFLOW_ERROR_MESSAGE", "").strip()
    if airflow_error:
      logger.info("Detected direct failure context from Airflow on_failure_callback!")
      run_id = os.environ.get("RUN_NAME", "airflow_run")
      model_name = os.environ.get("MAXTEXT_MODEL_NAME", "unknown_model")
      run_agent_workflow(run_id, model_name, airflow_error, "airflow_callback")
      return

    # Check if Airflow passed failure context via direct GCS trigger blob (roles/run.invoker compatible)
    from monitor.gcs_poller import check_for_direct_airflow_failures
    direct_trigger = check_for_direct_airflow_failures()
    if direct_trigger:
      logger.info("Detected direct failure trigger blob from GCS!")
      run_id = direct_trigger.get("run_name", "airflow_run")
      model_name = direct_trigger.get("maxtext_model_name", "unknown_model")
      error_msg = direct_trigger.get("airflow_error_message", "")
      if direct_trigger.get("hf_ref_code_url"):
        os.environ["HF_REF_CODE_URL"] = direct_trigger["hf_ref_code_url"]
      if direct_trigger.get("hf_config_url"):
        os.environ["HF_CONFIG_URL"] = direct_trigger["hf_config_url"]
      if direct_trigger.get("alert_recipient"):
        os.environ["ALERT_RECIPIENT"] = direct_trigger["alert_recipient"]
      if direct_trigger.get("maxtext_branch"):
        os.environ["MAXTEXT_BRANCH"] = direct_trigger["maxtext_branch"]
      run_agent_workflow(run_id, model_name, error_msg, "airflow_callback")
      return

    failure, blob_name = check_for_failures()
    if not failure:
      logger.info("No failures detected in GCS. Checking if any retry runs completed successfully...")
      state = load_state()
      _check_and_send_victory_laps(state)
      logger.info("Exiting cleanly.")
      return

    run_id = (
        failure.get("run_name") or failure.get("run_id") or failure.get("task") or failure.get("stage") or "unknown_run"
    )
    model_name = failure.get("model") or failure.get("model_name", "unknown")
    failure_log = failure.get("stderr") or failure.get("error_message") or failure.get("log", "")
    if not failure_log or failure_log == "Success":
      failure_log = failure.get("stdout", "No logs provided.")

    state = load_state()
    retries = state.get(run_id, 0)

    if retries >= MAX_RETRIES:
      logger.info("Run ID %s has hit the maximum of %s retries. Escalating to human.", run_id, MAX_RETRIES)
      dispatch_email_alert(run_id, model_name)
      state[run_id] = retries + 1  # Mark as handled
      save_state(state)
      mark_handled(blob_name)
    elif retries < MAX_RETRIES:
      logger.info("Detected failure for %s. Attempt %s/%s.", run_id, retries + 1, MAX_RETRIES)

      # Immediately mark the report as handled and increment retries to prevent duplicate concurrent triggers
      state[run_id] = retries + 1
      save_state(state)
      if not mark_handled(blob_name):
        logger.info("Report %s was already claimed or deleted by a concurrent worker. Exiting cleanly.", blob_name)
        return

      # Trigger the ADK workflow instead of shelling out to agentapi CLI
      run_agent_workflow(run_id, model_name, failure_log, blob_name)

  except Exception as e:
    logger.error("Error during job execution: %s", e)
    sys.exit(1)


if __name__ == "__main__":
  main()
