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

"""Email alerting module for manual escalation."""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_email_script_path() -> str:
  """Returns dynamic absolute path to send_email.py across Linux/macOS environments."""
  return str(Path(__file__).resolve().parents[2] / "send_email.py")


def _get_default_recipient() -> str:
  import os
  return os.environ.get("ALERT_RECIPIENT") or os.environ.get("USER_EMAIL") or ""


def dispatch_email_alert(run_id, model_name, recipient="", state_entry=None):
  """Triggers send_email.py for Terminal Failure (Distress Signal) with rich attempt history."""
  email_script = _get_email_script_path()
  recipient = recipient or _get_default_recipient()
  if not recipient:
    logger.warning("No email recipient configured (ALERT_RECIPIENT/USER_EMAIL empty). Skipping distress signal email.")
    return

  subject = f"Pipeline Halted: {model_name} (Run: {run_id})"

  if state_entry and isinstance(state_entry, dict) and state_entry.get("attempts"):
    last_attempt = state_entry["attempts"][-1]
    branch = last_attempt.get("branch", "unknown-branch")
    diagnosis = last_attempt.get("diagnosis", "No analysis recorded.")
    hypothesis = last_attempt.get("hypothesis", "No hypothesis recorded.")
    body = (
        f"Pipeline Halted: I attempted to auto-fix {model_name} 25 times but was unsuccessful.\n\n"
        f"Here is my analysis of the divergence:\n{diagnosis}\n\n"
        f"The code I tried to patch is on branch `{branch}`.\n\n"
        f"My hypothesis for how a human engineer can fix it:\n{hypothesis}"
    )
  else:
    body = (
        f"Pipeline Halted: I attempted to auto-fix {model_name} 25 times but was unsuccessful.\n\n"
        f"Please check Airflow logs and GCS reports for run_id `{run_id}`."
    )

  try:
    subprocess.run(["python3", email_script, "--subject", subject, "--body", body, "--recipient", recipient], check=True)
    logger.info("Distress Signal email dispatched to %s", recipient)
  except Exception as e:
    logger.error("Failed to send distress signal email: %s", e)


def dispatch_victory_lap_alert(
    run_id, model_name, pr_url="", log_url="", report_url="", recipient=""
):
  """Triggers send_email.py for The Remediation Report (Victory Lap) upon pipeline success."""
  email_script = _get_email_script_path()
  recipient = recipient or _get_default_recipient()
  if not recipient:
    logger.warning("No email recipient configured (ALERT_RECIPIENT/USER_EMAIL empty). Skipping victory lap email.")
    return

  subject = f"Pipeline Successful: {model_name} is validated and ready! (Run: {run_id})"
  body = (
      f"Pipeline Successful: The {model_name} model is validated and ready!\n\n"
      "Note: I encountered an error during execution, but I successfully auto-patched the MaxText codebase to fix it.\n\n"
      f"Quick Links:\n"
      f" Review my pull request here: {pr_url or 'N/A'}\n"
      f" View Successful Airflow Log: {log_url or 'N/A'}\n"
      f" View detailed report(s): {report_url or 'N/A'}\n\n"
      "NOTE: The final report follows a structured summary format."
  )

  try:
    subprocess.run(["python3", email_script, "--subject", subject, "--body", body, "--recipient", recipient], check=True)
    logger.info("Victory Lap email dispatched to %s", recipient)
  except Exception as e:
    logger.error("Failed to send victory lap email: %s", e)
