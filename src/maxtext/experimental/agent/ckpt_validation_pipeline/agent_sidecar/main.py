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

def main():
    """Entrypoint for the Cloud Run Job. Executes once and terminates."""
    logger.info("Overwatch Cloud Run Job started. Checking for pipeline failures...")
    
    try:
        failure, blob_name = check_for_failures()
        if not failure:
            logger.info("No failures detected in GCS. Exiting cleanly.")
            return

        run_id = failure.get("run_id") or failure.get("task")  # Fallback to task if run_id missing
        model_name = failure.get("model_name", "unknown")
        failure_log = failure.get("log", "") or failure.get("error_message", "")

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
            
            # Trigger the ADK workflow instead of shelling out to agentapi CLI
            run_agent_workflow(run_id, model_name, failure_log)
            
            state[run_id] = retries + 1
            save_state(state)
            mark_handled(blob_name)

    except Exception as e:
        logger.error("Error during job execution: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
