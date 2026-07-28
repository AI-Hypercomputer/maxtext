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

"""Overwatch Sidecar: Monitors Airflow pipelines and spawns autonomous agents on failure."""

import time
import subprocess
from pathlib import Path
import logging

from monitor.state_manager import load_state, save_state, MAX_RETRIES
from monitor.alerter import dispatch_email_alert
from monitor.gcs_poller import check_for_failures

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def spawn_agent(run_id, model_name, failure_log, **kwargs):
  """Uses the agentapi CLI to spawn a new contextualized agent conversation using the prompt chain."""
  prompts_dir = Path(__file__).parent / "fixer" / "prompts"

  maxtext_branch = kwargs.get("maxtext_branch", "main")
  hf_ref_code_url = kwargs.get("hf_ref_code_url", "")
  hf_config_url = kwargs.get("hf_config_url", "")

  try:
    with open(prompts_dir / "01_diagnose.txt", "r", encoding="utf-8") as f:
      p1 = f.read()
    with open(prompts_dir / "02_patch.txt", "r", encoding="utf-8") as f:
      p2 = f.read()
    with open(prompts_dir / "03_verify.txt", "r", encoding="utf-8") as f:
      p3 = f.read()
    with open(prompts_dir / "meta_agent.txt", "r", encoding="utf-8") as f:
      meta_prompt_template = f.read()

    prompt = meta_prompt_template.format(
        model_name=model_name,
        run_id=run_id,
        failure_log=failure_log,
        maxtext_branch=maxtext_branch,
        hf_ref_code_url=hf_ref_code_url,
        hf_config_url=hf_config_url,
        p1=p1,
        p2=p2,
        p3=p3,
    )
  except FileNotFoundError as e:
    logger.error("Prompt template missing: %s", e)
    return
  logger.info("Spawning agent for Run ID %s...", run_id)
  try:
    subprocess.run(["agentapi", "new-conversation", "--title", f"Fix {model_name} Pipeline", prompt], check=True)
    logger.info("Agent spawned successfully.")
  except subprocess.CalledProcessError as e:
    logger.error("Failed to spawn agent: %s", e)


def run_loop():
  """Main polling loop for the sidecar."""
  logger.info("Overwatch Sidecar started. Monitoring pipeline...")
  while True:
    try:
      failure = check_for_failures()
      if failure:
        run_id = failure.get("run_id")
        model_name = failure.get("model_name")
        maxtext_branch = failure.get("maxtext_branch", "main")
        hf_ref_code_url = failure.get("hf_ref_code_url", "")
        hf_config_url = failure.get("hf_config_url", "")

        state = load_state()
        retries = state.get(run_id, 0)

        if retries >= MAX_RETRIES:
          logger.info("Run ID %s has hit the maximum of %s retries. Escalating to human.", run_id, MAX_RETRIES)
          dispatch_email_alert(run_id, model_name)
          state[run_id] = retries + 1  # Mark as handled
          save_state(state)
        elif retries < MAX_RETRIES:
          logger.info("Detected failure for %s. Attempt %s/%s.", run_id, retries + 1, MAX_RETRIES)
          spawn_agent(
              run_id,
              model_name,
              failure.get("log", ""),
              maxtext_branch=maxtext_branch,
              hf_ref_code_url=hf_ref_code_url,
              hf_config_url=hf_config_url,
          )
          state[run_id] = retries + 1
          save_state(state)

      time.sleep(10)
    except (ValueError, IOError, KeyError, TypeError, OSError) as e:
      logger.error("Error in polling loop: %s", e)
      time.sleep(10)


if __name__ == "__main__":
  run_loop()
