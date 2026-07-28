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

logger = logging.getLogger(__name__)


def dispatch_email_alert(run_id, model_name, recipient="oncall-team@google.com"):
  """Triggers the send_email.py utility when the retry limit is exhausted."""
  email_script = (
      "/Users/fiyinbenstowe/Desktop/project/maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/send_email.py"
  )
  subject = f"Pipeline Halted: {model_name} (Run: {run_id})"
  body = f"The Overwatch agent attempted to auto-fix {model_name} 3 times but was unsuccessful. Please investigate."

  try:
    subprocess.run(["python3", email_script, "--subject", subject, "--body", body, "--recipient", recipient], check=True)
    logger.info("Alert email dispatched to %s", recipient)
  except subprocess.CalledProcessError as e:
    logger.error("Failed to send alert email: %s", e)
