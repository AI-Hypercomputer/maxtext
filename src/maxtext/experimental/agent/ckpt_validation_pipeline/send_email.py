# Copyright 2024 Google LLC
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

"""Utility script for the Overwatch agent to dispatch automated email alerts."""

# pylint: disable=logging-fstring-interpolation

import argparse
import base64
import json
import os
import sys

from maxtext.utils import max_logging as logger
from google.cloud import pubsub_v1

def send_alert(subject: str, body: str, recipient: str, attachment_path: str = None):
  """Dispatches an email alert by publishing it to an Application Integration Pub/Sub topic."""
  
  project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "tpu-prod-env-multipod")
  topic_id = "maxtext-validation-agent-alerts"

  payload = {
      "subject": subject,
      "body": body,
      "recipient": recipient,
  }

  if attachment_path and os.path.exists(attachment_path):
    with open(attachment_path, "r", encoding="utf-8") as f:
      payload["attachment_content"] = f.read()
      payload["attachment_filename"] = os.path.basename(attachment_path)

  data = json.dumps(payload).encode("utf-8")

  try:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    
    future = publisher.publish(topic_path, data)
    message_id = future.result(timeout=10)
    logger.info(f"Published alert to Pub/Sub topic {topic_path}. Message ID: {message_id}")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"Failed to push alert to Pub/Sub topic {topic_id}. Is the Integrations API reachable?")
    logger.error(f"Exception: {e}")
    sys.exit(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Send automated pipeline alerts via email.")
  parser.add_argument("--subject", type=str, required=True, help="The subject line of the email.")
  parser.add_argument("--body", type=str, required=True, help="The main content/body of the email.")
  parser.add_argument("--recipient", type=str, required=True, help="The destination email address.")
  parser.add_argument("--attachment", type=str, required=False, help="Path to a file to attach.")

  args = parser.parse_args()

  send_alert(args.subject, args.body, args.recipient, args.attachment)
