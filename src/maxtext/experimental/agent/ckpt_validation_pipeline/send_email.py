# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-;
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility script for the Overwatch agent to dispatch automated email alerts."""

# pylint: disable=logging-fstring-interpolation

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage

from maxtext.utils import max_logging as logger


def send_alert(subject: str, body: str, recipient: str):
  """Dispatches an email alert, gracefully degrading to local logging if SMTP is unavailable."""
  # In a production environment, you would pull these from a secure secrets manager or env.
  smtp_server = os.environ.get("SMTP_SERVER", "localhost")
  smtp_port = int(os.environ.get("SMTP_PORT", 1025))  # Default to a local mock server port
  sender_email = os.environ.get("SENDER_EMAIL", "overwatch-agent@ml-auto-solutions.com")

  msg = EmailMessage()
  msg.set_content(body)
  msg["Subject"] = subject
  msg["From"] = sender_email
  msg["To"] = recipient

  try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
      # server.login(user, password) # Add login here if authentication is required by the SMTP server
      server.send_message(msg)
    logger.info(f"Successfully sent email alert to {recipient} regarding: {subject}")
  except ConnectionRefusedError:
    logger.info(f"Connection to SMTP server {smtp_server}:{smtp_port} refused.")
    logger.info("Operating in MOCK/DEV mode. The following email WOULD have been sent:")
    logger.info("--- EMAIL START ---")
    logger.info(f"To: {recipient}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Body:\n{body}")
    logger.info("--- EMAIL END ---")
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(f"Failed to dispatch email. Exception: {e}")
    sys.exit(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Send automated pipeline alerts via email.")
  parser.add_argument("--subject", type=str, required=True, help="The subject line of the email.")
  parser.add_argument("--body", type=str, required=True, help="The main content/body of the email.")
  parser.add_argument("--recipient", type=str, required=True, help="The destination email address.")

  args = parser.parse_args()

  send_alert(args.subject, args.body, args.recipient)
