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

"""Poll an exact Airflow DAG run until it reaches a terminal state."""

import argparse
import json
import os
import sys
import time

import google.auth
import google.auth.transport.requests
import requests

AIRFLOW_URL = os.environ["AIRFLOW_WEBSERVER_URL"].rstrip("/")
TERMINAL_STATES = {"success", "failed"}


def wait_for_run(dag_id: str, dag_run_id: str, timeout_seconds: int, poll_seconds: int):
  credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  request = google.auth.transport.requests.Request()
  deadline = time.monotonic() + timeout_seconds
  url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
  while time.monotonic() < deadline:
    credentials.refresh(request)
    response = requests.get(
        url, headers={"Authorization": f"Bearer {credentials.token}", "Accept": "application/json"}, timeout=30
    )
    if response.status_code != 200:
      raise RuntimeError(f"Airflow status failed ({response.status_code}): {response.text}")
    payload = response.json()
    state = str(payload.get("state", "")).lower()
    if state in TERMINAL_STATES:
      result = {"ok": state == "success", "dag_id": dag_id, "dag_run_id": dag_run_id, "state": state}
      print(json.dumps(result))
      return result
    time.sleep(poll_seconds)
  raise TimeoutError(f"Timed out waiting for {dag_id}/{dag_run_id}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dag_id", required=True)
  parser.add_argument("--dag_run_id", required=True)
  parser.add_argument("--timeout_seconds", type=int, default=7200)
  parser.add_argument("--poll_seconds", type=int, default=30)
  args = parser.parse_args()
  try:
    wait_for_run(args.dag_id, args.dag_run_id, args.timeout_seconds, args.poll_seconds)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(json.dumps({"ok": False, "error": str(exc)}))
    sys.exit(1)
