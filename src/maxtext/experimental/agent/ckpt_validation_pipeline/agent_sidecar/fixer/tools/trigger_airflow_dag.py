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

"""Tool for the Overwatch Agent to remotely trigger Airflow DAG runs."""

import argparse
import requests
import os

# Note: In a production environment, the Airflow Webserver URL and auth token would be injected via environment variables.
AIRFLOW_URL = os.environ.get("AIRFLOW_WEBSERVER_URL", "http://localhost:8080")
DAG_ID = "maxtext_validation_master_dag"


def trigger_dag(branch_name):
  """Triggers the master Airflow DAG, passing the specified branch name into the configuration."""
  url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"

  # We override the maxtext_branch in the dag_run.conf so the agent can test its specific fix
  payload = {"conf": {"maxtext_branch": branch_name}}

  headers = {
      "Content-Type": "application/json",
      "Accept": "application/json",
      # In production, add authentication headers here:
      # "Authorization": f"Bearer {os.environ.get('AIRFLOW_API_TOKEN')}"
  }

  try:
    print(f"Triggering Airflow DAG '{DAG_ID}' on branch '{branch_name}'...")
    response = requests.post(url, json=payload, headers=headers, timeout=10)

    if response.status_code in (200, 201):
      run_info = response.json()
      print(f"✅ Successfully triggered DAG Run ID: {run_info.get('dag_run_id')}")
      print(f"Status: {run_info.get('state')}")
    else:
      print(f"❌ Failed to trigger DAG. Status Code: {response.status_code}")
      print(f"Response: {response.text}")
  except requests.exceptions.RequestException as e:
    print(f"❌ Error communicating with Airflow API: {e}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Trigger the MaxText Validation Airflow DAG on a specific branch.")
  parser.add_argument("--branch", type=str, required=True, help="The git branch name containing the bug fix to test.")
  args = parser.parse_args()

  trigger_dag(args.branch)
