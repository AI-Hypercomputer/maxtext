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

import google.auth
import google.auth.transport.requests

# Note: In a production environment, the Airflow Webserver URL and auth token would be injected via environment variables.
AIRFLOW_URL = os.environ.get(
    "AIRFLOW_WEBSERVER_URL",
    "https://4bae0a6de8f94e92aa8ee3a6ffc8b278-dot-us-central1.composer.googleusercontent.com",
)
DAG_ID = "maxtext_validation_master_dag"


def trigger_dag(branch_name, cluster_name=None, project_name=None, zone=None, overrides=None):
  """Triggers the master Airflow DAG, passing the specified branch and optional parameter overrides in conf."""
  url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"

  conf_dict = {"maxtext_branch": branch_name}
  if cluster_name:
    conf_dict["xpk_cluster_name"] = cluster_name
  if project_name:
    conf_dict["xpk_project"] = project_name
  if zone:
    conf_dict["xpk_zone"] = zone

  if overrides:
    if isinstance(overrides, dict):
      conf_dict.update(overrides)
    elif isinstance(overrides, str):
      try:
        import json
        parsed = json.loads(overrides)
        if isinstance(parsed, dict):
          conf_dict.update(parsed)
      except Exception:
        for kv in overrides.split(","):
          if "=" in kv:
            k, v = kv.split("=", 1)
            conf_dict[k.strip()] = v.strip()

  payload = {"conf": conf_dict}

  headers = {
      "Content-Type": "application/json",
      "Accept": "application/json",
  }
  try:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    if credentials.token:
      headers["Authorization"] = f"Bearer {credentials.token}"
  except Exception as auth_e:
    print(f"Warning: Failed to refresh Google auth credentials: {auth_e}")

  try:
    print(f"Triggering Airflow DAG '{DAG_ID}' on branch '{branch_name}' (conf: {conf_dict})...")
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
  parser.add_argument("--cluster_name", type=str, default=None, help="Optional override for TPU GKE cluster name (e.g. v5p-128-bodaborg-europe-west4-b).")
  parser.add_argument("--project_name", type=str, default=None, help="Optional override for GCP project (e.g. cloud-tpu-multipod-dev).")
  parser.add_argument("--zone", type=str, default=None, help="Optional override for GCP zone (e.g. europe-west4-b).")
  parser.add_argument("--overrides", type=str, default=None, help="Optional parameter overrides in conf (JSON string or key=val list).")
  args = parser.parse_args()

  trigger_dag(args.branch, args.cluster_name, args.project_name, args.zone, args.overrides)

