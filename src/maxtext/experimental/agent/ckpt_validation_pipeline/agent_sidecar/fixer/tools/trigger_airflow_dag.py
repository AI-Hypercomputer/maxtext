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
import json
import requests
import os
import sys

import google.auth
import google.auth.transport.requests

# Note: In a production environment, the Airflow Webserver URL and auth token would be injected via environment variables.
AIRFLOW_URL = os.environ.get(
    "AIRFLOW_WEBSERVER_URL",
    "https://4bae0a6de8f94e92aa8ee3a6ffc8b278-dot-us-central1.composer.googleusercontent.com",
)
DAG_ID = "maxtext_validation_master_dag"


def trigger_dag(branch_name, cluster_name=None, project_name=None, zone=None, overrides=None, dag_id=None):
  """Triggers a DAG and returns structured run metadata."""
  target_dag = dag_id or os.environ.get("TARGET_DAG_ID", DAG_ID)
  url = f"{AIRFLOW_URL}/api/v1/dags/{target_dag}/dagRuns"
  conf_dict = {}
  original_conf_str = os.environ.get("ORIGINAL_DAG_CONF")
  if original_conf_str:
    try:
      original_conf = json.loads(original_conf_str)
      # The failure log wraps the original clean Airflow config inside the 'dag_conf' key.
      # We extract it to avoid sending K8s manifests, error messages, and appended run_names
      # back into Airflow and creating infinitely nested configs.
      clean_conf = original_conf.get("dag_conf", original_conf)
      
      # Just in case we are dealing with an already nested config from before this fix,
      # gracefully un-nest it.
      while "dag_conf" in clean_conf:
        clean_conf = clean_conf["dag_conf"]
        
      conf_dict.update(clean_conf)
    except Exception as e:
      print(f"Warning: Failed to parse ORIGINAL_DAG_CONF: {e}")

  conf_dict["maxtext_branch"] = branch_name
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
        parsed = json.loads(overrides)
        if isinstance(parsed, dict):
          conf_dict.update(parsed)
      except json.JSONDecodeError:
        for item in overrides.split(","):
          if "=" in item:
            key, value = item.split("=", 1)
            conf_dict[key.strip()] = value.strip()

  headers = {"Content-Type": "application/json", "Accept": "application/json"}
  credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
  credentials.refresh(google.auth.transport.requests.Request())
  headers["Authorization"] = f"Bearer {credentials.token}"
  response = requests.post(url, json={"conf": conf_dict}, headers=headers, timeout=30)
  if response.status_code not in (200, 201):
    raise RuntimeError(f"Airflow trigger failed ({response.status_code}): {response.text}")
  result = response.json()
  output = {
      "ok": True,
      "dag_id": target_dag,
      "dag_run_id": result.get("dag_run_id"),
      "state": result.get("state"),
      "conf": conf_dict,
  }
  print(json.dumps(output))
  return output


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Trigger the MaxText Validation Airflow DAG on a specific branch.")
  parser.add_argument("--branch", type=str, required=True, help="The git branch name containing the bug fix to test.")
  parser.add_argument(
      "--cluster_name",
      type=str,
      default=None,
      help="Optional override for TPU GKE cluster name (e.g. v5p-128-bodaborg-europe-west4-b).",
  )
  parser.add_argument(
      "--project_name", type=str, default=None, help="Optional override for GCP project (e.g. cloud-tpu-multipod-dev)."
  )
  parser.add_argument("--zone", type=str, default=None, help="Optional override for GCP zone (e.g. europe-west4-b).")
  parser.add_argument(
      "--overrides", type=str, default=None, help="Optional parameter overrides in conf (JSON string or key=val list)."
  )
  parser.add_argument(
      "--dag_id",
      type=str,
      default=None,
      help="Specific Airflow DAG ID to re-trigger (e.g. dag_verify_forward_pass, dag_verify_decoding).",
  )
  args = parser.parse_args()

  try:
    trigger_dag(args.branch, args.cluster_name, args.project_name, args.zone, args.overrides, args.dag_id)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(json.dumps({"ok": False, "error": str(exc)}))
    sys.exit(1)
