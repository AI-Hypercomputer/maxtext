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

"""Persistent remediation-attempt state for Overwatch."""

import json
import os
from datetime import datetime, timezone
from google.cloud import storage

GCS_BUCKET_NAME = os.environ.get("GCS_REPORTS_BUCKET", "maxtext-validation-agent-reports")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))


def _get_blob(run_key: str):
  client = storage.Client()
  bucket = client.bucket(GCS_BUCKET_NAME)
  return bucket.blob(f"retry_state_{run_key}.json")


def load_state(run_key: str):
  try:
    blob = _get_blob(run_key)
    if blob.exists():
      content = blob.download_as_string()
      return json.loads(content)
  except Exception as e:
    print(f"Failed to load state from GCS for {run_key}: {e}")
  return {}


def save_state(run_key: str, state: dict):
  try:
    blob = _get_blob(run_key)
    blob.upload_from_string(json.dumps(state, indent=2, sort_keys=True), content_type="application/json")
  except Exception as e:
    print(f"Failed to save state to GCS for {run_key}: {e}")


def get_run_state(run_key: str) -> dict:
  entry = load_state(run_key)
  if isinstance(entry, int):
    entry = {"retries": entry}
  return {
      "retries": 0,
      "attempts": [],
      "status": "new",
      **entry,
  }


def can_attempt(run_key: str) -> bool:
  return get_run_state(run_key).get("retries", 0) < MAX_RETRIES


def record_attempt(run_key: str, **details) -> dict:
  entry = get_run_state(run_key)
  entry["retries"] += 1
  attempt = {
      "attempt": entry["retries"],
      "created_at": datetime.now(timezone.utc).isoformat(),
      **{key: value for key, value in details.items() if value not in (None, "")},
  }
  entry["attempts"].append(attempt)
  entry["status"] = details.get("status", "attempt_started")
  save_state(run_key, entry)
  return entry


def update_run_state(run_key: str, **updates) -> dict:
  entry = get_run_state(run_key)
  entry.update(updates)
  entry["updated_at"] = datetime.now(timezone.utc).isoformat()
  save_state(run_key, entry)
  return entry
