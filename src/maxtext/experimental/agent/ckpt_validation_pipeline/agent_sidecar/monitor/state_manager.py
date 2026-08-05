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
from pathlib import Path

DATA_DIR = os.environ.get("ANTIGRAVITY_EXECUTABLE_DATA_DIR", "./data")
STATE_FILE = Path(DATA_DIR) / "retry_state.json"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "25"))


def load_state():
  if STATE_FILE.exists():
    with open(STATE_FILE, "r", encoding="utf-8") as f:
      return json.load(f)
  return {}


def save_state(state):
  Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
  temporary = STATE_FILE.with_suffix(".tmp")
  with open(temporary, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2, sort_keys=True)
  temporary.replace(STATE_FILE)


def get_run_state(run_key: str) -> dict:
  entry = load_state().get(run_key, {})
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
  state = load_state()
  entry = get_run_state(run_key)
  entry["retries"] += 1
  attempt = {
      "attempt": entry["retries"],
      "created_at": datetime.now(timezone.utc).isoformat(),
      **{key: value for key, value in details.items() if value not in (None, "")},
  }
  entry["attempts"].append(attempt)
  entry["status"] = details.get("status", "attempt_started")
  state[run_key] = entry
  save_state(state)
  return entry


def update_run_state(run_key: str, **updates) -> dict:
  state = load_state()
  entry = get_run_state(run_key)
  entry.update(updates)
  entry["updated_at"] = datetime.now(timezone.utc).isoformat()
  state[run_key] = entry
  save_state(state)
  return entry
