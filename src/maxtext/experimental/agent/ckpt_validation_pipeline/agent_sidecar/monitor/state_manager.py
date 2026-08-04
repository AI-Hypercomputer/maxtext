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

"""State manager for tracking pipeline runs and retries."""

import json
import os
from pathlib import Path

# The system provides this environment variable for persistent state storage
DATA_DIR = os.environ.get("ANTIGRAVITY_EXECUTABLE_DATA_DIR", "./data")
STATE_FILE = Path(DATA_DIR) / "retry_state.json"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "25"))


def load_state():
  """Loads the retry tracking state from the persistent data directory."""
  if STATE_FILE.exists():
    with open(STATE_FILE, "r", encoding="utf-8") as f:
      return json.load(f)
  return {}


def save_state(state):
  """Saves the retry tracking state to the persistent data directory."""
  Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
  with open(STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)


def get_run_state(run_id: str) -> dict:
  """Returns a structured dictionary state for a given run_id."""
  state = load_state()
  entry = state.get(run_id, {"retries": 0, "attempts": []})
  if isinstance(entry, int):
    entry = {"retries": entry, "attempts": []}
  return entry


def record_attempt(run_id: str, branch: str = "", diagnosis: str = "", hypothesis: str = "") -> dict:
  """Records an attempt with rich context so terminal alerts include full analysis and history."""
  state = load_state()
  entry = get_run_state(run_id)
  entry["retries"] += 1
  if branch or diagnosis or hypothesis:
    entry["attempts"].append({
        "attempt": entry["retries"],
        "branch": branch,
        "diagnosis": diagnosis,
        "hypothesis": hypothesis,
    })
  state[run_id] = entry
  save_state(state)
  return entry
