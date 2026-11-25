# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for `.github/scripts/rerun_ci_for_pr.sh`."""

import json
import os
import pathlib
import subprocess
import sys
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RERUN_SCRIPT = REPO_ROOT / ".github" / "scripts" / "rerun_ci_for_pr.sh"
_VALID_SHA = "a" * 40
_MERGE_SHA = "c" * 40
_RUNS_ENDPOINT = (
    f"GET repos/AI-Hypercomputer/maxtext/actions/workflows/ci_pipeline.yml/runs"
    f"?head_sha={_VALID_SHA}&event=pull_request&per_page=100"
)
_ANALYZE_AT_MERGE = (
    f"GET repos/AI-Hypercomputer/maxtext/contents/.github/scripts/analyze_code_changes.sh?ref={_MERGE_SHA}"
)

_FAKE_GH = """\
import json, os, subprocess, sys
args = sys.argv[1:]
method, jq_expr, endpoint, i = "GET", None, None, 1
while i < len(args):
  if args[i] in ("-X", "--method"):
    method, i = args[i + 1], i + 2
  elif args[i] == "--jq":
    jq_expr, i = args[i + 1], i + 2
  elif args[i] in ("-H", "--header"):
    i += 2
  elif args[i].startswith("-"):
    i += 1
  else:
    endpoint, i = args[i], i + 1
with open(os.environ["FAKE_GH_CALLS"], "a", encoding="utf-8") as cf:
  cf.write(json.dumps([method, endpoint]) + "\\n")
with open(os.environ["FAKE_GH_STATE"], "r", encoding="utf-8") as sf:
  state = json.load(sf)
queue = state[f"{method} {endpoint}"]
item = queue.pop(0) if len(queue) > 1 else queue[0]
with open(os.environ["FAKE_GH_STATE"], "w", encoding="utf-8") as sf:
  json.dump(state, sf)
if item.get("rc", 0) != 0:
  sys.stderr.write(item.get("stderr", "HTTP 500\\n"))
  sys.exit(item["rc"])
body = item.get("body", {})
payload = body if isinstance(body, str) else json.dumps(body)
if jq_expr:
  sys.stdout.write(subprocess.run(["jq", "-r", jq_expr], input=payload, text=True, capture_output=True, check=True).stdout)
elif isinstance(body, str):
  sys.stdout.write(body)
"""


def _run_script(tmp_path, responses):
  """Execute `rerun_ci_for_pr.sh` with a stateful stub `gh` CLI."""
  bin_dir = tmp_path / "bin"
  bin_dir.mkdir(exist_ok=True)
  driver = tmp_path / "fake_gh.py"
  driver.write_text(_FAKE_GH, encoding="utf-8")
  state_path, calls_path = tmp_path / "state.json", tmp_path / "calls.jsonl"
  merged = {"GET repos/AI-Hypercomputer/maxtext/pulls/42": [{"body": {"state": "open", "head": {"sha": _VALID_SHA}}}]}
  merged.update(responses)
  state_path.write_text(json.dumps(merged), encoding="utf-8")
  calls_path.write_text("", encoding="utf-8")
  gh_bin = bin_dir / "gh"
  gh_bin.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{driver}' \"$@\"\n", encoding="utf-8")
  gh_bin.chmod(0o755)

  env = {
      **os.environ,
      "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
      "FAKE_GH_STATE": str(state_path),
      "FAKE_GH_CALLS": str(calls_path),
      "GITHUB_REPOSITORY": "AI-Hypercomputer/maxtext",
      "HEAD_SHA": _VALID_SHA,
      "PR_NUMBER": "42",
      "POLL_INTERVAL_SECONDS": "0",
      "FIND_RUN_ATTEMPTS": "2",
      "CANCEL_WAIT_ATTEMPTS": "4",
      "FORCE_CANCEL_AFTER_ATTEMPT": "2",
      "RERUN_ATTEMPTS": "2",
  }
  result = subprocess.run(
      ["bash", str(RERUN_SCRIPT)], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
  )
  calls = [tuple(json.loads(line)) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
  return result, calls


def test_reruns_newest_completed_run(tmp_path):
  runs = [
      {"id": 100, "status": "completed", "conclusion": "failure"},
      {
          "id": 250,
          "status": "completed",
          "conclusion": "failure",
          "referenced_workflows": [{"sha": _MERGE_SHA}],
      },
  ]
  result, calls = _run_script(
      tmp_path,
      {
          _RUNS_ENDPOINT: [{"body": {"workflow_runs": runs}}],
          _ANALYZE_AT_MERGE: [{"body": "#!/bin/bash\nfetch_pr_metadata_via_api\n"}],
          "POST repos/AI-Hypercomputer/maxtext/actions/runs/250/rerun": [{}],
      },
  )
  assert result.returncode == 0 and calls[-1] == ("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/250/rerun")


def test_cancels_active_run_and_escalates_to_force_cancel(tmp_path):
  result, calls = _run_script(
      tmp_path,
      {
          _RUNS_ENDPOINT: [{"body": {"workflow_runs": [{"id": 300, "status": "in_progress", "conclusion": None}]}}],
          "POST repos/AI-Hypercomputer/maxtext/actions/runs/300/cancel": [{}],
          "POST repos/AI-Hypercomputer/maxtext/actions/runs/300/force-cancel": [{"rc": 1, "stderr": "HTTP 502\n"}, {}],
          "GET repos/AI-Hypercomputer/maxtext/actions/runs/300": [
              {"rc": 1, "stderr": "HTTP 502\n"},
              {"body": {"status": "in_progress", "conclusion": None}},
              {"body": {"status": "completed", "conclusion": "cancelled"}},
          ],
          "POST repos/AI-Hypercomputer/maxtext/actions/runs/300/rerun": [{}],
      },
  )
  assert result.returncode == 0, result.stdout + result.stderr
  assert ("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/300/cancel") in calls
  assert calls.count(("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/300/force-cancel")) == 2
  assert calls[-1] == ("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/300/rerun")


def test_fails_fast_when_run_predates_live_label_support(tmp_path):
  legacy_run = [{"id": 275, "status": "in_progress", "conclusion": None, "referenced_workflows": [{"sha": _MERGE_SHA}]}]
  result, calls = _run_script(
      tmp_path,
      {
          _RUNS_ENDPOINT: [{"body": {"workflow_runs": legacy_run}}],
          _ANALYZE_AT_MERGE: [{"body": "#!/bin/bash\n# pre-6433 analyze_code_changes.sh\n"}],
      },
  )
  assert result.returncode == 1 and "predates live PR label re-runs" in result.stdout
  assert not [c for c in calls if c[0] == "POST"]


@pytest.mark.parametrize(
    "run_obj",
    [
        pytest.param({"id": 400, "status": "completed", "conclusion": "action_required"}, id="conclusion"),
        pytest.param({"id": 401, "status": "action_required", "conclusion": None}, id="status"),
    ],
)
def test_skips_action_required_fork_run(tmp_path, run_obj):
  result, calls = _run_script(tmp_path, {_RUNS_ENDPOINT: [{"body": {"workflow_runs": [run_obj]}}]})
  assert result.returncode == 0 and not [c for c in calls if c[0] == "POST"]


@pytest.mark.parametrize(
    "pr_states, expect_cancel",
    [
        pytest.param([{"body": {"state": "closed", "head": {"sha": _VALID_SHA}}}], False, id="closed-pr"),
        pytest.param(
            [
                {"body": {"state": "open", "head": {"sha": _VALID_SHA}}},
                {"body": {"state": "open", "head": {"sha": "b" * 40}}},
            ],
            True,
            id="head-moved-during-cancel",
        ),
    ],
)
def test_skips_when_pr_closed_or_head_sha_advanced(tmp_path, pr_states, expect_cancel):
  result, calls = _run_script(
      tmp_path,
      {
          "GET repos/AI-Hypercomputer/maxtext/pulls/42": pr_states,
          _RUNS_ENDPOINT: [{"body": {"workflow_runs": [{"id": 350, "status": "in_progress", "conclusion": None}]}}],
          "POST repos/AI-Hypercomputer/maxtext/actions/runs/350/cancel": [{}],
          "GET repos/AI-Hypercomputer/maxtext/actions/runs/350": [
              {"body": {"status": "completed", "conclusion": "cancelled"}}
          ],
      },
  )
  assert result.returncode == 0
  assert ("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/350/rerun") not in calls
  assert (("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/350/cancel") in calls) == expect_cancel


def test_missing_runs_api_errors_and_rerun_error_handling(tmp_path):
  empty_dir, api_err_dir, retry_dir, expired_dir = (tmp_path / n for n in ("empty", "api_err", "retry", "expired"))
  for d in (empty_dir, api_err_dir, retry_dir, expired_dir):
    d.mkdir()

  res_empty, _ = _run_script(empty_dir, {_RUNS_ENDPOINT: [{"body": {"workflow_runs": []}}]})
  assert res_empty.returncode == 0

  res_api_err, _ = _run_script(api_err_dir, {_RUNS_ENDPOINT: [{"rc": 1}]})
  assert res_api_err.returncode == 1

  completed = [{"body": {"workflow_runs": [{"id": 500, "status": "completed", "conclusion": "failure"}]}}]
  res_retry, calls_retry = _run_script(
      retry_dir,
      {_RUNS_ENDPOINT: completed, "POST repos/AI-Hypercomputer/maxtext/actions/runs/500/rerun": [{"rc": 1}, {}]},
  )
  assert (
      res_retry.returncode == 0
      and calls_retry.count(("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/500/rerun")) == 2
  )

  res_exp, calls_exp = _run_script(
      expired_dir,
      {
          _RUNS_ENDPOINT: completed,
          "POST repos/AI-Hypercomputer/maxtext/actions/runs/500/rerun": [
              {
                  "rc": 1,
                  "stderr": "Unable to retry this workflow run because it was created over a month ago (HTTP 403)\n",
              }
          ],
      },
  )
  assert res_exp.returncode == 1 and "Rebase PR #42 onto main" in res_exp.stdout
  assert calls_exp.count(("POST", "repos/AI-Hypercomputer/maxtext/actions/runs/500/rerun")) == 1
