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

"""Tests for the decisions of .github/workflows/auto_label_pr.yml.

The `Decide label updates` step's script runs under Node, as actions/github-script runs it, with stub `github`,
`context` and `core` objects that serve a pull request and its check runs and record the step outputs.
"""

import json
import os
import pathlib
import shutil
import subprocess
import textwrap

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "auto_label_pr.yml"

_HARNESS = textwrap.dedent(
    """\
    const scenario = JSON.parse(process.env.SCENARIO);
    const outputs = {};
    const core = {
      setOutput: (name, value) => { outputs[name] = String(value); },
      setFailed: (message) => { outputs.failed = message; },
      info: () => {},
    };
    const context = { eventName: scenario.eventName, payload: scenario.payload, repo: { owner: 'o', repo: 'r' } };
    const github = {
      graphql: async () => ({ repository: { pullRequest: scenario.pullRequest } }),
      paginate: async (method, params) => method(params),
      rest: {
        checks: { listForRef: async () => scenario.checkRuns },
        issues: {
          addLabels: async (params) => { outputs.added_labels = params.labels.join(','); },
          removeLabel: async (params) => { outputs.removed_label = params.name; },
        },
      },
    };
    (async () => {
    %s
    })().then(
      () => console.log(JSON.stringify(outputs)),
      (error) => { console.error(error); process.exit(1); },
    );
    """
)
_PASSED = [{"status": "completed", "conclusion": "success"}]
_FAILED = [{"status": "completed", "conclusion": "failure"}]


def _job():
  yaml = pytest.importorskip("yaml")
  return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["AddPullReady"]


def _decide_script():
  return next(step for step in _job()["steps"] if step.get("id") == "decide")["with"]["script"]


def _decide(tmp_path, *, labels, check_runs, event="pull_request_review", action="submitted", added_label=None):
  """Run the decide step for a 1-commit, twice-approved, non-draft PR; return the step outputs it set."""
  node = shutil.which("node")
  if node is None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
      pytest.fail("node is not installed on this runner")
    pytest.skip("node is not installed")
  harness = tmp_path / "decide.js"
  harness.write_text(_HARNESS % _decide_script(), encoding="utf-8")
  scenario = {
      "eventName": event,
      "payload": {"action": action, "pull_request": {"number": 7}, "label": {"name": added_label}},
      "pullRequest": {
          "reviewDecision": "APPROVED",
          "isDraft": False,
          "isCrossRepository": False,
          "headRefOid": "0123abc",
          "commits": {"totalCount": 1},
          "labels": {"nodes": [{"name": name} for name in labels]},
          "reviews": {"nodes": [{"state": "APPROVED", "author": {"login": login}} for login in ("ana", "bo")]},
      },
      "checkRuns": check_runs,
  }
  result = subprocess.run(
      [node, str(harness)],
      env={**os.environ, "SCENARIO": json.dumps(scenario)},
      capture_output=True,
      text=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr
  return json.loads(result.stdout)


@pytest.mark.parametrize(
    "labels, check_runs, removes_skip_ci_tests",
    [
        pytest.param(["skip-ci-tests"], [], True, id="never-ran"),
        pytest.param(["skip-ci-tests"], _FAILED, True, id="failed"),
        pytest.param(["skip-ci-tests"], _PASSED, False, id="already-passed"),
        pytest.param([], [], False, id="no-skip-ci-tests"),
        pytest.param(["run-tpu7x"], [], False, id="run-tpu7x-without-skip-ci-tests"),
        # The run that passed may predate the run-tpu7x label, which starts no TPU7X run while skip-ci-tests holds the
        # heavy suites back, so only removing skip-ci-tests runs the TPU7X tests the label asks for.
        pytest.param(["skip-ci-tests", "run-tpu7x"], _PASSED, True, id="run-tpu7x-after-a-passing-run"),
    ],
)
def test_two_approvals_add_pull_ready_and_start_full_ci_only_when_needed(
    tmp_path, labels, check_runs, removes_skip_ci_tests
):
  outputs = _decide(tmp_path, labels=labels, check_runs=check_runs)
  assert outputs.get("added_labels") == "pull ready"
  assert (outputs.get("remove_skip_ci_tests") == "true") == removes_skip_ci_tests, outputs


@pytest.mark.parametrize(
    "labels, removes_skip_ci_tests",
    [
        pytest.param(["pull ready", "skip-ci-tests"], False, id="already-passed"),
        pytest.param(["pull ready", "skip-ci-tests", "run-tpu7x"], True, id="run-tpu7x-after-a-passing-run"),
    ],
)
def test_pull_ready_added_by_hand_starts_full_ci_only_when_needed(tmp_path, labels, removes_skip_ci_tests):
  outputs = _decide(
      tmp_path, labels=labels, check_runs=_PASSED, event="pull_request_target", action="labeled", added_label="pull ready"
  )
  assert (outputs.get("remove_skip_ci_tests") == "true") == removes_skip_ci_tests, outputs


def test_the_job_runs_for_pull_ready_and_run_tpu7x_label_events_only():
  # labeled fires for every label; the job must react to the two labels that can call for a full CI run.
  assert " ".join(_job()["if"].split()) == (
      "github.event_name != 'pull_request_target' || github.event.action != 'labeled' || "
      "github.event.label.name == 'pull ready' || github.event.label.name == 'run-tpu7x'"
  )


@pytest.mark.parametrize(
    "labels, check_runs, removes_skip_ci_tests",
    [
        # After "pull ready", no later event re-runs the bot before Copybara imports on the earlier passing run, so the
        # label itself must start the full CI run with TPU7X.
        pytest.param(["pull ready", "skip-ci-tests", "run-tpu7x"], _PASSED, True, id="after-pull-ready"),
        pytest.param(["pull ready", "skip-ci-tests", "run-tpu7x"], [], True, id="after-pull-ready-never-ran"),
        pytest.param(["pull ready", "run-tpu7x"], _PASSED, False, id="after-pull-ready-without-skip-ci-tests"),
        # Before "pull ready", two approvals still decide when the heavy suites run: the label must not lift the gate.
        pytest.param(["skip-ci-tests", "run-tpu7x"], _PASSED, False, id="before-pull-ready"),
        pytest.param(["skip-ci-tests", "run-tpu7x"], [], False, id="before-pull-ready-never-ran"),
    ],
)
def test_run_tpu7x_added_by_hand_starts_full_ci_only_after_pull_ready(
    tmp_path, labels, check_runs, removes_skip_ci_tests
):
  outputs = _decide(
      tmp_path,
      labels=labels,
      check_runs=check_runs,
      event="pull_request_target",
      action="labeled",
      added_label="run-tpu7x",
  )
  assert (outputs.get("remove_skip_ci_tests") == "true") == removes_skip_ci_tests, outputs
