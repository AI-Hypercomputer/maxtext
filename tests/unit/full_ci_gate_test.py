# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the heavy TPU/GPU test gate (the gate step of ci_pipeline.yml) and the workflows around it."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pytest
import yaml

_ANALYZE_SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".github", "scripts", "analyze_code_changes.sh")
)
_WORKFLOW_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".github", "workflows", "ci_pipeline.yml")
)

# The gate's messages, word for word.
_SKIP_LABEL_MESSAGE = (
    "Heavy tests skipped because this PR has the 'skip-ci-tests' label. A bot removes it after 2 approvals once the "
    "other checks pass, which runs all tests; remove it yourself to run them sooner."
)
_NEW_PR_MESSAGE = (
    "Heavy tests skipped on the first run of a new PR, which gets the 'skip-ci-tests' label. A bot removes it after 2 "
    "approvals once the other checks pass, which runs all tests; remove it yourself to run them sooner."
)
_DRAFT_MESSAGE = (
    "Heavy tests skipped because this PR is a draft. Mark it 'Ready for review'; a bot then removes the "
    "'skip-ci-tests' label after 2 approvals once the other checks pass, which runs all tests."
)
_DRAFT_READY_MESSAGE = (
    "Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' to run all tests before merging."
)

# The pull_request actions that start ci_pipeline.yml.
_PR_ACTIONS = ("opened", "synchronize", "reopened", "labeled", "unlabeled", "ready_for_review")

# Flags as analyze_code_changes.sh emits them when a change needs every test suite.
_ALL_SUITES = {
    "RUN_TESTS": "true",
    "RUN_NOTEBOOKS": "false",
    "RUN_PRETRAIN_TESTS": "true",
    "RUN_POSTTRAIN_TESTS": "true",
    "RUN_PATHWAYS_TESTS": "true",
    "RUN_GPU_TESTS": "true",
}
# Flags for a change that needs no test suite, e.g. docs-only.
_NO_SUITES = dict.fromkeys(_ALL_SUITES, "false")
# The suites that have TPU/GPU flavors; any one of them (with RUN_TESTS) needs the heavy runners.
_SUITE_FLAGS = ("RUN_PRETRAIN_TESTS", "RUN_POSTTRAIN_TESTS", "RUN_PATHWAYS_TESTS", "RUN_GPU_TESTS")


def _run_gate(
    event_action="synchronize",
    is_draft="false",
    skip_ci_tests_label="false",
    scheduled_only_label="false",
    flags=None,
    copybara="false",
):
  """Runs the gate step of ci_pipeline.yml for one pull_request event (other events skip the step).

  Args:
    event_action: EVENT_ACTION, i.e. github.event.action; None leaves it unset.
    is_draft: IS_DRAFT, "true" for a draft PR.
    skip_ci_tests_label: HAS_SKIP_CI_TESTS_LABEL, "true" if the PR has the `skip-ci-tests` label.
    scheduled_only_label: HAS_SCHEDULED_ONLY_LABEL, "true" if the PR has the `scheduled-only` label.
    flags: The analyze_code_changes.sh flags (RUN_TESTS, RUN_NOTEBOOKS, ...); None means _ALL_SUITES.
    copybara: IS_COPYBARA_PR, "true" if Copybara opened the PR (copybara-service[bot]).

  Returns:
    (result, outputs): the subprocess.CompletedProcess and the key/value pairs the
    step wrote to GITHUB_OUTPUT.
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    output_path = os.path.join(tmp_dir, "github_output.txt")
    base_env = {k: v for k, v in os.environ.items() if k != "EVENT_ACTION"}
    env = {
        **base_env,
        **(_ALL_SUITES if flags is None else flags),
        "IS_DRAFT": is_draft,
        "HAS_SKIP_CI_TESTS_LABEL": skip_ci_tests_label,
        "HAS_SCHEDULED_ONLY_LABEL": scheduled_only_label,
        "IS_COPYBARA_PR": copybara,
        # Always override so a CI job's real GITHUB_OUTPUT file is never written to.
        "GITHUB_OUTPUT": output_path,
    }
    if event_action is not None:
      env["EVENT_ACTION"] = event_action
    # Runs in an empty directory: the step must not need any file from the checkout.
    result = _run_step(_gate_step_script(), env, cwd=tmp_dir)
    outputs = _read_outputs(output_path)
  return result, outputs


def _run_analyze_code_changes(extra_env):
  """Runs analyze_code_changes.sh with `extra_env`, without PR_NUMBER or a git repository to list changed files.

  Returns:
    (result, outputs), like _run_gate.
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    output_path = os.path.join(tmp_dir, "github_output.txt")
    unset = ("EVENT_NAME", "GITHUB_EVENT_NAME", "FORCE_ALL_TESTS", "SKIP_CI_TESTS", "PR_NUMBER", "GITHUB_REPOSITORY")
    env = {
        **{k: v for k, v in os.environ.items() if k not in unset},
        **extra_env,
        # Every git command fails, so a PR run finds no changed files.
        "GIT_DIR": os.path.join(tmp_dir, "no-repo"),
        "GITHUB_OUTPUT": output_path,
    }
    result = subprocess.run(
        ["bash", _ANALYZE_SCRIPT_PATH], cwd=tmp_dir, env=env, capture_output=True, text=True, check=False
    )
    outputs = _read_outputs(output_path)
  return result, outputs


def _read_outputs(output_path):
  """The key/value pairs written to a GITHUB_OUTPUT file; empty if the file was not created."""
  if not os.path.exists(output_path):
    return {}
  with open(output_path, encoding="utf-8") as f:
    return dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)


def _step_script(job_id, step):
  """The `run:` script of the step of a ci_pipeline.yml job whose `id` or `name` is `step`."""
  with open(_WORKFLOW_PATH, encoding="utf-8") as f:
    workflow = yaml.safe_load(f)
  return next(s["run"] for s in workflow["jobs"][job_id]["steps"] if step in (s.get("id"), s.get("name")))


def _gate_step_script():
  """The `run:` script of the gate step (id: gate) of the analyze_code_changes job in ci_pipeline.yml."""
  return _step_script("analyze_code_changes", "gate")


def _run_step(script, env, cwd=None):
  """Runs a workflow `run:` script the way GitHub runs it when the step sets no `shell:` (`bash -e`)."""
  return subprocess.run(["bash", "-e", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, check=False)


def _expected(run_heavy_tests, gated, message=""):
  """The outputs the gate step should write."""
  return {
      "run_heavy_tests": "true" if run_heavy_tests else "false",
      "heavy_tests_gated": "true" if gated else "false",
      "heavy_tests_gate_message": message,
  }


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not (os.path.isfile(_WORKFLOW_PATH) and os.path.isfile(_ANALYZE_SCRIPT_PATH)),
    reason="ci_pipeline.yml and analyze_code_changes.sh are not shipped in the MaxText image; run these tests from a "
    "source checkout",
)
class FullCiGateTest(unittest.TestCase):
  """Tests for the gate step of ci_pipeline.yml (id: gate), which decides what the `skip-ci-tests` label skips.

  The workflows and scripts live under .github/, which is not copied into the Docker image, so the
  class is skipped when the tests run from inside the image (maxtext_installed=true).
  """

  def _gate(self, **kwargs):
    """Runs the gate and returns its outputs, failing the test if the step itself failed."""
    result, outputs = _run_gate(**kwargs)
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return outputs

  def test_pr_without_skip_ci_tests_label_runs_heavy_tests(self):
    """E.g. once the author or the bot removed the label, or on a PR opened before the labeler existed."""
    for action in [action for action in _PR_ACTIONS if action != "opened"]:
      with self.subTest(action=action):
        self.assertEqual(self._gate(event_action=action), _expected(run_heavy_tests=True, gated=False))

  def test_pr_with_skip_ci_tests_label_is_gated(self):
    for action in _PR_ACTIONS:
      with self.subTest(action=action):
        outputs = self._gate(event_action=action, skip_ci_tests_label="true")
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_SKIP_LABEL_MESSAGE))

  def test_new_pr_is_gated_on_its_first_run(self):
    """add_skip_ci_tests_label.yml adds the label in parallel, so the `opened` payload doesn't have it yet."""
    outputs = self._gate(event_action="opened")
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_NEW_PR_MESSAGE))

  def test_copybara_pr_is_not_gated_on_its_first_run(self):
    """The labeler skips PRs that Copybara exports, so they run every suite as before.

    Copybara merges them itself without approvals here, so the bot would never remove the label.
    """
    outputs = self._gate(event_action="opened", copybara="true")
    self.assertEqual(outputs, _expected(run_heavy_tests=True, gated=False))

  def test_copybara_pr_with_skip_ci_tests_label_is_gated(self):
    """Only the labeler skips Copybara PRs: someone who adds the label by hand still gets the label's effect."""
    for action in _PR_ACTIONS:
      with self.subTest(action=action):
        outputs = self._gate(event_action=action, skip_ci_tests_label="true", copybara="true")
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_SKIP_LABEL_MESSAGE))

  def test_copybara_draft_pr_is_gated_with_draft_ready_message(self):
    """A Copybara draft doesn't get the label, so marking it ready is enough to run all tests."""
    for action in ("opened", "synchronize"):
      with self.subTest(action=action):
        outputs = self._gate(event_action=action, is_draft="true", copybara="true")
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_DRAFT_READY_MESSAGE))

  def test_scheduled_only_label_runs_heavy_tests(self):
    """`scheduled-only` asks for the full scheduled pipeline, so it wins over `skip-ci-tests` and the new-PR rule."""
    for action in _PR_ACTIONS:
      for skip_ci_tests_label in ("false", "true"):
        with self.subTest(action=action, skip_ci_tests_label=skip_ci_tests_label):
          outputs = self._gate(event_action=action, skip_ci_tests_label=skip_ci_tests_label, scheduled_only_label="true")
          self.assertEqual(outputs, _expected(run_heavy_tests=True, gated=False))

  def test_pr_that_needs_no_heavy_suites_is_not_gated(self):
    """Docs-only changes are not gated, so All Required Tests Passed passes whatever skipped the heavy suites."""
    cases = [
        ({"skip_ci_tests_label": "true"}, False),
        ({"event_action": "opened"}, False),
        ({"is_draft": "true"}, False),
        ({"is_draft": "true", "skip_ci_tests_label": "true"}, False),
        ({}, True),
    ]
    for kwargs, run_heavy_tests in cases:
      with self.subTest(**kwargs):
        outputs = self._gate(flags=_NO_SUITES, **kwargs)
        self.assertEqual(outputs, _expected(run_heavy_tests=run_heavy_tests, gated=False))

  def test_notebook_only_pr_is_gated(self):
    """Notebooks run on TPU runners, so they count as heavy."""
    outputs = self._gate(skip_ci_tests_label="true", flags={**_NO_SUITES, "RUN_NOTEBOOKS": "true"})
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_SKIP_LABEL_MESSAGE))

  def test_suite_flags_without_run_tests_are_not_gated(self):
    """Test suites only run when run_tests is true (gate_test_run), so nothing heavy is needed."""
    outputs = self._gate(skip_ci_tests_label="true", flags={**_ALL_SUITES, "RUN_TESTS": "false"})
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=False))

  def test_draft_pr_is_gated_with_draft_message(self):
    """The draft has the `skip-ci-tests` label, or is getting it as a new PR, so being ready is not enough."""
    for kwargs in (
        {"skip_ci_tests_label": "true"},
        {"skip_ci_tests_label": "true", "event_action": "opened"},
        {"event_action": "opened"},
    ):
      with self.subTest(**kwargs):
        outputs = self._gate(is_draft="true", **kwargs)
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_DRAFT_MESSAGE))

  def test_draft_pr_is_gated_with_draft_ready_message(self):
    """Marking the draft ready runs all tests: it has no `skip-ci-tests` label, or `scheduled-only` wins over it."""
    for kwargs in (
        {},
        {"event_action": "unlabeled"},
        {"scheduled_only_label": "true"},
        {"scheduled_only_label": "true", "skip_ci_tests_label": "true"},
        {"scheduled_only_label": "true", "event_action": "opened"},
    ):
      with self.subTest(**kwargs):
        outputs = self._gate(is_draft="true", **kwargs)
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_DRAFT_READY_MESSAGE))

  def test_each_heavy_suite_alone_is_gated(self):
    """analyze_code_changes.sh often selects one suite, e.g. for GPU-only or post-training-only changes."""
    for suite in _SUITE_FLAGS:
      with self.subTest(suite=suite):
        outputs = self._gate(skip_ci_tests_label="true", flags={**_NO_SUITES, "RUN_TESTS": "true", suite: "true"})
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_SKIP_LABEL_MESSAGE))

  def test_only_true_counts_as_true(self):
    """Boolean inputs such as "1" or "yes" count as false."""
    for value in ("1", "yes"):
      with self.subTest(value=value):
        self.assertEqual(self._gate(skip_ci_tests_label=value), _expected(run_heavy_tests=True, gated=False))

  def test_event_action_is_compared_exactly(self):
    """Only GitHub's exact value matches: "Opened" or an unset action is not `opened`."""
    runs = _expected(run_heavy_tests=True, gated=False)
    for event_action in ("Opened", None):
      with self.subTest(event_action=event_action):
        self.assertEqual(self._gate(event_action=event_action), runs)

  def test_analyze_code_changes_ignores_skip_ci_tests(self):
    """`skip-ci-tests` skips only the heavy suites, through this gate, so SKIP_CI_TESTS must not change any flag."""
    scenarios = {
        "non-PR event": {"EVENT_NAME": "schedule"},
        "scheduled-only PR": {"EVENT_NAME": "pull_request", "FORCE_ALL_TESTS": "true"},
        "PR without changed files": {"EVENT_NAME": "pull_request"},
    }
    for name, extra_env in scenarios.items():
      with self.subTest(name):
        _, expected = _run_analyze_code_changes(extra_env)
        self.assertEqual(expected.get("run_tests"), "true", "the scenario must enable tests to be meaningful")
        result, outputs = _run_analyze_code_changes({**extra_env, "SKIP_CI_TESTS": "true"})
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertEqual(outputs, expected)

  def test_analyze_code_changes_still_honors_force_all_tests(self):
    """The `scheduled-only` label (FORCE_ALL_TESTS) runs every suite on a PR, notebooks included."""
    result, outputs = _run_analyze_code_changes({"EVENT_NAME": "pull_request", "FORCE_ALL_TESTS": "true"})
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    self.assertEqual(outputs, {flag.lower(): "true" for flag in _ALL_SUITES})


_REPO = "AI-Hypercomputer/maxtext"
_HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"
_RUNS_ENDPOINT = f"repos/{_REPO}/actions/workflows/ci_pipeline.yml/runs"
# A run triggered by an unrelated label skips analyze_code_changes (so its gate outputs are empty), Code Quality and
# Docs.
_UNRELATED_LABEL_RUN = {
    "NEEDS_ANALYZE_CODE_CHANGES_RESULT": "skipped",
    "NEEDS_CODE_QUALITY_CHECK_RESULT": "skipped",
    "NEEDS_DOCS_BUILD_CHECK_RESULT": "skipped",
    "RUN_HEAVY_TESTS": "",
    "HEAVY_TESTS_GATED": "",
    "GATE_MESSAGE": "",
}

# A fake `gh` for the workflow step tests. It answers `gh api URL --jq FILTER` with the canned
# response for the URL's path in $FAKE_GH_ROUTES, filtered by the real jq, and logs every URL.
# A list of responses is served one per request, and its last response repeats.
# A response {"fake_gh_error": MESSAGE} makes the call fail.
_FAKE_GH = """
import json
import os
import subprocess
import sys

url = sys.argv[sys.argv.index("api") + 1]
jq_filter = sys.argv[sys.argv.index("--jq") + 1]
path = url.split("?", 1)[0]
with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as log:
  log.write(url + "\\n")
with open(os.environ["FAKE_GH_ROUTES"], encoding="utf-8") as f:
  response = json.load(f)[path]
if isinstance(response, list):
  with open(os.environ["FAKE_GH_LOG"], encoding="utf-8") as log:
    earlier = sum(line.split("?", 1)[0] == path for line in log.read().splitlines()) - 1
  response = response[min(earlier, len(response) - 1)]
if "fake_gh_error" in response:
  sys.exit(response["fake_gh_error"])
sys.exit(subprocess.run(["jq", "-r", jq_filter], input=json.dumps(response), text=True, check=False).returncode)
"""


class _FakeGh:
  """A temporary PATH whose `gh` serves canned API responses (see _FAKE_GH) and whose `sleep` returns at once."""

  def __init__(self, routes):
    self._routes = routes
    self._dir = None

  def __enter__(self):
    self._dir = tempfile.mkdtemp()
    self._write("gh", f"#!{sys.executable}\n{_FAKE_GH}", executable=True)
    self._write("sleep", '#!/bin/sh\necho "$@" >> "$FAKE_SLEEP_LOG"\n', executable=True)
    self._write("routes.json", json.dumps(self._routes))
    self._write("gh.log", "")
    self._write("sleep.log", "")
    return self

  def __exit__(self, *exc_info):
    shutil.rmtree(self._dir)

  def _write(self, name, content, executable=False):
    path = os.path.join(self._dir, name)
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)
    if executable:
      os.chmod(path, 0o755)

  def _lines(self, name):
    with open(os.path.join(self._dir, name), encoding="utf-8") as f:
      return f.read().splitlines()

  def env(self, extra):
    """os.environ plus `extra`, with the fakes first on PATH and the step summary in the temporary directory."""
    return {
        **os.environ,
        # A CI job's real step summary is never written to.
        "GITHUB_STEP_SUMMARY": os.path.join(self._dir, "summary.md"),
        **extra,
        "PATH": self._dir + os.pathsep + os.environ.get("PATH", ""),
        "FAKE_GH_ROUTES": os.path.join(self._dir, "routes.json"),
        "FAKE_GH_LOG": os.path.join(self._dir, "gh.log"),
        "FAKE_SLEEP_LOG": os.path.join(self._dir, "sleep.log"),
    }

  def requests(self):
    """The URLs requested with `gh api`, in order."""
    return self._lines("gh.log")

  def sleeps(self):
    """How many times `sleep` was called."""
    return len(self._lines("sleep.log"))

  def summary(self):
    """What the step wrote to its step summary."""
    path = os.path.join(self._dir, "summary.md")
    if not os.path.exists(path):
      return ""
    with open(path, encoding="utf-8") as f:
      return f.read()


def _jobs(full_run=True, atp="success", atp_status="completed"):
  """A response of the jobs endpoint for one run of ci_pipeline.yml.

  Args:
    full_run: whether the run ran the pipeline (its Code Quality check ran). Runs triggered by an
      unrelated label skip it, and analyze_code_changes too.
    atp: the conclusion of the run's "All Required Tests Passed" job once it completed.
    atp_status: the status of that job; None if the job does not exist yet (tests still running).
  """
  if full_run:
    code_quality = {"name": "Code Quality Check / Pre-commit Linters", "status": "completed", "conclusion": "success"}
  else:
    code_quality = {"name": "Code Quality Check", "status": "completed", "conclusion": "skipped"}
  analyze = "success" if full_run else "skipped"
  jobs = [{"name": "Analyze Code Changes for Test Orchestration", "status": "completed", "conclusion": analyze}]
  jobs.append(code_quality)
  if atp_status is not None:
    conclusion = atp if atp_status == "completed" else None
    jobs.append({"name": "All Required Tests Passed", "status": atp_status, "conclusion": conclusion})
  return {"jobs": jobs}


def _pipeline_runs(*runs):
  """Canned API responses for the runs of ci_pipeline.yml on the PR head commit.

  Args:
    *runs: (run_id, jobs) pairs in the order the runs endpoint lists them; jobs is a _jobs() response,
      or a list of them served one per request.
  """
  routes = {_RUNS_ENDPOINT: {"workflow_runs": [{"id": run_id} for run_id, _ in runs]}}
  routes.update({f"repos/{_REPO}/actions/runs/{run_id}/jobs": jobs for run_id, jobs in runs})
  return routes


def _label_names(expression):
  """The label names that a workflow expression compares github.event.label.name with."""
  return set(re.findall(r"github\.event\.label\.name == '([^']+)'", expression))


_LABELER_WORKFLOW_PATH = os.path.join(os.path.dirname(_WORKFLOW_PATH), "add_skip_ci_tests_label.yml")
_AUTO_LABEL_WORKFLOW_PATH = os.path.join(os.path.dirname(_WORKFLOW_PATH), "auto_label_pr.yml")


def _load_workflow(path=_WORKFLOW_PATH):
  """A parsed workflow file. PyYAML reads the `on` key as True (YAML 1.1), so it is put back under "on"."""
  with open(path, encoding="utf-8") as f:
    workflow = yaml.safe_load(f)
  if True in workflow:
    workflow["on"] = workflow.pop(True)
  return workflow


def _step_env(job_id, step_id):
  """The `env:` of the step of a ci_pipeline.yml job whose `id` is `step_id`."""
  return next(s for s in _load_workflow()["jobs"][job_id]["steps"] if s.get("id") == step_id)["env"]


def _conjuncts(expression):
  """The operands of the top-level `&&`s of a workflow expression; the whole expression if it has a top-level `||`."""
  parts, depth, start = [], 0, 0
  for i, char in enumerate(expression):
    depth += {"(": 1, ")": -1}.get(char, 0)
    if depth == 0 and expression.startswith("||", i):
      return [" ".join(expression.split())]
    if depth == 0 and expression.startswith("&&", i):
      parts.append(expression[start:i])
      start = i + 2
  parts.append(expression[start:])
  return [" ".join(part.split()) for part in parts]


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not os.path.isfile(_WORKFLOW_PATH),
    reason="ci_pipeline.yml is not shipped in the MaxText image; run these tests from a source checkout",
)
class CiPipelineStepTest(unittest.TestCase):
  """Tests for the ci_pipeline.yml settings and `run:` scripts around the heavy-test gate, and the labeler workflow."""

  # The run that a test runs the step in, triggered by an unrelated label. The runs endpoint lists it too.
  _CURRENT_RUN = (3000, _jobs(full_run=False, atp_status="in_progress"))

  def _all_tests_passed(self, routes, **env):
    """Runs the step of the All Required Tests Passed job against fake API responses.

    Skips the test if `routes` is not empty and jq is missing: the fake gh filters its responses with jq.
    Stores what the step wrote to its step summary in self.step_summary.

    Returns:
      (result, requested URLs, number of sleeps).
    """
    if routes and shutil.which("jq") is None:
      self.skipTest("the fake gh filters API responses with jq, which is not installed")
    base_env = {
        "NEEDS_ANALYZE_CODE_CHANGES_RESULT": "success",
        "NEEDS_CODE_QUALITY_CHECK_RESULT": "success",
        "NEEDS_DOCS_BUILD_CHECK_RESULT": "success",
        "ANY_JOB_FAILED": "false",
        "ANY_JOB_CANCELLED": "false",
        "RUN_HEAVY_TESTS": "true",
        "HEAVY_TESTS_GATED": "false",
        "GATE_MESSAGE": "",
        "GITHUB_REPOSITORY": _REPO,
        "HEAD_SHA": _HEAD_SHA,
    }
    with _FakeGh(routes) as fake:
      result = _run_step(_step_script("all_tests_passed", "Check test results"), fake.env({**base_env, **env}))
      self.step_summary = fake.summary()
      return result, fake.requests(), fake.sleeps()

  def _all_notebooks_passed(self, **env):
    """Runs the step of the All Notebooks Passed job."""
    base_env = {
        "NEEDS_ANALYZE_CODE_CHANGES_RESULT": "success",
        "NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_NOTEBOOKS": "true",
        "NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_HEAVY_TESTS": "true",
        "ANY_JOB_FAILED": "false",
        "ANY_JOB_CANCELLED": "false",
    }
    return _run_step(_step_script("all_notebooks_passed", "Check notebooks results"), {**os.environ, **base_env, **env})

  def test_pull_request_trigger_types(self):
    """Removing `skip-ci-tests` (`unlabeled`) and marking a draft ready for review (`ready_for_review`) start a run."""
    types = _load_workflow()["on"]["pull_request"]["types"]
    self.assertCountEqual(types, _PR_ACTIONS)

  def test_concurrency_group_and_entry_jobs_name_the_same_ci_labels(self):
    """Adding or removing any other label starts a run in its own concurrency group, so it never cancels a test run.

    That run skips the entry jobs, so it runs no test; All Required Tests Passed repeats the newest full run.
    """
    workflow = _load_workflow()
    expressions = {
        "concurrency": workflow["concurrency"]["group"],
        **{
            job: workflow["jobs"][job].get("if", "")
            for job in ("analyze_code_changes", "code_quality_check", "docs_build_check")
        },
    }
    for name, expression in expressions.items():
      with self.subTest(name):
        self.assertEqual(_label_names(expression), {"scheduled-only", "skip-ci-tests"})
        self.assertIn(
            "(github.event.action != 'labeled' && github.event.action != 'unlabeled')", " ".join(expression.split())
        )

  def test_gate_step_passes_the_gate_inputs(self):
    """The gate step passes exactly the inputs that its script reads (plus GITHUB_OUTPUT, which it writes)."""
    expected = {
        "EVENT_ACTION": "${{ github.event.action }}",
        "IS_DRAFT": "${{ github.event.pull_request.draft }}",
        "HAS_SKIP_CI_TESTS_LABEL": "${{ contains(github.event.pull_request.labels.*.name, 'skip-ci-tests') }}",
        "HAS_SCHEDULED_ONLY_LABEL": "${{ contains(github.event.pull_request.labels.*.name, 'scheduled-only') }}",
        # Must match the PRs that add_skip_ci_tests_label.yml skips.
        "IS_COPYBARA_PR": "${{ github.event.pull_request.user.login == 'copybara-service[bot]' }}",
        **{flag: "${{ steps.check.outputs." + flag.lower() + " }}" for flag in _ALL_SUITES},
    }
    self.assertEqual(_step_env("analyze_code_changes", "gate"), expected)
    read = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", _gate_step_script()))
    self.assertEqual(read, set(expected) | {"GITHUB_OUTPUT"})

  def test_gate_step_runs_only_on_pull_requests(self):
    """Scheduled, dispatched and release (workflow_call) runs are never gated, so they skip the gate step."""
    step = next(s for s in _load_workflow()["jobs"]["analyze_code_changes"]["steps"] if s.get("id") == "gate")
    self.assertEqual(step.get("if"), "github.event_name == 'pull_request'")

  def test_gate_outputs_run_the_heavy_tests_outside_pull_requests(self):
    """Without the gate step the job outputs run every heavy suite, but only outside pull requests.

    A PR whose gate step failed gets no outputs, so its heavy suites stay skipped and the merge stays blocked (a
    default like `steps.gate.outputs.run_heavy_tests || 'true'` would run them ungated).
    """
    outputs = _load_workflow()["jobs"]["analyze_code_changes"]["outputs"]
    self.assertEqual(
        outputs["run_heavy_tests"],
        "${{ github.event_name != 'pull_request' && 'true' || steps.gate.outputs.run_heavy_tests }}",
    )
    self.assertEqual(
        outputs["heavy_tests_gated"],
        "${{ github.event_name != 'pull_request' && 'false' || steps.gate.outputs.heavy_tests_gated }}",
    )
    # Empty when the step is skipped, which is what non-PR events need.
    self.assertEqual(outputs["heavy_tests_gate_message"], "${{ steps.gate.outputs.heavy_tests_gate_message }}")

  def test_skip_ci_tests_label_does_not_reach_analyze_code_changes(self):
    """`skip-ci-tests` skips only the heavy suites (through the gate step), so the CPU tests still run."""
    check_env = _step_env("analyze_code_changes", "check")
    self.assertNotIn("SKIP_CI_TESTS", check_env)
    self.assertIn("FORCE_ALL_TESTS", check_env)

  def test_heavy_jobs_run_only_when_the_gate_allows_it(self):
    """Each TPU/GPU job requires `run_heavy_tests` from the gate step, whatever else its `if:` checks."""
    jobs = _load_workflow()["jobs"]
    heavy_jobs = (
        "maxtext_jupyter_notebooks",
        "tpu-pretrain-tests",
        "tpu-posttrain-tests",
        "tpu7x-tests",
        "gpu-tests",
        "maxtext_tpu_pathways_unit_tests",
        "maxtext_tpu_pathways_integration_tests",
    )
    for job in heavy_jobs:
      with self.subTest(job):
        self.assertIn("needs.analyze_code_changes.outputs.run_heavy_tests == 'true'", _conjuncts(jobs[job]["if"]))

  def test_all_tests_passed_checks_the_gate(self):
    """The required All Required Tests Passed check itself fails when the gate skipped heavy suites; no extra job."""
    jobs = _load_workflow()["jobs"]
    self.assertNotIn("full_ci_gatekeeper", jobs)
    job = jobs["all_tests_passed"]
    self.assertEqual(
        job["needs"],
        [
            "analyze_code_changes",
            "build_and_upload_maxtext_package",
            "gate_test_run",
            "tpu-pretrain-tests",
            "tpu-posttrain-tests",
            "tpu7x-tests",
            "gpu-tests",
            "cpu-pretrain-tests",
            "cpu-posttrain-tests",
            "maxtext_tpu_pathways_unit_tests",
            "maxtext_tpu_pathways_integration_tests",
            "code_quality_check",
            "docs_build_check",
        ],
    )
    # A skipped required check counts as passing, so the job must run even when a job it needs failed.
    self.assertEqual(job["if"], "always()")
    env = job["steps"][0]["env"]
    self.assertEqual(env["ANY_JOB_FAILED"], "${{ contains(needs.*.result, 'failure') }}")
    self.assertEqual(env["ANY_JOB_CANCELLED"], "${{ contains(needs.*.result, 'cancelled') }}")
    self.assertEqual(env["NEEDS_ANALYZE_CODE_CHANGES_RESULT"], "${{ needs.analyze_code_changes.result }}")
    self.assertEqual(env["RUN_HEAVY_TESTS"], "${{ needs.analyze_code_changes.outputs.run_heavy_tests }}")
    self.assertEqual(env["HEAVY_TESTS_GATED"], "${{ needs.analyze_code_changes.outputs.heavy_tests_gated }}")
    self.assertEqual(env["GATE_MESSAGE"], "${{ needs.analyze_code_changes.outputs.heavy_tests_gate_message }}")

  def test_auto_label_pr_reads_the_annotation_that_all_tests_passed_writes(self):
    """auto_label_pr.yml removes `skip-ci-tests` only when All Required Tests Passed failed with this annotation."""
    title = "Heavy tests skipped"
    self.assertIn(f"::error title={title}::", _step_script("all_tests_passed", "Check test results"))
    with open(_AUTO_LABEL_WORKFLOW_PATH, encoding="utf-8") as f:
      bot = f.read()
    self.assertIn(f"'{title}'", bot)
    self.assertNotIn("Full CI Gatekeeper", bot)

  def test_notify_failure_needs_only_the_summary_jobs(self):
    """notify_failure files an issue when a job it needs failed in a scheduled run.

    All Required Tests Passed and All Notebooks Passed fail whenever a job they need failed or was cancelled, and
    together they need every job except the reporting ones, so needing these two is enough.
    """
    jobs = _load_workflow()["jobs"]
    self.assertEqual(jobs["notify_failure"]["needs"], ["all_tests_passed", "all_notebooks_passed"])
    self.assertEqual(jobs["notify_failure"]["if"], "${{ !cancelled() }}")
    covered = set(jobs["all_tests_passed"]["needs"]) | set(jobs["all_notebooks_passed"]["needs"])
    reporting = {"all_tests_passed", "all_notebooks_passed", "notify_failure", "investigate_failure", "track_performance"}
    self.assertEqual(set(jobs) - reporting - covered, set())

  def test_investigate_failure_runs_once_notify_failure_filed_an_issue(self):
    """investigate_failure starts when notify_failure filed (or commented on) the issue of a failed scheduled run.

    notify_failure succeeds when it does, so a `contains(needs.*.result, 'failure')` check would never start the
    investigation. Its issue_number output is set only then.
    """
    jobs = _load_workflow()["jobs"]
    notify, investigate = jobs["notify_failure"], jobs["investigate_failure"]
    create_issue = next(s for s in notify["steps"] if s.get("id") == "create_issue")
    self.assertEqual(create_issue["if"], "${{ contains(needs.*.result, 'failure') && github.event_name == 'schedule' }}")
    self.assertEqual(notify["outputs"]["issue_number"], "${{ steps.create_issue.outputs.issue-number }}")
    self.assertEqual(investigate["needs"], ["notify_failure"])
    self.assertEqual(
        " ".join(investigate["if"].split()),
        "${{ !cancelled() && github.event_name == 'schedule' && needs.notify_failure.outputs.issue_number != '' }}",
    )
    self.assertEqual(investigate["with"]["tracking_issue_number"], "${{ needs.notify_failure.outputs.issue_number }}")

  def test_skip_ci_tests_labeler_workflow_is_minimal(self):
    """add_skip_ci_tests_label.yml uses pull_request_target, so it runs with a write token on fork PRs too.

    It must never check out or run PR code, use a third-party action, or put event values into its script.
    """
    self.assertTrue(os.path.isfile(_LABELER_WORKFLOW_PATH), f"{_LABELER_WORKFLOW_PATH} is missing")
    with open(_LABELER_WORKFLOW_PATH, encoding="utf-8") as f:
      text = f.read()
    self.assertTrue(text.startswith("# Copyright 2026 Google LLC\n"))
    self.assertIn("Licensed under the Apache License, Version 2.0", text)
    self.assertIn("# zizmor: ignore[dangerous-triggers]", text)
    workflow = _load_workflow(_LABELER_WORKFLOW_PATH)
    self.assertEqual(workflow["name"], "Add skip-ci-tests Label")
    self.assertEqual(workflow["on"], {"pull_request_target": {"types": ["opened"]}})
    self.assertEqual(workflow["permissions"], {})
    self.assertEqual(list(workflow["jobs"]), ["add_skip_ci_tests_label"])
    job = workflow["jobs"]["add_skip_ci_tests_label"]
    self.assertEqual(job["name"], "Add skip-ci-tests label")
    self.assertEqual(job["permissions"], {"pull-requests": "write"})
    self.assertEqual(job["timeout-minutes"], 5)
    self.assertEqual(job["runs-on"], "ubuntu-latest")
    # `scheduled-only` wins over `skip-ci-tests`. Copybara merges the PRs it exports without approvals here, so the
    # bot would never remove the label from them (the gate step's IS_COPYBARA_PR names the same author).
    self.assertEqual(
        " ".join(job["if"].split()),
        "${{ !contains(github.event.pull_request.labels.*.name, 'scheduled-only') && "
        "github.event.pull_request.user.login != 'copybara-service[bot]' }}",
    )
    self.assertEqual(len(job["steps"]), 1)
    step = job["steps"][0]
    self.assertNotIn("uses", step)
    script, env = step["run"], step.get("env", {})
    self.assertNotIn("${{", script, "event values must reach the script through env")
    self.assertEqual(env.get("GH_TOKEN"), "${{ github.token }}")
    pr_number_vars = [name for name, value in env.items() if value == "${{ github.event.pull_request.number }}"]
    self.assertEqual(len(pr_number_vars), 1, f"one env var must hold the PR number: {env}")
    self.assertRegex(script, r"\$\{?" + re.escape(pr_number_vars[0]) + r"\b")
    self.assertIn("gh api", script)
    self.assertIn("skip-ci-tests", script)

  def test_all_tests_passed_passes_when_no_job_failed(self):
    result, requests, _ = self._all_tests_passed({})
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(requests, [], "a run that ran the pipeline needs no API call")

  def test_all_tests_passed_fails_when_a_job_failed_or_was_cancelled(self):
    for env in ({"ANY_JOB_FAILED": "true"}, {"ANY_JOB_CANCELLED": "true"}):
      with self.subTest(**env):
        result, _, _ = self._all_tests_passed({}, **env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("One or more jobs failed or were cancelled", result.stdout)

  def test_all_tests_passed_fails_when_the_gate_skipped_needed_heavy_tests(self):
    """The annotation title `Heavy tests skipped` tells auto_label_pr.yml that the label blocked the heavy tests."""
    result, _, _ = self._all_tests_passed(
        {}, RUN_HEAVY_TESTS="false", HEAVY_TESTS_GATED="true", GATE_MESSAGE=_SKIP_LABEL_MESSAGE
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(f"::error title=Heavy tests skipped::{_SKIP_LABEL_MESSAGE}\n", result.stdout)
    self.assertEqual(self.step_summary, f"### {_SKIP_LABEL_MESSAGE}\n")

  def test_all_tests_passed_fails_closed_on_missing_or_invalid_gate_outputs(self):
    """A broken gate must not skip every heavy suite silently, so missing or invalid outputs fail the check too."""
    for gated, run_heavy_tests in (("", "true"), ("false", ""), ("false", "yes"), ("", "")):
      with self.subTest(heavy_tests_gated=gated, run_heavy_tests=run_heavy_tests):
        result, _, _ = self._all_tests_passed({}, RUN_HEAVY_TESTS=run_heavy_tests, HEAVY_TESTS_GATED=gated)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("::error title=Heavy tests skipped::The heavy-test gate produced no decision", result.stdout)

  def test_all_tests_passed_passes_when_no_heavy_test_was_needed(self):
    """E.g. a docs-only PR with `skip-ci-tests`: the gate skipped the heavy suites, but the change needs none."""
    result, _, _ = self._all_tests_passed({}, RUN_HEAVY_TESTS="false", HEAVY_TESTS_GATED="false")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_unrelated_label_run_passes_after_a_passing_full_run(self):
    """The run tested nothing, so it repeats the result of the newest run on the commit that ran the pipeline.

    analyze_code_changes was skipped too, so the gate outputs are empty: the newest full run decides before the gate
    check, which only runs in full runs.
    """
    routes = _pipeline_runs(self._CURRENT_RUN, (2000, _jobs()))
    result, requests, _ = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn(f"head_sha={_HEAD_SHA}", requests[0])
    self.assertNotIn("Heavy tests skipped", result.stdout)

  def test_unrelated_label_run_fails_after_a_failing_full_run(self):
    """Only the newest full run counts, even when an older one passed (here the runs are listed oldest first)."""
    routes = _pipeline_runs((1000, _jobs()), (2000, _jobs(atp="failure")), self._CURRENT_RUN)
    result, _, _ = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("is 'completed failure'", result.stdout)

  def test_unrelated_label_run_fails_without_a_full_run(self):
    routes = _pipeline_runs(self._CURRENT_RUN, (2500, _jobs(full_run=False)))
    result, _, _ = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("there is no full run on this commit", result.stdout)

  def test_unrelated_label_run_ignores_earlier_unrelated_label_runs(self):
    """An earlier run for an unrelated label tested nothing either, so the failed full run before it decides."""
    routes = _pipeline_runs(self._CURRENT_RUN, (2500, _jobs(full_run=False)), (2000, _jobs(atp="failure")))
    result, _, _ = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("is 'completed failure'", result.stdout)

  def test_unrelated_label_run_waits_for_a_finishing_full_run(self):
    """The full run's own All Required Tests Passed job is still running: wait for its result."""
    routes = _pipeline_runs(self._CURRENT_RUN, (2000, [_jobs(atp_status="in_progress"), _jobs()]))
    result, _, sleeps = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(sleeps, 1)

  def test_unrelated_label_run_does_not_wait_for_a_full_run_still_testing(self):
    """That run's All Required Tests Passed check does not exist yet; it will be newer than this one anyway."""
    routes = _pipeline_runs(self._CURRENT_RUN, (2000, _jobs(atp_status=None)))
    result, _, sleeps = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("is still running its tests", result.stdout)
    self.assertEqual(sleeps, 0)

  def test_unrelated_label_run_stops_waiting(self):
    routes = _pipeline_runs(self._CURRENT_RUN, (2000, _jobs(atp_status="in_progress")))
    result, _, sleeps = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("is 'in_progress none'", result.stdout)
    self.assertEqual(sleeps, 10)

  def test_unrelated_label_run_fails_without_head_sha(self):
    routes = _pipeline_runs(self._CURRENT_RUN, (2000, _jobs()))
    result, _, _ = self._all_tests_passed(routes, **_UNRELATED_LABEL_RUN, HEAD_SHA="")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("analyze_code_changes was skipped in a run that is not for a pull request", result.stdout)

  def test_unrelated_label_run_fails_when_the_api_fails(self):
    result, _, _ = self._all_tests_passed({_RUNS_ENDPOINT: {"fake_gh_error": "HTTP 500"}}, **_UNRELATED_LABEL_RUN)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("HTTP 500", result.stderr)

  def test_all_notebooks_passed_without_notebook_changes(self):
    result = self._all_notebooks_passed(
        NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_NOTEBOOKS="false",
        NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_HEAVY_TESTS="false",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_all_notebooks_passed_when_the_notebooks_ran(self):
    result = self._all_notebooks_passed()
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_all_notebooks_passed_fails_when_the_gate_skipped_the_notebooks(self):
    """It must not report "passed" for notebooks that the heavy-test gate skipped."""
    result = self._all_notebooks_passed(NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_HEAVY_TESTS="false")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("see 'All Required Tests Passed'", result.stdout)

  def test_all_notebooks_passed_in_an_unrelated_label_run(self):
    """Such a run skips analyze_code_changes and every notebook; All Required Tests Passed repeats the full run."""
    result = self._all_notebooks_passed(
        NEEDS_ANALYZE_CODE_CHANGES_RESULT="skipped",
        NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_NOTEBOOKS="",
        NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_HEAVY_TESTS="",
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_all_notebooks_passed_reads_the_results_it_checks(self):
    """The step's inputs come from the jobs it summarizes, and it runs even when one of them failed."""
    job = _load_workflow()["jobs"]["all_notebooks_passed"]
    self.assertEqual(
        job["needs"], ["analyze_code_changes", "build_and_upload_maxtext_package", "maxtext_jupyter_notebooks"]
    )
    self.assertEqual(job["if"], "always()")
    env = job["steps"][0]["env"]
    self.assertEqual(env["ANY_JOB_FAILED"], "${{ contains(needs.*.result, 'failure') }}")
    self.assertEqual(env["ANY_JOB_CANCELLED"], "${{ contains(needs.*.result, 'cancelled') }}")
    self.assertEqual(env["NEEDS_ANALYZE_CODE_CHANGES_RESULT"], "${{ needs.analyze_code_changes.result }}")
    outputs = "${{ needs.analyze_code_changes.outputs."
    self.assertEqual(env["NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_NOTEBOOKS"], outputs + "run_notebooks }}")
    self.assertEqual(env["NEEDS_ANALYZE_CODE_CHANGES_OUTPUTS_RUN_HEAVY_TESTS"], outputs + "run_heavy_tests }}")

  def test_all_notebooks_passed_fails_when_a_job_failed(self):
    result = self._all_notebooks_passed(ANY_JOB_FAILED="true")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("One or more jobs failed or were cancelled", result.stdout)


if __name__ == "__main__":
  unittest.main()
