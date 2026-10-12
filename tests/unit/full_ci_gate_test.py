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

"""Unit tests for the heavy TPU/GPU test gate script (.github/scripts/full_ci_gate.sh)."""

import os
import subprocess
import tempfile
import unittest

import pytest

_SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".github", "scripts", "full_ci_gate.sh")
)

_MISSING_LABEL_MESSAGE = (
    "Heavy tests skipped. Please apply the 'ready_for_full_ci' label to run all tests before merging."
)
_DRAFT_MESSAGE = (
    "Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' and apply the "
    "'ready_for_full_ci' label to run all tests before merging."
)
_DRAFT_LABELED_MESSAGE = (
    "Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' to run all tests before merging."
)

# Flags as analyze_code_changes.sh emits them when a change needs every test suite.
_ALL_SUITES = {
    "RUN_TESTS": "true",
    "RUN_NOTEBOOKS": "false",
    "RUN_PRETRAIN_TESTS": "true",
    "RUN_POSTTRAIN_TESTS": "true",
    "RUN_PATHWAYS_TESTS": "true",
    "RUN_GPU_TESTS": "true",
}
# Flags for a change that needs no test suite, e.g. docs-only or the `skip-ci-tests` label.
_NO_SUITES = dict.fromkeys(_ALL_SUITES, "false")
# The suites that have TPU/GPU flavors; any one of them (with RUN_TESTS) needs the heavy runners.
_SUITE_FLAGS = ("RUN_PRETRAIN_TESTS", "RUN_POSTTRAIN_TESTS", "RUN_PATHWAYS_TESTS", "RUN_GPU_TESTS")


def _run_gate(
    event_name="pull_request", is_draft="false", full_ci_label="false", scheduled_only_label="false", flags=None
):
  """Runs the gate script for one event.

  Args:
    event_name: EVENT_NAME, i.e. github.event_name.
    is_draft: IS_DRAFT, "true" for a draft PR.
    full_ci_label: HAS_FULL_CI_LABEL, "true" if the PR has the `ready_for_full_ci` label.
    scheduled_only_label: HAS_SCHEDULED_ONLY_LABEL, "true" if the PR has the `scheduled-only` label.
    flags: The analyze_code_changes.sh flags (RUN_TESTS, RUN_NOTEBOOKS, ...); None means _ALL_SUITES.

  Returns:
    (result, outputs): the subprocess.CompletedProcess and the key/value pairs the
    script wrote to GITHUB_OUTPUT.
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    output_path = os.path.join(tmp_dir, "github_output.txt")
    env = {
        **os.environ,
        **(_ALL_SUITES if flags is None else flags),
        "EVENT_NAME": event_name,
        "IS_DRAFT": is_draft,
        "HAS_FULL_CI_LABEL": full_ci_label,
        "HAS_SCHEDULED_ONLY_LABEL": scheduled_only_label,
        # Always override so a CI job's real GITHUB_OUTPUT file is never written to.
        "GITHUB_OUTPUT": output_path,
    }
    result = subprocess.run(["bash", _SCRIPT_PATH], env=env, capture_output=True, text=True, check=False)
    outputs = {}
    if os.path.exists(output_path):
      with open(output_path, encoding="utf-8") as f:
        outputs = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
  return result, outputs


def _expected(run_heavy_tests, gated, message=""):
  """The outputs the gate script should write."""
  return {
      "run_heavy_tests": "true" if run_heavy_tests else "false",
      "heavy_tests_gated": "true" if gated else "false",
      "heavy_tests_gate_message": message,
  }


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not os.path.isfile(_SCRIPT_PATH),
    reason="full CI gate script is not shipped in the MaxText image; run these tests from a source checkout",
)
class FullCiGateTest(unittest.TestCase):
  """Tests for .github/scripts/full_ci_gate.sh.

  The script lives under .github/, which is not copied into the Docker image, so the
  class is skipped when the tests run from inside the image (maxtext_installed=true).
  """

  def _gate(self, **kwargs):
    """Runs the gate and returns its outputs, failing the test if the script itself failed."""
    result, outputs = _run_gate(**kwargs)
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return outputs

  def test_non_pr_event_runs_heavy_tests(self):
    """Scheduled, dispatched and release (via workflow_call) runs are never gated."""
    for event_name in ("schedule", "workflow_dispatch", "release"):
      with self.subTest(event_name=event_name):
        self.assertEqual(self._gate(event_name=event_name), _expected(run_heavy_tests=True, gated=False))

  def test_ready_pr_with_full_ci_label_runs_heavy_tests(self):
    self.assertEqual(self._gate(full_ci_label="true"), _expected(run_heavy_tests=True, gated=False))

  def test_ready_pr_with_scheduled_only_label_runs_heavy_tests(self):
    """`scheduled-only` already asks for the full pipeline, so it counts as opting in."""
    self.assertEqual(self._gate(scheduled_only_label="true"), _expected(run_heavy_tests=True, gated=False))

  def test_ready_pr_without_label_is_gated(self):
    self.assertEqual(self._gate(), _expected(run_heavy_tests=False, gated=True, message=_MISSING_LABEL_MESSAGE))

  def test_pr_without_label_that_needs_no_suites_is_not_gated(self):
    """Docs-only or `skip-ci-tests` PRs pass the gatekeeper without the label."""
    self.assertEqual(self._gate(flags=_NO_SUITES), _expected(run_heavy_tests=False, gated=False))

  def test_notebook_only_pr_without_label_is_gated(self):
    """Notebooks run on TPU runners, so they count as heavy."""
    outputs = self._gate(flags={**_NO_SUITES, "RUN_NOTEBOOKS": "true"})
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_MISSING_LABEL_MESSAGE))

  def test_suite_flags_without_run_tests_are_not_gated(self):
    """Test suites only run when run_tests is true (gate_test_run), so nothing heavy is needed."""
    outputs = self._gate(flags={**_ALL_SUITES, "RUN_TESTS": "false"})
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=False))

  def test_draft_pr_with_full_ci_label_is_gated(self):
    """The PR already opted in, so the message only asks to mark it ready for review."""
    outputs = self._gate(is_draft="true", full_ci_label="true")
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_DRAFT_LABELED_MESSAGE))

  def test_draft_pr_without_label_is_gated_with_draft_message(self):
    self.assertEqual(self._gate(is_draft="true"), _expected(run_heavy_tests=False, gated=True, message=_DRAFT_MESSAGE))

  def test_draft_pr_with_scheduled_only_label_is_gated(self):
    outputs = self._gate(is_draft="true", scheduled_only_label="true")
    self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_DRAFT_LABELED_MESSAGE))

  def test_each_heavy_suite_alone_is_gated(self):
    """analyze_code_changes.sh often selects one suite, e.g. for GPU-only or post-training-only changes."""
    for suite in _SUITE_FLAGS:
      with self.subTest(suite=suite):
        outputs = self._gate(flags={**_NO_SUITES, "RUN_TESTS": "true", suite: "true"})
        self.assertEqual(outputs, _expected(run_heavy_tests=False, gated=True, message=_MISSING_LABEL_MESSAGE))

  def test_inputs_are_case_insensitive(self):
    """Values such as "True" are honored; in each case that one input decides the outcome."""
    runs = _expected(run_heavy_tests=True, gated=False)
    gated = _expected(run_heavy_tests=False, gated=True, message=_MISSING_LABEL_MESSAGE)
    draft_gated = _expected(run_heavy_tests=False, gated=True, message=_DRAFT_LABELED_MESSAGE)
    cases = [
        ({"is_draft": "TRUE", "full_ci_label": "true"}, draft_gated),
        ({"full_ci_label": "True"}, runs),
        ({"scheduled_only_label": "True"}, runs),
        ({"flags": {**_NO_SUITES, "RUN_NOTEBOOKS": "True"}}, gated),
        ({"flags": {**_NO_SUITES, "RUN_TESTS": "True", "RUN_GPU_TESTS": "true"}}, gated),
    ]
    cases += [({"flags": {**_NO_SUITES, "RUN_TESTS": "true", suite: "True"}}, gated) for suite in _SUITE_FLAGS]
    for kwargs, expected in cases:
      with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
        self.assertEqual(self._gate(**kwargs), expected)


if __name__ == "__main__":
  unittest.main()
