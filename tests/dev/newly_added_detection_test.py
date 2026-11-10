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

"""Unit tests for the line-range test-change detector in ``newly_added_detection``."""

import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest

# The repository root: tests/dev/<this file>.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("newly_added_detection", REPO_ROOT / "tests/utils/newly_added_detection.py")
newly_added_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(newly_added_detection)

# pylint: disable=protected-access
_build_diff_commands = newly_added_detection._build_diff_commands
_is_eligible = newly_added_detection._is_eligible
_is_test_file = newly_added_detection._is_test_file
_marker_matches = newly_added_detection._marker_matches
_read_pytest_ini = newly_added_detection._read_pytest_ini
# pylint: enable=protected-access
TPU7X_FLAVORS = newly_added_detection.TPU7X_FLAVORS
is_changed_item = newly_added_detection.is_changed_item
parse_changed_line_map = newly_added_detection.parse_changed_line_map
touched_test_names = newly_added_detection.touched_test_names
touched_tests = newly_added_detection.touched_tests


# --- _is_test_file -----------------------------------------------------------


def test_is_test_file_accepts_test_suffixes_under_tests():
  assert _is_test_file("tests/unit/a_test.py")
  assert _is_test_file("tests/integration/b_tests.py")


def test_is_test_file_rejects_non_tests_and_non_suffix():
  assert not _is_test_file("tests/utils/test_helpers.py")  # helper, not *_test.py
  assert not _is_test_file("src/maxtext/foo_test.py")  # outside tests/
  assert not _is_test_file("tests/unit/helper.py")  # not a test suffix


# --- _build_diff_commands ----------------------------------------------------


def test_build_diff_commands_uses_only_merge_base_threedot_ranges():
  # Only three-dot (merge-base) ranges: remote first (CI / local-with-origin),
  # then local (developer without an origin remote). Two-dot tip-vs-tip ranges are
  # excluded because they over-report every commit the base gained past the fork
  # point, dragging unrelated scheduled_only tests into pre-submit.
  cmds = _build_diff_commands("main")
  assert cmds == [
      ["git", "diff", "--unified=0", "origin/main...HEAD"],
      ["git", "diff", "--unified=0", "main...HEAD"],
  ]
  # Guard against a two-dot form sneaking back in: every range arg is three-dot.
  range_args = [cmd[-1] for cmd in cmds]
  assert all("..." in arg for arg in range_args)


def test_build_diff_commands_honours_non_main_base():
  cmds = _build_diff_commands("release/v2")
  assert cmds == [
      ["git", "diff", "--unified=0", "origin/release/v2...HEAD"],
      ["git", "diff", "--unified=0", "release/v2...HEAD"],
  ]


# --- parse_changed_line_map --------------------------------------------------


def test_line_map_added_region_from_header_range():
  diff = (
      "diff --git a/tests/unit/a_test.py b/tests/unit/a_test.py\n"
      "--- a/tests/unit/a_test.py\n"
      "+++ b/tests/unit/a_test.py\n"
      "@@ -0,0 +1,3 @@\n"
      "+line1\n+line2\n+line3\n"
  )
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {1, 2, 3}}


def test_line_map_single_line_default_length():
  diff = "+++ b/tests/unit/a_test.py\n" "@@ -10 +12 @@\n" "+changed\n"
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {12}}


def test_line_map_pure_deletion_marks_boundary():
  # `+4,0` = pure deletion; the join sits between new-file lines 4 and 5.
  diff = "+++ b/tests/unit/a_test.py\n" "@@ -5,2 +4,0 @@\n" "-gone1\n-gone2\n"
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {4, 5}}


def test_line_map_accumulates_multiple_hunks_and_files():
  diff = (
      "+++ b/tests/unit/a_test.py\n"
      "@@ -0,0 +1,1 @@\n"
      "+x\n"
      "@@ -8,0 +10,2 @@\n"
      "+y\n+z\n"
      "+++ b/tests/unit/b_test.py\n"
      "@@ -0,0 +3,1 @@\n"
      "+w\n"
  )
  assert parse_changed_line_map(diff) == {
      "tests/unit/a_test.py": {1, 10, 11},
      "tests/unit/b_test.py": {3},
  }


def test_line_map_ignores_deleted_file():
  diff = "+++ /dev/null\n" "@@ -1,2 +0,0 @@\n" "-a\n-b\n"
  assert not parse_changed_line_map(diff)


# --- tests_touching_lines ----------------------------------------------------

_SOURCE = textwrap.dedent(
    """\
    import pytest


    class TestAlpha:

      @pytest.mark.scheduled_only
      def test_existing(self):
        x = 1
        assert x == 1

      def test_untouched(self):
        assert True


    def test_top_level_untouched():
      assert True


    async def test_async_new():
      assert True
    """
)
# Line numbers in _SOURCE:
#  6 @pytest.mark.scheduled_only
#  7 def test_existing        (span 6-9, decorator included)
# 11 def test_untouched       (span 11-12)
# 15 def test_top_level_untouched (span 15-16)
# 19 async def test_async_new (span 19-20)


def test_touching_body_line_flags_that_test():
  assert touched_test_names(_SOURCE, {9}) == {"test_existing"}


def test_touching_decorator_line_flags_that_test():
  assert touched_test_names(_SOURCE, {6}) == {"test_existing"}


def test_touching_async_test_is_detected():
  assert touched_test_names(_SOURCE, {20}) == {"test_async_new"}


def test_untouched_tests_are_not_flagged():
  # Editing test_existing must not drag in the neighbouring untouched tests.
  assert touched_test_names(_SOURCE, {8, 9}) == {"test_existing"}


def test_lines_outside_any_test_flag_nothing():
  assert touched_test_names(_SOURCE, {1, 2}) == set()  # imports / blank lines


def test_empty_touched_set_returns_empty():
  assert touched_test_names(_SOURCE, set()) == set()


def test_unparseable_source_returns_empty_gracefully():
  # pytest collection surfaces the syntax error itself; the parser must not raise.
  assert touched_test_names("def broken(:\n", {1}) == set()


def test_insertion_after_untouched_test_does_not_flag_it():
  # The regression that git's hunk-header heuristic caused: a new test added
  # right after an untouched test must flag ONLY the new test.
  source = textwrap.dedent(
      """\
      def test_old():
        assert True


      def test_new():
        assert True
      """
  )
  # test_old spans lines 1-2; test_new spans lines 5-6. The added lines are 5-6.
  assert touched_test_names(source, {5, 6}) == {"test_new"}


# --- touched_tests: markers, and class and module marker lines ---------------

_MARKED_SOURCE = textwrap.dedent(
    """\
    import pytest

    pytestmark = pytest.mark.integration_test


    @pytest.mark.tpu_only
    class TestTpu:

      def test_a(self):
        pass

      @pytest.mark.skip_on_tpu7x
      def test_b(self):
        pass


    @pytest.mark.post_training
    def test_c():
      pass


    class TestPost:
      pytestmark = [pytest.mark.tpu_only, pytest.mark.post_training()]

      def test_d(self):
        pass
    """
)
# Line numbers in _MARKED_SOURCE:
#  3 module pytestmark
#  6 class decorator, 7 class header (TestTpu's class lines: 6-8)
#  9 def test_a (span 9-10)
# 12 decorator of test_b (span 12-14)
# 17 decorator of test_c (span 17-19)
# 22 class TestPost, 23 its pytestmark
# 25 def test_d (span 25-26)


@pytest.mark.parametrize(
    "lines, expected",
    [
        pytest.param({10}, {"test_a"}, id="method-body"),
        pytest.param({6}, {"test_a", "test_b"}, id="class-decorator"),
        pytest.param({7}, {"test_a", "test_b"}, id="class-header"),
        pytest.param({23}, {"test_d"}, id="class-pytestmark"),
        pytest.param({3}, {"test_a", "test_b", "test_c", "test_d"}, id="module-pytestmark"),
        pytest.param({1, 4, 11, 20}, set(), id="outside-tests"),
    ],
)
def test_touched_tests_counts_class_and_module_marker_lines(lines, expected):
  assert set(touched_tests(_MARKED_SOURCE, lines)) == expected


def test_touched_tests_collects_test_class_and_module_markers():
  assert touched_tests(_MARKED_SOURCE, {3}) == {
      "test_a": {"integration_test", "tpu_only"},
      "test_b": {"integration_test", "tpu_only", "skip_on_tpu7x"},
      "test_c": {"integration_test", "post_training"},
      "test_d": {"integration_test", "tpu_only", "post_training"},
  }


# --- TPU7X flavors -----------------------------------------------------------

_INI = (("tests",), ("tests/dev", "tests/unit/ignored_test.py"))


@pytest.mark.parametrize(
    "path, markers, expected",
    [
        pytest.param("tests/unit/a_test.py", {"tpu_only"}, ["tpu7x-unit"], id="unit"),
        pytest.param(
            "tests/integration/a_test.py", {"tpu_only", "integration_test"}, ["tpu7x-integration"], id="integration"
        ),
        pytest.param(
            "tests/post_training/unit/a_test.py",
            {"tpu_only", "post_training"},
            ["tpu7x-post-training-unit"],
            id="post-training-unit",
        ),
        pytest.param(
            "tests/unit/a_test.py",
            {"tpu_only", "post_training"},
            ["tpu7x-post-training-unit"],
            id="post-training-in-unit",
        ),
        pytest.param(
            "tests/post_training/integration/a_test.py",
            {"tpu_only", "post_training", "integration_test"},
            [],
            id="post-training-integration-is-not-in-the-matrix",
        ),
        pytest.param("tests/unit/a_test.py", set(), [], id="no-hardware-marker-is-cpu-only"),
        pytest.param("tests/unit/a_test.py", {"tpu_backend"}, [], id="tpu-backend-is-cpu-only"),
        pytest.param("tests/unit/a_test.py", {"gpu_only"}, [], id="gpu-only"),
        pytest.param("tests/unit/a_test.py", {"tpu_only", "skip_on_tpu7x"}, [], id="skip-on-tpu7x"),
        pytest.param("tests/unit/a_test.py", {"tpu_only", "skip"}, [], id="skip"),
        pytest.param("tests/unit/ignored_test.py", {"tpu_only"}, [], id="ignored-in-pytest-ini"),
    ],
)
def test_is_eligible_routes_each_test_to_the_flavors_that_run_it(path, markers, expected):
  flavors = [name for name, flavor in TPU7X_FLAVORS.items() if _is_eligible(flavor, _INI, path, frozenset(markers))]
  assert flavors == expected


@pytest.mark.parametrize("expression", ["a or b", "(a and b)", "not not a"])
def test_marker_matches_rejects_expressions_it_cannot_evaluate(expression):
  with pytest.raises(ValueError):
    _marker_matches(expression, frozenset())


def test_read_pytest_ini_reads_testpaths_and_ignores(tmp_path):
  ini = tmp_path / "pytest.ini"
  ini.write_text(
      "[pytest]\ntestpaths =\n    tests\naddopts =\n    -rf --ignore=tests/dev\n    --ignore=tests/unit/a_test.py\n",
      encoding="utf-8",
  )
  assert _read_pytest_ini(ini) == (("tests",), ("tests/dev", "tests/unit/a_test.py"))
  assert _read_pytest_ini(tmp_path / "missing.ini") == (("tests",), ())
  testpaths, ignores = _read_pytest_ini(REPO_ROOT / "pytest.ini")
  assert testpaths == ("tests",) and "tests/dev" in ignores


# --- Command line ------------------------------------------------------------


def _git(repo, *args):
  """Run git in ``repo`` with a fixed identity and without the user's or the system's git config."""
  subprocess.run(
      ["git", "-c", "user.name=ci", "-c", "user.email=ci@example.com", "-c", "commit.gpgsign=false", *args],
      cwd=repo,
      env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
      check=True,
      capture_output=True,
  )


def _write_files(root, files):
  for relative, content in files.items():
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_text(content, encoding="utf-8")


def _make_pr_repo(tmp_path, base, change):
  """Return a checkout like CI's: ``origin/main`` has the ``base`` files, and HEAD is one commit that writes ``change``."""
  repo, origin = tmp_path / "repo", tmp_path / "origin.git"
  _git(tmp_path, "init", "-q", "--bare", str(origin))
  _git(tmp_path, "init", "-q", "-b", "main", str(repo))
  _git(repo, "remote", "add", "origin", str(origin))
  _write_files(repo, base)
  _git(repo, "add", "-A")
  _git(repo, "commit", "-q", "-m", "base")
  _git(repo, "push", "-q", "origin", "main")
  _git(repo, "checkout", "-q", "-b", "feature")
  _write_files(repo, change)
  _git(repo, "add", "-A")
  _git(repo, "commit", "-q", "-m", "change")
  return repo


def _run_cli(cwd, *args):
  """Run the detector like CI does, with ``python3 -I -S``, which can import only the standard library."""
  return subprocess.run(
      [sys.executable, "-I", "-S", newly_added_detection.__file__, "--base", "main", *args],
      cwd=cwd,
      capture_output=True,
      text=True,
      check=False,
  )


_TPU_TESTS = "import pytest\n\n\n@pytest.mark.tpu_only\ndef test_old():\n  pass\n"
_BASE = {
    "pytest.ini": "[pytest]\ntestpaths = tests\naddopts = --ignore=tests/dev\n",
    "tests/unit/tpu_test.py": _TPU_TESTS,
}
_CPU_TEST = {"tests/unit/cpu_test.py": "def test_cpu():\n  pass\n"}


def test_cli_lists_the_touched_tests_and_the_tpu7x_flavors_that_run_them(tmp_path):
  new_tpu_test = _TPU_TESTS + "\n\n@pytest.mark.tpu_only\ndef test_new():\n  pass\n"
  repo = _make_pr_repo(tmp_path, _BASE, {"tests/unit/tpu_test.py": new_tpu_test, **_CPU_TEST})
  listed = _run_cli(repo)
  assert listed.returncode == 0, listed.stderr
  assert listed.stdout.splitlines() == ["tests/unit/cpu_test.py::test_cpu", "tests/unit/tpu_test.py::test_new"]
  assert _run_cli(repo, "--flavor", "tpu7x-unit").stdout.splitlines() == ["tests/unit/tpu_test.py::test_new"]
  assert _run_cli(repo, "--flavor", "tpu7x-integration").stdout == ""
  eligible = _run_cli(repo, "--eligible-flavors")
  # analyze_code_changes.sh accepts only a single line that holds a compact JSON array.
  assert eligible.stdout == '["tpu7x-unit"]\n', eligible.stderr
  assert "tpu7x-unit: tests/unit/tpu_test.py::test_new" in eligible.stderr


def test_cli_prints_an_empty_array_when_no_tpu7x_flavor_runs_a_touched_test(tmp_path):
  repo = _make_pr_repo(tmp_path, _BASE, _CPU_TEST)
  assert _run_cli(repo, "--eligible-flavors").stdout == "[]\n"


def test_cli_exits_2_when_it_cannot_diff(tmp_path):
  result = _run_cli(tmp_path, "--eligible-flavors")
  assert result.returncode == 2 and "cannot diff" in result.stderr and not result.stdout


# --- is_changed_item (the tests/conftest.py match) ---------------------------


def test_is_changed_item_matches_absl_parameterized_methods_by_their_def_name():
  # pytest collects an absl parameterized def only under generated names, so matching on item.name or
  # item.originalname alone never marks it newly_added, and its TPU7X job would select nothing.
  parameterized = pytest.importorskip("absl.testing.parameterized")

  # Built here, not at module level, where pytest would collect it as real tests.
  class AbslCase(parameterized.TestCase):

    @parameterized.named_parameters(("bf16", 1), ("f32", 2))
    def test_named(self, x):
      del x

    @parameterized.parameters(1, 2)
    def test_plain(self, x):
      del x

  path = "tests/unit/absl_case_test.py"
  generated = sorted(name for name in vars(AbslCase) if name.startswith("test"))
  assert generated == ["test_named_bf16", "test_named_f32", "test_plain0", "test_plain1"]
  for name in generated:
    def_name = "test_named" if name.startswith("test_named") else "test_plain"
    # pytest collects a generated method under its generated name, as both name and originalname.
    item = types.SimpleNamespace(
        nodeid=f"{path}::AbslCase::{name}", name=name, originalname=name, function=getattr(AbslCase, name)
    )
    assert is_changed_item(item, {(path, def_name)}), name
    assert not is_changed_item(item, {("tests/unit/other_test.py", def_name)}), name


def test_is_changed_item_matches_pytest_parametrize_items_by_originalname():
  path = "tests/unit/param_test.py"
  item = types.SimpleNamespace(nodeid=f"{path}::test_param[1-a]", name="test_param[1-a]", originalname="test_param")
  assert is_changed_item(item, {(path, "test_param")})
  assert not is_changed_item(item, {(path, "test_other"), ("tests/unit/other_test.py", "test_param")})
  assert not is_changed_item(item, set())


# --- analyze_code_changes.sh -------------------------------------------------


def _run_analyze_script(tmp_path, env=None, detector_rc=0, detector_stdout=""):
  """Run analyze_code_changes.sh on a PR that changes only README.md, with a fake ``python3`` as the detector.

  Returns the GITHUB_OUTPUT writes as a dict (the last write of a key wins, as on GitHub) and the fake's calls.
  """
  repo = _make_pr_repo(tmp_path, {"README.md": "base\n"}, {"README.md": "changed\n"})
  fake = tmp_path / "bin" / "python3"
  fake.parent.mkdir()
  (tmp_path / "stdout").write_text(detector_stdout, encoding="utf-8")
  fake.write_text(
      f"#!/bin/sh\necho \"$@\" >> '{tmp_path}/calls'\ncat '{tmp_path}/stdout'\nexit {detector_rc}\n", encoding="utf-8"
  )
  fake.chmod(0o755)
  output = tmp_path / "github_output"
  output.touch()
  # Drop the values that the CI job running these tests has for the variables that the script reads.
  read_by_script = ("PR_NUMBER", "GITHUB_REPOSITORY", "FORCE_ALL_TESTS", "RUN_TPU7X", "EVENT_ACTION", "IS_DRAFT")
  run_env = {
      k: v for k, v in os.environ.items() if k not in read_by_script + ("HAS_SKIP_CI_TESTS_LABEL", "IS_COPYBARA_PR")
  }
  run_env.update(PATH=f"{fake.parent}{os.pathsep}{os.environ['PATH']}", EVENT_NAME="pull_request", GITHUB_BASE_REF="main")
  run_env.update(GITHUB_OUTPUT=str(output), **(env or {}))
  script = REPO_ROOT / ".github/scripts/analyze_code_changes.sh"
  result = subprocess.run(["bash", str(script)], cwd=repo, env=run_env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, result.stdout + result.stderr
  writes = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines() if "=" in line)
  calls = tmp_path / "calls"
  return writes, calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


@pytest.mark.parametrize(
    "detector_rc, detector_stdout, run_tpu7x_tests, tpu7x_flavors",
    [
        pytest.param(2, "", "true", None, id="detection-failed"),
        pytest.param(0, "not json\n", "true", None, id="malformed-output"),
        pytest.param(0, "[]\n", "false", None, id="no-flavor"),
        pytest.param(0, '["tpu7x-unit"]\n', "true", '["tpu7x-unit"]', id="one-flavor"),
    ],
)
def test_analyze_code_changes_maps_the_detector_output_to_the_tpu7x_outputs(
    tmp_path, detector_rc, detector_stdout, run_tpu7x_tests, tpu7x_flavors
):
  # A failed or malformed detection leaves tpu7x_flavors unset, so tpu7x-tests runs every flavor (fail-safe).
  writes, calls = _run_analyze_script(tmp_path, detector_rc=detector_rc, detector_stdout=detector_stdout)
  assert calls == ["tests/utils/newly_added_detection.py --base main --eligible-flavors"]
  assert writes["run_tpu7x_tests"] == run_tpu7x_tests
  assert writes.get("tpu7x_flavors") == tpu7x_flavors


def test_analyze_code_changes_run_tpu7x_selects_every_flavor_and_waits_for_skip_ci_tests(tmp_path):
  # run-tpu7x skips the detector and builds the package, but while the PR has skip-ci-tests the TPU7X tests are
  # gated like the other heavy suites, so All Required Tests Passed fails until the label is removed.
  env = {"RUN_TPU7X": "true", "HAS_SKIP_CI_TESTS_LABEL": "true", "EVENT_ACTION": "labeled"}
  writes, calls = _run_analyze_script(tmp_path, env=env)
  assert not calls
  assert writes["run_tpu7x_tests"] == "true" and "tpu7x_flavors" not in writes
  assert writes["run_tests"] == "true"
  assert writes["run_heavy_tests"] == "false" and writes["heavy_tests_gated"] == "true"
