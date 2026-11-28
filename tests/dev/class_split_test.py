# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for class_split.py."""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock
import pytest

# The repository root: tests/dev/<this file>.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("class_split", REPO_ROOT / "tests/utils/class_split.py")
class_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(class_split)

group_key = class_split.group_key
load_durations = class_split.load_durations
deal_groups = class_split.deal_groups
ClassSplitPlugin = class_split.ClassSplitPlugin


class GroupKeyTest(unittest.TestCase):

  def test_method_with_parameters_maps_to_its_class(self):
    self.assertEqual(
        group_key("tests/unit/moe_test.py::RoutedMoeTest::test_x[bf16-ep1]"),
        "tests/unit/moe_test.py::RoutedMoeTest",
    )

  def test_module_level_function_maps_to_itself_without_parameters(self):
    self.assertEqual(group_key("tests/unit/moe_test.py::test_fn[p]"), "tests/unit/moe_test.py::test_fn")

  def test_parameter_value_containing_separators_does_not_shift_the_key(self):
    self.assertEqual(
        group_key("tests/unit/a_test.py::ATest::test_x[path::with[brackets]]"),
        "tests/unit/a_test.py::ATest",
    )

  def test_bracket_in_a_directory_name_keeps_the_class_key(self):
    self.assertEqual(
        group_key("tests/unit/d[1]/a_test.py::ATest::test_x[p]"),
        "tests/unit/d[1]/a_test.py::ATest",
    )

  def test_bracket_in_a_directory_name_keeps_the_function_key(self):
    self.assertEqual(group_key("tests/unit/d[1]/a_test.py::test_fn[p]"), "tests/unit/d[1]/a_test.py::test_fn")

  def test_nodeid_without_separator_is_its_own_key(self):
    self.assertEqual(group_key("tests/unit/a_test.py"), "tests/unit/a_test.py")

  def test_at_sign_in_a_directory_name_is_not_stripped_as_an_xdist_group(self):
    self.assertEqual(
        group_key("tests/unit/dir@1/a_test.py::ATest::test_x"),
        "tests/unit/dir@1/a_test.py::ATest",
    )
    self.assertEqual(
        group_key("tests/unit/dir@1/a_test.py::test_fn@G"),
        "tests/unit/dir@1/a_test.py::test_fn",
    )


class LoadDurationsTest(unittest.TestCase):

  def test_missing_file_gives_empty_durations_and_a_warning(self):
    durations, warning = load_durations("/nonexistent/.test_durations")
    self.assertEqual(durations, {})
    self.assertIn("not found", warning)

  def test_valid_file_is_read_and_non_numeric_values_are_skipped(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, ".test_durations")
      with open(path, "w", encoding="utf-8") as f:
        json.dump({"tests/a_test.py::T::test_a": 12.5, "tests/a_test.py::T::test_b": "oops"}, f)
      durations, warning = load_durations(path)
    self.assertEqual(durations, {"tests/a_test.py::T::test_a": 12.5})
    self.assertEqual(warning, "")

  def test_unreadable_json_gives_empty_durations_and_a_warning(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, ".test_durations")
      with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
      durations, warning = load_durations(path)
    self.assertEqual(durations, {})
    self.assertIn("could not be read", warning)

  def test_json_that_is_not_an_object_gives_empty_durations_and_a_warning(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, ".test_durations")
      with open(path, "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
      durations, warning = load_durations(path)
    self.assertEqual(durations, {})
    self.assertIn("not a JSON object", warning)

  def test_load_durations_indexes_ungrouped_fallback_for_xdist_group_entries(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, ".test_durations")
      with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tests/a_test.py::A::test_1@G": 12.0,
                "tests/b_test.py::B::test_1[user@host]@G": 8.0,
                "tests/c_test.py::C::test_1[p]@G]": 9.0,
                "tests/d_test.py::D::test_1": 3.0,
                "tests/d_test.py::D::test_1@G": 30.0,
            },
            f,
        )
      durations, warning = load_durations(path)
    self.assertEqual(warning, "")
    self.assertEqual(durations["tests/a_test.py::A::test_1"], 12.0)
    self.assertEqual(durations["tests/b_test.py::B::test_1[user@host]"], 8.0)
    self.assertEqual(durations["tests/c_test.py::C::test_1[p]"], 9.0)
    # Existing exact ungrouped entry is preserved rather than overwritten by the suffixed entry.
    self.assertEqual(durations["tests/d_test.py::D::test_1"], 3.0)
    self.assertEqual(durations["tests/d_test.py::D::test_1@G"], 30.0)


class DealGroupsTest(unittest.TestCase):

  def test_keeps_classes_together_and_deals_heaviest_class_first(self):
    nodeids = [
        "tests/a_test.py::ATest::test_1",
        "tests/a_test.py::ATest::test_2",
        "tests/b_test.py::BTest::test_1",
        "tests/c_test.py::CTest::test_1",
    ]
    durations = {nodeids[0]: 10.0, nodeids[1]: 10.0, nodeids[2]: 15.0, nodeids[3]: 5.0}

    groups = deal_groups(nodeids, durations, 2)

    # ATest (20 s) goes to shard 1, BTest (15 s) to shard 2, CTest (5 s) to the lighter shard 2.
    self.assertEqual(groups, [[nodeids[0], nodeids[1]], [nodeids[2], nodeids[3]]])

  def test_unknown_durations_get_the_mean_of_the_known_ones(self):
    nodeids = ["tests/x_test.py::X::t", "tests/y_test.py::Y::t", "tests/z_test.py::Z::t"]
    durations = {nodeids[0]: 10.0, nodeids[2]: 30.0}

    groups = deal_groups(nodeids, durations, 3)

    # Y is charged 20 s: order is Z (30), Y (20), X (10).
    self.assertEqual(groups, [[nodeids[2]], [nodeids[1]], [nodeids[0]]])

  def test_without_durations_every_test_weighs_the_same_and_ties_go_to_the_lowest_shard(self):
    nodeids = ["tests/a_test.py::A::t", "tests/b_test.py::B::t", "tests/c_test.py::C::t", "tests/d_test.py::D::t"]

    groups = deal_groups(nodeids, {}, 2)

    self.assertEqual(groups, [[nodeids[0], nodeids[2]], [nodeids[1], nodeids[3]]])

  def test_items_keep_their_collection_order_inside_a_shard(self):
    nodeids = [
        "tests/b_test.py::B::test_1",
        "tests/a_test.py::A::test_1",
        "tests/b_test.py::B::test_2",
        "tests/a_test.py::A::test_2",
    ]
    durations = {n: 1.0 for n in nodeids}
    durations[nodeids[1]] = 5.0  # A is heavier, so A is dealt first

    groups = deal_groups(nodeids, durations, 2)

    self.assertEqual(groups, [[nodeids[1], nodeids[3]], [nodeids[0], nodeids[2]]])

  def test_every_test_lands_in_exactly_one_shard(self):
    nodeids = [f"tests/m{m}_test.py::C{c}::test_{t}" for m in range(3) for c in range(3) for t in range(4)]
    durations = {n: float(1 + (i * 7) % 11) for i, n in enumerate(nodeids)}

    groups = deal_groups(nodeids, durations, 4)

    dealt = [n for group in groups for n in group]
    self.assertEqual(sorted(dealt), sorted(nodeids))
    self.assertEqual(len(dealt), len(set(dealt)))
    for group in groups:
      keys = [group_key(n) for n in group]
      for key in set(keys):
        self.assertEqual(sum(1 for n in nodeids if group_key(n) == key), keys.count(key))

  def test_splits_below_one_is_rejected(self):
    with self.assertRaises(ValueError):
      deal_groups(["tests/a_test.py::A::t"], {}, 0)

  def test_duration_lookup_falls_back_to_nodeid_without_xdist_group_suffix(self):
    nodeids = [
        "tests/a_test.py::A::test_1@tests_a_test_A",
        "tests/b_test.py::B::test_1[user@host]@tests_b_test_B",
        "tests/c_test.py::C::test_1[user@host]",
    ]
    # Baseline recorded before `@<group>` suffixes existed (plus an exact `@` inside `[...]`).
    durations = {
        "tests/a_test.py::A::test_1": 25.0,
        "tests/b_test.py::B::test_1[user@host]": 50.0,
        "tests/c_test.py::C::test_1[user@host]": 10.0,
    }

    groups = deal_groups(nodeids, durations, 2)

    # B (50s) outweighs A (25s) + C (10s), so B lands alone on shard 1 (a miss on B drops it to 17.5s and flips shards).
    self.assertEqual(groups, [[nodeids[1]], [nodeids[0], nodeids[2]]])
    with tempfile.TemporaryDirectory() as tmpdir:
      durations_path = os.path.join(tmpdir, ".test_durations")
      with open(durations_path, "w", encoding="utf-8") as f:
        json.dump(durations, f)
      out = io.StringIO()
      items = [_fake_item(n) for n in nodeids]
      config = _fake_config(1, 1, [])
      with contextlib.redirect_stdout(out):
        ClassSplitPlugin(config, durations_path=durations_path).pytest_collection_modifyitems(config, items)
    self.assertIn("3 tests in 3 classes, 3 with a recorded duration", out.getvalue())

  def test_exact_xdist_group_suffixed_duration_wins_over_ungrouped_fallback(self):
    nodeids = [
        "tests/a_test.py::A::test_1@G",
        "tests/b_test.py::B::test_1",
    ]
    durations = {
        "tests/a_test.py::A::test_1": 1.0,
        "tests/a_test.py::A::test_1@G": 50.0,
        "tests/b_test.py::B::test_1": 10.0,
    }

    groups = deal_groups(nodeids, durations, 2)

    # Exact `@G` duration (50s) outweighs B (10s), so A lands on shard 1.
    self.assertEqual(groups, [[nodeids[0]], [nodeids[1]]])

  def test_zero_weight_nodeids_get_zero_weight_and_do_not_bias_fallback(self):
    skipped = [f"tests/skip_test.py::HugeSkipped::test_{i}" for i in range(10)]
    active_a = "tests/a_test.py::RealA::test_1"
    active_b = "tests/b_test.py::RealB::test_1"
    unknown_c = "tests/c_test.py::RealC::test_1"
    nodeids = [*skipped, active_a, active_b, unknown_c]
    # Give a skipped item a huge stale duration to verify zero_weight overrides durations and fallback mean.
    durations = {skipped[0]: 1000.0, active_a: 20.0, active_b: 10.0}

    groups = deal_groups(nodeids, durations, 2, zero_weight_nodeids=skipped)

    # Fallback mean is (20 + 10) / 2 = 15 (not inflated by skipped[0]'s stale 1000s), so RealA (20s)
    # lands on shard 1, RealC (15s) + RealB (10s) land on shard 2, and HugeSkipped (0s) does not steal shard 1.
    self.assertIn(active_a, groups[0])
    self.assertNotIn(active_b, groups[0])
    self.assertNotIn(unknown_c, groups[0])
    self.assertIn(active_b, groups[1])
    self.assertIn(unknown_c, groups[1])
    # All skipped items stay together in their class on the lighter shard (shard 1: 20s vs shard 2: 25s).
    self.assertEqual([n for n in groups[0] if n in skipped], skipped)

  def test_zero_weight_groups_distribute_across_tied_shards(self):
    nodeids = [f"tests/skip_test.py::S{i}::test_1" for i in range(4)]

    groups = deal_groups(nodeids, {}, 2, zero_weight_nodeids=nodeids)

    self.assertEqual(groups, [[nodeids[0], nodeids[2]], [nodeids[1], nodeids[3]]])


def _fake_config(splits, group, deselected, split_by="class"):
  """A stand-in for pytest's Config: options, a deselection hook that records, and no terminal reporter."""
  options = {"class_splits": splits, "class_group": group, "class_split_by": split_by}
  return types.SimpleNamespace(
      getoption=options.get,
      hook=types.SimpleNamespace(
          pytest_deselected=lambda items: deselected.extend(items)  # pylint: disable=unnecessary-lambda
      ),
      pluginmanager=types.SimpleNamespace(get_plugin=lambda name: None),
  )


def _fake_item(nodeid, xdist_group=None):
  def _get_closest_marker(name):
    if name == "xdist_group" and xdist_group is not None:
      return types.SimpleNamespace(args=(xdist_group,), kwargs={})
    return None

  return types.SimpleNamespace(nodeid=nodeid, get_closest_marker=_get_closest_marker)


class ClassSplitPluginTest(unittest.TestCase):

  def test_requires_both_options(self):
    with self.assertRaises(pytest.UsageError):
      ClassSplitPlugin(_fake_config(3, None, []))

  def test_rejects_a_group_outside_the_split_range(self):
    with self.assertRaises(pytest.UsageError):
      ClassSplitPlugin(_fake_config(3, 4, []))
    with self.assertRaises(pytest.UsageError):
      ClassSplitPlugin(_fake_config(3, 0, []))

  def test_rejects_fewer_than_one_split(self):
    with self.assertRaises(pytest.UsageError):
      ClassSplitPlugin(_fake_config(0, 1, []))

  def test_keeps_only_the_classes_dealt_to_this_shard_and_deselects_the_rest(self):
    nodeids = [
        "tests/a_test.py::ATest::test_1",
        "tests/a_test.py::ATest::test_2",
        "tests/b_test.py::BTest::test_1",
        "tests/c_test.py::CTest::test_1",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
      durations_path = os.path.join(tmpdir, ".test_durations")
      with open(durations_path, "w", encoding="utf-8") as f:
        json.dump({nodeids[0]: 10.0, nodeids[1]: 10.0, nodeids[2]: 15.0, nodeids[3]: 5.0}, f)
      deselected = []
      config = _fake_config(2, 2, deselected)
      items = [_fake_item(n) for n in nodeids]
      plugin = ClassSplitPlugin(config, durations_path=durations_path)

      plugin.pytest_collection_modifyitems(config, items)

    self.assertEqual([i.nodeid for i in items], [nodeids[2], nodeids[3]])
    self.assertEqual([i.nodeid for i in deselected], [nodeids[0], nodeids[1]])

  def test_works_without_a_durations_file(self):
    nodeids = ["tests/a_test.py::A::t", "tests/b_test.py::B::t"]
    deselected = []
    config = _fake_config(2, 1, deselected)
    items = [_fake_item(n) for n in nodeids]
    plugin = ClassSplitPlugin(config, durations_path="/nonexistent/.test_durations")

    plugin.pytest_collection_modifyitems(config, items)

    self.assertEqual([i.nodeid for i in items], [nodeids[0]])
    self.assertEqual([i.nodeid for i in deselected], [nodeids[1]])

  def test_empty_collection_is_left_alone(self):
    deselected = []
    config = _fake_config(2, 1, deselected)
    items = []

    ClassSplitPlugin(config, durations_path="/nonexistent/.test_durations").pytest_collection_modifyitems(config, items)

    self.assertEqual(items, [])
    self.assertEqual(deselected, [])

  def test_status_line_reports_how_many_tests_had_a_recorded_duration(self):
    nodeids = ["tests/a_test.py::A::t", "tests/b_test.py::B::t"]
    with tempfile.TemporaryDirectory() as tmpdir:
      durations_path = os.path.join(tmpdir, ".test_durations")
      with open(durations_path, "w", encoding="utf-8") as f:
        json.dump({"tests/zzz_test.py::Z::t": 5.0}, f)  # a valid file whose keys match nothing
      config = _fake_config(1, 1, [])
      items = [_fake_item(n) for n in nodeids]
      out = io.StringIO()
      with contextlib.redirect_stdout(out):
        ClassSplitPlugin(config, durations_path=durations_path).pytest_collection_modifyitems(config, items)

    self.assertIn("2 tests in 2 classes, 0 with a recorded duration", out.getvalue())

  def test_with_a_terminal_reporter_the_lines_wait_for_the_collection_summary(self):
    config = _fake_config(1, 1, [])
    config.pluginmanager = types.SimpleNamespace(get_plugin=lambda name: object())
    plugin = ClassSplitPlugin(config, durations_path="/nonexistent/.test_durations")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      plugin.pytest_collection_modifyitems(config, [_fake_item("tests/a_test.py::A::t")])

    self.assertEqual(out.getvalue(), "")
    lines = plugin.pytest_report_collectionfinish()
    self.assertEqual(len(lines), 2)
    self.assertIn("[class-split] warning: durations file", lines[0])
    self.assertEqual(lines[1], "[class-split] Running group 1/1: 1 tests in 1 classes, 0 with a recorded duration")

  def test_xdist_group_mode_keeps_grouped_items_together_and_splits_ungrouped_class(self):
    items = [
        _fake_item("tests/a_test.py::Grouped::test_1", xdist_group="G"),
        _fake_item("tests/a_test.py::Grouped::test_2", xdist_group="G"),
        _fake_item("tests/b_test.py::Stateless::test_1"),
        _fake_item("tests/b_test.py::Stateless::test_2"),
        _fake_item("tests/b_test.py::Stateless::test_3"),
    ]
    durations = {
        "tests/a_test.py::Grouped::test_1": 6.0,
        "tests/a_test.py::Grouped::test_2": 6.0,
        "tests/b_test.py::Stateless::test_1": 11.0,
        "tests/b_test.py::Stateless::test_2": 10.0,
        "tests/b_test.py::Stateless::test_3": 9.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
      durations_path = os.path.join(tmpdir, ".test_durations")
      with open(durations_path, "w", encoding="utf-8") as f:
        json.dump(durations, f)
      shard_items = []
      for group_idx in (1, 2):
        copy = list(items)
        config = _fake_config(2, group_idx, [], split_by="xdist_group")
        ClassSplitPlugin(config, durations_path=durations_path).pytest_collection_modifyitems(config, copy)
        shard_items.append([i.nodeid for i in copy])

    # Grouped (12 s total) stays together on shard 1; Stateless splits across both shards
    # (test_1=11 s and test_2=10 s on shard 2, test_3=9 s on shard 1).
    self.assertEqual(
        shard_items[0],
        [
            "tests/a_test.py::Grouped::test_1",
            "tests/a_test.py::Grouped::test_2",
            "tests/b_test.py::Stateless::test_3",
        ],
    )
    self.assertEqual(
        shard_items[1],
        ["tests/b_test.py::Stateless::test_1", "tests/b_test.py::Stateless::test_2"],
    )


# With no .test_durations file every test weighs 1. Once `-m "not integration_test"` has removed TestHeavy, TestA
# and TestB weigh 2 each and get one group each. Dealt before `-m`, TestHeavy would take group 1 and leave it empty.
SAMPLE_MODULE = """
import pytest

@pytest.mark.integration_test
class TestHeavy:
  def test_1(self): pass
  def test_2(self): pass
  def test_3(self): pass

class TestA:
  def test_1(self): pass
  def test_2(self): pass

class TestB:
  def test_1(self): pass
  def test_2(self): pass
"""


class ClassSplitEndToEndTest(unittest.TestCase):
  """Runs `pytest --collect-only` through the class_split plugin on a sample module in a temporary directory.

  `plugins=[class_split]` loads the plugin hooks directly (without importing JAX/absl from tests/conftest.py),
  and `-c pytest.ini` brings the repo's markers and --strict-markers, as in CI.
  """

  def _run_pytest(self, workdir, *args):
    """Runs `pytest --collect-only` on the sample module and returns its standard output."""
    out = io.StringIO()
    err = io.StringIO()
    sys.modules.pop("sample_test", None)
    try:
      with (
          mock.patch.dict(os.environ, {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}),
          contextlib.chdir(workdir),
          contextlib.redirect_stdout(out),
          contextlib.redirect_stderr(err),
      ):
        returncode = pytest.main(
            [
                "--collect-only",
                "-s",
                "-p",
                "no:cacheprovider",
                "-c",
                str(REPO_ROOT / "pytest.ini"),
                f"--rootdir={workdir}",
                "-m",
                "not integration_test",
                *args,
                "sample_test.py",
            ],
            plugins=[class_split],
        )
    finally:
      sys.modules.pop("sample_test", None)
    self.assertEqual(int(returncode), 0, out.getvalue() + err.getvalue())
    return out.getvalue()

  def _collect(self, workdir, *args):
    """Returns the nodeids that `pytest --collect-only -q` keeps from the sample module."""
    output = self._run_pytest(workdir, "-q", *args)
    return [line for line in output.splitlines() if line.startswith("sample_test.py::")]

  def test_options_register_and_the_split_runs_after_marker_deselection(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # resolve(): on macOS the temporary directory sits behind a symlink, which would change the printed nodeids.
      workdir = pathlib.Path(tmpdir).resolve()
      (workdir / "sample_test.py").write_text(SAMPLE_MODULE, encoding="utf-8")
      unsplit = self._collect(workdir)
      first = self._collect(workdir, "--class-splits=2", "--class-group=1")
      second = self._collect(workdir, "--class-splits=2", "--class-group=2")

    self.assertCountEqual(first + second, unsplit)
    self.assertEqual({nodeid.split("::")[1] for nodeid in first}, {"TestA"})
    self.assertEqual({nodeid.split("::")[1] for nodeid in second}, {"TestB"})

  def test_status_lines_are_printed_on_their_own_lines(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      workdir = pathlib.Path(tmpdir).resolve()
      (workdir / "sample_test.py").write_text(SAMPLE_MODULE, encoding="utf-8")
      # With -v and an output that is not a terminal (as in CI), pytest leaves "collecting ... " unterminated.
      output = self._run_pytest(workdir, "-v", "--class-splits=2", "--class-group=1")

    lines = [line for line in output.splitlines() if "[class-split]" in line]
    self.assertEqual(len(lines), 2, output)  # the missing-durations warning and the status line
    for line in lines:
      self.assertTrue(line.startswith("[class-split]"), line)
    self.assertEqual(lines[1], "[class-split] Running group 1/2: 2 tests in 1 classes, 0 with a recorded duration")

  def test_xdist_group_mode_end_to_end_splits_ungrouped_and_keeps_grouped_together(self):
    sample = """
import pytest

@pytest.mark.xdist_group("Grouped")
class TestGrouped:
  def test_1(self): pass
  def test_2(self): pass

class TestStateless:
  def test_1(self): pass
  def test_2(self): pass
"""
    with tempfile.TemporaryDirectory() as tmpdir:
      workdir = pathlib.Path(tmpdir).resolve()
      (workdir / "sample_test.py").write_text(sample, encoding="utf-8")
      unsplit = self._collect(workdir)
      first = self._collect(workdir, "--class-splits=2", "--class-group=1", "--class-split-by=xdist_group")
      second = self._collect(workdir, "--class-splits=2", "--class-group=2", "--class-split-by=xdist_group")

    self.assertCountEqual(first + second, unsplit)
    # TestGrouped (weight 2) lands together on group 1; TestStateless's two tests (weight 1 each) land on group 2.
    self.assertEqual(first, ["sample_test.py::TestGrouped::test_1", "sample_test.py::TestGrouped::test_2"])
    self.assertEqual(second, ["sample_test.py::TestStateless::test_1", "sample_test.py::TestStateless::test_2"])

  def test_unconditionally_skipped_tests_get_zero_weight_and_do_not_steal_shards(self):
    sample = """
import unittest
import pytest

class TestHugePytestSkip:
  @pytest.mark.skip(reason="disabled")
  @pytest.mark.parametrize("i", range(10))
  def test_skipped(self, i): pass

@unittest.skip("whole class disabled")
class TestUnittestSkippedClass(unittest.TestCase):
  def test_1(self): pass
  def test_2(self): pass
  def test_3(self): pass

@pytest.mark.skipif(True, reason="always skipped")
class TestStackedSkipIf:
  @pytest.mark.skipif(False, reason="not skipped")
  def test_1(self): pass

class TestRealA(unittest.TestCase):
  def test_active(self): pass
  @unittest.skip("method disabled")
  def test_skipped_method(self): pass
  @pytest.mark.skipif(condition=True, reason="kwarg condition")
  def test_kwarg_skipif_true(self): pass
  @pytest.mark.skipif(reason="unconditional skipif")
  def test_empty_skipif(self): pass

class TestRealB:
  @pytest.mark.skipif(False, reason="active")
  def test_active(self): pass
  @pytest.mark.skipif(condition=False, reason="kwarg active")
  def test_kwarg_active(self): pass
  @pytest.mark.skipif("1 == 1", reason="string condition is not evaluated at collection time")
  def test_string_skipif(self): pass
"""
    # Calibrated so every skip/non-skip branch flips shard 1 if broken:
    # - Active weights: TestRealA=30s, TestRealB=10s + 10s + fallback(50/3=16.67s) = 36.67s.
    # - If any TestRealB method were wrongly zeroed, TestRealB drops to <=30s and loses to TestRealA.
    # - If any skipped method/class failed to zero-weight, its stale 500s or fallback steals shard 1.
    durations = {
        "sample_test.py::TestStackedSkipIf::test_1": 500.0,
        "sample_test.py::TestRealA::test_active": 30.0,
        "sample_test.py::TestRealA::test_skipped_method": 500.0,
        "sample_test.py::TestRealA::test_kwarg_skipif_true": 500.0,
        "sample_test.py::TestRealA::test_empty_skipif": 500.0,
        "sample_test.py::TestRealB::test_active": 10.0,
        "sample_test.py::TestRealB::test_kwarg_active": 10.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
      workdir = pathlib.Path(tmpdir).resolve()
      (workdir / "sample_test.py").write_text(sample, encoding="utf-8")
      (workdir / ".test_durations").write_text(json.dumps(durations), encoding="utf-8")
      unsplit = self._collect(workdir)
      first = self._collect(workdir, "--class-splits=2", "--class-group=1")
      second = self._collect(workdir, "--class-splits=2", "--class-group=2")
      second_verbose = self._run_pytest(workdir, "-v", "--class-splits=2", "--class-group=2")
      first_xg = self._collect(workdir, "--class-splits=2", "--class-group=1", "--class-split-by=xdist_group")
      second_xg = self._collect(workdir, "--class-splits=2", "--class-group=2", "--class-split-by=xdist_group")

    self.assertCountEqual(first + second, unsplit)
    self.assertCountEqual(first_xg + second_xg, unsplit)
    self.assertIn("sample_test.py::TestRealB::test_active", first)
    self.assertIn("sample_test.py::TestRealB::test_kwarg_active", first)
    self.assertIn("sample_test.py::TestRealB::test_string_skipif", first)
    self.assertNotIn("sample_test.py::TestRealA::test_active", first)
    self.assertIn("sample_test.py::TestRealA::test_active", second)
    self.assertIn("sample_test.py::TestHugePytestSkip::test_skipped[0]", second)
    # Status line excludes stale durations on skipped items (only TestRealA::test_active counts).
    self.assertIn(", 1 with a recorded duration", second_verbose)
    # In xdist_group mode: TestRealA::test_active (30s) lands alone among active tests on shard 1,
    # while TestRealB's 3 active items (16.67s + 10s + 10s = 36.67s) land together on shard 2.
    self.assertIn("sample_test.py::TestRealA::test_active", first_xg)
    self.assertNotIn("sample_test.py::TestRealB::test_active", first_xg)
    self.assertNotIn("sample_test.py::TestRealB::test_kwarg_active", first_xg)
    self.assertNotIn("sample_test.py::TestRealB::test_string_skipif", first_xg)
    self.assertIn("sample_test.py::TestRealB::test_active", second_xg)
    self.assertIn("sample_test.py::TestRealB::test_kwarg_active", second_xg)
    self.assertIn("sample_test.py::TestRealB::test_string_skipif", second_xg)


if __name__ == "__main__":
  unittest.main()
