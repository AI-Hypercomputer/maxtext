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
import tempfile
import types
import unittest
import pytest

spec = importlib.util.spec_from_file_location("class_split", "tests/utils/class_split.py")
class_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(class_split)

group_key = class_split.group_key
load_durations = class_split.load_durations
deal_groups = class_split.deal_groups
ClassSplitPlugin = class_split.ClassSplitPlugin


@pytest.mark.cpu_only
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

  def test_nodeid_without_separator_is_its_own_key(self):
    self.assertEqual(group_key("tests/unit/a_test.py"), "tests/unit/a_test.py")


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


def _fake_config(splits, group, deselected):
  """A stand-in for pytest's Config: options, a deselection hook that records, and no terminal reporter."""
  options = {"class_splits": splits, "class_group": group}
  return types.SimpleNamespace(
      getoption=options.get,
      hook=types.SimpleNamespace(
          pytest_deselected=lambda items: deselected.extend(items)  # pylint: disable=unnecessary-lambda
      ),
      pluginmanager=types.SimpleNamespace(get_plugin=lambda name: None),
  )


def _fake_item(nodeid):
  return types.SimpleNamespace(nodeid=nodeid)


@pytest.mark.cpu_only
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


if __name__ == "__main__":
  unittest.main()
