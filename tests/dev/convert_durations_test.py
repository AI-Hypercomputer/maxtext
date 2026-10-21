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

"""Tests for convert_durations utility."""

# pylint: disable=redefined-outer-name

import pytest
from tools.dev import convert_durations


@pytest.fixture
def mock_repo_root(tmp_path):
  """Creates a temporary mock directory structure."""
  (tmp_path / "tests" / "unit").mkdir(parents=True)
  (tmp_path / "tests" / "gather_reduce_sc_test.py").touch()
  (tmp_path / "tests" / "unit" / "attention_test.py").touch()
  (tmp_path / "tests" / "unit" / "no_class_test.py").touch()
  return str(tmp_path)


def test_convert_entry_legacy_format(mock_repo_root):
  res = convert_durations.convert_entry(
      "tests.gather_reduce_sc_test.GatherReduceScTest.test_column0",
      0.003,
      repo_root=mock_repo_root,
  )
  assert res == (
      "",
      "tests/gather_reduce_sc_test.py::GatherReduceScTest::test_column0",
      0.003,
  )


def test_convert_entry_flavor_prefixed(mock_repo_root):
  res = convert_durations.convert_entry(
      "cpu-unit::tests.unit.attention_test.AttentionTest.test_dot_product",
      64.84,
      repo_root=mock_repo_root,
  )
  assert res == (
      "cpu-unit",
      "tests/unit/attention_test.py::AttentionTest::test_dot_product",
      64.84,
  )


def test_convert_entry_non_existent_file(mock_repo_root):
  res = convert_durations.convert_entry(
      "tests.unit.non_existent_test.FooTest.test_bar",
      1.0,
      repo_root=mock_repo_root,
  )
  assert res is None


def test_convert_durations_flavor_priority(mock_repo_root):
  raw_data = {
      "cpu-unit::tests.unit.attention_test.AttentionTest.test_dot_product": 60.0,
      "tpu-unit::tests.unit.attention_test.AttentionTest.test_dot_product": 1.5,
      "tests.gather_reduce_sc_test.GatherReduceScTest.test_column0": 0.05,
  }

  cpu_durations = convert_durations.convert_durations(raw_data, target_flavor="cpu-unit", repo_root=mock_repo_root)
  assert cpu_durations["tests/unit/attention_test.py::AttentionTest::test_dot_product"] == 60.0
  assert cpu_durations["tests/gather_reduce_sc_test.py::GatherReduceScTest::test_column0"] == 0.05

  tpu_durations = convert_durations.convert_durations(raw_data, target_flavor="tpu-unit", repo_root=mock_repo_root)
  assert tpu_durations["tests/unit/attention_test.py::AttentionTest::test_dot_product"] == 1.5


def test_convert_entry_parameterized_with_dots(mock_repo_root):
  """Tests that dots within parametrization brackets are preserved."""
  res = convert_durations.convert_entry(
      "cpu-integration::tests.unit.attention_test.AttentionTest.test_hlo_diff[qwen3_1.7b-qwen3-1.7b-overrides2]",
      45.2,
      repo_root=mock_repo_root,
  )
  assert res == (
      "cpu-integration",
      "tests/unit/attention_test.py::AttentionTest::test_hlo_diff[qwen3_1.7b-qwen3-1.7b-overrides2]",
      45.2,
  )


def test_convert_entry_absl_dots_and_split_class_remap(tmp_path):
  """Preserves absl method dots and remaps split-class methods while keeping @parameterized_class suffixes."""
  unit_dir = tmp_path / "tests" / "unit"
  unit_dir.mkdir(parents=True)
  (unit_dir / "m_test.py").write_text(
      "from absl.testing import parameterized\nfrom parameterized import parameterized_class\n"
      "class _Base(parameterized.TestCase): pass\n"
      "class A(_Base):\n  def test_x(self): pass\n"
      "class B(_Base):\n  @parameterized.named_parameters(('v1.5', True))\n  def test_param(self, f): pass\n"
      "class C(_Base):\n  def test_x_moved(self): pass\n"
      "@parameterized_class([{'k': 0}])\nclass P:\n  def test_p(self): pass\n",
      encoding="utf-8",
  )
  root = str(tmp_path)
  assert convert_durations.convert_entry("tpu-unit::tests.unit.m_test.A.test_x", 4.0, root) == (
      "tpu-unit",
      "tests/unit/m_test.py::A::test_x",
      4.0,
  )
  assert convert_durations.convert_entry("tpu-unit::tests.unit.m_test.A.test_x_moved", 5.2, root) == (
      "tpu-unit",
      "tests/unit/m_test.py::C::test_x_moved",
      5.2,
  )
  assert convert_durations.convert_entry("tpu-unit::tests.unit.m_test.A.test_param_v1.5", 44.5, root) == (
      "tpu-unit",
      "tests/unit/m_test.py::B::test_param_v1.5",
      44.5,
  )
  assert convert_durations.convert_entry("cpu-unit::tests.unit.m_test.P_0.test_p", 3.1, root) == (
      "cpu-unit",
      "tests/unit/m_test.py::P_0::test_p",
      3.1,
  )
