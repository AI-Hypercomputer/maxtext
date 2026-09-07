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
