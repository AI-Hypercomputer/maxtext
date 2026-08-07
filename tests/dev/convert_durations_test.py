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
