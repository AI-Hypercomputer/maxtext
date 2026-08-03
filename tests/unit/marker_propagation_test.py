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

"""Unit tests validating pytest marker propagation through decorator stacks."""

import functools
import unittest
from unittest import mock

from absl.testing import parameterized
import jax

from tests.conftest import pytest_collection_modifyitems


class FakeMarker:
  """Fake pytest marker for testing."""

  def __init__(self, name):
    self.name = name


class FakeItem:
  """Fake pytest item for testing."""

  def __init__(self, name, marker_names):
    self.nodeid = f"test_file.py::{name}"
    self.name = name
    self._markers = [FakeMarker(m) for m in marker_names]
    self.added_markers = []

  def iter_markers(self):
    return iter(self._markers + self.added_markers)

  def add_marker(self, marker):
    if hasattr(marker, "name"):
      self.added_markers.append(marker)
    elif hasattr(marker, "mark"):
      self.added_markers.append(marker.mark)
    else:
      self.added_markers.append(marker)


def dummy_decorator(func):
  """Standard transparent wrapper decorator preserving function metadata."""

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    return func(*args, **kwargs)

  return wrapper


class MarkerPropagationTest(parameterized.TestCase):
  """Validates that pytest markers propagate correctly through decorator stacks."""

  @parameterized.named_parameters(
      {"testcase_name": "default", "unused": None},
  )
  def test_parameterized_cpu_only_marker_propagation(self, unused):
    """Verifies cpu_only marker above @parameterized propagates to generated methods."""
    has_tpu = any(d.platform == "tpu" for d in jax.devices())
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    assert not has_tpu, "cpu_only parameterized test accidentally executed on TPU hardware"
    assert not has_gpu, "cpu_only parameterized test accidentally executed on GPU hardware"

  @dummy_decorator
  def test_standard_decorator_cpu_only_marker_propagation(self):
    """Verifies cpu_only marker above standard decorators propagates correctly."""
    has_tpu = any(d.platform == "tpu" for d in jax.devices())
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    assert not has_tpu, "cpu_only standard decorated test accidentally executed on TPU hardware"
    assert not has_gpu, "cpu_only standard decorated test accidentally executed on GPU hardware"

  def test_auto_apply_cpu_only_marker(self):
    """Verifies cpu_only is auto-applied to tests without explicit hardware markers."""
    item_no_markers = FakeItem("test_no_markers", [])
    item_other_marker = FakeItem("test_other_marker", ["scheduled_only", "integration_test"])
    item_tpu = FakeItem("test_tpu", ["tpu_only"])
    item_gpu = FakeItem("test_gpu", ["gpu_only"])
    item_tpu_backend = FakeItem("test_tpu_backend", ["tpu_backend"])
    item_cpu = FakeItem("test_cpu", ["cpu_only"])

    items = [item_no_markers, item_other_marker, item_tpu, item_gpu, item_tpu_backend, item_cpu]
    mock_config = mock.MagicMock()

    with (
        mock.patch("tests.conftest.get_changed_tests", return_value=set()),
        mock.patch("tests.conftest._has_tpu_backend_support", return_value=True),
    ):
      pytest_collection_modifyitems(mock_config, items)

    def get_added_names(item):
      return [m.name if hasattr(m, "name") else str(m) for m in item.added_markers]

    self.assertIn("cpu_only", get_added_names(item_no_markers))
    self.assertIn("cpu_only", get_added_names(item_other_marker))
    self.assertIn("cpu_only", get_added_names(item_tpu_backend))
    self.assertNotIn("cpu_only", get_added_names(item_tpu))
    self.assertNotIn("cpu_only", get_added_names(item_gpu))
    self.assertNotIn("cpu_only", get_added_names(item_cpu))


if __name__ == "__main__":
  unittest.main()
