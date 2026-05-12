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

from absl.testing import parameterized
import jax
import pytest


def dummy_decorator(func):
  """Standard transparent wrapper decorator preserving function metadata."""

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    return func(*args, **kwargs)

  return wrapper


class MarkerPropagationTest(parameterized.TestCase):
  """Validates that pytest markers propagate correctly through decorator stacks."""

  @pytest.mark.cpu_only
  @parameterized.named_parameters(
      {"testcase_name": "default", "unused": None},
  )
  def test_parameterized_cpu_only_marker_propagation(self, unused):
    """Verifies cpu_only marker above @parameterized propagates to generated methods."""
    has_tpu = any(d.platform == "tpu" for d in jax.devices())
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    assert not has_tpu, "cpu_only parameterized test accidentally executed on TPU hardware"
    assert not has_gpu, "cpu_only parameterized test accidentally executed on GPU hardware"

  @pytest.mark.cpu_only
  @dummy_decorator
  def test_standard_decorator_cpu_only_marker_propagation(self):
    """Verifies cpu_only marker above standard decorators propagates correctly."""
    has_tpu = any(d.platform == "tpu" for d in jax.devices())
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    assert not has_tpu, "cpu_only standard decorated test accidentally executed on TPU hardware"
    assert not has_gpu, "cpu_only standard decorated test accidentally executed on GPU hardware"


if __name__ == "__main__":
  unittest.main()
