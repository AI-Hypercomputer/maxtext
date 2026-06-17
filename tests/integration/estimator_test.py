# Copyright 2025 Google LLC
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

"""
E2E integration tests for the MaxText estimator.

These tests exercise the estimator against a real (small) model compilation
to verify the search_policy_only and is_oom functions work end-to-end.
The tests use a small default model (global_parameter_scale=1) to stay fast.
"""

import unittest
import pytest
import jax

from tests.utils.test_helpers import get_test_config_path
from maxtext.utils.estimator import (
    Action,
    RematPolicy,
    is_oom,
    search_policy_only,
)


@pytest.mark.skip_on_tpu7x
class EstimatorE2ETest(unittest.TestCase):
  """E2E tests for the estimator search logic."""

  def get_device_user_facing_name(self):
    """Gets TPU device user facing name to generate correct AOT arguments."""
    devices = jax.devices()
    if not devices or "tpu" not in devices[0].platform.lower():
      pytest.skip("This test requires a TPU environment.")

    num_devices = len(devices)
    device_kind = devices[0].device_kind
    device_info = {
        "TPU v4": ("v4", 2 * num_devices),
        "TPU v5 lite": ("v5e", num_devices),
        "TPU v5": ("v5p", 2 * num_devices),
        "TPU v6": ("v6e", num_devices),
    }

    prefix, topology_devices = next((v for k, v in device_info.items() if k in device_kind), (None, None))
    if prefix is None:
      raise ValueError(f"Unsupported TPU device kind for estimator test: {device_kind}")

    return f"{prefix}-{topology_devices}"

  def _make_base_argv(self, extra_args=None):
    """Build the base argv tuple for a small default model."""
    topology = self.get_device_user_facing_name()
    args = [
        None,
        get_test_config_path(),
        f"compile_topology={topology}",
        "compile_topology_num_slices=1",
        "global_parameter_scale=1",
        "per_device_batch_size=2",
        "max_target_length=128",
        "dataset_type=synthetic",
        "enable_checkpointing=False",
        "write_estimator_result=False",
        "log_config=False",
    ]
    if extra_args:
      args.extend(extra_args)
    return tuple(args)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_is_oom_returns_bool(self):
    """Verify is_oom returns a boolean for a small model with full remat."""
    base_argv = self._make_base_argv()
    tensor_names = ["context", "query_proj", "key_proj", "value_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"]
    policy = RematPolicy(tensor_names=tensor_names, initial_level=Action.REMAT)

    jax.clear_caches()
    result = is_oom(base_argv, policy, pdb=2.0)

    self.assertIsInstance(result, bool)
    # A small model with full remat and small batch should NOT OOM
    self.assertFalse(result, "Small model with full remat and pdb=2 should not OOM")

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_search_policy_only_small_model(self):
    """E2E: search_policy_only returns a valid policy for a small model."""
    base_argv = self._make_base_argv()
    tensor_names = ["context", "query_proj", "key_proj", "value_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"]

    jax.clear_caches()
    result = search_policy_only(tensor_names, base_argv, pdb=2.0)

    # Should return a RematPolicy
    self.assertIsInstance(result, RematPolicy)

    # The result's to_dict should have all expected tensor names
    result_dict = result.to_dict
    for name in tensor_names:
      self.assertIn(name, result_dict)
      self.assertIn(result_dict[name], ("remat", "offload", "device"))

    print(f"Estimator found policy: {result_dict}")
