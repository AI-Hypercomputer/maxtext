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
These tests verify the proper functioning of MaxText estimator.
"""

import unittest
import pytest
from unittest.mock import MagicMock, patch
import os

from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.estimator import (
    Action,
    RematPolicy,
    largest_batch_size,
    generate_priority_list,
    find_pdb_scalar,
    build_argv,
)


class TestRematEstimator(unittest.TestCase):
  """Tests for the MaxText estimator functions."""

  def setUp(self):
    """Set up standard test variables."""
    self.tensor_names = ["context", "mlpwo"]
    self.base_argv = (None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"))

  @pytest.mark.cpu_only
  def test_policy_to_dict_conversion(self):
    """Verify the Action enum correctly maps to MaxText string flags."""
    policy = RematPolicy(self.tensor_names, initial_level=Action.REMAT)
    d = policy.to_dict
    self.assertEqual(d["context"], "remat")

    policy.tensors["context"] = Action.DEVICE
    self.assertEqual(policy.to_dict["context"], "device")

  @pytest.mark.cpu_only
  def test_next_policy_logic(self):
    """
    Verify that next_policy moves through actions incrementally
    and respects the reversed priority order.
    """
    # Start at full remat
    policy = RematPolicy(self.tensor_names, initial_level=Action.REMAT)

    # mlpwo should be updated
    next_p = policy.next_policy()
    self.assertIsNotNone(next_p)
    self.assertEqual(next_p.tensors["mlpwo"], Action.OFFLOAD)
    self.assertEqual(next_p.tensors["context"], Action.REMAT)

    # previous_policy should reverse the result from next_policy
    same_p = next_p.previous_policy()
    self.assertEqual(same_p.tensors["mlpwo"], Action.REMAT)

  @pytest.mark.cpu_only
  def test_policy_terminal_state(self):
    """Verify next_policy returns None when no more upgrades are possible."""
    policy = RematPolicy(self.tensor_names, initial_level=Action.DEVICE)
    self.assertIsNone(policy.next_policy())

  @pytest.mark.cpu_only
  def test_find_pdb_scalar(self):
    """Verify pdb_scalar ignores data/fsdp axes and multiplies parallelism axes."""
    mock_config = MagicMock()
    mock_config.mesh_axes = ["data", "tensor", "expert"]
    mock_config.ici_tensor_parallelism = 4
    mock_config.ici_expert_parallelism = 8  # Should be ignored per your logic

    scalar = find_pdb_scalar(mock_config)
    # Per your code: if mesh_axis not in ("data", "fsdp", ... "expert")
    # Only "tensor" is left -> scalar should be 4.0
    self.assertEqual(scalar, 4.0)

  @pytest.mark.cpu_only
  @patch("MaxText.estimator.is_oom")
  def test_largest_batch_size_binary_search(self, mock_is_oom):
    """
    Simulate OOM checks to ensure binary search finds the exact boundary.
    Scenario: Fits at pdb=4, OOMs at pdb=5.
    """

    def oom_side_effect(argv, policy, pdb):
      return pdb >= 5.0  # OOM if batch size is 5 or higher

    mock_is_oom.side_effect = oom_side_effect

    policy = RematPolicy(self.tensor_names)
    # Search between 1 and 8
    result = largest_batch_size(self.base_argv, policy, min_pdb=1.0, max_pdb=8.0, pdb_scalar=1.0)
    self.assertEqual(result, 4.0)

  @pytest.mark.cpu_only
  def test_build_argv_default_layer_input(self):
    """Ensure decoder_layer_input=device is added if not present."""
    policy = RematPolicy(["context"], initial_level=Action.REMAT)
    argv = build_argv(self.base_argv[1:], policy, pdb=2)

    self.assertIn("decoder_layer_input=device", argv)
    self.assertIn("per_device_batch_size=2", argv)
    self.assertIn("context=remat", argv)

  @pytest.mark.cpu_only
  def test_priority_list_generation(self):
    """Test that generate_priority_list pulls the correct key from config."""
    mock_config = MagicMock()
    mock_config.fused_mlp = True
    mock_config.mlp_activations = [1, 2]  # Length 2
    mock_config.emb_dim = 1024
    mock_config.mlp_dim = 4096
    mock_config.head_dim = 64
    mock_config.num_query_heads = 16
    mock_config.num_kv_heads = 16
    mock_config.max_target_length = 2048

    # Tensors for (True, 2): ["context", "qkv_proj", "mlpwi_0", "mlpwi_1", "mlpwo", "out_proj"]
    plist = generate_priority_list(mock_config, provided_tensor_names=["context"])

    self.assertNotIn("context", plist)
    self.assertIn("mlpwi_0", plist)
    self.assertIn("mlpwi_1", plist)
