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

from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.utils.estimator import (
    Action,
    RematPolicy,
    largest_batch_size,
    generate_priority_list,
    find_pdb_scalar,
    build_argv,
    get_parameter_value,
    find_batch_size,
    find_remat_policy_tensor_names,
    generate_remat_config,
    generate_pdb_config,
    search,
    search_policy_only,
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
  def test_previous_policy_terminal_state(self):
    """Verify previous_policy returns None when no more downgrades are possible."""
    policy = RematPolicy(self.tensor_names, initial_level=Action.REMAT)
    self.assertIsNone(policy.previous_policy())

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
  @patch("maxtext.utils.estimator.is_oom")
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
  @patch("maxtext.utils.estimator.is_oom")
  def test_largest_batch_size_min_equals_max_no_oom(self, mock_is_oom):
    """When min_pdb == max_pdb and it does NOT OOM, return that batch size."""
    mock_is_oom.return_value = False
    policy = RematPolicy(self.tensor_names)
    result = largest_batch_size(self.base_argv, policy, min_pdb=4.0, max_pdb=4.0, pdb_scalar=1.0)
    self.assertEqual(result, 4.0)

  @pytest.mark.cpu_only
  @patch("maxtext.utils.estimator.is_oom")
  def test_largest_batch_size_min_equals_max_oom(self, mock_is_oom):
    """When min_pdb == max_pdb and it DOES OOM, return below min."""
    mock_is_oom.return_value = True
    policy = RematPolicy(self.tensor_names)
    result = largest_batch_size(self.base_argv, policy, min_pdb=4.0, max_pdb=4.0, pdb_scalar=1.0)
    self.assertEqual(result, 3.0)

  @pytest.mark.cpu_only
  def test_largest_batch_size_zero_scalar(self):
    """Verify ValueError is raised when pdb_scalar is zero."""
    policy = RematPolicy(self.tensor_names)
    with self.assertRaises(ValueError):
      largest_batch_size(self.base_argv, policy, min_pdb=1.0, max_pdb=8.0, pdb_scalar=0.0)

  @pytest.mark.cpu_only
  def test_largest_batch_size_invalid_range(self):
    """Verify ValueError when max_pdb < min_pdb."""
    policy = RematPolicy(self.tensor_names)
    with self.assertRaises(ValueError):
      largest_batch_size(self.base_argv, policy, min_pdb=8.0, max_pdb=4.0, pdb_scalar=1.0)

  @pytest.mark.cpu_only
  @patch("maxtext.utils.estimator.is_oom")
  def test_largest_batch_size_default_min_pdb(self, mock_is_oom):
    """Verify min_pdb defaults to 1/pdb_scalar when None."""
    mock_is_oom.return_value = False  # Never OOM
    policy = RematPolicy(self.tensor_names)
    result = largest_batch_size(self.base_argv, policy, min_pdb=None, max_pdb=8.0, pdb_scalar=4.0)
    # min_pdb defaults to 1/4 = 0.25; since no OOM at max, result should be 8.0
    self.assertEqual(result, 8.0)

  @pytest.mark.cpu_only
  def test_build_argv_default_layer_input(self):
    """Ensure decoder_layer_input=device is added if not present."""
    policy = RematPolicy(["context"], initial_level=Action.REMAT)
    argv = build_argv(self.base_argv[1:], policy, pdb=2)

    self.assertIn("decoder_layer_input=device", argv)
    self.assertIn("per_device_batch_size=2", argv)
    self.assertIn("context=remat", argv)

  @pytest.mark.cpu_only
  def test_build_argv_with_existing_layer_input(self):
    """Ensure decoder_layer_input is NOT added when already present."""
    base = self.base_argv[1:] + ("decoder_layer_input=offload",)
    policy = RematPolicy(["context"], initial_level=Action.REMAT)
    argv = build_argv(base, policy, pdb=2)

    # Should have exactly one decoder_layer_input
    count = sum(1 for a in argv if "decoder_layer_input" in a)
    self.assertEqual(count, 1)
    self.assertIn("decoder_layer_input=offload", argv)

  @pytest.mark.cpu_only
  def test_build_argv_with_dict_policy(self):
    """Ensure build_argv works with a dict policy (as stored in Mode 2 results)."""
    policy_dict = {"context": "remat", "mlpwo": "offload"}
    argv = build_argv(self.base_argv[1:], policy_dict, pdb=4)

    self.assertIn("remat_policy=custom", argv)
    self.assertIn("context=remat", argv)
    self.assertIn("mlpwo=offload", argv)
    self.assertIn("per_device_batch_size=4", argv)

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

  @pytest.mark.cpu_only
  def test_get_parameter_value_found(self):
    """Verify get_parameter_value finds a matching prefix."""
    config = ("foo=bar", "per_device_batch_size=4", "baz=qux")
    found, value = get_parameter_value(config, "per_device_batch_size=")
    self.assertTrue(found)
    self.assertEqual(value, "4")

  @pytest.mark.cpu_only
  def test_get_parameter_value_not_found(self):
    """Verify get_parameter_value returns (False, None) when not found."""
    config = ("foo=bar", "baz=qux")
    found, value = get_parameter_value(config, "per_device_batch_size=")
    self.assertFalse(found)
    self.assertIsNone(value)

  @pytest.mark.cpu_only
  def test_find_batch_size_provided(self):
    """Verify find_batch_size returns the batch size when present."""
    argv = ("base.yml", "per_device_batch_size=2.5", "model_name=llama")
    provided, pdb = find_batch_size(argv)
    self.assertTrue(provided)
    self.assertEqual(pdb, 2.5)

  @pytest.mark.cpu_only
  def test_find_batch_size_not_provided(self):
    """Verify find_batch_size returns (False, None) when not present."""
    argv = ("base.yml", "model_name=llama")
    provided, pdb = find_batch_size(argv)
    self.assertFalse(provided)
    self.assertIsNone(pdb)

  @pytest.mark.cpu_only
  def test_find_remat_policy_tensor_names(self):
    """Verify tensor names are detected from argv."""
    argv = ("base.yml", "context=offload", "mlpwo=remat", "model_name=llama")
    result = find_remat_policy_tensor_names(argv)
    self.assertIn("context", result)
    self.assertIn("mlpwo", result)
    self.assertNotIn("query_proj", result)

  @pytest.mark.cpu_only
  def test_find_remat_policy_tensor_names_none(self):
    """Verify empty list when no tensor names in argv."""
    argv = ("base.yml", "model_name=llama")
    result = find_remat_policy_tensor_names(argv)
    self.assertEqual(result, [])

  @pytest.mark.cpu_only
  def test_generate_remat_config_from_policy(self):
    """Verify generate_remat_config works with a RematPolicy object."""
    policy = RematPolicy(["context", "mlpwo"], initial_level=Action.REMAT)
    config = generate_remat_config(policy)
    self.assertIn("remat_policy=custom", config)
    self.assertIn("context=remat", config)
    self.assertIn("mlpwo=remat", config)

  @pytest.mark.cpu_only
  def test_generate_remat_config_from_dict(self):
    """Verify generate_remat_config works with a dict."""
    policy_dict = {"context": "offload", "mlpwo": "device"}
    config = generate_remat_config(policy_dict)
    self.assertIn("remat_policy=custom", config)
    self.assertIn("context=offload", config)
    self.assertIn("mlpwo=device", config)

  @pytest.mark.cpu_only
  def test_generate_pdb_config(self):
    """Verify generate_pdb_config returns the expected tuple."""
    result = generate_pdb_config(4.0)
    self.assertEqual(result, ("per_device_batch_size=4.0",))

  @pytest.mark.cpu_only
  @patch("maxtext.utils.estimator.is_oom")
  def test_search_policy_only_finds_fitting_policy(self, mock_is_oom):
    """Verify search_policy_only returns the first non-OOM policy."""
    tensor_names = ["context", "mlpwo"]

    # Simulate: full device OOMs, one step back (previous_policy) fits
    call_count = [0]

    def oom_side_effect(argv, policy, pdb):
      call_count[0] += 1
      if call_count[0] == 1:
        # First call: sanity check with full remat -> does NOT OOM
        return False
      if call_count[0] == 2:
        # Second call: full device policy -> OOMs
        return True
      if call_count[0] == 3:
        # Third call: one step back -> OOMs
        return True
      # Fourth call onwards: fits
      return False

    mock_is_oom.side_effect = oom_side_effect

    result = search_policy_only(tensor_names, self.base_argv, pdb=4.0)
    # Result should be a RematPolicy that fits
    self.assertIsInstance(result, RematPolicy)

  @pytest.mark.cpu_only
  @patch("maxtext.utils.estimator.is_oom")
  def test_search_policy_only_raises_on_impossible_batch(self, mock_is_oom):
    """Verify search_policy_only raises ValueError when full remat OOMs."""
    mock_is_oom.return_value = True  # Everything OOMs

    with self.assertRaises(ValueError):
      search_policy_only(self.tensor_names, self.base_argv, pdb=999.0)

  @pytest.mark.cpu_only
  @patch("maxtext.utils.estimator.is_oom")
  def test_search_returns_pareto_frontier(self, mock_is_oom):
    """Verify search returns a list of (pdb, dict) tuples."""

    def oom_side_effect(argv, policy, pdb):
      # OOM at pdb >= 5
      return pdb >= 5.0

    mock_is_oom.side_effect = oom_side_effect

    results = search(
        self.tensor_names,
        self.base_argv,
        min_pdb=1.0,
        max_pdb=8.0,
        pdb_scalar=1.0,
    )
    # Should have at least one result
    self.assertGreater(len(results), 0)
    for pdb, policy_dict in results:
      self.assertIsInstance(pdb, float)
      self.assertIsInstance(policy_dict, dict)
      # pdb should be > 0 (valid)
      self.assertGreater(pdb, 0)

  @pytest.mark.cpu_only
  def test_next_previous_policy_full_traversal(self):
    """Verify we can traverse from full remat to full device and back."""
    names = ["a", "b"]
    policy = RematPolicy(names, initial_level=Action.REMAT)
    visited = [policy.to_dict]

    # Traverse forward
    current = policy
    while True:
      nxt = current.next_policy()
      if nxt is None:
        break
      visited.append(nxt.to_dict)
      current = nxt

    # Final state should be full device
    self.assertEqual(current.to_dict, {"a": "device", "b": "device"})

    # Traverse backward
    while True:
      prev = current.previous_policy()
      if prev is None:
        break
      current = prev

    # Should be back at full remat
    self.assertEqual(current.to_dict, {"a": "remat", "b": "remat"})
