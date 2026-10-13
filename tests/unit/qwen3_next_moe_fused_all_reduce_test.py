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

"""Tests for folding the Qwen3-Next shared expert into the routed-expert all-reduce.

`Qwen3NextSparseMoeBlock` can hand the shared expert's *un-reduced* output
projection to `RoutedMoE` as `partial_output_to_add`, so that tpu_inference's
fused MoE reduces `routed_partial + sigmoid(gate) * shared_partial` with a
single all-reduce. The fused kernel itself is TPU-only; here the routed block
is replaced by a pure-JAX stand-in that reduces exactly what it is handed, so
the test checks the MaxText side of the contract on CPU with
`XLA_FLAGS=--xla_force_host_platform_device_count=N`.
"""

import os
import sys
import unittest
from unittest.mock import patch

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _make_config(num_devices, **overrides):
  config_arguments = {
      "run_name": "test",
      "enable_checkpointing": False,
      "per_device_batch_size": 1.0,
      "max_target_length": 8,
      "decoder_block": "qwen3_next",
      "base_emb_dim": 64,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 16,
      "base_num_decoder_layers": 2,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "dropout_rate": 0.0,
      "ici_tensor_parallelism": num_devices,
  }
  config_arguments.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **config_arguments)


class FusedSharedExpertAllReduceTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_devices = jax.device_count()
    if self.num_devices < 2:
      self.skipTest("needs >= 2 devices (set XLA_FLAGS=--xla_force_host_platform_device_count=4)")

  def _build(self, cfg):
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
    ctx = nn_partitioning.axis_rules(cfg.logical_axis_rules)
    ctx.__enter__()
    self.addCleanup(ctx.__exit__, None, None, None)
    mesh.__enter__()
    self.addCleanup(mesh.__exit__, None, None, None)
    block = qwen3.Qwen3NextSparseMoeBlock(config=cfg, mesh=mesh, quant=None, rngs=nnx.Rngs(params=0, dropout=1))
    return mesh, block

  def test_disabled_off_the_fused_path(self):
    """Without the vLLM fused MoE path the block keeps its separate-reduction code path."""
    cfg = _make_config(self.num_devices)
    _, block = self._build(cfg)
    self.assertFalse(block.routed_experts.uses_fused_moe_path())
    self.assertFalse(block._fuse_shared_expert_all_reduce())  # pylint: disable=protected-access

  def test_disabled_by_config_flag(self):
    cfg = _make_config(self.num_devices, fuse_shared_expert_all_reduce=False)
    _, block = self._build(cfg)
    with patch.object(moe.RoutedMoE, "fused_moe_partial_output_supported", return_value=True), patch.object(
        moe.RoutedMoE, "fused_moe_reduction_axes", return_value=("tensor",)
    ):
      self.assertFalse(block._fuse_shared_expert_all_reduce())  # pylint: disable=protected-access

  def test_disabled_when_reduction_groups_differ(self):
    cfg = _make_config(self.num_devices)
    _, block = self._build(cfg)
    with patch.object(moe.RoutedMoE, "fused_moe_partial_output_supported", return_value=True), patch.object(
        moe.RoutedMoE, "fused_moe_reduction_axes", return_value=("data",)
    ):
      self.assertFalse(block._fuse_shared_expert_all_reduce())  # pylint: disable=protected-access

  def test_fused_path_reduces_gated_shared_partial_once(self):
    """`routed(partial_output_to_add=g*shared_partial)` reproduces `routed + g*shared`.

    The routed block is replaced by a stand-in that returns
    `allreduce(partial_output_to_add)` (zero routed contribution) over the
    reduction axes it advertises -- which is exactly what the fused
    tpu_inference kernel does with the argument. The block output must then
    equal `sigmoid(gate) * shared_expert(x)` computed the regular way.
    """
    cfg = _make_config(self.num_devices)
    mesh, block = self._build(cfg)
    axes = block.shared_expert.unreduced_output_axes()
    self.assertEqual(set(axes), {"tensor"})

    calls = []

    def fake_routed_call(self_routed, inputs, out_sharding=None, forced_routed_experts=None, partial_output_to_add=None, **_):
      calls.append(partial_output_to_add is not None)
      if partial_output_to_add is None:
        return jnp.zeros_like(inputs), None, None
      reduced = jax.shard_map(
          lambda p: jax.lax.psum(p, axes),
          mesh=mesh,
          in_specs=P(),
          out_specs=P(),
          check_vma=False,
      )(partial_output_to_add)
      return reduced.astype(inputs.dtype), None, None

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 8, cfg.emb_dim), dtype=jnp.float32)

    def expected_fn(m, h):
      shared = m.shared_expert(h, deterministic=True)
      gate = jax.nn.sigmoid(m._shared_expert_gate_logits(h).astype(jnp.float32))  # pylint: disable=protected-access
      return gate * shared.astype(jnp.float32)

    def fused_fn(m, h):
      out, _ = m(h, deterministic=True)
      return out

    with patch.object(moe.RoutedMoE, "__call__", fake_routed_call), patch.object(
        moe.RoutedMoE, "fused_moe_partial_output_supported", return_value=True
    ), patch.object(moe.RoutedMoE, "fused_moe_reduction_axes", return_value=("tensor",)):
      self.assertTrue(block._fuse_shared_expert_all_reduce())  # pylint: disable=protected-access
      fused = nnx.jit(fused_fn)(block, x)
      expected = nnx.jit(expected_fn)(block, x)

    self.assertEqual(calls, [True])
    np.testing.assert_allclose(np.asarray(fused), np.asarray(expected), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
