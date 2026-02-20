# Copyright 2023–2026 Google LLC
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

"""Test for DeepSeek Manifold-Constrained Hyper Connections (mHC)."""

import os.path
import unittest
import pytest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from maxtext.common.common_types import HyperConnectionType
from maxtext.layers import attention_mla, linears, mhc, moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import maxtext_utils


class TestExpandReduce(unittest.TestCase):
  """Unit tests for MHC dimension expansion and reduction operations."""

  def setUp(self):
    self.rate = 4
    self.batch, self.seq_len, self.dim = 2, 8, 12
    self.shape = (self.batch, self.seq_len, self.dim)
    self.expand, self.reduce = mhc.get_functions(self.rate)

    # Consistent random data for testing
    self.key = jax.random.PRNGKey(0)
    self.x = jax.random.normal(self.key, self.shape)

  def test_expand_shape(self):
    """Verifies (B, S, D) -> (B, S, K, D)"""
    out = self.expand(self.x)
    expected_shape = (self.batch, self.seq_len, self.rate, self.dim)
    self.assertEqual(out.shape, expected_shape)

  def test_reduce_shape(self):
    """Verifies (B, S, K, D) -> (B, S, D)"""
    dummy_expanded = jnp.ones((self.batch, self.seq_len, self.rate, self.dim))
    out = self.reduce(dummy_expanded)
    self.assertEqual(out.shape, self.shape)

  def test_value_identity(self):
    """Mathematically, reduce(expand(x)) should equal expansion_rate * x."""
    out = self.reduce(self.expand(self.x))
    expected = self.x * self.rate
    np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestSinkhorn(unittest.TestCase):
  """Unit tests for MHC Sinkhorn Algorithm."""

  def setUp(self):
    self.key = jax.random.PRNGKey(42)
    self.matrix_shape = (8, 8)
    self.t = jax.random.normal(self.key, self.matrix_shape)

  def test_doubly_stochastic_property(self):
    """After many iterations, rows and columns should sum to approximately 1."""
    # Use more iterations to ensure convergence
    out = mhc.sinkhorn(self.t, iters=20)

    row_sums = jnp.sum(out, axis=-1)
    col_sums = jnp.sum(out, axis=-2)

    # Check if sums are close to 1.0
    np.testing.assert_allclose(row_sums, jnp.ones_like(row_sums), atol=1e-3)
    np.testing.assert_allclose(col_sums, jnp.ones_like(col_sums), atol=1e-3)


class TestMHC(unittest.TestCase):
  """Test for MHC module"""

  def setUp(self):
    self.dim = 16
    self.config = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        run_name="test_mhc",
        enable_checkpointing=False,
        model_name="deepseek-custom",
        per_device_batch_size=4,
        max_target_length=7,
        max_prefill_predict_length=7,
        base_emb_dim=self.dim,
        mhc_expansion_rate=3,
        num_experts=4,
        num_experts_per_tok=2,
        attention="dot_product",
        routed_bias_update_rate=0.01,
        load_balance_loss_weight=0.02,
    )
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)

    self.rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(42))
    self.x = jax.random.normal(
        jax.random.PRNGKey(0),
        (
            self.config.per_device_batch_size,
            self.config.max_target_length,
            self.config.mhc_expansion_rate,
            self.config.emb_dim,
        ),
    )

    self.pre_norm = RMSNorm(
        num_features=self.dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

  # Skip GPU due to NotImplementedError: dynamic grid bounds not supported in the Triton backend
  @pytest.mark.tpu_only
  def test_moe_layer_output_shape(self):
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)
      layer = moe.RoutedMoE(
          config=self.config,
          num_experts=self.config.num_experts,
          num_experts_per_tok=self.config.num_experts_per_tok,
          mesh=self.mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=self.config.base_mlp_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          rngs=self.rngs,
      )

      b, s, k, d = self.x.shape
      output, metadata = module(self.pre_norm, layer, x=self.x, mhc_type=HyperConnectionType.MLP_MOE)
      # metadata includes load_balance_loss & moe_bias_updates
      self.assertEqual(len(metadata), 2)
      for key, value in metadata.items():
        self.assertIsNotNone(value, f"Key '{key}' has a value of None")
      self.assertEqual(output.shape, (b, s, k, d))

  def test_dense_layer_output_shape(self):
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)
      layer = linears.MlpBlock(
          config=self.config,
          mesh=self.mesh,
          in_features=self.config.emb_dim,
          intermediate_dim=self.config.moe_mlp_dim,
          activations=self.config.mlp_activations,
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          model_mode=self.config.model_call_mode,
          rngs=self.rngs,
      )

      b, s, k, d = self.x.shape
      output, metadata = module(self.pre_norm, layer, x=self.x, mhc_type=HyperConnectionType.MLP_DENSE)
      self.assertDictEqual(metadata, {})
      self.assertEqual(output.shape, (b, s, k, d))

  def test_attention_layer_output_shape(self):
    inputs_shape = (self.config.per_device_batch_size, self.config.max_target_length, self.config.emb_dim)
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)
      layer = attention_mla.MLA(
          config=self.config,
          num_query_heads=self.config.num_query_heads,
          num_kv_heads=self.config.num_kv_heads,
          head_dim=self.config.head_dim,
          max_target_length=self.config.max_target_length,
          max_prefill_predict_length=self.config.max_prefill_predict_length,
          attention_kernel=self.config.attention,
          attention_type=self.config.attention_type,
          inputs_q_shape=inputs_shape,
          inputs_kv_shape=inputs_shape,
          mesh=self.mesh,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          dropout_rate=self.config.dropout_rate,
          name="self_attention",
          q_lora_rank=self.config.q_lora_rank,
          kv_lora_rank=self.config.kv_lora_rank,
          qk_nope_head_dim=self.config.qk_nope_head_dim,
          qk_rope_head_dim=self.config.qk_rope_head_dim,
          v_head_dim=self.config.v_head_dim,
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          mscale=self.config.mscale,
          rope_factor=self.config.rope_factor,
          model_mode="train",
          rngs=self.rngs,
          attn_logits_soft_cap=self.config.attn_logits_soft_cap,
      )

      b, s, k, d = self.x.shape
      output, metadata = module(self.pre_norm, layer, x=self.x, mhc_type=HyperConnectionType.ATTENTION)
      self.assertDictEqual(metadata, {})
      self.assertEqual(output.shape, (b, s, k, d))


if __name__ == "__main__":
  unittest.main()
