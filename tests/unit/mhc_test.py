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

import unittest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.common.common_types import HyperConnectionType
from maxtext.layers import attention_mla, linears, mhc, moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


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


class TestMHC(parameterized.TestCase):
  """Test for MHC module"""

  def _setup_mhc(self, rate, enable_mhc_lite=False):
    """Sets up the common configurations and modules for MHC testing."""
    self.dim = 16
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"test_mhc_k{rate}",
        enable_checkpointing=False,
        model_name="deepseek-custom",
        per_device_batch_size=jax.device_count(),
        max_target_length=7,
        max_prefill_predict_length=7,
        attention="dot_product",
        routed_bias_update_rate=0.01,
        load_balance_loss_weight=0.02,
        # override
        override_model_config=True,
        base_emb_dim=self.dim,
        mhc_expansion_rate=rate,
        enable_mhc_lite=enable_mhc_lite,
        num_experts=4,
        num_experts_per_tok=2,
        engram_layers=[],
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
  @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def test_moe_layer_output_shape(self, rate):
    self._setup_mhc(rate)

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
      self.assertLen(metadata, 2)
      for key, value in metadata.items():
        self.assertIsNotNone(value, f"Key '{key}' has a value of None")
      self.assertEqual(output.shape, (b, s, k, d))

  @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def test_dense_layer_output_shape(self, rate):
    self._setup_mhc(rate)
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

  @parameterized.named_parameters(("Rate3", 3), ("Rate4", 4))
  def test_attention_layer_output_shape(self, rate):
    self._setup_mhc(rate)
    inputs_shape = (
        self.config.per_device_batch_size,
        self.config.max_target_length,
        self.config.emb_dim,
    )
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

  def test_mhc_lite_doubly_stochastic(self):
    """Verify that mHC-lite output is doubly stochastic (rows/cols sum to 1)."""
    self._setup_mhc(4, enable_mhc_lite=True)
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)

      b, s, k, d = self.x.shape

      # Generate random input X
      random_x = jax.random.normal(jax.random.PRNGKey(42), (b, s, k * d))
      norm_x = module.mhc_norm(random_x)

      # Output from mHC-lite mapping
      res_mapping_out = module.res_mapping(norm_x)

      row_sums = jnp.sum(res_mapping_out, axis=-1)
      col_sums = jnp.sum(res_mapping_out, axis=-2)

      # Check if sums are close to 1.0
      np.testing.assert_allclose(row_sums, jnp.ones_like(row_sums), atol=1e-2)
      np.testing.assert_allclose(col_sums, jnp.ones_like(col_sums), atol=1e-2)

  def test_feature_flag_gates_lite(self):
    """Verify that setting enable_mhc_lite=False falls back to Sinkhorn."""
    self.dim = 16
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_mhc_lite_gated",
        enable_checkpointing=False,
        model_name="deepseek-custom",
        per_device_batch_size=4,
        max_target_length=7,
        max_prefill_predict_length=7,
        attention="dot_product",
        routed_bias_update_rate=0.01,
        load_balance_loss_weight=0.02,
        # override
        override_model_config=True,
        base_emb_dim=self.dim,
        mhc_expansion_rate=4,
        enable_mhc_lite=False,
        num_experts=4,
        num_experts_per_tok=2,
        engram_layers=[],
    )
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    self.rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(42))

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)

      # Shape of res_alpha should be (4*16, 4*4) = (64, 16) instead of (64, 24)
      self.assertEqual(module.res_alpha.shape, (64, 16))
      # Shape of res_beta should be (4, 4) instead of (24,)
      self.assertEqual(module.res_beta.shape, (4, 4))
      # Permutation matrices shouldn't be defined
      self.assertFalse(hasattr(module, "permutation_matrices"))


if __name__ == "__main__":
  unittest.main()
