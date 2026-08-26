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

import dataclasses
import itertools
import math
import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import HyperConnectionType
from maxtext.configs import pyconfig
from maxtext.kernels import mhc as mhc_kernel
from maxtext.kernels.mhc import common as mhc_kernel_common
from maxtext.layers import attention_mla, linears, mhc, moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import pytest


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

  def _setup_mhc(
      self,
      rate,
      enable_mhc_lite=False,
      use_mhc_pallas_kernel=False,
      mhc_pallas_kernel_fwd_block_size=None,
      mhc_pallas_kernel_bwd_block_size=None,
      dim=16,
      sequence_length=7,
      per_device_batch_size=None,
      dtype=None,
  ):
    """Sets up the common configurations and modules for MHC testing."""
    self.dim = dim
    if per_device_batch_size is None:
      per_device_batch_size = jax.device_count()
    kwargs = {
        "run_name": f"test_mhc_k{rate}",
        "enable_checkpointing": False,
        "model_name": "deepseek-custom",
        "per_device_batch_size": per_device_batch_size,
        "max_target_length": sequence_length,
        "max_prefill_predict_length": sequence_length,
        "attention": "dot_product",
        "attention_type": "mla",
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "load_balance_loss_weight": 0.02,
        # override
        "override_model_config": True,
        "base_emb_dim": self.dim,
        "mhc_expansion_rate": rate,
        "enable_mhc_lite": enable_mhc_lite,
        "use_mhc_pallas_kernel": use_mhc_pallas_kernel,
        "decoder_block": "deepseek",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "base_moe_mlp_dim": self.dim * 4,
        "base_mlp_dim": self.dim * 4,
        "engram_layers": [],
    }
    if mhc_pallas_kernel_fwd_block_size is not None:
      kwargs["mhc_pallas_kernel_fwd_block_size"] = mhc_pallas_kernel_fwd_block_size
    if mhc_pallas_kernel_bwd_block_size is not None:
      kwargs["mhc_pallas_kernel_bwd_block_size"] = mhc_pallas_kernel_bwd_block_size
    if dtype is not None:
      kwargs["dtype"] = dtype
      kwargs["weight_dtype"] = dtype
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        **kwargs,
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
        dtype=self.config.dtype,
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
      positions = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))
      output, metadata = module(
          self.pre_norm,
          layer,
          x=self.x,
          mhc_type=HyperConnectionType.ATTENTION,
          inputs_positions=positions,
      )
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

      # Output from mHC-lite mapping (using post_matmul API)
      res_alpha = jnp.asarray(module.res_alpha[...], module.dtype)
      h_res = jnp.einsum("bsm,mn -> bsn", norm_x, res_alpha, precision=module.matmul_precision)
      res_mapping_out = module.res_mapping(h_res)

      row_sums = jnp.sum(res_mapping_out, axis=-1)
      col_sums = jnp.sum(res_mapping_out, axis=-2)

      # Check if sums are close to 1.0
      np.testing.assert_allclose(row_sums, jnp.ones_like(row_sums), atol=1e-2)
      np.testing.assert_allclose(col_sums, jnp.ones_like(col_sums), atol=1e-2)

  def test_mhc_lite_sharding_eval_shape(self):
    """Verify that enable_mhc_lite=True works with nnx.eval_shape and nnx_construct_named_sharding."""
    self._setup_mhc(4, enable_mhc_lite=True)
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):

      def _create_model():
        rngs = nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(42))
        return mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, rngs)

      abs_model = nnx.eval_shape(_create_model)
      _, abs_var_state = nnx.split(abs_model)
      named_sharding_state = sharding.nnx_construct_named_sharding(abs_var_state, self.mesh)
      self.assertIsNotNone(named_sharding_state)
      self.assertFalse(hasattr(abs_model, "permutation_matrices"))

  def test_weight_concatenation_equivalence(self):
    """Verify that fused projection matches sequential projections."""
    self._setup_mhc(4)
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)

      b, s, k, d = self.x.shape
      x_flat = jnp.reshape(self.x, (b, s, k * d))
      norm_x = module.mhc_norm(x_flat)

      # Sequential Projections (Old way)
      pre_alpha = jnp.asarray(module.pre_alpha[...], module.dtype)
      post_alpha = jnp.asarray(module.post_alpha[...], module.dtype)
      res_alpha = jnp.asarray(module.res_alpha[...], module.dtype)

      h_pre_seq = jnp.einsum("bsm,mk -> bsk", norm_x, pre_alpha, precision=module.matmul_precision)
      h_post_seq = jnp.einsum("bsm,mk -> bsk", norm_x, post_alpha, precision=module.matmul_precision)
      h_res_seq = jnp.einsum("bsm,mn -> bsn", norm_x, res_alpha, precision=module.matmul_precision)

      # Fused Projection (New way)
      alpha_concat = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1)
      h_concat = jnp.einsum("bsm,mn -> bsn", norm_x, alpha_concat, precision=module.matmul_precision)

      h_pre_fused = h_concat[..., : module.k]
      h_post_fused = h_concat[..., module.k : 2 * module.k]
      h_res_fused = h_concat[..., 2 * module.k :]

      # Verify equivalence
      np.testing.assert_allclose(h_pre_seq, h_pre_fused, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(h_post_seq, h_post_fused, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(h_res_seq, h_res_fused, rtol=1e-5, atol=1e-5)

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
        attention_type="mla",
        routed_bias=True,
        routed_bias_update_rate=0.01,
        load_balance_loss_weight=0.02,
        # override
        override_model_config=True,
        base_emb_dim=self.dim,
        mhc_expansion_rate=4,
        enable_mhc_lite=False,
        decoder_block="deepseek",
        num_experts=4,
        num_experts_per_tok=2,
        base_moe_mlp_dim=self.dim * 4,
        base_mlp_dim=self.dim * 4,
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

  @parameterized.named_parameters(
      ("KernelEnabled", True),
      ("KernelDisabled", False),
  )
  def test_use_mhc_pallas_kernel_dispatch(self, use_mhc_pallas_kernel):
    """Verify that use_mhc_pallas_kernel flag controls kernel vs pure JAX dispatch."""
    self._setup_mhc(
        4,
        enable_mhc_lite=True,
        use_mhc_pallas_kernel=use_mhc_pallas_kernel,
        dim=128,
        sequence_length=256,
        per_device_batch_size=1,
        dtype="bfloat16",
    )
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

      real_pre = mhc.mhc_kernel.pre
      real_post = mhc.mhc_kernel.post

      def fake_pre(*args, **kwargs):
        config = kwargs.get("config", mhc.mhc_kernel.MhcKernelConfig())
        config = dataclasses.replace(config, interpret=True)
        kwargs["config"] = config
        return real_pre(*args, **kwargs)

      def fake_post(*args, **kwargs):
        config = kwargs.get("config", mhc.mhc_kernel.MhcKernelConfig())
        config = dataclasses.replace(config, interpret=True)
        kwargs["config"] = config
        return real_post(*args, **kwargs)

      with (
          mock.patch.object(mhc.mhc_kernel, "pre", side_effect=fake_pre) as mock_pre,
          mock.patch.object(mhc.mhc_kernel, "post", side_effect=fake_post) as mock_post,
      ):
        output, _ = module(
            self.pre_norm,
            layer,
            x=self.x,
            mhc_type=HyperConnectionType.MLP_DENSE,
        )

        if use_mhc_pallas_kernel:
          mock_pre.assert_called_once()
          mock_post.assert_called_once()
          _, kwargs_pre = mock_pre.call_args
          config_pre = kwargs_pre.get("config")
          self.assertIsNotNone(config_pre)
          self.assertEqual(config_pre.block_size, 256)
          self.assertEqual(config_pre.bwd_block_size, 128)

          _, kwargs_post = mock_post.call_args
          config_post = kwargs_post.get("config")
          self.assertIsNotNone(config_post)
          self.assertEqual(config_post.block_size, 256)
          self.assertEqual(config_post.bwd_block_size, 128)
        else:
          mock_pre.assert_not_called()
          mock_post.assert_not_called()

        self.assertEqual(output.shape, self.x.shape)

  def test_use_mhc_pallas_kernel_custom_block_size(self):
    """Verify that custom block sizes are passed to the kernel."""
    self._setup_mhc(
        4,
        enable_mhc_lite=True,
        use_mhc_pallas_kernel=True,
        mhc_pallas_kernel_fwd_block_size=128,
        mhc_pallas_kernel_bwd_block_size=64,
        dim=128,
        sequence_length=128,
        per_device_batch_size=1,
        dtype="bfloat16",
    )
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

      real_pre = mhc.mhc_kernel.pre
      real_post = mhc.mhc_kernel.post

      def fake_pre(*args, **kwargs):
        config = kwargs.get("config", mhc.mhc_kernel.MhcKernelConfig())
        config = dataclasses.replace(config, interpret=True)
        kwargs["config"] = config
        return real_pre(*args, **kwargs)

      def fake_post(*args, **kwargs):
        config = kwargs.get("config", mhc.mhc_kernel.MhcKernelConfig())
        config = dataclasses.replace(config, interpret=True)
        kwargs["config"] = config
        return real_post(*args, **kwargs)

      with (
          mock.patch.object(mhc.mhc_kernel, "pre", side_effect=fake_pre) as mock_pre,
          mock.patch.object(mhc.mhc_kernel, "post", side_effect=fake_post) as mock_post,
      ):
        output, _ = module(
            self.pre_norm,
            layer,
            x=self.x,
            mhc_type=HyperConnectionType.MLP_DENSE,
        )

        mock_pre.assert_called_once()
        mock_post.assert_called_once()
        _, kwargs_pre = mock_pre.call_args
        config_pre = kwargs_pre.get("config")
        self.assertIsNotNone(config_pre)
        self.assertEqual(config_pre.block_size, 128)
        self.assertEqual(config_pre.bwd_block_size, 64)

        _, kwargs_post = mock_post.call_args
        config_post = kwargs_post.get("config")
        self.assertIsNotNone(config_post)
        self.assertEqual(config_post.block_size, 128)
        self.assertEqual(config_post.bwd_block_size, 64)

        self.assertEqual(output.shape, self.x.shape)

  def test_use_mhc_pallas_kernel_requires_enable_mhc_lite(self):
    """Verify that use_mhc_pallas_kernel=True requires enable_mhc_lite=True."""
    # Test via pyconfig initialization
    with self.assertRaises(ValueError):
      self._setup_mhc(
          4,
          enable_mhc_lite=False,
          use_mhc_pallas_kernel=True,
      )

    # Test via direct layer initialization with mock config
    self._setup_mhc(4)
    mock_config = mock.MagicMock()
    mock_config.use_mhc_pallas_kernel = True
    mock_config.enable_mhc_lite = False
    mock_config.dtype = jnp.bfloat16
    mock_config.weight_dtype = jnp.bfloat16
    mock_config.matmul_precision = "default"
    mock_config.mhc_expansion_rate = 4
    with self.assertRaises(ValueError):
      mhc.ManifoldConstrainedHyperConnections(mock_config, 16, self.mesh, self.rngs)

  def test_layer_vjp_parity_kernel_vs_baseline(self):
    """Verify that ManifoldConstrainedHyperConnections with kernel matches baseline under VJP."""
    self._setup_mhc(
        4,
        enable_mhc_lite=True,
        use_mhc_pallas_kernel=False,
        dim=128,
        sequence_length=128,
        per_device_batch_size=1,
        dtype="bfloat16",
    )
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      module_baseline = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)

      def layer_fn(inputs):
        return inputs * 2.0

      def forward_baseline(x):
        out, _ = module_baseline(self.pre_norm, layer_fn, x=x, mhc_type=HyperConnectionType.MLP_DENSE)
        return out

      out_base, vjp_base = jax.vjp(forward_baseline, self.x)
      cotangent = jax.random.normal(jax.random.PRNGKey(123), self.x.shape, dtype=self.x.dtype)
      (dx_base,) = vjp_base(cotangent)

      self._setup_mhc(
          4,
          enable_mhc_lite=True,
          use_mhc_pallas_kernel=True,
          mhc_pallas_kernel_fwd_block_size=128,
          mhc_pallas_kernel_bwd_block_size=64,
          dim=128,
          sequence_length=128,
          per_device_batch_size=1,
          dtype="bfloat16",
      )
      module_kernel = mhc.ManifoldConstrainedHyperConnections(self.config, self.dim, self.mesh, self.rngs)

      real_pre = mhc.mhc_kernel.pre
      real_post = mhc.mhc_kernel.post

      def fake_pre(*args, **kwargs):
        cfg = kwargs.get("config", mhc.mhc_kernel.MhcKernelConfig())
        cfg = dataclasses.replace(cfg, interpret=True)
        kwargs["config"] = cfg
        return real_pre(*args, **kwargs)

      def fake_post(*args, **kwargs):
        cfg = kwargs.get("config", mhc.mhc_kernel.MhcKernelConfig())
        cfg = dataclasses.replace(cfg, interpret=True)
        kwargs["config"] = cfg
        return real_post(*args, **kwargs)

      with (
          mock.patch.object(mhc.mhc_kernel, "pre", side_effect=fake_pre),
          mock.patch.object(mhc.mhc_kernel, "post", side_effect=fake_post),
      ):

        def forward_kernel(x):
          out, _ = module_kernel(self.pre_norm, layer_fn, x=x, mhc_type=HyperConnectionType.MLP_DENSE)
          return out

        out_kern, vjp_kern = jax.vjp(forward_kernel, self.x)
        (dx_kern,) = vjp_kern(cotangent)

      np.testing.assert_allclose(out_kern, out_base, rtol=5e-2, atol=5e-2)
      np.testing.assert_allclose(
          np.asarray(dx_kern, np.float32),
          np.asarray(dx_base, np.float32),
          rtol=5e-2,
          atol=5e-2,
      )


def _get_permutation_matrices(k: int) -> jax.Array:
  """Generates all permutation matrices for k streams."""
  perms = jnp.array(list(itertools.permutations(range(k))))
  return jnp.eye(k, dtype=jnp.float32)[perms]


def _make_kernel_inputs(batch=2, sequence=64, streams=4, embedding=256, seed=0):
  """Generates synthetic inputs and parameters for testing mHC kernels."""
  key = jax.random.PRNGKey(seed)
  keys = jax.random.split(key, 12)
  x = jax.random.normal(keys[0], (batch, sequence, streams, embedding), dtype=jnp.bfloat16)
  norm_scale = jax.random.normal(keys[1], (streams * embedding,), dtype=jnp.bfloat16)
  pre_alpha = jax.random.normal(keys[2], (streams * embedding, streams), dtype=jnp.bfloat16) * 0.1
  pre_bias = jax.random.normal(keys[3], (streams,), dtype=jnp.bfloat16) * 0.1
  pre_scale = jnp.array([1.0], dtype=jnp.bfloat16)
  post_alpha = jax.random.normal(keys[5], (streams * embedding, streams), dtype=jnp.bfloat16) * 0.1
  post_bias = jax.random.normal(keys[6], (streams,), dtype=jnp.bfloat16) * 0.1
  post_scale = jnp.array([1.0], dtype=jnp.bfloat16)
  num_perms = math.factorial(streams)
  res_alpha = jax.random.normal(keys[8], (streams * embedding, num_perms), dtype=jnp.bfloat16) * 0.1
  res_bias = jax.random.normal(keys[9], (num_perms,), dtype=jnp.bfloat16) * 0.1
  res_scale = jnp.array([1.0], dtype=jnp.bfloat16)
  permutations = _get_permutation_matrices(streams)
  cotangent = jax.random.normal(keys[11], (batch, sequence, streams, embedding), dtype=jnp.bfloat16)
  weights = mhc_kernel_common.MhcWeights(
      norm_scale=norm_scale,
      pre_alpha=pre_alpha,
      pre_bias=pre_bias,
      pre_scale=pre_scale,
      post_alpha=post_alpha,
      post_bias=post_bias,
      post_scale=post_scale,
      res_alpha=res_alpha,
      res_bias=res_bias,
      res_scale=res_scale,
  )
  return x, weights, permutations, cotangent


def _run_pipeline_reference(x, weights: mhc_kernel_common.MhcWeights, permutations):
  """Runs the native JAX/XLA reference implementation of the mHC pipeline."""
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  flattened_size = streams * embedding
  permutation_count = permutations.shape[0]
  x_flat = x.reshape(tokens, streams, embedding)
  flattened_f32 = x_flat.reshape(tokens, flattened_size).astype(jnp.float32)
  normalized = (
      flattened_f32
      * jax.lax.rsqrt(jnp.mean(flattened_f32 * flattened_f32, axis=-1, keepdims=True) + 1e-5)
      * weights.norm_scale.astype(jnp.float32)
  ).astype(x.dtype)

  h_pre = (
      jax.nn.sigmoid(
          weights.pre_scale.astype(jnp.float32)
          * jnp.dot(
              normalized,
              weights.pre_alpha,
              preferred_element_type=jnp.float32,
          )
          + weights.pre_bias.astype(jnp.float32)
      )
      + 1e-6
  )
  layer_input = jnp.sum(
      h_pre[:, :, None] * x_flat.astype(jnp.float32),
      axis=1,
  ).astype(x.dtype)

  h_post = 2.0 * jax.nn.sigmoid(
      weights.post_scale.astype(jnp.float32)
      * jnp.dot(
          normalized,
          weights.post_alpha,
          preferred_element_type=jnp.float32,
      )
      + weights.post_bias.astype(jnp.float32)
  )
  weights_res = jax.nn.softmax(
      weights.res_scale.astype(jnp.float32)
      * jnp.dot(
          normalized,
          weights.res_alpha,
          preferred_element_type=jnp.float32,
      )
      + weights.res_bias.astype(jnp.float32),
      axis=-1,
  )
  residual = jnp.dot(
      weights_res,
      permutations.reshape(permutation_count, streams * streams).astype(jnp.float32),
  ).reshape(tokens, streams, streams)

  residual_mix = jnp.einsum(
      "tkj,tkd->tjd",
      residual.astype(x.dtype),
      x_flat,
      preferred_element_type=jnp.float32,
  )
  post_mix = h_post.astype(jnp.float32)[:, :, None] * layer_input.astype(jnp.float32)[:, None, :]
  return (residual_mix + post_mix).astype(x.dtype).reshape(x.shape)


def _run_pipeline_api(
    x,
    weights: mhc_kernel_common.MhcWeights,
    permutations,
    implementation=None,
    config: mhc_kernel_common.MhcKernelConfig | None = None,
    interpret=True,
):
  """Runs the mHC Pallas kernel API pipeline."""
  if config is None:
    config = mhc_kernel.MhcKernelConfig(interpret=interpret)
  else:
    config = dataclasses.replace(config, interpret=interpret)
  layer_input, context = mhc_kernel.pre(
      x,
      weights,
      permutations,
      config=config,
      implementation=implementation,
  )
  return mhc_kernel.post(layer_input, context, config=config)


class TestMhcKernelsFwd(parameterized.TestCase):
  """Unit tests for MaxText mHC-lite Pallas forward kernel."""

  def test_doubly_stochastic(self):
    x, weights, permutations, _ = _make_kernel_inputs(batch=1, sequence=128, streams=4, embedding=128)
    config = mhc_kernel.MhcKernelConfig(interpret=True)
    _, context = mhc_kernel.pre(
        x,
        weights,
        permutations,
        config=config,
    )
    row_sums = jnp.sum(context.residual, axis=-1)
    col_sums = jnp.sum(context.residual, axis=-2)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), rtol=1e-3, atol=1e-3)

  @parameterized.named_parameters(
      ("mosaic", "mosaic"),
      ("auto", None),
  )
  def test_forward_parity(self, implementation):
    x, weights, permutations, _ = _make_kernel_inputs(batch=2, sequence=64, streams=4, embedding=256)
    expected = _run_pipeline_reference(x, weights, permutations)
    actual = _run_pipeline_api(
        x,
        weights,
        permutations,
        implementation=implementation,
        interpret=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)

  def test_unsupported_shape_raises_error(self):
    x, weights, permutations, _ = _make_kernel_inputs(batch=1, sequence=16, streams=2, embedding=128)
    config = mhc_kernel.MhcKernelConfig(interpret=True)
    with self.assertRaises(mhc_kernel_common.UnsupportedInputError):
      mhc_kernel.pre(
          x,
          weights,
          permutations,
          config=config,
      )


class TestMhcKernelsBwd(parameterized.TestCase):
  """Unit tests for MaxText mHC-lite Pallas backward kernel."""

  def test_forward_and_backward_vjp_parity(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(batch=2, sequence=64, streams=4, embedding=256)
    expected_out, expected_vjp_fn = jax.vjp(
        lambda x_, w_: _run_pipeline_reference(x_, w_, permutations),
        x,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent)

    actual_out, actual_vjp_fn = jax.vjp(
        lambda x_, w_: _run_pipeline_api(x_, w_, permutations, implementation=None, interpret=True),
        x,
        weights,
    )
    actual_dx, actual_dw = actual_vjp_fn(cotangent)

    np.testing.assert_allclose(actual_out, expected_out, rtol=5e-2, atol=5e-2)

    actual_grads = (actual_dx,) + tuple(jax.tree_util.tree_leaves(actual_dw))
    expected_grads = (expected_dx,) + tuple(jax.tree_util.tree_leaves(expected_dw))
    self.assertEqual(len(actual_grads), len(expected_grads))
    for i, (actual_g, expected_g) in enumerate(zip(actual_grads, expected_grads)):
      tol = 0.05 if actual_g.size == 1 else 0.02
      scale = max(float(np.max(np.abs(np.asarray(expected_g, np.float32)))), 1e-7)
      np.testing.assert_allclose(
          np.asarray(actual_g, np.float32),
          np.asarray(expected_g, np.float32),
          rtol=0.0,
          atol=tol * scale,
          err_msg=f"Gradient leaf {i} mismatch",
      )

  def test_forward_and_backward_vjp_parity_feature_tiled(self):
    x, weights, permutations, cotangent = _make_kernel_inputs(batch=2, sequence=64, streams=4, embedding=256)
    config = mhc_kernel.MhcKernelConfig(bwd_feature_block_size=128)
    expected_out, expected_vjp_fn = jax.vjp(
        lambda x_, w_: _run_pipeline_reference(x_, w_, permutations),
        x,
        weights,
    )
    expected_dx, expected_dw = expected_vjp_fn(cotangent)

    actual_out, actual_vjp_fn = jax.vjp(
        lambda x_, w_: _run_pipeline_api(x_, w_, permutations, config=config, interpret=True),
        x,
        weights,
    )
    actual_dx, actual_dw = actual_vjp_fn(cotangent)

    np.testing.assert_allclose(actual_out, expected_out, rtol=5e-2, atol=5e-2)

    actual_grads = (actual_dx,) + tuple(jax.tree_util.tree_leaves(actual_dw))
    expected_grads = (expected_dx,) + tuple(jax.tree_util.tree_leaves(expected_dw))
    self.assertEqual(len(actual_grads), len(expected_grads))
    for i, (actual_g, expected_g) in enumerate(zip(actual_grads, expected_grads)):
      tol = 0.05 if actual_g.size == 1 else 0.02
      scale = max(float(np.max(np.abs(np.asarray(expected_g, np.float32)))), 1e-7)
      np.testing.assert_allclose(
          np.asarray(actual_g, np.float32),
          np.asarray(expected_g, np.float32),
          rtol=0.0,
          atol=tol * scale,
          err_msg=f"Feature-tiled gradient leaf {i} mismatch",
      )


class TestMhcCostEstimates(unittest.TestCase):
  """Unit tests for analytical CostEstimate computations on MhcDims."""

  def setUp(self):
    self.dims = mhc_kernel_common.MhcDims(tokens=256, streams=4, embedding=512, num_permutations=24)

  def test_dims_properties(self):
    self.assertEqual(self.dims.flattened_size, 4 * 512)
    self.assertEqual(self.dims.phi_cols, 2 * 4 + 24)
    self.assertEqual(self.dims.pre_slice, slice(0, 4))
    self.assertEqual(self.dims.post_slice, slice(4, 8))
    self.assertEqual(self.dims.res_slice, slice(8, 32))

  def test_cost_estimates_non_zero_and_valid(self):
    costs = [
        ("coeff_fwd", self.dims.coeff_fwd_cost()),
        ("pre_apply_fwd", self.dims.pre_apply_fwd_cost()),
        ("post_apply_fwd", self.dims.post_apply_fwd_cost()),
        ("pre_apply_bwd", self.dims.pre_apply_bwd_cost()),
        ("coeff_bwd", self.dims.coeff_bwd_cost()),
        ("post_apply_bwd", self.dims.post_apply_bwd_cost()),
    ]
    for name, cost in costs:
      self.assertIsInstance(cost.flops, int, msg=f"{name} flops must be int")
      self.assertGreater(cost.flops, 0, msg=f"{name} flops must be positive")
      self.assertIsInstance(cost.bytes_accessed, int, msg=f"{name} bytes_accessed must be int")
      self.assertGreater(cost.bytes_accessed, 0, msg=f"{name} bytes_accessed must be positive")
      self.assertIsInstance(cost.transcendentals, int, msg=f"{name} transcendentals must be int")
      self.assertGreaterEqual(cost.transcendentals, 0, msg=f"{name} transcendentals must be non-negative")

    # coeff kernels have transcendentals (sigmoid + softmax)
    self.assertGreater(self.dims.coeff_fwd_cost().transcendentals, 0)
    self.assertGreater(self.dims.coeff_bwd_cost().transcendentals, 0)
    # pre and post apply kernels have 0 transcendentals
    self.assertEqual(self.dims.pre_apply_fwd_cost().transcendentals, 0)
    self.assertEqual(self.dims.post_apply_fwd_cost().transcendentals, 0)
    self.assertEqual(self.dims.pre_apply_bwd_cost().transcendentals, 0)
    self.assertEqual(self.dims.post_apply_bwd_cost().transcendentals, 0)


if __name__ == "__main__":
  absltest.main()
