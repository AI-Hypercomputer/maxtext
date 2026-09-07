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

"""Integration tests for tensor distribution debugging in MaxText layers."""

import sys
import unittest
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.decoders import DecoderLayer, SequentialBlockDecoderLayers
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.moe import RoutedMoE
from maxtext.layers.nnx_decoders import NNXDecoderLayer
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.models.mixtral import MixtralDecoderLayer
from maxtext.utils import debug_tensor_interceptors
from maxtext.utils import debug_tensor_utils
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np

_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "debug_tensor_integration_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 1,
    "attention": "dot_product",
    "max_target_length": 16,
    "base_emb_dim": 64,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 128,
    "max_prefill_predict_length": 4,
    "scan_layers": False,
}


def _make_config(**overrides):
  merged = {**_BASE_CONFIG, **overrides}
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


class DebugTensorIntegrationTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def _make_inputs(self, cfg):
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    emb_dim = cfg.emb_dim
    inputs = jax.random.normal(self.rng, (batch, seq_len, emb_dim), dtype=jnp.float32)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return inputs, segment_ids, positions

  def test_linen_decoder_layer_debug_tensor(self):
    cfg_disabled = _make_config(debug_tensor_distribution=False)
    cfg_enabled = _make_config(debug_tensor_distribution=True)
    mesh = _make_mesh(cfg_disabled)

    inputs, segment_ids, positions = self._make_inputs(cfg_disabled)

    layer_disabled = DecoderLayer(
        config=cfg_disabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
    )
    layer_enabled = DecoderLayer(
        config=cfg_enabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
    )

    init_rng = jax.random.PRNGKey(42)
    variables = layer_disabled.init(
        init_rng,
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    # Forward pass
    out_disabled, _ = layer_disabled.apply(
        variables,
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    out_enabled, _ = layer_enabled.apply(
        variables,
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    np.testing.assert_allclose(np.array(out_disabled), np.array(out_enabled), rtol=1e-6, atol=1e-6)

    # Gradient computation
    def loss_disabled(x):
      out, _ = layer_disabled.apply(
          variables,
          x,
          segment_ids,
          positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out**2)

    def loss_enabled(x):
      out, _ = layer_enabled.apply(
          variables,
          x,
          segment_ids,
          positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out**2)

    grad_disabled = jax.grad(loss_disabled)(inputs)
    grad_enabled = jax.grad(loss_enabled)(inputs)

    np.testing.assert_allclose(np.array(grad_disabled), np.array(grad_enabled), rtol=1e-6, atol=1e-6)

  def test_nnx_decoder_layer_debug_tensor(self):
    cfg_disabled = _make_config(debug_tensor_distribution=False)
    cfg_enabled = _make_config(debug_tensor_distribution=True)
    mesh = _make_mesh(cfg_disabled)

    inputs, segment_ids, positions = self._make_inputs(cfg_disabled)

    rngs = nnx.Rngs(params=42, dropout=1)
    layer_disabled = NNXDecoderLayer(
        config=cfg_disabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )
    # Clone state into layer_enabled for exact weight parity
    rngs_en = nnx.Rngs(params=42, dropout=1)
    layer_enabled = NNXDecoderLayer(
        config=cfg_enabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs_en,
    )

    out_disabled, _ = layer_disabled(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    out_enabled, _ = layer_enabled(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    np.testing.assert_allclose(np.array(out_disabled), np.array(out_enabled), rtol=1e-6, atol=1e-6)

    # Gradient computation through NNX
    def loss_nnx(layer, x):
      out, _ = layer(
          x,
          segment_ids,
          positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out**2)

    grad_disabled = jax.grad(loss_nnx, argnums=1)(layer_disabled, inputs)
    grad_enabled = jax.grad(loss_nnx, argnums=1)(layer_enabled, inputs)
    np.testing.assert_allclose(np.array(grad_disabled), np.array(grad_enabled), rtol=1e-6, atol=1e-6)

  def test_moe_debug_tensor(self):
    cfg_disabled = _make_config(
        debug_tensor_distribution=False,
        base_num_decoder_layers=1,
        base_emb_dim=64,
        base_mlp_dim=128,
        moe_mlp_dim=128,
        base_moe_mlp_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    cfg_enabled = _make_config(
        debug_tensor_distribution=True,
        base_num_decoder_layers=1,
        base_emb_dim=64,
        base_mlp_dim=128,
        moe_mlp_dim=128,
        base_moe_mlp_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    mesh = _make_mesh(cfg_enabled)
    rngs_dis = nnx.Rngs(params=42, dropout=1)
    rngs_en = nnx.Rngs(params=42, dropout=1)

    moe_disabled = RoutedMoE(
        config=cfg_disabled,
        num_experts=4,
        num_experts_per_tok=2,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg_disabled.dtype,
        rngs=rngs_dis,
    )
    moe_enabled = RoutedMoE(
        config=cfg_enabled,
        num_experts=4,
        num_experts_per_tok=2,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg_enabled.dtype,
        rngs=rngs_en,
    )

    batch = cfg_enabled.global_batch_size_to_train_on
    seq_len = cfg_enabled.max_target_length
    emb_dim = cfg_enabled.emb_dim
    inputs = jax.random.normal(self.rng, (batch, seq_len, emb_dim), dtype=jnp.float32)

    out_dis, lb_loss_dis, _ = moe_disabled(inputs)
    out_en, lb_loss_en, _ = moe_enabled(inputs)

    self.assertEqual(out_en.shape, inputs.shape)
    np.testing.assert_allclose(np.array(out_dis), np.array(out_en), rtol=1e-6, atol=1e-6)
    if lb_loss_dis is not None and lb_loss_en is not None:
      np.testing.assert_allclose(np.array(lb_loss_dis), np.array(lb_loss_en), rtol=1e-6, atol=1e-6)
    else:
      self.assertEqual(lb_loss_dis, lb_loss_en)

    # MoE Gradient computation
    def moe_loss(layer, x):
      out, lb, _ = layer(x)
      loss_val = jnp.sum(out**2)
      if lb is not None:
        loss_val += lb
      return loss_val

    grad_dis = jax.grad(moe_loss, argnums=1)(moe_disabled, inputs)
    grad_en = jax.grad(moe_loss, argnums=1)(moe_enabled, inputs)
    np.testing.assert_allclose(np.array(grad_dis), np.array(grad_en), rtol=1e-6, atol=1e-6)

  def test_llama2_layer_unroll_debug_tensor(self):
    cfg_disabled = _make_config(debug_tensor_distribution=False)
    cfg_enabled = _make_config(debug_tensor_distribution=True)
    mesh = _make_mesh(cfg_disabled)

    inputs, segment_ids, positions = self._make_inputs(cfg_disabled)

    rngs_0 = nnx.Rngs(params=42, dropout=1)
    layer0_en = LlamaDecoderLayer(
        config=cfg_enabled,
        model_mode=MODEL_MODE_TRAIN,
        mesh=mesh,
        rngs=rngs_0,
    )
    layer0_en = debug_tensor_interceptors.wrap_nnx_module_for_debug(
        layer0_en, parent_path="decoder/layer_0", config=cfg_enabled
    )

    rngs_1 = nnx.Rngs(params=42, dropout=1)
    layer1_en = LlamaDecoderLayer(
        config=cfg_enabled,
        model_mode=MODEL_MODE_TRAIN,
        mesh=mesh,
        rngs=rngs_1,
    )
    layer1_en = debug_tensor_interceptors.wrap_nnx_module_for_debug(
        layer1_en, parent_path="decoder/layer_1", config=cfg_enabled
    )

    # Forward pass and gradient check
    out0, _ = layer0_en(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(out0.shape, inputs.shape)

    rngs_dis = nnx.Rngs(params=42, dropout=1)
    layer0_dis = LlamaDecoderLayer(
        config=cfg_disabled,
        model_mode=MODEL_MODE_TRAIN,
        mesh=mesh,
        rngs=rngs_dis,
    )
    out0_dis, _ = layer0_dis(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    np.testing.assert_allclose(np.array(out0), np.array(out0_dis), rtol=1e-6, atol=1e-6)

  def test_sequential_decoder_layer_unroll(self):
    cfg_disabled = _make_config(debug_tensor_distribution=False, base_num_decoder_layers=2, scan_layers=True)
    cfg_enabled = _make_config(debug_tensor_distribution=True, base_num_decoder_layers=2, scan_layers=True)
    mesh = _make_mesh(cfg_disabled)

    inputs, segment_ids, positions = self._make_inputs(cfg_disabled)

    seq_layers_dis = SequentialBlockDecoderLayers(
        decoder_layer=DecoderLayer,
        num_decoder_layers=2,
        config=cfg_disabled,
        mesh=mesh,
        quant=None,
        model_mode=MODEL_MODE_TRAIN,
    )
    seq_layers_en = SequentialBlockDecoderLayers(
        decoder_layer=DecoderLayer,
        num_decoder_layers=2,
        config=cfg_enabled,
        mesh=mesh,
        quant=None,
        model_mode=MODEL_MODE_TRAIN,
    )

    init_rng = jax.random.PRNGKey(42)
    variables = seq_layers_dis.init(
        init_rng,
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    out_dis = seq_layers_dis.apply(
        variables,
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled):
      out_en = seq_layers_en.apply(
          variables,
          inputs,
          segment_ids,
          positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    np.testing.assert_allclose(np.array(out_dis[0]), np.array(out_en[0]), rtol=1e-6, atol=1e-6)

  def test_mixtral_decoder_layer_debug_tensor(self):
    cfg_disabled = _make_config(
        debug_tensor_distribution=False,
        base_num_decoder_layers=1,
        base_emb_dim=64,
        base_mlp_dim=128,
        moe_mlp_dim=128,
        base_moe_mlp_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    cfg_enabled = _make_config(
        debug_tensor_distribution=True,
        base_num_decoder_layers=1,
        base_emb_dim=64,
        base_mlp_dim=128,
        moe_mlp_dim=128,
        base_moe_mlp_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    mesh = _make_mesh(cfg_enabled)
    inputs, segment_ids, positions = self._make_inputs(cfg_enabled)

    rngs_dis = nnx.Rngs(params=42, dropout=1)
    rngs_en = nnx.Rngs(params=42, dropout=1)

    layer_dis = MixtralDecoderLayer(
        config=cfg_disabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs_dis,
    )
    layer_en = MixtralDecoderLayer(
        config=cfg_enabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs_en,
    )
    layer_en = debug_tensor_interceptors.wrap_nnx_module_for_debug(
        layer_en, parent_path="decoder/layers_0", config=cfg_enabled
    )

    with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled, step=0):
      out_en, _ = layer_en(
          inputs,
          segment_ids,
          positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    out_dis, _ = layer_dis(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertEqual(out_en.shape, inputs.shape)
    np.testing.assert_allclose(np.array(out_en), np.array(out_dis), rtol=1e-6, atol=1e-6)

  def test_full_moe_telemetry_capture_and_visualizer(self):
    cfg_enabled = _make_config(
        debug_tensor_distribution=True,
        base_num_decoder_layers=1,
        base_emb_dim=64,
        base_mlp_dim=128,
        moe_mlp_dim=128,
        base_moe_mlp_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    mesh = _make_mesh(cfg_enabled)
    inputs, segment_ids, positions = self._make_inputs(cfg_enabled)

    rngs = nnx.Rngs(params=42, dropout=1)
    layer = MixtralDecoderLayer(
        config=cfg_enabled,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )
    layer = debug_tensor_interceptors.wrap_nnx_module_for_debug(layer, parent_path="decoder/layers_0", config=cfg_enabled)

    def log_capture_forward_backward():
      with debug_tensor_interceptors.debug_telemetry_scope(cfg_enabled, step=0):
        # Instrument dummy loss and embeddings
        emb = debug_tensor_utils.debug_tensor(inputs, "decoder/token_embeddings", step=0, enabled=True)

        def loss_fn(x):
          out, _ = layer(
              x,
              segment_ids,
              positions,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
          )
          logits = debug_tensor_utils.debug_tensor(out, "loss/logits", step=0, enabled=True)
          loss_val = jnp.sum(logits**2)
          return debug_tensor_utils.debug_tensor(loss_val, "loss/cross_entropy_per_token", step=0, enabled=True)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(emb)
        return g

    # Run the instrumented forward and backward pass
    grad_val = log_capture_forward_backward()
    self.assertEqual(grad_val.shape, inputs.shape)

  def test_debug_tensor_config_validation(self):
    # Valid configurations
    _make_config(
        debug_tensor_distribution=True,
        debug_tensor_distribution_layers="all",
        debug_tensor_distribution_step_interval=1,
    )
    _make_config(
        debug_tensor_distribution=True,
        base_num_decoder_layers=2,
        debug_tensor_distribution_layers="0,1",
        debug_tensor_distribution_step_interval=5,
    )
    _make_config(
        debug_tensor_distribution=True,
        base_num_decoder_layers=2,
        debug_tensor_distribution_layers="layers_0,layers_1,moe",
    )

    # Invalid: layer index >= model layers
    with self.assertRaises(ValueError):
      _make_config(
          debug_tensor_distribution=True,
          base_num_decoder_layers=2,
          debug_tensor_distribution_layers="2",
      )

    with self.assertRaises(ValueError):
      _make_config(
          debug_tensor_distribution=True,
          base_num_decoder_layers=2,
          debug_tensor_distribution_layers="layers_5",
      )

    with self.assertRaises(ValueError):
      _make_config(
          debug_tensor_distribution=True,
          base_num_decoder_layers=2,
          debug_tensor_distribution_layers="-1",
      )

    # Invalid: step_interval < 1
    with self.assertRaises((ValueError, Exception)):
      _make_config(
          debug_tensor_distribution=True,
          debug_tensor_distribution_step_interval=0,
      )


if __name__ == "__main__":
  absltest.main()
