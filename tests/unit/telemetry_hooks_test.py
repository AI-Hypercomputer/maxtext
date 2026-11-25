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

"""Unit tests for MaxText telemetry hooks A, B, and C."""

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import optax

from maxtext.common import common_types as ctypes
from maxtext.common import metric_logger
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.layers import initializers as max_initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_decoders
from maxtext.models import qwen3
from maxtext.models import qwen3_5
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import gradient_accumulation
from maxtext.utils import maxtext_utils
from maxtext.utils import pipeline_utils
from tests.utils.test_helpers import get_test_config_path


def _make_config(**overrides):
  """Builds a tiny CPU MaxText config using base.yml."""
  defaults = {
      "run_name": "telemetry_hooks_test",
      "hardware": "cpu",
      "skip_jax_distributed_system": True,
      "enable_checkpointing": False,
      "per_device_batch_size": 1.0,
      "max_target_length": 8,
      "base_emb_dim": 16,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "base_mlp_dim": 16,
      "base_moe_mlp_dim": 16,
      "head_dim": 8,
      "base_num_decoder_layers": 2,
      "vocab_size": 32,
      "dtype": "float32",
      "weight_dtype": "float32",
      "attention": "dot_product",
      "scan_layers": False,
      "remat_policy": "minimal",
      "enable_dropout": False,
      "dropout_rate": 0.0,
      "ici_fsdp_parallelism": -1,
      "ici_data_parallelism": 1,
      "ici_expert_parallelism": 1,
      "ici_tensor_parallelism": 1,
      "dcn_data_parallelism": 1,
      "dcn_fsdp_parallelism": 1,
      "dcn_expert_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "sparse_matmul": False,
      "megablox": False,
      "use_tokamax_gmm": False,
      "use_custom_sort_vjp": False,
      "capacity_factor": -1.0,
  }
  defaults.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **defaults)


def _make_mesh(config):
  devices_array = maxtext_utils.create_device_mesh(config)
  return jax.sharding.Mesh(devices_array, config.mesh_axes)


def _qwen3_5_overrides(**extra):
  """Returns base Qwen3.5 config overrides for tiny CPU tests."""
  base = {
      "decoder_block": "qwen3_5",
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "base_emb_dim": 16,
      "base_mlp_dim": 16,
      "base_moe_mlp_dim": 16,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 8,
      "inhomogeneous_layer_cycle_interval": 4,
      "base_num_decoder_layers": 4,
      "gdn_conv_kernel_dim": 2,
      "gdn_key_head_dim": 8,
      "gdn_value_head_dim": 8,
      "gdn_num_key_heads": 2,
      "gdn_num_value_heads": 2,
      "gdn_chunk_size": 4,
      "sparse_matmul": False,
      "capacity_factor": -1.0,
      "load_balance_loss_weight": 0.0,
  }
  base.update(extra)
  return base


def _clone_module_with_config(module_factory, base_module, new_config):
  """Creates a second module with `new_config` sharing bit-identical params/RNGs from `base_module`."""
  _, base_params, base_other = nnx.split(base_module, nnx.Param, ...)
  other_mod = module_factory(new_config)
  nnx.update(other_mod, base_params, base_other)
  return other_mod


def _assert_pytree_bit_equal(tree_a, tree_b):
  leaves_a = jax.tree_util.tree_leaves(tree_a)
  leaves_b = jax.tree_util.tree_leaves(tree_b)
  assert len(leaves_a) == len(leaves_b), f"Leaf count mismatch: {len(leaves_a)} vs {len(leaves_b)}"
  for a, b in zip(leaves_a, leaves_b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@dataclass
class _StubTrainCfg:
  """Minimal config matching pre_train.loss_fn / train_step / eval_step fields."""

  model_name: str = ""
  micro_batch_size_to_train_on: int = 2
  micro_batch_size_to_eval_on: int = 2
  global_batch_size_to_train_on: int = 2
  max_target_length: int = 4
  vocab_size: int = 8
  z_loss_multiplier: float = 0.0
  enable_dropout: bool = False
  use_multimodal: bool = False
  use_indexer: bool = False
  indexer_sparse_training: bool = False
  indexer_loss_scaling_factor: float = 0.0
  num_vocab_tiling: int = 1
  num_experts: int = 4
  num_decoder_layers: int = 2
  param_scan_axis: int = 0
  inhomogeneous_layer_cycle_interval: int = 1
  moe_dropless_fallback: str | None = None
  routed_bias: bool = False
  routed_bias_update_rate: float = 0.0
  routed_score_func: str = "softmax"
  moe_use_megatron_seq_aux_loss: bool = False
  te_moe_block: bool = False
  load_balance_loss_weight: float = 0.0
  moe_log_unweighted_lb_loss: bool = False
  log_per_module_norms: bool = False
  mtp_num_layers: int = 0
  mtp_eval_target_module: int = 0
  use_qk_clip: bool = False
  use_tunix_gradient_accumulation: bool = False
  gradient_accumulation_steps: int = 1
  shard_optimizer_over_data: bool = False
  optimizer_memory_host_offload: bool = False
  parameter_memory_host_offload: bool = False
  gradient_clipping_threshold: float = 0.0
  grad_dtype: jnp.dtype = jnp.float32
  record_internal_nn_metrics: int = 0
  skip_step_on_spikes: bool = False
  shard_mode: int = ctypes.ShardMode.AUTO
  debug_sharding: bool = False
  weight_sparsity_n: int = 0
  weight_sparsity_m: int = 0


class _TinyMoETrainModel(nnx.Module):
  """Minimal NNX model with unscanned layers_0 / layers_1 and optional unweighted LB loss."""

  def __init__(self, vocab_size: int, hidden: int, config, rngs: nnx.Rngs):
    self.config = config
    self.mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))
    self.token_embedder = nnx.Embed(vocab_size, hidden, rngs=rngs)
    self.layers_0 = nnx.Linear(hidden, hidden, rngs=rngs)
    self.layers_1 = nnx.Linear(hidden, hidden, rngs=rngs)
    self.router = nnx.Linear(hidden, config.num_experts, rngs=rngs)
    self.lm_head = nnx.Linear(hidden, vocab_size, rngs=rngs)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      encoder_images=None,
      encoder_image_masks=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks
    del enable_dropout, decoder_target_tokens, decoder_target_mask
    h = self.token_embedder(decoder_input_tokens)
    h0 = jax.nn.silu(self.layers_0(h))
    h1 = jax.nn.silu(self.layers_1(h0))
    router_logits = self.router(h1)
    probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    # Synthetic deterministic per-layer LB loss values depending on router_logits.
    unweighted_lb = jnp.mean(jnp.square(probs)) * self.config.num_experts
    if self.config.load_balance_loss_weight > 0.0:
      self.moe_lb_loss = nnx.Intermediate(unweighted_lb * self.config.load_balance_loss_weight)
    if getattr(self.config, "moe_log_unweighted_lb_loss", False):
      self.moe_lb_loss_unweighted = nnx.Intermediate(jax.lax.stop_gradient(unweighted_lb))
    if getattr(self.config, "record_internal_nn_metrics", 0):
      self.layers_0.activation_amax = nnx.Intermediate(jax.lax.stop_gradient(jnp.max(jnp.abs(h0)).astype(jnp.float32)))
      self.layers_1.activation_amax = nnx.Intermediate(jax.lax.stop_gradient(jnp.max(jnp.abs(h1)).astype(jnp.float32)))
    return self.lm_head(h1)


def _make_stub_batch(batch_size=2, seq_len=4, vocab_size=8):
  rng = np.random.default_rng(7)
  tokens = jnp.asarray(rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32))
  return {
      "inputs": tokens,
      "targets": jnp.roll(tokens, -1, axis=1),
      "inputs_position": jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len)),
      "inputs_segmentation": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
      "targets_segmentation": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
  }


class DefaultOffParityTest(unittest.TestCase):
  """Verifies all new telemetry flags default OFF and emit zero overhead / bit-exact results."""

  def test_config_defaults_off(self):
    cfg = _make_config()
    self.assertEqual(cfg.record_internal_nn_metrics, 0)
    self.assertFalse(cfg.log_per_module_norms)
    self.assertFalse(cfg.moe_log_unweighted_lb_loss)

  def test_pipeline_sown_microbatch_reductions_registered(self):
    self.assertIs(pipeline_utils.SOWN_MICROBATCH_REDUCTIONS["activation_amax"], jnp.max)
    self.assertIs(pipeline_utils.SOWN_MICROBATCH_REDUCTIONS["mlp_wo_input_amax"], jnp.max)
    self.assertIs(pipeline_utils.SOWN_MICROBATCH_REDUCTIONS["moe_wo_input_amax"], jnp.max)
    self.assertIs(pipeline_utils.SOWN_MICROBATCH_REDUCTIONS["moe_lb_loss_unweighted"], jnp.mean)

  def test_train_step_default_off_vs_all_on_bit_exact_loss_and_params(self):
    batch = _make_stub_batch()
    cfg_off = _StubTrainCfg(
        record_internal_nn_metrics=0,
        log_per_module_norms=False,
        moe_log_unweighted_lb_loss=False,
        load_balance_loss_weight=0.0,
    )
    cfg_on = _StubTrainCfg(
        record_internal_nn_metrics=1,
        log_per_module_norms=True,
        moe_log_unweighted_lb_loss=True,
        load_balance_loss_weight=0.0,
    )

    model_off = _TinyMoETrainModel(cfg_off.vocab_size, 8, cfg_off, rngs=nnx.Rngs(0))
    state_off = train_state_nnx.TrainStateNNX(model_off, nnx.Optimizer(model_off, optax.sgd(0.1), wrt=nnx.Param))
    graphdef_off, pure_state_off = nnx.split(state_off)

    model_on = _TinyMoETrainModel(cfg_on.vocab_size, 8, cfg_on, rngs=nnx.Rngs(0))
    state_on = train_state_nnx.TrainStateNNX(model_on, nnx.Optimizer(model_on, optax.sgd(0.1), wrt=nnx.Param))
    graphdef_on, pure_state_on = nnx.split(state_on)

    out_off, metrics_off = pre_train.train_step(graphdef_off, cfg_off, None, None, pure_state_off, batch)
    out_on, metrics_on = pre_train.train_step(graphdef_on, cfg_on, None, None, pure_state_on, batch)

    # Default OFF emits zero telemetry keys.
    for key in metrics_off["scalar"]:
      self.assertFalse(key.startswith("activ_"), key)
      self.assertFalse(key.startswith("grad_norm/"), key)
      self.assertFalse(key.startswith("grad_max/"), key)
      self.assertFalse(key.startswith("param_norm/"), key)
      self.assertFalse(key.startswith("layer_grad_"), key)
      self.assertFalse(key.startswith("layer_param_"), key)
      self.assertNotEqual(key, "learning/moe_lb_loss_unweighted")

    # Bit-exact loss and updated parameters when all diagnostic hooks are ON.
    np.testing.assert_array_equal(
        np.asarray(metrics_off["scalar"]["learning/loss"]),
        np.asarray(metrics_on["scalar"]["learning/loss"]),
    )
    _assert_pytree_bit_equal(out_off, out_on)

    # All 3 hooks emit expected metrics when ON.
    self.assertIn("activ_amax/layer_000", metrics_on["scalar"])
    self.assertIn("activ_amax/layer_001", metrics_on["scalar"])
    self.assertIn("layer_grad_norm/layer_000", metrics_on["scalar"])
    self.assertIn("layer_grad_norm/layer_001", metrics_on["scalar"])
    self.assertIn("learning/moe_lb_loss_unweighted", metrics_on["scalar"])
    self.assertGreater(float(metrics_on["scalar"]["learning/moe_lb_loss_unweighted"]), 0.0)


class HookAActivationTelemetryTest(parameterized.TestCase):
  """Tests Hook A across DecoderLayer, MlpBlock, Qwen3_5, Qwen3Moe, RoutedMoE, and record_activation_metrics."""

  def test_mlp_block_wo_input_amax_and_parity(self):
    cfg_off = _make_config(record_internal_nn_metrics=0)
    cfg_on = _make_config(record_internal_nn_metrics=1)
    mesh = _make_mesh(cfg_off)

    def _build(cfg):
      return linears.MlpBlock(
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=tuple(cfg.mlp_activations),
          intermediate_dropout_rate=0.0,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          config=cfg,
          mesh=mesh,
          rngs=nnx.Rngs(1),
      )

    mlp_off = _build(cfg_off)
    mlp_on = _clone_module_with_config(_build, mlp_off, cfg_on)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 4, cfg_off.emb_dim), dtype=jnp.float32)

    def _run(mod, inp):
      out = mod(inp, deterministic=True)
      inter = nnx.pop(mod, nnx.Intermediate).to_pure_dict()
      return jnp.sum(out), (out, inter)

    with jax.set_mesh(mesh), logical_axis_rules(cfg_off.logical_axis_rules):
      (_, (out_off, inter_off)), grads_off = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(mlp_off, x)
      (_, (out_on, inter_on)), grads_on = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(mlp_on, x)

    self.assertEqual(inter_off, {})
    self.assertIn("mlp_wo_input_amax", inter_on)
    _assert_pytree_bit_equal(out_off, out_on)
    _assert_pytree_bit_equal(grads_off, grads_on)

    # Compare against manual SwiGLU intermediate before wo.
    with jax.set_mesh(mesh), logical_axis_rules(cfg_off.logical_axis_rules):
      wi0 = jax.nn.silu(mlp_off.wi_0(x))
      wi1 = mlp_off.wi_1(x)
      expected_amax = jnp.max(jnp.abs(wi0 * wi1)).astype(jnp.float32)
    np.testing.assert_allclose(np.asarray(inter_on["mlp_wo_input_amax"]), np.asarray(expected_amax), rtol=1e-6, atol=1e-6)
    self.assertEqual(inter_on["mlp_wo_input_amax"].dtype, jnp.float32)

  def test_decoder_layer_activation_amax_and_stop_gradient(self):
    cfg_off = _make_config(record_internal_nn_metrics=0)
    cfg_on = _make_config(record_internal_nn_metrics=1)
    mesh = _make_mesh(cfg_off)

    def _build(cfg):
      return nnx_decoders.NNXDecoderLayer(config=cfg, mesh=mesh, model_mode=ctypes.MODEL_MODE_TRAIN, rngs=nnx.Rngs(3))

    layer_off = _build(cfg_off)
    layer_on = _clone_module_with_config(_build, layer_off, cfg_on)
    x = jax.random.normal(jax.random.PRNGKey(4), (4, 4, cfg_off.emb_dim), dtype=jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (4, 4))
    seg = jnp.ones((4, 4), dtype=jnp.int32)

    def _run(mod, inp):
      out, _ = mod(inp, seg, pos, deterministic=True, model_mode=ctypes.MODEL_MODE_TRAIN)
      inter = nnx.pop(mod, nnx.Intermediate).to_pure_dict()
      return jnp.sum(out), (out, inter)

    with jax.set_mesh(mesh), logical_axis_rules(cfg_off.logical_axis_rules):
      (_, (out_off, inter_off)), grads_off = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(layer_off, x)
      (_, (out_on, inter_on)), grads_on = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(layer_on, x)

    self.assertEqual(inter_off, {})
    for key in ("activation_mean", "activation_stdev", "activation_fraction_zero", "activation_amax"):
      vals = maxtext_utils.collect_intermediates_by_suffix(inter_on, key)
      self.assertLen(vals, 1, msg=f"Missing {key}")
    amax = maxtext_utils.collect_intermediates_by_suffix(inter_on, "activation_amax")[0][0]
    np.testing.assert_allclose(np.asarray(amax), np.asarray(jnp.max(jnp.abs(out_on)).astype(jnp.float32)), rtol=1e-6)
    self.assertLen(maxtext_utils.collect_intermediates_by_suffix(inter_on, "mlp_wo_input_amax"), 1)
    _assert_pytree_bit_equal(out_off, out_on)
    _assert_pytree_bit_equal(grads_off, grads_on)

  def test_qwen3_5_and_qwen3_moe_decoder_layers(self):
    # 1. Qwen3_5DecoderLayer (inherits Qwen3NextDecoderLayer.__call__).
    cfg_off = _make_config(**_qwen3_5_overrides(record_internal_nn_metrics=0))
    cfg_on = _make_config(**_qwen3_5_overrides(record_internal_nn_metrics=1))
    mesh = _make_mesh(cfg_off)

    def _build_q35(cfg):
      return qwen3_5.Qwen3_5DecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=ctypes.MODEL_MODE_TRAIN,
          layer_idx=3,
          is_full_attention_layer=True,
          rngs=nnx.Rngs(5),
      )

    layer_off = _build_q35(cfg_off)
    layer_on = _clone_module_with_config(_build_q35, layer_off, cfg_on)
    x = jax.random.normal(jax.random.PRNGKey(6), (4, 4, cfg_off.emb_dim), dtype=jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (4, 4))
    seg = jnp.ones((4, 4), dtype=jnp.int32)

    def _run(mod, inp):
      out, _ = mod(inp, seg, pos, deterministic=True, model_mode=ctypes.MODEL_MODE_TRAIN)
      inter = nnx.pop(mod, nnx.Intermediate).to_pure_dict()
      return jnp.sum(out), (out, inter)

    with jax.set_mesh(mesh), logical_axis_rules(cfg_off.logical_axis_rules):
      (_, (out_off, inter_off)), grads_off = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(layer_off, x)
      (_, (out_on, inter_on)), grads_on = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(layer_on, x)

    self.assertEqual(inter_off, {})
    for key in (
        "activation_mean",
        "activation_stdev",
        "activation_fraction_zero",
        "activation_amax",
        "mlp_wo_input_amax",
        "moe_wo_input_amax",
    ):
      vals = maxtext_utils.collect_intermediates_by_suffix(inter_on, key)
      self.assertLen(vals, 1, msg=f"Qwen3_5DecoderLayer missing {key}")
    amax = maxtext_utils.collect_intermediates_by_suffix(inter_on, "activation_amax")[0][0]
    np.testing.assert_allclose(np.asarray(amax), np.asarray(jnp.max(jnp.abs(out_on)).astype(jnp.float32)), rtol=1e-6)
    _assert_pytree_bit_equal(out_off, out_on)
    _assert_pytree_bit_equal(grads_off, grads_on)

    # 2. Qwen3MoeDecoderLayer.
    q3_moe_off = _make_config(
        decoder_block="qwen3_moe",
        num_experts=4,
        num_experts_per_tok=2,
        record_internal_nn_metrics=0,
    )
    q3_moe_on = _make_config(
        decoder_block="qwen3_moe",
        num_experts=4,
        num_experts_per_tok=2,
        record_internal_nn_metrics=1,
    )

    def _build_q3moe(cfg):
      return qwen3.Qwen3MoeDecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=ctypes.MODEL_MODE_TRAIN,
          quant=None,
          rngs=nnx.Rngs(7),
      )

    m_off = _build_q3moe(q3_moe_off)
    m_on = _clone_module_with_config(_build_q3moe, m_off, q3_moe_on)
    with jax.set_mesh(mesh), logical_axis_rules(q3_moe_off.logical_axis_rules):
      (_, (out_off2, inter_off2)), grads_off2 = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(m_off, x)
      (_, (out_on2, inter_on2)), grads_on2 = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(m_on, x)

    self.assertEqual(inter_off2, {})
    for key in (
        "activation_mean",
        "activation_stdev",
        "activation_fraction_zero",
        "activation_amax",
        "moe_wo_input_amax",
    ):
      self.assertLen(maxtext_utils.collect_intermediates_by_suffix(inter_on2, key), 1, msg=f"Qwen3Moe missing {key}")
    _assert_pytree_bit_equal(out_off2, out_on2)
    _assert_pytree_bit_equal(grads_off2, grads_on2)

  def test_qwen3_5_scannable_block_logical_order_scanned_and_unscanned(self):
    # 1. Exact logical interleaving check with synthetic 2 blocks x 4 layers + mtp_block ignored.
    fake_intermediates = {
        "decoder": {
            "layers": {
                "local_layers": {
                    "activation_amax": jnp.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
                    "mlp": {
                        "moe_block": {
                            "moe_wo_input_amax": jnp.array([[10.0, 11.0, 12.0], [14.0, 15.0, 16.0]], dtype=jnp.float32)
                        },
                        "shared_expert": {
                            "mlp_wo_input_amax": jnp.array([[20.0, 21.0, 22.0], [24.0, 25.0, 26.0]], dtype=jnp.float32)
                        },
                    },
                },
                "global_layer": {
                    "activation_amax": jnp.array([3.0, 7.0], dtype=jnp.float32),
                    "mlp": {
                        "moe_block": {"moe_wo_input_amax": jnp.array([13.0, 17.0], dtype=jnp.float32)},
                        "shared_expert": {"mlp_wo_input_amax": jnp.array([23.0, 27.0], dtype=jnp.float32)},
                    },
                },
            }
        },
        "mtp_block": {
            "mtp_layer_1": {
                "activation_amax": jnp.array(999.0, dtype=jnp.float32),
            }
        },
    }
    out_metrics = {"scalar": {}}
    metric_logger.record_activation_metrics(out_metrics, fake_intermediates, SimpleNamespace(num_decoder_layers=8))
    for i in range(8):
      self.assertEqual(float(out_metrics["scalar"][f"activ_amax/layer_{i:03d}"]), float(i))
      self.assertEqual(float(out_metrics["scalar"][f"activ_moe_wo_amax/layer_{i:03d}"]), float(10 + i))
      self.assertEqual(float(out_metrics["scalar"][f"activ_mlp_wo_amax/layer_{i:03d}"]), float(20 + i))

    # 2. Numeric sorting of unscanned layers_0 .. layers_11 (not lexicographic layers_10 before layers_2).
    unscanned_inter = {
        "decoder": {f"layers_{i}": {"activation_amax": jnp.array(float(i), dtype=jnp.float32)} for i in range(12)}
    }
    out_unscanned = {"scalar": {}}
    metric_logger.record_activation_metrics(out_unscanned, unscanned_inter, SimpleNamespace(num_decoder_layers=12))
    for i in range(12):
      self.assertEqual(float(out_unscanned["scalar"][f"activ_amax/layer_{i:03d}"]), float(i))

    # 3. End-to-end NNXDecoder with Qwen3_5ScannableBlock (scan_layers=True, 2 blocks x 4 layers = 8 layers)
    #    and scan_layers=False (8 layers).
    for scan_layers in (True, False):
      with self.subTest(scan_layers=scan_layers):
        cfg = _make_config(
            **_qwen3_5_overrides(
                base_num_decoder_layers=8,
                inhomogeneous_layer_cycle_interval=4,
                scan_layers=scan_layers,
                record_internal_nn_metrics=1,
            )
        )
        mesh = _make_mesh(cfg)
        shared_emb = nnx_decoders.Embed(
            num_embeddings=cfg.vocab_size,
            num_features=cfg.emb_dim,
            dtype=cfg.dtype,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            config=cfg,
            mesh=mesh,
            rngs=nnx.Rngs(11),
        )
        decoder = nnx_decoders.NNXDecoder(
            config=cfg,
            mesh=mesh,
            model_mode=ctypes.MODEL_MODE_TRAIN,
            rngs=nnx.Rngs(12),
        )
        tokens = jnp.ones((4, 4), dtype=jnp.int32)
        pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (4, 4))
        seg = jnp.ones((4, 4), dtype=jnp.int32)
        with jax.set_mesh(mesh), logical_axis_rules(cfg.logical_axis_rules):
          _ = decoder(shared_emb, tokens, pos, decoder_segment_ids=seg, deterministic=True)
        inter = nnx.pop(decoder, nnx.Intermediate).to_pure_dict()
        recorded = {"scalar": {}}
        metric_logger.record_activation_metrics(recorded, inter, cfg)
        for i in range(8):
          for prefix in (
              "activ_mean",
              "activ_stdev",
              "activ_fraction_zero",
              "activ_amax",
              "activ_mlp_wo_amax",
              "activ_moe_wo_amax",
          ):
            key = f"{prefix}/layer_{i:03d}"
            self.assertIn(key, recorded["scalar"], msg=f"Missing {key} for scan_layers={scan_layers}")
            self.assertTrue(np.isfinite(float(recorded["scalar"][key])), msg=f"Non-finite {key}")

  @parameterized.named_parameters(
      ("dense_no_cap", False, -1.0, False, 1, 1),
      ("dense_with_cap", False, 2.0, False, 1, 1),
      ("sparse_non_roe_fsdp4", True, -1.0, False, 1, 1),
      ("sparse_roe_ep2_chunks2", True, -1.0, True, 2, 2),
  )
  def test_routed_moe_wo_input_amax_and_parity(self, sparse_matmul, capacity_factor, use_roe, n_chunks, ep):
    fsdp = 4 // ep
    common = {
        "per_device_batch_size": 1.0,
        "max_target_length": 8,
        "base_emb_dim": 16,
        "base_moe_mlp_dim": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "ici_fsdp_parallelism": fsdp,
        "ici_expert_parallelism": ep,
        "sparse_matmul": sparse_matmul,
        "capacity_factor": capacity_factor,
        "use_ring_of_experts": use_roe,
        "num_moe_token_chunks": n_chunks,
        "load_balance_loss_weight": 0.0,
    }
    cfg_off = _make_config(record_internal_nn_metrics=0, **common)
    cfg_on = _make_config(record_internal_nn_metrics=1, **common)
    mesh = _make_mesh(cfg_off)

    def _build(cfg):
      return moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=nnx.Rngs(21),
      )

    m_off = _build(cfg_off)
    m_on = _clone_module_with_config(_build, m_off, cfg_on)
    x = jax.random.normal(jax.random.PRNGKey(22), (4, 8, cfg_off.emb_dim), dtype=jnp.float32)

    def _run(mod, inp):
      out, lb, _ = mod(inp)
      inter = nnx.pop(mod, nnx.Intermediate).to_pure_dict()
      return jnp.sum(out), (out, lb, inter)

    with jax.set_mesh(mesh), logical_axis_rules(cfg_off.logical_axis_rules):
      (_, (out_off, _, inter_off)), grads_off = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(m_off, x)
      (_, (out_on, _, inter_on)), grads_on = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(m_on, x)

    self.assertNotIn("moe_wo_input_amax", inter_off)
    self.assertIn("moe_wo_input_amax", inter_on)
    amax = inter_on["moe_wo_input_amax"]
    self.assertEqual(amax.dtype, jnp.float32)
    self.assertGreater(float(amax), 0.0)
    self.assertTrue(np.isfinite(float(amax)))
    _assert_pytree_bit_equal(out_off, out_on)
    _assert_pytree_bit_equal(grads_off, grads_on)

    # Exact numerical reference check on sparse_matmul: compute routed SwiGLU activation directly.
    if sparse_matmul:
      with jax.set_mesh(mesh), logical_axis_rules(cfg_off.logical_axis_rules):
        gate_logits, _ = m_off.gate(x)
        _, selected = m_off.get_topk(gate_logits, None, m_off.rngs)
        w0 = m_off.wi_0.get_value()
        w1 = m_off.wi_1.get_value()
        # Evaluate only selected experts per token: [B, S, K, H].
        h0 = jnp.einsum("bsm,emh->bseh", x, w0)
        h1 = jnp.einsum("bsm,emh->bseh", x, w1)
        act = m_off.apply_ffn_activation(h0, h1)
        chosen = jnp.take_along_axis(act, selected[..., None], axis=2)
        ref_amax = jnp.max(jnp.abs(chosen)).astype(jnp.float32)
      np.testing.assert_allclose(np.asarray(amax), np.asarray(ref_amax), rtol=1e-5, atol=1e-5)


class HookBPerModuleNormsTest(unittest.TestCase):
  """Tests Hook B (`compute_per_module_norms` and `train_step` wiring)."""

  def test_unscanned_layout_exact_norms_and_mtp_exclusion(self):
    raw_grads = {
        "token_embedder": {"embedding": jnp.array([3.0, 4.0], dtype=jnp.bfloat16)},
        "decoder": {
            "layers_0": {
                "attn": {"kernel": jnp.array([1.0, -2.0], dtype=jnp.float32)},
                "mlp": {"kernel": jnp.array([2.0, 0.0], dtype=jnp.float32)},
            },
            "layers_1": {
                "attn": {"kernel": jnp.array([0.0, 6.0], dtype=jnp.float32)},
            },
            "layers_10": {
                "attn": {"kernel": jnp.array([-5.0], dtype=jnp.float32)},
            },
        },
        "mtp_block": {
            "mtp_layer_1": {"kernel": jnp.array([100.0], dtype=jnp.float32)},
        },
    }
    model_params = {
        "token_embedder": {"embedding": jnp.array([6.0, 8.0], dtype=jnp.float32)},
        "decoder": {
            "layers_0": {
                "attn": {"kernel": jnp.array([3.0, 0.0], dtype=jnp.float32)},
                "mlp": {"kernel": jnp.array([0.0, 4.0], dtype=jnp.float32)},
            },
            "layers_1": {
                "attn": {"kernel": jnp.array([7.0, 0.0], dtype=jnp.float32)},
            },
            "layers_10": {
                "attn": {"kernel": jnp.array([9.0], dtype=jnp.float32)},
            },
        },
        "mtp_block": {
            "mtp_layer_1": {"kernel": jnp.array([50.0], dtype=jnp.float32)},
        },
    }
    cfg = SimpleNamespace(param_scan_axis=0, inhomogeneous_layer_cycle_interval=1)
    out = metric_logger.compute_per_module_norms(raw_grads, model_params, cfg)

    np.testing.assert_allclose(float(out["grad_norm/token_embedder/embedding"]), 5.0, rtol=1e-3)
    np.testing.assert_allclose(float(out["grad_max/token_embedder/embedding"]), 4.0, rtol=1e-3)
    np.testing.assert_allclose(float(out["param_norm/token_embedder/embedding"]), 10.0, rtol=1e-6)

    # Layer 0 aggregates attn ([1, -2] -> sq=5, max=2) and mlp ([2, 0] -> sq=4, max=2): total sq=9, norm=3.
    np.testing.assert_allclose(float(out["layer_grad_norm/layer_000"]), 3.0, rtol=1e-6)
    np.testing.assert_allclose(float(out["layer_grad_max/layer_000"]), 2.0, rtol=1e-6)
    np.testing.assert_allclose(float(out["layer_param_norm/layer_000"]), 5.0, rtol=1e-6)

    np.testing.assert_allclose(float(out["layer_grad_norm/layer_001"]), 6.0, rtol=1e-6)
    np.testing.assert_allclose(float(out["layer_grad_max/layer_001"]), 6.0, rtol=1e-6)
    np.testing.assert_allclose(float(out["layer_param_norm/layer_001"]), 7.0, rtol=1e-6)

    np.testing.assert_allclose(float(out["layer_grad_norm/layer_010"]), 5.0, rtol=1e-6)
    np.testing.assert_allclose(float(out["layer_grad_max/layer_010"]), 5.0, rtol=1e-6)
    np.testing.assert_allclose(float(out["layer_param_norm/layer_010"]), 9.0, rtol=1e-6)

    # MTP block gets per-module norms, not decoder layer norms.
    np.testing.assert_allclose(float(out["grad_norm/mtp_block/mtp_layer_1/kernel"]), 100.0, rtol=1e-6)
    self.assertNotIn("layer_grad_norm/layer_002", out)

  def test_standard_scanned_layout_reads_metadata_from_model_params(self):
    # param_scan_axis=1: kernel has shape (D=2, L=3, M=2) with sharding=("embed", "layers", "mlp").
    # raw_grads has NO metadata (plain dict, simulating stripped grad metadata), model_params uses nnx.Param.
    g_kernel = jnp.stack(
        [
            jnp.full((2, 2), 1.0, dtype=jnp.float32),  # layer 0: sq=4, max=1
            jnp.full((2, 2), 2.0, dtype=jnp.float32),  # layer 1: sq=16, max=2
            jnp.full((2, 2), -3.0, dtype=jnp.float32),  # layer 2: sq=36, max=3
        ],
        axis=1,
    )
    p_kernel = jnp.stack(
        [
            jnp.full((2, 2), 0.5, dtype=jnp.float32),  # layer 0: sq=1
            jnp.full((2, 2), 1.0, dtype=jnp.float32),  # layer 1: sq=4
            jnp.full((2, 2), 1.5, dtype=jnp.float32),  # layer 2: sq=9
        ],
        axis=1,
    )
    raw_grads = {"decoder": {"layers": {"attn": {"kernel": g_kernel}}}}
    model_params = {
        "decoder": {
            "layers": {
                "attn": {
                    "kernel": nnx.Param(p_kernel, sharding=("embed", "layers", "mlp"), param_scan_axis=1),
                }
            }
        }
    }
    cfg = SimpleNamespace(param_scan_axis=0, inhomogeneous_layer_cycle_interval=1)
    out = metric_logger.compute_per_module_norms(raw_grads, model_params, cfg)

    np.testing.assert_allclose(float(out["grad_norm/decoder/layers/attn/kernel"]), np.sqrt(4 + 16 + 36), rtol=1e-6)
    np.testing.assert_allclose(float(out["grad_max/decoder/layers/attn/kernel"]), 3.0, rtol=1e-6)
    for i, (exp_gn, exp_gm, exp_pn) in enumerate(((2.0, 1.0, 1.0), (4.0, 2.0, 2.0), (6.0, 3.0, 3.0))):
      np.testing.assert_allclose(float(out[f"layer_grad_norm/layer_{i:03d}"]), exp_gn, rtol=1e-6)
      np.testing.assert_allclose(float(out[f"layer_grad_max/layer_{i:03d}"]), exp_gm, rtol=1e-6)
      np.testing.assert_allclose(float(out[f"layer_param_norm/layer_{i:03d}"]), exp_pn, rtol=1e-6)

  def test_hybrid_qwen3_5_scannable_block_layout_synthetic_and_real(self):
    # 1. Synthetic hybrid pytree: B=2 blocks, K=3 local layers + 1 global layer -> 8 logical layers.
    # Use param_scan_axis=1 sharding metadata: ("embed", "layers", "local_layers", "mlp") and ("embed", "layers", "mlp").
    local_vals = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)  # layers 0,1,2 and 4,5,6
    global_vals = np.array([4.0, 8.0], dtype=np.float32)  # layers 3 and 7

    g_local = jnp.broadcast_to(local_vals[None, :, :, None], (2, 2, 3, 2))  # each slice has 4 elements -> norm = 2*v
    g_global = jnp.broadcast_to(global_vals[None, :, None], (2, 2, 2))
    p_local = g_local * 0.5
    p_global = g_global * 0.5

    raw_grads = {
        "decoder": {
            "layers": {
                "local_layers": {"kernel": g_local},
                "global_layer": {"kernel": g_global},
            }
        }
    }
    model_params = {
        "decoder": {
            "layers": {
                "local_layers": {
                    "kernel": nnx.Param(p_local, sharding=("embed", "layers", "local_layers", "mlp"), param_scan_axis=1)
                },
                "global_layer": {"kernel": nnx.Param(p_global, sharding=("embed", "layers", "mlp"), param_scan_axis=1)},
            }
        }
    }
    cfg = SimpleNamespace(param_scan_axis=1, inhomogeneous_layer_cycle_interval=4)
    out = metric_logger.compute_per_module_norms(raw_grads, model_params, cfg)

    for layer_idx in range(8):
      v = float(layer_idx + 1)
      np.testing.assert_allclose(float(out[f"layer_grad_norm/layer_{layer_idx:03d}"]), 2.0 * v, rtol=1e-6)
      np.testing.assert_allclose(float(out[f"layer_grad_max/layer_{layer_idx:03d}"]), v, rtol=1e-6)
      np.testing.assert_allclose(float(out[f"layer_param_norm/layer_{layer_idx:03d}"]), v, rtol=1e-6)

    # 2. Real NNXDecoder with Qwen3_5ScannableBlock (scan_layers=True, param_scan_axis=1, 2 blocks x 4 = 8 layers).
    real_cfg = _make_config(
        **_qwen3_5_overrides(
            base_num_decoder_layers=8,
            inhomogeneous_layer_cycle_interval=4,
            scan_layers=True,
            param_scan_axis=1,
        )
    )
    mesh = _make_mesh(real_cfg)
    decoder = nnx_decoders.NNXDecoder(
        config=real_cfg,
        mesh=mesh,
        model_mode=ctypes.MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(32),
    )
    params_state = nnx.state(decoder, nnx.Param)
    # Use a scaled copy as raw_grads with stripped metadata via nnx.to_pure_dict().
    grads_pure = jax.tree.map(lambda a: a * 0.25, params_state.to_pure_dict())
    real_out = metric_logger.compute_per_module_norms(grads_pure, params_state, real_cfg)
    for layer_idx in range(8):
      self.assertIn(f"layer_grad_norm/layer_{layer_idx:03d}", real_out)
      self.assertIn(f"layer_grad_max/layer_{layer_idx:03d}", real_out)
      self.assertIn(f"layer_param_norm/layer_{layer_idx:03d}", real_out)
      # Because grads == 0.25 * params, layer_grad_norm == 0.25 * layer_param_norm for every layer.
      np.testing.assert_allclose(
          float(real_out[f"layer_grad_norm/layer_{layer_idx:03d}"]),
          0.25 * float(real_out[f"layer_param_norm/layer_{layer_idx:03d}"]),
          rtol=1e-5,
      )

  def test_train_step_with_optimizer_host_offload_and_log_per_module_norms(self):
    batch = _make_stub_batch()
    cfg = _StubTrainCfg(log_per_module_norms=True, optimizer_memory_host_offload=True)
    model = _TinyMoETrainModel(cfg.vocab_size, 8, cfg, rngs=nnx.Rngs(41))
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param))
    graphdef, pure_state = nnx.split(state)

    # Provide CPU NamedShardings matching state_mesh_shardings.optimizer so host-offload path runs.
    mesh = jax.make_mesh((1,), ("data",))
    replicated = NamedSharding(mesh, P())
    opt_shardings = jax.tree.map(lambda _: replicated, nnx.state(state.optimizer))
    state_shardings = SimpleNamespace(optimizer=opt_shardings)

    _, metrics = pre_train.train_step(graphdef, cfg, state_shardings, None, pure_state, batch)
    self.assertIn("layer_grad_norm/layer_000", metrics["scalar"])
    self.assertIn("layer_param_norm/layer_000", metrics["scalar"])
    self.assertIn("layer_grad_max/layer_001", metrics["scalar"])


class HookCUnweightedLbLossTest(parameterized.TestCase):
  """Tests Hook C across RoutedMoE, loss_fn, train_step, eval_step, GA, and MetricLogger."""

  @parameterized.named_parameters(
      ("dense_switch", False, "softmax", False, False, False, 1),
      ("dense_megatron", False, "sigmoid", True, False, False, 1),
      ("sparse_switch_non_roe", True, "softmax", False, False, False, 1),
      ("sparse_megatron_non_roe", True, "sigmoid", True, False, False, 1),
      ("sparse_switch_roe_topk_before_ag", True, "softmax", False, True, True, 2),
      ("sparse_megatron_roe", True, "sigmoid", True, True, False, 2),
  )
  def test_routed_moe_unweighted_lb_loss_parity_and_zero_grad_leakage(
      self, sparse_matmul, score_func, use_megatron, use_roe, topk_before_ag, ep
  ):
    common = {
        "per_device_batch_size": 1.0,
        "max_target_length": 8,
        "base_emb_dim": 16,
        "base_moe_mlp_dim": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "ici_fsdp_parallelism": 4 // ep,
        "ici_expert_parallelism": ep,
        "sparse_matmul": sparse_matmul,
        "capacity_factor": -1.0,
        "routed_score_func": score_func,
        "moe_use_megatron_seq_aux_loss": use_megatron,
        "use_ring_of_experts": use_roe,
        "moe_topk_before_ep_all_gather": topk_before_ag,
    }
    cfg_off_w0 = _make_config(load_balance_loss_weight=0.0, moe_log_unweighted_lb_loss=False, **common)
    cfg_on_w0 = _make_config(load_balance_loss_weight=0.0, moe_log_unweighted_lb_loss=True, **common)
    cfg_off_w = _make_config(load_balance_loss_weight=0.01, moe_log_unweighted_lb_loss=False, **common)
    cfg_on_w = _make_config(load_balance_loss_weight=0.01, moe_log_unweighted_lb_loss=True, **common)
    mesh = _make_mesh(cfg_off_w0)

    def _build(cfg):
      return moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=nnx.Rngs(51),
      )

    m_off_w0 = _build(cfg_off_w0)
    m_on_w0 = _clone_module_with_config(_build, m_off_w0, cfg_on_w0)
    m_off_w = _clone_module_with_config(_build, m_off_w0, cfg_off_w)
    m_on_w = _clone_module_with_config(_build, m_off_w0, cfg_on_w)
    x = jax.random.normal(jax.random.PRNGKey(52), (4, 8, cfg_off_w0.emb_dim), dtype=jnp.float32)

    def _run(mod, inp):
      out, lb, _ = mod(inp)
      inter = nnx.pop(mod, nnx.Intermediate).to_pure_dict()
      total = jnp.sum(out) + (lb if lb is not None else 0.0)
      return total, (out, lb, inter)

    with jax.set_mesh(mesh), logical_axis_rules(cfg_off_w0.logical_axis_rules):
      (_, (out_off_w0, lb_off_w0, inter_off_w0)), grads_off_w0 = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(
          m_off_w0, x
      )
      (_, (out_on_w0, lb_on_w0, inter_on_w0)), grads_on_w0 = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(
          m_on_w0, x
      )
      (_, (out_off_w, lb_off_w, inter_off_w)), grads_off_w = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(
          m_off_w, x
      )
      (_, (out_on_w, lb_on_w, inter_on_w)), grads_on_w = nnx.value_and_grad(_run, argnums=(0, 1), has_aux=True)(m_on_w, x)

    # weight == 0.0: unweighted intermediate emitted ONLY when flag is ON; lb_loss stays None; zero grad leakage.
    self.assertIsNone(lb_off_w0)
    self.assertIsNone(lb_on_w0)
    self.assertNotIn("moe_lb_loss_unweighted", inter_off_w0)
    self.assertIn("moe_lb_loss_unweighted", inter_on_w0)
    unweighted_w0 = float(inter_on_w0["moe_lb_loss_unweighted"])
    self.assertGreater(unweighted_w0, 0.0)
    _assert_pytree_bit_equal(out_off_w0, out_on_w0)
    _assert_pytree_bit_equal(grads_off_w0, grads_on_w0)

    # weight == 0.01: unweighted == weighted / 0.01 == unweighted_w0; forward & grads unchanged vs flag OFF.
    self.assertNotIn("moe_lb_loss_unweighted", inter_off_w)
    self.assertIn("moe_lb_loss_unweighted", inter_on_w)
    self.assertIsNotNone(lb_off_w)
    self.assertIsNotNone(lb_on_w)
    np.testing.assert_array_equal(np.asarray(lb_off_w), np.asarray(lb_on_w))
    unweighted_w = float(inter_on_w["moe_lb_loss_unweighted"])
    np.testing.assert_allclose(unweighted_w, float(lb_on_w) / 0.01, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(unweighted_w0, unweighted_w, rtol=1e-5, atol=1e-5)
    _assert_pytree_bit_equal(out_off_w, out_on_w)
    _assert_pytree_bit_equal(grads_off_w, grads_on_w)

  def test_loss_fn_train_eval_step_and_grad_accum_unweighted_lb_loss(self):
    batch = _make_stub_batch(batch_size=2)
    cfg_off = _StubTrainCfg(load_balance_loss_weight=0.0, moe_log_unweighted_lb_loss=False)
    cfg_on = _StubTrainCfg(load_balance_loss_weight=0.0, moe_log_unweighted_lb_loss=True)

    m_off = _TinyMoETrainModel(cfg_off.vocab_size, 8, cfg_off, rngs=nnx.Rngs(61))
    m_on = _TinyMoETrainModel(cfg_on.vocab_size, 8, cfg_on, rngs=nnx.Rngs(61))

    # loss_fn parity + zero gradient leakage.
    (loss_off, aux_off), grads_off = nnx.value_and_grad(pre_train.loss_fn, argnums=0, has_aux=True)(
        m_off, cfg_off, batch, None, None, is_train=True
    )
    (loss_on, aux_on), grads_on = nnx.value_and_grad(pre_train.loss_fn, argnums=0, has_aux=True)(
        m_on, cfg_on, batch, None, None, is_train=True
    )
    np.testing.assert_array_equal(np.asarray(loss_off), np.asarray(loss_on))
    _assert_pytree_bit_equal(grads_off, grads_on)
    self.assertNotIn("moe_lb_loss_unweighted", aux_off)
    self.assertIn("moe_lb_loss_unweighted", aux_on)
    self.assertGreater(float(aux_on["moe_lb_loss_unweighted"]), 0.0)

    # eval_step emits evaluation/moe_lb_loss_unweighted only when ON.
    eval_m_off = _TinyMoETrainModel(cfg_off.vocab_size, 8, cfg_off, rngs=nnx.Rngs(61))
    eval_m_on = _TinyMoETrainModel(cfg_on.vocab_size, 8, cfg_on, rngs=nnx.Rngs(61))
    st_off = train_state_nnx.TrainStateNNX(eval_m_off, nnx.Optimizer(eval_m_off, optax.sgd(0.1), wrt=nnx.Param))
    st_on = train_state_nnx.TrainStateNNX(eval_m_on, nnx.Optimizer(eval_m_on, optax.sgd(0.1), wrt=nnx.Param))
    gd_off, ps_off = nnx.split(st_off)
    gd_on, ps_on = nnx.split(st_on)
    eval_off = pre_train.eval_step(gd_off, cfg_off, ps_off, batch)
    eval_on = pre_train.eval_step(gd_on, cfg_on, ps_on, batch)
    self.assertNotIn("evaluation/moe_lb_loss_unweighted", eval_off["scalar"])
    self.assertIn("evaluation/moe_lb_loss_unweighted", eval_on["scalar"])
    np.testing.assert_allclose(
        float(eval_on["scalar"]["evaluation/moe_lb_loss_unweighted"]),
        float(aux_on["moe_lb_loss_unweighted"]),
        rtol=1e-6,
    )

    # Gradient accumulation averages moe_lb_loss_unweighted over microbatches (does not sum, does not alter loss).
    ga_batch = _make_stub_batch(batch_size=4)
    cfg_ga = _StubTrainCfg(
        micro_batch_size_to_train_on=2,
        global_batch_size_to_train_on=4,
        gradient_accumulation_steps=2,
        load_balance_loss_weight=0.0,
        moe_log_unweighted_lb_loss=True,
    )
    m_ga = _TinyMoETrainModel(cfg_ga.vocab_size, 8, cfg_ga, rngs=nnx.Rngs(61))
    replicated = NamedSharding(m_ga.mesh, P())
    params_shardings = jax.tree.map(lambda _: replicated, nnx.state(m_ga, nnx.Param))
    _, aux_ga, _ = gradient_accumulation.gradient_accumulation_loss_and_grad(
        pre_train.loss_fn, cfg_ga, m_ga, None, params_shardings, ga_batch, None
    )
    # Compare with average of the two individual microbatches.
    mb0 = {k: v[:2] for k, v in ga_batch.items()}
    mb1 = {k: v[2:] for k, v in ga_batch.items()}
    cfg_mb = _StubTrainCfg(load_balance_loss_weight=0.0, moe_log_unweighted_lb_loss=True)
    _, aux_mb0 = pre_train.loss_fn(
        _TinyMoETrainModel(cfg_mb.vocab_size, 8, cfg_mb, rngs=nnx.Rngs(61)), cfg_mb, mb0, None, None, is_train=True
    )
    _, aux_mb1 = pre_train.loss_fn(
        _TinyMoETrainModel(cfg_mb.vocab_size, 8, cfg_mb, rngs=nnx.Rngs(61)), cfg_mb, mb1, None, None, is_train=True
    )
    expected_mean = 0.5 * (float(aux_mb0["moe_lb_loss_unweighted"]) + float(aux_mb1["moe_lb_loss_unweighted"]))
    np.testing.assert_allclose(float(aux_ga["moe_lb_loss_unweighted"]), expected_mean, rtol=1e-5)

  def test_metric_logger_train_and_eval_unweighted_lb_loss_logging(self):
    # pylint: disable=protected-access
    logger = metric_logger.MetricLogger.__new__(metric_logger.MetricLogger)
    logger.config = SimpleNamespace(
        num_experts=4,
        rampup_end_step=0,
        hide_profiler_step_metric=False,
        profiler="",
        skip_first_n_steps_for_profiler=0,
        profiler_steps=0,
        mtp_num_layers=0,
        use_indexer=False,
        log_step_diagnostics=False,
        enable_tensorboard=False,
        metrics_file=None,
        gcs_metrics=False,
        managed_mldiagnostics=False,
        enable_wandb=False,
        abort_on_nan_loss=False,
        abort_on_inf_loss=False,
        enable_mllog=False,
        target_eval_loss=0.0,
        elastic_enabled=False,
    )
    logger.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    logger._pending_eval_step_count = 0
    logger._eval_loop_start = None
    logger.start_step = 0

    base_train_scalars = {
        "learning/loss": 2.0,
        "learning/lm_loss": 2.0,
        "learning/perplexity": 7.389,
        "learning/total_weights": 16,
        "learning/moe_lb_loss": 0.0,
        "perf/step_time_seconds": 0.1,
        "perf/per_device_tflops_per_sec": 1.0,
        "perf/per_device_tokens_per_sec": 100.0,
    }
    with mock.patch.object(metric_logger.max_logging, "log") as mock_log:
      logger._log_training_metrics({"scalar": dict(base_train_scalars)}, step=0)
      msg_off = mock_log.call_args[0][0]
      self.assertNotIn("moe_lb_loss_unweighted", msg_off)

      with_unweighted = dict(base_train_scalars)
      with_unweighted["learning/moe_lb_loss_unweighted"] = 1.234567
      logger._log_training_metrics({"scalar": with_unweighted}, step=1)
      msg_on = mock_log.call_args[0][0]
      self.assertIn("moe_lb_loss_unweighted: 1.234567", msg_on)

    # Eval accumulation & finalization when evaluation/moe_lb_loss_unweighted is absent.
    logger.reset_eval_metrics()
    logger._pending_eval_step_count = 2
    logger._accumulate_eval_metrics({"scalar": {"evaluation/total_loss": 4.0, "evaluation/total_weights": 2.0}})
    logger._accumulate_eval_metrics({"scalar": {"evaluation/total_loss": 6.0, "evaluation/total_weights": 2.0}})
    with (
        mock.patch.object(metric_logger.max_logging, "log") as mock_log,
        mock.patch.object(metric_logger.mllog_utils, "eval_stop"),
    ):
      logger._finalize_eval_metrics(train_step=5)
      self.assertNotIn("eval/avg_moe_lb_loss_unweighted", logger.cumulative_eval_metrics["scalar"])
      self.assertNotIn("avg_moe_lb_loss_unweighted", mock_log.call_args[0][0])

    # Eval accumulation & finalization when evaluation/moe_lb_loss_unweighted is present.
    logger.reset_eval_metrics()
    logger._pending_eval_step_count = 2
    logger._accumulate_eval_metrics(
        {
            "scalar": {
                "evaluation/total_loss": 4.0,
                "evaluation/total_weights": 2.0,
                "evaluation/moe_lb_loss_unweighted": 1.2,
            }
        }
    )
    logger._accumulate_eval_metrics(
        {
            "scalar": {
                "evaluation/total_loss": 6.0,
                "evaluation/total_weights": 2.0,
                "evaluation/moe_lb_loss_unweighted": 1.8,
            }
        }
    )
    with (
        mock.patch.object(metric_logger.max_logging, "log") as mock_log,
        mock.patch.object(metric_logger.mllog_utils, "eval_stop"),
    ):
      logger._finalize_eval_metrics(train_step=6)
      np.testing.assert_allclose(logger.cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss_unweighted"], 1.5)
      self.assertIn("avg_moe_lb_loss_unweighted=1.500000", mock_log.call_args[0][0])


if __name__ == "__main__":
  absltest.main()
