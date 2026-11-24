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

"""Torch-free CPU unit tests for Inkling 2 Mini (`decoder_block: inkling2`).

Covers:
  1. Released `inkling2-mini.yml` config loading & negative Pydantic validators.
  2. `scan_layers=False` vs `scan_layers=True` (`param_scan_axis=0` and `1`)
     forward logits, gradients, and zero-dummy-param invariants.
  3. `remat_policy="none"` vs `"full"` forward & backward parity.
  4. Packed `decoder_segment_ids` & reset `decoder_positions` zero-leakage
     invariance across all 4 causal short convs (static + dynamic), relative
     position bias, log-scaling, and attention, plus padding NaN-safety and
     `unpadded_vocab_size` masking.
  5. `MODEL_MODE_PREFILL` + step-by-step `MODEL_MODE_AUTOREGRESSIVE` (48-slot /
     logical-visit KV + 4x short-conv cache across 3 loop passes) exact parity
     against `MODEL_MODE_TRAIN`.
  6. `forced_routed_experts` logical-visit router replay & `return_routed_experts`.
  7. 8-device virtual CPU sharding parity (`FSDP=2, TP=2, EP=2`) in both
     `shard_mode="auto"` and `shard_mode="explicit"`.
  8. FLOPs calculation (`calculate_tflops_training_per_device`) for 48 logical
     layer visits, sliding/full attention split, message gates, and GA scaling.

PyTorch/HuggingFace parity & `tiny-inkling2-mini` verification live in
`tests/unit/inkling2_vs_reference_test.py`.
"""

import os
import subprocess
import sys
import tempfile
import unittest

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
import optax
from optax.contrib._muon import MuonDimensionNumbers as mdn
import orbax.checkpoint as ocp
import pytest

from maxtext.common.common_types import (
    DecoderBlockType,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import moe
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import lora_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import muon_utils
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

pytestmark = [pytest.mark.cpu_only]

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

# Non-square toy dimensions that exercise every architectural branch (GQA,
# sliding + full attention, static + dynamic sconv, Block AttnRes, 3-pass loop
# with cross-pass message gates, Softcapped SwiGLU MoE, unpadded vocab masking).
_TOY_EMB_DIM = 64
_TOY_Q_HEADS = 4
_TOY_KV_HEADS = 2
_TOY_HEAD_DIM = 16
_TOY_MOE_MLP_DIM = 48
_TOY_NUM_EXPERTS = 8
_TOY_TOP_K = 2
_TOY_VOCAB_SIZE = 96
_TOY_UNPADDED_VOCAB_SIZE = 88
_TOY_NUM_LAYERS = 6
_TOY_ATTN_RES_BLOCK_SIZE = 2
_TOY_CORE_START = 2
_TOY_CORE_END = 4
_TOY_NUM_PASSES = 3
_TOY_GATE_RANK = 16
_TOY_D_REL = 8
_TOY_REL_EXTENT = 16
_TOY_SLIDING_WINDOW = 6
_TOY_LOG_SCALING_FLOOR = 4.0
_TOY_SEQ_LEN = 10


def _logical_visits(cfg) -> int:
  num_core = cfg.inkling2_core_end - cfg.inkling2_core_start
  return cfg.num_decoder_layers + (cfg.inkling2_num_passes - 1) * num_core


def _toy_config(**overrides):
  """Builds a real HyperParameters for inkling2-mini at non-square toy scale."""
  kwargs = {
      "model_name": "inkling2-mini",
      "override_model_config": True,
      "base_emb_dim": _TOY_EMB_DIM,
      "base_num_query_heads": _TOY_Q_HEADS,
      "base_num_kv_heads": _TOY_KV_HEADS,
      "head_dim": _TOY_HEAD_DIM,
      "base_mlp_dim": _TOY_MOE_MLP_DIM,
      "base_moe_mlp_dim": _TOY_MOE_MLP_DIM,
      "base_num_decoder_layers": _TOY_NUM_LAYERS,
      "attn_res_block_size": _TOY_ATTN_RES_BLOCK_SIZE,
      "inkling2_core_start": _TOY_CORE_START,
      "inkling2_core_end": _TOY_CORE_END,
      "inkling2_num_passes": _TOY_NUM_PASSES,
      "inkling2_message_gate_rank": _TOY_GATE_RANK,
      "inkling2_d_rel": _TOY_D_REL,
      "inkling2_rel_extent": _TOY_REL_EXTENT,
      "sliding_window_size": _TOY_SLIDING_WINDOW,
      "short_conv_kernel_size": 4,
      "inkling2_log_scaling_n_floor": _TOY_LOG_SCALING_FLOOR,
      "inkling2_log_scaling_alpha": 0.1,
      "vocab_size": _TOY_VOCAB_SIZE,
      "unpadded_vocab_size": _TOY_UNPADDED_VOCAB_SIZE,
      "num_experts": _TOY_NUM_EXPERTS,
      "num_experts_per_tok": _TOY_TOP_K,
      "shared_experts": 0,
      "sparse_matmul": False,
      "megablox": False,
      "capacity_factor": -1,
      "per_device_batch_size": 2,
      "max_target_length": _TOY_SEQ_LEN,
      "max_prefill_predict_length": 4,
      "scan_layers": False,
      "remat_policy": "none",
      "packing": False,
      "enable_checkpointing": False,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "dropout_rate": 0.0,
      "skip_jax_distributed_system": True,
      "hardware": "cpu",
  }
  kwargs.update(overrides)
  return pyconfig.initialize(["", _BASE_CONFIG_PATH], **kwargs)


def _make_mesh(cfg, devices=None):
  if devices is None:
    devices = jax.devices()[:1]
  return maxtext_utils.get_mesh_from_config(cfg, devices=devices)


def _randomize_inkling2_state(model, seed: int = 123):
  """Replaces zero/identity initializations with seeded non-trivial values.

  Ensures AttnRes queries, message queries, dynamic conv static weights,
  router biases, and norm scales do not trivially collapse during testing.
  """
  var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
  state = nnx.state(model, var_filter)
  rng = np.random.RandomState(seed)
  new_flat = []
  for path, var in nnx.to_flat_state(state):
    val = var.get_value()
    key = "/".join(str(p) for p in path)
    if key.endswith(("layernorm/scale", "embed_norm/scale", "decoder_norm/scale", "l2_clip_scale")):
      arr = 1.0 + 0.1 * rng.randn(*val.shape)
    elif key.endswith(("/query", "/bias", "/static_w", "/kernel")) and any(
        tag in key for tag in ("attention_input", "mlp_input", "final_input", "message_input", "sconv", "convolution")
    ):
      arr = 0.15 * rng.randn(*val.shape)
    else:
      arr = 0.08 * rng.randn(*val.shape)
    new_arr = jnp.asarray(arr, dtype=val.dtype)
    if hasattr(val, "sharding") and val.sharding is not None:
      new_arr = jax.device_put(new_arr, val.sharding)
    new_flat.append((path, var.replace(new_arr)))
  nnx.update(model, nnx.from_flat_state(new_flat))


def _copy_unscanned_to_scanned(unscanned_model, scanned_model, scanned_cfg):
  """Copies unscanned Inkling2 params & MoEBiasVar into the scanned layout."""
  block_size = scanned_cfg.attn_res_block_size
  core_start = scanned_cfg.inkling2_core_start
  core_end = scanned_cfg.inkling2_core_end
  num_core_blocks = (core_end - core_start) // block_size

  var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
  src = {tuple(map(str, k)): v for k, v in nnx.to_flat_state(nnx.state(unscanned_model, var_filter))}
  dst_state = nnx.state(scanned_model, var_filter)
  new_flat = []
  for path, var in nnx.to_flat_state(dst_state):
    key = tuple(map(str, path))
    if "scanned_blocks" in key:
      idx = key.index("scanned_blocks")
      sub_name = key[idx + 1]
      # Supports both `sub_layer_{s}` (contract) and `layers_{s}` naming.
      sub_idx = int(sub_name.rsplit("_", 1)[-1])
      slices = [
          src[key[:idx] + (f"layers_{core_start + b * block_size + sub_idx}",) + key[idx + 2 :]].get_value()
          for b in range(num_core_blocks)
      ]
      axis = scanned_cfg.param_scan_axis if isinstance(var, nnx.Param) else 0
      value = jnp.stack(slices, axis=axis)
    else:
      value = src[key].get_value()
    target_val = var.get_value()
    assert value.shape == target_val.shape, f"Shape mismatch at {key}: {value.shape} vs {target_val.shape}"
    if hasattr(target_val, "sharding") and target_val.sharding is not None:
      value = jax.device_put(value, target_val.sharding)
    new_flat.append((path, var.replace(value)))
  nnx.update(scanned_model, nnx.from_flat_state(new_flat))


def _copy_state_between_models(src_model, dst_model):
  """Copies Param and MoEBiasVar values across identically structured models."""
  var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
  src = {tuple(map(str, k)): v.get_value() for k, v in nnx.to_flat_state(nnx.state(src_model, var_filter))}
  dst_state = nnx.state(dst_model, var_filter)
  new_flat = []
  for path, var in nnx.to_flat_state(dst_state):
    key = tuple(map(str, path))
    value = src[key]
    target_val = var.get_value()
    if hasattr(target_val, "sharding") and target_val.sharding is not None:
      value = jax.device_put(value, target_val.sharding)
    new_flat.append((path, var.replace(value)))
  nnx.update(dst_model, nnx.from_flat_state(new_flat))


def _unstack_scanned_grads(scanned_grads, scanned_cfg):
  """Unstacks scanned_blocks gradients back to flat unscanned `layers_{i}` paths."""
  block_size = scanned_cfg.attn_res_block_size
  core_start = scanned_cfg.inkling2_core_start
  core_end = scanned_cfg.inkling2_core_end
  num_core_blocks = (core_end - core_start) // block_size
  axis = scanned_cfg.param_scan_axis

  out = {}
  for path, var in nnx.to_flat_state(scanned_grads):
    key = tuple(map(str, path))
    arr = np.asarray(var.get_value(), dtype=np.float32)
    if "scanned_blocks" in key:
      idx = key.index("scanned_blocks")
      sub_idx = int(key[idx + 1].rsplit("_", 1)[-1])
      for b in range(num_core_blocks):
        unscanned_key = key[:idx] + (f"layers_{core_start + b * block_size + sub_idx}",) + key[idx + 2 :]
        out[unscanned_key] = np.take(arr, b, axis=axis)
    else:
      out[key] = arr
  return out


def _run_forward(model, cfg, mesh, *, tokens, positions, segment_ids=None, **kwargs):
  """Runs a single deterministic forward pass under the given mesh and axis rules."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    return model(
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
        **kwargs,
    )


def _ce_loss_and_grads(model, cfg, mesh, *, tokens, positions, targets, segment_ids=None):
  """Computes standard CE loss over valid tokens and per-Param gradients."""

  def _loss_fn(m):
    logits = m(
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        model_mode=MODEL_MODE_TRAIN,
        enable_dropout=False,
    )
    one_hot = jax.nn.one_hot(targets, cfg.vocab_size, dtype=jnp.float32)
    xent, _ = max_utils.cross_entropy_with_logits(logits.astype(jnp.float32), one_hot, z_loss=0.0)
    if segment_ids is not None:
      mask = (segment_ids != 0).astype(jnp.float32)
      loss = jnp.sum(xent * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
      loss = jnp.mean(xent)
    return loss, logits

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    (loss, logits), grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)(model)
  return loss, logits, grads


def _make_inputs(batch: int, seq_len: int, vocab_limit: int, seed: int = 7):
  rng = np.random.RandomState(seed)
  tokens = jnp.asarray(rng.randint(1, vocab_limit, size=(batch, seq_len)), dtype=jnp.int32)
  targets = jnp.asarray(rng.randint(1, vocab_limit, size=(batch, seq_len)), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch, seq_len))
  segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
  return tokens, positions, targets, segment_ids


# ==============================================================================
# 1. Config & Validator Tests
# ==============================================================================


class Inkling2ConfigTest(unittest.TestCase):
  """Released `inkling2-mini.yml` config loading and negative validator checks."""

  def test_released_inkling2_mini_config_loads(self):
    cfg = pyconfig.initialize(
        ["", _BASE_CONFIG_PATH],
        model_name="inkling2-mini",
        skip_jax_distributed_system=True,
        hardware="cpu",
    )
    self.assertEqual(cfg.decoder_block, DecoderBlockType.INKLING2)
    self.assertEqual(cfg.num_decoder_layers, 24)
    self.assertEqual(cfg.emb_dim, 2304)
    self.assertEqual(cfg.num_query_heads, 24)
    self.assertEqual(cfg.num_kv_heads, 8)
    self.assertEqual(cfg.head_dim, 128)
    self.assertEqual(cfg.moe_mlp_dim, 1152)
    self.assertEqual(cfg.vocab_size, 201024)
    self.assertEqual(cfg.unpadded_vocab_size, 200058)
    self.assertFalse(cfg.logits_via_embedding)
    self.assertAlmostEqual(cfg.final_logits_soft_cap, 30.0)
    self.assertAlmostEqual(cfg.normalization_layer_epsilon, 1e-6)
    self.assertEqual(cfg.attn_res_block_size, 6)
    self.assertEqual(cfg.inkling2_core_start, 6)
    self.assertEqual(cfg.inkling2_core_end, 18)
    self.assertEqual(cfg.inkling2_num_passes, 3)
    self.assertEqual(cfg.inkling2_message_gate_rank, 64)
    self.assertEqual(cfg.sliding_window_size, 1024)
    self.assertEqual(cfg.short_conv_kernel_size, 4)
    self.assertEqual(cfg.inkling2_d_rel, 16)
    self.assertEqual(cfg.inkling2_rel_extent, 1024)
    self.assertAlmostEqual(cfg.inkling2_qk_l2_clip_threshold, 22.627416997969522)
    self.assertAlmostEqual(cfg.inkling2_rel_l2_clip_threshold, 8.0)
    self.assertAlmostEqual(cfg.inkling2_log_scaling_n_floor, 8192.0)
    self.assertAlmostEqual(cfg.inkling2_log_scaling_alpha, 0.1)
    self.assertEqual(cfg.num_experts, 128)
    self.assertEqual(cfg.num_experts_per_tok, 8)
    self.assertEqual(cfg.shared_experts, 0)
    self.assertEqual(list(cfg.mlp_activations), ["situ_gate", "situ_linear"])
    self.assertAlmostEqual(cfg.mlp_gate_softcap, 4.0)
    self.assertAlmostEqual(cfg.mlp_up_softcap, 25.0)
    self.assertEqual(cfg.routed_score_func, "sigmoid")
    self.assertAlmostEqual(cfg.routed_scaling_factor, 8.0)
    self.assertTrue(cfg.routed_bias)
    self.assertEqual(cfg.gate_matmul_precision, "highest")
    self.assertTrue(cfg.inkling2_contiguous_segment_positions)
    self.assertEqual(_logical_visits(cfg), 48)

  def test_negative_config_validators(self):
    # Verify the baseline toy config is valid first so negative checks cannot pass vacuously.
    base_cfg = _toy_config()
    self.assertEqual(base_cfg.decoder_block, DecoderBlockType.INKLING2)

    invalid_cases = [
        ("layers_not_divisible_by_block", {"base_num_decoder_layers": 7, "attn_res_block_size": 2}, "divisible"),
        ("core_start_zero", {"inkling2_core_start": 0, "inkling2_core_end": 4}, "0 < inkling2_core_start"),
        ("core_start_not_multiple", {"inkling2_core_start": 1, "inkling2_core_end": 4}, "divisible"),
        ("core_end_not_multiple", {"inkling2_core_start": 2, "inkling2_core_end": 5}, "divisible"),
        ("core_start_ge_core_end", {"inkling2_core_start": 4, "inkling2_core_end": 4}, "0 < inkling2_core_start"),
        ("core_end_gt_layers", {"inkling2_core_start": 2, "inkling2_core_end": 8}, "0 < inkling2_core_start"),
        ("unpadded_vocab_gt_vocab", {"vocab_size": 96, "unpadded_vocab_size": 100}, "unpadded_vocab_size"),
        ("explicit_ici_cp", {"shard_mode": "explicit", "ici_context_parallelism": 2}, "context parallelism"),
        ("explicit_dcn_cp", {"shard_mode": "explicit", "dcn_context_parallelism": 2}, "context parallelism"),
        ("auto_ici_cp", {"shard_mode": "auto", "ici_context_parallelism": 2}, "context parallelism"),
        ("auto_dcn_cp", {"shard_mode": "auto", "dcn_context_parallelism": 2}, "context parallelism"),
        (
            "auto_ulysses_cp",
            {"shard_mode": "auto", "ici_context_usp_ulysses_parallelism": 2},
            "context parallelism",
        ),
        (
            "routed_bias_update_rate_positive",
            {"routed_bias_update_rate": 1e-3},
            "routed_bias_update_rate|Loss-free load balancing",
        ),
        ("use_chunked_prefill_true", {"use_chunked_prefill": True}, "use_chunked_prefill"),
    ]
    for name, bad_overrides, err_regex in invalid_cases:
      with self.subTest(case=name):
        with self.assertRaisesRegex(ValueError, err_regex):
          _toy_config(**bad_overrides)

  def test_rl_config_non_inkling_forward_and_inkling2_init(self):
    """RLConfig runs forward cleanly on non-Inkling models and validates inkling2-mini."""
    rl_yml = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "rl.yml")

    # 1. Non-Inkling model (`llama3.1-8b`) via RLConfig must run forward without AttributeError.
    cfg_llama = pyconfig.initialize_pydantic(
        ["", rl_yml],
        config_class=config_types.RLConfig,
        run_name="inkling2_rl_llama_regression",
        model_name="llama3.1-8b",
        override_model_config=True,
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        hardware="cpu",
        attention="dot_product",
        base_num_decoder_layers=1,
        base_emb_dim=64,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        base_mlp_dim=128,
        vocab_size=64,
        max_target_length=8,
        max_prefill_predict_length=4,
        per_device_batch_size=1,
        dtype="float32",
        weight_dtype="float32",
    )
    self.assertEqual(cfg_llama.unpadded_vocab_size, 0)
    mesh = _make_mesh(cfg_llama)
    llama_model = model_creation_utils.from_config(cfg_llama, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    toks, pos, _, seg = _make_inputs(1, 8, 64, seed=5)
    logits = _run_forward(llama_model, cfg_llama, mesh, tokens=toks, positions=pos, segment_ids=seg)
    self.assertEqual(tuple(logits.shape), (1, 8, 64))
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))

    # 2. Inkling 2 Mini via RLConfig loads Inkling2 fields and enforces Inkling2 validators.
    cfg_ink_rl = pyconfig.initialize_pydantic(
        ["", rl_yml],
        config_class=config_types.RLConfig,
        run_name="inkling2_rl_init",
        model_name="inkling2-mini",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        hardware="cpu",
    )
    self.assertEqual(cfg_ink_rl.decoder_block, DecoderBlockType.INKLING2)
    self.assertEqual(cfg_ink_rl.unpadded_vocab_size, 200058)
    self.assertAlmostEqual(cfg_ink_rl.mlp_gate_softcap, 4.0)
    self.assertAlmostEqual(cfg_ink_rl.mlp_up_softcap, 25.0)
    self.assertEqual(cfg_ink_rl.gate_matmul_precision, "highest")
    with self.assertRaisesRegex(ValueError, "context parallelism"):
      pyconfig.initialize_pydantic(
          ["", rl_yml],
          config_class=config_types.RLConfig,
          run_name="inkling2_rl_cp_reject",
          model_name="inkling2-mini",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          ici_context_parallelism=2,
      )
    with self.assertRaisesRegex(ValueError, "routed_bias_update_rate"):
      pyconfig.initialize_pydantic(
          ["", rl_yml],
          config_class=config_types.RLConfig,
          run_name="inkling2_rl_bias_reject",
          model_name="inkling2-mini",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          hardware="cpu",
          routed_bias_update_rate=1e-3,
      )


# ==============================================================================
# 2. Core Model Invariants (Scan, Remat, Packing, Prefill+AR, Forced Routing)
# ==============================================================================


class Inkling2ModelTest(parameterized.TestCase):
  """Torch-free synthetic Inkling 2 forward, backward, packing, and decode tests."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._prev_matmul = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")

  @classmethod
  def tearDownClass(cls):
    jax.config.update("jax_default_matmul_precision", cls._prev_matmul)
    super().tearDownClass()

  @parameterized.named_parameters(
      ("one_core_block_axis0", 6, 2, 4, 0),
      ("one_core_block_axis1", 6, 2, 4, 1),
      ("two_core_blocks_axis0", 8, 2, 6, 0),
      ("two_core_blocks_axis1", 8, 2, 6, 1),
  )
  def test_scan_layers_equivalence_forward_and_grad(self, num_layers, core_start, core_end, param_scan_axis):
    """`scan_layers=False` vs `True` must match in logits & every leaf grad."""
    common = {
        "base_num_decoder_layers": num_layers,
        "inkling2_core_start": core_start,
        "inkling2_core_end": core_end,
        "param_scan_axis": param_scan_axis,
    }
    cfg_u = _toy_config(scan_layers=False, **common)
    cfg_s = _toy_config(scan_layers=True, **common)
    mesh = _make_mesh(cfg_u)

    model_u = model_creation_utils.from_config(cfg_u, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    model_s = model_creation_utils.from_config(cfg_s, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    _randomize_inkling2_state(model_u, seed=42)
    _copy_unscanned_to_scanned(model_u, model_s, cfg_s)

    tokens, positions, targets, seg = _make_inputs(2, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=11)
    loss_u, logits_u, grads_u = _ce_loss_and_grads(
        model_u, cfg_u, mesh, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )
    loss_s, logits_s, grads_s = _ce_loss_and_grads(
        model_s, cfg_s, mesh, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )

    np.testing.assert_allclose(np.asarray(loss_s), np.asarray(loss_u), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(logits_s)[..., :_TOY_UNPADDED_VOCAB_SIZE],
        np.asarray(logits_u)[..., :_TOY_UNPADDED_VOCAB_SIZE],
        rtol=1e-4,
        atol=1e-4,
    )

    flat_u = {tuple(map(str, p)): np.asarray(v.get_value(), dtype=np.float32) for p, v in nnx.to_flat_state(grads_u)}
    flat_s = _unstack_scanned_grads(grads_s, cfg_s)
    self.assertEqual(set(flat_u.keys()), set(flat_s.keys()), "Unscanned vs scanned Param keys differ")

    # Verify zero dummy/dead Param leaves: every active Param leaf must have finite non-zero grad.
    # Note: when `num_core_blocks == 1`, `message_input` attends over only 1 source (`prev_pass_y`),
    # so its 1-element softmax is identically 1.0 and `message_input.query` has exact mathematical 0 grad.
    num_core_blocks = (core_end - core_start) // cfg_s.attn_res_block_size
    for key, gu in flat_u.items():
      gs = flat_s[key]
      self.assertTrue(np.all(np.isfinite(gu)), f"Non-finite unscanned grad at {key}")
      self.assertTrue(np.all(np.isfinite(gs)), f"Non-finite scanned grad at {key}")
      np.testing.assert_allclose(gs, gu, rtol=1e-4, atol=1e-4, err_msg=f"Grad mismatch at {key}")
      if num_core_blocks == 1 and key == ("decoder", "message_input", "query"):
        self.assertEqual(float(np.linalg.norm(gu)), 0.0)
      else:
        self.assertGreater(np.linalg.norm(gu), 0.0, f"Zero/dead gradient on Param leaf {key}")

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_remat_none_vs_full_forward_and_grad(self, scan_layers):
    """`remat_policy='none'` vs `'full'` must produce identical logits & grads."""
    cfg_none = _toy_config(scan_layers=scan_layers, remat_policy="none")
    cfg_full = _toy_config(scan_layers=scan_layers, remat_policy="full")
    mesh = _make_mesh(cfg_none)

    m_none = model_creation_utils.from_config(cfg_none, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m_full = model_creation_utils.from_config(cfg_full, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    _randomize_inkling2_state(m_none, seed=19)
    _copy_state_between_models(m_none, m_full)

    tokens, positions, targets, seg = _make_inputs(2, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=21)
    loss_n, logits_n, grads_n = _ce_loss_and_grads(
        m_none, cfg_none, mesh, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )
    loss_f, logits_f, grads_f = _ce_loss_and_grads(
        m_full, cfg_full, mesh, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )

    np.testing.assert_allclose(np.asarray(loss_f), np.asarray(loss_n), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(logits_f), np.asarray(logits_n), rtol=1e-4, atol=1e-4)
    for (p_n, v_n), (p_f, v_f) in zip(nnx.to_flat_state(grads_n), nnx.to_flat_state(grads_f), strict=True):
      self.assertEqual(p_n, p_f)
      np.testing.assert_allclose(
          np.asarray(v_f.get_value(), dtype=np.float32),
          np.asarray(v_n.get_value(), dtype=np.float32),
          rtol=1e-4,
          atol=1e-4,
          err_msg=f"Remat grad mismatch at {p_n}",
      )

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_packed_segment_ids_and_positions_invariance(self, scan_layers):
    """Packed multi-doc + padding must have ZERO cross-doc leakage and ZERO NaNs."""
    # Length 14 packs doc1 (len 7 > sliding_window=6 & > log_scaling_floor=4),
    # doc2 (len 5 > conv_kernel=4), and 2 right-padding slots.
    packed_len = 14
    cfg = _toy_config(scan_layers=scan_layers, per_device_batch_size=1, max_target_length=packed_len)
    mesh = _make_mesh(cfg)
    model = model_creation_utils.from_config(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    _randomize_inkling2_state(model, seed=31)

    doc1_toks = jnp.array([[3, 17, 42, 8, 55, 71, 12]], dtype=jnp.int32)
    doc2_toks = jnp.array([[61, 5, 29, 80, 14]], dtype=jnp.int32)
    pad_a = jnp.array([[1, 2]], dtype=jnp.int32)
    pad_b = jnp.array([[77, 85]], dtype=jnp.int32)

    packed_toks_a = jnp.concatenate([doc1_toks, doc2_toks, pad_a], axis=1)
    packed_toks_b = jnp.concatenate([doc1_toks, doc2_toks, pad_b], axis=1)
    packed_pos = jnp.array([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 0, 0]], dtype=jnp.int32)
    packed_seg = jnp.array([[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0]], dtype=jnp.int32)

    # Unpacked single-doc runs padded to same length with seg=0 tail.
    def _pad_single(doc_toks):
      n = doc_toks.shape[1]
      toks = jnp.pad(doc_toks, ((0, 0), (0, packed_len - n)), constant_values=0)
      pos = jnp.pad(jnp.arange(n, dtype=jnp.int32)[None, :], ((0, 0), (0, packed_len - n)), constant_values=0)
      seg = jnp.pad(jnp.ones((1, n), dtype=jnp.int32), ((0, 0), (0, packed_len - n)), constant_values=0)
      return _run_forward(model, cfg, mesh, tokens=toks, positions=pos, segment_ids=seg, model_mode=MODEL_MODE_TRAIN)

    logits_packed_a = np.asarray(
        _run_forward(
            model,
            cfg,
            mesh,
            tokens=packed_toks_a,
            positions=packed_pos,
            segment_ids=packed_seg,
            model_mode=MODEL_MODE_TRAIN,
        )
    )
    logits_packed_b = np.asarray(
        _run_forward(
            model,
            cfg,
            mesh,
            tokens=packed_toks_b,
            positions=packed_pos,
            segment_ids=packed_seg,
            model_mode=MODEL_MODE_TRAIN,
        )
    )
    logits_doc1 = np.asarray(_pad_single(doc1_toks))
    logits_doc2 = np.asarray(_pad_single(doc2_toks))

    # Zero NaNs anywhere — including on padded query positions (seg == 0).
    self.assertTrue(np.all(np.isfinite(logits_packed_a)), "Non-finite logits on packed input with padding")
    # Padded vocab logits [unpadded_vocab_size:] must be masked to -1e9.
    np.testing.assert_allclose(logits_packed_a[..., _TOY_UNPADDED_VOCAB_SIZE:], -1e9, rtol=0, atol=1e-3)

    # Doc 1 and Doc 2 inside packed sequence must exactly match unpacked single docs.
    v = _TOY_UNPADDED_VOCAB_SIZE
    np.testing.assert_allclose(logits_packed_a[:, :7, :v], logits_doc1[:, :7, :v], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(logits_packed_a[:, 7:12, :v], logits_doc2[:, :5, :v], rtol=1e-4, atol=1e-4)
    # Mutating padded tokens must not perturb valid doc 1 or doc 2 logits.
    np.testing.assert_allclose(logits_packed_a[:, :12, :v], logits_packed_b[:, :12, :v], rtol=1e-5, atol=1e-5)

    # Backward pass with padded queries must also have zero NaNs/Infs.
    targets = jnp.roll(packed_toks_a, -1, axis=1)
    loss, _, grads = _ce_loss_and_grads(
        model, cfg, mesh, tokens=packed_toks_a, positions=packed_pos, targets=targets, segment_ids=packed_seg
    )
    self.assertTrue(bool(jnp.isfinite(loss)), "Non-finite loss with packed + padded queries")
    for path, var in nnx.to_flat_state(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(var.get_value()))), f"Non-finite grad with padding at {path}")

  @parameterized.named_parameters(
      ("one_core_block_unscanned_b1", 6, 2, 4, False, 1),
      ("two_core_blocks_unscanned_b1", 8, 2, 6, False, 1),
      ("two_core_blocks_scanned_b2", 8, 2, 6, True, 2),
  )
  def test_prefill_and_autoregressive_decode_match_train(self, num_layers, core_start, core_end, scan_layers, batch_size):
    """Logical-visit KV + 4x conv cache prefill+AR matches TRAIN (unscanned/scanned, B=1/B=2)."""
    total_len = 10  # Crosses sliding_window=6 and log_scaling_n_floor=4!
    prompt_len = 4
    prefill_pad = 6
    common = {
        "base_num_decoder_layers": num_layers,
        "inkling2_core_start": core_start,
        "inkling2_core_end": core_end,
        "scan_layers": scan_layers,
        "per_device_batch_size": batch_size,
        "max_target_length": total_len,
        "max_prefill_predict_length": prefill_pad,
    }
    cfg_train = _toy_config(**common)
    cfg_dec = _toy_config(**common)
    mesh = _make_mesh(cfg_train)

    m_train = model_creation_utils.from_config(cfg_train, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m_dec = model_creation_utils.from_config(cfg_dec, mesh=mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(1))
    _randomize_inkling2_state(m_train, seed=55)
    _copy_state_between_models(m_train, m_dec)

    # Verify all 8 Inkling2DecodeCache fields are sized for `batch_size` and carry `cache_batch` sharding metadata.
    cache_state = nnx.state(m_dec.decoder.inkling2_cache, nnx.Cache)
    cache_leaves = list(nnx.to_flat_state(cache_state))
    self.assertEqual(len(cache_leaves), 8)
    for path, var in cache_leaves:
      self.assertEqual(var.get_value().shape[1], batch_size, f"Cache batch size mismatch at {path}")
      sharding_axes = var.get_metadata().get("out_sharding") or var.get_metadata().get("sharding")
      self.assertIsNotNone(sharding_axes, f"Missing logical out_sharding on cache leaf {path}")
      self.assertIn("cache_batch", sharding_axes, f"Missing cache_batch axis on cache leaf {path}")

    tokens, positions, _, seg = _make_inputs(batch_size, total_len, _TOY_UNPADDED_VOCAB_SIZE, seed=56)
    ref_logits = np.asarray(
        _run_forward(
            m_train, cfg_train, mesh, tokens=tokens, positions=positions, segment_ids=seg, model_mode=MODEL_MODE_TRAIN
        )
    )

    # Prefill with padded prompt [prompt_len valid + (prefill_pad - prompt_len) pad].
    p_toks = jnp.pad(tokens[:, :prompt_len], ((0, 0), (0, prefill_pad - prompt_len)), constant_values=0)
    p_pos = jnp.broadcast_to(jnp.arange(prefill_pad, dtype=jnp.int32)[None, :], (batch_size, prefill_pad))
    p_seg = jnp.broadcast_to((jnp.arange(prefill_pad) < prompt_len).astype(jnp.int32)[None, :], (batch_size, prefill_pad))
    prefill_logits = np.asarray(
        _run_forward(
            m_dec,
            cfg_dec,
            mesh,
            tokens=p_toks,
            positions=p_pos,
            segment_ids=p_seg,
            model_mode=MODEL_MODE_PREFILL,
            true_length=prompt_len,
        )
    )
    v = _TOY_UNPADDED_VOCAB_SIZE
    np.testing.assert_allclose(
        prefill_logits[:, :prompt_len, :v],
        ref_logits[:, :prompt_len, :v],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Prefill logits mismatch vs TRAIN",
    )

    # Step-by-step autoregressive decode across the sliding window boundary.
    # `Transformer.__call__` requires `decoder_segment_ids=None` in AUTOREGRESSIVE mode.
    for t in range(prompt_len, total_len):
      step_tok = tokens[:, t : t + 1]
      step_pos = jnp.full((batch_size, 1), t, dtype=jnp.int32)
      ar_logits = np.asarray(
          _run_forward(
              m_dec,
              cfg_dec,
              mesh,
              tokens=step_tok,
              positions=step_pos,
              segment_ids=None,
              model_mode=MODEL_MODE_AUTOREGRESSIVE,
          )
      )
      np.testing.assert_allclose(
          ar_logits[:, 0, :v],
          ref_logits[:, t, :v],
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"Autoregressive decode mismatch at step t={t}",
      )

  def test_decode_cache_error_guards(self):
    """AR decode rejects `decoder_positions=None` and cache rejects batch > capacity."""
    cfg = _toy_config(per_device_batch_size=1, max_target_length=8, max_prefill_predict_length=4)
    mesh = _make_mesh(cfg)
    m_dec = model_creation_utils.from_config(cfg, mesh=mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))

    # 1. Batch exceeding Inkling2DecodeCache capacity must raise ValueError (not silently truncate/pad).
    big_toks, big_pos, _, big_seg = _make_inputs(3, 4, _TOY_UNPADDED_VOCAB_SIZE, seed=61)
    with self.assertRaisesRegex(ValueError, "batch capacity.*exceeded"):
      _run_forward(
          m_dec,
          cfg,
          mesh,
          tokens=big_toks,
          positions=big_pos,
          segment_ids=big_seg,
          model_mode=MODEL_MODE_PREFILL,
      )

    # 2. Autoregressive decode with `decoder_positions=None` must raise ValueError.
    step_tok = jnp.array([[5]], dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "requires explicit.*decoder_positions"):
      _run_forward(
          m_dec,
          cfg,
          mesh,
          tokens=step_tok,
          positions=None,
          segment_ids=None,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_forced_routed_experts_and_return_routed_experts(self, scan_layers):
    """Verifies per-visit `forced_routed_experts` replay & `return_routed_experts`."""
    cfg = _toy_config(scan_layers=scan_layers, return_routed_experts=True, per_device_batch_size=1)
    mesh = _make_mesh(cfg)
    model = model_creation_utils.from_config(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    _randomize_inkling2_state(model, seed=77)

    tokens, positions, _, seg = _make_inputs(1, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=78)
    n_visits = _logical_visits(cfg)

    logits_unforced = np.asarray(
        _run_forward(model, cfg, mesh, tokens=tokens, positions=positions, segment_ids=seg, model_mode=MODEL_MODE_TRAIN)
    )

    # Verify NNXDecoder returns the 4-tuple with per-logical-visit routed experts `(n_visits, B, T, top_k)`.
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      dec_out = model.decoder(
          shared_embedding=model.token_embedder,
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
    self.assertIsInstance(dec_out, tuple)
    self.assertEqual(len(dec_out), 4, "Expected return_routed_experts=True to return 4-tuple from NNXDecoder")
    dec_logits, _, _, expert_indices = dec_out
    self.assertEqual(
        tuple(expert_indices.shape),
        (n_visits, 1, _TOY_SEQ_LEN, _TOY_TOP_K),
        f"Expected ({n_visits}, 1, {_TOY_SEQ_LEN}, {_TOY_TOP_K}) logical visits in expert_indices",
    )
    np.testing.assert_allclose(np.asarray(dec_logits), logits_unforced, rtol=1e-6, atol=1e-6)

    # Inject two distinct 4D forced routings [B, T, n_visits, top_k] and verify exact replay + distinct outputs.
    rng = np.random.RandomState(88)
    forced_a = jnp.asarray(
        rng.randint(0, _TOY_NUM_EXPERTS, size=(1, _TOY_SEQ_LEN, n_visits, _TOY_TOP_K)), dtype=jnp.int32
    )
    forced_b = (forced_a + 1) % _TOY_NUM_EXPERTS

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      logits_a, _, _, replayed_a = model.decoder(
          shared_embedding=model.token_embedder,
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          forced_routed_experts=forced_a,
      )
      logits_b, _, _, replayed_b = model.decoder(
          shared_embedding=model.token_embedder,
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          forced_routed_experts=forced_b,
      )
    np.testing.assert_array_equal(np.asarray(replayed_a), np.moveaxis(np.asarray(forced_a), 2, 0))
    np.testing.assert_array_equal(np.asarray(replayed_b), np.moveaxis(np.asarray(forced_b), 2, 0))
    logits_a = np.asarray(logits_a)
    logits_b = np.asarray(logits_b)

    v = _TOY_UNPADDED_VOCAB_SIZE
    self.assertGreater(float(np.max(np.abs(logits_a[..., :v] - logits_b[..., :v]))), 1e-3)
    self.assertGreater(float(np.max(np.abs(logits_a[..., :v] - logits_unforced[..., :v]))), 1e-3)

  def test_forced_routed_experts_non_zero_bias_uses_pre_bias_weights(self):
    """Non-zero `e_score_correction_bias` must not perturb weights under `forced_routed_experts`."""
    cfg = _toy_config(scan_layers=False, per_device_batch_size=1)
    mesh = _make_mesh(cfg)
    model = model_creation_utils.from_config(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    _randomize_inkling2_state(model, seed=101)
    mlp = model.decoder.layers_0.mlp

    rng = np.random.RandomState(102)
    x = jnp.asarray(rng.randn(1, 6, _TOY_EMB_DIM).astype(np.float32))
    forced = jnp.asarray(rng.randint(0, _TOY_NUM_EXPERTS, size=(1, 6, _TOY_TOP_K)), dtype=jnp.int32)

    bias_zero = jnp.zeros((_TOY_NUM_EXPERTS,), dtype=jnp.float32)
    bias_large = jnp.linspace(-50.0, 50.0, _TOY_NUM_EXPERTS, dtype=jnp.float32)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      mlp.gate.bias.set_value(bias_zero)
      gl_z, pb_z = mlp.gate(x)
      _, unforced_idx_z = mlp.get_topk(gl_z, pb_z)
      w_z, idx_z = mlp.get_topk(gl_z, pb_z, forced_routed_experts=forced)
      out_z, _, _ = mlp(x, forced_routed_experts=forced)

      mlp.gate.bias.set_value(bias_large)
      gl_b, pb_b = mlp.gate(x)
      _, unforced_idx_b = mlp.get_topk(gl_b, pb_b)
      w_b, idx_b = mlp.get_topk(gl_b, pb_b, forced_routed_experts=forced)
      out_b, _, _ = mlp(x, forced_routed_experts=forced)

    # Confirm the large bias actually steers unforced selection, proving the test is non-vacuous.
    self.assertFalse(bool(jnp.array_equal(unforced_idx_z, unforced_idx_b)))
    # Under forced routing, indices, gathered weights (from pre_bias_logits), and MLP output are invariant.
    np.testing.assert_array_equal(np.asarray(idx_z), np.asarray(forced))
    np.testing.assert_array_equal(np.asarray(idx_b), np.asarray(forced))
    np.testing.assert_allclose(np.asarray(w_b), np.asarray(w_z), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_b), np.asarray(out_z), rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_qwix_lora_under_full_remat_scanned_and_unscanned(self, scan_layers):
    """Qwix LoRA adapters survive `remat_policy='full'` and receive finite non-zero grads."""
    lora_regex = "decoder/(layers(_[0-9]+)?|scanned_blocks)/.*self_attention/(query|key|value|out)"
    cfg = _toy_config(
        scan_layers=scan_layers,
        remat_policy="full",
        per_device_batch_size=1,
        lora={
            "enable_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_module_path": lora_regex,
        },
    )
    mesh = _make_mesh(cfg)
    model = model_creation_utils.from_config(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    _randomize_inkling2_state(model, seed=111)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      lora_model = lora_utils.apply_lora_to_model(model, mesh, cfg)

    lora_state = nnx.state(lora_model, nnx.LoRAParam)
    lora_leaves = list(nnx.to_flat_state(lora_state))
    self.assertGreater(len(lora_leaves), 0, f"Expected LoRAParam leaves > 0 (scan={scan_layers})")
    lora_paths = ["/".join(str(p) for p in path) for path, _ in lora_leaves]
    self.assertTrue(any("layers_0" in p for p in lora_paths), "Missing prelude LoRAParam")
    if scan_layers:
      self.assertTrue(any("scanned_blocks" in p for p in lora_paths), "Missing scanned_blocks LoRAParam")

    tokens, positions, targets, seg = _make_inputs(1, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=112)

    def _lora_loss(m):
      logits = m(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg,
          model_mode=MODEL_MODE_TRAIN,
          enable_dropout=False,
      )
      one_hot = jax.nn.one_hot(targets, cfg.vocab_size, dtype=jnp.float32)
      xent, _ = max_utils.cross_entropy_with_logits(logits.astype(jnp.float32), one_hot, z_loss=0.0)
      return jnp.mean(xent)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      loss, lora_grads = nnx.value_and_grad(_lora_loss, argnums=nnx.DiffState(0, nnx.LoRAParam))(lora_model)

    self.assertTrue(bool(jnp.isfinite(loss)))
    grad_norms = []
    for path, var in nnx.to_flat_state(lora_grads):
      arr = np.asarray(var.get_value(), dtype=np.float32)
      self.assertTrue(np.all(np.isfinite(arr)), f"Non-finite LoRA grad at {path}")
      grad_norms.append(float(np.linalg.norm(arr)))
    self.assertTrue(any(n > 0.0 for n in grad_norms), "All LoRA adapter gradients were zero")

  def test_contiguous_positions_hlo_zero_scatter_gather_and_non_unit_stride(self):
    """Static `inkling2_contiguous_segment_positions=True` emits zero gather/scatter/cond; `False` honors strides."""
    cfg_fast = _toy_config(inkling2_contiguous_segment_positions=True, per_device_batch_size=1)
    cfg_gen = _toy_config(inkling2_contiguous_segment_positions=False, per_device_batch_size=1)
    mesh = _make_mesh(cfg_fast)

    m_fast = model_creation_utils.from_config(cfg_fast, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m_gen = model_creation_utils.from_config(cfg_gen, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    _randomize_inkling2_state(m_fast, seed=121)
    _copy_state_between_models(m_fast, m_gen)

    # 1. Verify lowered attention HLO (sliding & full) under `contiguous=True` has zero gather/scatter/conditional.
    x_dummy = jnp.ones((1, _TOY_SEQ_LEN, _TOY_EMB_DIM), dtype=jnp.float32)
    pos_dummy = jnp.arange(_TOY_SEQ_LEN, dtype=jnp.int32)[None, :]
    seg_dummy = jnp.ones((1, _TOY_SEQ_LEN), dtype=jnp.int32)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_fast.logical_axis_rules):
      for lyr_idx in (0, 1):
        attn = getattr(m_fast.decoder, f"layers_{lyr_idx}").self_attention
        gd, st = nnx.split(attn)

        def _attn_fn(state, qkv, seg_in, pos_in, _gd=gd):
          q, k, v = qkv
          return nnx.merge(_gd, state)(q, k, v, seg_in, pos_in, MODEL_MODE_TRAIN)[0]

        hlo_txt = jax.jit(_attn_fn).lower(st, (x_dummy, x_dummy, x_dummy), seg_dummy, pos_dummy).as_text()
        self.assertNotIn("conditional(", hlo_txt, f"Unexpected lax.cond in layer {lyr_idx} attention HLO")
        self.assertNotIn("gather(", hlo_txt, f"Unexpected gather in layer {lyr_idx} attention HLO")
        self.assertNotIn("scatter(", hlo_txt, f"Unexpected scatter in layer {lyr_idx} attention HLO")

      # Full model forward HLO must also contain zero `conditional(`.
      toks_d = jnp.ones((1, _TOY_SEQ_LEN), dtype=jnp.int32)
      gd_m, st_m = nnx.split(m_fast)

      def _full_fn(state, t_in, pos_in, seg_in, _gd=gd_m):
        return nnx.merge(_gd, state)(
            decoder_input_tokens=t_in,
            decoder_positions=pos_in,
            decoder_segment_ids=seg_in,
            model_mode=MODEL_MODE_TRAIN,
            enable_dropout=False,
        )

      full_hlo = jax.jit(_full_fn).lower(st_m, toks_d, pos_dummy, seg_dummy).as_text()
      self.assertNotIn("conditional(", full_hlo, "Unexpected lax.cond in full Inkling 2 training HLO")

    # 2. On standard contiguous packed segments, `contiguous=True` and `False` must match identically.
    toks, _, _, _ = _make_inputs(1, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=122)
    packed_pos = jnp.array([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=jnp.int32)
    packed_seg = jnp.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=jnp.int32)
    logits_fast_packed = np.asarray(
        _run_forward(m_fast, cfg_fast, mesh, tokens=toks, positions=packed_pos, segment_ids=packed_seg)
    )
    logits_gen_packed = np.asarray(
        _run_forward(m_gen, cfg_gen, mesh, tokens=toks, positions=packed_pos, segment_ids=packed_seg)
    )
    v = _TOY_UNPADDED_VOCAB_SIZE
    np.testing.assert_allclose(logits_fast_packed[..., :v], logits_gen_packed[..., :v], rtol=1e-5, atol=1e-5)

    # 3. On non-unit-stride custom positions, `contiguous=False` honors actual `p_i - p_j` gaps (diverges from `True`).
    strided_pos = jnp.array([[0, 2, 5, 9, 10, 14, 20, 21, 25, 30]], dtype=jnp.int32)
    ones_seg = jnp.ones((1, _TOY_SEQ_LEN), dtype=jnp.int32)
    logits_fast_strided = np.asarray(
        _run_forward(m_fast, cfg_fast, mesh, tokens=toks, positions=strided_pos, segment_ids=ones_seg)
    )
    logits_gen_strided = np.asarray(
        _run_forward(m_gen, cfg_gen, mesh, tokens=toks, positions=strided_pos, segment_ids=ones_seg)
    )
    self.assertTrue(np.all(np.isfinite(logits_gen_strided)))
    self.assertGreater(float(np.max(np.abs(logits_gen_strided[..., :v] - logits_fast_strided[..., :v]))), 1e-4)

  def test_muon_weight_dimension_numbers_scanned_and_unscanned(self):
    """`muon_utils.get_muon_weight_dimension_numbers` classifies all Inkling 2 params consistently."""
    cfg_u = _toy_config(scan_layers=False, muon_include_routers=True)
    cfg_s = _toy_config(scan_layers=True, param_scan_axis=0, muon_include_routers=True)
    cfg_u_norouter = _toy_config(scan_layers=False, muon_include_routers=False)
    cfg_s_norouter = _toy_config(scan_layers=True, param_scan_axis=0, muon_include_routers=False)
    mesh = _make_mesh(cfg_u)

    m_u = model_creation_utils.from_config(cfg_u, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m_s = model_creation_utils.from_config(cfg_s, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))

    dims_u = {
        tuple(map(str, p)): v for p, v in nnx.to_flat_state(muon_utils.get_muon_weight_dimension_numbers(m_u, cfg_u))
    }
    dims_s = {
        tuple(map(str, p)): v for p, v in nnx.to_flat_state(muon_utils.get_muon_weight_dimension_numbers(m_s, cfg_s))
    }
    dims_u_nr = {
        tuple(map(str, p)): v
        for p, v in nnx.to_flat_state(muon_utils.get_muon_weight_dimension_numbers(m_u, cfg_u_norouter))
    }
    dims_s_nr = {
        tuple(map(str, p)): v
        for p, v in nnx.to_flat_state(muon_utils.get_muon_weight_dimension_numbers(m_s, cfg_s_norouter))
    }

    # Expert MLP weights: [E, D1, D2] unscanned and [B_core, E, D1, D2] scanned both map to ((-2,), (-1,)).
    for leaf in ("wi_0", "wi_1", "wo"):
      self.assertEqual(dims_u[("decoder", "layers_2", "mlp", leaf)], mdn((-2,), (-1,)))
      self.assertEqual(dims_s[("decoder", "scanned_blocks", "sub_layer_0", "mlp", leaf)], mdn((-2,), (-1,)))

    # Router gate: included when muon_include_routers=True, excluded (None) when False.
    self.assertEqual(dims_u[("decoder", "layers_2", "mlp", "gate", "kernel")], mdn((0,), (-1,)))
    self.assertIsNotNone(dims_s[("decoder", "scanned_blocks", "sub_layer_0", "mlp", "gate", "kernel")])
    self.assertIsNone(dims_u_nr[("decoder", "layers_2", "mlp", "gate", "kernel")])
    self.assertIsNone(dims_s_nr[("decoder", "scanned_blocks", "sub_layer_0", "mlp", "gate", "kernel")])

    # Depthwise conv kernels/static_w and AttnRes queries must be excluded (None) in both layouts.
    for key in (
        ("decoder", "layers_2", "post_attention_convolution", "kernel"),
        ("decoder", "layers_2", "post_mlp_convolution", "kernel"),
        ("decoder", "layers_2", "self_attention", "k_sconv", "kernel"),
        ("decoder", "layers_2", "self_attention", "v_sconv", "kernel"),
        ("decoder", "layers_3", "self_attention", "k_sconv", "static_w"),
        ("decoder", "layers_3", "self_attention", "v_sconv", "static_w"),
        ("decoder", "layers_2", "attention_input", "query"),
        ("decoder", "layers_2", "mlp_input", "query"),
        ("decoder", "final_input", "query"),
        ("decoder", "message_input", "query"),
    ):
      self.assertIsNone(dims_u[key], f"Expected None Muon dim for {key}")

    # 3D projections (`gate_attn`, `rel_proj`, dynamic `weight_proj`) map to ((0,), (-2, -1)) unscanned.
    for key in (
        ("decoder", "layers_2", "self_attention", "gate_attn", "kernel"),
        ("decoder", "layers_2", "self_attention", "rel_proj", "kernel"),
        ("decoder", "layers_3", "self_attention", "k_sconv", "weight_proj", "kernel"),
        ("decoder", "layers_3", "self_attention", "v_sconv", "weight_proj", "kernel"),
    ):
      self.assertEqual(dims_u[key], mdn((0,), (-2, -1)), f"Unexpected Muon dim for {key}")

    # Verify scanned vs unscanned include/exclude consistency across every single core leaf.
    core_start = cfg_s.inkling2_core_start
    for key_s, val_s in dims_s.items():
      if "scanned_blocks" not in key_s:
        self.assertEqual(val_s is None, dims_u[key_s] is None, f"Mismatch at {key_s}")
        continue
      idx = key_s.index("scanned_blocks")
      sub_idx = int(key_s[idx + 1].rsplit("_", 1)[-1])
      key_u = key_s[:idx] + (f"layers_{core_start + sub_idx}",) + key_s[idx + 2 :]
      self.assertEqual(val_s is None, dims_u[key_u] is None, f"Scanned vs unscanned Muon mismatch: {key_s} vs {key_u}")

  def test_moe_lb_loss_scan_vs_unscan_parity(self):
    """`moe_lb_loss` with `load_balance_loss_weight=0.05` matches identically across scan/unscan."""
    common = {
        "base_num_decoder_layers": 8,
        "inkling2_core_start": 2,
        "inkling2_core_end": 6,
        "load_balance_loss_weight": 0.05,
        "per_device_batch_size": 1,
    }
    cfg_u = _toy_config(scan_layers=False, **common)
    cfg_s = _toy_config(scan_layers=True, **common)
    mesh = _make_mesh(cfg_u)

    m_u = model_creation_utils.from_config(cfg_u, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m_s = model_creation_utils.from_config(cfg_s, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    _randomize_inkling2_state(m_u, seed=131)
    _copy_unscanned_to_scanned(m_u, m_s, cfg_s)

    toks, pos, tgts, seg = _make_inputs(1, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=132)
    batch = {
        "inputs": toks,
        "inputs_position": pos,
        "inputs_segmentation": seg,
        "targets": tgts,
        "targets_position": pos,
        "targets_segmentation": seg,
    }
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      loss_u, aux_u = pre_train.loss_fn(m_u, cfg_u, dict(batch), None, None, is_train=True)
      loss_s, aux_s = pre_train.loss_fn(m_s, cfg_s, dict(batch), None, None, is_train=True)

    lb_u = float(aux_u["moe_lb_loss"])
    lb_s = float(aux_s["moe_lb_loss"])
    self.assertGreater(lb_u, 0.0, "Expected positive moe_lb_loss when load_balance_loss_weight=0.05")
    np.testing.assert_allclose(lb_s, lb_u, rtol=1e-5, atol=1e-5, err_msg="moe_lb_loss scan vs unscan mismatch")
    np.testing.assert_allclose(float(loss_s), float(loss_u), rtol=1e-5, atol=1e-5)

  def test_untied_final_logits_soft_cap_preserves_dtype(self):
    """Untied `final_logits_soft_cap` preserves bf16 when `cast_logits_to_fp32=False`."""
    common = {
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "logits_via_embedding": False,
        "final_logits_soft_cap": 30.0,
        "per_device_batch_size": 1,
    }
    cfg_bf16 = _toy_config(cast_logits_to_fp32=False, **common)
    cfg_fp32 = _toy_config(cast_logits_to_fp32=True, **common)
    mesh = _make_mesh(cfg_bf16)

    m_bf16 = model_creation_utils.from_config(cfg_bf16, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m_fp32 = model_creation_utils.from_config(cfg_fp32, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(1))
    _copy_state_between_models(m_bf16, m_fp32)

    toks, pos, _, seg = _make_inputs(1, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=141)
    logits_bf16 = _run_forward(m_bf16, cfg_bf16, mesh, tokens=toks, positions=pos, segment_ids=seg)
    logits_fp32 = _run_forward(m_fp32, cfg_fp32, mesh, tokens=toks, positions=pos, segment_ids=seg)

    self.assertEqual(logits_bf16.dtype, jnp.bfloat16)
    self.assertEqual(logits_fp32.dtype, jnp.float32)
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits_bf16))))
    v = _TOY_UNPADDED_VOCAB_SIZE
    self.assertLessEqual(float(jnp.max(jnp.abs(logits_bf16[..., :v]))), 30.0 + 1e-2)

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_message_gate_zero_init_one_to_three_pass_identity_and_step1_grads(self, scan_layers):
    """Zero-init message gates make 1-pass == 3-pass at init & Orbax warm-start, with all grads active by step 1."""
    # Use 2 core blocks (`core=2..6, block_size=2`) so `message_input.query` attends over 2 core blocks.
    common = {
        "base_num_decoder_layers": 8,
        "inkling2_core_start": 2,
        "inkling2_core_end": 6,
        "scan_layers": scan_layers,
        "per_device_batch_size": 2,
    }
    cfg_p1 = _toy_config(inkling2_num_passes=1, **common)
    cfg_p3 = _toy_config(inkling2_num_passes=3, **common)
    mesh = _make_mesh(cfg_p1)

    m1 = model_creation_utils.from_config(cfg_p1, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    m3 = model_creation_utils.from_config(cfg_p3, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    # 1. Verify exact zero-init on `forget_gate_proj` and `message_gate_proj` (and non-zero `down_proj`).
    zero_gate_count = 0
    down_proj_count = 0
    for path, var in nnx.to_flat_state(nnx.state(m3, nnx.Param)):
      key = "/".join(str(p) for p in path)
      val_np = np.asarray(var.get_value(), dtype=np.float32)
      if key.endswith(("forget_gate_proj/kernel", "message_gate_proj/kernel")):
        self.assertEqual(float(np.max(np.abs(val_np))), 0.0, f"Expected exact zero-init on {key}")
        zero_gate_count += 1
      elif key.endswith("message_gate/down_proj/kernel") or "_message_gate/down_proj/kernel" in key:
        self.assertGreater(float(np.linalg.norm(val_np)), 0.0, f"Expected non-zero init on {key}")
        down_proj_count += 1
    self.assertGreater(zero_gate_count, 0)
    self.assertGreater(down_proj_count, 0)

    # 2. Fresh init: 1-pass and 3-pass must produce identical logits.
    tokens, positions, targets, seg = _make_inputs(2, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=161)
    v = _TOY_UNPADDED_VOCAB_SIZE
    logits_p1_init = np.asarray(_run_forward(m1, cfg_p1, mesh, tokens=tokens, positions=positions, segment_ids=seg))
    logits_p3_init = np.asarray(_run_forward(m3, cfg_p3, mesh, tokens=tokens, positions=positions, segment_ids=seg))
    self.assertLessEqual(float(np.max(np.abs(logits_p3_init[..., :v] - logits_p1_init[..., :v]))), 1e-5)

    # 3. Train `m1` (1-pass) for 1 AdamW step, save Orbax checkpoint, restore into 3-pass `m3_warm`:
    #    1-pass trained model and warm-started 3-pass model must still produce identical logits!
    tx_p1 = optax.adamw(learning_rate=1e-2, weight_decay=0.01)
    opt_p1 = tx_p1.init(nnx.state(m1, nnx.Param))
    _, _, grads_p1 = _ce_loss_and_grads(
        m1, cfg_p1, mesh, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )
    p1_params = nnx.state(m1, nnx.Param)
    upd_p1, _ = tx_p1.update(grads_p1, opt_p1, p1_params)
    nnx.update(m1, optax.apply_updates(p1_params, upd_p1))

    with tempfile.TemporaryDirectory(prefix="inkling2_1to3_warmstart_") as tmpdir:
      ckpt_path = os.path.join(tmpdir, "step1_p1")
      ckptr = ocp.StandardCheckpointer()
      var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
      payload = jax.tree.map(lambda x: x, nnx.to_pure_dict(nnx.state(m1, var_filter)))
      ckptr.save(ckpt_path, payload)
      ckptr.wait_until_finished()
      restored = ckptr.restore(ckpt_path, target=payload)

    m3_warm = model_creation_utils.from_config(cfg_p3, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(7))
    warm_state = nnx.state(m3_warm, var_filter)
    nnx.replace_by_pure_dict(warm_state, restored)
    nnx.update(m3_warm, warm_state)

    logits_p1_step1 = np.asarray(_run_forward(m1, cfg_p1, mesh, tokens=tokens, positions=positions, segment_ids=seg))
    logits_p3_warm = np.asarray(_run_forward(m3_warm, cfg_p3, mesh, tokens=tokens, positions=positions, segment_ids=seg))
    self.assertLessEqual(float(np.max(np.abs(logits_p3_warm[..., :v] - logits_p1_step1[..., :v]))), 1e-5)

    # 4. Step 0 on `m3_warm`: `forget_gate_proj` & `message_gate_proj` receive non-zero grads immediately;
    #    after 1 AdamW step, Step 1 produces finite non-zero grads on EVERY Param leaf (incl. `down_proj`
    #    and `message_input.query`).
    tx_p3 = optax.adamw(learning_rate=1e-2, weight_decay=0.01)
    opt_p3 = tx_p3.init(nnx.state(m3_warm, nnx.Param))
    loss_s0, _, grads_s0 = _ce_loss_and_grads(
        m3_warm, cfg_p3, mesh, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )
    self.assertTrue(bool(jnp.isfinite(loss_s0)))
    step0_gate_nonzero = 0
    for path, var in nnx.to_flat_state(grads_s0):
      key = "/".join(str(p) for p in path)
      g_arr = np.asarray(var.get_value(), dtype=np.float32)
      self.assertTrue(np.all(np.isfinite(g_arr)), f"Non-finite step-0 grad at {key}")
      if key.endswith(("forget_gate_proj/kernel", "message_gate_proj/kernel")):
        self.assertGreater(float(np.linalg.norm(g_arr)), 0.0, f"Zero step-0 grad on {key}")
        step0_gate_nonzero += 1
    self.assertEqual(step0_gate_nonzero, zero_gate_count)

    p3_params = nnx.state(m3_warm, nnx.Param)
    upd_p3, _ = tx_p3.update(grads_s0, opt_p3, p3_params)
    nnx.update(m3_warm, optax.apply_updates(p3_params, upd_p3))

    tokens2, positions2, targets2, seg2 = _make_inputs(2, _TOY_SEQ_LEN, _TOY_UNPADDED_VOCAB_SIZE, seed=162)
    loss_s1, _, grads_s1 = _ce_loss_and_grads(
        m3_warm, cfg_p3, mesh, tokens=tokens2, positions=positions2, targets=targets2, segment_ids=seg2
    )
    self.assertTrue(bool(jnp.isfinite(loss_s1)))
    step1_down_nonzero = 0
    step1_msg_query_nonzero = False
    for path, var in nnx.to_flat_state(grads_s1):
      key = "/".join(str(p) for p in path)
      g_arr = np.asarray(var.get_value(), dtype=np.float32)
      self.assertTrue(np.all(np.isfinite(g_arr)), f"Non-finite step-1 grad at {key}")
      g_norm = float(np.linalg.norm(g_arr))
      self.assertGreater(g_norm, 0.0, f"Zero/dead step-1 grad on Param leaf {key}")
      if "_message_gate/down_proj/kernel" in key:
        step1_down_nonzero += 1
      if tuple(map(str, path)) == ("decoder", "message_input", "query"):
        step1_msg_query_nonzero = True
    self.assertEqual(step1_down_nonzero, down_proj_count)
    self.assertTrue(step1_msg_query_nonzero, "Missing decoder/message_input/query in step-1 grads")


# ==============================================================================
# 3. 8-Device Virtual CPU Sharding Parity (FSDP=2, TP=2, EP=2; auto & explicit)
# ==============================================================================


def _run_8dev_sharding_checks():
  """Runs 8-device FSDP=2, TP=2, EP=2 parity checks (invoked in 8-CPU subproc)."""
  jax.config.update("jax_default_matmul_precision", "highest")
  devices = jax.devices()
  assert len(devices) >= 8, f"Expected >= 8 CPU devices, got {len(devices)}"

  for scan_layers in (False, True):
    # 1-device unsharded reference (`global_batch=4, seq_len=8` so `per_device_batch_size=0.5`
    # on 8 devices yields `0.5 * 8 = 4`, divisible by `num_vocab_tiling=1` and `FSDP*EP=4`).
    sh_seq_len = 8
    cfg_ref = _toy_config(
        scan_layers=scan_layers,
        per_device_batch_size=4,
        max_target_length=sh_seq_len,
        sparse_matmul=True,
        megablox=False,
        capacity_factor=-1.0,
        ici_fsdp_parallelism=1,
        ici_tensor_parallelism=1,
        ici_expert_parallelism=1,
        shard_mode="auto",
    )
    mesh_ref = _make_mesh(cfg_ref, devices=devices[:1])
    m_ref = model_creation_utils.from_config(cfg_ref, mesh=mesh_ref, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    _randomize_inkling2_state(m_ref, seed=91)

    tokens, positions, targets, seg = _make_inputs(4, sh_seq_len, _TOY_UNPADDED_VOCAB_SIZE, seed=92)
    loss_ref, logits_ref, grads_ref = _ce_loss_and_grads(
        m_ref, cfg_ref, mesh_ref, tokens=tokens, positions=positions, targets=targets, segment_ids=seg
    )
    loss_ref_np = np.asarray(loss_ref, dtype=np.float32)
    logits_ref_np = np.asarray(logits_ref, dtype=np.float32)
    grads_ref_map = {
        tuple(map(str, p)): np.asarray(v.get_value(), dtype=np.float32) for p, v in nnx.to_flat_state(grads_ref)
    }

    for shard_mode in ("auto", "explicit"):
      cfg_sh = _toy_config(
          scan_layers=scan_layers,
          per_device_batch_size=0.5,  # global batch = 0.5 * 8 = 4 (matches ref!)
          max_target_length=sh_seq_len,
          sparse_matmul=True,
          megablox=False,
          capacity_factor=-1.0,
          use_ring_of_experts=True,
          ici_fsdp_parallelism=2,
          ici_tensor_parallelism=2,
          ici_expert_parallelism=2,
          shard_mode=shard_mode,
      )
      mesh_sh = _make_mesh(cfg_sh, devices=devices[:8])
      _, abs_model = model_creation_utils.create_nnx_abstract_model(cfg_sh, mesh=mesh_sh, model_mode=MODEL_MODE_TRAIN)
      abs_shardings = {tuple(map(str, p)): v.get_value().sharding for p, v in nnx.to_flat_state(nnx.state(abs_model))}
      m_sh = model_creation_utils.from_config(cfg_sh, mesh=mesh_sh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      _copy_state_between_models(m_ref, m_sh)

      # Place every variable (Param, MoEBiasVar, RngState) onto its exact NamedSharding across all 8 devices.
      replicated = NamedSharding(mesh_sh, PartitionSpec())
      sh_state = nnx.state(m_sh)
      sharded_flat = [
          (p, v.replace(jax.device_put(v.get_value(), abs_shardings.get(tuple(map(str, p)), replicated))))
          for p, v in nnx.to_flat_state(sh_state)
      ]
      nnx.update(m_sh, nnx.from_flat_state(sharded_flat))

      data_sharding = sharding.get_input_data_sharding(cfg_sh, mesh_sh)
      sh_toks = jax.device_put(tokens, data_sharding)
      sh_pos = jax.device_put(positions, data_sharding)
      sh_tgts = jax.device_put(targets, data_sharding)
      sh_seg = jax.device_put(seg, data_sharding)

      graphdef, params_sh, non_params_sh = nnx.split(m_sh, nnx.Param, ...)

      @jax.jit
      def _jitted_loss_and_grads(p_state, np_state, batch_in, _gd=graphdef, _cfg=cfg_sh):
        t_in, pos_in, tgt_in, seg_in = batch_in

        def _loss(ps):
          m = nnx.merge(_gd, ps, np_state)
          logits = m(
              decoder_input_tokens=t_in,
              decoder_positions=pos_in,
              decoder_segment_ids=seg_in,
              model_mode=MODEL_MODE_TRAIN,
              enable_dropout=False,
          )
          one_hot = jax.nn.one_hot(tgt_in, _cfg.vocab_size, dtype=jnp.float32)
          xent, _ = max_utils.cross_entropy_with_logits(logits.astype(jnp.float32), one_hot, z_loss=0.0)
          mask = (seg_in != 0).astype(jnp.float32)
          loss = jnp.sum(xent * mask) / jnp.maximum(jnp.sum(mask), 1.0)
          return loss, logits

        return jax.value_and_grad(_loss, has_aux=True)(p_state)

      with jax.set_mesh(mesh_sh), nn_partitioning.axis_rules(cfg_sh.logical_axis_rules):
        (loss_sh, logits_sh), grads_sh = _jitted_loss_and_grads(
            params_sh, non_params_sh, (sh_toks, sh_pos, sh_tgts, sh_seg)
        )
      np.testing.assert_allclose(
          np.asarray(loss_sh, dtype=np.float32),
          loss_ref_np,
          rtol=1e-4,
          atol=1e-4,
          err_msg=f"Loss mismatch (scan={scan_layers}, shard_mode={shard_mode})",
      )
      np.testing.assert_allclose(
          np.asarray(logits_sh, dtype=np.float32)[..., :_TOY_UNPADDED_VOCAB_SIZE],
          logits_ref_np[..., :_TOY_UNPADDED_VOCAB_SIZE],
          rtol=1e-4,
          atol=1e-4,
          err_msg=f"Logits mismatch (scan={scan_layers}, shard_mode={shard_mode})",
      )
      for path, var in nnx.to_flat_state(grads_sh):
        key = tuple(map(str, path))
        np.testing.assert_allclose(
            np.asarray(var.get_value(), dtype=np.float32),
            grads_ref_map[key],
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Sharded grad mismatch at {key} (scan={scan_layers}, shard_mode={shard_mode})",
        )
  print("INKLING2_8DEV_SHARDING_PASSED")


def _run_4dev_vocab_tiling_explicit_checks():
  """Runs 4-device `FSDP=2, TP=2` `shard_mode='explicit'` `num_vocab_tiling=2` parity checks."""
  jax.config.update("jax_default_matmul_precision", "highest")
  devices = jax.devices()
  assert len(devices) >= 4, f"Expected >= 4 CPU devices, got {len(devices)}"

  vt_seq_len = 8
  for scan_layers in (False, True):
    common = {
        "scan_layers": scan_layers,
        "remat_policy": "full",
        "per_device_batch_size": 1,  # global_batch = 4, divisible by FSDP=2 and num_vocab_tiling=2
        "max_target_length": vt_seq_len,
        "sparse_matmul": True,
        "megablox": False,
        "capacity_factor": -1.0,
        "ici_fsdp_parallelism": 2,
        "ici_tensor_parallelism": 2,
        "ici_expert_parallelism": 1,
        "shard_mode": "explicit",
    }
    cfg_t1 = _toy_config(num_vocab_tiling=1, **common)
    cfg_t2 = _toy_config(num_vocab_tiling=2, **common)
    mesh = _make_mesh(cfg_t1, devices=devices[:4])

    def _build_sharded(cfg_item, target_mesh, src_model=None):
      _, abs_m = model_creation_utils.create_nnx_abstract_model(cfg_item, mesh=target_mesh, model_mode=MODEL_MODE_TRAIN)
      abs_sh = {tuple(map(str, p)): v.get_value().sharding for p, v in nnx.to_flat_state(nnx.state(abs_m))}
      replicated = NamedSharding(target_mesh, PartitionSpec())
      m = model_creation_utils.from_config(cfg_item, mesh=target_mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      if src_model is None:
        _randomize_inkling2_state(m, seed=151)
      else:
        _copy_state_between_models(src_model, m)
      sh_flat = [
          (p, v.replace(jax.device_put(v.get_value(), abs_sh.get(tuple(map(str, p)), replicated))))
          for p, v in nnx.to_flat_state(nnx.state(m))
      ]
      nnx.update(m, nnx.from_flat_state(sh_flat))
      return m

    m_t1 = _build_sharded(cfg_t1, mesh)
    m_t2 = _build_sharded(cfg_t2, mesh, src_model=m_t1)

    tokens, positions, targets, seg = _make_inputs(4, vt_seq_len, _TOY_UNPADDED_VOCAB_SIZE, seed=152)
    data_sh = sharding.get_input_data_sharding(cfg_t1, mesh)
    batch = {
        "inputs": jax.device_put(tokens, data_sh),
        "inputs_position": jax.device_put(positions, data_sh),
        "inputs_segmentation": jax.device_put(seg, data_sh),
        "targets": jax.device_put(targets, data_sh),
        "targets_position": jax.device_put(positions, data_sh),
        "targets_segmentation": jax.device_put(seg, data_sh),
    }

    def _eval_loss_and_grads(model, cfg_item, target_mesh, batch_in):
      gd, p_st, np_st = nnx.split(model, nnx.Param, ...)

      @jax.jit
      def _step(ps, nps, b_in, _gd=gd, _cfg=cfg_item):
        def _fn(params):
          loss, _ = pre_train.loss_fn(nnx.merge(_gd, params, nps), _cfg, b_in, None, None, is_train=True)
          return loss

        return jax.value_and_grad(_fn)(ps)

      with jax.set_mesh(target_mesh), nn_partitioning.axis_rules(cfg_item.logical_axis_rules):
        return _step(p_st, np_st, batch_in)

    loss_1, grads_1 = _eval_loss_and_grads(m_t1, cfg_t1, mesh, batch)
    loss_2, grads_2 = _eval_loss_and_grads(m_t2, cfg_t2, mesh, batch)

    np.testing.assert_allclose(
        np.asarray(loss_2, dtype=np.float32),
        np.asarray(loss_1, dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Vocab tiling loss mismatch (scan={scan_layers})",
    )
    g1_map = {tuple(map(str, p)): np.asarray(v.get_value(), dtype=np.float32) for p, v in nnx.to_flat_state(grads_1)}
    for path, var in nnx.to_flat_state(grads_2):
      key = tuple(map(str, path))
      np.testing.assert_allclose(
          np.asarray(var.get_value(), dtype=np.float32),
          g1_map[key],
          rtol=1e-4,
          atol=1e-4,
          err_msg=f"Vocab tiling grad mismatch at {key} (scan={scan_layers})",
      )
  print("INKLING2_4DEV_VOCAB_TILING_PASSED")


class Inkling2ShardingTest(unittest.TestCase):
  """Multi-device virtual CPU sharding parity (`FSDP/TP/EP` & `vocabulary_tiling`, auto & explicit)."""

  def test_8_device_fsdp_tp_ep_auto_and_explicit_sharding_parity(self):
    if len(jax.devices()) >= 8:
      _run_8dev_sharding_checks()
      return
    env = os.environ.copy()
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["_INKLING2_SHARDING_SUBPROC"] = "1"
    result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, f"8-device sharding subprocess failed:\n{result.stdout}\n{result.stderr}")
    self.assertIn("INKLING2_8DEV_SHARDING_PASSED", result.stdout)

  def test_vocabulary_tiling_4dev_explicit_sharding(self):
    if len(jax.devices()) >= 4:
      _run_4dev_vocab_tiling_explicit_checks()
      return
    env = os.environ.copy()
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["_INKLING2_SHARDING_SUBPROC"] = "vt4"
    result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, f"4-device vocab tiling subprocess failed:\n{result.stdout}\n{result.stderr}")
    self.assertIn("INKLING2_4DEV_VOCAB_TILING_PASSED", result.stdout)


# ==============================================================================
# 4. FLOPs Calculation Tests
# ==============================================================================


class Inkling2FlopsTest(unittest.TestCase):
  """Verifies `calculate_tflops_training_per_device` for Inkling 2 Mini."""

  def _cfg(self, **overrides):
    """Builds a full-scale inkling2-mini config for FLOPs verification."""
    kwargs = {
        "model_name": "inkling2-mini",
        "per_device_batch_size": 1,
        "max_target_length": 4096,
        "gradient_accumulation_steps": 1,
        "skip_jax_distributed_system": True,
        "hardware": "cpu",
    }
    kwargs.update(overrides)
    return pyconfig.initialize(["", _BASE_CONFIG_PATH], **kwargs)

  def test_inkling2_mini_flops_48_logical_visits_and_components(self):
    cfg3 = self._cfg()
    total3, weight3, attn3 = maxtext_utils.calculate_tflops_training_per_device(cfg3, log=False)
    self.assertGreater(weight3, 0.0)
    self.assertGreater(attn3, 0.0)
    self.assertAlmostEqual(total3, weight3 + attn3, places=6)

    # Analytic verification of 48 logical visits (40 sliding + 8 full) and 24 message visits.
    b_tok = cfg3.per_device_batch_size * cfg3.max_target_length
    t = cfg3.max_target_length
    h = cfg3.emb_dim
    hq = cfg3.num_query_heads
    hkv = cfg3.num_kv_heads
    d = cfg3.head_dim
    dr = cfg3.inkling2_d_rel
    re = cfg3.inkling2_rel_extent
    k_conv = cfg3.short_conv_kernel_size
    r_msg = cfg3.inkling2_message_gate_rank
    e = cfg3.num_experts
    top_k = cfg3.num_experts_per_tok
    inter = cfg3.moe_mlp_dim
    w = min(cfg3.sliding_window_size, t)

    n_visits = 48
    n_full = 8
    n_sliding = 40
    n_msg = 24

    full_attn_fwd = 2 * b_tok * t * hq * d
    sliding_attn_fwd = 4 * cfg3.per_device_batch_size * hq * d * (t * w - (w * w) / 2.0)
    expected_attn_tflops = (n_full * full_attn_fwd + n_sliding * sliding_attn_fwd) * 3.0 / 1e12

    attn_proj = 2 * b_tok * (h * (3 * hq * d + 2 * hkv * d + hq * dr + hq) + hq * dr * re)
    dyn_conv = 2 * b_tok * (2 * h * hkv * k_conv)
    moe_proj = 2 * b_tok * (h * e + 3 * top_k * h * inter)
    msg_proj = 2 * b_tok * 4 * (4 * h * r_msg)
    embed_flops = 2 * b_tok * h * cfg3.vocab_size
    expected_weight_tflops = (
        (n_visits * (attn_proj + moe_proj) + n_full * dyn_conv + n_msg * msg_proj + embed_flops) * 3.0 / 1e12
    )

    self.assertAlmostEqual(attn3, expected_attn_tflops, delta=expected_attn_tflops * 0.05)
    self.assertAlmostEqual(weight3, expected_weight_tflops, delta=expected_weight_tflops * 0.05)

    # 3 passes must be strictly larger than 1 pass (48 visits vs 24 visits + 24 message passes).
    cfg1 = self._cfg(override_model_config=True, inkling2_num_passes=1)
    total1, weight1, attn1 = maxtext_utils.calculate_tflops_training_per_device(cfg1, log=False)
    self.assertGreater(total3, total1)
    self.assertAlmostEqual(attn3 / attn1, 2.0, places=5)
    self.assertGreater(weight3, weight1 * 1.5)

    # Gradient accumulation steps scale per-step TFLOPs linearly.
    cfg_ga2 = self._cfg(gradient_accumulation_steps=2)
    total_ga2, weight_ga2, attn_ga2 = maxtext_utils.calculate_tflops_training_per_device(cfg_ga2, log=False)
    self.assertAlmostEqual(total_ga2, 2.0 * total3, places=5)
    self.assertAlmostEqual(weight_ga2, 2.0 * weight3, places=5)
    self.assertAlmostEqual(attn_ga2, 2.0 * attn3, places=5)


if __name__ == "__main__":
  _subproc_mode = os.environ.get("_INKLING2_SHARDING_SUBPROC")
  if _subproc_mode == "1":
    _run_8dev_sharding_checks()
  elif _subproc_mode == "vt4":
    _run_4dev_vocab_tiling_explicit_checks()
  else:
    unittest.main()
