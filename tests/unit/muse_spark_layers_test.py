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

"""Fast Torch-free CPU unit tests for MaxText Muse-Spark layers and scanned decoder."""

from __future__ import annotations

import os
import re
import types
import unittest

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common.common_types import (
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import quantizations
from maxtext.models import muse_spark
from maxtext.utils import maxtext_utils, model_creation_utils, muon_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

EMB, Q_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 16
# Expert parallelism dispatches token rows as `[dim / 128, 128]` tiles, so the expert dims are multiples of 128.
MLP, MOE_IN, MOE_MLP, EXPERTS, TOPK = 64, 128, 128, 4, 2
SLIDING_WINDOW = 4
VOCAB = 64

# Training flags for 256 TPU v7x chips (512 TensorCores), as listed in
# tests/end_to_end/tpu/muse_spark/Run_Muse_Spark.md. Keep the two in sync.
TPU7X_256_TRAINING_FLAGS = {
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    # Mesh: expert parallelism over the z ring, FSDP over the (x, y) plane and the two cores of a chip.
    "custom_mesh": "ep8_z_fsdp64_xycore",
    "ici_fsdp_parallelism": 64,
    "ici_expert_parallelism": 8,
    "logical_axis_rules": [["embed", ["fsdp"]], ["embed_attn", ["fsdp"]], ["embed_vocab", ["fsdp"]]],
    "sharding_tolerance": 0.05,
    # Tokamax Splash attention with sequence-minor layouts.
    "attention": "flash",
    "use_tokamax_splash": True,
    "sa_num_stacked_q_heads": 8,
    "sa_q_layout": "SEQ_MINOR",
    "sa_k_layout": "SEQ_MINOR",
    "sa_v_layout": "SEQ_MINOR",
    "local_sa_q_layout": "SEQ_MINOR",
    "local_sa_k_layout": "SEQ_MINOR",
    "local_sa_v_layout": "SEQ_MINOR",
    "sa_use_fused_bwd_kernel": True,
    "local_sa_use_fused_bwd_kernel": True,
    # Expert matmuls and dispatch.
    "use_tokamax_gmm": True,
    "use_gmm_v2": True,
    "ragged_buffer_factor": 1.5,
    "muse_spark_moe_dropless_overflow": False,
    "muse_spark_moe_num_token_chunks": 2,
    # Memory: what the backward pass keeps instead of recomputing.
    "remat_policy": "custom",
    "decoder_layer_input": "remat",
    "attention_out": "remat",
    "context": "device",
    "moe_combined": "device",
    "moe_routing_maps": "device",
    "out_proj": "device",
    "value_proj": "device",
    "query_proj": "remat",
    "key_proj": "remat",
    "mlpwi": "remat",
    "pre_expert_norm": "device",
    "mlpwo": "remat",
    "num_vocab_tiling": 1,
    "tie_lm_head_cotangents": True,
}

# FP8 expert matmuls must stay within this signal-to-quantization-noise ratio of the full-precision result. The CPU
# reference path measures 21 to 24 dB on the test below.
FP8_MIN_SQNR_DB = 18.0
# FP8 experts on 256 TPU v7x chips: applied on top of `TPU7X_256_TRAINING_FLAGS` (see Run_Muse_Spark.md).
TPU7X_256_FP8_TRAINING_FLAGS = {
    "quantization": "fp8_full",
    "use_qwix_quantization": True,
    "quantize_router_proj": False,
    "te_moe_dispatch_remat_location": "device",
    "wi_tile_fwd_batch_seq": 512,
    "wi_tile_fwd_embed_dim": 2048,
    "wi_tile_fwd_mlp_dim": 2048,
}


def _collect_while_body_collectives(hlo_txt: str) -> tuple[set[str], list[str]]:
  """Returns `(visited_while_bodies, collective_instructions)` transitively reachable from HLO `while` `body=`."""
  comps: dict[str, list[str]] = {}
  cur_name: str | None = None
  cur_instrs: list[str] = []
  for raw_line in hlo_txt.splitlines():
    stripped = raw_line.strip()
    if not stripped:
      continue
    if not raw_line.startswith(" ") and stripped.endswith("{"):
      m = re.match(r"^(?:ENTRY\s+)?%?([^\s(]+)\s*\(", stripped)
      cur_name = m.group(1) if m else None
      cur_instrs = []
    elif cur_name is not None and not raw_line.startswith(" ") and stripped == "}":
      comps[cur_name] = cur_instrs
      cur_name = None
    elif cur_name is not None:
      if stripped.startswith(("%", "ROOT ")) or not cur_instrs:
        cur_instrs.append(stripped)
      else:
        cur_instrs[-1] += " " + stripped

  queue: list[str] = []
  for instrs in comps.values():
    for instr in instrs:
      if re.search(r"\bwhile\(", instr):
        queue.extend(re.findall(r"\bbody=%?([A-Za-z0-9_.-]+)", instr))

  visited: set[str] = set()
  violations: list[str] = []
  while queue:
    cname = queue.pop()
    if cname in visited or cname not in comps:
      continue
    visited.add(cname)
    for instr in comps[cname]:
      queue.extend(
          re.findall(
              r"\b(?:body|condition|calls|to_apply|true_computation|false_computation)=%?([A-Za-z0-9_.-]+)",
              instr,
          )
      )
      if re.search(r"\b(?:all-gather|reduce-scatter)(?:-start|-done)?\(", instr):
        violations.append(f"{cname}: {instr}")
  return visited, violations


def _assert_no_while_body_collectives(test_case: unittest.TestCase, hlo_txt: str) -> None:
  """Asserts compiled HLO visits >= 1 `while` body computation and contains zero `all-gather`/`reduce-scatter` inside."""
  visited, violations = _collect_while_body_collectives(hlo_txt)
  test_case.assertGreaterEqual(len(visited), 1, "Expected >= 1 while body computation in compiled HLO")
  test_case.assertEqual(violations, [], f"Unexpected FSDP collectives inside while body: {violations[:3]}")


def make_config(**kwargs):
  """Builds a minimal Muse-Spark MaxText config for fast CPU unit tests."""
  overrides = {
      "model_name": "muse-spark-816b-a42b",
      "override_model_config": True,
      "base_emb_dim": EMB,
      "base_num_query_heads": Q_HEADS,
      "base_num_kv_heads": KV_HEADS,
      "head_dim": HEAD_DIM,
      "base_mlp_dim": MLP,
      "moe_expert_input_dim": MOE_IN,
      "base_moe_mlp_dim": MOE_MLP,
      "num_experts": EXPERTS,
      "num_experts_per_tok": TOPK,
      "shared_experts": 0,
      "routed_scaling_factor": 1.0,
      "routed_score_func": "sigmoid",
      "routed_bias": True,
      "norm_topk_prob": True,
      "first_num_dense_layers": 0,
      "inhomogeneous_layer_cycle_interval": 4,
      "sliding_window_size": SLIDING_WINDOW,
      "vocab_size": VOCAB,
      "base_num_decoder_layers": 6,
      "per_device_batch_size": 1,
      "max_target_length": 8,
      "max_prefill_predict_length": 4,
      "max_position_embeddings": 64,
      "original_max_position_embeddings": 64,
      "scan_layers": False,
      "attention": "dot_product",
      "sparse_matmul": False,
      "capacity_factor": -1,
      "matmul_precision": "highest",
      "dtype": "float32",
      "weight_dtype": "float32",
      "float32_logits": True,
      "float32_qk_product": True,
      "dropout_rate": 0.0,
      "skip_jax_distributed_system": True,
      "hardware": "cpu",
  }
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


def copy_unscanned_to_scanned(unscanned_model, scanned_model, cfg_s):
  """Copies parameters and MoE router biases from an unscanned Muse-Spark model into a scanned one."""
  scanned_model.token_embedder.embedding[...] = unscanned_model.token_embedder.embedding[...]
  scanned_model.decoder.decoder_norm.scale[...] = unscanned_model.decoder.decoder_norm.scale[...]
  scanned_model.decoder.logits_dense.kernel[...] = unscanned_model.decoder.logits_dense.kernel[...]

  num_prefix, num_blocks = muse_spark.scan_layout(cfg_s)
  cycle = cfg_s.inhomogeneous_layer_cycle_interval

  for i in range(num_prefix):
    nnx.update(getattr(scanned_model.decoder, f"layers_{i}"), nnx.state(getattr(unscanned_model.decoder, f"layers_{i}")))

  if num_blocks > 0:
    scan_axis = cfg_s.param_scan_axis
    for sub_id in range(cycle):
      dst_sub = getattr(scanned_model.decoder.scanned_blocks, f"layers_{sub_id}")
      src_states = [
          nnx.to_flat_state(nnx.state(getattr(unscanned_model.decoder, f"layers_{num_prefix + b * cycle + sub_id}")))
          for b in range(num_blocks)
      ]
      dst_flat = nnx.to_flat_state(nnx.state(dst_sub))
      updated_flat = []
      for idx, (path, dst_var) in enumerate(dst_flat):
        if isinstance(dst_var, (nnx.RngState, nnx.Intermediate)):
          updated_flat.append((path, dst_var))
          continue
        axis = scan_axis if isinstance(dst_var, nnx.Param) else 0
        stacked = jnp.stack([src_states[b][idx][1].get_value() for b in range(num_blocks)], axis=axis)
        updated_flat.append((path, dst_var.replace(stacked)))
      nnx.update(dst_sub, nnx.from_flat_state(updated_flat))


@pytest.mark.cpu_only
class MuseSparkStandaloneLayersTest(unittest.TestCase):
  """Fast Torch-free unit tests for Muse-Spark modules, sharding, and scan layout."""

  @classmethod
  def setUpClass(cls):
    cls.config = make_config()
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.config), cls.config.mesh_axes)

  def test_layer_pattern_and_scan_layout(self):
    def stub(num_layers, first_dense=0, cycle=4, experts=EXPERTS):
      return types.SimpleNamespace(
          first_num_dense_layers=first_dense,
          inhomogeneous_layer_cycle_interval=cycle,
          num_decoder_layers=num_layers,
          num_experts=experts,
      )

    cfg62 = stub(62)
    self.assertEqual(muse_spark.scan_layout(cfg62), (2, 15))
    full_indices = [i for i in range(62) if muse_spark.is_full_attention_layer(i, cfg62)]
    self.assertEqual(full_indices, list(range(1, 62, 4)))
    self.assertTrue(all(muse_spark.is_moe_layer(i, cfg62) for i in range(62)))

    # Dense prefix forces extra whole cycle(s) into unrolled prefix so scanned cycles stay homogeneous MoE.
    cfg_dense_prefix = stub(62, first_dense=3)
    self.assertEqual(muse_spark.scan_layout(cfg_dense_prefix), (6, 14))
    self.assertEqual([muse_spark.is_moe_layer(i, cfg_dense_prefix) for i in range(5)], [False, False, False, True, True])
    self.assertEqual(muse_spark.scan_layout(stub(6, experts=1)), (6, 0))

  def test_norms_and_residual_gate(self):
    x = jnp.asarray(np.random.RandomState(0).randn(2, 5, EMB), dtype=jnp.float32)
    y = jnp.asarray(np.random.RandomState(1).randn(2, 5, EMB), dtype=jnp.float32)

    rms_noscale = muse_spark.MuseSparkRMSNorm(EMB, epsilon=1e-8, dtype=jnp.float32, with_scale=False, rngs=nnx.Rngs(0))
    self.assertFalse(hasattr(rms_noscale, "scale") and rms_noscale.scale is not None)
    ref_noscale = x * jnp.power(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-8, -0.5)
    np.testing.assert_allclose(rms_noscale(x), ref_noscale, atol=1e-6, rtol=1e-6)
    self.assertEqual(rms_noscale(x.astype(jnp.bfloat16)).dtype, jnp.bfloat16)

    # Centered RMSNorm in bf16: `(bf16(1.0) + scale_bf16).astype(fp32)`.
    crms_bf16 = muse_spark.MuseSparkCenteredRMSNorm(
        EMB,
        epsilon=1e-5,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(2),
    )
    scale_bf16 = jnp.asarray(np.random.RandomState(3).randn(EMB) * 0.1, dtype=jnp.bfloat16)
    crms_bf16.scale[...] = scale_bf16
    x_bf16 = x.astype(jnp.bfloat16)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
      out_crms = crms_bf16(x_bf16)
    x_fp32 = x_bf16.astype(jnp.float32)
    expected_crms = (
        x_fp32
        * jax.lax.rsqrt(jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True) + 1e-5)
        * (jnp.asarray(1.0, dtype=jnp.bfloat16) + scale_bf16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out_crms), np.asarray(expected_crms))

    # Input-scaled RMSNorm rounds `x * (1 + scale)` to input dtype before fp32 variance.
    irms_bf16 = muse_spark.MuseSparkInputScaledRMSNorm(
        EMB, epsilon=1e-8, dtype=jnp.bfloat16, weight_dtype=jnp.bfloat16, rngs=nnx.Rngs(4)
    )
    irms_bf16.scale[...] = scale_bf16
    out_irms = irms_bf16(x_bf16)
    scaled_bf16 = (x_fp32 * (jnp.asarray(1.0, dtype=jnp.bfloat16) + scale_bf16).astype(jnp.float32)).astype(jnp.bfloat16)
    scaled_fp32 = scaled_bf16.astype(jnp.float32)
    expected_irms = (
        scaled_fp32 * jax.lax.rsqrt(jnp.mean(jnp.square(scaled_fp32), axis=-1, keepdims=True) + 1e-8)
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out_irms), np.asarray(expected_irms))

    # Residual gate always keeps `gate` in fp32 and initializes with normal(stddev=0.02).
    res_gate = muse_spark.MuseSparkResidualGate(EMB, temperature=0.3, weight_dtype=jnp.bfloat16, rngs=nnx.Rngs(5))
    gate_val = res_gate.gate.get_value()
    self.assertEqual(gate_val.dtype, jnp.float32)
    self.assertGreater(float(jnp.linalg.norm(gate_val)), 0.0)
    w = gate_val / 0.3
    beta = jax.nn.sigmoid(w)
    alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-w) * (1.0 + beta), 1e-3))
    expected_gate = alpha * x + beta * y
    np.testing.assert_allclose(res_gate(x, y), expected_gate, atol=1e-6, rtol=1e-6)

  def test_attention_flags_sliding_window_and_kv_cache(self):
    cfg_no_fp32 = make_config(float32_logits=False, float32_qk_product=False, attention_bias=True)
    attn_no_fp32 = muse_spark.MuseSparkAttention(
        config=cfg_no_fp32,
        num_query_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_target_length=8,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 6, EMB),
        inputs_kv_shape=(1, 6, EMB),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        is_full_attention=False,
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    self.assertFalse(attn_no_fp32.attention_op.float32_logits)
    self.assertFalse(attn_no_fp32.attention_op.float32_qk_product)
    self.assertTrue(attn_no_fp32.query.use_bias)
    self.assertFalse(attn_no_fp32.gate_proj.use_bias)

    seq = 6  # > SLIDING_WINDOW (4): token 5 cannot attend to token 0 on sliding layers, but can on full layers.
    x = jnp.asarray(np.random.RandomState(7).randn(1, seq, EMB), dtype=jnp.float32)
    x_perturbed = x.at[:, 0, :].add(5.0)
    pos = jnp.arange(seq, dtype=jnp.int32)[None, :]
    seg = jnp.ones((1, seq), dtype=jnp.int32)

    attn_sliding = muse_spark.MuseSparkAttention(
        config=self.config,
        num_query_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_target_length=8,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 4, EMB),
        inputs_kv_shape=(1, 4, EMB),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        max_prefill_predict_length=4,
        is_full_attention=False,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=10, dropout=11),
    )
    attn_full = muse_spark.MuseSparkAttention(
        config=self.config,
        num_query_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_target_length=8,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 4, EMB),
        inputs_kv_shape=(1, 4, EMB),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        max_prefill_predict_length=4,
        is_full_attention=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=12, dropout=13),
    )

    out_s1, _ = attn_sliding(x, x, pos, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN, deterministic=True)
    out_s2, _ = attn_sliding(
        x_perturbed, x_perturbed, pos, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN, deterministic=True
    )
    np.testing.assert_allclose(out_s1[:, -1], out_s2[:, -1], atol=1e-6, rtol=1e-6)

    out_f1, _ = attn_full(x, x, pos, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN, deterministic=True)
    out_f2, _ = attn_full(
        x_perturbed, x_perturbed, pos, decoder_segment_ids=seg, model_mode=MODEL_MODE_TRAIN, deterministic=True
    )
    self.assertGreater(float(jnp.max(jnp.abs(out_f1[:, -1] - out_f2[:, -1]))), 1e-3)

    # Prefill + AR decode parity against train-mode forward pass.
    prompt, pad = 3, 4
    xp = jnp.concatenate([x[:, :prompt], jnp.zeros((1, pad - prompt, EMB), dtype=jnp.float32)], axis=1)
    pos_p = jnp.arange(pad, dtype=jnp.int32)[None, :]
    seg_p = (jnp.arange(pad) < prompt).astype(jnp.int32)[None, :]
    for attn, ref in ((attn_sliding, out_s1), (attn_full, out_f1)):
      out_p, _ = attn(xp, xp, pos_p, decoder_segment_ids=seg_p, model_mode=MODEL_MODE_PREFILL, deterministic=True)
      np.testing.assert_allclose(out_p[:, :prompt], ref[:, :prompt], atol=1e-5, rtol=1e-5)
      for t in range(prompt, seq):
        out_t, _ = attn(
            x[:, t : t + 1],
            x[:, t : t + 1],
            jnp.array([[t]], dtype=jnp.int32),
            decoder_segment_ids=jnp.ones((1, 1), dtype=jnp.int32),
            model_mode=MODEL_MODE_AUTOREGRESSIVE,
            deterministic=True,
        )
        np.testing.assert_allclose(out_t[:, 0], ref[:, t], atol=1e-5, rtol=1e-5)

  def test_moe_losses_expert_parallel_guard_and_sharded_grads(self):
    aux_batch = max(2, jax.device_count())
    x = jnp.asarray(np.random.RandomState(22).randn(aux_batch, 4, EMB), dtype=jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None, :], (aux_batch, 4))
    seg = jnp.ones((aux_batch, 4), dtype=jnp.int32)
    for megatron_aux in (False, True):
      cfg_aux = make_config(load_balance_loss_weight=0.02, moe_use_megatron_seq_aux_loss=megatron_aux)
      layer_aux = muse_spark.MuseSparkDecoderLayer(
          config=cfg_aux, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, layer_idx=0, rngs=nnx.Rngs(params=20, dropout=21)
      )
      with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_aux.logical_axis_rules):
        _, _ = layer_aux(x, seg, pos, deterministic=True, model_mode=MODEL_MODE_TRAIN)
      intermediates = nnx.pop(layer_aux, nnx.Intermediate)
      self.assertIn("moe_lb_loss", intermediates)
      self.assertGreater(float(intermediates["moe_lb_loss"].get_value()[0]), 0.0)

    # Indivisible `num_experts % ep_size != 0` must raise a clear ValueError.
    if jax.device_count() >= 2:
      ep_mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(2), ("expert",))
      moe_bad_ep = muse_spark.MuseSparkMoE(make_config(num_experts=3), ep_mesh, rngs=nnx.Rngs(params=23))
      with self.assertRaisesRegex(ValueError, "divisible by expert mesh size"):
        moe_bad_ep(x)

    # Expert parallelism needs an expert input dim that is a multiple of 128.
    if jax.device_count() >= 2:
      cfg_bad_dim = make_config(
          moe_expert_input_dim=16, ici_fsdp_parallelism=1, ici_expert_parallelism=jax.device_count()
      )
      mesh_bad_dim = Mesh(maxtext_utils.create_device_mesh(cfg_bad_dim), cfg_bad_dim.mesh_axes)
      with jax.set_mesh(mesh_bad_dim), nn_partitioning.axis_rules(cfg_bad_dim.logical_axis_rules):
        moe_bad_dim = muse_spark.MuseSparkMoE(cfg_bad_dim, mesh_bad_dim, rngs=nnx.Rngs(params=26))
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
          moe_bad_dim(jnp.zeros((jax.device_count(), 4, EMB), jnp.float32))

    # `moe_tc_ragged_weights_on_activation=True` must raise ValueError because `post_expert_norm`
    # normalizes each expert output before weighting by `top_k_weights`.
    cfg_bad_moe = make_config(moe_tc_ragged_weights_on_activation=True)
    with self.assertRaisesRegex(ValueError, "moe_tc_ragged_weights_on_activation=False"):
      muse_spark.MuseSparkMoE(cfg_bad_moe, self.mesh, rngs=nnx.Rngs(params=25))

    # Compare single-device reference against multi-device FSDP+TP, pure EP, and FSDP+EP
    # on divisible and indivisible batches.
    single_mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * len(self.config.mesh_axes)), self.config.mesh_axes)
    ref_moe = muse_spark.MuseSparkMoE(self.config, single_mesh, rngs=nnx.Rngs(params=24))
    ref_flat = dict(nnx.to_flat_state(nnx.state(ref_moe)))

    mesh_overrides = [{}]
    if jax.device_count() >= 2 and jax.device_count() % 2 == 0:
      mesh_overrides.append({"ici_fsdp_parallelism": jax.device_count() // 2, "ici_tensor_parallelism": 2})
      mesh_overrides.append({"ici_fsdp_parallelism": jax.device_count() // 2, "ici_expert_parallelism": 2})
      mesh_overrides.append(
          {
              "ici_fsdp_parallelism": jax.device_count() // 2,
              "ici_expert_parallelism": 2,
              "ragged_buffer_factor": 2.0,
          }
      )
    if jax.device_count() >= 4 and EXPERTS % jax.device_count() == 0:
      mesh_overrides.append({"ici_expert_parallelism": jax.device_count()})

    @nnx.jit
    def eval_moe_and_grads(m, xin, pr, forced):
      def loss_fn(mod, x_arg):
        out, _ = mod(x_arg, forced_routed_experts=forced)
        return jnp.sum(out * pr), out

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    for batch in (1, 3, max(2, jax.device_count())):
      xb_np = np.random.RandomState(30 + batch).randn(batch, 5, EMB).astype(np.float32)
      probe_np = np.random.RandomState(40 + batch).randn(batch, 5, EMB).astype(np.float32)
      forced_np = np.zeros((batch, 5, TOPK), dtype=np.int32)
      forced_np[:, :, 0] = 1
      forced_np[:, :, 1] = 3
      forced_np[:, -1:, :] = -1

      xb_ref = jax.device_put(jnp.asarray(xb_np), single_mesh.devices.flat[0])
      pr_ref = jax.device_put(jnp.asarray(probe_np), single_mesh.devices.flat[0])
      with jax.set_mesh(single_mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
        (_, ref_y), (ref_grads, ref_dx) = eval_moe_and_grads(ref_moe, xb_ref, pr_ref, None)
        forced_ref = jax.device_put(jnp.asarray(forced_np), single_mesh.devices.flat[0])
        (_, ref_y_forced), _ = eval_moe_and_grads(ref_moe, xb_ref, pr_ref, forced_ref)

      for mesh_kw in mesh_overrides:
        cfg_m = make_config(**mesh_kw)
        mesh_m = Mesh(maxtext_utils.create_device_mesh(cfg_m), cfg_m.mesh_axes)
        with jax.set_mesh(mesh_m), nn_partitioning.axis_rules(cfg_m.logical_axis_rules):
          moe_m = muse_spark.MuseSparkMoE(cfg_m, mesh_m, rngs=nnx.Rngs(params=99))
          dst_flat = nnx.to_flat_state(nnx.state(moe_m))
          updated = []
          for path, dst_var in dst_flat:
            src_arr = np.asarray(ref_flat[path].get_value())
            dst_val = dst_var.get_value()
            shd = getattr(dst_val, "sharding", None)
            placed = jax.device_put(jnp.asarray(src_arr), shd) if shd is not None else jnp.asarray(src_arr)
            updated.append((path, dst_var.replace(placed)))
          nnx.update(moe_m, nnx.from_flat_state(updated))
          xb_m = jnp.asarray(xb_np)
          pr_m = jnp.asarray(probe_np)
          (_, y_m), (grads_m, dx_m) = eval_moe_and_grads(moe_m, xb_m, pr_m, None)
          (_, y_m_forced), _ = eval_moe_and_grads(moe_m, xb_m, pr_m, jnp.asarray(forced_np))
        np.testing.assert_allclose(np.asarray(y_m), np.asarray(ref_y), atol=1e-5, rtol=1e-5, err_msg=str(mesh_kw))
        np.testing.assert_allclose(np.asarray(y_m_forced), np.asarray(ref_y_forced), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(dx_m), np.asarray(ref_dx), atol=1e-5, rtol=1e-5, err_msg=str(mesh_kw))
        for p_ref, p_m in zip(nnx.to_flat_state(ref_grads), nnx.to_flat_state(grads_m)):
          np.testing.assert_allclose(
              np.asarray(p_m[1].get_value()),
              np.asarray(p_ref[1].get_value()),
              atol=1e-4,
              rtol=1e-4,
              err_msg=f"{mesh_kw}:{p_ref[0]}",
          )

  def test_moe_small_buffer_overflow_keeps_every_token(self):
    """A `ragged_buffer_factor` buffer that is too small for a shard's tokens must not change the result.

    With EP=4 (one local expert per shard), `k=1` and every token forced onto expert 0, shard 0 receives all
    `4 * 512` gathered slots, but its buffer holds only `align_to(2048 / 4 * 1.0, 512) = 512` rows. The other 1536
    slots go through the overflow loop, so outputs and gradients match the full-buffer (`ragged_buffer_factor=-1`) run.
    """
    if jax.device_count() < 4 or EXPERTS % 4 != 0:
      self.skipTest("Needs 4 devices and `num_experts % 4 == 0` for the EP=4 overflow test.")
    seq = 512
    mesh_kw = {"ici_fsdp_parallelism": 1, "ici_expert_parallelism": 4, "num_experts_per_tok": 1}
    cfg_small = make_config(ragged_buffer_factor=1.0, **mesh_kw)
    cfg_full = make_config(ragged_buffer_factor=-1.0, **mesh_kw)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_small), cfg_small.mesh_axes)

    x_np = np.random.RandomState(70).randn(4, seq, EMB).astype(np.float32)
    probe_np = np.random.RandomState(71).randn(4, seq, EMB).astype(np.float32)
    all_to_expert0 = np.zeros((4, seq, 1), dtype=np.int32)

    @nnx.jit
    def eval_moe_and_grads(m, xin, pr, forced):
      def loss_fn(mod, x_arg):
        out, _ = mod(x_arg, forced_routed_experts=forced)
        return jnp.sum(out * pr), out

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_small.logical_axis_rules):
      moe_small = muse_spark.MuseSparkMoE(cfg_small, mesh, rngs=nnx.Rngs(params=72))
      moe_full = muse_spark.MuseSparkMoE(cfg_full, mesh, rngs=nnx.Rngs(params=73))
      nnx.update(moe_full, nnx.state(moe_small))
      xb, pr = jnp.asarray(x_np), jnp.asarray(probe_np)
      (_, y_small), (g_small, dx_small) = eval_moe_and_grads(moe_small, xb, pr, jnp.asarray(all_to_expert0))
      (_, y_full), (g_full, dx_full) = eval_moe_and_grads(moe_full, xb, pr, jnp.asarray(all_to_expert0))
      overflow_small = int(np.sum(np.asarray(moe_small.moe_overflow_slots.get_value()[-1])))
      overflow_full = int(np.sum(np.asarray(moe_full.moe_overflow_slots.get_value()[-1])))

    # 2048 slots reach shard 0 and 512 fit in its buffer.
    self.assertEqual(overflow_small, 1536)
    self.assertEqual(overflow_full, 0)
    y_small, y_full = np.asarray(y_small), np.asarray(y_full)
    # Every batch row gets its MoE output, including the rows whose slots did not fit in the buffer.
    for row in range(4):
      self.assertGreater(float(np.max(np.abs(y_small[row]))), 1e-3)
    np.testing.assert_allclose(y_small, y_full, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dx_small), np.asarray(dx_full), atol=1e-5, rtol=1e-5)
    for p_full, p_small in zip(nnx.to_flat_state(g_full), nnx.to_flat_state(g_small)):
      np.testing.assert_allclose(
          np.asarray(p_small[1].get_value()),
          np.asarray(p_full[1].get_value()),
          atol=1e-4,
          rtol=1e-4,
          err_msg=str(p_full[0]),
      )

    # Also test non-divisible tail overflow (`total_slots_c=4096`, `buf_rows_c=1536`, `4096 % 1536 != 0`)
    # combined with `muse_spark_moe_num_token_chunks=2`.
    seq2 = 1024
    mesh_kw2 = {
        "num_experts": 8,
        "ici_fsdp_parallelism": 1,
        "ici_expert_parallelism": 4,
        "num_experts_per_tok": 2,
        "max_target_length": seq2,
    }
    cfg_tail_small = make_config(ragged_buffer_factor=1.5, muse_spark_moe_num_token_chunks=2, **mesh_kw2)
    cfg_tail_full = make_config(ragged_buffer_factor=-1.0, muse_spark_moe_num_token_chunks=1, **mesh_kw2)
    x2_np = np.random.RandomState(170).randn(4, seq2, EMB).astype(np.float32)
    probe2_np = np.random.RandomState(171).randn(4, seq2, EMB).astype(np.float32)
    forced_pair = np.broadcast_to(np.array([[[0, 1]]], dtype=np.int32), (4, seq2, 2)).copy()
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_tail_small.logical_axis_rules):
      moe_t_small = muse_spark.MuseSparkMoE(cfg_tail_small, mesh, rngs=nnx.Rngs(params=172))
      moe_t_full = muse_spark.MuseSparkMoE(cfg_tail_full, mesh, rngs=nnx.Rngs(params=173))
      nnx.update(moe_t_full, nnx.state(moe_t_small))
      xb2, pr2 = jnp.asarray(x2_np), jnp.asarray(probe2_np)
      (_, y_t_small), (g_t_small, dx_t_small) = eval_moe_and_grads(moe_t_small, xb2, pr2, jnp.asarray(forced_pair))
      (_, y_t_full), (g_t_full, dx_t_full) = eval_moe_and_grads(moe_t_full, xb2, pr2, jnp.asarray(forced_pair))
      ov_tail = int(np.sum(np.asarray(moe_t_small.moe_overflow_slots.get_value()[-1])))
    # Per chunk: 4096 slots reach shard 0, 1536 fit in buffer -> 2560 overflow slots per chunk x 2 chunks = 5120.
    self.assertEqual(ov_tail, 5120)
    np.testing.assert_allclose(np.asarray(y_t_small), np.asarray(y_t_full), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dx_t_small), np.asarray(dx_t_full), atol=1e-5, rtol=1e-5)
    for p_full, p_small in zip(nnx.to_flat_state(g_t_full), nnx.to_flat_state(g_t_small)):
      np.testing.assert_allclose(
          np.asarray(p_small[1].get_value()),
          np.asarray(p_full[1].get_value()),
          atol=1e-4,
          rtol=1e-4,
          err_msg=str(p_full[0]),
      )

  def test_moe_token_chunks_match_single_chunk(self):
    """`muse_spark_moe_num_token_chunks` changes the schedule of the expert-parallel collectives, not the result."""
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices for FSDP=2 x EP=2.")
    mesh_kw = {"ici_fsdp_parallelism": 2, "ici_expert_parallelism": 2}
    cfg_one = make_config(**mesh_kw)
    cfg_two = make_config(muse_spark_moe_num_token_chunks=2, **mesh_kw)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_one), cfg_one.mesh_axes)
    x = jnp.asarray(np.random.RandomState(80).randn(4, 8, EMB), dtype=jnp.float32)
    probe = jnp.asarray(np.random.RandomState(81).randn(4, 8, EMB), dtype=jnp.float32)

    @nnx.jit
    def eval_moe_and_grads(m, xin):
      def loss_fn(mod, x_arg):
        out, _ = mod(x_arg)
        return jnp.sum(out * probe), out

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_one.logical_axis_rules):
      moe_one = muse_spark.MuseSparkMoE(cfg_one, mesh, rngs=nnx.Rngs(params=82))
      moe_two = muse_spark.MuseSparkMoE(cfg_two, mesh, rngs=nnx.Rngs(params=83))
      nnx.update(moe_two, nnx.state(moe_one))
      (_, y_one), (g_one, dx_one) = eval_moe_and_grads(moe_one, x)
      (_, y_two), (g_two, dx_two) = eval_moe_and_grads(moe_two, x)

      # The chunk count must divide the per-device token count (2 batch rows x 8 tokens per FSDP shard here).
      moe_bad = muse_spark.MuseSparkMoE(
          make_config(muse_spark_moe_num_token_chunks=3, **mesh_kw), mesh, rngs=nnx.Rngs(params=84)
      )
      with self.assertRaisesRegex(ValueError, "muse_spark_moe_num_token_chunks"):
        moe_bad(x)

    np.testing.assert_allclose(np.asarray(y_two), np.asarray(y_one), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dx_two), np.asarray(dx_one), atol=1e-5, rtol=1e-5)
    for p_one, p_two in zip(nnx.to_flat_state(g_one), nnx.to_flat_state(g_two)):
      np.testing.assert_allclose(
          np.asarray(p_two[1].get_value()),
          np.asarray(p_one[1].get_value()),
          atol=1e-4,
          rtol=1e-4,
          err_msg=str(p_one[0]),
      )

  def test_moe_fp8_experts_track_full_precision_with_and_without_overflow(self):
    """`quantization="fp8_full"` runs the expert matmuls in FP8 and stays within quantization noise of full precision.

    With EP=4 (`num_experts=8`, `k=2`) and every token forced onto `[0, 1]` (both hosted on shard 0), shard 0 receives
    all `4 * 512 * 2 = 4096` slots while `ragged_buffer_factor=0.5` holds `512` rows (`3584` overflow slots), and
    `k=2` keeps `gate/kernel` gradients non-zero under `norm_topk_prob=True`.
    """
    from unittest import mock  # pylint: disable=import-outside-toplevel

    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices for the EP=4 and FSDP=2 x EP=2 FP8 tests.")
    seq = 512
    mesh_kw = {
        "num_experts": 8,
        "ici_fsdp_parallelism": 1,
        "ici_expert_parallelism": 4,
        "num_experts_per_tok": 2,
        "max_target_length": seq,
    }
    cfg_ref = make_config(ragged_buffer_factor=-1.0, **mesh_kw)
    cfg_fp8_full = make_config(quantization="fp8_full", ragged_buffer_factor=-1.0, **mesh_kw)
    cfg_fp8_small = make_config(quantization="fp8_full", ragged_buffer_factor=0.5, **mesh_kw)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)

    x_np = np.random.RandomState(80).randn(4, seq, EMB).astype(np.float32)
    probe_np = np.random.RandomState(81).randn(4, seq, EMB).astype(np.float32)
    all_to_shard0 = jnp.broadcast_to(jnp.array([[[0, 1]]], dtype=jnp.int32), (4, seq, 2))

    @nnx.jit
    def eval_moe_and_grads(m, xin, pr, forced):
      def loss_fn(mod, x_arg):
        out, _ = mod(x_arg, forced_routed_experts=forced)
        return jnp.sum(out * pr), out

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    def sqnr_db(reference, candidate):
      reference = np.asarray(reference, dtype=np.float64)
      noise = np.sum(np.square(reference - np.asarray(candidate, dtype=np.float64)))
      return 10.0 * np.log10(np.sum(np.square(reference)) / max(noise, 1e-30))

    results = {}
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      moe_ref = muse_spark.MuseSparkMoE(cfg_ref, mesh, rngs=nnx.Rngs(params=82))
      xb, pr = jnp.asarray(x_np), jnp.asarray(probe_np)
      for name, cfg in (("ref", cfg_ref), ("fp8_full", cfg_fp8_full), ("fp8_small", cfg_fp8_small)):
        moe = muse_spark.MuseSparkMoE(cfg, mesh, rngs=nnx.Rngs(params=83))
        nnx.update(moe, nnx.state(moe_ref))
        (_, y), (grads, dx) = eval_moe_and_grads(moe, xb, pr, all_to_shard0)
        overflow = int(np.sum(np.asarray(moe.moe_overflow_slots.get_value()[-1])))
        flat_grads = {"/".join(map(str, path)): np.asarray(var.get_value()) for path, var in nnx.to_flat_state(grads)}
        results[name] = (np.asarray(y), np.asarray(dx), flat_grads, overflow)

      # TPU `quantization="fp8_full"` without `use_tokamax_gmm=True, use_gmm_v2=True` must fail fast instead of
      # silently running dequantized float32 `ragged_dot`.
      moe_no_gmm = muse_spark.MuseSparkMoE(
          make_config(quantization="fp8_full", dtype="bfloat16", use_tokamax_gmm=False, use_gmm_v2=False, **mesh_kw),
          mesh,
          rngs=nnx.Rngs(params=84),
      )
      with mock.patch.object(muse_spark, "_mesh_is_tpu", return_value=True):
        with self.assertRaisesRegex(ValueError, "use_tokamax_gmm=True"):
          moe_no_gmm(xb.astype(jnp.bfloat16))

    y_ref, dx_ref, grads_ref, overflow_ref = results["ref"]
    self.assertEqual(overflow_ref, 0)
    self.assertEqual(results["fp8_full"][3], 0)
    self.assertEqual(results["fp8_small"][3], 3584)
    self.assertGreater(float(np.linalg.norm(grads_ref["gate/kernel"])), 0.0)
    for name in ("fp8_full", "fp8_small"):
      y, dx, grads, _ = results[name]
      self.assertTrue(np.all(np.isfinite(y)), name)
      # FP8 really ran: the result is close to, but not the same as, full precision.
      self.assertFalse(np.array_equal(y, y_ref), name)
      self.assertGreaterEqual(sqnr_db(y_ref, y), FP8_MIN_SQNR_DB, f"{name} output")
      self.assertGreaterEqual(sqnr_db(dx_ref, dx), FP8_MIN_SQNR_DB, f"{name} input gradient")
      for key, grad_ref in grads_ref.items():
        if np.any(grad_ref):
          self.assertGreaterEqual(sqnr_db(grad_ref, grads[key]), FP8_MIN_SQNR_DB, f"{name} gradient of {key}")
    # The overflow loop does not drop tokens under FP8 either.
    self.assertGreaterEqual(sqnr_db(results["fp8_full"][0], results["fp8_small"][0]), FP8_MIN_SQNR_DB)

    # Verify `FSDP=2 x EP=2, chunks=2, ragged_buffer_factor=0.5` pre-quantizes and gathers `PreGatheredFp8Weight`
    # once before `_route_overflow` so zero `all-gather`/`reduce-scatter` ops appear inside `while` bodies,
    # and forward/backward outputs + gradients match full-precision `FSDP=2 x EP=2`.
    fsdp_ep_base = {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "ici_fsdp_parallelism": 2,
        "ici_expert_parallelism": 2,
        "max_target_length": seq,
    }
    cfg_fsdp_ref = make_config(ragged_buffer_factor=-1.0, muse_spark_moe_num_token_chunks=1, **fsdp_ep_base)
    cfg_fsdp_fp8 = make_config(
        quantization="fp8_full",
        ragged_buffer_factor=0.5,
        muse_spark_moe_num_token_chunks=2,
        **fsdp_ep_base,
    )
    mesh_fsdp_fp8 = Mesh(maxtext_utils.create_device_mesh(cfg_fsdp_fp8), cfg_fsdp_fp8.mesh_axes)
    with jax.set_mesh(mesh_fsdp_fp8), nn_partitioning.axis_rules(cfg_fsdp_fp8.logical_axis_rules):
      moe_fsdp_ref = muse_spark.MuseSparkMoE(cfg_fsdp_ref, mesh_fsdp_fp8, rngs=nnx.Rngs(params=84))
      moe_fsdp_fp8 = muse_spark.MuseSparkMoE(cfg_fsdp_fp8, mesh_fsdp_fp8, rngs=nnx.Rngs(params=85))
      nnx.update(moe_fsdp_fp8, nnx.state(moe_fsdp_ref))
      hlo_fp8 = eval_moe_and_grads.lower(moe_fsdp_fp8, xb, pr, all_to_shard0).compile().as_text()
      _assert_no_while_body_collectives(self, hlo_fp8)
      (_, y_fsdp_ref), (g_fsdp_ref, dx_fsdp_ref) = eval_moe_and_grads(moe_fsdp_ref, xb, pr, all_to_shard0)
      (_, y_fsdp_fp8), (g_fsdp_fp8, dx_fsdp_fp8) = eval_moe_and_grads(moe_fsdp_fp8, xb, pr, all_to_shard0)
    self.assertFalse(np.array_equal(np.asarray(y_fsdp_fp8), np.asarray(y_fsdp_ref)))
    self.assertGreaterEqual(sqnr_db(np.asarray(y_fsdp_ref), np.asarray(y_fsdp_fp8)), FP8_MIN_SQNR_DB, "fsdp_fp8 output")
    self.assertGreaterEqual(sqnr_db(np.asarray(dx_fsdp_ref), np.asarray(dx_fsdp_fp8)), FP8_MIN_SQNR_DB, "fsdp_fp8 dx")
    for p_ref, p_fp8 in zip(nnx.to_flat_state(g_fsdp_ref), nnx.to_flat_state(g_fsdp_fp8)):
      gr, gf = np.asarray(p_ref[1].get_value()), np.asarray(p_fp8[1].get_value())
      if np.any(gr):
        self.assertGreaterEqual(sqnr_db(gr, gf), FP8_MIN_SQNR_DB, f"fsdp_fp8 grad {p_ref[0]}")

    # Also verify `ep_size == 1` (`ici_expert_parallelism=1, ici_fsdp_parallelism=4`) runs routed experts in FP8.
    ep1_kw = {"ici_fsdp_parallelism": 4, "ici_expert_parallelism": 1, "num_experts_per_tok": 2}
    cfg_ep1_ref = make_config(**ep1_kw)
    cfg_ep1_fp8 = make_config(quantization="fp8_full", **ep1_kw)
    mesh_ep1 = Mesh(maxtext_utils.create_device_mesh(cfg_ep1_ref), cfg_ep1_ref.mesh_axes)
    x_ep1 = jnp.asarray(np.random.RandomState(280).randn(4, 16, EMB).astype(np.float32))
    pr_ep1 = jnp.asarray(np.random.RandomState(281).randn(4, 16, EMB).astype(np.float32))
    with jax.set_mesh(mesh_ep1), nn_partitioning.axis_rules(cfg_ep1_ref.logical_axis_rules):
      moe_ep1_ref = muse_spark.MuseSparkMoE(cfg_ep1_ref, mesh_ep1, rngs=nnx.Rngs(params=282))
      moe_ep1_fp8 = muse_spark.MuseSparkMoE(cfg_ep1_fp8, mesh_ep1, rngs=nnx.Rngs(params=283))
      nnx.update(moe_ep1_fp8, nnx.state(moe_ep1_ref))
      (_, y_ep1_ref), (g_ep1_ref, dx_ep1_ref) = eval_moe_and_grads(moe_ep1_ref, x_ep1, pr_ep1, None)
      (_, y_ep1_fp8), (g_ep1_fp8, dx_ep1_fp8) = eval_moe_and_grads(moe_ep1_fp8, x_ep1, pr_ep1, None)
    y_ep1_ref_np, y_ep1_fp8_np = np.asarray(y_ep1_ref), np.asarray(y_ep1_fp8)
    self.assertFalse(np.array_equal(y_ep1_fp8_np, y_ep1_ref_np), "ep_size == 1 unexpectedly ran in full precision")
    self.assertGreaterEqual(sqnr_db(y_ep1_ref_np, y_ep1_fp8_np), FP8_MIN_SQNR_DB, "ep_size==1 output")
    self.assertGreaterEqual(sqnr_db(np.asarray(dx_ep1_ref), np.asarray(dx_ep1_fp8)), FP8_MIN_SQNR_DB, "ep_size==1 dx")
    for p_ref, p_fp8 in zip(nnx.to_flat_state(g_ep1_ref), nnx.to_flat_state(g_ep1_fp8)):
      gr, gf = np.asarray(p_ref[1].get_value()), np.asarray(p_fp8[1].get_value())
      if np.any(gr):
        self.assertGreaterEqual(sqnr_db(gr, gf), FP8_MIN_SQNR_DB, f"ep_size==1 grad {p_ref[0]}")

  def test_maybe_quantize_model_on_muse_spark_preserves_exclusions(self):
    """`quantizations.maybe_quantize_model` quantizes decoder linears while leaving excluded modules unquantized."""
    base_kw = {
        "base_emb_dim": 256,
        "moe_expert_input_dim": 256,
        "base_moe_mlp_dim": 256,
        "base_num_decoder_layers": 2,
        "first_num_dense_layers": 0,
        "scan_layers": False,
        "per_device_batch_size": 1,
        "load_balance_loss_weight": 1.0,
    }
    cfg_ref = make_config(**base_kw)
    cfg_fp8 = make_config(
        quantization="fp8_full",
        use_qwix_quantization=True,
        quantize_router_proj=False,
        quantize_logits_proj=True,
        **base_kw,
    )
    rules = quantizations.get_quantization_rule(cfg_fp8)
    self.assertEqual([r.weight_qtype for r in rules], [None, None, None, jnp.float8_e4m3fn])
    cfg_qproj_bf16 = make_config(
        quantization="fp8_full",
        use_qwix_quantization=True,
        quantize_router_proj=False,
        sa_qproj_bwd_bf16=True,
        **base_kw,
    )
    rules_qproj_bf16 = quantizations.get_quantization_rule(cfg_qproj_bf16)
    self.assertEqual([r.bwd_qtype for r in rules_qproj_bf16], [None, None, None, None, jnp.float8_e5m2])

    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref, jax.devices()[:1]), cfg_ref.mesh_axes)
    toks = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    pos = jnp.arange(8, dtype=jnp.int32)[None, :]
    seg = jnp.ones_like(toks)
    probe = jnp.asarray(np.random.RandomState(294).randn(1, 8, VOCAB).astype(np.float32))
    forced = jnp.broadcast_to(jnp.array([[[[0, 1], [2, 3]]]], dtype=jnp.int32), (1, 8, 2, TOPK))

    def sqnr_db(reference, candidate):
      reference = np.asarray(reference, dtype=np.float64)
      noise = np.sum(np.square(reference - np.asarray(candidate, dtype=np.float64)))
      return 10.0 * np.log10(np.sum(np.square(reference)) / max(noise, 1e-30))

    @nnx.jit
    def model_val_and_grad(mod):
      def loss_fn(m):
        out = m(toks, pos, seg, enable_dropout=False, forced_routed_experts=forced)
        return jnp.sum(out * probe), out

      return nnx.value_and_grad(loss_fn, has_aux=True)(mod)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      # 1. Non-excluded attention and MoE projections ARE quantized by Qwix (`fp8_full`), transient Intermediates are
      #    popped, and locked-routing forward logits + backward parameter gradients stay finite and accurate (with
      #    `sa_qproj_bwd_bf16=True` recovering full-precision `query` backward cotangents across scaleless QK-norm).
      m_ref = model_creation_utils.from_config(cfg_ref, mesh=mesh, rngs=nnx.Rngs(params=290, dropout=291))
      for layer_attr in ("layers_0", "layers_1"):
        lyr = getattr(m_ref.decoder, layer_attr)
        lyr.moe_block.gate.kernel.set_value(jnp.zeros_like(lyr.moe_block.gate.kernel.get_value()))
      m_fp8_raw = model_creation_utils.from_config(cfg_fp8, mesh=mesh, rngs=nnx.Rngs(params=292, dropout=293))
      nnx.update(m_fp8_raw, nnx.state(m_ref))
      m_fp8 = quantizations.maybe_quantize_model(m_fp8_raw, cfg_fp8)
      self.assertEqual(list(nnx.to_flat_state(nnx.state(m_fp8, nnx.Intermediate))), [])

      m_qbf16_raw = model_creation_utils.from_config(cfg_qproj_bf16, mesh=mesh, rngs=nnx.Rngs(params=292, dropout=293))
      nnx.update(m_qbf16_raw, nnx.state(m_ref))
      m_qbf16 = quantizations.maybe_quantize_model(m_qbf16_raw, cfg_qproj_bf16)

      (_, logits_ref_arr), grads_ref = model_val_and_grad(m_ref)
      (_, logits_fp8_arr), grads_fp8 = model_val_and_grad(m_fp8)
      _, grads_qbf16 = model_val_and_grad(m_qbf16)
      logits_ref = np.asarray(logits_ref_arr)
      logits_fp8 = np.asarray(logits_fp8_arr)
      self.assertTrue(np.all(np.isfinite(logits_fp8)))
      self.assertFalse(np.array_equal(logits_fp8, logits_ref))
      self.assertGreaterEqual(sqnr_db(logits_ref, logits_fp8), 14.0, "2-layer fp8_full logits SQNR")
      g_ref_dict = {"/".join(map(str, p)): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(grads_ref)}
      g_fp8_dict = {"/".join(map(str, p)): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(grads_fp8)}
      g_qbf16_dict = {"/".join(map(str, p)): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(grads_qbf16)}
      for arr in g_fp8_dict.values():
        self.assertTrue(np.all(np.isfinite(arr)))
      for qkey, min_db in (
          ("decoder/layers_1/moe_block/wi_0", 12.0),
          ("decoder/layers_1/self_attention/out/kernel", 12.0),
          ("decoder/layers_1/self_attention/gate_proj/kernel", 10.0),
      ):
        self.assertGreater(float(np.linalg.norm(g_ref_dict[qkey])), 0.0, qkey)
        self.assertGreaterEqual(sqnr_db(g_ref_dict[qkey], g_fp8_dict[qkey]), min_db, f"grad SQNR for {qkey}")
      q_key = "decoder/layers_1/self_attention/query/kernel"
      self.assertGreater(float(np.linalg.norm(g_ref_dict[q_key])), 0.0, q_key)
      self.assertGreaterEqual(sqnr_db(g_ref_dict[q_key], g_qbf16_dict[q_key]), 10.0, "sa_qproj_bwd_bf16 query grad SQNR")
      self.assertGreater(
          sqnr_db(g_ref_dict[q_key], g_qbf16_dict[q_key]),
          sqnr_db(g_ref_dict[q_key], g_fp8_dict[q_key]),
      )

      # Zeroing only `moe_block.post_expert_proj` still differs from `m_ref`, confirming `self_attention` runs in FP8.
      m_attn_only_ref = model_creation_utils.from_config(cfg_ref, mesh=mesh, rngs=nnx.Rngs(params=294, dropout=295))
      for layer_attr in ("layers_0", "layers_1"):
        lyr = getattr(m_attn_only_ref.decoder, layer_attr)
        lyr.moe_block.post_expert_proj.kernel.set_value(jnp.zeros_like(lyr.moe_block.post_expert_proj.kernel.get_value()))
      m_attn_only_fp8_raw = model_creation_utils.from_config(cfg_fp8, mesh=mesh, rngs=nnx.Rngs(params=294, dropout=295))
      nnx.update(m_attn_only_fp8_raw, nnx.state(m_attn_only_ref))
      m_attn_only_fp8 = quantizations.maybe_quantize_model(m_attn_only_fp8_raw, cfg_fp8)
      self.assertFalse(
          np.array_equal(
              np.asarray(m_attn_only_fp8(toks, pos, seg, enable_dropout=False)),
              np.asarray(m_attn_only_ref(toks, pos, seg, enable_dropout=False)),
          )
      )

      # 2. Zeroing `self_attention.out` and `moe_block.post_expert_proj` routes the forward pass exclusively through
      # excluded modules (`token_embedder`, norms, `MuseSparkResidualGate`, `moe_block.gate`, and `logits_dense` even
      # with `quantize_logits_proj=True`), which must stay bit-exact with the unquantized reference model.
      m_zero_ref = model_creation_utils.from_config(cfg_ref, mesh=mesh, rngs=nnx.Rngs(params=296, dropout=297))
      for layer_attr in ("layers_0", "layers_1"):
        lyr = getattr(m_zero_ref.decoder, layer_attr)
        lyr.self_attention.out.kernel.set_value(jnp.zeros_like(lyr.self_attention.out.kernel.get_value()))
        lyr.moe_block.post_expert_proj.kernel.set_value(jnp.zeros_like(lyr.moe_block.post_expert_proj.kernel.get_value()))
      m_zero_fp8_raw = model_creation_utils.from_config(cfg_fp8, mesh=mesh, rngs=nnx.Rngs(params=298, dropout=299))
      nnx.update(m_zero_fp8_raw, nnx.state(m_zero_ref))
      m_zero_fp8 = quantizations.maybe_quantize_model(m_zero_fp8_raw, cfg_fp8)

      logits_zero_ref = np.asarray(m_zero_ref(toks, pos, seg, enable_dropout=False))
      logits_zero_fp8 = np.asarray(m_zero_fp8(toks, pos, seg, enable_dropout=False))
      np.testing.assert_array_equal(logits_zero_fp8, logits_zero_ref)

      sown_ref = {k: np.asarray(v.get_value()) for k, v in nnx.to_flat_state(nnx.state(m_zero_ref, nnx.Intermediate))}
      sown_fp8 = {k: np.asarray(v.get_value()) for k, v in nnx.to_flat_state(nnx.state(m_zero_fp8, nnx.Intermediate))}
      self.assertTrue(sown_fp8)
      self.assertEqual(set(sown_fp8), set(sown_ref))
      for k in sown_ref:
        np.testing.assert_array_equal(sown_fp8[k], sown_ref[k])

    # 3. Sliding-window boundary check (`seq=12 > sliding_window_size=4`): mutating token 0 leaves final token 11
    #    unchanged on a sliding-window layer (`layers_0`) while changing a global-attention layer (`layers_1`).
    cfg_swa = make_config(base_num_decoder_layers=2, scan_layers=False, max_target_length=12, sliding_window_size=4)
    x_a = jnp.asarray(np.random.RandomState(295).randn(1, 12, EMB).astype(np.float32))
    x_b = x_a.at[:, 0, :].set(-x_a[:, 0, :] + 3.0)
    pos_12 = jnp.arange(12, dtype=jnp.int32)[None, :]
    seg_12 = jnp.ones((1, 12), dtype=jnp.int32)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_swa.logical_axis_rules):
      m_swa = model_creation_utils.from_config(cfg_swa, mesh=mesh, rngs=nnx.Rngs(params=401, dropout=402))
      attn_swa = m_swa.decoder.layers_0.self_attention
      attn_glb = m_swa.decoder.layers_1.self_attention
      self.assertFalse(attn_swa.is_full_attention)
      self.assertTrue(attn_glb.is_full_attention)
      y_swa_a, _ = attn_swa(x_a, x_a, pos_12, seg_12, deterministic=True)
      y_swa_b, _ = attn_swa(x_b, x_b, pos_12, seg_12, deterministic=True)
      y_glb_a, _ = attn_glb(x_a, x_a, pos_12, seg_12, deterministic=True)
      y_glb_b, _ = attn_glb(x_b, x_b, pos_12, seg_12, deterministic=True)
    np.testing.assert_allclose(np.asarray(y_swa_a[:, -1, :]), np.asarray(y_swa_b[:, -1, :]), atol=1e-6, rtol=1e-6)
    self.assertGreater(float(np.max(np.abs(np.asarray(y_glb_a[:, -1, :]) - np.asarray(y_glb_b[:, -1, :])))), 1e-4)

  def test_tpu7x_256_chip_fp8_training_flags(self):
    """The FP8 training flags of the user guide parse and are rejected for other quantization modes."""
    cfg = pyconfig.initialize(
        ["", BASE_CONFIG],
        model_name="muse-spark-816b-a42b",
        skip_jax_distributed_system=True,
        hardware="cpu",
        compile_topology="tpu7x-256",
        compile_topology_num_slices=1,
        **{**TPU7X_256_TRAINING_FLAGS, **TPU7X_256_FP8_TRAINING_FLAGS},
    )
    self.assertEqual(cfg.quantization, "fp8_full")
    self.assertIsNone(quantizations.configure_quantization(cfg))
    for bad_cfg in (
        make_config(quantization="int8"),
        make_config(quantization="serve_fp8_weight", weight_dtype="float8_e4m3fn"),
        types.SimpleNamespace(
            decoder_block=config_types.DecoderBlockType.MUSE_SPARK,
            quantization="int8",
            use_batch_split_schedule=True,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "only supports quantization='fp8_full'"):
        quantizations.configure_quantization(bad_cfg)
      with self.assertRaisesRegex(ValueError, "only supports quantization='fp8_full'"):
        quantizations.get_qt_provider(bad_cfg)

  def test_tpu7x_256_chip_training_flags_and_mesh_preset(self):
    """The 256-chip training flags of the user guide parse, and their mesh puts EP on the z ring, FSDP on (x, y, core)."""
    devices = [
        types.SimpleNamespace(id=idx, coords=(x, y, z), core_on_chip=core, process_index=0, platform="tpu")
        for idx, (x, y, z, core) in enumerate(
            (x, y, z, core) for x in range(4) for y in range(8) for z in range(8) for core in range(2)
        )
    ]
    cfg = pyconfig.initialize(
        ["", BASE_CONFIG],
        model_name="muse-spark-816b-a42b",
        skip_jax_distributed_system=True,
        hardware="cpu",
        compile_topology="tpu7x-256",
        compile_topology_num_slices=1,
        **TPU7X_256_TRAINING_FLAGS,
    )
    self.assertEqual(cfg.custom_mesh, "ep8_z_fsdp64_xycore")
    self.assertEqual(cfg.sa_num_stacked_q_heads, 8)
    self.assertEqual(cfg.ragged_buffer_factor, 1.5)
    self.assertFalse(cfg.muse_spark_moe_dropless_overflow)
    self.assertTrue(make_config().muse_spark_moe_dropless_overflow)
    mesh = maxtext_utils.create_device_mesh(cfg, devices)
    axes = list(cfg.mesh_axes)
    fsdp_axis, expert_axis = axes.index("fsdp"), axes.index("expert")
    self.assertEqual(mesh.shape[fsdp_axis], 64)
    self.assertEqual(mesh.shape[expert_axis], 8)
    self.assertEqual(len({d.id for d in mesh.flat}), 512)
    by_position = np.moveaxis(mesh, (fsdp_axis, expert_axis), (0, 1)).reshape(64, 8)
    for fsdp_idx in range(64):
      for expert_idx in range(8):
        dev = by_position[fsdp_idx, expert_idx]
        self.assertEqual(dev.coords[2], expert_idx)
        self.assertEqual((dev.coords[0] * 8 + dev.coords[1]) * 2 + dev.core_on_chip, fsdp_idx)

    # Mismatched topologies or incompatible mesh flags raise ValueError instead of silently falling back.
    with self.assertRaisesRegex(ValueError, "requires a single-slice 512-device"):
      maxtext_utils.create_device_mesh(cfg, devices[:64])
    cfg_bad_split = pyconfig.initialize(
        ["", BASE_CONFIG],
        model_name="muse-spark-816b-a42b",
        skip_jax_distributed_system=True,
        hardware="cpu",
        compile_topology="tpu7x-256",
        compile_topology_num_slices=1,
        allow_split_physical_axes=True,
        **TPU7X_256_TRAINING_FLAGS,
    )
    with self.assertRaisesRegex(ValueError, "incompatible with allow_split_physical_axes"):
      maxtext_utils.create_device_mesh(cfg_bad_split, devices)

    # Verify `M_MOE_COMBINED` and `M_PRE_EXPERT_NORM` environment variable overrides work via `base.yml`.
    from unittest import mock  # pylint: disable=import-outside-toplevel

    with mock.patch.dict(os.environ, {"M_MOE_COMBINED": "device", "M_PRE_EXPERT_NORM": "offload"}):
      cfg_env = make_config(remat_policy="custom")
    self.assertEqual(cfg_env.moe_combined, "device")
    self.assertEqual(cfg_env.pre_expert_norm, "offload")

  def test_moe_ep_sequence_axis_sharding_matches_single_device(self):
    """When `logical_axis_rules` shards sequence along the EP axis, tokens are gathered rather than `psum`-ed."""
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices for EP=4 sequence-sharded test.")
    seq_ep_rules = [
        ("activation_batch", ()),
        ("activation_norm_length", "expert"),
        ("exp", "expert"),
    ]
    cfg_ref = make_config(ici_fsdp_parallelism=1, ici_expert_parallelism=1)
    cfg_seq_ep = make_config(
        ici_fsdp_parallelism=1,
        ici_expert_parallelism=4,
        logical_axis_rules=seq_ep_rules,
    )
    mesh_ref = Mesh(maxtext_utils.create_device_mesh(cfg_ref, jax.devices()[:1]), cfg_ref.mesh_axes)
    mesh_ep = Mesh(maxtext_utils.create_device_mesh(cfg_seq_ep), cfg_seq_ep.mesh_axes)
    x_np = np.random.RandomState(190).randn(2, 8, EMB).astype(np.float32)

    with jax.set_mesh(mesh_ref), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      moe_ref = muse_spark.MuseSparkMoE(cfg_ref, mesh_ref, rngs=nnx.Rngs(params=191))
      ref_flat = dict(nnx.to_flat_state(nnx.state(moe_ref)))
      x_ref = jax.device_put(jnp.asarray(x_np), jax.sharding.NamedSharding(mesh_ref, jax.sharding.PartitionSpec()))
      y_ref, _ = moe_ref(x_ref)

    with jax.set_mesh(mesh_ep), nn_partitioning.axis_rules(cfg_seq_ep.logical_axis_rules):
      moe_ep = muse_spark.MuseSparkMoE(cfg_seq_ep, mesh_ep, rngs=nnx.Rngs(params=192))
      updated = []
      for path, dst_var in nnx.to_flat_state(nnx.state(moe_ep)):
        src_arr = np.asarray(ref_flat[path].get_value())
        shd = getattr(dst_var.get_value(), "sharding", None)
        placed = jax.device_put(jnp.asarray(src_arr), shd) if shd is not None else jnp.asarray(src_arr)
        updated.append((path, dst_var.replace(placed)))
      nnx.update(moe_ep, nnx.from_flat_state(updated))
      x_ep = jax.device_put(
          jnp.asarray(x_np),
          jax.sharding.NamedSharding(mesh_ep, jax.sharding.PartitionSpec(None, "expert", None)),
      )
      y_ep, _ = moe_ep(x_ep)

    np.testing.assert_allclose(np.asarray(y_ep), np.asarray(y_ref), atol=1e-5, rtol=1e-5)

  def test_scanned_vs_unscanned_decoder_get_layers_and_kv_caches(self):
    num_layers = 6
    cfg_u = make_config(
        base_num_decoder_layers=num_layers, first_num_dense_layers=1, scan_layers=False, return_routed_experts=True
    )
    cfg_s = make_config(
        base_num_decoder_layers=num_layers, first_num_dense_layers=1, scan_layers=True, return_routed_experts=True
    )
    # 6 layers with cycle=4 and first_num_dense_layers=1 -> 2 prefix layers (layer 0 dense, layer 1 MoE) + 1 scanned cycle.
    self.assertEqual(muse_spark.scan_layout(cfg_s), (2, 1))

    mu = model_creation_utils.from_config(cfg_u, mesh=self.mesh, rngs=nnx.Rngs(params=50, dropout=51))
    ms = model_creation_utils.from_config(cfg_s, mesh=self.mesh, rngs=nnx.Rngs(params=52, dropout=53))
    copy_unscanned_to_scanned(mu, ms, cfg_s)

    self.assertEqual(
        ms.decoder.get_layers(),
        [ms.decoder.layers_0, ms.decoder.layers_1, ms.decoder.scanned_blocks],
    )

    batch, seq = jax.device_count(), 6
    ids = jnp.asarray(np.random.RandomState(60).randint(1, VOCAB, (batch, seq)), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))
    seg = jnp.ones((batch, seq), dtype=jnp.int32)
    probe = jnp.asarray(np.random.RandomState(61).randn(batch, seq, VOCAB), dtype=jnp.float32)

    @nnx.jit
    def eval_decoder_and_grads(mod):
      def model_loss(m):
        logits = m(decoder_input_tokens=ids, decoder_positions=pos, decoder_segment_ids=seg, enable_dropout=False)
        return jnp.sum(logits * probe), logits

      return nnx.value_and_grad(model_loss, has_aux=True)(mod)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      (_, logits_u), grads_u = eval_decoder_and_grads(mu)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      (_, logits_s), grads_s = eval_decoder_and_grads(ms)

    def assert_grad_close(actual_var, expected_var, name: str):
      actual = np.asarray(actual_var, dtype=np.float32)
      expected = np.asarray(expected_var, dtype=np.float32)
      scale = float(np.max(np.abs(expected)))
      self.assertGreater(scale, 0.0, f"Zero gradient for {name}")
      self.assertLess(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)), 1e-4, name)
      np.testing.assert_allclose(actual, expected, atol=1e-4 * scale, rtol=1e-3, err_msg=name)

    np.testing.assert_allclose(logits_s, logits_u, atol=1e-5, rtol=1e-5)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      _, _, _, routed_u = mu.decoder(
          shared_embedding=mu.token_embedder,
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      _, _, _, routed_s = ms.decoder(
          shared_embedding=ms.token_embedder,
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
    self.assertEqual(routed_s.shape, (num_layers - 1, batch * seq, TOPK))
    np.testing.assert_array_equal(np.asarray(routed_s), np.asarray(routed_u))
    assert_grad_close(
        grads_s.token_embedder.embedding.get_value(),
        grads_u.token_embedder.embedding.get_value(),
        "token_embedder",
    )
    assert_grad_close(
        grads_s.decoder.logits_dense.kernel.get_value(),
        grads_u.decoder.logits_dense.kernel.get_value(),
        "logits_dense",
    )
    for sub_id in range(cfg_s.inhomogeneous_layer_cycle_interval):
      u_flat = dict(nnx.to_flat_state(getattr(grads_u.decoder, f"layers_{2 + sub_id}")))
      s_flat = dict(nnx.to_flat_state(getattr(grads_s.decoder.scanned_blocks, f"layers_{sub_id}")))
      for path, u_var in u_flat.items():
        s_sliced = np.take(np.asarray(s_flat[path].get_value()), 0, axis=cfg_s.param_scan_axis)
        assert_grad_close(s_sliced, u_var.get_value(), f"scanned_sub_{sub_id}/{path}")

    # Verify external `kv_caches` threading through prefix + scanned cycles, and mutual exclusion with forced routing.
    kv_list = [jnp.array([float(i)], dtype=jnp.float32) for i in range(num_layers)]
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      logits_kv = ms(
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          enable_dropout=False,
          kv_caches=kv_list,
      )
    np.testing.assert_allclose(logits_kv, logits_s, atol=1e-5, rtol=1e-5)
    for i in range(num_layers):
      np.testing.assert_array_equal(np.asarray(kv_list[i]), np.array([float(i)], dtype=np.float32))

    # Verify forced routing with `-1` padding sentinels works identically on unscanned and scanned models,
    # and raises NotImplementedError when combined with external `kv_caches`.
    forced = jnp.zeros((batch, seq, num_layers, TOPK), dtype=jnp.int32)
    forced = forced.at[:, :, :, 0].set(1).at[:, :, :, 1].set(3).at[:, -2:, :, :].set(-1)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      logits_forced_u = mu(
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          enable_dropout=False,
          forced_routed_experts=forced,
      )
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      logits_forced_s = ms(
          decoder_input_tokens=ids,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          enable_dropout=False,
          forced_routed_experts=forced,
      )
    self.assertTrue(np.all(np.isfinite(np.asarray(logits_forced_s))))
    np.testing.assert_allclose(logits_forced_s, logits_forced_u, atol=1e-5, rtol=1e-5)

    with self.assertRaises(NotImplementedError):
      with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
        ms(
            decoder_input_tokens=ids,
            decoder_positions=pos,
            decoder_segment_ids=seg,
            enable_dropout=False,
            kv_caches=kv_list,
            forced_routed_experts=forced,
        )

    # Verify `remat_policy_override="none"` disables remat cleanly in `_apply_muse_spark_scanned_blocks`.
    ms.decoder.remat_policy_override = "none"
    try:
      with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
        logits_no_remat = ms(
            decoder_input_tokens=ids,
            decoder_positions=pos,
            decoder_segment_ids=seg,
            enable_dropout=False,
        )
      np.testing.assert_allclose(logits_no_remat, logits_s, atol=1e-5, rtol=1e-5)
    finally:
      ms.decoder.remat_policy_override = None

  def test_muon_dimension_numbers_full_config_tflops_and_pipeline_rejection(self):
    cfg_s = make_config(scan_layers=True, base_num_decoder_layers=10, first_num_dense_layers=1, opt_type="muon")
    _, abstract_ms = model_creation_utils.create_nnx_abstract_model(cfg_s, mesh=self.mesh)
    mdn_tree = nnx.to_pure_dict(muon_utils.get_muon_weight_dimension_numbers(abstract_ms, cfg_s))["decoder"]

    # Both unrolled prefix and 2D scanned `MuseSparkResidualGate.gate` vectors must be excluded (`None` -> AdamW).
    self.assertIsNone(mdn_tree["layers_0"]["post_attention_residual_gate"]["gate"])
    self.assertIsNone(mdn_tree["layers_0"]["post_feedforward_residual_gate"]["gate"])
    self.assertIsNone(mdn_tree["scanned_blocks"]["layers_0"]["post_attention_residual_gate"]["gate"])
    self.assertIsNone(mdn_tree["scanned_blocks"]["layers_0"]["post_feedforward_residual_gate"]["gate"])

    # Both 3D unscanned and 4D scanned `MuseSparkAttention.gate_proj` must use head-aware axes `((0,), (-2, -1))`.
    for attn_mdn in (
        mdn_tree["layers_0"]["self_attention"]["gate_proj"]["kernel"],
        mdn_tree["scanned_blocks"]["layers_0"]["self_attention"]["gate_proj"]["kernel"],
    ):
      self.assertIsNotNone(attn_mdn)
      self.assertEqual(tuple(attn_mdn.reduction_axis), (0,))
      self.assertEqual(tuple(attn_mdn.output_axis), (-2, -1))

    # Full `muse-spark-816b-a42b.yml` loads (including `attention="flash"` without `override_model_config=True`)
    # and computes positive analytical training TFLOPs.
    cfg_816b = pyconfig.initialize(
        ["", BASE_CONFIG],
        model_name="muse-spark-816b-a42b",
        attention="flash",
        skip_jax_distributed_system=True,
        hardware="cpu",
    )
    total_tflops, qw_tflops, att_tflops = maxtext_utils.calculate_tflops_training_per_device(cfg_816b, log=False)
    self.assertGreater(qw_tflops, 0.0)
    self.assertGreater(att_tflops, 0.0)
    self.assertAlmostEqual(total_tflops, qw_tflops + att_tflops, places=5)

    # Enabling pipeline parallelism or constant-global-batch MoE padding is rejected.
    with self.assertRaisesRegex(ValueError, "muse_spark does not support pipeline parallelism"):
      make_config(ici_pipeline_parallelism=2, num_layers_per_pipeline_stage=1)
    with self.assertRaisesRegex(ValueError, "decoder_block=muse_spark"):
      make_config(
          elastic_simulate_job_devices=8,
          elastic_global_batch_size_mode="constant_padded",
          dataset_type="synthetic",
      )

    # `RLConfig` mirrors the Muse Spark guards, and `Decoder` catches pipeline parallelism without `MaxTextConfig`.
    rl_base = {
        "model_name": "muse-spark-816b-a42b",
        "decoder_block": "muse_spark",
        "tokenizer_path": "meta-models/Muse-Spark-Open-816B-A42B",
        "num_experts": EXPERTS,
        "routed_bias": True,
    }
    with self.assertRaisesRegex(ValueError, "muse_spark does not support pipeline parallelism"):
      config_types.RLConfig(**rl_base, ici_pipeline_parallelism=2)
    with self.assertRaisesRegex(ValueError, "not supported with 'explicit' sharding"):
      config_types.RLConfig(**rl_base, shard_mode="explicit")
    with self.assertRaisesRegex(ValueError, "Loss-free load balancing is only supported"):
      config_types.RLConfig(**rl_base, routed_bias_update_rate=0.001)
    with self.assertRaisesRegex(ValueError, "sa_num_stacked_q_heads must be >= 1"):
      config_types.RLConfig(**rl_base, sa_num_stacked_q_heads=0)
    with self.assertRaisesRegex(ValueError, "must divide num_query_heads // num_kv_heads"):
      config_types.RLConfig(**rl_base, base_num_query_heads=128, base_num_kv_heads=16, sa_num_stacked_q_heads=3)
    self.assertEqual(
        config_types.RLConfig(
            **rl_base, base_num_query_heads=128, base_num_kv_heads=16, sa_num_stacked_q_heads=8
        ).sa_num_stacked_q_heads,
        8,
    )
    cfg_no_flag = make_config()
    object.__setattr__(cfg_no_flag, "using_pipeline_parallelism", False)
    object.__setattr__(cfg_no_flag, "ici_pipeline_parallelism", 2)
    with self.assertRaisesRegex(ValueError, "muse_spark does not support pipeline parallelism"):
      model_creation_utils.create_nnx_abstract_model(cfg_no_flag, mesh=self.mesh)

  def test_moe_overflow_fsdp_pregather_bf16_accum_and_diloco_axes(self):
    """Verifies chunk-0 FSDP pre-gather outside `while`, `float32` overflow accumulation, `diloco` `pmax`, and guards."""
    from unittest import mock  # pylint: disable=import-outside-toplevel

    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices for FSDP=2 x EP=2 and DiLoCo=2 x EP=2 overflow tests.")

    # 1. TPU `dtype=float32` EP is rejected early with a clear ValueError.
    cfg_f32_ep = make_config(dtype="float32", ici_fsdp_parallelism=1, ici_expert_parallelism=2)
    mesh_ep2 = Mesh(maxtext_utils.create_device_mesh(cfg_f32_ep, jax.devices()[:2]), cfg_f32_ep.mesh_axes)
    with jax.set_mesh(mesh_ep2), nn_partitioning.axis_rules(cfg_f32_ep.logical_axis_rules):
      moe_f32 = muse_spark.MuseSparkMoE(cfg_f32_ep, mesh_ep2, rngs=nnx.Rngs(params=301))
      with mock.patch.object(muse_spark, "_mesh_is_tpu", return_value=True):
        with self.assertRaisesRegex(ValueError, "16-bit activation dtype"):
          moe_f32(jnp.zeros((2, 8, EMB), dtype=jnp.float32))

    # Sanity-check the transitive HLO parser catches collectives inside `%wide.region_...` and nested `calls=%...`.
    synthetic_hlo = """
%nested_sub.1 (p: f32[]) -> f32[] {
  ROOT %rs = f32[] reduce-scatter-start(p), dimensions={0}
}
%wide.region_71.91_spmd.clone (a: f32[]) -> f32[] {
  ROOT %c = f32[] call(a), to_apply=%nested_sub.1
}
ENTRY %main.2 (x: f32[]) -> f32[] {
  ROOT %w = f32[] while(x), condition=%cond.0, body=%wide.region_71.91_spmd.clone
}
"""
    syn_visited, syn_violations = _collect_while_body_collectives(synthetic_hlo)
    self.assertEqual(syn_visited, {"wide.region_71.91_spmd.clone", "nested_sub.1"})
    self.assertEqual(len(syn_violations), 1)

    # 2. Odd-128 `bfloat16` (`128 % 256 != 0`) falls back cleanly to 2D and matches `ragged_buffer_factor=-1`
    #    on `FSDP=2 x EP=2` with `weight_dtype="bfloat16"` across `chunks=1` and `chunks=2` (including
    #    `post_expert_norm.scale` grad), and compiled HLO keeps all FSDP collectives outside `while` bodies.
    seq = 512
    bf16_kw = {
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "moe_expert_input_dim": 128,
        "base_moe_mlp_dim": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "ici_fsdp_parallelism": 2,
        "ici_expert_parallelism": 2,
        "max_target_length": seq,
    }
    cfg_full = make_config(ragged_buffer_factor=-1.0, muse_spark_moe_num_token_chunks=1, **bf16_kw)
    mesh_fsdp_ep = Mesh(maxtext_utils.create_device_mesh(cfg_full), cfg_full.mesh_axes)
    x_bf16 = jnp.asarray(np.random.RandomState(302).randn(4, seq, EMB), dtype=jnp.bfloat16)
    pr_bf16 = jnp.asarray(np.random.RandomState(303).randn(4, seq, EMB), dtype=jnp.bfloat16)
    forced_ov = jnp.broadcast_to(jnp.array([[[0, 1]]], dtype=jnp.int32), (4, seq, 2))

    @nnx.jit
    def run_bf16_and_grad(m, xin):
      def loss_fn(mod, xa):
        out, _ = mod(xa, forced_routed_experts=forced_ov)
        return jnp.mean(out.astype(jnp.float32) * pr_bf16.astype(jnp.float32)), out

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    with jax.set_mesh(mesh_fsdp_ep), nn_partitioning.axis_rules(cfg_full.logical_axis_rules):
      m_full = muse_spark.MuseSparkMoE(cfg_full, mesh_fsdp_ep, rngs=nnx.Rngs(params=305))
      (_, y_fl), (g_fl, dx_fl) = run_bf16_and_grad(m_full, x_bf16)

      for num_chunks in (1, 2):
        cfg_ov = make_config(ragged_buffer_factor=0.5, muse_spark_moe_num_token_chunks=num_chunks, **bf16_kw)
        m_ov = muse_spark.MuseSparkMoE(cfg_ov, mesh_fsdp_ep, rngs=nnx.Rngs(params=304))
        nnx.update(m_ov, nnx.state(m_full))

        hlo_txt = run_bf16_and_grad.lower(m_ov, x_bf16).compile().as_text()
        _assert_no_while_body_collectives(self, hlo_txt)

        (_, y_ov), (g_ov, dx_ov) = run_bf16_and_grad(m_ov, x_bf16)
        np.testing.assert_allclose(np.asarray(y_ov, np.float32), np.asarray(y_fl, np.float32), atol=1.5e-2, rtol=1.5e-2)
        for actual_arr, expect_arr in (
            (dx_ov, dx_fl),
            (g_ov.wi_0.get_value(), g_fl.wi_0.get_value()),
            (g_ov.wi_1.get_value(), g_fl.wi_1.get_value()),
            (g_ov.wo.get_value(), g_fl.wo.get_value()),
            (g_ov.post_expert_norm.scale.get_value(), g_fl.post_expert_norm.scale.get_value()),
        ):
          actual_g = np.asarray(actual_arr, np.float32)
          expect_g = np.asarray(expect_arr, np.float32)
          scale_g = max(float(np.max(np.abs(expect_g))), 1e-6)
          self.assertGreater(float(np.linalg.norm(expect_g)), 0.0)
          self.assertLessEqual(float(np.linalg.norm(actual_g - expect_g) / np.linalg.norm(expect_g)), 2e-2)
          np.testing.assert_allclose(actual_g, expect_g, atol=2e-2 * scale_g, rtol=3e-2)

    # 3. `ici_diloco_parallelism=2, ici_expert_parallelism=2` forward `pmax` does NOT reduce across `diloco`
    #    (no 4-device `[0, 1, 2, 3]` `all_reduce` in forward StableHLO).
    cfg_diloco = make_config(
        ici_fsdp_parallelism=1,
        ici_diloco_parallelism=2,
        ici_expert_parallelism=2,
        ragged_buffer_factor=0.5,
        max_target_length=256,
    )
    mesh_diloco = Mesh(maxtext_utils.create_device_mesh(cfg_diloco), cfg_diloco.mesh_axes)

    @nnx.jit
    def fwd_diloco(m, xin):
      out, _ = m(xin)
      return out

    with jax.set_mesh(mesh_diloco), nn_partitioning.axis_rules(cfg_diloco.logical_axis_rules):
      m_diloco = muse_spark.MuseSparkMoE(cfg_diloco, mesh_diloco, rngs=nnx.Rngs(params=306))
      fwd_mlir = fwd_diloco.lower(m_diloco, jnp.zeros((4, 256, EMB), dtype=jnp.float32)).as_text()
    self.assertNotIn("dense<[[0, 1, 2, 3]]>", fwd_mlir)

  def test_moe_fixed_capacity_drop_mode_eliminates_while_loop_and_matches_when_unskewed(self):
    """`muse_spark_moe_dropless_overflow=False` omits `while` loops, matches `True` unskewed, and drops excess slots."""
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices for FSDP=2 x EP=2 fixed-capacity drop test.")
    seq = 512
    kw = {
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "moe_expert_input_dim": 128,
        "base_moe_mlp_dim": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "ici_fsdp_parallelism": 2,
        "ici_expert_parallelism": 2,
        "max_target_length": seq,
    }
    cfg_base = make_config(
        ragged_buffer_factor=1.5,
        muse_spark_moe_dropless_overflow=True,
        muse_spark_moe_num_token_chunks=2,
        **kw,
    )
    mesh_fsdp_ep = Mesh(maxtext_utils.create_device_mesh(cfg_base), cfg_base.mesh_axes)
    x_in = jnp.asarray(np.random.RandomState(501).randn(4, seq, EMB), dtype=jnp.bfloat16)
    probe = jnp.asarray(np.random.RandomState(502).randn(4, seq, EMB), dtype=jnp.bfloat16)
    # Balanced routing: half tokens route to shard 0 ([0, 1]) and half to shard 1 ([4, 5]) -> 0 overflow under rbf=1.5.
    bal_row = jnp.array([[0, 1], [4, 5]], dtype=jnp.int32)
    forced_bal = jnp.broadcast_to(jnp.tile(bal_row, (seq // 2, 1))[None, :, :], (4, seq, 2))
    # Skewed routing: every token routes to shard 0 ([0, 1]) -> overflows rbf=0.5 (`buf_rows=512 < 2048` per chunk).
    forced_skew = jnp.broadcast_to(jnp.array([[[0, 1]]], dtype=jnp.int32), (4, seq, 2))

    @nnx.jit
    def run_moe_and_grad(m, xin, fr):
      def loss_fn(mod, xa):
        out, _ = mod(xa, forced_routed_experts=fr)
        ov = mod.moe_overflow_slots.get_value()
        return jnp.sum(out.astype(jnp.float32) * probe.astype(jnp.float32)), (out, ov)

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    with jax.set_mesh(mesh_fsdp_ep), nn_partitioning.axis_rules(cfg_base.logical_axis_rules):
      m_dropless_bal = muse_spark.MuseSparkMoE(cfg_base, mesh_fsdp_ep, rngs=nnx.Rngs(params=503))
      cfg_drop_bal = make_config(
          ragged_buffer_factor=1.5,
          muse_spark_moe_dropless_overflow=False,
          muse_spark_moe_num_token_chunks=2,
          **kw,
      )
      m_drop_bal = muse_spark.MuseSparkMoE(cfg_drop_bal, mesh_fsdp_ep, rngs=nnx.Rngs(params=504))
      nnx.update(m_drop_bal, nnx.state(m_dropless_bal))

      (_, (y_dl_bal, ov_dl_bal)), (g_dl_bal, dx_dl_bal) = run_moe_and_grad(m_dropless_bal, x_in, forced_bal)
      (_, (y_dr_bal, ov_dr_bal)), (g_dr_bal, dx_dr_bal) = run_moe_and_grad(m_drop_bal, x_in, forced_bal)
      self.assertEqual(int(np.sum(np.asarray(ov_dl_bal))), 0)
      self.assertEqual(int(np.sum(np.asarray(ov_dr_bal))), 0)
      np.testing.assert_array_equal(np.asarray(y_dr_bal), np.asarray(y_dl_bal))
      np.testing.assert_array_equal(np.asarray(dx_dr_bal), np.asarray(dx_dl_bal))
      np.testing.assert_array_equal(np.asarray(g_dr_bal.wi_0.get_value()), np.asarray(g_dl_bal.wi_0.get_value()))
      np.testing.assert_array_equal(np.asarray(g_dr_bal.wo.get_value()), np.asarray(g_dl_bal.wo.get_value()))

      for num_chunks in (1, 2):
        cfg_dl_sk = make_config(
            ragged_buffer_factor=0.5,
            muse_spark_moe_dropless_overflow=True,
            muse_spark_moe_num_token_chunks=num_chunks,
            **kw,
        )
        cfg_dr_sk = make_config(
            ragged_buffer_factor=0.5,
            muse_spark_moe_dropless_overflow=False,
            muse_spark_moe_num_token_chunks=num_chunks,
            **kw,
        )
        m_dl_sk = muse_spark.MuseSparkMoE(cfg_dl_sk, mesh_fsdp_ep, rngs=nnx.Rngs(params=505))
        m_dr_sk = muse_spark.MuseSparkMoE(cfg_dr_sk, mesh_fsdp_ep, rngs=nnx.Rngs(params=506))
        nnx.update(m_dl_sk, nnx.state(m_dropless_bal))
        nnx.update(m_dr_sk, nnx.state(m_dropless_bal))

        hlo_dl = run_moe_and_grad.lower(m_dl_sk, x_in, forced_skew).compile().as_text()
        hlo_dr = run_moe_and_grad.lower(m_dr_sk, x_in, forced_skew).compile().as_text()
        self.assertIn(" while(", hlo_dl)
        self.assertNotIn(" while(", hlo_dr)

        (_, (y_dl_sk, ov_dl_sk)), (g_dl_sk, dx_dl_sk) = run_moe_and_grad(m_dl_sk, x_in, forced_skew)
        (_, (y_dr_sk, ov_dr_sk)), (g_dr_sk, dx_dr_sk) = run_moe_and_grad(m_dr_sk, x_in, forced_skew)
        self.assertGreater(int(np.max(np.asarray(ov_dr_sk))), 0)
        np.testing.assert_array_equal(np.asarray(ov_dr_sk), np.asarray(ov_dl_sk))
        self.assertTrue(np.all(np.isfinite(np.asarray(y_dr_sk, np.float32))))
        self.assertTrue(np.all(np.isfinite(np.asarray(dx_dr_sk, np.float32))))
        self.assertTrue(np.all(np.isfinite(np.asarray(g_dr_sk.wi_0.get_value(), np.float32))))
        self.assertFalse(np.allclose(np.asarray(y_dr_sk, np.float32), np.asarray(y_dl_sk, np.float32), atol=1e-2))
        dx_diff = np.linalg.norm(np.asarray(dx_dr_sk, np.float32) - np.asarray(dx_dl_sk, np.float32))
        self.assertGreater(float(dx_diff / np.linalg.norm(np.asarray(dx_dl_sk, np.float32))), 0.1)
        w0_dr = np.asarray(g_dr_sk.wi_0.get_value(), np.float32)
        w0_dl = np.asarray(g_dl_sk.wi_0.get_value(), np.float32)
        self.assertGreater(float(np.linalg.norm(w0_dr - w0_dl) / np.linalg.norm(w0_dl)), 0.1)


@pytest.mark.tpu_only
class MuseSparkMoETpuGmmTest(unittest.TestCase):
  """TPU-only test validating `use_gmm_v2=True` with dropless overflow, FSDP pre-gather, and `partial_sum` chunking."""

  def test_tpu_gmm_v2_overflow_and_token_chunks_match_full_buffer(self):
    try:
      tpu_devices = jax.devices("tpu")
    except RuntimeError:
      self.skipTest("Requires >= 4 TPU devices for `use_gmm_v2=True` test.")
    if len(tpu_devices) < 4:
      self.skipTest("Requires >= 4 TPU devices for `use_gmm_v2=True` test.")
    seq = 1024
    # Test both `dim=256` (3D `gmm_v2` path) and `dim=128` (odd-128 `bfloat16` 2D fallback path),
    # across pure EP (`FSDP=1, EP=4`) and hybrid FSDP+EP (`FSDP=2, EP=2`, covering `_pregather_fsdp_rhs` + `partial_sum`).
    for dim in (256, 128):
      for fsdp, ep in ((1, 4), (2, 2)):
        kw = {
            "hardware": "tpu",
            "use_tokamax_gmm": True,
            "use_gmm_v2": True,
            "dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "base_emb_dim": dim,
            "moe_expert_input_dim": dim,
            "base_moe_mlp_dim": dim,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "ici_fsdp_parallelism": fsdp,
            "ici_expert_parallelism": ep,
            "max_target_length": seq,
        }
        rbf_small = 1.5 if ep == 4 else 0.75
        cfg_small = make_config(ragged_buffer_factor=rbf_small, muse_spark_moe_num_token_chunks=2, **kw)
        cfg_full = make_config(ragged_buffer_factor=-1.0, muse_spark_moe_num_token_chunks=1, **kw)
        mesh = Mesh(maxtext_utils.create_device_mesh(cfg_small, devices=tpu_devices[:4]), cfg_small.mesh_axes)
        x = jnp.asarray(np.random.RandomState(201 + dim + fsdp).randn(4, seq, dim), dtype=jnp.bfloat16)
        probe = jnp.asarray(np.random.RandomState(202 + dim + fsdp).randn(4, seq, dim), dtype=jnp.bfloat16)
        forced = jnp.broadcast_to(jnp.array([[[0, 1]]], dtype=jnp.int32), (4, seq, 2))

        @nnx.jit
        def run_and_grad(m, xin, pr=probe, fr=forced):
          def loss_fn(mod, xa):
            out, _ = mod(xa, forced_routed_experts=fr)
            return jnp.mean(out.astype(jnp.float32) * pr.astype(jnp.float32)), out

          return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

        with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_small.logical_axis_rules):
          m_small = muse_spark.MuseSparkMoE(cfg_small, mesh, rngs=nnx.Rngs(params=203 + dim))
          m_full = muse_spark.MuseSparkMoE(cfg_full, mesh, rngs=nnx.Rngs(params=204 + dim))
          nnx.update(m_full, nnx.state(m_small))
          if fsdp > 1:
            hlo_tpu = run_and_grad.lower(m_small, x).compile().as_text()
            _assert_no_while_body_collectives(self, hlo_tpu)
          (_, y_small), (g_small, dx_small) = run_and_grad(m_small, x)
          (_, y_full), (g_full, dx_full) = run_and_grad(m_full, x)

        np.testing.assert_allclose(np.asarray(y_small, np.float32), np.asarray(y_full, np.float32), atol=2e-2, rtol=2e-2)
        for actual_arr, expect_arr in (
            (dx_small, dx_full),
            (g_small.wi_0.get_value(), g_full.wi_0.get_value()),
            (g_small.wi_1.get_value(), g_full.wi_1.get_value()),
            (g_small.wo.get_value(), g_full.wo.get_value()),
            (g_small.post_expert_norm.scale.get_value(), g_full.post_expert_norm.scale.get_value()),
        ):
          actual_g = np.asarray(actual_arr, np.float32)
          expect_g = np.asarray(expect_arr, np.float32)
          scale_g = max(float(np.max(np.abs(expect_g))), 1e-6)
          self.assertGreater(float(np.linalg.norm(expect_g)), 0.0)
          self.assertLessEqual(float(np.linalg.norm(actual_g - expect_g) / np.linalg.norm(expect_g)), 2e-2)
          np.testing.assert_allclose(actual_g, expect_g, atol=2e-2 * scale_g, rtol=3e-2)

  def test_tpu_splash_stacked_q_heads_and_fixed_capacity_drop(self):
    """Validates `sa_num_stacked_q_heads=8` Splash parity and `muse_spark_moe_dropless_overflow=False` on TPU."""
    try:
      tpu_devices = jax.devices("tpu")
    except RuntimeError:
      self.skipTest("Requires >= 4 TPU devices.")
    if len(tpu_devices) < 4:
      self.skipTest("Requires >= 4 TPU devices.")

    # 1. Global and sliding-window `MuseSparkAttention` with `sa_num_stacked_q_heads=8` match `sa_num_stacked_q_heads=1`
    #    in forward and backward on TPU Splash (`16Q / 2KV = 8` stacked heads per KV group, `head_dim=64`, `SEQ_MINOR`).
    seq_attn, dim_attn = 512, 256
    attn_kw = {
        "hardware": "tpu",
        "dtype": "bfloat16",
        "weight_dtype": "float32",
        "base_emb_dim": dim_attn,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 2,
        "head_dim": 64,
        "sliding_window_size": 256,
        "max_target_length": seq_attn,
        "attention": "flash",
        "use_tokamax_splash": True,
        "sa_q_layout": "SEQ_MINOR",
        "sa_k_layout": "SEQ_MINOR",
        "sa_v_layout": "SEQ_MINOR",
        "local_sa_q_layout": "SEQ_MINOR",
        "local_sa_k_layout": "SEQ_MINOR",
        "local_sa_v_layout": "SEQ_MINOR",
        "sa_use_fused_bwd_kernel": True,
        "local_sa_use_fused_bwd_kernel": True,
        "ici_fsdp_parallelism": 4,
        "ici_expert_parallelism": 1,
    }
    cfg_st1 = make_config(sa_num_stacked_q_heads=1, **attn_kw)
    cfg_st8 = make_config(sa_num_stacked_q_heads=8, **attn_kw)
    mesh_attn = Mesh(maxtext_utils.create_device_mesh(cfg_st1, devices=tpu_devices[:4]), cfg_st1.mesh_axes)
    x_attn = jnp.asarray(np.random.RandomState(601).randn(4, seq_attn, dim_attn), dtype=jnp.bfloat16)
    pr_attn = jnp.asarray(np.random.RandomState(602).randn(4, seq_attn, dim_attn), dtype=jnp.bfloat16)
    pos_attn = jnp.broadcast_to(jnp.arange(seq_attn, dtype=jnp.int32)[None, :], (4, seq_attn))
    seg_attn = jnp.ones((4, seq_attn), dtype=jnp.int32)

    @nnx.jit
    def run_attn_and_grad(m, xin):
      def loss_fn(mod, xa):
        y, _ = mod(xa, xa, pos_attn, seg_attn, deterministic=True, model_mode=MODEL_MODE_TRAIN)
        return jnp.mean(y.astype(jnp.float32) * pr_attn.astype(jnp.float32)), y

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    for is_full in (True, False):
      with jax.set_mesh(mesh_attn), nn_partitioning.axis_rules(cfg_st1.logical_axis_rules):
        a1 = muse_spark.MuseSparkDecoderLayer(
            cfg_st1,
            mesh_attn,
            MODEL_MODE_TRAIN,
            layer_idx=1 if is_full else 0,
            is_full_attention_override=is_full,
            is_moe_override=False,
            rngs=nnx.Rngs(params=603),
        ).self_attention
        a8 = muse_spark.MuseSparkDecoderLayer(
            cfg_st8,
            mesh_attn,
            MODEL_MODE_TRAIN,
            layer_idx=1 if is_full else 0,
            is_full_attention_override=is_full,
            is_moe_override=False,
            rngs=nnx.Rngs(params=604),
        ).self_attention
        nnx.update(a8, nnx.state(a1))
        self.assertEqual(
            a8.attention_op._create_sa_config(  # pylint: disable=protected-access
                a8.attention_op._splash_blocks(), seq_attn, seq_attn, None  # pylint: disable=protected-access
            ).num_stacked_q_heads,
            8,
        )
        (_, y1), (g1, dx1) = run_attn_and_grad(a1, x_attn)
        (_, y8), (g8, dx8) = run_attn_and_grad(a8, x_attn)
      np.testing.assert_allclose(np.asarray(y8, np.float32), np.asarray(y1, np.float32), atol=1e-2, rtol=1e-2)
      np.testing.assert_allclose(np.asarray(dx8, np.float32), np.asarray(dx1, np.float32), atol=1.5e-2, rtol=2e-2)
      np.testing.assert_allclose(
          np.asarray(g8.query.kernel.get_value(), np.float32),
          np.asarray(g1.query.kernel.get_value(), np.float32),
          atol=1.5e-2,
          rtol=2e-2,
      )

    # 2. TPU `use_gmm_v2=True` with `muse_spark_moe_dropless_overflow=False` compiles with zero `while` loops.
    moe_kw = {
        "hardware": "tpu",
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "base_emb_dim": 256,
        "moe_expert_input_dim": 256,
        "base_moe_mlp_dim": 256,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "ici_fsdp_parallelism": 2,
        "ici_expert_parallelism": 2,
        "max_target_length": 1024,
        "ragged_buffer_factor": 0.75,
        "muse_spark_moe_dropless_overflow": False,
        "muse_spark_moe_num_token_chunks": 2,
    }
    cfg_drop_tpu = make_config(**moe_kw)
    mesh_moe = Mesh(maxtext_utils.create_device_mesh(cfg_drop_tpu, devices=tpu_devices[:4]), cfg_drop_tpu.mesh_axes)
    x_moe = jnp.asarray(np.random.RandomState(610).randn(4, 1024, 256), dtype=jnp.bfloat16)
    pr_moe = jnp.asarray(np.random.RandomState(611).randn(4, 1024, 256), dtype=jnp.bfloat16)
    forced_moe = jnp.broadcast_to(jnp.array([[[0, 1]]], dtype=jnp.int32), (4, 1024, 2))

    @nnx.jit
    def run_tpu_drop(m, xin):
      def loss_fn(mod, xa):
        out, _ = mod(xa, forced_routed_experts=forced_moe)
        return jnp.mean(out.astype(jnp.float32) * pr_moe.astype(jnp.float32)), out

      return nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, xin)

    with jax.set_mesh(mesh_moe), nn_partitioning.axis_rules(cfg_drop_tpu.logical_axis_rules):
      m_drop_tpu = muse_spark.MuseSparkMoE(cfg_drop_tpu, mesh_moe, rngs=nnx.Rngs(params=612))
      hlo_drop_tpu = run_tpu_drop.lower(m_drop_tpu, x_moe).compile().as_text()
      self.assertNotIn(" while(", hlo_drop_tpu)
      (_, y_drop_tpu), (_, dx_drop_tpu) = run_tpu_drop(m_drop_tpu, x_moe)
    self.assertTrue(np.all(np.isfinite(np.asarray(y_drop_tpu, np.float32))))
    self.assertTrue(np.all(np.isfinite(np.asarray(dx_drop_tpu, np.float32))))


if __name__ == "__main__":
  absltest.main()
