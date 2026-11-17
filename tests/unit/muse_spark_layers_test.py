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
from maxtext.models import muse_spark
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

pytestmark = [pytest.mark.cpu_only]

BASE_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

EMB, Q_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 16
MLP, MOE_IN, MOE_MLP, EXPERTS, TOPK = 64, 128, 128, 4, 2
SLIDING_WINDOW = 4
VOCAB = 64


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

    # Expert parallelism (`expert` mesh axis > 1) must raise a clear ValueError.
    if jax.device_count() >= 2:
      ep_mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(2), ("expert",))
      moe_ep = muse_spark.MuseSparkMoE(self.config, ep_mesh, rngs=nnx.Rngs(params=23))
      with self.assertRaisesRegex(ValueError, "expert parallelism"):
        moe_ep(x)

    # `moe_tc_ragged_weights_on_activation=True` must raise ValueError because `post_expert_norm`
    # normalizes each expert output before weighting by `top_k_weights`.
    cfg_bad_moe = make_config(moe_tc_ragged_weights_on_activation=True)
    with self.assertRaisesRegex(ValueError, "moe_tc_ragged_weights_on_activation=False"):
      muse_spark.MuseSparkMoE(cfg_bad_moe, self.mesh, rngs=nnx.Rngs(params=25))

    # Compare single-device reference against multi-device FSDP+TP on divisible and indivisible batches.
    single_mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * len(self.config.mesh_axes)), self.config.mesh_axes)
    ref_moe = muse_spark.MuseSparkMoE(self.config, single_mesh, rngs=nnx.Rngs(params=24))
    ref_flat = dict(nnx.to_flat_state(nnx.state(ref_moe)))

    mesh_overrides = [{}]
    if jax.device_count() >= 2 and jax.device_count() % 2 == 0:
      mesh_overrides.append({"ici_fsdp_parallelism": jax.device_count() // 2, "ici_tensor_parallelism": 2})

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


if __name__ == "__main__":
  absltest.main()
