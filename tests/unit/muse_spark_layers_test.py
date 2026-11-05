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

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_shape, param_mapping
from maxtext.checkpoint_conversion.utils.utils import process_maxtext_param
from maxtext.common.common_types import (
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.models import muse_spark
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

pytestmark = [pytest.mark.cpu_only]

BASE_CONFIG = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

EMB, Q_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 16
MLP, MOE_IN, MOE_MLP, EXPERTS, TOPK = 64, 16, 24, 4, 2
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

  num_prefix, num_blocks, rem_start = muse_spark.scan_layout(cfg_s)
  cycle = cfg_s.inhomogeneous_layer_cycle_interval

  for i in range(num_prefix):
    nnx.update(getattr(scanned_model.decoder, f"layers_{i}"), nnx.state(getattr(unscanned_model.decoder, f"layers_{i}")))
  for i in range(rem_start, cfg_s.num_decoder_layers):
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


def make_hf_text_config(num_layers):
  """Builds a tiny HF `text_config` dict matching `make_config`, for Torch-free checkpoint conversion tests."""
  return {
      "text_config": {
          "hidden_size": EMB,
          "intermediate_size": MLP,
          "moe_hidden_size": MOE_IN,
          "moe_intermediate_size": MOE_MLP,
          "num_hidden_layers": num_layers,
          "num_attention_heads": Q_HEADS,
          "num_key_value_heads": KV_HEADS,
          "head_dim": HEAD_DIM,
          "num_local_experts": EXPERTS,
          "num_experts_per_tok": TOPK,
          "vocab_size": VOCAB,
          "mlp_layer_types": ["sparse"] * num_layers,
      }
  }


def convert_hf_state_to_maxtext(hf_state, hf_cfg_dict, cfg_m, scan):
  """Converts a {hf_key: np.ndarray} dict into the MaxText checkpoint tree through the `to_maxtext` code path."""
  abstract_dict, treedef = to_maxtext.get_maxtext_model_info(cfg_m)
  pmap = param_mapping.MUSE_SPARK_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, cfg_m, scan_layers=scan)
  hmap = param_mapping.MUSE_SPARK_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, cfg_m, scan_layers=scan, saving_to_hf=False)
  final_mt_weights = [None] * len(abstract_dict)
  for mt_key, hf_targets in pmap.items():
    idx_or_indices, shape_or_shapes = to_maxtext._get_maxtext_indices_and_shapes(  # pylint: disable=protected-access
        mt_key, abstract_dict
    )
    load_fn = to_maxtext._get_hf_loading_function(  # pylint: disable=protected-access
        hf_targets, lambda k: hf_state[k], hmap.get(mt_key), shape_or_shapes, cfg_m, mt_key
    )
    to_maxtext._get_maxtext_weight(  # pylint: disable=protected-access
        load_fn, idx_or_indices, shape_or_shapes, mt_key, final_mt_weights, "float32", use_lazy_load=False
    )
  return jax.tree_util.tree_unflatten(treedef, final_mt_weights), pmap, abstract_dict


def load_converted_tree(model, converted_tree):
  """Loads a converted MaxText checkpoint tree (params + MoEBiasVar) into an NNX model."""
  new_flat = []
  for path, var in nnx.to_flat_state(nnx.state(model, nnx.Any(nnx.Param, moe.MoEBiasVar))):
    node = converted_tree["MoEBiasVar" if isinstance(var, moe.MoEBiasVar) else "params"]
    for p in to_maxtext.param_key_parts_from_path(path):
      node = node[p]
    assert node.shape == var.get_value().shape, f"Shape mismatch for {path}: {node.shape} vs {var.get_value().shape}"
    new_flat.append((path, var.replace(jnp.asarray(node))))
  nnx.update(model, nnx.from_flat_state(new_flat))


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
    self.assertEqual(muse_spark.scan_layout(cfg62), (2, 15, 62))
    full_indices = [i for i in range(62) if muse_spark.is_full_attention_layer(i, cfg62)]
    self.assertEqual(full_indices, list(range(1, 62, 4)))
    self.assertTrue(all(muse_spark.is_moe_layer(i, cfg62) for i in range(62)))

    # Dense prefix forces extra whole cycle(s) into unrolled prefix so scanned cycles stay homogeneous MoE.
    cfg_dense_prefix = stub(62, first_dense=3)
    self.assertEqual(muse_spark.scan_layout(cfg_dense_prefix), (6, 14, 62))
    self.assertEqual([muse_spark.is_moe_layer(i, cfg_dense_prefix) for i in range(5)], [False, False, False, True, True])
    self.assertEqual(muse_spark.scan_layout(stub(6, experts=1)), (6, 0, 6))

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

    # Expert parallelism > 1 must raise a clear ValueError.
    if jax.device_count() >= 2:
      ep_mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(2), ("expert",))
      moe_ep = muse_spark.MuseSparkMoE(self.config, ep_mesh, rngs=nnx.Rngs(params=23))
      with self.assertRaisesRegex(ValueError, "expert parallelism"):
        moe_ep(x)

    # Compare single-device reference against multi-device FSDP / FSDP+TP on divisible and indivisible batches.
    single_mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * len(self.config.mesh_axes)), self.config.mesh_axes)
    ref_moe = muse_spark.MuseSparkMoE(self.config, single_mesh, rngs=nnx.Rngs(params=24))
    ref_state = nnx.state(ref_moe)

    mesh_overrides = [{}]
    if jax.device_count() >= 2 and jax.device_count() % 2 == 0:
      mesh_overrides.append({"ici_fsdp_parallelism": jax.device_count() // 2, "ici_tensor_parallelism": 2})

    for batch in (1, 3, max(2, jax.device_count())):
      xb = jnp.asarray(np.random.RandomState(30 + batch).randn(batch, 5, EMB), dtype=jnp.float32)
      probe = jnp.asarray(np.random.RandomState(40 + batch).randn(batch, 5, EMB), dtype=jnp.float32)

      def loss_fn(m, xin, pr=probe):
        out, _ = m(xin)
        return jnp.sum(out * pr), out

      with jax.set_mesh(single_mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
        (_, ref_y), (ref_grads, ref_dx) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(ref_moe, xb)

      for mesh_kw in mesh_overrides:
        cfg_m = make_config(**mesh_kw)
        mesh_m = Mesh(maxtext_utils.create_device_mesh(cfg_m), cfg_m.mesh_axes)
        moe_m = muse_spark.MuseSparkMoE(cfg_m, mesh_m, rngs=nnx.Rngs(params=99))
        nnx.update(moe_m, ref_state)
        with jax.set_mesh(mesh_m), nn_partitioning.axis_rules(cfg_m.logical_axis_rules):
          (_, y_m), (grads_m, dx_m) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(moe_m, xb)
        np.testing.assert_allclose(y_m, ref_y, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(dx_m, ref_dx, atol=1e-5, rtol=1e-5)
        for p_ref, p_m in zip(nnx.to_flat_state(ref_grads), nnx.to_flat_state(grads_m)):
          np.testing.assert_allclose(
              p_m[1].get_value(), p_ref[1].get_value(), atol=1e-4, rtol=1e-4, err_msg=str(p_ref[0])
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
    self.assertEqual(muse_spark.scan_layout(cfg_s), (2, 1, 6))

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

    def model_loss(m):
      logits = m(decoder_input_tokens=ids, decoder_positions=pos, decoder_segment_ids=seg, enable_dropout=False)
      return jnp.sum(logits * probe), logits

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      (_, logits_u), grads_u = nnx.value_and_grad(model_loss, has_aux=True)(mu)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      (_, logits_s), grads_s = nnx.value_and_grad(model_loss, has_aux=True)(ms)

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

  def test_hf_checkpoint_conversion_round_trip(self):
    """HF -> MaxText (to_maxtext) -> HF (to_huggingface) round trip on random HF weights, unscanned and scanned."""
    num_layers = 6
    hf_cfg_dict = make_hf_text_config(num_layers)
    shape_map = hf_shape.MUSE_SPARK_HF_WEIGHTS_TO_SHAPE(hf_cfg_dict)
    self.assertIn("model.language_model.layers.0.mlp.gate.e_score_correction_bias", shape_map)
    self.assertNotIn("model.language_model.layers.0.mlp.gate_proj.weight", shape_map)
    rng = np.random.RandomState(70)
    hf_state = {k: (0.1 * rng.randn(*shp)).astype(np.float32) for k, shp in shape_map.items()}

    batch, seq = jax.device_count(), 6
    ids = jnp.asarray(np.random.RandomState(71).randint(1, VOCAB, (batch, seq)), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))
    seg = jnp.ones((batch, seq), dtype=jnp.int32)

    logits = {}
    for scan in (False, True):
      cfg_m = make_config(base_num_decoder_layers=num_layers, scan_layers=scan)
      converted_tree, pmap, abstract_dict = convert_hf_state_to_maxtext(hf_state, hf_cfg_dict, cfg_m, scan)

      # Every MaxText variable is mapped, and every HF weight is consumed.
      flat_mt_keys, used_hf_keys = set(), set()
      for mt_key, hf_targets in pmap.items():
        flat_mt_keys.update(mt_key if isinstance(mt_key, tuple) else (mt_key,))
        used_hf_keys.update(np.asarray(hf_targets, dtype=object).ravel().tolist())
      self.assertEqual(flat_mt_keys, set(abstract_dict), f"scan={scan}")
      self.assertEqual(used_hf_keys, set(shape_map), f"scan={scan}")

      model = model_creation_utils.from_config(cfg_m, mesh=self.mesh, rngs=nnx.Rngs(params=72, dropout=73))
      load_converted_tree(model, converted_tree)
      with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_m.logical_axis_rules):
        logits[scan] = model(
            decoder_input_tokens=ids, decoder_positions=pos, decoder_segment_ids=seg, enable_dropout=False
        )

      # MaxText -> HF, mirroring `to_huggingface`: promote composite hook keys (wi_0, wi_1) into the param map.
      save_pmap = param_mapping.MUSE_SPARK_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, cfg_m, scan_layers=scan)
      save_hooks = param_mapping.MUSE_SPARK_MAXTEXT_TO_HF_PARAM_HOOK_FN(
          hf_cfg_dict, cfg_m, scan_layers=scan, saving_to_hf=True
      )
      for hook_key in [k for k in save_hooks if isinstance(k, tuple)]:
        if save_pmap.get(hook_key[0]) is not None:
          save_pmap[hook_key] = save_pmap[hook_key[0]]
          for k in hook_key:
            save_pmap.pop(k, None)
      flat_mt = {
          "-".join(to_maxtext.param_key_parts_from_path(p)): np.asarray(v)
          for p, v in jax.tree_util.tree_flatten_with_path(converted_tree)[0]
      }
      hf_out = {}
      for mt_key in save_pmap:
        weight = [flat_mt[k] for k in mt_key] if isinstance(mt_key, tuple) else flat_mt[mt_key]
        hf_out.update(process_maxtext_param(mt_key, weight, save_pmap, save_hooks, shape_map, cfg_m))
      self.assertEqual(set(hf_out), set(hf_state), f"scan={scan}")
      for k, v in hf_state.items():
        np.testing.assert_array_equal(np.asarray(hf_out[k], dtype=np.float32), v, err_msg=f"scan={scan}: {k}")

    self.assertTrue(np.all(np.isfinite(np.asarray(logits[False]))))
    np.testing.assert_allclose(logits[True], logits[False], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
