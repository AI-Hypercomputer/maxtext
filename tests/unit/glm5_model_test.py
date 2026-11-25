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

"""Targeted CPU unit tests for GLM-5, MLA IndexShare, MoE routing, NNXDecoder, and trainer bias updates."""

import copy
import tempfile
import types
import unittest
from unittest import mock

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common import common_types
from maxtext.configs import pyconfig
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.models import deepseek
from maxtext.models import glm5
from maxtext.models import models
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import maxtext_utils
import numpy as np
from tests.utils.test_helpers import get_test_config_path  # pylint: disable=no-name-in-module


def _tiny_glm5_args(
    tmpdir: str,
    *,
    model_name: str = "glm5.2-744b",
    scan_layers: bool = True,
    prune_shared_indexers: bool | None = None,
    num_layers: int = 2,
    first_dense: int = 1,
    pattern: str = "FS",
    seq_len: int = 6,
    max_prefill_len: int = 4,
    indexer_topk: int = 4,
    indexer_loss_scaling_factor: float = 0.0,
    indexer_use_approx_top_k: bool = False,
    mlp_activations_limit: float = 0.0,
    sparse_matmul: bool = False,
) -> list[str]:
  """Builds an ultra-small, multi-device-safe config argument list for GLM-5 unit tests."""
  args = [
      "",
      get_test_config_path(),
      f"model_name={model_name}",
      "override_model_config=True",
      "skip_jax_distributed_system=True",
      f"base_output_directory={tmpdir}",
      "run_name=glm5_model_unit",
      "enable_checkpointing=False",
      "remat_policy=none",
      f"scan_layers={scan_layers}",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "matmul_precision=highest",
      "attention=dot_product",
      "base_emb_dim=16",
      "base_mlp_dim=16",
      "base_moe_mlp_dim=8",
      f"base_num_decoder_layers={num_layers}",
      f"first_num_dense_layers={first_dense}",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "q_lora_rank=8",
      "kv_lora_rank=4",
      "qk_nope_head_dim=4",
      "qk_rope_head_dim=4",
      "v_head_dim=4",
      "indexer_n_heads=2",
      "indexer_head_dim=4",
      f"indexer_topk={indexer_topk}",
      f"indexer_loss_scaling_factor={indexer_loss_scaling_factor}",
      f"indexer_use_approx_top_k={indexer_use_approx_top_k}",
      f"mlp_activations_limit={mlp_activations_limit}",
      f"sparse_matmul={sparse_matmul}",
      "megablox=False",
      "num_experts=4",
      "num_experts_per_tok=2",
      "shared_experts=1",
      "n_routing_groups=1",
      "topk_routing_group=1",
      "vocab_size=32",
      "max_position_embeddings=64",
      "original_max_position_embeddings=64",
      f"max_prefill_predict_length={max_prefill_len}",
      f"max_target_length={seq_len}",
      "mla_naive_kvcache=False",
      "per_device_batch_size=1",
  ]
  if model_name == "glm5.2-744b":
    args.append(f"index_share_pattern={pattern}")
  if prune_shared_indexers is not None:
    args.append(f"prune_shared_indexers={prune_shared_indexers}")
  return args


def _make_mesh(cfg) -> Mesh:
  return Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)


def _make_mla(
    cfg, mesh: Mesh, *, is_shared_layer: bool = False, served_group_size: int = 1, model_mode: str | None = None
):
  mode = model_mode or common_types.MODEL_MODE_TRAIN
  batch = cfg.global_batch_size_to_train_on
  return attention_mla.MLA(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      mesh=mesh,
      attention_kernel="dot_product",
      inputs_q_shape=(batch, cfg.max_target_length, cfg.emb_dim),
      inputs_kv_shape=(batch, cfg.max_target_length, cfg.emb_dim),
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      q_lora_rank=cfg.q_lora_rank,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      v_head_dim=cfg.v_head_dim,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      mscale=cfg.mscale,
      rope_factor=cfg.rope_factor,
      model_mode=mode,
      is_shared_layer=is_shared_layer,
      served_group_size=served_group_size,
      rngs=nnx.Rngs(params=0),
  )


class _DummyGateHolder(nnx.Module):
  """Minimal module holding a GateLogit with MoEBiasVar for trainer path tests."""

  def __init__(self, num_experts: int = 4):
    self.gate = moe.GateLogit(
        in_features_shape=4,
        out_features_shape=num_experts,
        model_name="glm5.2-744b",
        decoder_block=common_types.DecoderBlockType.GLM5,
        mesh=None,
        rngs=nnx.Rngs(0),
        use_bias=True,
    )


class _DummyNoGateHolder(nnx.Module):
  """Minimal module without any GateLogit bias."""

  def __init__(self):
    self.scale = nnx.Param(jnp.ones((4,), dtype=jnp.float32))


class _DummyBiasRoot(nnx.Module):
  """Root module mixing attribute, dict, and bias-free children for _apply_moe_bias_updates_by_path."""

  def __init__(self, num_experts: int = 4):
    self.attr_layer = _DummyGateHolder(num_experts)
    self.dict_layers = nnx.data({"stage_0": _DummyGateHolder(num_experts)})
    self.no_gate = _DummyNoGateHolder()


class _NonTupleScanLayer(nnx.Module):
  """Minimal scannable layer returning a plain Array instead of a tuple."""

  def __init__(self, config, mesh=None, model_mode=common_types.MODEL_MODE_TRAIN, quant=None, rngs=None, **kwargs):
    del mesh, model_mode, quant, kwargs
    self.w = nnx.Param(jnp.ones((config.emb_dim,), dtype=jnp.float32))
    self.rngs = rngs

  def __call__(self, inputs, *args, **kwargs):
    del args, kwargs
    return inputs + self.w.get_value()


class Glm5ModelLayersCoverageTest(unittest.TestCase):
  """Covers GLM-5 model, MLA IndexShare, MoE routing, NNXDecoder, and trainer bias update paths."""

  # pylint: disable=protected-access

  def test_decode_and_apply_moe_bias_updates_branches(self):
    """Covers _decode_moe_bias_update and _apply_moe_bias_updates_by_path branches and error checks."""
    cfg_std = types.SimpleNamespace(
        using_pipeline_parallelism=False,
        routed_bias_update_rate=0.05,
        log_moe_bias_norms=False,
    )
    cfg_pp = types.SimpleNamespace(
        using_pipeline_parallelism=True,
        routed_bias_update_rate=0.05,
        log_moe_bias_norms=True,
    )

    # 1. Float update without pipeline parallelism returns raw update unchanged.
    raw_float = jnp.array([0.1, -0.1, 0.0, 0.2], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(pre_train._decode_moe_bias_update(raw_float, cfg_std)), np.asarray(raw_float))

    # 2. Integer counts tuple without pipeline parallelism decodes via sign(mean - counts) * rate.
    int_counts = (jnp.array([4, 0, 2, 2], dtype=jnp.int32),)
    decoded_int = np.asarray(pre_train._decode_moe_bias_update(int_counts, cfg_std))
    np.testing.assert_allclose(decoded_int, np.array([-0.05, 0.05, 0.0, 0.0], dtype=np.float32), atol=1e-7)

    # 3. Pipeline parallelism decodes even float counts via sign(mean - counts) * rate.
    float_counts = jnp.array([1.0, 3.0, 2.0, 2.0], dtype=jnp.float32)
    decoded_pp = np.asarray(pre_train._decode_moe_bias_update(float_counts, cfg_pp))
    np.testing.assert_allclose(decoded_pp, np.array([0.05, -0.05, 0.0, 0.0], dtype=np.float32), atol=1e-7)

    # 4. Default None optional arguments, dict traversal, missing path key, and non-gate target.
    root = _DummyBiasRoot(num_experts=4)
    upd_a = jnp.array([0.02, -0.02, 0.02, -0.02], dtype=jnp.float32)
    upd_b = jnp.array([-0.01, 0.01, -0.01, 0.01], dtype=jnp.float32)
    intermediates = {
        "unrelated_metric": jnp.array(1.0),
        "attr_layer": {"moe_bias_updates": (upd_a,)},
        "dict_layers": {
            "stage_0": {"moe_bias_updates": upd_b},
            "missing_stage": {"moe_bias_updates": upd_b},
        },
        "missing_top": {"moe_bias_updates": upd_a},
        "no_gate": {"moe_bias_updates": upd_a},
    }
    pre_train._apply_moe_bias_updates_by_path(root, intermediates, cfg_std)
    np.testing.assert_allclose(np.asarray(root.attr_layer.gate.bias.get_value()), np.asarray(upd_a), atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(root.dict_layers["stage_0"].gate.bias.get_value()), np.asarray(upd_b), atol=1e-7
    )

    # 5. Explicit diag lists + bias_metrics with log_moe_bias_norms=False and log_moe_bias_norms=True.
    diag_vals, diag_upds, metrics_off = [], [], {}
    pre_train._apply_moe_bias_updates_by_path(
        root,
        {"attr_layer": {"moe_bias_updates": upd_a}},
        cfg_std,
        diag_bias_values=diag_vals,
        diag_bias_updates=diag_upds,
        bias_metrics=metrics_off,
    )
    self.assertEqual(len(diag_vals), 1)
    self.assertEqual(len(diag_upds), 1)
    self.assertEqual(metrics_off, {})

    metrics_on = {}
    pre_train._apply_moe_bias_updates_by_path(
        root,
        {"dict_layers": {"stage_0": {"moe_bias_updates": float_counts}}},
        cfg_pp,
        diag_bias_values=diag_vals,
        diag_bias_updates=diag_upds,
        bias_metrics=metrics_on,
    )
    self.assertIn("learning/moe_bias_before_norm_decoder-dict_layers-stage_0", metrics_on)
    self.assertIn("learning/moe_bias_update_norm_decoder-dict_layers-stage_0", metrics_on)

    # 6. Shape mismatch raises ValueError.
    with self.assertRaisesRegex(ValueError, "moe_bias_updates at attr_layer/moe_bias_updates has shape"):
      pre_train._apply_moe_bias_updates_by_path(
          root,
          {"attr_layer": {"moe_bias_updates": jnp.ones((7,), dtype=jnp.float32)}},
          cfg_std,
      )

  def test_glm5_moe_routing_gate_logit_and_ffn_clipping(self):
    """Covers _uses_deepseek_routing, GateLogit forward branches, and RoutedMoE GLM5 FFN clipping/matmul."""
    self.assertTrue(moe._uses_deepseek_routing("deepseek3-671b", None))
    self.assertTrue(moe._uses_deepseek_routing("custom", common_types.DecoderBlockType.GLM5))
    self.assertTrue(moe._uses_deepseek_routing("custom", "glm5"))
    self.assertFalse(moe._uses_deepseek_routing("glm5.3-flash", common_types.DecoderBlockType.LLAMA2))

    x_gate = jnp.ones((2, 3, 8), dtype=jnp.float32)
    gate_glm = moe.GateLogit(
        in_features_shape=8,
        out_features_shape=4,
        model_name="glm5.2-744b",
        decoder_block=common_types.DecoderBlockType.GLM5,
        mesh=None,
        rngs=nnx.Rngs(0),
        use_bias=True,
        score_func="sigmoid",
    )
    logits_glm, pre_bias_glm = gate_glm(x_gate)
    self.assertEqual(logits_glm.shape, (2, 3, 4))
    self.assertIsNotNone(pre_bias_glm)

    gate_other = moe.GateLogit(
        in_features_shape=8,
        out_features_shape=4,
        model_name="llama2-7b",
        decoder_block=common_types.DecoderBlockType.LLAMA2,
        mesh=None,
        rngs=nnx.Rngs(1),
        use_bias=True,
    )
    logits_other, pre_bias_other = gate_other(x_gate)
    self.assertEqual(logits_other.shape, (2, 3, 4))
    self.assertIsNone(pre_bias_other)

    with tempfile.TemporaryDirectory() as tmpdir:
      # MaxTextConfig rejects routed_bias_update_rate > 0 on non-DeepSeek/GLM decoder blocks.
      with self.assertRaisesRegex(
          ValueError, "Loss-free load balancing is only supported for the DeepSeek/GLM decoder block."
      ):
        pyconfig.initialize(
            [
                "",
                get_test_config_path(),
                "model_name=mixtral-8x7b",
                "override_model_config=True",
                "skip_jax_distributed_system=True",
                f"base_output_directory={tmpdir}",
                "run_name=mixtral_bad_bias",
                "enable_checkpointing=False",
                "routed_bias=True",
                "routed_bias_update_rate=0.001",
            ]
        )

      # Dense matmul with mlp_activations_limit > 0 (hits GLM5 SwiGLU clipping) and sparse_matmul=True.
      dense_init = initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")
      for sparse_flag, clip_limit in ((False, 2.5), (True, 0.0)):
        cfg = pyconfig.initialize(
            _tiny_glm5_args(
                tmpdir,
                scan_layers=False,
                prune_shared_indexers=True,
                mlp_activations_limit=clip_limit,
                sparse_matmul=sparse_flag,
            )
        )
        mesh = _make_mesh(cfg)
        batch = cfg.global_batch_size_to_train_on
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
          routed_moe = moe.RoutedMoE(
              config=cfg,
              num_experts=cfg.num_experts,
              num_experts_per_tok=cfg.num_experts_per_tok,
              mesh=mesh,
              kernel_init=dense_init,
              kernel_axes=("embed", None),
              intermediate_dim=cfg.moe_mlp_dim,
              dtype=cfg.dtype,
              weight_dtype=cfg.weight_dtype,
              quant=quantizations.configure_quantization(cfg),
              rngs=nnx.Rngs(params=3, dropout=4),
          )
          inp = jnp.ones((batch, 4, cfg.emb_dim), dtype=jnp.float32) * 3.0
          out, lb_loss, bias_upd = routed_moe(inp)
          self.assertEqual(out.shape, (batch, 4, cfg.emb_dim))
          self.assertTrue(np.all(np.isfinite(np.asarray(out))))
          del lb_loss, bias_upd

      # Non-DeepSeek/GLM RoutedMoE (Mixtral) covers the False branches of _uses_deepseek_routing in dense & sparse paths.
      for sparse_flag in (False, True):
        cfg_mix = pyconfig.initialize(
            [
                "",
                get_test_config_path(),
                "model_name=mixtral-8x7b",
                "override_model_config=True",
                "skip_jax_distributed_system=True",
                f"base_output_directory={tmpdir}",
                "run_name=mixtral_moe_unit",
                "enable_checkpointing=False",
                "dtype=float32",
                "weight_dtype=float32",
                f"sparse_matmul={sparse_flag}",
                "megablox=False",
                "base_emb_dim=16",
                "base_mlp_dim=16",
                "base_moe_mlp_dim=16",
                "base_num_decoder_layers=2",
                "base_num_query_heads=2",
                "base_num_kv_heads=2",
                "num_experts=4",
                "num_experts_per_tok=2",
                "vocab_size=32",
                "max_target_length=4",
                "per_device_batch_size=1",
            ]
        )
        mesh_mix = _make_mesh(cfg_mix)
        batch_mix = cfg_mix.global_batch_size_to_train_on
        with jax.set_mesh(mesh_mix), nn_partitioning.axis_rules(cfg_mix.logical_axis_rules):
          mix_moe = moe.RoutedMoE(
              config=cfg_mix,
              num_experts=cfg_mix.num_experts,
              num_experts_per_tok=cfg_mix.num_experts_per_tok,
              mesh=mesh_mix,
              kernel_init=dense_init,
              kernel_axes=("embed", None),
              intermediate_dim=cfg_mix.moe_mlp_dim,
              dtype=cfg_mix.dtype,
              weight_dtype=cfg_mix.weight_dtype,
              quant=quantizations.configure_quantization(cfg_mix),
              rngs=nnx.Rngs(params=5, dropout=6),
          )
          out_mix, _, _ = mix_moe(jnp.ones((batch_mix, 4, cfg_mix.emb_dim), dtype=jnp.float32))
          self.assertEqual(out_mix.shape, (batch_mix, 4, cfg_mix.emb_dim))

  def test_indexer_short_sequence_index_share_branches(self):
    """Covers Indexer.__call__ short-sequence branches (k <= indexer_topk) under use_index_share=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_no_loss = pyconfig.initialize(
          _tiny_glm5_args(
              tmpdir,
              scan_layers=False,
              prune_shared_indexers=True,
              seq_len=6,
              max_prefill_len=4,
              indexer_topk=4,
              indexer_loss_scaling_factor=0.0,
          )
      )
      cfg_with_loss = pyconfig.initialize(
          _tiny_glm5_args(
              tmpdir,
              scan_layers=False,
              prune_shared_indexers=True,
              seq_len=6,
              max_prefill_len=4,
              indexer_topk=4,
              indexer_loss_scaling_factor=0.4,
              indexer_use_approx_top_k=True,
          )
      )
      mesh = _make_mesh(cfg_no_loss)
      batch = cfg_no_loss.global_batch_size_to_train_on

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_no_loss.logical_axis_rules):
        mla_ar = _make_mla(cfg_no_loss, mesh, model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE)
        mla_loss = _make_mla(cfg_with_loss, mesh, model_mode=common_types.MODEL_MODE_TRAIN)
        assert mla_ar.indexer is not None and mla_loss.indexer is not None

        short_len = 3  # <= indexer_topk=4
        x_q = jnp.ones((batch, short_len, cfg_no_loss.emb_dim), dtype=jnp.float32)
        q_low = jnp.ones((batch, short_len, cfg_no_loss.q_lora_rank), dtype=jnp.float32)
        pos = jnp.broadcast_to(jnp.arange(short_len, dtype=jnp.int32), (batch, short_len))
        seg = jnp.ones((batch, short_len), dtype=jnp.int32)

        # 1. Train mode, loss=0, attention_mask=None -> returns (None, all_indices, None).
        mask_none, idx_all, score_none = mla_ar.indexer(
            inputs_q=x_q,
            low_rank_q=q_low,
            inputs_kv=x_q,
            inputs_positions=pos,
            attention_mask=None,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        self.assertIsNone(mask_none)
        self.assertIsNone(score_none)
        self.assertEqual(idx_all.shape, (batch, short_len, short_len))

        # 2. Train mode, loss=0, attention_mask!=None, cached_s=None -> adds attention_mask only.
        attn_mask = jnp.zeros((batch, short_len, short_len), dtype=jnp.float32)
        mask_attn, idx_attn, _ = mla_ar.indexer(
            inputs_q=x_q,
            low_rank_q=q_low,
            inputs_kv=x_q,
            inputs_positions=pos,
            attention_mask=attn_mask,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        self.assertEqual(mask_attn.shape, (batch, short_len, short_len))
        self.assertEqual(idx_attn.shape, (batch, short_len, short_len))

        # 3. Prefill mode (kv_len=4 <= indexer_topk=4), attention_mask=None, kv_cache!=None -> cached_s!=None!
        x_pref = jnp.ones((batch, 4, cfg_no_loss.emb_dim), dtype=jnp.float32)
        q_pref = jnp.ones((batch, 4, cfg_no_loss.q_lora_rank), dtype=jnp.float32)
        pos_pref = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch, 4))
        seg_pref = jnp.ones((batch, 4), dtype=jnp.int32)
        mask_cached, idx_cached, _ = mla_ar.indexer(
            inputs_q=x_pref,
            low_rank_q=q_pref,
            inputs_kv=x_pref,
            inputs_positions=pos_pref,
            decoder_segment_ids=seg_pref,
            attention_mask=None,
            kv_cache=mla_ar.IndexerKVCache_0,
            model_mode=common_types.MODEL_MODE_PREFILL,
        )
        self.assertEqual(mask_cached.shape, (batch, 4, 4))
        self.assertEqual(idx_cached.shape, (batch, 4, 4))

        # 4. loss > 0 in non-train mode still takes fast-path; loss > 0 in TRAIN mode computes full score + approx_max_k.
        _, _, score_pref = mla_loss.indexer(
            inputs_q=x_pref,
            low_rank_q=q_pref,
            inputs_kv=x_pref,
            inputs_positions=pos_pref,
            decoder_segment_ids=seg_pref,
            attention_mask=None,
            kv_cache=mla_ar.IndexerKVCache_0,
            model_mode=common_types.MODEL_MODE_PREFILL,
        )
        self.assertIsNone(score_pref)

        mask_train_loss, idx_train_loss, score_train_loss = mla_loss.indexer(
            inputs_q=x_q,
            low_rank_q=q_low,
            inputs_kv=x_q,
            inputs_positions=pos,
            decoder_segment_ids=seg,
            attention_mask=attn_mask,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        self.assertEqual(mask_train_loss.shape, (batch, short_len, short_len))
        self.assertEqual(idx_train_loss.shape, (batch, short_len, short_len))
        self.assertEqual(score_train_loss.shape, (batch, short_len, short_len))

  def test_mla_reconstruct_mask_and_donor_validation_branches(self):
    """Covers _reconstruct_indexer_mask and _extract_and_validate_donor_state branches and ValueErrors."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = pyconfig.initialize(
          _tiny_glm5_args(tmpdir, scan_layers=False, prune_shared_indexers=True, seq_len=6, indexer_topk=4)
      )
      mesh = _make_mesh(cfg)
      batch = cfg.global_batch_size_to_train_on

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        mla = _make_mla(cfg, mesh)

        # _reconstruct_indexer_mask: kv_len <= topk with/without masks, and kv_len > topk with/without masks.
        idx_short = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), (batch, 3, 3))
        self.assertIsNone(mla._reconstruct_indexer_mask(idx_short, 3, None, None))
        pad_short = jnp.zeros((batch, 3), dtype=jnp.float32)
        m_short_pad = mla._reconstruct_indexer_mask(idx_short, 3, None, pad_short)
        self.assertEqual(m_short_pad.shape, (batch, 3, 3))

        idx_long = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch, 6, 4))
        m_long_bare = mla._reconstruct_indexer_mask(idx_long, 6, None, None)
        self.assertEqual(m_long_bare.shape, (batch, 6, 6))
        attn_long = jnp.zeros((batch, 6, 6), dtype=jnp.float32)
        pad_long = jnp.zeros((batch, 6), dtype=jnp.float32)
        m_long_full = mla._reconstruct_indexer_mask(idx_long, 6, attn_long, pad_long)
        self.assertEqual(m_long_full.shape, (batch, 6, 6))

        # _extract_and_validate_donor_state: None, malformed tuples, bad indices, and bad scores.
        self.assertIsNone(mla._extract_and_validate_donor_state(None, batch, 6, 4, 6, False, False))
        with self.assertRaisesRegex(ValueError, "Shared DSA layers require top-k indices"):
          mla._extract_and_validate_donor_state(None, batch, 6, 4, 6, False, True)

        for bad_tuple in ("not_a_tuple", (idx_long,), (idx_long, None, None)):
          with self.assertRaisesRegex(ValueError, "Expected cached_indexer_state to be a 2-tuple"):
            mla._extract_and_validate_donor_state(bad_tuple, batch, 6, 4, 6, False, True)

        for bad_indices in (
            None,
            "not_an_array",
            jnp.ones((batch, 6, 4), dtype=jnp.float32),
            jnp.zeros((batch, 6, 2), dtype=jnp.int32),
        ):
          with self.assertRaisesRegex(ValueError, "Expected cached topk_indices of integer shape"):
            mla._extract_and_validate_donor_state((bad_indices, None), batch, 6, 4, 6, False, True)

        for bad_score in (None, "not_an_array", jnp.zeros((batch, 6, 3), dtype=jnp.float32)):
          with self.assertRaisesRegex(ValueError, "Expected cached indexer_score of shape"):
            mla._extract_and_validate_donor_state((idx_long, bad_score), batch, 6, 4, 6, True, True)

        valid_score = jnp.zeros((batch, 6, 6), dtype=jnp.bfloat16)
        validated = mla._extract_and_validate_donor_state((idx_long, valid_score), batch, 6, 4, 6, True, True)
        self.assertIsInstance(validated, attention_mla.IndexerShareState)
        self.assertEqual(validated.indexer_score.dtype, jnp.float32)

  def test_mla_apply_index_share_static_dynamic_and_error_guards(self):
    """Covers _apply_index_share static layer_idx, out-of-bounds, schedule/indexer/shape guards."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = pyconfig.initialize(
          _tiny_glm5_args(
              tmpdir,
              scan_layers=False,
              prune_shared_indexers=False,
              seq_len=6,
              indexer_topk=4,
              indexer_loss_scaling_factor=0.5,
          )
      )
      mesh = _make_mesh(cfg)
      batch = cfg.global_batch_size_to_train_on

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        mla = _make_mla(cfg, mesh, is_shared_layer=False, served_group_size=2)
        x = jnp.ones((batch, 6, cfg.emb_dim), dtype=jnp.float32)
        pos = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32), (batch, 6))
        seg = jnp.ones((batch, 6), dtype=jnp.int32)

        # 1. Static Python int layer_idx=0 (Full) and layer_idx=1 (Shared, including static group_size loss scale).
        res_0 = mla(
            x,
            x,
            pos,
            seg,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_TRAIN,
            layer_idx=0,
            return_indexer_state=True,
        )
        out_0, _, donor_state = res_0  # pylint: disable=unbalanced-tuple-unpacking
        self.assertEqual(out_0.shape, (batch, 6, cfg.emb_dim))
        res_1 = mla(
            x,
            x,
            pos,
            seg,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_TRAIN,
            cached_indexer_state=donor_state,
            layer_idx=1,
            return_indexer_state=True,
        )
        out_1, _, shared_state = res_1  # pylint: disable=unbalanced-tuple-unpacking
        self.assertEqual(out_1.shape, (batch, 6, cfg.emb_dim))
        self.assertIsInstance(shared_state, attention_mla.IndexerShareState)

        # 2. Static layer_idx out of bounds (< 0 or >= num_decoder_layers).
        for bad_idx in (-1, 2):
          with self.assertRaisesRegex(ValueError, "out of bounds for num_decoder_layers"):
            mla(
                x,
                x,
                pos,
                seg,
                deterministic=True,
                model_mode=common_types.MODEL_MODE_TRAIN,
                cached_indexer_state=donor_state,
                layer_idx=bad_idx,
            )

        # 3. Direct _apply_index_share guards: bad attention_mask, missing schedule, missing indexer, bad outputs.
        q_dummy = jnp.ones((batch, 6, cfg.num_query_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim))
        k_dummy = jnp.ones((batch, 6, cfg.num_kv_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim))
        q_low = jnp.ones((batch, 6, cfg.q_lora_rank), dtype=jnp.float32)
        common_ais_kw = {
            "inputs_q": x,
            "low_rank_q": q_low,
            "inputs_kv": x,
            "inputs_positions": pos,
            "query": q_dummy,
            "key": k_dummy,
            "cached_values": [None],
            "decoder_segment_ids": seg,
            "previous_chunk": None,
            "model_mode": common_types.MODEL_MODE_TRAIN,
        }

        with self.assertRaisesRegex(ValueError, "Expected attention_mask last dim"):
          mla._apply_index_share(
              **common_ais_kw,
              attention_mask=jnp.zeros((batch, 6, 9), dtype=jnp.float32),
              cached_indexer_state=None,
              layer_idx=None,
          )

        saved_schedule = mla._index_share_schedule
        try:
          mla._index_share_schedule = None
          with self.assertRaisesRegex(ValueError, "Missing precomputed _index_share_schedule"):
            jax.jit(
                lambda dyn_idx: mla._apply_index_share(
                    **common_ais_kw,
                    attention_mask=None,
                    cached_indexer_state=donor_state,
                    layer_idx=dyn_idx,
                )
            )(jnp.int32(0))
        finally:
          mla._index_share_schedule = saved_schedule

        saved_indexer = mla.indexer
        try:
          mla.indexer = None
          with self.assertRaisesRegex(ValueError, "Full DSA layer requires an initialized self.indexer module"):
            mla._apply_index_share(
                **common_ais_kw,
                attention_mask=None,
                cached_indexer_state=None,
                layer_idx=None,
            )
          with self.assertRaisesRegex(ValueError, "Full DSA layer requires an initialized self.indexer module"):
            jax.jit(
                lambda dyn_idx: mla._apply_index_share(
                    **common_ais_kw,
                    attention_mask=None,
                    cached_indexer_state=donor_state,
                    layer_idx=dyn_idx,
                )
            )(jnp.int32(0))

          mla.indexer = lambda **kw: (None, None, None)
          with self.assertRaisesRegex(ValueError, "Full DSA indexer produced topk_indices of shape"):
            mla._apply_index_share(
                **common_ais_kw,
                attention_mask=None,
                cached_indexer_state=None,
                layer_idx=None,
            )

          mla.indexer = lambda **kw: (None, donor_state.topk_indices, None)
          with self.assertRaisesRegex(ValueError, "Full DSA indexer produced indexer_score of shape"):
            mla._apply_index_share(
                **common_ais_kw,
                attention_mask=None,
                cached_indexer_state=None,
                layer_idx=None,
            )
        finally:
          mla.indexer = saved_indexer

      # 4. Non-IndexShare MLA (GLM-5.1) with indexer_loss_scaling_factor > 0 and default 2-tuple return.
      cfg_51 = pyconfig.initialize(
          _tiny_glm5_args(
              tmpdir,
              model_name="glm5.1-744b",
              scan_layers=False,
              seq_len=6,
              indexer_topk=4,
              indexer_loss_scaling_factor=0.5,
          )
      )
      mesh_51 = _make_mesh(cfg_51)
      batch_51 = cfg_51.global_batch_size_to_train_on
      with jax.set_mesh(mesh_51), nn_partitioning.axis_rules(cfg_51.logical_axis_rules):
        mla_51 = _make_mla(cfg_51, mesh_51)
        x_51 = jnp.ones((batch_51, 6, cfg_51.emb_dim), dtype=jnp.float32)
        pos_51 = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32), (batch_51, 6))
        seg_51 = jnp.ones((batch_51, 6), dtype=jnp.int32)
        res_2tuple = mla_51(x_51, x_51, pos_51, seg_51, deterministic=True, model_mode=common_types.MODEL_MODE_TRAIN)
        self.assertEqual(len(res_2tuple), 2)
        self.assertTrue(hasattr(mla_51, "indexer_loss"))

  def test_mla_scanned_prefill_resizes_shared_indexer_cache(self):
    """Covers scanned Shared-layer IndexerKVCache_0 zero-resize when prompt length != max_prefill_predict_length."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = pyconfig.initialize(
          _tiny_glm5_args(
              tmpdir,
              scan_layers=True,
              prune_shared_indexers=False,
              num_layers=2,
              first_dense=0,
              pattern="FS",
              seq_len=6,
              max_prefill_len=4,
              indexer_topk=2,
          )
          + ["base_moe_mlp_dim=16"]
      )
      mesh = _make_mesh(cfg)
      batch = cfg.global_batch_size_to_train_on

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        model_ar = models.Transformer(
            cfg,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg),
            rngs=nnx.Rngs(params=11, dropout=12),
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        )
        # Initial IndexerKVCache_0 prefill buffer has length max_prefill_predict_length=4; pass q_len=2 != 4!
        short_prompt = jnp.ones((batch, 2), dtype=jnp.int32)
        short_pos = jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), (batch, 2))
        short_seg = jnp.ones((batch, 2), dtype=jnp.int32)
        logits_seg = model_ar(
            short_prompt,
            short_pos,
            decoder_segment_ids=short_seg,
            enable_dropout=False,
            model_mode=common_types.MODEL_MODE_PREFILL,
        )
        self.assertEqual(logits_seg.shape, (batch, 2, cfg.vocab_size))
        idx_cache = model_ar.decoder.moe_layers.self_attention.IndexerKVCache_0
        self.assertEqual(idx_cache.cached_prefill_key.get_value().shape[1], 2)
        self.assertEqual(idx_cache.cache_prefill_segment_id.get_value().shape[-1], 2)

        # Calling decoder directly with decoder_segment_ids=None covers the no-resize and None-segment branches.
        logits_noseg, *_ = model_ar.decoder(
            model_ar.token_embedder,
            short_prompt,
            short_pos,
            decoder_segment_ids=None,
            deterministic=True,
            model_mode=common_types.MODEL_MODE_PREFILL,
        )
        self.assertEqual(logits_noseg.shape, (batch, 2, cfg.vocab_size))

  def test_glm5_and_deepseek_layer_edge_branches(self):
    """Covers GLMGenericLayer layer_idx=-1, DeepSeek mHC + IndexShare guard, and batch-split _format_layer_return."""
    self.assertIsNotNone(glm5.GLMDenseLayerToLinen)
    self.assertIsNotNone(glm5.GLMMoELayerToLinen)

    with tempfile.TemporaryDirectory() as tmpdir:
      # 1. GLMGenericLayer with default layer_idx=-1 skips pattern lookup while still initializing MLA.
      cfg_glm = pyconfig.initialize(
          _tiny_glm5_args(tmpdir, scan_layers=False, prune_shared_indexers=True, seq_len=4, indexer_topk=2)
      )
      mesh = _make_mesh(cfg_glm)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_glm.logical_axis_rules):
        dense_neg_idx = glm5.GLMDenseLayer(
            config=cfg_glm,
            model_mode=common_types.MODEL_MODE_TRAIN,
            mesh=mesh,
            rngs=nnx.Rngs(params=0, dropout=1),
            layer_idx=-1,
        )
        self.assertFalse(dense_neg_idx.is_shared_layer)
        self.assertEqual(dense_neg_idx.served_group_size, 1)

      # 2. DeepSeek layer with mhc_expansion_rate=2: normal forward, self_attention_with_norm_op, and IndexShare guard.
      cfg_mhc = pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "model_name=deepseek3-671b",
              "override_model_config=True",
              "skip_jax_distributed_system=True",
              f"base_output_directory={tmpdir}",
              "run_name=ds_mhc_unit",
              "enable_checkpointing=False",
              "dtype=float32",
              "weight_dtype=float32",
              "attention=dot_product",
              "mhc_expansion_rate=2",
              "base_emb_dim=16",
              "base_mlp_dim=16",
              "base_moe_mlp_dim=8",
              "base_num_decoder_layers=2",
              "first_num_dense_layers=1",
              "base_num_query_heads=2",
              "base_num_kv_heads=2",
              "q_lora_rank=8",
              "kv_lora_rank=4",
              "qk_nope_head_dim=4",
              "qk_rope_head_dim=4",
              "v_head_dim=4",
              "num_experts=4",
              "num_experts_per_tok=2",
              "vocab_size=32",
              "max_position_embeddings=64",
              "original_max_position_embeddings=64",
              "max_target_length=4",
              "per_device_batch_size=1",
          ]
      )
      mesh_mhc = _make_mesh(cfg_mhc)
      batch_mhc = cfg_mhc.global_batch_size_to_train_on
      with jax.set_mesh(mesh_mhc), nn_partitioning.axis_rules(cfg_mhc.logical_axis_rules):
        ds_mhc = deepseek.DeepSeekDenseLayer(
            config=cfg_mhc,
            model_mode=common_types.MODEL_MODE_TRAIN,
            mesh=mesh_mhc,
            rngs=nnx.Rngs(params=2, dropout=3),
            layer_idx=0,
        )
        x_mhc = jnp.ones((batch_mhc, 4, 2, cfg_mhc.emb_dim), dtype=jnp.float32)
        pos_mhc = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_mhc, 4))
        seg_mhc = jnp.ones((batch_mhc, 4), dtype=jnp.int32)
        h_norm, inter_norm = ds_mhc.self_attention_with_norm_op(
            x_mhc, seg_mhc, pos_mhc, True, common_types.MODEL_MODE_TRAIN
        )
        self.assertEqual(h_norm.shape, x_mhc.shape)
        self.assertEqual(inter_norm.shape, x_mhc.shape)

        bad_cfg = copy.copy(cfg_mhc)
        object.__setattr__(bad_cfg, "use_index_share", True)
        ds_mhc.config = bad_cfg
        with self.assertRaisesRegex(ValueError, "use_index_share=True is not supported with mhc_expansion_rate > 1"):
          ds_mhc(x_mhc, seg_mhc, pos_mhc, True, common_types.MODEL_MODE_TRAIN)

      # 3. DeepSeekMoELayer use_batch_split_schedule=True return formatting (both qwix-fp8 and pure-JAX branches).
      cfg_ds = pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "model_name=deepseek3-671b",
              "override_model_config=True",
              "skip_jax_distributed_system=True",
              f"base_output_directory={tmpdir}",
              "run_name=ds_bs_unit",
              "enable_checkpointing=False",
              "scan_layers=False",
              "dtype=float32",
              "weight_dtype=float32",
              "attention=dot_product",
              "base_emb_dim=16",
              "base_mlp_dim=16",
              "base_moe_mlp_dim=8",
              "base_num_decoder_layers=2",
              "first_num_dense_layers=1",
              "base_num_query_heads=2",
              "base_num_kv_heads=2",
              "q_lora_rank=8",
              "kv_lora_rank=4",
              "qk_nope_head_dim=4",
              "qk_rope_head_dim=4",
              "v_head_dim=4",
              "num_experts=4",
              "num_experts_per_tok=2",
              "shared_experts=1",
              "n_routing_groups=1",
              "topk_routing_group=1",
              "vocab_size=32",
              "max_position_embeddings=64",
              "original_max_position_embeddings=64",
              "max_target_length=4",
              "per_device_batch_size=1",
          ]
      )
      mesh_ds = _make_mesh(cfg_ds)
      batch_ds = cfg_ds.global_batch_size_to_train_on
      with jax.set_mesh(mesh_ds), nn_partitioning.axis_rules(cfg_ds.logical_axis_rules):
        ds_moe = deepseek.DeepSeekMoELayer(
            config=cfg_ds,
            model_mode=common_types.MODEL_MODE_TRAIN,
            mesh=mesh_ds,
            rngs=nnx.Rngs(params=4, dropout=5),
            layer_idx=1,
        )
        x_ds = jnp.ones((batch_ds, 4, cfg_ds.emb_dim), dtype=jnp.float32)
        pos_ds = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_ds, 4))
        seg_ds = jnp.ones((batch_ds, 4), dtype=jnp.int32)

        cfg_bs_fp8 = copy.copy(cfg_ds)
        object.__setattr__(cfg_bs_fp8, "use_batch_split_schedule", True)
        object.__setattr__(cfg_bs_fp8, "batch_split_factor", 2)
        object.__setattr__(cfg_bs_fp8, "quantization", "fp8")
        object.__setattr__(cfg_bs_fp8, "use_qwix_quantization", True)
        object.__setattr__(cfg_bs_fp8, "use_manual_quantization", False)
        ds_moe.config = cfg_bs_fp8
        with mock.patch.object(deepseek, "deepseek_batchsplit_fp8") as mock_fp8:
          mock_fp8.split.side_effect = lambda arr, split_factor=2: [arr] * split_factor
          mock_fp8.fetch_weights.return_value = {}
          mock_fp8.batch_split_schedule.side_effect = lambda ubatches, *a, **kw: ubatches
          mock_fp8.merge.side_effect = lambda ubatches, split_factor=2: ubatches[0]
          res_fp8 = ds_moe((x_ds, None), seg_ds, pos_ds, True, common_types.MODEL_MODE_TRAIN)
        self.assertEqual(len(res_fp8), 2)
        self.assertIsNone(res_fp8[1])

        cfg_bs_jax = copy.copy(cfg_ds)
        object.__setattr__(cfg_bs_jax, "use_batch_split_schedule", True)
        object.__setattr__(cfg_bs_jax, "batch_split_factor", 2)
        object.__setattr__(cfg_bs_jax, "quantization", "")
        ds_moe.config = cfg_bs_jax
        with (
            mock.patch.object(deepseek, "deepseek_batchsplit") as mock_bs,
            mock.patch.object(jax, "reshard", side_effect=lambda arr, _: arr),
        ):
          mock_bs.initialize_yarn_freqs.return_value = jnp.ones((batch_ds, 4, 4), dtype=jnp.float32)
          mock_bs.initialize_yarn_mask.return_value = jnp.ones((4,), dtype=jnp.float32)
          mock_bs.init_splash_kernel.return_value = None
          mock_bs.split.side_effect = lambda arr, split_factor=2: [arr] * split_factor
          mock_bs.fetch_weights.return_value = {}
          mock_bs.gather_weights.return_value = {}
          mock_bs.batch_split_schedule.return_value = ([x_ds, (x_ds, x_ds, x_ds, pos_ds)], None)
          mock_bs.unroute_ubatch_shard_mapped.return_value = (x_ds, None)
          mock_bs.merge.side_effect = lambda ubatches, split_factor=2: ubatches[0]
          res_jax = ds_moe(x_ds, seg_ds, pos_ds, True, common_types.MODEL_MODE_TRAIN)
        self.assertEqual(len(res_jax), 2)
        self.assertIsNone(res_jax[1])

  def test_nnx_decoder_run_layers_sequentially_and_hy4_unscanned(self):
    """Covers NNXDecoder._run_layers_sequentially edge cases and unscanned HY4 prev_indexer_mask propagation."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = pyconfig.initialize(
          _tiny_glm5_args(
              tmpdir,
              scan_layers=True,
              prune_shared_indexers=False,
              num_layers=2,
              first_dense=1,
              pattern="FS",
              seq_len=6,
              max_prefill_len=4,
              indexer_topk=2,
          )
      )
      mesh = _make_mesh(cfg)
      batch = cfg.global_batch_size_to_train_on

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        model = models.Transformer(
            cfg,
            mesh=mesh,
            quant=quantizations.configure_quantization(cfg),
            rngs=nnx.Rngs(params=21, dropout=22),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        dec = model.decoder
        x = jnp.ones((batch, 4, cfg.emb_dim), dtype=jnp.float32)
        pos = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch, 4))
        seg = jnp.ones((batch, 4), dtype=jnp.int32)

        # 1. Starting scan on a Shared layer (start_layer_idx=1 in "FS") with cached_indexer_state=None raises.
        with self.assertRaisesRegex(ValueError, "Shared DSA layers require top-k indices"):
          dec._run_layers_sequentially(
              dec.moe_layers,
              x,
              seg,
              pos,
              True,
              common_types.MODEL_MODE_TRAIN,
              length=1,
              track_indexer_state=True,
              start_layer_idx=1,
              cached_indexer_state=None,
          )

        # 2. length=0 returns initial carry and cached_indexer_state immediately.
        y0, _, kv0, state0 = dec._run_layers_sequentially(
            dec.dense_layers,
            x,
            length=0,
            track_indexer_state=True,
            start_layer_idx=0,
            cached_indexer_state=None,
        )
        np.testing.assert_array_equal(np.asarray(y0), np.asarray(x))
        self.assertIsNone(kv0)
        self.assertIsNone(state0)

        # 3. Static unroll with kv_caches_stacked (use_kv=True) + track_indexer_state=True + scan_model_mode=None.
        y_kv, _, ret_kv, out_idx_state = dec._run_layers_sequentially(
            dec.dense_layers,
            x,
            seg,
            pos,
            True,
            common_types.MODEL_MODE_TRAIN,
            length=1,
            kv_caches_stacked=[None],
            track_indexer_state=True,
            start_layer_idx=0,
            cached_indexer_state=None,
            scan_model_mode=None,
        )
        self.assertEqual(y_kv.shape, x.shape)
        self.assertIsNone(ret_kv)
        self.assertIsInstance(out_idx_state, attention_mla.IndexerShareState)

        # 4. Scanned layer returning a bare Array (not tuple) in _apply_layers_sequentially (scan & static kv unroll).
        bare_scanned = dec._create_scanned_layers(
            _NonTupleScanLayer, length=2, metadata_axis_name="layers", rngs=nnx.Rngs(params=30)
        )
        y_bare, _, kv_bare = dec._apply_layers_sequentially(bare_scanned, x, length=2)
        np.testing.assert_allclose(np.asarray(y_bare), np.asarray(x) + 2.0, rtol=1e-6)
        self.assertIsNone(kv_bare)
        y_bare_kv, _, _ = dec._apply_layers_sequentially(bare_scanned, x, length=2, kv_caches_stacked=[None, None])
        np.testing.assert_allclose(np.asarray(y_bare_kv), np.asarray(x) + 2.0, rtol=1e-6)

      # 5. Scanned & unscanned GLM-5.1 (use_index_share=False) cover NNXDecoder non-IndexShare branches.
      for scan_51 in (True, False):
        cfg_51 = pyconfig.initialize(_tiny_glm5_args(tmpdir, model_name="glm5.1-744b", scan_layers=scan_51, seq_len=4))
        mesh_51 = _make_mesh(cfg_51)
        batch_51 = cfg_51.global_batch_size_to_train_on
        with jax.set_mesh(mesh_51), nn_partitioning.axis_rules(cfg_51.logical_axis_rules):
          m_51 = models.Transformer(
              cfg_51,
              mesh=mesh_51,
              quant=quantizations.configure_quantization(cfg_51),
              rngs=nnx.Rngs(params=35, dropout=36),
              model_mode=common_types.MODEL_MODE_TRAIN,
          )
          toks_51 = jnp.ones((batch_51, 4), dtype=jnp.int32)
          pos_51 = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_51, 4))
          seg_51 = jnp.ones((batch_51, 4), dtype=jnp.int32)
          out_51 = m_51(toks_51, pos_51, decoder_segment_ids=seg_51, enable_dropout=False)
          self.assertEqual(out_51.shape, (batch_51, 4, cfg_51.vocab_size))

      # 6. Unscanned HY4 decoder block forward pass (with remat_policy=full) covers prev_indexer_mask + checkpointed_fn.
      cfg_hy4 = pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "model_name=hy4-preview",
              "override_model_config=True",
              "skip_jax_distributed_system=True",
              f"base_output_directory={tmpdir}",
              "run_name=hy4_unscan_unit",
              "enable_checkpointing=False",
              "scan_layers=False",
              "remat_policy=full",
              "dtype=float32",
              "weight_dtype=float32",
              "attention=dot_product",
              "mhc_expansion_rate=2",
              "base_emb_dim=16",
              "base_mlp_dim=16",
              "base_moe_mlp_dim=8",
              "base_num_decoder_layers=2",
              "first_num_dense_layers=1",
              "base_num_query_heads=2",
              "base_num_kv_heads=2",
              "head_dim=8",
              "q_lora_rank=8",
              "kv_lora_rank=4",
              "qk_nope_head_dim=4",
              "qk_rope_head_dim=4",
              "v_head_dim=8",
              "use_indexer=True",
              "indexer_n_heads=2",
              "indexer_head_dim=4",
              "indexer_topk=2",
              "num_experts=4",
              "num_experts_per_tok=2",
              "shared_experts=1",
              "vocab_size=32",
              "max_position_embeddings=64",
              "original_max_position_embeddings=64",
              "max_target_length=4",
              "per_device_batch_size=1",
          ]
      )
      mesh_hy4 = _make_mesh(cfg_hy4)
      batch_hy4 = cfg_hy4.global_batch_size_to_train_on
      with jax.set_mesh(mesh_hy4), nn_partitioning.axis_rules(cfg_hy4.logical_axis_rules):
        m_hy4 = models.Transformer(
            cfg_hy4,
            mesh=mesh_hy4,
            quant=quantizations.configure_quantization(cfg_hy4),
            rngs=nnx.Rngs(params=41, dropout=42),
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        toks_hy4 = jnp.ones((batch_hy4, 4), dtype=jnp.int32)
        pos_hy4 = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (batch_hy4, 4))
        seg_hy4 = jnp.ones((batch_hy4, 4), dtype=jnp.int32)
        logits_hy4 = m_hy4(
            toks_hy4,
            pos_hy4,
            decoder_segment_ids=seg_hy4,
            enable_dropout=False,
            model_mode=common_types.MODEL_MODE_TRAIN,
        )
        self.assertEqual(logits_hy4.shape, (batch_hy4, 4, cfg_hy4.vocab_size))


if __name__ == "__main__":
  unittest.main()
