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

"""Multi-device CPU parity and unit tests for generalized batch-split (`enable_batch_split`)."""

from __future__ import annotations

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.configs import pyconfig
from maxtext.layers import batch_split
from maxtext.layers import fsdp_prefetch_overlap_transform
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path


def _make_dsv3_config(num_moe_layers: int, first_num_dense_layers: int = 1, **overrides):
  """Builds a tiny DeepSeek-V3 config suitable for 8-device (DP=2, FSDP=2, EP=2) CPU testing."""
  total_layers = first_num_dense_layers + num_moe_layers
  defaults = {
      "model_name": "deepseek3-671b",
      "decoder_block": "deepseek",
      "override_model_config": True,
      "run_name": "batch_split_unit_test",
      "base_emb_dim": 32,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 32,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 8,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 4,
      "qk_rope_head_dim": 4,
      "v_head_dim": 8,
      "vocab_size": 64,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "n_routing_groups": 2,
      "topk_routing_group": 1,
      "shared_experts": 1,
      "first_num_dense_layers": first_num_dense_layers,
      "base_num_decoder_layers": total_layers,
      "num_decoder_layers": total_layers,
      "max_target_length": 8,
      "per_device_batch_size": 2.0,
      "ici_data_parallelism": 2,
      "ici_fsdp_parallelism": 2,
      "ici_expert_parallelism": 2,
      "ici_tensor_parallelism": 1,
      "scan_layers": True,
      "sparse_matmul": True,
      "use_ring_of_experts": True,
      "use_ragged_sort": False,
      "megablox": False,
      "use_tokamax_gmm": False,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "remat_policy": "full",
      "dropout_rate": 0.0,
      "load_balance_loss_weight": 0.0,
      "routed_bias": True,
      "routed_bias_update_rate": 0.0,
      "routed_score_func": "sigmoid",
      "enable_batch_split": False,
      "prefetch_fsdp_weights": False,
      "prefetch_fsdp_defer_grad_reduce_scatter": False,
      "shard_mode": "explicit",
      "param_scan_axis": 1,
  }
  defaults.update(overrides)
  args = ["", get_test_config_path()]
  for k, v in defaults.items():
    args.append(f"{k}={v}")
  return pyconfig.initialize(args)


def _make_mesh(cfg):
  return maxtext_utils.get_mesh_from_config(cfg)


def _shard_nnx_module(module, mesh):
  graphdef, state = nnx.split(module)
  state = jax.tree.map(jax.device_put, state, sharding.nnx_construct_named_sharding(state, mesh))
  return nnx.merge(graphdef, state)


def _run_loss_and_grads(cfg):
  """Runs a jitted forward+backward pass on NNXDecoder and returns (loss, grads, intermediates)."""
  mesh = _make_mesh(cfg)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    shared_emb = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=0.5),
        config=cfg,
        mesh=mesh,
        rngs=nnx.Rngs(params=1),
    )
    nnx.pop(decoder, nnx.RngState, nnx.RngCount)
    decoder = _shard_nnx_module(decoder, mesh)
    shared_emb = _shard_nnx_module(shared_emb, mesh)

    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(42), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None], (batch, seq_len))
    if cfg.shard_mode == ShardMode.EXPLICIT:
      batch_pspec = sharding.logical_to_mesh_axes(
          ("activation_batch", "activation_length"), mesh, rules=cfg.logical_axis_rules
      )
      data_sharding = jax.sharding.NamedSharding(mesh, batch_pspec)
      ids = jax.device_put(ids, data_sharding)
      segment_ids = jax.device_put(segment_ids, data_sharding)
      positions = jax.device_put(positions, data_sharding)

    graphdef, params, state = nnx.split(decoder, nnx.Param, ...)

    def loss_fn(p):
      m = nnx.merge(graphdef, p, state, copy=True)
      logits, _, _ = m(
          shared_emb,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      loss = jnp.mean(logits.astype(jnp.float32) ** 2)
      inter = nnx.pop(m, nnx.Intermediate)
      # Fold sown moe_lb_loss into total loss so its gradients are verified too.
      inter_dict = inter.to_pure_dict() if len(inter) > 0 else {}
      lb_vals = _find_by_suffix(inter_dict, "moe_lb_loss")
      if lb_vals:
        lb_stacked = jnp.concatenate([jnp.ravel(jnp.asarray(v[0] if isinstance(v, tuple) else v)) for v in lb_vals])
        loss = loss + (jnp.sum(lb_stacked) if cfg.routed_score_func == "sigmoid" else jnp.mean(lb_stacked))
      return loss, inter_dict

    (loss, inter_dict), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
    return np.asarray(loss), grads, inter_dict


def _find_by_suffix(tree: dict, suffix: str) -> list:
  found = []
  for k, v in tree.items():
    if k == suffix:
      found.append(v)
    elif isinstance(v, dict):
      found.extend(_find_by_suffix(v, suffix))
  return found


class BatchSplitConfigValidationTest(unittest.TestCase):
  """Tests config validation rules for `enable_batch_split`."""

  def test_rejects_incompatible_flags(self):
    with self.assertRaisesRegex(ValueError, "scan_layers=True"):
      _make_dsv3_config(num_moe_layers=2, enable_batch_split=True, scan_layers=False)
    with self.assertRaisesRegex(ValueError, "use_batch_split_schedule"):
      _make_dsv3_config(num_moe_layers=2, enable_batch_split=True, use_batch_split_schedule=True)
    with self.assertRaisesRegex(ValueError, "use_ring_of_experts=True"):
      _make_dsv3_config(num_moe_layers=2, enable_batch_split=True, use_ring_of_experts=False)


class BatchSplitGradParityTest(parameterized.TestCase):
  """Verifies forward loss, parameter gradients, moe_lb_loss, and moe_bias_updates parity."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 8:
      self.skipTest(f"Requires 8 simulated CPU devices, got {jax.device_count()}")

  def _assert_parity(self, num_moe_layers: int, rtol: float = 2e-4, atol: float = 2e-5, **overrides):
    """Asserts loss, gradient, and aux metric parity between regular and batch-split runs."""
    ref_cfg = _make_dsv3_config(num_moe_layers=num_moe_layers, enable_batch_split=False, **overrides)
    bs_cfg = _make_dsv3_config(num_moe_layers=num_moe_layers, enable_batch_split=True, **overrides)

    ref_loss, ref_grads, ref_inter = _run_loss_and_grads(ref_cfg)
    bs_loss, bs_grads, bs_inter = _run_loss_and_grads(bs_cfg)

    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)

    ref_leaves, ref_def = jax.tree_util.tree_flatten(ref_grads)
    bs_leaves, bs_def = jax.tree_util.tree_flatten(bs_grads)
    self.assertEqual(ref_def, bs_def)
    for idx, (ref_g, bs_g) in enumerate(zip(ref_leaves, bs_leaves)):
      self.assertEqual(ref_g.shape, bs_g.shape, f"Gradient leaf {idx} shape mismatch")
      np.testing.assert_allclose(
          np.asarray(bs_g),
          np.asarray(ref_g),
          rtol=rtol,
          atol=atol,
          err_msg=f"Gradient leaf {idx} mismatch (shape={ref_g.shape})",
      )

    if overrides.get("load_balance_loss_weight", 0.0) > 0.0:
      ref_lbs = _find_by_suffix(ref_inter, "moe_lb_loss")
      bs_lbs = _find_by_suffix(bs_inter, "moe_lb_loss")
      self.assertEqual(len(ref_lbs), len(bs_lbs))
      self.assertNotEmpty(bs_lbs)
      for r, b in zip(ref_lbs, bs_lbs):
        r_arr = np.asarray(r[0] if isinstance(r, tuple) else r)
        b_arr = np.asarray(b[0] if isinstance(b, tuple) else b)
        np.testing.assert_allclose(b_arr, r_arr, rtol=1e-5, atol=1e-6)

    if overrides.get("routed_bias_update_rate", 0.0) > 0.0:
      ref_bu = _find_by_suffix(ref_inter, "moe_bias_updates")
      bs_bu = _find_by_suffix(bs_inter, "moe_bias_updates")
      self.assertEqual(len(ref_bu), len(bs_bu))
      self.assertNotEmpty(bs_bu)
      for r, b in zip(ref_bu, bs_bu):
        r_arr = np.asarray(r[0] if isinstance(r, tuple) else r)
        b_arr = np.asarray(b[0] if isinstance(b, tuple) else b)
        np.testing.assert_allclose(b_arr, r_arr, rtol=0.0, atol=1e-6)

  @parameterized.named_parameters(
      ("1_layer", 1),
      ("2_layers", 2),
      ("4_layers", 4),
      ("5_layers", 5),
      ("7_layers_ordered", 7),
  )
  def test_explicit_with_fsdp_prefetch_and_deferred_rs(self, num_moe_layers):
    self._assert_parity(
        num_moe_layers=num_moe_layers,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )

  def test_explicit_batch_split_prefetch_vs_regular_scan(self):
    ref_cfg = _make_dsv3_config(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        enable_batch_split=False,
    )
    bs_cfg = _make_dsv3_config(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        enable_batch_split=True,
    )
    ref_loss, ref_grads, _ = _run_loss_and_grads(ref_cfg)
    bs_loss, bs_grads, _ = _run_loss_and_grads(bs_cfg)
    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)
    ref_items = jax.tree_util.tree_leaves_with_path(ref_grads)
    bs_items = jax.tree_util.tree_leaves_with_path(bs_grads)
    for (p, r), (_, b) in zip(ref_items, bs_items):
      np.testing.assert_allclose(
          np.asarray(b),
          np.asarray(r),
          rtol=2e-4,
          atol=2e-5,
          err_msg=f"Grad mismatch at {jax.tree_util.keystr(p)} (shape={r.shape})",
      )

  @parameterized.named_parameters(
      ("custom_1_layer", 1, 0, "custom"),
      ("custom_5_layers", 5, 1, "custom"),
      ("minimal_with_context_4_layers", 4, 0, "minimal_with_context"),
  )
  def test_selective_remat_policy_parity(self, num_moe_layers, first_num_dense_layers, remat_policy):
    extra = {}
    if remat_policy == "custom":
      extra = {
          "context": "device",
          "out_proj": "device",
          "decoder_layer_input": "device",
          "query_wa_proj": "device",
          "kv_wa_proj": "device",
          "mla_q": "device",
          "mla_kv": "device",
          "moe_routing_maps": "device",
      }
    common = {
        "num_moe_layers": num_moe_layers,
        "first_num_dense_layers": first_num_dense_layers,
        "base_mlp_dim": 32,
        "shard_mode": "explicit",
        "remat_policy": remat_policy,
        "moe_topk_before_ep_all_gather": True,
        "moe_use_megatron_seq_aux_loss": True,
        "load_balance_loss_weight": 0.01,
        "routed_bias_update_rate": 0.001,
    } | extra
    # Compare against regular scan (`prefetch_fsdp_weights=False`) because `fsdp_prefetch.py` leaves
    # non-parameter NNX state read-only and drops sown `moe_lb_loss` / `moe_bias_updates`, whereas
    # `batch_split.py` preserves both.
    ref_cfg = _make_dsv3_config(
        enable_batch_split=False,
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        **common,
    )
    bs_cfg = _make_dsv3_config(
        enable_batch_split=True,
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        **common,
    )
    ref_loss, ref_grads, ref_inter = _run_loss_and_grads(ref_cfg)
    bs_loss, bs_grads, bs_inter = _run_loss_and_grads(bs_cfg)
    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)
    for (p, r), (_, b) in zip(
        jax.tree_util.tree_leaves_with_path(ref_grads), jax.tree_util.tree_leaves_with_path(bs_grads)
    ):
      np.testing.assert_allclose(
          np.asarray(b),
          np.asarray(r),
          rtol=2e-4,
          atol=2e-5,
          err_msg=f"Grad mismatch at {jax.tree_util.keystr(p)} (shape={r.shape})",
      )
    for key in ("moe_lb_loss", "moe_bias_updates"):
      ref_vals = _find_by_suffix(ref_inter, key)
      bs_vals = _find_by_suffix(bs_inter, key)
      self.assertEqual(len(ref_vals), len(bs_vals))
      self.assertNotEmpty(bs_vals)
      for r, b in zip(ref_vals, bs_vals):
        r_arr = np.asarray(r[0] if isinstance(r, tuple) else r)
        b_arr = np.asarray(b[0] if isinstance(b, tuple) else b)
        np.testing.assert_allclose(b_arr, r_arr, rtol=1e-5, atol=1e-6)

  def test_explicit_without_fsdp_prefetch(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
    )

  def test_auto_sharding_without_prefetch(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="auto",
        prefetch_fsdp_weights=False,
    )

  def test_topk_before_ep_all_gather_false(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        moe_topk_before_ep_all_gather=False,
    )

  def test_topk_before_ep_all_gather_true(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        moe_topk_before_ep_all_gather=True,
    )

  def test_sigmoid_megatron_lb_loss_and_routed_bias_updates(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=False,
        load_balance_loss_weight=0.05,
        routed_score_func="sigmoid",
        routed_bias=True,
        routed_bias_update_rate=0.01,
        moe_topk_before_ep_all_gather=True,
    )

  def test_softmax_lb_loss_and_routed_bias_updates(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=False,
        load_balance_loss_weight=0.05,
        routed_score_func="softmax",
        routed_bias=True,
        routed_bias_update_rate=0.01,
        moe_topk_before_ep_all_gather=True,
    )

  def test_param_scan_axis_zero(self):
    self._assert_parity(
        num_moe_layers=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        param_scan_axis=0,
    )

  def test_per_device_batch_size_one_falls_back_cleanly(self):
    self._assert_parity(
        num_moe_layers=2,
        shard_mode="explicit",
        per_device_batch_size=1.0,
        prefetch_fsdp_weights=False,
    )

  def _assert_cfg_pair_parity(
      self, ref_cfg, bs_cfg, *, expect_batch_split_used: bool = True, rtol: float = 2e-4, atol: float = 2e-5
  ):
    """Asserts loss and gradient parity for an explicit pair of reference and batch-split configs."""
    ref_loss, ref_grads, ref_inter = _run_loss_and_grads(ref_cfg)
    with mock.patch.object(
        batch_split,
        "_custom_vjp_batch_split_pipeline",
        wraps=batch_split._custom_vjp_batch_split_pipeline,  # pylint: disable=protected-access
    ) as bs_spy:
      bs_loss, bs_grads, bs_inter = _run_loss_and_grads(bs_cfg)

    self.assertEqual(bs_spy.call_count, 1 if expect_batch_split_used else 0)
    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)
    ref_items = jax.tree_util.tree_leaves_with_path(ref_grads)
    bs_items = jax.tree_util.tree_leaves_with_path(bs_grads)
    self.assertEqual(len(ref_items), len(bs_items))
    for (p, r), (_, b) in zip(ref_items, bs_items):
      np.testing.assert_allclose(
          np.asarray(b),
          np.asarray(r),
          rtol=rtol,
          atol=atol,
          err_msg=f"Grad mismatch at {jax.tree_util.keystr(p)} (shape={r.shape})",
      )
    if bs_cfg.load_balance_loss_weight > 0.0:
      ref_lbs = _find_by_suffix(ref_inter, "moe_lb_loss")
      bs_lbs = _find_by_suffix(bs_inter, "moe_lb_loss")
      self.assertEqual(len(ref_lbs), len(bs_lbs))
      self.assertNotEmpty(bs_lbs)
      for r, b in zip(ref_lbs, bs_lbs):
        r_arr = np.asarray(r[0] if isinstance(r, tuple) else r)
        b_arr = np.asarray(b[0] if isinstance(b, tuple) else b)
        np.testing.assert_allclose(b_arr, r_arr, rtol=1e-5, atol=1e-6)

  def test_qwen3_moe_explicit_prefetch_deferred_rs_and_lb_loss(self):
    def _qwen3_moe_cfg(enable_bs: bool, prefetch: bool):
      args = [
          "",
          get_test_config_path(),
          "model_name=qwen3-235b-a22b",
          "decoder_block=qwen3_moe",
          "override_model_config=True",
          "run_name=batch_split_qwen3_moe_unit_test",
          "base_emb_dim=32",
          "base_mlp_dim=32",
          "base_moe_mlp_dim=32",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=8",
          "use_qk_norm=True",
          "vocab_size=64",
          "num_experts=4",
          "num_experts_per_tok=2",
          "shared_experts=0",
          "norm_topk_prob=True",
          "load_balance_loss_weight=0.05",
          "base_num_decoder_layers=4",
          "num_decoder_layers=4",
          "max_target_length=8",
          "per_device_batch_size=2.0",
          "ici_data_parallelism=2",
          "ici_fsdp_parallelism=2",
          "ici_expert_parallelism=2",
          "ici_tensor_parallelism=1",
          "scan_layers=True",
          "param_scan_axis=1",
          "sparse_matmul=True",
          "use_ring_of_experts=True",
          "use_ragged_sort=False",
          "megablox=False",
          "use_tokamax_gmm=False",
          "capacity_factor=-1",
          "attention=dot_product",
          "dtype=float32",
          "weight_dtype=float32",
          "matmul_precision=highest",
          "remat_policy=full",
          "dropout_rate=0.0",
          "shard_mode=explicit",
          f"prefetch_fsdp_weights={prefetch}",
          f"prefetch_fsdp_defer_grad_reduce_scatter={prefetch}",
          f"enable_batch_split={enable_bs}",
      ]
      return pyconfig.initialize(args)

    self._assert_cfg_pair_parity(_qwen3_moe_cfg(False, False), _qwen3_moe_cfg(True, True))

  def test_mixtral_explicit_prefetch_deferred_rs_and_lb_loss(self):
    def _mixtral_cfg(enable_bs: bool, prefetch: bool):
      args = [
          "",
          get_test_config_path(),
          "model_name=mixtral-8x22b",
          "decoder_block=mixtral",
          "override_model_config=True",
          "run_name=batch_split_mixtral_unit_test",
          "base_emb_dim=32",
          "base_mlp_dim=32",
          "base_moe_mlp_dim=32",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=8",
          "vocab_size=64",
          "num_experts=4",
          "num_experts_per_tok=2",
          "shared_experts=0",
          "load_balance_loss_weight=0.05",
          "base_num_decoder_layers=4",
          "num_decoder_layers=4",
          "max_target_length=8",
          "per_device_batch_size=2.0",
          "ici_data_parallelism=2",
          "ici_fsdp_parallelism=2",
          "ici_expert_parallelism=2",
          "ici_tensor_parallelism=1",
          "scan_layers=True",
          "param_scan_axis=1",
          "sparse_matmul=True",
          "use_ring_of_experts=True",
          "use_ragged_sort=False",
          "megablox=False",
          "use_tokamax_gmm=False",
          "capacity_factor=-1",
          "attention=dot_product",
          "dtype=float32",
          "weight_dtype=float32",
          "matmul_precision=highest",
          "remat_policy=full",
          "dropout_rate=0.0",
          "shard_mode=explicit",
          f"prefetch_fsdp_weights={prefetch}",
          f"prefetch_fsdp_defer_grad_reduce_scatter={prefetch}",
          f"enable_batch_split={enable_bs}",
      ]
      return pyconfig.initialize(args)

    self._assert_cfg_pair_parity(_mixtral_cfg(False, False), _mixtral_cfg(True, True))

  def test_deepseek3_2_dsa_indexer_and_kimi_k2_high_experts(self):
    # 1. DeepSeek-V3.2 with DSA Indexer (`use_indexer=True`, `indexer_loss_scaling_factor=0.0`).
    dsv32_common = {
        "model_name": "deepseek3.2-671b",
        "num_moe_layers": 3,
        "first_num_dense_layers": 1,
        "use_indexer": True,
        "indexer_n_heads": 4,
        "indexer_head_dim": 8,
        "indexer_topk": 4,
        "indexer_loss_scaling_factor": 0.0,
        "rope_type": "yarn",
        "load_balance_loss_weight": 0.05,
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "shard_mode": "explicit",
    }
    self._assert_cfg_pair_parity(
        _make_dsv3_config(enable_batch_split=False, prefetch_fsdp_weights=False, **dsv32_common),
        _make_dsv3_config(
            enable_batch_split=True,
            prefetch_fsdp_weights=True,
            prefetch_fsdp_defer_grad_reduce_scatter=True,
            **dsv32_common,
        ),
    )

    # 2. When `indexer_loss_scaling_factor > 0.0` (dense warmup), `batch_split_supported()` cleanly falls back.
    warmup_common = dsv32_common | {
        "indexer_loss_scaling_factor": 0.1,
        "indexer_sparse_training": True,
        "load_balance_loss_weight": 0.0,
        "routed_bias_update_rate": 0.0,
    }
    self._assert_cfg_pair_parity(
        _make_dsv3_config(enable_batch_split=False, prefetch_fsdp_weights=False, **warmup_common),
        _make_dsv3_config(enable_batch_split=True, prefetch_fsdp_weights=False, **warmup_common),
        expect_batch_split_used=False,
    )

    # 3. Kimi-K2 (`model_name="kimi-k2-1t"`, higher expert count `E=8`, `routed_scaling_factor=2.827`).
    kimi_common = {
        "model_name": "kimi-k2-1t",
        "num_moe_layers": 3,
        "first_num_dense_layers": 1,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_routing_groups": 1,
        "topk_routing_group": 1,
        "routed_scaling_factor": 2.827,
        "load_balance_loss_weight": 0.05,
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "shard_mode": "explicit",
    }
    self._assert_cfg_pair_parity(
        _make_dsv3_config(enable_batch_split=False, prefetch_fsdp_weights=False, **kimi_common),
        _make_dsv3_config(
            enable_batch_split=True,
            prefetch_fsdp_weights=True,
            prefetch_fsdp_defer_grad_reduce_scatter=True,
            **kimi_common,
        ),
    )


class BatchSplitOverlapQueueParsingTest(unittest.TestCase):
  """Verifies `async_queue`, VMEM DMA filtering, and `use_compute_epochs` on SC0 EP and SC1 FSDP queues."""

  def test_extracts_top_level_core_ids_ignoring_nested_consumers(self):
    sc0_inst = (
        "%call-start.1 = ((bf16[4096,7168]), bf16[32768,7168]) async-start(%slice.1), "
        'async_execution_thread="sparsecore", '
        'backend_config={"operation_queue_id":"0","sparse_core_config":{"consumers":[{"core_ids":[]}],"core_ids":["0"]}}'
    )
    sc1_inst = (
        "%call-start.2 = ((bf16[448,7168]), bf16[7168,7168]) async-start(%slice.2), "
        'async_execution_thread="sparsecore", '
        'backend_config={"operation_queue_id":"0","sparse_core_config":{"consumers":[{"core_ids":[]}],"core_ids":["1"]}}'
    )
    self.assertEqual(
        fsdp_prefetch_overlap_transform.async_queue("kAsyncStart", sc0_inst),
        "sparsecore:0",
    )
    self.assertEqual(
        fsdp_prefetch_overlap_transform.async_queue("kAsyncStart", sc1_inst),
        "sparsecore:1",
    )

  def test_separate_sc0_sc1_queues_allow_sc1_prefetch_done_to_sink_past_sc0_ep_done(self):
    order = ("p0", "sc1_ag_start", "sc1_ag_done", "dot_attn", "sc0_ep_start", "sc0_ep_done", "dot_gmm", "root")
    opcodes = {
        "p0": "kParameter",
        "sc1_ag_start": "kAsyncStart",
        "sc1_ag_done": "kAsyncDone",
        "dot_attn": "kDot",
        "sc0_ep_start": "kAsyncStart",
        "sc0_ep_done": "kAsyncDone",
        "dot_gmm": "kDot",
        "root": "kTuple",
    }
    operands = {
        "p0": (),
        "sc1_ag_start": ("p0",),
        "sc1_ag_done": ("sc1_ag_start",),
        "dot_attn": ("p0",),
        "sc0_ep_start": ("dot_attn",),
        "sc0_ep_done": ("sc0_ep_start",),
        "dot_gmm": ("sc0_ep_done",),
        "root": ("dot_gmm", "sc1_ag_done"),
    }
    graph = fsdp_prefetch_overlap_transform.make_schedule_graph(
        order=order,
        opcodes=opcodes,
        operands=operands,
        queues={"sc1_ag_start": "sparsecore:1", "sc0_ep_start": "sparsecore:0"},
    )
    new_order = fsdp_prefetch_overlap_transform.reschedule(graph)
    fsdp_prefetch_overlap_transform.validate_order(graph, new_order)
    # SC1 FSDP prefetch starts before dot_attn and its done sinks past both sc0_ep_done and dot_gmm!
    self.assertLess(new_order.index("sc1_ag_start"), new_order.index("dot_attn"))
    self.assertGreater(new_order.index("sc1_ag_done"), new_order.index("dot_gmm"))

  def test_non_collective_vmem_dma_detection_ignores_pipelined_dynamic_slice(self):
    vmem_done = (
        "%dynamic-slice-done.12 = f32[2,64,64,64] async-done(%while.6512#1), "
        'backend_config={"dma_priority":0,"while_loop_async_pipelined_config":{"is_pipelined":true}}'
    )
    sc1_start = (
        "%call-start.2 = ((bf16[448,7168]), bf16[7168,7168]) async-start(%slice.2), "
        'async_execution_thread="sparsecore", backend_config={"sparse_core_config":{"core_ids":["1"]}}'
    )
    self.assertTrue(
        fsdp_prefetch_overlap_transform._is_non_collective_async(  # pylint: disable=protected-access
            "kAsyncDone", vmem_done
        )
    )
    self.assertFalse(
        fsdp_prefetch_overlap_transform._is_non_collective_async(  # pylint: disable=protected-access
            "kAsyncStart", sc1_start
        )
    )

  def test_compute_epochs_hoist_cross_phase_collective_before_post_splash_collective(self):
    # Reproduces F2(k) on SC0: `mb0_dispatch` is ready at `p0` (before `splash_mha`) and consumed at `f3_gmm`
    # (`raw_start + done = 1 + 6 = 7`), while `mb1_out_rs` only becomes ready after `splash_mha` (`raw_start = 3`,
    # `done = 4` -> `sum = 7` or `< 7` with formatting ops). With `use_compute_epochs=True`, `mb0_dispatch_start`
    # issues before `splash_mha` while `mb0_dispatch_done` stays FIFO-ordered before `mb1_out_rs_done`.
    order = (
        "p0",
        "splash_mha",
        "mb1_out_rs_start",
        "mb1_out_rs_done",
        "mb0_dispatch_start",
        "mb0_dispatch_done",
        "pad0",
        "pad1",
        "f3_gmm",
        "root",
    )
    opcodes = {
        "p0": "kParameter",
        "splash_mha": "kCustomCall",
        "mb1_out_rs_start": "kAsyncStart",
        "mb1_out_rs_done": "kAsyncDone",
        "mb0_dispatch_start": "kAsyncStart",
        "mb0_dispatch_done": "kAsyncDone",
        "pad0": "kFusion",
        "pad1": "kFusion",
        "f3_gmm": "kCustomCall",
        "root": "kTuple",
    }
    operands = {
        "p0": (),
        "splash_mha": ("p0",),
        "mb1_out_rs_start": ("splash_mha",),
        "mb1_out_rs_done": ("mb1_out_rs_start",),
        "mb0_dispatch_start": ("p0",),
        "mb0_dispatch_done": ("mb0_dispatch_start",),
        "pad0": ("mb1_out_rs_done",),
        "pad1": ("pad0",),
        "f3_gmm": ("pad1", "mb0_dispatch_done"),
        "root": ("f3_gmm",),
    }
    graph = fsdp_prefetch_overlap_transform.make_schedule_graph(
        order=order,
        opcodes=opcodes,
        operands=operands,
        queues={"mb0_dispatch_start": "sparsecore:0", "mb1_out_rs_start": "sparsecore:0"},
    )
    legacy_order = fsdp_prefetch_overlap_transform.reschedule(graph, use_compute_epochs=False)
    epoch_order = fsdp_prefetch_overlap_transform.reschedule(graph, use_compute_epochs=True)
    fsdp_prefetch_overlap_transform.validate_order(graph, legacy_order)
    fsdp_prefetch_overlap_transform.validate_order(graph, epoch_order)
    self.assertGreater(legacy_order.index("mb0_dispatch_start"), legacy_order.index("splash_mha"))
    self.assertLess(epoch_order.index("mb0_dispatch_start"), epoch_order.index("splash_mha"))
    self.assertGreater(epoch_order.index("mb0_dispatch_done"), epoch_order.index("splash_mha"))
    self.assertLess(epoch_order.index("mb0_dispatch_done"), epoch_order.index("mb1_out_rs_done"))

  def test_compute_epochs_ignore_kfusion_cliffs_within_same_custom_call_gap(self):
    # Reproduces the SC0 gap before `splash_mha_dkv`: 9 non-formatting `kFusion`s precede
    # `dispatch_bwd_start` (consumed after `splash_mha_dkv`), and 1 `kFusion` separates it from
    # `pre_splash_ag_start` (consumed before `splash_mha_dkv`). Because no `kCustomCall` lies
    # between the two starts, `pre_splash_ag_start` must issue first by `done_slot` tie-breaking so
    # `dispatch_bwd_done` can sink past `splash_mha_dkv` without being FIFO-capped before it.
    fusions = tuple(f"fus_{i}" for i in range(9))
    order = (
        "p0",
        "gmm_bwd",
        *fusions,
        "dispatch_bwd_start",
        "dispatch_bwd_done",
        "fus_9",
        "pre_splash_ag_start",
        "pre_splash_ag_done",
        "splash_mha_dkv",
        "post_consumer",
        "root",
    )
    opcodes = {
        "p0": "kParameter",
        "gmm_bwd": "kCustomCall",
        **{f: "kFusion" for f in (*fusions, "fus_9")},
        "dispatch_bwd_start": "kAsyncStart",
        "dispatch_bwd_done": "kAsyncDone",
        "pre_splash_ag_start": "kAsyncStart",
        "pre_splash_ag_done": "kAsyncDone",
        "splash_mha_dkv": "kCustomCall",
        "post_consumer": "kFusion",
        "root": "kTuple",
    }
    operands = {
        "p0": (),
        "gmm_bwd": ("p0",),
        "fus_0": ("gmm_bwd",),
        **{f"fus_{i}": (f"fus_{i - 1}",) for i in range(1, 9)},
        "dispatch_bwd_start": ("fus_8",),
        "dispatch_bwd_done": ("dispatch_bwd_start",),
        "fus_9": ("fus_8",),
        "pre_splash_ag_start": ("fus_9",),
        "pre_splash_ag_done": ("pre_splash_ag_start",),
        "splash_mha_dkv": ("pre_splash_ag_done",),
        "post_consumer": ("splash_mha_dkv", "dispatch_bwd_done"),
        "root": ("post_consumer",),
    }
    graph = fsdp_prefetch_overlap_transform.make_schedule_graph(
        order=order,
        opcodes=opcodes,
        operands=operands,
        queues={"dispatch_bwd_start": "sparsecore:0", "pre_splash_ag_start": "sparsecore:0"},
    )
    epoch_order = fsdp_prefetch_overlap_transform.reschedule(graph, use_compute_epochs=True)
    fsdp_prefetch_overlap_transform.validate_order(graph, epoch_order)
    self.assertLess(epoch_order.index("pre_splash_ag_start"), epoch_order.index("dispatch_bwd_start"))
    self.assertLess(epoch_order.index("dispatch_bwd_start"), epoch_order.index("splash_mha_dkv"))
    self.assertGreater(epoch_order.index("dispatch_bwd_done"), epoch_order.index("splash_mha_dkv"))

  def test_transform_module_skips_post_fusion_pipelined_body_in_legacy_and_epoch_modes(self):
    def _mk_inst(name: str, opcode_name: str, text: str = ""):
      inst = mock.MagicMock()
      inst.name = name
      inst.opcode.name = opcode_name
      inst.operands.return_value = []
      inst.control_predecessors.return_value = []
      inst.to_string.return_value = text
      return inst

    entry_comp = mock.MagicMock()
    entry_comp.name = "main.1"
    p0 = _mk_inst("p0", "kParameter")
    while_op = _mk_inst("while.1", "kWhile", "%while.1 = () while(%p0), body=%pipelined_post_fusion_pipelined_body.2")
    while_op.operands.return_value = [p0]
    entry_comp.instructions.return_value = [p0, while_op]

    pipelined_comp = mock.MagicMock()
    pipelined_comp.name = "pipelined_post_fusion_pipelined_body.2"
    bp0 = _mk_inst("bp0", "kParameter")
    gte = _mk_inst("gte", "kGetTupleElement")
    gte.operands.return_value = [bp0]
    orphan_done = _mk_inst("cloned.call-done.1", "kAsyncDone")
    orphan_done.operands.return_value = [gte]
    next_start = _mk_inst("cloned.call-start.1", "kAsyncStart")
    next_start.operands.return_value = [gte]
    root = _mk_inst("root", "kTuple")
    root.operands.return_value = [orphan_done, next_start]
    pipelined_comp.instructions.return_value = [bp0, gte, orphan_done, next_start, root]

    sched = mock.MagicMock()
    seqs = {"main.1": [p0, while_op], "pipelined_post_fusion_pipelined_body.2": [bp0, gte, orphan_done, next_start, root]}
    sched.sequence.side_effect = lambda c: list(seqs[c.name])

    module = mock.MagicMock()
    module.name = "test_mod"
    module.schedule.return_value = sched
    module.to_string.return_value = "ENTRY %main.1 () -> ()"
    module.make_nonfusion_computations.return_value = [entry_comp, pipelined_comp]
    module.computations.return_value = [entry_comp, pipelined_comp]

    for use_epochs in (False, True):
      rewritten = fsdp_prefetch_overlap_transform.transform_module(module, use_compute_epochs=use_epochs)
      self.assertNotIn("pipelined_post_fusion_pipelined_body.2", rewritten)

  def test_dcn_mesh_axes_identifies_cross_slice_axis_only(self):
    """Verifies `_dcn_mesh_axes` flags only mesh axes whose `slice_index` varies across slices."""

    class _FakeDev:

      def __init__(self, slice_index: int):
        self.slice_index = slice_index

    fake_devices = np.array(
        [[_FakeDev(0), _FakeDev(0)], [_FakeDev(1), _FakeDev(1)]],
        dtype=object,
    )
    fake_mesh = mock.MagicMock()
    fake_mesh.axis_names = ("data", "fsdp")
    fake_mesh.devices = fake_devices
    self.assertEqual(
        batch_split._dcn_mesh_axes(fake_mesh),  # pylint: disable=protected-access
        frozenset({"data"}),
    )


if __name__ == "__main__":
  absltest.main()
