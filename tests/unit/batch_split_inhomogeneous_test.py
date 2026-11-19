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

"""Multi-device CPU parity tests for Phase 2 inhomogeneous batch-split (`batch_split_inhomogeneous`)."""

from __future__ import annotations

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
from maxtext.layers import batch_split_inhomogeneous
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path


def _make_mesh(cfg):
  return maxtext_utils.get_mesh_from_config(cfg)


def _shard_nnx_module(module, mesh):
  graphdef, state = nnx.split(module)
  state = jax.tree.map(jax.device_put, state, sharding.nnx_construct_named_sharding(state, mesh))
  return nnx.merge(graphdef, state)


def _find_by_suffix(tree: dict, suffix: str) -> list:
  found = []
  for k, v in tree.items():
    if k == suffix:
      found.append(v)
    elif isinstance(v, dict):
      found.extend(_find_by_suffix(v, suffix))
  return found


def _make_qwen_hybrid_config(num_blocks: int = 2, cycle: int = 4, model_name: str = "qwen3.5-35b-a3b", **overrides):
  """Builds a tiny Qwen3.5 / Qwen3.8 inhomogeneous MoE config for 8-device (DP=2, FSDP=2, EP=2) CPU tests."""
  num_layers = num_blocks * cycle
  defaults = {
      "model_name": model_name,
      "override_model_config": True,
      "run_name": "batch_split_inhom_qwen_test",
      "base_emb_dim": 32,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 1,
      "head_dim": 16,
      "vocab_size": 64,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "inhomogeneous_layer_cycle_interval": cycle,
      "gdn_key_head_dim": 8,
      "gdn_value_head_dim": 8,
      "gdn_num_key_heads": 2,
      "gdn_num_value_heads": 4,
      "gdn_chunk_size": 4,
      "base_num_decoder_layers": num_layers,
      "num_decoder_layers": num_layers,
      "max_target_length": 8,
      "per_device_batch_size": 2.0,
      "ici_data_parallelism": 2,
      "ici_fsdp_parallelism": 2,
      "ici_expert_parallelism": 2,
      "ici_tensor_parallelism": 1,
      "use_mrope": False,
      "scan_layers": True,
      "sparse_matmul": True,
      "use_ring_of_experts": True,
      "use_ragged_sort": False,
      "megablox": False,
      "use_tokamax_gmm": False,
      "capacity_factor": -1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "remat_policy": "full",
      "dropout_rate": 0.0,
      "load_balance_loss_weight": 0.0,
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


def _make_deepseek4_config(num_blocks: int = 2, first_num_hash_layers: int = 0, **overrides):
  """Builds a tiny DeepSeek-V4 inhomogeneous MoE config (`[HCA, CSA]` + 4D mHC) for 8-device CPU tests."""
  total_layers = first_num_hash_layers + 2 * num_blocks
  prefix_ratios = [0] * max(first_num_hash_layers - 1, 0) + ([4] if first_num_hash_layers >= 1 else [])
  compress_ratios = prefix_ratios + [128, 4] * num_blocks
  defaults = {
      "model_name": "deepseek4-tiny",
      "decoder_block": "deepseek4",
      "override_model_config": True,
      "run_name": "batch_split_inhom_dsv4_test",
      "base_emb_dim": 64,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 64,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 1,
      "head_dim": 32,
      "qk_rope_head_dim": 32,
      "q_lora_rank": 16,
      "o_groups": 4,
      "o_lora_rank": 16,
      "sliding_window_size": 8,
      "mhc_expansion_rate": 4,
      "first_num_hash_layers": first_num_hash_layers,
      "base_num_decoder_layers": total_layers,
      "num_decoder_layers": total_layers,
      "compress_ratios": compress_ratios,
      "use_indexer": False,
      "indexer_topk": 4,
      "vocab_size": 128,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "routed_score_func": "sqrtsoftplus",
      "moe_use_megatron_seq_aux_loss": True,
      "routed_bias": True,
      "routed_bias_update_rate": 0.0,
      "load_balance_loss_weight": 0.0,
      "max_target_length": 16,
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
      "enable_batch_split": False,
      "prefetch_fsdp_weights": False,
      "prefetch_fsdp_defer_grad_reduce_scatter": False,
      "shard_mode": "auto",
      "param_scan_axis": 1,
  }
  defaults.update(overrides)
  args = ["", get_test_config_path()]
  for k, v in defaults.items():
    args.append(f"{k}={v}")
  return pyconfig.initialize(args)


def _make_gemma4_config(num_blocks: int = 2, model_name: str = "gemma4-26b", **overrides):
  """Builds a tiny Gemma4 `[5 LOCAL_SLIDING + 1 GLOBAL]` config for 8-device (DP=2, FSDP=2, EP=2) CPU tests."""
  num_layers = 6 * num_blocks
  defaults = {
      "model_name": model_name,
      "decoder_block": "gemma4",
      "override_model_config": True,
      "run_name": "batch_split_inhom_gemma4_test",
      "base_emb_dim": 32,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 8,
      "global_head_dim": 16,
      "global_num_kv_heads": 2,
      "vocab_size": 64,
      "sliding_window_size": 4,
      "use_post_attn_norm": True,
      "use_post_ffw_norm": True,
      "share_kv_projections": True,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "norm_topk_prob": True,
      "load_balance_loss_weight": 0.0,
      "base_num_decoder_layers": num_layers,
      "num_decoder_layers": num_layers,
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
      "capacity_factor": -1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "remat_policy": "full",
      "dropout_rate": 0.0,
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


def _make_gpt_oss_config(num_blocks: int = 2, **overrides):
  """Builds a tiny GPT-OSS `[1 LOCAL_SLIDING + 1 GLOBAL]` config for 8-device (DP=2, FSDP=2, EP=2) CPU tests."""
  num_layers = 2 * num_blocks
  defaults = {
      "model_name": "gpt-oss-20b",
      "decoder_block": "gpt_oss",
      "override_model_config": True,
      "run_name": "batch_split_inhom_gpt_oss_test",
      "base_emb_dim": 32,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 8,
      "sliding_window_size": 4,
      "attention_bias": True,
      "attention_sink": True,
      "mlp_activations": ["sigmoid", "linear"],
      "mlp_activations_limit": 7.0,
      "routed_bias": True,
      "routed_bias_update_rate": 0.0,
      "mlp_bias": True,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "shared_experts": 0,
      "inhomogeneous_layer_cycle_interval": 2,
      "load_balance_loss_weight": 0.0,
      "vocab_size": 64,
      "base_num_decoder_layers": num_layers,
      "num_decoder_layers": num_layers,
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
      "capacity_factor": -1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "remat_policy": "full",
      "dropout_rate": 0.0,
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


def _run_loss_and_grads(cfg):
  """Runs a jitted forward+backward pass on NNXDecoder and returns `(loss, grads, inter_dict)`."""
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
      inter_dict = inter.to_pure_dict() if len(inter) > 0 else {}
      lb_vals = _find_by_suffix(inter_dict, "moe_lb_loss")
      if lb_vals:
        lb_stacked = jnp.concatenate([jnp.ravel(jnp.asarray(v[0] if isinstance(v, tuple) else v)) for v in lb_vals])
        use_sum = cfg.routed_score_func == "sigmoid" or getattr(cfg, "moe_use_megatron_seq_aux_loss", False)
        loss = loss + (jnp.sum(lb_stacked) if use_sum else jnp.mean(lb_stacked))
      return loss, inter_dict

    (loss, inter_dict), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
    return np.asarray(loss), grads, inter_dict


class BatchSplitInhomogeneousParityTest(parameterized.TestCase):
  """Verifies Phase 2 nested-scan inhomogeneous batch-split on Qwen3.5/3.8 and DeepSeek-V4."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 8:
      self.skipTest(f"Requires 8 simulated CPU devices, got {jax.device_count()}")

  def _assert_cfg_parity(
      self,
      ref_cfg,
      bs_cfg,
      *,
      expected_sizes: tuple[int, ...] | None = None,
      expected_length: int | None = None,
      rtol: float = 1e-2,
      atol: float = 1e-3,
  ):
    """Asserts loss, gradient, and aux metric parity for inhomogeneous batch-split configs."""
    ref_loss, ref_grads, ref_inter = _run_loss_and_grads(ref_cfg)
    with mock.patch.object(
        batch_split_inhomogeneous,
        "_custom_vjp_nested_batch_split",
        wraps=batch_split_inhomogeneous._custom_vjp_nested_batch_split,  # pylint: disable=protected-access  # pylint: disable=protected-access
    ) as nested_spy:
      bs_loss, bs_grads, bs_inter = _run_loss_and_grads(bs_cfg)

    if expected_sizes is not None:
      self.assertEqual(nested_spy.call_count, 1)
      self.assertEqual(nested_spy.call_args.args[11], expected_sizes)
    if expected_length is not None:
      self.assertEqual(nested_spy.call_args.args[9], expected_length)

    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)

    ref_items = jax.tree_util.tree_leaves_with_path(ref_grads)
    bs_items = jax.tree_util.tree_leaves_with_path(bs_grads)
    self.assertEqual(len(ref_items), len(bs_items))
    for (p, r), (_, b) in zip(ref_items, bs_items):
      self.assertEqual(r.shape, b.shape, f"Grad shape mismatch at {jax.tree_util.keystr(p)}")
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
        self.assertEqual(r_arr.shape, b_arr.shape)
        np.testing.assert_allclose(b_arr, r_arr, rtol=1e-5, atol=1e-6)

    if bs_cfg.routed_bias and bs_cfg.routed_bias_update_rate > 0.0:
      ref_bu = _find_by_suffix(ref_inter, "moe_bias_updates")
      bs_bu = _find_by_suffix(bs_inter, "moe_bias_updates")
      self.assertEqual(len(ref_bu), len(bs_bu))
      self.assertNotEmpty(bs_bu)
      for r, b in zip(ref_bu, bs_bu):
        r_arr = np.asarray(r[0] if isinstance(r, tuple) else r)
        b_arr = np.asarray(b[0] if isinstance(b, tuple) else b)
        self.assertEqual(r_arr.shape, b_arr.shape)
        np.testing.assert_allclose(b_arr, r_arr, rtol=0.0, atol=1e-6)

  def test_qwen3_5_explicit_prefetch_and_deferred_rs_1_block(self):
    common = {
        "num_blocks": 1,
        "cycle": 4,
        "shard_mode": "explicit",
        "prefetch_fsdp_weights": True,
        "prefetch_fsdp_defer_grad_reduce_scatter": True,
    }
    self._assert_cfg_parity(
        _make_qwen_hybrid_config(enable_batch_split=False, **common),
        _make_qwen_hybrid_config(enable_batch_split=True, **common),
        expected_sizes=(3, 1),
        expected_length=1,
    )

  def test_qwen3_5_explicit_prefetch_deferred_rs_and_lb_loss_2_blocks(self):
    # Reference uses plain scan (prefetch=False) with load_balance_loss_weight > 0 so sown moe_lb_loss is compared directly.
    ref_cfg = _make_qwen_hybrid_config(
        num_blocks=2,
        cycle=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        load_balance_loss_weight=0.05,
        enable_batch_split=False,
    )
    bs_cfg = _make_qwen_hybrid_config(
        num_blocks=2,
        cycle=4,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        load_balance_loss_weight=0.05,
        enable_batch_split=True,
    )
    self._assert_cfg_parity(ref_cfg, bs_cfg, expected_sizes=(3, 1), expected_length=2)

  @parameterized.named_parameters(
      ("cycle_2_single_gdn", 2, (1, 1)),
      ("cycle_3_two_gdn_no_mid", 3, (2, 1)),
      ("cycle_5_four_gdn_unrolled_mid", 5, (4, 1)),
  )
  def test_qwen3_5_sublayer_group_sizes(self, cycle: int, expected_sizes: tuple[int, int]):
    common = {
        "num_blocks": 2,
        "cycle": cycle,
        "shard_mode": "explicit",
        "prefetch_fsdp_weights": True,
        "prefetch_fsdp_defer_grad_reduce_scatter": True,
    }
    self._assert_cfg_parity(
        _make_qwen_hybrid_config(enable_batch_split=False, **common),
        _make_qwen_hybrid_config(enable_batch_split=True, **common),
        expected_sizes=expected_sizes,
        expected_length=2,
    )

  def test_qwen3_5_without_shared_expert_and_param_scan_axis_zero(self):
    common = {
        "num_blocks": 2,
        "cycle": 4,
        "shared_experts": 0,
        "param_scan_axis": 0,
        "shard_mode": "explicit",
        "prefetch_fsdp_weights": True,
        "prefetch_fsdp_defer_grad_reduce_scatter": True,
    }
    self._assert_cfg_parity(
        _make_qwen_hybrid_config(enable_batch_split=False, **common),
        _make_qwen_hybrid_config(enable_batch_split=True, **common),
        expected_sizes=(3, 1),
        expected_length=2,
    )

  def test_qwen3_8_recipe_parity(self):
    common = {
        "num_blocks": 2,
        "cycle": 4,
        "model_name": "qwen3.8-2.4t-a95b",
        "shard_mode": "explicit",
        "prefetch_fsdp_weights": True,
        "prefetch_fsdp_defer_grad_reduce_scatter": True,
    }
    self._assert_cfg_parity(
        _make_qwen_hybrid_config(enable_batch_split=False, **common),
        _make_qwen_hybrid_config(enable_batch_split=True, **common),
        expected_sizes=(3, 1),
        expected_length=2,
    )

  def test_qwen3_5_dense_falls_back_cleanly(self):
    common = {
        "num_blocks": 1,
        "cycle": 4,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "sparse_matmul": False,
        "shard_mode": "explicit",
        "prefetch_fsdp_weights": True,
    }
    ref_cfg = _make_qwen_hybrid_config(enable_batch_split=False, **common)
    bs_cfg = _make_qwen_hybrid_config(enable_batch_split=True, **common)
    ref_loss, ref_grads, _ = _run_loss_and_grads(ref_cfg)
    with mock.patch.object(
        batch_split_inhomogeneous,
        "_custom_vjp_nested_batch_split",
        wraps=batch_split_inhomogeneous._custom_vjp_nested_batch_split,  # pylint: disable=protected-access
    ) as nested_spy:
      bs_loss, bs_grads, _ = _run_loss_and_grads(bs_cfg)
    self.assertEqual(nested_spy.call_count, 0)
    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)
    for r, b in zip(jax.tree_util.tree_leaves(ref_grads), jax.tree_util.tree_leaves(bs_grads)):
      np.testing.assert_allclose(np.asarray(b), np.asarray(r), rtol=2e-4, atol=2e-5)

  def test_deepseek4_mhc_lb_loss_and_routed_bias_updates(self):
    common = {
        "num_blocks": 2,
        "first_num_hash_layers": 0,
        "mhc_expansion_rate": 4,
        "load_balance_loss_weight": 0.05,
        "routed_bias": True,
        "routed_bias_update_rate": 0.01,
        "shard_mode": "auto",
        "prefetch_fsdp_weights": False,
    }
    self._assert_cfg_parity(
        _make_deepseek4_config(enable_batch_split=False, **common),
        _make_deepseek4_config(enable_batch_split=True, **common),
        expected_sizes=(1, 1),
        expected_length=2,
        rtol=5e-4,
        atol=5e-5,
    )

  def test_deepseek4_with_fsdp_prefetch_deferred_rs_and_indexer(self):
    ref_cfg = _make_deepseek4_config(
        num_blocks=2,
        first_num_hash_layers=0,
        mhc_expansion_rate=4,
        use_indexer=True,
        indexer_loss_scaling_factor=0.0,
        shard_mode="auto",
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        enable_batch_split=False,
    )
    bs_cfg = _make_deepseek4_config(
        num_blocks=2,
        first_num_hash_layers=0,
        mhc_expansion_rate=4,
        use_indexer=True,
        indexer_loss_scaling_factor=0.0,
        shard_mode="auto",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        enable_batch_split=True,
    )
    self._assert_cfg_parity(
        ref_cfg,
        bs_cfg,
        expected_sizes=(1, 1),
        expected_length=2,
        rtol=5e-4,
        atol=5e-5,
    )

  def test_gemma4_26b_moe_explicit_prefetch_deferred_rs_and_lb_loss(self):
    ref_cfg = _make_gemma4_config(
        num_blocks=2,
        load_balance_loss_weight=0.05,
        moe_topk_before_ep_all_gather=True,
        shard_mode="explicit",
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        enable_batch_split=False,
    )
    bs_cfg = _make_gemma4_config(
        num_blocks=2,
        load_balance_loss_weight=0.05,
        moe_topk_before_ep_all_gather=True,
        shard_mode="explicit",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        enable_batch_split=True,
    )
    self._assert_cfg_parity(
        ref_cfg,
        bs_cfg,
        expected_sizes=(5, 1),
        expected_length=2,
        rtol=5e-4,
        atol=5e-5,
    )

  def test_gemma4_26b_single_block_bf16(self):
    common = {
        "num_blocks": 1,
        "dtype": "bfloat16",
        "base_emb_dim": 64,
        "base_mlp_dim": 32,
        "base_moe_mlp_dim": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "load_balance_loss_weight": 0.01,
        "moe_topk_before_ep_all_gather": True,
        "moe_use_direct_token_gather": True,
        "shard_mode": "explicit",
    }
    ref_cfg = _make_gemma4_config(
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        enable_batch_split=False,
        **common,
    )
    bs_cfg = _make_gemma4_config(
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        enable_batch_split=True,
        **common,
    )
    self._assert_cfg_parity(
        ref_cfg,
        bs_cfg,
        expected_sizes=(5, 1),
        expected_length=1,
        rtol=5e-2,
        atol=2e-2,
    )

  def test_gpt_oss_20b_explicit_prefetch_deferred_rs_and_lb_loss(self):
    ref_cfg = _make_gpt_oss_config(
        num_blocks=2,
        load_balance_loss_weight=0.05,
        routed_bias=True,
        shard_mode="auto",
        prefetch_fsdp_weights=False,
        prefetch_fsdp_defer_grad_reduce_scatter=False,
        enable_batch_split=False,
    )
    bs_cfg = _make_gpt_oss_config(
        num_blocks=2,
        load_balance_loss_weight=0.05,
        routed_bias=True,
        shard_mode="auto",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
        enable_batch_split=True,
    )
    self._assert_cfg_parity(
        ref_cfg,
        bs_cfg,
        expected_sizes=(1, 1),
        expected_length=2,
        rtol=5e-4,
        atol=5e-5,
    )

  def test_gemma4_31b_dense_falls_back_cleanly(self):
    common = {
        "num_blocks": 1,
        "model_name": "gemma4-31b",
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "sparse_matmul": False,
        "shard_mode": "explicit",
        "prefetch_fsdp_weights": True,
    }
    ref_cfg = _make_gemma4_config(enable_batch_split=False, **common)
    bs_cfg = _make_gemma4_config(enable_batch_split=True, **common)
    ref_loss, ref_grads, _ = _run_loss_and_grads(ref_cfg)
    with mock.patch.object(
        batch_split_inhomogeneous,
        "_custom_vjp_nested_batch_split",
        wraps=batch_split_inhomogeneous._custom_vjp_nested_batch_split,  # pylint: disable=protected-access
    ) as nested_spy:
      bs_loss, bs_grads, _ = _run_loss_and_grads(bs_cfg)
    self.assertEqual(nested_spy.call_count, 0)
    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)
    for r, b in zip(jax.tree_util.tree_leaves(ref_grads), jax.tree_util.tree_leaves(bs_grads)):
      np.testing.assert_allclose(np.asarray(b), np.asarray(r), rtol=2e-4, atol=2e-5)

  def test_llama4_interleaved_dense_moe_falls_back_cleanly(self):
    def _llama4_cfg(enable_bs: bool):
      args = [
          "",
          get_test_config_path(),
          "model_name=llama4-17b-16e",
          "decoder_block=llama4",
          "override_model_config=True",
          "run_name=batch_split_llama4_fallback_test",
          "base_emb_dim=32",
          "base_mlp_dim=32",
          "base_moe_mlp_dim=32",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=8",
          "vocab_size=64",
          "num_experts=4",
          "num_experts_per_tok=1",
          "shared_experts=1",
          "use_qk_norm=True",
          "nope_layer_interval=4",
          "interleave_moe_layer_step=2",
          "inhomogeneous_layer_cycle_interval=4",
          "chunk_attn_window_size=8",
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
          "load_balance_loss_weight=0.0",
          "shard_mode=explicit",
          "prefetch_fsdp_weights=False",
          f"enable_batch_split={enable_bs}",
      ]
      return pyconfig.initialize(args)

    ref_loss, ref_grads, _ = _run_loss_and_grads(_llama4_cfg(False))
    with mock.patch.object(
        batch_split_inhomogeneous,
        "_custom_vjp_nested_batch_split",
        wraps=batch_split_inhomogeneous._custom_vjp_nested_batch_split,  # pylint: disable=protected-access
    ) as nested_spy:
      bs_loss, bs_grads, _ = _run_loss_and_grads(_llama4_cfg(True))
    self.assertEqual(nested_spy.call_count, 0)
    np.testing.assert_allclose(bs_loss, ref_loss, rtol=1e-5, atol=1e-6)
    for r, b in zip(jax.tree_util.tree_leaves(ref_grads), jax.tree_util.tree_leaves(bs_grads)):
      np.testing.assert_allclose(np.asarray(b), np.asarray(r), rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
  absltest.main()
