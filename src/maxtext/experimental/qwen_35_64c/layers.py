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

"""Pure-JAX Qwen3.5 layers: Gated DeltaNet, Splash Attention, Shared & Routed MoE."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas.ops.tpu import splash_attention
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.qwen_35_64c import moe_kernels
from maxtext.experimental.qwen_35_64c import ops
from maxtext.experimental.qwen_35_64c.config import Qwen35Config
from maxtext.experimental.qwen_35_64c.ext_kernels.gdn.gdn_bwd_pallas import gdn_decoupled_conv1d
from maxtext.experimental.qwen_35_64c.ext_kernels.ragged import ragged_gather_tc as tc
from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import splash_attention_kernel as tokamax_splash
from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import splash_attention_mask as tokamax_splash_mask
from maxtext.experimental.qwen_35_64c.ext_kernels.splash_attention import splash_attention_mask_info as tokamax_splash_mask_info
from maxtext.experimental.qwen_35_64c.params import AttnSublayerParams, GdnSublayerParams, MoeParams

# ---------------------------------------------------------------------------
# Gathered per-sublayer weights (produced via SC1 `fsdp` all-gathers)
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredMoeWeights:
  router_gate: jax.Array
  wi: jax.Array | moe_kernels.Fp8SteWeight
  wo: jax.Array | moe_kernels.Fp8SteWeight
  shared_wi_0: jax.Array
  shared_wi_1: jax.Array
  shared_wo: jax.Array
  shared_gate: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredGdnWeights:
  """All-gathered weights for a single GDN sublayer."""

  pre_attn_norm: jax.Array
  post_attn_norm: jax.Array
  in_proj_qkvz: jax.Array
  in_proj_ba: jax.Array
  conv1d: jax.Array
  A_log: jax.Array
  dt_bias: jax.Array
  gdn_norm: jax.Array
  out_proj: jax.Array
  moe: GatheredMoeWeights
  in_proj_qkv: jax.Array | None = None
  in_proj_z: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredAttnWeights:
  """All-gathered weights for a single full-attention sublayer."""

  pre_attn_norm: jax.Array
  post_attn_norm: jax.Array
  q_proj: jax.Array
  k_proj: jax.Array
  v_proj: jax.Array
  q_norm: jax.Array
  k_norm: jax.Array
  out_proj: jax.Array
  moe: GatheredMoeWeights
  w_q_grouped: jax.Array | None = None
  w_gate_grouped: jax.Array | None = None
  w_out_grouped: jax.Array | None = None


def split_gdn_qkvz_weight(in_proj_qkvz: jax.Array, cfg: Qwen35Config) -> tuple[jax.Array | None, jax.Array | None]:
  """Pre-slices gathered `[D, qkvz_dim]` GDN weight once per sublayer."""
  if not cfg.gdn_split_qkvz_proj:
    return None, None
  return (
      in_proj_qkvz[:, : cfg.gdn_qkv_dim],
      in_proj_qkvz[:, cfg.gdn_qkv_dim :],
  )


def group_attn_q_and_out_weights(
    q_proj: jax.Array, out_proj: jax.Array, cfg: Qwen35Config
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
  """Pre-groups `q_proj` and `out_proj` into `[..., K, G, D_h]` once per sublayer."""
  if not (cfg.attn_kv_grouped_layout and cfg.sa_fold_gqa_into_seq and not cfg.attn_qkv_2d_matmul):
    return None, None, None
  gqa_ratio = cfg.num_query_heads // cfg.num_kv_heads
  q_grouped = q_proj.reshape(cfg.emb_dim, cfg.num_kv_heads, gqa_ratio, 2 * cfg.head_dim)
  w_q = q_grouped[..., : cfg.head_dim]
  w_gate = q_grouped[..., cfg.head_dim :]
  w_out = out_proj.reshape(cfg.num_kv_heads, gqa_ratio, cfg.head_dim, cfg.emb_dim)
  return w_q, w_gate, w_out


class DispatchedMoeTokens(NamedTuple):
  tokens_ep: jax.Array | ops.Fp8DispatchedTokens  # [T_ep, D] or 3D [T_ep, D//128, 128]
  topk_weights_ep: jax.Array  # [T_ep, K] fp32
  topk_ids_ep: jax.Array  # [T_ep, K] s32


def order_dispatched_after(
    disp: DispatchedMoeTokens,
    deps: jax.Array | ops.Fp8DispatchedTokens | tuple[object, ...],
) -> DispatchedMoeTokens:
  """Attaches forward-only ordering tie(s) onto `disp.tokens_ep` (4B tie when FP8!)."""
  dep_tuple = deps if isinstance(deps, tuple) else (deps,)
  tok_ep = disp.tokens_ep
  for dep in dep_tuple:
    tok_ep = ops.order_after(tok_ep, dep)
  return DispatchedMoeTokens(
      tokens_ep=tok_ep,
      topk_weights_ep=disp.topk_weights_ep,
      topk_ids_ep=disp.topk_ids_ep,
  )


def gather_moe_weights(
    p: MoeParams,
    wi_absmax: jax.Array | None,
    wo_absmax: jax.Array | None,
    cfg: Qwen35Config,
) -> GatheredMoeWeights:
  """Gathers MoE + Shared Expert weights on SC1 (`fsdp`)."""
  return GatheredMoeWeights(
      router_gate=ops.gather_or_replicate_dense_weight(p.router_gate, cfg, param_name="router_gate", sharded_axis=0),
      wi=moe_kernels.gather_routed_weight_sc1(p.wi, wi_absmax, cfg, is_wi=True),
      wo=moe_kernels.gather_routed_weight_sc1(p.wo, wo_absmax, cfg, is_wi=False),
      shared_wi_0=ops.gather_or_replicate_dense_weight(p.shared_wi_0, cfg, param_name="shared_wi", sharded_axis=0),
      shared_wi_1=ops.gather_or_replicate_dense_weight(p.shared_wi_1, cfg, param_name="shared_wi", sharded_axis=0),
      shared_wo=ops.gather_or_replicate_dense_weight(p.shared_wo, cfg, param_name="shared_wo", sharded_axis=1),
      shared_gate=ops.gather_or_replicate_dense_weight(p.shared_gate, cfg, param_name="shared_gate", sharded_axis=0),
  )


def gather_gdn_sublayer_weights(
    p: GdnSublayerParams,
    wi_absmax: jax.Array | None,
    wo_absmax: jax.Array | None,
    cfg: Qwen35Config,
) -> GatheredGdnWeights:
  """Gathers all GDN + MoE weights of one sub-layer on SC1 (`fsdp`)."""
  in_proj_qkvz = ops.gather_or_replicate_dense_weight(p.in_proj_qkvz, cfg, param_name="in_proj_qkvz", sharded_axis=0)
  in_proj_qkv, in_proj_z = split_gdn_qkvz_weight(in_proj_qkvz, cfg)
  return GatheredGdnWeights(
      pre_attn_norm=ops.replicate_small_param(p.pre_attn_norm, cfg),
      post_attn_norm=ops.replicate_small_param(p.post_attn_norm, cfg),
      in_proj_qkvz=in_proj_qkvz,
      in_proj_ba=ops.gather_or_replicate_dense_weight(p.in_proj_ba, cfg, param_name="in_proj_ba", sharded_axis=0),
      conv1d=ops.replicate_small_param(p.conv1d, cfg),
      A_log=ops.replicate_small_param(p.A_log, cfg),
      dt_bias=ops.replicate_small_param(p.dt_bias, cfg),
      gdn_norm=ops.replicate_small_param(p.gdn_norm, cfg),
      out_proj=ops.gather_or_replicate_dense_weight(p.out_proj, cfg, param_name="gdn_out_proj", sharded_axis=0),
      moe=gather_moe_weights(p.moe, wi_absmax, wo_absmax, cfg),
      in_proj_qkv=in_proj_qkv,
      in_proj_z=in_proj_z,
  )


def gather_attn_sublayer_weights(
    p: AttnSublayerParams,
    wi_absmax: jax.Array | None,
    wo_absmax: jax.Array | None,
    cfg: Qwen35Config,
) -> GatheredAttnWeights:
  """Gathers all Full-Attention + MoE weights of one sub-layer on SC1 (`fsdp`)."""
  q_in = p.q_proj.reshape(p.q_proj.shape[0], -1) if cfg.attn_qkv_2d_matmul else p.q_proj
  k_in = p.k_proj.reshape(p.k_proj.shape[0], -1) if cfg.attn_qkv_2d_matmul else p.k_proj
  v_in = p.v_proj.reshape(p.v_proj.shape[0], -1) if cfg.attn_qkv_2d_matmul else p.v_proj
  q_proj = ops.gather_or_replicate_dense_weight(q_in, cfg, param_name="q_proj", sharded_axis=0)
  out_proj = ops.gather_or_replicate_dense_weight(p.out_proj, cfg, param_name="attn_out_proj", sharded_axis=0)
  w_q_g, w_gate_g, w_out_g = group_attn_q_and_out_weights(q_proj, out_proj, cfg)
  return GatheredAttnWeights(
      pre_attn_norm=ops.replicate_small_param(p.pre_attn_norm, cfg),
      post_attn_norm=ops.replicate_small_param(p.post_attn_norm, cfg),
      q_proj=q_proj,
      k_proj=ops.gather_or_replicate_dense_weight(k_in, cfg, param_name="kv_proj", sharded_axis=0),
      v_proj=ops.gather_or_replicate_dense_weight(v_in, cfg, param_name="kv_proj", sharded_axis=0),
      q_norm=ops.replicate_small_param(p.q_norm, cfg),
      k_norm=ops.replicate_small_param(p.k_norm, cfg),
      out_proj=out_proj,
      moe=gather_moe_weights(p.moe, wi_absmax, wo_absmax, cfg),
      w_q_grouped=w_q_g,
      w_gate_grouped=w_gate_g,
      w_out_grouped=w_out_g,
  )


# ---------------------------------------------------------------------------
# Context parallelism: dual-chunk load-balanced attention layout
# ---------------------------------------------------------------------------
#
# Activations are sequence-sharded over `cfg.cp_axis_names` in NATURAL order:
# CP rank `r` of `n` holds chunks `[2r, 2r + 1]` of the `2n` equal chunks of the
# sequence (contiguous, as GDN state passing requires). Causal attention runs in
# LOAD-BALANCED order instead: rank `d` holds chunks `[d, 2n - 1 - d]`, so every
# rank does the same causal work.


def _lb_chunk_dest(j: int, n: int) -> int:
  """CP rank holding chunk `j` (of `2n`) in load-balanced order."""
  return j if j < n else 2 * n - 1 - j


def cp_natural_to_lb(x: jax.Array, cfg: Qwen35Config, axis: int = 1) -> jax.Array:
  """Permutes a natural-order local sequence shard into load-balanced order."""
  n = cfg.cp_size
  axes = cfg.cp_axis_names
  half = x.shape[axis] // 2
  h0 = jax.lax.slice_in_dim(x, 0, half, axis=axis)  # chunk 2r
  h1 = jax.lax.slice_in_dim(x, half, 2 * half, axis=axis)  # chunk 2r + 1
  recv0 = jax.lax.ppermute(h0, axes, [(r, _lb_chunk_dest(2 * r, n)) for r in range(n)])
  recv1 = jax.lax.ppermute(h1, axes, [(r, _lb_chunk_dest(2 * r + 1, n)) for r in range(n)])
  # Rank d receives chunk d (slot 0) and chunk 2n-1-d (slot 1); for even d chunk d
  # is even (arrives via recv0), for odd d it is odd (arrives via recv1).
  even = jax.lax.axis_index(axes) % 2 == 0
  first = jnp.where(even, recv0, recv1)
  second = jnp.where(even, recv1, recv0)
  return jnp.concatenate([first, second], axis=axis)


def cp_lb_to_natural(x: jax.Array, cfg: Qwen35Config, axis: int = 1) -> jax.Array:
  """Inverse of `cp_natural_to_lb`."""
  n = cfg.cp_size
  axes = cfg.cp_axis_names
  half = x.shape[axis] // 2
  slot0 = jax.lax.slice_in_dim(x, 0, half, axis=axis)  # chunk d
  slot1 = jax.lax.slice_in_dim(x, half, 2 * half, axis=axis)  # chunk 2n-1-d
  even = jax.lax.axis_index(axes) % 2 == 0
  even_chunk = jnp.where(even, slot0, slot1)
  odd_chunk = jnp.where(even, slot1, slot0)
  recv_even = jax.lax.ppermute(even_chunk, axes, [(_lb_chunk_dest(2 * r, n), r) for r in range(n)])
  recv_odd = jax.lax.ppermute(odd_chunk, axes, [(_lb_chunk_dest(2 * r + 1, n), r) for r in range(n)])
  return jnp.concatenate([recv_even, recv_odd], axis=axis)


def cp_all_gather_lb_to_natural(x_lb: jax.Array, cfg: Qwen35Config, axis: int = 1) -> jax.Array:
  """All-gathers a load-balanced local shard over CP into the full natural-order sequence."""
  n = cfg.cp_size
  half = x_lb.shape[axis] // 2
  g = jax.lax.all_gather(x_lb, cfg.cp_axis_names, axis=0, tiled=False)  # [n, ...]
  g0 = jax.lax.slice_in_dim(g, 0, half, axis=axis + 1)  # chunk d from rank d
  g1 = jax.lax.slice_in_dim(g, half, 2 * half, axis=axis + 1)  # chunk 2n-1-d from rank d
  chunks = jnp.concatenate([g0, jnp.flip(g1, axis=0)], axis=0)  # [2n, ...] natural chunk order
  chunks = jnp.moveaxis(chunks, 0, axis)
  shape = list(x_lb.shape)
  shape[axis] = 2 * n * half
  return chunks.reshape(shape)


def cp_lb_global_index(local_len: int, cfg: Qwen35Config) -> jax.Array:
  """Global sequence index of each local load-balanced position on this CP rank."""
  n = cfg.cp_size
  half = local_len // 2
  d = jax.lax.axis_index(cfg.cp_axis_names)
  ar = jnp.arange(half, dtype=jnp.int32)
  return jnp.concatenate([d * half + ar, (2 * n - 1 - d) * half + ar])


def _lb_q_sequence(seq_len: int, n: int) -> np.ndarray:
  half = seq_len // (2 * n)
  ar = np.arange(half, dtype=np.int32)
  return np.concatenate([np.concatenate([d * half + ar, (2 * n - 1 - d) * half + ar]) for d in range(n)])


@functools.lru_cache(maxsize=16)
def _fast_causal_tokamax_mask_info(
    causal_mask: tokamax_splash_mask.CausalMask,
    block_shape: tuple[int, int],
    *,
    is_dkv: bool,
    return_dynamic_grid: bool,
    q_seq_shards: int,
) -> tuple[tokamax_splash_mask_info.MaskInfo, object]:
  """Vectorized `process_mask` for `CausalMask` with `q_sequence` (<1 ms vs ~15s at 1M rows)."""
  q_seq_len, kv_seq_len = causal_mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  if q_mod != 0:
    raise ValueError(f"{q_block_size=} should divide {q_seq_len=}.")
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)
  if kv_mod != 0:
    raise ValueError(f"{kv_block_size=} should divide {kv_seq_len=}.")
  q_seq_len_per_shard, qs_mod = divmod(q_seq_len, q_seq_shards)
  if qs_mod != 0:
    raise ValueError(f"{q_seq_shards=} should divide {q_seq_len=}.")
  q_blocks_per_shard, qbs_mod = divmod(q_seq_len_per_shard, q_block_size)
  if qbs_mod != 0:
    raise ValueError(f"{q_block_size=} should divide {q_seq_len_per_shard=}.")

  q_sequence = causal_mask.q_sequence
  mask_function = causal_mask.mask_function
  q_blocks_2d = q_sequence.reshape(q_blocks_count, q_block_size)
  q_min = q_blocks_2d.min(axis=1)[:, None] + causal_mask.offset
  q_max = q_blocks_2d.max(axis=1)[:, None] + causal_mask.offset
  kv_idx = np.arange(kv_blocks_count, dtype=np.int64)[None, :]
  kv_min = kv_idx * kv_block_size
  kv_max = (kv_idx + 1) * kv_block_size - 1

  state_grid = np.ones((q_blocks_count, kv_blocks_count), dtype=np.int32)
  state_grid[q_max < kv_min] = 0
  state_grid[q_min >= kv_max] = 2

  if (state_grid == 2).all():
    return (
        tokamax_splash_mask_info.MaskInfo(
            mask_next=None,
            active_rows=None,
            active_cols=None,
            block_mask=None,
            num_active_blocks=None,
            partial_mask_blocks=None,
            q_sequence=q_sequence,
        ),
        None,
    )

  partial_id_grid = np.full((q_blocks_count, kv_blocks_count), -1, dtype=np.int32)
  all_shards_metadata = []
  for q_shard_idx in range(q_seq_shards):
    q_slice = slice(
        q_shard_idx * q_blocks_per_shard,
        (q_shard_idx + 1) * q_blocks_per_shard,
    )
    all_shards_metadata.append(
        tokamax_splash_mask_info._generate_shard_metadata(  # pylint: disable=protected-access
            state_grid[q_slice, :],
            partial_id_grid[q_slice, :],
            is_dkv,
            return_dynamic_grid,
        )
    )

  (
      active_rows_slices,
      active_cols_slices,
      _,
      block_mask_slices,
      num_active_blocks,
  ) = zip(*all_shards_metadata)

  if return_dynamic_grid:
    max_size = max(num_active_blocks)
    pad_slice = lambda arr: np.pad(arr, (0, max_size - arr.shape[0]), mode="constant", constant_values=-1)
    active_rows = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
        np.concatenate(list(map(pad_slice, active_rows_slices)), axis=0)
    )
    active_cols = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
        np.concatenate(list(map(pad_slice, active_cols_slices)), axis=0)
    )
    block_mask_slices = list(map(pad_slice, block_mask_slices))
    num_active_blocks = np.array(num_active_blocks, dtype=np.int32)
  else:
    active_rows = active_cols = num_active_blocks = None

  block_mask = tokamax_splash_mask_info._downcast_to_small_type(  # pylint: disable=protected-access
      np.concatenate(block_mask_slices, axis=0)
  )
  return (
      tokamax_splash_mask_info.MaskInfo(
          mask_next=None,
          active_rows=active_rows,
          active_cols=active_cols,
          block_mask=block_mask,
          num_active_blocks=num_active_blocks,
          partial_mask_blocks=None,
          q_sequence=q_sequence,
      ),
      mask_function,
  )


def _make_fast_causal_tokamax_mha(
    causal_mask: tokamax_splash_mask.CausalMask,
    *,
    config: tokamax_splash.SplashConfig,
    q_seq_shards: int,
) -> tokamax_splash.SplashAttentionKernel:
  """Builds a Tokamax `SplashAttentionKernel` for `CausalMask` without materializing bool blocks."""

  def _mask_sparsity(info, block_shape: tuple[int, int]) -> float:
    if info.block_mask is None:
      return 1.0
    dense_blocks = (causal_mask.shape[0] // block_shape[0]) * (causal_mask.shape[1] // block_shape[1])
    return float(np.count_nonzero(info.block_mask > 0)) / float(dense_blocks)

  fwd_info_np, mask_fn_fwd = _fast_causal_tokamax_mask_info(
      causal_mask,
      (config.block_q, config.block_kv),
      is_dkv=False,
      return_dynamic_grid=True,
      q_seq_shards=q_seq_shards,
  )
  fwd_sparsity = _mask_sparsity(fwd_info_np, (config.block_q, config.block_kv))
  fwd_mask_info = jax.tree.map(jnp.array, fwd_info_np)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    assert bq_dkv is not None and bkv_dkv is not None
    dkv_info_np, _ = _fast_causal_tokamax_mask_info(
        causal_mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps in (1, 3),
        q_seq_shards=q_seq_shards,
    )
    q_blocks_per_shard = causal_mask.shape[0] // q_seq_shards // bq_dkv
    if config.dq_reduction_steps == 1 and not (
        tokamax_splash_mask_info.dkv_q_blocks_are_contiguous(dkv_info_np, q_blocks_per_shard)
    ):
      raise ValueError(
          "dq_reduction_steps=1 requires the dkv grid to visit every q block in"
          " one run of consecutive steps, which this mask and block sizes do"
          " not. Use dq_reduction_steps=3 or None."
      )
    dkv_sparsity = _mask_sparsity(dkv_info_np, (bq_dkv, bkv_dkv))
    dkv_mask_info = jax.tree.map(jnp.array, dkv_info_np)
  else:
    dkv_sparsity = 1.0

  return tokamax_splash.SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=False,
      save_residuals=False,
      mask_value=tokamax_splash.base.DEFAULT_MASK_VALUE,
      mask_function=mask_fn_fwd,
      fwd_mask_sparsity=fwd_sparsity,
      dkv_mask_sparsity=dkv_sparsity,
  )


def _shard_tokamax_arr(arr, rank: jax.Array, n: int):
  if arr is None or n == 1:
    return arr
  size = arr.shape[0] // n
  return jax.lax.dynamic_slice_in_dim(jnp.asarray(arr), rank * size, size, axis=0)


def _local_tokamax_mask_info(info, rank: jax.Array, n: int):
  if info is None:
    return None
  return info._replace(
      mask_next=_shard_tokamax_arr(info.mask_next, rank, n),
      active_rows=_shard_tokamax_arr(info.active_rows, rank, n),
      active_cols=_shard_tokamax_arr(info.active_cols, rank, n),
      num_active_blocks=_shard_tokamax_arr(info.num_active_blocks, rank, n),
      block_mask=_shard_tokamax_arr(info.block_mask, rank, n),
      q_sequence=_shard_tokamax_arr(info.q_sequence, rank, n),
  )


def _local_splash_kernel(seq_len: int, cfg: Qwen35Config):
  """Builds a q-sharded load-balanced causal Splash kernel and selects this rank's shard.

  Mirrors `SplashAttentionKernel.manual_sharding_spec(P(None, cp))` (or `P(cp)`
  for Tokamax): block tables are sharded along the q-shard dimension,
  `q_sequence` on dim 0, and static partial blocks are replicated.
  """
  n = cfg.cp_size
  gqa_ratio = cfg.num_query_heads // cfg.num_kv_heads if cfg.sa_fold_gqa_into_seq else 1
  q_seq_total = seq_len * gqa_ratio
  q_local = q_seq_total // n
  q_seq_arr = _lb_q_sequence(seq_len, n)
  if gqa_ratio > 1:
    q_seq_arr = np.repeat(q_seq_arr, gqa_ratio)
  rank = jax.lax.axis_index(cfg.cp_axis_names)

  if cfg.use_tokamax_splash:
    sa_layout = tokamax_splash.QKVLayout.SEQ_MINOR if cfg.sa_qk_seq_minor else tokamax_splash.QKVLayout.HEAD_DIM_MINOR
    sa_config = tokamax_splash.SplashConfig(
        block_q=min(cfg.sa_block_q, q_local),
        block_kv=min(cfg.sa_block_kv, seq_len),
        block_kv_compute=min(cfg.sa_block_kv_compute, seq_len),
        block_q_dkv=min(cfg.sa_block_q_dkv, q_local),
        block_kv_dkv=min(cfg.sa_block_kv_dkv, seq_len),
        block_kv_dkv_compute=min(cfg.sa_block_kv_dkv_compute, seq_len),
        use_fused_bwd_kernel=True,
        q_layout=sa_layout,
        k_layout=sa_layout,
        v_layout=sa_layout,
        residual_checkpoint_name="context",
        fuse_reciprocal=cfg.sa_fuse_reciprocal,
        use_base2_exp=cfg.sa_use_base2_exp,
        q_prescaled_log2e=cfg.sa_q_prescaled_log2e,
        skip_uniform_segment_mask=cfg.sa_skip_uniform_segment_mask,
        cast_s_to_v_dtype=cfg.sa_cast_s_to_v_dtype,
        interpret=cfg.sa_interpret,
        dq_reduction_steps=cfg.sa_dq_reduction_steps,
        use_experimental_scheduler=cfg.use_splash_scheduler,
    )
    causal = tokamax_splash_mask.CausalMask(shape=(q_seq_total, seq_len))
    causal.q_sequence = q_seq_arr
    if cfg.sa_fold_gqa_into_seq:
      kernel = _make_fast_causal_tokamax_mha(causal, config=sa_config, q_seq_shards=n)
    else:
      kernel = tokamax_splash.make_splash_mha(causal, config=sa_config, q_seq_shards=n)
    return tokamax_splash.SplashAttentionKernel(
        _local_tokamax_mask_info(kernel.fwd_mask_info, rank, n),
        _local_tokamax_mask_info(kernel.dkv_mask_info, rank, n),
        **kernel.kwargs,
    )

  sa_layout = splash_attention.QKVLayout.SEQ_MINOR if cfg.sa_qk_seq_minor else splash_attention.QKVLayout.HEAD_DIM_MINOR
  block_sizes = splash_attention.BlockSizes(
      block_q=min(cfg.sa_block_q, q_local),
      block_kv=min(cfg.sa_block_kv, seq_len),
      block_kv_compute=min(cfg.sa_block_kv_compute, seq_len),
      block_q_dkv=min(cfg.sa_block_q_dkv, q_local),
      block_kv_dkv=min(cfg.sa_block_kv_dkv, seq_len),
      block_kv_dkv_compute=min(cfg.sa_block_kv_dkv_compute, seq_len),
      use_fused_bwd_kernel=True,
      q_layout=sa_layout,
      k_layout=sa_layout,
      v_layout=sa_layout,
  )
  causal = splash_attention.CausalMask(shape=(q_seq_total, seq_len))
  causal.q_sequence = q_seq_arr
  num_heads_kernel = cfg.num_kv_heads if cfg.sa_fold_gqa_into_seq else cfg.num_query_heads
  mh_mask = splash_attention.MultiHeadMask(masks=[causal for _ in range(num_heads_kernel)])
  kernel = splash_attention.make_splash_mha(
      mh_mask,
      block_sizes=block_sizes,
      head_shards=1,
      q_seq_shards=n,
      residual_checkpoint_name="context",
  )

  def _shard(arr, dim):
    if arr is None:
      return None
    size = arr.shape[dim] // n
    return jax.lax.dynamic_slice_in_dim(jnp.asarray(arr), rank * size, size, axis=dim)

  def _local_info(info):
    if info is None:
      return None
    return info._replace(
        data_next=_shard(info.data_next, 1),
        mask_next=_shard(info.mask_next, 1),
        block_mask=_shard(info.block_mask, 1),
        partial_mask_blocks=(_shard(info.partial_mask_blocks, 1) if info.is_dynamic_mask else info.partial_mask_blocks),
        q_sequence=_shard(info.q_sequence, 0),
    )

  return type(kernel)(
      _local_info(kernel.fwd_mask_info),
      _local_info(kernel.dq_mask_info),
      _local_info(kernel.dkv_mask_info),
      **kernel.kwargs,
  )


# ---------------------------------------------------------------------------
# Partial RoPE + Full Attention (SplashAttention on TPU, dot-product on CPU)
# ---------------------------------------------------------------------------


def compute_rope_cos_sin(
    positions: jax.Array,
    rotary_dim: int,
    max_timescale: float = 10_000_000.0,
) -> tuple[jax.Array, jax.Array]:
  """Precomputes `(cos, sin)` (`[B, S, 1, rotary_dim // 2]` fp32, `512 KiB`/mb) outside `scan`."""
  half = rotary_dim // 2
  freq_seq = jnp.arange(0, half, dtype=jnp.float32)
  inv_freq = 1.0 / (max_timescale ** (freq_seq / float(half)))
  angles = positions.astype(jnp.float32)[:, :, None, None] * inv_freq[None, None, None, :]
  return jnp.cos(angles), jnp.sin(angles)


def apply_partial_rope(
    x: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
    max_timescale: float = 10_000_000.0,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    scale: float = 1.0,
    out_dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Applies NeoX-style half-rotated RoPE to the first `rotary_dim` channels.

  Supports both 4D `[B, S, H, D_h]` and 5D KV-grouped `[B, K, S, G, D_h]`
  inputs, optionally fusing scalar query scaling (`scale != 1.0`) and output
  casting (`out_dtype`, e.g. `float8_e4m3fn`).
  """
  orig_dtype = x.dtype
  target_dtype = jnp.dtype(orig_dtype if out_dtype is None else out_dtype)
  x_rot = x[..., :rotary_dim].astype(jnp.float32)
  x_pass = x[..., rotary_dim:]

  half = rotary_dim // 2
  if rope_cos_sin is not None:
    cos, sin = rope_cos_sin
  else:
    cos, sin = compute_rope_cos_sin(positions, rotary_dim, max_timescale)
  if x.ndim == 5:
    cos = cos[:, None, :, :, :]
    sin = sin[:, None, :, :, :]

  x1, x2 = x_rot[..., :half], x_rot[..., half:]
  y1 = x1 * cos - x2 * sin
  y2 = x2 * cos + x1 * sin
  if scale != 1.0 or target_dtype != orig_dtype:
    x_pass_f32 = x_pass.astype(jnp.float32)
    if scale != 1.0:
      y1 = y1 * scale
      y2 = y2 * scale
      x_pass_f32 = x_pass_f32 * scale
    return jnp.concatenate([y1, y2, x_pass_f32], axis=-1).astype(target_dtype)
  y_rot = jnp.concatenate([y1, y2], axis=-1).astype(orig_dtype)
  return jnp.concatenate([y_rot, x_pass], axis=-1)


def full_attn_mixer(
    x: jax.Array,
    w: GatheredAttnWeights,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: Qwen35Config,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Executes context-parallel Qwen3NextFullAttention and returns `x + attn_out`.

  `x` (`[B_mb, S_local, D]`) and `segment_ids` are in natural CP order;
  `positions`
  and `rope_cos_sin` are in load-balanced CP order (see `cp_natural_to_lb`). The
  attention input is permuted into load-balanced order, K/V are all-gathered
  over
  CP into the full natural sequence, and the output is permuted back.
  """
  bsz, seq_local, d_model = x.shape
  seq_len = seq_local * cfg.cp_size
  use_pallas_norm = cfg.use_pallas_kernels and cfg.use_pallas_rmsnorm
  use_grouped = (
      cfg.attn_kv_grouped_layout
      and cfg.sa_fold_gqa_into_seq
      and not cfg.attn_qkv_2d_matmul
      and w.w_q_grouped is not None
      and w.w_gate_grouped is not None
      and w.w_out_grouped is not None
  )
  normed = ops.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps, use_pallas=use_pallas_norm)
  normed = cp_natural_to_lb(normed, cfg)
  seg_q = cp_natural_to_lb(segment_ids, cfg)
  seg_kv = jax.lax.all_gather(segment_ids, cfg.cp_axis_names, axis=1, tiled=True)

  if use_grouped:
    q_raw = jnp.einsum("bsd,dkgc->bksgc", normed, w.w_q_grouped)
    q_raw = checkpoint_name(q_raw, "query_proj")
    gate = jnp.einsum("bsd,dkgc->bksgc", normed, w.w_gate_grouped)
    k_raw = jnp.einsum("bsd,dhc->bshc", normed, w.k_proj)
    k_raw = checkpoint_name(k_raw, "key_proj")
    v = jnp.einsum("bsd,dhc->bshc", normed, w.v_proj)
    v = checkpoint_name(v, "value_proj")
  elif cfg.attn_qkv_2d_matmul:
    qg = jnp.matmul(normed, w.q_proj.reshape(d_model, -1)).reshape(bsz, seq_local, cfg.num_query_heads, 2 * cfg.head_dim)
    q_raw, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
    q_raw = checkpoint_name(q_raw, "query_proj")
    k_raw = jnp.matmul(normed, w.k_proj.reshape(d_model, -1)).reshape(bsz, seq_local, cfg.num_kv_heads, cfg.head_dim)
    k_raw = checkpoint_name(k_raw, "key_proj")
    v = jnp.matmul(normed, w.v_proj.reshape(d_model, -1)).reshape(bsz, seq_local, cfg.num_kv_heads, cfg.head_dim)
    v = checkpoint_name(v, "value_proj")
  else:
    qg = jnp.einsum("bsd,dhc->bshc", normed, w.q_proj)
    q_raw, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
    q_raw = checkpoint_name(q_raw, "query_proj")

    k_raw = jnp.einsum("bsd,dhc->bshc", normed, w.k_proj)
    k_raw = checkpoint_name(k_raw, "key_proj")
    v = jnp.einsum("bsd,dhc->bshc", normed, w.v_proj)
    v = checkpoint_name(v, "value_proj")

  # Keep 4D/5D per-head q_norm/k_norm in pure JAX so XLA fuses them into RoPE
  # without splitting the packed head*dim minor lane dimension via HBM reshape.
  q = ops.rms_norm_zero_centered(q_raw, w.q_norm, cfg.norm_eps, use_pallas=False)
  k = ops.rms_norm_zero_centered(k_raw, w.k_norm, cfg.norm_eps, use_pallas=False)

  use_splash = cfg.use_pallas_kernels or (cfg.use_tokamax_splash and cfg.sa_interpret)
  use_tokamax_splash = use_splash and cfg.use_tokamax_splash
  scale = cfg.head_dim**-0.5
  q_scale = scale * math.log2(math.e) if (use_tokamax_splash and cfg.sa_q_prescaled_log2e) else scale
  q_dtype = jnp.float8_e4m3fn if (use_tokamax_splash and cfg.experimental_sa_quant_q_fp8) else x.dtype
  if use_grouped:
    q_scaled = apply_partial_rope(
        q,
        positions,
        cfg.rotary_dim,
        cfg.rope_max_timescale,
        rope_cos_sin=rope_cos_sin,
        scale=q_scale,
        out_dtype=q_dtype,
    )
  else:
    q = apply_partial_rope(
        q,
        positions,
        cfg.rotary_dim,
        cfg.rope_max_timescale,
        rope_cos_sin=rope_cos_sin,
    )
    # Fold LOG2E into f32 softmax scale before fp8 cast so q rounds once.
    q_scaled = (q.astype(jnp.float32) * q_scale).astype(q_dtype)
  k = apply_partial_rope(
      k,
      positions,
      cfg.rotary_dim,
      cfg.rope_max_timescale,
      rope_cos_sin=rope_cos_sin,
  )

  # K/V are small (2 heads): all-gather them over CP into the natural full sequence.
  k_full = cp_all_gather_lb_to_natural(k, cfg)
  v_full = cp_all_gather_lb_to_natural(v, cfg)

  gqa_ratio = cfg.num_query_heads // cfg.num_kv_heads
  attn_5d = None
  attn_out = None
  if use_splash:
    if use_grouped:
      q_bhsd = q_scaled.reshape(bsz, cfg.num_kv_heads, seq_local * gqa_ratio, cfg.head_dim)
      seg_q_splash = jnp.repeat(seg_q, gqa_ratio, axis=1)
    elif cfg.sa_fold_gqa_into_seq:
      q_bhsd = (
          q_scaled.reshape(bsz, seq_local, cfg.num_kv_heads, gqa_ratio, cfg.head_dim)
          .transpose(0, 2, 1, 3, 4)
          .reshape(bsz, cfg.num_kv_heads, seq_local * gqa_ratio, cfg.head_dim)
      )
      seg_q_splash = jnp.repeat(seg_q, gqa_ratio, axis=1)
    else:
      q_bhsd = jnp.swapaxes(q_scaled, 1, 2)
      seg_q_splash = seg_q
    k_bhsd = jnp.swapaxes(k_full, 1, 2)
    if use_tokamax_splash and cfg.experimental_sa_quant_k_fp8:
      k_bhsd = k_bhsd.astype(jnp.float8_e4m3fn)
    v_bhsd = jnp.swapaxes(v_full, 1, 2)
    splash_fn = _local_splash_kernel(seq_len, cfg)
    seg_cls = tokamax_splash.SegmentIds if cfg.use_tokamax_splash else splash_attention.SegmentIds

    def _single_batch_splash(q_i, k_i, v_i, seg_q_i, seg_kv_i):
      seg = seg_cls(q=seg_q_i, kv=seg_kv_i)
      return splash_fn(q_i, k_i, v_i, segment_ids=seg)

    if cfg.sa_unroll_batch and bsz > 1:
      attn_bhsd = jnp.stack(
          [_single_batch_splash(q_bhsd[i], k_bhsd[i], v_bhsd[i], seg_q_splash[i], seg_kv[i]) for i in range(bsz)],
          axis=0,
      )
    else:
      attn_bhsd = jax.vmap(_single_batch_splash)(q_bhsd, k_bhsd, v_bhsd, seg_q_splash, seg_kv)
    if use_grouped:
      attn_5d = attn_bhsd.reshape(bsz, cfg.num_kv_heads, seq_local, gqa_ratio, cfg.head_dim)
    elif cfg.sa_fold_gqa_into_seq:
      attn_out = (
          attn_bhsd.reshape(bsz, cfg.num_kv_heads, seq_local, gqa_ratio, cfg.head_dim)
          .transpose(0, 2, 1, 3, 4)
          .reshape(bsz, seq_local, cfg.num_query_heads, cfg.head_dim)
      )
    else:
      attn_out = jnp.swapaxes(attn_bhsd, 1, 2)
  elif cfg.sa_fold_gqa_into_seq:
    if use_grouped:
      q_folded = q_scaled.reshape(bsz, cfg.num_kv_heads, seq_local * gqa_ratio, cfg.head_dim)
    else:
      q_folded = (
          q_scaled.reshape(bsz, seq_local, cfg.num_kv_heads, gqa_ratio, cfg.head_dim)
          .transpose(0, 2, 1, 3, 4)
          .reshape(bsz, cfg.num_kv_heads, seq_local * gqa_ratio, cfg.head_dim)
      )
    seg_q_folded = jnp.repeat(seg_q, gqa_ratio, axis=1)
    logits = jnp.einsum(
        "bhsd,bthd->bhst",
        q_folded.astype(jnp.float32),
        k_full.astype(jnp.float32),
    )
    q_idx = jnp.repeat(cp_lb_global_index(seq_local, cfg), gqa_ratio)
    kv_idx = jnp.arange(seq_len, dtype=jnp.int32)
    causal_mask = q_idx[:, None] >= kv_idx[None, :]
    same_seg = (seg_q_folded[:, :, None] == seg_kv[:, None, :]) & (seg_q_folded[:, :, None] > 0)
    mask = causal_mask[None, None, :, :] & same_seg[:, None, :, :]
    logits = jnp.where(mask, logits, -1e30)
    probs = jax.nn.softmax(logits, axis=-1)
    attn_folded = jnp.einsum("bhst,bthd->bhsd", probs, v_full.astype(jnp.float32)).astype(x.dtype)
    if use_grouped:
      attn_5d = attn_folded.reshape(bsz, cfg.num_kv_heads, seq_local, gqa_ratio, cfg.head_dim)
      attn_5d = checkpoint_name(attn_5d, "context")
    else:
      attn_out = (
          attn_folded.reshape(bsz, cfg.num_kv_heads, seq_local, gqa_ratio, cfg.head_dim)
          .transpose(0, 2, 1, 3, 4)
          .reshape(bsz, seq_local, cfg.num_query_heads, cfg.head_dim)
      )
      attn_out = checkpoint_name(attn_out, "context")
  else:
    k_rep = jnp.repeat(k_full, gqa_ratio, axis=2)
    v_rep = jnp.repeat(v_full, gqa_ratio, axis=2)
    logits = jnp.einsum(
        "bshd,bthd->bhst",
        q_scaled.astype(jnp.float32),
        k_rep.astype(jnp.float32),
    )
    q_idx = cp_lb_global_index(seq_local, cfg)
    kv_idx = jnp.arange(seq_len, dtype=jnp.int32)
    causal_mask = q_idx[:, None] >= kv_idx[None, :]
    same_seg = (seg_q[:, :, None] == seg_kv[:, None, :]) & (seg_q[:, :, None] > 0)
    mask = causal_mask[None, None, :, :] & same_seg[:, None, :, :]
    logits = jnp.where(mask, logits, -1e30)
    probs = jax.nn.softmax(logits, axis=-1)
    attn_out = jnp.einsum("bhst,bthd->bshd", probs, v_rep.astype(jnp.float32)).astype(x.dtype)
    attn_out = checkpoint_name(attn_out, "context")

  if use_grouped:
    assert attn_5d is not None and w.w_out_grouped is not None
    gated_5d = (attn_5d.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(x.dtype)
    proj = cp_lb_to_natural(jnp.einsum("bksgc,kgco->bso", gated_5d, w.w_out_grouped), cfg)
    proj = checkpoint_name(proj, "out_proj")
    return x + proj

  assert attn_out is not None
  gate_3d = gate.reshape(bsz, seq_local, cfg.attn_q_out_dim)
  if cfg.use_pallas_kernels and cfg.attn_gate_bf16_align:
    gate_3d = ops.forward_layout_constraint(gate_3d, (2, 0, 1))
  gated = (
      attn_out.reshape(bsz, seq_local, cfg.attn_q_out_dim).astype(jnp.float32)
      * jax.nn.sigmoid(gate_3d.astype(jnp.float32))
  ).astype(x.dtype)
  proj = cp_lb_to_natural(jnp.matmul(gated, w.out_proj), cfg)
  proj = checkpoint_name(proj, "out_proj")
  return x + proj


# ---------------------------------------------------------------------------
# Gated DeltaNet Mixer (Fused Conv1D + WY Delta Rule Pallas Kernel)
# ---------------------------------------------------------------------------


def _pure_jax_gdn_cpu_fallback(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    segment_ids: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Zero-Flax pure-JAX segmented causal Conv1D + Gated Delta Rule reference."""
  bsz, seq_len, qkv_dim = qkv.shape
  k_conv = cfg.gdn_conv_kernel_size
  qkv_f32 = qkv.astype(jnp.float32)
  w_f32 = conv_weight.astype(jnp.float32)  # [K_conv, 1, qkv_dim]

  conv_acc = jnp.zeros_like(qkv_f32)
  for lag in range(k_conv):
    tap_w = w_f32[k_conv - 1 - lag, 0, :]
    if lag == 0:
      shifted = qkv_f32
      same_seg = segment_ids > 0
    else:
      shifted = jnp.pad(qkv_f32[:, :-lag, :], ((0, 0), (lag, 0), (0, 0)))
      seg_shifted = jnp.pad(segment_ids[:, :-lag], ((0, 0), (lag, 0)), constant_values=-1)
      same_seg = (segment_ids > 0) & (segment_ids == seg_shifted)
    conv_acc = conv_acc + jnp.where(same_seg[:, :, None], shifted * tap_w[None, None, :], 0.0)
  qkv_conv = jax.nn.silu(conv_acc)

  key_dim = cfg.gdn_key_dim
  q_c, k_c, v_c = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)
  q = q_c.reshape(bsz, seq_len, cfg.gdn_num_k_heads, cfg.gdn_head_k_dim)
  k = k_c.reshape(bsz, seq_len, cfg.gdn_num_k_heads, cfg.gdn_head_k_dim)
  v = v_c.reshape(bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim)

  q = q * jax.lax.rsqrt(jnp.sum(jnp.square(q), axis=-1, keepdims=True) + 1e-6)
  k = k * jax.lax.rsqrt(jnp.sum(jnp.square(k), axis=-1, keepdims=True) + 1e-6)
  q = q * (cfg.gdn_head_k_dim**-0.5)

  repeats = cfg.gdn_num_v_heads // cfg.gdn_num_k_heads
  if repeats > 1:
    q = jnp.repeat(q, repeats, axis=2)
    k = jnp.repeat(k, repeats, axis=2)

  beta = jax.nn.sigmoid(b.astype(jnp.float32))
  g = -jnp.exp(a_log.astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))

  init_state = jnp.zeros(
      (bsz, cfg.gdn_num_v_heads, cfg.gdn_head_k_dim, cfg.gdn_head_v_dim),
      dtype=jnp.float32,
  )
  init_prev_seg = jnp.full((bsz,), -1, dtype=jnp.int32)

  def _step(carry, tok_inputs):
    state, prev_seg = carry
    q_t, k_t, v_t, beta_t, g_t, seg_t = tok_inputs
    valid = seg_t > 0
    same_doc = valid & (seg_t == prev_seg)
    s_in = jnp.where(same_doc[:, None, None, None], state, 0.0)
    s_dec = s_in * jnp.exp(g_t)[:, :, None, None]
    kv_mem = jnp.einsum("bhk,bhkv->bhv", k_t, s_dec)
    delta = (v_t - kv_mem) * beta_t[:, :, None]
    s_next = s_dec + jnp.einsum("bhk,bhv->bhkv", k_t, delta)
    s_next = jnp.where(valid[:, None, None, None], s_next, 0.0)
    out_t = jnp.where(valid[:, None, None], jnp.einsum("bhk,bhkv->bhv", q_t, s_next), 0.0)
    return (s_next, seg_t), out_t

  xs = (
      jnp.swapaxes(q, 0, 1),
      jnp.swapaxes(k, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(beta, 0, 1),
      jnp.swapaxes(g, 0, 1),
      jnp.swapaxes(segment_ids, 0, 1),
  )
  _, outs_sbhd = jax.lax.scan(_step, (init_state, init_prev_seg), xs)
  return jnp.swapaxes(outs_sbhd, 0, 1).astype(qkv.dtype)


def gdn_mixer(
    x: jax.Array,
    w: GatheredGdnWeights,
    segment_ids: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Executes context-parallel Qwen3NextGatedDeltaNet and returns `x + gdn_out`.

  `x` is `[B_mb, S_local, D]` in natural CP order; the recurrent state and conv
  halo
  are passed across `cfg.cp_axis_names` ranks.
  """
  bsz, seq_len, _ = x.shape
  use_pallas_norm = cfg.use_pallas_kernels and cfg.use_pallas_rmsnorm
  normed = ops.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps, use_pallas=use_pallas_norm)

  if cfg.gdn_split_qkvz_proj and w.in_proj_qkv is not None and w.in_proj_z is not None:
    qkv = jnp.matmul(normed, w.in_proj_qkv)
    qkv = checkpoint_name(qkv, "qkv_proj")
    z = jnp.matmul(normed, w.in_proj_z).reshape(bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim)
  else:
    qkvz = jnp.matmul(normed, w.in_proj_qkvz)
    qkvz = checkpoint_name(qkvz, "qkv_proj")
    qkv = qkvz[..., : cfg.gdn_qkv_dim]
    z = qkvz[..., cfg.gdn_qkv_dim :].reshape(bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim)

  ba = jnp.matmul(normed, w.in_proj_ba)
  b = ba[..., : cfg.gdn_num_v_heads]
  a = ba[..., cfg.gdn_num_v_heads :]

  if cfg.use_pallas_kernels:
    cp_matmul_precision = jax.lax.Precision.HIGH if cfg.gdn_cp_matmul_high else jax.lax.Precision.HIGHEST
    core_out, _ = gdn_decoupled_conv1d(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=w.conv1d,
        conv_bias=None,
        a_log=w.A_log,
        dt_bias=w.dt_bias,
        conv_state=None,
        recurrent_state=None,
        num_k_heads=cfg.gdn_num_k_heads,
        num_v_heads=cfg.gdn_num_v_heads,
        head_k_dim=cfg.gdn_head_k_dim,
        head_v_dim=cfg.gdn_head_v_dim,
        conv_kernel_size=cfg.gdn_conv_kernel_size,
        chunk_size=cfg.gdn_chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.bfloat16,
        cp_axis_name=cfg.cp_axis_names,
        segment_ids=segment_ids,
        cp_matmul_precision=cp_matmul_precision,
        conv_input_in_qkv_dtype=True,
        conv_dense_token_stack=True,
        cp_m_scan=cfg.gdn_cp_m_scan,
        cp_dh_scan=cfg.gdn_cp_dh_scan,
        pass1_states_and_bf16_qkv=True,
        bwd_natural_layout=cfg.gdn_bwd_natural_layout,
        conv_natural_layout=True,
        fwd_head_group_size=cfg.gdn_fwd_head_group_size,
    )
    if cfg.gdn_norm_layout_mode in ("token_minor", "bf16_z_token_minor"):
      # Pin bf16 GDN output token-minor (0, 2, 3, 1) ahead of fp32 gated norm!
      core_out = ops.forward_layout_constraint(core_out, (0, 2, 3, 1))
      if cfg.gdn_norm_layout_mode == "bf16_z_token_minor":
        z = ops.forward_layout_constraint(z, (0, 2, 3, 1))
  else:
    # CPU reference: all-gather the sequence over CP, run the full recurrence, keep
    # this rank's natural-order shard.
    def _ag(t):
      return jax.lax.all_gather(t, cfg.cp_axis_names, axis=1, tiled=True)

    core_full = _pure_jax_gdn_cpu_fallback(
        qkv=_ag(qkv),
        b=_ag(b),
        a=_ag(a),
        conv_weight=w.conv1d,
        a_log=w.A_log,
        dt_bias=w.dt_bias,
        segment_ids=_ag(segment_ids),
        cfg=cfg,
    )
    core_out = jax.lax.dynamic_slice_in_dim(
        core_full,
        jax.lax.axis_index(cfg.cp_axis_names) * seq_len,
        seq_len,
        axis=1,
    )

  use_pallas_gated = cfg.use_pallas_kernels and cfg.use_pallas_gated_rmsnorm
  gated_normed = ops.rms_norm_gated(core_out, z, w.gdn_norm, cfg.norm_eps, use_pallas=use_pallas_gated)
  flat = gated_normed.reshape(bsz, seq_len, cfg.gdn_value_dim)
  # Token-minor layout on out_proj matmul (`gdn_out_proj_token_minor=true`)
  if cfg.use_pallas_kernels and cfg.gdn_norm_layout_mode != "natural":
    flat_tm = ops.forward_layout_constraint(flat, (2, 0, 1))
    proj = jnp.matmul(flat_tm, w.out_proj)
  else:
    proj = jnp.matmul(flat, w.out_proj)
  proj = checkpoint_name(proj, "out_proj")
  return x + proj


# ---------------------------------------------------------------------------
# Gated Shared Expert + Decomposed Routed MoE Stages (for Unsplit & BatchSplit)
# ---------------------------------------------------------------------------


def shared_expert_and_residual(
    x_post_mixer: jax.Array,
    normed_post_mixer: jax.Array,
    w_moe: GatheredMoeWeights,
) -> jax.Array:
  """Computes `x_post_mixer + sigmoid(shared_gate(normed)) * shared_mlp(normed)`."""
  g0 = jnp.matmul(normed_post_mixer, w_moe.shared_wi_0)
  u0 = jnp.matmul(normed_post_mixer, w_moe.shared_wi_1)
  act = (jax.nn.silu(g0.astype(jnp.float32)) * u0.astype(jnp.float32)).astype(x_post_mixer.dtype)
  shared_out = jnp.matmul(act, w_moe.shared_wo)
  gate_prob = jax.nn.sigmoid(jnp.matmul(normed_post_mixer, w_moe.shared_gate).astype(jnp.float32))
  gated_shared = (shared_out.astype(jnp.float32) * gate_prob).astype(x_post_mixer.dtype)
  return x_post_mixer + gated_shared


class MoeAuxStats(NamedTuple):
  """Local unnormalized Switch load-balancing sums `[E]` fp32 for MoE layers."""

  expert_counts: jax.Array  # stop_gradient count of valid top-k expert picks
  prob_sum: jax.Array  # differentiable sum of valid router softmax probs


def _compute_local_moe_aux_stats(
    probs: jax.Array,
    topk_ids: jax.Array,
    segment_ids: jax.Array | None,
    *,
    num_experts_per_tok: int,
) -> MoeAuxStats:
  """Computes local `(expert_counts, prob_sum)` over valid (`segment_ids > 0`) tokens."""
  t_len, num_experts = probs.shape
  ids_sg = jax.lax.stop_gradient(topk_ids)
  iota_e = jax.lax.broadcasted_iota(ids_sg.dtype, (1, num_experts), 1)
  if segment_ids is not None:
    valid_col = jax.lax.stop_gradient((segment_ids.reshape(t_len, 1) > 0).astype(jnp.float32))
    prob_sum = jnp.sum(probs * valid_col, axis=0)
  else:
    valid_col = jnp.ones((1, 1), dtype=jnp.float32)
    prob_sum = jnp.sum(probs, axis=0)
  counts = jnp.zeros((num_experts,), dtype=jnp.float32)
  for j in range(num_experts_per_tok):
    counts = counts + jnp.sum(
        jnp.where(ids_sg[:, j : j + 1] == iota_e, valid_col, 0.0),
        axis=0,
    )
  return MoeAuxStats(
      expert_counts=jax.lax.stop_gradient(counts),
      prob_sum=prob_sum,
  )


def add_moe_aux_stats(a: MoeAuxStats | None, b: MoeAuxStats | None) -> MoeAuxStats | None:
  if a is None or b is None:
    return None
  return MoeAuxStats(
      expert_counts=jax.lax.stop_gradient(a.expert_counts + b.expert_counts),
      prob_sum=a.prob_sum + b.prob_sum,
  )


def stack_moe_aux_stats(
    items: tuple[MoeAuxStats | None, ...],
    *,
    pack: bool = False,
) -> MoeAuxStats | jax.Array | None:
  """Stacks per-sublayer MoE aux stats across a superblock."""
  if any(x is None for x in items):
    return None
  expert_counts = jax.lax.stop_gradient(jnp.stack([x.expert_counts for x in items if x is not None], axis=0))
  prob_sum = jnp.stack([x.prob_sum for x in items if x is not None], axis=0)
  if pack:
    return jnp.stack([expert_counts, prob_sum], axis=1)
  return MoeAuxStats(
      expert_counts=expert_counts,
      prob_sum=prob_sum,
  )


def moe_local_route_with_aux(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: Qwen35Config,
    *,
    route_after: jax.Array | tuple[jax.Array, ...] | None = None,
    segment_ids: jax.Array | None = None,
    collect_aux: bool = False,
) -> tuple[jax.Array, jax.Array, MoeAuxStats | None]:
  """Computes local fp32 top-k routing plus optional Switch load-balancing stats."""
  bsz, seq_len, emb_dim = normed_tokens.shape
  flat_tokens = normed_tokens.reshape(bsz * seq_len, emb_dim)

  logits = jnp.matmul(flat_tokens.astype(jnp.float32), router_gate.astype(jnp.float32))
  if route_after is not None:
    deps = route_after if isinstance(route_after, tuple) else (route_after,)
    for dep in deps:
      logits = ops.order_after(logits, dep)

  probs = jax.nn.softmax(logits, axis=-1)
  k = cfg.num_experts_per_tok
  t_len, num_experts = probs.shape

  if cfg.moe_router_dense_bwd:
    probs_dtype = probs.dtype
    raw_vals, raw_ids = jax.lax.top_k(jax.lax.stop_gradient(probs), k)
    vals_sg = checkpoint_name(raw_vals, "moe_routing")
    topk_ids = checkpoint_name(raw_ids.astype(jnp.int32), "moe_routing")

    @jax.custom_vjp
    def _attach_topk_grad(p: jax.Array, v_sg: jax.Array, ids: jax.Array) -> jax.Array:
      del p, ids
      return v_sg

    def _attach_topk_grad_fwd(p: jax.Array, v_sg: jax.Array, ids: jax.Array):
      del p
      return v_sg, ids

    def _attach_topk_grad_bwd(ids: jax.Array, g_w: jax.Array):
      g_w = g_w.astype(probs_dtype)
      iota_e = jax.lax.broadcasted_iota(ids.dtype, (1, num_experts), 1)
      acc = jnp.zeros((t_len, num_experts), dtype=probs_dtype)
      for j in range(k):
        acc = acc + jnp.where(
            iota_e == ids[:, j : j + 1],
            g_w[:, j : j + 1],
            jnp.zeros((), dtype=probs_dtype),
        )
      return (acc, None, None)

    _attach_topk_grad.defvjp(_attach_topk_grad_fwd, _attach_topk_grad_bwd)
    topk_weights = _attach_topk_grad(probs, vals_sg, topk_ids)
  else:
    topk_weights, topk_ids = jax.lax.top_k(probs, k)
    topk_ids = topk_ids.astype(jnp.int32)
  aux_stats = _compute_local_moe_aux_stats(probs, topk_ids, segment_ids, num_experts_per_tok=k) if collect_aux else None
  denom = jnp.sum(topk_weights, axis=-1, keepdims=True)
  topk_weights = topk_weights / jnp.where(denom > 0.0, denom, 1.0)
  return topk_weights, topk_ids, aux_stats


def moe_local_route_only(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: Qwen35Config,
    *,
    route_after: jax.Array | tuple[jax.Array, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes local top-k routing `(topk_weights, topk_ids)` in fp32 ahead of Shared Expert."""
  topk_weights, topk_ids, _ = moe_local_route_with_aux(
      normed_tokens,
      router_gate,
      cfg,
      route_after=route_after,
      collect_aux=False,
  )
  return topk_weights, topk_ids


def moe_dispatch_from_routed(
    normed_tokens: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    cfg: Qwen35Config,
    *,
    order_tokens_on_router: bool = True,
) -> DispatchedMoeTokens:
  """Launches SC0-pinned zero-relayout 3D FP8 EP token + routing all-gathers."""
  bsz, seq_len, emb_dim = normed_tokens.shape
  tokens_disp = normed_tokens.reshape(bsz * seq_len, emb_dim // 128, 128)
  if order_tokens_on_router:
    tokens_disp = ops.order_after(tokens_disp, topk_weights)
  tokens_ep = ops.ep_token_dispatch_3d_fp8(tokens_disp, cfg)
  weights_ep = ops.ep_routing_weights_dispatch(topk_weights, cfg)
  ids_ep = ops.ep_routing_ids_dispatch(topk_ids, cfg)
  return DispatchedMoeTokens(
      tokens_ep=tokens_ep,
      topk_weights_ep=weights_ep,
      topk_ids_ep=ids_ep,
  )


def moe_local_route_and_dispatch_with_aux(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: Qwen35Config,
    *,
    route_after: jax.Array | tuple[jax.Array, ...] | None = None,
    segment_ids: jax.Array | None = None,
    collect_aux: bool = False,
) -> tuple[DispatchedMoeTokens, MoeAuxStats | None]:
  """Computes local top-k routing (+ optional aux stats) and launches SC0 EP all-gathers."""
  topk_weights, topk_ids, aux_stats = moe_local_route_with_aux(
      normed_tokens,
      router_gate,
      cfg,
      route_after=route_after,
      segment_ids=segment_ids,
      collect_aux=collect_aux,
  )
  disp = moe_dispatch_from_routed(
      normed_tokens,
      topk_weights,
      topk_ids,
      cfg,
      order_tokens_on_router=(route_after is not None),
  )
  return disp, aux_stats


def moe_local_route_and_dispatch(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: Qwen35Config,
    *,
    route_after: jax.Array | tuple[jax.Array, ...] | None = None,
) -> DispatchedMoeTokens:
  """Computes local top-k routing and launches SC0-pinned EP token + routing all-gathers."""
  disp, _ = moe_local_route_and_dispatch_with_aux(normed_tokens, router_gate, cfg, route_after=route_after)
  return disp


def moe_expert_gmm_and_unsort(
    dispatched: DispatchedMoeTokens,
    wi: moe_kernels.Fp8SteWeight,
    wo: moe_kernels.Fp8SteWeight,
    cfg: Qwen35Config,
    *,
    return_weights: bool = False,
    forward_tie: object | None = None,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
    use_external_bwd_rw: bool = False,
):
  """No-drop routed MoE: local sort -> chunked FP8 `wi` GMM + SwiGLU -> `wo` GMM -> unsort.

  Sorts ALL `T_ep * top_k` EP slots once with this rank's local-expert slots
  first and hands the full routing to `moe_kernels.routed_moe_forked_3d_fp8`,
  which processes them in chunks of `cfg.moe_chunk_capacity(T_ep // ep_size)`
  sorted slots so no slot is ever dropped. Routing weights are gathered (and
  their
  gradients scatter-added) only for the processed `capacity`-row windows.
  """
  if not isinstance(dispatched.tokens_ep, ops.Fp8DispatchedTokens):
    raise ValueError(
        "Routed MoE requires FP8 3D EP token dispatch"
        " (ops.Fp8DispatchedTokens), got"
        f" {type(dispatched.tokens_ep).__name__}."
    )
  if not (isinstance(wi, moe_kernels.Fp8SteWeight) and isinstance(wo, moe_kernels.Fp8SteWeight)):
    raise ValueError("Routed MoE requires Fp8SteWeight wi/wo, got" f" {type(wi).__name__}/{type(wo).__name__}.")
  num_tokens_ep = dispatched.tokens_ep.shape[0]
  topk = cfg.num_experts_per_tok
  e_local = cfg.local_num_experts
  num_slots = num_tokens_ep * topk
  capacity = cfg.moe_chunk_capacity(num_tokens_ep // cfg.ep_size)
  topk_ids_flat = dispatched.topk_ids_ep.reshape(-1).astype(jnp.int32)
  ep_idx = ops.ep_axis_index(cfg)
  if cfg.moe_local_65key_sort:
    # 65-key local sort: maps local experts to `0..63` and all non-local experts
    # to sentinel `64`, so local slots come first and `jnp.searchsorted` gives the
    # exact `local_gs` with ZERO `[327680, 512]` `one_hot`.
    local_expert = topk_ids_flat - ep_idx * e_local
    sort_key = jnp.where((local_expert >= 0) & (local_expert < e_local), local_expert, e_local)
    if cfg.use_pallas_kernels:
      sorted_keys, sort_indices = tc.packed_stable_sort(sort_key, e_local)
    else:
      sort_indices = jnp.argsort(sort_key, stable=True).astype(jnp.int32)
      sorted_keys = jnp.take(sort_key, sort_indices, axis=0)
    if cfg.moe_chunked_searchsorted:
      chunk = 512
      num_chunks = (num_slots + chunk - 1) // chunk
      key_chunks = jnp.pad(
          sorted_keys,
          (0, num_chunks * chunk - num_slots),
          constant_values=e_local,
      ).reshape(num_chunks, chunk)
      thresholds = jnp.arange(1, e_local + 1, dtype=jnp.int32)
      full_chunks = jnp.sum(key_chunks[:, -1] < thresholds[:, None], axis=1, dtype=jnp.int32)
      row = jnp.minimum(full_chunks, num_chunks - 1)
      in_chunk = jnp.sum(key_chunks[row] < thresholds[:, None], axis=1, dtype=jnp.int32)
      ends = row * chunk + in_chunk
    else:
      ends = jnp.searchsorted(
          sorted_keys,
          jnp.arange(1, e_local + 1, dtype=jnp.int32),
          side="left",
          method="scan_unrolled",
      ).astype(jnp.int32)
    local_gs = jnp.diff(ends, prepend=jnp.zeros((1,), dtype=jnp.int32))
  else:
    if cfg.use_pallas_kernels:
      _, topk_argsort_indices = tc.packed_stable_sort(topk_ids_flat, cfg.num_experts - 1)
    else:
      topk_argsort_indices = jnp.argsort(topk_ids_flat, stable=True).astype(jnp.int32)
    group_sizes_global = jax.nn.one_hot(topk_ids_flat, cfg.num_experts, dtype=jnp.int32).sum(axis=0)
    group_offsets = jnp.cumulative_sum(group_sizes_global, include_initial=True)
    start_raw = group_offsets[ep_idx * e_local].astype(jnp.int32)
    # Shift the global expert order so this rank's local slots come first.
    sort_indices = jax.lax.dynamic_slice_in_dim(
        jnp.pad(topk_argsort_indices, (0, num_slots)),
        start_raw,
        num_slots,
        axis=0,
    )
    local_gs = jax.lax.dynamic_slice_in_dim(group_sizes_global, ep_idx * e_local, e_local, axis=0)
  local_gs_raw = local_gs.astype(jnp.int32)
  count_total_raw = jnp.sum(local_gs_raw).astype(jnp.int32)
  # Save `sort_indices` itself: the kernel derives every processed window's token ids
  # (`// topk`) and routing weights from it, so remat never re-sorts (or re-all-gathers
  # `topk_ids`).
  sort_indices = checkpoint_name(sort_indices, "moe_routing")
  # Unsorted routing weights; saving them (same size as the old sorted `w_rows`, but no
  # gather) keeps remat from re-running the routing-weight all-gather.
  topk_weights_flat = checkpoint_name(jnp.ravel(dispatched.topk_weights_ep).astype(jnp.float32), "moe_routing")

  # Chunk 0's routing weights (0 past `count_total`): a `capacity`-row gather instead of
  # all `num_slots`; overflow windows gather theirs inside the kernel loops. Valid rows
  # are distinct slots and padding rows get distinct out-of-range indices, so the gather
  # is unique_indices. It is a stop_gradient view: the kernel returns the full
  # `d(topk_weights_flat)` (window-local unique scatter-adds, no scatter-sort).
  iota0 = jnp.arange(capacity, dtype=jnp.int32)
  w_rows0 = (
      jax.lax.stop_gradient(topk_weights_flat)
      .at[
          jnp.where(
              iota0 < count_total_raw,
              sort_indices[:capacity],
              num_slots + iota0,
          )
      ]
      .get(mode="fill", fill_value=0.0, unique_indices=True)
  )

  reduce_meta = (
      moe_kernels.chunk_reduce_metadata(
          jax.lax.stop_gradient(sort_indices[:capacity] // topk),
          jnp.minimum(count_total_raw, capacity),
          cfg,
          num_tokens_ep,
      )
      if cfg.use_pallas_kernels
      else None
  )
  w_rows0 = checkpoint_name(w_rows0, "moe_routing")
  if cfg.pack_scan_routing_metadata:
    if reduce_meta is not None:
      packed_small = checkpoint_name(
          jnp.concatenate([count_total_raw[None], local_gs_raw, reduce_meta.blocks], axis=0),
          "moe_routing",
      )
      packed_rows = checkpoint_name(
          jnp.stack([reduce_meta.src_rows, reduce_meta.tokens], axis=0),
          "moe_routing",
      )
      count_total = packed_small[0]
      local_gs = packed_small[1 : 1 + e_local]
      reduce_meta = dataclasses.replace(
          reduce_meta,
          src_rows=packed_rows[0],
          tokens=packed_rows[1],
          blocks=packed_small[1 + e_local :],
      )
    else:
      packed_small = checkpoint_name(
          jnp.concatenate([count_total_raw[None], local_gs_raw], axis=0),
          "moe_routing",
      )
      count_total = packed_small[0]
      local_gs = packed_small[1 : 1 + e_local]
  else:
    local_gs = checkpoint_name(local_gs_raw, "moe_routing")
    count_total = checkpoint_name(count_total_raw, "moe_routing")
    if reduce_meta is not None:
      reduce_meta = jax.tree.map(
          lambda v: checkpoint_name(v, "moe_routing") if isinstance(v, jax.Array) else v,
          reduce_meta,
      )
  outs = moe_kernels.routed_moe_forked_3d_fp8(
      dispatched.tokens_ep,
      w_rows0,
      topk_weights_flat,
      wi,
      wo,
      local_gs,
      sort_indices,
      count_total,
      reduce_meta,
      cfg,
      capacity=capacity,
      return_rhs=return_weights,
      forward_tie=forward_tie,
      wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
      sublayer_idx=sublayer_idx,
      use_external_bwd_rw=use_external_bwd_rw,
  )
  if not (return_weights or use_external_bwd_rw):
    return (outs, forward_tie) if forward_tie is not None else outs
  partial_ep, *rest = outs
  if forward_tie is not None:
    return (partial_ep, forward_tie, *rest)
  return (partial_ep, *rest)


def moe_combine_and_add_base(
    partial_ep: jax.Array,
    base_residual: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Combines partial EP expert outputs on SC0 (`psum_scatter`) and adds `base_residual`."""
  combined_flat = ops.ep_token_combine(partial_ep, cfg)
  return base_residual + combined_flat.reshape(base_residual.shape)
