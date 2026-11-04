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

"""Pure-JAX Qwen3.5-397B-A17B top-level forward, loss, Optax AdamW, and `jit_train_step`."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax

from maxtext.experimental.qwen_35_64c import layers
from maxtext.experimental.qwen_35_64c import moe_kernels
from maxtext.experimental.qwen_35_64c import ops
from maxtext.experimental.qwen_35_64c import params as params_lib
from maxtext.experimental.qwen_35_64c import schedule
from maxtext.experimental.qwen_35_64c.config import Qwen35Config

_GDN_SAVED_NAMES = (
    "gdn",
    "gdn_core_attn_out",
    "gdn_fwd_out",
    "gdn_qkv",
    "gdn_b",
    "gdn_a",
    "gdn_t_inv",
    "gdn_chunk_states",
    "gdn_m_local",
    "gdn_conv_state",
    "gdn_recurrent_state",
    "gdn_conv",
    "gdn_conv_out",
)


def data_pspec(cfg: Qwen35Config) -> P:
  """`[B, S]` token data: batch over 'fsdp', sequence over the CP (== EP) group."""
  return P("fsdp", cfg.cp_axis_names)


def build_remat_policy(cfg: Qwen35Config):
  """Builds JAX checkpoint and `S(5)` pinned-host offload policy."""
  save_names: list[str] = ["fwd_tie_scalar"]
  if cfg.save_key_value_proj:
    save_names.extend(["key_proj", "value_proj"])
  if cfg.save_gdn_residuals:
    save_names.extend(_GDN_SAVED_NAMES)
  if cfg.save_gdn_cp_state:
    for name in ("gdn_cp_conv_halo", "gdn_cp_m_local", "gdn_cp_s_in"):
      if name not in save_names:
        save_names.append(name)
  if (
      cfg.save_moe_routing
      or cfg.save_moe_gate_up_sublayers
      or cfg.offload_moe_gate_up_sublayers
      or cfg.moe_router_dense_bwd
  ):
    save_names.append("moe_routing")

  offload_names: list[str] = []
  for idx in cfg.save_moe_gate_up_sublayers:
    name = f"moe_gate_up_{idx}"
    if name not in save_names:
      save_names.append(name)
  for idx in cfg.offload_moe_gate_up_sublayers:
    name = f"moe_gate_up_{idx}"
    target = save_names if cfg.hbm_save_offload_targets else offload_names
    if name not in save_names and name not in target:
      target.append(name)
  for idx in cfg.save_moe_disp_ep_sublayers:
    name = f"moe_disp_ep_{idx}"
    if name not in save_names:
      save_names.append(name)
  for idx in cfg.offload_moe_disp_ep_sublayers:
    if not cfg.hbm_save_offload_targets:
      raise ValueError(
          "XLA TPU S(5) pinned_host offload does not support float8_e4m3fn"
          " moe_disp_ep buffers (`cannot query the word size without a memory"
          " space`); use `save_moe_disp_ep_sublayers` (HBM) or"
          " `offload_moe_gate_up_sublayers` (bf16 S(5))."
      )
    name = f"moe_disp_ep_{idx}"
    if name not in save_names:
      save_names.append(name)

  if cfg.hbm_save_offload_targets:
    save_names.extend(
        [
            "decoder_layer_input",
            "context",
            "query_proj",
            "out_proj",
            "qkv_proj",
        ]
    )
  else:
    if cfg.save_decoder_input_hbm:
      save_names.append("decoder_layer_input")
    elif cfg.offload_decoder_input:
      offload_names.append("decoder_layer_input")
    if cfg.save_attn_context_hbm:
      save_names.append("context")
    elif cfg.offload_attn_context:
      offload_names.append("context")
    if cfg.save_query_proj_hbm:
      save_names.append("query_proj")
    elif cfg.offload_query_proj:
      offload_names.append("query_proj")
    if cfg.save_out_proj_hbm:
      save_names.append("out_proj")
    elif cfg.offload_out_proj:
      offload_names.append("out_proj")
    if cfg.offload_gdn_qkv:
      offload_names.append("qkv_proj")

  if offload_names:
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=save_names,
        names_which_can_be_offloaded=offload_names,
        offload_src="device",
        offload_dst="pinned_host",
    )
  if save_names:
    return jax.checkpoint_policies.save_only_these_names(*save_names)
  return jax.checkpoint_policies.nothing_saveable


def _compute_hoisted_superblock_absmax(
    blocks_local: params_lib.SuperblockParams,
    cfg: Qwen35Config,
) -> schedule.SuperblockAbsmax:
  """Precomputes local (or per-EP-group `pmax`) MoE FP8 `absmax` across all superblocks."""
  if not cfg.moe_fp8_weights:
    empty = schedule.SublayerAbsmax(wi=None, wo=None)
    return schedule.SuperblockAbsmax(gdn_0=empty, gdn_1=empty, gdn_2=empty, attn_3=empty)

  forked_ep_scalar = (
      cfg.moe_use_forked_3d_gmmv2
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and cfg.moe_use_ragged_silu_3d_fp8
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_gmm_compute
  )
  per_expert_scalar = cfg.moe_fp8_gmm_compute

  def _one_w_amax(w_stacked: jax.Array) -> jax.Array:
    if forked_ep_scalar:
      # Single `[num_superblocks, 1, 1]` pmax over `'fsdp'` BEFORE `scan` eliminates
      # 184 per-layer SC1 `scale` all-gathers inside `scan`!
      amax_ep_local = jax.lax.stop_gradient(jnp.max(jnp.abs(w_stacked), axis=(1, 2, 3)).astype(jnp.float32))
      amax_ep = jax.lax.pmax(amax_ep_local, axis_name="fsdp")[:, None, None]
      return jax.lax.stop_gradient(amax_ep)
    return moe_kernels.compute_hoisted_moe_absmax(w_stacked, per_expert_scalar=per_expert_scalar)

  def _sub_amax(sub) -> schedule.SublayerAbsmax:
    return schedule.SublayerAbsmax(
        wi=_one_w_amax(sub.moe.wi),
        wo=_one_w_amax(sub.moe.wo),
    )

  return schedule.SuperblockAbsmax(
      gdn_0=_sub_amax(blocks_local.gdn_0),
      gdn_1=_sub_amax(blocks_local.gdn_1),
      gdn_2=_sub_amax(blocks_local.gdn_2),
      attn_3=_sub_amax(blocks_local.attn_3),
  )


def embed_tokens(
    tokens_local: jax.Array,
    token_embedder_local: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Embeds `[B_local, S]` token IDs using `token_embedder` (`P(None, ('fsdp', 'expert'))`)."""
  if cfg.embed_mode == "ep_all_to_all_fsdp_gather":
    bsz, seq_len = tokens_local.shape
    vocab_size, d_local = token_embedder_local.shape
    s2_ax = cfg.dense_stage2_axis_names
    s2_size = cfg.dense_stage2_size
    z_ep = cfg.ep_ici_size
    table_f = jax.lax.all_gather(token_embedder_local, s2_ax, axis=1, tiled=True)
    t_ep = jax.lax.all_gather(tokens_local, "expert", axis=0, tiled=False)
    y_flat = jnp.take(table_f.reshape(vocab_size, s2_size * d_local), t_ep, axis=0)
    y = y_flat.reshape(z_ep, bsz, seq_len, s2_size, d_local)
    y_full = jnp.squeeze(
        jax.lax.all_to_all(y, axis_name="expert", split_axis=0, concat_axis=4, tiled=True),
        axis=0,
    )
    return y_full.reshape(bsz, seq_len, cfg.emb_dim)

  if cfg.embed_mode == "vocab_all_to_all":
    t_ep = jax.lax.all_gather(tokens_local, "expert", axis=0, tiled=False)
    t_all = jax.lax.all_gather(t_ep, "fsdp", axis=0, tiled=False)
    y = jnp.take(token_embedder_local, t_all, axis=0)
    y_ep = jnp.squeeze(
        jax.lax.all_to_all(y, axis_name="expert", split_axis=1, concat_axis=4, tiled=True),
        axis=1,
    )
    y_full = jnp.squeeze(
        jax.lax.all_to_all(y_ep, axis_name="fsdp", split_axis=0, concat_axis=3, tiled=True),
        axis=0,
    )
    return y_full

  table_full = ops.gather_dense_weight(token_embedder_local, cfg, sharded_axis=1, is_fsdp_expert=True)
  return jnp.take(table_full, tokens_local, axis=0)


def chunked_lm_head_loss(
    hidden_local: jax.Array,
    decoder_norm_local: jax.Array,
    lm_head_local: jax.Array,
    targets_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Computes `decoder_norm` + 2D vocab-and-token-chunked cross-entropy on this device.

  Splits both vocabulary (`cfg.num_vocab_chunks`, e.g. `4 x 62,080`) and local
  tokens
  (`cfg.num_vocab_tiling`, e.g. `2 x 4,096`) so full `[8192, 248320]` (`3.79
  GiB`)
  weights or logits are NEVER materialized, saves only `32 KiB` per-token
  `logsumexp`,
  and uses fused elementwise compare-select (`ZERO` scatter offload) in
  backward!
  """
  norm_scale = ops.replicate_small_param(decoder_norm_local, cfg)
  normed = ops.rms_norm_zero_centered(
      hidden_local,
      norm_scale,
      cfg.norm_eps,
      use_pallas=(cfg.use_pallas_kernels and cfg.use_pallas_rmsnorm),
  )

  bsz, seq_len, emb_dim = normed.shape
  total_local_tokens = bsz * seq_len
  num_vchunks = cfg.num_vocab_chunks
  num_tiles = cfg.num_vocab_tiling
  vocab_size = cfg.vocab_size
  if total_local_tokens % num_tiles != 0:
    raise ValueError(f"total_local_tokens={total_local_tokens} must be divisible by" f" num_vocab_tiling={num_tiles}")
  if vocab_size % num_vchunks != 0:
    raise ValueError(f"vocab_size={vocab_size} must be divisible by" f" num_vocab_chunks={num_vchunks}")
  tile_tokens = total_local_tokens // num_tiles
  vocab_chunk = vocab_size // num_vchunks

  h_flat = normed.reshape(total_local_tokens, emb_dim)
  tgt_flat = targets_local.reshape(total_local_tokens)
  mask_flat = (segment_ids_local.reshape(total_local_tokens) > 0).astype(jnp.float32)

  local_valid = jnp.sum(mask_flat)
  global_valid = jnp.maximum(jax.lax.psum(local_valid, axis_name=cfg.all_sharded_axes), 1.0)
  denom = jax.lax.stop_gradient(global_valid)

  ag_core = ops._fsdp_ag_core_id(cfg, is_bwd_remat=False)  # pylint: disable=protected-access
  rs_core = ops._fsdp_rs_core_id(cfg)  # pylint: disable=protected-access
  s2_ax = cfg.dense_stage2_axis_names
  min_sc = cfg.sc_min_collective_bytes
  use_pipe = cfg.pipeline_lm_head_chunks and num_vchunks > 1

  def _gather_lm_chunk(w_c_l: jax.Array) -> jax.Array:
    w_ep = ops.sc_all_gather(w_c_l, "expert", core_id=ag_core, axis=0, min_sc_bytes=min_sc)
    return ops.sc_all_gather(w_ep, s2_ax, core_id=ag_core, axis=0, min_sc_bytes=min_sc)

  def _compute_lse_and_picked(h: jax.Array, w_l: jax.Array, safe_t: jax.Array):
    h_tiles = h.reshape(num_tiles, tile_tokens, emb_dim)
    t_tiles = safe_t.reshape(num_tiles, tile_tokens)
    w_chunks_l = jnp.swapaxes(w_l.reshape(w_l.shape[0], num_vchunks, vocab_chunk), 0, 1)

    m_init = jnp.full((num_tiles, tile_tokens), -jnp.inf, dtype=jnp.float32)
    s_init = jnp.zeros((num_tiles, tile_tokens), dtype=jnp.float32)
    p_init = jnp.zeros((num_tiles, tile_tokens), dtype=jnp.float32)

    def _fwd_chunk_math(m_prev, s_prev, p_prev, c_idx, w_c):
      v_off = c_idx * vocab_chunk
      m_list, s_list, p_list = [], [], []
      for ti in range(num_tiles):
        logits = jnp.matmul(h_tiles[ti], w_c, preferred_element_type=jnp.float32)
        m_cur = jnp.max(logits, axis=-1)
        m_new = jnp.maximum(m_prev[ti], m_cur)
        s_new = s_prev[ti] * jnp.exp(m_prev[ti] - m_new) + jnp.sum(jnp.exp(logits - m_new[:, None]), axis=-1)
        in_c = (t_tiles[ti] >= v_off) & (t_tiles[ti] < v_off + vocab_chunk)
        loc_t = jnp.where(in_c, t_tiles[ti] - v_off, 0)
        picked_c = jnp.squeeze(jnp.take_along_axis(logits, loc_t[:, None], axis=-1), axis=-1)
        p_new = p_prev[ti] + jnp.where(in_c, picked_c, 0.0)
        m_list.append(m_new)
        s_list.append(s_new)
        p_list.append(p_new)
      return jnp.stack(m_list), jnp.stack(s_list), jnp.stack(p_list)

    if use_pipe:
      w_first = _gather_lm_chunk(w_chunks_l[0])

      def _fwd_pipe_step(carry, xs):
        m_prev, s_prev, p_prev, w_cur = carry
        c_idx, w_next_l = xs
        w_next = _gather_lm_chunk(w_next_l)
        m_new, s_new, p_new = _fwd_chunk_math(m_prev, s_prev, p_prev, c_idx, w_cur)
        return (m_new, s_new, p_new, w_next), None

      (m_pen, s_pen, p_pen, w_last), _ = jax.lax.scan(
          _fwd_pipe_step,
          (m_init, s_init, p_init, w_first),
          (jnp.arange(num_vchunks - 1, dtype=jnp.int32), w_chunks_l[1:]),
      )
      m_fin, s_fin, p_fin = _fwd_chunk_math(m_pen, s_pen, p_pen, jnp.int32(num_vchunks - 1), w_last)
    else:

      def _fwd_vchunk(carry, xs):
        m_prev, s_prev, p_prev = carry
        c_idx, w_c_l = xs
        w_c = _gather_lm_chunk(w_c_l)
        return _fwd_chunk_math(m_prev, s_prev, p_prev, c_idx, w_c), None

      (m_fin, s_fin, p_fin), _ = jax.lax.scan(
          _fwd_vchunk,
          (m_init, s_init, p_init),
          (jnp.arange(num_vchunks, dtype=jnp.int32), w_chunks_l),
      )
    lse_tiles = m_fin + jnp.log(s_fin)
    return lse_tiles, p_fin

  @jax.custom_vjp
  def _tiled_loss(
      h: jax.Array,
      w_l: jax.Array,
      tgt: jax.Array,
      msk: jax.Array,
      inv_denom_val: jax.Array,
  ) -> jax.Array:
    safe_t = jnp.clip(tgt, 0, vocab_size - 1)
    lse_tiles, p_tiles = _compute_lse_and_picked(h, w_l, safe_t)
    local_sum = jnp.sum((lse_tiles.reshape(-1) - p_tiles.reshape(-1)) * msk)
    return local_sum * inv_denom_val

  def _tiled_loss_fwd(h, w_l, tgt, msk, inv_denom_val):
    safe_t = jnp.clip(tgt, 0, vocab_size - 1)
    lse_tiles, p_tiles = _compute_lse_and_picked(h, w_l, safe_t)
    out = jnp.sum((lse_tiles.reshape(-1) - p_tiles.reshape(-1)) * msk) * inv_denom_val
    return out, (h, w_l, safe_t, msk, lse_tiles, inv_denom_val)

  def _tiled_loss_bwd(res, g_out: jax.Array):
    h, w_l, safe_t, msk, lse_tiles, inv_denom_val = res
    h_tiles = h.reshape(num_tiles, tile_tokens, emb_dim)
    t_tiles = safe_t.reshape(num_tiles, tile_tokens)
    scale_tiles = (g_out.astype(jnp.float32) * inv_denom_val * msk).reshape(num_tiles, tile_tokens)
    w_chunks_l = jnp.swapaxes(w_l.reshape(w_l.shape[0], num_vchunks, vocab_chunk), 0, 1)
    v_iota = jnp.arange(vocab_chunk, dtype=jnp.int32)[None, :]

    def _bwd_chunk_math(dh_tiles_prev: jax.Array, c_idx: jax.Array, w_c: jax.Array) -> tuple[jax.Array, jax.Array]:
      v_off = c_idx * vocab_chunk
      dw_c_acc = jnp.zeros((emb_dim, vocab_chunk), dtype=w_l.dtype)
      dh_list = []
      for ti in range(num_tiles):
        logits = jnp.matmul(h_tiles[ti], w_c, preferred_element_type=jnp.float32)
        probs = jnp.exp(logits - lse_tiles[ti][:, None])
        in_c = (t_tiles[ti] >= v_off) & (t_tiles[ti] < v_off + vocab_chunk)
        one_hot = in_c[:, None] & (v_iota == (t_tiles[ti] - v_off)[:, None])
        dlogits = ((probs - one_hot.astype(jnp.float32)) * scale_tiles[ti][:, None]).astype(h.dtype)
        dh_ti = dh_tiles_prev[ti] + jnp.matmul(dlogits, jnp.swapaxes(w_c, 0, 1)).astype(h.dtype)
        dw_step = jnp.matmul(jnp.swapaxes(h_tiles[ti], 0, 1), dlogits).astype(w_l.dtype)
        dw_c_acc = dw_c_acc + dw_step
        dh_list.append(dh_ti)

      dw_c_fsdp = ops.sc_psum_scatter(
          dw_c_acc,
          s2_ax,
          core_id=rs_core,
          scatter_dimension=0,
          min_sc_bytes=min_sc,
      )
      dw_c_local = ops.sc_psum_scatter(
          dw_c_fsdp,
          "expert",
          core_id=rs_core,
          scatter_dimension=0,
          min_sc_bytes=min_sc,
      )
      return jnp.stack(dh_list), dw_c_local.astype(w_l.dtype)

    if use_pipe:
      w_first = _gather_lm_chunk(w_chunks_l[0])

      def _bwd_pipe_step(carry, xs):
        dh_prev, w_cur = carry
        c_idx, w_next_l = xs
        w_next = _gather_lm_chunk(w_next_l)
        dh_next, dw_cur_local = _bwd_chunk_math(dh_prev, c_idx, w_cur)
        return (dh_next, w_next), dw_cur_local

      (dh_penult, w_last), dw_head = jax.lax.scan(
          _bwd_pipe_step,
          (jnp.zeros_like(h_tiles), w_first),
          (jnp.arange(num_vchunks - 1, dtype=jnp.int32), w_chunks_l[1:]),
      )
      dh_tiles_fin, dw_last = _bwd_chunk_math(dh_penult, jnp.int32(num_vchunks - 1), w_last)
      dw_chunks_local = jnp.concatenate([dw_head, dw_last[None, ...]], axis=0)
    else:

      def _bwd_vchunk(dh_tiles_prev: jax.Array, xs):
        c_idx, w_c_l = xs
        w_c = _gather_lm_chunk(w_c_l)
        return _bwd_chunk_math(dh_tiles_prev, c_idx, w_c)

      dh_tiles_fin, dw_chunks_local = jax.lax.scan(
          _bwd_vchunk,
          jnp.zeros_like(h_tiles),
          (jnp.arange(num_vchunks, dtype=jnp.int32), w_chunks_l),
      )
    dh = dh_tiles_fin.reshape(total_local_tokens, emb_dim)
    dw_local = jnp.swapaxes(dw_chunks_local, 0, 1).reshape(w_l.shape)
    return dh, dw_local, None, None, None

  _tiled_loss.defvjp(_tiled_loss_fwd, _tiled_loss_bwd)
  return _tiled_loss(h_flat, lm_head_local, tgt_flat, mask_flat, 1.0 / denom)


def compute_moe_switch_aux_loss(
    sb_aux: layers.MoeAuxStats | jax.Array | None,
    targets_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Computes local contribution to Qwen3.5's Switch load-balancing loss.

  For each MoE layer `l in 0..L-1`, global expert fraction is
  `f_{l,e} = stop_gradient(psum(expert_counts_{l,e}) / N_valid)` (sums to
  `top_k`)
  and local router probability contribution is `s_{r,l,e} / N_valid`. Summing
  `coef * (E / L) * sum_{l,e} f_{l,e} * (s_{r,l,e} / N_valid)` across all mesh
  ranks in `make_value_and_grad_fn` produces exact global Switch loss value AND
  exact global `router_gate` / activation gradients with a single `[L, E]`
  forward-only `psum` outside `lax.scan`.
  """
  del targets_local
  if sb_aux is None or cfg.router_aux_loss_coef <= 0.0:
    return jnp.zeros((), dtype=jnp.float32)
  if isinstance(sb_aux, jax.Array):
    expert_counts = jax.lax.stop_gradient(sb_aux[..., 0, :])
    prob_sum = sb_aux[..., 1, :]
  else:
    expert_counts = sb_aux.expert_counts
    prob_sum = sb_aux.prob_sum
  mask_flat = (segment_ids_local.reshape(-1) > 0).astype(jnp.float32)
  global_valid = jnp.maximum(jax.lax.psum(jnp.sum(mask_flat), axis_name=cfg.all_sharded_axes), 1.0)
  denom = jax.lax.stop_gradient(global_valid)
  global_counts = jax.lax.stop_gradient(jax.lax.psum(expert_counts, axis_name=cfg.all_sharded_axes))
  f_expert = jax.lax.stop_gradient(global_counts / denom)
  scale = float(cfg.router_aux_loss_coef) * (float(cfg.num_experts) / float(cfg.num_layers))
  return jnp.sum(f_expert * (prob_sum / denom)) * scale


def _slice_sb_range(tree: Any, start: int, stop: int) -> Any:
  return jax.tree.map(lambda a: a[start:stop] if a is not None else None, tree)


def _index_sb(tree: Any, idx: int) -> Any:
  return jax.tree.map(lambda a: a[idx] if a is not None else None, tree)


def _concat_sb_aux(scan_aux: Any, tail_aux_list: list[Any], *, has_scan: bool) -> Any:
  """Concatenates scanned and peeled-tail superblock auxiliary outputs."""
  if not tail_aux_list:
    return scan_aux
  if tail_aux_list[0] is None:
    return None
  tail_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *tail_aux_list)
  if not has_scan:
    return tail_stacked
  if scan_aux is None:
    return None
  return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), scan_aux, tail_stacked)


def local_forward_loss(
    params_local: params_lib.ModelParams,
    tokens_local: jax.Array,
    targets_local: jax.Array,
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: Qwen35Config,
) -> jax.Array:
  """Executes single-device `shard_map` body returning local contribution to global loss."""
  x = embed_tokens(tokens_local, params_local.token_embedder, cfg)
  hoisted_absmax = _compute_hoisted_superblock_absmax(params_local.blocks, cfg)
  remat_policy = build_remat_policy(cfg)
  n_peel = cfg.peel_tail_hbm_superblocks
  n_scan = cfg.num_superblocks - n_peel
  if n_peel > 0:
    tail_save_gate_up = tuple(sorted(set(cfg.save_moe_gate_up_sublayers) | set(cfg.offload_moe_gate_up_sublayers)))
    tail_cfg = dataclasses.replace(
        cfg,
        save_moe_gate_up_sublayers=tail_save_gate_up,
        offload_moe_gate_up_sublayers=(),
    )
    tail_policy = build_remat_policy(tail_cfg)
  else:
    tail_cfg = cfg
    tail_policy = remat_policy

  # Attention runs in dual-chunk load-balanced CP order: hoist the positions permute.
  positions_lb = layers.cp_natural_to_lb(positions_local, cfg)

  if cfg.schedule_mode == "unsplit":
    rope_full = (
        layers.compute_rope_cos_sin(positions_lb, cfg.rotary_dim, cfg.rope_max_timescale)
        if cfg.precompute_rope_tables
        else None
    )

    def _make_unsplit_fn(step_cfg: Qwen35Config, step_policy):
      def _unsplit_step(x_c, sb_inputs):
        sb_p, sb_a = sb_inputs
        return schedule.run_superblock_unsplit(
            x_c,
            sb_p,
            sb_a,
            positions_lb,
            segment_ids_local,
            step_cfg,
            rope_cos_sin=rope_full,
        )

      return (
          jax.checkpoint(_unsplit_step, policy=step_policy, prevent_cse=False) if step_cfg.enable_remat else _unsplit_step
      )

    all_xs_u = (params_local.blocks, hoisted_absmax)
    x_curr = x
    sb_aux_scan = None
    if n_scan > 0:
      scan_fn = _make_unsplit_fn(cfg, remat_policy)
      x_curr, sb_aux_scan = jax.lax.scan(scan_fn, x_curr, _slice_sb_range(all_xs_u, 0, n_scan))
    tail_aux_list = []
    if n_peel > 0:
      tail_fn = _make_unsplit_fn(tail_cfg, tail_policy)
      for sb_idx in range(n_scan, cfg.num_superblocks):
        x_curr, aux_t = tail_fn(x_curr, _index_sb(all_xs_u, sb_idx))
        tail_aux_list.append(aux_t)
    x_out = x_curr
    sb_aux_all = _concat_sb_aux(sb_aux_scan, tail_aux_list, has_scan=(n_scan > 0))
  elif cfg.schedule_mode == "batchsplit":
    if x.shape[0] % 2 != 0:
      raise ValueError(f"batchsplit requires an even local batch, got {x.shape[0]}")
    half_b = x.shape[0] // 2
    x_mb0, x_mb1_init = x[:half_b], x[half_b:]
    pos_mb0, pos_mb1 = positions_lb[:half_b], positions_lb[half_b:]
    seg_mb0, seg_mb1 = segment_ids_local[:half_b], segment_ids_local[half_b:]
    rope_mb0 = (
        layers.compute_rope_cos_sin(pos_mb0, cfg.rotary_dim, cfg.rope_max_timescale)
        if cfg.precompute_rope_tables
        else None
    )
    rope_mb1 = (
        layers.compute_rope_cos_sin(pos_mb1, cfg.rotary_dim, cfg.rope_max_timescale)
        if cfg.precompute_rope_tables
        else None
    )

    t_ep_mb = half_b * x.shape[1] * cfg.ep_size
    use_3d_ep = cfg.moe_use_tc_ragged and cfg.moe_3d_ep_tokens and (cfg.emb_dim % 128 == 0)
    unperm_shape = (t_ep_mb, cfg.emb_dim // 128, 128) if use_3d_ep else (t_ep_mb, cfg.emb_dim)
    unperm_zero = jnp.zeros(unperm_shape, dtype=x.dtype)
    use_bwd_rw3_prefetch = (
        cfg.bwd_prefetch_prev_attn3_moe_weights
        and cfg.fsdp_pipeline_mode == "full_prefetch_defer_rs"
        and cfg.moe_fp8_weights
        and cfg.moe_use_forked_3d_gmmv2
        and ((not cfg.enable_remat) or (3 in cfg.save_moe_gate_up_sublayers) or (3 in cfg.offload_moe_gate_up_sublayers))
    )
    if use_bwd_rw3_prefetch:
      assert hoisted_absmax.attn_3.wi is not None and hoisted_absmax.attn_3.wo is not None
      attn3_wi_all_sg = jax.lax.stop_gradient(params_local.blocks.attn_3.moe.wi)
      attn3_wo_all_sg = jax.lax.stop_gradient(params_local.blocks.attn_3.moe.wo)
      attn3_wi_amax_all_sg = jax.lax.stop_gradient(hoisted_absmax.attn_3.wi)
      attn3_wo_amax_all_sg = jax.lax.stop_gradient(hoisted_absmax.attn_3.wo)
      init_bwd_rw3 = ops.zero_bwd_rw_prefetch(
          *moe_kernels.bwd_rw3_prefetch_shapes(cfg, amax_ndim=attn3_wi_amax_all_sg.ndim - 1)
      )
      wi_c_all, wi_s_all = jax.vmap(lambda w, a: moe_kernels.prequantize_routed_weight_local(w, a, cfg))(
          attn3_wi_all_sg, attn3_wi_amax_all_sg
      )
      wo_c_all, wo_s_all = jax.vmap(lambda w, a: moe_kernels.prequantize_routed_weight_local(w, a, cfg))(
          attn3_wo_all_sg, attn3_wo_amax_all_sg
      )

      def _shift_prev(arr: jax.Array) -> jax.Array:
        return jnp.concatenate([arr[:1], arr[:-1]], axis=0)

      prev_rw3_all = (
          _shift_prev(wi_c_all),
          _shift_prev(wi_s_all),
          _shift_prev(wo_c_all),
          _shift_prev(wo_s_all),
      )
      last_rw3_local = (wi_c_all[-1], wi_s_all[-1], wo_c_all[-1], wo_s_all[-1])
    else:
      init_bwd_rw3 = None
      prev_rw3_all = None
      last_rw3_local = None

    init_carry = schedule.BatchSplitCarry(
        x_mb0=x_mb0,
        unperm_mb1=unperm_zero,
        base_mb1=x_mb1_init,
        bwd_rw3_prefetch=init_bwd_rw3,
    )

    def _make_batchsplit_fn(step_cfg: Qwen35Config, step_policy):
      def _batchsplit_step(carry_c: schedule.BatchSplitCarry, sb_inputs):
        sb_p, sb_a, prev_rw3_l = sb_inputs
        return schedule.run_superblock_batchsplit(
            carry_c,
            sb_p,
            sb_a,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            step_cfg,
            rope_mb0=rope_mb0,
            rope_mb1=rope_mb1,
            prev_rw3_local=prev_rw3_l,
            use_attn3_external_bwd_rw=use_bwd_rw3_prefetch,
        )

      if step_cfg.enable_remat:
        return jax.checkpoint(_batchsplit_step, policy=step_policy, prevent_cse=False)
      return _batchsplit_step

    all_xs_bs = (params_local.blocks, hoisted_absmax, prev_rw3_all)
    carry_curr = init_carry
    sb_aux_scan = None
    if n_scan > 0:
      scan_fn = _make_batchsplit_fn(cfg, remat_policy)
      carry_curr, sb_aux_scan = jax.lax.scan(scan_fn, carry_curr, _slice_sb_range(all_xs_bs, 0, n_scan))
    tail_aux_list = []
    if n_peel > 0:
      tail_fn = _make_batchsplit_fn(tail_cfg, tail_policy)
      for sb_idx in range(n_scan, cfg.num_superblocks):
        carry_curr, aux_t = tail_fn(carry_curr, _index_sb(all_xs_bs, sb_idx))
        tail_aux_list.append(aux_t)
    final_carry = carry_curr
    sb_aux_all = _concat_sb_aux(sb_aux_scan, tail_aux_list, has_scan=(n_scan > 0))
    # Epilogue: drain final superblock's deferred mb1 EP Combine on SC0!
    x_mb1_final = layers.moe_combine_and_add_base(final_carry.unperm_mb1, final_carry.base_mb1, cfg)
    x_out = jnp.concatenate([final_carry.x_mb0, x_mb1_final], axis=0)
    if use_bwd_rw3_prefetch:
      assert final_carry.bwd_rw3_prefetch is not None and last_rw3_local is not None
      x_out = moe_kernels.attach_last_bwd_rw3(
          x_out,
          final_carry.bwd_rw3_prefetch,
          last_rw3_local,
          cfg,
      )
  else:
    raise ValueError(f"Unknown schedule_mode={cfg.schedule_mode!r}")

  ce_loss = chunked_lm_head_loss(
      x_out,
      params_local.decoder_norm,
      params_local.lm_head,
      targets_local,
      segment_ids_local,
      cfg,
  )
  aux_loss = compute_moe_switch_aux_loss(sb_aux_all, targets_local, segment_ids_local, cfg)
  return ce_loss + aux_loss


def create_optimizer(cfg: Qwen35Config) -> optax.GradientTransformation:
  """Creates pure Optax AdamW or stateless SGD optimizer."""
  lr: float | optax.Schedule = cfg.learning_rate
  if cfg.lr_warmup_steps > 0:
    lr = optax.warmup_constant_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.lr_warmup_steps,
    )
  if cfg.optimizer == "sgd":
    if cfg.grad_clip_norm > 0.0:
      return optax.chain(
          optax.clip_by_global_norm(cfg.grad_clip_norm),
          optax.sgd(learning_rate=lr),
      )
    return optax.sgd(learning_rate=lr)

  mu_dt = jnp.bfloat16 if cfg.mu_dtype == "bfloat16" else jnp.float32

  def _wd_mask(params: params_lib.ModelParams):
    def _is_decayed(path, leaf):
      del leaf
      p_str = jax.tree_util.keystr(path)
      if any(
          k in p_str
          for k in (
              "norm",
              "conv1d",
              "A_log",
              "dt_bias",
              "token_embedder",
          )
      ):
        return False
      return True

    return jax.tree_util.tree_map_with_path(_is_decayed, params)

  tx = optax.adamw(
      learning_rate=lr,
      b1=cfg.adam_b1,
      b2=cfg.adam_b2,
      eps=cfg.adam_eps,
      eps_root=cfg.adam_eps_root,
      weight_decay=cfg.adam_weight_decay,
      mu_dtype=mu_dt,
      mask=_wd_mask,
  )
  if cfg.grad_clip_norm > 0.0:
    tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), tx)
  return tx


def get_abstract_opt_state(
    cfg: Qwen35Config,
    mesh: Mesh,
    abstract_params: params_lib.ModelParams,
) -> Any:
  """Evaluates abstract Optax optimizer state with exact parameter shardings."""
  tx = create_optimizer(cfg)
  param_shardings = jax.tree.map(lambda x: x.sharding, abstract_params)
  raw_shapes = jax.eval_shape(tx.init, abstract_params)
  scalar_sharding = NamedSharding(mesh, P())

  def _assign_sharding(path, leaf):
    if getattr(leaf, "ndim", 0) == 0:
      return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=scalar_sharding)
    # Match corresponding leaf in `abstract_params` by shape when inside mu/nu
    p_str = jax.tree_util.keystr(path)
    for p_path, p_leaf in jax.tree.leaves_with_path(abstract_params):
      if p_str.endswith(jax.tree_util.keystr(p_path)) and leaf.shape == p_leaf.shape:
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=p_leaf.sharding)
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=scalar_sharding)

  return jax.tree_util.tree_map_with_path(_assign_sharding, raw_shapes)


def init_opt_state(
    cfg: Qwen35Config,
    mesh: Mesh,
    params: params_lib.ModelParams,
) -> Any:
  """Initializes sharded Optax state directly on device (multi-host safe)."""
  tx = create_optimizer(cfg)
  param_pspecs = params_lib.get_param_pspecs(cfg)
  param_shardings = jax.tree.map(lambda ps: NamedSharding(mesh, ps), param_pspecs)
  abstract_params = params_lib.get_abstract_params(cfg, mesh)
  abstract_opt_state = get_abstract_opt_state(cfg, mesh, abstract_params)
  opt_shardings = jax.tree.map(lambda x: x.sharding, abstract_opt_state)
  return jax.jit(
      tx.init,
      in_shardings=(param_shardings,),
      out_shardings=opt_shardings,
  )(params)


def init_synthetic_batch(
    cfg: Qwen35Config,
    mesh: Mesh,
    *,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Creates sharded synthetic `(tokens, targets, positions, segment_ids)` on device."""
  pspec = data_pspec(cfg)
  data_sharding = NamedSharding(mesh, pspec)
  local_b = cfg.local_batch_size
  local_s = cfg.local_seq_len
  vocab_size = cfg.vocab_size

  def _local_batch():
    dev_id = jnp.int32(0)
    for ax in cfg.all_sharded_axes:
      dev_id = dev_id * jnp.int32(mesh.shape[ax]) + jax.lax.axis_index(ax)
    k = jax.random.fold_in(jax.random.PRNGKey(seed), dev_id)
    k_tok, k_tgt = jax.random.split(k)
    tokens = jax.random.randint(k_tok, (local_b, local_s), minval=1, maxval=vocab_size, dtype=jnp.int32)
    targets = jax.random.randint(k_tgt, (local_b, local_s), minval=1, maxval=vocab_size, dtype=jnp.int32)
    seq_off = jax.lax.axis_index(cfg.cp_axis_names) * local_s
    positions = jnp.broadcast_to(
        (seq_off + jnp.arange(local_s, dtype=jnp.int32))[None, :],
        (local_b, local_s),
    )
    segment_ids = jnp.ones((local_b, local_s), dtype=jnp.int32)
    return tokens, targets, positions, segment_ids

  smap_fn = jax.shard_map(
      _local_batch,
      mesh=mesh,
      in_specs=(),
      out_specs=(pspec, pspec, pspec, pspec),
      check_vma=False,
  )
  return jax.jit(
      smap_fn,
      out_shardings=(
          data_sharding,
          data_sharding,
          data_sharding,
          data_sharding,
      ),
  )()


def make_value_and_grad_fn(cfg: Qwen35Config, mesh: Mesh):
  """Returns `shard_map`-wrapped `(loss, grads) = vg_fn(params, tokens, targets, positions, segment_ids)`."""
  param_pspecs = params_lib.get_param_pspecs(cfg)
  all_axes = cfg.all_sharded_axes
  d_pspec = data_pspec(cfg)
  ga_steps = cfg.grad_accum_steps
  if ga_steps < 1 or cfg.local_batch_size % ga_steps != 0:
    raise ValueError(f"local_batch_size={cfg.local_batch_size} must be divisible by" f" grad_accum_steps={ga_steps}")

  def _single_vg(p_l, tok_l, tgt_l, pos_l, seg_l):
    local_contrib, grads_l = jax.value_and_grad(local_forward_loss)(p_l, tok_l, tgt_l, pos_l, seg_l, cfg)
    global_loss = jax.lax.psum(local_contrib, axis_name=all_axes)
    return global_loss, grads_l

  def _local_vg(p_l, tok_l, tgt_l, pos_l, seg_l):
    if ga_steps == 1:
      return _single_vg(p_l, tok_l, tgt_l, pos_l, seg_l)
    mb = tok_l.shape[0] // ga_steps
    xs = tuple(arr.reshape(ga_steps, mb, *arr.shape[1:]) for arr in (tok_l, tgt_l, pos_l, seg_l))
    acc_dt = jnp.float32 if cfg.grad_accum_dtype == "float32" else jnp.bfloat16
    inv_ga = jnp.asarray(1.0 / float(ga_steps), dtype=jnp.float32)
    init_loss = jnp.zeros((), dtype=jnp.float32)
    init_grads = jax.tree.map(
        lambda x: jnp.zeros(
            x.shape,
            dtype=jnp.float32 if x.dtype == jnp.float32 else acc_dt,
        ),
        p_l,
    )

    def _ga_step(carry, mb_inputs):
      loss_acc, grads_acc = carry
      tok_i, tgt_i, pos_i, seg_i = mb_inputs
      loss_i, grads_i = _single_vg(p_l, tok_i, tgt_i, pos_i, seg_i)
      loss_acc = loss_acc + loss_i.astype(jnp.float32) * inv_ga
      grads_acc = jax.tree.map(
          lambda acc, g: (acc.astype(jnp.float32) + g.astype(jnp.float32) * inv_ga).astype(acc.dtype),
          grads_acc,
          grads_i,
      )
      return (loss_acc, grads_acc), None

    (total_loss, total_grads), _ = jax.lax.scan(_ga_step, (init_loss, init_grads), xs)
    total_grads = jax.tree.map(lambda g, p: g.astype(p.dtype), total_grads, p_l)
    return total_loss, total_grads

  return jax.shard_map(
      _local_vg,
      mesh=mesh,
      in_specs=(param_pspecs, d_pspec, d_pspec, d_pspec, d_pspec),
      out_specs=(P(), param_pspecs),
      check_vma=False,
  )


def make_train_step(
    cfg: Qwen35Config,
    mesh: Mesh,
    *,
    donate_state: bool = True,
    return_grad_norm: bool = False,
):
  """Builds top-level `@jax.jit` `train_step` (dumps XLA module `jit_train_step`)."""
  tx = create_optimizer(cfg)
  vg_fn = make_value_and_grad_fn(cfg, mesh)

  param_pspecs = params_lib.get_param_pspecs(cfg)
  param_shardings = jax.tree.map(lambda ps: NamedSharding(mesh, ps), param_pspecs)
  abstract_params = params_lib.get_abstract_params(cfg, mesh)
  abstract_opt_state = get_abstract_opt_state(cfg, mesh, abstract_params)
  opt_shardings = jax.tree.map(lambda x: x.sharding, abstract_opt_state)
  data_sharding = NamedSharding(mesh, data_pspec(cfg))
  scalar_sharding = NamedSharding(mesh, P())
  donate = (0, 1) if donate_state else ()

  if return_grad_norm:

    def train_step(params, opt_state, tokens, targets, positions, segment_ids):
      loss, grads = vg_fn(params, tokens, targets, positions, segment_ids)
      sq_norms = [jnp.sum(jnp.square(g.astype(jnp.float32))) for g in jax.tree.leaves(grads)]
      grad_norm = jnp.sqrt(jnp.sum(jnp.stack(sq_norms)))
      updates, new_opt_state = tx.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, loss, grad_norm

    out_shardings: tuple[Any, ...] = (
        param_shardings,
        opt_shardings,
        scalar_sharding,
        scalar_sharding,
    )
  else:

    def train_step(params, opt_state, tokens, targets, positions, segment_ids):
      loss, grads = vg_fn(params, tokens, targets, positions, segment_ids)
      updates, new_opt_state = tx.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, loss

    out_shardings = (param_shardings, opt_shardings, scalar_sharding)

  return jax.jit(
      train_step,
      in_shardings=(
          param_shardings,
          opt_shardings,
          data_sharding,
          data_sharding,
          data_sharding,
          data_sharding,
      ),
      out_shardings=out_shardings,
      donate_argnums=donate,
  )
