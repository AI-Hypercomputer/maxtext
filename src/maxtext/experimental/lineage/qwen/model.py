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

"""Pure-JAX Qwen 3.5 / Qwen MAX model, remat, chunked CE, Optax step, & MFU helpers."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import optax

from maxtext.experimental.lineage.qwen import layers
from maxtext.experimental.lineage.qwen import ops
from maxtext.experimental.lineage.qwen import params as params_lib
from maxtext.experimental.lineage.qwen.config import PEAK_TFLOPS_PER_CHIP_BF16_V7
from maxtext.experimental.lineage.qwen.config import QwenConfig

_PKG = "maxtext.experimental.lineage.qwen"

_GDN_SAVED_NAMES: tuple[str, ...] = (
    "gdn",
    "gdn_core_attn_out",
    "gdn_fwd_out",
    "gdn_fwd_conv",
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
    "qkv_proj",
    "out_proj",
)


def _maybe_peer(name: str) -> Any | None:
  try:
    return importlib.import_module(f"{_PKG}.{name}")
  except ImportError:
    return None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class OptimizerState:
  """Pure-JAX optimizer state supporting optional `fp32` master weights."""

  step: jax.Array
  master_params: Any
  inner_opt_state: Any


def _shard_map(fn, *, mesh: Mesh, in_specs: Any, out_specs: Any):
  sm_fn = getattr(jax, "shard_map", None)
  if sm_fn is None:
    from jax.experimental import shard_map as exp_sm  # pylint: disable=g-import-not-at-top

    sm_fn = exp_sm.shard_map
  return sm_fn(
      fn,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      check_vma=False,
  )


# ---------------------------------------------------------------------------
# Checkpoint / Remat Policy (Zero S(5) Offload by Default)
# ---------------------------------------------------------------------------


def build_remat_policy(cfg: QwenConfig):
  """Builds JAX checkpoint policy honoring `save_decoder_input_hbm` (zero `S(5)` default)."""
  save_names: list[str] = ["fwd_tie_scalar"]
  offload_names: list[str] = []

  def _route(name: str, *, save_hbm: bool, offload: bool) -> None:
    if save_hbm:
      if name not in save_names:
        save_names.append(name)
    elif offload:
      target = save_names if cfg.hbm_save_offload_targets else offload_names
      if name not in save_names and name not in target:
        target.append(name)

  _route(
      "decoder_layer_input",
      save_hbm=cfg.save_decoder_input_hbm,
      offload=cfg.offload_decoder_input,
  )
  _route(
      "context",
      save_hbm=cfg.save_attn_context_hbm,
      offload=cfg.offload_attn_context,
  )
  _route(
      "query_proj",
      save_hbm=cfg.save_query_proj_hbm,
      offload=cfg.offload_query_proj,
  )
  _route(
      "out_proj",
      save_hbm=cfg.save_out_proj_hbm,
      offload=cfg.offload_out_proj,
  )

  if cfg.save_key_value_proj:
    for nm in ("key_proj", "value_proj"):
      if nm not in save_names:
        save_names.append(nm)

  if cfg.save_gdn_residuals:
    for nm in _GDN_SAVED_NAMES:
      if nm not in save_names:
        save_names.append(nm)
  elif cfg.offload_gdn_qkv:
    _route("gdn_qkv", save_hbm=False, offload=True)

  if (
      cfg.save_moe_routing
      or cfg.save_moe_gate_up_sublayers
      or cfg.offload_moe_gate_up_sublayers
      or cfg.save_moe_disp_ep_sublayers
      or cfg.offload_moe_disp_ep_sublayers
  ):
    if "moe_routing" not in save_names:
      save_names.append("moe_routing")

  if cfg.moe_save_bank_in_superblock and "moe_act_bank" not in save_names:
    save_names.append("moe_act_bank")

  for idx in cfg.save_moe_gate_up_sublayers:
    nm = f"moe_gate_up_{idx}"
    if nm not in save_names:
      save_names.append(nm)
  for idx in cfg.offload_moe_gate_up_sublayers:
    _route(f"moe_gate_up_{idx}", save_hbm=False, offload=True)

  for idx in cfg.save_moe_disp_ep_sublayers:
    nm = f"moe_disp_ep_{idx}"
    if nm not in save_names:
      save_names.append(nm)
  for idx in cfg.offload_moe_disp_ep_sublayers:
    _route(f"moe_disp_ep_{idx}", save_hbm=False, offload=True)

  if offload_names:
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=save_names,
        names_which_can_be_offloaded=offload_names,
        offload_src="device",
        offload_dst="pinned_host",
    )
  return jax.checkpoint_policies.save_only_these_names(*save_names)


# ---------------------------------------------------------------------------
# Hoisted Superblock Absmax & Embeddings & Chunked Cross-Entropy Loss
# ---------------------------------------------------------------------------


def _compute_hoisted_superblock_absmax(
    blocks_local: params_lib.SuperblockParams,
    cfg: QwenConfig,
):
  """Precomputes stacked `SuperblockAbsmax` outside `scan` when `moe_fp8_weights=True`."""
  sched_mod = _maybe_peer("schedule")
  moe_kernels = _maybe_peer("moe_kernels") if cfg.use_pallas_kernels else None

  def _make_sub_absmax(moe_p: params_lib.MoeParams):
    if (
        cfg.moe_fp8_weights
        and moe_kernels is not None
        and hasattr(moe_kernels, "compute_hoisted_moe_absmax")
    ):
      wi_amax = moe_kernels.compute_hoisted_moe_absmax(moe_p.wi, cfg)
      wo_amax = moe_kernels.compute_hoisted_moe_absmax(moe_p.wo, cfg)
    else:
      wi_amax = None
      wo_amax = None
    if sched_mod is not None and hasattr(sched_mod, "SublayerAbsmax"):
      return sched_mod.SublayerAbsmax(wi=wi_amax, wo=wo_amax)
    return (wi_amax, wo_amax)

  sub_0 = _make_sub_absmax(blocks_local.gdn_0.moe)
  sub_1 = _make_sub_absmax(blocks_local.gdn_1.moe)
  sub_2 = _make_sub_absmax(blocks_local.gdn_2.moe)
  sub_3 = _make_sub_absmax(blocks_local.attn_3.moe)
  if sched_mod is not None and hasattr(sched_mod, "SuperblockAbsmax"):
    return sched_mod.SuperblockAbsmax(
        gdn_0=sub_0, gdn_1=sub_1, gdn_2=sub_2, attn_3=sub_3
    )
  return (sub_0, sub_1, sub_2, sub_3)


@jax.custom_vjp
def _gather_embed_rows(table: jax.Array, idx: jax.Array) -> jax.Array:
  """Gathers rows `table[idx]` with a scatter-free chunked one-hot VJP."""
  return table[idx]


def _gather_embed_rows_fwd(
    table: jax.Array, idx: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  # Pass a zero-size dummy array to carry `table.shape` / `table.dtype`
  # without putting a raw `np.dtype` object into JAX `custom_vjp` residuals.
  dummy = jax.lax.broadcast(jnp.zeros((), dtype=table.dtype), table.shape)
  return table[idx], (dummy, idx)


def _gather_embed_rows_bwd(
    res: tuple[jax.Array, jax.Array], g_out: jax.Array
) -> tuple[jax.Array, None]:
  dummy, idx = res
  vocab_size, d_dim = dummy.shape
  table_dtype = dummy.dtype
  idx_flat = jax.lax.stop_gradient(idx.reshape(-1)).astype(jnp.int32)
  g_flat = g_out.reshape(-1, d_dim)
  n_tok = idx_flat.shape[0]
  if vocab_size < 8192:
    one_hot = jax.nn.one_hot(idx_flat, vocab_size, dtype=jnp.float32)
    grad_table = jnp.matmul(
        one_hot.T, g_flat.astype(jnp.float32)
    ).astype(table_dtype)
    return grad_table, None

  # Sort tokens by vocab ID on TensorCore (0 scatter, 0 SC offload), then
  # matmul only the active token range per `chunk_v=2048` vocab bucket.
  # Cuts embedding backward FLOPs ~80x while staying exact on arbitrary
  # duplicate-heavy token distributions.
  sorted_vid, sorted_src = jax.lax.sort(
      (idx_flat, jnp.arange(n_tok, dtype=jnp.int32)),
      dimension=0,
      is_stable=False,
      num_keys=1,
  )
  g_sorted = jnp.take(g_flat, sorted_src, axis=0)

  chunk_v = 2048
  n_chunks = (vocab_size + chunk_v - 1) // chunk_v
  w_tok = min(512, max(128, ((n_tok + 127) // 128) * 128))
  sent_vid = jnp.int32(n_chunks * chunk_v)
  sorted_vid_pad = jnp.pad(
      sorted_vid, (0, w_tok), mode="constant", constant_values=sent_vid
  )
  g_sorted_pad = jnp.pad(
      g_sorted, ((0, w_tok), (0, 0)), mode="constant", constant_values=0
  )
  boundaries = jnp.arange(n_chunks + 1, dtype=jnp.int32) * chunk_v
  chunk_starts = jnp.searchsorted(
      sorted_vid, boundaries, side="left"
  ).astype(jnp.int32)
  iota_w = jnp.arange(w_tok, dtype=jnp.int32)
  grad_buf = jnp.zeros((n_chunks * chunk_v, d_dim), dtype=table_dtype)

  def _chunk_body(c, buf):
    v0 = c * chunk_v
    s = chunk_starts[c]
    e = chunk_starts[c + 1]
    init_acc = jnp.zeros((chunk_v, d_dim), dtype=jnp.float32)

    def _cond(state):
      pos, _ = state
      return pos < e

    def _tile_body(state):
      pos, acc = state
      vid_tile = jax.lax.dynamic_slice_in_dim(
          sorted_vid_pad, pos, w_tok, axis=0
      )
      g_tile = jax.lax.dynamic_slice_in_dim(
          g_sorted_pad, pos, w_tok, axis=0
      )
      valid = (pos + iota_w) < e
      local_id = jnp.where(valid, vid_tile - v0, jnp.int32(-1))
      oh = jax.nn.one_hot(local_id, chunk_v, dtype=g_flat.dtype)
      acc = acc + jnp.matmul(
          oh.T, g_tile, preferred_element_type=jnp.float32
      )
      return pos + w_tok, acc

    _, acc = jax.lax.while_loop(_cond, _tile_body, (s, init_acc))
    return jax.lax.dynamic_update_slice(buf, acc.astype(table_dtype), (v0, 0))

  grad_buf = jax.lax.fori_loop(0, n_chunks, _chunk_body, grad_buf)
  return grad_buf[:vocab_size], None


_gather_embed_rows.defvjp(_gather_embed_rows_fwd, _gather_embed_rows_bwd)


def embed_tokens(
    tokens_local: jax.Array,
    token_embedder_local: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Sharded token embedding lookup supporting `ep_all_to_all_fsdp_gather`, `vocab_all_to_all`, and `sc_gather`."""
  bsz_local, seq_len = tokens_local.shape
  if cfg.num_devices == 1:
    return token_embedder_local[tokens_local]

  if cfg.embed_mode == "sc_gather":
    full_embed = ops.gather_dense_weight(
        token_embedder_local,
        cfg,
        sharded_axis=1,
        is_fsdp_expert=True,
        use_fp8=False,
    )
    return _gather_embed_rows(full_embed, tokens_local)

  if cfg.embed_mode == "vocab_all_to_all":
    tok_all = jax.lax.all_gather(
        jax.lax.stop_gradient(tokens_local),
        axis_name=cfg.all_sharded_axes,
        axis=0,
        tiled=False,
    )  # [N_dev, B_local, S]
    gathered_slices = _gather_embed_rows(
        token_embedder_local, tok_all
    )  # [N_dev, B_local, S, D_local]
    emb = jax.lax.all_to_all(
        gathered_slices,
        axis_name=cfg.all_sharded_axes,
        split_axis=0,
        concat_axis=3,
        tiled=True,
    )
    return emb.reshape(bsz_local, seq_len, cfg.emb_dim)

  # Default: "ep_all_to_all_fsdp_gather"
  # 1) SC1 FSDP stage-2 gather along D (axis=1) WITHOUT psum('expert') in bwd,
  #    because each 'expert' shard holds a disjoint slice of D!
  # 2) EP all-gather of int32 token IDs + EP all_to_all along D (whose bwd
  #    already accumulates across all 'expert' ranks into each shard's D slice).
  ag_core = 1 if cfg.fsdp_ag_sc_pinning in ("sc1", "sc1_fwd_only") else None
  rs_core = 1 if cfg.fsdp_rs_sc_pinning == "sc1" else None
  s2_ax = cfg.dense_stage2_axis_names
  min_b = cfg.sc_min_collective_bytes

  @jax.custom_vjp
  def _gather_s2_axis1(w_local: jax.Array) -> jax.Array:
    return ops.sc_all_gather(
        w_local, s2_ax, core_id=ag_core, axis=1, min_collective_bytes=min_b
    )

  def _gather_s2_axis1_fwd(w_local: jax.Array):
    return _gather_s2_axis1(w_local), None

  def _gather_s2_axis1_bwd(_, gw: jax.Array):
    ops.assert_bf16_or_fp32_reduction(gw, "embed_tokens_s2_bwd")
    return (
        ops.sc_psum_scatter(
            gw,
            s2_ax,
            core_id=rs_core,
            scatter_dimension=1,
            min_collective_bytes=min_b,
        ),
    )

  _gather_s2_axis1.defvjp(_gather_s2_axis1_fwd, _gather_s2_axis1_bwd)
  s2_gathered = _gather_s2_axis1(token_embedder_local)

  ep_ax_rem = "expert"
  if cfg.ep_ici_size <= 1:
    return _gather_embed_rows(s2_gathered, tokens_local)
  tok_ep = jax.lax.all_gather(
      jax.lax.stop_gradient(tokens_local),
      axis_name=ep_ax_rem,
      axis=0,
      tiled=False,
  )  # [ep_ici_size, B_local, S]
  slices = _gather_embed_rows(s2_gathered, tok_ep)  # [ep_ici_size, B_local, S, D // ep_ici_size]
  # `token_embedder` is sharded on `cfg.all_sharded_axes` (`('fsdp', 'expert',
  # 'ep_d2d')` when core-minor), while `s2_gathered` gathered `('fsdp',
  # 'ep_d2d')`. Place `'expert'` between `'fsdp'` and `'ep_d2d'` via
  # `all_to_all(..., split_axis=0, concat_axis=4)` when core-minor, or at the
  # end otherwise.
  d_chunk = cfg.emb_dim // cfg.num_devices
  if cfg.all_sharded_axes == ("fsdp", "expert", "ep_d2d"):
    ep_d2d_size = cfg.ep_size // cfg.ep_ici_size
    slices_6d = slices.reshape(
        cfg.ep_ici_size,
        bsz_local,
        seq_len,
        cfg.fsdp_size,
        ep_d2d_size,
        d_chunk,
    )
    emb = jax.lax.all_to_all(
        slices_6d,
        axis_name=ep_ax_rem,
        split_axis=0,
        concat_axis=4,
        tiled=True,
    )
  else:
    s2_size = cfg.num_devices // cfg.ep_ici_size
    slices_5d = slices.reshape(
        cfg.ep_ici_size, bsz_local, seq_len, s2_size, d_chunk
    )
    emb = jax.lax.all_to_all(
        slices_5d,
        axis_name=ep_ax_rem,
        split_axis=0,
        concat_axis=4,
        tiled=True,
    )
  return emb.reshape(bsz_local, seq_len, cfg.emb_dim)


def chunked_lm_head_loss(
    hidden_local: jax.Array,
    decoder_norm_local: jax.Array,
    lm_head_local: jax.Array,
    targets_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Memory-bounded vocab-chunked & token-tiled cross-entropy loss (supports dense FP8)."""
  norm_w = ops.replicate_small_param(decoder_norm_local, cfg)
  normed = ops.rms_norm_zero_centered(hidden_local, norm_w, cfg.norm_eps)
  t_total = hidden_local.shape[0] * hidden_local.shape[1]
  x_all = normed.reshape(t_total, cfg.emb_dim)
  tgt_all = targets_local.reshape(t_total)
  mask_all = (segment_ids_local.reshape(t_total) > 0).astype(jnp.float32)

  n_tiles = max(1, min(int(cfg.num_vocab_tiling), t_total))
  tile_bounds: list[tuple[int, int]] = []
  base_t = t_total // n_tiles
  rem_t = t_total % n_tiles
  t_cursor = 0
  for i in range(n_tiles):
    sz = base_t + (1 if i < rem_t else 0)
    tile_bounds.append((t_cursor, t_cursor + sz))
    t_cursor += sz

  v_total = cfg.vocab_size
  n_chunks = max(1, min(int(cfg.num_vocab_chunks), v_total))
  chunk_bounds: list[tuple[int, int]] = []
  base_v = v_total // n_chunks
  rem_v = v_total % n_chunks
  v_cursor = 0
  for i in range(n_chunks):
    sz = base_v + (1 if i < rem_v else 0)
    chunk_bounds.append((v_cursor, v_cursor + sz))
    v_cursor += sz

  fp8_lm = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_lm_head)

  def _eval_chunk(
      x_tile: jax.Array,
      tgt_tile: jax.Array,
      lm_chunk_sharded: jax.Array,
      v_start: int,
      v_size: int,
  ) -> tuple[jax.Array, jax.Array]:
    @jax.checkpoint
    def _ckpt_chunk(x_in: jax.Array, w_sharded: jax.Array):
      if cfg.num_devices > 1:
        w_full = ops.gather_dense_weight(
            w_sharded,
            cfg,
            sharded_axis=0,
            is_fsdp_expert=True,
            use_fp8=fp8_lm,
        )
      else:
        w_full = w_sharded
      quant_unsharded = fp8_lm and (cfg.num_devices == 1)
      logits = ops.dense_fp8_matmul(
          x_in, w_full, quantize_bf16_weight=quant_unsharded
      ).astype(jnp.float32)
      lse_c = jax.nn.logsumexp(logits, axis=-1)
      in_chunk = (tgt_tile >= v_start) & (tgt_tile < v_start + v_size)
      safe_idx = jnp.where(in_chunk, tgt_tile - v_start, 0)
      mask_1h = (
          jnp.arange(v_size, dtype=safe_idx.dtype)[None, :] == safe_idx[:, None]
      ) & in_chunk[:, None]
      tgt_logit_c = jnp.sum(jnp.where(mask_1h, logits, 0.0), axis=-1)
      return lse_c, tgt_logit_c

    return _ckpt_chunk(x_tile, lm_chunk_sharded)

  local_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
  local_tok_count = jnp.sum(mask_all)

  if cfg.vocab_tiling_ag_once:
    # Gather each vocab chunk once across all token tiles, while processing one
    # `[t_tile, v_chunk]` logit tile at a time in both forward and backward so
    # peak HBM never holds full `[t_total, v_chunk]` logits or multiple chunks.
    def _add_w_grad(a: Any, b: Any) -> Any:
      if isinstance(a, ops.Fp8DenseWeight):
        return ops.Fp8DenseWeight(
            w_ste=a.w_ste + b.w_ste,
            codes=a.codes,
            scale=a.scale,
        )
      return a + b

    def _eval_vocab_chunk(
        x_in_all: jax.Array,
        lm_chunk_sharded: jax.Array,
        v_start: int,
        v_size: int,
    ) -> tuple[jax.Array, jax.Array]:
      quant_unsharded = fp8_lm and (cfg.num_devices == 1)

      def _gather_w(w_sh: jax.Array) -> jax.Array | ops.Fp8DenseWeight:
        if cfg.num_devices > 1:
          return ops.gather_dense_weight(
              w_sh,
              cfg,
              sharded_axis=0,
              is_fsdp_expert=True,
              use_fp8=fp8_lm,
          )
        return w_sh

      def _mm(x_t: jax.Array, w_f: jax.Array | ops.Fp8DenseWeight) -> jax.Array:
        return ops.dense_fp8_matmul(
            x_t, w_f, quantize_bf16_weight=quant_unsharded
        )

      def _tile_mask_1h(tgt_t: jax.Array) -> jax.Array:
        in_chunk = (tgt_t >= v_start) & (tgt_t < v_start + v_size)
        safe_idx = jnp.where(in_chunk, tgt_t - v_start, 0)
        return (
            jnp.arange(v_size, dtype=safe_idx.dtype)[None, :]
            == safe_idx[:, None]
        ) & in_chunk[:, None]

      @jax.custom_vjp
      def _ckpt_vocab_chunk(
          x_in: jax.Array, w_sh: jax.Array, tgt_in: jax.Array
      ) -> tuple[jax.Array, jax.Array]:
        w_full = _gather_w(w_sh)
        lse_tiles: list[jax.Array] = []
        tgt_tiles: list[jax.Array] = []
        for t_start, t_end in tile_bounds:
          logits = _mm(x_in[t_start:t_end], w_full).astype(jnp.float32)
          lse_tiles.append(jax.nn.logsumexp(logits, axis=-1))
          mask_1h = _tile_mask_1h(tgt_in[t_start:t_end])
          tgt_tiles.append(jnp.sum(jnp.where(mask_1h, logits, 0.0), axis=-1))
        return jnp.concatenate(lse_tiles, axis=0), jnp.concatenate(
            tgt_tiles, axis=0
        )

      def _ckpt_vocab_chunk_fwd(
          x_in: jax.Array, w_sh: jax.Array, tgt_in: jax.Array
      ):
        lse_c, tgt_c = _ckpt_vocab_chunk(x_in, w_sh, tgt_in)
        return (lse_c, tgt_c), (x_in, w_sh, tgt_in, lse_c)

      def _ckpt_vocab_chunk_bwd(res, cotangents):
        x_in, w_sh, tgt_in, lse_c = res
        g_lse_all, g_tgt_all = cotangents
        w_full, gather_vjp = jax.vjp(_gather_w, w_sh)
        dx_tiles: list[jax.Array] = []
        dw_acc: Any = None
        for t_start, t_end in tile_bounds:
          x_t = x_in[t_start:t_end]
          tgt_t = tgt_in[t_start:t_end]
          lse_t = lse_c[t_start:t_end]
          g_lse_t = g_lse_all[t_start:t_end]
          g_tgt_t = g_tgt_all[t_start:t_end]
          logits_raw, mm_vjp = jax.vjp(_mm, x_t, w_full)
          logits_f32 = logits_raw.astype(jnp.float32)
          probs_t = jnp.exp(logits_f32 - lse_t[:, None])
          mask_1h = _tile_mask_1h(tgt_t)
          dlogits_f32 = g_lse_t[:, None] * probs_t + jnp.where(
              mask_1h, g_tgt_t[:, None], 0.0
          )
          dx_t, dw_t = mm_vjp(dlogits_f32.astype(logits_raw.dtype))
          dx_tiles.append(dx_t)
          dw_acc = dw_t if dw_acc is None else _add_w_grad(dw_acc, dw_t)
        dx_all = jnp.concatenate(dx_tiles, axis=0)
        (dw_sh,) = gather_vjp(dw_acc)
        return dx_all, dw_sh, None

      _ckpt_vocab_chunk.defvjp(_ckpt_vocab_chunk_fwd, _ckpt_vocab_chunk_bwd)
      return _ckpt_vocab_chunk(x_in_all, lm_chunk_sharded, tgt_all)

    lse_acc = jnp.full((t_total,), -jnp.inf, dtype=jnp.float32)
    tgt_logit = jnp.zeros((t_total,), dtype=jnp.float32)
    for v_start, v_end in chunk_bounds:
      lse_c, t_c = _eval_vocab_chunk(
          x_all,
          lm_head_local[:, v_start:v_end],
          v_start,
          v_end - v_start,
      )
      lse_acc = jnp.logaddexp(lse_acc, lse_c)
      tgt_logit = tgt_logit + t_c
    ce_all = lse_acc - tgt_logit
    local_loss_sum = jnp.sum(ce_all * mask_all)
  else:
    for t_start, t_end in tile_bounds:
      x_tile = x_all[t_start:t_end]
      tgt_tile = tgt_all[t_start:t_end]
      mask_tile = mask_all[t_start:t_end]
      lse_acc = jnp.full((t_end - t_start,), -jnp.inf, dtype=jnp.float32)
      tgt_logit = jnp.zeros((t_end - t_start,), dtype=jnp.float32)
      for v_start, v_end in chunk_bounds:
        lse_c, t_c = _eval_chunk(
            x_tile,
            tgt_tile,
            lm_head_local[:, v_start:v_end],
            v_start,
            v_end - v_start,
        )
        lse_acc = jnp.logaddexp(lse_acc, lse_c)
        tgt_logit = tgt_logit + t_c
      ce_tile = lse_acc - tgt_logit
      local_loss_sum = local_loss_sum + jnp.sum(ce_tile * mask_tile)

  if cfg.num_devices > 1:
    reduce_axes = (
        ("fsdp", "expert", "ep_d2d")
        if (cfg.decouple_ep_d2d and cfg.ep_core_minor)
        else cfg.all_sharded_axes
    )
    global_loss_sum = ops.psum_bf16_or_fp32(
        local_loss_sum, reduce_axes
    )
    global_tok_count = ops.psum_bf16_or_fp32(
        local_tok_count, reduce_axes
    )
    return global_loss_sum / jnp.maximum(global_tok_count, 1.0)
  return local_loss_sum / jnp.maximum(local_tok_count, 1.0)


# ---------------------------------------------------------------------------
# Pure-JAX Superblock & Full Model Forward / Loss
# ---------------------------------------------------------------------------


def _run_unsplit_sublayer_fallback(
    x: jax.Array,
    sub_params: params_lib.GdnSublayerParams | params_lib.AttnSublayerParams,
    sub_absmax: Any,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    sublayer_idx: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
    batch_offset: int = 0,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None,
) -> jax.Array:
  """Runs one unsplit GDN or Full-Attn + MoE sub-layer without Pallas schedule."""
  wi_amax, wo_amax = (
      (sub_absmax[0], sub_absmax[1])
      if isinstance(sub_absmax, tuple)
      else (getattr(sub_absmax, "wi", None), getattr(sub_absmax, "wo", None))
  )
  if is_attn:
    w = layers.gather_attn_sublayer_weights(sub_params, wi_amax, wo_amax, cfg)
    h = layers.full_attn_mixer(
        x, w, positions, segment_ids, cfg, rope_cos_sin=rope_cos_sin
    )
  else:
    w = layers.gather_gdn_sublayer_weights(sub_params, wi_amax, wo_amax, cfg)
    h = layers.gdn_mixer(x, w, segment_ids, cfg)

  normed = ops.rms_norm_zero_centered(h, w.post_attn_norm, cfg.norm_eps)
  base = layers.shared_expert_and_residual(
      h, normed, w.moe, float32_gate=cfg.float32_gate_logits
  )
  disp = layers.moe_local_route_and_dispatch(
      normed,
      w.moe.router_gate,
      cfg,
      sublayer_idx=sublayer_idx,
      batch_offset=batch_offset,
      layer_idx=layer_idx,
      step=step,
  )
  unperm = layers.moe_expert_gmm_and_unsort(disp, w.moe.wi, w.moe.wo, cfg)
  return layers.moe_combine_and_add_base(unperm, base, cfg)


def run_superblock_unsplit_fallback(
    x: jax.Array,
    sb_params: params_lib.SuperblockParams,
    sb_absmax: Any,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    sb_idx: jax.Array | int = 0,
    step: jax.Array | int = 0,
    batch_offset: int = 0,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Pure-JAX `[GDN_0, GDN_1, GDN_2, FullAttn_3]` superblock without batchsplit."""
  base_layer_idx = sb_idx * 4
  x = jax.ad_checkpoint.checkpoint_name(x, "decoder_layer_input")
  x = _run_unsplit_sublayer_fallback(
      x,
      sb_params.gdn_0,
      sb_absmax.gdn_0 if hasattr(sb_absmax, "gdn_0") else sb_absmax[0],
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=0,
      layer_idx=base_layer_idx + 0,
      step=step,
      batch_offset=batch_offset,
      rope_cos_sin=None,
  )
  x = _run_unsplit_sublayer_fallback(
      x,
      sb_params.gdn_1,
      sb_absmax.gdn_1 if hasattr(sb_absmax, "gdn_1") else sb_absmax[1],
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=1,
      layer_idx=base_layer_idx + 1,
      step=step,
      batch_offset=batch_offset,
      rope_cos_sin=None,
  )
  x = _run_unsplit_sublayer_fallback(
      x,
      sb_params.gdn_2,
      sb_absmax.gdn_2 if hasattr(sb_absmax, "gdn_2") else sb_absmax[2],
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=2,
      layer_idx=base_layer_idx + 2,
      step=step,
      batch_offset=batch_offset,
      rope_cos_sin=None,
  )
  return _run_unsplit_sublayer_fallback(
      x,
      sb_params.attn_3,
      sb_absmax.attn_3 if hasattr(sb_absmax, "attn_3") else sb_absmax[3],
      positions,
      segment_ids,
      cfg,
      is_attn=True,
      sublayer_idx=3,
      layer_idx=base_layer_idx + 3,
      step=step,
      batch_offset=batch_offset,
      rope_cos_sin=rope_cos_sin,
  )


def local_forward_loss(
    params_local: params_lib.ModelParams,
    tokens_local: jax.Array,
    targets_local: jax.Array,
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: QwenConfig,
    *,
    step: jax.Array | int = 0,
) -> jax.Array:
  """Runs shard-local model forward pass (`unsplit` or `batchsplit`) and returns mean CE loss."""
  hidden = embed_tokens(tokens_local, params_local.token_embedder, cfg)
  hoisted_absmax = _compute_hoisted_superblock_absmax(params_local.blocks, cfg)
  sched_mod = _maybe_peer("schedule")
  remat_policy = build_remat_policy(cfg) if cfg.enable_remat else None
  sb_indices = jnp.arange(cfg.num_superblocks, dtype=jnp.int32)

  use_schedule_bs = (
      sched_mod is not None
      and cfg.schedule_mode == "batchsplit"
      and hidden.shape[0] >= 2
      and hasattr(sched_mod, "run_superblock_batchsplit")
  )
  use_schedule_unsplit = (
      sched_mod is not None
      and cfg.schedule_mode == "unsplit"
      and hasattr(sched_mod, "run_superblock_unsplit")
  )

  if use_schedule_bs:
    half_b = hidden.shape[0] // 2
    x_mb0, x_mb1 = hidden[:half_b], hidden[half_b:]
    pos_mb0, pos_mb1 = positions_local[:half_b], positions_local[half_b:]
    seg_mb0, seg_mb1 = segment_ids_local[:half_b], segment_ids_local[half_b:]
    rope_mb0 = None
    rope_mb1 = None
    if cfg.precompute_rope_tables:
      rope_mb0 = layers.compute_rope_cos_sin(
          pos_mb0, cfg.rotary_dim, cfg.rope_max_timescale
      )
      rope_mb1 = layers.compute_rope_cos_sin(
          pos_mb1, cfg.rotary_dim, cfg.rope_max_timescale
      )
    lineage_mod = (
        _maybe_peer("lineage_schedule")
        if cfg.use_lineage_pipelined_scan and cfg.num_superblocks >= 3
        else None
    )
    if lineage_mod is not None and hasattr(lineage_mod, "run_qwen_lineage_scan"):
      final_x_mb0, final_x_mb1 = lineage_mod.run_qwen_lineage_scan(
          x_mb0,
          x_mb1,
          params_local.blocks,
          hoisted_absmax,
          pos_mb0,
          pos_mb1,
          seg_mb0,
          seg_mb1,
          cfg,
          rope_mb0=rope_mb0,
          rope_mb1=rope_mb1,
          step=step,
      )
      hidden = jnp.concatenate([final_x_mb0, final_x_mb1], axis=0)
    else:
      t_ep_mb = x_mb1.shape[0] * x_mb1.shape[1] * cfg.ep_size
      use_3d_ep = (
          cfg.moe_use_tc_ragged
          and cfg.moe_3d_ep_tokens
          and (cfg.emb_dim % 128 == 0)
      )
      unperm_shape = (
          (t_ep_mb, cfg.emb_dim // 128, 128)
          if use_3d_ep
          else (t_ep_mb, cfg.emb_dim)
      )
      init_unperm = jnp.zeros(unperm_shape, dtype=x_mb1.dtype)
      if (
          cfg.enable_remat
          and cfg.remat_superblock_carry_only
          and hasattr(sched_mod, "run_superblock_batchsplit_scan")
      ):
        # Carry-only custom-VJP superblock scan: runs `full_prefetch_defer_rs`
        # strictly once in forward without saving `prefetched_w_dense_0` or
        # `unperm_mb1` in scan residuals.
        final_x_mb0, final_unperm_mb1, final_base_mb1 = (
            sched_mod.run_superblock_batchsplit_scan(
                x_mb0,
                init_unperm,
                x_mb1,
                params_local.blocks,
                hoisted_absmax,
                pos_mb0,
                pos_mb1,
                seg_mb0,
                seg_mb1,
                cfg,
                rope_mb0=rope_mb0,
                rope_mb1=rope_mb1,
                step=step,
            )
        )
      else:
        prefetched_0, next_gdn0s = sched_mod.prepare_loop_boundary_dense_pipeline(
            params_local.blocks, cfg
        )

        def _scan_body(carry, sb_tuple):
          sb_p, sb_a, next_gdn0, sb_i = sb_tuple
          return (
              sched_mod.run_superblock_batchsplit(
                  carry,
                  sb_p,
                  sb_a,
                  pos_mb0,
                  pos_mb1,
                  seg_mb0,
                  seg_mb1,
                  cfg,
                  sb_idx=sb_i,
                  step=step,
                  rope_mb0=rope_mb0,
                  rope_mb1=rope_mb1,
                  prev_rw3_local=None,
                  use_attn3_external_bwd_rw=False,
                  next_gdn0_params=next_gdn0,
              ),
              None,
          )

        if cfg.enable_remat:
          _scan_body = jax.checkpoint(
              _scan_body, policy=remat_policy, prevent_cse=False
          )

        init_unperm_bs = (
            sched_mod.maybe_wrap_init_unperm_partial_tokens(init_unperm, cfg)
            if hasattr(sched_mod, "maybe_wrap_init_unperm_partial_tokens")
            else init_unperm
        )
        init_carry = sched_mod.BatchSplitCarry(
            x_mb0=x_mb0,
            unperm_mb1=init_unperm_bs,
            base_mb1=x_mb1,
            bwd_rw3_prefetch=None,
            prefetched_w_dense_0=prefetched_0,
        )
        final_carry, _ = jax.lax.scan(
            _scan_body,
            init_carry,
            (params_local.blocks, hoisted_absmax, next_gdn0s, sb_indices),
        )
        final_x_mb0 = final_carry.x_mb0
        final_unperm_mb1 = final_carry.unperm_mb1
        final_base_mb1 = final_carry.base_mb1
      x_mb1_final = layers.moe_combine_and_add_base(
          final_unperm_mb1, final_base_mb1, cfg
      )
      hidden = jnp.concatenate([final_x_mb0, x_mb1_final], axis=0)
  elif cfg.schedule_mode == "batchsplit" and hidden.shape[0] >= 2:
    # Pure-JAX standalone batchsplit fallback without external schedule module.
    rope_all = None
    if cfg.precompute_rope_tables:
      rope_all = layers.compute_rope_cos_sin(
          positions_local, cfg.rotary_dim, cfg.rope_max_timescale
      )
    half_b = hidden.shape[0] // 2
    rope_mb0 = (
        (rope_all[0][:half_b], rope_all[1][:half_b])
        if rope_all is not None
        else None
    )
    rope_mb1 = (
        (rope_all[0][half_b:], rope_all[1][half_b:])
        if rope_all is not None
        else None
    )

    def _sb_step(h_pair, sb_tuple):
      h0, h1 = h_pair
      sb_p, sb_a, sb_i = sb_tuple
      h0_out = run_superblock_unsplit_fallback(
          h0,
          sb_p,
          sb_a,
          positions_local[:half_b],
          segment_ids_local[:half_b],
          cfg,
          sb_idx=sb_i,
          step=step,
          batch_offset=0,
          rope_cos_sin=rope_mb0,
      )
      h1_out = run_superblock_unsplit_fallback(
          h1,
          sb_p,
          sb_a,
          positions_local[half_b:],
          segment_ids_local[half_b:],
          cfg,
          sb_idx=sb_i,
          step=step,
          batch_offset=half_b,
          rope_cos_sin=rope_mb1,
      )
      return (h0_out, h1_out), None

    if cfg.enable_remat:
      _sb_step = jax.checkpoint(_sb_step, policy=remat_policy)
    (h0_f, h1_f), _ = jax.lax.scan(
        _sb_step,
        (hidden[:half_b], hidden[half_b:]),
        (params_local.blocks, hoisted_absmax, sb_indices),
    )
    hidden = jnp.concatenate([h0_f, h1_f], axis=0)
  else:
    rope_cos_sin = None
    if cfg.precompute_rope_tables:
      rope_cos_sin = layers.compute_rope_cos_sin(
          positions_local, cfg.rotary_dim, cfg.rope_max_timescale
      )

    def _sb_step(h, sb_tuple):
      sb_p, sb_a, sb_i = sb_tuple
      if use_schedule_unsplit:
        out = sched_mod.run_superblock_unsplit(
            h,
            sb_p,
            sb_a,
            positions_local,
            segment_ids_local,
            cfg,
            sb_idx=sb_i,
            step=step,
            rope_cos_sin=rope_cos_sin,
        )
      else:
        out = run_superblock_unsplit_fallback(
            h,
            sb_p,
            sb_a,
            positions_local,
            segment_ids_local,
            cfg,
            sb_idx=sb_i,
            step=step,
            rope_cos_sin=rope_cos_sin,
        )
      return out, None

    if cfg.enable_remat:
      _sb_step = jax.checkpoint(_sb_step, policy=remat_policy)
    hidden, _ = jax.lax.scan(
        _sb_step, hidden, (params_local.blocks, hoisted_absmax, sb_indices)
    )

  return chunked_lm_head_loss(
      hidden,
      params_local.decoder_norm,
      params_local.lm_head,
      targets_local,
      segment_ids_local,
      cfg,
  )


# ---------------------------------------------------------------------------
# Pure-Optax Optimizer (AdamW w/ Optional FP32 Master Weights & Stateless SGD)
# ---------------------------------------------------------------------------


def _build_weight_decay_mask(params: params_lib.ModelParams):
  def _is_decayed(path, leaf):
    del leaf
    p_str = jax.tree_util.keystr(path)
    if any(
        tok in p_str
        for tok in ("norm", "conv1d", "A_log", "dt_bias", "token_embedder")
    ):
      return False
    return True

  return jax.tree_util.tree_map_with_path(_is_decayed, params)


def create_optimizer(cfg: QwenConfig) -> optax.GradientTransformation:
  """Creates pure Optax AdamW (`mu_dtype`, optional `grad_clip_norm`) or stateless SGD."""
  if cfg.optimizer == "sgd":
    if cfg.grad_clip_norm > 0.0:
      return optax.chain(
          optax.clip_by_global_norm(cfg.grad_clip_norm),
          optax.sgd(learning_rate=cfg.learning_rate),
      )
    return optax.sgd(learning_rate=cfg.learning_rate)

  mu_dt = jnp.bfloat16 if cfg.mu_dtype == "bfloat16" else jnp.float32
  adamw_tx = optax.adamw(
      learning_rate=cfg.learning_rate,
      b1=cfg.adam_b1,
      b2=cfg.adam_b2,
      eps=cfg.adam_eps,
      eps_root=cfg.adam_eps_root,
      mu_dtype=mu_dt,
      weight_decay=cfg.adam_weight_decay,
      mask=_build_weight_decay_mask,
  )
  if cfg.grad_clip_norm > 0.0:
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), adamw_tx)
  return adamw_tx


def _map_opt_shardings(
    raw_shapes: Any,
    abstract_params: params_lib.ModelParams,
    mesh: Mesh,
) -> Any:
  """Maps Optax state shapes to matching parameter `NamedSharding`s."""
  scalar_sharding = NamedSharding(mesh, P())
  key_to_sharding = {
      jax.tree_util.keystr(p_path): p_leaf.sharding
      for p_path, p_leaf in jax.tree.leaves_with_path(abstract_params)
  }
  param_leaves = jax.tree.leaves_with_path(abstract_params)

  def _assign(path, leaf):
    if getattr(leaf, "ndim", 0) == 0:
      return jax.ShapeDtypeStruct(
          leaf.shape, leaf.dtype, sharding=scalar_sharding
      )
    p_str = jax.tree_util.keystr(path)
    for suf, sh in key_to_sharding.items():
      if p_str.endswith(suf):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sh)
    for _, p_leaf in param_leaves:
      if leaf.shape == p_leaf.shape:
        return jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=p_leaf.sharding
        )
    return jax.ShapeDtypeStruct(
        leaf.shape, leaf.dtype, sharding=scalar_sharding
    )

  return jax.tree.map_with_path(_assign, raw_shapes)


def get_abstract_opt_state(
    cfg: QwenConfig,
    mesh: Mesh,
    abstract_params: params_lib.ModelParams,
) -> Any:
  """Evaluates abstract optimizer state with exact sharded `NamedSharding`s."""
  tx = create_optimizer(cfg)
  if cfg.master_weights_fp32:
    master_abs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(
            x.shape, jnp.float32, sharding=x.sharding
        ),
        abstract_params,
    )
    inner_raw = jax.eval_shape(tx.init, master_abs)
    inner_abs = _map_opt_shardings(inner_raw, master_abs, mesh)
    return OptimizerState(
        step=jax.ShapeDtypeStruct(
            (), jnp.int32, sharding=NamedSharding(mesh, P())
        ),
        master_params=master_abs,
        inner_opt_state=inner_abs,
    )

  raw_shapes = jax.eval_shape(tx.init, abstract_params)
  return _map_opt_shardings(raw_shapes, abstract_params, mesh)


def init_opt_state(
    cfg: QwenConfig,
    mesh: Mesh,
    params: params_lib.ModelParams,
) -> Any:
  """Initializes sharded optimizer state directly on device (multi-host safe)."""
  tx = create_optimizer(cfg)
  param_pspecs = params_lib.get_param_pspecs(cfg)
  param_shardings = jax.tree.map(
      lambda ps: NamedSharding(mesh, ps), param_pspecs
  )
  abstract_params = params_lib.get_abstract_params(
      cfg, mesh, param_dtype=params.token_embedder.dtype
  )
  abstract_opt_state = get_abstract_opt_state(cfg, mesh, abstract_params)
  opt_shardings = jax.tree.map(lambda x: x.sharding, abstract_opt_state)

  if cfg.master_weights_fp32:

    def _init_master(p):
      master = jax.tree.map(lambda arr: arr.astype(jnp.float32), p)
      return OptimizerState(
          step=jnp.asarray(0, dtype=jnp.int32),
          master_params=master,
          inner_opt_state=tx.init(master),
      )

    return jax.jit(
        _init_master,
        in_shardings=(param_shardings,),
        out_shardings=opt_shardings,
    )(params)

  return jax.jit(
      tx.init,
      in_shardings=(param_shardings,),
      out_shardings=opt_shardings,
  )(params)


def init_synthetic_batch(
    cfg: QwenConfig,
    mesh: Mesh,
    *,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Creates sharded synthetic `(tokens, targets, positions, segment_ids)` directly on device."""
  global_b = cfg.num_devices * cfg.per_device_batch_size
  seq_len = cfg.seq_len
  data_sharding = NamedSharding(mesh, P(cfg.all_sharded_axes, None))

  @jax.jit
  def _build():
    key = jax.random.PRNGKey(seed)
    k_tok, k_tgt = jax.random.split(key)
    tokens = jax.random.randint(
        k_tok, (global_b, seq_len), 0, cfg.vocab_size, dtype=jnp.int32
    )
    targets = jax.random.randint(
        k_tgt, (global_b, seq_len), 0, cfg.vocab_size, dtype=jnp.int32
    )
    positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :], (global_b, seq_len)
    )
    segment_ids = jnp.ones((global_b, seq_len), dtype=jnp.int32)
    return tokens, targets, positions, segment_ids

  tokens, targets, positions, segment_ids = _build()
  return (
      jax.device_put(tokens, data_sharding),
      jax.device_put(targets, data_sharding),
      jax.device_put(positions, data_sharding),
      jax.device_put(segment_ids, data_sharding),
  )


def _extract_step_from_opt_state(opt_state: Any) -> jax.Array:
  """Extracts replicated `int32` step counter from `OptimizerState` or Optax state."""
  if isinstance(opt_state, OptimizerState):
    return opt_state.step.astype(jnp.int32)
  for path, leaf in jax.tree_util.tree_leaves_with_path(opt_state):
    if (
        path
        and getattr(path[-1], "name", None) in ("step", "count")
        and getattr(leaf, "ndim", -1) == 0
    ):
      return leaf.astype(jnp.int32)
  return jnp.asarray(0, dtype=jnp.int32)


def make_value_and_grad_fn(cfg: QwenConfig, mesh: Mesh):
  """Returns `(params, tokens, targets, positions, segment_ids, [step]) -> (loss, grads)`."""
  param_pspecs = params_lib.get_param_pspecs(cfg)
  data_pspec = P(cfg.all_sharded_axes, None)

  def _local_vag(
      params_local: params_lib.ModelParams,
      tokens_local: jax.Array,
      targets_local: jax.Array,
      positions_local: jax.Array,
      segment_ids_local: jax.Array,
      step_local: jax.Array,
  ) -> tuple[jax.Array, params_lib.ModelParams]:
    def _loss(p):
      return local_forward_loss(
          p,
          tokens_local,
          targets_local,
          positions_local,
          segment_ids_local,
          cfg,
          step=step_local,
      )

    return jax.value_and_grad(_loss)(params_local)

  sm_vag = _shard_map(
      _local_vag,
      mesh=mesh,
      in_specs=(
          param_pspecs,
          data_pspec,
          data_pspec,
          data_pspec,
          data_pspec,
          P(),
      ),
      out_specs=(P(), param_pspecs),
  )

  def _vag(
      params: params_lib.ModelParams,
      tokens: jax.Array,
      targets: jax.Array,
      positions: jax.Array,
      segment_ids: jax.Array,
      step: jax.Array | int | None = None,
  ) -> tuple[jax.Array, params_lib.ModelParams]:
    step_arr = (
        jnp.asarray(0, dtype=jnp.int32)
        if step is None
        else jnp.asarray(step, dtype=jnp.int32)
    )
    return sm_vag(params, tokens, targets, positions, segment_ids, step_arr)

  return _vag


def make_train_step(
    cfg: QwenConfig,
    mesh: Mesh,
    *,
    donate_state: bool = True,
):
  """Builds pure-JAX `jit_train_step(params, opt_state, tokens, targets, positions, segment_ids)`."""
  tx = create_optimizer(cfg)
  vag_fn = make_value_and_grad_fn(cfg, mesh)
  param_pspecs = params_lib.get_param_pspecs(cfg)
  param_shardings = jax.tree.map(
      lambda ps: NamedSharding(mesh, ps), param_pspecs
  )
  abstract_params = params_lib.get_abstract_params(cfg, mesh)
  opt_shardings = jax.tree.map(
      lambda x: x.sharding, get_abstract_opt_state(cfg, mesh, abstract_params)
  )
  data_sharding = NamedSharding(mesh, P(cfg.all_sharded_axes, None))
  scalar_sharding = NamedSharding(mesh, P())

  def train_step(
      params: params_lib.ModelParams,
      opt_state: Any,
      tokens: jax.Array,
      targets: jax.Array,
      positions: jax.Array,
      segment_ids: jax.Array,
  ) -> tuple[params_lib.ModelParams, Any, jax.Array]:
    step = _extract_step_from_opt_state(opt_state)
    loss, grads = vag_fn(
        params, tokens, targets, positions, segment_ids, step=step
    )
    if cfg.master_weights_fp32:
      grads_f32 = jax.tree.map(lambda g: g.astype(jnp.float32), grads)
      updates, new_inner = tx.update(
          grads_f32, opt_state.inner_opt_state, opt_state.master_params
      )
      new_master = optax.apply_updates(opt_state.master_params, updates)
      new_params = jax.tree.map(
          lambda m, p: m.astype(p.dtype), new_master, params
      )
      new_opt_state = OptimizerState(
          step=opt_state.step + jnp.asarray(1, dtype=jnp.int32),
          master_params=new_master,
          inner_opt_state=new_inner,
      )
    else:
      updates, new_opt_state = tx.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

  donate = (0, 1) if donate_state else ()
  compiler_opts: dict[str, Any] = {}
  if cfg.use_max_async_overlap_transform:
    overlap_mod = _maybe_peer("max_async_overlap_transform")
    if overlap_mod is not None and hasattr(
        overlap_mod, "register_max_async_overlap_transform"
    ):
      overlap_mod.register_max_async_overlap_transform(cfg)

  jit_fn = jax.jit(
      train_step,
      in_shardings=(
          param_shardings,
          opt_shardings,
          data_sharding,
          data_sharding,
          data_sharding,
          data_sharding,
      ),
      out_shardings=(param_shardings, opt_shardings, scalar_sharding),
      donate_argnums=donate,
      compiler_options=compiler_opts or None,
  )
  return jit_fn


# ---------------------------------------------------------------------------
# MFU & Superblock Stride Helpers
# ---------------------------------------------------------------------------


def estimate_active_params(
    cfg: QwenConfig, *, num_superblocks: int | None = None
) -> int:
  """Counts active non-embedding matmul parameters per token for Qwen 3.5 / Qwen MAX."""
  sb = cfg.num_superblocks if num_superblocks is None else num_superblocks
  d = cfg.emb_dim
  gdn_per_layer = (
      d * cfg.gdn_qkvz_dim
      + d * (2 * cfg.gdn_num_v_heads)
      + cfg.gdn_value_dim * d
  )
  attn_per_layer = (
      d * (cfg.num_query_heads * 2 * cfg.head_dim)
      + 2 * d * (cfg.num_kv_heads * cfg.head_dim)
      + cfg.attn_q_out_dim * d
  )
  moe_active_per_layer = (
      d * cfg.num_experts
      + cfg.num_experts_per_tok * (3 * d * cfg.moe_mlp_dim)
      + 3 * d * cfg.shared_mlp_dim
      + d
  )
  per_sb = 3 * (gdn_per_layer + moe_active_per_layer) + (
      attn_per_layer + moe_active_per_layer
  )
  return sb * per_sb + d * cfg.vocab_size


def estimate_step_flops(
    cfg: QwenConfig, *, num_superblocks: int | None = None
) -> float:
  """Estimates training FLOPs/step (`6 * N_active * tokens + attn + gdn`) for Ironwood MFU."""
  sb = cfg.num_superblocks if num_superblocks is None else num_superblocks
  tokens = float(cfg.num_devices * cfg.per_device_batch_size * cfg.seq_len)
  active = float(estimate_active_params(cfg, num_superblocks=sb))
  attn_flops_per_tok = float(
      sb * 12 * cfg.num_query_heads * cfg.head_dim * cfg.seq_len // 2
  )
  return (6.0 * active + attn_flops_per_tok) * tokens


def compute_mfu(
    cfg: QwenConfig,
    step_time_s: float,
    *,
    num_superblocks: int | None = None,
) -> float:
  """Computes TPU7x Ironwood BF16 MFU against `PEAK_TFLOPS_PER_CHIP_BF16_V7 = 2307.0`."""
  flops = estimate_step_flops(cfg, num_superblocks=num_superblocks)
  peak = (
      float(cfg.num_chips) * PEAK_TFLOPS_PER_CHIP_BF16_V7 * 1e12 * step_time_s
  )
  return flops / peak if peak > 0.0 else 0.0


def estimate_scan_stride_s(
    step_time_s: float,
    fixed_overhead_s: float,
    cfg: QwenConfig,
) -> float:
  """Estimates per-superblock scan stride in seconds from total step time and fixed overhead."""
  return max(0.0, step_time_s - fixed_overhead_s) / float(
      max(1, cfg.num_superblocks)
  )
