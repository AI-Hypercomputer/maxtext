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

"""True Lineage 8-phase (`program_order`) + `jax.lax.scan` FSDP/EP schedule for Qwen.

Performance Invariant (do not reintroduce — regressed +80-100 ms in v17 & v19a):
  - Never route per-slot MoE weights through `.at[rows].get()` / gather or
    replace the `v18` routing path with a saved `MoeCompactRouting` / padded
    `ragged_gather_reduce_tc` in backward. Both trigger a ~+77 ms `jvp()`
    fusion regression (`linear_combine_fwd` / gather) and double
    `transpose(jvp(ragged_gather_reduce_tc))` calls (+25 ms) in `phase4`/`phase5`.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

try:
  from jax.experimental.overlap import program_order
except ImportError:
  try:
    from jax._src.pjit import program_order
  except ImportError:

    def program_order(*unused_args, **unused_kwargs):
      return lambda fn: fn


from maxtext.experimental.lineage.qwen import config as config_lib
from maxtext.experimental.lineage.qwen import layers
from maxtext.experimental.lineage.qwen import ops
from maxtext.experimental.lineage.qwen import params as params_lib
from maxtext.experimental.lineage.qwen import pipelined_scan

QwenConfig = config_lib.QwenConfig


def _ubatch_wait(x: Any) -> Any:
  return x


def _ubatch_mixer(
    x: jax.Array,
    w_sub: Any,
    pos: jax.Array,
    seg: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    gdn_meta: Any = None,
) -> jax.Array:
  if is_attn:
    return layers.full_attn_mixer(
        x, w_sub, pos, seg, cfg, rope_cos_sin=rope_cos_sin
    )
  return layers.gdn_mixer(x, w_sub, seg, cfg, precomputed_meta=gdn_meta)


def _extract_gdn_mixer_grad(
    g_w: params_lib.GdnSublayerParams,
) -> params_lib.GdnSublayerParams:
  return params_lib.GdnSublayerParams(
      pre_attn_norm=g_w.pre_attn_norm,
      post_attn_norm=None,
      in_proj_qkvz=g_w.in_proj_qkvz,
      in_proj_ba=g_w.in_proj_ba,
      conv1d=g_w.conv1d,
      A_log=g_w.A_log,
      dt_bias=g_w.dt_bias,
      gdn_norm=g_w.gdn_norm,
      out_proj=g_w.out_proj,
      moe=None,
  )


def _extract_attn_mixer_grad(
    g_w: params_lib.AttnSublayerParams,
) -> params_lib.AttnSublayerParams:
  return params_lib.AttnSublayerParams(
      pre_attn_norm=g_w.pre_attn_norm,
      post_attn_norm=None,
      q_proj=g_w.q_proj,
      k_proj=g_w.k_proj,
      v_proj=g_w.v_proj,
      q_norm=g_w.q_norm,
      k_norm=g_w.k_norm,
      out_proj=g_w.out_proj,
      moe=None,
  )


def _zero_gdn_mixer_grad_like(
    w_sub: params_lib.GdnSublayerParams,
) -> params_lib.GdnSublayerParams:
  return params_lib.GdnSublayerParams(
      pre_attn_norm=jnp.zeros_like(w_sub.pre_attn_norm),
      post_attn_norm=None,
      in_proj_qkvz=jnp.zeros_like(w_sub.in_proj_qkvz),
      in_proj_ba=jnp.zeros_like(w_sub.in_proj_ba),
      conv1d=jnp.zeros_like(w_sub.conv1d),
      A_log=jnp.zeros_like(w_sub.A_log),
      dt_bias=jnp.zeros_like(w_sub.dt_bias),
      gdn_norm=jnp.zeros_like(w_sub.gdn_norm),
      out_proj=jnp.zeros_like(w_sub.out_proj),
      moe=None,
  )


def _zero_attn_mixer_grad_like(
    w_sub: params_lib.AttnSublayerParams,
) -> params_lib.AttnSublayerParams:
  return params_lib.AttnSublayerParams(
      pre_attn_norm=jnp.zeros_like(w_sub.pre_attn_norm),
      post_attn_norm=None,
      q_proj=jnp.zeros_like(w_sub.q_proj),
      k_proj=jnp.zeros_like(w_sub.k_proj),
      v_proj=jnp.zeros_like(w_sub.v_proj),
      q_norm=jnp.zeros_like(w_sub.q_norm),
      k_norm=jnp.zeros_like(w_sub.k_norm),
      out_proj=jnp.zeros_like(w_sub.out_proj),
      moe=None,
  )


def _ubatch_norm_and_route(
    h: jax.Array,
    post_attn_norm: jax.Array,
    router_gate: jax.Array,
    cfg: QwenConfig,
    *,
    sublayer_idx: int,
    batch_offset: int,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  normed = ops.rms_norm_zero_centered(h, post_attn_norm, cfg.norm_eps)
  topk_w, topk_ids = layers.moe_local_route_only(
      normed,
      router_gate,
      cfg,
      sublayer_idx=sublayer_idx,
      batch_offset=batch_offset,
      layer_idx=layer_idx,
      step=step,
  )
  return normed, topk_w, topk_ids


def _ubatch_norm_and_metadata(
    h: jax.Array,
    post_attn_norm: jax.Array,
    router_gate: jax.Array,
    cfg: QwenConfig,
    *,
    sublayer_idx: int,
    batch_offset: int,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]:
  """Phase 1/3: post-attn norm, router, SC0 metadata AG, and FP8 token quantize."""
  normed, topk_w, topk_ids = _ubatch_norm_and_route(
      h,
      post_attn_norm,
      router_gate,
      cfg,
      sublayer_idx=sublayer_idx,
      batch_offset=batch_offset,
      layer_idx=layer_idx,
      step=step,
  )
  use_3d = (
      cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
  )
  tok_in = ops.order_after(normed, topk_w)
  if use_3d:
    tok_flat = tok_in.reshape(-1, cfg.emb_dim // 128, 128)
  else:
    tok_flat = tok_in.reshape(-1, cfg.emb_dim)

  k_top = cfg.num_experts_per_tok
  w_flat = topk_w.reshape(-1, k_top)
  ids_flat = jax.lax.stop_gradient(topk_ids.reshape(-1, k_top).astype(jnp.int32))

  if cfg.ep_size > 1:
    if cfg.defer_meta_ag_to_dispatch:
      # Defer `(w_flat, ids_flat)` AllGather into phase2/phase4 bundled with
      # `codes_l` on SC0 so phase1/phase3 have zero exposed SC0 `call-done`.
      w_ep = w_flat
      ids_ep = ids_flat
    else:
      ep_ax = cfg.ep_axis_names
      core_id = 0 if cfg.ep_sc_pinning == "sc0" else None
      min_b = cfg.sc_min_collective_bytes
      use_sc0 = (
          core_id is not None
          and w_flat.size * w_flat.dtype.itemsize * cfg.ep_size >= min_b
          and ids_flat.size * ids_flat.dtype.itemsize * cfg.ep_size >= min_b
      )

      def _ag_meta(w_in, ids_in):
        return (
            jax.lax.all_gather(w_in, axis_name=ep_ax, axis=0, tiled=True),
            jax.lax.stop_gradient(
                jax.lax.all_gather(ids_in, axis_name=ep_ax, axis=0, tiled=True)
            ),
        )

      if use_sc0:
        w_ep, ids_ep = pipelined_scan.sc_compute_on(_ag_meta, core_id=core_id)(
            w_flat, ids_flat
        )
      else:
        w_ep, ids_ep = _ag_meta(w_flat, ids_flat)

    if use_3d and cfg.moe_fp8_token_dispatch:
      qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
      amax_local = jax.lax.stop_gradient(
          jnp.max(jnp.abs(tok_flat.astype(jnp.float32)))
      )
      amax_ep = ops._ep_scalar_pmax(amax_local, cfg).reshape(1, 1)
      scale = jax.lax.stop_gradient(
          jnp.where(amax_ep > 0.0, amax_ep / qmax, jnp.ones_like(amax_ep))
      )
      inv_scale = (1.0 / scale).reshape(1, *([1] * (tok_flat.ndim - 1)))
      codes_l = jax.lax.stop_gradient(
          jnp.clip(
              tok_flat.astype(jnp.float32) * inv_scale, -qmax, qmax
          ).astype(jnp.float8_e4m3fn)
      )
    else:
      codes_l = tok_flat
      scale = None
  else:
    w_ep = w_flat
    ids_ep = ids_flat
    codes_l = tok_flat
    scale = None

  return normed, codes_l, scale, w_ep, ids_ep


def _ubatch_dispatch_tokens(
    codes_l: jax.Array,
    scale: jax.Array | None,
    w_ep: jax.Array,
    ids_ep: jax.Array,
    normed_dtype: Any,
    cfg: QwenConfig,
    *,
    gather_meta: bool = False,
) -> layers.DispatchedMoeTokens:
  """Phase 2/4: SC0 token AllGather (optionally bundled with deferred meta AG)."""
  if cfg.ep_size > 1:
    core_id = 0 if cfg.ep_sc_pinning == "sc0" else None
    ep_ax = cfg.ep_axis_names
    min_b = cfg.sc_min_collective_bytes
    if gather_meta:
      w_in = w_ep
      ids_in = ids_ep
      use_sc0_all = (
          core_id is not None
          and codes_l.size * codes_l.dtype.itemsize * cfg.ep_size >= min_b
          and w_in.size * w_in.dtype.itemsize * cfg.ep_size >= min_b
          and ids_in.size * ids_in.dtype.itemsize * cfg.ep_size >= min_b
      )

      def _ag_all(c_in, w_arg, ids_arg):
        return (
            jax.lax.all_gather(c_in, axis_name=ep_ax, axis=0, tiled=True),
            jax.lax.all_gather(w_arg, axis_name=ep_ax, axis=0, tiled=True),
            jax.lax.stop_gradient(
                jax.lax.all_gather(
                    ids_arg, axis_name=ep_ax, axis=0, tiled=True
                )
            ),
        )

      if use_sc0_all:
        gathered_tok, w_ep, ids_ep = pipelined_scan.sc_compute_on(
            _ag_all, core_id=core_id
        )(codes_l, w_in, ids_in)
      else:
        gathered_tok = ops.sc_all_gather(
            codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
        )
        w_ep = ops.sc_all_gather(
            w_in, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
        )
        ids_ep = jax.lax.stop_gradient(
            ops.sc_all_gather(
                ids_in,
                ep_ax,
                core_id=core_id,
                axis=0,
                min_collective_bytes=min_b,
            )
        )
      if scale is not None:
        ep_shape = (codes_l.shape[0] * cfg.ep_size, *codes_l.shape[1:])
        ste = jax.lax.broadcast(jnp.zeros((), dtype=normed_dtype), ep_shape)
        tok_ep: Any = ops.Fp8DispatchedTokens(
            ste_carrier=ste, codes_ep=gathered_tok, scale=scale
        )
      else:
        tok_ep = gathered_tok
    elif scale is not None:
      ep_shape = (codes_l.shape[0] * cfg.ep_size, *codes_l.shape[1:])
      codes_ep = ops.sc_all_gather(
          codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
      )
      ste = jax.lax.broadcast(jnp.zeros((), dtype=normed_dtype), ep_shape)
      tok_ep = ops.Fp8DispatchedTokens(
          ste_carrier=ste, codes_ep=codes_ep, scale=scale
      )
    else:
      tok_ep = ops.sc_all_gather(
          codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
      )
  else:
    tok_ep = codes_l
  return layers.DispatchedMoeTokens(
      tokens_ep=tok_ep, topk_weights_ep=w_ep, topk_ids_ep=ids_ep
  )


def _ubatch_remat_dispatch(
    normed: jax.Array,
    topk_w: jax.Array,
    topk_ids: jax.Array,
    saved_scale: jax.Array | None,
    cfg: QwenConfig,
) -> layers.DispatchedMoeTokens:
  """BWD remat dispatch reusing FWD FP8 `saved_scale` (zero remat `pmax`)."""
  if saved_scale is None or cfg.ep_size <= 1:
    return layers.moe_dispatch_from_routed(
        normed, topk_w, topk_ids, cfg, order_tokens_on_router=False
    )
  use_3d = (
      cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
  )
  tok_flat = (
      normed.reshape(-1, cfg.emb_dim // 128, 128)
      if use_3d
      else normed.reshape(-1, cfg.emb_dim)
  )
  k_top = cfg.num_experts_per_tok
  w_flat = topk_w.reshape(-1, k_top)
  ids_flat = jax.lax.stop_gradient(topk_ids.reshape(-1, k_top).astype(jnp.int32))
  ep_ax = cfg.ep_axis_names
  core_id = 0 if cfg.ep_sc_pinning == "sc0" else None
  min_b = cfg.sc_min_collective_bytes
  use_sc0 = (
      core_id is not None
      and w_flat.size * w_flat.dtype.itemsize * cfg.ep_size >= min_b
      and ids_flat.size * ids_flat.dtype.itemsize * cfg.ep_size >= min_b
  )

  def _ag_meta(w_in, ids_in):
    return (
        jax.lax.all_gather(w_in, axis_name=ep_ax, axis=0, tiled=True),
        jax.lax.stop_gradient(
            jax.lax.all_gather(ids_in, axis_name=ep_ax, axis=0, tiled=True)
        ),
    )

  if use_sc0:
    w_ep, ids_ep = pipelined_scan.sc_compute_on(_ag_meta, core_id=core_id)(
        w_flat, ids_flat
    )
  else:
    w_ep, ids_ep = _ag_meta(w_flat, ids_flat)

  qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
  scale = jax.lax.stop_gradient(saved_scale)
  inv_scale = (1.0 / scale).reshape(1, *([1] * (tok_flat.ndim - 1)))
  codes_l = jax.lax.stop_gradient(
      jnp.clip(tok_flat.astype(jnp.float32) * inv_scale, -qmax, qmax).astype(
          jnp.float8_e4m3fn
      )
  )
  return _ubatch_dispatch_tokens(
      codes_l, scale, w_ep, ids_ep, normed.dtype, cfg
  )


def _ubatch_remat_dispatch_tokens_only(
    normed: jax.Array,
    saved_scale: jax.Array | None,
    cfg: QwenConfig,
) -> jax.Array | ops.Fp8DispatchedTokens:
  """SC0-pinned remat token-only AllGather reusing FWD `saved_scale` (zero metadata AG)."""
  use_3d = (
      cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
  )
  tok_flat = (
      normed.reshape(-1, cfg.emb_dim // 128, 128)
      if use_3d
      else normed.reshape(-1, cfg.emb_dim)
  )
  if cfg.ep_size <= 1:
    return tok_flat
  core_id = 0 if cfg.ep_sc_pinning == "sc0" else None
  ep_ax = cfg.ep_axis_names
  min_b = cfg.sc_min_collective_bytes
  if saved_scale is not None and use_3d and cfg.moe_fp8_token_dispatch:
    qmax = float(jnp.finfo(jnp.float8_e4m3fn).max)
    scale = jax.lax.stop_gradient(saved_scale)
    inv_scale = (1.0 / scale).reshape(1, *([1] * (tok_flat.ndim - 1)))
    codes_l = jax.lax.stop_gradient(
        jnp.clip(tok_flat.astype(jnp.float32) * inv_scale, -qmax, qmax).astype(
            jnp.float8_e4m3fn
        )
    )
    ep_shape = (codes_l.shape[0] * cfg.ep_size, *codes_l.shape[1:])
    codes_ep = ops.sc_all_gather(
        codes_l, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
    )
    ste = jax.lax.broadcast(jnp.zeros((), dtype=normed.dtype), ep_shape)
    return ops.Fp8DispatchedTokens(
        ste_carrier=ste, codes_ep=codes_ep, scale=scale
    )
  return ops.sc_all_gather(
      tok_flat, ep_ax, core_id=core_id, axis=0, min_collective_bytes=min_b
  )


def _ubatch_dispatch_bwd_and_reduce(
    g_disp: layers.DispatchedMoeTokens,
    deferred_reduce: pipelined_scan.DeferredReduce | None,
    cfg: QwenConfig,
    *,
    use_fp8_disp: bool,
    tok_local_shape: tuple[int, ...],
    finish_deferred: bool = True,
) -> tuple[jax.Array, jax.Array, Any]:
  """SC0 `dispatch` transpose (`psum_scatter`) + decoupled deferred EP-axis `AllReduce`."""
  core_id = 0 if cfg.ep_sc_pinning == "sc0" else None
  ep_ax = cfg.ep_axis_names
  min_b = cfg.sc_min_collective_bytes

  g_tok_ep = (
      g_disp.tokens_ep.ste_carrier
      if isinstance(g_disp.tokens_ep, ops.Fp8DispatchedTokens)
      else g_disp.tokens_ep
  )
  ops.assert_bf16_or_fp32_reduction(g_tok_ep, "dispatch_bwd_tokens")
  ops.assert_bf16_or_fp32_reduction(
      g_disp.topk_weights_ep, "dispatch_bwd_weights"
  )

  def _sc0_dispatch_bwd(tok_ep, w_ep):
    if cfg.ep_size > 1:
      g_tok = jax.lax.psum_scatter(
          tok_ep, axis_name=ep_ax, scatter_dimension=0, tiled=True
      )
      g_w = jax.lax.psum_scatter(
          w_ep, axis_name=ep_ax, scatter_dimension=0, tiled=True
      )
      return g_tok, g_w
    return tok_ep, w_ep

  if core_id is not None and cfg.ep_size > 1:
    g_tok_local, g_w_local = pipelined_scan.sc_compute_on(
        _sc0_dispatch_bwd, core_id=core_id
    )(g_tok_ep, g_disp.topk_weights_ep)
  else:
    g_tok_local, g_w_local = _sc0_dispatch_bwd(
        g_tok_ep, g_disp.topk_weights_ep
    )

  # Issue `deferred_reduce` in a separate SC0 `compute_on` block so `g_tok_local`
  # and `g_w_local` unblock phase6/phase8 `vjp_nr` immediately without waiting
  # for EP-axis weight-gradient AllReduces to complete. When `finish_deferred`
  # is False, return only `dr_out` so non-AR SC1 RS leaves (`wi`/`wo`, dense
  # FSDP+EP projections) stay completely outside `_layer_bwd` program-order
  # barriers and overlap all 8 phases.
  if deferred_reduce is not None:
    if core_id is not None:
      dr_out = pipelined_scan.sc_compute_on(
          deferred_reduce.local_fn, core_id=core_id
      )(deferred_reduce.operand)
    else:
      dr_out = deferred_reduce.local_fn(deferred_reduce.operand)
    reduced_w = deferred_reduce.finish(dr_out) if finish_deferred else dr_out
  else:
    reduced_w = None

  topk_w_shape = (*tok_local_shape[:-1], g_w_local.shape[-1])
  del use_fp8_disp, min_b
  return (
      g_tok_local.reshape(tok_local_shape),
      g_w_local.reshape(topk_w_shape),
      reduced_w,
  )


def _ubatch_combine_fwd(
    unperm_ep: Any,
    cfg: QwenConfig,
) -> jax.Array:
  if cfg.ep_size <= 1:
    return unperm_ep.y_ep if isinstance(unperm_ep, ops.MoePartialTokens) else unperm_ep
  return ops.ep_token_combine(unperm_ep, cfg)


def _ubatch_combine_bwd(
    g_comb: jax.Array,
    cfg: QwenConfig,
    *,
    as_partial_tokens: bool,
    unperm_shape: tuple[int, ...],
    precomputed_amax_ep: jax.Array | None = None,
) -> Any:
  g_comb_flat = g_comb.reshape((-1, *unperm_shape[1:]))
  if cfg.ep_size <= 1:
    if as_partial_tokens:
      raise ValueError("as_partial_tokens=True requires ep_size > 1")
    return g_comb_flat
  return ops.ep_token_combine_bwd_only(
      g_comb_flat,
      cfg,
      as_partial_tokens=as_partial_tokens,
      precomputed_amax_ep=precomputed_amax_ep,
  )


def _sublayer_fwd_8phase(
    h_mb0: jax.Array,
    x_mb1: jax.Array,
    w_curr_sub: Any,
    w_next_sub: Any | None,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn_curr: bool,
    is_attn_next: bool,
    sublayer_idx: int,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    gdn_meta_mb0: Any = None,
    gdn_meta_mb1: Any = None,
    save_h_mb0: bool = True,
    w_next_phase1_weights: tuple[jax.Array, jax.Array] | None = None,
    precomputed_phase1: tuple[Any, ...] | None = None,
) -> tuple[tuple[jax.Array, jax.Array], tuple[Any, ...]]:
  """Runs 1 sublayer in 8 `program_order` phases (`F1`..`F8`)."""
  batch_off = h_mb0.shape[0]
  eff_layer_idx = sublayer_idx if layer_idx is None else layer_idx
  wrap_pt = bool(
      cfg.use_pallas_kernels
      and cfg.ep_size > 1
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_weights
      and cfg.moe_fp8_gmm_compute
      and cfg.moe_use_forked_3d_gmmv2
      and cfg.moe_bwd_dy_fp8_e5m2
  )

  @jax.named_call
  @program_order(enforce=False)
  def phase1(h0_in):
    normed_0, codes_l_0, scale_0, w_ep_0, ids_ep_0 = _ubatch_norm_and_metadata(
        h0_in,
        w_curr_sub.post_attn_norm,
        w_curr_sub.moe.router_gate,
        cfg,
        sublayer_idx=sublayer_idx,
        batch_offset=0,
        layer_idx=eff_layer_idx,
        step=step,
    )
    return (normed_0, codes_l_0, scale_0, w_ep_0, ids_ep_0), ()

  save_gate_up = (sublayer_idx in cfg.save_moe_gate_up_sublayers) and wrap_pt
  use_ext_bwd_disp = bool(save_gate_up and cfg.save_moe_gate_up_recompute_disp)
  use_fp8_disp_fwd = bool(
      cfg.ep_size > 1
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
      and cfg.moe_fp8_token_dispatch
  )
  save_routing = bool(
      cfg.save_moe_routing
      and not save_gate_up
      and (sublayer_idx not in cfg.save_moe_disp_ep_sublayers)
      and (not cfg.use_pallas_kernels or use_fp8_disp_fwd)
  )
  save_curr_attn_res = bool(
      cfg.save_attn_residuals_in_lineage and is_attn_curr
  )
  save_curr_gdn_res = bool(
      (not is_attn_curr) and (sublayer_idx in cfg.save_gdn_residuals_sublayers)
  )
  save_next_attn_res = bool(
      cfg.save_attn_residuals_in_lineage
      and is_attn_next
      and w_next_sub is not None
  )
  save_next_gdn_res = bool(
      (not is_attn_next)
      and (((sublayer_idx + 1) % 4) in cfg.save_gdn_residuals_sublayers)
      and w_next_sub is not None
  )

  @jax.named_call
  @program_order(enforce=False)
  def phase2(codes_l_0, scale_0, w_ep_0, ids_ep_0, x1_in):
    disp_0 = _ubatch_dispatch_tokens(
        codes_l_0,
        scale_0,
        w_ep_0,
        ids_ep_0,
        x1_in.dtype,
        cfg,
        gather_meta=cfg.defer_meta_ag_to_dispatch,
    )
    if save_curr_attn_res:
      h1_out, attn_res_1 = layers.full_attn_mixer_fwd_save_res(
          x1_in,
          w_curr_sub,
          pos_mb1,
          seg_mb1,
          cfg,
          rope_cos_sin=rope_mb1,
      )
    elif save_curr_gdn_res:
      h1_out, attn_res_1 = layers.gdn_mixer_fwd_save_res(
          x1_in,
          w_curr_sub,
          seg_mb1,
          cfg,
          precomputed_meta=gdn_meta_mb1,
      )
    else:
      h1_out = _ubatch_mixer(
          x1_in,
          w_curr_sub,
          pos_mb1,
          seg_mb1,
          cfg,
          is_attn=is_attn_curr,
          rope_cos_sin=rope_mb1 if is_attn_curr else None,
          gdn_meta=gdn_meta_mb1 if not is_attn_curr else None,
      )
      attn_res_1 = None
    return (disp_0, h1_out, attn_res_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase3(disp_0, h1_in):
    disp_0_w = _ubatch_wait(disp_0)
    normed_1, codes_l_1, scale_1, w_ep_1, ids_ep_1 = _ubatch_norm_and_metadata(
        h1_in,
        w_curr_sub.post_attn_norm,
        w_curr_sub.moe.router_gate,
        cfg,
        sublayer_idx=sublayer_idx,
        batch_offset=batch_off,
        layer_idx=eff_layer_idx,
        step=step,
    )
    return (disp_0_w, normed_1, codes_l_1, scale_1, w_ep_1, ids_ep_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase4(disp_0, codes_l_1, scale_1, w_ep_1, ids_ep_1):
    disp_1 = _ubatch_dispatch_tokens(
        codes_l_1,
        scale_1,
        w_ep_1,
        ids_ep_1,
        h_mb0.dtype,
        cfg,
        gather_meta=cfg.defer_meta_ag_to_dispatch,
    )
    if save_gate_up:
      (unperm_0, _, _, _), vjp_moe_0 = jax.vjp(
          lambda d_tok, d_w, wi_w, wo_w: layers.moe_expert_gmm_and_unsort(
              layers.DispatchedMoeTokens(
                  tokens_ep=d_tok,
                  topk_weights_ep=d_w,
                  topk_ids_ep=disp_0.topk_ids_ep,
              ),
              wi_w,
              wo_w,
              cfg,
              return_weights=True,
              wrap_partial_fp8_bwd=wrap_pt,
              sublayer_idx=sublayer_idx,
              use_external_bwd_rw=True,
              use_external_bwd_disp=use_ext_bwd_disp,
          ),
          disp_0.tokens_ep,
          disp_0.topk_weights_ep,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
      )
      routing_0 = None
    elif save_routing:
      unperm_0, routing_0 = layers.moe_expert_gmm_and_unsort(
          disp_0,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
          cfg,
          return_weights=False,
          wrap_partial_fp8_bwd=wrap_pt,
          sublayer_idx=sublayer_idx,
          return_routing=True,
      )
      vjp_moe_0 = None
    else:
      unperm_0 = layers.moe_expert_gmm_and_unsort(
          disp_0,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
          cfg,
          return_weights=False,
          wrap_partial_fp8_bwd=wrap_pt,
          sublayer_idx=sublayer_idx,
      )
      vjp_moe_0 = None
      routing_0 = None
    return (unperm_0, disp_1, vjp_moe_0, routing_0), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase5(unperm_0, disp_1, h0_in, normed_0, h1_in, normed_1):
    comb_0 = _ubatch_combine_fwd(unperm_0, cfg)
    if save_gate_up:
      (unperm_1, _), vjp_moe_1 = jax.vjp(
          lambda d_tok, d_w, wi_w, wo_w: layers.moe_expert_gmm_and_unsort(
              layers.DispatchedMoeTokens(
                  tokens_ep=d_tok,
                  topk_weights_ep=d_w,
                  topk_ids_ep=disp_1.topk_ids_ep,
              ),
              wi_w,
              wo_w,
              cfg,
              return_weights=False,
              wrap_partial_fp8_bwd=wrap_pt,
              sublayer_idx=sublayer_idx,
              use_external_bwd_rw=True,
              use_external_bwd_disp=use_ext_bwd_disp,
          ),
          disp_1.tokens_ep,
          disp_1.topk_weights_ep,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
      )
      routing_1 = None
    elif save_routing:
      unperm_1, routing_1 = layers.moe_expert_gmm_and_unsort(
          disp_1,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
          cfg,
          return_weights=False,
          wrap_partial_fp8_bwd=wrap_pt,
          sublayer_idx=sublayer_idx,
          return_routing=True,
      )
      vjp_moe_1 = None
    else:
      unperm_1 = layers.moe_expert_gmm_and_unsort(
          disp_1,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
          cfg,
          return_weights=False,
          wrap_partial_fp8_bwd=wrap_pt,
          sublayer_idx=sublayer_idx,
      )
      vjp_moe_1 = None
      routing_1 = None
    # Run BOTH mb0 and mb1 shared-expert MLPs inside phase5 to hide `comb_0`
    # SC0 combine latency and keep shared-expert weights hot in cache.
    base_0 = layers.shared_expert_and_residual(
        h0_in, normed_0, w_curr_sub.moe
    )
    base_1 = layers.shared_expert_and_residual(
        h1_in, normed_1, w_curr_sub.moe
    )
    return (comb_0, base_0, base_1, unperm_1, vjp_moe_1, routing_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase6(comb_0, base_0, unperm_1):
    x_out_0 = base_0 + comb_0.reshape(base_0.shape).astype(base_0.dtype)
    unperm_1_w = _ubatch_wait(unperm_1)
    return (x_out_0, unperm_1_w), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(x_out_0, unperm_1):
    comb_1 = _ubatch_combine_fwd(unperm_1, cfg)
    if w_next_sub is not None:
      if save_next_attn_res:
        h_next_0, attn_res_next_0 = layers.full_attn_mixer_fwd_save_res(
            x_out_0,
            w_next_sub,
            pos_mb0,
            seg_mb0,
            cfg,
            rope_cos_sin=rope_mb0,
        )
      elif save_next_gdn_res:
        h_next_0, attn_res_next_0 = layers.gdn_mixer_fwd_save_res(
            x_out_0,
            w_next_sub,
            seg_mb0,
            cfg,
            precomputed_meta=gdn_meta_mb0,
        )
      else:
        h_next_0 = _ubatch_mixer(
            x_out_0,
            w_next_sub,
            pos_mb0,
            seg_mb0,
            cfg,
            is_attn=is_attn_next,
            rope_cos_sin=rope_mb0 if is_attn_next else None,
            gdn_meta=gdn_meta_mb0 if not is_attn_next else None,
        )
        attn_res_next_0 = None
    else:
      h_next_0 = x_out_0
      attn_res_next_0 = None
    if w_next_phase1_weights is not None:
      next_post_attn_norm, next_router_gate = w_next_phase1_weights
      next_phase1 = _ubatch_norm_and_metadata(
          h_next_0,
          next_post_attn_norm,
          next_router_gate,
          cfg,
          sublayer_idx=sublayer_idx + 1,
          batch_offset=0,
          layer_idx=eff_layer_idx + 1,
          step=step,
      )
    else:
      next_phase1 = None
    return (h_next_0, comb_1, attn_res_next_0, next_phase1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(h_next_0, comb_1, base_1, next_phase1):
    h_next_0_w = _ubatch_wait(h_next_0)
    next_phase1_w = (
        _ubatch_wait(next_phase1) if next_phase1 is not None else None
    )
    x_out_1 = base_1 + comb_1.reshape(base_1.shape).astype(base_1.dtype)
    return (h_next_0_w, x_out_1, next_phase1_w), ()

  @program_order(enforce=True)
  def _layer_fwd(h0_in, x1_in):
    if precomputed_phase1 is not None:
      normed_0, codes_l_0, scale_0, w_ep_0, ids_ep_0 = precomputed_phase1
    else:
      (normed_0, codes_l_0, scale_0, w_ep_0, ids_ep_0), _ = phase1(h0_in)
    (disp_0, h1_mid, attn_res_1), _ = phase2(
        codes_l_0, scale_0, w_ep_0, ids_ep_0, x1_in
    )
    (disp_0, normed_1, codes_l_1, scale_1, w_ep_1, ids_ep_1), _ = phase3(
        disp_0, h1_mid
    )
    (unperm_0, disp_1, vjp_moe_0, routing_0), _ = phase4(
        disp_0, codes_l_1, scale_1, w_ep_1, ids_ep_1
    )
    (comb_0, base_0, base_1, unperm_1, vjp_moe_1, routing_1), _ = phase5(
        unperm_0, disp_1, h0_in, normed_0, h1_mid, normed_1
    )
    (x0_out, unperm_1), _ = phase6(comb_0, base_0, unperm_1)
    (h_next_0, comb_1, attn_res_next_0, next_phase1), _ = phase7(
        x0_out, unperm_1
    )
    (h_next_0, x_out_1, next_phase1), _ = phase8(
        h_next_0, comb_1, base_1, next_phase1
    )
    return (h_next_0, x_out_1), (
        x0_out,
        disp_0,
        disp_1,
        vjp_moe_0,
        vjp_moe_1,
        routing_0,
        routing_1,
        scale_0,
        scale_1,
        attn_res_1,
        h1_mid,
        attn_res_next_0,
        next_phase1,
    )

  (h_next_0, x_out_1), (
      x_out_0,
      disp_0,
      disp_1,
      vjp_moe_0,
      vjp_moe_1,
      routing_0,
      routing_1,
      scale_0,
      scale_1,
      attn_res_1,
      h1_mid,
      attn_res_next_0,
      next_phase1,
  ) = _layer_fwd(h_mb0, x_mb1)

  saved_disp = (
      (disp_0, disp_1)
      if (not save_gate_up and sublayer_idx in cfg.save_moe_disp_ep_sublayers)
      else None
  )
  saved_moe_vjp = (
      (vjp_moe_0, disp_0.topk_ids_ep, vjp_moe_1, disp_1.topk_ids_ep)
      if save_gate_up
      else None
  )
  saved_routing_pair = (
      jax.tree.map(
          lambda v: jax.lax.stop_gradient(v) if isinstance(v, jax.Array) else v,
          (routing_0, routing_1),
      )
      if save_routing
      else None
  )
  saved_scales = (
      (scale_0, scale_1)
      if (not save_gate_up or cfg.save_moe_gate_up_recompute_disp)
      else (None, None)
  )
  save_curr_mixer_res = save_curr_attn_res or save_curr_gdn_res
  save_next_mixer_res = save_next_attn_res or save_next_gdn_res
  saved_attn_res = (
      (
          attn_res_1,
          h1_mid if save_curr_mixer_res else None,
          attn_res_next_0,
      )
      if (save_curr_mixer_res or save_next_mixer_res)
      else None
  )
  res_sub = (
      h_mb0 if save_h_mb0 else None,
      x_mb1,
      x_out_0,
      saved_disp,
      saved_moe_vjp,
      saved_scales,
      saved_attn_res,
      saved_routing_pair,
  )
  return (h_next_0, x_out_1), res_sub, next_phase1


def _sublayer_bwd_8phase(
    g_next_0: jax.Array,
    g_x_out_1: jax.Array,
    res_sub: tuple[Any, ...],
    w_curr_sub: Any,
    w_next_sub: Any | None,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn_curr: bool,
    is_attn_next: bool,
    sublayer_idx: int,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    gdn_meta_mb0: Any = None,
    gdn_meta_mb1: Any = None,
    deferred_reduce_attn: pipelined_scan.DeferredReduce | None = None,
    deferred_reduce_moe: pipelined_scan.DeferredReduce | None = None,
    prev_part_grads: Any = None,
    x0_in_remat: jax.Array | None = None,
    bypass_next_mixer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, Any, Any, Any, Any]:
  """Runs 1 sublayer backward in 8 `program_order` phases (`B8`..`B1`).

  When `prev_part_grads` is supplied, `phase2` fuses `mb0 + mb1` mixer wgrad
  adds (`prev_part_grads + g_w_next`) so XLA emits `convolution_add_fusion` and
  returns the complete unreduced `g_next_merged` for the caller to launch SC1 RS
  OUTSIDE `_layer_bwd` (preventing `@program_order(enforce=True)` from trapping
  `wi`/`wo` or dense RS `async-done` at phase boundaries).

  `deferred_reduce_attn` (in `phase5`) and `deferred_reduce_moe` (in `phase7`)
  run ONLY the small EP-axis SC0 `AllReduce` leaves inside `_layer_bwd`;
  `finish()` runs outside `_layer_bwd` so SC1 RS of `wi`/`wo` and FSDP+EP dense
  projections overlaps all 8 phases (`phase1..phase8`).

  Returns:
    `(g_h_mb0, g_x_mb1, g_curr_part, g_next_out, reduced_attn, reduced_moe)`
    where `g_curr_part` has unreduced `(post_attn_norm, mb1_mixer, moe)` grads
    for `w_curr_sub`.
  """
  (
      h_mb0,
      x_mb1,
      x_out_0,
      saved_disp,
      saved_moe_vjp,
      saved_scales,
      saved_attn_res,
      saved_routing_pair,
  ) = res_sub
  scale_0_sv, scale_1_sv = saved_scales
  attn_res_1, h_mb1_sv, attn_res_next_0 = (
      saved_attn_res if saved_attn_res is not None else (None, None, None)
  )
  if h_mb0 is None:
    assert x0_in_remat is not None
    h_mb0 = _ubatch_mixer(
        x0_in_remat,
        w_curr_sub,
        pos_mb0,
        seg_mb0,
        cfg,
        is_attn=is_attn_curr,
        rope_cos_sin=rope_mb0 if is_attn_curr else None,
        gdn_meta=gdn_meta_mb0 if not is_attn_curr else None,
    )
  use_ext_bwd_disp = bool(
      saved_moe_vjp is not None and cfg.save_moe_gate_up_recompute_disp
  )
  ext_bwd_rw = (
      ops.BwdRoutedWeightPrefetch(
          wi_codes=w_curr_sub.moe.wi.codes,
          wi_scale=w_curr_sub.moe.wi.scale,
          wo_codes=w_curr_sub.moe.wo.codes,
          wo_scale=w_curr_sub.moe.wo.scale,
      )
      if saved_moe_vjp is not None
      else None
  )
  batch_off = h_mb0.shape[0]
  eff_layer_idx = sublayer_idx if layer_idx is None else layer_idx
  wrap_pt = bool(
      cfg.use_pallas_kernels
      and cfg.ep_size > 1
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_weights
      and cfg.moe_fp8_gmm_compute
      and cfg.moe_use_forked_3d_gmmv2
      and cfg.moe_bwd_dy_fp8_e5m2
  )
  use_fp8_disp = bool(
      cfg.ep_size > 1
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
      and cfg.moe_fp8_token_dispatch
  )
  t_ep_mb = x_mb1.shape[0] * x_mb1.shape[1] * max(1, cfg.ep_size)
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
  tok_local_shape = h_mb0.shape
  sc0_dy_pmax = bool(
      wrap_pt
      and cfg.ep_size > 1
      and cfg.sc0_bwd_dy_pmax
      and cfg.use_pallas_kernels
      and cfg.ep_sc_pinning == "sc0"
  )
  has_saved_h1 = attn_res_1 is not None and h_mb1_sv is not None
  has_saved_routing = saved_routing_pair is not None
  pipe_ext_disp = bool(
      cfg.pipeline_bwd_remat_dispatch
      and ((saved_moe_vjp is not None and use_ext_bwd_disp) or has_saved_routing)
  )
  pipe_full_disp = bool(
      cfg.pipeline_bwd_remat_dispatch
      and saved_moe_vjp is None
      and saved_disp is None
      and not has_saved_routing
  )

  @jax.named_call
  @program_order(enforce=False)
  def phase1(g_n0, g_x1):
    g_n0_w = _ubatch_wait(g_n0)
    if sc0_dy_pmax:
      amax_l_1 = jax.lax.stop_gradient(
          jnp.max(jnp.abs(g_x1.astype(jnp.float32)))
      )
      amax_ep_1 = ops.sc_pmax(amax_l_1, cfg.ep_axis_names, core_id=0)
    else:
      amax_ep_1 = None
    return (g_n0_w, g_x1, g_x1, amax_ep_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase2(g_n0_w, g_comb_1, amax_ep_1):
    if w_next_sub is not None:
      if attn_res_next_0 is not None and is_attn_next:
        g_x_out_0, g_w_next_raw = layers.full_attn_mixer_bwd_from_res(
            g_n0_w,
            x_out_0,
            attn_res_next_0,
            w_next_sub,
            pos_mb0,
            seg_mb0,
            cfg,
            rope_cos_sin=rope_mb0,
        )
      elif attn_res_next_0 is not None and not is_attn_next:
        g_x_out_0, g_w_next_raw = layers.gdn_mixer_bwd_from_res(
            g_n0_w,
            x_out_0,
            attn_res_next_0,
            w_next_sub,
            seg_mb0,
            cfg,
            precomputed_meta=gdn_meta_mb0,
        )
      else:
        _, vjp_next_mixer = jax.vjp(
            lambda x, w: _ubatch_mixer(
                x,
                w,
                pos_mb0,
                seg_mb0,
                cfg,
                is_attn=is_attn_next,
                rope_cos_sin=rope_mb0 if is_attn_next else None,
                gdn_meta=gdn_meta_mb0 if not is_attn_next else None,
            ),
            x_out_0,
            w_next_sub,
        )
        g_x_out_0, g_w_next_raw = vjp_next_mixer(g_n0_w)
      g_w_next_mixer = (
          _extract_attn_mixer_grad(g_w_next_raw)
          if is_attn_next
          else _extract_gdn_mixer_grad(g_w_next_raw)
      )
      if bypass_next_mixer is not None:
        g_x_out_0 = jnp.where(bypass_next_mixer, g_n0_w, g_x_out_0)
        g_w_next_mixer = jax.tree.map(
            lambda g: jnp.where(bypass_next_mixer, jnp.zeros_like(g), g),
            g_w_next_mixer,
        )
    else:
      g_x_out_0 = g_n0_w
      g_w_next_mixer = None
    # Fuse mb0 + mb1 mixer wgrad accumulation inside phase2 so XLA emits
    # `convolution_add_fusion` and frees `g_w_next_mixer` immediately.
    g_next_merged = (
        _merge_sublayer_grads(prev_part_grads, g_w_next_mixer)
        if prev_part_grads is not None
        else g_w_next_mixer
    )
    g_unperm_1 = _ubatch_combine_bwd(
        g_comb_1,
        cfg,
        as_partial_tokens=wrap_pt,
        unperm_shape=unperm_shape,
        precomputed_amax_ep=amax_ep_1,
    )
    if pipe_ext_disp and has_saved_h1:
      normed_1_p2 = ops.rms_norm_zero_centered(
          h_mb1_sv, w_curr_sub.post_attn_norm, cfg.norm_eps
      )
      prefetched_disp_1 = _ubatch_remat_dispatch_tokens_only(
          normed_1_p2, scale_1_sv, cfg
      )
    elif pipe_full_disp and has_saved_h1:
      normed_1_p2, topk_w_1_p2, topk_ids_1_p2 = _ubatch_norm_and_route(
          h_mb1_sv,
          w_curr_sub.post_attn_norm,
          w_curr_sub.moe.router_gate,
          cfg,
          sublayer_idx=sublayer_idx,
          batch_offset=batch_off,
          layer_idx=eff_layer_idx,
          step=step,
      )
      prefetched_disp_1 = _ubatch_remat_dispatch(
          normed_1_p2, topk_w_1_p2, topk_ids_1_p2, scale_1_sv, cfg
      )
    else:
      prefetched_disp_1 = None
    return (g_x_out_0, g_next_merged, g_unperm_1, prefetched_disp_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase3(g_x0, g_unperm_1):
    g_unperm_1_w = _ubatch_wait(g_unperm_1)
    if sc0_dy_pmax:
      amax_l_0 = jax.lax.stop_gradient(
          jnp.max(jnp.abs(g_x0.astype(jnp.float32)))
      )
      amax_ep_0 = ops.sc_pmax(amax_l_0, cfg.ep_axis_names, core_id=0)
    else:
      amax_ep_0 = None
    if has_saved_h1:
      h_mb1 = h_mb1_sv
      vjp_mixer_1 = None
      prefetched_disp_0_p3 = None
    else:
      h_mb1, vjp_mixer_1 = jax.vjp(
          lambda x, w: _ubatch_mixer(
              x,
              w,
              pos_mb1,
              seg_mb1,
              cfg,
              is_attn=is_attn_curr,
              rope_cos_sin=rope_mb1 if is_attn_curr else None,
              gdn_meta=gdn_meta_mb1 if not is_attn_curr else None,
          ),
          x_mb1,
          w_curr_sub,
      )
      if pipe_ext_disp:
        normed_0_p3 = ops.rms_norm_zero_centered(
            h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
        )
        prefetched_disp_0_p3 = _ubatch_remat_dispatch_tokens_only(
            normed_0_p3, scale_0_sv, cfg
        )
      elif pipe_full_disp:
        normed_0_p3, topk_w_0_p3, topk_ids_0_p3 = _ubatch_norm_and_route(
            h_mb0,
            w_curr_sub.post_attn_norm,
            w_curr_sub.moe.router_gate,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=0,
            layer_idx=eff_layer_idx,
            step=step,
        )
        prefetched_disp_0_p3 = _ubatch_remat_dispatch(
            normed_0_p3, topk_w_0_p3, topk_ids_0_p3, scale_0_sv, cfg
        )
      else:
        prefetched_disp_0_p3 = None
    return (
        g_x0,
        g_x0,
        g_unperm_1_w,
        h_mb1,
        vjp_mixer_1,
        amax_ep_0,
        prefetched_disp_0_p3,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase4(
      g_comb_0,
      g_base_0,
      g_base_1,
      g_unperm_1,
      h_mb1,
      amax_ep_0,
      prefetched_disp_1,
  ):
    g_unperm_0 = _ubatch_combine_bwd(
        g_comb_0,
        cfg,
        as_partial_tokens=wrap_pt,
        unperm_shape=unperm_shape,
        precomputed_amax_ep=amax_ep_0,
    )
    normed_1 = ops.rms_norm_zero_centered(
        h_mb1, w_curr_sub.post_attn_norm, cfg.norm_eps
    )
    prefetched_disp_0_p4 = None
    g_h0_se_p4 = None
    g_n0_se_p4 = None
    if saved_moe_vjp is not None:
      _, _, vjp_moe_1, topk_ids_1_sv = saved_moe_vjp
      if use_ext_bwd_disp:
        if prefetched_disp_1 is not None:
          remat_disp_1 = prefetched_disp_1
        else:
          remat_disp_1 = _ubatch_remat_dispatch_tokens_only(
              normed_1, scale_1_sv, cfg
          )
        assert isinstance(remat_disp_1, ops.Fp8DispatchedTokens)
        ext_rw_1 = dataclasses.replace(
            ext_bwd_rw,
            disp_codes_ep=remat_disp_1.codes_ep,
            disp_scale=remat_disp_1.scale,
        )
        if pipe_ext_disp and has_saved_h1:
          normed_0_p4 = ops.rms_norm_zero_centered(
              h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
          )
          prefetched_disp_0_p4 = _ubatch_remat_dispatch_tokens_only(
              normed_0_p4, scale_0_sv, cfg
          )
      else:
        ext_rw_1 = ext_bwd_rw
      _, vjp_se_1 = jax.vjp(
          lambda h_in, n_in, w_moe: layers.shared_expert_and_residual(
              h_in, n_in, w_moe
          ),
          h_mb1,
          normed_1,
          w_curr_sub.moe,
      )
      g_h1_se, g_n1_se, g_w_moe_1 = vjp_se_1(g_base_1)
      g_dtok_1, g_dw_1, g_wi_1, g_wo_1 = vjp_moe_1((g_unperm_1, ext_rw_1))
      g_w_moe_1 = dataclasses.replace(g_w_moe_1, wi=g_wi_1, wo=g_wo_1)
      g_disp_1 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_1, topk_weights_ep=g_dw_1, topk_ids_ep=topk_ids_1_sv
      )
    elif has_saved_routing:
      _, routing_1_sv = saved_routing_pair
      if prefetched_disp_1 is not None:
        disp_tok_1 = prefetched_disp_1
      else:
        disp_tok_1 = _ubatch_remat_dispatch_tokens_only(
            normed_1, scale_1_sv, cfg
        )
      if pipe_ext_disp:
        normed_0_p4 = ops.rms_norm_zero_centered(
            h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
        )
        if has_saved_h1:
          prefetched_disp_0_p4 = _ubatch_remat_dispatch_tokens_only(
              normed_0_p4, scale_0_sv, cfg
          )
        _, vjp_se_1 = jax.vjp(
            lambda h_in, n_in, w_moe: layers.shared_expert_and_residual(
                h_in, n_in, w_moe
            ),
            h_mb1,
            normed_1,
            w_curr_sub.moe,
        )
        g_h1_se, g_n1_se, g_w_se_1 = vjp_se_1(g_base_1)
        _, vjp_se_0 = jax.vjp(
            lambda h_in, n_in, w_moe: layers.shared_expert_and_residual(
                h_in, n_in, w_moe
            ),
            h_mb0,
            normed_0_p4,
            w_curr_sub.moe,
        )
        g_h0_se_p4, g_n0_se_p4, g_w_se_0 = vjp_se_0(g_base_0)

        def _experts_1_sr(d_tok, w_r, wi_w, wo_w):
          return layers.moe_expert_gmm_and_unsort_from_routing(
              d_tok,
              w_r,
              routing_1_sv,
              wi_w,
              wo_w,
              cfg,
              return_weights=False,
              wrap_partial_fp8_bwd=wrap_pt,
              sublayer_idx=sublayer_idx,
          )

        _, vjp_e1_sr = jax.vjp(
            _experts_1_sr,
            disp_tok_1,
            routing_1_sv.w_rows.astype(jnp.float32),
            w_curr_sub.moe.wi,
            w_curr_sub.moe.wo,
        )
        g_dtok_1, g_w_rows_1, g_wi_1, g_wo_1 = vjp_e1_sr(g_unperm_1)
        g_dw_1 = layers.unsort_moe_row_weights(
            g_w_rows_1, routing_1_sv, t_ep_mb, cfg
        )
        g_w_se_sum = jax.tree.map(
            lambda a, b: a + b,
            dataclasses.replace(g_w_se_0, wi=None, wo=None),
            dataclasses.replace(g_w_se_1, wi=None, wo=None),
        )
        g_w_moe_1 = dataclasses.replace(g_w_se_sum, wi=g_wi_1, wo=g_wo_1)
      else:

        def _se_and_experts_1_sr(h_in, n_in, d_tok, w_r, w_moe):
          b = layers.shared_expert_and_residual(h_in, n_in, w_moe)
          u = layers.moe_expert_gmm_and_unsort_from_routing(
              d_tok,
              w_r,
              routing_1_sv,
              w_moe.wi,
              w_moe.wo,
              cfg,
              return_weights=False,
              wrap_partial_fp8_bwd=wrap_pt,
              sublayer_idx=sublayer_idx,
          )
          return b, u

        _, vjp_1_sr = jax.vjp(
            _se_and_experts_1_sr,
            h_mb1,
            normed_1,
            disp_tok_1,
            routing_1_sv.w_rows.astype(jnp.float32),
            w_curr_sub.moe,
        )
        g_h1_se, g_n1_se, g_dtok_1, g_w_rows_1, g_w_moe_1 = vjp_1_sr(
            (g_base_1, g_unperm_1)
        )
        g_dw_1 = layers.unsort_moe_row_weights(
            g_w_rows_1, routing_1_sv, t_ep_mb, cfg
        )
      g_disp_1 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_1,
          topk_weights_ep=g_dw_1,
          topk_ids_ep=jnp.zeros((0,), dtype=jnp.int32),
      )
    elif pipe_full_disp:
      if prefetched_disp_1 is not None:
        disp_1 = prefetched_disp_1
      else:
        _, topk_w_1, topk_ids_1 = _ubatch_norm_and_route(
            h_mb1,
            w_curr_sub.post_attn_norm,
            w_curr_sub.moe.router_gate,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=batch_off,
            layer_idx=eff_layer_idx,
            step=step,
        )
        disp_1 = _ubatch_remat_dispatch(
            normed_1, topk_w_1, topk_ids_1, scale_1_sv, cfg
        )
      normed_0_p4 = ops.rms_norm_zero_centered(
          h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
      )
      if has_saved_h1:
        _, topk_w_0_p4, topk_ids_0_p4 = _ubatch_norm_and_route(
            h_mb0,
            w_curr_sub.post_attn_norm,
            w_curr_sub.moe.router_gate,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=0,
            layer_idx=eff_layer_idx,
            step=step,
        )
        prefetched_disp_0_p4 = _ubatch_remat_dispatch(
            normed_0_p4, topk_w_0_p4, topk_ids_0_p4, scale_0_sv, cfg
        )
      # Run BOTH mb1 and mb0 shared-expert VJPs in phase4 so they overlap
      # behind `disp_1` SC0 AG when `disp_1` is rematerialized in phase4.
      _, vjp_se_1 = jax.vjp(
          lambda h_in, n_in, w_moe: layers.shared_expert_and_residual(
              h_in, n_in, w_moe
          ),
          h_mb1,
          normed_1,
          w_curr_sub.moe,
      )
      g_h1_se, g_n1_se, g_w_se_1 = vjp_se_1(g_base_1)
      _, vjp_se_0 = jax.vjp(
          lambda h_in, n_in, w_moe: layers.shared_expert_and_residual(
              h_in, n_in, w_moe
          ),
          h_mb0,
          normed_0_p4,
          w_curr_sub.moe,
      )
      g_h0_se_p4, g_n0_se_p4, g_w_se_0 = vjp_se_0(g_base_0)

      def _experts_1(d_tok, d_w, wi_w, wo_w):
        d = layers.DispatchedMoeTokens(
            tokens_ep=d_tok, topk_weights_ep=d_w, topk_ids_ep=disp_1.topk_ids_ep
        )
        return layers.moe_expert_gmm_and_unsort(
            d,
            wi_w,
            wo_w,
            cfg,
            return_weights=False,
            wrap_partial_fp8_bwd=wrap_pt,
            sublayer_idx=sublayer_idx,
        )

      _, vjp_e1 = jax.vjp(
          _experts_1,
          disp_1.tokens_ep,
          disp_1.topk_weights_ep,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
      )
      g_dtok_1, g_dw_1, g_wi_1, g_wo_1 = vjp_e1(g_unperm_1)
      g_w_se_sum = jax.tree.map(
          lambda a, b: a + b,
          dataclasses.replace(g_w_se_0, wi=None, wo=None),
          dataclasses.replace(g_w_se_1, wi=None, wo=None),
      )
      g_w_moe_1 = dataclasses.replace(g_w_se_sum, wi=g_wi_1, wo=g_wo_1)
      g_disp_1 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_1,
          topk_weights_ep=g_dw_1,
          topk_ids_ep=disp_1.topk_ids_ep,
      )
    else:
      if saved_disp is not None:
        _, disp_1 = saved_disp
      else:
        _, topk_w_1, topk_ids_1 = _ubatch_norm_and_route(
            h_mb1,
            w_curr_sub.post_attn_norm,
            w_curr_sub.moe.router_gate,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=batch_off,
            layer_idx=eff_layer_idx,
            step=step,
        )
        disp_1 = _ubatch_remat_dispatch(
            normed_1, topk_w_1, topk_ids_1, scale_1_sv, cfg
        )

      def _se_and_experts_1(h_in, n_in, d_tok, d_w, w_moe):
        b = layers.shared_expert_and_residual(h_in, n_in, w_moe)
        d = layers.DispatchedMoeTokens(
            tokens_ep=d_tok, topk_weights_ep=d_w, topk_ids_ep=disp_1.topk_ids_ep
        )
        u = layers.moe_expert_gmm_and_unsort(
            d,
            w_moe.wi,
            w_moe.wo,
            cfg,
            return_weights=False,
            wrap_partial_fp8_bwd=wrap_pt,
            sublayer_idx=sublayer_idx,
        )
        return b, u

      _, vjp_1 = jax.vjp(
          _se_and_experts_1,
          h_mb1,
          normed_1,
          disp_1.tokens_ep,
          disp_1.topk_weights_ep,
          w_curr_sub.moe,
      )
      g_h1_se, g_n1_se, g_dtok_1, g_dw_1, g_w_moe_1 = vjp_1(
          (g_base_1, g_unperm_1)
      )
      g_disp_1 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_1,
          topk_weights_ep=g_dw_1,
          topk_ids_ep=disp_1.topk_ids_ep,
      )
    return (
        g_unperm_0,
        g_h1_se,
        g_n1_se,
        g_disp_1,
        g_w_moe_1,
        prefetched_disp_0_p4,
        g_h0_se_p4,
        g_n0_se_p4,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase5(
      g_disp_1,
      g_base_0,
      g_unperm_0,
      g_w_moe_1,
      prefetched_disp_0,
      g_h0_se_in,
      g_n0_se_in,
  ):
    g_n1_routed, g_topk_w_1, dr_out_attn = _ubatch_dispatch_bwd_and_reduce(
        g_disp_1,
        deferred_reduce_attn,
        cfg,
        use_fp8_disp=use_fp8_disp,
        tok_local_shape=tok_local_shape,
        finish_deferred=False,
    )
    if saved_moe_vjp is not None:
      normed_0 = ops.rms_norm_zero_centered(
          h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
      )
      vjp_moe_0, topk_ids_0_sv, _, _ = saved_moe_vjp
      if use_ext_bwd_disp:
        if prefetched_disp_0 is not None:
          remat_disp_0 = prefetched_disp_0
        else:
          remat_disp_0 = _ubatch_remat_dispatch_tokens_only(
              normed_0, scale_0_sv, cfg
          )
        assert isinstance(remat_disp_0, ops.Fp8DispatchedTokens)
        ext_rw_0 = dataclasses.replace(
            ext_bwd_rw,
            disp_codes_ep=remat_disp_0.codes_ep,
            disp_scale=remat_disp_0.scale,
        )
      else:
        ext_rw_0 = ext_bwd_rw
      _, vjp_se_0 = jax.vjp(
          lambda h_in, n_in, w_moe: layers.shared_expert_and_residual(
              h_in, n_in, w_moe
          ),
          h_mb0,
          normed_0,
          w_curr_sub.moe,
      )
      g_h0_se, g_n0_se, g_w_moe_0 = vjp_se_0(g_base_0)
      g_dtok_0, g_dw_0, g_wi, g_wo = vjp_moe_0(
          (g_unperm_0, g_w_moe_1.wi, g_w_moe_1.wo, ext_rw_0)
      )
      g_w_moe_rest = jax.tree.map(
          lambda a, b: a + b,
          dataclasses.replace(g_w_moe_0, wi=None, wo=None),
          dataclasses.replace(g_w_moe_1, wi=None, wo=None),
      )
      g_w_moe = dataclasses.replace(g_w_moe_rest, wi=g_wi, wo=g_wo)
      g_disp_0 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_0, topk_weights_ep=g_dw_0, topk_ids_ep=topk_ids_0_sv
      )
    elif has_saved_routing:
      routing_0_sv, _ = saved_routing_pair
      if pipe_ext_disp:
        disp_tok_0 = prefetched_disp_0

        def _experts_0_sr(d_tok, w_r, wi_w, wo_w):
          return layers.moe_expert_gmm_and_unsort_from_routing(
              d_tok,
              w_r,
              routing_0_sv,
              wi_w,
              wo_w,
              cfg,
              return_weights=True,
              wrap_partial_fp8_bwd=wrap_pt,
              sublayer_idx=sublayer_idx,
          )

        _, vjp_e0_sr = jax.vjp(
            _experts_0_sr,
            disp_tok_0,
            routing_0_sv.w_rows.astype(jnp.float32),
            w_curr_sub.moe.wi,
            w_curr_sub.moe.wo,
        )
        g_dtok_0, g_w_rows_0, g_wi, g_wo = vjp_e0_sr(
            (g_unperm_0, g_w_moe_1.wi, g_w_moe_1.wo)
        )
        g_dw_0 = layers.unsort_moe_row_weights(
            g_w_rows_0, routing_0_sv, t_ep_mb, cfg
        )
        g_h0_se = g_h0_se_in
        g_n0_se = g_n0_se_in
        g_w_moe = dataclasses.replace(g_w_moe_1, wi=g_wi, wo=g_wo)
      else:
        normed_0 = ops.rms_norm_zero_centered(
            h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
        )
        disp_tok_0 = _ubatch_remat_dispatch_tokens_only(
            normed_0, scale_0_sv, cfg
        )

        def _se_and_experts_0_sr(h_in, n_in, d_tok, w_r, w_moe):
          b = layers.shared_expert_and_residual(h_in, n_in, w_moe)
          u, wi_c, wo_c = layers.moe_expert_gmm_and_unsort_from_routing(
              d_tok,
              w_r,
              routing_0_sv,
              w_moe.wi,
              w_moe.wo,
              cfg,
              return_weights=True,
              wrap_partial_fp8_bwd=wrap_pt,
              sublayer_idx=sublayer_idx,
          )
          return b, u, wi_c, wo_c

        _, vjp_0_sr = jax.vjp(
            _se_and_experts_0_sr,
            h_mb0,
            normed_0,
            disp_tok_0,
            routing_0_sv.w_rows.astype(jnp.float32),
            w_curr_sub.moe,
        )
        g_h0_se, g_n0_se, g_dtok_0, g_w_rows_0, g_w_moe = vjp_0_sr(
            (g_base_0, g_unperm_0, g_w_moe_1.wi, g_w_moe_1.wo)
        )
        g_dw_0 = layers.unsort_moe_row_weights(
            g_w_rows_0, routing_0_sv, t_ep_mb, cfg
        )
        g_w_moe_rest = jax.tree.map(
            lambda a, b: a + b,
            dataclasses.replace(g_w_moe, wi=None, wo=None),
            dataclasses.replace(g_w_moe_1, wi=None, wo=None),
        )
        g_w_moe = dataclasses.replace(g_w_moe_rest, wi=g_w_moe.wi, wo=g_w_moe.wo)
      g_disp_0 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_0,
          topk_weights_ep=g_dw_0,
          topk_ids_ep=jnp.zeros((0,), dtype=jnp.int32),
      )
    elif pipe_full_disp:
      disp_0 = prefetched_disp_0

      def _experts_0(d_tok, d_w, wi_w, wo_w):
        d = layers.DispatchedMoeTokens(
            tokens_ep=d_tok, topk_weights_ep=d_w, topk_ids_ep=disp_0.topk_ids_ep
        )
        return layers.moe_expert_gmm_and_unsort(
            d,
            wi_w,
            wo_w,
            cfg,
            return_weights=True,
            wrap_partial_fp8_bwd=wrap_pt,
            sublayer_idx=sublayer_idx,
        )

      _, vjp_e0 = jax.vjp(
          _experts_0,
          disp_0.tokens_ep,
          disp_0.topk_weights_ep,
          w_curr_sub.moe.wi,
          w_curr_sub.moe.wo,
      )
      g_dtok_0, g_dw_0, g_wi, g_wo = vjp_e0(
          (g_unperm_0, g_w_moe_1.wi, g_w_moe_1.wo)
      )
      g_h0_se = g_h0_se_in
      g_n0_se = g_n0_se_in
      g_w_moe = dataclasses.replace(g_w_moe_1, wi=g_wi, wo=g_wo)
      g_disp_0 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_0,
          topk_weights_ep=g_dw_0,
          topk_ids_ep=disp_0.topk_ids_ep,
      )
    else:
      normed_0 = ops.rms_norm_zero_centered(
          h_mb0, w_curr_sub.post_attn_norm, cfg.norm_eps
      )
      if saved_disp is not None:
        disp_0, _ = saved_disp
      else:
        _, topk_w_0, topk_ids_0 = _ubatch_norm_and_route(
            h_mb0,
            w_curr_sub.post_attn_norm,
            w_curr_sub.moe.router_gate,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=0,
            layer_idx=eff_layer_idx,
            step=step,
        )
        disp_0 = _ubatch_remat_dispatch(
            normed_0, topk_w_0, topk_ids_0, scale_0_sv, cfg
        )

      def _se_and_experts_0(h_in, n_in, d_tok, d_w, w_moe):
        b = layers.shared_expert_and_residual(h_in, n_in, w_moe)
        d = layers.DispatchedMoeTokens(
            tokens_ep=d_tok, topk_weights_ep=d_w, topk_ids_ep=disp_0.topk_ids_ep
        )
        u, wi_c, wo_c = layers.moe_expert_gmm_and_unsort(
            d,
            w_moe.wi,
            w_moe.wo,
            cfg,
            return_weights=True,
            wrap_partial_fp8_bwd=wrap_pt,
            sublayer_idx=sublayer_idx,
        )
        return b, u, wi_c, wo_c

      _, vjp_0 = jax.vjp(
          _se_and_experts_0,
          h_mb0,
          normed_0,
          disp_0.tokens_ep,
          disp_0.topk_weights_ep,
          w_curr_sub.moe,
      )
      g_h0_se, g_n0_se, g_dtok_0, g_dw_0, g_w_moe = vjp_0(
          (g_base_0, g_unperm_0, g_w_moe_1.wi, g_w_moe_1.wo)
      )
      g_w_moe_rest = jax.tree.map(
          lambda a, b: a + b,
          dataclasses.replace(g_w_moe, wi=None, wo=None),
          dataclasses.replace(g_w_moe_1, wi=None, wo=None),
      )
      g_w_moe = dataclasses.replace(g_w_moe_rest, wi=g_w_moe.wi, wo=g_w_moe.wo)
      g_disp_0 = layers.DispatchedMoeTokens(
          tokens_ep=g_dtok_0,
          topk_weights_ep=g_dw_0,
          topk_ids_ep=disp_0.topk_ids_ep,
      )
    return (
        g_n1_routed,
        g_topk_w_1,
        dr_out_attn,
        g_h0_se,
        g_n0_se,
        g_disp_0,
        g_w_moe,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase6(mb0_bundle, g_n1_routed, g_n1_se, g_topk_w_1, h_mb1):
    mb0_bundle_w = _ubatch_wait(mb0_bundle)
    _, vjp_nr_1 = jax.vjp(
        lambda h_in, pn, rg: _ubatch_norm_and_route(
            h_in,
            pn,
            rg,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=batch_off,
            layer_idx=eff_layer_idx,
            step=step,
        )[:2],
        h_mb1,
        w_curr_sub.post_attn_norm,
        w_curr_sub.moe.router_gate,
    )
    g_h1_nr, g_pn_1, g_rg_1 = vjp_nr_1((g_n1_routed + g_n1_se, g_topk_w_1))
    return (
        mb0_bundle_w,
        g_h1_nr,
        g_pn_1,
        g_rg_1,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(g_disp_0, g_h1_total, vjp_mixer_1):
    g_n0_routed, g_topk_w_0, dr_out_moe = _ubatch_dispatch_bwd_and_reduce(
        g_disp_0,
        deferred_reduce_moe,
        cfg,
        use_fp8_disp=use_fp8_disp,
        tok_local_shape=tok_local_shape,
        finish_deferred=False,
    )
    if attn_res_1 is not None and is_attn_curr:
      g_x_mb1, g_w_curr_raw_1 = layers.full_attn_mixer_bwd_from_res(
          g_h1_total,
          x_mb1,
          attn_res_1,
          w_curr_sub,
          pos_mb1,
          seg_mb1,
          cfg,
          rope_cos_sin=rope_mb1,
      )
    elif attn_res_1 is not None and not is_attn_curr:
      g_x_mb1, g_w_curr_raw_1 = layers.gdn_mixer_bwd_from_res(
          g_h1_total,
          x_mb1,
          attn_res_1,
          w_curr_sub,
          seg_mb1,
          cfg,
          precomputed_meta=gdn_meta_mb1,
      )
    else:
      g_x_mb1, g_w_curr_raw_1 = vjp_mixer_1(g_h1_total)
    g_w_curr_mixer_1 = (
        _extract_attn_mixer_grad(g_w_curr_raw_1)
        if is_attn_curr
        else _extract_gdn_mixer_grad(g_w_curr_raw_1)
    )
    return (
        g_n0_routed,
        g_topk_w_0,
        dr_out_moe,
        g_x_mb1,
        g_w_curr_mixer_1,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(g_n0_routed, g_n0_se, g_topk_w_0, g_x_mb1):
    g_x1_w = _ubatch_wait(g_x_mb1)
    _, vjp_nr_0 = jax.vjp(
        lambda h_in, pn, rg: _ubatch_norm_and_route(
            h_in,
            pn,
            rg,
            cfg,
            sublayer_idx=sublayer_idx,
            batch_offset=0,
            layer_idx=eff_layer_idx,
            step=step,
        )[:2],
        h_mb0,
        w_curr_sub.post_attn_norm,
        w_curr_sub.moe.router_gate,
    )
    g_h0_nr, g_pn_0, g_rg_0 = vjp_nr_0((g_n0_routed + g_n0_se, g_topk_w_0))
    return (g_h0_nr, g_pn_0, g_rg_0, g_x1_w), ()

  @program_order(enforce=True)
  def _layer_bwd(g_n0_in, g_x1_in):
    (g_n0_w, g_base_1, g_comb_1, amax_ep_1), _ = phase1(g_n0_in, g_x1_in)
    (g_x_out_0, g_next_merged, g_unperm_1, prefetched_disp_1), _ = phase2(
        g_n0_w, g_comb_1, amax_ep_1
    )
    (
        g_comb_0,
        g_base_0,
        g_unperm_1,
        h_mb1,
        vjp_mixer_1,
        amax_ep_0,
        prefetched_disp_0_p3,
    ), _ = phase3(g_x_out_0, g_unperm_1)
    (
        g_unperm_0,
        g_h1_se,
        g_n1_se,
        g_disp_1,
        g_w_moe_1,
        prefetched_disp_0_p4,
        g_h0_se_p4,
        g_n0_se_p4,
    ), _ = phase4(
        g_comb_0,
        g_base_0,
        g_base_1,
        g_unperm_1,
        h_mb1,
        amax_ep_0,
        prefetched_disp_1,
    )
    prefetched_disp_0 = (
        prefetched_disp_0_p4
        if prefetched_disp_0_p4 is not None
        else prefetched_disp_0_p3
    )
    (
        g_n1_routed,
        g_topk_w_1,
        dr_out_attn,
        g_h0_se,
        g_n0_se,
        g_disp_0,
        g_w_moe,
    ), _ = phase5(
        g_disp_1,
        g_base_0,
        g_unperm_0,
        g_w_moe_1,
        prefetched_disp_0,
        g_h0_se_p4,
        g_n0_se_p4,
    )
    ((g_h0_se, g_n0_se, g_disp_0, g_w_moe), g_h1_nr, g_pn_1, g_rg_1), _ = (
        phase6(
            (g_h0_se, g_n0_se, g_disp_0, g_w_moe),
            g_n1_routed,
            g_n1_se,
            g_topk_w_1,
            h_mb1,
        )
    )
    g_h1_total = g_h1_se + g_h1_nr
    (g_n0_routed, g_topk_w_0, dr_out_moe, g_x_mb1, g_w_curr_mixer_1), _ = (
        phase7(g_disp_0, g_h1_total, vjp_mixer_1)
    )
    (g_h0_nr, g_pn_0, g_rg_0, g_x_mb1), _ = phase8(
        g_n0_routed, g_n0_se, g_topk_w_0, g_x_mb1
    )
    g_h_mb0 = g_h0_se + g_h0_nr
    return (
        g_h_mb0,
        g_x_mb1,
        g_w_moe,
        g_rg_0,
        g_rg_1,
        g_pn_0,
        g_pn_1,
        g_w_curr_mixer_1,
        g_next_merged,
        dr_out_attn,
        dr_out_moe,
    ), ()

  (
      g_h_mb0,
      g_x_mb1,
      g_w_moe,
      g_rg_0,
      g_rg_1,
      g_pn_0,
      g_pn_1,
      g_w_curr_mixer_1,
      g_next_merged,
      dr_out_attn,
      dr_out_moe,
  ), _ = _layer_bwd(g_next_0, g_x_out_1)

  # Stitch deferred EP-axis AR outputs back into full reduced sublayer trees
  # OUTSIDE `_layer_bwd` so non-AR SC1 RS leaves (`wi`/`wo`, dense FSDP+EP
  # projections) overlap all 8 phases without program-order stalls.
  reduced_attn = (
      deferred_reduce_attn.finish(dr_out_attn)
      if deferred_reduce_attn is not None
      else None
  )
  reduced_moe = (
      deferred_reduce_moe.finish(dr_out_moe)
      if deferred_reduce_moe is not None
      else None
  )

  g_wi_raw = (
      g_w_moe.wi.w_ste if hasattr(g_w_moe.wi, "w_ste") else g_w_moe.wi
  )
  g_wo_raw = (
      g_w_moe.wo.w_ste if hasattr(g_w_moe.wo, "w_ste") else g_w_moe.wo
  )
  g_moe = dataclasses.replace(
      g_w_moe,
      wi=g_wi_raw,
      wo=g_wo_raw,
      router_gate=g_w_moe.router_gate + g_rg_0 + g_rg_1,
  )
  g_post_norm = g_pn_0 + g_pn_1
  g_curr_part = dataclasses.replace(
      g_w_curr_mixer_1, post_attn_norm=g_post_norm, moe=g_moe
  )
  return (
      g_h_mb0,
      g_x_mb1,
      g_curr_part,
      g_next_merged,
      reduced_attn,
      reduced_moe,
  )


def _combine_mixer_and_moe(
    mixer_p: Any,
    post_attn_norm: jax.Array,
    moe_p: params_lib.MoeParams,
    *,
    is_attn: bool,
) -> Any:
  if is_attn:
    return params_lib.AttnSublayerParams(
        pre_attn_norm=mixer_p.pre_attn_norm,
        post_attn_norm=post_attn_norm,
        q_proj=mixer_p.q_proj,
        k_proj=mixer_p.k_proj,
        v_proj=mixer_p.v_proj,
        q_norm=mixer_p.q_norm,
        k_norm=mixer_p.k_norm,
        out_proj=mixer_p.out_proj,
        moe=moe_p,
    )
  return params_lib.GdnSublayerParams(
      pre_attn_norm=mixer_p.pre_attn_norm,
      post_attn_norm=post_attn_norm,
      in_proj_qkvz=mixer_p.in_proj_qkvz,
      in_proj_ba=mixer_p.in_proj_ba,
      conv1d=mixer_p.conv1d,
      A_log=mixer_p.A_log,
      dt_bias=mixer_p.dt_bias,
      gdn_norm=mixer_p.gdn_norm,
      out_proj=mixer_p.out_proj,
      moe=moe_p,
  )


def _merge_sublayer_grads(g_part: Any, g_mixer: Any | None) -> Any:
  if g_mixer is None:
    return g_part
  return pipelined_scan.combine_w_grads(g_mixer, g_part)


def _superblock_fwd(
    carry: tuple[jax.Array, jax.Array, params_lib.GdnSublayerParams],
    w_sb_curr: params_lib.SuperblockParams,
    w_next_gdn0_s: params_lib.GdnSublayerParams | None,
    aux_curr: Any,
    aux_next_gdn0: Any,
    *,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    rope_mb0: tuple[jax.Array, jax.Array] | None,
    rope_mb1: tuple[jax.Array, jax.Array] | None,
    cfg: QwenConfig,
    sb_idx: jax.Array | int = 0,
    step: jax.Array | int = 0,
    bypass_next_mixer: jax.Array | None = None,
    gdn_meta_mb0: Any = None,
    gdn_meta_mb1: Any = None,
) -> tuple[tuple[jax.Array, jax.Array, params_lib.GdnSublayerParams | None], Any]:
  del bypass_next_mixer
  h_mb0, x_mb1, w0_full_in = carry
  a1 = getattr(aux_curr, "gdn_1", None) if aux_curr is not None else None
  a2 = getattr(aux_curr, "gdn_2", None) if aux_curr is not None else None
  a3 = getattr(aux_curr, "attn_3", None) if aux_curr is not None else None
  base_layer_idx = sb_idx * 4

  defer_comb1 = bool(cfg.lineage_defer_mb1_combine)

  @program_order(enforce=False)
  def sub0(h0_in, x1_in, w0_full):
    # Prefetch next sublayer mixer FIRST (needed at phase7), then MoE SECOND
    # (overlaps phase1..8, needed at sub1/phase1).
    w1_mix = pipelined_scan.collect_sublayer_mixer(
        w_sb_curr.gdn_1, cfg, is_attn=False
    )
    w1_moe = pipelined_scan.collect_sublayer_moe(w_sb_curr.gdn_1, a1, cfg)
    w1_full = _combine_mixer_and_moe(
        w1_mix, w_sb_curr.gdn_1.post_attn_norm, w1_moe, is_attn=False
    )
    (h0_out, x1_out), r0, p1_next = _sublayer_fwd_8phase(
        h0_in,
        x1_in,
        w0_full,
        w1_mix,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn_curr=False,
        is_attn_next=False,
        sublayer_idx=0,
        layer_idx=base_layer_idx + 0,
        step=step,
        gdn_meta_mb0=gdn_meta_mb0,
        gdn_meta_mb1=gdn_meta_mb1,
        w_next_phase1_weights=(
            (w1_full.post_attn_norm, w1_full.moe.router_gate)
            if defer_comb1
            else None
        ),
    )
    return (h0_out, x1_out, w1_full, r0, p1_next), ()

  @program_order(enforce=False)
  def sub1(h0_in, x1_in, w1_full, p1_in):
    w2_mix = pipelined_scan.collect_sublayer_mixer(
        w_sb_curr.gdn_2, cfg, is_attn=False
    )
    w2_moe = pipelined_scan.collect_sublayer_moe(w_sb_curr.gdn_2, a2, cfg)
    w2_full = _combine_mixer_and_moe(
        w2_mix, w_sb_curr.gdn_2.post_attn_norm, w2_moe, is_attn=False
    )
    (h0_out, x1_out), r1, p1_next = _sublayer_fwd_8phase(
        h0_in,
        x1_in,
        w1_full,
        w2_mix,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn_curr=False,
        is_attn_next=False,
        sublayer_idx=1,
        layer_idx=base_layer_idx + 1,
        step=step,
        gdn_meta_mb0=gdn_meta_mb0,
        gdn_meta_mb1=gdn_meta_mb1,
        w_next_phase1_weights=(
            (w2_full.post_attn_norm, w2_full.moe.router_gate)
            if defer_comb1
            else None
        ),
        precomputed_phase1=p1_in,
    )
    return (h0_out, x1_out, w2_full, r1, p1_next), ()

  @program_order(enforce=False)
  def sub2(h0_in, x1_in, w2_full, p1_in):
    w3_mix = pipelined_scan.collect_sublayer_mixer(
        w_sb_curr.attn_3, cfg, is_attn=True
    )
    w3_moe = pipelined_scan.collect_sublayer_moe(w_sb_curr.attn_3, a3, cfg)
    w3_full = _combine_mixer_and_moe(
        w3_mix, w_sb_curr.attn_3.post_attn_norm, w3_moe, is_attn=True
    )
    (h0_out, x1_out), r2, p1_next = _sublayer_fwd_8phase(
        h0_in,
        x1_in,
        w2_full,
        w3_mix,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn_curr=False,
        is_attn_next=True,
        sublayer_idx=2,
        layer_idx=base_layer_idx + 2,
        step=step,
        rope_mb0=rope_mb0,
        rope_mb1=rope_mb1,
        gdn_meta_mb0=gdn_meta_mb0,
        gdn_meta_mb1=gdn_meta_mb1,
        w_next_phase1_weights=(
            (w3_full.post_attn_norm, w3_full.moe.router_gate)
            if defer_comb1
            else None
        ),
        precomputed_phase1=p1_in,
    )
    return (h0_out, x1_out, w3_full, r2, p1_next), ()

  @program_order(enforce=False)
  def sub3(h0_in, x1_in, w3_full, p1_in):
    if w_next_gdn0_s is not None:
      w_next0_mix = pipelined_scan.collect_sublayer_mixer(
          w_next_gdn0_s, cfg, is_attn=False
      )
      w_next0_moe = pipelined_scan.collect_sublayer_moe(
          w_next_gdn0_s, aux_next_gdn0, cfg
      )
      w_next0_full = _combine_mixer_and_moe(
          w_next0_mix, w_next_gdn0_s.post_attn_norm, w_next0_moe, is_attn=False
      )
    else:
      w_next0_mix = None
      w_next0_full = None
    (h0_out, x1_out), r3, _ = _sublayer_fwd_8phase(
        h0_in,
        x1_in,
        w3_full,
        w_next0_mix,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn_curr=True,
        is_attn_next=False,
        sublayer_idx=3,
        layer_idx=base_layer_idx + 3,
        step=step,
        rope_mb0=rope_mb0,
        rope_mb1=rope_mb1,
        gdn_meta_mb0=gdn_meta_mb0,
        gdn_meta_mb1=gdn_meta_mb1,
        precomputed_phase1=p1_in,
    )
    return (h0_out, x1_out, w_next0_full, r3), ()

  @program_order(enforce=True)
  def _sb_fwd(h0_in, x1_in, w0_full):
    (h0_c, x1_c, w1_full, res0, p1_1), _ = sub0(h0_in, x1_in, w0_full)
    (h0_c, x1_c, w2_full, res1, p1_2), _ = sub1(h0_c, x1_c, w1_full, p1_1)
    (h0_c, x1_c, w3_full, res2, p1_3), _ = sub2(h0_c, x1_c, w2_full, p1_2)
    (h0_c, x1_c, w_next0_full, res3), _ = sub3(h0_c, x1_c, w3_full, p1_3)
    return (h0_c, x1_c, w_next0_full), (res0, res1, res2, res3)

  return _sb_fwd(h_mb0, x_mb1, w0_full_in)


def _reduce_sublayer_complete(
    g_unred: Any,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    defer_replicated_ar: bool = False,
    skip_routed_wi_wo: bool = False,
) -> Any:
  """Runs FSDP `ReduceScatter` (SC1) then EP-axis `AllReduce` (SC0) for 1 sublayer."""
  g_sc1 = pipelined_scan.reduce_sublayer_sc1(
      g_unred,
      cfg,
      is_attn=is_attn,
      defer_replicated_ar=defer_replicated_ar,
      skip_routed_wi_wo=skip_routed_wi_wo,
  )
  dr = pipelined_scan.reduce_sublayer_sc0(
      g_sc1,
      cfg,
      is_attn=is_attn,
      defer_replicated_ar=defer_replicated_ar,
  )
  if dr is not None:
    return dr.finish(
        pipelined_scan.sc_compute_on(dr.local_fn, core_id=0)(dr.operand)
    )
  return g_sc1


def _superblock_bwd_step(
    res_sb: Any,
    g_carry: tuple[jax.Array, jax.Array],
    w_sb_curr: params_lib.SuperblockParams,
    w_prev_attn3_s: params_lib.AttnSublayerParams,
    aux_curr: Any,
    aux_prev_attn3: Any,
    g_next_gdn0_unred: params_lib.GdnSublayerParams,
    g_prev_gdn1_merged: params_lib.GdnSublayerParams,
    w_curr_attn3_full: params_lib.AttnSublayerParams,
    w_next0_mix_in: params_lib.GdnSublayerParams,
    *,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    rope_mb0: tuple[jax.Array, jax.Array] | None,
    rope_mb1: tuple[jax.Array, jax.Array] | None,
    cfg: QwenConfig,
    sb_idx: jax.Array | int = 0,
    step: jax.Array | int = 0,
    bypass_next_mixer: jax.Array | None = None,
    gdn_meta_mb0: Any = None,
    gdn_meta_mb1: Any = None,
) -> tuple[
    tuple[jax.Array, jax.Array],
    params_lib.GdnSublayerParams,
    params_lib.GdnSublayerParams,
    params_lib.AttnSublayerParams,
    params_lib.GdnSublayerParams,
    params_lib.GdnSublayerParams,
    params_lib.GdnSublayerParams,
    params_lib.AttnSublayerParams,
    params_lib.GdnSublayerParams,
]:
  """Symmetric 4-sublayer backward pass with 1 AG + 1 full RS + 1 deferred AR per sublayer.

  Each sublayer launches SC1 `reduce_sublayer_sc1` + `collect_sublayer_full`
  OUTSIDE `_sublayer_bwd_8phase` so `wi`/`wo` and FSDP+EP dense RS plus next
  weight AG overlap all 8 phases (`phase1..phase8`), while dense/narrow EP-axis
  SC0 `AllReduce` runs in `phase7` overlapped with `vjp_mixer_1`.
  """
  res0, res1, res2, res3 = res_sb
  g_next_0, g_x_out_1 = g_carry
  a0 = getattr(aux_curr, "gdn_0", None) if aux_curr is not None else None
  a1 = getattr(aux_curr, "gdn_1", None) if aux_curr is not None else None
  a2 = getattr(aux_curr, "gdn_2", None) if aux_curr is not None else None
  base_layer_idx = sb_idx * 4

  @program_order(enforce=False)
  def sub3(g0_in, g1_in, w3_full, w_next0_mix, g_next0_u, g_prev_gdn1_m):
    g_1_sc1 = pipelined_scan.reduce_sublayer_sc1(
        g_prev_gdn1_m,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
        skip_routed_wi_wo=True,
    )
    dr_1 = pipelined_scan.reduce_sublayer_sc0(
        g_1_sc1,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
    )
    # Routed MoE `wi`/`wo` gradients are complete at the end of their own
    # sublayer; launch their SC1 ReduceScatter 1 sublayer earlier (here for
    # `g_next0_u` while `sub3` merges `mb0` mixer grads) so peak concurrent
    # unreduced `dwi`/`dwo` drops from 4 -> 2 (-6+ GiB HBM + lower heap frag).
    dw_next0_wi, dw_next0_wo = pipelined_scan.reduce_routed_wi_wo_sc1(
        g_next0_u.moe.wi, g_next0_u.moe.wo, cfg
    )
    w2_full = pipelined_scan.collect_sublayer_full(
        w_sb_curr.gdn_2, a2, cfg, is_attn=False
    )
    g_next0_no_routed = dataclasses.replace(
        g_next0_u, moe=dataclasses.replace(g_next0_u.moe, wi=None, wo=None)
    )
    g0_out, g1_out, g_3_part, g_next0_merged, _, dw_prev_gdn1_ep = (
        _sublayer_bwd_8phase(
            g0_in,
            g1_in,
            res3,
            w3_full,
            w_next0_mix,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn_curr=True,
            is_attn_next=False,
            sublayer_idx=3,
            layer_idx=base_layer_idx + 3,
            step=step,
            rope_mb0=rope_mb0,
            rope_mb1=rope_mb1,
            gdn_meta_mb0=gdn_meta_mb0,
            gdn_meta_mb1=gdn_meta_mb1,
            deferred_reduce_moe=dr_1,
            prev_part_grads=g_next0_no_routed,
            x0_in_remat=res2[2],
            bypass_next_mixer=bypass_next_mixer,
        )
    )
    g_next0_merged = dataclasses.replace(
        g_next0_merged,
        moe=dataclasses.replace(
            g_next0_merged.moe, wi=dw_next0_wi, wo=dw_next0_wo
        ),
    )
    dw_prev_gdn1 = (
        dw_prev_gdn1_ep if dw_prev_gdn1_ep is not None else g_1_sc1
    )
    return (
        g0_out,
        g1_out,
        dataclasses.replace(w3_full, moe=None),
        w2_full,
        g_3_part,
        g_next0_merged,
        dw_prev_gdn1,
    ), ()

  @program_order(enforce=False)
  def sub2(g0_in, g1_in, w3_mix, w2_full, g_3_part, g_next0_merged):
    g_next0_sc1 = pipelined_scan.reduce_sublayer_sc1(
        g_next0_merged,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
        skip_routed_wi_wo=True,
    )
    dr_next0 = pipelined_scan.reduce_sublayer_sc0(
        g_next0_sc1,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
    )
    dw_3_wi, dw_3_wo = pipelined_scan.reduce_routed_wi_wo_sc1(
        g_3_part.moe.wi, g_3_part.moe.wo, cfg
    )
    w1_full = pipelined_scan.collect_sublayer_full(
        w_sb_curr.gdn_1, a1, cfg, is_attn=False
    )
    g_3_no_routed = dataclasses.replace(
        g_3_part, moe=dataclasses.replace(g_3_part.moe, wi=None, wo=None)
    )
    g0_out, g1_out, g_2_part, g_3_merged, _, dw_next_gdn0_ep = (
        _sublayer_bwd_8phase(
            g0_in,
            g1_in,
            res2,
            w2_full,
            w3_mix,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn_curr=False,
            is_attn_next=True,
            sublayer_idx=2,
            layer_idx=base_layer_idx + 2,
            step=step,
            rope_mb0=rope_mb0,
            rope_mb1=rope_mb1,
            gdn_meta_mb0=gdn_meta_mb0,
            gdn_meta_mb1=gdn_meta_mb1,
            deferred_reduce_moe=dr_next0,
            prev_part_grads=g_3_no_routed,
            x0_in_remat=res1[2],
        )
    )
    g_3_merged = dataclasses.replace(
        g_3_merged,
        moe=dataclasses.replace(g_3_merged.moe, wi=dw_3_wi, wo=dw_3_wo),
    )
    dw_next_gdn0 = (
        dw_next_gdn0_ep if dw_next_gdn0_ep is not None else g_next0_sc1
    )
    return (
        g0_out,
        g1_out,
        dataclasses.replace(w2_full, moe=None),
        w1_full,
        g_2_part,
        g_3_merged,
        dw_next_gdn0,
    ), ()

  @program_order(enforce=False)
  def sub1(g0_in, g1_in, w2_mix, w1_full, g_2_part, g_3_merged):
    g_3_sc1 = pipelined_scan.reduce_sublayer_sc1(
        g_3_merged,
        cfg,
        is_attn=True,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
        skip_routed_wi_wo=True,
    )
    dr_3 = pipelined_scan.reduce_sublayer_sc0(
        g_3_sc1,
        cfg,
        is_attn=True,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
    )
    dw_2_wi, dw_2_wo = pipelined_scan.reduce_routed_wi_wo_sc1(
        g_2_part.moe.wi, g_2_part.moe.wo, cfg
    )
    w0_full = pipelined_scan.collect_sublayer_full(
        w_sb_curr.gdn_0, a0, cfg, is_attn=False
    )
    g_2_no_routed = dataclasses.replace(
        g_2_part, moe=dataclasses.replace(g_2_part.moe, wi=None, wo=None)
    )
    g0_out, g1_out, g_1_part, g_2_merged, _, dw_attn3_ep = (
        _sublayer_bwd_8phase(
            g0_in,
            g1_in,
            res1,
            w1_full,
            w2_mix,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn_curr=False,
            is_attn_next=False,
            sublayer_idx=1,
            layer_idx=base_layer_idx + 1,
            step=step,
            gdn_meta_mb0=gdn_meta_mb0,
            gdn_meta_mb1=gdn_meta_mb1,
            deferred_reduce_moe=dr_3,
            prev_part_grads=g_2_no_routed,
            x0_in_remat=res0[2],
        )
    )
    g_2_merged = dataclasses.replace(
        g_2_merged,
        moe=dataclasses.replace(g_2_merged.moe, wi=dw_2_wi, wo=dw_2_wo),
    )
    dw_attn3 = dw_attn3_ep if dw_attn3_ep is not None else g_3_sc1
    return (
        g0_out,
        g1_out,
        dataclasses.replace(w1_full, moe=None),
        w0_full,
        g_1_part,
        g_2_merged,
        dw_attn3,
    ), ()

  @program_order(enforce=False)
  def sub0(g0_in, g1_in, w1_mix, w0_full, g_1_part, g_2_merged):
    g_2_sc1 = pipelined_scan.reduce_sublayer_sc1(
        g_2_merged,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
        skip_routed_wi_wo=True,
    )
    dr_2 = pipelined_scan.reduce_sublayer_sc0(
        g_2_sc1,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
    )
    dw_1_wi, dw_1_wo = pipelined_scan.reduce_routed_wi_wo_sc1(
        g_1_part.moe.wi, g_1_part.moe.wo, cfg
    )
    w_prev_attn3_full = pipelined_scan.collect_sublayer_full(
        w_prev_attn3_s, aux_prev_attn3, cfg, is_attn=True
    )
    g_1_no_routed = dataclasses.replace(
        g_1_part, moe=dataclasses.replace(g_1_part.moe, wi=None, wo=None)
    )
    g0_out, g1_out, g_0_part, g_1_merged, _, dw_gdn2_ep = (
        _sublayer_bwd_8phase(
            g0_in,
            g1_in,
            res0,
            w0_full,
            w1_mix,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn_curr=False,
            is_attn_next=False,
            sublayer_idx=0,
            layer_idx=base_layer_idx + 0,
            step=step,
            gdn_meta_mb0=gdn_meta_mb0,
            gdn_meta_mb1=gdn_meta_mb1,
            deferred_reduce_moe=dr_2,
            prev_part_grads=g_1_no_routed,
        )
    )
    g_1_merged = dataclasses.replace(
        g_1_merged,
        moe=dataclasses.replace(g_1_merged.moe, wi=dw_1_wi, wo=dw_1_wo),
    )
    dw_gdn2 = dw_gdn2_ep if dw_gdn2_ep is not None else g_2_sc1
    w0_mix = dataclasses.replace(w0_full, post_attn_norm=None, moe=None)
    return (
        g0_out,
        g1_out,
        g_0_part,
        g_1_merged,
        dw_gdn2,
        w_prev_attn3_full,
        w0_mix,
    ), ()

  @program_order(enforce=True)
  def _sb_bwd(g0_in, g1_in, w3_full, w_next0_mix, g_next0_u, g_prev_gdn1_m):
    (
        g0_c,
        g1_c,
        w3_mix,
        w2_full,
        g_3_part,
        g_next0_merged,
        dw_prev_gdn1,
    ), _ = sub3(g0_in, g1_in, w3_full, w_next0_mix, g_next0_u, g_prev_gdn1_m)
    (
        g0_c,
        g1_c,
        w2_mix,
        w1_full,
        g_2_part,
        g_3_merged,
        dw_next_gdn0,
    ), _ = sub2(g0_c, g1_c, w3_mix, w2_full, g_3_part, g_next0_merged)
    (
        g0_c,
        g1_c,
        w1_mix,
        w0_full,
        g_1_part,
        g_2_merged,
        dw_attn3,
    ), _ = sub1(g0_c, g1_c, w2_mix, w1_full, g_2_part, g_3_merged)
    (
        g0_c,
        g1_c,
        g_0_part,
        g_1_merged,
        dw_gdn2,
        w_prev_attn3_full,
        w0_mix,
    ), _ = sub0(g0_c, g1_c, w1_mix, w0_full, g_1_part, g_2_merged)
    return (
        (g0_c, g1_c),
        g_0_part,
        g_1_merged,
        w_prev_attn3_full,
        w0_mix,
        dw_prev_gdn1,
        dw_gdn2,
        dw_attn3,
        dw_next_gdn0,
    )

  return _sb_bwd(
      g_next_0,
      g_x_out_1,
      w_curr_attn3_full,
      w_next0_mix_in,
      g_next_gdn0_unred,
      g_prev_gdn1_merged,
  )


def run_qwen_lineage_scan(
    x_mb0: jax.Array,
    x_mb1: jax.Array,
    blocks_local: params_lib.SuperblockParams,
    hoisted_absmax: Any,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: QwenConfig,
    *,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array]:
  """Executes the full Lineage 8-phase pipelined scan over `blocks_local`."""
  if cfg.num_superblocks < 2:
    raise ValueError(
        "run_qwen_lineage_scan requires num_superblocks >= 2, got"
        f" {cfg.num_superblocks}"
    )

  step_in = jnp.asarray(step, dtype=jnp.int32)
  gdn_meta_mb0 = layers.precompute_gdn_meta(seg_mb0, cfg)
  gdn_meta_mb1 = layers.precompute_gdn_meta(seg_mb1, cfg)

  @jax.custom_vjp
  def _scanned_blocks(
      x0: jax.Array,
      x1: jax.Array,
      w_blocks: params_lib.SuperblockParams,
      w_aux: Any,
      step_arr: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    w_sb0_gdn0 = jax.tree.map(lambda a: a[0], w_blocks.gdn_0)
    a_sb0_gdn0 = (
        jax.tree.map(lambda a: a[0], w_aux.gdn_0)
        if w_aux is not None
        else None
    )
    w0_mix = pipelined_scan.collect_sublayer_mixer(
        w_sb0_gdn0, cfg, is_attn=False
    )
    w0_moe = pipelined_scan.collect_sublayer_moe(w_sb0_gdn0, a_sb0_gdn0, cfg)
    w0_full_init = _combine_mixer_and_moe(
        w0_mix, w_sb0_gdn0.post_attn_norm, w0_moe, is_attn=False
    )
    h0 = _ubatch_mixer(
        x0, w0_mix, pos_mb0, seg_mb0, cfg, is_attn=False, gdn_meta=gdn_meta_mb0
    )

    num_sb = cfg.num_superblocks
    sb_indices = jnp.arange(num_sb, dtype=jnp.int32)

    def _body_fwd(carry, xs):
      w_sb_c, a_c, idx = xs
      next_idx = jnp.where(idx < (num_sb - 1), idx + 1, jnp.int32(0))
      w_n0_s = jax.tree.map(
          lambda a: jax.lax.dynamic_index_in_dim(
              a, next_idx, axis=0, keepdims=False
          ),
          w_blocks.gdn_0,
      )
      a_n0 = (
          jax.tree.map(
              lambda a: jax.lax.dynamic_index_in_dim(
                  a, next_idx, axis=0, keepdims=False
              ),
              w_aux.gdn_0,
          )
          if w_aux is not None
          else None
      )
      (h_out, x1_out, w_n0_full), res_sb = _superblock_fwd(
          carry,
          w_sb_c,
          w_n0_s,
          a_c,
          a_n0,
          pos_mb0=pos_mb0,
          pos_mb1=pos_mb1,
          seg_mb0=seg_mb0,
          seg_mb1=seg_mb1,
          rope_mb0=rope_mb0,
          rope_mb1=rope_mb1,
          cfg=cfg,
          sb_idx=idx,
          step=step_arr,
          gdn_meta_mb0=gdn_meta_mb0,
          gdn_meta_mb1=gdn_meta_mb1,
      )
      h_out = jnp.where(idx < (num_sb - 1), h_out, res_sb[3][2])
      return (h_out, x1_out, w_n0_full), None

    (out0, out1, _), _ = jax.lax.scan(
        _body_fwd,
        (h0, x1, w0_full_init),
        (w_blocks, w_aux, sb_indices),
    )
    return out0, out1

  def _scanned_blocks_fwd(
      x0: jax.Array,
      x1: jax.Array,
      w_blocks: params_lib.SuperblockParams,
      w_aux: Any,
      step_arr: jax.Array,
  ):
    w_sb0_gdn0 = jax.tree.map(lambda a: a[0], w_blocks.gdn_0)
    a_sb0_gdn0 = (
        jax.tree.map(lambda a: a[0], w_aux.gdn_0)
        if w_aux is not None
        else None
    )
    w0_mix = pipelined_scan.collect_sublayer_mixer(
        w_sb0_gdn0, cfg, is_attn=False
    )
    w0_moe = pipelined_scan.collect_sublayer_moe(w_sb0_gdn0, a_sb0_gdn0, cfg)
    w0_full_init = _combine_mixer_and_moe(
        w0_mix, w_sb0_gdn0.post_attn_norm, w0_moe, is_attn=False
    )
    if 0 in cfg.save_gdn_residuals_sublayers:
      h0, pro_gdn_res = layers.gdn_mixer_fwd_save_res(
          x0, w0_mix, seg_mb0, cfg, precomputed_meta=gdn_meta_mb0
      )
    else:
      h0 = _ubatch_mixer(
          x0, w0_mix, pos_mb0, seg_mb0, cfg, is_attn=False, gdn_meta=gdn_meta_mb0
      )
      pro_gdn_res = None

    num_sb = cfg.num_superblocks
    sb_indices = jnp.arange(num_sb, dtype=jnp.int32)

    def _body_fwd(carry, xs):
      w_sb_c, a_c, idx = xs
      next_idx = jnp.where(idx < (num_sb - 1), idx + 1, jnp.int32(0))
      w_n0_s = jax.tree.map(
          lambda a: jax.lax.dynamic_index_in_dim(
              a, next_idx, axis=0, keepdims=False
          ),
          w_blocks.gdn_0,
      )
      a_n0 = (
          jax.tree.map(
              lambda a: jax.lax.dynamic_index_in_dim(
                  a, next_idx, axis=0, keepdims=False
              ),
              w_aux.gdn_0,
          )
          if w_aux is not None
          else None
      )
      (h_out, x1_out, w_n0_full), res_sb = _superblock_fwd(
          carry,
          w_sb_c,
          w_n0_s,
          a_c,
          a_n0,
          pos_mb0=pos_mb0,
          pos_mb1=pos_mb1,
          seg_mb0=seg_mb0,
          seg_mb1=seg_mb1,
          rope_mb0=rope_mb0,
          rope_mb1=rope_mb1,
          cfg=cfg,
          sb_idx=idx,
          step=step_arr,
          gdn_meta_mb0=gdn_meta_mb0,
          gdn_meta_mb1=gdn_meta_mb1,
      )
      h_out = jnp.where(idx < (num_sb - 1), h_out, res_sb[3][2])
      return (h_out, x1_out, w_n0_full), res_sb

    (out0, out1, _), res_scan = jax.lax.scan(
        _body_fwd,
        (h0, x1, w0_full_init),
        (w_blocks, w_aux, sb_indices),
    )
    return (out0, out1), (x0, w_blocks, w_aux, res_scan, step_arr, pro_gdn_res)

  def _scanned_blocks_bwd(res, g_outs):
    x0, w_blocks, w_aux, res_scan, step_arr, pro_gdn_res = res
    g_out0, g_out1 = g_outs

    num_sb = cfg.num_superblocks
    sb_indices = jnp.arange(num_sb, dtype=jnp.int32)
    w_sb0_g0 = jax.tree.map(lambda a: a[0], w_blocks.gdn_0)
    w_last_a3 = jax.tree.map(lambda a: a[-1], w_blocks.attn_3)
    a_last_a3 = (
        jax.tree.map(lambda a: a[-1], w_aux.attn_3)
        if w_aux is not None
        else None
    )
    w_last_attn3_full = pipelined_scan.collect_sublayer_full(
        w_last_a3, a_last_a3, cfg, is_attn=True
    )
    w_init_next0_mix = pipelined_scan.collect_sublayer_mixer(
        w_sb0_g0, cfg, is_attn=False
    )

    d_m = cfg.emb_dim
    m_r = cfg.moe_mlp_dim
    m_s = cfg.shared_mlp_dim
    e_loc = cfg.local_num_experts
    g_dtype = (
        w_sb0_g0.pre_attn_norm.dtype
        if cfg.use_pallas_kernels
        else jnp.float32
    )
    a_dt_dtype = (
        w_sb0_g0.A_log.dtype if cfg.use_pallas_kernels else jnp.float32
    )
    w0_moe_shape = jax.eval_shape(
        lambda w, a: pipelined_scan.collect_sublayer_moe(w, a, cfg),
        w_sb0_g0,
        jax.tree.map(lambda a: a[0], w_aux.gdn_0)
        if w_aux is not None
        else None,
    )
    if hasattr(w0_moe_shape.wi, "w_ste"):
      wi_zero = jnp.zeros(
          w0_moe_shape.wi.w_ste.shape, dtype=w0_moe_shape.wi.w_ste.dtype
      )
      wo_zero = jnp.zeros(
          w0_moe_shape.wo.w_ste.shape, dtype=w0_moe_shape.wo.w_ste.dtype
      )
    else:
      wi_zero = jnp.zeros((e_loc, d_m, 2 * m_r), dtype=g_dtype)
      wo_zero = jnp.zeros((e_loc, m_r, d_m), dtype=g_dtype)
    g_next_gdn0_zero = params_lib.GdnSublayerParams(
        pre_attn_norm=jnp.zeros((d_m,), dtype=g_dtype),
        post_attn_norm=jnp.zeros((d_m,), dtype=g_dtype),
        in_proj_qkvz=jnp.zeros((d_m, cfg.gdn_qkvz_dim), dtype=g_dtype),
        in_proj_ba=jnp.zeros((d_m, 2 * cfg.gdn_num_v_heads), dtype=g_dtype),
        conv1d=jnp.zeros(w_sb0_g0.conv1d.shape, dtype=g_dtype),
        A_log=jnp.zeros(w_sb0_g0.A_log.shape, dtype=a_dt_dtype),
        dt_bias=jnp.zeros(w_sb0_g0.dt_bias.shape, dtype=a_dt_dtype),
        gdn_norm=jnp.zeros(w_sb0_g0.gdn_norm.shape, dtype=g_dtype),
        out_proj=jnp.zeros((cfg.gdn_value_dim, d_m), dtype=g_dtype),
        moe=params_lib.MoeParams(
            router_gate=jnp.zeros((d_m, cfg.num_experts), dtype=g_dtype),
            wi=wi_zero,
            wo=wo_zero,
            shared_wi_0=jnp.zeros((d_m, m_s), dtype=g_dtype),
            shared_wi_1=jnp.zeros((d_m, m_s), dtype=g_dtype),
            shared_wo=jnp.zeros((m_s, d_m), dtype=g_dtype),
            shared_gate=jnp.zeros((d_m, 1), dtype=g_dtype),
        ),
    )

    dw_gdn_leaf_spec = jax.eval_shape(
        lambda g: _reduce_sublayer_complete(
            g,
            cfg,
            is_attn=False,
            defer_replicated_ar=cfg.hoist_replicated_grad_ar,
            skip_routed_wi_wo=False,
        ),
        g_next_gdn0_zero,
    )
    g_prev_gdn1_zero = dataclasses.replace(
        g_next_gdn0_zero,
        moe=dataclasses.replace(
            g_next_gdn0_zero.moe,
            wi=jnp.zeros(
                dw_gdn_leaf_spec.moe.wi.shape,
                dtype=dw_gdn_leaf_spec.moe.wi.dtype,
            ),
            wo=jnp.zeros(
                dw_gdn_leaf_spec.moe.wo.shape,
                dtype=dw_gdn_leaf_spec.moe.wo.dtype,
            ),
        ),
    )
    dw_gdn0_init = jax.tree.map(
        lambda s: jax.lax.broadcast(
            jnp.zeros((), dtype=s.dtype), (num_sb, *s.shape)
        ),
        dw_gdn_leaf_spec,
    )
    dw_gdn1_init = jax.tree.map(
        lambda s: jax.lax.broadcast(
            jnp.zeros((), dtype=s.dtype), (num_sb, *s.shape)
        ),
        dw_gdn_leaf_spec,
    )

    def _body_bwd(carry, xs):
      (
          g_c,
          g_next_gdn0_u,
          g_prev_gdn1_m,
          w_a3_full,
          w_n0_mix,
          dw_gdn0_buf,
          dw_gdn1_buf,
      ) = carry
      w_sb_c, a_c, res_c, idx = xs
      prev_idx = jnp.maximum(idx - 1, jnp.int32(0))
      w_pa3_s = jax.tree.map(
          lambda a: jax.lax.dynamic_index_in_dim(
              a, prev_idx, axis=0, keepdims=False
          ),
          w_blocks.attn_3,
      )
      a_pa3 = (
          jax.tree.map(
              lambda a: jax.lax.dynamic_index_in_dim(
                  a, prev_idx, axis=0, keepdims=False
              ),
              w_aux.attn_3,
          )
          if w_aux is not None
          else None
      )
      is_last = idx == (num_sb - 1)
      (
          g_c,
          g_0_u,
          g_1_m,
          w_prev_a3_full,
          w0_mix_out,
          dw_prev_1,
          dw_2,
          dw_3,
          dw_next_0,
      ) = _superblock_bwd_step(
          res_c,
          g_c,
          w_sb_c,
          w_pa3_s,
          a_c,
          a_pa3,
          g_next_gdn0_u,
          g_prev_gdn1_m,
          w_a3_full,
          w_n0_mix,
          pos_mb0=pos_mb0,
          pos_mb1=pos_mb1,
          seg_mb0=seg_mb0,
          seg_mb1=seg_mb1,
          rope_mb0=rope_mb0,
          rope_mb1=rope_mb1,
          cfg=cfg,
          sb_idx=idx,
          step=step_arr,
          bypass_next_mixer=is_last,
          gdn_meta_mb0=gdn_meta_mb0,
          gdn_meta_mb1=gdn_meta_mb1,
      )
      target_idx = jnp.where(idx < (num_sb - 1), idx + 1, jnp.int32(0))
      dw_gdn0_buf = jax.tree.map(
          lambda buf, val: jax.lax.dynamic_update_slice_in_dim(
              buf, val[None, ...], target_idx, axis=0
          ),
          dw_gdn0_buf,
          dw_next_0,
      )
      dw_gdn1_buf = jax.tree.map(
          lambda buf, val: jax.lax.dynamic_update_slice_in_dim(
              buf, val[None, ...], target_idx, axis=0
          ),
          dw_gdn1_buf,
          dw_prev_1,
      )
      return (
          g_c,
          g_0_u,
          g_1_m,
          w_prev_a3_full,
          w0_mix_out,
          dw_gdn0_buf,
          dw_gdn1_buf,
      ), (
          dw_2,
          dw_3,
      )

    (
        g_carry,
        g_sb0_gdn0_unred,
        g_sb0_gdn1_merged,
        _,
        w0_mix,
        dw_gdn0_buf,
        dw_gdn1_buf,
    ), (
        dw_gdn2,
        dw_attn3,
    ) = jax.lax.scan(
        _body_bwd,
        (
            (g_out0, g_out1),
            g_next_gdn0_zero,
            g_prev_gdn1_zero,
            w_last_attn3_full,
            w_init_next0_mix,
            dw_gdn0_init,
            dw_gdn1_init,
        ),
        (
            w_blocks,
            w_aux,
            res_scan,
            sb_indices,
        ),
        reverse=True,
    )

    g_h0, g_x1 = g_carry
    if pro_gdn_res is not None:
      g_x0, g_w0_pro_raw = layers.gdn_mixer_bwd_from_res(
          g_h0,
          x0,
          pro_gdn_res,
          w0_mix,
          seg_mb0,
          cfg,
          precomputed_meta=gdn_meta_mb0,
      )
    else:
      _, vjp_pro = jax.vjp(
          lambda x, w: _ubatch_mixer(
              x, w, pos_mb0, seg_mb0, cfg, is_attn=False, gdn_meta=gdn_meta_mb0
          ),
          x0,
          w0_mix,
      )
      g_x0, g_w0_pro_raw = vjp_pro(g_h0)
    g_w0_pro_mix = _extract_gdn_mixer_grad(g_w0_pro_raw)
    dw_sb0_gdn1 = _reduce_sublayer_complete(
        g_sb0_gdn1_merged,
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
        skip_routed_wi_wo=True,
    )
    dw_sb0_gdn0 = _reduce_sublayer_complete(
        _merge_sublayer_grads(g_sb0_gdn0_unred, g_w0_pro_mix),
        cfg,
        is_attn=False,
        defer_replicated_ar=cfg.hoist_replicated_grad_ar,
        skip_routed_wi_wo=False,
    )

    dw_gdn0 = jax.tree.map(
        lambda buf, val: jax.lax.dynamic_update_slice_in_dim(
            buf, val[None, ...], jnp.int32(0), axis=0
        ),
        dw_gdn0_buf,
        dw_sb0_gdn0,
    )
    dw_gdn1 = jax.tree.map(
        lambda buf, val: jax.lax.dynamic_update_slice_in_dim(
            buf, val[None, ...], jnp.int32(0), axis=0
        ),
        dw_gdn1_buf,
        dw_sb0_gdn1,
    )
    dw_blocks = params_lib.SuperblockParams(
        gdn_0=dw_gdn0, gdn_1=dw_gdn1, gdn_2=dw_gdn2, attn_3=dw_attn3
    )
    if cfg.hoist_replicated_grad_ar:
      dw_blocks = pipelined_scan.reduce_hoisted_replicated_grads(
          dw_blocks, cfg
      )
    return g_x0, g_x1, dw_blocks, None, None

  _scanned_blocks.defvjp(_scanned_blocks_fwd, _scanned_blocks_bwd)
  return _scanned_blocks(x_mb0, x_mb1, blocks_local, hoisted_absmax, step_in)
