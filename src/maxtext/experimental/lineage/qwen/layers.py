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

"""Pure-JAX Qwen 3.5 / Qwen MAX sub-layers, MoE routing/GMM, GDN, and Full GQA."""

from __future__ import annotations

import dataclasses
import importlib
import math
from typing import Any, NamedTuple

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen import ops
from maxtext.experimental.lineage.qwen import params as params_lib
from maxtext.experimental.lineage.qwen.config import QwenConfig

_PKG = "maxtext.experimental.lineage.qwen"


def _maybe_peer(name: str) -> Any | None:
  """Optionally resolves a sibling kernel module without hard BUILD edges."""
  try:
    return importlib.import_module(f"{_PKG}.{name}")
  except ImportError:
    return None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredMoeWeights:
  """All-gathered MoE & shared-expert weights ready for sub-layer compute."""

  router_gate: Any
  wi: Any
  wo: Any
  shared_wi_0: Any
  shared_wi_1: Any
  shared_wo: Any
  shared_gate: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredGdnWeights:
  """All-gathered Gated DeltaNet + MoE sub-layer weights."""

  pre_attn_norm: jax.Array
  post_attn_norm: jax.Array
  in_proj_qkvz: Any
  in_proj_ba: Any
  conv1d: jax.Array
  A_log: jax.Array  # pylint: disable=invalid-name
  dt_bias: jax.Array
  gdn_norm: jax.Array
  out_proj: Any
  moe: GatheredMoeWeights


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredAttnWeights:
  """All-gathered Global Full Attention + MoE sub-layer weights."""

  pre_attn_norm: jax.Array
  post_attn_norm: jax.Array
  q_proj: Any
  k_proj: Any
  v_proj: Any
  q_norm: jax.Array
  k_norm: jax.Array
  out_proj: Any
  moe: GatheredMoeWeights


class DispatchedMoeTokens(NamedTuple):
  """EP-dispatched activations and routing metadata (`T_ep` leading dim)."""

  tokens_ep: Any
  topk_weights_ep: jax.Array
  topk_ids_ep: jax.Array


def order_dispatched_after(
    disp: DispatchedMoeTokens, deps: Any
) -> DispatchedMoeTokens:
  """Ties a `DispatchedMoeTokens` bundle after scheduling barrier `deps`."""
  return DispatchedMoeTokens(
      tokens_ep=ops.order_after(disp.tokens_ep, deps),
      topk_weights_ep=disp.topk_weights_ep,
      topk_ids_ep=disp.topk_ids_ep,
  )


def _routed_fsdp_gather_pure(
    w_sharded: jax.Array,
    cfg: QwenConfig,
    *,
    sharded_axis: int,
    start_after: Any = None,
) -> jax.Array:
  """Pure-JAX FSDP-only all-gather for routed MoE experts (`ep` stays sharded)."""
  w_in = (
      ops.order_after(w_sharded, start_after)
      if start_after is not None
      else w_sharded
  )
  if cfg.fsdp_size <= 1:
    return w_in

  ag_core = 1 if cfg.fsdp_ag_sc_pinning in ("sc1", "sc1_fwd_only") else None
  rs_core = 1 if cfg.fsdp_rs_sc_pinning == "sc1" else None
  min_b = cfg.sc_min_collective_bytes
  k_shards = cfg.moe_k_shards
  flat_k_sharded = k_shards > 1
  if flat_k_sharded:
    e_full = (w_in.shape[0] * cfg.fsdp_size) // k_shards
    full_k = w_in.shape[1] * k_shards
    full_shape = (e_full, full_k, w_in.shape[2])
  else:
    full_shape = None

  @jax.custom_vjp
  def _gather(w: jax.Array) -> jax.Array:
    w_f = ops.sc_all_gather(
        w,
        "fsdp",
        core_id=ag_core,
        axis=sharded_axis,
        min_collective_bytes=min_b,
    )
    return w_f.reshape(full_shape) if flat_k_sharded else w_f

  def _gather_fwd(w: jax.Array):
    return _gather(w), None

  def _gather_bwd(_, gw_full: jax.Array):
    ops.assert_bf16_or_fp32_reduction(gw_full, "routed_fsdp_gather_bwd")
    gw_in = (
        gw_full.reshape(
            gw_full.shape[0] * k_shards,
            gw_full.shape[1] // k_shards,
            gw_full.shape[2],
        )
        if flat_k_sharded
        else gw_full
    )
    return (
        ops.sc_psum_scatter(
            gw_in,
            "fsdp",
            core_id=rs_core,
            scatter_dimension=sharded_axis,
            min_collective_bytes=min_b,
        ),
    )

  _gather.defvjp(_gather_fwd, _gather_bwd)
  return _gather(w_in)


def gather_moe_weights(
    p: params_lib.MoeParams,
    wi_absmax: Any,
    wo_absmax: Any,
    cfg: QwenConfig,
    *,
    start_after: Any = None,
) -> GatheredMoeWeights:
  """Gathers `MoeParams` from sharded device layout for one sub-layer."""
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  fp8_se = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_shared_expert)
  moe_kernels = _maybe_peer("moe_kernels") if cfg.use_pallas_kernels else None

  if moe_kernels is not None and hasattr(
      moe_kernels, "gather_routed_weight_sc1"
  ):
    wi = moe_kernels.gather_routed_weight_sc1(
        p.wi, wi_absmax, cfg, start_after=start_after, is_wi=True
    )
    wo = moe_kernels.gather_routed_weight_sc1(
        p.wo, wo_absmax, cfg, start_after=start_after, is_wi=False
    )
  else:
    del wi_absmax, wo_absmax
    fsdp_axis = 1 if cfg.moe_k_shards == 0 else 0
    wi = _routed_fsdp_gather_pure(
        p.wi, cfg, sharded_axis=fsdp_axis, start_after=start_after
    )
    wo = _routed_fsdp_gather_pure(
        p.wo, cfg, sharded_axis=fsdp_axis, start_after=start_after
    )

  router_fe = cfg.dense_proj_fsdp_expert(cfg.num_experts)
  shared_fe = cfg.dense_proj_fsdp_expert(cfg.shared_mlp_dim)
  if cfg.replicate_shared_gate:
    shared_gate_w = ops.replicate_small_param(p.shared_gate, cfg)
  else:
    shared_gate_w = ops.gather_dense_weight(
        p.shared_gate,
        cfg,
        sharded_axis=0,
        is_fsdp_expert=is_fe,
        start_after=start_after,
        use_fp8=False,
    )
  return GatheredMoeWeights(
      router_gate=ops.gather_dense_weight(
          p.router_gate,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=router_fe,
          start_after=start_after,
          use_fp8=False,
      ),
      wi=wi,
      wo=wo,
      shared_wi_0=ops.gather_dense_weight(
          p.shared_wi_0,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=shared_fe,
          start_after=start_after,
          use_fp8=fp8_se,
      ),
      shared_wi_1=ops.gather_dense_weight(
          p.shared_wi_1,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=shared_fe,
          start_after=start_after,
          use_fp8=fp8_se,
      ),
      shared_wo=ops.gather_dense_weight(
          p.shared_wo,
          cfg,
          sharded_axis=1,
          is_fsdp_expert=shared_fe,
          start_after=start_after,
          use_fp8=fp8_se,
      ),
      shared_gate=shared_gate_w,
  )


def gather_gdn_sublayer_weights(
    p: params_lib.GdnSublayerParams,
    wi_absmax: Any,
    wo_absmax: Any,
    cfg: QwenConfig,
) -> GatheredGdnWeights:
  """All-gathers GDN + MoE weights and registers replicated norm/conv/dt grads."""
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  fp8_gdn = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_gdn)
  return GatheredGdnWeights(
      pre_attn_norm=ops.replicate_small_param(p.pre_attn_norm, cfg),
      post_attn_norm=ops.replicate_small_param(p.post_attn_norm, cfg),
      in_proj_qkvz=ops.gather_dense_weight(
          p.in_proj_qkvz,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          use_fp8=fp8_gdn,
      ),
      in_proj_ba=ops.gather_dense_weight(
          p.in_proj_ba,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=cfg.dense_proj_fsdp_expert(2 * cfg.gdn_num_v_heads),
          use_fp8=False,
      ),
      conv1d=ops.replicate_small_param(p.conv1d, cfg),
      A_log=ops.replicate_small_param(p.A_log, cfg),
      dt_bias=ops.replicate_small_param(p.dt_bias, cfg),
      gdn_norm=ops.replicate_small_param(p.gdn_norm, cfg),
      out_proj=ops.gather_dense_weight(
          p.out_proj,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          use_fp8=fp8_gdn,
      ),
      moe=gather_moe_weights(p.moe, wi_absmax, wo_absmax, cfg),
  )


def gather_attn_sublayer_weights(
    p: params_lib.AttnSublayerParams,
    wi_absmax: Any,
    wo_absmax: Any,
    cfg: QwenConfig,
) -> GatheredAttnWeights:
  """All-gathers Full-Attention + MoE weights and registers replicated norm grads."""
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  fp8_attn = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_attn)
  return GatheredAttnWeights(
      pre_attn_norm=ops.replicate_small_param(p.pre_attn_norm, cfg),
      post_attn_norm=ops.replicate_small_param(p.post_attn_norm, cfg),
      q_proj=ops.gather_dense_weight(
          p.q_proj,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          use_fp8=fp8_attn,
      ),
      k_proj=ops.gather_dense_weight(
          p.k_proj,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=cfg.dense_proj_fsdp_expert(
              cfg.num_kv_heads * cfg.head_dim
          ),
          use_fp8=fp8_attn,
      ),
      v_proj=ops.gather_dense_weight(
          p.v_proj,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=cfg.dense_proj_fsdp_expert(
              cfg.num_kv_heads * cfg.head_dim
          ),
          use_fp8=fp8_attn,
      ),
      q_norm=ops.replicate_small_param(p.q_norm, cfg),
      k_norm=ops.replicate_small_param(p.k_norm, cfg),
      out_proj=ops.gather_dense_weight(
          p.out_proj,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          use_fp8=fp8_attn,
      ),
      moe=gather_moe_weights(p.moe, wi_absmax, wo_absmax, cfg),
  )


# ---------------------------------------------------------------------------
# Partial RoPE & Global Full GQA Attention (with sigmoid output gate)
# ---------------------------------------------------------------------------


def compute_rope_cos_sin(
    positions: jax.Array,
    rotary_dim: int,
    max_timescale: float = 10_000_000.0,
) -> tuple[jax.Array, jax.Array]:
  """Precomputes half-split partial RoPE `(cos, sin)` `[B, S, 1, rotary_dim // 2]`."""
  attn_mod = _maybe_peer("attention")
  if attn_mod is not None and hasattr(attn_mod, "compute_rope_cos_sin"):
    return attn_mod.compute_rope_cos_sin(positions, rotary_dim, max_timescale)

  half_dim = rotary_dim // 2
  fraction = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / float(rotary_dim)
  inv_freq = 1.0 / (float(max_timescale) ** fraction)
  angles = positions.astype(jnp.float32)[..., None] * inv_freq.reshape(
      1, 1, half_dim
  )
  cos = jnp.cos(angles)[..., None, :]
  sin = jnp.sin(angles)[..., None, :]
  return cos, sin


def apply_partial_rope(
    x: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
    max_timescale: float = 10_000_000.0,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Applies NeoX half-split partial RoPE to leading `rotary_dim` channels."""
  attn_mod = _maybe_peer("attention")
  if attn_mod is not None and hasattr(attn_mod, "apply_partial_rope"):
    return attn_mod.apply_partial_rope(
        x,
        positions,
        rotary_dim,
        max_timescale,
        rope_cos_sin=rope_cos_sin,
    )

  if rotary_dim == 0:
    return x
  x_rot = x[..., :rotary_dim]
  x_pass = x[..., rotary_dim:]
  cos, sin = (
      rope_cos_sin
      if rope_cos_sin is not None
      else compute_rope_cos_sin(positions, rotary_dim, max_timescale)
  )
  x1, x2 = jnp.split(x_rot.astype(jnp.float32), 2, axis=-1)
  out_rot = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return jnp.concatenate([out_rot.astype(x.dtype), x_pass], axis=-1)


def full_attn_mixer(
    x: jax.Array,
    w: GatheredAttnWeights | Any,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Global Full GQA sub-layer (`q_proj [Q|gate]`, QK-norm, partial RoPE, residual)."""
  attn_mod = _maybe_peer("attention") if cfg.use_pallas_kernels else None
  if attn_mod is not None and hasattr(attn_mod, "full_attn_mixer"):
    return attn_mod.full_attn_mixer(
        x, w, positions, segment_ids, cfg, rope_cos_sin=rope_cos_sin
    )

  normed = ops.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)
  bsz, seq_len, _ = normed.shape
  h_q = cfg.num_query_heads
  h_kv = cfg.num_kv_heads
  d_h = cfg.head_dim

  if cfg.attn_qkv_2d_matmul and not isinstance(w.q_proj, ops.Fp8DenseWeight):
    q_w2d = w.q_proj.reshape(cfg.emb_dim, h_q * 2 * d_h)
    k_w2d = w.k_proj.reshape(cfg.emb_dim, h_kv * d_h)
    v_w2d = w.v_proj.reshape(cfg.emb_dim, h_kv * d_h)
    q_raw = ops.dense_fp8_attn(normed, q_w2d, is_4d_head_proj=False).reshape(
        bsz, seq_len, h_q, 2 * d_h
    )
    k = ops.dense_fp8_attn(normed, k_w2d, is_4d_head_proj=False).reshape(
        bsz, seq_len, h_kv, d_h
    )
    v = ops.dense_fp8_attn(normed, v_w2d, is_4d_head_proj=False).reshape(
        bsz, seq_len, h_kv, d_h
    )
  else:
    q_raw = ops.dense_fp8_attn(normed, w.q_proj, is_4d_head_proj=True)
    k = ops.dense_fp8_attn(normed, w.k_proj, is_4d_head_proj=True)
    v = ops.dense_fp8_attn(normed, w.v_proj, is_4d_head_proj=True)

  q, gate = jnp.split(q_raw, 2, axis=-1)
  q = ops.rms_norm_zero_centered(q, w.q_norm, cfg.norm_eps)
  k = ops.rms_norm_zero_centered(k, w.k_norm, cfg.norm_eps)

  rot_dim = cfg.rotary_dim
  q = apply_partial_rope(
      q, positions, rot_dim, cfg.rope_max_timescale, rope_cos_sin=rope_cos_sin
  )
  k = apply_partial_rope(
      k, positions, rot_dim, cfg.rope_max_timescale, rope_cos_sin=rope_cos_sin
  )

  # Standard GQA attention matching MaxText Attention (float32, d_h**-0.5).
  q_scaled = q * (float(d_h) ** -0.5)
  ctx = _pure_jax_gqa_core(q_scaled, k, v, segment_ids, cfg).astype(normed.dtype)

  gated = (
      ctx.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
  ).astype(normed.dtype)
  ctx_flat = gated.reshape(bsz, seq_len, h_q * d_h)
  proj = ops.dense_fp8_attn(ctx_flat, w.out_proj, is_4d_head_proj=False)
  return x + proj


def _pure_jax_gqa_core(
    q_scaled: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Pure-JAX GQA attention core matching `layers.full_attn_mixer` fallback."""
  cp_axis = cfg.cp_axis_name
  if cp_axis is not None:
    q_scaled = jax.lax.all_to_all(
        q_scaled, cp_axis, split_axis=2, concat_axis=1, tiled=True
    )
    k = jax.lax.all_to_all(k, cp_axis, split_axis=2, concat_axis=1, tiled=True)
    v = jax.lax.all_to_all(v, cp_axis, split_axis=2, concat_axis=1, tiled=True)
    segment_ids = jax.lax.all_gather(segment_ids, cp_axis, axis=1, tiled=True)

  bsz, seq_len, h_q, d_h = q_scaled.shape
  h_kv = k.shape[2]
  gqa_groups = h_q // h_kv
  q_g = q_scaled.reshape(bsz, seq_len, h_kv, gqa_groups, d_h)
  logits = jnp.einsum(
      "bshgd,bkhd->bhgsk",
      q_g.astype(jnp.float32),
      k.astype(jnp.float32),
  )
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  seg_q = segment_ids[:, None, :, None] > 0
  seg_eq = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
  mask = causal[None, None, :, :] & seg_q & seg_eq
  mask_5d = mask[:, None, :, :, :]
  logits = jnp.where(mask_5d, logits, -1e9)
  probs = jax.nn.softmax(logits, axis=-1)
  probs = jnp.where(mask_5d, probs, 0.0)
  ctx = jnp.einsum("bhgsk,bkhd->bshgd", probs, v.astype(jnp.float32)).reshape(
      bsz, seq_len, h_q, d_h
  )
  ctx = ctx.astype(q_scaled.dtype)
  if cp_axis is not None:
    ctx = jax.lax.all_to_all(
        ctx, cp_axis, split_axis=1, concat_axis=2, tiled=True
    )
  return ctx


def full_attn_mixer_fwd_save_res(
    x: jax.Array,
    w: GatheredAttnWeights | Any,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, tuple[Any, jax.Array]]:
  """Forward `full_attn_mixer` saving Splash/attention residuals with zero weight capture."""
  attn_mod = _maybe_peer("attention") if cfg.use_pallas_kernels else None
  if attn_mod is not None and hasattr(attn_mod, "full_attn_mixer_fwd_save_res"):
    return attn_mod.full_attn_mixer_fwd_save_res(
        x, w, positions, segment_ids, cfg, rope_cos_sin=rope_cos_sin
    )

  normed = ops.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)
  bsz, seq_len, _ = normed.shape
  h_q = cfg.num_query_heads
  h_kv = cfg.num_kv_heads
  d_h = cfg.head_dim
  if cfg.attn_qkv_2d_matmul and not isinstance(w.q_proj, ops.Fp8DenseWeight):
    q_raw = ops.dense_fp8_attn(
        normed, w.q_proj.reshape(cfg.emb_dim, h_q * 2 * d_h), is_4d_head_proj=False
    ).reshape(bsz, seq_len, h_q, 2 * d_h)
    k = ops.dense_fp8_attn(
        normed, w.k_proj.reshape(cfg.emb_dim, h_kv * d_h), is_4d_head_proj=False
    ).reshape(bsz, seq_len, h_kv, d_h)
    v = ops.dense_fp8_attn(
        normed, w.v_proj.reshape(cfg.emb_dim, h_kv * d_h), is_4d_head_proj=False
    ).reshape(bsz, seq_len, h_kv, d_h)
  else:
    q_raw = ops.dense_fp8_attn(normed, w.q_proj, is_4d_head_proj=True)
    k = ops.dense_fp8_attn(normed, w.k_proj, is_4d_head_proj=True)
    v = ops.dense_fp8_attn(normed, w.v_proj, is_4d_head_proj=True)
  q, gate = jnp.split(q_raw, 2, axis=-1)
  q = ops.rms_norm_zero_centered(q, w.q_norm, cfg.norm_eps)
  k = ops.rms_norm_zero_centered(k, w.k_norm, cfg.norm_eps)
  rot_dim = cfg.rotary_dim
  q = apply_partial_rope(
      q, positions, rot_dim, cfg.rope_max_timescale, rope_cos_sin=rope_cos_sin
  )
  k = apply_partial_rope(
      k, positions, rot_dim, cfg.rope_max_timescale, rope_cos_sin=rope_cos_sin
  )
  q_scaled = q * (float(d_h) ** -0.5)
  ctx, vjp_core = jax.vjp(
      lambda q_in, k_in, v_in: _pure_jax_gqa_core(
          q_in, k_in, v_in, segment_ids, cfg
      ),
      q_scaled,
      k,
      v,
  )
  gated = (
      ctx.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
  ).astype(normed.dtype)
  proj = ops.dense_fp8_attn(
      gated.reshape(bsz, seq_len, h_q * d_h), w.out_proj, is_4d_head_proj=False
  )
  return x + proj, (vjp_core, ctx)


def full_attn_mixer_bwd_from_res(
    g_y: jax.Array,
    x: jax.Array,
    saved_res: tuple[Any, jax.Array],
    w: GatheredAttnWeights | Any,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, Any]:
  """Backward `full_attn_mixer` reusing saved Splash/attention residuals."""
  attn_mod = _maybe_peer("attention") if cfg.use_pallas_kernels else None
  if attn_mod is not None and hasattr(attn_mod, "full_attn_mixer_bwd_from_res"):
    return attn_mod.full_attn_mixer_bwd_from_res(
        g_y, x, saved_res, w, positions, segment_ids, cfg, rope_cos_sin=rope_cos_sin
    )

  del segment_ids
  vjp_core, ctx = saved_res
  bsz, seq_len, _ = x.shape
  h_q = cfg.num_query_heads
  h_kv = cfg.num_kv_heads
  d_h = cfg.head_dim

  def _pre(x_in, pre_norm_w, q_proj_w, k_proj_w, v_proj_w, q_norm_w, k_norm_w):
    normed = ops.rms_norm_zero_centered(x_in, pre_norm_w, cfg.norm_eps)
    if cfg.attn_qkv_2d_matmul and not isinstance(q_proj_w, ops.Fp8DenseWeight):
      q_raw = ops.dense_fp8_attn(
          normed, q_proj_w.reshape(cfg.emb_dim, h_q * 2 * d_h), is_4d_head_proj=False
      ).reshape(bsz, seq_len, h_q, 2 * d_h)
      k = ops.dense_fp8_attn(
          normed, k_proj_w.reshape(cfg.emb_dim, h_kv * d_h), is_4d_head_proj=False
      ).reshape(bsz, seq_len, h_kv, d_h)
      v = ops.dense_fp8_attn(
          normed, v_proj_w.reshape(cfg.emb_dim, h_kv * d_h), is_4d_head_proj=False
      ).reshape(bsz, seq_len, h_kv, d_h)
    else:
      q_raw = ops.dense_fp8_attn(normed, q_proj_w, is_4d_head_proj=True)
      k = ops.dense_fp8_attn(normed, k_proj_w, is_4d_head_proj=True)
      v = ops.dense_fp8_attn(normed, v_proj_w, is_4d_head_proj=True)
    q, gate = jnp.split(q_raw, 2, axis=-1)
    q = ops.rms_norm_zero_centered(q, q_norm_w, cfg.norm_eps)
    k = ops.rms_norm_zero_centered(k, k_norm_w, cfg.norm_eps)
    rot_dim = cfg.rotary_dim
    q = apply_partial_rope(
        q, positions, rot_dim, cfg.rope_max_timescale, rope_cos_sin=rope_cos_sin
    )
    k = apply_partial_rope(
        k, positions, rot_dim, cfg.rope_max_timescale, rope_cos_sin=rope_cos_sin
    )
    q_scaled = q * (float(d_h) ** -0.5)
    return q_scaled, k, v, gate

  def _post(ctx_in, gate_in, out_proj_w):
    gated = (
        ctx_in.astype(jnp.float32) * jax.nn.sigmoid(gate_in.astype(jnp.float32))
    ).astype(x.dtype)
    return ops.dense_fp8_attn(
        gated.reshape(bsz, seq_len, h_q * d_h), out_proj_w, is_4d_head_proj=False
    )

  (_, _, _, gate), vjp_pre = jax.vjp(
      _pre,
      x,
      w.pre_attn_norm,
      w.q_proj,
      w.k_proj,
      w.v_proj,
      w.q_norm,
      w.k_norm,
  )
  _, vjp_post = jax.vjp(_post, ctx, gate, w.out_proj)
  g_ctx, g_gate, g_out_proj = vjp_post(g_y)
  g_q_scaled, g_k, g_v = vjp_core(g_ctx)
  (
      g_x_pre,
      g_pre_norm,
      g_q_proj,
      g_k_proj,
      g_v_proj,
      g_q_norm,
      g_k_norm,
  ) = vjp_pre((g_q_scaled, g_k, g_v, g_gate))
  return (
      g_y + g_x_pre,
      params_lib.AttnSublayerParams(
          pre_attn_norm=g_pre_norm,
          post_attn_norm=None,
          q_proj=g_q_proj,
          k_proj=g_k_proj,
          v_proj=g_v_proj,
          q_norm=g_q_norm,
          k_norm=g_k_norm,
          out_proj=g_out_proj,
          moe=None,
      ),
  )


# ---------------------------------------------------------------------------
# Gated DeltaNet Linear Attention Sub-layer (pure-JAX + optional Pallas peer)
# ---------------------------------------------------------------------------


def _l2_norm_last_dim(x: jax.Array, eps: float = 1e-6) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  inv_norm = jax.lax.rsqrt(
      jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
      + jnp.asarray(eps, jnp.float32)
  )
  return (x_f32 * inv_norm).astype(x.dtype)


def _gdn_pure_jax_recurrence(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    segment_ids: jax.Array,
) -> jax.Array:
  """Exact pure-JAX linear-attention gated-delta-rule recurrence `[B, S, H_v, D_v]`."""
  bsz, seq_len, h_v, d_k = q.shape
  d_v = v.shape[-1]

  # Normalize inputs to [S, B, H_v, ...], float32 for exact FP32 accumulation.
  q_s = jnp.swapaxes(q.astype(jnp.float32), 0, 1) * (float(d_k) ** -0.5)
  k_s = jnp.swapaxes(k.astype(jnp.float32), 0, 1)
  v_s = jnp.swapaxes(v.astype(jnp.float32), 0, 1)
  g_s = jnp.swapaxes(g.astype(jnp.float32), 0, 1)
  beta_s = jnp.swapaxes(beta.astype(jnp.float32), 0, 1)

  # Reset state at padding tokens or segment boundaries.
  seg_s = jnp.swapaxes(segment_ids, 0, 1)  # [S, B]
  prev_seg = jnp.concatenate([seg_s[:1], seg_s[:-1]], axis=0)
  same_seg = (seg_s > 0) & (
      (jnp.arange(seq_len)[:, None] == 0) | (seg_s == prev_seg)
  )
  active_s = (seg_s > 0).astype(jnp.float32)[:, :, None, None]

  init_state = jnp.zeros((bsz, h_v, d_k, d_v), dtype=jnp.float32)

  def _step(state, xs):
    q_t, k_t, v_t, g_t, beta_t, keep_t, act_t = xs
    keep_4d = keep_t.astype(jnp.float32)[:, None, None, None]
    state = state * keep_4d
    decay = jnp.exp(g_t)[:, :, None, None]
    state_decayed = state * decay
    # KV memory readout: [B, H_v, D_v]
    kv_mem = jnp.einsum("bhkd,bhk->bhd", state_decayed, k_t)
    delta = (v_t - kv_mem) * beta_t[:, :, None]
    new_state = state_decayed + k_t[:, :, :, None] * delta[:, :, None, :]
    out_t = jnp.einsum("bhkd,bhk->bhd", new_state, q_t) * act_t
    return new_state, out_t

  _, out_s = jax.lax.scan(
      _step, init_state, (q_s, k_s, v_s, g_s, beta_s, same_seg, active_s)
  )
  return jnp.swapaxes(out_s, 0, 1)


def _gdn_mixer_fallback(
    x: jax.Array,
    w: GatheredGdnWeights | Any,
    segment_ids: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Pure-JAX GDN mixer matching `Qwen3_5GatedDeltaNet`."""
  normed = ops.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)
  bsz, seq_len, _ = normed.shape

  qkvz = ops.dense_fp8_gdn(normed, w.in_proj_qkvz)
  ba = ops.dense_fp8_gdn(normed, w.in_proj_ba)

  qkv, z = jnp.split(qkvz, [cfg.gdn_qkv_dim], axis=-1)
  b, a = jnp.split(ba, [cfg.gdn_num_v_heads], axis=-1)

  # Causal depthwise Conv1D masking padded tokens (`segment_ids != 0`).
  qkv_masked = jnp.where((segment_ids != 0)[:, :, None], qkv, 0)
  k_conv = cfg.gdn_conv_kernel_size
  conv_w = w.conv1d.reshape(k_conv, 1, cfg.gdn_qkv_dim)
  qkv_conv = jax.lax.conv_general_dilated(
      qkv_masked.astype(jnp.float32),
      conv_w.astype(jnp.float32),
      window_strides=(1,),
      padding=((k_conv - 1, 0),),
      dimension_numbers=("NWC", "WIO", "NWC"),
      feature_group_count=cfg.gdn_qkv_dim,
  )
  qkv_conv = jax.nn.silu(qkv_conv).astype(normed.dtype)

  q_flat, k_flat, v_flat = jnp.split(
      qkv_conv, [cfg.gdn_key_dim, 2 * cfg.gdn_key_dim], axis=-1
  )
  h_k = cfg.gdn_num_k_heads
  h_v = cfg.gdn_num_v_heads
  d_k = cfg.gdn_head_k_dim
  d_v = cfg.gdn_head_v_dim
  q = _l2_norm_last_dim(
      q_flat.reshape(bsz, seq_len, h_k, d_k), eps=cfg.norm_eps
  )
  k = _l2_norm_last_dim(
      k_flat.reshape(bsz, seq_len, h_k, d_k), eps=cfg.norm_eps
  )
  v = v_flat.reshape(bsz, seq_len, h_v, d_v)

  if h_v > h_k:
    rep = h_v // h_k
    q = jnp.repeat(q, rep, axis=2)
    k = jnp.repeat(k, rep, axis=2)

  beta = jax.nn.sigmoid(b.astype(jnp.float32))
  g = -jnp.exp(w.A_log.astype(jnp.float32)) * jax.nn.softplus(
      a.astype(jnp.float32) + w.dt_bias.astype(jnp.float32)
  )

  # Dispatch Pallas recurrence cleanly if available, else exact scan.
  out_f32 = _gdn_pure_jax_recurrence(q, k, v, g, beta, segment_ids)
  z_4d = z.reshape(bsz, seq_len, h_v, d_v)
  normed_out = ops.rms_norm_gated(out_f32, z_4d, w.gdn_norm, eps=cfg.norm_eps)
  normed_flat = normed_out.reshape(bsz, seq_len, cfg.gdn_value_dim).astype(
      normed.dtype
  )
  proj = ops.dense_fp8_gdn(normed_flat, w.out_proj)
  return x + proj


def gdn_mixer(
    x: jax.Array,
    w: GatheredGdnWeights | Any,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    precomputed_meta: Any = None,
) -> jax.Array:
  """Gated DeltaNet sub-layer (`gdn_mixer` peer delegation with pure-JAX fallback)."""
  gdn_mod = _maybe_peer("gdn")
  if gdn_mod is not None and hasattr(gdn_mod, "gdn_mixer"):
    if cfg.use_pallas_kernels or hasattr(gdn_mod, "_pure_jax_gdn_cpu_fallback"):
      return gdn_mod.gdn_mixer(
          x, w, segment_ids, cfg, precomputed_meta=precomputed_meta
      )
  return _gdn_mixer_fallback(x, w, segment_ids, cfg)


def gdn_mixer_fwd_save_res(
    x: jax.Array,
    w: GatheredGdnWeights | Any,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    precomputed_meta: Any = None,
) -> tuple[jax.Array, tuple[Any, jax.Array, jax.Array]]:
  """Forward `gdn_mixer` saving GDN core residuals with zero big-weight capture."""
  gdn_mod = _maybe_peer("gdn")
  if gdn_mod is not None and hasattr(gdn_mod, "gdn_mixer_fwd_save_res"):
    return gdn_mod.gdn_mixer_fwd_save_res(
        x, w, segment_ids, cfg, precomputed_meta=precomputed_meta
    )
  raise RuntimeError(
      "gdn_mixer_fwd_save_res requires the sibling `models.qwen.gdn` module."
  )


def gdn_mixer_bwd_from_res(
    g_y: jax.Array,
    x: jax.Array,
    saved_res: tuple[Any, jax.Array, jax.Array],
    w: GatheredGdnWeights | Any,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    precomputed_meta: Any = None,
) -> tuple[jax.Array, Any]:
  """Backward `gdn_mixer` reusing saved GDN core residuals (zero GDN/proj FWD remat)."""
  gdn_mod = _maybe_peer("gdn")
  if gdn_mod is not None and hasattr(gdn_mod, "gdn_mixer_bwd_from_res"):
    return gdn_mod.gdn_mixer_bwd_from_res(
        g_y, x, saved_res, w, segment_ids, cfg, precomputed_meta=precomputed_meta
    )
  raise RuntimeError(
      "gdn_mixer_bwd_from_res requires the sibling `models.qwen.gdn` module."
  )


def precompute_gdn_meta(segment_ids: jax.Array | None, cfg: QwenConfig) -> Any:
  """Precomputes loop-invariant GDN segment and Pallas metadata outside scan."""
  if not cfg.use_pallas_kernels or segment_ids is None:
    return None
  gdn_mod = _maybe_peer("gdn")
  if gdn_mod is None or not hasattr(gdn_mod, "precompute_gdn_meta"):
    return None
  return gdn_mod.precompute_gdn_meta(
      segment_ids,
      chunk_size=cfg.gdn_chunk_size,
      conv_kernel_size=cfg.gdn_conv_kernel_size,
  )


# ---------------------------------------------------------------------------
# Shared Expert & Sparse MoE Blocks
# ---------------------------------------------------------------------------


def shared_expert_and_residual(
    x_post_mixer: jax.Array,
    normed_post_mixer: jax.Array,
    w_moe: GatheredMoeWeights | Any,
    *,
    float32_gate: bool = True,
) -> jax.Array:
  """Computes gated shared-expert FFN + `x_post_mixer` residual (BF16 or MXU FP8)."""
  shared_mlp = ops.dense_fp8_shared_expert(
      normed_post_mixer,
      w_moe.shared_wi_0,
      w_moe.shared_wi_1,
      w_moe.shared_wo,
  )
  if float32_gate and not isinstance(w_moe.shared_gate, ops.Fp8DenseWeight):
    gate_raw = jnp.matmul(
        normed_post_mixer.astype(jnp.float32),
        w_moe.shared_gate.astype(jnp.float32),
    )
  else:
    gate_raw = ops.dense_fp8_matmul(normed_post_mixer, w_moe.shared_gate)
  gate = jax.nn.sigmoid(gate_raw.astype(jnp.float32))
  return x_post_mixer + (shared_mlp.astype(jnp.float32) * gate).astype(
      x_post_mixer.dtype
  )


def moe_local_route_only(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: QwenConfig,
    *,
    route_after: Any = None,
    sublayer_idx: int = 0,
    batch_offset: int | None = None,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array]:
  """Computes Top-K router weights and IDs locally (preserving `[*prefix, K]`)."""
  x_in = (
      ops.order_after(normed_tokens, route_after)
      if route_after is not None
      else normed_tokens
  )
  orig_shape = x_in.shape[:-1]
  x_2d = x_in.reshape(-1, cfg.emb_dim)

  if cfg.float32_gate_logits:
    logits = jnp.matmul(
        x_2d.astype(jnp.float32), router_gate.astype(jnp.float32)
    )
  else:
    logits = jnp.matmul(x_2d, router_gate).astype(jnp.float32)

  if cfg.routing_mode == "router":
    _, topk_ids = jax.lax.top_k(
        jax.lax.stop_gradient(logits), cfg.num_experts_per_tok
    )
    topk_mask = jax.nn.one_hot(topk_ids, cfg.num_experts, dtype=logits.dtype)
    topk_logits = jnp.einsum("te,tke->tk", logits, topk_mask)
    topk_weights = jax.nn.softmax(
        topk_logits.astype(jnp.float32), axis=-1
    ).astype(x_in.dtype)
    if cfg.norm_topk_prob:
      denom = jnp.sum(topk_weights.astype(jnp.float32), axis=-1, keepdims=True)
      denom = jnp.where(denom > 0.0, denom, jnp.ones_like(denom))
      topk_weights = (topk_weights.astype(jnp.float32) / denom).astype(
          x_in.dtype
      )
  elif cfg.routing_mode == "random":
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    eff_layer_idx = sublayer_idx if layer_idx is None else layer_idx
    topk_ids = ops.sample_random_topk_ids(
        orig_shape,
        cfg,
        layer_idx=eff_layer_idx,
        step=step,
        batch_offset=0 if batch_offset is None else batch_offset,
        router_logits_f32=logits,
    )
    topk_weights_f32 = ops.dense_select_topk_probs(
        probs, topk_ids, cfg.num_experts
    ).astype(jnp.float32)
    if cfg.norm_topk_prob:
      denom = jnp.sum(topk_weights_f32, axis=-1, keepdims=True)
      denom = jnp.where(denom > 0.0, denom, jnp.ones_like(denom))
      topk_weights_f32 = topk_weights_f32 / denom
    topk_weights = topk_weights_f32.astype(x_in.dtype)
  else:
    raise ValueError(f"Unsupported routing_mode={cfg.routing_mode!r}")

  return (
      checkpoint_name(
          topk_weights.reshape(*orig_shape, cfg.num_experts_per_tok),
          "moe_routing",
      ),
      checkpoint_name(
          topk_ids.reshape(*orig_shape, cfg.num_experts_per_tok),
          "moe_routing",
      ),
  )


def moe_dispatch_from_routed(
    normed_tokens: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    cfg: QwenConfig,
    *,
    order_tokens_on_router: bool = True,
) -> DispatchedMoeTokens:
  """EP-dispatches flattened local tokens and routing metadata to `[T_ep, ...]`."""
  tok_in = (
      ops.order_after(normed_tokens, topk_weights)
      if order_tokens_on_router
      else normed_tokens
  )
  use_3d = (
      cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (cfg.emb_dim % 128 == 0)
  )
  if use_3d:
    tok_flat = tok_in.reshape(-1, cfg.emb_dim // 128, 128)
  else:
    tok_flat = tok_in.reshape(-1, cfg.emb_dim)

  k_top = cfg.num_experts_per_tok
  w_flat = topk_weights.reshape(-1, k_top)
  ids_flat = topk_ids.reshape(-1, k_top)

  if cfg.ep_size > 1:
    if use_3d and cfg.moe_fp8_token_dispatch:
      tok_ep = ops.ep_token_dispatch_3d_fp8(tok_flat, cfg)
    else:
      tok_ep = ops.ep_token_dispatch(tok_flat, cfg)
    w_ep = ops.ep_routing_weights_dispatch(w_flat, cfg)
    ids_ep = ops.ep_routing_ids_dispatch(ids_flat, cfg)
  else:
    tok_ep = tok_flat
    w_ep = w_flat
    ids_ep = jax.lax.stop_gradient(ids_flat.astype(jnp.int32))

  return DispatchedMoeTokens(
      tokens_ep=tok_ep,
      topk_weights_ep=w_ep,
      topk_ids_ep=ids_ep,
  )


def moe_local_route_and_dispatch(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: QwenConfig,
    *,
    route_after: Any = None,
    sublayer_idx: int = 0,
    batch_offset: int | None = None,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> DispatchedMoeTokens:
  """Routes locally then EP-dispatches tokens, weights, and IDs."""
  topk_weights, topk_ids = moe_local_route_only(
      normed_tokens,
      router_gate,
      cfg,
      route_after=route_after,
      sublayer_idx=sublayer_idx,
      batch_offset=batch_offset,
      layer_idx=layer_idx,
      step=step,
  )
  return moe_dispatch_from_routed(
      normed_tokens,
      topk_weights,
      topk_ids,
      cfg,
      order_tokens_on_router=True,
  )


def _format_moe_expert_return(
    unperm_ep: Any,
    wi_out: Any,
    wo_out: Any,
    forward_tie_out: Any,
    rw3_slot: Any,
    *,
    return_weights: bool,
    has_forward_tie: bool,
    use_external_bwd_rw: bool,
) -> Any:
  """Formats tuple returns identically to `schedule.py` batchsplit expectations."""
  if return_weights:
    if use_external_bwd_rw:
      return unperm_ep, wi_out, wo_out, rw3_slot
    return unperm_ep, wi_out, wo_out
  if has_forward_tie:
    if use_external_bwd_rw:
      return unperm_ep, forward_tie_out, rw3_slot
    return unperm_ep, forward_tie_out
  if use_external_bwd_rw:
    return unperm_ep, rw3_slot
  return unperm_ep


def _moe_expert_pure_jax_fallback(
    dispatched: DispatchedMoeTokens,
    wi: jax.Array,
    wo: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Exact pure-JAX expert SwiGLU MLP + weight multiply preserving EP sharding."""
  tok_raw = dispatched.tokens_ep
  if isinstance(tok_raw, ops.Fp8DispatchedTokens):
    # Reconstruct differentiable bf16/fp32 tokens via STE carrier for fallback.
    tok_f32 = (
        tok_raw.codes_ep.astype(jnp.float32)
        * tok_raw.scale.reshape(1, *([1] * (tok_raw.codes_ep.ndim - 1)))
    ).astype(tok_raw.ste_carrier.dtype) + tok_raw.ste_carrier
  else:
    tok_f32 = tok_raw

  orig_tok_shape = tok_f32.shape
  t_ep = orig_tok_shape[0]
  d_dim = cfg.emb_dim
  m_dim = cfg.moe_mlp_dim
  tok_2d = tok_f32.reshape(t_ep, d_dim)

  ep_idx = (
      ops.ep_axis_index(cfg) if cfg.ep_size > 1 else jnp.asarray(0, jnp.int32)
  )
  e_local = cfg.local_num_experts
  start_e = ep_idx * e_local

  w_ep = dispatched.topk_weights_ep.astype(jnp.float32)  # [T_ep, K]
  ids_ep = dispatched.topk_ids_ep  # [T_ep, K]
  wi_e = wi.reshape(e_local, d_dim, 2 * m_dim)
  wo_e = wo.reshape(e_local, m_dim, d_dim)

  def _expert_step(acc: jax.Array, expert_pair: tuple[Any, ...]):
    local_e, wi_slice, wo_slice = expert_pair
    global_e = start_e + local_e
    match_k = ids_ep == global_e  # [T_ep, K]
    coeff = jnp.sum(jnp.where(match_k, w_ep, 0.0), axis=-1)  # [T_ep]
    active = jnp.any(match_k, axis=-1).astype(tok_2d.dtype)[:, None]
    x_e = tok_2d * active
    gate_up = jnp.matmul(x_e, wi_slice)
    g_part, u_part = jnp.split(gate_up, 2, axis=-1)
    hidden = (
        jax.nn.silu(g_part.astype(jnp.float32)) * u_part.astype(jnp.float32)
    ).astype(tok_2d.dtype)
    y_e = jnp.matmul(hidden, wo_slice).astype(jnp.float32)
    return acc + y_e * coeff[:, None], None

  acc_init = jnp.zeros((t_ep, d_dim), dtype=jnp.float32)
  out_2d_f32, _ = jax.lax.scan(
      _expert_step,
      acc_init,
      (jnp.arange(e_local, dtype=jnp.int32), wi_e, wo_e),
  )
  return out_2d_f32.astype(tok_2d.dtype).reshape(orig_tok_shape)


def _build_pure_jax_saved_routing(
    dispatched: DispatchedMoeTokens, cfg: QwenConfig
) -> ops.MoeSavedRouting:
  """Pure-JAX CPU fallback building `MoeSavedRouting` for unit-test parity."""
  t_ep = dispatched.tokens_ep.shape[0]
  ubatch_tokens_local = max(1, t_ep // max(1, cfg.ep_size))
  buf_size = cfg.ragged_buffer_size(ubatch_tokens_local)
  ep_idx = (
      ops.ep_axis_index(cfg) if cfg.ep_size > 1 else jnp.asarray(0, jnp.int32)
  )
  e_local = cfg.local_num_experts
  topk_ids_flat = dispatched.topk_ids_ep.reshape(-1).astype(jnp.int32)
  local_expert = topk_ids_flat - ep_idx * e_local
  sort_key = jnp.where(
      (local_expert >= 0) & (local_expert < e_local), local_expert, e_local
  )
  sort_indices = jnp.argsort(sort_key, stable=True).astype(jnp.int32)
  sorted_keys = jnp.take(sort_key, sort_indices, axis=0)
  slot_order = sort_indices[:buf_size]
  sk_win = sorted_keys[:buf_size]
  ends = jnp.searchsorted(
      sk_win, jnp.arange(1, e_local + 1, dtype=jnp.int32), side="left"
  ).astype(jnp.int32)
  starts = jnp.concatenate(
      [jnp.zeros((1,), dtype=jnp.int32), ends[:-1]], axis=0
  )
  local_gs = ends - starts
  count = ends[-1]
  valid = jnp.arange(buf_size, dtype=jnp.int32) < count
  w_flat = jnp.ravel(dispatched.topk_weights_ep)
  w_rows = jnp.where(
      valid,
      jnp.take(w_flat, jnp.where(valid, slot_order, 0), axis=0),
      jnp.zeros((), dtype=w_flat.dtype),
  )
  return ops.MoeSavedRouting(
      slot_order=slot_order,
      local_gs=local_gs,
      count=count,
      reduce_meta=None,
      w_rows=w_rows,
  )


def _moe_expert_from_routing_pure_jax_fallback(
    tokens_ep: jax.Array | ops.Fp8DispatchedTokens,
    w_rows_f32: jax.Array,
    saved_routing: ops.MoeSavedRouting,
    wi: jax.Array,
    wo: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Pure-JAX CPU fallback executing routed MoE from `saved_routing`."""
  if isinstance(tokens_ep, ops.Fp8DispatchedTokens):
    tok_f = (
        tokens_ep.codes_ep.astype(jnp.float32)
        * tokens_ep.scale.reshape(1, *([1] * (tokens_ep.codes_ep.ndim - 1)))
    ).astype(tokens_ep.ste_carrier.dtype) + tokens_ep.ste_carrier
  else:
    tok_f = tokens_ep
  orig_shape = tok_f.shape
  t_ep = orig_shape[0]
  d_dim = cfg.emb_dim
  m_dim = cfg.moe_mlp_dim
  e_local = cfg.local_num_experts
  topk = cfg.num_experts_per_tok
  buf_size = saved_routing.slot_order.shape[0]
  tok_2d = tok_f.reshape(t_ep, d_dim)

  valid = jnp.arange(buf_size, dtype=jnp.int32) < saved_routing.count
  tok_idx = jax.lax.stop_gradient(
      jnp.where(valid, saved_routing.slot_order // topk, 0)
  )
  sorted_x = jnp.where(valid[:, None], jnp.take(tok_2d, tok_idx, axis=0), 0)

  ends = jnp.cumsum(saved_routing.local_gs)
  starts = ends - saved_routing.local_gs
  row_idx = jnp.arange(buf_size, dtype=jnp.int32)
  wi_e = wi.reshape(e_local, d_dim, 2 * m_dim)
  wo_e = wo.reshape(e_local, m_dim, d_dim)

  def _expert_step(acc, xs):
    s_e, e_e, wi_s, wo_s = xs
    in_e = (row_idx >= s_e) & (row_idx < e_e)
    x_e = jnp.where(in_e[:, None], sorted_x, 0)
    gu = jnp.matmul(x_e, wi_s)
    g_p, u_p = jnp.split(gu, 2, axis=-1)
    hid = (
        jax.nn.silu(g_p.astype(jnp.float32)) * u_p.astype(jnp.float32)
    ).astype(tok_2d.dtype)
    y_e = jnp.matmul(hid, wo_s).astype(jnp.float32)
    return acc + jnp.where(in_e[:, None], y_e, 0.0), None

  y_sorted, _ = jax.lax.scan(
      _expert_step,
      jnp.zeros((buf_size, d_dim), dtype=jnp.float32),
      (starts, ends, wi_e, wo_e),
  )
  weighted = jnp.where(valid[:, None], y_sorted * w_rows_f32[:, None], 0.0)
  safe_dst = jnp.where(valid, tok_idx, t_ep)
  out_buf = jnp.zeros((t_ep + 1, d_dim), dtype=jnp.float32)
  out_2d = out_buf.at[safe_dst].add(weighted)[:t_ep]
  return out_2d.astype(tok_2d.dtype).reshape(orig_shape)


def moe_expert_gmm_and_unsort(
    dispatched: DispatchedMoeTokens,
    wi: Any,
    wo: Any,
    cfg: QwenConfig,
    *,
    return_weights: bool = False,
    forward_tie: Any = None,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
    use_external_bwd_rw: bool = False,
    use_external_bwd_disp: bool = False,
    return_routing: bool = False,
) -> Any:
  """Executes EP-local ragged expert GMM+unsort, delegating to `moe` when available."""
  moe_mod = _maybe_peer("moe") if cfg.use_pallas_kernels else None
  if moe_mod is not None and hasattr(moe_mod, "moe_expert_gmm_and_unsort"):
    return moe_mod.moe_expert_gmm_and_unsort(
        dispatched,
        wi,
        wo,
        cfg,
        return_weights=return_weights,
        forward_tie=forward_tie,
        wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
        sublayer_idx=sublayer_idx,
        use_external_bwd_rw=use_external_bwd_rw,
        use_external_bwd_disp=use_external_bwd_disp,
        return_routing=return_routing,
    )

  del wrap_partial_fp8_bwd, sublayer_idx, use_external_bwd_disp
  unperm_ep = _moe_expert_pure_jax_fallback(dispatched, wi, wo, cfg)
  if return_routing:
    return unperm_ep, _build_pure_jax_saved_routing(dispatched, cfg)
  tie_out = forward_tie
  rw3_slot = None
  if use_external_bwd_rw:
    e_local = cfg.local_num_experts
    rw3_slot = ops.zero_bwd_rw_prefetch(
        (e_local, cfg.emb_dim, 2 * cfg.moe_mlp_dim),
        (1,),
        (e_local, cfg.moe_mlp_dim, cfg.emb_dim),
        (1,),
        dtype=jnp.bfloat16,
    )
  return _format_moe_expert_return(
      unperm_ep,
      wi,
      wo,
      tie_out,
      rw3_slot,
      return_weights=return_weights,
      has_forward_tie=(forward_tie is not None),
      use_external_bwd_rw=use_external_bwd_rw,
  )


def moe_expert_gmm_and_unsort_from_routing(
    tokens_ep: Any,
    w_rows_f32: jax.Array,
    saved_routing: ops.MoeSavedRouting,
    wi: Any,
    wo: Any,
    cfg: QwenConfig,
    *,
    return_weights: bool = False,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
) -> Any:
  """Executes routed MoE GMM+unsort directly from FWD `saved_routing`."""
  moe_mod = _maybe_peer("moe") if cfg.use_pallas_kernels else None
  if moe_mod is not None and hasattr(
      moe_mod, "moe_expert_gmm_and_unsort_from_routing"
  ):
    return moe_mod.moe_expert_gmm_and_unsort_from_routing(
        tokens_ep,
        w_rows_f32,
        saved_routing,
        wi,
        wo,
        cfg,
        return_weights=return_weights,
        wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
        sublayer_idx=sublayer_idx,
    )
  del wrap_partial_fp8_bwd, sublayer_idx
  unperm_ep = _moe_expert_from_routing_pure_jax_fallback(
      tokens_ep, w_rows_f32, saved_routing, wi, wo, cfg
  )
  if return_weights:
    return unperm_ep, wi, wo
  return unperm_ep


def unsort_moe_row_weights(
    g_w_rows: jax.Array,
    saved_routing: ops.MoeSavedRouting,
    num_tokens_ep: int,
    cfg: QwenConfig,
) -> jax.Array:
  """Maps `[buf_size]` buffer-row weight gradients back to `[num_tokens_ep, K]`."""
  moe_mod = _maybe_peer("moe") if cfg.use_pallas_kernels else None
  if moe_mod is not None and hasattr(moe_mod, "unsort_moe_row_weights"):
    return moe_mod.unsort_moe_row_weights(
        g_w_rows, saved_routing, num_tokens_ep, cfg
    )
  cap = g_w_rows.shape[0]
  topk = cfg.num_experts_per_tok
  valid = jnp.arange(cap, dtype=jnp.int32) < saved_routing.count
  g_valid = jnp.where(valid, g_w_rows.astype(jnp.float32), 0.0)
  n_slots = num_tokens_ep * topk
  safe_dst = jnp.where(valid, saved_routing.slot_order, n_slots)
  buf = jnp.zeros((n_slots + 1,), dtype=jnp.float32)
  buf = buf.at[safe_dst].add(g_valid)
  return buf[:n_slots].reshape(num_tokens_ep, topk).astype(
      saved_routing.w_rows.dtype
  )


def moe_combine_and_add_base(
    partial_ep: jax.Array | ops.MoePartialTokens,
    base_residual: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """SC0-pinned BF16+ EP combine `ReduceScatter` + adds `base_residual` (`[B, S, D]`)."""
  t_local = math.prod(base_residual.shape[:-1])
  if isinstance(partial_ep, ops.MoePartialTokens):
    combined = (
        ops.ep_token_combine(partial_ep, cfg)
        if cfg.ep_size > 1
        else partial_ep.y_ep
    )
  else:
    if cfg.ep_size > 1 and partial_ep.shape[0] == cfg.ep_size * t_local:
      combined = ops.ep_token_combine(partial_ep, cfg)
    else:
      combined = partial_ep
  return base_residual + combined.reshape(base_residual.shape).astype(
      base_residual.dtype
  )
