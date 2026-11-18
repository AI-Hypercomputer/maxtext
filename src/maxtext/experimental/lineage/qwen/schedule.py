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

"""Pure-JAX Unsplit & Dual-SparseCore (SC0 EP + SC1 FSDP) Batch-Split Schedules.

Supports both Qwen MAX (`2.4T-A95B`, `92L = 23` superblocks) and Qwen 3.5
(`397B-A17B`, `60L = 15` superblocks) with:
- Core-minor `[ICI, core]` `('expert', 'ep_d2d')` `EP=16` (`EP=8` on `64c`)
- 3-axis D2D-decoupled `('fsdp', 'ep_d2d')` dense Stage-2 collectives
- Symmetric 4-phase 2-microbatch interleaving (`F1..F4` / `B4..B1`)
- `f2_early_mb0_dispatch`, `bwd_early_mb1_dy_pmax`, `wi_rs_sc0`,
  `defer_mb1_combine_after_f1`, optional P1 dense FP8 `gather8`, and optional
  P2 cross-superblock loop-boundary `SC1` collective pipelining.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
from typing import Any, NamedTuple

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen import config as config_lib

Qwen38Config = config_lib.Qwen38Config
QwenConfig = config_lib.QwenConfig

_QWEN_PKG = "maxtext.experimental.lineage.qwen"


class SublayerAbsmax(NamedTuple):
  wi: jax.Array | None
  wo: jax.Array | None


class SuperblockAbsmax(NamedTuple):
  gdn_0: SublayerAbsmax
  gdn_1: SublayerAbsmax
  gdn_2: SublayerAbsmax
  attn_3: SublayerAbsmax


class BatchSplitCarry(NamedTuple):
  x_mb0: jax.Array  # [B_mb, S, D]
  unperm_mb1: Any  # [T_ep, D] or 3D [T_ep, D//128, 128] or MoePartialTokens
  base_mb1: jax.Array  # [B_mb, S, D]
  bwd_rw3_prefetch: Any = None
  prefetched_w_dense_0: Any = None


@dataclasses.dataclass(frozen=True)
class ScheduleModules:
  """Resolved peer module hooks (injectable for standalone CPU schedule tests)."""

  ops: Any
  params: Any
  layers: Any
  moe_kernels: Any


@functools.lru_cache(maxsize=8)
def _accepts_kwarg(fn: Any, kwarg: str) -> bool:
  try:
    sig = inspect.signature(fn)
  except (TypeError, ValueError):
    return False
  for p in sig.parameters.values():
    if p.kind == inspect.Parameter.VAR_KEYWORD:
      return True
  return kwarg in sig.parameters


def _resolve_modules(modules: ScheduleModules | None = None) -> ScheduleModules:
  """Lazily resolves `ops`, `params`, `layers`, and `moe_kernels` from `models.qwen`."""
  if modules is not None:
    return modules
  # Sibling modules (`ops`, `params`, `layers`, `moe_kernels`) are owned by
  # other workers and linked by the integrating binary, so resolve them
  # dynamically instead of via static BUILD deps.
  layers = importlib.import_module(f"{_QWEN_PKG}.layers")
  ops = importlib.import_module(f"{_QWEN_PKG}.ops")
  params = importlib.import_module(f"{_QWEN_PKG}.params")
  moe_kernels = layers
  for cand in ("moe_kernels", "moe"):
    try:
      moe_kernels = importlib.import_module(f"{_QWEN_PKG}.{cand}")
      break
    except ImportError:
      continue
  return ScheduleModules(
      ops=ops, params=params, layers=layers, moe_kernels=moe_kernels
  )


def _unwrap_dep_tensor(dep: Any) -> Any:
  """Extracts a tieable array from FP8 weight carriers when needed."""
  if hasattr(dep, "codes") and not hasattr(dep, "codes_ep"):
    return dep.codes
  return dep


def _replace_struct(obj: Any, **updates: Any) -> Any:
  """Returns a copy of a dataclass or NamedTuple with updated fields."""
  if dataclasses.is_dataclass(obj):
    return dataclasses.replace(obj, **updates)
  if hasattr(obj, "_replace"):
    return obj._replace(**updates)
  raise TypeError(f"Expected dataclass or NamedTuple, got {type(obj).__name__}")


def _assert_core_minor_config(cfg: QwenConfig) -> None:
  """Verifies core-minor `[ICI, core]` ordering unless explicitly testing core-major."""
  if cfg.decouple_ep_d2d and getattr(cfg, "ep_core_minor", True):
    if tuple(cfg.ep_axis_tuple) != ("expert", "ep_d2d"):
      raise ValueError(
          "Core-minor [ICI, core] invariant requires"
          f" ep_axis_tuple == ('expert', 'ep_d2d'), got {cfg.ep_axis_tuple!r}."
      )
    if tuple(cfg.dense_stage2_axis_names) != ("fsdp", "ep_d2d"):
      raise ValueError(
          "Core-minor [ICI, core] invariant requires dense_stage2_axis_names =="
          f" ('fsdp', 'ep_d2d'), got {cfg.dense_stage2_axis_names!r}."
      )


def _call_route_only(
    layers_mod: Any,
    normed_tokens: jax.Array,
    router_gate: Any,
    cfg: QwenConfig,
    *,
    route_after: Any = None,
    sublayer_idx: int = 0,
    batch_offset: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array]:
  """Calls `layers.moe_local_route_only` forwarding supported metadata args."""
  fn = layers_mod.moe_local_route_only
  kwargs: dict[str, Any] = {}
  if route_after is not None:
    kwargs["route_after"] = route_after
  if _accepts_kwarg(fn, "sublayer_idx"):
    kwargs["sublayer_idx"] = sublayer_idx
  if batch_offset and _accepts_kwarg(fn, "batch_offset"):
    kwargs["batch_offset"] = batch_offset
  if layer_idx is not None and _accepts_kwarg(fn, "layer_idx"):
    kwargs["layer_idx"] = layer_idx
  if _accepts_kwarg(fn, "step"):
    kwargs["step"] = step
  return fn(normed_tokens, router_gate, cfg, **kwargs)


def _call_route_and_dispatch(
    layers_mod: Any,
    normed_tokens: jax.Array,
    router_gate: Any,
    cfg: QwenConfig,
    *,
    route_after: Any = None,
    sublayer_idx: int = 0,
    batch_offset: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> Any:
  """Calls `layers.moe_local_route_and_dispatch` with supported metadata args."""
  fn = layers_mod.moe_local_route_and_dispatch
  kwargs: dict[str, Any] = {}
  if route_after is not None:
    kwargs["route_after"] = route_after
  if _accepts_kwarg(fn, "sublayer_idx"):
    kwargs["sublayer_idx"] = sublayer_idx
  if batch_offset and _accepts_kwarg(fn, "batch_offset"):
    kwargs["batch_offset"] = batch_offset
  if layer_idx is not None and _accepts_kwarg(fn, "layer_idx"):
    kwargs["layer_idx"] = layer_idx
  if _accepts_kwarg(fn, "step"):
    kwargs["step"] = step
  return fn(normed_tokens, router_gate, cfg, **kwargs)


def _gather_dense_proj(
    ops_mod: Any,
    w_sharded: jax.Array,
    cfg: QwenConfig,
    *,
    sharded_axis: int = 0,
    is_fsdp_expert: bool = False,
    start_after: Any = None,
    use_fp8: bool = False,
) -> Any:
  """Gathers a dense projection on SC1, optionally using FP8 `gather8` when supported."""
  sa = _unwrap_dep_tensor(start_after) if start_after is not None else None
  if use_fp8 and hasattr(ops_mod, "gather_dense_weight_fp8"):
    return ops_mod.gather_dense_weight_fp8(
        w_sharded,
        cfg,
        sharded_axis=sharded_axis,
        is_fsdp_expert=is_fsdp_expert,
        start_after=sa,
    )
  fn = ops_mod.gather_dense_weight
  kwargs: dict[str, Any] = {
      "sharded_axis": sharded_axis,
      "is_fsdp_expert": is_fsdp_expert,
      "start_after": sa,
  }
  if use_fp8 and _accepts_kwarg(fn, "use_fp8"):
    kwargs["use_fp8"] = True
  return fn(w_sharded, cfg, **kwargs)


# ---------------------------------------------------------------------------
# Reference 1-Microbatch ("unsplit") Sub-Layer & Superblock Schedule
# ---------------------------------------------------------------------------


def _run_sublayer_unsplit(
    x: jax.Array,
    sub_params: Any,
    sub_absmax: SublayerAbsmax,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    sublayer_idx: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    mods: ScheduleModules,
) -> jax.Array:
  """Executes a single sub-layer without microbatch splitting."""
  layers = mods.layers
  ops = mods.ops
  x = checkpoint_name(x, "decoder_layer_input")
  if is_attn:
    w = layers.gather_attn_sublayer_weights(
        sub_params, sub_absmax.wi, sub_absmax.wo, cfg
    )
    h = layers.full_attn_mixer(
        x, w, positions, segment_ids, cfg, rope_cos_sin=rope_cos_sin
    )
  else:
    w = layers.gather_gdn_sublayer_weights(
        sub_params, sub_absmax.wi, sub_absmax.wo, cfg
    )
    h = layers.gdn_mixer(x, w, segment_ids, cfg)

  normed = ops.rms_norm_zero_centered(h, w.post_attn_norm, cfg.norm_eps)
  base = layers.shared_expert_and_residual(h, normed, w.moe)
  disp = _call_route_and_dispatch(
      layers,
      normed,
      w.moe.router_gate,
      cfg,
      sublayer_idx=sublayer_idx,
      layer_idx=layer_idx,
      step=step,
  )
  wrap_pt = cfg.moe_use_forked_3d_gmmv2 and cfg.moe_bwd_dy_fp8_e5m2
  unperm = layers.moe_expert_gmm_and_unsort(
      disp,
      w.moe.wi,
      w.moe.wo,
      cfg,
      return_weights=False,
      wrap_partial_fp8_bwd=wrap_pt,
      sublayer_idx=sublayer_idx,
  )
  return layers.moe_combine_and_add_base(unperm, base, cfg)


def run_superblock_unsplit(
    x: jax.Array,
    sb_params: Any,
    sb_absmax: SuperblockAbsmax,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: Qwen38Config,
    *,
    sb_idx: jax.Array | int = 0,
    step: jax.Array | int = 0,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    modules: ScheduleModules | None = None,
) -> jax.Array:
  """Executes `[GDN_0, GDN_1, GDN_2, FullAttn_3]` sequentially on full local batch."""
  _assert_core_minor_config(cfg)
  mods = _resolve_modules(modules)
  base_layer_idx = sb_idx * 4
  x = _run_sublayer_unsplit(
      x,
      sb_params.gdn_0,
      sb_absmax.gdn_0,
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=0,
      layer_idx=base_layer_idx + 0,
      step=step,
      mods=mods,
  )
  x = _run_sublayer_unsplit(
      x,
      sb_params.gdn_1,
      sb_absmax.gdn_1,
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=1,
      layer_idx=base_layer_idx + 1,
      step=step,
      mods=mods,
  )
  x = _run_sublayer_unsplit(
      x,
      sb_params.gdn_2,
      sb_absmax.gdn_2,
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=2,
      layer_idx=base_layer_idx + 2,
      step=step,
      mods=mods,
  )
  x = _run_sublayer_unsplit(
      x,
      sb_params.attn_3,
      sb_absmax.attn_3,
      positions,
      segment_ids,
      cfg,
      is_attn=True,
      sublayer_idx=3,
      layer_idx=base_layer_idx + 3,
      step=step,
      rope_cos_sin=rope_cos_sin,
      mods=mods,
  )
  return x


# ---------------------------------------------------------------------------
# Dual-SparseCore (SC0 EP + SC1 FSDP) 2-Microbatch Batch-Split Schedule
# ---------------------------------------------------------------------------


def _tie_bwd_sharded_params(
    x_in: tuple[jax.Array, jax.Array],
    sub_params: Any,
    cfg: QwenConfig,
    *,
    ops_mod: Any,
):
  """Applies zero-residual post-RS backward barrier on sharded sub-layer weights."""
  if cfg.bwd_tie_mode == "none":
    return x_in, sub_params

  if cfg.bwd_tie_mode in ("moe_xmb0_only", "symmetric_4phase"):
    # Tie ONLY `x_mb0` (not `x_mb1`) to `(wi, wo)` post-RS cotangents so
    # `dx_mb1` launches cross-layer SC0 EP Combine-Bwd during `B1`.
    x_mb0_in, x_mb1_in = x_in
    x_mb0_t, (wi_t, wo_t) = ops_mod.backward_barrier_tie(
        x_mb0_in, (sub_params.moe.wi, sub_params.moe.wo)
    )
    moe_updated = _replace_struct(sub_params.moe, wi=wi_t, wo=wo_t)
    return (x_mb0_t, x_mb1_in), _replace_struct(sub_params, moe=moe_updated)

  if cfg.bwd_tie_mode == "moe_only":
    x_tied, (wi_t, wo_t) = ops_mod.backward_barrier_tie(
        x_in, (sub_params.moe.wi, sub_params.moe.wo)
    )
    moe_updated = _replace_struct(sub_params.moe, wi=wi_t, wo=wo_t)
    return x_tied, _replace_struct(sub_params, moe=moe_updated)

  # "all4": tie all sharded MoE + dense weights of this sub-layer!
  return ops_mod.backward_barrier_tie(x_in, sub_params)


def _tie_bwd_dense_only_params(
    anchor: Any,
    sub_params: Any,
    *,
    ops_mod: Any,
):
  """Ties ONLY dense + shared-expert + router sharded grads to `anchor`."""
  moe_no_rw = _replace_struct(sub_params.moe, wi=None, wo=None)
  sub_no_rw = _replace_struct(sub_params, moe=moe_no_rw)
  anchor_tied, sub_no_rw_tied = ops_mod.backward_barrier_tie(anchor, sub_no_rw)
  moe_restored = _replace_struct(
      sub_no_rw_tied.moe, wi=sub_params.moe.wi, wo=sub_params.moe.wo
  )
  return anchor_tied, _replace_struct(sub_no_rw_tied, moe=moe_restored)


def _proj_fsdp_expert(cfg: QwenConfig, unsharded_cols: int) -> bool:
  if hasattr(cfg, "dense_proj_fsdp_expert"):
    return cfg.dense_proj_fsdp_expert(unsharded_cols)
  return cfg.dense_sharding_mode == "fsdp_expert"


def _dense_rs_grads_only(g_p: Any) -> Any:
  """Keeps ONLY sharded dense/shared/router RS grads (excludes `wi`/`wo` & replicated params)."""
  g_moe = _replace_struct(g_p.moe, wi=None, wo=None, shared_gate=None)
  if hasattr(g_p, "q_proj"):
    return _replace_struct(
        g_p,
        pre_attn_norm=None,
        post_attn_norm=None,
        q_norm=None,
        k_norm=None,
        moe=g_moe,
    )
  return _replace_struct(
      g_p,
      pre_attn_norm=None,
      post_attn_norm=None,
      conv1d=None,
      A_log=None,
      dt_bias=None,
      gdn_norm=None,
      moe=g_moe,
  )


def _gather_moe_dense_bundled(
    p_moe: Any,
    ops: Any,
    layers: Any,
    cfg: QwenConfig,
    *,
    is_fe: bool,
    start_after: Any,
) -> Any:
  """Bundles dense MoE Stage-1 `'expert'` & Stage-2 `(router_gate, shared_gate)` collectives."""
  wi0 = p_moe.shared_wi_0
  wi1 = p_moe.shared_wi_1
  wo_t = p_moe.shared_wo.T
  rg = p_moe.router_gate
  sg = p_moe.shared_gate
  router_fe = _proj_fsdp_expert(cfg, cfg.num_experts)
  shared_fe = _proj_fsdp_expert(cfg, cfg.shared_mlp_dim)
  if (
      getattr(cfg, "replicate_shared_gate", False)
      or router_fe != is_fe
      or shared_fe != is_fe
      or len(
          {(a.shape[0], jnp.dtype(a.dtype)) for a in (wi0, wi1, wo_t, rg, sg)}
      )
      != 1
  ):
    return None
  sa = _unwrap_dep_tensor(start_after) if start_after is not None else None
  wi0_ep, wi1_ep, wo_t_ep, rg_ep, sg_ep = ops.gather_dense_stage1_bundled(
      (wi0, wi1, wo_t, rg, sg),
      cfg,
      is_fsdp_expert=is_fe,
      start_after=sa,
      min_collective_bytes=0,
  )
  (wi0_full,) = ops.gather_dense_stage2_bundled(
      (wi0_ep,), cfg, is_fsdp_expert=is_fe
  )
  (wi1_full,) = ops.gather_dense_stage2_bundled(
      (wi1_ep,), cfg, is_fsdp_expert=is_fe
  )
  (wo_t_full,) = ops.gather_dense_stage2_bundled(
      (wo_t_ep,), cfg, is_fsdp_expert=is_fe
  )
  rg_full, sg_full = ops.gather_dense_stage2_bundled(
      (rg_ep, sg_ep), cfg, is_fsdp_expert=is_fe
  )
  return layers.GatheredMoeWeights(
      router_gate=rg_full,
      wi=p_moe.wi,
      wo=p_moe.wo,
      shared_wi_0=wi0_full,
      shared_wi_1=wi1_full,
      shared_wo=wo_t_full.T,
      shared_gate=sg_full,
  )


def _gather_dense_and_norms(
    sub_params: Any,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    start_after: Any = None,
    mods: ScheduleModules,
):
  """Gathers dense mixer + shared-expert + router weights on SC1, leaving `wi`/`wo` separate."""
  ops = mods.ops
  layers = mods.layers
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  fp8_shared = bool(
      getattr(cfg, "dense_fp8_matmul", False)
      or getattr(cfg, "dense_fp8_shared_expert", False)
  )
  fp8_gdn = bool(
      getattr(cfg, "dense_fp8_matmul", False)
      or getattr(cfg, "dense_fp8_gdn", False)
  )
  fp8_attn = bool(
      getattr(cfg, "dense_fp8_matmul", False)
      or getattr(cfg, "dense_fp8_attn", False)
  )
  use_bundled = (
      bool(getattr(cfg, "bundle_stage1_dense_collectives", False))
      and hasattr(ops, "gather_dense_stage1_bundled")
      and hasattr(ops, "gather_dense_stage2_bundled")
      and hasattr(ops, "gather_dense_weights_bundled")
      and not fp8_shared
      and not (fp8_attn if is_attn else fp8_gdn)
  )

  p_moe = sub_params.moe
  moe_dense = None
  if use_bundled:
    moe_dense = _gather_moe_dense_bundled(
        p_moe,
        ops,
        layers,
        cfg,
        is_fe=is_fe,
        start_after=start_after,
    )
  if moe_dense is None:
    router_fe = _proj_fsdp_expert(cfg, cfg.num_experts)
    shared_fe = _proj_fsdp_expert(cfg, cfg.shared_mlp_dim)
    if getattr(cfg, "replicate_shared_gate", False):
      shared_gate_w = ops.replicate_small_param(p_moe.shared_gate, cfg)
    else:
      shared_gate_w = _gather_dense_proj(
          ops,
          p_moe.shared_gate,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          start_after=start_after,
          use_fp8=False,
      )
    moe_dense = layers.GatheredMoeWeights(
        router_gate=_gather_dense_proj(
            ops,
            p_moe.router_gate,
            cfg,
            sharded_axis=0,
            is_fsdp_expert=router_fe,
            start_after=start_after,
            use_fp8=False,  # Router gate stays BF16/FP32 for top-k logits
        ),
        wi=p_moe.wi,
        wo=p_moe.wo,
        shared_wi_0=_gather_dense_proj(
            ops,
            p_moe.shared_wi_0,
            cfg,
            sharded_axis=0,
            is_fsdp_expert=shared_fe,
            start_after=start_after,
            use_fp8=fp8_shared,
        ),
        shared_wi_1=_gather_dense_proj(
            ops,
            p_moe.shared_wi_1,
            cfg,
            sharded_axis=0,
            is_fsdp_expert=shared_fe,
            start_after=start_after,
            use_fp8=fp8_shared,
        ),
        shared_wo=_gather_dense_proj(
            ops,
            p_moe.shared_wo,
            cfg,
            sharded_axis=1,
            is_fsdp_expert=shared_fe,
            start_after=start_after,
            use_fp8=fp8_shared,
        ),
        shared_gate=shared_gate_w,
    )

  sa = _unwrap_dep_tensor(start_after) if start_after is not None else None
  if is_attn:
    attn_kv_fe = _proj_fsdp_expert(cfg, cfg.num_kv_heads * cfg.head_dim)
    q_in = (
        sub_params.q_proj.reshape(sub_params.q_proj.shape[0], -1)
        if cfg.attn_qkv_2d_matmul
        else sub_params.q_proj
    )
    k_in = (
        sub_params.k_proj.reshape(sub_params.k_proj.shape[0], -1)
        if cfg.attn_qkv_2d_matmul
        else sub_params.k_proj
    )
    v_in = (
        sub_params.v_proj.reshape(sub_params.v_proj.shape[0], -1)
        if cfg.attn_qkv_2d_matmul
        else sub_params.v_proj
    )
    q_2d = sub_params.q_proj.reshape(sub_params.q_proj.shape[0], -1)
    k_2d = sub_params.k_proj.reshape(sub_params.k_proj.shape[0], -1)
    v_2d = sub_params.v_proj.reshape(sub_params.v_proj.shape[0], -1)
    if (
        use_bundled
        and attn_kv_fe == is_fe
        and len({(a.shape[0], jnp.dtype(a.dtype)) for a in (q_2d, k_2d, v_2d)})
        == 1
    ):
      q_full_2d, k_full_2d, v_full_2d = ops.gather_dense_weights_bundled(
          (q_2d, k_2d, v_2d),
          cfg,
          is_fsdp_expert=is_fe,
          start_after=sa,
          min_collective_bytes=0,
      )
      if cfg.attn_qkv_2d_matmul:
        q_proj, k_proj, v_proj = q_full_2d, k_full_2d, v_full_2d
      else:
        q_proj, k_proj, v_proj = (
            full.reshape(full.shape[0], *orig.shape[1:])
            for full, orig in (
                (q_full_2d, sub_params.q_proj),
                (k_full_2d, sub_params.k_proj),
                (v_full_2d, sub_params.v_proj),
            )
        )
      (out_ep,) = ops.gather_dense_stage1_bundled(
          (sub_params.out_proj,),
          cfg,
          is_fsdp_expert=is_fe,
          start_after=sa,
          min_collective_bytes=0,
      )
      (out_proj,) = ops.gather_dense_stage2_bundled(
          (out_ep,), cfg, is_fsdp_expert=is_fe
      )
    else:
      q_proj = _gather_dense_proj(
          ops,
          q_in,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          start_after=start_after,
          use_fp8=fp8_attn,
      )
      k_proj = _gather_dense_proj(
          ops,
          k_in,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=attn_kv_fe,
          start_after=start_after,
          use_fp8=fp8_attn,
      )
      v_proj = _gather_dense_proj(
          ops,
          v_in,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=attn_kv_fe,
          start_after=start_after,
          use_fp8=fp8_attn,
      )
      out_proj = _gather_dense_proj(
          ops,
          sub_params.out_proj,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=is_fe,
          start_after=start_after,
          use_fp8=fp8_attn,
      )
    return layers.GatheredAttnWeights(
        pre_attn_norm=ops.replicate_small_param(sub_params.pre_attn_norm, cfg),
        post_attn_norm=ops.replicate_small_param(
            sub_params.post_attn_norm, cfg
        ),
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        q_norm=ops.replicate_small_param(sub_params.q_norm, cfg),
        k_norm=ops.replicate_small_param(sub_params.k_norm, cfg),
        out_proj=out_proj,
        moe=moe_dense,
    )

  gdn_ba_fe = _proj_fsdp_expert(cfg, 2 * cfg.gdn_num_v_heads)
  if (
      use_bundled
      and gdn_ba_fe == is_fe
      and len({
          (a.shape[0], jnp.dtype(a.dtype))
          for a in (sub_params.in_proj_qkvz, sub_params.in_proj_ba)
      })
      == 1
  ):
    in_proj_qkvz, in_proj_ba = ops.gather_dense_weights_bundled(
        (sub_params.in_proj_qkvz, sub_params.in_proj_ba),
        cfg,
        is_fsdp_expert=is_fe,
        start_after=sa,
        min_collective_bytes=0,
    )
    (out_ep,) = ops.gather_dense_stage1_bundled(
        (sub_params.out_proj,),
        cfg,
        is_fsdp_expert=is_fe,
        start_after=sa,
        min_collective_bytes=0,
    )
    (out_proj,) = ops.gather_dense_stage2_bundled(
        (out_ep,), cfg, is_fsdp_expert=is_fe
    )
  else:
    in_proj_qkvz = _gather_dense_proj(
        ops,
        sub_params.in_proj_qkvz,
        cfg,
        sharded_axis=0,
        is_fsdp_expert=is_fe,
        start_after=start_after,
        use_fp8=fp8_gdn,
    )
    in_proj_ba = _gather_dense_proj(
        ops,
        sub_params.in_proj_ba,
        cfg,
        sharded_axis=0,
        is_fsdp_expert=gdn_ba_fe,
        start_after=start_after,
        use_fp8=False,
    )
    out_proj = _gather_dense_proj(
        ops,
        sub_params.out_proj,
        cfg,
        sharded_axis=0,
        is_fsdp_expert=is_fe,
        start_after=start_after,
        use_fp8=fp8_gdn,
    )
  return layers.GatheredGdnWeights(
      pre_attn_norm=ops.replicate_small_param(sub_params.pre_attn_norm, cfg),
      post_attn_norm=ops.replicate_small_param(sub_params.post_attn_norm, cfg),
      in_proj_qkvz=in_proj_qkvz,
      in_proj_ba=in_proj_ba,
      conv1d=ops.replicate_small_param(sub_params.conv1d, cfg),
      A_log=ops.replicate_small_param(sub_params.A_log, cfg),
      dt_bias=ops.replicate_small_param(sub_params.dt_bias, cfg),
      gdn_norm=ops.replicate_small_param(sub_params.gdn_norm, cfg),
      out_proj=out_proj,
      moe=moe_dense,
  )


def _dense_loop_boundary_bridge(
    gdn0_sharded: Any,
    prefetched_w_dense_0: Any,
    cfg: QwenConfig,
    *,
    mods: ScheduleModules,
) -> Any:
  """Zero-residual cross-superblock bridge using prefetched `w_dense_0` in fwd.

  In forward, returns `prefetched_w_dense_0` with `()` empty residuals so
  `jax.checkpoint` saves ZERO `[num_superblocks, ...]` gathered weights.
  In backward, reduces `gw_dense_0` back to `gdn0_sharded` via
  `_gather_dense_and_norms`'s exact VJP transpose (`SC1` BF16 `psum_scatter`).

  Args:
    gdn0_sharded: Sharded parameters for the next superblock's `gdn_0` layer.
    prefetched_w_dense_0: Gathered forward dense weights (`stop_gradient`).
    cfg: Model and mesh configuration.
    mods: Resolved Qwen sibling module handle bundle.

  Returns:
    Gathered `w_dense_0` structure with restored `wi`/`wo` sharded MoE weights.
  """
  gdn0_no_rw = _replace_struct(
      gdn0_sharded,
      moe=_replace_struct(gdn0_sharded.moe, wi=None, wo=None),
  )
  pref_no_rw = _replace_struct(
      prefetched_w_dense_0,
      moe=_replace_struct(prefetched_w_dense_0.moe, wi=None, wo=None),
  )

  def _gather_only_dense(p_no_rw: Any) -> Any:
    p_full = _replace_struct(
        p_no_rw,
        moe=_replace_struct(
            p_no_rw.moe,
            wi=jnp.zeros((), jnp.bfloat16),
            wo=jnp.zeros((), jnp.bfloat16),
        ),
    )
    g = _gather_dense_and_norms(
        p_full, cfg, is_attn=False, start_after=None, mods=mods
    )
    return _replace_struct(g, moe=_replace_struct(g.moe, wi=None, wo=None))

  p_shape = jax.tree.map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), gdn0_no_rw
  )

  @jax.custom_vjp
  def _bridge(p_in: Any, pref_in: Any) -> Any:
    del p_in
    return pref_in

  def _bridge_fwd(p_in: Any, pref_in: Any):
    del p_in
    return pref_in, ()

  def _bridge_bwd(_, g_w_dense: Any):
    _, vjp_fn = jax.vjp(
        _gather_only_dense,
        jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), p_shape),
    )
    (g_p_sharded,) = vjp_fn(g_w_dense)
    g_pref_zero = jax.tree.map(
        lambda v: jnp.zeros_like(v) if isinstance(v, jax.Array) else v,
        g_w_dense,
    )
    return g_p_sharded, g_pref_zero

  _bridge.defvjp(_bridge_fwd, _bridge_bwd)
  bridged = _bridge(gdn0_no_rw, pref_no_rw)
  return _replace_struct(
      bridged,
      moe=_replace_struct(
          bridged.moe, wi=gdn0_sharded.moe.wi, wo=gdn0_sharded.moe.wo
      ),
  )


def prepare_loop_boundary_dense_pipeline(
    blocks_local: Any,
    cfg: QwenConfig,
    *,
    modules: ScheduleModules | None = None,
) -> tuple[Any, Any]:
  """Prepares `(init_w_dense_0, shifted_next_gdn0_sg)` for loop-boundary `SC1`.

  Args:
    blocks_local: Sharded superblock parameter stack (`[num_superblocks, ...]`).
    cfg: Model and mesh configuration.
    modules: Optional pre-resolved sibling module bundle.

  Returns:
    Tuple `(init_w_dense_0, shifted_next_gdn0_sg)` where `init_w_dense_0` is
    prologue-prefetched (`stop_gradient`) for `BatchSplitCarry` and
    `shifted_next_gdn0_sg` shifts `gdn_0[1..B-1, 0]` for `next_gdn0_params`.
  """
  if cfg.fsdp_pipeline_mode != "full_prefetch_defer_rs":
    return None, None
  mods = _resolve_modules(modules)
  gdn0_first = jax.tree.map(
      lambda v: jax.lax.stop_gradient(v[0]), blocks_local.gdn_0
  )
  init_w_dense_0 = jax.tree.map(
      jax.lax.stop_gradient,
      _gather_dense_and_norms(
          gdn0_first, cfg, is_attn=False, start_after=None, mods=mods
      ),
  )

  def _shift_next_sg(arr: jax.Array) -> jax.Array:
    sg = jax.lax.stop_gradient(arr)
    return jnp.concatenate([sg[1:], sg[:1]], axis=0)

  shifted_next_gdn0_sg = jax.tree.map(_shift_next_sg, blocks_local.gdn_0)
  return init_w_dense_0, shifted_next_gdn0_sg


def _mixer_only(
    x_mb: jax.Array,
    w_dense: Any,
    pos_mb: jax.Array,
    seg_mb: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    mods: ScheduleModules,
) -> tuple[jax.Array, jax.Array]:
  """Runs GDN or FullAttn mixer + post_attn_norm returning `(h, normed)`."""
  if is_attn:
    h = mods.layers.full_attn_mixer(
        x_mb, w_dense, pos_mb, seg_mb, cfg, rope_cos_sin=rope_cos_sin
    )
  else:
    h = mods.layers.gdn_mixer(x_mb, w_dense, seg_mb, cfg)
  normed = mods.ops.rms_norm_zero_centered(
      h, w_dense.post_attn_norm, cfg.norm_eps
  )
  return h, normed


def _mixer_and_shared(
    x_mb: jax.Array,
    w_dense: Any,
    pos_mb: jax.Array,
    seg_mb: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    mods: ScheduleModules,
) -> tuple[jax.Array, jax.Array]:
  """Runs GDN or FullAttn mixer + post_attn_norm + gated Shared Expert."""
  h, normed = _mixer_only(
      x_mb,
      w_dense,
      pos_mb,
      seg_mb,
      cfg,
      is_attn=is_attn,
      rope_cos_sin=rope_cos_sin,
      mods=mods,
  )
  base = mods.layers.shared_expert_and_residual(h, normed, w_dense.moe)
  return base, normed


def _run_sublayer_batchsplit(
    carry: BatchSplitCarry,
    sub_params: Any,
    sub_absmax: SublayerAbsmax,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: QwenConfig,
    *,
    is_attn: bool,
    sublayer_idx: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
    prefetched_w_dense: Any = None,
    prefetched_moe_rw: tuple[Any, Any] | None = None,
    moe_rw_start_after: Any = None,
    next_sub_params: Any = None,
    next_is_attn: bool = False,
    next_prefetch_stop_grad: bool = False,
    skip_own_bwd_tie: bool = False,
    is_superblock_last: bool = False,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    prev_rw3_local: (
        tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None
    ) = None,
    use_external_bwd_rw: bool = False,
    prefetched_x_mb1: jax.Array | None = None,
    return_entry_pair: bool = False,
    mods: ScheduleModules,
):
  """4-Phase Combine-Wrapped Sub-Layer with Strict SC0 (EP) / SC1 (FSDP) Separation."""
  ops = mods.ops
  layers = mods.layers
  moe_kernels = mods.moe_kernels

  x_mb0, unperm_mb1_in, base_mb1_in, bwd_rw3_in, _ = carry
  entry_x_mb0 = x_mb0
  is_partial_tok = isinstance(unperm_mb1_in, ops.MoePartialTokens)
  if cfg.bwd_tie_b1_before_prev_b4 and is_partial_tok:
    unperm_mb1_in, x_mb0 = ops.backward_barrier_tie(unperm_mb1_in, x_mb0)

  defer_f1_combine = cfg.defer_mb1_combine_after_f1 and cfg.bwd_tie_mode in (
      "moe_xmb0_only",
      "symmetric_4phase",
      "none",
  )
  if defer_f1_combine:
    # Launch `prev mb1` SC0 EP Combine `psum_scatter` before Phase F1,
    # deferring its `call-done` (`base_mb1_in + combined_mb1_prev`) past F1.
    if prefetched_x_mb1 is None:
      combined_mb1_prev = jax.lax.stop_gradient(
          ops.ep_token_combine(unperm_mb1_in, cfg)
      )
    else:
      combined_mb1_prev = None
    x_mb0 = checkpoint_name(x_mb0, "decoder_layer_input")
    if not skip_own_bwd_tie:
      (x_mb0, _), sub_params = _tie_bwd_sharded_params(
          (x_mb0, x_mb0), sub_params, cfg, ops_mod=ops
      )
  else:
    if prefetched_x_mb1 is None:
      x_mb1_raw = jax.lax.stop_gradient(
          layers.moe_combine_and_add_base(unperm_mb1_in, base_mb1_in, cfg)
      )
    else:
      x_mb1_raw = prefetched_x_mb1
    entry_x_mb1_raw = x_mb1_raw
    x_mb1 = ops.combine_bwd_bridge(cfg, unperm_mb1_in, base_mb1_in, x_mb1_raw)
    x_mb0 = checkpoint_name(x_mb0, "decoder_layer_input")
    x_mb1 = checkpoint_name(x_mb1, "decoder_layer_input")
    if not skip_own_bwd_tie:
      (x_mb0, x_mb1), sub_params = _tie_bwd_sharded_params(
          (x_mb0, x_mb1), sub_params, cfg, ops_mod=ops
      )

  w_dense = (
      prefetched_w_dense
      if prefetched_w_dense is not None
      else _gather_dense_and_norms(sub_params, cfg, is_attn=is_attn, mods=mods)
  )

  early_moe_rw = None
  if prefetched_moe_rw is None and moe_rw_start_after is not None:
    sa_dep = _unwrap_dep_tensor(moe_rw_start_after)
    early_moe_rw = (
        moe_kernels.gather_routed_weight_sc1(
            sub_params.moe.wi,
            sub_absmax.wi,
            cfg,
            start_after=sa_dep,
            is_wi=True,
        ),
        moe_kernels.gather_routed_weight_sc1(
            sub_params.moe.wo,
            sub_absmax.wo,
            cfg,
            start_after=sa_dep,
            is_wi=False,
        ),
    )

  # === Phase F1 & F2 ===
  if prefetched_x_mb1 is not None:
    w_f1 = w_dense
  else:
    w_f1 = _replace_struct(
        w_dense,
        pre_attn_norm=ops.order_after(w_dense.pre_attn_norm, unperm_mb1_in),
    )
  if cfg.route_before_shared_expert:
    h_mb0, normed_mb0 = _mixer_only(
        x_mb0,
        w_f1,
        pos_mb0,
        seg_mb0,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb0,
        mods=mods,
    )
    topk_w_mb0, topk_ids_mb0 = _call_route_only(
        layers,
        normed_mb0,
        w_dense.moe.router_gate,
        cfg,
        sublayer_idx=sublayer_idx,
        layer_idx=layer_idx,
        step=step,
    )
    disp_mb0 = layers.moe_dispatch_from_routed(
        normed_mb0, topk_w_mb0, topk_ids_mb0, cfg, order_tokens_on_router=True
    )
    normed_mb0_se = ops.order_after(normed_mb0, topk_w_mb0)
    base_mb0 = layers.shared_expert_and_residual(h_mb0, normed_mb0_se, w_f1.moe)
  elif cfg.f2_light_router_ties:
    base_mb0, normed_mb0 = _mixer_and_shared(
        x_mb0,
        w_f1,
        pos_mb0,
        seg_mb0,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb0,
        mods=mods,
    )
    topk_w_mb0, topk_ids_mb0 = _call_route_only(
        layers,
        normed_mb0,
        w_dense.moe.router_gate,
        cfg,
        route_after=base_mb0,
        sublayer_idx=sublayer_idx,
        layer_idx=layer_idx,
        step=step,
    )
    disp_mb0 = layers.moe_dispatch_from_routed(
        normed_mb0,
        topk_w_mb0,
        topk_ids_mb0,
        cfg,
        order_tokens_on_router=not cfg.f2_early_mb0_dispatch,
    )
  else:
    topk_w_mb0 = None
    base_mb0, normed_mb0 = _mixer_and_shared(
        x_mb0,
        w_f1,
        pos_mb0,
        seg_mb0,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb0,
        mods=mods,
    )
    disp_mb0 = _call_route_and_dispatch(
        layers,
        normed_mb0,
        w_dense.moe.router_gate,
        cfg,
        route_after=base_mb0,
        sublayer_idx=sublayer_idx,
        layer_idx=layer_idx,
        step=step,
    )

  if defer_f1_combine:
    if prefetched_x_mb1 is None:
      assert combined_mb1_prev is not None
      base_mb1_f1 = ops.order_after(base_mb1_in, base_mb0)
      if cfg.f2_early_mb0_dispatch and topk_w_mb0 is not None:
        base_mb1_f1 = ops.order_after(base_mb1_f1, topk_w_mb0)
      x_mb1_raw = jax.lax.stop_gradient(
          base_mb1_f1 + combined_mb1_prev.reshape(base_mb1_in.shape)
      )
    else:
      x_mb1_raw = ops.order_after(prefetched_x_mb1, base_mb0)
      if cfg.f2_early_mb0_dispatch and topk_w_mb0 is not None:
        x_mb1_raw = ops.order_after(x_mb1_raw, topk_w_mb0)
    entry_x_mb1_raw = x_mb1_raw
    unperm_for_bridge = unperm_mb1_in
    if cfg.bwd_early_mb1_dy_pmax and is_partial_tok:
      base_mb0, unperm_for_bridge = ops.tie_bwd_after_partial_dy_scale(
          base_mb0, unperm_mb1_in
      )
    x_mb1 = ops.combine_bwd_bridge(
        cfg, unperm_for_bridge, base_mb1_in, x_mb1_raw
    )
    x_mb1 = checkpoint_name(x_mb1, "decoder_layer_input")

  if prefetched_moe_rw is not None:
    wi_full, wo_full = prefetched_moe_rw
  elif early_moe_rw is not None:
    wi_full, wo_full = early_moe_rw
  else:
    if cfg.moe_ag_start_tie == "f2_x_mb1":
      moe_tie = x_mb1
    elif cfg.moe_ag_start_tie == "f2_base_mb0":
      moe_tie = base_mb0
    else:
      moe_tie = None
    wi_full = moe_kernels.gather_routed_weight_sc1(
        sub_params.moe.wi, sub_absmax.wi, cfg, start_after=moe_tie, is_wi=True
    )
    wo_full = moe_kernels.gather_routed_weight_sc1(
        sub_params.moe.wo, sub_absmax.wo, cfg, start_after=moe_tie, is_wi=False
    )

  if cfg.bwd_tie_mode == "symmetric_4phase":
    base_mb0, x_mb1 = ops.backward_barrier_tie(base_mb0, x_mb1)

  f2_pre_norm = ops.order_after(w_dense.pre_attn_norm, base_mb0)
  if cfg.f2_early_mb0_dispatch:
    if topk_w_mb0 is not None:
      f2_pre_norm = ops.order_after(f2_pre_norm, topk_w_mb0)
    if hasattr(disp_mb0.tokens_ep, "scale"):
      f2_pre_norm = ops.order_after(f2_pre_norm, disp_mb0.tokens_ep.scale)
  w_f2 = _replace_struct(w_dense, pre_attn_norm=f2_pre_norm)
  mb1_route_dep = (
      topk_w_mb0
      if (cfg.f2_light_router_ties and topk_w_mb0 is not None)
      else disp_mb0.tokens_ep
  )
  mb1_batch_off = x_mb0.shape[0]
  if cfg.route_before_shared_expert:
    h_mb1, normed_mb1 = _mixer_only(
        x_mb1,
        w_f2,
        pos_mb1,
        seg_mb1,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb1,
        mods=mods,
    )
    topk_w_mb1, topk_ids_mb1 = _call_route_only(
        layers,
        normed_mb1,
        w_dense.moe.router_gate,
        cfg,
        route_after=mb1_route_dep,
        sublayer_idx=sublayer_idx,
        batch_offset=mb1_batch_off,
        layer_idx=layer_idx,
        step=step,
    )
    disp_mb1 = layers.moe_dispatch_from_routed(
        normed_mb1, topk_w_mb1, topk_ids_mb1, cfg, order_tokens_on_router=True
    )
    normed_mb1_se = ops.order_after(normed_mb1, topk_w_mb1)
    base_mb1_out = layers.shared_expert_and_residual(
        h_mb1, normed_mb1_se, w_f2.moe
    )
  elif cfg.f2_light_router_ties:
    base_mb1_out, normed_mb1 = _mixer_and_shared(
        x_mb1,
        w_f2,
        pos_mb1,
        seg_mb1,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb1,
        mods=mods,
    )
    normed_mb1_r = ops.order_after(normed_mb1, base_mb1_out)
    topk_w_mb1, topk_ids_mb1 = _call_route_only(
        layers,
        normed_mb1_r,
        w_dense.moe.router_gate,
        cfg,
        route_after=mb1_route_dep,
        sublayer_idx=sublayer_idx,
        batch_offset=mb1_batch_off,
        layer_idx=layer_idx,
        step=step,
    )
    disp_mb1 = layers.moe_dispatch_from_routed(
        normed_mb1_r, topk_w_mb1, topk_ids_mb1, cfg, order_tokens_on_router=True
    )
  else:
    topk_w_mb1 = None
    base_mb1_out, normed_mb1 = _mixer_and_shared(
        x_mb1,
        w_f2,
        pos_mb1,
        seg_mb1,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb1,
        mods=mods,
    )
    normed_mb1_r = ops.order_after(normed_mb1, base_mb1_out)
    disp_mb1 = _call_route_and_dispatch(
        layers,
        normed_mb1_r,
        w_dense.moe.router_gate,
        cfg,
        route_after=disp_mb0.tokens_ep,
        sublayer_idx=sublayer_idx,
        batch_offset=mb1_batch_off,
        layer_idx=layer_idx,
        step=step,
    )

  # === Phase F3: mb0 MoE GMMs (TC) || mb1 EP AG (SC0) || w_dense(j+1) (SC1) ===
  if cfg.strict_f2_before_f3:
    f3_router_dep = (
        topk_w_mb1
        if (cfg.f2_light_router_ties and topk_w_mb1 is not None)
        else disp_mb1.topk_weights_ep
    )
    disp_mb0 = layers.order_dispatched_after(disp_mb0, base_mb1_out)
    disp_mb0 = layers.order_dispatched_after(disp_mb0, f3_router_dep)

  if cfg.bwd_tie_mode == "symmetric_4phase":
    base_mb1_out, tok_ep_mb0 = ops.backward_barrier_tie(
        base_mb1_out, disp_mb0.tokens_ep
    )
    disp_mb0 = layers.DispatchedMoeTokens(
        tokens_ep=tok_ep_mb0,
        topk_weights_ep=disp_mb0.topk_weights_ep,
        topk_ids_ep=disp_mb0.topk_ids_ep,
    )
  if prev_rw3_local is not None and bwd_rw3_in is not None:
    tok_ep_mb0_rw = moe_kernels.produce_prev_bwd_rw3(
        disp_mb0.tokens_ep, bwd_rw3_in, prev_rw3_local, cfg
    )
    disp_mb0 = layers.DispatchedMoeTokens(
        tokens_ep=tok_ep_mb0_rw,
        topk_weights_ep=disp_mb0.topk_weights_ep,
        topk_ids_ep=disp_mb0.topk_ids_ep,
    )

  wrap_pt_mb0 = cfg.moe_use_forked_3d_gmmv2 and cfg.moe_bwd_dy_fp8_e5m2
  wrap_pt_mb1 = wrap_pt_mb0 and (
      not is_superblock_last
      or _can_wrap_pt_across_superblock(cfg, int(x_mb0.shape[-1]), ops)
  )
  if use_external_bwd_rw:
    unperm_mb0, wi_chain, wo_chain, rw3_slot_mb0 = (
        layers.moe_expert_gmm_and_unsort(
            disp_mb0,
            wi_full,
            wo_full,
            cfg,
            return_weights=True,
            wrap_partial_fp8_bwd=wrap_pt_mb0,
            sublayer_idx=sublayer_idx,
            use_external_bwd_rw=True,
        )
    )
  else:
    rw3_slot_mb0 = None
    unperm_mb0, wi_chain, wo_chain = layers.moe_expert_gmm_and_unsort(
        disp_mb0,
        wi_full,
        wo_full,
        cfg,
        return_weights=True,
        wrap_partial_fp8_bwd=wrap_pt_mb0,
        sublayer_idx=sublayer_idx,
    )

  # === Deferred Next-Sublayer Dense Bwd Tie & SC1 `w_dense(j+1)` Prefetch ===
  next_params_tied = None
  w_dense_next = None
  if next_sub_params is not None:
    if next_prefetch_stop_grad:
      next_params_tied = next_sub_params
    else:
      unperm_mb0, next_params_tied = _tie_bwd_dense_only_params(
          unperm_mb0, next_sub_params, ops_mod=ops
      )
    wi_dep = _unwrap_dep_tensor(wi_full)
    wo_dep = _unwrap_dep_tensor(wo_full)
    prefetch_trigger = ops.order_after(
        ops.order_after(disp_mb0.tokens_ep, wi_dep), wo_dep
    )
    w_dense_next = _gather_dense_and_norms(
        next_params_tied,
        cfg,
        is_attn=next_is_attn,
        start_after=prefetch_trigger,
        mods=mods,
    )
    if next_prefetch_stop_grad:
      w_dense_next = jax.tree.map(
          lambda v: jax.lax.stop_gradient(v) if isinstance(v, jax.Array) else v,
          w_dense_next,
      )

  # === Phase F4: `mb1` MoE GMMs (TC) || `mb0` EP Combine (SC0) ===
  tok_ep_mb1_f4 = ops.order_after(disp_mb1.tokens_ep, unperm_mb0)
  if cfg.bwd_tie_mode == "symmetric_4phase":
    unperm_mb0, tok_ep_mb1_f4 = ops.backward_barrier_tie(
        unperm_mb0, tok_ep_mb1_f4
    )

  if cfg.strict_f2_before_f3:
    combined_mb0 = ops.ep_token_combine(unperm_mb0, cfg)
  else:
    combined_mb0 = None
    x_mb0_out = layers.moe_combine_and_add_base(unperm_mb0, base_mb0, cfg)

  disp_mb1_f4 = layers.DispatchedMoeTokens(
      tokens_ep=tok_ep_mb1_f4,
      topk_weights_ep=disp_mb1.topk_weights_ep,
      topk_ids_ep=disp_mb1.topk_ids_ep,
  )
  if use_external_bwd_rw:
    if cfg.tie_f4_before_wo:
      unperm_mb1_out, _, rw3_slot_mb1 = layers.moe_expert_gmm_and_unsort(
          disp_mb1_f4,
          wi_chain,
          wo_chain,
          cfg,
          return_weights=False,
          forward_tie=unperm_mb0,
          wrap_partial_fp8_bwd=wrap_pt_mb1,
          sublayer_idx=sublayer_idx,
          use_external_bwd_rw=True,
      )
    else:
      unperm_mb1_out, rw3_slot_mb1 = layers.moe_expert_gmm_and_unsort(
          disp_mb1_f4,
          wi_chain,
          wo_chain,
          cfg,
          return_weights=False,
          wrap_partial_fp8_bwd=wrap_pt_mb1,
          sublayer_idx=sublayer_idx,
          use_external_bwd_rw=True,
      )
    assert rw3_slot_mb0 is not None
    bwd_rw3_out = ops.merge_bwd_rw3_slots(rw3_slot_mb0, rw3_slot_mb1)
  elif cfg.tie_f4_before_wo:
    bwd_rw3_out = None
    unperm_mb1_out, _ = layers.moe_expert_gmm_and_unsort(
        disp_mb1_f4,
        wi_chain,
        wo_chain,
        cfg,
        return_weights=False,
        forward_tie=unperm_mb0,
        wrap_partial_fp8_bwd=wrap_pt_mb1,
        sublayer_idx=sublayer_idx,
    )
  else:
    bwd_rw3_out = None
    unperm_mb1_out = layers.moe_expert_gmm_and_unsort(
        disp_mb1_f4,
        wi_chain,
        wo_chain,
        cfg,
        return_weights=False,
        wrap_partial_fp8_bwd=wrap_pt_mb1,
        sublayer_idx=sublayer_idx,
    )

  if cfg.strict_f2_before_f3:
    assert combined_mb0 is not None
    base_mb0_f4 = ops.order_after(base_mb0, unperm_mb1_out)
    x_mb0_out = base_mb0_f4 + combined_mb0.reshape(base_mb0.shape)

  carry_out = BatchSplitCarry(
      x_mb0=x_mb0_out,
      unperm_mb1=unperm_mb1_out,
      base_mb1=base_mb1_out,
      bwd_rw3_prefetch=bwd_rw3_out,
      prefetched_w_dense_0=w_dense_next if next_prefetch_stop_grad else None,
  )
  entry_pair = (entry_x_mb0, entry_x_mb1_raw)
  if next_sub_params is not None and not next_prefetch_stop_grad:
    if return_entry_pair:
      return carry_out, next_params_tied, w_dense_next, entry_pair
    return carry_out, next_params_tied, w_dense_next
  if return_entry_pair:
    return carry_out, entry_pair
  return carry_out


def run_superblock_batchsplit(
    carry: BatchSplitCarry,
    sb_params: Any,
    sb_absmax: SuperblockAbsmax,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: Qwen38Config,
    *,
    sb_idx: jax.Array | int = 0,
    step: jax.Array | int = 0,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    prev_rw3_local: (
        tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None
    ) = None,
    use_attn3_external_bwd_rw: bool = False,
    next_gdn0_params: Any = None,
    return_sublayer_entries: bool = False,
    modules: ScheduleModules | None = None,
) -> (
    BatchSplitCarry
    | tuple[BatchSplitCarry, tuple[tuple[jax.Array, jax.Array], ...]]
):
  """Flat-unrolls `[GDN_0, GDN_1, GDN_2, FullAttn_3]` (`whiles=0` inside outer scan)."""
  _assert_core_minor_config(cfg)
  mods = _resolve_modules(modules)
  moe_kernels = mods.moe_kernels
  base_layer_idx = sb_idx * 4

  def _unpack_entry(res):
    if return_sublayer_entries:
      return res[:-1], res[-1]
    return res, None

  if cfg.fsdp_pipeline_mode == "full_prefetch_defer_rs":
    use_loop_dense_pipe = (
        next_gdn0_params is not None and carry.prefetched_w_dense_0 is not None
    )
    if use_loop_dense_pipe:
      w_dense_0 = _dense_loop_boundary_bridge(
          sb_params.gdn_0, carry.prefetched_w_dense_0, cfg, mods=mods
      )
    else:
      w_dense_0 = _gather_dense_and_norms(
          sb_params.gdn_0, cfg, is_attn=False, start_after=None, mods=mods
      )
    res0, e0 = _unpack_entry(
        _run_sublayer_batchsplit(
            carry,
            sb_params.gdn_0,
            sb_absmax.gdn_0,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=False,
            sublayer_idx=0,
            layer_idx=base_layer_idx + 0,
            step=step,
            prefetched_w_dense=w_dense_0,
            moe_rw_start_after=w_dense_0.out_proj,
            next_sub_params=sb_params.gdn_1,
            next_is_attn=False,
            skip_own_bwd_tie=False,
            prev_rw3_local=prev_rw3_local,
            return_entry_pair=return_sublayer_entries,
            mods=mods,
        )
    )
    carry, p_gdn_1, w_dense_1 = res0
    rw_tie_1 = (
        w_dense_1.out_proj if cfg.moe_rw_prefetch_in_f3 else carry.unperm_mb1
    )
    res1, e1 = _unpack_entry(
        _run_sublayer_batchsplit(
            carry,
            p_gdn_1,
            sb_absmax.gdn_1,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=False,
            sublayer_idx=1,
            layer_idx=base_layer_idx + 1,
            step=step,
            prefetched_w_dense=w_dense_1,
            moe_rw_start_after=rw_tie_1,
            next_sub_params=sb_params.gdn_2,
            next_is_attn=False,
            skip_own_bwd_tie=False,
            return_entry_pair=return_sublayer_entries,
            mods=mods,
        )
    )
    carry, p_gdn_2, w_dense_2 = res1
    rw_tie_2 = (
        w_dense_2.out_proj if cfg.moe_rw_prefetch_in_f3 else carry.unperm_mb1
    )
    res2, e2 = _unpack_entry(
        _run_sublayer_batchsplit(
            carry,
            p_gdn_2,
            sb_absmax.gdn_2,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=False,
            sublayer_idx=2,
            layer_idx=base_layer_idx + 2,
            step=step,
            prefetched_w_dense=w_dense_2,
            moe_rw_start_after=rw_tie_2,
            next_sub_params=sb_params.attn_3,
            next_is_attn=True,
            skip_own_bwd_tie=False,
            return_entry_pair=return_sublayer_entries,
            mods=mods,
        )
    )
    carry, p_attn_3, w_dense_3 = res2
    rw_tie_3 = (
        w_dense_3.out_proj if cfg.moe_rw_prefetch_in_f3 else carry.unperm_mb1
    )
    res3 = _run_sublayer_batchsplit(
        carry,
        p_attn_3,
        sb_absmax.attn_3,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn=True,
        sublayer_idx=3,
        layer_idx=base_layer_idx + 3,
        step=step,
        prefetched_w_dense=w_dense_3,
        moe_rw_start_after=rw_tie_3,
        next_sub_params=next_gdn0_params if use_loop_dense_pipe else None,
        next_is_attn=False,
        next_prefetch_stop_grad=use_loop_dense_pipe,
        skip_own_bwd_tie=False,
        is_superblock_last=True,
        rope_mb0=rope_mb0,
        rope_mb1=rope_mb1,
        use_external_bwd_rw=use_attn3_external_bwd_rw,
        return_entry_pair=return_sublayer_entries,
        mods=mods,
    )
    if return_sublayer_entries:
      carry, e3 = res3
      return carry, (e0, e1, e2, e3)
    return res3

  sublayer_specs = (
      ("gdn_0", False, 0),
      ("gdn_1", False, 1),
      ("gdn_2", False, 2),
      ("attn_3", True, 3),
  )
  entries: list[tuple[jax.Array, jax.Array]] = []
  use_sub_rw = cfg.fsdp_pipeline_mode == "sublayer_prefetch"

  def _gather_rw(sub_p: Any, sub_a: SublayerAbsmax):
    return (
        moe_kernels.gather_routed_weight_sc1(
            sub_p.moe.wi, sub_a.wi, cfg, is_wi=True
        ),
        moe_kernels.gather_routed_weight_sc1(
            sub_p.moe.wo, sub_a.wo, cfg, is_wi=False
        ),
    )

  pref_rws = [None, None, None, None]
  if use_sub_rw:
    pref_rws[0] = _gather_rw(sb_params.gdn_0, sb_absmax.gdn_0)
  for k, (attr, is_attn_k, idx_k) in enumerate(sublayer_specs):
    if use_sub_rw and k + 1 < 4:
      n_attr = sublayer_specs[k + 1][0]
      pref_rws[k + 1] = _gather_rw(
          getattr(sb_params, n_attr), getattr(sb_absmax, n_attr)
      )
    res_k = _run_sublayer_batchsplit(
        carry,
        getattr(sb_params, attr),
        getattr(sb_absmax, attr),
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn=is_attn_k,
        sublayer_idx=idx_k,
        layer_idx=base_layer_idx + idx_k,
        step=step,
        prefetched_moe_rw=pref_rws[k],
        is_superblock_last=(idx_k == 3),
        rope_mb0=rope_mb0 if is_attn_k else None,
        rope_mb1=rope_mb1 if is_attn_k else None,
        return_entry_pair=return_sublayer_entries,
        mods=mods,
    )
    if return_sublayer_entries:
      carry, ek = res_k
      entries.append(ek)
    else:
      carry = res_k
  if return_sublayer_entries:
    return carry, tuple(entries)
  return carry


def _make_dummy_unperm_mb1(
    unperm_shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    is_partial_tok: bool,
    ops_mod: Any,
) -> Any:
  if not is_partial_tok:
    return jnp.zeros(unperm_shape, dtype=dtype)
  return ops_mod.MoePartialTokens(
      y_ep=jnp.zeros(unperm_shape, dtype=dtype),
      dy_codes=jnp.zeros(unperm_shape, dtype=jnp.float8_e5m2),
      dy_scale=jnp.zeros((1, 1), dtype=jnp.float32),
  )


def _can_wrap_pt_across_superblock(
    cfg: QwenConfig, emb_dim: int, ops_mod: Any
) -> bool:
  """Returns True when cross-superblock `unperm_mb1` carry should wrap `MoePartialTokens`."""
  pt_cls = getattr(ops_mod, "MoePartialTokens", None)
  pt_has_yep = pt_cls is not None and (
      not hasattr(pt_cls, "_fields") or "y_ep" in pt_cls._fields
  )
  return bool(
      getattr(cfg, "wrap_pt_across_superblock", True)
      and cfg.use_pallas_kernels
      and cfg.ep_size > 1
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (emb_dim % 128 == 0)
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_weights
      and cfg.moe_fp8_gmm_compute
      and cfg.moe_use_forked_3d_gmmv2
      and cfg.moe_bwd_dy_fp8_e5m2
      and pt_has_yep
  )


def maybe_wrap_init_unperm_partial_tokens(
    init_unperm: Any,
    cfg: QwenConfig,
    *,
    emb_dim: int | None = None,
    modules: ScheduleModules | None = None,
) -> Any:
  """Wraps `init_unperm` in zero-gradient `MoePartialTokens` when cross-SB FP8 `dy` is active."""
  mods = _resolve_modules(modules)
  ops_mod = mods.ops
  pt_cls = getattr(ops_mod, "MoePartialTokens", None)
  if pt_cls is not None and isinstance(init_unperm, pt_cls):
    return init_unperm
  eff_emb_dim = int(cfg.emb_dim if emb_dim is None else emb_dim)
  if not _can_wrap_pt_across_superblock(cfg, eff_emb_dim, ops_mod):
    return init_unperm
  return ops_mod.MoePartialTokens(
      y_ep=init_unperm,
      dy_codes=jnp.zeros(init_unperm.shape, dtype=jnp.float8_e5m2),
      dy_scale=jnp.zeros((1, 1), dtype=jnp.float32),
  )


def run_superblock_batchsplit_scan(
    init_x_mb0: jax.Array,
    init_unperm_mb1: jax.Array,
    init_base_mb1: jax.Array,
    blocks_local: Any,
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
    modules: ScheduleModules | None = None,
) -> tuple[jax.Array, Any, jax.Array]:
  """Executes `lax.scan` over superblocks keeping `prefetched_w_dense_0` out of HBM residuals.

  Uses `@jax.custom_vjp` around the superblock scan so the forward scan runs
  `full_prefetch_defer_rs` strictly once (`F3/F4` -> `next_gdn0` `SC1` prefetch)
  while saving ONLY small `(x_mb0, x_mb1)` entry activations (`0` stacked
  `prefetched_w_dense_0` `[num_superblocks, ...]` buffers and `0` `[T_ep, D]`
  `unperm_mb1` buffers across the scan loop). In backward, when
  `save_decoder_input_hbm=True`, sublayers `3 -> 2 -> 1 -> 0` replay one
  sublayer at a time with `SC1` dense weights re-gathered cleanly.

  Args:
    init_x_mb0: Microbatch 0 entry activation `[B_local/2, S, D]`.
    init_unperm_mb1: Initial microbatch 1 unpermuted EP buffer.
    init_base_mb1: Microbatch 1 base activation `[B_local/2, S, D]`.
    blocks_local: Stacked superblock parameter PyTree (`[num_sb, ...]`).
    hoisted_absmax: Stacked per-superblock hoisted MoE absmax PyTree.
    pos_mb0: Microbatch 0 token positions `[B_local/2, S]`.
    pos_mb1: Microbatch 1 token positions `[B_local/2, S]`.
    seg_mb0: Microbatch 0 segment IDs `[B_local/2, S]`.
    seg_mb1: Microbatch 1 segment IDs `[B_local/2, S]`.
    cfg: Model and schedule configuration.
    rope_mb0: Optional precomputed `(cos, sin)` tables for microbatch 0.
    rope_mb1: Optional precomputed `(cos, sin)` tables for microbatch 1.
    modules: Optional pre-resolved sibling module bundle.

  Returns:
    Tuple `(final_x_mb0, final_unperm_mb1, final_base_mb1)` matching the final
    `BatchSplitCarry` fields needed for post-scan MoE combine.
  """
  _assert_core_minor_config(cfg)
  mods = _resolve_modules(modules)
  ops = mods.ops
  pt_cls = getattr(ops, "MoePartialTokens", None)
  already_pt = pt_cls is not None and isinstance(init_unperm_mb1, pt_cls)
  unperm_shape = (
      init_unperm_mb1.y_ep.shape if already_pt else init_unperm_mb1.shape
  )
  act_dtype = init_base_mb1.dtype
  emb_dim = int(init_base_mb1.shape[-1])
  pt_has_yep = pt_cls is not None and (
      not hasattr(pt_cls, "_fields") or "y_ep" in pt_cls._fields
  )
  wrap_pt_mb0 = bool(
      cfg.use_pallas_kernels
      and cfg.ep_size > 1
      and cfg.moe_use_tc_ragged
      and cfg.moe_3d_ep_tokens
      and (emb_dim % 128 == 0)
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_weights
      and cfg.moe_fp8_gmm_compute
      and cfg.moe_use_forked_3d_gmmv2
      and cfg.moe_bwd_dy_fp8_e5m2
      and pt_has_yep
  )
  wrap_pt_sb = _can_wrap_pt_across_superblock(cfg, emb_dim, ops)
  save_all_sublayer_inputs = bool(cfg.save_decoder_input_hbm)

  step_in = jnp.asarray(step, dtype=jnp.int32)

  @jax.custom_vjp
  def _scan_impl(x0_in, u1_in, b1_in, blks, h_absmax, step_arr):
    sb_indices = jnp.arange(cfg.num_superblocks, dtype=jnp.int32)
    prefetched_0, next_gdn0s = prepare_loop_boundary_dense_pipeline(
        blks, cfg, modules=mods
    )
    init_u1 = u1_in
    if wrap_pt_sb:
      init_u1 = maybe_wrap_init_unperm_partial_tokens(
          u1_in, cfg, emb_dim=emb_dim, modules=mods
      )
    init_carry = BatchSplitCarry(
        x_mb0=x0_in,
        unperm_mb1=init_u1,
        base_mb1=b1_in,
        bwd_rw3_prefetch=None,
        prefetched_w_dense_0=prefetched_0,
    )

    def _fwd_body(carry, sb_tuple):
      if next_gdn0s is not None:
        sb_p, sb_a, next_gdn0, sb_i = sb_tuple
      else:
        sb_p, sb_a, sb_i = sb_tuple
        next_gdn0 = None
      carry_out = run_superblock_batchsplit(
          carry,
          sb_p,
          sb_a,
          pos_mb0,
          pos_mb1,
          seg_mb0,
          seg_mb1,
          cfg,
          sb_idx=sb_i,
          step=step_arr,
          rope_mb0=rope_mb0,
          rope_mb1=rope_mb1,
          next_gdn0_params=next_gdn0,
          return_sublayer_entries=False,
          modules=mods,
      )
      return carry_out, None

    xs = (
        (blks, h_absmax, next_gdn0s, sb_indices)
        if next_gdn0s is not None
        else (blks, h_absmax, sb_indices)
    )
    final_c, _ = jax.lax.scan(_fwd_body, init_carry, xs)
    return final_c.x_mb0, final_c.unperm_mb1, final_c.base_mb1

  def _scan_fwd(x0_in, u1_in, b1_in, blks, h_absmax, step_arr):
    sb_indices = jnp.arange(cfg.num_superblocks, dtype=jnp.int32)
    prefetched_0, next_gdn0s = prepare_loop_boundary_dense_pipeline(
        blks, cfg, modules=mods
    )
    init_u1 = u1_in
    if wrap_pt_sb:
      init_u1 = maybe_wrap_init_unperm_partial_tokens(
          u1_in, cfg, emb_dim=emb_dim, modules=mods
      )
    init_carry = BatchSplitCarry(
        x_mb0=x0_in,
        unperm_mb1=init_u1,
        base_mb1=b1_in,
        bwd_rw3_prefetch=None,
        prefetched_w_dense_0=prefetched_0,
    )

    def _fwd_body(carry, sb_tuple):
      if next_gdn0s is not None:
        sb_p, sb_a, next_gdn0, sb_i = sb_tuple
      else:
        sb_p, sb_a, sb_i = sb_tuple
        next_gdn0 = None
      carry_out, entries = run_superblock_batchsplit(
          carry,
          sb_p,
          sb_a,
          pos_mb0,
          pos_mb1,
          seg_mb0,
          seg_mb1,
          cfg,
          sb_idx=sb_i,
          step=step_arr,
          rope_mb0=rope_mb0,
          rope_mb1=rope_mb1,
          next_gdn0_params=next_gdn0,
          return_sublayer_entries=True,
          modules=mods,
      )
      saved_y = entries if save_all_sublayer_inputs else entries[0]
      return carry_out, saved_y

    xs = (
        (blks, h_absmax, next_gdn0s, sb_indices)
        if next_gdn0s is not None
        else (blks, h_absmax, sb_indices)
    )
    final_c, scan_y = jax.lax.scan(_fwd_body, init_carry, xs)
    out = (final_c.x_mb0, final_c.unperm_mb1, final_c.base_mb1)
    return out, (scan_y, blks, h_absmax, step_arr)

  def _sublayer_bwd_step(
      g_carry_out: tuple[jax.Array, Any, jax.Array],
      x0_k: jax.Array,
      x1_k: jax.Array,
      sub_p: Any,
      sub_a: SublayerAbsmax,
      *,
      is_attn: bool,
      sublayer_idx: int,
      layer_idx: jax.Array | int,
      step_arr: jax.Array,
      is_superblock_last: bool,
      entry_is_partial_tok: bool,
      moe_rw_start_after: Any = None,
  ) -> tuple[tuple[jax.Array, Any, jax.Array], Any]:
    u_dummy = _make_dummy_unperm_mb1(
        unperm_shape, act_dtype, entry_is_partial_tok, ops
    )
    b_dummy = jnp.zeros_like(x1_k)

    def _sub_fn(x0_in, u_in, b_in, p_in):
      c_in = BatchSplitCarry(
          x_mb0=x0_in,
          unperm_mb1=u_in,
          base_mb1=b_in,
          bwd_rw3_prefetch=None,
          prefetched_w_dense_0=None,
      )
      c_out = _run_sublayer_batchsplit(
          c_in,
          p_in,
          sub_a,
          pos_mb0,
          pos_mb1,
          seg_mb0,
          seg_mb1,
          cfg,
          is_attn=is_attn,
          sublayer_idx=sublayer_idx,
          layer_idx=layer_idx,
          step=step_arr,
          moe_rw_start_after=moe_rw_start_after,
          is_superblock_last=is_superblock_last,
          rope_mb0=rope_mb0 if is_attn else None,
          rope_mb1=rope_mb1 if is_attn else None,
          prefetched_x_mb1=x1_k,
          mods=mods,
      )
      return c_out.x_mb0, c_out.unperm_mb1, c_out.base_mb1

    _, vjp_fn = jax.vjp(_sub_fn, x0_k, u_dummy, b_dummy, sub_p)
    g_x0, g_u, g_b, g_p = vjp_fn(g_carry_out)
    return (g_x0, g_u, g_b), g_p

  def _scan_bwd(res, g_outputs):
    scan_y, blks, h_absmax, step_arr = res
    sb_indices = jnp.arange(cfg.num_superblocks, dtype=jnp.int32)
    tie_mode = getattr(cfg, "remat_bwd_tie_mode", "dx_mb1")

    def _bwd_body(g_carry, xs_i):
      entry_i, sb_p, sb_a, sb_i = xs_i
      base_layer_idx = sb_i * 4
      if save_all_sublayer_inputs:
        sub_specs = (
            (3, "attn_3", True, True, wrap_pt_mb0),
            (2, "gdn_2", False, False, wrap_pt_mb0),
            (1, "gdn_1", False, False, wrap_pt_mb0),
            (0, "gdn_0", False, False, wrap_pt_sb),
        )
        g_subs: dict[str, Any] = {}
        g_cur = g_carry
        g_prev_dense: Any = None
        dense_depth = int(getattr(cfg, "remat_bwd_tie_prev_dense_depth", 0))
        wi_prefetch_prev = bool(
            getattr(cfg, "remat_bwd_wi_prefetch_prev_sublayer", False)
        )
        dense_history: list[Any] = []
        prev_x0_k: Any = None
        for step_idx, (idx_k, attr_k, is_attn_k, is_last_k, pt_k) in enumerate(
            sub_specs
        ):
          x0_k, x1_k = entry_i[idx_k]
          moe_rw_sa: Any = None
          if tie_mode == "dense_rs":
            if g_prev_dense is not None:
              # Keep sublayer backward execution sequential via `g_cur[0]`.
              g_cur = (
                  ops.order_after(g_cur[0], g_prev_dense),
                  g_cur[1],
                  g_cur[2],
              )
          elif tie_mode == "dx_mb1":
            if step_idx > 0:
              dep = jax.lax.stop_gradient(g_cur[2])
              x0_k = ops.order_after(x0_k, dep)
              x1_k = ops.order_after(x1_k, dep)
              moe_rw_sa = (
                  jax.lax.stop_gradient(prev_x0_k)
                  if (wi_prefetch_prev and prev_x0_k is not None)
                  else dep
              )
            else:
              moe_rw_sa = x0_k
            if dense_depth > 0 and len(dense_history) >= dense_depth:
              dense_dep = jax.lax.stop_gradient(dense_history[-dense_depth])
              x0_k = ops.order_after(x0_k, dense_dep)
              x1_k = ops.order_after(x1_k, dense_dep)
          elif tie_mode == "none":
            moe_rw_sa = x0_k
          prev_x0_k = x0_k
          g_cur, g_p_k = _sublayer_bwd_step(
              g_cur,
              x0_k,
              x1_k,
              getattr(sb_p, attr_k),
              getattr(sb_a, attr_k),
              is_attn=is_attn_k,
              sublayer_idx=idx_k,
              layer_idx=base_layer_idx + idx_k,
              step_arr=step_arr,
              is_superblock_last=is_last_k,
              entry_is_partial_tok=pt_k,
              moe_rw_start_after=moe_rw_sa,
          )
          g_subs[attr_k] = g_p_k
          g_prev_dense = _replace_struct(
              g_p_k, moe=_replace_struct(g_p_k.moe, wi=None, wo=None)
          )
          if tie_mode == "dx_mb1" and dense_depth > 0:
            dense_history.append(_dense_rs_grads_only(g_p_k))
            if (
                step_idx == len(sub_specs) - 1
                and len(dense_history) >= dense_depth
            ):
              g_cur = (
                  ops.order_after(
                      g_cur[0],
                      jax.lax.stop_gradient(dense_history[-dense_depth]),
                  ),
                  g_cur[1],
                  g_cur[2],
              )
        g_sb_p = type(sb_p)(**g_subs)
        return g_cur, g_sb_p

      # Entry-only replay (`save_decoder_input_hbm=False`): 2.88 GiB scan tape.
      x0_0, x1_0 = entry_i
      u_dummy_0 = _make_dummy_unperm_mb1(
          unperm_shape, act_dtype, wrap_pt_sb, ops
      )
      b_dummy_0 = jnp.zeros_like(x1_0)

      def _sb_replay_fn(x0_in, u_in, b_in, p_in):
        sub0_c, p_gdn1, w_dense_1 = _run_sublayer_batchsplit(
            BatchSplitCarry(
                x_mb0=x0_in,
                unperm_mb1=u_in,
                base_mb1=b_in,
                bwd_rw3_prefetch=None,
                prefetched_w_dense_0=None,
            ),
            p_in.gdn_0,
            sb_a.gdn_0,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=False,
            sublayer_idx=0,
            layer_idx=base_layer_idx + 0,
            step=step_arr,
            next_sub_params=p_in.gdn_1,
            next_is_attn=False,
            prefetched_x_mb1=x1_0,
            mods=mods,
        )
        # Replay remaining 3 sublayers starting cleanly from sublayer 0 output.
        c = sub0_c
        c, p_gdn2, w_dense_2 = _run_sublayer_batchsplit(
            c,
            p_gdn1,
            sb_a.gdn_1,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=False,
            sublayer_idx=1,
            layer_idx=base_layer_idx + 1,
            step=step_arr,
            prefetched_w_dense=w_dense_1,
            next_sub_params=p_in.gdn_2,
            next_is_attn=False,
            mods=mods,
        )
        c, p_attn3, w_dense_3 = _run_sublayer_batchsplit(
            c,
            p_gdn2,
            sb_a.gdn_2,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=False,
            sublayer_idx=2,
            layer_idx=base_layer_idx + 2,
            step=step_arr,
            prefetched_w_dense=w_dense_2,
            next_sub_params=p_in.attn_3,
            next_is_attn=True,
            mods=mods,
        )
        c = _run_sublayer_batchsplit(
            c,
            p_attn3,
            sb_a.attn_3,
            pos_mb0,
            pos_mb1,
            seg_mb0,
            seg_mb1,
            cfg,
            is_attn=True,
            sublayer_idx=3,
            layer_idx=base_layer_idx + 3,
            step=step_arr,
            prefetched_w_dense=w_dense_3,
            is_superblock_last=True,
            rope_mb0=rope_mb0,
            rope_mb1=rope_mb1,
            mods=mods,
        )
        return c.x_mb0, c.unperm_mb1, c.base_mb1

      _, vjp_fn = jax.vjp(_sb_replay_fn, x0_0, u_dummy_0, b_dummy_0, sb_p)
      g_x0, g_u, g_b, g_sb_p = vjp_fn(g_carry)
      return (g_x0, g_u, g_b), g_sb_p

    g_init, g_blks = jax.lax.scan(
        _bwd_body,
        g_outputs,
        (scan_y, blks, h_absmax, sb_indices),
        reverse=True,
    )
    g_h_absmax = jax.tree.map(
        lambda v: None if v is None else jnp.zeros_like(v), h_absmax
    )
    g_u_init = g_init[1]
    if (
        wrap_pt_sb
        and not already_pt
        and pt_cls is not None
        and isinstance(g_init[1], pt_cls)
    ):
      g_u_init = g_init[1].y_ep
    return g_init[0], g_u_init, g_init[2], g_blks, g_h_absmax, None

  _scan_impl.defvjp(_scan_fwd, _scan_bwd)
  return _scan_impl(
      init_x_mb0,
      init_unperm_mb1,
      init_base_mb1,
      blocks_local,
      hoisted_absmax,
      step_in,
  )
