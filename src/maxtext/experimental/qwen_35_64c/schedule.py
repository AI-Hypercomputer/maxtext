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

"""Pure-JAX Unsplit & Dual-SparseCore (SC0 EP + SC1 FSDP) Batch-Split Schedules."""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.experimental.qwen_35_64c import layers
from maxtext.experimental.qwen_35_64c import ops
from maxtext.experimental.qwen_35_64c.config import Qwen35Config
from maxtext.experimental.qwen_35_64c.params import (
    AttnSublayerParams,
    GdnSublayerParams,
    SuperblockParams,
)


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
  unperm_mb1: jax.Array | ops.MoePartialTokens  # [T_ep, D] or 3D [T_ep, D//128, 128]
  base_mb1: jax.Array  # [B_mb, S, D]
  bwd_rw3_prefetch: ops.BwdRoutedWeightPrefetch | None = None


# ---------------------------------------------------------------------------
# Reference 1-Microbatch ("unsplit") Sub-Layer & Superblock Schedule
# ---------------------------------------------------------------------------


def _run_sublayer_unsplit(
    x: jax.Array,
    sub_params: GdnSublayerParams | AttnSublayerParams,
    sub_absmax: SublayerAbsmax,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: Qwen35Config,
    *,
    is_attn: bool,
    sublayer_idx: int = 0,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Executes a single sub-layer without microbatch splitting."""
  x = checkpoint_name(x, "decoder_layer_input")
  if is_attn:
    assert isinstance(sub_params, AttnSublayerParams)
    w = layers.gather_attn_sublayer_weights(sub_params, sub_absmax.wi, sub_absmax.wo, cfg)
    h = layers.full_attn_mixer(x, w, positions, segment_ids, cfg, rope_cos_sin=rope_cos_sin)
  else:
    assert isinstance(sub_params, GdnSublayerParams)
    w = layers.gather_gdn_sublayer_weights(sub_params, sub_absmax.wi, sub_absmax.wo, cfg)
    h = layers.gdn_mixer(x, w, segment_ids, cfg)

  normed = ops.rms_norm_zero_centered(h, w.post_attn_norm, cfg.norm_eps)
  base = layers.shared_expert_and_residual(h, normed, w.moe)
  disp, aux_stats = layers.moe_local_route_and_dispatch_with_aux(
      normed,
      w.moe.router_gate,
      cfg,
      segment_ids=segment_ids,
      collect_aux=(cfg.router_aux_loss_coef > 0.0),
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
  return layers.moe_combine_and_add_base(unperm, base, cfg), aux_stats


def run_superblock_unsplit(
    x: jax.Array,
    sb_params: SuperblockParams,
    sb_absmax: SuperblockAbsmax,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: Qwen35Config,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, layers.MoeAuxStats | None]:
  """Executes `[GDN_0, GDN_1, GDN_2, FullAttn_3]` sequentially on full local batch."""
  x, aux_0 = _run_sublayer_unsplit(
      x,
      sb_params.gdn_0,
      sb_absmax.gdn_0,
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=0,
  )
  x, aux_1 = _run_sublayer_unsplit(
      x,
      sb_params.gdn_1,
      sb_absmax.gdn_1,
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=1,
  )
  x, aux_2 = _run_sublayer_unsplit(
      x,
      sb_params.gdn_2,
      sb_absmax.gdn_2,
      positions,
      segment_ids,
      cfg,
      is_attn=False,
      sublayer_idx=2,
  )
  x, aux_3 = _run_sublayer_unsplit(
      x,
      sb_params.attn_3,
      sb_absmax.attn_3,
      positions,
      segment_ids,
      cfg,
      is_attn=True,
      sublayer_idx=3,
      rope_cos_sin=rope_cos_sin,
  )
  return x, layers.stack_moe_aux_stats((aux_0, aux_1, aux_2, aux_3))


# ---------------------------------------------------------------------------
# Dual-SparseCore (SC0 EP + SC1 FSDP) 2-Microbatch Batch-Split Schedule
# ---------------------------------------------------------------------------


def _tie_bwd_sharded_params(
    x_in: tuple[jax.Array, jax.Array],
    sub_params: GdnSublayerParams | AttnSublayerParams,
    cfg: Qwen35Config,
):
  """Applies zero-residual post-RS backward barrier on sharded sub-layer weights.

  Forces sub-layer `j`'s SC1 FSDP weight-gradient `psum_scatter` to overlap
  `B_F2 + B_F1` and retire before sub-layer `j - 1`'s backward begins,
  immediately freeing `~6 GiB` of unreduced `[64, K, N]` `drhs` buffers!
  """
  if cfg.bwd_tie_mode == "none":
    return x_in, sub_params

  if cfg.bwd_tie_mode in ("moe_xmb0_only", "symmetric_4phase"):
    # Tie ONLY `x_mb0` (NOT `x_mb1`!) to `(wi, wo)` post-RS cotangents so `dx_mb1`
    # immediately launches cross-layer SC0 EP Combine-Bwd during `B1` (`mb0` Mixer bwd)!
    x_mb0_in, x_mb1_in = x_in
    x_mb0_t, (wi_t, wo_t) = ops.backward_barrier_tie(x_mb0_in, (sub_params.moe.wi, sub_params.moe.wo))
    moe_updated = type(sub_params.moe)(
        router_gate=sub_params.moe.router_gate,
        wi=wi_t,
        wo=wo_t,
        shared_wi_0=sub_params.moe.shared_wi_0,
        shared_wi_1=sub_params.moe.shared_wi_1,
        shared_wo=sub_params.moe.shared_wo,
        shared_gate=sub_params.moe.shared_gate,
    )
    return (x_mb0_t, x_mb1_in), dataclasses_replace(sub_params, moe=moe_updated)

  if cfg.bwd_tie_mode == "moe_only":
    x_tied, (wi_t, wo_t) = ops.backward_barrier_tie(x_in, (sub_params.moe.wi, sub_params.moe.wo))
    new_moe = SuperblockParams  # unused alias guard
    del new_moe
    moe_updated = type(sub_params.moe)(
        router_gate=sub_params.moe.router_gate,
        wi=wi_t,
        wo=wo_t,
        shared_wi_0=sub_params.moe.shared_wi_0,
        shared_wi_1=sub_params.moe.shared_wi_1,
        shared_wo=sub_params.moe.shared_wo,
        shared_gate=sub_params.moe.shared_gate,
    )
    return x_tied, dataclasses_replace(sub_params, moe=moe_updated)

  # "all4": tie all sharded MoE + dense weights of this sub-layer!
  return ops.backward_barrier_tie(x_in, sub_params)


def _tie_bwd_dense_only_params(
    anchor: jax.Array,
    sub_params: GdnSublayerParams | AttnSublayerParams,
    cfg: Qwen35Config,
):
  """Ties ONLY `fsdp_expert` dense sharded grads (~0.94 GiB) to `anchor`.

  Leaves `(wi, wo)` (`~6 GiB` unreduced `drhs`) to retire via sub-layer `j+1`'s
  own `x_mb0(j+1)` tie (`16 ms` window across `B3_dlhs + B2 + B1`, zero extra
  HBM!), while deferring `fsdp_expert` `dw_dense(j+1)` retirement across `B4(j)`
  (`SC1` idle!) to `B3(j)`. Masks out replicated and `fsdp_only` parameters so
  their TensorCore ICI `all-reduce`s remain leaf outputs and never block `dx`!

  Args:
    anchor: Activation tensor whose backward cotangent anchors the barrier.
    sub_params: Next sub-layer parameter PyTree (`GDN` or `FullAttn`).
    cfg: Model and schedule configuration.

  Returns:
    Tuple `(anchor_tied, sub_restored)` with `fsdp_expert` leaves tied.
  """
  is_attn = isinstance(sub_params, AttnSublayerParams)

  def _keep_fe(leaf: jax.Array, name: str) -> jax.Array | None:
    return leaf if cfg.dense_param_sharding_kind(name) == "fsdp_expert" else None

  p_moe = sub_params.moe
  moe_for_tie = dataclasses_replace(
      p_moe,
      router_gate=_keep_fe(p_moe.router_gate, "router_gate"),
      wi=None,
      wo=None,
      shared_wi_0=_keep_fe(p_moe.shared_wi_0, "shared_wi"),
      shared_wi_1=_keep_fe(p_moe.shared_wi_1, "shared_wi"),
      shared_wo=_keep_fe(p_moe.shared_wo, "shared_wo"),
      shared_gate=_keep_fe(p_moe.shared_gate, "shared_gate"),
  )
  if is_attn:
    sub_for_tie = dataclasses_replace(
        sub_params,
        pre_attn_norm=None,
        post_attn_norm=None,
        q_proj=_keep_fe(sub_params.q_proj, "q_proj"),
        k_proj=_keep_fe(sub_params.k_proj, "kv_proj"),
        v_proj=_keep_fe(sub_params.v_proj, "kv_proj"),
        q_norm=None,
        k_norm=None,
        out_proj=_keep_fe(sub_params.out_proj, "attn_out_proj"),
        moe=moe_for_tie,
    )
  else:
    assert isinstance(sub_params, GdnSublayerParams)
    sub_for_tie = dataclasses_replace(
        sub_params,
        pre_attn_norm=None,
        post_attn_norm=None,
        in_proj_qkvz=_keep_fe(sub_params.in_proj_qkvz, "in_proj_qkvz"),
        in_proj_ba=_keep_fe(sub_params.in_proj_ba, "in_proj_ba"),
        conv1d=None,
        A_log=None,
        dt_bias=None,
        gdn_norm=None,
        out_proj=_keep_fe(sub_params.out_proj, "gdn_out_proj"),
        moe=moe_for_tie,
    )
  anchor_tied, sub_tied = ops.backward_barrier_tie(anchor, sub_for_tie)
  moe_restored = dataclasses_replace(
      p_moe,
      router_gate=(sub_tied.moe.router_gate if sub_tied.moe.router_gate is not None else p_moe.router_gate),
      shared_wi_0=(sub_tied.moe.shared_wi_0 if sub_tied.moe.shared_wi_0 is not None else p_moe.shared_wi_0),
      shared_wi_1=(sub_tied.moe.shared_wi_1 if sub_tied.moe.shared_wi_1 is not None else p_moe.shared_wi_1),
      shared_wo=(sub_tied.moe.shared_wo if sub_tied.moe.shared_wo is not None else p_moe.shared_wo),
      shared_gate=(sub_tied.moe.shared_gate if sub_tied.moe.shared_gate is not None else p_moe.shared_gate),
  )
  if is_attn:
    assert isinstance(sub_tied, AttnSublayerParams)
    sub_restored = dataclasses_replace(
        sub_params,
        q_proj=(sub_tied.q_proj if sub_tied.q_proj is not None else sub_params.q_proj),
        k_proj=(sub_tied.k_proj if sub_tied.k_proj is not None else sub_params.k_proj),
        v_proj=(sub_tied.v_proj if sub_tied.v_proj is not None else sub_params.v_proj),
        out_proj=(sub_tied.out_proj if sub_tied.out_proj is not None else sub_params.out_proj),
        moe=moe_restored,
    )
  else:
    assert isinstance(sub_tied, GdnSublayerParams)
    sub_restored = dataclasses_replace(
        sub_params,
        in_proj_qkvz=(sub_tied.in_proj_qkvz if sub_tied.in_proj_qkvz is not None else sub_params.in_proj_qkvz),
        in_proj_ba=(sub_tied.in_proj_ba if sub_tied.in_proj_ba is not None else sub_params.in_proj_ba),
        out_proj=(sub_tied.out_proj if sub_tied.out_proj is not None else sub_params.out_proj),
        moe=moe_restored,
    )
  return anchor_tied, sub_restored


def dataclasses_replace(obj, **changes):
  import dataclasses

  return dataclasses.replace(obj, **changes)


def _gather_dense_and_norms(
    sub_params: GdnSublayerParams | AttnSublayerParams,
    cfg: Qwen35Config,
    *,
    is_attn: bool,
    start_after: jax.Array | None = None,
):
  """Gathers dense mixer + shared-expert + router weights on SC1 (`fsdp`), leaving `wi`/`wo` separate."""
  p_moe = sub_params.moe
  moe_dense = layers.GatheredMoeWeights(
      router_gate=ops.gather_or_replicate_dense_weight(
          p_moe.router_gate,
          cfg,
          param_name="router_gate",
          sharded_axis=0,
          start_after=start_after,
      ),
      wi=p_moe.wi,  # placeholder until SC1 routed weight gather
      wo=p_moe.wo,  # placeholder until SC1 routed weight gather
      shared_wi_0=ops.gather_or_replicate_dense_weight(
          p_moe.shared_wi_0,
          cfg,
          param_name="shared_wi",
          sharded_axis=0,
          start_after=start_after,
      ),
      shared_wi_1=ops.gather_or_replicate_dense_weight(
          p_moe.shared_wi_1,
          cfg,
          param_name="shared_wi",
          sharded_axis=0,
          start_after=start_after,
      ),
      shared_wo=ops.gather_or_replicate_dense_weight(
          p_moe.shared_wo,
          cfg,
          param_name="shared_wo",
          sharded_axis=1,
          start_after=start_after,
      ),
      shared_gate=ops.gather_or_replicate_dense_weight(
          p_moe.shared_gate,
          cfg,
          param_name="shared_gate",
          sharded_axis=0,
          start_after=start_after,
      ),
  )
  if is_attn:
    assert isinstance(sub_params, AttnSublayerParams)
    q_in = sub_params.q_proj.reshape(sub_params.q_proj.shape[0], -1) if cfg.attn_qkv_2d_matmul else sub_params.q_proj
    k_in = sub_params.k_proj.reshape(sub_params.k_proj.shape[0], -1) if cfg.attn_qkv_2d_matmul else sub_params.k_proj
    v_in = sub_params.v_proj.reshape(sub_params.v_proj.shape[0], -1) if cfg.attn_qkv_2d_matmul else sub_params.v_proj
    return layers.GatheredAttnWeights(
        pre_attn_norm=ops.replicate_small_param(sub_params.pre_attn_norm, cfg),
        post_attn_norm=ops.replicate_small_param(sub_params.post_attn_norm, cfg),
        q_proj=ops.gather_or_replicate_dense_weight(
            q_in,
            cfg,
            param_name="q_proj",
            sharded_axis=0,
            start_after=start_after,
        ),
        k_proj=ops.gather_or_replicate_dense_weight(
            k_in,
            cfg,
            param_name="kv_proj",
            sharded_axis=0,
            start_after=start_after,
        ),
        v_proj=ops.gather_or_replicate_dense_weight(
            v_in,
            cfg,
            param_name="kv_proj",
            sharded_axis=0,
            start_after=start_after,
        ),
        q_norm=ops.replicate_small_param(sub_params.q_norm, cfg),
        k_norm=ops.replicate_small_param(sub_params.k_norm, cfg),
        out_proj=ops.gather_or_replicate_dense_weight(
            sub_params.out_proj,
            cfg,
            param_name="attn_out_proj",
            sharded_axis=0,
            start_after=start_after,
        ),
        moe=moe_dense,
    )
  assert isinstance(sub_params, GdnSublayerParams)
  return layers.GatheredGdnWeights(
      pre_attn_norm=ops.replicate_small_param(sub_params.pre_attn_norm, cfg),
      post_attn_norm=ops.replicate_small_param(sub_params.post_attn_norm, cfg),
      in_proj_qkvz=ops.gather_or_replicate_dense_weight(
          sub_params.in_proj_qkvz,
          cfg,
          param_name="in_proj_qkvz",
          sharded_axis=0,
          start_after=start_after,
      ),
      in_proj_ba=ops.gather_or_replicate_dense_weight(
          sub_params.in_proj_ba,
          cfg,
          param_name="in_proj_ba",
          sharded_axis=0,
          start_after=start_after,
      ),
      conv1d=ops.replicate_small_param(sub_params.conv1d, cfg),
      A_log=ops.replicate_small_param(sub_params.A_log, cfg),
      dt_bias=ops.replicate_small_param(sub_params.dt_bias, cfg),
      gdn_norm=ops.replicate_small_param(sub_params.gdn_norm, cfg),
      out_proj=ops.gather_or_replicate_dense_weight(
          sub_params.out_proj,
          cfg,
          param_name="gdn_out_proj",
          sharded_axis=0,
          start_after=start_after,
      ),
      moe=moe_dense,
  )


def _mixer_only(
    x_mb: jax.Array,
    w_dense: layers.GatheredGdnWeights | layers.GatheredAttnWeights,
    pos_mb: jax.Array,
    seg_mb: jax.Array,
    cfg: Qwen35Config,
    *,
    is_attn: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Runs GDN or FullAttn mixer + post_attn_norm returning `(h, normed)`."""
  if is_attn:
    assert isinstance(w_dense, layers.GatheredAttnWeights)
    h = layers.full_attn_mixer(x_mb, w_dense, pos_mb, seg_mb, cfg, rope_cos_sin=rope_cos_sin)
  else:
    assert isinstance(w_dense, layers.GatheredGdnWeights)
    h = layers.gdn_mixer(x_mb, w_dense, seg_mb, cfg)
  normed = ops.rms_norm_zero_centered(h, w_dense.post_attn_norm, cfg.norm_eps)
  return h, normed


def _mixer_and_shared(
    x_mb: jax.Array,
    w_dense: layers.GatheredGdnWeights | layers.GatheredAttnWeights,
    pos_mb: jax.Array,
    seg_mb: jax.Array,
    cfg: Qwen35Config,
    *,
    is_attn: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
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
  )
  base = layers.shared_expert_and_residual(h, normed, w_dense.moe)
  return base, normed


def _run_sublayer_batchsplit(
    carry: BatchSplitCarry,
    sub_params: GdnSublayerParams | AttnSublayerParams,
    sub_absmax: SublayerAbsmax,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: Qwen35Config,
    *,
    is_attn: bool,
    sublayer_idx: int = 0,
    prefetched_w_dense: layers.GatheredGdnWeights | layers.GatheredAttnWeights | None = None,
    prefetched_moe_rw: tuple[object, object] | None = None,
    moe_rw_start_after: object | None = None,
    next_sub_params: GdnSublayerParams | AttnSublayerParams | None = None,
    next_is_attn: bool = False,
    skip_own_bwd_tie: bool = False,
    is_superblock_last: bool = False,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    prev_rw3_local: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
    use_external_bwd_rw: bool = False,
):
  """4-Phase Combine-Wrapped Sub-Layer with Strict SC0 (EP) / SC1 (FSDP) Separation.

  - Phase F1: TC runs `mb0` Mixer (+ Shared) || SC0 drains `prev mb1` EP Combine
  RS || SC1 gathers `wi`/`wo` (across F1+F2!).
  - Phase F2: SC0 launches `mb0` EP Dispatch AG immediately after `mb0`
  Mixer+Router || TC overlaps `mb1` Mixer + `mb1` Shared + `mb1` Router!
  - Phase F3: SC0 launches `mb1` EP Dispatch AG || TC overlaps `mb0` MoE GMMs ||
  SC1 prefetches `w_dense(j+1)` (across F3+F4)!
  - Phase F4: SC0 launches `mb0` EP Combine RS || TC overlaps `mb1` MoE GMMs;
              returns `unperm_mb1` in carry so `mb1` EP Combine RS overlaps next
              sub-layer's F1!
  """
  x_mb0, unperm_mb1_in, base_mb1_in, bwd_rw3_in = carry
  if cfg.bwd_tie_b1_before_prev_b4 and isinstance(unperm_mb1_in, ops.MoePartialTokens):
    # Force `B4(j-1)` (`g_unperm_mb1_in`) to wait for `B1(j)` (`g_x_mb0`) via 4-byte
    # `dy_scale` tie, while `combine_bwd_bridge` below still launches `dy_codes_ep`
    # and `disp_codes_ep` on SC0 immediately after `B2(j)` to overlap ALL of `B1(j)`!
    unperm_mb1_in, x_mb0 = ops.backward_barrier_tie(unperm_mb1_in, x_mb0)

  defer_f1_combine = cfg.defer_mb1_combine_after_f1 and cfg.bwd_tie_mode in (
      "moe_xmb0_only",
      "symmetric_4phase",
      "none",
  )
  if defer_f1_combine:
    # Launch `prev mb1` SC0 EP Combine `psum_scatter` BEFORE Phase F1 (`mb0` Mixer+Shared),
    # and defer its `call-done` (`base_mb1_in + combined_mb1_prev`) until AFTER `base_mb0`!
    combined_mb1_prev = jax.lax.stop_gradient(ops.ep_token_combine(unperm_mb1_in, cfg))
    x_mb0 = checkpoint_name(x_mb0, "decoder_layer_input")
    if not skip_own_bwd_tie:
      (x_mb0, _), sub_params = _tie_bwd_sharded_params((x_mb0, x_mb0), sub_params, cfg)
  else:
    # Zero-residual cross-layer combine bridge (`prev mb1` EP Combine on SC0):
    # avoids saving extra `[B_mb, S, D]` residuals across `jax.checkpoint`!
    x_mb1_raw = jax.lax.stop_gradient(layers.moe_combine_and_add_base(unperm_mb1_in, base_mb1_in, cfg))
    x_mb1 = ops.combine_bwd_bridge(cfg, unperm_mb1_in, base_mb1_in, x_mb1_raw)
    x_mb0 = checkpoint_name(x_mb0, "decoder_layer_input")
    x_mb1 = checkpoint_name(x_mb1, "decoder_layer_input")
    if not skip_own_bwd_tie:
      (x_mb0, x_mb1), sub_params = _tie_bwd_sharded_params((x_mb0, x_mb1), sub_params, cfg)

  w_dense = (
      prefetched_w_dense if prefetched_w_dense is not None else _gather_dense_and_norms(sub_params, cfg, is_attn=is_attn)
  )

  # Launch SC1 `wi`/`wo` routed weight gather across full `F1 + F2` when `moe_rw_start_after` is used!
  early_moe_rw = None
  if prefetched_moe_rw is None and moe_rw_start_after is not None:
    from maxtext.experimental.qwen_35_64c import moe_kernels

    early_moe_rw = (
        moe_kernels.gather_routed_weight_sc1(
            sub_params.moe.wi,
            sub_absmax.wi,
            cfg,
            start_after=moe_rw_start_after,
            is_wi=True,
        ),
        moe_kernels.gather_routed_weight_sc1(
            sub_params.moe.wo,
            sub_absmax.wo,
            cfg,
            start_after=moe_rw_start_after,
            is_wi=False,
        ),
    )

  # === Phase F1 & F2 ===
  w_f1 = dataclasses_replace(
      w_dense,
      pre_attn_norm=ops.order_after(w_dense.pre_attn_norm, unperm_mb1_in),
  )
  collect_aux = cfg.router_aux_loss_coef > 0.0
  if cfg.route_before_shared_expert:
    # Run `mb0` Mixer -> `mb0` Router -> launch SC0 `disp_mb0` BEFORE `mb0` Shared Expert,
    # overlapping BOTH `mb0` Shared Expert AND `mb1` Mixer + `mb1` Router with SC0 `disp_mb0`!
    h_mb0, normed_mb0 = _mixer_only(
        x_mb0,
        w_f1,
        pos_mb0,
        seg_mb0,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb0,
    )
    topk_w_mb0, topk_ids_mb0, aux_mb0 = layers.moe_local_route_with_aux(
        normed_mb0,
        w_dense.moe.router_gate,
        cfg,
        segment_ids=seg_mb0,
        collect_aux=collect_aux,
    )
    disp_mb0 = layers.moe_dispatch_from_routed(normed_mb0, topk_w_mb0, topk_ids_mb0, cfg, order_tokens_on_router=True)
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
    )
    topk_w_mb0, topk_ids_mb0, aux_mb0 = layers.moe_local_route_with_aux(
        normed_mb0,
        w_dense.moe.router_gate,
        cfg,
        route_after=base_mb0,
        segment_ids=seg_mb0,
        collect_aux=collect_aux,
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
    )
    disp_mb0, aux_mb0 = layers.moe_local_route_and_dispatch_with_aux(
        normed_mb0,
        w_dense.moe.router_gate,
        cfg,
        route_after=base_mb0,
        segment_ids=seg_mb0,
        collect_aux=collect_aux,
    )

  if defer_f1_combine:
    # Retire `prev mb1` SC0 EP Combine RS AFTER `base_mb0` (`Phase F1` compute) completes!
    base_mb1_f1 = ops.order_after(base_mb1_in, base_mb0)
    if cfg.f2_early_mb0_dispatch and topk_w_mb0 is not None:
      base_mb1_f1 = ops.order_after(base_mb1_f1, topk_w_mb0)
    x_mb1_raw = jax.lax.stop_gradient(base_mb1_f1 + combined_mb1_prev.reshape(base_mb1_in.shape))
    unperm_for_bridge = unperm_mb1_in
    if cfg.bwd_early_mb1_dy_pmax and isinstance(unperm_mb1_in, ops.MoePartialTokens):
      base_mb0, unperm_for_bridge = ops.tie_bwd_after_partial_dy_scale(base_mb0, unperm_mb1_in)
    x_mb1 = ops.combine_bwd_bridge(cfg, unperm_for_bridge, base_mb1_in, x_mb1_raw)
    x_mb1 = checkpoint_name(x_mb1, "decoder_layer_input")

  if prefetched_moe_rw is not None:
    wi_full, wo_full = prefetched_moe_rw
  elif early_moe_rw is not None:
    wi_full, wo_full = early_moe_rw
  else:
    from maxtext.experimental.qwen_35_64c import moe_kernels

    if cfg.moe_ag_start_tie == "f2_x_mb1":
      moe_tie = x_mb1
    elif cfg.moe_ag_start_tie == "f2_base_mb0":
      moe_tie = base_mb0
    else:
      moe_tie = None
    wi_full = moe_kernels.gather_routed_weight_sc1(sub_params.moe.wi, sub_absmax.wi, cfg, start_after=moe_tie, is_wi=True)
    wo_full = moe_kernels.gather_routed_weight_sc1(
        sub_params.moe.wo, sub_absmax.wo, cfg, start_after=moe_tie, is_wi=False
    )

  if cfg.bwd_tie_mode == "symmetric_4phase":
    # B2 -> B1 backward tie: `mb0` Mixer+Shared bwd (`B1`) runs after `mb1` Mixer
    # bwd (`B2`) so `dx_mb1` Cross-Layer SC0 EP Combine-Bwd overlaps `B1` on TC!
    base_mb0, x_mb1 = ops.backward_barrier_tie(base_mb0, x_mb1)

  f2_pre_norm = ops.order_after(w_dense.pre_attn_norm, base_mb0)
  if cfg.f2_early_mb0_dispatch:
    if topk_w_mb0 is not None:
      f2_pre_norm = ops.order_after(f2_pre_norm, topk_w_mb0)
    if hasattr(disp_mb0.tokens_ep, "scale"):
      f2_pre_norm = ops.order_after(f2_pre_norm, disp_mb0.tokens_ep.scale)
  w_f2 = dataclasses_replace(w_dense, pre_attn_norm=f2_pre_norm)
  mb1_route_dep = topk_w_mb0 if (cfg.f2_light_router_ties and topk_w_mb0 is not None) else disp_mb0.tokens_ep
  if cfg.route_before_shared_expert:
    # Run `mb1` Mixer -> `mb1` Router -> launch SC0 `disp_mb1` BEFORE `mb1` Shared Expert,
    # overlapping BOTH `mb1` Shared Expert AND `mb0` MoE GMMs with SC0 `disp_mb1`!
    h_mb1, normed_mb1 = _mixer_only(
        x_mb1,
        w_f2,
        pos_mb1,
        seg_mb1,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb1,
    )
    topk_w_mb1, topk_ids_mb1, aux_mb1 = layers.moe_local_route_with_aux(
        normed_mb1,
        w_dense.moe.router_gate,
        cfg,
        route_after=mb1_route_dep,
        segment_ids=seg_mb1,
        collect_aux=collect_aux,
    )
    disp_mb1 = layers.moe_dispatch_from_routed(normed_mb1, topk_w_mb1, topk_ids_mb1, cfg, order_tokens_on_router=True)
    normed_mb1_se = ops.order_after(normed_mb1, topk_w_mb1)
    base_mb1_out = layers.shared_expert_and_residual(h_mb1, normed_mb1_se, w_f2.moe)
  elif cfg.f2_light_router_ties:
    base_mb1_out, normed_mb1 = _mixer_and_shared(
        x_mb1,
        w_f2,
        pos_mb1,
        seg_mb1,
        cfg,
        is_attn=is_attn,
        rope_cos_sin=rope_mb1,
    )
    topk_w_mb1, topk_ids_mb1, aux_mb1 = layers.moe_local_route_with_aux(
        normed_mb1,
        w_dense.moe.router_gate,
        cfg,
        route_after=(base_mb1_out, mb1_route_dep),
        segment_ids=seg_mb1,
        collect_aux=collect_aux,
    )
    disp_mb1 = layers.moe_dispatch_from_routed(normed_mb1, topk_w_mb1, topk_ids_mb1, cfg, order_tokens_on_router=True)
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
    )
    disp_mb1, aux_mb1 = layers.moe_local_route_and_dispatch_with_aux(
        normed_mb1,
        w_dense.moe.router_gate,
        cfg,
        route_after=(base_mb1_out, disp_mb0.tokens_ep),
        segment_ids=seg_mb1,
        collect_aux=collect_aux,
    )
  sub_aux = layers.add_moe_aux_stats(aux_mb0, aux_mb1)

  # === Phase F3: `mb0` MoE GMMs (TC) || `mb1` EP Dispatch AG (SC0) || `w_dense(j+1)` (SC1) ===
  if cfg.strict_f2_before_f3:
    # Gate `mb0` MoE sort/GMMs on Phase F2 (`base_mb1_out` + `disp_mb1` router) via 4-byte
    # scale tie so `disp_mb0` (`SC0` AG) overlaps ALL of Phase F2 and `disp_mb1` (`SC0` AG)
    # launches BEFORE Phase F3 begins!
    f3_router_dep = topk_w_mb1 if (cfg.f2_light_router_ties and topk_w_mb1 is not None) else disp_mb1.topk_weights_ep
    disp_mb0 = layers.order_dispatched_after(disp_mb0, (base_mb1_out, f3_router_dep))

  if cfg.bwd_tie_mode == "symmetric_4phase":
    # B3 -> B2 backward tie: `mb1` Mixer+Shared bwd (`B2`) runs after `mb0` MoE bwd (`B3`)!
    base_mb1_out, tok_ep_mb0 = ops.backward_barrier_tie(base_mb1_out, disp_mb0.tokens_ep)
    disp_mb0 = layers.DispatchedMoeTokens(
        tokens_ep=tok_ep_mb0,
        topk_weights_ep=disp_mb0.topk_weights_ep,
        topk_ids_ep=disp_mb0.topk_ids_ep,
    )
  if prev_rw3_local is not None and bwd_rw3_in is not None:
    from maxtext.experimental.qwen_35_64c import moe_kernels

    tok_ep_mb0_rw = moe_kernels.produce_prev_bwd_rw3(disp_mb0.tokens_ep, bwd_rw3_in, prev_rw3_local, cfg)
    disp_mb0 = layers.DispatchedMoeTokens(
        tokens_ep=tok_ep_mb0_rw,
        topk_weights_ep=disp_mb0.topk_weights_ep,
        topk_ids_ep=disp_mb0.topk_ids_ep,
    )
  wrap_pt_mb0 = cfg.moe_use_forked_3d_gmmv2 and cfg.moe_bwd_dy_fp8_e5m2
  wrap_pt_mb1 = wrap_pt_mb0 and not is_superblock_last
  if use_external_bwd_rw:
    unperm_mb0, wi_chain, wo_chain, rw3_slot_mb0 = layers.moe_expert_gmm_and_unsort(
        disp_mb0,
        wi_full,
        wo_full,
        cfg,
        return_weights=True,
        wrap_partial_fp8_bwd=wrap_pt_mb0,
        sublayer_idx=sublayer_idx,
        use_external_bwd_rw=True,
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

  # === Deferred Next-Sublayer Dense Post-RS Bwd Tie + SC1 `w_dense(j+1)` Prefetch ===
  next_params_tied = None
  w_dense_next = None
  if next_sub_params is not None:
    # Defer ONLY `dw_dense(j+1)` (~0.94 GiB) across `B4(j)` (when SC1 is 100% idle!)
    # so it retires right before `B3(j)` launches `dwi(j)`/`dwo(j)` on SC1, while
    # `(dwi(j+1), dwo(j+1))` (~6 GiB) already retired at `dx_mb0(j+1)` (zero extra HBM!).
    unperm_mb0, next_params_tied = _tie_bwd_dense_only_params(unperm_mb0, next_sub_params, cfg)
    # Prefetch `w_dense(j+1)` (~0.94 GiB) on SC1 during `F3(j) + F4(j)` (~7 ms window)!
    wi_dep = wi_full.codes if hasattr(wi_full, "codes") else wi_full
    wo_dep = wo_full.codes if hasattr(wo_full, "codes") else wo_full
    prefetch_trigger = ops.order_after(ops.order_after(disp_mb0.tokens_ep, wi_dep), wo_dep)
    w_dense_next = _gather_dense_and_norms(
        next_params_tied,
        cfg,
        is_attn=next_is_attn,
        start_after=prefetch_trigger,
    )

  # === Phase F4: `mb1` MoE GMMs (TC) || `mb0` EP Combine (SC0) ===
  tok_ep_mb1_f4 = ops.order_after(disp_mb1.tokens_ep, unperm_mb0)
  if cfg.bwd_tie_mode == "symmetric_4phase":
    # B4 -> B3 backward tie: `mb0` MoE bwd (`B3`) runs after `mb1` MoE bwd (`B4`),
    # hiding `mb0` EP Combine-Bwd (`SC0` AG) behind `B4` and `mb1` EP Dispatch-Bwd (`SC0` RS) behind `B3`!
    unperm_mb0, tok_ep_mb1_f4 = ops.backward_barrier_tie(unperm_mb0, tok_ep_mb1_f4)

  if cfg.strict_f2_before_f3:
    # Launch `mb0` SC0 EP Combine `psum_scatter` BEFORE `mb1` MoE GMMs and retire AFTER!
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
  )
  if next_sub_params is not None:
    return carry_out, next_params_tied, w_dense_next, sub_aux
  return carry_out, sub_aux


def run_superblock_batchsplit(
    carry: BatchSplitCarry,
    sb_params: SuperblockParams,
    sb_absmax: SuperblockAbsmax,
    pos_mb0: jax.Array,
    pos_mb1: jax.Array,
    seg_mb0: jax.Array,
    seg_mb1: jax.Array,
    cfg: Qwen35Config,
    *,
    rope_mb0: tuple[jax.Array, jax.Array] | None = None,
    rope_mb1: tuple[jax.Array, jax.Array] | None = None,
    prev_rw3_local: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None,
    use_attn3_external_bwd_rw: bool = False,
) -> tuple[BatchSplitCarry, layers.MoeAuxStats | None]:
  """Flat-unrolls `[GDN_0, GDN_1, GDN_2, FullAttn_3]` (`whiles=0` inside outer scan)."""
  if cfg.fsdp_pipeline_mode == "full_prefetch_defer_rs":
    w_dense_0 = _gather_dense_and_norms(sb_params.gdn_0, cfg, is_attn=False, start_after=None)
    carry, p_gdn_1, w_dense_1, aux_0 = _run_sublayer_batchsplit(
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
        prefetched_w_dense=w_dense_0,
        moe_rw_start_after=w_dense_0.out_proj,
        next_sub_params=sb_params.gdn_1,
        next_is_attn=False,
        skip_own_bwd_tie=False,
        prev_rw3_local=prev_rw3_local,
    )
    rw_tie_1 = w_dense_1.out_proj if cfg.moe_rw_prefetch_in_f3 else carry.unperm_mb1
    carry, p_gdn_2, w_dense_2, aux_1 = _run_sublayer_batchsplit(
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
        prefetched_w_dense=w_dense_1,
        moe_rw_start_after=rw_tie_1,
        next_sub_params=sb_params.gdn_2,
        next_is_attn=False,
        skip_own_bwd_tie=False,
    )
    rw_tie_2 = w_dense_2.out_proj if cfg.moe_rw_prefetch_in_f3 else carry.unperm_mb1
    carry, p_attn_3, w_dense_3, aux_2 = _run_sublayer_batchsplit(
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
        prefetched_w_dense=w_dense_2,
        moe_rw_start_after=rw_tie_2,
        next_sub_params=sb_params.attn_3,
        next_is_attn=True,
        skip_own_bwd_tie=False,
    )
    rw_tie_3 = w_dense_3.out_proj if cfg.moe_rw_prefetch_in_f3 else carry.unperm_mb1
    carry, aux_3 = _run_sublayer_batchsplit(
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
        prefetched_w_dense=w_dense_3,
        moe_rw_start_after=rw_tie_3,
        next_sub_params=None,
        skip_own_bwd_tie=False,
        is_superblock_last=True,
        rope_mb0=rope_mb0,
        rope_mb1=rope_mb1,
        use_external_bwd_rw=use_attn3_external_bwd_rw,
    )
    return carry, layers.stack_moe_aux_stats((aux_0, aux_1, aux_2, aux_3))

  if cfg.fsdp_pipeline_mode == "sublayer_prefetch":
    from maxtext.experimental.qwen_35_64c import moe_kernels

    # Pipeline SC1 MoE routed weight gathers one sub-layer ahead!
    rw0 = (
        moe_kernels.gather_routed_weight_sc1(sb_params.gdn_0.moe.wi, sb_absmax.gdn_0.wi, cfg, is_wi=True),
        moe_kernels.gather_routed_weight_sc1(sb_params.gdn_0.moe.wo, sb_absmax.gdn_0.wo, cfg, is_wi=False),
    )
    rw1 = (
        moe_kernels.gather_routed_weight_sc1(sb_params.gdn_1.moe.wi, sb_absmax.gdn_1.wi, cfg, is_wi=True),
        moe_kernels.gather_routed_weight_sc1(sb_params.gdn_1.moe.wo, sb_absmax.gdn_1.wo, cfg, is_wi=False),
    )
    carry, aux_0 = _run_sublayer_batchsplit(
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
        prefetched_moe_rw=rw0,
    )
    rw2 = (
        moe_kernels.gather_routed_weight_sc1(sb_params.gdn_2.moe.wi, sb_absmax.gdn_2.wi, cfg, is_wi=True),
        moe_kernels.gather_routed_weight_sc1(sb_params.gdn_2.moe.wo, sb_absmax.gdn_2.wo, cfg, is_wi=False),
    )
    carry, aux_1 = _run_sublayer_batchsplit(
        carry,
        sb_params.gdn_1,
        sb_absmax.gdn_1,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn=False,
        sublayer_idx=1,
        prefetched_moe_rw=rw1,
    )
    rw3 = (
        moe_kernels.gather_routed_weight_sc1(sb_params.attn_3.moe.wi, sb_absmax.attn_3.wi, cfg, is_wi=True),
        moe_kernels.gather_routed_weight_sc1(sb_params.attn_3.moe.wo, sb_absmax.attn_3.wo, cfg, is_wi=False),
    )
    carry, aux_2 = _run_sublayer_batchsplit(
        carry,
        sb_params.gdn_2,
        sb_absmax.gdn_2,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn=False,
        sublayer_idx=2,
        prefetched_moe_rw=rw2,
    )
    carry, aux_3 = _run_sublayer_batchsplit(
        carry,
        sb_params.attn_3,
        sb_absmax.attn_3,
        pos_mb0,
        pos_mb1,
        seg_mb0,
        seg_mb1,
        cfg,
        is_attn=True,
        sublayer_idx=3,
        prefetched_moe_rw=rw3,
        is_superblock_last=True,
        rope_mb0=rope_mb0,
        rope_mb1=rope_mb1,
    )
    return carry, layers.stack_moe_aux_stats((aux_0, aux_1, aux_2, aux_3))

  carry, aux_0 = _run_sublayer_batchsplit(
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
  )
  carry, aux_1 = _run_sublayer_batchsplit(
      carry,
      sb_params.gdn_1,
      sb_absmax.gdn_1,
      pos_mb0,
      pos_mb1,
      seg_mb0,
      seg_mb1,
      cfg,
      is_attn=False,
      sublayer_idx=1,
  )
  carry, aux_2 = _run_sublayer_batchsplit(
      carry,
      sb_params.gdn_2,
      sb_absmax.gdn_2,
      pos_mb0,
      pos_mb1,
      seg_mb0,
      seg_mb1,
      cfg,
      is_attn=False,
      sublayer_idx=2,
  )
  carry, aux_3 = _run_sublayer_batchsplit(
      carry,
      sb_params.attn_3,
      sb_absmax.attn_3,
      pos_mb0,
      pos_mb1,
      seg_mb0,
      seg_mb1,
      cfg,
      is_attn=True,
      sublayer_idx=3,
      is_superblock_last=True,
      rope_mb0=rope_mb0,
      rope_mb1=rope_mb1,
  )
  return carry, layers.stack_moe_aux_stats((aux_0, aux_1, aux_2, aux_3))
