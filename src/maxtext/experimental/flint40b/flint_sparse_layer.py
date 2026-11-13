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

"""Sparse MoE decoder layer (`layers 2..48`) with per-layer remat for Flint."""

from __future__ import annotations

import jax

from maxtext.experimental.flint40b import flint_attention
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_experts
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops


def _sparse_layer_body(
    x_in: jax.Array,
    w_s: flint_types.FlintSparseLayerWeights[jax.Array],
    pos: jax.Array,
    seg: jax.Array,
    msk: jax.Array | None,
    rope_cs: tuple[jax.Array, jax.Array] | None,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
    compute_aux_stats: bool = True,
) -> tuple[jax.Array, flint_types.FlintMoeAux]:
  """Runs pre-norm hybrid GQA attention and MoE block without checkpointing."""
  eps = cfg.model.rms_norm_eps
  comp_dt = cfg.train.compute_jnp_dtype
  x_norm1 = jax.ad_checkpoint.checkpoint_name(ops.flint_rms_norm(x_in, w_s.input_layernorm, eps), "x_norm1")
  pre_gathered_gate_up = (
      flint_experts.gather_routed_gate_up(w_s.moe.routed_experts, cfg) if cfg.kernels.prefetch_routed_weights else None
  )
  pre_gathered_down = (
      flint_experts.gather_routed_down(w_s.moe.routed_experts, cfg) if cfg.kernels.prefetch_routed_down else None
  )
  attn_out = flint_attention.flint_attention_forward(
      x_norm1,
      w_s.self_attn,
      pos,
      seg,
      cfg,
      is_global=is_global,
      rope_cos_sin=None if is_global else rope_cs,
  )
  hidden = jax.ad_checkpoint.checkpoint_name(x_in.astype(comp_dt) + attn_out, "post_attn")
  x_norm2 = jax.ad_checkpoint.checkpoint_name(ops.flint_rms_norm(hidden, w_s.post_attention_layernorm, eps), "x_norm2")
  moe_out, aux = flint_experts.flint_moe_forward(
      x_norm2,
      w_s.moe,
      msk,
      cfg,
      pre_gathered_gate_up=pre_gathered_gate_up,
      pre_gathered_down=pre_gathered_down,
      compute_aux_stats=compute_aux_stats,
  )
  return hidden + moe_out, aux


def flint_sparse_layer_forward(
    x_local: jax.Array,
    weights_shard: flint_types.FlintSparseLayerWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
    inside_scan: bool = False,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs one sparse MoE Flint decoder layer for training.

  Only `expert_counts` (`f32[E]`) and `prob_sum` (`f32[E]`) leave the checkpoint
  alongside `h_out` so `lax.scan` never saves `[N_sb, B_loc, S_loc, K]`
  `topk_ids` or `topk_weights`. Sharded `weights_shard` enters `jax.checkpoint`
  directly so `gather_dense_weight` and `gather_routed_weight` execute strictly
  inside the checkpoint with zero gathered-weight residuals. Uses
  `prevent_cse=not inside_scan` so unscanned tail/standalone layers do not keep
  forward activations live across backward via CSE.

  Args:
    x_local: Local residual activations `[B_loc, S_loc, hidden_size]` in natural
      CP order.
    weights_shard: Sharded `FlintSparseLayerWeights` for a single sparse layer.
    positions_local: Local token positions `int32[B_loc, S_loc]`.
    segment_ids_local: Local document segment IDs `int32[B_loc, S_loc]`.
    loss_mask_local: Optional local validity mask `int32[B_loc, S_loc]`, or
      `None` when all local tokens are valid.
    cfg: Validated `FlintConfig`.
    is_global: If `True`, global causal NoPE attention; otherwise RoPE causal
      sliding-window (`W=513`) attention.
    inside_scan: `True` when called inside `jax.lax.scan` (`prevent_cse=False`);
      `False` for unscanned tail or standalone layers (`prevent_cse=True`).
    rope_cos_sin: Optional precomputed `(cos, sin)` tables for sliding layers.

  Returns:
    Tuple `(h_out, expert_counts, prob_sum)` where `h_out` has shape
    `[B_loc, S_loc, hidden_size]` in `compute_jnp_dtype` and both
    `expert_counts` and `prob_sum` have shape `float32[num_experts]`.

  Raises:
    ValueError: If `cfg.train.remat_policy` is unsupported.
  """
  eff_rope_cs = None if is_global else rope_cos_sin
  compute_aux = not (cfg.kernels.skip_router_aux_stats_when_zero_coef and cfg.train.router_aux_loss_coef <= 0.0)

  def _impl(
      x_in: jax.Array,
      w_s: flint_types.FlintSparseLayerWeights[jax.Array],
      pos: jax.Array,
      seg: jax.Array,
      msk: jax.Array | None,
      rope_cs: tuple[jax.Array, jax.Array] | None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    h_out, aux = _sparse_layer_body(
        x_in,
        w_s,
        pos,
        seg,
        msk,
        rope_cs,
        cfg,
        is_global=is_global,
        compute_aux_stats=compute_aux,
    )
    return h_out, aux.expert_counts, aux.prob_sum

  if cfg.train.remat_policy == "layer":
    remat_fn = jax.checkpoint(
        _impl,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=not inside_scan,
    )
    return remat_fn(
        x_local,
        weights_shard,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        eff_rope_cs,
    )
  if cfg.train.remat_policy == "selective":
    saved_names = (
        (
            *flint_config.SELECTIVE_SAVED_CHECKPOINT_NAMES,
            *flint_config.EXTRA_SELECTIVE_SAVED_CHECKPOINT_NAMES,
        )
        if cfg.kernels.save_extra_selective_residuals
        else flint_config.SELECTIVE_SAVED_CHECKPOINT_NAMES
    )
    remat_fn = jax.checkpoint(
        _impl,
        policy=jax.checkpoint_policies.save_only_these_names(*saved_names),
        prevent_cse=not inside_scan,
    )
    return remat_fn(
        x_local,
        weights_shard,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        eff_rope_cs,
    )
  if cfg.train.remat_policy == "global_selective":
    saved_names = (
        flint_config.GLOBAL_SELECTIVE_SAVED_CHECKPOINT_NAMES
        if is_global
        else flint_config.LOCAL_SELECTIVE_SAVED_CHECKPOINT_NAMES
    )
    remat_fn = jax.checkpoint(
        _impl,
        policy=jax.checkpoint_policies.save_only_these_names(*saved_names),
        prevent_cse=not inside_scan,
    )
    return remat_fn(
        x_local,
        weights_shard,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        eff_rope_cs,
    )
  if cfg.train.remat_policy == "none":
    return _impl(
        x_local,
        weights_shard,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        eff_rope_cs,
    )
  raise ValueError(f"Unsupported remat_policy: {cfg.train.remat_policy!r}.")


def flint_sparse_layer_forward_with_aux(
    x_local: jax.Array,
    weights_shard: flint_types.FlintSparseLayerWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, flint_types.FlintMoeAux]:
  """Runs one sparse MoE decoder layer returning full `FlintMoeAux`.

  Intended for forward inspection and HuggingFace parity verification where
  per-layer `topk_ids` and `topk_weights` are inspected directly.

  Args:
    x_local: Local residual activations `[B_loc, S_loc, hidden_size]` in natural
      CP order.
    weights_shard: Sharded `FlintSparseLayerWeights` for a single sparse layer.
    positions_local: Local token positions `int32[B_loc, S_loc]`.
    segment_ids_local: Local document segment IDs `int32[B_loc, S_loc]`.
    loss_mask_local: Optional local validity mask `int32[B_loc, S_loc]`, or
      `None` when all local tokens are valid.
    cfg: Validated `FlintConfig`.
    is_global: If `True`, global causal NoPE attention; otherwise RoPE causal
      sliding-window (`W=513`) attention.
    rope_cos_sin: Optional precomputed `(cos, sin)` tables for sliding layers.

  Returns:
    Tuple `(h_out, aux)` where `aux` is the full `FlintMoeAux`.
  """
  return _sparse_layer_body(
      x_local,
      weights_shard,
      positions_local,
      segment_ids_local,
      loss_mask_local,
      None if is_global else rope_cos_sin,
      cfg,
      is_global=is_global,
  )
