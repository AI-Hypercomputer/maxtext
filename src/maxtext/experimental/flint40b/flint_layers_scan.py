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

"""4-layer `(3 sliding + 1 global)` scanned MoE superblocks and decoder stack."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_dense_layer
from maxtext.experimental.flint40b import flint_sparse_layer
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops


def run_superblock(
    x_local: jax.Array,
    sb_layer_weights: flint_types.FlintSuperblockWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    inside_scan: bool = False,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs one 4-layer `(sliding_0..2, global_3)` MoE superblock sequentially.

  Each of the 4 sparse layers is individually wrapped in per-layer
  `jax.checkpoint` inside `flint_sparse_layer_forward` when
  `cfg.train.remat_policy == 'layer'`.

  Args:
    x_local: Local residual activations `[B_loc, S_loc, hidden_size]`.
    sb_layer_weights: Single-superblock `FlintSuperblockWeights` shard.
    positions_local: Local token positions `int32[B_loc, S_loc]`.
    segment_ids_local: Local document segment IDs `int32[B_loc, S_loc]`.
    loss_mask_local: Optional local validity mask `int32[B_loc, S_loc]`, or
      `None` when all local tokens are valid.
    cfg: Validated `FlintConfig`.
    inside_scan: `True` when called inside `jax.lax.scan` (`prevent_cse=False`);
      `False` for standalone calls outside `lax.scan` (`prevent_cse=True`).
    rope_cos_sin: Optional precomputed `(cos, sin)` tables for sliding layers.

  Returns:
    Tuple `(x_out, counts_4_e, prob_sum_4_e)` where `counts_4_e` and
    `prob_sum_4_e` have shape `float32[4, num_experts]`.
  """
  x0, c0, p0 = flint_sparse_layer.flint_sparse_layer_forward(
      x_local,
      sb_layer_weights.sliding_0,
      positions_local,
      segment_ids_local,
      loss_mask_local,
      cfg,
      is_global=False,
      inside_scan=inside_scan,
      rope_cos_sin=rope_cos_sin,
  )
  x1, c1, p1 = flint_sparse_layer.flint_sparse_layer_forward(
      x0,
      sb_layer_weights.sliding_1,
      positions_local,
      segment_ids_local,
      loss_mask_local,
      cfg,
      is_global=False,
      inside_scan=inside_scan,
      rope_cos_sin=rope_cos_sin,
  )
  x2, c2, p2 = flint_sparse_layer.flint_sparse_layer_forward(
      x1,
      sb_layer_weights.sliding_2,
      positions_local,
      segment_ids_local,
      loss_mask_local,
      cfg,
      is_global=False,
      inside_scan=inside_scan,
      rope_cos_sin=rope_cos_sin,
  )
  x3, c3, p3 = flint_sparse_layer.flint_sparse_layer_forward(
      x2,
      sb_layer_weights.global_3,
      positions_local,
      segment_ids_local,
      loss_mask_local,
      cfg,
      is_global=True,
      inside_scan=inside_scan,
  )
  counts_4_e = jnp.stack([c0, c1, c2, c3], axis=0)
  prob_sum_4_e = jnp.stack([p0, p1, p2, p3], axis=0)
  return x3, counts_4_e, prob_sum_4_e


def run_decoder(
    x_local: jax.Array,
    decoder_shard: flint_types.FlintDecoderWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs the full hybrid Flint decoder (`scan_layers=True` or `False`).

  Args:
    x_local: Local embedded activations `[B_loc, S_loc, hidden_size]`.
    decoder_shard: Sharded `FlintDecoderWeights` for all `L` layers.
    positions_local: Local token positions `int32[B_loc, S_loc]`.
    segment_ids_local: Local document segment IDs `int32[B_loc, S_loc]`.
    loss_mask_local: Optional local validity mask `int32[B_loc, S_loc]`, or
      `None` when all local tokens are valid.
    cfg: Validated `FlintConfig`.

  Returns:
    Tuple `(x_final, all_expert_counts, all_prob_sums)` where `x_final` has
    shape `[B_loc, S_loc, hidden_size]` and both routing summaries have shape
    `float32[num_moe_layers, num_experts]`.

  Raises:
    ValueError: If `decoder_shard` structure disagrees with `cfg.model`.
  """
  model = cfg.model
  num_experts = model.num_experts
  if len(decoder_shard.prefix_dense) != model.num_dense_prefix_layers:
    raise ValueError(
        f"Expected {model.num_dense_prefix_layers} prefix_dense layers, got" f" {len(decoder_shard.prefix_dense)}."
    )

  hidden = x_local.astype(cfg.train.compute_jnp_dtype)
  rope_cos_sin = (
      ops.compute_split_half_rope_cos_sin(
          positions_local,
          model.head_dim,
          rope_theta=model.rope_theta,
          dtype=cfg.train.compute_jnp_dtype,
      )
      if cfg.kernels.precompute_rope_tables
      else None
  )
  for dense_w in decoder_shard.prefix_dense:
    hidden = flint_dense_layer.flint_dense_layer_forward(hidden, dense_w, positions_local, segment_ids_local, cfg)

  if model.scan_layers:
    _, num_sb, tail_start = flint_config.flint_scan_layout(model)
    if len(decoder_shard.tail_moe) != 3:
      raise ValueError("Expected 3 tail_moe layers for scan_layers=True, got" f" {len(decoder_shard.tail_moe)}.")
    counts_parts: list[jax.Array] = []
    probs_parts: list[jax.Array] = []
    if num_sb > 0:
      if decoder_shard.scanned_blocks is None:
        raise ValueError(f"Expected non-None scanned_blocks when num_superblocks={num_sb}.")

      def _scan_body(
          carry_x: jax.Array,
          sb_w: flint_types.FlintSuperblockWeights[jax.Array],
      ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        next_x, c_4e, p_4e = run_superblock(
            carry_x,
            sb_w,
            positions_local,
            segment_ids_local,
            loss_mask_local,
            cfg,
            inside_scan=True,
            rope_cos_sin=rope_cos_sin,
        )
        return next_x, (c_4e, p_4e)

      hidden, (sb_counts, sb_probs) = jax.lax.scan(_scan_body, hidden, decoder_shard.scanned_blocks)
      counts_parts.append(sb_counts.reshape(4 * num_sb, num_experts))
      probs_parts.append(sb_probs.reshape(4 * num_sb, num_experts))

    tail_counts: list[jax.Array] = []
    tail_probs: list[jax.Array] = []
    for j, tail_w in enumerate(decoder_shard.tail_moe):
      is_glob = flint_config.is_global_layer(tail_start + j, model)
      hidden, c_e, p_e = flint_sparse_layer.flint_sparse_layer_forward(
          hidden,
          tail_w,
          positions_local,
          segment_ids_local,
          loss_mask_local,
          cfg,
          is_global=is_glob,
          rope_cos_sin=None if is_glob else rope_cos_sin,
      )
      tail_counts.append(c_e)
      tail_probs.append(p_e)
    counts_parts.append(jnp.stack(tail_counts, axis=0))
    probs_parts.append(jnp.stack(tail_probs, axis=0))
    all_counts = jnp.concatenate(counts_parts, axis=0)
    all_probs = jnp.concatenate(probs_parts, axis=0)
    return hidden, all_counts, all_probs

  expected_moe = model.num_hidden_layers - model.num_dense_prefix_layers
  if len(decoder_shard.unscanned_moe) != expected_moe:
    raise ValueError(f"Expected {expected_moe} unscanned_moe layers, got" f" {len(decoder_shard.unscanned_moe)}.")
  if expected_moe == 0:
    empty_stats = jnp.zeros((0, num_experts), dtype=jnp.float32)
    return hidden, empty_stats, empty_stats

  moe_counts: list[jax.Array] = []
  moe_probs: list[jax.Array] = []
  for j, moe_w in enumerate(decoder_shard.unscanned_moe):
    layer_idx = model.num_dense_prefix_layers + j
    is_glob = flint_config.is_global_layer(layer_idx, model)
    hidden, c_e, p_e = flint_sparse_layer.flint_sparse_layer_forward(
        hidden,
        moe_w,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        cfg,
        is_global=is_glob,
        rope_cos_sin=None if is_glob else rope_cos_sin,
    )
    moe_counts.append(c_e)
    moe_probs.append(p_e)
  return hidden, jnp.stack(moe_counts, axis=0), jnp.stack(moe_probs, axis=0)


def _run_one_sparse_inspect(
    hidden: jax.Array,
    moe_w: flint_types.FlintSparseLayerWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    is_global: bool,
    return_router_aux: bool,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, flint_types.FlintMoeAux | None]:
  """Runs one sparse layer with optional `FlintMoeAux` materialization."""
  if return_router_aux:
    return flint_sparse_layer.flint_sparse_layer_forward_with_aux(
        hidden,
        moe_w,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        cfg,
        is_global=is_global,
        rope_cos_sin=rope_cos_sin,
    )
  next_h, _, _ = flint_sparse_layer.flint_sparse_layer_forward(
      hidden,
      moe_w,
      positions_local,
      segment_ids_local,
      loss_mask_local,
      cfg,
      is_global=is_global,
      rope_cos_sin=rope_cos_sin,
  )
  return next_h, None


def run_decoder_with_intermediates(
    x_local: jax.Array,
    decoder_shard: flint_types.FlintDecoderWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    return_router_aux: bool = True,
) -> tuple[
    jax.Array,
    tuple[jax.Array, ...],
    tuple[flint_types.FlintMoeAux | None, ...],
]:
  """Runs the decoder and returns chronological per-layer hidden states and aux.

  Uses `jax.lax.scan` when `cfg.model.scan_layers=True` so full 49-layer 40B
  parity compiles in seconds, then unpacks `[N_sb, ...]` into chronological
  length-`num_hidden_layers` tuples.

  Args:
    x_local: Local embedded activations `[B_loc, S_loc, hidden_size]`.
    decoder_shard: Sharded `FlintDecoderWeights` for all `L` layers.
    positions_local: Local token positions `int32[B_loc, S_loc]`.
    segment_ids_local: Local document segment IDs `int32[B_loc, S_loc]`.
    loss_mask_local: Optional local validity mask `int32[B_loc, S_loc]`, or
      `None` when all local tokens are valid.
    cfg: Validated `FlintConfig`.
    return_router_aux: If `True`, populates `FlintMoeAux` for every MoE layer;
      if `False`, returns `None` for all entries in `layer_aux` to save memory.

  Returns:
    Tuple `(x_final, layer_hiddens, layer_aux)` where `layer_hiddens` has length
    `cfg.model.num_hidden_layers` in natural CP order and `layer_aux` has
    `None` on dense prefix layers (or all layers when `return_router_aux=False`)
    and `FlintMoeAux` on MoE layers.

  Raises:
    ValueError: If `decoder_shard` structure disagrees with `cfg.model`.
  """
  model = cfg.model
  if len(decoder_shard.prefix_dense) != model.num_dense_prefix_layers:
    raise ValueError(
        f"Expected {model.num_dense_prefix_layers} prefix_dense layers, got" f" {len(decoder_shard.prefix_dense)}."
    )

  hidden = x_local.astype(cfg.train.compute_jnp_dtype)
  rope_cos_sin = (
      ops.compute_split_half_rope_cos_sin(
          positions_local,
          model.head_dim,
          rope_theta=model.rope_theta,
          dtype=cfg.train.compute_jnp_dtype,
      )
      if cfg.kernels.precompute_rope_tables
      else None
  )
  layer_hiddens: list[jax.Array] = []
  layer_aux: list[flint_types.FlintMoeAux | None] = []

  for dense_w in decoder_shard.prefix_dense:
    hidden = flint_dense_layer.flint_dense_layer_forward(hidden, dense_w, positions_local, segment_ids_local, cfg)
    layer_hiddens.append(hidden)
    layer_aux.append(None)

  if model.scan_layers:
    _, num_sb, tail_start = flint_config.flint_scan_layout(model)
    if len(decoder_shard.tail_moe) != 3:
      raise ValueError("Expected 3 tail_moe layers for scan_layers=True, got" f" {len(decoder_shard.tail_moe)}.")
    if num_sb > 0:
      if decoder_shard.scanned_blocks is None:
        raise ValueError(f"Expected non-None scanned_blocks when num_superblocks={num_sb}.")

      def _sb_inspect_body(
          carry_x: jax.Array,
          sb_w: flint_types.FlintSuperblockWeights[jax.Array],
      ) -> tuple[
          jax.Array,
          tuple[
              tuple[jax.Array, jax.Array, jax.Array, jax.Array],
              tuple[
                  flint_types.FlintMoeAux | None,
                  flint_types.FlintMoeAux | None,
                  flint_types.FlintMoeAux | None,
                  flint_types.FlintMoeAux | None,
              ],
          ],
      ]:
        h0, a0 = _run_one_sparse_inspect(
            carry_x,
            sb_w.sliding_0,
            positions_local,
            segment_ids_local,
            loss_mask_local,
            cfg,
            is_global=False,
            return_router_aux=return_router_aux,
            rope_cos_sin=rope_cos_sin,
        )
        h1, a1 = _run_one_sparse_inspect(
            h0,
            sb_w.sliding_1,
            positions_local,
            segment_ids_local,
            loss_mask_local,
            cfg,
            is_global=False,
            return_router_aux=return_router_aux,
            rope_cos_sin=rope_cos_sin,
        )
        h2, a2 = _run_one_sparse_inspect(
            h1,
            sb_w.sliding_2,
            positions_local,
            segment_ids_local,
            loss_mask_local,
            cfg,
            is_global=False,
            return_router_aux=return_router_aux,
            rope_cos_sin=rope_cos_sin,
        )
        h3, a3 = _run_one_sparse_inspect(
            h2,
            sb_w.global_3,
            positions_local,
            segment_ids_local,
            loss_mask_local,
            cfg,
            is_global=True,
            return_router_aux=return_router_aux,
        )
        return h3, ((h0, h1, h2, h3), (a0, a1, a2, a3))

      hidden, (sb_hs, sb_auxs) = jax.lax.scan(_sb_inspect_body, hidden, decoder_shard.scanned_blocks)
      for sb_idx in range(num_sb):
        for slot_idx in range(4):
          layer_hiddens.append(sb_hs[slot_idx][sb_idx])
          slot_aux = sb_auxs[slot_idx]
          if slot_aux is None:
            layer_aux.append(None)
          else:
            layer_aux.append(jax.tree.map(lambda arr, idx=sb_idx: arr[idx], slot_aux))

    for j, tail_w in enumerate(decoder_shard.tail_moe):
      is_glob = flint_config.is_global_layer(tail_start + j, model)
      hidden, aux_j = _run_one_sparse_inspect(
          hidden,
          tail_w,
          positions_local,
          segment_ids_local,
          loss_mask_local,
          cfg,
          is_global=is_glob,
          return_router_aux=return_router_aux,
          rope_cos_sin=None if is_glob else rope_cos_sin,
      )
      layer_hiddens.append(hidden)
      layer_aux.append(aux_j)
    return hidden, tuple(layer_hiddens), tuple(layer_aux)

  expected_moe = model.num_hidden_layers - model.num_dense_prefix_layers
  if len(decoder_shard.unscanned_moe) != expected_moe:
    raise ValueError(f"Expected {expected_moe} unscanned_moe layers, got" f" {len(decoder_shard.unscanned_moe)}.")
  for j, moe_w in enumerate(decoder_shard.unscanned_moe):
    layer_idx = model.num_dense_prefix_layers + j
    is_glob = flint_config.is_global_layer(layer_idx, model)
    hidden, aux_j = _run_one_sparse_inspect(
        hidden,
        moe_w,
        positions_local,
        segment_ids_local,
        loss_mask_local,
        cfg,
        is_global=is_glob,
        return_router_aux=return_router_aux,
        rope_cos_sin=None if is_glob else rope_cos_sin,
    )
    layer_hiddens.append(hidden)
    layer_aux.append(aux_j)
  return hidden, tuple(layer_hiddens), tuple(layer_aux)
