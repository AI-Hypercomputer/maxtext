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

"""Dense prefix decoder layer (`layers 0..1`) with per-layer remat for Flint."""

from __future__ import annotations

import jax

from maxtext.experimental.flint40b import flint_attention
from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_mlp
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops


def flint_dense_layer_forward(
    x_local: jax.Array,
    weights_shard: flint_types.FlintDenseLayerWeights[jax.Array],
    positions_local: jax.Array,
    segment_ids_local: jax.Array,
    cfg: flint_config.FlintConfig,
) -> jax.Array:
  """Runs one dense prefix Flint decoder layer (`is_global=True`).

  Executes inside `jax.shard_map(..., check_vma=False)`. When
  `cfg.train.remat_policy == 'layer'`, sharded `weights_shard` is passed into
  `jax.checkpoint(..., policy=nothing_saveable, prevent_cse=True)` (since dense
  prefix layers execute outside `jax.lax.scan`) so all-gathered BF16 weights and
  forward intermediates are rematerialized in backward rather than kept live by
  CSE.

  Args:
    x_local: Local residual activations `[B_loc, S_loc, hidden_size]` in natural
      contiguous CP order.
    weights_shard: Sharded `FlintDenseLayerWeights` for a single dense layer.
    positions_local: Local token positions `int32[B_loc, S_loc]`.
    segment_ids_local: Local document segment IDs `int32[B_loc, S_loc]`.
    cfg: Validated `FlintConfig`.

  Returns:
    Local output activations `[B_loc, S_loc, hidden_size]` in
    `cfg.train.compute_jnp_dtype`.

  Raises:
    ValueError: If `cfg.train.remat_policy` is unsupported.
  """
  comp_dt = cfg.train.compute_jnp_dtype
  eps = cfg.model.rms_norm_eps

  def _impl(
      x_in: jax.Array,
      w_s: flint_types.FlintDenseLayerWeights[jax.Array],
      pos: jax.Array,
      seg: jax.Array,
  ) -> jax.Array:
    x_norm1 = jax.ad_checkpoint.checkpoint_name(ops.flint_rms_norm(x_in, w_s.input_layernorm, eps), "x_norm1")
    attn_out = flint_attention.flint_attention_forward(
        x_norm1,
        w_s.self_attn,
        pos,
        seg,
        cfg,
        is_global=True,
    )
    hidden = jax.ad_checkpoint.checkpoint_name(x_in.astype(comp_dt) + attn_out, "post_attn")
    x_norm2 = jax.ad_checkpoint.checkpoint_name(ops.flint_rms_norm(hidden, w_s.post_attention_layernorm, eps), "x_norm2")
    mlp_out = flint_mlp.flint_mlp_forward(
        x_norm2,
        w_s.mlp,
        cfg,
    )
    return hidden + mlp_out

  if cfg.train.remat_policy == "layer":
    remat_fn = jax.checkpoint(
        _impl,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=True,
    )
    return remat_fn(x_local, weights_shard, positions_local, segment_ids_local)
  if cfg.train.remat_policy == "selective":
    remat_fn = jax.checkpoint(
        _impl,
        policy=jax.checkpoint_policies.save_only_these_names(*flint_config.SELECTIVE_SAVED_CHECKPOINT_NAMES),
        prevent_cse=True,
    )
    return remat_fn(x_local, weights_shard, positions_local, segment_ids_local)
  if cfg.train.remat_policy == "global_selective":
    remat_fn = jax.checkpoint(
        _impl,
        policy=jax.checkpoint_policies.save_only_these_names(*flint_config.GLOBAL_SELECTIVE_SAVED_CHECKPOINT_NAMES),
        prevent_cse=True,
    )
    return remat_fn(x_local, weights_shard, positions_local, segment_ids_local)
  if cfg.train.remat_policy == "none":
    return _impl(x_local, weights_shard, positions_local, segment_ids_local)
  raise ValueError(f"Unsupported remat_policy: {cfg.train.remat_policy!r}.")
