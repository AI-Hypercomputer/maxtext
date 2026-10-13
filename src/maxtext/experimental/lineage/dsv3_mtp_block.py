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

"""Multi-token prediction (MTP) block of DSv3.

A faithful port of MaxText's `MultiTokenPredictionBlock`: per depth `k`, the
targets are rolled one token to the left within document boundaries, the
rolled tokens are embedded, the MTP layer combines them with the previous
depth's hidden state, and a dedicated final norm plus the shared LM head
produce the depth's logits, cross entropy and (in eval) predictions.

The embedding lookup, the MTP layer and the LM head are injected as callables
so this module stays independent of their weights and kernels.
"""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_embed
from maxtext.experimental.lineage import dsv3_loss
from maxtext.experimental.lineage import dsv3_types

T = TypeVar(
    "T",
    dsv3_types.ArrayType,
    dsv3_types.ShardingType,
    dsv3_types.AxisNameType,
)

# `embed_fn(tokens "B T") -> "B T D"` embeddings in the compute dtype.
EmbedFn = Callable[[jax.Array], jax.Array]
# `layer_fn(h_prev, emb, weights, expert_permutations) -> (out, aux)`.
LayerFn = Callable[
    [jax.Array, jax.Array, dsv3_types.DSv3MTPWeightsPytree, jax.Array | None],
    tuple[jax.Array, dsv3_types.DSv3RouterAux[Any]],
]
# `head_fn(normed "B T D") -> logits "B T V"`.
HeadFn = Callable[[jax.Array], jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MTPDepthWeightsPytree(Generic[T]):
  """Weights of one MTP depth.

  Attributes:
    layer: The MTP layer (`eh_proj` projection plus one sparse decoder layer).
    final_norm_scale: Scale `(emb,)` of the depth's final RMSNorm
      (`mtp_{k}_final_norm`).
  """

  layer: dsv3_types.DSv3MTPWeightsPytree[T] = dataclasses.field(default_factory=dsv3_types.DSv3MTPWeightsPytree[T])
  final_norm_scale: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MTPBlockOutputs:
  """Outputs of `dsv3_mtp_block`.

  Attributes:
    losses: Per-depth sums of the masked cross entropy, `(K,)` fp32; None unless
      `compute_losses`.
    weights: Per-depth counts of valid target tokens, `(K,)` fp32; None unless
      `compute_losses`.
    preds: fp32 argmax predictions `(1, B, T)` of the `eval_target_module`
      depth; None when that depth is 0.
    mask: Validity mask `(1, B, T)` (bool) of `preds`; None when absent.
    aux: Router aux of every depth, in depth order.
  """

  losses: jax.Array | None
  weights: jax.Array | None
  preds: jax.Array | None
  mask: jax.Array | None
  aux: list[dsv3_types.DSv3RouterAux[Any]]


def _shift_left_one(x: jax.Array) -> jax.Array:
  """Left-shifts `x` by one along axis 1 and zeroes the last position."""
  # `jnp.roll` cannot slice an explicitly sharded sequence axis: shift a copy
  # gathered along it and restore the sharding (as MaxText does).
  sharding = jax.typeof(x).sharding
  spec = tuple(sharding.spec)
  if len(spec) > 1 and spec[1] is not None:
    gathered = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(spec[0], None, *spec[2:]))
    return jax.reshard(_shift_left_one(jax.reshard(x, gathered)), sharding)
  local_rolled = jnp.roll(x, -1, axis=1)
  last_mask = jnp.arange(local_rolled.shape[1]) == local_rolled.shape[1] - 1
  last_mask = last_mask.reshape((1, -1) + (1,) * (local_rolled.ndim - 2))
  return jnp.where(last_mask, 0, local_rolled)


def _replicated(x: jax.Array) -> jax.Array:
  """Gathers an explicitly sharded array onto every device."""
  sharding = jax.typeof(x).sharding
  if all(axis is None for axis in sharding.spec):
    return x
  return jax.reshard(x, jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec()))


def roll_and_mask(x: jax.Array, *, shift: int = -1) -> jax.Array:
  """Rolls `x` left along the sequence axis and zeroes the vacated positions.

  Mirrors MaxText's `roll_and_mask` without context parallelism.

  Args:
    x: Array `[batch, seq, ...]`.
    shift: Number of positions to shift left (<= 0).

  Returns:
    The rolled array.
  """
  if shift == 0:
    return x
  if shift == -1:
    return _shift_left_one(x)
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


def roll_and_mask_by_segment(x: jax.Array, segment_ids: jax.Array | None) -> jax.Array:
  """Rolls `x` one position left within the documents of `segment_ids`.

  Mirrors MaxText's `roll_and_mask_by_segment`: positions whose successor
  belongs to another document, and padding positions (segment 0), are zeroed.

  Args:
    x: Array `[batch, seq, ...]`.
    segment_ids: Segment ids `[batch, seq]`, or None for a plain roll.

  Returns:
    The rolled array.
  """
  if segment_ids is None:
    return roll_and_mask(x)
  rolled = _shift_left_one(x)
  seg_next = _shift_left_one(segment_ids)
  is_boundary = (segment_ids != seg_next) | (segment_ids == 0)
  mask = is_boundary[(...,) + (None,) * (x.ndim - 2)]
  return jnp.where(mask, 0, rolled)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mtp_block(
    main_hidden_state: jt.Num[jax.Array, "B T D"],
    input_ids: jt.Int[jax.Array, "B T"],
    target_ids: jt.Int[jax.Array, "B T"],
    target_mask: jt.Int[jax.Array, "B T"],
    segment_ids: jt.Int[jax.Array, "B T"] | None,
    depth_weights: Sequence[DSv3MTPDepthWeightsPytree],
    *,
    embed_fn: EmbedFn,
    layer_fn: LayerFn,
    head_fn: HeadFn,
    dtype: jax.typing.DTypeLike,
    norm_epsilon: float,
    compute_losses: bool = True,
    eval_target_module: int = 0,
    expert_permutations: Sequence[jax.Array | None] | None = None,
    main_token_embeddings: jt.Num[jax.Array, "B T D"] | None = None,
    token_zero_embedding: jt.Num[jax.Array, "D"] | None = None,
) -> DSv3MTPBlockOutputs:
  """Runs the MTP depths, as MaxText's `MultiTokenPredictionBlock.__call__`.

  Args:
    main_hidden_state: Pre-final-norm hidden state of the main decoder.
    input_ids: Main decoder input tokens.
    target_ids: Main decoder target tokens.
    target_mask: Main decoder target mask or packed target segmentation; any
      non-zero entry marks a valid target.
    segment_ids: Main decoder segment ids, or None when unpacked.
    depth_weights: Weights of depths `1..K`, in order.
    embed_fn: Token embedding lookup.
    layer_fn: The MTP layer; receives the previous hidden state, the rolled
      token embeddings, the depth's layer weights and its expert permutation.
    head_fn: The shared LM head (applied after the depth's final norm).
    dtype: Compute dtype of the token embeddings entering the layers.
    norm_epsilon: Epsilon of the per-depth final norm.
    compute_losses: Accumulate the per-depth loss sums (MaxText does so in
      `MODEL_MODE_TRAIN`, which its eval step also uses).
    eval_target_module: 1-based depth whose predictions are returned for the
      acceptance rate (0 disables).
    expert_permutations: Per-depth expert permutations for `layer_fn`, or None.
    main_token_embeddings: Embeddings of `input_ids` from the main decoder; when
      given they are rolled instead of embedding the rolled tokens
      (`mtp_reuse_input_embedding`).
    token_zero_embedding: Embedding of token 0 that fills the rolled-in
      positions of `main_token_embeddings`.

  Returns:
    The block outputs.
  """
  num_depths = len(depth_weights)
  if expert_permutations is None:
    expert_permutations = [None] * num_depths
  if len(expert_permutations) != num_depths:
    raise ValueError(f"Got {len(expert_permutations)} expert permutations for" f" {num_depths} MTP depths.")
  if (main_token_embeddings is None) != (token_zero_embedding is None):
    raise ValueError("main_token_embeddings and token_zero_embedding must be given together.")

  target_mask = (jnp.asarray(target_mask) != 0).astype(jnp.int32)

  mtp_hidden_state = main_hidden_state
  rolled_input_ids = input_ids
  rolled_target_ids = target_ids
  rolled_target_mask = target_mask != 0
  rolled_segment_ids = segment_ids

  losses = []
  weights = []
  preds = []
  masks = []
  aux_per_depth = []

  rolled_token_embeddings = main_token_embeddings
  not_filled = jnp.ones(input_ids.shape, dtype=jnp.int32)
  if token_zero_embedding is not None:
    # The row keeps the table's embedding-dim sharding; broadcasting it against
    # the token-sharded fill mask below must not combine the two.
    token_zero_embedding = _replicated(token_zero_embedding)

  for k in range(1, num_depths + 1):
    with jax.named_scope(f"mtp_depth_{k}"):
      if rolled_token_embeddings is not None and token_zero_embedding is not None:
        # roll_and_mask_by_segment zero-fills the positions whose token id it
        # replaces with 0; those take the embedding of token 0.
        is_filled = roll_and_mask_by_segment(not_filled, rolled_segment_ids) == 0
        rolled_token_embeddings = jnp.where(
            is_filled[..., None],
            token_zero_embedding,
            roll_and_mask_by_segment(rolled_token_embeddings, rolled_segment_ids),
        )
      rolled_input_ids = roll_and_mask_by_segment(rolled_input_ids, rolled_segment_ids)
      rolled_target_ids = roll_and_mask_by_segment(rolled_target_ids, rolled_segment_ids)
      rolled_target_mask = roll_and_mask_by_segment(rolled_target_mask, rolled_segment_ids)
      if rolled_segment_ids is not None:
        rolled_segment_ids = roll_and_mask(rolled_segment_ids)

      if rolled_token_embeddings is not None:
        target_token_embedding = rolled_token_embeddings
      else:
        target_token_embedding = embed_fn(rolled_input_ids)
      target_token_embedding = target_token_embedding.astype(dtype)

      w = depth_weights[k - 1]
      mtp_hidden_state, aux = layer_fn(
          mtp_hidden_state,
          target_token_embedding,
          w.layer,
          expert_permutations[k - 1],
      )
      aux_per_depth.append(aux)

      assert w.final_norm_scale is not None
      normed = dsv3_embed.final_norm(mtp_hidden_state, w.final_norm_scale, epsilon=norm_epsilon)
      mtp_logits = head_fn(normed)
      mtp_xent = dsv3_loss.cross_entropy_with_integer_labels(mtp_logits, rolled_target_ids)
      mtp_xent_masked = mtp_xent * rolled_target_mask

      if compute_losses:
        losses.append(jnp.sum(mtp_xent_masked))
        weights.append(jnp.sum(rolled_target_mask).astype(jnp.float32))
      if eval_target_module == k:
        # fp32 to avoid gradient errors; cast back to int32 for the rate.
        preds.append(jnp.argmax(mtp_logits, axis=-1).astype(jnp.float32))
        masks.append(rolled_target_mask)

  return DSv3MTPBlockOutputs(
      losses=jnp.stack(losses) if losses else None,
      weights=jnp.stack(weights) if weights else None,
      preds=jnp.stack(preds) if preds else None,
      mask=jnp.stack(masks) if masks else None,
      aux=aux_per_depth,
  )


def mtp_loss(outputs: DSv3MTPBlockOutputs, *, scaling_factor: float) -> jax.Array | float:
  """Pools the depth losses, as MaxText's `calculate_mtp_loss`.

  Args:
    outputs: Block outputs of a training step.
    scaling_factor: `mtp_loss_scaling_factor`.

  Returns:
    `scaling_factor * sum(losses) / (sum(weights) + EPS)`, or 0.0 without
    losses.
  """
  if outputs.losses is None or outputs.weights is None:
    return 0.0
  avg = jnp.sum(outputs.losses) / (jnp.sum(outputs.weights) + dsv3_loss.EPS)
  return avg * scaling_factor


def mtp_acceptance_rate(
    outputs: DSv3MTPBlockOutputs,
    main_logits: jax.Array,
    *,
    eval_target_module: int,
) -> jax.Array | float:
  """Acceptance rate (%) of the target depth against the main predictions.

  Mirrors MaxText's `calculate_mtp_acceptance_rate`: the main model's argmax
  predictions are rolled `eval_target_module` times and compared with the
  depth's predictions over the valid positions.

  Args:
    outputs: Block outputs of an eval step.
    main_logits: Logits `[B, T, V]` of the main decoder.
    eval_target_module: 1-based depth whose predictions `outputs` carry.

  Returns:
    The rate in percent, or 0.0 when `outputs` carries no predictions.
  """
  if outputs.preds is None or outputs.mask is None:
    return 0.0
  mtp_preds = outputs.preds.astype(jnp.int32)
  rolled_main_preds = jnp.argmax(main_logits, axis=-1)
  for _ in range(eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)
  correct = jnp.sum((mtp_preds == rolled_main_preds) * outputs.mask)
  total_valid = jnp.sum(outputs.mask)
  return (correct / (total_valid + dsv3_loss.EPS)) * 100
