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

"""Loss functions of the DSv3 training objective.

* `cross_entropy_with_logits`: the T5X one-hot cross entropy with z-loss and a
  stable custom gradient, verbatim from `maxtext/utils/max_utils.py`.
* `cross_entropy_with_integer_labels`: the plain-autodiff cross entropy of the
  MTP depths (`multi_token_prediction.py`).
* `masked_token_loss`: the main-objective sums MaxText's `train.loss_fn`
  derives from the logits.
* MoE router auxiliaries: Megatron-LM's sigmoid seq_aux_loss, the Switch-style
  load balance loss and the loss-free routed-bias updates, computed from the
  Lineage `DSv3RouterAux`.
"""

import dataclasses
import functools
import operator
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_types

T = TypeVar(
    "T",
    dsv3_types.ArrayType,
    dsv3_types.ShardingType,
    dsv3_types.AxisNameType,
)

# `train.py` EPS: denominators of the per-token averages.
EPS = 1e-8


# Cross entropy implementation is taken from original T5X codebase:
# https://github.com/google-research/t5x/blob/ace831eea1e2742b4299cd1a9af7e4f302038351/t5x/losses.py#L25-L101
@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.

  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxiliary z-loss loss term.

  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[Any, ...]]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (
      logits,
      targets,
      z_loss,
      exp_shifted,
      sum_exp,
      log_z,
  )


def _cross_entropy_with_logits_bwd(
    res: tuple[Any, ...], cotangents: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[jnp.ndarray, None, None]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = cotangents[0]  # Ignore z_loss component; it is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv

  return (
      jnp.asarray(g_logits, logits.dtype),
      None,  # we don't need gradients on targets
      None,  # we don't need gradients on z_loss
  )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def cross_entropy_with_integer_labels(
    logits: jt.Float[jax.Array, "*batch V"],
    labels: jt.Int[jax.Array, "*batch"],
) -> jt.Float[jax.Array, "*batch"]:
  """Per-token cross entropy against integer labels (plain autodiff).

  Mirrors `multi_token_prediction._cross_entropy_with_integer_labels`.

  Args:
    logits: Logits `[*batch, vocab]`.
    labels: Integer labels `[*batch]`.

  Returns:
    The per-token losses `[*batch]`.
  """
  lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  target = jnp.take_along_axis(logits, labels[..., None], axis=-1)
  return jnp.squeeze(lse - target, axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3TokenLossSums(Generic[T]):
  """Sums of the main objective over the valid tokens of a batch.

  Attributes:
    xent: Sum of the per-token cross entropy (including the z-loss term).
    z_loss: Sum of the per-token z-loss.
    weights: Number of valid (non-padding) target tokens.
  """

  xent: T | None = None
  z_loss: T | None = None
  weights: T | None = None


def _constrain(x: jax.Array, sharding: jax.sharding.NamedSharding) -> jax.Array:
  """Places `x` on `sharding`: `reshard` under Explicit axes, else a constraint."""
  if sharding.mesh.abstract_mesh.are_all_axes_explicit:
    return jax.reshard(x, sharding)
  return jax.lax.with_sharding_constraint(x, sharding)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def masked_token_loss(
    logits: jt.Float[jax.Array, "B T V"],
    targets: jt.Int[jax.Array, "B T"],
    target_segmentation: jt.Int[jax.Array, "B T"],
    *,
    z_loss_multiplier: float = 0.0,
    token_sharding: jax.sharding.NamedSharding | None = None,
) -> DSv3TokenLossSums[jax.Array]:
  """Computes the main-objective sums of MaxText's `train.loss_fn`.

  Args:
    logits: Logits `[batch, seq, vocab]`.
    targets: Target token ids `[batch, seq]`.
    target_segmentation: Target segmentation `[batch, seq]`; zero marks padding.
    z_loss_multiplier: z-loss coefficient.
    token_sharding: Optional sharding constraint of the per-token losses.

  Returns:
    The summed cross entropy, z-loss and valid-token count.
  """
  one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
  xent, z_loss = cross_entropy_with_logits(logits, one_hot_targets, z_loss=z_loss_multiplier)
  if token_sharding is not None:
    xent = _constrain(xent, token_sharding)
    z_loss = _constrain(z_loss, token_sharding)
  valid = target_segmentation != 0
  xent = xent * valid
  z_loss = z_loss * valid
  return DSv3TokenLossSums[jax.Array](
      xent=jnp.sum(xent),
      z_loss=jnp.sum(z_loss),
      weights=jnp.sum(valid),
  )


def _with_layer_axis(x: jax.Array, ndim_per_layer: int) -> tuple[jax.Array, bool]:
  """Prepends a singleton layer axis to unscanned router aux arrays."""
  scanned = x.ndim == ndim_per_layer + 1
  return (x if scanned else x[None]), scanned


@jax.named_call
def moe_switch_load_balance_loss(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    weight: float,
) -> jax.Array:
  """Switch-style sequence-wise load balance loss, averaged over layers.

  Mirrors MaxText's `RoutedMoE.load_balance_loss` over the softmax-normalized
  pre-bias router scores (`aux.logits`), one layer at a time to avoid a
  `(layers, batch, seq, top_k, experts)` intermediate.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    num_experts_per_tok: Router top-k.
    weight: `load_balance_loss_weight`.

  Returns:
    The scalar loss.
  """
  selected, _ = _with_layer_axis(aux.selected_experts, 3)
  gates, _ = _with_layer_axis(aux.logits, 3)

  def layer_loss(selected_l, gates_l):
    probs = jax.nn.softmax(gates_l.astype(jnp.float32), axis=-1)
    expert_mask = jax.nn.one_hot(selected_l, num_classes=num_experts, dtype=jnp.float32)
    # Fraction of tokens dispatched to each expert, per sequence.
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / num_experts_per_tok
    # Fraction of routing probability allocated to each expert, per sequence.
    density_prob = jnp.mean(probs, axis=1)
    return jnp.mean(density * density_prob)

  num_layers = selected.shape[0]
  total = functools.reduce(
      operator.add,
      (layer_loss(selected[i], gates[i]) for i in range(num_layers)),
  )
  return total / num_layers * (num_experts**2) * weight


@jax.named_call
def moe_megatron_seq_aux_loss(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    weight: float,
) -> jax.Array:
  """Megatron-LM's sigmoid-router seq_aux_loss, one loss per layer.

  Mirrors MaxText's `RoutedMoE.megatron_seq_aux_loss`: the probabilities are
  the pre-bias sigmoid scores (`aux.logits`, upcast to fp32) normalized per
  token, the token fractions come from a plain top-k over those scores, and
  both are per-sequence statistics averaged over the batch and scaled by
  `num_experts**2 * weight`. MaxText sums the per-layer losses into the
  objective.

  The scores are floored at 1e-6 and the normalization uses a 1e-6 (rather
  than 1e-20) epsilon: bf16 sigmoid scores can underflow to 0, and the
  normalization's pullback divides by the squared row sum, which flushes to
  zero on TPU with the smaller epsilon and yields NaN gradients.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    num_experts_per_tok: Router top-k.
    weight: `load_balance_loss_weight`.

  Returns:
    The per-layer losses `(num_layers,)` for a scanned stack, or a scalar for
    a single layer.
  """
  logits, scanned = _with_layer_axis(aux.logits, 3)

  def layer_loss(logits_l):
    scores = jnp.maximum(logits_l.astype(jnp.float32), 1e-6)
    probs = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-6)
    _, top_k_indices = jax.lax.top_k(scores, k=num_experts_per_tok)
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=num_experts, dtype=jnp.int32)
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / num_experts_per_tok
    density_prob = jnp.mean(probs, axis=1)
    return jnp.mean(density * density_prob) * (num_experts**2) * weight

  losses = jnp.stack([layer_loss(logits[i]) for i in range(logits.shape[0])])
  return losses if scanned else losses[0]


def expert_counts(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int) -> jax.Array:
  """Full-batch int32 expert counts, `(num_layers, E)` if scanned else `(E,)`.

  Prefers the router's per-shard dispatch counts (`aux.group_sizes`), summing
  the shards; otherwise mirrors `moe.calculate_expert_counts` over
  `aux.selected_experts`, one layer at a time.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.

  Returns:
    The expert counts.
  """
  if aux.group_sizes is not None:
    return jnp.sum(aux.group_sizes, axis=-2, dtype=jnp.int32)
  selected, scanned = _with_layer_axis(aux.selected_experts, 3)
  counts = jnp.stack(
      [
          jnp.sum(
              jax.nn.one_hot(selected[i], num_experts, dtype=jnp.int32),
              axis=(0, 1, 2),
          )
          for i in range(selected.shape[0])
      ]
  )
  return counts if scanned else counts[0]


def load_balance_updates_from_counts(counts: jax.Array, *, num_experts: int, rate: float) -> jax.Array:
  """Mirrors `moe.load_balance_updates_from_counts`: rate * sign(mean - load)."""
  total_tokens = jnp.sum(counts, axis=-1, keepdims=True)
  average_load = total_tokens / num_experts
  return jnp.sign(average_load - counts) * rate


@jax.named_call
def routed_bias_updates(aux: dsv3_types.DSv3RouterAux[Any], *, num_experts: int, rate: float) -> jax.Array:
  """Loss-free load balancing updates of the routed gate bias.

  Mirrors `RoutedMoE` under `should_update_load_balance()`:
  `rate * sign(mean_load - load)` over the global batch's expert counts.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    rate: `routed_bias_update_rate`.

  Returns:
    The updates, `(num_layers, E)` for a scanned stack or `(E,)` for a single
    layer.
  """
  counts = expert_counts(aux, num_experts=num_experts)
  return load_balance_updates_from_counts(counts, num_experts=num_experts, rate=rate)


def moe_router_aux_outputs(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    load_balance_loss_weight: float,
    megatron_seq_aux_loss: bool,
    routed_bias_update_rate: float,
) -> tuple[jax.Array | None, jax.Array | None]:
  """Returns `(load balance loss, routed bias updates)` of a router aux.

  Args:
    aux: Router aux of `dsv3.dsv3` (scanned) or `dsv3_mtp_layer` (unscanned).
    num_experts: Number of routed experts.
    num_experts_per_tok: Router top-k.
    load_balance_loss_weight: Weight of the load balance loss; 0 disables it.
    megatron_seq_aux_loss: Use `moe_megatron_seq_aux_loss` (per layer) rather
      than `moe_switch_load_balance_loss` (averaged over layers).
    routed_bias_update_rate: Rate of the routed-bias updates; 0 disables them.

  Returns:
    The loss and the updates; either is None when disabled.
  """
  lb_loss = None
  if load_balance_loss_weight > 0.0:
    loss_fn = moe_megatron_seq_aux_loss if megatron_seq_aux_loss else moe_switch_load_balance_loss
    lb_loss = loss_fn(
        aux,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        weight=load_balance_loss_weight,
    )
  updates = None
  if routed_bias_update_rate > 0.0:
    updates = routed_bias_updates(aux, num_experts=num_experts, rate=routed_bias_update_rate)
  return lb_loss, updates
