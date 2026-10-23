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

"""Shared routed-expert runtime: the mechanics every MoE model has in common.

The split is "share the runtime, not the model semantics". Everything that
differs between MoE families -- how router scores are computed (softmax,
sigmoid), the routing bias, group-limited top-k, score normalization, the
load-balance loss, shared experts -- lives in each model file. What arrives
here is the routing *decision*: for every token, `top_k` expert indices and
the weight each expert's output gets. This module turns that into the expert
computation and the weighted combination, and nothing else. There is no
model-name branching and no config object; the arguments are the whole
interface.

The caller picks an `Executor`, normally once at construction with
`executor_from_config(config, sharding)`:

  * `"dense"` runs every expert on every token and masks the combine. Exact,
    simple, and what the parity tests compare against legacy's
    `sparse_matmul=False` path. Cost scales with `num_experts`, not `top_k`,
    so it is for small models and tests.
  * `"ragged_dot"` sorts the `(token, expert)` assignments by expert and runs
    one grouped matmul (`jax.lax.ragged_dot`) per projection. Cost scales with
    `top_k`. Dropless: no capacity factor, so no token is ever dropped and
    group sizes are data dependent. Plain `jnp`; XLA shards it.
  * `"megablox"` and `"tokamax"` are the same sort / grouped matmul / scatter
    with a Pallas kernel doing the matmuls (see `core/kernels.py`). Mosaic
    kernels cannot be partitioned by XLA, so this whole path runs per device
    under a `jax.shard_map`: tokens stay on their batch shard, expert weights
    are gathered over the FSDP axis at the boundary (as legacy does), the MLP
    dimension stays tensor-sharded and is reduced at the end. The sort is
    therefore local, which is the point. Expert parallelism -- a mesh axis
    over the experts themselves -- needs an all-to-all these executors do not
    have yet, and is refused rather than computed wrongly.

Capacity-factor token dropping is deliberately not here.
"""

import dataclasses
import functools
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.m3.core import kernels
from maxtext.m3.core.sharding import MeshAxes, Sharding, TensorType

ExecutorKind = Literal["dense", "ragged_dot", "megablox", "tokamax"]

# Legacy's `float32_weight_sum` default: experts' outputs are combined in
# float32 regardless of the compute dtype, then cast back.
_COMBINE_DTYPE = jnp.float32


@dataclasses.dataclass(frozen=True)
class Executor:
  """How `routed_experts` runs the expert matmuls.

  The mesh-axis fields matter only to the Pallas kinds, whose `shard_map`
  needs them: `batch` shards the tokens, `experts` is whatever shards the
  leading expert dimension of the weights (size 1 for now), and `mlp` shards
  the expert hidden dimension, computed per shard and summed at the end. They
  come from the model's `Sharding`, so the kernel moves with the strategy.
  Frozen, so Flax treats a module attribute holding one as static.
  """

  kind: ExecutorKind = "dense"
  batch: MeshAxes = None
  experts: MeshAxes = None
  mlp: MeshAxes = None


DENSE = Executor("dense")
RAGGED_DOT = Executor("ragged_dot")


def combine_weights(weights: jax.Array, indices: jax.Array, num_experts: int) -> jax.Array:
  """Scatters per-token top-k weights into a dense `(batch, length, num_experts)` table.

  Entries not selected are zero. `top_k` never repeats an expert for one
  token, so `set` and `add` agree; `add` is used so that a caller feeding
  replayed routing with padding slots (weight zero) gets a defined result.
  """
  batch, length, _ = indices.shape
  table = jnp.zeros((batch, length, num_experts), dtype=weights.dtype)
  batch_index = jnp.arange(batch)[:, None, None]
  length_index = jnp.arange(length)[None, :, None]
  return table.at[batch_index, length_index, indices].add(weights)


def load_balance_loss(indices: jax.Array, probs: jax.Array, *, weight: float) -> jax.Array:
  """Switch-Transformer load-balance loss, per sequence, scaled by `weight`.

  The product of two per-expert fractions, each computed over one sequence:
  the fraction of routing slots sent to the expert, and the mean router
  probability it received. Minimizing it pushes both towards uniform. The
  `num_experts**2` factor makes a perfectly balanced router score `weight`
  regardless of expert count. Same formula and reduction as legacy
  `RoutedMoE.load_balance_loss`, so the two stacks' losses compare directly.

  Args:
    indices: `(batch, length, top_k)` selected expert ids.
    probs: `(batch, length, num_experts)` router probabilities.
    weight: `load_balance_loss_weight`; the result is already multiplied by it.
  """
  num_experts = probs.shape[-1]
  top_k = indices.shape[-1]
  expert_mask = jax.nn.one_hot(indices, num_experts, dtype=jnp.int32)
  density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / top_k
  density_prob = jnp.mean(probs, axis=1)
  return jnp.mean(density * density_prob) * num_experts**2 * weight


def _dense(x, weights, indices, wi_0, wi_1, wo, *, num_experts, activation):
  """Every expert on every token, then a masked weighted sum over experts."""
  table = combine_weights(weights, indices, num_experts)
  hidden = activation(jnp.einsum("bse,neh->bsnh", x, wi_0)) * jnp.einsum("bse,neh->bsnh", x, wi_1)
  per_expert = jnp.einsum("bsnh,nhe->bsne", hidden.astype(x.dtype), wo)
  return jnp.einsum("bsne,bsn->bse", per_expert.astype(_COMBINE_DTYPE), table.astype(_COMBINE_DTYPE))


def _grouped(x, weights, indices, wi_0, wi_1, wo, *, num_experts, activation, matmul):
  """Sort assignments by expert, one grouped matmul per projection, scatter back."""
  batch, length, embed = x.shape
  top_k = indices.shape[-1]
  num_tokens = batch * length

  # Every (token, slot) assignment becomes a row. Sorting rows by expert makes
  # each expert's rows contiguous, which is the layout grouped matmuls consume.
  flat_indices = jnp.reshape(indices, (-1,))
  order = jnp.argsort(flat_indices, stable=True)
  token_of_row = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)[order]
  weight_of_row = jnp.reshape(weights, (-1,))[order]
  group_sizes = jnp.bincount(flat_indices, length=num_experts).astype(jnp.int32)

  rows = jnp.reshape(x, (num_tokens, embed))[token_of_row]
  gate = matmul(rows, wi_0, group_sizes)
  up = matmul(rows, wi_1, group_sizes)
  hidden = (activation(gate) * up).astype(x.dtype)
  out_rows = matmul(hidden, wo, group_sizes)
  out_rows = out_rows * weight_of_row[:, None].astype(_COMBINE_DTYPE)

  out = jnp.zeros((num_tokens, wo.shape[-1]), dtype=_COMBINE_DTYPE).at[token_of_row].add(out_rows)
  return jnp.reshape(out, (batch, length, wo.shape[-1]))


def _sharded_grouped(x, weights, indices, wi_0, wi_1, wo, *, num_experts, activation, executor: Executor):
  """`_grouped` per device under `shard_map`, for backends XLA cannot partition.

  Tokens (`x`, `weights`, `indices`) keep their batch sharding, so the sort
  and scatter are local. Weights arrive with their FSDP (embed) axis gathered
  -- that is what leaving it out of the `in_specs` makes XLA do -- and their
  expert-MLP axis still sharded, so each device computes a slice of the
  hidden dimension and the down-projection's partial sums are added across
  that axis at the end.
  """
  mesh = kernels.mesh_in_scope()
  if kernels.mesh_axis_size(mesh, executor.experts) > 1:
    raise ValueError(
        f"The {executor.kind!r} executor does not do expert parallelism yet; the mesh shards the expert dimension "
        f"{kernels.mesh_axis_size(mesh, executor.experts)} ways. Use 'ragged_dot' or a mesh with that axis at size 1."
    )
  batch_shards = kernels.mesh_axis_size(mesh, executor.batch)
  if x.shape[0] % batch_shards:
    raise ValueError(f"Batch {x.shape[0]} does not divide over the {batch_shards} token shards of {executor.batch!r}.")

  tokens_spec = P(executor.batch, None, None)
  wi_spec = P(executor.experts, None, executor.mlp)
  wo_spec = P(executor.experts, executor.mlp, None)
  reduce_over = kernels.mesh_axis_names(executor.mlp)
  matmul = functools.partial(kernels.grouped_matmul, backend=executor.kind)

  def per_device(x, weights, indices, wi_0, wi_1, wo):
    out = _grouped(x, weights, indices, wi_0, wi_1, wo, num_experts=num_experts, activation=activation, matmul=matmul)
    return jax.lax.psum(out, reduce_over) if reduce_over else out

  sharded = jax.shard_map(
      per_device,
      mesh=mesh,
      in_specs=(tokens_spec, tokens_spec, tokens_spec, wi_spec, wi_spec, wo_spec),
      out_specs=tokens_spec,
      check_vma=False,
  )
  return sharded(x, weights, indices, wi_0, wi_1, wo)


def routed_experts(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    *,
    executor: Executor,
    activation=jax.nn.silu,
) -> jax.Array:
  """Runs the selected experts on each token and combines their outputs.

  Args:
    x: Inputs, `(batch, length, embed)`. Sets the compute dtype; the kernels
      are cast to it.
    weights: Combine weight per selected expert, `(batch, length, top_k)`.
      Already normalized and scaled however the model's router wants.
    indices: Selected expert ids, `(batch, length, top_k)`, each in
      `[0, num_experts)`.
    wi_0: Gate projection kernels, `(num_experts, embed, mlp)`.
    wi_1: Up projection kernels, `(num_experts, embed, mlp)`.
    wo: Down projection kernels, `(num_experts, mlp, embed)`.
    executor: See the module docstring; `DENSE` and `RAGGED_DOT` are the two
      that need no mesh axes.
    activation: Applied to the gate projection; `silu` for SwiGLU.

  Returns:
    `(batch, length, embed)` in `x.dtype`.
  """
  num_experts = wi_0.shape[0]
  projections = (wi_0.astype(x.dtype), wi_1.astype(x.dtype), wo.astype(x.dtype))
  common = {"num_experts": num_experts, "activation": activation}
  match executor.kind:
    case "dense":
      out = _dense(x, weights, indices, *projections, **common)
    case "ragged_dot":
      matmul = functools.partial(kernels.grouped_matmul, backend="ragged_dot")
      out = _grouped(x, weights, indices, *projections, **common, matmul=matmul)
    case "megablox" | "tokamax":
      out = _sharded_grouped(x, weights, indices, *projections, **common, executor=executor)
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'megablox' or 'tokamax'.")
  return out.astype(x.dtype)


def executor_from_config(config, sharding: Sharding) -> Executor:
  """Picks the executor from legacy config fields.

  `sparse_matmul=False` is legacy's dense path. With `sparse_matmul=True`,
  `use_tokamax_gmm` selects tokamax's grouped matmul, otherwise `megablox`
  selects JAX's Pallas kernel, otherwise `jax.lax.ragged_dot`. The Pallas
  kinds take their mesh axes from `sharding` using the same logical names the
  models give their tokens and expert weights, so the `shard_map` boundary
  matches how those arrays are actually sharded. A positive `capacity_factor`
  asks for token dropping, which no executor does, so it is refused rather
  than silently ignored.
  """
  if config.capacity_factor > 0:
    raise ValueError(
        f"m3 MoE is dropless; capacity_factor={config.capacity_factor} is not supported yet. Set capacity_factor=-1."
    )
  if not config.sparse_matmul:
    return DENSE
  if config.use_tokamax_gmm:
    kind = "tokamax"
  elif config.megablox:
    kind = "megablox"
  else:
    return RAGGED_DOT
  tokens = sharding("moe_tokens", ("batch", "length", "embed"), TensorType.ACTIVATION)
  wi = sharding("wi_0", ("experts", "expert_embed", "expert_mlp"), TensorType.WEIGHT)
  return Executor(kind, batch=tokens[0], experts=wi[0], mlp=wi[2])
