# Copyright 2023–2026 Google LLC
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

"""DeepSeek MoE experts implementation."""

import functools
from typing import Protocol

import jax
import jax.numpy as jnp
import jaxtyping as jt
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops
import typeguard


class GmmFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "BT D"],
      w: jt.Num[jax.Array, "E D F"],
      group_sizes: jt.Num[jax.Array, "E"],
  ) -> jt.Num[jax.Array, "BT F"]:
    ...


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_shared_expert(
    x: jt.Num[jax.Array, "*BT D"],
    w: dsv3_types.DSv3MoESharedExpertWeightsPytree,
) -> jt.Num[jax.Array, "*BT D"]:
  """Performs computation for a shared expert.

  Args:
    x: Input tokens with any number of leading dimensions.
    w: Shared expert weights.

  Returns:
    Output tokens with the same shape as the input.
  """
  dot = functools.partial(jnp.tensordot, axes=1)
  return dot(jax.nn.silu(dot(x, w.gate_0)) * dot(x, w.gate_1), w.linear)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_impl(
    x: jt.Num[jax.Array, "BT D0 D1"],
    w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    *,
    gmm_fn: GmmFn,
) -> jt.Num[jax.Array, "BT D0 D1"]:
  """Performs routed expert computation on local tokens.

  Args:
    x: Input tokens, sorted by expert.
    w: Routed expert weights.
    group_sizes: Number of tokens assigned to each expert on the local device.
    gmm_fn: Function to perform Grouped Matrix Multiplication.

  Returns:
    Output tokens, sorted by expert.
  """
  d0, d1 = x.shape[-2], x.shape[-1]

  num_tokens = jnp.sum(group_sizes)
  x = ops.ragged_flatten(x, num_tokens, block_size_tokens=256)
  _gmm_fn = functools.partial(
      gmm_fn,
      group_sizes=group_sizes,
  )
  with jax.named_scope("gate"):
    gate_out = _gmm_fn(x, w.gate)
  with jax.named_scope("silu"):
    act = ops.ragged_silu_mul(
        gate_out,
        num_tokens,
        block_size_tokens_fwd=1024,
        block_size_hidden_fwd=2048,
        block_size_tokens_bwd=1024,
        block_size_hidden_bwd=1024,
    )
  with jax.named_scope("linear"):
    out = _gmm_fn(
        act,
        w.linear,
    )
  return ops.ragged_unflatten(out, (d0, d1), num_tokens, block_size_tokens=256)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts(
    x: jt.Num[jax.Array, "BT D0 D1"],
    w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    *,
    gmm_fn: GmmFn,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "BT D0 D1"]:
  """Performs computation for all routed experts.

  Args:
    x: Input tokens, sorted by expert.
    w: Routed expert weights.
    group_sizes: Number of tokens assigned to each expert.
    gmm_fn: Function to perform Grouped Matrix Multiplication.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Output tokens, sorted by expert.
  """
  out_specs = jax.typeof(x).sharding.spec
  return jax.shard_map(
      functools.partial(
          dsv3_routed_experts_impl,
          gmm_fn=gmm_fn,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x, w, group_sizes)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_chunk0_impl(
    x: jt.Num[jax.Array, "BT D0 D1"],
    w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    *,
    gmm_fn: GmmFn,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    jt.Num[jax.Array, "BT ..."],
]:
  """Performs routed expert computation on local tokens for chunk 0, returning gate_out.

  Args:
    x: Input tokens, sorted by expert.
    w: Routed expert weights.
    group_sizes: Number of tokens assigned to each expert on the local device.
    gmm_fn: Function to perform Grouped Matrix Multiplication.

  Returns:
    Tuple of (output tokens sorted by expert, gating GMM activations).
  """
  d0, d1 = x.shape[-2], x.shape[-1]

  num_tokens = jnp.sum(group_sizes)
  x_flat = ops.ragged_flatten(x, num_tokens, block_size_tokens=256)
  _gmm_fn = functools.partial(
      gmm_fn,
      group_sizes=group_sizes,
  )
  with jax.named_scope("gate"):
    gate_out = _gmm_fn(x_flat, w.gate)
  with jax.named_scope("silu"):
    act = ops.ragged_silu_mul(
        gate_out,
        num_tokens,
        block_size_tokens_fwd=1024,
        block_size_hidden_fwd=2048,
        block_size_tokens_bwd=1024,
        block_size_hidden_bwd=1024,
    )
  with jax.named_scope("linear"):
    out = _gmm_fn(
        act,
        w.linear,
    )
  routed_out = ops.ragged_unflatten(out, (d0, d1), num_tokens, block_size_tokens=256)
  return routed_out, gate_out


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_chunk0(
    x: jt.Num[jax.Array, "BT D0 D1"],
    w: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    *,
    gmm_fn: GmmFn,
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    jt.Num[jax.Array, "BT ..."],
]:
  """Performs computation for routed experts for chunk 0 across mesh, returning gate_out."""
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = x_pspec.partitions[0]
  gate_pspec = jax.sharding.PartitionSpec(token_axis, None)
  out_specs = (x_pspec, gate_pspec)
  return jax.shard_map(
      functools.partial(
          dsv3_routed_experts_chunk0_impl,
          gmm_fn=gmm_fn,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x, w, group_sizes)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_chunk0_bwd_impl(
    chunk0_grad_routed: jt.Num[jax.Array, "BT D0 D1"],
    chunk0_routed: jt.Num[jax.Array, "BT D0 D1"],
    chunk0_gate_out: jt.Num[jax.Array, "BT ..."],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    *,
    gmm_fn: GmmFn,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    dsv3_types.DSv3MoERoutedExpertWeightsPytree,
]:
  """Performs chunk 0 backward pass on local tokens by rematting SiLU from gate_out."""
  d0, d1 = chunk0_routed.shape[-2], chunk0_routed.shape[-1]
  num_tokens = jnp.sum(group_sizes)
  _gmm_fn = functools.partial(
      gmm_fn,
      group_sizes=group_sizes,
  )

  def _silu_linear_fwd(gate_out, w_linear):
    with jax.named_scope("silu"):
      act = ops.ragged_silu_mul(
          gate_out,
          num_tokens,
          block_size_tokens_fwd=1024,
          block_size_hidden_fwd=2048,
          block_size_tokens_bwd=1024,
          block_size_hidden_bwd=1024,
      )
    with jax.named_scope("linear"):
      out = _gmm_fn(act, w_linear)
    return ops.ragged_unflatten(out, (d0, d1), num_tokens, block_size_tokens=256)

  _, silu_linear_vjp = jax.vjp(_silu_linear_fwd, chunk0_gate_out, w_routed.linear)
  grad_gate_out, grad_w_linear = silu_linear_vjp(chunk0_grad_routed)

  def _gating_fwd(x, w_gate):
    x_flat = ops.ragged_flatten(x, num_tokens, block_size_tokens=256)
    with jax.named_scope("gate"):
      return _gmm_fn(x_flat, w_gate)

  _, gating_vjp = jax.vjp(_gating_fwd, chunk0_routed, w_routed.gate)
  chunk0_grad_routed_in, grad_w_gate = gating_vjp(grad_gate_out)
  grad_w_routed = dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=grad_w_gate, linear=grad_w_linear)
  return chunk0_grad_routed_in, grad_w_routed


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routed_experts_chunk0_bwd(
    chunk0_grad_routed: jt.Num[jax.Array, "BT D0 D1"],
    chunk0_routed: jt.Num[jax.Array, "BT D0 D1"],
    chunk0_gate_out: jt.Num[jax.Array, "BT ..."],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    group_sizes: jt.Num[jax.Array, "E"],
    *,
    gmm_fn: GmmFn,
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT D0 D1"],
    dsv3_types.DSv3MoERoutedExpertWeightsPytree,
]:
  """Sharded wrapper for chunk 0 backward pass."""
  out_specs = (
      jax.typeof(chunk0_routed).sharding.spec,
      jax.tree_util.tree_map(lambda g: jax.typeof(g).sharding.spec.to_ct_spec(), w_routed),
  )
  return jax.shard_map(
      functools.partial(
          dsv3_routed_experts_chunk0_bwd_impl,
          gmm_fn=gmm_fn,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(chunk0_grad_routed, chunk0_routed, chunk0_gate_out, w_routed, group_sizes)
