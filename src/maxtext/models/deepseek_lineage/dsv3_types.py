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

"""Types for DSv3."""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any, Generic, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from maxtext.models.deepseek_lineage import ops
import numpy as np

Initializer: TypeAlias = jax.nn.initializers.Initializer
ArrayType: TypeAlias = jax.Array | np.ndarray
ShardingType: TypeAlias = jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
T = TypeVar("T", ArrayType, ShardingType)

# Default initializers for DSv3 weights.
default_kernel_init = jax.nn.initializers.variance_scaling(
    scale=1.0,
    mode="fan_in",
    distribution="truncated_normal",
)
default_scale_init = jax.nn.initializers.ones
default_bias_init = jax.nn.initializers.constant(0.0)


def _resolve_shardings(
    out_shardings: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> Any:
  """Resolves logical PartitionSpec shardings to physical NamedSharding."""
  if not out_shardings:
    return out_shardings

  def _to_sharding(s: Any) -> Any:
    if isinstance(s, jax.sharding.PartitionSpec):
      return jax.sharding.NamedSharding(mesh, ops.physical_pspec(s, axis_mapping))
    return s

  return jax.tree.map(_to_sharding, out_shardings)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3RouterAux(Generic[T]):
  """Auxiliary routing data from the DSv3 router.

  Attributes:
    group_sizes: Token counts assigned to each expert per shard. Shape:
      (num_shards, num_experts) per layer, or (num_layers, num_shards,
      num_experts) for scanned layers.
    selected_experts: Integer array of expert indices selected for each token.
      Shape: (B, T, num_experts_per_tok) per layer, or (num_layers, B, T,
        num_experts_per_tok) for scanned layers.
    logits: Router logits for all experts. Shape: (B, T, num_experts) per layer,
      or (num_layers, B, T, num_experts) for scanned layers.
  """

  group_sizes: T
  selected_experts: T
  logits: T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MLAWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MLA weights or weight shardings."""

  # We could define our own overloaded `__init__` functions to only allow
  # None for shardings, but that's a lot of boilerplate. In practice, weights
  # are initialized for testing or loaded from a checkpoint, and neither of
  # those use this init function anyways.
  q_down: T | None = None  # (emb_dim, cq_dim)
  q_up: T | None = None  # (cq_dim, num_query_heads, qk_head_dim + rope_head_dim)
  q_norm_scale: T | None = None  # (cq_dim,)
  kv_down: T | None = None  # (emb_dim, ckv_dim + rope_head_dim)
  k_up: T | None = None  # (ckv_dim, num_kv_heads, qk_head_dim)
  v_up: T | None = None  # (ckv_dim, num_kv_heads, v_head_dim)
  kv_norm_scale: T | None = None  # (ckv_dim,)
  out: T | None = None  # (num_query_heads, v_head_dim, emb_dim)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MoERouterWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MoE router weights or weight shardings."""

  kernel: T | None = None
  bias: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MoESharedExpertWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MoE shared expert weights or weight shardings."""

  gate_0: T | None = None
  gate_1: T | None = None
  linear: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MoERoutedExpertWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MoE routed expert weights or weight shardings."""

  gate: T | None = None
  linear: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MoEWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MoE weights or weight shardings."""

  router: DSv3MoERouterWeightsPytree[T] = dataclasses.field(default_factory=DSv3MoERouterWeightsPytree[T])
  shared: DSv3MoESharedExpertWeightsPytree[T] = dataclasses.field(default_factory=DSv3MoESharedExpertWeightsPytree[T])
  routed: DSv3MoERoutedExpertWeightsPytree[T] = dataclasses.field(default_factory=DSv3MoERoutedExpertWeightsPytree[T])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MLPWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MLP weights or weight shardings."""

  gate_0: T | None = None
  gate_1: T | None = None
  linear: T | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3SparseLayerWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 sparse layer weights or weight shardings."""

  pre_attn_norm_scale: T | None = None
  mla: DSv3MLAWeightsPytree[T] = dataclasses.field(default_factory=DSv3MLAWeightsPytree[T])
  post_attn_norm_scale: T | None = None
  moe: DSv3MoEWeightsPytree[T] = dataclasses.field(default_factory=DSv3MoEWeightsPytree[T])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3DenseLayerWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 dense layer weights or weight shardings."""

  pre_attn_norm_scale: T | None = None
  mla: DSv3MLAWeightsPytree[T] = dataclasses.field(default_factory=DSv3MLAWeightsPytree[T])
  post_attn_norm_scale: T | None = None
  mlp: DSv3MLPWeightsPytree[T] = dataclasses.field(default_factory=DSv3MLPWeightsPytree[T])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3EHProjWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 EHProj weights or weight shardings."""

  enorm_scale: T | None = None
  hnorm_scale: T | None = None
  eh_proj: T | None = None  # (2 * emb_dim, emb_dim)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3MTPWeightsPytree(Generic[T]):
  """Pytree to hold DSv3 MTP layer weights or weight shardings."""

  ehproj: DSv3EHProjWeightsPytree[T] = dataclasses.field(default_factory=DSv3EHProjWeightsPytree[T])
  sparse: DSv3SparseLayerWeightsPytree[T] = dataclasses.field(default_factory=DSv3SparseLayerWeightsPytree[T])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3WeightsPytree(Generic[T]):
  """Pytree to hold full DSv3 model weights or weight shardings."""

  dense: DSv3DenseLayerWeightsPytree[T] = dataclasses.field(default_factory=DSv3DenseLayerWeightsPytree[T])
  sparse: DSv3SparseLayerWeightsPytree[T] = dataclasses.field(default_factory=DSv3SparseLayerWeightsPytree[T])


def _add_layer_dim(
    shape: int | tuple[int, ...],
    num_layers: int,
) -> tuple[int, ...]:
  """Adds the layer dimension to the shape.

  If num_layers is 1, the shape is returned as is. Otherwise, the layer
  dimension is prepended to the shape.

  Args:
    shape: The shape to add the layer dimension to.
    num_layers: The number of layers.

  Returns:
    The shape with the layer dimension added.
  """
  if isinstance(shape, int):
    shape = (shape,)
  return (num_layers,) + shape if num_layers > 1 else shape


def init_dsv3_mla_weights(
    rng: jax.Array,
    emb_dim: int,
    cq_dim: int,
    ckv_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MLAWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    scale_init: Initializer = default_scale_init,
) -> DSv3MLAWeightsPytree[ArrayType]:
  """Initializes the MLA weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    cq_dim: the compressed query dimension.
    ckv_dim: the compressed key/value dimension.
    num_query_heads: the number of query heads.
    num_kv_heads: the number of key/value heads.
    rope_head_dim: the RoPE dimension for each head.
    qk_head_dim: the query/key dimension for each head (without RoPE).
    v_head_dim: the value dimension for each head.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    scale_init: optional, custom scale initializer.

  Returns:
    The initialized MLA weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3MLAWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 7)
    k_init = functools.partial(kernel_init, dtype=weight_dtype)
    s_init = functools.partial(scale_init, dtype=weight_dtype)
    q_up = k_init(
        subkeys[1],
        add_layer_dim((cq_dim, num_query_heads, qk_head_dim + rope_head_dim)),
        out_sharding=out_shardings.q_up,
    )
    q_nope = q_up[..., :qk_head_dim]
    q_rope = q_up[..., qk_head_dim:]
    q_rope_perm = jnp.concatenate([q_rope[..., 0::2], q_rope[..., 1::2]], axis=-1)
    q_up = jnp.concatenate([q_nope, q_rope_perm], axis=-1)

    kv_down = k_init(
        subkeys[3],
        add_layer_dim((emb_dim, ckv_dim + rope_head_dim)),
        out_sharding=out_shardings.kv_down,
    )
    kv_comp = kv_down[..., :ckv_dim]
    k_rope = kv_down[..., ckv_dim:]
    k_rope_perm = jnp.concatenate([k_rope[..., 0::2], k_rope[..., 1::2]], axis=-1)
    kv_down = jnp.concatenate([kv_comp, k_rope_perm], axis=-1)

    kv_up = k_init(
        subkeys[4],
        add_layer_dim((ckv_dim, num_kv_heads, qk_head_dim + v_head_dim)),
    )
    k_up = kv_up[..., :qk_head_dim]
    v_up = kv_up[..., qk_head_dim:]

    return DSv3MLAWeightsPytree[ArrayType](
        q_down=k_init(
            subkeys[0],
            add_layer_dim((emb_dim, cq_dim)),
            out_sharding=out_shardings.q_down,
        ),
        q_up=q_up,
        q_norm_scale=s_init(
            subkeys[2],
            add_layer_dim(cq_dim),
            out_sharding=out_shardings.q_norm_scale,
        ),
        kv_down=kv_down,
        k_up=k_up,
        v_up=v_up,
        kv_norm_scale=s_init(
            subkeys[5],
            add_layer_dim(ckv_dim),
            out_sharding=out_shardings.kv_norm_scale,
        ),
        out=k_init(
            subkeys[6],
            add_layer_dim((num_query_heads, v_head_dim, emb_dim)),
            out_sharding=out_shardings.out,
        ),
    )

  return _init(rng)


def init_dsv3_moe_router_weights(
    rng: jax.Array,
    emb_dim: int,
    num_experts: int,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MoERouterWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
) -> DSv3MoERouterWeightsPytree[ArrayType]:
  """Initializes the router weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    num_experts: the number of experts.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    bias_init: optional, custom bias initializer.

  Returns:
    The initialized router weights.
  """
  out_shardings = out_shardings if out_shardings else DSv3MoERouterWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 2)
    k_init = functools.partial(kernel_init, dtype=weight_dtype)
    b_init = functools.partial(bias_init, dtype=weight_dtype)
    return DSv3MoERouterWeightsPytree[ArrayType](
        kernel=k_init(
            subkeys[0],
            add_layer_dim((emb_dim, num_experts)),
            out_sharding=out_shardings.kernel,
        ),
        bias=b_init(
            subkeys[1],
            add_layer_dim(num_experts),
            out_sharding=out_shardings.bias,
        ),
    )

  return _init(rng)


def init_dsv3_moe_shared_expert_weights(
    rng: jax.Array,
    emb_dim: int,
    expert_hidden_dim: int,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MoESharedExpertWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
) -> DSv3MoESharedExpertWeightsPytree[ArrayType]:
  """Initializes the shared expert weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    expert_hidden_dim: the hidden dimension of each expert.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.

  Returns:
    The initialized shared expert weights.
  """
  out_shardings = out_shardings if out_shardings else DSv3MoESharedExpertWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 3)
    k_init = functools.partial(kernel_init, dtype=weight_dtype)
    return DSv3MoESharedExpertWeightsPytree[ArrayType](
        gate_0=k_init(
            subkeys[0],
            add_layer_dim((emb_dim, expert_hidden_dim)),
            out_sharding=out_shardings.gate_0,
        ),
        gate_1=k_init(
            subkeys[1],
            add_layer_dim((emb_dim, expert_hidden_dim)),
            out_sharding=out_shardings.gate_1,
        ),
        linear=k_init(
            subkeys[2],
            add_layer_dim((expert_hidden_dim, emb_dim)),
            out_sharding=out_shardings.linear,
        ),
    )

  return _init(rng)


def init_dsv3_moe_routed_expert_weights(
    rng: jax.Array,
    num_experts: int,
    emb_dim: int,
    expert_hidden_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MoERoutedExpertWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
) -> DSv3MoERoutedExpertWeightsPytree[ArrayType]:
  """Initializes the routed expert weights.

  Args:
    rng: a PRNG key.
    num_experts: the number of experts.
    emb_dim: the embedding dimension.
    expert_hidden_dim: the hidden dimension of each expert.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.

  Returns:
    The initialized routed expert weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3MoERoutedExpertWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 3)
    k_init = functools.partial(kernel_init, dtype=weight_dtype)
    gate_0 = k_init(
        subkeys[0],
        add_layer_dim((num_experts, emb_dim, expert_hidden_dim)),
        out_sharding=out_shardings.gate,
    )
    gate_1 = k_init(
        subkeys[1],
        add_layer_dim((num_experts, emb_dim, expert_hidden_dim)),
        out_sharding=out_shardings.gate,
    )
    return DSv3MoERoutedExpertWeightsPytree[ArrayType](
        gate=jnp.concatenate([gate_0, gate_1], axis=-1),
        linear=k_init(
            subkeys[2],
            add_layer_dim((num_experts, expert_hidden_dim, emb_dim)),
            out_sharding=out_shardings.linear,
        ),
    )

  return _init(rng)


def init_dsv3_moe_weights(
    rng: jax.Array,
    num_experts: int,
    emb_dim: int,
    expert_hidden_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MoEWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
) -> DSv3MoEWeightsPytree[ArrayType]:
  """Initializes the MoE weights.

  Args:
    rng: a PRNG key.
    num_experts: the number of experts.
    emb_dim: the embedding dimension.
    expert_hidden_dim: the hidden dimension of each expert.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    bias_init: optional, custom bias initializer.

  Returns:
    The initialized MoE weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  subkeys = jax.random.split(rng, 3)
  out_shardings = out_shardings if out_shardings else DSv3MoEWeightsPytree[ShardingType]()
  return DSv3MoEWeightsPytree[ArrayType](
      router=init_dsv3_moe_router_weights(
          subkeys[0],
          emb_dim,
          num_experts,
          weight_dtype,
          num_layers,
          out_shardings=out_shardings.router,
          kernel_init=kernel_init,
          bias_init=bias_init,
      ),
      shared=init_dsv3_moe_shared_expert_weights(
          subkeys[1],
          emb_dim,
          expert_hidden_dim,
          weight_dtype,
          num_layers,
          out_shardings=out_shardings.shared,
          kernel_init=kernel_init,
      ),
      routed=init_dsv3_moe_routed_expert_weights(
          subkeys[2],
          num_experts,
          emb_dim,
          expert_hidden_dim,
          mesh,
          axis_mapping,
          weight_dtype=weight_dtype,
          num_layers=num_layers,
          out_shardings=out_shardings.routed,
          kernel_init=kernel_init,
      ),
  )


def init_dsv3_sparse_layer_weights(
    rng: jax.Array,
    emb_dim: int,
    cq_dim: int,
    ckv_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_experts: int,
    expert_hidden_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3SparseLayerWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    scale_init: Initializer = default_scale_init,
    bias_init: Initializer = default_bias_init,
) -> DSv3SparseLayerWeightsPytree[ArrayType]:
  """Initializes the sparse layer weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    cq_dim: the compressed query dimension.
    ckv_dim: the compressed key/value dimension.
    num_query_heads: the number of query heads.
    num_kv_heads: the number of key/value heads.
    rope_head_dim: the RoPE dimension for each head.
    qk_head_dim: the query/key dimension for each head (without RoPE).
    v_head_dim: the value dimension for each head.
    num_experts: the number of experts.
    expert_hidden_dim: the hidden dimension of each expert.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    scale_init: optional, custom scale initializer.
    bias_init: optional, custom bias initializer.

  Returns:
    The initialized sparse layer weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3SparseLayerWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 4)
    s_init = functools.partial(scale_init, dtype=weight_dtype)
    return DSv3SparseLayerWeightsPytree[ArrayType](
        pre_attn_norm_scale=s_init(
            subkeys[0],
            add_layer_dim(emb_dim),
            out_sharding=out_shardings.pre_attn_norm_scale,
        ),
        mla=init_dsv3_mla_weights(
            subkeys[1],
            emb_dim,
            cq_dim,
            ckv_dim,
            num_query_heads,
            num_kv_heads,
            rope_head_dim,
            qk_head_dim,
            v_head_dim,
            mesh,
            axis_mapping,
            weight_dtype=weight_dtype,
            num_layers=num_layers,
            out_shardings=out_shardings.mla,
            kernel_init=kernel_init,
            scale_init=scale_init,
        ),
        post_attn_norm_scale=s_init(
            subkeys[2],
            add_layer_dim(emb_dim),
            out_sharding=out_shardings.post_attn_norm_scale,
        ),
        moe=init_dsv3_moe_weights(
            subkeys[3],
            num_experts,
            emb_dim,
            expert_hidden_dim,
            mesh,
            axis_mapping,
            weight_dtype=weight_dtype,
            num_layers=num_layers,
            out_shardings=out_shardings.moe,
            kernel_init=kernel_init,
            bias_init=bias_init,
        ),
    )

  return _init(rng)


def init_dsv3_mlp_weights(
    rng: jax.Array,
    emb_dim: int,
    mlp_dim: int,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MLPWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
) -> DSv3MLPWeightsPytree[ArrayType]:
  """Initializes the MLP weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    mlp_dim: the hidden dimension of each MLP.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.

  Returns:
    The initialized MLP weights.
  """
  out_shardings = out_shardings if out_shardings else DSv3MLPWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 3)
    k_init = functools.partial(kernel_init, dtype=weight_dtype)
    return DSv3MLPWeightsPytree[ArrayType](
        gate_0=k_init(
            subkeys[0],
            add_layer_dim((emb_dim, mlp_dim)),
            out_sharding=out_shardings.gate_0,
        ),
        gate_1=k_init(
            subkeys[1],
            add_layer_dim((emb_dim, mlp_dim)),
            out_sharding=out_shardings.gate_1,
        ),
        linear=k_init(
            subkeys[2],
            add_layer_dim((mlp_dim, emb_dim)),
            out_sharding=out_shardings.linear,
        ),
    )

  return _init(rng)


def init_dsv3_dense_layer_weights(
    rng: jax.Array,
    emb_dim: int,
    cq_dim: int,
    ckv_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    mlp_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3DenseLayerWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    scale_init: Initializer = default_scale_init,
) -> DSv3DenseLayerWeightsPytree[ArrayType]:
  """Initializes the dense layer weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    cq_dim: the compressed query dimension.
    ckv_dim: the compressed key/value dimension.
    num_query_heads: the number of query heads.
    num_kv_heads: the number of key/value heads.
    rope_head_dim: the RoPE dimension for each head.
    qk_head_dim: the query/key dimension for each head (without RoPE).
    v_head_dim: the value dimension for each head.
    mlp_dim: the hidden dimension of each MLP.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    scale_init: optional, custom scale initializer.

  Returns:
    The initialized dense layer weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3DenseLayerWeightsPytree[ShardingType]()

  @functools.partial(
      jax.jit,
      out_shardings=out_shardings,
  )
  def _init(rng):
    add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
    subkeys = jax.random.split(rng, 4)
    s_init = functools.partial(scale_init, dtype=weight_dtype)
    return DSv3DenseLayerWeightsPytree[ArrayType](
        pre_attn_norm_scale=s_init(
            subkeys[0],
            add_layer_dim(emb_dim),
            out_sharding=out_shardings.pre_attn_norm_scale,
        ),
        mla=init_dsv3_mla_weights(
            subkeys[1],
            emb_dim,
            cq_dim,
            ckv_dim,
            num_query_heads,
            num_kv_heads,
            rope_head_dim,
            qk_head_dim,
            v_head_dim,
            mesh,
            axis_mapping,
            weight_dtype=weight_dtype,
            num_layers=num_layers,
            out_shardings=out_shardings.mla,
            kernel_init=kernel_init,
            scale_init=scale_init,
        ),
        post_attn_norm_scale=s_init(
            subkeys[2],
            add_layer_dim(emb_dim),
            out_sharding=out_shardings.post_attn_norm_scale,
        ),
        mlp=init_dsv3_mlp_weights(
            subkeys[3],
            emb_dim,
            mlp_dim,
            weight_dtype,
            num_layers,
            out_shardings=out_shardings.mlp,
            kernel_init=kernel_init,
        ),
    )

  return _init(rng)


def init_dsv3_weights(
    rng: jax.Array,
    emb_dim: int,
    cq_dim: int,
    ckv_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    mlp_dim: int,
    num_experts: int,
    expert_hidden_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_dense_layers: int = 3,
    num_sparse_layers: int = 58,
    out_shardings: DSv3WeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    scale_init: Initializer = default_scale_init,
    bias_init: Initializer = default_bias_init,
) -> DSv3WeightsPytree[ArrayType]:
  """Initializes full DSv3 model weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    cq_dim: the compressed query dimension.
    ckv_dim: the compressed key/value dimension.
    num_query_heads: the number of query heads.
    num_kv_heads: the number of key/value heads.
    rope_head_dim: the RoPE dimension for each head.
    qk_head_dim: the query/key dimension for each head (without RoPE).
    v_head_dim: the value dimension for each head.
    mlp_dim: the hidden dimension of dense MLP.
    num_experts: the number of experts.
    expert_hidden_dim: the hidden dimension of each expert.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_dense_layers: number of dense layers to stack weights for.
    num_sparse_layers: number of sparse layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    scale_init: optional, custom scale initializer.
    bias_init: optional, custom bias initializer.

  Returns:
    The initialized full DSv3 model weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3WeightsPytree[ShardingType]()
  subkeys = jax.random.split(rng, 2)
  return DSv3WeightsPytree[ArrayType](
      dense=init_dsv3_dense_layer_weights(
          subkeys[0],
          emb_dim,
          cq_dim,
          ckv_dim,
          num_query_heads,
          num_kv_heads,
          rope_head_dim,
          qk_head_dim,
          v_head_dim,
          mlp_dim,
          mesh,
          axis_mapping,
          weight_dtype=weight_dtype,
          num_layers=num_dense_layers,
          out_shardings=out_shardings.dense,
          kernel_init=kernel_init,
          scale_init=scale_init,
      ),
      sparse=init_dsv3_sparse_layer_weights(
          subkeys[1],
          emb_dim,
          cq_dim,
          ckv_dim,
          num_query_heads,
          num_kv_heads,
          rope_head_dim,
          qk_head_dim,
          v_head_dim,
          num_experts,
          expert_hidden_dim,
          mesh,
          axis_mapping,
          weight_dtype=weight_dtype,
          num_layers=num_sparse_layers,
          out_shardings=out_shardings.sparse,
          kernel_init=kernel_init,
          scale_init=scale_init,
          bias_init=bias_init,
      ),
  )


def init_dsv3_ehproj_weights(
    rng: jax.Array,
    emb_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3EHProjWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    scale_init: Initializer = default_scale_init,
) -> DSv3EHProjWeightsPytree[ArrayType]:
  """Initializes the EHProj weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    scale_init: optional, custom scale initializer.

  Returns:
    The initialized EHProj weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3EHProjWeightsPytree[ShardingType]()
  subkeys = jax.random.split(rng, 3)
  add_layer_dim = functools.partial(_add_layer_dim, num_layers=num_layers)
  s_init = functools.partial(scale_init, dtype=weight_dtype)
  k_init = functools.partial(kernel_init, dtype=weight_dtype)
  return DSv3EHProjWeightsPytree[ArrayType](
      enorm_scale=s_init(
          subkeys[0],
          add_layer_dim(emb_dim),
          out_sharding=out_shardings.enorm_scale,
      ),
      hnorm_scale=s_init(
          subkeys[1],
          add_layer_dim(emb_dim),
          out_sharding=out_shardings.hnorm_scale,
      ),
      eh_proj=k_init(
          subkeys[2],
          add_layer_dim((2 * emb_dim, emb_dim)),
          out_sharding=out_shardings.eh_proj,
      ),
  )


def init_dsv3_mtp_weights(
    rng: jax.Array,
    emb_dim: int,
    cq_dim: int,
    ckv_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_experts: int,
    expert_hidden_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jnp.dtype = jnp.float32,
    num_layers: int = 1,
    out_shardings: DSv3MTPWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_kernel_init,
    scale_init: Initializer = default_scale_init,
    bias_init: Initializer = default_bias_init,
) -> DSv3MTPWeightsPytree[ArrayType]:
  """Initializes the MTP layer weights.

  Args:
    rng: a PRNG key.
    emb_dim: the embedding dimension.
    cq_dim: the compressed query dimension.
    ckv_dim: the compressed key/value dimension.
    num_query_heads: the number of query heads.
    num_kv_heads: the number of key/value heads.
    rope_head_dim: the RoPE dimension for each head.
    qk_head_dim: the query/key dimension for each head (without RoPE).
    v_head_dim: the value dimension for each head.
    num_experts: the number of experts.
    expert_hidden_dim: the hidden dimension of each expert.
    mesh: physical mesh for logical sharding resolution.
    axis_mapping: mapping from logical to physical mesh axes.
    weight_dtype: the dtype of the weights.
    num_layers: number of layers to stack weights for.
    out_shardings: optional, output shardings for the weights.
    kernel_init: optional, custom kernel initializer.
    scale_init: optional, custom scale initializer.
    bias_init: optional, custom bias initializer.

  Returns:
    The initialized MTP layer weights.
  """
  out_shardings = _resolve_shardings(out_shardings, mesh, axis_mapping)
  out_shardings = out_shardings if out_shardings else DSv3MTPWeightsPytree[ShardingType]()
  subkeys = jax.random.split(rng, 2)
  return DSv3MTPWeightsPytree[ArrayType](
      ehproj=init_dsv3_ehproj_weights(
          subkeys[0],
          emb_dim,
          mesh,
          axis_mapping,
          weight_dtype=weight_dtype,
          num_layers=num_layers,
          out_shardings=out_shardings.ehproj,
          kernel_init=kernel_init,
          scale_init=scale_init,
      ),
      sparse=init_dsv3_sparse_layer_weights(
          subkeys[1],
          emb_dim,
          cq_dim,
          ckv_dim,
          num_query_heads,
          num_kv_heads,
          rope_head_dim,
          qk_head_dim,
          v_head_dim,
          num_experts,
          expert_hidden_dim,
          mesh,
          axis_mapping,
          weight_dtype=weight_dtype,
          num_layers=num_layers,
          out_shardings=out_shardings.sparse,
          kernel_init=kernel_init,
          scale_init=scale_init,
          bias_init=bias_init,
      ),
  )
