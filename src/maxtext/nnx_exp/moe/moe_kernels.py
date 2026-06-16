"""Sparse MoE grouped-matmul kernel adapters."""

import inspect
from typing import Any, Literal, TypeAlias

import jax
from jax.experimental.pallas.ops.tpu import megablox
import jax.numpy as jnp

from maxtext.nnx_exp.moe.moe_types import MegabloxConfig


SparseMoEKernel: TypeAlias = Literal["ragged_dot", "megablox"]


def spec_of(x: jax.Array) -> jax.sharding.PartitionSpec | None:
    if hasattr(x, "sharding"):
        sharding: Any = x.sharding
        return sharding.spec if hasattr(sharding, "spec") else sharding
    try:
        sharding: Any = jax.typeof(x).sharding
        return sharding.spec if hasattr(sharding, "spec") else sharding
    except Exception:
        return None


def abstract_mesh():
    try:
        return jax.sharding.get_abstract_mesh()
    except Exception:
        return None


def shard_map_options() -> dict[str, Any]:
    params = inspect.signature(jax.shard_map).parameters
    if "check_vma" in params:
        return {"check_vma": False}
    if "check_rep" in params:
        return {"check_rep": False}
    return {}


def has_nontrivial_mesh(mesh) -> bool:
    if mesh is None:
        return False
    axis_sizes = tuple(getattr(mesh, "shape", {}).values())
    return any(size > 1 for size in axis_sizes)


def _megablox_out_spec(
    lhs: jax.Array,
    rhs: jax.Array,
) -> jax.sharding.PartitionSpec | None:
    lhs_spec = spec_of(lhs)
    rhs_spec = spec_of(rhs)
    if lhs_spec is None or rhs_spec is None:
        return None
    if len(lhs_spec) < 1 or len(rhs_spec) < 3:
        return None
    return jax.P(lhs_spec[0], rhs_spec[2])


def _megablox_grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_dtype,
    megablox_config: MegabloxConfig,
    manual_mode: bool = False,
) -> jax.Array:
    interpret = megablox_config.use_interpret()

    def megablox_tiling(local_lhs: jax.Array, local_rhs: jax.Array) -> tuple[int, int, int]:
        if jax.default_backend() == "tpu" and not interpret:
            # TPU Megablox validates block shapes against the local per-device
            # arrays that reach the kernel body, so under shard_map we pick
            # the full local tile rather than the global sharded size.
            return (local_lhs.shape[0], local_lhs.shape[1], local_rhs.shape[-1])
        return (
            min(megablox_config.tile_batch_seq, local_lhs.shape[0]),
            min(megablox_config.tile_activation_dim, local_lhs.shape[1]),
            min(megablox_config.tile_weight_dim, local_rhs.shape[-1]),
        )

    tiling = megablox_tiling(lhs, rhs)
    if jax.default_backend() != "tpu" or interpret:
        return megablox.gmm(
            lhs,
            rhs,
            group_sizes.astype(jnp.int32),
            preferred_element_type=preferred_dtype,
            tiling=tiling,
            interpret=interpret,
        )
    if manual_mode:
        return megablox.gmm(
            lhs,
            rhs,
            group_sizes.astype(jnp.int32),
            preferred_element_type=preferred_dtype,
            tiling=tiling,
            interpret=False,
        )

    mesh = abstract_mesh()
    out_spec = _megablox_out_spec(lhs, rhs)
    lhs_spec = spec_of(lhs)
    rhs_spec = spec_of(rhs)
    group_sizes_spec = spec_of(group_sizes) or jax.P(None)
    if not has_nontrivial_mesh(mesh) or lhs_spec is None or rhs_spec is None or out_spec is None:
        return megablox.gmm(
            lhs,
            rhs,
            group_sizes.astype(jnp.int32),
            preferred_element_type=preferred_dtype,
            tiling=tiling,
            interpret=False,
        )

    # TPU Megablox lowers to a Mosaic/Pallas local kernel, so under explicit
    # sharding we have to switch just this grouped-matmul boundary into manual
    # mode rather than relying on automatic partitioning.
    return jax.shard_map(
        lambda lhs, rhs, group_sizes: megablox.gmm(
            lhs,
            rhs,
            group_sizes.astype(jnp.int32),
            preferred_element_type=preferred_dtype,
            tiling=megablox_tiling(lhs, rhs),
            interpret=False,
        ),
        mesh=mesh,
        in_specs=(lhs_spec, rhs_spec, group_sizes_spec),
        out_specs=out_spec,
        **shard_map_options(),
    )(lhs, rhs, group_sizes)


def grouped_matmul(
    executor: SparseMoEKernel,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    megablox_config: MegabloxConfig,
    manual_mode: bool = False,
) -> jax.Array:
    preferred_dtype = lhs.dtype if lhs.dtype == jnp.float32 else jnp.float32
    match executor:
        case "ragged_dot":
            return jax.lax.ragged_dot(
                lhs,
                rhs,
                group_sizes.astype(jnp.int32),
                preferred_element_type=preferred_dtype,
            )
        case "megablox":
            return _megablox_grouped_matmul(
                lhs,
                rhs,
                group_sizes,
                preferred_dtype=preferred_dtype,
                megablox_config=megablox_config,
                manual_mode=manual_mode,
            )
        case _:
            raise ValueError(f"Unsupported sparse MoE kernel: {executor}")
