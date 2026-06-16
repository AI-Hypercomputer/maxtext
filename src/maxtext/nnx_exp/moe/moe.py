"""Shared MoE dispatch planning and expert execution runtime."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import cast

import jax
import jax.numpy as jnp

from maxtext.nnx_exp.moe.moe_types import (
    MegabloxConfig,
    MoEExecutor,
    MoERuntimeConfig,
    RoutingMode,
)
from maxtext.nnx_exp.moe.moe_kernels import (
    SparseMoEKernel,
    abstract_mesh,
    grouped_matmul,
    has_nontrivial_mesh,
    shard_map_options,
    spec_of,
)

@dataclass(frozen=True)
class _DispatchPlan:
    mode: RoutingMode
    weights: jax.Array
    indices: jax.Array
    kept_mask: jax.Array
    combine_weights: jax.Array
    token_ids: jax.Array
    packed_weights: jax.Array
    packed_mask: jax.Array
    group_sizes: jax.Array
    expert_capacity: int | None
    dropped_assignments: jax.Array


def routing_mode(capacity_factor: float) -> RoutingMode:
    if capacity_factor > 0:
        return "capped"
    return "dropless"


def _combine_weights(
    weights: jax.Array,
    indices: jax.Array,
    *,
    num_experts: int,
    out_sharding=None,
) -> jax.Array:
    batch, length, _ = indices.shape
    combine = jnp.zeros((batch, length, num_experts), dtype=weights.dtype)
    batch_i = jnp.arange(batch)[:, None, None]
    length_i = jnp.arange(length)[None, :, None]
    combine = combine.at[batch_i, length_i, indices].add(weights, out_sharding=out_sharding)
    return combine


def _empty_sparse_fields(
    num_experts: int,
    dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=dtype),
        jnp.zeros((0,), dtype=dtype),
        jnp.zeros((num_experts,), dtype=jnp.int32),
    )


def _dropless_sparse_fields(
    weights: jax.Array,
    indices: jax.Array,
    *,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    batch, length, top_k = indices.shape
    num_tokens = batch * length
    flat_indices = jnp.reshape(indices, (-1,))
    flat_weights = jnp.reshape(weights, (-1,))
    token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)

    order = jnp.argsort(flat_indices, stable=True)
    sorted_indices = flat_indices[order]
    sorted_token_ids = token_ids[order]
    sorted_weights = flat_weights[order]
    group_sizes = jnp.bincount(sorted_indices, length=num_experts).astype(jnp.int32)
    packed_mask = jnp.ones_like(sorted_weights, dtype=weights.dtype)
    return sorted_token_ids, sorted_weights, packed_mask, group_sizes


def _expert_capacity(
    *,
    batch: int,
    length: int,
    top_k: int,
    num_experts: int,
    capacity_factor: float,
) -> int:
    tokens_per_batch = length * top_k
    return int(
        max(
            math.ceil(tokens_per_batch / num_experts) * capacity_factor,
            capacity_factor,
        )
    )


def _capped_assignments(
    weights: jax.Array,
    indices: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
) -> tuple[jax.Array, jax.Array, int, jax.Array]:
    batch, length, top_k = indices.shape
    expert_capacity = _expert_capacity(
        batch=batch,
        length=length,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    expert_mask = jax.nn.one_hot(indices, num_classes=num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(expert_mask, (batch, length * top_k, num_experts))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        (batch, length, top_k, num_experts),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity)
    kept_mask = jnp.sum(trunc_expert_mask, axis=-1).astype(bool)
    kept_weights = weights * kept_mask.astype(weights.dtype)
    dropped_assignments = jnp.size(kept_mask) - jnp.sum(kept_mask.astype(jnp.int32))
    return kept_weights, kept_mask, expert_capacity, dropped_assignments


def _capped_sparse_fields(
    kept_weights: jax.Array,
    kept_mask: jax.Array,
    indices: jax.Array,
    *,
    num_experts: int,
    expert_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    batch, length, top_k = indices.shape
    num_tokens = batch * length
    num_rows = num_experts * expert_capacity
    token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)

    expert_mask = jax.nn.one_hot(indices, num_classes=num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(expert_mask, (batch, length * top_k, num_experts))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        (batch, length, top_k, num_experts),
    )
    expert_slot = jnp.sum(expert_mask * expert_token_count, axis=-1) - 1
    expert_slot = jnp.maximum(expert_slot, 0)
    row_ids = jnp.reshape(indices * expert_capacity + expert_slot, (-1,))
    flat_kept = jnp.reshape(kept_mask, (-1,))
    flat_kept_i32 = flat_kept.astype(jnp.int32)
    flat_kept_w = flat_kept.astype(kept_weights.dtype)

    packed_token_ids = jnp.zeros((num_rows,), dtype=jnp.int32)
    packed_token_ids = packed_token_ids.at[row_ids].add(token_ids * flat_kept_i32)

    packed_weights = jnp.zeros((num_rows,), dtype=kept_weights.dtype)
    packed_weights = packed_weights.at[row_ids].add(jnp.reshape(kept_weights, (-1,)))

    packed_mask = jnp.zeros((num_rows,), dtype=kept_weights.dtype)
    packed_mask = packed_mask.at[row_ids].add(flat_kept_w)
    group_sizes = jnp.full((num_experts,), expert_capacity, dtype=jnp.int32)
    return packed_token_ids, packed_weights, packed_mask, group_sizes


def make_dense_dispatch_plan(
    weights: jax.Array,
    indices: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
    combine_sharding=None,
) -> _DispatchPlan:
    token_ids, packed_weights, packed_mask, group_sizes = _empty_sparse_fields(num_experts, weights.dtype)

    match routing_mode(capacity_factor):
        case "dropless":
            kept_mask = jnp.ones_like(weights, dtype=bool)
            return _DispatchPlan(
                mode="dropless",
                weights=weights,
                indices=indices,
                kept_mask=kept_mask,
                combine_weights=_combine_weights(
                    weights,
                    indices,
                    num_experts=num_experts,
                    out_sharding=combine_sharding,
                ),
                token_ids=token_ids,
                packed_weights=packed_weights,
                packed_mask=packed_mask,
                group_sizes=group_sizes,
                expert_capacity=None,
                dropped_assignments=jnp.array(0, dtype=jnp.int32),
            )
        case "capped":
            kept_weights, kept_mask, expert_capacity, dropped_assignments = _capped_assignments(
                weights,
                indices,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )
            return _DispatchPlan(
                mode="capped",
                weights=kept_weights,
                indices=indices,
                kept_mask=kept_mask,
                combine_weights=_combine_weights(
                    kept_weights,
                    indices,
                    num_experts=num_experts,
                    out_sharding=combine_sharding,
                ),
                token_ids=token_ids,
                packed_weights=packed_weights,
                packed_mask=packed_mask,
                group_sizes=group_sizes,
                expert_capacity=expert_capacity,
                dropped_assignments=dropped_assignments,
            )


def make_sparse_dispatch_plan(
    weights: jax.Array,
    indices: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
    combine_sharding=None,
) -> _DispatchPlan:
    match routing_mode(capacity_factor):
        case "dropless":
            token_ids, packed_weights, packed_mask, group_sizes = cast(
                Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
                jax.sharding.auto_axes(
                    lambda weights, indices: _dropless_sparse_fields(
                        weights,
                        indices,
                        num_experts=num_experts,
                    ),
                    out_sharding=(jax.P(None), jax.P(None), jax.P(None), jax.P(None)),
                ),
            )(weights, indices)
            kept_mask = jnp.ones_like(weights, dtype=bool)

            return _DispatchPlan(
                mode="dropless",
                weights=weights,
                indices=indices,
                kept_mask=kept_mask,
                combine_weights=_combine_weights(
                    weights,
                    indices,
                    num_experts=num_experts,
                    out_sharding=combine_sharding,
                ),
                token_ids=token_ids,
                packed_weights=packed_weights,
                packed_mask=packed_mask,
                group_sizes=group_sizes,
                expert_capacity=None,
                dropped_assignments=jnp.array(0, dtype=jnp.int32),
            )
        case "capped":
            kept_weights, kept_mask, expert_capacity, dropped_assignments = _capped_assignments(
                weights,
                indices,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )
            token_ids, packed_weights, packed_mask, group_sizes = cast(
                Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
                jax.sharding.auto_axes(
                    lambda kept_weights, kept_mask, indices: _capped_sparse_fields(
                        kept_weights,
                        kept_mask,
                        indices,
                        num_experts=num_experts,
                        expert_capacity=expert_capacity,
                    ),
                    out_sharding=(jax.P(None), jax.P(None), jax.P(None), jax.P(None)),
                ),
            )(kept_weights, kept_mask, indices)

            return _DispatchPlan(
                mode="capped",
                weights=kept_weights,
                indices=indices,
                kept_mask=kept_mask,
                combine_weights=_combine_weights(
                    kept_weights,
                    indices,
                    num_experts=num_experts,
                    out_sharding=combine_sharding,
                ),
                token_ids=token_ids,
                packed_weights=packed_weights,
                packed_mask=packed_mask,
                group_sizes=group_sizes,
                expert_capacity=expert_capacity,
                dropped_assignments=dropped_assignments,
            )


def make_dispatch_plan(
    weights: jax.Array,
    indices: jax.Array,
    *,
    executor: MoEExecutor,
    num_experts: int,
    capacity_factor: float,
    combine_sharding=None,
) -> _DispatchPlan:
    match executor:
        case "dense":
            return make_dense_dispatch_plan(
                weights,
                indices,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                combine_sharding=combine_sharding,
            )
        case "ragged_dot" | "megablox":
            return make_sparse_dispatch_plan(
                weights,
                indices,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                combine_sharding=combine_sharding,
            )
        case _:
            raise ValueError(f"Unknown MoE executor: {executor}")


def _dense_moe(
    x: jax.Array,
    plan: _DispatchPlan,
    gate_kernels: jax.Array,
    up_kernels: jax.Array,
    down_kernels: jax.Array,
    *,
    hidden_sharding=None,
    outputs_sharding=None,
    out_sharding=None,
) -> jax.Array:
    gate = jnp.einsum(
        "bse,neh->bsnh",
        x,
        gate_kernels,
        out_sharding=hidden_sharding,
    )
    up = jnp.einsum(
        "bse,neh->bsnh",
        x,
        up_kernels,
        out_sharding=hidden_sharding,
    )
    hidden = jax.nn.silu(gate) * up
    all_expert_out = jnp.einsum(
        "bsnh,nhe->bsne",
        hidden,
        down_kernels,
        out_sharding=outputs_sharding,
    )
    return jnp.einsum(
        "bsn,bsne->bse",
        plan.combine_weights,
        all_expert_out,
        out_sharding=out_sharding,
    )


def _sparse_moe_impl(
    x: jax.Array,
    plan: _DispatchPlan,
    gate_kernels: jax.Array,
    up_kernels: jax.Array,
    down_kernels: jax.Array,
    *,
    executor: SparseMoEKernel,
    megablox_config: MegabloxConfig,
    manual_mode: bool = False,
) -> jax.Array:
    flat_x = jnp.reshape(x, (-1, x.shape[-1]))
    packed_inputs = flat_x[plan.token_ids] * plan.packed_mask[:, None].astype(flat_x.dtype)
    gate = grouped_matmul(
        executor,
        packed_inputs,
        gate_kernels,
        plan.group_sizes,
        megablox_config=megablox_config,
        manual_mode=manual_mode,
    )
    up = grouped_matmul(
        executor,
        packed_inputs,
        up_kernels,
        plan.group_sizes,
        megablox_config=megablox_config,
        manual_mode=manual_mode,
    )
    hidden = jax.nn.silu(gate) * up
    out = grouped_matmul(
        executor,
        hidden,
        down_kernels,
        plan.group_sizes,
        megablox_config=megablox_config,
        manual_mode=manual_mode,
    )
    out = out * plan.packed_weights[:, None] * plan.packed_mask[:, None].astype(out.dtype)

    flat_out = jnp.zeros((flat_x.shape[0], down_kernels.shape[-1]), dtype=out.dtype)
    flat_out = flat_out.at[plan.token_ids].add(out)
    return jnp.reshape(flat_out, x.shape[:-1] + (down_kernels.shape[-1],)).astype(x.dtype)


def _sparse_moe(
    x: jax.Array,
    plan: _DispatchPlan,
    gate_kernels: jax.Array,
    up_kernels: jax.Array,
    down_kernels: jax.Array,
    *,
    executor: SparseMoEKernel,
    out_sharding=None,
    megablox_config: MegabloxConfig,
) -> jax.Array:
    mesh = abstract_mesh()
    x_spec = spec_of(x)
    gate_spec = spec_of(gate_kernels)
    up_spec = spec_of(up_kernels)
    down_spec = spec_of(down_kernels)
    token_ids_spec = spec_of(plan.token_ids) or jax.P(None)
    packed_weights_spec = spec_of(plan.packed_weights) or jax.P(None)
    packed_mask_spec = spec_of(plan.packed_mask) or jax.P(None)
    group_sizes_spec = spec_of(plan.group_sizes) or jax.P(None)
    manual_sparse_megablox = (
        executor == "megablox"
        and jax.default_backend() == "tpu"
        and not megablox_config.use_interpret()
        and has_nontrivial_mesh(mesh)
        and x_spec is not None
        and gate_spec is not None
        and up_spec is not None
        and down_spec is not None
        and out_sharding is not None
    )
    if manual_sparse_megablox:
        wrapped = jax.shard_map(
            lambda x, token_ids, packed_weights, packed_mask, group_sizes, gate, up, down: _sparse_moe_impl(
                x,
                _DispatchPlan(
                    mode=plan.mode,
                    weights=plan.weights,
                    indices=plan.indices,
                    kept_mask=plan.kept_mask,
                    combine_weights=plan.combine_weights,
                    token_ids=token_ids,
                    packed_weights=packed_weights,
                    packed_mask=packed_mask,
                    group_sizes=group_sizes,
                    expert_capacity=plan.expert_capacity,
                    dropped_assignments=plan.dropped_assignments,
                ),
                gate,
                up,
                down,
                executor=executor,
                megablox_config=megablox_config,
                manual_mode=True,
            ),
            mesh=mesh,
            in_specs=(
                x_spec,
                token_ids_spec,
                packed_weights_spec,
                packed_mask_spec,
                group_sizes_spec,
                gate_spec,
                up_spec,
                down_spec,
            ),
            out_specs=out_sharding,
            **shard_map_options(),
        )
        return wrapped(
            x,
            plan.token_ids,
            plan.packed_weights,
            plan.packed_mask,
            plan.group_sizes,
            gate_kernels,
            up_kernels,
            down_kernels,
        )

    wrapped = cast(
        Callable[..., jax.Array],
        jax.sharding.auto_axes(
            lambda x, token_ids, packed_weights, packed_mask, group_sizes, gate, up, down: _sparse_moe_impl(
                x,
                _DispatchPlan(
                    mode=plan.mode,
                    weights=plan.weights,
                    indices=plan.indices,
                    kept_mask=plan.kept_mask,
                    combine_weights=plan.combine_weights,
                    token_ids=token_ids,
                    packed_weights=packed_weights,
                    packed_mask=packed_mask,
                    group_sizes=group_sizes,
                    expert_capacity=plan.expert_capacity,
                    dropped_assignments=plan.dropped_assignments,
                ),
                gate,
                up,
                down,
                executor=executor,
                megablox_config=megablox_config,
                manual_mode=False,
            ),
            out_sharding=out_sharding,
        ),
    )
    return wrapped(
        x,
        plan.token_ids,
        plan.packed_weights,
        plan.packed_mask,
        plan.group_sizes,
        gate_kernels,
        up_kernels,
        down_kernels,
    )


def _runtime_config(
    runtime_config: MoERuntimeConfig | None,
    *,
    executor: MoEExecutor | None,
    capacity_factor: float | None = None,
    megablox_config: MegabloxConfig | None = None,
) -> MoERuntimeConfig:
    if runtime_config is None:
        if executor is None:
            raise ValueError("MoE runtime_config or executor must be provided.")
        return MoERuntimeConfig(
            executor=executor,
            capacity_factor=-1.0 if capacity_factor is None else capacity_factor,
            megablox=megablox_config or MegabloxConfig(),
        )

    if executor is not None and executor != runtime_config.executor:
        raise ValueError(f"Conflicting MoE executors: {executor!r} and {runtime_config.executor!r}.")
    if capacity_factor is not None and capacity_factor != runtime_config.capacity_factor:
        raise ValueError(
            "Conflicting MoE capacity factors: "
            f"{capacity_factor!r} and {runtime_config.capacity_factor!r}."
        )
    if megablox_config is not None and megablox_config != runtime_config.megablox:
        raise ValueError("Conflicting MoE MegaBlox configs.")
    return runtime_config


def execute_moe(
    x: jax.Array,
    plan: _DispatchPlan,
    gate_kernels: jax.Array,
    up_kernels: jax.Array,
    down_kernels: jax.Array,
    *,
    runtime_config: MoERuntimeConfig | None = None,
    executor: MoEExecutor | None = None,
    hidden_sharding=None,
    outputs_sharding=None,
    out_sharding=None,
    megablox_config: MegabloxConfig | None = None,
) -> jax.Array:
    runtime_config = _runtime_config(
        runtime_config,
        executor=executor,
        megablox_config=megablox_config,
    )

    match runtime_config.executor:
        case "dense":
            return _dense_moe(
                x,
                plan,
                gate_kernels,
                up_kernels,
                down_kernels,
                hidden_sharding=hidden_sharding,
                outputs_sharding=outputs_sharding,
                out_sharding=out_sharding,
            )
        case "ragged_dot" | "megablox":
            return _sparse_moe(
                x,
                plan,
                gate_kernels,
                up_kernels,
                down_kernels,
                executor=cast(SparseMoEKernel, runtime_config.executor),
                out_sharding=out_sharding,
                megablox_config=runtime_config.megablox,
            )
        case _:
            raise ValueError(f"Unknown MoE executor: {runtime_config.executor}")


def execute_routed_moe(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    gate_kernels: jax.Array,
    up_kernels: jax.Array,
    down_kernels: jax.Array,
    *,
    runtime_config: MoERuntimeConfig | None = None,
    executor: MoEExecutor | None = None,
    num_experts: int,
    capacity_factor: float | None = None,
    combine_sharding=None,
    hidden_sharding=None,
    outputs_sharding=None,
    out_sharding=None,
    megablox_config: MegabloxConfig | None = None,
) -> jax.Array:
    runtime_config = _runtime_config(
        runtime_config,
        executor=executor,
        capacity_factor=capacity_factor,
        megablox_config=megablox_config,
    )
    plan = make_dispatch_plan(
        weights,
        indices,
        executor=runtime_config.executor,
        num_experts=num_experts,
        capacity_factor=runtime_config.capacity_factor,
        combine_sharding=combine_sharding,
    )
    return execute_moe(
        x,
        plan,
        gate_kernels,
        up_kernels,
        down_kernels,
        runtime_config=runtime_config,
        hidden_sharding=hidden_sharding,
        outputs_sharding=outputs_sharding,
        out_sharding=out_sharding,
    )
