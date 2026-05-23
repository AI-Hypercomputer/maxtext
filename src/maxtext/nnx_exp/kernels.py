"""Optional dense-kernel adapters for nnx_exp."""

from collections.abc import Callable
import inspect
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp


def shard_map_options() -> dict[str, Any]:
    params = inspect.signature(jax.shard_map).parameters
    if "check_vma" in params:
        return {"check_vma": False}
    if "check_rep" in params:
        return {"check_rep": False}
    return {}



KernelBackend: TypeAlias = Literal["legacy", "auto", "tokamax", "splash"]
KernelImplementation: TypeAlias = str | None


def _sharding_of(x: Any) -> Any | None:
    if hasattr(x, "sharding"):
        return x.sharding
    try:
        return jax.typeof(x).sharding
    except Exception:
        return None


def _named_sharding_of(x: jax.Array) -> jax.sharding.NamedSharding | None:
    sharding = _sharding_of(x)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _normalized_spec_for_shape(
    spec: jax.sharding.PartitionSpec,
    shape: tuple[int, ...],
) -> jax.sharding.PartitionSpec:
    axes = tuple(spec) + (None,) * (len(shape) - len(spec))
    return jax.P(*(None if dim == 1 else axis for axis, dim in zip(axes, shape)))


def _normalize_broadcast_sharding(x: jax.Array | None) -> jax.Array | None:
    if x is None:
        return None

    named_sharding = _named_sharding_of(x)
    if named_sharding is None:
        return x

    normalized_spec = _normalized_spec_for_shape(named_sharding.spec, x.shape)
    if normalized_spec == named_sharding.spec:
        return x
    return jnp.reshape(x, x.shape, out_sharding=normalized_spec)


def _normalize_implementation(implementation: Any) -> str | None:
    match implementation:
        case None | "auto":
            return None
        case str() as impl:
            return impl
        case _:
            raise ValueError("Kernel implementation must be a single string, `auto`, or `None`.")


def _legacy_attention_implementation(implementation: KernelImplementation) -> Any:
    normalized = _normalize_implementation(implementation)
    if normalized is None and jax.default_backend() == "gpu":
        return "cudnn"
    return normalized


def _check_legacy_glu_implementation(implementation: str | None) -> None:
    match implementation:
        case None | "xla":
            return
        case _:
            raise ValueError("Legacy GLU is the inlined JAX/XLA path; only `auto`, `xla`, or `None` are valid.")


def _import_tokamax(required: bool = False) -> Any | None:
    try:
        import tokamax
    except ImportError:
        if required:
            raise ImportError(
                "Tokamax kernel backend requested but `tokamax` is not importable in this environment. "
                "Install `tokamax` and its Python dependencies, or switch the backend to `legacy` or `auto`."
            ) from None
        return None
    return tokamax


def _tokamax_available() -> bool:
    return _import_tokamax() is not None


def _use_tokamax(backend: KernelBackend) -> bool:
    match backend:
        case "legacy" | "splash":
            return False
        case "auto":
            return jax.default_backend() != "cpu" and _tokamax_available()
        case "tokamax":
            _import_tokamax(required=True)
            return True
        case _:
            raise ValueError(f"Unknown kernel backend: {backend}")


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    bias: jax.Array | None = None,
    mask: jax.Array | None = None,
    is_causal: bool = False,
    backend: KernelBackend = "legacy",
    implementation: KernelImplementation = None,
) -> jax.Array:
    # Tokamax's shard_map path drops sharding on broadcast dimensions of size 1.
    # Normalize our explicit-sharded inputs the same way so inference-time
    # batch=1 generation matches the manual in_specs Tokamax derives internally.
    query = cast(jax.Array, _normalize_broadcast_sharding(query))
    key = cast(jax.Array, _normalize_broadcast_sharding(key))
    value = cast(jax.Array, _normalize_broadcast_sharding(value))
    bias = cast(jax.Array | None, _normalize_broadcast_sharding(bias))
    mask = cast(jax.Array | None, _normalize_broadcast_sharding(mask))

    if _use_tokamax(backend):
        tokamax = _import_tokamax(required=True)
        return cast(
            jax.Array,
            tokamax.dot_product_attention(
                query,
                key,
                value,
                bias=bias,
                mask=mask,
                is_causal=is_causal,
                implementation=_normalize_implementation(implementation),
                q_sharding=_named_sharding_of(query),
            ),
        )

    if backend == "splash" or implementation == "splash":
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

        def _splash_impl(q, k, v):
            q_t = jnp.transpose(q, (0, 2, 1, 3))
            k_t = jnp.transpose(k, (0, 2, 1, 3))
            v_t = jnp.transpose(v, (0, 2, 1, 3))

            mask_shape = (q.shape[1], k.shape[1])
            if is_causal:
                base_mask = splash_attention_mask.CausalMask(shape=mask_shape)
            else:
                base_mask = splash_attention_mask.FullMask(shape=mask_shape)

            num_heads = q_t.shape[1]
            multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(base_mask,) * num_heads)

            splash_kernel = splash_attention_kernel.make_splash_mha(
                mask=multi_head_mask,
                head_shards=1,
                q_seq_shards=1,
            )

            out = jax.vmap(splash_kernel)(q_t, k_t, v_t)
            return jnp.transpose(out, (0, 2, 1, 3))

        sharding_q = _named_sharding_of(query)
        sharding_k = _named_sharding_of(key)
        sharding_v = _named_sharding_of(value)
        mesh = getattr(sharding_q, "mesh", None)
        if mesh is None:
            try:
                mesh = jax.sharding.get_abstract_mesh()
            except Exception:
                mesh = None

        if mesh is not None and sharding_q is not None and sharding_k is not None and sharding_v is not None:
            wrapped = jax.shard_map(
                _splash_impl,
                mesh=mesh,
                in_specs=(sharding_q.spec, sharding_k.spec, sharding_v.spec),
                out_specs=sharding_q.spec,
                **shard_map_options(),
            )
            return cast(jax.Array, wrapped(query, key, value))

        wrapped = jax.sharding.auto_axes(
            _splash_impl,
            out_sharding=_sharding_of(query) or jax.P(None),
        )
        return cast(jax.Array, wrapped(query, key, value))

    return cast(
        jax.Array,
        jax.nn.dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            is_causal=is_causal,
            implementation=_legacy_attention_implementation(implementation),
        ),
    )


def gated_linear_unit(
    x: jax.Array,
    weights: jax.Array | tuple[jax.Array, jax.Array],
    *,
    activation: Callable[[jax.Array], jax.Array],
    backend: KernelBackend = "legacy",
    implementation: KernelImplementation = None,
    precision: jax.lax.PrecisionLike = None,
    out_sharding: jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None = None,
) -> jax.Array:
    impl = _normalize_implementation(implementation)
    if _use_tokamax(backend):
        tokamax = _import_tokamax(required=True)
        out = cast(
            jax.Array,
            tokamax.gated_linear_unit(
                x,
                weights,
                activation=activation,
                implementation=impl,
                precision=precision,
            ),
        )
        if out_sharding is None:
            return out
        return jnp.reshape(out, out.shape, out_sharding=out_sharding)

    _check_legacy_glu_implementation(impl)
    if isinstance(weights, tuple):
        gate_kernel, up_kernel = weights
    else:
        gate_kernel, up_kernel = weights[:, 0, :], weights[:, 1, :]

    gate = jnp.tensordot(x, gate_kernel, axes=((-1,), (0,)), out_sharding=out_sharding)
    up = jnp.tensordot(x, up_kernel, axes=((-1,), (0,)), out_sharding=out_sharding)
    return activation(gate) * up
