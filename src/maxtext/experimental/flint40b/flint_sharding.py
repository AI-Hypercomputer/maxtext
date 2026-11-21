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

"""3D ('fsdp', 'cp', 'ep') mesh, PartitionSpecs, and sharded parameter init."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import jax
from jax.experimental import topologies
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_types

ALL_SHARD_AXES: tuple[str, str, str] = ("fsdp", "cp", "ep")
_TRUNCATED_NORMAL_STDDEV: float = 0.87962566103423978
P = PartitionSpec


def batch_pspec() -> PartitionSpec:
  """Returns the 2D batch `PartitionSpec(('fsdp', 'ep'), 'cp')`."""
  return P(("fsdp", "ep"), "cp")


def activation_pspec() -> PartitionSpec:
  """Returns the 3D residual activation `PartitionSpec`."""
  return P(("fsdp", "ep"), "cp", None)


def _normalize_aot_topology(aot_topology: str) -> str:
  if any(sep in aot_topology for sep in (":", "=", "_")):
    return aot_topology
  return f"tpu7x:{aot_topology}"


def create_mesh(
    sharding_cfg: flint_config.FlintShardingConfig,
    *,
    devices: Sequence[Any] | None = None,
    aot_topology: str | None = None,
) -> Mesh:
  """Constructs a torus-aligned 3D `Mesh` over `('fsdp', 'cp', 'ep')`.

  Args:
    sharding_cfg: 3D mesh axis sizes.
    devices: Explicit sequence of JAX devices, or `None` to query live/AOT.
    aot_topology: Offline TPU topology string (e.g. `'tpu7x:4x4x4'` or
      `'4x4x4'`), or `None` to use `jax.devices()`.

  Returns:
    A `jax.sharding.Mesh` with axis names `('fsdp', 'cp', 'ep')`.

  Raises:
    ValueError: If device count or torus coordinates are invalid.
  """
  if devices is None:
    if aot_topology is not None:
      topo_name = _normalize_aot_topology(aot_topology)
      topo = topologies.get_topology_desc(
          topology_name=topo_name,
          platform="tpu",
      )
      device_list = list(topo.devices)
    else:
      device_list = list(jax.devices())
  else:
    device_list = list(devices)

  if len(device_list) != sharding_cfg.num_devices:
    raise ValueError(
        f"Expected {sharding_cfg.num_devices} devices for mesh"
        f" ({sharding_cfg.fsdp_size}, {sharding_cfg.cp_size},"
        f" {sharding_cfg.ep_size}), got {len(device_list)}."
    )

  has_coords = hasattr(device_list[0], "coords") and hasattr(device_list[0], "core_on_chip")
  if has_coords:
    xs = sorted({d.coords[0] for d in device_list})
    ys = sorted({d.coords[1] for d in device_list})
    zs = sorted({d.coords[2] for d in device_list})
    cs = sorted({d.core_on_chip for d in device_list})
    x_map = {v: i for i, v in enumerate(xs)}
    y_map = {v: i for i, v in enumerate(ys)}
    z_map = {v: i for i, v in enumerate(zs)}
    c_map = {v: i for i, v in enumerate(cs)}
    grid = np.full((len(xs), len(ys), len(zs), len(cs)), None, dtype=object)
    for dev in device_list:
      x, y, z = dev.coords
      grid[x_map[x], y_map[y], z_map[z], c_map[dev.core_on_chip]] = dev
    if any(entry is None for entry in grid.flat):
      raise ValueError("Devices do not fill a dense" f" ({len(xs)}, {len(ys)}, {len(zs)}, {len(cs)}) torus grid.")
    arr = grid.reshape(
        sharding_cfg.fsdp_size,
        sharding_cfg.cp_size,
        sharding_cfg.ep_size,
    )
  else:
    arr = np.asarray(device_list, dtype=object).reshape(
        sharding_cfg.fsdp_size,
        sharding_cfg.cp_size,
        sharding_cfg.ep_size,
    )
  return Mesh(arr, ALL_SHARD_AXES)


def _prepend_scan_spec(spec: PartitionSpec, scanned: bool) -> PartitionSpec:
  if not scanned or spec == P():
    return spec
  return P(None, *spec)


def _attn_pspecs(scanned: bool) -> flint_types.FlintAttnWeights[PartitionSpec]:
  in_sharded = _prepend_scan_spec(P(ALL_SHARD_AXES, None), scanned)
  out_sharded = _prepend_scan_spec(P(None, ALL_SHARD_AXES), scanned)
  return flint_types.FlintAttnWeights(
      q_proj=in_sharded,
      k_proj=in_sharded,
      v_proj=in_sharded,
      o_proj=out_sharded,
      q_norm=P(),
      k_norm=P(),
      sinks=P(),
  )


def _mlp_pspecs(scanned: bool) -> flint_types.FlintMLPWeights[PartitionSpec]:
  in_sharded = _prepend_scan_spec(P(ALL_SHARD_AXES, None), scanned)
  out_sharded = _prepend_scan_spec(P(None, ALL_SHARD_AXES), scanned)
  return flint_types.FlintMLPWeights(
      gate_proj=in_sharded,
      up_proj=in_sharded,
      down_proj=out_sharded,
  )


def _sparse_layer_pspecs(
    scanned: bool,
    *,
    shard_routed_on_expert_axis: bool = False,
    replicate_router_gate: bool = False,
) -> flint_types.FlintSparseLayerWeights[PartitionSpec]:
  router_spec = P() if replicate_router_gate else _prepend_scan_spec(P(ALL_SHARD_AXES, None), scanned)
  if shard_routed_on_expert_axis:
    gu_spec = _prepend_scan_spec(P(("fsdp", "cp"), None, None), scanned)
    down_spec = _prepend_scan_spec(P(("fsdp", "cp"), None, None), scanned)
  else:
    gu_spec = _prepend_scan_spec(P("ep", ("fsdp", "cp"), None), scanned)
    down_spec = _prepend_scan_spec(P("ep", None, ("fsdp", "cp")), scanned)
  return flint_types.FlintSparseLayerWeights(
      input_layernorm=P(),
      self_attn=_attn_pspecs(scanned),
      post_attention_layernorm=P(),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(gate=router_spec),
          shared_expert=_mlp_pspecs(scanned),
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=gu_spec,
              down_proj=down_spec,
          ),
      ),
  )


def _dense_layer_pspecs() -> flint_types.FlintDenseLayerWeights[PartitionSpec]:
  return flint_types.FlintDenseLayerWeights(
      input_layernorm=P(),
      self_attn=_attn_pspecs(scanned=False),
      post_attention_layernorm=P(),
      mlp=_mlp_pspecs(scanned=False),
  )


def param_pspecs(
    cfg: flint_config.FlintConfig,
) -> flint_types.FlintParams[PartitionSpec]:
  """Returns the 3D `PartitionSpec` PyTree for all `FlintParams` leaves."""
  m = cfg.model
  k = cfg.kernels
  sparse_kw = dict(
      shard_routed_on_expert_axis=k.shard_routed_on_expert_axis,
      replicate_router_gate=k.replicate_router_gate,
  )
  prefix_specs = tuple(_dense_layer_pspecs() for _ in range(m.num_dense_prefix_layers))
  if m.scan_layers:
    _, num_sb, _ = flint_config.flint_scan_layout(m)
    if num_sb > 0:
      sb_layer = _sparse_layer_pspecs(scanned=True, **sparse_kw)
      scanned_blocks = flint_types.FlintSuperblockWeights(
          sliding_0=sb_layer,
          sliding_1=sb_layer,
          sliding_2=sb_layer,
          global_3=sb_layer,
      )
    else:
      scanned_blocks = None
    unscanned_moe = ()
    tail_moe = tuple(_sparse_layer_pspecs(scanned=False, **sparse_kw) for _ in range(3))
  else:
    scanned_blocks = None
    num_moe = m.num_hidden_layers - m.num_dense_prefix_layers
    unscanned_moe = tuple(_sparse_layer_pspecs(scanned=False, **sparse_kw) for _ in range(num_moe))
    tail_moe = ()

  return flint_types.FlintParams(
      embed=P(None, ALL_SHARD_AXES),
      decoder=flint_types.FlintDecoderWeights(
          prefix_dense=prefix_specs,
          scanned_blocks=scanned_blocks,
          unscanned_moe=unscanned_moe,
          tail_moe=tail_moe,
      ),
      final_norm=P(),
      lm_head=P(ALL_SHARD_AXES, None),
  )


def named_param_shardings(
    mesh: Mesh,
    cfg: flint_config.FlintConfig,
) -> flint_types.FlintParams[NamedSharding]:
  """Maps `param_pspecs(cfg)` onto `NamedSharding(mesh, spec)`."""
  return jax.tree.map(lambda s: NamedSharding(mesh, s), param_pspecs(cfg))


def _with_opt_lead(lead: int | None, shape: tuple[int, ...]) -> tuple[int, ...]:
  return shape if lead is None else (lead, *shape)


def _attn_shapes(
    m: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    lead: int | None,
) -> flint_types.FlintAttnWeights[jax.ShapeDtypeStruct]:
  d = m.hidden_size
  q_dim = m.num_attention_heads * m.head_dim
  kv_dim = m.num_key_value_heads * m.head_dim
  sds = jax.ShapeDtypeStruct
  return flint_types.FlintAttnWeights(
      q_proj=sds(_with_opt_lead(lead, (d, q_dim)), param_dt),
      k_proj=sds(_with_opt_lead(lead, (d, kv_dim)), param_dt),
      v_proj=sds(_with_opt_lead(lead, (d, kv_dim)), param_dt),
      o_proj=sds(_with_opt_lead(lead, (q_dim, d)), param_dt),
      q_norm=sds(_with_opt_lead(lead, (m.head_dim,)), param_dt),
      k_norm=sds(_with_opt_lead(lead, (m.head_dim,)), param_dt),
      sinks=sds(_with_opt_lead(lead, (m.num_attention_heads,)), jnp.float32),
  )


def _mlp_shapes(
    hidden_size: int,
    inter_size: int,
    param_dt: jnp.dtype,
    lead: int | None,
) -> flint_types.FlintMLPWeights[jax.ShapeDtypeStruct]:
  sds = jax.ShapeDtypeStruct
  return flint_types.FlintMLPWeights(
      gate_proj=sds(_with_opt_lead(lead, (hidden_size, inter_size)), param_dt),
      up_proj=sds(_with_opt_lead(lead, (hidden_size, inter_size)), param_dt),
      down_proj=sds(_with_opt_lead(lead, (inter_size, hidden_size)), param_dt),
  )


def _sparse_layer_shapes(
    m: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    lead: int | None,
) -> flint_types.FlintSparseLayerWeights[jax.ShapeDtypeStruct]:
  d = m.hidden_size
  e = m.num_experts
  moe_inter = m.moe_intermediate_size
  sds = jax.ShapeDtypeStruct
  return flint_types.FlintSparseLayerWeights(
      input_layernorm=sds(_with_opt_lead(lead, (d,)), param_dt),
      self_attn=_attn_shapes(m, param_dt, lead),
      post_attention_layernorm=sds(_with_opt_lead(lead, (d,)), param_dt),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(
              gate=sds(_with_opt_lead(lead, (d, e)), param_dt),
          ),
          shared_expert=_mlp_shapes(d, m.shared_expert_intermediate_size, param_dt, lead),
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=sds(_with_opt_lead(lead, (e, d, 2 * moe_inter)), param_dt),
              down_proj=sds(_with_opt_lead(lead, (e, moe_inter, d)), param_dt),
          ),
      ),
  )


def abstract_param_shapes(
    cfg: flint_config.FlintConfig,
) -> flint_types.FlintParams[jax.ShapeDtypeStruct]:
  """Returns abstract `ShapeDtypeStruct` leaves (`sinks` always `float32`)."""
  m = cfg.model
  param_dt = cfg.train.param_jnp_dtype
  d = m.hidden_size
  sds = jax.ShapeDtypeStruct

  dense_shape = flint_types.FlintDenseLayerWeights(
      input_layernorm=sds((d,), param_dt),
      self_attn=_attn_shapes(m, param_dt, lead=None),
      post_attention_layernorm=sds((d,), param_dt),
      mlp=_mlp_shapes(d, m.intermediate_size, param_dt, lead=None),
  )
  prefix_dense = tuple(dense_shape for _ in range(m.num_dense_prefix_layers))

  if m.scan_layers:
    _, num_sb, _ = flint_config.flint_scan_layout(m)
    if num_sb > 0:
      sb_shape = _sparse_layer_shapes(m, param_dt, lead=num_sb)
      scanned_blocks = flint_types.FlintSuperblockWeights(
          sliding_0=sb_shape,
          sliding_1=sb_shape,
          sliding_2=sb_shape,
          global_3=sb_shape,
      )
    else:
      scanned_blocks = None
    unscanned_moe = ()
    tail_moe = tuple(_sparse_layer_shapes(m, param_dt, lead=None) for _ in range(3))
  else:
    scanned_blocks = None
    num_moe = m.num_hidden_layers - m.num_dense_prefix_layers
    unscanned_moe = tuple(_sparse_layer_shapes(m, param_dt, lead=None) for _ in range(num_moe))
    tail_moe = ()

  return flint_types.FlintParams(
      embed=sds((m.vocab_size, d), param_dt),
      decoder=flint_types.FlintDecoderWeights(
          prefix_dense=prefix_dense,
          scanned_blocks=scanned_blocks,
          unscanned_moe=unscanned_moe,
          tail_moe=tail_moe,
      ),
      final_norm=sds((d,), param_dt),
      lm_head=sds((d, m.vocab_size), param_dt),
  )


def _pspec_divisor(
    shape: tuple[int, ...],
    spec: PartitionSpec,
    axis_sizes: dict[str, int],
) -> int:
  """Returns the product of sharded mesh-axis sizes across `shape`."""
  divisor = 1
  for dim_idx, entry in enumerate(spec):
    if entry is None:
      continue
    axes = (entry,) if isinstance(entry, str) else entry
    dim_div = 1
    for ax in axes:
      dim_div *= axis_sizes[ax]
    if shape[dim_idx] % dim_div != 0:
      raise ValueError(f"Shape {shape} dim {dim_idx} not divisible by spec {spec} ({dim_div}" " ways).")
    divisor *= dim_div
  return divisor


def estimate_persistent_state_bytes_per_device(
    cfg: flint_config.FlintConfig,
) -> tuple[int, int]:
  """Returns `(param_bytes_per_dev, adam_total_bytes_per_dev)`.

  `adam_total_bytes_per_dev` sums primary parameter bytes plus FP32 AdamW `mu`
  and `nu` moments (`8` bytes per local parameter element).

  Args:
    cfg: Validated `FlintConfig`.

  Returns:
    Tuple `(param_bytes_per_dev, adam_total_bytes_per_dev)` in bytes per device.
  """
  s = cfg.sharding
  axis_sizes = {
      "fsdp": s.fsdp_size,
      "cp": s.cp_size,
      "ep": s.ep_size,
  }
  shapes = abstract_param_shapes(cfg)
  specs = param_pspecs(cfg)
  param_bytes = 0
  moment_bytes = 0
  for sds, spec in zip(jax.tree.leaves(shapes), jax.tree.leaves(specs), strict=True):
    div = _pspec_divisor(sds.shape, spec, axis_sizes)
    local_elems = math.prod(sds.shape) // div
    param_bytes += local_elems * jnp.dtype(sds.dtype).itemsize
    moment_bytes += local_elems * 8
  return param_bytes, param_bytes + moment_bytes


def _sample_matrix(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    fan_in: int,
    model_cfg: flint_config.FlintModelConfig,
    out_sharding: NamedSharding,
) -> jax.Array:
  """Samples a weight tensor with exact 2D `fan_in` scale and `out_sharding`."""
  if model_cfg.init_mode == "fan_in_trunc_normal":
    stddev = (1.0 / math.sqrt(float(fan_in))) / _TRUNCATED_NORMAL_STDDEV
    raw = jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype)
    out = raw * jnp.asarray(stddev, dtype=dtype)
    return jax.lax.with_sharding_constraint(out, out_sharding)
  if model_cfg.init_mode == "normal":
    raw = jax.random.normal(key, shape, dtype)
    out = raw * jnp.asarray(model_cfg.initializer_range, dtype=dtype)
    return jax.lax.with_sharding_constraint(out, out_sharding)
  raise ValueError(f"Unsupported init_mode: {model_cfg.init_mode!r}.")


def _const_vec(
    shape: tuple[int, ...],
    val: float,
    dtype: jnp.dtype,
    out_sharding: NamedSharding,
) -> jax.Array:
  arr = jnp.full(shape, val, dtype=dtype)
  return jax.lax.with_sharding_constraint(arr, out_sharding)


def _init_attn(
    key: jax.Array,
    m: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    lead: int | None,
    shardings: flint_types.FlintAttnWeights[NamedSharding],
) -> flint_types.FlintAttnWeights[jax.Array]:
  d = m.hidden_size
  q_dim = m.num_attention_heads * m.head_dim
  kv_dim = m.num_key_value_heads * m.head_dim
  kq, kk, kv, ko = jax.random.split(key, 4)
  return flint_types.FlintAttnWeights(
      q_proj=_sample_matrix(kq, _with_opt_lead(lead, (d, q_dim)), param_dt, d, m, shardings.q_proj),
      k_proj=_sample_matrix(
          kk,
          _with_opt_lead(lead, (d, kv_dim)),
          param_dt,
          d,
          m,
          shardings.k_proj,
      ),
      v_proj=_sample_matrix(
          kv,
          _with_opt_lead(lead, (d, kv_dim)),
          param_dt,
          d,
          m,
          shardings.v_proj,
      ),
      o_proj=_sample_matrix(
          ko,
          _with_opt_lead(lead, (q_dim, d)),
          param_dt,
          q_dim,
          m,
          shardings.o_proj,
      ),
      q_norm=_const_vec(
          _with_opt_lead(lead, (m.head_dim,)),
          1.0,
          param_dt,
          shardings.q_norm,
      ),
      k_norm=_const_vec(
          _with_opt_lead(lead, (m.head_dim,)),
          1.0,
          param_dt,
          shardings.k_norm,
      ),
      sinks=_const_vec(
          _with_opt_lead(lead, (m.num_attention_heads,)),
          0.0,
          jnp.dtype(jnp.float32),
          shardings.sinks,
      ),
  )


def _init_mlp(
    key: jax.Array,
    hidden_size: int,
    inter_size: int,
    m: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    lead: int | None,
    shardings: flint_types.FlintMLPWeights[NamedSharding],
) -> flint_types.FlintMLPWeights[jax.Array]:
  kg, ku, kd = jax.random.split(key, 3)
  return flint_types.FlintMLPWeights(
      gate_proj=_sample_matrix(
          kg,
          _with_opt_lead(lead, (hidden_size, inter_size)),
          param_dt,
          hidden_size,
          m,
          shardings.gate_proj,
      ),
      up_proj=_sample_matrix(
          ku,
          _with_opt_lead(lead, (hidden_size, inter_size)),
          param_dt,
          hidden_size,
          m,
          shardings.up_proj,
      ),
      down_proj=_sample_matrix(
          kd,
          _with_opt_lead(lead, (inter_size, hidden_size)),
          param_dt,
          inter_size,
          m,
          shardings.down_proj,
      ),
  )


def _init_sparse_layer(
    key: jax.Array,
    m: flint_config.FlintModelConfig,
    param_dt: jnp.dtype,
    lead: int | None,
    shardings: flint_types.FlintSparseLayerWeights[NamedSharding],
) -> flint_types.FlintSparseLayerWeights[jax.Array]:
  """Initializes sharded weights for one (or `lead`-stacked) sparse MoE layer."""
  d = m.hidden_size
  e = m.num_experts
  moe_inter = m.moe_intermediate_size
  k_attn, k_router, k_shared, k_routed = jax.random.split(key, 4)
  k_gate = jax.random.fold_in(k_routed, 0)
  k_up = jax.random.fold_in(k_routed, 1)
  k_down = jax.random.fold_in(k_routed, 2)

  routed_half_shape = _with_opt_lead(lead, (e, d, moe_inter))
  gate_half = _sample_matrix(
      k_gate,
      routed_half_shape,
      param_dt,
      d,
      m,
      shardings.moe.routed_experts.gate_up_proj,
  )
  up_half = _sample_matrix(
      k_up,
      routed_half_shape,
      param_dt,
      d,
      m,
      shardings.moe.routed_experts.gate_up_proj,
  )
  gate_up = jnp.concatenate([gate_half, up_half], axis=-1)
  down = _sample_matrix(
      k_down,
      _with_opt_lead(lead, (e, moe_inter, d)),
      param_dt,
      moe_inter,
      m,
      shardings.moe.routed_experts.down_proj,
  )

  return flint_types.FlintSparseLayerWeights(
      input_layernorm=_const_vec(
          _with_opt_lead(lead, (d,)),
          1.0,
          param_dt,
          shardings.input_layernorm,
      ),
      self_attn=_init_attn(k_attn, m, param_dt, lead, shardings.self_attn),
      post_attention_layernorm=_const_vec(
          _with_opt_lead(lead, (d,)),
          1.0,
          param_dt,
          shardings.post_attention_layernorm,
      ),
      moe=flint_types.FlintMoEWeights(
          router=flint_types.FlintRouterWeights(
              gate=_sample_matrix(
                  k_router,
                  _with_opt_lead(lead, (d, e)),
                  param_dt,
                  d,
                  m,
                  shardings.moe.router.gate,
              ),
          ),
          shared_expert=_init_mlp(
              k_shared,
              d,
              m.shared_expert_intermediate_size,
              m,
              param_dt,
              lead,
              shardings.moe.shared_expert,
          ),
          routed_experts=flint_types.FlintRoutedExpertsWeights(
              gate_up_proj=gate_up,
              down_proj=down,
          ),
      ),
  )


def init_random_params(
    key: jax.Array,
    mesh: Mesh,
    cfg: flint_config.FlintConfig,
) -> flint_types.FlintParams[jax.Array]:
  """Initializes `FlintParams` directly sharded across `mesh` (multi-host safe).

  Uses `normal(stddev=1.0)` on `embed` under `fan_in_trunc_normal` (`fan_in = 1`
  for one-hot token lookup, matching MaxText `Embed`), exact 2D contraction
  `fan_in` on all 2D/3D linear projections (ignoring stacked `[N_sb]` and `[E]`
  axes), separate folded PRNG keys for `gate_proj` and `up_proj` halves of
  `routed_experts.gate_up_proj`, `1.0` for norms, and `0.0` (`float32`) for
  attention `sinks`.

  Args:
    key: Base JAX PRNG key.
    mesh: Target 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.
    cfg: Validated `FlintConfig`.

  Returns:
    Sharded `FlintParams` on `mesh`.
  """
  out_shardings = named_param_shardings(mesh, cfg)
  m = cfg.model
  param_dt = cfg.train.param_jnp_dtype
  d = m.hidden_size

  def _build_params(rng_key: jax.Array) -> flint_types.FlintParams[jax.Array]:
    k_embed = jax.random.fold_in(rng_key, 10_000)
    k_head = jax.random.fold_in(rng_key, 10_001)
    if m.init_mode == "fan_in_trunc_normal":
      embed = jax.lax.with_sharding_constraint(
          jax.random.normal(k_embed, (m.vocab_size, d), param_dt),
          out_shardings.embed,
      )
    else:
      embed = _sample_matrix(
          k_embed,
          (m.vocab_size, d),
          param_dt,
          d,
          m,
          out_shardings.embed,
      )
    prefix_layers = []
    for idx in range(m.num_dense_prefix_layers):
      lk = jax.random.fold_in(rng_key, idx)
      k_attn, k_mlp = jax.random.split(lk, 2)
      sh_i = out_shardings.decoder.prefix_dense[idx]
      prefix_layers.append(
          flint_types.FlintDenseLayerWeights(
              input_layernorm=_const_vec((d,), 1.0, param_dt, sh_i.input_layernorm),
              self_attn=_init_attn(k_attn, m, param_dt, None, sh_i.self_attn),
              post_attention_layernorm=_const_vec((d,), 1.0, param_dt, sh_i.post_attention_layernorm),
              mlp=_init_mlp(k_mlp, d, m.intermediate_size, m, param_dt, None, sh_i.mlp),
          )
      )

    scanned_blocks: flint_types.FlintSuperblockWeights[jax.Array] | None
    unscanned_moe: tuple[flint_types.FlintSparseLayerWeights[jax.Array], ...]
    tail_moe: tuple[flint_types.FlintSparseLayerWeights[jax.Array], ...]
    if m.scan_layers:
      _, num_sb, tail_start = flint_config.flint_scan_layout(m)
      sb_sh = out_shardings.decoder.scanned_blocks
      if num_sb > 0 and sb_sh is not None:
        scanned_blocks = flint_types.FlintSuperblockWeights(
            sliding_0=_init_sparse_layer(
                jax.random.fold_in(rng_key, 20_000),
                m,
                param_dt,
                num_sb,
                sb_sh.sliding_0,
            ),
            sliding_1=_init_sparse_layer(
                jax.random.fold_in(rng_key, 20_001),
                m,
                param_dt,
                num_sb,
                sb_sh.sliding_1,
            ),
            sliding_2=_init_sparse_layer(
                jax.random.fold_in(rng_key, 20_002),
                m,
                param_dt,
                num_sb,
                sb_sh.sliding_2,
            ),
            global_3=_init_sparse_layer(
                jax.random.fold_in(rng_key, 20_003),
                m,
                param_dt,
                num_sb,
                sb_sh.global_3,
            ),
        )
      else:
        scanned_blocks = None
      unscanned_moe = ()
      tail_moe = tuple(
          _init_sparse_layer(
              jax.random.fold_in(rng_key, tail_start + j),
              m,
              param_dt,
              None,
              out_shardings.decoder.tail_moe[j],
          )
          for j in range(3)
      )
    else:
      scanned_blocks = None
      num_moe = m.num_hidden_layers - m.num_dense_prefix_layers
      unscanned_moe = tuple(
          _init_sparse_layer(
              jax.random.fold_in(rng_key, m.num_dense_prefix_layers + j),
              m,
              param_dt,
              None,
              out_shardings.decoder.unscanned_moe[j],
          )
          for j in range(num_moe)
      )
      tail_moe = ()

    return flint_types.FlintParams(
        embed=embed,
        decoder=flint_types.FlintDecoderWeights(
            prefix_dense=tuple(prefix_layers),
            scanned_blocks=scanned_blocks,
            unscanned_moe=unscanned_moe,
            tail_moe=tail_moe,
        ),
        final_norm=_const_vec((d,), 1.0, param_dt, out_shardings.final_norm),
        lm_head=_sample_matrix(
            k_head,
            (d, m.vocab_size),
            param_dt,
            d,
            m,
            out_shardings.lm_head,
        ),
    )

  if isinstance(key, jax.core.Tracer):
    return jax.jit(_build_params, out_shardings=out_shardings)(key)

  is_typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
  raw_key = jax.random.key_data(key) if is_typed else key
  key_bits = tuple(int(x) for x in np.asarray(raw_key).reshape(-1))
  raw_shape = raw_key.shape

  @jax.jit(out_shardings=out_shardings)
  def _init_from_bits() -> flint_types.FlintParams[jax.Array]:
    arr = jnp.asarray(key_bits, dtype=jnp.uint32).reshape(raw_shape)
    rng_key = jax.random.wrap_key_data(arr) if is_typed else arr
    return _build_params(rng_key)

  return _init_from_bits()
