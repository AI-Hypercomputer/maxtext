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

"""Adapter bridging MaxText models/configs to Lineage DeepSeek-V3 (dsv3.py).

This module provides the integration layer between MaxText's model
representations and Lineage's DeepSeek-V3 model (dsv3.py).
Specifically, it handles:

1. Weight Translation and Layout Alignment:
   - `fetch_lineage_dense_weights`: Extracts and translates weights from MaxText
     dense layer parameter trees into Lineage's typed
     `DSv3DenseLayerWeightsPytree`.
   - `fetch_lineage_sparse_weights`: Extracts and translates weights from
     MaxText MoE layer parameter trees into Lineage's typed
     `DSv3SparseLayerWeightsPytree`.
   - `fetch_lineage_weights`: Packages dense and sparse weights into
     `DSv3WeightsPytree`.

2. Physical Mesh and Axis Mapping (`build_axis_mapping`):
   - Maps Lineage logical sharding axes (`attention`, `fsdp_attention`,
     `expert`, `fsdp_moe`) to the physical cluster mesh.

3. Kernel Initialization and Model Execution (`run_lineage_dsv3`):
   - Prepares input activation shardings and parameter distributions.
   - Derives YaRN rotary embedding frequencies (`dsv3_mla.get_yarn_freqs`).
   - Configures optimized Splash attention kernels
     (`dsv3_mla.init_splash_kernel`).
   - Dispatches execution directly through Lineage's top-level integration point
     `learning/performance/lineage/dsv3.py:dsv3`.
"""

from collections.abc import Mapping
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from maxtext.models.deepseek_lineage import dsv3
from maxtext.models.deepseek_lineage import dsv3_mla
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops


def validate_lineage_config(cfg: Any) -> None:
  """Validates that MaxText configuration matches the desired state for Lineage."""
  if cfg is None:
    raise ValueError("MaxText configuration (cfg) cannot be None for Lineage.")
  if hasattr(cfg, "use_lineage") and not cfg.use_lineage:
    raise ValueError("Lineage adapter requires use_lineage=True.")
  decoder_block = getattr(cfg, "decoder_block", "deepseek")
  if getattr(decoder_block, "value", decoder_block) not in ("deepseek", None):
    raise ValueError(f"Lineage requires decoder_block='deepseek', got {decoder_block!r}.")
  if hasattr(cfg, "scan_layers") and not cfg.scan_layers:
    raise ValueError("Lineage requires scan_layers=True.")
  attention_type = getattr(cfg, "attention_type", "mla")
  if getattr(attention_type, "value", attention_type) not in ("mla", None):
    raise ValueError(f"Lineage requires attention_type='mla', got {attention_type!r}.")
  rope_type = getattr(cfg, "rope_type", "yarn")
  if getattr(rope_type, "value", rope_type) not in ("yarn", None):
    raise ValueError(f"Lineage requires rope_type='yarn', got {rope_type!r}.")
  if getattr(cfg, "param_scan_axis", 0) not in (0, 1):
    raise ValueError(f"Unsupported param_scan_axis: {cfg.param_scan_axis}. Expected 0 or 1.")
  get_capacity_factor(cfg)


def _transform(
    val: Any,
    dtype: jax.typing.DTypeLike | None = None,
    param_scan_axis: int = 0,
) -> jax.Array | None:
  """Moves layer dimension to axis 0 for Lineage, updates sharding, and casts dtype."""
  if val is None:
    return None
  if dtype is not None:
    val = jnp.asarray(val, dtype=dtype)
  if 0 < param_scan_axis < val.ndim:
    sharding = getattr(val, "sharding", None)
    val = jnp.moveaxis(val, param_scan_axis, 0)
    mesh = getattr(sharding, "mesh", None)
    if sharding is not None and hasattr(sharding, "spec") and mesh is not None and not getattr(mesh, "empty", False):
      partitions = list(sharding.spec.partitions)
      while len(partitions) < val.ndim:
        partitions.append(None)
      partitions.insert(0, partitions.pop(param_scan_axis))
      val = jax.reshard(
          val,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*partitions)),
      )
  return val


def _unwrap(val: Any) -> Any:
  """Unwraps parameter container wrappers such as Flax/NNX Param or dict value."""
  while hasattr(val, "value") or (isinstance(val, Mapping) and "value" in val and len(val) == 1):
    val = val.value if hasattr(val, "value") else val["value"]
  return val


def _extract(
    container: Any,
    key: str,
    subkey: str,
    dtype: jax.typing.DTypeLike | None = None,
    param_scan_axis: int = 0,
) -> jax.Array | None:
  """Extracts a parameter leaf from nested dict or object containers."""
  if container is None:
    return None
  val = container.get(key) if isinstance(container, Mapping) else getattr(container, key, None)
  if val is None:
    return None
  val = _unwrap(val)
  if isinstance(val, Mapping):
    val = val.get(subkey, val if subkey in ("kernel", "scale") else None)
  elif hasattr(val, subkey):
    val = getattr(val, subkey)
  elif subkey not in ("kernel", "scale"):
    return None
  return _transform(_unwrap(val), dtype=dtype, param_scan_axis=param_scan_axis)


def _extract_mla(
    self_attn: Any,
    dtype: jax.typing.DTypeLike | None = None,
    param_scan_axis: int = 0,
    qk_head_dim: int | None = None,
) -> dsv3_types.DSv3MLAWeightsPytree:
  """Extracts MLA weights from self_attention container."""
  if self_attn is None:
    return dsv3_types.DSv3MLAWeightsPytree()
  ext = functools.partial(_extract, self_attn, dtype=dtype, param_scan_axis=param_scan_axis)
  kv_up = ext("wkv_b", "kernel")
  qk_dim = qk_head_dim or (kv_up.shape[-1] // 2 if kv_up is not None else None)
  return dsv3_types.DSv3MLAWeightsPytree(
      q_down=ext("wq_a", "kernel"),
      q_up=ext("wq_b", "kernel"),
      q_norm_scale=ext("q_norm", "scale"),
      kv_down=ext("wkv_a", "kernel"),
      k_up=kv_up[..., :qk_dim] if kv_up is not None else None,
      v_up=kv_up[..., qk_dim:] if kv_up is not None else None,
      kv_norm_scale=ext("kv_norm", "scale"),
      out=ext("out", "kernel"),
  )


def _check_pytree(params: Any, cls: Any, dtype: jax.typing.DTypeLike | None, param_scan_axis: int) -> Any:
  """Validates and optionally transforms pytree weights if already wrapped."""
  if isinstance(params, cls):
    if dtype is None and param_scan_axis == 0:
      return params
    return jax.tree.map(
        lambda x: _transform(x, dtype=dtype, param_scan_axis=param_scan_axis) if x is not None else None,
        params,
    )
  if params is None:
    raise ValueError("params cannot be None.")
  return None


def _common_weights(
    params: Any,
    ext: Any,
    get_obj: Any,
    dtype: jax.typing.DTypeLike | None,
    param_scan_axis: int,
    qk_head_dim: int | None,
) -> dict[str, Any]:
  """Extracts common pre/post attention norm scales and MLA weights."""
  return {
      "pre_attn_norm_scale": ext(params, "pre_self_attention_layer_norm", "scale"),
      "mla": _extract_mla(get_obj(params, "self_attention"), dtype, param_scan_axis, qk_head_dim),
      "post_attn_norm_scale": ext(params, "post_self_attention_layer_norm", "scale"),
  }


def fetch_lineage_dense_weights(
    params: Any,
    dtype: jax.typing.DTypeLike | None = None,
    param_scan_axis: int = 0,
    qk_head_dim: int | None = None,
) -> dsv3_types.DSv3DenseLayerWeightsPytree:
  """Fetches and translates MaxText dense layer weights into Lineage DSv3DenseLayerWeightsPytree."""
  res = _check_pytree(params, dsv3_types.DSv3DenseLayerWeightsPytree, dtype, param_scan_axis)
  if res is not None:
    return res

  ext = functools.partial(_extract, dtype=dtype, param_scan_axis=param_scan_axis)

  def get_obj(p, k):
    return p.get(k) if isinstance(p, Mapping) else getattr(p, k, None)

  mlp = get_obj(params, "mlp")
  return dsv3_types.DSv3DenseLayerWeightsPytree(
      **_common_weights(params, ext, get_obj, dtype, param_scan_axis, qk_head_dim),
      mlp=dsv3_types.DSv3MLPWeightsPytree(
          gate_0=ext(mlp, "wi_0", "kernel"),
          gate_1=ext(mlp, "wi_1", "kernel"),
          linear=ext(mlp, "wo", "kernel"),
      ),
  )


def fetch_lineage_sparse_weights(
    params: Any,
    dtype: jax.typing.DTypeLike | None = None,
    param_scan_axis: int = 0,
    qk_head_dim: int | None = None,
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  """Fetches and translates MaxText MoE layer weights into Lineage DSv3SparseLayerWeightsPytree."""
  res = _check_pytree(params, dsv3_types.DSv3SparseLayerWeightsPytree, dtype, param_scan_axis)
  if res is not None:
    return res

  ext = functools.partial(_extract, dtype=dtype, param_scan_axis=param_scan_axis)

  def get_obj(p, k):
    return p.get(k) if isinstance(p, Mapping) else getattr(p, k, None)

  ds_moe = get_obj(params, "DeepSeekMoeBlock_0")
  moe_block = get_obj(ds_moe, "MoeBlock_0")
  shared_experts = get_obj(ds_moe, "shared_experts")

  router_kernel = ext(moe_block, "gate", "kernel")
  router_bias = ext(moe_block, "gate", "bias")
  if router_bias is None and router_kernel is not None:
    bias_shape = (
        (router_kernel.shape[0], router_kernel.shape[-1]) if router_kernel.ndim == 3 else (router_kernel.shape[-1],)
    )
    router_bias = jnp.zeros(bias_shape, dtype=router_kernel.dtype)

  wi_0 = ext(moe_block, "wi_0", "kernel")
  wi_1 = ext(moe_block, "wi_1", "kernel")
  routed_gate = jnp.concatenate([wi_0, wi_1], axis=-1) if wi_0 is not None and wi_1 is not None else None

  return dsv3_types.DSv3SparseLayerWeightsPytree(
      **_common_weights(params, ext, get_obj, dtype, param_scan_axis, qk_head_dim),
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(kernel=router_kernel, bias=router_bias),
          routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(gate=routed_gate, linear=ext(moe_block, "wo", "kernel")),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(
              gate_0=ext(shared_experts, "wi_0", "kernel"),
              gate_1=ext(shared_experts, "wi_1", "kernel"),
              linear=ext(shared_experts, "wo", "kernel"),
          ),
      ),
  )


def fetch_lineage_weights(
    dense_params: Any,
    sparse_params: Any,
    dtype: jax.typing.DTypeLike | None = None,
    param_scan_axis: int = 0,
    qk_head_dim: int | None = None,
) -> dsv3_types.DSv3WeightsPytree:
  """Fetches and translates MaxText dense and MoE layer weights into Lineage DSv3WeightsPytree."""
  res = _check_pytree(dense_params, dsv3_types.DSv3WeightsPytree, dtype, param_scan_axis)
  if res is not None:
    return res
  if dense_params is not None:
    dense_weights = fetch_lineage_dense_weights(
        dense_params,
        dtype=dtype,
        param_scan_axis=param_scan_axis,
        qk_head_dim=qk_head_dim,
    )
  else:
    dense_weights = dsv3_types.DSv3DenseLayerWeightsPytree()

  if sparse_params is not None:
    sparse_weights = fetch_lineage_sparse_weights(
        sparse_params,
        dtype=dtype,
        param_scan_axis=param_scan_axis,
        qk_head_dim=qk_head_dim,
    )
  else:
    sparse_weights = dsv3_types.DSv3SparseLayerWeightsPytree()

  return dsv3_types.DSv3WeightsPytree(
      dense=dense_weights,
      sparse=sparse_weights,
  )


def _norm_axes(v: Any) -> Any:
  if isinstance(v, (list, tuple)):
    return tuple(v) if len(v) > 1 else (v[0] if len(v) == 1 else ())
  return v


def build_axis_mapping(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    cfg: Any = None,
) -> Mapping[str, str | tuple[str, ...]]:
  """Builds axis mapping from Lineage logical axes to MaxText mesh axes."""
  del mesh  # Lineage uses 4D physical coordinate mesh with logical rules.
  rules = dict(getattr(cfg, "logical_axis_rules", None) or [])
  attn = rules.get("activation_length", "core")
  fsdp_attn = rules.get("activation_batch_attn", ("x", "y", "z"))
  exp = rules.get("exp", ("x", "y", "core"))
  f_moe = rules.get("embed_moe", "z")
  norm_f_moe = _norm_axes(f_moe)
  return {
      "attention": _norm_axes(attn),
      "fsdp_attention": _norm_axes(fsdp_attn),
      "expert": _norm_axes(exp),
      "fsdp_moe": norm_f_moe,
      "z": norm_f_moe,
  }


def get_capacity_factor(cfg: Any) -> float:
  """Determines the capacity factor for Lineage layers from config."""
  if cfg is None:
    raise ValueError("cfg must be provided to get_capacity_factor.")
  for attr in ("capacity_factor", "lineage_capacity_factor"):
    val = getattr(cfg, attr, None)
    if val is not None and val > 0:
      return float(val)
  cap_val = getattr(cfg, "capacity_factor", getattr(cfg, "lineage_capacity_factor", None))
  raise ValueError(f"capacity_factor must be set and > 0, got: {cap_val}")


def _mesh_axis_size(axes: str | tuple[str, ...] | None, mesh: Any) -> int:
  if not axes:
    return 1
  shape = getattr(mesh, "shape", {})
  return math.prod(
      shape.get(a, 1) if isinstance(shape, Mapping) else getattr(shape, a, 1)
      for a in ((axes,) if isinstance(axes, str) else axes)
  )


def compute_effective_capacity_factor(
    inputs: jax.Array,
    mesh: Any,
    axis_mapping: Mapping[str, Any],
    base_capacity_factor: float,
    expert_axis_name: str = "expert",
    min_capacity: int = 1024,
) -> float:
  """Ensures max_capacity is at least min_capacity (1024) and aligned for Pallas kernels."""
  if mesh is None or getattr(mesh, "empty", False):
    return base_capacity_factor
  num_devices = _mesh_axis_size(axis_mapping.get(expert_axis_name, expert_axis_name), mesh)
  sharding = getattr(jax.typeof(inputs), "sharding", None) or getattr(inputs, "sharding", None)
  partitions = getattr(getattr(sharding, "spec", None), "partitions", ()) or ()
  local_b = (inputs.shape[0] if inputs.ndim > 0 else 1) // _mesh_axis_size(
      partitions[0] if len(partitions) > 0 else None, mesh
  )
  local_s = (inputs.shape[1] if inputs.ndim > 1 else 1) // _mesh_axis_size(
      partitions[1] if len(partitions) > 1 else None, mesh
  )
  tokens_per_expert = local_b * local_s * num_devices
  if tokens_per_expert <= 0:
    return base_capacity_factor
  raw = int(base_capacity_factor * tokens_per_expert)
  desired = ((max(min_capacity, raw) + min_capacity - 1) // min_capacity) * min_capacity
  return float((desired + 1e-6) / tokens_per_expert)


def run_lineage_dsv3(
    inputs: jax.Array,
    dense_params: Any,
    sparse_params: Any,
    decoder_positions: jax.Array,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    cfg: Any,
) -> jax.Array:
  """Executes Lineage DSv3 model (dense layers followed by sparse layers)."""
  validate_lineage_config(cfg)
  axis_mapping = build_axis_mapping(mesh, cfg)

  physical_activation_pspec = ops.physical_pspec(
      jax.sharding.PartitionSpec("fsdp_attention", "attention", None),
      axis_mapping,
  )
  inputs = jax.reshard(inputs, jax.sharding.NamedSharding(mesh, physical_activation_pspec))

  weights = fetch_lineage_weights(
      dense_params,
      sparse_params,
      dtype=cfg.dtype,
      param_scan_axis=getattr(cfg, "param_scan_axis", 0),
      qk_head_dim=getattr(cfg, "qk_nope_head_dim", 128),
  )

  p = jax.sharding.PartitionSpec

  def _reshard(arr: jax.Array, pspec: jax.sharding.PartitionSpec) -> jax.Array:
    return jax.reshard(
        arr,
        jax.sharding.NamedSharding(mesh, ops.physical_pspec(pspec, axis_mapping)),
    )

  scale_spec = p(None, None)
  for lw in (weights.dense, weights.sparse):
    lw.pre_attn_norm_scale = _reshard(lw.pre_attn_norm_scale, scale_spec)
    lw.post_attn_norm_scale = _reshard(lw.post_attn_norm_scale, scale_spec)
    lw.mla.q_down = _reshard(lw.mla.q_down, p(None, "fsdp_moe", None))
    lw.mla.q_up = _reshard(lw.mla.q_up, p(None, None, "attention", "fsdp_moe"))
    lw.mla.q_norm_scale = _reshard(lw.mla.q_norm_scale, scale_spec)
    lw.mla.kv_down = _reshard(lw.mla.kv_down, p(None, "fsdp_moe", None))
    lw.mla.k_up = _reshard(lw.mla.k_up, p(None, None, "attention", "fsdp_moe"))
    lw.mla.v_up = _reshard(lw.mla.v_up, p(None, None, "attention", "fsdp_moe"))
    lw.mla.kv_norm_scale = _reshard(lw.mla.kv_norm_scale, scale_spec)
    lw.mla.out = _reshard(lw.mla.out, p(None, "attention", None, "fsdp_moe"))

  mlp = weights.dense.mlp
  mlp.gate_0 = _reshard(mlp.gate_0, p(None, None, "fsdp_moe"))
  mlp.gate_1 = _reshard(mlp.gate_1, p(None, None, "fsdp_moe"))
  mlp.linear = _reshard(mlp.linear, p(None, "fsdp_moe", None))

  moe = weights.sparse.moe
  moe.router.kernel = _reshard(moe.router.kernel, p())
  moe.router.bias = _reshard(moe.router.bias, p())
  moe.routed.gate = _reshard(moe.routed.gate, p(None, "expert", None, "fsdp_moe"))
  moe.routed.linear = _reshard(moe.routed.linear, p(None, "expert", "fsdp_moe", None))
  moe.shared.gate_0 = _reshard(moe.shared.gate_0, p(None, None, "fsdp_moe"))
  moe.shared.gate_1 = _reshard(moe.shared.gate_1, p(None, None, "fsdp_moe"))
  moe.shared.linear = _reshard(moe.shared.linear, p(None, "fsdp_moe", None))

  kernel_out_spec = p("attention", None)

  yarn_freqs = dsv3_mla.get_yarn_freqs(
      decoder_positions,
      rope_head_dim=cfg.qk_rope_head_dim,
      rope_theta=getattr(cfg, "rope_max_timescale", 10000),
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      beta_fast=cfg.beta_fast,
      beta_slow=cfg.beta_slow,
      rope_factor=cfg.rope_factor,
      out_pspec=p("fsdp_attention", None, None),
      mesh=mesh,
      axis_mapping=axis_mapping,
      dtype=cfg.dtype,
  )

  num_query_heads = getattr(cfg, "num_query_heads", getattr(cfg, "base_num_query_heads", 128))
  sa_kwargs = {
      k: getattr(cfg, k, d)
      for k, d in (
          ("sa_block_q", 2048),
          ("sa_block_kv", 2048),
          ("sa_block_kv_compute", 1024),
          ("sa_block_q_dkv", 2048),
          ("sa_block_kv_dkv", 2048),
          ("sa_block_kv_dkv_compute", 2048),
          ("sa_q_layout", "HEAD_DIM_MINOR"),
          ("sa_k_layout", "HEAD_DIM_MINOR"),
          ("sa_v_layout", "HEAD_DIM_MINOR"),
      )
  }
  splash_kernel = dsv3_mla.init_splash_kernel(
      max_target_length=cfg.max_target_length,
      num_query_heads=num_query_heads,
      kernel_out_spec=kernel_out_spec,
      mesh=mesh,
      axis_mapping=axis_mapping,
      **sa_kwargs,
  )

  topk_routing_group = getattr(cfg, "topk_routing_group", 4)
  topk_in_group = getattr(
      cfg,
      "topk_in_group",
      cfg.num_experts_per_tok // topk_routing_group,
  )
  capacity_factor = compute_effective_capacity_factor(inputs, mesh, axis_mapping, get_capacity_factor(cfg))

  out, _ = dsv3.dsv3(
      inputs,
      weights,
      yarn_freqs,
      splash_kernel,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=float(cfg.routed_scaling_factor),
      n_routing_groups=getattr(cfg, "n_routing_groups", 8),
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      qk_head_dim=getattr(cfg, "qk_nope_head_dim", 128),
      rope_head_dim=cfg.qk_rope_head_dim,
      num_query_heads=num_query_heads,
      mscale=float(getattr(cfg, "mscale", 1.0)),
      kv_lora_rank=cfg.kv_lora_rank,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=int(cfg.rope_factor),
      mesh=mesh,
      norm_fn=functools.partial(ops.rms_norm, epsilon=cfg.normalization_layer_epsilon),
      rope_fn=dsv3_mla.yarn,
      gmm_fn=functools.partial(
          jax.lax.ragged_dot,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=cfg.dtype,
      ),
      axis_mapping=axis_mapping,
      expert_axis_name="expert",
      capacity_factor=capacity_factor,
      segment_ids=None,
  )
  return out
