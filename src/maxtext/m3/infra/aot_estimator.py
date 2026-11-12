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

"""Virtual-topology CPU AOT peak HBM estimator and abstract sharding verifier."""

from absl import logging
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import importlib.util
import math
import os
import re
from typing import Any

from flax import nnx
import jax
from jax.experimental import topologies
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import optax

from maxtext.m3 import models as m3_models
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core import moe as m3_moe
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.infra import named_scopes as m3_named_scopes
from maxtext.m3.infra import offload as m3_offload
from maxtext.m3.infra import quantization as m3_quantization
from maxtext.m3.infra import remat as m3_remat
from maxtext.m3.infra import scan as m3_scan
from maxtext.optimizers import optimizers
from maxtext.utils import accelerator_to_spec_map

_GIB_BYTES = 1024**3
_HOST_MEMORY_KINDS = frozenset({"pinned_host", "unpinned_host"})

TOPOLOGY_HBM_GIB_MAP: Mapping[str, int] = {
    "tpu7x": 95,
    "v5p": 95,
    "v6e": 32,
    "v5e": 16,
    "v5litepod": 16,
    "v4": 32,
    "cpu": 95,
}

COLLECTIVE_KINDS: tuple[str, ...] = (
    "all-gather",
    "reduce-scatter",
    "all-reduce",
    "ragged-all-to-all",
)

_COLLECTIVE_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?[a-zA-Z0-9_.-]+\s*=.*?\s"
    r"(all-gather|reduce-scatter|all-reduce|ragged-all-to-all)(?:-start)?\s*\(\s*%"
)


class CapabilityStatus(enum.Enum):
  """Whether a static model capability is supported or a configured training feature is active."""

  SUPPORTED = "SUPPORTED"
  NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclasses.dataclass(frozen=True)
class AbstractShardingReport:
  """Abstract parameter sharding and byte footprint traced via `nnx.eval_shape`."""

  mesh_shape: dict[str, int]
  param_shardings: dict[str, NamedSharding]
  param_specs: dict[str, P]
  param_shapes: dict[str, tuple[int, ...]]
  param_dtypes: dict[str, jnp.dtype]
  device_arg_bytes: int
  host_arg_bytes: int
  capabilities: dict[str, CapabilityStatus]
  model_capabilities: dict[str, CapabilityStatus] = dataclasses.field(default_factory=dict)
  active_features: dict[str, CapabilityStatus] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class AOTHbmEstimate:
  """Structured AOT compilation HBM estimate, sharding map, and collective counts."""

  topology: str
  num_slices: int
  mesh_shape: dict[str, int]
  resident_peak_bytes: int
  raw_xla_peak_bytes: int
  temp_bytes: int
  argument_bytes: int
  output_bytes: int
  alias_bytes: int
  device_arg_bytes: int
  host_arg_bytes: int
  hbm_capacity_bytes: int
  fits_in_hbm: bool
  param_shardings: dict[str, NamedSharding]
  param_specs: dict[str, P]
  param_shapes: dict[str, tuple[int, ...]]
  collective_counts: dict[str, int]
  capabilities: dict[str, CapabilityStatus]
  model_capabilities: dict[str, CapabilityStatus] = dataclasses.field(default_factory=dict)
  active_features: dict[str, CapabilityStatus] = dataclasses.field(default_factory=dict)


def get_topology_hbm_capacity_bytes(topology: str | None) -> int:
  """Returns per-device HBM capacity in bytes for a target topology string."""
  if topology is None:
    norm = "cpu"
  else:
    norm = str(topology).strip().lower()
    if not norm or ":" in norm:
      raise ValueError(
          f"Unsupported topology {topology!r} for HBM capacity lookup; "
          "expected a non-empty MaxText topology name such as 'tpu7x-8' or 'v5p-8'."
      )
  prefix = norm.split("-", 1)[0]
  if prefix not in TOPOLOGY_HBM_GIB_MAP:
    raise ValueError(
        f"Unsupported topology {topology!r} for HBM capacity lookup; "
        f"expected prefix in {sorted(TOPOLOGY_HBM_GIB_MAP)}."
    )
  return int(TOPOLOGY_HBM_GIB_MAP[prefix]) * _GIB_BYTES


def has_libtpu_topology_support() -> bool:
  """Returns True if the host environment supports virtual TPU topologies without live PJRT init."""
  jax_platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
  active_platforms = {p.strip() for p in jax_platforms.split(",") if p.strip()}
  if "proxy" in active_platforms:
    return False
  if importlib.util.find_spec("libtpu") is not None:
    return True
  if active_platforms and "tpu" not in active_platforms:
    return False
  try:
    desc = topologies.get_topology_desc(
        platform="tpu",
        topology_name="tpu7x:2x2x1",
        chips_per_host_bounds=(2, 2, 1),
        num_slices=1,
    )
    return bool(getattr(desc, "devices", None))
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.info("Virtual TPU topology probe failed; treating virtual topologies as unsupported: %r", e)
    return False


def get_virtual_topology_devices(topology: str, num_slices: int = 1) -> Sequence[Any]:
  """Builds virtual PJRT TPU topology devices on a CPU host."""
  if int(num_slices) < 1:
    raise ValueError(f"num_slices must be >= 1, got {num_slices}.")
  clean_topo = "" if topology is None else str(topology).strip()
  if not clean_topo or ":" in clean_topo:
    raise ValueError(f"Unsupported topology {topology!r}; expected a non-empty MaxText topology name such as 'tpu7x-8'.")
  sys_char = accelerator_to_spec_map.get_system_characteristics(clean_topo)
  try:
    desc = topologies.get_topology_desc(
        platform=sys_char.platform,
        topology_name=sys_char.topology_name,
        chip_config_name=sys_char.chip_config_name,
        chips_per_host_bounds=sys_char.chips_per_host_bounds,
        num_slices=int(num_slices),
        wrap=sys_char.wrap,
    )
  except TypeError:
    desc = topologies.get_topology_desc(
        platform=sys_char.platform,
        topology_name=sys_char.topology_name,
        chips_per_host_bounds=sys_char.chips_per_host_bounds,
        num_slices=int(num_slices),
    )
  return desc.devices


def get_virtual_topology_mesh(
    config: Any,
    *,
    topology: str | None = None,
    num_slices: int | None = None,
) -> Mesh:
  """Constructs the 8-axis m3 `Mesh` over virtual TPU topology devices."""
  topo_str = topology if topology else (str(config.compile_topology) if config.compile_topology else "")
  if not topo_str:
    raise ValueError("topology or config.compile_topology must be non-empty.")
  slices = int(
      num_slices
      if num_slices is not None
      else (config.compile_topology_num_slices if config.compile_topology_num_slices > 0 else 1)
  )
  devices = get_virtual_topology_devices(topo_str, num_slices=slices)
  return m3_sharding.mesh_from_config(config, devices=devices)


def count_hlo_collectives(hlo_text: str) -> dict[str, int]:
  """Counts HLO collective instructions without double-counting async wrappers."""
  counts = {kind: 0 for kind in COLLECTIVE_KINDS}
  for line in hlo_text.splitlines():
    match = _COLLECTIVE_RE.match(line)
    if match:
      counts[match.group(1)] += 1
    elif "custom-call(" in line and "ragged_all_to_all" in line.lower():
      counts["ragged-all-to-all"] += 1
  return counts


def _make_optimizer_tx(config: Any, *, model: Any = None, mesh: Any = None) -> optax.GradientTransformation:
  """Builds the optax transformation matching `m3.train.loop.make_optimizer`."""
  tx = optimizers.get_optimizer(config, config.learning_rate, model=model, mesh=mesh)
  if config.gradient_clipping_threshold > 0:
    return optax.chain(optax.clip_by_global_norm(config.gradient_clipping_threshold), tx)
  return tx


def _batch_spec(sharding: m3_sharding.Sharding) -> P:
  """Returns the `(batch, length)` activation `PartitionSpec` for `sharding`."""
  return sharding("input_tokens", ("batch", "length"), m3_sharding.TensorType.ACTIVATION)


def _setup_model_for_aot(
    config: Any,
    mesh: Mesh,
    sharding: m3_sharding.Sharding | None,
    *,
    include_optimizer: bool,
) -> tuple[nnx.Module, nnx.Optimizer | None]:
  """Runs the canonical m3 setup pipeline (`construct -> scan -> remat -> quantize -> scopes -> offload`)."""
  with m3_sharding.enter_mesh_scope(mesh), nnx.use_eager_sharding(True):
    create_kw = {"sharding": sharding} if sharding is not None else {}
    model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(config.init_weights_seed), **create_kw)
    model = m3_scan.maybe_apply_scan(model, config)
    model = m3_remat.maybe_apply_remat(model, config)
    sharding_obj = sharding or m3_sharding.sharding_from_config(config)
    m3_remat.check_model_tags(model, config, mesh=mesh, sharding=sharding_obj)
    in_sharding = NamedSharding(mesh, m3_sharding.project_spec_to_mesh(_batch_spec(sharding_obj), mesh))
    num_devices = int(mesh.devices.size)
    global_batch = max(1, int(round(float(config.per_device_batch_size) * num_devices)))
    in_shape = (global_batch, int(config.max_target_length))
    tag_inputs = (
        jax.ShapeDtypeStruct(in_shape, jnp.int32, sharding=in_sharding),
        jax.ShapeDtypeStruct(in_shape, jnp.int32, sharding=in_sharding),
    )
    if config.quantization and m3_sharding.ambient_mesh_is_explicit():
      raise ValueError(
          "quantization is not supported with shard_mode='explicit' yet: qwix drops out_sharding on quantized"
          " matmuls. Use shard_mode='auto'."
      )
    model = m3_quantization.maybe_quantize(model, config, *tag_inputs)
    model = m3_named_scopes.add_named_scopes(model)
    model = m3_offload.maybe_offload_parameters(model, config)
    if not include_optimizer:
      return model, None
    optimizer = m3_offload.make_optimizer(model, _make_optimizer_tx(config, model=model, mesh=mesh), config)
    return model, optimizer


def _bind_abstract_shardings(node: Any, mesh: Mesh) -> None:
  """Attaches physical `NamedSharding`s to abstract `ShapeDtypeStruct` variables, skipping `MaskedNode`."""
  if node is None:
    return
  for _, variable in nnx.to_flat_state(nnx.state(node)):
    val = variable.get_raw_value()
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
      continue
    sharding = getattr(val, "sharding", None)
    if isinstance(sharding, NamedSharding):
      if sharding.mesh == mesh:
        continue
      mem_kind = getattr(sharding, "memory_kind", None)
      named = NamedSharding(mesh, sharding.spec, memory_kind=mem_kind) if mem_kind else NamedSharding(mesh, sharding.spec)
    else:
      meta_spec = variable.get_metadata().get("out_sharding")
      spec = meta_spec if (isinstance(meta_spec, P) and len(meta_spec) <= len(val.shape)) else P()
      named = NamedSharding(mesh, spec)
    variable.set_raw_value(jax.ShapeDtypeStruct(shape=val.shape, dtype=val.dtype, sharding=named))


def _loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  """Mean next-token cross-entropy loss in float32."""
  logits = model(batch["inputs"], batch["positions"])
  if isinstance(logits, tuple):
    logits = logits[0]
  logits_fp32 = logits.astype(jnp.float32)
  if not jax.sharding.get_abstract_mesh().empty and m3_sharding.ambient_mesh_is_explicit():
    target_sharding = jax.typeof(batch["targets"]).sharding
    target_spec = target_sharding.spec if hasattr(target_sharding, "spec") else P()
    losses = jax.sharding.auto_axes(
        optax.softmax_cross_entropy_with_integer_labels,
        out_sharding=target_spec,
    )(logits_fp32, batch["targets"])
  else:
    losses = optax.softmax_cross_entropy_with_integer_labels(logits_fp32, batch["targets"])
  return jnp.mean(losses)


@nnx.jit(donate_argnums=(0, 1))
def _aot_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
) -> jax.Array:
  """Single forward/backward/optimizer update step with donated state."""
  loss, grads = nnx.value_and_grad(_loss_fn)(model, batch)
  optimizer.update(model, grads)
  return loss


@nnx.jit
def _aot_forward_step(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  """Single forward loss step without optimizer state."""
  return _loss_fn(model, batch)


def _spec_contains_axis(spec: P, target_axis: str) -> bool:
  """Returns True if `target_axis` appears anywhere in `spec`."""
  for entry in spec:
    if entry == target_axis:
      return True
    if isinstance(entry, tuple) and target_axis in entry:
      return True
  return False


def _detect_model_capabilities(param_specs: Mapping[str, P]) -> dict[str, CapabilityStatus]:
  """Detects static architecture capabilities solely from parameter `PartitionSpec`s."""
  has_routed_experts = any(
      ".experts." in path or (_spec_contains_axis(spec, "expert") and "experts" in path)
      for path, spec in param_specs.items()
  )
  has_seq_parallel_axes = any(
      _spec_contains_axis(spec, "tensor_sequence") or _spec_contains_axis(spec, "context")
      for spec in param_specs.values()
  )
  return {
      "expert_parallelism": (CapabilityStatus.SUPPORTED if has_routed_experts else CapabilityStatus.NOT_APPLICABLE),
      "sequence_parallel_capable": (
          CapabilityStatus.SUPPORTED if has_seq_parallel_axes else CapabilityStatus.NOT_APPLICABLE
      ),
  }


def _detect_active_features(
    config: Any,
    param_shardings: Mapping[str, NamedSharding],
) -> dict[str, CapabilityStatus]:
  """Detects active configured parallelism axes and infra transforms from `config` and shardings."""
  has_host_offload_params = any(
      getattr(sharding, "memory_kind", None) in _HOST_MEMORY_KINDS for sharding in param_shardings.values()
  )
  remat_str = str(config.remat_policy).lower() if config.remat_policy else "none"
  tsp_active = int(config.ici_tensor_sequence_parallelism) > 1 or int(config.dcn_tensor_sequence_parallelism) > 1
  cp_active = int(config.ici_context_parallelism) > 1 or int(config.dcn_context_parallelism) > 1
  return {
      "tensor_sequence_parallelism": (CapabilityStatus.SUPPORTED if tsp_active else CapabilityStatus.NOT_APPLICABLE),
      "context_parallelism": (CapabilityStatus.SUPPORTED if cp_active else CapabilityStatus.NOT_APPLICABLE),
      "sequence_parallel": (CapabilityStatus.SUPPORTED if (tsp_active or cp_active) else CapabilityStatus.NOT_APPLICABLE),
      "scan_layers": (CapabilityStatus.SUPPORTED if bool(config.scan_layers) else CapabilityStatus.NOT_APPLICABLE),
      "remat": (CapabilityStatus.SUPPORTED if remat_str != "none" else CapabilityStatus.NOT_APPLICABLE),
      "parameter_offload": (
          CapabilityStatus.SUPPORTED
          if (bool(config.parameter_memory_host_offload) or has_host_offload_params)
          else CapabilityStatus.NOT_APPLICABLE
      ),
      "optimizer_offload": (
          CapabilityStatus.SUPPORTED if bool(config.optimizer_memory_host_offload) else CapabilityStatus.NOT_APPLICABLE
      ),
  }


def _detect_capabilities(
    config: Any,
    param_specs: Mapping[str, P],
    param_shardings: Mapping[str, NamedSharding],
) -> tuple[dict[str, CapabilityStatus], dict[str, CapabilityStatus], dict[str, CapabilityStatus]]:
  """Returns `(combined_capabilities, model_capabilities, active_features)`."""
  model_caps = _detect_model_capabilities(param_specs)
  active_feats = _detect_active_features(config, param_shardings)
  return {**model_caps, **active_feats}, model_caps, active_feats


def _per_device_bytes(leaf: Any) -> tuple[int, int]:
  """Returns `(device_bytes, host_bytes)` held per device for `leaf`."""
  if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
    return 0, 0
  sharding = getattr(leaf, "sharding", None)
  shard_shape = sharding.shard_shape(leaf.shape) if sharding is not None else leaf.shape
  nbytes = math.prod(shard_shape) * leaf.dtype.itemsize
  if getattr(sharding, "memory_kind", None) in _HOST_MEMORY_KINDS:
    return 0, nbytes
  return nbytes, 0


def _tree_bytes(structure: Any) -> tuple[int, int]:
  """Returns total device and host bytes across leaves."""
  device, host = 0, 0
  for leaf in jax.tree.leaves(structure):
    d, h = _per_device_bytes(leaf)
    device += d
    host += h
  return device, host


def _extract_param_maps(
    model: nnx.Module,
    mesh: Mesh,
) -> tuple[
    dict[str, NamedSharding],
    dict[str, P],
    dict[str, tuple[int, ...]],
    dict[str, jnp.dtype],
]:
  """Extracts dot-separated parameter sharding, spec, shape, and dtype dictionaries."""
  param_shardings: dict[str, NamedSharding] = {}
  param_specs: dict[str, P] = {}
  param_shapes: dict[str, tuple[int, ...]] = {}
  param_dtypes: dict[str, jnp.dtype] = {}

  state = nnx.state(model, nnx.Param)
  for path, variable in nnx.to_flat_state(state):
    key = ".".join(str(part) for part in path)
    val = variable.get_value()
    sharding = getattr(val, "sharding", None)
    if isinstance(sharding, NamedSharding):
      named_sharding = sharding
      spec = sharding.spec
    else:
      meta_spec = variable.get_metadata().get("out_sharding")
      spec = meta_spec if isinstance(meta_spec, P) else P()
      named_sharding = NamedSharding(mesh, spec)
    param_shardings[key] = named_sharding
    param_specs[key] = spec
    param_shapes[key] = tuple(int(d) for d in val.shape)
    param_dtypes[key] = jnp.dtype(val.dtype)

  return param_shardings, param_specs, param_shapes, param_dtypes


def _resolve_mesh_and_topology(
    config: Any,
    *,
    mesh: Mesh | None,
    topology: str | None,
    num_slices: int | None,
) -> tuple[Mesh, str, int]:
  """Resolves target `Mesh`, topology label, and slice count."""
  topo_str = topology if topology else (str(config.compile_topology) if config.compile_topology else "")
  slices = int(
      num_slices
      if num_slices is not None
      else (config.compile_topology_num_slices if config.compile_topology_num_slices > 0 else 1)
  )
  if mesh is not None:
    resolved_topo = topo_str or str(getattr(mesh.devices.flat[0], "platform", "cpu"))
    return mesh, resolved_topo, slices
  if topo_str:
    return get_virtual_topology_mesh(config, topology=topo_str, num_slices=slices), topo_str, slices
  resolved_mesh = m3_sharding.mesh_from_config(config)
  resolved_topo = str(getattr(resolved_mesh.devices.flat[0], "platform", "cpu"))
  return resolved_mesh, resolved_topo, slices


def trace_abstract_sharding(
    config: Any,
    *,
    mesh: Mesh | None = None,
    sharding: m3_sharding.Sharding | None = None,
    topology: str | None = None,
    num_slices: int | None = None,
    include_optimizer: bool = True,
) -> AbstractShardingReport:
  """Traces abstract model and optimizer state via `nnx.eval_shape` without weight allocation."""
  m3_moe.validate_moe_config(config)
  resolved_mesh, _, _ = _resolve_mesh_and_topology(config, mesh=mesh, topology=topology, num_slices=num_slices)
  m3_kernels.validate_attention_config(config, mesh=resolved_mesh)
  is_tpu_cross = (
      resolved_mesh.devices.size > 0
      and getattr(resolved_mesh.devices.flat[0], "platform", "") == "tpu"
      and jax.default_backend() != "tpu"
  )
  # Clear outer abstract mesh around nnx.eval_shape so Flax _to_variable skips .shape on Optax MaskedNode.
  with (
      jax.default_matmul_precision(
          config.matmul_precision.value if hasattr(config.matmul_precision, "value") else str(config.matmul_precision)
      ),
      jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())),
      m3_remat.maybe_tokamax_cross_compile(is_tpu_cross),
  ):
    model, optimizer = nnx.eval_shape(
        lambda: _setup_model_for_aot(config, resolved_mesh, sharding, include_optimizer=include_optimizer)
    )
  _bind_abstract_shardings(model, resolved_mesh)
  _bind_abstract_shardings(optimizer, resolved_mesh)

  param_shardings, param_specs, param_shapes, param_dtypes = _extract_param_maps(model, resolved_mesh)
  state_tree = (nnx.state(model), nnx.state(optimizer)) if optimizer is not None else nnx.state(model)
  device_arg_bytes, host_arg_bytes = _tree_bytes(state_tree)
  capabilities, model_capabilities, active_features = _detect_capabilities(config, param_specs, param_shardings)
  return AbstractShardingReport(
      mesh_shape={str(k): int(v) for k, v in resolved_mesh.shape.items()},
      param_shardings=param_shardings,
      param_specs=param_specs,
      param_shapes=param_shapes,
      param_dtypes=param_dtypes,
      device_arg_bytes=int(device_arg_bytes),
      host_arg_bytes=int(host_arg_bytes),
      capabilities=capabilities,
      model_capabilities=model_capabilities,
      active_features=active_features,
  )


def compile_aot_hbm_estimate(
    config: Any,
    *,
    mesh: Mesh | None = None,
    sharding: m3_sharding.Sharding | None = None,
    topology: str | None = None,
    num_slices: int | None = None,
    hbm_capacity_bytes: int | None = None,
    include_optimizer: bool = True,
    compiler_options: Mapping[str, Any] | None = None,
) -> AOTHbmEstimate:
  """Compiles the m3 step AOT on a virtual or provided mesh and returns HBM & sharding metrics."""
  m3_moe.validate_moe_config(config)
  resolved_mesh, resolved_topo, resolved_slices = _resolve_mesh_and_topology(
      config, mesh=mesh, topology=topology, num_slices=num_slices
  )
  m3_kernels.validate_attention_config(config, mesh=resolved_mesh)
  capacity_bytes = (
      int(hbm_capacity_bytes) if hbm_capacity_bytes is not None else get_topology_hbm_capacity_bytes(resolved_topo)
  )

  is_tpu_cross = (
      resolved_mesh.devices.size > 0
      and getattr(resolved_mesh.devices.flat[0], "platform", "") == "tpu"
      and jax.default_backend() != "tpu"
  )
  with (
      jax.default_matmul_precision(
          config.matmul_precision.value if hasattr(config.matmul_precision, "value") else str(config.matmul_precision)
      ),
      jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())),
      m3_remat.maybe_tokamax_cross_compile(is_tpu_cross),
  ):
    model, optimizer = nnx.eval_shape(
        lambda: _setup_model_for_aot(config, resolved_mesh, sharding, include_optimizer=include_optimizer)
    )
  _bind_abstract_shardings(model, resolved_mesh)
  _bind_abstract_shardings(optimizer, resolved_mesh)

  param_shardings, param_specs, param_shapes, _ = _extract_param_maps(model, resolved_mesh)
  capabilities, model_capabilities, active_features = _detect_capabilities(config, param_specs, param_shardings)

  sharding_obj = sharding or m3_sharding.sharding_from_config(config)
  b_spec = m3_sharding.project_spec_to_mesh(_batch_spec(sharding_obj), resolved_mesh)
  batch_sharding = NamedSharding(resolved_mesh, b_spec)
  num_devices = int(resolved_mesh.devices.size)
  global_batch = max(1, int(round(float(config.per_device_batch_size) * num_devices)))
  seq_len = int(config.max_target_length)
  batch = {
      "inputs": jax.ShapeDtypeStruct((global_batch, seq_len), jnp.int32, sharding=batch_sharding),
      "targets": jax.ShapeDtypeStruct((global_batch, seq_len), jnp.int32, sharding=batch_sharding),
      "positions": jax.ShapeDtypeStruct((global_batch, seq_len), jnp.int32, sharding=batch_sharding),
  }

  state_tree = (nnx.state(model), nnx.state(optimizer), batch) if optimizer is not None else (nnx.state(model), batch)
  tree_device_bytes, tree_host_bytes = _tree_bytes(state_tree)

  with (
      jax.default_matmul_precision(
          config.matmul_precision.value if hasattr(config.matmul_precision, "value") else str(config.matmul_precision)
      ),
      jax.set_mesh(resolved_mesh),
      m3_remat.maybe_tokamax_cross_compile(is_tpu_cross),
  ):
    if optimizer is not None:
      lowered = _aot_train_step.lower(model, optimizer, batch)
    else:
      lowered = _aot_forward_step.lower(model, batch)
    compiled = lowered.compile(compiler_options=dict(compiler_options)) if compiler_options else lowered.compile()

  stats = compiled.memory_analysis()
  if stats is None:
    raise ValueError(f"Compiled executable returned no memory_analysis() for topology {resolved_topo!r}.")
  arg_bytes = int(getattr(stats, "argument_size_in_bytes", 0) or 0)
  out_bytes = int(getattr(stats, "output_size_in_bytes", 0) or 0)
  alias_bytes = int(getattr(stats, "alias_size_in_bytes", 0) or 0)
  temp_bytes = int(getattr(stats, "temp_size_in_bytes", 0) or 0)
  xla_host_arg_bytes = int(getattr(stats, "host_argument_size_in_bytes", 0) or 0)
  host_out_bytes = int(getattr(stats, "host_output_size_in_bytes", 0) or 0)
  host_temp_bytes = int(getattr(stats, "host_temp_size_in_bytes", 0) or 0)
  raw_xla_peak_bytes = int(getattr(stats, "peak_memory_in_bytes", 0) or 0)

  resident_peak_bytes = arg_bytes + out_bytes - alias_bytes + temp_bytes
  if xla_host_arg_bytes > 0:
    device_arg_bytes = arg_bytes
    host_arg_bytes = xla_host_arg_bytes
  elif tree_host_bytes > 0:
    host_arg_bytes = int(tree_host_bytes)
    device_arg_bytes = int(tree_device_bytes)
    # On XLA:CPU, memory_analysis() folds pinned_host buffers into argument_size_in_bytes.
    resident_peak_bytes = max(0, resident_peak_bytes - min(arg_bytes, host_arg_bytes))
  else:
    host_arg_bytes = 0
    device_arg_bytes = arg_bytes if arg_bytes > 0 else int(tree_device_bytes)

  has_host_buffers = any(x > 0 for x in (xla_host_arg_bytes, tree_host_bytes, host_out_bytes, host_temp_bytes))
  fits_in_hbm = (resident_peak_bytes <= capacity_bytes) and (has_host_buffers or raw_xla_peak_bytes <= capacity_bytes)
  collective_counts = count_hlo_collectives(compiled.as_text())

  return AOTHbmEstimate(
      topology=resolved_topo,
      num_slices=resolved_slices,
      mesh_shape={str(k): int(v) for k, v in resolved_mesh.shape.items()},
      resident_peak_bytes=resident_peak_bytes,
      raw_xla_peak_bytes=raw_xla_peak_bytes,
      temp_bytes=temp_bytes,
      argument_bytes=arg_bytes,
      output_bytes=out_bytes,
      alias_bytes=alias_bytes,
      device_arg_bytes=device_arg_bytes,
      host_arg_bytes=host_arg_bytes,
      hbm_capacity_bytes=capacity_bytes,
      fits_in_hbm=fits_in_hbm,
      param_shardings=param_shardings,
      param_specs=param_specs,
      param_shapes=param_shapes,
      collective_counts=collective_counts,
      capabilities=capabilities,
      model_capabilities=model_capabilities,
      active_features=active_features,
  )


def _axis_product(mesh_shape: Mapping[str, int], axis_entry: Any) -> int:
  """Returns the product of mesh axis sizes for one `PartitionSpec` dimension entry."""
  if axis_entry is None:
    return 1
  axes = (axis_entry,) if isinstance(axis_entry, str) else tuple(axis_entry)
  return math.prod(int(mesh_shape.get(ax, 1)) for ax in axes)


def verify_sharding_golden(
    target: AOTHbmEstimate | AbstractShardingReport | Mapping[str, NamedSharding | P],
    golden_specs: Mapping[str, P | CapabilityStatus],
    *,
    mesh: Mesh | None = None,
    param_shapes: Mapping[str, tuple[int, ...]] | None = None,
    check_divisibility: bool = True,
    check_unsharded_2d_weights: bool = False,
    unsharded_allowlist: Sequence[str] = (),
) -> dict[str, CapabilityStatus]:
  """Verifies parameter `PartitionSpec`s against a golden map without test-only shims."""
  model_cap_keys = frozenset({"expert_parallelism", "sequence_parallel_capable"})
  active_feat_keys = frozenset(
      {
          "tensor_sequence_parallelism",
          "context_parallelism",
          "sequence_parallel",
          "scan_layers",
          "remat",
          "parameter_offload",
          "optimizer_offload",
      }
  )
  all_cap_keys = model_cap_keys | active_feat_keys
  is_bare_mapping = not isinstance(target, (AOTHbmEstimate, AbstractShardingReport))

  if not is_bare_mapping:
    actual_specs = target.param_specs
    actual_shapes = dict(param_shapes) if param_shapes is not None else dict(target.param_shapes)
    mesh_shape = dict(target.mesh_shape)
    capabilities = dict(target.capabilities)
  else:
    actual_specs = {}
    inferred_mesh_shape: dict[str, int] = {}
    for key, val in target.items():
      if isinstance(val, NamedSharding):
        actual_specs[key] = val.spec
        if not inferred_mesh_shape and hasattr(val.mesh, "shape"):
          inferred_mesh_shape = {str(k): int(v) for k, v in val.mesh.shape.items()}
      elif isinstance(val, P):
        actual_specs[key] = val
      else:
        raise TypeError(f"Unsupported sharding entry for {key!r}: {type(val).__name__}")
    actual_shapes = dict(param_shapes) if param_shapes is not None else {}
    mesh_shape = {str(k): int(v) for k, v in mesh.shape.items()} if mesh is not None else inferred_mesh_shape
    capabilities = _detect_model_capabilities(actual_specs)

  results: dict[str, CapabilityStatus] = {}
  mismatches: list[str] = []

  for key, expected in golden_specs.items():
    if isinstance(expected, CapabilityStatus):
      if "." not in key and "/" not in key:
        if key not in all_cap_keys:
          mismatches.append(f"Unknown capability key {key!r}; expected one of {sorted(all_cap_keys)}.")
          continue
        if is_bare_mapping and key in active_feat_keys:
          mismatches.append(
              f"Active-feature capability {key!r} requires an AOTHbmEstimate or AbstractShardingReport target, "
              "not a bare Mapping."
          )
          continue
      if expected == CapabilityStatus.NOT_APPLICABLE:
        if key in capabilities and capabilities[key] != CapabilityStatus.NOT_APPLICABLE:
          mismatches.append(f"Capability {key!r} expected NOT_APPLICABLE, got {capabilities[key].value}.")
        elif key in actual_specs:
          mismatches.append(f"Parameter {key!r} expected NOT_APPLICABLE (absent), but found spec {actual_specs[key]!r}.")
        results[key] = CapabilityStatus.NOT_APPLICABLE
      else:
        if key in capabilities and capabilities[key] != CapabilityStatus.SUPPORTED:
          mismatches.append(f"Capability {key!r} expected SUPPORTED, got {capabilities[key].value}.")
        elif key not in capabilities and key not in actual_specs:
          mismatches.append(f"Expected supported key {key!r} missing from parameter specs.")
        results[key] = CapabilityStatus.SUPPORTED
      continue

    if key not in actual_specs:
      mismatches.append(f"Missing parameter {key!r} (expected {expected!r}).")
      continue
    actual = actual_specs[key]
    if actual != expected:
      mismatches.append(f"Parameter {key!r}: expected {expected!r}, got {actual!r}.")
    else:
      results[key] = CapabilityStatus.SUPPORTED

  if check_divisibility and mesh_shape and actual_shapes:
    for key, spec in actual_specs.items():
      shape = actual_shapes.get(key)
      if not shape:
        continue
      for dim_idx, axis_entry in enumerate(spec):
        if dim_idx >= len(shape) or axis_entry is None:
          continue
        shards = _axis_product(mesh_shape, axis_entry)
        if shards > 1 and (shape[dim_idx] % shards != 0):
          mismatches.append(
              f"Parameter {key!r} dim {dim_idx} (size {shape[dim_idx]}) is not divisible by "
              f"mesh axis {axis_entry!r} (size {shards})."
          )

  if check_unsharded_2d_weights and actual_shapes:
    total_mesh_devices = math.prod(mesh_shape.values()) if mesh_shape else 1
    if total_mesh_devices > 1:
      for key, spec in actual_specs.items():
        shape = actual_shapes.get(key, ())
        if len(shape) < 2:
          continue
        if any(allow in key for allow in unsharded_allowlist):
          continue
        has_active_shard = any(
            axis_entry is not None and _axis_product(mesh_shape, axis_entry) > 1 for axis_entry in spec
        )
        if not has_active_shard:
          mismatches.append(
              f"2D+ parameter {key!r} with shape {shape} is fully replicated ({spec!r}) " f"on mesh {mesh_shape}."
          )

  if mismatches:
    raise AssertionError("Sharding golden verification failed:\n  " + "\n  ".join(mismatches))

  return results
