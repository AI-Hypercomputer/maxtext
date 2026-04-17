"""Explicit sharding helpers for the NNX experimental track."""

from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain
from typing import TypeAlias

import jax
import numpy as np
from jax.sharding import AxisType, PartitionSpec as P


class TensorType(Enum):
  Weight = "weight"
  Activation = "activation"
  Input = "input"
  Output = "output"


MeshAxis: TypeAlias = str
MeshAxisMapping: TypeAlias = MeshAxis | tuple[MeshAxis, ...] | None
LogicalAxis: TypeAlias = str | None


MESH_AXES = (
  ("dp", ("ici_dp_parallelism", "ici_data_parallelism"), 1),
  ("pp", ("ici_pipeline_parallelism",), 1),
  ("fsdp", ("ici_fsdp_parallelism",), -1),
  ("fsdp_t", ("ici_fsdp_transpose_parallelism",), 1),
  ("sp", ("ici_sequence_parallelism",), 1),
  ("cp", ("ici_context_parallelism",), 1),
  ("tp", ("ici_tensor_parallelism",), 1),
  ("tp_t", ("ici_tensor_transpose_parallelism",), 1),
  ("tp_s", ("ici_tensor_sequence_parallelism",), 1),
  ("ep", ("ici_expert_parallelism",), 1),
)


def _flatten_mesh_axes(mesh_axes: MeshAxisMapping) -> tuple[MeshAxis, ...]:
  if mesh_axes is None:
    return ()
  if isinstance(mesh_axes, tuple):
    return tuple(chain.from_iterable(_flatten_mesh_axes(axis) for axis in mesh_axes))
  return (mesh_axes,)


class Sharding(ABC):
  """Maps semantic tensor axes to explicit mesh PartitionSpecs."""

  @abstractmethod
  def map_axis(
    self,
    axis: str,
    tensor_name: str,
    tensor_type: TensorType,
  ) -> MeshAxisMapping:
    """Map one logical axis to mesh axis name(s), or None."""
    ...

  def spec(self, tensor_name, axes, tensor_type=TensorType.Weight):
    return P(*(
      None if axis is None else self.map_axis(axis, tensor_name, tensor_type)
      for axis in axes
    ))

  def __call__(self, tensor_name, axes, tensor_type=TensorType.Weight):
    return self.spec(tensor_name, axes, tensor_type)

  def weight_spec(self, tensor_name, axes):
    return self.spec(tensor_name, axes, TensorType.Weight)

  def activation_spec_for(self, tensor_name, axes):
    return self.spec(tensor_name, axes, TensorType.Activation)

  def input_spec_for(self, tensor_name, axes):
    return self.spec(tensor_name, axes, TensorType.Input)

  def output_spec_for(self, tensor_name, axes):
    return self.spec(tensor_name, axes, TensorType.Output)

  def init_weight_spec(self, tensor_name, in_axes, out_axes):
    """Initializer-time sharding for layers that flatten grouped dimensions."""

    def collapse(axes):
      mapped = [
        self.map_axis(axis, tensor_name, TensorType.Weight)
        for axis in axes
        if axis is not None
      ]
      mapped = [axis for axis in mapped if axis is not None]
      if not mapped:
        return None
      flattened = tuple(chain.from_iterable(_flatten_mesh_axes(axis) for axis in mapped))
      if len(flattened) == 1:
        return flattened[0]
      return flattened

    return P(collapse(in_axes), collapse(out_axes))

  def mesh_axis_size(self, axis_name):
    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None:
      return 1
    return int(mesh.shape.get(axis_name, 1))

  def token_ids_spec(self, ndim=2):
    match ndim:
      case 2:
        return self.input_spec_for("tokens", ("batch", "length"))
      case 3:
        return P(None, self.map_axis("batch", "tokens", TensorType.Input), None)
      case _:
        raise ValueError(f"Unsupported token ids rank: {ndim}")

  def targets_spec(self, ndim=2):
    match ndim:
      case 2:
        return self.input_spec_for("targets", ("batch", "length"))
      case 3:
        return P(None, self.map_axis("batch", "targets", TensorType.Input), None)
      case _:
        raise ValueError(f"Unsupported targets rank: {ndim}")

  def positions_spec(self, ndim=2):
    match ndim:
      case 2:
        return self.input_spec_for("positions", ("batch", "length"))
      case 3:
        return P(None, self.map_axis("batch", "positions", TensorType.Input), None)
      case _:
        raise ValueError(f"Unsupported positions rank: {ndim}")

  def mask_spec(self, ndim=4):
    return P(*(None for _ in range(ndim)))

  def sequence_spec(self, tensor_name="sequence", *, length_axis="norm_length"):
    return self.activation_spec_for(tensor_name, ("batch", length_axis, "embed"))

  def attention_spec(
    self,
    tensor_name="attention",
    *,
    batch_axis="batch",
    length_axis="length",
    head_axis="heads",
    dim_axis="head_dim",
  ):
    return self.activation_spec_for(tensor_name, (batch_axis, length_axis, head_axis, dim_axis))

  def query_spec(self, tensor_name="query", *, length_axis="length", dim_axis="head_dim"):
    return self.attention_spec(
      tensor_name,
      length_axis=length_axis,
      head_axis="heads",
      dim_axis=dim_axis,
    )

  def kv_spec(self, tensor_name="key", *, batch_axis="batch", length_axis="kv_length", dim_axis="head_dim"):
    return self.attention_spec(
      tensor_name,
      batch_axis=batch_axis,
      length_axis=length_axis,
      head_axis="kv_heads",
      dim_axis=dim_axis,
    )

  def mlp_spec(self, tensor_name="mlpwi", *, width_axis="mlp"):
    return self.activation_spec_for(tensor_name, ("batch", "length", width_axis))

  def logits_spec(self, tensor_name="logits"):
    return self.output_spec_for(tensor_name, ("embed_and_logits_batch", "length", "vocab"))

  def place_inputs(self, tokens, targets=None, positions=None, mask=None):
    placed_tokens = jax.device_put(tokens, self.token_ids_spec(getattr(tokens, "ndim", np.ndim(tokens))))
    placed_targets = None
    if targets is not None:
      placed_targets = jax.device_put(targets, self.targets_spec(getattr(targets, "ndim", np.ndim(targets))))
    placed_positions = None
    if positions is not None:
      placed_positions = jax.device_put(
        positions,
        self.positions_spec(getattr(positions, "ndim", np.ndim(positions))),
      )
    placed_mask = None
    if mask is not None:
      placed_mask = jax.device_put(mask, self.mask_spec(getattr(mask, "ndim", np.ndim(mask))))
    return placed_tokens, placed_targets, placed_positions, placed_mask


class LlamaSharding(Sharding):
  """Concrete sharding rules for Llama model."""

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxisMapping:
    match axis, tensor_name, tensor_type:
      case "batch",                            _,                  _:             return "dp"
      case "embed_and_logits_batch",           _,                  _:             return ("dp", "pp")
      case "norm",                             _,                  _:             return ("fsdp", "fsdp_t")
      case "embed",                            _,                  _:             return ("fsdp", "fsdp_t")
      case "vocab",                            _,                  _:             return ("tp", "tp_s")
      case ("heads" | "q_heads" | "kv_heads" | "qkv_heads"), _,    _:             return ("tp", "tp_t")
      case "head_dim",                         _,                  _:             return None
      case ("length" | "kv_length"),           _,                  _:             return None
      case "norm_length",                      _,                  _:             return ("sp", "cp")
      case ("mlp" | "expert_mlp"),             _,                  _:             return ("tp", "tp_t", "tp_s")
      case _, _, _:
        assert False, f"Unexpected logical axis name for sharding: {axis=} {tensor_name=} {tensor_type=}"
        return None


def create_mesh(cfg=None, **overrides):
  """Create an explicit mesh from ici_*_parallelism settings."""
  cfg = cfg or {}

  def get(keys, default):
    for key in keys:
      if key in overrides:
        return overrides[key]
      if key in cfg:
        return cfg[key]
    return default

  num_devices = len(jax.devices())
  dims = {axis: get(keys, default) for axis, keys, default in MESH_AXES}

  auto = [key for key, value in dims.items() if value == -1]
  assert len(auto) <= 1, f"At most one axis can be -1, got: {auto}"

  if auto:
    fixed = int(np.prod([value for value in dims.values() if value != -1]))
    assert num_devices % fixed == 0, f"{num_devices} devices not divisible by fixed dims {fixed}"
    dims[auto[0]] = num_devices // fixed

  shape = tuple(int(value) for value in dims.values())
  total = int(np.prod(shape))
  assert total == num_devices, f"Mesh {dims} needs {total} devices, have {num_devices}"

  axis_types = (AxisType.Explicit,) * len(shape)
  return jax.make_mesh(shape, tuple(dims.keys()), axis_types=axis_types)
