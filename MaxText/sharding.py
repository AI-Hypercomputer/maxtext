import enum
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from importlib import import_module

import jax
from jax import Array
from jax.interpreters import pxla
from jax.sharding import Mesh, PartitionSpec
from flax.linen.spmd import get_logical_axis_rules, logical_to_mesh_axes
import numpy as np


class Axis(enum.Enum):
  dp = "data"
  fsdp = "fsdp"
  tp = "tensor"
  pp = "stage"
  sp = "sequence"
  cp = "context"
  cp_ar = "context_autoregressive"
  tp_s = "tensor_sequence"
  ep = "expert"
  ar = "autoregressive"


class TensorType(enum.Enum):
  Activation = "activation"
  Weight = "weight"


def check_valid_mesh_axes(axes):
  valid_axis_values = {member.value for member in Axis}
  valid_axes_str = ", ".join(f"'{v}'" for v in valid_axis_values)

  def _check(axis_name: str):
    if axis_name not in valid_axis_values:
      raise ValueError(f"Invalid axis: '{axis_name}'. Valid axes are: {valid_axes_str}")

  for axis in axes:
    if isinstance(axis, (list, tuple)):
      for sub_axis in axis:
        if sub_axis is not None:
          _check(sub_axis)
    elif axis is not None:
      _check(axis)


def create_mesh(axes: Dict[Any, int], validate_mesh_axes: bool = True) -> Mesh:
  if validate_mesh_axes:
    check_valid_mesh_axes(axes)

  return Mesh(
      devices=np.array(jax.devices()).reshape(tuple(axes.values())),
      axis_names=tuple([key.value for key in axes.keys()]),
  )


class MeshSharding(ABC):

  def __init__(self, validate_mesh_axes = True):
     self.validate_mesh_axes = validate_mesh_axes

  @abstractmethod
  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    pass

  @abstractmethod
  def check_final_state(self):
     pass

  def shard(self, tensor: Array, *args: Any, **kwargs) -> Array:  # returns PyTree with shardings
    mesh = pxla.thread_resources.env.physical_mesh
    # like linen, no-op outside the context of a mesh
    if not mesh.devices.shape:
      return tensor

    kwargs["tensor_type"] = TensorType.Activation
    shardings = self(*args, **kwargs)
    with mesh:
      return jax.lax.with_sharding_constraint(tensor, shardings)

  def maybe_check_valid_mesh_axes(self, axes: PartitionSpec | Tuple[str]) -> None:
    if self.validate_mesh_axes:
      check_valid_mesh_axes(axes)


class LogicalAxisRulesSharding(MeshSharding):

  def check_valid_logical_axes(self, axes, axis_rules):
     for axis in axes:
        axis_found = False
        for mapping in axis_rules:
           if axis == mapping[0]:
            axis_found = True
            break
        if not axis_found and axis is not None:
          raise Exception(f"Logical axis {axis} not found  in axis rules:\n {axis_rules}")

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    axis_rules = get_logical_axis_rules()

    self.check_valid_logical_axes(axes, axis_rules)
    mesh_axes = logical_to_mesh_axes(axes, axis_rules)
    self.maybe_check_valid_mesh_axes(mesh_axes)

    return mesh_axes

  def check_final_state(self):
     pass


def assert_matches_logical_axis_rules(axis_mappings, axes):
   lar_axis_mappings = LogicalAxisRulesSharding()(a=axes)
   assert PartitionSpec(*axis_mappings) == lar_axis_mappings, "Axis mappings do not match that of logical_axis_rules"


def create_sharding_rules(sharding_rules_name: str):
    """Instantiate a sharding rules class by name.

    The name can be either:
      * A bare class name defined in this module (e.g. "LogicalAxisRulesSharding").
      * A fully qualified path (e.g. "MaxText.sharding.LogicalAxisRulesSharding").
    """
    name = sharding_rules_name

    if "." in name:
        module_path, class_name = name.rsplit(".", 1)
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as e:
            raise ValueError(f"Module '{module_path}' not found while loading '{name}'.") from e
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ValueError(f"Class '{class_name}' not found in module '{module_path}'.")
    else:
        cls = globals().get(name)
        if cls is None:
            raise ValueError(f"Class '{name}' not found in module '{__name__}'. Available: {list(globals().keys())}")

    return cls()
