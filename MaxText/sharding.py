import enum
from abc import ABC, abstractmethod
from typing import Any, Dict
from importlib import import_module

import jax
from jax import Array
from jax.interpreters import pxla
from jax.sharding import Mesh, PartitionSpec
from flax.linen.spmd import get_logical_axis_rules, logical_to_mesh_axes
import numpy as np


class Axis(enum.Enum):
  DP = "data"
  FSDP = "fsdp"
  TP = "tensor"
  PP = "stage"
  FSDP_T = "fsdp_transpose"
  SP = "sequence"
  CP = "context"
  CP_AR = "context_autoregressive"
  TP_T = "tensor_transpose"
  TP_S = "tensor_sequence"
  EP = "expert"
  AR = "autoregressive"


class TensorType(enum.Enum):
  ACT = "activation"
  WT = "weight"
  CACHE = "cache"


ACT, WT, CACHE = TensorType


def create_mesh(axes: Dict[Axis, int]) -> Mesh:
  return Mesh(
      devices=np.array(jax.devices()).reshape(tuple(axes.values())),
      axis_names=tuple([key.value for key in axes.keys()]),
  )


class MeshSharding(ABC):

  @abstractmethod
  def get(self, *args: Any, **kwargs) -> tuple[str | None, ...] | None:
    pass

  def assert_valid_axes(self, axes: tuple[str, ...]) -> None:
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

  def shard(self, tensor: Array, *args: Any, **kwargs) -> Array:  # returns PyTree with shardings
    mesh = pxla.thread_resources.env.physical_mesh
    # like linen, no-op outside the context of a mesh
    if not mesh.devices.shape:
      return tensor

    shardings = self.get(*args, **kwargs)
    # Unpack the tuple to pass them as separate arguments to PartitionSpec
    spec = PartitionSpec(*shardings)
    with mesh:
      return jax.lax.with_sharding_constraint(tensor, spec)


class LogicalAxisRulesSharding(MeshSharding):

  def get(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    mesh_axes = logical_to_mesh_axes(axes, get_logical_axis_rules())
    self.assert_valid_axes(mesh_axes)
    return mesh_axes


def create_sharding_rules(sharding_rules_name: str):
    """Instantiate a sharding rules class by name.

    The name can be either:
      * A bare class name defined in this module (e.g. "LogicalAxisRulesSharding").
      * A fully qualified path (e.g. "MaxText.sharding.LogicalAxisRulesSharding").

    Args:
      sharding_rules_name: Class name or fully-qualified path to the class.
      expected_base: Base type the resolved class must inherit from.
    Returns:
      An instance of the resolved class.
    Raises:
      ValueError: If the class or module can't be found, or type check fails.
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
