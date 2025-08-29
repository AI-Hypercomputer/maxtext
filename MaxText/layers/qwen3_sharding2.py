import enum
import logging

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, Axis
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar = Axis


class FallbackPolicy(enum.Enum):
  Always = "always"
  Allowed = "allowed"
  Never = "never"


class MeshAxes:
  def __init__(self, config):
    self.used_axes = set()
    self.config = config
    # NOTE: these would be set in config / yml
    # TODO: actually specifying in code is better - should be in the below V3 class tho
    self.fallback_on_axis_clash = FallbackPolicy.Always
    self.allowed_fallbacks = set()

  def unused_axes(self, mesh_axes):
      mapped_axes = ()
      for axis in mesh_axes:
        if axis not in self.used_axes:
          self.used_axes.add(axis)
          mapped_axes.append(axis)
        else:
          # if we should never fall back or we should only fall back for allowed axes and this isn't one of them, err
          if (self.fallback_on_axis_clash == FallbackPolicy.Never or
              axis in self.allowed_fallbacks):
            raise Exception(f"{axis=} mappings clash with another axis in {mesh_axes=} and policy does not allow fallback")
          logging.warning("f{axis=} mappings clash with another axis in {mesh_axes=} so falling back to reduced axis set")

      return mapped_axes


class Qwen3AxisShardingV3(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)

    mesh_axes = MeshAxes(self.config)
    # we mimic linen's behavior, of matching axes right-to-left
    for axis in reversed(axes):
      match tensor_name, axis, tensor_type:
        case _, "embed", TT.Activation:
          mapped_axes = mesh_axes.unused_axes((tp,))

    return PartitionSpec(*mapped_axes)
