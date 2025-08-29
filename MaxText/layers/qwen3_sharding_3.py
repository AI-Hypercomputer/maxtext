import enum
from typing import Tuple

from jax.sharding import PartitionSpec

from MaxText.sharding import MeshSharding, TensorType as TT, Axis
dp, fsdp, tp, pp, sp, cp, cp_ar, tp_s, ep, ar = Axis


class MeshAxis(enum.Enum):
  x = "x"
  y = "y"
  z = "z"


class MeshAxisMap:
  def __init__(self, config):
    self.axis_map = {}
    self.config = config
    # NOTE: this would come from ici_fsdp_parallelism, etc.
    self.config.parallelisms = {
      fsdp: 8,
      cp: 8,
      tp: 4,
      ep: 4
    }
    # NOTE: this would be set in yml
    self.config.shared_axes = [
      set(ep, tp),
      set(fsdp, cp),
    ]

    self.init_axis_map()

  def init_axis_map(self):
    # TODO: handle 2 vs. 3 ICI axes
    available_mesh_axes = set(MeshAxis.x, MeshAxis.y, MeshAxis.z)

    # iterate through the sets of parallelisms that share axes and map the members of each set to an available mesh axis
    for shared_set in self.config.shared_axes:
      mesh_axis = available_mesh_axes.pop()
      for parallelism in shared_set:
        # if this axis is flagged as shared it should actually be enabled (TODO: or we could just ignore it?)
        assert self.config.parallelisms[parallelism] > 1
        self.axis_map[parallelism] = mesh_axis

    # find remaining, parallelisms, which do not share axes, and map each to an available mesh axis
    for parallelism, value in self.config.parallelisms.items():
      if value > 1:
        if parallelism not in self.axis_map:
          mesh_axis = available_mesh_axes.pop()
          self.axis_map[parallelism] = mesh_axis

  # NOTE: this is a tuple of tuples because it corresponds to all the parallelisms of one axis (one tuple) for all
  #       axes in the tensor (another tuple)
  def resolve_axis(self, mapped_axes: Tuple[Tuple[Axis]]) -> MeshAxis:
    mesh_axes = []
    for axis in mapped_axes:
      axis_mappings = []
      for parallelism in axis:
        axis_mappings.append(self.axis_map[parallelism])
      mesh_axes.append(axis_mappings)

    return mesh_axes


class Qwen3AxisShardingV4(MeshSharding):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.mesh_axis_map = MeshAxisMap(config)

  def __call__(self, *args: Any, **kwargs) -> PartitionSpec:
    axes = kwargs["a"]
    tensor_name = kwargs["t"]
    tensor_type = kwargs.get("tensor_type", TT.Weight)

    match tensor_name:
      case "mlp":
        # NOTE: each row here could feature multiple parallelisms but we leave each singular for now
        mapped_axes = ((ep,)
                       (fsdp,))
      case "attn":
        mapped_axes = ((tp,),
                       (cp,))

    resolved_axes = self.mesh_axis_map.resolve_axes(mapped_axes)

    return PartitionSpec(*resolved_axes)
