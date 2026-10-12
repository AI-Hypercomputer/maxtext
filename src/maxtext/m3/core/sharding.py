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

"""Sharding rules and mesh creation.

Model code never names a physical mesh axis. It refers to logical axes --
`"embed"`, `"heads"`, `"mlp"` -- and a `Sharding` instance supplied at
construction translates those into `PartitionSpec`s. Swapping a parallelism
strategy is then a change to one `Sharding` subclass, not a change to a model.

Two call sites, and only two:

  * At construction, for weight metadata::

      nnx.Linear(..., kernel_metadata={"sharding": s("gate", ("embed", "mlp"))})

  * Inside `__call__`, for activation constraints::

      x = s.constrain(x, "mlp_mid", ("batch", "length", "mlp"))

Parameters are sharded once by the initializer. Re-asserting a constraint on a
parameter afterwards collides with the training engine's deferred all-reduce
and Zero-1 resharding, so `constrain` is hardwired to `TensorType.ACTIVATION`
and cannot be used on weights.
"""

import abc
import enum
import math
from collections.abc import Mapping, Sequence

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# Data, fully-sharded data, tensor and expert parallelism, in mesh order.
#
# Order is not cosmetic: it is the order handed to `mesh_utils`, which places
# axes against the physical topology. Inserting an axis in the middle changes
# the placement of everything after it.
#
# TODO: This is a basic starting point, this list will be expanded soon.
# Legacy carries twelve axes, but that list is a union -- its seven strategy
# configs under `configs/custom_mesh_and_rule/` use between one and six. The
# ones m3 is likely to need next are `stage` (pipeline), `context` and
# `context_usp_ulysses` (context parallelism), and `tensor_sequence`.
MESH_AXIS_NAMES = ("dp", "fsdp", "tp", "ep")

# A logical axis maps to one mesh axis, several (their product shards the
# dimension), or None to leave the dimension replicated.
MeshAxes = str | tuple[str, ...] | None


class TensorType(enum.Enum):
  """Whether an axis is being mapped for a parameter or for an activation.

  The same logical axis often wants different treatment in each case. Under
  FSDP, `"embed"` shards weights across `fsdp` but is usually left replicated
  for activations.
  """

  WEIGHT = "weight"
  ACTIVATION = "activation"


def _ambient_mesh_is_explicit() -> bool:
  """Reports whether the mesh in scope uses `Explicit` axis types.

  The two sharding modes need different APIs and each rejects the other's:
  `with_sharding_constraint` asserts rather than shards under an explicit mesh,
  and `reshard` refuses a `PartitionSpec` naming auto axes. So `constrain` has
  to know which mode it is in.

  The mode is read off the mesh in scope rather than taken as a constructor
  argument or config flag, because the mesh is what the two APIs actually
  check. A flag can disagree with the mesh; the mesh cannot disagree with
  itself. Legacy does the same in `optimizers/reshape_utils.py`, though
  `utils/sharding.py` threads a `ShardMode` through instead.

  Raises:
    ValueError: If the mesh mixes explicit and non-explicit axes, where no
      single API is correct for the whole spec.
  """
  axis_types = tuple(jax.sharding.get_abstract_mesh().axis_types)
  explicit = sum(1 for t in axis_types if t == jax.sharding.AxisType.Explicit)
  if 0 < explicit < len(axis_types):
    raise ValueError(f"Mesh mixes explicit and non-explicit axis types ({axis_types}), which m3 does not support.")
  return explicit == len(axis_types) and explicit > 0


def _validate_mesh_axes(mapped: MeshAxes, axis: str, tensor_name: str) -> None:
  """Rejects mesh axis names that are not on the mesh.

  Without this a typo such as `"tensor"` for `"tp"` surfaces much later as an
  opaque sharding error, far from the rule that produced it.
  """
  if mapped is None:
    return
  names = (mapped,) if isinstance(mapped, str) else tuple(mapped)
  unknown = [name for name in names if name not in MESH_AXIS_NAMES]
  if unknown:
    raise ValueError(
        f"Sharding rule for logical axis {axis!r} of tensor {tensor_name!r} returned unknown mesh "
        f"axes {unknown}; expected names from {MESH_AXIS_NAMES}."
    )


class Sharding(abc.ABC):
  """Maps a model's logical axis names onto mesh axes.

  A subclass encodes one parallelism strategy. Strategies are meant to be
  reused across models, which is why the common logical vocabulary --
  `"embed"`, `"mlp"`, `"heads"`, `"kv_heads"`, `"norm"`, `"layers"` -- should
  mean the same thing everywhere. A model with a genuinely novel axis extends a
  strategy and delegates the rest::

      class PureFsdpSharding(Sharding):
        def map_axis(self, axis, tensor_name, tensor_type):
          match axis, tensor_type:
            case "batch", TensorType.ACTIVATION: return ("dp", "fsdp")
            case "embed", TensorType.WEIGHT: return "fsdp"
            case "embed", TensorType.ACTIVATION: return None
            case ("heads" | "kv_heads"), _: return "tp"
            case "mlp", _: return "tp"
            case _: raise KeyError(axis)

      class Qwen3NextSharding(PureFsdpSharding):
        def map_axis(self, axis, tensor_name, tensor_type):
          if axis == "gdn_head":
            return "tp"
          return super().map_axis(axis, tensor_name, tensor_type)

  Keeping rules per-strategy rather than per-model matters: legacy's rule table
  is model-agnostic, so N models and S strategies cost S tables. A rule set
  written per model costs N x S instead.

  Prefer raising on an unrecognised axis over a catch-all that returns None. A
  catch-all turns a misspelled logical axis into silent full replication, which
  costs performance without ever failing a test. Raising surfaces it the first
  time the model is constructed.
  """

  @abc.abstractmethod
  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    """Maps one logical axis to mesh axis name(s), or None to leave it replicated.

    Args:
      axis: Logical axis name, e.g. `"embed"`.
      tensor_name: Name of the tensor being sharded, for rules that need to
        distinguish otherwise identical axes.
      tensor_type: Whether this is a parameter or an activation.
    """

  def __call__(
      self,
      tensor_name: str,
      axes: Sequence[str | None],
      tensor_type: TensorType = TensorType.WEIGHT,
  ) -> P:
    """Resolves a tuple of logical axes into a `PartitionSpec`.

    A `None` entry in `axes` marks a dimension with no logical name, which is
    left replicated without consulting `map_axis`.

    Raises if two dimensions claim the same mesh axis. Legacy resolves that
    case through Flax's fallback rules, which skip a rule whose mesh axis is
    already taken and fall through to a shorter one -- hence the duplicate keys
    in `logical_axis_rules`. m3 has no fallback, so the conflict is reported
    here rather than surfacing as an opaque XLA error later.
    """
    resolved = []
    claimed: dict[str, str] = {}
    for axis in axes:
      if axis is None:
        resolved.append(None)
        continue
      mapped = self.map_axis(axis, tensor_name, tensor_type)
      _validate_mesh_axes(mapped, axis, tensor_name)
      for mesh_axis in (mapped,) if isinstance(mapped, str) else tuple(mapped or ()):
        if mesh_axis in claimed:
          raise ValueError(
              f"Tensor {tensor_name!r} maps both {claimed[mesh_axis]!r} and {axis!r} onto mesh axis "
              f"{mesh_axis!r}; a mesh axis can shard only one dimension of a tensor."
          )
        claimed[mesh_axis] = axis
      resolved.append(mapped)
    return P(*resolved)

  def constrain(self, x: jax.Array, tensor_name: str, axes: Sequence[str | None]) -> jax.Array:
    """Applies an activation sharding constraint to `x`.

    Activations only, by construction -- see the module docstring on why
    re-constraining parameters after init breaks the training engine.

    Dispatches on the sharding mode of the mesh in scope. Under an explicit
    mesh `with_sharding_constraint` is an assertion, not a constraint, and
    fails on anything not already sharded as asked; `reshard` is the operation
    that moves data there.
    """
    spec = self(tensor_name, axes, TensorType.ACTIVATION)
    if _ambient_mesh_is_explicit():
      return jax.sharding.reshard(x, spec)
    return jax.lax.with_sharding_constraint(x, spec)


def _resolve_axis_sizes(sizes: Sequence[int], target_product: int, kind: str, target_desc: str) -> tuple[int, ...]:
  """Fills in a single `-1` axis and checks the result tiles `target_product` exactly.

  Split out from `create_mesh` so the arithmetic is testable without needing a
  particular number of devices attached.

  Args:
    sizes: Per-axis sizes, in `MESH_AXIS_NAMES` order.
    target_product: What the sizes must multiply to.
    kind: `"ICI"` or `"DCN"`; names the network in error messages.
    target_desc: A count noun naming what `target_product` counts, such as
      `"slice count"`. Phrased so error messages read correctly at any value.

  Returns:
    The resolved sizes, in `MESH_AXIS_NAMES` order.

  Raises:
    ValueError: If more than one axis is -1, if any size is invalid, or if the
      sizes do not multiply to `target_product`.
  """
  resolved = list(sizes)

  if resolved.count(-1) > 1:
    raise ValueError(f"At most one {kind} mesh axis may be -1 (auto); got {dict(zip(MESH_AXIS_NAMES, resolved))}.")

  invalid = {name: size for name, size in zip(MESH_AXIS_NAMES, resolved) if size < 1 and size != -1}
  if invalid:
    raise ValueError(f"{kind} mesh axis sizes must be >= 1, or -1 for auto; got {invalid}.")

  if -1 in resolved:
    specified = math.prod(size for size in resolved if size != -1)
    if target_product % specified:
      raise ValueError(
          f"Cannot auto-size the -1 {kind} mesh axis: {target_desc} ({target_product}) is not divisible by "
          f"{specified}, the product of the specified axes {dict(zip(MESH_AXIS_NAMES, resolved))}."
      )
    resolved[resolved.index(-1)] = target_product // specified

  product = math.prod(resolved)
  if product != target_product:
    raise ValueError(
        f"{kind} mesh axes {dict(zip(MESH_AXIS_NAMES, resolved))} have product {product}, "
        f"which does not match the {target_desc} ({target_product})."
    )
  return tuple(resolved)


def _count_slices(devices: Sequence[jax.Device]) -> int:
  """Counts how many slices `devices` spans.

  Devices that are not part of a multi-slice job carry no `slice_index`
  attribute at all -- true of CPU devices and of single-slice TPU devices --
  so the default folds them into one group.
  """
  return len({getattr(device, "slice_index", 0) for device in devices})


def _shape_from_mapping(sizes: Mapping[str, int] | None, kind: str) -> tuple[int, ...]:
  """Turns `{axis: size}` into a shape tuple in `MESH_AXIS_NAMES` order.

  Unnamed axes default to 1, so a caller names only the axes it is actually
  splitting. Unknown names are rejected rather than ignored -- a typo would
  otherwise silently leave that axis at 1 and shard nothing.
  """
  sizes = dict(sizes or {})
  unknown = sorted(set(sizes) - set(MESH_AXIS_NAMES))
  if unknown:
    raise ValueError(f"Unknown {kind} mesh axes {unknown}; expected names from {MESH_AXIS_NAMES}.")
  return tuple(sizes.get(name, 1) for name in MESH_AXIS_NAMES)


def create_mesh(
    *,
    ici: Mapping[str, int] | None = None,
    dcn: Mapping[str, int] | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> Mesh:
  """Builds the device mesh over `MESH_AXIS_NAMES`.

  The mesh is the product of two shapes over the same axes. `ici` spreads the
  devices within one slice over the fast interconnect; `dcn` spreads the slices
  themselves over the data center network. Each may carry at most one `-1`,
  meaning "use whatever is left", and both may do so at once because they
  resolve against different totals: devices per slice, and slices.

  Axes left unnamed default to 1, so callers name only what they split::

      create_mesh(ici={"fsdp": -1})                 # every device on fsdp
      create_mesh(ici={"fsdp": -1, "tp": 2})        # tp within a slice
      create_mesh(ici={"fsdp": -1}, dcn={"dp": -1}) # fsdp in a slice, dp across

  Mappings rather than one keyword per axis: the axis list is expected to grow
  (see `MESH_AXIS_NAMES`), and a keyword pair per axis per network would scale
  to twenty-odd parameters and churn every call site each time one is added.

  The defaults -- `ici={"fsdp": -1}`, `dcn={"dp": -1}` -- put every device in a
  slice on `fsdp` and every slice on `dp`. That split is the usual starting
  point because DCN bandwidth comfortably carries a gradient all-reduce but not
  the per-layer traffic that FSDP or tensor parallelism generate. On a single
  device it degenerates to a mesh where every `PartitionSpec` resolves to
  replicated, which is what keeps the single-device code path identical to the
  distributed one instead of a special case.

  Device ordering is delegated to `jax.experimental.mesh_utils`, which lays the
  mesh out over the detected topology; this function only picks which of its
  two helpers to call. Hand-tuned orderings still beat it on some topologies --
  legacy exposes a couple as `custom_mesh` -- which m3 does not offer yet.
  https://docs.jax.dev/en/latest/jax.experimental.mesh_utils.html

  The returned mesh must be the same object the training driver uses. A
  `Sharding` resolved against a different mesh is silently replaced by full
  replication rather than raising.

  Args:
    ici: Axis sizes within a slice, e.g. `{"fsdp": -1, "tp": 2}`. Defaults to
      `{"fsdp": -1}`.
    dcn: Axis sizes across slices. Defaults to `{"dp": -1}`.
    devices: Devices to arrange; defaults to `jax.devices()`.

  Returns:
    A `Mesh` over `MESH_AXIS_NAMES`.

  Raises:
    ValueError: If `devices` is empty, if either mapping names an unknown axis,
      has more than one -1, contains an invalid size, or does not multiply to
      the device or slice count.
  """
  ici_sizes = _shape_from_mapping({"fsdp": -1} if ici is None else ici, "ICI")
  dcn_sizes = _shape_from_mapping({"dp": -1} if dcn is None else dcn, "DCN")

  devices = list(jax.devices() if devices is None else devices)
  if not devices:
    raise ValueError("Cannot build a mesh over zero devices.")

  num_slices = _count_slices(devices)
  if len(devices) % num_slices:
    raise ValueError(f"{len(devices)} devices do not divide evenly into {num_slices} slices.")

  ici_shape = _resolve_axis_sizes(ici_sizes, len(devices) // num_slices, "ICI", "per-slice device count")
  dcn_shape = _resolve_axis_sizes(dcn_sizes, num_slices, "DCN", "slice count")

  if num_slices > 1:
    device_array = mesh_utils.create_hybrid_device_mesh(ici_shape, dcn_shape, devices)
  else:
    # Not `create_hybrid_device_mesh` with an all-ones DCN shape: it requires
    # every device to expose `slice_index`, which single-slice TPU and CPU
    # devices do not, so it raises rather than degrading to the ICI-only case.
    device_array = mesh_utils.create_device_mesh(ici_shape, devices)

  return Mesh(device_array, MESH_AXIS_NAMES)
