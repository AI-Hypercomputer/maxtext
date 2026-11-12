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

  * At construction, for weight metadata (`out_sharding` is Flax's key; the
    design doc's `sharding` still works but is deprecated)::

      nnx.Linear(..., kernel_metadata={"out_sharding": s("gate", ("embed", "mlp"))})

  * Inside `__call__`, for activation constraints::

      x = s.constrain(x, "mlp_mid", ("batch", "length", "mlp"))

Parameters are sharded once by the initializer. Re-asserting a constraint on a
parameter afterwards collides with the training engine's deferred all-reduce
and Zero-1 resharding, so `constrain` is hardwired to `TensorType.ACTIVATION`
and cannot be used on weights.
"""

import abc
import collections
from collections.abc import Callable, Mapping, Sequence
import contextlib
import contextvars
import dataclasses
import enum
import math
from typing import Any, TypeVar
import warnings

from flax import nnx
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from maxtext.common.common_types import CustomRule, ShardMode

# Data, fully-sharded data, tensor and expert parallelism, in mesh order.
#
# Order is not cosmetic: it is the order handed to `mesh_utils`, which places
# axes against the physical topology. Inserting an axis in the middle changes
# the placement of everything after it.
#
# The names are legacy's rather than abbreviations because the mesh is shared
# with `maxtext.training_engine.maxtext_engine` (see `m3/README.md`), which
# reads the data axis by name: on a mesh with no axis called `"data"` it
# declines Zero-1 (one warning) and deferred gradient all-reduce (silently).
#
# Data, pipeline stage (`stage=1` placeholder in legacy mesh order), FSDP
# (`fsdp` and extra factor `fsdp_transpose`), context, tensor, tensor_sequence,
# and expert parallelism, in legacy's exact relative mesh order.
MESH_AXIS_NAMES = ("data", "stage", "fsdp", "fsdp_transpose", "context", "tensor", "tensor_sequence", "expert")
DEFAULT_SHARDING_AXES = ("data", "fsdp", "fsdp_transpose", "context", "tensor", "tensor_sequence", "expert")

HYBRID_RING_64X4 = "hybrid_ring_64x4"
HYBRID_RING_32X8 = "hybrid_ring_32x8"


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


def ambient_mesh_is_explicit() -> bool:
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
  """Rejects mesh axis names that are not in `MESH_AXIS_NAMES`."""
  if mapped is None:
    return
  names = (mapped,) if isinstance(mapped, str) else tuple(mapped)
  unknown = [name for name in names if name not in MESH_AXIS_NAMES]
  if unknown:
    raise ValueError(
        f"Sharding rule for logical axis {axis!r} of tensor {tensor_name!r} returned unknown mesh "
        f"axes {unknown}; expected names from {MESH_AXIS_NAMES}."
    )


def project_spec_to_mesh(
    spec: P,
    mesh_or_axes: Mesh | jax.sharding.AbstractMesh | Sequence[str],
    *,
    drop_explicit_unit_axes: bool = True,
) -> P:
  """Projects `spec` onto `mesh_or_axes`, dropping absent optional axes (and size-1 explicit axes when requested).

  Keeps 4-axis test meshes (e.g. `("data", "fsdp", "tensor", "expert")`) working
  with 7-axis rules. Activation specs (`drop_explicit_unit_axes=True`) also drop
  size-1 `Explicit` axes to match legacy input/loss shardings (`utils/sharding.py`),
  while parameter metadata (`drop_explicit_unit_axes=False`) stays identical across
  `auto` and `explicit` modes.
  """
  if hasattr(mesh_or_axes, "axis_names"):
    axis_names = set(mesh_or_axes.axis_names)
  else:
    axis_names = set(mesh_or_axes)

  explicit_shape: dict[str, int] | None = None
  axis_types = tuple(getattr(mesh_or_axes, "axis_types", ()))
  if (
      drop_explicit_unit_axes
      and axis_types
      and all(t == jax.sharding.AxisType.Explicit for t in axis_types)
      and hasattr(mesh_or_axes, "shape")
  ):
    explicit_shape = dict(mesh_or_axes.shape)

  projected = []
  for entry in spec:
    if entry is None:
      projected.append(None)
      continue
    names = (entry,) if isinstance(entry, str) else tuple(entry)
    kept = []
    for name in names:
      if name not in axis_names and name not in MESH_AXIS_NAMES:
        raise ValueError(f"PartitionSpec {spec!r} references unknown mesh axis {name!r}; mesh has {sorted(axis_names)}.")
      if name in axis_names and (explicit_shape is None or explicit_shape.get(name, 2) > 1):
        kept.append(name)
    if not kept:
      projected.append(None)
    elif len(kept) == 1:
      projected.append(kept[0])
    else:
      projected.append(tuple(kept))
  return P(*projected)


class Sharding(abc.ABC):
  """Maps a model's logical axis names onto mesh axes.

  A subclass encodes one parallelism strategy; `DefaultSharding` below is the
  one models start from. Strategies are meant to be reused across models,
  which is why the common logical vocabulary -- `"embed"`, `"mlp"`, `"heads"`,
  `"kv_heads"`, `"vocab"`, `"norm"` -- should mean the same thing everywhere.
  A model with a genuinely novel axis extends a strategy and delegates the
  rest::

      class Qwen3NextSharding(DefaultSharding):
        def map_axis(self, axis, tensor_name, tensor_type):
          if axis == "gdn_head":
            return "tensor"
          return super().map_axis(axis, tensor_name, tensor_type)

  Which mesh axes are live is the mesh's decision, not the strategy's. An axis
  of size 1 shards nothing, so one rule set serves pure FSDP, FSDP with tensor
  parallelism, and data parallelism across slices; the mesh shape selects the
  combination. A new strategy is only warranted when the same logical axis
  should land on a *different* mesh axis.

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
    spec = P(*resolved)
    ambient = jax.sharding.get_abstract_mesh()
    if not ambient.empty:
      ambient_mesh_is_explicit()
      return project_spec_to_mesh(spec, ambient, drop_explicit_unit_axes=tensor_type != TensorType.WEIGHT)
    return spec

  def out_sharding(self, tensor_name: str, axes: Sequence[str | None]) -> P | None:
    """Returns the explicit output `PartitionSpec` under an `Explicit` mesh, else `None`."""
    if jax.sharding.get_abstract_mesh().empty or not ambient_mesh_is_explicit():
      return None
    return self(tensor_name, axes, TensorType.ACTIVATION)

  def constrain(self, x: jax.Array, tensor_name: str, axes: Sequence[str | None]) -> jax.Array:
    """Applies an activation sharding constraint to `x`.

    Activations only, by construction -- see the module docstring on why
    re-constraining parameters after init breaks the training engine.

    Dispatches on the sharding mode of the mesh in scope. Under an explicit
    mesh `with_sharding_constraint` is an assertion, not a constraint, and
    fails on anything not already sharded as asked; `reshard` is the operation
    that moves data there.

    With no mesh in scope there is nothing to constrain against, so the call
    returns `x` unchanged -- a model built under a mesh can still be called
    eagerly outside one. Construction is where a missing mesh is an error.
    """
    if jax.sharding.get_abstract_mesh().empty:
      return x
    spec = self(tensor_name, axes, TensorType.ACTIVATION)
    if ambient_mesh_is_explicit():
      return jax.sharding.reshard(x, spec)
    if isinstance(getattr(x, "sharding", None), jax.sharding.SingleDeviceSharding):
      try:
        mesh = jax.sharding.get_mesh()
      except ValueError:
        mesh = None
      if isinstance(mesh, Mesh) and not mesh.empty:
        return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))
    return jax.lax.with_sharding_constraint(x, spec)

  def project(
      self,
      layer: Callable[..., jax.Array],
      x: jax.Array,
      tensor_name: str,
      axes: Sequence[str | None],
  ) -> jax.Array:
    """Calls `layer(x, out_sharding=...)` and constrains the output to `axes`."""
    out = layer(x, out_sharding=self.out_sharding(tensor_name, axes))
    return self.constrain(out, tensor_name, axes)


def param_metadata(sharding: Sharding, tensor_name: str, axes: Sequence[str | None]) -> dict[str, Any]:
  """The `*_metadata=` for one parameter: its resolved spec, placed at construction.

  `eager_sharding=True` makes Flax apply `out_sharding` when the variable is
  created, under the mesh in scope, for this variable regardless of the
  process-global switch (`flax_always_shard_variable`). That switch is turned
  off at import by legacy MaxText (`layers/nnx_wrappers.py`), and with it off
  the metadata is stored but never applied: every parameter comes out
  replicated, with nothing to say so. Declaring it per parameter keeps the
  model's placement independent of what else the process imported.
  The flag is for construction only. Flax rebuilds variables from their
  metadata (`nnx.Optimizer` creates each moment as
  `OptVariable(value, **param.get_metadata())`), and a flag left on would
  re-run placement there, which fails with no mesh in scope.
  `m3.models.create_model` turns it off once the model is built.
  """
  meta: dict[str, Any] = {
      "out_sharding": sharding(tensor_name, axes),
      "eager_sharding": not _ALLOW_UNSHARDED_CONSTRUCTION.get(),
  }
  if (
      isinstance(sharding, FSDPasDPForAttn)
      and not sharding.shard_router_on_fsdp
      and ("embed_router" in axes or ("embed" in axes and tensor_name in _ROUTER_TENSOR_NAMES))
  ):
    meta["preserve_unsharded"] = True
  return meta


def end_eager_placement(model: nnx.Module) -> nnx.Module:
  """Turns `eager_sharding` off on every variable placed at construction; returns `model`."""
  for _, node in nnx.iter_graph(model):
    if isinstance(node, nnx.Variable):
      metadata = node.get_metadata()
      if metadata.get("eager_sharding") and metadata.get("out_sharding") is not None:
        node.set_metadata(eager_sharding=False)
  return model


_ALLOW_UNSHARDED_CONSTRUCTION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_allow_unsharded_construction", default=False
)


@contextlib.contextmanager
def allow_unsharded_construction():
  """Explicit opt-in for building `param_metadata`-annotated modules with no mesh in scope."""
  warnings.warn(
      "Constructing m3 parameters under allow_unsharded_construction(): parameters are left unsharded on a single "
      "device and must not be passed to multi-device training.",
      UserWarning,
      stacklevel=2,
  )
  token = _ALLOW_UNSHARDED_CONSTRUCTION.set(True)
  try:
    yield
  finally:
    _ALLOW_UNSHARDED_CONSTRUCTION.reset(token)


_EnumT = TypeVar("_EnumT", bound=enum.Enum)


def parse_enum(value: Any, default: _EnumT) -> _EnumT:
  """Normalizes an Enum, string, or `None` config value to `type(default)`."""
  if value is None:
    return default
  enum_type = type(default)
  if isinstance(value, enum_type):
    return value
  if isinstance(value, str):
    try:
      return enum_type(value)
    except ValueError:
      return enum_type(value.lower())
  return enum_type(value)


def get_enum(config: Any, attr: str, default: _EnumT) -> _EnumT:
  """Reads Enum attribute `attr` from `config`, normalizing string or `None` values."""
  return parse_enum(getattr(config, attr, default), default)


def _in_jax_trace() -> bool:
  """Returns whether execution is under a JAX trace where `jax.set_mesh` is disallowed."""
  try:
    jax.sharding.get_mesh()
    return False
  except ValueError:
    return True


def _normalized_abstract_mesh(
    mesh: Mesh | jax.sharding.AbstractMesh, shard_mode: ShardMode | str = ShardMode.AUTO
) -> jax.sharding.AbstractMesh:
  abstract = mesh.abstract_mesh
  # `AbstractMesh(sizes, names)` defaults to `Explicit` axes in JAX even when
  # `shard_mode=auto`, which stamps `Explicit` shardings onto parameter tracers
  # and breaks post-init forward traces (such as Qwix quantization).
  if isinstance(mesh, jax.sharding.AbstractMesh) and parse_enum(shard_mode, ShardMode.AUTO) == ShardMode.AUTO:
    abstract = abstract.update(axis_types=(jax.sharding.AxisType.Auto,) * len(abstract.axis_names))
  return abstract


@contextlib.contextmanager
def enter_mesh_scope(mesh: Mesh | jax.sharding.AbstractMesh | None, *, shard_mode: ShardMode | str = ShardMode.AUTO):
  """Enters `mesh` (`set_mesh` eagerly or `use_abstract_mesh` under trace) when no mesh is in scope."""
  if mesh is None:
    yield
    return
  abstract = _normalized_abstract_mesh(mesh, shard_mode=shard_mode)
  ambient = jax.sharding.get_abstract_mesh()
  if ambient.empty:
    with contextlib.ExitStack() as stack:
      if isinstance(mesh, Mesh) and not _in_jax_trace():
        stack.enter_context(jax.set_mesh(mesh))
      else:
        stack.enter_context(jax.sharding.use_abstract_mesh(abstract))
      yield
    return
  if ambient not in (abstract, mesh.abstract_mesh):
    raise ValueError(f"create_model was passed mesh {mesh} while a different mesh is in scope: {ambient}.")
  yield


_REQUIRED_MESH_AXES = ("data", "fsdp", "tensor", "expert")
_OPTIONAL_MESH_AXES = frozenset(MESH_AXIS_NAMES) - frozenset(_REQUIRED_MESH_AXES)


def check_mesh(mesh: Mesh | jax.sharding.AbstractMesh, axis_names: Sequence[str] = DEFAULT_SHARDING_AXES) -> None:
  """Refuses a mesh that a strategy resolving to `axis_names` cannot use as asked.

  Specs name only `axis_names` (`_validate_mesh_axes`). On a mesh missing one
  of the required axes, placement fails with an opaque "resource axis not found"
  (`stage`, `fsdp_transpose`, `context`, and `tensor_sequence` are optional when
  absent or size 1 so 4-axis meshes continue to work). On a mesh that splits any
  other axis (including `stage > 1` until a pipeline transform exists),
  parameters are replicated along it and activations computed redundantly,
  without an error. Both reach m3 models through drivers that build their own
  mesh (legacy's `custom_mesh_and_rule`, ...), which `mesh_from_config` never
  sees.

  Raises:
    ValueError: Naming the missing axes and the split ones.
  """
  required = [axis for axis in axis_names if axis not in _OPTIONAL_MESH_AXES]
  missing = [axis for axis in required if axis not in mesh.axis_names]
  split = {axis: size for axis, size in dict(mesh.shape).items() if axis not in axis_names and size > 1}
  if missing or split:
    problems = []
    if missing:
      problems.append(f"lacks mesh axes {missing}")
    if split:
      problems.append(f"splits mesh axes {split} that it does not use")
    raise ValueError(
        f"m3 sharding resolves to mesh axes {tuple(axis_names)}, and the mesh in scope "
        f"{' and '.join(problems)} (see m3/README.md). Mesh: {dict(mesh.shape)}."
    )


# Frozen dataclass: models keep their strategy as a static attribute, and the
# generated `__eq__` / `__hash__` let two models built from one config share a
# graphdef (and so one `nnx.jit` trace), which identity comparison would not.
@dataclasses.dataclass(frozen=True)
class DefaultSharding(Sharding):
  """Default m3 parallelism rules across FSDP, context, tensor, tensor_sequence, and expert axes.

  Weights shard `embed` across `("fsdp", "fsdp_transpose", "context", "expert")`
  (`fsdp_transpose` acts as an additional FSDP factor rather than sharding MLP
  width, and `context` doubles as FSDP weight sharding so dense parameters are
  never replicated across `fsdp_transpose`, `context`, or `expert`);
  routed-expert weights shard `experts` over `"expert"`, `expert_embed` over
  `("fsdp", "fsdp_transpose", "context")`, and `expert_mlp` over
  `("tensor", "tensor_sequence")`; `heads`, `kv_heads`, `mlp`, and `vocab` shard
  over `("tensor", "tensor_sequence")`; `norm` and `head_dim` stay replicated.

  Activations shard `batch` over `("data", "fsdp", "fsdp_transpose", "expert")`,
  `length` over `context` (all-gather K/V in natural token order), and
  `norm_length` (the residual stream and RMSNorm inputs) over
  `("context", "tensor_sequence")`. When `tensor_sequence` is used in place of
  `tensor` (`tensor == 1` and `tensor_sequence > 1`, Megatron sequence
  parallelism), TP `all-reduce`s on residual projections are replaced with
  `reduce-scatter` + `all-gather`, cutting residual/norm activation memory by
  `1 / tensor_sequence` (mixing `tensor > 1` with `tensor_sequence > 1` retains
  the `tensor` `all-reduce`).

  Two places this follows the design doc rather than legacy's
  `logical_axis_rules`: legacy shards the residual-stream `embed` activation
  and the norm scale over `tensor`, whereas this leaves both replicated over
  `tensor` (use `tensor_sequence` instead of `tensor` to shard residual/norm
  activations along sequence).
  """

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case "batch", TensorType.ACTIVATION:
        return ("data", "fsdp", "fsdp_transpose", "expert")
      case "embed", TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose", "context", "expert")
      case "embed", TensorType.ACTIVATION:
        return None
      case "experts", _:
        return "expert"
      case "expert_embed", TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose", "context")
      case "expert_embed", TensorType.ACTIVATION:
        return None
      case ("heads" | "kv_heads" | "mlp" | "expert_mlp" | "vocab"), _:
        return ("tensor", "tensor_sequence")
      case ("length" | "norm_length" | "kv_length"), TensorType.WEIGHT:
        return None
      case "length", TensorType.ACTIVATION:
        return "context"
      case "norm_length", TensorType.ACTIVATION:
        return ("context", "tensor_sequence")
      case "kv_length", TensorType.ACTIVATION:
        return None
      case ("q_lora" | "kv_lora" | "embed_attn"), TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose", "expert")
      case ("q_lora" | "kv_lora" | "embed_attn"), TensorType.ACTIVATION:
        return None
      case "embed_router", TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose")
      case "embed_router", TensorType.ACTIVATION:
        return None
      case ("head_dim" | "norm" | "q_lora_up_proj" | "kv_lora_up_proj"), _:
        return None
      case _:
        raise KeyError(f"No sharding rule for logical axis {axis!r} ({tensor_type.value}) on tensor {tensor_name!r}.")


# Compound MoE physical axis tuples.
# `FSDPxEP` shards routed-expert `experts` across `("fsdp", "expert")` (or 2D
# `("fsdp", "fsdp_transpose", "expert")` via `FSDPxEP_2D` / `fsdp_x_ep_axes`)
# while keeping `expert_embed` unsharded (`None`) so non-elementwise optimizers
# such as Muon retain complete `(expert_embed, expert_mlp)` matrices per shard.
FSDPxEP: tuple[str, str] = ("fsdp", "expert")
FSDPxEP_2D: tuple[str, str, str] = ("fsdp", "fsdp_transpose", "expert")
TPxEP: tuple[str, str] = ("expert", "tensor")

_MLA_ATTN_TENSOR_NAMES = frozenset(
    {
        "q_lora",
        "kv_lora",
        "wq_a",
        "wkv_a",
        "wk",
        "q_a_proj",
        "kv_a_proj",
        "kv_a_proj_with_mqa",
        "attn_q_lora",
        "attn_kv_lora",
    }
)
_ROUTER_TENSOR_NAMES = frozenset({"router", "router_weight", "gate_router", "moe_router"})


def fsdp_x_ep_axes(*, two_d_fsdp: bool | None = None) -> tuple[str, ...]:
  """Returns `("fsdp", "expert")` or `("fsdp", "fsdp_transpose", "expert")` for compound FSDP x EP."""
  if two_d_fsdp is True:
    return FSDPxEP_2D
  if two_d_fsdp is False:
    return FSDPxEP
  ambient = jax.sharding.get_abstract_mesh()
  if not ambient.empty and int(ambient.shape.get("fsdp_transpose", 1)) > 1:
    return FSDPxEP_2D
  return FSDPxEP


def tp_x_ep_axes() -> tuple[str, str]:  # public-api
  """Returns `("expert", "tensor")` for compound TP x EP expert sharding."""
  return TPxEP


def compound_moe_spec(  # public-api
    sharding: Sharding | None,
    tensor_name: str,
    axes: Sequence[str | None],
    *,
    mode: str = "fsdp_x_ep",
    two_d_fsdp: bool | None = None,
    muon_friendly: bool = True,
    tensor_type: TensorType = TensorType.WEIGHT,
) -> P:
  """Resolves `axes` for a routed-expert tensor under `FSDPxEP` or `TPxEP`.

  Under `mode="fsdp_x_ep"`, `experts` shards across `("fsdp", "expert")` (or
  `("fsdp", "fsdp_transpose", "expert")` when `two_d_fsdp=True`) while
  `expert_embed` stays `None` (`muon_friendly=True`), keeping each expert's
  non-contracting `(expert_embed, expert_mlp)` matrix whole on its device shard.
  Under `mode="tp_x_ep"`, `experts` shards across `("expert", "tensor")` and
  `expert_mlp` shards over `"tensor_sequence"` so `"tensor"` is not claimed twice.
  When a preset `sharding` instance is supplied, its non-overridden axis rules
  and `two_d_fsdp` setting are preserved.
  """
  mode_key = mode.strip().lower().replace("-", "_")
  if mode_key in ("fsdp_x_ep", "fsdp_ep", "fsdpxep", "expert_on_fsdp_ep", "shard_exp_on_fsdp"):
    if isinstance(sharding, TPasEP):
      rep_kw: dict[str, Any] = {"compound_fsdp_ep": True, "compound_muon_friendly": muon_friendly}
      if two_d_fsdp is not None:
        rep_kw["two_d_fsdp"] = two_d_fsdp
      strategy: Sharding = dataclasses.replace(sharding, **rep_kw)
    elif isinstance(sharding, (EPasDP, EPasCP, CPasEP, FSDPasDPForAttn)):
      rep_kw = {"compound_fsdp_ep": True, "muon_friendly": muon_friendly}
      if two_d_fsdp is not None:
        rep_kw["two_d_fsdp"] = two_d_fsdp
      strategy = dataclasses.replace(sharding, **rep_kw)
    else:
      strategy = EPasDP(
          compound_fsdp_ep=True,
          muon_friendly=muon_friendly,
          two_d_fsdp=two_d_fsdp,
      )
    return strategy(tensor_name, axes, tensor_type)
  if mode_key in ("tp_x_ep", "tp_ep", "tpxep", "tp_as_ep", "tpasep"):
    if isinstance(sharding, TPasEP):
      rep_kw = {"muon_friendly": muon_friendly}
      if two_d_fsdp is not None:
        rep_kw["two_d_fsdp"] = two_d_fsdp
      strategy = dataclasses.replace(sharding, **rep_kw)
    else:
      eff_two_d = two_d_fsdp
      if eff_two_d is None and isinstance(sharding, DefaultSharding) and hasattr(sharding, "two_d_fsdp"):
        eff_two_d = sharding.two_d_fsdp
      strategy = TPasEP(
          muon_friendly=muon_friendly,
          two_d_fsdp=eff_two_d,
      )
    return strategy(tensor_name, axes, tensor_type)
  raise ValueError(f"Unknown compound MoE mode {mode!r}; expected 'fsdp_x_ep' or 'tp_x_ep'.")


def compound_moe_param_metadata(  # public-api
    sharding: Sharding | None,
    tensor_name: str,
    axes: Sequence[str | None],
    *,
    mode: str = "fsdp_x_ep",
    two_d_fsdp: bool | None = None,
    muon_friendly: bool = True,
) -> dict[str, Any]:
  """Returns `*_metadata` with `eager_sharding=True` for compound `FSDPxEP` or `TPxEP` expert weights."""
  return {
      "out_sharding": compound_moe_spec(
          sharding,
          tensor_name,
          axes,
          mode=mode,
          two_d_fsdp=two_d_fsdp,
          muon_friendly=muon_friendly,
          tensor_type=TensorType.WEIGHT,
      ),
      "eager_sharding": not _ALLOW_UNSHARDED_CONSTRUCTION.get(),
  }


@dataclasses.dataclass(frozen=True)
class EPasDP(DefaultSharding):
  """Expert axis acts as Data Parallelism outside core MoE (`ep-as-dp`).

  When `compound_fsdp_ep=True` (`expert-on-fsdp-ep` / `shard-exp-on-fsdp`),
  routed-expert `experts` shards across `FSDPxEP` (`("fsdp", "expert")` or
  `("fsdp", "fsdp_transpose", "expert")`) and `expert_embed` stays `None` so
  Muon-style non-elementwise optimizers avoid per-step FSDP all-gathers.
  """

  compound_fsdp_ep: bool = False
  muon_friendly: bool = True
  two_d_fsdp: bool | None = None

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    if self.compound_fsdp_ep:
      if axis == "experts":
        return fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp) if tensor_type == TensorType.WEIGHT else "expert"
      if axis == "expert_embed":
        if self.muon_friendly or tensor_type != TensorType.WEIGHT:
          return None
        claimed_fsdp = set(fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp))
        rem = tuple(ax for ax in ("fsdp", "fsdp_transpose", "context") if ax not in claimed_fsdp)
        return rem[0] if len(rem) == 1 else (rem if rem else None)
    return super().map_axis(axis, tensor_name, tensor_type)


@dataclasses.dataclass(frozen=True)
class EPasCP(DefaultSharding):
  """Expert axis acts as Context Parallelism outside core MoE (`ep-as-cp`).

  Sequence activations (`length`, `norm_length`) shard over `("context", "expert")`
  while `batch` shards over `("data", "fsdp", "fsdp_transpose")`, avoiding axis
  collisions on `(batch, length, ...)` tensors while repurposing `expert` for
  context parallelism in attention and dense blocks.
  """

  compound_fsdp_ep: bool = False
  muon_friendly: bool = True
  two_d_fsdp: bool | None = None

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case "batch", TensorType.ACTIVATION:
        return ("data", "fsdp", "fsdp_transpose")
      case "length", TensorType.ACTIVATION:
        return ("context", "expert")
      case "norm_length", TensorType.ACTIVATION:
        return ("context", "expert", "tensor_sequence")
      case "experts", TensorType.WEIGHT if self.compound_fsdp_ep:
        return fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp)
      case "expert_embed", _ if self.compound_fsdp_ep:
        if self.muon_friendly or tensor_type != TensorType.WEIGHT:
          return None
        claimed_fsdp = set(fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp))
        rem = tuple(ax for ax in ("fsdp", "fsdp_transpose", "context") if ax not in claimed_fsdp)
        return rem[0] if len(rem) == 1 else (rem if rem else None)
    return super().map_axis(axis, tensor_name, tensor_type)


@dataclasses.dataclass(frozen=True)
class CPasEP(DefaultSharding):
  """Context axis acts as secondary Expert Parallelism in core MoE (`cp-as-ep`).

  Routed-expert `experts` shards over `("context", "expert")` (or
  `("fsdp", "context", "expert")` / `("fsdp", "fsdp_transpose", "context", "expert")`
  when `compound_fsdp_ep=True`) while `expert_embed` weights shard over
  `("fsdp", "fsdp_transpose")` (or `None` when `compound_fsdp_ep=True`) so
  `"context"` is not claimed twice, and dense `embed` / MLA `q_lora` / `kv_lora`
  weights shard over `("fsdp", "fsdp_transpose", "context", "expert")`.
  """

  compound_fsdp_ep: bool = False
  muon_friendly: bool = True
  two_d_fsdp: bool | None = None

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case "experts", TensorType.WEIGHT if self.compound_fsdp_ep:
        base_fsdp = fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp)[:-1]
        return (*base_fsdp, "context", "expert")
      case "experts", _:
        return ("context", "expert")
      case "expert_embed", _ if self.compound_fsdp_ep:
        if self.muon_friendly or tensor_type != TensorType.WEIGHT:
          return None
        base_fsdp_set = set(fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp)[:-1])
        rem = tuple(ax for ax in ("fsdp", "fsdp_transpose") if ax not in base_fsdp_set)
        return rem[0] if len(rem) == 1 else (rem if rem else None)
      case "expert_embed", TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose")
      case ("embed" | "q_lora" | "kv_lora" | "embed_attn"), TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose", "context", "expert")
    return super().map_axis(axis, tensor_name, tensor_type)


@dataclasses.dataclass(frozen=True)
class TPasEP(DefaultSharding):
  """Tensor axis acts as secondary Expert Parallelism in core MoE (`TPxEP`).

  Routed-expert `experts` shards over `TPxEP = ("expert", "tensor")` while
  `expert_mlp` shards over `"tensor_sequence"` so `"tensor"` is not claimed twice
  on 3D/4D expert weights while preserving sequence-tensor parallelism. When
  `compound_fsdp_ep=True` or `muon_friendly=True`, `expert_embed` is also kept
  unsharded (`None`).
  """

  compound_fsdp_ep: bool = False
  muon_friendly: bool = False
  compound_muon_friendly: bool = True
  two_d_fsdp: bool | None = None

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case "experts", TensorType.WEIGHT if self.compound_fsdp_ep:
        return (*fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp), "tensor")
      case "experts", _:
        return TPxEP
      case "expert_mlp", _:
        return "tensor_sequence"
      case "expert_embed", _ if self.compound_fsdp_ep:
        if self.compound_muon_friendly or tensor_type != TensorType.WEIGHT:
          return None
        claimed_fsdp = set(fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp))
        rem = tuple(ax for ax in ("fsdp", "fsdp_transpose", "context") if ax not in claimed_fsdp)
        return rem[0] if len(rem) == 1 else (rem if rem else None)
      case "expert_embed", _ if self.muon_friendly:
        return None
    return super().map_axis(axis, tensor_name, tensor_type)


@dataclasses.dataclass(frozen=True)
class FSDPasDPForAttn(DefaultSharding):
  """Scale-tuned MLA preset where FSDP acts as DP for attention and EP acts as FSDP for attention (`fsdp-as-dp-for-attn-rf`).

  Small MLA compressed attention projections (`q_lora`, `kv_lora`, `embed_attn`)
  shard over `"expert"` and replicate across `("fsdp", "fsdp_transpose")` to
  avoid latency-bound tiny FSDP all-gathers at large scale (`>512` chips), while:
    * Attention and model activations (`batch`) still shard across
      `("data", "fsdp", "fsdp_transpose", "expert")` (never replicated on FSDP).
    * Routed-expert weights keep scaling across `("fsdp", "expert")`:
      `P(("fsdp", "expert"), None, ("tensor", "tensor_sequence"))` when
      `compound_fsdp_ep=True` (Muon-friendly) or
      `P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence"))`
      when `compound_fsdp_ep=False`.
    * Router weights (`embed_router`) shard over `("fsdp", "fsdp_transpose")` when
      `shard_router_on_fsdp=True` (the `-rf` default) and replicate when `False`.
  """

  compound_fsdp_ep: bool = False
  muon_friendly: bool = True
  shard_router_on_fsdp: bool = True
  two_d_fsdp: bool | None = None

  def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes:
    match axis, tensor_type:
      case ("q_lora" | "kv_lora" | "embed_attn"), TensorType.WEIGHT:
        return "expert"
      case ("q_lora" | "kv_lora" | "embed_attn"), TensorType.ACTIVATION:
        return None
      case "embed", TensorType.WEIGHT if tensor_name in _MLA_ATTN_TENSOR_NAMES:
        return "expert"
      case "embed", TensorType.WEIGHT if tensor_name in _ROUTER_TENSOR_NAMES:
        return ("fsdp", "fsdp_transpose") if self.shard_router_on_fsdp else None
      case "embed_router", TensorType.WEIGHT:
        return ("fsdp", "fsdp_transpose") if self.shard_router_on_fsdp else None
      case "embed_router", TensorType.ACTIVATION:
        return None
      case "experts", TensorType.WEIGHT if self.compound_fsdp_ep:
        return fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp)
      case "expert_embed", _ if self.compound_fsdp_ep:
        if self.muon_friendly or tensor_type != TensorType.WEIGHT:
          return None
        claimed_fsdp = set(fsdp_x_ep_axes(two_d_fsdp=self.two_d_fsdp))
        rem = tuple(ax for ax in ("fsdp", "fsdp_transpose", "context") if ax not in claimed_fsdp)
        return rem[0] if len(rem) == 1 else (rem if rem else None)
    return super().map_axis(axis, tensor_name, tensor_type)


SHARDING_PRESETS: dict[str, type[DefaultSharding]] = {
    "default": DefaultSharding,
    "ep_as_dp": EPasDP,
    "ep_as_cp": EPasCP,
    "cp_as_ep": CPasEP,
    "tp_as_ep": TPasEP,
    "fsdp_as_dp_for_attn": FSDPasDPForAttn,
    "fsdp_x_ep": EPasDP,
    "tp_x_ep": TPasEP,
}


def sharding_preset_from_name(
    name: str | DefaultSharding | None = None,
    **kwargs: Any,
) -> DefaultSharding:
  """Resolves a scale-tuned `DefaultSharding` preset by name or alias.

  Supported preset names (case-insensitive, `-` or `_` separated):
    * `"default"` / `""` / `None`: `DefaultSharding`
    * `"ep_as_dp"` / `"ep-as-dp"`: `EPasDP`
    * `"ep_as_cp"` / `"ep-as-cp"`: `EPasCP`
    * `"cp_as_ep"` / `"cp-as-ep"`: `CPasEP`
    * `"tp_as_ep"` / `"tp-as-ep"` / `"tp_x_ep"`: `TPasEP`
    * `"fsdp_as_dp_for_attn"` / `"fsdp-as-dp-for-attn"` / `"fsdp-as-dp-for-attn-rf"`: `FSDPasDPForAttn`
    * `"fsdp_x_ep"` / `"expert-on-fsdp-ep"` / `"shard-exp-on-fsdp"`: `EPasDP(compound_fsdp_ep=True)`
  """
  if isinstance(name, DefaultSharding):
    if not kwargs:
      return name
    try:
      return dataclasses.replace(name, **kwargs)
    except TypeError as exc:
      valid_fields = sorted(f.name for f in dataclasses.fields(name))
      hint = (
          "choose a preset subclass such as 'ep_as_dp' for compound MoE options."
          if not valid_fields
          else f"valid fields are {valid_fields}."
      )
      raise ValueError(
          f"Sharding preset instance {type(name).__name__} does not accept options {sorted(kwargs)}; {hint}"
      ) from exc
  raw = getattr(name, "value", name)
  if raw is None:
    raw = "default"
  if not isinstance(raw, str):
    raise TypeError(f"Expected str or None for sharding preset name, got {type(name).__name__}.")
  key = raw.strip().lower().replace("-", "_")
  if not key or key in ("default", "default_sharding"):
    if kwargs:
      raise ValueError(
          f"Sharding preset {name!r} (DefaultSharding) does not accept options {sorted(kwargs)}; "
          "choose a preset such as 'ep_as_dp' or omit sharding_preset='default'."
      )
    return DefaultSharding()
  if key in ("ep_as_dp", "epasdp"):
    return EPasDP(**kwargs)
  if key in ("fsdp_x_ep", "fsdpxep", "expert_on_fsdp_ep", "shard_exp_on_fsdp"):
    kwargs.setdefault("compound_fsdp_ep", True)
    return EPasDP(**kwargs)
  if key in ("ep_as_cp", "epascp"):
    return EPasCP(**kwargs)
  if key in ("cp_as_ep", "cpasep"):
    return CPasEP(**kwargs)
  if key in ("tp_as_ep", "tpasep", "tp_x_ep", "tpxep"):
    return TPasEP(**kwargs)
  if key in ("fsdp_as_dp_for_attn", "fsdpasdpforattn", "fsdp_as_dp_for_attn_rf"):
    return FSDPasDPForAttn(**kwargs)
  if key in ("fsdp_as_dp_for_attn_no_rf", "fsdp_as_dp_for_attn_replicated_router"):
    kwargs.setdefault("shard_router_on_fsdp", False)
    return FSDPasDPForAttn(**kwargs)
  raise ValueError(f"Unknown m3 sharding preset {name!r}; expected one of {sorted(SHARDING_PRESETS)}.")


def _is_plain_default_preset(name: Any) -> bool:
  """Returns True when `name` selects plain `DefaultSharding` rather than a preset subclass."""
  if isinstance(name, DefaultSharding):
    return not dataclasses.fields(name)
  raw = getattr(name, "value", name)
  if raw is None:
    return True
  if isinstance(raw, str):
    return raw.strip().lower().replace("-", "_") in ("", "default", "default_sharding")
  return False


def sharding_from_config(config: Any) -> DefaultSharding:
  """Builds `DefaultSharding` (or the configured scale-tuned preset subclass) from `config`."""
  preset_name = config.sharding_preset if hasattr(config, "sharding_preset") else None
  compound_fsdp_ep = bool(config.compound_fsdp_ep) if hasattr(config, "compound_fsdp_ep") else False
  ici_fsdp_t = (
      int(config.ici_fsdp_transpose_parallelism)
      if hasattr(config, "ici_fsdp_transpose_parallelism") and config.ici_fsdp_transpose_parallelism is not None
      else 1
  )
  dcn_fsdp_t = (
      int(config.dcn_fsdp_transpose_parallelism)
      if hasattr(config, "dcn_fsdp_transpose_parallelism") and config.dcn_fsdp_transpose_parallelism is not None
      else 1
  )
  two_d_fsdp = ici_fsdp_t > 1 or dcn_fsdp_t > 1
  if preset_name is not None and preset_name != "":
    kwargs: dict[str, Any] = {}
    if compound_fsdp_ep:
      kwargs["compound_fsdp_ep"] = True
    if two_d_fsdp and not _is_plain_default_preset(preset_name):
      has_explicit_two_d = (
          isinstance(preset_name, DefaultSharding)
          and hasattr(preset_name, "two_d_fsdp")
          and preset_name.two_d_fsdp is not None
      )
      if not has_explicit_two_d:
        kwargs["two_d_fsdp"] = True
    return sharding_preset_from_name(preset_name, **kwargs)
  if compound_fsdp_ep:
    return EPasDP(compound_fsdp_ep=True, two_d_fsdp=True if two_d_fsdp else None)
  return DefaultSharding()


def _resolve_axis_sizes(
    sizes: Sequence[int],
    target_product: int,
    kind: str,
    target_desc: str,
    axis_names: Sequence[str] = MESH_AXIS_NAMES,
) -> tuple[int, ...]:
  """Fills in a single `-1` axis and checks the result tiles `target_product` exactly.

  Split out from `create_mesh` so the arithmetic is testable without needing a
  particular number of devices attached.

  Args:
    sizes: Per-axis sizes, in `axis_names` order.
    target_product: What the sizes must multiply to.
    kind: `"ICI"` or `"DCN"`; names the network in error messages.
    target_desc: A count noun naming what `target_product` counts, such as
      `"slice count"`. Phrased so error messages read correctly at any value.
    axis_names: The mesh axes `sizes` refers to; names the axes in error messages.

  Returns:
    The resolved sizes, in `axis_names` order.

  Raises:
    ValueError: If more than one axis is -1, if any size is invalid, or if the
      sizes do not multiply to `target_product`.
  """
  resolved = list(sizes)

  if resolved.count(-1) > 1:
    raise ValueError(f"At most one {kind} mesh axis may be -1 (auto); got {dict(zip(axis_names, resolved))}.")

  invalid = {name: size for name, size in zip(axis_names, resolved) if size < 1 and size != -1}
  if invalid:
    raise ValueError(f"{kind} mesh axis sizes must be >= 1, or -1 for auto; got {invalid}.")

  if -1 in resolved:
    specified = math.prod(size for size in resolved if size != -1)
    if target_product % specified:
      raise ValueError(
          f"Cannot auto-size the -1 {kind} mesh axis: {target_desc} ({target_product}) is not divisible by "
          f"{specified}, the product of the specified axes {dict(zip(axis_names, resolved))}."
      )
    resolved[resolved.index(-1)] = target_product // specified

  product = math.prod(resolved)
  if product != target_product:
    raise ValueError(
        f"{kind} mesh axes {dict(zip(axis_names, resolved))} have product {product}, "
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


def _shape_from_mapping(
    sizes: Mapping[str, int] | None, kind: str, axis_names: Sequence[str] = MESH_AXIS_NAMES
) -> tuple[int, ...]:
  """Turns `{axis: size}` into a shape tuple in `axis_names` order.

  Unnamed axes default to 1, so a caller names only the axes it is actually
  splitting. Unknown names are rejected rather than ignored -- a typo would
  otherwise silently leave that axis at 1 and shard nothing.
  """
  sizes = dict(sizes or {})
  unknown = sorted(set(sizes) - set(axis_names))
  if unknown:
    raise ValueError(f"Unknown {kind} mesh axes {unknown}; expected names from {tuple(axis_names)}.")
  return tuple(sizes.get(name, 1) for name in axis_names)


def _default_sizes(axis: str, kind: str, axis_names: Sequence[str]) -> dict[str, int]:
  """`{axis: -1}` when `axis` is on the mesh; otherwise the caller has to say how to lay the devices out."""
  if axis not in axis_names:
    raise ValueError(
        f"The default {kind} layout puts every device on {axis!r}, which is not one of the mesh axes "
        f"{tuple(axis_names)}; pass `{kind.lower()}=` explicitly."
    )
  return {axis: -1}


def _reshape_mesh_to_rings(slice_mesh_16x16: Any, strategy: str) -> Any:
  """Reshapes a `(16, 16)` slice device array into `(64, 4)` or `(32, 8)` 2D torus rings."""
  if strategy == HYBRID_RING_64X4:
    tiles = slice_mesh_16x16.reshape(8, 2, 8, 2).transpose(0, 2, 1, 3).copy()
    tiles[:, :, 1, :] = tiles[:, :, 1, ::-1]
    return tiles.reshape(64, 4)
  if strategy == HYBRID_RING_32X8:
    tiles = slice_mesh_16x16.reshape(8, 2, 4, 4).transpose(0, 2, 1, 3).copy()
    tiles[:, :, 1, :] = tiles[:, :, 1, ::-1]
    return tiles.reshape(32, 8)
  raise ValueError(f"Unknown custom_mesh strategy {strategy!r}; expected {HYBRID_RING_64X4!r} or {HYBRID_RING_32X8!r}.")


def _is_valid_custom_mesh(ici_shape: Sequence[int], custom_mesh: str) -> bool:
  """Validates `custom_mesh` against the ordered non-unit dimensions of `ici_shape`."""
  if not custom_mesh:
    return False
  expected_non_unit = {
      HYBRID_RING_64X4: [64, 4],
      HYBRID_RING_32X8: [32, 8],
  }
  if custom_mesh not in expected_non_unit:
    raise ValueError(f"Invalid custom_mesh={custom_mesh!r}; supported strategies are {sorted(expected_non_unit)}.")
  non_unit = [dim for dim in ici_shape if dim != 1]
  if non_unit != expected_non_unit[custom_mesh]:
    raise ValueError(
        f"Invalid custom_mesh={custom_mesh!r} for ICI mesh shape {tuple(ici_shape)}; "
        f"expected ordered non-unit axes {tuple(expected_non_unit[custom_mesh])}, got {tuple(non_unit)}."
    )
  return True


def _create_custom_device_mesh(
    ici_shape: Sequence[int],
    dcn_shape: Sequence[int],
    devices: Sequence[jax.Device],
    custom_mesh: str,
) -> Any:
  """Builds a single- or multi-slice custom ring device mesh (`64x4` or `32x8` per 256-chip slice)."""
  if len(devices) % 256 != 0:
    raise ValueError(f"custom_mesh={custom_mesh!r} requires a multiple of 256 devices, got {len(devices)}.")
  if math.prod(dcn_shape) > 1 and any(not hasattr(d, "slice_index") for d in devices):
    raise ValueError(f"Device {devices[0]!r} is missing `slice_index` required for multi-slice custom_mesh.")
  granule_dict: dict[int, list[jax.Device]] = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, "slice_index", 0)].append(dev)
  granules = [granule_dict[k] for k in sorted(granule_dict)]
  if math.prod(dcn_shape) != len(granules):
    raise ValueError(f"Number of slices ({len(granules)}) must equal product of dcn_shape {tuple(dcn_shape)}.")
  if any(len(g) != 256 for g in granules):
    raise ValueError(f"custom_mesh={custom_mesh!r} requires 256 devices per slice, got {[len(g) for g in granules]}.")

  per_granule = [
      _reshape_mesh_to_rings(
          mesh_utils.create_device_mesh([16, 16], granule, allow_split_physical_axes=False),
          custom_mesh,
      ).reshape(ici_shape)
      for granule in granules
  ]
  if len(per_granule) == 1:
    return per_granule[0]

  ndim = len(ici_shape)
  stacked = per_granule[0][None].repeat(len(per_granule), axis=0).copy()
  for idx, g_mesh in enumerate(per_granule):
    stacked[idx] = g_mesh
  interleaved = stacked.reshape(*dcn_shape, *ici_shape).transpose(
      tuple(val for i in range(ndim) for val in (i, i + ndim))
  )
  return interleaved.reshape(tuple(d * m for d, m in zip(dcn_shape, ici_shape)))


def _optimize_mesh_for_tpu_v6e(device_array: Any, devices: Sequence[jax.Device]) -> Any:
  """Reorders the second half of a 1D-ring TPU v6e `(2, 4, 1)` host mesh into a Hamiltonian ring."""
  if getattr(devices[0], "device_kind", None) != "TPU v6 lite":
    return device_array
  num_devices = len(devices)
  if num_devices not in device_array.shape or not all(hasattr(d, "coords") for d in devices):
    return device_array
  coords = [d.coords for d in devices]
  dims = tuple(max(c[i] for c in coords) - min(c[i] for c in coords) + 1 for i in range(len(coords[0])))
  if dims != (2, 4, 1):
    return device_array
  axis_idx = device_array.shape.index(num_devices)
  ndim = device_array.ndim
  perm = (axis_idx, *range(axis_idx), *range(axis_idx + 1, ndim))
  inv_perm = (*range(1, axis_idx + 1), 0, *range(axis_idx + 1, ndim))
  moved = device_array.transpose(perm).copy()
  moved[4:] = moved[-1:3:-1]
  return moved.transpose(inv_perm)


def create_mesh(
    *,
    ici: Mapping[str, int] | None = None,
    dcn: Mapping[str, int] | None = None,
    devices: Sequence[jax.Device] | None = None,
    axis_names: Sequence[str] = MESH_AXIS_NAMES,
    explicit: bool = False,
    custom_mesh: str = "",
    allow_split_physical_axes: bool = False,
    optimize_mesh_for_tpu_v6e: bool = False,
) -> Mesh:
  """Builds the device mesh over `axis_names` (by default `MESH_AXIS_NAMES`).

  The mesh is the product of two shapes over the same axes. `ici` spreads the
  devices within one slice over the fast interconnect; `dcn` spreads the slices
  themselves over the data center network. Each may carry at most one `-1`,
  meaning "use whatever is left", and both may do so at once because they
  resolve against different totals: devices per slice, and slices.

  Axes left unnamed default to 1, so callers name only what they split::

      create_mesh(ici={"fsdp": -1})                     # every device on fsdp
      create_mesh(ici={"fsdp": -1, "tensor": 2})        # tensor within a slice
      create_mesh(ici={"fsdp": -1}, dcn={"data": -1})   # fsdp in a slice, data across

  Mappings rather than one keyword per axis: the axis list is expected to grow
  (see `MESH_AXIS_NAMES`), and a keyword pair per axis per network would scale
  to twenty-odd parameters and churn every call site each time one is added.

  The defaults -- `ici={"fsdp": -1}`, `dcn={"data": -1}` -- put every device in
  a slice on `fsdp` and every slice on `data`. That split is the usual starting
  point because DCN bandwidth comfortably carries a gradient all-reduce but not
  the per-layer traffic that FSDP or tensor parallelism generate. On a single
  device it degenerates to a mesh where every `PartitionSpec` resolves to
  replicated, which is what keeps the single-device code path identical to the
  distributed one instead of a special case.

  Models are constructed inside `jax.sharding.set_mesh(mesh)`, on one device as
  on many. Flax places each parameter from its `*_metadata={"out_sharding": ...}`
  at construction and, with no mesh in scope, raises rather than degrading to
  replicated.

  Device ordering defaults to `jax.experimental.mesh_utils` over the detected
  topology unless `custom_mesh` selects a hand-tuned ring/torus layout
  (`hybrid_ring_64x4` or `hybrid_ring_32x8`).
  https://docs.jax.dev/en/latest/jax.experimental.mesh_utils.html

  The returned mesh must be the same object the training driver uses. A
  `Sharding` resolved against a different mesh is silently replaced by full
  replication rather than raising.

  Args:
    ici: Axis sizes within a slice, e.g. `{"fsdp": -1, "tensor": 2}`. Defaults
      to `{"fsdp": -1}`.
    dcn: Axis sizes across slices. Defaults to `{"data": -1}`.
    devices: Devices to arrange; defaults to `jax.devices()`.
    axis_names: The mesh axes, in mesh order. A model whose strategy needs other
      axes (Wan's `context`, say) passes its own; the `ici` / `dcn` defaults then
      apply only if `fsdp` / `data` are among them, otherwise both must be given.
    explicit: Give every axis `AxisType.Explicit`, so shardings are part of
      array types and `constrain` reshards instead of hinting. Legacy's
      `shard_mode=explicit`, and what the training engine's Zero-1 and deferred
      gradient all-reduce require.
    custom_mesh: Optional hand-tuned physical device ordering (`""` for default
      `mesh_utils` placement, or `"hybrid_ring_64x4"` / `"hybrid_ring_32x8"`).
    allow_split_physical_axes: Forwarded to `mesh_utils.create_device_mesh` /
      `create_hybrid_device_mesh` when logical axis sizes split physical torus
      dimensions.
    optimize_mesh_for_tpu_v6e: Reorders devices on TPU v6e topologies to align
      mesh axes with physical ICI rings before wrapping in `Mesh`.

  Returns:
    A `Mesh` over `axis_names`.

  Raises:
    ValueError: If `devices` is empty, if either mapping names an unknown axis,
      has more than one -1, contains an invalid size, or does not multiply to
      the device or slice count; or if `ici` / `dcn` is omitted while its
      default axis is not in `axis_names`.
  """
  axis_names = tuple(axis_names)
  ici_sizes = _shape_from_mapping(_default_sizes("fsdp", "ICI", axis_names) if ici is None else ici, "ICI", axis_names)
  dcn_sizes = _shape_from_mapping(_default_sizes("data", "DCN", axis_names) if dcn is None else dcn, "DCN", axis_names)

  devices = list(jax.devices() if devices is None else devices)
  if not devices:
    raise ValueError("Cannot build a mesh over zero devices.")

  num_slices = _count_slices(devices)
  if len(devices) % num_slices:
    raise ValueError(f"{len(devices)} devices do not divide evenly into {num_slices} slices.")

  ici_shape = _resolve_axis_sizes(ici_sizes, len(devices) // num_slices, "ICI", "per-slice device count", axis_names)
  dcn_shape = _resolve_axis_sizes(dcn_sizes, num_slices, "DCN", "slice count", axis_names)

  if _is_valid_custom_mesh(ici_shape, custom_mesh):
    device_array = _create_custom_device_mesh(ici_shape, dcn_shape, devices, custom_mesh)
  elif num_slices > 1:
    if allow_split_physical_axes:
      device_array = mesh_utils.create_hybrid_device_mesh(ici_shape, dcn_shape, devices, allow_split_physical_axes=True)
    else:
      device_array = mesh_utils.create_hybrid_device_mesh(ici_shape, dcn_shape, devices)
  else:
    # Not `create_hybrid_device_mesh` with an all-ones DCN shape: it requires
    # every device to expose `slice_index`, which single-slice TPU and CPU
    # devices do not, so it raises rather than degrading to the ICI-only case.
    if allow_split_physical_axes:
      device_array = mesh_utils.create_device_mesh(ici_shape, devices, allow_split_physical_axes=True)
    else:
      device_array = mesh_utils.create_device_mesh(ici_shape, devices)

  if optimize_mesh_for_tpu_v6e:
    device_array = _optimize_mesh_for_tpu_v6e(device_array, devices)

  axis_types = (jax.sharding.AxisType.Explicit,) * len(axis_names) if explicit else None
  return Mesh(device_array, axis_names, axis_types=axis_types)


# Legacy parallelism axes that are not part of `MESH_AXIS_NAMES`. Setting one
# of them != 1 raises `ValueError` rather than running a silent mismatch.
_UNSUPPORTED_PARALLELISM_AXES = (
    "diloco",
    "sequence",
    "context_usp_ulysses",
    "context_autoregressive",
    "pipeline",
    "autoregressive",
)


_UNSUPPORTED_MESH_FIELDS = {
    "custom_mesh_and_rule": CustomRule.DEFAULT,
    "custom_mesh_and_rule_for_eval": CustomRule.DEFAULT,
}

_MODEL_MESH_VALIDATOR: Callable[[Any], dict[str, Any]] | None = None


def register_model_mesh_validator(validator: Callable[[Any], dict[str, Any]]) -> None:
  """Registers the model-capability validator from `m3.models` without circular imports."""
  global _MODEL_MESH_VALIDATOR
  _MODEL_MESH_VALIDATOR = validator


def unsupported_mesh_settings(config) -> dict[str, Any]:
  """Returns parallelism or sharding settings enabled on `config` that m3 (or the selected model) does not implement."""
  unsupported = {}
  for network in ("ici", "dcn"):
    for axis in _UNSUPPORTED_PARALLELISM_AXES:
      field = f"{network}_{axis}_parallelism"
      if getattr(config, field, 1) != 1:
        unsupported[field] = getattr(config, field)
    stage_field = f"{network}_stage_parallelism"
    if getattr(config, stage_field, 1) != 1:
      unsupported[stage_field] = getattr(config, stage_field)
  for field, off in _UNSUPPORTED_MESH_FIELDS.items():
    value = get_enum(config, field, off)
    if value != off:
      unsupported[field] = value.value

  cp_active = any(getattr(config, f"{net}_context_parallelism", 1) not in (1, None) for net in ("ici", "dcn"))
  if cp_active:
    cp_strategy = getattr(config, "context_parallel_strategy", "all_gather") or "all_gather"
    if str(cp_strategy).lower() != "all_gather":
      unsupported["context_parallel_strategy"] = cp_strategy
    if bool(getattr(config, "context_parallel_load_balance", False)):
      unsupported["context_parallel_load_balance"] = True

  # Registered via `register_model_mesh_validator` at package init (kept decoupled for Bazel strict_deps).
  if _MODEL_MESH_VALIDATOR is not None and getattr(config, "model_name", None) is not None:
    unsupported.update(_MODEL_MESH_VALIDATOR(config))
  return unsupported


def mesh_from_config(config, devices: Sequence[jax.Device] | None = None) -> Mesh:
  """Builds the mesh `config` describes over `MESH_AXIS_NAMES`.

  Supports `data`, `fsdp`, `fsdp_transpose`, `context`, `tensor`, `tensor_sequence`,
  and `expert` across ICI and DCN (`stage` is held at 1 until a pipeline transform
  exists), `shard_mode` (`auto`, or `explicit` on models that support it),
  `custom_mesh` (`hybrid_ring_64x4` / `hybrid_ring_32x8`), `allow_split_physical_axes`,
  and `optimize_mesh_for_tpu_v6e`.
  """
  unsupported = unsupported_mesh_settings(config)
  if unsupported:
    raise ValueError(
        f"m3 builds its mesh over {MESH_AXIS_NAMES}; it does not implement {unsupported} (see m3/README.md)."
    )
  ici = {
      axis: (1 if axis == "stage" else getattr(config, f"ici_{axis}_parallelism", -1 if axis == "fsdp" else 1))
      for axis in MESH_AXIS_NAMES
  }
  dcn = {
      axis: (1 if axis == "stage" else getattr(config, f"dcn_{axis}_parallelism", -1 if axis == "data" else 1))
      for axis in MESH_AXIS_NAMES
  }
  explicit = get_enum(config, "shard_mode", ShardMode.AUTO) == ShardMode.EXPLICIT
  return create_mesh(
      ici=ici,
      dcn=dcn,
      devices=devices,
      explicit=explicit,
      custom_mesh=getattr(config, "custom_mesh", "") or "",
      allow_split_physical_axes=bool(getattr(config, "allow_split_physical_axes", False)),
      optimize_mesh_for_tpu_v6e=bool(getattr(config, "optimize_mesh_for_tpu_v6e", False)),
  )
