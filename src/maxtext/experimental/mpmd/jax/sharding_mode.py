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

"""Which sharding mode the code being traced runs in, and the helpers that
model code shards through.

One stage function is traced in four situations, and this module is how it
tells them apart:

  none      No mesh in context: a single-device reference (`ds_ep_parity`'s
            run of the DeepSeek port in `deepseek/`), or `eval_shape` over
            plain avals.  Every helper is the identity, because there is
            nothing to shard.
  manual    Inside a `shard_map` over every axis (`ds_ep`'s per-stage
            programs, or `manual()` below), where values are per-device
            blocks.  Every helper is the identity again: a second `shard_map`
            over the same axes fails in every jax version we run.
  auto      A sub-mesh whose axes are `AxisType.Auto`, in which XLA
            propagates the shardings.  A helper emits exactly the expression
            the model wrote before this module existed: today's program.
  explicit  A sub-mesh whose axes are `AxisType.Explicit`.  Every value
            carries its sharding in its type, and an op whose result type
            would be ambiguous will not trace without an `out_sharding=` or a
            `reshard`.

The mode is read off the abstract mesh in context.  No production path
enters auto or explicit: `ds_ep` runs the models per device, inside its own
shard_maps (manual), and `ds_ep_parity`'s port reference runs them with no
mesh (none).  The auto and explicit branches of the helpers below,
`trace_context` (which sets the context), `manual()`'s shard_map and
`replicate()` are exercised only by the model tests (`sharding_mode_test`,
`deepseek_model_test`, `qwen35_explicit_test`), which trace the ports at
global view in both modes, so that the ports keep tracing there.  Such a
caller enters `trace_context` inside every program it traces under explicit;
under auto every helper is the model's own expression with or without a
mesh, and the context is not free there (see `trace_context`), so it is
entered only for `manual`'s per-device regions.  Reading the context rather
than an argument is what lets one stage function serve every case: under an
`eval_shape` of plain avals there is no context, so the helpers step aside.
The mode is never passed around as a Python flag, so there is no flag that
can disagree with the types.

The helpers come in three groups, and the grouping is the point:

  (1) Type obligations, explicit only: `out_sharding`, `retype`, `zeros`,
      and the forms a `custom_vjp` rule uses, `bwd_out_sharding`,
      `sharding_of` and `retype_like` (types read off an aval).
      They are what explicit needs in order to trace at all, and they emit
      NOTHING under auto, so that auto stays the same program, byte for
      byte.
  (2) Strategy forcing, both modes: `replicate`, the ONLY place auto gets a
      `with_sharding_constraint`; it forces the same layout in both modes,
      so that a comparison of the two is one of typing.  The models call it
      nowhere (`ShardPolicy` holds the default policy alone).
  (3) Ops that need an `out_sharding=` under explicit: `dot_general`,
      `einsum`, `take`, `set_at`, `bincount`, plus the `manual` region.  The
      auto/none branch of each but `manual` (a shard_map under auto, on
      purpose) is the literal expression the model used before, and
      `sharding_mode_test.py` asserts that its jaxpr is the same.

Two rules for the code that uses them.  First, model code never calls
`reshard`, `with_sharding_constraint` or `out_sharding=` itself, so that every
departure from today's program is in this file.  Second, a `custom_vjp` rule
types what it returns only from the primals' own avals: an op inside it gets
`out_sharding=bwd_out_sharding(primal)` (or `sharding_of(x)` for a result
typed exactly like `x`), and a value that comes out typed otherwise goes
through `retype_like(value, primal)`.  A cotangent must be
typed exactly as its primal's cotangent type (jax refuses a bwd rule that
returns anything else), and a bwd rule is traced lazily, during
transposition.  The abstract mesh is still captured there, but a Python
context variable set in the forward is not, so the primal's own type is the
one reliable witness.

Version notes, for jax 0.10.2 / 0.11.x / 0.12: this file never constructs an
`AbstractMesh(...)`, never calls `make_mesh` with keywords and never uses
`are_all_axes_*`; all three changed in 0.12.  Shardings are always
`NamedSharding(get_abstract_mesh(), spec)`, never a bare `PartitionSpec`:
a bare spec needs a concrete mesh set with `jax.set_mesh`, which is illegal
inside `jit` and would anyway be one mesh for a pipeline that has one per
stage.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

Mode = Literal["none", "manual", "auto", "explicit"]


# --------------------------------------------------------------------- policy
@dataclasses.dataclass(frozen=True)
class ShardPolicy:
  """How a model shards: the default policy, today's program, and no other.

  A config carries one (`config.shard`) and the helpers below never read it:
  what a helper emits is the mode's (see the module docstring).  The
  strategies a policy once forced in both modes -- gathered or
  tensor-parallel weights, other MoE dispatches, attention per device, a
  recomputed embedding lookup, fixed routing -- served whole-model pipelines
  that traced at global view, and went with them; ds_ep runs the models per
  device, inside its own shard_maps, where a policy has no say.  Frozen, so a
  config that holds one stays hashable.
  """


# ------------------------------------------------------------------- the mode
def _mode_of(mesh, what) -> Mode:
  """Classify an abstract mesh: empty, all Manual, all Explicit or all Auto."""
  if mesh.empty:
    return "none"
  types = set(mesh.axis_types)
  if types == {AxisType.Manual}:
    return "manual"
  if types == {AxisType.Explicit}:
    return "explicit"
  if types == {AxisType.Auto}:
    return "auto"
  # A mesh that mixes Explicit and Auto axes, or one only partly inside a
  # shard_map.  No sub-mesh in this repo is either, and guessing which axes
  # the batch rides on would be a silent remap, so it is an error instead.
  axes = ", ".join(f"{n}: {t}" for n, t in zip(mesh.axis_names, mesh.axis_types))
  raise ValueError(
      f"{what} mixes axis types ({axes}).  The sharding helpers "
      "need every axis Auto, every axis Explicit or every axis "
      "Manual."
  )


def context_mode() -> Mode:
  """The mode of the abstract mesh in context (see the module docstring)."""
  return _mode_of(jax.sharding.get_abstract_mesh(), "the context mesh")


def aval_mode(x) -> Mode:
  """The same classification, read off the mesh in `x`'s type.

  A committed array, a tracer, a `ShapeDtypeStruct` or an aval (a jaxpr
  variable's) all work.  For a value with no mesh in its type (a numpy array,
  a Python scalar, an untyped `ShapeDtypeStruct`) the answer is 'none'.
  """
  try:
    t = jax.typeof(x)
  except TypeError:  # already an aval, e.g. `ShapedArray`
    t = x
  mesh = getattr(getattr(t, "sharding", None), "mesh", None)
  if mesh is None:
    return "none"
  return _mode_of(mesh, f"the type of {t}")


def trace_context(abstract_mesh):
  """Trace with `abstract_mesh` in context.  `None` means no context.

  The models' one way to `jax.sharding.use_abstract_mesh` (their call-site
  tests refuse one of their own; `ds_ep_lineage`, which runs MaxText's
  Lineage code, enters it itself), and only the model tests enter it (see the
  module docstring).  Enter it INSIDE the function being traced, never around
  the call.  `jax.set_mesh` is illegal inside `jit`, and a context around the
  call becomes part of the jit cache key, which costs one extra trace per
  executable.  A concrete `Mesh` is accepted too and stands for its
  `abstract_mesh`.

  Under an Auto mesh it is not free, which is why a caller should enter it
  under auto only for `manual`'s per-device regions (Qwen's
  `gdn_core='manual'`).  Every aval traced under it carries the mesh, and
  Pallas's TPU lowering compares a kernel's avals with mesh-less ones it
  builds itself (`load_aval != aval_out` in its load rule), so a kernel
  traced under it gets an identity `vector.shape_cast` per load: jax's splash
  kernel lowers to a different Mosaic body, with bitwise the same results, in
  every program that calls it.
  """
  if abstract_mesh is None:
    return contextlib.nullcontext()
  return jax.sharding.use_abstract_mesh(getattr(abstract_mesh, "abstract_mesh", abstract_mesh))


def batch_axes() -> tuple[str, ...]:
  """The context mesh's non-Manual axes, i.e. the axes the batch rides on.

  Every axis of an MPMD sub-mesh shards the batch (`ds_ep` splits a stage's
  microbatch over all of them), so these are simply the axes `manual()` would
  map over.
  """
  m = jax.sharding.get_abstract_mesh()
  return tuple(n for n, t in zip(m.axis_names, m.axis_types) if t != AxisType.Manual)


def act_spec(ndim: int, batch_dim: int = 0) -> P:
  """The PartitionSpec of a batch-major activation of rank `ndim`.

  The batch dim is sharded over `batch_axes()` and nothing else is.  With no
  batch axes (no mesh, or inside `manual`) the spec is all None.
  """
  axes = batch_axes()
  spec = [None] * ndim
  if axes and ndim:
    spec[batch_dim] = axes[0] if len(axes) == 1 else axes
  return P(*spec)


def named(spec) -> NamedSharding:
  """`NamedSharding(get_abstract_mesh(), spec)`, the only sharding the helpers
  ever build.

  It needs no concrete mesh in context, only the abstract one that
  `trace_context` sets.  With no mesh at all it raises: the helpers only call
  it under auto or explicit, where there always is one, so a call without a
  mesh is a mistake to surface, not to paper over.
  """
  m = jax.sharding.get_abstract_mesh()
  if m.empty:
    raise ValueError(f"named({spec}) needs a mesh in context: call it while " "tracing under mode.trace_context(...).")
  return NamedSharding(m, spec)


def _type(x):
  try:
    return jax.typeof(x)
  except TypeError:  # already an aval, e.g. `ShapedArray`
    return x


def _partitions(spec):
  """A spec's partitions.  `tuple(spec)` raises once it has a reduced or
  unreduced set, so never iterate a spec that may carry one."""
  return tuple(spec.partitions)


# ---------------------------------------------------- (1) type obligations
def out_sharding(spec) -> NamedSharding | None:
  """The value for an op's `out_sharding=`: `named(spec)` under explicit, else
  None, which is the op's default."""
  return named(spec) if context_mode() == "explicit" else None


def retype(x, spec):
  """`reshard` `x` to `spec` under explicit, the identity in any other mode."""
  if context_mode() == "explicit":
    return jax.sharding.reshard(x, named(spec))
  return x


def zeros(shape, dtype, spec):
  """`jnp.zeros`, typed `spec` under explicit, e.g. for a scan carry.

  A scan's carry must have the same type on the way in as on the way out, and
  a `zeros` without an `out_sharding` is typed replicated.
  """
  if context_mode() == "explicit":
    return jnp.zeros(shape, dtype, out_sharding=named(spec))
  return jnp.zeros(shape, dtype)


def retype_like(x, like):
  """`x` typed as the cotangent of `like` when `like`'s type is Explicit,
  else `x`.  For every value but a reduced or unreduced one that is `like`'s
  own type (`bwd_out_sharding`).

  The form of `retype` a `custom_vjp` rule uses: the type comes from an aval
  (`bwd_out_sharding`), never from the context, which a lazily traced bwd
  rule does not reliably see.  A cotangent can come out of perfectly typed
  arithmetic typed otherwise than its primal -- the cross entropy's follows
  the targets when only they are batch-sharded -- and jax refuses a bwd rule
  that returns it.  `x` and `like` have the same shape.  Nothing is emitted
  when the two types already agree, so the common case costs no op at all.
  """
  sh = bwd_out_sharding(like)
  if sh is None or jax.typeof(x).sharding == sh:
    return x
  return jax.sharding.reshard(x, sh)


# --------------------------------------------------- (2) strategy forcing
def replicate(x):
  """Make `x` replicated over the stage, in both modes.

  A `reshard` under explicit and a `with_sharding_constraint` under auto.  A
  `reshard` on Auto axes would lower to nothing, so the dispatch matters: the
  wrong one is a silent no-op.  The identity under none and manual.
  """
  mode = context_mode()
  if mode == "explicit":
    return jax.sharding.reshard(x, named(P()))
  if mode == "auto":
    return jax.lax.with_sharding_constraint(x, named(P()))
  return x


# ------------------------------------------- (3) ops that need out_sharding
def dot_general(lhs, rhs, dims, *, out_spec, precision, preferred_element_type=None):
  """`jax.lax.dot_general`, with `out_sharding=named(out_spec)` under explicit.

  `out_spec=None` leaves the result type to the op's sharding rule in every
  mode.
  """
  if out_spec is not None and context_mode() == "explicit":
    return jax.lax.dot_general(
        lhs, rhs, dims, precision=precision, preferred_element_type=preferred_element_type, out_sharding=named(out_spec)
    )
  return jax.lax.dot_general(lhs, rhs, dims, precision=precision, preferred_element_type=preferred_element_type)


def einsum(eq, *operands, out_spec, precision=None, preferred_element_type=None):
  """`jnp.einsum`, with `out_sharding=named(out_spec)` under explicit."""
  if out_spec is not None and context_mode() == "explicit":
    return jnp.einsum(
        eq, *operands, precision=precision, preferred_element_type=preferred_element_type, out_sharding=named(out_spec)
    )
  return jnp.einsum(eq, *operands, precision=precision, preferred_element_type=preferred_element_type)


def take(table, ids, *, out_spec):
  """`table[ids]`: rows of `table` at integer `ids`, e.g. an embedding lookup.

  Under explicit, a gather with batch-sharded indices has no unambiguous
  result type, so it becomes `.at[ids].get(out_sharding=)`, which jax
  implements as an auto_axes island.  Anywhere else it is `table[ids]`,
  whose jaxpr is that of `.at[ids].get()`.  It is never `jnp.take`, which
  fills out-of-range rows with NaN rather than promising they are in bounds.
  """
  if context_mode() == "explicit":
    return table.at[ids].get(out_sharding=named(out_spec))
  return table[ids]


def set_at(x, ids, values, *, out_spec):
  """`x.at[ids].set(values)`, typed `out_spec` under explicit.

  The scatter twin of `take`: writing `values` at batch-sharded `ids` into a
  buffer has no unambiguous result type under explicit (the dense MoE's
  routing-weight scatter), so it needs one.  Anywhere else it is the plain
  expression.
  """
  if context_mode() == "explicit":
    return x.at[ids].set(values, out_sharding=named(out_spec))
  return x.at[ids].set(values)


def bincount(x, *, length):
  """`jnp.bincount`.  Under explicit the counts are typed replicated: a
  scatter-add of sharded indices into a replicated buffer needs one."""
  if context_mode() == "explicit":
    return jnp.bincount(x, length=length, out_sharding=named(P()))
  return jnp.bincount(x, length=length)


def manual(f, *, in_specs, out_specs, name):
  """`f` in a `shard_map` over `batch_axes()`, under auto and explicit.

  Under none and manual this returns `f` unchanged: there is nothing to map,
  or the axes are already Manual.  Call it while tracing, since that is when
  the mode is read.  Under explicit the arguments are first `retype`d to
  `in_specs`, because a `shard_map` refuses inputs typed otherwise.  The
  `shard_map` runs in `jax.named_scope(name)` and the retype does not, so a
  profile can measure the time in the scope -- the same per-device program
  in both modes -- apart from the retype, which is where the modes differ.

  Every operand needs a PartitionSpec; `P()` is a replicated one.  A `None`
  is refused in every mode: `shard_map` reads it as "close over the whole
  array", which under auto hands the body the full array and under explicit
  fails for any sharded operand, so the two modes would part ways.
  """
  if any(s is None for s in jax.tree.leaves(in_specs, is_leaf=lambda s: s is None or isinstance(s, P))):
    raise ValueError(
        f"manual({name!r}): in_specs {in_specs} has a None.  Give every "
        "operand a PartitionSpec; P() is replicated (shard_map reads None as "
        '"close over the whole array", which explicit refuses for a sharded '
        "operand and auto does not)."
    )
  mode = context_mode()
  if mode in ("none", "manual"):
    return f
  axes = frozenset(batch_axes())
  is_spec = lambda s: isinstance(s, P)

  def run(*args):
    if mode == "explicit":
      args = jax.tree.map(lambda spec, sub: jax.tree.map(lambda a: retype(a, spec), sub), in_specs, args, is_leaf=is_spec)
    with jax.named_scope(name):
      return jax.shard_map(f, mesh=None, in_specs=in_specs, out_specs=out_specs, axis_names=axes, check_vma=False)(*args)

  return run


def sharding_of(x):
  """The sharding of `x`'s type when it is Explicit, else None: the
  `out_sharding=` of an op whose result is typed exactly like `x`.

  For a `custom_vjp` forward whose output is typed like an input, and for a
  bwd rule that types a cotangent like another cotangent.  Both are the
  value's own type, which for a reduced or unreduced value is not its
  cotangent's (`bwd_out_sharding`).
  """
  if aval_mode(x) == "explicit":
    return _type(x).sharding
  return None


def bwd_out_sharding(primal):
  """The `out_sharding=` a `custom_vjp` bwd rule gives the cotangent of
  `primal`: the cotangent type of `primal`'s type when it is Explicit, and
  None otherwise.

  That is the primal's own type for every value but a reduced or unreduced
  one, whose cotangent has the two swapped: a value typed reduced (one copy
  promised equal on every device) gets an unreduced cotangent (each device's
  partial sum), and jax refuses a bwd rule that returns one typed like the
  primal.  See the module docstring for why the type, and not the context,
  decides.  A value typed exactly like another one is `sharding_of`.
  """
  if aval_mode(primal) != "explicit":
    return None
  sharding = _type(primal).sharding
  spec = sharding.spec
  reduced = frozenset(getattr(spec, "reduced", ()) or ())
  unreduced = frozenset(getattr(spec, "unreduced", ()) or ())
  if not (reduced or unreduced):
    return sharding
  return NamedSharding(sharding.mesh, P(*_partitions(spec), reduced=unreduced, unreduced=reduced))
