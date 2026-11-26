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

"""Autodiff of a stage's programs: `closure_convert_all`, and the split of a backward into I and W.

`closure_convert_all` is `jax.closure_convert` that hoists EVERY closed-over
tracer.  `ds_ep.ChunkPrograms.build_stash` closure-converts each layer's vjp
with it, so that everything a stash backward reads is an explicit constant --
int and bool ones too, which `jax.closure_convert` would bake into the callable
as escaped tracers -- and the forward can hand the backward its residuals as
arrays.

The split engine cuts a chunk's backward for the zero-bubble schedules (Qi et
al., "Zero Bubble Pipeline Parallelism", ICLR 2024) into I, what the input
gradient needs -- the recompute of the forward included -- and W, what the
weight gradients need beyond that, which the schedule runs later.  It cuts the
program the pipeline already runs, not the model: the chunk's fused backward
body is traced to one flat per-device jaxpr, and its equations are partitioned
by data dependence, each one whole.  The halves compute the same operations on
the same operands as the fused body, so they reproduce its outputs bit for bit
wherever XLA computes each operation the same way in either program.

  inline_eval  evaluates a jaxpr with the body of every nested jit and
               closed_call -- and of every remat (jax.checkpoint) -- evaluated
               in place, so that a trace of it holds none of those calls and
               the cut sees inside them.  An inlined remat's operands pass the
               optimization barrier its lowering puts on them, so its
               recompute still cannot merge with the forward's values.  Every
               other equation -- custom_vjp / custom_jvp calls, Pallas kernels,
               scan / cond / while, collectives -- is bound again as it is,
               atomic.
  trace_body   the flat per-device `Body` of jax.shard_map(fn) at abstract
               arguments: its jaxpr, the spec of each input and output, and
               where each input comes from -- a flat argument, or a constant of
               the trace that shard_map hoisted into an input (`Const`).
  split_jaxpr  the cut of a body's jaxpr (`JaxprSplit`): I computes the
               outputs it keeps, W every other output; the values W reads from
               I cross between them, in the order W first reads them.
  same_jaxpr   whether two traces made one program (`same_values` for
               constants).

No jax._src: jax.extend's jaxprs, equations, primitives and source info, and
each equation's bind parameters and context, which jax.core.eval_jaxpr re-binds
an equation with too.
"""

from __future__ import annotations

import dataclasses
import math

import jax
from jax import lax
from jax.extend import core as jcore
from jax.extend import source_info_util
from jax.sharding import PartitionSpec as P
import numpy as np

eval_jaxpr = jax.core.eval_jaxpr


def closure_convert_all(fun, *example_args):
  """Like `jax.closure_convert`, but hoists EVERY closed-over tracer.

  `jax.closure_convert` only hoists consts whose tangent space is nontrivial
  (`_maybe_perturbed`).  Any bool/int/PRNG-key residual -- i.e. every relu mask,
  every argmax/top-k routing index, every dropout key -- is instead *baked into*
  the returned callable as an escaped tracer, and blows up with
  `UnexpectedTracerError` the moment you call it outside the originating trace.
  """
  flat_args, in_tree = jax.tree.flatten(example_args)
  store = {}

  def flat_fun(*flat):
    out = fun(*jax.tree.unflatten(in_tree, list(flat)))
    out_flat, out_tree = jax.tree.flatten(out)
    store["out_tree"] = out_tree
    return out_flat

  closed = jax.make_jaxpr(flat_fun)(*flat_args)
  jaxpr, consts, out_tree = closed.jaxpr, list(closed.consts), store["out_tree"]
  n_top = len(example_args)

  def converted(*args_and_consts):
    args, cs = args_and_consts[:n_top], list(args_and_consts[n_top:])
    flat, tree = jax.tree.flatten(args)
    if tree != in_tree:
      raise TypeError(f"closure_convert_all: expected {in_tree}, got {tree}")
    return jax.tree.unflatten(out_tree, eval_jaxpr(jaxpr, cs, *flat))

  return converted, consts


# ------------------------------------------------------------ inline_eval
# The calls inline_eval evaluates in place: XLA inlines a jit's and a closed_call's computation anyway.  A remat's
# lowering emits its body in place too, behind an optimization barrier on the operands it flags (prevent_cse) once
# differentiated -- jax's ad_checkpoint._remat_lowering -- and inline_eval puts the same barrier there (_barrier).
_CALLS = frozenset({jcore.primitives.jit_p, jcore.primitives.closed_call_p, jcore.primitives.call_p})
_REMAT = jcore.primitives.remat_p


def _call_body(eqn):
  """The jaxpr an inlined call equation evaluates, and its constants' values."""
  j = eqn.params["jaxpr"] if "jaxpr" in eqn.params else eqn.params["call_jaxpr"]
  return getattr(j, "jaxpr", j), list(getattr(j, "consts", ()))


def _barrier(args, prevent_cse):
  """`args`, those `prevent_cse` flags (one bool for all, or one per argument) passed through ONE optimization
  barrier: what a differentiated remat's lowering does to its operands, so that XLA cannot merge the recompute
  behind it with the forward's values (common-subexpression elimination) and keep those alive instead."""
  flags = [prevent_cse] * len(args) if isinstance(prevent_cse, bool) else list(prevent_cse)
  idx = [k for k, f in enumerate(flags) if f]
  out = list(args)
  if idx:
    for k, v in zip(idx, lax.optimization_barrier(tuple(args[k] for k in idx))):
      out[k] = v
  return out


def inline_eval(jaxpr, consts, *args, inline_remat=True):
  """Evaluate `jaxpr` at `args` (`consts`: its constvars' values) under the current trace, as jax.core.eval_jaxpr
  does, but with the body of every jit and closed_call -- and with `inline_remat`, of every remat -- evaluated in
  place: a trace of it holds none of those calls.  A differentiated remat's operands with prevent_cse pass one
  optimization barrier first, as in its lowering.  Every other equation is bound again with its own parameters
  (`get_bind_params`), source info and context."""
  if len(consts) != len(jaxpr.constvars) or len(args) != len(jaxpr.invars):
    raise TypeError(
        f"inline_eval: {len(consts)} consts and {len(args)} args for a jaxpr of {len(jaxpr.constvars)} constvars "
        f"and {len(jaxpr.invars)} invars"
    )
  env = {}

  def read(v):
    return v.val if isinstance(v, jcore.Literal) else env[v]

  for v, x in zip(jaxpr.constvars, consts):
    env[v] = x
  for v, x in zip(jaxpr.invars, args):
    env[v] = x
  for eqn in jaxpr.eqns:
    ins = [read(v) for v in eqn.invars]
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
      if eqn.primitive in _CALLS:
        outs = inline_eval(*_call_body(eqn), *ins, inline_remat=inline_remat)
      elif eqn.primitive is _REMAT and inline_remat:
        if eqn.params["differentiated"]:
          with eqn.ctx.manager:
            ins = _barrier(ins, eqn.params["prevent_cse"])
        with source_info_util.extend_name_stack("checkpoint"):
          outs = inline_eval(*_call_body(eqn), *ins, inline_remat=True)
      else:
        with eqn.ctx.manager:
          outs = eqn.primitive.bind(*ins, **eqn.primitive.get_bind_params(eqn.params))
        if not eqn.primitive.multiple_results:
          outs = [outs]
    for v, x in zip(eqn.outvars, outs):
      if not isinstance(v, jcore.DropVar):
        env[v] = x
  return [read(v) for v in jaxpr.outvars]


# ------------------------------------------------------------- trace_body
@dataclasses.dataclass(frozen=True, eq=False)
class Const:
  """An input of a traced body that is a constant of the trace -- a splash attention mask table, rope frequencies, an
  array a jitted helper closes over -- which shard_map hoisted into an input of its own (replicated)."""

  value: object


@dataclasses.dataclass(frozen=True, eq=False)
class Body:
  """A flat per-device body (trace_body): `jaxpr`, the spec of each of its invars (`in_specs`) and outvars
  (`out_specs`), and per invar where its value comes from (`src`): an int, the position among the traced function's
  flat arguments, or a Const.  Its outvars are the function's flat outputs, in order."""

  jaxpr: object
  in_specs: tuple
  out_specs: tuple
  src: tuple

  def operands(self, flat_args):
    """The values of `jaxpr`'s invars, from the traced function's flat arguments (a Const: its value)."""
    return [flat_args[s] if isinstance(s, int) else s.value for s in self.src]


def _spec_leaves(specs):
  return jax.tree.leaves(specs, is_leaf=lambda x: x is None or isinstance(x, P))


def _shard_map_body(fn, avals, mesh, in_specs, out_specs):
  """The Body of jax.shard_map(fn) traced at `avals` (src: positions in `avals`, or Consts)."""
  closed = jax.make_jaxpr(
      jax.shard_map(fn, mesh=mesh, in_specs=tuple(in_specs), out_specs=tuple(out_specs), check_vma=False)
  )(*avals)
  outer = closed.jaxpr
  if len(outer.eqns) != 1 or outer.eqns[0].primitive.name != "shard_map":
    raise ValueError(f"trace_body: expected one shard_map, got {[e.primitive.name for e in outer.eqns]}")
  (sm,) = outer.eqns
  if list(sm.outvars) != list(outer.outvars):
    raise ValueError("trace_body: the shard_map's outputs are not the trace's")
  pos = {v: i for i, v in enumerate(outer.invars)}
  const = dict(zip(outer.constvars, closed.consts))
  src = []
  for v in sm.invars:
    if v in pos:
      src.append(pos[v])
    elif v in const:
      src.append(Const(const[v]))
    else:
      raise ValueError(f"trace_body: shard_map input {v} is neither an argument nor a constant of the trace")
  return Body(sm.params["jaxpr"], tuple(sm.params["in_specs"]), tuple(sm.params["out_specs"]), tuple(src))


def trace_body(fn, avals, mesh, in_specs, out_specs, *, inline_remat=True):
  """The flat per-device Body of jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
  check_vma=False) at `avals`, a tuple of fn's arguments as pytrees of ShapeDtypeStructs (in_specs / out_specs:
  pytrees of PartitionSpecs over its arguments and outputs, as for shard_map).  Its src positions are those of
  jax.tree.leaves(avals), and its outvars are jax.tree.leaves of fn's outputs.

  Two traces: shard_map(fn) itself, then its per-device jaxpr again through inline_eval (`inline_remat`: the remats
  too) inside a shard_map of the same specs, every input of the first an argument of the second -- the constants
  shard_map hoisted too.  Constants of the second trace (arrays an inlined jit closed over) become Consts."""
  flat, in_tree = jax.tree.flatten(avals)
  fin, fout = _spec_leaves(in_specs), _spec_leaves(out_specs)
  if len(fin) != len(flat):
    raise ValueError(f"trace_body: {len(fin)} in_specs for {len(flat)} arguments")

  def flat_fn(*xs):
    outs = jax.tree.leaves(fn(*jax.tree.unflatten(in_tree, list(xs))))
    if len(outs) != len(fout):
      raise ValueError(f"trace_body: {len(fout)} out_specs for {len(outs)} outputs")
    return tuple(outs)

  first = _shard_map_body(flat_fn, flat, mesh, fin, fout)
  j1, c1 = getattr(first.jaxpr, "jaxpr", first.jaxpr), list(getattr(first.jaxpr, "consts", ()))
  avals2 = [flat[s] if isinstance(s, int) else jax.ShapeDtypeStruct(s.value.shape, s.value.dtype) for s in first.src]

  def inlined(*xs):
    return tuple(inline_eval(j1, c1, *xs, inline_remat=inline_remat))

  second = _shard_map_body(inlined, avals2, mesh, first.in_specs, first.out_specs)
  src = tuple(first.src[s] if isinstance(s, int) else s for s in second.src)
  return Body(second.jaxpr, second.in_specs, second.out_specs, src)


# ------------------------------------------------------------ split_jaxpr
# Effects an equation may carry and still move to W, or out of a program that does not need it: a collective's named
# mesh axis, which types the equation rather than doing anything.  Any other -- a callback, an ordered or a state
# effect -- would run at another time in the schedule, or not at all, and split_jaxpr refuses the body.
MOVABLE_EFFECTS = ("NamedAxisEffect",)
# Equations that only move or retype a value.  A W of these alone computes nothing of its own -- I computed every
# deferred output, W would reshape or convert it (a backward whose remat stays one equation: inline_remat=False) --
# and split_jaxpr refuses it.
MOVES = frozenset(
    {"broadcast_in_dim", "reshape", "squeeze", "expand_dims", "transpose", "convert_element_type", "copy", "copy_p"}
)


@dataclasses.dataclass(frozen=True, eq=False)
class JaxprSplit:
  """The two halves of a body's jaxpr (split_jaxpr):

    i_jaxpr(*jaxpr's invars)                       -> (*outvars[i_out], *crossing)
    w_jaxpr(*crossing, *jaxpr's invars[w_inputs])  -> (*outvars[w_out])

  `crossing`: the values I computes that W reads -- W's equations in order, each one's operands in order, then the
  outputs W returns as they are -- the same layout in every trace of the same program, as every McJAX process
  traces its own.  An operand of an inlined remat that W reads crosses as its barrier's output.  `i_inputs` /
  `w_inputs`: the positions of the invars each half reads, ascending.  `stats`: the body's equations (`eqns`), I's
  and W's (`i_eqns`, `w_eqns`; any other is dead code, in neither), those I computes that W's outputs need too
  (`shared`), and the crossing set's size (`crossing`) and bytes (`crossing_bytes`, per device: a body's avals are
  per-device)."""

  i_jaxpr: object
  w_jaxpr: object
  i_out: tuple
  w_out: tuple
  crossing: tuple
  i_inputs: tuple
  w_inputs: tuple
  stats: dict

  @property
  def crossing_avals(self):
    return tuple(v.aval for v in self.crossing)


def _nbytes(aval):
  return math.prod(aval.shape) * aval.dtype.itemsize


def _closure(eqns, producer, roots):
  """The positions of the equations `roots` depend on, each equation whole: all of its operands' producers too."""
  need, stack, seen = set(), [v for v in roots if isinstance(v, jcore.Var)], set()
  while stack:
    v = stack.pop()
    if v in seen:
      continue
    seen.add(v)
    k = producer.get(v)
    if k is None or k in need:
      continue
    need.add(k)
    stack.extend(u for u in eqns[k].invars if isinstance(u, jcore.Var))
  return need


def _effects(eqns):
  return frozenset().union(*(e.effects for e in eqns))


def split_jaxpr(jaxpr, i_out):
  """Cut `jaxpr` (a Body's: per-device and flat) by data dependence -> JaxprSplit.

  `i_out`: the outvars I computes (their positions, or an int n: the first n) -- the input gradient, any
  accumulator leaf not deferred; W computes the others.  I is every equation its outputs need; W every equation the
  others need that I does not compute (each equation whole, in the body's order); the crossing set the values I
  computes that W reads.  No equation runs twice, and none is changed: a custom_vjp call, a kernel or a loop that
  both halves' outputs need is computed in I, its outputs crossing to W as needed.

  Refuses (ValueError) a body with an effect outside MOVABLE_EFFECTS, an I or a W that computes nothing -- no output
  kept or none deferred, no equation in I, or in W none but MOVES (every deferred output an input or one of I's
  values, reshaped or converted at most) -- and positions out of range."""
  outvars = list(jaxpr.outvars)
  i_out = tuple(sorted(set(range(i_out) if isinstance(i_out, int) else i_out)))
  if any(not 0 <= k < len(outvars) for k in i_out):
    raise ValueError(f"split_jaxpr: i_out {i_out} out of range for {len(outvars)} outputs")
  w_out = tuple(k for k in range(len(outvars)) if k not in i_out)
  if not i_out or not w_out:
    raise ValueError(
        f"split_jaxpr: I keeps {len(i_out)} of the {len(outvars)} outputs and W {len(w_out)}: nothing to split"
    )
  eqns = list(jaxpr.eqns)
  for k, e in enumerate(eqns):
    bad = sorted(str(x) for x in e.effects if type(x).__name__ not in MOVABLE_EFFECTS)
    if bad:
      raise ValueError(
          f"split_jaxpr: equation {k} ({e.primitive.name}) has effects {bad}: split, it would run at another "
          "time, or not at all"
      )
  producer = {v: k for k, e in enumerate(eqns) for v in e.outvars if not isinstance(v, jcore.DropVar)}
  need_i = _closure(eqns, producer, [outvars[k] for k in i_out])
  need_w = _closure(eqns, producer, [outvars[k] for k in w_out])
  i_eqns, w_eqns = sorted(need_i), sorted(need_w - need_i)
  if not i_eqns:
    raise ValueError(f"split_jaxpr: I would compute nothing: no equation of the {len(eqns)} computes the outputs kept")
  if all(eqns[k].primitive.name in MOVES for k in w_eqns):
    raise ValueError(
        f"split_jaxpr: W would compute nothing: the deferred outputs are inputs or I's values, which its "
        f"{len(w_eqns)} equation(s) {sorted({eqns[k].primitive.name for k in w_eqns})} only move"
    )
  w_defined = {v for k in w_eqns for v in eqns[k].outvars}
  invar_pos = {v: i for i, v in enumerate(jaxpr.invars)}
  constvars = set(jaxpr.constvars)
  crossing, crossed, w_reads = [], set(), set()

  def w_read(v):
    if not isinstance(v, jcore.Var) or v in w_defined or v in constvars:
      return
    if v in invar_pos:
      w_reads.add(invar_pos[v])
    elif v not in crossed:
      crossed.add(v)
      crossing.append(v)

  for k in w_eqns:
    for v in eqns[k].invars:
      w_read(v)
  for k in w_out:
    w_read(outvars[k])
  i_vars = [v for k in i_eqns for v in eqns[k].invars] + [outvars[k] for k in i_out]
  i_inputs = tuple(sorted({invar_pos[v] for v in i_vars if isinstance(v, jcore.Var) and v in invar_pos}))
  w_inputs = tuple(sorted(w_reads))
  i_jaxpr = jaxpr.replace(
      eqns=[eqns[k] for k in i_eqns],
      outvars=[outvars[k] for k in i_out] + crossing,
      effects=_effects(eqns[k] for k in i_eqns),
  )
  w_jaxpr = jaxpr.replace(
      invars=crossing + [jaxpr.invars[i] for i in w_inputs],
      eqns=[eqns[k] for k in w_eqns],
      outvars=[outvars[k] for k in w_out],
      effects=_effects(eqns[k] for k in w_eqns),
  )
  stats = {
      "eqns": len(eqns),
      "i_eqns": len(i_eqns),
      "w_eqns": len(w_eqns),
      "shared": len(need_i & need_w),
      "crossing": len(crossing),
      "crossing_bytes": sum(_nbytes(v.aval) for v in crossing),
  }
  return JaxprSplit(i_jaxpr, w_jaxpr, i_out, w_out, tuple(crossing), i_inputs, w_inputs, stats)


def same_values(a, b):
  """Whether `a` and `b` (arrays or scalars) are one value: the same shape, dtype and bits."""
  a, b = np.asarray(a), np.asarray(b)
  return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def same_jaxpr(a, b):
  """Whether jaxprs `a` and `b` (open or closed) are one program: printed alike -- every equation's primitive,
  parameters (nested jaxprs too) and variables, which the printer names in order of appearance, so that two traces
  of one function compare alike -- and their constants alike (`same_values`).  A parameter printed with an object's
  address (a function) makes two traces of one function differ: no false match."""
  ca, cb = list(getattr(a, "consts", ())), list(getattr(b, "consts", ()))
  return len(ca) == len(cb) and str(a) == str(b) and all(same_values(x, y) for x, y in zip(ca, cb))
