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

"""Tests for stage_ad: `closure_convert_all`, with which ds_ep's stash programs closure-convert each layer's vjp, and
the split engine that cuts a backward into I and W for the zero-bubble schedules.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/stage_ad_test.py -q

SplitJaxpr cuts the vjp of toy layers -- an MLP, a routed MoE layer over an expert-parallel mesh -- traced as a
pipeline chunk's backward is (stage_ad.trace_body), runs I then W as two programs, and holds their outputs to the
vjp's one program bit for bit; W must be the weight-gradient matmuls alone.  The programs compile without XLA:CPU's
YNNPACK fusions (testing.ynn_off), whose dots sum in an order that follows the fusion around them.
"""

import collections
import unittest

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax import stage_ad
from maxtext.experimental.mpmd.jax import testing


def _relerr(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


class ClosureConvertTest(unittest.TestCase):

  def test_hoists_non_differentiable_consts(self):
    """`jax.closure_convert` leaves int/bool consts as escaped tracers."""

    def f(x):
      mask = x > 0  # bool: zero tangent space
      idx = jnp.argmax(x).astype(jnp.int32)  # int: zero tangent space
      return lambda y: (y * mask, y * idx)

    x = jnp.array([-1.0, 2.0, -3.0, 4.0])
    conv, consts = stage_ad.closure_convert_all(f(x), jnp.zeros(4))
    self.assertEqual(len(consts), 2)  # both hoisted, none baked in
    self.assertTrue(all(isinstance(c, jax.Array) for c in consts))
    # Callable outside the originating trace -- the whole point.
    got = jax.jit(conv)(jnp.ones(4), *consts)
    self.assertEqual(_relerr(got[0], np.array([0.0, 1.0, 0.0, 1.0])), 0.0)

  def test_rejects_wrong_input_tree(self):
    conv, consts = stage_ad.closure_convert_all(lambda a, b: a + b, 1.0, 2.0)
    with self.assertRaisesRegex(TypeError, "closure_convert_all"):
      conv(1.0, *consts)


# ------------------------------------------------------------ the split engine
# The toy layers, per device: B rows of width D, hidden width H; the MoE's E experts (EL a device over EP = 2) take
# each of the N = 2 B rows its 'expert' ring gathers to K of them.  The widths differ, so no activation has a weight's
# shape.
B, D, H = 6, 16, 32
E, K, EP = 4, 2, 2
EL = E // EP
MATMULS = frozenset({"dot_general", "ragged_dot_general"})
LAYOUT = frozenset({"transpose", "reshape", "broadcast_in_dim", "squeeze", "convert_element_type"})


def mlp(w, x):
  return jnp.tanh(x @ w["w1"]) @ w["w2"]


def moe(w, x):
  """A routed MoE layer on an expert-parallel mesh, as ds_ep's XLA kernels run one: the 'expert' ring's rows
  gathered, routed top-k, sorted by local expert (argsort), the experts' ragged dots, unsorted, weighted, and the
  rows reduce-scattered back.  A row routed to another device's expert goes through this device's last one as
  zeros."""
  xg = lax.all_gather(x, "expert", tiled=True)
  probs = jax.nn.softmax(xg @ w["router"], axis=-1)
  gate, idx = lax.top_k(probs, K)
  gate = gate / gate.sum(-1, keepdims=True)
  local = idx - lax.axis_index("expert") * EL
  mine = (local >= 0) & (local < EL)
  key = jnp.where(mine, local, EL - 1).reshape(-1)
  order = jnp.argsort(key)
  rows = jnp.where(mine.reshape(-1)[order][:, None], xg[order // K], 0.0)
  sizes = jnp.bincount(key, length=EL).astype(jnp.int32)
  out = lax.ragged_dot(jax.nn.silu(lax.ragged_dot(rows, w["wi"], sizes)), w["wo"], sizes)
  y = jnp.einsum("nkd,nk->nd", out[jnp.argsort(order)].reshape(-1, K, D), gate)
  return lax.psum_scatter(y, "expert", tiled=True)


def bwd_of(f):
  """A chunk backward's body of `f`: (w, x, dy) -> (dx, each device's dW on a leading axis)."""

  def bwd(w, x, dy):
    _, vjp = jax.vjp(f, w, x)
    dw, dx = vjp(dy)
    return dx, jax.tree.map(lambda g: g[None], dw)

  return bwd


def bwda_mlp(w, x, dy, gacc):
  """The MLP's accumulating backward: (w, x, dy, gacc) -> (dx, gacc + dW)."""
  _, vjp = jax.vjp(mlp, w, x)
  dw, dx = vjp(dy)
  return dx, jax.tree.map(lambda a, g: a + g[None], gacc, dw)


def moe_bwd(w, x, dy):
  """The MoE's backward, the router's dW reduced over the mesh, the experts' over 'data'."""
  _, vjp = jax.vjp(moe, w, x)
  dw, dx = vjp(dy)
  reduce = {"router": ("data", "expert"), "wi": "data", "wo": "data"}
  return dx, {k: lax.psum(g, reduce[k]) for k, g in dw.items()}


@jax.custom_vjp
def rowscale(g):
  """Rows scaled by 1, 2, ...: a custom_vjp a backward calls."""
  return g * jnp.arange(1, g.shape[0] + 1, dtype=g.dtype)[:, None]


rowscale.defvjp(lambda g: (rowscale(g), None), lambda _, ct: (rowscale(ct),))


@jax.custom_vjp
def matmul(x, w):
  return x @ w


def _matmul_bwd(res, ct):
  """Both cotangents read one custom_vjp call's (rowscale) and one custom_jvp call's (relu) output."""
  x, w = res
  ct = jax.nn.relu(rowscale(ct))
  return ct @ w.T, x.T @ ct


matmul.defvjp(lambda x, w: (x @ w, (x, w)), _matmul_bwd)


def prims(jaxpr):
  return collections.Counter(e.primitive.name for e in jaxpr.eqns)


def _spec_leaves(specs):
  return jax.tree.leaves(specs, is_leaf=lambda s: isinstance(s, P))


def _program(fn, mesh, in_specs, out_specs):
  sm = jax.shard_map(fn, mesh=mesh, in_specs=tuple(in_specs), out_specs=tuple(out_specs), check_vma=False)
  return jax.jit(sm, compiler_options=testing.ynn_off())


def split_programs(body, split, mesh, flat_specs):
  """I and W as programs on the traced function's flat arguments (of specs `flat_specs`), as a pipeline runs them:
  i(*args) -> (*I's outputs, *crossing), each crossing value on a leading device axis; w(*crossing, *the arguments W
  reads) -> W's outputs.  -> (i, w, the positions of the arguments w takes after the crossing set)."""
  lead = tuple(mesh.axis_names)
  cross = [P(lead, *([None] * len(a.shape))) for a in split.crossing_avals]
  n_i, n_c = len(split.i_out), len(split.crossing)
  w_src = [body.src[k] for k in split.w_inputs]
  w_args = [s for s in w_src if isinstance(s, int)]

  def run_i(*args):
    outs = stage_ad.inline_eval(split.i_jaxpr, (), *body.operands(args))
    return (*outs[:n_i], *(c[None] for c in outs[n_i:]))

  def run_w(*args):
    kept = iter(args[n_c:])
    vals = [next(kept) if isinstance(s, int) else s.value for s in w_src]
    return tuple(stage_ad.inline_eval(split.w_jaxpr, (), *(c[0] for c in args[:n_c]), *vals))

  i = _program(run_i, mesh, flat_specs, [body.out_specs[k] for k in split.i_out] + cross)
  w = _program(run_w, mesh, cross + [flat_specs[s] for s in w_args], [body.out_specs[k] for k in split.w_out])
  return i, w, w_args


def _bits(a):
  a = np.asarray(a)
  return a.shape, a.dtype, a.tobytes()


@testing.needs_devices(4)
class SplitJaxpr(unittest.TestCase):
  """stage_ad.trace_body + split_jaxpr on the vjp of toy layers: I then W against the vjp's own program, bit for
  bit; what lands in W; the layout every trace agrees on; what stays whole; what the split refuses.  On every PR the
  routed MoE layer (test_moe), the rest scheduled_only."""

  @classmethod
  def setUpClass(cls):
    cls.mesh = Mesh(np.array(jax.devices()[:2]), ("data",))
    cls.ep_mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, EP), ("data", "expert"))

    def sds(mesh, shape, spec):
      return jax.ShapeDtypeStruct(shape, jnp.float32, sharding=NamedSharding(mesh, spec))

    m = cls.mesh
    cls.wspec = {"w1": P(), "w2": P()}
    cls.gspec = {"w1": P("data"), "w2": P("data")}
    w = {"w1": sds(m, (D, H), P()), "w2": sds(m, (H, D), P())}
    x = sds(m, (2 * B, D), P("data"))
    cls.g_av = {"w1": sds(m, (2, D, H), P("data")), "w2": sds(m, (2, H, D), P("data"))}
    # the MLP's backward: (w, x, dy) -> (dx, dW): arguments, their specs, the outputs' specs
    cls.mlp_case = ((w, x, x), (cls.wspec, P("data"), P("data")), (P("data"), cls.gspec))
    m, xs = cls.ep_mesh, P(("data", "expert"))
    wspec = {"router": P(), "wi": P("expert"), "wo": P("expert")}
    w = {"router": sds(m, (D, E), P()), "wi": sds(m, (E, D, H), wspec["wi"]), "wo": sds(m, (E, H, D), wspec["wo"])}
    x = sds(m, (4 * B, D), xs)
    cls.moe_case = ((w, x, x), (wspec, xs, xs), (xs, wspec))

  def check(self, fn, case, mesh=None, i_out=1, **trace_kw):
    """Trace `fn`'s body at `case` (avals, in_specs, out_specs), split it keeping the first `i_out` outputs in I, and
    hold I then W to fn's own program bit for bit -> (body, split, flat args, I's program)."""
    mesh = mesh or self.mesh
    avals, in_specs, out_specs = case
    body = stage_ad.trace_body(fn, avals, mesh, in_specs, out_specs, **trace_kw)
    split = stage_ad.split_jaxpr(body.jaxpr, i_out)
    fin, fout = _spec_leaves(in_specs), _spec_leaves(out_specs)
    leaves, tree = jax.tree.flatten(avals)
    rng = np.random.default_rng(0)
    flat = [jax.device_put(rng.normal(size=a.shape).astype(a.dtype), a.sharding) for a in leaves]
    want = _program(lambda *a: tuple(jax.tree.leaves(fn(*jax.tree.unflatten(tree, list(a))))), mesh, fin, fout)(*flat)
    i_prog, w_prog, w_args = split_programs(body, split, mesh, fin)
    i_outs = list(i_prog(*flat))
    n_i = len(split.i_out)
    w_outs = list(w_prog(*i_outs[n_i:], *(flat[s] for s in w_args)))
    got = dict(zip(split.i_out, i_outs[:n_i])) | dict(zip(split.w_out, w_outs))
    self.assertEqual(len(got), len(want))
    for k, b in enumerate(want):
      if _bits(got[k]) != _bits(b):
        diff = np.max(np.abs(np.asarray(got[k], np.float64) - np.asarray(b, np.float64)))
        self.fail(f"output {k} is not the fused program's: max |diff| {diff}")
    return body, split, flat, i_prog

  @pytest.mark.scheduled_only
  def test_mlp(self):
    """A two-layer MLP's backward, first and accumulating: W is the two dW matmuls, reading tanh(x @ w1) and its
    cotangent from I, and x, dy (and the accumulator) themselves; I never reads the accumulator."""
    body, split, _, _ = self.check(bwd_of(mlp), self.mlp_case)
    w_prims, i_prims = prims(split.w_jaxpr), prims(split.i_jaxpr)
    self.assertEqual(w_prims["dot_general"], 2)
    self.assertLessEqual(set(w_prims), MATMULS | LAYOUT)
    self.assertEqual(i_prims["dot_general"], 3)  # the recompute's x @ w1, then dy @ w2^T and dz @ w1^T
    self.assertEqual([a.shape for a in split.crossing_avals], [(B, H), (B, H)])
    self.assertEqual(split.stats["crossing_bytes"], 2 * B * H * 4)
    self.assertEqual([body.src[k] for k in split.w_inputs], [2, 3])  # x, dy (w1, w2 are 0, 1)
    self.assertEqual(split.stats["eqns"], split.stats["i_eqns"] + split.stats["w_eqns"] + 1)  # + the unused primal
    avals, in_specs, out_specs = self.mlp_case
    body, split, _, _ = self.check(bwda_mlp, (avals + (self.g_av,), in_specs + (self.gspec,), out_specs))
    w_prims = prims(split.w_jaxpr)
    self.assertEqual((w_prims["dot_general"], w_prims["add"]), (2, 2))
    self.assertLessEqual(set(w_prims), MATMULS | LAYOUT | {"add"})
    self.assertEqual([body.src[k] for k in split.w_inputs], [2, 3, 4, 5])
    self.assertFalse({4, 5} & {body.src[k] for k in split.i_inputs})

  def test_moe(self):
    """A routed MoE layer's backward over an expert-parallel mesh: W is the three dW matmuls (the router's dot, the
    experts' ragged dots) and their reductions over the mesh; the routing, the gathers and dx's collectives are I's."""
    _, split, _, _ = self.check(moe_bwd, self.moe_case, self.ep_mesh)
    w_prims = prims(split.w_jaxpr)
    self.assertLessEqual(set(w_prims), MATMULS | LAYOUT | {"psum"})
    self.assertEqual(w_prims["psum"], 3)
    weights = sorted(tuple(sorted(s)) for s in ((D, E), (EL, D, H), (EL, H, D)))  # dims sorted: up to a transpose

    def matmuls(jaxpr):
      return sorted(tuple(sorted(e.outvars[0].aval.shape)) for e in jaxpr.eqns if e.primitive.name in MATMULS)

    self.assertEqual(matmuls(split.w_jaxpr), weights)
    self.assertFalse(set(matmuls(split.i_jaxpr)) & set(weights))
    # the recompute's dispatch all_gather, then dx's transposes of the combine (an all_gather) and of the dispatch
    i_prims = prims(split.i_jaxpr)
    self.assertEqual((i_prims["all_gather"], i_prims["reduce_scatter"]), (2, 1))

  @pytest.mark.scheduled_only
  def test_same_layout_every_trace(self):
    """Two traces of one program -- as each McJAX process traces its own -- split alike: the crossing set in the
    order W first reads it, the same inputs, the same text of I and W (stage_ad.same_jaxpr: one program)."""
    m, (avals, in_specs, out_specs) = self.ep_mesh, self.moe_case
    a, b = (stage_ad.split_jaxpr(stage_ad.trace_body(moe_bwd, avals, m, in_specs, out_specs).jaxpr, 1) for _ in range(2))
    self.assertIsNot(a.crossing[0], b.crossing[0])
    self.assertEqual([str(v) for v in a.crossing_avals], [str(v) for v in b.crossing_avals])
    self.assertEqual(
        (a.i_out, a.w_out, a.i_inputs, a.w_inputs, a.stats), (b.i_out, b.w_out, b.i_inputs, b.w_inputs, b.stats)
    )
    self.assertEqual(str(a.i_jaxpr), str(b.i_jaxpr))
    self.assertEqual(str(a.w_jaxpr), str(b.w_jaxpr))
    self.assertEqual([stage_ad.same_jaxpr(a.i_jaxpr, x) for x in (b.i_jaxpr, a.w_jaxpr)], [True, False])
    self.assertEqual(
        [stage_ad.same_values(np.float32(1), x) for x in (np.ones((), np.float32), np.float64(1), np.ones(1))],
        [True, False, False],
    )
    cross = a.w_jaxpr.invars[: len(a.crossing)]
    first = list(dict.fromkeys(v for e in a.w_jaxpr.eqns for v in e.invars if any(v is c for c in cross)))
    self.assertEqual(first, cross)

  @pytest.mark.scheduled_only
  def test_custom_calls_kept_whole(self):
    """A custom_vjp call and a custom_jvp call of the backward (the custom bwd rule's rowscale and relu) stay one
    equation each, in I, which computes dx from their outputs; W reads their output across."""
    body, split, _, _ = self.check(bwd_of(lambda w, x: jnp.tanh(matmul(x, w["w1"])) @ w["w2"]), self.mlp_case)
    self.assertEqual((prims(body.jaxpr)["custom_vjp_call"], prims(body.jaxpr)["custom_jvp_call"]), (1, 1))
    self.assertEqual((prims(split.i_jaxpr)["custom_vjp_call"], prims(split.i_jaxpr)["custom_jvp_call"]), (1, 1))
    self.assertEqual(prims(split.w_jaxpr)["dot_general"], 2)
    self.assertLessEqual(set(prims(split.w_jaxpr)), MATMULS | LAYOUT)
    (jvp,) = [e for e in split.i_jaxpr.eqns if e.primitive.name == "custom_jvp_call"]
    self.assertIn(jvp.outvars[0], split.crossing)

  @pytest.mark.scheduled_only
  def test_remat_inlined_with_its_barrier(self):
    """jax.checkpoint's backward is one remat equation computing dx and every dW: kept whole, W would compute
    nothing.  Inlined, its operands pass one optimization barrier as in its lowering -- I's program keeps it -- and W
    is the two dW matmuls; without prevent_cse there is no barrier."""
    fn, (avals, in_specs, out_specs) = bwd_of(jax.checkpoint(mlp)), self.mlp_case
    whole = stage_ad.trace_body(fn, avals, self.mesh, in_specs, out_specs, inline_remat=False)
    (remat,) = [e for e in whole.jaxpr.eqns if e.primitive.name == "remat2"]
    self.assertTrue(remat.params["differentiated"] and remat.params["prevent_cse"])
    with self.assertRaisesRegex(ValueError, "W would compute nothing"):
      stage_ad.split_jaxpr(whole.jaxpr, 1)
    body, split, flat, i_prog = self.check(fn, self.mlp_case)
    self.assertNotIn("remat2", prims(body.jaxpr))
    (barrier,) = [e for e in body.jaxpr.eqns if e.primitive.name == "optimization_barrier"]
    self.assertEqual(len(barrier.invars), len(remat.invars))
    self.assertEqual(prims(split.i_jaxpr)["optimization_barrier"], 1)
    self.assertIn("optimization_barrier", i_prog.lower(*flat).as_text())
    self.assertEqual(prims(split.w_jaxpr)["dot_general"], 2)
    self.assertLessEqual(set(prims(split.w_jaxpr)), MATMULS | LAYOUT)
    lean = stage_ad.trace_body(bwd_of(jax.checkpoint(mlp, prevent_cse=False)), avals, self.mesh, in_specs, out_specs)
    self.assertNotIn("optimization_barrier", prims(lean.jaxpr))

  @pytest.mark.scheduled_only
  def test_hoisted_constants(self):
    """A numpy table the body closes over (shard_map hoists it) and an array a jitted helper closes over (hoisted
    once the jit is inlined) come in as Consts."""
    row = np.linspace(0.5, 1.5, H, dtype=np.float32)
    col = jnp.linspace(1.0, 2.0, H, dtype=jnp.float32)
    scale = jax.jit(lambda h: h * col)
    body, _, _, _ = self.check(bwd_of(lambda w, x: jnp.tanh(scale(x @ w["w1"]) * row) @ w["w2"]), self.mlp_case)
    self.assertNotIn("jit", prims(body.jaxpr))
    consts = sorted(np.asarray(s.value).tobytes() for s in body.src if isinstance(s, stage_ad.Const))
    self.assertEqual(consts, sorted([row.tobytes(), np.asarray(col).tobytes()]))
    self.assertEqual(sorted(s for s in body.src if isinstance(s, int)), [0, 1, 2, 3])

  @pytest.mark.scheduled_only
  def test_refuses_effects(self):
    """An effect but a collective's named axis -- here a debug callback nothing reads -- would run at another time
    in the schedule, or not at all."""

    def noisy(w, x, dy):
      dx, dw = bwd_of(mlp)(w, x, dy)
      jax.debug.callback(lambda *_: None, dx)
      return dx, dw

    avals, in_specs, out_specs = self.mlp_case
    body = stage_ad.trace_body(noisy, avals, self.mesh, in_specs, out_specs)
    with self.assertRaisesRegex(ValueError, "effects"):
      stage_ad.split_jaxpr(body.jaxpr, 1)

  @pytest.mark.scheduled_only
  def test_refuses_nothing_to_split(self):
    """No output kept or none deferred, a position out of range, or a W that would compute nothing: every
    deferred output one of I's values (dx reads both dW; W would only put them on their leading axis) or an input
    passed through."""
    avals, in_specs, out_specs = self.mlp_case
    body = stage_ad.trace_body(bwd_of(mlp), avals, self.mesh, in_specs, out_specs)
    for i_out in ((), (0, 1, 2), (0, 3)):
      with self.assertRaisesRegex(ValueError, "split_jaxpr"):
        stage_ad.split_jaxpr(body.jaxpr, i_out)

    def tied(w, x, dy):
      dx, dw = bwd_of(mlp)(w, x, dy)
      return dx * (jnp.sum(dw["w1"]) + jnp.sum(dw["w2"])), dw

    def passthrough(w, x, dy, gacc):
      return bwd_of(mlp)(w, x, dy)[0], gacc

    cases = ((tied, self.mlp_case), (passthrough, (avals + (self.g_av,), in_specs + (self.gspec,), out_specs)))
    for fn, (av, i_specs, o_specs) in cases:
      body = stage_ad.trace_body(fn, av, self.mesh, i_specs, o_specs)
      with self.assertRaisesRegex(ValueError, "W would compute nothing"):
        stage_ad.split_jaxpr(body.jaxpr, 1)


if __name__ == "__main__":
  unittest.main()
