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

"""Tests for `maxtext.experimental.mpmd.jax.sharding_mode`: the mode a helper sees wherever it can
run, what every helper emits in every mode, and that the policy is the
default one.

Two properties of comparing explicit with auto sharding rest on this module,
and each has tests here:

  * Auto is today's program.  Every helper's auto and none branch traces to
    the same jaxpr as the expression it replaces (`TodaysExpressionTest`),
    and the model's matmuls add no constraint (`LoweringTest`): only
    `replicate` and `manual` change the auto program, on purpose.
  * The mode is right in every place model code runs: with no mesh, on a stage
    mesh of either kind, in a shard_map body (`ContextTest`), and `manual()`
    steps aside inside an existing shard_map (`ManualTest`, where the real
    tokamax splash kernel runs when tokamax is installed).

A few seconds on 8 fake CPU devices:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/sharding_mode_test.py
"""

import re
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax import sharding_mode as mode

# Imported at collection, as ds_ep_test does.
try:
  import tokamax  # noqa: F401

  HAVE_TOKAMAX = True
except Exception:  # pragma: no cover
  HAVE_TOKAMAX = False

AUTO, EXPLICIT = AxisType.Auto, AxisType.Explicit
DIMS = (((1,), (0,)), ((), ()))  # [b, k] x [k, n] -> [b, n]


def _mesh(kind, n=2, names=("fsdp",), shape=None):
  devs = np.array(jax.devices()[:n]).reshape(shape or (n,))
  return Mesh(devs, names, axis_types=(kind,) * len(names))


def _put(mesh, x, spec):
  return jax.device_put(x, NamedSharding(mesh, spec))


def _in(mesh, f):
  """`f`, traced with `mesh` (None: no mesh) in context, entered inside the trace."""

  def g(*args):
    with mode.trace_context(None if mesh is None else mesh.abstract_mesh):
      return f(*args)

  return g


def _constraints(lowered_text):
  """Sharding constraints in lowered StableHLO, Shardy's or GSPMD's spelling."""
  return re.findall(r"sdy\.sharding_constraint[^\n]*|@Sharding\b[^\n]*", lowered_text)


class PolicyTest(unittest.TestCase):

  def test_the_default_is_the_only_policy(self):
    """No knob is left: every ShardPolicy is the default one, equal and
    hashable, so a config that holds one stays a static argument."""
    self.assertEqual(mode.ShardPolicy(), mode.ShardPolicy())
    self.assertEqual(hash(mode.ShardPolicy()), hash(mode.ShardPolicy()))
    self.assertEqual(vars(mode.ShardPolicy()), {})
    with self.assertRaises(TypeError):
      mode.ShardPolicy(weights="out")


class ContextTest(unittest.TestCase):

  def test_no_mesh_is_none(self):
    self.assertEqual(mode.context_mode(), "none")
    self.assertEqual(mode.batch_axes(), ())
    self.assertEqual(mode.act_spec(3), P(None, None, None))
    with mode.trace_context(None):  # None: no context at all
      self.assertEqual(mode.context_mode(), "none")

  def test_stage_meshes(self):
    for kind, want in ((AUTO, "auto"), (EXPLICIT, "explicit")):
      mesh, seen = _mesh(kind), {}

      def f(x):
        seen["mode"], seen["axes"] = mode.context_mode(), mode.batch_axes()
        seen["act"] = (mode.act_spec(3), mode.act_spec(2, batch_dim=1))
        seen["0-d"] = mode.act_spec(0)  # no dim to split
        return x

      jax.jit(_in(mesh, f))(jnp.ones(4))
      with self.subTest(kind=want):
        self.assertEqual(
            seen, {"mode": want, "axes": ("fsdp",), "act": (P("fsdp", None, None), P(None, "fsdp")), "0-d": P()}
        )
      with mode.trace_context(mesh):  # a concrete Mesh is accepted too
        self.assertEqual(mode.context_mode(), want)
    self.assertEqual(mode.context_mode(), "none")

  def test_two_axis_mesh_puts_the_batch_on_both(self):
    mesh = _mesh(EXPLICIT, 4, ("a", "b"), (2, 2))
    with mode.trace_context(mesh.abstract_mesh):
      self.assertEqual(mode.batch_axes(), ("a", "b"))
      self.assertEqual(mode.act_spec(2), P(("a", "b"), None))

  def test_shard_map_body_is_manual(self):
    for kind in (AUTO, EXPLICIT):
      mesh, seen = _mesh(kind), {}

      def body(a):
        seen["mode"], seen["axes"] = mode.context_mode(), mode.batch_axes()
        seen["act"], seen["aval"] = mode.act_spec(2), mode.aval_mode(a)
        return a

      def f(x):
        return jax.shard_map(
            body, mesh=jax.sharding.get_abstract_mesh(), in_specs=P("fsdp"), out_specs=P("fsdp"), check_vma=False
        )(x)

      jax.jit(_in(mesh, f))(_put(mesh, jnp.ones((4, 2)), P("fsdp", None)))
      with self.subTest(kind=str(kind)):
        self.assertEqual(seen, {"mode": "manual", "axes": (), "act": P(None, None), "aval": "manual"})

  def test_mixed_axis_types_are_an_error(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("a", "b"), axis_types=(EXPLICIT, AUTO))
    with mode.trace_context(mesh.abstract_mesh):
      with self.assertRaisesRegex(ValueError, "mixes axis types"):
        mode.context_mode()

  def test_aval_mode(self):
    me, ma = _mesh(EXPLICIT), _mesh(AUTO)
    xe = _put(me, jnp.ones((4, 2)), P("fsdp", None))
    xa = _put(ma, jnp.ones((4, 2)), P("fsdp", None))
    typed = jax.ShapeDtypeStruct((4, 2), jnp.float32, sharding=NamedSharding(me, P("fsdp", None)))
    self.assertEqual(
        [mode.aval_mode(v) for v in (xe, xa, typed, jax.ShapeDtypeStruct((4,), jnp.float32), np.ones(3), 1.0)],
        ["explicit", "auto", "explicit", "none", "none", "none"],
    )
    self.assertEqual(mode.bwd_out_sharding(xe), jax.typeof(xe).sharding)
    self.assertIsNone(mode.bwd_out_sharding(xa))
    self.assertIsNone(mode.bwd_out_sharding(np.ones(3)))


class NamedTest(unittest.TestCase):

  def test_without_a_mesh_it_raises(self):
    with self.assertRaisesRegex(ValueError, "needs a mesh in context"):
      mode.named(P())

  def test_the_abstract_mesh_is_enough(self):
    """No `jax.set_mesh` anywhere: the context from `trace_context` alone
    types an output-dim FSDP matmul, which needs an out_sharding."""
    mesh = _mesh(EXPLICIT)
    x = _put(mesh, jnp.ones((4, 8)), P("fsdp", None))
    w = _put(mesh, jnp.ones((8, 6)), P(None, "fsdp"))
    y = jax.jit(_in(mesh, lambda x, w: mode.einsum("bk,kn->bn", x, w, out_spec=P("fsdp", None))))(x, w)
    self.assertEqual(str(jax.typeof(y)), "float32[4@fsdp,6]")
    np.testing.assert_array_equal(y, np.full((4, 6), 8.0))


class TodaysExpressionTest(unittest.TestCase):
  """Under none and auto, a helper IS the expression the model wrote before.

  Compared as jaxprs, so nothing downstream (lowering, XLA) can tell the two
  apart.  Under auto both sides are traced with the stage mesh in context, as
  a caller's program traces them.
  """

  def setUp(self):
    # numpy, not jax.random: only the traces are compared, nothing runs.
    self.mesh = _mesh(AUTO)
    rng = np.random.default_rng(0)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32)
    self.x, self.w, self.table = f32(4, 8), f32(8, 6), f32(16, 8)
    self.y3, self.scale = f32(4, 3, 8), f32(8)
    self.ids = rng.integers(0, 16, (4, 3)).astype(np.int32)

  def assertSameJaxpr(self, mesh, got, want, *args):
    if mesh is not None:
      args = jax.tree.map(lambda a: _put(mesh, a, P()), args)
    self.assertEqual(str(jax.make_jaxpr(_in(mesh, got))(*args)), str(jax.make_jaxpr(_in(mesh, want))(*args)))

  def test_helpers(self):
    hi, f32 = jax.lax.Precision.HIGHEST, jnp.float32
    act3 = P("fsdp", None, None)
    cases = {
        "einsum (RMSNorm scale)": (
            lambda y, s: mode.einsum("...k,k->...k", y, s, out_spec=act3),
            lambda y, s: jnp.einsum("...k,k->...k", y, s),
            (self.y3, self.scale),
        ),
        "einsum precision, preferred": (
            lambda a, b: mode.einsum("bk,kn->bn", a, b, out_spec=P("fsdp"), precision=hi, preferred_element_type=f32),
            lambda a, b: jnp.einsum("bk,kn->bn", a, b, precision=hi, preferred_element_type=f32),
            (self.x, self.w),
        ),
        "dot_general (dense_general)": (
            lambda a, b: mode.dot_general(a, b, DIMS, out_spec=P("fsdp"), precision=jax.lax.Precision("default")),
            lambda a, b: jax.lax.dot_general(a, b, DIMS, precision=jax.lax.Precision("default")),
            (self.x, self.w),
        ),
        "take == table[ids]": (lambda t, i: mode.take(t, i, out_spec=act3), lambda t, i: t[i], (self.table, self.ids)),
        "take == .at[ids].get() (embed_lookup)": (
            lambda t, i: mode.take(t, i, out_spec=act3),
            lambda t, i: t.at[i].get(),
            (self.table, self.ids),
        ),
        "bincount": (
            lambda i: mode.bincount(i.reshape(-1), length=16),
            lambda i: jnp.bincount(i.reshape(-1), length=16),
            (self.ids,),
        ),
        "set_at (dense-MoE scatter)": (
            lambda y, i: mode.set_at(
                jnp.zeros((4, 3, 16)), (jnp.arange(4)[:, None], jnp.arange(3)[None, :], i), y[..., 0], out_spec=act3
            ),
            lambda y, i: jnp.zeros((4, 3, 16)).at[jnp.arange(4)[:, None], jnp.arange(3)[None, :], i].set(y[..., 0]),
            (self.y3, self.ids),
        ),
        "retype_like": (lambda a, b: mode.retype_like(a, b) * 2, lambda a, b: a * 2, (self.x, self.x)),
        "zeros": (
            lambda a: a + mode.zeros((4, 8), f32, P("fsdp", None)),
            lambda a: a + jnp.zeros((4, 8), f32),
            (self.x,),
        ),
        "retype": (lambda a: mode.retype(a, P(None, "fsdp")) * 2, lambda a: a * 2, (self.x,)),
    }
    for where, mesh in (("none", None), ("auto", self.mesh)):
      for name, (got, want, args) in cases.items():
        with self.subTest(mode=where, helper=name):
          self.assertSameJaxpr(mesh, got, want, *args)
      with self.subTest(mode=where, helper="out_sharding"):
        with mode.trace_context(None if mesh is None else mesh.abstract_mesh):
          self.assertIsNone(mode.out_sharding(P("fsdp")))

  def test_no_mesh_leaves_strategy_and_regions_alone(self):
    """`replicate` and `manual` change the program under auto (on purpose:
    `ManualTest`); without a mesh they do nothing."""
    self.assertSameJaxpr(None, lambda a: mode.replicate(a) * 2, lambda a: a * 2, self.x)
    self.assertIs(mode.manual(jnp.tanh, in_specs=P(), out_specs=P(), name="n"), jnp.tanh)


class CustomVjpHelperTest(unittest.TestCase):
  """The helpers' explicit branches: what they type, and when they emit
  nothing."""

  def test_zeros_replicate_bincount(self):
    """A typed scan carry, a replicated copy, and counts of sharded ids."""
    mesh = _mesh(EXPLICIT)
    x = _put(mesh, jnp.arange(8.0).reshape(4, 2), P("fsdp", None))
    ids = _put(mesh, np.array([1, 0, 3, 1], np.int32), P("fsdp"))

    def f(a, i):
      return (a + mode.zeros((4, 2), jnp.float32, P("fsdp", None)), mode.replicate(a), mode.bincount(i, length=4))

    z, r, n = jax.jit(_in(mesh, f))(x, ids)
    self.assertEqual([str(jax.typeof(v)) for v in (z, r, n)], ["float32[4@fsdp,2]", "float32[4,2]", "int32[4]"])
    np.testing.assert_array_equal(z, np.arange(8.0).reshape(4, 2))
    np.testing.assert_array_equal(r, np.arange(8.0).reshape(4, 2))
    np.testing.assert_array_equal(n, [1, 2, 0, 1])

  def test_retype_like_takes_the_type_of_its_witness(self):
    mesh = _mesh(EXPLICIT)
    x = _put(mesh, jnp.arange(8.0).reshape(4, 2), P("fsdp", None))
    rep = _put(mesh, jnp.zeros((4, 2)), P())
    # A reduced witness (one copy promised equal on every device): the
    # cotangent type, unreduced, never its own.
    reduced = jax.jit(_in(mesh, lambda w: jax.sharding.reshard(w, mode.named(P(None, None, reduced={"fsdp"})))))(rep)
    for like, want, n_reshard in (
        (rep, "float32[4,2]", 1),
        (x, "float32[4@fsdp,2]", 0),
        (np.zeros((4, 2)), "float32[4@fsdp,2]", 0),
        (reduced, "float32[4,2]{U:fsdp}", 1),
    ):

      def f(a, w):
        return mode.retype_like(a, w)

      with self.subTest(like=str(jax.typeof(like))):
        # No context at all: the witness's aval alone decides, as it must in
        # a lazily traced bwd rule.
        self.assertEqual(str(jax.typeof(jax.jit(f)(x, like))), want)
        prims = [e.primitive.name for e in jax.make_jaxpr(f)(x, like).eqns]
        self.assertEqual(prims.count("reshard"), n_reshard, prims)

  def test_set_at_types_a_scatter_of_sharded_indices(self):
    """The dense MoE's routing-weight scatter (deepseek.moe._experts_dense), its indices built as the model builds
    them: a batch and a sequence arange that broadcast against the batch-sharded top-k expert ids."""
    # Not index arrays of the ids' own shape, replicated beside them: jax 0.12.1 types this scatter itself (0.12.0.1
    # and before ran it in an auto_axes island) and gives index arrays one sharding only where it broadcasts them, so
    # it refuses those ("All operands should have the same sharding").
    mesh = _mesh(EXPLICIT)
    b, s, e = 4, 3, 8
    ids_np = np.argsort(np.random.default_rng(0).random((b, s, e)), axis=-1)[..., :2].astype(np.int32)  # top 2 of 8
    w_np = np.arange(1.0, 1.0 + b * s * 2, dtype=np.float32).reshape(b, s, 2)
    ids = _put(mesh, ids_np, P("fsdp", None, None))
    w = _put(mesh, w_np, P("fsdp", None, None))

    def f(i, v):
      batch_idx = jnp.arange(b)[:, None, None]
      seq_idx = jnp.arange(s)[None, :, None]
      return mode.set_at(jnp.zeros((b, s, e)), (batch_idx, seq_idx, i), v, out_spec=mode.act_spec(3))

    y = jax.jit(_in(mesh, f))(ids, w)
    self.assertEqual(str(jax.typeof(y)), "float32[4@fsdp,3,8]")
    want = np.zeros((b, s, e), np.float32)
    want[np.arange(b)[:, None, None], np.arange(s)[None, :, None], ids_np] = w_np
    np.testing.assert_array_equal(y, want)

  def test_take_types_a_gather_of_sharded_indices(self):
    """The embedding lookup (a vocab-sharded table) and the RoPE gather (a
    replicated one) with batch-sharded ids, untypeable as `table[ids]`: the
    rows typed `out_spec`, and the gradient the table's scatter-add, typed
    as the table."""
    mesh = _mesh(EXPLICIT)
    ids_np = np.array([[0, 15, 3], [7, 7, 1]], np.int32)
    ids = _put(mesh, ids_np, P("fsdp", None))
    table_np = np.arange(128.0, dtype=np.float32).reshape(16, 8)
    c = np.arange(48.0, dtype=np.float32).reshape(2, 3, 8)
    want = np.zeros_like(table_np)
    np.add.at(want, ids_np, c)

    def take(t, i):
      return mode.take(t, i, out_spec=mode.act_spec(3))

    for spec in (P("fsdp", None), P()):
      with self.subTest(table=str(spec)):
        table = _put(mesh, table_np, spec)
        y = jax.jit(_in(mesh, take))(table, ids)
        g = jax.jit(_in(mesh, jax.grad(lambda t, i: jnp.sum(take(t, i) * c))))(table, ids)
        self.assertEqual(str(jax.typeof(y)), "float32[2@fsdp,3,8]")
        np.testing.assert_array_equal(y, table_np[ids_np])
        self.assertEqual(jax.typeof(g), jax.typeof(table))
        np.testing.assert_array_equal(g, want)


class ManualTest(unittest.TestCase):

  def test_shard_map_over_the_batch_axes_in_a_named_scope(self):
    for kind in (AUTO, EXPLICIT):
      mesh, seen = _mesh(kind), {}

      def body(q):
        seen["block"], seen["mode"] = q.shape, mode.context_mode()
        return q * 2

      def f(q):
        return mode.manual(body, in_specs=(P("fsdp", None),), out_specs=P("fsdp", None), name="attn_core")(q)

      # Under explicit the input arrives typed otherwise and manual() retypes
      # it to in_specs first: a shard_map refuses a mismatched input type.
      spec = P(None, "fsdp") if kind == EXPLICIT else P("fsdp", None)
      q = _put(mesh, jnp.arange(32.0).reshape(4, 8), spec)
      g = jax.jit(_in(mesh, f))
      y = g(q)
      with self.subTest(kind=str(kind)):
        self.assertEqual(seen, {"block": (2, 8), "mode": "manual"})
        np.testing.assert_array_equal(y, 2 * np.arange(32.0).reshape(4, 8))
        prims = [e.primitive.name for e in jax.make_jaxpr(_in(mesh, f))(q).jaxpr.eqns]
        self.assertEqual(prims.count("shard_map"), 1, prims)
        self.assertIn("attn_core", g.lower(q).compile().as_text())
        if kind == EXPLICIT:
          self.assertEqual(str(jax.typeof(y)), "float32[4@fsdp,8]")

  def test_identity_inside_a_shard_map(self):
    """A second shard_map over the same axes fails in every jax we run, and
    ds_ep calls the model from inside its own: manual() must step aside."""
    mesh, seen = _mesh(AUTO), {}

    def body(q):
      inner = mode.manual(jnp.tanh, in_specs=P("fsdp"), out_specs=P("fsdp"), name="inner")
      seen["identity"] = inner is jnp.tanh
      return inner(q)

    jax.jit(lambda q: jax.shard_map(body, mesh=mesh, in_specs=P("fsdp"), out_specs=P("fsdp"), check_vma=False)(q))(
        _put(mesh, jnp.ones((4, 2)), P("fsdp", None))
    )
    self.assertTrue(seen["identity"])

  def test_a_none_in_spec_is_refused_in_every_mode(self):
    """shard_map reads None as "close over the whole array": the full array
    under auto, an error for a sharded operand under explicit."""
    for where, mesh in (("none", None), ("auto", _mesh(AUTO)), ("explicit", _mesh(EXPLICIT))):
      with self.subTest(mode=where):
        with mode.trace_context(None if mesh is None else mesh.abstract_mesh):
          with self.assertRaisesRegex(ValueError, r"P\(\) is replicated"):
            mode.manual(jnp.add, in_specs=(P("fsdp"), None), out_specs=P("fsdp"), name="n")

  @unittest.skipUnless(HAVE_TOKAMAX, "needs tokamax")
  def test_tokamax_splash_inside_a_shard_map(self):
    """The real kernel (interpret mode on CPU) through `attention_op` from
    inside a caller's shard_map, as ds_ep calls `mla`: no shard_map of its
    own, against the portable reference attention."""
    from maxtext.experimental.mpmd.jax.deepseek import attention
    from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig

    cfg = DeepSeekConfig(
        attention_kernel="tokamax_splash",
        num_query_heads=1,
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        dtype=jnp.float32,
    )
    mesh, b = _mesh(AUTO), P("fsdp")
    rng = np.random.default_rng(1)  # one head, one block: interpret mode
    q, kk, v = (rng.normal(size=(2, 128, 1, 128)).astype(np.float32) for _ in range(3))
    seg = np.ones((2, 128), np.int32)
    ref = attention.reference_attention(q, kk, v, seg)

    def op(*a):
      return attention.attention_op(*a, cfg)

    args = [_put(mesh, a, b) for a in (q, kk, v, seg)]
    traced = jax.jit(
        lambda *a: jax.shard_map(op, mesh=mesh, in_specs=(b, b, b, b), out_specs=b, check_vma=False)(*a)
    ).trace(*args)
    self.assertEqual(str(traced.jaxpr).count("shard_map"), 1)
    out = traced.lower().compile()(*args)
    err = float(jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref)))
    self.assertLess(err, 1e-4)


class LoweringTest(unittest.TestCase):
  """Constraints in the lowered program: auto at the default policy adds none."""

  def test_auto_free_emits_no_constraint(self):
    """Two FSDP matmuls through the helpers, as `dense_general` writes them:
    the cast weight, the result typed like the activation (a type auto does
    not have, so nothing)."""

    def f(x, w1, w2):
      h = jnp.tanh(mode.dot_general(x, jnp.asarray(w1, jnp.float32), DIMS, out_spec=mode.act_spec(2), precision=None))
      return mode.dot_general(h, jnp.asarray(w2, jnp.float32), DIMS, out_spec=mode.act_spec(2), precision=None)

    mesh = _mesh(AUTO)
    args = (
        _put(mesh, jnp.ones((4, 8)), P("fsdp", None)),
        _put(mesh, jnp.ones((8, 16)), P("fsdp", None)),
        _put(mesh, jnp.ones((16, 8)), P("fsdp", None)),
    )
    self.assertEqual(_constraints(jax.jit(_in(mesh, f)).lower(*args).as_text()), [])


if __name__ == "__main__":
  unittest.main()
