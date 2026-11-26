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

"""The Qwen3.5 port under explicit sharding, and the knobs it gained.

`src/maxtext/experimental/mpmd/jax/qwen35/*` shards through `src/maxtext/experimental/mpmd/jax/qwen35/_ops.py`, whose
bodies go
through `maxtext.experimental.mpmd.jax.sharding_mode` (as the DeepSeek port's do) and read
`config.gdn_core`.  Under auto that is part 1's program, which the
canonical-HLO A/B against a73a218 and `qwen35_model_test` pin numerically.
This file pins what the conversion adds:

  * OpsTodaysExpressionTest: under none and auto every `_ops` body traces to
    the jaxpr of the plain expression it replaces -- so auto cannot be
    anything but part 1's program.
  * OpsCallSiteTest: `_ops` reaches `reshard`, `with_sharding_constraint`,
    `shard_map` and `auto_axes` only through `mode`, and every
    `out_sharding=` it writes is one `mode` computed.
  * CustomVjpTest: the GDN inverse's custom VJP (E1) on an Explicit mesh
    with batch-sharded primals, both forwards (block doubling, the solve):
    the unsharded values, bit for bit, with the cotangent typed like its
    primal.
  * GdnLayerTest: one Gated DeltaNet layer on a 2-device Explicit mesh, its
    core typed op by op (E1-E4), against the unsharded layer: padding, and
    packed documents (gdn_packing's segmented conv and per-document rule).
  * ConfigTest: the new knobs are refused at construction when misspelt.
  * ExplicitModelTest (scheduled_only): the monolithic model on a
    2-device Explicit mesh -- and its auto twin -- with both GDN core
    placements, ragged and dense, against the single-device model at the
    measured fp32 floor.

The default set takes a few seconds on 4 fake CPU devices:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/qwen35_explicit_test.py
"""

import ast
import pathlib
import re
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import pytest

from tests.unit.experimental.mpmd.jax.qwen35_model_test import code_lines, max_rel, realize, small
from maxtext.experimental.mpmd.jax.qwen35 import _ops
from maxtext.experimental.mpmd.jax.qwen35 import config as qc
from maxtext.experimental.mpmd.jax.qwen35 import gdn
from maxtext.experimental.mpmd.jax.qwen35 import init as qinit
from maxtext.experimental.mpmd.jax.qwen35 import model as qm
from maxtext.experimental.mpmd.jax.qwen35 import train as qt
from maxtext.experimental.mpmd.jax import sharding_mode as mode
from maxtext.experimental.mpmd.jax.sharding_mode import ShardPolicy

AUTO, EXPLICIT = AxisType.Auto, AxisType.Explicit
OPS = pathlib.Path(_ops.__file__)


def _mesh(kind, n=2):
  return Mesh(np.array(jax.devices()[:n]), ("fsdp",), axis_types=(kind,))


def _put(mesh, x, spec):
  return jax.device_put(x, NamedSharding(mesh, spec))


def _in(mesh, f):
  """`f`, traced with `mesh` (None: no mesh) in context, entered inside the trace."""

  def g(*args):
    with mode.trace_context(None if mesh is None else mesh.abstract_mesh):
      return f(*args)

  return g


def _tree_relerr(got, ref):
  worst, where = 0.0, ""
  for (path, a), r in zip(jax.tree_util.tree_flatten_with_path(got)[0], jax.tree.leaves(ref)):
    e = max_rel(a, r)
    if e > worst:
      worst, where = e, jax.tree_util.keystr(path)
  return worst, where


def _layer(tree):
  """A one-block stacked tree -> one layer's (the block axis dropped)."""
  return jax.tree.map(lambda a: a[0], tree)


def _layer_specs(stacked_specs):
  return jax.tree.map(lambda s: P(*s[1:]), stacked_specs, is_leaf=lambda s: isinstance(s, P))


def _fsdp_spec(avals, n_shards, axis="fsdp"):
  """A PartitionSpec tree that stores every leaf of `avals` FSDP-sharded: on its
  biggest axis that `n_shards` divides, never a `decoder/layers` leaf's block
  axis 0 (the scan's), replicated where no axis qualifies.  Any axis gives the
  same values; this one is the most likely to divide evenly."""
  leaves, treedef = jax.tree_util.tree_flatten_with_path(avals)
  specs = []
  for path, a in leaves:
    shape = tuple(int(d) for d in a.shape)
    start = 1 if qinit.is_stacked(path) else 0
    cand = [d for d in range(start, len(shape)) if shape[d] >= n_shards and shape[d] % n_shards == 0]
    if n_shards <= 1 or not cand:
      specs.append(P())
      continue
    d = max(cand, key=lambda i: (shape[i], -i))
    specs.append(P(*([None] * d + [axis] + [None] * (len(shape) - d - 1))))
  return jax.tree.unflatten(treedef, specs)


# --------------------------------------------------------- today's program
class OpsTodaysExpressionTest(unittest.TestCase):
  """Under none and auto (the stage mesh in context, as a caller entering
  `manual`'s regions traces it), every `_ops` body is the expression part 1
  wrote, as a jaxpr; auto with no context at all is the none case."""

  def setUp(self):
    rng = np.random.default_rng(0)
    f32 = lambda *s: rng.standard_normal(s).astype(np.float32)
    self.cfg = small()
    self.mesh = _mesh(AUTO)
    self.v = f32(2, 5, 2, 8)  # [B, S, Hk, Vpk * Dv]
    self.s = f32(2, 3, 2, 8, 8)  # [B, N, H, C, C]
    self.x = f32(2, 5, 32)
    self.scale = f32(32)
    self.w = f32(32, 4)
    self.conv_x, self.conv_k = f32(2, 5, 6), f32(4, 1, 6)
    self.cum, self.g = f32(2, 3, 1, 8, 8), f32(2, 3, 4, 8)  # [B, N, 1, C, C], [B, N, H, C]
    self.taps, self.w2 = f32(2, 5, 4, 6), f32(4, 6)  # [B, S, K, C], [K, C]

  def assertSameJaxpr(self, mesh, got, want, *args):
    if mesh is not None:
      args = jax.tree.map(lambda a: _put(mesh, a, P()), args)
    self.assertEqual(str(jax.make_jaxpr(_in(mesh, got))(*args)), str(jax.make_jaxpr(_in(mesh, want))(*args)))

  def test_every_body(self):
    cfg, bf16, f32 = self.cfg, jnp.bfloat16, jnp.float32
    dims = (((2,), (0,)), ((), ()))
    prec = jax.lax.Precision("default")
    conv_kw = dict(
        window_strides=(1,),
        padding=((3, 0),),
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=6,
        precision=prec,
    )

    def norm_fp32_part1(x, scale):
      x = jnp.asarray(x, jnp.float32)
      mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
      y = x * jax.lax.rsqrt(mean2 + cfg.normalization_layer_epsilon)
      return jnp.einsum("...k,k->...k", y, jnp.asarray(scale, jnp.float32))

    def rule(q, k, v, g, b):
      return q + k + v + g[..., None] + b[..., None]

    def seg_rule(q, k, v, g, b, segment_ids):
      return rule(q, k, v, g, b) * segment_ids[..., None, None]

    cases = {
        "use_weight": (lambda w: _ops.use_weight(w, bf16, cfg), lambda w: jnp.asarray(w, bf16), (self.w,)),
        "small_weight": (lambda w: _ops.small_weight(w, bf16, cfg), lambda w: jnp.asarray(w, bf16), (self.w,)),
        "norm_fp32": (lambda x, s: _ops.norm_fp32({"scale": s}, x, cfg), norm_fp32_part1, (self.x, self.scale)),
        "dot_general (router)": (
            lambda x, w: _ops.dot_general(x, w, dims, precision=prec),
            lambda x, w: jax.lax.dot_general(x, w, dims, precision=prec),
            (self.x, self.w),
        ),
        "zeros (scan carry)": (
            lambda x: x + _ops.zeros((2, 5, 32), f32, batch_dim=0),
            lambda x: x + jnp.zeros((2, 5, 32), f32),
            (self.x,),
        ),
        "regroup": (lambda v: _ops.regroup(v, (2, 5, 4, 4)), lambda v: v.reshape((2, 5, 4, 4)), (self.v,)),
        "broadcast_like": (
            lambda s: _ops.broadcast_like(jnp.eye(8, dtype=s.dtype), s) + s,
            lambda s: jnp.broadcast_to(jnp.eye(8, dtype=s.dtype), s.shape) + s,
            (self.s,),
        ),
        "retype_like": (lambda s: _ops.retype_like(s * 2, s), lambda s: s * 2, (self.s,)),
        "conv_general_dilated": (
            lambda x, k: _ops.conv_general_dilated(x, k, **conv_kw),
            lambda x, k: jax.lax.conv_general_dilated(x, k, **conv_kw),
            (self.conv_x, self.conv_k),
        ),
        "gdn_core inline": (
            lambda *a: _ops.gdn_core(rule, *a, cfg),
            rule,
            (self.v, self.v, self.v, self.v[..., 0], self.v[..., 0]),
        ),
        "gdn_core inline, packed": (
            lambda q, k, v, g, b, s: _ops.gdn_core(seg_rule, q, k, v, g, b, cfg, segment_ids=s),
            lambda q, k, v, g, b, s: seg_rule(q, k, v, g, b, segment_ids=s),
            (self.v, self.v, self.v, self.v[..., 0], self.v[..., 0], self.v[..., 0, 0]),
        ),
        "segment_cumsum": (
            _ops.segment_cumsum,
            lambda m, g: jnp.einsum("bnhij,bnhj->bnhi", m, g, precision=jax.lax.Precision.HIGHEST),
            (self.cum, self.g),
        ),
        "conv_taps": (
            _ops.conv_taps,
            lambda t, w: jnp.einsum(
                "bskc,kc->bsc", t, w, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
            ),
            (self.taps, self.w2),
        ),
    }
    for where, mesh in (("none", None), ("auto", self.mesh)):
      for name, (got, want, args) in cases.items():
        with self.subTest(mode=where, op=name):
          self.assertSameJaxpr(mesh, got, want, *args)

  def test_strategies_act_only_with_a_mesh(self):
    """gdn_core='manual' changes the auto program on purpose (a shard_map);
    with no mesh it is the plain expression."""
    cfg = self.cfg.replace(gdn_core="manual")
    rule = lambda q, k, v, g, b: q * 2
    self.assertSameJaxpr(None, lambda *a: _ops.gdn_core(rule, *a, cfg), rule, *(self.v,) * 5)
    prims = str(
        jax.make_jaxpr(_in(self.mesh, lambda *a: _ops.gdn_core(rule, *a, cfg)))(
            *jax.tree.map(lambda a: _put(self.mesh, a, P("fsdp")), (self.v,) * 5)
        )
    )
    self.assertIn("shard_map", prims)


# -------------------------------------------------------------- call sites
class OpsCallSiteTest(unittest.TestCase):
  """`_ops` is the port's one sharding seam, and inside it every sharding
  decision is `mode`'s (the model code makes no sharding call of its own): no
  `reshard`, constraint, shard_map, island or concrete sharding of its own, and
  every `out_sharding=` is a value `mode.out_sharding` / `mode.sharding_of` /
  `mode.bwd_out_sharding` computed."""

  FORBIDDEN = (
      r"\breshard\(",
      r"with_sharding_constraint",
      r"\bauto_axes\(",
      r"\bshard_map\(",
      r"use_abstract_mesh",
      r"set_mesh\(",
      r"NamedSharding\(",
  )

  def test_no_sharding_calls_of_its_own(self):
    code = "\n".join(code_lines(OPS.read_text()))  # docstrings may name them
    for pat in self.FORBIDDEN:
      with self.subTest(pattern=pat):
        self.assertIsNone(re.search(pat, code))

  def test_out_sharding_only_from_mode(self):
    tree = ast.parse(OPS.read_text())
    seen = 0
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
      bound = {
          t.id
          for n in ast.walk(fn)
          if isinstance(n, ast.Assign)
          for t in n.targets
          if isinstance(t, ast.Name)
          and isinstance(n.value, ast.Call)
          and ast.unparse(n.value.func) in ("mode.out_sharding", "mode.sharding_of", "mode.bwd_out_sharding")
      }
      for node in ast.walk(fn):
        for kw in getattr(node, "keywords", ()):
          if kw.arg == "out_sharding":
            seen += 1
            with self.subTest(function=fn.name):
              self.assertTrue(isinstance(kw.value, ast.Name) and kw.value.id in bound, ast.unparse(kw.value))
    self.assertEqual(seen, 2)  # regroup, broadcast_like


# ------------------------------------------------------------ custom VJPs
class CustomVjpTest(unittest.TestCase):
  """`gdn.unit_lower_inv` (E1) on an Explicit 2-device mesh, the batch of
  matrices sharded: each matrix is inverted (and differentiated) on its own
  device, so the values are the unsharded ones bit for bit."""

  def setUp(self):
    if len(jax.devices()) < 2:
      self.skipTest("needs 2 devices")

  def test_unit_lower_inverse(self):
    rng = np.random.default_rng(0)
    c = 8
    s = (rng.standard_normal((4, 3, c, c)) * 0.3 * np.tril(np.ones((c, c)), -1)).astype(np.float32)
    probe = rng.standard_normal(s.shape).astype(np.float32)
    for method in qc.GDN_CHUNK_INVERSES:
      with self.subTest(method=method):

        def loss(m, method=method):
          a = gdn.unit_lower_inv(m, method)
          return jnp.sum(a * probe), a

        vg = jax.value_and_grad(loss, has_aux=True)
        (_, ref_a), ref_g = jax.jit(vg)(s)
        mesh = _mesh(EXPLICIT)
        ss = _put(mesh, s, P("fsdp"))
        (_, a), g = jax.jit(_in(mesh, vg))(ss)
        self.assertEqual(jax.typeof(a), jax.typeof(ss))
        self.assertEqual(jax.typeof(g), jax.typeof(ss))
        np.testing.assert_array_equal(a, ref_a)
        np.testing.assert_array_equal(g, ref_g)
        # strictly lower, as the primal's tangent space is
        self.assertEqual(float(np.abs(np.triu(np.asarray(g))).max()), 0.0)


# ----------------------------------------------------------------- the GDN
class GdnLayerTest(unittest.TestCase):
  """One Gated DeltaNet layer, fwd + bwd, on an Explicit 2-device mesh with
  FSDP-stored weights (each projection left sharded, its matmul typed like
  the activation), its core inline and per device."""

  def setUp(self):
    if len(jax.devices()) < 2:
      self.skipTest("needs 2 devices")

  def test_inline_core(self):
    """E1-E4: the core typed op by op.  The per-device core ('manual') is
    ExplicitModelTest's (scheduled_only)."""
    seg = np.ones((2, 20), np.int32)
    seg[1, 15:] = 0  # padding on the path
    for packing in (True, False):
      with self.subTest(gdn_packing=packing):
        self.check(small(gdn_packing=packing), seg)

  def test_packed_documents(self):
    """gdn_packing's path -- the segmented conv (both forms), the canonical
    documents, the per-document masks and cumulative decay -- typed op by op:
    two documents in a row, padding inside the other and at its end."""
    seg = np.ones((2, 20), np.int32)
    seg[0, 7:] = 2
    seg[1, 3:5] = 0
    seg[1, 15:] = 0
    for conv in qc.GDN_CONV_IMPLS:
      with self.subTest(gdn_conv_impl=conv):
        self.check(small(gdn_conv_impl=conv), seg)

  def check(self, cfg, seg):
    stacked = qm.gdn_layer_avals(cfg, 1)["attention"]
    p = _layer(realize({"a": stacked})["a"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 20, cfg.emb_dim)).astype(np.float32)

    def f(p, x, s, c):
      return jnp.sum(gdn.gated_delta_net(p, x, s, c) ** 2)

    vg = jax.value_and_grad(f, argnums=(0, 1))
    ref_l, ref_g = jax.jit(vg, static_argnums=3)(p, x, seg, cfg)
    mesh = _mesh(EXPLICIT)
    specs = _layer_specs(_fsdp_spec({"a": stacked}, 2)["a"])
    self.assertTrue(any(s != P() for s in jax.tree.leaves(specs, is_leaf=lambda s: isinstance(s, P))))
    ps = jax.tree.map(lambda a, s: _put(mesh, a, s), p, specs)
    xs, ss = _put(mesh, x, P("fsdp")), _put(mesh, seg, P("fsdp"))
    loss, (gp, gx) = jax.jit(_in(mesh, lambda p, x, s: vg(p, x, s, cfg)))(ps, xs, ss)
    self.assertLess(abs(float(loss) - float(ref_l)) / abs(float(ref_l)), 1e-6)
    err, where = _tree_relerr((gp, gx), ref_g)
    self.assertLess(err, 1e-5, where)
    self.assertEqual(jax.typeof(gx), jax.typeof(xs))
    for g, w in zip(jax.tree.leaves(gp), jax.tree.leaves(ps)):
      self.assertEqual(jax.typeof(g), jax.typeof(w))


# ------------------------------------------------------------------ config
class ConfigTest(unittest.TestCase):

  def test_new_knobs_are_checked_at_construction(self):
    self.assertEqual((small().gdn_core, small().shard), ("inline", ShardPolicy()))
    with self.assertRaisesRegex(ValueError, "gdn_core"):
      small(gdn_core="shard_map")
    with self.assertRaisesRegex(TypeError, "ShardPolicy"):
      small(shard={"weights": "out"})


# --------------------------------------------------- the model, compiled
@pytest.mark.scheduled_only
class ExplicitModelTest(unittest.TestCase):
  """`train.compute_loss` on one 2-device mesh (no pipeline) with both GDN
  core placements, ragged and dense, explicit and auto, against the
  single-device model.

  The bound is max(2e-5, 4 x floor) with the floor MEASURED: the same
  reference with the batch sharded on an Auto mesh, i.e. the fp32
  reassociation of splitting every batch reduction in two, which no
  sharded program avoids (the conv kernel's and dt_bias's gradients are
  long sums with cancellation).
  """

  def setUp(self):
    if len(jax.devices()) < 2:
      self.skipTest("needs 2 devices")

  def test_every_policy(self):
    for impl in ("ragged", "dense"):
      cfg = small(num_decoder_layers=4, moe_impl=impl)
      params = realize(qm.param_avals(cfg))
      batch = jax.tree.map(np.asarray, qt.synthetic_batch(cfg.vocab_size, 4, 20, seed=1, max_segments_per_seq=3))
      loss = lambda p, b, c: qt.compute_loss(p, b, c)[0]
      vg = jax.value_and_grad(loss)
      ref_l, ref_g = jax.jit(vg, static_argnums=2)(params, batch, cfg)
      auto = _mesh(AUTO)
      _, floor_g = jax.jit(_in(auto, lambda p, b: vg(p, b, cfg)))(_put(auto, params, P()), _put(auto, batch, P("fsdp")))
      tol = max(2e-5, 4 * _tree_relerr(floor_g, ref_g)[0])
      for shard_mode in ("explicit", "auto"):
        for core in qc.GDN_CORES:
          mesh = _mesh(EXPLICIT if shard_mode == "explicit" else AUTO)
          specs = _fsdp_spec(qm.param_avals(cfg), 2)
          ps = jax.tree.map(lambda a, s, m=mesh: _put(m, a, s), params, specs)
          bs = _put(mesh, batch, P("fsdp"))
          c = cfg.replace(gdn_core=core)
          got_l, got_g = jax.jit(_in(mesh, lambda p, b, c=c: vg(p, b, c)))(ps, bs)
          with self.subTest(moe_impl=impl, mode=shard_mode, gdn_core=core):
            self.assertLess(abs(float(got_l) - float(ref_l)) / abs(float(ref_l)), 1e-6)
            err, where = _tree_relerr(got_g, ref_g)
            self.assertLess(err, tol, where)
            for g, w in zip(jax.tree.leaves(got_g), jax.tree.leaves(ps)):
              self.assertEqual(jax.typeof(g), jax.typeof(w))


if __name__ == "__main__":
  unittest.main()
