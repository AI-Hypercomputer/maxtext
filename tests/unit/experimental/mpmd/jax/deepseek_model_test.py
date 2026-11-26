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

"""Model-level tests of the DeepSeek-V3 port (`src/maxtext/experimental/mpmd/jax/deepseek/`).

The port shards through `maxtext.experimental.mpmd.jax.sharding_mode`, which reads the
mode off the mesh in context; `ds_ep_test`, `exact_dsv3_test` and
`ds_ep_parity` hold its numerics (against the MPMD pipeline and the
monolith).  This file pins:

  * ConfigProtocolTest: `SharedLayerConfig` is exactly the config surface of
    the functions other models borrow, and they run on a config that has
    those fields and nothing else.
  * CustomVjpTest: the model's three custom VJPs on an Explicit mesh with
    sharded primals give the unsharded values -- bit for bit wherever the
    sharding splits no sum -- with cotangents typed like their primals.
  * PrecisionTest: `matmul_precision` reaches every matmul of a decoder
    layer, its backward included, so an fp32 parity run at 'highest' has
    no bf16 pass left on TPU.
  * CallSiteTest: the model reaches `reshard`, `with_sharding_constraint` and
    `out_sharding=` only through `mode` -- the one exception being custom-VJP
    rules typing results from their primals' avals.
  * QueryProjectionTest: `q_lora_rank == 0` is MaxText's plain query
    projection, and the queries reach the loss.
  * InitTest: `init.init_params` builds `model.param_avals`' tree, norm
    scales at one.

A few seconds on 4 fake CPU devices, nothing compiled but small programs:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/deepseek_model_test.py
"""

import ast
import inspect
import os
import re
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax import sharding_mode as mode
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax.deepseek import attention as ds_attention
from maxtext.experimental.mpmd.jax.deepseek import init as ds_init
from maxtext.experimental.mpmd.jax.deepseek import layers as ds_layers
from maxtext.experimental.mpmd.jax.deepseek import model as ds_model
from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe
from maxtext.experimental.mpmd.jax.deepseek import rope as ds_rope
from maxtext.experimental.mpmd.jax.deepseek import train as ds_train
from maxtext.experimental.mpmd.jax.deepseek.config import (SHARED_LAYER_FIELDS, DeepSeekConfig, SharedLayerConfig)

GB, SEQ = 8, 32


def _config(**kw):
  """Small enough for 8 CPU devices, wide enough to exercise dense + MoE."""
  base = dict(
      vocab_size=256,
      emb_dim=64,
      num_decoder_layers=8,
      first_num_dense_layers=2,
      num_query_heads=4,
      num_kv_heads=4,
      q_lora_rank=0,
      kv_lora_rank=32,
      qk_nope_head_dim=16,
      qk_rope_head_dim=16,
      v_head_dim=16,
      mlp_dim=128,
      moe_mlp_dim=64,
      num_experts=4,
      num_experts_per_tok=2,
      shared_experts=1,
      max_target_length=SEQ,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      attention_kernel="reference",
  )
  base.update(kw)
  return DeepSeekConfig(**base)


def _relerr(a, r):
  a = np.asarray(a, np.float64)
  r = np.asarray(r, np.float64)
  return float(np.max(np.abs(a - r)) / (np.max(np.abs(r)) + 1e-30))


def _mesh(kind, n=2):
  return Mesh(np.array(jax.devices()[:n]), ("fsdp",), axis_types=(kind,))


def _put(mesh, x, spec):
  return jax.device_put(x, NamedSharding(mesh, spec))


def _jit_in(mesh, f, **kw):
  """`jit(f)` traced with `mesh` in context, entered inside -- around the
  whole function, so a backward taken inside is covered."""

  def g(*args):
    with mode.trace_context(None if mesh is None else mesh.abstract_mesh):
      return f(*args)

  return jax.jit(g, **kw)


# --------------------------------------------------------- config surface
class ConfigProtocolTest(unittest.TestCase):
  """`SharedLayerConfig` is the config surface of the shared functions."""

  # The functions another model calls, plus the private helpers they reach
  # whose source reads the config.
  SHARED = (
      ds_layers.precision_of,
      ds_layers.precision_or_none,
      ds_layers.rms_norm,
      ds_layers.dense_general,
      ds_layers.embed_lookup,
      ds_layers.mlp_block,
      ds_moe.routed_experts,
      ds_moe._experts_dense,
      ds_moe._experts_ragged,
      ds_attention.attention_op,
      ds_attention._attention_kernel,
      ds_attention.dot_product_attention,
      ds_attention.tokamax_splash_attention,
      ds_model.remat_policy_for,
  )

  def test_it_lists_exactly_what_the_shared_functions_read(self):
    read = set()
    for fn in self.SHARED:
      read |= set(re.findall(r"\bconfig\.(\w+)", inspect.getsource(fn)))
    self.assertEqual(read, set(SHARED_LAYER_FIELDS))

  def test_deepseek_config_satisfies_it(self):
    self.assertIsInstance(DeepSeekConfig(), SharedLayerConfig)
    self.assertNotIsInstance(types.SimpleNamespace(dtype=jnp.float32), SharedLayerConfig)

  def test_the_shared_functions_need_nothing_else(self):
    """Traced with an object that has the protocol's fields and no others,
    they trace to exactly what the full config gives.  A trace is enough: a
    field read that the protocol misses is an AttributeError at trace
    time."""
    for impl in ("ragged", "dense"):
      full = _config(moe_impl=impl)
      bare = types.SimpleNamespace(**{f: getattr(full, f) for f in SHARED_LAYER_FIELDS})
      avals = ds_model.param_avals(full)
      layer = jax.tree.map(  # layer 0 of the MoE stack
          lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), avals["decoder"]["moe_layers"]["DeepSeekMoeBlock_0"]
      )
      tok = jax.ShapeDtypeStruct((2, SEQ), jnp.int32)

      def run(table, norm, moe_p, tokens, seg, cfg):
        x = ds_layers.embed_lookup(table, tokens, cfg)
        x = ds_layers.rms_norm(norm, x, cfg)
        gate = dict(moe_p["MoeBlock_0"]["gate"], bias=jnp.zeros(full.num_experts, jnp.float32))  # + the aux router bias
        w, idx = ds_moe.route(gate, x, full)
        y = ds_moe.routed_experts(moe_p["MoeBlock_0"], x, w, idx, cfg)
        y = y + ds_layers.mlp_block(moe_p["shared_experts"], x, cfg)
        q = y.reshape(2, SEQ, full.num_query_heads, -1)
        return ds_attention.attention_op(q, q, q, seg, cfg)

      args = (avals["token_embedder"], avals["decoder"]["decoder_norm"], layer, tok, tok)
      with self.subTest(moe_impl=impl):
        self.assertEqual(
            str(jax.make_jaxpr(lambda *a: run(*a, bare))(*args)), str(jax.make_jaxpr(lambda *a: run(*a, full))(*args))
        )


# ----------------------------------------------------------- custom VJPs
class CustomVjpTest(unittest.TestCase):
  """The model's custom VJPs on an Explicit 2-device mesh, sharded primals.

  Each against the same call on unsharded arrays.  "Typed like its primal" is
  jax's own check (it refuses a bwd rule whose cotangent is not), so the type
  asserts here mostly document; the values are the point.
  """

  def setUp(self):
    if len(jax.devices()) < 2:
      self.skipTest("needs 2 devices")
    self.mesh = _mesh(AxisType.Explicit)
    self.rng = np.random.default_rng(0)

  def _f32(self, *shape):
    return self.rng.standard_normal(shape).astype(np.float32)

  def test_grouped_matmul(self):
    """The global dispatch's case: rows replicated (the ragged axis cannot be
    sharded), the experts FSDP-sharded on their contracting or output dim.
    Both sides compile without XLA:CPU's YNNPACK fusions (testing.ynn_off): a
    YNNPACK dot sums in an order its shapes decide, so on jax 0.12.0.1
    exact_drhs's masked einsum over a device's half of the experts came out a
    few fp32 ulps off the unsharded one's (a third of the elements)."""
    m, g, k, n = 16, 4, 8, 6
    lhs, rhs, probe = self._f32(m, k), self._f32(g, k, n), self._f32(m, n)
    sizes = np.array([3, 5, 0, 8], np.int32)

    def loss(l, r, s, fp32_accum, exact_drhs):
      out = ds_moe._grouped_matmul(l, r, s, fp32_accum, exact_drhs)
      return jnp.sum(out * probe), out

    vg = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    opts = testing.ynn_off()
    for fp32_accum, exact_drhs in ((True, False), (False, False), (True, True)):
      (_, ref_out), ref_g = jax.jit(vg, static_argnums=(3, 4), compiler_options=opts)(
          lhs, rhs, sizes, fp32_accum, exact_drhs
      )
      for rhs_spec in (P(None, "fsdp", None), P(None, None, "fsdp")):
        args = (_put(self.mesh, lhs, P()), _put(self.mesh, rhs, rhs_spec), _put(self.mesh, sizes, P()))
        (_, out), (dl, dr) = _jit_in(self.mesh, vg, static_argnums=(3, 4), compiler_options=opts)(
            *args, fp32_accum, exact_drhs
        )
        with self.subTest(fp32_accum=fp32_accum, exact_drhs=exact_drhs, rhs=str(rhs_spec)):
          self.assertEqual(jax.typeof(dl), jax.typeof(args[0]))
          self.assertEqual(jax.typeof(dr), jax.typeof(args[1]))
          np.testing.assert_array_equal(dr, ref_g[1])
          # The forward contracts the experts' K dim and dlhs their N dim:
          # whichever is sharded has its sum split in two, the other is exact.
          for got, want, split in ((out, ref_out, rhs_spec[1]), (dl, ref_g[0], rhs_spec[2])):
            if split is None:
              np.testing.assert_array_equal(got, want)
            else:
              self.assertLess(_relerr(got, want), 1e-6)

  def test_grouped_matmul_takes_the_precision_in_every_matmul(self):
    """The forward and both transposes, which are written out by hand
    (PrecisionTest says why it matters), in every accumulation mode, with
    unsharded operands and on the Explicit mesh, whose transposes are
    spelled differently.  Traced, not compiled."""
    m, g, k, n = 16, 4, 8, 6
    shapes = ((m, k), (g, k, n), (g,))
    dtypes = (jnp.float32, jnp.float32, jnp.int32)
    specs = (P(), P(None, "fsdp", None), P())
    for fp32_accum, exact_drhs in ((True, False), (False, False), (True, True)):
      grad = jax.grad(
          lambda l, r, s, a=fp32_accum, e=exact_drhs: jnp.sum(
              ds_moe._grouped_matmul(l, r, s, a, e, jax.lax.Precision.HIGHEST)
          ),
          argnums=(0, 1),
      )
      for mesh in (None, self.mesh):
        args = [
            jax.ShapeDtypeStruct(sh, dt, sharding=None if mesh is None else NamedSharding(mesh, sp))
            for sh, dt, sp in zip(shapes, dtypes, specs)
        ]
        with self.subTest(fp32_accum=fp32_accum, exact_drhs=exact_drhs, mesh=mesh is not None):
          jaxpr = jax.make_jaxpr(_jit_in(mesh, grad))(*args).jaxpr
          self.assertEqual(_matmul_precisions(jaxpr), {_HIGHEST})

  def test_sort_activations(self):
    """Rows sharded, so both the gather and the inverse permutation would be
    untypeable as written; pure data movement either way, so bit for bit."""
    x, probe = self._f32(16, 8), self._f32(16, 8)
    perm = self.rng.permutation(16).astype(np.int32)

    def loss(a, p):
      out = ds_moe._sort_activations(a, p)
      return jnp.sum(out * probe), out

    vg = jax.value_and_grad(loss, has_aux=True)
    (_, ref_out), ref_g = jax.jit(vg)(x, perm)
    for x_spec, p_spec in ((P("fsdp", None), P("fsdp")), (P("fsdp", None), P()), (P(None, "fsdp"), P("fsdp"))):
      xs = _put(self.mesh, x, x_spec)
      (_, out), g = _jit_in(self.mesh, vg)(xs, _put(self.mesh, perm, p_spec))
      with self.subTest(x=str(x_spec), perm=str(p_spec)):
        self.assertEqual(jax.typeof(out), jax.typeof(xs))
        self.assertEqual(jax.typeof(g), jax.typeof(xs))
        np.testing.assert_array_equal(out, ref_out)
        np.testing.assert_array_equal(g, ref_g)

  def test_cross_entropy(self):
    """Batch-sharded logits (the model's head), vocab-sharded ones, and
    replicated logits against batch-sharded targets: the case whose
    elementwise cotangent comes out typed like the targets unless the rule
    retypes it from the logits."""
    b, s, v = 4, 8, 16
    logits = self._f32(b, s, v)
    targets = self.rng.integers(0, v, (b, s)).astype(np.int32)

    def loss(l, t):
      xent, _ = ds_train.cross_entropy_with_logits(l, jax.nn.one_hot(t, v), 1e-4)
      return jnp.sum(xent)

    ref = jax.jit(jax.grad(loss))(logits, targets)
    for l_spec, t_spec, exact in (
        (P("fsdp", None, None), P("fsdp", None), 1),
        (P(None, None, "fsdp"), P(), 0),
        (P(), P("fsdp", None), 1),
    ):
      ls = _put(self.mesh, logits, l_spec)
      g = _jit_in(self.mesh, jax.grad(loss))(ls, _put(self.mesh, targets, t_spec))
      with self.subTest(logits=str(l_spec), targets=str(t_spec)):
        self.assertEqual(jax.typeof(g), jax.typeof(ls))
        if exact:
          np.testing.assert_array_equal(g, ref)
        else:  # the softmax sum is split over vocab
          self.assertLess(_relerr(g, ref), 1e-6)


# ------------------------------------------------------- matmul precision
def _as_jaxprs(v):
  """The jaxprs in a jaxpr-equation parameter: Jaxpr, ClosedJaxpr or tuples of
  them.  Duck-typed, because jax 0.12 folded ClosedJaxpr into Jaxpr."""
  if hasattr(v, "eqns"):
    yield v
  elif hasattr(getattr(v, "jaxpr", None), "eqns"):
    yield v.jaxpr
  elif isinstance(v, (tuple, list)):
    for x in v:
      yield from _as_jaxprs(x)


def _matmul_precisions(jaxpr):
  """The set of `precision`s of the dot_generals and ragged_dot_generals in
  `jaxpr`, through every nested jaxpr."""
  out = set()
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ("dot_general", "ragged_dot_general"):
      out.add(eqn.params["precision"])
    for p in eqn.params.values():
      for sub in _as_jaxprs(p):
        out |= _matmul_precisions(sub)
  return out


_HIGHEST = (jax.lax.Precision.HIGHEST,) * 2


class PrecisionTest(unittest.TestCase):
  """`config.matmul_precision` reaches every matmul of a decoder layer, in
  the backward too.

  On TPU a DEFAULT-precision matmul of fp32 operands is one bf16 pass.  The
  grouped expert matmuls, RMSNorm's scale and the reference attention's QK
  product used to take no precision, so an fp32 parity run at 'highest'
  still rounded them -- on a v5p enough to flip top-k routing and put an
  auto-sharded run 0.12 off the monolithic model at an expert's gradient.  At
  'default' they pass None, as they always did, so today's program stands
  and `jax_default_matmul_precision` still governs them.  A trace each, no
  compile.
  """

  def _layer_precisions(self, **cfg_kw):
    cfg = _config(**cfg_kw)
    avals = ds_model.param_avals(cfg)
    layer = jax.tree.map(  # layer 0 of the MoE stack
        lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype),
        ds_model._merge(avals["decoder"]["moe_layers"], ds_model.aux_avals(cfg)["decoder"]["moe_layers"]),
    )
    x = jax.ShapeDtypeStruct((2, SEQ, cfg.emb_dim), jnp.float32)
    pos = jax.ShapeDtypeStruct((2, SEQ), jnp.int32)
    # Traced, not run: an eager rotary table is a dozen compiles.
    freqs = jax.eval_shape(lambda: ds_rope.freqs_cis_for(cfg, SEQ))

    def loss(p, x, pos, freqs):
      return jnp.sum(ds_model.decoder_layer(p, x, pos, pos, freqs, cfg, is_moe=True))

    return _matmul_precisions(jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(layer, x, pos, freqs).jaxpr)

  def test_every_matmul_of_a_layer_takes_it(self):
    """The ragged MoE, which holds every site that used to take none; the
    grouped matmul's other modes are CustomVjpTest's."""
    self.assertEqual(self._layer_precisions(matmul_precision="highest"), {_HIGHEST})

  def test_none_at_the_default(self):
    """The None those sites always passed, so the default program is the
    one it was; any other precision is passed as itself."""
    self.assertIsNone(ds_layers.precision_or_none(_config()))
    self.assertEqual(ds_layers.precision_or_none(_config(matmul_precision="highest")), jax.lax.Precision.HIGHEST)


# ------------------------------------------------------------- call sites
class CallSiteTest(unittest.TestCase):
  """The model never shards by itself: every departure from today's program
  is a `mode` helper, so `sharding_mode_test` and the A/B gate see all of
  them."""

  FILES = (
      "deepseek/layers.py",
      "deepseek/rope.py",
      "deepseek/attention.py",
      "deepseek/moe.py",
      "deepseek/model.py",
      "deepseek/train.py",
      "deepseek/init.py",
  )
  FORBIDDEN = (
      r"\breshard\(",
      r"with_sharding_constraint",
      r"\bauto_axes\(",
      r"\bshard_map\(",
      r"use_abstract_mesh",
      r"set_mesh\(",
      r"NamedSharding\(",
  )
  # A custom-VJP rule types what it returns from its primals' avals, the one
  # place `out_sharding=` is written out (see `mode`'s module docstring).
  VJP_RULES = ("_sort_activations", "_sort_activations_bwd", "_grouped_matmul", "_grouped_matmul_bwd")

  def _sources(self):
    for rel in self.FILES:
      with open(os.path.join(testing.package_dir(), rel)) as f:
        yield rel, f.read()

  def test_no_direct_sharding_calls(self):
    for rel, src in self._sources():
      code = "\n".join(line.split("#")[0] for line in src.splitlines())
      for pat in self.FORBIDDEN:
        with self.subTest(file=rel, pattern=pat):
          self.assertIsNone(re.search(pat, code))

  def test_out_sharding_only_from_a_primal_in_a_vjp_rule(self):
    def from_aval(value, assigned):
      if isinstance(value, ast.Call):
        return ast.unparse(value.func) in ("mode.bwd_out_sharding", "mode.sharding_of")
      return isinstance(value, ast.Name) and assigned.get(value.id) is True

    for rel, src in self._sources():
      for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef):
          continue
        # Names bound from `mode.bwd_out_sharding(...)` or
        # `mode.sharding_of(...)` in this function.
        assigned = {}
        for node in ast.walk(fn):
          if isinstance(node, ast.Assign):
            targets = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets
            values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value] * len(targets)
            for t, v in zip(targets, values):
              if isinstance(t, ast.Name):
                assigned[t.id] = assigned.get(t.id, True) and from_aval(v, {})
        for node in ast.walk(fn):
          for kw in getattr(node, "keywords", ()):
            if kw.arg == "out_sharding":
              with self.subTest(file=rel, function=fn.name, line=kw.value.lineno):
                self.assertIn(fn.name, self.VJP_RULES)
                self.assertTrue(from_aval(kw.value, assigned), ast.unparse(kw.value))


# ------------------------------------------------------ query projection
class QueryProjectionTest(unittest.TestCase):
  """`q_lora_rank == 0` is MaxText's plain `query` projection, not rank-0 LoRA.

  Built as a rank-0 LoRA, `wq_a` is `[emb, 0]` and every query is exactly
  zero: attention goes uniform over the causal window and nothing fails.  The
  pipeline tests' own configs ran at rank 0, so every parity test was passing
  on that degenerate model until the two paths were split.
  """

  def test_rank_zero_is_a_dense_query(self):
    att = ds_model.param_avals(_config(q_lora_rank=0))["decoder"]["moe_layers"]["self_attention"]
    self.assertIn("query", att)
    self.assertNotIn("wq_a", att)
    c = _config()
    self.assertEqual(
        tuple(att["query"]["kernel"].shape),
        (c.num_decoder_layers - c.first_num_dense_layers, c.emb_dim, c.num_query_heads, c.qk_head_dim),
    )

  def test_queries_reach_the_loss(self):
    for rank in (0, 16):
      cfg = _config(q_lora_rank=rank)
      params = ds_init.init_params(ds_model.param_avals(cfg), jax.random.key(5))
      aux = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), ds_model.aux_avals(cfg))
      batch = ds_train.synthetic_batch(cfg.vocab_size, GB, SEQ)
      g = jax.grad(lambda p: ds_train.compute_loss(p, aux, batch, cfg, ds_train.TrainConfig())[0])(params)
      att = g["decoder"]["dense_layers"]["self_attention"]
      key = "query" if rank == 0 else "wq_b"
      self.assertGreater(float(jnp.max(jnp.abs(att[key]["kernel"]))), 0.0, f"rank {rank}: no gradient reaches {key}")


# ------------------------------------------------------------------- init
class InitTest(unittest.TestCase):

  def test_matches_the_avals(self):
    cfg = _config()
    avals = ds_model.param_avals(cfg)
    p = ds_init.init_params(avals, jax.random.key(3))
    self.assertEqual(jax.tree.structure(p), jax.tree.structure(avals))
    for got, want in zip(jax.tree.leaves(p), jax.tree.leaves(avals)):
      self.assertEqual(got.shape, want.shape)
      self.assertEqual(got.dtype, want.dtype)

  def test_norm_scales_start_at_one(self):
    p = ds_init.init_params(ds_model.param_avals(_config()), jax.random.key(3))
    scale = p["decoder"]["decoder_norm"]["scale"]
    self.assertEqual(_relerr(scale, np.ones_like(scale)), 0.0)


if __name__ == "__main__":
  unittest.main()
