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

"""gdn_scan_impl 'triton' (src/maxtext/experimental/mpmd/jax/qwen35/gdn_triton.py): the GDN chunked delta rule's
inter-chunk recurrence as
Triton kernels, against the lax.scan it replaces.

CPU (any machine): the configuration and its refusals, and the restructured math -- the recurrence's outputs h and
vn, the chunk outputs batched over the chunks, and the hand-written VJP -- with the two kernels swapped for a JAX
emulation of exactly what they compute, against the scan's output and jax.grad of it.  GPU: the kernels themselves,
against the scan, run with
`JAX_PLATFORMS=cuda XLA_FLAGS= python -m pytest tests/unit/experimental/mpmd/jax/gdn_kernels_test.py`.
"""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.qwen35 import config as qc
from maxtext.experimental.mpmd.jax.qwen35 import gdn, gdn_triton
from maxtext.experimental.mpmd.jax.qwen35 import model as qm
from maxtext.experimental.mpmd.jax import testing

# The Triton kernels' packages: no requirements file of the package brings them (testing.requires_gpu).
TRITON = ("triton", "jax_triton")


def inputs(b, s, h, d, seed=0, dtype=jnp.float32):
  ks = jax.random.split(jax.random.key(seed), 6)
  q = jax.random.normal(ks[0], (b, s, h, d), jnp.float32).astype(dtype)
  k = jax.random.normal(ks[1], (b, s, h, d), jnp.float32).astype(dtype)
  v = jax.random.normal(ks[2], (b, s, h, d), jnp.float32).astype(dtype)
  g = -jax.nn.softplus(jax.random.normal(ks[3], (b, s, h), jnp.float32)) * 0.3
  beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, s, h), jnp.float32)).astype(dtype)
  ct = jax.random.normal(ks[5], (b, s, h, d), jnp.float32)
  return (q, k, v, g, beta), ct


def grouped(args, rep):
  """`inputs`' (q, k, v, g, beta) with H / rep key heads -- q and k every rep-th head -- for H value heads: value head
  j reads key head j // rep (MaxText's GQA-like Gated DeltaNet, Qwen3.5's 2:1 and 4:1)."""
  q, k, v, g, beta = args
  return (q[:, :, ::rep], k[:, :, ::rep], v, g, beta)


# (b, s, value heads, d, chunk, seed, value heads a key head): padded S, a key head for 2, 3 and 4 value heads
GROUPED = ((2, 64, 6, 16, 16, 0, 3), (1, 100, 4, 32, 32, 1, 4), (2, 45, 4, 16, 16, 2, 2))


def run(impl, args, ct, chunk, cdt):
  """The rule's output and the gradients of <o, ct> with respect to q, k, v, g, beta."""
  f = lambda *a: gdn.chunk_gated_delta_rule(*a, chunk, True, cdt, scan_impl=impl)
  out = jax.jit(f)(*args)
  grads = jax.jit(jax.grad(lambda *a: jnp.sum(f(*a).astype(jnp.float32) * ct), argnums=(0, 1, 2, 3, 4)))(*args)
  return (out,) + tuple(grads)


def rel(a, b):
  a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
  return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")


# --- a JAX emulation of exactly what the two kernels compute (per (b, h); chunks on axis 1) ---
def emulated_fwd(u, w, k, e, a, q, eg, block_v, precision, store_h):
  hi = jax.lax.Precision.HIGHEST
  f32 = jnp.float32

  def step(s, x):
    u_n, w_n, k_n, e_n, a_n, q_n, eg_n = x
    o = jnp.matmul(q_n.astype(f32) * eg_n[..., None], s, precision=hi)
    vn = u_n.astype(f32) - jnp.matmul(w_n.astype(f32), s, precision=hi)
    kd = k_n.astype(f32) * e_n[..., None]
    return s * a_n[..., None, None] + jnp.matmul(kd.swapaxes(-1, -2), vn, precision=hi), (o, vn, s)

  b, n, h, c, dk = w.shape
  xs = tuple(jnp.moveaxis(x, 1, 0) for x in (u, w, k, e.astype(f32), a.astype(f32), q, eg.astype(f32)))
  _, (os_, vns, hs) = jax.lax.scan(step, jnp.zeros((b, h, dk, u.shape[-1]), f32), xs)
  hs = jnp.moveaxis(hs, 0, 1) if store_h else jnp.zeros((1,), f32)
  return jnp.moveaxis(os_, 0, 1), jnp.moveaxis(vns, 0, 1), hs


def emulated_bwd(w, k, e, a, q, eg, do, dvn, block_v, precision):
  hi = jax.lax.Precision.HIGHEST
  f32 = jnp.float32

  def step(lam, x):
    w_n, k_n, e_n, a_n, q_n, eg_n, do_n, dvn_n = x
    kd = k_n.astype(f32) * e_n[..., None]
    dvt = dvn_n + jnp.matmul(kd, lam, precision=hi)
    dh = jnp.matmul((q_n.astype(f32) * eg_n[..., None]).swapaxes(-1, -2), do_n, precision=hi)
    return dh + lam * a_n[..., None, None] - jnp.matmul(w_n.astype(f32).swapaxes(-1, -2), dvt, precision=hi), (dvt, lam)

  b, n, h, c, dk = w.shape
  xs = tuple(
      jnp.moveaxis(x, 1, 0)
      for x in (w, k, e.astype(f32), a.astype(f32), q, eg.astype(f32), do.astype(f32), dvn.astype(f32))
  )
  _, (dvts, lams) = jax.lax.scan(step, jnp.zeros((b, h, dk, do.shape[-1]), f32), xs, reverse=True)
  return jnp.moveaxis(dvts, 0, 1), jnp.moveaxis(lams, 0, 1)


# --- a JAX emulation of the fused path's per-chunk kernels (FUSED: chunk_rule), what each computes, in fp32 ---
HI = jax.lax.Precision.HIGHEST


def _decay(g, inclusive):
  c = g.shape[-1]
  mask = jnp.tril(jnp.ones((c, c), dtype=bool), k=0 if inclusive else -1)
  return jnp.where(mask, jnp.exp(jnp.where(mask, g[..., :, None] - g[..., None, :], 0.0)), 0.0)


def emu_wy_fwd(k, v, beta, g, precision):
  f32 = jnp.float32
  kb = (k.astype(f32) * beta.astype(f32)[..., None]).astype(k.dtype).astype(f32)
  vb = (v.astype(f32) * beta.astype(f32)[..., None]).astype(v.dtype).astype(f32)
  # the kernel solves the triangular system I + S (wy_fwd: 16-row diagonal blocks, then the rows below them), so its
  # emulation is the solve, where the scan it is held to inverts by block doubling (gdn_chunk_inverse's default):
  # the same inverse to fp32 rounding, far inside the checks' 1e-5
  s = jnp.matmul(kb, k.astype(f32).swapaxes(-1, -2), precision=HI) * _decay(g, False)
  a = gdn.unit_lower_inv(s, "solve")
  u = jnp.matmul(a, vb, precision=HI).astype(k.dtype)
  w = jnp.matmul(a * jnp.exp(g)[..., None, :], kb, precision=HI).astype(k.dtype)
  return a, u, w


def emu_out(oi, q, k, g, vn, precision, out_dtype=jnp.float32):
  f32 = jnp.float32
  p = jnp.matmul(q.astype(f32), k.astype(f32).swapaxes(-1, -2), precision=HI) * _decay(g, True)
  return (oi + jnp.matmul(p, vn, precision=HI)).astype(out_dtype)


def emu_out_bwd(q, k, g, do, vn, precision):
  f32 = jnp.float32
  dm = _decay(g, True)
  qk = jnp.matmul(q.astype(f32), k.astype(f32).swapaxes(-1, -2), precision=HI)
  dvn = jnp.matmul((qk * dm).swapaxes(-1, -2), do, precision=HI)
  dqk = jnp.matmul(do, vn.swapaxes(-1, -2), precision=HI) * dm
  dq = jnp.matmul(dqk, k.astype(f32), precision=HI)
  dk = jnp.matmul(dqk.swapaxes(-1, -2), q.astype(f32), precision=HI)
  dl = dqk * qk
  return dvn, dq, dk, dl.sum(-1) - dl.sum(-2)


def emu_prod(dvt, h, do, vn, lam, k, q, e, eg, dq_o, dk_o, precision, dq_dtype=jnp.float32):
  f32 = jnp.float32
  ht = h.swapaxes(-1, -2)
  dqg = jnp.matmul(do, ht, precision=HI)
  dkd = jnp.matmul(vn, lam.swapaxes(-1, -2), precision=HI)
  return (
      -jnp.matmul(dvt, ht, precision=HI),
      (dqg * eg[..., None] + dq_o).astype(dq_dtype),
      dkd * e[..., None] + dk_o,
      (dqg * q.astype(f32)).sum(-1),
      (dkd * k.astype(f32)).sum(-1),
      (h * lam).sum((-2, -1)),
  )


def emu_wy_bwd(k, v, beta, g, a, du, dw, dk_in, precision, dk_dtype=jnp.float32, dv_dtype=jnp.float32):
  f32 = jnp.float32
  kf, vf, bf = k.astype(f32), v.astype(f32), beta.astype(f32)
  kb = (kf * bf[..., None]).astype(k.dtype).astype(f32)
  vb = (vf * bf[..., None]).astype(v.dtype).astype(f32)
  eg = jnp.exp(g)
  at = a.swapaxes(-1, -2)
  da = (
      jnp.matmul(du, vb.swapaxes(-1, -2), precision=HI)
      + jnp.matmul(dw, kb.swapaxes(-1, -2), precision=HI) * eg[..., None, :]
  )
  ds = -jnp.matmul(jnp.matmul(at, da, precision=HI), at, precision=HI)
  dkk = ds * _decay(g, False)
  dl = dkk * jnp.matmul(kb, kf.swapaxes(-1, -2), precision=HI)
  dkbe = jnp.matmul(at, dw, precision=HI)
  dkb = jnp.matmul(dkk, kf, precision=HI) + dkbe * eg[..., None]
  dk = jnp.matmul(dkk.swapaxes(-1, -2), kb, precision=HI) + dkb * bf[..., None]
  dvb = jnp.matmul(at, du, precision=HI)
  return (
      (dk + dk_in).astype(dk_dtype),
      (dvb * bf[..., None]).astype(dv_dtype),
      (dkb * kf).sum(-1) + (dvb * vf).sum(-1),
      dl.sum(-1) - dl.sum(-2) + eg * (dkbe * kb).sum(-1),
  )


EMULATED = dict(
    _fwd_call=emulated_fwd,
    _bwd_call=emulated_bwd,
    _wy_fwd_call=emu_wy_fwd,
    _out_call=emu_out,
    _out_bwd_call=emu_out_bwd,
    _prod_call=emu_prod,
    _wy_bwd_call=emu_wy_bwd,
)


class Config(unittest.TestCase):

  def test_values_and_refusals(self):
    """gdn_scan_impl is 'scan' unless set, takes GDN_SCAN_IMPLS only; the kernel path refuses a backend without
    Triton, a sharded (auto / explicit) context and packed documents -- before any custom call is traced -- and
    shapes its kernels cannot tile."""
    self.assertEqual(qc.GDN_SCAN_IMPLS, ("scan", "triton"))
    self.assertEqual(qc.Qwen35Config().gdn_scan_impl, "scan")
    self.assertEqual(qc.Qwen35Config(gdn_scan_impl="triton").gdn_scan_impl, "triton")
    with self.assertRaisesRegex(ValueError, "gdn_scan_impl"):
      qc.Qwen35Config(gdn_scan_impl="pallas")
    args, _ = inputs(1, 32, 2, 16)
    if jax.default_backend() != "gpu":
      with self.assertRaisesRegex(ValueError, "needs a GPU"):
        gdn.chunk_gated_delta_rule(*args, 16, True, jnp.float32, scan_impl="triton")
    with (
        mock.patch.object(jax, "default_backend", lambda: "gpu"),
        mock.patch("maxtext.experimental.mpmd.jax.sharding_mode.context_mode", lambda: "auto"),
        self.assertRaisesRegex(ValueError, "gdn_core='manual'"),
    ):
      gdn.chunk_gated_delta_rule(*args, 16, True, jnp.float32, scan_impl="triton")
    with self.assertRaisesRegex(ValueError, "'scan' or 'triton'"):
      gdn.chunk_gated_delta_rule(*args, 16, True, jnp.float32, scan_impl="pallas")
    # Packed documents: the kernels restart no state at a document, so the rule refuses segment ids on any backend,
    # before the GPU check, and so does a GDN layer given them under gdn_packing (the single-controller model passes
    # them always: qwen35.train, all one for unpacked rows); gdn_packing=False keeps them out of the rule.
    ones = jnp.ones((1, 32), jnp.int32)
    with self.assertRaisesRegex(ValueError, "no packed documents"):
      gdn.chunk_gated_delta_rule(*args, 16, True, jnp.float32, scan_impl="triton", segment_ids=ones)
    for packing, refusal in ((True, "no packed documents"), (False, "needs a GPU")):
      if not packing and jax.default_backend() == "gpu":  # past the refusal it runs the kernels there
        continue
      cfg = qc.Qwen35Config(
          emb_dim=64,
          gdn_num_key_heads=2,
          gdn_num_value_heads=2,
          gdn_key_head_dim=16,
          gdn_value_head_dim=16,
          gdn_chunk_size=16,
          dtype=jnp.float32,
          gdn_scan_impl="triton",
          gdn_packing=packing,
      )
      mixer = qm.gdn_layer_avals(cfg, 1)["attention"]  # one layer: the stacked avals without their layer axis
      mixer = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), mixer)
      x = jax.ShapeDtypeStruct((1, 32, cfg.emb_dim), jnp.float32)
      with self.subTest(gdn_packing=packing), self.assertRaisesRegex(ValueError, refusal):
        jax.eval_shape(lambda p, x, cfg=cfg: gdn.gated_delta_net(p, x, ones, cfg), mixer, x)
    self.assertTrue(gdn_triton.supported(64, 128, 128))
    for bad in ((8, 128, 128), (64, 96, 128), (64, 128, 8), (48, 128, 128)):
      self.assertFalse(gdn_triton.supported(*bad), bad)
    with self.assertRaisesRegex(ValueError, "powers of two"):
      gdn_triton.inter_chunk(*(jnp.zeros((1, 1, 1, 8, 8)),) * 4, jnp.zeros((1, 1, 1, 8)))
    with self.assertRaisesRegex(ValueError, "precision"):
      gdn_triton.inter_chunk(*(jnp.zeros((1, 1, 1, 16, 16)),) * 4, jnp.zeros((1, 1, 1, 16)), precision="bf16")


class RestructuredMath(unittest.TestCase):
  """The kernel path's math with the kernels emulated in JAX (`emulated_fwd` / `emulated_bwd`, what the kernels
  compute, step for step): the recurrence's outputs feed the chunk outputs batched over the chunks, and the
  hand-written VJP -- du, dw, dKd (-> dk and the decay's de), da, dq_g (-> dq and e^g's) -- must give jax.grad
  of the scan, in fp32 to fp32 rounding (1e-5 of each tensor's largest entry), padded S included.  A wrong VJP term
  is caught (the control:
  the state decay's gradient dropped)."""

  def setUp(self):
    self._ctx = mock.patch.object(gdn, "_triton_context", lambda: None)
    self._ctx.start()
    self.addCleanup(self._ctx.stop)
    unfused = mock.patch.object(gdn_triton, "FUSED", False)  # the recurrence's kernels alone
    unfused.start()
    self.addCleanup(unfused.stop)

  def check(self, b, s, h, d, chunk, seed, tol=1e-5, rep=1):
    args, ct = inputs(b, s, h, d, seed)
    args = grouped(args, rep)
    want = run("scan", args, ct, chunk, jnp.float32)
    with (
        mock.patch.object(gdn_triton, "_fwd_call", emulated_fwd),
        mock.patch.object(gdn_triton, "_bwd_call", emulated_bwd),
    ):
      got = run("triton", args, ct, chunk, jnp.float32)
    self.assertEqual([x.shape for x in got], [x.shape for x in want])
    errs = {n: rel(g, w) for n, g, w in zip(NAMES, got, want)}
    for n, err in errs.items():
      self.assertLess(err, tol, (n, errs))
    return errs

  def test_against_the_scan(self):
    for shape in ((2, 64, 3, 16, 16, 0), (1, 100, 2, 32, 16, 1), (2, 45, 2, 16, 32, 2)):
      with self.subTest(shape=shape):
        self.check(*shape)

  def test_grouped_heads(self):
    """Fewer key heads than value heads (`grouped`): the kernels take q and k repeated to the value heads
    (chunk_gated_delta_rule's heads()), and dq and dk come back at the key heads, each a group's sum."""
    for *shape, rep in GROUPED:
      with self.subTest(shape=shape, rep=rep):
        self.check(*shape, rep=rep)

  def test_a_broken_vjp_is_caught(self):
    real = gdn_triton._scan_bwd

    def no_decay_grad(block_v, precision, res, cts):
      du, dw, dk, de, da, dq, deg = real(block_v, precision, res, cts)
      return du, dw, dk, de, jnp.zeros_like(da), dq, deg

    primal = gdn_triton.scan.fun  # the undecorated primal, taken before the patch below
    broken = jax.custom_vjp(primal, nondiff_argnums=(7, 8))
    broken.defvjp(gdn_triton._scan_fwd, no_decay_grad)
    with mock.patch.object(gdn_triton, "scan", broken), self.assertRaisesRegex(AssertionError, r"\('dg', "):
      self.check(2, 64, 3, 16, 16, 0)


class FusedMath(unittest.TestCase):
  """FUSED (`gdn_triton.chunk_rule`: the WY factors, the triangular solve, the intra-chunk output and the whole
  backward on per-chunk kernels) with every kernel swapped for a JAX emulation of exactly what it computes
  (`EMULATED`): the output and all five input gradients must be the scan's and jax.grad of it, in fp32 to fp32
  rounding (1e-5 of each tensor's largest entry), padded S included; a wrong backward term is caught (the control:
  the WY factors' decay gradient dropped)."""

  def setUp(self):
    ctx = mock.patch.object(gdn, "_triton_context", lambda: None)
    ctx.start()
    self.addCleanup(ctx.stop)

  def check(self, b, s, h, d, chunk, seed, patches=None, rep=1):
    args, ct = inputs(b, s, h, d, seed)
    args = grouped(args, rep)
    want = run("scan", args, ct, chunk, jnp.float32)
    with (
        mock.patch.multiple(gdn_triton, **dict(EMULATED, **(patches or {}))),
        mock.patch.object(gdn_triton, "FUSED", True),
    ):
      got = run("triton", args, ct, chunk, jnp.float32)
    self.assertEqual([x.shape for x in got], [x.shape for x in want])
    errs = {n: rel(g, w) for n, g, w in zip(NAMES, got, want)}
    for n, err in errs.items():
      self.assertLess(err, 1e-5, (n, errs))
    return errs

  def test_against_the_scan(self):
    for shape in ((2, 64, 3, 16, 16, 0), (1, 100, 2, 32, 32, 1), (2, 150, 2, 16, 64, 2)):
      with self.subTest(shape=shape):
        self.check(*shape)

  def test_grouped_heads(self):
    """Fewer key heads than value heads (`grouped`), as RestructuredMath's: the fused kernels too take q and k
    repeated to the value heads, and dq and dk come back at the key heads."""
    for *shape, rep in GROUPED:
      with self.subTest(shape=shape, rep=rep):
        self.check(*shape, rep=rep)

  def test_a_broken_backward_is_caught(self):
    def no_decay_grad(*a, **kw):
      dk, dv, dbeta, dg = emu_wy_bwd(*a, **kw)
      return dk, dv, dbeta, jnp.zeros_like(dg)

    with self.assertRaisesRegex(AssertionError, r"\('dg', "):
      self.check(2, 64, 3, 16, 16, 0, patches=dict(_wy_bwd_call=no_decay_grad))

  def test_shapes_it_takes(self):
    self.assertEqual(gdn_triton.FUSED_CHUNKS, (16, 32, 64))
    self.assertTrue(gdn_triton.fused_supported(64, 128, 128))
    self.assertFalse(gdn_triton.fused_supported(128, 128, 128))  # the unfused kernels take chunk 128
    self.assertTrue(gdn_triton.supported(128, 128, 128))
    with self.assertRaisesRegex(ValueError, "chunk of"):
      gdn_triton.chunk(*(jnp.zeros((1, 1, 1, 128, 16)),) * 3, jnp.zeros((1, 1, 1, 128)), jnp.zeros((1, 1, 1, 128)))


@testing.requires_gpu(*TRITON, why="the Triton kernels")
class KernelsOnGpu(unittest.TestCase):
  """The kernels against the scan: in fp32 to fp32 rounding (1e-5 of each tensor's largest entry: the kernels sum
  their matmuls in another order than cuBLAS); in bf16, where every gradient is rounded to bf16 at the end, both
  are compared against the scan run in fp32 on the same bf16 inputs, and the kernel path must be no farther from it
  than 1.5x the scan's own bf16 run (plus a bf16 ulp)."""

  def test_fp32(self):
    for b, s, h, d, chunk, seed in (
        (2, 64, 3, 16, 16, 0),
        (2, 100, 4, 32, 16, 1),
        (2, 300, 3, 64, 32, 3),
        (1, 200, 2, 128, 64, 5),
    ):
      with self.subTest(shape=(b, s, h, d, chunk)):
        args, ct = inputs(b, s, h, d, seed)
        want, got = run("scan", args, ct, chunk, jnp.float32), run("triton", args, ct, chunk, jnp.float32)
        errs = {n: rel(g, w) for n, g, w in zip(NAMES, got, want)}
        for n, err in errs.items():
          self.assertLess(err, 1e-5, (n, errs))

  def test_split_precisions_against_fp32(self):
    """The exact-operand splits at the kernel level: on bf16 q, k, w, u and fp32 decays / cotangents, 'bf16x3e'
    and 'tf32x2e' must give fp32's result (the emulation's HIGHEST matmuls on the same values) to 1e-5 of each
    output's largest entry, for two Dv tile widths -- a split cut to one bf16 product (2^-8 off) fails this, where
    the end-to-end bf16 comparisons above, rounded to bf16 at the end, could not tell."""
    key = jax.random.split(jax.random.key(11), 9)
    b, n, h, c, dk, dv = 2, 8, 3, 32, 64, 64
    bf = lambda k_, shape: jax.random.normal(k_, shape, jnp.float32).astype(jnp.bfloat16)
    u, w, k, q = (
        bf(key[0], (b, n, h, c, dv)),
        bf(key[1], (b, n, h, c, dk)) * 0.1,
        bf(key[2], (b, n, h, c, dk)) * 0.1,
        bf(key[3], (b, n, h, c, dk)),
    )
    g = -jax.nn.softplus(jax.random.normal(key[4], (b, n, h, c), jnp.float32)) * 0.05
    g = jnp.cumsum(g, axis=-1)
    e, a, eg = jnp.exp(g[..., -1:] - g), jnp.exp(g[..., -1]), jnp.exp(g)
    do = jax.random.normal(key[5], (b, n, h, c, dv), jnp.float32)
    dvn = jax.random.normal(key[6], (b, n, h, c, dv), jnp.float32)
    want_f = emulated_fwd(u, w, k, e, a, q, eg, None, None, True)
    want_b = emulated_bwd(w, k, e, a, q, eg, do, dvn, None, None)
    for precision in ("bf16x3e", "tf32x2e"):
      for block_v in (16, 64):
        with self.subTest(precision=precision, block_v=block_v):
          got_f = jax.jit(lambda *x: gdn_triton._fwd_call(*x, block_v, precision, True))(u, w, k, e, a, q, eg)
          got_b = jax.jit(lambda *x: gdn_triton._bwd_call(*x, block_v, precision))(w, k, e, a, q, eg, do, dvn)
          for name, gv, wv in zip(("o", "vn", "h", "dvt", "lam"), got_f + got_b, want_f + want_b):
            self.assertLess(rel(gv, wv), 1e-5, name)

  def test_bf16(self):
    for b, s, h, d, chunk, seed in ((2, 100, 4, 32, 16, 1), (2, 300, 3, 64, 64, 4)):
      with self.subTest(shape=(b, s, h, d, chunk)):
        args, ct = inputs(b, s, h, d, seed, jnp.bfloat16)
        exact = run("scan", tuple(x.astype(jnp.float32) for x in args), ct, chunk, jnp.float32)
        scan, kern = run("scan", args, ct, chunk, jnp.bfloat16), run("triton", args, ct, chunk, jnp.bfloat16)
        for n, s_, k_, x_ in zip(NAMES, scan, kern, exact):
          self.assertLessEqual(rel(k_, x_), 1.5 * rel(s_, x_) + 2**-8, n)

  def test_precisions_and_blocks(self):
    """Every Dv tile width and dot precision, on both paths (FUSED, `chunk`; the recurrence's kernels alone,
    `inter_chunk`): in fp32 compute 'ieee' and 'tf32x3' match the scan to fp32-ish rounding; in bf16 compute the
    exact-operand splits ('bf16x3e', the default there, and 'tf32x2e') are as far from the fp32-compute scan as the
    bf16 scan is, and refuse fp32 operands."""
    args, ct = inputs(2, 128, 2, 64, 7)
    want = run("scan", args, ct, 32, jnp.float32)
    with self.assertRaisesRegex(ValueError, "bf16"):
      gdn_triton.inter_chunk(*(jnp.zeros((1, 1, 1, 16, 16)),) * 4, jnp.zeros((1, 1, 1, 16)), precision="bf16x3e")
    with self.assertRaisesRegex(ValueError, "bf16"):
      gdn_triton.chunk(
          *(jnp.zeros((1, 1, 1, 16, 16)),) * 3, jnp.zeros((1, 1, 1, 16)), jnp.zeros((1, 1, 1, 16)), precision="bf16x3e"
      )
    args16, _ = inputs(2, 128, 2, 64, 7, jnp.bfloat16)
    exact = run("scan", tuple(x.astype(jnp.float32) for x in args16), ct, 32, jnp.float32)
    scan16 = run("scan", args16, ct, 32, jnp.bfloat16)
    for fused, entry in ((True, "chunk"), (False, "inter_chunk")):
      orig = getattr(gdn_triton, entry)
      with mock.patch.object(gdn_triton, "FUSED", fused):
        for precision in ("bf16x3e", "tf32x2e"):
          with (
              self.subTest(fused=fused, precision=precision),
              mock.patch.object(gdn_triton, entry, lambda *a, p=precision, **kw: orig(*a, precision=p, **kw)),
          ):
            got = run("triton", args16, ct, 32, jnp.bfloat16)
            for n, g_, s_, x_ in zip(NAMES, got, scan16, exact):
              self.assertLessEqual(rel(g_, x_), 1.5 * rel(s_, x_) + 2**-8, n)
        for block_v, precision, tol in ((16, "ieee", 1e-5), (64, "ieee", 1e-5), (32, "tf32x3", 1e-4)):
          with (
              self.subTest(fused=fused, block_v=block_v, precision=precision),
              mock.patch.object(
                  gdn_triton, entry, lambda *a, b=block_v, p=precision, **kw: orig(*a, block_v=b, precision=p, **kw)
              ),
          ):
            got = run("triton", args, ct, 32, jnp.float32)
            for n, g, w in zip(NAMES, got, want):
              self.assertLess(rel(g, w), tol, n)

  def test_fused_kernels_against_their_emulation(self):
    """FUSED's per-chunk kernels one at a time against their emulation (`EMULATED`: HIGHEST matmuls on the same
    values): on fp32 operands in 'ieee' to 1e-5 of each output's largest entry; on bf16 q, k, v, beta (fp32 decays,
    states and their cotangents; the output's cotangent in bf16 -- exact, one-sided -- and in fp32) with the
    exact-operand splits to 1e-4: the products with both operands fp32 take two bf16 parts a side ('bf16x3', which
    drops terms of 2^-16), and one bf16 product (2^-8 off) fails it.  u and w, stored in bf16, to a bf16 rounding."""
    f32, bf16 = jnp.float32, jnp.bfloat16
    for b, n, h, c, dk, dv in ((1, 3, 2, 64, 128, 64), (2, 2, 2, 16, 32, 128)):
      key = iter(jax.random.split(jax.random.key(c + dk), 16))
      nrm = lambda shape, scale=1.0: jax.random.normal(next(key), shape, f32) * scale
      q, k, v = nrm((b, n, h, c, dk)), nrm((b, n, h, c, dk), dk**-0.5), nrm((b, n, h, c, dv))
      beta = jax.nn.sigmoid(nrm((b, n, h, c)))
      g = jnp.cumsum(-jax.nn.softplus(nrm((b, n, h, c))) * 0.1, axis=-1)
      e, a, eg = gdn_triton._decays(g)
      oi, vn, du, dvt = (nrm((b, n, h, c, dv)) for _ in range(4))
      dw, dq_o, dk_o = (nrm((b, n, h, c, dk)) for _ in range(3))
      hs, lam = nrm((b, n, h, dk, dv)), nrm((b, n, h, dk, dv))
      do32 = nrm((b, n, h, c, dv))
      for precision in ("ieee", "bf16x3e", "tf32x2e"):
        dt, tol = (f32, 1e-5) if precision == "ieee" else (bf16, 1e-4)
        qd, kd, vd, betad = (x.astype(dt) for x in (q, k, v, beta))
        a_inv = emu_wy_fwd(kd, vd, betad, g, None)[0]
        cases = [
            ("wy_fwd", ("A", "u", "w"), emu_wy_fwd, gdn_triton._wy_fwd_call, (kd, vd, betad, g)),
            ("out", ("o",), emu_out, gdn_triton._out_call, (oi, qd, kd, g, vn)),
            (
                "prod",
                ("dw", "dq", "dk", "deg", "de", "da"),
                emu_prod,
                gdn_triton._prod_call,
                (dvt, hs, None, vn, lam, kd, qd, e, eg, dq_o, dk_o),
            ),
            (
                "wy_bwd",
                ("dk", "dv", "dbeta", "dg"),
                emu_wy_bwd,
                gdn_triton._wy_bwd_call,
                (kd, vd, betad, g, a_inv, du, dw, dk_o),
            ),
            ("out_bwd", ("dvn", "dq", "dk", "dg"), emu_out_bwd, gdn_triton._out_bwd_call, (qd, kd, g, None, vn)),
        ]
        for name, outs, emu, kern, xs in cases:
          for do in ((do32,) if dt == f32 else (do32.astype(bf16), do32)) if None in xs else (None,):
            xs_ = tuple(do if x is None else x for x in xs)
            with self.subTest(
                shape=(b, n, h, c, dk, dv), precision=precision, kernel=name, do=None if do is None else do.dtype.name
            ):
              want = emu(*xs_, precision)
              got = jax.jit(lambda *x, kern=kern: kern(*x, precision))(*xs_)
              for o, gv, wv in zip(outs, got, want):
                bound = 2**-8 if (name, o) in (("wy_fwd", "u"), ("wy_fwd", "w")) and dt == bf16 else tol
                self.assertLess(rel(gv, wv), bound, (name, o, rel(gv, wv)))


if __name__ == "__main__":
  unittest.main()
