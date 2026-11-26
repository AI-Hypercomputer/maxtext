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

"""gpu_kernels: the Blackwell grouped matmul that reads the weight as stored (`grouped_matmul`) and the in-place dW
(`ragged_dw`).

CPU (any machine): the dispatch -- which products ds_ep hands `grouped_matmul` (`_own_mosaic`), which of
`ragged_dw`'s kernels a dW with a starting value runs per tokamax_gmm (`_ragged_dw`; and --inplace-dw auto by it) --
its tiles rule and argument checks, and `ragged_dw`'s off-GPU form.  The kernels themselves only run on an sm_100
GPU (GB200 / B200: tcgen05); those tests skip elsewhere and are run there with
`JAX_PLATFORMS=cuda XLA_FLAGS= python -m pytest tests/unit/experimental/mpmd/jax/gpu_kernels_test.py`.
"""

import itertools
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from maxtext.experimental.mpmd.jax import ds_ep, gpu_kernels, testing
from tests.unit.experimental.mpmd.jax.ds_ep_test import HAVE_TOKAMAX, _sys_modules

# The Triton kernel's packages: no requirements file of the package brings them (testing.requires_gpu).
TRITON = ("triton", "jax_triton")
_CON = lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())), lhs_ragged_dimensions=[0], rhs_group_dimensions=[]
)


def group_sizes(groups, valid, seed=0, skew=None, empty=()):
  rng = np.random.default_rng(seed)
  p = None if skew is None else rng.dirichlet(np.full(groups, skew))
  gs = np.bincount(rng.choice(groups, valid, p=p), minlength=groups).astype(np.int32)
  for e in empty:
    gs[e] = 0
  return gs


def per_group(lhs, rhs, gs, transpose_rhs=False):
  """The reference, one group at a time in fp32 (lax.ragged_dot's dense-per-group lowering runs out of memory at
  the pipeline's sizes): [sum(gs), N]."""
  off = np.concatenate([[0], np.cumsum(gs)])
  outs = [
      jnp.dot(lhs[off[e] : off[e + 1]], rhs[e].T if transpose_rhs else rhs[e], preferred_element_type=jnp.float32)
      for e in range(len(gs))
  ]
  return jnp.concatenate(outs)


class Dispatch(unittest.TestCase):
  """What ds_ep hands the kernel: tokamax_gmm 'mosaic' only, bf16 both sides, weight dims multiples of 64, on sm_100
  -- everything else stays on tokamax's chain.  (tokamax's own Mosaic kernel has the same tiling limit but does not
  fall through to Triton at it: its tiling error is a ValueError at lowering, after the backend was chosen -- such
  weights want --tokamax-gmm triton.)  The tcgen05 dW too is 'mosaic''s only (test_dw_kernel, test_dw_xla), and so
  is --inplace-dw auto's fp32 accumulator in place (test_inplace_dw_auto)."""

  def test_own_mosaic(self):
    a, w = jnp.zeros((8, 64), jnp.bfloat16), jnp.zeros((2, 64, 128), jnp.bfloat16)
    self.assertEqual(ds_ep._own_mosaic("mosaic", a, w), gpu_kernels.mosaic_supported())  # the gate as it is here
    # the rules, as on an sm_100 device (mosaic_supported is read as a module global)
    with mock.patch.object(gpu_kernels, "mosaic_supported", lambda device=None: True):
      self.assertTrue(ds_ep._own_mosaic("mosaic", a, w))
      self.assertTrue(ds_ep._own_mosaic("mosaic", jnp.zeros((1001, 64), jnp.bfloat16), w))  # any row count
      for impl in ("tokamax_mosaic", "triton", "xla"):
        self.assertFalse(ds_ep._own_mosaic(impl, a, w), impl)
      self.assertFalse(ds_ep._own_mosaic("mosaic", a.astype(jnp.float32), w.astype(jnp.float32)))
      self.assertFalse(ds_ep._own_mosaic("mosaic", a, w.astype(jnp.float32)))
      self.assertFalse(
          ds_ep._own_mosaic("mosaic", jnp.zeros((8, 72), jnp.bfloat16), jnp.zeros((2, 72, 128), jnp.bfloat16))
      )
      self.assertFalse(ds_ep._own_mosaic("mosaic", a, jnp.zeros((2, 64, 200), jnp.bfloat16)))
    fake = lambda cc: types.SimpleNamespace(platform="gpu", compute_capability=cc)
    self.assertEqual(
        [gpu_kernels.mosaic_supported(fake(cc)) for cc in ("8.0", "9.0", "10.0", "10.3", "12.0")],
        [False, False, True, True, False],
    )
    self.assertFalse(gpu_kernels.mosaic_supported(types.SimpleNamespace(platform="cpu")))

  def test_dw_kernel(self):
    """A dW with a starting value -- dw0 (in-place accumulation), or under acc the handed-on copy's cotangent (batch
    split's mb0 under split_dw 'acc'), or an accumulator of its own dtype (an fp32 one: tokamax_gmm's dw32, and its
    pass-through, mb0 handing its start on to mb1, whose accumulated value mb0's dW continues) -- runs
    gpu_kernels.ragged_dw (ds_ep._ragged_dw), told by tokamax_gmm which kernel: the tcgen05 one (where it takes the
    shapes) under 'mosaic' only, as grouped_matmul; the Triton one under 'tokamax_mosaic' and 'triton'; lax's form
    under 'xla'.  Off GPU ragged_dw is lax's whatever it is told: the programs here are one."""
    x, w = jnp.ones((64, 64), jnp.bfloat16), jnp.ones((2, 64, 128), jnp.bfloat16)
    a32 = jnp.ones(w.shape, jnp.float32)
    gs, valid = jnp.asarray([24, 30], jnp.int32), (jnp.arange(64) < 54)[:, None]
    lax_dot = lambda lhs, rhs, gs, dims, out_dtype, impls: lax.ragged_dot_general(  # tokamax's products, as lax's
        lhs, rhs, gs, dims, preferred_element_type=out_dtype
    )
    gmm = lambda impl, nm, x, w, **kw: ds_ep.tokamax_gmm(nm, impl, x, w, gs, valid, **kw)
    total = lambda *ys: sum(jnp.sum(y.astype(jnp.float32)) for y in ys)

    def handed_on(impl, w, x, a32):
      y0, wp = gmm(impl, "a", x, w, acc=True)
      return total(y0, gmm(impl, "b", x, wp))

    def handed_on32(impl, w, x, a32):
      y0, start = gmm(impl, "a", x, w, acc=True, dw0=a32)
      return total(y0, gmm(impl, "b", x, w, dw0=start))

    # loss, its ragged_dw calls, the starts' dtype
    losses = {
        "dw0": (lambda impl, w, x, a32: total(gmm(impl, "a", x, w, dw0=jnp.ones_like(w))), 1, jnp.bfloat16),
        "acc": (handed_on, 1, jnp.bfloat16),
        "dw32": (lambda impl, w, x, a32: total(gmm(impl, "a", x, w, dw0=a32)), 1, jnp.float32),
        "dw32 handed on": (handed_on32, 2, jnp.float32),
    }
    # (ragged_dw's `mosaic`, `xla`): None its default tiles where the tcgen05 kernel takes the shapes, False Triton
    want = {"mosaic": (None, False), "tokamax_mosaic": (False, False), "triton": (False, False), "xla": (False, True)}
    programs = {}
    for impl in ds_ep.TOKAMAX_GMM:
      for start, (loss, calls, dtype) in losses.items():
        with (
            self.subTest(impl=impl, start=start),
            mock.patch.object(ds_ep, "_tokamax_dot", lax_dot),
            mock.patch.object(gpu_kernels, "ragged_dw", wraps=gpu_kernels.ragged_dw) as dw,
        ):
          grad = jax.grad(lambda w, x, a: loss(impl, w, x, a), argnums=(0, 2))
          programs[impl, start] = str(jax.make_jaxpr(grad)(w, x, a32))
          self.assertEqual(dw.call_count, calls)
          for call in dw.call_args_list:
            self.assertEqual((call.kwargs.get("mosaic"), call.kwargs.get("xla", False)), want[impl])
            self.assertEqual(call.args[3].dtype, dtype)
    if jax.default_backend() != "gpu":
      for start in losses:
        self.assertEqual(len({programs[impl, start] for impl in ds_ep.TOKAMAX_GMM}), 1, start)

  def test_dw_xla(self):
    """What ragged_dw runs on an sm_100 GPU (faked, its kernels ones that fail if they run), told as ds_ep tells it
    per tokamax_gmm (test_dw_kernel), into an accumulator of the operands' dtype and an fp32 one (DW_MOSAIC_F32's
    tiles): the tcgen05 kernel by default ('mosaic'), the Triton one with mosaic=False ('tokamax_mosaic', 'triton'),
    and with `xla` ('xla', mosaic=False too) lax's form -- neither kernel, the program it traces off GPU."""
    lhs, g, gs = jnp.ones((64, 256), jnp.bfloat16), jnp.ones((64, 256), jnp.bfloat16), jnp.asarray([24, 30], jnp.int32)

    def lax_form(lhs, g, gs, acc):
      dw = lax.ragged_dot_general(lhs, g, gs, _CON, preferred_element_type=jnp.float32)
      return (acc.astype(jnp.float32) + dw).astype(acc.dtype)

    for dtype, tiles in ((jnp.bfloat16, gpu_kernels.DW_MOSAIC), (jnp.float32, gpu_kernels.DW_MOSAIC_F32)):
      acc = jnp.ones((2, 256, 256), dtype)  # shapes the tcgen05 kernel takes
      want = str(jax.make_jaxpr(lax_form)(lhs, g, gs, acc))
      trace = lambda acc=acc, **kw: str(jax.make_jaxpr(lambda *a: gpu_kernels.ragged_dw(*a, **kw))(lhs, g, gs, acc))
      with (
          self.subTest(acc=jnp.dtype(dtype).name),
          mock.patch.object(jax, "default_backend", lambda: "gpu"),
          mock.patch.object(gpu_kernels, "mosaic_supported", lambda device=None: True),
          mock.patch.object(gpu_kernels, "MOSAIC_F32_ACC", True),
          mock.patch.object(gpu_kernels, "_dw_mosaic", side_effect=AssertionError("the tcgen05 kernel ran")) as tc,
          _sys_modules({"jax_triton": None}),
      ):  # the Triton kernel's import halts
        with self.assertRaisesRegex(AssertionError, "the tcgen05 kernel ran"):  # the fake GPU is what it sees
          trace()
        self.assertEqual(tc.call_args.args[4], tiles)
        with self.assertRaisesRegex(ImportError, "jax_triton"):
          trace(mosaic=False)
        self.assertEqual(trace(mosaic=False, xla=True), want)
        self.assertEqual(trace(xla=True), want)

  def test_inplace_dw_auto(self):
    """--inplace-dw auto puts an fp32 accumulator's dW in place ('all') only where that runs ragged_dw's tcgen05
    kernel -- an sm_100 GPU (faked) with MOSAIC_F32_ACC, under tokamax_gmm 'mosaic': under the others _ragged_dw runs
    it on the Triton kernel (2.69 ms at DeepSeek-V3's EP 8 shapes, against 2.67 for tokamax's dW plus the add) or
    lax's form -- and build() resolves it against the command line's --tokamax-gmm.  A value given stays."""
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp

    ap = dsp.make_parser()
    args = {impl: ap.parse_args(["--gmm-impl", "tokamax", "--tokamax-gmm", impl]) for impl in ds_ep.TOKAMAX_GMM}

    class Built(Exception):  # build() stopped at its StageConfig, with the fields it was handed
      pass

    def stage_config(**kw):
      raise Built(kw)

    with (
        mock.patch.object(jax, "default_backend", lambda: "gpu"),
        mock.patch.object(gpu_kernels, "mosaic_supported", lambda device=None: True),
        mock.patch.object(gpu_kernels, "MOSAIC_F32_ACC", True),
    ):
      self.assertEqual(
          {impl: dsp.resolve_inplace_dw("auto", impl) for impl in ds_ep.TOKAMAX_GMM},
          {"mosaic": "all", "tokamax_mosaic": "on", "triton": "on", "xla": "on"},
      )
      self.assertEqual(dsp.resolve_inplace_dw("auto"), "all")  # tokamax_gmm's default
      for impl in ds_ep.TOKAMAX_GMM:
        self.assertEqual([dsp.resolve_inplace_dw(v, impl) for v in ("on", "off", "all")], ["on", "off", "all"])
        with (
            self.subTest(tokamax_gmm=impl),
            mock.patch.object(ds_ep, "StageConfig", stage_config),
            self.assertRaises(Built) as built,
        ):
          dsp.build(args[impl])
        kw = built.exception.args[0]
        self.assertEqual(
            (kw["tokamax_gmm"], kw["inplace_dw"], kw["inplace_dw_any_dtype"]), (impl, True, impl == "mosaic")
        )

  def test_not_off_gpu(self):
    if jax.default_backend() == "gpu":
      self.skipTest("a GPU process")
    self.assertFalse(gpu_kernels.mosaic_supported())

  def test_tokamax_gmm_values(self):
    self.assertEqual(ds_ep.TOKAMAX_GMM, ("mosaic", "tokamax_mosaic", "triton", "xla"))
    self.assertEqual(ds_ep._tokamax_impls("tokamax_mosaic"), ds_ep._tokamax_impls("mosaic"))
    import dataclasses
    from tests.unit.experimental.mpmd.jax.ds_ep_test import tiny_xla

    base = dataclasses.replace(tiny_xla(), gmm_impl="tokamax", use_ragged_sort=False)
    self.assertEqual(base.tokamax_gmm, "mosaic")
    self.assertEqual(dataclasses.replace(base, tokamax_gmm="tokamax_mosaic").tokamax_gmm, "tokamax_mosaic")
    with self.assertRaisesRegex(ValueError, "tokamax_gmm"):
      dataclasses.replace(base, tokamax_gmm="cutlass")

  def test_tiles(self):
    """128-row tiles at 4 stages where the groups average >= 768 buffer rows (DeepSeek-V3's 65536 over 64 local
    experts), else 64 at 6 (Qwen3.8's 61440 over 128): each shape's measured best on a GB200."""
    self.assertEqual(
        (gpu_kernels.mosaic_tiles(65536, 64)["block_m"], gpu_kernels.mosaic_tiles(65536, 64)["num_stages"]), (128, 4)
    )
    self.assertEqual(
        (gpu_kernels.mosaic_tiles(61440, 128)["block_m"], gpu_kernels.mosaic_tiles(61440, 128)["num_stages"]), (64, 6)
    )
    if not HAVE_TOKAMAX:
      self.skipTest("mosaic_config builds a tokamax Config: needs tokamax")
    cfg = gpu_kernels.mosaic_config(**gpu_kernels.mosaic_tiles(4096, 8))
    self.assertTrue(cfg.collective)
    self.assertEqual((cfg.block_n, cfg.block_k, cfg.grid_tile_width, cfg.persistent), (128, 64, 1, False))

  def test_argument_checks(self):
    a, w, gs = jnp.zeros((8, 64), jnp.bfloat16), jnp.zeros((2, 64, 128), jnp.bfloat16), jnp.zeros((2,), jnp.int32)
    with self.assertRaisesRegex(ValueError, "grouped_matmul"):
      gpu_kernels.grouped_matmul(a, jnp.zeros((2, 128, 64), jnp.bfloat16), gs)  # K mismatch
    with self.assertRaisesRegex(ValueError, "grouped_matmul"):
      gpu_kernels.grouped_matmul(a, w, gs, transpose_rhs=True)  # [G, N, K] wanted
    with self.assertRaisesRegex(ValueError, "grouped_matmul"):
      gpu_kernels.grouped_matmul(a, w, jnp.zeros((3,), jnp.int32))
    with self.assertRaisesRegex(ValueError, "bf16"):
      gpu_kernels.grouped_matmul(a.astype(jnp.float32), w.astype(jnp.float32), gs)

  def test_ragged_dw_off_gpu(self):
    """Off GPU ragged_dw is lax's ragged contraction plus the start, in fp32, rounded once; rows past the groups
    contribute nothing."""
    if jax.default_backend() == "gpu":
      self.skipTest("a GPU process")
    gs = group_sizes(5, 300, empty=(2,))
    lhs = jax.random.normal(jax.random.key(1), (400, 24), jnp.float32)
    g = jax.random.normal(jax.random.key(2), (400, 40), jnp.float32)
    acc = jax.random.normal(jax.random.key(3), (5, 24, 40), jnp.float32)
    got = gpu_kernels.ragged_dw(lhs, g, jnp.asarray(gs), acc)
    off = np.concatenate([[0], np.cumsum(gs)])
    want = jnp.stack([acc[e] + lhs[off[e] : off[e + 1]].T @ g[off[e] : off[e + 1]] for e in range(5)])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(acc[2]))  # the empty group keeps its start


@testing.requires_gpu(sm=10, why="the tcgen05 kernel (GB200 / B200)")
class GroupedMatmulOnBlackwell(unittest.TestCase):
  """The kernel against a per-group fp32 reference, at the pipeline's shapes and odd ones: bf16 rounding of the
  output is all that may differ (|err| <= 2^-8 (1 + |ref|): a bf16 half-ulp, the fp32 summation order's noise
  inside it), for uniform and skewed routings with empty groups, rows past the groups and a row count the kernel
  pads to its tiling; and dlhs equals tokamax's Mosaic kernel bit for bit (the same kernel on the same layout)."""

  SHAPES = (
      (64, 7168, 2048, 32768, 65536),  # DeepSeek-V3 at EP 4: groups, K, N, routed rows, buffer rows
      (128, 8192, 2048, 40960, 61440),  # Qwen3.8 at EP 4
      (4, 192, 640, 755, 1000),  # odd: K a multiple of 64 only, N not of 256, rows of nothing
      (5, 128, 256, 900, 1001),  # rows not a multiple of the tiling's 8 (padded)
      (3, 64, 128, 0, 256),
  )  # no rows routed at all

  def check(self, G, K, N, valid, rows, skew):
    gs = group_sizes(G, valid, skew=skew, empty=(0,) if G > 2 else ())
    v = int(gs.sum())
    x = jax.random.normal(jax.random.key(1), (rows, K), jnp.bfloat16)
    w = (jax.random.normal(jax.random.key(2), (G, K, N), jnp.float32) * 0.05).astype(jnp.bfloat16)
    gy = jax.random.normal(jax.random.key(3), (rows, N), jnp.bfloat16)
    fwd = jax.jit(lambda x, w, gs: gpu_kernels.grouped_matmul(x, w, gs))(x, w, jnp.asarray(gs))
    dlhs = jax.jit(lambda g, w, gs: gpu_kernels.grouped_matmul(g, w, gs, transpose_rhs=True))(gy, w, jnp.asarray(gs))
    self.assertEqual((fwd.shape, fwd.dtype, dlhs.shape), ((rows, N), jnp.bfloat16, (rows, K)))
    if v == 0:
      return
    for got, ref in ((fwd[:v], per_group(x, w, gs)), (dlhs[:v], per_group(gy, w, gs, transpose_rhs=True))):
      rel = jnp.abs(got.astype(jnp.float32) - ref) / (1.0 + jnp.abs(ref))
      self.assertLessEqual(float(jnp.max(rel)), 2.0**-8)
    if rows % 8:  # tokamax's own kernel cannot lower such a row count (its tiling, unpadded)
      return
    import tokamax

    dims = lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())), lhs_ragged_dimensions=[0], rhs_group_dimensions=[0]
    )
    tk = jax.jit(
        lambda g, w, gs: tokamax.ragged_dot_general(
            g, w.swapaxes(1, 2), gs, dims, preferred_element_type=jnp.bfloat16, implementation=("mosaic",)
        )
    )(gy, w, jnp.asarray(gs))
    self.assertTrue(bool(jnp.array_equal(dlhs[:v], tk[:v])))

  def test_shapes(self):
    for shape in self.SHAPES:
      for skew in (None, 0.3):
        with self.subTest(shape=shape, skew=skew):
          self.check(*shape, skew)

  def test_in_the_vjp(self):
    """ds_ep's tokamax_gmm under 'mosaic' (this kernel) against 'tokamax_mosaic' (tokamax's): the forward to bf16
    rounding, dlhs bitwise, dW (both on Triton, one kernel) bitwise."""
    G, K, N, valid, rows = 8, 256, 384, 900, 1024
    gs = jnp.asarray(group_sizes(G, valid, skew=0.5, empty=(3,)))
    valid_rows = (jnp.arange(rows) < int(gs.sum()))[:, None]
    x = jax.random.normal(jax.random.key(4), (rows, K), jnp.bfloat16)
    w = (jax.random.normal(jax.random.key(5), (G, K, N), jnp.float32) * 0.05).astype(jnp.bfloat16)
    ct = jax.random.normal(jax.random.key(6), (rows, N), jnp.bfloat16)

    def run(impl):
      f = lambda x, w: ds_ep.tokamax_gmm("gmm_t", impl, x, w, gs, valid_rows)
      out, vjp = jax.vjp(f, x, w)
      return (out,) + vjp(ct)

    mine, theirs = jax.jit(lambda: run("mosaic"))(), jax.jit(lambda: run("tokamax_mosaic"))()
    np.testing.assert_allclose(np.asarray(mine[0], np.float32), np.asarray(theirs[0], np.float32), rtol=2**-7, atol=2**-7)
    self.assertTrue(bool(jnp.array_equal(mine[1], theirs[1])))
    self.assertTrue(bool(jnp.array_equal(mine[2], theirs[2])))


class DwSides(unittest.TestCase):
  """The tcgen05 dW kernel's split of each group's rows (`_dw_sides`), on CPU: full block_k-row steps from the
  group's first 8-aligned row (the body: every TMA on a row the 8-row SMEM tiling divides) plus one or two steps
  of a zero-padded side buffer holding the rest -- which, summed as the kernel sums them, is each group's dW
  exactly (fp32 here): no row is lost, doubled or borrowed from a neighbour, for groups of 0, < 8, bk, bk + 1, ...
  rows and rows past the groups."""

  def test_steps_cover_each_group(self):
    bk = 64
    for gs in (
        [0, 3, 64, 65, 130, 7, 0, 200],
        [1, 1, 1, 9, 8, 16],
        [0, 0, 0],
        [500, 77, 8, 63, 129],
        list(group_sizes(16, 3000, seed=4, skew=0.4, empty=(2, 9))),
    ):
      with self.subTest(gs=gs[:8]):
        gs = np.asarray(gs, np.int32)
        rows = int(gs.sum()) + 13
        lhs = jax.random.normal(jax.random.key(1), (rows, 16), jnp.float32)
        g = jax.random.normal(jax.random.key(2), (rows, 8), jnp.float32)
        body, nbody, nsteps, side_l, side_g = (np.asarray(x) for x in gpu_kernels._dw_sides(lhs, g, jnp.asarray(gs), bk))
        self.assertEqual(side_l.shape, (len(gs) * 2 * bk, 16))
        off = np.concatenate([[0], np.cumsum(gs)])
        lhs, g = np.asarray(lhs), np.asarray(g)
        for e in range(len(gs)):
          self.assertEqual(body[e] % 8, 0)
          self.assertLessEqual(nsteps[e] - nbody[e], 2)
          dw = np.zeros((16, 8), np.float32)
          for ki in range(nsteps[e]):
            if ki < nbody[e]:
              r0 = body[e] + ki * bk
              self.assertGreaterEqual(r0, off[e])
              self.assertLessEqual(r0 + bk, off[e + 1])
              a, b = lhs[r0 : r0 + bk], g[r0 : r0 + bk]
            else:
              r0 = e * 2 * bk + (ki - nbody[e]) * bk
              a, b = side_l[r0 : r0 + bk], side_g[r0 : r0 + bk]
            dw += a.T @ b
          want = lhs[off[e] : off[e + 1]].T @ g[off[e] : off[e + 1]]
          np.testing.assert_allclose(dw, want, rtol=1e-5, atol=1e-4)
          self.assertEqual(nsteps[e] == 0, gs[e] == 0)

  def test_supported(self):
    """The tcgen05 dW takes bf16 / f16 operands, an accumulator of their dtype or fp32 (DW_MOSAIC_F32's tiles, unless
    MAXTEXT_MPMD_DW_MOSAIC_F32=0 turned MOSAIC_F32_ACC off), and acc dims multiples of its 256 x 256 cluster tile, on
    sm_100; and only tiles whose columns its epilogue's epi_chunk-column chunks cover (it adds whole chunks: the
    columns past them would be stored back without their dW), which the kernel itself refuses too -- into either
    accumulator."""
    a, g, acc = (
        jnp.zeros((8, 256), jnp.bfloat16),
        jnp.zeros((8, 512), jnp.bfloat16),
        jnp.zeros((2, 256, 512), jnp.bfloat16),
    )
    self.assertEqual(gpu_kernels.ragged_dw_mosaic_supported(a, g, acc), gpu_kernels.mosaic_supported())
    with mock.patch.object(gpu_kernels, "mosaic_supported", lambda device=None: True):
      self.assertTrue(gpu_kernels.ragged_dw_mosaic_supported(a, g, acc))
      self.assertTrue(
          gpu_kernels.ragged_dw_mosaic_supported(
              jnp.zeros((1001, 256), jnp.bfloat16), jnp.zeros((1001, 512), jnp.bfloat16), acc
          )
      )
      self.assertFalse(gpu_kernels.ragged_dw_mosaic_supported(a.astype(jnp.float32), g, acc))
      self.assertFalse(gpu_kernels.ragged_dw_mosaic_supported(a, g, acc.astype(jnp.float16)))
      # an fp32 accumulator (Megatron's main_grad) on DW_MOSAIC_F32's tiles, unless MOSAIC_F32_ACC is off: DW_MOSAIC's
      # at 3 stages -- its 128 KB epilogue tile and 3 x 32 KB of A / B tiles fill sm_100's 227 KB of SMEM, as the bf16
      # tile's 64 KB and 5 do (Mosaic GPU refuses a stage more of either at lowering: fp32's 4, 263308 > 232448 B)
      self.assertEqual(
          (gpu_kernels._dw_mosaic_cfg(jnp.bfloat16), gpu_kernels._dw_mosaic_cfg(jnp.float32)),
          (gpu_kernels.DW_MOSAIC, dict(gpu_kernels.DW_MOSAIC, num_stages=3)),
      )
      self.assertEqual(
          gpu_kernels._dw_mosaic_cfg(jnp.bfloat16, dict(block_n=64)), dict(gpu_kernels.DW_MOSAIC, block_n=64)
      )
      for f32 in (True, False):
        with mock.patch.object(gpu_kernels, "MOSAIC_F32_ACC", f32):
          self.assertEqual(gpu_kernels.ragged_dw_mosaic_supported(a, g, acc.astype(jnp.float32)), f32)
      self.assertFalse(
          gpu_kernels.ragged_dw_mosaic_supported(
              jnp.zeros((8, 128), jnp.bfloat16), g, jnp.zeros((2, 128, 512), jnp.bfloat16)
          )
      )
      self.assertFalse(
          gpu_kernels.ragged_dw_mosaic_supported(
              a, jnp.zeros((8, 384), jnp.bfloat16), jnp.zeros((2, 256, 384), jnp.bfloat16)
          )
      )
      self.assertTrue(
          gpu_kernels.ragged_dw_mosaic_supported(
              jnp.zeros((8, 128), jnp.bfloat16),
              g,
              jnp.zeros((2, 128, 512), jnp.bfloat16),
              cfg=dict(gpu_kernels.DW_MOSAIC, collective=False),
          )
      )
      a128, acc192 = jnp.zeros((8, 128), jnp.bfloat16), jnp.zeros((2, 256, 192), jnp.bfloat16)
      bad = {
          "192 columns": (a, jnp.zeros((8, 192), jnp.bfloat16), acc192, dict(block_n=96)),
          "64 columns": (
              a128,
              jnp.zeros((8, 64), jnp.bfloat16),
              jnp.zeros((2, 128, 64), jnp.bfloat16),
              dict(collective=False, block_n=64),
          ),
      }
      # 96-column chunks: a bf16 tile's 64-column SMEM tiling fails them at lowering, an fp32 tile's (32 columns)
      # lowers them -- 192 of 256 columns added -- so for an fp32 accumulator the gate is all that refuses them
      bad["96-column chunks"] = (a, g, acc, dict(epi_chunk=96))
      for (what, (lhs, gg, start, tiles)), dtype in itertools.product(bad.items(), (jnp.bfloat16, jnp.float32)):
        start = start.astype(dtype)  # the fp32 accumulator's tiles (DW_MOSAIC_F32) share the epilogue
        cfg = gpu_kernels._dw_mosaic_cfg(dtype, tiles)
        with self.subTest(what, acc=jnp.dtype(dtype).name), mock.patch.object(gpu_kernels, "MOSAIC_F32_ACC", True):
          self.assertFalse(gpu_kernels.ragged_dw_mosaic_supported(lhs, gg, start, cfg=cfg))
          self.assertTrue(gpu_kernels.ragged_dw_mosaic_supported(lhs, gg, start, cfg=dict(cfg, epi_chunk=64)))
          if HAVE_TOKAMAX:  # _dw_mosaic imports tokamax's Mosaic GPU helpers before it reads its tiles
            with self.assertRaisesRegex(ValueError, "epi_chunk"):
              gpu_kernels._dw_mosaic(lhs, gg, jnp.zeros((2,), jnp.int32), start, cfg)


@testing.requires_gpu()
class RaggedDwOnGpu(unittest.TestCase):
  """The in-place dW kernel against a per-group fp32 reference plus the start, one bf16 rounding (|err| <= 2^-8
  (1 + |ref|)) -- into an fp32 start (Megatron's main_grad; the tcgen05 kernel's DW_MOSAIC_F32 tiles;
  test_fp32_accumulator) only the two fp32 summation orders' noise (2^-14; ~2^-15.4 at these groups' sizes in a CPU
  model of the kernels' fp32 summation against XLA:CPU's reference, not measured on a GPU; one bf16 rounding reaches
  ~2^-8): the Triton kernel (`mosaic=False`) at tiles that divide the weight and ones that do not (masked), and the
  tcgen05 one -- which ragged_dw runs on sm_100, and which must take there every weight whose dims are multiples of
  its 256 x 256 cluster tile (into an fp32 start too, unless MOSAIC_F32_ACC is off) -- at both models' shapes,
  groups of 0, < 8, bk, bk + 1 rows, a row count it pads to its tiling, every group empty; empty groups keep their
  start exactly, rows past the groups are ignored."""

  CASES = (
      (8, 200, 72, None, 900, 1024),
      (5, 256, 384, None, 3000, 4096),
      (64, 7168, 2048, None, 32768, 65536),
      (128, 2048, 8192, None, 40960, 61440),
      (8, 256, 256, [0, 3, 64, 65, 130, 7, 0, 200], None, 512),
      (5, 256, 256, None, 999, 1001),
      (3, 256, 256, [0, 0, 0], None, 64),
      (6, 512, 256, [1, 1, 1, 9, 8, 16], None, 40),
  )

  def _against_reference(self, acc_dtype, tol):
    sm100 = gpu_kernels.mosaic_supported()
    for G, m, n, sizes, valid, rows in self.CASES:
      gs = np.asarray(sizes, np.int32) if sizes is not None else group_sizes(G, valid, seed=1, empty=(1,))
      lhs = jax.random.normal(jax.random.key(4), (rows, m), jnp.bfloat16)
      g = jax.random.normal(jax.random.key(5), (rows, n), jnp.bfloat16)
      acc = (jax.random.normal(jax.random.key(6), (G, m, n), jnp.float32) * 10).astype(acc_dtype)
      off = np.concatenate([[0], np.cumsum(gs)])
      ref = jnp.stack(
          [
              jnp.dot(lhs[off[e] : off[e + 1]].T, g[off[e] : off[e + 1]], preferred_element_type=jnp.float32)
              for e in range(G)
          ]
      ) + acc.astype(jnp.float32)
      kernels = [("triton", False)]
      if gpu_kernels.ragged_dw_mosaic_supported(lhs, g, acc):
        kernels.append(("tcgen05", None))
      elif sm100 and m % 256 == 0 and n % 256 == 0 and (acc.dtype == lhs.dtype or gpu_kernels.MOSAIC_F32_ACC):
        self.fail(f"the tcgen05 kernel does not take {(G, m, n)} with a {acc.dtype} accumulator")
      for name, mosaic in kernels:
        with self.subTest(shape=(G, m, n, rows), kernel=name, acc=acc.dtype.name):
          got = jax.jit(lambda *a, mo=mosaic: gpu_kernels.ragged_dw(*a, mosaic=mo))(lhs, g, jnp.asarray(gs), acc)
          self.assertEqual(got.dtype, acc.dtype)
          got = got.astype(jnp.float32)
          self.assertLessEqual(float(jnp.max(jnp.abs(got - ref) / (1 + jnp.abs(ref)))), tol)
          for e in np.flatnonzero(gs == 0):
            self.assertTrue(bool(jnp.array_equal(got[e], acc[e].astype(jnp.float32))), e)

  @testing.requires_gpu(*TRITON, why="the Triton kernel")
  def test_against_reference(self):
    self._against_reference(jnp.bfloat16, 2.0**-8)

  @testing.requires_gpu(*TRITON, why="the Triton kernel")
  def test_fp32_accumulator(self):
    """An fp32 accumulator (the dw32 path; on sm_100 the tcgen05 kernel at DW_MOSAIC_F32's tiles) on the same ragged
    groups: the products summed in fp32 onto the start and stored in fp32 -- within fp32 summation-order rounding
    of the reference (2^-14, where one bf16 rounding is up to 2^-8), empty groups keeping their start exactly."""
    self._against_reference(jnp.float32, 2.0**-14)

  @testing.requires_gpu(sm=10, why="the tcgen05 kernel")
  def test_tcgen05_in_place(self):
    """Donated, the tcgen05 dW updates the accumulator where it lies, a bf16 or (MOSAIC_F32_ACC) an fp32 one: no copy
    of it in the compiled program."""
    G, m, n = 8, 256, 512
    lhs, g = jnp.zeros((1024, m), jnp.bfloat16), jnp.zeros((1024, n), jnp.bfloat16)
    gs = jnp.asarray(group_sizes(G, 900, seed=2))
    f = jax.jit(gpu_kernels.ragged_dw, donate_argnums=(3,))
    for dtype in (jnp.bfloat16, jnp.float32) if gpu_kernels.MOSAIC_F32_ACC else (jnp.bfloat16,):
      with self.subTest(acc=jnp.dtype(dtype).name):
        hlo = f.lower(lhs, g, gs, jnp.zeros((G, m, n), dtype)).compile().as_text()
        self.assertIn("ragged_dw_sm100", hlo)
        self.assertFalse([l for l in hlo.splitlines() if (" copy(" in l or "copy-start" in l) and f"{G},{m},{n}" in l])


if __name__ == "__main__":
  unittest.main()
