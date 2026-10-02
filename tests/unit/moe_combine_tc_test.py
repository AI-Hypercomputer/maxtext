# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for moe_combine_tc (Pallas interpret mode on CPU, compiled on TPU)."""

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import moe_combine_tc


def _interpret(strict=True):
  if jax.default_backend() == "tpu":
    return False
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

  if strict:
    return pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
  return True


def _problem(seed, t, k, g, e, skew=0.0, wdtype=jnp.bfloat16):
  """Random routing (distinct experts per token), expert-sorted rows, weights."""
  rng = np.random.default_rng(seed)
  if skew > 0:
    p = np.exp(-skew * np.arange(g))
    p /= p.sum()
  else:
    p = np.full(g, 1.0 / g)
  experts = np.stack([rng.choice(g, size=k, replace=False, p=p) for _ in range(t)]).astype(np.int32)
  flat = jnp.asarray(experts.reshape(-1))
  sort_idx = jnp.argsort(flat)  # stable
  group_sizes = jnp.bincount(flat, length=g).astype(jnp.int32)
  x = jnp.asarray(rng.standard_normal((t * k, e)), jnp.float32).astype(jnp.bfloat16)
  w = jnp.asarray(rng.random((t, k)), jnp.float32).astype(wdtype)
  dy = jnp.asarray(rng.standard_normal((t, e)), jnp.float32).astype(jnp.bfloat16)
  return x, sort_idx, w, group_sizes, dy


def _assert_no_races(testcase):
  try:
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc  # pylint: disable=g-import-not-at-top
  except ImportError:
    return
  races = getattr(ipc, "races", None)
  if races is not None:
    testcase.assertFalse(races.races_found, "Pallas interpreter detected a DMA/VMEM race")


def _ref_grads(x, sort_idx, w, dy):
  _, vjp = jax.vjp(lambda a, b: moe_combine_tc.combine_reference(a, sort_idx, b), x, w)
  return vjp(dy)


class MoeCombineTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("base", 0, 256, 8, 16, 256, 128, 0.0),
      ("many_experts", 1, 256, 8, 128, 128, 128, 0.0),
      ("odd_groups_e384", 2, 384, 8, 9, 384, 128, 0.0),
      ("skewed_long_ranges", 3, 256, 8, 32, 128, 128, 1.0),
      ("k6_block256", 4, 512, 6, 12, 256, 256, 0.3),
      ("k2", 5, 256, 2, 5, 128, 128, 0.0),
      ("tiny_ranges_shared_granules", 8, 256, 2, 128, 128, 128, 0.0),
      ("tiny_ranges_align16", 9, 256, 2, 128, 128, 128, 0.0, 16),
      ("base_align16", 10, 256, 8, 16, 256, 128, 0.5, 16),
  )
  def test_fwd_bwd(self, seed, t, k, g, e, c, skew, align=8):
    self._check_fwd_bwd(seed, t, k, g, e, c, skew, align)

  def _check_fwd_bwd(self, seed, t, k, g, e, c, skew, align=8):
    x, sort_idx, w, gs, dy = _problem(seed, t, k, g, e, skew)
    f = functools.partial(moe_combine_tc.combine, block_tokens=c, interpret=_interpret(), align=align)
    y, vjp = jax.vjp(lambda a, b: f(a, sort_idx, b, gs), x, w)
    y_ref = moe_combine_tc.combine_reference(x, sort_idx, w)
    self.assertEqual(y.dtype, x.dtype)
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(y_ref, np.float32), rtol=1e-2, atol=1e-2)
    # f32 check of the forward sum before bf16 rounding differences: at most 1 bf16 ulp.
    diff = np.abs(np.asarray(y, np.float32) - np.asarray(y_ref, np.float32))
    ulp = np.abs(np.asarray(y_ref, np.float32)) * 2.0**-7 + 1e-6
    self.assertLessEqual(float(np.max(diff / ulp)), 1.01)

    dx, dw = vjp(dy)
    dx_ref, dw_ref = _ref_grads(x, sort_idx, w, dy)
    self.assertEqual(dx.dtype, x.dtype)
    self.assertEqual(dw.dtype, w.dtype)
    np.testing.assert_array_equal(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32))
    np.testing.assert_allclose(np.asarray(dw, np.float32), np.asarray(dw_ref, np.float32), rtol=1e-2, atol=1e-2)
    _assert_no_races(self)

  @parameterized.named_parameters(("pure_local", 0.0), ("mixed", 0.02))
  def test_adjacent_ranges_in_block(self, mix):
    """Experts mostly private to a token block -> many same-block shared boundary pairs."""
    t, k, g, e, c = 512, 4, 32, 128, 128
    rng = np.random.default_rng(21)
    per_block = g // (t // c)
    rows = []
    for tok in range(t):
      if rng.random() < mix:
        rows.append(rng.choice(g, size=k, replace=False))
      else:
        b = tok // c
        rows.append(b * per_block + rng.choice(per_block, size=k, replace=False))
    flat = jnp.asarray(np.stack(rows).astype(np.int32).reshape(-1))
    sort_idx = jnp.argsort(flat)
    gs = jnp.bincount(flat, length=g).astype(jnp.int32)
    x = jnp.asarray(rng.standard_normal((t * k, e)), jnp.float32).astype(jnp.bfloat16)
    w = jnp.asarray(rng.random((t, k)), jnp.float32).astype(jnp.bfloat16)
    dy = jnp.asarray(rng.standard_normal((t, e)), jnp.float32).astype(jnp.bfloat16)
    tab = moe_combine_tc._metadata(sort_idx, gs, t, k, c)  # pylint: disable=protected-access
    ntab = moe_combine_tc._NTAB  # pylint: disable=protected-access
    tab = np.asarray(tab).reshape(t // c, -1)[:, : ntab * g].reshape(t // c, ntab, g)
    # Granules shared by two ranges of the same block (tail of one == head of the next).
    shared = sum(
        len(set(tab[bb, 5][tab[bb, 5] >= 0]) & set(tab[bb, 4][tab[bb, 4] >= 0])) for bb in range(t // c)
    )
    self.assertGreater(shared, 0)
    f = functools.partial(moe_combine_tc.combine, block_tokens=c, interpret=_interpret())
    y, vjp = jax.vjp(lambda a, b: f(a, sort_idx, b, gs), x, w)
    y_ref = moe_combine_tc.combine_reference(x, sort_idx, w)
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(y_ref, np.float32), rtol=1e-2, atol=1e-2)
    dx, dw = vjp(dy)
    dx_ref, dw_ref = _ref_grads(x, sort_idx, w, dy)
    np.testing.assert_array_equal(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32))
    np.testing.assert_allclose(np.asarray(dw, np.float32), np.asarray(dw_ref, np.float32), rtol=1e-2, atol=1e-2)
    _assert_no_races(self)

  def test_f32_weights_split(self):
    x, sort_idx, w, gs, dy = _problem(7, 256, 8, 16, 128, 0.0, wdtype=jnp.float32)
    f = functools.partial(moe_combine_tc.combine, block_tokens=128, interpret=_interpret())
    y, vjp = jax.vjp(lambda a, b: f(a, sort_idx, b, gs), x, w)
    y_ref = moe_combine_tc.combine_reference(x, sort_idx, w)
    np.testing.assert_allclose(np.asarray(y, np.float32), np.asarray(y_ref, np.float32), rtol=1e-2, atol=1e-2)
    dx, dw = vjp(dy)
    dx_ref, dw_ref = _ref_grads(x, sort_idx, w, dy)
    np.testing.assert_allclose(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32), rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(dw_ref), rtol=1e-4, atol=1e-4)
    # The lo part must be used: y matches the f32-weight reference far better
    # than the bf16-rounded-weight one (bf16 output rounding flips aside).
    y_bw = moe_combine_tc.combine_reference(x, sort_idx, w.astype(jnp.bfloat16))
    f = lambda a: np.asarray(a, np.float32)
    self.assertLess(int(np.sum(f(y) != f(y_ref))) * 20, int(np.sum(f(y) != f(y_bw))))

  def test_check_grads_and_jit_scan_remat(self):
    # Off-TPU this needs the legacy HLO interpreter (the Mosaic interpreter's
    # callbacks are not allowed under remat), which has no dynamic-size DMAs.
    old = os.environ.get("MAXTEXT_G4_COMBINE_DYNDMA")
    if jax.default_backend() != "tpu":
      os.environ["MAXTEXT_G4_COMBINE_DYNDMA"] = "0"
    try:
      self._check_scan_remat()
    finally:
      if old is None:
        os.environ.pop("MAXTEXT_G4_COMBINE_DYNDMA", None)
      else:
        os.environ["MAXTEXT_G4_COMBINE_DYNDMA"] = old

  def _check_scan_remat(self):
    x, sort_idx, w, gs, _ = _problem(11, 256, 8, 16, 128, 0.5)
    f = functools.partial(moe_combine_tc.combine, block_tokens=128, interpret=_interpret(strict=False))

    def loss(fn, a, b):
      def step(carry, _):
        y = jax.checkpoint(lambda aa, bb: fn(aa, sort_idx, bb, gs))(a, b)
        return carry + jnp.sum(y.astype(jnp.float32) ** 2), None

      out, _ = jax.lax.scan(step, 0.0, None, length=2)
      return out

    ref = jax.jit(jax.value_and_grad(functools.partial(loss, lambda a, s, b, g_: moe_combine_tc.combine_reference(a, s, b)), argnums=(0, 1)))
    got = jax.jit(jax.value_and_grad(functools.partial(loss, f), argnums=(0, 1)))
    (lv_r, (gx_r, gw_r)), (lv, (gx, gw)) = ref(x, w), got(x, w)
    np.testing.assert_allclose(float(lv), float(lv_r), rtol=1e-3)
    np.testing.assert_allclose(np.asarray(gx, np.float32), np.asarray(gx_r, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(gw, np.float32), np.asarray(gw_r, np.float32), rtol=2e-2, atol=5e-1)

  @parameterized.named_parameters(
      ("static_chunks", {"MAXTEXT_G4_COMBINE_DYNDMA": "0"}),
      ("merge_f32", {"MAXTEXT_G4_COMBINE_MERGE": "f32"}),
      ("bucket256_en128", {"MAXTEXT_G4_COMBINE_BUCKET": "256", "MAXTEXT_G4_COMBINE_EN": "128"}),
      ("bwd_v4", {"MAXTEXT_G4_COMBINE_BWD": "v4"}),
      ("bwd_v5_rq512", {"MAXTEXT_G4_COMBINE_BWD": "v5", "MAXTEXT_G4_COMBINE_BWD_RQ": "512"}),
      ("rowinfo_iota", {"MAXTEXT_G4_ROWINFO_IOTA": "1"}),
  )
  def test_env_variants(self, env):
    old = {kk: os.environ.get(kk) for kk in env}
    os.environ.update(env)
    try:
      self._check_fwd_bwd(12, 256, 8, 32, 256, 128, 0.3)
    finally:
      for kk, v in old.items():
        if v is None:
          os.environ.pop(kk, None)
        else:
          os.environ[kk] = v

  def test_metadata_table_matches_argsort_ranges(self):
    t, k, g, c = 512, 8, 32, 128
    _, sort_idx, _, gs, _ = _problem(13, t, k, g, 128, 0.7)
    tab = moe_combine_tc._metadata(sort_idx, gs, t, k, c)  # pylint: disable=protected-access
    ntab = moe_combine_tc._NTAB  # pylint: disable=protected-access
    tab = np.asarray(tab).reshape(t // c, -1)[:, : ntab * g].reshape(t // c, ntab, g)
    si = np.asarray(sort_idx)
    ends = np.cumsum(np.asarray(gs))
    grow = np.searchsorted(ends, np.arange(t * k), side="right")
    blk = si // (k * c)
    used = 0
    for b in range(t // c):
      rows = np.nonzero(blk == b)[0]
      covered = set()
      for gi in range(g):
        rs, nl = tab[b, 0, gi], tab[b, 1, gi]
        covered |= set(range(rs, rs + nl))
      self.assertTrue(set(rows.tolist()) <= covered)
      used = max(used, len(covered))
      del grow
      grow = None
    self.assertLessEqual(used, moe_combine_tc._buffer_rows(c, k, g))  # pylint: disable=protected-access

  def test_fallback_unsupported_shape(self):
    x, sort_idx, w, gs, _ = _problem(3, 100, 8, 16, 128)
    y = moe_combine_tc.combine(x, sort_idx, w, gs, block_tokens=128, interpret=True)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(moe_combine_tc.combine_reference(x, sort_idx, w)))


if __name__ == "__main__":
  absltest.main()
