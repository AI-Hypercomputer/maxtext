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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels import moe_combine_tc


def _interpret(strict=True):
  """Interpret setting: compiled on TPU; otherwise the Mosaic interpreter with race detection."""
  if jax.default_backend() == "tpu":
    return False
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=g-import-not-at-top

  if strict:
    return pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
  return True


def _sorted_problem(rng, experts, t, k, g, e, wdtype=jnp.bfloat16):
  """Expert-sorted rows, stable sort index, group sizes, weights and dy for a [T, K] routing into G experts."""
  flat = jnp.asarray(experts.astype(np.int32).reshape(-1))
  sort_idx = jnp.argsort(flat)  # stable
  group_sizes = jnp.bincount(flat, length=g).astype(jnp.int32)
  x = jnp.asarray(rng.standard_normal((t * k, e)), jnp.float32).astype(jnp.bfloat16)
  w = jnp.asarray(rng.random((t, k)), jnp.float32).astype(wdtype)
  dy = jnp.asarray(rng.standard_normal((t, e)), jnp.float32).astype(jnp.bfloat16)
  return x, sort_idx, w, group_sizes, dy


def _problem(seed, t, k, g, e, skew=0.0, wdtype=jnp.bfloat16):
  """Random routing (distinct experts per token, optionally skewed towards low expert ids)."""
  rng = np.random.default_rng(seed)
  if skew > 0:
    p = np.exp(-skew * np.arange(g))
    p /= p.sum()
  else:
    p = np.full(g, 1.0 / g)
  experts = np.stack([rng.choice(g, size=k, replace=False, p=p) for _ in range(t)])
  return _sorted_problem(rng, experts, t, k, g, e, wdtype)


def _block_local_problem(seed, t, k, g, e, c, mix):
  """Routing where experts are mostly private to one token block of `c` tokens.

  Consecutive experts of a block then have adjacent row ranges (shared boundary
  granules); with `mix` > 0 a few tokens route anywhere, so some experts also
  span consecutive blocks.
  """
  rng = np.random.default_rng(seed)
  per_block = g // (t // c)
  rows = []
  for tok in range(t):
    if rng.random() < mix:
      rows.append(rng.choice(g, size=k, replace=False))
    else:
      rows.append((tok // c) * per_block + rng.choice(per_block, size=k, replace=False))
  return _sorted_problem(rng, np.stack(rows), t, k, g, e)


def _metadata_table(sort_idx, group_sizes, t, k, c):
  """Per-block scalar table as a [nb, _NTAB, G] numpy array."""
  g = group_sizes.shape[0]
  tab = moe_combine_tc._metadata(sort_idx, group_sizes, t, k, c)  # pylint: disable=protected-access
  ntab = moe_combine_tc._NTAB  # pylint: disable=protected-access
  return np.asarray(tab).reshape(t // c, -1)[:, : ntab * g].reshape(t // c, ntab, g)


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


def _f32(a):
  return np.asarray(a, np.float32)


class MoeCombineTest(parameterized.TestCase):

  def _check_combine(self, combine_fn, x, sort_idx, w, gs, dy, *, bf16_weights=True):
    """Forward and backward against the reference.

    With bf16 weights every product is exact, so y is within 1 bf16 ulp of the
    f32 reference and dx is bit-exact; non-bf16 weights go through the bf16
    hi + lo split and are checked with tolerances.
    """
    y, vjp = jax.vjp(lambda a, b: combine_fn(a, sort_idx, b, gs), x, w)
    y_ref = moe_combine_tc.combine_reference(x, sort_idx, w)
    self.assertEqual(y.dtype, x.dtype)
    np.testing.assert_allclose(_f32(y), _f32(y_ref), rtol=1e-2, atol=1e-2)
    if bf16_weights:
      ulp = np.abs(_f32(y_ref)) * 2.0**-7 + 1e-6
      self.assertLessEqual(float(np.max(np.abs(_f32(y) - _f32(y_ref)) / ulp)), 1.01)

    dx, dw = vjp(dy)
    dx_ref, dw_ref = _ref_grads(x, sort_idx, w, dy)
    self.assertEqual(dx.dtype, x.dtype)
    self.assertEqual(dw.dtype, w.dtype)
    if bf16_weights:
      np.testing.assert_array_equal(_f32(dx), _f32(dx_ref))
    else:
      np.testing.assert_allclose(_f32(dx), _f32(dx_ref), rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(_f32(dw), _f32(dw_ref), rtol=1e-2, atol=1e-2)
    _assert_no_races(self)
    return y, y_ref, dw

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
    x, sort_idx, w, gs, dy = _problem(seed, t, k, g, e, skew)
    f = functools.partial(moe_combine_tc.combine, block_tokens=c, interpret=_interpret(), align=align)
    self._check_combine(f, x, sort_idx, w, gs, dy)

  @parameterized.named_parameters(("pure_local", 0.0), ("mixed", 0.02))
  def test_adjacent_ranges_in_block(self, mix):
    """Adjacent expert ranges within a block and (mixed) experts spanning consecutive blocks."""
    t, k, g, e, c = 512, 4, 32, 128, 128
    x, sort_idx, w, gs, dy = _block_local_problem(21, t, k, g, e, c, mix)
    tab = _metadata_table(sort_idx, gs, t, k, c)
    hs, ts = tab[:, moe_combine_tc._T_HS], tab[:, moe_combine_tc._T_TS]  # pylint: disable=protected-access
    hcr, tcr = tab[:, moe_combine_tc._T_HCR], tab[:, moe_combine_tc._T_TCR]  # pylint: disable=protected-access
    staged_any = int(np.sum((hs >= 0) | (ts >= 0) | (hcr >= 0) | (tcr >= 0)))
    if mix > 0:
      # Partial granules holding rows of earlier blocks are staged from HBM or,
      # for experts spanning consecutive blocks, carried from the previous
      # block's output buffer.
      self.assertGreater(staged_any, 0)
      self.assertGreater(int(np.sum(hcr >= 0)), 0)
    else:
      # 64 rows per expert: every range boundary is aligned, nothing is staged.
      self.assertEqual(staged_any, 0)
    for b in range(t // c):  # a granule is never staged twice within a block
      staged = [v for v in np.concatenate([hs[b], ts[b]]) if v >= 0]
      self.assertEqual(len(staged), len(set(staged)))
    f = functools.partial(moe_combine_tc.combine, block_tokens=c, interpret=_interpret())
    self._check_combine(f, x, sort_idx, w, gs, dy)

  def test_f32_weights_split(self):
    x, sort_idx, w, gs, dy = _problem(7, 256, 8, 16, 128, 0.0, wdtype=jnp.float32)
    f = functools.partial(moe_combine_tc.combine, block_tokens=128, interpret=_interpret())
    y, y_ref, dw = self._check_combine(f, x, sort_idx, w, gs, dy, bf16_weights=False)
    _, dw_ref = _ref_grads(x, sort_idx, w, dy)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(dw_ref), rtol=1e-4, atol=1e-4)
    # The lo part must be used: y matches the f32-weight reference far better
    # than the bf16-rounded-weight one (bf16 output rounding flips aside).
    y_bw = moe_combine_tc.combine_reference(x, sort_idx, w.astype(jnp.bfloat16))
    self.assertLess(int(np.sum(_f32(y) != _f32(y_ref))) * 20, int(np.sum(_f32(y) != _f32(y_bw))))

  def test_check_grads_and_jit_scan_remat(self):
    # Off-TPU this needs the legacy HLO interpreter (`interpret=True`): the
    # Mosaic interpreter's callbacks are not allowed under remat.
    x, sort_idx, w, gs, _ = _problem(11, 256, 8, 16, 128, 0.5)
    f = functools.partial(moe_combine_tc.combine, block_tokens=128, interpret=_interpret(strict=False))

    def loss(fn, a, b):
      def step(carry, _):
        y = jax.checkpoint(lambda aa, bb: fn(aa, sort_idx, bb, gs))(a, b)
        return carry + jnp.sum(y.astype(jnp.float32) ** 2), None

      out, _ = jax.lax.scan(step, 0.0, None, length=2)
      return out

    ref = jax.jit(
        jax.value_and_grad(
            functools.partial(loss, lambda a, s, b, g_: moe_combine_tc.combine_reference(a, s, b)), argnums=(0, 1)
        )
    )
    got = jax.jit(jax.value_and_grad(functools.partial(loss, f), argnums=(0, 1)))
    (lv_r, (gx_r, gw_r)), (lv, (gx, gw)) = ref(x, w), got(x, w)
    np.testing.assert_allclose(float(lv), float(lv_r), rtol=1e-3)
    np.testing.assert_allclose(_f32(gx), _f32(gx_r), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(_f32(gw), _f32(gw_r), rtol=2e-2, atol=5e-1)

  def test_metadata_table_matches_argsort_ranges(self):
    """Every row of a token block lies in one of the block's windows, within the static buffer bound."""
    t, k, g, c = 512, 8, 32, 128
    _, sort_idx, _, gs, _ = _problem(13, t, k, g, 128, 0.7)
    tab = _metadata_table(sort_idx, gs, t, k, c)
    blk = np.asarray(sort_idx) // (k * c)
    used = 0
    for b in range(t // c):
      rows = np.nonzero(blk == b)[0]
      covered = set()
      for gi in range(g):
        rs, nl = tab[b, moe_combine_tc._T_RS, gi], tab[b, moe_combine_tc._T_NL, gi]  # pylint: disable=protected-access
        covered |= set(range(rs, rs + nl))
      self.assertTrue(set(rows.tolist()) <= covered)
      used = max(used, len(covered))
    self.assertLessEqual(used, moe_combine_tc._buffer_rows(c, k, g))  # pylint: disable=protected-access

  def test_fallback_unsupported_shape(self):
    x, sort_idx, w, gs, _ = _problem(3, 100, 8, 16, 128)
    y = moe_combine_tc.combine(x, sort_idx, w, gs, block_tokens=128, interpret=True)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(moe_combine_tc.combine_reference(x, sort_idx, w)))


if __name__ == "__main__":
  absltest.main()
