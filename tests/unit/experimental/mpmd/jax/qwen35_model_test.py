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

"""CPU tests for the Qwen3.5 port (`src/maxtext/experimental/mpmd/jax/qwen35/`): the gated delta rule's
numerics, the RMSNorms' two roundings (`norm_numerics`), the parameter tree and
init, the config, and where the sharding-sensitive call sites live.

    JAX_PLATFORMS=cpu python3 -m pytest -q tests/unit/experimental/mpmd/jax/qwen35_model_test.py
    JAX_PLATFORMS=cpu python3 -m unittest tests.unit.experimental.mpmd.jax.qwen35_model_test   # stacks without pytest

The default set is cheap (a few seconds: nothing here compiles a whole model).
The whole-model checks that do (the three remat layouts' gradients agree,
scanned == unrolled, train steps reduce the loss) are marked scheduled_only.  The port's
parity with MaxText's own `qwen3_5` model is `qwen35_maxtext_parity_test.py`.
"""

import ast
import dataclasses
import functools
import inspect
import pathlib
import re
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.experimental.mpmd.jax.deepseek import attention as ds_attention
from maxtext.experimental.mpmd.jax.deepseek import layers as ds_layers
from maxtext.experimental.mpmd.jax.deepseek import model as ds_model
from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe
from maxtext.experimental.mpmd.jax.deepseek.config import SHARED_LAYER_FIELDS, DeepSeekConfig, SharedLayerConfig
from maxtext.experimental.mpmd.jax.qwen35 import attention as qa
from maxtext.experimental.mpmd.jax.qwen35 import config as qc
from maxtext.experimental.mpmd.jax.qwen35 import convert
from maxtext.experimental.mpmd.jax.qwen35 import gdn
from maxtext.experimental.mpmd.jax.qwen35 import init as qs
from maxtext.experimental.mpmd.jax.qwen35 import layers
from maxtext.experimental.mpmd.jax.qwen35 import model as qm
from maxtext.experimental.mpmd.jax.qwen35 import train as qt

QWEN35 = pathlib.Path(qm.__file__).resolve().parent


def small(**kw):
  """fp32, every structural feature, tiny widths; a 2-layer [GDN, attention] block
  by default (half the compile time of the real 4-layer cycle, same code)."""
  base = dict(
      vocab_size=64,
      emb_dim=32,
      num_decoder_layers=8,
      inhomogeneous_layer_cycle_interval=2,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=32,
      mrope_section=(2, 1, 1),
      gdn_num_key_heads=2,
      gdn_num_value_heads=4,
      gdn_key_head_dim=8,
      gdn_value_head_dim=8,
      gdn_chunk_size=8,
      moe_mlp_dim=16,
      num_experts=4,
      num_experts_per_tok=2,
      max_target_length=20,
      dtype=jnp.float32,
      attention_kernel="reference",
  )
  base.update(kw)
  return qc.Qwen35Config(**base)


def realize(avals, seed=0):
  """Host-side random weights with the init's structure rules (no init compile;
  cast on the host too: `jnp.asarray(x, dtype)` compiles a convert per shape)."""
  rng = np.random.default_rng(seed)

  def leaf(path, a):
    keys = tuple(k.key for k in path)
    if keys[-1] == "scale":
      v = rng.standard_normal(a.shape) * 0.1
    elif keys[-1] == "A_log":
      v = np.log(rng.uniform(0.5, 8.0, a.shape))
    else:
      v = rng.standard_normal(a.shape) * qs.fan_in(keys, a.shape) ** -0.5
    return jax.device_put(np.asarray(v, a.dtype))

  return jax.tree_util.tree_map_with_path(leaf, avals)


def rel_rms(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b**2)))


def max_rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def flat(tree):
  return {jax.tree_util.keystr(p): v for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


# --------------------------------------------------------------------------- #
# The gated delta rule
# --------------------------------------------------------------------------- #
def gdn_inputs(b=2, s=37, h=3, dk=8, dv=12, seed=0):
  """Random core inputs with MaxText's gate statistics (A ~ U(0, 16], dt_bias 1)."""
  rng = np.random.default_rng(seed)
  q, k = rng.standard_normal((b, s, h, dk)), rng.standard_normal((b, s, h, dk))
  v = rng.standard_normal((b, s, h, dv))
  a_log = np.log(rng.uniform(1e-3, 16.0, size=(h,)))
  g = -np.exp(a_log) * np.log1p(np.exp(rng.standard_normal((b, s, h)) + 1.0))
  beta = 1 / (1 + np.exp(-rng.standard_normal((b, s, h))))
  # A few masked positions (segment id 0): k = v = 0, g = 0 -- no decay, no update.
  k[0, 5:9], v[0, 5:9], g[0, 5:9] = 0.0, 0.0, 0.0
  return q, k, v, g, beta


@functools.cache
def _chunked_rule(dtype, chunk, qk_norm):
  return jax.jit(lambda *a: gdn.chunk_gated_delta_rule(*a, chunk, qk_norm, dtype))


class GatedDeltaRuleTest(unittest.TestCase):

  def _chunked(self, args, dtype, chunk=8, qk_norm=True):
    return _chunked_rule(dtype, chunk, qk_norm)(*(jnp.asarray(np.asarray(x, dtype)) for x in args))

  def test_chunked_rule_is_the_token_recurrence_fp32(self):
    """The chunked WY algorithm == the definitional recurrence (float64 oracle).

    S = 37 is not a multiple of the chunk (the padding path), Dk != Dv, some
    tokens masked.  (At the real head dims, 128, it is ~5e-7 rel-RMS.)
    """
    args = gdn_inputs()
    truth = gdn.recurrent_gated_delta_rule(*args)
    got = self._chunked(args, jnp.float32)
    self.assertEqual(got.shape, truth.shape)
    self.assertEqual(got.dtype, jnp.float32)
    self.assertLess(rel_rms(got, truth), 2e-6)
    # without the q/k l2 norm too, and with a chunk longer than the sequence
    truth = gdn.recurrent_gated_delta_rule(*args, use_qk_norm=False)
    self.assertLess(rel_rms(self._chunked(args, jnp.float32, chunk=64, qk_norm=False), truth), 2e-6)
    # negative control: a wrong decay must not pass
    q, k, v, g, beta = args
    wrong = gdn.recurrent_gated_delta_rule(q, k, v, 0.5 * g, beta)
    self.assertGreater(rel_rms(self._chunked(args, jnp.float32), wrong), 1e-2)

  def test_chunked_rule_bf16_is_sane(self):
    """bf16 inputs and compute (MaxText's training dtype): ~6e-3 rel-RMS at the real dims."""
    args = gdn_inputs(seed=1)
    self.assertLess(rel_rms(self._chunked(args, jnp.bfloat16), gdn.recurrent_gated_delta_rule(*args)), 2e-2)

  def test_unit_lower_inverse(self):
    """'solve': the forward is MaxText's solve_triangular, bitwise; 'block_doubling' (MaxText's default from fb42453e1
    on, and unit_lower_inv's; bitwise MaxText's in qwen35_maxtext_parity_test) inverts I + S too, at any chunk, to
    fp32 rounding of the solve's.  The custom VJP of either is jax.scipy's VJP of the solve, and strictly lower."""
    rng = np.random.default_rng(0)
    for c in (16, 12):
      s = jnp.asarray(np.tril(rng.standard_normal((2, 3, 2, c, c)) * 0.3, -1), jnp.float32)
      ct = jnp.asarray(rng.standard_normal(s.shape), jnp.float32)
      eye = jnp.eye(c, dtype=jnp.float32)

      def scipy_inv(s, eye=eye):
        return jax.scipy.linalg.solve_triangular(eye + s, jnp.broadcast_to(eye, s.shape), lower=True, unit_diagonal=True)

      ref = jax.jit(scipy_inv)(s)
      g_ref = jax.jit(jax.grad(lambda s: jnp.sum(scipy_inv(s) * ct)))(s)
      for method in qc.GDN_CHUNK_INVERSES:
        with self.subTest(chunk=c, method=method):
          a = jax.jit(gdn.unit_lower_inv, static_argnums=1)(s, method)
          if method == "solve":
            self.assertTrue(bool(jnp.array_equal(a, ref)))
          else:
            self.assertTrue(bool(jnp.array_equal(a, jax.jit(gdn.unit_lower_inv)(s))))  # the default
            self.assertFalse(bool(jnp.array_equal(a, ref)))  # another computation...
            self.assertLess(max_rel(a, ref), 1e-6)  # ...of the same inverse
          np.testing.assert_allclose(np.asarray(a) @ np.asarray(eye + s), np.broadcast_to(np.eye(c), s.shape), atol=1e-5)
          g_new = jax.jit(jax.grad(lambda s, m=method: jnp.sum(gdn.unit_lower_inv(s, m) * ct)))(s)
          self.assertLess(max_rel(g_new, g_ref), 1e-5)
          # S enters strictly-lower only: its cotangent is zero on and above the diagonal
          upper = np.triu(np.ones((c, c), bool))
          self.assertEqual(float(np.abs(np.asarray(g_new)[..., upper]).max()), 0.0)
    with self.assertRaisesRegex(ValueError, "method"):
      gdn.unit_lower_inv(s, "neumann")

  def test_grouped_heads(self):
    """Grouped heads (MaxText's GDN: k k^T at the key heads, beta and the decay on it in fp32, the heads repeated
    inside the rule): the float64 recurrence on the repeated heads, to fp32 rounding at 2 key heads x 3 value heads
    each, both inverses (bf16 to its rounding); key heads that do not divide the value heads are refused."""
    q, k, v, g, beta = gdn_inputs(h=6, seed=4)
    q, k = q[:, :, ::3], k[:, :, ::3]  # 2 key heads; value head j reads key head j // 3
    truth = gdn.recurrent_gated_delta_rule(np.repeat(q, 3, axis=2), np.repeat(k, 3, axis=2), v, g, beta)
    for dt in (jnp.float32, jnp.bfloat16):
      for inverse in qc.GDN_CHUNK_INVERSES:
        f = jax.jit(lambda *a, inverse=inverse: gdn.chunk_gated_delta_rule(*a, 8, True, dt, chunk_inverse=inverse))
        o = f(*(jnp.asarray(np.asarray(x, dt)) for x in (q, k, v, g, beta)))
        with self.subTest(dtype=jnp.dtype(dt).name, chunk_inverse=inverse):
          self.assertLess(rel_rms(o, truth), 2e-6 if dt == jnp.float32 else 2e-2)
    q4, k4 = (np.concatenate([x, x], axis=2) for x in (q, k))  # 4 key heads for 6 value heads
    with self.assertRaisesRegex(ValueError, "value heads 6 vs key heads 4"):
      gdn.chunk_gated_delta_rule(*(jnp.asarray(x, jnp.float32) for x in (q4, k4, v, g, beta)), 8, True, jnp.float32)

  def test_canonical_segment_ids(self):
    """MaxText's canonicalisation: a new document wherever a valid token's id differs from the last valid one's,
    padding (<= 0) kept 0 and splitting nothing."""
    seg = jnp.asarray([[1, 1, 2, 2, 1, 0, 1, 3], [0, 0, 5, 5, 0, 5, 7, -1], [4, 4, 4, 4, 4, 4, 4, 4]], jnp.int32)
    want = [[1, 1, 2, 2, 3, 0, 3, 4], [0, 0, 1, 1, 0, 1, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    np.testing.assert_array_equal(np.asarray(gdn.canonical_segment_ids(seg)), want)

  def test_packed_documents_restart_the_rule(self):
    """With segment ids (gdn_packing) the chunked rule is the float64 recurrence run on each document alone, zero
    on padding: documents across chunk edges, padding at the start, inside a document (which it does not split)
    and at the end, an id that comes back as a new document, S = 37 (the padding path), 2 key heads x 3 value heads
    and both inverses; without segment ids the state runs on across documents (another result)."""
    q, k, v, g, beta = gdn_inputs(h=6, seed=5)
    q, k = q[:, :, ::3], k[:, :, ::3]
    seg = np.zeros((2, 37), np.int32)
    seg[0] = [1] * 5 + [2] * 12 + [0] * 3 + [2] * 4 + [3] * 13
    seg[1] = [0] * 4 + [1] * 9 + [2] * 10 + [1] * 8 + [0] * 6
    canon = np.asarray(gdn.canonical_segment_ids(jnp.asarray(seg)))
    truth = np.zeros(v.shape)
    for row in range(2):
      for d in set(canon[row]) - {0}:
        idx = np.nonzero(canon[row] == d)[0]
        part = [x[row : row + 1, idx] for x in (np.repeat(q, 3, axis=2), np.repeat(k, 3, axis=2), v, g, beta)]
        truth[row, idx] = gdn.recurrent_gated_delta_rule(*part)[0]
    ins = [jnp.asarray(x, jnp.float32) for x in (q, k, v, g, beta)]
    for inverse in qc.GDN_CHUNK_INVERSES:
      with self.subTest(chunk_inverse=inverse):
        rule = jax.jit(
            lambda *a, s=None, inverse=inverse: gdn.chunk_gated_delta_rule(
                *a, 8, True, jnp.float32, segment_ids=s, chunk_inverse=inverse
            )
        )
        got = rule(*ins, s=jnp.asarray(seg))
        self.assertLess(rel_rms(got, truth), 2e-6)
        self.assertEqual(float(jnp.max(jnp.abs(got[jnp.asarray(canon == 0)]))), 0.0)
        self.assertGreater(rel_rms(rule(*ins), truth), 1e-2)

  def test_segmented_conv(self):
    """`segmented_causal_conv` (gdn_packing) is the causal conv within each document -- output t sums the taps
    t-K+1..t of its own document only (positions, not tokens: padding inside a document is a zero tap) -- in fp32
    with the fp32 kernel, rounded once, zero on padding; 'shift' is the same sum to fp32 rounding (bitwise in bf16
    here).  In bf16 that is within one rounding (2^-8, relative) of the exact sum over the bf16 input, give or take
    the fp32 sum's own error (2^-20, absolute) where the taps cancel; with the kernel rounded to bf16 first, as the
    plain conv's is (`qwen35_maxtext_parity_test`'s PACKED_CONV_MUTANT), it is not.  Without documents it is
    `causal_depthwise_conv` on the masked input, to fp32 rounding."""
    rng = np.random.default_rng(6)
    b, s, ch, k = 2, 19, 12, 4
    x = jnp.asarray(rng.standard_normal((b, s, ch)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((k, 1, ch)), jnp.float32)
    seg = np.ones((b, s), np.int32)
    seg[0, 6:] = 2
    seg[0, 9] = 0  # padding inside document 2
    seg[0, 14:] = 1  # id 1 again: a third document
    seg[1, :3] = 0
    seg[1, 16:] = 0
    canon = np.asarray(gdn.canonical_segment_ids(jnp.asarray(seg)))
    ws = np.asarray(w, np.float64)[:, 0]

    def definition(x):  # the sum in float64, of the input the port gets
      xs, want = np.asarray(x, np.float64), np.zeros((b, s, ch))
      for r in range(b):
        for t in range(s):
          if canon[r, t]:
            for i in range(k):
              u = t - (k - 1) + i
              if u >= 0 and canon[r, u] == canon[r, t]:
                want[r, t] += ws[i] * xs[r, u]
      return want

    got = {}
    for dt in (jnp.float32, jnp.bfloat16):
      want = definition(x.astype(dt))
      for impl in qc.GDN_CONV_IMPLS:
        cfg = small(dtype=dt, gdn_conv_impl=impl)
        conv = jax.jit(lambda x, w, c=cfg: gdn.segmented_causal_conv(x, w, jnp.asarray(seg), c))
        got[dt, impl] = conv(x.astype(dt), w)
        with self.subTest(dtype=jnp.dtype(dt).name, impl=impl):
          self.assertEqual(got[dt, impl].dtype, dt)
          if dt == jnp.float32:
            np.testing.assert_allclose(np.asarray(got[dt, impl], np.float64), want, rtol=1e-5, atol=4e-5)
          else:  # atol 2^-20: the fp32 sum's own error, where the taps cancel
            np.testing.assert_allclose(np.asarray(got[dt, impl], np.float64), want, rtol=2**-8, atol=2**-20)
            rounded = np.asarray(conv(x.astype(dt), w.astype(dt).astype(jnp.float32)), np.float64)
            self.assertFalse(np.allclose(rounded, want, rtol=2**-8, atol=2**-20))  # they do differ
          self.assertEqual(float(jnp.max(jnp.abs(got[dt, impl][jnp.asarray(canon == 0)]))), 0.0)
    self.assertTrue(bool(jnp.array_equal(got[jnp.bfloat16, "conv"], got[jnp.bfloat16, "shift"])))
    ones = jnp.ones((b, s), jnp.int32)
    cfg = small()
    np.testing.assert_allclose(
        np.asarray(gdn.segmented_causal_conv(x, w, ones, cfg)),
        np.asarray(gdn.causal_depthwise_conv(x, w, cfg)),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_causal_conv(self):
    """One (K-1, 0) pad == MaxText's pad + 'CAUSAL' + slice (bitwise, fp32 and bf16) == the definition."""
    rng = np.random.default_rng(0)
    b, s, ch, k = 2, 13, 12, 4
    x = jnp.asarray(rng.standard_normal((b, s, ch)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((k, 1, ch)), jnp.float32)

    def maxtext_form(x, w):  # nnx.Conv(dtype=...) casts input AND kernel to its dtype
      xp = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
      y = jax.lax.conv_general_dilated(
          xp, w.astype(x.dtype), (1,), ((k - 1, 0),), dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=ch
      )
      return y[:, -s:]

    for dt in (jnp.bfloat16, jnp.float32):
      with self.subTest(dtype=jnp.dtype(dt).name):
        cfg = small(dtype=dt)
        got = jax.jit(lambda x, w, cfg=cfg: gdn.causal_depthwise_conv(x, w, cfg))(x.astype(dt), w)
        self.assertEqual(got.dtype, dt)
        self.assertTrue(bool(jnp.array_equal(got, jax.jit(maxtext_form)(x.astype(dt), w))))
    xp = np.pad(np.asarray(x, np.float64), ((0, 0), (k - 1, 0), (0, 0)))  # got is the fp32 run
    want = sum(xp[:, i : i + s] * np.asarray(w, np.float64)[i, 0] for i in range(k))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5)

  def test_causal_conv_shift(self):
    """gdn_conv_impl 'shift' (the GPU default: K shifted multiply-adds in fp32, one rounding) is the conv: in bf16
    the output and its input gradient bitwise here, the weight gradient (a reduction over batch and time the conv
    rounds differently) to bf16's precision; in fp32 all three to fp32 rounding; the definition either way.  A
    wrong shift (taps reversed) is caught."""
    rng = np.random.default_rng(1)
    b, s, ch, k = 2, 13, 12, 4
    x = jnp.asarray(rng.standard_normal((b, s, ch)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((k, 1, ch)), jnp.float32)
    ct = jnp.asarray(rng.standard_normal((b, s, ch)), jnp.float32)
    with self.assertRaisesRegex(ValueError, "gdn_conv_impl"):
      small(gdn_conv_impl="fft")

    def run(cfg, dt):
      f = lambda x, w: jnp.sum(gdn.causal_depthwise_conv(x, w, cfg).astype(jnp.float32) * ct)
      y = jax.jit(lambda x, w: gdn.causal_depthwise_conv(x, w, cfg))(x.astype(dt), w)
      return (y,) + jax.jit(jax.grad(f, argnums=(0, 1)))(x.astype(dt), w)

    for dt in (jnp.bfloat16, jnp.float32):
      with self.subTest(dtype=jnp.dtype(dt).name):
        conv, shift = run(small(dtype=dt), dt), run(small(dtype=dt, gdn_conv_impl="shift"), dt)
        self.assertEqual(shift[0].dtype, dt)
        for name, a, c in zip(("y", "dx", "dw"), conv, shift):
          if dt == jnp.bfloat16 and name != "dw":
            self.assertTrue(bool(jnp.array_equal(a, c)), name)
          elif dt == jnp.bfloat16:
            np.testing.assert_allclose(
                np.asarray(c), np.asarray(a), rtol=2**-7, atol=2**-7 * float(jnp.max(jnp.abs(a))), err_msg=name
            )
          else:
            np.testing.assert_allclose(np.asarray(c), np.asarray(a), rtol=1e-5, atol=1e-5, err_msg=name)
    xp = np.pad(np.asarray(x, np.float64), ((0, 0), (k - 1, 0), (0, 0)))
    want = sum(xp[:, i : i + s] * np.asarray(w, np.float64)[i, 0] for i in range(k))
    np.testing.assert_allclose(np.asarray(shift[0]), want, rtol=1e-5, atol=1e-5)
    flipped = gdn.causal_conv_shift(x, w[::-1])
    self.assertGreater(float(jnp.max(jnp.abs(flipped - shift[0]))), 0.1)


class GdnSegmentsTest(unittest.TestCase):
  """What segment ids mean to one Gated DeltaNet layer (fp32), by `gdn_packing` (MaxText's `packing`): the semantics
  behind qwen35_maxtext_parity_test's packed and padded cases, without MaxText."""

  def setUp(self):
    self.cfg = small()
    stacked = qm.gdn_layer_avals(self.cfg, 1)["attention"]
    self.p = jax.tree.map(lambda a: a[0], realize({"a": stacked})["a"])
    self.x = jnp.asarray(np.random.default_rng(7).standard_normal((2, 30, self.cfg.emb_dim)), jnp.float32)

  def layer(self, x, seg, cfg):
    return jax.jit(lambda p, x, s: gdn.gated_delta_net(p, x, s, cfg))(self.p, x, jnp.asarray(seg))

  def test_packed_documents_are_separate_sequences(self):
    """gdn_packing: each document of a packed row is the layer run on that document alone (to fp32 rounding:
    other chunk edges), and padding comes out 0.  Without it a row's later documents part from that: the
    recurrent state and the conv window run on into them."""
    seg = np.zeros((2, 30), np.int32)
    seg[0] = [1] * 7 + [2] * 13 + [3] * 6 + [0] * 4
    seg[1] = [5] * 11 + [9] * 19
    packed = self.layer(self.x, seg, self.cfg)
    unpacked = self.layer(self.x, seg, self.cfg.replace(gdn_packing=False))
    for row in range(2):
      for d in sorted(set(seg[row]) - {0}):
        idx = np.nonzero(seg[row] == d)[0]
        with self.subTest(row=row, document=d):
          alone = self.layer(self.x[row : row + 1, idx], np.ones((1, len(idx)), np.int32), self.cfg)[0]
          self.assertLess(max_rel(packed[row, idx], alone), 1e-5)
          if idx[0] > 0:
            self.assertGreater(max_rel(unpacked[row, idx], alone), 1e-3)
    self.assertEqual(float(jnp.max(jnp.abs(packed[0, 26:]))), 0.0)

  def test_padding_before_the_conv(self):
    """Left padding, zeroed before the conv as well as after it: the tokens after it are the sequence without it,
    packed or not."""
    pad = 5
    seg = np.ones((2, 30), np.int32)
    seg[:, :pad] = 0
    alone = self.layer(self.x[:, pad:], np.ones((2, 30 - pad), np.int32), self.cfg)
    for packing in (True, False):
      with self.subTest(gdn_packing=packing):
        got = self.layer(self.x, seg, self.cfg.replace(gdn_packing=packing))
        self.assertLess(max_rel(got[:, pad:], alone), 1e-5)


# --------------------------------------------------------------------------- #
# RMSNorm numerics
# --------------------------------------------------------------------------- #
def _normed32(x, eps):
  x32 = x.astype(jnp.float32)
  return x32 * jax.lax.rsqrt(jnp.mean(jax.lax.square(x32), axis=-1, keepdims=True) + eps)


class NormNumericsTest(unittest.TestCase):
  """`norm_numerics`, bitwise, against each source's own op sequence.

  Run eagerly, op by op, so that every rounding a formula writes happens: under
  jit, XLA may keep excess precision across a fused bf16 cast and make two
  different roundings agree (the parity test runs its bf16 programs with
  --xla_allow_excess_precision=false for that reason).  Weights are
  bf16-representable, as HF loads them.
  """

  @classmethod
  def setUpClass(cls):
    rng = np.random.default_rng(0)
    cls.cfg = qc.tiny()
    cls.eps = cls.cfg.normalization_layer_epsilon
    bf16 = lambda a: jnp.asarray(a, jnp.bfloat16)
    cls.x = bf16(rng.standard_normal((2, 9, cls.cfg.emb_dim)) * 3)  # residual stream
    cls.s = bf16(0.3 * rng.standard_normal(cls.cfg.emb_dim)).astype(jnp.float32)  # zero-centred scale
    cls.o = bf16(rng.standard_normal((2, 9, 4, 16)) * 3)  # GDN core output
    cls.z = bf16(rng.standard_normal((2, 9, 4, 16)) * 2)
    cls.w = bf16(1 + 0.3 * rng.standard_normal(16)).astype(jnp.float32)  # gated norm scale

  def test_zero_centred(self):
    """'fp32' = MaxText from 9a343b3d8 (b33d0f687) = HF: `(normed * (1 + s)).to(bf16)`;
    'bf16' = MaxText before it (pr-5060): bf16 normalised value, `bf16(s) + 1` in bf16."""
    normed = _normed32(self.x, self.eps)
    want = {
        "fp32": (normed * (1.0 + self.s)).astype(jnp.bfloat16),
        "bf16": normed.astype(jnp.bfloat16) * (self.s.astype(jnp.bfloat16) + 1.0),
    }
    self.assertFalse(bool(jnp.array_equal(want["fp32"], want["bf16"])))  # they do differ
    for n, w in want.items():
      with self.subTest(norm_numerics=n):
        got = layers.rms_norm({"scale": self.s}, self.x, self.cfg.replace(norm_numerics=n))
        self.assertEqual(got.dtype, jnp.bfloat16)
        self.assertTrue(bool(jnp.array_equal(got, w)))

  def test_gated(self):
    """'fp32' = MaxText from 9a343b3d8: fp32 until the one rounding after the gate;
    'bf16' = MaxText before it = HF (`weight * normed.to(bf16)`, then the fp32 gate)."""
    normed, gate = _normed32(self.o, self.eps), jax.nn.silu(self.z.astype(jnp.float32))
    want = {
        "fp32": (normed * self.w * gate).astype(jnp.bfloat16),
        "bf16": (self.w.astype(jnp.bfloat16) * normed.astype(jnp.bfloat16) * gate).astype(jnp.bfloat16),
    }
    self.assertFalse(bool(jnp.array_equal(want["fp32"], want["bf16"])))
    for n, w in want.items():
      with self.subTest(norm_numerics=n):
        got = gdn.gated_rms_norm({"rms_norm": {"scale": self.w}}, self.o, self.z, self.cfg.replace(norm_numerics=n))
        self.assertTrue(bool(jnp.array_equal(got, w)))

  def test_one_computation_in_fp32(self):
    """With `dtype=float32` the two settings trace to the same program."""
    cfg = self.cfg.replace(dtype=jnp.float32)
    x, o, z = (a.astype(jnp.float32) for a in (self.x, self.o, self.z))
    for name, fn, args in (
        ("zero-centred", lambda c, x, s: layers.rms_norm({"scale": s}, x, c), (x, self.s)),
        ("gated", lambda c, o, z, w: gdn.gated_rms_norm({"rms_norm": {"scale": w}}, o, z, c), (o, z, self.w)),
    ):
      with self.subTest(name):
        a, b = (str(jax.make_jaxpr(functools.partial(fn, cfg.replace(norm_numerics=n)))(*args)) for n in qc.NORM_NUMERICS)
        self.assertEqual(a, b)


# --------------------------------------------------------------------------- #
# Parameter tree and init
# --------------------------------------------------------------------------- #
class ParamTreeTest(unittest.TestCase):

  def test_tiny_tree_is_maxtexts_with_the_block_axis_first(self):
    """The tiny config: MaxText's 70 leaves / 795136 params, scan axis 1 -> 0."""
    cfg = qc.tiny()
    avals = flat(qm.param_avals(cfg))
    self.assertEqual(len(avals), 70)
    self.assertEqual(sum(int(np.prod(a.shape)) for a in avals.values()), 795136)
    lay = "['decoder']['layers']"
    want = {
        f"{lay}['layer_0']['attention']['in_proj_qkvz']['kernel']": (2, 64, 320),
        f"{lay}['layer_0']['attention']['in_proj_ba']['kernel']": (2, 64, 16),
        f"{lay}['layer_0']['attention']['conv1d']['kernel']": (2, 4, 1, 192),
        f"{lay}['layer_0']['attention']['A_log']": (2, 8),
        f"{lay}['layer_0']['attention']['norm']['rms_norm']['scale']": (2, 16),
        f"{lay}['layer_0']['attention']['out_proj']['kernel']": (2, 128, 64),
        f"{lay}['layer_2']['mlp']['routed_experts']['wi_0']": (2, 8, 64, 32),
        f"{lay}['layer_2']['mlp']['routed_experts']['wo']": (2, 8, 32, 64),
        f"{lay}['layer_2']['mlp']['shared_expert_gate']['kernel']": (2, 64, 1),
        f"{lay}['layer_3']['attention']['attention']['query']['kernel']": (2, 64, 4, 128),
        f"{lay}['layer_3']['attention']['attention']['key']['kernel']": (2, 64, 2, 64),
        f"{lay}['layer_3']['attention']['attention']['out']['kernel']": (2, 256, 64),
        f"{lay}['layer_3']['attention']['attention']['query_norm']['scale']": (2, 64),
        "['token_embedder']['embedding']": (256, 64),
        "['decoder']['logits_dense']['kernel']": (64, 256),
        "['decoder']['decoder_norm']['scale']": (64,),
    }
    for path, shape in want.items():
      self.assertEqual(tuple(avals[path].shape), shape, path)
    self.assertTrue(all(a.dtype == jnp.float32 for a in avals.values()))

  def test_full_size_parameter_counts(self):
    """qwen3.5-397b-a17b and -35b-a3b, from avals: the totals and active-per-token
    counts of MaxText's abstract model at full size."""
    for cfg, total, active in ((qc.qwen35_397b(), 396.346e9, 17.349e9), (qc.qwen35_35b(), 34.661e9, 3.455e9)):
      avals = flat(qm.param_avals(cfg))
      n = sum(int(np.prod(a.shape)) for a in avals.values())
      routed = sum(int(np.prod(a.shape)) for p, a in avals.items() if "['routed_experts']['w" in p)
      act = n - routed + routed * cfg.num_experts_per_tok // cfg.num_experts
      self.assertAlmostEqual(n / 1e9, total / 1e9, places=3)
      self.assertAlmostEqual(act / 1e9, active / 1e9, places=3)

  def test_shared_expert_is_optional(self):
    mlp = qm.param_avals(small(shared_experts=0))["decoder"]["layers"]["layer_0"]["mlp"]
    self.assertEqual(sorted(mlp), ["routed_experts"])

  def test_init_params_builds_the_aval_tree(self):
    """Structure, shapes and dtypes, via eval_shape (no compile)."""
    for cfg in (qc.tiny(), small(), small(weight_dtype=jnp.bfloat16)):
      avals = qm.param_avals(cfg)
      got = jax.eval_shape(lambda a=avals: qs.init_params(a, jax.random.key(0)))
      self.assertEqual(jax.tree.structure(got), jax.tree.structure(avals))
      for g, a in zip(jax.tree.leaves(got), jax.tree.leaves(avals)):
        self.assertEqual((g.shape, g.dtype), (a.shape, a.dtype))

  def test_init_rules(self):
    """Zero-centred norms 0, the gated GDN norm 1, A_log = log U(1e-9, 16),
    dt_bias 1, kernels normal / sqrt(fan_in) -- the conv's fan_in is K."""
    cfg = small()
    avals = qm.param_avals(cfg)
    lay = avals["decoder"]["layers"]
    zero_centred = (
        ("decoder", "decoder_norm", "scale"),
        ("decoder", "layers", "layer_0", "input_layernorm", "scale"),
        ("decoder", "layers", "layer_1", "attention", "attention", "query_norm", "scale"),
    )
    gated = ("decoder", "layers", "layer_0", "attention", "norm", "rms_norm", "scale")
    dt_bias = ("decoder", "layers", "layer_0", "attention", "dt_bias")
    a_log = ("decoder", "layers", "layer_0", "attention", "A_log")
    head = ("decoder", "logits_dense", "kernel")

    def aval(path):
      if path == head:
        return jax.ShapeDtypeStruct((256, 512), jnp.float32)
      node = avals
      for p in path:
        node = node[p]
      return node

    paths = zero_centred + (gated, dt_bias, a_log, head)
    # one program for all of them (eagerly, every random op compiles on its own)
    got = jax.jit(lambda k: [qs.init_leaf(p, aval(p).shape, aval(p).dtype, k) for p in paths])(jax.random.key(1))
    init = {p: np.asarray(v) for p, v in zip(paths, got)}
    for path in zero_centred:
      self.assertEqual(float(np.abs(init[path]).max()), 0.0, path)
    np.testing.assert_array_equal(init[gated], 1.0)
    np.testing.assert_array_equal(init[dt_bias], 1.0)
    a_log = init[a_log]
    self.assertTrue(np.all(a_log <= np.log(16.0)) and np.all(a_log > np.log(1e-9) - 1e-3))
    self.assertGreater(float(np.std(a_log)), 0.1)
    self.assertEqual(
        qs.fan_in(
            ("decoder", "layers", "layer_0", "attention", "conv1d", "kernel"),
            lay["layer_0"]["attention"]["conv1d"]["kernel"].shape,
        ),
        cfg.gdn_conv_kernel_dim,
    )
    self.assertEqual(
        qs.fan_in(
            ("decoder", "layers", "layer_1", "attention", "attention", "query", "kernel"),
            lay["layer_1"]["attention"]["attention"]["query"]["kernel"].shape,
        ),
        cfg.emb_dim,
    )
    self.assertEqual(
        qs.fan_in(
            ("decoder", "layers", "layer_0", "mlp", "routed_experts", "wo"),
            lay["layer_0"]["mlp"]["routed_experts"]["wo"].shape,
        ),
        cfg.moe_mlp_dim,
    )
    self.assertEqual(qs.fan_in(("token_embedder", "embedding"), (64, 32)), 1)
    self.assertAlmostEqual(float(np.std(init[head])) * 256**0.5, 1.0, delta=0.02)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def fake_maxtext_cfg(**kw):
  """The attributes `from_maxtext` reads, at the tiny shape."""
  c = qc.tiny()
  base = dict(
      decoder_block=types.SimpleNamespace(value="qwen3_5"),
      use_mrope=True,
      rope_type="default",
      routed_score_func="",
      routed_bias=False,
      routed_scaling_factor=1.0,
      logits_via_embedding=False,
      logits_dot_in_fp32=False,
      vocab_size=c.vocab_size,
      emb_dim=c.emb_dim,
      num_decoder_layers=c.num_decoder_layers,
      inhomogeneous_layer_cycle_interval=4,
      num_query_heads=c.num_query_heads,
      num_kv_heads=c.num_kv_heads,
      head_dim=c.head_dim,
      partial_rotary_factor=0.25,
      rope_min_timescale=1,
      rope_max_timescale=10_000_000,
      mrope_section=[3, 3, 2],
      gdn_num_key_heads=2,
      gdn_num_value_heads=8,
      gdn_key_head_dim=16,
      gdn_value_head_dim=16,
      gdn_conv_kernel_dim=4,
      gdn_chunk_size=16,
      use_qk_norm_in_gdn=True,
      # the GDN's options (`qc.maxtext_gdn_fields`)
      gdn_chunk_inverse="block_doubling",
      packing=True,
      moe_mlp_dim=32,
      num_experts=8,
      num_experts_per_tok=2,
      shared_experts=1,
      norm_topk_prob=True,
      mlp_activations=["silu", "linear"],
      sparse_matmul=True,
      megablox=True,
      use_tokamax_gmm=False,
      moe_use_direct_token_gather=False,
      float32_weight_sum=True,
      float32_gate_logits=False,
      load_balance_loss_weight=0.0,
      normalization_layer_epsilon=1e-6,
      dtype=jnp.bfloat16,
      weight_dtype=jnp.float32,
      matmul_precision="default",
      cast_logits_to_fp32=True,
      attention="autoselected",
      use_tokamax_splash=False,
      sa_block_q=512,
      sa_block_kv=512,
      sa_block_kv_compute=512,
      sa_block_q_dkv=512,
      sa_block_kv_dkv=512,
      sa_block_kv_dkv_compute=512,
      sa_use_base2_exp=True,
      max_target_length=64,
      scan_layers=True,
      remat_policy="full",
  )
  base.update(kw)
  return types.SimpleNamespace(**base)


class ConfigTest(unittest.TestCase):

  def test_from_maxtext_is_the_tiny_preset(self):
    got = qc.from_maxtext(fake_maxtext_cfg(), platform="cpu")
    want = qc.tiny(sa_block_q_dkv=512, sa_block_kv_dkv=512, sa_block_kv_dkv_compute=512)
    self.assertEqual(got, want)
    # the norm rounding is the MaxText revision's, not the pyconfig's: a parameter
    self.assertEqual(got.norm_numerics, "fp32")
    self.assertEqual(
        qc.from_maxtext(fake_maxtext_cfg(), platform="cpu", norm_numerics="bf16"), want.replace(norm_numerics="bf16")
    )

  def test_from_maxtext_reads_the_gdn_options(self):
    """The GDN's settings MaxText's options give, read off its config (`qc.maxtext_gdn_fields`): its packing and
    gdn_chunk_inverse, whose defaults are the port's."""
    main = qc.from_maxtext(fake_maxtext_cfg(), platform="cpu")
    fields = ("gdn_packing", "gdn_chunk_inverse")
    self.assertEqual([getattr(main, f) for f in fields], [getattr(qc.Qwen35Config(), f) for f in fields])
    self.assertEqual([getattr(main, f) for f in fields], [True, "block_doubling"])
    got = qc.from_maxtext(fake_maxtext_cfg(packing=False, gdn_chunk_inverse="solve"), platform="cpu")
    self.assertEqual(got, main.replace(gdn_packing=False, gdn_chunk_inverse="solve"))
    self.assertEqual(
        qc.maxtext_gdn_fields(fake_maxtext_cfg(packing=False)),
        dict(gdn_packing=False, gdn_chunk_inverse="block_doubling"),
    )

  def test_from_maxtext_carries_use_mrope(self):
    """MaxText's use_mrope picks the rotary embedding: false (qwen3.8-2.4t-a95b's yml, base.yml's default) is the
    partial RoPE (`attention.partial_rope`), whose config no longer checks the mRoPE sections."""
    got = qc.from_maxtext(fake_maxtext_cfg(use_mrope=False, mrope_section=[1, 1, 1]), platform="cpu")
    self.assertIs(got.use_mrope, False)
    self.assertEqual(got.mrope_section, (1, 1, 1))
    self.assertEqual(
        got.replace(use_mrope=True, mrope_section=(3, 3, 2)), qc.from_maxtext(fake_maxtext_cfg(), platform="cpu")
    )
    with self.assertRaisesRegex(ValueError, "mrope_section"):
      got.replace(use_mrope=True)

  def test_attention_kernel_resolution(self):
    """MaxText's autoselect: dot-product below 128 tokens or on CPU, splash on TPU."""
    r = qc.resolve_attention_kernel
    self.assertEqual(r(fake_maxtext_cfg(max_target_length=4096), "cpu"), "dot_product")
    self.assertEqual(r(fake_maxtext_cfg(max_target_length=64), "tpu"), "dot_product")
    self.assertEqual(r(fake_maxtext_cfg(max_target_length=4096), "tpu"), "splash")
    self.assertEqual(r(fake_maxtext_cfg(attention="flash", max_target_length=64), "tpu"), "splash")
    self.assertEqual(r(fake_maxtext_cfg(max_target_length=4096, use_tokamax_splash=True), "tpu"), "tokamax_splash")
    with self.assertRaisesRegex(ValueError, "unsupported attention"):
      r(fake_maxtext_cfg(attention="paged"), "tpu")

  def test_from_maxtext_refuses_what_it_does_not_implement(self):
    """Each of these makes MaxText compute something the port does not: plain
    q/k RMSNorms (use_qk_norm), token dropping (capacity_factor under the dense
    MoE), expert biases (mlp_bias), dropout, quantized matmuls, fp32 MLP
    activations (activations_in_float32), sinusoidal position embeddings
    (use_untrainable_positional_embedding), the LLaMA-3.1 or YaRN RoPE MaxText
    picks ahead of the partial one under use_mrope=false (rope_type, as a str
    or MaxText's RopeType enum, or a llama3.1 model_name) -- the last three add
    no parameter, so the tree check cannot catch them -- bf16 weights, under
    which MaxText before 9a343b3d8 keeps one kernel (shared_expert_gate) fp32,
    one expert, under which current MaxText builds the dense MLP
    (qwen3.5-9b), and the Tokamax GDN kernel, its two-pass bf16 refinement and
    FP8 GDN projections, which round differently from the port's GDN."""
    enum = types.SimpleNamespace  # stands in for MaxText's (str, Enum) RopeType: its .value
    for kw in (
        dict(decoder_block=types.SimpleNamespace(value="qwen3_next")),
        dict(routed_score_func="sigmoid"),
        dict(routed_bias=True),
        dict(routed_scaling_factor=2.5),
        dict(logits_via_embedding=True),
        dict(logits_dot_in_fp32=True),
        dict(use_qk_norm=True),
        dict(capacity_factor=0.5, sparse_matmul=False),
        dict(mlp_bias=True),
        dict(dropout_rate=0.1, enable_dropout=True),
        dict(quantization="int8"),
        dict(activations_in_float32=True),
        dict(use_untrainable_positional_embedding=True),
        dict(use_mrope=False, rope_type="yarn"),
        dict(use_mrope=False, rope_type="llama3.1"),
        dict(use_mrope=False, rope_type=enum(value="yarn")),
        dict(use_mrope=False, model_name="llama3.1-8b"),
        dict(weight_dtype=jnp.bfloat16),
        dict(num_experts=1, num_experts_per_tok=1),
        dict(use_tokamax_gdn=True),
        dict(gdn_two_pass_bf16=True),
        dict(gdn_fp8_projections=True),
    ):
      with self.subTest(**{k: str(v) for k, v in kw.items()}):
        with self.assertRaises(NotImplementedError):
          qc.from_maxtext(fake_maxtext_cfg(**kw), platform="cpu")
    # ...and these do not: MaxText's sparse_matmul path never reads capacity_factor,
    # its trainer applies no dropout without enable_dropout (the Qwen3.5 ymls' setting),
    # and under use_mrope=true its mRoPE comes before any rope_type.
    for kw in (
        dict(capacity_factor=0.5),
        dict(dropout_rate=0.1, enable_dropout=False),
        dict(
            capacity_factor=-1.0,
            sparse_matmul=False,
            use_qk_norm=False,
            mlp_bias=False,
            dropout_rate=0.0,
            enable_dropout=True,
            quantization="",
            activations_in_float32=False,
            use_untrainable_positional_embedding=False,
            weight_dtype=jnp.float32,
            use_tokamax_gdn=False,
            gdn_two_pass_bf16=False,
            gdn_fp8_projections=False,
        ),
        dict(rope_type="yarn"),
        dict(rope_type=enum(value="llama3.1"), model_name="llama3.1-8b"),
        dict(use_mrope=False, rope_type=enum(value="default"), model_name="qwen3.8-2.4t-a95b"),
    ):
      with self.subTest(accepted=str(kw)):
        got = qc.from_maxtext(fake_maxtext_cfg(**kw), platform="cpu")
        self.assertEqual((got.vocab_size, got.use_mrope), (256, kw.get("use_mrope", True)))

  def test_validation(self):
    for kw, msg in (
        (dict(num_decoder_layers=6), "multiple of"),
        (dict(gdn_num_value_heads=6, gdn_num_key_heads=4), "gdn_num_value_heads"),
        (dict(num_query_heads=3), "num_query_heads"),
        (dict(mrope_section=(3, 3, 3)), "mrope_section"),
        (dict(num_experts_per_tok=9), "num_experts_per_tok"),
        (dict(shared_experts=2), "shared_experts"),
        (dict(remat_granularity="layers"), "remat_granularity"),
        (dict(attention_kernel="flash"), "attention_kernel"),
        (dict(moe_impl="sparse"), "moe_impl"),
        (dict(norm_numerics="fp16"), "norm_numerics"),
        (dict(gdn_chunk_inverse="neumann"), "gdn_chunk_inverse"),
        (dict(moe_mlp_dim=-1), "moe_mlp_dim"),
    ):
      with self.subTest(**kw):
        with self.assertRaisesRegex(ValueError, msg):
          qc.tiny(**kw)

  def test_derived_and_presets(self):
    c = qc.qwen35_397b()
    self.assertEqual(
        (c.num_blocks, c.rotary_dim, c.gdn_v_per_k, c.gdn_conv_dim, c.gdn_qkvz_dim), (15, 64, 4, 12288, 20480)
    )
    self.assertEqual([i for i in range(8) if c.is_attention_layer(i)], [3, 7])
    m = qc.qwen35_35b()
    self.assertEqual((m.emb_dim, m.num_decoder_layers, m.num_experts, m.gdn_v_per_k), (2048, 40, 256, 2))
    s = qc.qwen35_397b(16, 32, 32768)  # depth, experts and vocab are knobs
    self.assertEqual((s.num_decoder_layers, s.num_experts, s.vocab_size, s.emb_dim), (16, 32, 32768, 4096))
    c = qc.tiny(mrope_section=[3, 3, 2], mlp_activations=["silu", "linear"])  # lists from yaml
    self.assertEqual(hash(c), hash(qc.tiny()))


class _Param:
  """Stands in for `nnx.Param` in a fake `flat_state()` (`convert` reads the type name)."""

  def __init__(self, value):
    self.value = value


_Param.__name__ = "Param"


class ConvertTest(unittest.TestCase):

  def test_tree_must_be_the_models(self):
    """`check_params` / `from_nnx_state(config=)` refuse a converted tree with a
    leaf the forward never reads (`mlp_bias`'s expert biases), a missing leaf,
    or another shape or dtype."""
    cfg = small()
    avals = qm.param_avals(cfg)
    convert.check_params(avals, cfg)
    entries = [
        (tuple(k.key for k in p), _Param(np.zeros(a.shape, a.dtype)))
        for p, a in jax.tree_util.tree_flatten_with_path(avals)[0]
    ]
    got = convert.from_nnx_state(types.SimpleNamespace(flat_state=lambda: entries), 0, config=cfg)
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(avals))
    bias = (
        ("decoder", "layers", "layer_0", "mlp", "routed_experts", "wi_0_bias"),
        _Param(np.zeros((cfg.num_blocks, cfg.num_experts, cfg.moe_mlp_dim), np.float32)),
    )
    state = types.SimpleNamespace(flat_state=lambda: entries + [bias])
    self.assertEqual(len(jax.tree.leaves(convert.from_nnx_state(state, 0))), len(entries) + 1)
    with self.assertRaisesRegex(ValueError, r"1 extra leaves .*wi_0_bias"):
      convert.from_nnx_state(state, 0, config=cfg)

    def edited(path, leaf):
      tree = qm.param_avals(cfg)
      node = tree
      for key in path[:-1]:
        node = node[key]
      if leaf is None:
        del node[path[-1]]
      else:
        node[path[-1]] = leaf
      return tree

    a_log = ("decoder", "layers", "layer_0", "attention", "A_log")
    n, hv = cfg.num_blocks, cfg.gdn_num_value_heads
    for tree, msg in (
        (edited(("decoder", "decoder_norm"), None), "1 missing"),
        (edited(a_log, jax.ShapeDtypeStruct((n, hv + 1), jnp.float32)), "1 with another"),
        (edited(a_log, jax.ShapeDtypeStruct((n, hv), jnp.bfloat16)), "1 with another"),
    ):
      with self.subTest(msg):
        with self.assertRaisesRegex(ValueError, msg):
          convert.check_params(tree, cfg)

  def test_nested_scan_layout(self):
    """MaxText's nested-scan tree (a block's cycle - 1 linear layers as one scanned `local_layers` subtree, each
    leaf with their axis right after the block axis, the full-attention layer as `global_layer`) converts to the
    port's layer_0..layer_{cycle-1} -- the port tree's own arrays, at param_scan_axis 0 and 1, and through
    `from_flat` on host arrays as views."""
    cfg = small(inhomogeneous_layer_cycle_interval=4)
    rng = np.random.default_rng(0)
    port = jax.tree.map(lambda a: rng.standard_normal(a.shape).astype(np.float32), qm.param_avals(cfg))
    cyc, layers = cfg.inhomogeneous_layer_cycle_interval, port["decoder"]["layers"]
    for axis in (0, 1):
      flat = []
      for path, a in jax.tree_util.tree_flatten_with_path(port)[0]:
        keys = [k.key for k in path]
        if keys[:2] != ["decoder", "layers"]:
          flat.append((keys, a))
      # MaxText's scans stack at param_scan_axis: the inner (local layers) one each block's layers, then the outer
      # (blocks) one those -- a per-layer leaf [d0, ...] becomes [d0, NB, cycle - 1, ...] at axis 1
      for path, _ in jax.tree_util.tree_flatten_with_path(layers["layer_0"])[0]:
        rest = [k.key for k in path]
        leaf = lambda i, b, rest=rest: functools.reduce(lambda t, k: t[k], rest, layers[f"layer_{i}"])[b]
        nested = np.stack(
            [np.stack([leaf(i, b) for i in range(cyc - 1)], axis=axis) for b in range(cfg.num_blocks)], axis=axis
        )
        flat.append((["decoder", "layers", "local_layers"] + rest, nested))
      for path, a in jax.tree_util.tree_flatten_with_path(layers[f"layer_{cyc - 1}"])[0]:
        flat.append((["decoder", "layers", "global_layer"] + [k.key for k in path], np.moveaxis(a, 0, axis)))
      with self.subTest(param_scan_axis=axis):
        got = convert.from_flat(flat, axis, config=cfg, asarray=np.asarray)
        self.assertEqual(jax.tree.structure(got), jax.tree.structure(port))
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(port)):
          np.testing.assert_array_equal(a, b)
          # the memory of an input (np.asarray passes a whole leaf through): no copy, local_layers' slices included
          self.assertTrue(any(np.shares_memory(a, v) for _, v in flat))
        state = types.SimpleNamespace(flat_state=lambda flat=flat: [(tuple(k), _Param(v)) for k, v in flat])
        for a, b in zip(jax.tree.leaves(convert.from_nnx_state(state, axis, config=cfg)), jax.tree.leaves(port)):
          np.testing.assert_array_equal(np.asarray(a), b)


class SharedLayerConfigTest(unittest.TestCase):
  """The DeepSeek functions the port reaches read exactly the fields of
  `maxtext.experimental.mpmd.jax.deepseek.config.SharedLayerConfig`, the one Protocol both configs
  are held to, and `Qwen35Config` satisfies it."""

  SHARED = (
      ds_layers.precision_of,
      ds_layers.precision_or_none,
      ds_layers.rms_norm,
      ds_layers.dense_general,
      ds_layers.embed_lookup,
      ds_layers.mlp_block,
      ds_moe.routed_experts,
      ds_moe._experts_dense,
      ds_moe._experts_ragged,  # pylint: disable=protected-access
      ds_attention.attention_op,
      ds_attention._attention_kernel,  # pylint: disable=protected-access
      ds_attention.dot_product_attention,
      ds_attention.tokamax_splash_attention,
      ds_model.remat_policy_for,
  )

  def test_protocol_lists_exactly_what_the_shared_functions_read(self):
    read = set()
    for fn in self.SHARED:
      read |= set(re.findall(r"\bconfig\.(\w+)", inspect.getsource(fn)))
    # attention_op passes these to the splash kernel positionally
    self.assertLessEqual({"num_query_heads", "sa_block_q"}, read)
    self.assertEqual(read, set(SHARED_LAYER_FIELDS))
    self.assertIs(qc.SharedLayerConfig, SharedLayerConfig)  # no second copy

  def test_both_configs_satisfy_it(self):
    self.assertIsInstance(qc.tiny(), SharedLayerConfig)
    self.assertIsInstance(DeepSeekConfig(), SharedLayerConfig)
    for field in SHARED_LAYER_FIELDS:  # the same meaning, hence the same name
      self.assertIn(field, {f.name for f in dataclasses.fields(qc.Qwen35Config)})

  def test_policy_table_is_the_shared_one(self):
    """One policy table for both models (review finding 7): every name in the
    dict `remat_policy_for` builds the policy from builds one for Qwen3.5 too,
    and the config refuses a name that is not in it."""
    for name, saved in ds_model.REMAT_POLICIES.items():
      policy = ds_model.remat_policy_for(small(remat_policy=name))
      self.assertEqual(policy is None, not saved, name)
    for field in ("remat_policy", "layer_remat_policy"):
      with self.assertRaisesRegex(ValueError, f"{field}='save_everything'"):
        small(**{field: "save_everything"})


WEIGHT_ARGS = ("params", "kernel", "stack", "blk")

# Uses of a weight that the rule in `weight_misuses` flags but that hand the
# weights on untouched, by module and exact source: the stacked blocks ARE the
# scan's xs; the unrolled path takes block i of every leaf; the stack's block
# count; a layer of the block (`decoder_layer`, reached through `_layer_fns`, so
# its callee is a local name).  Anything else done to `stack` / `blk` -- a cast
# of every leaf before the scan or inside its body, say -- is flagged.
ALLOWED_WEIGHT_USES = {
    "model.py": frozenset(
        {
            "jax.lax.scan(body, x, stack)",
            "jax.tree.map(lambda leaf, i=i: leaf[i], stack)",
            "jax.tree.leaves(stack)",
            "f(blk[f'layer_{i}'], carry, positions, segment_ids, inv_freq)",
        }
    ),
}


def weight_misuses(sources, only=None):
  """`{module: [line, ...]}`: where a model module (of `only`, default all)
  touches a weight other than by handing it on.

  A weight is a function argument named in `WEIGHT_ARGS` (`params`, `kernel`,
  and `apply_blocks`' `stack` and its scan body's `blk`, through which every
  decoder-layer weight enters), a subscript of one by a constant key
  (`params['attention']`, `params['decoder'][LAYERS]`), or a local alias of such
  a subscript (`routed = params['routed_experts']`).  The only allowed use is
  as an argument of an `_ops.*` call or of a function of the model modules
  (`layers.rms_norm`, `gdn.gated_delta_net`, `route`, ...), which get the same
  treatment, plus `ALLOWED_WEIGHT_USES`; a cast, an `.astype`, an index by a
  traced value or a raw `jnp` / `lax` op on a weight is a bypass of `_ops`.
  """
  trees = {name: ast.parse(src) for name, src in sources.items()}
  module_fns = {name[:-3]: {n.name for n in t.body if isinstance(n, ast.FunctionDef)} for name, t in trees.items()}
  out = {}
  for name, tree in trees.items():
    if only is not None and name not in only:
      continue
    mod = name[:-3]
    consts = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def callee_ok(f):
      if isinstance(f, ast.Name):
        return f.id in module_fns[mod]
      if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id == "_ops" or f.attr in module_fns.get(f.value.id, ())
      return False

    def chain_top(node, weights):
      """The outermost constant-key subscript of a weight name, or None."""
      if not (isinstance(node, ast.Name) and node.id in weights):
        return None
      while isinstance(parent.get(node), ast.Subscript) and parent[node].value is node:
        key = parent[node].slice
        if not (isinstance(key, (ast.Constant, ast.JoinedStr)) or (isinstance(key, ast.Name) and key.id in consts)):
          break  # params[tokens]: not a parameter subtree
        node = parent[node]
      return node

    bad = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
      weights = {a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg in WEIGHT_ARGS}
      if not weights:
        continue
      for _ in range(3):  # aliases of aliases
        for n in ast.walk(fn):
          if (
              isinstance(n, ast.Assign)
              and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name)
              and chain_top(_root(n.value), weights) is n.value
          ):
            weights.add(n.targets[0].id)
      for n in ast.walk(fn):
        if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in weights):
          continue
        top = chain_top(n, weights)
        up = parent[top]
        if isinstance(up, ast.keyword):
          up = parent[up]
        ok = (isinstance(up, ast.Call) and top is not up.func and callee_ok(up.func)) or (
            isinstance(up, ast.Assign)
            and up.value is top
            and len(up.targets) == 1
            and isinstance(up.targets[0], ast.Name)
        )
        if not ok:
          seg = ast.get_source_segment(sources[name], up) or ast.dump(up)
          # compared as ast.unparse gives them, so the allow-list does not depend on the source's formatting (quotes)
          if ast.unparse(up) not in {
              ast.unparse(ast.parse(s, mode="eval").body) for s in ALLOWED_WEIGHT_USES.get(name, ())
          }:
            bad.append(seg)
    out[name] = sorted(set(bad))
  return out


def _root(node):
  while isinstance(node, ast.Subscript):
    node = node.value
  return node


def code_lines(src):
  """`src`'s lines with comments and docstrings blanked: what the regexes read."""
  lines = src.splitlines()
  for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
      for i in range(node.lineno - 1, node.end_lineno):
        lines[i] = ""
  return [l.split("#")[0] for l in lines]


class CallSiteTest(unittest.TestCase):
  """Every sharding-sensitive call goes through `_ops` (the explicit conversion
  edits one file), and the model modules make no sharding call of their own
  (model code never calls reshard, with_sharding_constraint or out_sharding=
  directly; `src/maxtext/experimental/mpmd/jax/sharding_mode.py`, through `_ops`, does)."""

  MODEL_MODULES = ("layers.py", "gdn.py", "attention.py", "moe.py", "model.py")
  FORBIDDEN = (
      # the DeepSeek functions, reached through `_ops` only
      r"\bds_layers\.",
      r"\bds_moe\.",
      r"\bds_attention\.",
      r"import (layers|moe|attention) as ds_",
      r"from maxtext\.experimental\.mpmd\.jax\.deepseek import (layers|moe|attention)\b",
      # fresh arrays: a scan carry must be typed under explicit (`_ops.zeros`);
      # the [C, C] masks of the delta rule (bool, and the packed rule's fp32
      # cumulative-decay triangle) are replicated constants, fine
      r"jnp\.zeros\(",
      r"jnp\.(full|zeros_like|full_like|empty)\(",
      r"jnp\.ones(_like)?\((?!\(c, c\), dtype=(bool|jnp\.float32)\))",
      r"jnp\.broadcast_to\(",
      # ops that need an out_sharding under explicit
      r"(?<!_ops\.)conv_general_dilated\(",
      r"(?<!_ops\.)dot_general\(",
      r"jnp\.einsum\(",
      r"jnp\.dot\(",
      r"jnp\.take\(",
      r"\.at\[",
      # weights touched other than through `_ops` (`weight_misuses` does the rest)
      r"jnp\.asarray\(params",
      r"jnp\.asarray\(kernel",
      r"\b(params|kernel)(\[[^]]*\])*\.astype\(",
      # sharding calls of the model's own
      r"with_sharding_constraint",
      r"\breshard\(",
      r"out_sharding\s*=",
      r"shard_map",
      r"auto_axes|explicit_axes",
      r"NamedSharding|PartitionSpec",
  )

  @classmethod
  def sources(cls):
    return {name: (QWEN35 / name).read_text() for name in cls.MODEL_MODULES}

  def lint(self, sources, only=None):
    """`{module: [hit, ...]}` of the regex lint and `weight_misuses` together,
    for the modules in `only` (default all; the others still tell which calls
    are calls of the model's own functions)."""
    only = tuple(sources) if only is None else tuple(only)
    pats = [re.compile(p) for p in self.FORBIDDEN]
    out = {}
    for name in only:
      code = code_lines(sources[name])
      out[name] = [f"{p.pattern}: {l.strip()}" for p in pats for l in code if p.search(l)]
    for name, bad in weight_misuses(sources, only).items():
      out[name] += [f"weight use: {b}" for b in bad]
    return out

  def test_model_modules_only_reach_sharding_sensitive_ops_through_ops(self):
    self.assertEqual(self.lint(self.sources()), {name: [] for name in self.MODEL_MODULES})

  def test_the_lint_catches_the_likely_bypasses(self):
    """Each of these edits computes the same thing without `_ops`, or adds a
    sharding call of the model's own; the lint must flag every one."""
    sh = "jax.sharding.PartitionSpec"
    mutations = (
        ("moe.py", "_ops.dot_general(", "jax.lax.dot_general("),
        ("moe.py", '_ops.use_weight(params["kernel"], gate_dtype, config)', "params['kernel'].astype(gate_dtype)"),
        (
            "moe.py",
            '_ops.use_weight(params["kernel"], gate_dtype, config)',
            "jnp.asarray(routed_kernel := params['kernel'], gate_dtype)",
        ),
        (
            "moe.py",
            '  kernel = _ops.use_weight(params["kernel"], gate_dtype, config)\n',
            "  cast = lambda w: w.astype(gate_dtype)\n  kernel = cast(params['kernel'])\n",
        ),
        (
            "model.py",
            '_ops.embed(params["token_embedder"], tokens, config)',
            "params['token_embedder']['embedding'].astype(config.dtype)[tokens]",
        ),
        (
            "model.py",
            '  dec = params["decoder"]\n',
            "  dec = jax.tree.map(lambda w: w.astype(config.dtype), params['decoder'])\n",
        ),
        (
            "attention.py",
            '_ops.dense(params["out"], out, config)',
            "jnp.einsum('bsk,ke->bse', out, params['out']['kernel'].astype(config.dtype))",
        ),
        ("attention.py", '_ops.dense(params["out"], out, config)', "jax.lax.dot(out, w := params['out']['kernel'])"),
        ("gdn.py", '_ops.small_weight(params["A_log"], gate_dtype, config)', "params['A_log'] + 0"),
        (
            "gdn.py",
            '_ops.small_weight(params["A_log"], gate_dtype, config)',
            "params.get('A_log').astype(gate_dtype)",
        ),
        ("gdn.py", "_ops.small_weight(kernel, jnp.float32, config)", "kernel.astype(jnp.float32)"),
        # the packed path's contractions and the block doubling's identity (E1)
        ("gdn.py", "_ops.segment_cumsum(cum, g_c)", "jnp.einsum('bnhij,bnhj->bnhi', cum, g_c)"),
        ("gdn.py", "_ops.conv_taps(taps, w[:, 0, :])", "jnp.einsum('bskc,kc->bsc', taps, w[:, 0, :])"),
        (
            "gdn.py",
            "_ops.broadcast_like(jnp.eye(c, dtype=jnp.float32), s)",
            "jnp.broadcast_to(jnp.eye(c, dtype=jnp.float32), s.shape)",
        ),
        ("gdn.py", "out = _ops.conv_general_dilated(", "out = jax.lax.conv_general_dilated("),
        ("gdn.py", "_ops.small_weight(kernel, config.dtype, config)", "kernel * 1"),
        ("gdn.py", "_ops.zeros((b, h, dk, dv), jnp.float32, batch_dim=0)", "jnp.full((b, h, dk, dv), 0.0, jnp.float32)"),
        (
            "layers.py",
            'params = {"scale": _ops.use_weight(params["scale"], config.dtype, config) + 1.0}',
            "params = {k: v.astype(config.dtype) + 1.0 for k, v in params.items()}",
        ),
        # the fp32 norm site (norm_numerics='fp32')
        (
            "layers.py",
            '_ops.use_weight(params["scale"], jnp.float32, config)',
            "jnp.asarray(params['scale'], jnp.float32)",
        ),
        (
            "gdn.py",
            '_ops.norm_fp32(params["rms_norm"], x, config)',
            "jnp.einsum('...k,k->...k', x, params['rms_norm']['scale'])",
        ),
        # every decoder-layer weight enters through apply_blocks' `stack` / `blk`
        (
            "model.py",
            "  def body(carry, blk):\n",
            "  def body(carry, blk):\n    blk = jax.tree.map(lambda w: jnp.asarray(w, config.dtype), blk)\n",
        ),
        (
            "model.py",
            "jax.lax.scan(body, x, stack)",
            "jax.lax.scan(body, x, jax.tree.map(lambda w: w.astype(config.dtype), stack))",
        ),
        # sharding calls of the model's own
        (
            "gdn.py",
            '  conv = causal_depthwise_conv(qkv, params["conv1d"]["kernel"], config)\n',
            f"  qkv = jax.lax.with_sharding_constraint(qkv, {sh}('fsdp'))\n"
            "  conv = causal_depthwise_conv(qkv, params['conv1d']['kernel'], config)\n",
        ),
        (
            "attention.py",
            "  q = q * (config.head_dim**-0.5)\n",
            f"  q = jax.reshard(q * (config.head_dim ** -0.5), {sh}('fsdp'))\n",
        ),
        ("gdn.py", "_ops.regroup(v_raw, (b, s, hv, dv))", "jnp.reshape(v_raw, (b, s, hv, dv), out_sharding=s_out)"),
        (
            "moe.py",
            "  out = _ops.routed_experts(routed, x, weights, indices, config)\n",
            "  out = jax.sharding.auto_axes(lambda r, x, w, i: _ops.routed_experts(r, x, w, i, config), "
            "out_sharding=None)(routed, x, weights, indices)\n",
        ),
        (
            "model.py",
            '  return x + moe.sparse_moe_block(params["mlp"], h, config)\n',
            "  return x + jax.shard_map(lambda p, h: moe.sparse_moe_block(p, h, config), mesh=None, "
            "in_specs=None, out_specs=None)(params['mlp'], h)\n",
        ),
        ("gdn.py", "_ops.zeros((b, h, dk, dv), jnp.float32, batch_dim=0)", "jnp.ones((b, h, dk, dv), jnp.float32) * 0.0"),
        (
            "gdn.py",
            "_ops.gdn_core(rule, query, key, value, g, beta, config, segment_ids=segment_ids if packed else None)",
            "jax.shard_map(rule, mesh=None, in_specs=None, out_specs=None)(query, key, value, g, beta)",
        ),
    )
    clean = self.sources()
    for name, old, new in mutations:
      with self.subTest(module=name, edit=new):
        self.assertIn(old, clean[name], "the source moved on: update this mutation")
        hits = self.lint(dict(clean, **{name: clean[name].replace(old, new, 1)}), only=[name])[name]
        self.assertNotEqual(hits, [])

  def test_docstrings_are_not_code(self):
    """A docstring may name what it may not call."""
    src = (QWEN35 / "gdn.py").read_text()
    doc = '"""The Gated DeltaNet (linear-attention) layer of Qwen3.5.'
    self.assertIn(doc, src)
    edited = src.replace(doc, doc + "  MaxText runs the core in jax.shard_map, with_sharding_constraint.", 1)
    self.assertEqual(self.lint(dict(self.sources(), **{"gdn.py": edited}), only=["gdn.py"])["gdn.py"], [])

  def test_the_regroup_and_carry_sites_are_marked(self):
    src = {name: (QWEN35 / name).read_text() for name in ("gdn.py", "layers.py")}
    self.assertEqual(len(re.findall(r"_ops\.regroup\(", src["gdn.py"])), 4)  # v, z, b, a
    self.assertEqual(len(re.findall(r"_ops\.zeros\(", src["gdn.py"])), 1)  # the state carry
    # the conv kernel (plain conv, packed conv), A_log, dt_bias
    self.assertEqual(len(re.findall(r"_ops\.small_weight\(", src["gdn.py"])), 4)
    # the packed path's two contractions, and the conv in either form
    self.assertEqual(len(re.findall(r"_ops\.segment_cumsum\(", src["gdn.py"])), 1)
    self.assertEqual(len(re.findall(r"_ops\.conv_taps\(", src["gdn.py"])), 1)
    self.assertEqual(len(re.findall(r"_ops\.conv_general_dilated\(", src["gdn.py"])), 2)
    # the core is placed by `_ops.gdn_core` (config.gdn_core), its one call site
    self.assertEqual(len(re.findall(r"_ops\.gdn_core\(", src["gdn.py"])), 1)
    self.assertEqual(len(re.findall(r"chunk_gated_delta_rule\b", "\n".join(code_lines(src["gdn.py"])))), 2)
    for name in src:  # the fp32 norm: zero-centred, gated
      self.assertEqual(len(re.findall(r"_ops\.norm_fp32\(", src[name])), 1, name)


# --------------------------------------------------------------------------- #
# Model-level checks
# --------------------------------------------------------------------------- #
class ModelTest(unittest.TestCase):

  def test_inv_freq_argument_is_the_in_graph_table(self):
    """Passing `rope_inv_freq_jit` (a caller's precomputed table) == computing it in-graph, bitwise."""
    cfg = small()
    lay = qm.attention_layer_avals(cfg, 1)["attention"]["attention"]
    p = jax.tree.map(lambda a: a[0], realize({"a": lay})["a"])
    x = jnp.asarray(np.random.default_rng(0).standard_normal((2, 20, cfg.emb_dim)), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(20, dtype=jnp.int32), (2, 20))
    seg = jnp.ones((2, 20), jnp.int32)
    a = jax.jit(lambda p, x: qa.gated_attention(p, x, pos, seg, cfg))(p, x)
    b = jax.jit(lambda p, x, f: qa.gated_attention(p, x, pos, seg, cfg, f))(p, x, qa.rope_inv_freq_jit(cfg))
    self.assertTrue(bool(jnp.array_equal(a, b)))
    self.assertEqual(qa.rope_inv_freq_jit(cfg).shape, (cfg.rotary_dim // 2,))

  def test_partial_rope(self):
    """`partial_rope` (MaxText's PartialRotaryEmbedding, use_mrope=False) is the mRoPE's rotation: in fp32 the two
    agree to fp32 rounding (a division by the timescale against a multiply by its reciprocal), and the features
    past rotary_dim pass through untouched.  In bf16 it is MaxText's op sequence bit for bit -- sin / cos of
    position / timescale in fp32, each rounded to bf16, and the rotation in bf16, so a rotation in fp32 rounded
    once fails it -- with tables within a bf16 ulp of float64's, and it parts from the fp32-rotating mRoPE.
    gated_attention takes the one the config names (in bf16 the two part).  The bitwise check against MaxText
    itself is qwen35_maxtext_parity_test's (AgainstMaxTextQwen38, and the bf16 logits' qwen3.8 case set)."""
    cfg = small(dtype=jnp.float32)
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((2, 40, 3, cfg.head_dim)), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(1000, 1040, dtype=jnp.int32), (2, 40))
    part = jax.jit(lambda x: qa.partial_rope(x, pos, cfg))(x)
    mrope = jax.jit(lambda x: qa.mrope_text(x, pos, qa.rope_inv_freq(cfg), cfg))(x)
    rd = cfg.rotary_dim
    self.assertLess(float(jnp.max(jnp.abs(part - mrope))), 1e-4)
    np.testing.assert_array_equal(np.asarray(part[..., rd:]), np.asarray(x[..., rd:]))
    b16 = cfg.replace(dtype=jnp.bfloat16)
    xb = x.astype(jnp.bfloat16)
    part_b = qa.partial_rope(xb, pos, b16)  # eager, as `want`: every rounding happens
    sinusoid = pos[:, :, None, None] / qa.rope_timescale(cfg)
    sin, cos = (jnp.concatenate([f(sinusoid).astype(jnp.bfloat16)] * 2, axis=-1) for f in (jnp.sin, jnp.cos))
    x1, x2 = jnp.split(xb[..., :rd], 2, axis=-1)
    want = (xb[..., :rd] * cos) + (jnp.concatenate((-x2, x1), axis=-1) * sin)
    self.assertTrue(bool(jnp.array_equal(part_b, jnp.concatenate([want, xb[..., rd:]], axis=-1))))
    exponent = 2 * np.arange(rd // 2) / rd
    s64 = np.asarray(pos, np.float64)[:, :, None, None] / (
        cfg.rope_min_timescale * (cfg.rope_max_timescale / cfg.rope_min_timescale) ** exponent
    )
    for table, f in ((sin, np.sin), (cos, np.cos)):
      np.testing.assert_allclose(np.asarray(table[..., : rd // 2], np.float64), f(s64), rtol=0, atol=2**-8)
    self.assertFalse(bool(jnp.array_equal(part_b, qa.mrope_text(xb, pos, qa.rope_inv_freq(b16), b16))))
    lay = qm.attention_layer_avals(cfg, 1)["attention"]["attention"]
    p = jax.tree.map(lambda a: a[0], realize({"a": lay})["a"])
    h = jnp.asarray(rng.standard_normal((2, 40, cfg.emb_dim)), jnp.float32)
    seg = jnp.ones((2, 40), jnp.int32)
    attend = jax.jit(lambda p, h, c: qa.gated_attention(p, h, pos, seg, c), static_argnums=2)
    a, b = attend(p, h, cfg.replace(use_mrope=False)), attend(p, h, cfg)
    self.assertLess(float(jnp.max(jnp.abs(a - b))), 1e-5 * float(jnp.max(jnp.abs(b))))
    hb = h.astype(jnp.bfloat16)
    self.assertFalse(bool(jnp.array_equal(attend(p, hb, b16.replace(use_mrope=False)), attend(p, hb, b16))))

  def test_aux_loss_is_refused_up_front(self):
    """At construction, not on the first trace deep inside the scanned block."""
    with self.assertRaisesRegex(NotImplementedError, "load_balance_loss_weight"):
      small(load_balance_loss_weight=0.01)
    with self.assertRaisesRegex(NotImplementedError, "load_balance_loss_weight"):
      qc.from_maxtext(fake_maxtext_cfg(load_balance_loss_weight=0.01), platform="cpu")

  def test_remat_layouts(self):
    """Where the checkpoints are, with which policy and CSE setting (jaxpr only).

    The layer checkpoints carry the CSE barrier: their recomputes are mutually
    independent and XLA otherwise hoists them together (on v5p, one 397B-dims
    block's backward program: 22.6 GB of temporaries without, 12.0 GB with).
    """

    def checkpoints(cfg):
      lay = qm.block_avals(cfg, 2)
      x = jax.ShapeDtypeStruct((2, 20, cfg.emb_dim), cfg.dtype)
      pos = jax.ShapeDtypeStruct((2, 20), jnp.int32)
      jaxpr = jax.make_jaxpr(lambda p, x, q, s: qm.apply_blocks(p, x, q, s, cfg))(lay, x, pos, pos)
      found = []

      def walk(jx, depth):
        for eqn in jx.eqns:
          if eqn.primitive.name in ("checkpoint", "remat", "remat2"):
            found.append((depth, bool(eqn.params["prevent_cse"])))
          for v in eqn.params.values():
            for sub in v if isinstance(v, (list, tuple)) else [v]:
              inner = getattr(sub, "jaxpr", sub)
              if hasattr(inner, "eqns"):
                walk(inner, depth + (eqn.primitive.name in ("checkpoint", "remat", "remat2")))

      walk(jaxpr.jaxpr, 0)
      return sorted(found)

    cfg = small()
    cyc = cfg.inhomogeneous_layer_cycle_interval
    self.assertEqual(checkpoints(cfg), [(0, False)])
    self.assertEqual(checkpoints(cfg.replace(remat_granularity="layer")), [(0, True)] * cyc)
    self.assertEqual(checkpoints(cfg.replace(remat_granularity="block+layer")), [(0, False)] + [(1, True)] * cyc)
    self.assertEqual(checkpoints(cfg.replace(scan_layers=False, remat_granularity="block+layer")), [])
    self.assertFalse(cfg.replace(scan_layers=False).block_remat)
    with self.assertRaisesRegex(ValueError, "remat_policy"):
      qm.layer_remat_policy_for(cfg.replace(layer_remat_policy="save_all"))

  @pytest.mark.scheduled_only
  def test_remat_layouts_agree(self):
    """Same forward, same gradients (fp32): remat changes what is stored, not the math.

    The gradient bound is max(1e-5, 4 x floor), the floor MEASURED on the same
    weights and batch as scanned vs unrolled (`scan_layers=False`, no
    checkpoint at all): A_log's gradient is a sum with heavy cancellation, and
    fp32 reassociation between two compiled programs of the same model moves it
    by up to a few 1e-5 rel-to-max (block vs layer remat: 9e-6 at one input).
    """
    cfg = small(num_decoder_layers=4)
    params = realize(qm.param_avals(cfg))
    batch = qt.synthetic_batch(cfg.vocab_size, 2, 20, seed=1, max_segments_per_seq=3)
    out = {}
    for name, c in (
        ("block", cfg),
        ("layer", cfg.replace(remat_granularity="layer")),
        ("block+layer", cfg.replace(remat_granularity="block+layer")),
        ("unrolled", cfg.replace(scan_layers=False)),
    ):
      c = c.replace(layer_remat_policy="minimal")
      out[name] = jax.jit(jax.value_and_grad(lambda p, c=c: qt.compute_loss(p, batch, c)[0]))(params)
    floor = max(max_rel(a, b) for a, b in zip(jax.tree.leaves(out["unrolled"][1]), jax.tree.leaves(out["block"][1])))
    tol = max(1e-5, 4 * floor)
    for gran in ("layer", "block+layer"):
      self.assertEqual(float(out["block"][0]), float(out[gran][0]))
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(out["block"][1])[0], jax.tree.leaves(out[gran][1])):
        self.assertLess(max_rel(b, a), tol, f"{gran} {jax.tree_util.keystr(path)} (floor {floor:.2e})")

  @pytest.mark.scheduled_only
  def test_unscanned_matches_scanned(self):
    cfg = small(num_decoder_layers=4)
    params = realize(qm.param_avals(cfg))
    batch = qt.synthetic_batch(cfg.vocab_size, 2, 20, seed=1)
    a = jax.jit(lambda p: qt.compute_loss(p, batch, cfg)[0])(params)
    b = jax.jit(lambda p: qt.compute_loss(p, batch, cfg.replace(scan_layers=False))[0])(params)
    self.assertLess(abs(float(a) - float(b)), 1e-6 * abs(float(a)))

  @pytest.mark.scheduled_only
  def test_train_steps_reduce_the_loss(self):
    cfg = small(num_decoder_layers=4)
    params = qs.init_params(qm.param_avals(cfg), jax.random.key(0))
    batch = qt.synthetic_batch(cfg.vocab_size, 4, 20, seed=2)
    tcfg = qt.TrainConfig()
    opt = qt.adamw_init(params, tcfg)
    step = jax.jit(lambda p, o: qt.train_step(p, o, batch, cfg, tcfg, lambda _: 3e-3))
    losses = []
    for _ in range(4):
      params, opt, metrics = step(params, opt)
      losses.append(float(metrics["loss"]))
    self.assertTrue(all(np.isfinite(losses)), losses)
    self.assertLess(losses[-1], losses[0], losses)


class FromMaxTextBase2Test(unittest.TestCase):
  """from_maxtext carries MaxText's sa_use_base2_exp, both values."""

  def test_carries_sa_use_base2_exp(self):
    for v in (False, True):
      self.assertIs(qc.from_maxtext(fake_maxtext_cfg(sa_use_base2_exp=v), platform="cpu").sa_use_base2_exp, v)


if __name__ == "__main__":
  unittest.main()
