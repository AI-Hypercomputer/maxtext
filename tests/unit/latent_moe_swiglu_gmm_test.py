# Copyright 2023–2026 Google LLC
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

"""Numerics tests (Pallas interpret mode) for the fused SwiGLU latent MoE expert FFN."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from maxtext.kernels.latent_moe import swiglu_gmm
import numpy as np


_SMALL_TILING = swiglu_gmm.LatentFFNTiling(
    gate_up=(128, 128, 256),  # tiles_k = 2 exercises the fp32 accumulators.
    wo_fwd=(128, 512, 128),
    swiglu_bwd=(128, 128, 256),
    dx=(128, 256, 128),
    dwi=(128, 128, 256),
    dwo=(128, 128, 256),
)
_FULL_K_TILING = swiglu_gmm.LatentFFNTiling(
    gate_up=(128, 256, 256),
    wo_fwd=(128, 512, 128),
    swiglu_bwd=(64, 256, 256),
    dx=(128, 512, 128),
    dwi=(128, 256, 256),
    dwo=(128, 256, 512),
)


def _reference_ffn(x, wi_0, wi_1, wo, group_sizes):
  """Unfused fp32 reference on the (bf16-valued) inputs; rows past sum(group_sizes) are zero."""
  f32 = jnp.float32
  x, wi_0, wi_1, wo = (t.astype(f32) for t in (x, wi_0, wi_1, wo))
  hp = jax.lax.Precision.HIGHEST
  a = jax.lax.ragged_dot(x, wi_0, group_sizes, precision=hp)
  b = jax.lax.ragged_dot(x, wi_1, group_sizes, precision=hp)
  h = jax.nn.silu(a) * b
  return jax.lax.ragged_dot(h, wo, group_sizes, precision=hp)


def _rel_err(got, want):
  got = np.asarray(got, np.float32)
  want = np.asarray(want, np.float32)
  scale = max(float(np.max(np.abs(want))), 1e-6)
  return float(np.max(np.abs(got - want))) / scale


class LatentExpertFFNTest(parameterized.TestCase):

  def _inputs(self, m, d, f, e, seed=0):
    k = jax.random.split(jax.random.key(seed), 5)
    bf16 = jnp.bfloat16
    x = jax.random.normal(k[0], (m, d), jnp.float32).astype(bf16)
    wi_0 = (jax.random.normal(k[1], (e, d, f), jnp.float32) * d**-0.5).astype(bf16)
    wi_1 = (jax.random.normal(k[2], (e, d, f), jnp.float32) * d**-0.5).astype(bf16)
    wo = (jax.random.normal(k[3], (e, f, d), jnp.float32) * f**-0.5).astype(bf16)
    dy = jax.random.normal(k[4], (m, d), jnp.float32).astype(bf16)
    return x, wi_0, wi_1, wo, dy

  @parameterized.named_parameters(
      ("ragged_empty_padding", (100, 0, 237, 50), _SMALL_TILING),
      ("ragged_empty_padding_full_k", (100, 0, 237, 50), _FULL_K_TILING),
      ("aligned_full", (128, 128, 128, 128), _SMALL_TILING),
      ("single_group", (0, 0, 449, 0), _SMALL_TILING),
      ("leading_empty_unaligned", (0, 3, 129, 300), _FULL_K_TILING),
      (
          "pretransposed_dot_general",
          (100, 0, 237, 50),
          dataclasses.replace(_SMALL_TILING, pretranspose_weights=True, tgmm_lhs_mode="dot_general"),
      ),
      (
          "bwd_single_buffered_weights",
          (100, 0, 237, 50),
          dataclasses.replace(_FULL_K_TILING, swiglu_bwd=(128, 256, 256), swiglu_bwd_weight_buffers=1),
      ),
  )
  def test_forward_and_grads_match_reference(self, group_sizes, tiling):
    m, d, f, e = 512, 256, 512, 4
    x, wi_0, wi_1, wo, dy = self._inputs(m, d, f, e)
    gs = jnp.asarray(group_sizes, jnp.int32)
    valid = int(sum(group_sizes))
    row_mask = (jnp.arange(m) < valid)[:, None]

    def fused(x, wi_0, wi_1, wo):
      return swiglu_gmm.latent_expert_ffn(x, wi_0, wi_1, wo, gs, tiling=tiling, interpret=True)

    def ref(x, wi_0, wi_1, wo):
      return _reference_ffn(x, wi_0, wi_1, wo, gs)

    def loss(fn, *args):
      y = fn(*args).astype(jnp.float32)
      return jnp.sum(jnp.where(row_mask, y * dy.astype(jnp.float32), 0.0))

    y = fused(x, wi_0, wi_1, wo)
    y_ref = ref(x, wi_0, wi_1, wo)
    self.assertEqual(y.dtype, jnp.bfloat16)
    self.assertLess(_rel_err(y[:valid], y_ref[:valid]), 2e-2)

    grads = jax.grad(lambda *a: loss(fused, *a), argnums=(0, 1, 2, 3))(x, wi_0, wi_1, wo)
    grads_ref = jax.grad(lambda *a: loss(ref, *a), argnums=(0, 1, 2, 3))(
        *(t.astype(jnp.float32) for t in (x, wi_0, wi_1, wo))
    )
    names = ("dx", "dwi_0", "dwi_1", "dwo")
    for name, g, g_ref, primal in zip(names, grads, grads_ref, (x, wi_0, wi_1, wo)):
      self.assertEqual(g.shape, primal.shape, name)
      self.assertEqual(g.dtype, primal.dtype, name)
      g, g_ref = np.asarray(g, np.float32), np.asarray(g_ref, np.float32)
      self.assertTrue(np.all(np.isfinite(g if name != "dx" else g[:valid])), name)
      if name == "dx":
        g, g_ref = g[:valid], g_ref[:valid]
      self.assertLess(_rel_err(g, g_ref), 2e-2, name)

    # Empty experts get exactly-zero weight gradients.
    for gi, size in enumerate(group_sizes):
      if size == 0:
        for name, g in zip(names[1:], grads[1:]):
          self.assertEqual(float(jnp.max(jnp.abs(g[gi].astype(jnp.float32)))), 0.0, f"{name}[{gi}]")

  def test_padding_rows_do_not_leak_into_weight_grads(self):
    m, d, f, e = 512, 256, 512, 4
    x, wi_0, wi_1, wo, dy = self._inputs(m, d, f, e, seed=1)
    gs = jnp.asarray((64, 100, 0, 200), jnp.int32)
    valid = 364
    # Garbage (large / non-finite) values in the padding rows of x and dy.
    x_bad = x.at[valid:].set(jnp.inf)
    dy_bad = dy.at[valid:].set(jnp.nan)

    def wgrads(x_in, dy_in):
      _, vjp = jax.vjp(
          lambda w0, w1, w2: swiglu_gmm.latent_expert_ffn(x_in, w0, w1, w2, gs, tiling=_SMALL_TILING, interpret=True),
          wi_0,
          wi_1,
          wo,
      )
      return vjp(dy_in)

    clean = wgrads(x, dy.at[valid:].set(0))
    dirty = wgrads(x_bad, dy_bad)
    for g_clean, g_dirty in zip(clean, dirty):
      np.testing.assert_array_equal(np.asarray(g_clean, np.float32), np.asarray(g_dirty, np.float32))

  def test_group_offset_matches_local_slice(self):
    m, d, f = 512, 256, 512
    e_local, offset = 2, 1
    x, wi_0, wi_1, wo, dy = self._inputs(m, d, f, e_local, seed=2)
    # 4 global groups, the local shard owns groups [1, 3).
    global_sizes = jnp.asarray((70, 150, 90, 60), jnp.int32)
    start, local = 70, 240

    def fused(x, wi_0, wi_1, wo):
      return swiglu_gmm.latent_expert_ffn(
          x, wi_0, wi_1, wo, global_sizes, group_offset=offset, tiling=_SMALL_TILING, interpret=True
      )

    y = fused(x, wi_0, wi_1, wo)
    local_sizes = jnp.asarray((150, 90), jnp.int32)
    x_local = jnp.pad(x[start : start + local], ((0, m - local), (0, 0)))
    y_ref = _reference_ffn(x_local, wi_0, wi_1, wo, local_sizes)
    self.assertLess(_rel_err(y[start : start + local], y_ref[:local]), 2e-2)

    row_mask = ((jnp.arange(m) >= start) & (jnp.arange(m) < start + local))[:, None]
    dyf = dy.astype(jnp.float32)
    g = jax.grad(
        lambda *a: jnp.sum(jnp.where(row_mask, fused(*a).astype(jnp.float32) * dyf, 0.0)), argnums=(1, 2, 3)
    )(x, wi_0, wi_1, wo)
    dy_local = jnp.pad(dyf[start : start + local], ((0, m - local), (0, 0)))
    g_ref = jax.grad(
        lambda w0, w1, w2: jnp.sum(_reference_ffn(x_local, w0, w1, w2, local_sizes) * dy_local), argnums=(0, 1, 2)
    )(*(t.astype(jnp.float32) for t in (wi_0, wi_1, wo)))
    for gi, gr in zip(g, g_ref):
      self.assertLess(_rel_err(gi, gr), 2e-2)

  def test_unpadded_rows_are_padded_internally(self):
    m, d, f, e = 300, 256, 512, 4  # 300 is not a multiple of tm=128.
    x, wi_0, wi_1, wo, _ = self._inputs(m, d, f, e, seed=3)
    gs = jnp.asarray((50, 0, 120, 100), jnp.int32)
    y = swiglu_gmm.latent_expert_ffn(x, wi_0, wi_1, wo, gs, tiling=_SMALL_TILING, interpret=True)
    self.assertEqual(y.shape, (m, d))
    y_ref = _reference_ffn(x, wi_0, wi_1, wo, gs)
    self.assertLess(_rel_err(y[:270], y_ref[:270]), 2e-2)


if __name__ == "__main__":
  absltest.main()
