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

"""Tests for the Pallas Gated Delta Net kernels.

Compares the fused inter-chunk scan Pallas kernel (interpret mode, so the
test runs on CPU) against a pure-JAX lax.scan reference implementing the
same recurrence, for the forward outputs, the final state, and all input
gradients — including a non-zero initial state and a non-zero final-state
cotangent.

Also compares the blockwise unit-lower-triangular inversion kernel against
jax.scipy.linalg.solve_triangular, forward and gradients, including inputs
whose inverse grows far beyond the |s| < 1 radius where naive Neumann
doubling would overflow.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.models.qwen3 import jax_chunk_gated_delta_rule
from maxtext.kernels.attention.gated_delta_network import gdn_inter_chunk_scan, invert_unit_lower


def reference_inter_chunk_scan(w, u, q, k, g, h0):  # pylint: disable=too-many-positional-arguments
  """Pure-JAX reference for the inter-chunk recurrence (float32).

  Shapes: w/q/k [B, N, H, C, D_k], u [B, N, H, C, D_v], g [B, N, H, C],
  h0 [B, H, D_k, D_v].
  """
  chunk_size = q.shape[-2]

  def scan_body(h, xs):
    w_c, u_c, q_c, k_c, g_c = xs
    exp_g = jnp.exp(g_c)
    q_g = q_c * exp_g[..., None]
    attn_inter = jnp.einsum("bhcd,bhde->bhce", q_g, h)
    v_new = u_c - jnp.einsum("bhcd,bhde->bhce", w_c, h)
    p = jnp.einsum("bhcd,bhed->bhce", q_c, k_c)
    g_diff = g_c[..., :, None] - g_c[..., None, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
    o_c = attn_inter + jnp.einsum("bhce,bhef->bhcf", p * decay, v_new)
    g_last = g_c[..., -1]
    dvec = jnp.exp(g_last[..., None] - g_c)
    kd = k_c * dvec[..., None]
    h_new = h * jnp.exp(g_last)[..., None, None] + jnp.einsum("bhcd,bhce->bhde", kd, v_new)
    return h_new, o_c

  xs = tuple(jnp.moveaxis(x, 1, 0) for x in (w, u, q, k, g))
  h_final, o = jax.lax.scan(scan_body, h0, xs)
  return jnp.moveaxis(o, 0, 1), h_final


class GdnPallasKernelTest(unittest.TestCase):

  def setUp(self):
    batch, num_chunks, num_heads, chunk_size, d_k, d_v = 2, 4, 3, 128, 128, 128
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    scale = 0.3
    # In the real algorithm w comes from the WY factorization, which keeps the
    # state recurrence contractive; scale it so ||w @ h|| < ||h|| holds here
    # too, otherwise the state grows exponentially across chunks and float32
    # cancellation noise dominates the gradient comparison.
    w_scale = 0.05
    self.w = w_scale * jax.random.normal(keys[0], (batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32)
    self.u = scale * jax.random.normal(keys[1], (batch, num_chunks, num_heads, chunk_size, d_v), jnp.float32)
    self.q = scale * jax.random.normal(keys[2], (batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32)
    self.k = scale * jax.random.normal(keys[3], (batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32)
    # Cumulative log-decay: negative and decreasing within a chunk.
    g_raw = -jax.nn.softplus(jax.random.normal(keys[4], (batch, num_chunks, num_heads, chunk_size), jnp.float32))
    self.g = jnp.cumsum(0.05 * g_raw, axis=-1)
    self.h0 = scale * jax.random.normal(keys[5], (batch, num_heads, d_k, d_v), jnp.float32)

  def run_both(self, h0):
    o_ref, h_ref = reference_inter_chunk_scan(self.w, self.u, self.q, self.k, self.g, h0)
    o_ker, h_ker = gdn_inter_chunk_scan(self.w, self.u, self.q, self.k, self.g, h0, True, jnp.float32)
    return (o_ref, h_ref), (o_ker, h_ker)

  def test_forward_zero_initial_state(self):
    (o_ref, h_ref), (o_ker, h_ker) = self.run_both(jnp.zeros_like(self.h0))
    np.testing.assert_allclose(np.asarray(o_ker), np.asarray(o_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(h_ker), np.asarray(h_ref), rtol=1e-5, atol=1e-5)

  def test_forward_nonzero_initial_state(self):
    (o_ref, h_ref), (o_ker, h_ker) = self.run_both(self.h0)
    np.testing.assert_allclose(np.asarray(o_ker), np.asarray(o_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(h_ker), np.asarray(h_ref), rtol=1e-5, atol=1e-5)

  def test_gradients_output_only(self):
    def loss_ref(*args):
      o, _ = reference_inter_chunk_scan(*args)
      return jnp.sum(o**2)

    def loss_ker(*args):
      o, _ = gdn_inter_chunk_scan(*args, True, jnp.float32)
      return jnp.sum(o**2)

    args = (self.w, self.u, self.q, self.k, self.g, self.h0)
    grads_ref = jax.grad(loss_ref, argnums=tuple(range(6)))(*args)
    grads_ker = jax.grad(loss_ker, argnums=tuple(range(6)))(*args)
    for g_ref, g_ker in zip(grads_ref, grads_ker):
      g_ref, g_ker = np.asarray(g_ref), np.asarray(g_ker)
      # Kernel and reference associate the float32 reductions differently, so
      # near-zero elements carry rounding noise; compare against an absolute
      # tolerance scaled to the gradient's magnitude.
      atol = 1e-3 * np.abs(g_ref).max()
      np.testing.assert_allclose(g_ker, g_ref, rtol=2e-3, atol=atol)

  def test_bfloat16_inputs_differentiable(self):
    # Production passes bf16 tensors; cotangent dtypes must match the primals.
    args = (
        self.w.astype(jnp.bfloat16),
        self.u.astype(jnp.bfloat16),
        self.q.astype(jnp.bfloat16),
        self.k.astype(jnp.bfloat16),
        self.g,
        self.h0,
    )

    def loss(*a):
      o, h_final = gdn_inter_chunk_scan(*a, True, jnp.bfloat16)
      return jnp.sum(o**2) + jnp.sum(h_final**2)

    grads = jax.grad(loss, argnums=tuple(range(6)))(*args)
    for arg, grad in zip(args, grads):
      self.assertEqual(grad.dtype, arg.dtype)
      self.assertTrue(np.all(np.isfinite(np.asarray(grad, dtype=np.float32))))

  def test_gradients_with_final_state_cotangent(self):
    def loss_ref(*args):
      o, h_final = reference_inter_chunk_scan(*args)
      return jnp.sum(o**2) + jnp.sum(jnp.sin(h_final))

    def loss_ker(*args):
      o, h_final = gdn_inter_chunk_scan(*args, True, jnp.float32)
      return jnp.sum(o**2) + jnp.sum(jnp.sin(h_final))

    args = (self.w, self.u, self.q, self.k, self.g, self.h0)
    grads_ref = jax.grad(loss_ref, argnums=tuple(range(6)))(*args)
    grads_ker = jax.grad(loss_ker, argnums=tuple(range(6)))(*args)
    for g_ref, g_ker in zip(grads_ref, grads_ker):
      g_ref, g_ker = np.asarray(g_ref), np.asarray(g_ker)
      # Kernel and reference associate the float32 reductions differently, so
      # near-zero elements carry rounding noise; compare against an absolute
      # tolerance scaled to the gradient's magnitude.
      atol = 1e-3 * np.abs(g_ref).max()
      np.testing.assert_allclose(g_ker, g_ref, rtol=2e-3, atol=atol)


def strict_lower(x):
  """Zeroes everything on or above the diagonal of the last two dims."""
  size = x.shape[-1]
  return jnp.where(jnp.tril(jnp.ones((size, size), dtype=bool), k=-1), x, 0.0)


def reference_invert(s):
  """(I + s)^{-1} via the triangular solve the kernel replaces."""
  size = s.shape[-1]
  identity = jnp.eye(size, dtype=jnp.float32)
  identity_broadcasted = jnp.broadcast_to(identity, s.shape)
  return jax.scipy.linalg.solve_triangular(identity + s, identity_broadcasted, lower=True, unit_diagonal=True)


class InvertUnitLowerKernelTest(unittest.TestCase):
  """Pallas blockwise inversion vs jax.scipy solve_triangular."""

  def assert_forward_matches(self, s):
    a_ref = np.asarray(reference_invert(s))
    a_ker = np.asarray(invert_unit_lower(s, True))
    # The inverse's small entries are differences of large intermediates, so
    # their error is set by the large entries (for both kernel and solve);
    # compare against an absolute tolerance scaled to the inverse magnitude.
    np.testing.assert_allclose(a_ker, a_ref, rtol=1e-4, atol=1e-4 * np.abs(a_ref).max())

  def assert_gradients_match(self, s):
    """Compares the kernel's analytic VJP against autodiff through the solve."""

    def loss_ref(s_):
      return jnp.sum(jnp.sin(reference_invert(s_)))

    def loss_ker(s_):
      return jnp.sum(jnp.sin(invert_unit_lower(s_, True)))

    g_ref = np.asarray(jax.grad(loss_ref)(s))
    g_ker = np.asarray(jax.grad(loss_ker)(s))
    # Kernel (analytic dS = -A^T dA A^T) and reference (autodiff through the
    # solve) associate the float32 reductions differently; compare against
    # an absolute tolerance scaled to the gradient's magnitude.
    atol = 3e-3 * np.abs(g_ref).max()
    np.testing.assert_allclose(g_ker, g_ref, rtol=2e-3, atol=atol)
    # The cotangent must be strictly lower triangular like the primal.
    np.testing.assert_array_equal(g_ker, np.asarray(strict_lower(jnp.asarray(g_ker))))

  def test_random_bounded(self):
    # num_chunks = 8 exercises the kernel's chunk-tile grouping.
    shape = (1, 8, 2, 128, 128)
    s = strict_lower(jax.random.uniform(jax.random.PRNGKey(0), shape, jnp.float32, minval=-0.5, maxval=0.5))
    self.assert_forward_matches(s)
    self.assert_gradients_match(s)

  def test_all_ones(self):
    # (I + ones)^{-1} entries reach 2^(C-2): far outside the |s| < 1 radius
    # where Neumann doubling overflows, and every entry is a signed power of
    # two, so the blockwise ladder must reproduce the solve exactly.
    shape = (1, 4, 2, 128, 128)
    s = strict_lower(jnp.ones(shape, jnp.float32))
    self.assert_forward_matches(s)

  def test_all_ones_gradients(self):
    # At C = 128 the all-ones inverse reaches 2^126 and the gradient's
    # A^T dA A^T products overflow float32 for kernel and reference alike;
    # C = 32 keeps the analytic gradient finite while staying unbounded-|s|.
    shape = (1, 4, 2, 32, 32)
    s = strict_lower(jnp.ones(shape, jnp.float32))
    self.assert_forward_matches(s)
    self.assert_gradients_match(s)

  def test_gaussian_unnormalized(self):
    # Matches the production distribution: S = (k_beta @ k^T) * decay is not
    # normalized, so entries land at roughly this scale.
    shape = (1, 8, 2, 128, 128)
    s = strict_lower(0.3 * jax.random.normal(jax.random.PRNGKey(1), shape, jnp.float32))
    self.assert_forward_matches(s)
    self.assert_gradients_match(s)


class GdnPallasDispatchTest(unittest.TestCase):
  """End-to-end: the use_pallas dispatch in jax_chunk_gated_delta_rule matches the XLA path on CPU."""

  def test_use_pallas_matches_reference_end_to_end(self):
    batch, seq, heads, dim = 1, 256, 8, 64
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    q = jax.random.normal(keys[0], (batch, seq, heads, dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, seq, heads, dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, seq, heads, dim), jnp.float32)
    g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, seq, heads), jnp.float32))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, seq, heads), jnp.float32))
    kwargs = {"chunk_size": 128, "compute_dtype": jnp.float32, "use_qk_norm_in_gdn": True}

    o_ref, _ = jax_chunk_gated_delta_rule(q, k, v, g, beta, use_pallas=False, **kwargs)
    o_pal, _ = jax_chunk_gated_delta_rule(q, k, v, g, beta, use_pallas=True, **kwargs)
    np.testing.assert_allclose(np.asarray(o_pal), np.asarray(o_ref), rtol=5e-4, atol=5e-4)

    def loss(fn_q, use_pallas):
      o, _ = jax_chunk_gated_delta_rule(fn_q, k, v, g, beta, use_pallas=use_pallas, **kwargs)
      return jnp.sum(o.astype(jnp.float32) ** 2)

    g_ref = jax.grad(lambda x: loss(x, False))(q)
    g_pal = jax.grad(lambda x: loss(x, True))(q)
    np.testing.assert_allclose(np.asarray(g_pal), np.asarray(g_ref), rtol=2e-3, atol=1e-3 * float(np.abs(g_ref).max()))


if __name__ == "__main__":
  unittest.main()
