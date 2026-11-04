"""Tests for the fused Pallas MoE unsort + combine (CPU interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels import moe_combine


def _reference(x, idx, w):
  t, k = w.shape
  rows = x[idx].astype(jnp.float32).reshape(t, k, -1)
  return jnp.einsum("tkd,tk->td", rows, w.astype(jnp.float32)).astype(x.dtype)


def _inputs(t, k, d, seed=0):
  keys = jax.random.split(jax.random.PRNGKey(seed), 3)
  x = jax.random.normal(keys[0], (t * k, d)).astype(jnp.bfloat16)
  idx = jax.random.permutation(keys[1], t * k)
  w = jax.nn.softmax(jax.random.normal(keys[2], (t, k)), axis=-1).astype(jnp.float32)
  return x, idx, w


@pytest.mark.parametrize("t,k,d,block", [(32, 4, 256, 8), (24, 8, 128, 64), (10, 2, 128, 4)])
def test_forward_matches_reference(t, k, d, block):
  x, idx, w = _inputs(t, k, d)
  out = moe_combine.combine_fwd(x, idx, w, block_tokens=block, interpret=True)
  ref = _reference(x, idx, w)
  np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(ref, np.float32), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("with_inv", [False, True])
@pytest.mark.parametrize("t,k,d", [(16, 4, 128), (64, 8, 256)])
def test_grads_match_reference(with_inv, t, k, d):
  x, idx, w = _inputs(t, k, d, seed=1)
  inv = jnp.argsort(idx) if with_inv else None
  dy = jax.random.normal(jax.random.PRNGKey(3), (t, d)).astype(jnp.bfloat16)

  def ref_fn(x_, w_):
    return _reference(x_, idx, w_)

  def pal_fn(x_, w_):
    return moe_combine.fused_unsort_combine(x_, idx, w_, inv)

  o_r, vjp_r = jax.vjp(ref_fn, x, w)
  o_p, vjp_p = jax.vjp(pal_fn, x, w)
  np.testing.assert_allclose(np.asarray(o_p, np.float32), np.asarray(o_r, np.float32), rtol=1e-2, atol=1e-2)
  dx_r, dw_r = vjp_r(dy)
  dx_p, dw_p = vjp_p(dy)
  np.testing.assert_allclose(np.asarray(dx_p, np.float32), np.asarray(dx_r, np.float32), rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(np.asarray(dw_p, np.float32), np.asarray(dw_r, np.float32), rtol=2e-2, atol=5e-2)


# Add test for moe.top_k_dense_vjp
from maxtext.layers.moe import top_k_dense_vjp


def test_top_k_dense_vjp_vs_lax_top_k():
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (4, 16))
  dy_vals = jax.random.normal(key, (4, 2))

  def ref_fn(x_):
    v, _ = jax.lax.top_k(x_, 2)
    return v

  def dense_fn(x_):
    v, _ = top_k_dense_vjp(x_, 2)
    return v

  # Forward
  ref_out = ref_fn(x)
  dense_out = dense_fn(x)
  np.testing.assert_allclose(dense_out, ref_out)

  # Backward
  _, vjp_ref = jax.vjp(ref_fn, x)
  (dx_ref,) = vjp_ref(dy_vals)

  _, vjp_dense = jax.vjp(dense_fn, x)
  (dx_dense,) = vjp_dense(dy_vals)
  np.testing.assert_allclose(dx_dense, dx_ref)


def test_top_k_dense_vjp_in_scan():
  key = jax.random.PRNGKey(1)
  xs = jax.random.normal(key, (4, 8, 16))
  dy = jax.random.normal(key, (4, 8, 2))

  def scan_fn(carry, x_):
    v, _ = top_k_dense_vjp(x_, 2)
    return carry, v

  def run_scan(xs_):
    _, vs = jax.lax.scan(scan_fn, None, xs_)
    return vs

  _, vjp_scan = jax.vjp(run_scan, xs)
  (dx,) = vjp_scan(dy)
  assert dx.shape == xs.shape


# Add test for moe_combine.fused_unsort_combine with 3D expert_outputs input
def test_moe_combine_3d_vs_2d():
  t, k, d = 32, 4, 256
  x, idx, w = _inputs(t, k, d, seed=2)
  # x is (t*k, d); 3D handoff from backend.gmm(out_3d=True) has shape (t*k, d // 128, 128)
  x_3d = x.reshape(t * k, d // 128, 128)

  dy = jax.random.normal(jax.random.PRNGKey(3), (t, d)).astype(x.dtype)

  def fn(x_):
    return moe_combine.fused_unsort_combine(x_, idx, w)

  o2, vjp2 = jax.vjp(fn, x)
  o3, vjp3 = jax.vjp(fn, x_3d)

  np.testing.assert_allclose(o3, o2, atol=1e-5)

  (dx2,) = vjp2(dy)
  (dx3,) = vjp3(dy)

  np.testing.assert_allclose(dx3.reshape(dx2.shape), dx2, atol=1e-5)


# Add test for gmm_dual_up
from maxtext.kernels.megablox import ops


def test_gmm_dual_up_vs_separate():
  key = jax.random.PRNGKey(42)
  m, k, n = 32, 128, 128
  num_groups = 4

  lhs = jax.random.normal(key, (m, k)).astype(jnp.float32)
  rhs0 = jax.random.normal(jax.random.PRNGKey(43), (num_groups, k, n)).astype(jnp.float32)
  rhs1 = jax.random.normal(jax.random.PRNGKey(44), (num_groups, k, n)).astype(jnp.float32)
  dy0 = jax.random.normal(jax.random.PRNGKey(45), (m, n)).astype(jnp.float32)
  dy1 = jax.random.normal(jax.random.PRNGKey(46), (m, n)).astype(jnp.float32)
  group_sizes = jnp.full((num_groups,), m // num_groups, jnp.int32)
  tiling = (16, 128, 128, 16, 128, 128, 16, 128, 128)

  def dual_up_fn(l, r0, r1):
    return ops.gmm_dual_up(l, r0, r1, group_sizes, tiling=tiling, interpret=True)

  def sep_fn(l, r0, r1):
    out0 = ops.gmm(l, r0, group_sizes, tiling=tiling, interpret=True)
    out1 = ops.gmm(l, r1, group_sizes, tiling=tiling, interpret=True)
    return out0, out1

  o0_d, o1_d = dual_up_fn(lhs, rhs0, rhs1)
  o0_s, o1_s = sep_fn(lhs, rhs0, rhs1)

  np.testing.assert_allclose(o0_d, o0_s, atol=1e-4)
  np.testing.assert_allclose(o1_d, o1_s, atol=1e-4)

  # Backward
  (_, vjp_d) = jax.vjp(dual_up_fn, lhs, rhs0, rhs1)
  dx_d, dr0_d, dr1_d = vjp_d((dy0, dy1))

  (_, vjp_s) = jax.vjp(sep_fn, lhs, rhs0, rhs1)
  dx_s, dr0_s, dr1_s = vjp_s((dy0, dy1))

  np.testing.assert_allclose(dx_d, dx_s, atol=1e-3)
  np.testing.assert_allclose(dr0_d, dr0_s, atol=1e-3)
  np.testing.assert_allclose(dr1_d, dr1_s, atol=1e-3)
