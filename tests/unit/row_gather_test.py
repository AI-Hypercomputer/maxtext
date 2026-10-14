"""Tests for the Pallas row gather and its MoE custom-VJP wrappers (CPU interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels import row_gather
from maxtext.kernels import sort_activations
from maxtext.layers import moe


@pytest.mark.parametrize("n,m,d,block", [(64, 256, 256, 64), (48, 96, 128, 512), (32, 40, 128, 16)])
def test_take_rows_matches_xla(n, m, d, block):
  x = jax.random.normal(jax.random.PRNGKey(0), (n, d)).astype(jnp.bfloat16)
  idx = jax.random.randint(jax.random.PRNGKey(1), (m,), 0, n)
  out = row_gather.take_rows(x, idx, block_rows=block, interpret=True)
  np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(x[idx], np.float32))


def _sel(key, t, k, e):
  return jax.vmap(lambda kk: jax.random.permutation(kk, e)[:k])(jax.random.split(key, t))


def test_route_pallas_matches_route():
  t, k, e, d = 32, 4, 16, 128
  x = jax.random.normal(jax.random.PRNGKey(0), (t, d)).astype(jnp.bfloat16)
  sel = _sel(jax.random.PRNGKey(1), t, k, e)
  ct = jax.random.normal(jax.random.PRNGKey(2), (t * k, d)).astype(jnp.bfloat16)
  o_r, vjp_r = jax.vjp(lambda a: sort_activations.route(a, sel, False), x)
  o_p, vjp_p = jax.vjp(lambda a: moe._route_pallas(a, sel), x)  # pylint: disable=protected-access
  np.testing.assert_array_equal(np.asarray(o_p, np.float32), np.asarray(o_r, np.float32))
  np.testing.assert_allclose(
      np.asarray(vjp_p(ct)[0], np.float32), np.asarray(vjp_r(ct)[0], np.float32), rtol=1e-2, atol=1e-2
  )


@pytest.mark.parametrize("with_inv", [False, True])
def test_sort_pallas_matches_custom(with_inv):
  m, d = 128, 128
  x = jax.random.normal(jax.random.PRNGKey(0), (m, d)).astype(jnp.bfloat16)
  perm = jax.random.permutation(jax.random.PRNGKey(1), m)
  inv = jnp.argsort(perm) if with_inv else None
  ct = jax.random.normal(jax.random.PRNGKey(2), (m, d)).astype(jnp.bfloat16)
  o_c, vjp_c = jax.vjp(
      lambda a: moe._sort_activations_pallas(a, perm, inv_sort_indices=inv), x  # pylint: disable=protected-access
  )
  o_p, vjp_p = jax.vjp(
      lambda a: moe._sort_activations_pallas(a, perm, inv_sort_indices=inv), x  # pylint: disable=protected-access
  )
  np.testing.assert_array_equal(np.asarray(o_p, np.float32), np.asarray(o_c, np.float32))
  np.testing.assert_array_equal(np.asarray(vjp_p(ct)[0], np.float32), np.asarray(vjp_c(ct)[0], np.float32))
