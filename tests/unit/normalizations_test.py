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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from maxtext.layers import normalizations
import numpy as np


def _reference_rms_norm(x, scale, epsilon, dtype, scale_offset, with_scale):
  """Reference RMSNorm using keepdims and the einsum scale multiply."""
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
  normed_fp32 = x * lax.rsqrt(mean2 + epsilon)
  y = jnp.asarray(normed_fp32, dtype)
  if not with_scale:
    return y
  effective_scale = jnp.asarray(scale, jnp.float32) + scale_offset
  if scale_offset != 0.0:
    out = jnp.einsum(
        "...k,k->...k",
        normed_fp32,
        effective_scale,
        precision=lax.Precision.HIGHEST,
    )
    return jnp.asarray(out, dtype)
  effective_scale = jnp.asarray(effective_scale, dtype)
  return jnp.einsum(
      "...k,k->...k", y, effective_scale, precision=lax.Precision.HIGHEST
  )


def _random_init(key, shape, dtype):
  return jax.random.normal(key, shape, dtype)


class RMSNormTest(parameterized.TestCase):
  """Checks RMSNorm against the reference keepdims/einsum formulation."""

  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      scale_offset=(0.0, 1.0),
      with_scale=(True, False),
      global_norm=(False, True),
  )
  def test_matches_reference(
      self, dtype, scale_offset, with_scale, global_norm
  ):
    batch, seq, heads, head_dim = 2, 4, 2, 8
    epsilon = 1e-6
    shape = (
        (batch, seq, heads, head_dim)
        if global_norm
        else (batch, seq, heads * head_dim)
    )
    x = jax.random.normal(jax.random.key(1), shape, jnp.float32) * 3.0
    cotangent = jax.random.normal(jax.random.key(2), shape, jnp.float32)

    norm_cls = (
        normalizations.GlobalRMSNorm if global_norm else normalizations.RMSNorm
    )
    norm = norm_cls(
        num_features=heads * head_dim,
        epsilon=epsilon,
        dtype=dtype,
        scale_init=_random_init,
        scale_offset=scale_offset,
        with_scale=with_scale,
        rngs=nnx.Rngs(params=0),
    )
    scale = (
        norm.scale.get_value()
        if with_scale
        else jnp.ones((heads * head_dim,), jnp.float32)
    )

    def reference(x, scale):
      out = _reference_rms_norm(
          x.reshape(shape[:2] + (-1,)),
          scale,
          epsilon,
          dtype,
          scale_offset,
          with_scale,
      )
      return out.reshape(shape)

    graphdef, state = nnx.split(norm)

    def loss(state, x):
      out = nnx.merge(graphdef, state)(x)
      return jnp.sum(jnp.asarray(out, jnp.float32) * cotangent)

    def reference_loss(scale, x):
      return jnp.sum(jnp.asarray(reference(x, scale), jnp.float32) * cotangent)

    tol = 1e-5 if dtype == jnp.float32 else 1e-2
    out = norm(x)
    expected = reference(x, scale)
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(expected, np.float32),
        rtol=tol,
        atol=tol,
    )

    state_grad, x_grad = jax.grad(loss, argnums=(0, 1))(state, x)
    ref_scale_grad, ref_x_grad = jax.grad(reference_loss, argnums=(0, 1))(
        scale, x
    )
    np.testing.assert_allclose(x_grad, ref_x_grad, rtol=tol, atol=tol)
    if with_scale:
      (scale_grad,) = jax.tree.leaves(state_grad)
      np.testing.assert_allclose(scale_grad, ref_scale_grad, rtol=tol, atol=tol)


if __name__ == "__main__":
  absltest.main()
