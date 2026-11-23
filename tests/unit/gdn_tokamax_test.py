# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks `use_tokamax_gdn`'s recurrence against `jax_chunk_gated_delta_rule`."""

import unittest

import jax
import jax.numpy as jnp
import pytest
import tokamax

from maxtext.models import qwen3


def _rel(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


@pytest.mark.cpu_only
@unittest.skipUnless(
    hasattr(tokamax.ops, "gated_delta_net"),
    "Installed Tokamax has no gated_delta_net.",
)
class TokamaxGatedDeltaRuleTest(unittest.TestCase):

  def test_matches_jax_chunk_gated_delta_rule(self):
    b, s, h_k, h_v, d = 2, 128, 2, 4, 128
    ks = jax.random.split(jax.random.PRNGKey(0), 6)
    q = jax.random.normal(ks[0], (b, s, h_k, d), jnp.float32)
    k = jax.random.normal(ks[1], (b, s, h_k, d), jnp.float32)
    v = 0.5 * jax.random.normal(ks[2], (b, s, h_v, d), jnp.float32)
    g = -jax.nn.softplus(jax.random.normal(ks[3], (b, s, h_v)))
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, s, h_v)))
    do = jax.random.normal(ks[5], (b, s, h_v, d), jnp.float32)
    kwargs = {"chunk_size": 64, "use_qk_norm_in_gdn": True, "compute_dtype": jnp.float32}

    def tokamax_loss(*args):
      return jnp.sum(qwen3.tokamax_chunk_gated_delta_rule(*args, **kwargs) * do)

    def jax_loss(*args):
      out, _ = qwen3.jax_chunk_gated_delta_rule(*args, **kwargs)
      return jnp.sum(out * do)

    args = (q, k, v, g, beta)
    out = qwen3.tokamax_chunk_gated_delta_rule(*args, **kwargs)
    out_ref, _ = qwen3.jax_chunk_gated_delta_rule(*args, **kwargs)
    self.assertLess(_rel(out, out_ref), 1e-3)
    grads = jax.grad(tokamax_loss, argnums=tuple(range(5)))(*args)
    grads_ref = jax.grad(jax_loss, argnums=tuple(range(5)))(*args)
    for name, grad, grad_ref in zip(("q", "k", "v", "g", "beta"), grads, grads_ref):
      self.assertLess(_rel(grad, grad_ref), 1e-3, name)


if __name__ == "__main__":
  unittest.main()
