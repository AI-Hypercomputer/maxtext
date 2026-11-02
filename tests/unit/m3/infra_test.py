# Copyright 2023–2026 Google LLC
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

"""Tests for the m3 remat transform (`infra/remat.py`)."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.infra import remat


class RematPolicyTest(parameterized.TestCase):

  def test_none_means_no_remat(self):
    self.assertIsNone(remat.remat_policy(None))
    self.assertIsNone(remat.remat_policy("NONE"))

  @parameterized.parameters(*[p for p in remat.REMAT_POLICIES if p != "none"])
  def test_policies_are_callable(self, name):
    self.assertTrue(callable(remat.remat_policy(name.upper(), ["hidden_states"], ["query_proj"])))

  def test_unknown_policy_raises(self):
    with self.assertRaises(ValueError):
      remat.remat_policy("everything")

  @parameterized.parameters("full", "matmul_without_batch")
  def test_remat_keeps_gradients(self, name):
    """`nnx.remat` with the policy changes memory, not math."""
    model = nnx.Linear(4, 4, rngs=nnx.Rngs(0))
    x = jnp.arange(8.0).reshape(2, 4)

    def loss(m, fn):
      return jnp.sum(fn(m, x) ** 2)

    plain = nnx.grad(lambda m: loss(m, lambda mm, xx: mm(xx)))(model)
    rematted = nnx.grad(
        lambda m: loss(m, nnx.remat(lambda mm, xx: mm(xx), policy=remat.remat_policy(name)))  # pylint: disable=unnecessary-lambda
    )(model)
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(rematted)):
      np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
