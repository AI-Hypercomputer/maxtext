# Copyright 2023-2026 Google LLC
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

"""``moe_global_load_balancing``: the OLMoE3 load-balancing loss over a sharded batch equals the global-batch one.

The routed MoE splits the tokens across devices inside its ``shard_map`` when the batch is sharded by the expert
axis, and returns the loss through ``out_specs=P()`` with ``check_vma`` off. This builds the same ``shard_map``
around ``OLMoE3LatentRoutedMoE.load_balance_loss`` on 8 devices, with the batch split over a data and an expert
axis, and compares the value and the gradient with the function on the whole batch. (The full layer under expert
parallelism needs ragged_dot shapes the CPU backend rejects; without it, the routed path gets the whole batch.)

Needs 8 devices: on CPU, run this file in its own process (it sets the host device count before importing JAX);
it skips if JAX already started with fewer.

    JAX_PLATFORMS=cpu pytest tests/unit/olmoe3_global_lb_test.py
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import types  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position

import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from jax.sharding import Mesh, PartitionSpec as P  # pylint: disable=wrong-import-position

from maxtext.models import olmoe3  # pylint: disable=wrong-import-position

NUM_DEVICES = 8
NUM_EXPERTS, TOP_K, BATCH, SEQ = 32, 4, 8, 64


def _router(mesh, global_lb, sparse=True):
  """The attributes ``load_balance_loss`` reads, without building a layer."""
  cfg = types.SimpleNamespace(moe_global_load_balancing=global_lb, sparse_matmul=sparse, load_balance_loss_weight=0.01)
  return types.SimpleNamespace(config=cfg, mesh=mesh, num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K)


def _lb(router, logits):
  """The loss as the routed path computes it: softmax probabilities, top-k by logit."""
  probs = jax.nn.softmax(logits, axis=-1)
  _, top_k = jax.lax.top_k(logits, TOP_K)
  return olmoe3.OLMoE3LatentRoutedMoE.load_balance_loss(router, top_k, probs)


@unittest.skipUnless(jax.device_count() >= NUM_DEVICES, f"needs {NUM_DEVICES} devices; run this file on its own")
class OLMoE3GlobalLoadBalancingTest(unittest.TestCase):
  """Sharded with global balancing == whole batch, in value and in gradient; per-device balancing is not."""

  def setUp(self):
    super().setUp()
    self.mesh = Mesh(np.array(jax.devices()[:NUM_DEVICES]).reshape(4, 2), ("data", "expert"))
    # Skewed routing that differs by row, so every device's counts differ from the batch's.
    base = jax.random.normal(jax.random.PRNGKey(0), (BATCH, SEQ, NUM_EXPERTS))
    skew = jnp.linspace(0.0, 3.0, NUM_EXPERTS)[None, None, :] * jnp.cos(jnp.arange(BATCH))[:, None, None]
    self.logits = base + skew

  def _whole_batch(self):
    return jax.value_and_grad(lambda x: _lb(_router(self.mesh, False, sparse=False), x))(self.logits)

  def _sharded(self, global_lb):
    router = _router(self.mesh, global_lb)
    return jax.shard_map(
        lambda x: _lb(router, x),
        mesh=self.mesh,
        in_specs=P(("data", "expert")),
        out_specs=P(),
        check_vma=False,
    )

  def test_value_and_gradient_match_whole_batch(self):
    ref_value, ref_grad = self._whole_batch()
    value, grad = jax.jit(jax.value_and_grad(self._sharded(True)))(self.logits)
    np.testing.assert_allclose(float(value), float(ref_value), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(ref_grad), rtol=1e-5, atol=1e-9)

  def test_per_device_balancing_differs(self):
    """Without the flag the value is one device's and the gradient uses its own counts: this test can tell."""
    ref_value, ref_grad = self._whole_batch()
    value, grad = jax.jit(jax.value_and_grad(self._sharded(False)))(self.logits)
    self.assertGreater(abs(float(value) - float(ref_value)), 1e-3 * abs(float(ref_value)))
    self.assertGreater(float(jnp.max(jnp.abs(grad - ref_grad))), 1e-3 * float(jnp.max(jnp.abs(ref_grad))))

  def test_counts_carry_no_gradient(self):
    """Only the probabilities are differentiated, as in OLMo-core: d lb / d probs_e is the global count of e."""
    router = _router(self.mesh, True)
    _, top_k = jax.lax.top_k(self.logits, TOP_K)
    probs = jax.nn.softmax(self.logits, axis=-1)
    sharded = jax.shard_map(
        lambda t, p: olmoe3.OLMoE3LatentRoutedMoE.load_balance_loss(router, t, p),
        mesh=self.mesh,
        in_specs=(P(("data", "expert")), P(("data", "expert"))),
        out_specs=P(),
        check_vma=False,
    )
    grad = jax.jit(jax.grad(sharded, argnums=1))(top_k, probs)
    counts = jax.nn.one_hot(top_k, NUM_EXPERTS).sum(axis=(0, 1, 2))
    tokens = BATCH * SEQ
    expected = 0.01 * NUM_EXPERTS / (TOP_K * tokens) * counts / tokens
    np.testing.assert_allclose(np.asarray(grad), np.broadcast_to(expected, probs.shape), rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
