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
around ``OLMoE3LatentRoutedMoE.load_balance_loss`` on 4 devices (2 x 2), with the batch split over a data and an expert
axis, and compares the value and the gradient with the function on the whole batch. (The full layer under expert
parallelism needs ragged_dot shapes the CPU backend rejects; without it, the routed path gets the whole batch.)

The sparse path sets the token-sharding axes in ``moe.LB_REDUCE_AXES`` around its ``shard_map``; the tests do the same.

Needs 4 devices (CPU CI forces 4): on CPU, run this file in its own process (it sets the host device count before importing JAX);
it skips if JAX already started with fewer.

    JAX_PLATFORMS=cpu pytest tests/unit/olmoe3_global_lb_test.py
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import functools  # pylint: disable=wrong-import-position
import types  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position

import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from jax.sharding import Mesh, PartitionSpec as P  # pylint: disable=wrong-import-position

from maxtext.layers import moe  # pylint: disable=wrong-import-position
from maxtext.models import olmoe3  # pylint: disable=wrong-import-position

NUM_DEVICES = 4
NUM_EXPERTS, TOP_K, BATCH, SEQ = 32, 4, 8, 64


def _router(mesh, global_lb, sparse=True):
  """The attributes ``load_balance_loss`` reads, without building a layer."""
  cfg = types.SimpleNamespace(moe_global_load_balancing=global_lb, sparse_matmul=sparse, load_balance_loss_weight=0.01)
  return types.SimpleNamespace(config=cfg, mesh=mesh, num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K)


def _with_token_axes(fn, axes=("data", "expert")):
  """Run fn (a shard_map'd function) with ``moe.LB_REDUCE_AXES`` set, as the sparse routed path sets it."""

  def wrapped(*args):
    token = moe.LB_REDUCE_AXES.set(axes)
    try:
      return fn(*args)
    finally:
      moe.LB_REDUCE_AXES.reset(token)

  return wrapped


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
    self.mesh = Mesh(np.array(jax.devices()[:NUM_DEVICES]).reshape(2, 2), ("data", "expert"))
    # Skewed routing that differs by row, so every device's counts differ from the batch's.
    base = jax.random.normal(jax.random.PRNGKey(0), (BATCH, SEQ, NUM_EXPERTS))
    skew = jnp.linspace(0.0, 3.0, NUM_EXPERTS)[None, None, :] * jnp.cos(jnp.arange(BATCH))[:, None, None]
    self.logits = base + skew

  def _whole_batch(self):
    return jax.value_and_grad(lambda x: _lb(_router(self.mesh, False, sparse=False), x))(self.logits)

  def _sharded(self, global_lb):
    router = _router(self.mesh, global_lb)
    return _with_token_axes(
        jax.shard_map(
            lambda x: _lb(router, x),
            mesh=self.mesh,
            in_specs=P(("data", "expert")),
            out_specs=P(),
            check_vma=False,
        )
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
    sharded = _with_token_axes(
        jax.shard_map(
            lambda t, p: olmoe3.OLMoE3LatentRoutedMoE.load_balance_loss(router, t, p),
            mesh=self.mesh,
            in_specs=(P(("data", "expert")), P(("data", "expert"))),
            out_specs=P(),
            check_vma=False,
        )
    )
    grad = jax.jit(jax.grad(sharded, argnums=1))(top_k, probs)
    counts = jax.nn.one_hot(top_k, NUM_EXPERTS).sum(axis=(0, 1, 2))
    tokens = BATCH * SEQ
    expected = 0.01 * NUM_EXPERTS / (TOP_K * tokens) * counts / tokens
    np.testing.assert_allclose(np.asarray(grad), np.broadcast_to(expected, probs.shape), rtol=1e-5)


  def test_router_metrics_match_whole_batch(self):
    """``olmoe3_router_metrics``: [lb, load imbalance, global load imbalance] from inside the sharded MoE.

    The counts are gathered over the batch axes, so every device returns the whole batch's ratios: the local one
    over consecutive groups of ``router_metrics_group_seqs`` sequences (OLMo-core's per-GPU batch), and the global
    one over all of it. The lb entry is the loss itself, unchanged by the flag.
    """
    group = 2
    router = _router(self.mesh, True)
    router.config.olmoe3_router_metrics = True
    router.config.router_metrics_group_seqs = group
    router._logical_to_mesh_axes = lambda names: P(("data", "expert"), None, None)  # pylint: disable=protected-access
    router._router_load_imbalance = functools.partial(  # pylint: disable=protected-access
        olmoe3.OLMoE3LatentRoutedMoE._router_load_imbalance, router  # pylint: disable=protected-access
    )
    sharded = _with_token_axes(
        jax.shard_map(
            lambda x: _lb(router, x),
            mesh=self.mesh,
            in_specs=P(("data", "expert")),
            out_specs=P(),
            check_vma=False,
        )
    )
    lb, local, global_ = (float(v) for v in jax.jit(sharded)(self.logits))
    ref_value, _ = self._whole_batch()
    np.testing.assert_allclose(lb, float(ref_value), rtol=1e-6)

    _, top_k = jax.lax.top_k(self.logits, TOP_K)
    seq_counts = np.asarray(jax.nn.one_hot(top_k, NUM_EXPERTS).sum(axis=(1, 2)))  # [batch, experts]
    groups = seq_counts.reshape(BATCH // group, group, NUM_EXPERTS).sum(axis=1)
    expected_local = float(np.max(groups.max(axis=1) / groups.mean(axis=1)))
    total = seq_counts.sum(axis=0)
    expected_global = float(total.max() / total.mean())
    np.testing.assert_allclose(local, expected_local, rtol=1e-6)
    np.testing.assert_allclose(global_, expected_global, rtol=1e-6)
    self.assertGreater(local, global_)  # fewer tokens per group: a rank is less balanced than the batch


if __name__ == "__main__":
  unittest.main()
