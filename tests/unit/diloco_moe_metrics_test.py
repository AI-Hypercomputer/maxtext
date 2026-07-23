import importlib.util
import sys
from unittest.mock import MagicMock

# Mock maxtext.configs
sys.modules["maxtext"] = MagicMock()
sys.modules["maxtext.configs"] = MagicMock()
sys.modules["maxtext.configs.pyconfig"] = MagicMock()

spec = importlib.util.spec_from_file_location(
    "diloco",
    "src/maxtext/trainers/diloco/diloco.py",
)
diloco = importlib.util.module_from_spec(spec)
sys.modules["diloco"] = diloco
spec.loader.exec_module(diloco)

import unittest
import jax
import jax.numpy as jnp
import flax.struct
import optax


class DummyInnerState(flax.struct.PyTreeNode):
  step: jax.Array
  params: dict


class DiLoCoMoEMetricsUnitTest(unittest.TestCase):

  def test_inter_replica_router_distance_identical(self):
    # Identical router parameters across 2 replicas -> distance = 0.0
    w = jnp.ones((2, 4, 8))
    params = {"moe": {"gate": {"kernel": w}}}
    dist = diloco.compute_inter_replica_router_distance(params, num_replicas=2)
    self.assertAlmostEqual(float(dist), 0.0, places=5)

  def test_inter_replica_router_distance_orthogonal(self):
    # Orthogonal router parameters across 2 replicas -> distance = 1.0
    w1 = jnp.array([1.0, 0.0])
    w2 = jnp.array([0.0, 1.0])
    w = jnp.stack([w1, w2], axis=0)
    params = {"moe": {"gate": {"kernel": w}}}
    dist = diloco.compute_inter_replica_router_distance(params, num_replicas=2)
    self.assertAlmostEqual(float(dist), 1.0, places=5)

  def test_topk_token_routing_overlap_perfect(self):
    # Perfect top-k expert choices across replicas -> Jaccard = 1.0
    topk_same = jnp.array([[[[0, 1]], [[0, 1]]], [[[0, 1]], [[0, 1]]]])
    overlap = diloco.compute_topk_token_routing_overlap(topk_same, num_experts=4, num_replicas=2)
    self.assertAlmostEqual(float(overlap), 1.0, places=5)

  def test_topk_token_routing_overlap_disjoint(self):
    # Disjoint top-k expert choices across replicas -> Jaccard = 0.0
    topk_diff = jnp.array([[[[0, 1]]], [[[2, 3]]]])
    overlap = diloco.compute_topk_token_routing_overlap(topk_diff, num_experts=4, num_replicas=2)
    self.assertAlmostEqual(float(overlap), 0.0, places=5)

  def test_jensen_shannon_routing_divergence_identical(self):
    # Identical routing logits -> JS divergence = 0.0
    logits_same = jnp.ones((2, 1, 1, 4))
    rdi = diloco.compute_jensen_shannon_routing_divergence(logits_same, num_replicas=2)
    self.assertAlmostEqual(float(rdi), 0.0, places=5)

  def test_expert_utilization_entropy_uniform(self):
    # Uniform expert allocation across 4 experts -> Entropy = ln(4) = 1.386294
    topk_uniform = jnp.array([[[[0, 1]]], [[[2, 3]]]])  # counts: [1, 1, 1, 1]
    eue = diloco.compute_expert_utilization_entropy(topk_uniform, num_experts=4)
    expected = float(jnp.log(4))
    self.assertAlmostEqual(float(eue), expected, places=4)

  def test_expert_utilization_entropy_single(self):
    # Concentrated allocation to single expert -> Entropy = 0.0
    topk_single = jnp.zeros((2, 1, 1, 2), dtype=jnp.int32)
    eue = diloco.compute_expert_utilization_entropy(topk_single, num_experts=4)
    self.assertAlmostEqual(float(eue), 0.0, places=4)


if __name__ == "__main__":
  unittest.main()
