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

"""Unit tests for MoE routing distribution and load imbalance metrics."""

import math
import unittest
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.metric_logger import record_moe_routing_metrics
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class MoeRoutingMetricsTest(unittest.TestCase):

  def test_record_moe_routing_metrics_balanced(self):
    """Tests metrics computation on perfectly balanced routing."""
    # 2 layers, 4 experts, perfectly balanced with 100 tokens each
    expert_counts_l0 = jnp.array([100, 100, 100, 100], dtype=jnp.int32)
    expert_counts_l1 = jnp.array([100, 100, 100, 100], dtype=jnp.int32)

    intermediate_outputs = {
        "moe_expert_counts": (expert_counts_l0, expert_counts_l1)
    }
    metrics = {"scalar": {}, "scalars": {}}
    config = type("Config", (), {"num_experts": 4})()

    record_moe_routing_metrics(metrics, intermediate_outputs, config)

    # For uniform distribution: CV should be ~0, peak ratio should be ~1, dead experts = 0, entropy = 1.0
    self.assertAlmostEqual(float(metrics["scalar"]["moe_cv/layer_000"]), 0.0, places=4)
    self.assertAlmostEqual(float(metrics["scalar"]["moe_peak_ratio/layer_000"]), 1.0, places=4)
    self.assertEqual(int(metrics["scalar"]["moe_dead_experts/layer_000"]), 0)
    self.assertAlmostEqual(float(metrics["scalar"]["moe_entropy/layer_000"]), 1.0, places=4)

    self.assertAlmostEqual(float(metrics["scalar"]["moe/cv_max"]), 0.0, places=4)
    self.assertAlmostEqual(float(metrics["scalar"]["moe/cv_mean"]), 0.0, places=4)
    self.assertAlmostEqual(float(metrics["scalar"]["moe/peak_ratio_max"]), 1.0, places=4)
    self.assertEqual(int(metrics["scalar"]["moe/total_dead_experts"]), 0)

    # Verify histograms
    self.assertIn("moe/expert_token_counts/layer_000", metrics["moe_histograms"])
    self.assertIn("moe/expert_token_counts/layer_001", metrics["moe_histograms"])

  def test_record_moe_routing_metrics_imbalanced(self):
    """Tests metrics computation on collapsed/imbalanced routing."""
    # Layer 0: all 400 tokens routed to Expert 0, Experts 1-3 have 0 tokens
    expert_counts_l0 = jnp.array([400, 0, 0, 0], dtype=jnp.int32)

    intermediate_outputs = {
        "moe_expert_counts": (expert_counts_l0,)
    }
    metrics = {"scalar": {}, "scalars": {}}
    config = type("Config", (), {"num_experts": 4})()

    record_moe_routing_metrics(metrics, intermediate_outputs, config)

    # Peak ratio should be 400 / 100 = 4.0
    self.assertAlmostEqual(float(metrics["scalar"]["moe_peak_ratio/layer_000"]), 4.0, places=4)
    # 3 dead experts
    self.assertEqual(int(metrics["scalar"]["moe_dead_experts/layer_000"]), 3)
    # Entropy should be ~0.0
    self.assertAlmostEqual(float(metrics["scalar"]["moe_entropy/layer_000"]), 0.0, places=4)
    # CV should be sqrt(3) ~ 1.732
    self.assertAlmostEqual(float(metrics["scalar"]["moe_cv/layer_000"]), math.sqrt(3.0), places=3)
    self.assertEqual(int(metrics["scalar"]["moe/total_dead_experts"]), 3)

  def test_routed_moe_sow_intermediates(self):
    """Tests that RoutedMoE properly sows moe_expert_counts when enabled."""
    # Case 1: record_moe_routing_metrics = False
    cfg_disabled = pyconfig.initialize(
        [None, get_test_config_path()],
        num_experts=4,
        num_experts_per_tok=2,
        base_moe_mlp_dim=7168,
        record_moe_routing_metrics=False,
    )
    devices_disabled = maxtext_utils.create_device_mesh(cfg_disabled)
    mesh_disabled = jax.sharding.Mesh(devices_disabled, cfg_disabled.mesh_axes)
    rngs = nnx.Rngs(params=0)
    routed_moe_disabled = moe.RoutedMoE(
        config=cfg_disabled,
        num_experts=cfg_disabled.num_experts,
        num_experts_per_tok=cfg_disabled.num_experts_per_tok,
        mesh=mesh_disabled,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg_disabled.dtype,
        rngs=rngs,
    )

    batch_size, seq_len, emb_dim = 2, 8, cfg_disabled.emb_dim
    x = jnp.ones((batch_size, seq_len, emb_dim), dtype=jnp.float32)

    _, state_disabled = nnx.split(routed_moe_disabled)
    out_disabled, _, _ = routed_moe_disabled(x)
    self.assertEqual(out_disabled.shape, (batch_size, seq_len, emb_dim))
    # Verify no moe_expert_counts sown
    intermediates_disabled = nnx.state(routed_moe_disabled, nnx.Intermediate)
    intermediates_dict_disabled = nnx.to_pure_dict(intermediates_disabled)
    self.assertNotIn("moe_expert_counts", str(intermediates_dict_disabled))

    # Case 2: record_moe_routing_metrics = True
    cfg_enabled = pyconfig.initialize(
        [None, get_test_config_path()],
        num_experts=4,
        num_experts_per_tok=2,
        base_moe_mlp_dim=7168,
        record_moe_routing_metrics=True,
    )
    devices_enabled = maxtext_utils.create_device_mesh(cfg_enabled)
    mesh_enabled = jax.sharding.Mesh(devices_enabled, cfg_enabled.mesh_axes)
    rngs_enabled = nnx.Rngs(params=0)
    routed_moe_enabled = moe.RoutedMoE(
        config=cfg_enabled,
        num_experts=cfg_enabled.num_experts,
        num_experts_per_tok=cfg_enabled.num_experts_per_tok,
        mesh=mesh_enabled,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg_enabled.dtype,
        rngs=rngs_enabled,
    )

    out_enabled, _, _ = routed_moe_enabled(x)
    self.assertEqual(out_enabled.shape, (batch_size, seq_len, emb_dim))

    intermediates_enabled = nnx.state(routed_moe_enabled, nnx.Intermediate)
    intermediates_dict_enabled = nnx.to_pure_dict(intermediates_enabled)
    self.assertIn("moe_expert_counts", intermediates_dict_enabled)

    counts = intermediates_dict_enabled["moe_expert_counts"][0]
    self.assertEqual(counts.shape, (4,))
    # Total tokens routed must equal batch_size * seq_len * num_experts_per_tok
    expected_total_tokens = batch_size * seq_len * 2
    self.assertEqual(int(jnp.sum(counts)), expected_total_tokens)


if __name__ == "__main__":
  absltest.main()
