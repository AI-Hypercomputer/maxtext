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

"""Tests for forced routing mismatch calculation."""

# pylint: disable=protected-access,missing-function-docstring

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
from absl.testing import absltest


class ForcedRoutingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="forced_routing_test",
        enable_checkpointing=False,
        model_name="qwen3-30b-a3b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=80,
        per_device_batch_size=1,
        num_experts=8,
        num_experts_per_tok=2,
        override_model_config=True,
    )
    self.rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.model = moe.RoutedMoE(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
        rngs=self.rngs,
    )

  def test_calculate_mismatch_perfect_match(self):
    # Shape: [batch, length, num_experts_per_tok] -> [1, 2, 2]
    model_indices = jnp.array([[[0, 1], [2, 3]]])
    forced_indices = jnp.array([[[0, 1], [2, 3]]])
    mismatch = self.model._calculate_mismatch(model_indices, forced_indices)
    self.assertAlmostEqual(mismatch.item(), 0.0, places=6)

  def test_calculate_mismatch_complete_mismatch(self):
    model_indices = jnp.array([[[0, 1], [2, 3]]])
    forced_indices = jnp.array([[[4, 5], [6, 7]]])
    mismatch = self.model._calculate_mismatch(model_indices, forced_indices)
    self.assertAlmostEqual(mismatch.item(), 1.0, places=6)

  def test_calculate_mismatch_partial_match(self):
    model_indices = jnp.array([[[0, 1], [2, 3]]])
    forced_indices = jnp.array([[[0, 2], [2, 4]]])
    # Token 0: model={0,1}, forced={0,2}. Intersection={0} (size 1). Mismatch = 1 - 1/2 = 0.5
    # Token 1: model={2,3}, forced={2,4}. Intersection={2} (size 1). Mismatch = 1 - 1/2 = 0.5
    # Average = 0.5
    mismatch = self.model._calculate_mismatch(model_indices, forced_indices)
    self.assertAlmostEqual(mismatch.item(), 0.5, places=6)

  def test_calculate_mismatch_with_padding(self):
    model_indices = jnp.array([[[0, 1], [2, 3]]])
    # Token 1 is padded (starts with -1)
    forced_indices = jnp.array([[[0, 2], [-1, -1]]])
    # Token 0: model={0,1}, forced={0,2}. Mismatch = 0.5
    # Token 1: ignored
    # Average = 0.5
    mismatch = self.model._calculate_mismatch(model_indices, forced_indices)
    self.assertAlmostEqual(mismatch.item(), 0.5, places=6)

  def test_calculate_mismatch_all_padded(self):
    model_indices = jnp.array([[[0, 1], [2, 3]]])
    forced_indices = jnp.array([[[-1, -1], [-1, -1]]])
    mismatch = self.model._calculate_mismatch(model_indices, forced_indices)
    # Should handle all padded case without division by zero error, returning 0.0 due to 1e-8 eps
    self.assertAlmostEqual(mismatch.item(), 0.0, places=6)

  def test_routed_moe_calculates_and_sows_mismatch(self):
    batch_size = 1
    seq_len = 2
    emb_dim = self.cfg.emb_dim
    inputs = jax.random.normal(jax.random.key(1), (batch_size, seq_len, emb_dim))
    forced_experts = jnp.array([[[0, 1], [2, 3]]])

    _ = self.model(inputs, forced_routed_experts=forced_experts)

    self.assertTrue(hasattr(self.model, "mismatch_rate"))
    mismatch_rate = self.model.mismatch_rate
    self.assertIsInstance(mismatch_rate, moe.RoutingMismatchRate)
    self.assertIsInstance(mismatch_rate[...], jax.Array)
    self.assertTrue(0.0 <= mismatch_rate[...].item() <= 1.0)

    state = nnx.state(self.model, nnx.Intermediate)
    flat_state = list(nnx.to_flat_state(state))
    has_mismatch_rate = False
    for path, leaf in flat_state:
      if path[-1] == "mismatch_rate":
        has_mismatch_rate = True
        self.assertIsInstance(leaf, moe.RoutingMismatchRate)
        break
    self.assertTrue(has_mismatch_rate)


if __name__ == "__main__":
  absltest.main()
