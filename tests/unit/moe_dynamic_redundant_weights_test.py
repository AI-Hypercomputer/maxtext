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
"""Unit tests for MoonEP dynamic expert weight prefetch & backward gradient reducer in MaxText."""

import unittest
import functools
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from maxtext.layers.moe import _manage_dynamic_expert_weights_targeted


class MoeDynamicRedundantWeightsTest(unittest.TestCase):
  """Tests for the dynamic weight manager custom VJP."""

  def setUp(self):
    super().setUp()
    self.devices = jax.devices()
    self.num_devices = len(self.devices)
    self.num_ep = self.num_devices
    self.experts_per_rank = 4
    self.num_slots_B = 2
    self.H = 16
    self.H_ffn = 32
    self.mesh = Mesh(np.array(self.devices), ("expert",))

  def test_forward_prefetch_and_backward_grad_reduction(self):
    """Verifies forward prefetch copies correct weights and backward correctly sums slot gradients."""
    num_total_experts = self.num_ep * self.experts_per_rank

    # Initialize global weights: [num_ep, experts_per_rank, H, H_ffn]
    np_w = (np.arange(num_total_experts * self.H * self.H_ffn, dtype=np.float32) + 1.0).reshape(
        self.num_ep, self.experts_per_rank, self.H, self.H_ffn
    )
    w_sharded = jax.device_put(jnp.array(np_w), jax.sharding.NamedSharding(self.mesh, P("expert", None, None, None)))

    # Plan: duplicate Expert 0 (owned by Rank 0) to Rank 1 Slot 0
    # and duplicate Expert (experts_per_rank + 1) (owned by Rank 1) to Rank 0 Slot 0
    slot_assignments_np = np.full((self.num_ep, self.num_slots_B), -1, dtype=np.int32)
    if self.num_ep > 1:
      slot_assignments_np[1, 0] = 0  # Rank 1 gets Expert 0 in slot 0
      slot_assignments_np[0, 0] = self.experts_per_rank + 1  # Rank 0 gets an expert from Rank 1
    slot_assignments = jnp.array(slot_assignments_np)

    num_ep = self.num_ep
    experts_per_rank = self.experts_per_rank
    num_slots_B = self.num_slots_B

    # Forward loss function under shard_map
    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(P("expert", None, None, None), P()),
        out_specs=P(),
        check_vma=False,
    )
    def compute_loss(w_local, slots):
      # w_local shape inside shard_map: [1, experts_per_rank, H, H_ffn] -> squeeze out the leading shard dim
      w_local_squeezed = jnp.squeeze(w_local, axis=0)
      w_active = _manage_dynamic_expert_weights_targeted(
          w_local_squeezed, slots, num_ep, experts_per_rank, num_slots_B, "expert"
      )
      # Compute dummy loss: sum of squares
      local_sum = jnp.sum(w_active * w_active)
      return jax.lax.psum(local_sum, axis_name="expert")

    loss, grads = jax.value_and_grad(compute_loss)(w_sharded, slot_assignments)

    # Convert grads to numpy
    grads_np = np.array(grads)

    # Mathematical Verification:
    # d(sum(w_active^2))/dw_home[e] should equal 2 * w[e] * (1 + number_of_times_duplicated)
    expected_grads = 2.0 * np_w
    if self.num_ep > 1:
      expected_grads[0, 0] = 4.0 * np_w[0, 0]  # Expert 0 was duplicated to Rank 1
      expected_grads[1, 1] = 4.0 * np_w[1, 1]  # Expert (experts_per_rank + 1) was duplicated to Rank 0

    np.testing.assert_allclose(grads_np, expected_grads, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
