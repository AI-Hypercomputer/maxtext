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
"""End-to-end forward and backward tests for MoonEP dynamic redundant expert parallelism."""

import unittest
import sys
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

# Prevent absl flags crash when unittest passes CLI arguments like -s or -p
if not flags.FLAGS.is_parsed():
  try:
    flags.FLAGS(sys.argv[:1])
  except Exception:
    pass
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class MoeDynamicRedundantE2ETest(unittest.TestCase):
  """End-to-end test for dynamic redundant experts."""

  def test_forward_and_backward_dynamic_redundancy(self):
    """Executes a full forward and backward pass with dynamic redundancy enabled on TPU."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_dynamic_redundancy_e2e_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        ici_expert_parallelism=len(jax.devices()),
        ici_fsdp_parallelism=1,
        per_device_batch_size=1,
        max_target_length=128,
        enable_moe_dynamic_redundant_experts=True,
        moe_redundant_slots_per_rank=2,
        moe_rebalance_threshold_ratio=1.15,
        float32_gate_logits=True,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    rng = jax.random.PRNGKey(42)
    rng_init, rng_data = jax.random.split(rng)
    device_count = jax.device_count()
    batch_size = int(cfg.per_device_batch_size) * device_count

    hidden_states = jax.random.uniform(
        rng_data,
        (batch_size, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    model = moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    # Initialize model variables
    variables = model.init(
        {"params": rng_init, "dropout": rng_init},
        hidden_states,
    )

    # Forward loss function with value_and_grad
    def loss_fn(params, inputs):
      output, lb_loss, bias_updates = model.apply({"params": params}, inputs)
      return jnp.sum(output.astype(jnp.float32))

    jitted_step = jax.jit(jax.value_and_grad(loss_fn))
    loss_val, grads = jitted_step(variables["params"], hidden_states)

    # Convert to numpy for assertion checks
    loss_np = float(loss_val)
    self.assertTrue(np.isfinite(loss_np), f"Loss is not finite: {loss_np}")

    for param_name, grad_arr in grads.items():
      if isinstance(grad_arr, jax.Array):
        grad_np = np.array(grad_arr)
        self.assertTrue(np.isfinite(grad_np).all(), f"Gradient for {param_name} contains NaNs or Infs!")

    print(f"End-to-end Dynamic Redundant Expert Step Successful! Loss: {loss_np:.4f}")


if __name__ == "__main__":
  unittest.main()
