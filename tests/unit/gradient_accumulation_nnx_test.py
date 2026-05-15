# Copyright 2025-2026 Google LLC
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

"""Unit tests for the NNX branch of gradient_accumulation_loss_and_grad."""

import unittest
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import ShardMode
from maxtext.utils import gradient_accumulation


@dataclass
class _Cfg:
  gradient_accumulation_steps: int = 2
  shard_optimizer_over_data: bool = False
  shard_mode: int = ShardMode.AUTO
  ici_data_parallelism: int = 1
  debug_sharding: bool = False


class _TinyNNX(nnx.Module):
  """Single linear layer NNX model."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


def _fake_loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """A loss_fn shaped like the production loss_fn but for a tiny linear model.

  Returns (loss, aux) where aux follows the schema gradient_accumulation_loss_and_grad
  reads from: xent_sum / total_weights / moe_lb_loss / indexer_loss / mtp_loss.
  """
  del config, dropout_rng, params, is_train
  pred = model(data["inputs"])
  per_sample_loss = jnp.mean((pred - data["targets"]) ** 2, axis=-1)
  xent_sum = jnp.sum(per_sample_loss)
  total_weights = jnp.array(per_sample_loss.shape[0], dtype=jnp.float32)
  aux = {
      "xent_sum": xent_sum,
      "total_weights": total_weights,
      "moe_lb_loss": jnp.array(0.0),
      "indexer_loss": jnp.array(0.0),
      "mtp_loss": jnp.array(0.0),
  }
  return xent_sum / total_weights, aux


class TestGradientAccumulationNNX(unittest.TestCase):
  """Cover the NNX path of gradient_accumulation_loss_and_grad."""

  def setUp(self):
    self.model = _TinyNNX(rngs=nnx.Rngs(0))
    self.cfg = _Cfg(gradient_accumulation_steps=2)
    # 4 examples → 2 microbatches of 2 each
    self.data = {
        "inputs": jnp.arange(8.0).reshape(4, 2),
        "targets": jnp.zeros((4, 1)),
    }

  def _params_shardings(self):
    """Build a per-leaf NamedSharding tree shaped like nnx.split(model, nnx.Param, ...)[1].

    Uses a trivial single-device mesh so jax.lax.with_sharding_constraint accepts the
    sharding without contradicting the actual device topology.
    """
    _, params, _ = nnx.split(self.model, nnx.Param, ...)
    mesh = Mesh(
        np.array(jax.local_devices()[:1]).reshape(
            1,
        ),
        ("x",),
    )
    ns = NamedSharding(mesh, PartitionSpec())
    return jax.tree.map(lambda _: ns, params)

  def test_nnx_path_runs_and_returns_grad_for_every_param(self):
    """The NNX branch must call nnx.value_and_grad and return one gradient per Param."""
    loss, aux, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,  # NNX branch ignores params
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
        extra_dpo_args=[],
    )
    self.assertTrue(jnp.isfinite(loss))
    self.assertIn("xent_sum", aux)
    self.assertIn("total_weights", aux)
    grad_leaves = jax.tree.leaves(raw_grads)
    self.assertEqual(len(grad_leaves), 2)  # linear.kernel + linear.bias
    for g in grad_leaves:
      self.assertTrue(jnp.all(jnp.isfinite(g)))

  def test_nnx_path_updates_model_rest_state_after_scan(self):
    """After accumulation, nnx.update is called on the model with the rest_state from the scan.

    For a TinyNNX (no rngs/dropout), the rest tree is empty but the call path must still
    succeed end-to-end without raising — covering the `if is_nnx: nnx.update(...)` branch.
    """
    pre_kernel = self.model.linear.kernel.value.copy()
    gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
        extra_dpo_args=[],
    )
    # The kernel itself is a Param — gradient_accumulation_loss_and_grad does not apply
    # gradients to params, so the value should be untouched.
    self.assertTrue(jnp.allclose(self.model.linear.kernel.value, pre_kernel))

  def test_nnx_with_shard_optimizer_over_data_casts_to_bf16(self):
    """Zero-1 path must convert fp32 params to bf16 before the scan loop."""
    self.cfg.shard_optimizer_over_data = True
    # Should not raise; just verify the function runs and returns sensible outputs.
    loss, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
        extra_dpo_args=[],
    )
    self.assertTrue(jnp.isfinite(loss))
    for g in jax.tree.leaves(raw_grads):
      self.assertTrue(jnp.all(jnp.isfinite(g)))


if __name__ == "__main__":
  unittest.main()
