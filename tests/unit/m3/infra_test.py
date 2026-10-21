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

"""Tests for `maxtext.m3.infra.quantization` and `maxtext.m3.infra.offload`.

Both transforms are checked for the property that matters to every consumer
of the parameter tree: they leave it alone. Quantization is additionally
checked to produce a close-but-different forward pass (so it really is on),
and offload to keep the optimizer trajectory unchanged while the state lives
in host memory.
"""

from types import SimpleNamespace

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxtext.m3 import models as m3_models
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import offload, quantization
from maxtext.m3.train.loss import loss_fn
from tests.utils import m3_test_utils


def _batch(config):
  shape = (2, 8)
  tokens = jax.random.randint(jax.random.key(0), (shape[0], shape[1] + 1), 0, config.vocab_size)
  return {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": jnp.broadcast_to(jnp.arange(shape[1]), shape)}


class QuantizationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])
    self.enter_context(jax.set_mesh(self.mesh))

  def test_off_returns_the_same_model(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    self.assertIs(quantization.maybe_quantize(model, config, *quantization.example_batch(config)), model)

  def test_int8_keeps_parameter_tree_and_changes_numerics(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b", quantization="int8")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    batch = _batch(config)
    reference = model(batch["inputs"], batch["positions"]).logits
    before = m3_test_utils.flat_params(model)

    quantized = quantization.maybe_quantize(model, config, *quantization.example_batch(config))
    after = m3_test_utils.flat_params(quantized)
    self.assertEqual(list(before), list(after))
    self.assertEqual({type(v).__name__ for _, v in nnx.to_flat_state(nnx.state(quantized))}, {"Param"})

    logits = quantized(batch["inputs"], batch["positions"]).logits
    self.assertFalse(np.array_equal(np.asarray(logits), np.asarray(reference)), "int8 quantization had no effect")
    np.testing.assert_allclose(logits, reference, rtol=0.2, atol=0.5)

    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(quantized, batch)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertEqual([p for p, _ in nnx.to_flat_state(nnx.state(grads, nnx.Param))], list(before))

  def test_moe_router_stays_unquantized(self):
    """Routing decides which experts run; the rule set keeps the gate in full precision."""
    rules = quantization.quantization_rules(SimpleNamespace(quantization="int8"))
    self.assertIsNone(rules[0].weight_qtype)
    self.assertRegex("decoder/layers/0/moe_block/gate", rules[0].module_path)

  def test_unknown_quantization_raises(self):
    with self.assertRaisesRegex(ValueError, "m3 supports quantization"):
      quantization.quantization_rules(SimpleNamespace(quantization="int2"))


class _Model(nnx.Module):

  def __init__(self, rngs):
    self.lin = nnx.Linear(8, 8, rngs=rngs)

  def __call__(self, x):
    return self.lin(x)


@pytest.mark.tpu_only
class OffloadTest(absltest.TestCase):
  """Pinned host memory is an accelerator concept, so this runs on TPU only."""

  def _train(self, optimizer_cls, steps=3):
    """Trains a tiny model for `steps`; returns the per-step losses and the optimizer."""
    model = _Model(nnx.Rngs(0))
    optimizer = optimizer_cls(model, optax.adamw(1e-2), wrt=nnx.Param)
    x = jax.random.normal(jax.random.key(1), (4, 8))

    @nnx.jit
    def step(model, optimizer):
      loss, grads = nnx.value_and_grad(lambda m: jnp.sum(m(x) ** 2))(model)
      optimizer.update(model, grads)
      return loss

    return [float(step(model, optimizer)) for _ in range(steps)], optimizer

  def test_offloaded_state_lives_on_host_and_matches_device_trajectory(self):
    plain_losses, _ = self._train(nnx.Optimizer)
    offloaded_losses, optimizer = self._train(offload.HostOffloadedOptimizer)
    np.testing.assert_allclose(offloaded_losses, plain_losses, rtol=1e-6)
    kinds = {v[...].sharding.memory_kind for _, v in nnx.to_flat_state(nnx.state(optimizer.opt_state))}
    self.assertEqual(kinds, {"pinned_host"})

  def test_make_optimizer_reads_config(self):
    model = _Model(nnx.Rngs(0))
    on = offload.make_optimizer(model, optax.sgd(0.1), SimpleNamespace(optimizer_memory_host_offload=True))
    off = offload.make_optimizer(model, optax.sgd(0.1), SimpleNamespace(optimizer_memory_host_offload=False))
    self.assertIsInstance(on, offload.HostOffloadedOptimizer)
    self.assertNotIsInstance(off, offload.HostOffloadedOptimizer)


if __name__ == "__main__":
  absltest.main()
