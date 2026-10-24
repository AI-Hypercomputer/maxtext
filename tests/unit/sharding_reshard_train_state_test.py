# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for sharding.reshard_train_state (Zero-1 reshard before compile)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from maxtext.utils import sharding

# GitHub CI runs the unit tests on 4 CPU devices; every mesh below uses exactly the first 4 devices.
_NUM_DEVICES = 4


def _skip_unless_devices(test, num_devices=_NUM_DEVICES):
  if jax.device_count() < num_devices:
    test.skipTest(f"Needs {num_devices} devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count={num_devices}).")


def _mesh():
  return Mesh(np.array(jax.devices()[:_NUM_DEVICES]).reshape(2, 2), ("data", "expert"))


def _put(x, mesh, spec):
  return jax.device_put(x, NamedSharding(mesh, spec))


def _train_state_and_shardings(mesh):
  """A train-state-like pytree in a pre-Zero-1 layout plus its Zero-1 target shardings.

  Params start replicated / model-sharded, optimizer moments start replicated, and the step
  is a replicated scalar. The targets shard params and moments over the data axes, mirroring
  shard_optimizer_over_data.
  """
  kernel = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
  bias = jnp.arange(4, dtype=jnp.bfloat16)
  state = {
      "params": {
          "kernel": _put(kernel, mesh, P(None, "expert")),
          "bias": _put(bias, mesh, P()),
      },
      "opt_state": {
          "mu": {"kernel": _put(kernel * 0.5, mesh, P()), "bias": _put(bias * 2, mesh, P())},
          "nu": {"kernel": _put(kernel * kernel, mesh, P()), "bias": _put(bias + 1, mesh, P())},
      },
      "step": _put(jnp.array(7, dtype=jnp.int32), mesh, P()),
  }
  targets = {
      "params": {
          "kernel": NamedSharding(mesh, P("data", "expert")),
          "bias": NamedSharding(mesh, P(("data", "expert"))),
      },
      "opt_state": {
          "mu": {
              "kernel": NamedSharding(mesh, P(("data", "expert"), None)),
              "bias": NamedSharding(mesh, P("data")),
          },
          "nu": {
              "kernel": NamedSharding(mesh, P(None, ("data", "expert"))),
              "bias": NamedSharding(mesh, P("expert")),
          },
      },
      "step": NamedSharding(mesh, P()),
  }
  return state, targets


class ReshardTrainStateTest(parameterized.TestCase):
  """Tests for sharding.reshard_train_state."""

  def setUp(self):
    super().setUp()
    _skip_unless_devices(self)
    self.mesh = _mesh()

  def assert_resharded(self, before, after, targets):
    """Values are bit-identical and every leaf carries its target NamedSharding."""
    self.assertEqual(jax.tree.structure(before), jax.tree.structure(after))
    for b, a, t in zip(jax.tree.leaves(before), jax.tree.leaves(after), jax.tree.leaves(targets)):
      self.assertEqual(a.dtype, b.dtype)
      self.assertEqual(a.shape, b.shape)
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
      self.assertIsInstance(a.sharding, NamedSharding)
      self.assertEqual(a.sharding, t)

  def test_single_controller_reshards_on_device_without_device_put(self):
    x = _put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), self.mesh, P())  # replicated
    target = NamedSharding(self.mesh, P(("data", "expert")))
    with mock.patch.object(jax, "device_put", side_effect=AssertionError("device_put must not be called")) as dp:
      out = sharding.reshard_train_state(x, target, enable_single_controller=True)
    dp.assert_not_called()
    self.assert_resharded(x, out, target)
    # Each device holds one distinct row block (sharded, not replicated).
    self.assertLen(out.addressable_shards, _NUM_DEVICES)
    self.assertEqual(out.addressable_shards[0].data.shape, (1, 4))

  def test_single_controller_reshards_between_partitionings(self):
    x = _put(jnp.arange(32, dtype=jnp.int32).reshape(8, 4), self.mesh, P("data"))
    target = NamedSharding(self.mesh, P(("data", "expert")))
    with mock.patch.object(jax, "device_put", side_effect=AssertionError("device_put must not be called")):
      out = sharding.reshard_train_state(x, target, enable_single_controller=True)
    self.assert_resharded(x, out, target)

  def test_single_controller_uses_jit_with_out_shardings(self):
    state, targets = _train_state_and_shardings(self.mesh)
    with mock.patch.object(jax, "jit", wraps=jax.jit) as jit_spy:
      sharding.reshard_train_state(state, targets, enable_single_controller=True)
    jit_spy.assert_called_once()
    self.assertIs(jit_spy.call_args.kwargs["out_shardings"], targets)

  def test_multi_controller_calls_device_put_once(self):
    state, targets = _train_state_and_shardings(self.mesh)
    with (
        mock.patch.object(jax, "device_put", wraps=jax.device_put) as dp,
        mock.patch.object(jax, "jit", side_effect=AssertionError("jit must not be called")),
    ):
      out = sharding.reshard_train_state(state, targets, enable_single_controller=False)
    dp.assert_called_once_with(state, targets)
    self.assert_resharded(state, out, targets)

  @parameterized.named_parameters(("single_controller", True), ("multi_controller", False))
  def test_nested_train_state(self, enable_single_controller):
    state, targets = _train_state_and_shardings(self.mesh)
    out = sharding.reshard_train_state(state, targets, enable_single_controller)
    self.assert_resharded(state, out, targets)
    self.assertEqual(int(out["step"]), 7)
    self.assertEqual(out["params"]["bias"].dtype, jnp.bfloat16)

  @parameterized.named_parameters(("single_controller", True), ("multi_controller", False))
  def test_nnx_state(self, enable_single_controller):
    model = nnx.Linear(8, 4, rngs=nnx.Rngs(0))
    state = nnx.State({"model": nnx.state(model, nnx.Param), "step": jnp.array(3, dtype=jnp.int32)})
    state = jax.tree.map(lambda x: _put(x, self.mesh, P()), state)  # replicated, pre-Zero-1
    specs = {(8, 4): P("data", "expert"), (4,): P(("data", "expert")), (): P()}
    targets = jax.tree.map(lambda x: NamedSharding(self.mesh, specs[x.shape]), state)
    out = sharding.reshard_train_state(state, targets, enable_single_controller)
    self.assertIsInstance(out, nnx.State)
    self.assert_resharded(state, out, targets)
    self.assertEqual(out["model"]["kernel"].get_value().sharding, NamedSharding(self.mesh, P("data", "expert")))

  @parameterized.named_parameters(("single_controller", True), ("multi_controller", False))
  def test_idempotent(self, enable_single_controller):
    state, targets = _train_state_and_shardings(self.mesh)
    once = sharding.reshard_train_state(state, targets, enable_single_controller)
    twice = sharding.reshard_train_state(once, targets, enable_single_controller)
    self.assert_resharded(state, once, targets)
    self.assert_resharded(once, twice, targets)


if __name__ == "__main__":
  absltest.main()
