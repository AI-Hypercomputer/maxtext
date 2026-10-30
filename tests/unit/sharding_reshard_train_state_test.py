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
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.utils import sharding
import numpy as np

# GitHub CI runs the unit tests on 4 CPU devices. Every mesh below uses 4 devices: the first 4 on CPU, or the 2x2 chips at
# the TPU slice origin (one host), which Pathways can compile as a subslice.
_NUM_DEVICES = 4


def _skip_unless_devices(test, num_devices=_NUM_DEVICES):
  if jax.device_count() < num_devices:
    test.skipTest(f"Needs {num_devices} devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count={num_devices}).")


def _mesh():
  """2x2 mesh; on TPU, the 2x2 chips at the slice origin (one host) so Pathways can compile a subslice."""
  origin_chips = [
      device
      for device in jax.devices()
      if hasattr(device, "coords") and device.coords[0] < 2 and device.coords[1] < 2 and not any(device.coords[2:])
  ]
  devices = origin_chips[:_NUM_DEVICES] if len(origin_chips) >= _NUM_DEVICES else jax.devices()[:_NUM_DEVICES]
  return Mesh(np.array(devices).reshape(2, 2), ("data", "expert"))


def _put(x, mesh, spec):
  return jax.device_put(x, NamedSharding(mesh, spec))


def _train_state_and_shardings(mesh):
  """Builds a pre-Zero-1 train-state PyTree and its Zero-1 target shardings.

  Params start replicated or model-sharded, optimizer moments start replicated,
  and the step is a replicated scalar. The targets shard params and moments over
  the data axes, mirroring `shard_optimizer_over_data`.

  Args:
    mesh: The 2D JAX `Mesh` over `("data", "expert")`.

  Returns:
    A `(state, targets)` tuple of matching PyTrees.
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
    for before_leaf, after_leaf, target_sharding in zip(
        jax.tree.leaves(before), jax.tree.leaves(after), jax.tree.leaves(targets), strict=True
    ):
      self.assertEqual(after_leaf.dtype, before_leaf.dtype)
      self.assertEqual(after_leaf.shape, before_leaf.shape)
      np.testing.assert_array_equal(np.asarray(after_leaf), np.asarray(before_leaf))
      self.assertIsInstance(after_leaf.sharding, NamedSharding)
      self.assertEqual(after_leaf.sharding, target_sharding)

  def test_single_controller_reshards_on_device_without_device_put(self):
    x = _put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), self.mesh, P())  # replicated
    target = NamedSharding(self.mesh, P(("data", "expert")))
    with mock.patch.object(
        jax, "device_put", autospec=True, side_effect=AssertionError("device_put must not be called")
    ) as dp:
      out = sharding.reshard_train_state(x, target, enable_single_controller=True)
    dp.assert_not_called()
    self.assert_resharded(x, out, target)
    # Each device holds one distinct row block (sharded, not replicated).
    self.assertLen(out.addressable_shards, _NUM_DEVICES)
    self.assertEqual(out.addressable_shards[0].data.shape, (1, 4))

  def test_single_controller_reshards_between_partitionings(self):
    x = _put(jnp.arange(32, dtype=jnp.int32).reshape(8, 4), self.mesh, P("data"))
    target = NamedSharding(self.mesh, P(("data", "expert")))
    with mock.patch.object(jax, "device_put", autospec=True, side_effect=AssertionError("device_put must not be called")):
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
        mock.patch.object(jax, "device_put", autospec=True, wraps=jax.device_put) as dp,
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
