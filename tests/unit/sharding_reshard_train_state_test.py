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

import os
import socket
import subprocess
import sys
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

  def test_multi_controller_moves_typed_keys_as_raw_data_in_one_device_put(self):
    state, targets = _train_state_and_shardings(self.mesh)
    keys = jax.random.split(jax.random.key(0, impl="unsafe_rbg"), 4)  # stacked (num_layers,), as scan_layers does
    stacked_key = _put(keys, self.mesh, P(None))
    orig_ptr = jax.random.key_data(stacked_key).addressable_shards[0].data.unsafe_buffer_pointer()
    state["stacked_key"] = nnx.RngKey(stacked_key, tag="dropout")
    targets["stacked_key"] = nnx.RngKey(NamedSharding(self.mesh, P()), tag="dropout")
    state["moved_key"] = _put(keys, self.mesh, P(None))
    targets["moved_key"] = NamedSharding(self.mesh, P("data"))
    with (
        mock.patch.object(jax, "device_put", autospec=True, wraps=jax.device_put) as dp,
        mock.patch.object(jax, "jit", side_effect=AssertionError("jit must not be called")),
    ):
      out = sharding.reshard_train_state(state, targets, enable_single_controller=False)
    dp.assert_called_once()
    put_leaves = dp.call_args.args[0]
    self.assertFalse(any(jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) for x in put_leaves))
    self.assertIsInstance(out["stacked_key"], nnx.RngKey)
    self.assertEqual(out["stacked_key"].tag, "dropout")
    stacked_out = out["stacked_key"].get_value()
    stacked_target = targets["stacked_key"].get_value()
    self.assertEqual(jax.random.key_data(stacked_out).addressable_shards[0].data.unsafe_buffer_pointer(), orig_ptr)
    for out_key, target in ((stacked_out, stacked_target), (out["moved_key"], targets["moved_key"])):
      self.assertEqual(out_key.dtype, keys.dtype)
      self.assertEqual(jax.random.key_impl(out_key), jax.random.key_impl(keys))
      self.assertTrue(out_key.sharding.is_equivalent_to(target, out_key.ndim), out_key.sharding)
      np.testing.assert_array_equal(np.asarray(jax.random.key_data(out_key)), np.asarray(jax.random.key_data(keys)))
    plain = {k: v for k, v in targets.items() if not k.endswith("_key")}
    self.assert_resharded({k: state[k] for k in plain}, {k: out[k] for k in plain}, plain)


def _run_two_process_worker():
  """Runs in each of the two JAX processes started by ReshardTrainStateTwoProcessTest."""
  jax.distributed.initialize(
      coordinator_address=os.environ["COORDINATOR_ADDRESS"],
      num_processes=2,
      process_id=int(os.environ["PROCESS_ID"]),
      initialization_timeout=60,
  )
  try:
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("data", "expert"))
    num_layers = 4
    key_bits = np.stack([np.array([0, 17, 0, layer], dtype=np.uint32) for layer in range(num_layers)])
    raw = jax.make_array_from_callback(key_bits.shape, NamedSharding(mesh, P(None)), lambda i: key_bits[i])
    orig_ptr = raw.addressable_shards[0].data.unsafe_buffer_pointer()
    stacked_key = jax.random.wrap_key_data(raw, impl="unsafe_rbg")  # laid out P(None,), as scan_layers does
    kernel = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    replicated_kernel = jax.make_array_from_callback(kernel.shape, NamedSharding(mesh, P()), lambda i: kernel[i])
    state = {"stacked_key": stacked_key, "moved_key": stacked_key, "kernel": replicated_kernel}
    targets = {
        "stacked_key": NamedSharding(mesh, P()),
        "moved_key": NamedSharding(mesh, P("data")),
        "kernel": NamedSharding(mesh, P("data", "expert")),
    }
    out = sharding.reshard_train_state(state, targets, enable_single_controller=False)
    jax.block_until_ready(out)
    assert jax.random.key_data(out["stacked_key"]).addressable_shards[0].data.unsafe_buffer_pointer() == orig_ptr
    for name, target in targets.items():
      assert out[name].sharding.is_equivalent_to(target, out[name].ndim), (name, out[name].sharding)
    for name in ("stacked_key", "moved_key"):
      assert out[name].dtype == stacked_key.dtype, name
      assert jax.random.key_impl(out[name]) == jax.random.key_impl(stacked_key), name
      for shard in jax.random.key_data(out[name]).addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data), key_bits[shard.index])
    for shard in out["kernel"].addressable_shards:
      np.testing.assert_array_equal(np.asarray(shard.data), kernel[shard.index])
    print("RESHARD_OK", flush=True)
  finally:
    jax.distributed.shutdown()


class ReshardTrainStateTwoProcessTest(absltest.TestCase):
  """reshard_train_state across two real JAX CPU processes (2 devices each), as on a 2-host slice."""

  def test_scanned_typed_keys_across_processes(self):
    if "PROCESS_ID" in os.environ and "COORDINATOR_ADDRESS" in os.environ:
      _run_two_process_worker()
      return
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JAX_", "TPU_", "LIBTPU_", "XML_OUTPUT_FILE", "GTEST_")) and not k.lower().endswith("_proxy")
    }
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "PYTHONPATH": os.pathsep.join(sys.path),
        }
    )
    with socket.socket() as listener:
      listener.bind(("127.0.0.1", 0))
      env["COORDINATOR_ADDRESS"] = f"127.0.0.1:{listener.getsockname()[1]}"
    worker_cmd = [sys.executable, __file__] if sys.executable else [sys.argv[0]]
    workers = [
        subprocess.Popen(  # pylint: disable=consider-using-with
            worker_cmd,
            env={**env, "PROCESS_ID": str(process_id)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for process_id in range(2)
    ]
    try:
      outputs = [worker.communicate(timeout=300)[0] for worker in workers]
    finally:
      for worker in workers:
        if worker.poll() is None:
          worker.kill()
          worker.wait()
    for process_id, (worker, output) in enumerate(zip(workers, outputs)):
      self.assertEqual(worker.returncode, 0, f"process {process_id} failed:\n{output}")
      self.assertIn("RESHARD_OK", output, f"process {process_id}:\n{output}")


if __name__ == "__main__":
  if "PROCESS_ID" in os.environ and "COORDINATOR_ADDRESS" in os.environ:
    absltest.main(defaultTest="ReshardTrainStateTwoProcessTest")
  else:
    absltest.main()
