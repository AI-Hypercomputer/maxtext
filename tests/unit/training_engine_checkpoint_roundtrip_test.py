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

"""Real (unmocked) Orbax round trips through `training_engine.checkpointing.CheckpointManager`.

Uses the on-disk format that `ENABLE_PATHWAYS_PERSISTENCE=1` writes (`use_ocdbt=False`,
`use_zarr3=False`) with async saves, and restores into a separate manager and a differently
seeded model, as a resumed RL trainer would.
"""

import glob
import inspect
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.training_engine import checkpointing
import numpy as np
import optax


class _Model(nnx.Module):
  """Weights plus a dropout rng stream, so the saved tree carries typed PRNG keys like the trainer's."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(4, 8, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
    self.linear2 = nnx.Linear(8, 2, rngs=rngs)

  def __call__(self, x):
    return self.linear2(self.dropout(jax.nn.relu(self.linear1(x)), deterministic=True))


def _config():
  """The `CheckpointManager` fields it reads, set to the Pathways-persistence on-disk format."""
  return SimpleNamespace(
      checkpoint_period=1,
      max_num_checkpoints_to_keep=5,
      async_checkpointing=True,
      async_checkpointing_timeout_secs=1200,
      abandon_failed_checkpoint_saves=False,
      skip_checkpoint_save_if_in_progress=False,
      checkpoint_storage_use_ocdbt=False,
      checkpoint_storage_use_zarr3=False,
      checkpoint_storage_device_host_concurrent_gb=None,
  )


def _build(seed):
  model = _Model(nnx.Rngs(seed))
  return model, nnx.Optimizer(model, optax.adamw(1e-2), wrt=nnx.Param)


_X = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7.0
_Y = jnp.array([[1.0, -1.0], [0.5, 2.0], [-3.0, 0.25]])


def _train_step(model, optimizer):
  def loss_fn(m):
    return jnp.mean((m(_X) - _Y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)
  return loss


def _leaves(state):
  """Flat {path: np.ndarray}; typed PRNG keys compared through their raw key data."""
  return {
      jax.tree_util.keystr(p): np.asarray(
          jax.random.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x
      )
      for p, x in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(state))[0]
  }


class TrainingEngineCheckpointRoundTripTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)

  def _save_trained(self, steps=2):
    """Trains `steps` non-zero updates and saves them at `steps`; returns the live model/optimizer."""
    model, optimizer = _build(seed=0)
    for _ in range(steps):
      _train_step(model, optimizer)
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    saved = manager.save_checkpoint(
        step=steps,
        checkpoint_state=checkpointing.CheckpointState(model=model, optimizer=optimizer),
        custom_metadata={"additional_metadata": {"step": steps, "global_step": steps}},
    )
    manager.wait_until_finished()
    manager.close()
    self.assertTrue(saved)
    return model, optimizer

  def _restore_fresh(self, step=None):
    model, optimizer = _build(seed=123)
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.addCleanup(manager.close)
    restored = manager.restore_checkpoint(checkpointing.CheckpointState(model=model, optimizer=optimizer), step=step)
    return model, optimizer, restored

  def test_full_state_round_trips_bit_exactly(self):
    model, optimizer = self._save_trained(steps=2)
    want_params = _leaves(nnx.state(model))
    want_opt = _leaves(nnx.state(optimizer, nnx.optimizer.OptState))

    fresh_model, fresh_opt, (step, _, metadata) = self._restore_fresh()

    self.assertEqual(step, 2)
    self.assertEqual(metadata["micro_step_count"], 0)
    self.assertEqual(metadata["additional_metadata"], {"step": 2, "global_step": 2})
    got_params = _leaves(nnx.state(fresh_model))
    got_opt = _leaves(nnx.state(fresh_opt, nnx.optimizer.OptState))
    self.assertEqual(got_params.keys(), want_params.keys())
    self.assertEqual(got_opt.keys(), want_opt.keys())
    # Load-bearing counts: 2 kernels + 2 biases + dropout rng key/count; adam count + mu/nu x4 + step.
    self.assertEqual(len(got_params), 6)
    self.assertEqual(len(got_opt), 10)
    got_all = {**got_params, **got_opt}
    for name, want in {**want_params, **want_opt}.items():
      np.testing.assert_array_equal(got_all[name], want, err_msg=name)
    # Non-vacuous: moments are non-zero and the step counter reflects both updates.
    self.assertTrue(any(np.any(v != 0) for k, v in want_opt.items() if "mu" in k))
    self.assertTrue(any(int(np.max(v)) == 2 for k, v in want_opt.items() if "count" in k))

  def test_resumed_step_matches_uninterrupted_step(self):
    model, optimizer = self._save_trained(steps=2)
    fresh_model, fresh_opt, _ = self._restore_fresh()

    want_loss = _train_step(model, optimizer)
    got_loss = _train_step(fresh_model, fresh_opt)

    np.testing.assert_array_equal(np.asarray(got_loss), np.asarray(want_loss))
    got = _leaves(nnx.state(fresh_model, nnx.Param))
    for name, want in _leaves(nnx.state(model, nnx.Param)).items():
      np.testing.assert_array_equal(got[name], want, err_msg=name)

  def test_resume_skips_params_only_checkpoints(self):
    model, optimizer = self._save_trained(steps=2)
    want_opt = _leaves(nnx.state(optimizer, nnx.optimizer.OptState))
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    for step in (3, 4):
      _train_step(model, optimizer)
      self.assertTrue(manager.save_checkpoint(step=step, checkpoint_state=checkpointing.CheckpointState(model=model)))
    manager.wait_until_finished()
    manager.close()

    _, fresh_opt, (step, _, _) = self._restore_fresh()

    self.assertEqual(step, 2)
    got_opt = _leaves(nnx.state(fresh_opt, nnx.optimizer.OptState))
    for name, want in want_opt.items():
      np.testing.assert_array_equal(got_opt[name], want, err_msg=name)
    # The resumed run redoes steps 3 and 4, so they must be gone for Orbax to write them again.
    self.assertEqual(sorted(os.listdir(self.ckpt_dir)), ["2"])

  def test_params_only_checkpoint_needs_no_optimizer_to_restore(self):
    model, optimizer = self._save_trained(steps=2)
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    _train_step(model, optimizer)
    manager.save_checkpoint(step=3, checkpoint_state=checkpointing.CheckpointState(model=model))
    manager.wait_until_finished()
    manager.close()
    want_params = _leaves(nnx.state(model))

    # Resuming training from it would start the optimizer over, so that fails loudly.
    with self.assertRaisesRegex(checkpointing.CheckpointRestoreError, "no optimizer state"):
      self._restore_fresh(step=3)

    fresh_model, _ = _build(seed=123)
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.addCleanup(manager.close)
    step, _, _ = manager.restore_checkpoint(checkpointing.CheckpointState(model=fresh_model), step=3)

    self.assertEqual(step, 3)
    got_params = _leaves(nnx.state(fresh_model))
    for name, want in want_params.items():
      np.testing.assert_array_equal(got_params[name], want, err_msg=name)

  def test_unrestorable_checkpoint_raises_instead_of_fresh_start(self):
    self._save_trained(steps=2)
    # Keep `.zarray` so metadata still reads (a missing array dir already fails loudly there) and
    # drop the chunk data, so the failure lands inside `restore()` -- the path that used to be
    # swallowed into "no checkpoint".
    chunks = [
        f
        for f in glob.glob(os.path.join(self.ckpt_dir, "2", "model_params", "*kernel*", "*"))
        if not f.endswith(".zarray")
    ]
    self.assertTrue(chunks)
    for f in chunks:
      os.remove(f)

    with self.assertRaises(checkpointing.CheckpointRestoreError) as cm:
      self._restore_fresh()
    self.assertIsNotNone(cm.exception.__cause__)

  def test_no_checkpoint_keeps_fresh_start(self):
    model, optimizer = _build(seed=5)
    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.addCleanup(manager.close)
    state = checkpointing.CheckpointState(model=model, optimizer=optimizer)

    self.assertEqual(manager.restore_checkpoint(state), (None, state, None))

  def test_restore_into_pinned_host_optimizer_requests_device_memory_kind(self):
    os.environ["ENABLE_ORBAX_FINGERPRINT"] = "1"
    _, optimizer = self._save_trained(steps=2)
    want_opt = _leaves(nnx.state(optimizer, nnx.optimizer.OptState))

    fresh_model, fresh_opt = _build(seed=123)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    host_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()).with_memory_kind("pinned_host")
    offloaded_opt_state = jax.tree.map(
        lambda x: jax.device_put(x, host_sharding) if isinstance(x, jax.Array) else x,
        nnx.state(fresh_opt, nnx.optimizer.OptState),
    )
    nnx.update(fresh_opt, offloaded_opt_state)
    for leaf in jax.tree.leaves(nnx.state(fresh_opt, nnx.optimizer.OptState)):
      if isinstance(leaf, jax.Array):
        self.assertEqual(leaf.sharding.memory_kind, "pinned_host")

    manager = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.addCleanup(manager.close)
    step, _, _ = manager.restore_checkpoint(checkpointing.CheckpointState(model=fresh_model, optimizer=fresh_opt))
    self.assertEqual(step, 2)
    for leaf in jax.tree.leaves(nnx.state(fresh_opt, nnx.optimizer.OptState)):
      if isinstance(leaf, jax.Array):
        self.assertEqual(leaf.sharding.memory_kind, "device")
    got_opt = _leaves(nnx.state(fresh_opt, nnx.optimizer.OptState))
    for name, want in want_opt.items():
      np.testing.assert_array_equal(got_opt[name], want, err_msg=name)

  def test_maybe_register_colocated_python_without_env_raises(self):
    checkpointing._REGISTERED_IMPL = None
    self.addCleanup(setattr, checkpointing, "_REGISTERED_IMPL", None)
    for env_dict in ({}, {"ENABLE_PATHWAYS_PERSISTENCE": "0"}):
      with self.subTest(env=env_dict), mock.patch.dict(os.environ, env_dict, clear=True):
        with self.assertRaisesRegex(
            checkpointing.PathwaysCheckpointingUnavailableError,
            "pathways_checkpointing_impl='colocated_python' requires ENABLE_PATHWAYS_PERSISTENCE=1",
        ):
          checkpointing._maybe_register_pathways_persistence("colocated_python", d2h_concurrent_gb=16)

  def test_maybe_register_colocated_python_passes_deprioritized_callback(self):
    self.addCleanup(os.environ.pop, "ENABLE_PATHWAYS_PERSISTENCE", None)
    os.environ["ENABLE_PATHWAYS_PERSISTENCE"] = "1"
    checkpointing._REGISTERED_IMPL = None
    self.addCleanup(setattr, checkpointing, "_REGISTERED_IMPL", None)
    mock_handler = mock.MagicMock()
    mock_handler.has_dispatcher.return_value = True
    type(mock_handler).__name__ = "ArrayHandler"

    import orbax.checkpoint.pathways as ocp_pathways  # pylint: disable=g-import-not-at-top
    from orbax.checkpoint._src.serialization import type_handler_registry  # pylint: disable=g-import-not-at-top
    from orbax.checkpoint._src.serialization import types as serialization_types  # pylint: disable=g-import-not-at-top

    with (
        mock.patch.object(ocp_pathways, "register_type_handlers") as mock_reg,
        mock.patch.object(type_handler_registry, "get_type_handler", return_value=mock_handler),
        mock.patch.object(
            checkpointing,
            "_configure_colocated_python_handler",
            wraps=checkpointing._configure_colocated_python_handler,
        ) as spy_configure,
    ):
      checkpointing._maybe_register_pathways_persistence("colocated_python", d2h_concurrent_gb=16)
      spy_configure.assert_called_once_with(mock_handler, d2h_concurrent_gb=16)
      mock_reg.assert_called_once()
      call_kwargs = mock_reg.call_args.kwargs
      self.assertIn("callback", call_kwargs)
      cb = call_kwargs["callback"]
      self.assertEqual(
          cb.key_priority("any_param"),
          serialization_types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED,
      )
      # Orbax's ArrayHandler invokes status hooks (e.g. on_write_end) at commit time; a callback
      # missing any of them fails every colocated save. Derive the hook set from the installed
      # Orbax source so this test tracks the real contract instead of a hard-coded list.
      self.assertIsInstance(cb, serialization_types.DefaultSerializationStatusCallback)
      from orbax.checkpoint._src.serialization import jax_array_handlers  # pylint: disable=g-import-not-at-top

      invoked_hooks = set(re.findall(r"callback\.(on_\w+)", inspect.getsource(jax_array_handlers)))
      self.assertTrue(invoked_hooks, "expected ArrayHandler to invoke at least one callback hook")
      for hook in invoked_hooks:
        self.assertIsNone(getattr(cb, hook)(("any_param",)), hook)

  def test_colocated_transport_sharding_normalizer_strips_pinned_host(self):
    from orbax.checkpoint._src.multihost import colocated_transport  # pylint: disable=g-import-not-at-top

    with (
        mock.patch.object(colocated_transport, "_maxtext_Normalized", False, create=True),
        mock.patch.object(colocated_transport, "colocated_cpu_sharding", colocated_transport.colocated_cpu_sharding),
        mock.patch.object(
            colocated_transport,
            "_normalize_single_device_sharding_to_colocated_cpu",
            colocated_transport._normalize_single_device_sharding_to_colocated_cpu,
        ),
        mock.patch.object(colocated_transport.cp, "colocated_cpu_devices", side_effect=lambda devs: devs),
        mock.patch.object(colocated_transport, "colocated_cpu_mesh", side_effect=lambda m: m),
    ):
      checkpointing._normalize_colocated_cpu_shardings()
      self.assertTrue(colocated_transport._maxtext_Normalized)

      dev = jax.devices()[0]
      sds_pinned = jax.sharding.SingleDeviceSharding(dev).with_memory_kind("pinned_host")
      self.assertEqual(sds_pinned.memory_kind, "pinned_host")
      norm_sds = colocated_transport._normalize_single_device_sharding_to_colocated_cpu(sds_pinned)
      self.assertEqual(norm_sds.memory_kind, "device")

      mesh = jax.sharding.Mesh(np.array([dev]), ("data",))
      named_pinned = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()).with_memory_kind("pinned_host")
      norm_named = colocated_transport.colocated_cpu_sharding(named_pinned)
      self.assertEqual(norm_named.memory_kind, "device")

  def test_wrapped_dispatcher_batches_sync_deserialize_arrays_including_prng_key(self):
    dispatcher = mock.MagicMock()
    dispatcher._maxtext_wrapped = False
    dispatches = []

    def fake_dispatch(func, *, input_arrays=None, result_specs=None, func_args=(), func_kwargs=None):
      dispatches.append((func, list(result_specs), func_kwargs))
      return [f"out_{i}" for i in range(len(result_specs))]

    dispatcher.dispatch = fake_dispatch
    dummy_handler = SimpleNamespace(_dispatcher=dispatcher)

    # 1000 bytes budget: 800-byte float32 + 800-byte key<fry> + 200-byte float32 -> 3 batches
    checkpointing._configure_colocated_python_handler(dummy_handler, d2h_concurrent_gb=1e-6)
    self.assertTrue(dispatcher._maxtext_wrapped)

    def _sync_deserialize_arrays():
      pass

    specs = [
        jax.ShapeDtypeStruct((200,), jnp.float32),  # 800 bytes
        jax.ShapeDtypeStruct((100,), jax.random.key(0).dtype),  # 800 bytes (key<fry>)
        jax.ShapeDtypeStruct((50,), jnp.float32),  # 200 bytes
    ]
    kwargs = {"infos": [1, 2, 3], "args": [4, 5, 6], "shardings": [None, None, None]}
    res = dispatcher.dispatch(_sync_deserialize_arrays, result_specs=specs, func_kwargs=kwargs)
    self.assertEqual(res, ["out_0", "out_0", "out_0"])
    self.assertEqual(len(dispatches), 3)
    self.assertEqual(len(dispatches[0][1]), 1)
    self.assertEqual(len(dispatches[1][1]), 1)
    self.assertEqual(len(dispatches[2][1]), 1)

  def test_est_host_bytes_scales_with_v5p_and_v7x_devices_per_host(self):
    def _make_sharding(device_kind: str, num_devices: int):
      devs = [mock.Mock(device_kind=device_kind) for _ in range(num_devices)]
      return mock.Mock(
          spec=jax.sharding.NamedSharding,
          device_set=frozenset(devs),
          shard_shape=lambda shape: (shape[0] // num_devices, *shape[1:]),
      )

    # 16-device sharded (160, 4) float32 -> per-shard shape (10, 4) = 160 bytes/shard.
    # TPU v5p has 4 devices/host -> 4 * 160 = 640 bytes/host.
    v5p_spec = jax.ShapeDtypeStruct((160, 4), jnp.float32, sharding=_make_sharding("TPU v5p", 16))
    self.assertEqual(checkpointing._est_host_bytes(v5p_spec), 4 * 160)

    # TPU v7x has 8 devices/host -> 8 * 160 = 1280 bytes/host.
    v7x_spec = jax.ShapeDtypeStruct((160, 4), jnp.float32, sharding=_make_sharding("TPU v7x", 16))
    self.assertEqual(checkpointing._est_host_bytes(v7x_spec), 8 * 160)


if __name__ == "__main__":
  unittest.main()

