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
import os
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
    restored = manager.restore_checkpoint(
        checkpointing.CheckpointState(model=model, optimizer=optimizer), step=step
    )
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


if __name__ == "__main__":
  unittest.main()
