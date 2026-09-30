# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Format compatibility, FP8, and pinned_host tests for CheckpointManager."""

import glob
import json
import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.training_engine import checkpointing
import numpy as np
import optax


def _config():
  """Returns a CheckpointManager config for the on-disk Pathways persistence format."""
  return SimpleNamespace(
      checkpoint_period=1,
      max_num_checkpoints_to_keep=5,
      async_checkpointing=False,
      checkpoint_storage_use_ocdbt=False,
      checkpoint_storage_use_zarr3=False,
      checkpoint_storage_device_host_concurrent_gb=None,
      checkpoint_storage_concurrent_gb=96,
  )


class _Model(nnx.Module):
  """Simple NNX module with kernel and bias in specified dtype."""

  def __init__(self, dtype: jnp.dtype, factor: float = 1.0):
    self.kernel = nnx.Param(
        (jnp.array([[1.0, -0.5, 0.25, 2.0], [-1.0, 0.5, -0.25, -2.0]]) * factor).astype(dtype)
    )
    self.bias = nnx.Param((jnp.array([0.125, -0.25, 0.5, -0.5]) * factor).astype(dtype))


def _build_model_and_optimizer(dtype: jnp.dtype, factor: float = 1.0):
  """Builds model and optimizer with non-zero initial values parameterized by factor."""
  model = _Model(dtype, factor=factor)
  optimizer = nnx.Optimizer(model, optax.adamw(1e-2), wrt=nnx.Param)

  def _init_opt_leaf(leaf):
    if not isinstance(leaf, jax.Array):
      return leaf
    if leaf.ndim == 0:
      return jnp.array(int(factor * 2), dtype=leaf.dtype)
    val = jnp.arange(1, leaf.size + 1, dtype=jnp.float32).reshape(leaf.shape) * factor
    if jnp.issubdtype(leaf.dtype, jnp.floating):
      return val.astype(leaf.dtype)
    return leaf

  new_opt_state = jax.tree.map(_init_opt_leaf, nnx.state(optimizer, nnx.optimizer.OptState))
  nnx.update(optimizer, new_opt_state)
  return model, optimizer


def _apply_host_offload(optimizer: nnx.Optimizer) -> None:
  """Offloads optimizer state to pinned_host memory kind when supported on CPU."""
  devices = jax.devices()
  mesh = jax.sharding.Mesh(np.array(devices[:1]), ("data",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  try:
    host_sharding = sharding.with_memory_kind("pinned_host")
  except (ValueError, RuntimeError):
    host_sharding = sharding
  offloaded = jax.tree.map(
      lambda x: jax.device_put(x, host_sharding) if isinstance(x, jax.Array) else x,
      nnx.state(optimizer, nnx.optimizer.OptState),
  )
  nnx.update(optimizer, offloaded)


def _leaves(state):
  """Flat {path: array} mapping for leaf validation."""
  pure = nnx.to_pure_dict(state) if isinstance(state, nnx.State) else state
  return {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(pure)[0]}


class TrainingEngineCheckpointFormatTest(unittest.TestCase):
  """Tier B CPU tests verifying format compatibility, FP8 dtypes, and pinned_host offload."""

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)
    os.environ.pop("ENABLE_ORBAX_FINGERPRINT", None)

  def test_format_matrix_save_and_restore(self):
    dtypes = (
        jnp.bfloat16,
        jnp.float32,
        jnp.float8_e4m3fn,
        jnp.float8_e5m2,
    )
    offload_options = (False, True)

    for dtype in dtypes:
      for host_offload in offload_options:
        dtype_name = getattr(dtype, "__name__", str(dtype))
        with self.subTest(dtype=dtype_name, host_offload=host_offload):
          case_dir = os.path.join(self.ckpt_dir, f"{dtype_name}_{host_offload}")
          os.makedirs(case_dir, exist_ok=True)

          # 1. Save checkpoint
          save_model, save_opt = _build_model_and_optimizer(dtype, factor=1.0)
          if host_offload:
            _apply_host_offload(save_opt)

          save_mgr = checkpointing.CheckpointManager(case_dir, _config())
          saved = save_mgr.save_checkpoint(
              step=1,
              checkpoint_state=checkpointing.CheckpointState(model=save_model, optimizer=save_opt),
          )
          save_mgr.wait_until_finished()
          save_mgr.close()
          self.assertTrue(saved)

          # 2. Assert on-disk format: .zarray exists, no ocdbt, no zarr3
          step_dir = os.path.join(case_dir, "1")
          zarrays = glob.glob(os.path.join(step_dir, "**", ".zarray"), recursive=True)
          self.assertGreaterEqual(len(zarrays), 2)
          for zf in zarrays:
            with open(zf, "r", encoding="utf-8") as f:
              meta = json.load(f)
            self.assertIn("dtype", meta)
            self.assertIsInstance(meta["dtype"], str)
            self.assertGreater(len(meta["dtype"]), 0)

          ocdbt_paths = glob.glob(os.path.join(case_dir, "**", "*ocdbt*"), recursive=True)
          self.assertEqual(len(ocdbt_paths), 0)
          zarr3_paths = glob.glob(os.path.join(case_dir, "**", "zarr.json"), recursive=True)
          self.assertEqual(len(zarr3_paths), 0)

          # 3. Restore into fresh state with zero initial values
          fresh_model, fresh_opt = _build_model_and_optimizer(dtype, factor=0.0)
          if host_offload:
            _apply_host_offload(fresh_opt)

          restore_mgr = checkpointing.CheckpointManager(case_dir, _config())
          self.addCleanup(restore_mgr.close)
          restored_step, _, _ = restore_mgr.restore_checkpoint(
              checkpointing.CheckpointState(model=fresh_model, optimizer=fresh_opt),
              step=1,
          )
          self.assertEqual(restored_step, 1)

          # 4. Verify model parameters bit-exactness and dtype parity
          want_params = _leaves(nnx.state(save_model))
          got_params = _leaves(nnx.state(fresh_model))
          self.assertEqual(got_params.keys(), want_params.keys())
          self.assertEqual(len(got_params), 2)  # Load-bearing parameter leaf counter
          for name, want in want_params.items():
            got = got_params[name]
            self.assertEqual(got.dtype, want.dtype, msg=f"Param dtype mismatch at {name}")
            self.assertEqual(
                np.asarray(got).tobytes(),
                np.asarray(want).tobytes(),
                msg=f"Param bit mismatch at {name}",
            )
            self.assertTrue(bool(jnp.array_equal(got, want)), msg=f"Param array_equal at {name}")

          # 5. Verify optimizer state bit-exactness and dtype parity
          want_opt = _leaves(nnx.state(save_opt, nnx.optimizer.OptState))
          got_opt = _leaves(nnx.state(fresh_opt, nnx.optimizer.OptState))
          self.assertEqual(got_opt.keys(), want_opt.keys())
          self.assertEqual(len(got_opt), 6)  # Load-bearing optimizer leaf counter (step, count, mu x2, nu x2)
          for name, want in want_opt.items():
            got = got_opt[name]
            self.assertEqual(got.dtype, want.dtype, msg=f"OptState dtype mismatch at {name}")
            self.assertEqual(
                np.asarray(got).tobytes(),
                np.asarray(want).tobytes(),
                msg=f"OptState bit mismatch at {name}",
            )
            self.assertTrue(bool(np.array_equal(np.asarray(got), np.asarray(want))), msg=f"OptState array_equal at {name}")

  def test_fp8_restore_into_bfloat16_target_fails_or_mismatches(self):
    """Verifies Slice F mutation check: restoring FP8 checkpoint into bfloat16 target fails bit-exactness against saved FP8."""
    save_model, save_opt = _build_model_and_optimizer(jnp.float8_e4m3fn, factor=1.0)
    want_params = _leaves(nnx.state(save_model))
    save_mgr = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    saved = save_mgr.save_checkpoint(
        step=1,
        checkpoint_state=checkpointing.CheckpointState(model=save_model, optimizer=save_opt),
    )
    save_mgr.wait_until_finished()
    save_mgr.close()
    self.assertTrue(saved)

    target_model, target_opt = _build_model_and_optimizer(jnp.bfloat16, factor=0.0)
    restore_mgr = checkpointing.CheckpointManager(self.ckpt_dir, _config())
    self.addCleanup(restore_mgr.close)

    restore_mgr.restore_checkpoint(
        checkpointing.CheckpointState(model=target_model, optimizer=target_opt),
        step=1,
    )
    got_params = _leaves(nnx.state(target_model))
    with self.assertRaises(AssertionError):
      for name, want in want_params.items():
        self.assertEqual(got_params[name].dtype, want.dtype)
        self.assertEqual(np.asarray(got_params[name]).tobytes(), np.asarray(want).tobytes())


if __name__ == "__main__":
  unittest.main()
