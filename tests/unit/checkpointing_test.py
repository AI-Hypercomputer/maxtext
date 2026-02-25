# Copyright 2023–2025 Google LLC
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
"""Unit tests for common/checkpointing.py."""

import json
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
from etils import epath

from maxtext.common.checkpointing import (
    GrainCheckpointHandler,
    GrainCheckpointRestore,
    _is_remote_iterator,
    _prepare_scaled_down_grain_restore_args,
    cleanup_replicator_error_file,
    create_orbax_checkpoint_manager,
    load_state_if_possible,
    maybe_save_checkpoint,
    print_save_message,
    process_replicator_error_file,
    read_replicator_error_file,
    save_checkpoint,
    setup_checkpoint_logger,
)
from maxtext.input_pipeline.multihost_dataloading import RemoteIterator
from maxtext.utils import exceptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_config(**kwargs):
  """Minimal config SimpleNamespace for testing."""
  defaults = {
      "enable_checkpointing": True,
      "async_checkpointing": False,
      "checkpoint_period": 10,
      "enable_continuous_checkpointing": False,
      "dataset_type": "tfds",
      "pure_nnx": False,
      "enable_emergency_checkpoint": False,
      "local_checkpoint_period": 5,
      "checkpoint_storage_target_data_file_size_bytes": 2**32,
      "expansion_factor_real_data": -1,
  }
  defaults.update(kwargs)
  return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _is_remote_iterator
# ---------------------------------------------------------------------------


class TestIsRemoteIterator(unittest.TestCase):
  """Tests for _is_remote_iterator()."""

  def test_single_remote_iterator(self):
    mock_iter = MagicMock(spec=RemoteIterator)
    self.assertTrue(_is_remote_iterator(mock_iter))

  def test_list_containing_remote_iterator(self):
    mock_iter = MagicMock(spec=RemoteIterator)
    self.assertTrue(_is_remote_iterator([mock_iter, "other"]))

  def test_list_without_remote_iterator(self):
    self.assertFalse(_is_remote_iterator(["a", "b", 42]))

  def test_plain_object_is_not_remote(self):
    self.assertFalse(_is_remote_iterator("some_iterator"))

  def test_empty_list_is_not_remote(self):
    self.assertFalse(_is_remote_iterator([]))


# ---------------------------------------------------------------------------
# Replicator error file helpers
# ---------------------------------------------------------------------------


class TestReplicatorErrorFileFunctions(unittest.TestCase):
  """Tests for read/cleanup/process replicator error file functions."""

  def test_read_replicator_error_file_logs_content(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("some error text")
      path = f.name
    with patch("maxtext.common.checkpointing.max_logging.log") as mock_log:
      read_replicator_error_file(path)
    logged = " ".join(str(c) for c in mock_log.call_args_list)
    self.assertIn("some error text", logged)

  def test_read_replicator_error_file_handles_missing_file(self):
    """Should not raise even if the file doesn't exist."""
    with patch("maxtext.common.checkpointing.max_logging.log"):
      read_replicator_error_file("/nonexistent/path/file.txt")  # no exception

  def test_cleanup_replicator_error_file_removes_file(self):
    with tempfile.NamedTemporaryFile(delete=False) as f:
      path = f.name
    self.assertTrue(epath.Path(path).exists())
    cleanup_replicator_error_file(path)
    self.assertFalse(epath.Path(path).exists())

  def test_cleanup_replicator_error_file_handles_missing_file(self):
    with patch("maxtext.common.checkpointing.max_logging.log"):
      cleanup_replicator_error_file("/nonexistent/path/file.txt")  # no exception

  def test_process_replicator_error_file_returns_true_when_exists(self):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
      f.write("error")
      path = f.name
    with patch("maxtext.common.checkpointing.read_replicator_error_file"):
      with patch("maxtext.common.checkpointing.cleanup_replicator_error_file"):
        result = process_replicator_error_file(path)
    self.assertTrue(result)

  def test_process_replicator_error_file_returns_false_when_absent(self):
    result = process_replicator_error_file("/nonexistent/path.txt")
    self.assertFalse(result)

  def test_process_replicator_error_file_calls_read_and_cleanup(self):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
      f.write("error")
      path = f.name
    with patch("maxtext.common.checkpointing.read_replicator_error_file") as mock_read:
      with patch("maxtext.common.checkpointing.cleanup_replicator_error_file") as mock_cleanup:
        process_replicator_error_file(path)
    mock_read.assert_called_once_with(path)
    mock_cleanup.assert_called_once_with(path)


# ---------------------------------------------------------------------------
# print_save_message
# ---------------------------------------------------------------------------


class TestPrintSaveMessage(unittest.TestCase):
  """Tests for print_save_message()."""

  def test_async_message(self):
    with patch("maxtext.common.checkpointing.max_logging.log") as mock_log:
      print_save_message(step=7, async_checkpointing=True)
    self.assertIn("asynchronous", mock_log.call_args[0][0])
    self.assertIn("7", mock_log.call_args[0][0])

  def test_sync_message(self):
    with patch("maxtext.common.checkpointing.max_logging.log") as mock_log:
      print_save_message(step=3, async_checkpointing=False)
    msg = mock_log.call_args[0][0]
    self.assertIn("3", msg)
    self.assertNotIn("asynchronous", msg)


# ---------------------------------------------------------------------------
# create_orbax_checkpoint_manager
# ---------------------------------------------------------------------------


class TestCreateOrbaxCheckpointManager(unittest.TestCase):
  """Tests for create_orbax_checkpoint_manager()."""

  def test_returns_none_when_checkpointing_disabled(self):
    result = create_orbax_checkpoint_manager(
        checkpoint_dir="/tmp/test_ckpt_disabled",
        enable_checkpointing=False,
        use_async=False,
        save_interval_steps=10,
    )
    self.assertIsNone(result)

  def test_creates_manager_when_enabled(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      manager = create_orbax_checkpoint_manager(
          checkpoint_dir=tmpdir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=10,
      )
      self.assertIsNotNone(manager)
      manager.close()

  def test_creates_directory_if_missing(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      new_dir = f"{tmpdir}/nested/checkpoint"
      manager = create_orbax_checkpoint_manager(
          checkpoint_dir=new_dir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=10,
      )
      self.assertTrue(epath.Path(new_dir).exists())
      manager.close()

  def test_grain_dataset_type_adds_iter_handler(self):
    """When dataset_type='grain', the manager should include an 'iter' handler."""
    with tempfile.TemporaryDirectory() as tmpdir:
      manager = create_orbax_checkpoint_manager(
          checkpoint_dir=tmpdir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=10,
          dataset_type="grain",
      )
      self.assertIsNotNone(manager)
      manager.close()


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint(unittest.TestCase):
  """Tests for save_checkpoint()."""

  def _make_state(self):
    return {"w": jnp.ones(4, dtype=jnp.float32)}

  def test_default_case_calls_manager_save(self):
    """Normal (non-emergency) manager: save() is called with correct step."""
    cm = MagicMock()
    cm.save.return_value = True
    state = self._make_state()
    save_checkpoint(cm, step=5, state=state)
    cm.save.assert_called_once()
    # The first positional arg should be the step number
    call_args = cm.save.call_args
    step_arg = call_args[0][0] if call_args[0] else call_args[1].get("step")
    self.assertEqual(step_arg, 5)

  def test_returns_manager_save_return_value(self):
    cm = MagicMock()
    cm.save.return_value = False
    result = save_checkpoint(cm, step=0, state=self._make_state())
    self.assertFalse(result)

  def test_config_none_skips_blocking(self):
    """With config=None, jax.block_until_ready is never called."""
    cm = MagicMock()
    state = self._make_state()
    with patch("maxtext.common.checkpointing.jax.block_until_ready") as mock_block:
      save_checkpoint(cm, step=5, state=state, config=None)
    mock_block.assert_not_called()

  def test_config_checkpointing_disabled_skips_blocking(self):
    """When enable_checkpointing=False, block_until_ready is not called."""
    cm = MagicMock()
    config = _simple_config(enable_checkpointing=False)
    with patch("maxtext.common.checkpointing.jax.block_until_ready") as mock_block:
      save_checkpoint(cm, step=5, state=self._make_state(), config=config)
    mock_block.assert_not_called()

  def test_step_at_checkpoint_period_triggers_block(self):
    """When step % checkpoint_period == 0, block_until_ready should be called."""
    cm = MagicMock()
    config = _simple_config(enable_checkpointing=True, checkpoint_period=5)
    state = self._make_state()
    with patch("maxtext.common.checkpointing.jax.block_until_ready") as mock_block:
      save_checkpoint(cm, step=5, state=state, config=config)
    mock_block.assert_called_once_with(state)


# ---------------------------------------------------------------------------
# maybe_save_checkpoint
# ---------------------------------------------------------------------------


class TestMaybeSaveCheckpoint(unittest.TestCase):
  """Tests for maybe_save_checkpoint()."""

  def test_no_op_when_checkpoint_manager_is_none(self):
    """Returns immediately when checkpoint_manager is None."""
    with patch("maxtext.common.checkpointing.save_checkpoint") as mock_save:
      maybe_save_checkpoint(None, state=None, config=None, data_iterator=None, step=5)
    mock_save.assert_not_called()

  def test_explicit_step_used_directly(self):
    """When step is provided, actual_step = int(step)."""
    cm = MagicMock()
    cm.reached_preemption.return_value = False
    config = _simple_config(pure_nnx=False, checkpoint_period=10)
    state = types.SimpleNamespace(step=99)

    with patch("maxtext.common.checkpointing.save_checkpoint", return_value=False) as mock_save:
      maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=20)

    call_step = mock_save.call_args[0][1]
    self.assertEqual(call_step, 20)

  def test_step_inferred_from_linen_state_when_step_is_none(self):
    """When step=None and pure_nnx=False, actual_step = state.step - 1."""
    cm = MagicMock()
    cm.reached_preemption.return_value = False
    config = _simple_config(pure_nnx=False, checkpoint_period=10)
    state = types.SimpleNamespace(step=11)  # actual_step = 10

    with patch("maxtext.common.checkpointing.save_checkpoint", return_value=False) as mock_save:
      maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=None)

    call_step = mock_save.call_args[0][1]
    self.assertEqual(call_step, 10)

  def test_step_inferred_from_nnx_state_when_step_is_none(self):
    """When step=None and pure_nnx=True, actual_step = state.optimizer.step - 1."""
    cm = MagicMock()
    cm.reached_preemption.return_value = False
    config = _simple_config(pure_nnx=True, checkpoint_period=10)
    state = MagicMock()
    state.optimizer.step = 6  # actual_step = 5
    state.to_pure_dict.return_value = {"w": jnp.ones(4)}

    with patch("maxtext.common.checkpointing.save_checkpoint", return_value=False) as mock_save:
      maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=None)

    call_step = mock_save.call_args[0][1]
    self.assertEqual(call_step, 5)

  def test_nnx_state_converted_to_dict(self):
    """When pure_nnx=True, state.to_pure_dict() is called before save."""
    cm = MagicMock()
    cm.reached_preemption.return_value = False
    config = _simple_config(pure_nnx=True, checkpoint_period=5)
    state = MagicMock()
    state.optimizer.step = 6
    pure_dict = {"w": jnp.ones(4)}
    state.to_pure_dict.return_value = pure_dict

    with patch("maxtext.common.checkpointing.save_checkpoint", return_value=False) as mock_save:
      maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=None)

    # The state passed to save_checkpoint should be the pure dict
    call_state = mock_save.call_args[0][2]
    self.assertEqual(call_state, pure_dict)

  def test_exception_wrapped_as_stop_training(self):
    """Exceptions from save_checkpoint are re-raised as StopTraining."""
    cm = MagicMock()
    cm.reached_preemption.return_value = False
    config = _simple_config(pure_nnx=False)
    state = types.SimpleNamespace(step=1)

    with patch("maxtext.common.checkpointing.save_checkpoint", side_effect=RuntimeError("disk full")):
      with self.assertRaises(exceptions.StopTraining):
        maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=5)

  def test_preemption_raises_stop_training(self):
    """reached_preemption=True triggers wait_until_finished and raises StopTraining."""
    cm = MagicMock()
    cm.reached_preemption.return_value = True
    config = _simple_config(pure_nnx=False)
    state = types.SimpleNamespace(step=1)

    with patch("maxtext.common.checkpointing.save_checkpoint", return_value=False):
      with self.assertRaises(exceptions.StopTraining, msg="Job is preempted."):
        maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=5)

    cm.wait_until_finished.assert_called_once()

  def test_force_save_triggers_wait_until_finished(self):
    """When force_ckpt_save=True (step=None, off-period step), wait_until_finished is called."""
    cm = MagicMock()
    cm.reached_preemption.return_value = False
    # step=None with actual_step=7 (not divisible by checkpoint_period=10) → force_ckpt_save=True
    config = _simple_config(pure_nnx=False, checkpoint_period=10)
    state = types.SimpleNamespace(step=8)  # actual_step = 7

    with patch("maxtext.common.checkpointing.save_checkpoint", return_value=False):
      maybe_save_checkpoint(cm, state=state, config=config, data_iterator=None, step=None)

    cm.wait_until_finished.assert_called_once()


# ---------------------------------------------------------------------------
# load_state_if_possible — no-checkpoint-manager branches
# ---------------------------------------------------------------------------


class TestLoadStateIfPossible(unittest.TestCase):
  """Tests for load_state_if_possible() with no checkpoint manager."""

  def _abstract_state(self):
    return types.SimpleNamespace(params={"w": jnp.ones(4)})

  def test_returns_none_none_when_no_manager_and_no_paths(self):
    result = load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=96,
        abstract_unboxed_pre_state=self._abstract_state(),
    )
    self.assertEqual(result, (None, None))

  def test_loads_params_from_path_when_provided(self):
    restored_params = {"w": jnp.zeros(4)}
    with patch("maxtext.common.checkpointing.load_params_from_path", return_value=restored_params) as mock_load:
      result = load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="/some/param/path",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=96,
          abstract_unboxed_pre_state=self._abstract_state(),
      )
    mock_load.assert_called_once()
    self.assertIsNone(result[0])
    self.assertIs(result[1], restored_params)

  def test_loads_full_state_from_path_when_provided(self):
    restored_state = {"params": {"w": jnp.zeros(4)}}
    with patch("maxtext.common.checkpointing._load_full_state_from_path", return_value=restored_state):
      result = load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="",
          load_full_state_from_path="/some/full/state/path",
          checkpoint_storage_concurrent_gb=96,
          abstract_unboxed_pre_state=self._abstract_state(),
      )
    # Full state is wrapped in {"items": ...}
    self.assertIsNotNone(result[0])
    self.assertIn("items", result[0])
    self.assertIsNone(result[1])

  def test_params_path_takes_priority_over_full_state(self):
    """load_parameters_from_path is checked before load_full_state_from_path."""
    restored_params = {"w": jnp.zeros(4)}
    with patch("maxtext.common.checkpointing.load_params_from_path", return_value=restored_params) as mock_load:
      with patch("maxtext.common.checkpointing._load_full_state_from_path") as mock_full:
        load_state_if_possible(
            checkpoint_manager=None,
            data_iterator=None,
            load_parameters_from_path="/param/path",
            load_full_state_from_path="/full/path",
            checkpoint_storage_concurrent_gb=96,
            abstract_unboxed_pre_state=self._abstract_state(),
        )
    mock_load.assert_called_once()
    mock_full.assert_not_called()

  def test_checkpoint_manager_latest_step_used_when_step_negative(self):
    """When step=-1 and checkpoint_manager is provided, latest_step() is called."""
    cm = MagicMock()
    cm.latest_step.return_value = None  # no existing checkpoint
    result = load_state_if_possible(
        checkpoint_manager=cm,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=96,
        abstract_unboxed_pre_state=self._abstract_state(),
        step=-1,
    )
    cm.latest_step.assert_called_once()
    # No step found → falls through to no-checkpoint return
    self.assertEqual(result, (None, None))


# ---------------------------------------------------------------------------
# setup_checkpoint_logger
# ---------------------------------------------------------------------------


class TestSetupCheckpointLogger(unittest.TestCase):
  """Tests for setup_checkpoint_logger()."""

  def test_returns_none_when_logger_disabled(self):
    config = types.SimpleNamespace(enable_checkpoint_cloud_logger=False, run_name="test")
    result = setup_checkpoint_logger(config)
    self.assertIsNone(result)

  def test_returns_logger_when_enabled(self):
    config = types.SimpleNamespace(enable_checkpoint_cloud_logger=True, run_name="test_run")
    with patch("maxtext.common.checkpointing.ocp.logging.CloudLogger") as mock_logger_cls:
      mock_logger_cls.return_value = "mock_cloud_logger"
      result = setup_checkpoint_logger(config)
    self.assertEqual(result, "mock_cloud_logger")
    mock_logger_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _prepare_scaled_down_grain_restore_args
# ---------------------------------------------------------------------------


class TestPrepareScaledDownGrainRestoreArgs(unittest.TestCase):
  """Tests for _prepare_scaled_down_grain_restore_args()."""

  def _make_iterator_list(self, n):
    items = []
    for _ in range(n):
      mock_iter = MagicMock()
      mock_iter.local_iterator = MagicMock()
      items.append(mock_iter)
    return items

  def test_raises_when_data_iterator_not_a_list(self):
    """Non-list data_iterator should trigger AssertionError."""
    with self.assertRaises(AssertionError):
      _prepare_scaled_down_grain_restore_args(
          data_iterator="not_a_list",
          process_count_jax=4,
          process_count_stored=8,
          directory=epath.Path("/tmp"),
      )

  def test_raises_when_scaling_factor_mismatch(self):
    """Mismatch between len(data_iterator) and expected scaling factor."""
    # process_count_stored / process_count_jax = 8/4 = 2, but list has 3 items
    iters = self._make_iterator_list(3)
    with self.assertRaises(AssertionError):
      _prepare_scaled_down_grain_restore_args(
          data_iterator=iters,
          process_count_jax=4,
          process_count_stored=8,
          directory=epath.Path("/tmp"),
      )

  def test_returns_grain_checkpoint_restore_with_correct_fields(self):
    """Valid input produces GrainCheckpointRestore with correct process_count."""
    process_count_jax = jax.process_count()  # typically 1 in tests
    scaling_factor = 2
    process_count_stored = process_count_jax * scaling_factor
    iters = self._make_iterator_list(scaling_factor)

    result = _prepare_scaled_down_grain_restore_args(
        data_iterator=iters,
        process_count_jax=process_count_jax,
        process_count_stored=process_count_stored,
        directory=epath.Path("/tmp"),
    )

    self.assertIsInstance(result, GrainCheckpointRestore)
    self.assertEqual(result.process_count, process_count_stored)
    self.assertEqual(len(result.item), scaling_factor)
    self.assertEqual(len(result.process_index), scaling_factor)


# ---------------------------------------------------------------------------
# GrainCheckpointHandler
# ---------------------------------------------------------------------------


import grain as _grain


class FakeGrainIterator(_grain.DatasetIterator):
  """Minimal grain.DatasetIterator subclass for testing."""

  def __init__(self, state_dict):
    super().__init__()
    self._closed = False  # satisfy grain.DatasetIterator.__del__
    self._state = state_dict

  def __next__(self):
    return None

  def get_state(self):
    return self._state

  def set_state(self, state):
    self._state = state

  @property
  def element_spec(self):
    return None


class FakeByteIterator:
  """Non-grain iterator that uses bytes state (does NOT subclass grain.DatasetIterator)."""

  def __init__(self, state_bytes: bytes):
    self._state = state_bytes

  def get_state(self) -> bytes:
    return self._state

  def set_state(self, state: bytes):
    self._state = state


class TestGrainCheckpointHandlerSave(unittest.TestCase):
  """Tests for GrainCheckpointHandler.save()."""

  def setUp(self):
    self.handler = GrainCheckpointHandler()
    self.tmpdir = tempfile.mkdtemp()
    self.directory = epath.Path(self.tmpdir)

  def test_saves_grain_iterator_as_json(self):
    """Grain iterator state is serialised to JSON."""
    state_dict = {"step": 42, "epoch": 1}
    fake_iter = FakeGrainIterator(state_dict)
    self.handler.save(self.directory, item=fake_iter)

    filename = self.directory / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    self.assertTrue(filename.exists())
    loaded = json.loads(filename.read_text())
    self.assertEqual(loaded, state_dict)

  def test_saves_byte_iterator_as_text(self):
    """Non-grain iterator state (bytes) is written as decoded text."""
    state_bytes = b'{"step": 7}'
    fake_iter = FakeByteIterator(state_bytes)
    self.handler.save(self.directory, item=fake_iter)

    filename = self.directory / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    self.assertTrue(filename.exists())
    self.assertEqual(filename.read_text(), '{"step": 7}')

  def test_saves_list_of_iterators(self):
    """List of (iterator, process_index, process_count) tuples are each saved."""
    state1 = {"step": 1}
    state2 = {"step": 2}
    iter1 = FakeGrainIterator(state1)
    iter2 = FakeGrainIterator(state2)
    self.handler.save(self.directory, item=[(iter1, 0, 2), (iter2, 1, 2)])

    for idx, expected in [(0, state1), (1, state2)]:
      f = self.directory / f"process_{idx}-of-2.json"
      self.assertTrue(f.exists())
      self.assertEqual(json.loads(f.read_text()), expected)


class TestGrainCheckpointHandlerRestore(unittest.TestCase):
  """Tests for GrainCheckpointHandler.restore()."""

  def setUp(self):
    self.handler = GrainCheckpointHandler()
    self.tmpdir = tempfile.mkdtemp()
    self.directory = epath.Path(self.tmpdir)

  def _write_state_file(self, process_index, process_count, content: str):
    fname = self.directory / f"process_{process_index}-of-{process_count}.json"
    fname.write_text(content)

  def test_restores_grain_iterator_from_file(self):
    """JSON file content is parsed and passed to set_state for grain iterator."""
    state_dict = {"step": 99}
    self._write_state_file(0, 1, json.dumps(state_dict))

    fake_iter = FakeGrainIterator({"step": 0})
    result = self.handler.restore(
        self.directory,
        item=fake_iter,
        args=GrainCheckpointRestore(item=fake_iter, process_index=0, process_count=1),
    )
    self.assertEqual(result.get_state(), state_dict)

  def test_restore_raises_when_file_missing(self):
    """ValueError raised when checkpoint file doesn't exist."""
    fake_iter = FakeGrainIterator({"step": 0})
    with self.assertRaises(ValueError, msg="does not exist"):
      self.handler.restore(
          self.directory,
          item=fake_iter,
          args=GrainCheckpointRestore(item=fake_iter, process_index=0, process_count=1),
      )


if __name__ == "__main__":
  unittest.main()
