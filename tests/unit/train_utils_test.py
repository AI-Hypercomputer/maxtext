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

"""Unit tests for train_utils.py."""

import itertools
import pathlib
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
import grain
import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from orbax.checkpoint import v1 as ocp
import pytest

from maxtext.common import grain_utility
from maxtext.configs import pyconfig
from maxtext.input_pipeline import multihost_dataloading
from maxtext.trainers.pre_train import train
from maxtext.utils import elastic_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from tests.utils.test_helpers import get_test_config_path
from maxtext.utils.train_utils import (
    validate_train_config,
    create_training_optimizer,
    validate_completed_steps,
)


@dataclass
class MockConfig:
  """Minimal mock config for validate_train_config tests."""

  run_name: str = "test_run"
  dataset_path: str = "gs://test-bucket/data"
  base_output_directory: str = "gs://test-bucket/output"
  steps: int = 100
  quantization: str = ""
  gradient_accumulation_steps: int = 1
  packing: bool = False
  dataset_type: str = "synthetic"

  # Fields needed for create_training_optimizer
  opt_type: str = "adamw"
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1e-8
  adam_eps_root: float = 0.0
  adam_weight_decay: float = 0.1
  mu_dtype: str = ""
  learning_rate: float = 1e-4
  learning_rate_schedule_steps: int = 1000
  warmup_steps_fraction: float = 0.1
  cosine_learning_rate_final_fraction: float = 0.0
  steps: int = 100
  lr_schedule_type: str = "cosine"
  use_iota_embed: bool = False


class _FakeIterator:
  """Minimal resettable iterator for reorder-view tests."""

  def __init__(self, batches):
    self._batches = batches
    self._index = 0
    self.reset_calls = 0

  def __next__(self):
    batch = self._batches[self._index]
    self._index += 1
    return batch

  def reset(self):
    self.reset_calls += 1
    self._index = 0


class TestReorderedDataIterator(unittest.TestCase):
  """Tests for the load-balanced CP reorder view."""

  # pylint: disable=protected-access

  def test_reorders_batches_and_forwards_reset(self):
    inner = _FakeIterator([1, 2])
    wrapped = train_utils._ReorderedDataIterator(lambda batch: batch * 10, inner)

    self.assertEqual(next(wrapped), 10)
    self.assertEqual(next(wrapped), 20)
    self.assertIs(iter(wrapped), wrapped)
    wrapped.reset()
    self.assertEqual(inner.reset_calls, 1)
    self.assertEqual(next(wrapped), 10)

  def test_loader_view_wraps_single_iterator(self):
    inner = _FakeIterator([1])
    view = train_utils._reorder_data_iterator_for_loader(lambda batch: batch * 10, inner)

    self.assertIsInstance(view, train_utils._ReorderedDataIterator)
    self.assertIs(view.data_iterator, inner)

  def test_loader_view_wraps_each_list_element(self):
    inners = [_FakeIterator([1]), _FakeIterator([2])]
    view = train_utils._reorder_data_iterator_for_loader(lambda batch: batch * 10, inners)

    self.assertIsInstance(view, list)
    self.assertEqual(len(view), 2)
    for wrapped, inner in zip(view, inners):
      self.assertIsInstance(wrapped, train_utils._ReorderedDataIterator)
      self.assertIs(wrapped.data_iterator, inner)
    self.assertEqual(next(view[1]), 20)

  def test_grain_checkpoint_round_trip_through_reorder_view(self):
    """Batches consumed through the view advance the saved Grain state, and an
    in-place restore is visible through a view built before the restore."""
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "ckpt"
      iterator = iter(grain.MapDataset.range(10).to_iter_dataset())
      view = train_utils._ReorderedDataIterator(lambda batch: batch * 10, iterator)
      self.assertEqual(next(view), 0)
      self.assertEqual(next(view), 10)
      ocp.save_checkpointables(str(path), {"iter": grain_utility.GrainCheckpointable(iterator)})
      self.assertTrue((path / "iter" / "process_0-of-1.json").exists())

      restored = iter(grain.MapDataset.range(10).to_iter_dataset())
      restored_view = train_utils._ReorderedDataIterator(lambda batch: batch * 10, restored)
      ocp.load_checkpointables(str(path), {"iter": grain_utility.GrainCheckpointable(restored)})
      self.assertEqual(next(restored_view), 20)

  def test_eval_view_consume_reset_consume(self):
    """The eval view repeats identical reordered passes across a reset."""
    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    batches = [{"x": np.full((len(jax.devices()), 2), i, dtype=np.int32)} for i in range(2)]
    eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(batches, mesh)
    view = train_utils._ReorderedDataIterator(lambda batch: {"x": batch["x"] + 100}, eval_iterator)

    first_pass = [int(batch["x"][0, 0]) for batch in itertools.islice(view, 2)]
    view.reset()
    second_pass = [int(batch["x"][0, 0]) for batch in itertools.islice(view, 2)]

    self.assertEqual(first_pass, [100, 101])
    self.assertEqual(second_pass, first_pass)

  def test_eval_reset_then_for_loop_builds_one_iterator(self):
    """train.py resets the eval iterator and then runs a for loop over it: one local iterator per eval."""
    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    batches = [{"x": np.full((len(jax.devices()), 2), i, dtype=np.int32)} for i in range(2)]

    class _CountingLoader:
      builds = 0

      def __iter__(self):
        _CountingLoader.builds += 1
        return iter(batches)

    eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(_CountingLoader(), mesh)
    self.assertEqual(_CountingLoader.builds, 1)
    for _ in range(2):
      eval_iterator.reset()
      self.assertEqual([int(b["x"][0, 0]) for b in eval_iterator], [0, 1])
    self.assertEqual(_CountingLoader.builds, 3)
    # A for loop without an explicit reset still starts from the first batch.
    self.assertEqual([int(b["x"][0, 0]) for b in eval_iterator], [0, 1])
    self.assertEqual(_CountingLoader.builds, 4)


class TestEvalCachePrefill(unittest.TestCase):
  """Tests for start_eval_cache_prefill (eval_cache_prefill_in_background)."""

  # pylint: disable=protected-access

  @staticmethod
  def _counting_tf_dataset(reads):
    """A tf.data pipeline shaped like the c4_mlperf eval loader (... -> cache -> prefetch) that counts source reads."""
    tf = pytest.importorskip("tensorflow")

    def _read(x):
      reads.append(int(x))
      return np.full((jax.local_device_count(), 2), int(x), dtype=np.int32)

    ds = tf.data.Dataset.range(3)
    ds = ds.map(lambda x: {"x": tf.py_function(_read, [x], tf.int32)})
    return ds.cache().prefetch(1)

  def test_prefill_commits_tf_data_cache_through_reorder_view(self):
    reads = []
    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(self._counting_tf_dataset(reads), mesh)
    view = train_utils._ReorderedDataIterator(lambda batch: batch, eval_iterator)

    thread = train_utils.start_eval_cache_prefill(view)
    self.assertIsNotNone(thread)
    thread.join(timeout=60)
    self.assertFalse(thread.is_alive())
    self.assertEqual(reads, [0, 1, 2])
    # The eval loop resets and re-iterates the same dataset object: the cache is now served from memory.
    view.reset()
    self.assertEqual(len(list(itertools.islice(view, 3))), 3)
    self.assertEqual(reads, [0, 1, 2])

  def test_prefill_skips_non_tf_data_loaders(self):
    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    batches = [{"x": np.zeros((len(jax.devices()), 2), dtype=np.int32)}]
    eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(batches, mesh)
    self.assertIsNone(train_utils.start_eval_cache_prefill(eval_iterator))
    self.assertIsNone(train_utils.start_eval_cache_prefill(_FakeIterator([1])))


class TestValidateTrainConfig(unittest.TestCase):
  """Tests for validate_train_config."""

  def test_valid_config_passes(self):
    """Verifies no exception raised for a valid config."""
    config = MockConfig()
    # Should not raise
    validate_train_config(config)

  def test_missing_run_name_raises(self):
    """Verifies AssertionError when run_name is empty."""
    config = MockConfig(run_name="")
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_zero_steps_raises(self):
    """Verifies AssertionError when steps is 0."""
    config = MockConfig(steps=0)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_negative_steps_raises(self):
    """Verifies AssertionError when steps is negative."""
    config = MockConfig(steps=-5)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_fp8_with_grad_accumulation_raises(self):
    """Verifies AssertionError for fp8 quantization + gradient_accumulation_steps > 1."""
    config = MockConfig(quantization="fp8", gradient_accumulation_steps=2)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_nanoo_fp8_with_grad_accumulation_raises(self):
    """Verifies AssertionError for nanoo_fp8 quantization + gradient_accumulation_steps > 1."""
    config = MockConfig(quantization="nanoo_fp8", gradient_accumulation_steps=4)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_fp8_with_single_grad_accumulation_passes(self):
    """Verifies no error for fp8 with gradient_accumulation_steps=1."""
    config = MockConfig(quantization="fp8", gradient_accumulation_steps=1)
    validate_train_config(config)  # Should not raise

  def test_packing_with_synthetic_data_logs_warning(self):
    """Verifies no exception for packing + synthetic (just logs a warning)."""
    config = MockConfig(packing=True, dataset_type="synthetic")
    # Should not raise - just log a warning
    validate_train_config(config)

  def test_local_dataset_path_logs_warning(self):
    """Verifies no exception for local dataset_path (just logs a warning)."""
    config = MockConfig(dataset_path="/local/path/to/data")
    validate_train_config(config)  # Should not raise

  def test_local_output_directory_logs_warning(self):
    """Verifies no exception for local base_output_directory (just logs a warning)."""
    config = MockConfig(base_output_directory="/local/output")
    validate_train_config(config)  # Should not raise


class TestCreateTrainingOptimizer(unittest.TestCase):
  """Tests for create_training_optimizer."""

  def _make_config(self, opt_type="adamw", **kwargs):
    """Creates a mock config for optimizer tests."""
    cfg = MockConfig(opt_type=opt_type, **kwargs)
    return cfg

  def _mock_lr_schedule(self):
    """Returns a mock learning rate schedule that returns a fixed value."""
    return lambda step: 1e-4

  def test_adamw_optimizer_returns_schedule_and_tx(self):
    """Verifies create_training_optimizer returns a schedule and optax transform for adamw."""
    config = MagicMock()
    config.opt_type = "adamw"
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_eps = 1e-8
    config.adam_eps_root = 0.0
    config.adam_weight_decay = 0.01
    config.mu_dtype = None
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.1
    config.cosine_learning_rate_final_fraction = 0.0
    config.steps = 100
    config.learning_rate_schedule_steps = 100
    config.lr_schedule_type = "cosine"
    config.use_iota_embed = False

    schedule, tx = create_training_optimizer(config, model=None)

    self.assertIsNotNone(schedule)
    self.assertIsNotNone(tx)
    # Verify it's an optax GradientTransformation
    self.assertTrue(hasattr(tx, "init"))
    self.assertTrue(hasattr(tx, "update"))

  def test_adam_pax_optimizer_returns_tx(self):
    """Verifies create_training_optimizer works for adam_pax optimizer."""
    config = MagicMock()
    config.opt_type = "adam_pax"
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_eps = 1e-8
    config.adam_eps_root = 0.0
    config.adam_weight_decay = 0.01
    config.mu_dtype = None
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.1
    config.cosine_learning_rate_final_fraction = 0.0
    config.steps = 100
    config.learning_rate_schedule_steps = 100
    config.lr_schedule_type = "cosine"
    config.use_iota_embed = False

    _, tx = create_training_optimizer(config, model=None)

    self.assertIsNotNone(tx)
    self.assertTrue(hasattr(tx, "init"))
    self.assertTrue(hasattr(tx, "update"))

  def test_sgd_optimizer_returns_tx(self):
    """Verifies create_training_optimizer works for sgd optimizer."""
    config = MagicMock()
    config.opt_type = "sgd"
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.0
    config.cosine_learning_rate_final_fraction = 0.0
    config.steps = 100
    config.learning_rate_schedule_steps = 100
    config.lr_schedule_type = "cosine"
    config.use_iota_embed = False
    _, tx = create_training_optimizer(config, model=None)
    self.assertIsNotNone(tx)
    self.assertTrue(hasattr(tx, "init"))


class TestCreateCheckpointManager(unittest.TestCase):
  """Tests for create_checkpoint_manager."""

  def test_single_controller_mtc_registers_colocated_python_handlers(self):
    config = SimpleNamespace(
        enable_multi_tier_checkpointing=True,
        enable_checkpoint_cloud_logger=False,
        run_name="test_run",
        local_checkpoint_directory="/tmp/mtc",
        local_checkpoint_period=10,
        colocated_python_checkpointing=True,
        enable_single_controller=True,
        enable_single_replica_ckpt_restoring=True,
    )
    mesh = object()
    checkpoint_manager = object()
    checkpointing_impl = object()

    with (
        mock.patch.object(
            train_utils.emergency_checkpointing,
            "create_replicator_checkpoint_manager",
            return_value=checkpoint_manager,
        ) as create_manager,
        mock.patch.object(
            train_utils.ocp_pathways.CheckpointingImpl,
            "from_options",
            return_value=checkpointing_impl,
        ) as from_options,
        mock.patch.object(train_utils.ocp_pathways, "register_type_handlers") as register_type_handlers,
    ):
      result = train_utils.create_checkpoint_manager(config, mesh, init_state_fn=object())

    self.assertIs(result, checkpoint_manager)
    create_manager.assert_called_once_with(
        config.local_checkpoint_directory,
        config.local_checkpoint_period,
        mesh,
        config.colocated_python_checkpointing,
    )
    from_options.assert_called_once_with(use_colocated_python=True)
    register_type_handlers.assert_called_once_with(
        use_single_replica_array_handler=config.enable_single_replica_ckpt_restoring,
        checkpointing_impl=checkpointing_impl,
    )


class TestValidateCompletedSteps(unittest.TestCase):
  """Tests for validate_completed_steps."""

  def test_under_steps_passes(self):
    """Verifies no exception raised when completed_steps < config_steps."""
    # Should not raise
    validate_completed_steps(completed_steps=50, config_steps=100)

  def test_equal_steps_raises(self):
    """Verifies RuntimeError raised when completed_steps == config_steps."""
    with self.assertRaises(RuntimeError) as context:
      validate_completed_steps(completed_steps=100, config_steps=100)
    self.assertIn("Requested training up to step 100, but a checkpoint already exists at step 99", str(context.exception))

  def test_over_steps_raises(self):
    """Verifies RuntimeError raised when completed_steps > config_steps."""
    with self.assertRaises(RuntimeError) as context:
      validate_completed_steps(completed_steps=105, config_steps=100)
    self.assertIn(
        "Requested training up to step 100, but a checkpoint already exists at step 104", str(context.exception)
    )


class TestCompileXlaFlags(unittest.TestCase):
  """Tests that compile_xla_flags are applied to JIT and AOT-compiled train/eval steps without step-0 recompilation."""

  # pylint: disable=protected-access

  @staticmethod
  def _make_compile_config(compile_xla_flags: str):
    """Builds a minimal config namespace for jit_train_and_eval_step and aot_compile_step."""
    return SimpleNamespace(
        enable_diloco=False,
        compiled_trainstep_file="",
        logical_axis_rules=(("batch", "data"),),
        logical_axis_rules_for_eval=(("batch", "data"),),
        shard_mode="auto",
        input_data_sharding_logical_axes=("batch",),
        compile_xla_flags=compile_xla_flags,
    )

  @staticmethod
  def _dummy_train_step(model, config, state_mesh_shardings, params_shardings, state, batch):
    del model, config, state_mesh_shardings, params_shardings
    return {"w": state["w"] + batch["inputs"]}, {"loss": jax.numpy.sum(batch["inputs"])}

  @staticmethod
  def _dummy_eval_step(model, config, state, batch):
    del model, config
    return {"eval_loss": jax.numpy.sum(state["w"] * batch["inputs"])}

  def test_jit_train_and_eval_step_attaches_compile_xla_flags(self):
    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    config = self._make_compile_config("--xla_cpu_enable_fast_math=False --xla_sc_disable_megacore_partitioning=true")
    state = {"w": jax.device_put(np.ones((len(jax.devices()), 4), dtype=np.float32), replicated)}
    state_shardings = {"w": replicated}

    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        config,
        model=None,
        mesh=mesh,
        state=state,
        state_mesh_shardings=state_shardings,
        train_step=self._dummy_train_step,
        eval_step=self._dummy_eval_step,
        eval_data_iterator=[1],
        params_shardings=state_shardings,
    )

    expected_kvs = (
        ("xla_cpu_enable_fast_math", "False"),
        ("xla_sc_disable_megacore_partitioning", "true"),
    )
    self.assertEqual(p_train_step._jit_info.compiler_options_kvs, expected_kvs)
    self.assertEqual(p_eval_step._jit_info.compiler_options_kvs, expected_kvs)

  def test_aot_compile_step_and_step0_use_option_compiled_executable(self):
    pxla = sys.modules["jax._src.interpreters.pxla"]

    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    config = self._make_compile_config("--xla_cpu_enable_fast_math=False")
    compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
    batch_shape = (len(jax.devices()), 4)
    state = {"w": jax.device_put(np.ones(batch_shape, dtype=np.float32), replicated)}
    state_shardings = {"w": replicated}

    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        config,
        model=None,
        mesh=mesh,
        state=state,
        state_mesh_shardings=state_shardings,
        train_step=self._dummy_train_step,
        eval_step=self._dummy_eval_step,
        eval_data_iterator=[1],
        params_shardings=state_shardings,
    )

    data_sharding = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules)
    shaped_batch = {"inputs": jax.ShapeDtypeStruct(batch_shape, jax.numpy.float32, sharding=data_sharding)}
    real_batch = {"inputs": jax.device_put(np.ones(batch_shape, dtype=np.float32), data_sharding)}

    expected_kvs = (("xla_cpu_enable_fast_math", "False"),)
    aot_compile_fn = getattr(train_utils, "aot_compile_step", None)
    self.assertIsNotNone(aot_compile_fn, "train_utils.aot_compile_step must be defined")

    with mock.patch.object(
        pxla.UnloadedMeshExecutable, "from_hlo", wraps=pxla.UnloadedMeshExecutable.from_hlo
    ) as from_hlo_spy:
      with jax.set_mesh(mesh):
        compiled_train = aot_compile_fn(p_train_step, (state, shaped_batch), compiler_options, prefix="train")
        compiled_eval = aot_compile_fn(p_eval_step, (state, shaped_batch), compiler_options, prefix="eval")

        self.assertEqual(from_hlo_spy.call_count, 2)
        for call in from_hlo_spy.call_args_list:
          self.assertEqual(call.kwargs["compiler_options_kvs"], expected_kvs)

        # The option-compiled executables must be cached on the lowered MeshComputation objects.
        self.assertIs(p_train_step.lower(state, shaped_batch)._lowering._executable, compiled_train._executable)
        self.assertIs(p_eval_step.lower(state, shaped_batch)._lowering._executable, compiled_eval._executable)

        # Executing step 0 must reuse the option-compiled executables without recompiling.
        new_state, train_metrics = p_train_step(state, real_batch)
        eval_metrics = p_eval_step(new_state, real_batch)
        jax.block_until_ready((new_state, train_metrics, eval_metrics))

        self.assertEqual(from_hlo_spy.call_count, 2)

  def test_empty_compile_xla_flags_caches_executable_without_recompiling(self):
    pxla = sys.modules["jax._src.interpreters.pxla"]

    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    config = self._make_compile_config("")
    compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
    batch_shape = (len(jax.devices()), 4)
    state = {"w": jax.device_put(np.ones(batch_shape, dtype=np.float32), replicated)}
    state_shardings = {"w": replicated}

    p_train_step, _ = train_utils.jit_train_and_eval_step(
        config,
        model=None,
        mesh=mesh,
        state=state,
        state_mesh_shardings=state_shardings,
        train_step=self._dummy_train_step,
        params_shardings=state_shardings,
    )
    self.assertEqual(p_train_step._jit_info.compiler_options_kvs, ())

    data_sharding = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules)
    shaped_batch = {"inputs": jax.ShapeDtypeStruct(batch_shape, jax.numpy.float32, sharding=data_sharding)}
    real_batch = {"inputs": jax.device_put(np.ones(batch_shape, dtype=np.float32), data_sharding)}

    aot_compile_fn = getattr(train_utils, "aot_compile_step", None)
    self.assertIsNotNone(aot_compile_fn, "train_utils.aot_compile_step must be defined")

    with mock.patch.object(
        pxla.UnloadedMeshExecutable, "from_hlo", wraps=pxla.UnloadedMeshExecutable.from_hlo
    ) as from_hlo_spy:
      with jax.set_mesh(mesh):
        compiled_train = aot_compile_fn(p_train_step, (state, shaped_batch), compiler_options, prefix="train")
        self.assertEqual(from_hlo_spy.call_count, 1)
        self.assertIs(p_train_step.lower(state, shaped_batch)._lowering._executable, compiled_train._executable)

        new_state, _ = p_train_step(state, real_batch)
        jax.block_until_ready(new_state)
        self.assertEqual(from_hlo_spy.call_count, 1)


class TestElasticTrainWrapperHbmFallback(unittest.TestCase):
  """`get_train_func`'s `elastic_train_wrapper` rebuilds an attempt with the per-device batch bound on HBM overrun."""

  def setUp(self):
    super().setUp()
    self.config = SimpleNamespace(elastic_enabled=True)
    self.argv = ["train.py", "base.yml", "per_device_batch_size=2"]
    self.attempt_argvs = []

    def fake_initialize(argv):
      self.attempt_argvs.append(list(argv))
      return SimpleNamespace(attempt=len(self.attempt_argvs)), mock.sentinel.recorder

    def fake_elastic_retry(config, callback_fn=None, pre_callback_fn=None, retry_cache=None):
      del config, callback_fn, pre_callback_fn
      # Calls straight through, with the retry cache turned on as `elastic_retry` does for the run.
      retry_cache.reset(enabled=True)
      return lambda fn: fn

    patches = {
        "initialize": mock.patch.object(train, "initialize", side_effect=fake_initialize),
        "run": mock.patch.object(train, "run"),
        "elastic_retry": mock.patch.object(train.elastic_utils, "elastic_retry", side_effect=fake_elastic_retry),
        "cleanup_iterators": mock.patch.object(train.elastic_utils, "cleanup_iterators"),
        "create_snapshotter": mock.patch.object(train.snapshotting, "create_snapshotter", return_value=None),
    }
    self.mocks = {name: self.enterContext(patch) for name, patch in patches.items()}

  def test_hbm_budget_exceeded_rebuilds_the_attempt_with_the_bound(self):
    """The first attempt overruns HBM; the rebuild carries the override and succeeds without consuming a retry."""
    self.mocks["run"].side_effect = [elastic_utils.HbmBudgetExceededError(1, required_bytes=95, budget_bytes=90), None]

    train.get_train_func(self.config, mock.sentinel.recorder, self.argv)()

    self.assertEqual(
        self.attempt_argvs,
        [self.argv, [*self.argv, "elastic_max_per_device_batch_size=1"]],
    )
    self.assertEqual(self.mocks["run"].call_count, 2)
    # The rebuilt attempt runs with the config built from the overridden argv, not the cached one.
    self.assertEqual(self.mocks["run"].call_args.args[0].attempt, 2)
    self.mocks["cleanup_iterators"].assert_called_once()
    self.mocks["elastic_retry"].assert_called_once()

  def test_repeated_hbm_budget_exceeded_with_the_same_override_propagates(self):
    """A rebuilt attempt that still overruns HBM with the same bound raises instead of looping."""
    self.mocks["run"].side_effect = elastic_utils.HbmBudgetExceededError(1, required_bytes=95, budget_bytes=90)

    with self.assertRaises(elastic_utils.HbmBudgetExceededError):
      train.get_train_func(self.config, mock.sentinel.recorder, self.argv)()

    self.assertEqual(
        self.attempt_argvs,
        [self.argv, [*self.argv, "elastic_max_per_device_batch_size=1"]],
    )
    self.assertEqual(self.mocks["run"].call_count, 2)
    self.mocks["cleanup_iterators"].assert_called_once()


class TestDeepSeek41MultimodalJitStep(unittest.TestCase):
  """Tests jit_train_and_eval_step and DiLoCo jit_train_step across varying DS4.1 batch key sets."""

  # pylint: disable=protected-access

  def test_aot_and_variable_key_batches_with_cp_gt_num_images(self):
    if jax.device_count() < 2:
      self.skipTest("Requires >= 2 host devices to test CP=2 > max_num_images_per_example=1.")
    n_dp = max(jax.device_count() // 2, 1)
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        max_num_images_per_example=1,
        per_device_batch_size=1.0,
        eval_per_device_batch_size=1.0,
        max_target_length=32,
        ici_context_parallelism=2,
        ici_data_parallelism=n_dp,
        ici_fsdp_parallelism=1,
        skip_jax_distributed_system=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_shd = {"w": replicated}
    bsz = int(round(cfg.per_device_batch_size * jax.device_count()))

    def make_state():
      return {"w": jax.device_put(np.ones((bsz, cfg.max_target_length), dtype=np.float32), replicated)}

    def dummy_train(_m, _c, _ss, _ps, state, batch):
      total = jax.numpy.sum(batch["inputs"])
      if "images" in batch:
        total = total + jax.numpy.sum(batch["images"]).astype(jax.numpy.int32)
      if "forced_routed_experts" in batch:
        total = total + jax.numpy.sum(batch["forced_routed_experts"])
      if "image_masks" in batch:
        total = total + jax.numpy.sum(batch["image_masks"])
      return {"w": state["w"] + batch["inputs"].astype(jax.numpy.float32)}, {"loss": total}

    def dummy_eval(_m, _c, state, batch):
      total = jax.numpy.sum(state["w"]) + jax.numpy.sum(batch["inputs"]).astype(jax.numpy.float32)
      if "images" in batch:
        total = total + jax.numpy.sum(batch["images"]).astype(jax.numpy.float32)
      if "image_masks" in batch:
        total = total + jax.numpy.sum(batch["image_masks"]).astype(jax.numpy.float32)
      return {"eval_loss": total}

    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        cfg,
        model=None,
        mesh=mesh,
        state=make_state(),
        state_mesh_shardings=state_shd,
        train_step=dummy_train,
        eval_step=dummy_eval,
        eval_data_iterator=[1],
        params_shardings=state_shd,
    )

    train_base_shd = sharding.get_input_data_sharding(cfg, mesh, rules=cfg.logical_axis_rules)
    eval_base_shd = sharding.get_input_data_sharding(cfg, mesh, rules=cfg.logical_axis_rules_for_eval)
    shaped_train = maxtext_utils.get_shaped_batch(cfg, batch_sharding=train_base_shd)
    shaped_eval = maxtext_utils.get_shaped_batch(cfg, batch_sharding=eval_base_shd, is_eval=True)

    def make_real_batch(shaped, base_shd, extra_keys=(), drop_keys=()):
      raw = {
          k: np.ones(v.shape, dtype=np.int32 if v.dtype == jax.numpy.int32 else np.float32)
          for k, v in shaped.items()
          if k not in drop_keys
      }
      for k, shape in extra_keys:
        raw[k] = np.ones(shape, dtype=np.int32)
      return jax.device_put(raw, sharding.get_batch_sharding_for_batch(cfg, base_shd, raw))

    mm_train = make_real_batch(shaped_train, train_base_shd)
    mm_eval = make_real_batch(shaped_eval, eval_base_shd)
    text_train = make_real_batch(shaped_train, train_base_shd, drop_keys=("images",))
    text_eval = make_real_batch(shaped_eval, eval_base_shd, drop_keys=("images",))
    replay_train = make_real_batch(shaped_train, train_base_shd, extra_keys=(("forced_routed_experts", (bsz, 32, 2)),))
    masks_train = make_real_batch(shaped_train, train_base_shd, extra_keys=(("image_masks", (bsz, 1)),))
    masks_eval = make_real_batch(shaped_eval, eval_base_shd, extra_keys=(("image_masks", (bsz, 1)),))

    pxla = sys.modules["jax._src.interpreters.pxla"]
    with mock.patch.object(pxla.UnloadedMeshExecutable, "from_hlo", wraps=pxla.UnloadedMeshExecutable.from_hlo) as spy:
      with jax.set_mesh(mesh):
        train_utils.aot_compile_step(p_train_step, (make_state(), shaped_train), prefix="train")
        train_utils.aot_compile_step(p_eval_step, (make_state(), shaped_eval), prefix="eval")
        aot_calls = spy.call_count
        _, m_mm = p_train_step(make_state(), mm_train)
        ev_mm = p_eval_step(make_state(), mm_eval)
        jax.block_until_ready((m_mm, ev_mm))
        # Canonical 7-key batch reuses the AOT executables without recompiling.
        self.assertEqual(spy.call_count, aot_calls)

      for name, tr_b in (("text_only", text_train), ("router_replay", replay_train), ("image_masks", masks_train)):
        with self.subTest(train_variant=name):
          _, m = p_train_step(make_state(), tr_b)
          self.assertGreater(int(m["loss"]), 0)

      for name, ev_b in (("text_only", text_eval), ("image_masks", masks_eval)):
        with self.subTest(eval_variant=name):
          m = p_eval_step(make_state(), ev_b)
          self.assertGreater(float(m["eval_loss"]), 0.0)

  def test_diloco_jit_train_step_handles_mm_and_text_only_batches(self):
    if jax.device_count() < 4:
      self.skipTest("Requires >= 4 host devices to test DiLoCo=2 x CP=2 > max_num_images_per_example=1.")
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        max_num_images_per_example=1,
        enable_diloco=True,
        ici_diloco_parallelism=2,
        per_device_batch_size=1.0,
        max_target_length=32,
        ici_context_parallelism=2,
        ici_data_parallelism=1,
        ici_fsdp_parallelism=1,
        skip_jax_distributed_system=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    base_shd = sharding.get_input_data_sharding(cfg, mesh, rules=cfg.logical_axis_rules)
    shaped_batch = maxtext_utils.get_shaped_batch(cfg, batch_sharding=base_shd)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_shd = {"w": replicated}
    rng = jax.random.PRNGKey(0)

    def make_state():
      return {"w": jax.device_put(np.ones((2, 2, 32), dtype=np.float32), replicated)}

    def diloco_train(state, batch, step_rng):
      del step_rng
      total = jax.numpy.sum(batch["inputs"])
      if "images" in batch:
        total = total + jax.numpy.sum(batch["images"]).astype(jax.numpy.int32)
      return {"w": state["w"] + batch["inputs"].astype(jax.numpy.float32)}, {"loss": total}

    p_train = train_utils.jit_train_step(cfg, None, make_state(), state_shd, base_shd, diloco_train, state_shd, mesh=mesh)
    mm_raw = {
        k: np.ones(v.shape, dtype=np.int32 if v.dtype == jax.numpy.int32 else np.float32) for k, v in shaped_batch.items()
    }
    text_raw = {k: v for k, v in mm_raw.items() if k != "images"}
    mm_batch = jax.device_put(mm_raw, sharding.get_batch_sharding_for_batch(cfg, base_shd, mm_raw))
    text_batch = jax.device_put(text_raw, sharding.get_batch_sharding_for_batch(cfg, base_shd, text_raw))

    _, m_mm = p_train(make_state(), mm_batch, rng)
    _, m_text = p_train(make_state(), text_batch, rng)
    self.assertGreater(int(m_mm["loss"]), int(m_text["loss"]))


if __name__ == "__main__":
  absltest.main()
