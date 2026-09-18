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

"""Unit tests for MLPerf compliance logging utilities.

``mlperf_logging`` is an optional dependency that is not installed in the
default test environment, so these tests substitute a fake ``mllog`` module and
assert on the resulting event stream.
"""

# These are white-box tests: mllog_utils keeps its state in module-level
# globals, so asserting on them is the only way to verify configuration.
# pylint: disable=protected-access

import os
import tempfile
import types
import unittest
from unittest import mock

from maxtext.utils import mllog_utils


# Subset of mlperf_logging.mllog.constants used by mllog_utils. Values match the
# upstream constants so that assertions read like the emitted log lines.
_CONSTANTS = types.SimpleNamespace(
    ADAMW="adamw",
    BLOCK_START="block_start",
    BLOCK_STOP="block_stop",
    CACHE_CLEAR="cache_clear",
    CLOSED="closed",
    CLOUD="cloud",
    CONFIG_FILENAME="config_filename",
    CONTEXT_PARALLELISM="context_parallelism",
    DEEPSEEKV3_671B="deepseekv3_671b",
    EVAL_ACCURACY="eval_accuracy",
    EVAL_SAMPLES="eval_samples",
    EVAL_START="eval_start",
    EVAL_STOP="eval_stop",
    EXPERT_PARALLELISM="expert_parallelism",
    GLOBAL_BATCH_SIZE="global_batch_size",
    GRADIENT_ACCUMULATION_STEPS="gradient_accumulation_steps",
    INIT_CHECKPOINT_STEP="init_checkpoint_step",
    INIT_START="init_start",
    INIT_STOP="init_stop",
    LOWEST_NUMERICAL_PRECISION_IN_ATTN="lowest_numerical_precision_in_attn",
    LOWEST_NUMERICAL_PRECISION_IN_COMM="lowest_numerical_precision_in_comm",
    LOWEST_NUMERICAL_PRECISION_IN_LINEAR="lowest_numerical_precision_in_linear",
    MAX_SEQUENCE_LENGTH="max_sequence_length",
    MAX_STEPS="max_steps",
    MICRO_BATCH_SIZE="micro_batch_size",
    MOE_AUX_LOSS_COEFF="moe_aux_loss_coeff",
    OPT_ADAMW_BETA_1="opt_adamw_beta_1",
    OPT_ADAMW_BETA_2="opt_adamw_beta_2",
    OPT_ADAMW_EPSILON="opt_adamw_epsilon",
    OPT_ADAMW_WEIGHT_DECAY="opt_adamw_weight_decay",
    OPT_BASE_LR="opt_base_learning_rate",
    OPT_END_LR="opt_end_learning_rate",
    OPT_GRADIENT_CLIP_NORM="opt_gradient_clip_norm",
    OPT_LR_DECAY_SCHEDULE="opt_learning_rate_decay_schedule",
    OPT_LR_DECAY_STEPS="opt_learning_rate_decay_steps",
    OPT_LR_WARMUP_STEPS="opt_learning_rate_warmup_steps",
    OPT_NAME="opt_name",
    PIPELINE_PARALLELISM="pipeline_parallelism",
    RUN_START="run_start",
    RUN_STOP="run_stop",
    SAMPLES_COUNT="samples_count",
    SEED="seed",
    SUBMISSION_BENCHMARK="submission_benchmark",
    SUBMISSION_DIVISION="submission_division",
    SUBMISSION_ORG="submission_org",
    SUBMISSION_PLATFORM="submission_platform",
    SUBMISSION_STATUS="submission_status",
    TENSOR_PARALLELISM="tensor_parallelism",
    TRAIN_SAMPLES="train_samples",
)


class FakeMllogger:
  """Records the events that mllog_utils emits."""

  def __init__(self):
    self.events = []

  def start(self, key, value=None, metadata=None):
    self.events.append(("start", key, value, metadata or {}))

  def end(self, key, value=None, metadata=None):
    self.events.append(("end", key, value, metadata or {}))

  def event(self, key, value=None, metadata=None):
    self.events.append(("event", key, value, metadata or {}))

  def keys(self):
    return [key for _, key, _, _ in self.events]

  def value_of(self, key):
    return next(value for _, event_key, value, _ in self.events if event_key == key)

  def metadata_of(self, key):
    return next(metadata for _, event_key, _, metadata in self.events if event_key == key)


def make_config(**overrides):
  """Builds a config stub with the fields mllog_utils reads."""
  config = {
      "enable_mllog": True,
      "mllog_file": "",
      "run_name": "unit-test-run",
      "data_shuffle_seed": 1234,
      "steps": 12000,
      "global_batch_size_to_train_on": 16384,
      "global_batch_size_to_eval_on": 1024,
      "per_device_batch_size": 1.0,
      "gradient_accumulation_steps": 1,
      "max_target_length": 4096,
      "eval_steps": 1,
      "eval_interval": 100,
      "learning_rate": 2.4e-05,
      "learning_rate_schedule_steps": 12000,
      "learning_rate_final_fraction": 0.0,
      "warmup_steps_fraction": 1 / 3000,  # 4 warmup steps out of 12000.
      "adam_b1": 0.9,
      "adam_b2": 0.95,
      "adam_eps": 1e-08,
      "adam_weight_decay": 0.1,
      "gradient_clipping_threshold": 1.0,
      "load_balance_loss_weight": 0.01,
      "target_eval_loss": 3.6,
      "quantization": "fp8_full",
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "ici_tensor_parallelism": 1,
      "dcn_tensor_parallelism": 1,
      "ici_tensor_transpose_parallelism": 1,
      "dcn_tensor_transpose_parallelism": 1,
      "ici_pipeline_parallelism": 1,
      "dcn_pipeline_parallelism": 1,
      "ici_context_parallelism": 1,
      "dcn_context_parallelism": 1,
      "ici_expert_parallelism": 8,
      "dcn_expert_parallelism": 1,
  }
  config.update(overrides)
  return types.SimpleNamespace(**config)


class MllogUtilsTest(unittest.TestCase):
  """Unit tests for mllog_utils."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())  # pylint: disable=consider-using-with

    self.mllogger = FakeMllogger()
    self.fake_mllog = types.SimpleNamespace(constants=_CONSTANTS, config=mock.Mock())

    patches = {
        "mllog": self.fake_mllog,
        "mllogger": self.mllogger,
        "_enabled": False,
        "_is_configured": False,
        "_destination_path": None,
        "_local_staging_file": None,
        "_run_stopped": False,
        "_open_block": None,
        "_last_sync_time": 0.0,
        "_upload_thread": None,
        "_gcs_client": None,
    }
    for name, value in patches.items():
      patcher = mock.patch.object(mllog_utils, name, value)
      patcher.start()
      self.addCleanup(patcher.stop)

    process_index = mock.patch.object(mllog_utils.jax, "process_index", return_value=0)
    process_index.start()
    self.addCleanup(process_index.stop)

  def setup_local(self, **overrides):
    """Configures mllog against a local file in the test's temp directory."""
    log_file = os.path.join(self.tmp_dir, "mllog_1234.log")
    config = make_config(mllog_file=log_file, **overrides)
    mllog_utils.setup_mllog(config)
    return config

  # --- setup_mllog -----------------------------------------------------------

  def test_setup_local_file(self):
    config = self.setup_local()
    self.assertTrue(mllog_utils._enabled)
    self.assertTrue(os.path.exists(config.mllog_file))
    self.assertIsNone(mllog_utils._local_staging_file)
    self.fake_mllog.config.assert_called_once_with(filename=config.mllog_file)

  def test_setup_gcs_destination_stages_locally(self):
    config = make_config(mllog_file="gs://a-bucket/runs/mllog_1234.log")
    mllog_utils.setup_mllog(config)

    self.assertTrue(mllog_utils._enabled)
    self.assertEqual(mllog_utils._destination_path, config.mllog_file)
    self.assertEqual(mllog_utils._local_staging_file, "/tmp/mllog_unit-test-run.log")
    self.fake_mllog.config.assert_called_once_with(filename="/tmp/mllog_unit-test-run.log")

  def test_setup_disabled_when_flag_off(self):
    mllog_utils.setup_mllog(make_config(enable_mllog=False))
    self.assertFalse(mllog_utils._enabled)
    self.fake_mllog.config.assert_not_called()

  def test_setup_disabled_when_no_file_configured(self):
    mllog_utils.setup_mllog(make_config(mllog_file=""))
    self.assertFalse(mllog_utils._enabled)
    self.fake_mllog.config.assert_not_called()

  def test_setup_disabled_when_file_not_writable(self):
    """An unwritable log path must disable logging rather than crash later."""
    unwritable = os.path.join(self.tmp_dir, "not-a-dir")
    with open(unwritable, "w", encoding="utf8"):
      pass

    mllog_utils.setup_mllog(make_config(mllog_file=os.path.join(unwritable, "mllog.log")))

    self.assertFalse(mllog_utils._enabled)
    self.fake_mllog.config.assert_not_called()
    # Subsequent calls must be harmless no-ops.
    mllog_utils.run_start()
    self.assertEqual(self.mllogger.events, [])

  def test_setup_is_idempotent(self):
    self.setup_local()
    self.setup_local()
    self.fake_mllog.config.assert_called_once()

  def test_non_zero_process_does_not_log(self):
    with mock.patch.object(mllog_utils.jax, "process_index", return_value=1):
      config = self.setup_local()
      mllog_utils.run_start()
      mllog_utils.block_start(config, step=0)
    self.assertEqual(self.mllogger.events, [])

  # --- hyperparameter events -------------------------------------------------

  def test_init_print_decay_steps_excludes_warmup(self):
    """The checker requires decay_steps == schedule_steps - warmup_steps."""
    config = self.setup_local()
    mllog_utils.init_print(config)

    warmup = self.mllogger.value_of(_CONSTANTS.OPT_LR_WARMUP_STEPS)
    decay = self.mllogger.value_of(_CONSTANTS.OPT_LR_DECAY_STEPS)
    self.assertEqual(warmup, 4)
    self.assertEqual(decay, config.learning_rate_schedule_steps - warmup)

  def test_init_print_emits_required_keys(self):
    config = self.setup_local()
    mllog_utils.init_print(config)

    keys = self.mllogger.keys()
    for required in (
        _CONSTANTS.SUBMISSION_ORG,
        _CONSTANTS.SUBMISSION_PLATFORM,
        _CONSTANTS.SUBMISSION_DIVISION,
        _CONSTANTS.SUBMISSION_BENCHMARK,
        _CONSTANTS.GLOBAL_BATCH_SIZE,
        _CONSTANTS.TRAIN_SAMPLES,
        _CONSTANTS.EVAL_SAMPLES,
        _CONSTANTS.OPT_NAME,
        _CONSTANTS.LOWEST_NUMERICAL_PRECISION_IN_LINEAR,
        _CONSTANTS.LOWEST_NUMERICAL_PRECISION_IN_ATTN,
        _CONSTANTS.LOWEST_NUMERICAL_PRECISION_IN_COMM,
        _CONSTANTS.TENSOR_PARALLELISM,
        _CONSTANTS.PIPELINE_PARALLELISM,
        _CONSTANTS.CONTEXT_PARALLELISM,
        _CONSTANTS.EXPERT_PARALLELISM,
        _CONSTANTS.MICRO_BATCH_SIZE,
        _CONSTANTS.CONFIG_FILENAME,
    ):
      self.assertIn(required, keys)
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.SEED), config.data_shuffle_seed)
    self.assertEqual(
        self.mllogger.value_of(_CONSTANTS.EVAL_SAMPLES),
        config.global_batch_size_to_eval_on * config.eval_steps,
    )
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.LOWEST_NUMERICAL_PRECISION_IN_LINEAR), "fp8")
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.LOWEST_NUMERICAL_PRECISION_IN_ATTN), "bfloat16")
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.LOWEST_NUMERICAL_PRECISION_IN_COMM), "bfloat16")
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.EXPERT_PARALLELISM), 8)
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.MICRO_BATCH_SIZE), 2)
    self.assertEqual(self.mllogger.value_of(_CONSTANTS.CONFIG_FILENAME), "config.yml")

  def test_disabled_config_emits_nothing(self):
    config = make_config(enable_mllog=False)
    mllog_utils.setup_mllog(config)

    mllog_utils.init_print(config)
    mllog_utils.init_stop()
    mllog_utils.run_start()
    mllog_utils.block_start(config)
    mllog_utils.eval_start(config, step=1)
    mllog_utils.eval_stop(config, step=1, eval_loss=9.0)
    mllog_utils.tracked_stats(config, step=1, step_time=0.5, loss=9.0)
    mllog_utils.validation_time(step=1, seconds=0.5)
    mllog_utils.run_stop()

    self.assertEqual(self.mllogger.events, [])

  # --- run lifecycle ---------------------------------------------------------

  def test_full_lifecycle_ordering(self):
    config = self.setup_local()

    mllog_utils.init_start(config)
    mllog_utils.init_print(config)
    mllog_utils.init_stop()
    mllog_utils.run_start()
    mllog_utils.block_start(config, step=0)
    # Non-eval step 1 closes block 0 and reopens block 1.
    mllog_utils.tracked_stats(config, step=1, step_time=0.5, loss=9.1)
    mllog_utils.step_end(config, step=1)
    # Eval step 2 closes block 1, runs eval, and reopens block 2.
    mllog_utils.tracked_stats(config, step=2, step_time=0.5, loss=9.0)
    mllog_utils.eval_start(config, step=2)
    mllog_utils.eval_stop(config, step=2, eval_loss=9.0)
    mllog_utils.run_stop(current_epoch_num=2 * config.global_batch_size_to_train_on)

    lifecycle = [key for key in self.mllogger.keys() if key in vars(_CONSTANTS).values()]
    self.assertEqual(
        [
            key
            for key in lifecycle
            if key
            in (
                _CONSTANTS.INIT_START,
                _CONSTANTS.INIT_STOP,
                _CONSTANTS.RUN_START,
                _CONSTANTS.BLOCK_START,
                _CONSTANTS.EVAL_START,
                _CONSTANTS.EVAL_STOP,
                _CONSTANTS.BLOCK_STOP,
                _CONSTANTS.RUN_STOP,
            )
        ],
        [
            _CONSTANTS.INIT_START,
            _CONSTANTS.INIT_STOP,
            _CONSTANTS.RUN_START,
            _CONSTANTS.BLOCK_START,
            _CONSTANTS.BLOCK_STOP,
            _CONSTANTS.BLOCK_START,
            _CONSTANTS.BLOCK_STOP,
            _CONSTANTS.EVAL_START,
            _CONSTANTS.EVAL_STOP,
            # eval_stop reopened a block because the target was not reached, so
            # run_stop has to close it.
            _CONSTANTS.BLOCK_START,
            _CONSTANTS.BLOCK_STOP,
            _CONSTANTS.RUN_STOP,
        ],
    )

  def test_step_end_does_not_reopen_block_on_final_step(self):
    config = self.setup_local(steps=2)
    mllog_utils.block_start(config, step=0)
    mllog_utils.step_end(config, step=1)
    mllog_utils.step_end(config, step=2)
    mllog_utils.run_stop(current_epoch_num=2 * config.global_batch_size_to_train_on, step=2)

    self.assertEqual(
        self.mllogger.keys(),
        [
            _CONSTANTS.BLOCK_START,
            _CONSTANTS.BLOCK_STOP,
            _CONSTANTS.BLOCK_START,
            _CONSTANTS.BLOCK_STOP,
            _CONSTANTS.RUN_STOP,
        ],
    )

  def test_run_stop_closes_block_when_run_has_no_eval(self):
    """block_stop is required even when training ends without an eval."""
    config = self.setup_local()
    mllog_utils.block_start(config, step=0)
    mllog_utils.run_stop(current_epoch_num=4096, step=6)

    keys = self.mllogger.keys()
    self.assertEqual(keys, [_CONSTANTS.BLOCK_START, _CONSTANTS.BLOCK_STOP, _CONSTANTS.RUN_STOP])
    # The closing block_stop reports the step training actually reached.
    self.assertEqual(
        self.mllogger.metadata_of(_CONSTANTS.BLOCK_STOP),
        {"samples_count": config.global_batch_size_to_train_on, "step": 6},
    )

  def test_run_stop_does_not_duplicate_block_stop(self):
    """eval_start already closed the block, so run_stop must not close it again."""
    config = self.setup_local()
    mllog_utils.block_start(config, step=0)
    mllog_utils.eval_start(config, step=100)
    mllog_utils.eval_stop(config, step=100, eval_loss=1.0)  # Below target: stops the run.

    self.assertEqual(self.mllogger.keys().count(_CONSTANTS.BLOCK_STOP), 1)
    self.assertEqual(self.mllogger.keys().count(_CONSTANTS.RUN_STOP), 1)

  def test_run_stop_is_logged_once(self):
    config = self.setup_local()
    mllog_utils.block_start(config, step=0)
    mllog_utils.run_stop(current_epoch_num=4096)
    mllog_utils.run_stop(current_epoch_num=8192)

    self.assertEqual(self.mllogger.keys().count(_CONSTANTS.RUN_STOP), 1)
    self.assertEqual(self.mllogger.metadata_of(_CONSTANTS.RUN_STOP), {"status": "success", "samples_count": 4096})

  def test_eval_stop_below_target_stops_run(self):
    config = self.setup_local()
    mllog_utils.eval_stop(config, step=100, eval_loss=config.target_eval_loss - 0.1)

    keys = self.mllogger.keys()
    self.assertIn(_CONSTANTS.RUN_STOP, keys)
    self.assertNotIn(_CONSTANTS.BLOCK_START, keys)
    self.assertEqual(
        self.mllogger.metadata_of(_CONSTANTS.EVAL_ACCURACY),
        {"samples_count": 100 * config.global_batch_size_to_train_on},
    )

  def test_eval_stop_above_target_opens_next_block(self):
    config = self.setup_local()
    mllog_utils.eval_stop(config, step=100, eval_loss=config.target_eval_loss + 0.1)

    keys = self.mllogger.keys()
    self.assertIn(_CONSTANTS.BLOCK_START, keys)
    self.assertNotIn(_CONSTANTS.RUN_STOP, keys)

  def test_eval_stop_on_final_step_aborts_without_opening_a_block(self):
    """Matches the reference: a last-step eval that misses the target ends the run."""
    config = self.setup_local(steps=100)
    mllog_utils.eval_stop(config, step=100, eval_loss=config.target_eval_loss + 0.1)

    keys = self.mllogger.keys()
    self.assertNotIn(_CONSTANTS.BLOCK_START, keys)
    self.assertEqual(
        self.mllogger.metadata_of(_CONSTANTS.RUN_STOP),
        {"status": "aborted", "samples_count": 100 * config.global_batch_size_to_train_on},
    )

  def test_samples_count_is_relative_to_start_step(self):
    """Resumed runs must count samples from the step they restarted at."""
    config = self.setup_local()
    mllog_utils.eval_start(config, step=150, start_step=100)

    self.assertEqual(
        self.mllogger.metadata_of(_CONSTANTS.EVAL_START),
        {"samples_count": 50 * config.global_batch_size_to_train_on, "step": 150},
    )

  def test_tracked_stats_payload(self):
    config = self.setup_local()
    mllog_utils.tracked_stats(config, step=10, step_time=0.25, loss=7.5, start_step=5)

    self.assertEqual(
        self.mllogger.value_of("tracked_stats"),
        {"reduced_train_loss": 7.5, "train_step_time": 0.25},
    )
    self.assertEqual(
        self.mllogger.metadata_of("tracked_stats"),
        {"samples_count": 5 * config.global_batch_size_to_train_on},
    )

  def test_validation_time_is_keyed_by_step(self):
    """The reference keys this tracked_stats variant by step, not samples_count."""
    self.setup_local()
    mllog_utils.validation_time(step=50, seconds=1.31)

    self.assertEqual(self.mllogger.value_of("tracked_stats"), {"validation_time": 1.31})
    self.assertEqual(self.mllogger.metadata_of("tracked_stats"), {"step": 50})

  # --- GCS sync --------------------------------------------------------------

  def test_sync_is_noop_for_local_destinations(self):
    self.setup_local()
    with mock.patch.object(mllog_utils, "_upload_file_to_gcs") as upload:
      mllog_utils.sync_log(force=True)
    upload.assert_not_called()

  def test_sync_is_throttled_and_non_blocking(self):
    mllog_utils.setup_mllog(make_config(mllog_file="gs://a-bucket/mllog_1234.log"))

    with mock.patch.object(mllog_utils, "_upload_file_to_gcs") as upload:
      mllog_utils.sync_log()
      mllog_utils.sync_log()  # Within the throttling window: skipped.
      if mllog_utils._upload_thread is not None:
        mllog_utils._upload_thread.join()
      self.assertEqual(upload.call_count, 1)
      self.assertTrue(mllog_utils._upload_thread.daemon)

      # A forced sync bypasses the throttle and uploads inline.
      mllog_utils.sync_log(force=True)
      self.assertEqual(upload.call_count, 2)

  def test_forced_sync_uploads_inline(self):
    """The forced path must not depend on a background thread."""
    mllog_utils.setup_mllog(make_config(mllog_file="gs://a-bucket/mllog_1234.log"))

    with mock.patch.object(mllog_utils, "_upload_file_to_gcs") as upload:
      with mock.patch.object(mllog_utils.threading, "Thread") as thread:
        mllog_utils.sync_log(force=True)
      thread.assert_not_called()
      upload.assert_called_once_with("gs://a-bucket/mllog_1234.log", "/tmp/mllog_unit-test-run.log")

  def test_gcs_destination_registers_atexit_sync(self):
    """An early crash must still land the staged log in GCS."""
    with mock.patch.object(mllog_utils.atexit, "register") as register:
      mllog_utils.setup_mllog(make_config(mllog_file="gs://a-bucket/mllog_1234.log"))
    register.assert_called_once_with(mllog_utils.sync_log, force=True)

  def test_local_destination_does_not_register_atexit(self):
    with mock.patch.object(mllog_utils.atexit, "register") as register:
      self.setup_local()
    register.assert_not_called()

  def test_upload_reuses_cached_client(self):
    storage = mock.Mock()
    with mock.patch.dict("sys.modules", {"google.cloud": mock.Mock(storage=storage)}):
      src = os.path.join(self.tmp_dir, "staging.log")
      with open(src, "w", encoding="utf8") as f:
        f.write("line\n")

      mllog_utils._upload_file_to_gcs("gs://a-bucket/runs/mllog.log", src)
      mllog_utils._upload_file_to_gcs("gs://a-bucket/runs/mllog.log", src)

    storage.Client.assert_called_once()
    storage.Client.return_value.bucket.assert_called_with("a-bucket")
    storage.Client.return_value.bucket.return_value.blob.assert_called_with("runs/mllog.log")

  def test_upload_skips_missing_staging_file(self):
    storage = mock.Mock()
    with mock.patch.dict("sys.modules", {"google.cloud": mock.Mock(storage=storage)}):
      mllog_utils._upload_file_to_gcs("gs://a-bucket/mllog.log", os.path.join(self.tmp_dir, "absent.log"))
    storage.Client.assert_not_called()


if __name__ == "__main__":
  unittest.main()
