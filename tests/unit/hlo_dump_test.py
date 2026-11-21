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

"""Unit tests for fault-tolerant HLO dumping and GCS upload cleanup."""

import contextlib
import os
import shutil
import tempfile
import types as pytypes
import unittest
from unittest import mock

from maxtext.trainers.pre_train import train
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import gcs_utils


def _make_fake_storage_client(uploaded_blobs, fail_on_filename=None):
  """Creates a fake google.cloud.storage.Client capturing uploaded blobs."""

  class _FakeBlob:

    def __init__(self, name):
      self.name = name

    def upload_from_filename(self, filename):
      if fail_on_filename and os.path.basename(filename) == fail_on_filename:
        raise RuntimeError(f"Simulated upload failure for {filename}")
      with open(filename, "rb") as f:
        uploaded_blobs[self.name] = f.read()

  class _FakeBucket:

    def blob(self, blob_name):
      return _FakeBlob(blob_name)

  class _FakeClient:

    def __init__(self):
      self.requested_buckets = []

    def get_bucket(self, bucket_name):
      self.requested_buckets.append(bucket_name)
      return _FakeBucket()

  return _FakeClient


def _make_config(local_dir, gcs_dir="gs://test-bucket/run/xla_dump/", dump_hlo=True, **overrides):
  """Constructs a minimal config namespace for HLO dump tests."""
  defaults = {
      "dump_hlo": dump_hlo,
      "dump_step": -1,
      "dump_hlo_local_dir": local_dir,
      "dump_hlo_gcs_dir": gcs_dir,
      "dump_hlo_module_name": "",
      "dump_hlo_delete_local_after": True,
      "dump_hlo_upload_all": False,
      "steps": 1,
      "eval_interval": 0,
      "eval_steps": 0,
      "eval_start_step": -1,
      "eval_cache_prefill_in_background": False,
      "save_checkpoint_on_completion": False,
      "enable_diloco": False,
      "moe_dropless_fallback": "none",
      "shard_optimizer_over_data": False,
      "shard_mode": "auto",
      "compiled_trainstep_file": "",
      "compile_xla_flags": "",
      "logical_axis_rules": (),
      "logical_axis_rules_for_eval": (),
      "warmup_programs_in_init": False,
      "dcn_bandwidth_limit": "",
      "dcn_network_interface": "eth0",
      "dcn_throttle_interval_seconds": 0,
      "prng_impl": "unsafe_rbg",
      "debug_sharding": False,
      "elastic_enabled": False,
      "dataset_type": "synthetic",
      "logical_global_batch_size_to_train_on": 0,
      "global_batch_size_to_train_on": 1,
      "record_goodput": False,
      "enable_goodput_recording": False,
      "checkpoint_is_quantized": False,
      "retry_dropless_first_steps": 0,
      "first_phase_ragged_buffer_factor": 0.0,
      "record_required_ragged_buffer_factor": False,
      "max_checkify_steps": 0,
      "profiler": "",
      "save_config_to_gcs": False,
      "base_output_directory": "/tmp/out",
      "run_name": "test_run",
      "te_moe_overflow_stopping_enabled": False,
  }
  defaults.update(overrides)
  return pytypes.SimpleNamespace(**defaults)


class GcsUtilsUploadDumpTest(unittest.TestCase):
  """Tests fault tolerance of gcs_utils.upload_dump and upload_hlo_dump."""

  def test_upload_dump_skips_missing_local_dir(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      missing_dir = os.path.join(tmpdir, "does_not_exist")
      with mock.patch.object(gcs_utils.storage, "Client") as mock_client:
        gcs_utils.upload_dump(missing_dir, "gs://bucket/prefix", delete_local_after=True)
      mock_client.assert_not_called()

  def test_upload_dump_skips_non_gcs_target_and_keeps_local_dir(self):
    with tempfile.TemporaryDirectory() as local_dir:
      dump_file = os.path.join(local_dir, "module.txt")
      with open(dump_file, "w", encoding="utf-8") as f:
        f.write("HloModule test")

      for target in ("/tmp/local_out/xla_dump", "", None, "gs:///missing_bucket"):
        with self.subTest(target=target):
          with mock.patch.object(gcs_utils.storage, "Client") as mock_client:
            gcs_utils.upload_dump(local_dir, target, delete_local_after=True)
          mock_client.assert_not_called()
          self.assertTrue(os.path.exists(dump_file))

  def test_upload_dump_skips_when_no_files_match_and_keeps_local_dir(self):
    with tempfile.TemporaryDirectory() as local_dir:
      other_file = os.path.join(local_dir, "other.txt")
      with open(other_file, "w", encoding="utf-8") as f:
        f.write("data")

      with mock.patch.object(gcs_utils.storage, "Client") as mock_client:
        gcs_utils.upload_dump(local_dir, "gs://bucket/prefix", module_name="jit_train_step", delete_local_after=True)
      mock_client.assert_not_called()
      self.assertTrue(os.path.exists(other_file))

  def test_upload_dump_handles_gcs_client_and_bucket_errors(self):
    with tempfile.TemporaryDirectory() as local_dir:
      dump_file = os.path.join(local_dir, "module.txt")
      with open(dump_file, "w", encoding="utf-8") as f:
        f.write("HloModule test")

      with mock.patch.object(gcs_utils.storage, "Client", side_effect=RuntimeError("no credentials")):
        gcs_utils.upload_dump(local_dir, "gs://bucket/prefix", delete_local_after=True)
      self.assertTrue(os.path.exists(dump_file))

      fake_client = mock.MagicMock()
      fake_client.get_bucket.side_effect = RuntimeError("bucket missing")
      with mock.patch.object(gcs_utils.storage, "Client", return_value=fake_client):
        gcs_utils.upload_dump(local_dir, "gs://bucket/prefix", delete_local_after=True)
      self.assertTrue(os.path.exists(dump_file))

  def test_upload_dump_continues_on_file_error_and_preserves_local_dir(self):
    with tempfile.TemporaryDirectory() as local_dir:
      file_a = os.path.join(local_dir, "a_bad.txt")
      file_b = os.path.join(local_dir, "b_good.txt")
      with open(file_a, "w", encoding="utf-8") as f:
        f.write("bad")
      with open(file_b, "w", encoding="utf-8") as f:
        f.write("good")

      uploaded = {}
      fake_cls = _make_fake_storage_client(uploaded, fail_on_filename="a_bad.txt")
      with (
          mock.patch.object(gcs_utils.jax, "process_index", return_value=0),
          mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
          mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
      ):
        gcs_utils.upload_dump(local_dir, "gs://bucket/run/xla_dump/", delete_local_after=True)

      self.assertEqual(uploaded, {"run/xla_dump/b_good.txt": b"good"})
      self.assertTrue(os.path.exists(file_a))
      self.assertTrue(os.path.exists(file_b))

  def test_upload_dump_success_filters_uploads_and_deletes_local_dir(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    os.makedirs(os.path.join(local_dir, "nested"))
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    with open(os.path.join(local_dir, "jit_train_step_1.txt"), "w", encoding="utf-8") as f:
      f.write("hlo1")
    with open(os.path.join(local_dir, "nested", "jit_train_step_2.txt"), "w", encoding="utf-8") as f:
      f.write("hlo2")
    with open(os.path.join(local_dir, "unrelated.txt"), "w", encoding="utf-8") as f:
      f.write("ignore")

    uploaded = {}
    fake_cls = _make_fake_storage_client(uploaded)
    with (
        mock.patch.object(gcs_utils.jax, "process_index", return_value=0),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
    ):
      gcs_utils.upload_dump(
          local_dir,
          "gs://bucket/run/xla_dump/",
          module_name="jit_train_step",
          delete_local_after=True,
      )

    self.assertEqual(
        uploaded,
        {
            "run/xla_dump/jit_train_step_1.txt": b"hlo1",
            "run/xla_dump/nested/jit_train_step_2.txt": b"hlo2",
        },
    )
    self.assertFalse(os.path.exists(local_dir))

  def test_upload_dump_process_index_and_all_host_upload(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    os.makedirs(local_dir)
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    with open(os.path.join(local_dir, "m.hlo"), "w", encoding="utf-8") as f:
      f.write("hlo")

    uploaded = {}
    fake_cls = _make_fake_storage_client(uploaded)
    with (
        mock.patch.object(gcs_utils.jax, "process_index", return_value=1),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
    ):
      gcs_utils.upload_dump(local_dir, "gs://bucket/prefix/", all_host_upload=False)
      self.assertEqual(uploaded, {})
      self.assertTrue(os.path.exists(local_dir))

      with mock.patch.object(gcs_utils.socket, "gethostname", return_value="host-1"):
        gcs_utils.upload_dump(local_dir, "gs://bucket/prefix/", all_host_upload=True, delete_local_after=True)
      self.assertEqual(uploaded, {"prefix/host-1/m.hlo": b"hlo"})
      self.assertFalse(os.path.exists(local_dir))

  def test_upload_hlo_dump_config_helper(self):
    cfg_disabled = _make_config("/tmp/xla", dump_hlo=False)
    with mock.patch.object(gcs_utils, "upload_dump") as mock_upload:
      gcs_utils.upload_hlo_dump(cfg_disabled)
    mock_upload.assert_not_called()

    cfg_enabled = _make_config("/tmp/xla", "gs://b/p/", dump_hlo=True, dump_hlo_module_name="mod")
    with mock.patch.object(gcs_utils, "upload_dump") as mock_upload:
      gcs_utils.upload_hlo_dump(cfg_enabled)
    mock_upload.assert_called_once_with(
        "/tmp/xla",
        "gs://b/p/",
        module_name="mod",
        delete_local_after=True,
        all_host_upload=False,
    )

    with mock.patch.object(gcs_utils, "upload_dump", side_effect=RuntimeError("boom")):
      gcs_utils.upload_hlo_dump(cfg_enabled)


class TrainAndCompileHloFaultToleranceTest(unittest.TestCase):
  """Tests HLO dump fault tolerance across train.py and train_compile.py crashes."""

  def _patch_train_loop_setup(self, p_train_step, aot_side_effect=None):
    """Patches heavy setup/JAX helpers so train.train_loop executes hermetically."""
    fake_model = mock.MagicMock()
    fake_state = mock.MagicMock()
    fake_mesh = mock.MagicMock()
    fake_data_loader = mock.MagicMock()
    fake_data_loader.load_next_batch.return_value = {"inputs": [1]}
    fake_checkpoint_manager = mock.MagicMock()
    fake_lr = mock.MagicMock()

    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.object(
            train.train_utils,
            "setup_train_loop",
            return_value=(
                "init_rng",
                fake_checkpoint_manager,
                mock.MagicMock(),
                fake_model,
                fake_mesh,
                fake_lr,
                "data_iter",
                fake_data_loader,
                "rampup",
                None,
                fake_state,
            ),
        )
    )
    stack.enter_context(mock.patch.object(train.train_utils, "maybe_apply_dcn_throttling"))
    stack.enter_context(mock.patch.object(train.train_utils, "validate_completed_steps"))
    stack.enter_context(mock.patch.object(train.train_utils, "prepare_before_run_start"))
    stack.enter_context(mock.patch.object(train.train_utils, "maybe_cleanup_dcn_throttling"))
    stack.enter_context(
        mock.patch.object(train.train_utils, "jit_train_and_eval_step", return_value=(p_train_step, None))
    )
    stack.enter_context(mock.patch.object(train, "get_first_step", return_value=0))
    stack.enter_context(
        mock.patch.object(
            train.nnx,
            "split",
            side_effect=lambda *args: (fake_model, fake_state) if len(args) == 1 else (None, {}, None),
        )
    )
    stack.enter_context(mock.patch.object(train, "build_eval_graphdef", return_value=None))
    stack.enter_context(
        mock.patch.object(
            train.sharding,
            "maybe_update_params_sharding_with_opt",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        )
    )
    stack.enter_context(mock.patch.object(train.sharding, "get_input_data_sharding", return_value=mock.MagicMock()))
    stack.enter_context(mock.patch.object(train.maxtext_utils, "get_shaped_batch", return_value={}))
    stack.enter_context(mock.patch.object(train.maxtext_utils, "maybe_dump_jaxpr"))
    stack.enter_context(mock.patch.object(train.max_utils, "parse_libtpu_flags_to_dict", return_value={}))
    stack.enter_context(mock.patch.object(train.max_utils, "print_mem_stats"))
    stack.enter_context(mock.patch.object(train.profiler, "Profiler", return_value=mock.MagicMock()))
    stack.enter_context(mock.patch.object(train.metric_logger, "MetricLogger", return_value=mock.MagicMock()))
    stack.enter_context(mock.patch.object(train.elastic_utils, "record_elastic_reinit_end"))
    stack.enter_context(mock.patch.object(train, "mllog_utils"))
    stack.enter_context(mock.patch.object(train, "start_run_clock"))
    stack.enter_context(mock.patch.object(train, "record_goodput"))
    stack.enter_context(mock.patch.object(train, "maybe_record_goodput", return_value=contextlib.nullcontext()))
    stack.enter_context(mock.patch.object(train.jax, "set_mesh", return_value=contextlib.nullcontext()))
    stack.enter_context(mock.patch.object(train, "logical_axis_rules", return_value=contextlib.nullcontext()))
    stack.enter_context(mock.patch.object(train.checkpointing, "maybe_save_checkpoint"))
    stack.enter_context(mock.patch.object(train.checkpointing, "wait_until_finished"))
    stack.enter_context(mock.patch.object(train.snapshotting, "maybe_save_snapshot"))
    if aot_side_effect is not None:
      stack.enter_context(mock.patch.object(train, "aot_compile_step", side_effect=aot_side_effect))
    else:
      stack.enter_context(mock.patch.object(train, "aot_compile_step"))
    return stack

  def test_train_loop_uploads_hlo_when_aot_compile_raises(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    config = _make_config(local_dir, "gs://test-bucket/run/xla_dump/", dump_hlo=True)
    uploaded = {}
    fake_cls = _make_fake_storage_client(uploaded)

    def _crashing_aot_compile(*args, **kwargs):
      del args, kwargs
      os.makedirs(local_dir, exist_ok=True)
      with open(os.path.join(local_dir, "jit_train_step.before_opt.txt"), "w", encoding="utf-8") as f:
        f.write("partial hlo on compile crash")
      raise RuntimeError("XLA compilation OOM")

    with (
        self._patch_train_loop_setup(mock.MagicMock(), aot_side_effect=_crashing_aot_compile),
        mock.patch.object(gcs_utils.jax, "process_index", return_value=0),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
    ):
      with self.assertRaisesRegex(RuntimeError, "XLA compilation OOM"):
        train.train_loop(config, recorder=None)

    self.assertEqual(uploaded, {"run/xla_dump/jit_train_step.before_opt.txt": b"partial hlo on compile crash"})
    self.assertFalse(os.path.exists(local_dir))

  def test_train_loop_uploads_hlo_when_step0_raises_before_upload(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    os.makedirs(local_dir)
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
    with open(os.path.join(local_dir, "step0.hlo"), "w", encoding="utf-8") as f:
      f.write("step0 hlo")

    config = _make_config(local_dir, "gs://test-bucket/run/xla_dump/", dump_hlo=True)
    uploaded = {}
    fake_cls = _make_fake_storage_client(uploaded)
    p_train_step = mock.MagicMock(side_effect=RuntimeError("step 0 execution crash"))

    with (
        self._patch_train_loop_setup(p_train_step),
        mock.patch.object(gcs_utils.jax, "process_index", return_value=0),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
    ):
      with self.assertRaisesRegex(RuntimeError, "step 0 execution crash"):
        train.train_loop(config, recorder=None)

    self.assertEqual(uploaded, {"run/xla_dump/step0.hlo": b"step0 hlo"})

  def test_train_loop_uploads_hlo_when_block_until_ready_raises_at_step0(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    os.makedirs(local_dir)
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
    with open(os.path.join(local_dir, "async.hlo"), "w", encoding="utf-8") as f:
      f.write("async crash hlo")

    config = _make_config(local_dir, "gs://test-bucket/run/xla_dump/", dump_hlo=True)
    uploaded = {}
    fake_cls = _make_fake_storage_client(uploaded)
    p_train_step = mock.MagicMock(return_value=(mock.MagicMock(), {"learning/loss": 1.0}))

    with (
        self._patch_train_loop_setup(p_train_step),
        mock.patch.object(train.jax, "block_until_ready", side_effect=RuntimeError("async XLA failure")),
        mock.patch.object(gcs_utils.jax, "process_index", return_value=0),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
    ):
      with self.assertRaisesRegex(RuntimeError, "async XLA failure"):
        train.train_loop(config, recorder=None)

    self.assertEqual(uploaded, {"run/xla_dump/async.hlo": b"async crash hlo"})

  def test_train_loop_does_not_double_upload_after_step0_success(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    os.makedirs(local_dir)
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
    with open(os.path.join(local_dir, "ok.hlo"), "w", encoding="utf-8") as f:
      f.write("ok hlo")

    config = _make_config(local_dir, "gs://test-bucket/run/xla_dump/", dump_hlo=True, dump_hlo_delete_local_after=False)
    p_train_step = mock.MagicMock(return_value=(mock.MagicMock(), {"learning/loss": 1.0}))

    with (
        self._patch_train_loop_setup(p_train_step),
        mock.patch.object(train.jax, "block_until_ready"),
        mock.patch.object(gcs_utils, "upload_dump") as mock_upload_dump,
    ):
      train.train_loop(config, recorder=None)

    mock_upload_dump.assert_called_once_with(
        local_dir,
        "gs://test-bucket/run/xla_dump/",
        module_name="",
        delete_local_after=False,
        all_host_upload=False,
    )

  def test_train_loop_preserves_original_error_when_upload_fails(self):
    config = _make_config("/tmp/missing_xla", dump_hlo=True)
    with (
        self._patch_train_loop_setup(mock.MagicMock(), aot_side_effect=ValueError("original compile error")),
        mock.patch.object(gcs_utils, "upload_dump", side_effect=RuntimeError("GCS unavailable")),
    ):
      with self.assertRaisesRegex(ValueError, "original compile error"):
        train.train_loop(config, recorder=None)

  def test_train_compile_uploads_hlo_when_compile_raises(self):
    tmpdir = tempfile.mkdtemp()
    local_dir = os.path.join(tmpdir, "xla_dump")
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    config = _make_config(local_dir, "gs://test-bucket/compile/xla_dump/", dump_hlo=True)
    uploaded = {}
    fake_cls = _make_fake_storage_client(uploaded)

    def _crashing_jit_and_compile(*args, **kwargs):
      del args, kwargs
      os.makedirs(local_dir, exist_ok=True)
      with open(os.path.join(local_dir, "aot_crash.hlo"), "w", encoding="utf-8") as f:
        f.write("aot crash hlo")
      raise RuntimeError("AOT compile crash")

    with (
        mock.patch.dict(os.environ, {}, clear=False),
        mock.patch.object(train_compile.pyconfig, "initialize", return_value=config),
        mock.patch.object(train_compile, "validate_config"),
        mock.patch.object(train_compile, "get_topology_mesh", return_value=mock.MagicMock()),
        mock.patch.object(train_compile.max_utils, "print_system_information"),
        mock.patch.object(
            train_compile,
            "get_shaped_inputs",
            return_value=(("arg",), {}, mock.MagicMock(), mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(
            train_compile.sharding,
            "maybe_update_params_sharding_with_opt",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(
            train_compile.sharding, "build_zero1_input_state_mesh_shardings", return_value=mock.MagicMock()
        ),
        mock.patch.object(train_compile.sharding, "get_input_data_sharding", return_value=mock.MagicMock()),
        mock.patch.object(
            train_compile.maxtext_utils,
            "get_functional_train_with_signature",
            return_value=(mock.MagicMock(), (), (), (), 0),
        ),
        mock.patch.object(train_compile, "axis_rules", return_value=contextlib.nullcontext()),
        mock.patch.object(train_compile, "jit_and_compile", side_effect=_crashing_jit_and_compile),
        mock.patch.object(gcs_utils.jax, "process_index", return_value=0),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils.storage, "Client", side_effect=fake_cls),
    ):
      with self.assertRaisesRegex(RuntimeError, "AOT compile crash"):
        train_compile.main(["train_compile.py", "base.yml"])

    self.assertEqual(uploaded, {"compile/xla_dump/aot_crash.hlo": b"aot crash hlo"})

  def test_train_compile_skips_upload_when_dump_hlo_false(self):
    config = _make_config("/tmp/xla_dump", dump_hlo=False)
    with (
        mock.patch.dict(os.environ, {}, clear=False),
        mock.patch.object(train_compile.pyconfig, "initialize", return_value=config),
        mock.patch.object(train_compile, "validate_config"),
        mock.patch.object(train_compile, "get_topology_mesh", return_value=mock.MagicMock()),
        mock.patch.object(train_compile.max_utils, "print_system_information"),
        mock.patch.object(
            train_compile,
            "get_shaped_inputs",
            return_value=(("arg",), {}, mock.MagicMock(), mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(
            train_compile.sharding,
            "maybe_update_params_sharding_with_opt",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(
            train_compile.sharding, "build_zero1_input_state_mesh_shardings", return_value=mock.MagicMock()
        ),
        mock.patch.object(train_compile.sharding, "get_input_data_sharding", return_value=mock.MagicMock()),
        mock.patch.object(
            train_compile.maxtext_utils,
            "get_functional_train_with_signature",
            return_value=(mock.MagicMock(), (), (), (), 0),
        ),
        mock.patch.object(train_compile, "axis_rules", return_value=contextlib.nullcontext()),
        mock.patch.object(train_compile, "jit_and_compile", side_effect=RuntimeError("compile error")),
        mock.patch.object(gcs_utils, "upload_hlo_dump") as mock_upload,
    ):
      with self.assertRaisesRegex(RuntimeError, "compile error"):
        train_compile.main(["train_compile.py", "base.yml"])

    mock_upload.assert_not_called()


if __name__ == "__main__":
  unittest.main()
