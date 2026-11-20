# Copyright 2023–2026 Google LLC
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

"""Tests for automatic pipeline-parallel profiling, SPMD barriers, staging layout, and metadata."""

import json
import os
import sys
import tempfile
import types as pytypes
import unittest
from unittest.mock import patch

import numpy as np

from maxtext.common import profiler
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.utils import pipeline_trace_lens
from tests.utils.test_helpers import get_test_config_path


def _fake_device(device_id, process_index, coords=None, core_on_chip=0, device_kind="TPU fake"):
  return pytypes.SimpleNamespace(
      id=device_id,
      process_index=process_index,
      coords=coords if coords is not None else [device_id, 0, 0],
      core_on_chip=core_on_chip,
      device_kind=device_kind,
  )


def _fake_mesh(axis_names, device_grid):
  """Builds a mesh stand-in (axis_names + object ndarray of devices) without touching real hardware."""
  grid = np.empty(np.shape(device_grid)[: len(axis_names)], dtype=object)
  for idx in np.ndindex(*grid.shape):
    sub = device_grid
    for i in idx:
      sub = sub[i]
    grid[idx] = sub
  return pytypes.SimpleNamespace(axis_names=tuple(axis_names), devices=grid)


def _config(**overrides):
  """Minimal config with every field Profiler and build_pipeline_trace_metadata read."""
  cfg = {
      "profiler": "xplane",
      "tensorboard_dir": "/tmp/unused",
      "upload_all_profiler_results": False,
      "profile_cleanly": False,
      "profile_periodically_period": -1,
      "skip_first_n_steps_for_profiler": 3,
      "profiler_steps": 2,
      "steps": 20,
      "managed_mldiagnostics": False,
      "profile_power_events": False,
      "enable_tpu_profiling_options": False,
      "tpu_num_chips_to_profile_per_task": 1,
      "tpu_num_sparse_core_tiles_to_trace": 1,
      "tpu_num_sparse_cores_to_trace": 2,
      "enable_continuous_profiling": False,
      "enable_pipeline_stage_profiling": True,
      "using_pipeline_parallelism": True,
      "ici_pipeline_parallelism": 4,
      "dcn_pipeline_parallelism": 1,
      "num_pipeline_microbatches": 8,
      "num_pipeline_repeats": 1,
      "num_layers_per_pipeline_stage": 2,
      "pipeline_delay_activation_forwarding": True,
      "scan_pipeline_iterations": True,
      "pipeline_fsdp_ag_once": False,
  }
  cfg.update(overrides)
  return pytypes.SimpleNamespace(**cfg)


def _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=1):
  """Mesh axes (data, stage, fsdp); stage s owns processes [s*processes_per_stage, (s+1)*processes_per_stage)."""
  devices_per_process = devices_per_stage // processes_per_stage
  grid = [[[None] * devices_per_stage for _ in range(num_stages)]]
  for s in range(num_stages):
    for f in range(devices_per_stage):
      device_id = s * devices_per_stage + f
      grid[0][s][f] = _fake_device(device_id, s * processes_per_stage + f // devices_per_process)
  return _fake_mesh(("data", "stage", "fsdp"), grid)


def _v6e4_mesh():
  """Single-process PP=4 mesh whose stage order is the ring [0, 1, 3, 2], not device order."""
  devices = {i: _fake_device(i, 0, coords=[i % 2, i // 2, 0]) for i in range(4)}
  grid = [[[devices[0]], [devices[1]], [devices[3]], [devices[2]]]]
  return _fake_mesh(("data", "stage", "fsdp"), grid), [devices[i] for i in range(4)]


class StageProcessSelectionTest(unittest.TestCase):
  """Automatic selection of one profiled process per pipeline stage."""

  def test_one_process_per_stage(self):
    mesh = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=1)
    self.assertEqual(profiler.get_pipeline_stage_profile_processes(mesh), [0, 1, 2, 3])

  def test_first_device_owner_when_stage_spans_processes(self):
    mesh = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=2)
    self.assertEqual(profiler.get_pipeline_stage_profile_processes(mesh), [0, 2, 4, 6])

  def test_stage_axis_not_outermost(self):
    devices = [[_fake_device(s * 2 + d, s) for s in range(2)] for d in range(2)]
    mesh = _fake_mesh(("fsdp", "stage"), devices)
    self.assertEqual(profiler.get_pipeline_stage_profile_processes(mesh), [0, 1])

  def test_no_pipeline_returns_none(self):
    self.assertIsNone(profiler.get_pipeline_stage_profile_processes(None))
    no_stage = _fake_mesh(("data", "fsdp"), [[_fake_device(0, 0), _fake_device(1, 0)]])
    self.assertIsNone(profiler.get_pipeline_stage_profile_processes(no_stage))
    one_stage = _multihost_mesh(num_stages=1)
    self.assertIsNone(profiler.get_pipeline_stage_profile_processes(one_stage))

  def test_default_on_for_pipeline_parallel_xplane(self):
    mesh = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=2)
    prof = profiler.Profiler(_config(), mesh=mesh)
    self.assertTrue(prof.pipeline_stage_profiling_active)
    self.assertEqual(prof.profile_process_indices, [0, 2, 4, 6])
    for process_index in range(8):
      with patch("jax.process_index", return_value=process_index):
        self.assertEqual(prof._should_profile(), process_index in (0, 2, 4, 6))  # pylint: disable=protected-access
    self.assertEqual(prof._managed_process_index_list(), [0, 2, 4, 6])  # pylint: disable=protected-access

  def test_disable_flag_restores_legacy_process0_only(self):
    mesh = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=2)
    prof = profiler.Profiler(_config(enable_pipeline_stage_profiling=False), mesh=mesh)
    self.assertFalse(prof.pipeline_stage_profiling_active)
    self.assertIsNone(prof.profile_process_indices)
    for process_index, expected in ((0, True), (1, False), (2, False), (6, False)):
      with patch("jax.process_index", return_value=process_index):
        self.assertEqual(prof._should_profile(), expected)  # pylint: disable=protected-access

  def test_upload_all_overrides_stage_selection(self):
    mesh = _multihost_mesh()
    prof = profiler.Profiler(_config(upload_all_profiler_results=True), mesh=mesh)
    self.assertTrue(prof.pipeline_stage_profiling_active)
    with patch("jax.process_index", return_value=5):
      self.assertTrue(prof._should_profile())  # pylint: disable=protected-access
    self.assertIsNone(prof._managed_process_index_list())  # pylint: disable=protected-access

  def test_inactive_without_stage_axis_or_pp(self):
    mesh = _fake_mesh(("data", "fsdp"), [[_fake_device(0, 0), _fake_device(1, 1)]])
    prof = profiler.Profiler(_config(using_pipeline_parallelism=False), mesh=mesh)
    self.assertFalse(prof.pipeline_stage_profiling_active)
    self.assertIsNone(prof.profile_process_indices)
    with patch("jax.process_index", return_value=1):
      self.assertFalse(prof._should_profile())  # pylint: disable=protected-access


class StageLayoutAndSpmdBarrierTest(unittest.TestCase):
  """Collision-free stage trace staging, SPMD collectives, and in-place augmentation hooks."""

  def test_profile_output_path_staging(self):
    self.assertEqual(profiler.get_profile_output_path("/tb", 0, True), "/tb")
    self.assertEqual(
        profiler.get_profile_output_path("/tb", 2, True),
        f"/tb/{pipeline_trace_lens.STAGE_TRACES_SUBDIR}/process_00002",
    )
    self.assertEqual(profiler.get_profile_output_path("/tb", 2, False), "/tb")
    self.assertEqual(
        profiler.get_profile_output_path("gs://bucket/run", 3, True),
        f"gs://bucket/run/{pipeline_trace_lens.STAGE_TRACES_SUBDIR}/process_00003",
    )

  @patch.object(pipeline_trace_lens, "augment_host0_xplane_in_place")
  @patch.object(pipeline_trace_lens, "find_host0_xplane")
  @patch("jax.experimental.multihost_utils.sync_global_devices")
  @patch("jax.experimental.multihost_utils.broadcast_one_to_all")
  @patch("jax.profiler.stop_trace")
  @patch("jax.profiler.start_trace")
  def test_all_processes_hit_spmd_barriers_even_when_unprofiled(
      self,
      mock_start_trace,
      mock_stop_trace,
      mock_broadcast,
      mock_sync,
      mock_find_host0,
      mock_augment,
  ):
    mesh = _multihost_mesh(num_stages=2, devices_per_stage=4, processes_per_stage=2)  # selects [0, 2] out of [0, 1, 2, 3]
    with tempfile.TemporaryDirectory() as tmp:
      root_bytes = tmp.encode("utf-8")
      bcast_buf = np.zeros(4096, dtype=np.uint8)
      bcast_buf[:4] = np.frombuffer(len(root_bytes).to_bytes(4, "little"), dtype=np.uint8)
      bcast_buf[4 : 4 + len(root_bytes)] = np.frombuffer(root_bytes, dtype=np.uint8)
      mock_broadcast.return_value = bcast_buf
      mock_find_host0.return_value = os.path.join(tmp, "host0.xplane.pb")
      mock_augment.return_value = {
          "host0_xplane_path": os.path.join(tmp, "host0.xplane.pb"),
          "schedule": "gpipe",
          "num_stages": 2,
          "num_microbatches": 8,
          "stages_measured": [0, 1],
          "planes_added": 2,
          "events_added": 42,
          "align_p95_us": 1.2,
          "fallback_mode": False,
          "fallback_reason": "",
          "elapsed_s": 0.1,
      }

      for proc_idx in range(4):
        mock_start_trace.reset_mock()
        mock_stop_trace.reset_mock()
        mock_broadcast.reset_mock()
        mock_sync.reset_mock()
        mock_augment.reset_mock()
        local = [d for d in mesh.devices.flat if d.process_index == proc_idx]
        prof = profiler.Profiler(_config(tensorboard_dir=tmp), mesh=mesh)
        with (
            patch("jax.process_index", return_value=proc_idx),
            patch("jax.process_count", return_value=4),
            patch("jax.local_devices", return_value=local),
        ):
          prof.maybe_activate_profiler(prof.start_initial_profile_step, None)
          mock_broadcast.assert_called_once()
          if proc_idx == 0:
            mock_start_trace.assert_called_once()
            self.assertEqual(mock_start_trace.call_args.args[0], tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "pipeline_trace_metadata_process_00000.json")))
          elif proc_idx == 2:
            expected_dir = os.path.join(tmp, pipeline_trace_lens.STAGE_TRACES_SUBDIR, "process_00002")
            mock_start_trace.assert_called_once()
            self.assertEqual(mock_start_trace.call_args.args[0], expected_dir)
            self.assertTrue(os.path.exists(os.path.join(expected_dir, profiler.PIPELINE_TRACE_METADATA_FILENAME)))
          else:
            mock_start_trace.assert_not_called()

          prof.maybe_deactivate_profiler(prof.finished_initial_profile_step, None)
          mock_sync.assert_called_once_with("maxtext_pp_profile_flush")
          if proc_idx in (0, 2):
            mock_stop_trace.assert_called_once()
          else:
            mock_stop_trace.assert_not_called()
          if proc_idx == 0:
            mock_augment.assert_called_once_with(os.path.join(tmp, "host0.xplane.pb"), search_dir=tmp)
          else:
            mock_augment.assert_not_called()

          # Calling deactivate() again at the end of train_loop must be an idempotent no-op.
          prof.deactivate()
          mock_sync.assert_called_once()
          if proc_idx == 0:
            mock_augment.assert_called_once()

  @patch.object(pipeline_trace_lens, "augment_host0_xplane_in_place", side_effect=RuntimeError("boom"))
  @patch.object(pipeline_trace_lens, "find_host0_xplane", return_value="/tmp/fake.xplane.pb")
  @patch("jax.profiler.stop_trace")
  @patch("jax.profiler.start_trace")
  def test_augmentation_failure_never_crashes_training(
      self, mock_start_trace, mock_stop_trace, mock_find_host0, mock_augment
  ):
    del mock_start_trace, mock_stop_trace, mock_find_host0
    mesh, local = _v6e4_mesh()
    with tempfile.TemporaryDirectory() as tmp:
      prof = profiler.Profiler(_config(tensorboard_dir=tmp), mesh=mesh)
      with (
          patch("jax.process_index", return_value=0),
          patch("jax.process_count", return_value=1),
          patch("jax.local_devices", return_value=local),
          patch.object(profiler.max_logging, "log") as mock_log,
      ):
        prof.activate()
        prof.deactivate()  # Must not raise.
      mock_augment.assert_called_once()
      self.assertTrue(any("Failed to augment Process 0 trace" in str(c.args[0]) for c in mock_log.call_args_list))


class PipelineTraceMetadataTest(unittest.TestCase):
  """Contents of the pp_trace_metadata/v0 sidecar."""

  def test_stage_mapping_follows_mesh_not_device_order(self):
    mesh, local = _v6e4_mesh()
    meta = profiler.build_pipeline_trace_metadata(
        _config(), mesh, local, process_index=0, process_count=1, hostname="host-a"
    )
    json.dumps(meta)
    self.assertEqual(meta["schema"], profiler.PIPELINE_TRACE_METADATA_SCHEMA)
    self.assertEqual(meta["hostname"], "host-a")
    self.assertEqual(meta["mesh_axes"], ["data", "stage", "fsdp"])
    self.assertEqual(meta["mesh_shape"], [1, 4, 1])
    self.assertEqual(meta["mesh_device_ids"], [[[0], [1], [3], [2]]])
    self.assertEqual(meta["stage_axis"], 1)
    self.assertEqual([d["id"] for d in meta["local_devices"]], [0, 1, 2, 3])
    self.assertEqual([d["stage"] for d in meta["local_devices"]], [0, 1, 3, 2])
    self.assertEqual(meta["local_devices"][2]["mesh_index"], [0, 3, 0])
    self.assertEqual(meta["local_devices"][3]["coords"], [1, 1, 0])
    self.assertEqual(meta["pipeline"]["schedule"], "gpipe")
    self.assertTrue(meta["profiler"]["enable_pipeline_stage_profiling"])

  def test_schedule_inferred_from_repeats_when_config_has_no_schedule_field(self):
    mesh, local = _v6e4_mesh()
    meta_circ = profiler.build_pipeline_trace_metadata(
        _config(num_pipeline_repeats=2), mesh, local, process_index=0, process_count=1, hostname="host-a"
    )
    self.assertEqual(meta_circ["pipeline"]["schedule"], "circular")
    meta_explicit = profiler.build_pipeline_trace_metadata(
        _config(pipeline_schedule="custom_sched", num_pipeline_repeats=2),
        mesh,
        local,
        process_index=0,
        process_count=1,
        hostname="host-a",
    )
    self.assertEqual(meta_explicit["pipeline"]["schedule"], "custom_sched")


class ChipsToProfileTest(unittest.TestCase):
  """Automatic local-chip coverage for single-stage vs multi-stage processes."""

  def test_num_local_chips_counts_chips_not_cores(self):
    two_cores_per_chip = [_fake_device(i, 0, coords=[i // 2, 0, 0], core_on_chip=i % 2) for i in range(8)]
    self.assertEqual(profiler.num_local_chips(two_cores_per_chip), 4)
    no_coords = [pytypes.SimpleNamespace(id=i, coords=None) for i in range(3)]
    self.assertEqual(profiler.num_local_chips(no_coords), 3)

  def test_num_local_pipeline_stages(self):
    multihost = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=1)
    proc0_devices = [d for d in multihost.devices.flat if d.process_index == 0]
    self.assertEqual(profiler.num_local_pipeline_stages(multihost, proc0_devices), 1)

    v6e4_mesh, v6e4_local = _v6e4_mesh()
    self.assertEqual(profiler.num_local_pipeline_stages(v6e4_mesh, v6e4_local), 4)

    # PP=8 across 4 hosts (each process holds 2 stages across 4 local chips).
    pp8_grid = [[_fake_device(s * 2 + f, s // 2) for f in range(2)] for s in range(8)]
    pp8_mesh = _fake_mesh(("stage", "fsdp"), pp8_grid)
    pp8_proc0 = [d for d in pp8_mesh.devices.flat if d.process_index == 0]
    self.assertEqual(profiler.num_local_pipeline_stages(pp8_mesh, pp8_proc0), 2)
    self.assertEqual(profiler.num_local_pipeline_stages(None, proc0_devices), 0)

  def test_single_stage_per_process_keeps_requested_chips_without_warning(self):
    mesh = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=1)
    proc0_devices = [d for d in mesh.devices.flat if d.process_index == 0]
    cfg = _config(tpu_num_chips_to_profile_per_task=1)
    with patch.object(profiler.max_logging, "log") as mock_log:
      chips = profiler.resolve_tpu_num_chips_to_profile(
          cfg, 4, pipeline_stage_profiling=True, mesh=mesh, local_devices=proc0_devices
      )
    self.assertEqual(chips, 1)
    mock_log.assert_not_called()

  def test_upload_all_single_stage_logs_info_note_without_removed_flag(self):
    mesh = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=1)
    proc0_devices = [d for d in mesh.devices.flat if d.process_index == 0]
    cfg = _config(upload_all_profiler_results=True, tpu_num_chips_to_profile_per_task=1)
    with patch.object(profiler.max_logging, "log") as mock_log:
      chips = profiler.resolve_tpu_num_chips_to_profile(
          cfg, 4, pipeline_stage_profiling=True, mesh=mesh, local_devices=proc0_devices
      )
    self.assertEqual(chips, 1)
    self.assertEqual(mock_log.call_count, 1)
    msg = mock_log.call_args.args[0]
    self.assertIn("tpu_num_chips_to_profile_per_task", msg)
    self.assertNotIn("WARNING", msg)
    self.assertNotIn("all_local_chips", msg)

  def test_multi_stage_per_process_automatically_traces_all_local_chips(self):
    v6e4_mesh, v6e4_local = _v6e4_mesh()
    cfg = _config(tpu_num_chips_to_profile_per_task=1)
    with patch.object(profiler.max_logging, "log") as mock_log:
      chips = profiler.resolve_tpu_num_chips_to_profile(
          cfg, 4, pipeline_stage_profiling=True, mesh=v6e4_mesh, local_devices=v6e4_local
      )
    self.assertEqual(chips, 4)
    self.assertEqual(mock_log.call_count, 1)
    msg = mock_log.call_args.args[0]
    self.assertIn("tpu_num_chips_to_profile_per_task", msg)
    self.assertIn("4 pipeline stages", msg)
    self.assertNotIn("WARNING", msg)

    # PP=8 across 4 hosts (2 stages across 4 local chips on process 0).
    pp8_grid = [[_fake_device(s * 2 + f, s // 2) for f in range(2)] for s in range(8)]
    pp8_mesh = _fake_mesh(("stage", "fsdp"), pp8_grid)
    pp8_proc0 = [d for d in pp8_mesh.devices.flat if d.process_index == 0]
    with patch.object(profiler.max_logging, "log") as mock_log:
      chips_pp8 = profiler.resolve_tpu_num_chips_to_profile(
          cfg, 4, pipeline_stage_profiling=True, mesh=pp8_mesh, local_devices=pp8_proc0
      )
    self.assertEqual(chips_pp8, 4)
    self.assertEqual(mock_log.call_count, 1)
    self.assertIn("2 pipeline stages", mock_log.call_args.args[0])

  def test_profiler_init_tpu_options_respects_local_stage_count(self):
    multihost = _multihost_mesh(num_stages=4, devices_per_stage=4, processes_per_stage=1)
    mh_local = [d for d in multihost.devices.flat if d.process_index == 0]
    cfg = _config(enable_tpu_profiling_options=True, tpu_num_chips_to_profile_per_task=1)
    with patch("jax.local_devices", return_value=mh_local):
      prof_mh = profiler.Profiler(cfg, mesh=multihost)
    self.assertEqual(prof_mh.profiling_options.advanced_configuration["tpu_num_chips_to_profile_per_task"], 1)

    v6e4_mesh, v6e4_local = _v6e4_mesh()
    with patch("jax.local_devices", return_value=v6e4_local):
      prof_sp = profiler.Profiler(cfg, mesh=v6e4_mesh)
    self.assertEqual(prof_sp.profiling_options.advanced_configuration["tpu_num_chips_to_profile_per_task"], 4)


class ConfigDefaultsTest(unittest.TestCase):
  """Only enable_pipeline_stage_profiling is added on the config schema."""

  def test_config_defaults(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        run_name="test_profiler_pipeline_defaults",
        base_output_directory="/tmp",
    )
    self.assertTrue(config.enable_pipeline_stage_profiling)
    profiling_fields = list(getattr(config_types.Profiling, "model_fields").keys())
    all_fields = list(getattr(config_types.MaxTextConfig, "model_fields").keys())
    self.assertFalse(any("all_local_chips" in attr for attr in dir(config)))
    self.assertFalse(any("all_local_chips" in field for field in profiling_fields))
    self.assertFalse(any("all_local_chips" in field for field in all_fields))
    self.assertEqual([f for f in profiling_fields if "chips" in f], ["tpu_num_chips_to_profile_per_task"])
    self.assertEqual([f for f in profiling_fields if "pipeline" in f], ["enable_pipeline_stage_profiling"])


if __name__ == "__main__":
  unittest.main()
