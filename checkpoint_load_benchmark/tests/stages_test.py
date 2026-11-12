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

"""Tests checkpoint_load_benchmark/bench/stages.py."""

import io
import os
import sys
import types
import unittest

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import stages  # pylint: disable=wrong-import-position


class _FakeClock:
  """Deterministic monotonic clock for stage timing unit tests."""

  def __init__(self, start: float = 100.0):
    self.now = start

  def __call__(self) -> float:
    return self.now

  def advance(self, delta: float) -> None:
    self.now += delta


class StagesRecorderTest(unittest.TestCase):
  """Verifies stage nesting, wrappers, JAX monitoring hooks, and multi-host merging."""

  def test_nested_stages_account_inclusive_and_exclusive_time(self):
    mono = _FakeClock(10.0)
    wall = _FakeClock(1000.0)
    rec = stages.StageRecorder(clock=mono, wall_clock=wall)

    with rec.stage("outer"):
      mono.advance(2.0)
      wall.advance(2.0)
      with rec.stage("inner"):
        mono.advance(5.0)
        wall.advance(5.0)
        rec.add_metric("inner", "bytes", 1024)
      mono.advance(3.0)
      wall.advance(3.0)

    snap = rec.snapshot()
    self.assertAlmostEqual(snap["stages"]["outer"]["inclusive_s"], 10.0)
    self.assertAlmostEqual(snap["stages"]["outer"]["exclusive_s"], 5.0)
    self.assertAlmostEqual(snap["stages"]["inner"]["inclusive_s"], 5.0)
    self.assertAlmostEqual(snap["stages"]["inner"]["exclusive_s"], 5.0)
    self.assertEqual(snap["stages"]["inner"]["metrics"]["bytes"], 1024)

  def test_wrap_and_reentrant_guard_and_restore(self):
    mono = _FakeClock(0.0)
    rec = stages.StageRecorder(clock=mono, wall_clock=mono)

    def recursive(n: int) -> int:
      mono.advance(1.0)
      return 1 if n <= 0 else recursive(n - 1) + 1

    mod = types.SimpleNamespace(recursive=recursive)
    self.assertTrue(rec.wrap(mod, "recursive", "wrapped_rec", reentrant_guard=True, label="mod.recursive"))
    self.assertEqual(mod.recursive(2), 3)

    snap = rec.snapshot()
    self.assertEqual(snap["stages"]["wrapped_rec"]["count"], 1)
    self.assertAlmostEqual(snap["stages"]["wrapped_rec"]["inclusive_s"], 3.0)
    self.assertEqual(snap["active_timing_wrappers"], ["mod.recursive"])

    rec.restore_all()
    self.assertIs(mod.recursive, recursive)

  def test_wrap_sync_global_devices_records_barrier_time(self):
    mono = _FakeClock(0.0)
    rec = stages.StageRecorder(clock=mono, wall_clock=mono)

    def fake_sync(tag):
      mono.advance(1.25)
      return tag

    mu = types.SimpleNamespace(sync_global_devices=fake_sync)
    self.assertTrue(stages.wrap_sync_global_devices(mu, rec))
    with rec.stage("setup_device_mesh_build_seconds"):
      mono.advance(0.5)
      mu.sync_global_devices("init")

    snap = rec.snapshot()
    self.assertAlmostEqual(snap["barriers"]["init"]["total_s"], 1.25)
    self.assertEqual(snap["barriers"]["init"]["count"], 1)

  def test_monitoring_collector_captures_durations_and_scalars(self):
    dur_callbacks = []
    scalar_callbacks = []

    class FakeMonitoring:
      """Fake jax.monitoring callback registry."""

      def register_event_duration_secs_listener(self, cb):
        """Registers a duration listener."""
        dur_callbacks.append(cb)

      def register_scalar_listener(self, cb):
        """Registers a scalar listener."""
        scalar_callbacks.append(cb)

      def unregister_event_duration_secs_listener(self, cb):
        """Unregisters a duration listener."""
        dur_callbacks.remove(cb)

      def unregister_scalar_listener(self, cb):
        """Unregisters a scalar listener."""
        scalar_callbacks.remove(cb)

    rec = stages.StageRecorder()
    coll = stages.MonitoringCollector(rec)
    registered = coll.register(FakeMonitoring())
    self.assertTrue(registered)

    with rec.stage("transform"):
      dur_callbacks[0]("/jax/core/compile/backend_compile_duration", 4.5)
      scalar_callbacks[0]("/jax/orbax/read/safetensors/bytes_read", 4096)
      dur_callbacks[0]("/jax/untracked/ignored", 99.0)

    snap = coll.snapshot()
    self.assertAlmostEqual(snap["events"]["/jax/core/compile/backend_compile_duration"]["sum"], 4.5)
    self.assertNotIn("/jax/untracked/ignored", snap["events"])
    self.assertEqual(
        snap["by_stage"]["transform"]["/jax/orbax/read/safetensors/bytes_read"]["sum"],
        4096.0,
    )
    coll.unregister()
    self.assertEqual(dur_callbacks, [])
    self.assertEqual(scalar_callbacks, [])

  def test_emit_parse_and_merge_host_stages(self):
    buf = io.StringIO()
    p0 = {
        "process_index": 0,
        "slice_id": 0,
        "host_name": "host-0",
        "stages": {"load": {"inclusive_s": 10.0, "metrics": {"bytes": 100}}},
        "barriers": {"step2": {"total_s": 1.0}},
        "memory": {"peak_host_ram_container_gb": 50.0, "memory_tpu_hbm_peak_per_device_gb": 12.0},
        "active_timing_wrappers": ["a"],
    }
    p1 = {
        "process_index": 1,
        "slice_id": 1,
        "host_name": "host-1",
        "stages": {"load": {"inclusive_s": 15.0, "metrics": {"bytes": 200}}},
        "barriers": {"step2": {"total_s": 3.0}},
        "memory": {"peak_host_ram_container_gb": 80.5, "memory_tpu_hbm_peak_per_device_gb": 14.0},
        "active_timing_wrappers": ["a"],
    }
    stages.emit_host_stages_line(p0, stream=buf)
    buf.write("some unrelated log line\n")
    stages.emit_host_stages_line(p1, stream=buf)

    parsed = stages.parse_host_stages_lines(buf.getvalue())
    self.assertEqual(len(parsed), 2)
    merged = stages.merge_host_stages(parsed)
    self.assertEqual(merged["host_count"], 2)
    self.assertEqual(merged["stages_summary"]["load"]["slowest_host"], "host-1")
    self.assertAlmostEqual(merged["stages_summary"]["load"]["max_s"], 15.0)
    self.assertAlmostEqual(merged["stages_summary"]["load"]["metrics"]["bytes"]["sum"], 300.0)
    self.assertAlmostEqual(merged["memory_summary"]["memory_peak_host_ram_container_gb"], 80.5)
    self.assertEqual(merged["memory_summary"]["memory_peak_host_ram_host_name"], "host-1")
    self.assertAlmostEqual(merged["memory_summary"]["memory_tpu_hbm_peak_per_device_gb"], 14.0)

  def test_monitoring_collector_ignores_epoch_duration_noise_but_keeps_scalars(self):
    rec = stages.StageRecorder()
    mon = stages.MonitoringCollector(rec)
    dur = []
    scl = []
    fake_mod = types.SimpleNamespace(
        register_event_duration_secs_listener=dur.append,
        register_scalar_listener=scl.append,
    )
    mon.register(fake_mod)
    dur_cb = dur[0]
    scl_cb = scl[0]
    with rec.stage("load"):
      dur_cb("/jax/core/compile/jaxpr_trace_duration", 1794473596.75)
      dur_cb("/jax/core/compile/jaxpr_trace_duration", 1.88)
      scl_cb("/jax/orbax/read/safetensors/bytes_read", 2_000_000_000)
    snap = mon.snapshot()
    self.assertEqual(snap["events"]["/jax/core/compile/jaxpr_trace_duration"]["count"], 1)
    self.assertAlmostEqual(snap["events"]["/jax/core/compile/jaxpr_trace_duration"]["sum"], 1.88)
    self.assertEqual(snap["events"]["/jax/orbax/read/safetensors/bytes_read"]["sum"], 2_000_000_000.0)


if __name__ == "__main__":
  unittest.main()
