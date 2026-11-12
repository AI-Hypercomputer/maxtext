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

"""Tests checkpoint_load_benchmark/bench/harness_v2.py and Phase 2 harness integration."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import harness_v2  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position
from bench import stages  # pylint: disable=wrong-import-position


def _load_harness_module():
  harness_path = os.path.join(_BENCHMARK_DIR, "benchmark_ckpt_load_time.py")
  spec = importlib.util.spec_from_file_location("benchmark_ckpt_load_time_phase2_test", harness_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class HarnessV2Test(unittest.TestCase):
  """Verifies Phase 2 samplers, hook wrappers, failure attribution, checksums, and v2 records."""

  def test_resource_sampler_tracks_peak_ram_and_hbm_milestones(self):
    rec = stages.StageRecorder()
    ram_seq = iter([(10.0, 90.0, 100.0), (42.5, 57.5, 100.0), (30.0, 70.0, 100.0)])
    fake_devs = [
        types.SimpleNamespace(
            memory_stats=lambda: {
                "bytes_limit": 32_000_000_000,
                "bytes_in_use": next(dev_used, 12_000_000_000),
                "peak_bytes_in_use": 22_500_000_000,
            }
        )
    ]
    dev_used = iter([1_000_000_000, 18_000_000_000, 12_000_000_000])
    fake_jax = types.SimpleNamespace(local_devices=lambda: fake_devs)
    sampler = harness_v2.ResourceSampler(
        rec,
        jax_mod=fake_jax,
        ram_reader=lambda: next(ram_seq, (30.0, 70.0, 100.0)),
        rss_reader=lambda: 15.25,
    )
    with rec.stage("load"):
      sampler._sample_once()  # pylint: disable=protected-access
      sampler.record_hbm_milestone("after_read")
    sampler.record_hbm_milestone("after_load")
    sampler.stop()
    mem = sampler.memory_payload()
    self.assertAlmostEqual(mem["peak_host_ram_container_gb"], 42.5)
    self.assertAlmostEqual(mem["peak_process_rss_gb"], 15.25)
    self.assertAlmostEqual(mem["memory_tpu_hbm_usable_per_device_gb"], 32.0)
    self.assertAlmostEqual(mem["memory_tpu_hbm_used_after_read_per_device_gb"], 18.0)
    self.assertAlmostEqual(mem["memory_tpu_hbm_used_after_load_per_device_gb"], 12.0)
    self.assertAlmostEqual(mem["memory_tpu_hbm_peak_per_device_gb"], 22.5)

  def test_install_safetensors_dynamic_hooks_records_slowest_param_and_sub_stages(self):
    rec = stages.StageRecorder()

    def fake_get_hf_loader(_src, _get, _hk, _leaf, _cfg, _mt_key, num_parts=0):
      del num_parts
      return lambda: "loaded_tensor"

    load_dynamic = types.SimpleNamespace(
        _leading_axis_shardings=lambda x: x,
        get_hf_loading_function=fake_get_hf_loader,
    )
    tensor_handling = types.SimpleNamespace(
        apply_hook_fns=lambda w: w,
        _reshard=lambda x: x,
        _binary_chunked_stack=lambda x: x,
        _concatenate_pieces=lambda x: x,
        _cast_and_free=lambda x: x,
        _gather_to_host=lambda x: x,
        _device_put=lambda x: x,
        get_hf_loading_function=fake_get_hf_loader,
    )
    slowest = harness_v2.install_safetensors_dynamic_hooks(
        rec, load_dynamic_mod=load_dynamic, tensor_handling_mod=tensor_handling
    )
    with rec.stage("safetensors_dynamic_step2_transform_total_seconds"):
      fn = load_dynamic.get_hf_loading_function(None, None, None, None, None, "params-layers-weight")
      self.assertEqual(fn(), "loaded_tensor")
      tensor_handling.apply_hook_fns("w")
      tensor_handling._gather_to_host("g")  # pylint: disable=protected-access
      tensor_handling._gather_to_host("g")  # pylint: disable=protected-access

    self.assertEqual(slowest["slowest_name"], "params-layers-weight")
    self.assertGreater(slowest["slowest_s"], 0.0)
    snap = rec.snapshot()
    self.assertEqual(snap["stages"]["safetensors_dynamic_step2_gather_to_host_fallback_seconds"]["count"], 2)
    rec.restore_all()

  def test_classify_failure_maps_legend_codes(self):
    self.assertEqual(harness_v2.classify_failure("fail", ["[NOT-TPU] backend=cpu"])[0], "NOT-TPU")
    self.assertEqual(harness_v2.classify_failure("fail", ["[TIMEOUT_WATCHDOG]"])[0], "TIMEOUT")
    self.assertEqual(harness_v2.classify_failure("RESOURCE_EXHAUSTED: oom")[0], "OOM-HBM")
    self.assertEqual(harness_v2.classify_failure("killed", ["OOMKilled"], exit_code=-9)[0], "OOM-HOST")
    cat, sig, pre, _ = harness_v2.classify_failure("evicted", ["spot preemption"], exit_code=-15)
    self.assertEqual((cat, sig, pre), ("PREEMPTED", "SIGTERM", True))

  def test_compute_and_compare_parameter_checksums_on_cpu_jax(self):
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    tree_a = {
        "params": {
            f"k{i:02d}": jax.device_put(jnp.ones((2, 3), dtype=jnp.bfloat16) * (i + 1), jax.local_devices()[0])
            for i in range(10)
        }
    }
    tree_b = {
        f"k{i:02d}": jax.device_put(jnp.ones((2, 3), dtype=jnp.bfloat16) * (i + 1), jax.local_devices()[0])
        for i in range(10)
    }
    cs_a = harness_v2.compute_parameter_checksums(tree_a, jax, max_leaves=8)
    cs_b = harness_v2.compute_parameter_checksums(tree_b, jax, max_leaves=8)
    self.assertEqual(len(cs_a), 8)
    self.assertEqual(harness_v2.compare_parameter_checksums(cs_a, cs_b), [])

    cs_bad = dict(cs_b)
    cs_bad["k00"] = dict(cs_bad["k00"], sum=999.0)
    self.assertTrue(harness_v2.compare_parameter_checksums(cs_a, cs_bad))

  def test_compute_local_shard_gb_and_build_v2_phase_record_validates(self):
    fake_arr = types.SimpleNamespace(addressable_shards=[types.SimpleNamespace(data=np.zeros(2_000_000, dtype=np.uint8))])
    self.assertAlmostEqual(harness_v2.compute_local_shard_gb({"a": fake_arr}), 0.002)

    rec_line = stages.StageRecorder()
    with rec_line.stage("setup_tpu_check_seconds"):
      pass
    with rec_line.stage("safetensors_dynamic_step2_total_seconds"):
      rec_line.add_metric("safetensors_dynamic_step2_load_total_seconds", "bytes_needed_by_this_host_gb", 0.75)

    merged = stages.merge_host_stages(
        [
            {
                "process_index": 0,
                "slice_id": 0,
                "host_name": "pod-0",
                **rec_line.snapshot(),
                "monitoring": {"events": {"/jax/orbax/read/safetensors/bytes_read": {"sum": 750_000_000}}},
                "memory": {
                    "peak_host_ram_container_gb": 22.0,
                    "peak_process_rss_gb": 12.0,
                    "host_ram_size_gb": 100.0,
                    "memory_tpu_hbm_usable_per_device_gb": 32.0,
                    "memory_tpu_hbm_peak_per_device_gb": 2.0,
                },
                "active_timing_wrappers": ["load_dynamic._leading_axis_shardings"],
            }
        ]
    )

    v2_rec = harness_v2.build_v2_phase_record(
        phase="safetensors_dynamic",
        model_name="qwen3-0.6b",
        scan_mode="scanned",
        platform_name="v6e",
        num_slices=1,
        num_hosts_per_slice=1,
        topology="2x4",
        raw_worker_data={
            "hf_to_gcs_s": 1.0,
            "safetensors_and_convert_s": 8.0,
            "num_param_leaves": 10,
            "jax_backend_platform": "tpu",
            "tpu_device_kind": "TPU v6 lite",
            "total_tpu_devices_measured": 8,
            "tpu_devices_per_host_measured": 8,
            "tpu_hbm_per_device_gb_measured": 32.0,
        },
        merged_stages=merged,
        env_meta={"run_id": "v2-smoke", "container_image": "img:v1"},
        code_ver={
            "git_branch_name": "docs/checkpoint-load-benchmark",
            "git_commit": "dd4ecda6c",
            "git_working_tree_clean": True,
        },
        hw_info={"host_ram_gb_measured": 100.0, "host_vcpu_count": 32},
        pkg_versions={"jax_version": "0.6.0"},
        started_at_utc="2026-11-12T08:00:00Z",
        finished_at_utc="2026-11-12T08:01:00Z",
        wall_clock_s=15.0,
        run_status="OK",
        cache_state="empty",
    )
    self.assertEqual(schema_v2.validate_v2_phase_record(v2_rec), [])
    self.assertEqual(v2_rec["total_tpu_devices"], 8)
    self.assertAlmostEqual(v2_rec["safetensors_dynamic_step2_read_from_gcs_to_host_ram_gb"], 0.75)
    self.assertAlmostEqual(v2_rec["safetensors_dynamic_step2_bytes_read_divided_by_bytes_needed_ratio"], 1.0)

  def test_worker_subprocess_emits_stages_with_low_unattributed_time(self):
    harness = _load_harness_module()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    log_dir = os.path.join(tmp_dir, "logs", "qwen3-0.6b", "scanned")
    os.makedirs(log_dir, exist_ok=True)
    result_json = os.path.join(log_dir, "safetensors_dynamic_result.json")
    log_file = os.path.join(log_dir, "safetensors_dynamic.log")
    runner_py = os.path.join(tmp_dir, "mock_worker.py")
    harness_path = os.path.join(_BENCHMARK_DIR, "benchmark_ckpt_load_time.py")

    with open(runner_py, "w", encoding="utf-8") as rf:
      rf.write(
          f"""import importlib.util, os, sys, time
spec = importlib.util.spec_from_file_location("bclt_w", {harness_path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
def _fake_sd(args):
  rec = getattr(args, "_recorder", None)
  with rec.stage("safetensors_dynamic_step1_total_seconds"):
    time.sleep(0.05)
  with rec.stage("safetensors_dynamic_step2_total_seconds", log_start=True):
    time.sleep(0.45)
  return {{
      "hf_to_gcs_s": 0.05,
      "safetensors_and_convert_s": 0.45,
      "num_param_leaves": 4,
      "num_slices": 1,
      "global_devices": 8,
      "jax_backend_platform": "tpu",
      "tpu_device_kind": "TPU v7x",
      "total_tpu_devices_measured": 8,
      "tpu_devices_per_host_measured": 8,
      "tpu_hbm_per_device_gb_measured": 96.0,
      "parameter_checksums": {{"w": {{"shape": [2, 2], "dtype": "bfloat16", "sum": 4.0, "l2": 2.0, "abs_max": 1.0}}}},
  }}
mod.run_worker_safetensors_dynamic = _fake_sd
sys.argv = [
    {harness_path!r},
    "--worker-phase", "safetensors_dynamic",
    "--model", "qwen3-0.6b",
    "--scan-mode", "scanned",
    "--hardware", "tpu",
    "--platform", "v7x",
    "--slice-topology", "2x2x1",
    "--devices-per-slice", "8",
    "--gcs-run-dir", "gs://mesa-maxtext/mesa-bench-v2/v7x/run_test",
    "--local-hf-dir", {os.path.join(tmp_dir, "hf")!r},
    "--result-json", {result_json!r},
]
mod.main()
"""
      )

    env = os.environ.copy()
    env["BENCH_PROCESS_START_EPOCH"] = f"{time.time():.6f}"
    env["POD_NAME"] = "pod-w-0"
    env["NODE_NAME"] = "node-w-0"
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, runner_py], env=env, capture_output=True, text=True, check=True)
    wall_s = time.perf_counter() - t0
    with open(log_file, "w", encoding="utf-8") as lf:
      lf.write(proc.stdout + proc.stderr)
    with open(result_json, "r", encoding="utf-8") as rf:
      raw_data = json.load(rf)

    v2_rec = harness._finalize_v2_phase_result(  # pylint: disable=protected-access
        phase="safetensors_dynamic",
        model="qwen3-0.6b",
        scan_mode="scanned",
        platform_name="v7x",
        slice_topology="2x2x1",
        num_slices=1,
        num_hosts_per_slice=1,
        devices_per_slice=8,
        raw_data=raw_data,
        run_status="OK",
        started_at_utc="2026-11-12T08:00:00Z",
        finished_at_utc="2026-11-12T08:00:02Z",
        wall_clock_s=wall_s,
        cache_state="empty",
        env=env,
        repo_root=os.path.dirname(_BENCHMARK_DIR),
        gcs_run_dir="gs://mesa-maxtext/mesa-bench-v2/v7x/run_test",
        scan_dir_name="scanned",
        log_dir=log_dir,
        worker_cached_times=[],
    )
    self.assertEqual(schema_v2.validate_v2_phase_record(v2_rec), [])
    self.assertEqual(v2_rec["pod_names"], "pod-w-0")
    self.assertEqual(v2_rec["node_names"], "node-w-0")
    attributed = (
        float(v2_rec["setup_process_start_to_harness_code_seconds"])
        + float(v2_rec["setup_python_imports_seconds"])
        + float(v2_rec["safetensors_dynamic_step1_total_seconds"])
        + float(v2_rec["safetensors_dynamic_step2_total_seconds"])
        + float(v2_rec["teardown_write_results_and_exit_seconds"])
    )
    unattributed = abs(wall_s - attributed)
    self.assertLessEqual(unattributed / max(wall_s, 1e-6), 0.02)

  def test_multihost_log_merge_and_remote_timeout_attribution(self):
    harness = _load_harness_module()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    log_dir = os.path.join(tmp_dir, "logs", "llama3.1-8b", "scanned_slices2")
    os.makedirs(log_dir, exist_ok=True)
    host0 = {
        "process_index": 0,
        "slice_id": 0,
        "host_name": "pod-0",
        "pod_name": "pod-0",
        "node_name": "node-a",
        "stages": {"safetensors_dynamic_step2_total_seconds": {"inclusive_s": 5.0, "exclusive_s": 5.0, "count": 1}},
        "barriers": {},
        "monitoring": {"events": {}},
        "memory": {"peak_host_ram_container_gb": 10.0, "peak_process_rss_gb": 8.0, "host_ram_size_gb": 960.0},
    }
    host1 = {
        "process_index": 1,
        "slice_id": 1,
        "host_name": "pod-1",
        "pod_name": "pod-1",
        "node_name": "node-b",
        "stages": {"safetensors_dynamic_step2_total_seconds": {"inclusive_s": 7.5, "exclusive_s": 7.5, "count": 1}},
        "barriers": {},
        "monitoring": {"events": {}},
        "memory": {"peak_host_ram_container_gb": 14.0, "peak_process_rss_gb": 9.0, "host_ram_size_gb": 960.0},
    }
    with open(os.path.join(log_dir, "safetensors_dynamic.log"), "w", encoding="utf-8") as lf0:
      lf0.write("BENCH_STAGES_JSON " + json.dumps(host0) + "\n")
    with open(os.path.join(log_dir, "safetensors_dynamic_slice1.log"), "w", encoding="utf-8") as lf1:
      lf1.write(
          "[STAGE START] safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds\n"
          "[TIMEOUT_WATCHDOG] exceeded PHASE_TIMEOUT_S\n"
          "BENCH_STAGES_JSON " + json.dumps(host1) + "\n"
      )

    rec = harness._finalize_v2_phase_result(  # pylint: disable=protected-access
        phase="safetensors_dynamic",
        model="llama3.1-8b",
        scan_mode="scanned",
        platform_name="v7x",
        slice_topology="2x2x1",
        num_slices=2,
        num_hosts_per_slice=1,
        devices_per_slice=8,
        raw_data={"status": "FAILED", "error": "remote slice 1 failed", "num_slices": 2, "global_devices": 16},
        run_status="FAILED",
        started_at_utc="2026-11-12T08:00:00Z",
        finished_at_utc="2026-11-12T08:05:00Z",
        wall_clock_s=300.0,
        cache_state="empty",
        env=dict(os.environ),
        repo_root=os.path.dirname(_BENCHMARK_DIR),
        gcs_run_dir="gs://mesa-maxtext/mesa-bench-v2/v7x/run_test",
        scan_dir_name="scanned_slices2",
        log_dir=log_dir,
        worker_cached_times=[],
        failure_err_text="remote slice 1 failed",
        failure_exit_code=124,
    )
    self.assertEqual(schema_v2.validate_v2_phase_record(rec), [])
    self.assertEqual(rec["failure_category"], "TIMEOUT")
    self.assertEqual(rec["failed_stage_name"], "safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds")
    self.assertEqual(rec["pod_names"], "pod-0,pod-1")
    self.assertEqual(rec["node_names"], "node-a,node-b")
    self.assertEqual(rec["stages_merged"]["host_count"], 2)
    self.assertAlmostEqual(rec["safetensors_dynamic_step2_total_seconds"], 7.5)

  def test_main_writes_skipped_v2_jsons_and_verifies_checksums(self):
    harness = _load_harness_module()
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    work_skip = os.path.join(tmp_dir, "skip_run")
    argv_skip = [
        "benchmark_ckpt_load_time.py",
        "--models",
        "qwen3-0.6b",
        "--scan-modes",
        "scanned",
        "--num-slices-list",
        "1",
        "--platform",
        "v6e",
        "--slice-topology",
        "2x4",
        "--devices-per-slice",
        "8",
        "--gcs-base",
        os.path.join(tmp_dir, "gcs"),
        "--run-id",
        "r_skip",
        "--local-work-dir",
        work_skip,
        "--csv-report",
        os.path.join(tmp_dir, "skip.csv"),
        "--skip-to-maxtext",
        "--skip-safetensors",
    ]
    with mock.patch.object(sys, "argv", argv_skip):
      harness.main()
    for ph in ("to_maxtext_step1", "to_maxtext_step2", "to_maxtext_step3", "safetensors_dynamic"):
      p_json = os.path.join(work_skip, "logs", "qwen3-0.6b", "scanned", f"{ph}_result.json")
      with open(p_json, "r", encoding="utf-8") as f:
        rec = json.load(f)
      self.assertEqual(rec["run_status"], "SKIPPED")
      self.assertEqual(schema_v2.validate_v2_phase_record(rec), [])

    work_cs = os.path.join(tmp_dir, "cs_run")
    sd_dir = os.path.join(work_cs, "logs", "qwen3-0.6b", "scanned")
    os.makedirs(sd_dir, exist_ok=True)
    sd_json_path = os.path.join(sd_dir, "safetensors_dynamic_result.json")

    def _make_fake_invoke(sd_sum: float):
      def _fake_invoke(phase, **_kw):
        res = {"status": "OK", "run_status": "OK", "global_devices": 8}
        if phase == "to_maxtext_step3":
          res["load_orbax_s"] = 1.0
          res["parameter_checksums"] = {
              "w": {"shape": [2, 2], "dtype": "bfloat16", "sum": 4.0, "l2": 2.0, "abs_max": 1.0}
          }
        elif phase == "safetensors_dynamic":
          res["hf_to_gcs_s"] = 0.2
          res["safetensors_and_convert_s"] = 0.8
          res["parameter_checksums"] = {
              "w": {"shape": [2, 2], "dtype": "bfloat16", "sum": sd_sum, "l2": 2.0, "abs_max": 1.0}
          }
          with open(sd_json_path, "w", encoding="utf-8") as sf:
            json.dump(res, sf)
        return res

      return _fake_invoke

    argv_cs = [
        "benchmark_ckpt_load_time.py",
        "--models",
        "qwen3-0.6b",
        "--scan-modes",
        "scanned",
        "--num-slices-list",
        "1",
        "--platform",
        "v7x",
        "--slice-topology",
        "2x2x1",
        "--devices-per-slice",
        "8",
        "--verify-checksums",
        "--gcs-base",
        os.path.join(tmp_dir, "gcs"),
        "--run-id",
        "r_cs",
        "--local-work-dir",
        work_cs,
        "--csv-report",
        os.path.join(tmp_dir, "cs.csv"),
    ]
    with (
        mock.patch.object(sys, "argv", argv_cs),
        mock.patch.object(harness, "invoke_worker_subprocess", side_effect=_make_fake_invoke(4.0)),
    ):
      harness.main()
    with open(sd_json_path, "r", encoding="utf-8") as f:
      self.assertEqual(json.load(f)["checksum_verification"], "MATCH")

    with (
        mock.patch.object(sys, "argv", argv_cs),
        mock.patch.object(harness, "invoke_worker_subprocess", side_effect=_make_fake_invoke(99.0)),
    ):
      harness.main()
    with open(sd_json_path, "r", encoding="utf-8") as f:
      persisted = json.load(f)
      self.assertEqual(persisted["checksum_verification"], "MISMATCH")
      self.assertTrue(persisted["checksum_mismatches"])


if __name__ == "__main__":
  unittest.main()
