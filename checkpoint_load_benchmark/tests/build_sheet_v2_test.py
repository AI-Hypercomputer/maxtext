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

"""Golden tests for `checkpoint_load_benchmark/scripts/build_sheet_v2.py`."""

import json
import os
import shutil
import sys
import tempfile
import unittest

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import schema_v2  # pylint: disable=wrong-import-position
from scripts import build_sheet_v2  # pylint: disable=wrong-import-position


def _write_json(path: str, payload: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)


class BuildSheetV2Test(unittest.TestCase):
  """Tests planned matrix rows, repetition medians, speedups, Step 1b, and gsheets batches."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir)

  def test_builds_planned_matrix_medians_failures_and_batches(self):
    v2_root = os.path.join(self.tmp_dir, "v2_mirror")
    out_dir = os.path.join(self.tmp_dir, "sheet_out")

    # 1. Cold Step 1 for llama3.1-8b on v7x + DeepSeek Step 1 & Step 1b on v7x & v6e.
    _write_json(
        os.path.join(v2_root, "step1", "llama3.1-8b_step1_result.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "step1_hf_to_gcs",
            "tpu_platform": "v7x",
            "step1_gcs_bucket_region": "us-central1",
            "model_name": "llama3.1-8b",
            "step1_hf_repo_id": "meta-llama/Llama-3.1-8B",
            "step1_list_hf_repo_files_seconds": 0.4,
            "step1_files_already_in_gcs_count": 0,
            "step1_download_hf_to_gcs_seconds": 40.0,
            "step1_download_hf_to_gcs_gb": 16.1,
            "step1_download_hf_to_gcs_gb_per_second": 0.4025,
            "step1_number_of_files": 4,
            "step1_slowest_file_download_seconds": 12.0,
            "step1_download_workers_count": 16,
            "step1_host_machine_type": "n2d-standard-64",
            "step1_total_seconds": 40.4,
            "step1_value_source": "cold_download",
            "step1_measured_by_run_id": "s1-llama",
            "run_id": "s1-llama",
            "run_status": "OK",
            "phase_started_at_utc": "2026-11-12T08:00:00Z",
            "phase_finished_at_utc": "2026-11-12T08:00:41Z",
            "step1_log_url": "gs://mesa-maxtext-us-central1/mesa-bench-v2/step1/llama.log",
            "result_json_url": "gs://mesa-maxtext-us-central1/mesa-bench-v2/step1/llama3.1-8b_step1_result.json",
            "code_version": {
                "git_branch_name": "docs/checkpoint-load-benchmark",
                "git_commit": "deadbeef1234",
                "git_working_tree_clean": True,
                "code_tarball_sha256": "abcd1234",
            },
            "package_versions": {"python_version": "3.12.7", "jax_version": "0.11.1"},
        },
    )
    _write_json(
        os.path.join(v2_root, "step1", "deepseek4-284b_step1_result.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "step1_hf_to_gcs",
            "tpu_platform": "v7x",
            "step1_gcs_bucket_region": "us-central1",
            "model_name": "deepseek4-284b",
            "step1_hf_repo_id": "deepseek-ai/DeepSeek-V4-Flash",
            "step1_total_seconds": 300.0,
            "step1_download_hf_to_gcs_seconds": 299.0,
            "step1_files_already_in_gcs_count": 0,
            "step1_value_source": "cold_download",
            "step1_measured_by_run_id": "s1-ds284",
            "run_id": "s1-ds284",
            "run_status": "OK",
        },
    )
    _write_json(
        os.path.join(v2_root, "step1b", "deepseek4-284b_step1b_result.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "step1b_deepseek_dequantize",
            "tpu_platform": "v7x",
            "step1_gcs_bucket_region": "us-central1",
            "model_name": "deepseek4-284b",
            "step1b_deepseek_convert_fp8_fp4_to_bf16_seconds": 500.0,
            "step1b_deepseek_convert_input_gb": 159.6,
            "step1b_deepseek_convert_output_gb": 581.9,
            "step1b_deepseek_upload_bf16_to_gcs_seconds": 120.0,
            "step1b_deepseek_host_machine_type": "n2d-standard-224",
            "step1b_deepseek_bf16_source": "converted_by_this_benchmark",
            "step1b_deepseek_total_seconds": 620.0,
            "run_id": "s1b-ds284",
            "run_status": "OK",
        },
    )

    # 2. Three repetitions for llama3.1-8b scanned 1x2x2x1 on v7x for both pipelines -> median + min-max range.
    for rep, tm_load, sd_load in ((1, 12.0, 20.0), (2, 10.0, 18.0), (3, 14.0, 24.0)):
      base_dir = os.path.join(v2_root, "runs", "A", f"rep{rep}", "llama3.1-8b", "scanned")
      _write_json(
          os.path.join(base_dir, "to_maxtext_step1_result.json"),
          {
              "schema_version": schema_v2.SCHEMA_VERSION,
              "phase_name": "to_maxtext_step1",
              "pipeline_name": "to_maxtext",
              "tpu_platform": "v7x",
              "model_name": "llama3.1-8b",
              "scan_layers_mode": "scanned",
              "number_of_slices": 1,
              "slice_topology": "2x2x1",
              "repetition_index": rep,
              "run_id": f"tm-r{rep}",
              "run_status": "OK",
              "to_maxtext_step1_total_seconds": 25.0,
              "memory_peak_host_ram_container_gb": 18.0,
          },
      )
      _write_json(
          os.path.join(base_dir, "to_maxtext_step2_result.json"),
          {
              "schema_version": schema_v2.SCHEMA_VERSION,
              "phase_name": "to_maxtext_step2",
              "pipeline_name": "to_maxtext",
              "tpu_platform": "v7x",
              "model_name": "llama3.1-8b",
              "scan_layers_mode": "scanned",
              "number_of_slices": 1,
              "slice_topology": "2x2x1",
              "repetition_index": rep,
              "run_id": f"tm-r{rep}",
              "run_status": "OK",
              "to_maxtext_step2_total_seconds": 80.0,
              "memory_to_maxtext_step2_peak_host_ram_container_gb": 50.0,
              "memory_peak_host_ram_container_gb": 50.0,
          },
      )
      _write_json(
          os.path.join(base_dir, "to_maxtext_step3_result.json"),
          {
              "schema_version": schema_v2.SCHEMA_VERSION,
              "phase_name": "to_maxtext_step3",
              "pipeline_name": "to_maxtext",
              "tpu_platform": "v7x",
              "model_name": "llama3.1-8b",
              "scan_layers_mode": "scanned",
              "number_of_slices": 1,
              "slice_topology": "2x2x1",
              "repetition_index": rep,
              "run_id": f"tm-r{rep}",
              "run_status": "OK",
              "jax_backend_platform": "tpu",
              "tpu_device_kind": "TPU7x",
              "total_tpu_devices_measured": 8,
              "setup_total_before_load_seconds": 15.0,
              "to_maxtext_step3_total_seconds": tm_load,
              "memory_peak_host_ram_container_gb": 30.0,
              "memory_tpu_hbm_peak_per_device_gb": 2.2,
              "phase_started_at_utc": "2026-11-12T08:10:00Z",
              "phase_finished_at_utc": "2026-11-12T08:12:00Z",
              "step3_or_load_log_url": f"gs://mesa-maxtext-us-central1/mesa-bench-v2/runs/tm_r{rep}.log",
          },
      )
      _write_json(
          os.path.join(base_dir, "safetensors_dynamic_result.json"),
          {
              "schema_version": schema_v2.SCHEMA_VERSION,
              "phase_name": "safetensors_dynamic",
              "pipeline_name": "safetensors_dynamic",
              "tpu_platform": "v7x",
              "model_name": "llama3.1-8b",
              "scan_layers_mode": "scanned",
              "number_of_slices": 1,
              "slice_topology": "2x2x1",
              "repetition_index": rep,
              "run_id": f"sd-r{rep}",
              "run_status": "OK",
              "jax_backend_platform": "tpu",
              "tpu_device_kind": "TPU7x",
              "total_tpu_devices_measured": 8,
              "setup_total_before_load_seconds": 14.0,
              "safetensors_dynamic_step1_total_seconds": 0.4,
              "safetensors_dynamic_step2_total_seconds": sd_load,
              "memory_peak_host_ram_container_gb": 42.0,
              "memory_tpu_hbm_peak_per_device_gb": 3.5,
              "phase_started_at_utc": "2026-11-12T08:15:00Z",
              "phase_finished_at_utc": "2026-11-12T08:16:00Z",
              "step3_or_load_log_url": f"gs://mesa-maxtext-us-central1/mesa-bench-v2/runs/sd_r{rep}.log",
          },
      )

    # 3. Failed to_maxtext Step 2 (OOM-HOST) for deepseek4-284b must mark Step 3 BLOCKED.
    _write_json(
        os.path.join(v2_root, "runs", "C", "ds284", "deepseek4-284b", "scanned", "to_maxtext_step2_result.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "to_maxtext_step2",
            "pipeline_name": "to_maxtext",
            "tpu_platform": "v7x",
            "model_name": "deepseek4-284b",
            "scan_layers_mode": "scanned",
            "number_of_slices": 1,
            "slice_topology": "2x2x1",
            "repetition_index": 1,
            "run_id": "ds-tm-fail",
            "run_status": "FAILED",
            "failure_category": "OOM-HOST",
            "failed_stage_name": "to_maxtext_step2_transform_hooks_seconds",
            "error_summary": "MemoryError: host RAM exceeded",
            "memory_peak_host_ram_container_gb": 940.0,
            "step2_log_url": "gs://mesa-maxtext-us-central1/mesa-bench-v2/runs/ds_step2.log",
        },
    )

    impl_json = os.path.join(self.tmp_dir, "impl.json")
    with open(impl_json, "w", encoding="utf-8") as f:
      json.dump([{"task": "T4.1", "status": "Done", "commit": "abc1234", "notes": "tested"}], f)

    rc = build_sheet_v2.main(["--v2-root", v2_root, "--impl-json", impl_json, "--out-dir", out_dir])
    self.assertEqual(rc, 0)

    ds = build_sheet_v2.build_v2_dataset(v2_roots=[v2_root])
    self.assertEqual(ds.validate(), [])
    # Full planned v2 matrix must stay exact at 352 rows and group by Platform -> Model -> Pipeline -> Scan layers -> Slices.
    self.assertEqual(len(ds.summary_rows), 352)
    plat_blocks = []
    for r in ds.summary_rows:
      if not plat_blocks or plat_blocks[-1] != r["tpu_platform"]:
        plat_blocks.append(r["tpu_platform"])
    self.assertEqual(plat_blocks, ["v7x", "v6e", "v5e"])
    qwen06_v7x_groups = [
        (r["pipeline_name"], r["scan_layers_mode"], r["number_of_slices"])
        for r in ds.summary_rows
        if r["tpu_platform"] == "v7x" and r["model_name"] == "qwen3-0.6b"
    ]
    self.assertEqual(
        qwen06_v7x_groups,
        [
            ("to_maxtext", "scanned", 1),
            ("to_maxtext", "scanned", 2),
            ("to_maxtext", "unscanned", 1),
            ("to_maxtext", "unscanned", 2),
            ("safetensors_dynamic", "scanned", 1),
            ("safetensors_dynamic", "scanned", 2),
            ("safetensors_dynamic", "unscanned", 1),
            ("safetensors_dynamic", "unscanned", 2),
        ],
    )
    fmt_reqs = build_sheet_v2.build_summary_format_requests(ds.summary_rows)["requests"]
    merge_cols = {req["mergeCells"]["range"]["startColumnIndex"] for req in fmt_reqs if "mergeCells" in req}
    self.assertEqual(merge_cols, {0, 1, 2, 3, 4})

    by_id = {r["row_id"]: r for r in ds.summary_rows}
    tm_row = by_id["v2|v7x|llama3.1-8b|scanned|to_maxtext|1x2x2x1"]
    sd_row = by_id["v2|v7x|llama3.1-8b|scanned|safetensors_dynamic|1x2x2x1"]
    self.assertEqual(tm_row["run_status"], "OK")
    self.assertEqual(tm_row["repetitions_count"], 3)
    self.assertEqual(tm_row["load_into_tpu_hbm_total_seconds"], 12.0)
    self.assertEqual(tm_row["load_time_range_min_max_seconds"], "10.00\u201314.00")
    self.assertEqual(tm_row["memory_peak_host_ram_container_gb"], 50.0)
    self.assertAlmostEqual(tm_row["total_without_hf_download_seconds"], 117.0)
    self.assertAlmostEqual(tm_row["total_end_to_end_seconds"], 157.4)

    self.assertEqual(sd_row["run_status"], "OK")
    self.assertEqual(sd_row["load_into_tpu_hbm_total_seconds"], 20.0)
    self.assertEqual(sd_row["load_time_range_min_max_seconds"], "18.00\u201324.00")
    self.assertAlmostEqual(sd_row["total_without_hf_download_seconds"], 20.4)
    self.assertAlmostEqual(sd_row["total_end_to_end_seconds"], 60.8)
    self.assertAlmostEqual(
        sd_row["speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic"], round(117.0 / 20.4, 4)
    )
    self.assertAlmostEqual(sd_row["speedup_end_to_end_to_maxtext_vs_safetensors_dynamic"], round(157.4 / 60.8, 4))
    self.assertAlmostEqual(
        sd_row["speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2"], round(12.0 / 20.0, 4)
    )

    ds_fail = by_id["v2|v7x|deepseek4-284b|scanned|to_maxtext|1x2x2x1"]
    self.assertEqual(ds_fail["run_status"], "OOM-HOST")
    self.assertEqual(ds_fail["load_into_tpu_hbm_total_seconds"], "BLOCKED")
    self.assertIn("storage.cloud.google.com", ds_fail["log_url"])

    step1_run = next(
        r for r in ds.runs_rows if r["phase_name"] == "step1_hf_to_gcs" and r["model_name"] == "llama3.1-8b"
    )
    self.assertEqual(step1_run["git_commit"], "deadbeef1234")
    self.assertEqual(step1_run["code_tarball_sha256"], "abcd1234")
    self.assertEqual(step1_run["jax_version"], "0.11.1")

    with open(os.path.join(out_dir, "01_raw_batch.json"), "r", encoding="utf-8") as f:
      raw_ops = json.load(f)
    with open(os.path.join(out_dir, "02_formulas_batch.json"), "r", encoding="utf-8") as f:
      formula_ops = json.load(f)
    self.assertTrue(any(op.get("range", "").startswith("Progress!D58:H58") for op in raw_ops))
    self.assertTrue(any("HYPERLINK" in str(op.get("data")) for op in formula_ops))

  def test_v7x_step1b_copy_fallback_and_step1_oom_ram_cell(self):
    v2_root = os.path.join(self.tmp_dir, "v2_copy_fallback")

    _write_json(
        os.path.join(v2_root, "step1", "deepseek4-284b_step1_result.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "step1_hf_to_gcs",
            "tpu_platform": "v7x",
            "step1_gcs_bucket_region": "us-central1",
            "model_name": "deepseek4-284b",
            "step1_total_seconds": 120.0,
            "run_status": "OK",
        },
    )
    _write_json(
        os.path.join(v2_root, "step1b", "deepseek4-284b_copy_result.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "step1b_deepseek_copy_region",
            "tpu_platform": "v7x",
            "step1_gcs_bucket_region": "us-central1",
            "model_name": "deepseek4-284b",
            "step1b_deepseek_total_seconds": 45.0,
            "run_status": "OK",
        },
    )
    _write_json(
        os.path.join(v2_root, "runs", "deepseek4-284b_sd.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "safetensors_dynamic",
            "pipeline_name": "safetensors_dynamic",
            "tpu_platform": "v7x",
            "model_name": "deepseek4-284b",
            "scan_layers_mode": "scanned",
            "number_of_slices": 1,
            "slice_topology": "2x2x1",
            "repetition_index": 1,
            "run_status": "OK",
            "safetensors_dynamic_step1_total_seconds": 0.01,
            "safetensors_dynamic_step2_total_seconds": 34.99,
            "setup_total_before_load_seconds": 10.0,
            "memory_peak_host_ram_container_gb": 60.0,
            "memory_tpu_hbm_peak_per_device_gb": 50.0,
        },
    )
    _write_json(
        os.path.join(v2_root, "runs", "qwen3_235b_tm_s1_oom.json"),
        {
            "schema_version": schema_v2.SCHEMA_VERSION,
            "phase_name": "to_maxtext_step1",
            "pipeline_name": "to_maxtext",
            "tpu_platform": "v7x",
            "model_name": "qwen3-235b-a22b",
            "scan_layers_mode": "scanned",
            "number_of_slices": 1,
            "slice_topology": "2x2x1",
            "repetition_index": 1,
            "run_status": "FAILED",
            "failure_category": "OOM-HOST",
            "error_summary": "Host RAM staging limit exceeded",
        },
    )

    ds = build_sheet_v2.build_v2_dataset(v2_roots=[v2_root])
    ds284 = next(
        r for r in ds.summary_rows if r["row_id"] == "v2|v7x|deepseek4-284b|scanned|safetensors_dynamic|1x2x2x1"
    )
    self.assertEqual(ds284["step1b_deepseek_total_seconds"], 45.0)
    self.assertAlmostEqual(float(ds284["total_end_to_end_seconds"]), 200.0, places=2)

    q235 = next(r for r in ds.summary_rows if r["row_id"] == "v2|v7x|qwen3-235b-a22b|scanned|to_maxtext|1x2x2x1")
    self.assertEqual(q235["run_status"], "OOM-HOST")
    self.assertEqual(q235["memory_peak_host_ram_container_gb"], "OOM-HOST")
    self.assertEqual(q235["memory_tpu_hbm_peak_per_device_gb"], "BLOCKED")


if __name__ == "__main__":
  unittest.main()
