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

"""Tests checkpoint_load_benchmark/bench/schema_v2.py and configs/models_v2.json."""

import os
import sys
import unittest

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import schema_v2  # pylint: disable=wrong-import-position


class SchemaV2Test(unittest.TestCase):
  """Verifies v2 schema column contracts, validations, and models_v2.json matrix."""

  def test_column_keys_are_unique_across_each_tab(self):
    for tab_name, cols in schema_v2.DATA_TABS:
      keys = [c.key for c in cols]
      self.assertEqual(len(keys), len(set(keys)), f"Duplicate keys in {tab_name}")
      self.assertGreater(len(keys), 20, tab_name)

  def test_make_row_id_and_status_normalization(self):
    self.assertEqual(
        schema_v2.make_row_id("v2", "v7x", "qwen3-0.6b", "scanned", "to_maxtext", 2, "2x2x1"),
        "v2|v7x|qwen3-0.6b|scanned|to_maxtext|2x2x2x1",
    )
    self.assertEqual(schema_v2.normalize_sheet_status("OK"), "OK")
    self.assertEqual(schema_v2.normalize_sheet_status("NOT_RUN"), "NOT RUN")
    self.assertEqual(schema_v2.normalize_sheet_status("FAILED", "OOM-HBM"), "OOM-HBM")
    self.assertEqual(schema_v2.normalize_sheet_status("FAILED", None), "ERROR")

  def test_numeric_cell_validation_and_error_detection(self):
    self.assertTrue(schema_v2.is_valid_numeric_cell(12.34))
    self.assertTrue(schema_v2.is_valid_numeric_cell("LOCAL"))
    self.assertTrue(schema_v2.is_valid_numeric_cell("NOT RUN"))
    self.assertFalse(schema_v2.is_valid_numeric_cell("hello"))

    bad_row = {c.key: "" for c in schema_v2.SUMMARY_COLUMNS}
    bad_row["step1_total_seconds"] = "invalid_text"
    errs = schema_v2.validate_row_dict(bad_row, schema_v2.SUMMARY_COLUMNS)
    self.assertTrue(any("step1_total_seconds" in e for e in errs))

  def test_validate_step1_record_enforces_zero_existing_blobs_for_cold(self):
    good = {
        "tpu_platform": "v6e",
        "step1_gcs_bucket_region": "southamerica-west1",
        "model_name": "qwen3-0.6b",
        "step1_hf_repo_id": "Qwen/Qwen3-0.6B",
        "step1_files_already_in_gcs_count": 0,
        "step1_total_seconds": 14.2,
        "step1_value_source": "cold_download",
        "step1_measured_by_run_id": "s1-v6e-01",
    }
    self.assertEqual(schema_v2.validate_step1_record(good), [])

    warm_impostor = dict(good, step1_files_already_in_gcs_count=3)
    self.assertTrue(schema_v2.validate_step1_record(warm_impostor))

  def test_validate_v2_phase_record_requires_tpu_backend_for_load_ok(self):
    rec = {
        "schema_version": 2,
        "phase_name": "safetensors_dynamic",
        "pipeline_name": "safetensors_dynamic",
        "model_name": "qwen3-0.6b",
        "scan_layers_mode": "scanned",
        "tpu_platform": "v7x",
        "run_status": "OK",
        "jax_backend_platform": "cpu",
        "total_tpu_devices": 8,
        "total_tpu_devices_measured": 8,
    }
    errs = schema_v2.validate_v2_phase_record(rec)
    self.assertTrue(any("jax_backend_platform" in e for e in errs))

    rec["jax_backend_platform"] = "tpu"
    self.assertEqual(schema_v2.validate_v2_phase_record(rec), [])

  def test_models_v2_planned_matrix_counts(self):
    cfg = schema_v2.load_models_v2()
    self.assertEqual(len(cfg["models"]), 21)
    planned = list(schema_v2.iter_planned_matrix(cfg))
    self.assertEqual(len(planned), 352)
    by_platform = {}
    models_per_platform = {}
    for item in planned:
      by_platform[item["platform"]] = by_platform.get(item["platform"], 0) + 1
      models_per_platform.setdefault(item["platform"], set()).add(item["model_name"])
    self.assertEqual(by_platform, {"v7x": 164, "v6e": 140, "v5e": 48})
    self.assertEqual(
        {k: len(v) for k, v in models_per_platform.items()},
        {"v7x": 21, "v6e": 21, "v5e": 12},
    )


if __name__ == "__main__":
  unittest.main()
