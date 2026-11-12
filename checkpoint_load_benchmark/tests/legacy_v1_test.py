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

"""Tests checkpoint_load_benchmark/bench/legacy_v1.py."""

import json
import os
import shutil
import sys
import tempfile
import unittest

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import legacy_v1  # pylint: disable=wrong-import-position


def _write_json(root: str, model: str, cfg: str, filename: str, payload: dict) -> str:
  d = os.path.join(root, model, cfg)
  os.makedirs(d, exist_ok=True)
  path = os.path.join(d, filename)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
  return path


def _write_text(root: str, model: str, cfg: str, filename: str, content: str) -> None:
  d = os.path.join(root, model, cfg)
  os.makedirs(d, exist_ok=True)
  with open(os.path.join(d, filename), "w", encoding="utf-8") as f:
    f.write(content)


class LegacyV1AdapterTest(unittest.TestCase):
  """Golden-style tests mapping v1 JSONs, logs, and CSVs into validated v2 tables."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir)

  def test_adapts_real_synthetic_multihost_and_failures_honestly(self):
    p2_root = os.path.join(self.tmp_dir, "p2_logs")
    p1_root = os.path.join(self.tmp_dir, "p1_logs")
    before_root = os.path.join(self.tmp_dir, "before_logs")

    # 1. Real P2 model (`llama3.1-8b` scanned 1 slice OK on both pipelines).
    _write_json(
        p2_root, "llama3.1-8b", "scanned", "to_maxtext_step1_result.json", {"hf_to_gcs_s": 62.9, "hf_to_local_s": 44.95}
    )
    _write_json(
        p2_root,
        "llama3.1-8b",
        "scanned",
        "to_maxtext_step2_result.json",
        {
            "convert_to_orbax_s": 120.0,
            "init_abstract_model_s": 5.0,
            "transform_and_shard_s": 85.0,
            "orbax_save_to_gcs_s": 30.0,
            "peak_host_ram_gb": 90.0,
            "total_host_ram_gb": 1014.5,
        },
    )
    _write_json(
        p2_root,
        "llama3.1-8b",
        "scanned",
        "to_maxtext_step3_result.json",
        {"load_orbax_s": 10.0, "init_abstract_state_s": 2.0, "global_devices": 8, "peak_host_ram_gb": 40.0},
    )
    _write_json(
        p2_root,
        "llama3.1-8b",
        "scanned",
        "safetensors_dynamic_result.json",
        {
            "hf_to_gcs_s": 0.2,
            "safetensors_and_convert_s": 30.0,
            "derive_mappings_s": 0.01,
            "load_sharded_hf_s": 20.0,
            "transform_hf_to_mt_s": 9.9,
            "global_devices": 8,
            "peak_host_ram_gb": 55.0,
            "total_host_ram_gb": 1014.5,
        },
    )

    # 2. Failed P2 model (`gpt-oss-120b` HBM OOM logged inside `.log`).
    _write_json(
        p2_root,
        "gpt-oss-120b",
        "scanned",
        "safetensors_dynamic_result.json",
        {"status": "FAILED", "error": "Worker phase 'safetensors_dynamic' failed with exit code 1", "global_devices": 8},
    )
    _write_text(
        p2_root, "gpt-oss-120b", "scanned", "safetensors_dynamic.log", "RESOURCE_EXHAUSTED: HLO temporaries (200.32G)\n"
    )

    # 3. Synthetic DeepSeek model in P1 (`deepseek4-284b` -> LOCAL in Step 1, synthetic_sparse).
    _write_json(
        p1_root, "deepseek4-284b", "scanned", "to_maxtext_step1_result.json", {"hf_to_gcs_s": 0.45, "hf_to_local_s": 0.25}
    )
    _write_json(
        p1_root,
        "deepseek4-284b",
        "scanned",
        "to_maxtext_step2_result.json",
        {
            "status": "FAILED",
            "error": "Worker phase 'to_maxtext_step2' failed with exit code -9 (OOMKilled / Exit Code 137)",
        },
    )
    _write_json(
        p1_root,
        "deepseek4-284b",
        "scanned",
        "safetensors_dynamic_result.json",
        {
            "hf_to_gcs_s": 0.0,
            "safetensors_and_convert_s": 453.27,
            "load_sharded_hf_s": 366.34,
            "transform_hf_to_mt_s": 86.85,
            "global_devices": 8,
            "peak_host_ram_gb": 963.4,
        },
    )

    # 4. Multi-host CSV subset (`qwen3-0.6b` + `deepseek4-284b`) + multi-host `to_maxtext_step3`.
    mh_csv = os.path.join(self.tmp_dir, "mh.csv")
    with open(mh_csv, "w", encoding="utf-8") as f:
      f.write(
          "model,scan_mode,layout,hosts,devices,hbm_gb,link,status,total_s,load_s,transform_s,peak_host_ram_gb,note,source\n"
      )
      f.write("llama3.1-8b,scanned,1 × 2x2x2,2,16,1627.6,ICI,OK,25.0,15.0,10.0,60.0,,lance\n")

    _write_json(
        p2_root,
        "llama3.1-8b",
        "scanned_hosts2",
        "to_maxtext_step3_result.json",
        {"load_orbax_s": 12.0, "global_devices": 16, "peak_host_ram_gb": 50.0},
    )
    _write_json(
        before_root,
        "deepseek4-284b",
        "scanned",
        "safetensors_dynamic_result.json",
        {"status": "FAILED", "error": "TIMEOUT: exceeded PHASE_TIMEOUT_S=1800s"},
    )

    ds = legacy_v1.adapt_v1_dataset(
        p2_roots=[p2_root],
        p1_roots=[p1_root],
        before_roots=[before_root],
        multihost_csv=mh_csv,
    )
    self.assertEqual(ds.validate(), [])

    by_id = {r["row_id"]: r for r in ds.summary_rows}
    # Real OK row check + speedup check.
    sd_llama = by_id["legacy|v7x|llama3.1-8b|scanned|safetensors_dynamic|1x2x2x1"]
    self.assertEqual(sd_llama["run_status"], "OK")
    self.assertEqual(sd_llama["checkpoint_weights_kind"], "real")
    self.assertEqual(sd_llama["load_backend_and_tpu_device_kind"], "")
    self.assertAlmostEqual(sd_llama["speedup_end_to_end_to_maxtext_vs_safetensors_dynamic"], 2.56, places=2)

    # Multi-host paired TM + SD check.
    self.assertIn("legacy|v7x|llama3.1-8b|scanned|to_maxtext|1x2x2x2", by_id)
    sd_mh = by_id["legacy|v7x|llama3.1-8b|scanned|safetensors_dynamic|1x2x2x2"]
    self.assertAlmostEqual(sd_mh["speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic"], 7.08, places=2)

    # Synthetic DeepSeek must mark LOCAL (never 0.00) and synthetic_sparse.
    ds_row = by_id["legacy|v7x|deepseek4-284b|scanned|safetensors_dynamic|1x2x2x1"]
    self.assertEqual(ds_row["checkpoint_weights_kind"], "synthetic_sparse")
    self.assertEqual(ds_row["step1_total_seconds"], "LOCAL")
    self.assertEqual(ds_row["total_end_to_end_seconds"], "LOCAL")

    # Failure categorization from log & error string.
    self.assertEqual(by_id["legacy|v7x|gpt-oss-120b|scanned|safetensors_dynamic|1x2x2x1"]["run_status"], "OOM-HBM")
    self.assertEqual(by_id["legacy|v7x|deepseek4-284b|scanned|to_maxtext|1x2x2x1"]["run_status"], "OOM-HOST")

    # Stage rows leave unmeasured fine-grained columns blank and label RAM host-0 only.
    sd_stage = next(r for r in ds.stages_sd_rows if r["row_id"] == sd_llama["row_id"])
    self.assertEqual(sd_stage["safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds"], "")
    self.assertEqual(sd_stage["memory_peak_host_ram_host_name"], "host-0 (v1 only recorded host 0)")

    # Step 1 rows preserve cold real seconds and LOCAL for synthetic DeepSeek.
    step1_by_model = {r["model_name"]: r for r in ds.step1_rows}
    self.assertEqual(step1_by_model["deepseek4-284b"]["step1_total_seconds"], "LOCAL")
    self.assertEqual(step1_by_model["llama3.1-8b"]["step1_total_seconds"], 62.9)


if __name__ == "__main__":
  unittest.main()
