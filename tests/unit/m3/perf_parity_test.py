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

"""Unit and end-to-end CPU tests for the m3 vs. legacy performance parity harness."""

import dataclasses
import json
import math
import os

from absl.testing import absltest
import jax

from maxtext.m3.train import perf_parity
from tests.utils.test_helpers import get_test_config_path


def _tiny_qwen3_argv(output_dir: str, **overrides: str) -> list[str]:
  """Builds a tiny 2-layer qwen3-0.6b argv suitable for fast 4-device CPU tests."""
  args = {
      "model_name": "qwen3-0.6b",
      "override_model_config": "true",
      "base_num_decoder_layers": "2",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": "256",
      "per_device_batch_size": "2",
      "max_target_length": "32",
      "ici_fsdp_parallelism": str(len(jax.devices())),
      "scan_layers": "true",
      "base_output_directory": output_dir,
      "run_name": "perf_parity_test",
  }
  args.update({k: str(v) for k, v in overrides.items()})
  return ["perf_parity_test", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()]


def _make_result(
    label: str,
    *,
    num_params: int = 1000,
    tok_s: float | None = 1000.0,
    step_s: float | None = 0.01,
    compiled_total: int = 100 * 1024**2,
    compiled_peak: int | None = 120 * 1024**2,
    last_loss: float | None = 5.0,
) -> perf_parity.BackendResult:
  """Constructs a synthetic BackendResult for unit-testing gate evaluation and formatting."""
  return perf_parity.BackendResult(
      label=label,
      driver="train",
      model_name="qwen3-0.6b",
      num_devices=4,
      platform="cpu",
      num_params=num_params,
      compile_time_s=1.0,
      compiled_memory=perf_parity.CompiledMemoryStats(
          argument_bytes=compiled_total // 2,
          output_bytes=compiled_total // 2,
          temp_bytes=compiled_total // 2,
          alias_bytes=compiled_total // 2,
          host_temp_bytes=0,
          compiled_peak_bytes=compiled_peak,
      ),
      hlo_collectives=perf_parity.HloCollectives(all_gather=4, reduce_scatter=2, all_reduce=1, all_to_all=0),
      warmup_steps=1,
      measure_steps=3 if tok_s is not None else 0,
      mean_step_time_s=step_s,
      median_step_time_s=step_s,
      tokens_per_sec_per_device=tok_s,
      tflops_per_sec_per_device=10.0 if tok_s is not None else None,
      mfu=0.5 if tok_s is not None else None,
      first_loss=last_loss,
      last_loss=last_loss,
  )


class PerfParityUnitTest(absltest.TestCase):
  """Tests argument rewriting, HLO parsing, gate evaluation, and report formatting."""

  def test_override_argv_replaces_existing_keys_and_preserves_others(self):
    base = [
        "prog",
        "base.yml",
        "model_name=qwen3-0.6b",
        "use_m3_model=false",
        "eval_interval=10",
        "scan_layers=true",
    ]
    out = perf_parity.override_argv(base, {"use_m3_model": "true", "eval_interval": "0"})
    self.assertEqual(
        out,
        [
            "prog",
            "base.yml",
            "model_name=qwen3-0.6b",
            "scan_layers=true",
            "use_m3_model=true",
            "eval_interval=0",
        ],
    )

  def test_override_argv_rejects_missing_config_path(self):
    with self.assertRaisesRegex(ValueError, "Expected at least a program name and config YAML path"):
      perf_parity.override_argv(["prog"], {"use_m3_model": "true"})

  def test_count_hlo_collectives_handles_sync_and_async_without_double_counting_done(self):
    hlo = """
      %ag0 = bf16[4,64] all-gather(bf16[1,64] %p0), dimensions={0}
      %ag1 = (bf16[1,64], bf16[4,64]) all-gather-start(bf16[1,64] %p1), dimensions={0}
      %ag1.done = bf16[4,64] all-gather-done((bf16[1,64], bf16[4,64]) %ag1)
      %rs0 = bf16[1,64] reduce-scatter(bf16[4,64] %g0), to_apply=%add
      %rs1 = (bf16[4,64], bf16[1,64]) reduce-scatter-start(bf16[4,64] %g1), to_apply=%add
      %rs1.done = bf16[1,64] reduce-scatter-done((bf16[4,64], bf16[1,64]) %rs1)
      %ar0 = f32[] all-reduce(f32[] %loss), to_apply=%add
      %a2a0 = (bf16[4,8], bf16[4,8]) all-to-all-start(bf16[4,8] %tok)
      %a2a0.done = bf16[4,8] all-to-all-done((bf16[4,8], bf16[4,8]) %a2a0)
    """
    counts = perf_parity.count_hlo_collectives(hlo)
    self.assertEqual(counts, perf_parity.HloCollectives(all_gather=2, reduce_scatter=2, all_reduce=1, all_to_all=1))
    self.assertEqual(counts.total, 6)

  def test_evaluate_parity_passes_at_thresholds_and_fails_on_regressions(self):
    legacy = _make_result("legacy", tok_s=1000.0, compiled_peak=100 * 1024**2)
    m3_ok = _make_result("m3", tok_s=980.0, compiled_peak=102 * 1024**2)
    report_ok = perf_parity.evaluate_parity(legacy, m3_ok)
    self.assertTrue(report_ok.passed, perf_parity.format_report(report_ok))
    self.assertAlmostEqual(report_ok.throughput_ratio, 0.98)
    self.assertAlmostEqual(report_ok.hbm_ratio, 1.02)
    self.assertEqual(report_ok.hbm_metric_name, "compiled_peak_hbm")

    m3_slow = _make_result("m3", tok_s=950.0, compiled_peak=100 * 1024**2)
    report_slow = perf_parity.evaluate_parity(legacy, m3_slow)
    self.assertFalse(report_slow.passed)
    by_name = {o.name: o.passed for o in report_slow.outcomes}
    self.assertFalse(by_name["throughput"])
    self.assertTrue(by_name["hbm"])

    m3_bloated = _make_result("m3", tok_s=1000.0, compiled_peak=105 * 1024**2)
    report_bloated = perf_parity.evaluate_parity(legacy, m3_bloated)
    self.assertFalse(report_bloated.passed)
    self.assertFalse({o.name: o.passed for o in report_bloated.outcomes}["hbm"])

    m3_nan = _make_result("m3", num_params=999, last_loss=float("nan"))
    report_nan = perf_parity.evaluate_parity(legacy, m3_nan)
    by_name_nan = {o.name: o.passed for o in report_nan.outcomes}
    self.assertFalse(by_name_nan["param_count"])
    self.assertFalse(by_name_nan["finite_loss"])

  def test_hbm_gate_omitted_when_compiled_memory_missing(self):
    legacy = dataclasses.replace(_make_result("legacy"), compiled_memory=None)
    m3 = dataclasses.replace(_make_result("m3"), compiled_memory=None)
    report = perf_parity.evaluate_parity(legacy, m3)
    self.assertIsNone(report.hbm_metric_name)
    self.assertIsNone(report.hbm_ratio)
    self.assertNotIn("hbm", {o.name for o in report.outcomes})
    self.assertTrue(report.passed)

  def test_hbm_gate_ignores_partial_compiled_peak_smaller_than_temp(self):
    """CPU XLA reports peak_memory_in_bytes < temp_bytes; gate must use total_estimated_bytes."""
    legacy = _make_result("legacy", compiled_total=100 * 1024**2, compiled_peak=10 * 1024**2)
    m3 = _make_result("m3", compiled_total=175 * 1024**2, compiled_peak=10 * 1024**2)
    report = perf_parity.evaluate_parity(legacy, m3)
    self.assertEqual(report.hbm_metric_name, "compiled_total_hbm")
    self.assertAlmostEqual(report.hbm_ratio, 1.75)
    self.assertFalse({o.name: o.passed for o in report.outcomes}["hbm"])

  def test_format_report_and_write_json_round_trip(self):
    report = perf_parity.evaluate_parity(_make_result("legacy"), _make_result("m3"))
    text = perf_parity.format_report(report)
    self.assertIn("Throughput (tok/s/dev)", text)
    self.assertIn("Gates overall: PASS", text)
    out_path = os.path.join(self.create_tempdir().full_path, "nested", "report.json")
    perf_parity.write_report_json(report, out_path)
    with open(out_path, encoding="utf-8") as f:
      payload = json.load(f)
    self.assertTrue(payload["passed"])
    self.assertIn("total_estimated_bytes", payload["m3"]["compiled_memory"])

  def test_rejects_invalid_driver_and_parameters(self):
    argv = _tiny_qwen3_argv(self.create_tempdir().full_path)
    with self.assertRaisesRegex(ValueError, "m3_driver must be 'loop' or 'train'"):
      perf_parity.run_parity_benchmark(argv, m3_driver="invalid")
    with self.assertRaisesRegex(ValueError, "warmup_steps must be >= 0"):
      perf_parity.run_parity_benchmark(argv, warmup_steps=-1)
    with self.assertRaisesRegex(ValueError, "measure_steps must be >= 1"):
      perf_parity.run_parity_benchmark(argv, measure_steps=0, compile_only=False)
    with self.assertRaisesRegex(ValueError, "peak_tflops must be > 0"):
      perf_parity.run_parity_benchmark(argv, peak_tflops=0.0)
    with self.assertRaisesRegex(ValueError, "m3 loop driver does not support gradient accumulation"):
      perf_parity.run_parity_benchmark(
          _tiny_qwen3_argv(self.create_tempdir().full_path, gradient_accumulation_steps="2"),
          m3_driver="loop",
          compile_only=True,
      )


class PerfParityEndToEndTest(absltest.TestCase):
  """Runs tiny Qwen3 side-by-side comparisons end-to-end on CPU."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_train_driver_end_to_end_fsdp(self):
    report = perf_parity.run_parity_benchmark(
        _tiny_qwen3_argv(self.output_dir),
        m3_driver="train",
        warmup_steps=1,
        measure_steps=2,
        peak_tflops=100.0,
    )
    self.assertEqual(report.legacy.num_params, report.m3.num_params)
    self.assertGreater(report.m3.num_params, 0)
    self.assertTrue(math.isfinite(report.legacy.last_loss))
    self.assertTrue(math.isfinite(report.m3.last_loss))
    self.assertGreater(report.legacy.tokens_per_sec_per_device, 0.0)
    self.assertGreater(report.m3.tokens_per_sec_per_device, 0.0)
    self.assertGreater(report.m3.mfu, 0.0)
    self.assertIsNotNone(report.legacy.compiled_memory)
    self.assertIsNotNone(report.m3.compiled_memory)
    self.assertGreater(report.m3.compiled_memory.total_estimated_bytes, 0)
    if len(jax.devices()) > 1:
      self.assertGreater(report.legacy.hlo_collectives.all_gather, 0)
      self.assertGreater(report.m3.hlo_collectives.all_gather, 0)

  def test_default_loop_driver_end_to_end_fsdp(self):
    report = perf_parity.run_parity_benchmark(
        _tiny_qwen3_argv(self.output_dir),
        warmup_steps=1,
        measure_steps=2,
    )
    self.assertEqual(report.legacy.driver, "train")
    self.assertEqual(report.m3.driver, "loop")
    self.assertEqual(report.legacy.num_params, report.m3.num_params)
    self.assertTrue(math.isfinite(report.m3.last_loss))
    self.assertGreater(report.m3.tokens_per_sec_per_device, 0.0)
    self.assertIsNotNone(report.m3.compiled_memory)

  def test_compile_only_skips_step_execution(self):
    report = perf_parity.run_parity_benchmark(
        _tiny_qwen3_argv(self.output_dir),
        compile_only=True,
    )
    self.assertEqual(report.legacy.driver, "train")
    self.assertEqual(report.m3.driver, "loop")
    self.assertEqual(report.legacy.measure_steps, 0)
    self.assertEqual(report.m3.measure_steps, 0)
    self.assertIsNone(report.m3.mean_step_time_s)
    self.assertIsNone(report.m3.last_loss)
    self.assertIsNone(report.throughput_ratio)
    self.assertIsNotNone(report.hbm_ratio)
    self.assertEqual(report.legacy.num_params, report.m3.num_params)
    gate_names = {o.name for o in report.outcomes}
    self.assertEqual(gate_names, {"param_count", "hbm"})


if __name__ == "__main__":
  absltest.main()
