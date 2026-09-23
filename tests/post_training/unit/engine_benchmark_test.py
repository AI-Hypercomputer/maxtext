# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `training_engine/engine_benchmark.py`, the engine's standalone harness.

Two kinds. The arithmetic ones pin the throughput accounting to numbers measured on hardware:
the harness is only useful if its tokens/s/chip means what `train.py`'s does. The end-to-end ones
run the script itself on a four-device CPU mesh with a tiny Qwen3.5 -- GDN and MoE layers, the
decoder router replay is supported on -- and each gate they rely on is also shown to fire on an
input built to trip it, because a check that has never failed proves nothing.

The end-to-end tests run the script in a subprocess: the CPU device count is read only at backend
initialization, which sibling modules have already triggered by the time pytest imports this one
(see `maxtext_engine_xaot_test.py`).
"""

# The payload helpers are module-private; these tests read them directly.
# pylint: disable=protected-access

import importlib.util
import json
import os
import subprocess
import sys
import types
from unittest import mock

from absl.testing import absltest
import numpy as np
import pytest

from maxtext.training_engine import engine_benchmark
from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_DEVICES = 4


class ThroughputAccountingTest(absltest.TestCase):
  """The throughput a report quotes, checked against two benchmarks measured on v7x."""

  def _rates(self, per_device_batch_size, accumulation, step_time_s, devices, chip_devices):
    config = types.SimpleNamespace(
        max_target_length=65536, per_device_batch_size=per_device_batch_size, gradient_accumulation_steps=accumulation
    )
    # The TFLOP count needs a whole model config; its division by the step time is the same as
    # the token count's, which the real helper computes here.
    with mock.patch.object(
        engine_benchmark.maxtext_utils, "calculate_tflops_training_per_device", return_value=(1.0, 0.0, 0.0)
    ), mock.patch.object(engine_benchmark.jax, "device_count", return_value=devices):
      return engine_benchmark.throughput(config, step_time_s, chip_devices)

  def test_reproduces_the_128_chip_benchmark(self):
    """128 v7x chips, MBS 64 x 16, 255.9 s per step: measured 2,048.8 tokens/s/chip."""
    rates = self._rates(0.25, 16, 255.9, devices=256, chip_devices=2)
    self.assertAlmostEqual(rates["tokens_per_sec_per_chip"], 2048.8, delta=0.05)
    self.assertEqual(rates["tokens_per_step"], 1024 * 65536)

  def test_reproduces_the_256_chip_benchmark(self):
    """256 v7x chips, MBS 128 x 8, 141.8 s per step: measured 1,849 tokens/s/chip."""
    rates = self._rates(0.25, 8, 141.8, devices=512, chip_devices=2)
    self.assertAlmostEqual(rates["tokens_per_sec_per_chip"], 1849.0, delta=0.5)

  def test_per_chip_is_not_per_device_on_v7x(self):
    """The failure this guards: counting a v7x core as a chip halves every per-chip number."""
    rates = self._rates(0.25, 16, 255.9, devices=256, chip_devices=1)
    self.assertNotAlmostEqual(rates["tokens_per_sec_per_chip"], 2048.8, delta=1.0)

  def test_mfu_is_per_chip_tflops_over_the_chip_peak(self):
    config = types.SimpleNamespace(max_target_length=8, per_device_batch_size=1, gradient_accumulation_steps=1)
    with mock.patch.object(
        engine_benchmark.maxtext_utils, "calculate_tflops_training_per_device", return_value=(50.0, 0.0, 0.0)
    ):
      rates = engine_benchmark.throughput(config, step_time_s=2.0, chip_devices=2, peak_per_chip=100.0)
    # 50 TFLOP per device per step / 2 s = 25 per device = 50 per chip, of a 100 peak.
    self.assertAlmostEqual(rates["mfu"], 0.5)


class DevicesPerChipTest(absltest.TestCase):

  def _device(self, coords, core=0, slice_index=0):
    return types.SimpleNamespace(coords=coords, core_on_chip=core, slice_index=slice_index)

  def test_two_cores_per_chip(self):
    devices = [self._device((x, 0, 0), core) for x in range(4) for core in range(2)]
    self.assertEqual(engine_benchmark.devices_per_chip(devices), 2)

  def test_one_device_per_chip(self):
    self.assertEqual(engine_benchmark.devices_per_chip([self._device((x, 0, 0)) for x in range(4)]), 1)

  def test_the_same_coordinates_in_two_slices_are_two_chips(self):
    devices = [self._device((0, 0, 0), slice_index=s) for s in range(2)]
    self.assertEqual(engine_benchmark.devices_per_chip(devices), 1)

  def test_devices_without_coordinates_count_one_each(self):
    self.assertEqual(engine_benchmark.devices_per_chip([types.SimpleNamespace(), types.SimpleNamespace()]), 1)


class CompileLogTest(absltest.TestCase):

  def test_a_retraced_kernel_is_flagged_and_helper_compiles_are_not(self):
    kernels = ["jit(fwd_bwd)", "jit(update)"]
    self.assertEqual(engine_benchmark.recompiled_kernels(kernels, ["jit(broadcast_in_dim)"]), [])
    self.assertEqual(engine_benchmark.recompiled_kernels(kernels, ["jit(update)", "jit(update)"]), ["jit(update)"])

  def test_it_sees_a_real_recompile(self):
    """The instrument itself, two-sided: a new input shape recompiles, a repeated one does not."""
    log = engine_benchmark.CompileLog()

    def kernel_under_test(x):
      return x * 2

    f = engine_benchmark.jax.jit(kernel_under_test)
    x3, x4 = np.ones(3, np.float32), np.ones(4, np.float32)
    log.phase = "compile"
    f(x3)
    log.phase = "repeat"
    f(x3)
    log.phase = "reshaped"
    f(x4)
    name = "jit(kernel_under_test)"
    self.assertIn(name, log.names("compile"))
    self.assertEqual(engine_benchmark.recompiled_kernels(log.names("compile"), log.names("repeat")), [])
    self.assertEqual(engine_benchmark.recompiled_kernels(log.names("compile"), log.names("reshaped")), [name])


class DevicePeakTest(absltest.TestCase):

  def test_device_bytes_counts_the_shard_not_the_global_array(self):
    mesh = engine_benchmark.jax.sharding.Mesh(np.array(engine_benchmark.jax.devices()[:1]), ("x",))
    sharded = engine_benchmark.jax.ShapeDtypeStruct(
        (8, 4),
        np.float32,
        sharding=engine_benchmark.jax.sharding.NamedSharding(mesh, engine_benchmark.jax.sharding.PartitionSpec("x")),
    )
    self.assertEqual(engine_benchmark.device_bytes({"a": sharded, "b": None}), 8 * 4 * 4)
    fake = types.SimpleNamespace(
        shape=(8, 4), dtype=np.float32, sharding=types.SimpleNamespace(shard_shape=lambda s: (1, 4))
    )
    self.assertEqual(engine_benchmark.device_bytes([fake]), 1 * 4 * 4)

  def test_peaks_add_what_each_kernel_does_not_see(self):
    kernels = {name: {"resident_gib": 10.0} for name in ("fwd_bwd", "fwd_bwd_accum", "update", "eval")}
    kernels["oom_kernel"] = {"oom": True}
    peaks = engine_benchmark.device_peaks(kernels, optimizer_gib=3.0, accumulator_gib=2.0, other_payloads_gib=0.5)
    self.assertEqual(peaks, {"fwd_bwd": 13.5, "fwd_bwd_accum": 13.5, "update": 10.5, "eval": 15.5})


class ParseOomTest(absltest.TestCase):

  def test_reads_the_messages_xla_prints(self):
    verbatim = (
        "RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory required for HLO temporaries (98.58G) "
        "exceeds available HBM (94.74G). HLO module: jit_fwd_bwd."
    )
    self.assertEqual(engine_benchmark.parse_oom(verbatim), {"temp_gib": 98.58, "hbm_gib_reported": 94.74})
    used = engine_benchmark.parse_oom("Ran out of memory in memory space hbm. Used 115.94G of 95.73G hbm.")
    self.assertEqual(used, {"temp_gib": 115.94, "hbm_gib_reported": 95.73})
    self.assertAlmostEqual(
        engine_benchmark.parse_oom("HLO temporaries (512M) exceeds available HBM (1T)")["temp_gib"], 0.5
    )

  def test_says_nothing_about_a_message_that_is_not_an_oom(self):
    self.assertEqual(engine_benchmark.parse_oom("INVALID_ARGUMENT: shapes do not match"), {})


class PayloadPiecesTest(absltest.TestCase):

  def test_token_range_excludes_pad_and_eos(self):
    self.assertEqual(engine_benchmark._token_range(248320, 248044, 248044), (0, 248044))
    self.assertEqual(engine_benchmark._token_range(128, 0, 1), (2, 128))
    with self.assertRaises(ValueError):
      engine_benchmark._token_range(2, 0, 1)

  def test_routed_experts_are_distinct_and_in_range(self):
    fill = engine_benchmark._routed_experts(num_experts=16)
    block = fill((slice(0, 2), slice(0, 8), slice(0, 3), slice(0, 4)), np.random.default_rng(0))
    self.assertEqual(block.shape, (2, 8, 3, 4))
    self.assertTrue(((block >= 0) & (block < 16)).all())
    self.assertTrue(all(len(set(row)) == 4 for row in block.reshape(-1, 4).tolist()))

  def test_positions_follow_the_global_sequence_offset(self):
    block = engine_benchmark._positions((slice(0, 2), slice(8, 12)), None)
    np.testing.assert_array_equal(block, [[8, 9, 10, 11], [8, 9, 10, 11]])


def _tiny_qwen35(*overrides: str) -> list[str]:
  """MaxText args for a Qwen3.5 small enough for a CPU: GDN and attention layers, 8 experts."""
  return [
      get_test_config_path("base.yml"),
      "model_name=qwen3.5-397b-a17b",
      "override_model_config=true",
      "run_name=engine_benchmark_test",
      "use_multimodal=false",
      "base_emb_dim=64",
      "base_num_decoder_layers=4",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      # Production's MRoPE path, shrunk: rotary_dim = head_dim * partial_rotary_factor(0.25) = 8,
      # and mrope_section must sum to rotary_dim / 2.
      "head_dim=32",
      "mrope_section=[2,1,1]",
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=8",
      "num_experts_per_tok=2",
      "shared_experts=1",
      "gdn_key_head_dim=16",
      "gdn_value_head_dim=16",
      "gdn_num_key_heads=2",
      "gdn_num_value_heads=4",
      "gdn_chunk_size=16",
      "sparse_matmul=false",
      "megablox=false",
      "attention=dot_product",
      "dtype=float32",
      "weight_dtype=float32",
      "scan_layers=true",
      "max_target_length=64",
      f"ici_fsdp_parallelism={_DEVICES}",
      "per_device_batch_size=1",
      "gradient_accumulation_steps=2",
      "opt_type=adamw",
      "learning_rate=1e-4",
      "warmup_steps_fraction=0.0",
      "enable_checkpointing=false",
      "enable_tensorboard=false",
      "skip_jax_distributed_system=true",
      *overrides,
  ]


def _run_script(tmp_dir: str, flags: list[str], maxtext_args: list[str]) -> tuple[subprocess.CompletedProcess, dict]:
  """Runs the harness on a four-device CPU mesh and returns the process and its JSON report."""
  report_path = os.path.join(tmp_dir, "report.json")
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_DEVICES}".strip()
  if "--mode=aot" in flags:
    # Compiling for a TPU topology needs the TPU plugin, which a CPU-only platform list hides.
    env.pop("JAX_PLATFORMS", None)
  else:
    env["JAX_PLATFORMS"] = "cpu"
  command = [sys.executable, "-m", "maxtext.training_engine.engine_benchmark", *flags, f"--report_path={report_path}"]
  result = subprocess.run(command + maxtext_args, env=env, capture_output=True, text=True, check=False)
  report = {}
  if os.path.exists(report_path):
    with open(report_path, encoding="utf-8") as f:
      report = json.load(f)
  return result, report


def _tail(result: subprocess.CompletedProcess) -> str:
  return f"rc={result.returncode}\nstdout tail:\n{result.stdout[-4000:]}\nstderr tail:\n{result.stderr[-6000:]}"


@pytest.mark.cpu_only
class LiveRunTest(absltest.TestCase):
  """The script end to end on CPU: steps, validation pass, log-prob scoring, and its gates."""

  def test_grpo_with_router_replay_runs_and_every_check_passes(self):
    flags = [
        "--mode=run",
        "--loss_type=grpo",
        "--router_replay",
        "--steps=2",
        "--prompt_length=16",
        "--eval_batches=2",
        "--logprob_batches=1",
    ]
    result, report = _run_script(self.create_tempdir().full_path, flags, _tiny_qwen35())
    self.assertEqual(result.returncode, 0, _tail(result))
    self.assertEqual(
        {name: c["status"] for name, c in report["checks"].items()},
        {
            "loss_finite": "PASS",
            "grad_norm_finite": "PASS",
            "updates_applied": "PASS",
            "no_kernel_recompile_in_timed_steps": "PASS",
            "run_eval": "PASS",
            "logprob_scoring": "PASS",
        },
    )
    self.assertLen(report["step_times_s"], 2)
    # micro-batch 4 (1 per device) x 2 accumulation steps x 64 tokens.
    self.assertEqual(report["tokens_per_step"], 4 * 2 * 64)
    self.assertEqual(report["eval_metrics"]["eval_batches"], 2)

  def test_sft_runs_and_every_check_passes(self):
    flags = ["--mode=run", "--loss_type=sft", "--steps=2", "--eval_batches=1"]
    result, report = _run_script(self.create_tempdir().full_path, flags, _tiny_qwen35())
    self.assertEqual(result.returncode, 0, _tail(result))
    self.assertTrue(all(c["status"] == "PASS" for c in report["checks"].values()), report["checks"])

  def test_profile_steps_traces_a_default_config(self):
    """`--profile_steps` writes a trace of whole steps, and base.yml's default profiler_steps (5) does not block it."""
    tmp = self.create_tempdir().full_path
    trace_dir = os.path.join(tmp, "trace")
    flags = [
        "--mode=run",
        "--loss_type=grpo",
        "--steps=2",
        "--prompt_length=16",
        "--profile_steps=1",
        f"--profile_dir={trace_dir}",
    ]
    result, report = _run_script(tmp, flags, _tiny_qwen35())
    self.assertEqual(result.returncode, 0, _tail(result))
    self.assertEqual(report["profile_dir"], trace_dir)
    traces = [f for _, _, files in os.walk(trace_dir) for f in files if f.endswith((".xplane.pb", ".trace.json.gz"))]
    self.assertNotEmpty(traces, f"no trace under {trace_dir}")
    # The traced step is outside the statistics: two timed steps, not three.
    self.assertLen(report["step_times_s"], 2)

  def test_profile_steps_refuses_the_engines_own_profiler(self):
    """With MaxText's profiler also on, the engine would open a trace inside the harness's."""
    flags = ["--mode=run", "--loss_type=grpo", "--steps=1", "--prompt_length=16", "--profile_steps=1"]
    result, _ = _run_script(self.create_tempdir().full_path, flags, _tiny_qwen35("profiler=xplane"))
    self.assertNotEqual(result.returncode, 0, _tail(result))
    self.assertIn("not both", result.stderr + result.stdout)

  def test_grpo_refuses_vocab_tiling(self):
    """Vocab tiling leaves the decoder with no logits for the GRPO loss; the harness must say so."""
    flags = ["--mode=run", "--loss_type=grpo", "--steps=1", "--prompt_length=16"]
    result, _ = _run_script(self.create_tempdir().full_path, flags, _tiny_qwen35("num_vocab_tiling=2"))
    self.assertNotEqual(result.returncode, 0, _tail(result))
    self.assertIn("needs num_vocab_tiling=1", result.stderr + result.stdout)

  def _first_step(self, *extra_flags: str, overrides: tuple[str, ...] = ()) -> dict:
    flags = ["--mode=run", "--loss_type=grpo", "--steps=1", "--warmup_steps=0", "--prompt_length=16", *extra_flags]
    result, report = _run_script(self.create_tempdir().full_path, flags, _tiny_qwen35(*overrides))
    self.assertEqual(result.returncode, 0, _tail(result))
    return report

  def test_chunked_logps_give_the_unchunked_loss(self):
    """Chunking changes memory, never the answer: same data and weights, same loss and gradient norm."""
    full = self._first_step()
    chunked = self._first_step("--compute_logps_chunk_size=16")
    self.assertEqual(chunked["compute_logps_chunk_size"], 16)
    self.assertAlmostEqual(chunked["loss"][0], full["loss"][0], delta=1e-4 * max(1.0, abs(full["loss"][0])))
    self.assertAlmostEqual(chunked["grad_norm"][0], full["grad_norm"][0], delta=1e-3 * max(1.0, full["grad_norm"][0]))

  def test_chunked_logps_run_under_vocab_tiling(self):
    """Chunked, the model hands back hidden states under vocab tiling too, and the loss is unchanged."""
    full = self._first_step()
    tiled = self._first_step("--compute_logps_chunk_size=16", overrides=("num_vocab_tiling=2",))
    self.assertAlmostEqual(tiled["loss"][0], full["loss"][0], delta=1e-4 * max(1.0, abs(full["loss"][0])))

  def test_a_diverging_run_fails_the_loss_gate(self):
    """Positive control for the finiteness gates: a learning rate of 1e30 must be caught."""
    flags = ["--mode=run", "--loss_type=grpo", "--steps=3", "--prompt_length=16"]
    result, report = _run_script(self.create_tempdir().full_path, flags, _tiny_qwen35("learning_rate=1e30"))
    self.assertNotEqual(result.returncode, 0, _tail(result))
    statuses = {name: c["status"] for name, c in report.get("checks", {}).items()}
    self.assertIn("FAIL", statuses.values(), f"no gate fired on a diverging run: {statuses}\n{_tail(result)}")
    # Once the gradients go non-finite, `skip_step_on_nan` refuses the update, and the report must say so.
    self.assertEqual(statuses.get("updates_applied"), "FAIL", f"{statuses}\n{report.get('step_skipped')}")


@pytest.mark.cpu_only
@pytest.mark.skipif(importlib.util.find_spec("libtpu") is None, reason="AOT for a TPU topology needs libtpu.")
class AotTest(absltest.TestCase):
  """`--mode=aot` compiles all four kernels, eval included, and its fit verdict can say no."""

  def _aot(self, *args: str) -> tuple[subprocess.CompletedProcess, dict]:
    flags = [a for a in args if a.startswith("--")]
    overrides = [a for a in args if not a.startswith("--")]
    maxtext_args = _tiny_qwen35("compile_topology=v6e-4", "compile_topology_num_slices=1", *overrides)
    return _run_script(self.create_tempdir().full_path, ["--mode=aot", "--loss_type=grpo", *flags], maxtext_args)

  def test_compiles_every_kernel_including_eval(self):
    result, report = self._aot("--router_replay", "--prompt_length=16", "--hbm_gib_per_device=31.25")
    self.assertEqual(result.returncode, 0, _tail(result))
    self.assertEqual(set(report["kernels"]), {"fwd_bwd", "fwd_bwd_accum", "update", "eval"})
    self.assertEqual(report["device_peak_kernels"], ["eval", "fwd_bwd", "fwd_bwd_accum", "update"])
    self.assertTrue(all(k["resident_gib"] > 0 for k in report["kernels"].values()), report["kernels"])
    self.assertTrue(report["fits"])
    # adamw keeps two moments; with fp32 weights they are 8 bytes a parameter, so the optimizer
    # state is non-zero and the device peak must exceed the largest kernel's own resident.
    self.assertGreater(report["resident_outside_kernels_gib"]["optimizer_state"], 0.0)
    self.assertGreater(report["device_peak_gib"], report["peak_resident_gib"])
    # Cross-check of the sizing itself: adamw's two moments at fp32 weights are twice the weights.
    ratio = report["resident_outside_kernels_gib"]["optimizer_state"] / report["parameters_gib"]
    self.assertAlmostEqual(ratio, 2.0, delta=0.01)

  def test_compiles_under_the_production_cp_as_ep_rules(self):
    """Qwen3.5 under `cp-as-ep`, whose rules leave `norm` unmapped.

    The abstract engine's graph trace used to re-derive shardings through flax's own rule lookup,
    which raises on that (and, under the default rules, on `fsdp_transpose` claimed twice).
    """
    result, report = self._aot("--prompt_length=16", "--hbm_gib_per_device=31.25", "custom_mesh_and_rule=cp-as-ep")
    self.assertEqual(result.returncode, 0, _tail(result))
    self.assertEqual(set(report["kernels"]), {"fwd_bwd", "fwd_bwd_accum", "update", "eval"})

  def test_the_fit_verdict_can_fail(self):
    result, report = self._aot("--prompt_length=16", "--hbm_gib_per_device=0.000001")
    self.assertEqual(result.returncode, 0, _tail(result))
    self.assertFalse(report["fits"])


if __name__ == "__main__":
  absltest.main()
