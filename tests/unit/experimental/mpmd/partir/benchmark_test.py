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

"""Tests for the MPMD pipeline's step statistics, throughput and trace (benchmark.py, CPU only)."""

import sys
import types
import unittest
from unittest import mock

import jax.numpy as jnp

from maxtext.experimental.mpmd.partir import benchmark
from maxtext.experimental.mpmd.partir import config
from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.utils import maxtext_utils


def loss_step(losses):
  """A step function that returns the next of `losses` and leaves the state alone."""

  def step(state, batch):
    del batch
    return state, jnp.float32(next(losses))

  return step


class StepStatsTest(unittest.TestCase):

  def test_first_and_final_loss(self):
    _, stats = benchmark.run_steps(loss_step(iter([5.0, 4.0, 3.0, 2.0])), 0, None, num_steps=2, warmup_steps=2)
    self.assertEqual((stats.first_loss, stats.loss, stats.num_steps), (5.0, 2.0, 2))
    _, stats = benchmark.run_steps(loss_step(iter([3.0, 1.0])), 0, None, num_steps=2)
    self.assertEqual((stats.first_loss, stats.loss), (3.0, 1.0))

  def test_abort_on_the_first_non_finite_loss(self):
    step = loss_step(iter([5.0, 4.0, float("nan"), 2.0]))
    with self.assertRaises(benchmark.NonFiniteLossError) as raised:
      benchmark.run_steps(step, 0, None, num_steps=3, warmup_steps=1, abort_on_nonfinite=True)
    self.assertEqual((raised.exception.step, raised.exception.losses[:2]), (3, (5.0, 4.0)))
    self.assertIn("(losses from step 1: 5, 4, nan)", str(raised.exception))
    # without the flag it runs on
    _, stats = benchmark.run_steps(loss_step(iter([5.0, float("inf")])), 0, None, num_steps=2)
    self.assertEqual(stats.loss, float("inf"))

  def test_steps_numbered_after_an_earlier_call(self):
    """train.py's profiled steps after 1 warm-up and 3 timed steps: a NaN in the second is step 6."""
    step = loss_step(iter([4.47, float("nan")]))
    with self.assertRaises(benchmark.NonFiniteLossError) as raised:
      benchmark.run_steps(step, 0, None, num_steps=2, first_step=4, abort_on_nonfinite=True)
    self.assertEqual(raised.exception.step, 6)
    self.assertIn("step 6 returned a loss of nan (losses from step 5: 4.47, nan)", str(raised.exception))

  def test_loss_warning_only_when_the_loss_rose(self):
    falling = benchmark.StepStats(step_times_ms=(1.0,), loss=1.2, first_loss=24.8)
    self.assertEqual(benchmark.loss_warning(falling, 1e-5), [])
    rising = benchmark.StepStats(step_times_ms=(1.0,), loss=675.0, first_loss=24.8)
    (line,) = benchmark.loss_warning(rising, 1e-3)
    self.assertIn("rose from 24.8 to 675", line)
    self.assertIn("--learning_rate 0.001", line)
    self.assertEqual(benchmark.loss_warning(benchmark.StepStats(step_times_ms=(), loss=1.0), 1e-3), [])
    (nan_line,) = benchmark.loss_warning(
        benchmark.StepStats(step_times_ms=(1.0, 1.0), loss=float("nan"), first_loss=20.99), 1e-5
    )
    self.assertIn("ERROR", nan_line)
    self.assertIn("nan after 2 timed steps", nan_line)


class ThroughputTest(unittest.TestCase):

  def test_tokens_and_flops_of_the_mpmd_model(self):
    """A step's tokens and FLOPs are one stage's batch's, over every device; the MPMD model has no MTP block."""
    cfg = config.build_maxtext_config(hp.Hparams(model_name="deepseek3-tiny", num_layers=2, num_stages=1))
    stats = benchmark.StepStats(step_times_ms=(400.0, 500.0, 900.0), loss=1.0)  # median 0.5 s

    def throughput(maxtext_config):
      return benchmark.compute_throughput(
          stats=stats, maxtext_config=maxtext_config, devices_per_stage=2, num_devices=8, peak_tflops_per_device=10.0
      )

    tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
    result = throughput(cfg)
    self.assertEqual(result.tokens_per_step, 2 * 2 * 512)  # per_device_batch_size x devices per stage x sequence
    self.assertAlmostEqual(result.tflops_per_step_per_device, tflops * 2 / 8)
    self.assertAlmostEqual(result.tflops_per_second_per_device, tflops * 2 / 8 / 0.5)
    self.assertAlmostEqual(result.mfu, tflops * 2 / 8 / 0.5 / 10.0)
    self.assertAlmostEqual(result.tokens_per_second_per_device, 2048 / 0.5 / 8)
    # MaxText counts an MTP block's FLOPs (config.py refuses the setting; replace() bypasses it): none are charged.
    mtp = cfg.replace(mtp_num_layers=1)
    self.assertGreater(maxtext_utils.calculate_tflops_training_per_device(mtp, log=False)[0], tflops * 1.05)
    self.assertEqual(throughput(mtp), result)


class ProfileTest(unittest.TestCase):

  def _profile(self, start_trace):
    """benchmark.profile around nothing, off Pathways, with jax's tracer replaced by `start_trace`."""
    fake = types.SimpleNamespace(is_pathways_backend_used=lambda: False)
    with (
        mock.patch.dict(sys.modules, {"pathwaysutils": fake}),
        mock.patch("jax.profiler.start_trace", side_effect=start_trace) as start,
        mock.patch("jax.profiler.stop_trace"),
        self.assertLogs("absl", level="INFO") as logs,
    ):
      with benchmark.profile(True, tag="t", profile_dir="/tmp/xprof_test") as urls:
        pass
    return start.call_args_list, urls, "\n".join(logs.output)

  def test_one_tpu_one_sparsecore_one_tile(self):
    calls, urls, _ = self._profile(lambda *a, **k: None)
    (call,) = calls
    self.assertEqual(call.args, ("/tmp/xprof_test/t",))
    self.assertEqual(
        call.kwargs["profiler_options"].advanced_configuration,
        {
            "tpu_num_chips_to_profile_per_task": 1,
            "tpu_num_sparse_cores_to_trace": 1,
            "tpu_num_sparse_core_tiles_to_trace": 1,
        },
    )
    self.assertEqual(urls, ["/tmp/xprof_test/t"])

  def test_refused_options_trace_untrimmed(self):
    def start_trace(*args, profiler_options=None):
      if profiler_options is not None:
        raise RuntimeError("Bad PluginProgram")

    calls, urls, out = self._profile(start_trace)
    self.assertEqual([sorted(c.kwargs) for c in calls], [["profiler_options"], []])
    self.assertIn("The trimmed trace was refused (Bad PluginProgram); tracing untrimmed.", out)
    self.assertEqual(urls, ["/tmp/xprof_test/t"])


if __name__ == "__main__":
  unittest.main()
