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

"""Tests for the trainer's main loop and its NaN stop, end to end on CPU (no Shardy).

train.main runs with the Pathways-only pieces swapped for the SPMD step: Shardy MPMD's program compiles only on
Pathways.
"""

import json
import logging
import os
import tempfile
import types
import unittest
from unittest import mock

from absl import flags
import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.partir import train
from maxtext.experimental.mpmd.partir import train_step

ARGV = [
    "train",
    "--model_name=simple",
    "--num_layers=2",
    "--num_stages=2",
    "--emb_dim=64",
    "--mlp_dim=128",
    "--vocab_size=256",
    "--max_target_length=16",
    "--per_device_batch_size=2",
    "--num_microbatches=2",
    "--warmup_steps=1",
    "--steps=3",
    "--run_name=nan-test",
]


def run_main(argv, nan_at=0):
  """Runs train.main on `argv` with the compiled step's call `nan_at` (1-based; 0: none) returning a NaN loss.

  Returns:
    (what main raised or None, what it logged).
  """
  flag_values = flags.FlagValues()
  train.define_flags(flag_values)
  flag_values(argv)
  calls = [0]
  real_build = train.runner_lib.build

  def build(hparams, *, mode, **kw):
    del mode  # the MPMD program compiles only on Pathways: run the SPMD step instead
    return real_build(hparams, mode=train_step.ExecutionMode.SPMD, **kw)

  class Lowered:
    """Stands in for the lowered MPMD program: compiles the SPMD step instead."""

    function_named_shardings = types.SimpleNamespace(input_specs=None)

    def __init__(self, fn):
      self.fn = fn

    def compile(self):
      jitted = jax.jit(self.fn)

      def step(state, batch):
        calls[0] += 1
        state, loss = jitted(state, batch)
        return state, (jnp.float32(float("nan")) if calls[0] == nan_at else loss)

      return step

  logged = _Collect()
  absl_logger = logging.getLogger("absl")
  level = absl_logger.level
  absl_logger.addHandler(logged)
  absl_logger.setLevel(logging.INFO)
  raised = None
  with (
      mock.patch.object(train.runner_lib, "build", build),
      mock.patch.object(train.runner_lib.Runnable, "lower", lambda runnable: Lowered(runnable.train_step)),
      mock.patch.object(train.runner_lib, "place_on_stage_meshes", lambda inputs, specs: inputs),
  ):
    try:
      train.main(["train"], flag_values=flag_values)
    except SystemExit as error:
      raised = error
    finally:
      absl_logger.removeHandler(logged)
      absl_logger.setLevel(level)
  return raised, "\n".join(logged.messages)


class _Collect(logging.Handler):
  """Keeps the messages logged through absl (max_logging)."""

  def __init__(self):
    super().__init__(level=logging.INFO)
    self.messages = []

  def emit(self, record):
    self.messages.append(record.getMessage())


class TrainMainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < 4:
      self.skipTest("needs 4 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")

  def test_a_finished_run_prints_its_record(self):
    with tempfile.TemporaryDirectory() as tmp:
      out_path = os.path.join(tmp, "run.json")
      raised, logged = run_main([*ARGV, f"--result_json={out_path}"])
      self.assertIsNone(raised)
      with open(out_path, encoding="utf-8") as f:
        record = json.load(f)
    self.assertEqual((record["run"], record["stages"], record["microbatches"]), ("nan-test", 2, 2))
    self.assertLess(record["loss"], record["loss_first"])
    self.assertIn("Results for nan-test", logged)
    self.assertIn("Execution finished successfully.", logged)

  def test_a_nan_in_a_profiled_step_is_numbered_among_all_steps(self):
    """1 warm-up and 3 timed steps, then a NaN in the second profiled step: the error record calls it step 6, lists
    the losses from step 5, exits with an error, and writes --result_json."""
    with tempfile.TemporaryDirectory() as tmp:
      out_path = os.path.join(tmp, "run.json")
      argv = [
          *ARGV,
          "--enable_profile",
          "--profile_steps=2",
          f"--profile_dir={os.path.join(tmp, 'xprof')}",
          f"--result_json={out_path}",
      ]
      raised, logged = run_main(argv, nan_at=6)
      self.assertIsInstance(raised, SystemExit)
      self.assertIn("MPMD training failed", str(raised))
      with open(out_path, encoding="utf-8") as f:
        record = json.load(f)
    self.assertRegex(
        record["error"], r"^non-finite loss: step 6 returned a loss of nan \(losses from step 5: [0-9.e+-]+, nan\)$"
    )
    self.assertEqual(record["run"], "nan-test")
    self.assertIn("MPMD_RESULT ", logged)


if __name__ == "__main__":
  unittest.main()
