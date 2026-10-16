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

"""Tests for the minimal m3 training loop."""

import glob
import math
import os

from absl.testing import absltest
import jax
import numpy as np

from maxtext.configs import pyconfig
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.train import loop
from tests.utils.test_helpers import get_test_config_path


def _tiny_config(output_dir: str, **overrides):
  """A 2-layer qwen3-0.6b variant small enough to train on CPU in seconds.

  The `base_*` keys are what the model YAML sets; overriding the derived names
  (`emb_dim`, `mlp_dim`, ...) is silently ignored.
  """
  args = {
      "model_name": "qwen3-0.6b",
      "use_m3_model": "true",
      "scan_layers": "false",
      "override_model_config": "true",
      "base_num_decoder_layers": "2",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": "256",
      "per_device_batch_size": "2",
      "max_target_length": "16",
      "steps": "3",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_loop_test",
  }
  args.update({k: str(v) for k, v in overrides.items()})
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


class TrainLoopTest(absltest.TestCase):
  """Runs the loop end to end on a tiny model."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def test_tiny_config_takes_effect(self):
    """Guards the `base_*` overrides; a full-size model here would make the suite slow."""
    config = _tiny_config(self.output_dir)
    self.assertEqual(config.emb_dim, 64)
    self.assertEqual(config.num_decoder_layers, 2)

  def test_synthetic_batch_targets_are_shifted_inputs(self):
    config = _tiny_config(self.output_dir)
    batch = loop.make_synthetic_batch(config, create_mesh(), jax.random.key(0))
    inputs, targets = np.asarray(batch["inputs"]), np.asarray(batch["targets"])
    self.assertEqual(inputs.shape, (config.global_batch_size_to_train_on, config.max_target_length))
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])

  def test_train_loop_returns_finite_loss(self):
    loss = loop.train_loop(_tiny_config(self.output_dir))
    self.assertTrue(math.isfinite(loss), f"loss={loss}")

  def test_train_loop_writes_xplane_profile(self):
    config = _tiny_config(self.output_dir, profiler="xplane", skip_first_n_steps_for_profiler=1, profiler_steps=1)
    loop.train_loop(config)
    traces = glob.glob(os.path.join(config.tensorboard_dir, "**", "*.xplane.pb"), recursive=True)
    self.assertNotEmpty(traces, f"no .xplane.pb under {config.tensorboard_dir}")

  def test_rejects_gradient_accumulation(self):
    with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps=2"):
      loop.train_loop(_tiny_config(self.output_dir, gradient_accumulation_steps=2))

  def test_rejects_batch_that_does_not_split_over_mesh(self):
    """`per_device_batch_size=0.5` never splits: batch 0 on one device, n // 2 on n devices."""
    with self.assertRaisesRegex(ValueError, "does not split over"):
      loop.train_loop(_tiny_config(self.output_dir, per_device_batch_size=0.5))


if __name__ == "__main__":
  absltest.main()
