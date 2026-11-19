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

"""Single-host integration test for the constant-global-batch elastic modes (`constant_padded`, `constant_grad_accum`).

A replica resize cannot be exercised on one host (it needs Pathways), but its batch arithmetic can:
`elastic_simulate_job_devices` sizes the logical global batch for a larger job and plans and pads the physical batch
for the devices present, exactly as after a resize. The same logical batch is then trained with every factoring the
planner can produce, and all of them must agree:

*   `constant_padded` with the legacy `elastic_max_per_device_batch_size=-1`: a wider per-device batch, same
    accumulation steps;
*   `constant_padded` with `elastic_max_per_device_batch_size=per_device_batch_size`: same per-device batch, more
    accumulation steps;
*   `constant_padded` with a looser bound: whatever pads least;
*   `constant_grad_accum`: same per-device batch, more accumulation steps, with no bound flag at all.

Because the padded rows carry no tokens, `learning/total_weights` must be identical across the runs, and the loss and
gradient norm must agree to floating-point tolerance.
"""

import json
import os
import pathlib
import random
import string
import tempfile
import unittest

import jax
import numpy as np
import pytest

from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path

_STEPS = 3
_SEQ_LEN = 64


def _suffix():
  """A random suffix that keeps the metrics files of repeated runs apart."""
  return "".join(random.choice(string.ascii_letters) for _ in range(8))


class ElasticConstantPaddedTest(unittest.TestCase):
  """Trains the same logical batch under every constant-global-batch factoring and compares the metrics."""

  def _run(self, name, extra_args):
    """Trains `_STEPS` steps of a tiny dense model and returns its per-step metrics keyed by step."""
    metrics_file = os.path.join(tempfile.gettempdir(), f"{name}_{_suffix()}.txt")
    self.addCleanup(pathlib.Path(metrics_file).unlink, missing_ok=True)
    train_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={get_test_base_output_directory()}",
            f"run_name={name}",
            f"metrics_file={metrics_file}",
            "dataset_type=synthetic",
            "packing=false",
            "gradient_clipping_threshold=0",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "decoder_block=simple",
            "base_emb_dim=256",
            "base_num_decoder_layers=2",
            f"max_target_length={_SEQ_LEN}",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            f"steps={_STEPS}",
            *extra_args,
        ]
    )
    with open(metrics_file, "rt", encoding="utf8") as metrics:
      rows = [json.loads(line) for line in metrics if line.strip()]
    return {int(row["step"]): row for row in rows if "step" in row}

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_every_factoring_trains_the_same_logical_batch(self):
    """Every factoring of the padded batch trains on the same real tokens and matches the legacy run's metrics."""
    self._compare_factorings("padded")

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_every_factoring_trains_the_same_logical_batch_moe(self):
    """MoE: the padded rows stay out of routing, capacity, the aux loss and the routed-bias update (`token_mask`)."""
    moe_args = [
        "decoder_block=mixtral",
        "base_mlp_dim=256",
        "base_moe_mlp_dim=256",
        "num_experts=4",
        "num_experts_per_tok=2",
        "load_balance_loss_weight=0.01",
    ]
    with self.subTest(path="dense"):
      self._compare_factorings("padded_moe_dense", moe_args + ["sparse_matmul=false", "megablox=false"])
    with self.subTest(path="dense_capacity"):
      self._compare_factorings(
          "padded_moe_capacity", moe_args + ["sparse_matmul=false", "megablox=false", "capacity_factor=1.25"]
      )
    with self.subTest(path="sparse"):
      self._compare_factorings("padded_moe_sparse", moe_args + ["sparse_matmul=true", "megablox=true"])

  def _compare_factorings(self, prefix, model_args=()):
    """Trains every factoring of the simulated resize with `model_args` and compares their metrics."""
    live_devices = len(jax.devices())
    # Pretend a job of 1.5x the present devices with per-device batch 3: the logical batch is 4.5 * live_devices
    # rows, which no factoring on the live devices covers exactly, so every run below pads. With 4 devices: 18
    # logical rows.
    job_devices = live_devices * 3 // 2
    shared = [
        *model_args,
        f"elastic_simulate_job_devices={job_devices}",
        "per_device_batch_size=3",
        "gradient_accumulation_steps=1",
    ]
    padded = shared + ["elastic_global_batch_size_mode=constant_padded"]
    logical_rows = 3 * job_devices
    runs = {
        # B' = ceil(4.5) = 5, K' = 1: physical 5 * live_devices rows (20 with 4 devices, 2 padded).
        "legacy": self._run(f"{prefix}_legacy", padded + ["elastic_max_per_device_batch_size=-1"]),
        # B' = 3, K' = 2: physical 6 * live_devices rows (24, 6 padded); the per-device batch is unchanged.
        "fixed_b": self._run(f"{prefix}_fixed_b", padded + ["elastic_max_per_device_batch_size=3"]),
        # Least padding within [3, 8] is B' = 5, K' = 1 again, found by the search rather than the legacy rule.
        "bounded": self._run(f"{prefix}_bounded", padded + ["elastic_max_per_device_batch_size=8"]),
        # The recommended mode: B' = 3, K' = 2 like "fixed_b", with no bound flag.
        "grad_accum": self._run(f"{prefix}_grad_accum", shared + ["elastic_global_batch_size_mode=constant_grad_accum"]),
    }
    reference = runs["legacy"]
    for step in range(_STEPS):
      for name, run in runs.items():
        with self.subTest(step=step, run=name):
          self.assertEqual(float(run[step]["learning/total_weights"]), float(logical_rows * _SEQ_LEN))
          np.testing.assert_allclose(run[step]["learning/loss"], reference[step]["learning/loss"], rtol=1e-2)
          np.testing.assert_allclose(run[step]["learning/grad_norm"], reference[step]["learning/grad_norm"], rtol=1e-2)
          if "learning/moe_lb_loss" in reference[step]:
            # The aux loss is a per-sequence statistic averaged over the real rows only, so it must not scale with
            # the number of padded rows. The factorings pad a different number of tail rows (2 or 6 with 4 devices),
            # so on the sparse path (loss formed inside the shard_map) this also checks the cross-shard reduction of
            # the masked loss: the runs then agree on the same 18-row mean (identical at step 0 on v6e-4, within
            # 3e-4 after the drift of the training steps), whereas a per-shard statistic (shards with 3 / 5 and
            # 0 / 6 real rows) would differ by several percent.
            np.testing.assert_allclose(
                run[step]["learning/moe_lb_loss"], reference[step]["learning/moe_lb_loss"], rtol=1e-3
            )


if __name__ == "__main__":
  unittest.main()
