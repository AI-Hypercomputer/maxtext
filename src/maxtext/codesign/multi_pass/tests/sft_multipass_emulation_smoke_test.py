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

"""Scheduled nightly TPU end-to-end SFT test with custom Multi-Pass Precision Emulation."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import pytest

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path
from maxtext.codesign.multi_pass.checkpoint_utils import inspect_model_or_pytree_multipass
from maxtext.codesign.hooks import install_emulation_hooks
from maxtext.codesign.interception import (
    set_active_multipass_config,
    set_current_emulation_step,
)

# Scheduled nightly on TPU via tpu-post-training-unit (v6e-4 and tpu7x-8), NOT presubmit.
pytestmark = [
    pytest.mark.post_training,
    pytest.mark.tpu_only,
    pytest.mark.scheduled_only,
]


def _build_tiny_debug_sft_config(run_name: str, output_dir: str, **overrides):
  """Builds a tiny 4-layer unrolled (`scan_layers=False`) SFT config for TPU e2e testing."""
  install_emulation_hooks(workflow="sft")
  metrics_file = os.path.join(output_dir, f"{run_name}_metrics.jsonl")
  init_kwargs = {
      "run_name": run_name,
      "base_output_directory": output_dir,
      "use_sft": True,
      "dataset_type": "synthetic",
      "model_name": "default",
      "override_model_config": True,
      "convert_checkpoint_if_possible": False,
      "scan_layers": False,
      "per_device_batch_size": 1.0,
      "base_emb_dim": 64,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "base_mlp_dim": 128,
      "base_num_decoder_layers": 4,
      "head_dim": 64,
      "max_target_length": 128,
      "vocab_size": 256,
      "steps": 4,
      "eval_interval": -1,
      "enable_checkpointing": True,
      "async_checkpointing": False,
      "checkpoint_period": 4,
      "enable_tensorboard": True,
      "metrics_file": metrics_file,
      "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
      "enable_goodput_recording": False,
      "enable_checkpoint_cloud_logger": False,
      "monitor_goodput": False,
      "use_tunix_gradient_accumulation": False,
      "ici_fsdp_parallelism": 1,
      "ici_tensor_parallelism": 1,
      "ici_data_parallelism": -1,
      "attention": "dot_product",
  }
  init_kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **init_kwargs)


class SftMultipassEmulationSmokeTest(unittest.TestCase):
  """Scheduled nightly TPU e2e test comparing baseline SFT vs custom multi-pass SFT."""

  def setUp(self):
    super().setUp()
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("INPUTS_IS_SCHEDULED_RUN", "").lower() == "false":
      self.skipTest("Scheduled-only e2e SFT test; skipped during presubmit CI.")
    self._orig_wandb_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"
    self.test_dir = tempfile.mkdtemp(prefix="sft_multipass_smoke_")
    set_active_multipass_config(None)
    set_current_emulation_step(0)

  def tearDown(self):
    set_active_multipass_config(None)
    set_current_emulation_step(0)
    if self._orig_wandb_mode is None:
      os.environ.pop("WANDB_MODE", None)
    else:
      os.environ["WANDB_MODE"] = self._orig_wandb_mode
    if hasattr(self, "test_dir") and os.path.isdir(self.test_dir):
      shutil.rmtree(self.test_dir, ignore_errors=True)
    super().tearDown()

  def test_tiny_sft_baseline_vs_custom_heterogeneous_multipass_at_step_k(self):
    """Runs 4-step baseline SFT vs 4-step custom 3-pass heterogeneous SFT on layer 2 starting at step 2."""
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from maxtext.trainers.post_train.sft import train_sft  # pylint: disable=import-outside-toplevel

    # 1. Run A: Native Baseline SFT (4 steps)
    baseline_cfg = _build_tiny_debug_sft_config(
        run_name="smoke_sft_native_baseline",
        output_dir=self.test_dir,
        emulate_multipass_enabled=False,
    )
    trainer_base, _ = train_sft.train(baseline_cfg, goodput_recorder=None)
    self.assertEqual(int(trainer_base.train_steps), 4)

    # 2. Run B: Custom 3-Pass Heterogeneous Multi-Pass Emulation (`bf16 -> [fp8_e5m2, fp8_e4m3, fp4_e2m1]`)
    #    with MX-style power-of-two block scaling (`per_block_pow2`, block=32), triangular X6 cross-terms,
    #    forward + dLHS backward multi-pass, targeting middle layer 2 `mlp/wo` starting at step K=2.
    mp_cfg = _build_tiny_debug_sft_config(
        run_name="smoke_sft_multipass_L2_step2",
        output_dir=self.test_dir,
        emulate_multipass_enabled=True,
        emulate_src_dtype="bf16",
        emulate_num_passes=3,
        emulate_pass_dtypes="fp8_e5m2,fp8_e4m3,fp4_e2m1",
        emulate_scale_mode="per_block_pow2",
        emulate_block_size=32,
        emulate_cross_term_mode="triangular",
        emulate_acc_dtype="fp32",
        emulate_fwd=True,
        emulate_dlhs=True,
        emulate_drhs=False,
        emulate_layers="2",
        emulate_target_modules="mlp",
        emulate_target_submodules="wo",
        emulate_start_step=2,
    )
    trainer_mp, _ = train_sft.train(mp_cfg, goodput_recorder=None)
    self.assertEqual(int(trainer_mp.train_steps), 4)

    # 3. Verify step-gated metric logging (`learning/emulation_active` = 0.0 for steps < 2, 1.0 for steps >= 2)
    mp_metrics_path = os.path.join(self.test_dir, "smoke_sft_multipass_L2_step2_metrics.jsonl")
    self.assertTrue(os.path.isfile(mp_metrics_path))
    with open(mp_metrics_path, "r", encoding="utf-8") as f:
      logged_rows = [json.loads(line) for line in f if line.strip()]
    self.assertEqual(len(logged_rows), 4)
    self.assertEqual(logged_rows[0].get("learning/emulation_active"), 0.0)
    self.assertEqual(logged_rows[1].get("learning/emulation_active"), 0.0)
    self.assertEqual(logged_rows[2].get("learning/emulation_active"), 1.0)
    self.assertEqual(logged_rows[3].get("learning/emulation_active"), 1.0)
    self.assertEqual(logged_rows[2].get("learning/emulation_lhs_passes"), 3.0)
    self.assertEqual(logged_rows[2].get("learning/emulation_rhs_passes"), 3.0)
    self.assertEqual(logged_rows[2].get("learning/emulation_active_gemm_pairs"), 6.0)

    # 4. Verify that ONLY middle layer 2's `mlp/wo` is targeted and exhibits monotonic 3-pass error reduction
    pure_state = nnx.to_pure_dict(nnx.state(trainer_mp.model, nnx.Param))
    report = inspect_model_or_pytree_multipass(pure_state, mp_cfg.multipass_emulation)
    self.assertEqual(report["num_targeted_tensors"], 1)
    self.assertGreater(report["targeted_parameters"], 0)
    tensor_detail = report["tensors"][0]
    self.assertIn("layers_2", tensor_detail["path"])
    errs = tensor_detail["per_pass_relative_l2_error"]
    self.assertEqual(len(errs), 3)
    self.assertLess(errs[1], errs[0])
    self.assertLess(errs[2], errs[1])


if __name__ == "__main__":
  unittest.main()
