# Copyright 2025-2026 Google LLC
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

"""Integration test for end-to-end Flax NNX LoRA checkpointing, resume, and adapter restoration across trainers."""

import os
import shutil
import sys
import tempfile
import unittest

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path
import pytest


def _tiny_lora_pyconfig(run_name, checkpoint_dir, **overrides):
  """Build a tiny pyconfig for E2E LoRA testing."""
  init_kwargs = {
      "run_name": run_name,
      "base_output_directory": checkpoint_dir,
      "enable_checkpointing": True,
      "dataset_type": "synthetic",
      "model_name": "default",
      "pure_nnx": True,
      "per_device_batch_size": 1.0,
      "base_emb_dim": 8,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "base_mlp_dim": 32,
      "base_num_decoder_layers": 2,
      "head_dim": 128,
      "max_target_length": 128,
      "vocab_size": 256,
      "steps": 10,
      "async_checkpointing": False,
      "checkpoint_period": 10,
      "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
      "enable_goodput_recording": False,
      "enable_checkpoint_cloud_logger": False,
      "monitor_goodput": False,
      "override_model_config": True,
      "use_tunix_gradient_accumulation": False,
      "ici_fsdp_parallelism": 1,
      "ici_tensor_parallelism": 1,
      "ici_expert_parallelism": 1,
      "ici_data_parallelism": -1,
      "num_experts": 2,
      "num_experts_per_tok": 1,
      "shared_experts": 1,
      "base_moe_mlp_dim": 32,
      "attention": "dot_product",
  }
  init_kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **init_kwargs)


@pytest.mark.integration_test
class LoraE2ENnxIntegrationTest(unittest.TestCase):
  """E2E integration test for NNX LoRA lifecycle.

  Covers base generation, LoRA train, resume, and standalone restore.
  """

  def setUp(self):
    self.test_dir = tempfile.mkdtemp(prefix="lora_e2e_test_")

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def _run_e2e_flow(self, model_name, use_sft, lora_weight_qtype=None, scan_layers=True):
    """Executes a full 4-step E2E LoRA checkpoint/resume/restore flow."""
    from maxtext.trainers.pre_train import train  # pylint: disable=import-outside-toplevel

    base_run_name = f"b_{model_name}_{use_sft}_run"
    lora_run_name = f"w_{model_name}_{use_sft}_run"

    # Step 1: Generate base-only checkpoint (steps=2)
    config_step1 = _tiny_lora_pyconfig(
        run_name=base_run_name,
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=use_sft,
        scan_layers=scan_layers,
        steps=2,
        checkpoint_period=2,
        lora={"enable_lora": False},
    )
    state_step1 = train.train_loop(config_step1, recorder=None)
    self.assertEqual(int(state_step1.optimizer.step.get_value()), 2)

    base_ckpt_dir = os.path.join(self.test_dir, base_run_name, "checkpoints", "1")
    self.assertTrue(os.path.exists(base_ckpt_dir), f"Base checkpoint path does not exist: {base_ckpt_dir}")
    base_ckpt_path = os.path.join(base_ckpt_dir, "items")

    lora_config = {"enable_lora": True, "lora_rank": 4}
    if lora_weight_qtype:
      lora_config["lora_weight_qtype"] = lora_weight_qtype
      lora_config["lora_tile_size"] = 4

    # Step 2: Train with LoRA starting from base checkpoint (steps=4)
    config_step2 = _tiny_lora_pyconfig(
        run_name=lora_run_name,
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=use_sft,
        scan_layers=scan_layers,
        load_parameters_path=base_ckpt_path,
        steps=4,
        checkpoint_period=2,
        lora=lora_config,
    )
    state_step2 = train.train_loop(config_step2, recorder=None)
    self.assertEqual(int(state_step2.optimizer.step.get_value()), 4)

    lora_ckpt_dir = os.path.join(self.test_dir, lora_run_name, "checkpoints", "3")
    self.assertTrue(os.path.exists(lora_ckpt_dir), f"Saved LoRA checkpoint path does not exist: {lora_ckpt_dir}")
    lora_ckpt_path = os.path.join(lora_ckpt_dir, "items")

    # Step 3: Resume training under same run name (steps=6)
    config_step3 = _tiny_lora_pyconfig(
        run_name=lora_run_name,
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=use_sft,
        scan_layers=scan_layers,
        steps=6,
        checkpoint_period=2,
        lora=lora_config,
    )
    state_step3 = train.train_loop(config_step3, recorder=None)
    self.assertEqual(int(state_step3.optimizer.step.get_value()), 6)

    # Step 4: Standalone restore of LoRA adapter onto base checkpoint (steps=2)
    lora_restore_config = dict(lora_config)
    lora_restore_config["lora_restore_path"] = lora_ckpt_path
    config_step4 = _tiny_lora_pyconfig(
        run_name=f"restore_{model_name}_{use_sft}_run",
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=use_sft,
        scan_layers=scan_layers,
        load_parameters_path=base_ckpt_path,
        steps=2,
        checkpoint_period=2,
        lora=lora_restore_config,
    )
    state_step4 = train.train_loop(config_step4, recorder=None)
    self.assertEqual(int(state_step4.optimizer.step.get_value()), 2)

  def _run_e2e_flow_sft(self, model_name, lora_weight_qtype=None, scan_layers=True):
    """Executes a full 4-step E2E LoRA checkpoint/resume/restore flow for SFT (Tunix)."""
    from maxtext.trainers.post_train.sft import train_sft  # pylint: disable=import-outside-toplevel

    base_run_name = f"b_{model_name}_sft_run"
    lora_run_name = f"w_{model_name}_sft_run"

    # Step 1: Generate base-only checkpoint (steps=2)
    config_step1 = _tiny_lora_pyconfig(
        run_name=base_run_name,
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=True,
        scan_layers=scan_layers,
        steps=2,
        checkpoint_period=2,
        lora={"enable_lora": False},
    )
    trainer_step1, _ = train_sft.train(config_step1, goodput_recorder=None)
    self.assertEqual(int(trainer_step1.train_steps), 2)

    base_ckpt_dir = os.path.join(self.test_dir, base_run_name, "checkpoints", "1")
    self.assertTrue(os.path.exists(base_ckpt_dir), f"Base checkpoint path does not exist: {base_ckpt_dir}")
    base_ckpt_path = os.path.join(base_ckpt_dir, "model_params")

    lora_config = {"enable_lora": True, "lora_rank": 4}
    if lora_weight_qtype:
      lora_config["lora_weight_qtype"] = lora_weight_qtype
      lora_config["lora_tile_size"] = 4

    # Step 2: Train with LoRA starting from base checkpoint (steps=4)
    config_step2 = _tiny_lora_pyconfig(
        run_name=lora_run_name,
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=True,
        scan_layers=scan_layers,
        load_parameters_path=base_ckpt_path,
        steps=4,
        checkpoint_period=2,
        lora=lora_config,
    )
    trainer_step2, _ = train_sft.train(config_step2, goodput_recorder=None)
    self.assertEqual(int(trainer_step2.train_steps), 4)

    lora_ckpt_dir = os.path.join(self.test_dir, lora_run_name, "checkpoints", "4")
    self.assertTrue(os.path.exists(lora_ckpt_dir), f"Saved LoRA checkpoint path does not exist: {lora_ckpt_dir}")
    lora_ckpt_path = os.path.join(lora_ckpt_dir, "model_params")

    # Step 3: Resume training under same run name (steps=6)
    config_step3 = _tiny_lora_pyconfig(
        run_name=lora_run_name,
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=True,
        scan_layers=scan_layers,
        steps=6,
        checkpoint_period=2,
        lora=lora_config,
    )
    trainer_step3, _ = train_sft.train(config_step3, goodput_recorder=None)
    self.assertEqual(int(trainer_step3.train_steps), 6)

    # Step 4: Standalone restore of LoRA adapter onto base checkpoint (steps=2)
    lora_restore_config = dict(lora_config)
    lora_restore_config["lora_restore_path"] = lora_ckpt_path
    config_step4 = _tiny_lora_pyconfig(
        run_name=f"restore_{model_name}_sft_run",
        checkpoint_dir=self.test_dir,
        model_name=model_name,
        use_sft=True,
        scan_layers=scan_layers,
        load_parameters_path=base_ckpt_path,
        steps=2,
        checkpoint_period=2,
        lora=lora_restore_config,
    )
    trainer_step4, _ = train_sft.train(config_step4, goodput_recorder=None)
    self.assertEqual(int(trainer_step4.train_steps), 2)

  # --- LoRA Unquantized Tests (Gemma4) ---
  def test_lora_e2e_gemma4_pretrain(self):
    self._run_e2e_flow("gemma4-26b", use_sft=False)

  def test_lora_e2e_gemma4_sft_native(self):
    self._run_e2e_flow("gemma4-26b", use_sft=True)

  @pytest.mark.post_training
  def test_lora_e2e_gemma4_sft(self):
    self._run_e2e_flow_sft("gemma4-26b")

  # --- QLoRA NF4 Tests (Gemma4, Qwen3, GPT-OSS) ---
  def test_qlora_e2e_gemma4_pretrain_nf4(self):
    self._run_e2e_flow("gemma4-26b", use_sft=False, lora_weight_qtype="nf4")

  def test_qlora_e2e_gemma4_sft_native_nf4(self):
    self._run_e2e_flow("gemma4-26b", use_sft=True, lora_weight_qtype="nf4")

  @pytest.mark.post_training
  def test_qlora_e2e_gemma4_sft_nf4(self):
    self._run_e2e_flow_sft("gemma4-26b", lora_weight_qtype="nf4")

  def test_qlora_e2e_qwen3_pretrain_nf4(self):
    self._run_e2e_flow("qwen3-4b", use_sft=False, lora_weight_qtype="nf4")

  def test_qlora_e2e_qwen3_sft_native_nf4(self):
    self._run_e2e_flow("qwen3-4b", use_sft=True, lora_weight_qtype="nf4")

  @pytest.mark.post_training
  def test_qlora_e2e_qwen3_sft_nf4(self):
    self._run_e2e_flow_sft("qwen3-4b", lora_weight_qtype="nf4")

  def test_qlora_e2e_gptoss_unscanned_nf4(self):
    self._run_e2e_flow("gpt-oss-20b", use_sft=False, lora_weight_qtype="nf4", scan_layers=False)


if __name__ == "__main__":
  unittest.main()
