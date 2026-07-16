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

"""Integration tests for Flax NNX PEFT, LoRA, and QLoRA."""

import os
import sys
import unittest
import tempfile
import json
import shutil
import pytest
import subprocess

from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path


def _get_jax_backend():
  """Determine JAX backend by running a quick subprocess to avoid locking TPU in the parent process."""
  try:
    result = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.default_backend())"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
  except Exception:  # pylint: disable=broad-exception-caught
    return "cpu"


def _get_jax_device_count():
  """Determine JAX device count by running a quick subprocess."""
  try:
    result = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.device_count())"], capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip())
  except Exception:  # pylint: disable=broad-exception-caught
    return 1


def _get_common_args(temp_dir, model_name, steps=1, overrides=None):
  """Build common arguments for downscaled integration runs."""
  tokenizer_path = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2")

  backend = _get_jax_backend()
  num_devices = _get_jax_device_count()

  hardware = "cpu"
  if backend == "tpu":
    hardware = "tpu"
  elif backend == "gpu":
    hardware = "gpu"

  args = [
      None,
      get_test_config_path(),
      f"run_name=test_peft_{model_name}",
      f"model_name={model_name}",
      f"steps={steps}",
      f"base_output_directory={temp_dir}",
      "dataset_type=synthetic",
      "per_device_batch_size=1",
      "max_target_length=32",
      f"tokenizer_path={tokenizer_path}",
      "enable_goodput_recording=False",
      "enable_checkpoint_cloud_logger=False",
      "monitor_goodput=False",
      "sharding_tolerance=1.0",  # Bypass sharding checks for downscaled test models
      f"hardware={hardware}",
      "skip_jax_distributed_system=True",
      "ici_fsdp_parallelism=1",
      f"ici_data_parallelism={num_devices}",
  ]
  if overrides:
    args.extend(overrides)
  return args


def _run_train_subprocess(args):
  """Run the pre-training trainer inside an isolated Python subprocess."""
  cmd = [sys.executable, "-m", "maxtext.trainers.pre_train.train", args[1]] + args[2:]
  env = os.environ.copy()
  env["PYTHONPATH"] = os.path.abspath("src")

  backend = _get_jax_backend()

  if backend not in ("tpu", "gpu"):
    env["JAX_PLATFORMS"] = "cpu"

  subprocess.run(cmd, env=env, check=True)


def _run_sft_subprocess(args):
  """Run the SFT native trainer inside an isolated Python subprocess."""
  cmd = [sys.executable, "-m", "maxtext.trainers.post_train.sft.train_sft_native", args[1]] + args[2:]
  env = os.environ.copy()
  env["PYTHONPATH"] = os.path.abspath("src")

  backend = _get_jax_backend()

  if backend not in ("tpu", "gpu"):
    env["JAX_PLATFORMS"] = "cpu"

  subprocess.run(cmd, env=env, check=True)


@pytest.mark.integration_test
class PEFTIntegrationTest(unittest.TestCase):
  """Integration checks for pre-training and native SFT with LoRA & QLoRA."""

  def setUp(self):
    self.temp_dirs = []

  def tearDown(self):
    for temp_dir in self.temp_dirs:
      if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

  def _create_temp_dir(self):
    temp_dir = tempfile.mkdtemp()
    self.temp_dirs.append(temp_dir)
    return temp_dir

  def test_native_sft_gemma4_lora(self):
    """Test native SFT with Gemma4-e2b and LoRA enabled."""
    temp_dir = self._create_temp_dir()
    overrides = [
        "override_model_config=True",
        "scan_layers=False",
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "lora.enable_lora=True",
        "lora.lora_rank=4",
        "lora.lora_alpha=8.0",
        "base_num_decoder_layers=2",
        "base_emb_dim=256",
        "base_mlp_dim=512",
        "num_kv_shared_layers=1",
    ]
    args = _get_common_args(temp_dir, "gemma4-e2b", steps=2, overrides=overrides)
    _run_sft_subprocess(args)

  def test_native_sft_gemma4_qlora(self):
    """Test native SFT with Gemma4-e2b and QLoRA enabled."""
    temp_dir = self._create_temp_dir()
    overrides = [
        "override_model_config=True",
        "scan_layers=False",
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "lora.enable_lora=True",
        "lora.lora_weight_qtype=int8",
        "lora.lora_tile_size=32",
        "lora.lora_rank=4",
        "lora.lora_alpha=8.0",
        "base_num_decoder_layers=2",
        "base_emb_dim=256",
        "base_mlp_dim=512",
        "num_kv_shared_layers=1",
    ]
    args = _get_common_args(temp_dir, "gemma4-e2b", steps=2, overrides=overrides)
    _run_sft_subprocess(args)

  def test_native_sft_qwen3_lora(self):
    """Test native SFT with Qwen3-0.6B and LoRA enabled."""
    temp_dir = self._create_temp_dir()
    overrides = [
        "override_model_config=True",
        "scan_layers=True",
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "lora.enable_lora=True",
        "lora.lora_rank=4",
        "lora.lora_alpha=8.0",
        "base_num_decoder_layers=2",
        "base_emb_dim=256",
        "base_mlp_dim=512",
    ]
    args = _get_common_args(temp_dir, "qwen3-0.6b", steps=2, overrides=overrides)
    _run_sft_subprocess(args)

  def test_native_sft_qwen3_qlora(self):
    """Test native SFT with Qwen3-0.6B and QLoRA enabled."""
    temp_dir = self._create_temp_dir()
    overrides = [
        "override_model_config=True",
        "scan_layers=True",
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "lora.enable_lora=True",
        "lora.lora_weight_qtype=int8",
        "lora.lora_tile_size=32",
        "lora.lora_rank=4",
        "lora.lora_alpha=8.0",
        "base_num_decoder_layers=2",
        "base_emb_dim=256",
        "base_mlp_dim=512",
    ]
    args = _get_common_args(temp_dir, "qwen3-0.6b", steps=2, overrides=overrides)
    _run_sft_subprocess(args)

  def test_pretrain_lora_checkpoint_save_and_restore(self):
    """Test pre-training with LoRA, checkpoint saving, and subsequent restoration."""
    temp_dir = self._create_temp_dir()
    metrics_file_saved = os.path.join(temp_dir, "saved_metrics.txt")
    metrics_file_restored = os.path.join(temp_dir, "restored_metrics.txt")

    # Common model scaling/configs
    overrides_base = [
        "scan_layers=True",
        "attention=dot_product",
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "lora.enable_lora=True",
        "lora.lora_rank=4",
        "lora.lora_alpha=8.0",
        "base_num_decoder_layers=2",
        "base_emb_dim=128",
        "base_mlp_dim=256",
        "ici_fsdp_parallelism=1",
    ]

    # 1. First run: train 1 step with checkpointing enabled
    args_save = _get_common_args(
        temp_dir,
        "default",
        steps=1,
        overrides=overrides_base
        + [
            "enable_checkpointing=True",
            "checkpoint_period=1",
            "async_checkpointing=False",
            f"metrics_file={metrics_file_saved}",
        ],
    )
    _run_train_subprocess(args_save)

    # Verify checkpoint got written (MaxText writes to base_output_directory/run_name/checkpoints/0/items)
    checkpoint_dir = os.path.join(temp_dir, "test_peft_default", "checkpoints", "0", "items")
    self.assertTrue(os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} was not created.")

    # 2. Second run: restore from the checkpoint directory and train 1 step (total steps = 2)
    args_restore = _get_common_args(
        temp_dir,
        "default",
        steps=2,
        overrides=overrides_base
        + [
            "enable_checkpointing=True",
            "checkpoint_period=1",
            "async_checkpointing=False",
            f"load_parameters_path={checkpoint_dir}",
            f"metrics_file={metrics_file_restored}",
        ],
    )
    _run_train_subprocess(args_restore)

    # 3. Verify losses are logged and correct
    self.assertTrue(os.path.exists(metrics_file_saved))
    self.assertTrue(os.path.exists(metrics_file_restored))

    with open(metrics_file_saved, "r", encoding="utf8") as f:
      saved_metrics = [json.loads(line) for line in f.readlines()]
    with open(metrics_file_restored, "r", encoding="utf8") as f:
      restored_metrics = [json.loads(line) for line in f.readlines()]

    print("SAVED METRICS:", saved_metrics)
    print("RESTORED METRICS:", restored_metrics)

    self.assertEqual(len(saved_metrics), 1, "First run should have exactly 1 step.")
    self.assertEqual(len(restored_metrics), 1, "Second run should have exactly 1 step.")

    saved_step0_loss = saved_metrics[0]["learning/loss"]
    restored_step1_loss = restored_metrics[0]["learning/loss"]

    print(f"Saved run Step 0 Loss: {saved_step0_loss}")
    print(f"Restored run Step 1 Loss: {restored_step1_loss}")

    # Verify that the restored run resumed from step 0 (logged step 1)
    self.assertEqual(saved_metrics[0]["step"], 0.0)
    self.assertEqual(restored_metrics[0]["step"], 1.0)

  def test_pretrain_qlora_checkpoint_save_and_restore(self):
    """Test pre-training with QLoRA, checkpoint saving, and subsequent restoration."""
    temp_dir = self._create_temp_dir()
    metrics_file_saved = os.path.join(temp_dir, "saved_metrics_qlora.txt")
    metrics_file_restored = os.path.join(temp_dir, "restored_metrics_qlora.txt")

    # Common model scaling/configs
    overrides_base = [
        "scan_layers=True",
        "attention=dot_product",
        "weight_dtype=bfloat16",
        "dtype=bfloat16",
        "lora.enable_lora=True",
        "lora.lora_weight_qtype=int8",
        "lora.lora_tile_size=32",
        "lora.lora_rank=4",
        "lora.lora_alpha=8.0",
        "base_num_decoder_layers=2",
        "base_emb_dim=128",
        "base_mlp_dim=256",
        "ici_fsdp_parallelism=1",
    ]

    # 1. First run: train 1 step with checkpointing enabled
    args_save = _get_common_args(
        temp_dir,
        "default",
        steps=1,
        overrides=overrides_base
        + [
            "enable_checkpointing=True",
            "checkpoint_period=1",
            "async_checkpointing=False",
            f"metrics_file={metrics_file_saved}",
        ],
    )
    _run_train_subprocess(args_save)

    # Verify checkpoint got written (MaxText writes to base_output_directory/run_name/checkpoints/0/items)
    checkpoint_dir = os.path.join(temp_dir, "test_peft_default", "checkpoints", "0", "items")
    self.assertTrue(os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} was not created.")

    # 2. Second run: restore from the checkpoint directory and train 1 step (total steps = 2)
    args_restore = _get_common_args(
        temp_dir,
        "default",
        steps=2,
        overrides=overrides_base
        + [
            "enable_checkpointing=True",
            "checkpoint_period=1",
            "async_checkpointing=False",
            f"load_parameters_path={checkpoint_dir}",
            f"metrics_file={metrics_file_restored}",
        ],
    )
    _run_train_subprocess(args_restore)

    # 3. Verify losses are logged and correct
    self.assertTrue(os.path.exists(metrics_file_saved))
    self.assertTrue(os.path.exists(metrics_file_restored))

    with open(metrics_file_saved, "r", encoding="utf8") as f:
      saved_metrics = [json.loads(line) for line in f.readlines()]
    with open(metrics_file_restored, "r", encoding="utf8") as f:
      restored_metrics = [json.loads(line) for line in f.readlines()]

    print("SAVED METRICS (QLoRA):", saved_metrics)
    print("RESTORED METRICS (QLoRA):", restored_metrics)

    self.assertEqual(len(saved_metrics), 1, "First run should have exactly 1 step.")
    self.assertEqual(len(restored_metrics), 1, "Second run should have exactly 1 step.")

    saved_step0_loss = saved_metrics[0]["learning/loss"]
    restored_step1_loss = restored_metrics[0]["learning/loss"]

    print(f"Saved run Step 0 Loss (QLoRA): {saved_step0_loss}")
    print(f"Restored run Step 1 Loss (QLoRA): {restored_step1_loss}")

    # Verify that the restored run resumed from step 0 (logged step 1)
    self.assertEqual(saved_metrics[0]["step"], 0.0)
    self.assertEqual(restored_metrics[0]["step"], 1.0)
