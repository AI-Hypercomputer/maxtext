# Copyright 2023–2025 Google LLC
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

"""
Integration tests for generating a decode-only checkpoint from a training checkpoint
and then running decode with it.
"""
from datetime import datetime
import os
import pytest

from MaxText.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_PKG_DIR
from MaxText.train import main as train_main
from MaxText.decode import main as decode_main
from MaxText.generate_param_only_checkpoint import main as generate_param_only_ckpt_main
from tests.integration_tests.checkpointing_test import get_checkpointing_command


def get_model_params(quantization):
  return [
      f"quantization={quantization}",
      "base_emb_dim=384",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "base_mlp_dim=192",
      "base_num_decoder_layers=8",
      "head_dim=128",
  ]


def run_e2e_test_flow(hardware, model_config, attention_type="autoselected", state_path=None):
  """Helper function to run training, generate parameter-only checkpoint, and decode."""
  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  test_config = [
      None,
      os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
      "base_output_directory=gs://runner-src/MaxText-logs",
      "async_checkpointing=False",
      f"hardware={hardware}",
      f"attention={attention_type}",
      "max_target_length=128",
      "per_device_batch_size=1",
  ] + model_config

  if state_path is None:
    # Run training to get a checkpoint
    train_main(
        get_checkpointing_command(
            run_date=run_date,
            hardware=hardware,
            steps=1,
            metrics_file="run_metrics.txt",
            attention_type=attention_type,
            dataset_type="tfds",
            dataset_path="gs://src/MaxText-dataset",
        )
    )
    state_path = f"gs://runner-src/MaxText-logs/runner_{run_date}/checkpoints/0/items"

  # Generate parameter-only checkpoint
  generate_param_only_ckpt_config = test_config + [
      f"run_name=generate_param_{run_date}",
      f"load_full_state_path={state_path}",
  ]
  generate_param_only_ckpt_main(generate_param_only_ckpt_config)

  # Run inference on parameter-only checkpoint
  decode_config = test_config + [
      f"run_name=decode_{run_date}",
      f"load_parameters_path=gs://runner-src/MaxText-logs/generate_param_{run_date}/checkpoints/0/items",
  ]
  decode_main(decode_config)


@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.parametrize("quantization", [(""), ("int8")])
def test_param_ckpt_generation_with_autoselected_attention(quantization, capsys):
  """Tests the parameter-only checkpoint generation and decode flow on TPU with autoselected attention."""
  model_config = get_model_params(quantization)
  run_e2e_test_flow(hardware="tpu", attention_type="autoselected", model_config=model_config)
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out


@pytest.mark.integration_test
@pytest.mark.gpu_only
@pytest.mark.parametrize("quantization", [(""), ("int8")])
def test_param_ckpt_generation_with_dot_product(quantization, capsys):
  """Tests the parameter-only checkpoint generation and decode flow on GPU with dot product attention."""
  os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
  model_config = get_model_params(quantization)
  run_e2e_test_flow(hardware="gpu", attention_type="dot_product", model_config=model_config)
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out


@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.scheduled_only
def test_param_ckpt_generation_with_pre_generated_ckpt(capsys):
  """Tests the parameter-only checkpoint generation and decode flow with a pre-generated Gemma-2b model checkpoint."""
  model_config = [
      "model_name=gemma-2b",
      f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.gemma')}",
  ]
  run_e2e_test_flow(
      hardware="tpu",
      model_config=model_config,
      state_path="gs://runner-src/MaxText-logs/runner_finetune_2025-08-15-04-05/checkpoints/5/items",
  )
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out
