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

from MaxText.globals import PKG_DIR
from MaxText.train import main as train_main
from MaxText.decode import main as decode_main
from MaxText.generate_param_only_checkpoint import main as generate_param_only_ckpt_main
from MaxText.tests.integration_tests.checkpointing_test import get_checkpointing_command


def run_generate_param_only_checkpoint(hardware, attention_type, quantization):
  """Tests generating a parameter-only checkpoint."""

  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  model_params = [
      f"quantization={quantization}",
      "base_emb_dim=384",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "base_mlp_dim=192",
      "base_num_decoder_layers=8",
      "head_dim=128",
  ]

  train_main(
      get_checkpointing_command(
          run_date,
          hardware=hardware,
          steps=1,
          metrics_file="run_metrics.txt",
          attention_type=attention_type,
          dataset_type="tfds",
          dataset_path="gs://maxtext-dataset",
      )
      + model_params
  )

  state_path = f"gs://runner-maxtext-logs/runner_{run_date}/checkpoints/0/items"
  generate_param_only_ckpt_main(
      [
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          f"hardware={hardware}",
          f"run_name=generate_param_{run_date}",
          "base_output_directory=gs://runner-maxtext-logs",
          "dataset_path=gs://maxtext-dataset",
          "async_checkpointing=False",
          f"attention={attention_type}",
          f"load_full_state_path={state_path}",
      ]
      + model_params
  )

  decode_ckpt_path = f"gs://runner-maxtext-logs/generate_param_{run_date}/checkpoints/0/items"
  decode_main(
      [
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          f"hardware={hardware}",
          f"run_name=decode_{run_date}",
          "base_output_directory=gs://runner-maxtext-logs",
          "dataset_path=gs://maxtext-dataset",
          f"load_parameters_path={decode_ckpt_path}",
          f"attention={attention_type}",
          "max_target_length=128",
          "ici_tensor_parallelism=4",
          "per_device_batch_size=1",
      ]
      + model_params
  )


@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.parametrize("quantization", [(""), ("int8")])
def test_autoselected_attention(quantization, capsys):
  run_generate_param_only_checkpoint("tpu", "autoselected", quantization)
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out


@pytest.mark.integration_test
@pytest.mark.gpu_only
@pytest.mark.parametrize("quantization", [(""), ("int8")])
def test_with_dot_product(quantization, capsys):
  os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
  run_generate_param_only_checkpoint("gpu", "dot_product", quantization)
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out
