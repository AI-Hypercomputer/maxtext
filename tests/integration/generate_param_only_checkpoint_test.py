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

from maxtext.inference.decode import main as decode_main
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.utils.generate_param_only_checkpoint import main as generate_param_only_ckpt_main
from tests.integration.checkpointing_test import get_checkpointing_command
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory


def get_model_params(quantization):
  return [
      f"quantization={quantization}",
      "base_emb_dim=128",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "base_mlp_dim=128",
      "base_num_decoder_layers=1",
      "head_dim=64",
  ]


def run_e2e_test_flow(
    hardware,
    model_config,
    attention_type="autoselected",
    state_path=None,
    train_attention_type=None,
    decode_config_extra=None,
):
  """Helper function to run training, generate parameter-only checkpoint, and decode."""
  base_output_directory = get_test_base_output_directory()
  dataset_path = get_test_dataset_path()
  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  train_attention = train_attention_type if train_attention_type is not None else attention_type
  decode_extra = decode_config_extra or []
  shared_config = [
      None,
      get_test_config_path(),
      f"base_output_directory={base_output_directory}",
      "async_checkpointing=False",
      "enable_checkpointing=True",
      f"hardware={hardware}",
      "max_target_length=128",
      "per_device_batch_size=1",
  ] + model_config

  pathways_command = []
  if os.getenv("JAX_PLATFORMS") == "proxy":
    pathways_command = ["enable_single_controller=True"]

  if state_path is None:
    # Run training to get a checkpoint
    train_main(
        get_checkpointing_command(
            run_date=run_date,
            hardware=hardware,
            steps=1,
            metrics_file="run_metrics.txt",
            attention_type=train_attention,
            dataset_type="synthetic",
            dataset_path=dataset_path,
        )
    )
    state_path = f"{base_output_directory}/runner_{run_date}/checkpoints/0/items"

  # Generate parameter-only checkpoint
  generate_param_only_ckpt_config = (
      shared_config
      + [
          f"attention={train_attention}",
          f"run_name=generate_param_{run_date}",
          f"load_full_state_path={state_path}",
      ]
      + pathways_command
  )
  generate_param_only_ckpt_main(generate_param_only_ckpt_config)

  # Run inference on parameter-only checkpoint
  decode_config = (
      shared_config
      + [
          f"attention={attention_type}",
          f"run_name=decode_{run_date}",
          f"load_parameters_path={base_output_directory}/generate_param_{run_date}/checkpoints/0/items",
      ]
      + decode_extra
      + pathways_command
  )
  decode_main(decode_config)


@pytest.mark.skipif(
    is_decoupled(),
    reason="Bypassed in offline decoupled runs (no GCS/internet)",
)
@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.parametrize(
    "quantization",
    [
        "",
        pytest.param(
            "int8",
            marks=pytest.mark.skip(
                reason=(
                    "NNX int8 param-only generation is a convert-on-load case"
                    " (the fp32 training checkpoint has no AqtDotGeneral state"
                    " the int8 model expects); tracked as a follow-up alongside"
                    " layerwise_quantization."
                )
            ),
        ),
    ],
)
def test_param_ckpt_generation_with_autoselected_attention(quantization, capsys):
  """Tests the parameter-only checkpoint generation and decode flow on TPU with autoselected attention."""
  model_config = get_model_params(quantization)
  run_e2e_test_flow(hardware="tpu", attention_type="autoselected", model_config=model_config)
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out


@pytest.mark.external_serving
@pytest.mark.integration_test
@pytest.mark.gpu_only
@pytest.mark.parametrize(
    "quantization",
    [
        "",
        pytest.param(
            "int8",
            marks=pytest.mark.skip(
                reason=(
                    "NNX int8 param-only generation is a convert-on-load case"
                    " (the fp32 training checkpoint has no AqtDotGeneral state"
                    " the int8 model expects); tracked as a follow-up alongside"
                    " layerwise_quantization."
                )
            ),
        ),
    ],
)
def test_param_ckpt_generation_with_dot_product(quantization, capsys):
  """Tests the parameter-only checkpoint generation and decode flow on GPU with dot product attention."""
  os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
  model_config = get_model_params(quantization)
  run_e2e_test_flow(hardware="gpu", attention_type="dot_product", model_config=model_config)
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out


AR_SDPA_DECODE_OVERRIDES = [
    "ici_fsdp_parallelism=1",
    "ici_autoregressive_parallelism=-1",
    "skip_jax_distributed_system=True",
    "max_prefill_predict_length=8",
    "max_target_length=16",
]


@pytest.mark.external_serving
@pytest.mark.integration_test
@pytest.mark.gpu_only
def test_param_ckpt_generation_with_cudnn_flash_jax_ar_decode(capsys):
  model_config = get_model_params("")
  run_e2e_test_flow(
      hardware="gpu",
      attention_type="cudnn_flash_jax",
      train_attention_type="dot_product",
      model_config=model_config,
      decode_config_extra=AR_SDPA_DECODE_OVERRIDES,
  )
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out


@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.scheduled_only
@pytest.mark.external_serving  # Requires pre-generated checkpoint (Gemma-2b)
def test_param_ckpt_generation_with_pre_generated_ckpt(capsys):
  """Tests the parameter-only checkpoint generation and decode flow with a pre-generated Gemma-2b model checkpoint."""
  model_config = [
      "model_name=gemma-2b",
      f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.gemma')}",
  ]
  run_e2e_test_flow(
      hardware="tpu",
      model_config=model_config,
      state_path="gs://runner-maxtext-logs/runner_finetune_2025-08-15-04-05/checkpoints/5/items",
  )
  captured = capsys.readouterr()
  expected_output = "Input `I love to`"
  assert expected_output in captured.out
