# Copyright 2023–2026 Google LLC
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

"""Script to check correctness of `dpo_trainer` in MaxText with `DPOTrainer` in TRL & generate golden data for `dpo_trainer`.

Usage:

To generate golden data for default configuration:
  python3 -m tests.assets.logits_generation.generate_dpo_golden_data

To check correctness against TRL for Llama2-7b:
  python3 -m tests.assets.logits_generation.generate_dpo_golden_data

To check correctness against TRL for other models (e.g., ungated deepseek):
  python3 -m tests.assets.logits_generation.generate_dpo_golden_data \
    --model-name=deepseek2-16b \
    --tokenizer-path=deepseek-ai/DeepSeek-V2-Lite-chat \
    --hf-model-path=deepseek-ai/DeepSeek-V2-Lite-chat \
    --model-ckpt-path=<MaxText-compatible checkpoint for the model>
"""

import argparse
import functools
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import jsonlines
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.input_pipeline import dpo_utils
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.trainers.post_train.dpo import dpo_utils as post_train_dpo_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_PKG_DIR, MAXTEXT_TEST_ASSETS_ROOT


DATA = {
    "prompt": ["Context: What is the capital of France?"],
    "chosen": [" Answer: Paris"],
    "rejected": [" Answer: London"],
}


def initialize_maxtext_config(config):
  """Initializes configuration for MaxText."""
  cfg_with_ckpt = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "post_train", "dpo.yml")],
      run_name="compare_maxtext_with_trl_logits",
      model_name=config.model_name,
      tokenizer_path=config.tokenizer_path,
      enable_checkpointing=bool(config.model_ckpt_path),
      load_parameters_path=config.model_ckpt_path,
      max_target_length=32,
      per_device_batch_size=2,
      dataset_type="synthetic",
      dtype="float32",
      matmul_precision="high",
      logits_dot_in_fp32=True,
  )

  cfg_without_ckpt = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "post_train", "dpo.yml")],
      run_name="generate_dpo_golden_data",
      model_name="default",
      enable_checkpointing=False,
      max_target_length=32,
      per_device_batch_size=2,
      dataset_type="synthetic",
      dtype="float32",
      matmul_precision="high",
      logits_dot_in_fp32=True,
  )
  return cfg_with_ckpt, cfg_without_ckpt


def get_hf_model(model_path):
  """Load model from Hugging Face."""
  return AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=torch.float32,
  )


def get_tokenizer(tokenizer_path, max_target_length):
  """Get tokenizer from Hugging Face."""
  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_path,
      add_bos_token=False,
      add_eos_token=False,
      model_max_length=max_target_length,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def setup_dpo_trainer(hf_model, tokenizer, max_target_length):
  """Setup DPO Trainer in TRL."""
  training_args = TrainingArguments(
      per_device_train_batch_size=1,
      bf16=True,
      use_cpu=True,
      output_dir="/tmp/dpo_trainer",
  )
  dataset = Dataset.from_dict({"prompt": ["dummy"], "chosen": ["dummy"], "rejected": ["dummy"]})
  return DPOTrainer(
      model=hf_model,
      ref_model=None,
      processing_class=tokenizer,
      train_dataset=dataset,
      args=DPOConfig(
          max_length=max_target_length,
          **training_args.to_dict(),
      ),
  )


def prepare_trl_inputs(tokenizer_path, max_target_length):
  """Get tokenized inputs for TRL."""
  tokenizer = get_tokenizer(tokenizer_path, max_target_length)

  sample = {
      "chosen": np.array(tokenizer.encode(DATA["prompt"][0] + DATA["chosen"][0])),
      "rejected": np.array(tokenizer.encode(DATA["prompt"][0] + DATA["rejected"][0])),
  }
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  prep = dpo_utils.DPOTunixPrep(
      pad_id=pad_id, max_target_length=max_target_length, data_column_names=("chosen", "rejected")
  )
  batch = prep.map(sample)

  chosen_inputs = torch.tensor(
      [np.concatenate([batch["prompt_ids"], batch["chosen_ids"]], axis=-1)],
      dtype=torch.long,
  )
  rejected_inputs = torch.tensor(
      [np.concatenate([batch["prompt_ids"], batch["rejected_ids"]], axis=-1)],
      dtype=torch.long,
  )

  chosen_mask = torch.tensor(
      [np.concatenate([batch["prompt_mask"], batch["chosen_mask"]], axis=-1)],
      dtype=torch.long,
  )
  rejected_mask = torch.tensor(
      [np.concatenate([batch["prompt_mask"], batch["rejected_mask"]], axis=-1)],
      dtype=torch.long,
  )

  chosen_completion_mask = torch.tensor(
      [
          np.concatenate(
              [np.zeros_like(batch["prompt_ids"]), np.ones_like(batch["chosen_ids"])],
              axis=-1,
          )
      ],
      dtype=torch.long,
  )
  rejected_completion_mask = torch.tensor(
      [
          np.concatenate(
              [np.zeros_like(batch["prompt_ids"]), np.ones_like(batch["rejected_ids"])],
              axis=-1,
          )
      ],
      dtype=torch.long,
  )

  return {
      "input_ids": torch.cat([chosen_inputs, rejected_inputs], dim=0),
      "attention_mask": torch.cat([chosen_mask, rejected_mask], dim=0),
      "completion_mask": torch.cat([chosen_completion_mask, rejected_completion_mask], dim=0),
  }


def setup_maxtext_model(config):
  """Setup MaxText model."""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  quant = quantizations.configure_quantization(config)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  maxtext_model = models.transformer_as_linen(config=config, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
  init_state_fn = functools.partial(maxtext_utils.init_initial_state, maxtext_model, None, config, False, init_rng)
  state, _ = maxtext_utils.setup_decode_state(config, mesh, None, init_state_fn)
  return maxtext_model, state, init_rng


def get_maxtext_logits_and_loss(config, maxtext_data):
  """Get logits and loss generated by MaxText."""
  maxtext_model, state, rng = setup_maxtext_model(config)
  maxtext_logits, _ = maxtext_model.apply(
      state.params,
      maxtext_data["inputs"],
      maxtext_data["inputs_position"],
      decoder_segment_ids=maxtext_data["inputs_segmentation"],
      enable_dropout=False,
      rngs=rng,
      mutable="intermediates",
  )
  data = {
      "chosen": maxtext_data["inputs"][0:1],
      "rejected": maxtext_data["inputs"][1:2],
      "chosen_segmentation": maxtext_data["inputs_segmentation"][0:1],
      "rejected_segmentation": maxtext_data["inputs_segmentation"][1:2],
      "chosen_position": maxtext_data["inputs_position"][0:1],
      "rejected_position": maxtext_data["inputs_position"][1:2],
  }
  loss, _ = post_train_dpo_utils.dpo_loss_fn(
      model=maxtext_model,
      config=config,
      data=data,
      dropout_rng=rng,
      params=state.params,
      reference_params=state.params["params"],
      is_train=False,
  )
  return maxtext_logits, float(loss)


def get_trl_logits_and_loss(config, trl_data, max_target_length):
  """Get logits and loss generated by TRL/HF."""
  hf_model = get_hf_model(config.hf_model_path)
  tokenizer = get_tokenizer(config.tokenizer_path, max_target_length)
  trl_trainer = setup_dpo_trainer(hf_model, tokenizer, max_target_length)
  loss, trl_outputs = trl_trainer.compute_loss(hf_model, trl_data, return_outputs=True)
  trl_logits = trl_outputs.logits.detach().numpy()
  return trl_logits, float(loss.detach().item())


def prepare_maxtext_inputs(maxtext_data, config):
  """Get tokenized inputs for MaxText."""
  tokenizer = get_tokenizer(config.tokenizer_path, config.max_target_length)

  sample = {
      "chosen": np.array(tokenizer.encode(maxtext_data["prompt"][0] + maxtext_data["chosen"][0])),
      "rejected": np.array(tokenizer.encode(maxtext_data["prompt"][0] + maxtext_data["rejected"][0])),
  }

  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  prep = dpo_utils.DPOTunixPrep(
      pad_id=pad_id, max_target_length=config.max_target_length, data_column_names=("chosen", "rejected")
  )
  batch = prep.map(sample)

  chosen_inputs = np.concatenate([batch["prompt_ids"], batch["chosen_ids"]], axis=-1)
  rejected_inputs = np.concatenate([batch["prompt_ids"], batch["rejected_ids"]], axis=-1)
  chosen_mask = np.concatenate([batch["prompt_mask"], batch["chosen_mask"]], axis=-1)
  rejected_mask = np.concatenate([batch["prompt_mask"], batch["rejected_mask"]], axis=-1)

  inputs = jnp.stack([chosen_inputs, rejected_inputs])
  inputs_segmentation = jnp.stack([chosen_mask, rejected_mask])
  inputs_position = jnp.stack([jnp.arange(config.max_target_length, dtype=jnp.int32)] * 2)

  return {
      "inputs": inputs,
      "inputs_segmentation": inputs_segmentation,
      "inputs_position": inputs_position,
      "batch": batch,
  }


def get_token_log_probs(logits, inputs):
  """Computes per-token log probabilities."""
  targets = inputs[:, 1:]
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  log_probs = log_probs[:, :-1, :]
  token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  return token_log_probs


def test_with_trl_and_save_golden_data(config):
  """Compare input data and logits generated by MaxText with TRL and save golden data."""
  if not os.path.exists(config.tokenizer_path):
    print(f"Downloading tokenizer to {config.tokenizer_path}...")
    subprocess.call(
        [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            "gs://maxtext-dataset/hf/llama2-chat-tokenizer",
            os.path.join(MAXTEXT_ASSETS_ROOT, ""),
        ]
    )

  maxtext_config_with_ckpt, maxtext_config_without_ckpt = initialize_maxtext_config(config)
  maxtext_data = prepare_maxtext_inputs(dict(DATA), maxtext_config_with_ckpt)

  trl_data = prepare_trl_inputs(config.tokenizer_path, maxtext_config_with_ckpt.max_target_length)

  # Compare input tokens generated by TRL and MaxText
  assert trl_data["input_ids"][0].tolist() == maxtext_data["inputs"][0].tolist()
  assert trl_data["input_ids"][1].tolist() == maxtext_data["inputs"][1].tolist()

  # Compare logits and loss generated by TRL and MaxText
  trl_logits, trl_loss = get_trl_logits_and_loss(config, trl_data, maxtext_config_with_ckpt.max_target_length)
  maxtext_logits, maxtext_loss = get_maxtext_logits_and_loss(maxtext_config_with_ckpt, maxtext_data)
  assert jax.numpy.allclose(
      maxtext_logits,
      trl_logits,
      rtol=1e-05,
      atol=0.09,
      equal_nan=False,
  )
  assert jax.numpy.allclose(
      maxtext_loss,
      trl_loss,
      rtol=1e-05,
      atol=1e-05,
      equal_nan=False,
  )

  # With MaxText's implementation verified, create a model without a checkpoint and save its per-token log probabilities
  maxtext_logits_no_ckpt, _ = get_maxtext_logits_and_loss(maxtext_config_without_ckpt, maxtext_data)
  token_log_probs = get_token_log_probs(maxtext_logits_no_ckpt, maxtext_data["inputs"])

  # Prompt length is 16, so response log probs start at index 15
  response_log_probs = token_log_probs[:, 15:]
  batch = maxtext_data["batch"]
  chosen_logps = jnp.sum(response_log_probs[0] * batch["chosen_mask"], axis=-1)
  rejected_logps = jnp.sum(response_log_probs[1] * batch["rejected_mask"], axis=-1)

  data_to_save = {
      "sample": {"chosen": DATA["prompt"][0] + DATA["chosen"][0], "rejected": DATA["prompt"][0] + DATA["rejected"][0]},
      "inputs": maxtext_data["inputs"].tolist(),
      "inputs_segmentation": maxtext_data["inputs_segmentation"].tolist(),
      "token_log_probs": token_log_probs.tolist(),
      "chosen_logps": float(chosen_logps),
      "rejected_logps": float(rejected_logps),
  }

  model_output_path = os.path.join(
      MAXTEXT_TEST_ASSETS_ROOT, f"golden_data_dpo_{maxtext_config_without_ckpt.model_name}.jsonl"
  )
  with jsonlines.open(model_output_path, "w") as f:
    f.write(data_to_save)
  print(f"Successfully saved DPO golden data to {model_output_path}")


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", type=str, required=False, default="llama2-7b")
  parser.add_argument(
      "--tokenizer-path", type=str, required=False, default=os.path.join(MAXTEXT_ASSETS_ROOT, "llama2-chat-tokenizer")
  )
  parser.add_argument("--hf-model-path", type=str, required=False, default="meta-llama/Llama-2-7b-chat-hf")
  parser.add_argument(
      "--model-ckpt-path",
      type=str,
      required=False,
      default="",
  )

  dpo_config_args = parser.parse_args(sys.argv[1:])
  test_with_trl_and_save_golden_data(dpo_config_args)
