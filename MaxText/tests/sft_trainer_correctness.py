#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Runs SFT trainer correctness with TRL implementation.

Usage:
python3 -m MaxText.tests.sft_trainer_correctness --model-name=llama3.1-8b --tokenizer-path=meta-llama/Llama-3.1-8B --model-ckpt-path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items
"""

import argparse
import jax
import jax.numpy as jnp
import jsonlines
import numpy as np
import os
import sys

from jax.sharding import Mesh
from transformers import AutoTokenizer

from MaxText.globals import PKG_DIR

from MaxText import max_logging
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.layers import models
from MaxText.layers import quantizations


def initialize_config(config):
  return pyconfig.initialize(
      [sys.argv[0], os.path.join(PKG_DIR, "configs", "sft.yml")],
      run_name="test-sft-trainer-correctness",
      model_name=config.model_name,
      tokenizer_path=config.tokenizer_path,
      enable_checkpointing=True,
      load_parameters_path=config.model_ckpt_path,
      max_target_length=config.max_target_length,
      per_device_batch_size=1,
      max_prefill_predict_length=32,
      dataset_type="synthetic",
      dtype="float32",
      matmul_precision="high",
      logits_dot_in_fp32=True,
      skip_jax_distributed_system=True,
  )


def get_golden_data(config):
  """Get the golden data for SFTTrainer in TRL."""
  golden_data_path = os.path.join(PKG_DIR, "test_assets", f"golden_data_sft_{config.model_name}.jsonl")
  with jsonlines.open(golden_data_path, "r") as f:
    golden_data = list(f)
  return golden_data[0]


def setup_maxtext_model(config):
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  init_rng, rng1 = jax.random.split(init_rng)
  quant = quantizations.configure_quantization(config)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  maxtext_model = models.Transformer(config=config, mesh=mesh, quant=quant)
  state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng1, mesh, None)
  return maxtext_model, state, init_rng


def prepare_maxtext_inputs(maxtext_data, config):
  data = _input_pipeline_utils.extract_messages_and_mask(maxtext_data, "messages")
  tokenizer = AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      add_bos_token=False,
      add_eos_token=False,
      model_max_length=config.max_target_length,
  )
  tokenized_data = _input_pipeline_utils.tokenization(
      data,
      hf_tokenizer=tokenizer,
      truncation=False,
      max_length=config.max_target_length,
      column_names=["messages"],
  )
  masked_inputs = _input_pipeline_utils.SFTPromptMasking(
      text_column_name="messages",
      completion_only=False,
      max_target_length=config.max_target_length,
      add_bos=True,
      add_eos=True,
      bos_id=tokenizer.bos_token_id,
      eos_id=tokenizer.eos_token_id,
      unk_id=tokenizer.unk_token_id,
  ).map(tokenized_data)

  global_batch_size = int(jax.device_count() * config.per_device_batch_size * config.gradient_accumulation_steps)
  inputs = jnp.stack([np.asarray(masked_inputs["inputs"], dtype=np.int32) for _ in range(global_batch_size)])
  inputs_segmentation = jnp.stack([(masked_inputs["inputs"] != 0).astype(np.int32) for _ in range(global_batch_size)])
  inputs_position = jnp.stack(
      [np.arange(masked_inputs["inputs"].shape[0], dtype=np.int32) for _ in range(global_batch_size)]
  )
  return inputs, inputs_segmentation, inputs_position


def get_maxtext_logits(inputs, inputs_position, inputs_segmentation, config):
  maxtext_model, state, rng = setup_maxtext_model(config)
  maxtext_logits, _ = maxtext_model.apply(
      state.params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=False,
      rngs={"aqt": rng},
      mutable="intermediates",
  )
  return maxtext_logits


def get_kl_div(maxtext_logits, hf_logits):
  maxtext_probabilities = jax.nn.softmax(maxtext_logits, axis=-1)
  hf_probabilities = jax.nn.softmax(hf_logits, axis=-1)
  kl_div = jax.numpy.sum(jax.scipy.special.kl_div(hf_probabilities, maxtext_probabilities), axis=-1)
  return kl_div


def main(config, test_args):
  golden_data = get_golden_data(config)
  inputs, inputs_segmentation, inputs_position = prepare_maxtext_inputs(golden_data["data"], config)
  maxtext_data = {
      "inputs": inputs,
      "inputs_segmentation": inputs_segmentation,
      "inputs_position": inputs_position,
  }
  maxtext_logits = get_maxtext_logits(inputs, inputs_position, inputs_segmentation, config)

  assert golden_data["tokens"] == maxtext_data["inputs"][0].tolist()
  assert golden_data["attention_mask"] == maxtext_data["inputs_segmentation"][0].tolist()

  trl_logits = np.array(golden_data["trl_logits"])

  max_logging.log(f"Max numerical difference: {np.max(np.abs(np.subtract(maxtext_logits[0], trl_logits)))}")
  assert jax.numpy.allclose(
      maxtext_logits[0],
      trl_logits,
      rtol=float(test_args.rtol),
      atol=float(test_args.atol),
      equal_nan=False,
  )

  kl_div = get_kl_div(maxtext_logits, trl_logits)
  max_logging.log(f"KL divergence: {kl_div}, max KL divergence: {jnp.max(kl_div)}")
  assert jax.numpy.all(kl_div < float(test_args.kl_div))


def get_argument_parser():
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument("--model-name", type=str, required=True)
  argument_parser.add_argument("--tokenizer-path", type=str, required=True)
  argument_parser.add_argument("--model-ckpt-path", type=str, required=True)
  argument_parser.add_argument("--max-target-length", type=int, required=False, default=64)
  argument_parser.add_argument("--atol", type=float, required=True)
  argument_parser.add_argument("--rtol", type=float, required=True)
  argument_parser.add_argument("--kl-div", type=float, required=True)
  return argument_parser


if __name__ == "__main__":
  parser = get_argument_parser()
  test_args = parser.parse_args(sys.argv[1:])
  config = initialize_config(test_args)
  main(config, test_args)
