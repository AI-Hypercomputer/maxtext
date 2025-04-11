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
Runs GRPO trainer unit test correctness with golden logits generated from maxtext/MaxText/scratch_code/generate_grpo_golden_logits.py

Usage:
python3 -m MaxText.tests.grpo_trainer_correctness_test --model-name=llama3.1-8b --tokenizer-path=meta-llama/Llama-3.1-8B
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

current_dir = os.path.dirname(os.path.abspath(__file__))
maxtext_parent_dir = os.path.dirname(current_dir)
sys.path.append(maxtext_parent_dir)

from MaxText import max_logging
from MaxText import max_utils
from MaxText import pyconfig
from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.layers import models
from MaxText.layers import quantizations

from MaxText.experimental.rl.grpo_trainer import compute_log_probs, grpo_loss_fn, _merge_grpo_state, generate_completions
from MaxText import maxengine
import transformers


def initialize_config(config):
  return pyconfig.initialize(
      # [sys.argv[0], os.path.join(PKG_DIR, "configs", "experimental/rl/grpo.yml")],
      [None, "MaxText/experimental/rl/grpo.yml"],
      run_name="test-grpo-trainer-correctness-test",
      model_name=config.model_name,
      tokenizer_path=config.tokenizer_path,
      enable_checkpointing=False,
      # load_parameters_path=config.model_ckpt_path,
      max_target_length=32,
      per_device_batch_size=1,
      max_prefill_predict_length=16,
      dataset_type="synthetic",
      dtype="float32",
      matmul_precision="high",
      logits_dot_in_fp32=True,
      skip_jax_distributed_system=True,
      init_weights_seed=42,
      base_num_decoder_layers=8,
      base_emb_dim=1024,
  )


def get_golden_data(config):
  """Get the golden data for GrpoTrainer from maxtext/MaxText/scratch_code/generate_grpo_golden_logits.py."""
  golden_data_path = os.path.join(PKG_DIR, "test_assets", f"golden_data_grpo_{config.model_name}.jsonl")
  with jsonlines.open(golden_data_path, "r") as f:
    golden_data = list(f)
  return golden_data[0]


def setup_maxtext_model(config):
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  init_rng, rng1 = jax.random.split(init_rng)
  quant = quantizations.configure_quantization(config)
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  maxtext_model = models.Transformer(config=config, mesh=mesh, quant=quant)
  state, _ = max_utils.setup_decode_state(maxtext_model, config, rng1, mesh, None)
  reference_params = jax.tree.map(jnp.copy, state.params["params"])
  state = _merge_grpo_state(state, reference_params)
  tokenizer_model = transformers.AutoTokenizer.from_pretrained(
      "meta-llama/Llama-3.1-8B",
      add_bos_token=config.add_bos,
      add_eos_token=config.add_eos,
      model_max_length=config.max_target_length,
      legacy=False,
      token=config.hf_access_token,
      padding_side="left",
  )
  tokenizer_model.add_special_tokens({"pad_token": "<pad>"})
  return maxtext_model, state, reference_params, init_rng, tokenizer_model


def _prepare_maxtext_inputs(input_str, tokenizer_model):
  prompt = tokenizer_model.encode(input_str)
  input_ids = jnp.pad(
      jnp.tile(jnp.concat([jnp.array(prompt), jnp.array(prompt)], axis=-1), (4, 1)),
      ((0, 0), (0, 4)),
      constant_values=tokenizer_model.pad_token_type_id,
  )  # pad some tokens at the end of input prompt
  input_segmentation = (input_ids > 0).astype(jnp.int32)
  input_position = jnp.where(input_segmentation, jnp.arange(input_segmentation.shape[1]), 0)
  completion_segmentation = jnp.tile(
      jnp.pad(jnp.array([0] * len(prompt) + [1] * len(prompt)), (0, input_ids.shape[1] - 2 * len(prompt))), (4, 1)
  )
  return input_ids, input_segmentation, input_position, completion_segmentation


def get_maxtext_logits(maxtext_model, state, input_ids, input_position, input_segmentation, completion_segmentation, config):

  maxtext_per_token_logps, _ = compute_log_probs(
      maxtext_model,
      state.params,
      input_ids,
      input_position,
      input_segmentation,
      completion_segmentation,
      config,
      is_train=False,
  )
  return maxtext_per_token_logps.cpu().numpy().astype("float32")


def main(config, config_inference, test_args):
  golden_data = get_golden_data(config)
  maxtext_model, state, reference_params, rng, tokenizer_model = setup_maxtext_model(config)
  input_ids, input_segmentation, input_position, completion_segmentation = _prepare_maxtext_inputs(
      config.prompt, tokenizer_model
  )
  maxtext_per_token_logps = get_maxtext_logits(
      maxtext_model, state, input_ids, input_position, input_segmentation, completion_segmentation, config
  )

  # get maxtext grpo loss
  data = {
      "prompt_completions": input_ids,
      "prompt_completions_position": input_position,
      "prompt_completions_segmentation": input_segmentation,
      "ar_completions_segmentation": completion_segmentation,
  }
  maxtext_loss, aux = grpo_loss_fn(maxtext_model, config, data, rng, state.params, reference_params)

  assert maxtext_loss.tolist() == golden_data["maxtext_loss"]
  assert aux["avg_kl"].tolist() == golden_data["avg_kl"]
  golden_maxtext_logits = np.array(golden_data["maxtext_per_token_logps_no_ckpt_loading"])

  max_logging.log(
      f"Max numerical difference: {np.max(np.abs(np.subtract(maxtext_per_token_logps[0], golden_maxtext_logits)))}"
  )
  assert jax.numpy.allclose(
      maxtext_per_token_logps[0],
      golden_maxtext_logits,
      rtol=float(test_args.rtol),
      atol=float(test_args.atol),
      equal_nan=False,
  )

  # get_generated_completions(config_inference):
  engine = maxengine.MaxEngine(config_inference)
  generated_completions = generate_completions(
      config, tokenizer_model, engine, tokenizer_model.encode(config.prompt), state.params["params"], rng
  )
  assert generated_completions == golden_data["generated_completions"]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", type=str, required=True)
  parser.add_argument("--tokenizer-path", type=str, required=True)
  parser.add_argument("--atol", type=float, required=True)
  parser.add_argument("--rtol", type=float, required=True)
  test_args = parser.parse_args(sys.argv[1:])
  test_args.prompt = "Hello world this is a test"
  config = initialize_config(test_args)
  # modify test_args for config inference
  new_per_decide_batch_size = test_args.per_device_batch_size * test_args.num_generations
  test_args.per_device_batch_size = new_per_decide_batch_size
  test_args.ici_tensor_parallelism = 4
  config_inference = initialize_config(test_args)
  main(config, config_inference, test_args)
