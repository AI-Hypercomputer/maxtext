# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

export CKPT_13B=gs://runner-maxtext-logs/direct_generate_param_only_llama_2_13b_checkpoint_2024-03-27-04-27/checkpoints/0/items
export DATA_13B=/mnt/disks/persist-data/data/llama13b_openorca_question.json
export idx=$(date +%Y-%m-%d-%H-%M)

python MaxText/inference.py \
  MaxText/configs/base.yml \
  base_output_directory=gs://morgandu-tpu/maxtext-logs/llama2-13b/inference/static \
  run_name=${idx} \
  tokenizer_path=assets/tokenizer.llama2 \
  dataset_path=${DATA_13B} \
  model_name=llama2-13b \
  load_parameters_path=${CKPT_13B} \
  async_checkpointing=false \
  weight_dtype=bfloat16 \
  scan_layers=false \
  per_device_batch_size=4 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  ici_fsdp_parallelism=1 \
  ici_autoregressive_parallelism=-1 \
  enable_profiler=true \
  profiler_steps=128 \
  inference_batching_mode=static \
  inference_profiler_batch_indices="5,15,25" \
  inference_output_path="/tmp/inference_output.json"
"""

from absl import app

import jax
import numpy as np

from typing import Sequence

import os

import maxengine
import pyconfig


import json
from typing import List

from jetstream.engine import token_utils

import max_utils

_LOGGING_INTERVAL = 10

def delete_pytree(p):
  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_map(delete_leaf, p)


def validate_inference_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."
  assert config.inference_batching_mode in ["static", "continuous"], "Inference batching mode supports only static batching and continuous batching."


def load_openorca_dataset(
    dataset_path: str
) -> List[tuple[str]]:
  # Load the dataset.

  with open(dataset_path) as f:
    dataset_json = json.load(f)

  # Tokenize the prompts and completions.
  prompts = dataset_json["prompts"]
  results = dataset_json["results"]

  return prompts, results


def static_batching(config, engine, tokenizer, n, all_padded_tokens, true_lengths, stop_tokens):

  max_output_length = config.max_target_length - config.max_prefill_predict_length

  params = engine.load_params()
  decode_state = engine.init_decode_state()

  num_slots = engine.max_concurrent_decodes
  num_batch = n // num_slots + 1

  profiler_batch_indices = [int(batch) for batch in config.inference_profiler_batch_indices.split(",")]
  profiler_generate_steps = config.profiler_steps

  print(f"{num_slots} slots.")
  print(f"Total {num_batch} batches.")

  slot_generation_complete = np.zeros(num_slots)
  slot_generate_result_tokens = dict()
  slot_generate_results = dict()

  generate_results = []
  all_generate_result_tokens = []

  for batch in range(num_batch):
    if batch in profiler_batch_indices:
      profile_generate_done = False
      max_utils.activate_profiler(config)
    start_i = num_slots * batch
    end_i = min(num_slots * (batch + 1) - 1, n - 1)

    for i in range(start_i, end_i+1):
      padded_tokens = all_padded_tokens[i]
      true_length=true_lengths[i]
      slot = i - start_i

      prefill_result = engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
      )

      decode_state = engine.insert(
        prefix=prefill_result,
        decode_state=decode_state,
        slot=slot
      )
      jax.block_until_ready(decode_state)
      delete_pytree(prefill_result)

    for slot in range(num_slots):

      slot_generation_complete[slot] = 0.
      slot_generate_result_tokens[slot] = []
      slot_generate_results[slot] = None

    for step in range(max_output_length):
      decode_state, result_tokens = engine.generate(
        params, decode_state
      )
      jax.block_until_ready(decode_state)

      for i in range(start_i, end_i+1):
        slot = i - start_i
        slot_data = result_tokens.get_result_at_slot(slot)
        slot_tokens = slot_data.tokens
        slot_lengths = slot_data.lengths

        token_id = slot_tokens[slot, 0].item()
        if slot_lengths > max_output_length or token_id in stop_tokens:
          slot_generation_complete[slot] = 1

        if slot_generation_complete[slot]==0:
          slot_generate_result_tokens[slot].append(token_id)
        else:
          slot_generate_results[slot] = tokenizer.detokenize(slot_generate_result_tokens[slot])

      if batch in profiler_batch_indices and (step + 1) == profiler_generate_steps:
        profile_generate_done = True
        max_utils.deactivate_profiler(config)

      if np.sum(slot_generation_complete[:end_i+1-start_i]) == 1:
        if batch in profiler_batch_indices and profile_generate_done is False:
          profile_generate_done = True
          max_utils.deactivate_profiler(config)

        if batch % _LOGGING_INTERVAL == 0:
          print(f"All generations for batch {batch} are completed at step {step}.")

        break

    for i in range(start_i, end_i+1):
      slot = i - start_i
      all_generate_result_tokens.append(slot_generate_result_tokens[slot])
      generate_results.append(slot_generate_results[slot])

    if batch % _LOGGING_INTERVAL == 0:
      print(f"Finished batch {batch} over {num_batch} batches.")

  return generate_results, all_generate_result_tokens


def inference(config):

  prompts, results = load_openorca_dataset(config.dataset_path)
  n = len(prompts)

  engine = maxengine.MaxEngine(config)
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  stop_tokens = [vocab.eos_id, vocab.pad_id]
  print(f"stop_tokens: {stop_tokens}")

  all_padded_tokens = []
  true_lengths = []
  for prompt in prompts:
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        prompt,
        vocab,
        is_bos=True,
        prefill_lengths=[config.max_prefill_predict_length]
    )
    all_padded_tokens.append(padded_tokens)
    true_lengths.append(true_length)

  if config.inference_batching_mode == "static":
    generate_results, all_generate_result_tokens = static_batching(
      config,
      engine,
      tokenizer,
      n,
      all_padded_tokens,
      true_lengths,
      stop_tokens
    )
  elif config.inference_batching_mode == "continuous":
    raise NotImplementedError("Continuous batching is not implemented yet.")

  inference_output_json = dict()
  inference_output_json["prompts"] = prompts
  inference_output_json["original_results"] = results
  inference_output_json["generate_results"] = generate_results
  inference_output_json["all_generate_result_tokens"] = all_generate_result_tokens

  if config.inference_output_path:
    with open(config.inference_output_path, "w", encoding="utf-8") as f:
      json.dump(inference_output_json, f)

  return


def main(argv: Sequence[str]) -> None:
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_inference_config(config)
  inference(config)


if __name__ == "__main__":
  app.run(main)
