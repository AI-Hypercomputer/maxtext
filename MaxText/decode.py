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

"""CLI utility for running inference on a single stream"""

from collections import defaultdict
import jax

import max_utils
import maxengine

import os
import pyconfig

from typing import Sequence
from absl import app
from flax import struct
import common_types
Array = common_types.Array
import jax.numpy as jnp

@struct.dataclass
class ChunkMetadata:
  tokens_entire_sequence: Array
  true_length: int
  true_length_chunk: int
  chunk_padded: Array
  processed: bool
  chunk_seq_start_index: int


def create_chunked_metadata(tokens, true_length, chunk_size):
  start = 0
  chunk_metadata_list = []
  
  while start < len(tokens):
    end = min(start + chunk_size, true_length)
    cur_chunk_tokens = tokens[start:end]

    chunk_metadata_list.append(ChunkMetadata(tokens_entire_sequence=tokens, 
                                             true_length=true_length, 
                                             true_length_chunk=chunk_size, 
                                             chunk_padded=cur_chunk_tokens, 
                                             processed=False, 
                                             chunk_seq_start_index=start))
    
    start = start + chunk_size
  return chunk_metadata_list


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_config(config)
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  chunk_size = config.chunk_size
  tokens = tokens[:config.max_prefill_predict_length]
  true_length = config.max_prefill_predict_length
  chunked_metadata_list = create_chunked_metadata(tokens, true_length, chunk_size)
  assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"

  # Split RNG before calling prefill
  rng, rng_prefill = jax.random.split(rng)
  slot = 0
  rng, rng_init_decode = jax.random.split(rng)

  prefill_result = None
  prefill_results_dict = defaultdict(list)
  for i,chunk_metadata in enumerate(chunked_metadata_list):
      position_og = jnp.arange(config.max_prefill_predict_length)
      decode_state = engine.init_decode_state(rng_init_decode)
      postion_mask_1 = jnp.where(position_og >= i*chunk_size, position_og, 0)
      postion_mask = jnp.where(position_og < (i+1)*chunk_size, position_og, 0)
      print(postion_mask)
      prefill_result, first_token = engine.prefill(existing_prefix=prefill_result, 
                                                   params=params, 
                                                   padded_tokens=chunk_metadata.chunk_padded, 
                                                   true_length=true_length, 
                                                   rng=rng_prefill, 
                                                   position_mask_cur=postion_mask)
      # for k, v in prefill_result.items():
      #   prefill_results_dict[k].append(v)
      
      # import pdb
      # pdb.set_trace()

  # rng, rng_init_decode = jax.random.split(rng)
  # decode_state = engine.init_decode_state(rng_init_decode)
  # decode_state = engine.insert(prefill_result, decode_state, slot=slot)

  # steps = range(config.max_prefill_predict_length, config.max_target_length)
  # sampled_tokens_list = []
  # sampled_tokens_list.append(first_token)
  # for _ in steps:
  #   rng, rng_generate = jax.random.split(rng)
  #   decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)
  #   sampled_tokens_list.append(sampled_tokens)

  # results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  # output = tokenizer_model.decode(results)
  # print(f"Input `{text}` -> `{output}`")

  # if config.autoregressive_decode_assert != "":
  #   assert (
  #       output == config.autoregressive_decode_assert
  #   ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  app.run(main)
