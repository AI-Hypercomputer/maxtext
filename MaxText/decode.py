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

import jax

import max_utils
import maxengine

import os
import pyconfig

from typing import Sequence
from absl import app
import common_types
Array = common_types.Array
from flax import struct
from typing import Any, List, Optional, Tuple, Callable, Union
import numpy as np
import math
import jax.numpy as jnp


# TODO: move this to Jetstream after all features are done for chunked prefill
# this is in decode to enable testing and debugging

@struct.dataclass
class ChunkMetadata:
  processed_chunks: Any
  next_pos: Union[jax.Array, np.ndarray]
  true_length: int
  chunk_size: int
  chunk_padded_tokens: Union[jax.Array, np.ndarray]
  processed: bool
  is_first_chunk: bool
  is_last_chunk:bool

def create_chunked_metadata(tokens, true_length, chunk_size):
  start = 0
  chunk_metadata_list = []
  num_chunks =int(math.ceil(len(tokens)/chunk_size))

  for chunk_num in range(num_chunks):
    true_length_of_chunk = chunk_size
    start = int(chunk_num*chunk_size)
    end = int(min(chunk_num+1)*chunk_size, true_length)
    true_length_of_chunk = end - start
    next_pos = jnp.full((1, 1), start + true_length_of_chunk, dtype=jnp.int32)
    is_first_chunk = chunk_num == 0
    is_last_chunk = chunk_num == num_chunks - 1
    
    chunk_metadata_list.append(ChunkMetadata(processed_chunks=None, 
                                             next_pos=next_pos, true_length=true_length_of_chunk,
                                              chunk_size=chunk_size, 
                                              chunk_padded_tokens=tokens[start:max(end, true_length_of_chunk)],
                                              processed=False,
                                              is_first_chunk=is_first_chunk,
                                              is_last_chunk=is_last_chunk, ))
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
  if config.use_chunked_prefill:
    text = config.prompt_long

  
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"

  # Split RNG before calling prefill
  rng, rng_prefill = jax.random.split(rng)

  if not config.use_chunked_prefill:
    prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill)
  else:
    prefill_result, first_token = chunked_prefill(params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill)
  
  slot = 0

  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)
  decode_state = engine.insert(prefill_result, decode_state, slot=slot)

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  sampled_tokens_list.append(first_token)
  for _ in steps:
    rng, rng_generate = jax.random.split(rng)
    decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer_model.decode(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert (
        output == config.autoregressive_decode_assert
    ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"


def chunked_prefill(engine, config,params, padded_tokens, true_length, rng):
  """
  function breaks long padded tokens into chunks and calls prefill for each chunk
  returns the prefill result and first token after processing last chunk
  Note - this is not working yet, the function only shows high level calls.
  """
  chunked_metadata_list = create_chunked_metadata(padded_tokens, true_length, config.chunk_size)
  prefill_result = None
  first_token = None
  #TODO: move this logic to JetStream orchestrator once testing is complete
  for chunk_metadata in chunked_metadata_list:
      rng, rng_prefill = jax.random.split(rng)
      if chunk_metadata.is_first_chunk:
        prefill_result, first_token = engine.prefill(processed_prefix=prefill_result,
                                                    params=params, 
                                                    padded_tokens=chunk_metadata.chunk_padded, 
                                                    true_length=chunk_metadata.true_length, 
                                                    rng=rng_prefill,)
      else:
        prefill_result, first_token = engine.prefill(processed_prefix=prefill_result, 
                                                    params=params | {"cache": prefill_result["cache"]}, 
                                                    padded_tokens=chunk_metadata.chunk_padded, 
                                                    true_length=chunk_metadata.true_length, 
                                                    rng=rng_prefill,)
      chunk_metadata.is_processed = True
  return prefill_result, first_token, chunked_metadata_list
      


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  app.run(main)
