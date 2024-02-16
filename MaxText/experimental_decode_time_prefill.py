"""Tests for a mock version of the engine API, used in integration tests elsewhere.

What should we expect?

Prefill: Doubles the sequence by multiplying it with a weight [2].
Insert: Writes this sequence into a cache row
Generate step: Return sum(prefill_cache) + sum(generate_cache)/weight

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] /2  = 399
266 + [266, 399] /2 = 598
I.e. ['Ċ', 'Ə', 'ɖ'] when converted back with chr()
"""
import datetime
import sys
sys.path.append("/home/rwitten/disaggregation/")
import jax
import jax.numpy as jnp
import numpy as np

import max_utils

import myengine
from inference_engine import token_utils
from absl.testing import absltest

import os
import pyconfig
import sys

import max_logging
import maxtext_utils

def main(config):
  engine = myengine.TestEngine(config)
  params = engine.load_params()

  text = config.prompt
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  tokens, true_length = token_utils.tokenize_and_pad(text, vocab, is_bos=True)

  decode_state = engine.init_decode_state()
  slot=0

  prefill_result = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length
  )
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )

  jax.block_until_ready(decode_state)
  max_utils.activate_profiler(config)
  start = datetime.datetime.now()

  for i in range(config.steps):
    prefill_result = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = engine.insert(
        prefill_result, decode_state, slot=int(slot)
    )
    slot = i % (jax.device_count() * config.per_device_batch_size)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  max_utils.deactivate_profiler(config)

  prefill_average_ms = 1000 * (end-start).total_seconds()/config.steps
  num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
  max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")
  total_tflops = maxtext_utils.calculate_tflops_prefill(num_model_parameters, tokens.size, config)

  print(f"Prefill took on average {prefill_average_ms:.2f} milliseconds")
  print(f"TFLOP/s/device achieved: {total_tflops/jax.device_count()/(prefill_average_ms/1000):.2f}")



  
  

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  main(config)