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
import sys
sys.path.append("/home/rwitten/disaggregation/")
import jax
import jax.numpy as jnp
import numpy as np

import myengine
from inference_engine import token_utils
from absl.testing import absltest

import os
import pyconfig
import sys



def main(config):
  engine = myengine.TestEngine(config)
  params = engine.load_params()

  text = config.prompt
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  tokens, true_length = token_utils.tokenize_and_pad(text, vocab, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])

  prefill_result = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length
  )
  slot=1

  decode_state = engine.init_decode_state()
  decode_state = engine.insert(
      prefix=prefill_result, decode_state=decode_state, slot=slot
  )

  #breakpoint()

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  for i in steps:
    decode_state, sampled_tokens = engine.generate(
      params=params, decode_state=decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot)[0].item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer.detokenize(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert output==config.autoregressive_decode_assert, \
    f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"
  

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  main(config)