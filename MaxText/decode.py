import sys
import jax
import jax.numpy as jnp
import numpy as np

import maxengine
from jetstream.engine import token_utils
from absl.testing import absltest

import os
import pyconfig
import sys



def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  text = config.prompt
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer
  tokens, true_length = token_utils.tokenize_and_pad(text, vocab, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  assert tokens.size <= config.max_prefill_predict_length, "can't take too many tokens"

  prefill_result = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length
  )
  slot=1

  decode_state = engine.init_decode_state()
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  for i in steps:
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer.detokenize(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert output==config.autoregressive_decode_assert, \
    f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"

def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  validate_config(config)
  main(config)
