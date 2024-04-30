"""CLI Utility for Running Open LLM Evaluation Test"""

from typing import List, Optional, Tuple, Union
import random
import numpy as np

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval import tasks, evaluator

import max_logging
import maxengine
from jetstream.engine import token_utils
from transformers import AutoTokenizer

import jax
import jax.numpy as jnp
import os
import pyconfig
import sys

def hf_tokenize_and_pad(
  s: str,
  tokenizer,
  #is_bos: bool = True,
  prefill_lengths: Optional[List[int]] = None,
  max_prefill_length: Optional[int] = None,
  jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
  """Tokenize and pads a string using HF format tokenizer

  Args:
    s: String to tokenize.
    tokenizer: Tokenizer loaded with AutoTokenizer.from_pretrain()
    is_bos: Whether or not this is the beginning of a sequence. Default to yes
      as prefill is typically used when beginning sequences.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    max_prefill_length: Maximum bucket to use.
    jax_padding: convert to JAX padded tokens if True.

  Returns:
    tokens: Tokenized into integers.
    true_length: Actual length of the non-padded sequence.
  """
  if prefill_lengths is None:
    prefill_lengths = [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]
  if max_prefill_length is not None:
    prefill_lengths = prefill_lengths[
        : prefill_lengths.index(max_prefill_length)
    ] + [
        max_prefill_length,
    ]

  assert tokenizer.pad_token_id == 0, "Further logic required if pad_id not 0."
  tokens = np.array(tokenizer(s)['input_ids'])
   #padded_tokens = np.array(tokenizer(s, padding='max_length', max_length=padded_length, truncation=True))  # [Length]

  # Add a beginning of sequence token if this is the beginning.
  true_length = tokens.shape[-1]
  padded_length = token_utils.take_nearest_length(prefill_lengths, true_length)
  padding = padded_length - true_length
  #import pdb; pdb.set_trace()

  if padding < 0:
    max_logging.log("Provided sequence longer than available.")
    # Take the last N tokens if we have too many.
    padded_tokens = tokens[-padded_length:]
  else:
    padded_tokens = np.pad(tokens, (0, padding))
  if jax_padding:
    padded_tokens = jnp.array(padded_tokens)

  return padded_tokens, true_length


class MaxTextLM(LM):
    def __init__(self, config) -> None:
        super().__init__()

        self.lm_config = config
        self.engine = maxengine.MaxEngine(config)
        self.params = self.engine.load_params()
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, add_bos_token=True)

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for req in tqdm(requests, disable=disable_tqdm):
            ctx, gen_config = req.args
            # gen_config: {'until': ['\n\n'], 'do_sample': False}
            if not gen_config['do_sample']:
                assert self.lm_config.decode_sampling_strategy == "greedy"
            
            tokens, true_length = hf_tokenize_and_pad(
                ctx, self.tokenizer, prefill_lengths=[self.lm_config.max_prefill_predict_length]
            )
            assert tokens.size <= self.lm_config.max_prefill_predict_length, "can't take too many tokens"
            assert self.lm_config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
            prefill_result = self.engine.prefill(params=self.params, padded_tokens=tokens, true_length=true_length)
            slot = 0

            decode_state = self.engine.init_decode_state()
            decode_state = self.engine.insert(prefill_result, decode_state, slot=slot)

            steps = range(self.lm_config.max_prefill_predict_length, self.lm_config.max_target_length)
            sampled_tokens_list = []
            for _ in steps:
                decode_state, sampled_tokens = self.engine.generate(self.params, decode_state)
                sampled_tokens_list.append(sampled_tokens)

            results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
            output = self.tokenizer.detokenize(results)
            res.append(ctx + " " + output)

        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for req in tqdm(requests, disable=disable_tqdm):
            ctx, cont = req.args
            
            tokens, true_length = hf_tokenize_and_pad(
                ctx, self.tokenizer, prefill_lengths=[self.lm_config.max_prefill_predict_length]
            )
            prefill_result = self.engine.prefill(params=self.params, padded_tokens=tokens, true_length=true_length)
            assert tokens.size <= self.lm_config.max_prefill_predict_length, "can't take too many tokens"
            assert self.lm_config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
            
            slot = 0

            decode_state = self.engine.init_decode_state()
            decode_state = self.engine.insert(prefill_result, decode_state, slot=slot)

            target_tokens = np.array(self.tokenizer(cont)['input_ids'])
            steps = range(len(target_tokens)-1)

            log_probs = jax.nn.log_softmax(decode_state["logits"][0][0])
            index = jnp.argmax(log_probs).item()  # greedy for now
            ll = log_probs[index].item()
            #log_probs_list = [ll]

            for _ in steps:
                decode_state, sampled_tokens = self.engine.generate(self.params, decode_state)
                
                log_probs = jax.nn.log_softmax(decode_state["logits"][0][0])
                index = jnp.argmax(log_probs).item()  # greedy for now
                ll += log_probs[index].item()
                #log_probs_list.append(log_probs[index].item())

            res.append((ll, self.lm_config.decode_sampling_strategy == "greedy"))

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError


def run_eval_harness(config, bootstrap_iters=2):
    lm = MaxTextLM(config)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],  # arc_challenge hellaswag winogrande mmlu truthfulqa_gen
        num_fewshot=10,
        limit=5,  # Limit the number of examples per task 
        bootstrap_iters=bootstrap_iters,  # default: 10000
        log_samples=False,
    )
    return results

    # ARC challenge(25-shot)  38.14
    # HellaSwag(10-shot)      67.27
    # Winogrande(5-shot)      62.43
    # Truthfulness(0-shot)    35.70
    # MMLU(5-shot)            23.67


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  run_eval_harness(cfg)
