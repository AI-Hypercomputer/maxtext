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
  tokens, true_length = token_utils.tokenize_and_pad(text, vocab, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])

  decode_state = engine.init_decode_state()
  slot=0

  ### just to make sure we don't OOM TODO(FIX)
  prefill_result = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length
  )
  decode_state = engine.insert(
      prefill_result, decode_state, slot=slot
  )
  ### making sure we don't OOM

  #WARM IT UP START
  decode_state, sampled_tokens = engine.generate(
      params, decode_state
  )
  decode_state, sampled_tokens = engine.generate(
      params, decode_state
  )
  #WARM IT UP END

  num_params, bytes_params, bytes_per_param = max_utils.summarize_size_from_pytree(params)
  max_logging.log(f"Number of params={num_params/10**9:.3f} billion, memory usage={bytes_params/2**30:.3f}GB, "
                  f"bytes per param={bytes_per_param:.3f}")
  num_cache, bytes_cache, bytes_per_cache = max_utils.summarize_size_from_pytree(decode_state['cache'])
  max_logging.log(f"Number of cache entries={num_cache/10**9:.3f} billion, memory usage={bytes_cache/2**30:.3f}GB, "
                  f"bytes per cache={bytes_per_cache:.3f}")

  jax.block_until_ready(decode_state)
  max_utils.activate_profiler(config)
  start = datetime.datetime.now()

  for i in range(config.steps):
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  max_utils.deactivate_profiler(config)

  seconds_per_step = (end-start).total_seconds() / config.steps
  global_batch_size = jax.device_count() * config.per_device_batch_size
  total_tok_per_sec = (jax.device_count() * config.per_device_batch_size) / seconds_per_step

  GB_per_step_per_device = (bytes_params+bytes_cache)/(2**30 * jax.device_count())
  print(f"Each tick took {seconds_per_step*1000:.2f}ms, with a global_batch_size {global_batch_size}"
        f" so {total_tok_per_sec:.2f} tokens/second" )
  
  print(f"Per device memory bandwidth {GB_per_step_per_device/seconds_per_step:.2f} GB/s")

  
AGGRESSIVE_VMEM_PREFETCHING = {
    'xla_tpu_enable_all_experimental_scheduler_features': 'false',
    'xla_jf_rematerialization_percent_shared_memory_limit': 145,
    'xla_tpu_async_copy_bandwidth_scaling_factor': 0.684022,
    'xla_tpu_copy_elision_analysis_allowance': 46489,
    'xla_tpu_copy_fusion_pad_unpad_ratio': 5.736092,
    'xla_tpu_copy_insertion_use_region_analysis_limit': 2749,
    'xla_tpu_enable_aggressive_loop_fusion_layout_opt': 'false',
    'xla_tpu_enable_experimental_fusion_cost_model': 'true',
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_licm_size_inflation_ratio': 0.229577,
    'xla_tpu_nd_short_transfer_max_chunks': 3473,
    'xla_tpu_pack_vloads': 'true',
    'xla_tpu_prefetch_interval_picker_size_override': 198848,
    'xla_tpu_scoped_vmem_limit_kib': 12801,
    'xla_tpu_use_repeated_instance_for_preferred_prefetch_time': 'true',
    'xla_vf_vmem_max_outstanding_evictions': 328,
    'xla_vf_vmem_max_outstanding_prefetches': 96,
    'xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio': 2.948259,
    'xla_vf_vmem_max_repacks': 1,
    'xla_vf_vmem_max_retries': 8,
    'xla_vf_vmem_min_overlap_to_async_copy_ratio': 0.57965,
    'xla_vf_vmem_preferred_overlap_to_async_copy_ratio': 3.316116,
    'xla_tpu_vmem_use_telamalloc': 'true',
    'xla_tpu_enable_multi_level_nested_loop_fusion': 'true',
}
args = ""
for k in AGGRESSIVE_VMEM_PREFETCHING:
  args += f" --{k}={AGGRESSIVE_VMEM_PREFETCHING[k]} "

if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  #os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + args

  #os.environ["TPU_STDERR_LOG_LEVEL"] = "0"
  #os.environ["TPU_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  main(config)