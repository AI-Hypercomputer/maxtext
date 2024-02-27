"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""


import functools
from typing import Sequence
import datetime
import flax
import orbax
import os
from absl import app
import numpy as np
import pyconfig
import max_utils
import inference_utils
from input_pipeline.input_pipeline_interface import create_data_iterator_with_tokenizer
from layers import models, quantizations
import common_types
import jax
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental.compilation_cache import compilation_cache as cc
import max_logging
from cuda_api import cudaProfilerStart, cudaProfilerStop
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

Transformer = models.Transformer


def replicate_globally(np_array, mesh):
  arrays = [jax.device_put(np_array, dev) for dev in mesh.local_devices]
  sharding = jax.sharding.NamedSharding(mesh, P())
  return jax.make_array_from_single_device_arrays(np_array.shape, sharding, arrays)

def match_input_and_output_stream(prompt, outputs, tokenizer):
  for i in range(len(prompt)):
    prompt_mini = prompt[0:i+1]
    prompt_mini_arr = np.array(prompt_mini, dtype=np.int32)
    prompt_mini_str = decode_tokens(prompt_mini_arr, tokenizer)
    output_mini = outputs[i:i+1]
    output_mini_arr = np.array(output_mini, dtype=np.int32)
    output_mini_str = decode_tokens(output_mini_arr, tokenizer)
    print(f"{prompt_mini_str} -> {output_mini_str}")

def decode_tokens(toks, tokenizer):
  return tokenizer.detokenize(toks).numpy().decode("utf-8"), len(toks)

def default_prompts(config):
  return [config.prompt] * int(config.per_device_batch_size * jax.device_count())


def encode_strings(strs, max_len, tokenizer, mesh):
  """Pack prefill prompts into Jax.Array. The prompts are `right-aligned`, i.e. padded with zeros and all ending on the same
     index."""
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  positions = np.zeros((len(strs), max_len), np.int32)
  segment_ids = np.zeros((len(strs), max_len), np.int32)

  for i, s in enumerate(strs):
    toks = tokenizer.tokenize(s).numpy()
    assert toks.shape[0] <= max_len, f"We aren't able to tokenize input {i}, it is too long"
    prompt = toks
    start_index = max_len - prompt.shape[0]
    tokenized_batch[i, start_index:] = prompt
    padded_start_index = start_index
    segment_ids[i, padded_start_index:] = common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    positions[i, padded_start_index:] = np.arange(len(prompt))
  return replicate_globally(tokenized_batch, mesh), \
      replicate_globally(positions, mesh), \
      replicate_globally(segment_ids, mesh)


def prefill_predict_step(inputs, input_positions, decoder_segment_ids,
                         model_vars, rngkey, model=None, init_aqt=False):
  """Prefill KV Cache and output logits"""
  flat_logits, new_vars = model.apply(
    model_vars,
    inputs,
    input_positions,
    decoder_segment_ids=decoder_segment_ids,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_PREFILL,
    rngs={'params': rngkey},
    mutable=True
  )
  prefill_cache = new_vars['cache']
  aqt_vars = new_vars['aqt'] if init_aqt else None
  return flat_logits, prefill_cache, aqt_vars


def ar_predict_single_token(previous_logits, token_position, kv_cache, model_vars, rngkey, model, config):
  """Predict one token, return new cache"""

  new_token = inference_utils.sampling(previous_logits, rngkey, config.decode_sampling_strategy,\
                                       topk=config.decode_sampling_top_k, nucleus_topp=config.decode_sampling_nucleus_p,
                                       temperature=config.decode_sampling_temperature)

  flat_logits, new_vars = model.apply(
    model_vars | {'cache': kv_cache},
    new_token,
    token_position,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
    rngs={'params': rngkey},
    mutable=['cache'])
  new_flat_cache = new_vars["cache"]
  return token_position+1, new_flat_cache, flat_logits, new_token

def compute_prefill(config, model, model_vars, prompts, rng, sp_tokenizer, mesh,
                    kv_cache_mesh_shardings, replicated_sharding, init_aqt=False):
  """Compute the necessary prefill state."""

  # Encode the demo prompt -- to measure performance we encode it multiple times.
  tokenized_prompts, prompt_decoder_positions, prompt_decoder_segment_ids  = encode_strings(
    prompts,config.max_prefill_predict_length, sp_tokenizer, mesh)

  partial_prefill_predict_step = functools.partial(prefill_predict_step, model=model, init_aqt=init_aqt)
  p_prefill_predict_step = jax.jit(
      partial_prefill_predict_step,
      in_shardings=(replicated_sharding, replicated_sharding, replicated_sharding, None, None),
      out_shardings=(replicated_sharding, kv_cache_mesh_shardings, None)
  )

  prefill_output, prefill_cache, aqt_vars = p_prefill_predict_step(
    tokenized_prompts, prompt_decoder_positions, prompt_decoder_segment_ids, model_vars, rng)

  with jax.spmd_mode('allow_all'):
    updated_prompt_decoder_positions = prompt_decoder_positions[:, -1:] + 1

  return prefill_cache, prefill_output[:, -1:], updated_prompt_decoder_positions, aqt_vars

def decode_ar_one_step(config, model, model_vars, new_cache, pos, logits, rng,
                      kv_cache_mesh_shardings, replicated_sharding):
  """Compute the necessary prefill state."""
  partial_ar_predict_step = functools.partial(ar_predict_single_token, model=model, config=config)
  partial_ar_predict_step.__name__ = "partial_ar_predict_step"
  p_ar_predict_step = jax.jit(
      partial_ar_predict_step,
      in_shardings=(replicated_sharding, replicated_sharding, kv_cache_mesh_shardings, None, None),
      out_shardings=(replicated_sharding, kv_cache_mesh_shardings, replicated_sharding, replicated_sharding),
      donate_argnums=2
  )
  new_pos, updated_cache, last_logit, selected_id = p_ar_predict_step(
    logits, pos, new_cache, model_vars, rng)
  return new_pos, updated_cache, last_logit, selected_id

def prefill_or_load(config, model, model_vars, prompts, rng, sp_tokenizer, mesh,
                    kv_cache_mesh_shardings, replicated_sharding):
  """We either load the necessary prefill state or generate it.  """
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  if config.load_from_prefill_dir:
    kv_cache_restore_args = jax.tree_map(
      lambda sharding: orbax.checkpoint.type_handlers.ArrayRestoreArgs(sharding=sharding),
      kv_cache_mesh_shardings
      )
    next_token_restore_args = orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=P(None))
    pos_restore_args = orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=P(None))
    restore_args = {"cache": kv_cache_restore_args, "next_token": next_token_restore_args, "pos": pos_restore_args}
    blob = orbax_checkpointer.restore(config.prefill_cache_dir, restore_args=restore_args)
    blob["cache"] = jax.tree_map(lambda x : flax.linen.spmd.LogicallyPartitioned(x, mesh.axis_names), blob["cache"])
    max_logging.log(f"Restored prefill cache from {config.prefill_cache_dir}")
    return blob["cache"], blob["last_logit"], blob["pos"]
  else:
    cache, last_logit, pos, _ = compute_prefill(config, model, model_vars,
                                                prompts, rng, sp_tokenizer, mesh,
                                                kv_cache_mesh_shardings,
                                                replicated_sharding)
    max_logging.log(f"Computed prefill cache {config.prefill_cache_dir}")

    if config.prefill_cache_dir != "":
      blob = {"cache":cache, "last_logit":last_logit, "pos":pos}
      orbax_checkpointer.save(config.prefill_cache_dir, max_utils.unbox_logicallypartioned(blob))
      max_logging.log(f"Wrote prefill cache to {config.prefill_cache_dir}")
    return cache, last_logit, pos

def aqt_serving_conversion(model, mesh, config, model_vars, rng, sp_tokenizer):
  """Compute the necessary prefill state"""

  # Use configured prompt to generate default prompts
  prompts = default_prompts(config)

  # Compute shardings
  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None))
  kv_cache_annotations = max_utils.get_kv_cache_annotations(model, config, rng, mesh)
  kv_cache_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), kv_cache_annotations)

  model.quant.quant_mode = quantizations.get_quant_mode('convert')
  # Run prefill step
  _, _, _, aqt_vars = compute_prefill(
    config, model, model_vars, prompts, rng, sp_tokenizer, mesh,
    kv_cache_mesh_shardings, replicated_sharding, init_aqt=True)

  # Update aqt state
  model_vars['aqt'] = aqt_vars

  # Set quant mode for model to serve
  model.quant.quant_mode = quantizations.get_quant_mode('serve')
  return model, model_vars

def init_decode(config):
  """Initialize decode model, vars and tokennizer."""
  rng = random.PRNGKey(0)
  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  # Model definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh = mesh, quant=quant)
  # Tokenizer
  _, _, sp_tokenizer = create_data_iterator_with_tokenizer(config, mesh, add_bos = True, add_eos=False)
  # Load model vars
  model_vars = max_utils.load_decode_model_vars(model, config, rng, mesh)
  num_params, bytes_params, bytes_per_param = max_utils.summarize_size_from_pytree(model_vars['params'])
  max_logging.log(f"Number of model params loaded ={num_params/10**9:.3f} billion, memory usage={bytes_params/2**30:.3f}GB, "
                  f"bytes per param={bytes_per_param:.3f}")
  # Update aqt state
  if model.quant:
    model, model_vars = aqt_serving_conversion(
      model, mesh, config, model_vars, rng, sp_tokenizer)

  return model, model_vars, sp_tokenizer, rng


def decode_loop(model, model_vars, sp_tokenizer, rng, prompts):
  """Decoding loop for the Transformer model."""
  mesh = model.mesh
  config = model.config

  num_params, bytes_params, bytes_per_param = max_utils.summarize_size_from_pytree(model_vars['params'])
  max_logging.log(f"Number of model params={num_params/10**9:.3f} billion, memory usage={bytes_params/2**30:.3f}GB, "
                  f"bytes per param={bytes_per_param:.3f}")

  bytes_aqt_params = 0
  if model.quant:
    num_aqt_params, bytes_aqt_params, bytes_per_aqt_param = max_utils.summarize_size_from_pytree(model_vars['aqt'])
    max_logging.log(f"Number of aqt params={num_aqt_params/10**9:.3f} billion, memory usage={bytes_aqt_params/2**30:.3f}GB, "
                    f"bytes per aqt param={bytes_per_aqt_param:.3f}")

  # Compute shardings
  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None))
  kv_cache_annotations = max_utils.get_kv_cache_annotations(model, config, rng, mesh)
  kv_cache_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), kv_cache_annotations)

  prefill_cache, next_logit, new_position = prefill_or_load(
    config, model, model_vars, prompts, rng, sp_tokenizer, mesh,
    kv_cache_mesh_shardings, replicated_sharding)
  num_cache, bytes_cache, bytes_per_cache = max_utils.summarize_size_from_pytree(prefill_cache)
  max_logging.log(f"Number of cache entries={num_cache/10**9:.3f} billion, memory usage={bytes_cache/2**30:.3f}GB, "
                  f"bytes per cache={bytes_per_cache:.3f}")

  total_memory_GB = (bytes_params + bytes_aqt_params + bytes_cache)/2**30
  max_logging.log(f"Total memory for cache and params (and any quantization state) {total_memory_GB:.3f} GB")

  new_cache = prefill_cache
  first_profiling_step = config.max_prefill_predict_length + config.skip_first_n_steps_for_profiler
  last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1,
                                first_profiling_step, config.max_target_length - 1)

  outputs = []
  max_logging.log("Generate first predicted token")
  new_position, new_cache, next_logit, selected_id = decode_ar_one_step(
    config, model, model_vars, new_cache, new_position, next_logit, rng,
    kv_cache_mesh_shardings, replicated_sharding)
  outputs.append(selected_id)
  jax.block_until_ready(new_cache)

  starttime = datetime.datetime.now()
  steps = range(config.max_prefill_predict_length + 1, config.max_target_length)
  rngs = [random.PRNGKey(i) for i in range(config.max_target_length)]
  max_logging.log(f"Generate remaining {len(steps)} predicted tokens")

  for step in steps:
    if step == first_profiling_step:
      max_utils.activate_profiler(config)
    
    if step == 70 and jax.process_index() == 0:
      cudaProfilerStart()
      print("====================step 70=================")
    if step == 75 and jax.process_index() == 0:
      cudaProfilerStop()
      print("=====================step 75================")

    new_position, new_cache, next_logit, selected_id = decode_ar_one_step(
    config, model, model_vars, new_cache, new_position, next_logit, rngs[step],
    kv_cache_mesh_shardings, replicated_sharding)
    outputs.append(selected_id)
    if step == last_profiling_step:
      jax.block_until_ready(outputs)
      max_utils.deactivate_profiler(config)
  endtime = datetime.datetime.now()

  new_text, _ = decode_tokens([int(x[0, 0]) for x in outputs], sp_tokenizer)
  max_logging.log(f"Completion: `{config.prompt}` -> `{new_text}`")
  if config.autoregressive_decode_assert != "":
    assert new_text==config.autoregressive_decode_assert, \
    f"generated text mismatch {new_text=} {config.autoregressive_decode_assert=}"

  num_steps = len(steps)
  elapsed_time = (endtime-starttime).total_seconds() * 1000
  seqs = config.per_device_batch_size * jax.device_count()

  per_step_time = elapsed_time/num_steps
  memory_bandwidth_per_device_GB_per_sec = total_memory_GB/(elapsed_time/num_steps)/jax.device_count()
  max_logging.log(f"Did {num_steps} steps in {elapsed_time:.3f} milliseconds for {seqs} sequences"
                  f" with a total memory footprint of {total_memory_GB:.3f} GB")
  max_logging.log(f"Therefore, a per-generate time of {per_step_time:.4f} seconds, a throughput of {seqs/per_step_time:.3f} "
                  f"tok/s and {memory_bandwidth_per_device_GB_per_sec:.3f} GB/s/device")

def validate_config(config):
  assert config.load_full_state_path == "", "Decode doesn't operate on full states! Convert to parameter checkpoint first."\
                                            "Using generate_param_only_checkpoint."

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  config = pyconfig.config
  validate_config(config)
  model, model_vars, tokenizer, rng = init_decode(config)
  prompts = default_prompts(config)
  decode_loop(model, model_vars, tokenizer, rng, prompts)

if __name__ == "__main__":
  app.run(main)
