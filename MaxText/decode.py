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

import flax
import orbax

import os
from absl import app
import numpy as np

import pyconfig
import max_utils
from input_pipeline import create_data_iterator_with_tokenizer
from layers import models

import checkpointing
import common_types

import jax
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

import max_logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

Transformer = models.Transformer

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
  return jax.device_put(tokenized_batch, jax.sharding.NamedSharding(mesh, P())),\
         jax.device_put(positions, jax.sharding.NamedSharding(mesh, P())),\
         jax.device_put(segment_ids, jax.sharding.NamedSharding(mesh, P()))

def prefill_predict_step(inputs, input_positions, decoder_segment_ids,
                 state,
                 rngkey,
                 model=None):
  """Prefill KV Cache and output logits"""
  flat_logits, new_vars = model.apply(
    {
        "params": state.params
    },
    inputs,
    input_positions,
    decoder_segment_ids=decoder_segment_ids,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_PREFILL,
    rngs={'aqt': rngkey},
    mutable=["cache"]
  )

  return flat_logits, new_vars['cache']

def ar_predict_single_token(token_input, token_position, kv_cache, state, rngkey, model):
  """Predict one token, return new cache"""
  flat_logits, new_vars = model.apply(
    {
        "params": state.params,
        "cache": kv_cache
    },
    token_input,
    token_position,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
    rngs={'aqt': rngkey},
    mutable=["cache"])
  new_flat_cache = new_vars["cache"]
  return token_position+1, new_flat_cache, jax.numpy.argmax(flat_logits, axis=2)

def compute_prefill(config, model, state, rng, sp_tokenizer, mesh, state_mesh_shardings,
                    kv_cache_mesh_shardings):
  """Compute the necessary prefill state."""

  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None))
  tokenized_prompt = [config.prompt] * int(config.per_device_batch_size * jax.device_count())

  # Encode the demo prompt -- to measure performance we encode it multiple times.
  tokenized_prompts, prompt_decoder_positions, prompt_decoder_segment_ids  = encode_strings(tokenized_prompt,\
      config.max_prefill_predict_length, sp_tokenizer, mesh)

  partial_prefill_predict_step = functools.partial(prefill_predict_step, model=model)
  p_prefill_predict_step = jax.jit(
      partial_prefill_predict_step,
      in_shardings=(replicated_sharding, replicated_sharding, replicated_sharding, state_mesh_shardings, None),
      out_shardings=(replicated_sharding, kv_cache_mesh_shardings)
  )

  prefill_output, prefill_cache = p_prefill_predict_step(tokenized_prompts, prompt_decoder_positions,\
                                                        prompt_decoder_segment_ids, state, rng)
  indices = jax.numpy.argmax(prefill_output, axis=2)
  match_input_and_output_stream(tokenized_prompts[0, :], np.array(indices[0,:]), sp_tokenizer)

  last_index = indices[:, -1:]
  return prefill_cache, last_index, prompt_decoder_positions[:, -1:]+1

def prefill_or_load(config, model, state, rng, sp_tokenizer, mesh, state_mesh_shardings,
                    kv_cache_mesh_shardings):
  """We either load the necessary prefill state or generate it.  """
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  if config.load_from_prefill_dir:
    kv_cache_restore_args = jax.tree_map(lambda sharding: orbax.checkpoint.type_handlers.ArrayRestoreArgs(sharding=sharding),
                                         kv_cache_mesh_shardings)
    next_token_restore_args = orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=P(None))
    pos_restore_args = orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=P(None))
    restore_args = {"cache": kv_cache_restore_args, "next_token": next_token_restore_args, "pos": pos_restore_args}
    blob = orbax_checkpointer.restore(config.prefill_cache_dir, restore_args=restore_args)
    blob["cache"] = jax.tree_map(lambda x : flax.linen.spmd.LogicallyPartitioned(x, mesh.axis_names), blob["cache"])
    max_logging.log(f"Restored prefill cache from {config.prefill_cache_dir}")
    return blob["cache"], blob["next_token"], blob["pos"]
  else:
    cache, next_token, pos = compute_prefill(config, model, state, rng, sp_tokenizer, mesh,
                                             state_mesh_shardings, kv_cache_mesh_shardings)
    max_logging.log(f"Computed prefill cache {config.prefill_cache_dir}")

    if config.prefill_cache_dir != "":
      blob = {"cache":cache, "next_token":next_token, "pos":pos}
      orbax_checkpointer.save(config.prefill_cache_dir, max_utils.unbox_logicallypartioned(blob))
      max_logging.log(f"Wrote prefill cache to {config.prefill_cache_dir}")
    return cache, next_token, pos




def decode_loop(config, state=None):
  """Decoding loop for the Transformer model."""
  assert config.add_eos is False,\
    "For decoding, we must set add_eos=False"
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir,
                                                                     config.enable_checkpointing,
                                                                     config.async_checkpointing,
                                                                     config.checkpoint_period)
  rng = random.PRNGKey(0)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  model = Transformer(config, mesh = mesh)
  _, sp_tokenizer = create_data_iterator_with_tokenizer(config, mesh)
  state, state_mesh_annotations = max_utils.setup_decode_state(
    model, config, rng, mesh, checkpoint_manager
  )
  kv_cache_annotations = max_utils.get_kv_cache_annotations(model, config, rng, mesh)

  assert state.opt_state == {}, "non null opt_state in checkpoint"
  num_params = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"Number of model params={num_params/10**9:.3f} billion")

  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  kv_cache_mesh_shardings = jax.tree_map(
    lambda p: jax.sharding.NamedSharding(mesh, p), kv_cache_annotations)
  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None))

  prefill_cache, new_id, new_position = prefill_or_load(config, model, state, rng, sp_tokenizer,\
                                                   mesh, state_mesh_shardings, kv_cache_mesh_shardings)

  partial_ar_predict_step = functools.partial(ar_predict_single_token, model=model)
  partial_ar_predict_step.__name__ = "partial_ar_predict_step"
  p_ar_predict_step = jax.jit(
      partial_ar_predict_step,
      in_shardings=(replicated_sharding, replicated_sharding, kv_cache_mesh_shardings, state_mesh_shardings, None),
      out_shardings=(replicated_sharding, kv_cache_mesh_shardings, replicated_sharding),
      donate_argnums=2
  )

  new_cache = prefill_cache
  outputs = []
  first_profiling_step = config.max_prefill_predict_length + config.skip_first_n_steps_for_profiler
  last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1,
                                first_profiling_step, config.max_target_length - 1)

  outputs = []
  #add the new_id which is the first generated token to outputs
  outputs = [new_id]

  for step in range(config.max_prefill_predict_length, config.max_target_length-1):
    if step == first_profiling_step:
      max_utils.activate_profiler(config)
    new_position, new_cache, new_id = p_ar_predict_step(new_id, new_position, new_cache, state, rng)
    outputs.append(new_id)
    if step == last_profiling_step:
      jax.block_until_ready(outputs)
      max_utils.deactivate_profiler(config)

  new_text, _ = decode_tokens([int(x[0,0]) for x in outputs], sp_tokenizer)
  max_logging.log(f"Completion: `{config.prompt}` -> `{new_text}`")
  if config.autoregressive_decode_assert != "":
    assert new_text==config.autoregressive_decode_assert, \
    f"generated text mismatch {new_text=} {config.autoregressive_decode_assert=}"

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


  decode_loop(pyconfig.config)

if __name__ == "__main__":
  app.run(main)
