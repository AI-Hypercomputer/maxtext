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

import os
from absl import app
import datetime
from flax.linen import partitioning as nn_partitioning
import numpy as np
import optax

from layers import Transformer
import pyconfig
from input_pipeline import create_data_iterator_with_tokenizer
import max_utils
import temperature_sampler

import checkpointing

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

import sys

import max_logging

cc.initialize_cache(os.path.expanduser("~/jax_cache"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

def match_input_and_output_stream(prompt, outputs, tokenizer):
  for i in range(len(prompt)):
      prompt_mini = prompt[0:i+1]
      prompt_mini_arr = np.array(prompt_mini, dtype=np.int32)
      prompt_mini_str = decode_tokens(prompt_mini_arr, tokenizer)
      output_mini = outputs[i:i+1]
      output_mini_arr = np.array(output_mini, dtype=np.int32)
      print(output_mini_arr)
      output_mini_str = decode_tokens(output_mini_arr, tokenizer)
      print(f"{prompt_mini_str} -> {output_mini_str}")

def decode_tokens(toks, tokenizer):
  return tokenizer.detokenize(toks).numpy().decode("utf-8"), len(toks)


def encode_strings(strs, max_len, tokenizer):
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    toks = tokenizer.tokenize(s).numpy()
    # Remove EOS token in prompt.
    tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
  return tokenized_batch


def prefill_predict_step(inputs,
                 state,
                 rngkey,
                 model,
                 config):
  decoder_segment_ids = jax.numpy.zeros( inputs.shape)

  #cache = initial_variables["cache"]
  cache = None

  flat_logits = model.apply(
    {
        "params": state.params
    },
    inputs,
    None,
    decoder_segment_ids=decoder_segment_ids,
    enable_dropout=False,
    model_mode="train",
    rngs={'aqt': rngkey},
    max_decode_length=config.max_predict_length 
  )
  #cache = new_vars["cache"]
  #jax.debug.print("cache: {}", cache)

  #cache = initial_variables["cache"]
  return flat_logits


def ar_predict_step(inputs,
                 state,
                 rngkey,
                 model,
                 config):
  """Predict language model on a batch."""
  target_shape = (inputs.shape[0], config.max_predict_length)

  initial_variables = model.init(
      {'params': rngkey, 'dropout': rngkey, 'aqt': rngkey},
      jnp.ones(target_shape, config.dtype),
      None,
      enable_dropout=False,
      model_mode="autoregressive",
      max_decode_length=config.max_predict_length
  )
  cache = initial_variables["cache"]

  def tokens_ids_to_logits(flat_ids, flat_cache, aqt_rng):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = model.apply(
        {
            "params": state.params,
            "cache": flat_cache
        },
        flat_ids,
        None,
        enable_dropout=False,
        model_mode="autoregressive",
        rngs={'aqt': aqt_rng},
        max_decode_length=config.max_predict_length,
        mutable=["cache"])
    new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # search over possible sequences given input encoding.
  seqs = temperature_sampler.temperature_sample(
      inputs,
      cache,
      tokens_ids_to_logits,
      rngkey,
      temperature=config.sampling_temperature,
      topk=config.sampling_top_k,
      eos_token=config.eos_id)

  return seqs

def decode_loop(config, state=None):
  """Decoding loop for the Transformer model."""
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir,
                                                                     config.enable_checkpointing,
                                                                     config.async_checkpointing,
                                                                     config.save_period)
  rng = random.PRNGKey(0)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  model = Transformer(config, mesh = mesh)

  tx = optax.adamw(
    max_utils.create_learning_rate_schedule(config)
  ) # TODO: we need an optax.GradientTransformation to form a TrainState, but we don't use it when decoding


  _, sp_tokenizer = create_data_iterator_with_tokenizer(config, mesh)

  do_weird_visualization = False
  if do_weird_visualization:
    prompt = [0, 4153,213,  1586,  2247,  1080,    25,  2049,  1625,     8,     5, 23135, 5289,     8,  1127,    42]
    output = [25,213,  1586,  2247,     4,  1127,  2520, 10506,     8,     5, 23135,  5289,   8,  1127,    42, 10802,]

    match_input_and_output_stream(prompt, output, sp_tokenizer)
  
    sys.exit(1)
  

  state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, rng, mesh, checkpoint_manager)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None, None))

  # Encode the demo prompt.
  np_tokenized_prompts = encode_strings(
      [config.prompt], config.max_prefill_predict_length, sp_tokenizer)
  np_tokenized_prompts = [0, 6970, 12643,    13,   176,    12,     0,     0,     0,     0,
            0,     0,     0,     0,     0, 0]
  print(np_tokenized_prompts)
  #np_tokenized_prompts=  [0, 4153,213,  1586,  2247,  1080,    25,  2049,  1625,     8,     5, 23135, 5289,     8,  1127,    42]
  
  tokenized_prompts = jax.device_put(np.vstack([np_tokenized_prompts]*(int(config.per_device_batch_size) * jax.device_count())), jax.sharding.NamedSharding(mesh, P()))

  partial_prefill_predict_step = functools.partial(prefill_predict_step, model=model, config=config)
  p_prefill_predict_step = jax.jit(
      partial_prefill_predict_step,
      in_shardings=(replicated_sharding, state_mesh_shardings, None),
      out_shardings=None
  )

  #import pdb ; pdb.set_trace()
  prefill_output = p_prefill_predict_step(tokenized_prompts, state, rng)
  #import pdb ; pdb.set_trace()
  print(f"{prefill_output.shape=}")
  print(f"{tokenized_prompts.shape=}")
  indices = jax.numpy.argmax(prefill_output, axis=2)
  decoded_tokens = decode_tokens(np.array(indices)[0], sp_tokenizer)

  match_input_and_output_stream(np_tokenized_prompts, indices[0,:], sp_tokenizer)

  sys.exit(0)


  partial_ar_predict_step = functools.partial(ar_predict_step, model=model, config=config)
  p_ar_predict_step = jax.jit(
      partial_ar_predict_step,
      in_shardings=(replicated_sharding, state_mesh_shardings, None),
      out_shardings=None
  )

  if config.metrics_file:
    local_metrics_file = open(config.metrics_file, 'a', encoding="utf8")
    metrics= {'scalar': {} }
  max_utils.activate_profiler(config)

  last_step_completion = datetime.datetime.now()
  step_times = []
  for step in np.arange(config.steps):
    rng, rng_to_use = jax.random.split(rng)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      seqs = jax.block_until_ready(p_ar_predict_step(tokenized_prompts, state, rng_to_use))
      new_time = datetime.datetime.now()
      step_times.append((new_time-last_step_completion).total_seconds())
      last_step_completion = new_time

  print(f"Median Step Time Seconds {sorted(step_times)[len(step_times)//2]} {step_times=}")

  decoded_string, num_tokens_decoded = decode_tokens(np.array(seqs)[0], sp_tokenizer, config.eos_id)
  max_logging.log(f"Decoding #{step} (num tokens {num_tokens_decoded}):\n\t{decoded_string}")
  if config.metrics_file:
    metrics['scalar']['num_tokens'] = num_tokens_decoded
    max_utils.write_metrics_locally(metrics, step, config, local_metrics_file)
  max_utils.deactivate_profiler(config)



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path

  
  decode_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
