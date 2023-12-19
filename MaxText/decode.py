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
from flax.linen import partitioning as nn_partitioning
import numpy as np

import pyconfig
import max_utils
import temperature_sampler
from input_pipeline import create_data_iterator_with_tokenizer
# from layers import models
from layers import llama2

import checkpointing

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

import max_logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

models = llama2
Transformer = models.Transformer


def decode_tokens(toks, tokenizer, eos_id):
  if np.argmax(toks == eos_id) > 0:
    valid_toks = toks[:np.argmax(toks == eos_id)]
  else:
    valid_toks = toks
    valid_toks[-1] = eos_id

  valid_toks = valid_toks.astype(np.int32)
  return tokenizer.detokenize(valid_toks).numpy().decode("utf-8"), len(valid_toks)


def encode_strings(strs, max_len, tokenizer):
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    toks = tokenizer.tokenize(s).numpy()
    # Remove EOS token in prompt.
    tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
  return tokenized_batch

def predict_step(inputs,
                 state,
                 rngkey,
                 model,
                 config):
  """Predict language model on a batch."""
  # NOTE: wtf are we adding inputs.shape[2:] here?  it's almost always empty??
  target_shape = (inputs.shape[0], config.max_predict_length) + inputs.shape[2:]

  initial_variables = model.init(
      {'params': rngkey, 'dropout': rngkey, 'aqt': rngkey},
      jnp.ones(target_shape, config.dtype),
      None,
      enable_dropout=False,
      decode=True,
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
        decode=True,
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
  _, sp_tokenizer = create_data_iterator_with_tokenizer(config, mesh)
  state, state_mesh_annotations = max_utils.setup_decode_state(
    model, config, rng, mesh, checkpoint_manager
    )
  assert state.opt_state == {}, "non null opt_state in checkpoint"
  num_params = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"Number of model params={num_params/10**9:.3f} billion")

  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  replicated_sharding = jax.sharding.NamedSharding(mesh, P(None, None))
  p_predict_step = jax.jit(
      functools.partial(predict_step, model=model, config=config),
      in_shardings=(replicated_sharding, state_mesh_shardings, None),
      out_shardings=None
  )

  # Encode the demo prompt.
  tokenized_prompts = encode_strings(
      [config.prompt], config.max_predict_length, sp_tokenizer)

  if config.metrics_file:
    local_metrics_file = open(config.metrics_file, 'a', encoding="utf8")
    metrics= {'scalar': {} }
  max_utils.activate_profiler(config)
  for step in np.arange(config.steps):
    rng, rng_to_use = jax.random.split(rng)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      seqs = p_predict_step(tokenized_prompts, state, rng_to_use)
      max_logging.log(f"Anisha:seqs={seqs}")
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
