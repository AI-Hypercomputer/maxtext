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

# pylint: disable=bare-except, consider-using-generator
"""Utils that are only interesting to MaxText. """

import jax
from jax.sharding import PartitionSpec as P
from jax.experimental.serialize_executable import deserialize_and_load


import pickle
import functools
import input_pipeline
from mlperf_logging import mllog


def get_functional_train_with_signature(train_step, mesh, state_mesh_annotations, model, config, is_train):
  """ Get the shardings (both state and data) for train_step """
  functional_train = get_functional_train_step(train_step, model, config, is_train)
  functional_train.__name__ = "train_step" if is_train else "eval_step"
  data_pspec = P(*config.data_sharding)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None) # State, batch, rng
  out_shardings = (state_mesh_shardings, None) # State, metrics
  static_argnums = () # We partial out the static argnums of model, config and is_train
  # This is the index of the state - we allow the compiler to make use of this memory is_train
  donate_argnums = 0 if is_train else ()
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums

def get_functional_train_step(train_step, model, config, is_train):
  return functools.partial(train_step, model, config, is_train=is_train)

def load_compiled(config, partial_train, state):
  """ # Loading a serialized compiled train step function."""
  # Currently partial_train and state  are needed to reconstruct
  # input/output shapes to construct the in_trees and out_trees for load API
  # Parker is working on a serializing these
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as f:
      serialized_compiled = pickle.load(f)
    return serialized_compiled

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = input_pipeline.get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree)
  return p_train_step

def init_mllog(config, start_step):
  """an initial mllog for mlperf sumbission compliance check."""
  if jax.process_index() == 0 and config.model_name.startswith("gpt3"):
    mllogger = mllog.get_mllogger()
    mllogger.event(mllog.constants.CACHE_CLEAR)
    mllogger.start(mllog.constants.INIT_START)
    mllogger.event(mllog.constants.SUBMISSION_ORG, 'Google')
    mllogger.event(mllog.constants.SUBMISSION_PLATFORM, 'tpu-v5p')
    mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
    mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)
    mllogger.event(mllog.constants.SUBMISSION_BENCHMARK, mllog.constants.GPT3)
    mllogger.event(mllog.constants.OPT_NAME, mllog.constants.ADAM)
    mllogger.event(mllog.constants.OPT_BASE_LR, config.learning_rate)
    mllogger.event(mllog.constants.OPT_END_LR, config.cosine_learning_rate_final_fraction)
    mllogger.event(mllog.constants.OPT_WEIGHT_DECAY, config.adam_weight_decay)
    mllogger.event(mllog.constants.OPT_LR_DECAY_STEPS,
                   int(config.learning_rate_schedule_steps * (1 - config.warmup_steps_fraction)))
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS,
                   int(config.learning_rate_schedule_steps * config.warmup_steps_fraction + 1))
    mllogger.event(mllog.constants.OPT_LR_DECAY_SCHEDULE, 'cosine with linear warmup')
    mllogger.event(mllog.constants.INIT_CHECKPOINT_STEP, start_step)
    mllogger.event(mllog.constants.OPT_ADAM_BETA_1, config.adam_b1)
    mllogger.event(mllog.constants.OPT_ADAM_BETA_2, config.adam_b2)
    mllogger.event(mllog.constants.OPT_ADAM_EPSILON, config.adam_eps)
    mllogger.event(mllog.constants.OPT_GRADIENT_CLIP_NORM, config.gradient_clipping_threshold)
    mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE, config.global_batch_size_to_train_on)
    mllogger.event(mllog.constants.MAX_SEQUENCE_LENGTH, config.max_target_length)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, 1)
    mllogger.event(mllog.constants.EVAL_SAMPLES, 24567)

    mllogger.end(mllog.constants.INIT_STOP)
    mllogger.start(mllog.constants.RUN_START)

def is_early_stop_mllog(config, step, eval_loss) -> bool:
  """an early stop function with mllog for mlperf sumbission compliance check."""
  is_early_stop = eval_loss <= config.target_eval_loss
  if jax.process_index() == 0 and config.model_name.startswith("gpt3"):
    eval_frequency_tokens = config.eval_interval * config.global_batch_size_to_train_on * config.max_target_length
    current_epoch_num = step * config.global_batch_size_to_train_on * config.max_target_length
    first_epoch_num = current_epoch_num - eval_frequency_tokens
    mllogger = mllog.get_mllogger()
    mllogger.end(
      mllog.constants.BLOCK_STOP,
      metadata={'first_epoch_num': first_epoch_num},
    )
    mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      eval_loss,
      metadata={'epoch_num': current_epoch_num},
    )
    if is_early_stop:
      mllogger.end(mllog.constants.RUN_STOP, metadata={'status': 'success'})
      mllogger.event(mllog.constants.TRAIN_SAMPLES, current_epoch_num)
    else:
      mllogger.start(
        mllog.constants.BLOCK_START,
        metadata={
          'epoch_count': eval_frequency_tokens,
          'first_epoch_num': current_epoch_num,
        },
      )
  return is_early_stop
