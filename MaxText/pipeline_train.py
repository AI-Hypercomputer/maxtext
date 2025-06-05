import functools
import os.path
import sys
import unittest

import pytest

import jax
from jax.sharding import Mesh
import jax.numpy as jnp

from flax.core import meta
from flax import linen as nnx

from MaxText import maxtext_utils
from MaxText import max_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.globals import PKG_DIR
from MaxText.layers import pipeline
from MaxText.layers import simple_layer
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.train import main as train_main
from MaxText.layers import deepseek
from MaxText.train import (
    check_example_batch,
    create_goodput_recorder,
    eval_step,
    EPS,
    get_first_step,
    load_next_batch,
    record_goodput,
    record_scalar_metrics,
    save_checkpoint,
    setup_mesh_and_model,
    train_step,
    validate_train_config,
    loss_fn
)
from flax.linen import partitioning as nn_partitioning


def train_step_grads(model, config, state_mesh_shardings, state, data, dropout_rng):
  """
  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    grads: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
  """
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = [], [], [], loss_fn
  
  grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
  (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True)

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/total_weights": aux["total_weights"],
      "learning/grad_norm": max_utils.l2norm_pytree(grads)
  }
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }
  return raw_grads, metrics


config = pyconfig.initialize(
    [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
    enable_checkpointing=False,
    enable_goodput_recording=False,
    run_name="circular_ag_once",
    max_target_length=128,
    base_emb_dim=28,
    ici_pipeline_parallelism=2,
    base_num_decoder_layers=8,
    num_pipeline_microbatches=8,
    per_device_batch_size=4,
    pipeline_fsdp_ag_once=True,
    decoder_block="simple",
    dataset_type="synthetic",
)

def get_state_and_grads(config)
init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
    model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
)


(
    functional_train,
    in_shard_train,
    out_shard_train,
    static_argnums_train,
    donate_argnums_train,
) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

#functional_train=functools.partial(train_step, model, config, state_mesh_shardings)
functional_train=functools.partial(train_step_grads, model, config, state_mesh_shardings)

p_train_step = jax.jit(
    functional_train,
    in_shardings=in_shard_train,
    #out_shardings=out_shard_train,
    static_argnums=static_argnums_train,
    donate_argnums=donate_argnums_train,
)

example_batch = load_next_batch(data_iterator, None, config) # None=example_batch
with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    grads, metrics = p_train_step(state, example_batch, init_rng) #using same rng

loss = metrics['scalar']['learning/loss']
print(f"{loss=}", flush=True)


def convert_pp_params_to_non_pp():
    return False



