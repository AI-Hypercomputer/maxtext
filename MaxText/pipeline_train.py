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
from jax.tree_util import tree_leaves_with_path # Import the function to get paths




def print_pytree_leaf_shapes(pytree):
  """
  Prints the shapes of all leaf nodes in a given JAX PyTree,
  along with their respective keypaths.

  Args:
    pytree: The PyTree whose leaf shapes and keypaths are to be printed.
  """
  print("Shapes of PyTree leaves with keypaths:")
  # Use tree_leaves_with_path to get both the path and the leaf
  leaves_with_paths = tree_leaves_with_path(pytree)

  for i, (key_path, leaf) in enumerate(leaves_with_paths):
    # Format the key_path for readability (e.g., 'a.b.c' or '0.1.key')
    path_str = ".".join(str(p) for p in key_path) if key_path else "root"

    if hasattr(leaf, 'shape'):
      print(f"  Leaf {i}: Path = '{path_str}', Shape = {leaf.shape}")
    else:
      # For non-JAX array leaves, print the path and a note
      print(f"  Leaf {i}: Path = '{path_str}', (Not a JAX array or does not have a shape attribute)")
ppl = print_pytree_leaf_shapes

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
  
  grad_func = jax.value_and_grad(loss_fn, argnums=4, has_aux=True)
  (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, is_train=True)

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




def get_state_and_grads(config, inputs=None, params=None, run_train=True):
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )
  if params is not None:
    state = state.replace(params=params)


  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)
  donate_argnums_train = () # Keep the state 

  #functional_train=functools.partial(train_step, model, config, state_mesh_shardings)
  functional_train=functools.partial(train_step_grads, model, config, state_mesh_shardings)

  p_train_step = jax.jit(
      functional_train,
      #in_shardings=in_shard_train,
      #out_shardings=out_shard_train,
      static_argnums=static_argnums_train,
      donate_argnums=donate_argnums_train,
  )

  if run_train:
    if not inputs:
      example_batch = load_next_batch(data_iterator, None, config) # None=example_batch
    else:
      example_batch = inputs
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          grads, metrics = p_train_step(state, example_batch, init_rng) #using same rng
  else:
    return state, None, None

  loss = metrics['scalar']['learning/loss']
  print(f"{loss=}", flush=True)
  return state, grads, example_batch


def reshape_tree(pytree):
  def reshape_pp_to_dp(arr, scan_index=1):
    arr_shape = jnp.shape(arr)
    repeats, stages = arr_shape[0], arr_shape[1]
    arr_flat = jnp.reshape(arr, (repeats*stages,) + arr_shape[2:])
    if scan_index > 0:
      # swap first index with scan_index using jnp.transpose:
      perm = [scan_index] + list(range(scan_index)) + list(range(scan_index + 1, len(arr_flat.shape)))
      arr_flat = jnp.transpose(arr_flat, perm)
    return arr_flat
  # Perform a tree map of reshape_pp_to_dp
  dp_pytree = jax.tree_util.tree_map(reshape_pp_to_dp, pytree)
  return dp_pytree

pp_config = pyconfig.initialize(
      [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
      enable_checkpointing=False,
      enable_goodput_recording=False,
      run_name="circular_moe",
      max_target_length=128,
      base_emb_dim=28,
      base_moe_mlp_dim=56,
      ici_pipeline_parallelism=4,
      base_num_decoder_layers=8,
      num_pipeline_microbatches=8,
      per_device_batch_size=4,
      num_experts=4,
      num_experts_per_tok=2,
      megablox=False,
      sparse_matmul=False,
      capacity_factor=1,
      dataset_type="synthetic",
      attention_type="mla",
      decoder_block="deepseek",
      opt_type="sgd",
  )

dp_config = pyconfig.initialize(
      [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
      enable_checkpointing=False,
      enable_goodput_recording=False,
      run_name="circular_moe",
      max_target_length=128,
      base_emb_dim=28,
      base_moe_mlp_dim=56,
      ici_data_parallelism=4,
      base_num_decoder_layers=8,
      per_device_batch_size=4,
      num_experts=4,
      num_experts_per_tok=2,
      megablox=False,
      sparse_matmul=False,
      capacity_factor=1,
      dataset_type="synthetic",
      attention_type="mla",
      decoder_block="deepseek",
      opt_type="sgd",
  )


# Run PP to get real grads and weights
pp_state, pp_grads, inputs = get_state_and_grads(pp_config)
# Run DP to get pytree structure for DP (random possibly different weights)
dp_state, _, _ = get_state_and_grads(dp_config, run_train=False)

# Copy PP Params into DP
pp_module_subtree = pp_state.params['params']['decoder']['pipeline_module']['layers']
dp_module_subtree = reshape_tree(pp_module_subtree)
dp_params_copy=dp_state.params.copy()
dp_params_copy['params']['decoder']['layers'] = dp_module_subtree

# Run DP with same weights and inputs as PP
dp_rp_state, dp_rp_grads, _ = get_state_and_grads(dp_config, params=dp_params_copy, inputs=inputs)

# Reshape PP grads so identical pytree structure as pp grads
pp_layer_grads = reshape_tree(pp_grads['params']['decoder']['pipeline_module']['layers'])
#pp_layer_grads = reshape_tree(pp_grads['params']['decoder']['pipeline_module']['layers']['weights'])
dp_layer_grads = dp_rp_grads['params']['decoder']['layers']
breakpoint()
#assert jax.numpy.allclose(f1_value, f2_value, rtol=1e-2, equal_nan=False) # Loss
#assert jax.numpy.allclose(pp_layer_grads, dp_layer_grads, rtol=1e-1, equal_nan=False) # why is this broken
assert jax.numpy.allclose(pp_layer_grads, dp_layer_grads, atol=1e-3, equal_nan=False)
