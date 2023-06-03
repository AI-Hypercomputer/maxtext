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
""" Common Max Utils needed by multiple modules"""
import checkpointing
import functools

import max_logging

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils

from jax.experimental.pjit import pjit

import json
import flax
from flax.training import train_state
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import optax


def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jax.tree_util.tree_reduce(
      lambda x, y: x + jax.numpy.sum(y ** 2), x, initializer=0.0
  ) ** 0.5

def activate_profiler(config):
  if config.enable_profiler:
    jax.profiler.start_trace(config.tensorboard_dir)

def deactivate_profiler(config):
  if config.enable_profiler:
    jax.profiler.stop_trace()

def write_metrics_locally(metrics, step, last_step, file):
  """Writes metrics locally for testing"""
  if step == 0:
    file.truncate(0)

  aux = {}
  for val in metrics['scalar']:
    aux[val] = float(metrics['scalar'][val])
  file.write(str(json.dumps(aux))+'\n')

  if step == last_step:
    file.close()

def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product/np.product(parallelism_vals)*-1

    assert determined_val >= 1 and determined_val.is_integer, f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == 'DCN' else "devices per slice"

  assert np.product(parallelism_vals) == target_product, f"Number of {target_type} {target_product} does not match\
    the product of the {parallelism_type} parallelism {np.product(parallelism_vals)}"

  return parallelism_vals

def create_device_mesh(config):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas """
  devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1
  num_devices_per_slice = num_devices//num_slices
  max_logging.log(f"Devices: {devices} (num_devices: {num_devices})")
  assert len(devices) > 1, "You must have at least two devices"

  multi_slice_env = hasattr(jax.devices()[0], 'slice_index')

  dcn_parallelism = [config.dcn_data_parallelism, config.dcn_fsdp_parallelism, config.dcn_tensor_parallelism]
  ici_parallelism = [config.ici_data_parallelism, config.ici_fsdp_parallelism, config.ici_tensor_parallelism]

  # Find possible unspecified parallelisms
  dcn_parallelism = fill_unspecified_mesh_axes(dcn_parallelism, num_slices, 'DCN')
  ici_parallelism = fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, 'ICI')

  if multi_slice_env:
    mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism)

  max_logging.log(f"Decided on mesh: {mesh}")

  return mesh

def unbox_logicallypartioned_trainstate(
    boxed_train_state: train_state.TrainState):
  """ Unboxes the flax.LogicallyPartitioned pieces in a train state.

    Args:
      boxed_train_state: a train state that includes LogicallyPartitioned
        leaves.
    Returns:
      a TrainState where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(lambda x: x.unbox() if \
        isinstance(x, flax.linen.spmd.LogicallyPartitioned) \
        else x, boxed_train_state, \
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned))

def init_train_state(model, tx, config, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, key
  """
  input_shape = (
      len(jax.devices()) * config.per_device_batch_size,
      config.max_target_length
  )
  model_vars = model.init({'params': key, 'dropout': key, 'aqt': key},
                          jnp.ones(input_shape),
                          jnp.ones(input_shape))
  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=model_vars['params'],
      tx=tx)
  return state


def setup_initial_state(model, tx, config, rng, mesh, checkpoint_manager):
  """ We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """
  init_train_state_partial = functools.partial(init_train_state, model, tx,
                                               config)
  abstract_state = jax.eval_shape(init_train_state_partial, rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  unboxed_abstract_state = unbox_logicallypartioned_trainstate(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
    state, raw_params = checkpointing.load_state_if_possible(checkpoint_manager,
                                                config.load_parameters_path,
                                                unboxed_abstract_state,
                                                mesh,
                                                state_mesh_annotations)

    if not state:
      state = pjit(
          init_train_state_partial,
          in_axis_resources=None,
          out_axis_resources=state_mesh_annotations
      )(rng)
      if raw_params: # If we loaded a partial state, we need to merge it.
        state = state.replace(params = raw_params)
    raw_params = None

  state = unbox_logicallypartioned_trainstate(state)
  return state, state_mesh_annotations


# Learning Rate Schedule
# -----------------------------------------------------------------------------
# learning rate scheduling
def rsqrt_schedule(init_value: float, shift: int = 0):
  def schedule(count):
    return init_value * (1 + count + shift)**-.5 * shift**.5
  return schedule

def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0,
          end_value=learning_rate,
          transition_steps=warmup_steps
          ),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
      ], boundaries=[warmup_steps],
      )
