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
import common_types
import functools

import max_logging

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils


import json
import yaml
import flax
from flax.training import train_state
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import optax
import os
from typing import Tuple
from tensorboardX import writer

from google.cloud import storage

def find_nans_and_infs(pytree):
  def finder(x):
    return jnp.any(jnp.isinf(x) | jnp.isnan(x))
  bad_pytree = jax.tree_map(finder, pytree)
  return jax.tree_util.tree_flatten(bad_pytree)

def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jax.tree_util.tree_reduce(
      lambda x, y: x + jax.numpy.sum(jax.numpy.square(y)), x, initializer=0.0
  ) ** 0.5

def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters

def calculate_bytes_from_pytree(params):
  params_bytes = jax.tree_util.tree_map(lambda x : x.nbytes, params)
  total_bytes = jax.tree_util.tree_reduce(lambda x, y: x + y, params_bytes)
  return total_bytes

def summarize_size_from_pytree(params):
  num_params = calculate_num_params_from_pytree(params)
  num_bytes = calculate_bytes_from_pytree(params)
  return num_params, num_bytes, num_bytes/num_params

def activate_profiler(config):
  if config.enable_profiler and (config.upload_all_profiler_results or jax.process_index() == 0):
    jax.profiler.start_trace(config.tensorboard_dir)

def deactivate_profiler(config):
  if config.enable_profiler and (config.upload_all_profiler_results or jax.process_index() == 0):
    jax.profiler.stop_trace()

def initialize_summary_writer(config):
  return writer.SummaryWriter(config.tensorboard_dir) if jax.process_index() == 0 else None

def close_summary_writer(summary_writer):
  if jax.process_index() == 0:
    summary_writer.close()

def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {}
  for val in metrics['scalar']:
    metrics_dict[val] = float(metrics['scalar'][val])
  metrics_dict['step'] = float(step)
  metrics_dict['run_name'] = run_name
  return metrics_dict

def write_metrics_locally(metrics, step, config, file):
  """Writes metrics locally for testing"""
  if step == 0:
    file.truncate(0)

  metrics_dict = _prepare_metrics_for_json(metrics, step, config.run_name)
  file.write(str(json.dumps(metrics_dict))+'\n')

  if step == config.steps - 1:
    file.close()

def add_config_to_summary_writer(config, summary_writer):
  """Writes config params to tensorboard"""
  if jax.process_index() == 0:
    for key, value in config.get_keys().items():
      add_text_to_summary_writer(key, str(value), summary_writer)

def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary"""
  if jax.process_index() == 0:
    summary_writer.add_text(key, value)

def write_metrics_for_gcs(metrics, step, config, running_metrics):
  """Writes metrics to gcs"""
  metrics_dict_step = _prepare_metrics_for_json(metrics, step, config.run_name)
  running_metrics.append(metrics_dict_step)
  if (step + 1) % config.log_period == 0 or step == config.steps - 1:
    start_step = (step // config.log_period) * config.log_period
    metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
    with open(metrics_filename, 'w', encoding="utf8") as metrics_for_gcs:
      for metrics_step in running_metrics:
        metrics_for_gcs.write(str(json.dumps(metrics_step))+'\n')

    metrics_for_gcs.close()
    gcs_filename=os.path.join(config.metrics_dir, metrics_filename)
    max_logging.log(f"Moving file {metrics_filename} to GCS...")
    upload_blob(gcs_filename, metrics_filename)
    max_logging.log(f"File {metrics_filename} moved successfully!")
    running_metrics = [] # reset running_metrics to empty list
  return running_metrics

def write_config_raw_keys_for_gcs(raw_keys):
  """Writes config raw keys to GCS"""
  if not raw_keys["save_config_to_gcs"] or jax.process_index() != 0:
    return
  max_logging.log("Writing config to GCS...")

  raw_keys_dict = dict(raw_keys)
  filename = "config.yml"
  with open(filename, 'w', encoding="utf8") as config_for_gcs:
    yaml.dump(raw_keys_dict, config_for_gcs)
  config_for_gcs.close()

  gcs_filename=os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], filename)
  max_logging.log(f"Moving file {filename} to GCS...")
  upload_blob(gcs_filename, filename)
  max_logging.log(f"File {filename} moved successfully!")

def parse_gcs_bucket_and_prefix(destination_gcs_name):
  path_parts = destination_gcs_name.replace("gs://", "").split("/")
  bucket = path_parts.pop(0)
  key = "/".join(path_parts)
  return bucket, key

def upload_blob(destination_gcs_name, source_file_name):
  """Uploads a file to a GCS location"""
  bucket_name, prefix_name = parse_gcs_bucket_and_prefix(destination_gcs_name)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(prefix_name)
  blob.upload_from_filename(source_file_name)

def initialize_jax_distributed_system():
  """ The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
      indirection in MaxText to avoid breaking the call sites unnecessarily.

      Currently jax.distributed.initialize() fully works as expected!
  """
  max_logging.log("Attempting to initialize the jax distributed system...")
  jax.distributed.initialize()
  max_logging.log("Jax distributed system initialized!")

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

def create_device_mesh(config, devices=None):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas """
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1
  num_devices_per_slice = num_devices//num_slices

  multi_slice_env = num_slices > 1

  dcn_parallelism = [config.dcn_data_parallelism, config.dcn_fsdp_parallelism,
                     config.dcn_sequence_parallelism, config.dcn_tensor_parallelism,
                     config.dcn_autoregressive_parallelism]
  ici_parallelism = [config.ici_data_parallelism, config.ici_fsdp_parallelism,
                     config.ici_sequence_parallelism, config.ici_tensor_parallelism,
                     config.ici_autoregressive_parallelism]

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, 'ICI')

  if multi_slice_env:
    dcn_parallelism = fill_unspecified_mesh_axes(dcn_parallelism, num_slices, 'DCN')
    mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism, devices)
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)

  max_logging.log(f"Num_devices: {num_devices}, shape {mesh.shape}")

  return mesh

def unbox_logicallypartioned(
    boxed_pytree):
  """ Unboxes the flax.LogicallyPartitioned pieces

    Args:
      boxed_pytree: a pytree that includes LogicallyPartitioned
        leaves.
    Returns:
      a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(lambda x: x.unbox() if \
        isinstance(x, flax.linen.spmd.LogicallyPartitioned) \
        else x, boxed_pytree, \
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned))

def init_decode_state(apply_fn, params):
  """Init train state with null opt state for decode."""
  state = train_state.TrainState(
    step=0,
    apply_fn=apply_fn,
    params=params,
    tx=None, # type: ignore
    opt_state={}
    )
  return state

def init_training_state(apply_fn, params, tx):
  """Init train state with null opt state for decode."""
  state = train_state.TrainState.create(
    apply_fn=apply_fn,
    params=params,
    tx=tx
    )
  return state

def init_initial_state(model, tx, config, is_training, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, is_training, key
  """
  input_shape = (
      config.global_batch_size_to_load,
      config.max_target_length
  )
  model_vars = model.init({'params': key, 'dropout': key, 'aqt': key},
                          jnp.ones(input_shape, dtype=jnp.int32),
                          jnp.ones(input_shape, dtype=jnp.int32))
  if is_training:
    return init_training_state(model.apply, model_vars['params'], tx)
  return init_decode_state(model.apply, model_vars['params'])

def setup_decode_state(model, config, rng, mesh, checkpoint_manager):
  is_training = False
  return setup_initial_state(model, None, config, rng, mesh, checkpoint_manager, is_training)

def setup_training_state(model, tx, config, rng, mesh, checkpoint_manager):
  is_training = True
  return setup_initial_state(model, tx, config, rng, mesh, checkpoint_manager, is_training)

def setup_initial_state(model, tx, config, rng, mesh, checkpoint_manager, is_training=True):
  """ We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    is_training: True to initialize training state, False for decode state

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  unboxed_abstract_state, state_mesh_annotations = get_abstract_state(model, tx, config, rng, mesh, is_training)

  # Initialization
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state, raw_params = checkpointing.load_state_if_possible(checkpoint_manager,
                                                config.load_parameters_path,
                                                config.load_full_state_path,
                                                unboxed_abstract_state,
                                                mesh,
                                                state_mesh_annotations)

    state_mesh_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
    if not state:
      init_state_partial = functools.partial(init_initial_state, model, tx, config, is_training)
      state = jax.jit(
          init_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings
      )(rng)
      if raw_params: # If we loaded a partial state, we need to merge it.
        state = state.replace(params = raw_params)
    raw_params = None

  state = unbox_logicallypartioned(state)
  return state, state_mesh_annotations


# Learning Rate Schedule
# -----------------------------------------------------------------------------

def create_learning_rate_schedule(config):
  """Creates a warmup and cosine decay learning rate schedule:
  We take inspiration from Llama2's learning rate (LR) schedule, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  Learning rate schedule has either two or three parts:
  1) Linear warmup from 0 to [learning_rate] over steps 0 to [learning_rate_schedule_steps * warmup_steps_fraction]
  2) Cosine from [learning_rate] to [learning_rate * cosine_learning_rate_final_fraction] until learning_rate_schedule_steps
  3) Constant learning rate of 0 from learning_rate_schedule_steps to steps.
  The zero learning rate section can be used to more accurately measure the fully trained model's performance.
  """
  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = (step) / len_steps
      a = 0.5 * (jnp.cos(jnp.pi*pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr
    return schedule

  lr = config.learning_rate
  cos_final_lr = lr * config.cosine_learning_rate_final_fraction

  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  cos_steps = config.learning_rate_schedule_steps - warmup_steps
  constant_zero_steps = config.steps - config.learning_rate_schedule_steps

  warmup_schedule = optax.linear_schedule(
      init_value=0.0,
      end_value=lr,
      transition_steps=warmup_steps
  )
  cos_schedule = make_cos_schedule(lr, cos_final_lr, cos_steps)
  constant_schedule = optax.constant_schedule(0.0)

  pieces = [warmup_schedule, cos_schedule]
  boundaries=[
   warmup_steps,
   warmup_steps + cos_steps,
   ]

  if constant_zero_steps > 0:
    pieces.append(constant_schedule)
    boundaries.append(warmup_steps + cos_steps + constant_zero_steps)

  return optax.join_schedules(pieces, boundaries)


# Cross entropy implementation is taken from original T5X codebase:
# https://github.com/google-research/t5x/blob/ace831eea1e2742b4299cd1a9af7e4f302038351/t5x/losses.py#L25-L101
@jax.custom_vjp
def cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray,
                              z_loss: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.
  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.
  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxilliary z-loss loss term.
  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                 jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (logits, targets, z_loss, exp_shifted, sum_exp, #pytype: disable=bad-return-type  #jax-ndarray
                                log_softmax, log_z)


def _cross_entropy_with_logits_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
               jnp.ndarray, jnp.ndarray], g: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]  # Ignore z_loss component as that is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = (
      jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp -
      targets)
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (jnp.asarray(g_logits,
                      logits.dtype), jnp.asarray(g_targets, targets.dtype),
          jnp.array(0.0))  # sets z-loss coeff gradient to 0

cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd,
                                 _cross_entropy_with_logits_bwd)

def get_abstract_state(model, tx, config, rng, mesh, is_training=True):
  """ Get a shaped abstraction of the state (including optimizer)"""
  init_state_partial = functools.partial(init_initial_state, model, tx,
                                              config, is_training)
  abstract_state = jax.eval_shape(init_state_partial, rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  unboxed_abstract_state = unbox_logicallypartioned(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return unboxed_abstract_state, state_mesh_annotations

def get_kv_cache_annotations(model, config, rng, mesh):
  """ Get a shaped abstraction of the state (including optimizer)"""

  def init_kv_cache(model, config):
    input_shape = (
      config.global_batch_size_to_load,
      config.max_target_length
    )

    model_vars = model.init({'params': rng, 'dropout': rng, 'aqt': rng},
                          jnp.ones(input_shape),
                          jnp.ones(input_shape),
                          model_mode=common_types.MODEL_MODE_PREFILL)
    return model_vars['cache']

  init_kv_cache_partial = functools.partial(init_kv_cache, model,
                                              config)
  abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations
