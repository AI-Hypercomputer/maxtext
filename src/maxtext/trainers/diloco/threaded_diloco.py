#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# pylint: disable=protected-access
"""Non-SPMD, multi-threaded streaming DiLoCo implementation with single client Pathways."""

import copy
import datetime
import functools
import threading
import traceback
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from flax import linen as nn, nnx, struct
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.experimental import colocated_python
import numpy as np
import optax

from maxtext.common import checkpointing, profiler, metric_logger
from maxtext.common.goodput import maybe_record_goodput, GoodputEvent
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager, LearnerTransport, SyncerTransport
from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator
from maxtext.utils import exceptions
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from maxtext.utils.mesh_utils import partition_mesh_by_diloco_axis, stack_across_meshes_pytree


@jax.jit
def mix_frags(i_frag, o_frag, alpha):
  return jax.tree_util.tree_map(lambda x, y: alpha * x + (1 - alpha) * y, i_frag, o_frag)


def _normalize_to_null_layout(tree):
  """Rebuild every JAX array from raw shard data to ensure a consistent null layout.

  Pathways caches internal jits (jit__take, jit_scatter, etc.) by (shape, dtype, sharding)
  WITHOUT layout. When syncer params have mixed layouts across tensors that share the same
  (shape, dtype, sharding) — e.g. some tiled from learner transport, others null from
  device_put — the second call to the same cached jit fails with a layout mismatch.

  Rebuilding via make_array_from_single_device_arrays is metadata-only for CPU arrays:
  np.asarray(shard.data) accesses existing CPU memory with no copy, and
  device_put(numpy, SingleDeviceSharding(cpu)) creates a fresh null-layout IFRT buffer.
  """

  def normalize_leaf(x):
    if not isinstance(x, jax.Array) or not hasattr(x, "addressable_shards"):
      return x
    local_arrays = [
        jax.device_put(np.asarray(shard.data), jax.sharding.SingleDeviceSharding(shard.device))
        for shard in x.addressable_shards
    ]
    return jax.make_array_from_single_device_arrays(x.shape, x.sharding, local_arrays)

  return jax.tree_util.tree_map(normalize_leaf, tree)


class SyncerState(struct.PyTreeNode):
  params: Any
  opt_state: optax.OptState
  step: int


def get_first_step(model, state):
  if isinstance(model, nn.Module):
    return int(state.step)
  return int(state.optimizer.step.get_value())


def make_learner_config(config, learner_idx, num_learners):
  """Creates a modified deep copy of the global configuration for a specific learner."""
  learner_config = copy.deepcopy(config)

  # Remove 'diloco' from mesh_axes
  mesh_axes = list(learner_config.mesh_axes)
  if "diloco" in mesh_axes:
    mesh_axes.remove("diloco")
  learner_config._flat_config["mesh_axes"] = mesh_axes

  # Adjust logical_axis_rules to remove 'diloco'
  new_logical_axis_rules = []
  for logical_axis, physical_axes in learner_config.logical_axis_rules:
    if isinstance(physical_axes, str):
      if physical_axes == "diloco":
        continue
    elif isinstance(physical_axes, (list, tuple)):
      physical_axes = [ax for ax in physical_axes if ax != "diloco"]
    new_logical_axis_rules.append((logical_axis, physical_axes))
  learner_config._flat_config["logical_axis_rules"] = new_logical_axis_rules

  # Enable local data loading for each learner
  learner_config._flat_config["enable_local_data_loading"] = True
  learner_config._flat_config["learner_idx"] = learner_idx
  learner_config._flat_config["num_learners"] = num_learners

  # Disable SPMD diloco for learners
  learner_config._flat_config["enable_streaming_diloco"] = False
  learner_config._flat_config["enable_diloco"] = False

  return learner_config


def get_abstract_syncer_state(config, local_cpu_mesh):
  """Computes abstract state shapes and types for the syncer's parameters and optimizer."""

  if config.pure_nnx:
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      _, abstract_model = model_creation_utils.create_nnx_abstract_model(config, local_cpu_mesh)
      abstract_params = nnx.state(abstract_model, nnx.Param)
      outer_optimizer = optax.sgd(
          learning_rate=config.diloco_outer_lr,
          momentum=config.diloco_outer_momentum,
          nesterov=True,
      )

      @jax.jit
      def init_opt(p):
        return outer_optimizer.init(p)

      abstract_opt_state = init_opt.eval_shape(abstract_params)
  else:
    model = model_creation_utils.from_config(config, mesh=local_cpu_mesh)
    abstract_vars = maxtext_utils.get_abstract_param(model, config)
    abstract_params = abstract_vars["params"]

    params_logical_annotations = nn.get_partition_spec(abstract_params)
    params_mesh_shardings = nn.logical_to_mesh_sharding(
        params_logical_annotations, local_cpu_mesh, config.logical_axis_rules
    )

    @jax.jit
    def dummy_init():
      return abstract_params

    abstract_params = jax.jit(dummy_init, out_shardings=params_mesh_shardings).eval_shape()

    outer_optimizer = optax.sgd(
        learning_rate=config.diloco_outer_lr,
        momentum=config.diloco_outer_momentum,
        nesterov=True,
    )

    @jax.jit
    def init_opt(p):
      return outer_optimizer.init(p)

    opt_state_shardings = (optax.TraceState(trace=params_mesh_shardings), optax.EmptyState())
    abstract_opt_state = jax.jit(init_opt, out_shardings=opt_state_shardings).eval_shape(abstract_params)

  return abstract_params, abstract_opt_state


def _run_learner_loop(
    learner_idx, config, submesh, local_cpu_mesh, transport, recorder, train_step, eval_step, init_lock
):
  """Runs the main training and communication loop for a single learner replica."""
  max_logging.log(f"Learner {learner_idx}: Starting loop")
  learner_config = make_learner_config(config, learner_idx, config.num_diloco_replicas)
  learner_config._flat_config["run_name"] = config.run_name + f"_learner_{learner_idx}"

  with jax.set_mesh(submesh), submesh, nn_partitioning.axis_rules(learner_config.logical_axis_rules):
    learner_config._flat_config["checkpoint_dir"] = config.checkpoint_dir + f"/learner_{learner_idx}"

    max_logging.log(f"Learner {learner_idx}: setup_train_loop starting")
    (
        init_rng,
        checkpoint_manager,
        state_mesh_shardings,
        model,
        mesh,
        learning_rate_schedule,
        data_iterator,
        data_loader,
        rampup_manager,
        eval_data_iterator,
        state,
    ) = train_utils.setup_train_loop(learner_config, recorder, mesh=submesh)
    max_logging.log(f"Learner {learner_idx}: setup_train_loop done")

    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
        learner_config, state_mesh_shardings
    )
    flat_params_shardings, _ = jax.tree_util.tree_flatten_with_path(params_shardings)
    flat_params_shardings = {jax.tree_util.keystr(p): leaf for p, leaf in flat_params_shardings}

    if isinstance(model, nn.Module):
      jit_model = model
    else:
      jit_model, state = nnx.split(state)

    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        learner_config,
        jit_model,
        mesh,
        state,
        state_mesh_shardings,
        train_step,
        eval_step,
        eval_data_iterator,
        params_shardings,
    )

    start_step = get_first_step(model, state)

    # Synchronized Initialization / Resume
    try:
      if start_step == 0:
        if learner_idx == 0:
          params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
          max_logging.log(f"Learner {learner_idx}: sending init params")
          transport.send_to_syncer(step=0, fragment_id=-1, data=params)
          max_logging.log(f"Learner {learner_idx}: waiting for init params")
          initial_params = transport.recv_from_syncer(step=0, fragment_id=-1)
          max_logging.log(f"Learner {learner_idx}: received init params")
        else:
          max_logging.log(f"Learner {learner_idx}: waiting for init params")
          initial_params = transport.recv_from_syncer(step=0, fragment_id=-1)
          max_logging.log(f"Learner {learner_idx}: received init params")

        tpu_param_sharding = jax.tree_util.tree_map(
            lambda s: jax.sharding.NamedSharding(submesh, s.spec), params_shardings
        )
        initial_params_tpu = jax.device_put(initial_params, tpu_param_sharding)
        if learner_config.pure_nnx:
          non_param_model = nnx.filter_state(state.model, nnx.Not(nnx.Param))
          new_model = nnx.merge_state(non_param_model, initial_params_tpu)
          new_state = type(state)({})
          new_state["model"] = new_model
          new_state["optimizer"] = state["optimizer"]
          state = new_state
        else:
          state = state.replace(params=initial_params_tpu)
      else:
        global_params = transport.recv_from_syncer(step=start_step, fragment_id=-1)
        tpu_param_sharding = jax.tree_util.tree_map(
            lambda s: jax.sharding.NamedSharding(submesh, s.spec), params_shardings
        )
        global_params_tpu = jax.device_put(global_params, tpu_param_sharding)
        if learner_config.pure_nnx:
          non_param_model = nnx.filter_state(state.model, nnx.Not(nnx.Param))
          new_model = nnx.merge_state(non_param_model, global_params_tpu)
          new_state = type(state)({})
          new_state["model"] = new_model
          new_state["optimizer"] = state["optimizer"]
          state = new_state
        else:
          state = state.replace(params=global_params_tpu)
    except Exception as e:
      max_logging.error(f"Learner {learner_idx} crashed in init: {e}")
      max_logging.error(traceback.format_exc())
      raise e

    params_template = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
    manipulator = FragmentedTreeManipulator.create(params_template, learner_config)
    num_fragments = manipulator.num_fragments

    tau = learner_config.num_communication_overlapping_steps
    alpha = learner_config.communication_overlapping_alpha

    steps_between_syncs_plus_1 = int(round(learner_config.diloco_sync_period / num_fragments))
    steps_between_syncs_plus_1 = max(1, steps_between_syncs_plus_1)
    period = num_fragments * steps_between_syncs_plus_1

    prof = profiler.Profiler(learner_config, offset_step=start_step)
    metric_logger_instance = metric_logger.MetricLogger(
        config=learner_config, learning_rate_schedule=learning_rate_schedule
    )
    metric_logger_instance.write_setup_info_to_tensorboard(params_template)

    # Pre-compile the mix function for each fragment to avoid concurrent compilation crashes
    with init_lock:
      for f_idx in range(num_fragments):
        dummy_frag = manipulator.get_flat_fragment(params_template, f_idx)
        _ = mix_frags(dummy_frag, dummy_frag, alpha)

    try:
      last_step_completion = datetime.datetime.now()
      for step in range(start_step, learner_config.steps):
        max_logging.log(f"Learner {learner_idx}: Step {step} starting")
        prof.maybe_activate_profiler(step, state)

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
          if isinstance(model, nn.Module):
            step_rng_args = (jax.jit(jax.random.fold_in)(init_rng, step),)
          else:
            step_rng_args = ()

          with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
            with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
              if learner_config.shard_optimizer_over_data and isinstance(model, nn.Module):
                state = sharding.maybe_shard_with_name(state, state_mesh_shardings, learner_config.shard_mode)
              state, metrics = p_train_step(state, example_batch, *step_rng_args)
              # Force block to catch async errors immediately
              for leaf in jax.tree_util.tree_flatten((state, metrics))[0]:
                if hasattr(leaf, "block_until_ready"):
                  leaf.block_until_ready()

          max_logging.log(f"Learner {learner_idx}: Step {step} finished")
          step_time_delta = datetime.datetime.now() - last_step_completion

          completed_step = step + 1

          if completed_step > 0 and completed_step % steps_between_syncs_plus_1 == 0:
            frag_idx = (completed_step % period) // steps_between_syncs_plus_1
            params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
            frag_data = manipulator.get_flat_fragment(params, frag_idx)
            transport.send_to_syncer_async(completed_step, frag_idx, frag_data)

          if completed_step - tau > 0 and (completed_step - tau) % steps_between_syncs_plus_1 == 0:
            frag_idx = ((completed_step - tau) % period) // steps_between_syncs_plus_1
            received_frag = transport.recv_from_syncer(completed_step - tau, frag_idx)

            tpu_frag_sharding = {
                k: jax.sharding.NamedSharding(submesh, flat_params_shardings[k].spec) for k in received_frag.keys()
            }
            received_frag_tpu = jax.device_put(received_frag, tpu_frag_sharding)

            params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
            inner_frag = manipulator.get_flat_fragment(params, frag_idx)

            mixed_frag = mix_frags(inner_frag, received_frag_tpu, alpha)
            new_params = manipulator.apply_flat_fragment(params, frag_idx, mixed_frag)

            if learner_config.pure_nnx:
              non_param_model = nnx.filter_state(state.model, nnx.Not(nnx.Param))
              new_model = nnx.merge_state(non_param_model, new_params)
              new_state = type(state)({})
              new_state["model"] = new_model
              new_state["optimizer"] = state["optimizer"]
              state = new_state
            else:
              state = state.replace(params=new_params)

          checkpointing.maybe_save_checkpoint(checkpoint_manager, state, learner_config, data_iterator, step)

          metric_logger_instance.buffer_and_write_metrics(metrics, step, step_time_delta)

          eval_step_count = None
          if learner_config.eval_interval > 0 and step > start_step and (step + 1) % learner_config.eval_interval == 0:
            assert eval_data_iterator
            eval_data_iterator.reset()
            metric_logger_instance.reset_eval_metrics()
            max_logging.log(f"Learner {learner_idx}: Starting eval after train step {step}")

            eval_step_count = 0
            last_eval_step_completion = datetime.datetime.now()
            for eval_batch in eval_data_iterator:
              eval_batch = jax.device_put(eval_batch, sharding.get_input_data_sharding(learner_config, mesh))
              if learner_config.eval_steps > 0 and eval_step_count >= learner_config.eval_steps:
                break
              with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
                eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
              eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
              last_eval_step_completion = datetime.datetime.now()
              metric_logger_instance.buffer_and_write_metrics(
                  eval_metrics, eval_step_count, step_time_delta=eval_step_time_delta, is_training=False
              )
              eval_step_count += 1

          prof.maybe_deactivate_profiler(step, state)
          last_step_completion = datetime.datetime.now()

      if checkpoint_manager is not None:
        checkpoint_manager.wait_until_finished()
      if learner_config.save_checkpoint_on_completion:
        checkpointing.maybe_save_checkpoint(checkpoint_manager, state, learner_config, data_iterator, step=step)

    except exceptions.StopTraining as e:
      prof.deactivate()
      max_logging.log(f"Learner {learner_idx} training stopped: {str(e)}")
    finally:
      metric_logger_instance.flush_metrics_and_cleanup()
      transport.close()


def learner_loop(learner_idx, config, submesh, local_cpu_mesh, transport, recorder, train_step, eval_step, init_lock):
  """Wrapper to run the learner loop and handle/log top-level exceptions."""
  try:
    _run_learner_loop(learner_idx, config, submesh, local_cpu_mesh, transport, recorder, train_step, eval_step, init_lock)
  except Exception as e:
    max_logging.error(f"Learner {learner_idx} crashed: {e}")
    max_logging.error(traceback.format_exc())
    raise e


def syncer_loop(
    config, global_cpu_mesh, cpu_submeshes, transport, recorder, abstract_params=None, abstract_opt_state=None
):
  """Wrapper to run the syncer loop and handle/log top-level exceptions."""
  try:
    _run_syncer_loop(config, global_cpu_mesh, cpu_submeshes, transport, recorder, abstract_params, abstract_opt_state)
  except Exception as e:
    max_logging.error(f"Syncer crashed: {e}")
    max_logging.error(traceback.format_exc())
    raise e


def make_step_fns(global_cpu_mesh, flat_params_shardings, frag_keys, trace_keys, outer_optimizer):
  """Creates JIT-compiled functions for computing gradients and applying outer steps."""
  global_sharding_tree = {
      k: jax.sharding.NamedSharding(global_cpu_mesh, flat_params_shardings[k].spec) for k in frag_keys
  }
  stacked_sharding_tree = {
      k: jax.sharding.NamedSharding(global_cpu_mesh, jax.sharding.PartitionSpec("diloco", *flat_params_shardings[k].spec))
      for k in frag_keys
  }
  opt_state_sharding_tree = (optax.TraceState(trace=global_sharding_tree), optax.EmptyState())

  # Use plain NamedSharding (no Format/layout constraint) for all in_shardings.
  # Syncer params come from learner transport with tiled layout and remain tiled throughout
  # (device_put to NamedSharding is a no-op in Pathways when sharding matches).
  # Constraining in_shardings to null-layout Format causes mismatches when tiled arrays
  # are passed. Plain NamedSharding lets JAX accept whatever layout the actual inputs have.
  @functools.partial(
      jax.jit,
      in_shardings=(global_sharding_tree, stacked_sharding_tree),
      out_shardings=global_sharding_tree,
  )
  def compute_grad(o_frag, stacked_i_frag):
    averaged_i_frag = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), stacked_i_frag)
    return jax.tree_util.tree_map(lambda x, y: x - y, o_frag, averaged_i_frag)

  @functools.partial(
      jax.jit,
      in_shardings=(global_sharding_tree, opt_state_sharding_tree, global_sharding_tree),
      out_shardings=(global_sharding_tree, opt_state_sharding_tree),
  )
  def apply_outer_step(g_frag, o_state_frag, p_frag):
    updates_frag, new_o_state_frag = outer_optimizer.update(g_frag, o_state_frag, params=p_frag)
    new_p_frag = optax.apply_updates(p_frag, updates_frag)
    return new_p_frag, new_o_state_frag

  return compute_grad, apply_outer_step


def _run_syncer_loop(
    config, global_cpu_mesh, cpu_submeshes, transport, recorder, abstract_params=None, abstract_opt_state=None
):
  """Runs the main syncer loop that coordinates parameter averaging and outer optimization."""
  max_logging.log("Syncer: Starting loop")

  num_learners = config.num_diloco_replicas

  if abstract_params is None or abstract_opt_state is None:
    abstract_params, abstract_opt_state = get_abstract_syncer_state(config, global_cpu_mesh)
  abstract_step = jax.ShapeDtypeStruct(
      (), jnp.int32, sharding=jax.sharding.NamedSharding(global_cpu_mesh, jax.sharding.PartitionSpec())
  )
  abstract_syncer_state = SyncerState(params=abstract_params, opt_state=abstract_opt_state, step=abstract_step)

  # Init(1): Loading checkpoints
  logger = checkpointing.setup_checkpoint_logger(config)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.checkpoint_period,
      config.dataset_type,
      logger,
      config.checkpoint_storage_use_ocdbt,
      config.checkpoint_storage_use_zarr3,
      config.enable_continuous_checkpointing,
      config.max_num_checkpoints_to_keep,
      config.checkpoint_storage_concurrent_gb,
      config.enable_single_controller,
      config.colocated_python_checkpointing,
      config.enable_single_replica_ckpt_restoring,
      config.enable_autocheckpoint,
      config.checkpoint_todelete_subdir,
      config.checkpoint_todelete_full_path,
  )

  restored_state, _ = checkpointing.load_state_if_possible(
      checkpoint_manager=checkpoint_manager,
      data_iterator=None,
      load_parameters_from_path="",
      load_full_state_from_path="",
      checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
      abstract_unboxed_pre_state=abstract_syncer_state,
      enable_single_replica_ckpt_restoring=config.enable_single_replica_ckpt_restoring,
      dataset_type=config.dataset_type,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )

  # Get abstract shardings for params and opt_state
  params_shardings = jax.tree_util.tree_map(lambda x: x.sharding, abstract_params)
  flat_params_shardings = {
      jax.tree_util.keystr(k): v for k, v in jax.tree_util.tree_flatten_with_path(params_shardings)[0]
  }

  if restored_state is None:  # (1,a) No checkpoint found, start from scratch
    max_logging.log("Syncer: waiting for init params from Learner 0")
    initial_params_l0 = transport.recv_from_learner(learner_idx=0, step=0, fragment_id=-1)
    max_logging.log("Syncer: received init params from Learner 0")
    with jax.set_mesh(global_cpu_mesh):
      global_params = jax.device_put(initial_params_l0, params_shardings)
      # Normalize to null layout: learner transport delivers tiled params, and different
      # tensors may have mixed layouts. Consistent null layout prevents jit__take /
      # jit__scatter from seeing layout mismatches across params with the same signature.
      global_params = _normalize_to_null_layout(global_params)
      outer_optimizer = optax.sgd(
          learning_rate=config.diloco_outer_lr,
          momentum=config.diloco_outer_momentum,
          nesterov=True,
      )
      outer_opt_state = outer_optimizer.init(global_params)
      outer_opt_state = _normalize_to_null_layout(outer_opt_state)

    syncer_state = SyncerState(params=global_params, opt_state=outer_opt_state, step=0)
    start_step = 0

  else:  # loading checkpoints successfully
    syncer_state = restored_state["items"]
    start_step = int(syncer_state.step)
    max_logging.log(f"Syncer restored from step {start_step}")

  # Init (2): send initial params to each learner directly.
  # Params on global_cpu_mesh have no diloco axis in their spec, so they are replicated
  # across all diloco slices. Rebinding to each cpu_submesh is a metadata-only operation
  # that avoids the jit-level layout checking that caused Pathways layout mismatches.
  for i, submesh in enumerate(cpu_submeshes):
    local_sharding = jax.tree_util.tree_map(
        lambda s, submesh=submesh: jax.sharding.NamedSharding(submesh, s.spec),
        params_shardings,
    )
    local_params = jax.device_put(syncer_state.params, local_sharding)
    max_logging.log(f"Syncer: sending params to Learner {i} at step {start_step}")
    transport.send_to_learner(learner_idx=i, step=start_step, fragment_id=-1, data=local_params)
    max_logging.log(f"Syncer: sent params to Learner {i} at step {start_step}")

  manipulator = FragmentedTreeManipulator.create(syncer_state.params, config)
  num_fragments = manipulator.num_fragments

  steps_between_syncs_plus_1 = int(round(config.diloco_sync_period / num_fragments))
  steps_between_syncs_plus_1 = max(1, steps_between_syncs_plus_1)
  period = num_fragments * steps_between_syncs_plus_1

  outer_optimizer = optax.sgd(
      learning_rate=config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )

  # steps that syncing is happening
  sync_steps = [step for step in range(start_step + 1, config.steps + 1) if step % steps_between_syncs_plus_1 == 0]

  # Sharding for the full params tree on the global CPU mesh (used for null-layout resets).
  params_full_sharding = jax.tree_util.tree_map(
      lambda s: jax.sharding.NamedSharding(global_cpu_mesh, s.spec), params_shardings
  )

  # Pre-build JIT step functions for each fragment index.
  # Fragment 0 (non-scanned params): call get_flat_fragment directly — no jnp.take, safe.
  # Fragment >0 (scanned layer slices): avoid eager jnp.take by computing abstract shapes from
  # the full scanned arrays. ShapeDtypeStruct has .shape so null_format rank derivation works.
  step_fns_by_frag = {}
  with jax.set_mesh(global_cpu_mesh):
    for f_idx in range(num_fragments):
      if f_idx == 0:
        frag_dict = manipulator.get_flat_fragment(syncer_state.params, f_idx)
        trace_dict = manipulator.get_flat_fragment(syncer_state.opt_state[0].trace, f_idx)
      else:
        indices = manipulator.fragment_to_layer_indices[f_idx]
        num_layer_indices = len(indices)
        frag_dict = {}
        trace_dict = {}
        for keystr, v in [
            (jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_flatten_with_path(syncer_state.params)[0]
        ]:
          if manipulator.keypath_to_is_scanned.get(keystr, False):
            frag_shape = (num_layer_indices,) + v.shape[1:]
            frag_dict[keystr] = jax.ShapeDtypeStruct(frag_shape, v.dtype)
        for keystr, v in [
            (jax.tree_util.keystr(k), v)
            for k, v in jax.tree_util.tree_flatten_with_path(syncer_state.opt_state[0].trace)[0]
        ]:
          if manipulator.keypath_to_is_scanned.get(keystr, False):
            frag_shape = (num_layer_indices,) + v.shape[1:]
            trace_dict[keystr] = jax.ShapeDtypeStruct(frag_shape, v.dtype)
      step_fns_by_frag[f_idx] = make_step_fns(
          global_cpu_mesh, flat_params_shardings, frag_dict, trace_dict, outer_optimizer
      )

  # Start main syncer loop
  for step in sync_steps:  # e.g. 50, 100, 150... if sync_period=50
    max_logging.log(f"Syncer: Step {step} sync starting")
    frag_idx = (step % period) // steps_between_syncs_plus_1

    learner_frags = []

    # receive the fragment of the current step from each learner.
    for i in range(num_learners):
      frag_i = transport.recv_from_learner(learner_idx=i, step=step, fragment_id=frag_idx)
      learner_frags.append(frag_i)
    max_logging.log(f"Syncer: received all fragments for step {step}")

    stacked_inner_frag = stack_across_meshes_pytree(learner_frags, global_cpu_mesh, "diloco")
    # Normalize stacked fragment: concatenate_by_mesh_axis output layout is non-deterministic.
    stacked_inner_frag = _normalize_to_null_layout(stacked_inner_frag)
    max_logging.log(f"Syncer: Step {step} stacking done")

    with jax.set_mesh(global_cpu_mesh):
      # use_null_layout_jit=True avoids the eager jnp.take Pathways rejects for scanned
      # fragments; normalize afterwards since concatenate/jit outputs can still vary.
      outer_params_frag = _normalize_to_null_layout(
          manipulator.get_flat_fragment(syncer_state.params, frag_idx, use_null_layout_jit=True)
      )
      trace_frag = _normalize_to_null_layout(
          manipulator.get_flat_fragment(syncer_state.opt_state[0].trace, frag_idx, use_null_layout_jit=True)
      )
      opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

      compute_grad, apply_outer_step = step_fns_by_frag[frag_idx]

      pseudo_grad_frag = _normalize_to_null_layout(compute_grad(outer_params_frag, stacked_inner_frag))

      new_outer_params_frag, new_opt_state_frag = apply_outer_step(pseudo_grad_frag, opt_state_frag, outer_params_frag)
      # Normalize jit outputs before scatter so v.at[].set(frag) always sees null-layout frag.
      new_outer_params_frag = _normalize_to_null_layout(new_outer_params_frag)
      new_opt_state_trace = _normalize_to_null_layout(new_opt_state_frag[0].trace)

      new_params = manipulator.apply_flat_fragment(
          syncer_state.params, frag_idx, new_outer_params_frag, use_null_layout_jit=True
      )
      new_params = _normalize_to_null_layout(jax.device_put(new_params, params_full_sharding))

      new_trace = manipulator.apply_flat_fragment(
          syncer_state.opt_state[0].trace, frag_idx, new_opt_state_trace, use_null_layout_jit=True
      )
      new_trace = _normalize_to_null_layout(jax.device_put(new_trace, params_full_sharding))

      new_opt_state = (optax.TraceState(trace=new_trace), syncer_state.opt_state[1])

      syncer_state = syncer_state.replace(params=new_params, opt_state=new_opt_state, step=step)
    max_logging.log(f"Syncer: Step {step} outer step applied")

    # Send updated fragment directly to each learner's submesh via device_put.
    # new_outer_params_frag has no diloco axis in its sharding (replicated across slices),
    # so rebinding to each cpu_submesh is a metadata-only operation with no layout checking.
    for i, submesh in enumerate(cpu_submeshes):
      frag_local_sharding = {
          k: jax.sharding.NamedSharding(submesh, flat_params_shardings[k].spec) for k in new_outer_params_frag
      }
      local_frag = jax.device_put(new_outer_params_frag, frag_local_sharding)
      transport.send_to_learner(learner_idx=i, step=step, fragment_id=frag_idx, data=local_frag)

    # SyncerState is a plain PyTreeNode, not a NNX TrainState — force the Linen save path.
    syncer_ckpt_config = copy.copy(config)
    syncer_ckpt_config._flat_config["pure_nnx"] = False
    checkpointing.maybe_save_checkpoint(
        checkpoint_manager=checkpoint_manager,
        state=syncer_state,
        config=syncer_ckpt_config,
        data_iterator=None,
        step=step,
    )
    max_logging.log(f"Syncer: Step {step} sync finished")

  if checkpoint_manager is not None:
    checkpoint_manager.wait_until_finished()


def run_threaded_diloco(config, recorder, train_step, eval_step):
  """Orchestrator for multi-threaded DiLoCo."""
  max_logging.log("Starting run_threaded_diloco")
  num_learners = config.num_diloco_replicas

  max_logging.log("Creating global mesh")
  global_mesh = maxtext_utils.get_mesh_from_config(config)
  max_logging.log("Partitioning global mesh")
  tpu_submeshes = partition_mesh_by_diloco_axis(global_mesh, num_learners)
  cpu_submeshes = [colocated_python.colocated_cpu_devices(submesh) for submesh in tpu_submeshes]
  global_cpu_mesh = colocated_python.colocated_cpu_devices(global_mesh)

  transport_manager = ThreadedTransportManager(num_learners)

  # Get abstract syncer state first, on main thread, before spawning learner threads.
  max_logging.log("Getting abstract syncer state")
  abstract_params, abstract_opt_state = get_abstract_syncer_state(config, global_cpu_mesh)
  max_logging.log("Got abstract syncer state")

  init_lock = threading.Lock()

  max_logging.log("Spawning learner threads")
  with ThreadPoolExecutor(max_workers=num_learners) as executor:
    futures = []
    for i in range(num_learners):
      # Determine if this learner should run on this process
      learner_devices = tpu_submeshes[i].devices.flat
      should_run = any(d in jax.local_devices() for d in learner_devices)

      if should_run:
        learner_transport = LearnerTransport(transport_manager, i, cpu_submeshes[i])

        futures.append(
            executor.submit(
                learner_loop,
                i,
                config,
                tpu_submeshes[i],  # each learner will only see its local TPU submesh
                cpu_submeshes[i],
                learner_transport,
                recorder,
                train_step,
                eval_step,
                init_lock=init_lock,
            )
        )
      else:
        max_logging.log(f"Learner {i} is remote, not spawning thread")

    syncer_transport = SyncerTransport(transport_manager)
    max_logging.log("Starting syncer loop")
    syncer_loop(
        config,
        global_cpu_mesh,
        cpu_submeshes,
        syncer_transport,
        recorder,
        abstract_params=abstract_params,
        abstract_opt_state=abstract_opt_state,
    )

    max_logging.log("Waiting for learner threads to finish")
    for f in futures:
      f.result()
    max_logging.log("Finished run_threaded_diloco")
