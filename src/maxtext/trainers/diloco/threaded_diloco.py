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
import gc
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flax import linen as nn, nnx, struct
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.experimental import colocated_python
import optax
from pathwaysutils.experimental import reshard as pathways_reshard

from maxtext.common import checkpointing, profiler, metric_logger
from maxtext.common.goodput import maybe_record_goodput, GoodputEvent
from maxtext.trainers.diloco.decomposed_transport import (
    LearnerTransport,
    SyncerTransport,
    ThreadedTransportManager,
    TransportClosedError,
)
from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator
from maxtext.utils import exceptions
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from maxtext.utils.mesh_utils import jit_with_layout_canonicalized_inputs, partition_mesh_by_diloco_axis


INITIAL_PARAMS_FRAGMENT_ID = -1
INITIAL_PARAMS_ACK_FRAGMENT_ID = -2
START_STEP_FRAGMENT_ID = -3


@jax.jit
def mix_frags(i_frag, o_frag, alpha):
  return jax.tree_util.tree_map(lambda x, y: alpha * x + (1 - alpha) * y, i_frag, o_frag)


# pylint: disable=abstract-method
class SyncerState(struct.PyTreeNode):
  params: Any
  opt_state: optax.OptState
  step: int


def _shardings_on_mesh(sharding_tree, mesh):
  """Rebinds logical shardings to ``mesh`` while preserving specs and memory kinds."""
  return jax.tree_util.tree_map(
      lambda s: jax.sharding.NamedSharding(mesh, s.spec, memory_kind=s.memory_kind),
      sharding_tree,
  )


def _flat_fragment_shardings(mesh, flat_params_shardings, fragment):
  """Builds the target sharding tree for a flat parameter fragment."""
  return {
      key: jax.sharding.NamedSharding(
          mesh,
          flat_params_shardings[key].spec,
          memory_kind=flat_params_shardings[key].memory_kind,
      )
      for key in fragment
  }


def _reshard_tree(tree, target_shardings, *, donate):
  """Moves a tree between disjoint Pathways meshes without controller-host staging."""
  tree_leaves = jax.tree_util.tree_leaves(tree)
  target_leaves = jax.tree_util.tree_leaves(target_shardings)
  is_identity_reshard = (
      jax.tree_util.tree_structure(tree) == jax.tree_util.tree_structure(target_shardings)
      and len(tree_leaves) == len(target_leaves)
      and all(isinstance(value, jax.Array) and value.sharding == target for value, target in zip(tree_leaves, target_leaves))
  )
  if is_identity_reshard and donate:
    # The experimental API does not fast-path an identity reshard. Adopting the
    # tree directly avoids a plugin plan for learner 0 (the coordinator slice).
    # ``donate=True`` remains a caller ownership promise: the next reduction
    # consumes these arrays with a donated JIT.
    return tree
  if is_identity_reshard:
    # An outgoing transport payload must not alias persistent coordinator
    # state. In particular, fragment 0 is inserted directly into that state and
    # can otherwise remain queued until its next donated update.
    return jax.device_put(tree, target_shardings, may_alias=False)
  return pathways_reshard.reshard(
      tree,
      target_shardings,
      donate=donate,
      may_alias=None,
      cache_resharding_plans=True,
  )


def make_streaming_mean_fns(fragment_example, fragment_shardings, num_learners):
  """Compiles donated fragment-sized reductions for a numerically safe mean."""
  if num_learners < 1:
    raise ValueError(f"num_learners must be positive, got {num_learners}")

  original_dtypes = jax.tree_util.tree_map(lambda x: jnp.dtype(x.dtype), fragment_example)
  accumulator_dtypes = jax.tree_util.tree_map(
      lambda dtype: jnp.float32 if jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < 4 else dtype,
      original_dtypes,
  )

  def initialize_sum(fragment):
    return jax.tree_util.tree_map(lambda value, dtype: value.astype(dtype), fragment, accumulator_dtypes)

  def add_to_sum(running_sum, fragment):
    return jax.tree_util.tree_map(
        lambda total, value: total + value.astype(total.dtype),
        running_sum,
        fragment,
    )

  def finish_mean(running_sum):
    return jax.tree_util.tree_map(
        lambda total, dtype: (total / jnp.asarray(num_learners, dtype=total.dtype)).astype(dtype),
        running_sum,
        original_dtypes,
    )

  initialize_sum = jit_with_layout_canonicalized_inputs(
      initialize_sum,
      in_shardings=(fragment_shardings,),
      out_shardings=fragment_shardings,
      donate_argnums=(0,),
  )
  add_to_sum = jit_with_layout_canonicalized_inputs(
      add_to_sum,
      in_shardings=(fragment_shardings, fragment_shardings),
      out_shardings=fragment_shardings,
      donate_argnums=(0, 1),
  )
  finish_mean = jit_with_layout_canonicalized_inputs(
      finish_mean,
      in_shardings=(fragment_shardings,),
      out_shardings=fragment_shardings,
      donate_argnums=(0,),
  )
  return initialize_sum, add_to_sum, finish_mean


def stream_learner_mean(  # pylint: disable=too-many-arguments
    transport,
    *,
    num_learners,
    step,
    fragment_id,
    target_shardings,
    reduction_fns,
    reshard_fn=_reshard_tree,
):
  """Receives, reshards, and reduces one learner fragment at a time.

  Blocking each donated accumulation before receiving the next fragment makes
  the live payload count independent of ``num_learners``.
  """
  initialize_sum, add_to_sum, finish_mean = reduction_fns
  running_sum = None
  for learner_idx in range(num_learners):
    learner_fragment = transport.recv_from_learner(
        learner_idx=learner_idx,
        step=step,
        fragment_id=fragment_id,
    )
    coordinator_fragment = reshard_fn(learner_fragment, target_shardings, donate=True)
    del learner_fragment

    if running_sum is None:
      running_sum = initialize_sum(coordinator_fragment)
    else:
      running_sum = add_to_sum(running_sum, coordinator_fragment)
    running_sum = jax.block_until_ready(running_sum)
    del coordinator_fragment

  averaged_fragment = finish_mean(running_sum)
  averaged_fragment = jax.block_until_ready(averaged_fragment)
  del running_sum
  return averaged_fragment


def _save_checkpoint_serialized(  # pylint: disable=too-many-arguments
    checkpoint_manager,
    state,
    config,
    data_iterator,
    checkpoint_lock,
    *,
    step,
    skip_step_zero=False,
    force=False,
):
  """Prevents asynchronous full-state snapshots from overlapping donated updates."""
  if checkpoint_manager is None or (skip_step_zero and step == 0):
    return
  with checkpoint_lock:
    save_kwargs = {
        "checkpoint_manager": checkpoint_manager,
        "state": state,
        "config": config,
        "data_iterator": data_iterator,
        "step": step,
    }
    if force:
      save_kwargs["force"] = True
    checkpointing.maybe_save_checkpoint(**save_kwargs)
    checkpoint_manager.wait_until_finished()


def get_first_step(model, state):
  if isinstance(model, nn.Module):
    return int(state.step)
  return int(state.optimizer.step.get_value())


def _delayed_response_step(completed_step, overlap_steps, sync_interval, start_step):
  """Returns the response step due now, excluding messages from before resume."""
  delayed_step = completed_step - overlap_steps
  if delayed_step <= start_step or delayed_step % sync_interval != 0:
    return None
  return delayed_step


def _validate_checkpoint_alignment(config):
  """Rejects checkpoint schedules that cannot produce aligned DiLoCo states."""
  if not config.enable_checkpointing:
    return
  unsupported_modes = [
      name
      for name in (
          "enable_continuous_checkpointing",
          "enable_autocheckpoint",
          "enable_emergency_checkpoint",
          "enable_multi_tier_checkpointing",
      )
      if getattr(config, name, False)
  ]
  if unsupported_modes:
    raise ValueError(
        "Non-SPMD streaming DiLoCo requires fixed, coordinated checkpoint "
        f"boundaries; unsupported modes enabled: {unsupported_modes}"
    )
  sync_interval = max(1, int(round(config.diloco_sync_period / (config.num_diloco_fragments + 1))))
  if config.checkpoint_period % sync_interval != 0:
    raise ValueError(
        "Non-SPMD streaming DiLoCo requires checkpoint_period to be divisible "
        f"by the fragment sync interval ({sync_interval}); got {config.checkpoint_period}"
    )
  if config.save_checkpoint_on_completion and config.steps % sync_interval != 0:
    raise ValueError(
        "Non-SPMD streaming DiLoCo can save an aligned completion checkpoint "
        f"only on a fragment sync step (interval {sync_interval}); got steps={config.steps}"
    )


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

      params_mesh_shardings = jax.tree_util.tree_map(lambda x: x.sharding, abstract_params)
      opt_state_shardings = (
          optax.TraceState(trace=params_mesh_shardings),
          optax.EmptyState(),
      )
      abstract_opt_state = jax.jit(init_opt, out_shardings=opt_state_shardings).eval_shape(abstract_params)
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

    opt_state_shardings = (
        optax.TraceState(trace=params_mesh_shardings),
        optax.EmptyState(),
    )
    abstract_opt_state = jax.jit(init_opt, out_shardings=opt_state_shardings).eval_shape(abstract_params)

  return abstract_params, abstract_opt_state


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def _run_learner_loop(
    learner_idx,
    config,
    submesh,
    local_cpu_mesh,
    transport,
    recorder,
    train_step,
    eval_step,
    init_lock,
    checkpoint_lock,
):
  """Runs the main training and communication loop for a single learner replica."""
  max_logging.log(f"Learner {learner_idx}: Starting loop")
  learner_config = make_learner_config(config, learner_idx, config.num_diloco_replicas)
  learner_config._flat_config["run_name"] = config.run_name + f"_learner_{learner_idx}"

  with (
      jax.set_mesh(submesh),
      submesh,
      nn_partitioning.axis_rules(learner_config.logical_axis_rules),
  ):
    learner_config._flat_config["checkpoint_dir"] = config.checkpoint_dir + f"/learner_{learner_idx}"

    # Model/optimizer initialization can transiently materialize large host
    # trees. Serialize it across learner threads; the first train compilations
    # are serialized separately below after every learner is initialized.
    with init_lock:
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
    loop_exception = None
    try:
      # Checkpoint managers are independent. Advertise the restored learner
      # step before exchanging model-sized payloads so a partial checkpoint set
      # fails explicitly instead of leaving fresh/restored peers waiting on
      # incompatible initialization branches.
      transport.send_control_to_syncer(
          step=start_step,
          fragment_id=START_STEP_FRAGMENT_ID,
      )
      if start_step == 0:
        if learner_idx == 0:
          params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
          max_logging.log(f"Learner {learner_idx}: sending init params")
          transport.send_to_syncer(step=0, fragment_id=INITIAL_PARAMS_FRAGMENT_ID, data=params)
          del params
      else:
        max_logging.log(f"Learner {learner_idx}: waiting for restored global params")

      max_logging.log(f"Learner {learner_idx}: waiting for init params")
      initial_params = transport.recv_from_syncer(
          step=start_step,
          fragment_id=INITIAL_PARAMS_FRAGMENT_ID,
      )
      max_logging.log(f"Learner {learner_idx}: received init params")

      tpu_param_sharding = _shardings_on_mesh(params_shardings, submesh)
      initial_params_tpu = jax.device_put(
          initial_params,
          tpu_param_sharding,
          may_alias=False,
      )
      initial_params_tpu = jax.block_until_ready(initial_params_tpu)
      if learner_config.pure_nnx:
        non_param_model = nnx.filter_state(state.model, nnx.Not(nnx.Param))
        new_model = nnx.merge_state(non_param_model, initial_params_tpu)
        new_state = type(state)({})
        new_state["model"] = new_model
        new_state["optimizer"] = state["optimizer"]
        state = new_state
        del non_param_model, new_model, new_state
      else:
        state = state.replace(params=initial_params_tpu)

      # The syncer broadcasts full trees sequentially. Acknowledge only after
      # the TPU copy has completed and this learner has released its CPU tree.
      del initial_params, initial_params_tpu
      gc.collect()
      transport.send_control_to_syncer(
          step=start_step,
          fragment_id=INITIAL_PARAMS_ACK_FRAGMENT_ID,
      )
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
    metric_logger_instance = metric_logger.MetricLogger(config=learner_config, learning_rate_schedule=learning_rate_schedule)
    metric_logger_instance.write_setup_info_to_tensorboard(params_template)

    # Pre-compile the mix function for each fragment to avoid concurrent compilation crashes
    with init_lock:
      for f_idx in range(num_fragments):
        dummy_frag = manipulator.get_flat_fragment(params_template, f_idx)
        mixed_dummy = mix_frags(dummy_frag, dummy_frag, alpha)
        jax.block_until_ready(mixed_dummy)
        del dummy_frag, mixed_dummy
    del params_template

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
            with (
                jax.set_mesh(mesh),
                nn_partitioning.axis_rules(learner_config.logical_axis_rules),
            ):
              if learner_config.shard_optimizer_over_data and isinstance(model, nn.Module):
                state = sharding.maybe_shard_with_name(
                    state,
                    state_mesh_shardings,
                    learner_config.shard_mode,
                )
              # Serialize only the first expensive train-step compilation. All
              # later executions run concurrently.
              if step == start_step:
                with init_lock:
                  state, metrics = p_train_step(state, example_batch, *step_rng_args)
              else:
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
            del frag_data, params

          response_step = _delayed_response_step(
              completed_step,
              tau,
              steps_between_syncs_plus_1,
              start_step,
          )
          if response_step is not None:
            frag_idx = (response_step % period) // steps_between_syncs_plus_1
            received_frag = transport.recv_from_syncer(response_step, frag_idx)

            tpu_frag_sharding = _flat_fragment_shardings(submesh, flat_params_shardings, received_frag)
            received_frag_tpu = jax.device_put(received_frag, tpu_frag_sharding, may_alias=False)
            received_frag_tpu = jax.block_until_ready(received_frag_tpu)
            del received_frag

            params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
            inner_frag = manipulator.get_flat_fragment(params, frag_idx)

            mixed_frag = mix_frags(inner_frag, received_frag_tpu, alpha)
            mixed_frag = jax.block_until_ready(mixed_frag)
            del inner_frag, received_frag_tpu
            new_params = manipulator.apply_flat_fragment(params, frag_idx, mixed_frag)

            if learner_config.pure_nnx:
              non_param_model = nnx.filter_state(state.model, nnx.Not(nnx.Param))
              new_model = nnx.merge_state(non_param_model, new_params)
              new_state = type(state)({})
              new_state["model"] = new_model
              new_state["optimizer"] = state["optimizer"]
              state = new_state
              del non_param_model, new_model, new_state
            else:
              state = state.replace(params=new_params)
            del mixed_frag, new_params, params

          _save_checkpoint_serialized(
              checkpoint_manager,
              state,
              learner_config,
              data_iterator,
              checkpoint_lock,
              # State counters represent completed updates. Using the same
              # completed-step key as the syncer keeps restore start steps
              # aligned instead of saving a step+1 learner under key ``step``.
              step=completed_step,
              skip_step_zero=True,
          )

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
              eval_batch = jax.device_put(
                  eval_batch,
                  sharding.get_input_data_sharding(learner_config, mesh),
              )
              if learner_config.eval_steps > 0 and eval_step_count >= learner_config.eval_steps:
                break
              with (
                  jax.set_mesh(mesh),
                  nn_partitioning.axis_rules(learner_config.logical_axis_rules),
              ):
                eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
              eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
              last_eval_step_completion = datetime.datetime.now()
              metric_logger_instance.buffer_and_write_metrics(
                  eval_metrics,
                  eval_step_count,
                  step_time_delta=eval_step_time_delta,
                  is_training=False,
              )
              eval_step_count += 1

          prof.maybe_deactivate_profiler(step, state)
          last_step_completion = datetime.datetime.now()

      if learner_config.save_checkpoint_on_completion:
        _save_checkpoint_serialized(
            checkpoint_manager,
            state,
            learner_config,
            data_iterator,
            checkpoint_lock,
            step=get_first_step(model, state),
            force=True,
        )
      elif checkpoint_manager is not None:
        with checkpoint_lock:
          checkpoint_manager.wait_until_finished()

    except exceptions.StopTraining as e:
      loop_exception = e
      transport.manager.close()
      prof.deactivate()
      max_logging.log(f"Learner {learner_idx} training stopped: {str(e)}")
    except Exception as e:
      loop_exception = e
      transport.manager.close()
      raise
    finally:
      metric_logger_instance.flush_metrics_and_cleanup()
      try:
        transport.close()
      except Exception:  # pylint: disable=broad-exception-caught
        # A background offload closes the manager before the learner wakes.
        # Prefer that originating Future error over the resulting cancellation.
        if loop_exception is None or isinstance(loop_exception, TransportClosedError):
          raise


# pylint: disable=too-many-positional-arguments,too-many-arguments
def learner_loop(
    learner_idx,
    config,
    submesh,
    local_cpu_mesh,
    transport,
    recorder,
    train_step,
    eval_step,
    init_lock,
    checkpoint_lock,
):
  """Wrapper to run the learner loop and handle/log top-level exceptions."""
  try:
    _run_learner_loop(
        learner_idx,
        config,
        submesh,
        local_cpu_mesh,
        transport,
        recorder,
        train_step,
        eval_step,
        init_lock,
        checkpoint_lock,
    )
  except Exception as e:
    transport.manager.close()
    max_logging.error(f"Learner {learner_idx} crashed: {e}")
    max_logging.error(traceback.format_exc())
    raise e


# pylint: disable=too-many-positional-arguments,too-many-arguments
def syncer_loop(
    config,
    syncer_cpu_mesh,
    cpu_submeshes,
    transport,
    recorder,
    checkpoint_lock,
    abstract_params=None,
    abstract_opt_state=None,
):
  """Wrapper to run the syncer loop and handle/log top-level exceptions."""
  try:
    _run_syncer_loop(
        config,
        syncer_cpu_mesh,
        cpu_submeshes,
        transport,
        recorder,
        checkpoint_lock,
        abstract_params,
        abstract_opt_state,
    )
  except Exception as e:
    transport.close()
    max_logging.error(f"Syncer crashed: {e}")
    max_logging.error(traceback.format_exc())
    raise e


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def make_step_fns(syncer_cpu_mesh, flat_params_shardings, frag_keys, trace_keys, outer_optimizer):
  """Compiles layout-adapted, donated fragment-level outer optimizer steps."""
  fragment_shardings = _flat_fragment_shardings(syncer_cpu_mesh, flat_params_shardings, frag_keys)
  trace_shardings = _flat_fragment_shardings(syncer_cpu_mesh, flat_params_shardings, trace_keys)
  opt_state_shardings = (optax.TraceState(trace=trace_shardings), optax.EmptyState())

  def compute_grad(o_frag, averaged_i_frag):
    return jax.tree_util.tree_map(lambda x, y: x - y, o_frag, averaged_i_frag)

  def apply_outer_step(g_frag, o_state_frag, p_frag):
    updates_frag, new_o_state_frag = outer_optimizer.update(g_frag, o_state_frag, params=p_frag)
    new_p_frag = optax.apply_updates(p_frag, updates_frag)
    return new_p_frag, new_o_state_frag

  compute_grad = jit_with_layout_canonicalized_inputs(
      compute_grad,
      in_shardings=(fragment_shardings, fragment_shardings),
      out_shardings=fragment_shardings,
      donate_argnums=(1,),
  )
  apply_outer_step = jit_with_layout_canonicalized_inputs(
      apply_outer_step,
      in_shardings=(fragment_shardings, opt_state_shardings, fragment_shardings),
      out_shardings=(fragment_shardings, opt_state_shardings),
      donate_argnums=(0, 1, 2),
  )
  return compute_grad, apply_outer_step


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def _run_syncer_loop(
    config,
    syncer_cpu_mesh,
    cpu_submeshes,
    transport,
    recorder,
    checkpoint_lock,
    abstract_params=None,
    abstract_opt_state=None,
):
  """Runs a memory-bounded coordinator-only outer optimization loop."""
  max_logging.log("Syncer: Starting loop")

  num_learners = config.num_diloco_replicas

  if abstract_params is None or abstract_opt_state is None:
    syncer_model_config = make_learner_config(config, learner_idx=0, num_learners=num_learners)
    abstract_params, abstract_opt_state = get_abstract_syncer_state(syncer_model_config, syncer_cpu_mesh)
  abstract_step = jax.ShapeDtypeStruct(
      (),
      jnp.int32,
      sharding=jax.sharding.NamedSharding(syncer_cpu_mesh, jax.sharding.PartitionSpec()),
  )
  abstract_syncer_state = SyncerState(params=abstract_params, opt_state=abstract_opt_state, step=abstract_step)

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

  params_shardings = jax.tree_util.tree_map(lambda x: x.sharding, abstract_params)
  flat_params_shardings = {jax.tree_util.keystr(k): v for k, v in jax.tree_util.tree_flatten_with_path(params_shardings)[0]}
  opt_state_shardings = (optax.TraceState(trace=params_shardings), optax.EmptyState())
  outer_optimizer = optax.sgd(
      learning_rate=config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )

  checkpoint_start_step = 0 if restored_state is None else int(restored_state["items"].step)
  for learner_idx in range(num_learners):
    transport.recv_from_learner(
        learner_idx=learner_idx,
        step=checkpoint_start_step,
        fragment_id=START_STEP_FRAGMENT_ID,
    )
  max_logging.log(f"Syncer: every learner checkpoint is aligned at step {checkpoint_start_step}")

  if restored_state is None:
    max_logging.log("Syncer: waiting for init params from Learner 0")
    initial_params_l0 = transport.recv_from_learner(
        learner_idx=0,
        step=0,
        fragment_id=INITIAL_PARAMS_FRAGMENT_ID,
    )
    max_logging.log("Syncer: received init params from Learner 0")
    coordinator_params = _reshard_tree(initial_params_l0, params_shardings, donate=True)
    del initial_params_l0
    coordinator_params = jax.block_until_ready(coordinator_params)
    # This is a full-model boundary, so pin the initializer to the received
    # params' existing physical formats instead of risking a model-sized
    # conversion chosen from logical shardings alone.
    params_formats = jax.tree_util.tree_map(lambda x: x.format, coordinator_params)
    initialize_outer_optimizer = jit_with_layout_canonicalized_inputs(
        outer_optimizer.init,
        in_shardings=(params_formats,),
        out_shardings=opt_state_shardings,
    )
    outer_opt_state = initialize_outer_optimizer(coordinator_params)
    outer_opt_state = jax.block_until_ready(outer_opt_state)
    syncer_state = SyncerState(params=coordinator_params, opt_state=outer_opt_state, step=0)
    start_step = 0
    del coordinator_params, outer_opt_state, params_formats, initialize_outer_optimizer

  else:
    syncer_state = restored_state["items"]
    start_step = checkpoint_start_step
    syncer_state = jax.block_until_ready(syncer_state)
    del restored_state
    max_logging.log(f"Syncer restored from step {start_step}")

  def send_resharded_to_learner(learner_idx, step, fragment_id, tree, target_shardings):
    """Reserves queue capacity before allocating a destination-mesh payload."""
    transport.reserve_to_learner(learner_idx)
    reservation_owned = True
    try:
      payload = _reshard_tree(tree, target_shardings, donate=False)
      payload = jax.block_until_ready(payload)
      # publish_to_learner owns and releases the reservation on failure.
      reservation_owned = False
      transport.publish_to_learner(learner_idx, step, fragment_id, payload)
      return payload
    finally:
      if reservation_owned:
        transport.cancel_to_learner_reservation(learner_idx)

  # A coordinator-only state makes each initial broadcast a real full-model
  # transfer. Send one at a time and wait until the learner has completed its
  # TPU copy before allocating the next destination copy.
  for i, submesh in enumerate(cpu_submeshes):
    local_sharding = _shardings_on_mesh(params_shardings, submesh)
    max_logging.log(f"Syncer: sending params to Learner {i} at step {start_step}")
    local_params = send_resharded_to_learner(
        i,
        start_step,
        INITIAL_PARAMS_FRAGMENT_ID,
        syncer_state.params,
        local_sharding,
    )
    transport.recv_from_learner(
        learner_idx=i,
        step=start_step,
        fragment_id=INITIAL_PARAMS_ACK_FRAGMENT_ID,
    )
    del local_params, local_sharding
    gc.collect()
    max_logging.log(f"Syncer: sent params to Learner {i} at step {start_step}")

  manipulator = FragmentedTreeManipulator.create(syncer_state.params, config)
  num_fragments = manipulator.num_fragments

  steps_between_syncs_plus_1 = int(round(config.diloco_sync_period / num_fragments))
  steps_between_syncs_plus_1 = max(1, steps_between_syncs_plus_1)
  period = num_fragments * steps_between_syncs_plus_1

  sync_steps = [step for step in range(start_step + 1, config.steps + 1) if step % steps_between_syncs_plus_1 == 0]

  def fragment_descriptor(tree, fragment_idx):
    """Returns fragment metadata without allocating or taking from full arrays."""
    result = {}
    for keypath, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
      key = jax.tree_util.keystr(keypath)
      is_scanned = manipulator.keypath_to_is_scanned.get(key, False)
      if fragment_idx == 0 and is_scanned:
        continue
      if fragment_idx > 0 and not is_scanned:
        continue

      shape = list(value.shape)
      if fragment_idx > 0:
        axis = manipulator.param_scan_axis % len(shape)
        shape[axis] = len(manipulator.fragment_to_layer_indices[fragment_idx])
      result[key] = jax.ShapeDtypeStruct(
          tuple(shape),
          value.dtype,
          sharding=jax.sharding.NamedSharding(
              syncer_cpu_mesh,
              flat_params_shardings[key].spec,
              memory_kind=flat_params_shardings[key].memory_kind,
          ),
      )
    return result

  step_fns_by_frag = {}
  reduction_fns_by_frag = {}
  fragment_shardings_by_frag = {}
  for f_idx in range(num_fragments):
    frag_dict = fragment_descriptor(syncer_state.params, f_idx)
    trace_dict = fragment_descriptor(syncer_state.opt_state[0].trace, f_idx)
    fragment_shardings = _flat_fragment_shardings(syncer_cpu_mesh, flat_params_shardings, frag_dict)
    fragment_shardings_by_frag[f_idx] = fragment_shardings
    reduction_fns_by_frag[f_idx] = make_streaming_mean_fns(
        frag_dict,
        fragment_shardings,
        num_learners,
    )
    step_fns_by_frag[f_idx] = make_step_fns(
        syncer_cpu_mesh,
        flat_params_shardings,
        frag_dict,
        trace_dict,
        outer_optimizer,
    )

  syncer_ckpt_config = copy.deepcopy(config)
  syncer_ckpt_config._flat_config["pure_nnx"] = False

  for step in sync_steps:
    max_logging.log(f"Syncer: Step {step} sync starting")
    frag_idx = (step % period) // steps_between_syncs_plus_1

    averaged_inner_frag = stream_learner_mean(
        transport,
        num_learners=num_learners,
        step=step,
        fragment_id=frag_idx,
        target_shardings=fragment_shardings_by_frag[frag_idx],
        reduction_fns=reduction_fns_by_frag[frag_idx],
    )
    max_logging.log(f"Syncer: received and reduced fragments for step {step}")

    with jax.set_mesh(syncer_cpu_mesh):
      # The take adapter accepts whatever physical layout Pathways reshard or
      # checkpoint restore produced. Scatters donate full scanned buffers.
      outer_params_frag = manipulator.get_flat_fragment(
          syncer_state.params,
          frag_idx,
          use_null_layout_jit=True,
      )
      trace_frag = manipulator.get_flat_fragment(
          syncer_state.opt_state[0].trace,
          frag_idx,
          use_null_layout_jit=True,
      )
      opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

      compute_grad, apply_outer_step = step_fns_by_frag[frag_idx]
      pseudo_grad_frag = compute_grad(outer_params_frag, averaged_inner_frag)
      pseudo_grad_frag = jax.block_until_ready(pseudo_grad_frag)
      del averaged_inner_frag
      new_outer_params_frag, new_opt_state_frag = apply_outer_step(pseudo_grad_frag, opt_state_frag, outer_params_frag)
      new_outer_params_frag, new_opt_state_frag = jax.block_until_ready((new_outer_params_frag, new_opt_state_frag))
      new_opt_state_trace = new_opt_state_frag[0].trace
      del (
          pseudo_grad_frag,
          opt_state_frag,
          outer_params_frag,
          trace_frag,
          new_opt_state_frag,
      )

      new_params = manipulator.apply_flat_fragment(
          syncer_state.params,
          frag_idx,
          new_outer_params_frag,
          use_null_layout_jit=True,
          donate_full_array=True,
      )
      new_trace = manipulator.apply_flat_fragment(
          syncer_state.opt_state[0].trace,
          frag_idx,
          new_opt_state_trace,
          use_null_layout_jit=True,
          donate_full_array=True,
      )
      new_params, new_trace = jax.block_until_ready((new_params, new_trace))
      new_opt_state = (
          optax.TraceState(trace=new_trace),
          syncer_state.opt_state[1],
      )
      syncer_state = syncer_state.replace(params=new_params, opt_state=new_opt_state, step=step)
      del new_params, new_trace, new_opt_state, new_opt_state_trace
    max_logging.log(f"Syncer: Step {step} outer step applied")

    for i, submesh in enumerate(cpu_submeshes):
      target_fragment_shardings = _flat_fragment_shardings(
          submesh,
          flat_params_shardings,
          new_outer_params_frag,
      )
      local_frag = send_resharded_to_learner(
          i,
          step,
          frag_idx,
          new_outer_params_frag,
          target_fragment_shardings,
      )
      del local_frag, target_fragment_shardings

    _save_checkpoint_serialized(
        checkpoint_manager,
        syncer_state,
        syncer_ckpt_config,
        None,
        checkpoint_lock,
        step=step,
    )
    max_logging.log(f"Syncer: Step {step} sync finished")
    del new_outer_params_frag
    gc.collect()

  if config.save_checkpoint_on_completion:
    _save_checkpoint_serialized(
        checkpoint_manager,
        syncer_state,
        syncer_ckpt_config,
        None,
        checkpoint_lock,
        step=int(syncer_state.step),
        force=True,
    )
  elif checkpoint_manager is not None:
    with checkpoint_lock:
      checkpoint_manager.wait_until_finished()


def _validate_colocated_cpu_clients(tpu_submeshes, cpu_submeshes):
  """Ensures side-channel resharding sees one Pathways IFRT client."""
  accelerator_client = getattr(tpu_submeshes[0].devices.flat[0], "client", None)
  incompatible_devices = [
      device
      for mesh in cpu_submeshes
      for device in mesh.devices.flat
      if getattr(device, "client", None) is not accelerator_client
  ]
  if accelerator_client is None or incompatible_devices:
    raise RuntimeError(
        "Non-SPMD streaming DiLoCo requires colocated CPU devices from the same "
        "single-controller Pathways client as the TPU devices. "
        f"Incompatible CPU devices: {incompatible_devices}"
    )


def run_threaded_diloco(config, recorder, train_step, eval_step):
  """Orchestrator for multi-threaded DiLoCo."""
  max_logging.log("Starting run_threaded_diloco")
  num_learners = config.num_diloco_replicas
  _validate_checkpoint_alignment(config)

  max_logging.log("Creating global mesh")
  global_mesh = maxtext_utils.get_mesh_from_config(config)
  max_logging.log("Partitioning global mesh")
  tpu_submeshes = partition_mesh_by_diloco_axis(global_mesh, num_learners)
  cpu_submeshes = [colocated_python.colocated_cpu_devices(submesh) for submesh in tpu_submeshes]
  _validate_colocated_cpu_clients(tpu_submeshes, cpu_submeshes)
  syncer_cpu_mesh = cpu_submeshes[0]

  max_pending_fragments = max(1, int(config.num_communication_overlapping_steps) + 1)
  transport_manager = ThreadedTransportManager(
      num_learners,
      max_pending_fragments=max_pending_fragments,
  )

  # Build the outer state for only the coordinator CPU slice. The learner
  # config removes the now-absent diloco mesh axis and logical-axis mappings.
  max_logging.log("Getting abstract syncer state")
  syncer_model_config = make_learner_config(config, learner_idx=0, num_learners=num_learners)
  abstract_params, abstract_opt_state = get_abstract_syncer_state(syncer_model_config, syncer_cpu_mesh)
  max_logging.log("Got abstract syncer state")

  init_lock = threading.Lock()
  checkpoint_lock = threading.Lock()

  max_logging.log("Spawning learner threads")
  with ThreadPoolExecutor(max_workers=num_learners) as executor:
    futures = []
    local_devices = frozenset(jax.local_devices())

    def cancel_transport_on_failure(future):
      if future.cancelled():
        transport_manager.close()
        return
      if future.exception() is not None:
        transport_manager.close()

    for i in range(num_learners):
      learner_devices = tpu_submeshes[i].devices.flat
      should_run = all(device in local_devices for device in learner_devices)

      if should_run:
        learner_transport = LearnerTransport(transport_manager, i, cpu_submeshes[i])

        future = executor.submit(
            learner_loop,
            i,
            config,
            tpu_submeshes[i],
            cpu_submeshes[i],
            learner_transport,
            recorder,
            train_step,
            eval_step,
            init_lock=init_lock,
            checkpoint_lock=checkpoint_lock,
        )
        future.add_done_callback(cancel_transport_on_failure)
        futures.append(future)
      else:
        max_logging.log(f"Learner {i} is remote, not spawning thread")

    if len(futures) != num_learners:
      transport_manager.close()
      raise RuntimeError(
          "Non-SPMD streaming DiLoCo requires every learner submesh to be "
          "addressable by the single controller; "
          f"spawned {len(futures)} of {num_learners} learners"
      )

    syncer_transport = SyncerTransport(transport_manager)
    try:
      max_logging.log("Starting syncer loop")
      syncer_loop(
          config,
          syncer_cpu_mesh,
          cpu_submeshes,
          syncer_transport,
          recorder,
          checkpoint_lock,
          abstract_params=abstract_params,
          abstract_opt_state=abstract_opt_state,
      )

      max_logging.log("Waiting for learner threads to finish")
      for future in futures:
        future.result()
      max_logging.log("Finished run_threaded_diloco")
    finally:
      transport_manager.close()
