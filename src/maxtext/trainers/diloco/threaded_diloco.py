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
import gc
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
from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator, _get_tree_mesh
from maxtext.utils import exceptions
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from maxtext.utils.mesh_utils import partition_mesh_by_diloco_axis, stack_across_meshes_pytree
from jax._src.array import ArrayImpl
from jax.experimental.layout import Format, Layout

# Realign Python ArrayImpl.format for CPU devices to report clean untiled Layout(tiling=())
# matching the true physical untiled memory layout in Pathways (CL 959475984).
if not hasattr(ArrayImpl, "_orig_format_prop"):
  ArrayImpl._orig_format_prop = ArrayImpl.format

  def _patched_format(self):
    if hasattr(self, "sharding") and hasattr(self.sharding, "mesh") and self.sharding.mesh is not None:
      if len(self.sharding.mesh.devices.flat) > 0 and self.sharding.mesh.devices.flat[0].platform == "cpu":
        null_layout = Layout(major_to_minor=tuple(range(self.ndim)), tiling=())
        return Format(layout=null_layout, sharding=self.sharding)
    return ArrayImpl._orig_format_prop.fget(self)

  ArrayImpl.format = property(_patched_format)

# Prevent recursive device_put(Format) calls in JAX 0.10 during array layout coercion on Pathways
from jax._src import api

if not hasattr(api, "_orig_device_put"):
  api._orig_device_put = api.device_put

  def _safe_device_put(x, device=None, *args, **kwargs):
    if isinstance(device, Format):
      if hasattr(x, "sharding") and x.sharding == device.sharding:
        return x
    return api._orig_device_put(x, device, *args, **kwargs)

  api.device_put = _safe_device_put
  jax.device_put = _safe_device_put


def _normalize_to_null_layout(tree):
  return tree


def _slice_global_mesh_to_submesh(
    tree: Any,
    submesh: jax.sharding.Mesh,
    learner_idx: int,
    num_devices_per_mesh: int,
    target_shardings: Any,
    num_learners: int = 2,
    target_shapes: Any = None,
) -> Any:
  """Slices shards of a global_mesh array to construct a submesh array without cross-device communication."""
  def _slice_leaf(leaf, sharding_spec, shape=None):
    if not isinstance(leaf, jax.Array):
      return leaf
    target_sharding = (
        sharding_spec
        if isinstance(sharding_spec, (jax.sharding.NamedSharding, Format))
        else jax.sharding.NamedSharding(submesh, getattr(sharding_spec, "spec", jax.sharding.PartitionSpec()))
    )
    if shape is not None:
      target_shape = shape
    else:
      target_shape = leaf.shape[1:] if leaf.ndim > 0 and leaf.shape[0] == num_learners else leaf.shape

    if leaf.ndim > 0 and leaf.shape[0] == num_learners:
      if isinstance(target_sharding, Format):
        target_spec = target_sharding.sharding.spec if hasattr(target_sharding.sharding, "spec") else jax.sharding.PartitionSpec()
      else:
        target_spec = target_sharding.spec if hasattr(target_sharding, "spec") else jax.sharding.PartitionSpec()
      target_named_sharding = (
          target_sharding
          if isinstance(target_sharding, jax.sharding.NamedSharding)
          else jax.sharding.NamedSharding(submesh, target_spec)
      )
      if hasattr(leaf, "addressable_shards") and leaf.addressable_shards:
        start_idx = learner_idx * num_devices_per_mesh
        end_idx = start_idx + num_devices_per_mesh
        tpu_devices = list(submesh.devices.flat)
        if len(target_shape) == 3:
          # Scanned 3D layer weight: transpose axes 0 and 1 on CPU to match XLA {2, 1, 0} memory order
          tpu_shards = []
          for shard_idx, shard in enumerate(leaf.addressable_shards[start_idx:end_idx]):
            s = shard.data
            while s.ndim > len(target_shape) and s.shape[0] == 1:
              s = s.squeeze(0)
            s_t = jnp.swapaxes(s, 0, 1)
            tpu_shards.append(jax.device_put(s_t, tpu_devices[shard_idx]))
          shape_t = (target_shape[1], target_shape[0], target_shape[2])
          spec = target_named_sharding.spec
          spec_t = jax.sharding.PartitionSpec(
              spec[1] if len(spec) > 1 else None,
              spec[0] if len(spec) > 0 else None,
              spec[2] if len(spec) > 2 else None,
          )
          sharding_t = jax.sharding.NamedSharding(submesh, spec_t)
          tpu_arr_t = jax.make_array_from_single_device_arrays(shape_t, sharding_t, tpu_shards)
          return jnp.swapaxes(tpu_arr_t, 0, 1)
        elif len(target_shape) == 2 and target_shape[1] == 36:
          # Scanned 2D layer weight (e.g. layer norm / bias): transpose axes 0 and 1 on CPU to match XLA {1, 0} memory order
          tpu_shards = []
          for shard_idx, shard in enumerate(leaf.addressable_shards[start_idx:end_idx]):
            s = shard.data
            while s.ndim > len(target_shape) and s.shape[0] == 1:
              s = s.squeeze(0)
            s_t = jnp.swapaxes(s, 0, 1)
            tpu_shards.append(jax.device_put(s_t, tpu_devices[shard_idx]))
          shape_t = (target_shape[1], target_shape[0])
          spec = target_named_sharding.spec
          spec_t = jax.sharding.PartitionSpec(
              spec[1] if len(spec) > 1 else None,
              spec[0] if len(spec) > 0 else None,
          )
          sharding_t = jax.sharding.NamedSharding(submesh, spec_t)
          tpu_arr_t = jax.make_array_from_single_device_arrays(shape_t, sharding_t, tpu_shards)
          return jnp.swapaxes(tpu_arr_t, 0, 1)
        else:
          tpu_shards = []
          for shard_idx, shard in enumerate(leaf.addressable_shards[start_idx:end_idx]):
            s = shard.data
            while s.ndim > len(target_shape) and s.shape[0] == 1:
              s = s.squeeze(0)
            tpu_shards.append(jax.device_put(s, tpu_devices[shard_idx]))
          return jax.make_array_from_single_device_arrays(target_shape, target_named_sharding, tpu_shards)
      else:
        with jax.set_mesh(leaf.sharding.mesh if hasattr(leaf, "sharding") and leaf.sharding.mesh is not None else None):
          sliced_leaf = leaf[learner_idx]
        return _normalize_to_null_layout(jax.device_put(sliced_leaf, target_sharding))

    return _normalize_to_null_layout(jax.device_put(leaf, target_sharding))

  if target_shapes is not None:
    return jax.tree_util.tree_map(_slice_leaf, tree, target_shardings, target_shapes)
  return jax.tree_util.tree_map(_slice_leaf, tree, target_shardings)


# pylint: disable=abstract-method
class SyncerState(struct.PyTreeNode):
  params: Any
  opt_state: optax.OptState
  step: int


def get_first_step(model, state):
  """Extracts step integer safely across Linen (state.step) and NNX/Optax (state.optimizer.step).

  Prevents AttributeError and multislice host transfer errors.
  """
  try:
    if isinstance(model, nn.Module):
      val = getattr(state, "step", 0)
      if isinstance(val, (int, np.integer)):
        return int(val)
      return 0
    if hasattr(state, "optimizer") and hasattr(state.optimizer, "step"):
      val = state.optimizer.step
      if hasattr(val, "get_value"):
        val = val.get_value()
      elif hasattr(val, "value"):
        val = val.value
      if isinstance(val, (int, np.integer)):
        return int(val)
    return 0
  except Exception as e:
    max_logging.log(f"get_first_step encountered exception reading step, defaulting to 0: {e}")
    return 0


def _extract_scalar_metrics(tree):
  """Extracts Python scalar numbers from a JAX metric PyTree safely while inside mesh context."""

  def _leaf_to_scalar(x):
    if isinstance(x, jax.Array):
      if hasattr(x, "addressable_shards") and len(x.addressable_shards) > 0:
        s = x.addressable_shards[0].data
        val = jax.device_get(s)
      else:
        val = jax.device_get(x)
      if isinstance(val, np.ndarray):
        return float(val.mean())
      return float(val)
    elif isinstance(x, (np.ndarray, np.generic)):
      return float(x.mean())
    return x

  return jax.tree_util.tree_map(_leaf_to_scalar, tree)


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

  # Only enable profiling on the first island (learner 0); disable for all other islands
  if learner_idx != 0:
    learner_config._flat_config["profiler"] = ""

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


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def _run_learner_loop(
    learner_idx, config, submesh, transport, recorder, train_step, eval_step, init_lock
):
  """Runs the main training and communication loop for a single learner replica."""
  max_logging.log(f"Learner {learner_idx}: Starting loop")
  learner_config = make_learner_config(config, learner_idx, config.num_diloco_replicas)
  learner_config._flat_config["run_name"] = config.run_name + f"_learner_{learner_idx}"

  with jax.set_mesh(submesh), submesh, nn_partitioning.axis_rules(learner_config.logical_axis_rules):
    learner_config._flat_config["checkpoint_dir"] = config.checkpoint_dir + f"/learner_{learner_idx}"

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

    raw_state = state
    if isinstance(model, nn.Module):
      jit_model = model
    else:
      jit_model, state = nnx.split(raw_state)

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

    start_step = get_first_step(model, raw_state)

    # Synchronized Initialization / Resume
    try:
      if start_step == 0:
        params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else raw_state.params
        max_logging.log(f"Learner {learner_idx}: sending init params")
        transport.send_to_syncer(step=0, fragment_id=-1, data=params)
        max_logging.log(f"Learner {learner_idx}: sent init params")
      else:
        global_params = transport.recv_from_syncer(step=start_step, fragment_id=-1)
        if learner_config.pure_nnx:
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
            nnx.update(state.model, global_params)
        else:
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
            raw_state = raw_state.replace(params=global_params)
    except Exception as e:
      max_logging.error(f"Learner {learner_idx} crashed in init: {e}")
      max_logging.error(traceback.format_exc())
      raise e

    params_template = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else raw_state.params
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

    def p_mix_frags(i_frag, o_frag):
      if alpha == 0.0:
        return o_frag
      return jax.tree_util.tree_map(lambda x, y: alpha * x + (1.0 - alpha) * y, i_frag, o_frag)

    try:
      last_step_completion = datetime.datetime.now()
      for step in range(start_step, learner_config.steps):
        max_logging.log(f"Learner {learner_idx}: Step {step} starting")
        prof.maybe_activate_profiler(step, state)

        with jax.profiler.StepTraceAnnotation(f"train_learner_{learner_idx}", step_num=step):
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

            metrics = _extract_scalar_metrics(metrics)

          max_logging.log(f"Learner {learner_idx}: Step {step} finished")
          step_time_delta = datetime.datetime.now() - last_step_completion

          completed_step = step + 1

          if completed_step > 0 and completed_step % steps_between_syncs_plus_1 == 0:
            frag_idx = (completed_step % period) // steps_between_syncs_plus_1
            with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
              params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
              frag_data = manipulator.get_flat_fragment(params, frag_idx)
              frag_data = jax.tree_util.tree_map(
                  lambda leaf: jnp.copy(leaf) if isinstance(leaf, jax.Array) else leaf,
                  frag_data,
              )
            transport.send_to_syncer_async(completed_step, frag_idx, frag_data)

          if completed_step - tau > 0 and (completed_step - tau) % steps_between_syncs_plus_1 == 0:
            frag_idx = ((completed_step - tau) % period) // steps_between_syncs_plus_1
            received_leaves = transport.recv_from_syncer(completed_step - tau, frag_idx)

            with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
              params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
              new_params = manipulator.replace_leaves_from_dict(params, received_leaves)

              if learner_config.pure_nnx:
                nnx.update(state.model, new_params)
              else:
                state = state.replace(params=new_params)

          with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
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
                eval_metrics = _extract_scalar_metrics(eval_metrics)
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
      try:
        prof.deactivate()
      except Exception:
        pass
      max_logging.log(f"Learner {learner_idx} training stopped: {str(e)}")
    finally:
      metric_logger_instance.flush_metrics_and_cleanup()
      transport.close()


# pylint: disable=too-many-positional-arguments,too-many-arguments
def learner_loop(learner_idx, config, submesh, transport, recorder, train_step, eval_step, init_lock):
  """Wrapper to run the learner loop and handle/log top-level exceptions."""
  try:
    _run_learner_loop(learner_idx, config, submesh, transport, recorder, train_step, eval_step, init_lock)
  except Exception as e:
    max_logging.error(f"Learner {learner_idx} crashed: {e}")
    max_logging.error(traceback.format_exc())
    raise e


# pylint: disable=too-many-positional-arguments,too-many-arguments
def syncer_loop(
    config, global_mesh, submeshes, transport, recorder, abstract_params=None, abstract_opt_state=None
):
  """Wrapper to run the syncer loop and handle/log top-level exceptions."""
  try:
    _run_syncer_loop(config, global_mesh, submeshes, transport, recorder, abstract_params, abstract_opt_state)
  except Exception as e:
    max_logging.error(f"Syncer crashed: {e}")
    max_logging.error(traceback.format_exc())
    raise e


@jax.jit
def _compute_grad_jit(o_frag, stacked_i_frag):
  averaged_i_frag = jax.tree_util.tree_map(
      lambda x, o: jnp.broadcast_to(
          jnp.mean(x, axis=0, keepdims=(x.ndim == o.ndim)),
          x.shape if x.ndim == o.ndim else o.shape,
      ),
      stacked_i_frag,
      o_frag,
  )
  return jax.tree_util.tree_map(lambda x, y: x - y, o_frag, averaged_i_frag)


_APPLY_OUTER_STEP_CACHE = {}


def _get_apply_outer_step_jit(outer_optimizer):
  opt_key = id(outer_optimizer)
  if opt_key not in _APPLY_OUTER_STEP_CACHE:

    @functools.partial(jax.jit, donate_argnums=(1, 2))
    def _apply_outer_step_jit(g_frag, o_state_frag, p_frag):
      updates_frag, new_o_state_frag = outer_optimizer.update(g_frag, o_state_frag, params=p_frag)
      new_p_frag = optax.apply_updates(p_frag, updates_frag)
      return new_p_frag, new_o_state_frag

    _APPLY_OUTER_STEP_CACHE[opt_key] = _apply_outer_step_jit
  return _APPLY_OUTER_STEP_CACHE[opt_key]


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def make_step_fns(global_mesh, flat_params_shardings, frag_keys, trace_keys, outer_optimizer):
  """Creates JIT functions for computing gradients and applying outer steps."""

  def compute_grad(o_frag, stacked_i_frag):
    mesh = _get_tree_mesh(o_frag)
    if mesh is not None:
      with jax.set_mesh(mesh):
        return _compute_grad_jit(o_frag, stacked_i_frag)
    return _compute_grad_jit(o_frag, stacked_i_frag)

  _apply_fn = _get_apply_outer_step_jit(outer_optimizer)

  def apply_outer_step(g_frag, o_state_frag, p_frag):
    mesh = _get_tree_mesh(p_frag)
    if mesh is not None:
      with jax.set_mesh(mesh):
        return _apply_fn(g_frag, o_state_frag, p_frag)
    return _apply_fn(g_frag, o_state_frag, p_frag)

  return compute_grad, apply_outer_step


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def _run_syncer_loop(
    config, global_mesh, submeshes, transport, recorder, abstract_params=None, abstract_opt_state=None
):
  """Runs the main syncer loop that coordinates parameter averaging and outer optimization."""
  max_logging.log("Syncer: Starting loop")

  num_learners = config.num_diloco_replicas

  if abstract_params is None or abstract_opt_state is None:
    abstract_params, abstract_opt_state = get_abstract_syncer_state(config, global_mesh)
  abstract_step = jax.ShapeDtypeStruct(
      (), jnp.int32, sharding=jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec())
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

  devices_per_mesh = len(submeshes[0].devices.flat)
  cpu_submeshes = partition_mesh_by_diloco_axis(global_mesh, num_learners)

  if restored_state is None:  # (1,a) No checkpoint found, start from scratch
    learner_raw_params = [
        transport.recv_from_learner(learner_idx=i, step=0, fragment_id=-1)
        for i in range(num_learners)
    ]
    learner_formats = [
        {jax.tree_util.keystr(kp): v.format for kp, v in jax.tree_util.tree_leaves_with_path(p)}
        for p in learner_raw_params
    ]
    initial_learner_params = [
        _normalize_to_null_layout(
            jax.tree_util.tree_map(
                lambda x, s, submesh=cpu_submeshes[i]: jax.device_put(x, jax.sharding.NamedSharding(submesh, s.spec)),
                learner_raw_params[i],
                params_shardings,
            )
        )
        for i in range(num_learners)
    ]
    max_logging.log("Syncer: received init params from all learners, stacking across meshes")
    global_params = stack_across_meshes_pytree(initial_learner_params, global_mesh, "diloco")
    with jax.set_mesh(global_mesh):
      outer_optimizer = optax.sgd(
          learning_rate=config.diloco_outer_lr,
          momentum=config.diloco_outer_momentum,
          nesterov=True,
      )
      outer_opt_state = outer_optimizer.init(global_params)

    syncer_state = SyncerState(params=global_params, opt_state=outer_opt_state, step=0)
    start_step = 0

  else:  # loading checkpoints successfully
    syncer_state = restored_state["items"]
    start_step = int(syncer_state.step)
    max_logging.log(f"Syncer restored from step {start_step}")
    for i, submesh in enumerate(submeshes):
      local_sharding = jax.tree_util.tree_map(
          lambda s, submesh=submesh: jax.sharding.NamedSharding(submesh, s.spec),
          params_shardings,
      )
      target_shapes = jax.tree_util.tree_map(lambda x: x.shape, abstract_params)
      local_params = _slice_global_mesh_to_submesh(
          syncer_state.params,
          submesh,
          i,
          devices_per_mesh,
          local_sharding,
          num_learners,
          target_shapes,
      )
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

  params_full_sharding = jax.tree_util.tree_map(
      lambda s: jax.sharding.NamedSharding(global_mesh, s.spec), params_shardings
  )

  compute_grad, apply_outer_step = make_step_fns(
      global_mesh, flat_params_shardings, None, None, outer_optimizer
  )

  # Start main syncer loop
  for step in sync_steps:  # e.g. 50, 100, 150... if sync_period=50
    max_logging.log(f"Syncer: Step {step} sync starting")
    frag_idx = (step % period) // steps_between_syncs_plus_1

    learner_frags = []

    # receive the fragment of the current step from each learner.
    for i in range(num_learners):
      frag_i = transport.recv_from_learner(learner_idx=i, step=step, fragment_id=frag_idx)
      frag_i_cpu = _normalize_to_null_layout(
          jax.tree_util.tree_map(
              lambda x, submesh=cpu_submeshes[i]: jax.device_put(x, jax.sharding.NamedSharding(submesh, x.sharding.spec)),
              frag_i,
          )
      )
      learner_frags.append(frag_i_cpu)
    max_logging.log(f"Syncer: received all fragments for step {step}")

    stacked_inner_frag = stack_across_meshes_pytree(learner_frags, global_mesh, "diloco")
    max_logging.log(f"Syncer: Step {step} stacking done")

    with jax.set_mesh(global_mesh):
      outer_params_frag = manipulator.get_flat_fragment(syncer_state.params, frag_idx, has_replica_dim=True)
      trace_frag = manipulator.get_flat_fragment(syncer_state.opt_state[0].trace, frag_idx, has_replica_dim=True)
      opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

      pseudo_grad_frag = compute_grad(outer_params_frag, stacked_inner_frag)
      new_outer_params_frag, new_opt_state_frag = apply_outer_step(pseudo_grad_frag, opt_state_frag, outer_params_frag)
      new_opt_state_trace = new_opt_state_frag[0].trace

      new_params = manipulator.apply_flat_fragment(
          syncer_state.params, frag_idx, new_outer_params_frag, has_replica_dim=True
      )

      new_trace = manipulator.apply_flat_fragment(
          syncer_state.opt_state[0].trace, frag_idx, new_opt_state_trace, has_replica_dim=True
      )

      new_opt_state = (optax.TraceState(trace=new_trace), syncer_state.opt_state[1])

      syncer_state = syncer_state.replace(params=new_params, opt_state=new_opt_state, step=step)
    max_logging.log(f"Syncer: Step {step} outer step applied")

    # Send updated full parameter leaves directly to each learner's submesh.
    updated_full_leaves = manipulator.get_leaves_for_fragment(new_params, frag_idx)
    for i, submesh in enumerate(submeshes):
      target_shardings = {
          k: jax.sharding.NamedSharding(submesh, flat_params_shardings[k].spec) for k in updated_full_leaves
      }
      target_shapes = {k: updated_full_leaves[k].shape[1:] for k in updated_full_leaves}
      local_leaves = _slice_global_mesh_to_submesh(
          updated_full_leaves,
          submesh,
          i,
          devices_per_mesh,
          target_shardings,
          num_learners=num_learners,
          target_shapes=target_shapes,
      )
      transport.send_to_learner(learner_idx=i, step=step, fragment_id=frag_idx, data=local_leaves)

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
    del learner_frags, stacked_inner_frag, outer_params_frag, trace_frag
    del opt_state_frag, pseudo_grad_frag, new_outer_params_frag, new_opt_state_trace
    gc.collect()

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

  # Create colocated CPU mesh for syncer outer step execution
  max_logging.log("Creating colocated CPU mesh for syncer")
  colocated_cpu_mesh = colocated_python.colocated_cpu_devices(global_mesh)

  transport_manager = ThreadedTransportManager(num_learners)

  # Get abstract syncer state on colocated CPU mesh
  max_logging.log("Getting abstract syncer state on colocated CPU mesh")
  abstract_params, abstract_opt_state = get_abstract_syncer_state(config, colocated_cpu_mesh)
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
        learner_transport = LearnerTransport(transport_manager, i)

        futures.append(
            executor.submit(
                learner_loop,
                i,
                config,
                tpu_submeshes[i],  # each learner will only see its local TPU submesh
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
    max_logging.log("Starting syncer loop on colocated CPU mesh")
    syncer_loop(
        config,
        colocated_cpu_mesh,
        tpu_submeshes,
        syncer_transport,
        recorder,
        abstract_params=abstract_params,
        abstract_opt_state=abstract_opt_state,
    )

    max_logging.log("Waiting for learner threads to finish")
    for f in futures:
      f.result()
    max_logging.log("Finished run_threaded_diloco")
