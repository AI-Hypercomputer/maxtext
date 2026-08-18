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

# Ensure native uncorrupted device_put
from jax._src import api

if hasattr(api, "_orig_device_put"):
  api.device_put = api._orig_device_put
  jax.device_put = api._orig_device_put


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
      s = np.asarray(leaf[learner_idx])
      while s.ndim > len(target_shape) and s.shape[0] == 1:
        s = s.reshape(s.shape[1:])
      return s

    return np.asarray(leaf)

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
_fold_in_jit = jax.jit(jax.random.fold_in)


@functools.partial(jax.jit, static_argnames=("manipulator",))
def _extract_scanned_fragment_jit(params, manipulator, layer_idx: jax.Array):
  """Extracts any scanned layer fragment via native dynamic slice (single JIT for all layers)."""
  return manipulator.dynamic_extract_scanned_fragment(params, layer_idx)


def _extract_fragment(params, manipulator, frag_idx: int):
  """Extracts fragment using pure Python for Fragment 0 and 1 single static JIT across all scanned layers."""
  if frag_idx == 0:
    return manipulator.get_flat_fragment(params, 0)
  return _extract_scanned_fragment_jit(params, manipulator, jnp.asarray(frag_idx - 1, dtype=jnp.int32))


@functools.partial(jax.jit, static_argnames=("manipulator",))
def _apply_scanned_fragment_jit(params, manipulator, layer_idx: jax.Array, frag_dict: dict[str, Any]):
  """Applies any scanned layer fragment via native dynamic update slice (single JIT for all layers)."""
  return manipulator.dynamic_apply_scanned_fragment(params, layer_idx, frag_dict)


def _apply_fragment(params, manipulator, frag_idx: int, frag_dict: dict[str, Any]):
  """Applies fragment using pure Python for Fragment 0 and 1 single static JIT across all scanned layers."""
  if frag_idx == 0:
    return manipulator.apply_flat_fragment(params, 0, frag_dict)
  return _apply_scanned_fragment_jit(params, manipulator, jnp.asarray(frag_idx - 1, dtype=jnp.int32), frag_dict)


def _extract_scalar_metrics(tree):
  """Extracts Python scalar numbers from a JAX metric PyTree safely while inside mesh context.

  Packs all device JAX array leaves into a single 1D tensor to perform a single
  batched D2H transfer over gRPC (inspired by jax_pack).
  """
  try:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if not leaves:
      return tree

    jax_indices = []
    jax_scalars = []
    result_leaves = list(leaves)

    for i, leaf in enumerate(leaves):
      if isinstance(leaf, jax.Array):
        s_mean = jnp.mean(leaf).astype(jnp.float32)
        jax_indices.append(i)
        jax_scalars.append(jnp.reshape(s_mean, (1,)))
      elif isinstance(leaf, (np.ndarray, np.generic)):
        result_leaves[i] = float(leaf.mean())
      elif isinstance(leaf, (int, float)):
        result_leaves[i] = float(leaf)
      else:
        result_leaves[i] = leaf

    if jax_scalars:
      packed_device = jnp.concatenate(jax_scalars)
      packed_host = jax.device_get(packed_device)
      for idx, scalar_val in zip(jax_indices, packed_host):
        result_leaves[idx] = float(scalar_val)

    return jax.tree_util.tree_unflatten(treedef, result_leaves)
  except Exception as e:
    max_logging.log(f"Error in _extract_scalar_metrics: {e}")
    return tree


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

  # Adjust batch sizes for the learner's submesh (num_devices = total // num_learners)
  if hasattr(config, "num_target_devices") and config.num_target_devices:
    learner_config._flat_config["num_target_devices"] = config.num_target_devices // num_learners

  if hasattr(config, "global_batch_size_to_train_on") and config.global_batch_size_to_train_on:
    learner_config._flat_config["global_batch_size_to_train_on"] = config.global_batch_size_to_train_on // num_learners

  if hasattr(config, "global_batch_size_to_load") and config.global_batch_size_to_load:
    learner_config._flat_config["global_batch_size_to_load"] = config.global_batch_size_to_load // num_learners

  if hasattr(config, "micro_batch_size_to_train_on") and config.micro_batch_size_to_train_on:
    learner_config._flat_config["micro_batch_size_to_train_on"] = config.micro_batch_size_to_train_on // num_learners

  if hasattr(config, "global_batch_size_to_eval_on") and config.global_batch_size_to_eval_on:
    learner_config._flat_config["global_batch_size_to_eval_on"] = config.global_batch_size_to_eval_on // num_learners

  if hasattr(config, "global_batch_size_to_load_eval") and config.global_batch_size_to_load_eval:
    learner_config._flat_config["global_batch_size_to_load_eval"] = config.global_batch_size_to_load_eval // num_learners

  if hasattr(config, "micro_batch_size_to_eval_on") and config.micro_batch_size_to_eval_on:
    learner_config._flat_config["micro_batch_size_to_eval_on"] = config.micro_batch_size_to_eval_on // num_learners

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

      abstract_opt_state = jax.eval_shape(outer_optimizer.init, abstract_params)
  else:
    model = model_creation_utils.from_config(config, mesh=local_cpu_mesh)
    abstract_vars = maxtext_utils.get_abstract_param(model, config)
    abstract_params = abstract_vars["params"]

    params_logical_annotations = nn.get_partition_spec(abstract_params)
    params_mesh_shardings = nn.logical_to_mesh_sharding(
        params_logical_annotations, local_cpu_mesh, config.logical_axis_rules
    )

    abstract_params = jax.eval_shape(
        jax.jit(lambda: abstract_params, out_shardings=params_mesh_shardings)
    )

    outer_optimizer = optax.sgd(
        learning_rate=config.diloco_outer_lr,
        momentum=config.diloco_outer_momentum,
        nesterov=True,
    )

    opt_state_shardings = (optax.TraceState(trace=params_mesh_shardings), optax.EmptyState())
    abstract_opt_state = jax.eval_shape(
        jax.jit(outer_optimizer.init, out_shardings=opt_state_shardings), abstract_params
    )

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
        max_logging.log(f"Learner {learner_idx}: sent init params, waiting for syncer ack")
        transport.recv_from_syncer(step=0, fragment_id=-1)
        max_logging.log(f"Learner {learner_idx}: received syncer ack, starting training")
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

    if learner_idx > 0:
      learner_config._flat_config["profiler"] = ""
    prof = profiler.Profiler(learner_config, offset_step=start_step)
    metric_logger_instance = metric_logger.MetricLogger(
        config=learner_config, learning_rate_schedule=learning_rate_schedule
    )
    metric_logger_instance.write_setup_info_to_tensorboard(params_template)

    # AOT pre-warm extract and apply kernels on TPU mesh and record fragment leaf layouts & shardings
    fragment_leaf_layouts = {}
    fragment_leaf_shardings = {}
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
      dummy_frag0 = _extract_fragment(params_template, manipulator, 0)
      _ = _apply_fragment(params_template, manipulator, 0, dummy_frag0)
      fragment_leaf_layouts[0] = {
          k: getattr(getattr(v, "format", None), "layout", None) for k, v in dummy_frag0.items()
      }
      fragment_leaf_shardings[0] = {k: getattr(v, "sharding", None) for k, v in dummy_frag0.items()}

      dummy_frag1 = _extract_fragment(params_template, manipulator, 1)
      _ = _apply_fragment(params_template, manipulator, 1, dummy_frag1)
      fragment_leaf_layouts[1] = {
          k: getattr(getattr(v, "format", None), "layout", None) for k, v in dummy_frag1.items()
      }
      fragment_leaf_shardings[1] = {k: getattr(v, "sharding", None) for k, v in dummy_frag1.items()}
    max_logging.log(f"Learner {learner_idx}: AOT pre-compiled extract and apply kernels for non-scanned and scanned fragments")

    try:
      last_step_completion = datetime.datetime.now()
      for step in range(start_step, learner_config.steps):
        max_logging.log(f"Learner {learner_idx}: Step {step} starting")
        try:
          prof.maybe_activate_profiler(step, None)
        except Exception:
          pass

        with jax.profiler.StepTraceAnnotation(f"train_learner_{learner_idx}", step_num=step):
          example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
          if isinstance(model, nn.Module):
            step_rng_args = (_fold_in_jit(init_rng, step),)
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
              frag_data = _extract_fragment(params, manipulator, frag_idx)
              frag_data = jax.block_until_ready(frag_data)
            transport.send_to_syncer_async(completed_step, frag_idx, frag_data)

          if completed_step - tau > 0 and (completed_step - tau) % steps_between_syncs_plus_1 == 0:
            frag_idx = ((completed_step - tau) % period) // steps_between_syncs_plus_1
            received_leaves = transport.recv_from_syncer(completed_step - tau, frag_idx)

            with jax.set_mesh(mesh), nn_partitioning.axis_rules(learner_config.logical_axis_rules):
              frag_type = 0 if frag_idx == 0 else 1
              received_leaves_tpu = {}
              for k, v_cpu in received_leaves.items():
                target_shd = fragment_leaf_shardings.get(frag_type, {}).get(k)
                if target_shd is None:
                  target_shd = flat_params_shardings.get(k)
                if target_shd is None:
                  target_shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
                elif not isinstance(target_shd, jax.sharding.NamedSharding):
                  target_shd = jax.sharding.NamedSharding(
                      mesh, getattr(target_shd, "spec", jax.sharding.PartitionSpec())
                  )
                received_leaves_tpu[k] = jax.device_put(v_cpu, target_shd)

              params = nnx.state(state.model, nnx.Param) if learner_config.pure_nnx else state.params
              new_params = _apply_fragment(params, manipulator, frag_idx, received_leaves_tpu)

              if learner_config.pure_nnx:
                nnx.update(state.model, new_params)
              else:
                state = state.replace(params=new_params)

            del received_leaves, received_leaves_tpu

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

          try:
            prof.maybe_deactivate_profiler(step, None)
          except Exception:
            pass
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


def _compute_grad_flat(o_leaves, stacked_i_leaves):
  averaged_i = tuple(
      jnp.broadcast_to(
          jnp.mean(x, axis=0, keepdims=(x.ndim == o.ndim)),
          x.shape if x.ndim == o.ndim else o.shape,
      )
      for x, o in zip(stacked_i_leaves, o_leaves)
  )
  return tuple(o - avg for o, avg in zip(o_leaves, averaged_i))


_FLAT_GRAD_JIT = jax.jit(_compute_grad_flat)
_FLAT_STEP_JIT_CACHE = {}


def _get_apply_outer_step_flat_jit(outer_optimizer):
  opt_key = id(outer_optimizer)
  if opt_key not in _FLAT_STEP_JIT_CACHE:

    @jax.jit
    def _apply_outer_step_flat_jit(g_leaves, trace_leaves, p_leaves):
      o_state = (optax.TraceState(trace=trace_leaves), optax.EmptyState())
      updates, new_o_state = outer_optimizer.update(g_leaves, o_state, params=p_leaves)
      new_p_leaves = optax.apply_updates(p_leaves, updates)
      return new_p_leaves, new_o_state[0].trace

    _FLAT_STEP_JIT_CACHE[opt_key] = _apply_outer_step_flat_jit
  return _FLAT_STEP_JIT_CACHE[opt_key]


# pylint: disable=too-many-positional-arguments,too-many-arguments,unused-argument
def make_step_fns(
    global_mesh,
    flat_params_shardings,
    frag_keys,
    trace_keys,
    outer_optimizer,
    abstract_params=None,
    abstract_opt_state=None,
    manipulator=None,
    num_learners=2,
):
  """Creates AOT-compiled / flat-tuple JIT functions for outer optimization."""
  jit_apply_fn = _get_apply_outer_step_flat_jit(outer_optimizer)
  jit_grad_fn = _FLAT_GRAD_JIT

  aot_grad_executables = {}
  aot_step_executables = {}

  if abstract_params is not None and manipulator is not None and abstract_opt_state is not None:
    with jax.set_mesh(global_mesh):
      num_frags = getattr(manipulator, "num_fragments", 2)
      frag_indices = [0, 1] if num_frags > 1 else [0]
      has_replica = any(
          hasattr(l, "shape") and len(l.shape) > 0 and l.shape[0] == num_learners
          for l in jax.tree_util.tree_leaves(abstract_params)
      )
      for f_idx in frag_indices:
        try:
          p_frag = manipulator.get_flat_fragment(abstract_params, f_idx, has_replica_dim=has_replica)
          t_frag = manipulator.get_flat_fragment(abstract_opt_state[0].trace, f_idx, has_replica_dim=has_replica)
          raw_p_leaves, _ = jax.tree_util.tree_flatten(p_frag)
          raw_t_leaves, _ = jax.tree_util.tree_flatten(t_frag)
          if has_replica:
            p_leaves = [
                jax.ShapeDtypeStruct(l.shape, l.dtype, sharding=getattr(l, "sharding", None)) for l in raw_p_leaves
            ]
            t_leaves = [
                jax.ShapeDtypeStruct(l.shape, l.dtype, sharding=getattr(l, "sharding", None)) for l in raw_t_leaves
            ]
          else:
            p_leaves = [
                jax.ShapeDtypeStruct(
                    (num_learners, *l.shape),
                    l.dtype,
                    sharding=jax.sharding.NamedSharding(
                        global_mesh,
                        jax.sharding.PartitionSpec(
                            "diloco",
                            *(l.sharding.spec if hasattr(l, "sharding") and l.sharding is not None else ()),
                        ),
                    ),
                )
                for l in raw_p_leaves
            ]
            t_leaves = [
                jax.ShapeDtypeStruct(
                    (num_learners, *l.shape),
                    l.dtype,
                    sharding=jax.sharding.NamedSharding(
                        global_mesh,
                        jax.sharding.PartitionSpec(
                            "diloco",
                            *(l.sharding.spec if hasattr(l, "sharding") and l.sharding is not None else ()),
                        ),
                    ),
                )
                for l in raw_t_leaves
            ]
          i_leaves = list(p_leaves)
          aot_grad_executables[f_idx] = jit_grad_fn.lower(tuple(p_leaves), tuple(i_leaves)).compile()
          aot_step_executables[f_idx] = jit_apply_fn.lower(tuple(p_leaves), tuple(t_leaves), tuple(p_leaves)).compile()
          max_logging.log(f"Syncer: AOT pre-compiled outer optimization executable for fragment type {f_idx}")
        except Exception as e:
          max_logging.log(f"Syncer: AOT compilation for fragment {f_idx} deferred to JIT: {e}")

  def compute_grad(o_frag, stacked_i_frag, frag_idx=None):
    o_leaves, treedef_o = jax.tree_util.tree_flatten(o_frag)
    i_leaves, _ = jax.tree_util.tree_flatten(stacked_i_frag)

    exec_key = 0 if frag_idx == 0 else 1
    if exec_key in aot_grad_executables:
      g_leaves = aot_grad_executables[exec_key](tuple(o_leaves), tuple(i_leaves))
    else:
      mesh = _get_tree_mesh(o_frag)
      if mesh is not None:
        with jax.set_mesh(mesh):
          g_leaves = jit_grad_fn(tuple(o_leaves), tuple(i_leaves))
      else:
        g_leaves = jit_grad_fn(tuple(o_leaves), tuple(i_leaves))
    return jax.tree_util.tree_unflatten(treedef_o, g_leaves)

  def apply_outer_step(g_frag, o_state_frag, p_frag, frag_idx=None):
    g_leaves, _ = jax.tree_util.tree_flatten(g_frag)
    t_leaves, treedef_t = jax.tree_util.tree_flatten(o_state_frag[0].trace)
    p_leaves, treedef_p = jax.tree_util.tree_flatten(p_frag)

    exec_key = 0 if frag_idx == 0 else 1
    if exec_key in aot_step_executables:
      new_p_leaves, new_t_leaves = aot_step_executables[exec_key](tuple(g_leaves), tuple(t_leaves), tuple(p_leaves))
    else:
      mesh = _get_tree_mesh(p_frag)
      if mesh is not None:
        with jax.set_mesh(mesh):
          new_p_leaves, new_t_leaves = jit_apply_fn(tuple(g_leaves), tuple(t_leaves), tuple(p_leaves))
      else:
        new_p_leaves, new_t_leaves = jit_apply_fn(tuple(g_leaves), tuple(t_leaves), tuple(p_leaves))

    new_p_frag = jax.tree_util.tree_unflatten(treedef_p, new_p_leaves)
    new_t_frag = jax.tree_util.tree_unflatten(treedef_t, new_t_leaves)
    new_o_state_frag = (optax.TraceState(trace=new_t_frag), o_state_frag[1])
    return new_p_frag, new_o_state_frag

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
    for i in range(num_learners):
      transport.send_to_learner(learner_idx=i, step=0, fragment_id=-1, data=True)
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
      global_mesh,
      flat_params_shardings,
      None,
      None,
      outer_optimizer,
      abstract_params=syncer_state.params,
      abstract_opt_state=syncer_state.opt_state,
      manipulator=manipulator,
      num_learners=num_learners,
  )

  # Start main syncer loop
  for step in sync_steps:  # e.g. 50, 100, 150... if sync_period=50
    max_logging.log(f"Syncer: Step {step} sync starting")
    frag_idx = (step % period) // steps_between_syncs_plus_1

    frag_template = manipulator.get_flat_fragment(syncer_state.params, frag_idx, has_replica_dim=True)
    frag_specs = jax.tree_util.tree_map(
        lambda s: jax.sharding.PartitionSpec(*s.sharding.spec[1:])
        if hasattr(s, "sharding") and hasattr(s.sharding, "spec") and len(s.sharding.spec) > 1
        else jax.sharding.PartitionSpec(),
        frag_template,
    )

    learner_frags = []

    # receive the fragment of the current step from each learner.
    for i in range(num_learners):
      frag_i = transport.recv_from_learner(learner_idx=i, step=step, fragment_id=frag_idx)
      frag_i_cpu = jax.tree_util.tree_map(
          lambda x, spec, submesh=cpu_submeshes[i]: jax.device_put(
              x, jax.sharding.NamedSharding(submesh, spec)
          ),
          frag_i,
          frag_specs,
      )
      learner_frags.append(frag_i_cpu)
    max_logging.log(f"Syncer: received all fragments for step {step}")

    stacked_inner_frag = stack_across_meshes_pytree(learner_frags, global_mesh, "diloco")
    max_logging.log(f"Syncer: Step {step} stacking done")

    with jax.set_mesh(global_mesh):
      outer_params_frag = manipulator.get_flat_fragment(syncer_state.params, frag_idx, has_replica_dim=True)
      trace_frag = manipulator.get_flat_fragment(syncer_state.opt_state[0].trace, frag_idx, has_replica_dim=True)
      opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

      pseudo_grad_frag = compute_grad(outer_params_frag, stacked_inner_frag, frag_idx=frag_idx)
      new_outer_params_frag, new_opt_state_frag = apply_outer_step(
          pseudo_grad_frag, opt_state_frag, outer_params_frag, frag_idx=frag_idx
      )
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

    # Send updated fragment leaves directly to each learner's colocated CPU submesh.
    for i, cpu_submesh in enumerate(cpu_submeshes):
      target_shardings = {
          k: jax.sharding.NamedSharding(cpu_submesh, flat_params_shardings[k].spec) for k in new_outer_params_frag
      }
      target_shapes = {k: new_outer_params_frag[k].shape[1:] for k in new_outer_params_frag}
      local_leaves = _slice_global_mesh_to_submesh(
          new_outer_params_frag,
          cpu_submesh,
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
