# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Pipeline layer wrapping a decoder layer(s). Supports circular pipelining """

import functools
from typing import Any, Type

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh
import jax
import jax.ad_checkpoint

from flax import nnx
from flax import linen as nn

from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.maxtext_utils import all_gather_over_fsdp
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers


class Pipeline(nnx.Module):
  """Module that implements pipelining across stages."""

  def __init__(
      self,
      config: Config,
      layer_module: Type[nnx.Module],
      mesh: Mesh,
      remat_policy: Any = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy

    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    def create_layer(rngs_pytree):
      raw_rngs = {}
      rngs_dict = dict(rngs_pytree) if isinstance(rngs_pytree, nnx.Rngs) else rngs_pytree

      for name, val in rngs_dict.items():
        if hasattr(val, "tag") and hasattr(val, "value"):
          raw_rngs[name] = val.value
        else:
          raw_rngs[name] = val

      layer_class = layer_module.func if isinstance(layer_module, functools.partial) else layer_module
      if issubclass(layer_class, nnx.Module):
        if layer_class is nnx_wrappers.LinenToNnxWrapper:
          return layer_module(config=self.config, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(**raw_rngs))
        else:
          return layer_module(config=self.config, rngs=nnx.Rngs(**raw_rngs))
      else:
        return layer_module(config=self.config, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN)

    vmap_over_stages = nnx.vmap(
        create_layer,
        in_axes=0,
        out_axes=0,
        spmd_axis_name="stage",
    )
    if self.config.num_pipeline_repeats > 1:
      vmap_over_repeats = nnx.vmap(
          vmap_over_stages,
          in_axes=0,
          out_axes=0,
          spmd_axis_name="circular_repeats",
      )

    rngs_pure_dict = {}
    for name, stream in rngs.items():
      if hasattr(stream, "value"):
        rngs_pure_dict[name] = stream.value
      elif hasattr(stream, "key") and hasattr(stream.key, "value"):
        rngs_pure_dict[name] = stream.key.value
      else:
        rngs_pure_dict[name] = stream

    def stack_keys(key):
      if isinstance(key, int):
        key = jax.random.PRNGKey(key)
      elif (
          hasattr(key, "dtype")
          and hasattr(key, "ndim")
          and key.ndim == 0
          and not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
      ):
        key = jax.random.PRNGKey(key)

      if self.config.num_pipeline_repeats > 1:
        return jax.random.split(key, (self.config.num_pipeline_repeats, self.num_stages))
      else:
        return jax.random.split(key, self.num_stages)

    init_rngs_pytree = jax.tree.map(stack_keys, rngs_pure_dict)

    temp_rngs_dict = rngs_pure_dict

    layer_class = layer_module.func if isinstance(layer_module, functools.partial) else layer_module
    if issubclass(layer_class, nnx.Module):
      if layer_class is nnx_wrappers.LinenToNnxWrapper:
        temp_layer = layer_module(config=self.config, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(**temp_rngs_dict))
      else:
        temp_layer = layer_module(config=self.config, rngs=nnx.Rngs(**temp_rngs_dict))
    else:
      temp_layer = layer_module(config=self.config, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN)

    dummy_single_shape = (self.pipeline_microbatch_size, self.config.max_target_length, self.config.emb_dim)
    dummy_single_inputs = jnp.zeros(dummy_single_shape, dtype=jnp.float32)
    dummy_single_pos = jnp.zeros((self.pipeline_microbatch_size, self.config.max_target_length), dtype=jnp.int32)
    dummy_single_seg = jnp.zeros((self.pipeline_microbatch_size, self.config.max_target_length), dtype=jnp.int32)

    try:
        temp_layer(
            dummy_single_inputs,
            dummy_single_seg,
            dummy_single_pos,
            deterministic=False,
            model_mode=MODEL_MODE_TRAIN
        )
    except Exception as e:
         print(f"WARNING: Single layer pre-init failed: {e}")

    self.single_layer_graphdef, _ = nnx.split(temp_layer)
    del temp_layer

    if self.config.num_pipeline_repeats > 1:
      self.layers = vmap_over_repeats(init_rngs_pytree)
    else:
      self.layers = vmap_over_stages(init_rngs_pytree)
      graphdef, state = nnx.split(self.layers)
      state_with_axis = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
      nnx.update(self.layers, state_with_axis)

    total_microbatches = self.num_stages * self.microbatches_per_stage
    dummy_input_shape = (
        total_microbatches,
        self.pipeline_microbatch_size,
        self.config.max_target_length,
        self.config.emb_dim
    )
    dummy_inputs = jnp.zeros(dummy_input_shape, dtype=jnp.float32)
    dummy_pos = jnp.zeros(
        (total_microbatches, self.pipeline_microbatch_size, self.config.max_target_length),
        dtype=jnp.int32
    )
    dummy_seg = jnp.zeros(
        (total_microbatches, self.pipeline_microbatch_size, self.config.max_target_length),
        dtype=jnp.int32
    )

    dummy_loop_state = self.init_states(dummy_inputs)

    try:
        _, new_layers_state = self.run_one_iteration(
            dummy_loop_state,
            dummy_pos,
            dummy_seg,
            deterministic=False,
            model_mode=MODEL_MODE_TRAIN
        )
        nnx.update(self.layers, new_layers_state)
    except Exception as e:
        print(f"WARNING: Full pipeline eager init failed! Error: {e}")
        raise e

  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def init_states(self, inputs):
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = nn.with_logical_constraint(
        shift,
        ("activation_stage", "activation_batch", "activation_length", "activation_embed"),
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )

    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = nn.with_logical_constraint(
          prev_outputs,
          ("activation_stage", "activation_batch", "activation_length", "activation_embed"),
          rules=self.config.logical_axis_rules,
          mesh=self.mesh,
      )
    else:
      prev_outputs = None

    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    state_io = nn.with_logical_constraint(
        state_io,
        ("activation_stage", None, "activation_batch", "activation_length", "activation_embed"),
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )

    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
      circ_storage_mover = shift
    else:
      circ_storage = None
      circ_storage_mover = None

    init_loop_state = {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": circ_storage_mover,
        "loop_iteration": 0,
        "prev_outputs": prev_outputs,
    }
    return init_loop_state

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]

    if self.use_circ_storage:
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      circular_stage_in = shift

    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)

    def select_state_or_input(first_stage_in, shift):
      return jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_stage_in, shift)

    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = nn.with_logical_constraint(
        stages_in,
        ("activation_stage", "activation_batch", "activation_length", "activation_embed"),
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )
    return stages_in

  def shard_dim_by_stages(self, x, dim: int):
    dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*dims_mapping))
    return jax.lax.with_sharding_constraint(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0)
    weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    stage_weights = self.shard_dim_by_stages(stage_weights, gathered_weights_stage_dim)
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
    def _gather_one(x, i):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, i, 1, ids_dim), ids_dim)

    ids = self.shard_dim_by_stages(ids, 0)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0)

  def get_new_loop_state(self, output, loop_state):
    old_state_io = loop_state["state_io"]
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    old_prev_outputs = loop_state["prev_outputs"]

    def _rotate_right(arr):
      last = jax.lax.slice_in_dim(arr, self.num_stages - 1, self.num_stages, axis=0)
      except_last = jax.lax.slice_in_dim(arr, 0, self.num_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)

    def _shift_right(arr):
      padding = [[1, 0]] + [[0, 0]] * (arr.ndim - 1)
      return jax.lax.slice(jnp.pad(arr, padding), [0] * arr.ndim, arr.shape)

    def _update_shift(output_in):
      if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
        return _shift_right(output_in)
      else:
        return _rotate_right(output_in)

    if self.config.pipeline_delay_activation_forwarding:
      new_shift = _update_shift(old_prev_outputs)
      new_prev_outputs = output
    else:
      new_shift = _update_shift(output)
      new_prev_outputs = None

    if self.use_circ_storage:
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = _rotate_right(circ_storage_mover_in)
        rotated = jnp.expand_dims(rotated, 1)
        offset = (
            loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1
        ) % self.config.num_pipeline_microbatches
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _update_state_io(state_in, stream_slice, output):
      padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
      stream_slice = jax.lax.slice_in_dim(jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
      stream_slice = jnp.where(
          jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1, output, stream_slice
      )
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output)

    new_loop_state = {
        "state_io": new_state,
        "shift": new_shift,
        "circ_storage": new_circ_storage,
        "circ_storage_mover": new_circ_storage_mover,
        "loop_iteration": loop_iteration + 1,
        "prev_outputs": new_prev_outputs,
    }
    return new_loop_state

  def permute_output_micro_per_stage_dim(self, output):
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    output = output[:, permutation]
    return output

  def get_current_repeat_from_stages(self, full_state: nnx.State, loop_iteration: int):
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    def gather_pytree(path, pytree):
      del path
      return jax.tree.map(
          functools.partial(
              self.vmap_parallel_gather,
              repeat_ids=repeat_ids,
              repeat_dim_in_weights=0,
              stages_dim_in_weights=1,
          ),
          pytree,
      )

    return full_state.map(gather_pytree)

  def run_one_iteration(
      self,
      loop_state,
      positions,
      segment_ids,
      deterministic,
      model_mode,
  ):
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    _, state = nnx.split(self.layers)

    stage_vars_pytree = self.get_current_repeat_from_stages(state, loop_iteration)

    def run_stage_vmap_fn(
        layer_graphdef,
        stage_vars_slice,
        stage_input,
        stage_segment_id,
        stage_position,
    ):
      module = nnx.merge(layer_graphdef, stage_vars_slice)
      output = module(
          stage_input,
          stage_segment_id,
          stage_position,
          deterministic=deterministic,
          model_mode=model_mode,
      )
      if isinstance(output, tuple) and len(output) == 2 and output[1] is None:
        output, _ = output
      _, new_vars_slice = nnx.split(module)
      return output, new_vars_slice

    vmapped_run_stage = nnx.vmap(
        run_stage_vmap_fn,
        in_axes=(None, 0, 0, 0, 0),
        out_axes=0,
        spmd_axis_name="stage",
    )

    stages_output, new_stage_vars_pytree = vmapped_run_stage(
        self.single_layer_graphdef,
        stage_vars_pytree,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
    )

    def update_state(full_state_pytree, new_stage_state_pytree, repeat_ids):
      stage_ids = jnp.arange(self.num_stages)

      def scatter_leaf(full_array, update_slice):
        return full_array.at[repeat_ids, stage_ids].set(update_slice)

      return jax.tree.map(scatter_leaf, full_state_pytree, new_stage_state_pytree)

    new_full_state = update_state(state, new_stage_vars_pytree, repeat_ids)
    new_loop_state = self.get_new_loop_state(stages_output, loop_state)
    return new_loop_state, new_full_state

  def get_pipeline_remat_policy(self):
    if self.config.remat_policy == "custom":
      return self.remat_policy

    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    else:
      remat_policy = save_input_policy
    return remat_policy

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      partition_spec=None,
  ) -> jnp.ndarray:

    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        )
    )
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
    if positions is not None:
      positions = jax.lax.with_sharding_constraint(positions, ag_sharding)
      positions = positions.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
    if segment_ids is not None:
      segment_ids = jax.lax.with_sharding_constraint(segment_ids, ag_sharding)
      segment_ids = segment_ids.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    loop_state = self.init_states(inputs)

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    def run_iteration_scannable(merged_carry):
      scan_self, loop_state = merged_carry
      new_loop_state, new_layers_state = scan_self.run_one_iteration(
          loop_state, positions, segment_ids, deterministic, model_mode
      )
      nnx.update(scan_self.layers, new_layers_state)
      return (scan_self, new_loop_state), None

    if self.config.set_remat_policy_on_pipeline_iterations:
      run_iteration_rematted = nnx.remat(
          run_iteration_scannable,
          prevent_cse=not self.config.scan_pipeline_iterations,
          policy=self.get_pipeline_remat_policy(),
      )
    else:
      run_iteration_rematted = run_iteration_scannable

    if self.config.scan_pipeline_iterations:
      run_all_iterations_scanned = nnx.scan(
          run_iteration_rematted,
          length=total_iterations,
          in_axes=nnx.Carry,
      )
      (updated_self, final_loop_carry), _ = run_all_iterations_scanned((self, loop_state))
      _, new_state = nnx.split(updated_self)
      nnx.update(self, new_state)
      loop_state = final_loop_carry

    else:
      graphdef, state = nnx.split(self)

      @functools.partial(
          jax.remat,
          prevent_cse=True,
          policy=self.get_pipeline_remat_policy(),
      )
      def jax_remat_body(state_pytree, loop_state_in, rngs: nnx.Rngs):
        model = nnx.merge(graphdef, state_pytree)
        new_loop_state, new_layers_state = model.run_one_iteration(
            loop_state_in,
            positions,
            segment_ids,
            deterministic,
            model_mode,
            rngs=rngs,
        )
        model.layers = new_layers_state
        _, new_state_pytree = nnx.split(model)
        return new_state_pytree, new_loop_state

      current_state_pytree = state
      rngs = self.rngs
      for _ in range(total_iterations):
        rngs, iter_rngs = rngs.fork()
        new_state_pytree, loop_state = jax_remat_body(current_state_pytree, loop_state, iter_rngs)
        current_state_pytree = new_state_pytree

      nnx.update(self, current_state_pytree)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])

    final_output = jnp.reshape(
        final_output, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim)
    )

    return final_output


PipelineToLinen = nnx_wrappers.to_linen_class(
    Pipeline,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
