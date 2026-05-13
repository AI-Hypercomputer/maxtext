# Copyright 2023-2026 Google LLC
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

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining."""

import functools
from typing import Any

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from aqt.jax.v2 import aqt_tensor
from flax import linen as nn
from flax import nnx
from maxtext.layers import initializers
from maxtext.layers.nnx_wrappers import is_linen_initializing, to_linen_class

from maxtext.common.common_types import Config, MODEL_MODE_TRAIN, ShardMode
from maxtext.utils.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)
from maxtext.utils import pipeline_utils


def _is_static_param(path, v):
  """Predicate matching nnx.Param and FP8 _overwrite_with_gradient variables.

  Used throughout the pipeline to split state into trainable params vs other state.
  Must be consistent everywhere to prevent tree structure mismatches.
  """
  return isinstance(v, nnx.Param) or type(v).__name__ == "_overwrite_with_gradient"


def _advance_rng_state(state, iteration):
  """Fold loop_iteration into all RNG keys to produce unique dropout masks per scan step.

  jax.lax.scan has no split_rngs mechanism (unlike Linen's nn.scan), so every
  iteration would otherwise see the same dropout mask. This mirrors the effect
  of ``nn.scan(split_rngs={"random": True})`` from the Linen pipeline.

  Only typed PRNG key variables (``RngKey``) are folded. RNG counters
  (``RngCount``) are uint32 arrays and must be left untouched — calling
  ``jax.random.fold_in`` on raw uint32 data triggers a PRNG-impl shape
  mismatch (e.g. shape ``(N, 2)`` vs ``unsafe_rbg`` expecting ``(4,)``).

  Args:
    state: An ``nnx.State`` (or partition thereof) that may contain
        ``nnx.RngState`` variable entries whose ``.value`` is a JAX PRNG key.
    iteration: A scalar integer (the loop counter) folded into each key via
        ``jax.random.fold_in``.

  Returns:
    A new state with the same tree structure, where every typed PRNG key
    entry has a unique key derived from the original key and *iteration*.
  """

  def _fold_if_rng(x):
    if isinstance(x, nnx.Variable) and issubclass(x.type, nnx.RngState):
      val = x.value
      # Only fold typed PRNG keys (RngKey). Skip uint32 RNG counters
      # (RngCount) — fold_in would try to wrap them with the default PRNG
      # impl and fail on shape mismatch after vmap batching.
      if jax.dtypes.issubdtype(val.dtype, jax.dtypes.prng_key):
        # fold_in requires a scalar key (shape ()). After nnx.vmap over
        # stages and repeats, keys are batched arrays of shape e.g.
        # (num_repeats, num_stages). Nest jax.vmap over each batch
        # dimension so fold_in sees individual scalar keys.
        def folded(k):
          return jax.random.fold_in(k, iteration)

        for _ in range(val.ndim):
          folded = jax.vmap(folded)
        return x.replace(value=folded(val))
    return x

  return jax.tree.map(_fold_if_rng, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def is_spec_leaf(x):
  """Predicate matching leaves in the bsw_pps treedef, which can be either P or None (if no sharding)."""
  return isinstance(x, P) or x is None


class NNXPipelineBase(nnx.Module):
  """Base module for NNX pipeline parallelism. Contains shared utilities."""

  def _setup_pipeline_attributes(self):
    """Initializes the configuration, calculating num_stages, delay, axes, and partition specs."""
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    self.batch_axis_name = "activation_batch"
    self.seq_len_axis_name = "activation_length"
    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    self.stages_in_logical = ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.stages_in_sharding = (
        NamedSharding(self.mesh, self.stages_in_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )

    self.state_io_logical = ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.state_io_spec = logical_to_mesh_axes(self.state_io_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.state_io_sharding = (
        NamedSharding(self.mesh, self.state_io_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    self.input_sharding = (
        create_sharding(
            self.mesh,
            (None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    self.output_sharding = (
        create_sharding(
            self.mesh,
            (self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )

  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    # Return the number of iterations it takes for microbatch 0 to finish a repeat
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    # Return the number of iterations it takes for microbatch 0 to finish the last stage of the last repeat
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def _maybe_shard_with_logical(self, inputs, logical_axes):
    """Wrapper of maybe_shard_with_logical"""
    return maybe_shard_with_logical(
        inputs,
        logical_axes,
        shard_mode=self.config.shard_mode,
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    """Wrapper of maybe_shard_with_name"""
    return maybe_shard_with_name(
        inputs,
        sharding_name,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """
    Construct stages_in: the global array that is operated on for this iteration, shape same as
    shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from
    state_io or an old one from circ_storage
    """
    # Setup potential input from state_io, which has a rotating microbatch index (size of microbatches_per_stage)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    if self.use_circ_storage:
      # Setup potential input from circ_storage, which also has a rotating index for microbatch,
      # size of num_microbatches
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      # The last stage immediately flows into the first stage, use this rotated shift instead of circular storage
      circular_stage_in = shift

    # For early loop iterations we grab a new input for stage 0 from the state_io. Once each microbatch has left
    # state_io we instead grab from the last stage's output (possibly buffered when num_microbatches > num_stages, e.g.
    # from circ_storage).
    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)
    first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)

    # Note that first_stage_in may correspond to bubble computation during the last few iterations.
    # However, these bubble computation results remain in the shift buffer (do not make it back to state_io) and are
    # thus discarded / not returned.
    # The final returned output is stored in the state_io, which has the appropriate total size of num_microbatches. The
    # state_io will not contain bubble results at the end of the last iteration.

    def select_state_or_input(first_stage_in, shift):
      # Selects input for stage 0, shift for other stages
      return jnp.where(
          jax.lax.broadcasted_iota("int32", shift.shape, 0, out_sharding=self.stages_in_sharding) == 0,
          first_stage_in,
          shift,
      )

    # Selects input (from stream_io) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(first_stage_in, shift)
    return self._maybe_shard_with_logical(stages_in, self.stages_in_logical)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and
    non-circular"""
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is 1 behind due to bubble, etc for other stages
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatches_processed = self._maybe_shard_with_name(microbatches_processed, NamedSharding(self.mesh, P("stage")))
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline.

    Saves two named tensors during jax.checkpoint recomputation:
      - "iteration_input": routed microbatch data entering the decoder
      - "decoder_layer_input": input to the decoder layer itself
    Everything else (including BSW) is recomputed during backward to save memory.
    """
    if self.config.remat_policy == "custom":
      return self.remat_policy
    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    return save_input_policy

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    """Removes 'fsdp' and 'fsdp_transpose' from physical partition spec."""
    if isinstance(pps, P):
      new_spec = []
      # Iterate through each axis in the original PartitionSpec.
      for axis in pps:
        if axis is None:
          new_spec.append(None)
        elif isinstance(axis, str):
          # If the axis is 'fsdp', replace it with None to signify replication.
          if axis not in ("fsdp", "fsdp_transpose"):
            new_spec.append(axis)
          else:
            new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          # If the axis is a collection, filter out 'fsdp'.
          new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else:
          raise ValueError(f"Unsupported_axis_type: {type(axis)}")
        # Return a new sharding object with the modified spec.
      return P(*new_spec)
    return pps

  def init_states(self, inputs):
    """Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
    Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

    Returns a dictionary with properties
      shift: zeros shape [num_stages, micro_size, sequence, embed]
      prev_outputs: same shape as shift, only used when pipeline_delay_activation_forwarding is set to true, else None
      state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
      circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed] when needed, else None
      circ_storage_mover: zeros[num_stages, micro_size, sequence, embed] when needed, else None
      loop_iteration: scalar set initially to 0
      bsw: pytree of identical structure as weights with leaf arrays leading dimension of num_repeats replaced by 2, e.g.
        a leaf of shape [num_repeats, stages, mlp, embed] is mapped to [2, num_stages, mlp, embed].
    """
    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    # Prev outputs has the same shape of the output (and shift)
    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
    else:
      prev_outputs = None

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs
    #   as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(
        inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:], out_sharding=self.state_io_sharding
    )

    # We shard the pipeline_microbatch_size axis by data/fsdp, not num_microbatches since those are looped over.
    state_io = self._maybe_shard_with_logical(state_io, self.state_io_logical)

    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only
    # needed when num_microbatches > num_stages, else instead the final stage will immediately pass to the first without
    # additional storage.
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed].
    # Note that this shape is a factor of num_stages larger than necessary - each stage holds the global batch, but only
    # stage 0 holds the real activations (since it will use them), the rest hold dummy ones. This amount of storage
    # [global_batch, sequence, embed] is fine as long as there is some amount of additional sharding axes, e.g. FSDP,
    # TP, DP (e.g. there are many devices that shard stage 0)
    # We may look into alternatives using less storage if this becomes an issue (ideas in b/347603101).
    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage with one buffer iteration
    # of delay circ_storage_mover shape is same as shift: [num_stages, micro_size, sequence, embed]
    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype, out_sharding=self.state_io_sharding)
      circ_storage_mover = shift
    else:
      circ_storage = None
      circ_storage_mover = None

    return {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": circ_storage_mover,
        "loop_iteration": 0,
        "prev_outputs": prev_outputs,
    }

  def shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False):
    """Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at
    the specified dimension."""
    placeholder = None if self.config.shard_mode == ShardMode.EXPLICIT else P.UNCONSTRAINED
    if physical_partition_spec is None:
      dims_mapping = [placeholder] * x.ndim
    else:
      physical_partition_spec = self._remove_fsdp_from_physical_partition_spec(physical_partition_spec)
      dims_mapping = list(physical_partition_spec)
      # If not a stage weight, we handle the repeat dimension offset
      if not is_stage_weight:
        dims_mapping = [placeholder] * (dim + 1) + dims_mapping[dim:]  # inflat one dimension for num_repeats
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    # We add reduced rule only when pspec is given for a stage weight
    if physical_partition_spec and is_stage_weight and self.config.shard_mode == ShardMode.EXPLICIT:
      batch_mesh_axis = ["data", "fsdp"]
      reduced_mark = [mesh_axis for mesh_axis in batch_mesh_axis if self.mesh.shape[mesh_axis] > 1]
      pspec = P(*dims_mapping, reduced=set(reduced_mark))
    else:
      pspec = P(*dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
    return self._maybe_shard_with_name(x, sharding)

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.
    Args:
      weights: Per-stage data to be gathered from.
      repeat_ids: Integer tensor of shape [num_stages], the repeats of the stages.
      repeat_dim_in_weights: The dimension in weights where repeat_ids are applied. The output will not
        have this dimension.
      stages_dim_in_weights: The dimension in weights that represents parallel stages.

    Returns:
      The per-stage gathered values. The shape is weights.shape but with repeat_dim_in_weights
        removed.
    """

    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
    weights = self.shard_dim_by_stages(
        weights, stages_dim_in_weights, physical_partition_spec=physical_partition_spec, is_stage_weight=False
    )
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    return self.shard_dim_by_stages(
        stage_weights, gathered_weights_stage_dim, physical_partition_spec=physical_partition_spec, is_stage_weight=True
    )

  def vmap_gather(self, xs, ids, ids_dim):
    """Use vmap to implement a stage-wise sharded gather.

    The stages share the same input, but they have different offsets.

    Args:
      xs: Data shared by all stages, to be gathered from.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. In the output, this
        dimension will be [num_stages], since each stage gets one slice.

    Returns:
      The per-stage gathered values. The shape is xs.shape but with ids_dim size
        replaced with [num_stages].
    """
    xs = jnp.asarray(xs)
    ndim = xs.ndim

    def _gather_one(x, i):
      idx = tuple(i if d == ids_dim else slice(None) for d in range(ndim))
      replicated_sharding = NamedSharding(self.mesh, P())
      return x.at[idx].get(out_sharding=replicated_sharding)

    ids = self.shard_dim_by_stages(ids, 0, physical_partition_spec=None)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0, physical_partition_spec=None)

  def get_new_loop_state(self, output, loop_state):
    """
    Update the various buffers given the output of the most recent iteration
    * state_io: rotates left/up by 1 (the whole created in the last slot is filled with the most recent pipeline output)
       * Pushing inputs up from top of state_io into first stage of shift
       * Pulling outputs up from last stage of shift into bottom of state_io
    * shift: rotate output (or prev_outputs if using delay) right/down by 1 - we imagine the pipeline moves to
               right/down
    * circ_storage: pushes circ_storage_mover (the output of the previous iteration) into rotating index of circ_storage
    * circ_storage_mover: assigned to rotated output and pushed into circ_storage on the next iteration
    * prev_outputs: is set to the current output
    """
    old_state_io = loop_state["state_io"]
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    old_prev_outputs = loop_state["prev_outputs"]

    @jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
    def _rotate_right(arr):
      # we use +1 for right shifting
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

    @jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
    def _shift_right(arr):
      stage_idx = jax.lax.axis_index("stage")
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)

    # Shift either rotates or shifts depending on if the last stage immediately must send to first or not
    # For non-circular pipelines, the last stage does not need to send to first
    # For circular pipelines with #micro = #stages, last stage immediately sends to first
    # For circular pipelines with #micro > stages (circ_storage), last stage sends to circ storage
    def _update_shift(output_in):
      if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
        return _shift_right(output_in)  # last stage does not have to send to first immediately
      else:
        return _rotate_right(output_in)  # last stage must immediately send to first

    if self.config.pipeline_delay_activation_forwarding:
      new_shift = _update_shift(old_prev_outputs)
      new_prev_outputs = output
    else:
      new_shift = _update_shift(output)
      new_prev_outputs = None

    if self.use_circ_storage:
      # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
      # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped
      # compute/async transfers
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = jnp.expand_dims(_rotate_right(circ_storage_mover_in), 1)
        # We rotate the pushing index into circ storage, and ensure that microbatch 0 lands in index 0
        offset = (
            loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1
        ) % self.config.num_pipeline_microbatches
        # previous output - using circ_storage_mover before it is updated
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    # Rotate stream_io left/up by 1 on rotating micro/stage index (stream_buf_idx), replacing the last/bottom with the
    # last stage output
    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _rotate_left(arr, stage_size):
      # we use -1 for left shifting
      perm = [(i, (i - 1) % stage_size) for i in range(stage_size)]
      return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

    def _shift_left(arr, stage_size, output):
      stage_idx = jax.lax.axis_index("stage")
      arr = _rotate_left(arr, stage_size)
      return jnp.where(stage_idx == stage_size - 1, output, arr)

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, P()),
        out_specs=self.state_io_spec,
    )
    def _update_state_io(state_in, stream_slice, output, stream_buf_idx):
      # Shift the current slice to the left, then fill the last stage with the final output.
      stage_size = jax.lax.axis_size("stage")
      stream_slice = _shift_left(stream_slice, stage_size, output)
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output, stream_buf_idx)

    return {
        "state_io": new_state,
        "shift": new_shift,
        "circ_storage": new_circ_storage,
        "circ_storage_mover": new_circ_storage_mover,
        "loop_iteration": loop_iteration + 1,
        "prev_outputs": new_prev_outputs,
    }

  def permute_output_micro_per_stage_dim(self, output):
    """
    Permutes the output microbatches to match the input order.

    The pipeline execution introduces a delay (bubble) for each stage.
    Consequently, the first microbatch (index 0) finishes after a certain number of iterations
    and lands at a shifted position in the output buffer (`state_io`).
    This function calculates the offset (`microbatch_0_idx`) and permutes the output
    along the microbatch dimension so that microbatch 0 is at index 0, microbatch 1 at index 1, etc.
    """
    # The first real output (microbatch 0) takes a certain amount of loop iterations to finish and be pushed to
    # state_io - it will land on a different index of state_io depending on the number of iterations.
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    return output[:, permutation]

  def realign_output_microbatches(self, output):
    """Reorders the output tensor to reverse the circular shifts applied during execution.

    Because the pipeline operates circularly, the output microbatches are shifted
    out of order by the time the final stage is completed. This rolls them back
    into their original sequential layout.
    """
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    output = jnp.roll(output, shift=-microbatch_0_idx, axis=1)
    return self._maybe_shard_with_logical(output, self.state_io_logical)

  # ---- NNX-specific methods ----

  def get_weight_sharding(self, *init_args):
    """Returns a pytree of logical-name PartitionSpecs mirroring the params state."""

    state = nnx.state(self.layers, _is_static_param)

    def get_spec(x):
      if not isinstance(x, nnx.Variable):
        # Non-VariableState leaf (e.g., nnx.Empty): treat as replicated.
        return P()
      # _overwrite_with_gradient variables (FP8 amax history / scales) carry no
      # partition metadata; return replicated to keep the tree aligned.
      if x.type.__name__ == "_overwrite_with_gradient":
        return P()
      # AQT QTensor values are a pytree wrapping quantized data; mirror the
      # skip-list in variable_to_logically_partitioned (initializers.py:81-83).
      if isinstance(x.value, aqt_tensor.QTensor):
        return P()
      if isinstance(x.value, nn.spmd.LogicallyPartitioned):
        # Dead in the NNX-first flow; retained as a forward-compat guard in
        # case a Linen-wrapped param is ever merged into this module.
        return x.value.partitions
      metadata = x.get_metadata()
      # Try each known metadata key in order; first hit wins.
      sharding = metadata.get("out_sharding")
      if sharding is None:
        sharding = metadata.get("sharding_names")
      if sharding is None:
        sharding = metadata.get("sharding")
      # Already a PartitionSpec - pass through.
      if isinstance(sharding, P):
        return sharding
      # Happy path: tuple/list of logical axis names from nnx.Param(sharding=...).
      if isinstance(sharding, (tuple, list)):
        return P(*sharding)
      # Non-PartitionSpec wrapper with an explicit ``.spec`` attribute (kept
      # for forward compatibility with future Flax wrapper types).
      if sharding is not None and hasattr(sharding, "spec"):
        return sharding.spec
      # Fallback: replicated sharding (valid for shard_map, unlike None).
      return P()

    return jax.tree.map(get_spec, state, is_leaf=lambda x: isinstance(x, nnx.Variable))

  def get_main_vmap_func_for_iterations(self):
    def func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      module = nnx.merge(graph, state)
      out = module(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)
      return out, nnx.state(module)

    return nnx.vmap(
        func_to_vmap,
        in_axes=(None, 0, 0, 0, 0, None, None),
        out_axes=(0, 0),
        spmd_axis_name=self.spmd_axis_name,
    )

  @staticmethod
  def _stamp_at_current_trace(weights):
    """Pass each leaf through a no-op dynamic_slice so JAX creates new arrays
    at the *current* trace level.  This prevents trace-level mismatches when
    outer-trace values (e.g. closed-over by ``jax.lax.scan``) are later fed
    into ``nnx.merge`` inside the scan body.

    The operation is semantically an identity: ``x[0 : x.shape[0]]`` along
    axis 0, which XLA will optimise away.
    """

    def _identity_slice(x):
      if hasattr(x, "shape") and len(x.shape) > 0:
        return jax.lax.dynamic_slice_in_dim(x, 0, x.shape[0], axis=0)
      return x  # scalars / non-array leaves pass through unchanged

    return jax.tree.map(_identity_slice, weights)

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    """
    Gets the current weights used for one iteration. Outputs a pytree whose arrays have leading dimension of stages, e.g.
    {'mlp': 'wo': [stages, mlp, embed]}. Stage 0 will use the 0th index of this pytree, Stage 1 the 1st index, etc.
    For non-circular pipelines, this simply returns all weights - every weight is used in every iteraiton. However
    for circular pipelines each stage grabs only the weights corresponding to the current repeat.
    """
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(
          pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    # Stamp weights at the current trace level so that nnx.merge inside
    # func_to_vmap does not hit a trace-level mismatch when running under
    # jax.lax.scan (the weights may originate from an outer trace).
    return self._stamp_at_current_trace(pipeline_weights)

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    """
    all-gathers the variables over fsdp if fsdp is in the logical partition spec.
    """
    if logical_partition_spec is None:
      return variables

    def _gather_leaf(var, spec):
      if spec is None:
        return var
      physical = logical_to_mesh_axes(spec, self.mesh, rules=self.config.logical_axis_rules)
      no_fsdp = self._remove_fsdp_from_physical_partition_spec(physical)
      sharding = NamedSharding(self.mesh, no_fsdp)
      if isinstance(var, nnx.Variable):
        var.value = self._maybe_shard_with_name(var.value, sharding)
        return var
      return self._maybe_shard_with_name(var, sharding)

    # nnx.Variable and PartitionSpec are JAX pytree nodes — treat them as leaves
    # so the two trees align at the dict level. None must also be a leaf to avoid
    # being treated as an empty container (0 children) vs the Variable's 1 child.
    def is_leaf(x):
      return isinstance(x, (nnx.Variable, P)) or x is None

    return jax.tree.map(_gather_leaf, variables, logical_partition_spec, is_leaf=is_leaf)

  def get_logical_spec_repeats_removed(self, full_logical):
    """Returns a new logical spec with 'circular_repeats' removed."""
    if full_logical is None or self.config.num_pipeline_repeats == 1:
      return full_logical

    def _remove_from_spec(spec):
      if not isinstance(spec, P):
        return spec
      if spec and (spec[0] == "circular_repeats" or spec[0] is None):
        return jax.sharding.PartitionSpec(*spec[1:])
      return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])

    return jax.tree.map(_remove_from_spec, full_logical, is_leaf=lambda x: isinstance(x, P))

  def __init__(
      self,
      config: Config,
      stage_factory: Any,
      mesh: Mesh,
      remat_policy: Any = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    self._setup_pipeline_attributes()

    def build_batched_rngs(shape):
      kwargs = {}
      rng_state = nnx.state(rngs, nnx.RngState)
      leaves, _ = jax.tree_util.tree_flatten_with_path(rng_state)
      for path, key in leaves:
        stream_name = getattr(path[0], "key", str(path[0]))
        if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
          key = jax.random.key(key)
        num_splits = int(np.prod(shape))
        flat_keys = jax.random.split(key, num_splits)
        kwargs[stream_name] = flat_keys.reshape(shape + key.shape)
      return nnx.Rngs(**kwargs)

    def create_stage_fn(r):
      stage = stage_factory(r)
      # Split into (GraphDef, Param State, Rest of State)
      return nnx.split(stage, nnx.Param, ...)

    vmap_stages = nnx.vmap(
        create_stage_fn,
        in_axes=0,
        out_axes=(None, 0, 0),
        spmd_axis_name=self.spmd_axis_name,
        transform_metadata={nnx.PARTITION_NAME: "layers"},
    )

    if self.config.num_pipeline_repeats > 1:
      vmap_repeats = nnx.vmap(
          vmap_stages,
          in_axes=0,
          out_axes=(None, 0, 0),
          transform_metadata={nnx.PARTITION_NAME: "circular_repeats"},
      )
      batched_rngs = build_batched_rngs((self.config.num_pipeline_repeats, self.num_stages))
      graphdef, params, rest = vmap_repeats(batched_rngs)
    else:
      batched_rngs = build_batched_rngs((self.num_stages,))
      graphdef, params, rest = vmap_stages(batched_rngs)

    # Merge the batched states back into the module
    self.layers = nnx.merge(graphdef, params, rest)


class NNXPipeline(NNXPipelineBase):
  """Original Pipeline implementation adapted for NNX."""

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(
          pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    return self._stamp_at_current_trace(pipeline_weights)

  def get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None):
    """Fetches the weights for the current repeat from the stages."""
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    def gather_weights_for_stages_in(w, spec=None):
      if w is None:
        return None
      return self.vmap_parallel_gather(
          w, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1, physical_partition_spec=spec
      )

    if physical_partition_spec is None:
      return jax.tree.map(gather_weights_for_stages_in, weights)

    _, weights_params, weights_rest = nnx.split(weights, _is_static_param, ...)

    spec_leaves = jax.tree_util.tree_leaves(physical_partition_spec, is_leaf=is_spec_leaf)
    assert len(spec_leaves) == len(jax.tree_util.tree_leaves(weights_params)), (
        f"Spec tree leaf count ({len(spec_leaves)}) != weights tree leaf count "
        f"({len(jax.tree_util.tree_leaves(weights_params))}). "
        "The _is_static_param predicate may have diverged between get_weight_sharding and __call__."
    )
    spec_iter = iter(spec_leaves)
    gathered_params = jax.tree.map(
        lambda w: gather_weights_for_stages_in(w, next(spec_iter)),
        weights_params,
    )

    # Non-params gathered without sharding hints.
    gathered_rest = jax.tree.map(gather_weights_for_stages_in, weights_rest)

    return nnx.State.merge(gathered_params, gathered_rest)

  def run_one_iteration(
      self,
      loop_state,
      pipeline_weights_graph,
      pipeline_weights_state,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      logical_partition_spec=None,
  ):
    """Executes the logic for a single microbatch iteration, including routing inputs and weights, and advancing buffers."""
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)
    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    vmap_func = self.get_main_vmap_func_for_iterations()

    stage_weights_state = self.get_current_stage_weights(
        pipeline_weights_state, loop_iteration, physical_partition_spec=physical_partition_spec
    )

    # Strip nnx.Variable wrappers to raw arrays before nnx.vmap.
    # When called inside jax.lax.scan, outer-scope Variables have
    # _can_update=False, causing check_consistent_aliasing to reject them.
    # nnx.merge inside func_to_vmap creates fresh Variables from raw values.
    stage_weights_state = jax.tree.map(
        lambda x: x.value if isinstance(x, nnx.Variable) else x,
        stage_weights_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    stages_output, updated_stage_weights_state = vmap_func(
        pipeline_weights_graph,
        stage_weights_state,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
        deterministic,
        model_mode,
    )

    if self.config.scan_layers:
      stages_output = stages_output[0]

    if self.config.num_pipeline_repeats > 1:
      _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

      def _scatter_update(fw, uw, spec=None):
        if fw is None or uw is None:
          return fw

        def _update_one_stage(f_s, u_s, r_id):
          return jax.lax.dynamic_update_slice_in_dim(f_s, jnp.expand_dims(u_s, 0), r_id, axis=0)

        r_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
        updated_fw = jax.vmap(_update_one_stage, in_axes=(1, 0, 0), out_axes=1)(fw, uw, r_ids)
        return self.shard_dim_by_stages(updated_fw, 1, physical_partition_spec=spec, is_stage_weight=False)

      pipeline_weights_state = jax.tree.map(_scatter_update, pipeline_weights_state, updated_stage_weights_state)
    else:
      pipeline_weights_state = updated_stage_weights_state

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state, pipeline_weights_state

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,  # Pytree of sharding specifications of the weights (aka self.layers.variables)
  ) -> jnp.ndarray:
    """The main method that maps the series of decoder layer inputs to final layer outputs.
    Has the same signature of a single decoder layer, and expects the same shapes, e.g. the inputs should have shape
    [global_batch], and internally this will be reshapped into microbatches.
    """
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
    if positions is not None:
      positions = self._maybe_shard_with_name(positions, ag_sharding).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
    if segment_ids is not None:
      segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    loop_state = self.init_states(inputs)

    # MISS-1: Short-circuit during Linen init (to_linen_class wrapper path).
    # NNX modules eagerly initialize weights in __init__, so the full scan is
    # unnecessary during init — Linen only needs the output shape/dtype.
    # Returns zeros matching the pipeline output shape.
    # Assumption: output shape is (micro_batch_size, max_target_length, emb_dim).
    # This matches decoder-only models; update if pipeline is used for other architectures.
    if is_linen_initializing():
      return jnp.zeros(
          (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
          dtype=inputs.dtype,
      )

    # Each microbatch should go through each stage (with repeats) - so there is num_micro * (num_stages * repeats)
    # compute to perform
    # Each iteration is vmapped by num_stages, so the number of iterations should be
    # num_micro * num_stages * repeats / num_stages = num_micro * repeats
    # However due to the pipeline bubble some iterations process less than num_stages microbatches. It takes
    # num_micro * repeat iterations for the last microbatch to start the final repeat, then an additional
    # num_stages - 1 to finish the final repeat.
    # Thus the total iterations is num_micro * repeat + num_stages - 1, & we may consider the num_stages - 1 as bubble.
    # The bubble doubles when we use forwarding delay.
    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    logical_partition_spec = self.get_logical_spec_repeats_removed(logical_partition_spec)

    layers_graph, layers_state = nnx.split(self.layers)

    def is_lp(x):
      return isinstance(x, nn.spmd.LogicallyPartitioned)

    def unbox_val(x):
      return x.value if is_lp(x) else x

    layers_state = jax.tree.map(unbox_val, layers_state, is_leaf=is_lp)

    # Split BEFORE all_gather_over_fsdp so the tree handed to it aligns with
    # logical_partition_spec. logical_partition_spec comes from get_weight_sharding
    # which filters to the same _is_static_param predicate (nnx.Param +
    # _overwrite_with_gradient), so layers_params and the spec tree are
    # structurally identical by construction. Passing the unfiltered layers_state
    # would include dropout/RNG state that the spec tree lacks, causing
    # jax.tree.map to raise "Mismatch custom node data". Mirrors Linen
    # where all_gather_over_fsdp operates on self.layers.variables (the params collection only).
    _, layers_params, layers_metrics, layers_mutables = nnx.split(layers_state, _is_static_param, nnx.Intermediate, ...)

    # layers_mutables catch-all should contain ONLY RngState variables (RngKey/RngCount).
    # If non_trainable state (e.g. BatchStat) appears here,
    # it is being carried through scan instead of broadcast.
    # NOTE: is_leaf stops jax.tree.leaves from traversing *into* Variable nodes,
    # so we see actual Variable instances (not raw arrays).
    assert all(
        isinstance(v, nnx.RngState)
        for v in jax.tree.leaves(layers_mutables, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ), (
        "Non-RngState variable found in layers_mutables catch-all partition. "
        "Only RngState variables (RngKey/RngCount) should be present."
    )

    if self.config.pipeline_fsdp_ag_once:
      layers_params = self.all_gather_over_fsdp(layers_params, logical_partition_spec)

    def scan_body(carry, _):
      current_loop_state, current_layer_mutables = carry
      # Fold loop_iteration into RNG keys so each scan step gets a unique
      # dropout mask — mirrors Linen's nn.scan(split_rngs={"random": True}).
      iteration = current_loop_state["loop_iteration"]
      advanced_mutables = _advance_rng_state(current_layer_mutables, iteration)
      current_layer_state = nnx.State.merge(layers_params, layers_metrics, advanced_mutables)

      new_loop_state, new_layer_state = self.run_one_iteration(
          current_loop_state,
          layers_graph,
          current_layer_state,
          positions,
          segment_ids,
          deterministic,
          model_mode,
          logical_partition_spec,
      )

      _, _, new_layer_metrics, new_layer_mutables = nnx.split(new_layer_state, _is_static_param, nnx.Intermediate, ...)
      return (new_loop_state, new_layer_mutables), new_layer_metrics

    if self.config.set_remat_policy_on_pipeline_iterations:
      scan_body = jax.checkpoint(
          scan_body, policy=self.get_pipeline_remat_policy(), prevent_cse=not self.config.scan_pipeline_iterations
      )

    if self.config.scan_pipeline_iterations:
      (loop_state, final_layer_mutables), stacked_metrics = jax.lax.scan(
          scan_body, (loop_state, layers_mutables), None, length=total_iterations
      )
    else:
      current_carry = (loop_state, layers_mutables)
      metrics_history = []
      for _ in range(total_iterations):
        current_carry, step_metrics = scan_body(current_carry, None)
        metrics_history.append(step_metrics)
      loop_state, final_layer_mutables = current_carry
      stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metrics_history) if metrics_history else layers_metrics

    final_layer_state = nnx.State.merge(layers_params, stacked_metrics, final_layer_mutables)
    nnx.update(self.layers, final_layer_state)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])
    return jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )


class NNXCircularPipeline(NNXPipelineBase):
  """NNX circular pipeline with nested scan and BSW weight caching.

  Uses a nested scan: outer loop over repeats (creates BSW once per repeat),
  inner loop over microbatches (reuses BSW via closure). This reduces FSDP
  all-gathers from total_iterations to num_repeats+1.
  """

  def gather_microbatch_inputs_vmap(self, xs, ids, ids_dim):
    """Slices out the specific sequence inputs (e.g., positions, segments) for the current microbatch."""
    if xs is None:
      return None

    xs = jnp.asarray(xs)
    ndim = xs.ndim

    def _gather_one(x, i):
      idx = tuple(i if d == ids_dim else slice(None) for d in range(ndim))
      positions_sharding = (
          create_sharding(self.mesh, (None, "layers", "activation_length"))
          if self.config.shard_mode == ShardMode.EXPLICIT
          else None
      )
      return x.at[idx].get(out_sharding=positions_sharding)

    return jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)

  def gather_weights_across_stages_vmap(self, weights_state, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    """Uses jax.vmap to dynamically slice and gather weights for specific pipeline repeats."""

    def _gather_repeat_leaf(w_leaf, rep_id):
      if w_leaf is None:
        return None
      return jnp.squeeze(
          jax.lax.dynamic_slice_in_dim(w_leaf, rep_id, 1, axis=repeat_dim_in_weights), axis=repeat_dim_in_weights
      )

    vmap_gather = jax.vmap(_gather_repeat_leaf, in_axes=(stages_dim_in_weights, 0), out_axes=0)
    return jax.tree.map(lambda w: vmap_gather(w, repeat_ids) if w is not None else None, weights_state)

  def from_all_variables_to_repeat_weights(self, weights_state, loop_iteration):
    """Slices out the specific repeat's weights from the full weights state."""
    if self.config.num_pipeline_repeats == 1:
      return weights_state

    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    return self.gather_weights_across_stages_vmap(
        weights_state, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
    )

  def from_repeat_weights_to_bsw(
      self,
      repeat_weights,
      physical_partition_spec,
      axes_to_gather=("fsdp", "fsdp_transpose", "context", "expert"),
      # TODO (chengnuojin) set use_shardmap=true after JAX >= 0.10.0 and use all_gather(..., to='invarying')
      use_shardmap=False,  # using shardmap produces additional reduce-scatter in backward pass
  ):
    """Executes the FSDP-like all-gathers to fully materialize a block of weights for the BSW."""
    axes_to_remove = ["fsdp", "fsdp_transpose", "context"]
    if physical_partition_spec is not None:
      bsw_pps = pipeline_utils.derive_stage_weight_partition_specs(physical_partition_spec, axes_to_remove)
    else:
      bsw_pps = None

    def _from_repeat_weights_to_bsw_shardmap(
        repeat_weights,
        physical_partition_spec,
        axes_to_gather,
    ):
      repeat_weights_pps = jax.tree.map(
          lambda p: P(*p[1:]) if isinstance(p, P) else p,
          physical_partition_spec,
          is_leaf=is_spec_leaf,
      )

      # Dynamically gather the index pytrees for all specified axes
      axis_indices_dict = {
          axis: pipeline_utils.get_mesh_axis_dim_indices(physical_partition_spec, axis) for axis in axes_to_gather
      }

      axis_names = list(axis_indices_dict.keys())
      axis_pytrees = list(axis_indices_dict.values())

      def should_skip_gather(axis_name, path_keys):
        """Defines specific rule-based exceptions for gathering certain axes."""
        if axis_name == "expert" and "MoeBlock_0" in path_keys:
          return True
        # Add more exclusion rules for other axes here if needed in the future
        return False

      # Strip nnx.Variable wrappers via treedef roundtrip (same pattern as
      # get_current_weights_from_bsw). weights_treedef captures Variable nodes;
      # pps_treedef stops at plain P leaves and has the same leaf count by
      # invariant (8) -- both filtered by the same is_static_param predicate
      # upstream. Flatten repeat_weights to raw arrays, rebuild with
      # pps_treedef so the shard_map input tree matches the spec tree, then
      # re-wrap into Variables via weights_treedef on the way out.
      weights_treedef = jax.tree.structure(repeat_weights)
      pps_treedef = jax.tree.structure(repeat_weights_pps, is_leaf=is_spec_leaf)
      weights_leaves = jax.tree.leaves(repeat_weights)
      assert pps_treedef.num_leaves == len(weights_leaves), (
          f"repeat_weights/spec leaf count mismatch: specs={pps_treedef.num_leaves}, " f"weights={len(weights_leaves)}"
      )
      raw_weights = pps_treedef.unflatten(weights_leaves)

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(repeat_weights_pps, None),  # 'None' covers the entire axis_pytrees list
          out_specs=bsw_pps,
          check_vma=False,
      )
      def _shard_map_gather_weights(sharded_weights, indices_pytrees_list):

        def _gather_tensor_along_axes(path, x, *indices):
          path_keys = [getattr(p, "key", str(p)) for p in path]

          # Iterate through the provided axes and their corresponding indices
          for axis_name, axis_idx in zip(axis_names, indices):
            if axis_idx >= 0 and not should_skip_gather(axis_name, path_keys):
              x = jax.lax.all_gather(x, axis_name=axis_name, axis=axis_idx - 1, tiled=True)
          return x

        return jax.tree_util.tree_map_with_path(_gather_tensor_along_axes, sharded_weights, *indices_pytrees_list)

      raw_bsw = _shard_map_gather_weights(raw_weights, axis_pytrees)
      return weights_treedef.unflatten(jax.tree.leaves(raw_bsw))

    def _from_repeat_weights_to_bsw_hint(repeat_weights):
      def _apply_sharding_hint(weight, pspec):
        if pspec is None or weight is None:
          return weight
        sharding_name = NamedSharding(self.mesh, pspec)
        return maybe_shard_with_name(
            weight,
            sharding_name,
            shard_mode=self.config.shard_mode,
            debug_sharding=self.config.debug_sharding,
            extra_stack_level=0,
        )

      spec_leaves = jax.tree_util.tree_leaves(bsw_pps, is_leaf=is_spec_leaf)
      spec_iter = iter(spec_leaves)
      return jax.tree.map(lambda w: _apply_sharding_hint(w, next(spec_iter)), repeat_weights)

    if bsw_pps is None:
      return repeat_weights

    if use_shardmap:
      return _from_repeat_weights_to_bsw_shardmap(repeat_weights, physical_partition_spec, axes_to_gather=axes_to_gather)
    return _from_repeat_weights_to_bsw_hint(repeat_weights)

  def weight_prefetching(self, weights_state, physical_partition_spec, loop_iteration):
    """Prefetch next repeat's weights for the Buffer Sliding Window."""
    nxt_repeat_weights = self.from_all_variables_to_repeat_weights(weights_state, loop_iteration + 1)
    return self.from_repeat_weights_to_bsw(nxt_repeat_weights, physical_partition_spec)

  def fetch_active_stage_weights(self, bsw, loop_iteration, physical_partition_spec=None):
    """The module fetches the actively prefetched weights
    from the Buffer Sliding Window to avoid mid-iteration FSDP all-gathers.
    """
    return self.get_current_weights_from_bsw(bsw, loop_iteration, physical_partition_spec)

  def get_current_weights_from_bsw(self, bsw, loop_iteration, physical_partition_spec):
    """Pulls the fully gathered parameters for the current repeat from the BSW."""
    if bsw[0] is bsw[1]:
      treedef = jax.tree.structure(bsw[0])
      leaves = jax.tree.leaves(bsw[0])
      return treedef.unflatten(leaves)

    bsw_pps = jax.tree.map(self._remove_fsdp_from_physical_partition_spec, physical_partition_spec)
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
    stage0_repeat_id = jnp.maximum(loop_iteration, 0) // self.config.num_pipeline_microbatches

    if bsw_pps is not None:
      # Strip nnx.Variable containers from BSW for shard_map pytree compatibility.
      # BSW has Param(array) nodes at leaves; shard_map specs are plain P() leaves.
      # Treedef roundtrip:
      #   1. Capture bsw_treedef (includes Param nodes) for reconstruction later
      #   2. Flatten BSW leaves (raw arrays extracted from inside Param nodes)
      #   3. Rebuild BSW with pps_treedef (no Param nodes) so it matches bsw_pps
      #   4. Run shard_map on the raw-array BSW
      #   5. Reconstruct nnx.Variable wrappers via bsw_treedef.unflatten
      # Leaf counts match by construction: bsw and bsw_pps are co-derived from
      # the same weight tree (via get_weight_sharding + from_repeat_weights_to_bsw).
      bsw_treedef = jax.tree.structure(bsw[0])

      pps_treedef = jax.tree.structure(bsw_pps, is_leaf=is_spec_leaf)
      bsw0_leaves = jax.tree.leaves(bsw[0])
      bsw1_leaves = jax.tree.leaves(bsw[1])
      # Defensive: both BSW halves and the spec tree must agree on leaf count.
      # Stricter: bsw[0] and bsw[1] must have the same *structure*, not just
      # the same leaf count — they are co-produced by from_repeat_weights_to_bsw
      # called on cur_repeat_weights / nxt_repeat_weights so in practice this
      # always holds, but catching a divergence early beats a confusing
      # shard_map error later.
      assert bsw_treedef == jax.tree.structure(
          bsw[1]
      ), "BSW half-tree structure mismatch: bsw[0] and bsw[1] must be structurally identical but differ."
      assert pps_treedef.num_leaves == len(bsw0_leaves) == len(bsw1_leaves), (
          f"BSW/spec leaf count mismatch: specs={pps_treedef.num_leaves}, "
          f"bsw0={len(bsw0_leaves)}, bsw1={len(bsw1_leaves)}"
      )
      raw_bsw_0 = pps_treedef.unflatten(bsw0_leaves)
      raw_bsw_1 = pps_treedef.unflatten(bsw1_leaves)

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=((bsw_pps, bsw_pps), P("stage")),
          out_specs=bsw_pps,
          check_vma=True,
      )
      # [0]: shard_map passes repeat_id as a (1,)-shaped per-stage slice, not
      # a scalar. raw_bsw leaves are all arrays (the treedef roundtrip above
      # reconstructed pps_treedef with the raw array leaves from bsw), so no
      # None-guard is needed here — matches Linen old_pipeline.py:1134.
      def select_weights_from_bsw(bsw_inner, repeat_id):
        return jax.tree.map(
            lambda x, y: jax.lax.select(repeat_id[0] == stage0_repeat_id, y, x),
            bsw_inner[0],
            bsw_inner[1],
        )

      raw_weights = select_weights_from_bsw((raw_bsw_0, raw_bsw_1), repeat_ids)
      # Reconstruct nnx.Variable wrappers so downstream nnx.State.merge works.
      # raw_weights has pps_treedef structure; re-flatten and unflatten into bsw_treedef.
      weights = bsw_treedef.unflatten(jax.tree.leaves(raw_weights))
    else:
      # Fallback: no partition spec provided (e.g. initialization path where
      # logical_partition_spec is None); use vmap over the repeat dim. NNX
      # Variable wrappers are handled natively by jax.vmap — no treedef
      # roundtrip needed.
      def select_weights_from_bsw(bsw_inner, repeat_id):
        return jax.tree.map(
            lambda x, y: jax.lax.select(repeat_id == stage0_repeat_id, y, x) if x is not None else None,
            bsw_inner[0],
            bsw_inner[1],
        )

      weights = jax.vmap(select_weights_from_bsw, in_axes=((0, 0), 0), out_axes=0)(bsw, repeat_ids)

    return weights

  def run_one_iteration(
      self,
      loop_state,
      bsw,
      pipeline_weights_graph,
      layers_metrics,
      current_layer_mutables,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      logical_partition_spec,
  ):
    """Executes the forward/backward logic for a single microbatch inside the circular pipeline.

    Fetches params from BSW (params-only), gathers metrics/mutables directly for the current
    repeat, merges into full state for the forward pass, then scatter-updates only non-params
    back (params are static in scan and handled by AD/gradient).
    """
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)
    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")

    stages_positions = self.gather_microbatch_inputs_vmap(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = (
        self.gather_microbatch_inputs_vmap(segment_ids, microbatch_ids, 0) if segment_ids is not None else None
    )

    vmap_func = self.get_main_vmap_func_for_iterations()

    # 1. Fetch params from BSW (params-only, tree matches physical_partition_spec)
    stage_params = self.fetch_active_stage_weights(
        bsw,
        loop_iteration,
        physical_partition_spec=physical_partition_spec,
    )

    # 2. Gather non-params (metrics, mutables) for current repeat directly
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)
    if self.config.num_pipeline_repeats > 1:
      stage_metrics = self.gather_weights_across_stages_vmap(
          layers_metrics, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
      )
      stage_mutables = self.gather_weights_across_stages_vmap(
          current_layer_mutables, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1
      )
    else:
      # Stamp at current trace level to avoid nnx.merge trace-level mismatch
      # (layers_metrics is closed over from outer scope in scan).
      stage_metrics = self._stamp_at_current_trace(layers_metrics)
      stage_mutables = current_layer_mutables  # already at scan trace level (from carry)

    # 3. Merge into full state for forward pass
    stage_weights_state = nnx.State.merge(stage_params, stage_metrics, stage_mutables)

    stages_output, updated_stage_weights_state = vmap_func(
        pipeline_weights_graph,
        stage_weights_state,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
        deterministic,
        model_mode,
    )

    if self.config.scan_layers:
      stages_output = stages_output[0]

    # Scatter-back: only update non-params (params are handled by AD/gradient, not carried in scan)
    if self.config.num_pipeline_repeats > 1:

      def _scatter_update(fw, uw):
        if fw is None or uw is None:
          return fw

        def _update_one_stage(f_s, u_s, r_id):
          return jax.lax.dynamic_update_slice_in_dim(f_s, jnp.expand_dims(u_s, 0), r_id, axis=0)

        r_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
        updated_fw = jax.vmap(_update_one_stage, in_axes=(1, 0, 0), out_axes=1)(fw, uw, r_ids)
        return self.shard_dim_by_stages(updated_fw, 1, physical_partition_spec=None, is_stage_weight=False)

      # Extract non-params from updated stage state
      _, _, updated_stage_metrics, updated_stage_mutables = nnx.split(
          updated_stage_weights_state, _is_static_param, nnx.Intermediate, ...
      )
      updated_stage_non_params = nnx.State.merge(updated_stage_metrics, updated_stage_mutables)
      current_non_params = nnx.State.merge(layers_metrics, current_layer_mutables)
      new_layer_state = jax.tree.map(_scatter_update, current_non_params, updated_stage_non_params)
    else:
      # Filter to non-params for consistency with num_pipeline_repeats > 1 path
      _, _, else_metrics, else_mutables = nnx.split(updated_stage_weights_state, _is_static_param, nnx.Intermediate, ...)
      new_layer_state = nnx.State.merge(else_metrics, else_mutables)

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state, new_layer_state

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,
  ) -> jnp.ndarray:
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )

    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
    if positions is not None:
      positions = self._maybe_shard_with_name(positions, ag_sharding).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
    if segment_ids is not None:
      segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    loop_state = self.init_states(inputs)

    # NNX modules eagerly initialize weights in __init__, so the full scan is
    # unnecessary during init — Linen only needs the output shape/dtype.
    # Returns zeros matching the pipeline output shape.
    # Assumption: output shape is (micro_batch_size, max_target_length, emb_dim).
    # This matches decoder-only models; update if pipeline is used for other architectures.
    if is_linen_initializing():
      return jnp.zeros(
          (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
          dtype=inputs.dtype,
      )

    # Two spec variants needed:
    # - Full spec (with circular_repeats axis) -> BSW creation inside scan_body via
    #   from_all_variables_to_repeat_weights + from_repeat_weights_to_bsw.
    #   from_repeat_weights_to_bsw's derive_stage_weight_partition_specs drops the
    #   first dim (repeat), so the input must still have it.
    # - Stripped logical spec (circular_repeats removed) -> BSW consumption via
    #   run_one_iteration. get_current_weights_from_bsw uses _remove_fsdp_from_
    #   physical_partition_spec, which only removes fsdp; the repeat axis must
    #   already be gone to match the 3-dim BSW arrays (repeat gathered away by
    #   from_all_variables_to_repeat_weights).
    physical_partition_spec_full = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    logical_partition_spec_stripped = pipeline_utils.strip_pipeline_repeat_logical_axis(logical_partition_spec)

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)

    layers_graph, layers_state = nnx.split(self.layers)

    def is_lp(x):
      return isinstance(x, nn.spmd.LogicallyPartitioned)

    def unbox_val(x):
      return x.value if is_lp(x) else x

    layers_state = jax.tree.map(unbox_val, layers_state, is_leaf=is_lp)

    _, layers_params, layers_metrics, layers_mutables = nnx.split(layers_state, _is_static_param, nnx.Intermediate, ...)

    # layers_mutables catch-all should contain ONLY RngState variables (RngKey/RngCount).
    # If non_trainable state (e.g. BatchStat) appears here,
    # it is being carried through scan instead of broadcast.
    # NOTE: is_leaf stops jax.tree.leaves from traversing *into* Variable nodes,
    # so we see actual Variable instances (not raw arrays).
    assert all(
        isinstance(v, nnx.RngState)
        for v in jax.tree.leaves(layers_mutables, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ), (
        "Non-RngState variable found in layers_mutables catch-all partition. "
        "Only RngState variables (RngKey/RngCount) should be present."
    )

    # ---- Explicit fwd/bwd custom_vjp (deepseek_batchsplit pattern) ----
    #
    # Instead of nested L1/L2/L3 custom_vjp with opaque vjp closure residuals,
    # use a SINGLE custom_vjp with:
    #   - Forward: scan over repeats × microbatches, save explicit tensor residuals
    #   - Backward: scan in reverse, recompute per-iteration, accumulate gradients
    #
    # This follows deepseek_batchsplit.py's architecture which achieves optimal
    # XLA fission without nn.scan or nn.remat.

    from maxtext.utils import max_logging
    num_microbatches = self.config.num_pipeline_microbatches
    num_repeats = self.config.num_pipeline_repeats
    total_iterations = num_microbatches * num_repeats
    initial_w_curr = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype), layers_params)

    max_logging.log(
        f"[PIPELINE-DIAG] === EXPLICIT FWD/BWD (deepseek pattern) ===\n"
        f"[PIPELINE-DIAG]   num_repeats={num_repeats}, "
        f"num_microbatches={num_microbatches}, "
        f"total_iterations={total_iterations}, "
        f"bubble_iterations={bubble_iterations}"
    )

    def _run_iteration_pure(loop_st, layer_mut, bsw):
      """Pure forward for one microbatch iteration."""
      iteration = loop_st["loop_iteration"]
      advanced_mut = _advance_rng_state(layer_mut, iteration)
      new_loop_st, new_layer_state = self.run_one_iteration(
          loop_st, bsw, layers_graph, layers_metrics, advanced_mut,
          positions, segment_ids, deterministic, model_mode,
          logical_partition_spec_stripped,
      )
      _, _, _, new_mut = nnx.split(
          new_layer_state, _is_static_param, nnx.Intermediate, ...
      )
      return new_loop_st, new_mut

    @jax.custom_vjp
    def run_all_repeats(loop_state_init, layers_mutables_init, w_curr_init, pipeline_params):
      return _all_repeats_fwd(loop_state_init, layers_mutables_init, w_curr_init, pipeline_params)[0]

    def _all_repeats_fwd(loop_state_init, layers_mutables_init, w_curr_init, pipeline_params):
      max_logging.log("[PIPELINE-DIAG] _all_repeats_fwd: entering")

      bsw_ref = [None]

      def inner_body(carry, _):
        """One microbatch iteration inside a repeat."""
        loop_st, layer_mut = carry
        new_loop_st, new_mut = _run_iteration_pure(loop_st, layer_mut, bsw_ref[0])
        return (new_loop_st, new_mut), None

      if self.config.set_remat_policy_on_pipeline_iterations:
        inner_body = jax.checkpoint(
            inner_body, policy=self.get_pipeline_remat_policy(),
            prevent_cse=True,
        )

      def outer_body(carry, _):
        """One repeat: prefetch weights, run microbatches."""
        loop_st, layer_mut, w_curr = carry
        iteration = loop_st["loop_iteration"]

        w_next = self.weight_prefetching(
            pipeline_params, physical_partition_spec_full, iteration
        )
        bsw_ref[0] = (w_curr, w_next)

        max_logging.log(
            f"[PIPELINE-DIAG] outer_body fwd: w_curr_leaves="
            f"{len(jax.tree.leaves(w_curr))}, "
            f"w_next_leaves={len(jax.tree.leaves(w_next))}"
        )

        if self.config.scan_pipeline_iterations:
          (new_loop_st, new_mut), _ = jax.lax.scan(
              inner_body, (loop_st, layer_mut), None,
              length=num_microbatches,
          )
        else:
          inner_carry = (loop_st, layer_mut)
          for _ in range(num_microbatches):
            inner_carry, _ = inner_body(inner_carry, None)
          new_loop_st, new_mut = inner_carry

        return (new_loop_st, new_mut, w_next), None

      max_logging.log(
          f"[PIPELINE-DIAG] outer scan fwd: "
          f"scan_pipeline_iterations={self.config.scan_pipeline_iterations}, "
          f"length={num_repeats}"
      )

      if self.config.scan_pipeline_iterations:
        (final_loop_st, final_mut, final_w_curr), _ = jax.lax.scan(
            outer_body,
            (loop_state_init, layers_mutables_init, w_curr_init),
            None, length=num_repeats,
        )
      else:
        carry = (loop_state_init, layers_mutables_init, w_curr_init)
        for _ in range(num_repeats):
          carry, _ = outer_body(carry, None)
        final_loop_st, final_mut, final_w_curr = carry

      max_logging.log("[PIPELINE-DIAG] _all_repeats_fwd: done")
      return (
          (final_loop_st, final_mut, final_w_curr),
          (loop_state_init, layers_mutables_init, w_curr_init, pipeline_params),
      )

    def _all_repeats_bwd(residuals, g_output):
      """Backward: FIX-A only (recompute w_curr, keep fwd recompute scan)."""
      max_logging.log("[PIPELINE-DIAG] _all_repeats_bwd: entering (FIX-A only)")
      saved_loop_st, saved_mut, saved_w_curr, saved_params = residuals
      g_loop_st, g_mut, g_w_curr = g_output

      # Inner body for backward recompute: BSW passed through carry
      # to avoid tracer leak from bsw_ref side effect.
      def inner_body_remat_with_bsw(carry, _):
        loop_st, layer_mut, bsw = carry
        new_loop_st, new_mut = _run_iteration_pure(loop_st, layer_mut, bsw)
        return (new_loop_st, new_mut, bsw), None

      # Recompute forward to get per-repeat states (FIX-A: w_curr excluded)
      def fwd_outer_body(carry, _):
        loop_st, layer_mut, w_curr = carry
        iteration = loop_st["loop_iteration"]
        w_next = self.weight_prefetching(
            saved_params, physical_partition_spec_full, iteration
        )
        bsw = (w_curr, w_next)
        if self.config.scan_pipeline_iterations:
          (new_loop_st, new_mut, _), _ = jax.lax.scan(
              inner_body_remat_with_bsw, (loop_st, layer_mut, bsw), None,
              length=num_microbatches,
          )
        else:
          inner = (loop_st, layer_mut, bsw)
          for _ in range(num_microbatches):
            inner, _ = inner_body_remat_with_bsw(inner, None)
          new_loop_st, new_mut, _ = inner
        return (new_loop_st, new_mut, w_next), (loop_st, layer_mut)

      max_logging.log("[PIPELINE-DIAG] bwd: recomputing forward for per-repeat states")
      _, per_repeat_states = jax.lax.scan(
          fwd_outer_body,
          (saved_loop_st, saved_mut, saved_w_curr),
          None, length=num_repeats,
      )

      initial_iteration = saved_loop_st["loop_iteration"]

      # Backward scan over repeats in reverse.
      def bwd_outer_body(carry, per_repeat_xs):
        g_loop_st_c, g_mut_c, g_w_curr_c, d_params_c = carry
        loop_st_r, mut_r = per_repeat_xs
        iteration_r = loop_st_r["loop_iteration"]
        w_curr_r = jax.lax.cond(
            iteration_r == initial_iteration,
            lambda: saved_w_curr,
            lambda: self.weight_prefetching(
                saved_params, physical_partition_spec_full,
                iteration_r - num_microbatches,
            ),
        )
        w_next_r = self.weight_prefetching(
            saved_params, physical_partition_spec_full, iteration_r
        )
        bsw_r = (w_curr_r, w_next_r)

        weight_prefetching_t = jax.linear_transpose(
            functools.partial(
                self.weight_prefetching,
                physical_partition_spec=physical_partition_spec_full,
                loop_iteration=iteration_r,
            ),
            saved_params,
        )

        # Recompute inner scan + vjp for this repeat
        def _inner_fwd(state, bsw):
          if self.config.scan_pipeline_iterations:
            (result_st, result_mut, _), _ = jax.lax.scan(
                inner_body_remat_with_bsw, (*state, bsw), None,
                length=num_microbatches,
            )
          else:
            c = (*state, bsw)
            for _ in range(num_microbatches):
              c, _ = inner_body_remat_with_bsw(c, None)
            result_st, result_mut, _ = c
          return (result_st, result_mut)

        _inner_fwd_remat = jax.remat(
            _inner_fwd, policy=self.get_pipeline_remat_policy()
        )
        _, inner_vjp = jax.vjp(
            _inner_fwd_remat, (loop_st_r, mut_r), bsw_r
        )

        # Backward through inner scan
        d_state, d_bsw = inner_vjp((g_loop_st_c, g_mut_c))
        d_w_curr_inner, d_w_next = d_bsw

        # Reduce-scatter w_next gradient → param gradient
        (d_params_repeat,) = weight_prefetching_t(d_w_next)
        new_d_params = jax.tree.map(
            lambda a, b: a + b, d_params_c, d_params_repeat
        )

        # g_w_curr_c is the gradient flowing from repeat r+1 through the
        # w_next_r -> w_curr_{r+1} link. Since w_curr is now recomputed via
        # weight_prefetching, we must transpose g_w_curr_c back to params.
        # For repeat num_repeats-1 (first in reverse), g_w_curr_c comes from
        # the output gradient and is typically zero, so this is a no-op there.
        # For repeat 0, w_curr was zeros (not from weight_prefetching), so
        # we skip the transpose to avoid incorrect gradient accumulation.
        def _transpose_w_curr_grad(g_w_curr_val):
          w_curr_prefetch_t = jax.linear_transpose(
              functools.partial(
                  self.weight_prefetching,
                  physical_partition_spec=physical_partition_spec_full,
                  loop_iteration=iteration_r - num_microbatches,
              ),
              saved_params,
          )
          (d_params_from_wcurr,) = w_curr_prefetch_t(g_w_curr_val)
          return d_params_from_wcurr

        def _zero_params(_):
          return jax.tree.map(jnp.zeros_like, saved_params)

        d_params_from_w_curr = jax.lax.cond(
            iteration_r == initial_iteration,
            _zero_params,
            _transpose_w_curr_grad,
            g_w_curr_c,
        )
        new_d_params = jax.tree.map(
            lambda a, b: a + b, new_d_params, d_params_from_w_curr
        )

        g_loop_st_new, g_mut_new = d_state
        return (g_loop_st_new, g_mut_new, d_w_curr_inner, new_d_params), None

      max_logging.log("[PIPELINE-DIAG] bwd: reverse scan over repeats")
      initial_d_params = jax.tree.map(jnp.zeros_like, saved_params)
      (g_loop_st_final, g_mut_final, g_w_curr_final, d_params_final), _ = jax.lax.scan(
          bwd_outer_body,
          (g_loop_st, g_mut, g_w_curr, initial_d_params),
          per_repeat_states,
          reverse=True,
      )

      max_logging.log("[PIPELINE-DIAG] _all_repeats_bwd: done")
      return g_loop_st_final, g_mut_final, g_w_curr_final, d_params_final

    run_all_repeats.defvjp(_all_repeats_fwd, _all_repeats_bwd)

    # ---- Execute ----
    final_loop_st, final_mut, final_w_curr = run_all_repeats(
        loop_state, layers_mutables, initial_w_curr, layers_params
    )
    loop_state = final_loop_st
    final_layer_mutables = final_mut

    max_logging.log(
        f"[PIPELINE-DIAG] post-repeats: "
        f"loop_state keys={list(loop_state.keys())}, "
        f"final_mut_leaves={len(jax.tree.leaves(final_layer_mutables))}"
    )

    # ---- Bubble iterations (pipeline drain) ----
    bsw_ref_bubble = [None]

    def bubble_inner_body(carry, _):
      current_loop_state, current_layer_mutables = carry
      new_loop_st, new_mut = _run_iteration_pure(
          current_loop_state, current_layer_mutables, bsw_ref_bubble[0]
      )
      return (new_loop_st, new_mut), None

    if self.config.set_remat_policy_on_pipeline_iterations:
      bubble_inner_body = jax.checkpoint(
          bubble_inner_body,
          policy=self.get_pipeline_remat_policy(),
          prevent_cse=True,
      )

    if bubble_iterations > 0:
      bsw_ref_bubble[0] = (final_w_curr, final_w_curr)
      max_logging.log(
          f"[PIPELINE-DIAG] bubble: iterations={bubble_iterations}, "
          f"bsw=(final_w_curr, final_w_curr)"
      )
      if self.config.scan_pipeline_iterations:
        (loop_state, final_layer_mutables), _ = jax.lax.scan(
            bubble_inner_body, (loop_state, final_layer_mutables),
            None, length=bubble_iterations,
        )
      else:
        bubble_carry = (loop_state, final_layer_mutables)
        for _ in range(bubble_iterations):
          bubble_carry, _ = bubble_inner_body(bubble_carry, None)
        loop_state, final_layer_mutables = bubble_carry

    final_layer_state = nnx.State.merge(layers_params, layers_metrics, final_layer_mutables)
    nnx.update(self.layers, final_layer_state)

    final_output = self.realign_output_microbatches(loop_state["state_io"])
    max_logging.log(f"[PIPELINE-DIAG] final output: {final_output.shape}")
    return jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )


def create_nnx_pipeline(
    config: Config, stage_factory: Any, mesh: Mesh, remat_policy: Any = None, *, rngs: nnx.Rngs
) -> NNXPipeline | NNXCircularPipeline:
  """Factory function to instantiate the NNX Pipeline module."""
  if config.pipeline_fsdp_ag_per_repeat:
    return NNXCircularPipeline(
        config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs
    )
  return NNXPipeline(config=config, stage_factory=stage_factory, mesh=mesh, remat_policy=remat_policy, rngs=rngs)


Pipeline = to_linen_class(
    NNXPipeline,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
CircularPipeline = to_linen_class(
    NNXCircularPipeline,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


def create_pipeline(
    config: Config,
    layers,
    mesh: Mesh = None,
    remat_policy: Any = None,
) -> nn.Module:
  """Factory function to instantiate an NNX pipeline module wrapped in ToLinen.

  Args:
    config: Model configuration.
    layers: A callable ``rngs -> nnx.Module`` (stage factory).
    mesh: JAX device mesh for sharding.
    remat_policy: Optional rematerialization policy.
  """
  if config.pipeline_fsdp_ag_per_repeat:
    return CircularPipeline(config=config, stage_factory=layers, mesh=mesh, remat_policy=remat_policy)
  return Pipeline(config=config, stage_factory=layers, mesh=mesh, remat_policy=remat_policy)
