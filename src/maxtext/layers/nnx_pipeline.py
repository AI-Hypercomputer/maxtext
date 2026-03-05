from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.common.common_types import EP_AS_CONTEXT, MODEL_MODE_TRAIN, Config, ShardMode
from maxtext.utils.sharding import (
    all_gather_over_fsdp,
    create_sharding,
    logical_to_mesh,
    logical_to_mesh_axes,
    maybe_shard_with_logical,
    maybe_shard_with_name,
)
from maxtext.utils import max_logging


class Pipeline(nnx.Module):
  """Module that implements pipelining across stages using NNX."""

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

    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    # TODO(b/470167805): replace self.spmd_axis_name with "stage" when JAX >= 0.8.2.
    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    # -------------------------------------------------------------------------
    # Native NNX Pipeline State Initialization
    # -------------------------------------------------------------------------
    def build_batched_rngs(shape):
      max_logging.log(f"Building batched RNGs with shape {shape}...")
      kwargs = {}
      
      # Extract only the RNG components of the state
      rng_state = nnx.state(rngs, nnx.RngState)
      
      # Use bulletproof JAX tree utilities to flatten it, bypassing NNX dict internals
      leaves, _ = jax.tree_util.tree_flatten_with_path(rng_state)
      
      for path, key in leaves:
        stream_name = getattr(path[0], 'key', str(path[0]))
        
        if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            key = jax.random.key(key)
            
        num_splits = int(np.prod(shape))
        flat_keys = jax.random.split(key, num_splits)
        kwargs[stream_name] = flat_keys.reshape(shape + key.shape)
        
      max_logging.log(f"Successfully generated batched nnx.Rngs for streams: {list(kwargs.keys())}")
      return nnx.Rngs(**kwargs)

    # Vmap over stages natively adds 'layers' metadata to the logical partition spec!
    vmap_stages = nnx.vmap(
        stage_factory,
        in_axes=0,
        out_axes=nnx.StateAxes({nnx.Param: 0, ...: 0}),
        axis_name=self.spmd_axis_name,
        transform_metadata={nnx.PARTITION_NAME: "layers"}
    )

    if self.config.num_pipeline_repeats > 1:
      # Vmap over repeats. We use None to signify this dimension is physically replicated,
      # which prevents MaxText's logical_to_mesh from crashing on an unknown axis name.
      vmap_repeats = nnx.vmap(
          vmap_stages,
          in_axes=0,
          out_axes=nnx.StateAxes({nnx.Param: 0, ...: 0}),
          transform_metadata={nnx.PARTITION_NAME: None}
      )
      batched_rngs = build_batched_rngs((self.config.num_pipeline_repeats, self.num_stages))
      self.layers = vmap_repeats(batched_rngs)
    else:
      batched_rngs = build_batched_rngs((self.num_stages,))
      self.layers = vmap_stages(batched_rngs)

    # -------------------------------------------------------------------------
    # Sharding Configs
    # -------------------------------------------------------------------------
    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      self.batch_axis_name = "activation_batch_no_exp"
      self.seq_len_axis_name = "activation_length"
    else:
      self.batch_axis_name = "activation_batch"
      self.seq_len_axis_name = "activation_length_no_exp"

    self.stages_in_logical = ("layers", self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.stages_in_sharding = (
        NamedSharding(self.mesh, self.stages_in_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )

    self.state_io_logical = ("layers", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
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
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def _maybe_shard_with_logical(self, inputs, logical_axes):
    return maybe_shard_with_logical(
        inputs,
        logical_axes,
        shard_mode=self.config.shard_mode,
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
        debug_sharding=self.config.debug_sharding,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    return maybe_shard_with_name(
        inputs,
        sharding_name,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def init_states(self, inputs):
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
    else:
      prev_outputs = None

    state_io = jnp.reshape(
        inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:], out_sharding=self.state_io_sharding
    )
    state_io = self._maybe_shard_with_logical(state_io, self.state_io_logical)

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

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    if self.use_circ_storage:
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      circular_stage_in = shift

    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)
    first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)

    def select_state_or_input(first_stage_in, shift):
      return jnp.where(
          jax.lax.broadcasted_iota("int32", shift.shape, 0, out_sharding=self.stages_in_sharding) == 0,
          first_stage_in,
          shift,
      )

    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = self._maybe_shard_with_logical(stages_in, self.stages_in_logical)
    return stages_in

  def shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False):
    placeholder = None if self.config.shard_mode == ShardMode.EXPLICIT else P.UNCONSTRAINED
    if physical_partition_spec is None:
      dims_mapping = [placeholder] * x.ndim
      if dim < x.ndim:
        dims_mapping[dim] = "stage"
    else:
      physical_partition_spec = self._remove_fsdp_from_physical_partition_spec(physical_partition_spec)
      prefix = [placeholder] * dim +["stage"]
      dims_mapping = prefix + list(physical_partition_spec)

    dims_mapping = tuple(dims_mapping)

    if physical_partition_spec and is_stage_weight and self.config.shard_mode == ShardMode.EXPLICIT:
      batch_mesh_axis =["data", "fsdp"]
      reduced_mark =[mesh_axis for mesh_axis in batch_mesh_axis if self.mesh.shape[mesh_axis] > 1]
      pspec = P(*dims_mapping, reduced=set(reduced_mark))
    else:
      pspec = P(*dims_mapping)

    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
    return self._maybe_shard_with_name(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
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

    stage_weights = self.shard_dim_by_stages(
        stage_weights, gathered_weights_stage_dim, physical_partition_spec=physical_partition_spec, is_stage_weight=True
    )
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
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
    old_state_io = loop_state["state_io"]
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    old_prev_outputs = loop_state["prev_outputs"]

    @jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
    def _rotate_right(arr):
      stage_size = jax.lax.axis_size("stage")
      perm =[(i, (i + 1) % stage_size) for i in range(stage_size)]
      return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

    @jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec, out_specs=self.stages_in_spec, check_vma=True)
    def _shift_right(arr):
      stage_idx = jax.lax.axis_index("stage")
      stage_size = jax.lax.axis_size("stage")
      perm =[(i, (i + 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)

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
        rotated = jnp.expand_dims(_rotate_right(circ_storage_mover_in), 1)
        offset = (loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1) % self.config.num_pipeline_microbatches
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    @jax.shard_map(mesh=self.mesh, in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, P()), out_specs=self.state_io_spec)
    def _update_state_io(state_in, stream_slice, output, stream_buf_idx):
      stage_size = jax.lax.axis_size("stage")
      perm =[(i, (i - 1) % stage_size) for i in range(stage_size)]
      stream_slice = jax.lax.ppermute(stream_slice, axis_name="stage", perm=perm)
      stream_slice = jnp.where(jax.lax.axis_index("stage") == stage_size - 1, output, stream_slice)
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
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    output = output[:, permutation]
    return output

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec)
    else:
      return pipeline_weights

  def get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None):
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    def gather_weights_for_stages_in(w, spec=None):
      if w is None:
          return None
      return self.vmap_parallel_gather(w, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1, physical_partition_spec=spec)

    if physical_partition_spec is None:
      return jax.tree.map(gather_weights_for_stages_in, weights)
    else:
      return jax.tree.map(gather_weights_for_stages_in, weights, physical_partition_spec)

  def update_current_repeat_from_stages(self, full_weights, updated_stage_weights, loop_iteration, physical_partition_spec=None):
    """Scatters updated stage weights (like advanced RNGs) back into the full pipeline weights tensor."""
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    def _scatter_update(fw, uw, spec=None):
      if fw is None or uw is None:
        return fw
      def _update_one_stage(f_s, u_s, r_id):
        return jax.lax.dynamic_update_slice_in_dim(f_s, jnp.expand_dims(u_s, 0), r_id, axis=0)

      r_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
      updated_fw = jax.vmap(_update_one_stage, in_axes=(1, 0, 0), out_axes=1)(fw, uw, r_ids)
      return self.shard_dim_by_stages(updated_fw, 1, physical_partition_spec=spec, is_stage_weight=False)

    if physical_partition_spec is None:
      return jax.tree.map(_scatter_update, full_weights, updated_stage_weights)
    else:
      return jax.tree.map(_scatter_update, full_weights, updated_stage_weights, physical_partition_spec)

  def get_main_vmap_func_for_iterations(self):
    def func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      module = nnx.merge(graph, state)
      out = module(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)
      return out, nnx.state(module)

    vmap_func = nnx.vmap(
        func_to_vmap,
        in_axes=(None, 0, 0, 0, 0, None, None),
        out_axes=(0, 0),
        axis_name=self.spmd_axis_name,
    )
    return vmap_func

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

    if self.config.num_pipeline_repeats > 1:
      stage_weights_state = self.get_current_stage_weights(
          pipeline_weights_state, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    else:
      stage_weights_state = pipeline_weights_state

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
      pipeline_weights_state = self.update_current_repeat_from_stages(
          pipeline_weights_state, updated_stage_weights_state, loop_iteration, physical_partition_spec
      )
    else:
      pipeline_weights_state = updated_stage_weights_state

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state, pipeline_weights_state

  def get_pipeline_remat_policy(self):
    if self.config.remat_policy == "custom":
      return self.remat_policy
    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    return save_input_policy

  def get_weight_sharding(self, *init_args):
    """Safely extract metadata specs directly from VariableStates."""
    state = nnx.state(self.layers)

    def get_spec(x):
      if not isinstance(x, nnx.VariableState):
        return None
      if isinstance(x.value, nn.spmd.LogicallyPartitioned):
        return x.value.partitions
      meta = x.get_metadata()
      sharding = meta.get("sharding")
      if sharding and hasattr(sharding, "spec"):
        return sharding.spec
      return None

    logical_partition_spec = jax.tree.map(get_spec, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
    return logical_partition_spec

  def get_logical_spec_repeats_removed(self, full_logical):
    """Strips the None mapping from the repeat dimension if repeats are enabled."""
    if full_logical is None or self.config.num_pipeline_repeats == 1:
      return full_logical

    def _remove_from_spec(spec):
      if not isinstance(spec, P):
        return spec
      # The repeat dim is the 0th dim, slice it off to get the underlying stage spec
      return jax.sharding.PartitionSpec(*spec[1:])

    return jax.tree.map(_remove_from_spec, full_logical, is_leaf=lambda x: isinstance(x, P))

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    if isinstance(pps, P):
      new_spec =[]
      for axis in pps:
        if axis is None:
          new_spec.append(None)
        elif isinstance(axis, str):
          if axis not in ("fsdp", "fsdp_transpose"):
            new_spec.append(axis)
          else:
            new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          new_axis =[a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else:
          raise ValueError(f"Unsupported_axis_type: {type(axis)}")
      return P(*new_spec)
    return pps

  def all_gather_over_fsdp(self, state, logical_partition_spec):
    return all_gather_over_fsdp(
        state, logical_partition_spec, self.mesh, self.config.logical_axis_rules, self.config.shard_mode
    )

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
        (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length, self.config.emb_dim),
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
    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    logical_partition_spec = self.get_logical_spec_repeats_removed(logical_partition_spec)

    # 1. Split NNX Graph and State
    layers_graph, layers_state = nnx.split(self.layers)

    # 2. Globally unbox LogicallyPartitioned values before entering loop
    def is_lp(x):
      return isinstance(x, nn.spmd.LogicallyPartitioned)
    def unbox_val(x):
      return x.value if is_lp(x) else x
    layers_state = jax.tree.map(unbox_val, layers_state, is_leaf=is_lp)

    # 3. Handle Optional FSDP All-Gather
    if self.config.pipeline_fsdp_ag_once:
      layers_state = self.all_gather_over_fsdp(layers_state, logical_partition_spec)

    # 4. Split State: Separate heavy static Params from lightweight dynamic Mutables to prevent scan OOM
    _, layers_params, layers_mutables = nnx.split(layers_state, nnx.Param, ...)

    # 5. Checkpointable Scan Body
    def scan_body(carry, _):
      current_loop_state, current_layer_mutables = carry
      
      # Merge static params from closure with carried mutables
      current_layer_state = nnx.State.merge(layers_params, current_layer_mutables)
      
      new_loop_state, new_layer_state = self.run_one_iteration(
          current_loop_state, layers_graph, current_layer_state,
          positions, segment_ids, deterministic, model_mode, logical_partition_spec,
      )
      
      # Re-split and ONLY pass the mutables to the next loop iteration to prevent parameter buffering memory bloat
      _, _, new_layer_mutables = nnx.split(new_layer_state, nnx.Param, ...)
      
      return (new_loop_state, new_layer_mutables), None

    if self.config.set_remat_policy_on_pipeline_iterations:
      scan_body = jax.checkpoint(scan_body, policy=self.get_pipeline_remat_policy())

    # 6. Execute Loop
    if self.config.scan_pipeline_iterations:
      (loop_state, final_layer_mutables), _ = jax.lax.scan(
          scan_body, (loop_state, layers_mutables), None, length=total_iterations
      )
    else:
      current_carry = (loop_state, layers_mutables)
      for _ in range(total_iterations):
        current_carry, _ = scan_body(current_carry, None)
      loop_state, final_layer_mutables = current_carry

    # 7. Re-merge the full state to safely update metrics and RNGs without TraceContextErrors
    final_layer_state = nnx.State.merge(layers_params, final_layer_mutables)
    self.layers = nnx.merge(layers_graph, final_layer_state)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )

    return final_output