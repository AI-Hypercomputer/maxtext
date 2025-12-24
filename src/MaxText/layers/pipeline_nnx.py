"""
Pipeline Parallelism Module for MaxText using Flax NNX.
Native NNX Vectorized version for memory/speed efficiency.
"""

import functools
from typing import Any, Optional, Dict, Type, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import nnx
import flax.linen as nn_linen

from MaxText.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT


# --- Helpers ---
def debug_pytree_stats(name, tree):
  """Prints the number of leaves and total memory footprint of a Pytree."""
  leaves = jax.tree_util.tree_leaves(tree)
  num_leaves = len(leaves)

  # Calculate size in GB (assuming most are BF16/FP32)
  total_bytes = sum(x.nbytes if hasattr(x, "nbytes") else 0 for x in leaves)
  total_gb = total_bytes / (1024**3)

  # Only print on Lead Host to avoid log spam
  if jax.process_index() == 0:
    print(f"--- [DEBUG] {name} ---")
    print(f"  Count: {num_leaves} arrays")
    print(f"  Size:  {total_gb:.4f} GB")

    # Look for unexpected non-array leaves (potential overhead)
    non_arrays = [type(x) for x in leaves if not isinstance(x, (jnp.ndarray, jax.Array))]
    if non_arrays:
      print(f"  Warning: Found {len(non_arrays)} non-array leaves: {set(non_arrays)}")


def cast_to_dtype(node, dtype):
  """Recursively casts all floating point arrays in a Pytree/State to the target dtype."""

  def _cast(leaf):
    if isinstance(leaf, (jax.Array, jnp.ndarray)) and jnp.issubdtype(leaf.dtype, jnp.floating):
      return leaf.astype(dtype)
    return leaf

  return jax.tree_util.tree_map(_cast, node)


def to_pure_dict(x):
  """Recursively converts any nnx.State or custom mapping into a plain Python dict."""
  if hasattr(x, "items") and not isinstance(x, (jnp.ndarray, jax.Array)):
    return {k: to_pure_dict(v) for k, v in x.items()}
  return x


def with_logical_constraint(x, logical_axis_names, rules, mesh):
  if mesh is None:
    return x
  sharding_or_spec = nn_linen.logical_to_mesh_sharding(PartitionSpec(*logical_axis_names), mesh=mesh, rules=rules)
  if isinstance(sharding_or_spec, NamedSharding):
    return jax.lax.with_sharding_constraint(x, sharding_or_spec)
  elif isinstance(sharding_or_spec, PartitionSpec):
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, sharding_or_spec))
  return x


# --- NNX Pipeline Module ---


class InternalMetrics(nnx.Variable):
  """Custom variable for diagnostic metrics."""

  pass


class Pipeline(nnx.Module):

  def __init__(
      self, layers: nnx.Module, config: Config, mesh: Mesh, remat_policy: Any = None, rngs: nnx.Rngs | None = None
  ):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    self.rngs = rngs

    # 1. Pipeline Dimensions
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages

    num_repeats = self.config.num_pipeline_repeats if self.config.num_pipeline_repeats > 1 else 1
    self.total_instances = num_repeats * self.num_stages

    # 2. Logical Axis Setup
    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      self.batch_axis_name, self.seq_len_axis_name = "activation_batch_no_exp", "activation_length"
    else:
      self.batch_axis_name, self.seq_len_axis_name = "activation_batch", "activation_length_no_exp"
    self.use_circ_storage = self.need_circ_storage()

    if rngs is None:
      raise ValueError("Pipeline requires 'rngs' for initialization.")
    v_rngs = self.rngs.fork(split=self.total_instances)

    factory_kwargs = {
        "config": self.config,
        "mesh": self.mesh,
        "decoder_layer": getattr(layers, "decoder_layer", None),
        "num_decoder_layers": getattr(layers, "num_decoder_layers", 0),
        "model_mode": getattr(layers, "model_mode", MODEL_MODE_TRAIN),
        "quant": getattr(layers, "quant", None),
        "scan_layers": getattr(layers, "scan_layers", False),
        "dtype": self.config.dtype,
    }
    LayerCls = type(layers)

    # Warm-up Probe to define the structural template
    def get_full_metadata():
      m = LayerCls(rngs=nnx.Rngs(0), **factory_kwargs)
      # Run dummy pass to create metric slots
      m(
          jnp.zeros((1, 1, self.config.emb_dim), dtype=self.config.dtype),
          jnp.zeros((1, 1), dtype=jnp.int32),
          jnp.zeros((1, 1), dtype=jnp.int32),
          deterministic=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      # POP RNGs: This makes the GraphDef smaller and memory-clean
      nnx.pop(m, nnx.RngStream)
      m_def, m_state = nnx.split(m)
      # Capture sharding names for hierarchical distribution
      names = jax.tree_util.tree_map(lambda x: getattr(x, "sharding_names", None), m_state)
      return m_def, to_pure_dict(names)

    # graphdef is now "structurally complete" (has slots for metrics)
    # sharding_state_abstract contains the keys for every variable
    self.stage_graphdef, self.sharding_metadata = jax.eval_shape(get_full_metadata)

    v_rngs = self.rngs.fork(split=self.total_instances)

    def create_sharded_stage(r):
      m = type(layers)(rngs=r, **factory_kwargs)
      m(
          jnp.zeros((1, 1, self.config.emb_dim)),
          jnp.zeros((1, 1), dtype=jnp.int32),
          jnp.zeros((1, 1), dtype=jnp.int32),
          deterministic=False,
          model_mode=MODEL_MODE_TRAIN,
      )

      _, state = nnx.split(m)
      bf16_state = cast_to_dtype(state, self.config.dtype)
      nnx.update(m, bf16_state)

      nnx.pop(m, nnx.RngStream)
      return m

    with self.mesh:
      self.layers = nnx.vmap(create_sharded_stage, in_axes=0, spmd_axis_name="stage")(v_rngs)

  # --- MISSING HELPER: Determine active tokens and weights ---
  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Determines which data and weights are active for the current step."""
    # Calculate how many microbatches each physical stage has processed
    # This accounts for the bubble iterations (forwarding delay)
    processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = processed % self.config.num_pipeline_microbatches
    repeat_ids = processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def get_pipeline_remat_policy(self):
    """
    Returns the JAX rematerialization policy for this pipeline.
    This policy ensures that 'iteration_input' is saved to memory
    to avoid redundant recomputation of stages during the backward pass.
    """
    # 1. Check if the user has a custom override in the config
    if self.config.remat_policy == "custom":
      return self.remat_policy

    # 2. Define the Base Policy
    # We MUST save 'iteration_input' and 'decoder_layer_input'.
    # These names must match the 'jax.ad_checkpoint.checkpoint_name' calls
    # we added inside our scan_fn and Decoder layers.
    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")

    # 3. Combine with the Layer-Specific Policy
    # If the Pipeline was initialized with a remat_policy (e.g., 'minimal'),
    # we merge them so we save BOTH the inputs and the dots.
    if self.remat_policy is not None:
      # save_from_both_policies is the standard JAX utility for this.
      return jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)

    return save_input_policy

  def __call__(self, inputs, segment_ids=None, positions=None, deterministic=False, model_mode=MODEL_MODE_TRAIN):
    # 1. Inputs conversion (Same as before)
    inputs = jnp.asarray(inputs).reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        )
    )

    # Symmetrical Reshaping for Metadata
    # We must turn [Total_Batch, Length] into [Micro, Micro_Size, Length]
    if segment_ids is not None:
      segment_ids = jnp.asarray(segment_ids).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    if positions is not None:
      positions = jnp.asarray(positions).reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    # 2. Split State into Weights (Broadcast) and Metrics (Carry)
    # We separate things that change (Metrics) from things that are constant (Params)
    layers_def, layers_state = nnx.split(self.layers)

    # Bucket 1: Params (24)
    # Bucket 2: Metrics (Small)
    # Bucket 3: Remainder (The 30 internal RNG counters we need for the blueprint)
    params_state, metrics_state, remainder_state = layers_state.split(nnx.Param, InternalMetrics, ...)

    # --- MISSION 42 DIAGNOSTICS ---
    debug_pytree_stats("BUCKET 1: Params (Weights)", params_state)
    debug_pytree_stats("BUCKET 2: Metrics (Diagnostic)", metrics_state)
    debug_pytree_stats("BUCKET 3: Remainder (RNGs/Metadata)", remainder_state)
    # ------------------------------
    rng_def, rng_state = nnx.split(self.rngs)

    # The Carry now ONLY contains small tensors
    scan_carry = {
        "loop_state": self.init_loop_state(inputs),
        "metrics_state": to_pure_dict(metrics_state),  # Small metrics
        "rng_state": to_pure_dict(rng_state),
    }

    # The weights are passed as a closure (Broadcasted)
    # JAX will only keep ONE copy of params_pure_dict in memory
    params_pure_dict = to_pure_dict(params_state)
    remainder_pure_dict = to_pure_dict(remainder_state)

    def scan_fn(carry, _):
      l_state = carry["loop_state"]
      loop_iter = l_state["loop_iteration"]

      # (it_inputs, indices, and RNG fork logic same as Mission 25)
      micro_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iter)
      it_inputs = self.get_iteration_inputs(loop_iter, l_state["state_io"], l_state["circ_storage"], l_state["shift"])
      it_inputs = jax.ad_checkpoint.checkpoint_name(it_inputs, "iteration_input")

      it_pos = jnp.take(positions, micro_ids, axis=0) if positions is not None else None
      it_seg = jnp.take(segment_ids, micro_ids, axis=0) if segment_ids is not None else None
      if it_pos is not None:
        it_pos = self.shard_dim_by_stages(it_pos, 0)
      if it_seg is not None:
        it_seg = self.shard_dim_by_stages(it_seg, 0)

      it_rngs = nnx.merge(rng_def, nnx.State(carry["rng_state"]))
      vmap_rngs_obj = it_rngs.fork(split=self.num_stages)
      _, next_rng_state = nnx.split(it_rngs)
      _, vmap_rng_state = nnx.split(vmap_rngs_obj)

      stage_indices = jnp.arange(self.num_stages)
      target_indices = (
          stage_indices if self.config.num_pipeline_repeats <= 1 else (repeat_ids * self.num_stages + stage_indices)
      )

      # --- GATHER SLICES ---
      # Gather slices for both weights and stats
      active_params = jax.tree_util.tree_map(lambda x: x[target_indices], params_pure_dict)
      active_metrics = jax.tree_util.tree_map(lambda x: x[target_indices], carry["metrics_state"])
      active_remainder = jax.tree_util.tree_map(lambda x: x[target_indices], remainder_pure_dict)

      def run_stage(p_raw, m_raw, r_raw, x, seg, pos, r_keys):
        # Only merge what is necessary for the call
        m = nnx.merge(layers_def, nnx.State(p_raw), nnx.State(m_raw), nnx.State(r_raw))

        # Reseed using a more direct method to save Python cycles
        # it_m_rngs_state = nnx.State(r_keys)
        nnx.update(m, nnx.State(r_keys))

        # EXECUTE
        out, _ = m(x, decoder_segment_ids=seg, decoder_positions=pos, deterministic=deterministic, model_mode=model_mode)

        # Split back ONLY metrics.
        # Discarding GraphDef/Params/Remainder here is what allows 129 tokens/s.
        _, _, final_metrics, _ = nnx.split(m, nnx.Param, InternalMetrics, ...)
        return out, to_pure_dict(final_metrics)

      # VMAP execution
      stages_out, updated_metrics = nnx.vmap(run_stage)(
          active_params, active_metrics, active_remainder, it_inputs, it_seg, it_pos, to_pure_dict(vmap_rng_state)
      )

      # Update the metrics carry
      new_metrics_state = jax.tree_util.tree_map(
          lambda full, sub: full.at[target_indices].set(sub), carry["metrics_state"], updated_metrics
      )

      new_carry = {
          "loop_state": self.get_new_loop_state(stages_out, l_state),
          "metrics_state": new_metrics_state,
          "rng_state": to_pure_dict(next_rng_state),
      }
      return new_carry, None

    # 4. Execute Scan with Checkpointing
    policy = self.get_pipeline_remat_policy()
    scannable_fn = (
        jax.checkpoint(scan_fn, policy=policy) if self.config.set_remat_policy_on_pipeline_iterations else scan_fn
    )

    total_steps = (self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats) + self.forwarding_delay * (
        self.num_stages - 1
    )

    final_carry, _ = jax.lax.scan(scannable_fn, scan_carry, None, length=total_steps)

    # 5. SYNC BACK TO OBJECT
    # Re-combine the constant params and the updated stats for the final object sync

    nnx.update(self.layers, nnx.State(final_carry["metrics_state"]))
    nnx.update(self.rngs, nnx.State(final_carry["rng_state"]))

    out = self.permute_output_micro_per_stage_dim(final_carry["loop_state"]["state_io"])
    return jnp.reshape(
        out, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim)
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

  def shard_dim_by_stages(self, x, dim: int):
    if self.mesh is None:
      return x
    dims = [PartitionSpec.UNCONSTRAINED] * x.ndim
    dims[dim] = "stage"
    sharding = NamedSharding(self.mesh, PartitionSpec(*dims))
    return jax.lax.with_sharding_constraint(x, sharding)

  def init_loop_state(self, inputs):
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = with_logical_constraint(
        shift,
        ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
        self.config.logical_axis_rules,
        self.mesh,
    )
    prev_outputs = jnp.zeros_like(shift) if self.config.pipeline_delay_activation_forwarding else None
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    state_io = with_logical_constraint(
        state_io,
        ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
        self.config.logical_axis_rules,
        self.mesh,
    )
    circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype) if self.use_circ_storage else None
    circ_mover = shift if self.use_circ_storage else None
    return {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": circ_mover,
        "loop_iteration": jnp.array(0, dtype=jnp.int32),
        "prev_outputs": prev_outputs,
    }

  def get_iteration_inputs(self, loop_iter, state_io, circ_storage, shift):
    state_io_slice = state_io[:, loop_iter % self.microbatches_per_stage]
    circ_in = circ_storage[:, loop_iter % self.config.num_pipeline_microbatches] if self.use_circ_storage else shift
    first_in = jnp.where(loop_iter < self.config.num_pipeline_microbatches, state_io_slice, circ_in)
    stages_in = jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_in, shift)
    return with_logical_constraint(
        stages_in,
        ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
        self.config.logical_axis_rules,
        self.mesh,
    )

  def get_new_loop_state(self, output, loop_state):
    loop_iter = loop_state["loop_iteration"]

    def _rotate_right(a):
      return jnp.concatenate(
          [
              jax.lax.slice_in_dim(a, self.num_stages - 1, self.num_stages, axis=0),
              jax.lax.slice_in_dim(a, 0, self.num_stages - 1, axis=0),
          ],
          axis=0,
      )

    def _shift_right(a):
      return jax.lax.slice(jnp.pad(a, [[1, 0]] + [[0, 0]] * (a.ndim - 1)), [0] * a.ndim, a.shape)

    shift_out = (
        _shift_right(output)
        if (self.config.num_pipeline_repeats == 1 or self.use_circ_storage)
        else _rotate_right(output)
    )
    new_prev = output if self.config.pipeline_delay_activation_forwarding else None
    new_shift = (
        _shift_right(loop_state["prev_outputs"]) if self.config.pipeline_delay_activation_forwarding else shift_out
    )
    new_circ = loop_state["circ_storage"]
    new_mover = loop_state["circ_storage_mover"]
    if self.use_circ_storage:
      rot_mover = jnp.expand_dims(_rotate_right(new_mover), 1)
      off = (
          loop_iter - self.iterations_to_complete_first_microbatch_one_repeat() - 1
      ) % self.config.num_pipeline_microbatches
      new_circ = jax.lax.dynamic_update_slice_in_dim(new_circ, rot_mover, off, axis=1)
      new_mover = output
    stream_idx = loop_iter % self.microbatches_per_stage
    stream_slice = loop_state["state_io"][:, stream_idx]
    padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
    padded_stream = jnp.pad(stream_slice, padding)
    stream_slice = jax.lax.slice_in_dim(padded_stream, 1, stream_slice.shape[0] + 1, axis=0)
    stream_slice = jnp.where(
        jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1, output, stream_slice
    )
    new_state_io = jax.lax.dynamic_update_slice_in_dim(
        loop_state["state_io"], jnp.expand_dims(stream_slice, 1), stream_idx, axis=1
    )
    return {
        "state_io": new_state_io,
        "shift": new_shift,
        "circ_storage": new_circ,
        "circ_storage_mover": new_mover,
        "loop_iteration": loop_iter + 1,
        "prev_outputs": new_prev,
    }

  def permute_output_micro_per_stage_dim(self, output):
    idx0 = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    perm = (np.arange(self.microbatches_per_stage) + idx0) % self.microbatches_per_stage
    return output[:, perm]
