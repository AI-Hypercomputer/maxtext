"""
Pipeline Parallelism Module for MaxText using Flax NNX.
Refactored to use VMAP over a single template module for memory/speed efficiency.
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

def _strip_spec(spec):
    """Removes 'fsdp' and 'fsdp_transpose' from a PartitionSpec."""
    if spec is None: return None
    new_axes = []
    for axis in spec:
        if axis in ("fsdp", "fsdp_transpose"):
            new_axes.append(None)
        elif isinstance(axis, (list, tuple)):
            new_sub_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
            new_axes.append(tuple(new_sub_axis) if new_sub_axis else None)
        else:
            new_axes.append(axis)
    return PartitionSpec(*new_axes)

def with_logical_constraint(x, logical_axis_names, rules, mesh):
    if mesh is None: return x
    sharding_or_spec = nn_linen.logical_to_mesh_sharding(
        PartitionSpec(*logical_axis_names), mesh=mesh, rules=rules
    )
    if isinstance(sharding_or_spec, NamedSharding):
        return jax.lax.with_sharding_constraint(x, sharding_or_spec)
    elif isinstance(sharding_or_spec, PartitionSpec):
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, sharding_or_spec))
    return x

# --- NNX Pipeline Module ---

class Pipeline(nnx.Module):
    def __init__(
        self, 
        layers: nnx.Module, 
        config: Config, 
        mesh: Mesh, 
        remat_policy: Any = None, 
        rngs: nnx.Rngs | None = None
    ):
        self.config = config
        self.mesh = mesh
        self.remat_policy = remat_policy
        
        # Dimensions
        self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
        self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
        self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
        self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
        self.use_circ_storage = self.need_circ_storage()

        # Logical Axis Names
        if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
            self.batch_axis_name = "activation_batch_no_exp"
            self.seq_len_axis_name = "activation_length"
        else:
            self.batch_axis_name = "activation_batch"
            self.seq_len_axis_name = "activation_length_no_exp"

        if rngs is None:
            raise ValueError("Pipeline requires 'rngs' to initialize stage parameters.")

        # --- OPTIMIZED INITIALIZATION (VMAP) ---
        num_repeats = self.config.num_pipeline_repeats if self.config.num_pipeline_repeats > 1 else 1
        LayerCls = type(layers)
        
        # Extract init kwargs
        kwargs = {}
        for attr in ['decoder_layer', 'num_decoder_layers', 'quant', 'model_mode', 'scan_layers']:
            if hasattr(layers, attr):
                kwargs[attr] = getattr(layers, attr)

        # Helper to instantiate a single stage
        def create_stage(key_s):
            stage_rngs = nnx.Rngs(params=key_s) 
            return LayerCls(config=self.config, mesh=self.mesh, rngs=stage_rngs, **kwargs)

        # Generate keys for all stages
        total_instances = num_repeats * self.num_stages
        root_key = rngs.params()
        keys = jax.random.split(root_key, total_instances)
        
        # 1. Instantiate the template (Graph definition)
        template_module = create_stage(keys[0])
        self.graphdef, _ = nnx.split(template_module)
        
        # 2. VMAP Initialization to get Stacked States
        def get_layer_state(k):
            m = create_stage(k)
            return nnx.state(m)

        self.stacked_state = jax.vmap(get_layer_state)(keys)
        
        # 3. Apply Sharding to the Stacked State
        if self.mesh is not None:
            def shard_leading_dim(leaf):
                axes = ("stage",) + (None,) * (leaf.ndim - 1)
                spec = PartitionSpec(*axes)
                sharding = NamedSharding(self.mesh, spec)
                return jax.device_put(leaf, sharding)
            
            # FIX: Use jax.tree.map instead of jax.tree_map
            self.stacked_state = jax.tree.map(shard_leading_dim, self.stacked_state)

    def need_circ_storage(self):
        return (self.config.num_pipeline_repeats > 1 and 
                self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay)

    def iterations_to_complete_first_microbatch_one_repeat(self):
        return self.forwarding_delay * (self.num_stages - 1)

    def iterations_to_complete_first_microbatch(self):
        return (self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + 
                self.iterations_to_complete_first_microbatch_one_repeat())

    def get_pipeline_remat_policy(self):
        if self.config.remat_policy == "custom": return self.remat_policy
        save_input = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
        return (jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input) 
                if self.remat_policy else save_input)

    def get_weight_sharding(self, *args, **kwargs):
        def get_spec(leaf):
            if hasattr(leaf, 'sharding') and isinstance(leaf.sharding, NamedSharding):
                return leaf.sharding.spec
            return None
        # FIX: Use jax.tree.map
        return jax.tree.map(get_spec, self.stacked_state)

    def all_gather_over_fsdp(self):
        def apply_ag(leaf):
            if hasattr(leaf, 'sharding') and isinstance(leaf.sharding, NamedSharding):
                new_spec = _strip_spec(leaf.sharding.spec)
                target = NamedSharding(leaf.sharding.mesh, new_spec)
                return jax.lax.with_sharding_constraint(leaf, target)
            return leaf
        
        # FIX: Use jax.tree.map
        self.stacked_state = jax.tree.map(apply_ag, self.stacked_state)

    def shard_dim_by_stages(self, x, dim: int):
        if self.mesh is None: return x
        dims = [PartitionSpec.UNCONSTRAINED] * x.ndim
        dims[dim] = "stage"
        sharding = NamedSharding(self.mesh, PartitionSpec(*dims))
        return jax.lax.with_sharding_constraint(x, sharding)

    # ... (Loop Helpers: init_loop_state, get_iteration_inputs... COPY FROM PREVIOUS) ...
    def get_microbatch_and_repeat_ids(self, loop_iteration):
        processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
        return processed % self.config.num_pipeline_microbatches, processed // self.config.num_pipeline_microbatches

    def init_loop_state(self, inputs):
        shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
        shift = with_logical_constraint(shift, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), self.config.logical_axis_rules, self.mesh)
        prev_outputs = jnp.zeros_like(shift) if self.config.pipeline_delay_activation_forwarding else None
        if prev_outputs is not None:
            prev_outputs = with_logical_constraint(prev_outputs, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), self.config.logical_axis_rules, self.mesh)
        state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
        state_io = with_logical_constraint(state_io, ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), self.config.logical_axis_rules, self.mesh)
        circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype) if self.use_circ_storage else None
        circ_mover = shift if self.use_circ_storage else None
        return {"state_io": state_io, "shift": shift, "circ_storage": circ_storage, "circ_storage_mover": circ_mover, "loop_iteration": jnp.array(0, dtype=jnp.int32), "prev_outputs": prev_outputs}

    def get_iteration_inputs(self, loop_iter, state_io, circ_storage, shift):
        state_io_slice = state_io[:, loop_iter % self.microbatches_per_stage]
        circ_in = circ_storage[:, loop_iter % self.config.num_pipeline_microbatches] if self.use_circ_storage else shift
        first_in = jnp.where(loop_iter < self.config.num_pipeline_microbatches, state_io_slice, circ_in)
        stages_in = jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_in, shift)
        return with_logical_constraint(stages_in, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), self.config.logical_axis_rules, self.mesh)

    def get_new_loop_state(self, output, loop_state):
        loop_iter = loop_state["loop_iteration"]
        def _rotate_right(a): return jnp.concatenate([jax.lax.slice_in_dim(a, self.num_stages - 1, self.num_stages, axis=0), jax.lax.slice_in_dim(a, 0, self.num_stages - 1, axis=0)], axis=0)
        def _shift_right(a): return jax.lax.slice(jnp.pad(a, [[1, 0]] + [[0, 0]] * (a.ndim - 1)), [0] * a.ndim, a.shape)
        shift_out = _shift_right(output) if (self.config.num_pipeline_repeats == 1 or self.use_circ_storage) else _rotate_right(output)
        new_prev = output if self.config.pipeline_delay_activation_forwarding else None
        new_shift = _shift_right(loop_state["prev_outputs"]) if self.config.pipeline_delay_activation_forwarding else shift_out
        new_circ = loop_state["circ_storage"]
        new_mover = loop_state["circ_storage_mover"]
        if self.use_circ_storage:
            rot_mover = jnp.expand_dims(_rotate_right(new_mover), 1)
            off = (loop_iter - self.iterations_to_complete_first_microbatch_one_repeat() - 1) % self.config.num_pipeline_microbatches
            new_circ = jax.lax.dynamic_update_slice_in_dim(new_circ, rot_mover, off, axis=1)
            new_mover = output
        stream_idx = loop_iter % self.microbatches_per_stage
        stream_slice = loop_state["state_io"][:, stream_idx]
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        padded_stream = jnp.pad(stream_slice, padding)
        stream_slice = jax.lax.slice_in_dim(padded_stream, 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1, output, stream_slice)
        new_state_io = jax.lax.dynamic_update_slice_in_dim(loop_state["state_io"], jnp.expand_dims(stream_slice, 1), stream_idx, axis=1)
        return {"state_io": new_state_io, "shift": new_shift, "circ_storage": new_circ, "circ_storage_mover": new_mover, "loop_iteration": loop_iter + 1, "prev_outputs": new_prev}

    def permute_output_micro_per_stage_dim(self, output):
        idx0 = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
        perm = (np.arange(self.microbatches_per_stage) + idx0) % self.microbatches_per_stage
        return output[:, perm]

    # --- MAIN CALL ---

    def __call__(self, inputs, segment_ids=None, positions=None, deterministic=False, model_mode=MODEL_MODE_TRAIN, partition_spec=None):
        # 0. Convert & Reshape Inputs
        inputs = jnp.asarray(inputs).reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length, self.config.emb_dim))
        
        ag_sharding = NamedSharding(self.mesh, PartitionSpec(None, None))
        if positions is not None:
            positions = jax.lax.with_sharding_constraint(jnp.asarray(positions), ag_sharding).reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))
        if segment_ids is not None:
            segment_ids = jax.lax.with_sharding_constraint(jnp.asarray(segment_ids), ag_sharding).reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))

        if self.config.pipeline_fsdp_ag_once: self.all_gather_over_fsdp()

        loop_state = self.init_loop_state(inputs)

        # --- OPTIMIZED SCAN ---
        def scan_fn(carry, _):
            loop_iter = carry["loop_iteration"]
            stages_inputs = self.get_iteration_inputs(loop_iter, carry["state_io"], carry["circ_storage"], carry["shift"])
            stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
            
            micro_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iter)
            
            s_pos = positions[micro_ids] if positions is not None else None
            s_seg = segment_ids[micro_ids] if segment_ids is not None else None
            if s_pos is not None: s_pos = self.shard_dim_by_stages(s_pos, 0)
            if s_seg is not None: s_seg = self.shard_dim_by_stages(s_seg, 0)

            stage_indices = jnp.arange(self.num_stages)
            target_indices = stage_indices 
            if self.config.num_pipeline_repeats > 1:
                target_indices = repeat_ids * self.num_stages + stage_indices
            
            # Select weights for this step using jax.tree.map
            def gather_state(stacked, idxs):
                return jax.vmap(lambda i: jax.tree.map(lambda l: l[i], stacked))(idxs)
            
            current_states = jax.tree.map(lambda leaf: gather_state(leaf, target_indices), self.stacked_state)

            # Run Layer (Pure Vmap)
            def run_layer(state, x, seg, pos):
                model = nnx.merge(self.graphdef, state)
                out = model(x, decoder_segment_ids=seg, decoder_positions=pos, deterministic=deterministic, model_mode=model_mode)
                return out

            in_axes_seg = 0 if s_seg is not None else None
            in_axes_pos = 0 if s_pos is not None else None
            
            stages_out = jax.vmap(run_layer, in_axes=(0, 0, in_axes_seg, in_axes_pos))(
                current_states, stages_inputs, s_seg, s_pos
            )

            if self.config.scan_layers and isinstance(stages_out, tuple):
                stages_out = stages_out[0]

            return self.get_new_loop_state(stages_out, carry), None

        total_steps = (self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats) + self.forwarding_delay * (self.num_stages - 1)
        policy = self.get_pipeline_remat_policy() if self.config.set_remat_policy_on_pipeline_iterations else None
        
        if self.config.scan_pipeline_iterations:
             scan_fn = jax.checkpoint(scan_fn, policy=policy, prevent_cse=not self.config.scan_pipeline_iterations)
             final_loop_state, _ = jax.lax.scan(scan_fn, loop_state, None, length=total_steps)
        else:
             curr = loop_state
             scan_fn = jax.checkpoint(scan_fn, policy=policy) if policy else scan_fn
             for _ in range(total_steps): curr, _ = scan_fn(curr, None)
             final_loop_state = curr
        
        out = self.permute_output_micro_per_stage_dim(final_loop_state["state_io"])
        return jnp.reshape(out, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))