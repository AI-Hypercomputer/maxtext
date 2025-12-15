import functools
from typing import Any, Optional, Dict, Type, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import nnx
import flax.linen as nn_linen # Only for logical_to_mesh_sharding helper

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
    if mesh is None:
        return x
    sharding_or_spec = nn_linen.logical_to_mesh_sharding(
        PartitionSpec(*logical_axis_names), mesh=mesh, rules=rules
    )
    if isinstance(sharding_or_spec, NamedSharding):
        return jax.lax.with_sharding_constraint(x, sharding_or_spec)
    elif isinstance(sharding_or_spec, PartitionSpec):
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, sharding_or_spec))
    else:
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
        
        # Pipeline Dimensions
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

        # --- Flattening / Initialization Logic ---
        num_repeats = self.config.num_pipeline_repeats if self.config.num_pipeline_repeats > 1 else 1
        LayerCls = type(layers)
        
        # Extract init kwargs from the template layer
        kwargs = {}
        for attr in ['decoder_layer', 'num_decoder_layers', 'quant', 'model_mode']:
            if hasattr(layers, attr):
                kwargs[attr] = getattr(layers, attr)
        
        if rngs is None:
            raise ValueError("Pipeline requires 'rngs' to initialize stage parameters.")

        # Robust RNG Bulk Splitting
        total_layers = num_repeats * self.num_stages
        layer_keys_dicts = [{} for _ in range(total_layers)]
        target_keys = ['params', 'dropout', 'aqt', 'gate', 'random']
        
        for name in target_keys:
            if name in rngs:
                root_key = rngs[name]()
                split_keys = jax.random.split(root_key, total_layers)
                for i in range(total_layers):
                    layer_keys_dicts[i][name] = split_keys[i]

        repeats_list = []
        for r_idx in range(num_repeats):
            stages_list = []
            for s_idx in range(self.num_stages):
                flat_idx = r_idx * self.num_stages + s_idx
                stage_rngs_dict = layer_keys_dicts[flat_idx]
                layer_rngs = nnx.Rngs(**stage_rngs_dict)
                
                # --- FIX: Calculate Layer Index Offset ---
                # Ensure layers in subsequent stages continue the index sequence
                stage_kwargs = kwargs.copy()
                if 'num_decoder_layers' in kwargs:
                    # e.g., if 4 layers per stage: Stage 0 is 0-3, Stage 1 is 4-7
                    start_layer = s_idx * kwargs['num_decoder_layers']
                    stage_kwargs['layer_idx'] = start_layer

                # Instantiate independent layer for this stage
                new_layer = LayerCls(config=self.config, mesh=self.mesh, rngs=layer_rngs, **stage_kwargs)
                stages_list.append(new_layer)
            repeats_list.append(nnx.List(stages_list))
        
        self.layers = nnx.List(repeats_list) # shape: [repeats, stages]
        self.template_layer = layers # Keep reference

    def need_circ_storage(self):
        return (self.config.num_pipeline_repeats > 1 and 
                self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay)

    def iterations_to_complete_first_microbatch_one_repeat(self):
        return self.forwarding_delay * (self.num_stages - 1)

    def iterations_to_complete_first_microbatch(self):
        return (self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + 
                self.iterations_to_complete_first_microbatch_one_repeat())

    def get_pipeline_remat_policy(self):
        if self.config.remat_policy == "custom":
            return self.remat_policy
        save_input = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
        return (jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input) 
                if self.remat_policy else save_input)

    # --- Fix #3: Full State Spec ---
    def get_weight_sharding(self, *args, **kwargs):
        def get_spec(leaf):
            if hasattr(leaf, 'sharding') and isinstance(leaf.sharding, NamedSharding):
                return leaf.sharding.spec
            return None
        
        # Capture ALL state (params, batch_stats, quantization state, etc.)
        return jax.tree.map(get_spec, nnx.state(self.layers))

    def all_gather_over_fsdp(self):
        """
        Fix #2: Transforms the module variables in-place to enforce 'AG' (All-Gather) 
        sharding constraints, effectively stripping FSDP axes.
        """
        def apply_ag_constraint(leaf):
            if hasattr(leaf, 'sharding') and isinstance(leaf.sharding, NamedSharding):
                # 1. Get Physical Spec (Mesh + Spec)
                current_mesh = leaf.sharding.mesh
                current_spec = leaf.sharding.spec
                
                # 2. Strip FSDP from the spec
                new_spec = _strip_spec(current_spec)
                
                # 3. Create new target sharding
                target_sharding = NamedSharding(current_mesh, new_spec)
                
                # 4. Apply constraint (forces gather)
                return jax.lax.with_sharding_constraint(leaf, target_sharding)
            return leaf

        # Update the module state in-place with the constrained variables
        full_state = nnx.state(self.layers)
        new_state = jax.tree.map(apply_ag_constraint, full_state)
        nnx.update(self.layers, new_state)

    # --- Loop Logic Helpers ---
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
        
        return {
            "state_io": state_io, 
            "shift": shift, 
            "circ_storage": circ_storage, 
            "circ_storage_mover": circ_mover, 
            "loop_iteration": jnp.array(0, dtype=jnp.int32), 
            "prev_outputs": prev_outputs
        }

    def get_iteration_inputs(self, loop_iter, state_io, circ_storage, shift):
        state_io_slice = state_io[:, loop_iter % self.microbatches_per_stage]
        circ_in = circ_storage[:, loop_iter % self.config.num_pipeline_microbatches] if self.use_circ_storage else shift
        first_in = jnp.where(loop_iter < self.config.num_pipeline_microbatches, state_io_slice, circ_in)
        stages_in = jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_in, shift)
        
        return with_logical_constraint(stages_in, ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"), self.config.logical_axis_rules, self.mesh)

    def get_new_loop_state(self, output, loop_state):
        loop_iter = loop_state["loop_iteration"]
        
        def _rotate_right(a):
            return jnp.concatenate([jax.lax.slice_in_dim(a, self.num_stages - 1, self.num_stages, axis=0), jax.lax.slice_in_dim(a, 0, self.num_stages - 1, axis=0)], axis=0)
        
        def _shift_right(a):
            return jax.lax.slice(jnp.pad(a, [[1, 0]] + [[0, 0]] * (a.ndim - 1)), [0] * a.ndim, a.shape)

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
        
        return {
            "state_io": new_state_io, 
            "shift": new_shift, 
            "circ_storage": new_circ, 
            "circ_storage_mover": new_mover, 
            "loop_iteration": loop_iter + 1, 
            "prev_outputs": new_prev
        }

    def permute_output_micro_per_stage_dim(self, output):
        idx0 = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
        perm = (np.arange(self.microbatches_per_stage) + idx0) % self.microbatches_per_stage
        return output[:, perm]

    # --- MAIN CALL ---

    def __call__(
        self, 
        inputs: jnp.ndarray, 
        segment_ids: Optional[jnp.ndarray] = None, 
        positions: Optional[jnp.ndarray] = None, 
        deterministic: bool = False, 
        model_mode = MODEL_MODE_TRAIN, 
        partition_spec = None
    ):
        # 0. Convert inputs
        inputs = jnp.asarray(inputs)
        if positions is not None: positions = jnp.asarray(positions)
        if segment_ids is not None: segment_ids = jnp.asarray(segment_ids)

        # 1. Reshape Inputs
        inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length, self.config.emb_dim))
        if positions is not None: positions = positions.reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))
        if segment_ids is not None: segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length))

        # --- Fix #2: Apply FSDP All-Gather Once ---
        if self.config.pipeline_fsdp_ag_once:
             # This forces the model variables to be gathered from FSDP shards
             self.all_gather_over_fsdp()

        # 2. Init Loop State
        loop_state = self.init_loop_state(inputs)

        # 3. Prepare Flattened Modules
        flattened_modules = []
        if self.config.num_pipeline_repeats > 1:
            for r in range(self.config.num_pipeline_repeats):
                for s in range(self.num_stages):
                    flattened_modules.append(self.layers[r][s])
        else:
            for s in range(self.num_stages):
                flattened_modules.append(self.layers[0][s])

        # 4. Define Scan Function
        def scan_fn(carry, _):
            loop_iter = carry["loop_iteration"]
            stages_inputs = self.get_iteration_inputs(loop_iter, carry["state_io"], carry["circ_storage"], carry["shift"])
            stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
            
            micro_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iter)
            
            s_pos = positions[micro_ids] if positions is not None else None
            s_seg = segment_ids[micro_ids] if segment_ids is not None else None
            
            in_axes_seg = 0 if s_seg is not None else None
            in_axes_pos = 0 if s_pos is not None else None

            # 5. VMAP with Switch logic to handle heterogeneous stage modules
            def run_stage_logic(x, seg, pos, stage_idx, repeat_idx):
                if self.config.num_pipeline_repeats > 1:
                    target_idx = repeat_idx * self.num_stages + stage_idx
                else:
                    target_idx = stage_idx
                
                target_idx = jnp.clip(target_idx, 0, len(flattened_modules) - 1)
                
                branches = []
                for mod in flattened_modules:
                    def _branch(inputs, module=mod): 
                        x_i, seg_i, pos_i = inputs
                        return module(x_i, decoder_segment_ids=seg_i, decoder_positions=pos_i, deterministic=deterministic, model_mode=model_mode)
                    branches.append(_branch)
                
                return jax.lax.switch(target_idx, branches, (x, seg, pos))

            stage_indices = jnp.arange(self.num_stages)
            stages_out = nnx.vmap(run_stage_logic, in_axes=(0, in_axes_seg, in_axes_pos, 0, 0), out_axes=0)(
                stages_inputs, s_seg, s_pos, stage_indices, repeat_ids
            )
            
            if self.config.scan_layers:
                 # Flattened modules often return (out, kv), we only need out
                 stages_out = stages_out[0]

            return self.get_new_loop_state(stages_out, carry), None

        # 6. Execute Scan
        total_steps = (self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats) + \
                      self.forwarding_delay * (self.num_stages - 1)
        
        if self.config.scan_pipeline_iterations:
             policy = self.get_pipeline_remat_policy() if self.config.set_remat_policy_on_pipeline_iterations else None
             scan_fn = jax.checkpoint(scan_fn, policy=policy, prevent_cse=not self.config.scan_pipeline_iterations)
             
        final_loop_state, _ = jax.lax.scan(scan_fn, loop_state, None, length=total_steps)
        
        out = self.permute_output_micro_per_stage_dim(final_loop_state["state_io"])
        
        return jnp.reshape(out, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))