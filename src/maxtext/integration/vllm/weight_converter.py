import abc
import re
import jax
import jax.numpy as jnp
import gc
from typing import List, Union, Any, Dict
from flax import traverse_util

_MOE_MLP_WEIGHTS = ('wi_0', 'wi_1', 'wi', 'wo', 'gate', 'w13_weight', 'w2_weight', 'wi_0.kernel', 'wi_1.kernel', 'wo.kernel', 'wi.kernel', 'gate.kernel')

# ==========================================
# 1. Operations
# ==========================================
class Operation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, tensors: List[Any]) -> Any: pass

class Concatenate(Operation):
    def __init__(self, dim: int): self.dim = dim
    def __call__(self, tensors): return jnp.concatenate(tensors, axis=self.dim)

class SliceAxis(Operation):
    def __init__(self, axis: int, index: int):
        self.axis = axis
        self.index = index
    def __call__(self, tensors): 
        t = tensors[0] if isinstance(tensors, list) else tensors
        return jnp.take(t, self.index, axis=self.axis)

class Cast(Operation):
    def __init__(self, dtype): self.dtype = dtype
    def __call__(self, tensors): 
        t = tensors[0] if isinstance(tensors, list) else tensors
        return t.astype(self.dtype)

class PadAxes(Operation):
    def __init__(self, pad_specs): self.pad_specs = pad_specs
    def __call__(self, tensors): 
        t = tensors[0] if isinstance(tensors, list) else tensors
        return jnp.pad(t, self.pad_specs)

class InterleavedPadAxes(Operation):
    def __init__(self, pad_specs): self.pad_specs = pad_specs
    def __call__(self, tensors):
        out = tensors[0] if isinstance(tensors, list) else tensors
        for axis, n_shards, per_shard_extra in self.pad_specs:
            if per_shard_extra <= 0: continue
            src_dim = out.shape[axis]
            src_chunk_size = src_dim // n_shards
            split_shape = list(out.shape)
            split_shape.insert(axis + 1, src_chunk_size)
            split_shape[axis] = n_shards
            arr_split = out.reshape(split_shape)
            pad_width = [(0, 0)] * arr_split.ndim
            pad_width[axis + 1] = (0, per_shard_extra)
            arr_padded = jnp.pad(arr_split, pad_width)
            final_shape = list(out.shape)
            final_shape[axis] = src_dim + per_shard_extra * n_shards
            out = arr_padded.reshape(final_shape)
        return out

class RepeatAxes(Operation):
    def __init__(self, repeats): self.repeats = repeats
    def __call__(self, tensors): 
        res = tensors[0] if isinstance(tensors, list) else tensors
        for axis, rep in self.repeats:
            res = jnp.repeat(res, rep, axis=axis)
        return res


class TransposeSingle(Operation):
    def __init__(self, axes): self.axes = axes
    def __call__(self, tensors):
        out = tensors[0] if isinstance(tensors, list) else tensors
        return jnp.transpose(out, self.axes)

class Transpose(Operation):
    def __init__(self, axes): self.axes = axes
    def __call__(self, tensors):
        if isinstance(tensors, list):
            return [jnp.transpose(t, self.axes) for t in tensors]
        else:
            return jnp.transpose(tensors, self.axes)

# interleaving logic as _make_attn_compute() in qwen3_moe.py
class FuseQwen3MoEGateUp(Operation):
    def __init__(self, tp: int): 
        # tp is config.rollout_tensor_parallelism
        self.tp = tp
        
    def __call__(self, tensors):
        w0, w1 = tensors
        # [experts, d_model, d_inner] -> [experts, d_inner, d_model]
        w0 = jnp.transpose(w0, (0, 2, 1))
        w1 = jnp.transpose(w1, (0, 2, 1))
        num_experts, d_inner, d_model = w0.shape
        chunk_size = d_inner // self.tp
        
        # Pad each TP chunk to the next multiple of 128 for TPU GMM alignment
        padded_chunk_size = ((chunk_size + 127) // 128) * 128
        pad_amount = padded_chunk_size - chunk_size
        
        gate_chunks = w0.reshape(num_experts, self.tp, chunk_size, d_model)
        up_chunks = w1.reshape(num_experts, self.tp, chunk_size, d_model)
        
        if pad_amount > 0:
            gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
            up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
            
        combined = jnp.stack([gate_chunks, up_chunks], axis=2)
        fused = combined.reshape(num_experts, 2 * padded_chunk_size * self.tp, d_model)
        return jnp.transpose(fused, (0, 2, 1))

# logic as _make_fuse_all() in qwen3_moe.py
class FuseQwen3MoEQKV(Operation):
    def __init__(self, tp: int): 
        self.tp = tp
        
    def __call__(self, tensors):
        q, k, v = tensors
        # MaxText Qwen3 scanned attention kernels are (d_model, num_heads, head_dim)
        d_model, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]
        
        # Transpose to [num_heads, head_dim, d_model] for HF packing
        q = jnp.transpose(q, (1, 2, 0))
        k = jnp.transpose(k, (1, 2, 0))
        v = jnp.transpose(v, (1, 2, 0))

        # Attention caps TP to the number of KV Heads
        # different from moe gate
        tp = min(self.tp, num_kv_heads)
            
        q_per_tp = num_q_heads // tp
        kv_per_tp = num_kv_heads // tp
        
        q_by_tp = q.reshape(tp, q_per_tp, head_dim, d_model)
        k_by_tp = k.reshape(tp, kv_per_tp, head_dim, d_model)
        v_by_tp = v.reshape(tp, kv_per_tp, head_dim, d_model)
        
        qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=1)
        qkv_flat = qkv_by_tp.reshape(-1, d_model)
        return qkv_flat

class TransposeAttentionOut(Operation):
    def __call__(self, tensors):
        out = tensors[0] if isinstance(tensors, list) else tensors
        if len(out.shape) == 3:
            return jnp.transpose(out.reshape(-1, out.shape[2]), (1, 0))
        return jnp.transpose(out)

class TransposeNorm(Operation):
    def __call__(self, tensors):
        return jnp.transpose(tensors[0] if isinstance(tensors, list) else tensors)

# ==========================================
# 2. Rule
# ==========================================
class Rule:
    """Unified rule format for converting weights."""
    def __init__(self, source_patterns: Union[str, List[str]], target_pattern: str, operations: List[Operation] = None):
        if isinstance(source_patterns, str):
            self.source_patterns = [source_patterns]
        else:
            self.source_patterns = source_patterns
        self.target_pattern = target_pattern
        self.operations = operations or []

# ==========================================
# 3. Engine
# ==========================================
class WeightConverter(abc.ABC):
    def __init__(self, rules: List[Rule], tp: int = 1):
        self.rules = rules
        self.tp = tp

    def convert(self, src_pytree: Any, target_state: Any = None) -> Dict[str, Any]:
        if target_state is not None:
            # Attempt to dynamically extract maximum tp from target_state's sharding mesh
            try:
                from flax.traverse_util import flatten_dict
                ts_dict = target_state.to_pure_dict() if hasattr(target_state, "to_pure_dict") else dict(target_state)
                flat_ts = flatten_dict(ts_dict)
                max_tp = self.tp
                for k, v in flat_ts.items():
                    if hasattr(v, "sharding") and hasattr(v.sharding, "mesh") and hasattr(v.sharding.mesh, "shape"):
                        mesh_shape = v.sharding.mesh.shape
                        if "tensor" in mesh_shape:
                            max_tp = max(max_tp, mesh_shape["tensor"])
                
                if max_tp != self.tp:
                    self.tp = max_tp
                    print(f"DEBUG: Dynamically extracted tp={self.tp} from target_state sharding max.")
            except Exception as e:
                print(f"DEBUG: Failed to extract tp from target_state: {e}")

        if hasattr(src_pytree, 'to_pure_dict'):
            src_pytree = src_pytree.to_pure_dict()
            
        if isinstance(src_pytree, dict) and "base" in src_pytree:
            src_pytree = src_pytree["base"]
        
        flat_src_tuples = traverse_util.flatten_dict(src_pytree)
        flat_src = {'.'.join(str(k) for k in keys): v for keys, v in flat_src_tuples.items()}
        
        # Unroll scanned layers seamlessly
        unrolled_src = {}
        import numpy as np
        for src_key, src_val in flat_src.items():
            if "layers." in src_key and ".layers_" not in src_key:
                # Scanned! We need to unstack
                # MaxText scan parameters for Qwen3 appear at index 1 uniformly
                layer_dim = 1
                if hasattr(src_val, 'shape') and len(src_val.shape) > layer_dim:
                    num_layers = src_val.shape[layer_dim]
                    for i in range(num_layers):
                        new_key = src_key.replace("layers.", f"layers_{i}.")
                        unrolled_src[new_key] = src_val[:, i]
                else:
                    unrolled_src[src_key] = src_val
            else:
                unrolled_src[src_key] = src_val
        flat_src = unrolled_src
        
        dst_dict = {}
        
        rules_to_apply = list(self.rules)
        if target_state is not None and not self.rules:
            direct_rules = build_converter_rules(src_pytree, target_state)
            print(f"DEBUG: Generated {len(direct_rules)} direct rules.")
            if len(direct_rules) > 0:
                print(f"DEBUG: First direct rule sample: {direct_rules[0].__dict__}")
            rules_to_apply.extend(direct_rules)

        for rule in rules_to_apply:
            import collections
            matched_groups = collections.defaultdict(list)
            
            source_regexes = [re.compile(p) for p in rule.source_patterns]
            
            # Search for all keys matching each source pattern
            for i, pattern in enumerate(source_regexes):
                for src_key, src_val in flat_src.items():
                    m = pattern.search(src_key)
                    if m:
                        # Extract all digits to form a group key for matching multiple source patterns
                        digits = tuple(re.findall(r'\d+', src_key))
                        if len(rule.source_patterns) == 1:
                            # For single source pattern without groups, just use the string as is to replace
                            if not digits:
                                target_key = pattern.sub(rule.target_pattern, src_key)
                                dst_dict[target_key] = src_val
                                break
                            
                        matched_groups[digits].append((i, src_val))
            
            for digits, matched_items in matched_groups.items():
                # Check if we have all required source patterns matched
                if len(matched_items) == len(rule.source_patterns):
                    # Sort by index i to pass tensors in the correct order
                    matched_items.sort(key=lambda x: x[0])
                    tensors = [x[1] for x in matched_items]
                    
                    # Apply operations
                    result = tensors
                    for op in rule.operations:
                        # inject the correct tp, override the default tp=0
                        if (isinstance(op, FuseQwen3MoEGateUp) or isinstance(op, FuseQwen3MoEQKV)) and op.tp == 0:
                            op.tp = self.tp
                        result = op(result)
                    
                    if len(rule.source_patterns) == 1 and not rule.operations and digits:
                        # Simple renaming with grouped strings e.g. "model.layers.\1.input_layernorm.weight"
                        src_key = next(k for k in flat_src.keys() if source_regexes[0].search(k) and tuple(re.findall(r'\d+', k)) == digits)
                        target_key = source_regexes[0].sub(rule.target_pattern, src_key)
                    else:
                        # Format target pattern
                        if '{}' in rule.target_pattern:
                            target_key = rule.target_pattern.format(*digits)
                        else:
                            target_key = rule.target_pattern
                    
                    dst_dict[target_key] = result[0] if isinstance(result, list) and len(result) == 1 else result

        target_tuple_map = {}
        if target_state is not None:
            def to_pure_node(node):
                if hasattr(node, 'to_pure_dict'): node = node.to_pure_dict()
                if isinstance(node, dict): return {k: to_pure_node(v) for k, v in node.items()}
                if hasattr(node, 'value'): return node.value
                return node
            
            tgt_flat = traverse_util.flatten_dict(to_pure_node(target_state))
            for true_tuple in tgt_flat.keys():
                key_str = '.'.join(str(k) for k in true_tuple)
                target_tuple_map[key_str] = true_tuple

        dst_dict_tuples = {}
        if target_state is not None:
            dst_dict_tuples = dict(tgt_flat)

        for k_str, val in dst_dict.items():
            if k_str in target_tuple_map:
                dst_dict_tuples[target_tuple_map[k_str]] = val
            elif f"vllm_model.{k_str}" in target_tuple_map:
                dst_dict_tuples[target_tuple_map[f"vllm_model.{k_str}"]] = val
            else:
                # If not mapped by target state, just split the string stringly
                dst_dict_tuples[tuple(k_str.split('.'))] = val
                
        dst_pytree = traverse_util.unflatten_dict(dst_dict_tuples)
        return dst_pytree

# ==========================================
# 4. Registries and Builders
# ==========================================
# To replace the legacy transfer_state_with_mappings()
_MODEL_TO_CONVERSION_RULES = {
    "qwen3": [
        Rule(r"token_embedder\.embedding", "model.embed_tokens.weight"),
        Rule(r"decoder\.decoder_norm\.scale", "model.norm.weight"),
        Rule([r"(?:decoder\.)?logits_dense\.kernel"], "lm_head.weight", [TransposeAttentionOut()]),
        Rule(r"decoder\.layers_(\d+)\.pre_self_attention_layer_norm\.scale", r"model.layers.\1.input_layernorm.weight"),
        Rule(r"decoder\.layers_(\d+)\.post_self_attention_layer_norm\.scale", r"model.layers.\1.post_attention_layernorm.weight"),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.out\.kernel"], r"model.layers.{}.self_attn.o_proj.weight", [TransposeAttentionOut()]),
        Rule([r"decoder\.layers_(\d+)\.mlp\.wo(?:\.kernel)?"], r"model.layers.{}.mlp.down_proj.weight", [TransposeAttentionOut()]),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.query\.kernel", r"decoder\.layers_(\d+)\.self_attention\.key\.kernel", r"decoder\.layers_(\d+)\.self_attention\.value\.kernel"], r"model.layers.{}.self_attn.qkv_proj.weight", [FuseQwen3MoEQKV(tp=0)]),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.query_norm\.scale"], r"model.layers.{}.self_attn.q_norm.weight", [TransposeNorm()]),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.key_norm\.scale"], r"model.layers.{}.self_attn.k_norm.weight", [TransposeNorm()]),
        Rule([r"decoder\.layers_(\d+)\.mlp\.wi_0(?:\.kernel)?", r"decoder\.layers_(\d+)\.mlp\.wi_1(?:\.kernel)?"], r"model.layers.{}.mlp.gate_up_proj.weight", [Concatenate(dim=2)]),
    ],
    "qwen3_moe": [
        Rule(r"token_embedder\.embedding", "model.embed_tokens.weight"),
        Rule(r"decoder\.decoder_norm\.scale", "model.norm.weight"),
        Rule([r"(?:decoder\.)?logits_dense\.kernel"], "lm_head.weight", [TransposeAttentionOut()]),
        Rule(r"decoder\.layers_(\d+)\.pre_self_attention_layer_norm\.scale", r"model.layers.\1.input_layernorm.weight"),
        Rule(r"decoder\.layers_(\d+)\.post_self_attention_layer_norm\.scale", r"model.layers.\1.post_attention_layernorm.weight"),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.out\.kernel"], r"model.layers.{}.self_attn.o_proj.weight", [TransposeAttentionOut()]),
        Rule([r"decoder\.layers_(\d+)\.mlp\.wo(?:\.kernel)?"], r"model.layers.{}.mlp.down_proj.weight", [TransposeAttentionOut()]),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.query\.kernel", r"decoder\.layers_(\d+)\.self_attention\.key\.kernel", r"decoder\.layers_(\d+)\.self_attention\.value\.kernel"], r"model.layers.{}.self_attn.qkv_proj.weight", [FuseQwen3MoEQKV(tp=0)]),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.query_norm\.scale"], r"model.layers.{}.self_attn.q_norm.weight", [TransposeNorm()]),
        Rule([r"decoder\.layers_(\d+)\.self_attention\.key_norm\.scale"], r"model.layers.{}.self_attn.k_norm.weight", [TransposeNorm()]),
        Rule([r"decoder\.layers_(\d+)\.moe_block\.gate\.kernel"], r"model.layers.{}.mlp.gate.weight", [TransposeSingle(axes=(1, 0))]),
        Rule(
            source_patterns=[r"decoder\.layers_(\d+)\.moe_block\.wo(?:\.kernel)?"],
            target_pattern="model.layers.{}.mlp.experts.w2_weight",
            operations=[]
        ),
        Rule(
            source_patterns=[
                r"decoder\.layers_(\d+)\.moe_block\.wi_0(?:\.kernel)?",
                r"decoder\.layers_(\d+)\.moe_block\.wi_1(?:\.kernel)?"
            ],
            target_pattern="model.layers.{}.mlp.experts.w13_weight",
            operations=[FuseQwen3MoEGateUp(tp=0)] # tp is dynamically injected after this
        )
    ]
}


def _shapes_are_repeatable(candidate_shape, tgt_shape):
    if len(candidate_shape) != len(tgt_shape): return False
    for s, t in zip(candidate_shape, tgt_shape):
        if s > t or t % s != 0: return False
    return True

# inspects the shapes to perform routine dimensional adjustments  
# (padding, repeating, and sharding) dynamically to fit the target mesh
def build_converter_rules(src_pytree, target_state) -> list:
    # Unwrap Source
    if isinstance(src_pytree, dict) and 'base' in src_pytree:
        src_pytree = src_pytree['base']
        
    src_flat_raw = traverse_util.flatten_dict(src_pytree)
    src_flat = {tuple(str(k) for k in keys): v for keys, v in src_flat_raw.items()}
    
    # Helper to strip nnx Params
    def to_pure_spec(node):
        if hasattr(node, 'to_pure_dict'): node = node.to_pure_dict()
        if isinstance(node, dict): return {k: to_pure_spec(v) for k, v in node.items()}
        if hasattr(node, 'value'): return node.value
        return node
        
    tgt_flat_raw = traverse_util.flatten_dict(to_pure_spec(target_state))
    tgt_flat = {tuple(str(k) for k in keys): v for keys, v in tgt_flat_raw.items()}
    rules = []
    
    def get_sharding_spec(tgt_sharding, axis):
        if not hasattr(tgt_sharding, 'spec'): return None
        return tgt_sharding.spec[axis] if axis < len(tgt_sharding.spec) else None
        
    def get_partition_size(partition, mesh):
        if partition is None: return 1
        names = (partition,) if isinstance(partition, str) else tuple(partition)
        size = 1
        for n in names: size *= mesh.shape[n]
        return size
    
    for tgt_key_tuple, tgt_val in tgt_flat.items():
        tgt_key_str = '.'.join(str(k) for k in tgt_key_tuple)
        
        # Determine candidate source keys
        candidate_key_tuples = [tgt_key_tuple]
        
        # If target has a vllm_model wrapper (e.g. from vllm tpu inference wrapper)
        inner_tgt = tgt_key_tuple
        if inner_tgt and inner_tgt[0] == 'vllm_model':
            inner_tgt = inner_tgt[1:]
            candidate_key_tuples.append(inner_tgt)

        if inner_tgt and inner_tgt[0] == 'model':
            candidate_key_tuples.append(inner_tgt[1:])
            candidate_key_tuples.append(('decoder',) + inner_tgt[1:])
            
        # Scanned logic
        layer_idx = -1
        match_index = -1
        is_sequential = False
        for i, part in enumerate(tgt_key_tuple):
            if isinstance(part, str) and part.startswith('layers_'):
                m = re.match(r'^layers_(\d+)$', part)
                if m:
                    layer_idx = int(m.group(1))
                    match_index = i
                    break
            elif isinstance(part, str) and part == 'layers' and i + 1 < len(tgt_key_tuple):
                next_part = tgt_key_tuple[i + 1]
                if isinstance(next_part, (int, str)) and str(next_part).isdigit():
                    layer_idx = int(next_part)
                    match_index = i
                    is_sequential = True
                    break
                    
        if match_index != -1:
            cand_a_scanned = list(tgt_key_tuple)
            cand_b_scanned = list(tgt_key_tuple)
            
            if is_sequential:
                cand_a_scanned.pop(match_index + 1)
                cand_b_scanned.pop(match_index + 1)
            else:
                cand_a_scanned[match_index] = 'layers'
                cand_b_scanned.pop(match_index)
                
            base_cands = [list(tgt_key_tuple), cand_a_scanned]
            
            for cand in base_cands:
                if tgt_key_tuple[0] == 'model':
                    candidate_key_tuples.append(tuple(cand[1:]))
                    candidate_key_tuples.append(('decoder',) + tuple(cand[1:]))
                candidate_key_tuples.append(tuple(cand))
            
        # Add .kernel and .scale fallback if target ends with .weight
        extended_candidates = []
        for cand in candidate_key_tuples:
            extended_candidates.append(cand)
            if cand and cand[-1] == 'weight':
                cand_kernel = tuple(list(cand[:-1]) + ['kernel'])
                extended_candidates.append(cand_kernel)
                cand_scale = tuple(list(cand[:-1]) + ['scale'])
                extended_candidates.append(cand_scale)
        candidate_key_tuples = extended_candidates
            
        found_src_tuple = None
        for cand in candidate_key_tuples:
            if cand in src_flat:
                found_src_tuple = cand
                break
        
        operations = []
        
        if found_src_tuple:
            src_val = src_flat[found_src_tuple]
            src_shape = src_val.shape
            
            # Unstack scanned?
            if match_index != -1 and len(src_shape) == len(tgt_val.shape) + 1:
                scan_axis = None
                for i in range(len(src_shape)):
                    candidate = src_shape[:i] + src_shape[i + 1 :]
                    if _shapes_are_repeatable(candidate, tgt_val.shape):
                        scan_axis = i
                        break
                if scan_axis is not None:
                    operations.append(SliceAxis(axis=scan_axis, index=layer_idx))
                    src_shape = src_shape[:scan_axis] + src_shape[scan_axis+1:]
            
            # Align shapes
            if src_shape != tgt_val.shape:
                mismatches = []
                for axis, (s, t) in enumerate(zip(src_shape, tgt_val.shape)):
                    if s != t: mismatches.append((axis, s, t))
                
                if tgt_key_tuple[-1] in _MOE_MLP_WEIGHTS:
                    tgt_sharding = getattr(tgt_val, 'sharding', None)
                    if hasattr(tgt_sharding, 'mesh'):
                        pad_specs = []
                        for axis, s, t in mismatches:
                            n_shards = get_partition_size(get_sharding_spec(tgt_sharding, axis), tgt_sharding.mesh)
                            pad_specs.append((axis, n_shards, (t - s) // n_shards))
                        operations.append(InterleavedPadAxes(pad_specs))
                    else:
                        pad_specs = [(axis, 1, t - s) for axis, s, t in mismatches]
                        operations.append(PadAxes(pad_specs))
                else:
                    repeats = [(axis, t // s) for axis, s, t in mismatches]
                    operations.append(RepeatAxes(repeats))
                    
            if getattr(src_val, 'dtype', None) != tgt_val.dtype:
                operations.append(Cast(tgt_val.dtype))
                
            # Emit exact rule
            src_key_str = '.'.join(str(k) for k in found_src_tuple)
            
            rules.append(Rule(
                source_patterns=src_key_str,
                target_pattern=tgt_key_str,
                operations=operations
            ))
            
    return rules
