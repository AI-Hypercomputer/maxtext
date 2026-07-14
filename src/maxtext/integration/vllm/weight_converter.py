import abc
import re
import jax
import jax.numpy as jnp
import gc
from typing import List, Union, Any, Dict
from flax import traverse_util

# ==========================================
# 1. Operations
# ==========================================
class Operation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, tensors: List[Any]) -> Any: pass

class Concatenate(Operation):
    def __init__(self, dim: int): self.dim = dim
    def __call__(self, tensors): return jnp.concatenate(tensors, axis=self.dim)

class Transpose(Operation):
    def __init__(self, axes: tuple): self.axes = axes
    def __call__(self, tensors): return jnp.transpose(tensors[0], self.axes)

class UnstackScanned(Operation):
    def __init__(self, scan_axis: int = 1): self.scan_axis = scan_axis
    def __call__(self, tensors): return tuple(jnp.moveaxis(tensors[0], self.scan_axis, 0))

class SliceAxis(Operation):
    def __init__(self, axis: int, index: int):
        self.axis = axis
        self.index = index
    def __call__(self, tensors): return jnp.take(tensors[0], self.index, axis=self.axis)

class Cast(Operation):
    def __init__(self, dtype): self.dtype = dtype
    def __call__(self, tensors): return tensors[0].astype(self.dtype)

class FlattenSpatial(Operation):
    # Flattens the last two dimensions (num_heads, head_dim) into a single dimension per tensor
    def __call__(self, tensors):
        out = []
        for t in tensors:
            out.append(t.reshape(*t.shape[:-2], -1))
        return out

class PadAxes(Operation):
    def __init__(self, pad_specs): self.pad_specs = pad_specs
    def __call__(self, tensors): return jnp.pad(tensors[0], self.pad_specs)

class InterleavedPadAxes(Operation):
    def __init__(self, pad_specs): self.pad_specs = pad_specs
    def __call__(self, tensors):
        out = tensors[0]
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
        res = tensors[0]
        for axis, rep in self.repeats:
            res = jnp.repeat(res, rep, axis=axis)
        return res

# ==========================================
# 2. Rules
# ==========================================
class ExactWeightRule:
    def __init__(self, source_key: str, target_key: str, operations: List[Operation] = None):
        self.source_key = source_key
        self.target_key = target_key
        self.operations = operations or []

class ExactMultiWeightRule:
    def __init__(self, source_keys: List[str], target_key: str, operations: List[Operation] = None):
        self.source_keys = source_keys
        self.target_key = target_key
        self.operations = operations or []

class WeightRenaming:
    def __init__(self, source_pattern: str, target_pattern: str):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern

class WeightConverterRule:
    def __init__(self, source_patterns: List[str], target_pattern: str, operations: List[Operation]):
        self.source_patterns = source_patterns
        self.target_pattern = target_pattern
        self.operations = operations

# ==========================================
# 3. Engine
# ==========================================
class WeightConverter(abc.ABC):
    def __init__(self, rules: List[Union[WeightRenaming, WeightConverterRule]]):
        self.rules = rules

    def convert(self, src_pytree: Any, target_state: Any = None) -> Dict[str, Any]:
        if hasattr(src_pytree, 'to_pure_dict'):
            src_pytree = src_pytree.to_pure_dict()
            
        if isinstance(src_pytree, dict) and "base" in src_pytree:
            src_pytree = src_pytree["base"]
        
        flat_src_tuples = traverse_util.flatten_dict(src_pytree)
        flat_src = {'.'.join(str(k) for k in keys): v for keys, v in flat_src_tuples.items()}
        
        # print(f"DEBUG: src_pytree flat keys (first 10): {list(flat_src.keys())[:10]}")
        
        dst_dict = {}
        
        rules_to_apply = list(self.rules)
        if target_state is not None:
            direct_rules = build_converter_rules(src_pytree, target_state)
            print(f"DEBUG: Generated {len(direct_rules)} direct rules.")
            if len(direct_rules) > 0:
                print(f"DEBUG: First direct rule sample: {direct_rules[0].__dict__}")
            rules_to_apply.extend(direct_rules)

        for rule in rules_to_apply:
            if isinstance(rule, ExactWeightRule):
                if rule.source_key in flat_src:
                    val = flat_src[rule.source_key]
                    for op in rule.operations:
                        val = op([val])
                    dst_dict[rule.target_key] = val
            elif isinstance(rule, ExactMultiWeightRule):
                if all(k in flat_src for k in rule.source_keys):
                    tensors = [flat_src[k] for k in rule.source_keys]
                    for op in rule.operations:
                        tensors = op(tensors)
                    dst_dict[rule.target_key] = tensors
            elif isinstance(rule, WeightRenaming):
                # Compile the regex and handle simple string replacements
                source_pattern = re.compile(rule.source_pattern)
                for src_key, src_val in flat_src.items():
                    if source_pattern.search(src_key):
                        target_key = source_pattern.sub(rule.target_pattern, src_key)
                        dst_dict[target_key] = src_val
            elif isinstance(rule, WeightConverterRule):
                # Used r"layers\.\d+\.attention\.wq\.kernel"
                # matched to target_pattern=r"layers.{}.attention.qkv_proj.weight".
                import collections
                matched_groups = collections.defaultdict(list)
                
                source_regexes = [re.compile(p) for p in rule.source_patterns]
                
                # Search for all keys matching each source pattern
                for i, pattern in enumerate(source_regexes):
                    for src_key, src_val in flat_src.items():
                        m = pattern.search(src_key)
                        if m:
                            # Extract all digits to form a group key
                            digits = tuple(re.findall(r'\d+', src_key))
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
                            result = op(result)
                        
                        # Format target pattern
                        target_key = rule.target_pattern.format(*digits)
                        dst_dict[target_key] = result

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
        for k_str, val in dst_dict.items():
            if k_str in target_tuple_map:
                dst_dict_tuples[target_tuple_map[k_str]] = val
            else:
                # If not mapped by target state, just split the string stringly
                dst_dict_tuples[tuple(k_str.split('.'))] = val
                
        dst_pytree = traverse_util.unflatten_dict(dst_dict_tuples)
        return dst_pytree

# ==========================================
# 4. Registries and Builders
# ==========================================
_MODEL_TO_CONVERSION_RULES = {
    "qwen3": [
        WeightRenaming(r"token_embedder\.embedding", "model.embed_tokens.weight"),
        WeightRenaming(r"decoder\.decoder_norm\.scale", "model.norm.weight"),
        WeightRenaming(r"logits_dense\.kernel", "lm_head.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.pre_self_attention_layer_norm\.scale", r"model.layers.\1.input_layernorm.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.post_self_attention_layer_norm\.scale", r"model.layers.\1.post_attention_layernorm.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.self_attention\.out\.kernel", r"model.layers.\1.self_attn.o_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.mlp\.wo\.kernel", r"model.layers.\1.mlp.down_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.pre_self_attention_layer_norm\.scale", r"model.layers.\1.input_layernorm.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.post_self_attention_layer_norm\.scale", r"model.layers.\1.post_attention_layernorm.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.self_attention\.out\.kernel", r"model.layers.\1.self_attn.o_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.mlp\.wo\.kernel", r"model.layers.\1.mlp.down_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.self_attention\.query\.kernel", r"model.layers.\1.self_attn.q_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.self_attention\.key\.kernel", r"model.layers.\1.self_attn.k_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.self_attention\.value\.kernel", r"model.layers.\1.self_attn.v_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.self_attention\.query\.kernel", r"model.layers.\1.self_attn.q_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.self_attention\.key\.kernel", r"model.layers.\1.self_attn.k_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.self_attention\.value\.kernel", r"model.layers.\1.self_attn.v_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.mlp\.wi_0\.kernel", r"model.layers.\1.mlp.gate_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.mlp\.wi_1\.kernel", r"model.layers.\1.mlp.up_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.mlp\.wi_0\.kernel", r"model.layers.\1.mlp.gate_proj.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.mlp\.wi_1\.kernel", r"model.layers.\1.mlp.up_proj.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.self_attention\.query_norm\.scale", r"model.layers.\1.self_attn.q_norm.weight"),
        WeightRenaming(r"decoder\.layers_(\d+)\.self_attention\.key_norm\.scale", r"model.layers.\1.self_attn.k_norm.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.self_attention\.query_norm\.scale", r"model.layers.\1.self_attn.q_norm.weight"),
        WeightRenaming(r"decoder\.layers\.(\d+)\.self_attention\.key_norm\.scale", r"model.layers.\1.self_attn.k_norm.weight"),
    ],
}
def _shapes_are_repeatable(candidate_shape, tgt_shape):
    if len(candidate_shape) != len(tgt_shape): return False
    for s, t in zip(candidate_shape, tgt_shape):
        if s > t or t % s != 0: return False
    return True

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
        if tgt_key_tuple and tgt_key_tuple[0] == 'model':
            candidate_key_tuples.append(tgt_key_tuple[1:])
            candidate_key_tuples.append(('decoder',) + tgt_key_tuple[1:])
            
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
        
        # Fallbacks for fused weights
        src_keys_to_fuse = None
        if not found_src_tuple and tgt_key_tuple[-1] == 'wi':
            for cand in candidate_key_tuples:
                wi_0_cand = tuple(list(cand[:-1]) + ['wi_0'])
                wi_1_cand = tuple(list(cand[:-1]) + ['wi_1'])
                if wi_0_cand in src_flat and wi_1_cand in src_flat:
                    src_keys_to_fuse = [wi_0_cand, wi_1_cand]
                    break
        elif not found_src_tuple and tgt_key_tuple[-2:] == ('qkv_proj', 'weight'):
            for cand in candidate_key_tuples:
                prefix = list(cand[:-2])
                wq = tuple(prefix + ['wq', 'kernel'])
                wk = tuple(prefix + ['wk', 'kernel'])
                wv = tuple(prefix + ['wv', 'kernel'])
                if wq in src_flat and wk in src_flat and wv in src_flat:
                    src_keys_to_fuse = [wq, wk, wv]
                    break
        elif not found_src_tuple and tgt_key_tuple[-2:] == ('gate_up_proj', 'weight'):
            for cand in candidate_key_tuples:
                prefix = list(cand[:-2])
                wi_0 = tuple(prefix + ['wi_0', 'kernel'])
                wi_1 = tuple(prefix + ['wi_1', 'kernel'])
                if wi_0 in src_flat and wi_1 in src_flat:
                    src_keys_to_fuse = [wi_0, wi_1]
                    break

        operations = []
        
        # If we need to fuse
        if src_keys_to_fuse:
            src_val_shape = src_flat[src_keys_to_fuse[0]].shape
            if match_index != -1 and len(src_val_shape) == len(tgt_val.shape) + 1:
                # Scanned! We slice first!
                scan_axis = 0 # simple assumption
                operations.append(SliceAxis(axis=scan_axis, index=layer_idx))
                src_val_shape = src_val_shape[:scan_axis] + src_val_shape[scan_axis+1:]
                
            dim = len(src_val_shape) - 1
            operations.append(Concatenate(dim=dim))
            # new shape after concat
            src_val_shape = list(src_val_shape)
            src_val_shape[dim] *= len(src_keys_to_fuse)
            src_val_shape = tuple(src_val_shape)
            
            # Align shapes
            if src_val_shape != tgt_val.shape:
                mismatches = []
                for axis, (s, t) in enumerate(zip(src_val_shape, tgt_val.shape)):
                    if s != t: mismatches.append((axis, s, t))
                
                # wi is MoE MLP weight -> pad
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
                    
            if getattr(src_flat[src_keys_to_fuse[0]], 'dtype', None) != tgt_val.dtype:
                operations.append(Cast(tgt_val.dtype))
                
            rules.append(ExactMultiWeightRule(
                source_keys=['.'.join(str(k) for k in sk) for sk in src_keys_to_fuse],
                target_key=tgt_key_str,
                operations=operations
            ))
            
        elif found_src_tuple:
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
            
            rules.append(ExactWeightRule(
                source_key=src_key_str,
                target_key=tgt_key_str,
                operations=operations
            ))
            
    return rules
