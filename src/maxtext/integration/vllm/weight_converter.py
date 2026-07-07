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

# ==========================================
# 2. Rules
# ==========================================
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

    def convert(self, src_pytree: Any) -> Dict[str, Any]:
        flat_src = traverse_util.flatten_dict(src_pytree, sep='.')
        dst_dict = {}

        for rule in self.rules:
            if isinstance(rule, WeightRenaming):
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

        return traverse_util.unflatten_dict(dst_dict, sep='.')

# ==========================================
# 4. Registries and Builders
# ==========================================
_MODEL_TO_CONVERSION_RULES = {
    "qwen3_moe": [
        WeightConverterRule(
            source_patterns=[r"layers\.(\d+)\.attention\.wq\.kernel", r"layers\.(\d+)\.attention\.wk\.kernel", r"layers\.(\d+)\.attention\.wv\.kernel"],
            target_pattern=r"layers.{}.attention.qkv_proj.weight",
            operations=[Concatenate(dim=-1)]
        ),
        WeightRenaming(r"lm_head\.kernel", "lm_head.weight"),
    ],
}

def build_torchax_rules(mapping_config) -> list:
    return [WeightRenaming(src, tgt) for src, tgt in mapping_config.items()]

def build_maxtext_direct_rules(src_pytree, target_state) -> list:
    # Dynamic structural intersection logic
    return []
