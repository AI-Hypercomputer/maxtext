import abc
import re
import jax
import jax.numpy as jnp
import numpy as np
import gc
from typing import List, Union, Any, Dict, Optional, Callable, Mapping
from flax import traverse_util, nnx
from maxtext.integration.vllm.convert_utils import intersect_trees

# ==========================================
# 1. Operations
# ==========================================
class Operation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, tensors: List[Any], **kwargs) -> Any: pass

class Concatenate(Operation):
    def __init__(self, dim: int): self.dim = dim
    def __call__(self, tensors, **kwargs):
        @jax.jit
        def _f(*ts): return jnp.concatenate(ts, axis=self.dim)
        return _f(*tensors)

class Transpose(Operation):
    def __init__(self, axes): self.axes = axes
    def __call__(self, tensors, **kwargs):
        @jax.jit
        def _f(t): return jnp.transpose(t, self.axes)
        return _f(tensors[0])

class TransposeUnstack(Operation):
    def __init__(self, axes): self.axes = axes
    def __call__(self, tensors, **kwargs):
        @jax.jit
        def _f(t): return jnp.transpose(t, self.axes)
        return list(jnp.unstack(_f(tensors[0]), axis=0))

class AttentionOut(Operation):
    def __call__(self, tensors, **kwargs):
        @jax.jit
        def _f(t):
            # t shape: [heads?, num_layers, head_dim, d_model]
            o = jnp.transpose(t, (1, 3, 0, 2))
            return o.reshape(o.shape[0], o.shape[1], -1)
        return list(jnp.unstack(_f(tensors[0]), axis=0))

class AttentionQKV(Operation):
    def __call__(self, tensors, **kwargs):
        tp = kwargs.get('tp', 1)

        @jax.jit
        def _f(q_in, k_in, v_in):
            q = jnp.transpose(q_in, (1, 0, 2, 3))
            k = jnp.transpose(k_in, (1, 0, 2, 3))
            v = jnp.transpose(v_in, (1, 0, 2, 3))

            num_layers, d_model, num_q_heads, head_dim = q.shape
            num_kv_heads = k.shape[2]
            actual_tp = min(tp, num_kv_heads)

            kv_per_tp = num_kv_heads // actual_tp
            q_per_tp = num_q_heads // actual_tp

            q_by_tp = q.reshape(num_layers, d_model, actual_tp, q_per_tp, head_dim)
            k_by_tp = k.reshape(num_layers, d_model, actual_tp, kv_per_tp, head_dim)
            v_by_tp = v.reshape(num_layers, d_model, actual_tp, kv_per_tp, head_dim)

            qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)
            qkv_flat = qkv_by_tp.reshape(num_layers, d_model, -1)
            qkv_proj = jnp.transpose(qkv_flat, (0, 2, 1))
            return qkv_proj

        res = _f(tensors[0], tensors[1], tensors[2])
        return list(jnp.unstack(res, axis=0))

class MoEExpertDown(Operation):
    def __call__(self, tensors, **kwargs):
        @jax.jit
        def _f(t_in):
            # t_in is shape [num_experts, num_layers, d_inner, d_model]
            # transpose (1, 0, 2, 3) makes it [num_layers, num_experts, d_inner, d_model]
            return jnp.transpose(t_in, (1, 0, 2, 3))

        t_transposed = _f(tensors[0])
        return list(jnp.unstack(t_transposed, axis=0))

class MoEFuseGateUp(Operation):
    def __call__(self, tensors, **kwargs):
        tp = kwargs.get('tp', 1)

        @jax.jit
        def _fuse_all(wi_0, wi_1):
            wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))
            wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

            def _fuse_single(w0, w1):
                w0 = jnp.transpose(w0, (0, 2, 1))
                w1 = jnp.transpose(w1, (0, 2, 1))
                num_experts, d_inner, d_model = w0.shape
                chunk_size = d_inner // tp
                padded_chunk_size = ((chunk_size + 127) // 128) * 128
                pad_amount = padded_chunk_size - chunk_size
                gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
                up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
                if pad_amount > 0:
                    gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
                    up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
                combined = jnp.stack([gate_chunks, up_chunks], axis=2)
                res = combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)
                return jnp.transpose(res, (0, 2, 1))

            return jax.vmap(_fuse_single)(wi_0, wi_1)

        fused = _fuse_all(tensors[0], tensors[1])
        return list(jnp.unstack(fused, axis=0))

class MoEFuseGateUpPrefused(Operation):
    def __call__(self, tensors, **kwargs):
        tp = kwargs.get('tp', 1)

        @jax.jit
        def _fuse_all(wi):
            wi = jnp.transpose(wi, (1, 0, 2, 3))

            def _fuse_single(w):
                w = jnp.transpose(w, (0, 2, 1))
                num_experts, double_d_inner, d_model = w.shape
                d_inner = double_d_inner // 2
                w0 = w[:, :d_inner, :]
                w1 = w[:, d_inner:, :]
                chunk_size = d_inner // tp
                padded_chunk_size = ((chunk_size + 127) // 128) * 128
                pad_amount = padded_chunk_size - chunk_size
                gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
                up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
                if pad_amount > 0:
                    gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
                    up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
                combined = jnp.stack([gate_chunks, up_chunks], axis=2)
                res = combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)
                return jnp.transpose(res, (0, 2, 1))

            return jax.vmap(_fuse_single)(wi)

        fused = _fuse_all(tensors[0])
        return list(jnp.unstack(fused, axis=0))


class Identity(Operation):
    def __call__(self, tensors, **kwargs):
        return tensors[0]

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
    def __init__(self, rules: List[Rule], tp: int = 1, num_kv_heads: Optional[int] = None, head_dim: Optional[int] = None):
        self.rules = rules
        self.tp = tp
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def convert(
        self, 
        src_pytree: Any, 
        target_state: Any = None
    ) -> Any:
        def _to_pure(x):
            if hasattr(x, 'to_pure_dict'): x = x.to_pure_dict()
            if hasattr(x, 'unfreeze'): x = x.unfreeze()
            if hasattr(x, 'items'): return {k: _to_pure(v) for k, v in x.items()}
            if hasattr(x, 'value'): return x.value
            return x

        pure_src = _to_pure(src_pytree)
        flat_src = traverse_util.flatten_dict(pure_src, sep='.')

        # Free source references to prevent HBM peaks when evaluating rules
        del pure_src
        if hasattr(src_pytree, 'clear'):
            src_pytree.clear()
        gc.collect()

        # Extract pure state dictionary from the target for matching Sharding specs
        if target_state is None:
            full_target_spec = None
        elif isinstance(target_state, nnx.State):
            state_dict = target_state.to_pure_dict() if hasattr(target_state, "to_pure_dict") else dict(target_state)
            full_target_spec = _to_pure(state_dict)
        else:
            full_target_spec = _to_pure(target_state)

        # DYNAMIC ROUTING: Rules (HF) vs Direct Tree Intersection (MT)
        if self.rules:
            # 1. HuggingFace Mappings via Dedicated Rules
            self.rules = build_hf_rules(flat_src, target_state, self.rules)

            result = {}
            for rule in self.rules:
                tensors = []
                for src_pat in rule.source_patterns:
                    if src_pat in flat_src:
                        tensors.append(flat_src.pop(src_pat))
                    else:
                        pass # Ignore unfound keys gracefully
    
                if not tensors:
                    continue
    
                out = tensors
                for op in rule.operations:
                    out = op(out, tp=self.tp)
                    if not isinstance(out, list) and op != rule.operations[-1]:
                        out = [out]
    
                if isinstance(out, list) and len(out) > 1 and "{}" in rule.target_pattern:
                    for i, tensor in enumerate(out):
                        result[rule.target_pattern.format(i)] = tensor
                elif isinstance(out, list) and len(out) == 1:
                    result[rule.target_pattern] = out[0]
                else:
                    result[rule.target_pattern] = out
    
                del out
                del tensors
                gc.collect()
                
            vllm_state = traverse_util.unflatten_dict(result, sep='.')
        else:
            # 2. MaxText-to-MaxText Direct Sync (no rules entirely)
            if full_target_spec is None:
                raise ValueError(
                    "target_state must be provided when converting with empty rules (MaxText-to-MaxText direct tree intersection)."
                )
            # Repack the flat_src dictionary with tuple keys for the algorithm
            pure_src_unflat = traverse_util.unflatten_dict(flat_src, sep='.')
            has_model_prefix = isinstance(full_target_spec, dict) and "model" in full_target_spec
            if isinstance(pure_src_unflat, dict) and "base" in pure_src_unflat:
                pure_src_unflat = pure_src_unflat["base"]
            while isinstance(full_target_spec, dict) and "model" in full_target_spec:
                full_target_spec = full_target_spec["model"]
            
            final_source, _ = intersect_trees(pure_src_unflat, full_target_spec)
            vllm_state = {"model": final_source} if has_model_prefix else final_source

        return vllm_state

# ==========================================
# 4. Registries and Builders
# ==========================================
# To replace the legacy transfer_state_with_mappings()
_MODEL_TO_CONVERSION_RULES = {
    "qwen3": [],
    "qwen3_moe": [
        Rule(["base.token_embedder.embedding"], "vllm_model.model.embed_tokens.weight"),
        Rule(["base.decoder.decoder_norm.scale"], "vllm_model.model.norm.weight"),
        Rule(["base.decoder.logits_dense.kernel"], "vllm_model.lm_head.weight", [Transpose(axes=(1, 0))]),
        Rule(["base.decoder.layers.pre_self_attention_layer_norm.scale"], "vllm_model.model.layers.{}.input_layernorm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.post_self_attention_layer_norm.scale"], "vllm_model.model.layers.{}.post_attention_layernorm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.self_attention.out.kernel"], "vllm_model.model.layers.{}.self_attn.o_proj.weight", [AttentionOut()]),
        Rule(["base.decoder.layers.self_attention.query.kernel", "base.decoder.layers.self_attention.key.kernel", "base.decoder.layers.self_attention.value.kernel"], "vllm_model.model.layers.{}.self_attn.qkv_proj.weight", [AttentionQKV()]),
        Rule(["base.decoder.layers.self_attention.query_norm.scale"], "vllm_model.model.layers.{}.self_attn.q_norm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.self_attention.key_norm.scale"], "vllm_model.model.layers.{}.self_attn.k_norm.weight", [TransposeUnstack(axes=(1, 0))]),
        Rule(["base.decoder.layers.moe_block.gate.kernel"], "vllm_model.model.layers.{}.mlp.gate.weight", [TransposeUnstack(axes=(1, 2, 0))]),
        Rule(["base.decoder.layers.moe_block.wo"], "vllm_model.model.layers.{}.mlp.experts.w2_weight", [MoEExpertDown()]),
        Rule(["base.decoder.layers.moe_block.wi_0", "base.decoder.layers.moe_block.wi_1"], "vllm_model.model.layers.{}.mlp.experts.w13_weight", [MoEFuseGateUp()]),
        Rule(["base.decoder.layers.moe_block.wi"], "vllm_model.model.layers.{}.mlp.experts.w13_weight", [MoEFuseGateUpPrefused()]),
    ],
    "qwen35_moe": []
}

def build_hf_rules(flat_src: Dict[str, Any], target_state: Any, rules: List[Rule]) -> list:
    return rules
