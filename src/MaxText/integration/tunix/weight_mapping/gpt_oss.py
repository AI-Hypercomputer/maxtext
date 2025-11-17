# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the weight mapping from MaxText's GPT-OSS MoE model to a vLLM-compatible format."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GPTOSS_VLLM_MAPPING:
    """Mapping MaxText GPT-OSS MoE weights to vLLM's GPT-OSS MoE weights."""

    @staticmethod
    def to_hf_hook_fns() -> Dict[str, callable]:
        """Returns a dictionary of hook functions to be applied to MaxText weights.

        Hooks are needed to combine Q/K/V kernels, combine MLP projections, 
        and handle the MoE expert dimension.
        """
        # NOTE: Actual implementation of these functions would be in a separate utility module.
        # For this mapping, we define the *need* for them.
        return {
            "combine_qkv_kernels": "model.layers.*.attn.qkv_proj.weight",
            "combine_qkv_biases": "model.layers.*.attn.qkv_proj.bias",
            "combine_mlp_w13_kernels": "model.layers.*.mlp.experts.w13_weight",
            "combine_mlp_w13_biases": "model.layers.*.mlp.experts.w13_bias",
        }

    @staticmethod
    def to_hf_transpose_keys() -> Dict[str, Tuple[int, int]]:
        """Returns a dictionary of keys for weights that need to be transposed."""
        # MaxText often uses [in_dim, out_dim] while HuggingFace/PyTorch uses [out_dim, in_dim].
        # In a distributed MaxText checkpoint, dimensions are often sharded (e.g., [sharded_dim, other_dim]).
        # The common practice is to transpose all kernel weights.
        return {
            # Standard Kernels
            "base.decoder.logits_dense.kernel": (0, 1),
            # Attention Projections (MaxText's out.kernel needs transposition)
            "base.decoder.layers.attention.out.kernel": (0, 1),
            # MLP Down-Projection (wo.kernel)
            "base.decoder.layers.mlp.wo.kernel": (2, 3),  # (Expert, Layer, Out, In) -> (Expert, Layer, In, Out)
            # Embedding Layer (transpose usually required for the vocabulary dim)
            "base.token_embedder.embedding": (0, 1),
        }

    @staticmethod
    def to_hf_mapping() -> Dict[str, Tuple[str, Tuple[str | None, ...]]]:
        """Mapping from MaxText model to HuggingFace vLLM model."""
        return {
            # --- Global Parameters ---
            # Token Embeddings (vLLM splits embedding and final LM head) - Shard Vocab Dim
            "params.params.token_embedder.embedding": (
                "vllm_model.model.embedding.weight",
                ("vocab", "model"),
            ),
            # Final Layer Norm (Decoder Norm) - No Sharding
            "params.params.decoder.decoder_norm.scale": (
                "vllm_model.model.norm.weight",
                (None,),
            ),
            # LM Head (Logits Dense) - Shard Vocab Dim
            "params.params.decoder.logits_dense.kernel": (
                "vllm_model.lm_head.weight",
                ("model", "vocab"),
            ),

            # --- Layer-Specific Parameters (Scanned to Unscanned) ---

            # --- Attention (Needs Hooks to Combine Q/K/V) ---
            # The vLLM QKV weight/bias must be constructed by concatenating the MaxText Q, K, V kernels/biases.
            # MaxText: layers.*.GptOssAttention.query.kernel, .key.kernel, .value.kernel
            # vLLM: layers.*.attn.qkv_proj.weight
            # NOTE: The resulting combined QKV tensor in vLLM is often: [qkv_dim, hidden_dim]
            # MaxText weights have an extra "head" dimension, which implies a manual reshape/concatenate.
            
            # The mappings below mark the source components that are consumed by the hooks defined above.
            # MaxText Q/K/V kernels (consumed by combine_qkv_kernels hook)
            "params.params.decoder.layers.layers_*.GptOssAttention.query.kernel": None,
            "params.params.decoder.layers.layers_*.GptOssAttention.key.kernel": None,
            "params.params.decoder.layers.layers_*.GptOssAttention.value.kernel": None,
            
            # MaxText Q/K/V biases (consumed by combine_qkv_biases hook)
            "params.params.decoder.layers.layers_*.GptOssAttention.query.bias": None,
            "params.params.decoder.layers.layers_*.GptOssAttention.key.bias": None,
            "params.params.decoder.layers.layers_*.GptOssAttention.value.bias": None,
            
            # Output Projection (MaxText's out.kernel/bias -> vLLM's o_proj.weight/bias)
            "params.params.decoder.layers.layers_*.GptOssAttention.out.kernel": (
                "vllm_model.model.layers.*.attn.o_proj.weight",
                ("model", "heads", "head_dim", None), # Example sharding, confirm actual axis
            ),
            "params.params.decoder.layers.layers_*.GptOssAttention.out.bias": (
                "vllm_model.model.layers.*.attn.o_proj.bias",
                (None,),
            ),
            # Rotational Embedding Sinks (vLLM has this explicitly)
            "params.params.decoder.layers.layers_*.GptOssAttention.sinks": (
                "vllm_model.model.layers.*.attn.sinks",
                (None,),
            ),

            # --- MLP (MoE) (Needs Hooks to Combine w1/w3) ---
            # MaxText has gate, wi_0, wi_1 (w1, w3) - combined into vLLM's w13_weight/bias
            # MaxText w1/w3 kernels (consumed by combine_mlp_w13_kernels hook)
            "params.params.decoder.layers.layers_*.GptOssMlp.wi_0": None, # Should be combined to form w1
            "params.params.decoder.layers.layers_*.GptOssMlp.wi_1": None, # Should be combined to form w3
            # NOTE: MaxText might use GptOssMlp.gate.kernel for the *real* w1, and wi_0/wi_1 for up-projections, this is a common variance.
            # Assuming MaxText's `gate.kernel` (Gate) and `wi_0` (Up) are combined for vLLM's `w13_weight`. A hook is required here.
            "params.params.decoder.layers.layers_*.GptOssMlp.gate.kernel": None, # Source 1 for combined w13
            "params.params.decoder.layers.layers_*.GptOssMlp.wi_0": None, # Source 2 for combined w13
            
            # MaxText w1/w3 biases (consumed by combine_mlp_w13_biases hook)
            "params.params.decoder.layers.layers_*.GptOssMlp.gate.bias": None, # Source 1 for combined w13_bias
            "params.params.decoder.layers.layers_*.GptOssMlp.wi_0_bias": None, # Source 2 for combined w13_bias

            # MLP Down-Projection (wo.kernel/bias -> vLLM's w2_weight/bias)
            "params.params.decoder.layers.layers_*.GptOssMlp.wo": (
                "vllm_model.model.layers.*.mlp.experts.w2_weight",
                ("expert", "layer", None, "model"), # Example sharding (Expert, Layer, Out, In)
            ),
            "params.params.decoder.layers.layers_*.GptOssMlp.wo_bias": (
                "vllm_model.model.layers.*.mlp.experts.w2_bias",
                ("expert", "layer", None), # Example sharding (Expert, Layer, Out)
            ),
            
            # Router Weights
            # MaxText doesn't show a clear router key, assuming a separate router module in MaxText
            # VLLM uses `router.weight`
            # For now, map MaxText's `GptOssMlp.gate.kernel` (if it's the router) or define a new source if needed.
            # Assuming MaxText's `GptOssMlp.gate.kernel` is the router's input gate/weight.
            "params.params.decoder.layers.layers_*.GptOssMlp.gate.kernel": (
                "vllm_model.model.layers.*.mlp.router.weight",
                ("expert", "model"),
            ),
            "params.params.decoder.layers.layers_*.GptOssMlp.gate.bias": (
                "vllm_model.model.layers.*.mlp.router.bias",
                ("expert",),
            ),

            # --- Layer Norms ---
            # Pre-Attention Layer Norm (input_layernorm)
            "params.params.decoder.layers.layers_*.pre_self_attention_layer_norm.scale": (
                "vllm_model.model.layers.*.input_layernorm.weight",
                ("model", "layer"),
            ),
            # Post-Attention Layer Norm (post_attention_layernorm)
            "params.params.decoder.layers.layers_*.post_self_attention_layer_norm.scale": (
                "vllm_model.model.layers.*.post_attention_layernorm.weight",
                ("model", "layer"),
            ),
        }