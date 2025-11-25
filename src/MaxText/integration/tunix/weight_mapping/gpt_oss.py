# Copyright 2023–2025 Google LLC
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

"""Defines the weight mapping from MaxText's GPT-OSS model to a vLLM-compatible format.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class GptOssMaxTextMapping:
    """
    Mapping definition from MaxText GPT-OSS (Scanned/Interleaved) to vLLM JAX NNX.

    Supports:
    - Modulo Interleaving (e.g., Block 0 -> Layers 0, 2, 4...)
    - MoE Quantized Weights (MXFP4 nested qvalue/scale)
    """

    @staticmethod
    def to_hf_hook_fns():
        return {}

    @staticmethod
    def to_hf_transpose_keys():
        # If your specific checkpoint requires transposing kernels, add them here.
        # e.g. "layers.*.attn.kernel_q_DNH": (0, 2, 1)
        return {}

    @staticmethod
    def to_hf_mapping(
        layer_cycle_interval: int = 2,
        total_num_layers: int = 96,
        interleave_style: str = "modulo"
    ) -> Dict[str, Tuple[str, Tuple[Optional[str], ...]]]:

        mapping = {}

        # --- 1. Global Parameters ---
        mapping.update({
            "base.token_embedder.embedding": ("embedder.input_embedding_table_VD", (('data', 'model'), None)),
            "base.decoder.decoder_norm.scale": ("final_norm.scale", (None,)),
            "base.decoder.logits_dense.kernel": ("lm_head.input_embedding_table_DV", (None, ('data', 'model'))),
        })

        # --- 2. Layer Mapping Loop ---
        layers_per_block = total_num_layers // layer_cycle_interval

        for block_idx in range(layer_cycle_interval):
            src_block = f"base.decoder.layers.layers_{block_idx}"

            # Calculate Target Indices
            if interleave_style == "modulo":
                target_indices = range(block_idx, total_num_layers, layer_cycle_interval)
            else:
                start = block_idx * layers_per_block
                target_indices = range(start, start + layers_per_block)

            # Create Regex: "layers\.(0|2|4|...)\."
            regex_indices = "|".join(map(str, target_indices))
            layer_regex = f"layers\.({regex_indices})"

            # --- 3. Block Mappings ---
            
            # Layer Norms & Attention (Standard)
            mapping.update({
                f"{src_block}.pre_self_attention_layer_norm.scale": (f"{layer_regex}.pre_attention_norm.scale", (None, "layer")),
                f"{src_block}.post_self_attention_layer_norm.scale": (f"{layer_regex}.pre_mlp_norm.scale", (None, "layer")),
                
                f"{src_block}.GptOssAttention.query.kernel": (f"{layer_regex}.attn.kernel_q_DNH", (None, "layer", "model", None)),
                f"{src_block}.GptOssAttention.key.kernel":   (f"{layer_regex}.attn.kernel_k_DKH", (None, "layer", "model", None)),
                f"{src_block}.GptOssAttention.value.kernel": (f"{layer_regex}.attn.kernel_v_DKH", (None, "layer", "model", None)),
                f"{src_block}.GptOssAttention.out.kernel":   (f"{layer_regex}.attn.kernel_o_proj_NHD", ("model", "layer", None, None)),
                
                f"{src_block}.GptOssAttention.query.bias": (f"{layer_regex}.attn.bias_q_NH", (None, "layer", None)),
                f"{src_block}.GptOssAttention.key.bias":   (f"{layer_regex}.attn.bias_k_KH", (None, "layer", None)),
                f"{src_block}.GptOssAttention.value.bias": (f"{layer_regex}.attn.bias_v_KH", (None, "layer", None)),
                f"{src_block}.GptOssAttention.out.bias":   (f"{layer_regex}.attn.bias_o_D", (None, "layer")),
                f"{src_block}.GptOssAttention.sinks":      (f"{layer_regex}.attn.sinks_N", (None, "layer")),
            })

            # MoE Router
            mapping.update({
                f"{src_block}.GptOssMlp.gate.kernel": (f"{layer_regex}.custom_module.router.kernel_DE", (None, "layer", "model")),
                f"{src_block}.GptOssMlp.gate.bias":   (f"{layer_regex}.custom_module.router.bias_E", (None, "layer")),
            })

            # --- MOE EXPERTS (FIXED for Split Weights) ---
            # Source uses 'wi_0', 'wi_1', 'wo'. Target uses 'mlp1', 'mlp2'.
            # We map BOTH wi_0 and wi_1 to the same target container.

            # MLP1 BIAS
            mapping.update({
                f"{src_block}.GptOssMlp.wi_0_bias": (f"{layer_regex}.custom_module.mlp1_bias_EF2", ("expert", "layer", None)),
                f"{src_block}.GptOssMlp.wi_1_bias": (f"{layer_regex}.custom_module.mlp1_bias_EF2", ("expert", "layer", None)),
            })

            # MLP1 WEIGHTS (Split -> Fused QValue)
            # Note: Removed '.kernel' from source based on your error logs
            mapping.update({
                f"{src_block}.GptOssMlp.wi_0": (
                    f"{layer_regex}.custom_module.mlp1_weight_EDF2.array.qvalue", 
                    (None, "layer", "expert", "model", None)
                ),
                f"{src_block}.GptOssMlp.wi_1": (
                    f"{layer_regex}.custom_module.mlp1_weight_EDF2.array.qvalue", 
                    (None, "layer", "expert", "model", None)
                ),
            })

            # MLP2 (Down Projection)
            mapping.update({
                f"{src_block}.GptOssMlp.wo_bias": (f"{layer_regex}.custom_module.mlp2_bias_ED", ("expert", "layer", None)),
                
                f"{src_block}.GptOssMlp.wo": (
                    f"{layer_regex}.custom_module.mlp2_weight_EFD.array.qvalue", 
                    (None, "layer", "expert", "model", None)
                ),
            })

        return mapping

        