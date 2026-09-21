# Copyright 2023–2026 Google LLC
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

"""Self-contained Cosmos 3 model family package.

Layout:
  * ``cosmos3.py`` — Config, sequence packing, masks, 3D M-RoPE,
    ``CosmosDualAttention``, ``Cosmos3MLP``, and ``Cosmos3MoTDecoderLayer``.
"""

from maxtext.m3.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Config,
    Cosmos3MLP,
    Cosmos3MoTDecoderLayer,
    CosmosAttention,
    CosmosDualAttention,
    CosmosPackingMetadata,
    apply_rotary_pos_emb,
    build_causal_understanding_mask,
    build_causal_understanding_splash_mask,
    build_cosmos_packing_metadata,
    build_full_generative_mask,
    build_full_generative_splash_mask,
    causal_understanding_attention,
    compile_cosmos_splash_mask,
    compute_3d_mrope_cos_sin,
    full_generative_attention,
    reinterleave_streams,
    rotate_half,
    unpack_streams,
)

__all__ = [
    "Cosmos3Config",
    "Cosmos3MLP",
    "Cosmos3MoTDecoderLayer",
    "CosmosAttention",
    "CosmosDualAttention",
    "CosmosPackingMetadata",
    "apply_rotary_pos_emb",
    "build_causal_understanding_mask",
    "build_causal_understanding_splash_mask",
    "build_cosmos_packing_metadata",
    "build_full_generative_mask",
    "build_full_generative_splash_mask",
    "causal_understanding_attention",
    "compile_cosmos_splash_mask",
    "compute_3d_mrope_cos_sin",
    "full_generative_attention",
    "reinterleave_streams",
    "rotate_half",
    "unpack_streams",
]
