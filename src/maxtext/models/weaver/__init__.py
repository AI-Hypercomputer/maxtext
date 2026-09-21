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

"""Self-contained Weaver model family package.

Layout:
  * ``weaver.py`` — Config, sequence packing, masks, 3D M-RoPE,
    ``WeaverDualAttention``, ``WeaverMLP``, and ``WeaverMoTDecoderLayer``.
"""

from maxtext.models.weaver.weaver import (
    WeaverAttention,
    WeaverConfig,
    WeaverDualAttention,
    WeaverMLP,
    WeaverMoTDecoderLayer,
    WeaverPackingMetadata,
    apply_rotary_pos_emb,
    build_causal_understanding_mask,
    build_causal_understanding_splash_mask,
    build_full_generative_mask,
    build_full_generative_splash_mask,
    build_weaver_packing_metadata,
    causal_understanding_attention,
    compile_weaver_splash_mask,
    compute_3d_mrope_cos_sin,
    full_generative_attention,
    reinterleave_streams,
    rotate_half,
    unpack_streams,
)

__all__ = [
    "WeaverAttention",
    "WeaverConfig",
    "WeaverDualAttention",
    "WeaverMLP",
    "WeaverMoTDecoderLayer",
    "WeaverPackingMetadata",
    "apply_rotary_pos_emb",
    "build_causal_understanding_mask",
    "build_causal_understanding_splash_mask",
    "build_full_generative_mask",
    "build_full_generative_splash_mask",
    "build_weaver_packing_metadata",
    "causal_understanding_attention",
    "compile_weaver_splash_mask",
    "compute_3d_mrope_cos_sin",
    "full_generative_attention",
    "reinterleave_streams",
    "rotate_half",
    "unpack_streams",
]
