# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.8-Flash-Next model family package."""

# pylint: disable=no-name-in-module
from maxtext.m3.models.qwen38_flash_next import gdn
from maxtext.m3.models.qwen38_flash_next import gdn_recurrence
from maxtext.m3.models.qwen38_flash_next import hyper_connection
from maxtext.m3.models.qwen38_flash_next import moe
from maxtext.m3.models.qwen38_flash_next import qsa_attention

init_hyper_connection_streams = hyper_connection.init_hyper_connection_streams
l2norm = hyper_connection.l2norm
Qwen38Experts = moe.Qwen38Experts
Qwen38GatedDeltaNet = gdn.Qwen38GatedDeltaNet
Qwen38GatedResidual = hyper_connection.Qwen38GatedResidual
Qwen38MLP = moe.Qwen38MLP
Qwen38QSAIndexer = qsa_attention.Qwen38QSAIndexer
Qwen38RMSNorm = hyper_connection.Qwen38RMSNorm
Qwen38RMSNormGated = hyper_connection.Qwen38RMSNormGated
Qwen38RotaryEmbedding = qsa_attention.Qwen38RotaryEmbedding
Qwen38SparseAttention = qsa_attention.Qwen38SparseAttention
Qwen38SparseMoeBlock = moe.Qwen38SparseMoeBlock
Qwen38TopKRouter = moe.Qwen38TopKRouter
Qwen38ZeroCenteredRMSNorm = hyper_connection.Qwen38ZeroCenteredRMSNorm
apply_rotary_pos_emb = qsa_attention.apply_rotary_pos_emb
jax_chunk_gated_delta_rule = gdn_recurrence.jax_chunk_gated_delta_rule
jax_recurrent_gated_delta_rule = gdn_recurrence.jax_recurrent_gated_delta_rule
rotate_half = qsa_attention.rotate_half

__all__ = [
    "Qwen38Experts",
    "Qwen38GatedDeltaNet",
    "Qwen38GatedResidual",
    "Qwen38MLP",
    "Qwen38QSAIndexer",
    "Qwen38RMSNorm",
    "Qwen38RMSNormGated",
    "Qwen38RotaryEmbedding",
    "Qwen38SparseAttention",
    "Qwen38SparseMoeBlock",
    "Qwen38TopKRouter",
    "Qwen38ZeroCenteredRMSNorm",
    "apply_rotary_pos_emb",
    "init_hyper_connection_streams",
    "jax_chunk_gated_delta_rule",
    "jax_recurrent_gated_delta_rule",
    "l2norm",
    "rotate_half",
]
