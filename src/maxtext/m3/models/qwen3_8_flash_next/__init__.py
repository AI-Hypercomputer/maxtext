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
from maxtext.m3.models.qwen3_8_flash_next import gdn
from maxtext.m3.models.qwen3_8_flash_next import gdn_recurrence
from maxtext.m3.models.qwen3_8_flash_next import hyper_connection
from maxtext.m3.models.qwen3_8_flash_next import loss
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
from maxtext.m3.models.qwen3_8_flash_next import moe
from maxtext.m3.models.qwen3_8_flash_next import mtp
from maxtext.m3.models.qwen3_8_flash_next import ple
from maxtext.m3.models.qwen3_8_flash_next import qsa_attention

create_qwen3_8_flash_next_model = modeling_qwen3_8_flash_next.create_qwen3_8_flash_next_model
init_hyper_connection_streams = hyper_connection.init_hyper_connection_streams
l2norm = hyper_connection.l2norm
Qwen3_8Decoder = modeling_qwen3_8_flash_next.Qwen3_8Decoder
Qwen3_8Experts = moe.Qwen3_8Experts
Qwen3_8FlashNextModel = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel
Qwen3_8GatedDeltaNet = gdn.Qwen3_8GatedDeltaNet
Qwen3_8GatedResidual = hyper_connection.Qwen3_8GatedResidual
Qwen3_8MLP = moe.Qwen3_8MLP
Qwen3_8MTPBlock = mtp.Qwen3_8MTPBlock
Qwen3_8NGramEmbedding = ple.Qwen3_8NGramEmbedding
Qwen3_8PLELayer = ple.Qwen3_8PLELayer
Qwen3_8QSAIndexer = qsa_attention.Qwen3_8QSAIndexer
Qwen3_8RMSNorm = hyper_connection.Qwen3_8RMSNorm
Qwen3_8RMSNormGated = hyper_connection.Qwen3_8RMSNormGated
Qwen3_8RotaryEmbedding = qsa_attention.Qwen3_8RotaryEmbedding
Qwen3_8SparseAttention = qsa_attention.Qwen3_8SparseAttention
Qwen3_8SparseMoeBlock = moe.Qwen3_8SparseMoeBlock
Qwen3_8TopKRouter = moe.Qwen3_8TopKRouter
Qwen3_8ZeroCenteredRMSNorm = hyper_connection.Qwen3_8ZeroCenteredRMSNorm
apply_rotary_pos_emb = qsa_attention.apply_rotary_pos_emb
compute_ple_buffers = ple.compute_ple_buffers
compute_total_loss = loss.compute_total_loss
cross_entropy_loss = loss.cross_entropy_loss
jax_chunk_gated_delta_rule = gdn_recurrence.jax_chunk_gated_delta_rule
jax_recurrent_gated_delta_rule = gdn_recurrence.jax_recurrent_gated_delta_rule
rotate_half = qsa_attention.rotate_half
router_load_balancing_loss = loss.router_load_balancing_loss
splitmix64 = ple.splitmix64

__all__ = [
    "Qwen3_8Decoder",
    "Qwen3_8Experts",
    "Qwen3_8FlashNextModel",
    "Qwen3_8GatedDeltaNet",
    "Qwen3_8GatedResidual",
    "Qwen3_8MLP",
    "Qwen3_8MTPBlock",
    "Qwen3_8NGramEmbedding",
    "Qwen3_8PLELayer",
    "Qwen3_8QSAIndexer",
    "Qwen3_8RMSNorm",
    "Qwen3_8RMSNormGated",
    "Qwen3_8RotaryEmbedding",
    "Qwen3_8SparseAttention",
    "Qwen3_8SparseMoeBlock",
    "Qwen3_8TopKRouter",
    "Qwen3_8ZeroCenteredRMSNorm",
    "apply_rotary_pos_emb",
    "compute_ple_buffers",
    "compute_total_loss",
    "create_qwen3_8_flash_next_model",
    "cross_entropy_loss",
    "init_hyper_connection_streams",
    "jax_chunk_gated_delta_rule",
    "jax_recurrent_gated_delta_rule",
    "l2norm",
    "rotate_half",
    "router_load_balancing_loss",
    "splitmix64",
]
