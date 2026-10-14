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

init_hyper_connection_streams = hyper_connection.init_hyper_connection_streams
l2norm = hyper_connection.l2norm
Qwen38GatedDeltaNet = gdn.Qwen38GatedDeltaNet
Qwen38GatedResidual = hyper_connection.Qwen38GatedResidual
Qwen38RMSNorm = hyper_connection.Qwen38RMSNorm
Qwen38RMSNormGated = hyper_connection.Qwen38RMSNormGated
Qwen38ZeroCenteredRMSNorm = hyper_connection.Qwen38ZeroCenteredRMSNorm
jax_chunk_gated_delta_rule = gdn_recurrence.jax_chunk_gated_delta_rule
jax_recurrent_gated_delta_rule = gdn_recurrence.jax_recurrent_gated_delta_rule

__all__ = [
    "Qwen38GatedDeltaNet",
    "Qwen38GatedResidual",
    "Qwen38RMSNorm",
    "Qwen38RMSNormGated",
    "Qwen38ZeroCenteredRMSNorm",
    "init_hyper_connection_streams",
    "jax_chunk_gated_delta_rule",
    "jax_recurrent_gated_delta_rule",
    "l2norm",
]
