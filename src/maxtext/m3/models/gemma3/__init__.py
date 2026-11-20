# Copyright 2026 Google LLC
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

"""Gemma 3 model family in MaxText m3 (pure Flax NNX)."""

from maxtext.m3.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3Config,
    Gemma3CycleBlock,
    Gemma3Decoder,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3Sharding,
    Gemma3TextScaledWordEmbedding,
    apply_gemma3_rope,
    build_gemma3_attention_mask,
    calculate_gemma3_tflops_per_device,
    compute_gemma3_inv_freq,
    compute_gemma3_loss,
    configure_gemma3_engine,
    create_gemma3_model,
    gemma3_engine_loss_fn,
)

__all__ = [
    "Gemma3Attention",
    "Gemma3Config",
    "Gemma3CycleBlock",
    "Gemma3Decoder",
    "Gemma3DecoderLayer",
    "Gemma3ForCausalLM",
    "Gemma3MLP",
    "Gemma3RMSNorm",
    "Gemma3Sharding",
    "Gemma3TextScaledWordEmbedding",
    "apply_gemma3_rope",
    "build_gemma3_attention_mask",
    "calculate_gemma3_tflops_per_device",
    "compute_gemma3_inv_freq",
    "compute_gemma3_loss",
    "configure_gemma3_engine",
    "create_gemma3_model",
    "gemma3_engine_loss_fn",
]
