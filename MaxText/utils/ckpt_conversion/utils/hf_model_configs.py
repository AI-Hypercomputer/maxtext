"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
This config defines the architectural configurations of the Hugging Face version of a model.
"""


import transformers


gemma3text_4b_config = transformers.Gemma3TextConfig(
    vocab_size=262144,
    hidden_size=2560,
    intermediate_size=10240,
    num_hidden_layers=34,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    hidden_activation="gelu",
    max_position_embeddings=163840,
    rms_norm_eps=1e-06,
    rope_theta=10000,
    sliding_window=1024,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)

gemma3text_12b_config = transformers.Gemma3TextConfig(
    vocab_size=262144,
    hidden_size=3840,
    intermediate_size=15360,
    num_attention_heads=16,
    num_hidden_layers=48,
    num_key_value_heads=8,
    head_dim=256,
    hidden_activation="gelu",
    max_position_embeddings=163840,
    rms_norm_eps=1e-06,
    rope_theta=10000,
    sliding_window=1024,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)

gemma3text_27b_config = transformers.Gemma3TextConfig(
    vocab_size=262144,
    hidden_size=5376,
    intermediate_size=21504,
    num_attention_heads=32,
    num_hidden_layers=62,
    num_key_value_heads=16,
    head_dim=128,
    hidden_activation="gelu",
    max_position_embeddings=163840,
    rms_norm_eps=1e-06,
    rope_theta=10000,
    sliding_window=1024,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
    query_pre_attn_scalar=168,
)


gemma2_2b_config = transformers.Gemma2Config(
    num_hidden_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    hidden_size=2304,
    intermediate_size=9216,
)

gemma2_9b_config = transformers.Gemma2Config(
    num_hidden_layers=42,
    num_attention_heads=16,
    num_key_value_heads=8,
    hidden_size=3584,
    intermediate_size=14336,
    final_logit_softcapping=30.0,
    attn_logit_softcapping=50.0,
    head_dim=256,
    sliding_window=4096,
    query_pre_attn_scalar=224,
)

gemma2_27b_config = transformers.Gemma2Config(
    num_hidden_layers=46,
    num_attention_heads=32,
    num_key_value_heads=16,
    hidden_size=4608,
    intermediate_size=36864,
    final_logit_softcapping=30.0,
    attn_logit_softcapping=50.0,
    head_dim=128,
    sliding_window=4096,
    query_pre_attn_scalar=144,
)

qwen3_0_6b_config = transformers.Qwen3Config(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    rms_norm_eps=1.0e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)

qwen3_4b_config = transformers.Qwen3Config(
    vocab_size=151936,
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    rms_norm_eps=1.0e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)

qwen3_8b_config = transformers.Qwen3Config(
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    rms_norm_eps=1.0e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)

qwen3_14b_config = transformers.Qwen3Config(
    vocab_size=151936,
    hidden_size=5120,
    intermediate_size=17408,
    num_hidden_layers=40,
    num_attention_heads=40,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    rms_norm_eps=1.0e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)

HF_MODEL_CONFIGS = {
    "gemma2-2b": gemma2_2b_config,
    "gemma2-9b": gemma2_9b_config,
    "gemma2-27b": gemma2_27b_config,
    "gemma3-4b": gemma3text_4b_config,
    "gemma3-12b": gemma3text_12b_config,
    "gemma3-27b": gemma3text_27b_config,
    "qwen3-0.6b": qwen3_0_6b_config,
    "qwen3-4b": qwen3_4b_config,
    "qwen3-8b": qwen3_8b_config,
    "qwen3-14b": qwen3_14b_config,
}
