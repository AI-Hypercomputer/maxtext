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

"""Per-device training TFLOPs calculation for dense Qwen3 and Qwen3-VL m3 models.

Registered in `maxtext.m3.models.MODEL_FLOPS_REGISTRY` so
`maxtext_utils.calculate_tflops_training_per_device` dispatches here when
`use_m3_model=True` without introducing an import cycle back into
`maxtext.utils.maxtext_utils`.
"""

from typing import Any


def calculate_qwen3_tflops_per_device(config: Any, log: bool = True) -> tuple[float, float, float]:
  """Calculates `(total_tflops, learnable_weight_tflops, attention_tflops)` per device."""
  batch = config.per_device_batch_size
  seq_len = config.max_target_length
  emb_dim = config.emb_dim
  mlp_dim = config.mlp_dim
  num_layers = config.num_decoder_layers
  num_q_heads = config.num_query_heads
  num_kv_heads = config.num_kv_heads
  head_dim = config.head_dim

  ffn_flops_per_layer = (
      2 * batch * seq_len * mlp_dim * emb_dim * len(config.mlp_activations) + 2 * batch * seq_len * mlp_dim * emb_dim
  )
  total_ffn_flops_all_layers = ffn_flops_per_layer * num_layers

  kv_multiplier = 1 if getattr(config, "share_kv_projections", False) else 2
  qkv_flops = 2 * batch * seq_len * emb_dim * (num_q_heads + kv_multiplier * num_kv_heads) * head_dim
  projection_flops = 2 * batch * seq_len * emb_dim * num_q_heads * head_dim
  causal_attention_flops = 2 * batch * (seq_len**2) * num_q_heads * head_dim
  embedding_flops = 2 * batch * seq_len * emb_dim * config.vocab_size

  grad_accum = getattr(config, "gradient_accumulation_steps", 1)
  learnable_weight_tflops = (
      (total_ffn_flops_all_layers + (qkv_flops + projection_flops) * num_layers + embedding_flops)
      * 3
      / 10**12
      * grad_accum
  )
  attention_tflops = causal_attention_flops * num_layers * 3 / 10**12 * grad_accum

  if getattr(config, "use_dpo", False):
    reference_model_tflops = learnable_weight_tflops / 3
    attention_tflops = attention_tflops + (attention_tflops / 3)
  else:
    reference_model_tflops = 0.0

  total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops
  if log:
    print(
        "Per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops / total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops / total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops
