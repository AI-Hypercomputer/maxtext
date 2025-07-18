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


def GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING(config):
  """Returns mapping between HuggingFace weights path and weights shape.

  Args:
      config (dict): Model configuration dictionary, defined in `model_configs.py`

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a List
  """

  mapping = {
      "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
      "model.norm.weight": [config["hidden_size"]],
  }
  for layer_idx in range(config["num_hidden_layers"]):
    layer_mapping = {
        f"model.layers.{layer_idx}.input_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.down_proj.weight": [
            config["hidden_size"],
            config["intermediate_size"],
        ],
        f"model.layers.{layer_idx}.mlp.up_proj.weight": [
            config["intermediate_size"],
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": [
            config["intermediate_size"],
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.post_feedforward_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": [
            config["num_key_value_heads"] * config["head_dim"],
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": [
            config["hidden_size"],
            config["num_attention_heads"] * config["head_dim"],
        ],
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": [
            config["num_attention_heads"] * config["head_dim"],
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": [
            config["num_key_value_heads"] * config["head_dim"],
            config["hidden_size"],
        ],
    }
    mapping = {**mapping, **layer_mapping}
  return mapping


def GEMMA3TEXT_HF_WEIGHTS_TO_SHAPE_MAPPING(config):
  """Returns mapping between HuggingFace Gemma3Text weights path and weights shape.

  Args:
      config (dict): Model configuration dictionary (from HF Gemma3TextConfig.to_dict())

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a List
  """
  head_dim = config["head_dim"]

  mapping = {
      "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
      "model.norm.weight": [config["hidden_size"]],
  }
  for layer_idx in range(config["num_hidden_layers"]):
    layer_mapping = {
        f"model.layers.{layer_idx}.input_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": [
            config["num_attention_heads"] * head_dim,
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": [
            config["num_key_value_heads"] * head_dim,
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": [
            config["num_key_value_heads"] * head_dim,
            config["hidden_size"],
        ],
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": [
            config["hidden_size"],
            config["num_attention_heads"] * head_dim,
        ],
        f"model.layers.{layer_idx}.self_attn.q_norm.weight": [head_dim],
        f"model.layers.{layer_idx}.self_attn.k_norm.weight": [head_dim],
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": [config["intermediate_size"], config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.up_proj.weight": [config["intermediate_size"], config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.down_proj.weight": [config["hidden_size"], config["intermediate_size"]],
        f"model.layers.{layer_idx}.post_feedforward_layernorm.weight": [config["hidden_size"]],
    }
    mapping.update(layer_mapping)
  return mapping


def QWEN3_HF_WEIGHTS_TO_SHAPE_MAPPING(config):
  """Returns mapping between HuggingFace Qwen3 weights path and the HuggingFace weights shape.

  To check this mapping, dump the huggingface model shapes:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",
    )
    for name, val in model.named_parameters():
      print(name, val.shape)

  Args:
      config (dict): Model configuration dictionary (from HF Qwen3TextConfig.to_dict())
                     Expected keys: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a List
  """
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config["num_key_value_heads"]
  head_dim = config.get(
      "head_dim", config["hidden_size"] // config["num_attention_heads"]
  )  # head_dim might not always be present

  mapping = {
      "model.embed_tokens.weight": [config["vocab_size"], hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [config["vocab_size"], hidden_size],
  }

  # Determine if the model is MoE based on config keys
  num_experts = config.get("num_experts", 0)

  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"
    layer_mapping = {
        f"{layer_prefix}.input_layernorm.weight": [hidden_size],
        f"{layer_prefix}.post_attention_layernorm.weight": [hidden_size],
        # Attention projections
        f"{layer_prefix}.self_attn.q_proj.weight": [num_attention_heads * head_dim, hidden_size],
        f"{layer_prefix}.self_attn.k_proj.weight": [num_key_value_heads * head_dim, hidden_size],
        f"{layer_prefix}.self_attn.v_proj.weight": [num_key_value_heads * head_dim, hidden_size],
        f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, num_attention_heads * head_dim],
        # QK Norm weights (applied per head to the head_dim dimension)
        f"{layer_prefix}.self_attn.q_norm.weight": [head_dim],
        f"{layer_prefix}.self_attn.k_norm.weight": [head_dim],
    }

    if num_experts > 1:
      # MoE MLP layers
      moe_ffn_intermediate_size = config.get("moe_intermediate_size")
      if moe_ffn_intermediate_size is None:
        # moe_intermediate_size refers to the intermediate size of the routed expert
        # For Qwen MoE, moe_intermediate_size is distinct from intermediate_size (for dense layers)
        # Fall back to intermediate_size
        moe_ffn_intermediate_size = config.get("intermediate_size")
        if moe_ffn_intermediate_size is None:
          raise ValueError(
              "MoE model detected (num_experts > 1) but 'moe_intermediate_size' or 'intermediate_size' not found in config."
          )

      layer_mapping.update(
          {
              f"{layer_prefix}.mlp.gate.weight": [num_experts, hidden_size],
          }
      )
      for expert_j in range(num_experts):
        expert_prefix = f"{layer_prefix}.mlp.experts.{expert_j}"
        layer_mapping.update(
            {
                f"{expert_prefix}.gate_proj.weight": [moe_ffn_intermediate_size, hidden_size],
                f"{expert_prefix}.up_proj.weight": [moe_ffn_intermediate_size, hidden_size],
                f"{expert_prefix}.down_proj.weight": [hidden_size, moe_ffn_intermediate_size],
            }
        )
    else:
      # Dense MLP layers
      dense_ffn_intermediate_size = config.get("intermediate_size")
      if dense_ffn_intermediate_size is None:
        raise ValueError("'intermediate_size' not found in config for a dense MLP.")
      layer_mapping.update(
          {
              f"{layer_prefix}.mlp.gate_proj.weight": [dense_ffn_intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.up_proj.weight": [dense_ffn_intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.down_proj.weight": [hidden_size, dense_ffn_intermediate_size],
          }
      )
    mapping.update(layer_mapping)
  return mapping


SHAPE_MAPPING = {
    "gemma2-2b": GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "gemma2-9b": GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "gemma2-27b": GEMMA2_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "gemma3-4b": GEMMA3TEXT_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "gemma3-12b": GEMMA3TEXT_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "gemma3-27b": GEMMA3TEXT_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "qwen3-0.6b": QWEN3_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "qwen3-4b": QWEN3_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "qwen3-8b": QWEN3_HF_WEIGHTS_TO_SHAPE_MAPPING,
    "qwen3-14b": QWEN3_HF_WEIGHTS_TO_SHAPE_MAPPING,
}
