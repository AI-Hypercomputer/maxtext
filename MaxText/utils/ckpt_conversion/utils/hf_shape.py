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

"""Hugging Face shape checkpoint conversion utils."""

def GEMMA3_HF_WEIGHTS_TO_SHAPE(config):
  """Generates a shape mapping for Hugging Face Gemma3 parameters.

  This function computes the expected shapes for all parameters in a Hugging
  Face Gemma3 model, including both the text and vision components. The shapes
  are derived from the provided model configuration.

  Args:
    config (dict): The Hugging Face model configuration dictionary. It must
      contain 'text_config' and 'vision_config' sub-dictionaries with all
      necessary architectural details (e.g., hidden_size, num_layers).

  Returns:
    dict: A dictionary where keys are Hugging Face parameter names (e.g.,
    'model.language_model.embed_tokens.weight') and values are lists of
    integers representing the tensor's shape.
  """
  shapes = {}

  # Config-derived dimensions
  text_config = config["text_config"]
  vision_config = config["vision_config"]

  lm_hidden_size = text_config["hidden_size"]
  lm_intermediate_size = text_config["intermediate_size"]
  lm_num_layers = text_config["num_hidden_layers"]
  lm_q_heads = text_config["num_attention_heads"]
  lm_kv_heads = text_config["num_key_value_heads"]
  lm_head_dim = text_config["head_dim"]
  lm_q_dim = lm_q_heads * lm_head_dim
  lm_kv_dim = lm_kv_heads * lm_head_dim

  vision_hidden_size = vision_config["hidden_size"]
  vision_intermediate_size = vision_config["intermediate_size"]
  vision_num_layers = vision_config["num_hidden_layers"]
  vision_patch_size = vision_config["patch_size"]
  vision_num_channels = vision_config["num_channels"]
  vision_image_size = vision_config["image_size"]
  vision_num_positions = (vision_image_size / vision_patch_size) ** 2

  vocab_size = text_config["vocab_size"]

  # Vision Tower embeddings
  shapes["model.vision_tower.vision_model.embeddings.patch_embedding.weight"] = [
      vision_hidden_size,
      vision_num_channels,
      vision_patch_size,
      vision_patch_size,
  ]
  shapes["model.vision_tower.vision_model.embeddings.patch_embedding.bias"] = [vision_hidden_size]
  shapes["model.vision_tower.vision_model.embeddings.position_embedding.weight"] = [
      vision_num_positions,
      vision_hidden_size,
  ]

  # Vision Encoder layers
  for i in range(vision_num_layers):
    # LayerNorm 1
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = [vision_hidden_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = [vision_hidden_size]
    # Attention
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = [
        vision_hidden_size,
        vision_hidden_size,
    ]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = [vision_hidden_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = [
        vision_hidden_size,
        vision_hidden_size,
    ]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = [vision_hidden_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = [
        vision_hidden_size,
        vision_hidden_size,
    ]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = [vision_hidden_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = [
        vision_hidden_size,
        vision_hidden_size,
    ]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = [vision_hidden_size]
    # MLP
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = [vision_hidden_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = [vision_hidden_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = [
        vision_intermediate_size,
        vision_hidden_size,
    ]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = [vision_intermediate_size]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = [
        vision_hidden_size,
        vision_intermediate_size,
    ]
    shapes[f"model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = [vision_hidden_size]

  # Vision post-norm
  shapes["model.vision_tower.vision_model.post_layernorm.weight"] = [vision_hidden_size]
  shapes["model.vision_tower.vision_model.post_layernorm.bias"] = [vision_hidden_size]

  # Multi-Modal Projector
  shapes["model.multi_modal_projector.mm_input_projection_weight"] = [vision_hidden_size, lm_hidden_size]
  shapes["model.multi_modal_projector.mm_soft_emb_norm.weight"] = [vision_hidden_size]

  # Language Model embeddings
  shapes["model.language_model.embed_tokens.weight"] = [vocab_size, lm_hidden_size]

  # Language Model layers
  for i in range(lm_num_layers):
    # Self-Attn
    shapes[f"model.language_model.layers.{i}.self_attn.q_proj.weight"] = [lm_q_dim, lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.self_attn.q_proj.bias"] = [lm_q_dim]
    shapes[f"model.language_model.layers.{i}.self_attn.k_proj.weight"] = [lm_kv_dim, lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.self_attn.k_proj.bias"] = [lm_kv_dim]
    shapes[f"model.language_model.layers.{i}.self_attn.v_proj.weight"] = [lm_kv_dim, lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.self_attn.v_proj.bias"] = [lm_kv_dim]
    shapes[f"model.language_model.layers.{i}.self_attn.o_proj.weight"] = [lm_hidden_size, lm_q_dim]
    shapes[f"model.language_model.layers.{i}.self_attn.o_proj.bias"] = [lm_hidden_size]
    # Norms
    shapes[f"model.language_model.layers.{i}.self_attn.q_norm.weight"] = [lm_head_dim]
    shapes[f"model.language_model.layers.{i}.self_attn.k_norm.weight"] = [lm_head_dim]
    shapes[f"model.language_model.layers.{i}.input_layernorm.weight"] = [lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.post_attention_layernorm.weight"] = [lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.pre_feedforward_layernorm.weight"] = [lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.post_feedforward_layernorm.weight"] = [lm_hidden_size]
    # MLP
    shapes[f"model.language_model.layers.{i}.mlp.gate_proj.weight"] = [lm_intermediate_size, lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.mlp.gate_proj.bias"] = [lm_intermediate_size]
    shapes[f"model.language_model.layers.{i}.mlp.up_proj.weight"] = [lm_intermediate_size, lm_hidden_size]
    shapes[f"model.language_model.layers.{i}.mlp.up_proj.bias"] = [lm_intermediate_size]
    shapes[f"model.language_model.layers.{i}.mlp.down_proj.weight"] = [lm_hidden_size, lm_intermediate_size]
    shapes[f"model.language_model.layers.{i}.mlp.down_proj.bias"] = [lm_hidden_size]

  # Final norm & LM head
  shapes["model.language_model.norm.weight"] = [lm_hidden_size]
  shapes["lm_head.weight"] = [vocab_size, lm_hidden_size]
  return shapes


def GEMMA2_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and weights shape.

  Args:
      config (dict): Model configuration dictionary, defined in `model_configs.py`

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a list
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


def QWEN3_HF_WEIGHTS_TO_SHAPE(config):
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
          - Values are parameter shape as a list
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


def LLAMA31_HF_WEIGHTS_TO_SHAPE(config):
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
      "lm_head.weight": [config["vocab_size"], config["hidden_size"]],
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


HF_SHAPE = {
    "gemma2-2b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-9b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-27b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma3-4b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-12b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-27b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "qwen3-0.6b": QWEN3_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b": QWEN3_HF_WEIGHTS_TO_SHAPE,
    "qwen3-8b": QWEN3_HF_WEIGHTS_TO_SHAPE,
    "qwen3-14b": QWEN3_HF_WEIGHTS_TO_SHAPE,
    "qwen3-32b": QWEN3_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-70b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-405b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b": QWEN3_HF_WEIGHTS_TO_SHAPE,
    "qwen3-480b-a35b": QWEN3_HF_WEIGHTS_TO_SHAPE,
}
