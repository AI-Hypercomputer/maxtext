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


def GEMMA4_HF_WEIGHTS_TO_SHAPE(config):
  """Generates shape mapping for Hugging Face Gemma4 parameters.

  Handles both multimodal (with vision tower) and text-only variants, as well
  as MoE (26B) and dense (31B) text configurations. Shapes are per-layer aware:
  local (sliding) attention layers use head_dim, while global (full) attention
  layers use global_head_dim and num_global_key_value_heads.

  Args:
    config (dict): The Hugging Face model configuration dictionary. Must contain
      'text_config' with architectural details. May contain 'vision_config' for
      multimodal models.

  Returns:
    dict: A dictionary mapping Hugging Face parameter names to their shapes.
  """
  shapes = {}

  text_cfg = config.get("text_config", config)
  vision_cfg = config.get("vision_config", {})
  # text_base matches GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING logic
  text_base = "model.language_model" if vision_cfg else "model"

  hidden_size = text_cfg["hidden_size"]
  intermediate_size = text_cfg["intermediate_size"]
  num_hidden_layers = text_cfg["num_hidden_layers"]
  num_attention_heads = text_cfg["num_attention_heads"]
  num_key_value_heads = text_cfg["num_key_value_heads"]
  num_global_key_value_heads = text_cfg.get("num_global_key_value_heads", num_key_value_heads)
  head_dim = text_cfg["head_dim"]
  global_head_dim = text_cfg.get("global_head_dim", head_dim)
  vocab_size = text_cfg["vocab_size"]

  num_experts = text_cfg.get("num_experts")
  num_experts = num_experts if num_experts is not None else 1
  # "moe_intermediate_size" is the canonical key in Gemma4 config; fall back to "expert_intermediate_size"
  expert_intermediate_size = text_cfg.get("moe_intermediate_size") or text_cfg.get("expert_intermediate_size")

  shapes[f"{text_base}.embed_tokens.weight"] = [vocab_size, hidden_size]
  shapes[f"{text_base}.norm.weight"] = [hidden_size]

  for i in range(num_hidden_layers):
    hf_prefix = f"{text_base}.layers.{i}"
    is_global = (i % 6) == 5

    if is_global:
      q_dim = num_attention_heads * global_head_dim
      kv_dim = num_global_key_value_heads * global_head_dim
      norm_dim = global_head_dim
    else:
      q_dim = num_attention_heads * head_dim
      kv_dim = num_key_value_heads * head_dim
      norm_dim = head_dim

    shapes[f"{hf_prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    shapes[f"{hf_prefix}.self_attn.k_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{hf_prefix}.self_attn.v_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{hf_prefix}.self_attn.o_proj.weight"] = [hidden_size, q_dim]
    shapes[f"{hf_prefix}.self_attn.q_norm.weight"] = [norm_dim]
    shapes[f"{hf_prefix}.self_attn.k_norm.weight"] = [norm_dim]
    # v_norm is conditional on maxtext_config.v_norm_with_scale; included here for completeness
    shapes[f"{hf_prefix}.self_attn.v_norm.weight"] = [norm_dim]

    shapes[f"{hf_prefix}.input_layernorm.weight"] = [hidden_size]
    shapes[f"{hf_prefix}.post_attention_layernorm.weight"] = [hidden_size]
    shapes[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{hf_prefix}.post_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{hf_prefix}.layer_scalar"] = [1]

    if num_experts > 1:
      shapes[f"{hf_prefix}.pre_feedforward_layernorm_2.weight"] = [hidden_size]
      shapes[f"{hf_prefix}.post_feedforward_layernorm_1.weight"] = [hidden_size]
      shapes[f"{hf_prefix}.post_feedforward_layernorm_2.weight"] = [hidden_size]
      # router.scale has shape [hidden_size] (pre_forward_scale_2 in MaxText)
      shapes[f"{hf_prefix}.router.scale"] = [hidden_size]
      shapes[f"{hf_prefix}.router.proj.weight"] = [num_experts, hidden_size]
      shapes[f"{hf_prefix}.router.per_expert_scale"] = [num_experts]
      # Routed experts fused: gate_up [E, 2*FF, H], down [E, H, FF]
      shapes[f"{hf_prefix}.experts.gate_up_proj"] = [num_experts, 2 * expert_intermediate_size, hidden_size]
      shapes[f"{hf_prefix}.experts.down_proj"] = [num_experts, hidden_size, expert_intermediate_size]
      # Shared expert dense MLP
      shapes[f"{hf_prefix}.mlp.gate_proj.weight"] = [intermediate_size, hidden_size]
      shapes[f"{hf_prefix}.mlp.up_proj.weight"] = [intermediate_size, hidden_size]
      shapes[f"{hf_prefix}.mlp.down_proj.weight"] = [hidden_size, intermediate_size]
    else:
      shapes[f"{hf_prefix}.mlp.gate_proj.weight"] = [intermediate_size, hidden_size]
      shapes[f"{hf_prefix}.mlp.up_proj.weight"] = [intermediate_size, hidden_size]
      shapes[f"{hf_prefix}.mlp.down_proj.weight"] = [hidden_size, intermediate_size]

  if vision_cfg:
    vis_hidden = vision_cfg["hidden_size"]
    vis_intermediate = vision_cfg["intermediate_size"]
    vis_num_layers = vision_cfg["num_hidden_layers"]
    vis_num_heads = vision_cfg["num_attention_heads"]
    vis_head_dim = vision_cfg["head_dim"]
    vis_q_dim = vis_num_heads * vis_head_dim
    vis_kv_heads = vision_cfg.get("num_key_value_heads", vis_num_heads)
    vis_kv_dim = vis_kv_heads * vis_head_dim
    vis_pos_emb_size = vision_cfg.get("position_embedding_size", 10240)
    vis_patch_size = vision_cfg.get("patch_size", 16)
    num_channels = 3  # RGB
    patch_flat = num_channels * vis_patch_size * vis_patch_size

    # VisionEntry: input_proj is a linear [patch_flat, vis_hidden] transposed to [vis_hidden, patch_flat]
    shapes["model.vision_tower.patch_embedder.input_proj.weight"] = [vis_hidden, patch_flat]
    # pos_emb_param MaxText shape (N, 2, D) -> transpose(1,0,2) -> HF (2, N, D)
    shapes["model.vision_tower.patch_embedder.position_embedding_table"] = [2, vis_pos_emb_size, vis_hidden]
    shapes["model.vision_tower.std_scale"] = [vis_hidden]
    shapes["model.vision_tower.std_bias"] = [vis_hidden]
    # Vision projector: [vis_hidden, hidden_size] -> reshape_kernel -> [hidden_size, vis_hidden]
    shapes["model.embed_vision.embedding_projection.weight"] = [hidden_size, vis_hidden]

    for i in range(vis_num_layers):
      vis_prefix = f"model.vision_tower.encoder.layers.{i}"
      shapes[f"{vis_prefix}.self_attn.q_proj.linear.weight"] = [vis_q_dim, vis_hidden]
      shapes[f"{vis_prefix}.self_attn.k_proj.linear.weight"] = [vis_kv_dim, vis_hidden]
      shapes[f"{vis_prefix}.self_attn.v_proj.linear.weight"] = [vis_kv_dim, vis_hidden]
      shapes[f"{vis_prefix}.self_attn.o_proj.linear.weight"] = [vis_hidden, vis_q_dim]
      shapes[f"{vis_prefix}.self_attn.q_norm.weight"] = [vis_head_dim]
      shapes[f"{vis_prefix}.self_attn.k_norm.weight"] = [vis_head_dim]
      shapes[f"{vis_prefix}.input_layernorm.weight"] = [vis_hidden]
      shapes[f"{vis_prefix}.post_attention_layernorm.weight"] = [vis_hidden]
      shapes[f"{vis_prefix}.pre_feedforward_layernorm.weight"] = [vis_hidden]
      shapes[f"{vis_prefix}.post_feedforward_layernorm.weight"] = [vis_hidden]
      shapes[f"{vis_prefix}.mlp.gate_proj.linear.weight"] = [vis_intermediate, vis_hidden]
      shapes[f"{vis_prefix}.mlp.up_proj.linear.weight"] = [vis_intermediate, vis_hidden]
      shapes[f"{vis_prefix}.mlp.down_proj.linear.weight"] = [vis_hidden, vis_intermediate]

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


def DEEPSEEK_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and their shape derived from HF config.

  Args:
      config (dict): HF configuration dictionary
        e.g., https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a list

  To check expected mapping:
    from transformers import AutoModelForCausalLM
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    for name, val in model.named_parameters():
      print(name, val.shape)
  """
  # --- Core Config Values ---
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  attention_bias = config.get("attention_bias", False)

  # --- Attention-related Dimensions ---
  q_lora_rank = config["q_lora_rank"]
  kv_lora_rank = config["kv_lora_rank"]
  num_attention_heads = config["num_attention_heads"]

  # qk_head_dim is present in DeepseekV3Config, but missing from DeepseekV2Config
  qk_head_dim = config.get("qk_head_dim", config["qk_nope_head_dim"] + config["qk_rope_head_dim"])
  # Q projection dim
  q_dim = num_attention_heads * qk_head_dim

  # kv_b_proj output dim
  kv_b_dim = num_attention_heads * (config["qk_nope_head_dim"] + config["v_head_dim"])
  # Output projection dim (input)
  o_proj_in_dim = num_attention_heads * config["v_head_dim"]
  # kv_a_proj_with_mqa output dim
  kv_a_proj_out_dim = config["kv_lora_rank"] + config["qk_rope_head_dim"]

  # --- MLP-related Dimensions ---
  intermediate_size = config["intermediate_size"]  # For dense layers
  moe_intermediate_size = config["moe_intermediate_size"]  # For expert layers
  n_routed_experts = config["n_routed_experts"]
  n_shared_experts = config.get("n_shared_experts", 0)
  # This key determines which layers are dense vs. MoE
  first_k_dense = config.get("first_k_dense_replace", 0)

  # --- Indexer Configuration (Optional) ---
  index_head_dim = config.get("index_head_dim")
  index_n_heads = config.get("index_n_heads")

  # --- Non-layer-specific weights ---
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }

  # --- Loop Over Layers ---
  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"

    # --- Attention weights ---
    layer_mapping = {
        # norm
        f"{layer_prefix}.input_layernorm.weight": [hidden_size],
        f"{layer_prefix}.post_attention_layernorm.weight": [hidden_size],
        # kv projection
        f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.weight": [kv_a_proj_out_dim, hidden_size],
        f"{layer_prefix}.self_attn.kv_a_layernorm.weight": [kv_lora_rank],
        f"{layer_prefix}.self_attn.kv_b_proj.weight": [kv_b_dim, kv_lora_rank],
        # output projection
        f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, o_proj_in_dim],
    }

    # query projection
    if q_lora_rank is None:
      layer_mapping[f"{layer_prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    else:
      layer_mapping.update(
          {
              f"{layer_prefix}.self_attn.q_a_proj.weight": [q_lora_rank, hidden_size],
              f"{layer_prefix}.self_attn.q_a_layernorm.weight": [q_lora_rank],
              f"{layer_prefix}.self_attn.q_b_proj.weight": [q_dim, q_lora_rank],
          }
      )

    # bias
    if attention_bias:
      if q_lora_rank is not None:
        layer_mapping[f"{layer_prefix}.self_attn.q_a_proj.bias"] = [q_lora_rank]
      layer_mapping.update(
          {
              f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.bias": [kv_a_proj_out_dim],
              f"{layer_prefix}.self_attn.o_proj.bias": [hidden_size],
          }
      )

    # indexer for sparse attention
    if index_head_dim is not None and index_n_heads is not None and q_lora_rank is not None:
      wq_b_dim_out = index_n_heads * index_head_dim
      indexer_prefix = f"{layer_prefix}.self_attn.indexer"
      layer_mapping.update(
          {
              f"{indexer_prefix}.k_norm.bias": [index_head_dim],
              f"{indexer_prefix}.k_norm.weight": [index_head_dim],
              f"{indexer_prefix}.weights_proj.weight": [index_n_heads, hidden_size],
              f"{indexer_prefix}.wk.weight": [index_head_dim, hidden_size],
              f"{indexer_prefix}.wq_b.weight": [wq_b_dim_out, q_lora_rank],
          }
      )

    # --- MLP weights (Dense vs. MoE) ---
    if layer_idx < first_k_dense:
      # This is a DENSE MLP layer
      layer_mapping.update(
          {
              f"{layer_prefix}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.up_proj.weight": [intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.down_proj.weight": [hidden_size, intermediate_size],
          }
      )
    else:
      # This is a MoE MLP layer
      # Add the router gate
      layer_mapping.update(
          {
              f"{layer_prefix}.mlp.gate.weight": [n_routed_experts, hidden_size],
              f"{layer_prefix}.mlp.gate.e_score_correction_bias": [n_routed_experts],
          }
      )

      # Add routed experts
      for expert_j in range(n_routed_experts):
        expert_prefix = f"{layer_prefix}.mlp.experts.{expert_j}"
        layer_mapping.update(
            {
                f"{expert_prefix}.gate_proj.weight": [moe_intermediate_size, hidden_size],
                f"{expert_prefix}.up_proj.weight": [moe_intermediate_size, hidden_size],
                f"{expert_prefix}.down_proj.weight": [hidden_size, moe_intermediate_size],
            }
        )

      # Add shared experts (if any)
      if n_shared_experts > 0:
        shared_intermediate_size = moe_intermediate_size * n_shared_experts
        layer_mapping.update(
            {
                f"{layer_prefix}.mlp.shared_experts.gate_proj.weight": [shared_intermediate_size, hidden_size],
                f"{layer_prefix}.mlp.shared_experts.up_proj.weight": [shared_intermediate_size, hidden_size],
                f"{layer_prefix}.mlp.shared_experts.down_proj.weight": [hidden_size, shared_intermediate_size],
            }
        )

    mapping.update(layer_mapping)

  return mapping


def QWEN3_NEXT_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen3-Next weights path and their shape."""
  # --- Extract Core Config Values ---
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config["num_key_value_heads"]
  num_experts = config["num_experts"]
  head_dim = config["head_dim"]
  linear_conv_kernel_dim = config["linear_conv_kernel_dim"]
  linear_key_head_dim = config["linear_key_head_dim"]
  linear_num_key_heads = config["linear_num_key_heads"]
  linear_num_value_heads = config["linear_num_value_heads"]
  moe_intermediate_size = config["moe_intermediate_size"]
  shared_expert_intermediate_size = config["shared_expert_intermediate_size"]
  cycle_interval = config["full_attention_interval"]

  # --- Calculated Values ---
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim

  linear_k_dim = linear_num_key_heads * linear_key_head_dim
  linear_v_dim = linear_num_value_heads * head_dim
  conv_dim = 2 * linear_k_dim + linear_v_dim
  qkvz_dim = 2 * linear_k_dim + 2 * linear_v_dim
  ba_dim = 2 * linear_num_value_heads

  # --- Initialize Mapping ---
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }

  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"

    # Standard Layer Norms
    mapping[f"{layer_prefix}.input_layernorm.weight"] = [hidden_size]
    mapping[f"{layer_prefix}.post_attention_layernorm.weight"] = [hidden_size]

    is_full_attention_layer = (layer_idx + 1) % cycle_interval == 0

    if is_full_attention_layer:
      # Full Attention Block
      mapping.update(
          {
              f"{layer_prefix}.self_attn.q_proj.weight": [2 * q_dim, hidden_size],
              f"{layer_prefix}.self_attn.k_proj.weight": [kv_dim, hidden_size],
              f"{layer_prefix}.self_attn.v_proj.weight": [kv_dim, hidden_size],
              f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, q_dim],
              f"{layer_prefix}.self_attn.q_norm.weight": [head_dim],
              f"{layer_prefix}.self_attn.k_norm.weight": [head_dim],
          }
      )
    else:
      # Linear Attention (GDN) Block
      mapping.update(
          {
              f"{layer_prefix}.linear_attn.in_proj_qkvz.weight": [qkvz_dim, hidden_size],
              f"{layer_prefix}.linear_attn.in_proj_ba.weight": [ba_dim, hidden_size],
              f"{layer_prefix}.linear_attn.conv1d.weight": [conv_dim, 1, linear_conv_kernel_dim],
              f"{layer_prefix}.linear_attn.A_log": [linear_num_value_heads],
              f"{layer_prefix}.linear_attn.dt_bias": [linear_num_value_heads],
              f"{layer_prefix}.linear_attn.norm.weight": [head_dim],
              f"{layer_prefix}.linear_attn.out_proj.weight": [hidden_size, linear_v_dim],
          }
      )

    # --- MLP Logic (MoE + Shared) ---
    mapping.update(
        {
            # Router
            f"{layer_prefix}.mlp.gate.weight": [num_experts, hidden_size],
            # Shared Experts (SwiGLU - Separate Weights)
            f"{layer_prefix}.mlp.shared_expert.gate_proj.weight": [shared_expert_intermediate_size, hidden_size],
            f"{layer_prefix}.mlp.shared_expert.up_proj.weight": [shared_expert_intermediate_size, hidden_size],
            f"{layer_prefix}.mlp.shared_expert.down_proj.weight": [hidden_size, shared_expert_intermediate_size],
            # Shared Expert Gate (learned scaling factor)
            f"{layer_prefix}.mlp.shared_expert_gate.weight": [1, hidden_size],
        }
    )

    # Routed Experts Loop
    # Note: HF typically stores experts as a ModuleList
    for e in range(num_experts):
      mapping.update(
          {
              f"{layer_prefix}.mlp.experts.{e}.gate_proj.weight": [moe_intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.experts.{e}.up_proj.weight": [moe_intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.experts.{e}.down_proj.weight": [hidden_size, moe_intermediate_size],
          }
      )


def GPT_OSS_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace GptOss weights path and their shape."""
  # --- Extract Core Config Values ---
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
  intermediate_size = config["intermediate_size"]
  num_local_experts = config["num_local_experts"]
  attention_bias = config.get("attention_bias", False)
  # Calculate head dimension if not explicitly provided
  head_dim = config.get("head_dim", hidden_size // num_attention_heads)

  # Calculate Attention dimensions
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim

  # --- Initialize Mapping ---
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }

  # --- Loop Over Layers ---
  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"

    # --- Standard Layer Components ---
    layer_mapping = {
        f"{layer_prefix}.input_layernorm.weight": [hidden_size],
        f"{layer_prefix}.post_attention_layernorm.weight": [hidden_size],
    }

    # --- Attention Weights (GptOssAttention) ---
    layer_mapping.update(
        {
            # Standard QKV projections (Linear: [out_features, in_features])
            f"{layer_prefix}.self_attn.q_proj.weight": [q_dim, hidden_size],
            f"{layer_prefix}.self_attn.k_proj.weight": [kv_dim, hidden_size],
            f"{layer_prefix}.self_attn.v_proj.weight": [kv_dim, hidden_size],
            f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, q_dim],
            # Attention sinks unique to this model
            f"{layer_prefix}.self_attn.sinks": [num_attention_heads],
        }
    )

    if attention_bias:
      layer_mapping.update(
          {
              f"{layer_prefix}.self_attn.q_proj.bias": [q_dim],
              f"{layer_prefix}.self_attn.k_proj.bias": [kv_dim],
              f"{layer_prefix}.self_attn.v_proj.bias": [kv_dim],
              f"{layer_prefix}.self_attn.o_proj.bias": [hidden_size],
          }
      )

    # --- MoE MLP Weights (GptOssMLP) ---
    # Router (GptOssTopKRouter)
    layer_mapping.update(
        {
            f"{layer_prefix}.mlp.router.weight": [num_local_experts, hidden_size],
            f"{layer_prefix}.mlp.router.bias": [num_local_experts],
        }
    )

    # Experts (GptOssExperts)
    layer_mapping.update(
        {
            # Fused gate and up projection: [num_experts, hidden, 2 * intermediate]
            f"{layer_prefix}.mlp.experts.gate_up_proj": [num_local_experts, hidden_size, 2 * intermediate_size],
            f"{layer_prefix}.mlp.experts.gate_up_proj_bias": [num_local_experts, 2 * intermediate_size],
            # Down projection: [num_experts, intermediate, hidden]
            f"{layer_prefix}.mlp.experts.down_proj": [num_local_experts, intermediate_size, hidden_size],
            f"{layer_prefix}.mlp.experts.down_proj_bias": [num_local_experts, hidden_size],
        }
    )

    mapping.update(layer_mapping)

  return mapping


def QWEN_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen weights path and the HuggingFace weights shape.

  Args:
      config (dict): HF configuration dictionary (from Qwen3TextConfig.to_dict())
          e.g., https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a list

  To check expected mapping:
    from transformers import AutoModelForCausalLM
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    for name, val in model.named_parameters():
      print(name, val.shape)
  """
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config["num_key_value_heads"]
  head_dim = config.get(
      "head_dim", config["hidden_size"] // config["num_attention_heads"]
  )  # head_dim might not always be present
  attention_bias = config.get("attention_bias", False)

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

    if attention_bias:
      layer_mapping.update(
          {
              f"{layer_prefix}.self_attn.q_proj.bias": [num_attention_heads * head_dim],
              f"{layer_prefix}.self_attn.k_proj.bias": [num_key_value_heads * head_dim],
              f"{layer_prefix}.self_attn.v_proj.bias": [num_key_value_heads * head_dim],
          }
      )

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


def MIXTRAL_HF_WEIGHTS_TO_SHAPE(config):
  """
  Returns a mapping of Hugging Face parameter names to their tensor shapes.

  Args:
      config (dict): The model configuration dictionary.

  Returns:
      A dictionary mapping Hugging Face parameter paths to their tensor shapes.
  """
  shapes = {}

  # Embedding and LM Head
  shapes["model.embed_tokens.weight"] = [config["vocab_size"], config["hidden_size"]]
  shapes["lm_head.weight"] = [config["vocab_size"], config["hidden_size"]]

  # Final LayerNorm
  shapes["model.norm.weight"] = [config["hidden_size"]]

  # Calculated dimensions
  head_dim = config["hidden_size"] // config["num_attention_heads"]
  kv_dim = config["num_key_value_heads"] * head_dim

  # Decoder Layers
  for i in range(config["num_hidden_layers"]):
    # Attention Projections
    shapes[f"model.layers.{i}.self_attn.q_proj.weight"] = [
        config["hidden_size"],
        config["hidden_size"],
    ]
    shapes[f"model.layers.{i}.self_attn.k_proj.weight"] = [
        kv_dim,
        config["hidden_size"],
    ]
    shapes[f"model.layers.{i}.self_attn.v_proj.weight"] = [
        kv_dim,
        config["hidden_size"],
    ]
    shapes[f"model.layers.{i}.self_attn.o_proj.weight"] = [
        config["hidden_size"],
        config["hidden_size"],
    ]

    # LayerNorms
    shapes[f"model.layers.{i}.input_layernorm.weight"] = [config["hidden_size"]]
    shapes[f"model.layers.{i}.post_attention_layernorm.weight"] = [config["hidden_size"]]

    # MOE Gate
    shapes[f"model.layers.{i}.block_sparse_moe.gate.weight"] = [
        config["num_local_experts"],
        config["hidden_size"],
    ]

    # MOE Experts
    for j in range(config["num_local_experts"]):
      shapes[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"] = [
          config["intermediate_size"],
          config["hidden_size"],
      ]
      shapes[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"] = [
          config["hidden_size"],
          config["intermediate_size"],
      ]
      shapes[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"] = [
          config["intermediate_size"],
          config["hidden_size"],
      ]

  return shapes


# {maxtext model name: {hf weight name: hf shape}}
HF_SHAPE = {
    "gemma2-2b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-9b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-27b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma3-4b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-12b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-27b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma4-26b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-31b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-1.5b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-7b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-0.6b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b-thinking-2507": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-8b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-32b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-70b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-405b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-235b-a22b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-480b-a35b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "deepseek2-16b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3.2-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-20b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-120b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x7b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x22b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
}
