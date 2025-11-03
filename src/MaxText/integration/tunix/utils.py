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

"""Utils for Tunix integration."""

import re

import maxtext.src.maxtext.integration.tunix.weight_mapping as weight_mapping  # pylint: disable=consider-using-from-import
from MaxText.utils.ckpt_conversion.utils.param_mapping import PARAM_MAPPING
from MaxText.utils.ckpt_conversion.utils.param_mapping import VLLM_HOOK_FNS

STANDALONE_VLLM_WEIGHT_MAPPING = weight_mapping.StandaloneVllmWeightMapping()


class VllmWeightMapping:
  """Mapping MaxText model weights to vLLM's model weights."""

  def __init__(self, model_name, config=None, use_standalone_mappings=False):
    self.model_name = model_name
    self.config = config
    self.use_standalone_mappings = use_standalone_mappings
    self._sharding_knowledge_map = _SHARDING_KNOWLEDGE_MAP[self.model_name.split("-")[0]]
    self._maxtext_keys_to_vllm_keys = _MAXTEXT_TO_VLLM_KEY_MAP[self.model_name.split("-")[0]]

  def to_hf_mapping(self):
    """Returns a mapping from MaxText parameter names to HuggingFace parameter names."""
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_mapping()

    config = self.config
    mapping = self.convert_hf_map_to_sharding_map(PARAM_MAPPING[self.model_name](config, scan_layers=True))
    return mapping

  def to_hf_transpose_keys(self):
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_transpose_keys()

    return {}

  def to_hf_hook_fns(self):
    """Returns a mapping from MaxText parameter names to transformation functions."""
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_hook_fns()

    model_family = self.model_name.split("-")[0]
    return VLLM_HOOK_FNS[model_family]()

  def lora_to_hf_mappings(self):
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].lora_to_hf_mappings()

    return None

  def _generalize_maxtext_key(self, maxtext_key):
    """
    Universal generalizer for Qwen3, DeepSeek, and Llama3.1 keys.
    Converts raw MaxText keys to a standardized 'base' format for sharding maps.
    """
    # 1. Standardize separators and prefix
    # 'params-decoder-...' -> 'base.decoder....'
    # 'thinker.params-decoder-...' -> 'thinker.base.decoder....' (Qwen3-Omni)
    generic_key = maxtext_key.replace("params-", "base.").replace("-", ".")

    # 2. Normalize standard layer indexing (crucial for unscanned models)
    # Llama/Qwen unscanned: 'base.decoder.layers_5.self_attention...' -> 'base.decoder.layers.self_attention...'
    generic_key = re.sub(r"\.layers_\d+\.", ".layers.", generic_key)

    # 3. Normalize DeepSeek specific layer indexing (if unscanned support is added later)
    # Preserves the distinction between 'dense_layers' and 'moe_layers'
    generic_key = re.sub(r"\.dense_layers_\d+\.", ".dense_layers.", generic_key)
    generic_key = re.sub(r"\.moe_layers_\d+\.", ".moe_layers.", generic_key)

    return generic_key

  def _generalize_hf_value(self, hf_value):
    """Extracts and generalizes the Hugging Face name from the hf_value."""
    return self._maxtext_keys_to_vllm_keys(hf_value)

  def convert_hf_map_to_sharding_map(self, hf_mapping):
    """Converts a MaxText-to-HF name map into a generic MaxText-to-vLLM sharding map.

    Args:
      hf_mapping (dict): The output from QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING.
        - Keys are MaxText param names (e.g., "params-decoder-layers...").
        - Values are HF param names (str) or lists of names (list).

    Returns:
      dict: A mapping from generalized MaxText names (e.g.,
        "base.decoder.layers.mlp.wi_0.kernel") to a tuple containing:
        (str: generalized HF/vLLM name, tuple: sharding specification).
    """
    sharding_map = {}
    for maxtext_key, hf_value in hf_mapping.items():
      # 1. Generalize the MaxText key
      generic_key = self._generalize_maxtext_key(maxtext_key)

      # 2. Generalize the Hugging Face (HF) value name
      corrected_value = self._generalize_hf_value(hf_value)
      if corrected_value is None:
        continue

      # 4. Look up the sharding tuple
      sharding_tuple = self._sharding_knowledge_map.get(generic_key)

      if sharding_tuple is None:
        # This warning is fine if it's for unscanned layers,
        # as we only want the generic "base.decoder.layers.*" key
        if "layers." not in generic_key:
          print(f"Warning: No sharding rule found for key: {generic_key}")
        continue
      # 5. Assemble the final map entry
      sharding_map[generic_key] = (corrected_value, sharding_tuple)

    return sharding_map


def GENERAL_HF_KEYS_TO_VLLM_KEYS(hf_value):
  """Converts a concrete HF key (or list of keys) to a vLLM template string."""
  first_name = ""
  if isinstance(hf_value, str):
    first_name = hf_value
  elif isinstance(hf_value, list):
    if not hf_value:
      return None
    if isinstance(hf_value[0], list):
      first_name = hf_value[0][0]  # Scanned MoE
    else:
      first_name = hf_value[0]  # Scanned Dense / Unscanned MoE
  else:
    raise TypeError(f"Unknown value type in map: {type(hf_value)}")

  # Replace layer and expert indices with wildcards
  wildcard_name = re.sub(r"layers\.(\d+)\.", "layers.*.", first_name)
  wildcard_name = re.sub(r"experts\.(\d+)\.", "experts.*.", wildcard_name)

  if "layernorm.weight" in wildcard_name or "_norm.weight" in wildcard_name:
    # Fix all layer norms
    wildcard_name = wildcard_name.replace(".weight", ".scale")
  elif wildcard_name == "model.embed_tokens.weight":
    wildcard_name = "model.embed.embedding"
  elif wildcard_name == "lm_head.weight":
    wildcard_name = "model.lm_head"
  elif wildcard_name == "model.norm.weight":
    wildcard_name = "model.norm.scale"
  elif wildcard_name.endswith(".weight"):
    # Fix all other weights (MLP, Attn)
    wildcard_name = wildcard_name.replace(".weight", ".kernel")
  return wildcard_name


def DEEPSEEK_HF_KEYS_TO_VLLM_KEYS(hf_input):
  """
  Converts a concrete HF key (or list of keys) to a vLLM template string.
  Handles both single strings and lists of strings.
  """
  if not hf_input:
    return None

  # 1. Standardize input to a single representative sample string
  if isinstance(hf_input, str):
    sample_key = hf_input
  elif isinstance(hf_input, list):
    if not hf_input:
      return None
    if isinstance(hf_input[0], list):
      sample_key = hf_input[0][0]  # Scanned MoE
    else:
      sample_key = hf_input[0]  # Scanned Dense / Unscanned MoE
  else:
    raise TypeError(f"Unknown value type in map: {type(hf_input)}")

  # 2. Structural Generalization (convert specific indices to wildcards)
  # Replace 'model.layers.{N}.' with 'layers.*.'

  template = re.sub(r"^model\.layers\.\d+\.", "layers.*.", sample_key)

  # 3. Leaf Node Renaming (HF -> vLLM intermediate names)
  leaf_replacements = [
      # --- Globals ---
      (r"^model\.norm\.weight$", "final_norm.scale"),
      (r"^model\.embed_tokens\.weight$", "embedder.input_embedding_table_VD"),
      (r"^lm_head\.weight$", "lm_head.input_embedding_table_DV"),
      # --- MoE: Router (Gate) ---
      # specific to DeepSeek's 'mlp.gate'
      (r"mlp\.gate\.weight$", "custom_module.router.kernel_DE"),
      (r"mlp\.gate\.e_score_correction_bias$", "custom_module.router.bias_E"),
      # --- MoE: Shared Experts ---
      # specific to DeepSeek's 'mlp.shared_experts'
      (r"mlp\.shared_experts\.gate_proj\.weight$", "shared_experts.kernel_gating_DF"),
      (r"mlp\.shared_experts\.up_proj\.weight$", "shared_experts.kernel_up_proj_DF"),
      (r"mlp\.shared_experts\.down_proj\.weight$", "shared_experts.kernel_down_proj_FD"),
      # --- MoE: Routed Experts (individual experts) ---
      # Matches 'mlp.experts.0.gate_proj.weight' etc.
      # We generalize the expert ID to 'experts.*' if you want them all to map to one template key
      # OR if your target mapping uses 'custom_module.kernel_gating_EDF' directly for all:
      (r"mlp\.experts\.\d+\.gate_proj\.weight$", "custom_module.kernel_gating_EDF"),
      (r"mlp\.experts\.\d+\.up_proj\.weight$", "custom_module.kernel_up_proj_EDF"),
      (r"mlp\.experts\.\d+\.down_proj\.weight$", "custom_module.kernel_down_proj_EFD"),
      # --- Standard Dense MLP (Fallback for non-MoE layers) ---
      (r"mlp\.gate_proj\.weight$", "custom_module.kernel_gating_DF"),
      (r"mlp\.up_proj\.weight$", "custom_module.kernel_up_proj_DF"),
      (r"mlp\.down_proj\.weight$", "custom_module.kernel_down_proj_FD"),
      # --- Attention & Norms (Standard) ---
      (r"input_layernorm\.weight$", "pre_attention_norm.scale"),
      (r"post_attention_layernorm\.weight$", "pre_mlp_norm.scale"),
      (r"self_attn\.q_a_layernorm\.weight$", "attn.q_rms_norm.scale"),
      (r"self_attn\.kv_a_layernorm\.weight$", "attn.kv_rms_norm.scale"),
      (r"self_attn\.q_proj\.weight$", "attn.kernel_q_proj"),
      (r"self_attn\.q_a_proj\.weight$", "attn.kernel_q_down_proj_DA"),
      (r"self_attn\.q_b_proj\.weight$", "attn.kernel_q_up_proj_ANH"),
      (r"self_attn\.kv_a_proj_with_mqa\.weight$", "attn.kernel_kv_down_proj_DA"),
      (r"self_attn\.kv_b_proj\.weight$", "attn.kernel_kv_up_proj_ANH"),
      (r"self_attn\.o_proj\.weight$", "attn.kernel_o_proj_NHD"),
  ]

  for pattern, replacement in leaf_replacements:
    template = re.sub(pattern, replacement, template)

  return template


GENERAL_SHARDING_MAP = {
    # Non-layer parameters
    "base.token_embedder.embedding": ("model", None),
    "base.decoder.decoder_norm.scale": (None,),
    "base.decoder.logits_dense.kernel": (None, "model"),
    # --- Attention (generic for scanned/unscanned) ---
    "base.decoder.layers.pre_self_attention_layer_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.query.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.key.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.value.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.query_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.key_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.out.kernel": (
        "model",
        "layer",
        None,
        None,
    ),
    "base.decoder.layers.post_self_attention_layer_norm.scale": (None, "layer"),
    # --- Dense MLP (generic for scanned/unscanned) ---
    "base.decoder.layers.mlp.wi_0.kernel": (None, "layer", "model"),
    "base.decoder.layers.mlp.wi_1.kernel": (None, "layer", "model"),
    "base.decoder.layers.mlp.wo.kernel": ("model", "layer", None),
    # --- MoE (generic for scanned/unscanned) ---
    "base.decoder.layers.moe_block.gate.kernel": (None, "layer", "model"),
    "base.decoder.layers.moe_block.wi_0": ("expert", "layer", None, "model"),
    "base.decoder.layers.moe_block.wi_1": ("expert", "layer", None, "model"),
    "base.decoder.layers.moe_block.wo": ("expert", "layer", "model", None),
    # --- Deepseek Attention ---
    "base.decoder.layers.self_attention.wq_a.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.wq_b.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.q_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.wkv_a.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.wkv_b.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.kv_norm.scale": (None, "layer"),
    # --- Deepseek MoE ---
    "base.decoder.layers.moe_block.shared_experts.wi_0.kernel": (
        None,
        "layer",
        "model",
    ),
    "base.decoder.layers.moe_block.shared_experts.wi_1.kernel": (
        None,
        "layer",
        "model",
    ),
    "base.decoder.layers.moe_block.shared_experts.wo.kernel": (
        "model",
        "layer",
        None,
    ),
    "base.decoder.layers.moe_block.gate.bias": (None, "layer", "model"),
}


DEEPSEEK_SHARDING_MAP = {
    # --- Non-Layer Parameters ---
    "base.token_embedder.embedding": ("model", None),
    "base.decoder.decoder_norm.scale": (None,),
    "base.decoder.logits_dense.kernel": (None, "model"),
    # ==============================================================================
    # DENSE LAYERS MAPPING
    # ==============================================================================
    "base.decoder.dense_layers.pre_self_attention_layer_norm.scale": (None, "layer"),
    "base.decoder.dense_layers.post_self_attention_layer_norm.scale": (None, "layer"),
    # --- Attention (MLA) ---
    # Q projections (Down/Up)
    "base.decoder.dense_layers.self_attention.wq_a.kernel": (None, "layer", "model", None),
    "base.decoder.dense_layers.self_attention.wq_b.kernel": (None, "layer", "model", None),
    # KV projections (Down/Up with MQA)
    "base.decoder.dense_layers.self_attention.wkv_a.kernel": (None, "layer", "model", None),
    "base.decoder.dense_layers.self_attention.wkv_b.kernel": (None, "layer", "model", None),
    # Output projection
    "base.decoder.dense_layers.self_attention.out.kernel": ("model", "layer", None, None),
    # MLA Norms
    "base.decoder.dense_layers.self_attention.kv_norm.scale": (None, "layer"),
    "base.decoder.dense_layers.self_attention.q_norm.scale": (None, "layer"),
    # --- Dense MLP ---
    "base.decoder.dense_layers.mlp.wi_0.kernel": (None, "layer", "model"),
    "base.decoder.dense_layers.mlp.wi_1.kernel": (None, "layer", "model"),
    "base.decoder.dense_layers.mlp.wo.kernel": ("model", "layer", None),
    # ==============================================================================
    # MOE LAYERS MAPPING
    # ==============================================================================
    "base.decoder.moe_layers.pre_self_attention_layer_norm.scale": (None, "layer"),
    "base.decoder.moe_layers.post_self_attention_layer_norm.scale": (None, "layer"),
    # --- Attention (MLA + Decoupled RoPE) for MoE Layers ---
    "base.decoder.moe_layers.self_attention.wq_a.kernel": (None, "layer", "model", None),
    "base.decoder.moe_layers.self_attention.wq_b.kernel": (None, "layer", "model", None),
    "base.decoder.moe_layers.self_attention.wkv_a.kernel": (None, "layer", "model", None),
    "base.decoder.moe_layers.self_attention.wkv_b.kernel": (None, "layer", "model", None),
    "base.decoder.moe_layers.self_attention.out.kernel": ("model", "layer", None, None),
    "base.decoder.moe_layers.self_attention.kv_norm.scale": (None, "layer"),
    "base.decoder.moe_layers.self_attention.q_norm.scale": (None, "layer"),
    # --- DeepSeek MoE Blocks ---
    # Shared Experts
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel": (None, "layer", "model"),
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel": (None, "layer", "model"),
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.shared_experts.wo.kernel": ("model", "layer", None),
    # Gating (Router)
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel": (None, "layer", "model"),
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias": (None, "layer", "model"),
    # Routed Experts
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_0": ("expert", "layer", None, "model"),
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_1": ("expert", "layer", None, "model"),
    "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wo": ("expert", "layer", "model", None),
}


_SHARDING_KNOWLEDGE_MAP = {
    "qwen3": GENERAL_SHARDING_MAP,
    "llama3": GENERAL_SHARDING_MAP,
    "deepseek3": DEEPSEEK_SHARDING_MAP,
}

_MAXTEXT_TO_VLLM_KEY_MAP = {
    "qwen3": GENERAL_HF_KEYS_TO_VLLM_KEYS,
    "llama3": GENERAL_HF_KEYS_TO_VLLM_KEYS,
    "deepseek3": DEEPSEEK_HF_KEYS_TO_VLLM_KEYS,
}
