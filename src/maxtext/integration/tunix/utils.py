# Copyright 2023–2026 Google LLC
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

import maxtext.integration.tunix.weight_mapping as weight_mapping  # pylint: disable=consider-using-from-import
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.param_mapping import VLLM_HOOK_FNS

STANDALONE_VLLM_WEIGHT_MAPPING = weight_mapping.StandaloneVllmWeightMapping()

# This static map provides the architectural knowledge (sharding) that is
# not present in the original HF mapping.
# Keys are the "generalized" MaxText names (e.g., base.decoder.layers...).
_SHARDING_KNOWLEDGE_MAP = {
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


class VllmWeightMapping:
  """Mapping MaxText model weights to vLLM's model weights."""

  def __init__(self, model_name, config=None, use_standalone_mappings=False):
    self.model_name = model_name
    self.config = config
    self.use_standalone_mappings = use_standalone_mappings
    self._sharding_knowledge_map = _SHARDING_KNOWLEDGE_MAP

  def to_hf_mapping(self):
    """Returns a mapping from MaxText parameter names to HuggingFace parameter names."""
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].to_hf_mapping()

    config = self.config
    mapping = self.convert_hf_map_to_sharding_map(
        PARAM_MAPPING[self.model_name](config, maxtext_config=None, scan_layers=True)
    )
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
    if model_family in VLLM_HOOK_FNS:
      return VLLM_HOOK_FNS[model_family]()
    else:
      return {}

  def lora_to_hf_mappings(self):
    if self.use_standalone_mappings:
      return STANDALONE_VLLM_WEIGHT_MAPPING[self.model_name].lora_to_hf_mappings()

    return None

  def _generalize_maxtext_key(self, maxtext_key):
    """Generalizes the MaxText key to a common vLLM format."""
    # 'params-decoder-layers_0-mlp-...' -> 'base.decoder.layers_0.mlp....'
    generic_key = maxtext_key.replace("params-", "base.").replace("-", ".")
    # 'base.decoder.dense_layers.mlp....' -> 'base.decoder.layers.mlp....'
    generic_key = re.sub(r"\.dense_layers\.", ".layers.", generic_key)
    # 'base.decoder.moe_layers.mlp....' -> 'base.decoder.layers.mlp....'
    generic_key = re.sub(r"\.moe_layers\.", ".layers.", generic_key)
    # '...layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_0' -> '...layers.moe_block.wi_0'
    generic_key = re.sub(r"DeepSeekMoeBlock_0\.MoeBlock_0\.", "moe_block.", generic_key)
    # Handle shared experts
    generic_key = re.sub(
        r"DeepSeekMoeBlock_0\.shared_experts\.",
        "moe_block.shared_experts.",
        generic_key,
    )
    # Keep original rule for other models
    generic_key = re.sub(r"layers_(\d+)\.", "layers.", generic_key)
    return generic_key

  def _generalize_hf_value(self, hf_value):
    """Extracts and generalizes the Hugging Face name from the hf_value."""
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
    return wildcard_name

  def _correct_hf_wildcard_name(self, wildcard_name):
    """Corrects the generated Hugging Face wildcard name."""
    corrected_name = wildcard_name
    if "layernorm.weight" in corrected_name or "_norm.weight" in corrected_name:
      # Fix all layer norms
      corrected_name = corrected_name.replace(".weight", ".scale")
    elif corrected_name == "model.embed_tokens.weight":
      corrected_name = "model.embed.embedding"
    elif corrected_name == "lm_head.weight":
      corrected_name = "model.lm_head"
    elif corrected_name == "model.norm.weight":
      corrected_name = "model.norm.scale"
    elif corrected_name.endswith(".weight"):
      # Fix all other weights (MLP, Attn)
      corrected_name = corrected_name.replace(".weight", ".kernel")
    return corrected_name

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
      wildcard_name = self._generalize_hf_value(hf_value)
      if wildcard_name is None:
        continue

      # 3. Correct the generated wildcard name
      corrected_name = self._correct_hf_wildcard_name(wildcard_name)

      # 4. Look up the sharding tuple
      sharding_tuple = self._sharding_knowledge_map.get(generic_key)

      if sharding_tuple is None:
        # This warning is fine if it's for unscanned layers,
        # as we only want the generic "base.decoder.layers.*" key
        if "layers." not in generic_key:
          print(f"Warning: No sharding rule found for key: {generic_key}")
        continue
      # 5. Assemble the final map entry
      sharding_map[generic_key] = (corrected_name, sharding_tuple)

    return sharding_map
