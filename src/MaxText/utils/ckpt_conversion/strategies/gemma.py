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

"""Gemma family parameter mapping strategies."""

import warnings
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp

from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy


class Gemma2Mapper(ParamMapperStrategy):
  """Strategy for Gemma2 models (2B, 9B, 27B)."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Returns the parameter mapping dictionary from MaxText to HuggingFace for Gemma2.

    MaxText uses a paired layer approach for Gemma2 where two HF decoder layers are
    treated as one MaxText decoder layer. Local components map to even-numbered HF
    layers, and global components map to odd-numbered HF layers.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.

    Returns:
      A dictionary mapping MaxText parameter keys to HuggingFace keys.
    """
    nlayers = hf_config["num_hidden_layers"]
    mapping = {
        "params-token_embedder-embedding": "model.embed_tokens.weight",
        "params-decoder-decoder_norm-scale": "model.norm.weight",
    }
    if scan_layers:
      mapping.update(
          {
              "params-decoder-layers-pre_self_attention_norm_global-scale": [
                  f"model.layers.{i}.input_layernorm.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-mlp_global-wo-kernel": [
                  f"model.layers.{i}.mlp.down_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-mlp_global-wi_1-kernel": [
                  f"model.layers.{i}.mlp.up_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-mlp_global-wi_0-kernel": [
                  f"model.layers.{i}.mlp.gate_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-post_self_attention_norm_global-scale": [
                  f"model.layers.{i}.post_attention_layernorm.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-post_ffw_norm_global-scale": [
                  f"model.layers.{i}.post_feedforward_layernorm.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-pre_ffw_norm_global-scale": [
                  f"model.layers.{i}.pre_feedforward_layernorm.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_global-key-kernel": [
                  f"model.layers.{i}.self_attn.k_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_global-out-kernel": [
                  f"model.layers.{i}.self_attn.o_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_global-query-kernel": [
                  f"model.layers.{i}.self_attn.q_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_global-value-kernel": [
                  f"model.layers.{i}.self_attn.v_proj.weight" for i in range(1, nlayers, 2)
              ],
              "params-decoder-layers-pre_self_attention_norm_local-scale": [
                  f"model.layers.{i}.input_layernorm.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-mlp_local-wo-kernel": [
                  f"model.layers.{i}.mlp.down_proj.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-mlp_local-wi_1-kernel": [
                  f"model.layers.{i}.mlp.up_proj.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-mlp_local-wi_0-kernel": [
                  f"model.layers.{i}.mlp.gate_proj.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-post_self_attention_norm_local-scale": [
                  f"model.layers.{i}.post_attention_layernorm.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-post_ffw_norm_local-scale": [
                  f"model.layers.{i}.post_feedforward_layernorm.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-pre_ffw_norm_local-scale": [
                  f"model.layers.{i}.pre_feedforward_layernorm.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_local-key-kernel": [
                  f"model.layers.{i}.self_attn.k_proj.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_local-out-kernel": [
                  f"model.layers.{i}.self_attn.o_proj.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_local-query-kernel": [
                  f"model.layers.{i}.self_attn.q_proj.weight" for i in range(0, nlayers, 2)
              ],
              "params-decoder-layers-self_attention_local-value-kernel": [
                  f"model.layers.{i}.self_attn.v_proj.weight" for i in range(0, nlayers, 2)
              ],
          }
      )
    else:
      for maxtext_layer_idx in range(0, nlayers // 2):
        local_layer_idx = maxtext_layer_idx * 2
        global_layer_idx = maxtext_layer_idx * 2 + 1
        prefix = f"params-decoder-layers_{maxtext_layer_idx}"
        mapping.update(
            {
                f"{prefix}-pre_self_attention_norm_global-scale": f"model.layers.{global_layer_idx}.input_layernorm.weight",
                f"{prefix}-mlp_global-wo-kernel": f"model.layers.{global_layer_idx}.mlp.down_proj.weight",
                f"{prefix}-mlp_global-wi_1-kernel": f"model.layers.{global_layer_idx}.mlp.up_proj.weight",
                f"{prefix}-mlp_global-wi_0-kernel": f"model.layers.{global_layer_idx}.mlp.gate_proj.weight",
                f"{prefix}-post_self_attention_norm_global-scale": (
                    f"model.layers.{global_layer_idx}" f".post_attention_layernorm.weight"
                ),
                f"{prefix}-post_ffw_norm_global-scale": f"model.layers.{global_layer_idx}.post_feedforward_layernorm.weight",
                f"{prefix}-pre_ffw_norm_global-scale": f"model.layers.{global_layer_idx}.pre_feedforward_layernorm.weight",
                f"{prefix}-self_attention_global-key-kernel": f"model.layers.{global_layer_idx}.self_attn.k_proj.weight",
                f"{prefix}-self_attention_global-out-kernel": f"model.layers.{global_layer_idx}.self_attn.o_proj.weight",
                f"{prefix}-self_attention_global-query-kernel": f"model.layers.{global_layer_idx}.self_attn.q_proj.weight",
                f"{prefix}-self_attention_global-value-kernel": f"model.layers.{global_layer_idx}.self_attn.v_proj.weight",
                f"{prefix}-pre_self_attention_norm_local-scale": f"model.layers.{local_layer_idx}.input_layernorm.weight",
                f"{prefix}-mlp_local-wo-kernel": f"model.layers.{local_layer_idx}.mlp.down_proj.weight",
                f"{prefix}-mlp_local-wi_1-kernel": f"model.layers.{local_layer_idx}.mlp.up_proj.weight",
                f"{prefix}-mlp_local-wi_0-kernel": f"model.layers.{local_layer_idx}.mlp.gate_proj.weight",
                f"{prefix}-post_self_attention_norm_local-scale": (
                    f"model.layers.{local_layer_idx}.post_attention_layernorm.weight"
                ),
                f"{prefix}-post_ffw_norm_local-scale": f"model.layers.{local_layer_idx}.post_feedforward_layernorm.weight",
                f"{prefix}-pre_ffw_norm_local-scale": f"model.layers.{local_layer_idx}.pre_feedforward_layernorm.weight",
                f"{prefix}-self_attention_local-key-kernel": f"model.layers.{local_layer_idx}.self_attn.k_proj.weight",
                f"{prefix}-self_attention_local-out-kernel": f"model.layers.{local_layer_idx}.self_attn.o_proj.weight",
                f"{prefix}-self_attention_local-query-kernel": f"model.layers.{local_layer_idx}.self_attn.q_proj.weight",
                f"{prefix}-self_attention_local-value-kernel": f"model.layers.{local_layer_idx}.self_attn.v_proj.weight",
            }
        )
    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Returns the hook functions dictionary for Gemma2 parameter transformation.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.
      saving_to_hf: Boolean indicating direction (True for MT->HF, False for HF->MT).

    Returns:
      A dictionary mapping keys to hook functions.
    """
    nlayers = hf_config["num_hidden_layers"]

    def pad_hf_embedding_layer(input_tensor, target_shape):
      """Pads/unpads and scales the embedding layer."""
      normalizer = np.dtype("float32").type(hf_config["hidden_size"] ** 0.5)

      if saving_to_hf:
        target_tensor = input_tensor[: target_shape[0], : target_shape[1]]
        target_tensor = target_tensor / normalizer
        target_tensor = target_tensor.astype(input_tensor.dtype)
        return target_tensor
      else:
        target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
        target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
        target_tensor = target_tensor * normalizer
        target_tensor = target_tensor.astype(input_tensor.dtype)
        return target_tensor

    def reshape_kernel(input_tensor, target_shape):
      if saving_to_hf:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).T
      else:
        return input_tensor.T.reshape(target_shape)

    def scale_rmsnorm_layer(input_tensor, target_shape):
      if saving_to_hf:
        return (input_tensor - 1.0).reshape(target_shape)
      else:
        return (input_tensor + 1.0).reshape(target_shape)

    def scale_query_layer(input_tensor, target_shape):
      if saving_to_hf:
        depth_scale = np.dtype("float32").type(np.sqrt(hf_config["head_dim"]))
        return (input_tensor * depth_scale).astype(input_tensor.dtype)
      else:
        depth_scale = np.dtype("float32").type(1 / np.sqrt(hf_config["head_dim"]))
        return (input_tensor * depth_scale).astype(input_tensor.dtype)

    # hook order does not affect result
    query_hook_chain = [reshape_kernel, scale_query_layer]

    mapping = {
        "params-token_embedder-embedding": pad_hf_embedding_layer,
        "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
    }

    if scan_layers:
      mapping.update(
          {
              "params-decoder-layers-self_attention_global-query-kernel": query_hook_chain,
              "params-decoder-layers-self_attention_local-query-kernel": query_hook_chain,
              "params-decoder-layers-self_attention_global-key-kernel": reshape_kernel,
              "params-decoder-layers-self_attention_local-key-kernel": reshape_kernel,
              "params-decoder-layers-self_attention_global-value-kernel": reshape_kernel,
              "params-decoder-layers-self_attention_local-value-kernel": reshape_kernel,
              "params-decoder-layers-mlp_global-wo-kernel": reshape_kernel,
              "params-decoder-layers-mlp_global-wi_1-kernel": reshape_kernel,
              "params-decoder-layers-mlp_global-wi_0-kernel": reshape_kernel,
              "params-decoder-layers-self_attention_global-out-kernel": reshape_kernel,
              "params-decoder-layers-mlp_local-wo-kernel": reshape_kernel,
              "params-decoder-layers-mlp_local-wi_1-kernel": reshape_kernel,
              "params-decoder-layers-mlp_local-wi_0-kernel": reshape_kernel,
              "params-decoder-layers-self_attention_local-out-kernel": reshape_kernel,
              "params-decoder-layers-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
              "params-decoder-layers-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
              "params-decoder-layers-post_ffw_norm_global-scale": scale_rmsnorm_layer,
              "params-decoder-layers-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
              "params-decoder-layers-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
              "params-decoder-layers-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
              "params-decoder-layers-post_ffw_norm_local-scale": scale_rmsnorm_layer,
              "params-decoder-layers-pre_ffw_norm_local-scale": scale_rmsnorm_layer,
          }
      )
    else:
      for maxtext_layer_idx in range(nlayers // 2):
        mapping.update(
            {
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": query_hook_chain,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": query_hook_chain,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-key-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-value-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-value-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wo-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_1-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_0-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-out-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wo-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_1-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_0-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-out-kernel": reshape_kernel,
                f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_global-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_local-scale": scale_rmsnorm_layer,
                f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_local-scale": scale_rmsnorm_layer,
            }
        )
    return mapping


class Gemma3Mapper(ParamMapperStrategy):
  """Strategy for Gemma3 models (Multimodal)."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Returns the parameter mapping dictionary from MaxText to HuggingFace for Gemma3.

    Handles both text and vision components of the model.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.

    Returns:
      A dictionary mapping MaxText parameter keys to HuggingFace keys.
    """
    tcfg = hf_config["text_config"]
    vcfg = hf_config["vision_config"]
    num_dec = tcfg["num_hidden_layers"]
    num_vis = vcfg["num_hidden_layers"]

    mapping = {
        # Embedding & final norm
        "params-token_embedder-embedding": "model.language_model.embed_tokens.weight",
        "params-decoder-decoder_norm-scale": "model.language_model.norm.weight",
        # Vision embed & pos
        "params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-kernel": (
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight"
        ),
        "params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-bias": (
            "model.vision_tower.vision_model.embeddings.patch_embedding.bias"
        ),
        "params-vision_encoder-Gemma3VisionEncoderLayer_0-pos_embedding": (
            "model.vision_tower.vision_model.embeddings.position_embedding.weight"
        ),
        "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoder_norm-scale": (
            "model.vision_tower.vision_model.post_layernorm.weight"
        ),
        "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoder_norm-bias": (
            "model.vision_tower.vision_model.post_layernorm.bias"
        ),
        # Multi-modal projector
        "params-vision_encoder-VisionEmbedder_0-mm_input_projection-w": (
            "model.multi_modal_projector.mm_input_projection_weight"
        ),
        "params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale": (
            "model.multi_modal_projector.mm_soft_emb_norm.weight"
        ),
    }

    vision_params = [
        ("LayerNorm_0-scale", "layer_norm1.weight"),
        ("LayerNorm_0-bias", "layer_norm1.bias"),
        ("LayerNorm_1-scale", "layer_norm2.weight"),
        ("LayerNorm_1-bias", "layer_norm2.bias"),
        ("MultiHeadDotProductAttention_0-query-kernel", "self_attn.q_proj.weight"),
        ("MultiHeadDotProductAttention_0-query-bias", "self_attn.q_proj.bias"),
        ("MultiHeadDotProductAttention_0-key-kernel", "self_attn.k_proj.weight"),
        ("MultiHeadDotProductAttention_0-key-bias", "self_attn.k_proj.bias"),
        ("MultiHeadDotProductAttention_0-value-kernel", "self_attn.v_proj.weight"),
        ("MultiHeadDotProductAttention_0-value-bias", "self_attn.v_proj.bias"),
        ("MultiHeadDotProductAttention_0-out-kernel", "self_attn.out_proj.weight"),
        ("MultiHeadDotProductAttention_0-out-bias", "self_attn.out_proj.bias"),
        ("MlpBlockViT_0-Dense_0-kernel", "mlp.fc1.weight"),
        ("MlpBlockViT_0-Dense_0-bias", "mlp.fc1.bias"),
        ("MlpBlockViT_0-Dense_1-kernel", "mlp.fc2.weight"),
        ("MlpBlockViT_0-Dense_1-bias", "mlp.fc2.bias"),
    ]

    # Vision layers mapping
    if scan_layers:
      for i in range(num_vis):
        for mx, hf in vision_params:
          key = f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock-{mx}"
          mapping[key] = f"model.vision_tower.vision_model.encoder.layers.{i}.{hf}"
    else:
      for i in range(num_vis):
        for mx, hf in vision_params:
          key = f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{i}-{mx}"
          mapping[key] = f"model.vision_tower.vision_model.encoder.layers.{i}.{hf}"

    # Text decoder mapping
    text_params = [
        ("pre_self_attention_norm-scale", "input_layernorm.weight"),
        ("post_self_attention_norm-scale", "post_attention_layernorm.weight"),
        ("self_attention-query_norm-scale", "self_attn.q_norm.weight"),
        ("self_attention-key_norm-scale", "self_attn.k_norm.weight"),
        ("pre_ffw_norm-scale", "pre_feedforward_layernorm.weight"),
        ("post_ffw_norm-scale", "post_feedforward_layernorm.weight"),
        ("self_attention-query-kernel", "self_attn.q_proj.weight"),
        ("self_attention-key-kernel", "self_attn.k_proj.weight"),
        ("self_attention-value-kernel", "self_attn.v_proj.weight"),
        ("self_attention-out-kernel", "self_attn.o_proj.weight"),
        ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
        ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
        ("mlp-wo-kernel", "mlp.down_proj.weight"),
    ]

    if scan_layers:
      for mx, hf in text_params:
        key = f"params-decoder-layers-{mx}"
        mapping[key] = [f"model.language_model.layers.{i}.{hf}" for i in range(num_dec)]
    else:
      for i in range(num_dec):
        for mx, hf in text_params:
          key = f"params-decoder-layers_{i}-{mx}"
          mapping[key] = f"model.language_model.layers.{i}.{hf}"

    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Returns the hook functions dictionary for Gemma3 parameter transformation.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.
      saving_to_hf: Boolean indicating direction (True for MT->HF, False for HF->MT).

    Returns:
      A dictionary mapping keys to hook functions.
    """
    hooks = {}

    def pad_and_scale_embedding(input_tensor, target_shape):
      source_vocab_size, _ = input_tensor.shape
      target_vocab_size, target_hidden_size = target_shape

      # MaxText embedding = original_embedding * sqrt(hidden_size)
      # HF embedding = original_embedding (HF model forward pass applies scaling)
      # Note: config["hidden_size"] is the HF hidden size from the HF config object
      normalizer = np.dtype("bfloat16").type(hf_config["text_config"]["hidden_size"] ** 0.5)

      # Apply scaling first
      if saving_to_hf:  # MaxText to HF
        scaled_tensor = (input_tensor / normalizer).astype(input_tensor.dtype)
      else:  # HF to MaxText
        scaled_tensor = (input_tensor * normalizer).astype(input_tensor.dtype)

      # Handle padding/truncation
      if source_vocab_size > target_vocab_size:
        warnings.warn(
            f"source vocab={source_vocab_size} > target vocab={target_vocab_size}, truncate output layer for MaxText."
        )
        output_tensor = scaled_tensor[:target_vocab_size, :]
      elif source_vocab_size < target_vocab_size:
        warnings.warn(
            f"source vocab={source_vocab_size} < target vocab={target_vocab_size}, pad output layer for MaxText."
        )
        padding_shape = (target_vocab_size - source_vocab_size, target_hidden_size)
        # Use jnp.zeros for JAX arrays, np.zeros for numpy arrays
        padding = (
            jnp.zeros(padding_shape, dtype=scaled_tensor.dtype)
            if isinstance(scaled_tensor, jax.Array)
            else np.zeros(padding_shape, dtype=scaled_tensor.dtype)
        )
        output_tensor = (
            jnp.concatenate([scaled_tensor, padding], axis=0)
            if isinstance(scaled_tensor, jax.Array)
            else np.concatenate([scaled_tensor, padding], axis=0)
        )
      else:  # Vocab sizes match
        output_tensor = scaled_tensor

      return output_tensor

    def scale_rmsnorm(x, target_shape):
      # MaxText norm = HF norm +1; HF norm = MaxText norm -1
      if saving_to_hf:
        return (x - 1.0).reshape(target_shape)
      return (x + 1.0).reshape(target_shape)

    def reshape_kernel(x, target_shape):
      if saving_to_hf:
        flipped = np.flip(np.array(target_shape))
        return x.reshape(flipped).T
      else:
        return x.T.reshape(target_shape)

    def vis_bias(x, target_shape):
      if saving_to_hf:
        return x.flatten()
      else:
        return x.reshape(target_shape)

    def vision_patch(x, target_shape):
      if saving_to_hf:
        return x.transpose(3, 2, 0, 1)
      else:
        return x.transpose(2, 3, 1, 0)

    def pos_embed(x, target_shape):
      if saving_to_hf:
        return x.squeeze(0)
      return x[None, :, :]

    # ---Embedding & final norm---
    hooks["params-token_embedder-embedding"] = pad_and_scale_embedding
    hooks["params-decoder-decoder_norm-scale"] = scale_rmsnorm
    # [1, 4096, 1152]
    hooks["params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-kernel"] = vision_patch
    hooks["params-vision_encoder-Gemma3VisionEncoderLayer_0-pos_embedding"] = pos_embed

    hooks["params-vision_encoder-VisionEmbedder_0-mm_input_projection-w"] = lambda x, _: x
    hooks["params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale"] = scale_rmsnorm

    # Text layers
    tc = hf_config.get("text_config", {})
    nlayers = tc.get("num_hidden_layers", 0)
    layer_ids = [None] if scan_layers else list(range(nlayers))
    for i in layer_ids:
      pref = f"params-decoder-layers_{i}-" if i is not None else "params-decoder-layers-"
      # Attention Q/K/V/O
      hooks[f"{pref}self_attention-query-kernel"] = reshape_kernel
      hooks[f"{pref}self_attention-key-kernel"] = reshape_kernel
      hooks[f"{pref}self_attention-value-kernel"] = reshape_kernel
      hooks[f"{pref}self_attention-out-kernel"] = reshape_kernel
      # Norm scales
      for nm in (
          "pre_self_attention_norm-scale",
          "post_self_attention_norm-scale",
          "self_attention-query_norm-scale",
          "self_attention-key_norm-scale",
          "pre_ffw_norm-scale",
          "post_ffw_norm-scale",
      ):
        hooks[pref + nm] = scale_rmsnorm
      # MLP
      hooks[f"{pref}mlp-wi_0-kernel"] = reshape_kernel
      hooks[f"{pref}mlp-wi_1-kernel"] = reshape_kernel
      hooks[f"{pref}mlp-wo-kernel"] = reshape_kernel

    # Vision layers
    vc = hf_config.get("vision_config", {})
    nvis = vc.get("num_hidden_layers", 0)
    vision_layer_ids = [None] if scan_layers else list(range(nvis))
    for i in vision_layer_ids:
      base = (
          f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{i}-"
          if i is not None
          else "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock-"
      )
      # Attention kernels & biases
      for qkv in ["query", "key", "value"]:
        hooks[f"{base}MultiHeadDotProductAttention_0-{qkv}-kernel"] = reshape_kernel
        hooks[f"{base}MultiHeadDotProductAttention_0-{qkv}-bias"] = vis_bias
      # [1152, 1152] -> [16, 72, 1152]
      hooks[f"{base}MultiHeadDotProductAttention_0-out-kernel"] = reshape_kernel
      hooks[f"{base}MultiHeadDotProductAttention_0-out-bias"] = vis_bias
      # MLP ViT kernels & biases
      for dense in "Dense_0", "Dense_1":
        hooks[f"{base}MlpBlockViT_0-{dense}-kernel"] = reshape_kernel

    return hooks
