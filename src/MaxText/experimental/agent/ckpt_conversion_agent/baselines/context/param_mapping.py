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

"""Example code for agent, only gemma2 mappings"""

import numpy as np


def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers=False):
  """Returns mapping between MaxText and HuggingFace Gemma2 weight paths.

  Args:
    config (dict): Model configuration dictionary containing at least
      'num_hidden_layers'.
    scan_layers (bool, optional): Whether the MaxText model uses layer
      scanning optimization. When True, decoder layers are stacked into a
      single tensor. Defaults to False.

  Returns:
    dict: A mapping where keys are MaxText parameter paths and values are
      either single strings (HF parameter path) for unscanned parameters or
      lists of strings (HF parameter paths) for stacked layers when
      `scan_layers=True`.

  Notes:
    * MaxText uses a paired layer approach where two HF decoder layers are
      treated as one MaxText decoder layer.
    * MaxText layer `i` corresponds to HF layers `2i` and `2i+1`.
    * Local components map to even-numbered HF decoder layers (0, 2, 4...).
    * Global components map to odd-numbered HF decoder layers (1, 3, 5...).
  """

  nlayers = config["num_hidden_layers"]
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
  }
  if scan_layers:
    mapping = {
        **mapping,
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
  # Case 2: scan_layer=False
  else:
    for maxtext_layer_idx in range(0, nlayers // 2):
      local_layer_idx = maxtext_layer_idx * 2
      global_layer_idx = maxtext_layer_idx * 2 + 1
      # pylint: disable=line-too-long
      layer_mapping = {
          f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale": f"model.layers.{global_layer_idx}.input_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wo-kernel": f"model.layers.{global_layer_idx}.mlp.down_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_1-kernel": f"model.layers.{global_layer_idx}.mlp.up_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_0-kernel": f"model.layers.{global_layer_idx}.mlp.gate_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_global-scale": f"model.layers.{global_layer_idx}.post_attention_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_global-scale": f"model.layers.{global_layer_idx}.post_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_global-scale": f"model.layers.{global_layer_idx}.pre_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": f"model.layers.{global_layer_idx}.self_attn.k_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-out-kernel": f"model.layers.{global_layer_idx}.self_attn.o_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": f"model.layers.{global_layer_idx}.self_attn.q_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-value-kernel": f"model.layers.{global_layer_idx}.self_attn.v_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_local-scale": f"model.layers.{local_layer_idx}.input_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wo-kernel": f"model.layers.{local_layer_idx}.mlp.down_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_1-kernel": f"model.layers.{local_layer_idx}.mlp.up_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_0-kernel": f"model.layers.{local_layer_idx}.mlp.gate_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_local-scale": f"model.layers.{local_layer_idx}.post_attention_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_local-scale": f"model.layers.{local_layer_idx}.post_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_local-scale": f"model.layers.{local_layer_idx}.pre_feedforward_layernorm.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-key-kernel": f"model.layers.{local_layer_idx}.self_attn.k_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-out-kernel": f"model.layers.{local_layer_idx}.self_attn.o_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": f"model.layers.{local_layer_idx}.self_attn.q_proj.weight",
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-value-kernel": f"model.layers.{local_layer_idx}.self_attn.v_proj.weight",
      }
      mapping = {**mapping, **layer_mapping}
  return mapping


def GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Gemma2 conversion.

  This function generates a mapping of transformation functions that handle the
  necessary conversions between MaxText and HuggingFace parameter formats for
  Gemma2, including operations like padding, reshaping, and scaling.

  Args:
    config (dict): Model configuration dictionary that must contain:

      * num_hidden_layers (int): Number of layers in the model.
      * head_dim (int): Dimension of attention heads.
      * hidden_size (int): Model's hidden dimension size.
    scan_layers (bool, optional): Controls the output format for layer
      parameters. True for batched, False for individual. Defaults to False.
    saving_to_hf (bool, optional): Determines the direction of transformation.
      True for MaxText to HuggingFace, False for the reverse. Defaults to
      False.

  Returns:
    dict: A mapping from MaxText parameter names to transformation functions.
      The value can be a single function or a list of functions to be
      applied sequentially.
  """
  nlayers = config["num_hidden_layers"]

  def pad_hf_embedding_layer(input_tensor, target_shape):
    """Pads/unpads and scales the embedding layer.

    Note:
        HF embedding weights shape =  [256000, d_model]
        MaxText embedding weights shape = [256128, d_model]
        MaxText pads Gemma2 embedding to 256128 for better performance.
    """
    # TODO(wenxindongwork), Perhaps, this dtype should be the activation dtype
    normalizer = np.dtype("float32").type(config["hidden_size"] ** 0.5)

    def to_hf():
      target_tensor = input_tensor[: target_shape[0], : target_shape[1]]
      target_tensor = target_tensor / normalizer
      target_tensor = target_tensor.astype(input_tensor.dtype)
      return target_tensor

    def from_hf():
      target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
      target_tensor = target_tensor * normalizer
      target_tensor = target_tensor.astype(input_tensor.dtype)
      return target_tensor

    if saving_to_hf:
      return to_hf()
    else:
      return from_hf()

  def reshape_kernel(input_tensor, target_shape):
    def to_hf():
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T

    def from_hf():
      return input_tensor.T.reshape(target_shape)

    if saving_to_hf:
      return to_hf()
    else:
      return from_hf()

  def scale_rmsnorm_layer(input_tensor, target_shape):
    def to_hf():
      return (input_tensor - 1.0).reshape(target_shape)

    def from_hf():
      return (input_tensor + 1.0).reshape(target_shape)

    if saving_to_hf:
      return to_hf()
    else:
      return from_hf()

  def scale_query_layer(input_tensor, target_shape):
    def to_hf():
      depth_scale = np.dtype("float32").type(np.sqrt(config["head_dim"]))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)

    def from_hf():
      depth_scale = np.dtype("float32").type(1 / np.sqrt(config["head_dim"]))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)

    if saving_to_hf:
      return to_hf()
    else:
      return from_hf()

  mapping = {
      "params-token_embedder-embedding": pad_hf_embedding_layer,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }
  if scan_layers:
    mapping = {
        **mapping,
        "params-decoder-layers-self_attention_global-query-kernel": [
            reshape_kernel,
            scale_query_layer,
        ],
        "params-decoder-layers-self_attention_local-query-kernel": [
            reshape_kernel,
            scale_query_layer,
        ],
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
  else:
    for maxtext_layer_idx in range(nlayers // 2):
      mapping = {
          **mapping,
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": [
              reshape_kernel,
              scale_query_layer,
          ],
          f"params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": [
              reshape_kernel,
              scale_query_layer,
          ],
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
  return mapping
