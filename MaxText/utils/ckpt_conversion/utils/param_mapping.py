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

import numpy as np
import jax
import jax.numpy as jnp


def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers=False):
  """Returns mapping between MaxText and HuggingFace Gemma2 weight paths.

  Args:
      config (dict): Model configuration dictionary containing at least 'num_hidden_layers'.
      scan_layers (bool, optional): Whether the MaxText model uses layer scanning optimization.
          When True, decoder layers are stacked into a single tensor [dim1, #layers, dim2].
          Defaults to False.

  Returns:
      dict: A mapping where:
          - Keys are MaxText parameter paths
          - Values are either:
              - Single strings (HF parameter path) for unscanned parameters
              - Lists of strings (HF parameter paths) for stacked layers when scan_layers=True

  Notes:
      - MaxText uses a paired layer approach where two HF decoder layers are treated as
          one MaxText decoder layer.
      - MaxText layer i corresponds to HF layers 2i and 2i+1
      - Local components map to even-numbered HF decoder layers (0, 2, 4...)
      - Global components map to odd-numbered HF decoder layers (1, 3, 5...)
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
  """Creates parameter transformation functions for converting between MaxText and
  HuggingFace formats.

  This function generates a mapping of transformation functions that handle the necessary
  conversions between MaxText and HuggingFace parameter formats, including operations like
  padding, reshaping, and scaling.

  Args:
      config (dict): Model configuration dictionary that must contain:
          - num_hidden_layers (int): Number of layers in the model
          - head_dim (int): Dimension of attention heads
          - hidden_size (int): Model's hidden dimension size

      scan_layers (bool, optional): Controls the output format for layer parameters:
          - True: Returns transformation functions for batched layer parameters
          - False: Returns transformation functions for individual layer parameters
          Defaults to False.

      saving_to_hf (bool, optional): Determines the direction of transformation:
          - True: MaxText → HuggingFace conversion
          - False: HuggingFace → MaxText conversion
          Defaults to False.

  Returns:
      dict: Parameter transformation mapping where:
          - Keys: MaxText parameter names (str)
          - Values: Either:
              - callable: Single transformation function
              - list[callable]: List of transformation functions to be applied in sequence

  Transformation Details:
      The function handles several types of parameter transformations:
      1. Embedding layer padding:
          - HF shape: [256000, d_model]
          - MaxText shape: [256128, d_model] (padded for performance)
      2. Layer normalization scaling:
          - Adds/subtracts 1.0 depending on direction
      3. Attention query scaling:
          - Scales by sqrt(head_dim) or its inverse

      4. Kernel reshaping:
          - Handles dimension transposition and reshaping between formats
  """
  nlayers = config["num_hidden_layers"]

  def pad_hf_embedding_layer(input_tensor, target_shape):
    """
    Note:
        HF embedding weights shape =  [256000,d_model]
        MaxText embedding weights shape = [256128,d_model]
        MaxText pad Gemma2 embedding to 256128 for better performance.
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
        f"params-decoder-layers-self_attention_global-query-kernel": [
            reshape_kernel,
            scale_query_layer,
        ],
        f"params-decoder-layers-self_attention_local-query-kernel": [
            reshape_kernel,
            scale_query_layer,
        ],
        f"params-decoder-layers-self_attention_global-key-kernel": reshape_kernel,
        f"params-decoder-layers-self_attention_local-key-kernel": reshape_kernel,
        f"params-decoder-layers-self_attention_global-value-kernel": reshape_kernel,
        f"params-decoder-layers-self_attention_local-value-kernel": reshape_kernel,
        f"params-decoder-layers-mlp_global-wo-kernel": reshape_kernel,
        f"params-decoder-layers-mlp_global-wi_1-kernel": reshape_kernel,
        f"params-decoder-layers-mlp_global-wi_0-kernel": reshape_kernel,
        f"params-decoder-layers-self_attention_global-out-kernel": reshape_kernel,
        f"params-decoder-layers-mlp_local-wo-kernel": reshape_kernel,
        f"params-decoder-layers-mlp_local-wi_1-kernel": reshape_kernel,
        f"params-decoder-layers-mlp_local-wi_0-kernel": reshape_kernel,
        f"params-decoder-layers-self_attention_local-out-kernel": reshape_kernel,
        f"params-decoder-layers-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-post_ffw_norm_global-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-post_ffw_norm_local-scale": scale_rmsnorm_layer,
        f"params-decoder-layers-pre_ffw_norm_local-scale": scale_rmsnorm_layer,
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


def GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers=False):
  """
  Returns a dictionary mapping from MaxText parameter names to
  HuggingFace Gemma3Text parameter names.

  Args:
      config (dict): HF model configuration dictionary containing:
          - num_hidden_layers (int): The number of decoder layers.
      scan_layers (bool, optional): If True, MaxText layers are 'stacked'
          into a single param. Defaults to False.

  Returns:
      dict: A mapping from MaxText parameter names to HF parameter names (str)
            or lists of names (if scan_layers=True).
  """
  n_layers = config["num_hidden_layers"]

  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
  }

  if scan_layers:
    mapping.update(
        {
            "params-decoder-layers-pre_self_attention_norm-scale": [
                f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query-kernel": [
                f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-key-kernel": [
                f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-value-kernel": [
                f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-out-kernel": [
                f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query_norm-scale": [
                f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)  # Assuming q_norm exists
            ],
            "params-decoder-layers-self_attention-key_norm-scale": [
                f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)  # Assuming k_norm exists
            ],
            "params-decoder-layers-post_self_attention_norm-scale": [
                f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-pre_ffw_norm-scale": [  # MaxText pre_ffw_norm maps to HF pre_feedforward_layernorm
                f"model.layers.{i}.pre_feedforward_layernorm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-mlp-wi_0-kernel": [  # gate_proj
                f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-mlp-wi_1-kernel": [  # up_proj
                f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-mlp-wo-kernel": [  # down_proj
                f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-post_ffw_norm-scale": [  # MaxText post_ffw_norm maps to HF post_feedforward_layernorm
                f"model.layers.{i}.post_feedforward_layernorm.weight" for i in range(n_layers)
            ],
        }
    )
  else:
    for i in range(n_layers):
      mapping.update(
          {
              f"params-decoder-layers_{i}-pre_self_attention_norm-scale": f"model.layers.{i}.input_layernorm.weight",
              f"params-decoder-layers_{i}-self_attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
              f"params-decoder-layers_{i}-self_attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
              f"params-decoder-layers_{i}-self_attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
              f"params-decoder-layers_{i}-self_attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
              f"params-decoder-layers_{i}-self_attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
              f"params-decoder-layers_{i}-self_attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
              f"params-decoder-layers_{i}-post_self_attention_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
              f"params-decoder-layers_{i}-pre_ffw_norm-scale": f"model.layers.{i}.pre_feedforward_layernorm.weight",
              f"params-decoder-layers_{i}-mlp-wi_0-kernel": f"model.layers.{i}.mlp.gate_proj.weight",
              f"params-decoder-layers_{i}-mlp-wi_1-kernel": f"model.layers.{i}.mlp.up_proj.weight",
              f"params-decoder-layers_{i}-mlp-wo-kernel": f"model.layers.{i}.mlp.down_proj.weight",
              f"params-decoder-layers_{i}-post_ffw_norm-scale": f"model.layers.{i}.post_feedforward_layernorm.weight",
          }
      )
  return mapping


def GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Gemma3Text."""

  nlayers = config["num_hidden_layers"]

  def pad_and_scale_embedding_gemma3(input_tensor, target_shape):
    """
    Handles potential vocabulary size mismatch and scaling for Gemma3 embedding.
    Assumes input_tensor is the source embedding, target_shape is the destination shape.
    """
    source_vocab_size, source_hidden_size = input_tensor.shape
    target_vocab_size, target_hidden_size = target_shape

    # MaxText embedding = original_embedding * sqrt(hidden_size)
    # HF embedding = original_embedding (HF model forward pass applies scaling)
    # Note: config["hidden_size"] is the HF hidden size from the HF config object
    normalizer = np.dtype("bfloat16").type(config["hidden_size"] ** 0.5)

    # Apply scaling first
    if saving_to_hf:  # MaxText to HF
      scaled_tensor = (input_tensor / normalizer).astype(input_tensor.dtype)
    else:  # HF to MaxText
      scaled_tensor = (input_tensor * normalizer).astype(input_tensor.dtype)

    # Handle padding/truncation
    if source_vocab_size > target_vocab_size:
      output_tensor = scaled_tensor[:target_vocab_size, :]
    elif source_vocab_size < target_vocab_size:
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

  def scale_rmsnorm_layer(input_tensor, target_shape):
    # MaxText norm_scale = hf_norm_scale + 1.0
    if saving_to_hf:  # MaxText to HF
      output_tensor = (input_tensor - 1.0).reshape(target_shape)
      return output_tensor
    else:  # HF to MaxText
      output_tensor = (input_tensor + 1.0).reshape(target_shape)
      return output_tensor

  def reshape_kernel(input_tensor, target_shape):
    # Used for O, MLP kernels (transpose-like behavior)
    # MaxText (A, B) -> HF (B, A) or MaxText (A,B,C) -> HF (C, A*B) etc.
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      output_tensor = input_tensor.reshape(flipped_target_shape).T
      return output_tensor
    else:
      output_tensor = input_tensor.T.reshape(target_shape)
      return output_tensor

  mapping = {
      "params-token_embedder-embedding": pad_and_scale_embedding_gemma3,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }

  layer_hooks = {
      "pre_self_attention_norm-scale": scale_rmsnorm_layer,
      "self_attention-query-kernel": reshape_kernel,
      "self_attention-key-kernel": reshape_kernel,
      "self_attention-value-kernel": reshape_kernel,
      "self_attention-out-kernel": reshape_kernel,  # O proj is like MLP kernel
      "self_attention-query_norm-scale": scale_rmsnorm_layer,
      "self_attention-key_norm-scale": scale_rmsnorm_layer,
      "post_self_attention_norm-scale": scale_rmsnorm_layer,
      "pre_ffw_norm-scale": scale_rmsnorm_layer,
      "mlp-wi_0-kernel": reshape_kernel,
      "mlp-wi_1-kernel": reshape_kernel,
      "mlp-wo-kernel": reshape_kernel,
      "post_ffw_norm-scale": scale_rmsnorm_layer,
  }

  if scan_layers:
    for key, hook in layer_hooks.items():
      mapping[f"params-decoder-layers-{key}"] = hook
  else:
    for i in range(nlayers):
      for key, hook in layer_hooks.items():
        mapping[f"params-decoder-layers_{i}-{key}"] = hook
  return mapping


def QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers=False):
  """
  Returns mapping from MaxText to HuggingFace Qwen3 weight paths.
  Handles both dense and MoE models.
  """
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", 0)

  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }

  if scan_layers:
    # Common Attention and Norms for both Dense and MoE
    mapping.update(
        {
            "params-decoder-layers-pre_self_attention_layer_norm-scale": [
                f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query-kernel": [
                f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-key-kernel": [
                f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-value-kernel": [
                f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-out-kernel": [
                f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-query_norm-scale": [
                f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-self_attention-key_norm-scale": [
                f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)
            ],
            "params-decoder-layers-post_self_attention_layer_norm-scale": [
                f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)
            ],
        }
    )

    if num_experts > 1:
      # MoE MLP layers
      mapping.update(
          {
              "params-decoder-layers-moe_block-gate-kernel": [f"model.layers.{i}.mlp.gate.weight" for i in range(n_layers)],
              # NOTE: The conversion script needs to handle unstacking on both layer and expert axes.
              "params-decoder-layers-moe_block-wi_0-kernel": [
                  [f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight" for j in range(num_experts)] for i in range(n_layers)
              ],
              "params-decoder-layers-moe_block-wi_1-kernel": [
                  [f"model.layers.{i}.mlp.experts.{j}.up_proj.weight" for j in range(num_experts)] for i in range(n_layers)
              ],
              "params-decoder-layers-moe_block-wo-kernel": [
                  [f"model.layers.{i}.mlp.experts.{j}.down_proj.weight" for j in range(num_experts)] for i in range(n_layers)
              ],
          }
      )
    else:
      # Dense MLP layers
      mapping.update(
          {
              "params-decoder-layers-mlp-wi_0-kernel": [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)],
              "params-decoder-layers-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)],
              "params-decoder-layers-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)],
          }
      )
  else:  # not scan_layers
    for i in range(n_layers):
      # Common Attention and Norms
      mapping.update(
          {
              f"params-decoder-layers_{i}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
              f"params-decoder-layers_{i}-self_attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
              f"params-decoder-layers_{i}-self_attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
              f"params-decoder-layers_{i}-self_attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
              f"params-decoder-layers_{i}-self_attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
              f"params-decoder-layers_{i}-self_attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
              f"params-decoder-layers_{i}-self_attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
              f"params-decoder-layers_{i}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
          }
      )
      if num_experts > 1:
        # MoE MLP layers
        # NOTE: The conversion script needs to handle splitting the expert tensor along axis 0.
        mapping.update(
            {
                f"params-decoder-layers_{i}-moe_block-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
                f"params-decoder-layers_{i}-moe_block-wi_0-kernel": [
                    f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight" for j in range(num_experts)
                ],
                f"params-decoder-layers_{i}-moe_block-wi_1-kernel": [
                    f"model.layers.{i}.mlp.experts.{j}.up_proj.weight" for j in range(num_experts)
                ],
                f"params-decoder-layers_{i}-moe_block-wo-kernel": [
                    f"model.layers.{i}.mlp.experts.{j}.down_proj.weight" for j in range(num_experts)
                ],
            }
        )
      else:
        # Dense MLP layers
        mapping.update(
            {
                f"params-decoder-layers_{i}-mlp-wi_0-kernel": f"model.layers.{i}.mlp.gate_proj.weight",
                f"params-decoder-layers_{i}-mlp-wi_1-kernel": f"model.layers.{i}.mlp.up_proj.weight",
                f"params-decoder-layers_{i}-mlp-wo-kernel": f"model.layers.{i}.mlp.down_proj.weight",
            }
        )

  return mapping


def QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Qwen3."""
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", 0)

  def pad_embedding_layer(input_tensor, target_shape):
    """Pads or truncates embedding layer to match target vocab size."""
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]

    if source_vocab_size == target_vocab_size:
      return input_tensor

    if saving_to_hf:  # MaxText to HF, truncate
      return input_tensor[:target_vocab_size, :]
    else:  # HF to MaxText, pad
      padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      padded_tensor[:source_vocab_size, :] = input_tensor
      return padded_tensor

  def reshape_kernel(input_tensor, target_shape):
    """Reshapes and transposes kernel weights between MaxText and HF."""
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  mapping = {
      "params-token_embedder-embedding": pad_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
  }

  kernel_hooks = [
      "self_attention-query-kernel",
      "self_attention-key-kernel",
      "self_attention-value-kernel",
      "self_attention-out-kernel",
      "mlp-wi_0-kernel",
      "mlp-wi_1-kernel",
      "mlp-wo-kernel",
  ]
  moe_kernel_hooks = [
      "moe_block-gate-kernel",
      "moe_block-wi_0-kernel",
      "moe_block-wi_1-kernel",
      "moe_block-wo-kernel",
  ]

  if scan_layers:
    for key in kernel_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_kernel
    if num_experts > 1:
      for key in moe_kernel_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_kernel
  else:
    for i in range(n_layers):
      for key in kernel_hooks:
        mapping[f"params-decoder-layers_{i}-{key}"] = reshape_kernel
      if num_experts > 1:
        for key in moe_kernel_hooks:
          mapping[f"params-decoder-layers_{i}-{key}"] = reshape_kernel
  return mapping


PARAM_MAPPING = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-4b": GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-12b": GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-27b": GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-0.6b": QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b": QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b": QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b": QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING,
}

HOOK_FNS = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-4b": GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-12b": GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-27b": GEMMA3TEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-0.6b": QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b": QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b": QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b": QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
}
