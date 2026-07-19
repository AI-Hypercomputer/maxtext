# Copyright 2025 Google LLC
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

"""Qwen3 ckpt conversation agent ground truth hook functions."""

import numpy as np


def QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Qwen3.

  This function provides a dictionary of transformation functions (hooks) for
  converting Qwen3 model parameters between MaxText and Hugging Face formats.
  It handles embedding padding and kernel reshaping.

  Args:
    config (dict): Model configuration dictionary, including
      'num_hidden_layers' and optionally 'num_experts'.
    scan_layers (bool, optional): Whether the model uses scanned layers.
      Defaults to False.
    saving_to_hf (bool, optional): The direction of conversion. True for
      MaxText to Hugging Face, False for the reverse. Defaults to False.

  Returns:
    dict: A dictionary mapping MaxText parameter names to their corresponding
      transformation functions.
  """
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
