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


def GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """
  Hook functions for Gemma3 parameter conversion between MaxText and Hugging Face formats.
  Handles embedding padding/scaling, RMSNorm scaling, kernel reshaping, and vision-specific transforms.
  """
  hooks = {}

  # ---- Embedding pad & scale ----
  def pad_and_scale_embedding(input_tensor, target_shape):
    source_vocab_size, source_hidden_size = input_tensor.shape
    target_vocab_size, target_hidden_size = target_shape

    # MaxText embedding = original_embedding * sqrt(hidden_size)
    # HF embedding = original_embedding (HF model forward pass applies scaling)
    # Note: config["hidden_size"] is the HF hidden size from the HF config object
    normalizer = np.dtype("bfloat16").type(config["text_config"]["hidden_size"] ** 0.5)

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

  # ---- RMSNorm scale ----
  def scale_rmsnorm(x, target_shape):
    # MaxText norm = HF norm +1; HF norm = MaxText norm -1
    if saving_to_hf:
      return (x - 1.0).reshape(target_shape)
    return (x + 1.0).reshape(target_shape)

  # ---- Generic reshape ----
  def reshape_kernel(x, target_shape):
    if saving_to_hf:
      flipped = np.flip(np.array(target_shape))
      return x.reshape(flipped).T
    else:
      return x.T.reshape(target_shape)

  # ---- Vision reshape ----
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

  hooks["params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale"] = scale_rmsnorm

  # Text layers
  tc = config.get("text_config", {})
  nlayers = tc.get("num_hidden_layers", 0)
  layer_ids = [None] if scan_layers else list(range(nlayers))
  for i in layer_ids:
    pref = f"params-decoder-layers_{i}-" if i is not None else "params-decoder-layers-"
    # Attention Q/K/V/O
    hooks[pref + "self_attention-query-kernel"] = reshape_kernel
    hooks[pref + "self_attention-key-kernel"] = reshape_kernel
    hooks[pref + "self_attention-value-kernel"] = reshape_kernel
    hooks[pref + "self_attention-out-kernel"] = reshape_kernel
    # Norm scales
    for nm in [
        "pre_self_attention_norm-scale",
        "post_self_attention_norm-scale",
        "self_attention-query_norm-scale",
        "self_attention-key_norm-scale",
        "pre_ffw_norm-scale",
        "post_ffw_norm-scale",
    ]:
      hooks[pref + nm] = scale_rmsnorm
    # MLP
    hooks[pref + "mlp-wi_0-kernel"] = reshape_kernel
    hooks[pref + "mlp-wi_1-kernel"] = reshape_kernel
    hooks[pref + "mlp-wo-kernel"] = reshape_kernel

  # Vision layers
  vc = config.get("vision_config", {})
  nvis = vc.get("num_hidden_layers", 0)
  vision_layer_ids = list(range(nvis))
  for i in vision_layer_ids:
    base = (
        f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{i}-"
        if i is not None
        else "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock-"
    )
    # Attention kernels & biases
    for qkv in ["query", "key", "value"]:
      hooks[base + f"MultiHeadDotProductAttention_0-{qkv}-kernel"] = reshape_kernel
      hooks[base + f"MultiHeadDotProductAttention_0-{qkv}-bias"] = vis_bias
    # [1152, 1152] -> [16, 72, 1152]
    hooks[base + "MultiHeadDotProductAttention_0-out-kernel"] = reshape_kernel
    hooks[base + "MultiHeadDotProductAttention_0-out-bias"] = vis_bias
    # MLP ViT kernels & biases
    for dense in ["Dense_0", "Dense_1"]:
      hooks[base + f"MlpBlockViT_0-{dense}-kernel"] = reshape_kernel

  return hooks
