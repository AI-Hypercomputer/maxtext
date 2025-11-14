# Copyright 2025 Google LLC
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

"""Shared utilities for MaxText tests."""

import jax.numpy as jnp
import numpy as np
import torch


def create_random_jax_torch(*shape, dtype=np.float32):
  """Create random array and return both JAX and PyTorch versions.

  Args:
      *shape: Shape of the array
      dtype: NumPy dtype (default: np.float32)

  Returns:
      tuple: (jax_array, torch_tensor)
  """
  np_array = np.random.randn(*shape).astype(dtype)
  return jnp.array(np_array), torch.from_numpy(np_array)


def split_into_patches(x, temporal_patch_size, patch_size):
  """Split a 5D tensor into patches for PyTorch vision encoder input.

  Converts from full image format (batch, channels, temporal, height, width) to
  patch format (num_patches, channels, temporal_patch_size, patch_size, patch_size).

  Returns:
      Tensor of shape (num_patches, channels, temporal_patch_size, patch_size, patch_size)
      where num_patches = (temporal//temporal_patch_size) * (height//patch_size) * (width//patch_size)
  """
  B, C, T, H, W = x.shape
  assert T % temporal_patch_size == 0, f"Temporal dimension {T} must be divisible by {temporal_patch_size}"
  assert H % patch_size == 0, f"Height {H} must be divisible by {patch_size}"
  assert W % patch_size == 0, f"Width {W} must be divisible by {patch_size}"

  x = x.reshape(B, C, T, H // patch_size, patch_size, W // patch_size, patch_size)
  x = x.permute(0, 3, 5, 1, 2, 4, 6)  # (B, H//patch_size, W//patch_size, C, T, patch_size, patch_size)
  return x.reshape(-1, C, T, patch_size, patch_size)


def assert_all_close_jax_torch(jax_tensor, torch_tensor, rtol, atol, error_msg=""):
  """Compare JAX and PyTorch tensors for numerical closeness.

  Args:
      jax_tensor: JAX array to compare
      torch_tensor: PyTorch tensor to compare
      rtol: Relative tolerance
      atol: Absolute tolerance
      error_msg: Optional error message prefix
  """
  np.testing.assert_allclose(
      torch_tensor.numpy(),
      np.array(jax_tensor),
      rtol=rtol,
      atol=atol,
      err_msg=error_msg,
  )


def copy_linear_weights(torch_linear, jax_linear):
  """Copy weights from PyTorch Linear to JAX nnx.Linear."""
  jax_linear.kernel.value = jnp.array(torch_linear.weight.detach().cpu().numpy().T)
  if torch_linear.bias is not None and jax_linear.bias is not None:
    jax_linear.bias.value = jnp.array(torch_linear.bias.detach().cpu().numpy())


def copy_layernorm_weights(torch_ln, jax_ln):
  """Copy weights from PyTorch LayerNorm to JAX nnx.LayerNorm."""
  jax_ln.scale.value = jnp.array(torch_ln.weight.detach().cpu().numpy())
  jax_ln.bias.value = jnp.array(torch_ln.bias.detach().cpu().numpy())


def copy_conv2d_weights(torch_conv, jax_conv):
  """Copy weights from PyTorch Conv2d to JAX nnx.Conv."""
  # PyTorch: (out_channels, in_channels, kH, kW)
  # JAX: (kH, kW, in_channels, out_channels)
  torch_weight = torch_conv.weight.detach().cpu().numpy()
  jax_weight = np.transpose(torch_weight, (2, 3, 1, 0))
  jax_conv.kernel.value = jnp.array(jax_weight)
  jax_conv.bias.value = jnp.array(torch_conv.bias.detach().cpu().numpy())


def copy_densegeneral_qkv_weights(torch_linear, jax_densegeneral, num_heads, head_dim):
  """Copy weights from PyTorch Linear to JAX DenseGeneral for Q/K/V projections.

  PyTorch Linear has weight shape (out_features, in_features)
  JAX DenseGeneral has kernel shape (in_features, num_heads, head_dim)
  """
  # Get PyTorch weight: (out_features, in_features) where out_features = num_heads * head_dim
  torch_weight = torch_linear.weight.detach().cpu().numpy()  # (out_features, in_features)

  # Transpose and reshape: (in_features, out_features) -> (in_features, num_heads, head_dim)
  torch_weight_t = torch_weight.T  # (in_features, out_features)
  jax_weight = torch_weight_t.reshape(-1, num_heads, head_dim)  # (in_features, num_heads, head_dim)

  jax_densegeneral.kernel.value = jnp.array(jax_weight)
  if torch_linear.bias is not None and jax_densegeneral.bias is not None:
    # Bias shape for DenseGeneral: (num_heads, head_dim)
    torch_bias = torch_linear.bias.detach().cpu().numpy()
    jax_bias = torch_bias.reshape(num_heads, head_dim)
    jax_densegeneral.bias.value = jnp.array(jax_bias)


def copy_densegeneral_out_weights(torch_linear, jax_densegeneral, num_heads, head_dim, output_dim):
  """Copy weights from PyTorch Linear to JAX DenseGeneral for output projection.

  PyTorch Linear has weight shape (output_dim, input_dim) where input_dim = num_heads * head_dim
  JAX DenseGeneral has kernel shape (num_heads, head_dim, output_dim)
  """
  # Get PyTorch weight: (output_dim, num_heads * head_dim)
  torch_weight = torch_linear.weight.detach().cpu().numpy()

  # Transpose: (num_heads * head_dim, output_dim)
  torch_weight_t = torch_weight.T

  # Reshape: (num_heads, head_dim, output_dim)
  jax_weight = torch_weight_t.reshape(num_heads, head_dim, output_dim)

  jax_densegeneral.kernel.value = jnp.array(jax_weight)
  if torch_linear.bias is not None and jax_densegeneral.bias is not None:
    jax_densegeneral.bias.value = jnp.array(torch_linear.bias.detach().cpu().numpy())


def copy_attention_weights(torch_attn, jax_attn):
  """Copy attention layer weights from PyTorch to JAX."""
  copy_linear_weights(torch_attn.q_proj, jax_attn.q_proj)
  copy_linear_weights(torch_attn.k_proj, jax_attn.k_proj)
  copy_linear_weights(torch_attn.v_proj, jax_attn.v_proj)
  copy_linear_weights(torch_attn.out_proj, jax_attn.out_proj)


def copy_attention_weights_to_maxtext(torch_attn, maxtext_attn, fused_qkv=False):
  """Copy attention weights from PyTorch to MaxText's Attention module.

  Args:
      torch_attn: PyTorch attention with either:
          - Separate q_proj, k_proj, v_proj, out_proj (fused_qkv=False, for audio)
          - Fused qkv and proj (fused_qkv=True, for vision)
      maxtext_attn: MaxText Attention module with separate q/k/v projections
      fused_qkv: If True, torch_attn has fused qkv projection that needs splitting
  """
  if not hasattr(maxtext_attn, "query"):
    raise NotImplementedError("Unsupported MaxText Attention structure")

  num_heads = maxtext_attn.num_query_heads
  head_dim = maxtext_attn.head_dim
  hidden_size = num_heads * head_dim
  output_dim = hidden_size

  # Extract Q/K/V weights and biases
  if fused_qkv:
    # Vision: Split fused QKV projection
    qkv_weight = torch_attn.qkv.weight.detach().cpu().numpy()
    qkv_bias = torch_attn.qkv.bias.detach().cpu().numpy()

    q_weight = qkv_weight[:hidden_size, :]
    k_weight = qkv_weight[hidden_size : 2 * hidden_size, :]
    v_weight = qkv_weight[2 * hidden_size :, :]
    q_bias = qkv_bias[:hidden_size]
    k_bias = qkv_bias[hidden_size : 2 * hidden_size]
    v_bias = qkv_bias[2 * hidden_size :]

    out_proj = torch_attn.proj
  else:
    # Audio: Extract from separate projections
    q_weight = torch_attn.q_proj.weight.detach().cpu().numpy()
    k_weight = torch_attn.k_proj.weight.detach().cpu().numpy()
    v_weight = torch_attn.v_proj.weight.detach().cpu().numpy()
    q_bias = torch_attn.q_proj.bias.detach().cpu().numpy()
    k_bias = torch_attn.k_proj.bias.detach().cpu().numpy()
    v_bias = torch_attn.v_proj.bias.detach().cpu().numpy()

    out_proj = torch_attn.out_proj

  # Copy Q/K/V weights (common logic for both)
  maxtext_attn.query.kernel.value = jnp.array(q_weight.T.reshape(hidden_size, num_heads, head_dim))
  maxtext_attn.query.bias.value = jnp.array(q_bias.reshape(num_heads, head_dim))

  maxtext_attn.key.kernel.value = jnp.array(k_weight.T.reshape(hidden_size, num_heads, head_dim))
  maxtext_attn.key.bias.value = jnp.array(k_bias.reshape(num_heads, head_dim))

  maxtext_attn.value.kernel.value = jnp.array(v_weight.T.reshape(hidden_size, num_heads, head_dim))
  maxtext_attn.value.bias.value = jnp.array(v_bias.reshape(num_heads, head_dim))

  # Copy output projection (common logic for both)
  out_weight = out_proj.weight.detach().cpu().numpy()
  out_bias = out_proj.bias.detach().cpu().numpy()
  maxtext_attn.out.kernel.value = jnp.array(out_weight.T.reshape(num_heads, head_dim, output_dim))
  maxtext_attn.out.bias.value = jnp.array(out_bias)


def copy_encoder_layer_weights(torch_layer, jax_layer):
  """Copy encoder layer weights from PyTorch to JAX."""
  copy_attention_weights(torch_layer.self_attn, jax_layer.self_attn)
  copy_linear_weights(torch_layer.fc1, jax_layer.fc1)
  copy_linear_weights(torch_layer.fc2, jax_layer.fc2)
  copy_layernorm_weights(torch_layer.self_attn_layer_norm, jax_layer.self_attn_layer_norm)
  copy_layernorm_weights(torch_layer.final_layer_norm, jax_layer.final_layer_norm)


def copy_encoder_weights(torch_encoder, jax_encoder):
  """Copy full encoder weights from PyTorch to JAX."""
  # Copy convolutional layers
  copy_conv2d_weights(torch_encoder.conv2d1, jax_encoder.conv2d1)
  copy_conv2d_weights(torch_encoder.conv2d2, jax_encoder.conv2d2)
  copy_conv2d_weights(torch_encoder.conv2d3, jax_encoder.conv2d3)

  # Copy linear projections
  copy_linear_weights(torch_encoder.conv_out, jax_encoder.conv_out)
  copy_linear_weights(torch_encoder.proj1, jax_encoder.proj1)
  copy_linear_weights(torch_encoder.proj2, jax_encoder.proj2)

  # Copy layer norm
  copy_layernorm_weights(torch_encoder.ln_post, jax_encoder.ln_post)

  # Copy positional embeddings
  jax_encoder.positional_embedding.positional_embedding.value = jnp.array(
      torch_encoder.positional_embedding.positional_embedding.detach().cpu().numpy()
  )

  # Copy encoder layers
  for torch_layer, jax_layer in zip(torch_encoder.layers, jax_encoder.layers):
    copy_encoder_layer_weights(torch_layer, jax_layer)


def copy_maxtext_encoder_layer_weights(torch_layer, maxtext_layer):
  """Copy encoder layer weights from PyTorch to MaxText AudioEncoderLayer.

  Args:
      torch_layer: PyTorch TorchQwen3OmniMoeAudioEncoderLayer
      maxtext_layer: MaxText AudioEncoderLayer
  """
  # Copy layer norms
  copy_layernorm_weights(torch_layer.self_attn_layer_norm, maxtext_layer.input_layer_norm)
  copy_layernorm_weights(torch_layer.final_layer_norm, maxtext_layer.post_attention_layer_norm)

  # Copy attention weights to MaxText Attention module
  copy_attention_weights_to_maxtext(torch_layer.self_attn, maxtext_layer.self_attention_audio)

  copy_linear_weights(torch_layer.fc1, maxtext_layer.AudioMLP.wi)
  copy_linear_weights(torch_layer.fc2, maxtext_layer.AudioMLP.wo)


def copy_audio_model(torch_model, maxtext_model, config):
  """Copy full AudioModel weights from PyTorch to MaxText.

  Args:
      torch_model: PyTorch TorchQwen3OmniMoeAudioEncoder
      maxtext_model: MaxText AudioModel
      config: MaxText config with encoder_layers
  """
  copy_conv2d_weights(torch_model.conv2d1, maxtext_model.conv2d1)
  copy_conv2d_weights(torch_model.conv2d2, maxtext_model.conv2d2)
  copy_conv2d_weights(torch_model.conv2d3, maxtext_model.conv2d3)
  copy_linear_weights(torch_model.conv_out, maxtext_model.conv_out)

  maxtext_model.positional_embedding.positional_embedding.value = jnp.array(
      torch_model.positional_embedding.positional_embedding.detach().cpu().numpy()
  )

  copy_layernorm_weights(torch_model.ln_post, maxtext_model.layernorm_post)

  for torch_layer, maxtext_layer in zip(
      torch_model.layers,
      [getattr(maxtext_model.audio_encoder, f"layers_{i}") for i in range(config.encoder_layers_for_audio)],
  ):
    copy_maxtext_encoder_layer_weights(torch_layer, maxtext_layer)

  copy_linear_weights(torch_model.proj1, maxtext_model.audio_projector.proj1)
  copy_linear_weights(torch_model.proj2, maxtext_model.audio_projector.proj2)


def copy_maxtext_encoder_weights(torch_encoder, maxtext_encoder):
  # Copy weights for each encoder layer
  for torch_layer, maxtext_layer in zip(
      torch_encoder.layers,
      [getattr(maxtext_encoder, f"layers_{i}") for i in range(len(torch_encoder.layers))],
  ):
    copy_maxtext_encoder_layer_weights(torch_layer, maxtext_layer)


# Vision-specific weight copying utilities
def copy_conv3d_weights(torch_conv, jax_conv):
  """Copy weights from PyTorch Conv3d to JAX nnx.Conv (3D)."""
  # PyTorch Conv3d: (out_channels, in_channels, kD, kH, kW)
  # JAX Conv (3D): (kD, kH, kW, in_channels, out_channels)
  torch_weight = torch_conv.weight.detach().cpu().numpy()
  jax_weight = np.transpose(torch_weight, (2, 3, 4, 1, 0))
  jax_conv.kernel.value = jnp.array(jax_weight)
  jax_conv.bias.value = jnp.array(torch_conv.bias.detach().cpu().numpy())


def copy_patch_embed_weights(torch_embed, jax_embed):
  """Copy patch embed weights from PyTorch to JAX."""
  copy_conv3d_weights(torch_embed.proj, jax_embed.proj)


def copy_mlp_weights(torch_mlp, jax_mlp):
  """Copy MLP weights from PyTorch to JAX."""
  copy_linear_weights(torch_mlp.linear_fc1, jax_mlp.linear_fc1)
  copy_linear_weights(torch_mlp.linear_fc2, jax_mlp.linear_fc2)


def copy_patch_merger_weights(torch_merger, jax_merger):
  """Copy patch merger weights from PyTorch to JAX."""
  copy_layernorm_weights(torch_merger.ln_q, jax_merger.ln_q)
  copy_linear_weights(torch_merger.mlp[0], jax_merger.mlp_0)
  copy_linear_weights(torch_merger.mlp[2], jax_merger.mlp_2)


def copy_vision_encoder_weights(torch_encoder, jax_encoder):
  """Copy all weights from PyTorch vision encoder to JAX vision encoder.

  Args:
      torch_encoder: PyTorch Qwen3OmniMoeVisionEncoder
      jax_encoder: JAX Qwen3OmniMoeVisionEncoder
  """
  # Copy patch embedding
  copy_patch_embed_weights(torch_encoder.patch_embed, jax_encoder.patch_embed)

  # Copy positional embedding weights
  torch_pos_embed = torch_encoder.pos_embed.weight.detach().cpu().numpy()
  jax_encoder.pos_embed_interpolate.pos_embed.value = jnp.array(torch_pos_embed)

  # Copy encoder blocks
  for torch_block, jax_block in zip(torch_encoder.blocks, jax_encoder.blocks):
    # Copy layer norms
    copy_layernorm_weights(torch_block.norm1, jax_block.ln1)
    copy_layernorm_weights(torch_block.norm2, jax_block.ln2)

    # Copy attention weights (vision uses fused QKV)
    copy_attention_weights_to_maxtext(torch_block.attn, jax_block.attn.attn, fused_qkv=True)

    # Copy MLP weights (vision MLP uses DenseGeneral)
    copy_linear_weights(torch_block.mlp.linear_fc1, jax_block.mlp)
    copy_linear_weights(torch_block.mlp.linear_fc2, jax_block.mlp_out)

  # Copy merger weights (deep mergers only, final_merger is now in projector)
  for torch_merger, jax_merger in zip(torch_encoder.merger_list, jax_encoder.merger_list):
    copy_patch_merger_weights(torch_merger, jax_merger)


# Audio-specific utilities
def create_block_diagonal_attention_mask(cu_seqlens, dtype):
  """Create block-diagonal attention mask from cumulative sequence lengths.

  PyTorch's eager attention implementation doesn't automatically respect cu_seqlens boundaries.
  This function creates an explicit block-diagonal mask that prevents attention across
  different sequences in the batch.

  Args:
      cu_seqlens: Cumulative sequence lengths, e.g., [0, 12, 24] for 2 sequences of length 12
      dtype: Data type for the attention mask

  Returns:
      Attention mask of shape (1, 1, total_seq_len, total_seq_len) where:
      - 0.0 means "can attend" (within same sequence)
      - finfo(dtype).min means "cannot attend" (across sequences)

  Example:
      >>> cu_seqlens = torch.tensor([0, 12, 24], dtype=torch.int32)
      >>> mask = create_block_diagonal_attention_mask(cu_seqlens, torch.float32)
      >>> # Positions 0-11 can only attend to 0-11
      >>> # Positions 12-23 can only attend to 12-23
  """
  total_seq_len = cu_seqlens[-1].item()
  attention_mask = torch.full(
      [1, 1, total_seq_len, total_seq_len],
      torch.finfo(dtype).min,
      device=cu_seqlens.device,
      dtype=dtype,
  )

  # Create blocks: allow attention within each sequence boundary
  for i in range(1, len(cu_seqlens)):
    start = cu_seqlens[i - 1].item()
    end = cu_seqlens[i].item()
    attention_mask[..., start:end, start:end] = 0

  return attention_mask
