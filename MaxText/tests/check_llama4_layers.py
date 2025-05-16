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

""" Tests for Llama4 Vision RoPE """
import math
import torch
from torch import nn
import torch.nn.functional as F
import jax
import unittest
import jax.numpy as jnp
from MaxText.layers import embeddings, llama4
import numpy as np


"""  
Llama4 Vision RoPE 
Details https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
"""


### original Pytorch Reference implementation
def reshape_for_broadcast(freqs_ci: torch.Tensor, query: torch.Tensor):
  """Reshape the frequency tensor for broadcasting."""
  ndim = query.ndim
  shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(query.shape)]
  return freqs_ci.view(*shape)


def vision_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
):
  """Apply the rotary embedding to the query and key tensors."""
  query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
  key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
  freqs_ci = reshape_for_broadcast(freqs_ci=freqs_ci, query=query_)  # freqs_ci[:,:,None,:]
  freqs_ci = freqs_ci.to(query_.device)
  query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
  key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)
  return query_out.type_as(query), key_out.type_as(key)  # but this drops to 8e-3


class Llama4VisionRotaryEmbedding(nn.Module):
  """Llama4 Vision RoPE implementation."""

  def __init__(self, image_size, patch_size, hidden_size, num_attention_heads, rope_theta):
    super().__init__()
    idx = image_size // patch_size
    img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
    img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
    img_idx[-1, -1] = -2  # ID_CLS_TOKEN
    frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
    frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
    freq_dim = hidden_size // num_attention_heads // 2
    rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim))
    freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
    freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
    freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
    freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
    freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
    self.freqs_ci = freq_cis  # idx**2, idx**2, idx * 2

  def forward(self):
    return self.freqs_ci


### original Pytorch Reference implementation


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


class Llama4VisionRotaryEmbeddingTest(unittest.TestCase):
  """Test for the Llama4 Vision RoPE implementation."""

  def test_rope_multiple_seq(self):
    image_size = 336
    patch_size = 14
    hidden_size = 1408
    num_attention_heads = 16
    rope_theta = 10000
    seq_len = (image_size // patch_size) ** 2 + 1

    for batch_size in [10, 100, 1000]:
      with self.subTest(batch_size=batch_size):
        freqs_ci = Llama4VisionRotaryEmbedding(image_size, patch_size, hidden_size, num_attention_heads, rope_theta)
        freqs_ci = freqs_ci.forward()
        # Create random queries and keys
        q = torch.randn(batch_size, seq_len, num_attention_heads, hidden_size // num_attention_heads)
        k = torch.randn(batch_size, seq_len, num_attention_heads, hidden_size // num_attention_heads)

        q_rope_pt, k_rope_pt = vision_apply_rotary_emb(q, k, freqs_ci=freqs_ci)

        # # Create and initialize the JAX Llama4 Vision RoPE
        model_jax = embeddings.LlamaVisionRotaryEmbedding(
            image_size, patch_size, hidden_size, num_attention_heads, rope_theta
        )
        params = model_jax.init(jax.random.PRNGKey(0), to_jax(k))

        # Apply the JAX RoPE
        q_rope_jax = model_jax.apply(params, to_jax(q))
        k_rope_jax = model_jax.apply(params, to_jax(k))

        # Compare outputs from the PyTorch and JAX implementations
        np.testing.assert_allclose(to_jax(q_rope_pt), q_rope_jax, rtol=1e-3, atol=0.05)
        np.testing.assert_allclose(to_jax(k_rope_pt), k_rope_jax, rtol=1e-3, atol=0.05)


class Llama4UnfoldConvolutionTest(unittest.TestCase):
  """Test for the Llama4 Unfold Convolution implementation."""

  def __copy_weights(self, pt_model, params):
    """Copy weights from PyTorch model to JAX model.

    Args:
      pt_model: PyTorch Llama4UnfoldConvolution model
      params: JAX model parameters
    """
    # Create new params with copied weights
    updated_params = jax.tree_util.tree_map(lambda x: x, params)
    updated_params["params"]["vit_unfold_linear"]["kernel"] = to_jax(pt_model.linear.weight).T
    return updated_params

  def test_unfold_convolution(self):
    """Test for the Llama4 Unfold Convolution implementation."""
    # Test parameters
    # following the llama4 config
    # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
    batch_size = 10
    num_channels = 3
    image_size = 336
    patch_size = 14
    hidden_size = 1408

    # Create random input tensor
    inputs_pt = torch.randn(batch_size, num_channels, image_size, image_size)

    # PyTorch implementation
    # following llama4 implementation in
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py#L1279
    class Llama4UnfoldConvolution(nn.Module):
      """Llama4 Unfold Convolution implementation."""

      def __init__(self):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        kernel_size = patch_size
        if isinstance(kernel_size, int):
          kernel_size = (kernel_size, kernel_size)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=patch_size)
        self.linear = nn.Linear(num_channels * kernel_size[0] * kernel_size[1], hidden_size, bias=False)

      def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # num_patches = (self.image_size // self.patch_size) ** 2
        # hidden_states shape: torch.Size([batch_size, num_channels, img, img])
        hidden_states = self.unfold(hidden_states)
        # hidden_states shape: torch.Size([batch_size, num_channels * patch_size * patch_size, num_patches])
        hidden_states = hidden_states.permute(0, 2, 1)
        # hidden_states shape: torch.Size([batch_size, num_patches, num_channels * patch_size * patch_size])
        hidden_states = self.linear(hidden_states)
        # hidden_states shape: torch.Size([batch_size, num_patches, hidden_size])
        return hidden_states

    # Initialize PyTorch model
    pt_model = Llama4UnfoldConvolution()
    pt_model.eval()
    pt_output = pt_model(inputs_pt)

    # JAX implementation
    class JaxConfig:

      def __init__(self):
        self.patch_size_for_vit = patch_size
        self.hidden_size_for_vit = hidden_size
        self.dtype_mm = jnp.float32

    # Initialize JAX model
    jax_model = llama4.Llama4UnfoldConvolution(JaxConfig())
    params = jax_model.init(jax.random.PRNGKey(0), to_jax(inputs_pt))

    # Copy weights from PyTorch to JAX
    pt_params = self.__copy_weights(pt_model, params)

    # Run JAX forward pass with updated params
    jax_output = jax_model.apply(pt_params, to_jax(inputs_pt))

    # Compare shapes
    self.assertEqual(pt_output.shape, jax_output.shape)

    # Compare outputs with reasonable tolerances
    np.testing.assert_allclose(to_jax(pt_output), jax_output, rtol=1e-3, atol=0.05)


class Llama4VisionPixelShuffleMLPTest(unittest.TestCase):
  """Test for the Llama4 Vision Pixel Shuffle MLP implementation."""

  def __copy_weights(self, pt_model, params):
    """Copy weights from PyTorch model to JAX model.

    Args:
      pt_model: PyTorch Llama4VisionPixelShuffleMLP model
      params: JAX model parameters
    """
    # Create new params with copied weights
    updated_params = jax.tree_util.tree_map(lambda x: x, params)
    # Copy weights for both MLP layers
    updated_params["params"]["pixel_shuffle_mlp"]["vit_pixel_shuffle_mlp_fc1"]["kernel"] = to_jax(pt_model.mlp.fc1.weight).T
    updated_params["params"]["pixel_shuffle_mlp"]["vit_pixel_shuffle_mlp_fc2"]["kernel"] = to_jax(pt_model.mlp.fc2.weight).T
    return updated_params

  def test_pixel_shuffle_mlp(self):
    """Test for the Llama4 Vision Pixel Shuffle MLP implementation."""
    # Test parameters
    # following config https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
    batch_size = 10
    num_patches = 24 * 24  # 336/14 = 24 patches per side
    hidden_size = 1408
    intermediate_size = 5632
    projector_input_dim = 4096
    projector_output_dim = 4096
    pixel_shuffle_ratio = 0.5
    projector_dropout = 0.0

    def pixel_shuffle(input_tensor, shuffle_ratio):
      # input_tensor: [batch_size, num_patches, channels]
      batch_size, num_patches, channels = input_tensor.shape
      patch_size = int(math.sqrt(num_patches))

      input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)
      batch_size, height, width, channels = input_tensor.size()

      reshaped_tensor = input_tensor.view(batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio))
      reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

      reshaped_tensor = reshaped_tensor.view(
          batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio), int(channels / (shuffle_ratio**2))
      )
      reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

      output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
      return output_tensor

    # PyTorch implementation
    class Llama4VisionMLP2(nn.Module):
      """Llama4 Vision MLP2 implementation."""

      def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = nn.Linear(self.intermediate_size, config.projector_input_dim, bias=False)
        self.fc2 = nn.Linear(config.projector_output_dim, config.projector_output_dim, bias=False)
        self.activation_fn = nn.GELU()
        self.dropout = config.projector_dropout

      def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return self.activation_fn(self.fc2(hidden_states))

    class Llama4VisionPixelShuffleMLP(nn.Module):
      """Llama4 Vision Pixel Shuffle MLP implementation."""

      def __init__(self, config):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(config.projector_input_dim // (self.pixel_shuffle_ratio**2))
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP2(config)

      def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        # encoded_patches shape: torch.Size([batch_size, num_patches, hidden_size])
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)
        # result shape: torch.Size([batch_size, num_patches * (pixel_shuffle_rate**2), projector_output_dim])

    # Initialize PyTorch model
    class Config:

      def __init__(self):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projector_input_dim = projector_input_dim
        self.projector_output_dim = projector_output_dim
        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self.projector_dropout = projector_dropout

    # Create random input tensor
    inputs_pt = torch.randn(batch_size, num_patches, hidden_size)

    pt_model = Llama4VisionPixelShuffleMLP(Config())
    pt_model.eval()
    pt_output = pt_model(inputs_pt)

    # JAX implementation
    class JaxConfig:

      def __init__(self):
        self.pixel_shuffle_ratio_for_vit = pixel_shuffle_ratio
        self.projector_input_dim_for_vit = projector_input_dim
        self.projector_output_dim_for_vit = projector_output_dim
        self.dtype_mm = jnp.float32
        self.projector_dropout_for_vit = projector_dropout

    # Initialize JAX model
    jax_model = llama4.Llama4VisionPixelShuffleMLP(JaxConfig())
    params = jax_model.init(jax.random.PRNGKey(0), to_jax(inputs_pt))

    # Copy weights from PyTorch to JAX
    pt_params = self.__copy_weights(pt_model, params)

    # Run JAX forward pass with updated params
    jax_output = jax_model.apply(pt_params, to_jax(inputs_pt), deterministic=True)

    # Compare shapes
    self.assertEqual(pt_output.shape, jax_output.shape)

    # Compare outputs with reasonable tolerances
    np.testing.assert_allclose(to_jax(pt_output), jax_output, rtol=1e-3, atol=0.05)


if __name__ == "__main__":
  unittest.main()
