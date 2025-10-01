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

""" Tests for Llama4 Vision RoPE """
from typing import Callable, NamedTuple
import os.path
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
import jax
import unittest
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.common_types import MODEL_MODE_TRAIN, AttentionType
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.layers import attentions, embeddings, llama4
import numpy as np

Attention = attentions.Attention

# pylint: disable=line-too-long, missing-function-docstring

"""
Llama4 Vision RoPE
Details https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
"""


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().numpy())


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

        # Apply the JAX RoPE
        q_rope_jax = model_jax(to_jax(q))
        k_rope_jax = model_jax(to_jax(k))

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
    matmul_precision = "float32"

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
        self.dtype_mm = jnp.float32
        self.hidden_size_for_vit = hidden_size
        self.image_size_for_vit = 896
        self.num_channels_for_vit = 3
        self.patch_size_for_vit = patch_size
        self.per_device_batch_size = batch_size
        self.matmul_precision = matmul_precision

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
    updated_params["params"]["pixel_shuffle_mlp"]["vit_pixel_shuffle_mlp_fc1"]["kernel"] = to_jax(
        pt_model.mlp.fc1.weight
    ).T
    updated_params["params"]["pixel_shuffle_mlp"]["vit_pixel_shuffle_mlp_fc2"]["kernel"] = to_jax(
        pt_model.mlp.fc2.weight
    ).T
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
    matmul_precision = "float32"

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
        self.dtype_mm = jnp.float32
        self.intermediate_size_for_vit = 5632
        self.projector_dropout_for_vit = projector_dropout
        self.projector_input_dim_for_vit = projector_input_dim
        self.projector_output_dim_for_vit = projector_output_dim
        self.pixel_shuffle_ratio_for_vit = pixel_shuffle_ratio
        self.matmul_precision = matmul_precision

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


class Llama4MultiModalProjectorTest(unittest.TestCase):
  """Test for the Llama4 Multi Modal Projector implementation.
  # Test parameters follows config https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
  """

  def __copy_weights(self, pt_model, params):
    """Copy weights from PyTorch model to JAX model.

    Args:
      pt_model: PyTorch Llama4MultiModalProjector model
      params: JAX model parameters
    """
    # Create new params with copied weights
    updated_params = jax.tree_util.tree_map(lambda x: x, params)
    updated_params["params"]["vit_multi_modal_projector"]["kernel"] = to_jax(pt_model.linear_1.weight).T
    return updated_params

  def test_multi_modal_projector(self):
    """Test for the Llama4 Multi Modal Projector implementation."""
    # Test parameters
    # following config https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
    batch_size = 10
    num_patches = 24 * 24  # 336/14 = 24 patches per side
    pixel_shuffle_ratio = 0.5
    vision_output_dim = 4096
    hidden_size = 5120
    dtype_mm = jnp.float32
    matmul_precision = "float32"
    mesh_shape_1d = (len(jax.devices()),)
    mesh_axes = ["data"]
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), mesh_axes)

    # PyTorch implementation
    class VisionConfig:

      def __init__(self):
        self.vision_output_dim = vision_output_dim

    class TextConfig:

      def __init__(self):
        self.hidden_size = hidden_size

    class Config:

      def __init__(self):
        self.vision_config = VisionConfig()
        self.text_config = TextConfig()

    class Llama4MultiModalProjector(nn.Module):
      """Llama4 Multi Modal Projector pytorch original implementation."""

      def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=False,
        )

      def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        return hidden_states

    # Create random input tensor
    # Shape: [batch_size*num_patches*(pixel_shuffle_ratio**2), vision_output_dim]
    inputs = torch.randn(batch_size, num_patches, int(pixel_shuffle_ratio**2), vision_output_dim)
    inputs_pt = inputs.reshape(batch_size * num_patches * int(pixel_shuffle_ratio**2), vision_output_dim)

    # Initialize PyTorch model
    pt_model = Llama4MultiModalProjector(Config())
    pt_model.eval()
    pt_output = pt_model(inputs_pt)

    # JAX implementation
    class JaxConfig:

      def __init__(self):
        self.dtype_mm = dtype_mm
        self.matmul_precision = matmul_precision
        self.vision_output_dim_for_vit = vision_output_dim
        self.base_emb_dim = hidden_size

    # Initialize JAX model
    jax_model = llama4.Llama4MultiModalProjector(JaxConfig(), mesh)
    params = jax_model.init(jax.random.PRNGKey(0), to_jax(inputs))

    # Copy weights from PyTorch to JAX
    pt_params = self.__copy_weights(pt_model, params)

    # Run JAX forward pass with updated params
    jax_output = jax_model.apply(pt_params, to_jax(inputs))
    jax_output = jax_output.reshape(batch_size * num_patches * int(pixel_shuffle_ratio**2), hidden_size)

    # Compare shapes
    self.assertEqual(pt_output.shape, jax_output.shape)

    # Compare outputs with reasonable tolerances
    np.testing.assert_allclose(to_jax(pt_output), jax_output, rtol=1e-3, atol=0.05)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  Pytorch implementation from HuggingFace:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
  This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: None | torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
  """
  Pytorch implementation from HuggingFace:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
  """
  key_states = repeat_kv(key, module.num_key_value_groups)
  value_states = repeat_kv(value, module.num_key_value_groups)
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
  if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

  attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(query.dtype)
  attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()

  return attn_output, attn_weights


class Llama4VisionAttention(nn.Module):
  """
  Pytorch implementation from HuggingFace:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
  """

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = config.hidden_size // config.num_attention_heads
    self.num_key_value_groups = 1
    self.attention_dropout = config.attention_dropout
    seed_value = 4
    torch.manual_seed(seed_value)
    self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

  def forward(
      self,
      hidden_states: torch.Tensor,
      freqs_ci: torch.Tensor,
      attention_mask: None | torch.Tensor = None,
      past_key_value: None | torch.Tensor = None,
      **kwargs,
  ) -> tuple[torch.Tensor, None | torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    query_states, key_states = vision_apply_rotary_emb(query_states, key_states, freqs_ci=freqs_ci)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Comment the following block and only test eager_attention_forward in this test
    # # flex disable because breaks on TP 8, embed is 88 not power of 2
    # if self.config._attn_implementation not in ["eager", "flex_attention"]:
    #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
    #         logger.warning_once(
    #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
    #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
    #         )
    #     else:
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attention_interface: Callable = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        None,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=None,
        is_causal=False,  # HAS TO BE ENFORCED
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class Llama4VisionAttentionTest(unittest.TestCase):
  """Test for the Llama4 vision attention."""

  def _copy_weights_bias(self, pt_model, params, hidden_size=256, num_head=2):
    """Copy weights from PyTorch model to JAX model.

    Args:
      pt_model: PyTorch Llama4VisionAttention model
      params: JAX model parameters
    """
    head_dim = hidden_size // num_head
    updated_params = jax.tree_util.tree_map(lambda x: x, params)
    # update weights
    updated_params["params"]["query"]["kernel"] = to_jax(pt_model.q_proj.weight.T.view(hidden_size, num_head, head_dim))
    updated_params["params"]["key"]["kernel"] = to_jax(pt_model.k_proj.weight.T.view(hidden_size, num_head, head_dim))
    updated_params["params"]["value"]["kernel"] = to_jax(pt_model.v_proj.weight.T.view(hidden_size, num_head, head_dim))
    updated_params["params"]["out"]["kernel"] = to_jax(pt_model.o_proj.weight.T.view(num_head, head_dim, hidden_size))
    # update bias
    updated_params["params"]["query"]["bias"] = to_jax(pt_model.q_proj.bias.view(num_head, head_dim))
    updated_params["params"]["key"]["bias"] = to_jax(pt_model.k_proj.bias.view(num_head, head_dim))
    updated_params["params"]["value"]["bias"] = to_jax(pt_model.v_proj.bias.view(num_head, head_dim))
    updated_params["params"]["out"]["bias"] = to_jax(pt_model.o_proj.bias)
    return updated_params

  class Config(NamedTuple):
    num_attention_heads: int
    hidden_size: int
    attention_dropout: int = 0

  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "model_name": "llama4-17b-16e",
      "scan_layers": False,
      "dtype_mm": "float32",
      "float32_qk_product": True,
      "float32_logits": True,
  }

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
    )
    self.rng = jax.random.PRNGKey(0)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.seq_len_for_vit = (self.cfg.image_size_for_vit // self.cfg.patch_size_for_vit) ** 2 + 1

  def test_vision_attention(self):
    """Test for the Llama4 vision attention."""

    pt_config = self.Config(
        num_attention_heads=self.cfg.num_attention_heads_for_vit, hidden_size=self.cfg.hidden_size_for_vit
    )

    model_pt = Llama4VisionAttention(pt_config)
    hidden_states_pt = torch.rand(
        self.cfg.global_batch_size_to_load, self.seq_len_for_vit, self.cfg.hidden_size_for_vit, dtype=torch.float32
    )

    # Create proper freq_ci using the existing rotary embedding class
    freqs_ci_model = Llama4VisionRotaryEmbedding(
        self.cfg.image_size_for_vit,
        self.cfg.patch_size_for_vit,
        self.cfg.hidden_size_for_vit,
        self.cfg.num_attention_heads_for_vit,
        self.cfg.rope_theta_for_vit,
    )
    freqs_ci = freqs_ci_model.forward()
    attn_output_pt, _ = model_pt(hidden_states_pt, freqs_ci=freqs_ci)

    lnx = to_jax(hidden_states_pt)
    attention_layer = attentions.attention_as_linen(
        config=self.cfg,
        num_query_heads=self.cfg.num_attention_heads_for_vit,
        num_kv_heads=self.cfg.num_attention_heads_for_vit,
        head_dim=self.cfg.hidden_size_for_vit // self.cfg.num_attention_heads_for_vit,
        max_target_length=self.seq_len_for_vit,
        attention_kernel="dot_product",  # TODO aireenmei: support flash attention
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        mesh=self.mesh,
        dropout_rate=0,
        name="self_attention_vision",
        attention_type=AttentionType.FULL,
        is_nope_layer=True, #False,
        use_bias_in_projections=True,
        is_vision=True,
        use_qk_norm=False,
        query_pre_attn_scalar=1 / math.sqrt(self.cfg.hidden_size_for_vit // self.cfg.num_attention_heads_for_vit),
        model_mode=MODEL_MODE_TRAIN,
    )

    key = jax.random.PRNGKey(0)
    attention_layer_params = attention_layer.init(
        key,
        lnx,
        lnx,
    )
    params_from_pt = self._copy_weights_bias(
        model_pt, attention_layer_params, self.cfg.hidden_size_for_vit, self.cfg.num_attention_heads_for_vit
    )
    attn_ouput_jax = attention_layer.apply(
        params_from_pt,
        lnx,
        lnx,
        deterministic=True,
    )
    np.testing.assert_allclose(attn_ouput_jax, to_jax(attn_output_pt), rtol=1e-3, atol=0.05)


### PyTorch Reference Implementations


class Llama4VisionMLP(nn.Module):
  """
  Pytorch implementation from HuggingFace of Llama4 Vision MLP.
  """

  def __init__(self, config):
    super().__init__()
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
    self.activation_fn = nn.GELU()

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states


class Llama4VisionEncoderLayer(nn.Module):
  """
  Pytorch implementation from HuggingFace of Llama4 Vision Encoder Layer.
  """

  def __init__(self, config):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = Llama4VisionAttention(config)
    self.mlp = Llama4VisionMLP(config)
    self.input_layernorm = nn.LayerNorm(config.hidden_size)
    self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

  def forward(
      self,
      hidden_state: torch.Tensor,
      freqs_ci: torch.Tensor,
      attention_mask: None | torch.Tensor = None,
      output_attentions: None | bool = None,
  ):
    # Self Attention
    residual = hidden_state
    hidden_state = self.input_layernorm(hidden_state)
    hidden_state, attn_weights = self.self_attn(
        hidden_state,
        freqs_ci=freqs_ci,
        attention_mask=attention_mask,
    )
    hidden_state = residual + hidden_state

    # Feed forward
    residual = hidden_state
    hidden_state = self.post_attention_layernorm(hidden_state)
    hidden_state = self.mlp(hidden_state)
    hidden_state = residual + hidden_state

    outputs = (hidden_state,)
    if output_attentions:
      outputs += (attn_weights,)
    return outputs


class Llama4VisionEncoder(nn.Module):
  """
  Pytorch implementation from HuggingFace of Llama4 Vision Encoder.
  """

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList([Llama4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(
      self,
      hidden_states: torch.Tensor,
      freqs_ci: torch.Tensor,
      attention_mask: None | torch.Tensor = None,
      output_attentions: None | bool = None,
      output_hidden_states: None | bool = None,
      return_dict: None | bool = None,
  ):
    # all_hidden_states = () if output_hidden_states else None
    # all_attentions = () if output_attentions else None

    for encoder_layer in self.layers:
      layer_outputs = encoder_layer(
          hidden_state=hidden_states,
          attention_mask=attention_mask,
          output_attentions=output_attentions,
          freqs_ci=freqs_ci,
      )

      # if output_attentions:
      #   all_attentions = all_attentions + (layer_outputs[1],)

      hidden_states = layer_outputs[0]

    # if output_hidden_states:
    #   all_hidden_states = all_hidden_states + (hidden_states,)

    return hidden_states


class Llama4VisionEncoderTest(unittest.TestCase):
  """Test for the Llama4 vision encoder."""

  def _copy_weights(self, pt_model, params, hidden_size=256, num_head=2):
    """Copy weights from PyTorch model to JAX model.

    Args:
      pt_model: PyTorch Llama4VisionEncoder model
      params: JAX model parameters
    """
    head_dim = hidden_size // num_head
    updated_params = jax.tree_util.tree_map(lambda x: x, params)

    # Copy weights for each encoder layer
    for i, _ in enumerate(pt_model.layers):
      # Copy attention weights
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["query"]["kernel"] = to_jax(
          pt_model.layers[i].self_attn.q_proj.weight.T.view(hidden_size, num_head, head_dim)
      )
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["key"]["kernel"] = to_jax(
          pt_model.layers[i].self_attn.k_proj.weight.T.view(hidden_size, num_head, head_dim)
      )
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["value"]["kernel"] = to_jax(
          pt_model.layers[i].self_attn.v_proj.weight.T.view(hidden_size, num_head, head_dim)
      )
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["out"]["kernel"] = to_jax(
          pt_model.layers[i].self_attn.o_proj.weight.T.view(num_head, head_dim, hidden_size)
      )

      # Copy attention biases
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["query"]["bias"] = to_jax(
          pt_model.layers[i].self_attn.q_proj.bias.view(num_head, head_dim)
      )
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["key"]["bias"] = to_jax(
          pt_model.layers[i].self_attn.k_proj.bias.view(num_head, head_dim)
      )
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["value"]["bias"] = to_jax(
          pt_model.layers[i].self_attn.v_proj.bias.view(num_head, head_dim)
      )
      updated_params["params"][f"layers_{i}"]["self_attention_vision"]["out"]["bias"] = to_jax(
          pt_model.layers[i].self_attn.o_proj.bias
      )

      # Copy MLP weights
      updated_params["params"][f"layers_{i}"]["Llama4VisionMLP_0"]["vit_encoder_layer_mlp_fc1"]["kernel"] = to_jax(
          pt_model.layers[i].mlp.fc1.weight.T
      )
      updated_params["params"][f"layers_{i}"]["Llama4VisionMLP_0"]["vit_encoder_layer_mlp_fc2"]["kernel"] = to_jax(
          pt_model.layers[i].mlp.fc2.weight.T
      )

      # Copy MLP biases
      updated_params["params"][f"layers_{i}"]["Llama4VisionMLP_0"]["vit_encoder_layer_mlp_fc1"]["bias"] = to_jax(
          pt_model.layers[i].mlp.fc1.bias
      )
      updated_params["params"][f"layers_{i}"]["Llama4VisionMLP_0"]["vit_encoder_layer_mlp_fc2"]["bias"] = to_jax(
          pt_model.layers[i].mlp.fc2.bias
      )

      # Copy layer norm weights
      updated_params["params"][f"layers_{i}"]["input_layer_norm"]["scale"] = to_jax(
          pt_model.layers[i].input_layernorm.weight
      )
      updated_params["params"][f"layers_{i}"]["input_layer_norm"]["bias"] = to_jax(
          pt_model.layers[i].input_layernorm.bias
      )
      updated_params["params"][f"layers_{i}"]["post_attention_layer_norm"]["scale"] = to_jax(
          pt_model.layers[i].post_attention_layernorm.weight
      )
      updated_params["params"][f"layers_{i}"]["post_attention_layer_norm"]["bias"] = to_jax(
          pt_model.layers[i].post_attention_layernorm.bias
      )

    return updated_params

  class Config(NamedTuple):
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    attention_dropout: int = 0

  # Llama4 has 34 ViT layers, but test currently passes with a maximum of 31 layers
  config_arguments = {
      "run_name": "test",
      "enable_checkpointing": False,
      "model_name": "llama4-17b-16e",
      "scan_layers": False,
      "num_hidden_layers_for_vit": 34,
      "dtype": "float32",
      "matmul_precision": "float32",
      "float32_qk_product": True,
      "float32_logits": True,
  }

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
    )
    self.rng = jax.random.PRNGKey(0)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.seq_len_for_vit = (self.cfg.image_size_for_vit // self.cfg.patch_size_for_vit) ** 2 + 1

  def test_vision_encoder(self):
    """Test the full vision encoder implementation."""

    # PyTorch config using the loaded configuration values
    pt_config = self.Config(
        num_attention_heads=self.cfg.num_attention_heads_for_vit,
        hidden_size=self.cfg.hidden_size_for_vit,
        intermediate_size=self.cfg.intermediate_size_for_vit,
        num_hidden_layers=self.cfg.num_hidden_layers_for_vit,
        attention_dropout=0,
    )

    # Create PyTorch model
    pt_model = Llama4VisionEncoder(pt_config)
    pt_model.eval()

    # Create JAX model
    jax_model = llama4.Llama4VisionEncoder(config=self.cfg, mesh=self.mesh)

    # Create test input using config dimensions
    batch_size = 4
    # Generate random numbers in (-1, 1)
    key = jax.random.PRNGKey(0)
    inputs = jax.random.uniform(
      key,
      (batch_size, self.seq_len_for_vit, self.cfg.hidden_size_for_vit),
      minval=-0.1,
      maxval=0.1,
      dtype=jnp.float32,
    )
    #inputs = np.load("/home/aireenmei_google_com/golden/llama4_pre_vision_encoder_image_only_randomw_mx.npy")
    print(f"Input shape: {inputs.shape}")
    print(f"Input mean: {np.mean(inputs)}")
    print(f"Input: {inputs}")

    # Initialize JAX parameters
    params = jax_model.init(self.rng, inputs, deterministic=True)

    # Copy weights from PyTorch to JAX
    params = self._copy_weights(pt_model, params, self.cfg.hidden_size_for_vit, self.cfg.num_attention_heads_for_vit)

    freqs_ci_model = Llama4VisionRotaryEmbedding(
        self.cfg.image_size_for_vit,
        self.cfg.patch_size_for_vit,
        self.cfg.hidden_size_for_vit,
        self.cfg.num_attention_heads_for_vit,
        self.cfg.rope_theta_for_vit,
    )
    freqs_ci = freqs_ci_model.forward()

    # Forward pass through PyTorch model
    pt_inputs = torch.from_numpy(np.array(inputs)).to(torch.float32)
    pt_outputs = pt_model(hidden_states=pt_inputs, freqs_ci=freqs_ci)

    # Forward pass through JAX model
    jax_outputs = jax_model.apply(params, inputs, deterministic=True)
    print(f"pt mean: {torch.mean(pt_outputs)}")
    print(f"{pt_outputs=}")
    print(f"jax mean: {np.mean(jax_outputs)}")
    print(f"{jax_outputs=}")  
    # Compare outputs
    diff = np.abs(jax_outputs - to_jax(pt_outputs))
    actual_atol = np.max(diff)
    eps = 1e-8
    actual_rtol = np.max(diff / (np.maximum(np.abs(to_jax(pt_outputs)), eps)))
    print(f"Actual atol: {actual_atol}")
    print(f"Actual rtol: {actual_rtol}")
    np.testing.assert_allclose(jax_outputs, to_jax(pt_outputs), rtol=0.05, atol=1e-3)


if __name__ == "__main__":
  unittest.main()
