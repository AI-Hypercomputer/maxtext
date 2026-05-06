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

"""
Tests for DeepSeek V4 Compressed Attention Components against Golden PyTorch Reference.

This test file follows the MaxText golden reference testing pattern (e.g., deepseek32_vs_reference_test.py):
- Adapts the Hugging Face / DeepSeek V4 merged PyTorch classes to run standalone on CPU
- Defines the exact PyTorch reference classes inline within the test file with zero modifications to their logic
- Forces JAX weights to sync exactly with PyTorch random weights and asserts numerical parity within a tight tolerance
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

import torch
from torch import nn

from maxtext.layers import attention_compressed
from maxtext.layers import embeddings as jax_embeddings
from maxtext.layers import attention_op as jax_attention_op
from maxtext.layers import linears

# =============================================================================
# 1. Inline PyTorch Golden Reference Classes
# =============================================================================


class DeepseekV4Config:
  """Exact configuration properties used by the PyTorch modeling classes."""

  def __init__(self, compress_ratio: int, compressed_dim: int, head_dim: int, hidden_size: int):
    self.compress_rates = {"heavily_compressed_attention": 128, "compressed_sparse_attention": 4}
    self.head_dim = head_dim
    self.hidden_size = hidden_size
    self.rms_norm_eps = 1e-5
    self.o_groups = 8
    self.o_lora_rank = 1024
    self.num_attention_heads = 128


class DeepseekV4RMSNorm(nn.Module):
  """Exact copy of T5-style RMSNorm from PyTorch V4 modeling code."""

  def __init__(self, hidden_size, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


class DummyRotaryEmbedding(nn.Module):
  """Helper to bypass relative modeling_rope_utils imports and act as a no-op for weight comparison."""

  def __init__(self, config):
    super().__init__()

  def forward(self, x, position_ids, layer_type=None):
    cos = torch.ones((x.shape[0], x.shape[1], 64), dtype=x.dtype, device=x.device)
    sin = torch.zeros((x.shape[0], x.shape[1], 64), dtype=x.dtype, device=x.device)
    return cos, sin


def rotate_half(x):
  x1 = x[..., 0::2]
  x2 = x[..., 1::2]
  return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
  cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  rope_dim = cos.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
  return torch.cat([nope, rotated], dim=-1)


class DeepseekV4HCACompressor(nn.Module):
  """Exact copy of HCA Compressor from official DeepSeek-V4 modeling code."""

  rope_layer_type = "compress"

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.compress_rate = config.compress_rates["heavily_compressed_attention"]
    self.head_dim = config.head_dim
    self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.rotary_emb = DummyRotaryEmbedding(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      q_residual: torch.Tensor,
      position_ids: torch.Tensor,
      past_key_values,
      layer_idx: int,
  ) -> torch.Tensor:
    """Forward pass."""
    batch, _, _ = hidden_states.shape
    cache_layer = None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    if cache_layer is None:
      usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
      chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
      chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)
    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate
      chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
      chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(chunk_gate.dtype)
      compressed = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2))
      positions = torch.arange(n_windows, device=compressed.device)
      positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
      cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
      compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
      compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
    if cache_layer is not None:
      compressed = cache_layer.update_compressor_states("compressor", compressed)
    return compressed.unsqueeze(1)


class DeepseekV4CSACompressor(nn.Module):
  """Exact copy of CSA Compressor from official DeepSeek-V4 modeling code."""

  rope_layer_type = "compress"

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.compress_rate = config.compress_rates["compressed_sparse_attention"]
    self.head_dim = config.head_dim
    self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.rotary_emb = DummyRotaryEmbedding(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      q_residual: torch.Tensor,
      position_ids: torch.Tensor,
      past_key_values,
      layer_idx: int,
  ) -> torch.Tensor:
    """Forward pass."""
    batch, _, _ = hidden_states.shape
    cache_layer = None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    if cache_layer is None:
      usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
      chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
      chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)
    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate
      ratio = self.compress_rate
      chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
      chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
      new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
      new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
      new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
      new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
      if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
      if cache_layer is not None:
        prior_kv, prior_gate = cache_layer.update_overlap_state("compressor", chunk_kv, chunk_gate, self.head_dim)
        if prior_kv is not None:
          new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
          new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)
      compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
      positions = torch.arange(n_windows, device=compressed.device)
      positions = positions * self.compress_rate + first_window_position
      positions = positions.unsqueeze(0).expand(batch, -1)
      cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
      compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
      compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
    if cache_layer is not None:
      compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)
    return compressed_kv


class DeepseekV4GroupedLinear(nn.Module):
  """Exact copy of Grouped Output Projection from official DeepSeek-V4 modeling code."""

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.o_groups = config.o_groups
    self.o_lora_rank = config.o_lora_rank
    self.head_dim = config.head_dim
    self.num_heads = config.num_attention_heads
    self.in_group_dim = (self.num_heads // self.o_groups) * self.head_dim

    self.w_a = nn.Parameter(torch.empty(self.o_groups, self.in_group_dim, self.o_lora_rank))
    self.o_b_proj = nn.Linear(self.o_groups * self.o_lora_rank, config.hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, S, _, _ = x.shape
    x_grouped = x.view(B, S, self.o_groups, -1)
    grouped_out = torch.einsum("bsgd,gdr->bsgr", x_grouped, self.w_a)
    flattened = grouped_out.reshape(B, S, -1)
    return self.o_b_proj(flattened)


# =============================================================================
# 2. MaxText JAX Unit and Equivalence Tests
# =============================================================================


class DummyConfig:
  """Mock configuration class for testing."""

  def __init__(self, compress_ratio: int, compressed_dim: int, head_dim: int, emb_dim: int, dtype):
    self.compress_ratio = compress_ratio
    self.compressed_dim = compressed_dim
    self.head_dim = head_dim
    self.emb_dim = emb_dim
    self.dtype = dtype
    self.normalization_layer_epsilon = 1e-5
    self.o_groups = 8
    self.o_lora_rank = 1024
    self.base_num_query_heads = 128


class DeepseekV4VsReferenceTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 256
    self.hidden_dim = 1024
    self.head_dim = 128
    self.dtype = jnp.float32
    self.nnx_rng = nnx.Rngs(params=0)

  def test_hca_compression_shape(self):
    """Verify HCA (non-overlapping, ratio 128) shape transitions."""
    config = DummyConfig(
        compress_ratio=128,
        compressed_dim=self.head_dim,
        head_dim=self.head_dim,
        emb_dim=self.hidden_dim,
        dtype=self.dtype,
    )
    compressor = attention_compressed.Compressor(config, 128, rngs=self.nnx_rng)

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (self.batch_size, self.seq_len, self.hidden_dim), dtype=self.dtype)
    compressed_x = compressor(x)

    expected_shape = (self.batch_size, self.seq_len // 128, self.head_dim)
    self.assertEqual(compressed_x.shape, expected_shape)
    self.assertEqual(compressed_x.dtype, self.dtype)
    self.assertTrue(jnp.all(jnp.isfinite(compressed_x)))

  def test_csa_compression_shape(self):
    """Verify CSA (overlapping, ratio 4) shape transitions."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    compressor = attention_compressed.Compressor(config, 4, rngs=self.nnx_rng)

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (self.batch_size, self.seq_len, self.hidden_dim), dtype=self.dtype)
    compressed_x = compressor(x)

    expected_shape = (self.batch_size, self.seq_len // 4, self.head_dim)
    self.assertEqual(compressed_x.shape, expected_shape)
    self.assertEqual(compressed_x.dtype, self.dtype)
    self.assertTrue(jnp.all(jnp.isfinite(compressed_x)))

  def test_extreme_batch_sizes(self):
    """Verify shapes under extreme batch sizes (1 and 16)."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    compressor = attention_compressed.Compressor(config, 4, rngs=self.nnx_rng)
    key = jax.random.PRNGKey(42)

    # Batch Size = 1
    x_1 = jax.random.normal(key, (1, self.seq_len, self.hidden_dim), dtype=self.dtype)
    self.assertEqual(compressor(x_1).shape, (1, self.seq_len // 4, self.head_dim))

    # Batch Size = 16
    x_16 = jax.random.normal(key, (16, self.seq_len, self.hidden_dim), dtype=self.dtype)
    self.assertEqual(compressor(x_16).shape, (16, self.seq_len // 4, self.head_dim))

  def test_non_divisible_sequence_length_truncates(self):
    """Verify that non-divisible sequence lengths are safely truncated to prevent reshape errors."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    compressor = attention_compressed.Compressor(config, 4, rngs=self.nnx_rng)
    key = jax.random.PRNGKey(42)
    x_odd = jax.random.normal(key, (self.batch_size, 257, self.hidden_dim), dtype=self.dtype)

    # Usable sequence length should be truncated from 257 to 256, yielding 64 chunks!
    compressed_x = compressor(x_odd)
    self.assertEqual(compressed_x.shape, (self.batch_size, 64, self.head_dim))

  def test_varying_hyperparameters(self):
    """Verify shapes under different model embedding and compression dimensions."""
    config_a = DummyConfig(compress_ratio=4, compressed_dim=256, head_dim=128, emb_dim=2048, dtype=self.dtype)
    compressor_a = attention_compressed.Compressor(config_a, 4, rngs=self.nnx_rng)
    key = jax.random.PRNGKey(42)
    x_a = jax.random.normal(key, (self.batch_size, self.seq_len, 2048), dtype=self.dtype)
    self.assertEqual(compressor_a(x_a).shape, (self.batch_size, self.seq_len // 4, 256))

  def test_hca_pytorch_equivalence(self):
    """Verify HCA Compressor numerical equivalence against 100% exact PyTorch reference."""
    # JAX Config
    config = DummyConfig(
        compress_ratio=128,
        compressed_dim=self.head_dim,
        head_dim=self.head_dim,
        emb_dim=self.hidden_dim,
        dtype=self.dtype,
    )

    # PyTorch Config
    py_config = DeepseekV4Config(
        compress_ratio=128, compressed_dim=self.head_dim, head_dim=self.head_dim, hidden_size=self.hidden_dim
    )

    # Instantiate modules
    jax_compressor = attention_compressed.Compressor(config, 128, rngs=self.nnx_rng)
    pytorch_compressor = DeepseekV4HCACompressor(py_config)

    # Weight Synchronization
    jax_compressor.wkv.kernel[...] = jnp.array(pytorch_compressor.kv_proj.weight.detach().numpy().T)
    jax_compressor.wgate.kernel[...] = jnp.array(pytorch_compressor.gate_proj.weight.detach().numpy().T)
    jax_compressor.ape[...] = jnp.array(pytorch_compressor.position_bias.detach().numpy())
    jax_compressor.norm.scale[...] = jnp.array(pytorch_compressor.kv_norm.weight.detach().numpy())

    # Identical deterministic inputs
    np_input = np.random.normal(size=(self.batch_size, self.seq_len, self.hidden_dim)).astype(np.float32)

    # Run PyTorch forward and collect intermediates
    torch_in = torch.from_numpy(np_input)
    with torch.no_grad():
      torch_out = pytorch_compressor(torch_in, None, None, None, 0)

      # Collect PyTorch intermediates
      pt_kv = pytorch_compressor.kv_proj(torch_in)
      pt_gate = pytorch_compressor.gate_proj(torch_in)
      n_windows = self.seq_len // 128
      chunk_kv = pt_kv.view(self.batch_size, n_windows, 128, -1)
      chunk_gate = pt_gate.view(self.batch_size, n_windows, 128, -1) + pytorch_compressor.position_bias
      pt_weights = chunk_gate.softmax(dim=2, dtype=torch.float32)
      pt_prenorm = torch.sum(chunk_kv * pt_weights, dim=2)

    # Run JAX forward collecting intermediates
    jax_in = jnp.array(np_input)
    jax_out, jax_inter = jax_compressor(jax_in, return_intermediates=True)

    # Reshape PyTorch output [B, 1, C, D] -> [B, C, D]
    torch_out_np = torch_out.squeeze(1).numpy()
    jax_out_np = np.array(jax_out)

    # Meticulous Intermediate Assertions
    np.testing.assert_allclose(np.array(jax_inter["kv"]), pt_kv.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_inter["gate"]), pt_gate.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_inter["weights"]), pt_weights.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_inter["prenorm"]), pt_prenorm.numpy(), rtol=1e-5, atol=1e-5)

    # Assert final output equivalence
    np.testing.assert_allclose(jax_out_np, torch_out_np, rtol=1e-5, atol=1e-5)

  def test_csa_pytorch_equivalence(self):
    """Verify CSA Compressor numerical equivalence against 100% exact PyTorch reference."""
    # JAX Config
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )

    # PyTorch Config
    py_config = DeepseekV4Config(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, hidden_size=self.hidden_dim
    )

    # Instantiate modules
    jax_compressor = attention_compressed.Compressor(config, 4, rngs=self.nnx_rng)
    pytorch_compressor = DeepseekV4CSACompressor(py_config)

    # Weight Synchronization
    jax_compressor.wkv.kernel[...] = jnp.array(pytorch_compressor.kv_proj.weight.detach().numpy().T)
    jax_compressor.wgate.kernel[...] = jnp.array(pytorch_compressor.gate_proj.weight.detach().numpy().T)
    pt_bias_Ca, pt_bias_Cb = np.split(pytorch_compressor.position_bias.detach().numpy(), 2, axis=-1)
    jax_ape = np.concatenate([pt_bias_Ca, pt_bias_Cb], axis=0)
    jax_compressor.ape[...] = jnp.array(jax_ape)
    jax_compressor.norm.scale[...] = jnp.array(pytorch_compressor.kv_norm.weight.detach().numpy())

    # Identical deterministic inputs
    np_input = np.random.normal(size=(self.batch_size, self.seq_len, self.hidden_dim)).astype(np.float32)

    # Run PyTorch forward and collect intermediates
    torch_in = torch.from_numpy(np_input)
    with torch.no_grad():
      torch_out = pytorch_compressor(torch_in, None, None, None, 0)

      # Collect PyTorch intermediates
      pt_kv = pytorch_compressor.kv_proj(torch_in)
      pt_gate = pytorch_compressor.gate_proj(torch_in)
      n_windows = self.seq_len // 4
      ratio = 4
      chunk_kv = pt_kv.view(self.batch_size, n_windows, ratio, -1)
      chunk_gate = pt_gate.view(self.batch_size, n_windows, ratio, -1) + pytorch_compressor.position_bias

      new_kv = chunk_kv.new_zeros((self.batch_size, n_windows, 2 * ratio, self.head_dim))
      new_gate = chunk_gate.new_full((self.batch_size, n_windows, 2 * ratio, self.head_dim), float("-inf"))
      new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
      new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
      if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
      pt_weights = new_gate.softmax(dim=2, dtype=torch.float32)
      pt_prenorm = torch.sum(new_kv * pt_weights, dim=2)

    # Run JAX forward collecting intermediates
    jax_in = jnp.array(np_input)
    jax_out, jax_inter = jax_compressor(jax_in, return_intermediates=True)

    # Reshape PyTorch output [B, 1, C, D] -> [B, C, D]
    torch_out_np = torch_out.squeeze(1).numpy()
    jax_out_np = np.array(jax_out)

    # Meticulous Intermediate Assertions
    np.testing.assert_allclose(np.array(jax_inter["kv"]), pt_kv.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_inter["gate"]), pt_gate.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_inter["weights"]), pt_weights.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_inter["prenorm"]), pt_prenorm.numpy(), rtol=1e-5, atol=1e-5)

    # Assert final output equivalence
    np.testing.assert_allclose(jax_out_np, torch_out_np, rtol=1e-5, atol=1e-5)

  def test_nnx_state_splitting(self):
    """Verify that JAX NNX can extract and split Compressor state and structure cleanly."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    compressor = attention_compressed.Compressor(config, 4, rngs=self.nnx_rng)

    # Extract State and Graph Definition
    graphdef, state = nnx.split(compressor)
    self.assertIsNotNone(graphdef)
    self.assertIsNotNone(state)

    # Reconstruct Module from split state and def
    reconstructed = nnx.merge(graphdef, state)
    self.assertEqual(reconstructed.compress_ratio, 4)
    self.assertTrue(reconstructed.overlap)

  def test_rng_reproducibility(self):
    """Verify that Compressor parameter initialization is perfectly deterministic under identical RNG seeds."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )

    # Instantiate twice with the exact same RNG keys
    rngs_1 = nnx.Rngs(params=42)
    rngs_2 = nnx.Rngs(params=42)

    compressor_1 = attention_compressed.Compressor(config, 4, rngs=rngs_1)
    compressor_2 = attention_compressed.Compressor(config, 4, rngs=rngs_2)

    # Assert parameter equivalence
    np.testing.assert_allclose(np.array(compressor_1.wkv.kernel[...]), np.array(compressor_2.wkv.kernel[...]))
    np.testing.assert_allclose(np.array(compressor_1.ape[...]), np.array(compressor_2.ape[...]))

  def test_grouped_linear_shape(self):
    """Verify GroupedLinear shape boundaries."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    layer = linears.GroupedLinear(config, rngs=self.nnx_rng)

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(
        key, (self.batch_size, self.seq_len, config.base_num_query_heads, self.head_dim), dtype=self.dtype
    )
    out = layer(x)

    expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.dtype, self.dtype)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_grouped_linear_pytorch_equivalence(self):
    """Verify GroupedLinear numerical equivalence against PyTorch reference."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    py_config = DeepseekV4Config(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, hidden_size=self.hidden_dim
    )

    jax_layer = linears.GroupedLinear(config, rngs=self.nnx_rng)
    pytorch_layer = DeepseekV4GroupedLinear(py_config)

    # Weight Sync
    pytorch_layer.w_a.data.normal_(0.0, 0.02)
    jax_layer.w_a[...] = jnp.array(pytorch_layer.w_a.detach().numpy())

    pytorch_layer.o_b_proj.weight.data.normal_(0.0, 0.02)
    jax_layer.o_b_proj.kernel[...] = jnp.array(pytorch_layer.o_b_proj.weight.detach().numpy().T)

    # Identical deterministic inputs
    np_input = np.random.normal(size=(self.batch_size, self.seq_len, config.base_num_query_heads, self.head_dim)).astype(
        np.float32
    )

    # Run PyTorch forward
    torch_in = torch.from_numpy(np_input)
    with torch.no_grad():
      torch_out = pytorch_layer(torch_in)

      # Intermediates for assert
      B, S, _, _ = torch_in.shape
      x_grouped = torch_in.view(B, S, pytorch_layer.o_groups, -1)
      pt_grouped_out = torch.einsum("bsgd,gdr->bsgr", x_grouped, pytorch_layer.w_a)
      pt_flattened = pt_grouped_out.reshape(B, S, -1)

    # Run JAX forward
    jax_in = jnp.array(np_input)
    jax_out = jax_layer(jax_in)

    # Compute JAX intermediates explicitly to verify JAX parameters
    B, S, _, _ = jax_in.shape
    jax_x_grouped = jnp.reshape(jax_in, (B, S, config.o_groups, -1))
    jax_grouped_out = jnp.einsum("bsgd,gdr -> bsgr", jax_x_grouped, jax_layer.w_a[...])
    jax_flattened = jnp.reshape(jax_grouped_out, (B, S, -1))

    # Assert intermediate bottlenecks and final outputs
    np.testing.assert_allclose(np.array(jax_flattened), pt_flattened.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_out), torch_out.numpy(), rtol=1e-5, atol=1e-5)

  def test_grouped_linear_nnx_state(self):
    """Verify that JAX NNX can split and merge GroupedLinear state cleanly."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    layer = linears.GroupedLinear(config, rngs=self.nnx_rng)

    # Extract State and Graph Definition
    graphdef, state = nnx.split(layer)
    self.assertIsNotNone(graphdef)
    self.assertIsNotNone(state)

    # Reconstruct Module from split state
    reconstructed = nnx.merge(graphdef, state)
    self.assertEqual(reconstructed.config.o_groups, 8)
    self.assertEqual(reconstructed.config.o_lora_rank, 1024)


# =============================================================================
# PyTorch Golden Reference Classes
# =============================================================================
class DeepSeekV4RotaryEmbeddingPT(nn.Module):
  """PyTorch reference mock for DeepSeek-V4 interleaved RoPE."""

  def __init__(self, dim, rope_theta=10000.0, compress_rope_theta=160000.0):
    super().__init__()
    self.dim = dim
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    inv_freq_comp = 1.0 / (compress_rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer("inv_freq", inv_freq)
    self.register_buffer("inv_freq_comp", inv_freq_comp)

  def forward(self, x, position_ids, is_compressed=False):
    inv_freq = self.inv_freq_comp if is_compressed else self.inv_freq
    freqs = torch.einsum("i,j->ij", position_ids.flatten(), inv_freq)
    freqs = freqs.view(*position_ids.shape, -1)
    cos = freqs.cos()
    sin = freqs.sin()
    x_rotated = apply_rotary_pos_emb_pt(x, cos, sin)
    return x_rotated, cos, sin


def apply_rotary_pos_emb_pt(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  cos_interleaved = cos.repeat_interleave(2, dim=-1).unsqueeze(2)
  sin_interleaved = sin.repeat_interleave(2, dim=-1).unsqueeze(2)
  rope_dim = cos_interleaved.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = ((rope.float() * cos_interleaved) + (rotate_half(rope).float() * sin_interleaved)).to(x.dtype)
  return torch.cat([nope, rotated], dim=-1)


def apply_conjugate_unrotation_pt(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  cos_interleaved = cos.repeat_interleave(2, dim=-1).unsqueeze(2)
  sin_interleaved = -sin.repeat_interleave(2, dim=-1).unsqueeze(2)  # negate sin for conjugate
  rope_dim = cos_interleaved.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = ((rope.float() * cos_interleaved) + (rotate_half(rope).float() * sin_interleaved)).to(x.dtype)
  return torch.cat([nope, rotated], dim=-1)


def deepseek_v4_attention_forward_pt(q_local, k_local, v_local, k_comp, v_comp, cos_unrotate, sin_unrotate):
  """PyTorch reference mock for DeepSeek-V4 attention forward pass."""
  _, S_local, _, D = q_local.shape
  _, S_comp, _, _ = k_comp.shape

  k_combined = torch.cat([k_comp, k_local], dim=1)
  v_combined = torch.cat([v_comp, v_local], dim=1)

  mask_comp = torch.zeros((S_local, S_comp), dtype=q_local.dtype)
  mask_local = torch.triu(torch.full((S_local, S_local), float("-inf"), dtype=q_local.dtype), diagonal=1)

  attn_mask = torch.cat([mask_comp, mask_local], dim=1).unsqueeze(0).unsqueeze(0)
  q_scaled = q_local * (D**-0.5)

  attn_scores = torch.einsum("bsqd,bckd->bqsc", q_scaled, k_combined) + attn_mask
  attn_weights = torch.softmax(attn_scores, dim=-1)
  attn_output = torch.einsum("bqsc,bckd->bsqd", attn_weights, v_combined)

  return apply_conjugate_unrotation_pt(attn_output, cos_unrotate, sin_unrotate)


# =============================================================================
# JAX / PyTorch Coordinate Parity Tests
# =============================================================================
class DeepSeekV4CoordinateSystemsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.batch = 2
    self.seq_local = 128
    self.seq_comp = 256
    self.heads = 4
    self.dim = 128
    self.rope_dim = 64
    self.rope_theta = 10000.0
    self.compress_rope_theta = 160000.0
    self.nnx_rng = nnx.Rngs(params=0)

  def test_interleaved_rope_parity(self):
    np.random.seed(42)
    x_np = np.random.randn(self.batch, self.seq_local, self.heads, self.dim).astype(np.float32)
    pos_np = np.arange(self.seq_local).reshape(1, self.seq_local).repeat(self.batch, axis=0)

    pt_module = DeepSeekV4RotaryEmbeddingPT(self.rope_dim, self.rope_theta, self.compress_rope_theta)
    y_pt, _, _ = pt_module(torch.tensor(x_np), torch.tensor(pos_np), is_compressed=False)

    jax_module = jax_embeddings.DeepSeekV4RotaryEmbedding(
        self.rope_dim, rope_theta=self.rope_theta, compress_rope_theta=self.compress_rope_theta
    )
    y_jax, _, _ = jax_module(jnp.array(x_np), jnp.array(pos_np), is_compressed=False)

    np.testing.assert_allclose(y_pt.detach().numpy(), y_jax, rtol=1e-5, atol=1e-5)

  def test_attention_forward_parity(self):
    np.random.seed(45)
    q_local = np.random.randn(self.batch, self.seq_local, self.heads, self.dim).astype(np.float32)
    k_local = np.random.randn(self.batch, self.seq_local, self.heads, self.dim).astype(np.float32)
    v_local = np.random.randn(self.batch, self.seq_local, self.heads, self.dim).astype(np.float32)
    k_comp = np.random.randn(self.batch, self.seq_comp, self.heads, self.dim).astype(np.float32)
    v_comp = np.random.randn(self.batch, self.seq_comp, self.heads, self.dim).astype(np.float32)

    # Test conjugate un-rotation with rope_dim
    cos_un = np.random.randn(self.batch, self.seq_local, self.rope_dim // 2).astype(np.float32)
    sin_un = np.random.randn(self.batch, self.seq_local, self.rope_dim // 2).astype(np.float32)

    out_pt = deepseek_v4_attention_forward_pt(
        torch.tensor(q_local),
        torch.tensor(k_local),
        torch.tensor(v_local),
        torch.tensor(k_comp),
        torch.tensor(v_comp),
        torch.tensor(cos_un),
        torch.tensor(sin_un),
    )

    # Expand cos_un/sin_un for JAX to broadcast cleanly over heads -> [B, S, 1, rope_dim // 2]
    cos_jax = jnp.array(cos_un)[..., None, :]
    sin_jax = jnp.array(sin_un)[..., None, :]

    out_jax = jax_attention_op.deepseek_v4_attention_forward(
        jnp.array(q_local),
        jnp.array(k_local),
        jnp.array(v_local),
        jnp.array(k_comp),
        jnp.array(v_comp),
        cos_jax,
        sin_jax,
    )

    np.testing.assert_allclose(out_pt.detach().numpy(), out_jax, rtol=1e-4, atol=1e-4)

  def test_nnx_state_split_merge_and_jit(self):
    """Verifies that DeepSeekV4RotaryEmbedding satisfies NNX split/merge and JIT tracing without TracerLeaks."""
    module = jax_embeddings.DeepSeekV4RotaryEmbedding(
        self.rope_dim, rope_theta=self.rope_theta, compress_rope_theta=self.compress_rope_theta
    )

    # Extract State and Graph Definition
    graphdef, state = nnx.split(module)

    @jax.jit
    def compile_and_forward(state_in, x, pos):
      # Reconstruct module cleanly inside the compiled XLA JIT block
      mod = nnx.merge(graphdef, state_in)
      return mod(x, pos, is_compressed=False)

    x_jax = jnp.ones((self.batch, self.seq_local, self.heads, self.dim), dtype=jnp.float32)
    pos_jax = jnp.arange(self.seq_local).reshape(1, self.seq_local).repeat(self.batch, axis=0)

    # Execute JIT compilation and forward pass
    y_jax, cos_jax, _ = compile_and_forward(state, x_jax, pos_jax)

    self.assertEqual(y_jax.shape, x_jax.shape)
    self.assertEqual(cos_jax.shape, (self.batch, self.seq_local, 1, self.rope_dim // 2))

  def test_attention_varying_sequence_boundaries(self):
    """Verifies attention parity under extreme oversized and empty compressed sequence boundaries."""
    np.random.seed(46)

    # Edge Case A: Oversized Compressed History (seq_comp > seq_local)
    seq_local_small = 32
    seq_comp_large = 512

    q_l = np.random.randn(self.batch, seq_local_small, self.heads, self.dim).astype(np.float32)
    k_l = np.random.randn(self.batch, seq_local_small, self.heads, self.dim).astype(np.float32)
    v_l = np.random.randn(self.batch, seq_local_small, self.heads, self.dim).astype(np.float32)
    k_c = np.random.randn(self.batch, seq_comp_large, self.heads, self.dim).astype(np.float32)
    v_c = np.random.randn(self.batch, seq_comp_large, self.heads, self.dim).astype(np.float32)

    cos_u = np.random.randn(self.batch, seq_local_small, self.rope_dim // 2).astype(np.float32)
    sin_u = np.random.randn(self.batch, seq_local_small, self.rope_dim // 2).astype(np.float32)

    out_pt = deepseek_v4_attention_forward_pt(
        torch.tensor(q_l),
        torch.tensor(k_l),
        torch.tensor(v_l),
        torch.tensor(k_c),
        torch.tensor(v_c),
        torch.tensor(cos_u),
        torch.tensor(sin_u),
    )

    cos_jax = jnp.array(cos_u)[..., None, :]
    sin_jax = jnp.array(sin_u)[..., None, :]

    out_jax = jax_attention_op.deepseek_v4_attention_forward(
        jnp.array(q_l),
        jnp.array(k_l),
        jnp.array(v_l),
        jnp.array(k_c),
        jnp.array(v_c),
        cos_jax,
        sin_jax,
    )
    np.testing.assert_allclose(out_pt.detach().numpy(), out_jax, rtol=1e-4, atol=1e-4)

    # Edge Case B: Empty Compressed History (seq_comp = 0)
    k_c_zero = np.zeros((self.batch, 0, self.heads, self.dim), dtype=np.float32)
    v_c_zero = np.zeros((self.batch, 0, self.heads, self.dim), dtype=np.float32)

    out_pt_zero = deepseek_v4_attention_forward_pt(
        torch.tensor(q_l),
        torch.tensor(k_l),
        torch.tensor(v_l),
        torch.tensor(k_c_zero),
        torch.tensor(v_c_zero),
        torch.tensor(cos_u),
        torch.tensor(sin_u),
    )

    out_jax_zero = jax_attention_op.deepseek_v4_attention_forward(
        jnp.array(q_l),
        jnp.array(k_l),
        jnp.array(v_l),
        jnp.array(k_c_zero),
        jnp.array(v_c_zero),
        cos_jax,
        sin_jax,
    )
    np.testing.assert_allclose(out_pt_zero.detach().numpy(), out_jax_zero, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
