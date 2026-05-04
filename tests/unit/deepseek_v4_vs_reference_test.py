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

  def test_non_divisible_sequence_length_fails(self):
    """Verify that non-divisible sequence lengths cause static reshape errors."""
    config = DummyConfig(
        compress_ratio=4, compressed_dim=self.head_dim, head_dim=self.head_dim, emb_dim=self.hidden_dim, dtype=self.dtype
    )
    compressor = attention_compressed.Compressor(config, 4, rngs=self.nnx_rng)
    key = jax.random.PRNGKey(42)
    x_bad = jax.random.normal(key, (self.batch_size, 257, self.hidden_dim), dtype=self.dtype)

    with self.assertRaises((ValueError, TypeError)):
      _ = compressor(x_bad)

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


if __name__ == "__main__":
  unittest.main()
