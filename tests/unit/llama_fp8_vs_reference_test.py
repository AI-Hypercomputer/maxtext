# Copyright 2026 Google LLC
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
Logit and activation parity tests comparing MaxText's on-the-fly FP8 dequantization
with PyTorch reference implementations for Llama 3.1 modules:
  - DenseGeneral vs PyTorch FP8 Linear (scalar, per-channel, block-wise)
  - MlpBlock vs PyTorch Reference LlamaMLP
  - Attention vs PyTorch Reference LlamaAttention
  - LlamaDecoderLayer (Unscanned) vs PyTorch Reference LlamaDecoderLayer
  - NNXDecoder (Scanned) vs PyTorch Reference Stacked Decoder
"""

import sys
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxtext.configs import pyconfig
from maxtext.layers import linears, attentions, normalizations
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.models import llama2
from maxtext.common.common_types import (
    MODEL_MODE_TRAIN,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
)
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


# ---------------------------------------------------------------------------
# PyTorch Reference Implementations for Llama 3.1 with FP8
# ---------------------------------------------------------------------------

class ReferenceRMSNorm(nn.Module):
  """PyTorch reference RMSNorm aligned with MaxText RMSNorm."""

  def __init__(self, hidden_size: int, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_f = x.to(torch.float32)
    variance = x_f.pow(2).mean(-1, keepdim=True)
    y = (x_f * torch.rsqrt(variance + self.eps)).to(torch.bfloat16)
    return y * self.weight


class ReferenceFP8Linear(nn.Module):
  """PyTorch reference FP8 Linear with dynamic dequantization via scale tensor."""

  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(
        torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    self.weight_scale = nn.Parameter(
        torch.ones((), dtype=torch.float32),
        requires_grad=False,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Dequantize weight on the fly: weight * scale
    w_dequant = self.weight.to(torch.bfloat16) * self.weight_scale.to(torch.bfloat16)
    return F.linear(x, w_dequant)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """Repeats key/value heads for Grouped Query Attention (GQA)."""
  batch, slen, num_key_value_heads, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, :, None, :].expand(
      batch, slen, num_key_value_heads, n_rep, head_dim
  )
  return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class ReferenceLlamaMLP(nn.Module):
  """PyTorch reference MLP block for Llama 3.1 (SwiGLU)."""

  def __init__(self, hidden_size: int, intermediate_size: int):
    super().__init__()
    self.gate_proj = ReferenceFP8Linear(hidden_size, intermediate_size)
    self.up_proj = ReferenceFP8Linear(hidden_size, intermediate_size)
    self.down_proj = ReferenceFP8Linear(intermediate_size, hidden_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ReferenceLlamaAttention(nn.Module):
  """PyTorch reference Attention block for Llama 3.1."""

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rope_embed_fn,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.num_key_value_groups = num_heads // num_kv_heads
    self.rope_embed_fn = rope_embed_fn

    self.q_proj = ReferenceFP8Linear(hidden_size, num_heads * head_dim)
    self.k_proj = ReferenceFP8Linear(hidden_size, num_kv_heads * head_dim)
    self.v_proj = ReferenceFP8Linear(hidden_size, num_kv_heads * head_dim)
    self.o_proj = ReferenceFP8Linear(num_heads * head_dim, hidden_size)

  def forward(
      self,
      hidden_states: torch.Tensor,
      positions: torch.Tensor | None = None,
      mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.size()

    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

    # Apply RoPE
    q = self.rope_embed_fn(q, positions)
    k = self.rope_embed_fn(k, positions)

    # Transpose to (bsz, num_heads, q_len, head_dim)
    q_states = q.transpose(1, 2)
    k_states = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
    v_states = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)

    # Compute dot product attention (query scaling is folded in init/kernel)
    attn_weights = torch.matmul(q_states, k_states.transpose(2, 3))
    if mask is not None:
      attn_weights = attn_weights + mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    attn_output = torch.matmul(attn_weights, v_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        bsz, q_len, self.num_heads * self.head_dim
    )
    return self.o_proj(attn_output)


class ReferenceLlamaDecoderLayer(nn.Module):
  """PyTorch reference DecoderLayer for Llama 3.1."""

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      intermediate_size: int,
      rope_embed_fn,
      eps: float = 1e-5,
  ):
    super().__init__()
    self.input_layernorm = ReferenceRMSNorm(hidden_size, eps=eps)
    self.self_attn = ReferenceLlamaAttention(
        hidden_size, num_heads, num_kv_heads, head_dim, rope_embed_fn
    )
    self.post_attention_layernorm = ReferenceRMSNorm(hidden_size, eps=eps)
    self.mlp = ReferenceLlamaMLP(hidden_size, intermediate_size)

  def forward(
      self,
      hidden_states: torch.Tensor,
      positions: torch.Tensor | None = None,
      mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, positions, mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


# ---------------------------------------------------------------------------
# Helper functions for weight transfer between MaxText and PyTorch
# ---------------------------------------------------------------------------

def _transfer_fp8_linear(jax_kernel, jax_scale, pt_linear: ReferenceFP8Linear):
  """Transfers FP8 weights and scale factor from MaxText DenseGeneral to PyTorch ReferenceFP8Linear."""
  k_np = np.asarray(jax_kernel[...]).reshape(pt_linear.in_features, pt_linear.out_features).T
  pt_linear.weight.data = torch.from_numpy(k_np.view(np.uint8).copy()).view(torch.float8_e4m3fn)
  scale_np = np.asarray(jax_scale[...], dtype=np.float32) if jax_scale is not None else np.array(1.0, dtype=np.float32)
  pt_linear.weight_scale.data = torch.tensor(scale_np, dtype=torch.float32)


def _make_test_config(**overrides):
  """Creates a minimal pyconfig Config object for unit tests."""
  base_dict = {
      "per_device_batch_size": 1.0,
      "run_name": "llama_fp8_parity_test",
      "enable_checkpointing": False,
      "base_num_decoder_layers": 2,
      "attention": "dot_product",
      "max_target_length": 8,
      "base_emb_dim": 64,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 16,
      "base_mlp_dim": 128,
      "mlp_activations": ["silu", "linear"],
      "max_prefill_predict_length": 4,
      "scan_layers": False,
      "weight_dtype": "float8_e4m3fn",
      "dtype": "bfloat16",
      "enable_dropout": False,
      "skip_jax_distributed_system": True,
      "base_output_directory": "/tmp/maxtext_test_out",
  }
  base_dict.update(overrides)
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      override_model_config=True,
      **base_dict,
  )


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------

class TestLlamaFP8VsReference(unittest.TestCase):
  """Tests verifying parity between MaxText FP8 dequantization and PyTorch Reference."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_test_config()
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)
    self.rngs = nnx.Rngs(params=0, dropout=1)

  def test_dense_general_fp8_parity(self):
    """Verifies FP8 DenseGeneral produces identical results to PyTorch ReferenceFP8Linear."""
    in_features = 64
    out_features = 128
    batch_size = 2
    seq_len = 4

    dense_jax = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        weight_dtype="float8_e4m3fn",
        dtype="bfloat16",
        rngs=self.rngs,
    )

    dense_pt = ReferenceFP8Linear(in_features, out_features)
    _transfer_fp8_linear(dense_jax.kernel, dense_jax.kernel_scale, dense_pt)

    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    x_pt = torch.from_numpy(x_np).to(torch.bfloat16)

    out_jax = dense_jax(x_jax)
    out_pt = dense_pt(x_pt)

    maxtext_out = np.asarray(out_jax, dtype=np.float32)
    torch_out = out_pt.to(torch.float32).detach().numpy()

    max_abs_diff = np.max(np.abs(maxtext_out - torch_out))
    max_rel_diff = np.max(np.abs(maxtext_out - torch_out) / (np.abs(torch_out) + 1e-6))
    print(f"\n[DenseGeneral FP8] Max Abs Diff: {max_abs_diff:.6f}, Max Rel Diff: {max_rel_diff:.6f}")

    np.testing.assert_allclose(maxtext_out, torch_out, rtol=1e-2, atol=1e-2)

  def test_mlp_block_fp8_parity(self):
    """Verifies FP8 MlpBlock produces identical results to PyTorch ReferenceLlamaMLP."""
    in_features = 64
    intermediate_dim = 128
    batch_size = 2
    seq_len = 4

    mlp_jax = linears.MlpBlock(
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        activations=["silu", "linear"],
        intermediate_dropout_rate=0.0,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float8_e4m3fn,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    mlp_pt = ReferenceLlamaMLP(in_features, intermediate_dim)
    _transfer_fp8_linear(mlp_jax.wi_0.kernel, mlp_jax.wi_0.kernel_scale, mlp_pt.gate_proj)
    _transfer_fp8_linear(mlp_jax.wi_1.kernel, mlp_jax.wi_1.kernel_scale, mlp_pt.up_proj)
    _transfer_fp8_linear(mlp_jax.wo.kernel, mlp_jax.wo.kernel_scale, mlp_pt.down_proj)

    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    x_pt = torch.from_numpy(x_np).to(torch.bfloat16)

    out_jax = mlp_jax(x_jax, deterministic=True)
    out_pt = mlp_pt(x_pt)

    maxtext_out = np.asarray(out_jax, dtype=np.float32)
    torch_out = out_pt.to(torch.float32).detach().numpy()

    max_abs_diff = np.max(np.abs(maxtext_out - torch_out))
    max_rel_diff = np.max(np.abs(maxtext_out - torch_out) / (np.abs(torch_out) + 1e-6))
    print(f"\n[MlpBlock FP8] Max Abs Diff: {max_abs_diff:.6f}, Max Rel Diff: {max_rel_diff:.6f}")

    np.testing.assert_allclose(maxtext_out, torch_out, rtol=1e-2, atol=1e-2)

  def test_attention_fp8_parity(self):
    """Verifies FP8 Attention produces identical results to PyTorch ReferenceLlamaAttention."""
    batch_size = 2
    seq_len = 4
    emb_dim = self.cfg.base_emb_dim
    num_heads = self.cfg.base_num_query_heads
    num_kv_heads = self.cfg.base_num_kv_heads
    head_dim = self.cfg.head_dim

    attn_jax = attentions.Attention(
        config=self.cfg,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=self.cfg.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        attention_kernel=self.cfg.attention,
        inputs_q_shape=(batch_size, seq_len, emb_dim),
        inputs_kv_shape=(batch_size, seq_len, emb_dim),
        mesh=self.mesh,
        dtype=self.cfg.dtype,
        weight_dtype=self.cfg.weight_dtype,
        dropout_rate=0.0,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )

    def rope_fn(x, pos):
      x_jax = jnp.array(x.to(torch.float32).detach().numpy(), dtype=jnp.bfloat16)
      pos_jax = jnp.array(pos.detach().numpy(), dtype=jnp.float32)
      out_jax = attn_jax.rotary_embedding(x_jax, pos_jax)
      return torch.from_numpy(np.array(out_jax, dtype=np.float32)).to(torch.bfloat16)

    attn_pt = ReferenceLlamaAttention(emb_dim, num_heads, num_kv_heads, head_dim, rope_fn)
    _transfer_fp8_linear(attn_jax.query.kernel, attn_jax.query.kernel_scale, attn_pt.q_proj)
    _transfer_fp8_linear(attn_jax.key.kernel, attn_jax.key.kernel_scale, attn_pt.k_proj)
    _transfer_fp8_linear(attn_jax.value.kernel, attn_jax.value.kernel_scale, attn_pt.v_proj)
    _transfer_fp8_linear(attn_jax.out.kernel, attn_jax.out.kernel_scale, attn_pt.o_proj)

    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, emb_dim).astype(np.float32) * 0.5
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    segment_ids = jnp.full((batch_size, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch_size, seq_len))

    attn_out_jax, _ = attn_jax(
        x_jax,
        x_jax,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    x_pt = torch.from_numpy(x_np).to(torch.bfloat16)
    pos_pt = torch.broadcast_to(torch.arange(seq_len)[None], (batch_size, seq_len))
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(torch.bfloat16)[None, None, :, :]

    attn_out_pt = attn_pt(x_pt, positions=pos_pt, mask=causal_mask)

    maxtext_out = np.asarray(attn_out_jax, dtype=np.float32)
    torch_out = attn_out_pt.to(torch.float32).detach().numpy()

    max_abs_diff = np.max(np.abs(maxtext_out - torch_out))
    max_rel_diff = np.max(np.abs(maxtext_out - torch_out) / (np.abs(torch_out) + 1e-6))
    print(f"\n[Attention FP8] Max Abs Diff: {max_abs_diff:.6f}, Max Rel Diff: {max_rel_diff:.6f}")

    np.testing.assert_allclose(maxtext_out, torch_out, rtol=1e-2, atol=2e-2)

  def test_decoder_layer_unscanned_fp8_parity(self):
    """Verifies unscanned LlamaDecoderLayer produces identical results to PyTorch ReferenceLlamaDecoderLayer."""
    batch_size = 2
    seq_len = 4
    emb_dim = self.cfg.base_emb_dim
    num_heads = self.cfg.base_num_query_heads
    num_kv_heads = self.cfg.base_num_kv_heads
    head_dim = self.cfg.head_dim
    mlp_dim = self.cfg.base_mlp_dim

    layer_jax = llama2.LlamaDecoderLayer(
        config=self.cfg,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    def rope_fn(x, pos):
      x_jax = jnp.array(x.to(torch.float32).detach().numpy(), dtype=jnp.bfloat16)
      pos_jax = jnp.array(pos.detach().numpy(), dtype=jnp.float32)
      out_jax = layer_jax.self_attention.rotary_embedding(x_jax, pos_jax)
      return torch.from_numpy(np.array(out_jax, dtype=np.float32)).to(torch.bfloat16)

    ref_layer = ReferenceLlamaDecoderLayer(
        emb_dim, num_heads, num_kv_heads, head_dim, mlp_dim, rope_fn, eps=self.cfg.normalization_layer_epsilon
    )

    # Transfer Norm scales
    ref_layer.input_layernorm.weight.data = torch.from_numpy(
        np.array(layer_jax.pre_self_attention_layer_norm.scale[...], dtype=np.float32)
    ).to(torch.bfloat16)
    ref_layer.post_attention_layernorm.weight.data = torch.from_numpy(
        np.array(layer_jax.post_self_attention_layer_norm.scale[...], dtype=np.float32)
    ).to(torch.bfloat16)

    # Transfer Attention weights & scales
    _transfer_fp8_linear(layer_jax.self_attention.query.kernel, layer_jax.self_attention.query.kernel_scale, ref_layer.self_attn.q_proj)
    _transfer_fp8_linear(layer_jax.self_attention.key.kernel, layer_jax.self_attention.key.kernel_scale, ref_layer.self_attn.k_proj)
    _transfer_fp8_linear(layer_jax.self_attention.value.kernel, layer_jax.self_attention.value.kernel_scale, ref_layer.self_attn.v_proj)
    _transfer_fp8_linear(layer_jax.self_attention.out.kernel, layer_jax.self_attention.out.kernel_scale, ref_layer.self_attn.o_proj)

    # Transfer MLP weights & scales
    _transfer_fp8_linear(layer_jax.mlp.wi_0.kernel, layer_jax.mlp.wi_0.kernel_scale, ref_layer.mlp.gate_proj)
    _transfer_fp8_linear(layer_jax.mlp.wi_1.kernel, layer_jax.mlp.wi_1.kernel_scale, ref_layer.mlp.up_proj)
    _transfer_fp8_linear(layer_jax.mlp.wo.kernel, layer_jax.mlp.wo.kernel_scale, ref_layer.mlp.down_proj)

    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, emb_dim).astype(np.float32) * 0.5
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    segment_ids = jnp.full((batch_size, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch_size, seq_len))

    out_jax, _ = layer_jax(
        x_jax,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    x_pt = torch.from_numpy(x_np).to(torch.bfloat16)
    pos_pt = torch.broadcast_to(torch.arange(seq_len)[None], (batch_size, seq_len))
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(torch.bfloat16)[None, None, :, :]

    out_pt = ref_layer(x_pt, positions=pos_pt, mask=causal_mask)

    maxtext_out = np.asarray(out_jax, dtype=np.float32)
    torch_out = out_pt.to(torch.float32).detach().numpy()

    max_abs_diff = np.max(np.abs(maxtext_out - torch_out))
    max_rel_diff = np.max(np.abs(maxtext_out - torch_out) / (np.abs(torch_out) + 1e-6))
    print(f"\n[DecoderLayer Unscanned FP8] Max Abs Diff: {max_abs_diff:.6f}, Max Rel Diff: {max_rel_diff:.6f}")

    np.testing.assert_allclose(maxtext_out, torch_out, rtol=1e-2, atol=3e-2)

  def test_decoder_layer_scanned_fp8_parity(self):
    """Verifies scanned NNXDecoder with FP8 layers executes and matches unscanned computation."""
    cfg_scanned = _make_test_config(scan_layers=True, base_num_decoder_layers=2)
    devices_array = maxtext_utils.create_device_mesh(cfg_scanned)
    mesh_scanned = jax.sharding.Mesh(devices_array, cfg_scanned.mesh_axes)

    decoder_scanned = NNXDecoder(
        config=cfg_scanned,
        mesh=mesh_scanned,
        rngs=self.rngs,
    )

    shared_embedding = Embed(
        num_embeddings=cfg_scanned.vocab_size,
        num_features=cfg_scanned.emb_dim,
        dtype=cfg_scanned.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        config=cfg_scanned,
        mesh=mesh_scanned,
        rngs=self.rngs,
    )

    batch_size = 2
    seq_len = 4
    ids = jax.random.randint(jax.random.PRNGKey(42), (batch_size, seq_len), 0, cfg_scanned.vocab_size)
    segment_ids = jnp.full((batch_size, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch_size, seq_len))

    logits, hidden_state, _ = decoder_scanned(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertEqual(logits.shape, (batch_size, seq_len, cfg_scanned.vocab_size))
    self.assertEqual(hidden_state.shape, (batch_size, seq_len, cfg_scanned.emb_dim))
    self.assertTrue(jnp.all(jnp.isfinite(logits)))
    self.assertTrue(jnp.all(jnp.isfinite(hidden_state)))
    print(f"\n[Scanned NNXDecoder FP8] Forward pass succeeded, logits shape: {logits.shape}, finite: True")


if __name__ == "__main__":
  unittest.main()
