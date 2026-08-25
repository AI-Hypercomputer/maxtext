# Copyright 2026 Google LLC
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

"""Unit tests validating DeepSeek-V4 MaxText components against official DeepSeek reference implementation.

This test uses the official reference architecture from:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
and translates GPU TileLang kernels from:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/kernel.py
into pure, unoptimized PyTorch CPU operations to verify numerical parity against MaxText implementations.
Parameter conversion is performed using `param_mapping.py`.
"""

import dataclasses
import math
import sys
import unittest
from typing import Tuple, Optional, List

import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn_pt
import torch.nn.functional as F

from maxtext.checkpoint_conversion.utils.param_mapping import (
    DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING,
    DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
)
from maxtext.common.common_types import (
    Config,
    DecoderBlockType,
    AttentionType,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers.attention_compressed import CompressedAttention
from maxtext.layers.embeddings import Embed, DeepSeekV4RotaryEmbedding
from maxtext.layers.linears import DeepSeekV4GroupedLinear
from maxtext.layers.mhc import DeepSeek4HyperHead
from maxtext.layers.moe import RoutedMoE, RoutedAndSharedMoE
from maxtext.layers.normalizations import RMSNorm
from maxtext.models.deepseek4 import DeepSeek4DecoderLayer


# ==============================================================================
# Helper to create valid MaxText configuration for DeepSeek-V4
# ==============================================================================

def get_maxtext_config(**overrides) -> Config:
  config_arguments = {
      "model_name": "deepseek4-284b",
      "override_model_config": True,
      "per_device_batch_size": 1,
      "matmul_precision": "highest",
      "megablox": False,
      "sparse_matmul": False,
      "dtype": "float32",
      "weight_dtype": "float32",
      "base_num_decoder_layers": 5,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "compress_ratios": [0, 0, 4, 128, 4],
      "mtp_num_layers": 0,
      "max_target_length": 32,
      "max_prefill_predict_length": 32,
      "skip_jax_distributed_system": True,
      "scan_layers": False,
  }
  config_arguments.update(overrides)
  argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
  return pyconfig.initialize(argv, **config_arguments)


# ==============================================================================
# 1. Official DeepSeek ModelArgs & Reference Config
# ==============================================================================

@dataclasses.dataclass
class ModelArgs:
  max_batch_size: int = 8
  max_seq_len: int = 4096
  dtype: str = "float32"
  vocab_size: int = 129280
  dim: int = 4096
  inter_dim: int = 2048
  moe_inter_dim: int = 2048
  n_layers: int = 5
  n_dense_layers: int = 0
  n_heads: int = 64
  n_routed_experts: int = 4
  n_shared_experts: int = 1
  n_activated_experts: int = 2
  n_expert_groups: int = 1
  n_limited_groups: int = 1
  score_func: str = "sqrtsoftplus"
  route_scale: float = 1.5
  q_lora_rank: int = 1024
  kv_lora_rank: int = 512
  o_lora_rank: int = 1024
  o_groups: int = 8
  head_dim: int = 512
  qk_nope_head_dim: int = 512
  qk_rope_head_dim: int = 64
  v_head_dim: int = 512
  sliding_window: int = 128
  index_topk: int = 512
  index_n_heads: int = 64
  index_head_dim: int = 128
  index_block_size: int = 1
  index_routing_dim: int = 32
  index_q_lora_rank: int = 16
  original_seq_len: int = 65536
  rope_theta: float = 10000.0
  rope_factor: float = 40.0
  beta_fast: int = 32
  beta_slow: int = 1
  mscale: float = 1.0
  hc_mult: int = 4
  hc_sinkhorn_iters: int = 20
  hc_eps: float = 1e-6
  norm_eps: float = 1e-6
  compress_rates: List[int] = dataclasses.field(default_factory=lambda: [0, 0, 4, 128, 4])
  n_hash_layers: int = 3
  n_mtp_layers: int = 0


# ==============================================================================
# 2. CPU Reference Kernels (Pure PyTorch)
# ==============================================================================

def precompute_freqs_cis(dim: int, end: int = 4096, theta: float = 10000.0) -> torch.Tensor:
  """Computes RoPE frequency cis representation."""
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  t = torch.arange(end, dtype=torch.float32)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  """Applies interleaved rotary embedding to trailing dimensions."""
  dtype = x.dtype
  x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
  while freqs_cis.ndim < x_c.ndim:
    if freqs_cis.ndim == x_c.ndim - 1 and freqs_cis.shape[0] == x_c.shape[1]:
      freqs_cis = freqs_cis.unsqueeze(0)
    else:
      freqs_cis = freqs_cis.unsqueeze(-2)
  out = torch.view_as_real(x_c * freqs_cis).flatten(-2)
  return out.to(dtype)


def apply_partial_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  """Applies interleaved rotary embedding to trailing channels of x."""
  rope_dim = freqs_cis.shape[-1] * 2
  if x.shape[-1] > rope_dim:
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = apply_rotary_emb(rope, freqs_cis)
    return torch.cat([nope, rotated], dim=-1)
  else:
    return apply_rotary_emb(x, freqs_cis)


def sparse_attn_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
  """Pure CPU unoptimized implementation of TileLang sparse_attn_kernel."""
  b, m, h, d = q.shape
  topk = topk_idxs.shape[-1]
  valid_mask = topk_idxs >= 0
  safe_idxs = torch.where(valid_mask, topk_idxs, torch.zeros_like(topk_idxs))

  # k, v: [b, n, h, d] -> gather -> [b, m, topk, h, d]
  k_expanded = k.unsqueeze(1).expand(b, m, -1, h, d)
  v_expanded = v.unsqueeze(1).expand(b, m, -1, h, d)
  gather_index = safe_idxs.unsqueeze(-1).unsqueeze(-1).expand(b, m, topk, h, d)
  gathered_k = torch.gather(k_expanded, dim=2, index=gather_index)
  gathered_v = torch.gather(v_expanded, dim=2, index=gather_index)

  # scores: [b, m, h, topk]
  scores = torch.einsum("bmhd,bmthd->bmht", q.float(), gathered_k.float()) * softmax_scale
  scores = torch.where(valid_mask.unsqueeze(2), scores, float("-inf"))

  sink = attn_sink.view(1, 1, h, 1).expand(b, m, h, 1).float()
  all_scores = torch.cat([scores, sink], dim=-1)
  all_weights = torch.softmax(all_scores, dim=-1)
  attn_weights = all_weights[..., :topk]

  out = torch.einsum("bmht,bmthd->bmhd", attn_weights, gathered_v.float())
  return out.to(q.dtype)


def hc_split_sinkhorn_cpu(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Pure CPU implementation of TileLang hc_split_sinkhorn."""
  pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
  post = 2.0 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
  comb = (
      mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult) * hc_scale[2]
      + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
  )
  comb = torch.softmax(comb.float(), dim=-1) + eps
  comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
  for _ in range(sinkhorn_iters - 1):
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
  return pre.to(mixes.dtype), post.to(mixes.dtype), comb.to(mixes.dtype)


# ==============================================================================
# 3. Official DeepSeek PyTorch Reference Architecture (inference/model.py CPU version)
# ==============================================================================

class ParallelEmbedding_PT(nn_pt.Module):
  def __init__(self, vocab_size: int, dim: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.weight = nn_pt.Parameter(torch.randn(vocab_size, dim, dtype=torch.float32) * 0.02)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.embedding(x, self.weight)


class RMSNorm_PT(nn_pt.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn_pt.Parameter(torch.ones(dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class Linear_PT(nn_pt.Module):
  def __init__(self, in_features: int, out_features: int, bias: bool = False):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn_pt.Parameter(torch.randn(out_features, in_features, dtype=torch.float32) * 0.02)
    if bias:
      self.bias = nn_pt.Parameter(torch.zeros(out_features, dtype=torch.float32))
    else:
      self.register_parameter("bias", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear_PT(Linear_PT):
  pass


class RowParallelLinear_PT(Linear_PT):
  pass


class Compressor_PT(nn_pt.Module):
  def __init__(self, rate: int, head_dim: int, args: ModelArgs):
    super().__init__()
    self.rate = rate
    self.head_dim = head_dim
    self.args = args
    if rate > 0:
      proj_dim = (2 if rate == 4 else 1) * head_dim
      self.wkv = Linear_PT(args.dim, proj_dim)
      self.wgate = Linear_PT(args.dim, proj_dim)
      self.norm = RMSNorm_PT(head_dim, args.norm_eps)
      self.ape = nn_pt.Parameter(torch.randn(rate, proj_dim, dtype=torch.float32) * 0.02)
      self.register_buffer(
          "freqs_cis",
          precompute_freqs_cis(
              args.qk_rope_head_dim,
              args.max_seq_len,
              args.rope_theta * (16.0 if rate > 0 else 1.0),
          ),
          persistent=False,
      )

  def overlap_transform(self, tensor: torch.Tensor, value=0.0):
    b, s, _, _ = tensor.size()
    ratio, d = self.rate, self.head_dim
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    if s > 1:
      new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor

  def forward(self, x: torch.Tensor, start_pos: int = 0) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if self.rate == 0:
      return None, None
    bsz, seqlen, _ = x.size()
    ratio, d = self.rate, self.head_dim
    kv = self.wkv(x.float())
    score = self.wgate(x.float())
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    if cutoff == 0:
      return torch.zeros(bsz, 0, d, dtype=x.dtype, device=x.device), None
    if remainder > 0:
      kv = kv[:, :cutoff]
      score = score[:, :cutoff]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + self.ape
    if ratio == 4:
      kv = self.overlap_transform(kv, 0.0)
      score = self.overlap_transform(score, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    kv = self.norm(kv.to(x.dtype))
    freqs = self.freqs_cis[:cutoff:ratio]
    kv = apply_partial_rotary_emb(kv, freqs)
    return kv, freqs


class Indexer_PT(nn_pt.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.wq_b = Linear_PT(args.q_lora_rank, args.index_n_heads * args.index_head_dim)
    self.weights_proj = Linear_PT(args.dim, args.index_n_heads)
    self.compressor = Compressor_PT(4, args.index_head_dim, args)
    self.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(args.qk_rope_head_dim, args.max_seq_len, args.rope_theta),
        persistent=False,
    )

  def forward(self, x: torch.Tensor, q_latent: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    b, m, _ = x.shape
    weights = self.weights_proj(x).float() * (self.args.index_n_heads ** -0.5)
    q = self.wq_b(q_latent).view(b, m, self.args.index_n_heads, self.args.index_head_dim)
    qr = self.freqs_cis[start_pos : start_pos + m]
    q = apply_partial_rotary_emb(q, qr)
    k, _ = self.compressor(x, start_pos)
    n_win = k.size(1) if k is not None else 0
    if n_win == 0:
      return torch.zeros(b, m, 0, dtype=torch.long, device=x.device), torch.zeros(m, 0, dtype=torch.bool, device=x.device)
    # k has shape [b, n_win, index_head_dim]
    scores = torch.einsum("bmhd,btd->bmth", q.float(), k.float()) * (self.args.index_head_dim ** -0.5)
    scores = (scores.relu() * weights.unsqueeze(2)).sum(dim=-1)

    causal_mask = torch.arange(n_win, device=x.device).unsqueeze(0) >= (torch.arange(1, m + 1, device=x.device).unsqueeze(1) // 4)
    scores = scores + torch.where(causal_mask.unsqueeze(0), float("-inf"), 0.0)

    topk_block_idxs = scores.topk(min(self.args.index_topk, n_win), dim=-1).indices
    invalid = causal_mask.unsqueeze(0).expand(b, -1, -1).gather(dim=-1, index=topk_block_idxs)
    final_topk_idxs = torch.where(invalid, torch.full_like(topk_block_idxs, -1), topk_block_idxs)
    return final_topk_idxs, causal_mask


class Attention_PT(nn_pt.Module):
  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.layer_id = layer_id
    self.args = args
    self.rate = args.compress_rates[layer_id]
    self.n_heads = args.n_heads
    self.head_dim = args.head_dim
    self.v_head_dim = args.v_head_dim

    self.wq_a = Linear_PT(args.dim, args.q_lora_rank)
    self.q_norm = RMSNorm_PT(args.q_lora_rank, args.norm_eps)
    self.wq_b = Linear_PT(args.q_lora_rank, args.n_heads * args.head_dim)
    self.wkv = Linear_PT(args.dim, args.head_dim)
    self.kv_norm = RMSNorm_PT(args.qk_nope_head_dim, args.norm_eps)

    self.wo_a = ColumnParallelLinear_PT(args.n_heads * args.v_head_dim // args.o_groups, args.o_groups * args.o_lora_rank)
    self.wo_b = RowParallelLinear_PT(args.o_groups * args.o_lora_rank, args.dim)
    self.attn_sink = nn_pt.Parameter(torch.randn(args.n_heads, dtype=torch.float32) * 0.02)

    self.compressor = Compressor_PT(self.rate, args.v_head_dim, args) if self.rate > 0 else None
    self.indexer = Indexer_PT(args) if self.rate == 4 else None

    self.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(args.qk_rope_head_dim, args.max_seq_len, args.rope_theta),
        persistent=False,
    )

  def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    b, m, _ = x.shape
    q_latent = self.q_norm(self.wq_a(x))
    q = self.wq_b(q_latent).view(b, m, self.n_heads, self.head_dim)
    qr = self.freqs_cis[start_pos : start_pos + m]
    q = apply_partial_rotary_emb(q, qr)
    kv = self.wkv(x)
    if self.args.qk_nope_head_dim > 0:
      kv = self.kv_norm(kv)
    k_roped = apply_partial_rotary_emb(kv.unsqueeze(2), qr).squeeze(2)
    v_unroped = kv

    # For standard/sliding attention, k has head dimension (v_head_dim)
    k_proj = k_roped.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
    v_proj = v_unroped.unsqueeze(2).expand(-1, -1, self.n_heads, -1)

    scale = self.head_dim ** -0.5
    if self.rate == 0:
      # Sliding window causal attention
      scores = torch.einsum("bmhd,bnhd->bmhn", q.float(), k_proj.float()) * scale
      pos_q = torch.arange(start_pos, start_pos + m, device=x.device).unsqueeze(1)
      pos_k = torch.arange(0, m, device=x.device).unsqueeze(0)
      causal_mask = pos_q < pos_k
      sliding_mask = (pos_q - pos_k) >= self.args.sliding_window
      mask = causal_mask | sliding_mask
      scores = torch.where(mask.unsqueeze(1).unsqueeze(0), float("-inf"), scores)
      sink = self.attn_sink.view(1, 1, self.n_heads, 1).expand(b, m, self.n_heads, 1).float()
      all_scores = torch.cat([scores, sink], dim=-1)
      weights = torch.softmax(all_scores, dim=-1)[..., :m]
      out = torch.einsum("bmhn,bnhd->bmhd", weights, v_proj.float()).to(x.dtype)
    elif self.rate == 4:
      # CSA: compound attention with local sliding window + topk compressed blocks
      comp_kv, _ = self.compressor(x, start_pos)
      n_win = comp_kv.size(1) if comp_kv is not None else 0
      if n_win > 0:
        topk_block_idxs, causal_block_mask = self.indexer(x, q_latent, start_pos)
        
        valid = topk_block_idxs >= 0
        entry_indices = torch.arange(n_win, device=x.device).view(1, 1, 1, n_win)
        is_in_topk = topk_block_idxs.unsqueeze(-1) == entry_indices
        is_valid_and_in_topk = is_in_topk & valid.unsqueeze(-1)
        block_selected = is_valid_and_in_topk.any(dim=2)  # [b, m, n_win]

        comp_kv_expanded = comp_kv.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        all_k = torch.cat([k_proj, comp_kv_expanded], dim=1)
        all_v = torch.cat([v_proj, comp_kv_expanded], dim=1)

        scores = torch.einsum("bmhd,bnhd->bmhn", q.float(), all_k.float()) * scale

        pos_q = torch.arange(start_pos, start_pos + m, device=x.device).unsqueeze(1)
        pos_k = torch.arange(0, m, device=x.device).unsqueeze(0)
        causal_mask = pos_q < pos_k
        sliding_mask = (pos_q - pos_k) >= self.args.sliding_window
        uncompressed_mask = causal_mask | sliding_mask

        scores[:, :, :, :m] = torch.where(uncompressed_mask.unsqueeze(0).unsqueeze(2), float("-inf"), scores[:, :, :, :m])
        scores[:, :, :, m:] = torch.where(block_selected.unsqueeze(2), scores[:, :, :, m:], float("-inf"))

        sink = self.attn_sink.view(1, 1, self.n_heads, 1).expand(b, m, self.n_heads, 1).float()
        all_scores = torch.cat([scores, sink], dim=-1)
        weights = torch.softmax(all_scores, dim=-1)[..., : m + n_win]
        out = torch.einsum("bmhn,bnhd->bmhd", weights, all_v.float()).to(x.dtype)
      else:
        scores = torch.einsum("bmhd,bnhd->bmhn", q.float(), k_proj.float()) * scale
        pos_q = torch.arange(start_pos, start_pos + m, device=x.device).unsqueeze(1)
        pos_k = torch.arange(0, m, device=x.device).unsqueeze(0)
        causal_mask = pos_q < pos_k
        sliding_mask = (pos_q - pos_k) >= self.args.sliding_window
        mask = causal_mask | sliding_mask
        scores = torch.where(mask.unsqueeze(1).unsqueeze(0), float("-inf"), scores)
        sink = self.attn_sink.view(1, 1, self.n_heads, 1).expand(b, m, self.n_heads, 1).float()
        all_scores = torch.cat([scores, sink], dim=-1)
        weights = torch.softmax(all_scores, dim=-1)[..., :m]
        out = torch.einsum("bmhn,bnhd->bmhd", weights, v_proj.float()).to(x.dtype)
    else:
      # HCA: compressed attention (rate > 4)
      comp_kv, _ = self.compressor(x, start_pos)
      n_win = comp_kv.size(1) if comp_kv is not None else 0
      if n_win > 0:
        comp_kv_expanded = comp_kv.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        all_k = torch.cat([k_proj, comp_kv_expanded], dim=1)
        all_v = torch.cat([v_proj, comp_kv_expanded], dim=1)

        scores = torch.einsum("bmhd,bnhd->bmhn", q.float(), all_k.float()) * scale

        pos_q = torch.arange(start_pos, start_pos + m, device=x.device).unsqueeze(1)
        pos_k = torch.arange(0, m, device=x.device).unsqueeze(0)
        causal_mask = pos_q < pos_k
        sliding_mask = (pos_q - pos_k) >= self.args.sliding_window
        uncompressed_mask = causal_mask | sliding_mask

        scores[:, :, :, :m] = torch.where(uncompressed_mask.unsqueeze(0).unsqueeze(2), float("-inf"), scores[:, :, :, :m])

        pos_comp_k = torch.arange(0, n_win, device=x.device).unsqueeze(0)
        causal_comp_mask = pos_comp_k >= ((pos_q + 1) // self.rate)
        scores[:, :, :, m:] = torch.where((~causal_comp_mask).unsqueeze(0).unsqueeze(2), scores[:, :, :, m:], float("-inf"))

        sink = self.attn_sink.view(1, 1, self.n_heads, 1).expand(b, m, self.n_heads, 1).float()
        all_scores = torch.cat([scores, sink], dim=-1)
        weights = torch.softmax(all_scores, dim=-1)[..., : m + n_win]
        out = torch.einsum("bmhn,bnhd->bmhd", weights, all_v.float()).to(x.dtype)
      else:
        scores = torch.einsum("bmhd,bnhd->bmhn", q.float(), k_proj.float()) * scale
        pos_q = torch.arange(start_pos, start_pos + m, device=x.device).unsqueeze(1)
        pos_k = torch.arange(0, m, device=x.device).unsqueeze(0)
        causal_mask = pos_q < pos_k
        sliding_mask = (pos_q - pos_k) >= self.args.sliding_window
        mask = causal_mask | sliding_mask
        scores = torch.where(mask.unsqueeze(1).unsqueeze(0), float("-inf"), scores)
        sink = self.attn_sink.view(1, 1, self.n_heads, 1).expand(b, m, self.n_heads, 1).float()
        all_scores = torch.cat([scores, sink], dim=-1)
        weights = torch.softmax(all_scores, dim=-1)[..., :m]
        out = torch.einsum("bmhn,bnhd->bmhd", weights, v_proj.float()).to(x.dtype)

    # Output projection: Grouped wo_a + wo_b
    out = out.reshape(b, m, self.args.o_groups, -1)
    # wo_a is applied per group
    wo_a_w = self.wo_a.weight.reshape(self.args.o_groups, self.args.o_lora_rank, -1)
    out = torch.einsum("bmgi,goi->bmgo", out.float(), wo_a_w.float()).reshape(b, m, -1)
    out = self.wo_b(out.to(x.dtype))
    return out


class Gate_PT(nn_pt.Module):
  def __init__(self, is_hash_layer: bool, args: ModelArgs):
    super().__init__()
    self.is_hash_layer = is_hash_layer
    self.args = args
    self.weight = nn_pt.Parameter(torch.randn(args.n_routed_experts, args.dim, dtype=torch.float32) * 0.02)
    if is_hash_layer:
      tids = torch.stack([
          torch.randperm(args.n_routed_experts)[: args.n_activated_experts] for _ in range(args.vocab_size)
      ]).to(torch.int32)
      self.tid2eid = nn_pt.Parameter(tids, requires_grad=False)
      self.bias = None
    else:
      self.tid2eid = None
      self.bias = nn_pt.Parameter(torch.randn(args.n_routed_experts, dtype=torch.float32) * 0.02)

  def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(x.float(), self.weight.float())
    if self.args.score_func == "softmax":
      scores = scores.softmax(dim=-1)
    elif self.args.score_func == "sigmoid":
      scores = scores.sigmoid()
    elif self.args.score_func == "sqrtsoftplus":
      scores = F.softplus(scores).sqrt()
    original_scores = scores
    if getattr(self, "bias", None) is not None:
      scores = scores + self.bias
    if self.is_hash_layer:
      assert input_ids is not None
      indices = self.tid2eid[input_ids].long()
    else:
      indices = scores.topk(self.args.n_activated_experts, dim=-1).indices
    weights = torch.gather(original_scores, -1, indices)
    if self.args.score_func in ("sigmoid", "sqrtsoftplus"):
      weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * self.args.route_scale
    return weights, indices


class Expert_PT(nn_pt.Module):
  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    super().__init__()
    self.gate_proj = Linear_PT(in_features, hidden_features)
    self.down_proj = Linear_PT(hidden_features, out_features)
    self.up_proj = Linear_PT(in_features, hidden_features)
    self.w1 = self.gate_proj
    self.w2 = self.down_proj
    self.w3 = self.up_proj

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = torch.clamp(self.gate_proj(x), max=10.0)
    up = torch.clamp(self.up_proj(x), min=-10.0, max=10.0)
    return self.down_proj(F.silu(gate) * up)


class MoE_PT(nn_pt.Module):
  def __init__(self, is_hash_layer: bool, args: ModelArgs):
    super().__init__()
    self.args = args
    self.gate = Gate_PT(is_hash_layer, args)
    self.shared_experts = Expert_PT(args.dim, args.n_shared_experts * args.moe_inter_dim, args.dim)
    self.experts = nn_pt.ModuleList([
        Expert_PT(args.dim, args.moe_inter_dim, args.dim) for _ in range(args.n_routed_experts)
    ])

  def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    weights, indices = self.gate(x, input_ids)
    b, m, d = x.shape
    out = torch.zeros_like(x)
    for k in range(indices.shape[-1]):
      idx_k = indices[..., k]
      w_k = weights[..., k].unsqueeze(-1)
      for e in range(self.args.n_routed_experts):
        mask = (idx_k == e)
        if mask.any():
          x_e = x[mask]
          out_e = self.experts[e](x_e)
          out[mask] += out_e * w_k[mask]
    out += self.shared_experts(x)
    return out


class Block_PT(nn_pt.Module):
  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.layer_id = layer_id
    self.args = args
    self.attn_norm = RMSNorm_PT(args.dim, args.norm_eps)
    self.ffn_norm = RMSNorm_PT(args.dim, args.norm_eps)
    self.attn = Attention_PT(layer_id, args)
    self.ffn = MoE_PT(layer_id < args.n_hash_layers, args)

    self.hc_mult = args.hc_mult
    hc_dim = self.hc_mult * args.dim
    mix_hc = (2 + self.hc_mult) * self.hc_mult
    self.hc_attn_fn = nn_pt.Parameter(torch.randn(mix_hc, hc_dim, dtype=torch.float32) * 0.02)
    self.hc_ffn_fn = nn_pt.Parameter(torch.randn(mix_hc, hc_dim, dtype=torch.float32) * 0.02)
    self.hc_attn_base = nn_pt.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
    self.hc_ffn_base = nn_pt.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
    self.hc_attn_scale = nn_pt.Parameter(torch.ones(3, dtype=torch.float32))
    self.hc_ffn_scale = nn_pt.Parameter(torch.ones(3, dtype=torch.float32))

  def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.args.norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn_cpu(
        mixes, hc_scale, hc_base, self.hc_mult, self.args.hc_sinkhorn_iters, self.args.hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb

  def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.to(x.dtype)

  def forward(self, x: torch.Tensor, start_pos: int = 0, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_attn, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    attn_out = self.attn(self.attn_norm(x_attn), start_pos)
    x = self.hc_post(attn_out, x, post, comb)

    x_ffn, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    ffn_out = self.ffn(self.ffn_norm(x_ffn), input_ids)
    x = self.hc_post(ffn_out, x, post, comb)
    return x


class ParallelHead_PT(nn_pt.Module):
  def __init__(self, vocab_size: int, dim: int, args: ModelArgs):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    self.args = args
    self.weight = nn_pt.Parameter(torch.randn(vocab_size, dim, dtype=torch.float32) * 0.02)

  def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.args.norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.args.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype)

  def forward(
      self,
      x: torch.Tensor,
      hc_fn: torch.Tensor,
      hc_scale: torch.Tensor,
      hc_base: torch.Tensor,
      norm: RMSNorm_PT,
  ) -> torch.Tensor:
    x = self.hc_head(x, hc_fn, hc_scale, hc_base)
    x = norm(x)
    return F.linear(x, self.weight)


# ==============================================================================
# 4. Parameter Mapping Transfer Helpers
# ==============================================================================

def _get_nested_pt_attr(obj, path: str):
  """Fetches a nested attribute from a PyTorch module."""
  if path is None:
    return None
  parts = path.split(".")
  curr = obj
  for part in parts:
    if part.isdigit():
      curr = curr[int(part)]
    elif hasattr(curr, part):
      curr = getattr(curr, part)
    elif isinstance(curr, dict) and part in curr:
      curr = curr[part]
    else:
      return None
  return curr


def _apply_global_param_mapping(mt_model, pt_model, pt_config_dict: dict, mx_config: Config):
  mapping = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(pt_config_dict, mx_config, scan_layers=False)
  hooks = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(pt_config_dict, mx_config, scan_layers=False, saving_to_hf=False)

  for mt_key, hf_key in mapping.items():
    if "layers" in mt_key or (hf_key is not None and "layers" in str(hf_key)):
      continue # handled by layer mapper
      
    if hf_key is None:
      continue
      
    # Map directly
    def set_value(obj, val):
      import flax.nnx as nnx
      if isinstance(obj, nnx.Variable):
        obj.value = val
      else:
        obj.value = val

    if mt_key == "params-token_embedder-embedding":
      val = pt_model.embed.weight.detach().numpy()
      target_shape = mt_model.token_embedder.embedding.value.shape
      if mt_key in hooks:
        val = hooks[mt_key](val, target_shape=target_shape)
      set_value(mt_model.token_embedder.embedding, val)
    elif mt_key == "params-decoder-decoder_norm-scale":
      val = pt_model.norm.weight.detach().numpy()
      target_shape = mt_model.decoder.decoder_norm.scale.value.shape
      if mt_key in hooks:
        val = hooks[mt_key](val, target_shape=target_shape)
      set_value(mt_model.decoder.decoder_norm.scale, val)
    elif mt_key == "params-decoder-logits_dense-kernel":
      val = pt_model.head.weight.detach().numpy()
      target_shape = mt_model.decoder.logits_dense.kernel.value.shape
      if mt_key in hooks:
        val = hooks[mt_key](val, target_shape=target_shape)
      set_value(mt_model.decoder.logits_dense.kernel, val)
    elif mt_key == "params-decoder-hc_head-hc_fn":
      if hasattr(pt_model, "hc_head_fn"):
        val = pt_model.hc_head_fn.detach().numpy()
        target_shape = mt_model.decoder.hc_head.hc_fn.value.shape
        if mt_key in hooks:
          val = hooks[mt_key](val, target_shape=target_shape)
        set_value(mt_model.decoder.hc_head.hc_fn, val)
    elif mt_key == "params-decoder-hc_head-hc_base":
      if hasattr(pt_model, "hc_head_base"):
        val = pt_model.hc_head_base.detach().numpy()
        set_value(mt_model.decoder.hc_head.hc_base, val)
    elif mt_key == "params-decoder-hc_head-hc_scale":
      if hasattr(pt_model, "hc_head_scale"):
        val = pt_model.hc_head_scale.detach().numpy()
        set_value(mt_model.decoder.hc_head.hc_scale, val)


def _build_scanned_maxtext_params(pt_model, pt_config_dict: dict, mx_config: Config):
  """Builds a complete Linen params dict for scanned DeepSeek-V4 using checkpoint conversion utilities."""
  from maxtext.checkpoint_conversion.to_maxtext import _get_hf_loading_function
  from maxtext.checkpoint_conversion.utils.utils import param_key_parts_from_path
  from maxtext.models import models
  from maxtext.utils import maxtext_utils

  hf_tensors = {}
  for name, param in pt_model.named_parameters():
    hf_tensors[name] = param.detach().cpu().numpy()
  for name, buf in pt_model.named_buffers():
    hf_tensors[name] = buf.detach().cpu().numpy()

  def tensor_getter(key):
    if key in hf_tensors:
      return hf_tensors[key]
    alt_key = (
        key.replace(".w1.weight", ".gate_proj.weight")
        .replace(".w2.weight", ".down_proj.weight")
        .replace(".w3.weight", ".up_proj.weight")
        .replace(".w1.bias", ".gate_proj.bias")
        .replace(".w2.bias", ".down_proj.bias")
        .replace(".w3.bias", ".up_proj.bias")
    )
    if alt_key in hf_tensors:
      return hf_tensors[alt_key]
    raise KeyError(f"Key {key} (and alt {alt_key}) not found in hf_tensors.")

  mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
  maxtext_model = models.transformer_as_linen(mx_config, mesh, quant=None, model_mode=MODEL_MODE_TRAIN)

  abstract_params_tree = maxtext_utils.get_abstract_param(maxtext_model, mx_config)
  abstract_params_flat, abstract_params_treedef = jax.tree_util.tree_flatten_with_path(
      abstract_params_tree,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )

  maxtext_abstract_dict = {}
  for mt_target_idx, (path_tuple, abstract_leaf_value) in enumerate(abstract_params_flat):
    mt_param_key = "-".join(param_key_parts_from_path(path_tuple))
    if isinstance(abstract_leaf_value, nn.LogicallyPartitioned):
      mt_target_shape = abstract_leaf_value.value.shape
    else:
      mt_target_shape = abstract_leaf_value.shape
    maxtext_abstract_dict[mt_param_key] = (mt_target_idx, mt_target_shape)

  mapping = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(pt_config_dict, mx_config, scan_layers=True)
  hooks = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(pt_config_dict, mx_config, scan_layers=True, saving_to_hf=False)

  final_mt_weights = [None] * len(abstract_params_flat)

  for mt_key, hf_key in mapping.items():
    if mt_key not in maxtext_abstract_dict:
      continue
    target_idx, target_shape = maxtext_abstract_dict[mt_key]
    hook_fn = hooks.get(mt_key, None)
    load_fn = _get_hf_loading_function(hf_key, tensor_getter, hook_fn, target_shape, mx_config, mt_key=mt_key)
    val = load_fn()
    final_mt_weights[target_idx] = val

  params = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  return maxtext_model, params


def _apply_layer_param_mapping(mt_layer, pt_model, layer_idx: int, pt_config_dict: dict, mx_config: Config):
  """Applies parameter mapping from PyTorch Block_PT to MaxText DeepSeek4DecoderLayer."""
  mapping = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(pt_config_dict, mx_config, scan_layers=False)
  hooks = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(pt_config_dict, mx_config, scan_layers=False, saving_to_hf=False)

  pt_prefix = f"layers.{layer_idx}."
  for mt_key, hf_key in mapping.items():
    if f"layers_{layer_idx}" not in mt_key:
      continue

    # Extract target NNX path
    if "Tid2EidVar" in mt_key:
      prefix = f"Tid2EidVar-decoder-layers_{layer_idx}-"
    elif "MoEBiasVar" in mt_key:
      prefix = f"MoEBiasVar-decoder-layers_{layer_idx}-"
    else:
      prefix = f"params-decoder-layers_{layer_idx}-"

    nnx_subpath = mt_key.replace(prefix, "").replace("-", ".")
    parts = nnx_subpath.split(".")
    obj = mt_layer
    valid = True
    for part in parts:
      if hasattr(obj, part):
        obj = getattr(obj, part)
      else:
        valid = False
        break
    if not valid or obj is None or not hasattr(obj, "value"):
      continue

    target_shape = obj.value.shape
    hook_fn = hooks.get(mt_key, lambda x, target_shape=None: x)

    if hf_key is None:
      val = hook_fn(None, target_shape=target_shape)
    elif isinstance(hf_key, list):
      pt_vals = [_get_nested_pt_attr(pt_model, k.replace(pt_prefix, "")) for k in hf_key]
      if any(v is None for v in pt_vals):
        print(f"FAILED LIST LOOKUP for {mt_key}: {hf_key}", flush=True)
        continue
      pt_vals = [v.detach().numpy() for v in pt_vals]
      slice_shape = target_shape[1:]
      processed_vals = [hook_fn(v, target_shape=slice_shape) for v in pt_vals]
      val = np.stack(processed_vals, axis=0)
      print(f"HOOK CALL (LIST): mt_key={mt_key}, val_shape={val.shape}, target_shape={target_shape}", flush=True)
    elif isinstance(hf_key, tuple):
      pt_vals = [_get_nested_pt_attr(pt_model, k.replace(pt_prefix, "")) for k in hf_key]
      if any(v is None for v in pt_vals):
        print(f"FAILED TUPLE LOOKUP for {mt_key}: {hf_key}", flush=True)
        continue
      pt_vals = tuple(v.detach().numpy() for v in pt_vals)
      val = hook_fn(pt_vals, target_shape=target_shape)
      print(f"HOOK CALL (TUPLE): mt_key={mt_key}, val_shape={val.shape}, target_shape={target_shape}", flush=True)
    else:
      pt_attr = _get_nested_pt_attr(pt_model, hf_key.replace(pt_prefix, ""))
      if pt_attr is None:
        print(f"FAILED SINGLE LOOKUP for {mt_key}: {hf_key.replace(pt_prefix, '')}", flush=True)
        continue
      pt_val = pt_attr.detach().numpy()
      # print(f"DEBUG: mt_key={mt_key}, hf_key={hf_key}, pt_val={pt_val.shape}, target_shape={target_shape}", flush=True)
      print(f"HOOK CALL: mt_key={mt_key}, hf_key={hf_key}, pt_shape={pt_val.shape}, target_shape={target_shape}", flush=True)
      val = hook_fn(pt_val, target_shape=target_shape)

    if val is not None:
      setattr(obj, "value", jnp.array(val))


# ==============================================================================
# 5. Unit Tests
# ==============================================================================

class DeepSeekV4FlashEmbeddingTest(unittest.TestCase):
  """Validates ParallelEmbedding_PT against MaxText Embed via parameter mapping."""

  def setUp(self):
    self.args = ModelArgs()
    self.vocab_size = self.args.vocab_size
    self.dim = self.args.dim
    self.mx_config = get_maxtext_config()
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    self.rngs = nnx.Rngs(0)

  def test_embedding_parity(self):
    pt_embed = ParallelEmbedding_PT(self.vocab_size, self.dim)
    mt_embed = Embed(
        config=self.mx_config,
        num_embeddings=self.vocab_size,
        num_features=self.dim,
        dtype=jnp.float32,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Use mapping hook
    hooks = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN({"num_hidden_layers": 5}, self.mx_config)
    unpad_fn = hooks["params-token_embedder-embedding"]
    mt_embed.embedding.value = jnp.array(unpad_fn(pt_embed.weight.detach().numpy(), mt_embed.embedding.value.shape))

    input_ids_np = np.random.randint(0, self.vocab_size, size=(2, 16))
    pt_out = pt_embed(torch.tensor(input_ids_np, dtype=torch.long)).detach().numpy()
    mt_out = np.array(mt_embed(jnp.array(input_ids_np)))

    np.testing.assert_allclose(mt_out, pt_out, rtol=1e-5, atol=1e-5)


class DeepSeekV4FlashRMSNormTest(unittest.TestCase):
  """Validates RMSNorm_PT against MaxText RMSNorm."""

  def setUp(self):
    self.args = ModelArgs()
    self.dim = self.args.dim
    self.eps = self.args.norm_eps
    self.rngs = nnx.Rngs(0)

  def test_rmsnorm_parity(self):
    pt_norm = RMSNorm_PT(self.dim, self.eps)
    mt_norm = RMSNorm(num_features=self.dim, epsilon=self.eps, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=self.rngs)

    mt_norm.scale.value = jnp.array(pt_norm.weight.detach().numpy())

    x_np = np.random.normal(size=(2, 16, self.dim)).astype(np.float32)
    pt_out = pt_norm(torch.tensor(x_np)).detach().numpy()
    mt_out = np.array(mt_norm(jnp.array(x_np)))

    np.testing.assert_allclose(mt_out, pt_out, rtol=1e-5, atol=1e-5)


class DeepSeekV4FlashRotaryEmbeddingTest(unittest.TestCase):
  """Validates apply_rotary_emb / precompute_freqs_cis against DeepSeekV4RotaryEmbedding."""

  def setUp(self):
    self.args = ModelArgs()
    self.head_dim = self.args.head_dim
    self.qk_rope_head_dim = self.args.qk_rope_head_dim
    self.seq_len = 32
    self.batch_size = 2
    self.num_heads = self.args.n_heads

  def test_main_rope(self):
    self._run_rope_test(theta=10000.0)

  def test_compressed_rope(self):
    self._run_rope_test(theta=160000.0)

  def _run_rope_test(self, theta: float):
    freqs_cis = precompute_freqs_cis(self.qk_rope_head_dim, self.seq_len, theta)
    mt_rope = DeepSeekV4RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=self.qk_rope_head_dim / self.head_dim,
        rope_theta=theta,
    )

    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32)
    pos_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)

    # PyTorch reference
    x_pt = torch.tensor(x_np)
    qr = freqs_cis[pos_np[0]].unsqueeze(1)  # [seq_len, 1, qk_rope_head_dim//2]
    pt_out = apply_partial_rotary_emb(x_pt, qr).detach().numpy()

    # MaxText
    mt_out = np.array(mt_rope(jnp.array(x_np), jnp.array(pos_np), unsqueeze_dim=2))

    np.testing.assert_allclose(mt_out, pt_out, rtol=1e-5, atol=1e-5)


class DeepSeekV4FlashGroupedLinearTest(unittest.TestCase):
  """Validates ColumnParallelLinear_PT (wo_a) against MaxText DeepSeekV4GroupedLinear."""

  def setUp(self):
    self.args = ModelArgs()
    self.in_features_per_group = (self.args.n_heads * self.args.v_head_dim) // self.args.o_groups
    self.out_features_per_group = self.args.o_lora_rank
    self.n_groups = self.args.o_groups
    self.total_out_features = self.n_groups * self.out_features_per_group
    self.rngs = nnx.Rngs(0)
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))

  def test_grouped_linear_parity(self):
    pt_linear = ColumnParallelLinear_PT(self.in_features_per_group, self.total_out_features)
    mt_linear = DeepSeekV4GroupedLinear(
        in_features_per_group=self.in_features_per_group,
        out_features=self.total_out_features,
        n_groups=self.n_groups,
        rngs=self.rngs,
    )

    # Use hook reshape_o_a_proj
    hooks = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN({"num_hidden_layers": 5}, None)
    reshape_fn = hooks["params-decoder-layers_0-self_attention-o_a_proj-kernel"]
    mt_linear.kernel.value = jnp.array(reshape_fn(pt_linear.weight.detach().numpy(), mt_linear.kernel.value.shape))

    x_np = np.random.normal(size=(2, 16, self.n_groups, self.in_features_per_group)).astype(np.float32)
    # PyTorch evaluation per group
    w_groups = pt_linear.weight.view(self.n_groups, self.out_features_per_group, self.in_features_per_group)
    pt_out = torch.einsum("bmgi,goi->bmgo", torch.tensor(x_np), w_groups).detach().numpy()

    mt_out = np.array(mt_linear(jnp.array(x_np)))

    np.testing.assert_allclose(mt_out, pt_out, rtol=1e-5, atol=1e-5)


class DeepSeekV4FlashMoEGateTest(unittest.TestCase):
  """Validates Gate_PT (Hash and TopK) against MaxText RoutedMoE gate."""

  def setUp(self):
    self.args = ModelArgs()
    self.dim = self.args.dim
    self.vocab_size = self.args.vocab_size
    self.num_experts = self.args.n_routed_experts
    self.num_activated = self.args.n_activated_experts
    self.mx_config = get_maxtext_config()
    self.args.score_func = self.mx_config.routed_score_func
    self.args.route_scale = self.mx_config.routed_scaling_factor
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    self.rngs = nnx.Rngs(0)

  def test_hash_routing_gate_parity(self):
    pt_gate = Gate_PT(is_hash_layer=True, args=self.args)
    mt_moe = RoutedMoE(
        config=self.mx_config,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_activated,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", None),
        is_hash_routing=True,
        rngs=self.rngs,
    )

    # Copy weights & tid2eid
    mt_moe.gate.kernel.value = jnp.array(pt_gate.weight.detach().numpy().T)
    mt_moe.tid2eid.value = jnp.array(pt_gate.tid2eid.detach().numpy())

    input_ids_np = np.random.randint(0, self.vocab_size, size=(2, 16))
    x_np = np.random.normal(size=(2, 16, self.dim)).astype(np.float32)

    pt_weights, pt_indices = pt_gate(torch.tensor(x_np), torch.tensor(input_ids_np, dtype=torch.long))

    gate_logits, pre_bias_logits = mt_moe.gate(jnp.array(x_np))
    mt_weights, mt_indices = mt_moe.get_topk(
        gate_logits, pre_bias_logits, rngs=self.rngs, input_ids=jnp.array(input_ids_np)
    )

    np.testing.assert_array_equal(np.array(mt_indices), pt_indices.detach().numpy())
    np.testing.assert_allclose(np.array(mt_weights), pt_weights.detach().numpy(), rtol=1e-5, atol=1e-5)


class DeepSeekV4FlashMoEBlockTest(unittest.TestCase):
  """Validates full MoE_PT (Hash and TopK) against MaxText RoutedAndSharedMoE."""

  def setUp(self):
    self.args = ModelArgs()
    self.dim = self.args.dim
    self.inter_dim = self.args.inter_dim
    self.vocab_size = self.args.vocab_size
    self.num_experts = self.args.n_routed_experts
    self.num_activated = self.args.n_activated_experts
    self.mx_config = get_maxtext_config()
    self.args.score_func = self.mx_config.routed_score_func
    self.args.route_scale = self.mx_config.routed_scaling_factor
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    self.rngs = nnx.Rngs(0)

  def test_hash_routing_moe_parity(self):
    pt_moe = MoE_PT(is_hash_layer=True, args=self.args)
    mt_moe = RoutedAndSharedMoE(
        config=self.mx_config,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", None),
        rngs=self.rngs,
        is_hash_routing=True,
    )

    class DummyLayer:
      def __init__(self, moe):
        self.mlp = moe

    class DummyPTLayer:
      def __init__(self, moe):
        self.ffn = moe

    pt_config_dict = {
        "num_hidden_layers": 1,
        "first_k_dense_replace": 0,
        "n_routed_experts": self.args.n_routed_experts,
        "num_experts_per_tok": self.args.n_activated_experts,
        "first_num_hash_layers": 1,
    }
    _apply_layer_param_mapping(DummyLayer(mt_moe), DummyPTLayer(pt_moe), layer_idx=0, pt_config_dict=pt_config_dict, mx_config=self.mx_config)

    input_ids_np = np.random.randint(0, self.vocab_size, size=(2, 16))
    x_np = np.random.normal(size=(2, 16, self.dim)).astype(np.float32)

    pt_out = pt_moe(torch.tensor(x_np), torch.tensor(input_ids_np, dtype=torch.long)).detach().numpy()
    mt_out, _, _ = mt_moe(jnp.array(x_np), input_ids=jnp.array(input_ids_np))

    np.testing.assert_allclose(np.array(mt_out), pt_out, rtol=5e-2, atol=5e-2)


class DeepSeekV4FlashHyperHeadTest(unittest.TestCase):
  """Validates ParallelHead_PT.hc_head against MaxText DeepSeek4HyperHead."""

  def setUp(self):
    self.args = ModelArgs()
    self.dim = self.args.dim
    self.hc_mult = self.args.hc_mult
    self.vocab_size = self.args.vocab_size
    self.mx_config = get_maxtext_config()
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    self.rngs = nnx.Rngs(0)

  def test_hyperhead_parity(self):
    pt_head = ParallelHead_PT(self.vocab_size, self.dim, self.args)
    hc_fn = torch.randn(self.hc_mult, self.hc_mult * self.dim, dtype=torch.float32) * 0.02
    hc_base = torch.zeros(self.hc_mult, dtype=torch.float32)
    hc_scale = torch.ones(1, dtype=torch.float32)

    mt_head = DeepSeek4HyperHead(
        config=self.mx_config,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    mt_head.hc_fn.value = jnp.array(hc_fn.detach().numpy().T)
    mt_head.hc_base.value = jnp.array(hc_base.detach().numpy())
    mt_head.hc_scale.value = jnp.array(hc_scale.detach().numpy())

    x_np = np.random.normal(size=(2, 16, self.hc_mult, self.dim)).astype(np.float32)
    pt_out = pt_head.hc_head(torch.tensor(x_np), hc_fn, hc_scale, hc_base).detach().numpy()
    mt_out = np.array(mt_head(jnp.array(x_np)))

    np.testing.assert_allclose(mt_out, pt_out, rtol=2e-5, atol=2e-5)


class DeepSeekV4FlashDecoderLayerTest(unittest.TestCase):
  """Validates Block_PT vs MaxText DeepSeek4DecoderLayer across all layer types using param_mapping.py."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 16
    self.args = ModelArgs()
    self.dim = self.args.dim
    self.vocab_size = self.args.vocab_size
    self.mx_config = get_maxtext_config()
    self.args.score_func = self.mx_config.routed_score_func
    self.args.route_scale = self.mx_config.routed_scaling_factor
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    self.rngs = nnx.Rngs(0)

  def _run_layer_test(self, layer_idx: int):
    pt_block = Block_PT(layer_id=layer_idx, args=self.args)
    mt_layer = DeepSeek4DecoderLayer(
        config=self.mx_config,
        model_mode="train",
        mesh=self.mesh,
        rngs=self.rngs,
        layer_idx=layer_idx,
        compress_ratio=self.args.compress_rates[layer_idx],
        is_hash_routing=(layer_idx < self.args.n_hash_layers),
    )

    pt_config_dict = {
        "num_hidden_layers": self.args.n_layers,
        "first_k_dense_replace": 0,
        "n_routed_experts": self.args.n_routed_experts,
        "num_experts_per_tok": self.args.n_activated_experts,
        "first_num_hash_layers": self.args.n_hash_layers,
    }
    _apply_layer_param_mapping(mt_layer, pt_block, layer_idx, pt_config_dict, self.mx_config)

    x_np = np.random.uniform(0.1, 1.0, size=(self.batch_size, self.seq_len, self.args.hc_mult, self.dim)).astype(np.float32)
    pos_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)
    input_ids_np = np.random.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len))

    # PyTorch forward
    pt_out = pt_block(
        x=torch.tensor(x_np),
        start_pos=0,
        input_ids=torch.tensor(input_ids_np, dtype=torch.long),
    ).detach().numpy()

    # MaxText forward
    mt_out, _ = mt_layer(
        inputs=jnp.array(x_np),
        decoder_segment_ids=jnp.ones_like(pos_np, dtype=jnp.int32),
        decoder_positions=jnp.array(pos_np),
        deterministic=True,
        model_mode="train",
        decoder_input_tokens=jnp.array(input_ids_np),
    )
    mt_out_np = np.array(mt_out)

    max_diff = np.max(np.abs(mt_out_np - pt_out))
    mean_diff = np.mean(np.abs(mt_out_np - pt_out))
    print(f"Layer {layer_idx} Parity -> max_diff: {max_diff:.6e}, mean_diff: {mean_diff:.6e}")
    np.testing.assert_allclose(mt_out_np, pt_out, rtol=8e-2, atol=8e-2)

  def test_layer_0_sliding_hash(self):
    self._run_layer_test(0)

  def test_layer_2_csa_hash(self):
    self._run_layer_test(2)

  def test_layer_3_hca_topk(self):
    self._run_layer_test(3)

  def test_layer_4_csa_topk(self):
    self._run_layer_test(4)


# ==============================================================================
# Full Model Functional & Parity Test
# ==============================================================================
from maxtext.models.models import Transformer

class Transformer_PT(nn_pt.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.n_layers = args.n_layers
    self.embed = ParallelEmbedding_PT(args.vocab_size, args.dim)
    self.layers = nn_pt.ModuleList([Block_PT(i, args) for i in range(args.n_layers)])
    self.norm = RMSNorm_PT(args.dim, args.norm_eps)
    self.head = ParallelHead_PT(args.vocab_size, args.dim, args)
    if args.hc_mult > 1:
      self.hc_head_fn = nn_pt.Parameter(torch.randn(args.hc_mult, args.hc_mult * args.dim, dtype=torch.float32) * 0.02)
      self.hc_head_base = nn_pt.Parameter(torch.zeros(args.hc_mult, dtype=torch.float32))
      self.hc_head_scale = nn_pt.Parameter(torch.ones(1, dtype=torch.float32))

  def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
    h = self.embed(input_ids)
    if self.args.hc_mult > 1:
      h = h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1)
    for layer in self.layers:
      h = layer(h, start_pos, input_ids)
    
    if self.args.hc_mult > 1:
      logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
    else:
      h_norm = self.norm(h)
      logits = F.linear(h_norm, self.head.weight)
    
    return logits


class DeepSeekV4FlashFullModelTest(unittest.TestCase):
  """Validates full PyTorch Transformer_PT against MaxText Transformer (DeepSeek4)."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 8
    self.vocab_size = 256
    self.args = ModelArgs(vocab_size=self.vocab_size)
    self.dim = self.args.dim
    self.mx_config = get_maxtext_config(vocab_size=self.vocab_size)

  def test_full_model_parity(self):
    rng = jax.random.PRNGKey(0)
    pt_config_dict = {"num_hidden_layers": 5, "num_hash_layers": 3}

    pt_model = Transformer_PT(self.args)
    pt_model.eval()

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",))
    mt_model = Transformer(
        config=self.mx_config,
        mesh=mesh,
        quant=None,
        model_mode="train",
        rngs=nnx.Rngs(0),
    )

    input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])

    # Map Decoder Layers
    for i in range(self.args.n_layers):
        _apply_layer_param_mapping(getattr(mt_model.decoder, f'layers_{i}'), pt_model.layers[i], i, pt_config_dict, self.mx_config)
        
    _apply_global_param_mapping(mt_model, pt_model, pt_config_dict, self.mx_config)

    # Do forward pass
    pt_model.eval()
    import torch
    with torch.no_grad():
        logits_pt = pt_model(torch.tensor(input_ids.tolist(), dtype=torch.long))
    
    mt_out = mt_model(
        decoder_input_tokens=input_ids,
        decoder_positions=jnp.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0),
        model_mode="train",
    )
    # 1. Embedding check
    pt_emb = pt_model.embed(torch.tensor(input_ids.tolist(), dtype=torch.long)).detach().numpy()
    mt_emb = np.array(mt_model.token_embedder(input_ids))
    print(f"EMBEDDING DIFF: max={np.max(np.abs(pt_emb - mt_emb)):.6e}", flush=True)

    # Compare Layer 0 sub-blocks
    pt_block = pt_model.layers[0]
    mt_block = getattr(mt_model.decoder, "layers_0")
    
    pt_h = torch.tensor(pt_emb)
    if self.args.hc_mult > 1:
      pt_h = pt_h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1)

    positions = jnp.broadcast_to(jnp.arange(self.seq_len, dtype=jnp.int32)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32)

    # 2a. Attention pre-mhc
    pt_x_attn, pt_post, pt_comb = pt_block.hc_pre(pt_h, pt_block.hc_attn_fn, pt_block.hc_attn_scale, pt_block.hc_attn_base)
    print(f"PT HC_PRE ATTN: {pt_x_attn.shape}", flush=True)

    # 2b. Compare Attention core
    pt_attn_in = pt_block.attn_norm(pt_x_attn)
    pt_attn_out = pt_block.attn(pt_attn_in, 0).detach().numpy()
    
    mt_attn_in = np.array(mt_block.pre_self_attention_layer_norm(jnp.array(pt_x_attn.detach().numpy())))
    print(f"ATTN NORM DIFF: max={np.max(np.abs(pt_attn_in.detach().numpy() - mt_attn_in)):.6e}", flush=True)

    # Q projection
    pt_q_latent = pt_block.attn.q_norm(pt_block.attn.wq_a(pt_attn_in))
    mt_q_latent = mt_block.self_attention.q_norm(mt_block.self_attention.wq_a(jnp.array(mt_attn_in)))
    print(f"ATTN Q_LATENT DIFF: max={np.max(np.abs(pt_q_latent.detach().numpy() - np.array(mt_q_latent))):.6e}", flush=True)

    pt_q = pt_block.attn.wq_b(pt_q_latent).view(self.batch_size, self.seq_len, self.args.n_heads, self.args.head_dim)
    mt_q = mt_block.self_attention.wq_b(mt_q_latent)
    print(f"ATTN Q_UP DIFF: max={np.max(np.abs(pt_q.detach().numpy() - np.array(mt_q))):.6e}", flush=True)

    # Q RoPE
    qr_pt = pt_block.attn.freqs_cis[0 : self.seq_len]
    pt_q_roped = apply_partial_rotary_emb(pt_q, qr_pt)
    mt_q_roped = mt_block.self_attention._apply_rotary_embedding_v4(mt_q, positions, unsqueeze_dim=-2)
    print(f"ATTN Q_ROPED DIFF: max={np.max(np.abs(pt_q_roped.detach().numpy() - np.array(mt_q_roped))):.6e}", flush=True)

    # KV projection
    pt_kv = pt_block.attn.wkv(pt_attn_in)
    if self.args.qk_nope_head_dim > 0:
      pt_kv = pt_block.attn.kv_norm(pt_kv)
    pt_k_roped = apply_partial_rotary_emb(pt_kv.unsqueeze(2), qr_pt).squeeze(2).unsqueeze(2).expand(-1, -1, self.args.n_heads, -1)
    
    mt_k_roped, _ = mt_block.self_attention.compressed_kv_projection(jnp.array(mt_attn_in), positions, "train")
    print(f"ATTN KV_ROPED DIFF: max={np.max(np.abs(pt_k_roped.detach().numpy() - np.array(mt_k_roped))):.6e}", flush=True)
    
    # Test CompressedAttention directly
    mt_attn_out, _ = mt_block.self_attention(
        inputs_q=jnp.array(pt_attn_in.detach().numpy()),
        inputs_kv=jnp.array(pt_attn_in.detach().numpy()),
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode="train",
    )
    mt_attn_out = np.array(mt_attn_out)
    print(f"ATTENTION CORE DIFF: max={np.max(np.abs(pt_attn_out - mt_attn_out)):.6e}", flush=True)

    # 2c. Compare MoE core
    pt_x_ffn, _, _ = pt_block.hc_pre(pt_h, pt_block.hc_ffn_fn, pt_block.hc_ffn_scale, pt_block.hc_ffn_base)
    pt_ffn_in = pt_block.ffn_norm(pt_x_ffn)
    pt_ffn_out = pt_block.ffn(pt_ffn_in, torch.tensor(input_ids.tolist(), dtype=torch.long)).detach().numpy()

    mt_ffn_in = np.array(mt_block.post_self_attention_layer_norm(jnp.array(pt_x_ffn.detach().numpy())))
    print(f"FFN NORM DIFF: max={np.max(np.abs(pt_ffn_in.detach().numpy() - mt_ffn_in)):.6e}", flush=True)

    # Compare Shared Experts
    pt_shared = pt_block.ffn.shared_experts(pt_ffn_in).detach().numpy()
    mt_shared = np.array(mt_block.mlp.shared_experts(jnp.array(pt_ffn_in.detach().numpy())))
    print(f"MOE SHARED EXPERTS DIFF: max={np.max(np.abs(pt_shared - mt_shared)):.6e}", flush=True)

    # Compare Gate / Routing
    pt_w, pt_idx = pt_block.ffn.gate(pt_ffn_in, torch.tensor(input_ids.tolist(), dtype=torch.long))
    gate_in = jnp.array(pt_ffn_in.detach().numpy())
    mt_gl, mt_pbl = mt_block.mlp.routed_moe.gate(gate_in)
    mt_w, mt_idx = mt_block.mlp.routed_moe.get_topk(mt_gl, mt_pbl, input_ids=input_ids)
    print(f"GATE INDICES MATCH: {np.array_equal(pt_idx.detach().numpy(), np.array(mt_idx))}", flush=True)
    print(f"GATE WEIGHTS DIFF: max={np.max(np.abs(pt_w.detach().numpy() - np.array(mt_w))):.6e}", flush=True)

    # Compare Routed Experts
    mt_routed, _, _ = mt_block.mlp.routed_moe(gate_in, input_ids=input_ids)
    pt_routed = (pt_block.ffn(pt_ffn_in, torch.tensor(input_ids.tolist(), dtype=torch.long)) - pt_block.ffn.shared_experts(pt_ffn_in)).detach().numpy()
    print(f"MOE ROUTED EXPERTS DIFF: max={np.max(np.abs(pt_routed - np.array(mt_routed))):.6e}", flush=True)

    mt_ffn_out, _, _ = mt_block.mlp(
        jnp.array(pt_ffn_in.detach().numpy()),
        input_ids=input_ids,
    )
    print(f"MOE CORE DIFF: max={np.max(np.abs(pt_ffn_out - np.array(mt_ffn_out))):.6e}", flush=True)

    # Layer by Layer comparison
    pt_curr_h = pt_h.clone()
    mt_curr_h = jnp.array(pt_h.detach().numpy())
    for layer_i in range(self.args.n_layers):
      pt_layer = pt_model.layers[layer_i]
      mt_layer = getattr(mt_model.decoder, f"layers_{layer_i}")
      
      if layer_i == 2:
        # Step by step layer 2 breakdown
        l2_pt_x_attn, _, _ = pt_layer.hc_pre(pt_curr_h, pt_layer.hc_attn_fn, pt_layer.hc_attn_scale, pt_layer.hc_attn_base)
        l2_pt_attn_in = pt_layer.attn_norm(l2_pt_x_attn)
        l2_mt_attn_in = np.array(mt_layer.pre_self_attention_layer_norm(jnp.array(l2_pt_x_attn.detach().numpy())))
        print(f"L2 ATTN NORM DIFF: {np.max(np.abs(l2_pt_attn_in.detach().numpy() - l2_mt_attn_in)):.6e}", flush=True)

        l2_pt_q_latent = pt_layer.attn.q_norm(pt_layer.attn.wq_a(l2_pt_attn_in))
        l2_mt_q_latent = mt_layer.self_attention.q_norm(mt_layer.self_attention.wq_a(jnp.array(l2_mt_attn_in)))
        print(f"L2 Q_LATENT DIFF: {np.max(np.abs(l2_pt_q_latent.detach().numpy() - np.array(l2_mt_q_latent))):.6e}", flush=True)

        l2_pt_comp_kv, _ = pt_layer.attn.compressor(l2_pt_attn_in, 0)
        l2_mt_comp_kv, l2_mt_comp_mask = mt_layer.self_attention.csa_compressor(
            jnp.array(l2_pt_attn_in.detach().numpy()),
            jnp.array(l2_pt_q_latent.detach().numpy()),
            positions,
            None,
            "train",
        )
        print(f"L2 COMP KV DIFF: {np.max(np.abs(l2_pt_comp_kv.detach().numpy() - np.array(l2_mt_comp_kv[:, :, 0, :]))):.6e}", flush=True)

        l2_pt_topk, l2_pt_mask = pt_layer.attn.indexer(l2_pt_attn_in, l2_pt_q_latent, 0)
        l2_mt_topk = mt_layer.self_attention.csa_compressor.indexer(
            jnp.array(l2_pt_attn_in.detach().numpy()),
            jnp.array(l2_pt_q_latent.detach().numpy()),
            positions,
            None,
            "train",
        )
        print(f"L2 INDEXER TOPK MATCH: {np.array_equal(l2_pt_topk.detach().numpy(), np.array(l2_mt_topk))}", flush=True)
        print(f"L2 PT TOPK:\n{l2_pt_topk}", flush=True)
        print(f"L2 MT TOPK:\n{l2_mt_topk}", flush=True)

        l2_pt_attn_out = pt_layer.attn(l2_pt_attn_in, 0).detach().numpy()
        l2_mt_attn_out, _ = mt_layer.self_attention(
            inputs_q=jnp.array(l2_pt_attn_in.detach().numpy()),
            inputs_kv=jnp.array(l2_pt_attn_in.detach().numpy()),
            decoder_segment_ids=segment_ids,
            inputs_positions=positions,
            deterministic=True,
            model_mode="train",
        )
        print(f"L2 ATTN OUT DIFF: {np.max(np.abs(l2_pt_attn_out - np.array(l2_mt_attn_out))):.6e}", flush=True)

      pt_curr_h = pt_layer(pt_curr_h, 0, torch.tensor(input_ids.tolist(), dtype=torch.long))
      mt_curr_h, _ = mt_layer(
          mt_curr_h,
          decoder_segment_ids=segment_ids,
          decoder_positions=positions,
          deterministic=True,
          model_mode="train",
          decoder_input_tokens=input_ids,
      )
      diff = np.max(np.abs(pt_curr_h.detach().numpy() - np.array(mt_curr_h)))
      print(f"LAYER {layer_i} [rate={self.args.compress_rates[layer_i]}]: OUTPUT DIFF: max={diff:.6e}", flush=True)

    # Head comparison
    pt_head_out = pt_model.head.hc_head(pt_curr_h, pt_model.hc_head_fn, pt_model.hc_head_scale, pt_model.hc_head_base)
    mt_head_out = mt_model.decoder.hc_head(mt_curr_h)
    print(f"pt_head_out: shape={pt_head_out.shape}, min={pt_head_out.min():.4f}, max={pt_head_out.max():.4f}", flush=True)
    print(f"mt_head_out: shape={mt_head_out.shape}, min={mt_head_out.min():.4f}, max={mt_head_out.max():.4f}", flush=True)
    print(f"HC_HEAD DIFF: max={np.max(np.abs(pt_head_out.detach().numpy() - np.array(mt_head_out))):.6e}", flush=True)

    pt_norm_out = pt_model.norm(pt_head_out)
    mt_norm_out = mt_model.decoder.decoder_norm(mt_head_out)
    print(f"pt_norm_out: shape={pt_norm_out.shape}, min={pt_norm_out.min():.4f}, max={pt_norm_out.max():.4f}", flush=True)
    print(f"mt_norm_out: shape={mt_norm_out.shape}, min={mt_norm_out.min():.4f}, max={mt_norm_out.max():.4f}", flush=True)
    print(f"DECODER NORM DIFF: max={np.max(np.abs(pt_norm_out.detach().numpy() - np.array(mt_norm_out))):.6e}", flush=True)

    pt_logits = F.linear(pt_norm_out, pt_model.head.weight)
    mt_logits = mt_model.decoder.logits_dense(mt_norm_out)
    print(f"LOGITS DIFF: max={np.max(np.abs(pt_logits.detach().numpy() - np.array(mt_logits))):.6e}", flush=True)

    def assert_close(a, b, name):
        max_diff = jnp.max(jnp.abs(a - b))
        mean_diff = jnp.mean(jnp.abs(a - b))
        print(f"{name} Parity -> max_diff: {max_diff:e}, mean_diff: {mean_diff:e}", flush=True)
        np.testing.assert_allclose(a, b, atol=1e-2, rtol=1e-2)
        
    assert_close(np.array(mt_out), logits_pt.detach().numpy(), "Full Model Logits")

  def test_scanned_full_model_parity(self):
    """Validates full PyTorch Transformer_PT against MaxText Transformer (DeepSeek4) with scan_layers=True."""
    # 7 layers: 3 prefix [0, 0, 4] + 4 scanned (2 blocks of [128, 4])
    n_layers = 7
    compress_rates = [0, 0, 4, 128, 4, 128, 4]
    pt_config_dict = {
        "num_hidden_layers": n_layers,
        "num_hash_layers": 3,
    }

    test_vocab_size = 256
    scanned_args = ModelArgs(
        vocab_size=test_vocab_size,
        n_layers=n_layers,
        compress_rates=compress_rates,
        n_hash_layers=3,
        n_routed_experts=4,
        n_activated_experts=2,
    )

    scanned_mx_config = get_maxtext_config(
        vocab_size=test_vocab_size,
        base_num_decoder_layers=n_layers,
        compress_ratios=compress_rates,
        first_num_hash_layers=3,
        num_experts=4,
        num_experts_per_tok=2,
        scan_layers=True,
    )

    pt_model = Transformer_PT(scanned_args)
    pt_model.eval()

    maxtext_model, params = _build_scanned_maxtext_params(pt_model, pt_config_dict, scanned_mx_config)

    input_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])

    # PyTorch Forward Pass
    with torch.no_grad():
      logits_pt = pt_model(torch.tensor(input_ids.tolist(), dtype=torch.long))

    # MaxText Linen Forward Pass
    positions = jnp.broadcast_to(jnp.arange(self.seq_len, dtype=jnp.int32)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32)

    @jax.jit
    def run_mt(p, tokens, pos, seg):
      return maxtext_model.apply(
          p,
          decoder_input_tokens=tokens,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          enable_dropout=False,
          model_mode="train",
          decoder_target_tokens=tokens,
          decoder_target_mask=seg,
      )

    mt_out = run_mt(params, input_ids, positions, segment_ids)

    def assert_close(a, b, name):
      max_diff = jnp.max(jnp.abs(a - b))
      mean_diff = jnp.mean(jnp.abs(a - b))
      print(f"{name} Parity -> max_diff: {max_diff:e}, mean_diff: {mean_diff:e}", flush=True)
      np.testing.assert_allclose(a, b, atol=1e-2, rtol=1e-2)

    assert_close(np.array(mt_out), logits_pt.detach().numpy(), "Scanned Full Model Logits")


if __name__ == "__main__":
  unittest.main()



