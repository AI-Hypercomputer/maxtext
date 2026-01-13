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


from types import SimpleNamespace
import os.path
import unittest

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import jax
from jax.sharding import Mesh
import jax.numpy as jnp

from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.layers import attentions, moe, embeddings, attention_mla
from MaxText.layers.initializers import nd_dense_init
from flax import nnx
import chex


"""
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pytest -v --pyargs tests.check_deepseek32_vs_reference -rP -s 
python3 -m pytest -v --pyargs tests.check_deepseek32_vs_reference -rP -s -k "DeepseekV32IndexerTest"
python3 -m pytest -v --pyargs tests.check_deepseek32_vs_reference -rP -s -k "DeepseekV32MLATest"
https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


world_size = 1
rank = 0
block_size = 128


from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN


@dataclass
class ModelArgs:
  pass
#   """
#   Data class for defining model arguments and hyperparameters.
#   Attributes:
#       max_batch_size (int): Maximum batch size.
#       max_seq_len (int): Maximum sequence length.
#       dtype (Literal["bf16", "fp8"]): Data type for computations.
#       scale_fmt (Optional[str]): Format for quantization scale.
#       vocab_size (int): Vocabulary size.
#       dim (int): Model dimension.
#       inter_dim (int): Intermediate dimension for MLP layers.
#       moe_inter_dim (int): Intermediate dimension for MoE layers.
#       n_layers (int): Number of transformer layers.
#       n_dense_layers (int): Number of dense layers in the model.
#       n_heads (int): Number of attention heads.
#       n_routed_experts (int): Number of routed experts for MoE layers.
#       n_shared_experts (int): Number of shared experts for MoE layers.
#       n_activated_experts (int): Number of activated experts in MoE layers.
#       n_expert_groups (int): Number of expert groups.
#       n_limited_groups (int): Number of limited groups for MoE routing.
#       score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
#       route_scale (float): Scaling factor for routing scores.
#       q_lora_rank (int): LoRA rank for query projections.
#       kv_lora_rank (int): LoRA rank for key-value projections.
#       qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
#       qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
#       v_head_dim (int): Dimension for value projections.
#       original_seq_len (int): Original sequence length.
#       rope_theta (float): Base for rotary positional encoding.
#       rope_factor (float): Scaling factor for extended sequence lengths.
#       beta_fast (int): Fast beta correction factor.
#       beta_slow (int): Slow beta correction factor.
#       mscale (float): Scaling factor for extended attention.
#       index_head_dim (int): Dimension for index head.
#       index_topk (int): Top-k for index head.
#   """

#   max_batch_size: int = 8
#   max_seq_len: int = 4096 * 4
#   dtype: Literal["bf16", "fp8"] = "bf16"
#   scale_fmt: Optional[str] = None
#   vocab_size: int = 102400
#   dim: int = 2048
#   inter_dim: int = 10944
#   moe_inter_dim: int = 1408
#   n_layers: int = 27
#   n_dense_layers: int = 1
#   n_heads: int = 16
#   # moe
#   n_routed_experts: int = 64
#   n_shared_experts: int = 2
#   n_activated_experts: int = 6
#   n_expert_groups: int = 1
#   n_limited_groups: int = 1
#   score_func: Literal["softmax", "sigmoid"] = "softmax"
#   route_scale: float = 1.0
#   # mla
#   q_lora_rank: int = 0
#   kv_lora_rank: int = 512
#   qk_nope_head_dim: int = 128
#   qk_rope_head_dim: int = 64
#   v_head_dim: int = 128
#   # yarn
#   original_seq_len: int = 4096
#   rope_theta: float = 10000.0
#   rope_factor: float = 40
#   beta_fast: int = 32
#   beta_slow: int = 1
#   mscale: float = 1.0
#   # index
#   index_n_heads: int = 64
#   index_head_dim: int = 128
#   index_topk: int = 2048


class Config:
  """A configuration class for holding hyperparameters for the tests."""

  base_mlp_dim = 18432
  base_moe_mlp_dim = 2048
  base_num_decoder_layers = 61
  first_num_dense_layers = 3
  mlp_activations = ["silu", "linear"]
  vocab_size = 129280
  enable_dropout = False
  logits_via_embedding = False
  normalization_layer_epsilon = 1.0e-6
  num_experts = 256
  num_experts_per_tok = 8
  shared_experts = 1
  routed_scaling_factor = 2.5
  routed_score_func = "sigmoid"
  routed_bias = True
  decoder_block = "deepseek"
  # MLA
  base_emb_dim = 7168
  base_num_query_heads = 128
  base_num_kv_heads = 128
  attention_type = "mla"
  q_lora_rank = 1536
  kv_lora_rank = 512
  qk_nope_head_dim = 128
  qk_rope_head_dim = 64
  v_head_dim = 128
  # RoPE
  rope_type = "yarn"
  original_max_position_embeddings = 4096
  rope_max_timescale = 10_000
  max_position_embeddings = 163840
  rope_factor = 40
  beta_fast = 32
  beta_slow = 1
  mscale = 1.0
  rope_interleave = True
  rope_truncate = True
  rope_attention_scaling = False
  # Indexer
  use_sparse_indexer = True
  index_n_heads = 64
  index_head_dim = 128
  index_topk = 4


config = Config()

# 1. Setup PyTorch Config & Model
pt_args = {
    "max_batch_size": 8,
    "scale_fmt":  None,
    "max_seq_len": config.max_position_embeddings,  # Mapped from 163840
    "dtype": "bf16",
    "vocab_size": config.vocab_size,
    "dim": config.base_emb_dim,  # 7168
    "inter_dim": config.base_mlp_dim,  # 18432
    "moe_inter_dim": config.base_moe_mlp_dim,  # 2048
    "n_layers": config.base_num_decoder_layers,  # 61
    "n_dense_layers": config.first_num_dense_layers,  # 3
    "n_heads": config.base_num_query_heads,  # 128
    # moe
    "n_routed_experts": config.num_experts,  # 256
    "n_shared_experts": config.shared_experts,  # 1
    "n_activated_experts": config.num_experts_per_tok,  # 8
    "n_expert_groups": 1,  # Default (not in config)
    "n_limited_groups": 1,  # Default (not in config)
    "score_func": config.routed_score_func,  # "sigmoid"
    "route_scale": config.routed_scaling_factor,  # 2.5
    # mla
    "q_lora_rank": config.q_lora_rank,  # 1536
    "kv_lora_rank": config.kv_lora_rank,  # 512
    "qk_nope_head_dim": config.qk_nope_head_dim,  # 128
    "qk_rope_head_dim": config.qk_rope_head_dim,  # 64
    "v_head_dim": config.v_head_dim,  # 128
    # yarn
    "original_seq_len": config.original_max_position_embeddings,  # 4096
    "rope_theta": float(config.rope_max_timescale),  # 10000.0
    "rope_factor": float(config.rope_factor),  # 40.0
    "beta_fast": config.beta_fast,  # 32
    "beta_slow": 1,  # Default (not in config)
    "mscale": config.mscale,  # 1.0
    # index
    "index_n_heads": config.index_n_heads,  # 64
    "index_head_dim": config.index_head_dim,  # 128
    "index_topk": config.index_topk,  # 2048
}


class ParallelEmbedding(nn.Module):
  """
  Embedding layer with parallelism support across distributed processes.
  Args:
      vocab_size (int): Vocabulary size.
      dim (int): Embedding dimension.
  """

  def __init__(self, vocab_size: int, dim: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.dim = dim
    assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
    self.part_vocab_size = vocab_size // world_size
    self.vocab_start_idx = rank * self.part_vocab_size
    self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
    self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for parallel embedding layer.
    Args:
        x (torch.Tensor): Input tensor containing token indices.
    Returns:
        torch.Tensor: Embedded representations.
    Raises:
        ValueError: If `world_size` is not defined.
    """
    if world_size > 1:
      mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
      x = x - self.vocab_start_idx
      x[mask] = 0
    y = F.embedding(x, self.weight)
    if world_size > 1:
      y[mask] = 0
      dist.all_reduce(y)
    return y


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None
) -> torch.Tensor:
  """
  Applies a linear transformation to the incoming data: y = xA^T + b.
  This function supports specialized implementations based on quantization
  and tensor formats.
  Args:
      x (torch.Tensor): The input tensor.
      weight (torch.Tensor): The weight tensor. It may be quantized and
          requires dequantization for certain cases.
      bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
      scale_fmt (Optional[str]): The format of scaling factors.
  Returns:
      torch.Tensor: The result of the linear transformation, which may involve
      quantization-aware computations depending on the input parameters.
  Notes:
      - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
        is used for computation.
      - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
  """
  assert bias is None

  if weight.dtype != torch.float8_e4m3fn:
    return F.linear(x, weight)
  # CHANGE
  # else:
  #     x, scale = act_quant(x, block_size, scale_fmt)
  #     return fp8_gemm(x, scale, weight, weight.scale)


class Linear(nn.Module):
  """
  Custom linear layer with support for quantized weights and optional bias.
  Args:
      in_features (int): Number of input features.
      out_features (int): Number of output features.
      bias (bool): Whether to include a bias term. Defaults to False.
      dtype (optional): Data type for the layer. Defaults to `torch.float32`.
  """

  dtype = torch.float32
  scale_fmt: Optional[str] = None

  def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
    if self.weight.element_size() == 1:
      scale_out_features = (out_features + block_size - 1) // block_size
      scale_in_features = (in_features + block_size - 1) // block_size
      self.weight.scale = self.scale = nn.Parameter(
          torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
      )
    else:
      self.register_parameter("scale", None)
    if bias:
      self.bias = nn.Parameter(torch.empty(out_features))
    else:
      self.register_parameter("bias", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the custom linear layer.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Transformed tensor after linear computation.
    """
    return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
  """
  Linear layer with column parallelism, splitting output features across distributed processes.
  Args:
      in_features (int): Number of input features.
      out_features (int): Total number of output features.
      bias (bool): Whether to include a bias term. Defaults to False.
      dtype (optional): Data type for the layer. Defaults to `torch.float32`.
  """

  def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
    assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
    self.part_out_features = out_features // world_size
    super().__init__(in_features, self.part_out_features, bias, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for column parallel linear layer.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Transformed tensor with column-parallel computation.
    """
    y = linear(x, self.weight, self.bias, self.scale_fmt)
    return y


class RowParallelLinear(Linear):
  """
  Linear layer with row parallelism, splitting input features across distributed processes.
  Args:
      in_features (int): Total number of input features.
      out_features (int): Number of output features.
      bias (bool): Whether to include a bias term. Defaults to False.
      dtype (optional): Data type for the layer. Defaults to `torch.float32`.
  """

  def __init__(self, in_features: int, out_features: int, bias: bool = False, reduce_output=True, dtype=None):
    assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
    self.part_in_features = in_features // world_size
    self.reduce_output = reduce_output
    super().__init__(self.part_in_features, out_features, bias, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for row parallel linear layer.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Transformed tensor with row-parallel computation.
    """
    y = linear(x, self.weight, None, self.scale_fmt)
    if self.reduce_output and world_size > 1:
      y = y.float()
      dist.all_reduce(y)
    if self.bias is not None:
      y += self.bias
    return y.type_as(x)


class RMSNorm(nn.Module):
  """
  Root Mean Square Layer Normalization (RMSNorm).
  Args:
      dim (int): Dimension of the input tensor.
      eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
  """

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    """
    Forward pass for RMSNorm.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
    """
    dtype = x.dtype
    if residual is None:
      x = x.float()
      var = x.pow(2).mean(-1, keepdim=True)
      x = x * torch.rsqrt(var + self.eps)
      return (self.weight * x).to(dtype)
    else:
      x = residual = x.float() + residual.float()
      var = x.pow(2).mean(-1, keepdim=True)
      x = x * torch.rsqrt(var + self.eps)
      return (self.weight * x).to(dtype), residual.to(dtype)


class LayerNorm(nn.Module):
  """
  Layer Normalization.
  """

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
    self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor):
    return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
  """
  Precomputes frequency-based complex exponential values for rotary positional embeddings.
  Args:
      args (ModelArgs): Model arguments containing positional embedding parameters.
  Returns:
      torch.Tensor: Precomputed complex exponential values for positional embeddings.
  """
  dim = args.qk_rope_head_dim
  seqlen = args.max_seq_len
  beta_fast = args.beta_fast
  beta_slow = args.beta_slow
  base = args.rope_theta
  factor = args.rope_factor

  def find_correction_dim(num_rotations, dim, base, max_seq_len):
    """
    Computes the correction dimension for a given number of rotations in the rotary positional embedding.
    Args:
        num_rotations (float): Number of rotations to compute the correction for.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_seq_len (int): Maximum sequence length.
    Returns:
        float: The correction dimension based on the input parameters.
    """
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

  def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    """
    Computes the range of correction dimensions for rotary positional embeddings.
    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_seq_len (int): Maximum sequence length.
    Returns:
        Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
    """
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min, max, dim):
    """
    Computes a linear ramp function used to smooth values between a minimum and maximum range.
    Args:
        min (float): Minimum value for the ramp function.
        max (float): Maximum value for the ramp function.
        dim (int): Dimensionality of the ramp tensor.
    Returns:
        torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
            clamped to the range [0, 1].
    """
    if min == max:
      max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

  freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  if seqlen > args.original_seq_len:
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth

  t = torch.arange(seqlen)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
  """
  Applies rotary positional embeddings to the input tensor.
  Args:
      x (torch.Tensor): Input tensor with positional embeddings to be applied.
      freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.
  Returns:
      torch.Tensor: Tensor with rotary embeddings applied.
  """
  dtype = x.dtype
  shape = x.shape
  if not interleaved:
    x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
  x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
  freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
  y = torch.view_as_real(x * freqs_cis).flatten(3)
  if not interleaved:
    y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
  return y.to(dtype)


# def rotate_activation(x: torch.Tensor) -> torch.Tensor:
#   assert x.dtype == torch.float32
#   from fast_hadamard_transform import hadamard_transform

#   hidden_size = x.size(-1)
#   return hadamard_transform(x, scale=hidden_size**-0.5)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
  return x


class Indexer(torch.nn.Module):

  def __init__(self, args: ModelArgs):
    super().__init__()
    self.dim: int = args.dim
    self.n_heads: int = args.index_n_heads
    self.n_local_heads = args.index_n_heads // world_size
    self.head_dim: int = args.index_head_dim
    self.rope_head_dim: int = args.qk_rope_head_dim
    self.index_topk: int = args.index_topk
    self.q_lora_rank: int = args.q_lora_rank
    self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
    self.wk = Linear(self.dim, self.head_dim)
    self.k_norm = LayerNorm(self.head_dim)
    # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
    self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
    self.softmax_scale = self.head_dim**-0.5
    self.scale_fmt = args.scale_fmt

    # for name, param in self.weights_proj.named_parameters():
    #   print(name)
    #   print(param)

    # CHANGE
    # self.register_buffer(
    #     "k_cache",
    #     torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn),
    #     persistent=False,
    # )
    # self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)
    self.register_buffer(
        "k_cache",
        torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float32),
        persistent=False,
    )

  def forward(
      self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], debug: bool = False
  ):
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen

    q = self.wq_b(qr)
    q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
    # print("before rope")
    # return None, q
    q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    # rope in indexer is not interleaved
    q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
    q = torch.cat([q_pe, q_nope], dim=-1)
    # print("after rope")
    # return None, q
    q = rotate_activation(q)

    k = self.wk(x)
    k = self.k_norm(k)
    k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    # rope in indexer is not interleaved
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    k = rotate_activation(k)

    # return None, k

    # CHANGE
    # q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
    # k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
    # self.k_cache[:bsz, start_pos:end_pos] = k_fp8
    # self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
    self.k_cache[:bsz, start_pos:end_pos] = k

    weights = self.weights_proj(x.float()) * self.n_heads**-0.5
    # CHANGE
    # weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    weights = weights * self.softmax_scale
    # print(f"DEBUG: weights max={weights.max().item()}")

    # CHANGE
    # index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
    # fp8_index is defined by: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/kernel.py#L254
    # Replace fp_index with standard PyTorch: Sum_h( ReLU(Q @ K.T) * Weights
    # b: Batch size, h: number of attention heads, d: head dim
    # s (Sequence Length): The number of new tokens currently being processed (the queries). In inference, this is often 1.
    # t (Total Cache Length): The number of tokens currently stored in the memory (the keys). This includes all past tokens plus the current ones.
    k_active = self.k_cache[:bsz, :end_pos]
    # Compute Logits: Q (b,s,h,d) @ K (b,t,d) -> (b,s,t,h)
    # logits = torch.einsum("bshd, btd -> bsth", q, k_active)
    # logits = F.relu(logits)
    # # Weighted Sum: Logits (b,s,t,h) * Weights (b,s,h) -> (b,s,t)
    # index_score = torch.einsum("bsth, bsh -> bst", logits, weights)

    # return k_active, None

    # logits is bfloat16
    logits = torch.einsum("bshd, btd -> bsth", q.to(torch.float32), k_active.to(torch.float32))
    logits = F.relu(logits)
    # FIX: Cast logits to weights.dtype (float32) so einsum inputs match
    index_score = torch.einsum("bsth, bsh -> bst", logits.to(weights.dtype), weights)

    if mask is not None:
      index_score += mask
    topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]

    print("torch_index_score", index_score)

    if debug:
      return topk_indices, index_score
      # return None, {
      #     "k": k,
      #     "q": q,
      #     "weights": weights,
      #     "logits": logits,
      #     "index_score": index_score,
      # }
    return topk_indices

    # dist.broadcast(topk_indices_, src=0)
    # assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
    # return topk_indices


# def weight_dequant(weight, scale):
#     shape = weight.shape
#     assert weight.dim() == 2
#     weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size).transpose(1, 2).contiguous().view(-1, block_size * block_size)
#     weight = (weight.float() * scale.view(-1, 1).float()).to(torch.get_default_dtype()).view(shape[0] // block_size, shape[1] // block_size, block_size, block_size).transpose(1, 2).contiguous().view(shape)
#     return weight


class MLA(nn.Module):
  """
  Multi-Head Latent Attention (MLA) Layer.
  Attributes:
      dim (int): Dimensionality of the input features.
      n_heads (int): Number of attention heads.
      n_local_heads (int): Number of local attention heads for distributed systems.
      q_lora_rank (int): Rank for low-rank query projection.
      kv_lora_rank (int): Rank for low-rank key/value projection.
      qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
      qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
      qk_head_dim (int): Total dimensionality of query/key projections.
      v_head_dim (int): Dimensionality of value projections.
      softmax_scale (float): Scaling factor for softmax in attention computation.
  """

  def __init__(self, args: ModelArgs):
    super().__init__()
    self.dim = args.dim
    self.n_heads = args.n_heads
    self.n_local_heads = args.n_heads // world_size
    self.q_lora_rank = args.q_lora_rank
    self.kv_lora_rank = args.kv_lora_rank
    self.qk_nope_head_dim = args.qk_nope_head_dim
    self.qk_rope_head_dim = args.qk_rope_head_dim
    self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
    self.v_head_dim = args.v_head_dim

    self.wq_a = Linear(self.dim, self.q_lora_rank)
    self.q_norm = RMSNorm(self.q_lora_rank)
    self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
    self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
    self.kv_norm = RMSNorm(self.kv_lora_rank)
    self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
    self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
    self.softmax_scale = self.qk_head_dim**-0.5
    self.scale_fmt = args.scale_fmt
    if args.max_seq_len > args.original_seq_len:
      mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    self.indexer = Indexer(args)

    self.register_buffer(
        "kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False
    )
    self.register_buffer(
        "pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False
    )
    self.dequant_wkv_b = None

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
    """
    Forward pass for the Multi-Head Latent Attention (MLA) Layer.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        start_pos (int): Starting position in the sequence for caching.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
        mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.
    Returns:
        torch.Tensor: Output tensor with the same shape as the input.
    """
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    qr = self.q_norm(self.wq_a(x))
    q = self.wq_b(qr)
    q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    kv = self.wkv_a(x)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv = self.kv_norm(kv)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

    # we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to bf16.
    # CHANGE
    # kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
    # kv = (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)

    self.kv_cache[:bsz, start_pos:end_pos] = kv
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
    # train also uses MHA
    if mask is not None:  # MHA prefill
      q = torch.cat([q_nope, q_pe], dim=-1)
      kv = self.wkv_b(kv)
      kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
      k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
      k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
      scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

      # indexer
      topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
      index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
      index_mask += mask
      scores += index_mask.unsqueeze(2)

      scores = scores.softmax(dim=-1)
      x = torch.einsum("bsht,bthd->bshd", scores, v)
    else:  # MQA decode
      # CHANGE
      # if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
      #     self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
      wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
      wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
      q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim])
      scores = (
          torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
          + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
      ) * self.softmax_scale

      # indexer
      topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
      index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
      scores += index_mask.unsqueeze(2)

      scores = scores.softmax(dim=-1)
      x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
      x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
    x = self.wo(x.flatten(2))
    return x


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


# def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
#   # return jnp.asarray(pt_tensor.detach().cpu().numpy())
#   # return jax.dlpack.from_dlpack(pt_tensor.detach())
#   # 1. Handle BFloat16 (NumPy doesn't support it natively)
#   if pt_tensor.dtype == torch.float32:
#     return jnp.asarray(
#         pt_tensor.detach().cpu().to(torch.float32).numpy()
#     ).astype(jnp.bfloat16)  # Cast back to BF16 on the JAX device

#   # 2. Handle standard types (Float32, Int64, etc.)
#   return jnp.asarray(pt_tensor.detach().cpu().numpy())


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  # return jnp.asarray(pt_tensor.detach().cpu().numpy())
  # return jax.dlpack.from_dlpack(pt_tensor.detach())
  # 1. Handle BFloat16 (NumPy doesn't support it natively)
  if pt_tensor.dtype != torch.float32:
    return jnp.asarray(
        pt_tensor.detach().cpu().to(torch.float32).numpy()
    )  # .astype(jnp.bfloat16)  # Cast back to BF16 on the JAX device

  # 2. Handle standard types (Float32, Int64, etc.)
  return jnp.asarray(pt_tensor.detach().cpu().numpy())


def to_torch(jax_array: jax.Array) -> torch.Tensor:
  return torch.from_numpy(np.array(jax_array))


def init_torch_weights(module, std=1):
  """
  Initialize all Linear layers in the module.
  Weights -> Normal distribution (mean=0.0, std=std)
  Biases  -> Zero
  """
  with torch.no_grad():
    for name, param in module.named_parameters():
      # Check if it's a weight or a bias
      if "weight" in name:
        if param.dim() >= 2:  # Linear weights are usually 2D
          torch.nn.init.normal_(param, mean=0.0, std=std)
      elif "bias" in name:
        # torch.nn.init.zeros_(param)
        torch.nn.init.normal_(param, mean=0.0, std=std)


class DeepseekV32IndexerTest(unittest.TestCase):
  """Tests for the Sparse Indexer."""

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)  # precision
    torch.manual_seed(42)

    # jax config
    self.config = Config()
    # data, test long context
    # self.dtype = "bfloat16"
    self.dtype = "float32"  # precision
    self.batch_size = 2
    self.seq_len = 8

    # pt args
    self.pt_args = SimpleNamespace(**pt_args)
    self.pt_args.max_batch_size = self.batch_size

    # Indexer
    self.pt_indexer = Indexer(self.pt_args)
    self.pt_indexer = self.pt_indexer.float()
    self.pt_indexer.eval()

    # === FIX: Initialize the zeroed weights ===
    # Using normal distribution (standard for linear layers)
    init_torch_weights(self.pt_indexer)

    # 4. Input Data
    self.x = torch.randn(self.batch_size, self.seq_len, self.pt_args.dim)
    self.qr = torch.randn(self.batch_size, self.seq_len, self.pt_args.q_lora_rank)

    # RoPE freqs for PyTorch
    self.freqs_cis = precompute_freqs_cis(self.pt_args).to(self.x.device)

  def test_indexer_forward_pass(self):
    """Verifies Indexer output mask matches PyTorch Top-K selection."""

    causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).expand(self.batch_size, -1, -1)
    causal_mask_bias = torch.where(causal_mask == 1, 0.0, float("-inf"))
    mask = causal_mask_bias
    # mask = None

    # C. Run PyTorch Reference
    self.start_pos = 0
    self.freqs_cis_slice = self.freqs_cis[self.start_pos : self.start_pos + self.seq_len]
    with torch.no_grad():
      # Returns indices [B, K]
      pt_indices, pt_index_score = self.pt_indexer(
          self.x, self.qr, self.start_pos, self.freqs_cis_slice, mask=mask, debug=True
      )
      pt_mask = torch.full((self.batch_size, self.seq_len, self.seq_len), float("-inf"), device=self.x.device).scatter_(
          -1, pt_indices, 0
      )
      if mask is not None:
        pt_mask += mask

    # jax position, Shape: [B, S]
    start_pos = self.start_pos
    end_pos = self.start_pos + self.seq_len
    positions = jnp.arange(start_pos, end_pos, dtype=jnp.int32)[None, :]
    positions = jnp.broadcast_to(positions, (self.batch_size, self.seq_len))

    # 3. Setup Mesh
    cfg = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        run_name="gpt_oss_attention_test_flash",
        enable_checkpointing=False,
        model_name="default",
        dtype=self.dtype,
        weight_dtype="float32",  # precision
        matmul_precision="highest",  # precision
        per_device_batch_size=self.batch_size,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        attention="dot_product",
        # attention
        base_emb_dim=self.config.base_emb_dim,
        base_num_query_heads=self.config.base_num_query_heads,
        base_num_kv_heads=self.config.base_num_kv_heads,
        # mla
        attention_type=self.config.attention_type,
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        # RoPE
        rope_type=self.config.rope_type,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        rope_max_timescale=self.config.rope_max_timescale,
        max_position_embeddings=self.config.max_position_embeddings,
        rope_factor=self.config.rope_factor,
        beta_fast=self.config.beta_fast,
        mscale=self.config.mscale,
        rope_interleave=self.config.rope_interleave,
        rope_truncate=self.config.rope_truncate,
        rope_attention_scaling=self.config.rope_attention_scaling,
        # Indexer
        use_sparse_indexer=self.config.use_sparse_indexer,
        index_n_heads=self.config.index_n_heads,
        index_head_dim=self.config.index_head_dim,
        index_topk=self.config.index_topk,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    self.mesh = Mesh(devices_array, cfg.mesh_axes)

    yarn_rope = embeddings.YarnRotaryEmbedding(
        max_position_embeddings=self.config.max_position_embeddings,
        mesh=self.mesh,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        beta_fast=self.config.beta_fast,
        beta_slow=self.config.beta_slow,
        rope_theta=self.config.rope_max_timescale,
        rope_factor=self.config.rope_factor,
        embedding_dims=self.config.qk_rope_head_dim,
        fprop_dtype=self.dtype,
        interleave=False, #self.config.rope_interleave,
        truncate=self.config.rope_truncate,
        attention_scaling=self.config.rope_attention_scaling,
        # shard_mode=self.config.shard_mode,
        # rngs=self.rngs,
    )

    jax_indexer = attention_mla.Indexer(
        config=cfg,
        rngs=nnx.Rngs(0),
        rotary_embedding=yarn_rope,
    )

    # B. Copy Weights (PyTorch -> JAX)
    indexer_state = {
        "wq_b": {"kernel": to_jax(self.pt_indexer.wq_b.weight.T)},
        "wk": {"kernel": to_jax(self.pt_indexer.wk.weight.T)},
        "weights_proj": {"kernel": to_jax(self.pt_indexer.weights_proj.weight.T)},
        "k_norm": {
            "scale": to_jax(self.pt_indexer.k_norm.weight),
            "bias": to_jax(self.pt_indexer.k_norm.bias),
        },
    }
    nnx.update(jax_indexer, indexer_state)

    # D. Run JAX Forward
    # Returns bias mask [B, 1, S, T]
    jax_bias, jax_indices, jax_index_score = jax_indexer(
        to_jax(self.x),
        to_jax(self.qr),
        inputs_positions=positions,
        mask=to_jax(mask.unsqueeze(1)) if mask is not None else None,
    )

    np.testing.assert_allclose(jax_index_score, to_jax(pt_index_score), rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(jax_bias.squeeze(1) == 0, to_jax(pt_mask == 0))
    # np.testing.assert_array_equal(jax_indices, to_jax(pt_indices))
    # chex.assert_trees_all_close(jax_index_score, jax.tree.map(to_jax, pt_index_score), rtol=1e-3, atol=1e-3)


class DeepseekV32MLATest(unittest.TestCase):
  """Tests for MLA Attention with Sparse Indexing."""

  def setUp(self):
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)

    # jax config
    self.config = Config()
    # data, test long context
    # self.dtype = "bfloat16"
    self.dtype = "float32"
    self.batch_size = 1
    self.seq_len = 5
    self.start_pos = 0

    # pt args
    self.pt_args = SimpleNamespace(**pt_args)
    self.pt_args.max_batch_size = self.batch_size

    # MLA
    self.pt_mla = MLA(self.pt_args)
    self.pt_mla.eval()
    init_torch_weights(self.pt_mla)

    # input
    self.x = torch.randn(self.batch_size, self.seq_len, self.pt_args.dim)

  def test_mla_prefill_match(self):
    """Verifies MLA prefill output matches PyTorch (including sparse bias)."""

    # PyTorch: mask is needed for MHA mode in reference code
    # Shape [B, S, S], causal mask
    start_pos = 0
    causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).expand(self.batch_size, -1, -1)
    causal_mask_bias = torch.where(causal_mask == 1, 0.0, float("-inf"))
    # RoPE
    self.freqs_cis = precompute_freqs_cis(self.pt_args).to(self.x.device)
    self.freqs_cis_slice = self.freqs_cis[start_pos : start_pos + self.seq_len]
    with torch.no_grad():
      pt_out = self.pt_mla(self.x, start_pos=start_pos, freqs_cis=self.freqs_cis_slice, mask=causal_mask_bias)

    cfg = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        run_name="gpt_oss_attention_test_flash",
        enable_checkpointing=False,
        model_name="default",
        dtype=self.dtype,
        weight_dtype="float32",  # precision
        matmul_precision="highest",  # precision
        per_device_batch_size=self.batch_size,
        max_target_length=self.seq_len,
        max_prefill_predict_length=self.seq_len,
        attention="dot_product",
        # attention
        base_emb_dim=self.config.base_emb_dim,
        base_num_query_heads=self.config.base_num_query_heads,
        base_num_kv_heads=self.config.base_num_kv_heads,
        # mla
        attention_type=self.config.attention_type,
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        # RoPE
        rope_type=self.config.rope_type,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        rope_max_timescale=self.config.rope_max_timescale,
        max_position_embeddings=self.config.max_position_embeddings,
        rope_factor=self.config.rope_factor,
        beta_fast=self.config.beta_fast,
        mscale=self.config.mscale,
        rope_interleave=self.config.rope_interleave,
        rope_truncate=self.config.rope_truncate,
        rope_attention_scaling=self.config.rope_attention_scaling,
        # Indexer
        use_sparse_indexer=self.config.use_sparse_indexer,
        index_n_heads=self.config.index_n_heads,
        index_head_dim=self.config.index_head_dim,
        index_topk=self.config.index_topk,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    self.mesh = Mesh(devices_array, cfg.mesh_axes)

    # A. JAX Init
    jax_mla = attention_mla.MLA(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        # mla
        attention_type="mla",
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        max_target_length=self.seq_len,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(self.batch_size, self.seq_len, cfg.emb_dim),
        inputs_kv_shape=(self.batch_size, self.seq_len, cfg.emb_dim),
        rngs=nnx.Rngs(0),
    )

    # B. Copy Weights
    # reshape to match maxtext
    base_emb_dim = self.config.base_emb_dim
    base_num_query_heads = self.config.base_num_query_heads
    q_lora_rank = self.config.q_lora_rank
    kv_lora_rank = self.config.kv_lora_rank
    qk_nope_head_dim = self.config.qk_nope_head_dim
    qk_rope_head_dim = self.config.qk_rope_head_dim
    v_head_dim = self.config.v_head_dim

    # wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
    # out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
    # wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

    mla_state = {
        # 1. Main MLA Weights
        "wq_a": {"kernel": to_jax(self.pt_mla.wq_a.weight.T)},
        "q_norm": {"scale": to_jax(self.pt_mla.q_norm.weight)},
        "wq_b": {
            "kernel": to_jax(self.pt_mla.wq_b.weight.T).reshape(
                [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)]
            )
        },
        "wkv_a": {"kernel": to_jax(self.pt_mla.wkv_a.weight.T)},
        "kv_norm": {"scale": to_jax(self.pt_mla.kv_norm.weight)},
        "wkv_b": {
            "kernel": to_jax(self.pt_mla.wkv_b.weight.T).reshape(
                [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)]
            )
        },
        "out": {"kernel": to_jax(self.pt_mla.wo.weight.T).reshape([base_num_query_heads, v_head_dim, base_emb_dim])},
        # 2. Indexer Weights
        "indexer": {
            "wq_b": {"kernel": to_jax(self.pt_mla.indexer.wq_b.weight.T)},
            "wk": {"kernel": to_jax(self.pt_mla.indexer.wk.weight.T)},
            "weights_proj": {"kernel": to_jax(self.pt_mla.indexer.weights_proj.weight.T)},
            "k_norm": {
                "scale": to_jax(self.pt_mla.indexer.k_norm.weight),
                "bias": to_jax(self.pt_mla.indexer.k_norm.bias),
            },
        },
    }
    # Apply the update
    nnx.update(jax_mla, mla_state)

    # C. Run Forward, JAX
    # self.rng = jax.random.PRNGKey(0)
    decoder_segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    decoder_positions = jnp.broadcast_to(
        jnp.arange(start_pos, start_pos + self.seq_len, dtype=jnp.int32), (self.batch_size, self.seq_len)
    )

    jax_out, _ = jax_mla(
        inputs_q=to_jax(self.x),
        inputs_kv=to_jax(self.x),
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=MODEL_MODE_TRAIN,
    )

    print("torch out", to_jax(pt_out))
    print("jax out", jax_out)

    # D. Compare
    # Tolerances slightly loose due to potential BF16/FP32 differences inside modules
    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  unittest.main()
