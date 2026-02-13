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
Tests for DeepSeek V3.2: Indexer, MLA

DeepSeek 3.2 PyTorch implementation at: 
https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py

We adapt the reference implementation to run on CPU:
- Original code is GPU-specific, due to quantization and fp8 kernel
- Remove quantization logic. Use float32 for dtype and weight_dytpe
- Replace fp8 kernel with naive dot product
- Replace fast_hadamard_transform.hadamard_transform with F.linear
- Changes other than dtype are marked with `# [CHANGE]`, primarily in Indexer and MLA

To run the test
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python3 -m pytest -v --pyargs tests.unit.deepseek32_vs_reference_test -rP -s
"""


import math
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import scipy
import unittest
from absl.testing import parameterized

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import jax
from jax.sharding import Mesh
import jax.numpy as jnp
from flax import nnx

from MaxText import pyconfig
from MaxText.layers import embeddings, attention_mla
from MaxText.common_types import MODEL_MODE_TRAIN
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


world_size = 1
rank = 0
block_size = 128


@dataclass
class Config:
  """MaxText config"""

  # attention
  base_emb_dim: int = 71
  base_num_query_heads: int = 128
  base_num_kv_heads: int = 128
  # mla
  attention_type: str = "mla"
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  use_tokamax_splash: bool = True
  # yarn
  rope_type: str = "yarn"
  original_max_position_embeddings: int = 4096
  rope_max_timescale: int = 10_000
  max_position_embeddings: int = 163840
  rope_factor: int = 40
  beta_fast: int = 32
  beta_slow: int = 1
  mscale: float = 1.0
  rope_interleave: bool = True
  rope_truncate: bool = True
  rope_attention_scaling: bool = False
  # indexer
  use_sparse_indexer: bool = True
  index_n_heads: int = 64
  index_head_dim: int = 128  # > qk_rope_head_dim


class ModelArgs:
  """
  Arguments for the PyTorch Reference Model.
  Maps MaxText Config keys to the specific variable names expected by the reference implementation.
  """

  def __init__(self, config: Config, max_batch_size: int = 8, index_topk: int = 4):
    self.max_batch_size = max_batch_size
    self.scale_fmt = None
    self.max_seq_len = config.max_position_embeddings
    self.dim = config.base_emb_dim
    # mla
    self.n_heads = config.base_num_query_heads
    self.q_lora_rank = config.q_lora_rank
    self.kv_lora_rank = config.kv_lora_rank
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.v_head_dim = config.v_head_dim
    self.use_tokamax_splash = config.use_tokamax_splash
    # yarn
    self.original_seq_len = config.original_max_position_embeddings
    self.rope_theta = float(config.rope_max_timescale)
    self.rope_factor = float(config.rope_factor)
    self.beta_fast = config.beta_fast
    self.beta_slow = config.beta_slow
    self.mscale = config.mscale
    # indexer
    self.index_n_heads = config.index_n_heads
    self.index_head_dim = config.index_head_dim
    self.index_topk = index_topk


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def linear(  # pylint: disable=inconsistent-return-statements
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
  # [CHANGE]: remove
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

  def linear_ramp_factor(min, max, dim):  # pylint: disable=redefined-builtin
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


# [CHANGE]
# fast_hadamard_transform is gpu specific: https://github.com/Dao-AILab/fast-hadamard-transform
# `hadamard_transform(x, scale)` is equivalent to `F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale`
# OLD
# def rotate_activation(x: torch.Tensor) -> torch.Tensor:
#     assert x.dtype == torch.bfloat16
#     from fast_hadamard_transform import hadamard_transform
#     hidden_size = x.size(-1)
#     return hadamard_transform(x, scale=hidden_size ** -0.5)
# NEW
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
  hidden_size = x.size(-1)
  return F.linear(x, torch.tensor(scipy.linalg.hadamard(hidden_size), dtype=x.dtype, device=x.device)) * hidden_size**-0.5


class Indexer(torch.nn.Module):  # pylint: disable=missing-class-docstring

  def __init__(self, args: ModelArgs, index_topk: int = 4):
    super().__init__()
    self.dim: int = args.dim
    self.n_heads: int = args.index_n_heads
    self.n_local_heads = args.index_n_heads // world_size
    self.head_dim: int = args.index_head_dim
    self.rope_head_dim: int = args.qk_rope_head_dim
    self.index_topk: int = index_topk
    self.q_lora_rank: int = args.q_lora_rank
    self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
    self.wk = Linear(self.dim, self.head_dim)
    self.k_norm = LayerNorm(self.head_dim)
    # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
    self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
    self.softmax_scale = self.head_dim**-0.5
    self.scale_fmt = args.scale_fmt

    # [CHANGE]
    # OLD
    # self.register_buffer(
    #     "k_cache",
    #     torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn),
    #     persistent=False,
    # )
    # self.register_buffer(
    #     "k_scale_cache",
    #     torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32),
    #     persistent=False,
    # )
    # NEW
    self.register_buffer(
        "k_cache",
        torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float32),
        persistent=False,
    )

  def forward(  # pylint: disable=missing-function-docstring
      self,
      x: torch.Tensor,
      qr: torch.Tensor,
      start_pos: int,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      debug: bool = False,
  ):
    bsz, seqlen, _ = x.size()
    end_pos = start_pos + seqlen
    q = self.wq_b(qr)
    q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
    q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    # rope in indexer is not interleaved
    q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
    q = torch.cat([q_pe, q_nope], dim=-1)
    k = self.wk(x)
    k = self.k_norm(k)
    k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    # rope in indexer is not interleaved
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)
    q = rotate_activation(q)
    k = rotate_activation(k)

    # [CHANGE]
    # OLD
    # q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
    # k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
    # self.k_cache[:bsz, start_pos:end_pos] = k_fp8
    # self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
    # NEW
    self.k_cache[:bsz, start_pos:end_pos] = k

    weights = self.weights_proj(x.float()) * self.n_heads**-0.5

    # [CHANGE]
    # OLD
    # weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    # NEW
    weights = weights * self.softmax_scale

    # [CHANGE]
    # fp8_index is defined by: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/kernel.py#L254 # pylint: disable=line-too-long
    # Replace fp8_index with standard PyTorch: Sum_h( ReLU(Q @ K.T) * Weights
    # OLD
    # index_score = fp8_index(
    #     q_fp8.contiguous(),
    #     weights,
    #     self.k_cache[:bsz, :end_pos].contiguous(),
    #     self.k_scale_cache[:bsz, :end_pos].contiguous(),
    # )
    # NEW
    logits = torch.einsum("bthd, bsd -> btsh", q, self.k_cache[:bsz, :end_pos])
    logits = F.relu(logits)
    index_score = torch.einsum("btsh, bth -> bts", logits, weights)

    if mask is not None:
      index_score += mask
    topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]

    # [CHANGE]: add
    # additionally return index_score for indexer test
    if debug:
      return topk_indices, index_score

    return topk_indices


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

  def __init__(self, args: ModelArgs, index_topk: int):
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

    self.indexer = Indexer(args, index_topk)

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
    # [CHANGE]: remove
    # kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
    # kv = (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)

    self.kv_cache[:bsz, start_pos:end_pos] = kv
    self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
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
      # [CHANGE]: remove
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
# Test JAX Module
# -----------------------------------------------------------------------------


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  """Converts a PyTorch tensor to a JAX array.

  Args:
    pt_tensor: The PyTorch tensor to convert.

  Returns:
    The equivalent JAX array.
  """
  return jnp.asarray(pt_tensor.detach().numpy())


def init_torch_weights(module, std=1):
  """
  Initialize all parameters in the module with N(0,std).
  This simple strategy is intended only for unit test.
  """
  with torch.no_grad():
    for _, param in module.named_parameters():
      torch.nn.init.normal_(param, mean=0.0, std=std)


def get_jax_indexer_weights(pt_indexer):
  """Extracts weights for the Indexer module."""
  return {
      "wq_b": {"kernel": to_jax(pt_indexer.wq_b.weight.T)},
      "wk": {"kernel": to_jax(pt_indexer.wk.weight.T)},
      "weights_proj": {"kernel": to_jax(pt_indexer.weights_proj.weight.T)},
      "k_norm": {
          "scale": to_jax(pt_indexer.k_norm.weight),
          "bias": to_jax(pt_indexer.k_norm.bias),
      },
  }


def get_jax_mla_weights(pt_mla, cfg):
  """Extracts weights for the MLA module based on jax config (cfg)."""
  return {
      "wq_a": {"kernel": to_jax(pt_mla.wq_a.weight.T)},
      "q_norm": {"scale": to_jax(pt_mla.q_norm.weight)},
      "wq_b": {
          "kernel": to_jax(pt_mla.wq_b.weight.T).reshape(
              [cfg.q_lora_rank, cfg.base_num_query_heads, (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)]
          )
      },
      "wkv_a": {"kernel": to_jax(pt_mla.wkv_a.weight.T)},
      "kv_norm": {"scale": to_jax(pt_mla.kv_norm.weight)},
      "wkv_b": {
          "kernel": to_jax(pt_mla.wkv_b.weight.T).reshape(
              [cfg.kv_lora_rank, cfg.base_num_query_heads, (cfg.qk_nope_head_dim + cfg.v_head_dim)]
          )
      },
      "out": {"kernel": to_jax(pt_mla.wo.weight.T).reshape([cfg.base_num_query_heads, cfg.v_head_dim, cfg.base_emb_dim])},
      # Reuse the helper function
      "indexer": get_jax_indexer_weights(pt_mla.indexer),
  }


def get_cfg_and_mesh(config, run_name, dtype, batch_size, seq_len, attention, index_topk):
  """Returns MaxText configuration and mesh."""
  cfg = pyconfig.initialize(
      [None, get_test_config_path()],
      run_name=run_name,
      enable_checkpointing=False,
      model_name="default",
      dtype=dtype,
      # high precision
      weight_dtype="float32",
      matmul_precision="highest",
      float32_qk_product=True,
      float32_logits=True,
      per_device_batch_size=batch_size,
      max_target_length=seq_len,
      max_prefill_predict_length=seq_len,
      attention=attention,
      index_topk=index_topk,
      **asdict(config),
  )
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  return cfg, mesh


class DeepseekTestBase(parameterized.TestCase):
  """Base class handling common setup for DeepSeek V3.2"""

  def setUp(self):
    """Initializes the configuration for each test"""
    super().setUp()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)

    self.dtype = "float32"
    self.batch_size = 4
    self.start_pos = 0
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))
    # jax config
    self.config = Config()
    # torch config
    self.pt_args = ModelArgs(self.config, self.batch_size)

  def get_data(self, seq_len):
    """Initializes and returns synchronized data/masks for Torch and JAX."""
    self.seq_len = seq_len

    # --- PyTorch Inputs ---
    x = torch.randn(self.batch_size, seq_len, self.pt_args.dim)
    qr = torch.randn(self.batch_size, seq_len, self.pt_args.q_lora_rank)
    # RoPE
    freqs_cis = precompute_freqs_cis(self.pt_args).to(x.device)
    freqs_cis_slice = freqs_cis[self.start_pos : self.start_pos + seq_len]
    # Mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(self.batch_size, -1, -1)
    pt_mask = torch.where(causal_mask == 1, 0.0, float("-inf"))

    torch_inputs = {"x": x, "qr": qr, "freqs_cis_slice": freqs_cis_slice, "mask": pt_mask}

    # --- JAX Inputs ---
    decoder_positions = jnp.broadcast_to(
        jnp.arange(self.start_pos, self.start_pos + seq_len, dtype=jnp.int32), (self.batch_size, seq_len)
    )
    decoder_segment_ids = jnp.ones((self.batch_size, seq_len), dtype=jnp.int32)

    jax_inputs = {
        "x": to_jax(x),
        "qr": to_jax(qr),
        "positions": decoder_positions,
        "segment_ids": decoder_segment_ids,
        "mask": to_jax(pt_mask),
    }

    return torch_inputs, jax_inputs


class DeepseekV32IndexerTest(DeepseekTestBase):
  """Tests for the Sparse Indexer (Top-K Selection)."""

  # index_topk=4
  def test_indexer_match(self, seq_len=8):
    """Verifies Indexer output matches PyTorch output."""
    torch_inputs, jax_inputs = self.get_data(seq_len)
    pt_mask = torch_inputs["mask"]

    # 1. PyTorch Run
    pt_indexer = Indexer(self.pt_args)
    init_torch_weights(pt_indexer)
    pt_indexer.eval()

    with torch.no_grad():
      pt_indices, pt_index_score = pt_indexer(
          torch_inputs["x"],
          torch_inputs["qr"],
          self.start_pos,
          torch_inputs["freqs_cis_slice"],
          mask=pt_mask,
          debug=True,
      )
      # Reconstruct Mask
      pt_index_mask = torch.full((self.batch_size, self.seq_len, self.seq_len), float("-inf")).scatter_(-1, pt_indices, 0)
      if pt_mask is not None:
        pt_index_mask += pt_mask

    # 2. JAX Run
    cfg, mesh = get_cfg_and_mesh(
        config=self.config,
        run_name="deepseek_indexer_test",
        dtype=self.dtype,
        batch_size=self.batch_size,
        seq_len=self.seq_len,
        attention="dot_product",
        index_topk=4,
    )

    # Indexer specific RoPE (interleave=False)
    yarn_rope = embeddings.YarnRotaryEmbedding(
        max_position_embeddings=cfg.max_position_embeddings,
        mesh=mesh,
        original_max_position_embeddings=cfg.original_max_position_embeddings,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
        rope_theta=cfg.rope_max_timescale,
        rope_factor=cfg.rope_factor,
        embedding_dims=cfg.qk_rope_head_dim,
        fprop_dtype=self.dtype,
        interleave=False,
        truncate=cfg.rope_truncate,
        attention_scaling=cfg.rope_attention_scaling,
        rngs=self.nnx_rng,
    )

    jax_indexer = attention_mla.Indexer(config=cfg, rngs=self.nnx_rng, rotary_embedding=yarn_rope)

    # Copy Weights
    nnx.update(jax_indexer, get_jax_indexer_weights(pt_indexer))

    jax_index_mask, _, jax_index_score = jax_indexer(
        inputs_q=jax_inputs["x"],
        low_rank_q=jax_inputs["qr"],
        inputs_kv=jax_inputs["x"],
        inputs_positions=jax_inputs["positions"],
        attention_mask=jax_inputs["mask"],
    )

    # 3 Compare
    print("torch index score", pt_index_score)
    print("jax index score", jax_index_score)
    # check index score is close
    np.testing.assert_allclose(jax_index_score, to_jax(pt_index_score), rtol=1e-3, atol=1e-3)
    # check index mask is equal
    np.testing.assert_array_equal(jax_index_mask == 0, to_jax(pt_index_mask == 0))


class DeepseekV32MLATest(DeepseekTestBase):
  """Tests for MLA Attention with Sparse Indexing."""

  @parameterized.named_parameters(
      {
          "testcase_name": "dot_product_s2_k4",
          "attention": "dot_product",
          "seq_len": 2,
          "index_topk": 4,
      },
      {
          "testcase_name": "dot_product_s8_k4",
          "attention": "dot_product",
          "seq_len": 8,
          "index_topk": 4,
      },
      {
          "testcase_name": "dot_product_s128_k4",
          "attention": "dot_product",
          "seq_len": 128,
          "index_topk": 4,
          "check_norm": True,
      },
      {
          "testcase_name": "dot_product_s128_k128",
          "attention": "dot_product",
          "seq_len": 128,
          "index_topk": 128,
          "check_norm": True,
      },
      {
          "testcase_name": "flash_s128_k4",
          "attention": "flash",
          "seq_len": 128,
          "index_topk": 4,
          "check_norm": True,
      },
      {
          "testcase_name": "flash_s128_k128",
          "attention": "flash",
          "seq_len": 128,
          "index_topk": 128,
          "check_norm": True,
      },
  )
  def test_mla_parity(self, attention, seq_len, index_topk, check_norm=False):
    """Verifies JAX MLA output against the PyTorch reference implementation."""
    torch_inputs, jax_inputs = self.get_data(seq_len)

    # 1. PyTorch Run
    pt_mla = MLA(self.pt_args, index_topk)
    init_torch_weights(pt_mla)
    pt_mla.eval()

    with torch.no_grad():
      # MHA mode is activated by mask
      pt_out = pt_mla(
          torch_inputs["x"],
          start_pos=self.start_pos,
          freqs_cis=torch_inputs["freqs_cis_slice"],
          mask=torch_inputs["mask"],
      )

    # 2. JAX Run
    cfg, mesh = get_cfg_and_mesh(
        config=self.config,
        run_name="deepseek_mla_test",
        dtype=self.dtype,
        batch_size=self.batch_size,
        seq_len=self.seq_len,
        attention=attention,
        index_topk=index_topk,
    )

    jax_mla = attention_mla.MLA(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        attention_type="mla",
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        original_max_position_embeddings=cfg.original_max_position_embeddings,
        mscale=cfg.mscale,
        rope_factor=cfg.rope_factor,
        max_target_length=self.seq_len,
        mesh=mesh,
        attention_kernel=attention,
        inputs_q_shape=(self.batch_size, self.seq_len, cfg.emb_dim),
        inputs_kv_shape=(self.batch_size, self.seq_len, cfg.emb_dim),
        rngs=self.nnx_rng,
    )

    # Copy Weights
    nnx.update(jax_mla, get_jax_mla_weights(pt_mla, self.config))

    jax_out, _ = jax_mla(
        inputs_q=jax_inputs["x"],
        inputs_kv=jax_inputs["x"],
        inputs_positions=jax_inputs["positions"],
        decoder_segment_ids=jax_inputs["segment_ids"],
        model_mode=MODEL_MODE_TRAIN,
    )

    # 3. Compare
    if check_norm:
      expected = to_jax(pt_out) / jnp.linalg.norm(to_jax(pt_out))
      actual = jax_out / jnp.linalg.norm(jax_out)
    else:
      expected = to_jax(pt_out)
      actual = jax_out

    print("torch out", expected)
    print("jax out", actual)
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  unittest.main()
