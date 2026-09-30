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
#
# Reimplements the semantics of DeepSeek-V4-Flash inference/kernel.py
# (Copyright (c) 2023 DeepSeek, MIT License; see LICENSE in this directory).

"""Pure-torch CPU replacement for DeepSeek-V4 inference/kernel.py (tilelang).

Line ranges refer to upstream inference/kernel.py:

  function            | reproduces upstream kernel.py lines
  --------------------+----------------------------------------------------
  _fast_round_scale   | 22-37   fast_log2_ceil / fast_pow2 / fast_round_scale
  act_quant           | 40-125  act_quant_kernel + act_quant wrapper
  fp4_act_quant       | 128-200 fp4_quant_kernel + fp4_act_quant wrapper
  fp8_gemm            | 203-273 (NotImplementedError; bf16/fp32/fp64 weights only)
  sparse_attn         | 276-368 sparse_attn_kernel + sparse_attn wrapper
  hc_split_sinkhorn   | 371-438 hc_split_sinkhorn_kernel + wrapper
  fp4_gemm            | 441-536 (NotImplementedError; bf16/fp32/fp64 weights only)

MODE (see set_mode):
  'train': act_quant / fp4_act_quant are identity (inplace: x untouched and
           returned; non-inplace: (x, None)).
  'qat'  : forward emulation of the quant-dequant kernels with a
           straight-through (identity) gradient.
"""

import torch

MODE = "train"

_FP8_MAX = 448.0
_FP4_MAX = 6.0
# e2m1 magnitude grid and midpoints between consecutive grid values.
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Midpoints whose round-to-nearest-even result is the upper grid value.
_FP4_TIE_UP = (0.75, 1.75, 3.5)
_ATTN_BLOCK = 64


def set_mode(mode: str):
  """Selects 'train' (identity quant) or 'qat' (quant-dequant + STE)."""
  global MODE
  assert mode in ("train", "qat"), mode
  MODE = mode


def _fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
  """2**ceil(log2(amax * max_inv)) via IEEE-754 fp32 bit ops (lines 22-37)."""
  # .to() rather than .float(): fp64_promotion() makes .float() a no-op on fp64.
  x = amax.to(torch.float32) * torch.tensor(max_inv, dtype=torch.float32)
  bits = x.view(torch.int32)
  exp = (bits >> 23) & 0xFF
  man = bits & ((1 << 23) - 1)
  e = exp - 127 + (man != 0).to(torch.int32)
  return ((e + 127) << 23).view(torch.float32)


def _block_amax(x: torch.Tensor, block_size: int):
  n = x.size(-1)
  xb = x.float().unflatten(-1, (n // block_size, block_size))
  return xb, xb.abs().amax(-1, keepdim=True)


def _cast_fp4_e2m1(y: torch.Tensor) -> torch.Tensor:
  """Rounds values in [-6, 6] to the e2m1 grid, ties-to-even."""
  a = y.abs()
  mids = torch.tensor(_FP4_MIDS, dtype=a.dtype, device=a.device)
  grid = torch.tensor(_FP4_GRID, dtype=a.dtype, device=a.device)
  idx = torch.bucketize(a, mids, right=False)
  tie_up = torch.zeros_like(a, dtype=torch.bool)
  for t in _FP4_TIE_UP:
    tie_up |= a == t
  idx = idx + tie_up.to(idx.dtype)
  return torch.sign(y) * grid[idx]


def _ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
  """Forward value q, identity backward to x."""
  return x + (q - x).detach()


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
  """Block-wise FP8 (e4m3) quant; inplace=True writes quant-dequant into x."""
  if MODE == "train":
    return x if inplace else (x, None)
  n = x.size(-1)
  assert n % block_size == 0
  xb, amax = _block_amax(x, block_size)
  amax = amax.clamp(min=1e-4)
  if scale_fmt is not None:
    s = _fast_round_scale(amax, 1.0 / _FP8_MAX)
  else:
    s = amax * torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32)
  y = torch.clamp(xb / s, -_FP8_MAX, _FP8_MAX)
  if inplace:
    deq = (y.to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)
    x.copy_(_ste(x, deq))
    return x
  return y.flatten(-2).to(torch.float8_e4m3fn), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
  """Block-wise FP4 (e2m1) quant with ue8m0 scale; inplace=True dequantizes."""
  if MODE == "train":
    return x if inplace else (x, None)
  n = x.size(-1)
  assert n % block_size == 0
  xb, amax = _block_amax(x, block_size)
  amax = amax.clamp(min=6 * (2**-126))
  s = _fast_round_scale(amax, 1.0 / _FP4_MAX)
  y = torch.clamp(xb / s, -_FP4_MAX, _FP4_MAX)
  q = _cast_fp4_e2m1(y)
  if inplace:
    deq = (q * s).flatten(-2).to(x.dtype)
    x.copy_(_ste(x, deq))
    return x
  # Unpacked logical fp4 values instead of float4_e2m1fn_x2 packing.
  return q.flatten(-2), s.squeeze(-1).to(torch.float8_e8m0fnu)


def fp8_gemm(*args, **kwargs):
  raise NotImplementedError("fp8_gemm: only bf16/fp32/fp64 weights are supported")


def fp4_gemm(*args, **kwargs):
  raise NotImplementedError("fp4_gemm: only bf16/fp32/fp64 weights are supported")


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
  """Index-gathered MQA with attention sink (lines 276-368).

  q: [b, m, h, d]; kv: [b, n, d]; attn_sink: [h]; topk_idxs: [b, m, topk]
  (-1 = masked). Emulates the kernel's 64-wide online softmax: the running max
  excludes the sink, probabilities are cast to q.dtype before the PV gemm
  (acc_s_cast), row sums use the uncast values, and the sink adds
  exp(sink - max) to the denominator only.
  """
  b, m, _, d = q.shape
  topk = topk_idxs.size(-1)
  cdt = torch.promote_types(q.dtype, torch.float32)
  idx = topk_idxs.long()
  # Upstream masks exactly idx == -1 (kernel.py:325,327); other negatives are not special-cased.
  valid = idx != -1
  gidx = torch.where(valid, idx, 0)
  kvg = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2, gidx.unsqueeze(-1).expand(b, m, topk, d))
  kvg = torch.where(valid.unsqueeze(-1), kvg, kvg.new_zeros(()))
  kvg = kvg.to(cdt)
  scores = torch.einsum("bmhd,bmkd->bmhk", q.to(cdt), kvg)
  scores = torch.where(valid.unsqueeze(2), scores, scores.new_full((), float("-inf")))
  scores = scores * softmax_scale
  nb = -(-topk // _ATTN_BLOCK)
  pad = nb * _ATTN_BLOCK - topk
  if pad:
    scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
  sb = scores.unflatten(-1, (nb, _ATTN_BLOCK))
  # Max is a pure shift; detached so gradients equal the exact softmax's.
  with torch.no_grad():
    m_run = torch.cummax(sb.amax(-1), dim=-1).values
    m_fin = m_run[..., -1:]
    factor = torch.exp(m_run - m_fin)
  p = torch.exp(sb - m_run.unsqueeze(-1))
  p_cast = p.to(q.dtype).to(cdt)
  sum_exp = (p.sum(-1) * factor).sum(-1)
  sum_exp = sum_exp + torch.exp(attn_sink.to(cdt) - m_fin.squeeze(-1))
  w = (p_cast * factor.unsqueeze(-1)).flatten(-2)
  if pad:
    w = w[..., :topk]
  o = torch.einsum("bmhk,bmkd->bmhd", w, kvg) / sum_exp.unsqueeze(-1)
  return o.to(q.dtype)


def hc_split_sinkhorn(  # pylint: disable=too-many-positional-arguments
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
  """Hyper-connection pre/post/comb split + Sinkhorn (lines 371-438); upstream signature."""
  hc = hc_mult
  pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
  post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
  comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
  # comb = softmax(comb, -1) + eps; comb /= colsum + eps
  comb = torch.exp(comb - comb.amax(-1, keepdim=True))
  comb = comb / comb.sum(-1, keepdim=True) + eps
  comb = comb / (comb.sum(-2, keepdim=True) + eps)
  for _ in range(sinkhorn_iters - 1):
    comb = comb / (comb.sum(-1, keepdim=True) + eps)
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
  return pre, post, comb
