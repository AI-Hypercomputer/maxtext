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

"""Plain-PyTorch replacements for DeepSeek-V4.1-Flash `inference/kernel.py`.

The upstream release (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference) implements
these entry points as TileLang GPU kernels, several of which simulate the FP8 / FP4 numerics used in training:
  * `act_quant(..., inplace=True)` quantizes each `block_size` group to FP8 E4M3 (amax floor `1e-4`, power-of-two
    UE8M0 scale `2 ** ceil(log2(amax / 448))` when `scale_fmt` is set) and dequantizes back in place; upstream
    `linear()` applies the same FP8 round trip to the input of every FP8 / FP4 GEMM.
  * `fp4_act_quant(..., inplace=True)` quantizes each `block_size` group to FP4 E2M1 (round-to-nearest-even) with
    either a power-of-two UE8M0 scale (amax floor `6 * 2**-126`, indexer `q`/`k` at block 32) or an E4M3 scale
    (amax floor `6 * 2**-9`, compressed KV at block 16) and dequantizes back in place.

By default `activation_quant` is `False` so unquantized reference callers (`ds41_activation_quant=False`) stay
identity; `set_activation_quant(True)` enables both in-place round trips and `model.linear()` input quantization.
The TileLang `fp8_gemm` / `fp4_gemm` entry points and non-inplace quantizers still raise because `Linear` stores
dequantized floating-point weights, and `sparse_attn` / `hc_split_sinkhorn` run as plain PyTorch ops on CPU.
"""

from contextlib import contextmanager
from typing import Optional

import torch

activation_quant = False


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
  """2 ** ceil(log2(v)) for positive normal float32 v (`fast_round_scale` in upstream `inference/kernel.py`)."""
  m, e = torch.frexp(v)
  return torch.ldexp(torch.ones_like(v), torch.where(m > 0.5, e, e - 1))


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
  """Round-to-nearest-even onto the FP4 E2M1 grid on [-6, 6]."""
  a = x.abs()
  q = torch.zeros_like(a)
  for edge, value, inclusive in (
      (0.25, 0.5, False),
      (0.75, 1.0, True),
      (1.25, 1.5, False),
      (1.75, 2.0, True),
      (2.5, 3.0, False),
      (3.5, 4.0, True),
      (5.0, 6.0, False),
  ):
    q = torch.where((a >= edge) if inclusive else (a > edge), value, q)
  return torch.where(x < 0, -q, q)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
  """Upstream `inference/kernel.py` block-wise FP8 E4M3 quantization (in-place QDQ when `activation_quant` is on)."""
  del scale_dtype
  if not inplace:
    raise NotImplementedError("act_quant without inplace=True is only used by quantized GEMMs.")
  if not activation_quant:
    return x
  assert x.size(-1) % block_size == 0, (x.shape, block_size)
  g = x.float().unflatten(-1, (-1, block_size))
  raw_scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) * (1.0 / 448.0)
  s = _pow2_ceil(raw_scale) if scale_fmt is not None else raw_scale
  y = (g / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
  x.copy_(y.flatten(-2).to(x.dtype))
  return x


act_quant.enabled = False  # type: ignore[attr-defined]


@contextmanager
def set_activation_quant(enabled: bool = True):
  """Toggles upstream FP8/FP4 activation quantize-dequantize round trips (`ds41_activation_quant`)."""
  global activation_quant  # pylint: disable=global-statement
  prev = activation_quant
  activation_quant = enabled
  act_quant.enabled = enabled  # type: ignore[attr-defined]
  try:
    yield
  finally:
    activation_quant = prev
    act_quant.enabled = prev  # type: ignore[attr-defined]


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
  """Upstream `inference/kernel.py` block-wise FP4 E2M1 quantization (in-place QDQ when `activation_quant` is on)."""
  if not inplace:
    raise NotImplementedError("fp4_act_quant without inplace=True is not used by the reference.")
  if not activation_quant:
    return x
  assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn), scale_dtype
  assert x.size(-1) % block_size == 0, (x.shape, block_size)
  g = x.float().unflatten(-1, (-1, block_size))
  amax = g.abs().amax(dim=-1, keepdim=True)
  if scale_dtype == torch.float8_e4m3fn:
    s = (amax.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
  else:
    s = _pow2_ceil(amax.clamp_min(6.0 * 2.0**-126) * (1.0 / 6.0))
  y = _round_e2m1((g / s).clamp(-6.0, 6.0)) * s
  x.copy_(y.flatten(-2).to(x.dtype))
  return x


def fp8_gemm(*args, **kwargs):
  raise NotImplementedError("The unquantized reference never runs FP8 GEMMs.")


def fp4_gemm(*args, **kwargs):
  raise NotImplementedError("The unquantized reference never runs FP4 GEMMs.")


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
  """Sparse multi-query attention with a learnable per-head attention sink.

  Equivalent to upstream's `sparse_attn_kernel`: every query attends to the KV rows listed in `topk_idxs`
  (-1 marks an unused slot), `kv` serves as both key and value, and `exp(attn_sink[h])` is added to the softmax
  denominator. A query with no valid slot returns zeros, matching the kernel's finite running-max convention.

  Args:
    q: [b, m, h, d] queries.
    kv: [b, n, d] shared key/value rows.
    attn_sink: [h] per-head sink logits (float32).
    topk_idxs: [b, m, topk] int indices into `kv`'s n axis, -1 for unused slots.
    softmax_scale: scale applied to q.k.

  Returns:
    [b, m, h, d] attention output in q's dtype.
  """
  b, m, h, _ = q.shape
  n = kv.size(1)
  idx = topk_idxs.long()
  valid = idx >= 0
  # Scatter the index lists into a dense [b, m, n] visibility mask. Indices within a row are unique.
  mask = torch.zeros(b, m, n + 1, dtype=torch.bool, device=q.device)
  mask.scatter_(2, torch.where(valid, idx, n), True)
  mask = mask[..., :n]

  qf, kvf = q.float(), kv.float()
  scores = torch.einsum("bmhd,bnd->bhmn", qf, kvf) * softmax_scale
  scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
  sink = attn_sink.float().view(1, h, 1, 1).expand(b, h, m, 1)
  logits = torch.cat([scores, sink], dim=-1)
  probs = torch.softmax(logits, dim=-1)[..., :n]
  o = torch.einsum("bhmn,bnd->bmhd", probs, kvf)
  return o.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
  """Splits the Hyper-Connection mixes into pre / post / comb coefficients (upstream `hc_split_sinkhorn_kernel`).

  Args:
    mixes: [b, s, (2 + hc) * hc] float32.
    hc_scale: [3] float32.
    hc_base: [(2 + hc) * hc] float32.

  Returns:
    pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc] (comb made doubly stochastic by Sinkhorn).
  """
  hc = hc_mult
  mixes = mixes.float()
  pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
  post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
  comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
  comb = comb.softmax(dim=-1) + eps
  comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
  for _ in range(sinkhorn_iters - 1):
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
  return pre, post, comb
