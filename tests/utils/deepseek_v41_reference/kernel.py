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

"""Plain, unquantized PyTorch replacements for DeepSeek-V4.1-Flash `inference/kernel.py`.

The upstream reference (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference) implements
these entry points as TileLang GPU kernels, several of which simulate the FP8 / FP4 numerics used in training.
This module keeps the exact call signatures so `model.py` can stay close to upstream, but:

  * `act_quant` / `fp4_act_quant` with `inplace=True` (the fused quantize->dequantize round trip that upstream
    applies to KV, index keys/queries and the attention output) are the identity: the reference runs unquantized.
  * The quantized GEMMs (`fp8_gemm`, `fp4_gemm`) and the non-inplace quantizers are never reached, because every
    `Linear` holds a plain floating-point weight. They raise to make any accidental use obvious.
  * `sparse_attn` and `hc_split_sinkhorn` are re-written as straightforward PyTorch ops that run on CPU.
"""

from typing import Optional

import torch


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
  """Upstream: block-wise FP8 quantization. Unquantized reference: identity for the in-place round trip."""
  del block_size, scale_fmt, scale_dtype
  if inplace:
    return x
  raise NotImplementedError("act_quant without inplace=True is only used by quantized GEMMs.")


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
  """Upstream: block-wise FP4 quantization. Unquantized reference: identity for the in-place round trip."""
  del block_size, scale_dtype
  if inplace:
    return x
  raise NotImplementedError("fp4_act_quant without inplace=True is not used by the reference.")


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
