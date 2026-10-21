# Copyright 2023-2026 Google LLC
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

"""Long-context (L=2048, 4096, 8192) MaxText vs. PyTorch OLMo 3.5 parity tests.

Verifies layer-by-layer hidden states, forward logits, position-window stability
across 8,192 tokens, multi-document EOS-segmented EMo routing, auxiliary losses
(load-balance + router z-loss), and backward parameter gradients between:
  1. PyTorch OLMo 3.5 reference (`allenai/OLMo-core` / `standalone_model.py`)
  2. MaxText OLMoE3 XLA Reference (`kda_pallas_fused=False`)
  3. MaxText OLMoE3 Production Pallas Stack (`kda_pallas_fused=True`,
     `kda_flat_head_layout=True`, `moe_fused_swiglu_gmm=True`,
     `fused_lm_head_ce=True`)
"""

import gc
import math
import os
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
import maxtext.models.models as models
from maxtext.models import olmoe3
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import vocabulary_tiling
from maxtext.utils.globals import MAXTEXT_PKG_DIR


# -----------------------------------------------------------------------------
# Metrics & Comparison Helpers
# -----------------------------------------------------------------------------


def _unbox_jax(x):
  while hasattr(x, "value"):
    x = x.value
  return x


def _to_f32_np(x) -> np.ndarray:
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().float().numpy()
  return np.asarray(jax.device_get(_unbox_jax(x)), dtype=np.float32)


def _compare_metrics(a, b) -> dict[str, float]:
  """Computes max_abs, mean_abs, rel_rms, cos_sim, and top1_acc in streaming blocks."""
  a_np = _to_f32_np(a)
  b_np = _to_f32_np(b)
  last_dim = a_np.shape[-1] if a_np.ndim >= 1 else 1
  rows = a_np.size // max(last_dim, 1)
  a_2d = a_np.reshape(rows, last_dim) if a_np.flags["C_CONTIGUOUS"] else None
  b_2d = b_np.reshape(rows, last_dim) if b_np.flags["C_CONTIGUOUS"] else None
  n = a_np.size
  row_block = max(1, 4_000_000 // max(last_dim, 1))
  max_abs = 0.0
  sum_abs = 0.0
  sum_sq_diff = 0.0
  sum_sq_a = 0.0
  sum_sq_b = 0.0
  dot_ab = 0.0
  if a_2d is not None and b_2d is not None:
    for r in range(0, rows, row_block):
      ai = a_2d[r : r + row_block].reshape(-1).astype(np.float64)
      bi = b_2d[r : r + row_block].reshape(-1).astype(np.float64)
      d = ai - bi
      ad = np.abs(d)
      if ad.size:
        max_abs = max(max_abs, float(np.max(ad)))
      sum_abs += float(np.sum(ad))
      sum_sq_diff += float(np.dot(d, d))
      sum_sq_a += float(np.dot(ai, ai))
      sum_sq_b += float(np.dot(bi, bi))
      dot_ab += float(np.dot(ai, bi))
  else:
    for idx in np.ndindex(a_np.shape[:-1]):
      ai = a_np[idx].reshape(-1).astype(np.float64)
      bi = b_np[idx].reshape(-1).astype(np.float64)
      d = ai - bi
      ad = np.abs(d)
      if ad.size:
        max_abs = max(max_abs, float(np.max(ad)))
      sum_abs += float(np.sum(ad))
      sum_sq_diff += float(np.dot(d, d))
      sum_sq_a += float(np.dot(ai, ai))
      sum_sq_b += float(np.dot(bi, bi))
      dot_ab += float(np.dot(ai, bi))
  mean_abs = sum_abs / max(n, 1)
  rms_diff = math.sqrt(sum_sq_diff / max(n, 1))
  rms_ref = math.sqrt(sum_sq_b / max(n, 1)) + 1e-12
  rel_rms = rms_diff / rms_ref
  denom = (math.sqrt(sum_sq_a) * math.sqrt(sum_sq_b)) + 1e-12
  cos_sim = dot_ab / denom
  metrics = {
      "max_abs": max_abs,
      "mean_abs": mean_abs,
      "rel_rms": rel_rms,
      "cos_sim": cos_sim,
  }
  if a_np.ndim == 3 and a_np.shape[-1] > 1000:
    matches = 0
    total_tok = a_np.shape[0] * a_np.shape[1]
    for b_idx in range(a_np.shape[0]):
      for s in range(0, a_np.shape[1], 512):
        top1_a = np.argmax(a_np[b_idx, s : s + 512], axis=-1)
        top1_b = np.argmax(b_np[b_idx, s : s + 512], axis=-1)
        matches += int(np.sum(top1_a == top1_b))
    metrics["top1_acc"] = matches / max(total_tok, 1)
  return metrics


def _compute_logit_alignment_metrics(
    logits_mt,
    logits_pt,
    topk_list: tuple[int, ...] = (1, 5, 10, 20, 50),
) -> dict[str, float]:
  """Computes comprehensive end-to-end logit alignment metrics on [B, L, V] logits.

  Measures:
    - Full-sequence & last-token (next-token prediction at t=L-1) cosine similarity
    - First / top-1 logit value alignment (|max_v MT - max_v PT| and at PT's argmax)
    - Top-1 and Top-5 token prediction agreement (%)
    - KL divergence KL(P_PT || P_MT) and KL(P_MT || P_PT) in nats
    - Jaccard similarity @k for k in topk_list (|S_MT,k n S_PT,k| / |S_MT,k u S_PT,k|)
  """
  mt_np = _to_f32_np(logits_mt)
  pt_np = _to_f32_np(logits_pt)
  assert mt_np.shape == pt_np.shape and mt_np.ndim == 3
  bsz, seq_len, _ = mt_np.shape
  total_tok = bsz * seq_len

  base = _compare_metrics(mt_np, pt_np)
  max_k = max(topk_list)

  top1_matches = 0
  top5_exact_matches = 0
  top1_in_pt_top5 = 0
  first_logit_abs_diffs = []
  pt_argmax_logit_abs_diffs = []
  mt_first_logits = []
  pt_first_logits = []
  kl_pt_to_mt_sum = 0.0
  kl_mt_to_pt_sum = 0.0
  tail_kl_pt_to_mt_sum = 0.0
  tail_top1_matches = 0
  tail_len = min(512, seq_len)
  tail_start = seq_len - tail_len
  jaccard_sums = {k: 0.0 for k in topk_list}
  tail_jaccard10_sum = 0.0

  chunk_s = 256
  for b_idx in range(bsz):
    for s in range(0, seq_len, chunk_s):
      e = min(s + chunk_s, seq_len)
      cur_len = e - s
      mt_t = torch.from_numpy(mt_np[b_idx, s:e]).float()  # [cur_len, V]
      pt_t = torch.from_numpy(pt_np[b_idx, s:e]).float()  # [cur_len, V]

      mt_top_vals, mt_top_idx = torch.topk(mt_t, k=max_k, dim=-1)
      pt_top_vals, pt_top_idx = torch.topk(pt_t, k=max_k, dim=-1)

      mt_top1_val = mt_top_vals[:, 0]
      pt_top1_val = pt_top_vals[:, 0]
      mt_top1_id = mt_top_idx[:, 0]
      pt_top1_id = pt_top_idx[:, 0]

      mt_first_logits.append(mt_top1_val.numpy())
      pt_first_logits.append(pt_top1_val.numpy())
      first_logit_abs_diffs.append(torch.abs(mt_top1_val - pt_top1_val).numpy())

      mt_at_pt_top1 = torch.gather(mt_t, dim=-1, index=pt_top1_id.unsqueeze(-1)).squeeze(-1)
      pt_argmax_logit_abs_diffs.append(torch.abs(mt_at_pt_top1 - pt_top1_val).numpy())

      t1_eq = (mt_top1_id == pt_top1_id)
      top1_matches += int(t1_eq.sum().item())
      top5_exact_matches += int(torch.all(mt_top_idx[:, :5] == pt_top_idx[:, :5], dim=-1).sum().item())
      top1_in_pt_top5 += int((mt_top1_id.unsqueeze(-1) == pt_top_idx[:, :5]).any(dim=-1).sum().item())

      # Jaccard similarity @k: intersection / (2*k - intersection)
      for k in topk_list:
        mt_k = mt_top_idx[:, :k].unsqueeze(2)  # [cur_len, k, 1]
        pt_k = pt_top_idx[:, :k].unsqueeze(1)  # [cur_len, 1, k]
        inter = (mt_k == pt_k).any(dim=-1).float().sum(dim=-1)  # [cur_len]
        jacc = inter / torch.clamp(2.0 * k - inter, min=1.0)
        jaccard_sums[k] += float(jacc.sum().item())
        if k == 10 and e > tail_start:
          ts = max(0, tail_start - s)
          tail_jaccard10_sum += float(jacc[ts:].sum().item())

      # KL divergence in float32 log_softmax
      log_p_pt = F.log_softmax(pt_t, dim=-1)
      log_p_mt = F.log_softmax(mt_t, dim=-1)
      p_pt = torch.exp(log_p_pt)
      p_mt = torch.exp(log_p_mt)
      kl_pt_mt = torch.sum(p_pt * (log_p_pt - log_p_mt), dim=-1)  # [cur_len]
      kl_mt_pt = torch.sum(p_mt * (log_p_mt - log_p_pt), dim=-1)  # [cur_len]
      kl_pt_to_mt_sum += float(kl_pt_mt.sum().item())
      kl_mt_to_pt_sum += float(kl_mt_pt.sum().item())

      if e > tail_start:
        ts = max(0, tail_start - s)
        tail_top1_matches += int(t1_eq[ts:].sum().item())
        tail_kl_pt_to_mt_sum += float(kl_pt_mt[ts:].sum().item())

  mt_first_all = np.concatenate(mt_first_logits)
  pt_first_all = np.concatenate(pt_first_logits)
  first_diffs_all = np.concatenate(first_logit_abs_diffs)
  pt_argmax_diffs_all = np.concatenate(pt_argmax_logit_abs_diffs)

  first_cos = float(
      np.dot(mt_first_all.astype(np.float64), pt_first_all.astype(np.float64))
      / (
          np.linalg.norm(mt_first_all.astype(np.float64))
          * np.linalg.norm(pt_first_all.astype(np.float64))
          + 1e-12
      )
  )

  # Next-token prediction at the final position t = seq_len - 1
  last_mt = torch.from_numpy(mt_np[0, -1]).float()
  last_pt = torch.from_numpy(pt_np[0, -1]).float()
  last_mt_vals, last_mt_idx = torch.topk(last_mt, k=max_k)
  last_pt_vals, last_pt_idx = torch.topk(last_pt, k=max_k)
  last_cos = float(
      F.cosine_similarity(last_mt.unsqueeze(0), last_pt.unsqueeze(0), dim=-1).item()
  )
  last_log_pt = F.log_softmax(last_pt, dim=-1)
  last_log_mt = F.log_softmax(last_mt, dim=-1)
  last_kl_pt_mt = float(
      torch.sum(torch.exp(last_log_pt) * (last_log_pt - last_log_mt)).item()
  )
  last_jacc = {}
  for k in topk_list:
    s_mt = set(last_mt_idx[:k].tolist())
    s_pt = set(last_pt_idx[:k].tolist())
    last_jacc[k] = len(s_mt & s_pt) / max(len(s_mt | s_pt), 1)

  tail_m = _compare_metrics(mt_np[:, -tail_len:, :], pt_np[:, -tail_len:, :])

  out = {
      **base,
      "top1_acc": top1_matches / max(total_tok, 1),
      "top5_exact_acc": top5_exact_matches / max(total_tok, 1),
      "top1_in_pt_top5": top1_in_pt_top5 / max(total_tok, 1),
      "first_logit_mae": float(np.mean(first_diffs_all)),
      "first_logit_max_diff": float(np.max(first_diffs_all)),
      "first_logit_cos_sim": first_cos,
      "pt_top1_logit_mae": float(np.mean(pt_argmax_diffs_all)),
      "kl_pt_to_mt": kl_pt_to_mt_sum / max(total_tok, 1),
      "kl_mt_to_pt": kl_mt_to_pt_sum / max(total_tok, 1),
      "tail512_cos_sim": tail_m["cos_sim"],
      "tail512_top1_acc": tail_top1_matches / max(bsz * tail_len, 1),
      "tail512_kl_pt_to_mt": tail_kl_pt_to_mt_sum / max(bsz * tail_len, 1),
      "tail512_jaccard@10": tail_jaccard10_sum / max(bsz * tail_len, 1),
      "last_tok_mt_first_logit": float(last_mt_vals[0].item()),
      "last_tok_pt_first_logit": float(last_pt_vals[0].item()),
      "last_tok_first_logit_diff": float(torch.abs(last_mt_vals[0] - last_pt_vals[0]).item()),
      "last_tok_mt_top1_id": int(last_mt_idx[0].item()),
      "last_tok_pt_top1_id": int(last_pt_idx[0].item()),
      "last_tok_top1_match": float(int(last_mt_idx[0].item() == last_pt_idx[0].item())),
      "last_tok_cos_sim": last_cos,
      "last_tok_kl_pt_to_mt": last_kl_pt_mt,
  }
  for k in topk_list:
    out[f"jaccard@{k}"] = jaccard_sums[k] / max(total_tok, 1)
    out[f"last_tok_jaccard@{k}"] = last_jacc[k]
  return out


# -----------------------------------------------------------------------------
# PyTorch Reference Implementation of OLMo 3.5 (`standalone_model.py`)
# -----------------------------------------------------------------------------


def _pt_matmul(a: torch.Tensor, b: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
  """Matmul with `dt` operands/output and FP32 accumulation (avoids slow CPU scalar bfloat16 fallback)."""
  a_dt = a.to(dt)
  b_dt = b.to(dt)
  if dt == torch.bfloat16:
    return (a_dt.float() @ b_dt.float()).to(dt)
  return a_dt @ b_dt


def _pt_einsum(eq: str, a: torch.Tensor, b: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
  """Einsum with `dt` operands/output and FP32 accumulation (avoids slow CPU scalar bfloat16 fallback)."""
  a_dt = a.to(dt)
  b_dt = b.to(dt)
  if dt == torch.bfloat16:
    return torch.einsum(eq, a_dt.float(), b_dt.float()).to(dt)
  return torch.einsum(eq, a_dt, b_dt)


def torch_rms_norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Matches MaxText / OLMo-core RMSNorm (variance in fp32, gain in compute_dtype)."""
  x_f32 = x.float()
  mean2 = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
  y = (x_f32 * torch.rsqrt(mean2 + eps)).to(compute_dtype)
  s = scale.to(compute_dtype)
  return (y.float() * s.float()).to(compute_dtype) if compute_dtype == torch.bfloat16 else y * s


def torch_causal_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
  """Causal depthwise conv1d over time + SiLU, with packed-document masking.

  Args:
    x: [B, T, C]
    weight: [C, K] where tap=K-1 is lag 0 and tap=0 is lag K-1.
    segment_ids: Optional [B, T] int32 segment IDs.
  """
  _, seq_len, _ = x.shape
  kernel_size = weight.shape[-1]
  x_f = x.float()
  w_f = weight.float()
  acc = torch.zeros_like(x_f)
  for tap in range(kernel_size):
    lag = kernel_size - 1 - tap
    if lag == 0:
      shifted = x_f
    else:
      shifted = F.pad(x_f, (0, 0, lag, 0))[:, :seq_len, :]
      if segment_ids is not None:
        shifted_ids = F.pad(segment_ids, (lag, 0), value=-1)[:, :seq_len]
        mask = (shifted_ids == segment_ids).unsqueeze(-1)
        shifted = torch.where(mask, shifted, torch.zeros_like(shifted))
    acc = acc + shifted * w_f[:, tap]
  return F.silu(acc.to(x.dtype).float()).to(x.dtype)


def torch_l2_normalize(x: torch.Tensor, eps: float = 1e-6, eps_inside: bool = True) -> torch.Tensor:
  """Per-head L2 normalization in fp32."""
  x_f = x.float()
  sum_sq = torch.sum(x_f * x_f, dim=-1, keepdim=True)
  if eps_inside:
    return x_f * torch.rsqrt(sum_sq + eps)
  return x_f * torch.rsqrt(torch.clamp(sum_sq, min=eps))


def torch_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    resets: torch.Tensor,
) -> torch.Tensor:
  """Exact step-by-step recurrent Kimi Delta Attention in float32.

  S_t = exp(log_decay_t) * S_{t-1} + k_t (x) (beta_t * (v_t - S_{t-1}^T k_t))
  o_t = S_t^T q_t
  """
  bsz, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  q_f = q.float()
  k_f = k.float()
  v_f = v.float()
  decay_f = torch.exp(log_decay.float())
  beta_f = beta.float()

  state = torch.zeros((bsz, heads, dk, dv), dtype=torch.float32, device=q.device)
  outputs = []
  for t in range(seq_len):
    rst = resets[:, t].view(bsz, 1, 1, 1)
    state = torch.where(rst, torch.zeros_like(state), state)
    qt = q_f[:, t]          # [B, H, dk]
    kt = k_f[:, t]          # [B, H, dk]
    vt = v_f[:, t]          # [B, H, dv]
    dt = decay_f[:, t]      # [B, H, dk]
    bt = beta_f[:, t]       # [B, H]

    state = state * dt.unsqueeze(-1)
    proj = torch.einsum("bhkv,bhk->bhv", state, kt)
    delta = bt.unsqueeze(-1) * (vt - proj)
    state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
    ot = torch.einsum("bhkv,bhk->bhv", state, qt)
    outputs.append(ot)
  return torch.stack(outputs, dim=1)


def torch_kda_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    resets: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Chunked WY / triangular-inverse Kimi Delta Attention in PyTorch (float32).

  Mathematically identical to `torch_kda_recurrent` while executing only
  `T // chunk_size` recurrent steps over chunks via batched matrix multiplies.
  """
  bsz, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  if seq_len % chunk_size != 0:
    return torch_kda_recurrent(q, k, v, log_decay, beta, resets)

  num_chunks = seq_len // chunk_size
  c = chunk_size
  q_c = q.float().view(bsz, num_chunks, c, heads, dk).permute(0, 3, 1, 2, 4)
  k_c = k.float().view(bsz, num_chunks, c, heads, dk).permute(0, 3, 1, 2, 4)
  v_c = v.float().view(bsz, num_chunks, c, heads, dv).permute(0, 3, 1, 2, 4)
  g_c = log_decay.float().view(bsz, num_chunks, c, heads, dk).permute(0, 3, 1, 2, 4)
  b_c = beta.float().view(bsz, num_chunks, c, heads).permute(0, 3, 1, 2)
  r_c = resets.view(bsz, num_chunks, c)

  # Segment IDs within each chunk to reset cumulative sums at document boundaries
  seg_in_chunk = torch.cumsum(r_c.to(torch.int32), dim=-1)  # [B, N, C]
  same_seg = (seg_in_chunk.unsqueeze(-1) == seg_in_chunk.unsqueeze(-2))  # [B, N, C, C]
  same_seg_h = same_seg.unsqueeze(1)  # [B, 1, N, C, C]

  # Cumulative log_decay within each segment of each chunk
  idx = torch.arange(c, device=q.device)
  causal = idx.unsqueeze(-1) >= idx.unsqueeze(-2)  # [C, C]
  strict_lower = idx.unsqueeze(-1) > idx.unsqueeze(-2)  # [C, C]
  mask_cum = causal.view(1, 1, 1, c, c) & same_seg_h  # [B, 1, N, C, C]
  g_cum = torch.einsum("bhnij,bhnjd->bhnid", mask_cum.float(), g_c)  # [B, H, N, C, dk]

  # Pairwise log decay g_cum[i] - g_cum[j] for i >= j in same segment
  g_diff = g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)  # [B, H, N, C, C, dk]
  g_diff = torch.where(mask_cum.unsqueeze(-1), g_diff, torch.zeros_like(g_diff))
  decay_pair = torch.where(
      mask_cum.unsqueeze(-1), torch.exp(g_diff), torch.zeros_like(g_diff)
  )

  # M_{i,j} = beta_i * <k_i * exp(g_i - g_j), k_j> for i > j
  k_scaled = k_c.unsqueeze(-2) * decay_pair  # [B, H, N, C, C, dk]
  kk = torch.einsum("bhnijd,bhnjd->bhnij", k_scaled, k_c)  # [B, H, N, C, C]
  m_mat = b_c.unsqueeze(-1) * (kk * (strict_lower.view(1, 1, 1, c, c) & same_seg_h).float())

  eye = torch.eye(c, dtype=torch.float32, device=q.device).view(1, 1, 1, c, c)
  m_inv = torch.linalg.solve_triangular(eye + m_mat, eye, upper=False, unitriangular=True)

  v_beta = v_c * b_c.unsqueeze(-1)
  k_cum_beta = k_c * torch.exp(g_cum) * b_c.unsqueeze(-1)
  u_c = torch.einsum("bhnij,bhnjv->bhniv", m_inv, v_beta)
  w_c = torch.einsum("bhnij,bhnjk->bhnik", m_inv, k_cum_beta)

  # Intra-chunk attention scores: A_{i,j} = <q_i * exp(g_i - g_j), k_j> for i >= j
  q_scaled = q_c.unsqueeze(-2) * decay_pair
  a_qk = torch.einsum("bhnijd,bhnjd->bhnij", q_scaled, k_c) * mask_cum.float()

  q_cum = q_c * torch.exp(g_cum)

  # Last position in each segment within the chunk
  next_seg = F.pad(seg_in_chunk[:, :, 1:], (0, 1), value=-1)
  is_last_in_seg = (seg_in_chunk != next_seg)  # [B, N, C]
  pos_grid = idx.view(1, 1, c).expand(bsz, num_chunks, c)
  last_marker = torch.where(is_last_in_seg, pos_grid, torch.full_like(pos_grid, c - 1))
  last_idx = torch.clamp(
      torch.flip(torch.cummin(torch.flip(last_marker, dims=[-1]), dim=-1)[0], dims=[-1]),
      min=0,
      max=c - 1,
  )  # [B, N, C]
  g_last = torch.gather(
      g_cum, dim=3, index=last_idx.unsqueeze(1).unsqueeze(-1).expand(-1, heads, -1, -1, dk)
  )
  k_decay_to_end = k_c * torch.exp(torch.clamp(g_last - g_cum, max=0.0))

  # Scan across chunks
  state = torch.zeros((bsz, heads, dk, dv), dtype=torch.float32, device=q.device)
  out_chunks = []
  for n in range(num_chunks):
    # Check if chunk n has any resets
    r_n = r_c[:, n, :]  # [B, C]
    if not torch.any(r_n):
      u_eff = u_c[:, :, n] - torch.einsum("bhcd,bhdk->bhck", w_c[:, :, n], state)
      o_n = (
          torch.einsum("bhcd,bhdk->bhck", q_cum[:, :, n], state)
          + torch.einsum("bhij,bhjv->bhiv", a_qk[:, :, n], u_eff)
      )
      state = state * torch.exp(g_cum[:, :, n, -1, :]).unsqueeze(-1) + torch.einsum(
          "bhcd,bhcv->bhdv", k_decay_to_end[:, :, n], u_eff
      )
      out_chunks.append(o_n)
    else:
      # Fallback per-step within chunk when resets occur inside the chunk
      q_n = q_c[:, :, n].permute(0, 2, 1, 3)
      k_n = k_c[:, :, n].permute(0, 2, 1, 3)
      v_n = v_c[:, :, n].permute(0, 2, 1, 3)
      g_n = g_c[:, :, n].permute(0, 2, 1, 3)
      b_n = b_c[:, :, n].permute(0, 2, 1)
      o_n_list = []
      for t in range(c):
        rst = r_n[:, t].view(bsz, 1, 1, 1)
        state = torch.where(rst, torch.zeros_like(state), state)
        state = state * torch.exp(g_n[:, t]).unsqueeze(-1)
        proj = torch.einsum("bhkv,bhk->bhv", state, k_n[:, t])
        delta = b_n[:, t].unsqueeze(-1) * (v_n[:, t] - proj)
        state = state + torch.einsum("bhk,bhv->bhkv", k_n[:, t], delta)
        o_n_list.append(torch.einsum("bhkv,bhk->bhv", state, q_n[:, t]))
      out_chunks.append(torch.stack(o_n_list, dim=2))

  out = torch.stack(out_chunks, dim=2)  # [B, H, N, C, dv]
  return out.permute(0, 2, 3, 1, 4).contiguous().view(bsz, seq_len, heads, dv)


class PyTorchKDA(nn.Module):
  """PyTorch OLMo 3.5 Kimi Delta Attention layer."""

  def __init__(self, cfg, compute_dtype: torch.dtype):
    super().__init__()
    self.cfg = cfg
    self.compute_dtype = compute_dtype
    self.num_heads = cfg.gdn_num_value_heads
    self.dk = cfg.gdn_key_head_dim
    self.dv = cfg.gdn_value_head_dim
    key_width = self.num_heads * self.dk
    val_width = self.num_heads * self.dv
    conv_size = cfg.gdn_conv_kernel_dim

    self.w_q = nn.Parameter(torch.zeros(cfg.emb_dim, key_width))
    self.w_k = nn.Parameter(torch.zeros(cfg.emb_dim, key_width))
    self.w_v = nn.Parameter(torch.zeros(cfg.emb_dim, val_width))
    self.q_conv = nn.Parameter(torch.zeros(key_width, conv_size))
    self.k_conv = nn.Parameter(torch.zeros(key_width, conv_size))
    self.v_conv = nn.Parameter(torch.zeros(val_width, conv_size))
    self.f_proj_1 = nn.Parameter(torch.zeros(cfg.emb_dim, self.dv))
    self.f_proj_2 = nn.Parameter(torch.zeros(self.dv, key_width))
    self.w_b = nn.Parameter(torch.zeros(cfg.emb_dim, self.num_heads))
    self.A_log = nn.Parameter(torch.zeros(self.num_heads))
    self.dt_bias = nn.Parameter(torch.zeros(key_width))
    self.g_proj_1 = nn.Parameter(torch.zeros(cfg.emb_dim, self.dv))
    self.g_proj_2_kernel = nn.Parameter(torch.zeros(self.dv, val_width))
    self.g_proj_2_bias = nn.Parameter(torch.zeros(val_width))
    self.o_norm_scale = nn.Parameter(torch.ones(self.dv))
    self.w_out = nn.Parameter(torch.zeros(val_width, cfg.emb_dim))

  def forward(
      self,
      x: torch.Tensor,
      segment_ids: torch.Tensor | None = None,
      use_recurrent: bool = False,
  ) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    h, dk, dv = self.num_heads, self.dk, self.dv
    dt = self.compute_dtype
    x_dt = x.to(dt)

    q_in = _pt_matmul(x_dt, self.w_q, dt)
    k_in = _pt_matmul(x_dt, self.w_k, dt)
    v_in = _pt_matmul(x_dt, self.w_v, dt)
    f_in = _pt_matmul(x_dt, self.f_proj_1, dt)
    g_in = _pt_matmul(x_dt, self.g_proj_1, dt)
    b_in = _pt_matmul(x_dt, self.w_b, dt)

    conv_dt = dt if self.cfg.kda_conv_in_compute_dtype else self.q_conv.dtype
    q = torch_causal_depthwise_conv(q_in.to(conv_dt), self.q_conv.to(conv_dt), segment_ids).view(bsz, seq_len, h, dk)
    k = torch_causal_depthwise_conv(k_in.to(conv_dt), self.k_conv.to(conv_dt), segment_ids).view(bsz, seq_len, h, dk)
    v = torch_causal_depthwise_conv(v_in.to(conv_dt), self.v_conv.to(conv_dt), segment_ids).view(bsz, seq_len, h, dv)

    raw_g = _pt_matmul(f_in, self.f_proj_2, dt).view(bsz, seq_len, h, dk)
    beta_scale = 2.0 if self.cfg.kda_allow_neg_eigval else 1.0
    beta = beta_scale * torch.sigmoid(b_in.float())

    # L2 norm on q and k
    q = torch_l2_normalize(q, eps=1e-6, eps_inside=True) * (dk ** -0.5)
    k = torch_l2_normalize(k, eps=1e-6, eps_inside=True)

    log_decay = -torch.exp(self.A_log.float()).view(1, 1, h, 1) * F.softplus(
        raw_g.float() + self.dt_bias.float().view(1, 1, h, dk)
    )

    if segment_ids is None:
      resets = torch.zeros((bsz, seq_len), dtype=torch.bool, device=x.device)
    else:
      prev_ids = F.pad(segment_ids, (1, 0), value=-1)[:, :seq_len]
      resets = segment_ids != prev_ids

    # When dk >= 128 (e.g. olmoe3-3p5b dk=256), 6D chunked [B, H, N, C, C, dk] tensors at L=8192 are 4.3 GB each;
    # recurrent uses a single [B, H, dk, dv] state tensor (4 MB) with identical math.
    if use_recurrent or self.dk >= 128:
      out = torch_kda_recurrent(q, k, v.to(dt), log_decay, beta, resets)
    else:
      out = torch_kda_chunked(
          q, k, v.to(dt), log_decay, beta, resets, chunk_size=max(self.cfg.gdn_chunk_size, 1)
      )

    gate_lin = _pt_matmul(g_in, self.g_proj_2_kernel, dt) + self.g_proj_2_bias.to(dt)
    gate = torch.sigmoid(gate_lin.float().view(bsz, seq_len, h, dv)).to(dt)
    out_normed = torch_rms_norm(out.to(dt), self.o_norm_scale, eps=1e-5, compute_dtype=dt)
    gated_out = (out_normed.float() * gate.float()).to(dt).reshape(bsz, seq_len, h * dv)
    return _pt_matmul(gated_out, self.w_out, dt)


class PyTorchFullAttention(nn.Module):
  """PyTorch OLMo 3.5 Full Attention (GQA, NoPE, QK-norm, Scalable Softmax, Sigmoid Gate)."""

  def __init__(self, cfg, compute_dtype: torch.dtype):
    super().__init__()
    self.cfg = cfg
    self.compute_dtype = compute_dtype
    self.hq = cfg.num_query_heads
    self.hkv = cfg.num_kv_heads
    self.head_dim = cfg.head_dim

    self.query_kernel = nn.Parameter(torch.zeros(cfg.emb_dim, self.hq, 2 * self.head_dim))
    self.key_kernel = nn.Parameter(torch.zeros(cfg.emb_dim, self.hkv, self.head_dim))
    self.value_kernel = nn.Parameter(torch.zeros(cfg.emb_dim, self.hkv, self.head_dim))
    self.out_kernel = nn.Parameter(torch.zeros(self.hq * self.head_dim, cfg.emb_dim))

    if getattr(cfg, "qk_norm_per_head", False):
      self.query_norm_scale = nn.Parameter(torch.ones(self.hq, self.head_dim))
      self.key_norm_scale = nn.Parameter(torch.ones(self.hkv, self.head_dim))
    else:
      self.query_norm_scale = nn.Parameter(torch.ones(self.head_dim))
      self.key_norm_scale = nn.Parameter(torch.ones(self.head_dim))

    if getattr(cfg, "use_scalable_softmax", False):
      self.ssmax_scale = nn.Parameter(torch.ones(self.hq))
    else:
      self.ssmax_scale = None

  def forward(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    dt = self.compute_dtype
    x_dt = x.to(dt)

    qg = _pt_einsum("btd,dhk->bthk", x_dt, self.query_kernel, dt)
    q, gate = torch.split(qg, self.head_dim, dim=-1)
    gate = gate.reshape(bsz, seq_len, self.hq * self.head_dim)

    k = _pt_einsum("btd,dhk->bthk", x_dt, self.key_kernel, dt)
    v = _pt_einsum("btd,dhk->bthk", x_dt, self.value_kernel, dt)

    eps = self.cfg.normalization_layer_epsilon
    q = torch_rms_norm(q, self.query_norm_scale, eps=eps, compute_dtype=dt)
    k = torch_rms_norm(k, self.key_norm_scale, eps=eps, compute_dtype=dt)

    q = (q.float() * (self.head_dim ** -0.5)).to(dt)
    if self.ssmax_scale is not None:
      span = torch.log(positions.float() + 1.0).view(bsz, seq_len, 1, 1)
      gain = self.ssmax_scale.float().view(1, 1, self.hq, 1)
      q = (q.float() * span * gain).to(dt)

    # GQA repeat K and V
    rep = self.hq // self.hkv
    if rep > 1:
      k = k.repeat_interleave(rep, dim=2)
      v = v.repeat_interleave(rep, dim=2)

    # Causal + optional document segment mask (computed per head to bound peak RAM at L=8192)
    idx = torch.arange(seq_len, device=x.device)
    mask = idx.view(1, seq_len, 1) >= idx.view(1, 1, seq_len)
    if segment_ids is not None:
      same_seg = segment_ids.view(bsz, seq_len, 1) == segment_ids.view(bsz, 1, seq_len)
      mask = mask & same_seg
    inv_mask = ~mask
    attn_out = torch.empty_like(q)
    for h in range(self.hq):
      scores_h = torch.einsum("btd,bsd->bts", q[:, :, h, :].float(), k[:, :, h, :].float())
      scores_h.masked_fill_(inv_mask, -1e30)
      probs_h = F.softmax(scores_h, dim=-1).to(dt)
      attn_out[:, :, h, :] = _pt_einsum("bts,bsd->btd", probs_h, v[:, :, h, :], dt)
    attn_flat = attn_out.reshape(bsz, seq_len, self.hq * self.head_dim)
    attn_gated = (attn_flat.float() * torch.sigmoid(gate.float())).to(dt)
    return _pt_matmul(attn_gated, self.out_kernel, dt)


class PyTorchSwiGLU(nn.Module):
  """SwiGLU MLP block matching MaxText MlpBlock."""

  def __init__(self, in_dim: int, hidden_dim: int, compute_dtype: torch.dtype):
    super().__init__()
    self.compute_dtype = compute_dtype
    self.wi_0 = nn.Parameter(torch.zeros(in_dim, hidden_dim))
    self.wi_1 = nn.Parameter(torch.zeros(in_dim, hidden_dim))
    self.wo = nn.Parameter(torch.zeros(hidden_dim, in_dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dt = self.compute_dtype
    x_dt = x.to(dt)
    h0 = F.silu(_pt_matmul(x_dt, self.wi_0, dt).float()).to(dt)
    h1 = _pt_matmul(x_dt, self.wi_1, dt)
    return _pt_matmul((h0.float() * h1.float()).to(dt), self.wo, dt)


class PyTorchLatentMoE(nn.Module):
  """PyTorch OLMo 3.5 Latent Routed MoE with EMo document-pool routing."""

  def __init__(self, cfg, compute_dtype: torch.dtype):
    super().__init__()
    self.cfg = cfg
    self.compute_dtype = compute_dtype
    self.num_experts = cfg.num_experts
    self.top_k = cfg.num_experts_per_tok
    self.latent_dim = cfg.moe_expert_input_dim
    self.moe_mlp_dim = cfg.moe_mlp_dim

    self.latent_down = nn.Parameter(torch.zeros(cfg.emb_dim, self.latent_dim))
    self.latent_up = nn.Parameter(torch.zeros(self.latent_dim, cfg.emb_dim))
    self.gate_kernel = nn.Parameter(torch.zeros(cfg.emb_dim, self.num_experts))
    self.wi_0 = nn.Parameter(torch.zeros(self.num_experts, self.latent_dim, self.moe_mlp_dim))
    self.wi_1 = nn.Parameter(torch.zeros(self.num_experts, self.latent_dim, self.moe_mlp_dim))
    self.wo = nn.Parameter(torch.zeros(self.num_experts, self.moe_mlp_dim, self.latent_dim))

  def _emo_mask_logits(self, gate_logits: torch.Tensor, segment_ids: torch.Tensor | None) -> torch.Tensor:
    bsz, seq_len, num_experts = gate_logits.shape
    pool_size = int(self.cfg.emo_eval_document_expert_pool)
    if pool_size >= num_experts:
      return gate_logits
    if segment_ids is None:
      segment_ids = torch.zeros((bsz, seq_len), dtype=torch.int32, device=gate_logits.device)
    scores = F.softmax(gate_logits.float(), dim=-1)
    masked = gate_logits.clone()
    for b in range(bsz):
      seg_b = segment_ids[b]
      # Contiguous document spans
      change = torch.ones(seq_len, dtype=torch.bool, device=gate_logits.device)
      change[1:] = seg_b[1:] != seg_b[:-1]
      starts = torch.nonzero(change, as_tuple=False).squeeze(-1).tolist()
      ends = starts[1:] + [seq_len]
      for s, e in zip(starts, ends):
        doc_score = scores[b, s:e].sum(dim=0)  # [E]
        # Stable top-pool_size experts
        order = torch.argsort(-doc_score, stable=True)
        keep_mask = torch.zeros(num_experts, dtype=torch.bool, device=gate_logits.device)
        keep_mask[order[:pool_size]] = True
        masked[b, s:e] = torch.where(
            keep_mask.unsqueeze(0),
            gate_logits[b, s:e],
            torch.full_like(gate_logits[b, s:e], -float("inf")),
        )
    return masked

  def forward(
      self,
      ffn_in: torch.Tensor,
      segment_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = self.compute_dtype
    ffn_in_dt = ffn_in.to(dt)
    latent_in = _pt_matmul(ffn_in_dt, self.latent_down, dt)  # [B, T, d_latent]

    gate_dtype = torch.float32 if self.cfg.float32_gate_logits else dt
    raw_logits = _pt_matmul(ffn_in, self.gate_kernel, gate_dtype)  # [B, T, E]

    if self.cfg.emo_enabled:
      routed_logits = self._emo_mask_logits(raw_logits, segment_ids)
    else:
      routed_logits = raw_logits

    top_k_vals, top_k_idx = torch.topk(routed_logits, k=self.top_k, dim=-1)
    top_k_weights = F.softmax(top_k_vals.float(), dim=-1).to(dt) * self.top_k

    # Aux losses use the unmasked router logits as in OLMo-core & MaxText
    z_loss = olmoe3._ROUTER_Z_LOSS_WEIGHT * torch.mean(
        torch.logsumexp(raw_logits.float(), dim=-1) ** 2
    )
    softmax_probs = F.softmax(raw_logits.float(), dim=-1).to(dt)
    one_hot = F.one_hot(top_k_idx, num_classes=self.num_experts).float()
    counts = one_hot.sum(dim=(0, 1, 2))
    tokens = raw_logits.shape[0] * raw_logits.shape[1]
    mean_probs = softmax_probs.float().mean(dim=(0, 1))
    lb_loss = (
        (mean_probs * counts).sum()
        * self.num_experts
        / (self.top_k * tokens)
        * self.cfg.load_balance_loss_weight
    )

    # Routed expert execution in latent space
    bsz, seq_len, d_lat = latent_in.shape
    flat_z = latent_in.view(-1, d_lat)             # [N_tok, d_latent]
    flat_idx = top_k_idx.view(-1, self.top_k)      # [N_tok, top_k]
    flat_w = top_k_weights.view(-1, self.top_k)    # [N_tok, top_k]
    routed_out = torch.zeros((flat_z.shape[0], d_lat), dtype=torch.float32, device=flat_z.device)

    wi_0_dt = self.wi_0.to(dt)
    wi_1_dt = self.wi_1.to(dt)
    wo_dt = self.wo.to(dt)
    for e in range(self.num_experts):
      mask_e = flat_idx == e  # [N_tok, top_k]
      if not torch.any(mask_e):
        continue
      tok_indices, k_slots = torch.nonzero(mask_e, as_tuple=True)
      z_e = flat_z[tok_indices]  # [M_e, d_latent]
      w_e = flat_w[tok_indices, k_slots].float().unsqueeze(-1)  # [M_e, 1]
      h0 = F.silu(_pt_matmul(z_e, wi_0_dt[e], dt).float()).to(dt)
      h1 = _pt_matmul(z_e, wi_1_dt[e], dt)
      y_e = _pt_matmul((h0.float() * h1.float()).to(dt), wo_dt[e], dt).float()
      routed_out.index_add_(0, tok_indices, y_e * w_e)

    routed_out = routed_out.to(dt).view(bsz, seq_len, d_lat)
    up_out = _pt_matmul(routed_out, self.latent_up, dt)
    return up_out, lb_loss, z_loss


class PyTorchOLMoE3Block(nn.Module):
  """Single OLMo 3.5 decoder layer in PyTorch."""

  def __init__(self, cfg, layer_idx: int, compute_dtype: torch.dtype):
    super().__init__()
    self.cfg = cfg
    self.layer_idx = layer_idx
    self.compute_dtype = compute_dtype
    self.eps = cfg.normalization_layer_epsilon

    self.attn_in_norm_scale = nn.Parameter(torch.ones(cfg.emb_dim))
    self.attn_out_norm_scale = nn.Parameter(torch.ones(cfg.emb_dim))
    self.ffn_in_norm_scale = nn.Parameter(torch.ones(cfg.emb_dim))
    self.ffn_out_norm_scale = nn.Parameter(torch.ones(cfg.emb_dim))

    self.is_full_attn = (layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    if self.is_full_attn:
      self.mixer = PyTorchFullAttention(cfg, compute_dtype)
    else:
      self.mixer = PyTorchKDA(cfg, compute_dtype)

    self.is_dense = layer_idx < cfg.first_num_dense_layers
    shared_dim = cfg.mlp_dim if self.is_dense else cfg.shared_expert_mlp_dim
    self.shared_ffn = PyTorchSwiGLU(cfg.emb_dim, shared_dim, compute_dtype)
    self.moe = None if self.is_dense else PyTorchLatentMoE(cfg, compute_dtype)

  def forward(
      self,
      x: torch.Tensor,
      positions: torch.Tensor,
      segment_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    dt = self.compute_dtype
    normed = torch_rms_norm(x, self.attn_in_norm_scale, eps=self.eps, compute_dtype=dt)
    mixer_seg_ids = None if self.cfg.olmoe3_segments_for_emo_only else segment_ids

    if self.is_full_attn:
      mixer_out = self.mixer(normed, positions, mixer_seg_ids)
    else:
      mixer_out = self.mixer(normed, mixer_seg_ids)

    hidden = x.to(dt) + torch_rms_norm(mixer_out, self.attn_out_norm_scale, eps=self.eps, compute_dtype=dt)
    ffn_in = torch_rms_norm(hidden, self.ffn_in_norm_scale, eps=self.eps, compute_dtype=dt)
    ffn_out = self.shared_ffn(ffn_in)

    aux_loss = None
    if self.moe is not None:
      routed_up, lb_loss, z_loss = self.moe(ffn_in, segment_ids)
      ffn_out = ffn_out + routed_up
      aux_loss = lb_loss + z_loss

    layer_out = hidden + torch_rms_norm(ffn_out, self.ffn_out_norm_scale, eps=self.eps, compute_dtype=dt)
    return layer_out, aux_loss


class PyTorchOLMoE3Model(nn.Module):
  """Complete OLMo 3.5 decoder model in PyTorch."""

  def __init__(self, cfg, compute_dtype: torch.dtype = torch.float32):
    super().__init__()
    self.cfg = cfg
    self.compute_dtype = compute_dtype
    self.eps = cfg.normalization_layer_epsilon

    self.token_embedder = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.emb_dim))
    self.embedding_norm_scale = nn.Parameter(torch.ones(cfg.emb_dim))
    self.layers = nn.ModuleList(
        [PyTorchOLMoE3Block(cfg, i, compute_dtype) for i in range(cfg.num_decoder_layers)]
    )
    self.decoder_norm_scale = nn.Parameter(torch.ones(cfg.emb_dim))
    self.logits_dense = nn.Parameter(torch.zeros(cfg.emb_dim, cfg.vocab_size))

  def forward(
      self,
      input_tokens: torch.Tensor,
      positions: torch.Tensor,
      segment_ids: torch.Tensor | None = None,
      return_layer_outputs: bool = False,
  ):
    dt = self.compute_dtype
    raw_emb = F.embedding(input_tokens, self.token_embedder.to(dt))
    scaled_emb = raw_emb * (self.cfg.emb_dim ** 0.5)
    h = torch_rms_norm(scaled_emb, self.embedding_norm_scale, eps=self.eps, compute_dtype=dt)

    layer_states = {"embed_norm": h}
    aux_losses = []
    for i, layer in enumerate(self.layers):
      h, aux = layer(h, positions, segment_ids)
      layer_states[f"layers_{i}"] = h
      if aux is not None:
        aux_losses.append(aux)

    final_norm = torch_rms_norm(h, self.decoder_norm_scale, eps=self.eps, compute_dtype=dt)
    layer_states["decoder_norm"] = final_norm
    logits_dt = torch.float32 if self.cfg.logits_dot_in_fp32 else dt
    logits = _pt_matmul(final_norm, self.logits_dense, logits_dt).float()
    if return_layer_outputs:
      return logits, aux_losses, layer_states
    return logits, aux_losses


# -----------------------------------------------------------------------------
# Weight Synchronization: MaxText -> PyTorch (1-to-1 Exact Copy)
# -----------------------------------------------------------------------------


def _copy_array_to_param(param: nn.Parameter, arr) -> int:
  np_arr = np.array(jax.device_get(_unbox_jax(arr)), dtype=np.float32, copy=True)
  assert tuple(param.shape) == np_arr.shape, f"Shape mismatch: {tuple(param.shape)} vs {np_arr.shape}"
  with torch.no_grad():
    param.copy_(torch.from_numpy(np_arr))
  return np_arr.size


def sync_maxtext_to_pytorch(mt_params: dict, pt_model: PyTorchOLMoE3Model) -> int:
  """Copies all parameters from MaxText unrolled `params` dict to `PyTorchOLMoE3Model`."""
  copied = 0
  copied += _copy_array_to_param(pt_model.token_embedder, mt_params["token_embedder"]["embedding"])
  dec = mt_params["decoder"]
  copied += _copy_array_to_param(pt_model.embedding_norm_scale, dec["embedding_norm"]["scale"])
  copied += _copy_array_to_param(pt_model.decoder_norm_scale, dec["decoder_norm"]["scale"])
  copied += _copy_array_to_param(pt_model.logits_dense, dec["logits_dense"]["kernel"])

  for i, pt_layer in enumerate(pt_model.layers):
    ld = dec[f"layers_{i}"]
    copied += _copy_array_to_param(pt_layer.attn_in_norm_scale, ld["attn_in_norm"]["scale"])
    copied += _copy_array_to_param(pt_layer.attn_out_norm_scale, ld["attn_out_norm"]["scale"])
    copied += _copy_array_to_param(pt_layer.ffn_in_norm_scale, ld["ffn_in_norm"]["scale"])
    copied += _copy_array_to_param(pt_layer.ffn_out_norm_scale, ld["ffn_out_norm"]["scale"])

    if pt_layer.is_full_attn:
      attn = ld["mixer"]["attention"]
      copied += _copy_array_to_param(pt_layer.mixer.query_kernel, attn["query"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.key_kernel, attn["key"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.value_kernel, attn["value"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.out_kernel, attn["out"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.query_norm_scale, attn["query_norm"]["scale"])
      copied += _copy_array_to_param(pt_layer.mixer.key_norm_scale, attn["key_norm"]["scale"])
      if pt_layer.mixer.ssmax_scale is not None:
        copied += _copy_array_to_param(pt_layer.mixer.ssmax_scale, attn["ssmax_scale"])
    else:
      kda = ld["mixer"]
      copied += _copy_array_to_param(pt_layer.mixer.w_q, kda["w_q"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.w_k, kda["w_k"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.w_v, kda["w_v"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.q_conv, kda["q_conv"])
      copied += _copy_array_to_param(pt_layer.mixer.k_conv, kda["k_conv"])
      copied += _copy_array_to_param(pt_layer.mixer.v_conv, kda["v_conv"])
      copied += _copy_array_to_param(pt_layer.mixer.f_proj_1, kda["f_proj_1"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.f_proj_2, kda["f_proj_2"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.w_b, kda["w_b"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.A_log, kda["A_log"])
      copied += _copy_array_to_param(pt_layer.mixer.dt_bias, kda["dt_bias"])
      copied += _copy_array_to_param(pt_layer.mixer.g_proj_1, kda["g_proj_1"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.g_proj_2_kernel, kda["g_proj_2"]["kernel"])
      copied += _copy_array_to_param(pt_layer.mixer.g_proj_2_bias, kda["g_proj_2"]["bias"])
      copied += _copy_array_to_param(pt_layer.mixer.o_norm_scale, kda["o_norm"]["scale"])
      copied += _copy_array_to_param(pt_layer.mixer.w_out, kda["w_out"]["kernel"])

    sffn = ld["shared_ffn"]
    copied += _copy_array_to_param(pt_layer.shared_ffn.wi_0, sffn["wi_0"]["kernel"])
    copied += _copy_array_to_param(pt_layer.shared_ffn.wi_1, sffn["wi_1"]["kernel"])
    copied += _copy_array_to_param(pt_layer.shared_ffn.wo, sffn["wo"]["kernel"])

    if not pt_layer.is_dense:
      copied += _copy_array_to_param(pt_layer.moe.latent_down, ld["latent_down"]["kernel"])
      copied += _copy_array_to_param(pt_layer.moe.latent_up, ld["latent_up"]["kernel"])
      mb = ld["moe_block"]
      copied += _copy_array_to_param(pt_layer.moe.gate_kernel, mb["gate"]["kernel"])
      copied += _copy_array_to_param(pt_layer.moe.wi_0, mb["wi_0"])
      copied += _copy_array_to_param(pt_layer.moe.wi_1, mb["wi_1"])
      copied += _copy_array_to_param(pt_layer.moe.wo, mb["wo"])

  mt_total = max_utils.calculate_num_params_from_pytree(mt_params)
  assert copied == mt_total, f"Copied {copied:,} params != MaxText total {mt_total:,}"
  return copied


def _make_config(seq_len: int, dtype: str = "float32", extra=()):
  return pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "model_name=olmoe3-30m",
          "override_model_config=True",
          "run_name=olmoe3_pytorch_parity_test",
          "enable_checkpointing=False",
          "scan_layers=False",
          "per_device_batch_size=1",
          f"max_target_length={seq_len}",
          f"dtype={dtype}",
          "weight_dtype=float32",
          "matmul_precision=highest",
          "attention=dot_product",
          "float32_qk_product=True",
          "float32_logits=True",
          "megablox=False",
          "sparse_matmul=False",
          "use_tokamax_kda=False",
          "kda_pallas_fused=False",
          "kda_flat_head_layout=False",
          "kda_conv_in_compute_dtype=False",
          "gdn_state_dtype=float32",
          "skip_jax_distributed_system=True",
          *extra,
      ]
  )


def _make_packed_inputs(seq_len: int, num_docs: int = 1, vocab_size: int = 100352, seed: int = 42):
  """Builds token IDs, positions, and EOS-delimited segment IDs of length `seq_len`."""
  rng = np.random.default_rng(seed)
  tokens = rng.integers(10, vocab_size - 10, size=(1, seq_len), dtype=np.int32)
  positions = np.broadcast_to(np.arange(seq_len, dtype=np.int32), (1, seq_len)).copy()
  if num_docs <= 1:
    segment_ids = np.ones((1, seq_len), dtype=np.int32)
  else:
    boundaries = np.linspace(0, seq_len, num_docs + 1, dtype=np.int32)
    segment_ids = np.ones((1, seq_len), dtype=np.int32)
    for d in range(num_docs):
      s, e = int(boundaries[d]), int(boundaries[d + 1])
      segment_ids[:, s:e] = d + 1
  return tokens, positions, segment_ids


def _run_maxtext_layerwise(cfg, mesh, model, params, tokens, positions, segment_ids):
  """Runs MaxText model and also captures per-layer hidden states for layerwise comparison."""
  logits, intermediates = model.apply(
      {"params": params},
      tokens,
      positions,
      decoder_segment_ids=segment_ids,
      enable_dropout=False,
      mutable=["intermediates"],
  )
  # Extract per-layer outputs by running the NNX graph or reading intermediates
  return logits, intermediates


class OLMoE3PyTorchLongContextParityTest(unittest.TestCase):
  """Long-context (L=2048, 4096, 8192) MaxText vs. PyTorch OLMo 3.5 verification."""

  def setUp(self):
    super().setUp()
    # Ensure raw jnp.einsum / jnp.matmul calls in KDA and MoE use highest (fp32 / 3-pass bf16) precision on TPU
    jax.config.update("jax_default_matmul_precision", "highest")

  def test_long_context_kda_recurrent_vs_chunked_vs_maxtext_8192(self):
    """Verifies KDA at L=2048, 4096, and 8192: PyTorch Recurrent vs PyTorch Chunked vs MaxText XLA vs MaxText Pallas."""
    print("\n=== [Test 1] Long-Context KDA Mixer Parity (L=2048, 4096, 8192; 1-Doc & 16-Doc) ===")
    is_tpu = jax.devices()[0].platform == "tpu"
    for seq_len in (2048, 4096, 8192):
      cfg_xla = _make_config(
          seq_len,
          dtype="float32",
          extra=["kda_pallas_fused=False", "kda_flat_head_layout=False"],
      )
      cfg_pallas = _make_config(
          seq_len,
          dtype="float32",
          extra=["kda_pallas_fused=True", "kda_flat_head_layout=True"],
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg_xla), cfg_xla.mesh_axes)
      with mesh:
        mt_kda_xla = olmoe3.OLMoE3KimiDeltaAttention(cfg_xla, mesh, rngs=nnx.Rngs(0))
        mt_kda_pallas = olmoe3.OLMoE3KimiDeltaAttention(cfg_pallas, mesh, rngs=nnx.Rngs(0))
        nnx.update(mt_kda_pallas, nnx.state(mt_kda_xla))

        pt_kda = PyTorchKDA(cfg_xla, compute_dtype=torch.float32)
        kda_state = nnx.to_pure_dict(nnx.state(mt_kda_xla, nnx.Param))
        _copy_array_to_param(pt_kda.w_q, kda_state["w_q"]["kernel"])
        _copy_array_to_param(pt_kda.w_k, kda_state["w_k"]["kernel"])
        _copy_array_to_param(pt_kda.w_v, kda_state["w_v"]["kernel"])
        _copy_array_to_param(pt_kda.q_conv, kda_state["q_conv"])
        _copy_array_to_param(pt_kda.k_conv, kda_state["k_conv"])
        _copy_array_to_param(pt_kda.v_conv, kda_state["v_conv"])
        _copy_array_to_param(pt_kda.f_proj_1, kda_state["f_proj_1"]["kernel"])
        _copy_array_to_param(pt_kda.f_proj_2, kda_state["f_proj_2"]["kernel"])
        _copy_array_to_param(pt_kda.w_b, kda_state["w_b"]["kernel"])
        _copy_array_to_param(pt_kda.A_log, kda_state["A_log"])
        _copy_array_to_param(pt_kda.dt_bias, kda_state["dt_bias"])
        _copy_array_to_param(pt_kda.g_proj_1, kda_state["g_proj_1"]["kernel"])
        _copy_array_to_param(pt_kda.g_proj_2_kernel, kda_state["g_proj_2"]["kernel"])
        _copy_array_to_param(pt_kda.g_proj_2_bias, kda_state["g_proj_2"]["bias"])
        _copy_array_to_param(pt_kda.o_norm_scale, kda_state["o_norm"]["scale"])
        _copy_array_to_param(pt_kda.w_out, kda_state["w_out"]["kernel"])

        rng = np.random.default_rng(100 + seq_len)
        x_np = rng.standard_normal((1, seq_len, cfg_xla.emb_dim)).astype(np.float32)
        _, _, seg_16_np = _make_packed_inputs(seq_len, num_docs=16)

        for doc_mode, seg_np in (("1-doc", None), ("16-doc", seg_16_np)):
          x_jax = jnp.asarray(x_np)
          seg_jax = None if seg_np is None else jnp.asarray(seg_np)
          x_pt = torch.from_numpy(x_np)
          seg_pt = None if seg_np is None else torch.from_numpy(seg_np)

          out_mt_xla = mt_kda_xla(x_jax, seg_jax)
          out_mt_pallas = mt_kda_pallas(x_jax, seg_jax)
          with torch.no_grad():
            out_pt_chunk = pt_kda(x_pt, seg_pt, use_recurrent=False)
            out_pt_rec = pt_kda(x_pt, seg_pt, use_recurrent=True)

          m_rec_vs_chunk = _compare_metrics(out_pt_rec, out_pt_chunk)
          m_xla_vs_pt = _compare_metrics(out_mt_xla, out_pt_rec)
          m_pallas_vs_pt = _compare_metrics(out_mt_pallas, out_pt_rec)
          # Also check tail window [seq_len - 512 : seq_len] for long-context drift
          tail_xla_vs_pt = _compare_metrics(out_mt_xla[:, -512:, :], out_pt_rec[:, -512:, :])
          tail_pallas_vs_pt = _compare_metrics(out_mt_pallas[:, -512:, :], out_pt_rec[:, -512:, :])

          print(
              f"  [KDA L={seq_len:4d} {doc_mode:6s}] "
              f"PT_rec vs PT_chunk: max={m_rec_vs_chunk['max_abs']:.2e} | "
              f"MT_XLA vs PT_rec: max={m_xla_vs_pt['max_abs']:.2e} rel_rms={m_xla_vs_pt['rel_rms']:.2e} "
              f"cos={m_xla_vs_pt['cos_sim']:.8f} (tail_max={tail_xla_vs_pt['max_abs']:.2e}) | "
              f"MT_Pallas vs PT_rec: max={m_pallas_vs_pt['max_abs']:.2e} rel_rms={m_pallas_vs_pt['rel_rms']:.2e} "
              f"cos={m_pallas_vs_pt['cos_sim']:.8f} (tail_max={tail_pallas_vs_pt['max_abs']:.2e})"
          )
          self.assertLess(m_rec_vs_chunk["max_abs"], 1e-5)
          self.assertLess(m_xla_vs_pt["max_abs"], 1e-4 if is_tpu else 1e-5)
          self.assertLess(m_pallas_vs_pt["max_abs"], 2e-4 if is_tpu else 1e-5)
          self.assertGreater(m_xla_vs_pt["cos_sim"], 0.999999)
          self.assertGreater(m_pallas_vs_pt["cos_sim"], 0.99999 if is_tpu else 0.999999)
      gc.collect()

  def test_long_context_full_model_logits_and_windows_2k_4k_8k(self):
    """Verifies whole-model forward logits & position-window parity at L=2048, 4096, and 8192 (fp32 & bf16)."""
    print("\n=== [Test 2] Whole-Model 5-Layer OLMo 3.5 Logit Parity (L=2048, 4096, 8192) ===")
    is_tpu = jax.devices()[0].platform == "tpu"
    for seq_len in (2048, 4096, 8192):
      # Use deterministic 32-expert EMo pool so train-mode random pool sampling does not diverge from PyTorch
      cfg_xla = _make_config(
          seq_len,
          dtype="float32",
          extra=[
              "qk_norm_per_head=True",
              "use_scalable_softmax=True",
              "emo_min_document_expert_pool=32",
              "kda_pallas_fused=False",
              "kda_flat_head_layout=False",
              "moe_fused_swiglu_gmm=False",
          ],
      )
      cfg_pallas = _make_config(
          seq_len,
          dtype="float32",
          extra=[
              "qk_norm_per_head=True",
              "use_scalable_softmax=True",
              "emo_min_document_expert_pool=32",
              "kda_pallas_fused=True",
              "kda_flat_head_layout=True",
              "moe_fused_swiglu_gmm=True",
          ],
      )
      mesh, model_xla = olmoe3_build(cfg_xla)
      _, model_pallas = olmoe3_build(cfg_pallas)

      tokens_np, pos_np, seg_np = _make_packed_inputs(seq_len, num_docs=1, seed=200 + seq_len)
      tokens_jax = jnp.asarray(tokens_np)
      pos_jax = jnp.asarray(pos_np)
      seg_jax = jnp.asarray(seg_np)

      rng = jax.random.PRNGKey(0)
      variables = model_xla.init(
          {"params": rng, "dropout": rng},
          tokens_jax,
          pos_jax,
          decoder_segment_ids=seg_jax,
          enable_dropout=False,
      )
      params = variables["params"]
      # Give ssmax_scale non-trivial per-head values to exercise log(pos+1) scaling across 8,192 tokens
      old_ssmax = params["decoder"]["layers_4"]["mixer"]["attention"]["ssmax_scale"]
      new_ssmax_val = jnp.array([0.85], dtype=jnp.float32)
      if hasattr(old_ssmax, "replace"):
        params["decoder"]["layers_4"]["mixer"]["attention"]["ssmax_scale"] = old_ssmax.replace(
            value=new_ssmax_val
        )
      else:
        params["decoder"]["layers_4"]["mixer"]["attention"]["ssmax_scale"] = new_ssmax_val

      pt_model = PyTorchOLMoE3Model(cfg_xla, compute_dtype=torch.float32)
      copied = sync_maxtext_to_pytorch(params, pt_model)
      self.assertEqual(copied, 32_323_588 + 1)  # +1 for ssmax_scale (1 head)

      tokens_pt = torch.from_numpy(tokens_np).long()
      pos_pt = torch.from_numpy(pos_np)
      seg_pt = torch.from_numpy(seg_np)
      with torch.no_grad():
        logits_pt, aux_pt, pt_states = pt_model(
            tokens_pt, pos_pt, seg_pt, return_layer_outputs=True
        )
      del pt_model
      gc.collect()

      logits_mt_xla, inter_xla = model_xla.apply(
          {"params": params},
          tokens_jax,
          pos_jax,
          decoder_segment_ids=seg_jax,
          enable_dropout=False,
          mutable=["intermediates"],
      )
      m_xla = _compare_metrics(logits_mt_xla, logits_pt)
      mt_aux_xla_list = [
          float(inter_xla["intermediates"]["decoder"][f"layers_{i}"]["moe_lb_loss"][0])
          for i in range(1, 5)
      ]
      del logits_mt_xla, inter_xla, model_xla
      gc.collect()

      logits_mt_pallas, inter_pallas = model_pallas.apply(
          {"params": params},
          tokens_jax,
          pos_jax,
          decoder_segment_ids=seg_jax,
          enable_dropout=False,
          mutable=["intermediates"],
      )
      m_pallas = _compare_metrics(logits_mt_pallas, logits_pt)
      del model_pallas

      # Check early [0:512], mid [seq_len//2 - 256 : seq_len//2 + 256], and tail [seq_len - 512 : seq_len]
      w_early = _compare_metrics(logits_mt_pallas[:, :512, :], logits_pt[:, :512, :])
      mid = seq_len // 2
      w_mid = _compare_metrics(logits_mt_pallas[:, mid - 256 : mid + 256, :], logits_pt[:, mid - 256 : mid + 256, :])
      w_tail = _compare_metrics(logits_mt_pallas[:, -512:, :], logits_pt[:, -512:, :])

      # Aux loss comparison (4 MoE layers: layers 1, 2, 3, 4)
      mt_aux_pallas_list = [
          float(inter_pallas["intermediates"]["decoder"][f"layers_{i}"]["moe_lb_loss"][0])
          for i in range(1, 5)
      ]
      del inter_pallas, logits_mt_pallas, logits_pt
      gc.collect()

      pt_aux_list = [float(a.item()) for a in aux_pt]
      max_aux_xla_diff = max(abs(a - b) for a, b in zip(mt_aux_xla_list, pt_aux_list))
      max_aux_pallas_diff = max(abs(a - b) for a, b in zip(mt_aux_pallas_list, pt_aux_list))

      print(
          f"  [Full Model FP32 L={seq_len:4d}] "
          f"MT_XLA vs PT: max={m_xla['max_abs']:.2e} mean={m_xla['mean_abs']:.2e} "
          f"rel_rms={m_xla['rel_rms']:.2e} cos={m_xla['cos_sim']:.8f} top1={m_xla['top1_acc']*100:.2f}% "
          f"aux_diff={max_aux_xla_diff:.2e} | "
          f"MT_Pallas vs PT: max={m_pallas['max_abs']:.2e} mean={m_pallas['mean_abs']:.2e} "
          f"rel_rms={m_pallas['rel_rms']:.2e} cos={m_pallas['cos_sim']:.8f} top1={m_pallas['top1_acc']*100:.2f}% "
          f"aux_diff={max_aux_pallas_diff:.2e}"
      )
      print(
          f"    [Window Drift Check L={seq_len:4d}] "
          f"early[0:512]: max={w_early['max_abs']:.2e} top1={w_early['top1_acc']*100:.2f}% | "
          f"mid[{mid-256}:{mid+256}]: max={w_mid['max_abs']:.2e} top1={w_mid['top1_acc']*100:.2f}% | "
          f"tail[{seq_len-512}:{seq_len}]: max={w_tail['max_abs']:.2e} top1={w_tail['top1_acc']*100:.2f}%"
      )

      self.assertLess(m_xla["max_abs"], 1e-2 if is_tpu else 1e-4)
      self.assertLess(m_xla["mean_abs"], 1e-5)
      self.assertGreater(m_xla["cos_sim"], 0.999999)
      self.assertGreater(m_xla["top1_acc"], 0.999)
      self.assertLess(max_aux_xla_diff, 1e-5 if is_tpu else 1e-6)
      if not is_tpu:
        self.assertLess(m_pallas["max_abs"], 1e-4)
        self.assertGreater(m_pallas["cos_sim"], 0.999999)
        self.assertGreater(m_pallas["top1_acc"], 0.999)
      else:
        self.assertLess(m_pallas["mean_abs"], 1e-5)
        self.assertGreater(m_pallas["cos_sim"], 0.999999)
        self.assertGreater(m_pallas["top1_acc"], 0.999)

      # At L=8192, also verify layer-by-layer hidden states via NNX direct module call
      if seq_len == 8192:
        self._verify_layer_by_layer_8192(cfg_pallas, mesh, params, tokens_jax, pos_jax, seg_jax, pt_states, is_tpu)

      del pt_states, params, variables
      jax.clear_caches()
      gc.collect()

  def _verify_layer_by_layer_8192(self, cfg, mesh, params, tokens_jax, pos_jax, seg_jax, pt_states, is_tpu: bool):
    """Checks hidden-state parity after Embedding+Norm and after each of the 5 decoder layers at L=8192."""
    print("  --- Layer-by-Layer Hidden State Parity at L=8192 (MaxText Pallas vs. PyTorch OLMo 3.5) ---")
    with mesh:
      nnx_model = models.Transformer(
          config=cfg,
          mesh=mesh,
          quant=quantizations.configure_quantization(cfg),
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(0),
      )
      # Load params into nnx_model (stop traversal at nnx.Variable so we don't descend into LogicallyPartitioned)
      state = nnx.state(nnx_model, nnx.Param)
      flat_state = jax.tree_util.tree_flatten_with_path(
          state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      )[0]
      for path, param_var in flat_state:
        keys = [str(getattr(k, "key", k)) for k in path]
        cur = params
        for k in keys:
          cur = cur[k]
        param_var.set_value(_unbox_jax(cur))

      dec = nnx_model.decoder
      y = dec._apply_embedding(
          nnx_model.token_embedder,
          tokens_jax,
          pos_jax,
          True,
          MODEL_MODE_TRAIN,
      )
      m_emb = _compare_metrics(y, pt_states["embed_norm"])
      print(
          f"    Stage embed_norm : max_abs={m_emb['max_abs']:.2e} mean_abs={m_emb['mean_abs']:.2e} "
          f"rel_rms={m_emb['rel_rms']:.2e} cos_sim={m_emb['cos_sim']:.8f}"
      )
      self.assertLess(m_emb["max_abs"], 1e-5)

      for i in range(cfg.num_decoder_layers):
        layer = getattr(dec, f"layers_{i}")
        layer_kind = "FullAttn+MoE" if layer.is_full_attention_layer else ("KDA+Dense" if layer.is_dense_layer else "KDA+MoE")
        y, _ = layer(
            y,
            seg_jax,
            pos_jax,
            True,
            MODEL_MODE_TRAIN,
        )
        m_l = _compare_metrics(y, pt_states[f"layers_{i}"])
        print(
            f"    Stage layers_{i} ({layer_kind:12s}): max_abs={m_l['max_abs']:.2e} "
            f"mean_abs={m_l['mean_abs']:.2e} rel_rms={m_l['rel_rms']:.2e} cos_sim={m_l['cos_sim']:.8f}"
        )
        if not is_tpu:
          self.assertLess(m_l["mean_abs"], 1e-5)
          self.assertGreater(m_l["cos_sim"], 0.999999)
        else:
          self.assertGreater(m_l["cos_sim"], 0.9999)

      y_norm = dec.decoder_norm(y)
      m_dn = _compare_metrics(y_norm, pt_states["decoder_norm"])
      print(
          f"    Stage decoder_norm: max_abs={m_dn['max_abs']:.2e} mean_abs={m_dn['mean_abs']:.2e} "
          f"rel_rms={m_dn['rel_rms']:.2e} cos_sim={m_dn['cos_sim']:.8f}"
      )
      if not is_tpu:
        self.assertLess(m_dn["mean_abs"], 1e-5)
        self.assertGreater(m_dn["cos_sim"], 0.999999)
      else:
        self.assertGreater(m_dn["cos_sim"], 0.9999)

  def test_long_context_multi_doc_emo_and_bfloat16_8192(self):
    """Verifies L=8192 16-document EOS-packed EMo routing (emo_pool=24 < 32, both emo_only=True & False) + bfloat16."""
    print("\n=== [Test 3] L=8192 16-Document Packed EMo Routing & BFloat16 Parity ===")
    is_tpu = jax.devices()[0].platform == "tpu"
    seq_len = 8192
    tokens_np, pos_np, seg_np = _make_packed_inputs(seq_len, num_docs=16, seed=8192)
    tokens_jax = jnp.asarray(tokens_np)
    pos_jax = jnp.asarray(pos_np)
    seg_jax = jnp.asarray(seg_np)
    tokens_pt = torch.from_numpy(tokens_np).long()
    pos_pt = torch.from_numpy(pos_np)
    seg_pt = torch.from_numpy(seg_np)

    for emo_only in (True, False):
      for dtype_str, pt_dt in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        cfg = _make_config(
            seq_len,
            dtype=dtype_str,
            extra=[
                "qk_norm_per_head=True",
                "use_scalable_softmax=True",
                "emo_enabled=True",
                "emo_eval_document_expert_pool=24",  # Active document pool masking (< 32 experts)
                f"olmoe3_segments_for_emo_only={emo_only}",
                "kda_pallas_fused=True",
                "kda_flat_head_layout=True",
                "moe_fused_swiglu_gmm=True",
                "model_call_mode=inference",  # Deterministic eval pool = 24
            ],
        )
        _, model_mt = olmoe3_build(cfg)
        rng = jax.random.PRNGKey(7)
        params = model_mt.init(
            {"params": rng, "dropout": rng},
            tokens_jax,
            pos_jax,
            decoder_segment_ids=seg_jax,
            enable_dropout=False,
        )["params"]

        pt_model = PyTorchOLMoE3Model(cfg, compute_dtype=pt_dt)
        sync_maxtext_to_pytorch(params, pt_model)

        logits_mt = model_mt.apply(
            {"params": params},
            tokens_jax,
            pos_jax,
            decoder_segment_ids=seg_jax,
            enable_dropout=False,
        )
        with torch.no_grad():
          logits_pt, _ = pt_model(tokens_pt, pos_pt, seg_pt)

        m = _compare_metrics(logits_mt, logits_pt)
        tail_m = _compare_metrics(logits_mt[:, -1024:, :], logits_pt[:, -1024:, :])
        print(
            f"  [L=8192 16-Doc EMo(pool=24/32) emo_only={str(emo_only):5s} dtype={dtype_str:8s}] "
            f"max_abs={m['max_abs']:.2e} mean_abs={m['mean_abs']:.2e} rel_rms={m['rel_rms']:.2e} "
            f"cos={m['cos_sim']:.6f} top1={m['top1_acc']*100:.2f}% "
            f"(tail_1k: cos={tail_m['cos_sim']:.6f} top1={tail_m['top1_acc']*100:.2f}%)"
        )
        if dtype_str == "float32" and not is_tpu:
          self.assertLess(m["mean_abs"], 1e-5)
          self.assertLess(m["rel_rms"], 1e-4)
          self.assertGreater(m["cos_sim"], 0.999999)
          self.assertGreater(m["top1_acc"], 0.999)
        elif dtype_str == "float32" and is_tpu:
          self.assertLess(m["mean_abs"], 1e-5)
          self.assertGreater(m["cos_sim"], 0.9999)
          self.assertGreater(m["top1_acc"], 0.98)
        else:
          self.assertGreater(m["cos_sim"], 0.998)
          self.assertGreater(m["top1_acc"], 0.88)
        del logits_mt, logits_pt, pt_model, params, model_mt
        jax.clear_caches()
        gc.collect()

  def test_long_context_loss_and_backward_gradient_parity(self):
    """Verifies forward total/aux loss and backward parameter gradients (dW) at L=2048 and L=4096."""
    print("\n=== [Test 4] Long-Context Loss & Backward Parameter Gradient Parity (PyTorch vs. MaxText) ===")
    is_tpu = jax.devices()[0].platform == "tpu"
    # On CPU Forge containers (16 GB RAM), run L=2048 backward to stay within host RAM; on TPU v4 run L=2048 & 4096.
    seq_lens = (2048, 4096) if is_tpu else (2048,)
    for seq_len in seq_lens:
      cfg = _make_config(
          seq_len,
          dtype="float32",
          extra=[
              "qk_norm_per_head=True",
              "use_scalable_softmax=True",
              "emo_min_document_expert_pool=32",
              "olmoe3_segments_for_emo_only=True",
              "moe_aux_loss_reduction=sum",
              "kda_pallas_fused=True",
              "kda_flat_head_layout=True",
              "moe_fused_swiglu_gmm=True",
          ],
      )
      _, model_mt = olmoe3_build(cfg)
      tokens_np, pos_np, seg_np = _make_packed_inputs(seq_len, num_docs=8, seed=500 + seq_len)
      targets_np = np.roll(tokens_np, -1, axis=1)
      targets_seg_np = seg_np.copy()
      targets_seg_np[:, -1] = 0  # Mask final boundary token as in ShiftToInputsTargets

      tokens_jax = jnp.asarray(tokens_np)
      pos_jax = jnp.asarray(pos_np)
      seg_jax = jnp.asarray(seg_np)
      targets_jax = jnp.asarray(targets_np)
      tseg_jax = jnp.asarray(targets_seg_np)

      rng = jax.random.PRNGKey(11)
      params = model_mt.init(
          {"params": rng, "dropout": rng},
          tokens_jax,
          pos_jax,
          decoder_segment_ids=seg_jax,
          enable_dropout=False,
      )["params"]

      pt_model = PyTorchOLMoE3Model(cfg, compute_dtype=torch.float32)
      sync_maxtext_to_pytorch(params, pt_model)

      # Run PyTorch forward + backward first and free activation graph
      pt_model.zero_grad()
      tokens_pt = torch.from_numpy(tokens_np).long()
      pos_pt = torch.from_numpy(pos_np)
      seg_pt = torch.from_numpy(seg_np)
      targets_pt = torch.from_numpy(targets_np).long()
      mask_pt = torch.from_numpy((targets_seg_np != 0).astype(np.float32))

      logits_pt, aux_pt = pt_model(tokens_pt, pos_pt, seg_pt)
      log_probs_pt = F.log_softmax(logits_pt, dim=-1)
      nll_pt = -torch.gather(log_probs_pt, dim=-1, index=targets_pt.unsqueeze(-1)).squeeze(-1)
      pt_xent = (nll_pt * mask_pt).sum() / mask_pt.sum()
      pt_aux = torch.stack(aux_pt).sum()
      pt_total_loss = pt_xent + pt_aux
      pt_total_loss.backward()
      pt_total_loss_val = float(pt_total_loss.item())
      pt_xent_val = float(pt_xent.item())
      pt_aux_val = float(pt_aux.item())
      del logits_pt, log_probs_pt, nll_pt, pt_xent, pt_aux, pt_total_loss
      gc.collect()

      def mt_loss_fn(p):
        logits, inter = model_mt.apply(
            {"params": p},
            tokens_jax,
            pos_jax,
            decoder_segment_ids=seg_jax,
            enable_dropout=False,
            mutable=["intermediates"],
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        xent = -jnp.take_along_axis(log_probs, targets_jax[..., None], axis=-1).squeeze(-1)
        mask = (tseg_jax != 0).astype(jnp.float32)
        total_xent = jnp.sum(xent * mask) / jnp.sum(mask)
        moe_lbs = [
            inter["intermediates"]["decoder"][f"layers_{i}"]["moe_lb_loss"][0]
            for i in range(1, cfg.num_decoder_layers)
        ]
        aux_loss = jnp.sum(jnp.stack(moe_lbs))
        return total_xent + aux_loss, (total_xent, aux_loss)

      (mt_total_loss, (mt_xent, mt_aux)), mt_grads = jax.value_and_grad(mt_loss_fn, has_aux=True)(params)

      loss_diff = abs(float(mt_total_loss) - pt_total_loss_val)
      xent_diff = abs(float(mt_xent) - pt_xent_val)
      aux_diff = abs(float(mt_aux) - pt_aux_val)

      # Compare gradients across key parameter groups
      grad_pairs = [
          ("token_embedder", mt_grads["token_embedder"]["embedding"], pt_model.token_embedder.grad),
          ("embedding_norm", mt_grads["decoder"]["embedding_norm"]["scale"], pt_model.embedding_norm_scale.grad),
          ("L0_kda_w_q", mt_grads["decoder"]["layers_0"]["mixer"]["w_q"]["kernel"], pt_model.layers[0].mixer.w_q.grad),
          ("L0_kda_A_log", mt_grads["decoder"]["layers_0"]["mixer"]["A_log"], pt_model.layers[0].mixer.A_log.grad),
          ("L0_kda_q_conv", mt_grads["decoder"]["layers_0"]["mixer"]["q_conv"], pt_model.layers[0].mixer.q_conv.grad),
          ("L0_dense_wi_0", mt_grads["decoder"]["layers_0"]["shared_ffn"]["wi_0"]["kernel"], pt_model.layers[0].shared_ffn.wi_0.grad),
          ("L1_moe_gate", mt_grads["decoder"]["layers_1"]["moe_block"]["gate"]["kernel"], pt_model.layers[1].moe.gate_kernel.grad),
          ("L1_moe_wi_0", mt_grads["decoder"]["layers_1"]["moe_block"]["wi_0"], pt_model.layers[1].moe.wi_0.grad),
          ("L1_latent_down", mt_grads["decoder"]["layers_1"]["latent_down"]["kernel"], pt_model.layers[1].moe.latent_down.grad),
          ("L4_attn_query", mt_grads["decoder"]["layers_4"]["mixer"]["attention"]["query"]["kernel"], pt_model.layers[4].mixer.query_kernel.grad),
          ("L4_attn_ssmax", mt_grads["decoder"]["layers_4"]["mixer"]["attention"]["ssmax_scale"], pt_model.layers[4].mixer.ssmax_scale.grad),
          ("decoder_norm", mt_grads["decoder"]["decoder_norm"]["scale"], pt_model.decoder_norm_scale.grad),
          ("logits_dense", mt_grads["decoder"]["logits_dense"]["kernel"], pt_model.logits_dense.grad),
      ]
      print(
          f"  [Loss Parity L={seq_len:4d}] "
          f"total_loss: MT={float(mt_total_loss):.8f} vs PT={pt_total_loss_val:.8f} (|diff|={loss_diff:.2e}) | "
          f"xent |diff|={xent_diff:.2e} | aux |diff|={aux_diff:.2e}"
      )
      self.assertLess(loss_diff, 5e-4 if is_tpu else 1e-5)
      self.assertLess(xent_diff, 5e-4 if is_tpu else 1e-5)
      self.assertLess(aux_diff, 5e-5 if is_tpu else 1e-6)

      for name, g_mt, g_pt in grad_pairs:
        gm = _compare_metrics(g_mt, g_pt)
        print(
            f"    [Grad {name:15s} L={seq_len:4d}] "
            f"max_abs={gm['max_abs']:.2e} rel_rms={gm['rel_rms']:.2e} cos_sim={gm['cos_sim']:.8f}"
        )
        if not is_tpu:
          self.assertLess(gm["max_abs"], 1e-4, f"{name} grad max_abs too large at L={seq_len}")
          self.assertGreater(gm["cos_sim"], 0.99999, f"{name} grad cos_sim too low at L={seq_len}")
        else:
          self.assertGreater(gm["cos_sim"], 0.999, f"{name} grad cos_sim too low on TPU at L={seq_len}")
      del mt_grads, pt_model, params, model_mt
      jax.clear_caches()
      gc.collect()

  def test_long_context_3p5b_dimensions_kda_and_full_attn_8192(self):
    """Verifies exact `olmoe3-3p5b.yml` dimensions (D=1792, dk=256, dv=512, d_lat=768, d_ff=1792) at L=2048, 4096, 8192."""
    if jax.devices()[0].platform != "tpu":
      self.skipTest("Full 3p5b-dimension (D=1792, dk=256, dv=512) 8k test runs on TPU v4 hardware.")
    print(
        "\n=== [Test 5] TPU v4 `olmoe3-3p5b.yml` Exact Dimensions (D=1792, dk=256, dv=512) at L=2048, 4096, 8192 ===",
        flush=True,
    )
    dtype_str, pt_dt = "float32", torch.float32
    common_args = [
        "",
        os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
        "model_name=olmoe3-3p5b",
        "override_model_config=True",
        "run_name=olmoe3_3p5b_pytorch_parity_tpu",
        "enable_checkpointing=False",
        "scan_layers=False",
        "base_num_decoder_layers=2",
        "inhomogeneous_layer_cycle_interval=2",  # Layer 0: Dense+KDA, Layer 1: LatentMoE+FullAttn
        "vocab_size=4096",
        "num_experts=32",
        "num_experts_per_tok=16",
        "emo_min_document_expert_pool=16",
        "emo_max_document_expert_pool=32",
        "emo_eval_document_expert_pool=24",
        "qk_norm_per_head=True",
        "use_scalable_softmax=True",
        "olmoe3_segments_for_emo_only=True",
        "per_device_batch_size=1",
        f"dtype={dtype_str}",
        "weight_dtype=float32",
        "matmul_precision=highest",
        "attention=dot_product",
        "float32_qk_product=True",
        "float32_logits=True",
        "megablox=False",
        "sparse_matmul=False",
        "use_tokamax_kda=False",
        "kda_conv_in_compute_dtype=False",
        "gdn_state_dtype=float32",
        "model_call_mode=inference",
        "skip_jax_distributed_system=True",
    ]
    # Build model_pallas once and initialize parameters on a small 64-token dummy input under jax.jit
    cfg_pallas = pyconfig.initialize(
        [*common_args, "max_target_length=8192", "kda_pallas_fused=True", "kda_flat_head_layout=True", "moe_fused_swiglu_gmm=True"]
    )
    _, model_pallas = olmoe3_build(cfg_pallas)
    rng = jax.random.PRNGKey(23)
    dummy_tok = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_pos = jnp.arange(64, dtype=jnp.int32)[None, :]
    dummy_seg = jnp.ones((1, 64), dtype=jnp.int32)
    params = jax.jit(
        lambda r, t, pos, seg: model_pallas.init(
            {"params": r, "dropout": r}, t, pos, decoder_segment_ids=seg, enable_dropout=False
        )["params"]
    )(rng, dummy_tok, dummy_pos, dummy_seg)
    jax.clear_caches()
    gc.collect()

    pt_model = PyTorchOLMoE3Model(cfg_pallas, compute_dtype=pt_dt)
    sync_maxtext_to_pytorch(params, pt_model)

    run_pallas = jax.jit(
        lambda p, t, pos, seg: model_pallas.apply(
            {"params": p}, t, pos, decoder_segment_ids=seg, enable_dropout=False
        )
    )

    for seq_len in (2048, 4096, 8192):
      tokens_np, pos_np, seg_np = _make_packed_inputs(seq_len, num_docs=16, vocab_size=4096, seed=700 + seq_len)
      tokens_jax = jnp.asarray(tokens_np)
      pos_jax = jnp.asarray(pos_np)
      seg_jax = jnp.asarray(seg_np)

      logits_mt_pallas = np.asarray(run_pallas(params, tokens_jax, pos_jax, seg_jax), dtype=np.float32)
      del tokens_jax, pos_jax, seg_jax
      jax.clear_caches()
      gc.collect()

      with torch.no_grad():
        logits_pt, _ = pt_model(
            torch.from_numpy(tokens_np).long(),
            torch.from_numpy(pos_np),
            torch.from_numpy(seg_np),
        )
      m_pallas = _compare_metrics(logits_mt_pallas, logits_pt)
      tail_pallas = _compare_metrics(logits_mt_pallas[:, -512:, :], logits_pt[:, -512:, :])
      print(
          f"  [3p5b-Dim TPU v4 L={seq_len:4d} dtype={dtype_str:8s}] "
          f"MT_Pallas vs PT: max={m_pallas['max_abs']:.2e} mean={m_pallas['mean_abs']:.2e} "
          f"rel_rms={m_pallas['rel_rms']:.2e} cos={m_pallas['cos_sim']:.8f} top1={m_pallas['top1_acc']*100:.2f}% "
          f"(tail_512: cos={tail_pallas['cos_sim']:.8f} top1={tail_pallas['top1_acc']*100:.2f}%)",
          flush=True,
      )
      self.assertLess(m_pallas["rel_rms"], 2e-3)
      self.assertGreater(m_pallas["cos_sim"], 0.9999)
      self.assertGreater(m_pallas["top1_acc"], 0.98)
      del logits_mt_pallas, logits_pt
      gc.collect()

  def _run_30layer_long_context_logit_alignment(self, regimes: list[tuple[str, str, str]]):
    """Helper for 30-layer end-to-end long-context logit alignment across precision regimes."""
    if jax.devices()[0].platform != "tpu":
      self.skipTest("Full 30-layer long-context TPU vs PyTorch alignment benchmark runs on TPU v4.")
    num_layers = 30
    cycle_interval = 5  # 6 full cycles of (4 KDA + 1 FullAttn), Layer 0 Dense + 29 Routed Latent MoE

    # Initialize 30 distinct layers by compiling two 5-layer cycle inits (Cycle 0 with Dense L0, Cycles 1..5 all-MoE)
    common_extra = [
        f"inhomogeneous_layer_cycle_interval={cycle_interval}",
        "qk_norm_per_head=True",
        "use_scalable_softmax=True",
        "emo_enabled=True",
        "emo_eval_document_expert_pool=32",  # Full causal pool so prefix [0:L] of L=8192 is exact for L=1024,2048,4096,8192
        "olmoe3_segments_for_emo_only=True",
        "model_call_mode=inference",
        "kda_pallas_fused=True",
        "kda_flat_head_layout=True",
        "moe_fused_swiglu_gmm=True",
    ]
    cfg_5l_c0 = _make_config(
        8192, dtype="float32", extra=[f"base_num_decoder_layers={cycle_interval}", "first_num_dense_layers=1", *common_extra]
    )
    cfg_5l_cm = _make_config(
        8192, dtype="float32", extra=[f"base_num_decoder_layers={cycle_interval}", "first_num_dense_layers=0", *common_extra]
    )
    base_cfg = _make_config(
        8192, dtype="float32", extra=[f"base_num_decoder_layers={num_layers}", "first_num_dense_layers=1", *common_extra]
    )
    _, model_5l_c0 = olmoe3_build(cfg_5l_c0)
    _, model_5l_cm = olmoe3_build(cfg_5l_cm)
    dummy_tok = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_pos = jnp.arange(64, dtype=jnp.int32)[None, :]
    dummy_seg = jnp.ones((1, 64), dtype=jnp.int32)
    init_c0 = jax.jit(
        lambda r, t, pos, seg: model_5l_c0.init(
            {"params": r, "dropout": r}, t, pos, decoder_segment_ids=seg, enable_dropout=False
        )["params"]
    )
    init_cm = jax.jit(
        lambda r, t, pos, seg: model_5l_cm.init(
            {"params": r, "dropout": r}, t, pos, decoder_segment_ids=seg, enable_dropout=False
        )["params"]
    )
    p0 = init_c0(jax.random.PRNGKey(3035), dummy_tok, dummy_pos, dummy_seg)
    raw_dec = dict(p0["decoder"])
    for c in range(1, num_layers // cycle_interval):
      pc = init_cm(jax.random.PRNGKey(3035 + c), dummy_tok, dummy_pos, dummy_seg)
      for j in range(cycle_interval):
        raw_dec[f"layers_{c * cycle_interval + j}"] = pc["decoder"][f"layers_{j}"]
    raw_params = {"token_embedder": p0["token_embedder"], "decoder": raw_dec}
    params = _rescale_params_to_trained_regime(raw_params, base_cfg)
    del raw_params, raw_dec, p0, model_5l_c0, model_5l_cm, init_c0, init_cm
    jax.clear_caches()
    gc.collect()

    # Generate a single continuous 8,192-token long text; by strict causality, prefix [0:L] is exact for L=1024, 2048, 4096, 8192
    tokens_np, pos_np, seg_np = _make_long_text_tokens(
        8192, vocab_size=base_cfg.vocab_size, num_docs=1
    )
    tokens_jax = jnp.asarray(tokens_np)
    pos_jax = jnp.asarray(pos_np)
    seg_jax = jnp.asarray(seg_np)
    tokens_pt = torch.from_numpy(tokens_np).long()
    pos_pt = torch.from_numpy(pos_np)
    seg_pt = torch.from_numpy(seg_np)

    pt_model_fp32 = PyTorchOLMoE3Model(base_cfg, compute_dtype=torch.float32)
    sync_maxtext_to_pytorch(params, pt_model_fp32)
    with torch.no_grad():
      logits_pt_fp32, _ = pt_model_fp32(tokens_pt, pos_pt, seg_pt)
    logits_pt_fp32_np = logits_pt_fp32.detach().cpu().float().numpy()
    del pt_model_fp32, logits_pt_fp32
    gc.collect()

    needs_bf16 = any(dt == "bfloat16" for _, dt, _ in regimes)
    logits_pt_bf16_np = None
    if needs_bf16:
      pt_model_bf16 = PyTorchOLMoE3Model(base_cfg, compute_dtype=torch.bfloat16)
      sync_maxtext_to_pytorch(params, pt_model_bf16)
      with torch.no_grad():
        logits_pt_bf16, _ = pt_model_bf16(tokens_pt, pos_pt, seg_pt)
      logits_pt_bf16_np = logits_pt_bf16.detach().cpu().float().numpy()
      del pt_model_bf16, logits_pt_bf16
      gc.collect()

    try:
      for regime_name, mt_dtype, matmul_prec in regimes:
        jax.config.update("jax_default_matmul_precision", matmul_prec)
        cfg_r = _make_config(
            8192,
            dtype=mt_dtype,
            extra=[
                f"base_num_decoder_layers={num_layers}",
                "first_num_dense_layers=1",
                f"matmul_precision={matmul_prec}",
                *common_extra,
            ],
        )
        _, model_r = olmoe3_build(cfg_r)
        logits_mt_full = np.asarray(
            jax.jit(
                lambda p, t, pos, seg: model_r.apply(
                    {"params": p}, t, pos, decoder_segment_ids=seg, enable_dropout=False
                )
            )(params, tokens_jax, pos_jax, seg_jax),
            dtype=np.float32,
        )
        del model_r
        jax.clear_caches()
        gc.collect()

        for seq_len in (1024, 2048, 4096, 8192):
          logits_mt = logits_mt_full[:, :seq_len, :]
          ref_pt = (logits_pt_bf16_np if mt_dtype == "bfloat16" else logits_pt_fp32_np)[:, :seq_len, :]
          m = _compute_logit_alignment_metrics(logits_mt, ref_pt)
          print(
              f"  [30L E2E L={seq_len:4d} {regime_name:12s}] "
              f"top1={m['top1_acc']*100:6.2f}% (tail512={m['tail512_top1_acc']*100:6.2f}% top1_in_top5={m['top1_in_pt_top5']*100:6.2f}%) | "
              f"cos={m['cos_sim']:.6f} | KL(PT||MT)={m['kl_pt_to_mt']:.4e} KL(MT||PT)={m['kl_mt_to_pt']:.4e} | "
              f"1st_logit: mae={m['first_logit_mae']:.3e} max={m['first_logit_max_diff']:.3e} cos={m['first_logit_cos_sim']:.6f} | "
              f"Jaccard @1={m['jaccard@1']:.4f} @5={m['jaccard@5']:.4f} @10={m['jaccard@10']:.4f} "
              f"@20={m['jaccard@20']:.4f} @50={m['jaccard@50']:.4f}",
              flush=True,
          )
          print(
              f"    [Last-Token Next-Pred t={seq_len-1:4d} {regime_name:12s}] "
              f"top1_match={int(m['last_tok_top1_match'])} (MT_id={m['last_tok_mt_top1_id']} vs PT_id={m['last_tok_pt_top1_id']}) | "
              f"1st_logit: MT={m['last_tok_mt_first_logit']:.5f} vs PT={m['last_tok_pt_first_logit']:.5f} "
              f"(|diff|={m['last_tok_first_logit_diff']:.3e}) | "
              f"cos={m['last_tok_cos_sim']:.6f} | KL={m['last_tok_kl_pt_to_mt']:.4e} | "
              f"Jaccard @5={m['last_tok_jaccard@5']:.4f} @10={m['last_tok_jaccard@10']:.4f} "
              f"@20={m['last_tok_jaccard@20']:.4f} @50={m['last_tok_jaccard@50']:.4f}",
              flush=True,
          )
          if mt_dtype == "bfloat16":
            m_vs_fp32 = _compute_logit_alignment_metrics(logits_mt, logits_pt_fp32_np[:, :seq_len, :])
            print(
                f"  [30L E2E L={seq_len:4d} bf16_vs_pt32] "
                f"top1={m_vs_fp32['top1_acc']*100:6.2f}% (tail512={m_vs_fp32['tail512_top1_acc']*100:6.2f}% top1_in_top5={m_vs_fp32['top1_in_pt_top5']*100:6.2f}%) | "
                f"cos={m_vs_fp32['cos_sim']:.6f} | KL(PT||MT)={m_vs_fp32['kl_pt_to_mt']:.4e} KL(MT||PT)={m_vs_fp32['kl_mt_to_pt']:.4e} | "
                f"1st_logit: mae={m_vs_fp32['first_logit_mae']:.3e} max={m_vs_fp32['first_logit_max_diff']:.3e} cos={m_vs_fp32['first_logit_cos_sim']:.6f} | "
                f"Jaccard @1={m_vs_fp32['jaccard@1']:.4f} @5={m_vs_fp32['jaccard@5']:.4f} @10={m_vs_fp32['jaccard@10']:.4f} "
                f"@20={m_vs_fp32['jaccard@20']:.4f} @50={m_vs_fp32['jaccard@50']:.4f}",
                flush=True,
            )
            print(
                f"    [Last-Token Next-Pred t={seq_len-1:4d} bf16_vs_pt32] "
                f"top1_match={int(m_vs_fp32['last_tok_top1_match'])} (MT_id={m_vs_fp32['last_tok_mt_top1_id']} vs PT_id={m_vs_fp32['last_tok_pt_top1_id']}) | "
                f"1st_logit: MT={m_vs_fp32['last_tok_mt_first_logit']:.5f} vs PT={m_vs_fp32['last_tok_pt_first_logit']:.5f} "
                f"(|diff|={m_vs_fp32['last_tok_first_logit_diff']:.3e}) | "
                f"cos={m_vs_fp32['last_tok_cos_sim']:.6f} | KL={m_vs_fp32['last_tok_kl_pt_to_mt']:.4e} | "
                f"Jaccard @5={m_vs_fp32['last_tok_jaccard@5']:.4f} @10={m_vs_fp32['last_tok_jaccard@10']:.4f} "
                f"@20={m_vs_fp32['last_tok_jaccard@20']:.4f} @50={m_vs_fp32['last_tok_jaccard@50']:.4f}",
                flush=True,
            )

          if regime_name == "fp32_highest":
            self.assertGreater(m["cos_sim"], 0.9999)
            self.assertGreater(m["top1_acc"], 0.99)
          elif regime_name == "fp32_default":
            self.assertGreater(m["cos_sim"], 0.99)
            self.assertGreater(m["top1_acc"], 0.80)
          else:
            self.assertGreater(m["cos_sim"], 0.97)
            self.assertGreater(m["top1_acc"], 0.70)
        del logits_mt_full
        gc.collect()
    finally:
      jax.config.update("jax_default_matmul_precision", "highest")

  def test_end_to_end_30layer_long_context_logit_alignment_fp32(self):
    """End-to-end 30-layer OLMo 3.5 long-context logit alignment in fp32 (highest & default MXU)."""
    print(
        "\n=== [Test 6a] End-to-End 30-Layer OLMo 3.5 Long-Context Text Logit Alignment — FP32 (L=1024..8192) ===",
        flush=True,
    )
    self._run_30layer_long_context_logit_alignment([
        ("fp32_highest", "float32", "highest"),
        ("fp32_default", "float32", "default"),
    ])

  def test_end_to_end_30layer_long_context_logit_alignment_bfloat16(self):
    """End-to-end 30-layer OLMo 3.5 long-context logit alignment in bfloat16."""
    print(
        "\n=== [Test 6b] End-to-End 30-Layer OLMo 3.5 Long-Context Text Logit Alignment — BFloat16 (L=1024..8192) ===",
        flush=True,
    )
    self._run_30layer_long_context_logit_alignment([
        ("bfloat16", "bfloat16", "default"),
    ])


def _replace_leaf(leaf, new_arr):
  if hasattr(leaf, "replace") and hasattr(leaf, "value"):
    return leaf.replace(value=new_arr)
  return new_arr


def _scale_leaf(leaf, factor: float):
  arr = _unbox_jax(leaf)
  return _replace_leaf(leaf, arr * jnp.asarray(factor, dtype=arr.dtype))


def _rescale_params_to_trained_regime(params: dict, cfg) -> dict:
  """Rescales `olmoe3_init(stddev=0.02)` parameters to realistic trained-model fan-in variance.

  Why needed for realistic end-to-end MoE sensitivity testing:
    In `OLMoE3DecoderLayer`, `ffn_out = shared_ffn(ffn_in) + routed_up` is normalized by
    `ffn_out_norm(ffn_out)`. Under uniform `_INIT_STD = 0.02`:
      - `shared_ffn` has 2 linear stages (`[wi_0, wi_1] -> wo`), scaling as `O(0.02^3)`.
      - `routed_up` (Latent MoE) has 4 linear stages without an internal norm
        (`latent_down -> [wi_0, wi_1] -> wo -> latent_up`), scaling as `O(0.02^5)`.
    Thus at raw `_INIT_STD = 0.02`, `routed_up` has 28x smaller amplitude (<0.15% variance)
    than `shared_ffn`, so `ffn_out_norm(ffn_out)` is >99.8% Dense `shared_ffn` and mutes MoE!
    Rescaling each matrix to `1 / sqrt(fan_in)` restores balanced `routed_up` vs `shared_ffn`
    contributions, realistic router logit sharpness (`std ~ 1.8`), multi-timescale KDA decay,
    and realistic output logit spread (`std ~ 3.2`).
  """
  init_std = 0.02
  d_model = cfg.emb_dim
  d_lat = cfg.moe_expert_input_dim
  d_ff = cfg.moe_mlp_dim
  top_k = cfg.num_experts_per_tok
  num_layers = cfg.num_decoder_layers
  res_scale = 1.0 / math.sqrt(num_layers)

  dec = dict(params["decoder"])
  dec["logits_dense"] = {
      "kernel": _scale_leaf(dec["logits_dense"]["kernel"], (3.2 / math.sqrt(d_model)) / init_std)
  }

  for i in range(num_layers):
    ld = dict(dec[f"layers_{i}"])
    ld["attn_out_norm"] = {
        "scale": _replace_leaf(
            ld["attn_out_norm"]["scale"],
            jnp.full((d_model,), res_scale, dtype=jnp.float32),
        )
    }
    ld["ffn_out_norm"] = {
        "scale": _replace_leaf(
            ld["ffn_out_norm"]["scale"],
            jnp.full((d_model,), res_scale, dtype=jnp.float32),
        )
    }

    is_full_attn = (i + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    if is_full_attn:
      attn = dict(ld["mixer"]["attention"])
      attn["query"] = {"kernel": _scale_leaf(attn["query"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
      attn["key"] = {"kernel": _scale_leaf(attn["key"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
      attn["value"] = {"kernel": _scale_leaf(attn["value"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
      attn["out"] = {
          "kernel": _scale_leaf(
              attn["out"]["kernel"],
              (1.0 / math.sqrt(cfg.num_query_heads * cfg.head_dim)) / init_std,
          )
      }
      if "ssmax_scale" in attn:
        attn["ssmax_scale"] = _replace_leaf(
            attn["ssmax_scale"],
            jnp.linspace(0.80, 1.20, cfg.num_query_heads, dtype=jnp.float32),
        )
      ld["mixer"] = {"attention": attn}
    else:
      kda = dict(ld["mixer"])
      h = cfg.gdn_num_value_heads
      dk = cfg.gdn_key_head_dim
      dv = cfg.gdn_value_head_dim
      k_conv = cfg.gdn_conv_kernel_dim
      for k_name in ("w_q", "w_k", "w_v", "f_proj_1", "g_proj_1", "w_b"):
        kda[k_name] = {"kernel": _scale_leaf(kda[k_name]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
      for c_name in ("q_conv", "k_conv", "v_conv"):
        kda[c_name] = _scale_leaf(kda[c_name], (1.0 / math.sqrt(k_conv)) / init_std)
      kda["f_proj_2"] = {"kernel": _scale_leaf(kda["f_proj_2"]["kernel"], (1.0 / math.sqrt(dv)) / init_std)}
      kda["g_proj_2"] = {
          "kernel": _scale_leaf(kda["g_proj_2"]["kernel"], (1.0 / math.sqrt(dv)) / init_std),
          "bias": kda["g_proj_2"]["bias"],
      }
      kda["A_log"] = _replace_leaf(
          kda["A_log"],
          jnp.linspace(-3.5, -0.5, h, dtype=jnp.float32),
      )
      kda["dt_bias"] = _replace_leaf(
          kda["dt_bias"],
          jnp.full((h * dk,), -1.0, dtype=jnp.float32),
      )
      kda["w_out"] = {"kernel": _scale_leaf(kda["w_out"]["kernel"], (1.0 / math.sqrt(h * dv)) / init_std)}
      ld["mixer"] = kda

    is_dense = i < cfg.first_num_dense_layers
    s_dim = cfg.mlp_dim if is_dense else cfg.shared_expert_mlp_dim
    sffn = dict(ld["shared_ffn"])
    sffn["wi_0"] = {"kernel": _scale_leaf(sffn["wi_0"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
    sffn["wi_1"] = {"kernel": _scale_leaf(sffn["wi_1"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
    sffn["wo"] = {"kernel": _scale_leaf(sffn["wo"]["kernel"], (1.0 / math.sqrt(s_dim)) / init_std)}
    ld["shared_ffn"] = sffn

    if not is_dense:
      ld["latent_down"] = {
          "kernel": _scale_leaf(ld["latent_down"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)
      }
      ld["latent_up"] = {
          "kernel": _scale_leaf(ld["latent_up"]["kernel"], (1.0 / math.sqrt(d_lat)) / init_std)
      }
      mb = dict(ld["moe_block"])
      mb["gate"] = {"kernel": _scale_leaf(mb["gate"]["kernel"], (1.0 / math.sqrt(d_model)) / init_std)}
      mb["wi_0"] = _scale_leaf(mb["wi_0"], (1.4 / math.sqrt(d_lat)) / init_std)
      mb["wi_1"] = _scale_leaf(mb["wi_1"], (1.4 / math.sqrt(d_lat)) / init_std)
      mb["wo"] = _scale_leaf(mb["wo"], (1.0 / (top_k * math.sqrt(d_ff))) / init_std)
      ld["moe_block"] = mb

    dec[f"layers_{i}"] = ld

  return {"token_embedder": params["token_embedder"], "decoder": dec}


def _make_long_text_tokens(seq_len: int, vocab_size: int = 100352, num_docs: int = 4):
  """Generates deterministic long-context token sequences from a multi-document technical text corpus."""
  paragraphs = [
      (
          "Large language models trained with hybrid sequence mixers combine linear-time recurrent "
          "state updates with periodic full self-attention layers to capture both local syntactic "
          "structure and long-range retrieval across thousands of tokens in a document."
      ),
      (
          "In OLMo 3.5 Small, four out of every five layers use Kimi Delta Attention with chunked "
          "WY matrix factorization and per-channel exponential decay gates, while every fifth layer "
          "applies Grouped Query Attention without positional embeddings, QK RMSNorm, and Scalable "
          "Softmax log-position scaling."
      ),
      (
          "Each Mixture-of-Experts feed-forward block projects the residual stream into a compressed "
          "latent bottleneck before dispatching tokens to the top-k routed SwiGLU experts selected by "
          "an Effort-Balanced MoE document-level expert pool router."
      ),
      (
          "When comparing forward-pass next-token logits between JAX on TPU MXU and PyTorch reference "
          "implementations over 2048, 4096, and 8192 token contexts, minor bfloat16 rounding differences "
          "accumulate across deep layers and occasionally flip borderline top-k expert assignments."
      ),
  ]
  # Map words + subword n-grams deterministically across the full [0, vocab_size) vocabulary
  base_tokens = []
  for doc_idx in range(num_docs):
    for p_idx, para in enumerate(paragraphs):
      words = para.replace(",", " ,").replace(".", " .").replace("-", " - ").split()
      for w_idx, w in enumerate(words):
        # Deterministic hash into [16, vocab_size - 16) with Zipf-like core + full-vocab tail
        h = (hash((w, p_idx, (doc_idx + w_idx) % 7)) & 0x7FFFFFFF)
        if w_idx % 3 == 0:
          tok_id = 16 + (h % min(4096, vocab_size - 32))
        else:
          tok_id = 16 + (h % (vocab_size - 32))
        base_tokens.append(tok_id)
  repeats = (seq_len // len(base_tokens)) + 2
  tiled = (base_tokens * repeats)[:seq_len]
  tokens_np = np.asarray(tiled, dtype=np.int32).reshape(1, seq_len)
  _, pos_np, seg_np = _make_packed_inputs(seq_len, num_docs=num_docs, vocab_size=vocab_size)
  # Reset per-document positions at document boundaries
  if num_docs > 1:
    boundaries = np.linspace(0, seq_len, num_docs + 1, dtype=np.int32)
    for d in range(num_docs):
      s, e = int(boundaries[d]), int(boundaries[d + 1])
      pos_np[0, s:e] = np.arange(e - s, dtype=np.int32)
  return tokens_np, pos_np, seg_np


def olmoe3_build(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = models.transformer_as_linen(cfg, mesh, quant=quantizations.configure_quantization(cfg))
  return mesh, model


if __name__ == "__main__":
  absltest.main()
