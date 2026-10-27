# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zero-Generation verification metrics, golden-tensor I/O, and prefill runner."""

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import importlib
import sys
from typing import Any
from etils import epath
import jax.numpy as jnp
import numpy as np

# Deterministic golden batch corpus spanning English prose, code/JSON/LaTeX math,
# and multilingual Unicode text.
_PROSE_TEXT = (
    "The James Webb Space Telescope (JWST) is an infrared space observatory"
    " launched in 2021 to conduct infrared astronomy. Its high-resolution and"
    " high-sensitivity instruments allow it to view objects too old, distant,"
    " or faint for the Hubble Space Telescope."
)

_CODE_JSON_MATH_TEXT = (
    "class DistributedAttentionCoordinator:\n"
    "  def __init__(self, num_heads: int, head_dim: int, mesh_shape: tuple[int,"
    " ...]) -> None:\n"
    '    """Initializes multi-head distributed communication topology."""\n'
    "    self.num_heads = num_heads\n"
    "    self.head_dim = head_dim\n"
    "    self.mesh_shape = mesh_shape\n\n"
    "  def compute_alltoall_cost(self, sequence_length: int, batch_size: int)"
    " -> float:\n"
    "    transfer_bytes = 2.0 * batch_size * sequence_length * self.num_heads *"
    " self.head_dim\n"
    "    return transfer_bytes / 1e9\n\n"
    '{\n  "model_config": {\n    "name": "glm5.3-flash",\n    "d_model":'
    ' 4096,\n    "moe_experts": 288,\n'
    '    "top_k": 8,\n    "sinkhorn_iters": 20,\n    "fp8_block_size": [128,'
    ' 128],\n    "clamped_swiglu": true\n  }\n}\n'
    r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot"
    r" \nabla)\mathbf{u} = "
    r"-\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}; \quad "
    r"i\hbar \frac{\partial \psi}{\partial t} = \left("
    r" -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) \right)\psi; \quad "
    r"R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi"
    r" G}{c^4} T_{\mu\nu}."
)

_MULTILINGUAL_TEXT = (
    "混合专家模型（Mixture of"
    " Experts）在大规模语言模型训练中具有显著的计算效率优势。通过稀疏路由机制，"
    "输入序列中的每个标记仅激活一小部分专家网络，"
    "从而在保持总参数规模庞大的同时，有效控制每个前向传递所需的浮点运算次数。"
    "超连接流（Hyper-Connections）通过多流残差传播和双随机Sinkhorn迭代投影进一步稳定了深层网络的数值梯度传播。"
    "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。ان علم الفلك"
    " والرياضيات من اقدم العلوم التي ارست دعائم الحضارة الانسانية عبر مراصد"
    " بغداد ودمشق والقاهرة. Теория дифференциальных уравнений и функциональный"
    " анализ позволяют строго описывать динамические системы в бесконечномерных"
    " гильбертовых пространствах."
    " 量子計算における超伝導キュービットのコヒーレンス時間は量子誤り訂正の閾値を達成する上で極めて重要である。"
    "Die Konvergenzrate iterativer Optimierungsalgorithmen hängt fundamental"
    " von der Konditionszahl der Hesse-Matrix ab. 🚀🔭🌌✨⚡📐🧮🪐"
)

GOLDEN_BATCH_TEXT: str = f"{_PROSE_TEXT}\n\n{_CODE_JSON_MATH_TEXT}\n\n{_MULTILINGUAL_TEXT}"

GOLDEN_BATCH_SECTIONS: dict[str, str] = {
    "prose": _PROSE_TEXT,
    "code_json_math": _CODE_JSON_MATH_TEXT,
    "multilingual": _MULTILINGUAL_TEXT,
}

GOLDEN_BATCH: tuple[str, ...] = (GOLDEN_BATCH_TEXT,)
GOLDEN_PROMPTS = GOLDEN_BATCH


@dataclasses.dataclass(frozen=True)
class MaskedPrefillMetrics:
  """Logit parity and loss metrics for multi-token masked prefill verification."""

  cos_sim: float
  l_inf: float
  max_abs_diff: float
  max_kl_div: float
  logit_std: float
  target_ce_loss: float
  zero_norm_token_count: int = 0


def _normalize_texts(text: str | Sequence[str], batch_size: int) -> list[str]:
  """Normalizes a single string or sequence of strings to a batch list of length batch_size."""
  if isinstance(text, str):
    if not text.strip():
      raise ValueError("text must not be empty.")
    return [text] * batch_size
  if not text:
    raise ValueError("text sequence must not be empty.")
  texts = list(text)
  if len(texts) < batch_size:
    texts = (texts * ((batch_size // len(texts)) + 1))[:batch_size]
  return texts


def _build_batch_arrays(
    all_token_ids: list[list[int]],
    *,
    batch_size: int,
    seq_len: int,
    pad_token_id: int,
    target_start_fraction: float,
    valid_lengths: Sequence[int] | None = None,
    padding_side: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds padded token IDs, segment IDs, and target masks from per-row token lists."""
  if padding_side not in ("right", "left"):
    raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side!r}.")

  if valid_lengths is not None:
    if len(valid_lengths) != batch_size:
      raise ValueError(f"len(valid_lengths)={len(valid_lengths)} != batch_size={batch_size}.")
    for idx, v_len in enumerate(valid_lengths):
      if v_len <= 0 or v_len > seq_len:
        raise ValueError(f"valid_lengths[{idx}]={v_len} out of range (0, {seq_len}].")

  tokens = np.full((batch_size, seq_len), pad_token_id, dtype=np.int32)
  decoder_segment_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
  target_mask = np.zeros((batch_size, seq_len), dtype=bool)

  for b_idx in range(batch_size):
    raw_ids = all_token_ids[b_idx]
    if valid_lengths is not None:
      v_len = valid_lengths[b_idx]
      if len(raw_ids) < v_len:
        reps = (v_len + len(raw_ids) - 1) // len(raw_ids)
        raw_ids = (list(raw_ids) * reps)[:v_len]
    else:
      v_len = min(seq_len, len(raw_ids))

    t_start = int(v_len * target_start_fraction)
    if t_start >= v_len:
      t_start = max(0, v_len - 1)

    if padding_side == "right":
      tokens[b_idx, :v_len] = raw_ids[:v_len]
      decoder_segment_ids[b_idx, :v_len] = 1
      target_mask[b_idx, t_start:v_len] = True
    else:
      pad_len = seq_len - v_len
      tokens[b_idx, pad_len:seq_len] = raw_ids[:v_len]
      decoder_segment_ids[b_idx, pad_len:seq_len] = 1
      target_mask[b_idx, pad_len + t_start : seq_len] = True

  if int(np.sum(target_mask)) == 0:
    raise ValueError("Target mask contains zero valid target tokens.")

  return tokens, decoder_segment_ids, target_mask


def tokenize_golden_batch(
    text: str | Sequence[str],
    *,
    tokenizer: Callable[..., Mapping[str, Sequence[int]]],
    seq_len: int,
    pad_token_id: int,
    batch_size: int,
    target_start_fraction: float,
    valid_lengths: Sequence[int] | None = None,
    padding_side: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Tokenizes text into padded masked prefill batches using an HF callable tokenizer."""
  if tokenizer is None:
    raise ValueError("tokenizer must not be None")
  if seq_len <= 0:
    raise ValueError(f"seq_len must be positive, got {seq_len}.")
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}.")
  if pad_token_id < 0:
    raise ValueError(f"pad_token_id must be non-negative, got {pad_token_id}.")
  if not 0.0 <= target_start_fraction < 1.0:
    raise ValueError(f"target_start_fraction must be in [0.0, 1.0), got {target_start_fraction}.")

  texts = _normalize_texts(text, batch_size)
  all_token_ids: list[list[int]] = []
  for b_idx in range(batch_size):
    row_text = texts[b_idx]
    enc = tokenizer(row_text, add_special_tokens=False)
    raw_ids = list(enc["input_ids"])
    if not raw_ids:
      raise ValueError(f"Tokenizer produced empty token list for batch row {b_idx}.")
    all_token_ids.append(raw_ids)

  return _build_batch_arrays(
      all_token_ids,
      batch_size=batch_size,
      seq_len=seq_len,
      pad_token_id=pad_token_id,
      target_start_fraction=target_start_fraction,
      valid_lengths=valid_lengths,
      padding_side=padding_side,
  )


def synthetic_golden_batch(
    text: str | Sequence[str],
    *,
    vocab_size: int,
    seq_len: int,
    pad_token_id: int,
    batch_size: int,
    target_start_fraction: float,
    valid_lengths: Sequence[int] | None = None,
    padding_side: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generates synthetic deterministic token batches for hermetic testing."""
  if vocab_size <= 1:
    raise ValueError(f"vocab_size must be greater than 1, got {vocab_size}.")
  if not 0 <= pad_token_id < vocab_size:
    raise ValueError(f"pad_token_id must be in [0, {vocab_size}), got {pad_token_id}.")
  if seq_len <= 0:
    raise ValueError(f"seq_len must be positive, got {seq_len}.")
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}.")
  if not 0.0 <= target_start_fraction < 1.0:
    raise ValueError(f"target_start_fraction must be in [0.0, 1.0), got {target_start_fraction}.")

  texts = _normalize_texts(text, batch_size)
  all_token_ids: list[list[int]] = []
  for b_idx in range(batch_size):
    row_text = texts[b_idx]
    mod = max(1, vocab_size - 1)
    needed = seq_len * 2
    seed_bytes = hashlib.sha256(f"{b_idx}:{row_text}".encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.default_rng(seed=seed)
    raw_ids = rng.integers(1, mod + 1, size=needed).tolist()
    raw_ids = [tid if tid != pad_token_id else ((tid % mod) + 1) for tid in raw_ids]
    all_token_ids.append(raw_ids)

  return _build_batch_arrays(
      all_token_ids,
      batch_size=batch_size,
      seq_len=seq_len,
      pad_token_id=pad_token_id,
      target_start_fraction=target_start_fraction,
      valid_lengths=valid_lengths,
      padding_side=padding_side,
  )


def compute_masked_prefill_metrics(
    jax_logits: Any,
    ref_logits: Any,
    target_mask: Any,
    *,
    targets: Any | None = None,
) -> MaskedPrefillMetrics:
  """Computes Tier 1B masked-token logit metrics and cross-entropy loss."""

  def _to_f32(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
      return x.detach().cpu().float().numpy()
    if hasattr(x, "shape"):
      return np.array(x, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)

  jl = _to_f32(jax_logits)
  rl = _to_f32(ref_logits)
  mask = np.asarray(target_mask, dtype=bool)

  if jl.shape != rl.shape:
    raise ValueError(f"Logits shape mismatch: jax {jl.shape} != ref {rl.shape}.")
  if jl.shape[:2] != mask.shape:
    raise ValueError(f"Mask shape mismatch: logits {jl.shape[:2]} != mask {mask.shape}.")

  num_valid = int(np.sum(mask))
  if num_valid == 0:
    raise ValueError("target_mask must contain at least one valid target token.")

  j_masked = jl[mask]
  r_masked = rl[mask]

  norm_j = np.linalg.norm(j_masked, axis=-1)
  norm_r = np.linalg.norm(r_masked, axis=-1)
  both_zero = (norm_j < 1e-7) & (norm_r < 1e-7)
  either_zero = (norm_j < 1e-7) | (norm_r < 1e-7)
  zero_norm_token_count = int(np.sum(either_zero))
  valid_norm = ~either_zero

  dots = np.sum(j_masked * r_masked, axis=-1)
  denom = np.where(valid_norm, norm_j * norm_r, 1.0)
  cos_per_token = np.where(both_zero, 1.0, np.where(either_zero, 0.0, dots / denom))
  mean_cos_sim = float(np.mean(cos_per_token))

  max_abs_diff = float(np.max(np.abs(j_masked - r_masked)))

  r_shift = r_masked - np.max(r_masked, axis=-1, keepdims=True)
  log_p = r_shift - np.log(np.sum(np.exp(r_shift), axis=-1, keepdims=True))
  p = np.exp(log_p)

  j_shift = j_masked - np.max(j_masked, axis=-1, keepdims=True)
  log_q = j_shift - np.log(np.sum(np.exp(j_shift), axis=-1, keepdims=True))

  kl_per_token = np.sum(p * (log_p - log_q), axis=-1)
  max_kl_div = float(np.max(np.maximum(kl_per_token, 0.0)))

  logit_std = float(np.std(j_masked))

  if targets is not None:
    t_arr = np.asarray(targets)
    t_masked = t_arr[mask] if t_arr.shape == mask.shape else t_arr
    row_idx = np.arange(len(t_masked))
    target_ce_loss = float(-np.mean(log_q[row_idx, t_masked]))
  else:
    target_ce_loss = float(-np.mean(np.sum(p * log_q, axis=-1)))

  return MaskedPrefillMetrics(
      cos_sim=mean_cos_sim,
      l_inf=max_abs_diff,
      max_abs_diff=max_abs_diff,
      max_kl_div=max_kl_div,
      logit_std=logit_std,
      target_ce_loss=target_ce_loss,
      zero_norm_token_count=zero_norm_token_count,
  )


def evaluate_masked_prefill_parity(
    *,
    jax_logits: Any,
    ref_logits: Any,
    target_mask: Any,
    jax_expert_indices: Sequence[Any],
    ref_expert_indices: Sequence[Any],
    jax_dsa_indices: Sequence[Any],
    ref_dsa_indices: Sequence[Any],
    single_row_logits: Any,
    decoder_segment_ids: Any | None = None,
    dsa_eval_mask: Any | None = None,
    index_topk: int | None = None,
    index_kpool: int = 4,
    targets: Any | None = None,
    cos_sim_threshold: float = 0.999,
    l_inf_threshold: float = 1e-3,
    max_kl_threshold: float = 1e-3,
    logit_std_min_threshold: float = 1e-4,
    batch_diff_threshold: float = 1e-4,
    expert_match_threshold: float = 1.0,
    dsa_match_threshold: float = 1.0,
) -> dict[str, Any]:
  """Performs masked prefill verification across Tier 1A, 1B, determinism."""
  if jax_expert_indices is None or len(jax_expert_indices) == 0:
    raise ValueError("jax_expert_indices must be a non-empty sequence.")
  if ref_expert_indices is None or len(ref_expert_indices) == 0:
    raise ValueError("ref_expert_indices must be a non-empty sequence.")
  if len(jax_expert_indices) != len(ref_expert_indices):
    raise ValueError(f"Expert layers mismatch: jax {len(jax_expert_indices)} vs ref {len(ref_expert_indices)}.")

  if jax_dsa_indices is None or len(jax_dsa_indices) == 0:
    raise ValueError("jax_dsa_indices must be a non-empty sequence.")
  if ref_dsa_indices is None or len(ref_dsa_indices) == 0:
    raise ValueError("ref_dsa_indices must be a non-empty sequence.")
  if len(jax_dsa_indices) != len(ref_dsa_indices):
    raise ValueError(f"DSA layers mismatch: jax {len(jax_dsa_indices)} vs ref {len(ref_dsa_indices)}.")

  if single_row_logits is None:
    raise ValueError("single_row_logits must be provided for batch determinism verification.")

  metrics = compute_masked_prefill_metrics(jax_logits, ref_logits, target_mask, targets=targets)

  def _to_np_arr(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
      return x.detach().cpu().numpy()
    if hasattr(x, "shape"):
      return np.array(x)
    return np.asarray(x)

  valid_token_mask: np.ndarray | None = None
  if decoder_segment_ids is not None:
    valid_token_mask = (np.asarray(decoder_segment_ids) > 0).reshape(-1)

  total_matched = 0
  total_tokens = 0
  for j_idx, r_idx in zip(jax_expert_indices, ref_expert_indices):
    ja = _to_np_arr(j_idx)
    ra = _to_np_arr(r_idx)
    ja_flat = ja.reshape(-1, ja.shape[-1])
    ra_flat = ra.reshape(-1, ra.shape[-1])
    if valid_token_mask is not None:
      ja_flat = ja_flat[valid_token_mask]
      ra_flat = ra_flat[valid_token_mask]
    ja_s = np.sort(ja_flat, axis=-1)
    ra_s = np.sort(ra_flat, axis=-1)
    matched = np.all(ja_s == ra_s, axis=-1)
    total_matched += int(np.sum(matched))
    total_tokens += int(matched.size)
  expert_match_rate = float(total_matched / max(1, total_tokens))
  expert_mismatch_count = int(total_tokens - total_matched)

  total_dsa_matched = 0
  total_dsa_tokens = 0
  for j_dsa, r_dsa in zip(jax_dsa_indices, ref_dsa_indices):
    jd = _to_np_arr(j_dsa)
    rd = _to_np_arr(r_dsa)
    jd_flat = jd.reshape(-1, jd.shape[-1])
    rd_flat = rd.reshape(-1, rd.shape[-1])

    dsa_token_mask: np.ndarray | None = None
    if dsa_eval_mask is not None:
      dsa_token_mask = np.asarray(dsa_eval_mask, dtype=bool).reshape(-1)
      if valid_token_mask is not None:
        dsa_token_mask = dsa_token_mask & valid_token_mask
    elif index_topk is not None:
      topk_pools = index_topk // index_kpool
      if decoder_segment_ids is not None:
        vis_tokens = np.cumsum(np.asarray(decoder_segment_ids) > 0, axis=-1)
      else:
        vis_tokens = np.repeat(np.arange(1, jd.shape[1] + 1)[None, :], jd.shape[0], axis=0)
      vis_pools = vis_tokens // index_kpool
      dsa_token_mask = (vis_pools > topk_pools).reshape(-1)
      if valid_token_mask is not None:
        dsa_token_mask = dsa_token_mask & valid_token_mask
    elif valid_token_mask is not None:
      dsa_token_mask = valid_token_mask

    if dsa_token_mask is not None:
      jd_flat = jd_flat[dsa_token_mask]
      rd_flat = rd_flat[dsa_token_mask]

    if jd_flat.shape[0] > 0:
      jd_s = np.sort(jd_flat, axis=-1)
      rd_s = np.sort(rd_flat, axis=-1)
      d_match = np.all(jd_s == rd_s, axis=-1)
      total_dsa_matched += int(np.sum(d_match))
      total_dsa_tokens += int(d_match.size)

  dsa_match_rate = float(total_dsa_matched / total_dsa_tokens) if total_dsa_tokens > 0 else float("nan")

  j_batched = _to_np_arr(jax_logits)
  batch_size = j_batched.shape[0]
  if isinstance(single_row_logits, Sequence) and not hasattr(single_row_logits, "shape"):
    s_list = [_to_np_arr(x) for x in single_row_logits]
  else:
    s_arr = _to_np_arr(single_row_logits)
    if s_arr.shape[0] == 1 and batch_size > 1:
      s_list = [s_arr[0]]
    else:
      s_list = [s_arr[b] if s_arr.ndim == 3 else s_arr[b : b + 1] for b in range(s_arr.shape[0])]

  row_diffs: list[float] = []
  if len(s_list) != batch_size:
    raise ValueError(f"Single-row logits list length ({len(s_list)}) does not match batch size ({batch_size}).")
  for b_idx in range(batch_size):
    jb_row = j_batched[b_idx]
    s_row = s_list[b_idx]
    if s_row.ndim == 3 and s_row.shape[0] == 1:
      s_row = s_row[0]
    if decoder_segment_ids is not None:
      row_valid = np.asarray(decoder_segment_ids)[b_idx] > 0
      if int(np.sum(row_valid)) > 0:
        jb_valid = jb_row[row_valid]
        if s_row.shape[0] == jb_valid.shape[0]:
          s_valid = s_row
        elif s_row.shape[0] == jb_row.shape[0]:
          s_valid = s_row[row_valid]
        else:
          raise ValueError(
              f"Single-row logits length ({s_row.shape[0]}) does not match"
              f" batched valid length ({jb_valid.shape[0]}) or full sequence"
              f" length ({jb_row.shape[0]})."
          )
        row_diffs.append(float(np.mean(np.abs(jb_valid - s_valid))))
    else:
      row_diffs.append(float(np.mean(np.abs(jb_row - s_row))))

  mean_batch_diff = float(np.mean(row_diffs)) if row_diffs else 0.0

  dsa_ok = (dsa_match_rate >= dsa_match_threshold) if total_dsa_tokens > 0 else True
  tier_1a_passed = bool((expert_match_rate >= expert_match_threshold) and dsa_ok)
  tier_1b_passed = bool(
      metrics.cos_sim > cos_sim_threshold
      and metrics.l_inf < l_inf_threshold
      and metrics.max_kl_div < max_kl_threshold
      and metrics.logit_std > logit_std_min_threshold
  )
  batch_determinism_passed = bool(mean_batch_diff <= batch_diff_threshold)
  all_passed = bool(tier_1a_passed and tier_1b_passed and batch_determinism_passed)

  return {
      "metrics": metrics,
      "cos_sim": metrics.cos_sim,
      "l_inf": metrics.l_inf,
      "max_abs_diff": metrics.max_abs_diff,
      "max_kl_div": metrics.max_kl_div,
      "logit_std": metrics.logit_std,
      "target_ce_loss": metrics.target_ce_loss,
      "zero_norm_token_count": metrics.zero_norm_token_count,
      "expert_index_match_rate": expert_match_rate,
      "expert_mismatch_count": expert_mismatch_count,
      "dsa_index_match_rate": dsa_match_rate,
      "dsa_eval_token_count": int(total_dsa_tokens),
      "mean_batch_diff": mean_batch_diff,
      "tier_1a_passed": tier_1a_passed,
      "tier_1b_passed": tier_1b_passed,
      "batch_determinism_passed": batch_determinism_passed,
      "all_passed": all_passed,
  }


# pylint: disable=too-many-positional-arguments


def _compare_arrays(arr_a: np.ndarray, arr_b: np.ndarray) -> dict[str, object]:
  """Compares two arrays element-wise with explicit shape validation."""
  if arr_a.shape != arr_b.shape:
    out: dict[str, object] = {"passed": False, "match_rate": 0.0, "num_mismatches": -1, "mismatches": []}
    return {**out, "first_mismatch": None, "shape_mismatch": (arr_a.shape, arr_b.shape)}
  equal_mask = arr_a == arr_b
  mismatches = np.argwhere(~equal_mask)
  first = tuple(mismatches[0].tolist()) if len(mismatches) else None
  out = {"passed": bool(np.array_equal(arr_a, arr_b)), "match_rate": float(np.mean(equal_mask))}
  out.update({"num_mismatches": int(len(mismatches)), "mismatches": mismatches.tolist()[:100]})
  return {**out, "first_mismatch": first, "shape_mismatch": None}


def check_routing_parity(jax_routing_indices: np.ndarray, hf_routing_indices: np.ndarray) -> dict[str, object]:
  """Compares top-k MoE expert routing indices (layers, seq, k) for exact match."""
  return _compare_arrays(np.asarray(jax_routing_indices), np.asarray(hf_routing_indices))


def check_indexer_block_parity(jax_selected_blocks: np.ndarray, hf_selected_blocks: np.ndarray) -> dict[str, object]:
  """Compares QSA indexer selected block indices along the block axis (ndim >= 2)."""
  jax_idx, hf_idx = np.asarray(jax_selected_blocks), np.asarray(hf_selected_blocks)
  if jax_idx.ndim < 2:
    raise ValueError(f"Expected indexer block indices with ndim >= 2, got {jax_idx.shape}")
  return _compare_arrays(np.sort(jax_idx, axis=-1), np.sort(hf_idx, axis=-1))


def get_dynamic_threshold(base_threshold: float, seq_len: int, base_seq_len: int = 512) -> float:
  """Scales base threshold dynamically using sqrt(seq_len / base_seq_len)."""
  return float(base_threshold * max(1.0, float(np.sqrt(seq_len / base_seq_len))))


DEFAULT_BASE_SEQ_LEN: int = 512
DEFAULT_BASE_MAX_ABS_ERR: float = 1e-2
DEFAULT_BASE_KL_TOL: float = 1e-4
DEFAULT_MIN_COSINE_SIM: float = 1.0 - 1e-4
DEFAULT_TOL_NATS: float = 1e-4


def _log_softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def check_logits_parity(
    jax_logits: np.ndarray,
    hf_logits: np.ndarray,
    mask: np.ndarray | None = None,
    base_max_abs_err: float = DEFAULT_BASE_MAX_ABS_ERR,
    base_kl_tol: float = DEFAULT_BASE_KL_TOL,
    min_cosine_sim: float = DEFAULT_MIN_COSINE_SIM,
    target_mask: np.ndarray | None = None,
    base_seq_len: int = DEFAULT_BASE_SEQ_LEN,
    **kwargs: Any,
) -> dict[str, object]:
  """Computes cosine similarity, max abs error, and KL divergence parity.

  The criteria are scaled by `get_dynamic_threshold` using the sequence length of the comparison: the size of the
  sequence axis of `jax_logits`, or the last compared position + 1 when a mask is given (padding does not relax
  the criteria).

  Args:
    jax_logits: Logits under test, shaped `[..., seq, vocab]`.
    hf_logits: Reference logits with the same shape.
    mask: Optional boolean `[..., seq]` mask selecting the positions to compare.
    base_max_abs_err: Max |jax - hf| allowed at `base_seq_len` tokens.
    base_kl_tol: Mean KL(hf || jax) allowed at `base_seq_len` tokens.
    min_cosine_sim: Minimum per-token cosine similarity at `base_seq_len` tokens.
    target_mask: Alias of `mask`.
    base_seq_len: Sequence length at which the base criteria apply (see `get_dynamic_threshold`).
    **kwargs: Ignored, so callers can forward a shared criteria mapping.

  Returns:
    Metrics, the effective tolerances (`max_abs_tol`, `kl_tol`, `cos_tol`), and the overall `passed` verdict.
  """
  del kwargs
  mask = target_mask if mask is None else mask
  jax_f64, hf_f64 = np.asarray(jax_logits, dtype=np.float64), np.asarray(hf_logits, dtype=np.float64)
  if jax_f64.shape != hf_f64.shape or jax_f64.ndim < 2:
    raise ValueError(f"Logits shape mismatch: {jax_f64.shape} vs {hf_f64.shape}")
  seq_len = int(jax_f64.shape[-2])
  if mask is not None:
    mask_b = np.asarray(mask, dtype=bool)
    if mask_b.shape != jax_f64.shape[:-1] or not np.any(mask_b):
      raise ValueError(f"Invalid mask {mask_b.shape} for logits {jax_f64.shape}")
    seq_len = int(np.flatnonzero(np.any(mask_b.reshape(-1, seq_len), axis=0))[-1]) + 1
    jax_f64, hf_f64 = jax_f64[mask_b], hf_f64[mask_b]

  denom = np.maximum(np.linalg.norm(jax_f64, axis=-1) * np.linalg.norm(hf_f64, axis=-1), 1e-7)
  cos_per_tok = np.sum(jax_f64 * hf_f64, axis=-1) / denom
  log_p_jax, log_p_hf = _log_softmax(jax_f64), _log_softmax(hf_f64)
  kl_div = float(np.mean(np.sum(np.exp(log_p_hf) * (log_p_hf - log_p_jax), axis=-1)))
  max_abs_tol = get_dynamic_threshold(base_max_abs_err, seq_len, base_seq_len)
  kl_tol = get_dynamic_threshold(base_kl_tol, seq_len, base_seq_len)
  cos_tol = 1.0 - get_dynamic_threshold(1.0 - min_cosine_sim, seq_len, base_seq_len)
  cos_min, cos_mean = float(np.min(cos_per_tok)), float(np.mean(cos_per_tok))
  diff = np.abs(jax_f64 - hf_f64)
  max_err, mean_err = float(np.max(diff)), float(np.mean(diff))
  passed = bool((cos_min >= cos_tol) and (max_err <= max_abs_tol) and (kl_div <= kl_tol))
  out: dict[str, object] = {"passed": passed, "cosine_sim": cos_min, "cosine_sim_min": cos_min}
  out.update({"cosine_sim_mean": cos_mean, "max_abs_err": max_err, "mean_abs_err": mean_err, "kl_div": kl_div})
  return {**out, "max_abs_tol": max_abs_tol, "kl_tol": kl_tol, "cos_tol": cos_tol}


def compute_masked_ce_loss(
    logits: np.ndarray, target_tokens: np.ndarray, target_mask: np.ndarray | None = None, shift: bool = True
) -> float:
  """Computes cross-entropy loss in nats. If shift=True, slices logits[:, :-1] against tokens[:, 1:]."""
  logits_arr, tokens_arr = np.asarray(logits, dtype=np.float64), np.asarray(target_tokens, dtype=np.int64)
  mask_arr = np.ones(tokens_arr.shape, dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)
  if logits_arr.ndim != 3 or tokens_arr.ndim != 2 or mask_arr.ndim != 2:
    raise ValueError(f"Dim mismatch: {logits_arr.shape}, {tokens_arr.shape}, {mask_arr.shape}")
  if logits_arr.shape[:2] != tokens_arr.shape or tokens_arr.shape != mask_arr.shape:
    raise ValueError(f"Shape mismatch: {logits_arr.shape}, {tokens_arr.shape}, {mask_arr.shape}")
  if shift and logits_arr.shape[1] < 2:
    raise ValueError(f"Seq len must be >= 2 for shift=True, got {logits_arr.shape[1]}")

  eval_logits = logits_arr[:, :-1, :] if shift else logits_arr
  eval_tokens = tokens_arr[:, 1:] if shift else tokens_arr
  eval_mask = mask_arr[:, 1:] if shift else mask_arr
  if not np.any(eval_mask):
    raise ValueError("target_mask contains no active tokens (all False).")
  active_logits, active_tokens = eval_logits[eval_mask], eval_tokens[eval_mask]
  if np.any((active_tokens < 0) | (active_tokens >= active_logits.shape[-1])):
    raise ValueError(f"Active tokens out of bounds: vocab_size={active_logits.shape[-1]}")
  log_probs = _log_softmax(active_logits)
  return float(np.mean(-np.take_along_axis(log_probs, active_tokens[:, None], axis=-1).squeeze(-1)))


def check_tinymmlu_loss_delta(
    jax_logits: np.ndarray,
    hf_logits: np.ndarray,
    target_tokens: np.ndarray,
    target_mask: np.ndarray | None = None,
    tol_nats: float = DEFAULT_TOL_NATS,
    shift: bool = True,
    mask: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, object]:
  """Verifies TinyMMLU cross-entropy loss delta between JAX and HF."""
  del kwargs
  target_mask = mask if target_mask is None else target_mask
  loss_jax = compute_masked_ce_loss(jax_logits, target_tokens, target_mask, shift=shift)
  loss_hf = compute_masked_ce_loss(hf_logits, target_tokens, target_mask, shift=shift)
  loss_delta, ppl_jax, ppl_hf = float(abs(loss_jax - loss_hf)), float(np.exp(loss_jax)), float(np.exp(loss_hf))
  out: dict[str, object] = {"passed": bool(loss_delta <= tol_nats), "loss_delta": loss_delta, "ce_jax": loss_jax}
  out.update({"ce_hf": loss_hf, "ppl_jax": ppl_jax, "ppl_hf": ppl_hf})
  return {**out, "ppl_delta": float(abs(ppl_jax - ppl_hf)), "tol_nats": tol_nats}


def check_holdout_ce(
    post_sft_logits: Any, pre_sft_logits: Any, target_tokens: Any, target_mask: Any = None, shift: bool = True
) -> dict[str, object]:
  """Verifies post-SFT holdout CE loss improves over pre-SFT baseline."""
  loss_post = compute_masked_ce_loss(post_sft_logits, target_tokens, target_mask, shift=shift)
  loss_pre = compute_masked_ce_loss(pre_sft_logits, target_tokens, target_mask, shift=shift)
  is_finite, loss_improved = bool(np.isfinite(loss_post) and np.isfinite(loss_pre)), bool(loss_post < loss_pre)
  out: dict[str, object] = {"passed": bool(is_finite and loss_improved), "ce_post_sft": loss_post}
  out.update({"ce_pre_sft": loss_pre, "loss_delta": float(loss_post - loss_pre), "is_finite": is_finite})
  prob_post, prob_pre = float(np.exp(-loss_post)), float(np.exp(-loss_pre))
  return {**out, "prob_trend_upward": loss_improved, "prob_post_sft": prob_post, "prob_pre_sft": prob_pre}


def save_golden_npz(
    path: str | epath.Path,
    tokens: np.ndarray,
    logits: np.ndarray,
    routing_indices: np.ndarray | None = None,
    indexer_selected_blocks: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    mean_var: Mapping[str, np.ndarray] | None = None,
    seq_len: int | None = None,
    dtype_str: str | None = None,
) -> None:
  """Serializes golden tokens, logits, and parity metadata to .npz."""
  p = epath.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, np.ndarray] = {"tokens": np.asarray(tokens), "logits": np.asarray(logits)}
  for key, arr in (
      ("routing_indices", routing_indices),
      ("indexer_selected_blocks", indexer_selected_blocks),
      ("target_mask", target_mask),
  ):
    if arr is not None:
      payload[key] = np.asarray(arr)
  if mean_var:
    for name, arr in mean_var.items():
      payload[f"mean_var__{name}"] = np.asarray(arr)
  if seq_len is not None:
    payload["meta__seq_len"] = np.asarray(seq_len, dtype=np.int64)
  if dtype_str is not None:
    payload["meta__dtype"] = np.asarray(dtype_str)
  with p.open("wb") as f:
    np.savez_compressed(f, **payload)


def load_golden_npz(path: str | epath.Path) -> dict[str, object]:
  """Loads golden tensors from .npz file."""
  p = epath.Path(path)
  if not p.exists():
    raise FileNotFoundError(f"Golden npz file not found at {path}")
  with p.open("rb") as f, np.load(f, allow_pickle=False) as loaded:
    out: dict[str, object] = {k: loaded[k] for k in loaded.files if not k.startswith(("mean_var__", "meta__"))}
    if mean_var := {k.removeprefix("mean_var__"): loaded[k] for k in loaded.files if k.startswith("mean_var__")}:
      out["mean_var"] = mean_var
    if meta := {
        k.removeprefix("meta__"): (int(loaded[k]) if k == "meta__seq_len" else str(loaded[k]))
        for k in loaded.files
        if k.startswith("meta__")
    }:
      out["meta"] = meta
    return out


def extract_model_prefill_outputs(model: Any, tokens: np.ndarray, *, return_aux: bool = True) -> dict[str, Any]:
  """Executes prefill forward pass and extracts logits and optional routing/indexer aux."""
  toks, out = jnp.asarray(tokens, dtype=jnp.int32), {"tokens": np.asarray(tokens, dtype=np.int64)}
  res = model(toks, return_aux=True) if return_aux else model(toks)
  is_pair = return_aux and isinstance(res, tuple) and len(res) == 2
  logits, aux = res if is_pair else (res[0] if isinstance(res, (tuple, list)) else res, None)
  out["logits"] = np.asarray(logits, dtype=np.float32)
  if aux is not None:
    layer_auxes = aux["layer_aux"] if isinstance(aux, Mapping) and "layer_aux" in aux else aux
    for k, out_k in (("router_indices", "routing_indices"), ("selected_block_indices", "indexer_selected_blocks")):
      if isinstance(layer_auxes, Mapping) and k in layer_auxes:
        out[out_k] = np.asarray(layer_auxes[k])
      elif isinstance(layer_auxes, Sequence) and (
          lst := [np.asarray(a[k]) for a in layer_auxes if isinstance(a, Mapping) and k in a]
      ):
        out[out_k] = np.stack(lst, axis=0)
  return out


def run_prefill_logits_check(
    model: Any,
    golden_path: Any,
    jax_tokens: Any = None,
    *,
    mask: Any = None,
    require_aux: bool = True,
    base_max_abs_err: float = DEFAULT_BASE_MAX_ABS_ERR,
    base_kl_tol: float = DEFAULT_BASE_KL_TOL,
    min_cosine_sim: float = DEFAULT_MIN_COSINE_SIM,
    base_seq_len: int = DEFAULT_BASE_SEQ_LEN,
) -> dict[str, Any]:
  """Verifies Tier 1A routing, indexer blocks, and Tier 1B logits (criteria: see DEFAULT_*) against golden .npz."""
  golden = load_golden_npz(golden_path)
  eval_tokens = np.asarray(golden["tokens"], dtype=np.int64)
  if jax_tokens is not None and not np.array_equal(np.asarray(jax_tokens), eval_tokens):
    raise ValueError("Token ID mismatch between jax_tokens and golden tokens.")
  outputs = extract_model_prefill_outputs(model, eval_tokens, return_aux=require_aux)

  def _chk_aux(k: str, fn: Any) -> dict[str, Any]:
    if not (require_aux and k in golden):
      return {"passed": True, "skipped": True}
    return fn(outputs[k], golden[k]) if k in outputs else {"passed": False, "error": f"Golden contains {k}."}

  routing = _chk_aux("routing_indices", check_routing_parity)
  indexer = _chk_aux("indexer_selected_blocks", check_indexer_block_parity)
  logits = check_logits_parity(
      outputs["logits"],
      golden["logits"],
      mask=mask,
      base_max_abs_err=base_max_abs_err,
      base_kl_tol=base_kl_tol,
      min_cosine_sim=min_cosine_sim,
      base_seq_len=base_seq_len,
  )
  passed = bool(routing["passed"] and indexer["passed"] and logits["passed"])
  return {"passed": passed, "routing": routing, "indexer_blocks": indexer, "logits": logits}


def run_tinymmlu_check(
    model: Any, golden_path: Any, jax_tokens: Any = None, *, tol_nats: float = 1e-4, shift: bool = True
) -> dict[str, Any]:
  """Checks TinyMMLU cross-entropy and perplexity parity against golden outputs."""
  golden = load_golden_npz(golden_path)
  eval_tokens = np.asarray(golden["tokens"], dtype=np.int64)
  if jax_tokens is not None and not np.array_equal(np.asarray(jax_tokens), eval_tokens):
    raise ValueError("Token ID mismatch between jax_tokens and golden tokens.")
  jax_logits = extract_model_prefill_outputs(model, eval_tokens, return_aux=False)["logits"]
  return check_tinymmlu_loss_delta(
      jax_logits, golden["logits"], eval_tokens, target_mask=golden.get("target_mask"), tol_nats=tol_nats, shift=shift
  )


def run_holdout_ce_check(
    post_sft_model: Any, pre_sft_source: Any, tokens: Any = None, target_mask: Any = None, shift: bool = True
) -> dict[str, Any]:
  """Verifies post-SFT holdout CE loss improvement over pre-SFT baseline."""
  if isinstance(pre_sft_source, (str, epath.Path)):
    golden = load_golden_npz(pre_sft_source)
    eval_tokens = np.asarray(tokens if tokens is not None else golden["tokens"], dtype=np.int64)
    pre_logits = np.asarray(golden["logits"], dtype=np.float32)
    eval_mask = target_mask if target_mask is not None else golden.get("target_mask")
  else:
    if tokens is None:
      raise ValueError("tokens must be provided when pre_sft_source is a model.")
    eval_tokens = np.asarray(tokens, dtype=np.int64)
    pre_logits = extract_model_prefill_outputs(pre_sft_source, eval_tokens, return_aux=False)["logits"]
    eval_mask = target_mask
  post_logits = extract_model_prefill_outputs(post_sft_model, eval_tokens, return_aux=False)["logits"]
  return check_holdout_ce(post_logits, pre_logits, eval_tokens, target_mask=eval_mask, shift=shift)


def build_arg_parser() -> argparse.ArgumentParser:
  """Returns the parser for the checker's own flags; run with `--help` to list them.

  Unknown arguments are not errors: `main` forwards them verbatim to `pyconfig.initialize`, so MaxText config keys
  (any key in `src/maxtext/configs/base.yml`, e.g. `model_name=... load_parameters_path=... dtype=float32`) are
  passed alongside these flags.
  """
  parser = argparse.ArgumentParser(
      description="Zero-Generation verification of a MaxText model against golden tensors.",
      epilog="Any other argument is forwarded to pyconfig.initialize (MaxText config key=value pairs).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--mode",
      choices=["prefill_logits", "tier1b_logits", "tinymmlu", "holdout_ce"],
      default="prefill_logits",
      help="prefill_logits: routing + indexer + logits; tier1b_logits: logits only; tinymmlu: CE delta; holdout_ce.",
  )
  parser.add_argument("--golden_tensor_path", type=str, required=True, help="Golden .npz written by save_golden_npz.")
  parser.add_argument(
      "--base_max_abs_err",
      type=float,
      default=DEFAULT_BASE_MAX_ABS_ERR,
      help="Max |jax - hf| logit error allowed at --base_seq_len tokens (prefill_logits / tier1b_logits).",
  )
  parser.add_argument(
      "--base_kl_tol",
      type=float,
      default=DEFAULT_BASE_KL_TOL,
      help="Mean KL(hf || jax) allowed at --base_seq_len tokens (prefill_logits / tier1b_logits).",
  )
  parser.add_argument(
      "--min_cosine_sim",
      type=float,
      default=DEFAULT_MIN_COSINE_SIM,
      help="Minimum per-token cosine similarity at --base_seq_len tokens (prefill_logits / tier1b_logits).",
  )
  parser.add_argument(
      "--base_seq_len",
      type=int,
      default=DEFAULT_BASE_SEQ_LEN,
      help="Sequence length the base criteria are calibrated at; they relax by sqrt(seq_len / base_seq_len).",
  )
  parser.add_argument(
      "--tol_nats", type=float, default=DEFAULT_TOL_NATS, help="Max |CE_jax - CE_hf| in nats (tinymmlu mode)."
  )
  return parser


def main(argv: Sequence[str] | None = None, *, model: Any | None = None) -> int:
  """Entry point dispatching Zero-Gen modes using model_creation_utils.from_pretrained.

  Args:
    argv: Arguments (defaults to `sys.argv[1:]`): the flags of `build_arg_parser` plus MaxText config `key=value`
      pairs, which are forwarded to `pyconfig.initialize`.
    model: Model to check; when None, one is created with `model_creation_utils.from_pretrained`.

  Returns:
    0 when the selected check passed, 1 otherwise.
  """
  args, mt_args = build_arg_parser().parse_known_args(sys.argv[1:] if argv is None else list(argv))
  if model is None:
    pyconfig = importlib.import_module("maxtext.src.maxtext.configs.pyconfig")
    model_creation_utils = importlib.import_module("maxtext.src.maxtext.utils.model_creation_utils")
    config = pyconfig.initialize([sys.argv[0], *mt_args])
    model, _ = model_creation_utils.from_pretrained(config)
  golden_path = epath.Path(args.golden_tensor_path)
  if args.mode in {"prefill_logits", "tier1b_logits"}:
    res = run_prefill_logits_check(
        model,
        golden_path,
        require_aux=args.mode != "tier1b_logits",
        base_max_abs_err=args.base_max_abs_err,
        base_kl_tol=args.base_kl_tol,
        min_cosine_sim=args.min_cosine_sim,
        base_seq_len=args.base_seq_len,
    )
  elif args.mode == "tinymmlu":
    res = run_tinymmlu_check(model, golden_path, tol_nats=args.tol_nats)
  else:
    res = run_holdout_ce_check(model, golden_path)
  return 0 if res.get("passed", False) else 1


if __name__ == "__main__":
  sys.exit(main())
