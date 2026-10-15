# Copyright 2026 Google LLC
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

"""Multi-token masked prefill logits, routing agreement, and batch determinism checker."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any
import numpy as np

# Deterministic golden batch corpus spanning 60% English prose, 20% code/JSON/LaTeX math,
# and 20% multilingual Unicode text per plan Section 4.1.
_PROSE_TEXT = (
    "The James Webb Space Telescope (JWST) operates in a halo orbit around the Sun-Earth L2 "
    "Lagrange point, approximately 1.5 million kilometers outside Earth's orbit. Its primary mirror "
    "consists of 18 hexagonal beryllium segments coated with ultra-thin gold, designed to detect "
    "faint infrared light from the earliest luminous structures that formed in the universe after the "
    "Big Bang. Classical mechanics establishes that every particle in the universe attracts every other "
    "particle with a force proportional to the product of their masses and inversely proportional to the "
    "square of the distance between their centers. In an inertial reference frame, the acceleration of an "
    "object is directly proportional to the net force acting upon it and inversely proportional to its mass, "
    "yielding the dynamical governing equation F = dp/dt. When celestial bodies interact under mutual "
    "gravitational potential, conservation of angular momentum restricts planar motion to conical sections. "
    "Spectroscopic observations across cosmological redshifts confirm that vacuum energy density dominates "
    "the late-time accelerated expansion of the metric tensor, requiring precision instrumentation to isolate "
    "extragalactic photons from local zodiacal dust scattering."
)

_CODE_JSON_MATH_TEXT = (
    "class DistributedAttentionCoordinator:\n"
    "  def __init__(self, num_heads: int, head_dim: int, mesh_shape: tuple[int, ...]) -> None:\n"
    '    """Initializes multi-head distributed communication topology."""\n'
    "    self.num_heads = num_heads\n"
    "    self.head_dim = head_dim\n"
    "    self.mesh_shape = mesh_shape\n\n"
    "  def compute_alltoall_cost(self, sequence_length: int, batch_size: int) -> float:\n"
    "    transfer_bytes = 2.0 * batch_size * sequence_length * self.num_heads * self.head_dim\n"
    "    return transfer_bytes / 1e9\n\n"
    '{\n  "model_config": {\n    "name": "glm5.3-flash",\n    "d_model": 4096,\n    "moe_experts": 288,\n'
    '    "top_k": 8,\n    "sinkhorn_iters": 20,\n    "fp8_block_size": [128, 128],\n    "clamped_swiglu": true\n  }\n}\n'
    r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = "
    r"-\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}; \quad "
    r"i\hbar \frac{\partial \psi}{\partial t} = \left( -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) \right)\psi; \quad "
    r"R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}."
)

_MULTILINGUAL_TEXT = (
    "混合专家模型（Mixture of Experts）在大规模语言模型训练中具有显著的计算效率优势。"
    "通过稀疏路由机制，输入序列中的每个标记仅激活一小部分专家网络，从而在保持总参数规模庞大的同时，"
    "有效控制每个前向传递所需的浮点运算次数。超连接流（Hyper-Connections）通过多流残差传播和双随机"
    "Sinkhorn迭代投影进一步稳定了深层网络的数值梯度传播。道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。"
    "ان علم الفلك والرياضيات من اقدم العلوم التي ارست دعائم الحضارة الانسانية عبر مراصد بغداد ودمشق والقاهرة. "
    "Теория дифференциальных уравнений и функциональный анализ позволяют строго описывать динамические "
    "системы в бесконечномерных гильбертовых пространствах. "
    "量子計算における超伝導キュービットのコヒーレンス時間は量子誤り訂正の閾値を達成する上で極めて重要である。"
    "Die Konvergenzrate iterativer Optimierungsalgorithmen hängt fundamental von der Konditionszahl der Hesse-Matrix ab. "
    "🚀🔭🌌✨⚡📐🧮🪐"
)

GOLDEN_BATCH_TEXT: str = f"{_PROSE_TEXT}\n\n{_CODE_JSON_MATH_TEXT}\n\n{_MULTILINGUAL_TEXT}"

GOLDEN_BATCH_SECTIONS: dict[str, str] = {
    "prose": _PROSE_TEXT,
    "code_json_math": _CODE_JSON_MATH_TEXT,
    "multilingual": _MULTILINGUAL_TEXT,
}


@dataclasses.dataclass(frozen=True)
class MaskedPrefillMetrics:
  """Logit parity and loss metrics for multi-token masked prefill verification."""

  cos_sim: float
  l_inf: float
  max_abs_diff: float
  max_kl_div: float
  logit_std: float
  target_ce_loss: float


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds right-padded token IDs, segment IDs, and target masks from per-row token lists."""
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

    tokens[b_idx, :v_len] = raw_ids[:v_len]
    decoder_segment_ids[b_idx, :v_len] = 1

    t_start = int(v_len * target_start_fraction)
    if t_start >= v_len:
      t_start = max(0, v_len - 1)
    target_mask[b_idx, t_start:v_len] = True

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Tokenizes text into right-padded masked prefill batches using an HF callable tokenizer."""
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
    b_bytes = row_text.encode("utf-8")
    mod = max(1, vocab_size - 1)
    raw_ids = [int(b) % mod + 1 for b in b_bytes]
    raw_ids = [tid if tid != pad_token_id else ((tid % mod) + 1) for tid in raw_ids]
    needed = seq_len * 2
    if needed > len(raw_ids):
      padded_ids = [
          ((raw_ids[i % len(raw_ids)] - 1 + (i // len(raw_ids)) * 31 + (i // 17)) % mod) + 1 for i in range(needed)
      ]
      padded_ids = [tid if tid != pad_token_id else ((tid % mod) + 1) for tid in padded_ids]
      all_token_ids.append(padded_ids)
    else:
      all_token_ids.append(raw_ids[:needed])

  return _build_batch_arrays(
      all_token_ids,
      batch_size=batch_size,
      seq_len=seq_len,
      pad_token_id=pad_token_id,
      target_start_fraction=target_start_fraction,
      valid_lengths=valid_lengths,
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
  valid_norm = (norm_j >= 1e-7) & (norm_r >= 1e-7)
  dots = np.sum(j_masked * r_masked, axis=-1)
  denom = np.where(valid_norm, norm_j * norm_r, 1.0)
  cos_per_token = np.where(valid_norm, dots / denom, 1.0)
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
    if valid_token_mask is not None:
      jd_flat = jd_flat[valid_token_mask]
      rd_flat = rd_flat[valid_token_mask]
    jd_s = np.sort(jd_flat, axis=-1)
    rd_s = np.sort(rd_flat, axis=-1)
    d_match = np.all(jd_s == rd_s, axis=-1)
    total_dsa_matched += int(np.sum(d_match))
    total_dsa_tokens += int(d_match.size)
  dsa_match_rate = float(total_dsa_matched / max(1, total_dsa_tokens))

  j_batched_0 = _to_np_arr(jax_logits)[0]
  s_single_0 = _to_np_arr(single_row_logits)[0]
  if decoder_segment_ids is not None:
    row0_valid = np.asarray(decoder_segment_ids)[0] > 0
    mean_batch_diff = float(np.mean(np.abs(j_batched_0[row0_valid] - s_single_0[row0_valid])))
  else:
    mean_batch_diff = float(np.mean(np.abs(j_batched_0 - s_single_0)))

  tier_1a_passed = bool((expert_match_rate >= expert_match_threshold) and (dsa_match_rate >= dsa_match_threshold))
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
      "expert_index_match_rate": expert_match_rate,
      "expert_mismatch_count": expert_mismatch_count,
      "dsa_index_match_rate": dsa_match_rate,
      "mean_batch_diff": mean_batch_diff,
      "tier_1a_passed": tier_1a_passed,
      "tier_1b_passed": tier_1b_passed,
      "batch_determinism_passed": batch_determinism_passed,
      "all_passed": all_passed,
  }
