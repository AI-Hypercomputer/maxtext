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

"""Zero-Generation verification metrics and golden-tensor I/O library.

Variant of masked_prefill_logits_checker from PR #4447.
"""

from collections.abc import Mapping, Sequence
import sys
from etils import epath
import numpy as np

# Verbatim Golden Batch string from Accelerating MaxText Bring-Up via Zero-Generation Verification.
GOLDEN_BATCH_TEXT: str = """
The James Webb Space Telescope (JWST) is an infrared space observatory launched in 2021 to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope.
To calculate the gravitational force at the Sun-Earth L2 Lagrange point, we use Newton's law of universal gravitation:
F = G * (m1 * m2) / r^2
Here is a Python script to model the baseline orbital parameters:
def calculate_orbital_velocity(mass_kg, radius_m):
    G = 6.67430e-11 # gravitational constant
    velocity = (G * mass_kg / radius_m) ** 0.5
    return {"velocity_m_s": round(velocity, 2), "stable": True}
# Payload Configuration
payload_config = {
    "instruments": ["NIRCam", "NIRSpec", "MIRI", "FGS/NIRISS"],
    "operating_temp_k": 50,
    "sunshield_deployed": true
}
Testing multilingual and unicode tokenizer fallbacks.
Chinese (Simplified): 韦伯太空望远镜 (Webb Space Telescope).
Arabic: تلسكوب جيمس ويب الفضائي
Russian: Космический телескоп Джеймса Уэбба
Emoji byte-pair encoding test: 🚀🔭✨🌌🌍
"""

GOLDEN_BATCH: tuple[str, ...] = (GOLDEN_BATCH_TEXT,)
GOLDEN_PROMPTS = GOLDEN_BATCH


# pylint: disable=too-many-positional-arguments


def _compare_arrays(arr_a: np.ndarray, arr_b: np.ndarray) -> dict[str, object]:
  """Compares two arrays element-wise with explicit shape validation."""
  if arr_a.shape != arr_b.shape:
    return {
        "passed": False,
        "match_rate": 0.0,
        "num_mismatches": -1,
        "mismatches": [],
        "first_mismatch": None,
        "shape_mismatch": (arr_a.shape, arr_b.shape),
    }
  equal_mask = arr_a == arr_b
  mismatches = np.argwhere(~equal_mask)
  return {
      "passed": bool(np.array_equal(arr_a, arr_b)),
      "match_rate": float(np.mean(equal_mask)),
      "num_mismatches": int(len(mismatches)),
      "mismatches": mismatches.tolist()[:100],
      "first_mismatch": (tuple(mismatches[0].tolist()) if len(mismatches) else None),
      "shape_mismatch": None,
  }


def check_routing_parity(jax_routing_indices: np.ndarray, hf_routing_indices: np.ndarray) -> dict[str, object]:
  """Compares top-k MoE expert routing indices (layers, seq, k) for exact match."""
  return _compare_arrays(np.asarray(jax_routing_indices), np.asarray(hf_routing_indices))


def check_indexer_block_parity(jax_selected_blocks: np.ndarray, hf_selected_blocks: np.ndarray) -> dict[str, object]:
  """Compares QSA indexer selected block indices along the block axis.

  Expected tensor layout: (layers, batch, seq_len, num_blocks) with ndim >= 2.
  """
  jax_idx, hf_idx = np.asarray(jax_selected_blocks), np.asarray(hf_selected_blocks)
  if jax_idx.ndim < 2:
    raise ValueError(f"Expected indexer block indices with ndim >= 2, got {jax_idx.shape}")
  return _compare_arrays(np.sort(jax_idx, axis=-1), np.sort(hf_idx, axis=-1))


def get_dynamic_threshold(base_threshold: float, seq_len: int, base_seq_len: int = 512) -> float:
  """Scales base threshold dynamically using sqrt(seq_len / base_seq_len)."""
  scale = max(1.0, float(np.sqrt(seq_len / base_seq_len)))
  return float(base_threshold * scale)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def check_logits_parity(
    jax_logits: np.ndarray,
    hf_logits: np.ndarray,
    mask: np.ndarray | None = None,
    seq_len: int = 512,
    base_max_abs_err: float = 1e-2,
    base_kl_tol: float = 1e-4,
    min_cosine_sim: float = 1.0 - 1e-4,
) -> dict[str, object]:
  """Computes cosine similarity, max abs error, and KL divergence parity.

  Note: "cosine_sim" in the return dict is an alias for "cosine_sim_min".
  """
  jax_f64, hf_f64 = np.asarray(jax_logits, dtype=np.float64), np.asarray(hf_logits, dtype=np.float64)
  if jax_f64.shape != hf_f64.shape or jax_f64.ndim < 2:
    raise ValueError(f"Logits shape mismatch: {jax_f64.shape} vs {hf_f64.shape}")
  if mask is not None:
    mask_b = np.asarray(mask, dtype=bool)
    if mask_b.shape != jax_f64.shape[:-1]:
      raise ValueError(f"Mask {mask_b.shape} != prefix {jax_f64.shape[:-1]}")
    if not np.any(mask_b):
      raise ValueError("mask contains no active tokens (all False).")
    jax_f64, hf_f64 = jax_f64[mask_b], hf_f64[mask_b]

  denom = np.maximum(np.linalg.norm(jax_f64, axis=-1) * np.linalg.norm(hf_f64, axis=-1), 1e-7)
  cos_per_tok = np.sum(jax_f64 * hf_f64, axis=-1) / denom
  log_p_jax, log_p_hf = _log_softmax(jax_f64), _log_softmax(hf_f64)
  kl_div = float(np.mean(np.sum(np.exp(log_p_hf) * (log_p_hf - log_p_jax), axis=-1)))
  max_abs_tol = get_dynamic_threshold(base_max_abs_err, seq_len)
  kl_tol = get_dynamic_threshold(base_kl_tol, seq_len)
  cos_tol = 1.0 - get_dynamic_threshold(1.0 - min_cosine_sim, seq_len)
  cos_min, cos_mean = float(np.min(cos_per_tok)), float(np.mean(cos_per_tok))
  diff = np.abs(jax_f64 - hf_f64)
  max_err, mean_err = float(np.max(diff)), float(np.mean(diff))
  passed = bool((cos_min >= cos_tol) and (max_err <= max_abs_tol) and (kl_div <= kl_tol))
  return {
      "passed": passed,
      "cosine_sim": cos_min,
      "cosine_sim_min": cos_min,
      "cosine_sim_mean": cos_mean,
      "max_abs_err": max_err,
      "mean_abs_err": mean_err,
      "kl_div": kl_div,
      "max_abs_tol": max_abs_tol,
      "kl_tol": kl_tol,
      "cos_tol": cos_tol,
  }


def compute_masked_ce_loss(
    logits: np.ndarray,
    target_tokens: np.ndarray,
    target_mask: np.ndarray | None = None,
    shift: bool = True,
) -> float:
  """Computes cross-entropy loss in nats. If shift=True, slices logits[:, :-1] against tokens[:, 1:]."""
  logits_arr, tokens_arr = np.asarray(logits, dtype=np.float64), np.asarray(target_tokens, dtype=np.int64)
  mask_arr = np.ones(tokens_arr.shape, dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)

  if logits_arr.ndim != 3 or tokens_arr.ndim != 2 or mask_arr.ndim != 2:
    raise ValueError(f"Dim mismatch: {logits_arr.shape}, {tokens_arr.shape}," f" {mask_arr.shape}")
  if logits_arr.shape[:2] != tokens_arr.shape or tokens_arr.shape != mask_arr.shape:
    raise ValueError(f"Shape mismatch: {logits_arr.shape}, {tokens_arr.shape}," f" {mask_arr.shape}")
  if shift and logits_arr.shape[1] < 2:
    raise ValueError(f"Seq len must be >= 2 for shift=True, got {logits_arr.shape[1]}")

  eval_logits = logits_arr[:, :-1, :] if shift else logits_arr
  eval_tokens = tokens_arr[:, 1:] if shift else tokens_arr
  eval_mask = mask_arr[:, 1:] if shift else mask_arr
  if not np.any(eval_mask):
    raise ValueError("target_mask contains no active tokens (all False).")
  active_logits, active_tokens = eval_logits[eval_mask], eval_tokens[eval_mask]
  vocab_size = active_logits.shape[-1]
  if np.any((active_tokens < 0) | (active_tokens >= vocab_size)):
    raise ValueError(f"Active tokens out of bounds: vocab_size={vocab_size}")
  log_probs = _log_softmax(active_logits)
  token_losses = -np.take_along_axis(log_probs, active_tokens[:, None], axis=-1).squeeze(-1)
  return float(np.mean(token_losses))


def check_tinymmlu_loss_delta(
    jax_logits: np.ndarray,
    hf_logits: np.ndarray,
    target_tokens: np.ndarray,
    target_mask: np.ndarray | None = None,
    tol_nats: float = 1e-4,
    shift: bool = True,
) -> dict[str, object]:
  """Verifies cross-entropy loss and perplexity delta between JAX and HF."""
  loss_jax = compute_masked_ce_loss(jax_logits, target_tokens, target_mask, shift=shift)
  loss_hf = compute_masked_ce_loss(hf_logits, target_tokens, target_mask, shift=shift)
  loss_delta = float(abs(loss_jax - loss_hf))
  ppl_jax, ppl_hf = float(np.exp(loss_jax)), float(np.exp(loss_hf))
  return {
      "passed": bool(loss_delta <= tol_nats),
      "loss_delta": loss_delta,
      "ce_jax": loss_jax,
      "ce_hf": loss_hf,
      "ppl_jax": ppl_jax,
      "ppl_hf": ppl_hf,
      "ppl_delta": float(abs(ppl_jax - ppl_hf)),
      "tol_nats": tol_nats,
  }


def check_holdout_ce(
    post_sft_logits: np.ndarray,
    pre_sft_logits: np.ndarray,
    target_tokens: np.ndarray,
    target_mask: np.ndarray | None = None,
    shift: bool = True,
) -> dict[str, object]:
  """Verifies post-SFT holdout CE loss improves over pre-SFT baseline."""
  loss_post = compute_masked_ce_loss(post_sft_logits, target_tokens, target_mask, shift=shift)
  loss_pre = compute_masked_ce_loss(pre_sft_logits, target_tokens, target_mask, shift=shift)
  is_finite = bool(np.isfinite(loss_post) and np.isfinite(loss_pre))
  loss_improved = bool(loss_post < loss_pre)
  return {
      "passed": bool(is_finite and loss_improved),
      "ce_post_sft": loss_post,
      "ce_pre_sft": loss_pre,
      "loss_delta": float(loss_post - loss_pre),
      "is_finite": is_finite,
      "prob_trend_upward": loss_improved,
      "prob_post_sft": float(np.exp(-loss_post)),
      "prob_pre_sft": float(np.exp(-loss_pre)),
  }


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
  payload: dict[str, np.ndarray] = {
      "tokens": np.asarray(tokens),
      "logits": np.asarray(logits),
  }
  opt = (
      ("routing_indices", routing_indices),
      ("indexer_selected_blocks", indexer_selected_blocks),
      ("target_mask", target_mask),
  )
  for key, arr in opt:
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
    mean_var = {k.removeprefix("mean_var__"): loaded[k] for k in loaded.files if k.startswith("mean_var__")}
    meta = {
        k.removeprefix("meta__"): (int(loaded[k]) if k == "meta__seq_len" else str(loaded[k]))
        for k in loaded.files
        if k.startswith("meta__")
    }
    if mean_var:
      out["mean_var"] = mean_var
    if meta:
      out["meta"] = meta
    return out


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point dispatching Zero-Gen modes."""
  mode = "prefill_logits"
  for arg in argv if argv is not None else sys.argv[1:]:
    if arg.startswith(("mode=", "--mode=")):
      mode = arg.split("=", 1)[1]
  if mode not in {
      "prefill_logits",
      "tinymmlu",
      "holdout_ce",
      "generate_goldens",
  }:
    raise ValueError(f"Unsupported mode: {mode}")
  raise NotImplementedError(f"mode {mode!r}: end-to-end model driver lands in CL 8b")


if __name__ == "__main__":
  sys.exit(main())
