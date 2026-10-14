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

from collections.abc import Mapping, Sequence
import importlib
import sys
from typing import Any
from etils import epath
import jax.numpy as jnp
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
    target_mask: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, object]:
  """Computes cosine similarity, max abs error, and KL divergence parity."""
  del kwargs
  mask = target_mask if mask is None else mask
  jax_f64, hf_f64 = np.asarray(jax_logits, dtype=np.float64), np.asarray(hf_logits, dtype=np.float64)
  if jax_f64.shape != hf_f64.shape or jax_f64.ndim < 2:
    raise ValueError(f"Logits shape mismatch: {jax_f64.shape} vs {hf_f64.shape}")
  if mask is not None:
    mask_b = np.asarray(mask, dtype=bool)
    if mask_b.shape != jax_f64.shape[:-1] or not np.any(mask_b):
      raise ValueError(f"Invalid mask {mask_b.shape} for logits {jax_f64.shape}")
    jax_f64, hf_f64 = jax_f64[mask_b], hf_f64[mask_b]

  denom = np.maximum(np.linalg.norm(jax_f64, axis=-1) * np.linalg.norm(hf_f64, axis=-1), 1e-7)
  cos_per_tok = np.sum(jax_f64 * hf_f64, axis=-1) / denom
  log_p_jax, log_p_hf = _log_softmax(jax_f64), _log_softmax(hf_f64)
  kl_div = float(np.mean(np.sum(np.exp(log_p_hf) * (log_p_hf - log_p_jax), axis=-1)))
  max_abs_tol, kl_tol = get_dynamic_threshold(base_max_abs_err, seq_len), get_dynamic_threshold(base_kl_tol, seq_len)
  cos_tol = 1.0 - get_dynamic_threshold(1.0 - min_cosine_sim, seq_len)
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
    tol_nats: float = 1e-4,
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
    seq_len: Any = None,
    mask: Any = None,
    require_aux: bool = True,
) -> dict[str, Any]:
  """Verifies Tier 1A routing, QSA indexer blocks, and Tier 1B logits against golden .npz."""
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
      outputs["logits"], golden["logits"], mask=mask, seq_len=int(seq_len or eval_tokens.shape[1])
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


def main(argv: Sequence[str] | None = None, *, model: Any | None = None) -> int:
  """Entry point dispatching Zero-Gen modes using model_creation_utils.from_pretrained."""
  args = list(sys.argv[1:] if argv is None else argv)
  cli_keys = {"mode", "golden_tensor_path", "seq_len"}
  parsed = {
      a.split("=", 1)[0].removeprefix("--"): a.split("=", 1)[1]
      for a in args
      if "=" in a and a.split("=", 1)[0].removeprefix("--") in cli_keys
  }
  mt_args = [a for a in args if not ("=" in a and a.split("=", 1)[0].removeprefix("--") in cli_keys)]
  mode = parsed.get("mode", "prefill_logits")
  if mode not in {"prefill_logits", "tier1b_logits", "tinymmlu", "holdout_ce"}:
    raise ValueError(f"Unsupported mode: {mode}")
  if not parsed.get("golden_tensor_path"):
    raise ValueError("golden_tensor_path is required for CLI execution.")
  if model is None:
    pyconfig = importlib.import_module("maxtext.src.maxtext.configs.pyconfig")
    maxtext_utils = importlib.import_module("maxtext.src.maxtext.utils.maxtext_utils")
    model_creation_utils = importlib.import_module("maxtext.src.maxtext.utils.model_creation_utils")
    config = pyconfig.initialize([sys.argv[0], *mt_args])
    model = model_creation_utils.from_pretrained(config, mesh=maxtext_utils.create_device_mesh(config))
  golden_path = epath.Path(parsed["golden_tensor_path"])
  seq_len = int(parsed["seq_len"]) if "seq_len" in parsed else None
  if mode in {"prefill_logits", "tier1b_logits"}:
    require_aux = mode != "tier1b_logits"
    res = run_prefill_logits_check(model, golden_path, seq_len=seq_len, require_aux=require_aux)
  elif mode == "tinymmlu":
    res = run_tinymmlu_check(model, golden_path)
  else:
    res = run_holdout_ce_check(model, golden_path)
  return 0 if res.get("passed", False) else 1


if __name__ == "__main__":
  sys.exit(main())
