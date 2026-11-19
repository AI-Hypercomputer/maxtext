# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end output checks by output kind, with a negative control.

Output kinds match the conformance spec's `output_kind`:
  logits   - mean KL(golden || candidate) and top-1 agreement;
  tokens   - exact-match ratio of token ids under a fixed seed;
  allclose - continuous outputs (latents, action chunks) via
  layer_parity.compare.
For routed MoE models where sub-roundoff router ties flip expert selection,
use teacher-forced routing (`FrozenTopK`) for the end-to-end `LogitsTol` check
combined with `check_router_agreement` (`RouterAgreementTol`) on per-layer
router scores.
The test runs the candidate twice, once with its parameters and once with one
leaf perturbed by negctl.perturb_one_leaf, asserts `check_outputs` passes on the
first run and `negative_run_status` returns FIRED for the pair.
"""

import dataclasses
from typing import Union

import numpy as np

from tests.unit.m3.testing import layer_parity


@dataclasses.dataclass(frozen=True)
class LogitsTol:
  kl_mean_max: float
  top1_min: float


@dataclasses.dataclass(frozen=True)
class TokensTol:
  match_min: float


@dataclasses.dataclass(frozen=True)
class RouterAgreementTol:
  """Tolerance for MoE router top-k agreement outside near-tie margins."""

  top_k: int
  min_agreement: float
  tie_margin_ulp: float


OutputTol = Union[LogitsTol, TokensTol, layer_parity.Tolerance]


def _is_inexact_or_float(dtype: np.dtype) -> bool:
  """Returns True for floating-point dtypes, including bfloat16 (`np.generic` subtype outside `np.floating`)."""
  return not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_))


def _log_softmax(x: np.ndarray) -> np.ndarray:
  x = x - np.max(x, axis=-1, keepdims=True)
  return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


def check_router_agreement(
    golden_scores: np.ndarray,
    cand_scores: np.ndarray,
    tol: RouterAgreementTol,
) -> tuple[bool, dict[str, float]]:
  """Checks top-k expert set agreement on tokens whose kth vs (k+1)th margin exceeds tie_margin_ulp."""
  if tol.top_k < 1:
    raise ValueError(f"Router top_k must be >= 1, got {tol.top_k}")
  if tol.tie_margin_ulp < 0:
    raise ValueError(f"Router tie_margin_ulp must be >= 0, got {tol.tie_margin_ulp}")
  g_raw = np.asarray(golden_scores)
  c_raw = np.asarray(cand_scores)
  if g_raw.shape != c_raw.shape:
    raise ValueError(f"Router score shape mismatch: {c_raw.shape} vs {g_raw.shape}")
  if g_raw.size == 0:
    raise ValueError(f"Cannot check router agreement on zero-size arrays of shape {g_raw.shape}")
  if g_raw.ndim < 2 or g_raw.shape[-1] <= tol.top_k:
    raise ValueError(f"Router scores {g_raw.shape} must have last dim > top_k={tol.top_k}")
  g = np.asarray(g_raw, dtype=np.float32).reshape(-1, g_raw.shape[-1])
  c = np.asarray(c_raw, dtype=np.float32).reshape(-1, c_raw.shape[-1])
  if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
    return False, {
        "agreement": np.nan,
        "non_tie_fraction": np.nan,
        "max_abs_diff": np.nan,
    }

  g_sorted = np.sort(g, axis=-1)[:, ::-1]
  kth = g_sorted[:, tol.top_k - 1]
  next_k = g_sorted[:, tol.top_k]
  ulp_at_k = np.spacing(np.abs(kth))
  non_tie_mask = (kth - next_k) >= (tol.tie_margin_ulp * ulp_at_k)

  g_topk = np.argsort(g, axis=-1)[:, -tol.top_k :]
  c_topk = np.argsort(c, axis=-1)[:, -tol.top_k :]
  matches = np.array(
      [set(gr.tolist()) == set(cr.tolist()) for gr, cr in zip(g_topk, c_topk)],
      dtype=bool,
  )
  if not np.any(non_tie_mask):
    raise ValueError(
        f"Every token is a top-{tol.top_k} near-tie (non_tie_fraction=0.0 at {tol.tie_margin_ulp} ULP); "
        "router agreement cannot be evaluated"
    )
  agreement = float(np.mean(matches[non_tie_mask]))
  metrics = {
      "agreement": agreement,
      "non_tie_fraction": float(np.mean(non_tie_mask)),
      "max_abs_diff": float(np.max(np.abs(g.astype(np.float64) - c.astype(np.float64)))),
  }
  return agreement >= tol.min_agreement, metrics


def compare_one(golden: np.ndarray, cand: np.ndarray, tol: OutputTol) -> tuple[bool, dict[str, float]]:
  """Compares one output array; the tolerance type selects the output kind."""
  g_raw = np.asarray(golden)
  c_raw = np.asarray(cand)
  if g_raw.shape != c_raw.shape:
    raise ValueError(f"Shape mismatch: candidate {c_raw.shape} vs golden {g_raw.shape}")
  if g_raw.size == 0:
    raise ValueError(f"Cannot compare zero-size arrays of shape {g_raw.shape}")
  if isinstance(tol, LogitsTol):
    if g_raw.ndim < 1:
      raise ValueError(f"logits mode requires at least 1 dimension, got shape {g_raw.shape}")
    if not (_is_inexact_or_float(g_raw.dtype) and _is_inexact_or_float(c_raw.dtype)):
      raise TypeError(f"logits mode needs floating logits, got {g_raw.dtype} and {c_raw.dtype}")
    g = np.asarray(g_raw, dtype=np.float64)
    c = np.asarray(c_raw, dtype=np.float64)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
      return False, {"kl_mean": np.nan, "top1": np.nan}
    lp, lq = _log_softmax(g), _log_softmax(c)
    kl_mean = float(np.mean(np.sum(np.exp(lp) * (lp - lq), axis=-1)))
    top1 = float(np.mean(np.argmax(g, -1) == np.argmax(c, -1)))
    return kl_mean <= tol.kl_mean_max and top1 >= tol.top1_min, {"kl_mean": kl_mean, "top1": top1}
  if isinstance(tol, TokensTol):
    if not (np.issubdtype(g_raw.dtype, np.integer) and np.issubdtype(c_raw.dtype, np.integer)):
      raise TypeError(f"tokens mode needs integer token ids, got {g_raw.dtype} and {c_raw.dtype}")
    ratio = float(np.mean(g_raw == c_raw))
    return ratio >= tol.match_min, {"match": ratio}
  if isinstance(tol, layer_parity.Tolerance):
    ok, m = layer_parity.compare(c_raw, g_raw, tol)
    return ok, {"max_abs": m.max_abs, "frob_rel": m.frob_rel}
  raise TypeError(f"Unsupported tolerance type {type(tol).__name__}")


def check_outputs(
    golden: dict[str, np.ndarray], cand: dict[str, np.ndarray], keys: list[str], tol: OutputTol
) -> tuple[bool, dict[str, dict[str, float]], list[str]]:
  """Compares every key; a key missing on either side raises."""
  if not keys:
    raise ValueError("keys is empty; name every compared output explicitly")
  missing = [k for k in keys if k not in golden or k not in cand]
  if missing:
    raise KeyError(f"Keys missing from golden or candidate outputs: {missing}")
  metrics, failures = {}, []
  for key in keys:
    ok, m = compare_one(golden[key], cand[key], tol)
    metrics[key] = m
    if not ok:
      failures.append(f"{key}: {m}")
  return not failures, metrics, failures


def negative_run_status(
    golden: dict[str, np.ndarray],
    primary: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    keys: list[str],
    tol: OutputTol,
) -> tuple[str, list[str]]:
  """Classifies a negative control as FIRED, MISSED or ERROR, with the reasons.

  ERROR means the control proves nothing: the unperturbed run already fails,
  the perturbed outputs are not finite, or they equal the unperturbed outputs
  (the perturbed leaf is not on the compared path). Only FIRED is a pass.
  """
  primary_ok, _, primary_failures = check_outputs(golden, primary, keys, tol)
  if not primary_ok:
    return "ERROR", [f"unperturbed run fails, so the control is meaningless: {primary_failures}"]
  missing_perturbed = [k for k in keys if k not in perturbed]
  if missing_perturbed:
    raise KeyError(f"Keys missing from perturbed outputs: {missing_perturbed}")
  p_np = {k: np.asarray(perturbed[k]) for k in keys}
  u_np = {k: np.asarray(primary[k]) for k in keys}
  problems = [
      f"perturbed output {k!r} is not finite (a crash-like result is not a fired control)"
      for k in keys
      if _is_inexact_or_float(p_np[k].dtype) and not np.all(np.isfinite(np.asarray(p_np[k], dtype=np.float64)))
  ]
  if all(np.array_equal(p_np[k], u_np[k]) for k in keys):
    problems.append("perturbed outputs equal the unperturbed outputs: the perturbed leaf is not on the compared path")
  if problems:
    return "ERROR", problems
  perturbed_ok, _, _ = check_outputs(golden, perturbed, keys, tol)
  if perturbed_ok:
    return "MISSED", ["perturbed outputs still pass"]
  return "FIRED", []
