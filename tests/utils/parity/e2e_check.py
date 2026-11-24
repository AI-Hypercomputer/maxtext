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

"""Model-agnostic end-to-end output checks by output kind, with a negative control.

Output kinds:
  logits   - mean (and optional max) KL(golden || candidate) and top-1 agreement;
  tokens   - exact-match ratio of token ids under a fixed seed;
  allclose - continuous outputs via `layer_parity.compare`.
For routed MoE models where sub-roundoff router ties flip expert selection,
use teacher-forced routing (`FrozenTopK`) for end-to-end `LogitsTol` checks
combined with `check_router_agreement` (`RouterAgreementTol`) on per-layer
router scores.
"""

import dataclasses
from typing import Union

import numpy as np

from tests.utils.parity import layer_parity
from tests.utils.parity import negctl


@dataclasses.dataclass(frozen=True)
class LogitsTol:
  """Tolerance for end-to-end logit distributions."""

  kl_mean_max: float
  top1_min: float
  kl_max: float | None = None

  def __post_init__(self) -> None:
    if not (np.isfinite(self.kl_mean_max) and np.isfinite(self.top1_min)):
      raise ValueError(f"LogitsTol thresholds must be finite, got {self}")
    if self.kl_mean_max < 0.0:
      raise ValueError(f"LogitsTol KL thresholds must be >= 0, got {self}")
    if self.kl_max is not None and (not np.isfinite(self.kl_max) or self.kl_max < 0.0):
      raise ValueError(f"LogitsTol kl_max must be finite and >= 0, got {self}")
    if not 0.0 <= self.top1_min <= 1.0:
      raise ValueError(f"LogitsTol top1_min must be in [0, 1], got {self.top1_min}")


@dataclasses.dataclass(frozen=True)
class TokensTol:
  """Tolerance for discrete token-id outputs."""

  match_min: float

  def __post_init__(self) -> None:
    if not np.isfinite(self.match_min) or not 0.0 <= self.match_min <= 1.0:
      raise ValueError(f"TokensTol match_min must be finite and in [0, 1], got {self.match_min}")


@dataclasses.dataclass(frozen=True)
class RouterAgreementTol:
  """Tolerance for MoE router top-k agreement outside near-tie margins."""

  top_k: int
  min_agreement: float
  tie_margin_ulp: float


OutputTol = Union[LogitsTol, TokensTol, layer_parity.Tolerance]


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
  g_arr = np.asarray(golden)
  c_arr = np.asarray(cand)
  if g_arr.shape != c_arr.shape:
    raise ValueError(f"Shape mismatch: candidate {c_arr.shape} vs golden {g_arr.shape}")
  if g_arr.size == 0:
    raise ValueError(f"Cannot compare zero-size arrays of shape {g_arr.shape}")
  if isinstance(tol, LogitsTol):
    if g_arr.ndim < 1 or g_arr.shape[-1] < 2:
      raise ValueError(
          f"logits mode requires at least 1 dimension with vocab axis shape[-1] >= 2, got shape {g_arr.shape}"
      )
    if not (negctl.is_floating_dtype(g_arr.dtype) and negctl.is_floating_dtype(c_arr.dtype)):
      raise TypeError(f"logits mode needs floating logits, got {g_arr.dtype} and {c_arr.dtype}")
    g = np.asarray(g_arr, dtype=np.float64)
    c = np.asarray(c_arr, dtype=np.float64)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
      return False, {"kl_mean": np.nan, "kl_max": np.nan, "top1": np.nan}
    lp, lq = _log_softmax(g), _log_softmax(c)
    kl_per_pos = np.sum(np.exp(lp) * (lp - lq), axis=-1)
    kl_mean = float(np.mean(kl_per_pos))
    kl_max = float(np.max(kl_per_pos))
    top1 = float(np.mean(np.argmax(g, -1) == np.argmax(c, -1)))
    passed = kl_mean <= tol.kl_mean_max and top1 >= tol.top1_min
    if tol.kl_max is not None:
      passed = passed and kl_max <= tol.kl_max
    return passed, {"kl_mean": kl_mean, "kl_max": kl_max, "top1": top1}
  if isinstance(tol, TokensTol):
    if not (np.issubdtype(g_arr.dtype, np.integer) and np.issubdtype(c_arr.dtype, np.integer)):
      raise TypeError(f"tokens mode needs integer token ids, got {g_arr.dtype} and {c_arr.dtype}")
    ratio = float(np.mean(g_arr == c_arr))
    return ratio >= tol.match_min, {"match": ratio}
  if isinstance(tol, layer_parity.Tolerance):
    ok, m = layer_parity.compare(c_arr, g_arr, tol)
    return ok, {"max_abs": m.max_abs, "frob_rel": m.frob_rel}
  raise TypeError(f"Unsupported tolerance type {type(tol).__name__}")


def check_outputs(
    golden: dict[str, np.ndarray],
    cand: dict[str, np.ndarray],
    keys: list[str],
    tol: OutputTol,
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


def _is_array_finite(arr: np.ndarray) -> bool:
  """Returns True if `arr` is finite across all entries (including bfloat16)."""
  raw = np.asarray(arr)
  if negctl.is_floating_dtype(raw.dtype):
    return bool(np.all(np.isfinite(np.asarray(raw, dtype=np.float64))))
  return True


def negative_run_status(
    golden: dict[str, np.ndarray],
    primary: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    keys: list[str],
    tol: OutputTol,
) -> tuple[str, list[str]]:
  """Classifies a negative control as FIRED, MISSED or ERROR, with reasons."""
  if not keys:
    raise ValueError("keys is empty; name every compared output explicitly")
  missing = [k for k in keys if k not in golden or k not in primary or k not in perturbed]
  if missing:
    raise KeyError(f"Keys missing from golden, primary, or perturbed outputs: {missing}")
  primary_ok, _, primary_failures = check_outputs(golden, primary, keys, tol)
  if not primary_ok:
    return "ERROR", [f"unperturbed run fails, so the control is meaningless: {primary_failures}"]
  problems = [
      f"perturbed output {k!r} is not finite (a crash-like result is not a fired control)"
      for k in keys
      if not _is_array_finite(perturbed[k])
  ]
  if all(np.array_equal(np.asarray(perturbed[k]), np.asarray(primary[k])) for k in keys):
    problems.append("perturbed outputs equal the unperturbed outputs: the perturbed leaf is not on the compared path")
  if problems:
    return "ERROR", problems
  perturbed_ok, perturbed_metrics, _ = check_outputs(golden, perturbed, keys, tol)
  non_finite_metrics = [
      f"perturbed metrics for {k!r} are not finite: {m}"
      for k, m in perturbed_metrics.items()
      if not all(np.isfinite(v) for v in m.values())
  ]
  if non_finite_metrics:
    return "ERROR", non_finite_metrics
  if perturbed_ok:
    return "MISSED", ["perturbed outputs still pass"]
  return "FIRED", []
