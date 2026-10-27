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

"""End-to-end output checks by output kind, with a negative control.

Output kinds:
  logits   - mean/max KL(golden || candidate) and top-1 agreement;
  tokens   - exact-match ratio of token ids under a fixed seed (note: `tokens`
             checks whose in-band leaf perturbation cannot flip discrete token
             ids run the negative control on the underlying `logits` and
             document that choice);
  allclose - continuous outputs via `layer_parity.compare`.
The test runs the candidate twice, once with its parameters and once with one
leaf perturbed by `negctl.perturb_one_leaf`, asserts `check_outputs` passes on
the first run and `negative_run_status` returns `"FIRED"` for the pair.
"""

from collections.abc import Mapping
import dataclasses
import types
from typing import Union

import jax.numpy as jnp
import numpy as np

from tests.utils.parity import layer_parity

# Approved CPU full-model logits ceilings from tolerance_table.md row T13 / D9.
LOGITS_KL_CEILING = 1e-6
LOGITS_TOP1_FLOOR = 1.0


@dataclasses.dataclass(frozen=True)
class LogitsTol:
  """Tolerance for end-to-end logit distributions."""

  kl_mean_max: float
  kl_max: float
  top1_min: float

  def __post_init__(self) -> None:
    if not (np.isfinite(self.kl_mean_max) and np.isfinite(self.kl_max) and np.isfinite(self.top1_min)):
      raise ValueError(f"LogitsTol thresholds must be finite, got {self}")
    if self.kl_mean_max < 0.0 or self.kl_max < 0.0:
      raise ValueError(f"LogitsTol KL thresholds must be >= 0, got {self}")
    if not 0.0 <= self.top1_min <= 1.0:
      raise ValueError(f"LogitsTol top1_min must be in [0, 1], got {self.top1_min}")
    if self.kl_mean_max > LOGITS_KL_CEILING or self.kl_max > LOGITS_KL_CEILING or self.top1_min < LOGITS_TOP1_FLOOR:
      raise ValueError(
          "LogitsTol exceeds approved CPU logits ceilings "
          f"(kl_mean_max <= {LOGITS_KL_CEILING}, kl_max <= {LOGITS_KL_CEILING}, "
          f"top1_min >= {LOGITS_TOP1_FLOOR}): {self}"
      )


CPU_LOGITS_TOL = LogitsTol(
    kl_mean_max=LOGITS_KL_CEILING,
    kl_max=LOGITS_KL_CEILING,
    top1_min=LOGITS_TOP1_FLOOR,
)

# Named-row registry for CPU logits rows from tolerance_table.md. TPU rows are
# registered by the TPU e2e unit; callers select rows by ID.
_APPROVED_LOGITS_ROWS: Mapping[str, LogitsTol] = types.MappingProxyType(
    {
        "T13": CPU_LOGITS_TOL,
    }
)


def logits_tol_for_row(row_id: str) -> LogitsTol:
  """Returns the registered `LogitsTol` for `tolerance_table.md` row `row_id`."""
  if row_id in _APPROVED_LOGITS_ROWS:
    return _APPROVED_LOGITS_ROWS[row_id]
  registered = ", ".join(sorted(_APPROVED_LOGITS_ROWS))
  raise ValueError(f"Unregistered logits tolerance row {row_id!r}; registered rows: [{registered}]")


@dataclasses.dataclass(frozen=True)
class TokensTol:
  """Tolerance for discrete token-id outputs.

  When an in-band parameter perturbation (`negctl.REL_NORM_BAND`) cannot flip
  discrete argmax/sampled token ids, run the negative control on the underlying
  logits with `LogitsTol` instead.
  """

  match_min: float

  def __post_init__(self) -> None:
    if not np.isfinite(self.match_min) or not 0.0 <= self.match_min <= 1.0:
      raise ValueError(f"TokensTol match_min must be finite and in [0, 1], got {self.match_min}")


OutputTol = Union[LogitsTol, TokensTol, layer_parity.Tolerance]


def _log_softmax(x: np.ndarray) -> np.ndarray:
  x = x - np.max(x, axis=-1, keepdims=True)
  return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


def compare_one(golden: np.ndarray, cand: np.ndarray, tol: OutputTol) -> tuple[bool, dict[str, float]]:
  """Compares one output array; the tolerance type selects the output kind."""
  g_arr = np.asarray(golden)
  c_arr = np.asarray(cand)
  if g_arr.shape != c_arr.shape:
    raise ValueError(f"Shape mismatch: candidate {c_arr.shape} vs golden {g_arr.shape}")
  if g_arr.size == 0:
    raise ValueError(f"compare_one requires non-empty arrays, got shape {g_arr.shape}")
  if isinstance(tol, LogitsTol):
    if not (jnp.issubdtype(g_arr.dtype, jnp.floating) and jnp.issubdtype(c_arr.dtype, jnp.floating)):
      raise TypeError(f"logits mode needs floating-point arrays, got {g_arr.dtype} and {c_arr.dtype}")
    if g_arr.ndim < 1 or g_arr.shape[-1] < 2:
      raise ValueError(f"logits mode requires vocab axis shape[-1] >= 2, got shape {g_arr.shape}")
    g = np.asarray(g_arr, dtype=np.float64)
    c = np.asarray(c_arr, dtype=np.float64)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
      return False, {"kl_mean": np.nan, "kl_max": np.nan, "top1": np.nan}
    lp, lq = _log_softmax(g), _log_softmax(c)
    kl_per_pos = np.sum(np.exp(lp) * (lp - lq), axis=-1)
    kl_mean = float(np.mean(kl_per_pos))
    kl_max = float(np.max(kl_per_pos))
    top1 = float(np.mean(np.argmax(g, -1) == np.argmax(c, -1)))
    passed = kl_mean <= tol.kl_mean_max and kl_max <= tol.kl_max and top1 >= tol.top1_min
    return passed, {"kl_mean": kl_mean, "kl_max": kl_max, "top1": top1}
  if isinstance(tol, TokensTol):
    if not (jnp.issubdtype(g_arr.dtype, jnp.integer) and jnp.issubdtype(c_arr.dtype, jnp.integer)):
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
  if jnp.issubdtype(raw.dtype, jnp.floating):
    return bool(np.all(np.isfinite(np.asarray(raw, dtype=np.float64))))
  return True


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
  (the perturbed leaf is not on the compared path, or a discrete `tokens` output
  did not flip — in which case the control must be run on the underlying
  `logits`). Only FIRED is a pass; a parity test whose negative control passes
  (returns MISSED) is a test failure.
  """
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
