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

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import importlib.util
import pathlib
import sys
from typing import Any, Union

import numpy as np


def _load_sibling(name: str) -> Any:
  """Loads a sibling helper module by filename without requiring sys.path edits."""
  qual = f"_maxtext_bringup_{name}"
  if qual in sys.modules:
    return sys.modules[qual]
  path = pathlib.Path(__file__).resolve().with_name(f"{name}.py")
  spec = importlib.util.spec_from_file_location(qual, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load sibling helper {path}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[qual] = mod
  spec.loader.exec_module(mod)
  return mod


layer_parity = _load_sibling("layer_parity")


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
  min_agreement: float = 0.99
  tie_margin_ulp: float = 8.0


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
  if golden_scores.shape != cand_scores.shape:
    raise ValueError(f"Router score shape mismatch: {cand_scores.shape} vs" f" {golden_scores.shape}")
  if golden_scores.ndim < 2 or golden_scores.shape[-1] <= tol.top_k:
    raise ValueError(f"Router scores {golden_scores.shape} must have last dim >" f" top_k={tol.top_k}")
  g = golden_scores.astype(np.float32).reshape(-1, golden_scores.shape[-1])
  c = cand_scores.astype(np.float32).reshape(-1, cand_scores.shape[-1])
  if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
    return False, {
        "agreement": np.nan,
        "non_tie_fraction": np.nan,
        "max_abs_diff": np.nan,
    }

  g_sorted = np.sort(g, axis=-1)[:, ::-1]
  kth = g_sorted[:, tol.top_k - 1]
  next_k = g_sorted[:, tol.top_k]
  ulp_at_k = np.maximum(np.spacing(np.abs(kth)), np.finfo(np.float32).eps)
  non_tie_mask = (kth - next_k) >= (tol.tie_margin_ulp * ulp_at_k)

  g_topk = np.argsort(g, axis=-1)[:, -tol.top_k :]
  c_topk = np.argsort(c, axis=-1)[:, -tol.top_k :]
  matches = np.array(
      [set(gr.tolist()) == set(cr.tolist()) for gr, cr in zip(g_topk, c_topk)],
      dtype=bool,
  )
  evaluated = matches[non_tie_mask] if np.any(non_tie_mask) else matches
  agreement = float(np.mean(evaluated))
  metrics = {
      "agreement": agreement,
      "non_tie_fraction": float(np.mean(non_tie_mask)),
      "max_abs_diff": float(np.max(np.abs(g.astype(np.float64) - c.astype(np.float64)))),
  }
  return agreement >= tol.min_agreement, metrics


def compare_one(golden: np.ndarray, cand: np.ndarray, tol: OutputTol) -> tuple[bool, dict[str, float]]:
  """Compares one output array; the tolerance type selects the output kind."""
  if golden.shape != cand.shape:
    raise ValueError(f"Shape mismatch: candidate {cand.shape} vs golden {golden.shape}")
  if isinstance(tol, LogitsTol):
    g, c = golden.astype(np.float64), cand.astype(np.float64)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(c))):
      return False, {"kl_mean": np.nan, "top1": np.nan}
    lp, lq = _log_softmax(g), _log_softmax(c)
    kl_mean = float(np.mean(np.sum(np.exp(lp) * (lp - lq), axis=-1)))
    top1 = float(np.mean(np.argmax(g, -1) == np.argmax(c, -1)))
    return kl_mean <= tol.kl_mean_max and top1 >= tol.top1_min, {"kl_mean": kl_mean, "top1": top1}
  if isinstance(tol, TokensTol):
    if not (np.issubdtype(golden.dtype, np.integer) and np.issubdtype(cand.dtype, np.integer)):
      raise TypeError(f"tokens mode needs integer token ids, got {golden.dtype} and {cand.dtype}")
    ratio = float(np.mean(golden == cand))
    return ratio >= tol.match_min, {"match": ratio}
  if isinstance(tol, layer_parity.Tolerance):
    ok, m = layer_parity.compare(cand, golden, tol)
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
  problems = [
      f"perturbed output {k!r} is not finite (a crash-like result is not a fired control)"
      for k in keys
      if np.issubdtype(perturbed[k].dtype, np.floating) and not np.all(np.isfinite(perturbed[k]))
  ]
  if all(np.array_equal(perturbed[k], primary[k]) for k in keys):
    problems.append("perturbed outputs equal the unperturbed outputs: the perturbed leaf is not on the compared path")
  if problems:
    return "ERROR", problems
  perturbed_ok, _, _ = check_outputs(golden, perturbed, keys, tol)
  if perturbed_ok:
    return "MISSED", ["perturbed outputs still pass"]
  return "FIRED", []


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for end-to-end output and router-agreement checks."""
  parser = argparse.ArgumentParser(
      description="Verify end-to-end logits, token IDs, continuous outputs, and MoE router agreement."
  )
  parser.add_argument(
      "--self-test",
      action="store_true",
      help="Run a pure-NumPy self-check on logits and negative-control classification.",
  )
  args = parser.parse_args(argv)
  if args.self_test:
    tol = LogitsTol(kl_mean_max=1e-4, top1_min=1.0)
    g = {"logits": np.array([[2.0, -1.0, 0.5]], dtype=np.float32)}
    p = {"logits": np.array([[-2.0, 3.0, 0.5]], dtype=np.float32)}
    ok, _, _ = check_outputs(g, g, ["logits"], tol)
    status, reasons = negative_run_status(g, g, p, ["logits"], tol)
    if not ok or status != "FIRED":
      raise AssertionError(f"Self-test failed: ok={ok}, status={status}, reasons={reasons}")
    print("E2E_CHECK_SELF_TEST PASS status=FIRED")
  return 0


if __name__ == "__main__":
  sys.exit(main())
