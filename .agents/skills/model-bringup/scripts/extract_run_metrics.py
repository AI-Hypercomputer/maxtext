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

"""Deterministic step-log extractor, throughput-based ETA calculator, and prose number verifier.

Eliminates hand-typed metrics and ungrounded ETAs in status notes and handoff
documents:
1. `extract_step_metrics` parses raw trainer logs into structured JSON
(steady-state step time,
   TFLOP/s/device, MFU, loss trajectory, and verbatim step lines).
2. `compute_eta_seconds` computes remaining wall time strictly from measured
per-item durations.
3. `verify_prose_numbers` checks that every floating-point metric in a rendered
markdown section
   appears in the authoritative JSON dictionary.
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Sequence

import numpy as np

_STEP_RE = re.compile(
    r"completed step:\s*(?P<step>\d+),\s*seconds:\s*(?P<seconds>[0-9.]+),\s*"
    r"TFLOP/s/device:\s*(?P<tflops_per_sec>[0-9.]+),\s*Tokens/s/device:\s*(?P<tokens_per_sec>[0-9.]+)"
    r".*?loss:\s*(?P<loss>-?[0-9.]+(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
_TOTAL_TFLOPS_RE = re.compile(r"Total TFLOPs:\s*(?P<tflops>[0-9.]+)", re.IGNORECASE)
_FLOAT_TOKEN_RE = re.compile(r"(?<![a-zA-Z0-9_])(?P<val>\d+\.\d+)(?![a-zA-Z0-9_])")
# MaxText `log_step_diagnostics=true` appends "diag: grad_norm=1.234567890e+00 ..."
# to the step line; absent values print as "n/a".
_DIAG_RE = re.compile(r"\bdiag:\s*(?P<body>.*)$")
_DIAG_PAIR_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<val>\S+)")


def _parse_diag(line: str) -> dict[str, float | None]:
  """Returns the `diag:` key/value pairs of a step line; `n/a` maps to None."""
  m = _DIAG_RE.search(line)
  if not m:
    return {}
  out: dict[str, float | None] = {}
  for pair in _DIAG_PAIR_RE.finditer(m.group("body")):
    val = pair.group("val")
    out[pair.group("key")] = None if val == "n/a" else float(val)
  return out


def extract_step_metrics(
    log_text: str,
    steady_start_step: int = 10,
    steady_end_step: int = 29,
    peak_tflops_per_device: float = 1153.5,
) -> dict[str, Any]:
  """Extracts verbatim step records and computes steady-state step time and MFU."""
  steps: list[dict[str, float | int | str]] = []
  total_tflops_candidates: list[float] = []

  for line in log_text.splitlines():
    m_tf = _TOTAL_TFLOPS_RE.search(line)
    if m_tf:
      total_tflops_candidates.append(float(m_tf.group("tflops")))
    m_step = _STEP_RE.search(line)
    if m_step:
      steps.append(
          {
              "step": int(m_step.group("step")),
              "seconds": float(m_step.group("seconds")),
              "tflops_per_sec": float(m_step.group("tflops_per_sec")),
              "tokens_per_sec": float(m_step.group("tokens_per_sec")),
              "loss": float(m_step.group("loss")),
              "diag": _parse_diag(line),
              "verbatim": line.strip(),
          }
      )

  if not steps:
    raise ValueError("No 'completed step:' lines found in log_text")

  steady = [s for s in steps if steady_start_step <= int(s["step"]) <= steady_end_step]
  if not steady:
    steady = steps

  mean_sec = float(np.mean([float(s["seconds"]) for s in steady]))
  total_tflops = total_tflops_candidates[-1] if total_tflops_candidates else float(steady[0]["tflops_per_sec"]) * mean_sec
  achieved_tflops_per_dev = total_tflops / mean_sec if mean_sec > 0 else 0.0
  mfu_pct = (achieved_tflops_per_dev / peak_tflops_per_device) * 100.0 if peak_tflops_per_device > 0 else 0.0

  return {
      "num_steps_total": len(steps),
      "num_steps_steady": len(steady),
      "steady_start_step": int(steady[0]["step"]),
      "steady_end_step": int(steady[-1]["step"]),
      "total_tflops_per_device": round(total_tflops, 4),
      "mean_step_seconds": round(mean_sec, 4),
      "achieved_tflops_per_sec_per_device": round(achieved_tflops_per_dev, 4),
      "mfu_percent": round(mfu_pct, 3),
      "first_loss": float(steps[0]["loss"]),
      "last_loss": float(steps[-1]["loss"]),
      "losses": [float(s["loss"]) for s in steps],
      # Step -> grad_norm; None when the step line has no diag value (never 0).
      "grad_norms": {int(s["step"]): s["diag"].get("grad_norm") for s in steps},
      "verbatim_sample": [str(s["verbatim"]) for s in steady[:3]],
  }


def max_relative_grad_norm_delta(
    grad_norms_a: dict[int, float | None],
    grad_norms_b: dict[int, float | None],
    steps: Sequence[int] = (1, 2, 3),
) -> float:
  """Largest relative grad_norm difference over `steps`, relative to run A.

  Used twice by the Stage P3 screen: baseline vs. baseline gives the noise
  floor, candidate vs. baseline gives the delta compared against it. Raises if
  a step is missing or was logged without a grad_norm, so a run without
  `log_step_diagnostics=true` can never pass.
  """
  deltas = []
  for step in steps:
    a, b = grad_norms_a.get(step), grad_norms_b.get(step)
    if a is None or b is None:
      raise ValueError(f"grad_norm missing at step {step}; set log_step_diagnostics=true")
    if not (np.isfinite(a) and np.isfinite(b)):
      raise ValueError(f"non-finite grad_norm at step {step}: {a}, {b}")
    deltas.append(abs(b - a) / max(abs(a), 1e-30))
  return float(max(deltas))


def compute_eta_seconds(
    completed_durations_sec: Sequence[float],
    remaining_items: int,
    concurrency: int = 1,
) -> float:
  """Computes remaining wall-clock seconds strictly from measured item durations."""
  if remaining_items <= 0:
    return 0.0
  if not completed_durations_sec:
    raise ValueError("Cannot compute ETA without at least one measured item duration")
  if concurrency <= 0:
    raise ValueError(f"concurrency must be >= 1, got {concurrency}")
  median_sec = float(np.median(np.asarray(completed_durations_sec, dtype=np.float64)))
  return float((remaining_items * median_sec) / concurrency)


def _collect_json_floats(obj: Any, out: set[str]) -> None:
  """Collects formatted decimal string representations from nested JSON values."""
  if isinstance(obj, float):
    for decimals in (1, 2, 3, 4):
      out.add(f"{obj:.{decimals}f}")
    out.add(str(obj))
  elif isinstance(obj, dict):
    for v in obj.values():
      _collect_json_floats(v, out)
  elif isinstance(obj, (list, tuple)):
    for v in obj:
      _collect_json_floats(v, out)


def verify_prose_numbers(prose_text: str, source_data: dict[str, Any]) -> tuple[bool, list[str]]:
  """Checks that every decimal number in `prose_text` matches a number in `source_data`."""
  allowed: set[str] = set()
  _collect_json_floats(source_data, allowed)
  unverified: list[str] = []
  for match in _FLOAT_TOKEN_RE.finditer(prose_text):
    token = match.group("val")
    if token not in allowed:
      unverified.append(token)
  return not unverified, unverified


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint to extract step metrics JSON from a MaxText trainer log."""
  parser = argparse.ArgumentParser(
      description="Extract steady-state step time, TFLOP/s/device, MFU, and loss metrics from a trainer log."
  )
  parser.add_argument("log_path", type=pathlib.Path, help="Path to worker trainer log file.")
  parser.add_argument("--steady-start-step", type=int, default=10)
  parser.add_argument("--steady-end-step", type=int, default=29)
  parser.add_argument("--peak-tflops-per-device", type=float, default=1153.5)
  args = parser.parse_args(argv)
  text = args.log_path.read_text(encoding="utf-8")
  metrics = extract_step_metrics(
      text,
      steady_start_step=args.steady_start_step,
      steady_end_step=args.steady_end_step,
      peak_tflops_per_device=args.peak_tflops_per_device,
  )
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  sys.dont_write_bytecode = True
  raise SystemExit(main())
