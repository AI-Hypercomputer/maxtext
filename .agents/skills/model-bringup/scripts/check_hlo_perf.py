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

"""Deterministic CPU AOT HBM, static HLO pathology, and loss-trajectory gates for Phase P.

Used in Stage P2 (CPU AOT compile gate) and Stage P3 (batched accelerator micro-benchmark)
of `m3-perf-loop` so configurations that do not fit in HBM, inflate MXU FLOPs via
multi-pass fp32 lowering or dense masks, leave collectives exposed, spill to host, or
corrupt training loss are rejected before or during benchmark sweeps.
"""

import argparse
import collections
import dataclasses
import json
import pathlib
import re
import sys
from typing import Iterable, Sequence

import numpy as np

_GIB = float(1024**3)
_COMPUTATION_HEADER_RE = re.compile(r"^(?:ENTRY\s+)?%?(?P<name>[\w.\-]+)\s.*\{\s*$")
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<type>.*?)\s(?P<op>[a-z][\w\-]*)\((?P<args>.*)$"
)
_OPERAND_RE = re.compile(r"%([\w.\-]+)")
_CALLS_RE = re.compile(r"calls=%([\w.\-]+)")
_CALLEE_RE = re.compile(r"(?:calls|to_apply)=%([\w.\-]+)")
_COLLECTIVE_KINDS = (
    "ragged-all-to-all",
    "all-to-all",
    "all-gather",
    "reduce-scatter",
    "all-reduce",
    "collective-permute",
)
_LEGACY_START_OPS = {f"{k}-start": k for k in ("all-gather", "all-reduce", "collective-permute")}
_LEGACY_DONE_OPS = {f"{k}-done" for k in ("all-gather", "all-reduce", "collective-permute")}
# Instructions that do no compute; they never count as overlap between an async start and done.
_NON_COMPUTE_OPS = frozenset(
    {
        "parameter",
        "constant",
        "get-tuple-element",
        "tuple",
        "bitcast",
        "opt-barrier",
        "async-start",
        "async-update",
        "async-done",
        "copy-start",
        "copy-done",
    }
    | set(_LEGACY_START_OPS)
    | _LEGACY_DONE_OPS
)
_SPARSECORE_RE = re.compile(r'async_execution_thread="sparsecore"|sparse_core|sparsecore', re.IGNORECASE)
_HOST_MEMORY_SPACE_RE = re.compile(r"S\(5\)")


@dataclasses.dataclass(frozen=True)
class AOTMemoryStats:
  """Byte counts from XLA `compiled.memory_analysis()`."""

  temp_bytes: int
  argument_bytes: int
  output_bytes: int
  alias_bytes: int
  host_temp_bytes: int

  @property
  def naive_step_gib(self) -> float:
    return max(0, self.temp_bytes + self.argument_bytes + self.output_bytes - self.alias_bytes) / _GIB


@dataclasses.dataclass(frozen=True)
class HLOGateConfig:
  """Thresholds for Stage P2 CPU AOT & Static HLO verification."""

  hbm_ceiling_gib: float = 92.0
  runtime_reserve_gib: float = 6.5
  max_flop_inflation: float = 1.35
  allow_host_spill: bool = False
  require_sparsecore_offload: bool = False
  min_async_overlap_ratio: float = 0.50
  forbid_host_offload_copies: bool = False


def compute_hbm_estimate(
    stats: AOTMemoryStats, runtime_reserve_gib: float, calibrated_load_gib: float | None = None
) -> dict[str, float | str]:
  """Returns the naive HBM sum and the value the gate uses.

  The naive sum (temp + non-aliased arguments/outputs + runtime reserve) is a conservative
  upper bound. On platforms whose `temp_bytes` already includes an executable reservation
  (observed on TPU7x), it over-counts; there the caller passes the project's back-tested
  load-time estimate as `calibrated_load_gib`, and the gate uses that instead.
  """
  naive_peak = stats.naive_step_gib + runtime_reserve_gib
  if calibrated_load_gib is not None and calibrated_load_gib <= 0:
    raise ValueError(f"calibrated_load_gib must be > 0 when given, got {calibrated_load_gib}")
  gating = naive_peak if calibrated_load_gib is None else float(calibrated_load_gib)
  return {
      "naive_step_gib": stats.naive_step_gib,
      "runtime_reserve_gib": runtime_reserve_gib,
      "naive_peak_gib": naive_peak,
      "calibrated_load_gib": float("nan") if calibrated_load_gib is None else float(calibrated_load_gib),
      "gating_peak_gib": gating,
      "gating_source": "naive_sum" if calibrated_load_gib is None else "calibrated_load",
      "host_spill_gib": stats.host_temp_bytes / _GIB,
  }


def _collective_kind(name: str) -> str | None:
  for kind in _COLLECTIVE_KINDS:
    if name.startswith(kind):
      return kind
  return None


def audit_hlo_text(lines: Iterable[str]) -> dict[str, int | float | bool | dict[str, int]]:
  """Audits optimized HLO text for collective overlap, SparseCore offload, host copies and fp32 dots.

  Handles both legacy `all-gather-start`/`-done` pairs and generic `async-start` ->
  `async-update` -> `async-done` chains. A pair counts as overlapped when at least one compute
  instruction sits between its start and done in the same computation. Synchronous collectives
  outside async computations count as exposed (not overlapped).
  """
  async_bodies: set[str] = set()
  call_edges: dict[str, set[str]] = collections.defaultdict(set)
  sync_by_computation: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
  computation = ""
  index_in_computation = 0
  starts: dict[str, tuple[str, int, int, str]] = {}  # name -> (computation, index, compute_count_at_start, kind)
  update_to_start: dict[str, str] = {}
  compute_count = 0
  pairs = collections.Counter()
  overlapped = collections.Counter()
  unresolved_done = 0
  fp32_dot_count = bf16_dot_count = sparsecore_markers = host_offload_copies = 0

  for line in lines:
    header = _COMPUTATION_HEADER_RE.match(line)
    if header and not line.startswith(" "):
      computation, index_in_computation, compute_count = header.group("name"), 0, 0
      continue
    inst = _INSTRUCTION_RE.match(line)
    if not inst:
      continue
    index_in_computation += 1
    name, op, args = inst.group("name"), inst.group("op"), inst.group("args")
    if _SPARSECORE_RE.search(line):
      sparsecore_markers += 1
    if op in ("dot", "convolution", "ragged-dot"):
      dtype = inst.group("type").lstrip("(")
      if dtype.startswith("f32["):
        fp32_dot_count += 1
      elif dtype.startswith("bf16["):
        bf16_dot_count += 1
    if op == "copy-start" and _HOST_MEMORY_SPACE_RE.search(inst.group("type")):
      host_offload_copies += 1
    operands = _OPERAND_RE.findall(args.split(")", 1)[0])
    calls = _CALLS_RE.search(line)
    call_edges[computation].update(_CALLEE_RE.findall(line))

    if op == "async-start":
      if calls:
        async_bodies.add(calls.group(1))
      kind = _collective_kind(name) or "other"
      starts[name] = (computation, index_in_computation, compute_count, kind)
    elif op in _LEGACY_START_OPS:
      starts[name] = (computation, index_in_computation, compute_count, _LEGACY_START_OPS[op])
    elif op == "async-update":
      if operands:
        update_to_start[name] = update_to_start.get(operands[0], operands[0])
    elif op == "async-done" or op in _LEGACY_DONE_OPS:
      start_name = update_to_start.get(operands[0], operands[0]) if operands else None
      if start_name is None or start_name not in starts or starts[start_name][0] != computation:
        unresolved_done += 1
        continue
      _, _, compute_at_start, kind = starts.pop(start_name)
      pairs[kind] += 1
      if compute_count > compute_at_start:
        overlapped[kind] += 1
    elif op in _COLLECTIVE_KINDS:
      sync_by_computation[computation][op] += 1

    if op not in _NON_COMPUTE_OPS:
      compute_count += 1

  # Collectives inside computations reachable from an async-start body are that async op's payload.
  async_reachable: set[str] = set()
  frontier = list(async_bodies)
  while frontier:
    comp = frontier.pop()
    if comp in async_reachable:
      continue
    async_reachable.add(comp)
    frontier.extend(call_edges.get(comp, ()))
  sync_exposed = collections.Counter()
  for comp, counts in sync_by_computation.items():
    if comp not in async_reachable:
      sync_exposed.update(counts)

  collective_pairs = sum(v for k, v in pairs.items() if k != "other")
  collective_overlapped = sum(v for k, v in overlapped.items() if k != "other")
  sync_total = sum(sync_exposed.values())
  collectives_total = collective_pairs + sync_total
  return {
      "async_collective_pairs": collective_pairs,
      "async_collective_overlapped": collective_overlapped,
      "sync_collectives_exposed": sync_total,
      "sync_collectives_by_kind": dict(sync_exposed),
      "collective_overlap_ratio": (collective_overlapped / collectives_total) if collectives_total else float("nan"),
      "other_async_pairs": pairs["other"],
      "other_async_overlapped": overlapped["other"],
      "unresolved_async_done": unresolved_done,
      "fp32_dot_count": fp32_dot_count,
      "bf16_dot_count": bf16_dot_count,
      "sparsecore_markers": sparsecore_markers,
      "has_sparsecore_offload": sparsecore_markers > 0,
      "host_offload_copies": host_offload_copies,
  }


def check_hlo_gate(
    mem_stats: AOTMemoryStats,
    compiled_hlo_flops: float,
    canonical_model_flops: float,
    hlo_lines: Iterable[str],
    *,
    cfg: HLOGateConfig = HLOGateConfig(),
    calibrated_load_gib: float | None = None,
) -> tuple[bool, dict[str, object], list[str]]:
  """Runs all Stage P2 CPU AOT & Static HLO checks."""
  if canonical_model_flops <= 0:
    raise ValueError(f"canonical_model_flops must be > 0, got {canonical_model_flops}")
  mem_report = compute_hbm_estimate(mem_stats, cfg.runtime_reserve_gib, calibrated_load_gib)
  hlo_report = audit_hlo_text(hlo_lines)
  flop_inflation = float(compiled_hlo_flops) / float(canonical_model_flops)

  failures: list[str] = []
  warnings: list[str] = []
  if mem_report["gating_peak_gib"] > cfg.hbm_ceiling_gib:
    failures.append(
        f"HBM {mem_report['gating_peak_gib']:.2f} GiB ({mem_report['gating_source']}) exceeds ceiling "
        f"{cfg.hbm_ceiling_gib:.2f} GiB (naive sum {mem_report['naive_peak_gib']:.2f} GiB)"
    )
  elif mem_report["naive_peak_gib"] > cfg.hbm_ceiling_gib:
    warnings.append(
        f"naive HBM sum {mem_report['naive_peak_gib']:.2f} GiB exceeds ceiling; gate passed on calibrated load "
        f"{mem_report['gating_peak_gib']:.2f} GiB"
    )
  if flop_inflation > cfg.max_flop_inflation:
    failures.append(
        f"HLO FLOP inflation ratio {flop_inflation:.3f}x exceeds max {cfg.max_flop_inflation:.3f}x "
        f"(fp32_dots={hlo_report['fp32_dot_count']}, bf16_dots={hlo_report['bf16_dot_count']})"
    )
  if not cfg.allow_host_spill and mem_stats.host_temp_bytes > 0:
    failures.append(f"Unexpected host memory spill: {mem_report['host_spill_gib']:.3f} GiB")
  if cfg.forbid_host_offload_copies and hlo_report["host_offload_copies"] > 0:
    failures.append(f"{hlo_report['host_offload_copies']} host-memory (S(5)) copy-start ops present")
  if cfg.require_sparsecore_offload and not hlo_report["has_sparsecore_offload"]:
    failures.append("SparseCore offload required, but no SparseCore async thread found in optimized HLO")
  if hlo_report["unresolved_async_done"] > 0:
    failures.append(
        f"{hlo_report['unresolved_async_done']} async-done ops could not be matched to a start; parser and HLO disagree"
    )
  ratio = hlo_report["collective_overlap_ratio"]
  if not np.isnan(ratio) and ratio < cfg.min_async_overlap_ratio:
    failures.append(
        f"Collective overlap ratio {ratio:.2f} below {cfg.min_async_overlap_ratio:.2f} "
        f"({hlo_report['async_collective_overlapped']} overlapped of {hlo_report['async_collective_pairs']} async "
        f"+ {hlo_report['sync_collectives_exposed']} synchronous {hlo_report['sync_collectives_by_kind']})"
    )

  metrics: dict[str, object] = {**mem_report, **hlo_report, "flop_inflation_ratio": flop_inflation, "warnings": warnings}
  return not failures, metrics, failures


def check_loss_trajectory(
    baseline_losses: Sequence[float],
    candidate_losses: Sequence[float],
    max_rel_delta: float = 0.01,
    ragged_buffer_factor: float = 2.0,
) -> tuple[bool, dict[str, float], list[str]]:
  """Stage P3 anti-cheating numerical gate: checks finiteness, token-drop floor, and fixed-seed loss parity."""
  if not baseline_losses or not candidate_losses:
    raise ValueError("baseline_losses and candidate_losses must be non-empty")
  b = np.asarray(baseline_losses, dtype=np.float64)
  c = np.asarray(candidate_losses, dtype=np.float64)
  failures: list[str] = []
  if not (np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
    failures.append("Non-finite loss value (NaN or Inf) detected in trajectory")
    return False, {"max_rel_delta": np.nan, "mean_rel_delta": np.nan}, failures
  if ragged_buffer_factor != -1.0 and ragged_buffer_factor < 2.0:
    failures.append(f"ragged_buffer_factor={ragged_buffer_factor} is below minimum allowed 2.0 (or -1.0 for dropless)")

  n = min(len(b), len(c))
  denom = np.maximum(np.abs(b[:n]), 1e-8)
  rel_deltas = np.abs(c[:n] - b[:n]) / denom
  max_rel = float(np.max(rel_deltas))
  mean_rel = float(np.mean(rel_deltas))
  if max_rel > max_rel_delta:
    failures.append(f"Fixed-seed loss max relative delta {max_rel:.4f} exceeds {max_rel_delta:.4f}")
  return not failures, {"max_rel_delta": max_rel, "mean_rel_delta": mean_rel, "steps_compared": float(n)}, failures


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Run Stage P2 static HLO and AOT memory gates.")
  parser.add_argument("--summary-json", required=True, type=pathlib.Path, help="JSON file with memory and FLOP stats.")
  parser.add_argument("--hlo-path", required=True, type=pathlib.Path, help="Path to optimized HLO text dump.")
  parser.add_argument("--hbm-ceiling-gib", type=float, default=92.0)
  parser.add_argument("--runtime-reserve-gib", type=float, default=6.5)
  parser.add_argument("--max-flop-inflation", type=float, default=1.35)
  parser.add_argument("--min-overlap-ratio", type=float, default=0.50)
  parser.add_argument(
      "--calibrated-load-gib", type=float, help="Back-tested load-time HBM estimate; gates instead of the naive sum."
  )
  parser.add_argument("--require-sparsecore", action="store_true")
  parser.add_argument("--forbid-host-offload", action="store_true", help="Fail if any S(5) host-memory copy remains.")
  args = parser.parse_args(argv)

  data = json.loads(args.summary_json.read_text(encoding="utf-8"))
  required = (
      "temp_bytes",
      "argument_bytes",
      "output_bytes",
      "alias_bytes",
      "host_temp_bytes",
      "compiled_hlo_flops",
      "canonical_model_flops",
  )
  missing = [k for k in required if k not in data]
  if missing:
    raise KeyError(f"{args.summary_json} is missing required keys: {missing}")
  mem_stats = AOTMemoryStats(
      temp_bytes=int(data["temp_bytes"]),
      argument_bytes=int(data["argument_bytes"]),
      output_bytes=int(data["output_bytes"]),
      alias_bytes=int(data["alias_bytes"]),
      host_temp_bytes=int(data["host_temp_bytes"]),
  )
  cfg = HLOGateConfig(
      hbm_ceiling_gib=args.hbm_ceiling_gib,
      runtime_reserve_gib=args.runtime_reserve_gib,
      max_flop_inflation=args.max_flop_inflation,
      require_sparsecore_offload=args.require_sparsecore,
      min_async_overlap_ratio=args.min_overlap_ratio,
      forbid_host_offload_copies=args.forbid_host_offload,
  )
  with args.hlo_path.open(encoding="utf-8") as hlo_file:
    ok, metrics, failures = check_hlo_gate(
        mem_stats=mem_stats,
        compiled_hlo_flops=float(data["compiled_hlo_flops"]),
        canonical_model_flops=float(data["canonical_model_flops"]),
        hlo_lines=hlo_file,
        cfg=cfg,
        calibrated_load_gib=args.calibrated_load_gib,
    )
  print(json.dumps({"ok": ok, "metrics": metrics, "failures": failures}, indent=2, sort_keys=True, default=str))
  return 0 if ok else 2


if __name__ == "__main__":
  sys.dont_write_bytecode = True
  raise SystemExit(main())
