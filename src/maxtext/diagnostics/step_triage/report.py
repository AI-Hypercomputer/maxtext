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

"""Structured `straggler_report` JSON builder, cause-share attribution, and human formatter."""

from __future__ import annotations

from datetime import datetime, timezone
import statistics
from typing import Any


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
  return max(lo, min(hi, val))


def _median_or_none(vals: list[float] | None) -> float | None:
  if not vals:
    return None
  return float(statistics.median(vals))


def compute_cause_shares(
    baseline_step_ms: float | None = None,
    anomalous_step_ms: float | None = None,
    p0_step_ms: float | None = None,
    p1_step_ms: float | None = None,
    p3_collective_ms: float | None = None,
    baseline_collective_ms: float | None = None,
    has_slow_core: bool = False,
    has_bad_link: bool = False,
) -> dict[str, float]:
  """Computes multi-cause step-time shares following Section 5.3 equations.

  Args:
    baseline_step_ms: `T_base`, median step time in the healthy window.
    anomalous_step_ms: `T_curr`, median step time in the anomalous window.
    p0_step_ms: `median(T_P0)` with synthetic data (`dataset_type=synthetic`).
    p1_step_ms: `median(T_P1)` with callbacks/checkpointer disabled.
    p3_collective_ms: `T_P3_collective_only` from microbenchmark.
    baseline_collective_ms: `T_base_coll` expected healthy collective time.
    has_slow_core: Whether P2 (Approach 1 or 2) isolated a slow core.
    has_bad_link: Whether P2 (Approach 1) isolated a degraded ICI link.

  Returns:
    Dictionary mapping cause names to normalized shares in `[0.0, 1.0]`.
  """
  share_p0 = 0.0
  share_p1 = 0.0
  share_p3 = 0.0

  if baseline_step_ms is not None and anomalous_step_ms is not None and anomalous_step_ms > baseline_step_ms:
    delta_t = anomalous_step_ms - baseline_step_ms
    if p0_step_ms is not None:
      share_p0 = _clamp((anomalous_step_ms - p0_step_ms) / delta_t)
    ref_before_p1 = p0_step_ms if p0_step_ms is not None else anomalous_step_ms
    if p1_step_ms is not None:
      share_p1 = _clamp((ref_before_p1 - p1_step_ms) / delta_t)
    if p3_collective_ms is not None and baseline_collective_ms is not None:
      share_p3 = _clamp((p3_collective_ms - baseline_collective_ms) / delta_t)

  remaining = _clamp(1.0 - share_p0 - share_p1 - share_p3)
  share_slow_core = 0.0
  share_bad_link = 0.0
  if has_slow_core and has_bad_link:
    share_slow_core = round(remaining * 0.5, 4)
    share_bad_link = round(remaining - share_slow_core, 4)
  elif has_slow_core:
    share_slow_core = round(remaining, 4)
  elif has_bad_link:
    share_bad_link = round(remaining, 4)

  total_assigned = share_p0 + share_p1 + share_p3 + share_slow_core + share_bad_link
  unexplained = (
      _clamp(1.0 - total_assigned)
      if total_assigned > 0
      else (0.0 if (not has_slow_core and not has_bad_link and baseline_step_ms is None) else 1.0)
  )

  return {
      "TPU_SLOW_CORE": round(share_slow_core, 4),
      "ICI_LINK_DEGRADATION": round(share_bad_link, 4),
      "HOST_INPUT_PIPELINE": round(share_p0, 4),
      "HOST_CALLBACK_BARRIER": round(share_p1, 4),
      "FABRIC_CONTENTION": round(share_p3, 4),
      "UNEXPLAINED": round(unexplained, 4),
  }


def next_probe_decision(
    baseline_step_ms: float | None = None,
    anomalous_step_ms: float | None = None,
    p0_ran: bool = False,
    p0_step_ms: float | None = None,
    p1_ran: bool = False,
    p1_step_ms: float | None = None,
    p2_sflag_ran: bool = False,
    p2_sflag_culprits: list[str] | None = None,
    p2_ici_ran: bool = False,
    p2_ici_culprits: list[str] | None = None,
    p3_ran: bool = False,
) -> str:
  """Implements the Section 7 escalation decision tree (`P0 -> P1 -> P2 -> P3`)."""
  if baseline_step_ms is not None and anomalous_step_ms is not None and anomalous_step_ms <= 1.15 * baseline_step_ms:
    return "HEALTHY_NO_PROBE"

  shares = compute_cause_shares(
      baseline_step_ms=baseline_step_ms,
      anomalous_step_ms=anomalous_step_ms,
      p0_step_ms=p0_step_ms if p0_ran else None,
      p1_step_ms=p1_step_ms if p1_ran else None,
  )

  if not p0_ran:
    return "RUN_P0_SYNTHETIC_DATA"
  if shares["HOST_INPUT_PIPELINE"] >= 0.80:
    return "EMIT_REPORT"

  if not p1_ran:
    return "RUN_P1_DISABLE_CALLBACKS"
  if shares["HOST_INPUT_PIPELINE"] + shares["HOST_CALLBACK_BARRIER"] >= 0.80:
    return "EMIT_REPORT"

  if not p2_sflag_ran:
    return "RUN_P2_APPROACH2_SFLAG"
  if p2_sflag_culprits and len(p2_sflag_culprits) == 1:
    return "EMIT_REPORT"

  if not p2_ici_ran:
    return "RUN_P2_APPROACH1_ICI_SDC"
  if p2_ici_culprits:
    return "EMIT_REPORT"

  if not p3_ran:
    return "RUN_P3_MICROBENCHMARKS"
  return "EMIT_REPORT"


def build_straggler_report(
    analysis: dict[str, Any],
    job_name: str = "tpu_workload",
    trigger_reason: str = "offline_log_triage",
    baseline_step_times_ms: list[float] | None = None,
    anomalous_step_times_ms: list[float] | None = None,
    p0_step_times_ms: list[float] | None = None,
    p1_step_times_ms: list[float] | None = None,
    p3_allreduce_gibs: float | None = None,
    p3_expected_gibs: float | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
  """Builds the canonical Section 6 `straggler_report` JSON dictionary."""
  topo = analysis.get("topology") or {}
  bounds = topo.get("bounds") or [0, 0, 0]
  cpc = topo.get("cores_per_chip") or 2
  chips_count = bounds[0] * bounds[1] * bounds[2] if all(bounds) else 0
  cores_count = topo.get("total_cores") or (chips_count * cpc)
  topo_str = f"{bounds[0]}x{bounds[1]}x{bounds[2]}" if all(bounds) else "unknown"

  a1_slow = analysis.get("suspected_slow_cores") or []
  a1_links = analysis.get("suspected_links") or []
  sflag_sum = analysis.get("approach2_sflag_summary") or {}
  a2_slow = sflag_sum.get("suspected_straggler_cores") or []

  has_slow_core = bool(a1_slow or a2_slow)
  has_bad_link = bool(a1_links)

  t_base = _median_or_none(baseline_step_times_ms)
  t_curr = _median_or_none(anomalous_step_times_ms)
  t_p0 = _median_or_none(p0_step_times_ms)
  t_p1 = _median_or_none(p1_step_times_ms)
  delta_ms = round(t_curr - t_base, 3) if (t_base is not None and t_curr is not None) else None

  shares = compute_cause_shares(
      baseline_step_ms=t_base,
      anomalous_step_ms=t_curr,
      p0_step_ms=t_p0,
      p1_step_ms=t_p1,
      has_slow_core=has_slow_core,
      has_bad_link=has_bad_link,
  )

  # Collect suspect cores, chips, hosts, and links.
  suspect_cores: list[str] = []
  suspect_chips: list[str] = []
  suspect_hosts: list[str] = []
  suspect_links: list[str] = list(analysis.get("culprit_links") or [])

  for item in list(a1_slow) + list(a2_slow):
    c = item.get("core")
    ch = item.get("chip")
    h = item.get("host")
    if c and c not in suspect_cores:
      suspect_cores.append(c)
    if ch and ch not in suspect_chips:
      suspect_chips.append(ch)
    if h and h not in suspect_hosts:
      suspect_hosts.append(h)

  for sl in a1_links:
    for ch_k in ("src_chip", "dst_chip"):
      ch = sl.get(ch_k)
      if ch and ch != "unknown" and ch not in suspect_chips:
        suspect_chips.append(ch)
    for h_k in ("src_host", "dst_host"):
      h = sl.get(h_k)
      if h and h not in suspect_hosts:
        suspect_hosts.append(h)

  # Determine primary_cause, confidence, and recommended_action.
  raw_verdict = analysis.get("overall_verdict", "CLEAN")
  if has_slow_core and has_bad_link:
    primary_cause = "TPU_SLOW_CORE_AND_ICI_LINK"
    recommended_action = "DRAIN_AND_REPLACE_HOST"
    confidence = 0.95
  elif has_slow_core:
    primary_cause = "TPU_SLOW_CORE"
    recommended_action = "DRAIN_AND_REPLACE_HOST"
    if a1_slow and a2_slow:
      confidence = 0.99
    elif a1_slow:
      confidence = round(max(0.85, float(a1_slow[0].get("mean_neighbor_coverage", 1.0))), 2)
    else:
      top_a2 = a2_slow[0]
      confidence = 0.98 if top_a2.get("consistent_steps", 0) >= 3 else 0.90
  elif has_bad_link:
    primary_cause = "ICI_LINK_DEGRADATION"
    recommended_action = "INSPECT_ICI_LINK"
    confidence = 0.95
  elif shares["HOST_INPUT_PIPELINE"] >= 0.50:
    primary_cause = "HOST_INPUT_PIPELINE"
    recommended_action = "OPTIMIZE_HOST_PIPELINE"
    confidence = round(shares["HOST_INPUT_PIPELINE"], 2)
  elif shares["HOST_CALLBACK_BARRIER"] >= 0.50:
    primary_cause = "HOST_CALLBACK_BARRIER"
    recommended_action = "OPTIMIZE_HOST_PIPELINE"
    confidence = round(shares["HOST_CALLBACK_BARRIER"], 2)
  elif raw_verdict == "LAUNCH_BARRIER_NOISE_ONLY":
    primary_cause = "LAUNCH_BARRIER_NOISE_ONLY"
    recommended_action = "NO_ACTION"
    confidence = 0.95
  elif raw_verdict == "NOISE_ONLY":
    primary_cause = "NOISE_ONLY"
    recommended_action = "NO_ACTION"
    confidence = 0.90
  else:
    primary_cause = "CLEAN"
    recommended_action = "NO_ACTION"
    confidence = 1.0

  waves = analysis.get("waves") or []
  bad_link_waves = sum(1 for w in waves if w.get("verdict") == "SUSPECTED_BAD_LINK")
  slow_core_waves = sum(1 for w in waves if w.get("verdict") == "SUSPECTED_SLOW_CORE")
  noise_waves_cnt = analysis.get("noise_wave_count", 0)

  p2_a1_culprits = [
      {
          "core": sc["core"],
          "global_core_index": sc.get("global_core_index"),
          "host": sc.get("host", ""),
          "waves_attributed": sc["wave_count"],
          "blind_waves_attributed": sc.get("blind_wave_count", sc["wave_count"]),
          "logged_delay_iterations_P": sc.get("logged_p"),
          "effective_delay_iterations_P": sc.get("effective_p"),
          "s0_iterations": sc.get("s0_iterations", []),
          "p_divides_s0_plus_1": (
              all((s + 1) % sc["effective_p"] == 0 for s in sc.get("s0_iterations", []))
              if sc.get("effective_p")
              else None
          ),
          "mean_s0_recv_neighbor_coverage": sc.get("mean_neighbor_coverage", 1.0),
      }
      for sc in a1_slow
  ]

  p2_a2_culprits = [
      {
          "core": sc["core"],
          "global_core_index": sc.get("global_core_index"),
          "host": sc.get("host", ""),
          "peer_Wait_sflag_timeouts": sc.get("peer_Wait_sflag_timeouts", 0),
          "total_cores": sc.get("total_cores", cores_count),
          "earliest_hlo_op": sc.get("earliest_hlo_op", ""),
          "alloc_label": sc.get("alloc_label", ""),
          "consistent_steps": sc.get("consistent_steps", 0),
          "self_satisfied_timeouts": sc.get("self_satisfied_timeouts", 0),
      }
      for sc in a2_slow
  ]

  hosts_set = {h for h in suspect_hosts if h}
  for g_info in (analysis.get("ground_truth_p") or {}).values():
    if g_info.get("host"):
      hosts_set.add(g_info["host"])

  ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

  return {
      "schema_version": "1.0",
      "job": {
          "name": job_name,
          "topology": topo_str,
          "chips": chips_count,
          "cores": cores_count,
          "hosts": len(hosts_set) or max(1, chips_count // 4),
      },
      "trigger": {
          "reason": trigger_reason,
          "baseline_step_ms": t_base,
          "anomalous_step_ms": t_curr,
          "delta_step_ms": delta_ms,
          "timestamp_utc": ts,
      },
      "verdict": {
          "primary_cause": primary_cause,
          "overall_verdict": raw_verdict,
          "confidence": confidence,
          "cause_shares": shares,
          "recommended_action": recommended_action,
          "suspect_hosts": suspect_hosts,
          "suspect_chips": suspect_chips,
          "suspect_cores": suspect_cores,
          "suspect_links": suspect_links,
      },
      "probes": {
          "p0_synthetic_data": {
              "ran": t_p0 is not None,
              "step_ms": t_p0,
              "share": shares["HOST_INPUT_PIPELINE"],
          },
          "p1_noop_callbacks": {
              "ran": t_p1 is not None,
              "step_ms": t_p1,
              "share": shares["HOST_CALLBACK_BARRIER"],
          },
          "p2_approach2_sflag": {
              "ran": sflag_sum.get("raw_event_count", 0) > 0,
              "total_events": sflag_sum.get("unique_event_count", 0),
              "launch_barrier_filtered": sflag_sum.get("launch_barrier_count", 0),
              "collective_events": sflag_sum.get("collective_event_count", 0),
              "verdict": sflag_sum.get("verdict", "CLEAN"),
              "culprit_cores": p2_a2_culprits,
          },
          "p2_approach1_ici_sdc": {
              "ran": analysis.get("total_xfer_timeout_events", 0) > 0 or bool(analysis.get("ground_truth_p")),
              "pipeline_depth": analysis.get("pipeline_depth", 13),
              "hop_step": analysis.get("expected_hop_step", 14),
              "total_waves": analysis.get("total_waves_clustered", 0),
              "noise_waves": noise_waves_cnt,
              "bad_link_waves": bad_link_waves,
              "slow_core_waves": slow_core_waves,
              "culprit_cores": p2_a1_culprits,
              "culprit_links": suspect_links,
          },
          "p3_microbenchmarks": {
              "ran": p3_allreduce_gibs is not None,
              "allreduce_1gib_gibs": p3_allreduce_gibs,
              "expected_gibs": p3_expected_gibs,
              "share": shares["FABRIC_CONTENTION"],
          },
      },
      "analysis": analysis,
  }


def format_human_report(report: dict[str, Any], max_waves: int = 20) -> str:
  """Renders a concise, human-readable summary of a `straggler_report` dict."""
  job = report.get("job") or {}
  verdict = report.get("verdict") or {}
  probes = report.get("probes") or {}
  analysis = report.get("analysis") or {}
  p2_a1 = probes.get("p2_approach1_ici_sdc") or {}
  p2_a2 = probes.get("p2_approach2_sflag") or {}

  lines: list[str] = []
  lines.append("=" * 76)
  lines.append("MaxText Step-Triage Straggler Attribution Report")
  lines.append("=" * 76)
  lines.append(f"Topology          : {job.get('topology')} " f"({job.get('chips')} chips, {job.get('cores')} cores)")
  lines.append(f"Primary cause     : {verdict.get('primary_cause')}")
  lines.append(f"Overall verdict   : {verdict.get('overall_verdict')}")
  lines.append(f"Confidence        : {verdict.get('confidence')}")
  lines.append(f"Recommended action: {verdict.get('recommended_action')}")
  if verdict.get("suspect_cores"):
    lines.append(f"Suspect core(s)   : {', '.join(verdict['suspect_cores'])}")
  if verdict.get("suspect_links"):
    lines.append(f"Suspect link(s)   : {', '.join(verdict['suspect_links'])}")
  if verdict.get("suspect_hosts"):
    lines.append(f"Suspect host(s)   : {', '.join(verdict['suspect_hosts'])}")

  lines.append(
      f"\n--- Approach 1 (ICI-SDC Active Probe) ---\n"
      f"  Pipeline depth D={p2_a1.get('pipeline_depth')} (hop step D+1={p2_a1.get('hop_step')})\n"
      f"  Waves clustered : {p2_a1.get('total_waves')} "
      f"(slow_core={p2_a1.get('slow_core_waves')}, "
      f"bad_link={p2_a1.get('bad_link_waves')}, "
      f"noise={p2_a1.get('noise_waves')})"
  )
  for sc in p2_a1.get("culprit_cores") or []:
    lines.append(
        f"  * Slow core {sc['core']} (idx={sc.get('global_core_index')}, "
        f"host={sc.get('host')}): waves={sc['waves_attributed']}, "
        f"P={sc.get('effective_delay_iterations_P')}, "
        f"mean_neighbor_cov={sc.get('mean_s0_recv_neighbor_coverage')}"
    )
  for sl in p2_a1.get("culprit_links") or []:
    lines.append(f"  * Bad link {sl}")

  lines.append(
      f"\n--- Approach 2 (Passive Collective sflag Timeouts) ---\n"
      f"  Unique events   : {p2_a2.get('total_events')} "
      f"(collective={p2_a2.get('collective_events')}, "
      f"launch_barrier_filtered={p2_a2.get('launch_barrier_filtered')})\n"
      f"  Verdict         : {p2_a2.get('verdict')}"
  )
  for sc in p2_a2.get("culprit_cores") or []:
    lines.append(
        f"  * Silent straggler {sc['core']} (idx={sc.get('global_core_index')}, "
        f"host={sc.get('host')}): "
        f"{sc['peer_Wait_sflag_timeouts']}/{sc['total_cores']} peers timed out on "
        f"{sc['earliest_hlo_op']} across {sc['consistent_steps']} step(s) "
        f"(self_satisfied={sc.get('self_satisfied_timeouts', 0)})"
    )

  waves = analysis.get("waves") or []
  if waves:
    lines.append("\n--- Wave Breakdown ---")
    for w in waves[:max_waves]:
      lines.append(
          f"  [Wave {w['wave_id']}] S0={w['s0_iteration']} "
          f"events={w['total_events']} cores={w['affected_cores_count']} "
          f"verdict={w['verdict']} "
          f"blind_top={w.get('blind_top_core')} p_top={w.get('p_checked_top_core')}"
      )
    if len(waves) > max_waves:
      lines.append(f"  ... ({len(waves) - max_waves} additional waves omitted)")

  return "\n".join(lines) + "\n"
