<!--
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
 -->

# MaxText Step-Triage: Offline TPU Straggler & ICI Wave Attribution

`step_triage` is a zero-dependency, pure Python 3 standard-library package for
attributing step-time stragglers and degraded Inter-Chip Interconnect (ICI)
links from TPU runtime logs (`TPU_LOG_DIR`) or in-memory `libtpu.sdk.debug`
event-log callbacks.

## Quick Start (Customer `TPU_LOG_DIR` Dump)

Run directly as a module inside MaxText:

```bash
python3 -m maxtext.diagnostics.step_triage.cli \
  --logs /tmp/tpu_logs/ \
  --json straggler_report.json
```

Or specify explicit topology overrides:

```bash
python3 -m maxtext.diagnostics.step_triage.cli \
  --logs /tmp/tpu_logs/ \
  --topology 4x4x8 --cores_per_chip 2 --pipeline_depth 13 \
  --json straggler_report.json
```

Omit `--json` to print a concise human-readable attribution report to stdout.

## Scope & Attribution Rules

### Approach 1: Active ICI-SDC Probe Wave Attribution
Parses standard (`ICI transfer wait on SEND|RECV flag timed out ...`), pair
(`ICI SDC Test transfer SEND|RECV timeout : <src> -> <dst> ...`), and
compact-codegen (`ICI SDC Test transfer send|recv timeout (dynamic) ...`) lines:
1. **Wave clustering (`D + 1` hop step):** Groups timeout events by normalized
   dispatch `@run_id` and clusters propagation hops separated by
   `pipeline_depth + 1` iterations (default `13 + 1 = 14`), absorbing 1-to-2
   iteration stutters on the same hop.
2. **Bad link (`SUSPECTED_BAD_LINK`):** `SEND` timeout(s) at `S0` (minimum
   iteration in the wave). Identifies the exact directed or bidirectional link
   `<src_core> -> <dst_core>` (`<src_chip> <-> <dst_chip>`).
3. **Slow core (`SUSPECTED_SLOW_CORE`):** Zero `SEND` timeouts at `S0`, while a
   silent center core's 1-hop torus/mesh neighbors time out on `RECV` at `S0`
   and the center core remains silent at `S1 = S0 + D + 1` (disambiguating
   `TC0` vs. `TC1` on dual-core chips). When `IciSdcTest :: ICI BW: ... delay
   iterations: P` is present, cross-checks `(S0 + 1) % P == 0`.
4. **Noise (`NOISE`):** Isolated single-core `RECV` timeout with no `SEND` at
   `S0`, no 1-hop neighbor ring, and no `D + 1` propagation wave.

### Approach 2: Passive Collective `sflag` Wait/Short Timeout Attribution
Parses `Wait timeout on sflag` and `Short timeout on sflag` lines emitted by
`llo_log_report_handler.cc`:
1. **Launch-barrier filter:** Separates `hlo: <no-hlo-instruction>` launch
   sync-flag noise from real collective HLOs (`all-reduce`, `all-gather`,
   `reduce-scatter`, `all-to-all`, `psum`).
2. **Unsatisfied peer complement:** For the earliest collective HLO in each
   steady-state program launch, computes the complement
   `all_cores - {c | latest_val < for_val}`. When `N - 1` of `N` cores time out
   waiting for a sync flag (`latest_val < for_val`), the single silent core in
   the complement is the straggler that entered the collective late.
3. **Self-satisfied delay-injection signature:** Records cores that emit a
   timeout where `latest_val >= for_val` (e.g., `for value {4} ... latest value
   seen {4}` when `xla_tpu_enable_ici_Wait_sflag_delay` is active on a target
   core) and requires step consistency across steady-state launches.

### Multi-Cause Share Attribution & Decision Tree
`report.compute_cause_shares` and `report.next_probe_decision` implement the
staged `P0 -> P1 -> P2 -> P3` decision tree and step-time regression share
equations (`TPU_SLOW_CORE`, `ICI_LINK_DEGRADATION`, `HOST_INPUT_PIPELINE`,
`HOST_CALLBACK_BARRIER`, `FABRIC_CONTENTION`, `UNEXPLAINED`).

## Known Limits

- **Approach 2 collective group size:** Requires the collective to span at least
  4 cores (or the full slice) so that the `N - 1` unsatisfied peer complement
  has cardinality `<= N / 4`. For 2-core sub-collectives, escalate to Approach 1.
- **Overlapping multi-straggler waves in Approach 1:** When 4+ cores simultaneously
  inject delays at coprime periods `P`, a small fraction (~2.4%) of waves
  collide at the same `S0` iteration, masking part of a neighbor ring during
  purely blind attribution; cross-checking `(S0 + 1) % P == 0` resolves 100% of
  overlapping waves.

## Comparison with Legacy `analyze_ici_wave.py`

Running the legacy `analyze_ici_wave.py` script against the same hardware
fixtures exposes five failure modes fixed by `step_triage`:
1. **Coordinate regex (`_COORD_RE`):** `analyze_ici_wave.py` only matches
   `[v\d+[^\]]*]` or `(\d+,\s*\d+,\s*\d+[^)]*)`, returning `s0_culprit_chips: []`
   on all standard `XxYxZ_TCc` hardware logs (`[job.0x3x0_TC0@1]` and
   `[1x0x0_TC0@2:7e388229:-]`).
2. **Case-sensitive line filter:** Filters lines with `"SEND" in line or "RECV"
   in line`, completely missing lowercase compact-codegen lines (`ICI SDC Test
   transfer recv timeout (dynamic) ...`).
3. **Victim-vs-culprit inversion on slow cores:** Reports the `S0` `RECV` cores
   themselves as `s0_culprit_chips`, blaming the 12 healthy 1-hop neighbor
   victims instead of the silent center core that stalled.
4. **No wave clustering, `@run_id` normalization, or `TC0`/`TC1` disambiguation:**
   Takes a single global `min_iteration = iters[0]` across the entire file
   without clustering by `D + 1 = 14` or checking `silent_at_S1`.
5. **Misclassification of Approach-2 `sflag` lines:** The substring `"send sflag
   tracker"` triggers `_SEND_RE` (`is_send=True`) with `min_iteration=0`,
   misclassifying every `sflag` line (including `<no-hlo-instruction>` barrier
   noise) as an `ICI_STRAGGLER_DELAY_WAVE` at `N_min=0`.
