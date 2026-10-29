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
staged `Stage A (host timing) -> Stage B (metrics + compute probe) -> Stage C (ICI probe) -> Stage D (diagnostic steps)`
decision tree (`STAGE_A_HOST_TIMING`, `STAGE_B_METRICS`, `STAGE_C_ICI_PROBE`, `STAGE_D_DIAG_STEPS`) and step-time regression share
equations (`TPU_SLOW_CORE`, `ICI_LINK_DEGRADATION`, `HOST_INPUT_PIPELINE`,
`HOST_CALLBACK_BARRIER`, `FABRIC_CONTENTION`, `UNEXPLAINED`).

### Step-Time Regression Trigger (`trigger.py`)
`trigger.StepTimeTrigger` (configured via `trigger.TriggerConfig`) provides a
pure-stdlib per-step trigger with `"target"`, `"baseline"` (rolling clean-step
median over `window` after `warmup_steps`), and `"both"` modes:
- **Workload exclusions:** Tagged steps (`profiler`, `checkpoint`, `eval`,
  `dropless_replay`, `recompile`, `warmup`, or custom tags) never count toward
  a slow streak, do not reset an in-progress streak, and are excluded from the
  clean baseline window. Because MaxText's `train.py` computes `step_time_delta`
  (`L981`) before `maybe_save_checkpoint` (`L986`), the `eval` block
  (`L1001–1046`), and `maybe_deactivate_profiler` (`L1055`), `checkpoint` also
  excludes the next `exclude_after_checkpoint_steps` steps (default `2`), while
  `eval` and `profiler` also exclude step `N + 1`.
- **Data-wait subtraction:** `device_step_ms = step_ms - data_wait_ms`. When
  `device_step_ms` is within threshold but `step_ms` exceeds threshold for
  `consecutive_steps` steps, the trigger fires with `cause="INPUT_BOUND"` and
  recommends `STAGE_A_HOST_TIMING` only (skipping Stages C/D).

Example integration at MaxText's `train.py` hook points:

```python
from maxtext.diagnostics.step_triage import report
from maxtext.diagnostics.step_triage import trigger

step_trigger = trigger.StepTimeTrigger(
    trigger.TriggerConfig.from_dict(config.get_keys())
)

for step in range(start_step, config.steps):
  # ... load batch (measuring host data_wait_ms) & run train_step ...
  # L976-979: dropless MoE replay retry
  if replayed_step:
    step_trigger.mark(step, "dropless_replay")

  # L981: step_time_delta = datetime.datetime.now() - last_step_completion
  decision = step_trigger.observe(
      step=int(step),
      step_ms=step_time_delta.total_seconds() * 1000.0,
      data_wait_ms=data_wait_ms,
  )

  # L986: maybe_save_checkpoint (cost lands in step+1..step+2)
  if saved_checkpoint:
    step_trigger.mark(int(step), "checkpoint")
  # L1001-1053: eval loop (cost lands in step+1)
  if ran_eval:
    step_trigger.mark(int(step), "eval")
  # L1055: profiler activation / deactivation boundary (cost lands in step+1)
  if profiler_boundary_on_step:
    step_trigger.mark(int(step), "profiler")

  if decision.fired:
    # decision.recommended_stages is ("STAGE_A_HOST_TIMING",) for INPUT_BOUND
    # or ("STAGE_A_HOST_TIMING", ..., "STAGE_D_DIAG_STEPS") for DEVICE_SLOW.
    straggler_rep = report.build_straggler_report(
        analysis={},
        job_name=config.run_name,
        trigger_summary=step_trigger.summary(),
    )
```

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
