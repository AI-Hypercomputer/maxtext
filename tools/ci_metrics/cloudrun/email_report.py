# Copyright 2026 Google LLC
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

"""Builds and sends the 8-hour CI Pulse HTML digest (or failure alert) via GCP Application Integration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import html
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Mapping, Sequence
import urllib.request

from collector import derive as ci_derive
from collector import rows as ci_rows
from collector import store as ci_store
from collector import views as ci_views

LIVE_URL = "https://ai-hypercomputer.github.io/maxtext/dev/bench/#ci"
BENCH_BASE_URL = "https://ai-hypercomputer.github.io/maxtext/dev/bench"
REPO_URL = "https://github.com/AI-Hypercomputer/maxtext"
METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
IAM_CREDENTIALS_URL = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateAccessToken"
INTEGRATIONS_EXECUTE_URL = "https://integrations.googleapis.com/v1/{integration}:execute"

GROUPS = (
    ("runs", "Run + job rows"),
    ("suites", "Test suite rows"),
    ("flaky", "Flaky job rows"),
    ("queue", "Queue wait rows"),
    ("workflows", "Workflow-day rows"),
)
CELL = "border: 1px solid #dadce0; padding: 6px 12px;"
TABLE = 'style="border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px;"'


@dataclass(frozen=True)
class EmailSettings:
  """How and when the Cloud Run job sends its HTML report."""

  recipients: tuple[str, ...] = ()
  integration: str = ""
  trigger_id: str = "api_trigger/mesa_ci_pulse_email"
  invoker_sa: str = ""
  interval_seconds: int = 27600
  force: bool = False

  @property
  def enabled(self) -> bool:
    """True when both recipients and an Application Integration resource name are configured."""
    return bool(self.recipients and self.integration)


def parse_time(stamp: Any) -> datetime | None:
  """Parses a UTC ISO-8601 timestamp like `2026-11-10T01:20:00Z`."""
  if not isinstance(stamp, str) or not stamp.strip():
    return None
  try:
    return datetime.strptime(stamp.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
  except ValueError:
    return None


def fmt_time(stamp: Any) -> str:
  """Formats a UTC timestamp for human-readable email display."""
  parsed = parse_time(stamp)
  return parsed.strftime("%b %d, %H:%M UTC") if parsed is not None else str(stamp or "unknown")


def fmt_duration(seconds: float) -> str:
  """Formats a duration in seconds as `X min Y s` or `X h Y min`."""
  whole = int(round(max(0.0, seconds)))
  if whole < 60:
    return f"{whole} s"
  minutes, rem = divmod(whole, 60)
  if minutes < 60:
    return f"{minutes} min {rem} s"
  hours, minutes = divmod(minutes, 60)
  return f"{hours} h {minutes} min"


def fmt_short_dur(seconds: float) -> str:
  """Compact duration format (`12m 05s`) used in insight bullet lines."""
  whole = int(round(max(0.0, seconds)))
  if whole < 60:
    return f"{whole}s"
  minutes, rem = divmod(whole, 60)
  if minutes < 60:
    return f"{minutes}m {rem:02d}s" if rem else f"{minutes}m"
  hours, minutes = divmod(minutes, 60)
  return f"{hours}h {minutes:02d}m"


def should_send_digest(
    now: datetime,
    last_emailed_at: str | None,
    interval_seconds: int,
    force: bool = False,
) -> bool:
  """True when the 8-hour digest window has elapsed (or on the first run / forced run)."""
  if force:
    return True
  previous = parse_time(last_emailed_at)
  if previous is None:
    return True
  return (now - previous).total_seconds() >= interval_seconds


def _kv(key: str, text: str, html_value: str | None = None) -> tuple[str, str, str]:
  return (key, text, html.escape(text) if html_value is None else html_value)


def _link(url: str, text: str) -> str:
  return f'<a href="{html.escape(url)}">{html.escape(text)}</a>'


def _rich_html_line(text: str) -> str:
  esc = html.escape(text)
  esc = re.sub(
      r"`([^`]+)`",
      r'<code style="background: #f1f3f4; padding: 1px 4px; border-radius: 3px; font-size: 12px;">\1</code>',
      esc,
  )
  return re.sub(r"#(\d+)", rf'<a href="{REPO_URL}/pull/\1">#\1</a>', esc)


def _change_rows(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> list[tuple[str, int, int]]:
  """Returns `(label, before, after)` rows comparing two `meta.json` documents."""
  old_map = old or {}
  rows = [
      (
          "Merged PRs (one bar each)",
          len(old_map.get("pull_requests", {}) or {}),
          len(new.get("pull_requests", {}) or {}),
      )
  ]
  for key, label in GROUPS:
    before = int((old_map.get("totals") or {}).get(key, 0) or 0)
    after = int((new.get("totals") or {}).get(key, 0) or 0)
    rows.append((label, before, after))
  return rows


def _new_prs(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> list[str]:
  before = set((old or {}).get("pull_requests") or {})
  current = new.get("pull_requests") or {}
  return sorted((str(n) for n in current if str(n) not in before), key=int)


def _fetch_text(url: str, timeout: float = 10.0) -> str | None:
  """Reads a public text file over HTTPS; returns None if unreachable."""
  try:
    req = urllib.request.Request(url, headers={"User-Agent": "ci-pulse-collector"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.read().decode("utf-8", errors="replace")
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _load_json_file(path: Path) -> dict[str, Any]:
  if not path.is_file():
    return {}
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
  except (OSError, ValueError):
    return {}


def _table_rows(doc: Mapping[str, Any], table_name: str) -> list[dict[str, Any]]:
  table = (doc.get("tables") or {}).get(table_name) or {}
  cols = table.get("columns") or []
  rows = table.get("rows") or []
  return [dict(zip(cols, row)) for row in rows if isinstance(row, list)]


def _slow_and_new_test_insights(
    store_dir: Path,
    recent_prs: Sequence[int],
    ptb: Mapping[str, Any],
) -> tuple[list[tuple[str, list[str]]], dict[tuple[str, Any], tuple[float, str, int | None]]]:
  """Computes slowed tests and new heavy tests across recent merged PRs."""
  slowed: dict[tuple[str, str, float], list[tuple[float, int, Any]]] = defaultdict(list)
  new_heavy: dict[tuple[str, str], list[tuple[float, int, Any]]] = defaultdict(list)
  heaviest: dict[tuple[str, Any], tuple[float, str, int | None]] = defaultdict(lambda: (0.0, "", None))

  for pr_num in recent_prs:
    pr_doc = _load_json_file(store_dir / f"views/pr/{pr_num}.json")
    seen_in_pr: dict[str, tuple[float, str, Any, str]] = {}
    for td in _table_rows(pr_doc, "tests"):
      dur = float(td.get("duration") or 0.0)
      if dur <= 0:
        continue
      short_class = str(td.get("classname") or "").rsplit(".", 1)[-1]
      short_name = f"{short_class}.{td.get('name')}"
      key = f"{td.get('flavor')}::{td.get('classname')}.{td.get('name')}"
      if key not in seen_in_pr or dur > seen_in_pr[key][0]:
        seen_in_pr[key] = (dur, str(td.get("flavor")), td.get("worker"), short_name)
    for key, (dur, flavor, worker, short_name) in seen_in_pr.items():
      if dur > heaviest[(flavor, worker)][0]:
        heaviest[(flavor, worker)] = (dur, short_name, pr_num)
      base_raw = ptb.get(key)
      if base_raw is None:
        if dur >= 30:
          new_heavy[(flavor, short_name)].append((dur, pr_num, worker))
      else:
        base_dur = float(base_raw)
        if base_dur >= 15 and dur >= base_dur * 1.35 and (dur - base_dur) >= 45:
          slowed[(flavor, short_name, base_dur)].append((dur, pr_num, worker))

  groups: list[tuple[str, list[str]]] = []
  slow_items = []
  for (flavor, short_name, base_dur), hits in slowed.items():
    max_dur, _, _ = max(hits, key=lambda item: item[0])
    diff = max_dur - base_dur
    pct = (diff / base_dur) * 100.0
    prs = sorted({h[1] for h in hits}, reverse=True)
    slow_items.append((diff, flavor, short_name, max_dur, base_dur, pct, prs))
  slow_lines = []
  for diff, flavor, short_name, max_dur, base_dur, pct, prs in sorted(slow_items, reverse=True)[:4]:
    pr_str = ", ".join(f"#{p}" for p in prs[:4]) + (f" +{len(prs) - 4} more" if len(prs) > 4 else "")
    slow_lines.append(
        f"`{flavor}` - `{short_name}`: up to {fmt_short_dur(max_dur)} vs. {fmt_short_dur(base_dur)} baseline "
        f"(+{fmt_short_dur(diff)} / +{pct:.0f}% on {pr_str})"
    )
  if slow_lines:
    groups.append(("Slowed Tests (vs. per_test_baseline.json)", slow_lines))

  new_items = []
  for (flavor, short_name), hits in new_heavy.items():
    max_dur, _, max_w = max(hits, key=lambda item: item[0])
    prs = sorted({h[1] for h in hits}, reverse=True)
    new_items.append((max_dur, flavor, max_w, short_name, prs))
  new_lines = [
      f"`{flavor}` (worker {max_w}) - `{short_name}`: {fmt_short_dur(max_dur)} (on {', '.join(f'#{p}' for p in prs[:3])})"
      for max_dur, flavor, max_w, short_name, prs in sorted(new_items, reverse=True)[:3]
  ]
  if new_lines:
    groups.append(("New Heavy Tests (>=30s, not yet in baseline)", new_lines))
  return groups, heaviest


def _shard_imbalance_lines(
    j_rows: Sequence[Mapping[str, Any]],
    recent_prs: set[int],
    heaviest: Mapping[tuple[str, Any], tuple[float, str, int | None]],
) -> list[str]:
  """Finds worker shards that run much longer than their sibling workers."""
  by_run_flavor: dict[tuple[Any, Any, str], list[tuple[Any, float]]] = defaultdict(list)
  for job_row in j_rows:
    if (
        job_row.get("pr") in recent_prs
        and job_row.get("worker") is not None
        and job_row.get("run_seconds") is not None
        and job_row.get("conclusion") == "success"
    ):
      by_run_flavor[(job_row["run_id"], job_row["pr"], str(job_row["flavor"]))].append(
          (job_row["worker"], float(job_row["run_seconds"]))
      )

  flavor_gaps: dict[tuple[str, Any], list[tuple[float, float, float, int]]] = defaultdict(list)
  for (_, _, flavor), workers in by_run_flavor.items():
    if len(workers) >= 3:
      ordered = sorted(workers, key=lambda item: item[1], reverse=True)
      max_w, max_s = ordered[0]
      others_avg = sum(item[1] for item in ordered[1:]) / (len(ordered) - 1)
      if max_s >= 600 and (max_s - others_avg) >= 300:
        flavor_gaps[(flavor, max_w)].append((max_s - others_avg, max_s, others_avg, len(ordered)))

  lines = []
  ranked = sorted(flavor_gaps.items(), key=lambda item: statistics.median([h[0] for h in item[1]]), reverse=True)
  for (flavor, max_w), hits in ranked[:3]:
    med_gap = statistics.median([h[0] for h in hits])
    med_max = statistics.median([h[1] for h in hits])
    med_oth = statistics.median([h[2] for h in hits])
    n_workers = max(h[3] for h in hits)
    top_dur, top_test, _ = heaviest.get((flavor, max_w), (0.0, "", None))
    top_note = f"; heaviest test: `{top_test}` at {fmt_short_dur(top_dur)}" if top_dur >= 180 else ""
    lines.append(
        f"`{flavor}` ({n_workers} workers): worker {max_w} median {med_max / 60:.1f}m vs. others {med_oth / 60:.1f}m "
        f"(+{med_gap / 60:.1f}m straggler gap across {len(hits)} run(s){top_note})"
    )
  return lines


QUEUE_LONG_WAIT_SECONDS = 300.0
QUEUE_WINDOW = timedelta(hours=48)
QUEUE_BASELINE = timedelta(days=7)
QUEUE_TEST_LANES = (ci_derive.LANE_TPU, ci_derive.LANE_GPU, ci_derive.LANE_CPU)
# A label missing from derive.RUNNER_LABEL_LANES is hardware nobody announced: show it rather than drop it.
QUEUE_UNLISTED_LANE = "unlisted label"
QUEUE_MIN_JOBS_SHOWN = 5
QUEUE_MIN_JOBS_FOR_TREND = 20
# A pool is flagged when its share of long waits, in the whole percentages the email prints, is this many
# points above the 7 days before. Backtested on the 96 emails of Oct 10 - Nov 11: it fired mostly during
# real congestion (Oct 13-17, Oct 24-25, Nov 4-6), never on TPU7x, and 3 times on the 8-chip pool.
QUEUE_WORSE_POINTS = 10
QUEUE_LAG_NOTE_SECONDS = 3600
QUEUE_WINDOW_TEXT = f"{QUEUE_WINDOW.total_seconds() / 3600:.0f}h"
QUEUE_BASELINE_TEXT = f"{QUEUE_BASELINE.days} days before"
QUEUE_LONG_TEXT = f"{QUEUE_LONG_WAIT_SECONDS / 60:.0f}+ min"
QUEUE_HOW_TO_READ = (
    "Wait = time from a job being ready to a runner starting it. Counts every run of the CI workflows: open and "
    "merged PRs, scheduled, manual, cancelled and re-run. Runs still in progress at the last update are counted "
    "next time; jobs cancelled before getting a runner are not counted."
)
QUEUE_HOW_TO_FLAG = (
    f"WORSE = the share of {QUEUE_LONG_TEXT} waits is {QUEUE_WORSE_POINTS}+ points above the {QUEUE_BASELINE_TEXT}, "
    f"with {QUEUE_MIN_JOBS_FOR_TREND}+ jobs in both periods."
)
_RUN_KINDS = {
    "schedule": "a scheduled run",
    "workflow_dispatch": "a manual run",
    "push": "a push run",
    "pull_request": "a PR run",
}


@dataclass(frozen=True)
class _JobWait:
  """One job that held a runner on a test pool: when it became ready and how long it waited."""

  pool: str
  lane: str
  ready: datetime
  seconds: float
  job_id: int
  run_id: int


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
  """Returns the nearest-rank percentile of a non-empty list."""
  ordered = sorted(values)
  return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _share_long(seconds: Sequence[float]) -> float:
  """Returns the share of a non-empty list of waits that lasted 5 minutes or more."""
  return sum(value >= QUEUE_LONG_WAIT_SECONDS for value in seconds) / len(seconds)


def _months_between(first: datetime, last: datetime) -> list[str]:
  """Lists the `YYYY-MM` months from `first` to `last`, both included."""
  months, year, month = [], first.year, first.month
  while (year, month) <= (last.year, last.month):
    months.append(f"{year:04d}-{month:02d}")
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
  return months


def _run_labels(runs: Sequence[Mapping[str, Any]]) -> dict[int, str]:
  """Names each run: its PR and what the rows know of that PR's state, or the kind of run.

  The state is decided per PR across all of its runs. The collector writes a merge correction for one run
  of a PR per tick, and none for a PR closed without merging, so a single run row is often out of date.
  """
  by_pr: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
  for run in runs:
    if run.get("pr_number") is not None:
      by_pr[int(run["pr_number"])].append(run)
  states = {}
  for number, pr_runs in by_pr.items():
    if any(run.get("pr_merged_at") for run in pr_runs):
      states[number] = "merged PR"
    else:
      freshest = max(pr_runs, key=lambda run: str(run.get("collected_at") or ""))
      states[number] = "closed PR" if freshest.get("pr_state") == "closed" else "not merged"
  labels = {}
  for run in runs:
    number, event = run.get("pr_number"), run.get("event")
    if number is not None:
      labels[int(run["run_id"])] = f"#{number}, {states[int(number)]}"
    else:
      labels[int(run["run_id"])] = _RUN_KINDS.get(event, f"a {event} run") if event else "an unknown run"
  return labels


def _read_job_waits(store_dir: Path, months: Sequence[str], since: datetime) -> tuple[list[_JobWait], dict[int, str]]:
  """Reads the jobs that became ready since `since` and held a runner on a test pool, and labels their runs.

  Works on the stored payloads (the store applies its correction rule) and filters before anything else:
  two months of job rows hold about 150k jobs, most of them out of range or not on a test pool.
  """
  data = ci_store.Store(store_dir)
  newest: dict[int, Mapping[str, Any]] = {}
  for run in data.read(ci_rows.KIND_RUN, sorted(set(months) & set(data.months(ci_rows.KIND_RUN)))):
    run_id = int(run.get("run_id") or 0)
    if run_id not in newest or int(run.get("attempt") or 0) > int(newest[run_id].get("attempt") or 0):
      newest[run_id] = run
  waits = []
  for job in data.read(ci_rows.KIND_JOB, sorted(set(months) & set(data.months(ci_rows.KIND_JOB)))):
    ready = ci_derive.parse_timestamp(job.get("created_at"))
    pool = ci_views.pool_label(job)
    if ready is None or ready < since or pool is None:
      continue
    lane = ci_derive.RUNNER_LABEL_LANES.get(pool, QUEUE_UNLISTED_LANE)
    if lane not in QUEUE_TEST_LANES and lane != QUEUE_UNLISTED_LANE:
      continue
    seconds = ci_derive.queue_seconds(job)
    if seconds is not None:
      waits.append(_JobWait(pool, lane, ready, seconds, int(job.get("job_id") or 0), int(job.get("run_id") or 0)))
  return waits, _run_labels(list(newest.values()))


def _queue_window_end(store_dir: Path, generated: datetime) -> datetime:
  """Ends the window at the newest run the collector has listed, when that is earlier than this update."""
  state = _load_json_file(store_dir / ci_store.DATA_DIRNAME / ci_store.STATE_FILENAME)
  newest_run = ci_derive.parse_timestamp(state.get("watermark_created_at"))
  return min(generated, newest_run) if newest_run is not None else generated


def _pool_line(
    pool: str, items: Sequence[_JobWait], before: Sequence[float], labels: Mapping[int, str]
) -> tuple[bool, float, str]:
  """Builds one pool's watchdog line; returns `(worse, share, line)`."""
  seconds = [item.seconds for item in items]
  share = _share_long(seconds)
  percent = round(share * 100)
  base_percent = round(_share_long(before) * 100) if len(before) >= QUEUE_MIN_JOBS_SHOWN else None
  worse = (
      base_percent is not None
      and min(len(items), len(before)) >= QUEUE_MIN_JOBS_FOR_TREND
      and percent - base_percent >= QUEUE_WORSE_POINTS
  )
  longest = max(items, key=lambda item: (item.seconds, -item.job_id))
  base_text = "too few jobs" if base_percent is None else f"{base_percent}%"
  line = (
      f"{'WORSE: ' if worse else ''}`{pool}` ({items[0].lane}): {percent}% of {len(items):,} jobs waited "
      f"{QUEUE_LONG_TEXT} ({QUEUE_BASELINE_TEXT}: {base_text}) - p90 {_nearest_rank(seconds, 0.9) / 60:.1f}m - "
      f"longest {longest.seconds / 60:.1f}m ({labels.get(longest.run_id, 'an unknown run')})"
  )
  return worse, share, line


def _queue_watchdog(store_dir: Path, generated: datetime) -> tuple[str, list[str]] | None:
  """Summarizes, per test pool, how long the jobs of every CI run waited for a runner."""
  end = _queue_window_end(store_dir, generated)
  start = end - QUEUE_WINDOW
  since = start - QUEUE_BASELINE
  title = f"Runner Queue Watchdog (every CI run, {QUEUE_WINDOW_TEXT} to {end:%b %d, %H:%M} UTC)"
  try:
    # Job rows are filed under their run's month, so jobs of runs created before the first month read (9 to
    # about 40 days before the email) are missed. The tick re-lists only the last 2 days (14 for merges), so
    # in practice that is a rare late re-run.
    waits, labels = _read_job_waits(store_dir, _months_between(since, end), since)
  except (ci_store.StoreError, ValueError, OSError) as error:  # ValueError covers RowError and bad UTF-8.
    message = str(error).replace(str(Path(store_dir).resolve()), "the store")
    return title, [f"Could not read the job rows: {type(error).__name__}: {message[:200]}"]
  if not waits:
    return None

  current: dict[str, list[_JobWait]] = defaultdict(list)
  before: dict[str, list[float]] = defaultdict(list)
  for wait in waits:
    if start <= wait.ready <= end:
      current[wait.pool].append(wait)
    elif wait.ready < start:
      before[wait.pool].append(wait.seconds)

  ranked = []
  for pool, items in current.items():
    if len(items) >= QUEUE_MIN_JOBS_SHOWN:
      worse, share, line = _pool_line(pool, items, before[pool], labels)
      ranked.append((not worse, -share, pool, line))
  lines = [line for *_, line in sorted(ranked)] or [f"No test pool ran {QUEUE_MIN_JOBS_SHOWN}+ jobs in this window."]
  lag = (generated - end).total_seconds()
  if lag > QUEUE_LAG_NOTE_SECONDS:
    lines.append(
        f"Newest CI run read: {end:%b %d, %H:%M} UTC, {fmt_duration(lag)} before this update; the window ends there."
    )
  lines += [QUEUE_HOW_TO_READ, QUEUE_HOW_TO_FLAG]
  return title, lines


def _health_and_bench_lines(
    store_dir: Path,
    month: str,
    cutoff_48h: datetime,
    recent_prs: set[int],
    raw_bench: str | None,
) -> list[str]:
  """Summarizes flaky retries, cross-PR failing tests, and Benchmarks suite shifts."""
  lines: list[str] = []
  flaky_doc = _load_json_file(store_dir / f"views/flaky-{month}.json")
  recent_rc = []
  for row in _table_rows(flaky_doc, "rescues"):
    ts = parse_time(row.get("created_at"))
    if ts is not None and ts >= cutoff_48h and row.get("job_name") != "All Required Tests Passed":
      recent_rc.append(row)
  rescued = [r for r in recent_rc if r.get("rescued")]
  unrescued = [r for r in recent_rc if not r.get("rescued")]
  if rescued or unrescued:
    r_waste = sum(float(r.get("wasted_seconds") or 0.0) for r in rescued)
    u_waste = sum(float(r.get("wasted_seconds") or 0.0) for r in unrescued)
    lines.append(
        f"Flaky & failed job retries (48h): {len(rescued)} job(s) passed on retry (wasted {r_waste / 60:.1f}m) - "
        f"{len(unrescued)} job failure(s) stayed red (wasted {u_waste / 60:.1f}m)"
    )

  by_fail_test: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
  for row in _table_rows(flaky_doc, "rescue_tests"):
    if row.get("pr") in recent_prs:
      short_c = str(row.get("classname") or "").rsplit(".", 1)[-1]
      by_fail_test[(str(row.get("flavor")), f"{short_c}.{row.get('name')}")].append(
          (int(row["pr"]), str(row.get("failure_message") or ""))
      )
  for (flavor, tname), hits in sorted(by_fail_test.items(), key=lambda item: len(item[1]), reverse=True)[:2]:
    prs = sorted({h[0] for h in hits}, reverse=True)
    if len(prs) >= 2:
      msg = (hits[-1][1] or "").splitlines()[0][:85]
      pr_str = ", ".join(f"#{p}" for p in prs[:4]) + (f" +{len(prs) - 4} more" if len(prs) > 4 else "")
      lines.append(f"Cross-PR failing test: `{flavor}` - `{tname}` failed on {len(prs)} PRs ({pr_str}) — {msg}")

  if raw_bench and "window.BENCHMARK_DATA = " in raw_bench:
    try:
      bench_data = json.loads(raw_bench.split("window.BENCHMARK_DATA = ", 1)[1].rstrip(";\n "))
      entries = (bench_data.get("entries") or {}).get("MaxText Test Execution Times") or []
      series: dict[str, list[tuple[Any, str, float]]] = defaultdict(list)
      for entry in entries:
        commit = entry.get("commit") or {}
        for bench in entry.get("benches") or []:
          series[str(bench["name"])].append(
              (commit.get("timestamp"), str(commit.get("message") or "").splitlines()[0], float(bench["value"]))
          )
      suite_names = sorted(
          {key.replace("Total ", "").replace(" Tests Duration", "").replace(" Tests Count", "") for key in series}
      )
      for sname in suite_names:
        d_pts = series.get(f"Total {sname} Tests Duration", [])
        c_pts = series.get(f"Total {sname} Tests Count", [])
        if len(d_pts) >= 5 and len(c_pts) >= 5:
          ts_raw, msg, d_val = d_pts[-1]
          ts = parse_time(ts_raw)
          if ts is not None and ts < cutoff_48h:
            continue
          _, _, c_val = c_pts[-1]
          d_base = statistics.median([x[2] for x in d_pts[-15:-1]])
          c_base = int(round(statistics.median([x[2] for x in c_pts[-15:-1]])))
          c_diff = int(round(c_val)) - c_base
          d_diff = d_val - d_base
          d_pct = (d_diff / d_base * 100.0) if d_base > 0 else 0.0
          if abs(c_diff) >= 2 or (abs(d_diff) >= 60 and abs(d_pct) >= 10):
            pr_m = re.search(r"#(\d+)", msg)
            pr_ref = f" (at #{pr_m.group(1)})" if pr_m else ""
            sign = "+" if d_diff >= 0 else "-"
            lines.append(
                f"Benchmarks shift (`{sname}`): {int(c_val)} tests ({c_diff:+d} vs. {c_base}) - "
                f"total duration {fmt_short_dur(d_val)} ({sign}{fmt_short_dur(abs(d_diff))} / "
                f"{d_pct:+.1f}% vs. {fmt_short_dur(d_base)} median){pr_ref}"
            )
    except (ValueError, KeyError, TypeError):
      pass
  return lines


def build_ci_insights(
    store_dir: Path,
    new_meta: Mapping[str, Any],
    fetcher: Callable[[str], str | None] = _fetch_text,
) -> list[tuple[str, list[str]]]:
  """Analyzes CI Pulse views and Benchmarks data for regressions and bottlenecks."""
  now_dt = parse_time(new_meta.get("generated_at"))
  if now_dt is None:
    return []
  cutoff_48h = now_dt - timedelta(hours=48)
  month = str(new_meta.get("generated_at"))[:7]

  raw_ptb = fetcher(f"{BENCH_BASE_URL}/per_test_baseline.json")
  try:
    ptb = json.loads(raw_ptb) if raw_ptb else {}
  except ValueError:
    ptb = {}

  runs_doc = _load_json_file(store_dir / f"views/runs-{month}.json")
  r_rows = _table_rows(runs_doc, "runs")
  j_rows = _table_rows(runs_doc, "jobs")

  full_runs = [r for r in r_rows if (r.get("jobs_with_tests") or 0) > 0 and r.get("pr")]
  full_runs.sort(key=lambda r: str(r.get("merged_at") or r.get("created_at") or ""))
  recent_full = [r for r in full_runs if (parse_time(r.get("merged_at") or r.get("created_at")) or now_dt) >= cutoff_48h]
  if len(recent_full) < 5:
    recent_full = full_runs[-8:]
  recent_prs = sorted({int(r["pr"]) for r in recent_full if r.get("pr")})
  recent_pr_set = set(recent_prs)

  groups, heaviest = _slow_and_new_test_insights(store_dir, recent_prs, ptb if isinstance(ptb, dict) else {})
  shard_lines = _shard_imbalance_lines(j_rows, recent_pr_set, heaviest)
  if shard_lines:
    groups.append(("Worker Shard Imbalance (slowest worker sets suite wall time)", shard_lines))

  queue_group = _queue_watchdog(store_dir, now_dt)
  if queue_group:
    groups.append(queue_group)

  raw_bench = fetcher(f"{BENCH_BASE_URL}/data.js")
  health_lines = _health_and_bench_lines(store_dir, month, cutoff_48h, recent_pr_set, raw_bench)
  if health_lines:
    groups.append(("Failures, Flaky Rescues & Benchmark Suite Shifts", health_lines))
  return groups


def _render_html(headline: str, sections: Sequence[Mapping[str, Any]], footer: str) -> str:
  """Renders the report sections into an HTML email body."""
  parts = [
      '<div style="font-family: Arial, sans-serif; font-size: 14px; color: #202124;">',
      f'<p style="font-size: 16px;"><b>{html.escape(headline)}</b></p>',
  ]
  for sec in sections:
    if sec.get("title"):
      parts.append(f'<p style="margin: 16px 0 6px;"><b>{html.escape(str(sec["title"]))}</b></p>')
    kind = sec.get("kind")
    if kind == "kv":
      rows = "".join(
          f'<tr><td style="{CELL} background: #f8f9fa; font-weight: bold;">{html.escape(key)}</td>'
          f'<td style="{CELL}">{value}</td></tr>'
          for key, _, value in sec["rows"]
      )
      parts.append(f"<table {TABLE}>{rows}</table>")
    elif kind == "changes":
      head = "".join(
          f'<th style="{CELL} background: #f8f9fa; text-align: {align};">{name}</th>'
          for name, align in (("What", "left"), ("Before", "right"), ("After", "right"), ("Change", "right"))
      )
      rows = "".join(
          f'<tr><td style="{CELL}">{html.escape(label)}</td>'
          f'<td style="{CELL} text-align: right;">{b:,}</td>'
          f'<td style="{CELL} text-align: right;">{a:,}</td>'
          f'<td style="{CELL} text-align: right;">{a - b:+,}</td></tr>'
          for label, b, a in sec["rows"]
      )
      parts.append(f"<table {TABLE}><tr>{head}</tr>{rows}</table>")
    elif kind == "insights":
      blocks = []
      for gtitle, items in sec["rows"]:
        lis = "".join(f'<li style="margin: 3px 0;">{_rich_html_line(item)}</li>' for item in items)
        blocks.append(
            f'<div style="margin: 8px 0 10px;">'
            f'<div style="font-weight: bold; color: #3c4043; margin-bottom: 3px;">{html.escape(gtitle)}</div>'
            f'<ul style="margin: 0; padding-left: 20px; line-height: 1.45;">{lis}</ul></div>'
        )
      parts.append(
          '<div style="border: 1px solid #dadce0; background: #fcfcfc; padding: 8px 14px; border-radius: 4px;">'
          + "".join(blocks)
          + "</div>"
      )
    elif kind == "pre":
      parts.append(
          '<pre style="background: #f8f9fa; border: 1px solid #dadce0; padding: 8px; '
          f'font-size: 12px; white-space: pre-wrap;">{html.escape(str(sec["rows"]))}</pre>'
      )
    else:
      parts.append(f'<p style="margin: 0;">{sec.get("html") or html.escape(str(sec.get("rows") or ""))}</p>')
  parts.append(f'<p style="color: #5f6368; font-size: 12px; margin-top: 20px;">{html.escape(footer)}</p></div>')
  return "".join(parts)


def build_report_email(
    store_dir: Path,
    old_meta: Mapping[str, Any] | None,
    new_meta: Mapping[str, Any] | None,
    facts: Mapping[str, Any],
    fetcher: Callable[[str], str | None] = _fetch_text,
) -> tuple[str, str]:
  """Builds the `(subject, html_body)` for one Cloud Run execution."""
  prefix = "[MaxText CI Pulse]"
  footer = (
      "Sent by the MaxText CI Pulse Cloud Run collector (mesa-ci-pulse-collector in Google Cloud). "
      "This digest goes out about every 8 hours; a failed collector run sends an alert right away."
  )
  code = int(facts.get("exit_code") or 0)
  elapsed = fmt_duration(float(facts.get("seconds") or 0.0))
  execution = str(facts.get("execution") or "local")
  bucket = str(facts.get("bucket") or "mesa-ci-pulse")

  if code == 0 and new_meta:
    added = _new_prs(old_meta, new_meta)
    when = fmt_time(new_meta.get("generated_at"))
    before = fmt_time((old_meta or {}).get("generated_at"))
    pr_text = (
        "no new merged PRs" if not added else ("1 new merged PR" if len(added) == 1 else f"{len(added)} new merged PRs")
    )
    subject = f"{prefix} Dashboard updated: data to {when}, {pr_text}"
    headline = "The MaxText CI Pulse dataset was updated on Google Cloud."
    sections: list[dict[str, Any]] = [
        {
            "title": "",
            "kind": "kv",
            "rows": [
                _kv("Repository", "AI-Hypercomputer/maxtext", _link(REPO_URL, "AI-Hypercomputer/maxtext")),
                _kv("Live page", LIVE_URL, _link(LIVE_URL, LIVE_URL)),
                _kv("Data collected", f"{before} -> {when}"),
                _kv("Runs not read yet", str(new_meta.get("uncollected_runs", 0))),
            ],
        }
    ]
    sections.append(
        {"title": "What changed (before -> after)", "kind": "changes", "rows": _change_rows(old_meta, new_meta)}
    )
    if added:
      shown = added[:40]
      more = f" and {len(added) - 40} more" if len(added) > 40 else ""
      sections.append(
          {
              "title": f"New merged PRs ({len(added)})",
              "kind": "text",
              "rows": ", ".join(f"#{n}" for n in shown) + more,
              "html": ", ".join(_link(f"{REPO_URL}/pull/{n}", f"#{n}") for n in shown) + html.escape(more),
          }
      )
    insights = build_ci_insights(store_dir, new_meta, fetcher=fetcher)
    if insights:
      sections.append(
          {
              "title": "CI Health & Test Duration Insights (last 48h vs. baseline)",
              "kind": "insights",
              "rows": insights,
          }
      )
    uploaded = facts.get("uploaded")
    up_text = (
        f"gs://{bucket} ({uploaded.get('files', 0)} changed, {uploaded.get('unchanged', 0)} unchanged)"
        if isinstance(uploaded, Mapping)
        else f"gs://{bucket} ({uploaded})"
    )
    sections.append(
        {
            "title": "Publish & Run",
            "kind": "kv",
            "rows": [
                _kv("GCS store", up_text),
                _kv("GitHub Pages publisher", ".github/workflows/ci_pulse_publish.yml (runs every 4h at :40 UTC)"),
                _kv("Execution", f"{execution} (took {elapsed})"),
                _kv("Collector", str(facts.get("tick_summary") or "ok")),
                _kv("API budget", str(facts.get("tick_budget") or "n/a")),
            ],
        }
    )
  else:
    reason = str(facts.get("error") or facts.get("tick_summary") or f"exit code {code}")
    subject = f"{prefix} ALERT: Cloud Run collector exited with code {code}"
    headline = "The scheduled MaxText CI Pulse Cloud Run collector reported an error."
    sections = [
        {
            "title": "",
            "kind": "kv",
            "rows": [
                _kv("Repository", "AI-Hypercomputer/maxtext", _link(REPO_URL, "AI-Hypercomputer/maxtext")),
                _kv("Live page", LIVE_URL, _link(LIVE_URL, LIVE_URL)),
                _kv("Execution", f"{execution} (failed after {elapsed})"),
                _kv("Exit code", str(code)),
                _kv("Reason", reason),
                _kv("Collector", str(facts.get("tick_summary") or "none")),
            ],
        }
    ]
    problems = facts.get("tick_problems") or []
    if problems:
      sections.append(
          {"title": "Problems reported by the collector", "kind": "pre", "rows": "\n".join(str(p) for p in problems)}
      )

  return subject, _render_html(headline, sections, footer)


def _metadata_access_token(timeout: float = 10.0) -> str:
  """Reads the Cloud Run service account's access token from the GCE metadata server."""
  req = urllib.request.Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    data = json.loads(resp.read().decode("utf-8"))
  return str(data["access_token"])


def _impersonated_token(source_token: str, target_sa: str, timeout: float = 10.0) -> str:
  """Exchanges the ambient token for a short-lived token of `target_sa` via IAM Credentials."""
  url = IAM_CREDENTIALS_URL.format(sa=target_sa)
  payload = json.dumps({"scope": ["https://www.googleapis.com/auth/cloud-platform"]}).encode("utf-8")
  req = urllib.request.Request(
      url,
      data=payload,
      method="POST",
      headers={"Authorization": f"Bearer {source_token}", "Content-Type": "application/json"},
  )
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    data = json.loads(resp.read().decode("utf-8"))
  return str(data["accessToken"])


def send_via_integration(
    settings: EmailSettings,
    subject: str,
    html_body: str,
    token_getter: Callable[[EmailSettings], str] | None = None,
) -> str:
  """Calls GCP Application Integration `EmailTask` (`:execute`) and returns the execution ID."""
  if token_getter is not None:
    token = token_getter(settings)
  else:
    ambient = _metadata_access_token()
    token = _impersonated_token(ambient, settings.invoker_sa) if settings.invoker_sa else ambient

  url = INTEGRATIONS_EXECUTE_URL.format(integration=settings.integration)
  payload = {
      "triggerId": settings.trigger_id,
      "inputParameters": {
          "email_to": {"stringArray": {"stringValues": list(settings.recipients)}},
          "email_subject": {"stringValue": subject},
          "email_body": {"stringValue": html_body},
      },
  }
  req = urllib.request.Request(
      url,
      data=json.dumps(payload).encode("utf-8"),
      method="POST",
      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
  )
  with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read().decode("utf-8"))
  return str(result.get("executionId") or "ok")
