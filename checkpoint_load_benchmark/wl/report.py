# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builds the benchmark v3 workbook and the tables of the report from the CSV files of reports/v3/.

  report.py --csv-dir reports/v3 --xlsx OUT.xlsx --markdown OUT.md

The CSV files (written by collect.py) are the source of truth: this script only arranges them, so the workbook and
the report can be rebuilt at any time. The workbook has one row per configuration on every tab, no merged cells,
and frozen, filterable headers. It needs openpyxl.
"""

import argparse
import collections
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wl import common  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

CONFIG = ["platform", "model", "topology", "slices", "hosts", "chips", "scan_mode"]
# A change counts as faster or slower only beyond both of these.
CHANGE_PCT, CHANGE_S = 10.0, 10
STATUS_FILL = {"OK": "E2F0D9", "NOT RUN": "E7E6E6", "": None}
OTHER_STATUS_FILL = "F8CBAD"


def _read(csv_dir: str, name: str) -> list[dict]:
  path = os.path.join(csv_dir, name)
  if not os.path.exists(path):
    return []
  with open(path, "r", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def _num(value):
  """Returns `value` as an int or float if it is a number, else None."""
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return int(number) if number == int(number) else number


def _sort_key(sizes: dict[str, float]):
  order = {p: i for i, p in enumerate(common.PLATFORMS)}
  return lambda r: (
      order.get(r["platform"], 9),
      sizes.get(r["model"], 0.0),
      r["model"],
      _num(r["slices"]) or 0,
      _num(r.get("hosts")) or 0,
      r.get("scan_mode", ""),
      r.get("build", ""),
  )


def stage_changes(stages: list[dict]) -> dict[tuple, dict]:
  """Returns, per configuration, how much the read, reshard and other transform time changed from baseline to candidate."""
  values = collections.defaultdict(dict)
  for s in stages:
    if s["phase"] == "safetensors_dynamic":
      key = (s["platform"], s["model"], s["topology"], s["slices"], s["scan_mode"])
      values[(key, s["build"])][s["stage"]] = float(s["slowest_host_s"])
  out = {}
  for (key, build), base in values.items():
    cand = values.get((key, "candidate"))
    if build != "baseline" or cand is None:
      continue

    def delta(stage, base=base, cand=cand):
      return cand.get(stage, 0.0) - base.get(stage, 0.0)

    read = delta("step2_read_from_gcs_to_host_ram")
    reshard = delta("step2_reshard_to_target_layout")
    other = delta("step2_transform_total") - reshard
    parts = {"GCS read": read, "reshard": reshard, "other transform": other}
    biggest = max(parts, key=lambda k, parts=parts: abs(parts[k]))
    out[key] = {
        "read_change_s": round(read),
        "reshard_change_s": round(reshard),
        "other_transform_change_s": round(other),
        "main_change": biggest if abs(parts[biggest]) >= 5 else "",
    }
  return out


def before_after_rows(results: list[dict], stages: list[dict], planned: list[dict]) -> list[dict]:
  """Returns one row per configuration that has a baseline row planned."""
  groups = {
      (p["platform"], p["model"], p["topology"], p["slices"], p["scan_mode"]): p["fix_group"]
      for p in planned
      if p["build"] == "baseline"
  }
  changes = stage_changes(stages)
  rows = []
  for r in results:
    key = (r["platform"], r["model"], r["topology"], r["slices"], r["scan_mode"])
    if key not in groups:
      continue
    base_s, cand_s = _num(r["sd_baseline_total_s"]), _num(r["sd_candidate_total_s"])
    verdict = ""
    if r["sd_baseline_status"] not in ("OK", "") and r["sd_candidate_status"] == "OK":
      verdict = "fixed"
    elif r["sd_baseline_status"] == "OK" and r["sd_candidate_status"] not in ("OK", ""):
      verdict = "BROKEN"
    elif base_s is not None and cand_s is not None:
      diff, pct = cand_s - base_s, _num(r["sd_change_pct"]) or 0.0
      verdict = (
          "faster"
          if diff <= -CHANGE_S and pct <= -CHANGE_PCT
          else ("SLOWER" if diff >= CHANGE_S and pct >= CHANGE_PCT else "no change")
      )
    row = {k: r[k] for k in CONFIG}
    row.update(
        {
            "fix_group": groups[key],
            "baseline_status": r["sd_baseline_status"],
            "baseline_s": r["sd_baseline_total_s"],
            "candidate_status": r["sd_candidate_status"],
            "candidate_s": r["sd_candidate_total_s"],
            "change_s": r["sd_change_s"],
            "change_pct": r["sd_change_pct"],
            "verdict": verdict,
            **changes.get(
                key, {"read_change_s": "", "reshard_change_s": "", "other_transform_change_s": "", "main_change": ""}
            ),
            "valid": r["valid"],
            "note": r["note"] or r["expected_outcome"],
        }
    )
    rows.append(row)
  return rows


def pipeline_rows(results: list[dict]) -> list[dict]:
  """Returns one row per configuration: `safetensors_dynamic` (candidate) against `to_maxtext`."""
  rows = []
  for r in results:
    row = {k: r[k] for k in CONFIG}
    sd_s, load_s = _num(r["sd_candidate_total_s"]), _num(r["tm_step3_load_s"])
    row.update(
        {
            "sd_status": r["sd_candidate_status"],
            "sd_load_s": r["sd_candidate_total_s"],
            "tm_status": r["tm_status"],
            "tm_convert_s": r["tm_step2_convert_s"],
            "tm_load_s": r["tm_step3_load_s"],
            "tm_convert_plus_load_s": r["tm_total_s"],
            "tm_over_sd": r["tm_over_sd_speedup"],
            "tm_load_over_sd": round(load_s / sd_s, 2) if sd_s and load_s is not None else "",
            "sd_peak_ram_gb": r["sd_candidate_peak_ram_gb"],
            "sd_peak_hbm_gb": r["sd_candidate_peak_hbm_gb"],
            "note": r["note"] or r["expected_outcome"],
        }
    )
    rows.append(row)
  return rows


def stage_rows(stages: list[dict]) -> tuple[list[dict], list[str]]:
  """Returns the stage times pivoted to one row per configuration, build and pipeline, and the column order."""
  rows, columns = {}, []
  for s in stages:
    key = (s["platform"], s["model"], s["topology"], s["slices"], s["scan_mode"], s["build"], s["pipeline"])
    row = rows.setdefault(
        key, dict(zip(["platform", "model", "topology", "slices", "scan_mode", "build", "pipeline"], key))
    )
    column = (
        f"{s['phase'].removeprefix('to_maxtext_')}:{s['stage']}_s" if s["pipeline"] == "to_maxtext" else f"{s['stage']}_s"
    )
    row[column] = round(float(s["slowest_host_s"]), 1)
    if column not in columns:
      columns.append(column)
  head = ["platform", "model", "topology", "slices", "scan_mode", "build", "pipeline"]
  return list(rows.values()), head + sorted(columns)


def summary_tables(results: list[dict], before_after: list[dict]) -> dict[str, tuple[list[str], list[list]]]:
  """Returns the headline tables as `title -> (header, rows)`."""
  tables = {}
  header = ["platform", "cell", "planned", "OK", "failed as expected", "failed", "not run yet"]
  rows = []
  cells = (
      ("safetensors_dynamic baseline", "sd_baseline_status"),
      ("safetensors_dynamic candidate", "sd_candidate_status"),
      ("to_maxtext", "tm_status"),
  )
  for platform in common.PLATFORMS:
    for label, column in cells:
      mine = [r for r in results if r["platform"] == platform]
      if column == "sd_baseline_status":
        keys = {
            (b["model"], b["topology"], b["slices"], b["scan_mode"]) for b in before_after if b["platform"] == platform
        }
        mine = [r for r in mine if (r["model"], r["topology"], r["slices"], r["scan_mode"]) in keys]
      ok = sum(1 for r in mine if r[column] == "OK")
      missing = sum(1 for r in mine if r[column] == "")
      failed = [r for r in mine if r[column] not in ("OK", "")]
      expected = sum(
          1 for r in failed if r["expected_outcome"] and any(w in r["expected_outcome"] for w in (r[column], "NOT RUN"))
      )
      rows.append([platform, label, len(mine), ok, expected, len(failed) - expected, missing])
  tables["Rows by outcome"] = (header, rows)

  header = [
      "fix group",
      "rows compared",
      "median change %",
      "best %",
      "worst %",
      "faster",
      "no change",
      "slower",
      "fixed",
      "broken",
  ]
  rows = []
  for group in sorted({b["fix_group"] for b in before_after}):
    mine = [b for b in before_after if b["fix_group"] == group]
    pcts = [_num(b["change_pct"]) for b in mine if _num(b["change_pct"]) is not None]
    count = collections.Counter(b["verdict"] for b in mine)
    rows.append(
        [
            group,
            len(pcts),
            round(statistics.median(pcts), 1) if pcts else "",
            min(pcts) if pcts else "",
            max(pcts) if pcts else "",
            count["faster"],
            count["no change"],
            count["SLOWER"],
            count["fixed"],
            count["BROKEN"],
        ]
    )
  tables["Before and after by fix group (safetensors_dynamic, candidate against baseline)"] = (header, rows)

  header = [
      "platform",
      "rows with both pipelines",
      "median (convert + load) / safetensors_dynamic",
      "median load only / safetensors_dynamic",
  ]
  rows = []
  for platform in common.PLATFORMS:
    mine = [r for r in results if r["platform"] == platform and _num(r["tm_over_sd_speedup"]) is not None]
    load = [_num(r["tm_step3_load_s"]) / _num(r["sd_candidate_total_s"]) for r in mine if _num(r["sd_candidate_total_s"])]
    rows.append(
        [
            platform,
            len(mine),
            round(statistics.median(_num(r["tm_over_sd_speedup"]) for r in mine), 2) if mine else "",
            round(statistics.median(load), 2) if load else "",
        ]
    )
  tables[
      "Pipelines (to_maxtext time divided by safetensors_dynamic time; above 1 means safetensors_dynamic is faster)"
  ] = (header, rows)
  return tables


def readme_lines(builds: dict) -> list[str]:
  """Returns the text of the Read me tab."""
  b = builds.get("builds", {})
  return [
      "Checkpoint-load benchmark v3: safetensors_dynamic against to_maxtext, and the effect of PRs #6513 and #6519.",
      "",
      "What was measured: the time to load each model's weights into a sharded MaxText state on TPU, from GCS,",
      "with a cold JAX compilation cache.",
      "  safetensors_dynamic: read the Hugging Face .safetensors files and convert on the TPU devices (one step).",
      "  to_maxtext: convert on CPU to an Orbax checkpoint (Step 2), then restore it on TPU (Step 3).",
      "",
      f"Builds: both on main {builds.get('pinned_main', '')[:9]} plus PRs #6353 and #6354.",
      f"  baseline  {b.get('baseline', {}).get('commit', '')[:9]}  ({b.get('baseline', {}).get('branch', '')})",
      f"  candidate {b.get('candidate', {}).get('commit', '')[:9]}  ({b.get('candidate', {}).get('branch', '')})",
      "            = baseline + #6513 + #6519",
      f"Image: {builds.get('container_image_digest') or builds.get('container_image', '')}",
      "",
      "Conditions: every task ran in new pods; every phase had an empty JAX compilation cache and HF cache;",
      "baseline and candidate of a configuration ran back to back on the same cluster; all runs read the same weights;",
      "checksums of every loaded leaf were compared.",
      "A row is valid when its cache evidence, code version, weight source, stage-time sum and checksums all pass.",
      "",
      "Tabs:",
      "  Summary           headline tables.",
      "  Before and after  one row per configuration that ran on both builds.",
      "                    verdict: faster and slower need both 10% and 10 s.",
      "  Pipelines         one row per configuration: safetensors_dynamic (candidate) against to_maxtext.",
      "  Stages            stage times (slowest host), one row per configuration, build and pipeline.",
      "  Failures          every failed or invalid phase and every interrupted attempt, with its cause.",
      "  Runs              every phase result with all raw fields.",
      "",
      "Status words: OK; OOM-HOST (host RAM); OOM-HBM (TPU memory);",
      "NOT RUN (the harness refused or an earlier step failed);",
      "TIMEOUT; PREEMPTED; ERROR; INVALID (ran, but failed a validity check). An empty status means not run yet.",
      "Times are whole seconds: the median of the valid repetitions (three for Tier A, one otherwise).",
  ]


def write_workbook(
    path: str, sheets: list[tuple[str, list[str], list[dict] | list[list]]], readme: list[str], summary: dict
) -> None:
  """Writes the workbook: Read me, Summary, then one filterable sheet per (title, columns, rows)."""
  import openpyxl  # pylint: disable=import-outside-toplevel
  from openpyxl.styles import Font, PatternFill  # pylint: disable=import-outside-toplevel
  from openpyxl.utils import get_column_letter  # pylint: disable=import-outside-toplevel

  bold = Font(bold=True)
  book = openpyxl.Workbook()
  sheet = book.active
  sheet.title = "Read me"
  for line in readme:
    sheet.append([line])
  sheet["A1"].font = bold
  sheet.column_dimensions["A"].width = 140

  sheet = book.create_sheet("Summary")
  for title, (header, rows) in summary.items():
    sheet.append([title])
    sheet.cell(row=sheet.max_row, column=1).font = bold
    sheet.append(header)
    for cell in sheet[sheet.max_row]:
      cell.font = bold
    for row in rows:
      sheet.append(row)
    sheet.append([])
  for i in range(1, 12):
    sheet.column_dimensions[get_column_letter(i)].width = 30 if i <= 2 else 16

  for title, columns, rows in sheets:
    sheet = book.create_sheet(title)
    sheet.append(columns)
    for cell in sheet[1]:
      cell.font = bold
    for row in rows:
      values = [row.get(c, "") for c in columns] if isinstance(row, dict) else row
      sheet.append([_num(v) if _num(v) is not None and not isinstance(v, bool) else v for v in values])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for i, column in enumerate(columns, 1):
      longest = max(
          [len(column)] + [len(str(r.get(column, "") if isinstance(r, dict) else r[i - 1])) for r in rows[:500]]
      )
      sheet.column_dimensions[get_column_letter(i)].width = min(60, max(8, longest + 2))
      if column.endswith("status"):
        for cell in sheet[get_column_letter(i)][1:]:
          fill = STATUS_FILL.get(cell.value or "", OTHER_STATUS_FILL)
          if fill:
            cell.fill = PatternFill("solid", fgColor=fill)
  book.save(path)


def markdown_table(header: list[str], rows: list[list]) -> str:
  lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
  lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
  return "\n".join(lines)


def write_markdown(
    path: str, summary: dict, before_after: list[dict], pipelines: list[dict], failures: list[dict]
) -> None:
  """Writes the tables of the report; the findings above them are written by hand."""
  out = ["# Checkpoint-load benchmark v3: tables", ""]
  for title, (header, rows) in summary.items():
    out += [f"## {title}", "", markdown_table(header, rows), ""]
  columns = [
      "platform",
      "model",
      "topology",
      "slices",
      "scan_mode",
      "baseline_status",
      "baseline_s",
      "candidate_status",
      "candidate_s",
  ]
  columns += ["change_s", "change_pct", "verdict", "main_change"]
  changed = [b for b in before_after if b["verdict"] not in ("no change", "")]
  out += ["## Configurations that changed", "", markdown_table(columns, [[b[c] for c in columns] for b in changed]), ""]
  columns = [
      "model",
      "topology",
      "slices",
      "scan_mode",
      "sd_status",
      "sd_load_s",
      "tm_status",
      "tm_convert_s",
      "tm_load_s",
      "tm_over_sd",
  ]
  for platform in common.PLATFORMS:
    mine = [p for p in pipelines if p["platform"] == platform and (p["sd_status"] or p["tm_status"])]
    out += [f"## Pipelines on {platform}", "", markdown_table(columns, [[p[c] for c in columns] for p in mine]), ""]
  columns = [
      "platform",
      "model",
      "topology",
      "slices",
      "scan_mode",
      "build",
      "phase",
      "attempt",
      "category",
      "expected",
      "cause",
  ]
  out += ["## Failures and retries", "", markdown_table(columns, [[f.get(c, "") for c in columns] for f in failures]), ""]
  with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--csv-dir", default=common.REPORT_DIR)
  parser.add_argument("--xlsx")
  parser.add_argument("--markdown")
  args = parser.parse_args()
  sizes = {m["model_name"]: float(m["loaded_bf16_gb"]) for m in schema_v2.load_models_v2()["models"]}
  key = _sort_key(sizes)
  results = sorted(_read(args.csv_dir, "results.csv"), key=key)
  stages = _read(args.csv_dir, "stages.csv")
  planned = _read(args.csv_dir, "planned_rows.csv")
  failures = sorted(_read(args.csv_dir, "failures.csv"), key=key)
  runs = sorted(_read(args.csv_dir, "runs.csv"), key=key)
  builds = {}
  if os.path.exists(os.path.join(args.csv_dir, "builds.json")):
    with open(os.path.join(args.csv_dir, "builds.json"), "r", encoding="utf-8") as f:
      builds = json.load(f)
  before_after = before_after_rows(results, stages, planned)
  pipelines = pipeline_rows(results)
  stage_table, stage_columns = stage_rows(stages)
  summary = summary_tables(results, before_after)
  if args.xlsx:
    sheets = [
        ("Before and after", list(before_after[0]) if before_after else CONFIG, before_after),
        ("Pipelines", list(pipelines[0]) if pipelines else CONFIG, pipelines),
        ("Stages", stage_columns, sorted(stage_table, key=key)),
        ("Failures", list(failures[0]) if failures else ["platform"], failures),
        ("Runs", list(runs[0]) if runs else ["platform"], runs),
    ]
    write_workbook(args.xlsx, sheets, readme_lines(builds), summary)
    print(f"wrote {args.xlsx}")
  if args.markdown:
    write_markdown(args.markdown, summary, before_after, pipelines, failures)
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
  main()
