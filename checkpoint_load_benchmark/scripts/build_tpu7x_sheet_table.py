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

"""Builds the >200B `to_maxtext` vs `safetensor` table for the "tpu-v7x" tab of the benchmark Google Sheet.

The table has the tab's 10 columns, with a `to_maxtext` row and a `safetensor` row per (model, scan mode, slices).
It combines:
  * the P1 runs on single-host 2x2x1 slices (`1 (8)`, `2 (16)`, `4 (32)`), read from the per-phase result JSONs
    (same local mirrors, override order and `--cold-s1` rule as build_tpu7x_tables.py), and
  * Lance Wang's runs on multi-host slices (`1 (16)` = 1 x 2x2x2, `1 (32)` = 1 x 2x2x4, `2 (32)` = 2 x 2x2x2,
    `4 (64)` = 4 x 2x2x2), `safetensors_dynamic` only, read from reports/tpu7x_multihost_results.csv.

Every cell is plain text (no bold, italics or code spans), so a copy of the rendered markdown table pastes into
Google Sheets as plain cells. The same lines are also written as tab-separated text in a fenced block: Sheets splits
pasted tab-separated text into cells.

Usage:
  B=gs://mesa-maxtext/mesakh_tpu7x_bench
  gcloud storage rsync -r $B/p1_lance_prs/run_tpu7x_200b/logs /tmp/t7/p1
  gcloud storage rsync -r $B/p1_lance_prs/run_tpu7x_200b_pstream/logs /tmp/t7/p1_pstream
  python3 checkpoint_load_benchmark/scripts/build_tpu7x_sheet_table.py /tmp/t7/p1 /tmp/t7/p1_pstream \\
      --cold-s1 qwen3.5-397b-a17b=814.53 \\
      --multihost-csv checkpoint_load_benchmark/reports/tpu7x_multihost_results.csv \\
      --md-into checkpoint_load_benchmark/reports/tpu7x_priority1_progress.md
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tpu7x_tables import Logs, build_rows  # pylint: disable=wrong-import-position

# Sheet order and the label shown next to the model name.
MODELS = {
    "qwen3-235b-a22b": "470.4 GB MoE",
    "deepseek4-284b": "581.9 GB MoE, synthetic sparse files",
    "qwen3.5-397b-a17b": "795.5 GB Hybrid MoE",
    "qwen3-480b-a35b": "961.0 GB MoE",
    "deepseek3-671b": "1,342.4 GB MoE, synthetic sparse files",
    "deepseek3.2-671b": "1,367.0 GB MoE, synthetic sparse files",
    "deepseek4-1600b": "3,197.7 GB MoE, synthetic sparse files",
}

TITLE = (
    "TPU v7x (bodaborg-tpu7x-nap), >200B models: to_maxtext vs safetensor "
    "(safetensors_dynamic with PRs #5706, #5828, #5705)"
)
LEGEND = [
    "For each (Model, Scan Mode, Slices) configuration, to_maxtext and safetensor are on consecutive rows.",
    "Col 1 (HF -> GCS): to_maxtext Step 1 vs safetensor Step 1, the same cold download of the .safetensors shards, "
    "measured once per model. DeepSeek models read synthetic sparse files staged on each host (local stage).",
    "Col 2 (Convert to Orbax): to_maxtext Step 2, a CPU conversion on one host that every slice count shares. "
    "safetensor has no conversion step (—).",
    "Col 3 (Load into MaxText): to_maxtext Step 3 (Load Orbax) vs safetensor Step 2 (Safetensors+Convert = "
    "load_sharded_hf_state + transform_hf_state_to_mt_state). Peak host RAM is the runner host's.",
    "Slices (Devices): a tpu7x host has 4 chips = 8 TPU cores, 813.8 GB HBM and 1,014.5 GB RAM. "
    "1 (8), 2 (16), 4 (32) = 1, 2, 4 × 2x2x1 single-host slices (P1 runs). "
    "1 (16) = 1 × 2x2x2, 1 (32) = 1 × 2x2x4, 2 (32) = 2 × 2x2x2, 4 (64) = 4 × 2x2x2 multi-host slices "
    "(Lance Wang's runs, safetensor only).",
    "Speedup: to_maxtext total / safetensor total, end-to-end / without Step 1. N/A: no total because a step failed "
    "or was not run.",
]
HEADER = [
    "Model",
    "Scan Mode",
    "Slices (Devices)",
    "Pipeline",
    "1. HF -> GCS (s)",
    "2. Convert to Orbax (s)",
    "3. Load into MaxText (s)",
    "Total (No Step 1) (s)",
    "Total (End-to-End) (s)",
    "Speedup (E2E / No Step 1)",
]

# Failure and not-run texts that the result JSONs don't hold in full; each was checked against the phase logs.
# Keyed by (model, scan mode, slices, pipeline); a None slices entry applies to every slice count.
NOTES = {
    ("qwen3.5-397b-a17b", "scanned", 1, "safetensor"): (
        "HBM OOM in transform (load 942.10 s; the 795.5 GB checkpoint fills 98% of the 813.8 GB HBM)"
    ),
    ("qwen3.5-397b-a17b", "scanned", 2, "safetensor"): (
        "HBM OOM in transform (load 483.84 s; 11.25 GiB needed, 2.77 GiB free; same with hbmfix)"
    ),
    ("qwen3.5-397b-a17b", "unscanned", 1, "safetensor"): "Not run (the 795.5 GB checkpoint fills 98% of the HBM)",
    ("qwen3.5-397b-a17b", "unscanned", None, "to_maxtext"): (
        "Host OOM-killed at the start of the Orbax save (after 6,860 s of transform; ~807 GB converted state)"
    ),
    ("qwen3-480b-a35b", "scanned", 1, "safetensor"): "HBM OOM (961.0 GB > 813.8 GB HBM)",
    ("qwen3-480b-a35b", "unscanned", 1, "safetensor"): "Not run (961.0 GB > 813.8 GB HBM)",
    ("deepseek3-671b", "scanned", 1, "safetensor"): "HBM OOM (1,342.4 GB > 813.8 GB HBM)",
    ("deepseek3-671b", "scanned", 2, "safetensor"): (
        "Killed (-9) on slice 0 at 742.1 GB sampled peak host RAM; loads on 4 slices"
    ),
    ("deepseek3-671b", None, None, "to_maxtext"): "Not run (1.34 TB can't be converted on one host)",
}
# Extra text after a successful `to_maxtext` Step 2 time, keyed by (model, scan mode); the 2-slice row says "shared".
TM_S2_NOTES = {("qwen3-235b-a22b", "unscanned"): "16 parallel shard reads; single stream: 5,150.13"}
# Short forms of build_tpu7x_tables.failure_reason() for `to_maxtext` Step 2.
TM_REASONS = {
    "source must be staged in RAM next to the converted state; exceeds host RAM": (
        "Does not fit one host (the source must be staged in RAM next to the converted state)"
    ),
    "converted state exceeds host RAM": "Does not fit one host (the converted state exceeds host RAM)",
}
# Configurations with no record at all, shown as one pair before the scan mode's other rows.
SKIPPED = {("deepseek3-671b", "unscanned"): ("1 (8) / 2 (16)", "Skipped (same capacity limits as scanned)")}


def _fmt(x: float) -> str:
  return f"{x:,.2f}"


def _note(model: str, scan: str, n: int, pipeline: str) -> str | None:
  for key in ((model, scan, n, pipeline), (model, scan, None, pipeline), (model, None, None, pipeline)):
    if key in NOTES:
      return NOTES[key]
  return None


def _s1(model: str, seconds: float) -> str:
  return f"{_fmt(seconds)} (local stage)" if "synthetic" in MODELS[model] else _fmt(seconds)


def _st_cell(total: float, load: float, transform: float, peak: float) -> str:
  return (
      f"{_fmt(total)} (Safetensors+Convert: load {_fmt(load)} + transform {_fmt(transform)}; "
      f"peak host RAM {peak:.1f} GB)"
  )


def _cap(text: str) -> str:
  """Capitalized failure_reason(), with its peak host RAM in parentheses."""
  text = re.sub(r", (peak host RAM [\d.]+ GB)$", r" (\1)", text)
  return text[:1].upper() + text[1:]


def p1_pairs(logs: Logs, row: dict) -> list[list[str]]:
  """The `to_maxtext` and `safetensor` rows of one P1 configuration."""
  model, scan, n = row["model"], row["scan_mode"], row["num_slices"]
  label, slices = f"{model} ({MODELS[model]})", f"{n} ({row['global_devices']})"
  tm_status, st_status = row["to_maxtext_status"], row["safetensor_status"]
  tm_ok, st_ok = tm_status == "OK", st_status == "OK"
  tm_no1 = row["to_maxtext_2_convert_to_orbax_s"] + row["to_maxtext_3_load_orbax_s"]
  st_no1 = row["safetensor_2_safetensors_and_convert_s"]

  if tm_status == "not run":
    tm = ["—", _note(model, scan, n, "to_maxtext") or "Not run", "—", "N/A", "N/A", "N/A"]
  elif tm_ok:
    s2_note = TM_S2_NOTES.get((model, scan))
    s2 = _fmt(row["to_maxtext_2_convert_to_orbax_s"])
    if s2_note:
      s2 += f" ({s2_note})" if n == 1 else " (shared with 1 (8))"
    tm = [
        _s1(model, row["to_maxtext_1_hf_to_gcs_s"]),
        s2,
        f"{_fmt(row['to_maxtext_3_load_orbax_s'])} (Load Orbax)",
        _fmt(tm_no1),
        _fmt(row["to_maxtext_total_s"]),
        "1.00x (baseline)",
    ]
  else:
    reason = tm_status.removeprefix("S2: ").removeprefix("S3: ")
    text = _note(model, scan, n, "to_maxtext") or TM_REASONS.get(reason) or _cap(reason)
    tm = [_s1(model, row["to_maxtext_1_hf_to_gcs_s"]), text, "—", "N/A", "N/A", "N/A"]

  if st_ok:
    cfg = scan if n == 1 else f"{scan}_slices{n}"
    rec, _ = logs.load(model, cfg, "safetensors_dynamic")
    peak = float((rec or {}).get("peak_host_ram_gb", 0.0) or 0.0)
    if tm_ok:
      speedup = f"{row['to_maxtext_total_s'] / row['safetensor_total_s']:.2f}x / {tm_no1 / st_no1:.2f}x"
    else:
      speedup = "Only safetensor loads this config"
    st = [
        _st_cell(st_no1, row["safetensor_2b_load_sharded_hf_s"], row["safetensor_2c_transform_hf_to_mt_s"], peak),
        _fmt(st_no1),
        _fmt(row["safetensor_total_s"]),
        speedup,
    ]
  else:
    st = [_note(model, scan, n, "safetensor") or _cap(st_status), "N/A", "N/A", "N/A"]
  st = [_s1(model, row["safetensor_1_hf_to_gcs_s"]), "—"] + st
  return [[label, scan, slices, "to_maxtext"] + tm, ["↳", scan, slices, "safetensor"] + st]


def skipped_pair(model: str, scan: str, slices: str, text: str) -> list[list[str]]:
  label = f"{model} ({MODELS[model]})"
  tm_text = _note(model, scan, 1, "to_maxtext") or "Not run"
  return [
      [label, scan, slices, "to_maxtext", "—", tm_text, "—", "N/A", "N/A", "N/A"],
      ["↳", scan, slices, "safetensor", "—", "—", text, "N/A", "N/A", "N/A"],
  ]


def multihost_pair(rec: dict, cold_s1: float) -> list[list[str]]:
  """The rows of one multi-host run: `to_maxtext` was not run there."""
  model, scan = rec["model"], rec["scan_mode"]
  label, slices = f"{model} ({MODELS[model]})", f"{rec['layout'].split(' ')[0]} ({rec['devices']})"
  tm = [label, scan, slices, "to_maxtext", "—", "Not run (to_maxtext was not run on multi-host slices)", "—"]
  tm += ["N/A", "N/A", "N/A"]
  if rec["status"] == "OK":
    total = float(rec["total_s"])
    cell = _st_cell(total, float(rec["load_s"]), float(rec["transform_s"]), float(rec["peak_host_ram_gb"]))
    st_tail = [cell, _fmt(total), _fmt(cold_s1 + total), "N/A (to_maxtext not run on multi-host)"]
  else:
    st_tail = [rec["note"].replace("`", ""), "N/A", "N/A", "N/A"]
  return [tm, ["↳", scan, slices, "safetensor", _s1(model, cold_s1), "—"] + st_tail]


def build(roots: list[str], overrides: dict[str, float], multihost_csv: str) -> list[list[str]]:
  """Table rows in sheet order: per model and scan mode, the P1 configurations, then the multi-host runs."""
  logs = Logs(roots)
  p1 = [r for r in build_rows(roots, overrides) if r["model"] in MODELS]
  with open(multihost_csv, encoding="utf-8") as f:
    mh = [r for r in csv.DictReader(f) if r["source"] == "lance" and r["model"] in MODELS]
  out = []
  for model in MODELS:
    rows = [r for r in p1 if r["model"] == model]
    cold_s1 = rows[0]["safetensor_1_hf_to_gcs_s"] if rows else 0.0
    for scan in ("scanned", "unscanned"):
      if (model, scan) in SKIPPED:
        out += skipped_pair(model, scan, *SKIPPED[(model, scan)])
      for r in sorted((r for r in rows if r["scan_mode"] == scan), key=lambda r: r["num_slices"]):
        out += p1_pairs(logs, r)
      runs = [r for r in mh if r["model"] == model and r["scan_mode"] == scan]
      for r in sorted(runs, key=lambda r: (int(r["devices"]), int(r["layout"].split(" ")[0]))):
        out += multihost_pair(r, cold_s1)
  return out


def markdown(rows: list[list[str]]) -> str:
  """Title, legend and table as markdown, then the same lines as tab-separated text in a collapsed block."""
  lines = [TITLE, ""] + [f"- {line}" for line in LEGEND] + [""]
  lines.append("| " + " | ".join(HEADER) + " |")
  # Same alignment as the tab: a rendered table's cell alignment carries over when it is pasted into Sheets.
  lines.append("| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
  lines += ["| " + " | ".join(r) + " |" for r in rows]
  tsv = [TITLE, ""] + LEGEND + ["", "\t".join(HEADER)] + ["\t".join(r) for r in rows]
  lines += [
      "",
      "<details>",
      "<summary>The same title, legend and table as tab-separated text</summary>",
      "",
      "```text",
      *tsv,
      "```",
      "",
      "</details>",
  ]
  return "\n".join(lines) + "\n"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("roots", nargs="+", help="Local mirrors of the P1 <run dir>/logs (later roots override earlier ones)")
  p.add_argument("--cold-s1", action="append", default=[], help="MODEL=SECONDS cold HF->GCS override")
  p.add_argument("--multihost-csv", required=True, help="reports/tpu7x_multihost_results.csv")
  p.add_argument("--md", help="Output markdown path")
  p.add_argument(
      "--md-into",
      action="append",
      default=[],
      help="Markdown file whose text between `<!-- SHEET TABLE START -->` and `<!-- SHEET TABLE END -->` is replaced",
  )
  args = p.parse_args()
  overrides = {k: float(v) for k, v in (o.split("=", 1) for o in args.cold_s1)}
  rows = build(args.roots, overrides, args.multihost_csv)
  text = markdown(rows)
  if args.md:
    with open(args.md, "w", encoding="utf-8") as f:
      f.write(text)
  for path in args.md_into:
    with open(path, encoding="utf-8") as f:
      doc = f.read()
    start, end = "<!-- SHEET TABLE START -->\n", "<!-- SHEET TABLE END -->"
    i, j = doc.index(start) + len(start), doc.index(end)
    with open(path, "w", encoding="utf-8") as f:
      f.write(doc[:i] + text + doc[j:])
  print(f"Wrote {len(rows)} table rows ({len(rows) // 2} configurations)")


if __name__ == "__main__":
  main()
