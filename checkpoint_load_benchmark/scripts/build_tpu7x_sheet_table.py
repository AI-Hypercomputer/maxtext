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

"""Builds the plain-text `to_maxtext` vs `safetensor` tables for the tpu7x tabs of the benchmark Google Sheet.

Each table has the tabs' 10 columns, with a `to_maxtext` row and a `safetensor` row per (model, scan mode, slices):
  * `--suite p1`, the "tpu-v7x" tab: the >200B models. It combines the P1 runs on single-host 2x2x1 slices
    (`1 (8)`, `2 (16)`, `4 (32)`) with Lance Wang's runs on multi-host slices (`1 (16)` = 1 x 2x2x2,
    `1 (32)` = 1 x 2x2x4, `2 (32)` = 2 x 2x2x2, `4 (64)` = 4 x 2x2x2; `safetensors_dynamic` only), read from
    reports/tpu7x_multihost_results.csv.
  * `--suite p2`, a tab like "tpu-v5p": the 14 Priority 2 models up to 120B, on 1 and 2 slices.

The P1 and P2 rows come from the per-phase result JSONs and logs (same local mirrors, override order and `--cold-s1`
rule as build_tpu7x_tables.py).

Every cell is plain text (no bold, italics or code spans), so a copy of the rendered markdown table pastes into
Google Sheets as plain cells. The same lines are also written as tab-separated text in a fenced block: Sheets splits
pasted tab-separated text into cells.

Usage:
  B=gs://mesa-maxtext/mesakh_tpu7x_bench
  gcloud storage rsync -r $B/p1_lance_prs/run_tpu7x_200b/logs /tmp/t7/p1
  gcloud storage rsync -r $B/p1_lance_prs/run_tpu7x_200b_pstream/logs /tmp/t7/p1_pstream
  gcloud storage rsync -r $B/p2_lance_prs/run_tpu7x_16m/logs /tmp/t7/p2
  gcloud storage rsync -r $B/p2_lance_prs/run_tpu7x_16m_rerun/logs /tmp/t7/p2_rerun
  python3 checkpoint_load_benchmark/scripts/build_tpu7x_sheet_table.py --suite p1 /tmp/t7/p1 /tmp/t7/p1_pstream \\
      --cold-s1 qwen3.5-397b-a17b=814.53 \\
      --multihost-csv checkpoint_load_benchmark/reports/tpu7x_multihost_results.csv \\
      --md-into checkpoint_load_benchmark/reports/tpu7x_priority1_progress.md
  python3 checkpoint_load_benchmark/scripts/build_tpu7x_sheet_table.py --suite p2 /tmp/t7/p2 /tmp/t7/p2_rerun \\
      --md-into checkpoint_load_benchmark/reports/tpu7x_benchmark_ckpt_load_time_summary.md
"""

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tpu7x_tables import Logs, build_rows, failure_reason  # pylint: disable=wrong-import-position

# The label shown next to each model name, in sheet order.
P2_MODELS = {
    "qwen3-0.6b": "1.5 GB Dense",
    "gemma2-2b": "10.5 GB Dense, FP32 checkpoint",
    "qwen3-4b": "8.0 GB Dense",
    "gemma3-4b": "8.6 GB Dense",
    "llama3.1-8b": "16.1 GB Dense",
    "qwen3-8b": "16.4 GB Dense",
    "qwen3-14b": "29.5 GB Dense",
    "qwen3-32b": "65.5 GB Dense",
    "deepseek2-16b": "31.4 GB MoE",
    "gpt-oss-20b": "41.8 GB MoE",
    "qwen3-30b-a3b": "61.1 GB MoE",
    "mixtral-8x7b": "93.4 GB MoE",
    "llama3.1-70b": "141.1 GB Dense",
    "gpt-oss-120b": "233.7 GB MoE",
}
P1_MODELS = {
    "qwen3-235b-a22b": "470.4 GB MoE",
    "deepseek4-284b": "581.9 GB MoE, synthetic sparse files",
    "qwen3.5-397b-a17b": "795.5 GB Hybrid MoE",
    "qwen3-480b-a35b": "961.0 GB MoE",
    "deepseek3-671b": "1,342.4 GB MoE, synthetic sparse files",
    "deepseek3.2-671b": "1,367.0 GB MoE, synthetic sparse files",
    "deepseek4-1600b": "3,197.7 GB MoE, synthetic sparse files",
}
MODELS = P2_MODELS | P1_MODELS

_CODE = "(safetensors_dynamic with PRs #5706, #5828, #5705)"
_ROWS = "For each (Model, Scan Mode, Slices) configuration, to_maxtext and safetensor are on consecutive rows."
_COL1 = (
    "Col 1 (HF -> GCS): to_maxtext Step 1 vs safetensor Step 1, a cold download of the .safetensors shards, measured "
    "once and shared by the configurations that reuse it."
)
_COL2 = (
    "Col 2 (Convert to Orbax): to_maxtext Step 2, a CPU conversion on one host that every slice count shares. "
    "safetensor has no conversion step (—)."
)
_COL3 = (
    "Col 3 (Load into MaxText): to_maxtext Step 3 (Load Orbax) vs safetensor Step 2 (Safetensors+Convert = "
    "load_sharded_hf_state + transform_hf_state_to_mt_state). Peak host RAM is the runner host's."
)
_HOST = "a tpu7x host has 4 chips = 8 TPU cores, 813.8 GB HBM and 1,014.5 GB RAM."
_SPEEDUP = (
    "Speedup: to_maxtext total / safetensor total, end-to-end / without Step 1. N/A: no total because a step failed "
    "or was not run."
)
SUITES = {
    "p1": {
        "models": P1_MODELS,
        "title": f"TPU v7x (bodaborg-tpu7x-nap), >200B models: to_maxtext vs safetensor {_CODE}",
        "legend": [
            _ROWS,
            _COL1 + " DeepSeek models read synthetic sparse files staged on each host (local stage).",
            _COL2,
            _COL3,
            f"Slices (Devices): {_HOST} 1 (8), 2 (16), 4 (32) = 1, 2, 4 × 2x2x1 single-host slices (P1 runs). "
            "1 (16) = 1 × 2x2x2, 1 (32) = 1 × 2x2x4, 2 (32) = 2 × 2x2x2, 4 (64) = 4 × 2x2x2, 1 (64) = 1 × 2x4x4 "
            "multi-host slices (safetensor: Lance Wang's runs and the 10-30 runs; to_maxtext: Orbax loads of the "
            "checkpoint that the single-host conversion wrote, so Steps 1-2 are the ones of 1 (8)).",
            _SPEEDUP,
            "Status symbols: ⏳ In progress = currently running on the cluster; ⏳ Queued = queued in Kueue waiting "
            "for spot slice capacity.",
        ],
    },
    "p2": {
        "models": P2_MODELS,
        "title": f"TPU v7x (bodaborg-tpu7x-nap), models up to 120B: to_maxtext vs safetensor {_CODE}",
        "legend": [
            _ROWS,
            _COL1,
            _COL2,
            _COL3,
            f"Slices (Devices): {_HOST} 1 (8) = 1 slice (2x2x1, one host); 2 (16) = 2 slices over DCN; "
            "1 (16) = 1 × 2x2x2, 2 (32) = 2 × 2x2x2 multi-host slices (smoke test).",
            _SPEEDUP,
        ],
    },
}
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

# Failure and not-run texts that the result JSONs and logs don't give in full; each was checked against the phase
# logs. Keyed by (model, scan mode, slices, pipeline); a None entry matches any value.
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
    ("deepseek3.2-671b", None, None, "to_maxtext"): "Not run (1.37 TB can't be converted on one host)",
    ("deepseek4-1600b", None, None, "to_maxtext"): "Not run (3.2 TB can't be converted on one host)",
}
# Extra text after a successful `to_maxtext` Step 2 time, keyed by (model, scan mode); the 2-slice row says "shared".
TM_S2_NOTES = {
    ("qwen3-235b-a22b", "unscanned"): "16 parallel shard reads; single stream: 5,150.13",
    ("qwen3.5-397b-a17b", "unscanned"): (
        "with this branch's Orbax save-buffer cap, OOM-killed at the save without it; 16 parallel shard reads"
    ),
}
# Step 3 text of a configuration whose conversion worked but whose Orbax load was not run, keyed by (model, scan, n).
TM_S3_NOTES = {
    ("qwen3.5-397b-a17b", "unscanned", 1): "Not run (the 795.5 GB checkpoint fills 98% of the 813.8 GB HBM)",
}
# Step 3 text of a multi-host `to_maxtext` Orbax load that is currently running or queued on the cluster.
TM_MH_PENDING = {
    ("qwen3.5-397b-a17b", "unscanned", "1 × 2x2x4"): (
        "⏳ Queued (waiting for 1 × 2x2x4 spot slice to load the converted Orbax checkpoint)"
    ),
}
# Multi-host `safetensors_dynamic` runs currently running or queued that don't have a CSV row yet.
PENDING_MULTIHOST = [
    ("deepseek3.2-671b", "scanned", "1 × 2x2x4", 4, 32, "⏳ Queued (waiting for 1 × 2x2x4 spot slice)"),
    ("deepseek3.2-671b", "unscanned", "1 × 2x2x4", 4, 32, "⏳ Queued (waiting for 1 × 2x2x4 spot slice)"),
    ("deepseek4-1600b", "scanned", "1 × 2x4x4", 8, 64, "⏳ Queued (waiting for 1 × 2x4x4 spot slice)"),
    ("deepseek4-1600b", "unscanned", "1 × 2x4x4", 8, 64, "⏳ Queued (waiting for 1 × 2x4x4 spot slice)"),
]
# Legend line for the `safetensor (<label>)` rows of a fix variant (`--variant`), added when the table has such rows.
VARIANT_LEGEND = {
    "with fix": (
        "safetensor (with fix): the same configuration re-run with this branch's MaxText fixes for two HBM peaks "
        "(composite leaves stacked one by one; reshard slices joined in pairs), from a separate code tree. The plain "
        "safetensor rows don't include these fixes."
    ),
}
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


def _hbm_oom(log_path: str) -> str | None:
  """The phase and sizes of a TPU HBM OOM, from a phase's logs (one per slice), or None without an HBM OOM."""
  text = ""
  for path in [log_path] + sorted(glob.glob(log_path.removesuffix(".log") + "_slice*.log")):
    if os.path.exists(path):
      with open(path, encoding="utf-8", errors="replace") as f:
        text += f.read()
  if "RESOURCE_EXHAUSTED" not in text:
    return None
  loads = [float(x) for x in re.findall(r"load_sharded_hf_state completed in ([\d.]+)s", text)]
  details = [f"load {_fmt(max(loads))} s"] if loads else []
  temps = re.search(r"HLO temporaries \(([\d.]+)G\) exceeds available HBM \(([\d.]+)G\)", text)
  need = re.search(r"Attempting to (?:allocate|reserve) ([\d.]+)([MG])", text)
  free = re.search(r"([\d.]+)([MG]) free", text)
  if temps:
    details.append(f"{temps.group(1)} GiB of HLO temporaries per device, {temps.group(2)} GiB available")
  elif need and free:
    details.append(f"{need.group(1)} {need.group(2)}iB needed, {free.group(1)} {free.group(2)}iB free")
  where = "HBM OOM in transform" if loads else "HBM OOM during load"
  return f"{where} ({'; '.join(details)})" if details else where


def config_pairs(logs: Logs, row: dict) -> list[list[str]]:
  """The `to_maxtext` and `safetensor` rows of one configuration from build_tpu7x_tables.build_rows()."""
  model, scan, n = row["model"], row["scan_mode"], row["num_slices"]
  label, slices = f"{model} ({MODELS[model]})", f"{n} ({row['global_devices']})"
  tm_status, st_status = row["to_maxtext_status"], row["safetensor_status"]
  tm_ok, st_ok = tm_status == "OK", st_status == "OK"
  tm_no1 = row["to_maxtext_2_convert_to_orbax_s"] + row["to_maxtext_3_load_orbax_s"]
  st_no1 = row["safetensor_2_safetensors_and_convert_s"]
  cfg = scan if n == 1 else f"{scan}_slices{n}"

  if tm_status == "not run":
    tm = ["—", _note(model, scan, n, "to_maxtext") or "Not run", "—", "N/A", "N/A", "N/A"]
  elif tm_ok or tm_status.startswith("S3: "):
    s2_note = TM_S2_NOTES.get((model, scan))
    s2 = _fmt(row["to_maxtext_2_convert_to_orbax_s"])
    if s2_note:
      s2 += f" ({s2_note})" if n == 1 else " (shared with 1 (8))"
    if tm_ok:
      tm = [
          _s1(model, row["to_maxtext_1_hf_to_gcs_s"]),
          s2,
          f"{_fmt(row['to_maxtext_3_load_orbax_s'])} (Load Orbax)",
          _fmt(tm_no1),
          _fmt(row["to_maxtext_total_s"]),
          "1.00x (baseline)",
      ]
    else:
      # The conversion worked; only the Orbax load of this configuration is missing or failed.
      reason = tm_status.removeprefix("S3: ")
      s3 = TM_S3_NOTES.get((model, scan, n)) or ("Not run" if reason == "not run" else _cap(reason))
      pending = s3.startswith("⏳")
      tm = [
          _s1(model, row["to_maxtext_1_hf_to_gcs_s"]),
          s2,
          s3,
          "⏳" if pending else "N/A",
          "⏳" if pending else "N/A",
          "⏳ In progress" if pending else "N/A",
      ]
  else:
    reason = tm_status.removeprefix("S2: ").removeprefix("S3: ")
    text = _note(model, scan, n, "to_maxtext") or TM_REASONS.get(reason) or _cap(reason)
    tm = [_s1(model, row["to_maxtext_1_hf_to_gcs_s"]), text, "—", "N/A", "N/A", "N/A"]

  rec, log_path = logs.load(model, cfg, "safetensors_dynamic")
  if st_ok:
    peak = float((rec or {}).get("peak_host_ram_gb", 0.0) or 0.0)
    s3_note = TM_S3_NOTES.get((model, scan, n), "")
    if tm_ok:
      speedup = f"{row['to_maxtext_total_s'] / row['safetensor_total_s']:.2f}x / {tm_no1 / st_no1:.2f}x"
    elif tm_status == "S3: not run" and s3_note.startswith("⏳"):
      speedup = "⏳ In progress (waiting for to_maxtext Step 3)"
    elif tm_status == "S3: not run":
      speedup = "N/A (to_maxtext load not run)"
    else:
      speedup = "Only safetensor loads this config"
    st = [
        _st_cell(st_no1, row["safetensor_2b_load_sharded_hf_s"], row["safetensor_2c_transform_hf_to_mt_s"], peak),
        _fmt(st_no1),
        _fmt(row["safetensor_total_s"]),
        speedup,
    ]
  else:
    text = _note(model, scan, n, "safetensor") or (rec and _hbm_oom(log_path)) or _cap(st_status)
    st = [text, "N/A", "N/A", "Only to_maxtext loads this config" if tm_ok else "N/A"]
  st = [_s1(model, row["safetensor_1_hf_to_gcs_s"]), "—"] + st
  return [[label, scan, slices, "to_maxtext"] + tm, ["↳", scan, slices, "safetensor"] + st]


def skipped_pair(model: str, scan: str, slices: str, text: str) -> list[list[str]]:
  label = f"{model} ({MODELS[model]})"
  tm_text = _note(model, scan, 1, "to_maxtext") or "Not run"
  return [
      [label, scan, slices, "to_maxtext", "—", tm_text, "—", "N/A", "N/A", "N/A"],
      ["↳", scan, slices, "safetensor", "—", "—", text, "N/A", "N/A", "N/A"],
  ]


def _multihost_cfg(rec: dict) -> str:
  """The harness's result directory of a multi-host run: `<scan>[_slices<N>]_hosts<hosts per slice>`."""
  n = int(rec["layout"].split(" ")[0])
  return f"{rec['scan_mode']}{f'_slices{n}' if n > 1 else ''}_hosts{int(rec['hosts']) // n}"


def _mh_st_tail(
    rec: dict,
    cold_s1: float,
    tm_total: float = 0.0,
    tm_no1: float = 0.0,
    no_tm: str = "N/A (to_maxtext not run on multi-host)",
) -> list[str]:
  """Step 3 to speedup cells of a multi-host safetensors_dynamic run (a row of tpu7x_multihost_results.csv).

  `no_tm` is the speedup text when there is no to_maxtext total to compare with.
  """
  if rec["status"] == "PENDING":
    tag = "⏳ In progress" if "In progress" in rec["note"] else "⏳ Queued"
    return [rec["note"], "⏳", "⏳", tag]
  if rec["status"] != "OK":
    return [rec["note"].replace("`", ""), "N/A", "N/A", "N/A"]
  total = float(rec["total_s"])
  cell = _st_cell(total, float(rec["load_s"]), float(rec["transform_s"]), float(rec["peak_host_ram_gb"]))
  speedup = f"{tm_total / (cold_s1 + total):.2f}x / {tm_no1 / total:.2f}x" if tm_total else no_tm
  return [cell, _fmt(total), _fmt(cold_s1 + total), speedup]


def _tm_converted(row: dict | None) -> bool:
  """True if the `to_maxtext` conversion (Steps 1-2) of a build_rows() row worked, whatever its Orbax load did."""
  return bool(row) and (row["to_maxtext_status"] == "OK" or row["to_maxtext_status"].startswith("S3: "))


def _mh_tm_missing(model: str, scan: str, tm_ref: dict | None) -> str:
  """The Step 2 text of a multi-host to_maxtext row without an Orbax load result."""
  note = _note(model, scan, 0, "to_maxtext")
  if _tm_converted(tm_ref):
    return "Not run on multi-host slices"
  if tm_ref is None or tm_ref["to_maxtext_status"] == "not run":
    return note or "Not run (no single-host conversion)"
  return "No checkpoint to load (the single-host conversion fails, see 1 (8))"


def multihost_pair(rec: dict, cold_s1: float, logs: Logs | None = None, tm_ref: dict | None = None) -> list[list[str]]:
  """The rows of one multi-host run.

  The to_maxtext row has values when the harness loaded the model's converted checkpoint on the same layout (an
  Orbax load result in the run dirs of `logs`, from a run with a `run_dir` in MESAKH_MH_RUNS); its conversion
  (Steps 1-2) is the one of `tm_ref`, the model's P1 row for the scan mode.
  """
  model, scan = rec["model"], rec["scan_mode"]
  label, slices = f"{model} ({MODELS[model]})", f"{rec['layout'].split(' ')[0]} ({rec['devices']})"
  tm3 = logs.load(model, _multihost_cfg(rec), "to_maxtext_step3")[0] if logs else None
  s2 = tm_ref["to_maxtext_2_convert_to_orbax_s"] if _tm_converted(tm_ref) else 0.0
  tm_pending = TM_MH_PENDING.get((model, scan, rec["layout"]))
  tm_total = tm_no1 = 0.0
  if tm3 is None and tm_pending and s2 and tm_ref is not None:
    s1 = tm_ref["to_maxtext_1_hf_to_gcs_s"]
    tag = "⏳ In progress" if "In progress" in tm_pending else "⏳ Queued"
    tm = [label, scan, slices, "to_maxtext", _s1(model, s1), f"{_fmt(s2)} (shared with 1 (8))", tm_pending, "⏳", "⏳", tag]
    no_tm = f"{tag} (waiting for to_maxtext Step 3)"
  elif tm3 is None or not s2 or tm_ref is None:
    tm = [label, scan, slices, "to_maxtext", "—", _mh_tm_missing(model, scan, tm_ref), "—", "N/A", "N/A", "N/A"]
    no_tm = "N/A (to_maxtext not run on this layout)" if _tm_converted(tm_ref) else "Only safetensor loads this config"
  else:
    s1 = tm_ref["to_maxtext_1_hf_to_gcs_s"]
    head = [label, scan, slices, "to_maxtext", _s1(model, s1), f"{_fmt(s2)} (shared with 1 (8))"]
    if tm3.get("status") == "FAILED":
      log_path = logs.load(model, _multihost_cfg(rec), "to_maxtext_step3")[1] if logs else ""
      reason = _cap(failure_reason(tm3, log_path))
      tm = head + [reason, "N/A", "N/A", "N/A"]
      no_tm = "Only safetensor loads this config"
    else:
      load = float(tm3["load_orbax_s"])
      tm_no1, tm_total = s2 + load, s1 + s2 + load
      tm = head + [f"{_fmt(load)} (Load Orbax)", _fmt(tm_no1), _fmt(tm_total), "1.00x (baseline)"]
      no_tm = "N/A (to_maxtext not run on this layout)" if _tm_converted(tm_ref) else "Only safetensor loads this config"
  st = ["↳", scan, slices, "safetensor", _s1(model, cold_s1), "—"]
  st += _mh_st_tail(rec, cold_s1, tm_total, tm_no1, no_tm)
  return [tm, st]


def variant_row(rec: dict, cold_s1: float, label: str, tm_ref: dict | None = None) -> list[str]:
  """One extra safetensor row for a multi-host run of a fix variant (a CSV row whose note names the variant)."""
  model, scan = rec["model"], rec["scan_mode"]
  slices = f"{rec['layout'].split(' ')[0]} ({rec['devices']})"
  no_tm = "N/A (to_maxtext not run on this layout)" if _tm_converted(tm_ref) else "Only safetensor loads this config"
  head = ["↳", scan, slices, f"safetensor ({label})", _s1(model, cold_s1), "—"]
  return head + _mh_st_tail(rec, cold_s1, no_tm=no_tm)


def variant_config_row(row: dict, logs: Logs, label: str) -> list[str] | None:
  """One extra safetensor row for a single-host configuration of a fix variant (a result in the variant's mirror)."""
  model, scan, n = row["model"], row["scan_mode"], row["num_slices"]
  rec, log_path = logs.load(model, scan if n == 1 else f"{scan}_slices{n}", "safetensors_dynamic")
  if rec is None:
    return None
  slices = f"{n} ({row['global_devices']})"
  cold_s1 = row["safetensor_1_hf_to_gcs_s"]
  if rec.get("status") == "FAILED":
    text = _hbm_oom(log_path) or _cap(failure_reason(rec, log_path))
    tail = [text, "N/A", "N/A", "N/A"]
  else:
    total = float(rec["safetensors_and_convert_s"])
    peak = float(rec.get("peak_host_ram_gb", 0.0) or 0.0)
    cell = _st_cell(total, float(rec["load_sharded_hf_s"]), float(rec["transform_hf_to_mt_s"]), peak)
    tm_ok = row["to_maxtext_status"] == "OK"
    tm_no1 = row["to_maxtext_2_convert_to_orbax_s"] + row["to_maxtext_3_load_orbax_s"]
    s3_note = TM_S3_NOTES.get((model, scan, n), "")
    speedup = (
        f"{row['to_maxtext_total_s'] / (cold_s1 + total):.2f}x / {tm_no1 / total:.2f}x"
        if tm_ok
        else "⏳ In progress (waiting for to_maxtext Step 3)"
        if row["to_maxtext_status"] == "S3: not run" and s3_note.startswith("⏳")
        else "N/A (to_maxtext load not run)"
        if row["to_maxtext_status"] == "S3: not run"
        else "Only safetensor loads this config"
    )
    tail = [cell, _fmt(total), _fmt(cold_s1 + total), speedup]
  return ["↳", scan, slices, f"safetensor ({label})", _s1(model, cold_s1), "—"] + tail


def build(
    suite: str,
    roots: list[str],
    overrides: dict[str, float],
    multihost_csv: str | None,
    variants: list[tuple[str, str]] = (),
) -> list[list[str]]:
  """Table rows in sheet order: per model and scan mode, the run configurations, then the multi-host runs.

  Multi-host runs are the rows of `multihost_csv` on multi-host slices (not the 2x2x1 P1 rows it lists for
  comparison). A row whose note starts with a variant label, and a result in a variant's mirror (`variants`:
  (label, root) pairs), add a `safetensor (<label>)` row after the configuration's pair.
  """
  models = SUITES[suite]["models"]
  logs = Logs(roots)
  variant_logs = [(label, Logs([root])) for label, root in variants]
  runs = [r for r in build_rows(roots, overrides) if r["model"] in models]
  mh = []
  if multihost_csv:
    with open(multihost_csv, encoding="utf-8") as f:
      mh = [r for r in csv.DictReader(f) if r["model"] in models and not r["layout"].endswith("2x2x1")]
  labels = [label for label, _ in variants]
  out = []
  for model in models:
    rows = [r for r in runs if r["model"] == model]
    cold_s1 = rows[0]["safetensor_1_hf_to_gcs_s"] if rows else 0.0
    for scan in ("scanned", "unscanned"):
      blocks = []
      if (model, scan) in SKIPPED:
        blocks.append(((0, 0), skipped_pair(model, scan, *SKIPPED[(model, scan)])))
      scan_rows = sorted((r for r in rows if r["scan_mode"] == scan), key=lambda r: r["num_slices"])
      for r in scan_rows:
        blk = config_pairs(logs, r)
        for label, vlogs in variant_logs:
          extra = variant_config_row(r, vlogs, label)
          if extra:
            blk.append(extra)
        blocks.append(((int(r["global_devices"]), int(r["num_slices"])), blk))
      tm_ref = scan_rows[0] if scan_rows else None
      model_mh = [r for r in mh if r["model"] == model and r["scan_mode"] == scan]
      have_layouts = set()
      base = []
      for r in model_mh:
        if not any(r.get("note", "").startswith(label) for label in labels) and r["layout"] not in have_layouts:
          have_layouts.add(r["layout"])
          base.append(r)
      for p_model, p_scan, p_layout, p_hosts, p_devices, p_note in PENDING_MULTIHOST:
        if p_model == model and p_scan == scan and p_layout not in have_layouts:
          base.append({
              "model": model,
              "scan_mode": scan,
              "layout": p_layout,
              "hosts": str(p_hosts),
              "devices": str(p_devices),
              "status": "PENDING",
              "note": p_note,
          })
      for r in base:
        blk = multihost_pair(r, cold_s1, logs, tm_ref)
        for v in model_mh:
          label = next((label for label in labels if v.get("note", "").startswith(label)), None)
          if label and v["layout"] == r["layout"]:
            blk.append(variant_row(v, cold_s1, label, tm_ref))
        blocks.append(((int(r["devices"]), int(r["layout"].split(" ")[0])), blk))
      for _, blk in sorted(blocks, key=lambda item: item[0]):
        out += blk
  return out


def markdown(suite: str, rows: list[list[str]]) -> str:
  """Title, legend and table as markdown, then the same lines as tab-separated text in a collapsed block."""
  title, legend = SUITES[suite]["title"], list(SUITES[suite]["legend"])
  shown = {r[3].removeprefix("safetensor (").removesuffix(")") for r in rows if r[3].startswith("safetensor (")}
  legend += [VARIANT_LEGEND[label] for label in VARIANT_LEGEND if label in shown]
  lines = [title, ""] + [f"- {line}" for line in legend] + [""]
  lines.append("| " + " | ".join(HEADER) + " |")
  # Same alignment as the tabs: a rendered table's cell alignment carries over when it is pasted into Sheets.
  lines.append("| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
  lines += ["| " + " | ".join(r) + " |" for r in rows]
  tsv = [title, ""] + legend + ["", "\t".join(HEADER)] + ["\t".join(r) for r in rows]
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
  p.add_argument("roots", nargs="+", help="Local mirrors of <run dir>/logs (later roots override earlier ones)")
  p.add_argument("--suite", required=True, choices=sorted(SUITES), help="p1: >200B models; p2: models up to 120B")
  p.add_argument("--cold-s1", action="append", default=[], help="MODEL=SECONDS cold HF->GCS override")
  p.add_argument("--multihost-csv", help="reports/tpu7x_multihost_results.csv (adds the multi-host rows to p1)")
  p.add_argument(
      "--variant",
      action="append",
      default=[],
      help="LABEL=ROOT: a local mirror of a fix variant's run dir; its results add `safetensor (LABEL)` rows",
  )
  p.add_argument("--md", help="Output markdown path")
  p.add_argument(
      "--md-into",
      action="append",
      default=[],
      help="Markdown file whose text between `<!-- SHEET TABLE START -->` and `<!-- SHEET TABLE END -->` is replaced",
  )
  args = p.parse_args()
  overrides = {k: float(v) for k, v in (o.split("=", 1) for o in args.cold_s1)}
  variants = [tuple(v.split("=", 1)) for v in args.variant]
  rows = build(args.suite, args.roots, overrides, args.multihost_csv, variants)
  text = markdown(args.suite, rows)
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
