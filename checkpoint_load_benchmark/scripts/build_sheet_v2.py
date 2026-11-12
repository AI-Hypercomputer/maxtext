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

"""Builds Google Sheet v2 CSVs and `gsheets mutate batch` JSON files from v2 JSONs and legacy v1 results."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
import dataclasses
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
from typing import Any

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import legacy_v1  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

DEFAULT_SPREADSHEET_ID = "1dOUCtUNgbmJMBf4Jc1AcwF71_6THB3Zt9mYUVQrGP9I"
GSHEETS_BIN = "/google/bin/releases/gemini-agents-gsheets/gsheets"
README_UPDATED_CELL = "README!B18"
PROGRESS_UPDATED_CELL = "Progress!B2"
_SHEETS_EPOCH = datetime.datetime(1899, 12, 30, tzinfo=datetime.timezone.utc)
_SKIP_DIR_PARTS = frozenset({"attempts", "locks", "done"})
_SKIP_FILENAMES = frozenset({"tasks.json", "CODE_VERSION.json"})
_PHASE_PIPELINE_MAP = {
    "safetensors_dynamic": "safetensors_dynamic",
    "to_maxtext_step1": "to_maxtext",
    "to_maxtext_step2": "to_maxtext",
    "to_maxtext_step3": "to_maxtext",
}
_TIER_SORT_ORDER = {"S": 0, "A": 1, "C": 2, "B": 3}
_PLAT_SORT_ORDER = {"v7x": 0, "v6e": 1, "v5e": 2}
_PIPE_SORT_ORDER = {"to_maxtext": 0, "safetensors_dynamic": 1}
_PROGRESS_IMPL_ROWS: dict[str, int] = {
    "T1": 22,
    "T2": 23,
    "T3": 24,
    "T4": 25,
    "T5": 26,
    "Review": 27,
    "T0.3": 28,
    "T0.4": 29,
    "T0.5": 30,
    "T0.6": 31,
    "S0": 32,
    "S1": 33,
    "S2": 34,
    "S3": 35,
    "S4": 36,
    "T1.1": 37,
    "T1.2": 38,
    "T1.3": 39,
    "T1.4": 40,
    "T1.5": 41,
    "T1.6": 42,
    "T2.1": 43,
    "T2.2": 44,
    "T2.3": 45,
    "T2.4": 46,
    "T2.5": 47,
    "T2.6": 48,
    "T2.7": 49,
    "T2.8": 50,
    "T2.9": 51,
    "T2.10": 52,
    "T3.1": 53,
    "T3.2": 54,
    "T3.3": 55,
    "T3.4": 56,
    "T3.5": 57,
    "T4.1": 58,
    "T4.2": 59,
    "T4.3": 60,
    "T4.4": 61,
    "T5.1": 62,
    "T5.2": 63,
    "T5.3": 64,
}


@dataclasses.dataclass
class V2Dataset:
  """Parsed v2 records and formatted rows across all five Sheet v2 data tabs."""

  summary_rows: list[dict[str, Any]]
  step1_rows: list[dict[str, Any]]
  stages_sd_rows: list[dict[str, Any]]
  stages_tm_rows: list[dict[str, Any]]
  runs_rows: list[dict[str, Any]]

  def validate(self) -> list[str]:
    """Validates every tab row against `schema_v2` column constraints."""
    errors: list[str] = []
    tab_pairs = (
        ("Summary", self.summary_rows, schema_v2.SUMMARY_COLUMNS),
        ("Step 1 downloads", self.step1_rows, schema_v2.STEP1_COLUMNS),
        ("Stages: safetensors_dynamic", self.stages_sd_rows, schema_v2.STAGES_SD_COLUMNS),
        ("Stages: to_maxtext", self.stages_tm_rows, schema_v2.STAGES_TM_COLUMNS),
        ("Runs", self.runs_rows, schema_v2.RUNS_COLUMNS),
    )
    for tab_name, rows, cols in tab_pairs:
      for idx, row in enumerate(rows):
        for err in schema_v2.validate_row_dict(row, cols):
          errors.append(f"{tab_name}[{idx}] ({row.get('row_id', row.get('model_name'))}): {err}")
    return errors


def col_letter(index: int) -> str:
  """Converts a 0-based column index to A1 column letters (0 -> A, 26 -> AA)."""
  if index < 0:
    raise ValueError(f"Column index must be non-negative, got {index}")
  out: list[str] = []
  idx = index + 1
  while idx > 0:
    idx, rem = divmod(idx - 1, 26)
    out.append(chr(ord("A") + rem))
  return "".join(reversed(out))


def sheets_serial(ts: datetime.datetime | str | None) -> float | str:
  """Converts a UTC datetime or ISO-8601 string into a Google Sheets serial number."""
  if ts is None or ts == "":
    return ""
  if isinstance(ts, (int, float)) and not isinstance(ts, bool):
    return float(ts)
  if isinstance(ts, str):
    raw = ts.strip()
    if not raw:
      return ""
    if raw.endswith("Z"):
      raw = raw[:-1] + "+00:00"
    try:
      dt = datetime.datetime.fromisoformat(raw)
    except ValueError:
      return ts
  else:
    dt = ts
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=datetime.timezone.utc)
  return round((dt.astimezone(datetime.timezone.utc) - _SHEETS_EPOCH).total_seconds() / 86400.0, 6)


def gcs_to_https(url: str | None, *, is_dir: bool = False) -> str:
  """Converts `gs://` URIs to clickable Google Cloud console / storage URLs."""
  if not url:
    return ""
  text = str(url).strip()
  if not text.startswith("gs://"):
    return text
  stripped = text.removeprefix("gs://").lstrip("/")
  if is_dir:
    return f"https://console.cloud.google.com/storage/browser/{stripped}"
  return f"https://storage.cloud.google.com/{stripped}"


def _empty_row(columns: Sequence[schema_v2.ColumnSpec]) -> dict[str, Any]:
  return {c.key: "" for c in columns}


def _is_number(val: Any) -> bool:
  return isinstance(val, (int, float)) and not isinstance(val, bool)


def _round_num(val: float, digits: int = 4) -> float | int:
  rounded = round(float(val), digits)
  if rounded.is_integer() and digits == 0:
    return int(rounded)
  return rounded


def _record_priority(rec: Mapping[str, Any]) -> tuple[int, str, str]:
  return (
      1 if rec.get("run_status") == "OK" else 0,
      str(rec.get("phase_finished_at_utc") or ""),
      str(rec.get("run_id") or ""),
  )


def _load_json_file(path: str) -> dict[str, Any] | None:
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError):
    return None
  return data if isinstance(data, dict) else None


def _should_skip_json(path: str, *, include_smoke: bool, include_val: bool) -> bool:
  parts = set(path.split(os.sep))
  if os.path.basename(path) in _SKIP_FILENAMES or (parts & _SKIP_DIR_PARTS):
    return True
  if not include_smoke and "smoke" in parts:
    return True
  if not include_val and "val" in parts:
    return True
  return False


def _effective_record_slices(rec: Mapping[str, Any], phase: str, run_id: str) -> int:
  """Returns slice count for `rec`, inferring multi-slice `to_maxtext` Step 1/2 target slices from `run_id`."""
  slices = int(rec.get("number_of_slices") or 1)
  if phase in ("to_maxtext_step1", "to_maxtext_step2"):
    m_slices = re.search(r"-(\d+)x[0-9x]+-r\d+$", run_id)
    if m_slices:
      return int(m_slices.group(1))
  return slices


def scan_v2_roots(
    roots: Sequence[str],
    *,
    include_smoke: bool = False,
    include_val: bool = False,
) -> dict[str, Any]:
  """Walks local mirrored GCS v2 trees and indexes Step 1, Step 1b, and phase records."""
  step1_by_pm: dict[tuple[str, str], dict[str, Any]] = {}
  step1b_convert_by_model: dict[str, dict[str, Any]] = {}
  step1b_copy_by_pm: dict[tuple[str, str], dict[str, Any]] = {}
  # (platform, model, scan, slices, topo, pipeline) -> rep_idx -> phase_name -> rec
  phases_by_cfg: dict[tuple[str, str, str, int, str, str], dict[int, dict[str, dict[str, Any]]]] = {}
  runs_map: dict[tuple[Any, ...], dict[str, Any]] = {}

  for root in [r for r in roots if r and os.path.isdir(r)]:
    for dirpath, _, filenames in os.walk(root):
      for fname in sorted(filenames):
        if not fname.endswith(".json"):
          continue
        fpath = os.path.join(dirpath, fname)
        if _should_skip_json(fpath, include_smoke=include_smoke, include_val=include_val):
          continue
        rec = _load_json_file(fpath)
        if not rec or rec.get("schema_version") != schema_v2.SCHEMA_VERSION:
          continue
        phase = str(rec.get("phase_name") or "")
        platform = str(rec.get("tpu_platform") or "")
        model = str(rec.get("model_name") or "")
        run_id = str(rec.get("run_id") or "")

        if phase == "step1_hf_to_gcs" and platform and model:
          pm = (platform, model)
          if pm not in step1_by_pm or _record_priority(rec) > _record_priority(step1_by_pm[pm]):
            step1_by_pm[pm] = rec
          runs_map[(phase, platform, model, "", 0, "", run_id, 1)] = rec
        elif phase == "step1b_deepseek_dequantize" and model:
          if model not in step1b_convert_by_model or _record_priority(rec) > _record_priority(
              step1b_convert_by_model[model]
          ):
            step1b_convert_by_model[model] = rec
          runs_map[(phase, platform, model, "", 0, "", run_id, 1)] = rec
        elif phase == "step1b_deepseek_copy_region" and platform and model:
          pm = (platform, model)
          if pm not in step1b_copy_by_pm or _record_priority(rec) > _record_priority(step1b_copy_by_pm[pm]):
            step1b_copy_by_pm[pm] = rec
          runs_map[(phase, platform, model, "", 0, "", run_id, 1)] = rec
        elif phase in _PHASE_PIPELINE_MAP and platform and model:
          pipeline = str(rec.get("pipeline_name") or _PHASE_PIPELINE_MAP[phase])
          scan_mode = str(rec.get("scan_layers_mode") or "scanned")
          slices = _effective_record_slices(rec, phase, run_id)
          topology = str(rec.get("slice_topology") or "")
          rep = int(rec.get("repetition_index") or 1)
          cfg_key = (platform, model, scan_mode, slices, topology, pipeline)
          by_rep = phases_by_cfg.setdefault(cfg_key, {})
          by_phase = by_rep.setdefault(rep, {})
          if phase not in by_phase or _record_priority(rec) > _record_priority(by_phase[phase]):
            by_phase[phase] = rec
          run_key = (phase, platform, model, scan_mode, slices, topology, run_id, rep)
          if run_key not in runs_map or _record_priority(rec) > _record_priority(runs_map[run_key]):
            runs_map[run_key] = rec

  return {
      "step1_by_pm": step1_by_pm,
      "step1b_convert_by_model": step1b_convert_by_model,
      "step1b_copy_by_pm": step1b_copy_by_pm,
      "phases_by_cfg": phases_by_cfg,
      "runs_records": list(runs_map.values()),
  }


def _build_step1_rows(
    v2_data: Mapping[str, Any],
    legacy_step1_rows: Sequence[Mapping[str, Any]],
    models_by_name: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
  """Merges cold Step 1 and Step 1b records per platform/model and appends legacy Step 1 rows."""
  out: list[dict[str, Any]] = []
  step1_by_pm: Mapping[tuple[str, str], Mapping[str, Any]] = v2_data["step1_by_pm"]
  conv_by_m: Mapping[str, Mapping[str, Any]] = v2_data["step1b_convert_by_model"]
  copy_by_pm: Mapping[tuple[str, str], Mapping[str, Any]] = v2_data["step1b_copy_by_pm"]

  all_keys = sorted(
      set(step1_by_pm.keys()) | set(copy_by_pm.keys()) | {("v7x", m) for m in conv_by_m},
      key=lambda pm: (_PLAT_SORT_ORDER.get(pm[0], 9), models_by_name.get(pm[1], {}).get("_idx", 999), pm[1]),
  )
  for plat, model in all_keys:
    m_info = models_by_name.get(model, {})
    needs_1b = bool(m_info.get("needs_step1b"))
    s1 = step1_by_pm.get((plat, model), {})
    conv = conv_by_m.get(model, {}) if (needs_1b and plat == "v7x") else {}
    cpy = copy_by_pm.get((plat, model), {}) if needs_1b else {}
    if not s1 and not conv and not cpy:
      continue

    row = _empty_row(schema_v2.STEP1_COLUMNS)
    profile = platforms.get_profile(plat)
    row["tpu_platform"] = plat
    row["step1_gcs_bucket_region"] = (
        s1.get("step1_gcs_bucket_region") or cpy.get("step1_gcs_bucket_region") or profile.gcp_region
    )
    row["model_name"] = model
    row["step1_hf_repo_id"] = s1.get("step1_hf_repo_id") or m_info.get("hf_repo_id", "")

    for col in schema_v2.STEP1_COLUMNS:
      if col.key.startswith("step1_") and col.key in s1 and s1[col.key] not in (None, ""):
        row[col.key] = _round_num(s1[col.key], 4) if _is_number(s1[col.key]) else s1[col.key]

    if not needs_1b:
      for col in schema_v2.STEP1_COLUMNS:
        if col.key.startswith("step1b_"):
          row[col.key] = "N/A"
    else:
      merged_1b = dict(cpy)
      merged_1b.update(conv)
      if merged_1b:
        for col in schema_v2.STEP1_COLUMNS:
          if col.key.startswith("step1b_") and col.key in merged_1b and merged_1b[col.key] not in (None, ""):
            val = merged_1b[col.key]
            row[col.key] = _round_num(val, 4) if _is_number(val) else val
        if (
            not row.get("step1b_deepseek_copy_bf16_to_region_seconds")
            and row.get("step1b_deepseek_copy_bf16_to_region_seconds") != 0.0
        ):
          row["step1b_deepseek_copy_bf16_to_region_seconds"] = 0.0 if plat == "v7x" else "N/A"
        if plat != "v7x" and cpy:
          row["step1b_deepseek_convert_fp8_fp4_to_bf16_seconds"] = "N/A"
          row["step1b_deepseek_convert_input_gb"] = "N/A"
          row["step1b_deepseek_upload_bf16_to_gcs_seconds"] = "N/A"
          env_c = cpy.get("env_cluster") or {}
          row["step1b_deepseek_host_machine_type"] = env_c.get("host_machine_type") or profile.cpu_machine_type
      else:
        for col in schema_v2.STEP1_COLUMNS:
          if col.key.startswith("step1b_"):
            row[col.key] = "PENDING"

    out.append(row)

  out.extend(dict(r) for r in legacy_step1_rows)
  return out


def _effective_step1_seconds(
    step1_rec: Mapping[str, Any] | None,
    phase_step1_rec: Mapping[str, Any] | None,
) -> float | str:
  """Returns verified cold Step 1 duration in seconds or status code."""
  if step1_rec:
    if step1_rec.get("run_status") == "OK" and _is_number(step1_rec.get("step1_total_seconds")):
      return float(step1_rec["step1_total_seconds"])
    return schema_v2.normalize_sheet_status(
        str(step1_rec.get("run_status") or "ERROR"), step1_rec.get("failure_category")
    )
  if phase_step1_rec and phase_step1_rec.get("run_status") == "OK":
    # Fallback only if standalone cold Step 1 record wasn't run separately (e.g. validation runs).
    raw_val = phase_step1_rec.get("step1_total_seconds")
    if _is_number(raw_val) and float(raw_val) > 0:
      return float(raw_val)
  return "PENDING"


def _effective_step1b_seconds(
    model_info: Mapping[str, Any],
    platform: str,
    v2_data: Mapping[str, Any],
) -> float | str:
  """Returns DeepSeek Step 1b conversion/copy duration or `N/A`/`PENDING`."""
  if not model_info.get("needs_step1b"):
    return "N/A"
  if platform == "v7x":
    rec = v2_data["step1b_convert_by_model"].get(model_info["model_name"])
  else:
    rec = v2_data["step1b_copy_by_pm"].get((platform, model_info["model_name"]))
  if not rec:
    return "PENDING"
  if rec.get("run_status") == "OK" and _is_number(rec.get("step1b_deepseek_total_seconds")):
    return float(rec["step1b_deepseek_total_seconds"])
  return schema_v2.normalize_sheet_status(str(rec.get("run_status") or "ERROR"), rec.get("failure_category"))


def _deciding_to_maxtext_phase(rep_phases: Mapping[str, Mapping[str, Any]]) -> tuple[str, Mapping[str, Any], str]:
  """Returns `(normalized_status, deciding_record, deciding_phase_name)` for one `to_maxtext` repetition."""
  missing_phase = ""
  for p_name in ("to_maxtext_step1", "to_maxtext_step2", "to_maxtext_step3"):
    rec = rep_phases.get(p_name)
    if rec is None:
      missing_phase = missing_phase or p_name
      continue
    if rec.get("run_status") != "OK":
      code = schema_v2.normalize_sheet_status(str(rec.get("run_status") or "ERROR"), rec.get("failure_category"))
      return code, rec, p_name
  if "to_maxtext_step3" in rep_phases:
    return "OK", rep_phases["to_maxtext_step3"], "to_maxtext_step3"
  return "PENDING", next(reversed(rep_phases.values())), missing_phase or "to_maxtext_step1"


def _median_numeric_keys(
    records: Sequence[Mapping[str, Any]],
    columns: Sequence[schema_v2.ColumnSpec],
) -> dict[str, Any]:
  """Computes median values for numeric columns and preserves latest non-numeric metadata."""
  merged: dict[str, Any] = {}
  for col in columns:
    vals = [r.get(col.key) for r in records if r.get(col.key) not in (None, "")]
    if not vals:
      continue
    if col.fmt in schema_v2.NUMERIC_FORMATS:
      nums = [float(v) for v in vals if _is_number(v)]
      if nums:
        med = statistics.median(nums)
        merged[col.key] = (
            int(round(med)) if col.fmt == "n" and all(float(x).is_integer() for x in nums) else _round_num(med, 4)
        )
      else:
        merged[col.key] = vals[-1]
    else:
      merged[col.key] = vals[-1]
  return merged


def _build_stage_row_from_rep(
    *,
    row_id: str,
    pipeline: str,
    platform: str,
    model_name: str,
    scan_mode: str,
    slices: int,
    topology: str,
    rep_records: Sequence[Mapping[str, Any]],
    step1_cold_s: float | str,
) -> dict[str, Any]:
  """Aggregates per-phase JSON records into a `Stages` tab row."""
  cols = schema_v2.STAGES_SD_COLUMNS if pipeline == "safetensors_dynamic" else schema_v2.STAGES_TM_COLUMNS
  row = _empty_row(cols)
  row.update(_median_numeric_keys(rep_records, cols))
  row["row_id"] = row_id
  row["tpu_platform"] = platform
  row["model_name"] = model_name
  row["scan_layers_mode"] = scan_mode
  row["number_of_slices"] = slices
  row["slice_topology"] = topology
  if pipeline == "to_maxtext":
    s3_s = row.get("to_maxtext_step3_total_seconds")
    if _is_number(s3_s):
      row["to_maxtext_load_only_seconds"] = _round_num(float(s3_s), 4)
    s2_s = row.get("to_maxtext_step2_total_seconds")
    if _is_number(step1_cold_s) and _is_number(s2_s) and _is_number(s3_s):
      row["to_maxtext_total_end_to_end_old_formula_seconds"] = _round_num(
          float(step1_cold_s) + float(s2_s) + float(s3_s), 4
      )
    else:
      row["to_maxtext_total_end_to_end_old_formula_seconds"] = "—"
  return row


def _attach_v2_pair_speedups(summary_rows: Sequence[dict[str, Any]]) -> None:
  """Computes `to_maxtext / safetensors_dynamic` speedup ratios onto paired `safetensors_dynamic` rows."""
  by_pair: dict[tuple[str, str, str, int, str], dict[str, dict[str, Any]]] = {}
  for row in summary_rows:
    key = (
        str(row["tpu_platform"]),
        str(row["model_name"]),
        str(row["scan_layers_mode"]),
        int(row["number_of_slices"]),
        str(row["slice_topology"]),
    )
    by_pair.setdefault(key, {})[str(row["pipeline_name"])] = row

  for pair in by_pair.values():
    tm = pair.get("to_maxtext")
    sd = pair.get("safetensors_dynamic")
    if not (tm and sd):
      continue
    if tm.get("run_status") != "OK" or sd.get("run_status") != "OK":
      continue
    tm_nohf = tm.get("total_without_hf_download_seconds")
    sd_nohf = sd.get("total_without_hf_download_seconds")
    tm_load = tm.get("load_into_tpu_hbm_total_seconds")
    sd_load = sd.get("load_into_tpu_hbm_total_seconds")
    tm_e2e = tm.get("total_end_to_end_seconds")
    sd_e2e = sd.get("total_end_to_end_seconds")
    if _is_number(tm_nohf) and _is_number(sd_nohf) and float(sd_nohf) > 0:
      sd["speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic"] = _round_num(
          float(tm_nohf) / float(sd_nohf), 4
      )
    if _is_number(tm_load) and _is_number(sd_load) and float(sd_load) > 0:
      sd["speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2"] = _round_num(
          float(tm_load) / float(sd_load), 4
      )
    if _is_number(tm_e2e) and _is_number(sd_e2e) and float(sd_e2e) > 0:
      sd["speedup_end_to_end_to_maxtext_vs_safetensors_dynamic"] = _round_num(float(tm_e2e) / float(sd_e2e), 4)


def _format_failure_note(deciding_rec: Mapping[str, Any], deciding_phase: str) -> str:
  parts = [f"{deciding_phase}: {deciding_rec.get('run_status')}"]
  if deciding_rec.get("failure_category"):
    parts[0] = f"{deciding_phase}: {deciding_rec['failure_category']}"
  if deciding_rec.get("failed_stage_name"):
    parts.append(f"stage={deciding_rec['failed_stage_name']}")
  if deciding_rec.get("error_summary"):
    parts.append(str(deciding_rec["error_summary"]))
  return " | ".join(parts)


def _decide_log_url(rec: Mapping[str, Any], phase_name: str, ok: bool) -> str:
  if ok:
    return str(rec.get("step3_or_load_log_url") or "")
  key_map = {
      "to_maxtext_step1": "pipeline_step1_log_url",
      "to_maxtext_step2": "step2_log_url",
      "to_maxtext_step3": "step3_or_load_log_url",
      "safetensors_dynamic": "step3_or_load_log_url",
  }
  return str(rec.get(key_map.get(phase_name, "step3_or_load_log_url"), "") or rec.get("step3_or_load_log_url") or "")


def _build_v2_matrix_rows(
    v2_data: Mapping[str, Any],
    models_cfg: Mapping[str, Any],
    *,
    source_prefix: str = "v2",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
  """Builds all planned `Summary`, `Stages: safetensors_dynamic`, and `Stages: to_maxtext` rows."""
  summary_rows: list[dict[str, Any]] = []
  stages_sd_rows: list[dict[str, Any]] = []
  stages_tm_rows: list[dict[str, Any]] = []

  model_order = {m["model_name"]: idx for idx, m in enumerate(models_cfg["models"])}
  planned = list(schema_v2.iter_planned_matrix(models_cfg))
  planned.sort(
      key=lambda item: (
          _PLAT_SORT_ORDER.get(item["platform"], 99),
          model_order.get(item["model_name"], 999),
          _PIPE_SORT_ORDER.get(item["pipeline"], 99),
          0 if item["scan_mode"] == "scanned" else 1,
          item["slices"],
          item["topology"],
      )
  )

  for item in planned:
    plat = item["platform"]
    m_name = item["model_name"]
    tier = item["tier"]
    slices = item["slices"]
    topo = item["topology"]
    scan_mode = item["scan_mode"]
    pipeline = item["pipeline"]
    model_info = item["model"]
    profile = platforms.get_profile(plat)
    hosts_per_slice = profile.hosts_per_slice(topo)
    devs_per_slice = profile.devices_per_slice(topo)
    row_id = schema_v2.make_row_id(source_prefix, plat, m_name, scan_mode, pipeline, slices, topo)

    sum_row = _empty_row(schema_v2.SUMMARY_COLUMNS)
    sum_row.update(
        {
            "tpu_platform": plat,
            "model_test_tier": tier,
            "model_name": m_name,
            "pipeline_name": pipeline,
            "scan_layers_mode": scan_mode,
            "number_of_slices": slices,
            "slice_topology": topo,
            "hosts_per_slice": hosts_per_slice,
            "tpu_devices_per_slice": devs_per_slice,
            "total_tpu_devices": slices * devs_per_slice,
            "hf_checkpoint_dtype": model_info.get("hf_dtype", ""),
            "loaded_checkpoint_dtype": model_info.get("loaded_dtype", ""),
            "maxtext_load_weight_dtype": model_info.get("maxtext_weight_dtype", "bfloat16"),
            "checkpoint_weights_kind": "real",
            "checkpoint_size_gb": model_info.get("loaded_bf16_gb", ""),
            "speedup_end_to_end_to_maxtext_vs_safetensors_dynamic": "—",
            "speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic": "—",
            "speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2": "—",
            "row_id": row_id,
        }
    )

    step1_rec = v2_data["step1_by_pm"].get((plat, m_name))
    step1_s = _effective_step1_seconds(step1_rec, None)
    step1b_s = _effective_step1b_seconds(model_info, plat, v2_data)
    sum_row["step1_total_seconds"] = _round_num(step1_s, 4) if _is_number(step1_s) else step1_s
    sum_row["step1b_deepseek_total_seconds"] = _round_num(step1b_s, 4) if _is_number(step1b_s) else step1b_s

    cfg_key = (plat, m_name, scan_mode, slices, topo, pipeline)
    reps_map: Mapping[int, Mapping[str, Mapping[str, Any]]] = v2_data["phases_by_cfg"].get(cfg_key, {})

    if not reps_map:
      sum_row["run_status"] = "PENDING"
      if pipeline == "safetensors_dynamic":
        sum_row["to_maxtext_step1_total_seconds"] = "N/A"
        sum_row["to_maxtext_step2_total_seconds"] = "N/A"
      else:
        sum_row["to_maxtext_step1_total_seconds"] = "PENDING"
        sum_row["to_maxtext_step2_total_seconds"] = "PENDING"
      sum_row["load_into_tpu_hbm_total_seconds"] = "PENDING"
      sum_row["setup_total_before_load_seconds"] = "PENDING"
      sum_row["total_without_hf_download_seconds"] = "—"
      sum_row["total_end_to_end_seconds"] = "—"
      sum_row["memory_peak_host_ram_container_gb"] = "PENDING"
      sum_row["memory_tpu_hbm_peak_per_device_gb"] = "PENDING"
      sum_row["repetitions_count"] = 0
      sum_row["note"] = "Queued / awaiting execution."
      summary_rows.append(sum_row)
      continue

    ok_stage_rows: list[dict[str, Any]] = []
    ok_rep_records: list[Mapping[str, Any]] = []
    ok_tm_peak_rams: list[float] = []
    latest_status = "PENDING"
    latest_deciding_rec: Mapping[str, Any] = {}
    latest_deciding_phase = pipeline
    latest_stage_row: dict[str, Any] | None = None

    for rep_idx in sorted(reps_map):
      rep_phases = reps_map[rep_idx]
      if pipeline == "safetensors_dynamic":
        rec = rep_phases.get("safetensors_dynamic")
        if rec is None:
          continue
        rep_status = (
            "OK"
            if rec.get("run_status") == "OK"
            else schema_v2.normalize_sheet_status(str(rec.get("run_status") or "ERROR"), rec.get("failure_category"))
        )
        rep_recs: list[Mapping[str, Any]] = [rec]
        deciding_rec, deciding_phase = rec, "safetensors_dynamic"
      else:
        rep_status, deciding_rec, deciding_phase = _deciding_to_maxtext_phase(rep_phases)
        rep_recs = [
            rep_phases[p] for p in ("to_maxtext_step1", "to_maxtext_step2", "to_maxtext_step3") if p in rep_phases
        ]

      s1_fallback = _effective_step1_seconds(step1_rec, rep_phases.get("to_maxtext_step1") or deciding_rec)
      if _is_number(s1_fallback):
        step1_s = s1_fallback
        sum_row["step1_total_seconds"] = _round_num(float(s1_fallback), 4)

      st_row = _build_stage_row_from_rep(
          row_id=row_id,
          pipeline=pipeline,
          platform=plat,
          model_name=m_name,
          scan_mode=scan_mode,
          slices=slices,
          topology=topo,
          rep_records=rep_recs,
          step1_cold_s=step1_s,
      )
      latest_stage_row = st_row
      latest_status = rep_status
      latest_deciding_rec = deciding_rec
      latest_deciding_phase = deciding_phase

      if rep_status == "OK":
        ok_stage_rows.append(st_row)
        ok_rep_records.append(deciding_rec)
        if pipeline == "to_maxtext":
          ram_candidates = [
              float(r["memory_peak_host_ram_container_gb"])
              for r in rep_recs
              if _is_number(r.get("memory_peak_host_ram_container_gb"))
          ]
          if _is_number(st_row.get("memory_to_maxtext_step2_peak_host_ram_container_gb")):
            ram_candidates.append(float(st_row["memory_to_maxtext_step2_peak_host_ram_container_gb"]))
          if ram_candidates:
            ok_tm_peak_rams.append(max(ram_candidates))

    if ok_stage_rows:
      target_cols = schema_v2.STAGES_SD_COLUMNS if pipeline == "safetensors_dynamic" else schema_v2.STAGES_TM_COLUMNS
      agg_stage = _build_stage_row_from_rep(
          row_id=row_id,
          pipeline=pipeline,
          platform=plat,
          model_name=m_name,
          scan_mode=scan_mode,
          slices=slices,
          topology=topo,
          rep_records=ok_stage_rows,
          step1_cold_s=step1_s,
      )
      dec_rec = ok_rep_records[-1]
      sum_row["run_status"] = "OK"
      sum_row["repetitions_count"] = len(ok_stage_rows)
      load_key = (
          "safetensors_dynamic_step2_total_seconds"
          if pipeline == "safetensors_dynamic"
          else "to_maxtext_step3_total_seconds"
      )
      load_vals = [float(r[load_key]) for r in ok_stage_rows if _is_number(r.get(load_key))]
      if len(load_vals) > 1:
        sum_row["load_time_range_min_max_seconds"] = f"{min(load_vals):.2f}\u2013{max(load_vals):.2f}"

      backend = dec_rec.get("jax_backend_platform") or ""
      kind = dec_rec.get("tpu_device_kind") or ""
      sum_row["load_backend_and_tpu_device_kind"] = f"{backend} · {kind}" if (backend or kind) else ""
      sum_row["hf_checkpoint_dtype"] = dec_rec.get("hf_checkpoint_dtype") or sum_row["hf_checkpoint_dtype"]
      sum_row["loaded_checkpoint_dtype"] = dec_rec.get("loaded_checkpoint_dtype") or sum_row["loaded_checkpoint_dtype"]
      sum_row["maxtext_load_weight_dtype"] = (
          dec_rec.get("maxtext_load_weight_dtype") or sum_row["maxtext_load_weight_dtype"]
      )
      if _is_number(dec_rec.get("checkpoint_size_gb")):
        sum_row["checkpoint_size_gb"] = _round_num(float(dec_rec["checkpoint_size_gb"]), 4)

      load_s = agg_stage.get(load_key)
      setup_s = agg_stage.get("setup_total_before_load_seconds")
      sum_row["load_into_tpu_hbm_total_seconds"] = load_s if _is_number(load_s) else "—"
      sum_row["setup_total_before_load_seconds"] = setup_s if _is_number(setup_s) else ""
      sum_row["memory_tpu_hbm_peak_per_device_gb"] = agg_stage.get("memory_tpu_hbm_peak_per_device_gb", "")

      if pipeline == "safetensors_dynamic":
        sum_row["to_maxtext_step1_total_seconds"] = "N/A"
        sum_row["to_maxtext_step2_total_seconds"] = "N/A"
        sd_s1 = agg_stage.get("safetensors_dynamic_step1_total_seconds")
        sum_row["memory_peak_host_ram_container_gb"] = agg_stage.get("memory_peak_host_ram_container_gb", "")
        if _is_number(sd_s1) and _is_number(load_s):
          nohf = float(sd_s1) + float(load_s)
          sum_row["total_without_hf_download_seconds"] = _round_num(nohf, 4)
        else:
          sum_row["total_without_hf_download_seconds"] = "—"
      else:
        tm_s1 = agg_stage.get("to_maxtext_step1_total_seconds")
        tm_s2 = agg_stage.get("to_maxtext_step2_total_seconds")
        sum_row["to_maxtext_step1_total_seconds"] = tm_s1 if _is_number(tm_s1) else ""
        sum_row["to_maxtext_step2_total_seconds"] = tm_s2 if _is_number(tm_s2) else ""
        sum_row["memory_peak_host_ram_container_gb"] = (
            _round_num(statistics.median(ok_tm_peak_rams), 4)
            if ok_tm_peak_rams
            else agg_stage.get("memory_peak_host_ram_container_gb", "")
        )
        if _is_number(tm_s1) and _is_number(tm_s2) and _is_number(load_s):
          sum_row["total_without_hf_download_seconds"] = _round_num(float(tm_s1) + float(tm_s2) + float(load_s), 4)
        else:
          sum_row["total_without_hf_download_seconds"] = "—"

      nohf_val = sum_row["total_without_hf_download_seconds"]
      if _is_number(step1_s) and _is_number(nohf_val):
        e2e = float(step1_s) + float(nohf_val)
        if model_info.get("needs_step1b"):
          if _is_number(step1b_s):
            sum_row["total_end_to_end_seconds"] = _round_num(e2e + float(step1b_s), 4)
          else:
            sum_row["total_end_to_end_seconds"] = "—"
        else:
          sum_row["total_end_to_end_seconds"] = _round_num(e2e, 4)
      else:
        sum_row["total_end_to_end_seconds"] = "—"

      sum_row["log_url"] = gcs_to_https(_decide_log_url(dec_rec, pipeline, ok=True))
      if len(ok_stage_rows) != len(reps_map):
        sum_row["note"] = f"{len(ok_stage_rows)}/{len(reps_map)} repetitions OK; non-OK attempt excluded from median."
      del target_cols
      if pipeline == "safetensors_dynamic":
        stages_sd_rows.append(agg_stage)
      else:
        stages_tm_rows.append(agg_stage)
    else:
      sum_row["run_status"] = latest_status
      sum_row["repetitions_count"] = len(reps_map)
      sum_row["total_without_hf_download_seconds"] = "—"
      sum_row["total_end_to_end_seconds"] = "—"
      sum_row["log_url"] = gcs_to_https(_decide_log_url(latest_deciding_rec, latest_deciding_phase, ok=False))
      sum_row["note"] = _format_failure_note(latest_deciding_rec, latest_deciding_phase)
      if pipeline == "safetensors_dynamic":
        sum_row["to_maxtext_step1_total_seconds"] = "N/A"
        sum_row["to_maxtext_step2_total_seconds"] = "N/A"
        sum_row["load_into_tpu_hbm_total_seconds"] = latest_status
        sum_row["setup_total_before_load_seconds"] = (
            latest_deciding_rec.get("setup_total_before_load_seconds")
            if _is_number(latest_deciding_rec.get("setup_total_before_load_seconds"))
            else latest_status
        )
        sum_row["memory_peak_host_ram_container_gb"] = (
            latest_deciding_rec.get("memory_peak_host_ram_container_gb")
            if _is_number(latest_deciding_rec.get("memory_peak_host_ram_container_gb"))
            else latest_status
        )
        sum_row["memory_tpu_hbm_peak_per_device_gb"] = latest_status
      else:
        if latest_deciding_phase == "to_maxtext_step1":
          sum_row["to_maxtext_step1_total_seconds"] = latest_status
          sum_row["to_maxtext_step2_total_seconds"] = "BLOCKED"
          sum_row["load_into_tpu_hbm_total_seconds"] = "BLOCKED"
          sum_row["setup_total_before_load_seconds"] = "BLOCKED"
          sum_row["memory_tpu_hbm_peak_per_device_gb"] = "BLOCKED"
        elif latest_deciding_phase == "to_maxtext_step2":
          sum_row["to_maxtext_step1_total_seconds"] = (
              latest_stage_row.get("to_maxtext_step1_total_seconds", "") if latest_stage_row else ""
          )
          sum_row["to_maxtext_step2_total_seconds"] = latest_status
          sum_row["load_into_tpu_hbm_total_seconds"] = "BLOCKED"
          sum_row["setup_total_before_load_seconds"] = "BLOCKED"
          sum_row["memory_tpu_hbm_peak_per_device_gb"] = "BLOCKED"
          ram_s2 = latest_deciding_rec.get("memory_peak_host_ram_container_gb")
          sum_row["memory_peak_host_ram_container_gb"] = (
              _round_num(float(ram_s2), 4) if _is_number(ram_s2) else latest_status
          )
        else:
          sum_row["to_maxtext_step1_total_seconds"] = (
              latest_stage_row.get("to_maxtext_step1_total_seconds", "") if latest_stage_row else ""
          )
          sum_row["to_maxtext_step2_total_seconds"] = (
              latest_stage_row.get("to_maxtext_step2_total_seconds", "") if latest_stage_row else ""
          )
          sum_row["load_into_tpu_hbm_total_seconds"] = latest_status
          sum_row["setup_total_before_load_seconds"] = (
              latest_deciding_rec.get("setup_total_before_load_seconds")
              if _is_number(latest_deciding_rec.get("setup_total_before_load_seconds"))
              else latest_status
          )
          sum_row["memory_peak_host_ram_container_gb"] = (
              latest_deciding_rec.get("memory_peak_host_ram_container_gb")
              if _is_number(latest_deciding_rec.get("memory_peak_host_ram_container_gb"))
              else latest_status
          )
          sum_row["memory_tpu_hbm_peak_per_device_gb"] = latest_status
      if latest_stage_row is not None:
        if pipeline == "safetensors_dynamic":
          stages_sd_rows.append(latest_stage_row)
        else:
          stages_tm_rows.append(latest_stage_row)

    summary_rows.append(sum_row)

  _attach_v2_pair_speedups(summary_rows)
  return summary_rows, stages_sd_rows, stages_tm_rows


def _build_runs_rows(
    v2_data: Mapping[str, Any],
    legacy_runs_rows: Sequence[Mapping[str, Any]],
    models_by_name: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
  """Builds `Runs` tab rows from executed v2 phase JSONs and legacy v1 run records."""
  planned_ids = {entry["row_id"] for entry in schema_v2.iter_planned_matrix()}
  rows: list[dict[str, Any]] = []
  sorted_recs = sorted(
      v2_data["runs_records"],
      key=lambda r: (
          _PLAT_SORT_ORDER.get(str(r.get("tpu_platform") or ""), 9),
          models_by_name.get(str(r.get("model_name") or ""), {}).get("_idx", 999),
          str(r.get("phase_started_at_utc") or ""),
          str(r.get("phase_name") or ""),
      ),
  )
  for rec in sorted_recs:
    row = _empty_row(schema_v2.RUNS_COLUMNS)
    plat = str(rec.get("tpu_platform") or "")
    model = str(rec.get("model_name") or "")
    phase = str(rec.get("phase_name") or "")
    scan = str(rec.get("scan_layers_mode") or "")
    pipe = str(rec.get("pipeline_name") or _PHASE_PIPELINE_MAP.get(phase, ""))
    slices = int(rec.get("number_of_slices") or 1)
    topo = str(rec.get("slice_topology") or "")
    candidate_id = (
        schema_v2.make_row_id("v2", plat, model, scan, pipe, slices, topo) if (pipe and topo and scan) else ""
    )
    row_id = candidate_id if candidate_id in planned_ids else f"{phase}|{plat}|{model}|{rec.get('run_id', '')}"

    for col in schema_v2.RUNS_COLUMNS:
      if col.key in rec and rec[col.key] not in (None, ""):
        val = rec[col.key]
        if col.fmt == "url":
          row[col.key] = gcs_to_https(val, is_dir=col.key == "all_hosts_log_folder_url")
        elif col.fmt in schema_v2.NUMERIC_FORMATS and _is_number(val):
          row[col.key] = _round_num(float(val), 4)
        else:
          row[col.key] = val

    env_c = rec.get("env_cluster") or {}
    hw = rec.get("host_hardware") or {}
    pkgs = rec.get("package_versions") or {}
    cv_outer = rec.get("code_version") or {}
    cv_inner = cv_outer.get("code_version") if isinstance(cv_outer.get("code_version"), Mapping) else {}
    cv = {**cv_inner, **cv_outer}
    for key in (
        "git_branch_name",
        "git_commit",
        "git_working_tree_clean",
        "code_tarball_sha256",
    ):
      if row.get(key) in (None, "") and cv.get(key) not in (None, ""):
        row[key] = cv[key]
    for key, val in pkgs.items():
      if key in row and row.get(key) in (None, "") and val not in (None, ""):
        row[key] = val
    if not row.get("host_ram_gb_measured") and _is_number(hw.get("host_ram_gb")):
      row["host_ram_gb_measured"] = _round_num(float(hw["host_ram_gb"]), 2)
    if not row.get("host_vcpu_count") and _is_number(hw.get("host_vcpu_count")):
      row["host_vcpu_count"] = int(hw["host_vcpu_count"])
    if not row.get("host_machine_type") and env_c.get("host_machine_type"):
      row["host_machine_type"] = env_c["host_machine_type"]
    if not row.get("gke_cluster_name") and env_c.get("gke_cluster_name"):
      row["gke_cluster_name"] = env_c["gke_cluster_name"]

    row["row_id"] = row_id
    m_info = models_by_name.get(model, {})
    if not row.get("model_parameter_count_billions") and m_info.get("params_b"):
      row["model_parameter_count_billions"] = m_info["params_b"]
    if not row.get("model_architecture") and m_info.get("architecture"):
      row["model_architecture"] = m_info["architecture"]
    if not row.get("model_test_tier") and m_info.get("tier"):
      row["model_test_tier"] = m_info["tier"]
    rows.append(row)

  rows.extend(dict(r) for r in legacy_runs_rows)
  return rows


def _summary_cfg_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int, str]:
  """Returns the canonical `(platform, model, pipeline, scan_mode, slices, topology)` key for a Summary row."""
  slices = int(row.get("number_of_slices") or 0) if _is_number(row.get("number_of_slices")) else 0
  return (
      str(row.get("tpu_platform") or ""),
      str(row.get("model_name") or ""),
      str(row.get("pipeline_name") or ""),
      str(row.get("scan_layers_mode") or ""),
      slices,
      str(row.get("slice_topology") or ""),
  )


def _merge_v2_and_legacy_summary_rows(
    summary_v2: Sequence[Mapping[str, Any]],
    legacy_summary: Sequence[Mapping[str, Any]],
    models_by_name: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
  """Merges overlapping `v2` (`PENDING`) and `legacy` v1 rows and orders by `Model -> Pipeline -> Scan layers`."""
  legacy_by_key: dict[tuple[str, str, str, str, int, str], dict[str, Any]] = {}
  for leg in legacy_summary:
    key = _summary_cfg_key(leg)
    prev = legacy_by_key.get(key)
    if prev is None or (leg.get("run_status") == "OK" and prev.get("run_status") != "OK"):
      legacy_by_key[key] = dict(leg)

  merged: list[dict[str, Any]] = []
  seen_keys: set[tuple[str, str, str, str, int, str]] = set()
  for raw_v2 in summary_v2:
    row = dict(raw_v2)
    key = _summary_cfg_key(row)
    seen_keys.add(key)
    row["_v2_run_status"] = str(row.get("run_status") or "PENDING")
    leg = legacy_by_key.get(key)
    if row["_v2_run_status"] == "PENDING" and leg and leg.get("run_status") not in ("", "PENDING"):
      v2_s1 = row.get("step1_total_seconds")
      v2_s1b = row.get("step1b_deepseek_total_seconds")
      v2_tier = row.get("model_test_tier")
      for col in schema_v2.SUMMARY_COLUMNS:
        if leg.get(col.key) not in (None, ""):
          row[col.key] = leg[col.key]
      row["model_test_tier"] = v2_tier
      if _is_number(v2_s1):
        row["step1_total_seconds"] = _round_num(float(v2_s1), 4)
      if _is_number(v2_s1b) or v2_s1b in ("N/A", "PENDING"):
        row["step1b_deepseek_total_seconds"] = v2_s1b
      if row.get("pipeline_name") == "safetensors_dynamic":
        row["to_maxtext_step1_total_seconds"] = "N/A"
        row["to_maxtext_step2_total_seconds"] = "N/A"
      nohf = row.get("total_without_hf_download_seconds")
      s1_val = row.get("step1_total_seconds")
      m_info = models_by_name.get(str(row.get("model_name") or ""), {})
      if _is_number(s1_val) and _is_number(nohf):
        e2e = float(s1_val) + float(nohf)
        if m_info.get("needs_step1b"):
          s1b_val = row.get("step1b_deepseek_total_seconds")
          row["total_end_to_end_seconds"] = _round_num(e2e + float(s1b_val), 4) if _is_number(s1b_val) else "—"
        else:
          row["total_end_to_end_seconds"] = _round_num(e2e, 4)
      leg_note = str(leg.get("note") or "")
      row["note"] = f"v1 baseline (v2 re-run queued); {leg_note}" if leg_note else "v1 baseline (v2 re-run queued)"
    merged.append(row)

  for key, leg in legacy_by_key.items():
    if key in seen_keys:
      continue
    leg_row = dict(leg)
    leg_row["_v2_run_status"] = "LEGACY_ONLY"
    if leg_row.get("pipeline_name") == "safetensors_dynamic":
      leg_row["to_maxtext_step1_total_seconds"] = "N/A"
      leg_row["to_maxtext_step2_total_seconds"] = "N/A"
    merged.append(leg_row)

  _attach_v2_pair_speedups(merged)

  def _summary_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    m_name = str(row.get("model_name") or "")
    plat = str(row.get("tpu_platform") or "")
    pipe = str(row.get("pipeline_name") or "")
    scan = str(row.get("scan_layers_mode") or "")
    slices = int(row.get("number_of_slices") or 0) if _is_number(row.get("number_of_slices")) else 0
    topo = str(row.get("slice_topology") or "")
    model_order_idx = models_by_name.get(m_name, {}).get("_idx", 999)
    return (
        _PLAT_SORT_ORDER.get(plat, 99),
        model_order_idx,
        m_name if model_order_idx == 999 else "",
        _PIPE_SORT_ORDER.get(pipe, 99),
        0 if scan == "scanned" else 1,
        slices,
        topo,
        str(row.get("row_id") or ""),
    )

  merged.sort(key=_summary_sort_key)
  return merged


def build_v2_dataset(
    *,
    v2_roots: Sequence[str] = (),
    include_smoke: bool = False,
    include_val: bool = False,
    p2_roots: Sequence[str] = (),
    p1_roots: Sequence[str] = (),
    variant_roots: Sequence[tuple[str, Sequence[str]]] = (),
    before_roots: Sequence[str] = (),
    multihost_csv: str | None = None,
    cold_s1_overrides: Mapping[str, float] | None = None,
) -> V2Dataset:
  """Combines v2 GCS mirrors, planned matrix rows, and optional legacy v1 datasets."""
  models_cfg = schema_v2.load_models_v2()
  models_by_name: dict[str, dict[str, Any]] = {}
  for idx, m in enumerate(models_cfg["models"]):
    entry = dict(m)
    entry["_idx"] = idx
    models_by_name[m["model_name"]] = entry

  if p2_roots or p1_roots or before_roots or (multihost_csv and os.path.exists(multihost_csv)):
    v1_ds = legacy_v1.adapt_v1_dataset(
        p2_roots=p2_roots,
        p1_roots=p1_roots,
        variant_roots=variant_roots,
        before_roots=before_roots,
        multihost_csv=multihost_csv,
        cold_s1_overrides=cold_s1_overrides,
        models_cfg=models_cfg,
    )
  else:
    v1_ds = legacy_v1.LegacyV1Dataset([], [], [], [], [])

  v2_data = scan_v2_roots(v2_roots, include_smoke=include_smoke, include_val=include_val)
  step1_rows = _build_step1_rows(v2_data, v1_ds.step1_rows, models_by_name)
  summary_v2, stages_sd_v2, stages_tm_v2 = _build_v2_matrix_rows(v2_data, models_cfg)
  runs_rows = _build_runs_rows(v2_data, v1_ds.runs_rows, models_by_name)
  merged_summary = _merge_v2_and_legacy_summary_rows(summary_v2, v1_ds.summary_rows, models_by_name)

  return V2Dataset(
      summary_rows=merged_summary,
      step1_rows=step1_rows,
      stages_sd_rows=stages_sd_v2 + list(v1_ds.stages_sd_rows),
      stages_tm_rows=stages_tm_v2 + list(v1_ds.stages_tm_rows),
      runs_rows=runs_rows,
  )


def _format_cell_for_raw(
    val: Any, col: schema_v2.ColumnSpec, row_idx: int, links_out: list[tuple[str, int, str, str]]
) -> Any:
  """Converts timestamps/booleans/URLs for `gsheets mutate batch --value-input RAW`."""
  if val is None or val == "":
    return ""
  if col.fmt == "url":
    url_str = str(val).strip()
    if url_str:
      label = "dir" if col.key == "all_hosts_log_folder_url" else ("json" if col.key == "result_json_url" else "log")
      links_out.append((col.key, row_idx, url_str, label))
    return ""
  if col.fmt == "ts":
    return sheets_serial(val)
  if col.fmt == "bool":
    if isinstance(val, bool):
      return val
    return str(val).upper() == "TRUE"
  return val


def _tab_sheet_range_prefix(tab_name: str) -> str:
  return f"'{tab_name}'!" if any(ch in tab_name for ch in " :") else f"{tab_name}!"


def _write_csv(path: str, rows: Iterable[Mapping[str, Any]], columns: Sequence[schema_v2.ColumnSpec]) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  keys = [c.key for c in columns]
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: row.get(k, "") for k in keys})


def build_gsheets_batches(
    dataset: V2Dataset,
    *,
    impl_updates: Sequence[Mapping[str, Any]] = (),
    now_utc: datetime.datetime | None = None,
    chunk_size: int = 400,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Builds `(raw_ops, formula_ops)` ready for `gsheets mutate batch`."""
  raw_ops: list[dict[str, Any]] = []
  formula_ops: list[dict[str, Any]] = []
  ts_now = sheets_serial(now_utc or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0))

  tabs = (
      ("Summary", dataset.summary_rows, schema_v2.SUMMARY_COLUMNS),
      ("Step 1 downloads", dataset.step1_rows, schema_v2.STEP1_COLUMNS),
      ("Stages: safetensors_dynamic", dataset.stages_sd_rows, schema_v2.STAGES_SD_COLUMNS),
      ("Stages: to_maxtext", dataset.stages_tm_rows, schema_v2.STAGES_TM_COLUMNS),
      ("Runs", dataset.runs_rows, schema_v2.RUNS_COLUMNS),
  )

  for tab_name, rows, cols in tabs:
    prefix = _tab_sheet_range_prefix(tab_name)
    end_col = col_letter(len(cols) - 1)
    raw_ops.append({"op": "write", "range": f"{prefix}A1:{end_col}1", "data": [[c.header for c in cols]]})
    raw_ops.append({"op": "clear", "range": f"{prefix}A2:{end_col}2000"})

    if not rows:
      continue

    links: list[tuple[str, int, str, str]] = []
    matrix: list[list[Any]] = []
    for r_offset, row in enumerate(rows):
      sheet_row = 2 + r_offset
      matrix.append([_format_cell_for_raw(row.get(c.key, ""), c, sheet_row, links) for c in cols])

    for start in range(0, len(matrix), chunk_size):
      batch = matrix[start : start + chunk_size]
      r0 = 2 + start
      r1 = r0 + len(batch) - 1
      raw_ops.append({"op": "write", "range": f"{prefix}A{r0}:{end_col}{r1}", "data": batch})

    col_index = {c.key: i for i, c in enumerate(cols)}
    by_col: dict[str, dict[int, str]] = {}
    for c_key, sheet_row, url_str, label in links:
      safe_url = url_str.replace('"', "%22")
      by_col.setdefault(c_key, {})[sheet_row] = f'=HYPERLINK("{safe_url}","{label}")'

    for c_key, row_map in by_col.items():
      letter = col_letter(col_index[c_key])
      sorted_rows = sorted(row_map)
      run_start = sorted_rows[0]
      run_vals: list[list[str]] = [[row_map[run_start]]]
      prev = run_start
      for sr in sorted_rows[1:]:
        if sr == prev + 1:
          run_vals.append([row_map[sr]])
        else:
          formula_ops.append({"op": "write", "range": f"{prefix}{letter}{run_start}:{letter}{prev}", "data": run_vals})
          run_start = sr
          run_vals = [[row_map[sr]]]
        prev = sr
      formula_ops.append({"op": "write", "range": f"{prefix}{letter}{run_start}:{letter}{prev}", "data": run_vals})

  raw_ops.append({"op": "write", "range": PROGRESS_UPDATED_CELL, "data": [[ts_now]]})
  raw_ops.append({"op": "write", "range": README_UPDATED_CELL, "data": [[ts_now]]})

  step1_src_col = col_letter([c.key for c in schema_v2.STEP1_COLUMNS].index("step1_measured_by_run_id"))
  for r in (7, 8, 9):
    formula_ops.append(
        {
            "op": "write",
            "range": f"Progress!D{r}",
            "data": [
                [
                    "=COUNTIFS('Step 1 downloads'!$A$2:$A,$A"
                    f"{r},'Step 1 downloads'!${step1_src_col}$2:${step1_src_col},\"<>v1_tpu7x\")"
                ]
            ],
        }
    )

  planned_v2_rows = [
      r for r in dataset.summary_rows if str(r.get("_v2_run_status") or r.get("run_status")) != "LEGACY_ONLY"
  ]
  for r_idx, plat in ((7, "v7x"), (8, "v6e"), (9, "v5e")):
    p_rows = [r for r in planned_v2_rows if r.get("tpu_platform") == plat]
    ok_cnt = sum(1 for r in p_rows if str(r.get("_v2_run_status") or r.get("run_status")) == "OK")
    fail_cnt = sum(
        1 for r in p_rows if str(r.get("_v2_run_status") or r.get("run_status")) in schema_v2.FAILURE_CATEGORY_CODES
    )
    skip_cnt = sum(
        1
        for r in p_rows
        if str(r.get("_v2_run_status") or r.get("run_status")) in ("NOT RUN", "SKIPPED", "BLOCKED", "N/A")
    )
    pend_cnt = sum(1 for r in p_rows if str(r.get("_v2_run_status") or r.get("run_status")) == "PENDING")
    raw_ops.append(
        {"op": "write", "range": f"Progress!F{r_idx}:I{r_idx}", "data": [[ok_cnt, fail_cnt, skip_cnt, pend_cnt]]}
    )

  for r_idx, tier in ((14, "S"), (15, "A"), (16, "B"), (17, "C")):
    for col_l, plat in (("C", "v7x"), ("E", "v6e"), ("G", "v5e")):
      done_cnt = sum(
          1
          for r in planned_v2_rows
          if r.get("tpu_platform") == plat
          and r.get("model_test_tier") == tier
          and str(r.get("_v2_run_status") or r.get("run_status")) not in ("", "PENDING", "LEGACY_ONLY")
      )
      raw_ops.append({"op": "write", "range": f"Progress!{col_l}{r_idx}", "data": [[done_cnt]]})

  for upd in impl_updates:
    task_id = str(upd.get("task") or "")
    row_num = _PROGRESS_IMPL_ROWS.get(task_id)
    if row_num is None:
      raise ValueError(f"Unknown Progress task id {task_id!r}")
    status = str(upd.get("status") or "Done")
    started = sheets_serial(upd.get("started_utc") or "")
    finished = sheets_serial(upd.get("finished_utc") or "")
    commit = str(upd.get("commit") or "")
    notes = str(upd.get("notes") or "")
    raw_ops.append(
        {
            "op": "write",
            "range": f"Progress!D{row_num}:H{row_num}",
            "data": [[status, started, finished, commit, notes]],
        }
    )

  return raw_ops, formula_ops


def build_summary_format_requests(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    sheet_id: int = 300,
) -> dict[str, Any]:
  """Builds a Google Sheets `BatchUpdateSpreadsheetRequest` merging hierarchical group cells in `Summary`."""
  total_cols = len(schema_v2.SUMMARY_COLUMNS)
  requests: list[dict[str, Any]] = [
      {
          "unmergeCells": {
              "range": {
                  "sheetId": sheet_id,
                  "startRowIndex": 1,
                  "endRowIndex": 2000,
                  "startColumnIndex": 0,
                  "endColumnIndex": 5,
              }
          }
      },
      {
          "updateBorders": {
              "range": {
                  "sheetId": sheet_id,
                  "startRowIndex": 1,
                  "endRowIndex": 2000,
                  "startColumnIndex": 0,
                  "endColumnIndex": total_cols,
              },
              "top": {"style": "NONE"},
              "bottom": {"style": "NONE"},
              "innerHorizontal": {"style": "NONE"},
          }
      },
  ]

  if not summary_rows:
    return {"requests": requests}

  end_row = 1 + len(summary_rows)
  requests.extend(
      [
          {
              "repeatCell": {
                  "range": {
                      "sheetId": sheet_id,
                      "startRowIndex": 1,
                      "endRowIndex": end_row,
                      "startColumnIndex": 0,
                      "endColumnIndex": 5,
                  },
                  "cell": {
                      "userEnteredFormat": {
                          "verticalAlignment": "MIDDLE",
                      }
                  },
                  "fields": "userEnteredFormat.verticalAlignment",
              }
          },
          {
              "repeatCell": {
                  "range": {
                      "sheetId": sheet_id,
                      "startRowIndex": 1,
                      "endRowIndex": end_row,
                      "startColumnIndex": 0,
                      "endColumnIndex": 2,
                  },
                  "cell": {
                      "userEnteredFormat": {
                          "verticalAlignment": "MIDDLE",
                          "textFormat": {"bold": True},
                      }
                  },
                  "fields": "userEnteredFormat.verticalAlignment,userEnteredFormat.textFormat.bold",
              }
          },
      ]
  )

  group_specs = (
      (0, ("tpu_platform",)),
      (1, ("tpu_platform", "model_name")),
      (2, ("tpu_platform", "model_name", "model_test_tier")),
      (3, ("tpu_platform", "model_name", "pipeline_name")),
      (4, ("tpu_platform", "model_name", "pipeline_name", "scan_layers_mode")),
  )
  for col_idx, key_fields in group_specs:
    run_start = 0
    prev_key = tuple(str(summary_rows[0].get(k) or "") for k in key_fields)
    for idx in range(1, len(summary_rows) + 1):
      cur_key = tuple(str(summary_rows[idx].get(k) or "") for k in key_fields) if idx < len(summary_rows) else None
      if cur_key != prev_key:
        if idx - run_start >= 2:
          requests.append(
              {
                  "mergeCells": {
                      "range": {
                          "sheetId": sheet_id,
                          "startRowIndex": 1 + run_start,
                          "endRowIndex": 1 + idx,
                          "startColumnIndex": col_idx,
                          "endColumnIndex": col_idx + 1,
                      },
                      "mergeType": "MERGE_COLUMNS",
                  }
              }
          )
        if idx < len(summary_rows) and cur_key is not None:
          if col_idx == 0:
            requests.append(
                {
                    "updateBorders": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1 + idx - 1,
                            "endRowIndex": 1 + idx,
                            "startColumnIndex": 0,
                            "endColumnIndex": total_cols,
                        },
                        "bottom": {
                            "style": "SOLID_THICK",
                            "color": {"red": 0.25, "green": 0.25, "blue": 0.25},
                        },
                    }
                }
            )
          elif col_idx == 1 and cur_key[0] == prev_key[0]:
            requests.append(
                {
                    "updateBorders": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1 + idx - 1,
                            "endRowIndex": 1 + idx,
                            "startColumnIndex": 1,
                            "endColumnIndex": total_cols,
                        },
                        "bottom": {
                            "style": "SOLID_MEDIUM",
                            "color": {"red": 0.45, "green": 0.45, "blue": 0.45},
                        },
                    }
                }
            )
          elif col_idx == 3 and cur_key[:2] == prev_key[:2]:
            requests.append(
                {
                    "updateBorders": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 1 + idx - 1,
                            "endRowIndex": 1 + idx,
                            "startColumnIndex": 3,
                            "endColumnIndex": total_cols,
                        },
                        "bottom": {
                            "style": "SOLID",
                            "color": {"red": 0.72, "green": 0.72, "blue": 0.72},
                        },
                    }
                }
            )
        run_start = idx
        prev_key = cur_key  # type: ignore[assignment]

  return {"requests": requests}


def apply_batches(spreadsheet_id: str, raw_path: str, formulas_path: str, format_path: str = "") -> None:
  """Uploads RAW, USER_ENTERED, and raw-batch formatting updates via `gsheets mutate`."""
  if format_path and os.path.exists(format_path):
    subprocess.run(
        [
            GSHEETS_BIN,
            "mutate",
            "unmerge",
            spreadsheet_id,
            "--sheet-id",
            "300",
            "--start-row",
            "1",
            "--end-row",
            "2000",
            "--start-col",
            "0",
            "--end-col",
            "5",
        ],
        check=True,
    )
  subprocess.run(
      [GSHEETS_BIN, "mutate", "batch", spreadsheet_id, "-f", raw_path, "--value-input", "RAW"],
      check=True,
  )
  with open(formulas_path, "r", encoding="utf-8") as f:
    has_formulas = bool(json.load(f))
  if has_formulas:
    subprocess.run(
        [GSHEETS_BIN, "mutate", "batch", spreadsheet_id, "-f", formulas_path, "--value-input", "USER_ENTERED"],
        check=True,
    )
  if format_path and os.path.exists(format_path):
    with open(format_path, "r", encoding="utf-8") as f:
      fmt_payload = json.load(f)
    if fmt_payload.get("requests"):
      subprocess.run(
          [GSHEETS_BIN, "mutate", "raw-batch", spreadsheet_id, "-f", format_path],
          check=True,
      )


def main(argv: Sequence[str] | None = None) -> int:
  """Parses CLI args, builds CSVs + gsheets batch files, validates rows, and optionally pushes to Sheets."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--v2-root", action="append", default=[], help="Local mirrored GCS root for v2 result JSONs")
  parser.add_argument("--include-smoke", action="store_true")
  parser.add_argument("--include-val", action="store_true")
  parser.add_argument("--v1-p2-root", action="append", default=[])
  parser.add_argument("--v1-p1-root", action="append", default=[])
  parser.add_argument("--v1-variant-root", action="append", default=[], help="LABEL=PATH")
  parser.add_argument("--v1-before-root", action="append", default=[])
  parser.add_argument("--v1-multihost-csv", default="")
  parser.add_argument("--v1-cold-s1", action="append", default=[], help="MODEL=SECONDS")
  parser.add_argument("--impl-json", default="", help="Optional JSON file listing Progress task updates")
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--spreadsheet-id", default=DEFAULT_SPREADSHEET_ID)
  parser.add_argument("--apply", action="store_true", help="Push generated batch files to Google Sheets")
  args = parser.parse_args(argv)

  variants: dict[str, list[str]] = {}
  for spec in args.v1_variant_root:
    lbl, path = spec.split("=", 1)
    variants.setdefault(lbl, []).append(path)
  overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.v1_cold_s1)}
  impl_updates: list[dict[str, Any]] = []
  if args.impl_json and os.path.exists(args.impl_json):
    with open(args.impl_json, "r", encoding="utf-8") as f:
      loaded_impl = json.load(f)
      if isinstance(loaded_impl, list):
        impl_updates = loaded_impl

  ds = build_v2_dataset(
      v2_roots=args.v2_root,
      include_smoke=args.include_smoke,
      include_val=args.include_val,
      p2_roots=args.v1_p2_root,
      p1_roots=args.v1_p1_root,
      variant_roots=list(variants.items()),
      before_roots=args.v1_before_root,
      multihost_csv=args.v1_multihost_csv or None,
      cold_s1_overrides=overrides,
  )
  errs = ds.validate()
  if errs:
    print(f"Sheet v2 validation failed ({len(errs)} errors; first 10):", file=sys.stderr)
    for err in errs[:10]:
      print(f"  - {err}", file=sys.stderr)
    return 1

  os.makedirs(args.out_dir, exist_ok=True)
  _write_csv(os.path.join(args.out_dir, "summary.csv"), ds.summary_rows, schema_v2.SUMMARY_COLUMNS)
  _write_csv(os.path.join(args.out_dir, "step1_downloads.csv"), ds.step1_rows, schema_v2.STEP1_COLUMNS)
  _write_csv(
      os.path.join(args.out_dir, "stages_safetensors_dynamic.csv"),
      ds.stages_sd_rows,
      schema_v2.STAGES_SD_COLUMNS,
  )
  _write_csv(os.path.join(args.out_dir, "stages_to_maxtext.csv"), ds.stages_tm_rows, schema_v2.STAGES_TM_COLUMNS)
  _write_csv(os.path.join(args.out_dir, "runs.csv"), ds.runs_rows, schema_v2.RUNS_COLUMNS)

  raw_ops, formula_ops = build_gsheets_batches(ds, impl_updates=impl_updates)
  fmt_reqs = build_summary_format_requests(ds.summary_rows)
  raw_path = os.path.join(args.out_dir, "01_raw_batch.json")
  formulas_path = os.path.join(args.out_dir, "02_formulas_batch.json")
  format_path = os.path.join(args.out_dir, "03_format_batch.json")
  with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(raw_ops, f, indent=2)
  with open(formulas_path, "w", encoding="utf-8") as f:
    json.dump(formula_ops, f, indent=2)
  with open(format_path, "w", encoding="utf-8") as f:
    json.dump(fmt_reqs, f, indent=2)

  print(
      f"Built Sheet v2: Summary={len(ds.summary_rows)} Step1={len(ds.step1_rows)} "
      f"StagesSD={len(ds.stages_sd_rows)} StagesTM={len(ds.stages_tm_rows)} Runs={len(ds.runs_rows)}"
  )
  if args.apply:
    apply_batches(args.spreadsheet_id, raw_path, formulas_path, format_path)
    print(f"Synced Sheet v2 ({args.spreadsheet_id}).")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
