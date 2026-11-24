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

"""Adapts legacy v1 tpu7x result JSONs, logs, and CSVs into Benchmark v2 rows."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
import dataclasses
import glob
import os
import re
import sys
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_DIR not in sys.path:
  sys.path.insert(0, _BENCH_DIR)

from bench import schema_v2  # pylint: disable=wrong-import-position
from scripts import build_tpu7x_sheet_table as bsh  # pylint: disable=wrong-import-position
from scripts import build_tpu7x_tables as bt  # pylint: disable=wrong-import-position

_SYNTHETIC_DEEPSEEK_MODELS = frozenset(
    {
        "deepseek4-284b",
        "deepseek3-671b",
        "deepseek3.2-671b",
        "deepseek4-1600b",
    }
)
_TOPOLOGY_BY_HOSTS = {
    1: "2x2x1",
    2: "2x2x2",
    4: "2x2x4",
    8: "2x4x4",
}
_V1_CAVEAT_NOTE = (
    "v1 imported row: pre-dates v2 per-stage breakdown, pre-load TPU check, CODE_VERSION.json, "
    "and multi-host RAM tracking (host 0 only)"
)


@dataclasses.dataclass
class LegacyV1Dataset:
  """Container holding adapted v2 rows across all five sheet data tabs."""

  summary_rows: list[dict[str, Any]]
  step1_rows: list[dict[str, Any]]
  stages_sd_rows: list[dict[str, Any]]
  stages_tm_rows: list[dict[str, Any]]
  runs_rows: list[dict[str, Any]]

  def validate(self) -> list[str]:
    """Validates every adapted row against the v2 Sheet column specifications."""
    errors: list[str] = []
    tab_pairs = (
        ("Summary", self.summary_rows, schema_v2.SUMMARY_COLUMNS),
        ("Step 1 downloads", self.step1_rows, schema_v2.STEP1_COLUMNS),
        (
            "Stages: safetensors_dynamic",
            self.stages_sd_rows,
            schema_v2.STAGES_SD_COLUMNS,
        ),
        ("Stages: to_maxtext", self.stages_tm_rows, schema_v2.STAGES_TM_COLUMNS),
        ("Runs", self.runs_rows, schema_v2.RUNS_COLUMNS),
    )
    for tab_name, rows, cols in tab_pairs:
      for idx, row in enumerate(rows):
        row_errs = schema_v2.validate_row_dict(row, cols)
        for err in row_errs:
          errors.append(f"{tab_name}[{idx}] ({row.get('row_id', row.get('model_name'))}): {err}")
    return errors


def topology_for_hosts_per_slice(hosts_per_slice: int) -> str:
  """Maps v1 v7x hosts-per-slice count to canonical 3D torus topology."""
  if hosts_per_slice in _TOPOLOGY_BY_HOSTS:
    return _TOPOLOGY_BY_HOSTS[hosts_per_slice]
  raise ValueError(f"Unsupported v7x hosts_per_slice in v1 data: {hosts_per_slice!r}")


def parse_v1_config_dir(cfg: str) -> tuple[str, int, int]:
  """Parses v1 config dir names like `scanned`, `scanned_slices2`, `unscanned_hosts4`."""
  parts = cfg.split("_")
  scan_mode = parts[0] if parts[0] in ("scanned", "unscanned") else "scanned"
  num_slices = 1
  hosts_per_slice = 1
  for part in parts[1:]:
    if part.startswith("slices"):
      num_slices = int(part.removeprefix("slices"))
    elif part.startswith("hosts"):
      hosts_per_slice = int(part.removeprefix("hosts"))
  return scan_mode, num_slices, hosts_per_slice


def _read_phase_log_tail(log_path: str) -> str:
  if not log_path:
    return ""
  chunks: list[str] = []
  paths = [log_path] + sorted(glob.glob(log_path.removesuffix(".log") + "_slice*.log"))
  for p in paths:
    if os.path.exists(p):
      with open(p, encoding="utf-8", errors="replace") as f:
        chunks.append(f.read()[-200_000:])
  return "\n".join(chunks)


def _extract_exit_code(err: str) -> int | str:
  m = re.search(r"exit code (-?\d+)", err)
  return int(m.group(1)) if m else ""


def classify_v1_failure(rec: Mapping[str, Any] | None, log_path: str = "") -> tuple[str, str, str, int | str, str]:
  """Classifies a v1 failure into `(sheet_status, failure_category, summary, exit_code, signal)`."""
  if rec is None:
    return "NOT RUN", "", "not run", "", ""
  err = str(rec.get("error", "") or "")
  log_tail = _read_phase_log_tail(log_path)
  text = f"{err}\n{log_tail}"
  reason = bt.failure_reason(dict(rec), log_path)
  exit_code = _extract_exit_code(err)
  signal = "SIGKILL" if exit_code in (-9, 137) else ""

  if "RESOURCE_EXHAUSTED" in text or "pjrt.OOM" in text or "TPU HBM OOM" in reason:
    return "OOM-HBM", "OOM-HBM", reason, exit_code, signal
  if "TIMEOUT: exceeded PHASE_TIMEOUT_S" in text or "DEADLINE_EXCEEDED: Barrier timed out" in text:
    return "TIMEOUT", "TIMEOUT", reason, exit_code, signal
  if (
      "holds the whole converted state" in text
      or "exceeds ~" in text
      and "usable" in text
      or re.search(r"needs all \d+ shards .* staged", text)
      or "OOMKilled" in err
      or exit_code in (-9, 137)
  ):
    return "OOM-HOST", "OOM-HOST", reason, exit_code, signal
  return (
      "ERROR",
      "ERROR",
      reason or (err.splitlines()[0] if err else "failed"),
      exit_code,
      signal,
  )


def _empty_row(columns: Sequence[schema_v2.ColumnSpec]) -> dict[str, Any]:
  return {col.key: "" for col in columns}


def _round_opt(val: Any, ndigits: int = 2) -> float | str:
  if val in (None, "", "N/A"):
    return ""
  fval = float(val)
  return round(fval, ndigits) if fval != 0.0 else ""


def _model_lookup(models_cfg: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
  cfg = models_cfg or schema_v2.load_models_v2()
  return {m["model_name"]: m for m in cfg["models"]}


def _checkpoint_fields(model_info: Mapping[str, Any] | None, is_synthetic: bool) -> dict[str, Any]:
  """Extracts checkpoint metadata columns from `models_v2` plus synthetic flag."""
  if not model_info:
    return {
        "model_test_tier": "",
        "hf_repo_id": "",
        "model_parameter_count_billions": "",
        "model_architecture": "",
        "hf_checkpoint_dtype": "",
        "loaded_checkpoint_dtype": "BF16" if is_synthetic else "",
        "maxtext_load_weight_dtype": "",
        "checkpoint_weights_kind": "synthetic_sparse" if is_synthetic else "real",
        "checkpoint_size_gb": "",
        "checkpoint_number_of_files": "",
    }
  return {
      "model_test_tier": model_info.get("tier", ""),
      "hf_repo_id": model_info.get("hf_repo_id", ""),
      "model_parameter_count_billions": model_info.get("params_billions", ""),
      "model_architecture": model_info.get("architecture", ""),
      # Synthetic DeepSeek checkpoints did not download real HF FP8/FP4 weights.
      "hf_checkpoint_dtype": "" if is_synthetic else model_info.get("hf_dtype", ""),
      "loaded_checkpoint_dtype": "BF16" if is_synthetic else model_info.get("loaded_dtype", ""),
      "maxtext_load_weight_dtype": model_info.get("weight_dtype", ""),
      "checkpoint_weights_kind": "synthetic_sparse" if is_synthetic else "real",
      "checkpoint_size_gb": model_info.get("loaded_bf16_gb", ""),
      "checkpoint_number_of_files": model_info.get("num_files", ""),
  }


def _make_v1_row_id(  # pylint: disable=too-many-positional-arguments
    model_name: str,
    scan_mode: str,
    pipeline: str,
    num_slices: int,
    topology: str,
    variant_tag: str = "",
) -> str:
  base = schema_v2.make_row_id("legacy", "v7x", model_name, scan_mode, pipeline, num_slices, topology)
  return f"{base}|{variant_tag}" if variant_tag else base


def _v1_environment_common(
    model_name: str,
    num_slices: int,
    hosts_per_slice: int,
    topology: str,
    *,
    run_id: str,
    experiment_name: str,
) -> dict[str, Any]:
  """Returns baseline v1 TPU v7x environment fields for `Runs` rows."""
  del model_name
  devices_per_slice = 8 * hosts_per_slice
  total_hosts = num_slices * hosts_per_slice
  total_devices = num_slices * devices_per_slice
  interconnect = (
      "ICI+DCN"
      if (num_slices > 1 and hosts_per_slice > 1)
      else ("DCN" if num_slices > 1 else ("ICI" if hosts_per_slice > 1 else "Single-host"))
  )
  return {
      "run_id": run_id,
      "experiment_name": experiment_name,
      "repetition_index": 1,
      "queue_attempt_number": 1,
      "jobset_restart_attempt": "",
      "tpu_platform": "v7x",
      "gke_cluster_name": "bodaborg-tpu7x-nap",
      "gcp_project_id": "cloud-tpu-shared-capacity",
      "gcp_region": "us-central1",
      "gcp_zone": "us-central1-c",
      "capacity_type": "spot",
      "host_machine_type": "tpu7x-highmem",
      "tpu_accelerator_type": "tpu7x",
      "number_of_slices": num_slices,
      "slice_topology": topology,
      "hosts_per_slice": hosts_per_slice,
      "tpu_chips_per_host": 4,
      "tpu_devices_per_chip": 2,
      "tpu_devices_per_slice": devices_per_slice,
      "total_hosts": total_hosts,
      "total_tpu_devices": total_devices,
      "interconnect_type": interconnect,
      "total_tpu_hbm_gb": round(813.8 * total_hosts, 2),
      "tpu_hbm_per_device_gb_measured": "",
      "host_ram_gb_measured": 1014.5,
      "was_preempted": False,
      "git_branch_name": "docs/checkpoint-load-benchmark",
      "active_timing_wrappers": "v1_coarse_phase_timers",
  }


def adapt_single_v1_phase_json(
    result_path: str,
    *,
    model_name: str,
    cfg: str,
    phase: str,
    run_id: str,
    experiment_name: str,
    models_by_name: Mapping[str, dict[str, Any]] | None = None,
    cold_s1: float | None = None,
    variant_tag: str = "",
) -> dict[str, Any]:
  """Adapts one legacy `<phase>_result.json` (and adjacent `.log`) into a validated `Runs` tab dict."""
  models = models_by_name if models_by_name is not None else _model_lookup()
  rec = bt._load(result_path) or {}  # pylint: disable=protected-access
  log_path = os.path.join(os.path.dirname(result_path), f"{phase}.log")
  scan_mode, num_slices, hosts_per_slice = parse_v1_config_dir(cfg)
  topology = topology_for_hosts_per_slice(hosts_per_slice)
  pipeline = "safetensors_dynamic" if phase == "safetensors_dynamic" else "to_maxtext"
  is_synthetic = model_name in _SYNTHETIC_DEEPSEEK_MODELS
  row_id = _make_v1_row_id(model_name, scan_mode, pipeline, num_slices, topology, variant_tag=variant_tag)

  row = _empty_row(schema_v2.RUNS_COLUMNS)
  row.update(
      _v1_environment_common(
          model_name,
          num_slices,
          hosts_per_slice,
          topology,
          run_id=run_id,
          experiment_name=experiment_name,
      )
  )
  row.update(_checkpoint_fields(models.get(model_name), is_synthetic))
  row["row_id"] = row_id
  row["phase_name"] = phase
  row["model_name"] = model_name
  row["scan_layers_mode"] = scan_mode
  row["pipeline_name"] = pipeline
  row["number_of_parameter_leaves"] = rec.get("num_param_leaves", "") if rec.get("num_param_leaves") else ""
  row["result_json_url"] = result_path
  if os.path.exists(log_path):
    row["step3_or_load_log_url"] = log_path
    row["_local_log_files"] = [log_path]
  row["_local_json_path"] = result_path

  if rec.get("status") == "FAILED":
    sheet_status, fail_cat, err_sum, exit_code, sig = classify_v1_failure(rec, log_path)
    row["run_status"] = "FAILED"
    row["failure_category"] = fail_cat or sheet_status
    row["error_summary"] = err_sum
    row["process_exit_code"] = exit_code
    row["process_signal"] = sig
  else:
    row["run_status"] = "OK"
    if cold_s1 and not is_synthetic:
      row["step1_total_seconds"] = round(cold_s1, 2)
  return row


def _apply_common_summary_identity(
    row: dict[str, Any],
    *,
    row_id: str,
    model_name: str,
    pipeline: str,
    scan_mode: str,
    num_slices: int,
    hosts_per_slice: int,
    topology: str,
    ckpt_fields: Mapping[str, Any],
) -> None:
  """Populates shared platform/model/topology columns in a Summary row."""
  devices_per_slice = 8 * hosts_per_slice
  row.update(
      {
          "row_id": row_id,
          "tpu_platform": "v7x",
          "model_test_tier": ckpt_fields["model_test_tier"],
          "model_name": model_name,
          "pipeline_name": pipeline,
          "scan_layers_mode": scan_mode,
          "number_of_slices": num_slices,
          "slice_topology": topology,
          "hosts_per_slice": hosts_per_slice,
          "tpu_devices_per_slice": devices_per_slice,
          "total_tpu_devices": num_slices * devices_per_slice,
          "hf_checkpoint_dtype": ckpt_fields["hf_checkpoint_dtype"],
          "loaded_checkpoint_dtype": ckpt_fields["loaded_checkpoint_dtype"],
          "maxtext_load_weight_dtype": ckpt_fields["maxtext_load_weight_dtype"],
          "checkpoint_weights_kind": ckpt_fields["checkpoint_weights_kind"],
          "checkpoint_size_gb": ckpt_fields["checkpoint_size_gb"],
          "load_backend_and_tpu_device_kind": "",
          "repetitions_count": 1,
      }
  )


def _make_sd_stage_row(
    *,
    row_id: str,
    model_name: str,
    scan_mode: str,
    num_slices: int,
    topology: str,
    rec: Mapping[str, Any] | None,
    cold_s1: float,
    is_synthetic: bool,
    ok: bool,
) -> dict[str, Any]:
  """Adapts coarse v1 `safetensors_dynamic` timings without fabricating fine stages."""
  row = _empty_row(schema_v2.STAGES_SD_COLUMNS)
  row.update(
      {
          "row_id": row_id,
          "tpu_platform": "v7x",
          "model_name": model_name,
          "scan_layers_mode": scan_mode,
          "number_of_slices": num_slices,
          "slice_topology": topology,
      }
  )
  if not rec:
    return row
  if ok:
    row["setup_abstract_train_state_build_seconds"] = _round_opt(rec.get("init_abstract_state_s"))
    row["safetensors_dynamic_step1_total_seconds"] = (
        "" if is_synthetic else (_round_opt(cold_s1) if cold_s1 > 0 else _round_opt(rec.get("hf_to_gcs_s")))
    )
    row["safetensors_dynamic_step2_build_hf_to_maxtext_mappings_seconds"] = _round_opt(rec.get("derive_mappings_s"))
    row["safetensors_dynamic_step2_load_total_seconds"] = _round_opt(rec.get("load_sharded_hf_s"))
    row["safetensors_dynamic_step2_transform_total_seconds"] = _round_opt(rec.get("transform_hf_to_mt_s"))
    row["safetensors_dynamic_step2_total_seconds"] = _round_opt(rec.get("safetensors_and_convert_s"))
  peak_ram = rec.get("peak_host_ram_gb")
  if peak_ram:
    row["memory_peak_host_ram_container_gb"] = round(float(peak_ram), 2)
    row["memory_peak_host_ram_host_name"] = "host-0 (v1 only recorded host 0)"
  if rec.get("total_host_ram_gb"):
    row["memory_host_ram_size_gb"] = round(float(rec["total_host_ram_gb"]), 2)
  return row


def _make_tm_stage_row(
    *,
    row_id: str,
    model_name: str,
    scan_mode: str,
    num_slices: int,
    topology: str,
    tm1: Mapping[str, Any] | None,
    tm2: Mapping[str, Any] | None,
    tm3: Mapping[str, Any] | None,
    cold_s1: float,
    is_synthetic: bool,
) -> dict[str, Any]:
  """Adapts coarse v1 `to_maxtext` Step 1-3 timings without fabricating fine stages."""
  row = _empty_row(schema_v2.STAGES_TM_COLUMNS)
  row.update(
      {
          "row_id": row_id,
          "tpu_platform": "v7x",
          "model_name": model_name,
          "scan_layers_mode": scan_mode,
          "number_of_slices": num_slices,
          "slice_topology": topology,
      }
  )
  if bt._ok(dict(tm1) if tm1 else None) and not is_synthetic:  # pylint: disable=protected-access
    row["to_maxtext_step1_total_seconds"] = _round_opt(tm1.get("hf_to_local_s"))
  if bt._ok(dict(tm2) if tm2 else None):  # pylint: disable=protected-access
    row["to_maxtext_step2_build_abstract_model_seconds"] = _round_opt(tm2.get("init_abstract_model_s"))
    row["to_maxtext_step2_transform_total_seconds"] = _round_opt(tm2.get("transform_and_shard_s"))
    row["to_maxtext_step2_orbax_save_total_seconds"] = _round_opt(tm2.get("orbax_save_to_gcs_s"))
    row["to_maxtext_step2_total_seconds"] = _round_opt(tm2.get("convert_to_orbax_s"))
    if tm2.get("peak_host_ram_gb"):
      row["memory_to_maxtext_step2_peak_host_ram_container_gb"] = round(float(tm2["peak_host_ram_gb"]), 2)
    if tm2.get("total_host_ram_gb"):
      row["memory_to_maxtext_step2_host_ram_size_gb"] = round(float(tm2["total_host_ram_gb"]), 2)
  if bt._ok(dict(tm3) if tm3 else None):  # pylint: disable=protected-access
    row["setup_abstract_train_state_build_seconds"] = _round_opt(tm3.get("init_abstract_state_s"))
    row["to_maxtext_step3_total_seconds"] = _round_opt(tm3.get("load_orbax_s"))
    row["to_maxtext_load_only_seconds"] = _round_opt(tm3.get("load_orbax_s"))
    if tm3.get("peak_host_ram_gb"):
      row["memory_peak_host_ram_container_gb"] = round(float(tm3["peak_host_ram_gb"]), 2)
      row["memory_peak_host_ram_host_name"] = "host-0 (v1 only recorded host 0)"
    if tm3.get("total_host_ram_gb"):
      row["memory_host_ram_size_gb"] = round(float(tm3["total_host_ram_gb"]), 2)
  if (
      not is_synthetic
      and bt._ok(dict(tm2) if tm2 else None)  # pylint: disable=protected-access
      and bt._ok(dict(tm3) if tm3 else None)  # pylint: disable=protected-access
  ):
    s1_val = cold_s1 or float((tm1 or {}).get("hf_to_gcs_s", 0.0) or 0.0)
    old_e2e = s1_val + float(tm2.get("convert_to_orbax_s", 0.0) or 0.0) + float(tm3.get("load_orbax_s", 0.0) or 0.0)
    row["to_maxtext_total_end_to_end_old_formula_seconds"] = round(old_e2e, 2)
  return row


def _lookup_log_url(model: str, scan: str, n: int, global_devices: int, pipeline_label: str) -> str:
  """Looks up known gpaste/log URL for a v1 configuration."""
  raw = bsh._log_url(model, scan, f"{n} ({global_devices})", pipeline_label)  # pylint: disable=protected-access
  if raw == "—" or not raw:
    return ""
  m_url = re.search(r"https://paste\.googleplex\.com/\d+", str(raw))
  return m_url.group(0) if m_url else ""


def _build_single_host_pair(
    *,
    logs: bt.Logs,
    summary_base: Mapping[str, Any],
    models_by_name: Mapping[str, dict[str, Any]],
    runs_out: list[dict[str, Any]],
    stages_sd_out: list[dict[str, Any]],
    stages_tm_out: list[dict[str, Any]],
    summary_out: list[dict[str, Any]],
    variant_logs: Sequence[tuple[str, bt.Logs]] = (),
    run_id_prefix: str = "v1",
) -> None:
  """Populates Summary, Stages, and Runs entries for one P1/P2 configuration pair."""
  model = str(summary_base["model"])
  scan = str(summary_base["scan_mode"])
  n = int(summary_base["num_slices"])
  cfg = scan if n == 1 else f"{scan}_slices{n}"
  hosts_per_slice = max(1, int(summary_base["global_devices"]) // (8 * n))
  topology = topology_for_hosts_per_slice(hosts_per_slice)
  is_synthetic = model in _SYNTHETIC_DEEPSEEK_MODELS
  ckpt = _checkpoint_fields(models_by_name.get(model), is_synthetic)
  s1 = float(summary_base["safetensor_1_hf_to_gcs_s"] or 0.0)

  tm1, tm1_log = logs.load(model, scan, "to_maxtext_step1")
  tm2, tm2_log = logs.load(model, scan, "to_maxtext_step2")
  tm3, tm3_log = logs.load(model, cfg, "to_maxtext_step3")
  st, st_log = logs.load(model, cfg, "safetensors_dynamic")

  for phase_name, p_rec, p_cfg in (
      ("to_maxtext_step1", tm1, scan),
      ("to_maxtext_step2", tm2, scan),
      ("to_maxtext_step3", tm3, cfg),
      ("safetensors_dynamic", st, cfg),
  ):
    if p_rec is not None:
      p_path = logs.path(model, p_cfg, phase_name)
      runs_out.append(
          adapt_single_v1_phase_json(
              p_path,
              model_name=model,
              cfg=p_cfg,
              phase=phase_name,
              run_id=run_id_prefix,
              experiment_name=run_id_prefix,
              models_by_name=models_by_name,
              cold_s1=s1,
          )
      )

  tm_ok = summary_base["to_maxtext_status"] == "OK"
  st_ok = summary_base["safetensor_status"] == "OK"

  # Build `to_maxtext` Summary + Stages row.
  tm_id = _make_v1_row_id(model, scan, "to_maxtext", n, topology)
  tm_row = _empty_row(schema_v2.SUMMARY_COLUMNS)
  _apply_common_summary_identity(
      tm_row,
      row_id=tm_id,
      model_name=model,
      pipeline="to_maxtext",
      scan_mode=scan,
      num_slices=n,
      hosts_per_slice=hosts_per_slice,
      topology=topology,
      ckpt_fields=ckpt,
  )
  if tm_ok:
    tm_s1_local = float((tm1 or {}).get("hf_to_local_s", 0.0) or 0.0)
    tm_s2 = float(summary_base["to_maxtext_2_convert_to_orbax_s"] or 0.0)
    tm_s3 = float(summary_base["to_maxtext_3_load_orbax_s"] or 0.0)
    tm_no_hf = (0.0 if is_synthetic else tm_s1_local) + tm_s2 + tm_s3
    tm_e2e = (0.0 if is_synthetic else float(summary_base["to_maxtext_1_hf_to_gcs_s"] or s1)) + tm_no_hf
    tm_row.update(
        {
            "run_status": "OK",
            "step1_total_seconds": "LOCAL"
            if is_synthetic
            else _round_opt(summary_base["to_maxtext_1_hf_to_gcs_s"] or s1),
            "to_maxtext_step1_total_seconds": "" if is_synthetic else _round_opt(tm_s1_local),
            "to_maxtext_step2_total_seconds": round(tm_s2, 2),
            "load_into_tpu_hbm_total_seconds": round(tm_s3, 2),
            "total_without_hf_download_seconds": round(tm_no_hf, 2),
            "total_end_to_end_seconds": "LOCAL" if is_synthetic else round(tm_e2e, 2),
            "speedup_end_to_end_to_maxtext_vs_safetensors_dynamic": 1.0 if not is_synthetic else "",
            "speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic": 1.0,
            "speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2": 1.0,
            "memory_peak_host_ram_container_gb": _round_opt((tm3 or {}).get("peak_host_ram_gb")),
            "log_url": _lookup_log_url(model, scan, n, int(summary_base["global_devices"]), "to_maxtext"),
            "note": _V1_CAVEAT_NOTE,
        }
    )
  else:
    if tm1 is not None and not bt._ok(tm1):  # pylint: disable=protected-access
      st_code, _, err_sum, _, _ = classify_v1_failure(tm1, tm1_log)
    elif tm2 is not None and not bt._ok(tm2):  # pylint: disable=protected-access
      st_code, _, err_sum, _, _ = classify_v1_failure(tm2, tm2_log)
    elif tm2 is None:
      st_code = "NOT RUN"
      err_sum = bsh._note(model, scan, n, "to_maxtext") or "not run"  # pylint: disable=protected-access
    elif tm3 is None:
      st_code = "NOT RUN"
      err_sum = bsh.TM_S3_NOTES.get((model, scan, n)) or "Step 3 not run"
    else:
      st_code, _, err_sum, _, _ = classify_v1_failure(tm3, tm3_log)
    tm_row.update(
        {
            "run_status": st_code,
            "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
            "to_maxtext_step2_total_seconds": _round_opt((tm2 or {}).get("convert_to_orbax_s"))
            if bt._ok(tm2)  # pylint: disable=protected-access
            else "",
            "log_url": _lookup_log_url(model, scan, n, int(summary_base["global_devices"]), "to_maxtext"),
            "note": f"{err_sum}; {_V1_CAVEAT_NOTE}",
        }
    )
  tm_row["_local_log_files"] = [p for p in (tm1_log, tm2_log, tm3_log) if p and os.path.exists(p)]
  summary_out.append(tm_row)
  stages_tm_out.append(
      _make_tm_stage_row(
          row_id=tm_id,
          model_name=model,
          scan_mode=scan,
          num_slices=n,
          topology=topology,
          tm1=tm1,
          tm2=tm2,
          tm3=tm3,
          cold_s1=s1,
          is_synthetic=is_synthetic,
      )
  )

  # Build `safetensors_dynamic` Summary + Stages row.
  st_id = _make_v1_row_id(model, scan, "safetensors_dynamic", n, topology)
  st_row = _empty_row(schema_v2.SUMMARY_COLUMNS)
  _apply_common_summary_identity(
      st_row,
      row_id=st_id,
      model_name=model,
      pipeline="safetensors_dynamic",
      scan_mode=scan,
      num_slices=n,
      hosts_per_slice=hosts_per_slice,
      topology=topology,
      ckpt_fields=ckpt,
  )
  st_s2 = float(summary_base["safetensor_2_safetensors_and_convert_s"] or 0.0)
  if st_ok:
    st_e2e = s1 + st_s2
    st_row.update(
        {
            "run_status": "OK",
            "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
            "load_into_tpu_hbm_total_seconds": round(st_s2, 2),
            "total_without_hf_download_seconds": round(st_s2, 2),
            "total_end_to_end_seconds": "LOCAL" if is_synthetic else round(st_e2e, 2),
            "memory_peak_host_ram_container_gb": _round_opt((st or {}).get("peak_host_ram_gb")),
            "log_url": _lookup_log_url(model, scan, n, int(summary_base["global_devices"]), "safetensor"),
            "note": _V1_CAVEAT_NOTE,
        }
    )
    if tm_ok and st_s2 > 0:
      tm_no_hf = float(tm_row["total_without_hf_download_seconds"])
      tm_s3_val = float(tm_row["load_into_tpu_hbm_total_seconds"])
      st_row["speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic"] = round(tm_no_hf / st_s2, 2)
      st_row["speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2"] = round(tm_s3_val / st_s2, 2)
      if not is_synthetic and st_e2e > 0:
        tm_e2e = float(tm_row["total_end_to_end_seconds"])
        st_row["speedup_end_to_end_to_maxtext_vs_safetensors_dynamic"] = round(tm_e2e / st_e2e, 2)
  else:
    st_code, _, err_sum, _, _ = classify_v1_failure(st, st_log)
    if st is None and bsh._note(model, scan, n, "safetensor"):  # pylint: disable=protected-access
      st_code = "NOT RUN"
      err_sum = str(bsh._note(model, scan, n, "safetensor"))  # pylint: disable=protected-access
    st_row.update(
        {
            "run_status": st_code,
            "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
            "log_url": _lookup_log_url(model, scan, n, int(summary_base["global_devices"]), "safetensor"),
            "note": f"{err_sum}; {_V1_CAVEAT_NOTE}",
        }
    )
  st_row["_local_log_files"] = [st_log] if st_log and os.path.exists(st_log) else []
  summary_out.append(st_row)
  stages_sd_out.append(
      _make_sd_stage_row(
          row_id=st_id,
          model_name=model,
          scan_mode=scan,
          num_slices=n,
          topology=topology,
          rec=st,
          cold_s1=s1,
          is_synthetic=is_synthetic,
          ok=st_ok,
      )
  )

  # Add fix-variant rows (e.g. `with_fix` on qwen3.5-397b-a17b and gpt-oss-120b).
  for label, vlogs in variant_logs:
    v_rec, v_log = vlogs.load(model, cfg, "safetensors_dynamic")
    if v_rec is None:
      continue
    tag = label.lower().replace(" ", "_")
    v_id = _make_v1_row_id(model, scan, "safetensors_dynamic", n, topology, variant_tag=tag)
    v_ok = bt._ok(v_rec)  # pylint: disable=protected-access
    v_row = _empty_row(schema_v2.SUMMARY_COLUMNS)
    _apply_common_summary_identity(
        v_row,
        row_id=v_id,
        model_name=model,
        pipeline="safetensors_dynamic",
        scan_mode=scan,
        num_slices=n,
        hosts_per_slice=hosts_per_slice,
        topology=topology,
        ckpt_fields=ckpt,
    )
    if v_ok:
      v_s2 = float(v_rec.get("safetensors_and_convert_s", 0.0) or 0.0)
      v_e2e = s1 + v_s2
      v_row.update(
          {
              "run_status": "OK",
              "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
              "load_into_tpu_hbm_total_seconds": round(v_s2, 2),
              "total_without_hf_download_seconds": round(v_s2, 2),
              "total_end_to_end_seconds": "LOCAL" if is_synthetic else round(v_e2e, 2),
              "memory_peak_host_ram_container_gb": _round_opt(v_rec.get("peak_host_ram_gb")),
              "note": f"variant={label}; {_V1_CAVEAT_NOTE}",
          }
      )
    else:
      v_code, _, v_err, _, _ = classify_v1_failure(v_rec, v_log)
      v_row.update(
          {
              "run_status": v_code,
              "note": f"variant={label}; {v_err}; {_V1_CAVEAT_NOTE}",
          }
      )
    summary_out.append(v_row)
    stages_sd_out.append(
        _make_sd_stage_row(
            row_id=v_id,
            model_name=model,
            scan_mode=scan,
            num_slices=n,
            topology=topology,
            rec=v_rec,
            cold_s1=s1,
            is_synthetic=is_synthetic,
            ok=v_ok,
        )
    )
    runs_out.append(
        adapt_single_v1_phase_json(
            vlogs.path(model, cfg, "safetensors_dynamic"),
            model_name=model,
            cfg=cfg,
            phase="safetensors_dynamic",
            run_id=f"{run_id_prefix}_{tag}",
            experiment_name=f"{run_id_prefix}_{tag}",
            models_by_name=models_by_name,
            cold_s1=s1,
            variant_tag=tag,
        )
    )


def _add_multihost_csv_rows(
    *,
    multihost_csv: str,
    cold_s1_by_model: Mapping[str, float],
    models_by_name: Mapping[str, dict[str, Any]],
    logs: bt.Logs | None,
    summary_out: list[dict[str, Any]],
    stages_sd_out: list[dict[str, Any]],
    runs_out: list[dict[str, Any]],
) -> None:
  """Ingests multi-host `safetensors_dynamic` CSV records and links matching JSONs."""
  seen_ids = {r["row_id"] for r in summary_out}
  with open(multihost_csv, encoding="utf-8") as f:
    for rec in csv.DictReader(f):
      if rec.get("layout", "").endswith("2x2x1"):
        continue
      model = rec["model"]
      scan = rec["scan_mode"]
      layout = rec["layout"]  # e.g. "1 × 2x2x2" or "2 × 2x2x2"
      num_slices = int(layout.split()[0])
      topo = layout.split()[-1]
      hosts_per_slice = max(1, int(rec["hosts"]) // num_slices)
      is_synthetic = model in _SYNTHETIC_DEEPSEEK_MODELS
      ckpt = _checkpoint_fields(models_by_name.get(model), is_synthetic)
      note_raw = (rec.get("note") or "").strip()
      variant_tag = "with_fix" if note_raw.startswith("with fix") else ""
      row_id = _make_v1_row_id(
          model,
          scan,
          "safetensors_dynamic",
          num_slices,
          topo,
          variant_tag=variant_tag,
      )
      if row_id in seen_ids:
        continue
      seen_ids.add(row_id)

      s1 = float(cold_s1_by_model.get(model, 0.0) or 0.0)
      status = rec["status"]
      row = _empty_row(schema_v2.SUMMARY_COLUMNS)
      _apply_common_summary_identity(
          row,
          row_id=row_id,
          model_name=model,
          pipeline="safetensors_dynamic",
          scan_mode=scan,
          num_slices=num_slices,
          hosts_per_slice=hosts_per_slice,
          topology=topo,
          ckpt_fields=ckpt,
      )
      sd_stage = _empty_row(schema_v2.STAGES_SD_COLUMNS)
      sd_stage.update(
          {
              "row_id": row_id,
              "tpu_platform": "v7x",
              "model_name": model,
              "scan_layers_mode": scan,
              "number_of_slices": num_slices,
              "slice_topology": topo,
          }
      )

      if status == "OK":
        total_s = float(rec["total_s"])
        load_s = float(rec["load_s"])
        transform_s = float(rec["transform_s"])
        peak_ram = float(rec["peak_host_ram_gb"])
        row.update(
            {
                "run_status": "OK",
                "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
                "load_into_tpu_hbm_total_seconds": round(total_s, 2),
                "total_without_hf_download_seconds": round(total_s, 2),
                "total_end_to_end_seconds": "LOCAL" if is_synthetic else round(s1 + total_s, 2),
                "memory_peak_host_ram_container_gb": round(peak_ram, 2),
                "note": f"multihost source={rec.get('source', 'lance')}; {note_raw}; {_V1_CAVEAT_NOTE}".strip("; "),
            }
        )
        sd_stage.update(
            {
                "safetensors_dynamic_step1_total_seconds": "" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
                "safetensors_dynamic_step2_load_total_seconds": round(load_s, 2),
                "safetensors_dynamic_step2_transform_total_seconds": round(transform_s, 2),
                "safetensors_dynamic_step2_total_seconds": round(total_s, 2),
                "memory_peak_host_ram_container_gb": round(peak_ram, 2),
                "memory_peak_host_ram_host_name": "host-0 (v1 only recorded host 0)",
            }
        )
      else:
        fail_code = "OOM-HBM" if "HBM OOM" in note_raw else "ERROR"
        row.update(
            {
                "run_status": fail_code,
                "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
                "note": f"multihost source={rec.get('source', 'lance')}; {note_raw}; {_V1_CAVEAT_NOTE}",
            }
        )
      row["log_url"] = _lookup_log_url(model, scan, num_slices, int(rec["hosts"]) * 8, "safetensor")

      # Create corresponding Runs entry for completeness.
      run_row = _empty_row(schema_v2.RUNS_COLUMNS)
      run_row.update(
          _v1_environment_common(
              model,
              num_slices,
              hosts_per_slice,
              topo,
              run_id=f"v1_multihost_{rec.get('source', 'lance')}",
              experiment_name="v1_multihost",
          )
      )
      run_row.update(ckpt)
      run_row.update(
          {
              "row_id": row_id,
              "phase_name": "safetensors_dynamic",
              "model_name": model,
              "scan_layers_mode": scan,
              "pipeline_name": "safetensors_dynamic",
              "run_status": "OK" if status == "OK" else "FAILED",
              "failure_category": "" if status == "OK" else str(row["run_status"]),
              "error_summary": "" if status == "OK" else note_raw,
              "result_json_url": multihost_csv,
          }
      )
      if logs is not None:
        cfg = bsh._multihost_cfg(rec)  # pylint: disable=protected-access
        p = logs.path(model, cfg, "safetensors_dynamic")
        if os.path.exists(p):
          run_row["result_json_url"] = p
          run_row["_local_json_path"] = p
        p_log = os.path.join(os.path.dirname(p), "safetensors_dynamic.log") if p else ""
        if p_log and os.path.exists(p_log):
          row["_local_log_files"] = [p_log]
          run_row["_local_log_files"] = [p_log]
      summary_out.append(row)
      stages_sd_out.append(sd_stage)
      runs_out.append(run_row)


def _add_multihost_to_maxtext_rows(
    *,
    logs: bt.Logs,
    cold_s1_by_model: Mapping[str, float],
    models_by_name: Mapping[str, dict[str, Any]],
    summary_out: list[dict[str, Any]],
    stages_tm_out: list[dict[str, Any]],
    runs_out: list[dict[str, Any]],
) -> None:
  """Ingests multi-host (`*_hosts*`) `to_maxtext_step3_result.json` results paired with single-host Step 1-2."""
  seen_ids = {r["row_id"] for r in summary_out}
  for model in logs.models():
    is_synthetic = model in _SYNTHETIC_DEEPSEEK_MODELS
    ckpt = _checkpoint_fields(models_by_name.get(model), is_synthetic)
    s1 = float(cold_s1_by_model.get(model, 0.0) or 0.0)
    for cfg in logs.configs(model):
      if "hosts" not in cfg:
        continue
      tm3, tm3_log = logs.load(model, cfg, "to_maxtext_step3")
      if tm3 is None:
        continue
      scan, num_slices, hosts_per_slice = parse_v1_config_dir(cfg)
      topo = topology_for_hosts_per_slice(hosts_per_slice)
      row_id = _make_v1_row_id(model, scan, "to_maxtext", num_slices, topo)
      if row_id in seen_ids:
        continue
      seen_ids.add(row_id)

      tm1, tm1_log = logs.load(model, scan, "to_maxtext_step1")
      tm2, tm2_log = logs.load(model, scan, "to_maxtext_step2")
      runs_out.append(
          adapt_single_v1_phase_json(
              logs.path(model, cfg, "to_maxtext_step3"),
              model_name=model,
              cfg=cfg,
              phase="to_maxtext_step3",
              run_id="v1_multihost_tm",
              experiment_name="v1_multihost_tm",
              models_by_name=models_by_name,
              cold_s1=s1,
          )
      )
      row = _empty_row(schema_v2.SUMMARY_COLUMNS)
      _apply_common_summary_identity(
          row,
          row_id=row_id,
          model_name=model,
          pipeline="to_maxtext",
          scan_mode=scan,
          num_slices=num_slices,
          hosts_per_slice=hosts_per_slice,
          topology=topo,
          ckpt_fields=ckpt,
      )
      tm2_ok = bt._ok(tm2)  # pylint: disable=protected-access
      tm3_ok = bt._ok(tm3)  # pylint: disable=protected-access
      if tm2_ok and tm3_ok:
        tm1_local = float((tm1 or {}).get("hf_to_local_s", 0.0) or 0.0) if bt._ok(tm1) else 0.0  # pylint: disable=protected-access
        tm2_s = float((tm2 or {}).get("convert_to_orbax_s", 0.0) or 0.0)
        tm3_s = float(tm3.get("load_orbax_s", 0.0) or 0.0)
        tm_no_hf = (0.0 if is_synthetic else tm1_local) + tm2_s + tm3_s
        tm_e2e = (0.0 if is_synthetic else s1) + tm_no_hf
        row.update(
            {
                "run_status": "OK",
                "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
                "to_maxtext_step1_total_seconds": "" if is_synthetic else _round_opt(tm1_local),
                "to_maxtext_step2_total_seconds": round(tm2_s, 2),
                "load_into_tpu_hbm_total_seconds": round(tm3_s, 2),
                "total_without_hf_download_seconds": round(tm_no_hf, 2),
                "total_end_to_end_seconds": "LOCAL" if is_synthetic else round(tm_e2e, 2),
                "memory_peak_host_ram_container_gb": _round_opt(tm3.get("peak_host_ram_gb")),
                "note": f"multihost to_maxtext Step 3 (Steps 1-2 shared from 1x2x2x1); {_V1_CAVEAT_NOTE}",
            }
        )
      else:
        st_code, _, err_sum, _, _ = classify_v1_failure(tm3, tm3_log)
        row.update(
            {
                "run_status": st_code,
                "step1_total_seconds": "LOCAL" if is_synthetic else (_round_opt(s1) if s1 > 0 else ""),
                "note": f"{err_sum}; {_V1_CAVEAT_NOTE}",
            }
        )
      row["_local_log_files"] = [p for p in (tm1_log, tm2_log, tm3_log) if p and os.path.exists(p)]
      summary_out.append(row)
      stages_tm_out.append(
          _make_tm_stage_row(
              row_id=row_id,
              model_name=model,
              scan_mode=scan,
              num_slices=num_slices,
              topology=topo,
              tm1=tm1,
              tm2=tm2,
              tm3=tm3,
              cold_s1=s1,
              is_synthetic=is_synthetic,
          )
      )


def _attach_pair_speedups(summary_out: list[dict[str, Any]]) -> None:
  """Populates speedup columns across paired `to_maxtext` and baseline `safetensors_dynamic` rows."""
  by_config: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = {}
  for r in summary_out:
    # Skip variant rows (e.g. `|with_fix`) when looking up canonical baseline speedups.
    if r["row_id"].count("|") > 5:
      continue
    key = (
        r["model_name"],
        r["scan_layers_mode"],
        int(r["number_of_slices"]),
        r["slice_topology"],
    )
    by_config.setdefault(key, {})[r["pipeline_name"]] = r

  for pair in by_config.values():
    tm = pair.get("to_maxtext")
    sd = pair.get("safetensors_dynamic")
    if not (tm and sd and tm["run_status"] == "OK" and sd["run_status"] == "OK"):
      continue
    sd_load = float(sd["load_into_tpu_hbm_total_seconds"] or 0.0)
    tm_load = float(tm["load_into_tpu_hbm_total_seconds"] or 0.0)
    tm_no_hf = float(tm["total_without_hf_download_seconds"] or 0.0)
    if sd_load > 0 and tm_load > 0 and tm_no_hf > 0:
      tm["speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic"] = 1.0
      tm["speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2"] = 1.0
      sd["speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic"] = round(tm_no_hf / sd_load, 2)
      sd["speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2"] = round(tm_load / sd_load, 2)
    tm_e2e = tm.get("total_end_to_end_seconds")
    sd_e2e = sd.get("total_end_to_end_seconds")
    if isinstance(tm_e2e, (int, float)) and isinstance(sd_e2e, (int, float)) and sd_e2e > 0:
      tm["speedup_end_to_end_to_maxtext_vs_safetensors_dynamic"] = 1.0
      sd["speedup_end_to_end_to_maxtext_vs_safetensors_dynamic"] = round(tm_e2e / sd_e2e, 2)


def adapt_v1_dataset(
    *,
    p2_roots: Sequence[str] = (),
    p1_roots: Sequence[str] = (),
    variant_roots: Sequence[tuple[str, Sequence[str]]] = (),
    before_roots: Sequence[str] = (),
    multihost_csv: str | None = None,
    cold_s1_overrides: Mapping[str, float] | None = None,
    models_cfg: Mapping[str, Any] | None = None,
) -> LegacyV1Dataset:
  """Adapts mirrored v1 result directories + CSV into complete v2 tab row lists."""
  models_by_name = _model_lookup(models_cfg)
  overrides = dict(cold_s1_overrides or {})
  all_main_roots = [r for r in (*p2_roots, *p1_roots) if os.path.isdir(r)]
  summary_out: list[dict[str, Any]] = []
  step1_out: list[dict[str, Any]] = []
  stages_sd_out: list[dict[str, Any]] = []
  stages_tm_out: list[dict[str, Any]] = []
  runs_out: list[dict[str, Any]] = []
  cold_s1_by_model: dict[str, float] = dict(overrides)

  for suite_name, roots in (("v1_p2", p2_roots), ("v1_p1", p1_roots)):
    valid_roots = [r for r in roots if os.path.isdir(r)]
    if not valid_roots:
      continue
    logs = bt.Logs(valid_roots)
    vlogs = [(lbl, bt.Logs([vr for vr in vroots if os.path.isdir(vr)])) for lbl, vroots in variant_roots]
    vlogs = [(lbl, vl) for lbl, vl in vlogs if vl.roots]
    base_rows = bt.build_rows(valid_roots, overrides)
    for b_row in base_rows:
      model = b_row["model"]
      if b_row["safetensor_1_hf_to_gcs_s"]:
        cold_s1_by_model.setdefault(model, float(b_row["safetensor_1_hf_to_gcs_s"]))
      _build_single_host_pair(
          logs=logs,
          summary_base=b_row,
          models_by_name=models_by_name,
          runs_out=runs_out,
          stages_sd_out=stages_sd_out,
          stages_tm_out=stages_tm_out,
          summary_out=summary_out,
          variant_logs=vlogs,
          run_id_prefix=suite_name,
      )

  # Build one Step 1 downloads row per model observed in v1.
  seen_models = sorted({r["model_name"] for r in summary_out})
  for model in seen_models:
    is_synthetic = model in _SYNTHETIC_DEEPSEEK_MODELS
    m_info = models_by_name.get(model, {})
    s1_s = cold_s1_by_model.get(model, 0.0)
    if not is_synthetic and s1_s <= 0.0:
      continue
    step1_row = _empty_row(schema_v2.STEP1_COLUMNS)
    step1_row.update(
        {
            "tpu_platform": "v7x",
            "step1_gcs_bucket_region": "us-central1",
            "model_name": model,
            "step1_hf_repo_id": m_info.get("hf_repo_id", ""),
            "step1_download_hf_to_gcs_seconds": "" if is_synthetic else round(s1_s, 2),
            "step1_download_hf_to_gcs_gb": m_info.get("hf_download_gb", "") if not is_synthetic else "",
            "step1_number_of_files": m_info.get("num_files", ""),
            "step1_total_seconds": "LOCAL" if is_synthetic else round(s1_s, 2),
            "step1_value_source": "local_synthetic_files" if is_synthetic else "cold_download",
            "step1_measured_by_run_id": "v1_tpu7x",
        }
    )
    step1_out.append(step1_row)

  if multihost_csv and os.path.exists(multihost_csv):
    _add_multihost_csv_rows(
        multihost_csv=multihost_csv,
        cold_s1_by_model=cold_s1_by_model,
        models_by_name=models_by_name,
        logs=bt.Logs(all_main_roots) if all_main_roots else None,
        summary_out=summary_out,
        stages_sd_out=stages_sd_out,
        runs_out=runs_out,
    )

  if all_main_roots:
    _add_multihost_to_maxtext_rows(
        logs=bt.Logs(all_main_roots),
        cold_s1_by_model=cold_s1_by_model,
        models_by_name=models_by_name,
        summary_out=summary_out,
        stages_tm_out=stages_tm_out,
        runs_out=runs_out,
    )

  _attach_pair_speedups(summary_out)

  # Optionally ingest pre-PR (`origin/main`) BEFORE phase JSONs as separate tagged Runs rows.
  for b_root in [r for r in before_roots if os.path.isdir(r)]:
    for res_path in sorted(glob.glob(os.path.join(b_root, "*", "*", "*_result.json"))):
      rel = os.path.relpath(res_path, b_root).split(os.sep)
      if len(rel) != 3:
        continue
      m_name, cfg_name, fname = rel
      phase_name = fname.removesuffix("_result.json")
      runs_out.append(
          adapt_single_v1_phase_json(
              res_path,
              model_name=m_name,
              cfg=cfg_name,
              phase=phase_name,
              run_id="v1_origin_main_before",
              experiment_name="v1_before",
              models_by_name=models_by_name,
              cold_s1=cold_s1_by_model.get(m_name),
              variant_tag="origin_main",
          )
      )

  return LegacyV1Dataset(
      summary_rows=summary_out,
      step1_rows=step1_out,
      stages_sd_rows=stages_sd_out,
      stages_tm_rows=stages_tm_out,
      runs_rows=runs_out,
  )


def _write_csv(
    path: str,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[schema_v2.ColumnSpec],
) -> None:
  """Writes `rows` to `path` as a UTF-8 CSV using `columns` field order."""
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  keys = [c.key for c in columns]
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in rows:
      writer.writerow({k: r.get(k, "") for k in keys})


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--p2-root", action="append", default=[])
  parser.add_argument("--p1-root", action="append", default=[])
  parser.add_argument("--variant-root", action="append", default=[], help="LABEL=PATH")
  parser.add_argument("--before-root", action="append", default=[])
  parser.add_argument("--multihost-csv", default="")
  parser.add_argument("--cold-s1", action="append", default=[], help="MODEL=SECONDS")
  parser.add_argument("--out-dir", required=True)
  args = parser.parse_args(argv)

  variants: dict[str, list[str]] = {}
  for spec in args.variant_root:
    lbl, path = spec.split("=", 1)
    variants.setdefault(lbl, []).append(path)
  overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.cold_s1)}

  ds = adapt_v1_dataset(
      p2_roots=args.p2_root,
      p1_roots=args.p1_root,
      variant_roots=list(variants.items()),
      before_roots=args.before_root,
      multihost_csv=args.multihost_csv or None,
      cold_s1_overrides=overrides,
  )
  errs = ds.validate()
  if errs:
    print(
        f"Validation failed with {len(errs)} errors (showing first 10):",
        file=sys.stderr,
    )
    for err in errs[:10]:
      print(f"  - {err}", file=sys.stderr)
    return 1

  _write_csv(
      os.path.join(args.out_dir, "summary_v1.csv"),
      ds.summary_rows,
      schema_v2.SUMMARY_COLUMNS,
  )
  _write_csv(
      os.path.join(args.out_dir, "step1_v1.csv"),
      ds.step1_rows,
      schema_v2.STEP1_COLUMNS,
  )
  _write_csv(
      os.path.join(args.out_dir, "stages_sd_v1.csv"),
      ds.stages_sd_rows,
      schema_v2.STAGES_SD_COLUMNS,
  )
  _write_csv(
      os.path.join(args.out_dir, "stages_tm_v1.csv"),
      ds.stages_tm_rows,
      schema_v2.STAGES_TM_COLUMNS,
  )
  _write_csv(os.path.join(args.out_dir, "runs_v1.csv"), ds.runs_rows, schema_v2.RUNS_COLUMNS)
  print(
      f"Wrote validated v1 rows: Summary={len(ds.summary_rows)}, Step1={len(ds.step1_rows)}, "
      f"StagesSD={len(ds.stages_sd_rows)}, StagesTM={len(ds.stages_tm_rows)}, Runs={len(ds.runs_rows)}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
