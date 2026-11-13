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

"""AOT remat selector and MFU/HBM metrics parser for DeepSeek-V4.1-Flash W2 (64c) and W3 (128c) benchmarks."""

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils

BASE_CONFIG_PATH = "src/maxtext/configs/base.yml"
# Per-TensorCore (1 JAX device = half of a 2-core TPU7x chip) BF16 peak TFLOP/s (2307.0 / 2).
TPU7X_BF16_PEAK_TFLOPS_PER_DEVICE = 1153.5
TPU7X_DEVICES_PER_CHIP = 2
# Physical per-device HBM capacity enforced by XLA's TPU7x heap simulator (94.74 GiB).
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 94.74.
TPU7X_HBM_CAPACITY_GIB = 94.74
# TODO(yaoyuchen): Revisit 85.0 GiB minimal-to-full remat threshold once v7x compiler HBM scheduling improves.
DEFAULT_REMAT_FALLBACK_HBM_GIB = 85.0
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 1.1.
TT6_MAX_HBM_RATIO = 1.1
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 1e-6.
TFLOPS_MATCH_ATOL = 1e-6
DEFAULT_STEADY_START_STEP = 10
DEFAULT_STEADY_END_STEP = 29

_DS41_MM_NORMS_JSON_RE = re.compile(r"DS41_MM_NORMS:\s*(\{[^\r\n}]*\})")
_STEP0_MOE_DROPPED_RE = re.compile(
    r"MOE_DROPPED_FRAC\s+step=(?P<step>\d+)\s+"
    r"ragged_buffer_factor=(?P<rbf>[-+]?\d+(?:\.\d+)?)\s+"
    r"mean=(?P<mean>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"max=(?P<max>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"per_layer=\[(?P<per_layer>[^\]]*)\]"
)
_PEAK_HBM_KV_RE = re.compile(r"\bpeak_hbm_gb=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\b")
_BYTES_IN_USE_KV_RE = re.compile(r"\bbytes_in_use_gib=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\b")
_STAGE_START_UTC_RE = re.compile(r"STAGE_START_UTC=(\S+)\s+(\S+)")
_STAGE_END_UTC_RE = re.compile(r"STAGE_END_UTC=(\S+)\s+(\S+)")

_PRESET_NAMES_BY_TOPOLOGY: dict[str, dict[str, str]] = {
    "64c": {"mm_minimal": "A8", "text_minimal": "A8t", "mm_full": "A8b", "text_full": "A8tb"},
    "128c": {"mm_minimal": "A9", "text_minimal": "A9t", "mm_full": "A9b", "text_full": "A9tb"},
}


def _locate_aot_entry(
    aot_payload: Mapping[str, Any],
    *,
    preset_key: str | None = None,
) -> Mapping[str, Any]:
  """Resolves the target preset dictionary inside a single-preset or batch AOT JSON payload."""
  has_key = preset_key is not None and len(preset_key.strip()) > 0
  if has_key:
    assert preset_key is not None
    if "results" in aot_payload and isinstance(aot_payload["results"], Mapping) and preset_key in aot_payload["results"]:
      entry = aot_payload["results"][preset_key]
      if not isinstance(entry, Mapping):
        raise ValueError(f"Expected mapping for AOT preset {preset_key!r}, got {type(entry).__name__}.")
      return entry
    if preset_key in aot_payload and isinstance(aot_payload[preset_key], Mapping):
      return aot_payload[preset_key]
    if "temp_hbm_gib" in aot_payload or "naive_sum_hbm_gib" in aot_payload or "compile_succeeded" in aot_payload:
      return aot_payload
    raise KeyError(f"Preset {preset_key!r} not found in AOT payload keys={sorted(aot_payload.keys())}.")
  if "temp_hbm_gib" in aot_payload or "naive_sum_hbm_gib" in aot_payload or "compile_succeeded" in aot_payload:
    return aot_payload
  raise KeyError(
      "AOT payload lacks unclamped HBM fields ('temp_hbm_gib'/'naive_sum_hbm_gib') and no preset_key supplied."
  )


def _is_recorded_aot_failure(entry: Mapping[str, Any]) -> bool:
  """Returns True when an AOT JSON entry records failed compilation, `passed=False`, or `temp_hbm_gib > limit`."""
  if "compile_succeeded" in entry and not bool(entry["compile_succeeded"]):
    return True
  if "passed" in entry and not bool(entry["passed"]):
    return True
  limit_gib = float(entry["tpu7x_heap_limit_gib"]) if "tpu7x_heap_limit_gib" in entry else TPU7X_HBM_CAPACITY_GIB
  if "temp_hbm_gib" in entry and entry["temp_hbm_gib"] is not None:
    if float(entry["temp_hbm_gib"]) > limit_gib:
      return True
  return False


def _build_scalar_aot_record(hbm_gib: float, *, preset_key: str = "") -> dict[str, Any]:
  """Builds an unclamped AOT record from a scalar `--*_hbm_gib` CLI override, enforcing `<= 94.74 GiB`."""
  val = float(hbm_gib)
  if not math.isfinite(val) or val <= 0.0:
    raise ValueError(f"Invalid scalar AOT HBM={val!r}; expected positive finite float.")
  if val > TPU7X_HBM_CAPACITY_GIB:
    raise AssertionError(
        f"Scalar AOT HBM {val:.6f} GiB exceeds TPU7x heap limit {TPU7X_HBM_CAPACITY_GIB:.2f} GiB (preset={preset_key!r})."
    )
  return {
      "preset": preset_key,
      "aot_compile_succeeded": True,
      "aot_temp_hbm_gib": val,
      "aot_non_aliased_io_hbm_gib": 0.0,
      "aot_naive_sum_hbm_gib": val,
      "aot_clamped_heap_cap_gib": round(min(TPU7X_HBM_CAPACITY_GIB, val), 6),
      "aot_tpu7x_heap_limit_gib": TPU7X_HBM_CAPACITY_GIB,
  }


def extract_aot_record(
    aot_payload: Mapping[str, Any],
    *,
    preset_key: str | None = None,
) -> dict[str, Any]:
  """Extracts unclamped AOT HBM components and explicit `aot_clamped_heap_cap_gib` from a passing AOT JSON entry."""
  target = _locate_aot_entry(aot_payload, preset_key=preset_key)
  if _is_recorded_aot_failure(target):
    err_detail = str(target["compile_error"]) if "compile_error" in target and target["compile_error"] else "passed=False"
    raise AssertionError(f"AOT preset {preset_key!r} failed Option B compilation/HBM gate: {err_detail}")
  for req_field in ("temp_hbm_gib", "non_aliased_io_hbm_gib", "naive_sum_hbm_gib", "compile_succeeded"):
    if req_field not in target or target[req_field] is None:
      raise KeyError(f"Selected AOT entry for {preset_key!r} lacks required field {req_field!r}: keys={sorted(target)}.")

  compile_ok = bool(target["compile_succeeded"])
  temp_gib = float(target["temp_hbm_gib"])
  io_gib = float(target["non_aliased_io_hbm_gib"])
  naive_gib = float(target["naive_sum_hbm_gib"])
  limit_gib = float(target["tpu7x_heap_limit_gib"]) if "tpu7x_heap_limit_gib" in target else TPU7X_HBM_CAPACITY_GIB
  if not compile_ok:
    raise AssertionError(f"AOT preset {preset_key!r} has compile_succeeded=False.")
  if not math.isfinite(temp_gib) or temp_gib <= 0.0:
    raise ValueError(f"Invalid temp_hbm_gib={temp_gib!r} in AOT preset {preset_key!r}.")
  if not math.isfinite(io_gib) or io_gib < 0.0:
    raise ValueError(f"Invalid non_aliased_io_hbm_gib={io_gib!r} in AOT preset {preset_key!r}.")
  if not math.isfinite(naive_gib) or naive_gib <= 0.0:
    raise ValueError(f"Invalid naive_sum_hbm_gib={naive_gib!r} in AOT preset {preset_key!r}.")
  if temp_gib > limit_gib or io_gib > limit_gib:
    raise AssertionError(
        f"AOT preset {preset_key!r} exceeded TPU7x heap limit {limit_gib:.2f} GiB: "
        f"temp_hbm_gib={temp_gib:.6f}, non_aliased_io_hbm_gib={io_gib:.6f}."
    )
  clamped_cap = (
      float(target["clamped_heap_cap_gib"])
      if "clamped_heap_cap_gib" in target and target["clamped_heap_cap_gib"] is not None
      else round(min(limit_gib, naive_gib), 6)
  )
  resolved_preset = str(target["preset"]) if "preset" in target else (preset_key if preset_key is not None else "")
  return {
      "preset": resolved_preset,
      "aot_compile_succeeded": compile_ok,
      "aot_temp_hbm_gib": temp_gib,
      "aot_non_aliased_io_hbm_gib": io_gib,
      "aot_naive_sum_hbm_gib": naive_gib,
      "aot_clamped_heap_cap_gib": clamped_cap,
      "aot_tpu7x_heap_limit_gib": limit_gib,
  }


def _resolve_optional_aot_record(
    *,
    json_path: str | None,
    hbm_gib: float | None,
    preset_key: str,
    label: str,
    allow_recorded_failure: bool = False,
) -> tuple[dict[str, Any] | None, bool]:
  """Resolves an optional `(json_path, hbm_gib)` pair into `(unclamped_aot_record_or_none, recorded_failure)`."""
  has_json = json_path is not None and len(json_path.strip()) > 0
  has_val = hbm_gib is not None
  if has_json and has_val:
    raise ValueError(f"Specify at most one of --{label}_json or --{label}_hbm_gib, not both.")
  if has_val:
    assert hbm_gib is not None
    if allow_recorded_failure and float(hbm_gib) > TPU7X_HBM_CAPACITY_GIB:
      return None, True
    return _build_scalar_aot_record(float(hbm_gib), preset_key=preset_key), False
  if has_json:
    assert json_path is not None
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
      raise ValueError(f"Expected JSON object in {json_path!r}, got {type(payload).__name__}.")
    if allow_recorded_failure:
      entry = _locate_aot_entry(payload, preset_key=preset_key)
      if _is_recorded_aot_failure(entry):
        return None, True
    return extract_aot_record(payload, preset_key=preset_key), False
  return None, False


def select_remat_and_aot_hbm(
    *,
    topology_mode: str,
    enabled_stages: Sequence[str],
    explicit_remat_policy: str | None = None,
    mm_aot_json: str | None = None,
    mm_aot_hbm_gib: float | None = None,
    mm_fallback_aot_json: str | None = None,
    mm_fallback_aot_hbm_gib: float | None = None,
    text_aot_json: str | None = None,
    text_aot_hbm_gib: float | None = None,
    text_fallback_aot_json: str | None = None,
    text_fallback_aot_hbm_gib: float | None = None,
    remat_fallback_hbm_gib: float = DEFAULT_REMAT_FALLBACK_HBM_GIB,
) -> dict[str, Any]:
  """Selects `remat_policy` (`minimal` vs `full`) from unclamped AOT records and fails closed if selected AOT failed."""
  if topology_mode not in _PRESET_NAMES_BY_TOPOLOGY:
    raise ValueError(f"Unsupported topology_mode={topology_mode!r}; expected one of {sorted(_PRESET_NAMES_BY_TOPOLOGY)}.")
  stage_set = {s.strip().upper() for s in enabled_stages if s.strip()}
  if not stage_set or not stage_set.issubset({"MM", "TEXT"}):
    raise ValueError(f"enabled_stages must be a non-empty subset of {{'MM', 'TEXT'}}, got {enabled_stages!r}.")

  presets = _PRESET_NAMES_BY_TOPOLOGY[topology_mode]
  mm_min_rec, mm_min_failed = _resolve_optional_aot_record(
      json_path=mm_aot_json,
      hbm_gib=mm_aot_hbm_gib,
      preset_key=presets["mm_minimal"],
      label="mm_aot",
      allow_recorded_failure=True,
  )
  mm_full_rec, mm_full_failed = _resolve_optional_aot_record(
      json_path=mm_fallback_aot_json,
      hbm_gib=mm_fallback_aot_hbm_gib,
      preset_key=presets["mm_full"],
      label="mm_fallback_aot",
      allow_recorded_failure=True,
  )
  txt_min_rec, txt_min_failed = _resolve_optional_aot_record(
      json_path=text_aot_json,
      hbm_gib=text_aot_hbm_gib,
      preset_key=presets["text_minimal"],
      label="text_aot",
      allow_recorded_failure=True,
  )
  txt_full_rec, txt_full_failed = _resolve_optional_aot_record(
      json_path=text_fallback_aot_json,
      hbm_gib=text_fallback_aot_hbm_gib,
      preset_key=presets["text_full"],
      label="text_fallback_aot",
      allow_recorded_failure=True,
  )

  fb_thresh = float(remat_fallback_hbm_gib)
  cleaned_explicit = explicit_remat_policy.strip() if explicit_remat_policy is not None else ""
  if cleaned_explicit:
    if cleaned_explicit not in ("minimal", "full"):
      raise ValueError(f"explicit_remat_policy must be 'minimal' or 'full', got {cleaned_explicit!r}.")
    selected_remat = cleaned_explicit
    selection_source = "explicit_override"
  else:
    primary_failed = ("MM" in stage_set and mm_min_failed) or ("TEXT" in stage_set and txt_min_failed)
    all_primary_present = ("MM" not in stage_set or mm_min_rec is not None) and (
        "TEXT" not in stage_set or txt_min_rec is not None
    )
    if primary_failed:
      selected_remat = "full"
      selection_source = "primary_aot_failed_fallback"
    elif all_primary_present:
      stage_min_recs: list[dict[str, Any]] = []
      if "MM" in stage_set and mm_min_rec is not None:
        stage_min_recs.append(mm_min_rec)
      if "TEXT" in stage_set and txt_min_rec is not None:
        stage_min_recs.append(txt_min_rec)
      if all(
          float(r["aot_temp_hbm_gib"]) <= fb_thresh and float(r["aot_naive_sum_hbm_gib"]) <= fb_thresh
          for r in stage_min_recs
      ):
        selected_remat = "minimal"
        selection_source = "aot_le_threshold"
      else:
        selected_remat = "full"
        selection_source = "aot_gt_threshold_fallback"
    elif ("MM" not in stage_set or mm_full_rec is not None) and ("TEXT" not in stage_set or txt_full_rec is not None):
      selected_remat = "full"
      selection_source = "fallback_aot_only"
    else:
      if "MM" in stage_set and mm_full_failed:
        _resolve_optional_aot_record(
            json_path=mm_fallback_aot_json,
            hbm_gib=mm_fallback_aot_hbm_gib,
            preset_key=presets["mm_full"],
            label="mm_fallback_aot",
            allow_recorded_failure=False,
        )
      if "TEXT" in stage_set and txt_full_failed:
        _resolve_optional_aot_record(
            json_path=text_fallback_aot_json,
            hbm_gib=text_fallback_aot_hbm_gib,
            preset_key=presets["text_full"],
            label="text_fallback_aot",
            allow_recorded_failure=False,
        )
      raise ValueError(
          "Cannot auto-select remat_policy: provide primary AOT HBM (--mm_aot_json / --mm_aot_hbm_gib or "
          "--text_aot_json / --text_aot_hbm_gib), fallback full-remat AOT HBM, or explicit REMAT_POLICY."
      )

  resolved_mm_rec: dict[str, Any] | None = None
  if "MM" in stage_set:
    if selected_remat == "minimal":
      if mm_min_failed:
        _resolve_optional_aot_record(
            json_path=mm_aot_json,
            hbm_gib=mm_aot_hbm_gib,
            preset_key=presets["mm_minimal"],
            label="mm_aot",
            allow_recorded_failure=False,
        )
      if mm_min_rec is None:
        raise ValueError("Stage MM with remat_policy='minimal' requires --mm_aot_json or --mm_aot_hbm_gib.")
      resolved_mm_rec = mm_min_rec
    else:
      if mm_full_failed:
        _resolve_optional_aot_record(
            json_path=mm_fallback_aot_json,
            hbm_gib=mm_fallback_aot_hbm_gib,
            preset_key=presets["mm_full"],
            label="mm_fallback_aot",
            allow_recorded_failure=False,
        )
      if mm_full_rec is not None:
        resolved_mm_rec = mm_full_rec
      elif selection_source == "explicit_override" and mm_min_rec is not None:
        resolved_mm_rec = mm_min_rec
      else:
        raise ValueError(
            f"Stage MM auto-selected remat_policy='full' (source={selection_source}) "
            f"requires {presets['mm_full']} fallback AOT report (--mm_fallback_aot_json or --mm_fallback_aot_hbm_gib)."
        )

  resolved_txt_rec: dict[str, Any] | None = None
  if "TEXT" in stage_set:
    if selected_remat == "minimal":
      if txt_min_failed:
        _resolve_optional_aot_record(
            json_path=text_aot_json,
            hbm_gib=text_aot_hbm_gib,
            preset_key=presets["text_minimal"],
            label="text_aot",
            allow_recorded_failure=False,
        )
      if txt_min_rec is None:
        raise ValueError("Stage TEXT with remat_policy='minimal' requires --text_aot_json or --text_aot_hbm_gib.")
      resolved_txt_rec = txt_min_rec
    else:
      if txt_full_failed:
        _resolve_optional_aot_record(
            json_path=text_fallback_aot_json,
            hbm_gib=text_fallback_aot_hbm_gib,
            preset_key=presets["text_full"],
            label="text_fallback_aot",
            allow_recorded_failure=False,
        )
      if txt_full_rec is not None:
        resolved_txt_rec = txt_full_rec
      elif selection_source == "explicit_override" and txt_min_rec is not None:
        resolved_txt_rec = txt_min_rec
      else:
        raise ValueError(
            f"Stage TEXT with auto-selected remat_policy='full' (source={selection_source}) requires "
            f"{presets['text_full']} fallback AOT report (--text_fallback_aot_json or --text_fallback_aot_hbm_gib)."
        )

  return {
      "topology_mode": topology_mode,
      "remat_policy": selected_remat,
      "selection_source": selection_source,
      "remat_fallback_hbm_gib": fb_thresh,
      "mm_aot": resolved_mm_rec,
      "text_aot": resolved_txt_rec,
  }


def parse_live_hbm_telemetry(log_text: str) -> dict[str, Any]:
  """Extracts per-step live TPU `peak_hbm_gb` and `bytes_in_use_gib` emitted on `DS41_MM_NORMS` lines."""
  peaks: list[float] = []
  in_use_vals: list[float] = []
  for raw_line in log_text.splitlines():
    if "DS41_MM_NORMS" not in raw_line:
      continue
    json_m = _DS41_MM_NORMS_JSON_RE.search(raw_line)
    parsed_peak: float | None = None
    parsed_in_use: float | None = None
    if json_m is not None:
      obj = json.loads(json_m.group(1))
      if isinstance(obj, Mapping):
        if "live_peak_hbm_gib" in obj and obj["live_peak_hbm_gib"] is not None:
          parsed_peak = float(obj["live_peak_hbm_gib"])
        elif "peak_hbm_gb" in obj and obj["peak_hbm_gb"] is not None:
          parsed_peak = float(obj["peak_hbm_gb"])
        if "bytes_in_use_gib" in obj and obj["bytes_in_use_gib"] is not None:
          parsed_in_use = float(obj["bytes_in_use_gib"])
    if parsed_peak is None:
      kv_m = _PEAK_HBM_KV_RE.search(raw_line)
      if kv_m is not None:
        parsed_peak = float(kv_m.group(1))
    if parsed_in_use is None:
      in_use_m = _BYTES_IN_USE_KV_RE.search(raw_line)
      if in_use_m is not None:
        parsed_in_use = float(in_use_m.group(1))
    if parsed_peak is not None:
      if not math.isfinite(parsed_peak) or parsed_peak <= 0.0:
        raise ValueError(f"Non-positive or non-finite DS41_MM_NORMS peak_hbm_gb={parsed_peak!r} in training log.")
      peaks.append(parsed_peak)
    if parsed_in_use is not None:
      if not math.isfinite(parsed_in_use) or parsed_in_use < 0.0:
        raise ValueError(f"Negative or non-finite DS41_MM_NORMS bytes_in_use_gib={parsed_in_use!r} in training log.")
      in_use_vals.append(parsed_in_use)
  return {
      "ds41_mm_norms_hbm_step_count": len(peaks),
      "live_peak_hbm_gib": max(peaks) if peaks else None,
      "live_max_bytes_in_use_gib": max(in_use_vals) if in_use_vals else None,
  }


def parse_step0_moe_dropped_frac(log_text: str) -> dict[str, Any] | None:
  """Parses `MOE_DROPPED_FRAC step=0 ...` from training stdout/stderr logs."""
  match = _STEP0_MOE_DROPPED_RE.search(log_text)
  if match is None:
    return None
  raw_layers = [tok.strip() for tok in match.group("per_layer").split(",") if tok.strip()]
  per_layer = [float(val) for val in raw_layers]
  mean_val = float(match.group("mean"))
  max_val = float(match.group("max"))
  return {
      "step": int(match.group("step")),
      "ragged_buffer_factor": float(match.group("rbf")),
      "mean": mean_val,
      "max": max_val,
      "per_layer": per_layer,
      "step0_moe_dropped_frac_mean": mean_val,
      "step0_moe_dropped_frac_max": max_val,
      "step0_moe_dropped_frac_per_layer": per_layer,
  }


def parse_metrics_jsonl(metrics_path: Path) -> dict[int, dict[str, Any]]:
  """Parses `metrics.jsonl` into a step-keyed dictionary, validating required scalar keys."""
  if not metrics_path.is_file():
    raise FileNotFoundError(f"Metrics JSONL file does not exist: {metrics_path}")
  required_keys = (
      "step",
      "learning/loss",
      "perf/step_time_seconds",
      "perf/per_device_tflops",
      "perf/per_device_tflops_per_sec",
      "perf/per_device_tokens_per_sec",
  )
  by_step: dict[int, dict[str, Any]] = {}
  raw_lines = metrics_path.read_text(encoding="utf-8").splitlines()
  for line_no, raw in enumerate(raw_lines, start=1):
    stripped = raw.strip()
    if not stripped:
      continue
    rec = json.loads(stripped)
    if not isinstance(rec, dict):
      raise ValueError(f"Expected JSON object at {metrics_path}:{line_no}, got {type(rec).__name__}.")
    missing = [k for k in required_keys if k not in rec]
    if missing:
      raise KeyError(f"Metrics record at {metrics_path}:{line_no} missing required keys {missing}.")
    step_idx = int(rec["step"])
    by_step[step_idx] = rec
  if not by_step:
    raise ValueError(f"Metrics JSONL file {metrics_path} contains zero step records.")
  return by_step


def compute_expected_per_device_tflops(
    *,
    use_multimodal: bool,
    topology_mode: str,
    per_device_batch_size: float = 1.0,
    max_target_length: int = 4096,
    max_num_images_per_example: int = 4,
    ds41_vision_grid: Sequence[int] = (39, 39),
    freeze_vision_encoder_params: bool = True,
    remat_policy: str = "minimal",
    base_config_path: str = BASE_CONFIG_PATH,
) -> dict[str, float]:
  """Computes exact expected per-device training and vision TFLOPs on `deepseek4.1-flash.yml`."""
  if topology_mode == "64c":
    compile_topology = "tpu7x-128"
    ici_fsdp = 8
    opt_offload = True
  elif topology_mode == "128c":
    compile_topology = "tpu7x-256"
    ici_fsdp = 16
    opt_offload = True
  else:
    raise ValueError(f"Unsupported topology_mode={topology_mode!r}; expected '64c' or '128c'.")

  common_kw: dict[str, Any] = {
      "model_name": "deepseek4.1-flash",
      "override_model_config": True,
      "run_name": "ds41_bench_tflops_calc",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "compile_topology": compile_topology,
      "compile_topology_num_slices": 1,
      "ici_fsdp_parallelism": ici_fsdp,
      "ici_expert_parallelism": 16,
      "optimizer_memory_host_offload": opt_offload,
      "grad_dtype": "bfloat16",
      "num_vocab_tiling": 8,
      "quantization_local_shard_count": 8,
      "sparse_matmul": True,
      "megablox": True,
      "use_tokamax_gmm": True,
      "use_gmm_v2": True,
      "allow_split_physical_axes": True,
      "ragged_buffer_factor": 2.0,
      "ds41_activation_quant": False,
      "capacity_factor": -1,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "dtype_mm": "bfloat16",
      "matmul_precision": "default",
      "remat_policy": remat_policy,
      "max_target_length": int(max_target_length),
      "per_device_batch_size": float(per_device_batch_size),
      "vision_encoder_block": "deepseek41",
      "ds41_vision_grid": list(ds41_vision_grid),
      "dataset_type": "synthetic",
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 0.0,
      "log_period": 1,
  }
  cfg_txt = pyconfig.initialize(
      [sys.argv[0], base_config_path],
      **common_kw,
      use_multimodal=False,
      freeze_vision_encoder_params=False,
      max_num_images_per_example=1,
  )
  backbone_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(cfg_txt, log=False)
  if not use_multimodal:
    return {
        "total_per_device_tflops": float(backbone_tflops),
        "backbone_per_device_tflops": float(backbone_tflops),
        "vision_per_device_tflops": 0.0,
    }
  cfg_mm = pyconfig.initialize(
      [sys.argv[0], base_config_path],
      **common_kw,
      use_multimodal=True,
      freeze_vision_encoder_params=bool(freeze_vision_encoder_params),
      max_num_images_per_example=int(max_num_images_per_example),
  )
  total_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(cfg_mm, log=False)
  vision_tflops = maxtext_utils.calculate_vision_encoder_tflops(cfg_mm)[0]
  if abs(float(total_tflops) - float(backbone_tflops + vision_tflops)) > 1e-9:
    raise AssertionError(
        f"Non-additive multimodal TFLOPs detected: total={total_tflops} vs "
        f"backbone={backbone_tflops} + vision={vision_tflops}."
    )
  return {
      "total_per_device_tflops": float(total_tflops),
      "backbone_per_device_tflops": float(backbone_tflops),
      "vision_per_device_tflops": float(vision_tflops),
  }


def format_stage_summary_table(summary: Mapping[str, Any]) -> str:
  """Renders a side-by-side MFU + Live TPU HBM vs unclamped AOT HBM Markdown summary table."""
  live_val = summary["live_peak_hbm_gib"]
  delta_t = summary["live_vs_aot_temp_delta_gib"]
  delta_n = summary["live_vs_aot_naive_delta_gib"]
  live_s = "MISSING" if live_val is None else f"{float(live_val):.6f}"
  delta_t_s = "N/A" if delta_t is None else f"{float(delta_t):+.6f}"
  delta_n_s = "N/A" if delta_n is None else f"{float(delta_n):+.6f}"
  stage_id = f"{summary['topology_mode']}/{summary['modality']}"
  headers = (
      "| stage | remat | mfu_pct | step_s | aot_compile_succeeded | live_peak_hbm_gib | "
      "aot_temp_hbm_gib | aot_non_aliased_io_hbm_gib | aot_naive_sum_hbm_gib | "
      "aot_clamped_heap_cap_gib | live_vs_aot_temp_delta_gib | live_vs_aot_naive_delta_gib |"
  )
  sep = "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
  row = (
      f"| {stage_id} | {summary['remat_policy']} | {float(summary['mfu_pct']):.2f} | "
      f"{float(summary['steady_median_step_time_s']):.4f} | {bool(summary['aot_compile_succeeded'])} | "
      f"{live_s} | {float(summary['aot_temp_hbm_gib']):.6f} | "
      f"{float(summary['aot_non_aliased_io_hbm_gib']):.6f} | {float(summary['aot_naive_sum_hbm_gib']):.6f} | "
      f"{float(summary['aot_clamped_heap_cap_gib']):.6f} | {delta_t_s} | {delta_n_s} |"
  )
  moe_mode = summary["moe_routing_mode"] if "moe_routing_mode" in summary else None
  rbf_val = summary["ragged_buffer_factor"] if "ragged_buffer_factor" in summary else None
  drop_obj = summary["step0_moe_dropped_frac"] if "step0_moe_dropped_frac" in summary else None
  if moe_mode is not None or rbf_val is not None or drop_obj is not None:
    headers += " moe_routing_mode | ragged_buffer_factor | step0_moe_dropped_frac |"
    sep += " :--- | ---: | :--- |"
    moe_mode_s = "N/A" if moe_mode is None else str(moe_mode)
    rbf_s = "N/A" if rbf_val is None else f"{float(rbf_val)}"
    drop_s = "N/A" if drop_obj is None else f"mean={float(drop_obj['mean']):.6f} max={float(drop_obj['max']):.6f}"
    row += f" {moe_mode_s} | {rbf_s} | {drop_s} |"
  return "\n".join((headers, sep, row))


def summarize_benchmark_stage(
    *,
    metrics_path: Path,
    log_path: Path,
    summary_json_path: Path,
    topology_mode: str,
    modality: str,
    remat_policy: str,
    aot_record: Mapping[str, Any],
    require_live_hbm: bool = True,
    require_step0_moe_dropped_frac: bool | None = None,
    bench_steps: int = 30,
    xprof_steps: int = 3,
    steady_start_step: int = DEFAULT_STEADY_START_STEP,
    steady_end_step: int = DEFAULT_STEADY_END_STEP,
    per_device_batch_size: float = 1.0,
    max_target_length: int = 4096,
    max_num_images_per_example: int = 4,
    ds41_vision_grid: Sequence[int] = (39, 39),
    freeze_vision_encoder_params: bool = True,
    xprof_trace_gcs_uri: str = "",
    stage_start_utc: str = "",
    stage_end_utc: str = "",
    expected_per_device_tflops_override: float | None = None,
) -> dict[str, Any]:
  """Parses benchmark metrics and `DS41_MM_NORMS` live HBM, enforces Option B + TT6 gates, and writes summary JSON."""
  if topology_mode not in ("64c", "128c"):
    raise ValueError(f"topology_mode must be '64c' or '128c', got {topology_mode!r}.")
  if modality not in ("mm", "text"):
    raise ValueError(f"modality must be 'mm' or 'text', got {modality!r}.")
  if remat_policy not in ("minimal", "full"):
    raise ValueError(f"remat_policy must be 'minimal' or 'full', got {remat_policy!r}.")
  if bench_steps <= 0 or steady_start_step < 0 or steady_end_step < steady_start_step or steady_end_step >= bench_steps:
    raise ValueError(f"Invalid step window: bench_steps={bench_steps}, steady=[{steady_start_step}, {steady_end_step}].")
  if xprof_steps < 0:
    raise ValueError(f"xprof_steps must be >= 0, got {xprof_steps}.")

  aot_compile_ok = bool(aot_record["aot_compile_succeeded"])
  aot_temp_gib = float(aot_record["aot_temp_hbm_gib"])
  aot_io_gib = float(aot_record["aot_non_aliased_io_hbm_gib"])
  aot_naive_gib = float(aot_record["aot_naive_sum_hbm_gib"])
  aot_limit_gib = (
      float(aot_record["aot_tpu7x_heap_limit_gib"])
      if "aot_tpu7x_heap_limit_gib" in aot_record
      else TPU7X_HBM_CAPACITY_GIB
  )
  aot_clamped_cap_gib = (
      float(aot_record["aot_clamped_heap_cap_gib"])
      if "aot_clamped_heap_cap_gib" in aot_record
      else round(min(aot_limit_gib, aot_naive_gib), 6)
  )
  if not aot_compile_ok or aot_temp_gib <= 0.0 or aot_temp_gib > aot_limit_gib:
    raise AssertionError(
        f"Invalid or failed AOT record supplied to summarize_benchmark_stage: "
        f"aot_compile_succeeded={aot_compile_ok}, aot_temp_hbm_gib={aot_temp_gib}, limit={aot_limit_gib}."
    )

  if not log_path.is_file():
    raise FileNotFoundError(f"Training log file does not exist: {log_path}")
  log_text = log_path.read_text(encoding="utf-8")
  by_step = parse_metrics_jsonl(metrics_path)

  failure_reasons: list[str] = []
  missing_bench_steps = [s for s in range(bench_steps) if s not in by_step]
  if missing_bench_steps:
    failure_reasons.append(f"missing_bench_steps={missing_bench_steps}")

  nan_or_inf_steps: list[int] = []
  for step_idx in sorted(by_step):
    loss_val = float(by_step[step_idx]["learning/loss"])
    if not math.isfinite(loss_val) or loss_val <= 0.0:
      nan_or_inf_steps.append(step_idx)
  loss_finite_all_bench_steps = len(missing_bench_steps) == 0 and all(
      s not in nan_or_inf_steps for s in range(bench_steps)
  )
  if nan_or_inf_steps:
    failure_reasons.append(f"non_finite_or_non_positive_loss_steps={nan_or_inf_steps}")

  use_mm = modality == "mm"
  if expected_per_device_tflops_override is not None:
    expected_tflops = float(expected_per_device_tflops_override)
    tflops_breakdown = {
        "total_per_device_tflops": expected_tflops,
        "backbone_per_device_tflops": expected_tflops,
        "vision_per_device_tflops": 0.0,
    }
  else:
    tflops_breakdown = compute_expected_per_device_tflops(
        use_multimodal=use_mm,
        topology_mode=topology_mode,
        per_device_batch_size=per_device_batch_size,
        max_target_length=max_target_length,
        max_num_images_per_example=max_num_images_per_example,
        ds41_vision_grid=ds41_vision_grid,
        freeze_vision_encoder_params=freeze_vision_encoder_params,
        remat_policy=remat_policy,
    )
    expected_tflops = tflops_breakdown["total_per_device_tflops"]

  steady_steps = [s for s in range(steady_start_step, steady_end_step + 1) if s in by_step]
  if len(steady_steps) != (steady_end_step - steady_start_step + 1):
    raise ValueError(
        f"Incomplete steady-state step window [{steady_start_step}..{steady_end_step}]: present={steady_steps}."
    )

  steady_step_times_s = [float(by_step[s]["perf/step_time_seconds"]) for s in steady_steps]
  if any((not math.isfinite(t) or t <= 0.0) for t in steady_step_times_s):
    failure_reasons.append(f"invalid_steady_step_times={steady_step_times_s}")

  logged_per_device_tflops = float(by_step[steady_steps[0]]["perf/per_device_tflops"])
  tflops_verified = math.isfinite(logged_per_device_tflops) and (
      abs(logged_per_device_tflops - expected_tflops) <= TFLOPS_MATCH_ATOL
  )
  if not tflops_verified:
    failure_reasons.append(
        f"per_device_tflops_mismatch: logged={logged_per_device_tflops:.9f} expected={expected_tflops:.9f}"
    )

  steady_median_s = float(statistics.median(steady_step_times_s))
  steady_mean_s = float(statistics.fmean(steady_step_times_s))
  steady_min_s = float(min(steady_step_times_s))
  steady_max_s = float(max(steady_step_times_s))

  tflops_per_sec_dev = expected_tflops / steady_median_s
  mean_tflops_per_sec_dev = expected_tflops / steady_mean_s
  logged_steady_tflops_per_sec = [float(by_step[s]["perf/per_device_tflops_per_sec"]) for s in steady_steps]
  logged_median_tflops_per_sec_dev = float(statistics.median(logged_steady_tflops_per_sec))
  mfu_pct = (tflops_per_sec_dev / TPU7X_BF16_PEAK_TFLOPS_PER_DEVICE) * 100.0

  tokens_per_dev = float(per_device_batch_size) * float(max_target_length)
  tokens_per_sec_dev = tokens_per_dev / steady_median_s
  tokens_per_sec_chip = float(TPU7X_DEVICES_PER_CHIP) * tokens_per_sec_dev
  tflops_per_sec_chip = float(TPU7X_DEVICES_PER_CHIP) * tflops_per_sec_dev

  resolved_require_step0_moe = (
      os.environ.get("DS41_LOG_STEP0_MOE_DROPPED") == "1"
      if require_step0_moe_dropped_frac is None
      else bool(require_step0_moe_dropped_frac)
  )
  hbm_telemetry = parse_live_hbm_telemetry(log_text)
  moe_dropped_info = parse_step0_moe_dropped_frac(log_text)
  if moe_dropped_info is not None:
    rbf_val = float(moe_dropped_info["ragged_buffer_factor"])
    moe_routing_mode: str | None = "token_dropping" if rbf_val > 0.0 else "dropless"
    ragged_buffer_factor_val: float | None = rbf_val
    step0_moe_dropped_frac_dict: dict[str, Any] | None = {
        "mean": moe_dropped_info["mean"],
        "max": moe_dropped_info["max"],
        "per_layer": moe_dropped_info["per_layer"],
    }
  else:
    moe_routing_mode = None
    ragged_buffer_factor_val = None
    step0_moe_dropped_frac_dict = None
    if resolved_require_step0_moe:
      failure_reasons.append("missing_step0_moe_dropped_frac_telemetry")
  live_peak_hbm_gib: float | None = hbm_telemetry["live_peak_hbm_gib"]
  live_max_bytes_in_use_gib: float | None = hbm_telemetry["live_max_bytes_in_use_gib"]
  tt6_max_allowed_hbm_gb = min(aot_limit_gib, TT6_MAX_HBM_RATIO * aot_clamped_cap_gib)

  if live_peak_hbm_gib is None:
    tt6_hbm_ratio: float | None = None
    tt6_hbm_passed = not bool(require_live_hbm)
    live_vs_aot_temp_delta_gib: float | None = None
    live_vs_aot_naive_delta_gib: float | None = None
    if require_live_hbm:
      failure_reasons.append("missing_live_peak_hbm_gib_in_ds41_mm_norms")
  else:
    live_val = float(live_peak_hbm_gib)
    live_vs_aot_temp_delta_gib = live_val - aot_temp_gib
    live_vs_aot_naive_delta_gib = live_val - aot_naive_gib
    tt6_hbm_ratio = live_val / aot_clamped_cap_gib
    within_heap_limit = bool(math.isfinite(live_val) and 0.0 < live_val <= aot_limit_gib)
    if not within_heap_limit:
      failure_reasons.append(
          f"live_peak_hbm_exceeded_tpu7x_limit: live_peak_hbm_gib={live_val:.6f} > {aot_limit_gib:.2f}"
      )
    tt6_hbm_passed = bool(within_heap_limit and live_val <= tt6_max_allowed_hbm_gb)
    if not tt6_hbm_passed and within_heap_limit:
      failure_reasons.append(
          f"tt6_peak_hbm_exceeded: live_peak_hbm_gib={live_val:.6f} > "
          f"tt6_max_allowed_hbm_gb={tt6_max_allowed_hbm_gb:.6f}"
      )

  cleaned_xprof_uri = xprof_trace_gcs_uri.strip()
  if xprof_steps > 0 and not cleaned_xprof_uri:
    failure_reasons.append("missing_xprof_trace_gcs_uri_when_xprof_steps_gt_0")

  resolved_start_utc = stage_start_utc.strip()
  if not resolved_start_utc:
    start_m = _STAGE_START_UTC_RE.search(log_text)
    if start_m is not None:
      resolved_start_utc = start_m.group(2)
  resolved_end_utc = stage_end_utc.strip()
  if not resolved_end_utc:
    end_m = _STAGE_END_UTC_RE.search(log_text)
    if end_m is not None:
      resolved_end_utc = end_m.group(2)

  first_loss = float(by_step[0]["learning/loss"]) if 0 in by_step else float("nan")
  last_bench_step = bench_steps - 1
  last_bench_loss = float(by_step[last_bench_step]["learning/loss"]) if last_bench_step in by_step else float("nan")
  passed = len(failure_reasons) == 0

  summary: dict[str, Any] = {
      "passed": passed,
      "failure_reasons": failure_reasons,
      "topology_mode": topology_mode,
      "modality": modality,
      "use_multimodal": use_mm,
      "remat_policy": remat_policy,
      "bench_steps": int(bench_steps),
      "xprof_steps": int(xprof_steps),
      "total_recorded_steps": len(by_step),
      "steady_window_steps": [int(steady_start_step), int(steady_end_step)],
      "steady_step_count": len(steady_steps),
      "steady_step_times_s": steady_step_times_s,
      "steady_median_step_time_s": steady_median_s,
      "steady_mean_step_time_s": steady_mean_s,
      "steady_min_step_time_s": steady_min_s,
      "steady_max_step_time_s": steady_max_s,
      "per_device_tflops": expected_tflops,
      "logged_per_device_tflops": logged_per_device_tflops,
      "backbone_per_device_tflops": tflops_breakdown["backbone_per_device_tflops"],
      "vision_per_device_tflops": tflops_breakdown["vision_per_device_tflops"],
      "tflops_verified": tflops_verified,
      "tflops_per_sec_per_device": tflops_per_sec_dev,
      "mean_tflops_per_sec_per_device": mean_tflops_per_sec_dev,
      "logged_median_tflops_per_sec_per_device": logged_median_tflops_per_sec_dev,
      "tflops_per_sec_per_chip": tflops_per_sec_chip,
      "tpu7x_bf16_peak_tflops_per_device": TPU7X_BF16_PEAK_TFLOPS_PER_DEVICE,
      "mfu_pct": mfu_pct,
      "tokens_per_sec_per_device": tokens_per_sec_dev,
      "tokens_per_sec_per_chip": tokens_per_sec_chip,
      "aot_preset": str(aot_record["preset"]) if "preset" in aot_record else "",
      "aot_compile_succeeded": aot_compile_ok,
      "aot_temp_hbm_gib": aot_temp_gib,
      "aot_non_aliased_io_hbm_gib": aot_io_gib,
      "aot_naive_sum_hbm_gib": aot_naive_gib,
      "aot_clamped_heap_cap_gib": aot_clamped_cap_gib,
      "tpu7x_heap_limit_gib": aot_limit_gib,
      "require_live_hbm": bool(require_live_hbm),
      "ds41_mm_norms_hbm_step_count": int(hbm_telemetry["ds41_mm_norms_hbm_step_count"]),
      "live_max_bytes_in_use_gib": live_max_bytes_in_use_gib,
      "live_peak_hbm_gib": live_peak_hbm_gib,
      "peak_hbm_gb": live_peak_hbm_gib,
      "live_vs_aot_temp_delta_gib": live_vs_aot_temp_delta_gib,
      "live_vs_aot_naive_delta_gib": live_vs_aot_naive_delta_gib,
      "tt6_max_hbm_ratio": TT6_MAX_HBM_RATIO,
      "tt6_max_allowed_hbm_gb": tt6_max_allowed_hbm_gb,
      "tt6_hbm_ratio": tt6_hbm_ratio,
      "tt6_hbm_passed": tt6_hbm_passed,
      "loss_finite_all_bench_steps": loss_finite_all_bench_steps,
      "nan_or_inf_steps": nan_or_inf_steps,
      "first_step_loss": first_loss,
      "last_bench_step_loss": last_bench_loss,
      "moe_routing_mode": moe_routing_mode,
      "ragged_buffer_factor": ragged_buffer_factor_val,
      "require_step0_moe_dropped_frac": resolved_require_step0_moe,
      "step0_moe_dropped_frac": step0_moe_dropped_frac_dict,
      "step0_moe_dropped_frac_mean": None if moe_dropped_info is None else moe_dropped_info["mean"],
      "step0_moe_dropped_frac_max": None if moe_dropped_info is None else moe_dropped_info["max"],
      "step0_moe_dropped_frac_per_layer": (None if moe_dropped_info is None else moe_dropped_info["per_layer"]),
      "xprof_trace_gcs_uri": cleaned_xprof_uri,
      "stage_start_utc": resolved_start_utc,
      "stage_end_utc": resolved_end_utc,
  }
  summary_json_path.parent.mkdir(parents=True, exist_ok=True)
  summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(format_stage_summary_table(summary), flush=True)
  if not passed:
    raise AssertionError(f"Benchmark stage {topology_mode}/{modality} failed gates: {failure_reasons}")
  return summary


def _str2bool(val: str) -> bool:
  """Parses a CLI boolean string (`true`/`false`/`1`/`0`/`yes`/`no`)."""
  lower = val.strip().lower()
  if lower in ("true", "1", "yes", "y"):
    return True
  if lower in ("false", "0", "no", "n"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean string, got {val!r}")


def _parse_grid(val: str) -> list[int]:
  """Parses a 2-element vision grid CLI string (`'H,W'` or `'[H,W]'`)."""
  cleaned = val.strip()
  if cleaned.startswith("[") and cleaned.endswith("]"):
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list) or len(parsed) != 2:
      raise argparse.ArgumentTypeError(f"Expected 2-element grid list, got {val!r}")
    return [int(parsed[0]), int(parsed[1])]
  parts = [p.strip() for p in cleaned.split(",") if p.strip()]
  if len(parts) != 2:
    raise argparse.ArgumentTypeError(f"Expected 'H,W' or '[H,W]' vision grid, got {val!r}")
  return [int(parts[0]), int(parts[1])]


def _resolve_stage_aot_record_from_cli(args: argparse.Namespace) -> dict[str, Any]:
  """Resolves the stage AOT record for `--mode=summarize_bench` from selection JSON, AOT JSON, or scalar HBM."""
  if args.aot_selection_json is not None and len(args.aot_selection_json.strip()) > 0:
    sel_payload = json.loads(Path(args.aot_selection_json).read_text(encoding="utf-8"))
    if not isinstance(sel_payload, Mapping):
      raise ValueError(f"Expected JSON object in {args.aot_selection_json!r}.")
    stage_key = "mm_aot" if args.modality == "mm" else "text_aot"
    if stage_key not in sel_payload or not isinstance(sel_payload[stage_key], Mapping):
      raise KeyError(f"Stage AOT record {stage_key!r} missing from {args.aot_selection_json!r}.")
    return dict(sel_payload[stage_key])

  rec, _ = _resolve_optional_aot_record(
      json_path=args.aot_summary_json,
      hbm_gib=args.aot_hbm_gib,
      preset_key=args.aot_preset_key if args.aot_preset_key is not None else "",
      label="aot",
      allow_recorded_failure=False,
  )
  if rec is None:
    raise ValueError("--mode=summarize_bench requires --aot_selection_json, --aot_summary_json, or --aot_hbm_gib.")
  return rec


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for AOT remat policy selection and benchmark metrics summarization."""
  parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash W2/W3 benchmark AOT selector & metrics parser.")
  parser.add_argument(
      "--mode",
      type=str,
      required=True,
      choices=("select_remat", "summarize_bench"),
      help="Operation mode.",
  )
  parser.add_argument("--topology_mode", type=str, required=True, choices=("64c", "128c"))
  parser.add_argument("--enabled_stages", type=str, default="MM,TEXT")
  parser.add_argument("--explicit_remat_policy", type=str, default="")
  parser.add_argument("--mm_aot_json", type=str, default=None)
  parser.add_argument("--mm_aot_hbm_gib", type=float, default=None)
  parser.add_argument("--mm_fallback_aot_json", type=str, default=None)
  parser.add_argument("--mm_fallback_aot_hbm_gib", type=float, default=None)
  parser.add_argument("--text_aot_json", type=str, default=None)
  parser.add_argument("--text_aot_hbm_gib", type=float, default=None)
  parser.add_argument("--text_fallback_aot_json", type=str, default=None)
  parser.add_argument("--text_fallback_aot_hbm_gib", type=float, default=None)
  parser.add_argument("--remat_fallback_hbm_gib", type=float, default=DEFAULT_REMAT_FALLBACK_HBM_GIB)
  parser.add_argument("--output_json", type=str, default=None)

  # summarize_bench arguments
  parser.add_argument("--modality", type=str, choices=("mm", "text"), default=None)
  parser.add_argument("--remat_policy", type=str, choices=("minimal", "full"), default=None)
  parser.add_argument("--aot_selection_json", type=str, default=None)
  parser.add_argument("--aot_hbm_gib", type=float, default=None)
  parser.add_argument("--aot_summary_json", type=str, default=None)
  parser.add_argument("--aot_preset_key", type=str, default=None)
  parser.add_argument("--require_live_hbm", type=_str2bool, default=True)
  parser.add_argument("--require_step0_moe_dropped_frac", type=_str2bool, default=None)
  parser.add_argument("--metrics_file", type=str, default=None)
  parser.add_argument("--log_file", type=str, default=None)
  parser.add_argument("--summary_json", type=str, default=None)
  parser.add_argument("--bench_steps", type=int, default=30)
  parser.add_argument("--xprof_steps", type=int, default=3)
  parser.add_argument("--steady_start_step", type=int, default=DEFAULT_STEADY_START_STEP)
  parser.add_argument("--steady_end_step", type=int, default=DEFAULT_STEADY_END_STEP)
  parser.add_argument("--per_device_batch_size", type=float, default=1.0)
  parser.add_argument("--max_target_length", type=int, default=4096)
  parser.add_argument("--max_num_images_per_example", type=int, default=4)
  parser.add_argument("--ds41_vision_grid", type=_parse_grid, default=(39, 39))
  parser.add_argument("--freeze_vision_encoder_params", type=_str2bool, default=True)
  parser.add_argument("--xprof_trace_gcs_uri", type=str, default="")
  parser.add_argument("--stage_start_utc", type=str, default="")
  parser.add_argument("--stage_end_utc", type=str, default="")
  parser.add_argument("--expected_per_device_tflops_override", type=float, default=None)
  args = parser.parse_args(argv)

  if args.mode == "select_remat":
    stages = [s.strip() for s in args.enabled_stages.split(",") if s.strip()]
    selection = select_remat_and_aot_hbm(
        topology_mode=args.topology_mode,
        enabled_stages=stages,
        explicit_remat_policy=args.explicit_remat_policy,
        mm_aot_json=args.mm_aot_json,
        mm_aot_hbm_gib=args.mm_aot_hbm_gib,
        mm_fallback_aot_json=args.mm_fallback_aot_json,
        mm_fallback_aot_hbm_gib=args.mm_fallback_aot_hbm_gib,
        text_aot_json=args.text_aot_json,
        text_aot_hbm_gib=args.text_aot_hbm_gib,
        text_fallback_aot_json=args.text_fallback_aot_json,
        text_fallback_aot_hbm_gib=args.text_fallback_aot_hbm_gib,
        remat_fallback_hbm_gib=args.remat_fallback_hbm_gib,
    )
    rendered = json.dumps(selection, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output_json:
      Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    return 0

  if args.modality is None or args.remat_policy is None or args.metrics_file is None or args.log_file is None:
    raise ValueError("--mode=summarize_bench requires --modality, --remat_policy, --metrics_file, and --log_file.")
  if args.summary_json is None:
    raise ValueError("--mode=summarize_bench requires --summary_json.")

  stage_aot_rec = _resolve_stage_aot_record_from_cli(args)
  summary = summarize_benchmark_stage(
      metrics_path=Path(args.metrics_file),
      log_path=Path(args.log_file),
      summary_json_path=Path(args.summary_json),
      topology_mode=args.topology_mode,
      modality=args.modality,
      remat_policy=args.remat_policy,
      aot_record=stage_aot_rec,
      require_live_hbm=args.require_live_hbm,
      require_step0_moe_dropped_frac=args.require_step0_moe_dropped_frac,
      bench_steps=args.bench_steps,
      xprof_steps=args.xprof_steps,
      steady_start_step=args.steady_start_step,
      steady_end_step=args.steady_end_step,
      per_device_batch_size=args.per_device_batch_size,
      max_target_length=args.max_target_length,
      max_num_images_per_example=args.max_num_images_per_example,
      ds41_vision_grid=args.ds41_vision_grid,
      freeze_vision_encoder_params=args.freeze_vision_encoder_params,
      xprof_trace_gcs_uri=args.xprof_trace_gcs_uri,
      stage_start_utc=args.stage_start_utc,
      stage_end_utc=args.stage_end_utc,
      expected_per_device_tflops_override=args.expected_per_device_tflops_override,
  )
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
