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

"""Builds the tpu7x summary CSV and the row-by-row `to_maxtext` vs `safetensor` markdown table.

Reads local mirrors of one or more `<gcs-base>/run_<run-id>/logs/` trees (the per-phase `<phase>_result.json`
files, plus the `<phase>.log` files when present, which give failure reasons) and writes:
  * a CSV with the same columns as `reports/benchmark_ckpt_load_time_summary.csv` (v5p), plus a status column per
    pipeline, and
  * a markdown table with a `to_maxtext` row and a `safetensor` row per (model, scan mode, slices).

The cold HF->GCS time (Step 1) is measured once per model and shared by both pipelines and every configuration.
It is the largest Step 1 time recorded for the model (warm cache checks take ~0.2-1 s);
`--cold-s1 MODEL=SECONDS` overrides it when no record of this run dir holds the cold download.

Usage (later mirrors override earlier ones file by file, so list re-run dirs after the runs they repeat):
  B=gs://mesa-maxtext/mesakh_tpu7x_bench
  gcloud storage rsync -r $B/p2_lance_prs/run_tpu7x_16m/logs /tmp/t7/p2
  gcloud storage rsync -r $B/p2_lance_prs/run_tpu7x_16m_rerun/logs /tmp/t7/p2_rerun
  gcloud storage rsync -r $B/p1_lance_prs/run_tpu7x_200b/logs /tmp/t7/p1
  python3 checkpoint_load_benchmark/scripts/build_tpu7x_tables.py /tmp/t7/p2 /tmp/t7/p2_rerun /tmp/t7/p1 \\
      --cold-s1 qwen3.5-397b-a17b=814.53 \\
      --csv checkpoint_load_benchmark/reports/tpu7x_benchmark_ckpt_load_time_summary.csv --md /tmp/t7/table.md \\
      --md-into checkpoint_load_benchmark/reports/tpu7x_benchmark_ckpt_load_time_summary.md

The 397B override is the cold download recorded in `results/p1_st_lance_b.csv` (its result JSON was later replaced).
"""

import argparse
import csv
import glob
import json
import os
import re

# Table order: by size within each tier, as in the v5p summary.
MODEL_ORDER = [
    "qwen3-0.6b",
    "gemma2-2b",
    "qwen3-4b",
    "gemma3-4b",
    "llama3.1-8b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "deepseek2-16b",
    "gpt-oss-20b",
    "qwen3-30b-a3b",
    "mixtral-8x7b",
    "llama3.1-70b",
    "gpt-oss-120b",
    "qwen3-235b-a22b",
    "deepseek4-284b",
    "qwen3.5-397b-a17b",
    "qwen3-480b-a35b",
    "deepseek3-671b",
]

CSV_FIELDS = [
    "model",
    "scan_mode",
    "num_slices",
    "global_devices",
    "to_maxtext_1_hf_to_gcs_s",
    "to_maxtext_1b_hf_to_local_s",
    "to_maxtext_2_convert_to_orbax_s",
    "to_maxtext_2a_init_abstract_model_s",
    "to_maxtext_2b_transform_and_shard_s",
    "to_maxtext_2c_orbax_save_to_gcs_s",
    "to_maxtext_3_load_orbax_s",
    "to_maxtext_total_s",
    "safetensor_1_hf_to_gcs_s",
    "safetensor_1_worker_upload_s",
    "safetensor_1_warm_cache_s",
    "safetensor_2_safetensors_and_convert_s",
    "safetensor_2a_derive_mappings_s",
    "safetensor_2b_load_sharded_hf_s",
    "safetensor_2c_transform_hf_to_mt_s",
    "safetensor_total_s",
    "speedup_total_vs_to_maxtext_x",
    "load_time_diff_safetensor_step2_vs_orbax_step3_s",
    "to_maxtext_status",
    "safetensor_status",
]


def _load(path: str) -> dict | None:
  try:
    with open(path, encoding="utf-8") as f:
      return json.load(f)
  except (OSError, ValueError):
    return None


def _ok(rec: dict | None) -> bool:
  return rec is not None and rec.get("status") != "FAILED"


def failure_reason(rec: dict | None, log_path: str) -> str:
  """Short reason for a failed phase, from its result JSON and (when mirrored) its logs.

  A multi-slice phase writes one log per slice (`<phase>.log` for slice 0, `<phase>_slice<N>.log` for the others),
  and the error that ended the run can be in any of them.
  """
  if rec is None:
    return "not run"
  err = rec.get("error", "") or ""
  log_tail = ""
  for path in [log_path] + sorted(glob.glob(log_path.removesuffix(".log") + "_slice*.log")):
    if os.path.exists(path):
      with open(path, encoding="utf-8", errors="replace") as f:
        log_tail += f.read()[-200_000:] + "\n"
  text = err + "\n" + log_tail
  peak = re.search(r"Peak RAM: ([\d.]+)/", err)
  peak_s = f", peak host RAM {float(peak.group(1)):.1f} GB" if peak else ""
  m = re.search(r"TIMEOUT: exceeded PHASE_TIMEOUT_S=(\d+)s", text)
  if m:
    return f"timeout at {m.group(1)} s{peak_s}"
  if "holds the whole converted state" in text:
    return "converted state exceeds host RAM"
  if re.search(r"needs all \d+ shards .* staged", text):
    return "source must be staged in RAM next to the converted state; exceeds host RAM"
  if "RESOURCE_EXHAUSTED" in text:
    return "TPU HBM OOM"
  if "KeyError" in text:
    return "KeyError (no parameter mapping)"
  if "OOMKilled" in err:
    # A multi-slice phase also reports -9 on slice 0 when a remote slice dies, so only single-slice kills are OOMs.
    return f"killed (-9) on slice 0{peak_s}" if "Multi-slice" in err else f"host OOM-killed{peak_s}"
  m = re.search(r"exit code (-?\d+)", err)
  return f"failed (exit {m.group(1)})" if m else "failed"


class Logs:
  """Merged view of several logs mirrors: for each file, the last root that has it wins (pass re-runs last)."""

  def __init__(self, roots: list[str]):
    self.roots = roots

  def models(self) -> list[str]:
    names = {n for r in self.roots for n in os.listdir(r) if os.path.isdir(os.path.join(r, n))}
    return sorted(names, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else len(MODEL_ORDER), n))

  def configs(self, model: str) -> list[str]:
    return sorted(
        {c for r in self.roots if os.path.isdir(os.path.join(r, model)) for c in os.listdir(os.path.join(r, model))}
    )

  def path(self, model: str, cfg: str, phase: str) -> str:
    """Path of `<phase>_result.json`; its `<phase>.log` is in the same directory."""
    for root in reversed(self.roots):
      p = os.path.join(root, model, cfg, f"{phase}_result.json")
      if os.path.exists(p):
        return p
    return os.path.join(self.roots[-1], model, cfg, f"{phase}_result.json")

  def load(self, model: str, cfg: str, phase: str) -> tuple[dict | None, str]:
    p = self.path(model, cfg, phase)
    return _load(p), os.path.join(os.path.dirname(p), f"{phase}.log")


def cold_s1(logs: Logs, model: str, override: float | None) -> float:
  """Cold HF->GCS seconds for the model: the override, else the `safetensors_dynamic` cold download, else the
  `to_maxtext` one (when only that pipeline downloaded). Times under 10 s are warm cache checks (the smallest cold
  download, qwen3-0.6b, takes 16 s).
  """
  if override is not None:
    return override
  times = {"safetensors_dynamic": [], "to_maxtext_step1": []}
  for cfg in logs.configs(model):
    for phase, vals in times.items():
      rec, _ = logs.load(model, cfg, phase)
      if rec:
        vals.append(float(rec.get("hf_to_gcs_s", 0.0) or 0.0))
  st_best = max(times["safetensors_dynamic"], default=0.0)
  if st_best >= 10.0:
    return st_best
  best = max(times["safetensors_dynamic"] + times["to_maxtext_step1"], default=0.0)
  return best if best >= 10.0 else 0.0


def build_rows(roots: list[str], overrides: dict[str, float]) -> list[dict]:
  logs = Logs(roots)
  rows = []
  for model in logs.models():
    s1 = cold_s1(logs, model, overrides.get(model))
    for scan in ("scanned", "unscanned"):
      tm1, tm1_log = logs.load(model, scan, "to_maxtext_step1")
      tm2, tm2_log = logs.load(model, scan, "to_maxtext_step2")
      for n in (1, 2, 4):
        cfg = scan if n == 1 else f"{scan}_slices{n}"
        st, st_log = logs.load(model, cfg, "safetensors_dynamic")
        tm3, tm3_log = logs.load(model, cfg, "to_maxtext_step3")
        if st is None and tm3 is None and (n > 1 or tm2 is None):
          continue
        tm_ok = _ok(tm1) and _ok(tm2) and _ok(tm3)
        # Staging the source for the conversion happens in Step 1 or, on a resumed run, at the start of Step 2, so a
        # Step 1 failure is reported as a conversion (S2) failure.
        if tm_ok:
          tm_status = "OK"
        elif tm1 is not None and not _ok(tm1):
          tm_status = "S2: " + failure_reason(tm1, tm1_log)
        elif tm2 is not None and not _ok(tm2):
          tm_status = "S2: " + failure_reason(tm2, tm2_log)
        elif tm2 is None:
          tm_status = "not run"
        elif tm3 is None:
          tm_status = "S3: not run"
        else:
          tm_status = "S3: " + failure_reason(tm3, tm3_log)
        st_status = "OK" if _ok(st) else failure_reason(st, st_log)
        g = lambda rec, k: float((rec or {}).get(k, 0.0) or 0.0) if _ok(rec) else 0.0  # noqa: E731
        tm_s1 = g(tm1, "hf_to_gcs_s") or s1
        tm_total = tm_s1 + g(tm2, "convert_to_orbax_s") + g(tm3, "load_orbax_s") if tm_ok else 0.0
        st_total = s1 + g(st, "safetensors_and_convert_s") if _ok(st) else 0.0
        rows.append(
            {
                "model": model,
                "scan_mode": scan,
                "num_slices": n,
                "global_devices": int((st or tm3 or {}).get("global_devices", 8 * n) or 8 * n),
                "to_maxtext_1_hf_to_gcs_s": tm_s1,
                "to_maxtext_1b_hf_to_local_s": g(tm1, "hf_to_local_s"),
                "to_maxtext_2_convert_to_orbax_s": g(tm2, "convert_to_orbax_s"),
                "to_maxtext_2a_init_abstract_model_s": g(tm2, "init_abstract_model_s"),
                "to_maxtext_2b_transform_and_shard_s": g(tm2, "transform_and_shard_s"),
                "to_maxtext_2c_orbax_save_to_gcs_s": g(tm2, "orbax_save_to_gcs_s"),
                "to_maxtext_3_load_orbax_s": g(tm3, "load_orbax_s"),
                "to_maxtext_total_s": tm_total,
                "safetensor_1_hf_to_gcs_s": s1,
                "safetensor_1_worker_upload_s": g(st, "worker_upload_max_s"),
                "safetensor_1_warm_cache_s": g(st, "hf_to_gcs_s") if g(st, "hf_to_gcs_s") < 10.0 else 0.0,
                "safetensor_2_safetensors_and_convert_s": g(st, "safetensors_and_convert_s"),
                "safetensor_2a_derive_mappings_s": g(st, "derive_mappings_s"),
                "safetensor_2b_load_sharded_hf_s": g(st, "load_sharded_hf_s"),
                "safetensor_2c_transform_hf_to_mt_s": g(st, "transform_hf_to_mt_s"),
                "safetensor_total_s": st_total,
                "speedup_total_vs_to_maxtext_x": tm_total / st_total if tm_total and st_total else 0.0,
                "load_time_diff_safetensor_step2_vs_orbax_step3_s": (
                    g(st, "safetensors_and_convert_s") - g(tm3, "load_orbax_s") if tm_ok and _ok(st) else 0.0
                ),
                "to_maxtext_status": tm_status,
                "safetensor_status": st_status,
            }
        )
  return rows


def _fmt(x: float) -> str:
  return f"{x:,.2f}"


def markdown(rows: list[dict]) -> str:
  out = [
      "| Model | Scan Mode | Slices (Devices) | Pipeline | 1. `HF -> GCS` (s) | 2. `Convert to Orbax` (s) | "
      "3. `Load into MaxText` (s) | **Total (No Step 1) (s)** | **Total (End-to-End) (s)** | **Speedup (E2E / No Step 1)** |",
      "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
  ]
  for r in rows:
    dev = f"`{r['num_slices']} ({r['global_devices']})`"
    tm_ok, st_ok = r["to_maxtext_status"] == "OK", r["safetensor_status"] == "OK"
    tm_no1 = r["to_maxtext_2_convert_to_orbax_s"] + r["to_maxtext_3_load_orbax_s"]
    st_no1 = r["safetensor_2_safetensors_and_convert_s"]
    if tm_ok:
      tm_cells = [
          _fmt(r["to_maxtext_2_convert_to_orbax_s"]),
          _fmt(r["to_maxtext_3_load_orbax_s"]),
          _fmt(tm_no1),
          _fmt(r["to_maxtext_total_s"]),
          "1.00x (baseline)",
      ]
    elif r["to_maxtext_status"].startswith("S3: "):
      reason = r["to_maxtext_status"][4:]
      tm_cells = [_fmt(r["to_maxtext_2_convert_to_orbax_s"]), f"**{reason}**", "N/A", "N/A", "N/A"]
    else:
      reason = r["to_maxtext_status"].removeprefix("S2: ")
      tm_cells = [f"**{reason}**", "—", "N/A", "N/A", "N/A"]
    if st_ok and tm_ok:
      speed = f"**{r['to_maxtext_total_s'] / r['safetensor_total_s']:.2f}x / {tm_no1 / st_no1:.2f}x**"
    elif st_ok:
      speed = "**Only `safetensor` loads this config**"
    else:
      speed = "—"
    st_cells = [
        "**—**",
        f"**{_fmt(st_no1)}**" if st_ok else f"**{r['safetensor_status']}**",
        f"**{_fmt(st_no1)}**" if st_ok else "—",
        f"**{_fmt(r['safetensor_total_s'])}**" if st_ok else "—",
        speed,
    ]
    out.append(
        f"| **`{r['model']}`** | `{r['scan_mode']}` | {dev} | `to_maxtext` | {_fmt(r['to_maxtext_1_hf_to_gcs_s'])} | "
        + " | ".join(tm_cells)
        + " |"
    )
    out.append(
        f"| ↳ | `{r['scan_mode']}` | {dev} | **`safetensor`** | **{_fmt(r['safetensor_1_hf_to_gcs_s'])}** | "
        + " | ".join(st_cells)
        + " |"
    )
  return "\n".join(out) + "\n"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("roots", nargs="+", help="Local mirrors of <run dir>/logs (later roots override earlier ones)")
  p.add_argument("--cold-s1", action="append", default=[], help="MODEL=SECONDS cold HF->GCS override")
  p.add_argument("--csv", required=True, help="Output CSV path")
  p.add_argument("--md", required=True, help="Output markdown table path")
  p.add_argument(
      "--md-into",
      action="append",
      default=[],
      help="Markdown file whose text between `<!-- TABLE START -->` and `<!-- TABLE END -->` is replaced by the table",
  )
  args = p.parse_args()
  overrides = {k: float(v) for k, v in (o.split("=", 1) for o in args.cold_s1)}
  rows = build_rows(args.roots, overrides)
  with open(args.csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()
    for r in rows:
      w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
  table = markdown(rows)
  with open(args.md, "w", encoding="utf-8") as f:
    f.write(table)
  for path in args.md_into:
    with open(path, encoding="utf-8") as f:
      text = f.read()
    start, end = "<!-- TABLE START -->\n", "<!-- TABLE END -->"
    i, j = text.index(start) + len(start), text.index(end)
    with open(path, "w", encoding="utf-8") as f:
      f.write(text[:i] + table + text[j:])
  print(f"Wrote {len(rows)} configurations to {args.csv} and {args.md}")


if __name__ == "__main__":
  main()
