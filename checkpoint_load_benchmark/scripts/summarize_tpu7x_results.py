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

"""Prints a markdown status matrix from the per-phase result JSONs of a tpu7x benchmark run.

The benchmark harness writes one `<phase>_result.json` per model and configuration under
`<gcs-base>/run_<run-id>/logs/<model>/<config>/`, where `<config>` is `scanned`, `unscanned`,
`scanned_slices2` or `unscanned_slices2`. This script reads a local mirror of that `logs/` tree,
so it also shows partial progress before the harness writes its final CSV.

Usage:
  # 1. Mirror the result JSONs (the per-phase .log files are large, so skip them):
  gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/p2_lance_prs/run_tpu7x_16m/logs \
      /tmp/tpu7x_results/p2 --exclude='.*\\.log$'
  # 2. Print the matrix:
  python3 checkpoint_load_benchmark/scripts/summarize_tpu7x_results.py /tmp/tpu7x_results/p2

Columns: `TM` = to_maxtext (S1 HF->GCS, S2 CPU conversion to Orbax, S3 Orbax load into TPU HBM);
`ST` = safetensors_dynamic Step 2 (sharded safetensors load + in-memory conversion). to_maxtext
S1/S2 run once per scan mode on 1 slice and are shared by the 2-slice S3 load.

Cell legend: a time = completed; `⏳` = no result yet; `❌ killed (-9)` = the worker got SIGKILL,
which is either a host OOM kill or the PHASE_TIMEOUT_S watchdog (the harness labels both
"OOMKilled", so check the log for `[TIMEOUT_WATCHDOG]`); `❌ HBM OOM` = RESOURCE_EXHAUSTED on TPU;
`❌ exit N` = any other failure.
"""

import argparse
import json
import os

PREFERRED_ORDER = [
    "qwen3-0.6b",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "llama3.1-8b",
    "llama3.1-70b",
    "gemma2-2b",
    "gemma3-4b",
    "deepseek2-16b",
    "gpt-oss-20b",
    "qwen3-30b-a3b",
    "mixtral-8x7b",
    "gpt-oss-120b",
    "qwen3-235b-a22b",
    "deepseek4-284b",
    "qwen3.5-397b-a17b",
    "qwen3-480b-a35b",
    "deepseek3-671b",
]
CONFIGS = [
    ("scanned", "scanned", 1),
    ("unscanned", "unscanned", 1),
    ("scanned_slices2", "scanned", 2),
    ("unscanned_slices2", "unscanned", 2),
    ("scanned_slices4", "scanned", 4),  # only deepseek3-671b (needs 4 slices); rows without results are skipped
    ("unscanned_slices4", "unscanned", 4),
]


def _load(root, model, cfg, phase):
  path = os.path.join(root, model, cfg, f"{phase}_result.json")
  if not os.path.exists(path):
    return None
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def _cell(result, key):
  if result is None:
    return "⏳"
  if result.get("status") == "FAILED":
    err = str(result.get("error", ""))
    if "RESOURCE_EXHAUSTED" in err or "pjrt.OOM" in err:
      return "❌ HBM OOM"
    if "exit code -9" in err or "exit code 137" in err:
      return "❌ killed (-9)"
    code = err.split("exit code ", 1)[1].split()[0] if "exit code " in err else "?"
    return f"❌ exit {code}"
  value = result.get(key)
  return f"{value:.1f}s" if isinstance(value, (int, float)) else "?"


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("root", help="Local mirror of <gcs-base>/run_<run-id>/logs")
  parser.add_argument("--models", nargs="*", help="Models to show (default: every model directory found)")
  args = parser.parse_args()

  found = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
  models = args.models or [m for m in PREFERRED_ORDER if m in found] + [m for m in found if m not in PREFERRED_ORDER]

  print(
      "| Model | Config | TM S1 HF→GCS | TM S2 convert | TM S3 load Orbax | ST S2 load+convert | ST peak host RAM (GB) |"
  )
  print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
  for model in models:
    for cfg, scan, slices in CONFIGS:
      if slices > 2 and not os.path.isdir(os.path.join(args.root, model, cfg)):
        continue
      s3 = _load(args.root, model, cfg, "to_maxtext_step3")
      st = _load(args.root, model, cfg, "safetensors_dynamic")
      if slices == 1:
        s1 = _cell(_load(args.root, model, scan, "to_maxtext_step1"), "hf_to_gcs_s")
        s2 = _cell(_load(args.root, model, scan, "to_maxtext_step2"), "convert_to_orbax_s")
      else:
        s1 = s2 = "(shared)"
      ram = f"{st['peak_host_ram_gb']:.1f}" if st and st.get("status") != "FAILED" and "peak_host_ram_gb" in st else "-"
      label = f"{scan}, {slices} slice{'s' if slices > 1 else ''}"
      print(
          f"| `{model}` | {label} | {s1} | {s2} | {_cell(s3, 'load_orbax_s')} | "
          f"{_cell(st, 'safetensors_and_convert_s')} | {ram} |"
      )


if __name__ == "__main__":
  main()
