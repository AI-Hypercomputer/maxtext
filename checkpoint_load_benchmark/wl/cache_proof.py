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

"""Proves that the cache evidence of a phase result tells a cold load from a warm one.

Runs inside one benchmark pod (it is the runner script of the proof JobSet) and loads one small model with
`safetensors_dynamic` three times in that pod:
  run 1: the first load in the pod;
  run 2: the same load again under the normal rules, so with a new empty cache directory: it must take as long as
         run 1 and report no cache hit;
  run 3: the same load with the cache directory of run 2 kept (KEEP_JAX_CACHE=1): it must report cache hits, be
         flagged `cache_valid: false`, and be faster.
It prints one line per run and writes the verdict to <gcs-base>/cache_proof_result.json. Runs 2 and 3 are flagged
invalid for sharing the pod with an earlier task, which is expected here and reported separately from the cache hits.
"""

import argparse
import json
import os
import subprocess
import sys

from etils import epath

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_ckpt_load_time.py")


def run_load(args, n: int, extra_env: dict[str, str]) -> dict:
  """Runs the scanned `safetensors_dynamic` load once, as run `n`, and returns its phase result."""
  run_id = f"{args.run_prefix}-r{n}"
  work_dir = f"/ramdisk/cache_proof_r{n}"
  cmd = [sys.executable, "-u", HARNESS, "--models", args.model, "--scan-modes", "scanned", "--num-slices-list", "1"]
  cmd += ["--num-hosts-per-slice", "1", "--devices-per-slice", str(args.devices_per_slice), "--hardware", "tpu"]
  cmd += ["--platform", args.platform, "--slice-topology", args.topology, "--gcs-base", args.gcs_base]
  cmd += ["--run-id", run_id, "--local-work-dir", work_dir, "--allow-step-failure", "--skip-to-maxtext"]
  cmd += ["--csv-report", f"/tmp/cache_proof_r{n}.csv"]
  env = dict(os.environ, BENCH_RUN_ID=run_id, BENCH_SAFETENSORS_GCS_DIR=args.weights, KEEP_HF_CACHE="1")
  env.update(extra_env)
  subprocess.run(cmd, env=env, check=True)
  with open(f"{work_dir}/logs/{args.model}/scanned/safetensors_dynamic_result.json", "r", encoding="utf-8") as f:
    return json.load(f)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--platform", required=True)
  parser.add_argument("--topology", required=True)
  parser.add_argument("--devices-per-slice", type=int, required=True)
  parser.add_argument("--model", default="qwen3-0.6b")
  parser.add_argument("--weights", required=True, help="gs:// directory with the model's .safetensors files")
  parser.add_argument("--gcs-base", required=True)
  parser.add_argument("--run-prefix", default="wl3-cache-proof")
  args = parser.parse_args()

  runs = [run_load(args, 1, {}), run_load(args, 2, {})]
  runs.append(run_load(args, 3, {"KEEP_JAX_CACHE": "1", "BENCH_JAX_CACHE_DIR": runs[1]["cache_jax_dir"]}))
  rows = []
  for n, rec in enumerate(runs, 1):
    rows.append(
        {
            "run": n,
            "status": rec.get("run_status"),
            "total_s": rec.get("safetensors_dynamic_step2_total_seconds"),
            "xla_compile_s": rec.get("safetensors_dynamic_step2_xla_compile_seconds"),
            "cache_dir": rec.get("cache_jax_dir"),
            "files_at_start": rec.get("cache_jax_files_at_start"),
            "files_at_end": rec.get("cache_jax_files_at_end"),
            "hits": rec.get("cache_jax_hits"),
            "misses": rec.get("cache_jax_misses"),
            "cache_valid": rec.get("cache_valid"),
            "cache_invalid_reason": rec.get("cache_invalid_reason"),
        }
    )
    print(f"CACHE_PROOF {json.dumps(rows[-1], sort_keys=True)}", flush=True)
  r1, r2, r3 = rows[0], rows[1], rows[2]
  checks = {
      "all runs OK": all(r["status"] == "OK" for r in rows),
      "run 1 is valid": r1["cache_valid"] is True,
      "runs 1 and 2 start with an empty cache directory and have no hits": all(
          r["files_at_start"] == 0 and r["hits"] == 0 for r in (r1, r2)
      ),
      "runs 1 and 2 leave files in their own cache directories": all(r["files_at_end"] > 0 for r in (r1, r2))
      and r1["cache_dir"] != r2["cache_dir"],
      "run 2 is only flagged for the shared pod": r2["cache_invalid_reason"] == "pod had run 1 other tasks",
      "run 2 takes as long as run 1 (within 25%)": abs(r2["total_s"] - r1["total_s"]) <= 0.25 * r1["total_s"],
      "run 3 has cache hits and is flagged for them": r3["hits"] > 0 and "cache hits" in r3["cache_invalid_reason"],
      "run 3 is faster than run 2": r3["total_s"] < r2["total_s"],
  }
  verdict = {"passed": all(checks.values()), "checks": checks, "runs": rows}
  print(f"CACHE_PROOF_VERDICT {json.dumps(verdict, sort_keys=True)}", flush=True)
  epath.Path(f"{args.gcs_base.rstrip('/')}/cache_proof_result.json").write_text(json.dumps(verdict, indent=1))
  return 0 if verdict["passed"] else 1


if __name__ == "__main__":
  sys.exit(main())
