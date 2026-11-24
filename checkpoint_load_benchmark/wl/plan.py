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

"""The benchmark v3 matrix: the planned rows and the tasks that produce them.

The matrix is `configs/models_v2.json`. The candidate build runs all of it with both pipelines. The baseline build
runs `safetensors_dynamic` on BASELINE_SUBSET only: the models that the two fixes under test are expected to change,
plus three controls.

A task is one harness invocation in pods of its own: one model, one placement (platform, topology, slices), one
repetition, one build, one pipeline, both scan modes. A unit is what a lane of the driver runs without interruption:
one task, or the baseline and the candidate task of the same configuration back to back, in alternating order.

Usage:
  plan.py rows                      # writes reports/v3/planned_rows.csv
  plan.py tasks --phase smoke|sd|tm [--platforms v5p ...] [--models m ...] --out tasks.json
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wl import common  # pylint: disable=wrong-import-position
from bench import manifests_v2  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

# Why each model is in the baseline subset.
BASELINE_SUBSET = {
    "qwen3-14b": "hidden size not a power of two",
    "qwen3-32b": "hidden size not a power of two",
    "gpt-oss-20b": "hidden size not a power of two",
    "gpt-oss-120b": "hidden size not a power of two",
    "qwen3-30b-a3b": "fine-grained MoE on several hosts or slices",
    "qwen3-235b-a22b": "fine-grained MoE on several hosts or slices",
    "qwen3-480b-a35b": "fine-grained MoE on several hosts or slices",
    "deepseek3-671b": "scanned stacked MoE on many chips",
    "qwen3.5-397b-a17b": "scanned stacked MoE on many chips",
    "deepseek4-1600b": "scanned stacked MoE on many chips",
    "qwen3-0.6b": "control",
    "llama3.1-8b": "control",
    "llama3.1-70b": "control",
}
PIPELINES = {"sd": "safetensors_dynamic", "tm": "to_maxtext"}
# The (pipeline, build) combinations that a configuration is run with; baseline only for the subset.
ROW_VARIANTS = (("sd", "candidate"), ("sd", "baseline"), ("tm", "candidate"))
SMOKE_MODEL = "qwen3-0.6b"
# Baseline `safetensors_dynamic` on 64 v5e hosts reads these with tf.io.gfile until the host runs out of memory.
_V5E_BASELINE_OOM = ("qwen3-30b-a3b", "qwen3-235b-a22b", "qwen3-480b-a35b")
ROW_FIELDS = (
    "platform",
    "model",
    "tier",
    "topology",
    "slices",
    "hosts",
    "chips",
    "scan_mode",
    "pipeline",
    "build",
    "repetitions",
    "fix_group",
    "expected_outcome",
)


def _models() -> list[dict]:
  return schema_v2.load_models_v2()["models"]


def _reps(model: dict) -> int:
  return int(schema_v2.load_models_v2()["repetitions_by_tier"].get(model["tier"], 1))


def expected_outcome(model: dict, platform: str, topology: str, slices: int, scan: str, pipe: str, build: str) -> str:
  """Returns what the PR tests and the v2 benchmark lead us to expect of a row: a hint for reading results."""
  name = model["model_name"]
  profile = platforms.get_profile(platform)
  if pipe == "tm":
    ram_gb = profile.host_ram_gib_by_topology[topology] * 1.074
    if 2 * float(model["loaded_bf16_gb"]) > 0.9 * ram_gb:
      return "OOM-HOST or NOT RUN: the conversion does not fit in one host's RAM"
    return "OK"
  if name == "deepseek4-1600b" and platform == "v5e" and scan == "scanned":
    return "OOM-HBM: stacking needs more than 16 GiB per chip (open issue)"
  if build == "baseline" and platform == "v5e" and name in _V5E_BASELINE_OOM:
    return "OOM-HOST: the bug that the GCS read fix removes"
  del slices
  return "OK"


def _placements(model: dict, platform: str) -> list[tuple[str, int]]:
  return [(str(p["topology"]), int(p["slices"])) for p in model["topologies_by_platform"].get(platform, [])]


def planned_rows() -> list[dict]:
  """Returns one row per configuration, scan mode, pipeline and build that v3 runs."""
  rows = []
  configurations = [(m, p, t, s) for m in _models() for p in common.PLATFORMS for t, s in _placements(m, p)]
  variants = [(scan, pipe, build) for scan in ("scanned", "unscanned") for pipe, build in ROW_VARIANTS]
  for model, platform, topology, slices in configurations:
    name = model["model_name"]
    profile = platforms.get_profile(platform)
    hosts = profile.hosts_per_slice(topology) * slices
    for scan, pipe, build in variants:
      if build == "baseline" and name not in BASELINE_SUBSET:
        continue
      rows.append(
          {
              "platform": platform,
              "model": name,
              "tier": model["tier"],
              "topology": topology,
              "slices": slices,
              "hosts": hosts,
              "chips": hosts * profile.chips_per_host(topology),
              "scan_mode": scan,
              "pipeline": PIPELINES[pipe],
              "build": build,
              "repetitions": _reps(model),
              "fix_group": BASELINE_SUBSET.get(name, ""),
              "expected_outcome": expected_outcome(model, platform, topology, slices, scan, pipe, build),
          }
      )
  return rows


def write_planned_rows(path: str) -> int:
  rows = planned_rows()
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  return len(rows)


def build_task(platform: str, model: str, topology: str, slices: int, rep: int, build: str, pipe: str) -> dict:
  """Returns the queue task of one harness invocation, with every path under this build's part of wl-bench-v3."""
  root = common.regional_root(platform)
  prefix = f"wl3-{build}-{pipe}"
  planned = manifests_v2.build_planned_queue_tasks(
      platform=platform,
      experiment_name=common.EXPERIMENT,
      run_prefix=prefix,
      models=[model],
      verify_checksums_tiers=("S", "A", "B", "C"),
      skip_to_maxtext=pipe == "sd",
  )
  matches = [t for t in planned if t["topology"] == topology and t["slices"] == slices and t["repetition"] == rep]
  if len(matches) != 1:
    raise ValueError(f"no single planned task for {platform} {model} {slices}x{topology} r{rep}")
  task = matches[0]
  args, env = task["args"], task["env"]
  task["id"] = f"{task['id']}_{pipe}"
  task["csv"] = f"{task['id']}.csv"
  task["build"], task["pipe"] = build, pipe
  if pipe == "tm":
    args.append("--skip-safetensors")
  i = args.index("--gcs-base")
  args[i + 1] = args[i + 1].replace(f"{root}/runs/", f"{root}/runs-{build}/", 1)
  args[args.index("--local-work-dir") + 1] = f"/ramdisk/bench_{task['id']}"
  env["BENCH_FULL_CHECKSUMS"] = "1"
  # Rules that the task must satisfy; the builders above already do.
  run_id = args[args.index("--run-id") + 1]
  if not (run_id.startswith(prefix) and env["BENCH_RUN_ID"] == run_id):
    raise ValueError(f"unexpected run id in {task['id']}: {run_id}")
  if not args[i + 1].startswith(f"{root}/runs-{build}/"):
    raise ValueError(f"unexpected run root in {task['id']}: {args[i + 1]}")
  if not env["BENCH_SAFETENSORS_GCS_DIR"].startswith(f"{root}/hf_cache"):
    raise ValueError(f"weights of {task['id']} are not in {root}: {env['BENCH_SAFETENSORS_GCS_DIR']}")
  for forbidden in ("STAGE_SPARSE_DEEPSEEK_OOM", "KEEP_JAX_CACHE", "USE_LOCAL_HF_FOR_SAFETENSORS"):
    if forbidden in env:
      raise ValueError(f"{forbidden} must not be set in {task['id']}")
  if f"/{common.V2_APP}" in json.dumps(task):
    raise ValueError(f"{task['id']} references a {common.V2_APP} path")
  task["run_dir"] = f"{args[i + 1]}/run_{run_id}"
  task["queue_root"] = f"{root}/queue/{build}/{task['id']}"
  return task


def _unit_groups(phase: str, name: str, pair_index: int) -> list[list[tuple[str, str]]]:
  """Returns the (build, pipeline) task groups of one configuration; a pair alternates its order with `pair_index`."""
  if phase == "tm":
    return [[("candidate", "tm")]]
  if phase == "sd" and name not in BASELINE_SUBSET:
    return [[("candidate", "sd")]]
  pair = [("baseline", "sd"), ("candidate", "sd")]
  groups = [pair if pair_index % 2 == 0 else pair[::-1]]
  if phase == "smoke":
    groups.append([("baseline", "tm"), ("candidate", "tm")])
  return groups


def build_units(phase: str, want_platforms=None, want_models=None) -> list[dict]:
  """Returns the units of `phase` in running order: per platform, by tier (S, A, C, B) and then by model size.

  smoke: the smallest model on the smallest placement of every platform, and on a second slice where the matrix has
  one, with both builds and both pipelines. sd: `safetensors_dynamic`, candidate everywhere and baseline on the
  subset. tm: `to_maxtext`, candidate only.
  """
  tier_order = {"S": 0, "A": 1, "C": 2, "B": 3}
  models = sorted(_models(), key=lambda m: (tier_order[m["tier"]], float(m["loaded_bf16_gb"])))
  units = []
  for platform in want_platforms or common.PLATFORMS:
    profile = platforms.get_profile(platform)
    wanted = [
        m
        for m in models
        if (not want_models or m["model_name"] in want_models) and (phase != "smoke" or m["model_name"] == SMOKE_MODEL)
    ]
    runs = [
        (m, topology, slices, rep)
        for m in wanted
        for topology, slices in _placements(m, platform)
        for rep in range(1, (1 if phase == "smoke" else _reps(m)) + 1)
    ]
    pair_index = 0
    for model, topology, slices, rep in runs:
      name = model["model_name"]
      groups = _unit_groups(phase, name, pair_index)
      pair_index += len(groups[0]) == 2
      for group in groups:
        units.append(
            {
                "platform": platform,
                "model": name,
                "tier": model["tier"],
                "topology": topology,
                "slices": slices,
                "hosts_per_slice": profile.hosts_per_slice(topology),
                "tasks": [build_task(platform, name, topology, slices, rep, b, p) for b, p in group],
            }
        )
  return units


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)
  rows = sub.add_parser("rows")
  rows.add_argument("--out", default=os.path.join(common.REPORT_DIR, "planned_rows.csv"))
  tasks = sub.add_parser("tasks")
  tasks.add_argument("--phase", required=True, choices=["smoke", "sd", "tm"])
  tasks.add_argument("--platforms", nargs="+", choices=common.PLATFORMS)
  tasks.add_argument("--models", nargs="+")
  tasks.add_argument("--out", required=True)
  args = parser.parse_args()
  common.use_v3_profiles()
  if args.cmd == "rows":
    print(f"{args.out}: {write_planned_rows(args.out)} rows")
    return
  units = build_units(args.phase, args.platforms, args.models)
  with open(args.out, "w", encoding="utf-8") as f:
    json.dump({"phase": args.phase, "units": units}, f, indent=1)
  print(f"{args.out}: {len(units)} units, {sum(len(u['tasks']) for u in units)} tasks")


if __name__ == "__main__":
  main()
