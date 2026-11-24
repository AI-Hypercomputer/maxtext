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

"""Mirrors the benchmark v3 results and turns them into the CSV files of reports/v3/.

  collect.py sync --mirror DIR                 # copy every result JSON from GCS into DIR
  collect.py csv --mirror DIR [--state DIR ...] [--out reports/v3]
  collect.py check --mirror DIR                # print every problem: failed, invalid, wrong code, checksum mismatch

Files written by `csv`:
  runs.csv      one row per phase result, including the results of interrupted attempts, with all scalar fields
  results.csv   one row per configuration and scan mode: both builds of safetensors_dynamic, and to_maxtext
  stages.csv    one row per configuration, build, pipeline and stage
  failures.csv  one row per failed or invalid phase result and per interrupted attempt

A phase result is valid when it is OK, its cache evidence is valid (`cache_valid`), it ran the commit of its build
(reports/v3/builds.json), a `safetensors_dynamic` load read its weights from GCS, its stage times add up to its
total, and its checksums equal those of the other runs of the same model and scan mode. Total seconds are the
median over the valid repetitions.
"""

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wl import common  # pylint: disable=wrong-import-position
from wl import plan  # pylint: disable=wrong-import-position

RUN_ID_RE = re.compile(r"wl3-(baseline|candidate)-(sd|tm)-(v\d\w)-(.+)-(\d+)x([\dx]+)-r(\d+)$")
SD_TOTAL = "safetensors_dynamic_step2_total_seconds"
# The parts of a safetensors_dynamic load that together must make up SD_TOTAL.
SD_PARTS = (
    "safetensors_dynamic_step2_build_hf_to_maxtext_mappings_seconds",
    "safetensors_dynamic_step2_load_total_seconds",
    "safetensors_dynamic_step2_transform_total_seconds",
    "safetensors_dynamic_step2_harness_memory_cleanup_seconds",
    "safetensors_dynamic_step2_final_weight_check_and_sync_seconds",
)
# The harness refuses these conversions before it starts them: the model was not run, it did not run out of memory.
NOT_RUN_PATTERNS = ("holds the whole converted state", "exceeds ~", "shards .* staged")
CONFIG_FIELDS = ("platform", "model", "tier", "topology", "slices", "hosts", "chips", "scan_mode")
# The gpt-oss MLP leaves, whose bits may differ in the sign of zeros between two correct scanned loads.
SIGNED_ZERO_MODEL_PREFIX = "gpt-oss"
SIGNED_ZERO_LEAF_RE = re.compile(r"(^|[.\-_])(wo|wi_0|wi_1)($|[.\-_])")


def sync(mirror: str) -> None:
  """Copies the result JSONs of every region and build, and the Step 1 results, into `mirror`."""
  roots = sorted({common.regional_root(p) for p in common.PLATFORMS})
  for root in roots:
    region = root.split("/")[2].removeprefix("mesa-maxtext-")
    for sub in [f"runs-{b}" for b in common.BUILDS] + ["step1", "step1b"]:
      dst = os.path.join(mirror, region, sub)
      os.makedirs(dst, exist_ok=True)
      cmd = ["gcloud", "storage", "rsync", "-r", f"{root}/{sub}", dst]
      if sub.startswith("runs-"):
        cmd += ["--exclude", r".*(?<!result\.json)$"]
      res = common.run(cmd, check=False, timeout=3600)
      print(f"{root}/{sub}: {'ok' if res.returncode == 0 else 'nothing or failed: ' + res.stderr.strip()[-200:]}")


def _checksum_signatures(rec: dict) -> tuple[str, str, dict]:
  """Returns digests of the exact and of the zero-sign-insensitive bit checksums of `rec`, and the per-leaf values."""
  leaves = rec.get("parameter_checksums") or {}
  exact = {
      k: (v.get("bits_sum"), v.get("bits_pos"), tuple(v.get("shape", ())), v.get("dtype")) for k, v in leaves.items()
  }
  unsigned = {
      k: (v.get("bitsz_sum", v.get("bits_sum")), v.get("bitsz_pos", v.get("bits_pos")), tuple(v.get("shape", ())))
      for k, v in leaves.items()
  }
  if not leaves or any(v[0] is None for v in exact.values()):
    return "", "", exact

  def _digest(d):
    return hashlib.sha1(json.dumps(sorted(d.items()), default=str).encode()).hexdigest()[:16]

  return _digest(exact), _digest(unsigned), exact


def load(mirror: str) -> list[dict]:
  """Returns one record per mirrored phase result that belongs to a v3 task, final results and archived attempts."""
  records = []
  for path in glob.glob(os.path.join(mirror, "*", "runs-*", "*", "run_*", "**", "*_result.json"), recursive=True):
    parts = path[len(mirror) :].strip(os.sep).split(os.sep)
    match = RUN_ID_RE.match(parts[3].removeprefix("run_"))
    if not match or os.path.basename(path) == "cache_proof_result.json":
      continue
    try:
      with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    except ValueError:
      continue
    status = rec.get("run_status") or rec.get("status") or ""
    if status == "SKIPPED" and str(rec.get("error_summary", "")).startswith("--"):
      continue  # The placeholder of a pipeline that this task was told not to run.
    build, pipe, platform, model, slices, topology, rep = match.groups()
    attempt = int(parts[5][1:]) if parts[4] == "attempts" else 0
    phase = rec.get("phase_name") or os.path.basename(path).removesuffix("_result.json")
    exact, unsigned, leaves = _checksum_signatures(rec)
    records.append(
        {
            "platform": platform,
            "model": model,
            "topology": topology,
            "slices": int(slices),
            "scan_mode": rec.get("scan_layers_mode") or parts[-2].split("_")[0],
            "build": build,
            "pipe": pipe,
            "phase": phase,
            "repetition": int(rep),
            "attempt": attempt,
            "status": status,
            "rec": rec,
            "checksum_exact": exact,
            "checksum_unsigned_zero": unsigned,
            "leaves": leaves,
            "path": path,
        }
    )
  return records


def _total_seconds(r: dict):
  rec = r["rec"]
  return rec.get(SD_TOTAL) if r["phase"] == "safetensors_dynamic" else rec.get(f"{r['phase']}_total_seconds")


def short_status(r: dict) -> str:
  """Returns OK, or the short word for how the phase ended."""
  rec = r["rec"]
  if r["status"] == "OK":
    return "OK"
  if r["status"] == "SKIPPED":
    return "NOT RUN"
  category = rec.get("failure_category") or "ERROR"
  text = f"{rec.get('error_summary', '')} {rec.get('error', '')}"
  if (
      category == "OOM-HOST"
      and rec.get("process_signal") != "SIGKILL"
      and any(re.search(p, text) for p in NOT_RUN_PATTERNS)
  ):
    return "NOT RUN"
  return category


def validate(records: list[dict], commits: dict[str, str]) -> None:
  """Sets `valid` and `invalid_reason` on every record, and `checksum_check` on the loads that have checksums."""
  groups = collections.defaultdict(list)
  for r in records:
    if r["status"] == "OK" and r["attempt"] == 0 and r["phase"] in ("safetensors_dynamic", "to_maxtext_step3"):
      groups[(r["model"], r["scan_mode"])].append(r)
  for group in groups.values():
    loads = [r for r in group if r["phase"] == "safetensors_dynamic" and r["checksum_exact"]]
    baseline = [r for r in loads if r["build"] == "baseline"]
    reference = (baseline or loads or [None])[0]
    for r in group:
      if reference is None or not r["checksum_exact"]:
        r["checksum_check"] = "no checksums"
      elif r["checksum_exact"] == reference["checksum_exact"]:
        r["checksum_check"] = "identical"
      else:
        differing = sorted(k for k in r["leaves"] if r["leaves"][k] != reference["leaves"].get(k))
        only_zero_sign = r["checksum_unsigned_zero"] == reference["checksum_unsigned_zero"]
        gpt_oss = r["model"].startswith(SIGNED_ZERO_MODEL_PREFIX)
        if only_zero_sign and gpt_oss and all(SIGNED_ZERO_LEAF_RE.search(k) for k in differing):
          r["checksum_check"] = "identical but for the sign of zeros"
        else:
          r["checksum_check"] = f"MISMATCH in {len(differing)} leaves, e.g. {differing[:3]}"
  for r in records:
    rec, problems = r["rec"], []
    if r["status"] != "OK":
      problems.append(short_status(r))
    else:
      if rec.get("cache_valid") is not True:
        problems.append(f"cache: {rec.get('cache_invalid_reason') or 'no evidence'}")
      commit = (rec.get("code_version") or {}).get("git_commit") or rec.get("git_commit")
      if commits.get(r["build"]) and commit != commits[r["build"]]:
        problems.append(f"ran {str(commit)[:9]}, not the {r['build']} build")
      if r["phase"] == "safetensors_dynamic":
        source = str(rec.get("checkpoint_load_source_uri", ""))
        if (
            not (source.startswith("gs://") and f"/{common.APP}/" in source)
            or rec.get("checkpoint_weights_kind") != "real"
        ):
          problems.append(f"weights not real files from {common.APP}: {source}")
        total = rec.get(SD_TOTAL)
        if isinstance(total, (int, float)):
          unattributed = total - sum(float(rec.get(k) or 0.0) for k in SD_PARTS)
          if abs(unattributed) > max(5.0, 0.02 * total):
            problems.append(f"{unattributed:.1f} s of {total:.1f} s not attributed to a stage")
      if str(r.get("checksum_check", "")).startswith("MISMATCH"):
        problems.append(f"checksums: {r['checksum_check']}")
    r["valid"] = not problems
    r["invalid_reason"] = "; ".join(problems)


def _config_key(r: dict) -> tuple:
  return (r["platform"], r["model"], r["topology"], r["slices"], r["scan_mode"])


def _median_record(rows: list[dict]):
  """Returns the valid final record whose total is the median of the repetitions (the lower one of an even count)."""
  good = sorted((r for r in rows if r["valid"] and r["attempt"] == 0), key=_total_seconds)
  return good[(len(good) - 1) // 2] if good else None


def _summarize(rows: list[dict]) -> dict:
  """Returns status, median total seconds, spread and repetition count of the final results of one cell."""
  final = [r for r in rows if r["attempt"] == 0]
  if not final:
    return {"status": "", "total_s": "", "min_s": "", "max_s": "", "reps": 0, "valid": "", "note": ""}
  good = [r for r in final if r["valid"]]
  totals = [_total_seconds(r) for r in good if isinstance(_total_seconds(r), (int, float))]
  bad = [r for r in final if not r["valid"]]
  status = "OK" if good else short_status(bad[0])
  if not good and bad[0]["status"] == "OK":
    status = "INVALID"
  return {
      "status": status,
      "total_s": round(statistics.median(totals)) if totals else "",
      "min_s": round(min(totals)) if totals else "",
      "max_s": round(max(totals)) if totals else "",
      "reps": len(good),
      "valid": bool(good) and not bad,
      "note": "; ".join(sorted({f"r{r['repetition']}: {r['invalid_reason']}" for r in bad}))[:300],
  }


def build_results(records: list[dict], planned: list[dict]) -> list[dict]:
  """Returns one row per planned configuration and scan mode with the three measured cells side by side."""
  cells = collections.defaultdict(list)
  for r in records:
    cells[(_config_key(r), r["build"], r["phase"])].append(r)
  seen, rows = set(), []
  for p in planned:
    key = (p["platform"], p["model"], p["topology"], int(p["slices"]), p["scan_mode"])
    if key in seen:
      continue
    seen.add(key)
    row = {f: p[f] for f in CONFIG_FIELDS}
    base = _summarize(cells.get((key, "baseline", "safetensors_dynamic"), []))
    cand = _summarize(cells.get((key, "candidate", "safetensors_dynamic"), []))
    steps = {n: _summarize(cells.get((key, "candidate", f"to_maxtext_step{n}"), [])) for n in (1, 2, 3)}
    failed_step = next((steps[n] for n in (1, 2, 3) if steps[n]["status"] not in ("OK", "")), None)
    tm_done = all(steps[n]["status"] == "OK" for n in (2, 3))
    row.update(
        {
            "sd_baseline_status": base["status"],
            "sd_baseline_total_s": base["total_s"],
            "sd_candidate_status": cand["status"],
            "sd_candidate_total_s": cand["total_s"],
            "sd_candidate_min_s": cand["min_s"],
            "sd_candidate_max_s": cand["max_s"],
            "sd_candidate_reps": cand["reps"] or "",
            "tm_status": failed_step["status"] if failed_step else ("OK" if tm_done else steps[3]["status"]),
            "tm_step2_convert_s": steps[2]["total_s"],
            "tm_step3_load_s": steps[3]["total_s"],
            "tm_total_s": steps[2]["total_s"] + steps[3]["total_s"] if tm_done else "",
        }
    )
    both = isinstance(base["total_s"], int) and isinstance(cand["total_s"], int)
    row["sd_change_s"] = cand["total_s"] - base["total_s"] if both else ""
    row["sd_change_pct"] = (
        round(100.0 * (cand["total_s"] - base["total_s"]) / base["total_s"], 1) if both and base["total_s"] else ""
    )
    with_tm = tm_done and isinstance(cand["total_s"], int) and cand["total_s"] > 0
    row["tm_over_sd_speedup"] = round(row["tm_total_s"] / cand["total_s"], 2) if with_tm else ""
    median = _median_record(cells.get((key, "candidate", "safetensors_dynamic"), []))
    row["sd_candidate_peak_ram_gb"] = median["rec"].get("memory_peak_host_ram_container_gb", "") if median else ""
    row["sd_candidate_peak_hbm_gb"] = median["rec"].get("memory_tpu_hbm_peak_per_device_gb", "") if median else ""
    measured = [c for c in (base, cand, *steps.values()) if c["status"]]
    row["valid"] = bool(measured) and all(c["valid"] is True for c in measured)
    row["expected_outcome"] = p["expected_outcome"] if p["expected_outcome"] != "OK" else ""
    row["note"] = "; ".join(c["note"] for c in measured if c["note"])[:500]
    rows.append(row)
  return rows


def build_stages(records: list[dict]) -> list[dict]:
  """Returns the stage times of the median repetition of every cell, one row per stage."""
  cells = collections.defaultdict(list)
  for r in records:
    cells[(_config_key(r), r["build"], r["phase"])].append(r)
  rows = []
  for (key, build, phase), cell in sorted(cells.items()):
    median = _median_record(cell)
    if median is None:
      continue
    total = _total_seconds(median)
    summary = (median["rec"].get("stages_merged") or {}).get("stages_summary", {})
    for stage, values in summary.items():
      if not stage.startswith(phase) or not isinstance(values, dict):
        continue
      rows.append(
          {
              "platform": key[0],
              "model": key[1],
              "topology": key[2],
              "slices": key[3],
              "scan_mode": key[4],
              "build": build,
              "pipeline": plan.PIPELINES[median["pipe"]],
              "phase": phase,
              "stage": stage.removeprefix(f"{phase}_").removesuffix("_seconds"),
              "median_host_s": round(values.get("median_s", 0.0), 2),
              "slowest_host_s": round(values.get("max_s", 0.0), 2),
              "share_of_total_pct": round(100.0 * values.get("max_s", 0.0) / total, 1) if total else "",
          }
      )
  return rows


def build_runs(records: list[dict]) -> tuple[list[dict], list[str]]:
  """Returns one flat row per phase result with every scalar field of the result, and the column order."""
  head = ["platform", "model", "topology", "slices", "scan_mode", "build", "phase", "repetition", "attempt", "status"]
  head += ["valid", "invalid_reason", "total_s", "checksum_exact", "checksum_unsigned_zero", "checksum_check"]
  rows, extra = [], []
  for r in sorted(records, key=lambda r: (_config_key(r), r["build"], r["phase"], r["repetition"], r["attempt"])):
    row = {k: r.get(k, "") for k in head}
    row["status"], row["total_s"] = short_status(r), _total_seconds(r)
    for key, value in r["rec"].items():
      if isinstance(value, (dict, list)) or key in row or key in ("status", "harness_environment_flags"):
        continue
      if key in ("pod_names", "node_names"):
        names = [n for n in str(value).split(",") if n]
        value = f"{len(names)}: {names[0]}" if names else ""
      row[key] = value
      if key not in extra:
        extra.append(key)
    rows.append(row)
  return rows, head + extra


def build_failures(records: list[dict], planned: list[dict], states: list[str]) -> list[dict]:
  """Returns the failed or invalid phase results, and the attempts that the drivers recorded as interrupted."""
  expected = {
      (p["platform"], p["model"], p["topology"], int(p["slices"]), p["scan_mode"], p["build"], p["pipeline"]): p[
          "expected_outcome"
      ]
      for p in planned
  }
  rows = []
  for r in records:
    if r["valid"]:
      continue
    hint = expected.get((*_config_key(r), r["build"], plan.PIPELINES[r["pipe"]]), "")
    word = short_status(r)
    rows.append(
        {
            **{k: r[k] for k in ("platform", "model", "topology", "slices", "scan_mode", "build", "phase", "repetition")},
            "attempt": r["attempt"] or "final",
            "category": word if word != "OK" else "INVALID",
            "cause": (r["invalid_reason"] if word == "OK" else str(r["rec"].get("error_summary", "")))[:300],
            "failed_stage": r["rec"].get("failed_stage_name", ""),
            "expected": "yes" if word != "OK" and any(w in hint for w in (word, "NOT RUN")) and hint != "OK" else "no",
            "action": "kept as an earlier attempt; the task was retried" if r["attempt"] else "",
        }
    )
  for state_dir in states:
    with open(os.path.join(state_dir, "state.json"), "r", encoding="utf-8") as f:
      state = json.load(f)
    for key, task in state["tasks"].items():
      for attempt in task["attempts"]:
        if attempt.get("outcome") == "interrupted":
          build, _, task_id = key.partition("/")
          match = re.match(r"(v\d\w)_(.+)_(\d+)x([\dx]+)_r(\d+)_(sd|tm)$", task_id)
          platform, model, slices, topology, rep, pipe = match.groups() if match else ("", task_id, "", "", "", "")
          rows.append(
              {
                  "platform": platform,
                  "model": model,
                  "topology": topology,
                  "slices": slices,
                  "scan_mode": "both",
                  "build": build,
                  "phase": f"{plan.PIPELINES.get(pipe, pipe)} task",
                  "repetition": rep,
                  "attempt": attempt["n"],
                  "category": "INTERRUPTED",
                  "cause": attempt.get("reason", "")[:300],
                  "expected": "no",
                  "action": "retried" if attempt["n"] < len(task["attempts"]) else "no attempts left",
              }
          )
  return rows


def _write(path: str, rows: list[dict], fields: list[str]) -> None:
  with open(path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  print(f"{path}: {len(rows)} rows")


def _load_validated(mirror: str, out_dir: str) -> tuple[list[dict], list[dict]]:
  """Returns the validated records of `mirror` and the planned rows of `out_dir`."""
  with open(os.path.join(out_dir, "planned_rows.csv"), "r", encoding="utf-8") as f:
    planned = list(csv.DictReader(f))
  commits = {}
  builds_path = os.path.join(out_dir, "builds.json")
  if os.path.exists(builds_path):
    with open(builds_path, "r", encoding="utf-8") as f:
      commits = {name: build["commit"] for name, build in json.load(f)["builds"].items()}
  records = load(mirror)
  validate(records, commits)
  return records, planned


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("cmd", choices=["sync", "csv", "check"])
  parser.add_argument("--mirror", required=True)
  parser.add_argument("--out", default=common.REPORT_DIR)
  parser.add_argument("--state", nargs="*", default=[], help="Driver state directories, for interrupted attempts")
  args = parser.parse_args()
  common.use_v3_profiles()
  if args.cmd == "sync":
    sync(args.mirror)
    return
  records, planned = _load_validated(args.mirror, args.out)
  if args.cmd == "check":
    bad = [r for r in records if not r["valid"]]
    for r in bad:
      where = "/".join(map(str, _config_key(r)))
      print(f"{where} {r['build']} {r['phase']} r{r['repetition']} a{r['attempt']}: {r['invalid_reason']}")
    checks = collections.Counter(r.get("checksum_check", "") for r in records if r.get("checksum_check"))
    print(f"{len(records)} phase results, {len(bad)} failed or invalid; checksums: {dict(checks)}")
    return
  results = build_results(records, planned)
  _write(os.path.join(args.out, "results.csv"), results, list(results[0]))
  stages = build_stages(records)
  _write(os.path.join(args.out, "stages.csv"), stages, list(stages[0]) if stages else ["platform"])
  runs, run_fields = build_runs(records)
  _write(os.path.join(args.out, "runs.csv"), runs, run_fields)
  failure_fields = ["platform", "model", "topology", "slices", "scan_mode", "build", "phase", "repetition", "attempt"]
  failure_fields += ["category", "cause", "failed_stage", "expected", "action"]
  _write(os.path.join(args.out, "failures.csv"), build_failures(records, planned, args.state), failure_fields)


if __name__ == "__main__":
  main()
