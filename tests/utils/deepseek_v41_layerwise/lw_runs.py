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

"""Plans and summarizes the v5p-8 layerwise runs (driven by 4_test_deepseek41_mm_layerwise_v5p_8.sh).

  plan:      checks that LW_RUNS covers the pinned required list (layerwise_required_runs.json) and writes the runs in
             execution order: the HBM probe first, then the others in LW_RUNS order, then every run whose v5p-8 AOT
             peak exceeds the pinned threshold (each run is its own process).
  summarize: reads each planned run's exit code and report JSON, prints a status table with the special-status
             counts, and exits nonzero unless every planned run exited 0 with status PASS.

Usage:
  python3 -m tests.utils.deepseek_v41_layerwise.lw_runs plan --lw-runs="L1:code_parity:M4:forward ..." \\
      --aot-dir=<dir> --num-devices=4 --output=<plan.tsv>
  python3 -m tests.utils.deepseek_v41_layerwise.lw_runs summarize --plan=<plan.tsv> --exit-codes=<codes.tsv> \\
      --output-dir=<dir>
"""

import argparse
import dataclasses
import json
import pathlib
import re
import time
import traceback
from collections.abc import Sequence
from typing import Any

import jax
from absl import app, flags

from tests.utils.deepseek_v41_layerwise import lw_bundle, lw_checks, lw_config, lw_stages

REQUIRED_RUNS_PATH = lw_checks.TOLERANCES_PATH.parent / "layerwise_required_runs.json"
STAGES = ("L1", "L2", "L3", "L4")


@dataclasses.dataclass(frozen=True)
class RunSpec:
  """One harness run "<stage>:<tier>:<sample>:<target>"."""

  stage: str
  tier: str
  sample: str
  target: str

  @property
  def text(self) -> str:
    return f"{self.stage}:{self.tier}:{self.sample}:{self.target}"

  @property
  def tag(self) -> str:
    """File-name tag of the run's report and log."""
    return f"{self.stage}_{self.tier}_{self.sample}_{re.sub('[^A-Za-z0-9]', '-', self.target)}"


def parse_run(text: str) -> RunSpec:
  """ "L3:code_parity:M4:moe:3" -> RunSpec (the target keeps any further colons)."""
  parts = text.split(":", 3)
  if len(parts) != 4 or parts[0] not in STAGES or parts[1] not in lw_config.PRECISION_TIERS or not parts[3]:
    raise ValueError(f"Run {text!r} is not <stage {STAGES}>:<tier {lw_config.PRECISION_TIERS}>:<sample>:<target>")
  return RunSpec(*parts)


def load_required(path: pathlib.Path) -> dict[str, Any]:
  """The pinned run list (exact keys; every run parses; no duplicates)."""
  data = lw_checks.require_keys(
      lw_checks.load_pinned_json(path, expected_sha256=None).data,
      (
          "schema_version",
          "probe_first",
          "probe_reason",
          "hbm_last_threshold_gib",
          "hbm_last_reason",
          "dropped",
          "required",
          "step5_chain_required",
      ),
      str(path),
  )
  if data["schema_version"] != 1:
    raise ValueError(f"{path} schema_version {data['schema_version']} != 1")
  required = [parse_run(r).text for r in data["required"]]
  if len(set(required)) != len(required) or data["probe_first"] not in required:
    raise ValueError(f"{path}: required runs repeat, or probe_first {data['probe_first']!r} is not one of them")
  return data


def aot_peak_gib(spec: RunSpec, aot_dir: pathlib.Path, num_devices: int) -> float:
  """Largest v5p-8 AOT peak (temp + IO, GiB/device) over the programs of `spec` (`lw_stages.run_programs`)."""
  path = aot_dir / f"lw_aot_v5p_8_{spec.sample}.json"
  records = lw_checks.require(lw_checks.load_pinned_json(path, expected_sha256=None).data, "programs", str(path))
  peaks = []
  for program in lw_stages.run_programs(spec.stage, spec.target, num_devices):
    found = [r for r in records if "program" in r and r["program"] == program and r["tier"] == spec.tier]
    if len(found) != 1 or not found[0]["compiled"]:
      raise ValueError(f"{path}: {len(found)} compiled records for {program} / {spec.tier} (run {spec.text})")
    peaks.append(float(found[0]["temp_plus_io_gib"]))
  return max(peaks)


def plan(
    lw_runs: str, required_doc: dict[str, Any], aot_dir: pathlib.Path, num_devices: int
) -> list[tuple[RunSpec, float]]:
  """The runs of `lw_runs` in execution order with their AOT peaks; raises unless `lw_runs` covers the required list."""
  specs = [parse_run(r) for r in lw_runs.split()]
  given = [s.text for s in specs]
  if len(set(given)) != len(given):
    raise ValueError(f"LW_RUNS repeats a run: {sorted({g for g in given if given.count(g) > 1})}")
  missing = sorted(set(required_doc["required"]) - set(given))
  if missing:
    raise ValueError(f"LW_RUNS lacks {len(missing)} required runs ({REQUIRED_RUNS_PATH.name}): {missing}")
  peaks = {s.text: aot_peak_gib(s, aot_dir, num_devices) for s in specs}
  threshold = float(required_doc["hbm_last_threshold_gib"])
  probe = [s for s in specs if s.text == required_doc["probe_first"]]
  middle = [s for s in specs if s.text != required_doc["probe_first"] and peaks[s.text] <= threshold]
  last = [s for s in specs if s.text != required_doc["probe_first"] and peaks[s.text] > threshold]
  return [(s, peaks[s.text]) for s in probe + middle + last]


def summarize(planned: Sequence[RunSpec], exit_codes: dict[str, int], output_dir: pathlib.Path) -> tuple[list[str], bool]:
  """Status table lines and whether every planned run exited 0 with status PASS (a missing code / report fails)."""
  lines = [f"{'run':70s} {'exit':>4s} {'status':8s} special statuses / controls not fired"]
  ok = True
  totals: dict[str, int] = {}
  for spec in planned:
    report_path = output_dir / f"{spec.tag}.json"
    code = exit_codes[spec.tag] if spec.tag in exit_codes else None
    if code is None or not report_path.is_file():
      lines.append(
          f"{spec.text:70s} {'-':>4s} {'MISSING':8s} exit code recorded: {code is not None}, report: {report_path.is_file()}"
      )
      ok = False
      continue
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = report["status"]
    special = (
        {k: v for k, v in report["status_counts"].items() if k != lw_checks.STATUS_PASS and v}
        if "status_counts" in report
        else {"error": report["error"]}
    )
    for k, v in special.items():
      if isinstance(v, int):
        totals[k] = totals[k] + v if k in totals else v
    not_fired = len(report["negative_controls_not_fired"]) if "negative_controls_not_fired" in report else "n/a"
    lines.append(f"{spec.text:70s} {code:>4d} {status:8s} {special} / {not_fired}")
    ok = ok and code == 0 and status == lw_checks.STATUS_PASS
  lines.append(f"special-status totals: {totals}; overall: {'PASS' if ok else 'FAIL'}")
  return lines, ok


def run_gate(args: argparse.Namespace) -> None:
  """Executes a single layerwise stage gate run (replaces deleted layerwise_harness.py)."""
  t0 = time.time()
  report: dict[str, Any] = {
      "stage": args.stage,
      "tier": args.precision_tier,
      "sample": args.sample,
      "target": args.target,
      "memory": [],
  }
  out = pathlib.Path(args.output_json)
  out.parent.mkdir(parents=True, exist_ok=True)
  try:
    platform = jax.default_backend()
    report["runtime"] = lw_config.configure_runtime(args.precision_tier, platform=platform)
    report["platform"] = platform
    report["devices"] = [str(d) for d in jax.devices()]
    bundle_dir = pathlib.Path(args.bundle_dir)
    index = lw_bundle.load_index(bundle_dir)
    tolerances = lw_checks.load_tolerances(lw_checks.TOLERANCES_PATH)
    floors = lw_bundle.load_noise_floors(bundle_dir)
    aot = lw_checks.load_pinned_json(pathlib.Path(args.aot_json), expected_sha256=None)
    lw_stages.preflight_aot(
        stage=args.stage,
        tier=args.precision_tier,
        sample=args.sample,
        target=args.target,
        aot_json_data=aot.data,
        aot_json_path=str(aot.path),
        num_devices=len(jax.devices()),
    )
    if getattr(args, "preflight_only", False):
      print(f"[PREFLIGHT] AOT record for {args.stage} {args.precision_tier} {args.sample} {args.target} OK.")
      return
    expected_na = lw_stages.load_expected_not_applicable(lw_stages.EXPECTED_NOT_APPLICABLE_PATH)
    report["inputs_sha256"] = {
        "bundle_index": index.sha256,
        "tolerances": tolerances.sha256,
        "measured_noise_floors": floors.sha256,
        "aot_json": aot.sha256,
        "expected_not_applicable": expected_na.sha256,
        "harness_code": lw_checks.harness_code_sha256(),
        "model_code": lw_checks.model_code_sha256(),
    }
    stage_run = lw_stages.StageRun(
        stage=args.stage,
        tier=args.precision_tier,
        sample=args.sample,
        target=args.target,
        index=index,
        hf_dir=pathlib.Path(args.hf_ckpt_dir),
        tokenizer_path=args.tokenizer_path,
        tolerances=tolerances,
        floors=floors,
        aot=aot,
        expected_na=expected_na,
        memory=report["memory"],
        perturb_mode=args.perturb_mode,
        run_defect2_multi_block_conditioning_gate=args.run_defect2_multi_block_conditioning_gate,
    )
    report.update(lw_stages.finalize_report(lw_stages.run_stage(stage_run), stage_run))
  except Exception as e:  # pylint: disable=broad-exception-caught
    report["status"] = lw_checks.STATUS_FAIL
    report["error"] = f"{type(e).__name__}: {e}"[:8000]
    report["traceback_tail"] = traceback.format_exc()[-8000:]
  report["elapsed_seconds"] = time.time() - t0
  out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
  if report["status"] != lw_checks.STATUS_PASS:
    raise SystemExit(
        f"Layerwise {args.stage} {args.precision_tier} {args.sample} {args.target}: {report['status']} (see {out})"
    )


def flags_parser(argv: list[str]) -> argparse.Namespace:
  """Parses the CLI with argparse and marks absl flags parsed (absl app.run flags_parser contract)."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  p_plan = sub.add_parser("plan")
  p_plan.add_argument("--lw-runs", required=True)
  p_plan.add_argument("--aot-dir", required=True)
  p_plan.add_argument("--num-devices", type=int, required=True)
  p_plan.add_argument("--output", required=True)
  p_sum = sub.add_parser("summarize")
  p_sum.add_argument("--plan", required=True)
  p_sum.add_argument("--exit-codes", required=True)
  p_sum.add_argument("--output-dir", required=True)
  p_run = sub.add_parser("run")
  p_run.add_argument("--stage", choices=STAGES, required=True)
  p_run.add_argument("--precision-tier", choices=lw_config.PRECISION_TIERS, required=True)
  p_run.add_argument("--sample", required=True, help="Pinned input tag, e.g. M4 or S2048.")
  p_run.add_argument(
      "--target", required=True, help='L1: "forward" | "vjp"; L2: "0,1,2,3"; L3: "2|3;4|5" or "moe:3"; L4: "20".'
  )
  p_run.add_argument("--bundle-dir", required=True)
  p_run.add_argument("--hf-ckpt-dir", required=True)
  p_run.add_argument("--tokenizer-path", required=True)
  p_run.add_argument("--aot-json", required=True)
  p_run.add_argument("--output-json", required=True)
  p_run.add_argument(
      "--preflight-only",
      "--preflight_only",
      action="store_true",
      help="Run only the fast AOT preflight validation (< 2s) and exit.",
  )
  p_run.add_argument(
      "--perturb-mode",
      "--perturb_mode",
      choices=["none", "scale_gate_1ulp_bf16_l9", "scale_gate_1ulp_bf16", "swap_experts_l9"],
      default="none",
      help="Perturbation mode for Defect 1 testing.",
  )
  p_run.add_argument(
      "--run-defect2-multi-block-conditioning-gate",
      "--run_defect2_multi_block_conditioning_gate",
      action="store_true",
      help="Run multi-block RoPE inv_freq conditioning check matching Appendix B.3 Defect 2.",
  )
  args = parser.parse_args(argv[1:])
  flags.FLAGS.mark_as_parsed()
  return args


def main(args: argparse.Namespace) -> None:
  if args.command == "run":
    run_gate(args)
    return
  if args.command == "plan":
    ordered = plan(args.lw_runs, load_required(REQUIRED_RUNS_PATH), pathlib.Path(args.aot_dir), args.num_devices)
    rows = [f"{s.tag}\t{s.stage}\t{s.tier}\t{s.sample}\t{s.target}\t{peak:.1f}" for s, peak in ordered]
    pathlib.Path(args.output).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    return
  planned = [RunSpec(*line.split("\t")[1:5]) for line in pathlib.Path(args.plan).read_text(encoding="utf-8").splitlines()]
  exit_codes = {}
  for line in pathlib.Path(args.exit_codes).read_text(encoding="utf-8").splitlines():
    tag, code = line.split("\t")
    exit_codes[tag] = int(code)
  lines, ok = summarize(planned, exit_codes, pathlib.Path(args.output_dir))
  print("\n".join(lines))
  if not ok:
    raise SystemExit(1)


if __name__ == "__main__":
  app.run(main, flags_parser=flags_parser)
