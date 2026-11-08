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

"""Layerwise verification entry point with fast AOT preflight gating (< 2s).

Usage:
  # Fast preflight check across a sweep:
  python3 -m tests.utils.deepseek_v41_layerwise.verify_layerwise --preflight-only \
      --lw-runs="L1:production:M4:forward L2:code_parity:M4:0,1,2,3" --aot-dir=/path/to/aot

  # Execute a single gate with automatic preflight:
  python3 -m tests.utils.deepseek_v41_layerwise.verify_layerwise \
      --stage=L4 --precision-tier=production --sample=M4 --target=20 \
      --aot-json=/path/to/aot.json --output-json=/path/to/out.json ...
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
src_path = str(REPO_ROOT / "src")
if src_path in sys.path:
  sys.path.remove(src_path)
sys.path.insert(0, src_path)
root_str = str(REPO_ROOT)
if root_str in sys.path:
  sys.path.remove(root_str)
sys.path.insert(0, root_str)
for mod in list(sys.modules.keys()):
  if mod == "maxtext" or mod.startswith("maxtext."):
    del sys.modules[mod]

from absl import app, flags

from tests.utils.deepseek_v41_layerwise import lw_config, lw_runs, lw_stages


def flags_parser(argv: list[str]) -> argparse.Namespace:
  """Parses CLI flags with argparse and marks absl flags parsed."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      "--lw-runs",
      "--lw_runs",
      default="",
      help="Space-separated run specs '<stage>:<tier>:<sample>:<target>'.",
  )
  parser.add_argument("--stage", choices=lw_runs.STAGES, default="", help="Stage name.")
  parser.add_argument(
      "--precision-tier",
      "--precision_tier",
      choices=lw_config.PRECISION_TIERS,
      default="",
      help="Precision tier ('code_parity' or 'production').",
  )
  parser.add_argument("--sample", default="", help="Pinned sample tag, e.g. M4, L2x, S2048.")
  parser.add_argument("--target", default="", help="Stage target, e.g. forward, vjp, 0,1,2,3, 20.")
  parser.add_argument("--aot-dir", "--aot_dir", default="", help="Directory holding sample AOT reports.")
  parser.add_argument("--aot-json", "--aot_json", default="", help="Path to single sample AOT report JSON.")
  parser.add_argument("--bundle-dir", "--bundle_dir", default="", help="Local directory with reference bundles.")
  parser.add_argument("--hf-ckpt-dir", "--hf_ckpt_dir", default="", help="HF checkpoint directory.")
  parser.add_argument("--tokenizer-path", "--tokenizer_path", default="", help="Tokenizer directory.")
  parser.add_argument("--output-json", "--output_json", default="", help="Output JSON report path.")
  parser.add_argument("--output-dir", "--output_dir", default="", help="Output directory for sweep JSON reports.")
  parser.add_argument("--num-devices", "--num_devices", type=int, default=4, help="Number of devices (default 4).")
  parser.add_argument(
      "--preflight-only",
      "--preflight_only",
      action="store_true",
      help="Run only the fast AOT preflight validation (< 2s) and exit.",
  )
  parser.add_argument(
      "--perturb-mode",
      "--perturb_mode",
      choices=["none", "scale_gate_1ulp_bf16_l9", "scale_gate_1ulp_bf16", "swap_experts_l9"],
      default="none",
      help="Perturbation mode for testing.",
  )
  parser.add_argument(
      "--run-defect2-multi-block-conditioning-gate",
      "--run_defect2_multi_block_conditioning_gate",
      action="store_true",
      help="Run multi-block RoPE inv_freq conditioning check.",
  )

  args = parser.parse_args(argv[1:])
  flags.FLAGS.mark_as_parsed()
  return args


def _populate_args_from_spec(args: argparse.Namespace, spec: lw_runs.RunSpec) -> None:
  """Populates stage, precision_tier, sample, target from a single run spec if unset."""
  if not getattr(args, "stage", None):
    args.stage = spec.stage
  if not getattr(args, "precision_tier", None):
    args.precision_tier = spec.tier
  if not getattr(args, "sample", None):
    args.sample = spec.sample
  if not getattr(args, "target", None):
    args.target = spec.target


def _parse_plan_file(plan_path: pathlib.Path) -> list[lw_runs.RunSpec]:
  """Parses a TSV plan file into a list of RunSpecs."""
  specs = []
  for line in plan_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    parts = line.split("\t")
    if len(parts) >= 5:
      specs.append(lw_runs.RunSpec(parts[1], parts[2], parts[3], parts[4]))
    else:
      specs.append(lw_runs.parse_run(line))
  return specs


def parse_runs(args: argparse.Namespace) -> list[lw_runs.RunSpec]:
  """Extracts list of RunSpec objects from either --lw-runs or single stage flags."""
  if args.lw_runs.strip():
    raw = args.lw_runs.strip()
    if "\n" not in raw and len(raw) < 4096:
      try:
        plan_path = pathlib.Path(raw)
        if plan_path.is_file():
          specs = _parse_plan_file(plan_path)
          if len(specs) == 1:
            _populate_args_from_spec(args, specs[0])
          return specs
      except OSError:
        pass
    specs = [lw_runs.parse_run(tok) for tok in raw.split()]
    if len(specs) == 1:
      _populate_args_from_spec(args, specs[0])
    return specs
  if args.stage and args.precision_tier and args.sample and args.target:
    return [lw_runs.RunSpec(args.stage, args.precision_tier, args.sample, args.target)]
  raise ValueError("Must provide either --lw-runs or all of (--stage, --precision-tier, --sample, --target).")


def run_preflight_checks(runs: Sequence[lw_runs.RunSpec], args: argparse.Namespace) -> None:
  """Fast preflight check (< 2s) verifying require_aot upfront for every stage in the sweep."""
  aot_cache: dict[str, dict[str, Any]] = {}
  for r in runs:
    if args.aot_json:
      aot_path = pathlib.Path(args.aot_json)
    elif args.aot_dir:
      aot_path = pathlib.Path(args.aot_dir) / f"lw_aot_v5p_8_{r.sample}.json"
    else:
      raise ValueError("Either --aot-dir or --aot-json must be provided for preflight verification.")

    if not aot_path.is_file():
      raise ValueError(f"Required AOT report file does not exist: {aot_path}")

    path_str = str(aot_path)
    if path_str not in aot_cache:
      aot_cache[path_str] = json.loads(aot_path.read_text(encoding="utf-8"))

    lw_stages.preflight_aot(
        stage=r.stage,
        tier=r.tier,
        sample=r.sample,
        target=r.target,
        aot_json_data=aot_cache[path_str],
        aot_json_path=path_str,
        num_devices=args.num_devices,
    )


def main(args: argparse.Namespace) -> None:
  t0 = time.time()
  runs = parse_runs(args)

  # Mandatory fast preflight check at the very top of main before any JAX init or model load:
  run_preflight_checks(runs, args)
  preflight_elapsed = time.time() - t0
  print(f"[PREFLIGHT] All {len(runs)} stage AOT records verified successfully in {preflight_elapsed:.3f}s")

  if args.preflight_only:
    return

  # If not preflight-only, delegate single gate execution to lw_runs:
  if len(runs) == 1:
    if not getattr(args, "output_json", None):
      raise ValueError("Must provide --output-json for single gate execution.")
    _populate_args_from_spec(args, runs[0])
    if not getattr(args, "aot_json", None) and getattr(args, "aot_dir", None):
      args.aot_json = str(pathlib.Path(args.aot_dir) / f"lw_aot_v5p_8_{args.sample}.json")
    lw_runs.run_gate(args)
  else:
    raise NotImplementedError(
        "For multi-run execution sweeps, use lw_runs plan and run runner script. "
        "verify_layerwise supports single-gate execution or --preflight-only for sweeps."
    )


if __name__ == "__main__":
  app.run(main, flags_parser=flags_parser)
