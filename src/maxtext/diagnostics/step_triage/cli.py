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

"""Command-line interface for offline TPU ICI-SDC and sflag straggler triage.

Usage:
  python3 -m maxtext.diagnostics.step_triage.cli \\
    --logs <files or dirs> \\
    [--topology 4x4x8 --cores_per_chip 2 --pipeline_depth 13] \\
    [--json out.json]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys

from maxtext.diagnostics.step_triage import analysis
from maxtext.diagnostics.step_triage import events
from maxtext.diagnostics.step_triage import report


def build_arg_parser() -> argparse.ArgumentParser:
  """Constructs the argument parser for the `step_triage` CLI."""
  parser = argparse.ArgumentParser(
      description=(
          "Offline TPU straggler attribution for ICI-SDC transfer timeout waves "
          "(Approach 1) and passive collective sflag wait timeouts (Approach 2)."
      )
  )
  parser.add_argument(
      "paths",
      nargs="*",
      help="Log file(s) or directory(ies) to analyze.",
  )
  parser.add_argument(
      "--logs",
      nargs="+",
      default=None,
      help="Log file(s) or directory(ies) to analyze.",
  )
  parser.add_argument(
      "--topology",
      type=str,
      default=None,
      help="Optional chip grid dimensions WxH or WxHxZ (default: inferred from logs).",
  )
  parser.add_argument(
      "--cores_per_chip",
      "--cores-per-chip",
      dest="cores_per_chip",
      type=int,
      default=None,
      help="Optional TensorCores per chip (default: inferred from _TC0/_TC1).",
  )
  parser.add_argument(
      "--pipeline_depth",
      "--pipeline-depth",
      dest="pipeline_depth",
      type=int,
      default=None,
      help="Override ICI-SDC pipeline depth D (default: from IciSdcTestEmitter or 13).",
  )
  parser.add_argument(
      "--wrap",
      choices=("auto", "true", "false"),
      default="auto",
      help="Torus wrap mode for neighbor calculation (default: auto).",
  )
  parser.add_argument(
      "--job_name",
      "--job-name",
      dest="job_name",
      type=str,
      default="tpu_workload",
      help="Job name recorded in the straggler_report JSON output.",
  )
  parser.add_argument(
      "--json",
      nargs="?",
      const="-",
      default=None,
      help=("Write structured straggler_report JSON to the given file path, " "or to stdout if '-' or omitted."),
  )
  parser.add_argument(
      "--max_waves",
      "--max-waves",
      dest="max_waves",
      type=int,
      default=20,
      help="Maximum waves to display in human-readable summary (default: 20).",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint returning 0 on success."""
  parser = build_arg_parser()
  args = parser.parse_args(argv)

  target_paths: list[str] = []
  if args.logs:
    target_paths.extend(args.logs)
  if args.paths:
    target_paths.extend(args.paths)

  if not target_paths or target_paths == ["-"]:
    parsed = events.parse_lines(sys.stdin, file="<stdin>")
  else:
    parsed = events.parse_paths(target_paths)

  topo_override = analysis.parse_topology_str(args.topology)
  analysis_res = analysis.analyze_parsed_logs(
      parsed=parsed,
      pipeline_depth=args.pipeline_depth,
      topology_override=topo_override,
      cores_per_chip_override=args.cores_per_chip,
      wrap_mode=args.wrap,
  )
  rep = report.build_straggler_report(
      analysis=analysis_res,
      job_name=args.job_name,
  )

  if args.json and args.json != "-":
    with open(args.json, "w", encoding="utf-8") as f:
      json.dump(rep, f, indent=2)

  if args.json == "-":
    print(json.dumps(rep, indent=2))
  else:
    sys.stdout.write(report.format_human_report(rep, max_waves=args.max_waves))

  return 0


if __name__ == "__main__":
  sys.exit(main())
