#!/usr/bin/env python3
# Copyright 2026 Google LLC
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

"""Summarizes aot_e2e.sh logs: per kernel (fwd_bwd, accumulate, update) the CompiledMemoryStats and cost analysis,
and the tool's TRAIN-KERNEL DEVICE PEAK against an HBM cap.

  python3 aot_summarize.py <log> [<log> ...] [--cap-mib 84992]
"""
import argparse
import ast
import re


def summarize(path, cap_gib):
  with open(path, errors="replace") as f:
    s = f.read()
  print("==", path)
  costs = [ast.literal_eval(m) for m in re.findall(r"Cost analysis: (\{.*\})", s)]
  mems = re.findall(r"Memory analysis: CompiledMemoryStats\(([^)]*)\)", s)
  for name, c, m in zip(("fwd_bwd", "accumulate", "update"), costs, mems):
    d = dict(kv.split("=") for kv in m.split(", "))
    print(
        f"  {name:10s} flops={c.get('flops', 0):.6e} bytes={c.get('bytes accessed', 0):.6e} "
        f"temp={int(d['temp_size_in_bytes']):,} arg={int(d['argument_size_in_bytes']):,} "
        f"out={int(d['output_size_in_bytes']):,} code={int(d['generated_code_size_in_bytes']):,} "
        f"host_temp={int(d['host_temp_size_in_bytes']):,}"
    )
  peak = re.search(r"TRAIN-KERNEL DEVICE PEAK: ([0-9.]+) GiB", s)
  if peak:
    p = float(peak.group(1))
    print(f"  PEAK {p:.2f} GiB vs cap {cap_gib:.2f} GiB -> headroom {cap_gib - p:+.2f} GiB")
  err = re.findall(r"^(\w*Error[^\n]*|.*RESOURCE_EXHAUSTED[^\n]*)", s, re.M)
  if err and not peak:
    print("  FIRST ERROR:", err[0][:400])
  if "Finished training_engine" not in s and not err:
    print("  (running or incomplete)")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("logs", nargs="+")
  ap.add_argument("--cap-mib", type=float, default=84992, help="the trainer's --xla_tpu_max_hbm_size_mib")
  a = ap.parse_args()
  for path in a.logs:
    summarize(path, a.cap_mib / 1024)


if __name__ == "__main__":
  main()
