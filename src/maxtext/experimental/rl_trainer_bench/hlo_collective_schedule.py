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

"""Static schedule summary of the collectives in a scheduled (after_optimizations) XLA TPU HLO text dump.

  python3 hlo_collective_schedule.py <module.after_optimizations.txt> [--min-mib 1] [--detail]

For every async-start whose wrapped op is a collective (all-gather, reduce-scatter, all-reduce, all-to-all,
collective-permute; recognised by the start's name, e.g. all-gather.2222.cloned.1.call-start), it finds the matching
async-done in the same computation and counts what the schedule places between them: instructions, and "compute"
instructions (fusion / convolution / custom-call / dot). The instruction order in a scheduled dump is the TensorCore
program order, so "compute between start and done" is the static overlap the latency-hiding scheduler granted. It
also counts synchronous collectives (all-gather(...) etc. issued directly in a scheduled computation, not wrapped).
Sizes are from the done's result shape (per device). This is a static view: it cannot see runtime latency,
SparseCore queueing behind other SC work, or ICI contention.
"""
import argparse
import collections
import re
import statistics
import sys

DT = {
    "bf16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "f16": 2,
    "s8": 1,
    "u8": 1,
    "pred": 1,
    "f8e4m3fn": 1,
    "s16": 2,
    "u16": 2,
    "f64": 8,
    "s64": 8,
    "u64": 8,
    "f8e5m2": 1,
}
KINDS = ("all-gather", "reduce-scatter", "all-reduce", "all-to-all", "collective-permute")
COMPUTE = re.compile(r"\b(fusion|convolution|custom-call|dot)\(")
SHAPE = re.compile(r"\b([a-z0-9]+)\[([0-9,]*)\]")


def nbytes(shape_text):
  """Returns the byte size of an HLO shape string (tuples summed)."""
  tot = 0
  for dt, dims in SHAPE.findall(shape_text):
    if dt not in DT:
      continue
    n = 1
    for d in dims.split(","):
      if d:
        n *= int(d)
    tot += n * DT[dt]
  return tot


def kind_of(name):
  base = name.lstrip("%")
  for k in KINDS:
    if base.startswith(k):
      return k
  return None


def parse(path):
  """Splits an HLO text dump into {computation name: [(instruction name, right-hand side)]}."""
  comps = {}
  cur = None
  with open(path, encoding="utf-8") as f:
    lines = f.readlines()
  for line in lines:
    if cur is None:
      m = re.match(r"^(ENTRY )?%([\w.\-]+) .*\{\s*$", line)  # headers start at column 0, instructions are indented
      if m:
        cur = m.group(2)
        comps[cur] = []
      continue
    if line.rstrip() == "}":  # multi-line backend configs also start lines with "}}"
      cur = None
      continue
    m = re.match(r"^\s*(ROOT )?%([\w.\-]+) = (.*)$", line)
    if m:
      comps[cur].append((m.group(2), m.group(3)))
  return comps


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("path")
  ap.add_argument("--min-mib", type=float, default=0.0)
  ap.add_argument("--detail", action="store_true")
  a = ap.parse_args()
  comps = parse(a.path)
  rows = []
  sync = collections.Counter()
  sync_bytes = collections.Counter()
  for cname, instrs in comps.items():
    # Only scheduled computations that hold async-starts or wrapped collectives matter; computations that are
    # the async bodies themselves (async_computation.*, wrapped_*) are skipped as places.
    for i, (n, rhs) in enumerate(instrs):
      native = re.search(r"\) (all-gather|reduce-scatter|all-reduce|all-to-all|collective-permute)-start\(", rhs)
      if " async-start(" in rhs or native:
        k = native.group(1) if native else kind_of(n)
        if not k:
          continue
        thread = re.search(r'async_execution_thread="(\w+)"', rhs)
        thread = thread.group(1) if thread else "tensorcore"
        done_op = f" {k}-done(" if native else " async-done("
        done = None
        for j in range(i + 1, len(instrs)):
          if done_op in instrs[j][1] and f"%{n})" in instrs[j][1]:
            done = j
            break
        if done is None:
          continue
        between = instrs[i + 1 : done]
        ncomp = sum(1 for _, r in between if COMPUTE.search(r.split(" metadata=")[0]))
        size = nbytes(instrs[done][1].split(done_op)[0])
        rows.append((k, thread, "entry" if cname.startswith("main.") else "loop", size, done - i, ncomp, cname, n))
      else:
        m = re.search(r"[\]\}\)] (all-gather|reduce-scatter|all-reduce|all-to-all|collective-permute)\(", rhs)
        if m and any(" async-start(" in r for _, r in instrs):  # only scheduled computations, not reducers
          sync[(m.group(1), "entry" if cname.startswith("main.") else "loop")] += 1
          sync_bytes[(m.group(1), "entry" if cname.startswith("main.") else "loop")] += nbytes(rhs.split(m.group(0))[0])
  rows = [r for r in rows if r[3] >= a.min_mib * 2**20]
  agg = collections.defaultdict(list)
  for r in rows:
    agg[(r[0], r[1], r[2])].append(r)
  print(
      f"{'kind':16s} {'thread':10s} {'where':5s} {'n':>5s} {'MiB med':>8s} {'MiB tot':>9s} {'instr med':>9s} "
      f"{'compute med':>11s} "
      f"{'compute=0':>9s}"
  )
  for (k, t, w), rs in sorted(agg.items()):
    print(
        f"{k:16s} {t:10s} {w:5s} {len(rs):5d} {statistics.median(r[3] for r in rs) / 2**20:8.2f} "
        f"{sum(r[3] for r in rs) / 2**20:9.1f} {statistics.median(r[4] for r in rs):9.0f} "
        f"{statistics.median(r[5] for r in rs):11.0f} {sum(1 for r in rs if r[5] == 0):9d}"
    )
  print("synchronous collectives (no start/done; the TensorCore waits on them) in scheduled computations:")
  for key in sorted(sync):
    print(f"  {key[0]:16s} {key[1]:5s} n={sync[key]:4d} MiB tot={sync_bytes[key] / 2**20:9.1f}")
  if a.detail:
    for r in sorted(rows, key=lambda r: (r[0], r[6], r[7])):
      print(f"  {r[0]:15s} {r[1]:10s} {r[3] / 2**20:8.2f} MiB instr={r[4]:5d} compute={r[5]:4d} {r[6][:40]} {r[7]}")


if __name__ == "__main__":
  sys.exit(main())
