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

"""Diffs the resolved MaxText "Config param" dumps of two logs (e.g. e2e trainer vs harness client).

  python3 diff_config.py <e2e_cfg_or_log> <bench_log>
Lines are matched on "Config param <key>: <value>" anywhere in the line; prints only differing keys.
"""
import re
import sys


def load(path):
  out = {}
  for line in open(path, errors="replace"):
    m = re.search(r"Config param ([A-Za-z0-9_]+): (.*)$", line.rstrip("\n"))
    if m:
      out[m.group(1)] = m.group(2).strip()
  return out


def main():
  a, b = load(sys.argv[1]), load(sys.argv[2])
  print(f"{len(a)} keys in {sys.argv[1]}, {len(b)} keys in {sys.argv[2]}")
  for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
      print(f"{k}\n  e2e:   {a.get(k)}\n  bench: {b.get(k)}")


if __name__ == "__main__":
  main()
