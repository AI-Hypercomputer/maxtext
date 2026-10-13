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

"""Structured diff of two render_e2e.py JobSets: every YAML leaf except the client command, plus the client command
split into the env prefix (MAXTEXT_EXTRA_FLAGS tokens, LIBTPU_INIT_ARGS tokens, other env), the trainer argv
(E2E_TRAINER_ARGV), the orchestrator argv (E2E_ORCH_ARGV) and the driver env. Prints only differences.

  python3 structdiff_render.py a.yaml b.yaml
"""
import shlex
import sys

import yaml

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from render_e2e import _proc_main  # pylint: disable=g-import-not-at-top


def leaves(x, path=""):
  if isinstance(x, dict):
    for k, v in x.items():
      yield from leaves(v, f"{path}.{k}")
  elif isinstance(x, list):
    for i, v in enumerate(x):
      name = v.get("name") if isinstance(v, dict) and "name" in v else i
      yield from leaves(v, f"{path}[{name}]")
  else:
    yield path, x


def flagmap(tokens):
  out = {}
  for t in tokens:
    k, _, v = t.partition("=")
    out.setdefault(k, []).append(v)
  return out


def parse(path):
  with open(path) as f:
    js = yaml.safe_load(f)
  _, main_c = _proc_main(js)
  script = main_c["command"][-1]
  main_c["command"] = ["<client script>"]
  line = next(l for l in script.splitlines() if "exec python /tmp/e2e_driver.py" in l)
  toks = shlex.split(line)
  env = {}
  for t in toks:
    if t in ("exec", "python", "/tmp/e2e_driver.py"):
      continue
    k, _, v = t.partition("=")
    env[k] = v
  sec = {
      "leaf": dict(leaves(js)),
      "MAXTEXT_EXTRA_FLAGS": flagmap(env.pop("MAXTEXT_EXTRA_FLAGS", "").split()),
      "LIBTPU_INIT_ARGS": flagmap(env.pop("LIBTPU_INIT_ARGS", "").split()),
      "trainer_argv": flagmap(shlex.split(env.pop("E2E_TRAINER_ARGV", ""))),
      "orch_argv": flagmap(shlex.split(env.pop("E2E_ORCH_ARGV", ""))),
      "env": env,
  }
  return sec


def main_args(path_a, path_b):
  """Prints one line per difference between two renders and returns the number of differences."""
  a, b = parse(path_a), parse(path_b)
  n = 0
  for s in a:
    for k in sorted(set(a[s]) | set(b[s])):
      va, vb = a[s].get(k, "<absent>"), b[s].get(k, "<absent>")
      if va != vb:
        n += 1
        print(f"[{s}] {k}: {str(va)[:200]} -> {str(vb)[:200]}")
  print(f"{n} differences")
  return n


def main():
  main_args(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
  main()
