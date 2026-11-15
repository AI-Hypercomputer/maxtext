# Copyright 2023-2026 Google LLC
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

"""Pairwise gradient comparison from one-step SGD twins (see grad_twin/run.sh and PORT_NOTES.md).

Each twin ran pre_train/train.py for one step of plain SGD at learning rate LR from the same initial params p0, so its
step-1 params are p1 = p0 - LR * g, and g = (p0 - p1) / LR. With LR = 1e8 the update dwarfs p0, so the fp32 rounding of
p1 (about 2**-24 * |p1| / |p0 - p1| relative, the "floor" column) sits far below the differences being measured.

Usage: python compare.py P0_ITEMS P1A_ITEMS P1B_ITEMS [--lr 1e8] [--top 10] [--json out.json]
  P0_ITEMS: the initial params (e.g. <p0 run>/checkpoints/0/items, a learning_rate=0 step);
  P1A/P1B_ITEMS: <twin run>/checkpoints/0/items (MaxText labels the checkpoint after the first update step 0).
Prints the whole-model relative difference |g_a - g_b| / |g_a| and the worst tensors.
"""

import argparse
import json

import numpy as np
import orbax.checkpoint as ocp


def load_params(path):
  """Flat {path: np.ndarray} of the "params" subtree of a MaxText training or params-only checkpoint."""
  tree = ocp.PyTreeCheckpointer().restore(path)
  node = tree["params"] if "params" in tree else tree
  if isinstance(node, dict) and "params" in node:
    node = node["params"]
  flat = {}

  def walk(prefix, x):
    if isinstance(x, dict):
      for k, v in x.items():
        walk(f"{prefix}/{k}" if prefix else str(k), v)
    else:
      flat[prefix] = np.asarray(x, dtype=np.float64)

  walk("", node)
  return flat


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("p0")
  ap.add_argument("p1a")
  ap.add_argument("p1b")
  ap.add_argument("--lr", type=float, default=1e8)
  ap.add_argument("--top", type=int, default=10)
  ap.add_argument("--json")
  args = ap.parse_args()
  p0, pa, pb = load_params(args.p0), load_params(args.p1a), load_params(args.p1b)
  assert set(p0) == set(pa) == set(pb), "param trees differ"
  rows, num, den, floor_num = [], 0.0, 0.0, 0.0
  for k in sorted(p0):
    ga, gb = (p0[k] - pa[k]) / args.lr, (p0[k] - pb[k]) / args.lr
    d, n = np.linalg.norm(ga - gb), np.linalg.norm(ga)
    step = np.linalg.norm(p0[k] - pa[k])
    floor = 2.0**-24 * np.linalg.norm(pa[k]) / step if step > 0 else float("inf")
    rows.append({"tensor": k, "rel": float(d / n) if n > 0 else float("nan"), "abs": float(d), "norm": float(n),
                 "floor": float(floor)})
    num, den = num + d * d, den + n * n
    floor_num += (2.0**-24 * np.linalg.norm(pa[k]) / args.lr) ** 2
  whole = float(np.sqrt(num / den)) if den > 0 else float("nan")
  whole_floor = float(np.sqrt(floor_num / den)) if den > 0 else float("nan")
  print(f"GRAD-TWIN tensors={len(rows)} |g_a|={np.sqrt(den):.6e} rel(whole model)={whole:.3e} floor={whole_floor:.1e}")
  worst = sorted((r for r in rows if np.isfinite(r["rel"])), key=lambda r: -r["rel"])[: args.top]
  for r in worst:
    print(f"  {r['rel']:.3e}  (floor {r['floor']:.1e}, |g|={r['norm']:.3e})  {r['tensor']}")
  if args.json:
    with open(args.json, "w") as f:
      json.dump({"whole_rel": whole, "whole_floor": whole_floor, "tensors": rows}, f, indent=1)


if __name__ == "__main__":
  main()
