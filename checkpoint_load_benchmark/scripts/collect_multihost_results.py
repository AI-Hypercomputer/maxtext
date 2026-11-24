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

"""Adds multi-host safetensors_dynamic results to reports/tpu7x_multihost_results.csv.

Reads local mirrors of multi-host run dirs (`<run dir>/logs/<model>/<scan>[_slices<N>]_hosts<H>/`, from
`run_tpu7x_cluster.sh --priority multihost`) and appends one row per result JSON that the CSV does not have yet for
the same (model, scan mode, layout, source). Existing rows are never changed, so other owners' rows stay as they are.

Usage:
  gcloud storage rsync -r gs://mesa-maxtext/mesakh_tpu7x_bench/lance_prs/run_tpu7x_multihost/logs /tmp/mh
  python3 checkpoint_load_benchmark/scripts/collect_multihost_results.py /tmp/mh --source mesakh \\
      --csv checkpoint_load_benchmark/reports/tpu7x_multihost_results.csv
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tpu7x_tables import failure_reason  # pylint: disable=wrong-import-position

TOPOLOGY = {2: "2x2x2", 4: "2x2x4", 8: "2x4x4"}  # hosts per tpu7x slice -> topology
HBM_PER_HOST_GB = 813.8  # a tpu7x host: 4 chips, 8 TPU devices
FIELDS = [
    "model",
    "scan_mode",
    "layout",
    "hosts",
    "devices",
    "hbm_gb",
    "link",
    "status",
    "total_s",
    "load_s",
    "transform_s",
    "peak_host_ram_gb",
    "note",
    "source",
]
CFG_RE = re.compile(r"^(scanned|unscanned)(?:_slices(\d+))?_hosts(\d+)$")


def rows_from_mirror(root: str, source: str, note: str) -> list[dict]:
  """CSV rows for the safetensors_dynamic result JSONs of the multi-host configurations in `root`."""
  rows = []
  for model in sorted(os.listdir(root)):
    for cfg in sorted(os.listdir(os.path.join(root, model))):
      m = CFG_RE.match(cfg)
      path = os.path.join(root, model, cfg, "safetensors_dynamic_result.json")
      if not m or not os.path.exists(path):
        continue
      scan, slices, hosts = m.group(1), int(m.group(2) or 1), int(m.group(3))
      with open(path, encoding="utf-8") as f:
        rec = json.load(f)
      ok = rec.get("status") != "FAILED"
      row = {
          "model": model,
          "scan_mode": scan,
          "layout": f"{slices} × {TOPOLOGY[hosts]}",
          "hosts": slices * hosts,
          "devices": slices * hosts * 8,
          "hbm_gb": f"{slices * hosts * HBM_PER_HOST_GB:.1f}",
          "link": "ICI" if slices == 1 else "ICI + DCN",
          "status": "OK" if ok else "FAILED",
          "total_s": f"{rec['safetensors_and_convert_s']:.2f}" if ok else "",
          "load_s": f"{rec['load_sharded_hf_s']:.2f}" if ok else "",
          "transform_s": f"{rec['transform_hf_to_mt_s']:.2f}" if ok else "",
          "peak_host_ram_gb": f"{rec['peak_host_ram_gb']:.2f}" if ok and rec.get("peak_host_ram_gb") else "",
          "note": note
          if ok
          else "; ".join(
              x for x in (failure_reason(rec, os.path.join(root, model, cfg, "safetensors_dynamic.log")), note) if x
          ),
          "source": source,
      }
      rows.append(row)
  return rows


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("roots", nargs="+", help="Local mirrors of <multi-host run dir>/logs")
  p.add_argument("--source", required=True, help="Value of the CSV's source column for the new rows")
  p.add_argument("--note", default="", help="Note for every new row (e.g. the code tree of a fix variant)")
  p.add_argument("--csv", required=True, help="reports/tpu7x_multihost_results.csv (rows are appended)")
  args = p.parse_args()
  with open(args.csv, encoding="utf-8", newline="") as f:
    existing = list(csv.DictReader(f))
  have = {(r["model"], r["scan_mode"], r["layout"], r["source"], r.get("note", "")) for r in existing}
  new = []
  for root in args.roots:
    for row in rows_from_mirror(root, args.source, args.note):
      key = (row["model"], row["scan_mode"], row["layout"], row["source"], row["note"])
      if key not in have:
        have.add(key)
        new.append(row)
  with open(args.csv, "a", encoding="utf-8", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writerows(new)
  for row in new:
    print(f"Added {row['model']} {row['scan_mode']} {row['layout']}: {row['status']} {row['total_s']} {row['note']}")


if __name__ == "__main__":
  main()
