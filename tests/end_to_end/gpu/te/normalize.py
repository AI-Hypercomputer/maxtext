# Copyright 2023–2025 Google LLC
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
""" Normalize the raw results to get the percentage difference from the baseline"""

# Usage: python normalize.py input_raw_results.csv output_summary.{csv|txt} format
#   format = 'csv' for comma-separated, 'txt' or 'tsv' for tab-separated

import csv
import sys

if len(sys.argv) < 4:
  print("Usage: normalize.py input_raw_results.csv output_summary.{csv|txt} format")
  print("  format = 'csv' for comma-separated, 'txt' or 'tsv' for tab-separated")
  sys.exit(1)

input_csv = sys.argv[1]
output_file = sys.argv[2]
format_type = sys.argv[3].lower()

data = {}
key_order = []  # preserve order of keys

# Read input TSV
with open(input_csv, encoding="utf-8") as f:
  reader = csv.DictReader(f, delimiter="\t")
  for row in reader:
    key = tuple(row.get(k) if row.get(k) not in [None, ""] else "NA" for k in ["dp", "tpsp", "fsdp"])
    if not row.get("test"):
      continue
    if key not in data:
      data[key] = {}
      key_order.append(key)  # remember when first seen
    data[key][row["test"]] = row

header = ["test", "dp", "tpsp", "fsdp", "mean", "stddev", "normalized"]
rows = []

# iterate keys in first-seen order
for key in key_order:
  rowset = data[key]
  baseline = rowset.get("fp8", {})
  base_mean = baseline.get("mean", "NA")
  try:
    base_mean_val = float(base_mean)
    has_baseline = True
  except ValueError:
    base_mean_val = 1.0  # dummy value for pylint
    has_baseline = False

    # iterate tests in first-seen order
  for testname in rowset:
    row = rowset[testname]
    mean = row["mean"]
    stddev = row["stddev"]
    if mean == "NA":
      normalized = "-"
    elif testname == "fp8":
      testname = "maxtext_fp8"
      normalized = "0.00%" if has_baseline else "-"
    elif has_baseline and mean != "NA":
      try:
        normalized = f"{(float(mean) / base_mean_val - 1) * 100:.2f}%"
      except ValueError:
        normalized = "-"
    else:
      normalized = "-"
    rows.append(
        [
            testname,
            row["dp"],
            row["tpsp"],
            row["fsdp"],
            mean,
            stddev,
            normalized,
        ]
    )

if format_type in ("csv",):
  with open(output_file, "a", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(header)
    writer.writerows(rows)
elif format_type in ("txt", "tsv"):
  with open(output_file, "a", encoding="utf-8") as out:
    out.write("\t".join(header) + "\n")
    for r in rows:
      out.write("\t".join(r) + "\n")
else:
  print("Invalid format type! Use 'csv' or 'txt'/'tsv'.")
  sys.exit(2)

print(f"Done. Wrote summary to {output_file} as {format_type}.")
