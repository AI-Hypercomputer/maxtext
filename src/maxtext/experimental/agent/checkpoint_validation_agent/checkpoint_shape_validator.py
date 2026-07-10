"""Validates structural consistency between a MaxText blueprint and an Orbax checkpoint."""

import sys


def load_shapes(filepath):
  """Parses a file to extract key-shape pairs."""
  shapes = {}
  with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
      if "key:" in line and "|" in line:
        parts = line.split("|")
        shapes[parts[0].replace("key:", "").strip()] = parts[1].replace("shape:", "").strip()
  return shapes


ideal = load_shapes("/tmp/ideal_shapes.txt")
actual = load_shapes("/tmp/actual_shapes.txt")

all_keys = sorted(set(ideal.keys()) | set(actual.keys()))
has_mismatch = False

for k in all_keys:
  exp = ideal.get(k, "MISSING")
  got = actual.get(k, "MISSING")
  if exp == got:
    print(f"MATCH: {k} | Expected: {exp} -> Got: {got}")
  else:
    print(f"MISMATCH: {k} | Expected: {exp} -> Got: {got}")
    has_mismatch = True

if has_mismatch:
  print("\nERROR: Structural mismatches found!")
  sys.exit(1)
print("\nSUCCESS: All parameters match perfectly.")
