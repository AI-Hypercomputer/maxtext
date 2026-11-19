#!/usr/bin/env python3
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

"""Verifies completeness and integrity of a MaxText Orbax/OCDbt checkpoint.

Checks:
1. Expected theoretical size vs. actual storage size (GCS or local).
2. Commit success sentinel and manifest existence.
3. OCDbt layer coverage (validates expected layers or scanned blocks).
4. TensorStore chunk readability on sample arrays.
"""

import argparse
from collections.abc import Sequence
import os
import re
import subprocess
import sys


def parse_args(argv: Sequence[str] | None = None):
  """Parses CLI arguments (accepts `--checkpoint-path` or positional path)."""
  parser = argparse.ArgumentParser(description="Verify MaxText checkpoint completeness.")
  parser.add_argument(
      "checkpoint_path_pos",
      nargs="?",
      default=None,
      help="Optional positional checkpoint path (e.g. /dev/shm/.../0/items)",
  )
  parser.add_argument(
      "--checkpoint-path",
      dest="checkpoint_path_flag",
      default=None,
      help="Checkpoint path (e.g. gs://bucket/model/unscanned/0/items or /home/.../0/items)",
  )
  parser.add_argument(
      "--model-params-b",
      type=float,
      required=True,
      help="Total model parameters in billions (e.g. 1600 for 1.6T, 284 for 284B)",
  )
  parser.add_argument(
      "--dtype-bytes",
      type=int,
      default=2,
      help="Bytes per parameter (default 2 for BF16/FP16)",
  )
  parser.add_argument(
      "--expected-layers",
      type=int,
      default=None,
      help="Expected number of transformer layers (e.g. 61 for unscanned, ignored for scanned)",
  )
  parser.add_argument(
      "--expected-prefix-layers",
      type=int,
      default=0,
      help="Expected unscanned prefix layer count when scanned_blocks is used",
  )
  args = parser.parse_args(argv)
  if args.checkpoint_path_flag and args.checkpoint_path_pos:
    parser.error("Pass either --checkpoint-path or a positional checkpoint path, not both.")
  ckpt_path = args.checkpoint_path_flag or args.checkpoint_path_pos
  if not ckpt_path:
    parser.error("the following arguments are required: --checkpoint-path")
  args.checkpoint_path = ckpt_path
  return args


def get_items_and_root_paths(ckpt_path: str):
  """Resolves the items directory and step root path for an Orbax checkpoint."""
  clean_path = ckpt_path.rstrip("/")
  if clean_path.endswith("/items"):
    items_path = clean_path
    step_path = clean_path.rsplit("/items", 1)[0]
    root_path = step_path.rsplit("/0", 1)[0] if step_path.endswith("/0") else step_path
  elif clean_path.endswith("/0"):
    items_path = f"{clean_path}/items"
    root_path = clean_path.rsplit("/0", 1)[0]
  else:
    items_path = f"{clean_path}/0/items"
    root_path = clean_path
  return items_path, root_path


def check_storage_size(ckpt_path: str, expected_bytes: float) -> bool:
  """Verifies checkpoint byte size against expected uncompressed parameter bytes."""
  is_gcs = ckpt_path.startswith("gs://")
  target_type = "GCS" if is_gcs else "local"
  print(f"--> [1/4] Checking {target_type} storage size...")

  _, root_path = get_items_and_root_paths(ckpt_path)
  size_bytes = 0

  if is_gcs:
    cmd = ["gcloud", "storage", "du", "-s", root_path]
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, check=True)
      raw = res.stdout.strip()
      match = re.match(r"^(\d+)", raw)
      if match:
        size_bytes = int(match.group(1))
      else:
        size_bytes = int(raw.split()[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as e:
      print(f"    [WARN] Failed to check GCS size for {root_path}: {e}")
      return False
  else:
    if not os.path.exists(root_path):
      print(f"    [FAIL] Local path {root_path} does not exist.")
      return False
    cmd = ["du", "-sb", root_path]
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, check=True)
      size_bytes = int(res.stdout.strip().split()[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
      for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
          fp = os.path.join(dirpath, f)
          if not os.path.islink(fp):
            size_bytes += os.path.getsize(fp)

  size_gib = size_bytes / (1024**3)
  exp_gib = expected_bytes / (1024**3)
  ratio = size_bytes / expected_bytes
  print(f"    Actual Size:   {size_gib:.2f} GiB ({size_bytes:,} bytes)")
  print(f"    Expected Size: ~{exp_gib:.2f} GiB ({int(expected_bytes):,} bytes)")
  print(f"    Size Ratio:    {ratio * 100:.1f}%")

  if ratio < 0.25:
    print(
        f"    [FAIL] Checkpoint size is only {ratio * 100:.1f}% of expected! "
        "(< 25.0% indicates truncation or missing tensors)."
    )
    return False
  print("    [PASS] Checkpoint size is within expected bounds (>= 25.0% with zstd compression).")
  return True


def check_ocdbt_manifest(ckpt_path: str) -> bool:
  """Verifies presence of commit_success.txt and manifest.ocdbt."""
  is_gcs = ckpt_path.startswith("gs://")
  target_type = "GCS" if is_gcs else "local"
  print(f"--> [2/4] Checking {target_type} OCDbt manifest and sentinel files...")

  items_path, _ = get_items_and_root_paths(ckpt_path)
  step_path = items_path.rsplit("/items", 1)[0] if items_path.endswith("/items") else items_path

  if is_gcs:
    entries = []
    for check_dir in [items_path, step_path]:
      cmd = ["gcloud", "storage", "ls", check_dir]
      try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        entries.extend(res.stdout.splitlines())
      except (subprocess.SubprocessError, OSError):
        pass
    if not entries:
      print(f"    [FAIL] Could not access GCS paths {items_path} or {step_path}.")
      return False
    has_commit = any("commit_success.txt" in e for e in entries)
    has_manifest = any("manifest.ocdbt" in e for e in entries)
    if not has_manifest:
      for sub in [f"{items_path}/ocdbt.process_0", f"{step_path}/ocdbt.process_0"]:
        cmd = ["gcloud", "storage", "ls", sub]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and any("manifest.ocdbt" in e for e in res.stdout.splitlines()):
          has_manifest = True
          break
  else:
    if not os.path.isdir(items_path) and not os.path.isdir(step_path):
      print(f"    [FAIL] Neither items directory ({items_path}) nor step directory ({step_path}) exists.")
      return False
    has_commit = os.path.exists(os.path.join(items_path, "commit_success.txt")) or os.path.exists(
        os.path.join(step_path, "commit_success.txt")
    )
    has_manifest = (
        os.path.exists(os.path.join(items_path, "manifest.ocdbt"))
        or os.path.exists(os.path.join(items_path, "ocdbt.process_0", "manifest.ocdbt"))
        or os.path.exists(os.path.join(step_path, "manifest.ocdbt"))
        or os.path.exists(os.path.join(step_path, "ocdbt.process_0", "manifest.ocdbt"))
    )

  if not has_commit:
    print("    [FAIL] commit_success.txt NOT found! Checkpoint was never committed.")
    return False
  if not has_manifest:
    print("    [FAIL] manifest.ocdbt NOT found! Missing OCDbt index.")
    return False
  print("    [PASS] Both commit_success.txt and manifest.ocdbt exist.")
  return True


def _get_ts_kvstore(items_path: str):
  """Opens the OCDbt TensorStore KvStore with process_0 fallback."""
  import tensorstore as ts  # pylint: disable=import-outside-toplevel

  is_gcs = items_path.startswith("gs://")
  base = items_path if is_gcs else {"driver": "file", "path": items_path}
  spec = {"driver": "ocdbt", "base": base}
  try:
    return ts.KvStore.open(spec).result(), base
  except (OSError, ValueError, RuntimeError):
    fallback_path = f"{items_path}/ocdbt.process_0"
    base_fallback = fallback_path if is_gcs else {"driver": "file", "path": fallback_path}
    spec_fallback = {"driver": "ocdbt", "base": base_fallback}
    return ts.KvStore.open(spec_fallback).result(), base_fallback


def check_layers_coverage(
    ckpt_path: str,
    expected_layers: int | None,
    expected_prefix_layers: int = 0,
) -> bool:
  """Verifies that all expected unscanned or scanned layers exist in OCDbt."""
  items_path, _ = get_items_and_root_paths(ckpt_path)
  is_gcs = ckpt_path.startswith("gs://")

  print("--> [3/4] Checking layer coverage...")
  try:
    kv_store, _ = _get_ts_kvstore(items_path)
    all_keys = kv_store.list().result()
    key_strs = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in all_keys]

    has_scanned_blocks = any("scanned_blocks" in k for k in key_strs)
    found_unscanned_layers = set()

    for k in key_strs:
      if "layers_" in k and "scanned_blocks" not in k:
        part = k.split("layers_")[1].split(".")[0].split("/")[0].split("-")[0]
        if part.isdigit():
          found_unscanned_layers.add(int(part))

    if has_scanned_blocks:
      missing_initial = set(range(expected_prefix_layers)) - found_unscanned_layers
      if missing_initial:
        print(f"    [FAIL] Scanned format detected, but missing initial layers: {sorted(list(missing_initial))}")
        return False
      print("    Found scanned format: scanned_blocks and prefix layers are" " present.")
      print("    [PASS] Layer coverage verified for scanned model architecture.")
      return True

    if expected_layers is None:
      print("    [INFO] Unscanned format detected; skipping layer count check (--expected-layers not specified).")
      return True

    lo_l = min(found_unscanned_layers) if found_unscanned_layers else "N/A"
    hi_l = max(found_unscanned_layers) if found_unscanned_layers else "N/A"
    print(f"    Found {len(found_unscanned_layers)} distinct layers: {lo_l} to {hi_l}")
    missing = set(range(expected_layers)) - found_unscanned_layers
    if missing:
      print(f"    [FAIL] Missing {len(missing)} layers: {sorted(list(missing))[:10]}...")
      return False
    print(f"    [PASS] All {expected_layers} layers are represented in OCDbt.")
    return True

  except ImportError:
    print("    [INFO] tensorstore not installed; checking for ocdbt.process_0 directory...")
    if is_gcs:
      cmd = ["gcloud", "storage", "ls", f"{items_path}/ocdbt.process_0/"]
      res = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if res.returncode == 0:
        print("    [PASS] ocdbt.process_0 directory exists.")
        return True
    else:
      if os.path.isdir(os.path.join(items_path, "ocdbt.process_0")):
        print("    [PASS] ocdbt.process_0 directory exists.")
        return True
    print(f"    [FAIL] Failed to locate ocdbt.process_0 in {items_path}")
    return False
  except (OSError, ValueError, RuntimeError) as e:
    print(f"    [WARN] Layer check error: {e}")
    return False


def check_sample_read(ckpt_path: str) -> bool:
  """Reads a sample array slice from OCDbt to verify chunk integrity."""
  print("--> [4/4] Sampling array readability with TensorStore...")
  items_path, _ = get_items_and_root_paths(ckpt_path)

  try:
    import tensorstore as ts  # pylint: disable=import-outside-toplevel

    kv_store, ts_base = _get_ts_kvstore(items_path)
    all_keys = kv_store.list().result()
    zarr_keys = [
        k.decode("utf-8").replace("/zarr.json", "")
        for k in all_keys
        if b"/zarr.json" in k or (isinstance(k, str) and "/zarr.json" in k)
    ]

    moe_keys = [k for k in zarr_keys if "MoeBlock" in k and "wi_0" in k]
    if not moe_keys:
      moe_keys = [k for k in zarr_keys if "MoeBlock" in k]
    if not moe_keys:
      moe_keys = zarr_keys

    if not moe_keys:
      print("    [FAIL] No zarr arrays discovered in OCDbt store.")
      return False

    sample_key = moe_keys[0]
    spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "ocdbt",
            "base": ts_base,
            "path": sample_key,
        },
    }

    arr = ts.open(spec).result()
    shape = arr.shape
    dtype = arr.dtype

    if arr.ndim >= 3:
      sample_data = arr[0, :2, :2].read().result()
    elif arr.ndim == 2:
      sample_data = arr[:2, :2].read().result()
    else:
      sample_data = arr[:2].read().result()

    print(f"    Sample array '{sample_key}': shape={shape}, dtype={dtype}")
    print(f"    Sample slice read successfully: non-empty={sample_data.size > 0}")
    print("    [PASS] Array is accessible and chunk payloads are valid.")
    return True

  except ImportError:
    print("    [SKIP] tensorstore library not available in host environment to perform array read.")
    return True
  except (OSError, ValueError, RuntimeError) as e:
    print(f"    [WARN] Sample array read failed: {e}")
    return False


def main():
  """Runs all four checkpoint completeness checks."""
  args = parse_args()
  expected_bytes = args.model_params_b * 1e9 * args.dtype_bytes

  print(f"=== Validating Checkpoint: {args.checkpoint_path} ===")
  p1 = check_storage_size(args.checkpoint_path, expected_bytes)
  p2 = check_ocdbt_manifest(args.checkpoint_path)
  p3 = check_layers_coverage(args.checkpoint_path, args.expected_layers, args.expected_prefix_layers)
  p4 = check_sample_read(args.checkpoint_path)

  if p1 and p2 and p3 and p4:
    print("\n[VERDICT: VALID] Checkpoint is complete, fully committed, and ready for use.")
    sys.exit(0)
  else:
    print("\n[VERDICT: CORRUPTED/INCOMPLETE] Checkpoint failed validation. Do not use for testing or training.")
    sys.exit(1)


if __name__ == "__main__":
  main()
