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

"""Stream-downloads and stages the ~300GB ibm-granite/ChartNet 'core' subset file-by-file directly to GCS.

Operates with minimal local disk usage (< 3 GB peak) by downloading 1 parquet file
at a time, uploading immediately to GCS, and instantly removing the local file.
Supports resuming from where it left off.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from huggingface_hub import HfApi, hf_hub_download


def get_gcs_existing_files(gcs_dir: str) -> set[str]:
  """Returns the set of filenames already present in the destination GCS directory."""
  if not gcs_dir.startswith("gs://"):
    return set()
  try:
    cmd = ["gcloud", "storage", "ls", f"{gcs_dir.rstrip('/')}/*.parquet"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
      return set()
    existing = set()
    for line in result.stdout.strip().splitlines():
      line = line.strip()
      if line:
        existing.add(os.path.basename(line))
    return existing
  except Exception as e:
    print(f"Warning: Could not list existing GCS files ({e}). Starting fresh.")
    return set()


def cleanup_shm_caches():
  """Cleans up partial Hugging Face and staging caches to free /dev/shm memory."""
  paths_to_clean = [
      "/dev/shm/chartnet_core_raw",
      "/dev/shm/chartnet_core_staged",
      "/dev/shm/chartnet_core_sharded_staging",
      "/dev/shm/chartnet_core_temp",
      "/dev/shm/huggingface/hub/datasets--ibm-granite--ChartNet",
      "/dev/shm/huggingface/datasets/ibm-granite___chart_net",
      "/dev/shm/huggingface/xet",
  ]
  for p in paths_to_clean:
    if os.path.exists(p):
      try:
        shutil.rmtree(p, ignore_errors=True)
      except Exception:
        pass


def main():
  parser = argparse.ArgumentParser(
      description="Stream download ChartNet 'core' parquet files one-by-one to GCS without exceeding local disk."
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      help="GCS bucket directory (e.g., gs://YOUR_BUCKET/datasets/chartnet_core_sharded) or local path.",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      default=os.environ.get("HF_TOKEN", None),
      help="HuggingFace access token if needed.",
  )
  parser.add_argument(
      "--clean_shm",
      action="store_true",
      default=True,
      help="Clean stale /dev/shm cache directories before starting.",
  )
  args = parser.parse_args()

  if args.clean_shm:
    print("Cleaning up stale /dev/shm cache directories...")
    cleanup_shm_caches()

  api = HfApi(token=args.hf_token)
  print("Discovering files in ibm-granite/ChartNet repository...")
  all_files = api.list_repo_files(repo_id="ibm-granite/ChartNet", repo_type="dataset")
  
  # Find all parquet files belonging to the 'core' subset
  core_files = sorted([f for f in all_files if f.startswith("core/") and f.endswith(".parquet")])
  if not core_files:
    # Check alternative naming
    core_files = sorted([f for f in all_files if "core" in f and f.endswith(".parquet")])

  total_files = len(core_files)
  if total_files == 0:
    print("Error: No core parquet files found in ibm-granite/ChartNet repository.")
    sys.exit(1)

  print(f"Found {total_files} parquet files for 'core' subset in Hugging Face repository.")
  is_gcs = args.output_dir.startswith("gs://")

  # Check existing files in GCS for resuming
  existing_files = get_gcs_existing_files(args.output_dir) if is_gcs else set()
  if existing_files:
    print(f"Found {len(existing_files)} files already uploaded in {args.output_dir}. Resuming...")

  # Working temp directory (use /dev/shm or /tmp)
  temp_base = "/dev/shm/chartnet_core_temp" if os.path.exists("/dev/shm") else "/tmp/chartnet_core_temp"
  os.makedirs(temp_base, exist_ok=True)

  for idx, hf_file in enumerate(core_files):
    target_filename = f"train-{idx:05d}-of-{total_files:05d}.parquet"
    
    if target_filename in existing_files:
      print(f"[{idx + 1}/{total_files}] Skipping {target_filename} (already exists in GCS)")
      continue

    print(f"\n[{idx + 1}/{total_files}] Downloading {hf_file} from Hugging Face...")
    file_temp_dir = tempfile.mkdtemp(dir=temp_base)
    try:
      downloaded_file = hf_hub_download(
          repo_id="ibm-granite/ChartNet",
          repo_type="dataset",
          filename=hf_file,
          token=args.hf_token,
          local_dir=file_temp_dir,
      )

      file_size_gb = os.path.getsize(downloaded_file) / (1024 ** 3)
      print(f"  Downloaded {os.path.basename(downloaded_file)} ({file_size_gb:.2f} GB).")

      if is_gcs:
        gcs_dest = f"{args.output_dir.rstrip('/')}/{target_filename}"
        print(f"  Uploading to {gcs_dest}...")
        subprocess.run(["gcloud", "storage", "cp", downloaded_file, gcs_dest], check=True)
      else:
        os.makedirs(args.output_dir, exist_ok=True)
        dest_path = os.path.join(args.output_dir, target_filename)
        shutil.copyfile(downloaded_file, dest_path)
        print(f"  Saved to {dest_path}")

      print(f"  [OK] Shard {idx + 1}/{total_files} successfully uploaded.")
    finally:
      # Immediately clean up downloaded file and temp dir to free disk space
      shutil.rmtree(file_temp_dir, ignore_errors=True)
      # Also clean HF cache dir if created
      hf_hub_cache = os.path.join(temp_base, ".huggingface")
      if os.path.exists(hf_hub_cache):
        shutil.rmtree(hf_hub_cache, ignore_errors=True)

  shutil.rmtree(temp_base, ignore_errors=True)
  print(f"\n==================================================================")
  print(f"All {total_files} shards successfully uploaded to {args.output_dir}!")
  print(f"==================================================================")


if __name__ == "__main__":
  main()
