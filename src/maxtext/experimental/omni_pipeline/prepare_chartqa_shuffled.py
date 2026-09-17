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

import argparse
import os
import shutil
import subprocess
import tempfile

if os.path.exists("/dev/shm"):
  os.environ.setdefault("TMPDIR", "/dev/shm")
  os.environ.setdefault("HF_HOME", "/dev/shm/huggingface")
  os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/huggingface/datasets")
  os.environ.setdefault("TRANSFORMERS_CACHE", "/dev/shm/huggingface/transformers")
  os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/dev/shm/huggingface/hub")
  tempfile.tempdir = "/dev/shm"

from datasets import load_dataset


def main():
  parser = argparse.ArgumentParser(description="Prepare shuffled ChartQA parquet files for MaxText SFT.")
  parser.add_argument(
      "--output_dir",
      type=str,
      help="GCS bucket or local directory path to store parquet shards (e.g., gs://YOUR_BUCKET/datasets/chartqa_shuffled)",
  )
  parser.add_argument(
      "--num_shards",
      type=int,
      default=32,
      help="Number of shards (should be >= number of TPU hosts; 32 or 64 is ideal for TPU v4-128).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for global shuffling.",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      default=os.environ.get("HF_TOKEN", None),
      help="HuggingFace access token if needed.",
  )
  args = parser.parse_args()

  is_gcs = args.output_dir.startswith("gs://")
  if is_gcs:
    local_staging_dir = "/dev/shm/chartqa_shuffled_staging" if os.path.exists("/dev/shm") else "/tmp/chartqa_shuffled_staging"
    os.makedirs(local_staging_dir, exist_ok=True)
    out_prefix = local_staging_dir
  else:
    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = args.output_dir.rstrip("/")

  ds_train = load_dataset("HuggingFaceM4/ChartQA", split="train", token=args.hf_token)
  total_samples = len(ds_train)
  print(f"Loaded {total_samples} training samples.")

  # Perform TRUE global shuffle across all samples
  print(f"Performing global shuffle with seed={args.seed}...")
  ds_train_shuffled = ds_train.shuffle(seed=args.seed)

  for shard_idx in range(args.num_shards):
    shard = ds_train_shuffled.shard(num_shards=args.num_shards, index=shard_idx, contiguous=True)
    out_file = f"{out_prefix}/train-{shard_idx:05d}-of-{args.num_shards:05d}.parquet"
    print(f"  Writing shard {shard_idx + 1}/{args.num_shards} ({len(shard)} samples) -> {out_file}")
    shard.to_parquet(out_file)

  # Prepare val split sharded across hosts
  ds_val = load_dataset("HuggingFaceM4/ChartQA", split="val", token=args.hf_token)
  for shard_idx in range(args.num_shards):
    shard = ds_val.shard(num_shards=args.num_shards, index=shard_idx, contiguous=True)
    val_file = f"{out_prefix}/val-{shard_idx:05d}-of-{args.num_shards:05d}.parquet"
    print(f"  Writing val shard {shard_idx + 1}/{args.num_shards} ({len(shard)} samples) -> {val_file}")
    shard.to_parquet(val_file)

  # Upload to GCS if needed
  if is_gcs:
    print(f"\nUploading from local staging ({local_staging_dir}) to GCS ({args.output_dir})...")
    subprocess.run(["gcloud", "storage", "cp", "-r", f"{local_staging_dir}/*", args.output_dir.rstrip("/") + "/"], check=True)
    shutil.rmtree(local_staging_dir, ignore_errors=True)

  print(f"\nDone! ChartQA parquet dataset saved to {args.output_dir}")


if __name__ == "__main__":
  main()

