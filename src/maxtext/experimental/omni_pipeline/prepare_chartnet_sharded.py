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
  parser = argparse.ArgumentParser(description="Prepare sharded ChartNet parquet files for MaxText pretraining.")
  parser.add_argument(
      "--output_dir",
      type=str,
      help="GCS bucket or local directory path to store parquet shards (e.g., gs://YOUR_BUCKET/datasets/chartnet_sharded)",
  )
  parser.add_argument(
      "--num_shards",
      type=int,
      default=32,
      help="Number of shards (32 or 64 is ideal for TPU v4-128 / 32 hosts).",
  )
  parser.add_argument(
      "--shuffle",
      action="store_true",
      default=False,
      help="Whether to shuffle the dataset before sharding (default: False).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for shuffling if --shuffle is enabled.",
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
    local_staging_dir = "/dev/shm/chartnet_sharded_staging" if os.path.exists("/dev/shm") else "/tmp/chartnet_sharded_staging"
    os.makedirs(local_staging_dir, exist_ok=True)
    out_prefix = local_staging_dir
  else:
    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = args.output_dir.rstrip("/")

  # 1. Prepare train split (~92,643 examples)
  ds_train = load_dataset("ibm-granite/ChartNet", name="human_verified", split="train", token=args.hf_token)
  total_samples = len(ds_train)
  print(f"Loaded {total_samples} ChartNet human_verified training samples.")

  if args.shuffle:
    print(f"Shuffling training dataset with seed={args.seed}...")
    ds_train = ds_train.shuffle(seed=args.seed)

  for shard_idx in range(args.num_shards):
    shard = ds_train.shard(num_shards=args.num_shards, index=shard_idx, contiguous=True)
    out_file = f"{out_prefix}/train-{shard_idx:05d}-of-{args.num_shards:05d}.parquet"
    print(f"  Writing train shard {shard_idx + 1}/{args.num_shards} ({len(shard)} samples) -> {out_file}")
    shard.to_parquet(out_file)

  # 2. Prepare test / eval split (~2,000 examples)
  ds_test = load_dataset("ibm-granite/ChartNet", name="human_verified", split="test", token=args.hf_token)
  for shard_idx in range(args.num_shards):
    shard = ds_test.shard(num_shards=args.num_shards, index=shard_idx, contiguous=True)
    val_file = f"{out_prefix}/test-{shard_idx:05d}-of-{args.num_shards:05d}.parquet"
    print(f"  Writing test shard {shard_idx + 1}/{args.num_shards} ({len(shard)} samples) -> {val_file}")
    shard.to_parquet(val_file)

  # Upload to GCS if needed
  if is_gcs:
    print(f"\nUploading from local staging ({local_staging_dir}) to GCS ({args.output_dir})...")
    subprocess.run(["gcloud", "storage", "cp", "-r", f"{local_staging_dir}/*", args.output_dir.rstrip("/") + "/"], check=True)
    shutil.rmtree(local_staging_dir, ignore_errors=True)

  print(f"\nDone! ChartNet parquet dataset saved to {args.output_dir}")


if __name__ == "__main__":
  main()
