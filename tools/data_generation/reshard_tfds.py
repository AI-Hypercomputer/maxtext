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

"""
Script to reshard a TFDS dataset into a specific number of shards.
This is useful when the number of hosts for dataloading is larger than the number of shards in the dataset.

Example (num_workers, buffer_size, dataset_name are optional):

For Single Split Dataset:

  python3 tools/data_generation/reshard_tfds.py \
    --src_dir gs://your-bucket/origin_folder \
    --dst_dir gs://your-bucket/new_folder \
    --num_shards 2048 \
    --split train \
    --num_workers 16 \
    --buffer_size 33554432

For Multiple Splits Dataset:

  python3 tools/data_generation/reshard_tfds.py \
    --src_dir gs://your-bucket/origin_folder \
    --dst_dir gs://your-bucket/new_folder \
    --num_shards 2048 \
    --split train,validation \
    --num_workers 16 \
    --buffer_size 33554432

"""

import argparse
import os
import json
import multiprocessing
import queue
import threading

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def parse_args():
  """Parses command-line arguments for the resharding script."""
  parser = argparse.ArgumentParser(description="Reshard a TFDS dataset.")
  parser.add_argument("--src_dir", type=str, required=True, help="Source TFDS directory (e.g., gs://bucket/c4/en/3.0.1)")
  parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory")
  parser.add_argument("--num_shards", type=int, default=2048, help="Number of shards for the output (default: 2048)")
  parser.add_argument("--split", type=str, default="train", help="Split(s) to reshard, comma-separated (default: train)")
  parser.add_argument(
      "--dataset_name", type=str, default=None, help="Optional dataset name. If not set, inferred from metadata."
  )
  parser.add_argument("--num_workers", type=int, default=16, help="Optional number of workers (default: 16)")
  parser.add_argument(
      "--buffer_size",
      type=int,
      default=32 * 1024 * 1024,
      help="Optional buffer size in bytes for TFRecordDataset (default: 32MB)",
  )
  return parser.parse_args()


def get_shard_path(dst_dir, dataset_name, split, shard_index, total_shards):
  """Constructs the standard TFDS filename for a specific shard."""
  shard_name = f"{dataset_name}-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}"
  return os.path.join(dst_dir, shard_name)


def reshard_raw_bytes_worker(
    worker_id, num_workers, src_files, dst_dir, dataset_name, split, total_shards, buffer_size, progress_queue
):
  """
  Worker function that reads raw TFRecord bytes and distributes them to target shards.

  Each worker reads a subset of the source dataset and writes to a specific subset
  of target shards (based on its worker_id) to avoid write collisions.
  """
  # Dictionary to keep track of active writers and the number of records written to each
  writers = {}
  shard_lengths = {}

  def get_writer(shard_idx):
    """Helper to lazily initialize a TFRecordWriter for a given target shard."""
    if shard_idx not in writers:
      path = get_shard_path(dst_dir, dataset_name, split, shard_idx, total_shards)
      writers[shard_idx] = tf.io.TFRecordWriter(path)
      shard_lengths[shard_idx] = 0
    return writers[shard_idx]

  # Initialize a tf.data.Dataset to read raw bytes from the source TFRecord files.
  # A large buffer size (default 32MB) is used to improve I/O throughput, especially on GCS.
  ds = tf.data.TFRecordDataset(src_files, compression_type=None, buffer_size=buffer_size)

  # Shard the dataset so this worker only processes its designated portion of the data
  ds = ds.shard(num_workers, worker_id)

  # Iterate through the worker's data slice and write each record to its target shard
  i = -1
  for i, record_bytes in enumerate(ds):
    # Calculate the global index of this record among all records processed
    i_global = i * num_workers + worker_id

    # Determine which target shard this record belongs to (round-robin distribution)
    target_shard_idx = i_global % total_shards

    writer = get_writer(target_shard_idx)
    writer.write(record_bytes.numpy())
    shard_lengths[target_shard_idx] += 1

    # Send progress update every 1000 records
    if (i + 1) % 1000 == 0:
      progress_queue.put(1000)

  # Send any remaining progress
  remainder = (i + 1) % 1000
  if remainder > 0:
    progress_queue.put(remainder)

  # Close all writers opened by this worker to ensure data is flushed to disk
  for writer in writers.values():
    writer.close()

  return shard_lengths


def progress_listener(q, total_examples):
  """Listens to the progress queue and updates a single tqdm progress bar."""
  pbar = tqdm(total=total_examples, desc="Resharding Progress", unit=" records", unit_scale=True)
  while True:
    try:
      # Block briefly to wait for updates
      update = q.get(timeout=0.1)
      if update == "DONE":
        break
      pbar.update(update)
    except queue.Empty:
      continue
  pbar.close()


def main():
  """Main execution flow for reading metadata, sharding data, and updating dataset info."""
  args = parse_args()

  # Create destination directory if it doesn't exist
  if not tf.io.gfile.exists(args.dst_dir):
    tf.io.gfile.makedirs(args.dst_dir)

  target_splits = [s.strip() for s in args.split.split(",") if s.strip()]

  # Load source metadata once
  print(f"Loading metadata from {args.src_dir}...")
  info_path = os.path.join(args.src_dir, "dataset_info.json")
  if not tf.io.gfile.exists(info_path):
    raise FileNotFoundError(f"Required metadata file not found: {info_path}")

  with tf.io.gfile.GFile(info_path, "r") as f:
    info_json = json.load(f)

  dataset_name = args.dataset_name or info_json.get("name")
  if not dataset_name:
    try:
      # Attempt to verify dataset name using TFDS standard builder
      builder = tfds.builder_from_directory(args.src_dir)
      dataset_name = builder.name
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: Could not load metadata via tfds.builder_from_directory: {e}")
      print("Warning: Dataset name could not be determined, and output filenames will use 'unknown'.")
      dataset_name = "unknown"

  # Use a multiprocessing Manager to share a queue between workers and the main process
  with multiprocessing.Manager() as manager:
    for split_name in target_splits:
      print(f"\n--- Processing split: {split_name} ---")
      num_examples = 0

      # Handle splits metadata whether it's a list or dictionary
      splits_meta = info_json.get("splits", {})
      if isinstance(splits_meta, list):
        split_item = next((s for s in splits_meta if s["name"] == split_name), None)
        if split_item:
          num_examples = int(split_item.get("numExamples", split_item.get("num_examples", 0)))
      else:
        split_item = splits_meta.get(split_name)
        if split_item:
          num_examples = int(split_item.get("numExamples", split_item.get("num_examples", 0)))

      # Find source TFRecord files using common TFDS naming patterns
      pattern = os.path.join(args.src_dir, f"{dataset_name}-{split_name}.tfrecord*")
      src_files = tf.io.gfile.glob(pattern)
      src_files.sort()

      if not src_files:
        pattern = os.path.join(args.src_dir, f"{split_name}.tfrecord*")
        src_files = tf.io.gfile.glob(pattern)
        src_files.sort()

      if not src_files:
        raise FileNotFoundError(f"Could not find TFRecord files for split '{split_name}' in {args.src_dir}")

      print(f"Found {len(src_files)} source files for split '{split_name}' ({num_examples} examples).")

      # Setup multiprocessing pool
      num_workers = args.num_workers

      # Ensure the target number of shards is divisible by the number of workers
      # to maintain proper load balancing and deterministic write distributions
      if args.num_shards % num_workers != 0:
        for i in range(num_workers, 0, -1):
          if args.num_shards % i == 0:
            num_workers = i
            break
        print(f"Adjusted num_workers to {num_workers} to be a factor of {args.num_shards}")

      print(f"Resharding into {args.num_shards} shards using {num_workers} workers...")

      progress_queue = manager.Queue()

      # Start the listener thread in the background to consume progress updates
      listener_thread = threading.Thread(
          target=progress_listener, args=(progress_queue, num_examples if num_examples > 0 else None)
      )
      listener_thread.start()

      # Prepare worker arguments and launch the pool
      tasks = []
      for i in range(num_workers):
        tasks.append(
            (
                i,
                num_workers,
                src_files,
                args.dst_dir,
                dataset_name,
                split_name,
                args.num_shards,
                args.buffer_size,
                progress_queue,
            )
        )

      with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(reshard_raw_bytes_worker, tasks)

      # Signal the listener thread that work is complete and wait for it to join
      progress_queue.put("DONE")
      listener_thread.join()

      # Aggregate the results (shard lengths) from all workers
      all_shard_lengths = {}
      for r in results:
        all_shard_lengths.update(r)

      # Verify the total number of examples processed matches the original metadata
      total_count = sum(all_shard_lengths.values())
      print(f"Successfully resharded {total_count} examples for '{split_name}'.")
      if num_examples > 0 and total_count != num_examples:
        print(f"Warning: Total examples {total_count} does not match original {num_examples} for split '{split_name}'.")

      # Update the shard count and lengths in the JSON metadata for this split
      shard_lengths_list = [all_shard_lengths.get(i, 0) for i in range(args.num_shards)]

      if "splits" not in info_json:
        info_json["splits"] = {}
      splits_meta = info_json["splits"]

      if isinstance(splits_meta, list):
        found = False
        for split_item in splits_meta:
          if split_item.get("name") == split_name:
            split_item["shardLengths"] = [str(l) for l in shard_lengths_list]
            split_item["numShards"] = str(args.num_shards)
            split_item["numExamples"] = str(total_count)
            found = True
            break
        if not found:
          splits_meta.append(
              {
                  "name": split_name,
                  "shardLengths": [str(l) for l in shard_lengths_list],
                  "numShards": str(args.num_shards),
                  "numExamples": str(total_count),
              }
          )
      else:
        if split_name in splits_meta:
          splits_meta[split_name]["shardLengths"] = [str(l) for l in shard_lengths_list]
          splits_meta[split_name]["numShards"] = str(args.num_shards)
          if "numExamples" in splits_meta[split_name]:
            splits_meta[split_name]["numExamples"] = str(total_count)
          else:
            splits_meta[split_name]["num_examples"] = str(total_count)
        else:
          splits_meta[split_name] = {
              "shardLengths": [str(l) for l in shard_lengths_list],
              "numShards": str(args.num_shards),
              "numExamples": str(total_count),
          }

  # Create and save updated dataset_info.json for the new dataset
  print("\nCreating new dataset_info.json...")
  dst_info_path = os.path.join(args.dst_dir, "dataset_info.json")
  with tf.io.gfile.GFile(dst_info_path, "w") as f:
    json.dump(info_json, f, indent=4)

  # Copy features.json if it exists (necessary for some TFDS versions/formats)
  features_path = os.path.join(args.src_dir, "features.json")
  if tf.io.gfile.exists(features_path):
    tf.io.gfile.copy(features_path, os.path.join(args.dst_dir, "features.json"), overwrite=True)

  print(f"Done! Resharded dataset available at {args.dst_dir}")


if __name__ == "__main__":
  main()
