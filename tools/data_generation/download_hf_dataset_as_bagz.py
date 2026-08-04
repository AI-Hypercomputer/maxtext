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
Download a HuggingFace dataset via streaming and save as Bagz files.
Only supports text dataset for now.

Supports writing to local path, including GCS bucket mounted via GCSFUSE. 
Produces filenames: {name}-XXXXX-of-YYYYY.bagz
User can control file-size-mb and workers for parallelism. 

GCS output recommendation:
    Mount the bucket with write-optimized GCSFUSE flags and pass the mount path as --output.
"""

import argparse
import json
import multiprocessing
import os
import pathlib
import shutil
import sys
import time

from maxtext.input_pipeline.protos import example_pb2
from maxtext.input_pipeline.protos import feature_pb2

from datasets import load_dataset
import requests
import bagz
MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 30


def parse_args():
  parser = argparse.ArgumentParser(
      description="Convert a HuggingFace streaming dataset to Bagz files.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g. Salesforce/wikitext)")
  parser.add_argument("--config", default=None, help="Dataset config as a JSON string for load_dataset")
  parser.add_argument("--split", default="train", help="Dataset split to convert (default: train)")
  parser.add_argument("--output", required=True, help="Output directory (local path or GCSFUSE mount).")
  parser.add_argument("--name-prefix", default=None, help="Filename prefix (default: derived from dataset)")
  parser.add_argument("--file-size-mb", type=float, default=1000.0, help="Target file size per Bagz shard in MB")
  parser.add_argument("--workers", type=int, default=None, help="Number of concurrent worker processes")
  parser.add_argument("--token", default=None, help="HuggingFace auth token for gated datasets")
  return parser.parse_args()


def _to_feature(value):
  if isinstance(value, bool):
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[int(value)]))
  if isinstance(value, int):
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[value]))
  if isinstance(value, float):
    return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[value]))
  if isinstance(value, (str, bytes)):
    v = value.encode("utf-8") if isinstance(value, str) else value
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[v]))
  if isinstance(value, list):
    if not value:
      return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[]))
    first = value[0]
    if isinstance(first, (bool, int)):
      return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[int(v) for v in value]))
    if isinstance(first, float):
      return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=value))
    if isinstance(first, (str, bytes)):
      v = [x.encode("utf-8") if isinstance(x, str) else x for x in value]
      return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=v))

  json_v = json.dumps(value, ensure_ascii=False).encode("utf-8") if value is not None else b""
  return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[json_v] if json_v else []))


def serialize_example(example: dict) -> bytes:
  features = {k: _to_feature(v) for k, v in example.items()}
  return example_pb2.Example(features=feature_pb2.Features(feature=features)).SerializeToString()


def process_shard(task):
  worker_id = task["worker_id"]
  num_workers = task["num_workers"]
  dataset_name = task["dataset"]
  config = task["config"]
  split = task["split"]
  token = task["token"]
  output_dir = task["output"]
  file_size_bytes = task["file_size_bytes"]
  checkpoint_path = os.path.join(output_dir, f".checkpoint-worker-{worker_id:04d}.json")

  kwargs = {"path": dataset_name, "streaming": True, "split": split, "token": token}
  if config:
    kwargs.update(json.loads(config))

  def load_and_restore():
    ds = load_dataset(**kwargs).shard(num_shards=num_workers, index=worker_id)
    if os.path.exists(checkpoint_path):
      with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
      ds.load_state_dict(ckpt["ds_state"])
      return ds, ckpt
    return ds, {
        "local_file_idx": 0,
        "total_bytes": 0,
        "total_written": 0,
        "filenames": [],
        "adjusted_file_size_bytes": file_size_bytes,
    }

  for attempt in range(MAX_RETRIES):
    try:
      ds, state = load_and_restore()
      break
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
      print(f"Worker {worker_id}: Connection failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
      if attempt + 1 == MAX_RETRIES:
        raise
      time.sleep(RETRY_WAIT_SECONDS)

  local_file_idx, total_bytes, total_written, filenames, adjusted_file_size_bytes = (
      state[k] for k in ["local_file_idx", "total_bytes", "total_written", "filenames", "adjusted_file_size_bytes"]
  )

  file_bytes, writer, t0 = 0, None, time.time()

  def open_writer():
    nonlocal writer
    fname = f"worker-{worker_id:04d}-{local_file_idx:06d}.bagz"
    writer = bagz.Writer(os.path.join(output_dir, fname))
    filenames.append(fname)

  def close_writer():
    nonlocal writer
    if writer:
      writer.close()
      writer = None

  def save_checkpoint():
    ckpt = {
        "worker_id": worker_id,
        "local_file_idx": local_file_idx,
        "total_bytes": total_bytes,
        "total_written": total_written,
        "filenames": filenames,
        "adjusted_file_size_bytes": adjusted_file_size_bytes,
        "ds_state": ds.state_dict(),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
      json.dump(ckpt, f)

  open_writer()
  for attempt in range(MAX_RETRIES):
    try:
      for example in ds:
        record = serialize_example(example)
        record_size = len(record)
        writer.write(record)
        file_bytes += record_size
        total_bytes += record_size
        total_written += 1

        if file_bytes >= adjusted_file_size_bytes:
          close_writer()
          actual_size = os.path.getsize(os.path.join(output_dir, filenames[-1]))
          if actual_size > 0:
            adjusted_file_size_bytes = int(file_size_bytes / (actual_size / file_bytes))

          elapsed = time.time() - t0
          speed = (file_bytes / 1024 / 1024) / elapsed
          print(f"Worker {worker_id}: Finished shard {local_file_idx} ({actual_size/1024/1024:.1f} MB) in {elapsed:.1f}s ({speed:.1f} MB/s)")
          
          local_file_idx += 1
          file_bytes = 0
          save_checkpoint()
          open_writer()
          t0 = time.time()
      
      close_writer()
      if file_bytes == 0:
        filenames.pop()
      else:
        actual_size = os.path.getsize(os.path.join(output_dir, filenames[-1]))
        elapsed = time.time() - t0
        speed = (file_bytes / 1024 / 1024) / elapsed
        print(f"Worker {worker_id}: Finished FINAL shard {local_file_idx} ({actual_size/1024/1024:.1f} MB) in {elapsed:.1f}s ({speed:.1f} MB/s)")
        save_checkpoint()

      if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
      return filenames

    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
      print(f"Worker {worker_id}: Connection lost during iteration (attempt {attempt+1}/{MAX_RETRIES}): {e}")
      close_writer()
      if attempt + 1 == MAX_RETRIES:
        raise
      time.sleep(RETRY_WAIT_SECONDS)

  return filenames


def rename_files(all_filenames, output_dir, prefix):
  print("\nVerifying downloaded files...")
  existing_files = []
  for fname in all_filenames:
    fpath = os.path.join(output_dir, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
      existing_files.append(fname)
    else:
      print(f"  [Warning] File not found or empty, skipping: {fpath}")

  total_files = len(existing_files)
  if total_files == 0:
    print("No valid files to rename.")
    return

  print(f"\nRenaming {total_files} verified files...")
  existing_files.sort()
  width = max(5, len(str(total_files)))

  for idx, old_name in enumerate(existing_files):
    new_name = f"{prefix}-{idx:0{width}d}-of-{total_files:0{width}d}.bagz"
    if old_name != new_name:
      os.rename(os.path.join(output_dir, old_name), os.path.join(output_dir, new_name))

  print(f"  Renamed to: {prefix}-00000-of-{total_files:0{width}d}.bagz ... {prefix}-{total_files - 1:0{width}d}-of-{total_files:0{width}d}.bagz")


def convert(args):
  if args.output.startswith("gs://"):
    raise ValueError("gs:// paths are not supported. Mount the bucket with GCSFUSE and pass the mount path instead.")
  num_workers = args.workers or os.cpu_count()
  file_size_bytes = int(args.file_size_mb * 1024 * 1024)

  dataset_name_safe = args.dataset.replace("/", "_")
  prefix = args.name_prefix or dataset_name_safe
  output_dir = os.path.abspath(args.output)
  os.makedirs(output_dir, exist_ok=True)

  tasks = []
  for i in range(num_workers):
    tasks.append({
        "worker_id": i,
        "num_workers": num_workers,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "token": args.token,
        "output": output_dir,
        "file_size_bytes": file_size_bytes,
    })

  start_time = time.time()
  print(f"Starting conversion of '{args.dataset}' with {num_workers} workers...")
  
  if num_workers == 1:
    results = [process_shard(tasks[0])]
  else:
    with multiprocessing.Pool(processes=num_workers) as pool:
      results = pool.map(process_shard, tasks)

  all_filenames = []
  for worker_filenames in results:
    all_filenames.extend(worker_filenames)

  rename_files(all_filenames, output_dir, prefix)

  total_time = time.time() - start_time
  print(f"\nSuccessfully converted dataset in {total_time/60:.1f} minutes.")


if __name__ == "__main__":
  convert(parse_args())
