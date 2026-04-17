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

"""
Download a HuggingFace dataset via streaming and save as ArrayRecord files.
Only supports text dataset for now.

Supports writing to local path, including GCS bucket mounted vis GCSFUSE. 
Produces filenames: {name}-XXXXX-of-YYYYY.arrayrecord
User can control file-size-mb, group-size, and workers for parallelism. 

The script runs the following steps:
1. Spawns multiple worker processes.
2. Each worker streams a shard of the dataset.
3. Each example is serialized into a tf.train.Example protobuf message through a
   local copy of the tf protobuf to avoid tf dependency.
4. Workers write records to a temporary file (e.g., worker-000-000123.arrayrecord)
   until it reaches the target --file-size-mb.
5. After the first file, the script learns the on-disk compression ratio and
   adjusts its byte counter to create subsequent files that are more
   accurately sized.
6. After each file is written, a checkpoint is saved. If the script fails
   with a connection error, it retries, rolling back to the last checkpoint.
7. Once all workers are finished, the temporary files are collected, sorted,
   and renamed to a contiguous, conventional naming scheme, e.g.,
   my-dataset-00000-of-01024.arrayrecord.

GCS output recommendation:
    When writing to a GCS bucket, it is recommended to use GCSFUSE to mount
    the bucket and write to the mount path. This avoids local temp files and
    streams data directly to GCS.

    Mount the bucket with write-optimized flags:

        mkdir -p /mnt/gcs-bucket
        gcsfuse \
            --implicit-dirs \
            --enable-streaming-writes=true \
            --write-global-max-blocks=64 \
            --max-conns-per-host=100 \
            --stat-cache-max-size-mb=-1 \
            --metadata-cache-ttl-secs=-1 \
            --type-cache-max-size-mb=-1 \
            my-bucket /mnt/gcs-bucket

    Key flags:
        --enable-streaming-writes=true  Upload directly without local staging
                                        (default in GCSFUSE >= 3.0)
        --write-global-max-blocks=64    Allow up to 64 concurrent streaming
                                        writes (each uses ~96 MB RAM). Set
                                        this >= your --workers count.
        --max-conns-per-host=100        More parallel HTTP connections to GCS
        --implicit-dirs                 Recognize directories from object paths
        --stat/metadata/type-cache      Cache metadata to reduce GCS API calls

    Then pass the mount path as --output.

Usage examples:
    python download_hf_dataset_as_arrayrecord.py \
        --dataset OptimalScale/ClimbMix \
        --output /mnt/gcs-bucket/datasets/climbmix/ \
        --file-size-mb 1000 \
        --workers 8

For large datasets, the conversion can take hours or even days. It is
recommended to run the script inside a tmux or screen session so it
survives disconnections:

    # Using tmux
    tmux new -s ar-convert
    python download_hf_dataset_as_arrayrecord.py <options> ...
    # Detach with Ctrl-b d, reattach later with: tmux attach -t ar-convert

    # Using screen (with logging)
    screen -L -S ar-convert
    python download_hf_dataset_as_arrayrecord.py <options> ...
    # Detach with Ctrl-a d, reattach later with: screen -r ar-convert
"""

import argparse
import json
import multiprocessing
import os
import time

import requests

from maxtext.input_pipeline.protos import example_pb2
from maxtext.input_pipeline.protos import feature_pb2

from array_record.python.array_record_module import ArrayRecordWriter
from datasets import load_dataset

MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 30


def parse_args():
  """parse custom args"""
  parser = argparse.ArgumentParser(
      description="Convert a HuggingFace streaming dataset to ArrayRecord files.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--dataset",
      required=True,
      help="HuggingFace dataset name (e.g. OptimalScale/ClimbMix)",
  )
  parser.add_argument(
      "--config",
      default=None,
      help='Dataset config as a JSON string of kwargs for load_dataset (e.g. \'{"name": "subset", "data_dir": "subdir"}\')',
  )
  parser.add_argument(
      "--split",
      default="train",
      help="Dataset split to convert (default: train)",
  )
  parser.add_argument(
      "--output",
      required=True,
      help="Output directory (local path or GCSFUSE mount).",
  )
  parser.add_argument(
      "--name-prefix",
      default=None,
      help="Filename prefix (default: derived from dataset name)",
  )
  parser.add_argument(
      "--file-size-mb",
      type=int,
      default=500,
      help="Target file size in MB before rotating to next file (default: 500)",
  )
  parser.add_argument(
      "--group-size",
      type=int,
      default=1,
      help="ArrayRecord group_size. Use 1 for random access support (default: 1)",
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=None,
      help="Number of parallel worker processes (default: number of CPU cores)",
  )
  parser.add_argument(
      "--token",
      default=None,
      help="HuggingFace auth token for gated datasets",
  )
  return parser.parse_args()


def _to_feature(value):
  """Convert a Python value to a feature_pb2.Feature.

  Supports str, int, float, bool, bytes, and lists of these types.
  Nested dicts, None, and other types are serialized as JSON bytes.
  """
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
  """Serialize a dict as a example_pb2.Example protobuf byte string."""
  features = {k: _to_feature(v) for k, v in example.items()}
  return example_pb2.Example(features=feature_pb2.Features(feature=features)).SerializeToString()


def process_shard(task):
  """
  Process one shard of the dataset. Each worker calls this independently.

  Each example is serialized as a example_pb2.Example protobuf and written
  as one record. Files are rotated when the accumulated byte size exceeds
  the target file size (adjusted for compression after the first file).

  After each file is written, a checkpoint is saved containing the dataset
  iterator state and progress. On restart, if a checkpoint exists, the
  worker resumes from where it left off.
  """
  worker_id = task["worker_id"]
  num_workers = task["num_workers"]
  dataset_name = task["dataset"]
  config = task["config"]
  split = task["split"]
  token = task["token"]
  output_dir = task["output"]
  file_size_bytes = task["file_size_bytes"]
  group_size = task["group_size"]
  checkpoint_path = os.path.join(output_dir, f".checkpoint-worker-{worker_id:04d}.json")
  kwargs = {"path": dataset_name, "streaming": True, "split": split, "token": token}
  if config:
    kwargs.update(json.loads(config))

  def load_and_restore():
    """Unify dataset loading and restoration logic."""
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

  ds, state = load_and_restore()
  if state["filenames"]:
    print(f"  [Worker {worker_id}] Resumed: {state['total_written']:,} records, {len(state['filenames'])} files written")

  local_file_idx, total_bytes, total_written, filenames, adjusted_file_size_bytes = (
      state[k] for k in ["local_file_idx", "total_bytes", "total_written", "filenames", "adjusted_file_size_bytes"]
  )

  file_bytes, writer, t0 = 0, None, time.time()

  def open_writer():
    nonlocal writer
    fname = f"worker-{worker_id:04d}-{local_file_idx:06d}.arrayrecord"
    writer = ArrayRecordWriter(os.path.join(output_dir, fname), f"group_size:{group_size}")
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
          # Check actual file size on disk
          actual_size = os.path.getsize(os.path.join(output_dir, filenames[-1]))
          # Adjust threshold
          if actual_size > 0:
            adjusted_file_size_bytes = int(file_size_bytes / (actual_size / file_bytes))

          elapsed = time.time() - t0
          print(
              f"  [Worker {worker_id}] File {local_file_idx} written | "
              f"{actual_size/1e6:.0f} MB | {total_written/elapsed:,.0f} rec/sec"
          )
          local_file_idx += 1
          file_bytes = 0
          save_checkpoint()
          open_writer()
      break
    except (requests.exceptions.RequestException, ConnectionError) as e:
      close_writer()
      if attempt >= MAX_RETRIES - 1:
        print(f"  [Worker {worker_id}] Max retries reached. Re-run the script to resume from checkpoint.")
        raise
      print(
          f"  [Worker {worker_id}] Connection error: {e}. Rolling back to last checkpoint."
          f"Retrying in {RETRY_WAIT_SECONDS}s, ({attempt+1}/{MAX_RETRIES})..."
      )
      time.sleep(RETRY_WAIT_SECONDS)
      # Reload state from the last successful checkpoint
      ds, state = load_and_restore()
      local_file_idx, total_bytes, total_written, filenames, adjusted_file_size_bytes = (
          state[k] for k in ["local_file_idx", "total_bytes", "total_written", "filenames", "adjusted_file_size_bytes"]
      )
      file_bytes = 0
      open_writer()

  close_writer()
  elapsed = time.time() - t0
  print(f"  [Worker {worker_id}] Done: {total_written:,} records, {len(filenames)} files in {elapsed:.1f}s")
  return {"worker_id": worker_id, "total_records": total_written, "total_bytes": total_bytes, "filenames": filenames}


def rename_to_convention(output_dir, all_filenames, prefix):
  """Rename worker temp files to a contiguous naming scheme."""
  # Filter for existing, non-empty files first
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
    new_name = f"{prefix}-{idx:0{width}d}-of-{total_files:0{width}d}.arrayrecord"
    if old_name != new_name:
      os.rename(os.path.join(output_dir, old_name), os.path.join(output_dir, new_name))

  print(
      f"  Renamed to: {prefix}-00000-of-{total_files:0{width}d}.arrayrecord ... "
      f"{prefix}-{total_files - 1:0{width}d}-of-{total_files:0{width}d}.arrayrecord"
  )


def convert(args):
  """Main conversion orchestrator."""
  if args.output.startswith("gs://"):
    raise ValueError("gs:// paths are not supported. Mount the bucket with GCSFUSE and pass the mount path instead.")
  num_workers = args.workers or os.cpu_count()

  # Check shard count to avoid worker > shards error
  kwargs_shards = {"path": args.dataset, "streaming": True, "split": args.split, "token": args.token}
  if args.config:
    kwargs_shards.update(json.loads(args.config))
  ds_for_shards = load_dataset(**kwargs_shards)
  if hasattr(ds_for_shards, "n_shards") and ds_for_shards.n_shards:
    if ds_for_shards.n_shards < num_workers:
      print(
          f"{num_workers=} exceeds available data shard: {ds_for_shards.n_shards}. "
          f"Setting workers to {ds_for_shards.n_shards}"
      )
      num_workers = ds_for_shards.n_shards

  prefix = args.name_prefix or args.dataset.split("/")[-1].lower().replace(" ", "-")
  output_dir = os.path.abspath(args.output)
  os.makedirs(output_dir, exist_ok=True)

  # Build tasks
  tasks = [
      {
          "worker_id": i,
          "num_workers": num_workers,
          "dataset": args.dataset,
          "config": args.config,
          "split": args.split,
          "token": args.token,
          "output": output_dir,
          "file_size_bytes": args.file_size_mb * 1_000_000,
          "group_size": args.group_size,
      }
      for i in range(num_workers)
  ]
  print(f"Dataset: {args.dataset} (split: {args.split}) | Target: {args.file_size_mb} MB | Workers: {num_workers}\n")

  # Run workers
  t0 = time.time()
  with multiprocessing.Pool(processes=num_workers) as pool:
    results = pool.map(process_shard, tasks)
  write_elapsed = time.time() - t0

  # Collect filenames and rename
  all_filenames = []
  for r in sorted(results, key=lambda x: x["worker_id"]):
    all_filenames.extend(r["filenames"])

  rename_to_convention(output_dir, all_filenames, prefix)

  # Clean up checkpoint files now that everything is successful
  for i in range(num_workers):
    path = os.path.join(output_dir, f".checkpoint-worker-{i:04d}.json")
    if os.path.exists(path):
      os.remove(path)

  total_elapsed, total_records = time.time() - t0, sum(r["total_records"] for r in results)
  print(f"\nDone! Wrote {total_records:,} records in {total_elapsed:.1f}s ({total_records/write_elapsed:,.0f} rec/sec)")


def main():
  args = parse_args()
  convert(args)


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn", force=True)
  main()
