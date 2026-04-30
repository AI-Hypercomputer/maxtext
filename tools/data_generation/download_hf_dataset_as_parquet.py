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
Download a HuggingFace dataset via streaming and save as Parquet files.
Only supports text dataset for now.

Supports writing to local disk or GCS (gs://) paths. When using GCS,
mounting the bucket via GCSFUSE is recommended for better perf. 
Produces HuggingFace-style filenames: {name}-XXXXX-of-YYYYY.parquet
User can control num-files, row-group-size, and workers for parallelism. 

The script runs the following steps:
1. Auto-detects or accepts total dataset size to calculate a JSONL byte
   budget for each of the N output Parquet files.
2. Spawns multiple worker processes.
3. Each worker streams a shard of the dataset, collecting examples until its
   byte budget for a single file is met.
4. The worker writes the collected examples to a temporary Parquet file,
   e.g., worker-000-000123.parquet. Checkpoints are saved after each file.
5. If a worker fails with a connection error, it retries, rolling back to
   the last successful checkpoint.
6. After all workers finish, if more files were produced than desired (due
   to dataset size estimation variance), the smallest files are merged.
7. Finally, all temporary files are renamed to the HuggingFace convention,
   e.g., my-dataset-00000-of-02048.parquet.

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
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --row-group-size 50000 \
        --total-size-gb 1900 \
        --output /mnt/gcs-bucket/datasets/climbmix/ \
        --workers 8

For large datasets, the conversion can take hours or even days. It is
recommended to run the script inside a tmux or screen session so it
survives disconnections:

    # Using tmux
    tmux new -s parquet-convert
    python download_hf_dataset_as_parquet.py <options> ...
    # Detach with Ctrl-b d, reattach later with: tmux attach -t parquet-convert

    # Using screen (with logging)
    screen -L -S parquet-convert
    python download_hf_dataset_as_parquet.py <options> ...
    # Detach with Ctrl-a d, reattach later with: screen -r parquet-convert
"""

import argparse
import json
import multiprocessing
import os
import shutil
import time

import requests
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 30


def parse_args():
  """parse custom args"""
  parser = argparse.ArgumentParser(
      description="Convert a HuggingFace streaming dataset to Parquet files.",
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
      "--num-files",
      type=int,
      required=True,
      help="The exact number of Parquet files to produce.",
  )
  parser.add_argument(
      "--output",
      required=True,
      help="Output directory. Supports local paths or gs:// GCS paths.",
  )
  parser.add_argument(
      "--name-prefix",
      default=None,
      help="Filename prefix (default: derived from dataset name)",
  )
  parser.add_argument(
      "--compression",
      default="zstd",
      choices=["zstd", "snappy", "gzip", "none"],
      help="Parquet compression codec (default: zstd)",
  )
  parser.add_argument(
      "--row-group-size",
      type=int,
      default=None,
      help="Number of rows per Parquet row group. If not set, each file is "
      "written as a single row group. HuggingFace recommends targeting "
      "100-300 MB uncompressed per row group.",
  )
  parser.add_argument(
      "--total-size-gb",
      type=float,
      default=None,
      help="Total source data size in GB. Used to calculate per-file byte budgets. "
      "If not provided, auto-detected from HuggingFace repo metadata.",
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


def get_filesystem_and_path(output_path: str):
  """Return (pyarrow filesystem, base_path) for local or GCS paths."""
  is_gcs = output_path.startswith("gs://")
  if is_gcs:
    from pyarrow.fs import GcsFileSystem  # pylint: disable=import-outside-toplevel

    fs, base_path = GcsFileSystem(), output_path[len("gs://") :]
  else:
    from pyarrow.fs import LocalFileSystem  # pylint: disable=import-outside-toplevel

    fs, base_path = LocalFileSystem(), os.path.abspath(output_path)
    os.makedirs(base_path, exist_ok=True)

  return fs, base_path if base_path.endswith("/") else base_path + "/"


def get_total_size_from_hub(dataset: str, token: str = None):
  """Query HuggingFace Hub for total dataset file sizes."""
  from huggingface_hub import HfApi  # pylint: disable=import-outside-toplevel

  api = HfApi()
  entries = api.list_repo_tree(dataset, repo_type="dataset", token=token, recursive=True)
  return sum(e.size for e in entries if hasattr(e, "size") and e.size)


def process_shard(task):
  """
  Process one shard of the dataset. Each worker calls this independently.

  Each worker accumulates examples until the JSONL byte budget for one file
  is met, then writes the entire file at once. PyArrow's write_table with
  row_group_size handles row group splitting internally.

  Files are named worker-{worker_id}-{local_file_idx}.parquet. These can
  be renamed to HuggingFace convention after all workers complete.
  """
  worker_id = task["worker_id"]
  num_workers = task["num_workers"]
  dataset_name = task["dataset"]
  config = task["config"]
  split = task["split"]
  token = task["token"]
  output_path = task["output"]
  row_group_size = task["row_group_size"]
  compression = task["compression"]
  jsonl_bytes_per_file = task["jsonl_bytes_per_file"]
  local_checkpoint_dir = (
      os.path.abspath(output_path)
      if not output_path.startswith("gs://")
      else os.path.join("/tmp", output_path.replace("gs://", "gs_"))
  )
  os.makedirs(local_checkpoint_dir, exist_ok=True)
  checkpoint_path = os.path.join(local_checkpoint_dir, f".checkpoint-worker-{worker_id:04d}.json")
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
    return ds, {"local_file_idx": 0, "total_jsonl_bytes": 0, "total_written": 0, "filenames": []}

  ds, state = load_and_restore()
  local_file_idx, total_jsonl_bytes, total_written, filenames = (
      state[k] for k in ["local_file_idx", "total_jsonl_bytes", "total_written", "filenames"]
  )
  fs, base_path = get_filesystem_and_path(output_path)
  comp = None if compression == "none" else compression
  file_examples, file_jsonl_bytes, t0 = [], 0, time.time()

  def write_file():
    nonlocal local_file_idx, total_written
    fname = f"worker-{worker_id:04d}-{local_file_idx:06d}.parquet"
    pq.write_table(
        pa.Table.from_pylist(file_examples),
        base_path + fname,
        filesystem=fs,
        compression=comp,
        row_group_size=row_group_size,
    )
    filenames.append(fname)
    total_written += len(file_examples)
    elapsed = time.time() - t0
    print(f"  [Worker {worker_id}] File {local_file_idx} written | {total_written/elapsed:,.0f} rows/sec")
    local_file_idx += 1
    ckpt = {
        "worker_id": worker_id,
        "local_file_idx": local_file_idx,
        "total_jsonl_bytes": total_jsonl_bytes,
        "total_written": total_written,
        "filenames": filenames,
        "ds_state": ds.state_dict(),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
      json.dump(ckpt, f)

  for attempt in range(MAX_RETRIES):
    try:
      for example in ds:
        size = len(json.dumps(example, ensure_ascii=False).encode("utf-8"))
        file_examples.append(example)
        file_jsonl_bytes += size
        total_jsonl_bytes += size
        if file_jsonl_bytes >= jsonl_bytes_per_file:
          write_file()
          file_examples, file_jsonl_bytes = [], 0
      break
    except (requests.exceptions.RequestException, ConnectionError) as e:
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
      local_file_idx, total_jsonl_bytes, total_written, filenames = (
          state[k] for k in ["local_file_idx", "total_jsonl_bytes", "total_written", "filenames"]
      )
      file_examples, file_jsonl_bytes = [], 0

  if file_examples:
    write_file()
  elapsed = time.time() - t0
  print(f"  [Worker {worker_id}] Done: {total_written:,} records, {len(filenames)} files in {elapsed:.1f}s")
  return {"worker_id": worker_id, "total_rows": total_written, "total_bytes": total_jsonl_bytes, "filenames": filenames}


def rename_to_hf_convention(fs, base_path, all_filenames, prefix):
  """
  Rename worker temp files to HuggingFace convention.

  Collects all filenames from all workers, sorts them (by worker_id then
  local_file_idx, which is the natural sort order of the temp names),
  and renames to: prefix-00000-of-NNNNN.parquet
  """
  # Filter for existing, non-empty files first
  print("\nVerifying downloaded files...")
  existing_files = []
  for fname in all_filenames:
    fpath = base_path + fname
    try:
      if fs.get_file_info(fpath).size > 0:
        existing_files.append(fname)
      else:
        print(f"  [Warning] File is empty, skipping: {fpath}")
    except FileNotFoundError:
      print(f"  [Warning] File not found, skipping: {fpath}")

  total_files = len(existing_files)
  if total_files == 0:
    print("No valid files to rename.")
    return

  print(f"\nRenaming {total_files} verified files to HuggingFace convention...")
  existing_files.sort()
  width = max(5, len(str(total_files)))

  for idx, old_name in enumerate(existing_files):
    new_name = f"{prefix}-{idx:0{width}d}-of-{total_files:0{width}d}.parquet"
    if old_name != new_name:
      fs.move(base_path + old_name, base_path + new_name)

  print(f"  Renamed to: {prefix}-00000-of-{total_files:0{width}d}.parquet ...")


def convert(args):
  """Main conversion orchestrator."""
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

  # Get total source size
  total_bytes = args.total_size_gb * 1e9 if args.total_size_gb else get_total_size_from_hub(args.dataset, args.token)
  jsonl_bytes_per_file = total_bytes / args.num_files

  # Build tasks
  tasks = [
      {
          "worker_id": i,
          "num_workers": num_workers,
          "dataset": args.dataset,
          "config": args.config,
          "split": args.split,
          "token": args.token,
          "output": args.output,
          "row_group_size": args.row_group_size,
          "compression": args.compression,
          "jsonl_bytes_per_file": jsonl_bytes_per_file,
      }
      for i in range(num_workers)
  ]
  print(f"Plan: {total_bytes/1e9:.1f} GB -> ~{args.num_files} files | Workers: {num_workers}\n")

  # Run workers
  t0 = time.time()
  with multiprocessing.Pool(processes=num_workers) as pool:
    results = pool.map(process_shard, tasks)
  write_elapsed = time.time() - t0

  # Collect filenames, merge the smallest files to reach num-files and rename
  all_filenames = []
  for r in sorted(results, key=lambda x: x["worker_id"]):
    all_filenames.extend(r["filenames"])
  fs, base_path = get_filesystem_and_path(args.output)

  if len(all_filenames) > args.num_files:
    print(f"\nMerging {len(all_filenames) - args.num_files + 1} smallest files into one...")
    infos = sorted([(f, fs.get_file_info(base_path + f).size) for f in all_filenames], key=lambda x: x[1])
    num_merge = len(all_filenames) - args.num_files + 1
    to_merge, remain = [x[0] for x in infos[:num_merge]], [x[0] for x in infos[num_merge:]]
    merged_name = f"merged-{int(time.time())}.parquet"
    pq.write_table(
        pa.concat_tables([pq.read_table(base_path + f, filesystem=fs) for f in to_merge]),
        base_path + merged_name,
        filesystem=fs,
        compression=None if args.compression == "none" else args.compression,
        row_group_size=args.row_group_size,
    )
    for f in to_merge:
      fs.delete_file(base_path + f)
    all_filenames = remain + [merged_name]

  rename_to_hf_convention(
      fs, base_path, all_filenames, args.name_prefix or args.dataset.split("/")[-1].lower().replace(" ", "-")
  )

  # Clean up checkpoint files
  is_gcs = args.output.startswith("gs://")
  local_checkpoint_dir = (
      os.path.join("/tmp", args.output.replace("gs://", "gs_")) if is_gcs else os.path.abspath(args.output)
  )
  for i in range(num_workers):
    path = os.path.join(local_checkpoint_dir, f".checkpoint-worker-{i:04d}.json")
    if os.path.exists(path):
      os.remove(path)
  if is_gcs and os.path.isdir(local_checkpoint_dir):
    shutil.rmtree(local_checkpoint_dir)

  total_rows = sum(r["total_rows"] for r in results)
  print(f"\nDone! Wrote {total_rows:,} rows in {time.time()-t0:.1f}s ({total_rows/write_elapsed:,.0f} rows/sec)")


def main():
  args = parse_args()
  convert(args)


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn", force=True)
  main()
