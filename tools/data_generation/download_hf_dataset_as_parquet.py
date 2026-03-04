#!/usr/bin/env python3
"""
Download a HuggingFace dataset via streaming and save as Parquet files.
Designed for datasets stored in HuggingFace Hub with a non-Parquet format, such as JSONL, TXT.
For datasets that are already in parquet format, use HuggingFace download instructions in:
https://huggingface.co/docs/hub/en/datasets-downloading
 
This script supports writing to local disk or GCS (gs:// paths).
Produces HuggingFace-style filenames: {name}-XXXXX-of-YYYYY.parquet

The script streams the dataset so it never loads the full dataset into memory.
Row group sizes are automatically estimated by sampling to hit a target
uncompressed size (default 200 MB), following HuggingFace's recommendation
of 100-300 MB per row group.

Usage examples:
    # Basic: auto-detect row group size, write 2048 files to GCS
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --output gs://my-bucket/datasets/climbmix/

    # Custom split, compression, and row group target
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --split train \
        --num-files 2048 \
        --output ./output/ \
        --compression snappy \
        --target-row-group-mb 200

    # Skip counting pass (provide total if you already know it)
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --output gs://my-bucket/data/ \
        --total-examples 123456789

    # Skip both estimation and counting (provide row group size and total)
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --output gs://my-bucket/data/ \
        --row-group-size 50000 \
        --total-examples 123456789

    # Specify a subset/config
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --config default \
        --num-files 2048 \
        --output ./output/

    # Parallel streaming with 8 workers (~8x faster for large datasets).
    # Strongly recommended to pass --total-examples and --row-group-size to
    # avoid two extra sequential pre-passes before the parallel work starts.
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --output gs://my-bucket/datasets/climbmix/ \
        --num-workers 8 \
        --total-examples 123456789 \
        --row-group-size 50000

For large datasets, the conversion can take hours or even days. It is
recommended to run the script inside a tmux or screen session so it
survives disconnections:

    # Using tmux
    tmux new -s parquet-convert
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --output gs://my-bucket/datasets/climbmix/
    # Detach with Ctrl-b d, reattach later with: tmux attach -t parquet-convert

    # Using screen
    screen -L -S parquet-convert
    python download_hf_dataset_as_parquet.py \
        --dataset OptimalScale/ClimbMix \
        --num-files 2048 \
        --output gs://my-bucket/datasets/climbmix/
    # Detach with Ctrl-a d, reattach later with: screen -r parquet-convert
"""

import argparse
import math
import multiprocessing
import os
import time
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def parse_args():
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
      help="Dataset config/subset name (default: None)",
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
      help="Exact number of Parquet files to produce",
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
      "--target-row-group-mb",
      type=int,
      default=200,
      help="Target uncompressed row group size in MB (default: 200)",
  )
  parser.add_argument(
      "--sample-size",
      type=int,
      default=1000,
      help="Number of examples to sample for row size estimation (default: 1000)",
  )
  parser.add_argument(
      "--row-group-size",
      type=int,
      default=None,
      help="Exact number of rows per row group (skip sampling estimation if provided)",
  )
  parser.add_argument(
      "--total-examples",
      type=int,
      default=None,
      help="Total number of examples (skip counting pass if provided)",
  )
  parser.add_argument(
      "--token",
      default=None,
      help="HuggingFace auth token for gated datasets",
  )
  parser.add_argument(
      "--num-workers",
      type=int,
      default=1,
      help=(
          "Number of parallel worker processes (default: 1). "
          "Each worker streams an independent shard via IterableDataset.shard(). "
          "num-workers must not exceed the dataset's number of source shards "
          "(check dataset.n_shards). Pass --total-examples and --row-group-size "
          "to skip the two sequential pre-passes and go straight to parallel work."
      ),
  )
  return parser.parse_args()


def get_filesystem_and_path(output_path: str):
  """Return (pyarrow filesystem, base_path) for local or GCS paths."""
  if output_path.startswith("gs://"):
    from pyarrow.fs import GcsFileSystem

    fs = GcsFileSystem()
    base_path = output_path.replace("gs://", "")
  else:
    from pyarrow.fs import LocalFileSystem

    fs = LocalFileSystem()
    base_path = os.path.abspath(output_path)
    os.makedirs(base_path, exist_ok=True)

  if not base_path.endswith("/"):
    base_path += "/"

  return fs, base_path


def load_streaming_dataset(args):
  """Load the dataset in streaming mode."""
  kwargs = dict(
      path=args.dataset,
      streaming=True,
      split=args.split,
      token=args.token,
  )
  if args.config:
    kwargs["name"] = args.config
  return load_dataset(**kwargs)


def estimate_row_group_size(ds, sample_size: int, target_mb: int):
  """Sample rows to estimate how many rows fit in target_mb uncompressed."""
  print(f"Sampling {sample_size} examples to estimate row sizes...")
  sample = list(islice(ds, sample_size))
  table = pa.Table.from_pylist(sample)
  total_bytes = table.nbytes
  avg_bytes = total_bytes / len(sample)

  print(f"  Sample Arrow size: {total_bytes / 1e6:.2f} MB for {len(sample)} rows")
  print(f"  Avg row size: {avg_bytes:.0f} bytes")

  row_group_rows = int(target_mb * 1e6 / avg_bytes)
  row_group_rows = max(row_group_rows, 1)
  print(f"  Row group size for ~{target_mb} MB: {row_group_rows:,} rows")

  return row_group_rows


def count_examples(args):
  """Count total examples by streaming through the dataset."""
  print("Counting total examples (this may take a while)...")
  ds = load_streaming_dataset(args)
  total = 0
  t0 = time.time()
  for _ in ds:
    total += 1
    if total % 1_000_000 == 0:
      elapsed = time.time() - t0
      rate = total / elapsed
      print(f"  Counted {total:,} rows ({rate:,.0f} rows/sec)...")
  elapsed = time.time() - t0
  print(f"  Total: {total:,} examples in {elapsed:.1f}s")
  return total


def make_filename(prefix: str, file_idx: int, num_files: int) -> str:
  """Generate HuggingFace-style filename: prefix-00000-of-02048.parquet"""
  width = len(str(num_files))
  return f"{prefix}-{file_idx:0{width}d}-of-{num_files:0{width}d}.parquet"


def _worker(
    args,
    worker_idx: int,
    num_workers: int,
    file_start: int,
    file_end: int,
    rows_per_file: int,
    row_group_size: int,
    prefix: str,
):
  """
  Worker entry point: streams shard `worker_idx` of `num_workers` and writes
  output files in the range [file_start, file_end).
  """
  tag = f"[worker {worker_idx}]"
  fs, base_path = get_filesystem_and_path(args.output)
  compression = None if args.compression == "none" else args.compression

  ds = load_streaming_dataset(args)
  if num_workers > 1:
    ds = ds.shard(num_shards=num_workers, index=worker_idx)

  num_files_total = args.num_files
  schema = None
  writer = None
  file_idx = file_start
  rows_in_file = 0
  total_written = 0
  batch = []
  t0 = time.time()

  def open_writer(idx):
    nonlocal writer
    filename = make_filename(prefix, idx, num_files_total)
    filepath = base_path + filename
    writer = pq.ParquetWriter(
        filepath,
        schema,
        filesystem=fs,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    )
    print(f"{tag} Opening file {idx}: {filename}", flush=True)
    return writer

  for example in ds:
    batch.append(example)

    last_file = file_idx == file_end - 1
    flush_for_row_group = len(batch) >= row_group_size
    flush_for_file_boundary = not last_file and (rows_in_file + len(batch) >= rows_per_file)

    if flush_for_row_group or flush_for_file_boundary:
      table = pa.Table.from_pylist(batch)
      if schema is None:
        schema = table.schema
      if writer is None:
        open_writer(file_idx)

      writer.write_table(table)
      rows_in_file += len(batch)
      total_written += len(batch)
      batch = []

      # Rotate to next file when we've hit the per-file row target
      if rows_in_file >= rows_per_file and file_idx < file_end - 1:
        writer.close()
        writer = None
        file_idx += 1
        rows_in_file = 0

        if (file_idx - file_start) % 100 == 0:
          elapsed = time.time() - t0
          rate = total_written / elapsed
          print(
              f"{tag} file {file_idx - file_start}/{file_end - file_start} " f"({rate:,.0f} rows/sec)",
              flush=True,
          )

  # Flush the remaining batch into the last file
  if batch:
    table = pa.Table.from_pylist(batch)
    if schema is None:
      schema = table.schema
    if writer is None:
      open_writer(file_idx)
    writer.write_table(table)
    total_written += len(batch)

  if writer:
    writer.close()

  elapsed = time.time() - t0
  files_written = file_idx - file_start + 1
  print(
      f"{tag} Done. {total_written:,} rows -> {files_written} files in {elapsed:.1f}s",
      flush=True,
  )
  return total_written, files_written


def convert(args):
  compression = None if args.compression == "none" else args.compression

  prefix = args.name_prefix
  if prefix is None:
    prefix = args.dataset.split("/")[-1].lower().replace(" ", "-")

  # --- Step 1: Determine row group size ---
  if args.row_group_size is not None:
    row_group_size = args.row_group_size
    print(f"Using provided row group size: {row_group_size:,} rows")
  else:
    print(f"Estimating row group size (target {args.target_row_group_mb} MB uncompressed)...")
    ds = load_streaming_dataset(args)
    row_group_size = estimate_row_group_size(ds, args.sample_size, args.target_row_group_mb)
    print(f"Estimated row group size: {row_group_size:,} rows")

  # --- Step 2: Count total examples ---
  if args.total_examples is not None:
    total = args.total_examples
    print(f"Using provided total: {total:,} examples")
  else:
    total = count_examples(args)

  rows_per_file = math.ceil(total / args.num_files)

  # --- Step 3: Validate and assign file ranges to workers ---
  num_workers = args.num_workers

  # Warn if num_workers exceeds dataset source shards (some workers would get nothing)
  ds_check = load_streaming_dataset(args)
  n_shards = getattr(ds_check, "n_shards", None)
  if n_shards is not None and num_workers > n_shards:
    print(
        f"Warning: --num-workers {num_workers} exceeds the dataset's source shard "
        f"count ({n_shards}). Capping workers to {n_shards}."
    )
    num_workers = n_shards

  files_per_worker = math.ceil(args.num_files / num_workers)
  worker_ranges = []
  for w in range(num_workers):
    file_start = w * files_per_worker
    file_end = min(file_start + files_per_worker, args.num_files)
    if file_start >= args.num_files:
      break
    worker_ranges.append((file_start, file_end))
  actual_workers = len(worker_ranges)

  print(
      f"\nPlan: {total:,} examples -> {args.num_files} files "
      f"(~{rows_per_file:,} rows/file), {actual_workers} worker(s)"
  )
  print(f"Compression: {args.compression}")
  print(f"Output: {args.output}\n")

  worker_args = [
      (args, w, actual_workers, file_start, file_end, rows_per_file, row_group_size, prefix)
      for w, (file_start, file_end) in enumerate(worker_ranges)
  ]

  t0 = time.time()

  if actual_workers == 1:
    total_written, files_written = _worker(*worker_args[0])
  else:
    with multiprocessing.Pool(processes=actual_workers) as pool:
      results = pool.starmap(_worker, worker_args)
    total_written = sum(r[0] for r in results)
    files_written = sum(r[1] for r in results)

  elapsed = time.time() - t0
  print(f"\nDone! Wrote {total_written:,} rows to {files_written} files in {elapsed:.1f}s")

  if files_written < args.num_files:
    print(
        f"Warning: Only {files_written} files were created (target: {args.num_files}). "
        f"The dataset may have fewer examples than expected."
    )


def main():
  args = parse_args()
  convert(args)


if __name__ == "__main__":
  multiprocessing.set_start_method("spawn", force=True)
  main()
