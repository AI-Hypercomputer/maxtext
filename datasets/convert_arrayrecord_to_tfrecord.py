#!/usr/bin/env python3
"""Convert minimal C4 ArrayRecord shards to TFRecord shards with TFDS-compatible naming.

This script scans a version directory containing files like:
    c4-train.array_record-00000-of-00008
    c4-validation.array_record-00000-of-00002
and produces TFRecord files named:
    __local_c4_builder-train.tfrecord-00000-of-00008
    __local_c4_builder-validation.tfrecord-00000-of-00002

These TFRecord files allow TFDS to iterate using standard TFRecordDataset logic
while preserving the shard structure expected by your metadata (8 train, 2 validation).

Usage:
    python convert_arrayrecord_to_tfrecord.py --version-dir c4_en_dataset_minimal/c4/en/3.0.1 --builder-name __local_c4_builder --force

Options:
    --dry-run           Only show planned conversions.
    --force             Overwrite existing TFRecord output files.

Dependencies:
    array_record (Python bindings)
    tensorflow

Limitations:
    Records are copied verbatim; compression is not applied.
"""
from __future__ import annotations
import os, argparse, glob, sys
from typing import List

try:
    from array_record.python.array_record_module import ArrayRecordReader
except ModuleNotFoundError as e:
    print("Error: array_record module not found. Install appropriate package before running.")
    sys.exit(1)

import tensorflow as tf

def discover_shards(version_dir: str, split: str) -> List[str]:
    pattern = os.path.join(version_dir, f"c4-{split}.array_record-*")
    return sorted(glob.glob(pattern))

def parse_shard_numbers(fname: str):
    # fname example: c4-train.array_record-00003-of-00008
    base = os.path.basename(fname)
    parts = base.split('-')
    # last two parts are shard index and total, e.g. 00003, of, 00008
    shard_idx = parts[-3]
    total = parts[-1]
    return shard_idx, total

def convert_shard(arrayrecord_path: str, output_path: str, force: bool):
    if os.path.exists(output_path) and not force:
        print(f"Skip existing: {output_path}")
        return
    reader = ArrayRecordReader(arrayrecord_path)
    count = reader.num_records()
    written = 0
    batch_size = 1024  # iterate in batches for efficiency
    with tf.io.TFRecordWriter(output_path) as writer:
        start = 0
        while start < count:
            end = min(start + batch_size, count)
            # reader.read(start, end) returns list of records in [start,end)
            batch = reader.read(start, end)
            for rec in batch:
                writer.write(rec)
                written += 1
            start = end
    print(f"Converted {arrayrecord_path} -> {output_path} ({written} / {count} records)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version-dir', required=True, help='Directory like c4_en_dataset_minimal/c4/en/3.0.1')
    ap.add_argument('--builder-name', default='__local_c4_builder', help='Prefix used for TFRecord shard filenames.')
    ap.add_argument('--dry-run', action='store_true', help='Only list planned conversions.')
    ap.add_argument('--force', action='store_true', help='Overwrite existing TFRecord shards if present.')
    args = ap.parse_args()

    if not os.path.isdir(args.version_dir):
        print(f"Version directory not found: {args.version_dir}")
        sys.exit(1)

    for split in ['train', 'validation']:
        shards = discover_shards(args.version_dir, split)
        if not shards:
            print(f"No ArrayRecord shards found for split '{split}' in {args.version_dir}")
            continue
        print(f"Found {len(shards)} {split} ArrayRecord shards.")
        for shard in shards:
            shard_idx, total = parse_shard_numbers(shard)
            tfrec_name = f"{args.builder_name}-{split}.tfrecord-{shard_idx}-of-{total}"
            out_path = os.path.join(args.version_dir, tfrec_name)
            if args.dry_run:
                print(f"Would create: {out_path} from {shard}")
            else:
                convert_shard(shard, out_path, args.force)

    print("Done.")

if __name__ == '__main__':
    main()

