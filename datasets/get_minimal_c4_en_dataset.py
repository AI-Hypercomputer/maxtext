#!/usr/bin/env python3
"""
This script outputs the minimal versions of the c4/en dataset to be able to decouple the 
maxtext repository from Google Cloud dependencies. Using this script, we are also pushing
a minimal version of the 3.0.1 version of the c4/en dataset. The minimal version that is 
generated has similar sharding with original dataset, and train and validation 
array_records.

Technical Details:
-Connects to MinIO
-Chooses the smallest shard per split (train/validation) for each c4/en version
-Range-downloads TFRecord shards to avoid large downloads
-Writes multiple ArrayRecord shards locally with strict per-shard byte caps (to keep files small)
-The script uses per-shard byte caps to ensure no output file exceeds your target. Tune MAX_OUTPUT_SHARD_BYTES for stricter limits.
-Range downloads are only used for TFRecord shards; ArrayRecord readers expect full files, so we always download the smallest full ArrayRecord shard if available.
-If you need even smaller totals, reduce EXACT_TRAIN_RECORDS and EXACT_VAL_RECORDS and/or shard counts.
-Uses round-robin distribution across shards to simulate real behavior

Note: Replace the MINIO_ACCESS_KEY and MINIO_SECRET_KEY with your keys.
"""

import os
import glob
import sys
import argparse
from typing import List, Tuple

from minio import Minio
from minio.error import S3Error

# ArrayRecord Python bindings
from array_record.python.array_record_module import ArrayRecordWriter, ArrayRecordReader
import tensorflow as tf


# -----------------------
# Configurable parameters
# -----------------------

# ------------ MinIO Connection Config (override via env) ------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio-frameworks.amd.com")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "hidden")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "hidden")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() == "true"
BUCKET = os.environ.get("MINIO_C4_BUCKET", "datasets.dl")

# Versions of c4/en to sample
#VERSIONS = ["3.0.1", "3.0.5", "3.0.7", "3.0.8", "3.0.9"]
VERSIONS = ["3.0.1"]

# Local output base
LOCAL_BASE = "datasets/c4_en_dataset_minimal/c4/en"

# Shard counts (simulate real behavior)
NUM_SHARDS_TRAIN = 8
NUM_SHARDS_VAL = 2

# Record caps: adjust to control total content size
EXACT_TRAIN_RECORDS = 1000
EXACT_VAL_RECORDS = 200

# Per-output-shard hard cap (bytes) so each file stays under target size
# Adjust as needed; 20 MiB per shard keeps total per version < 50MB.
MAX_OUTPUT_SHARD_BYTES = 20 * 1024 * 1024  # 20 MiB per shard

# Per-version soft cap (for info/warning)
MAX_VERSION_BYTES = 50 * 1024 * 1024  # 50 MiB

# Temp download cap for TFRecord range GET (no need download the full shard)
MAX_TEMP_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MiB

# Prefixes in MinIO
ARRAY_RECORD_TRAIN_PREFIX = "c4/en/{ver}/c4-train.array_record-"
ARRAY_RECORD_VAL_PREFIX   = "c4/en/{ver}/c4-validation.array_record-"
TFRECORD_TRAIN_PREFIX     = "c4/en/{ver}/c4-train.tfrecord-"
TFRECORD_VAL_PREFIX       = "c4/en/{ver}/c4-validation.tfrecord-"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_matching(client: Minio, bucket: str, prefix: str) -> List:
    """Return a sorted list of objects under prefix."""
    return sorted(
        (obj for obj in client.list_objects(bucket, prefix=prefix, recursive=True)),
        key=lambda o: o.object_name
    )

def pick_smallest(objects):
    """Pick the smallest object by size; falls back gracefully if size is unavailable."""
    return min(objects, key=lambda o: getattr(o, "size", float("inf")))

def download_shard_with_optional_range(client, bucket, obj, tmp_dir, allow_range=False, max_bytes=MAX_TEMP_DOWNLOAD_BYTES):
    """
    Download shard to a temp file. If allow_range is True and obj.size > max_bytes,
    download only the first max_bytes bytes (safe for TFRecord; not for ArrayRecord).
    """
    ensure_dir(tmp_dir)
    local_tmp = os.path.join(tmp_dir, os.path.basename(obj.object_name))

    if allow_range and getattr(obj, "size", None) and obj.size > max_bytes:
        # Range GET: read first max_bytes bytes for TFRecord; iterator stops at incomplete record
        response = client.get_object(bucket, obj.object_name, offset=0, length=max_bytes)
        try:
            with open(local_tmp, "wb") as f:
                for d in response.stream(32 * 1024):
                    f.write(d)
        finally:
            response.close()
            response.release_conn()
    else:
        client.fget_object(bucket, obj.object_name, local_tmp)
    return local_tmp

def write_sharded_with_byte_caps_from_arrayrecord(src_path, dst_dir, split_name,
                                                  num_shards, max_total_records, max_shard_bytes):
    """
    Read ArrayRecord src and write into multiple ArrayRecord shards with per-shard byte caps.
    Round-robin distribution; stops if shard would exceed cap.
    """
    ensure_dir(dst_dir)
    writers = []
    shard_bytes = [0] * num_shards
    for i in range(num_shards):
        shard_name = f"c4-{split_name}.array_record-{i:05d}-of-{num_shards:05d}"
        writers.append(ArrayRecordWriter(os.path.join(dst_dir, shard_name), "group_size:1"))

    reader = ArrayRecordReader(src_path)
    n = min(max_total_records, reader.num_records())
    shard_idx = 0
    written = 0
    for i in range(n):
        rec = reader.read(i)
        rec_len = len(rec)
        # If current shard would exceed cap, move to next shard
        if shard_bytes[shard_idx] + rec_len > max_shard_bytes:
            shard_idx = (shard_idx + 1) % num_shards
            # If next shard is also full, stop early
            if shard_bytes[shard_idx] + rec_len > max_shard_bytes:
                break
        writers[shard_idx].write(rec)
        shard_bytes[shard_idx] += rec_len
        written += 1
        shard_idx = (shard_idx + 1) % num_shards

    for w in writers:
        w.close()

    print(f"[{split_name}] Wrote {written} records across {num_shards} shards; "
          f"per-shard sizes: {[round(b/1024/1024, 2) for b in shard_bytes]} MiB")
    return written, shard_bytes

def write_sharded_with_byte_caps_from_tfrecord(src_path, dst_dir, split_name,
                                               num_shards, max_total_records, max_shard_bytes):
    """
    Read TFRecord src and write into multiple ArrayRecord shards with per-shard byte caps.
    """
    ensure_dir(dst_dir)
    writers = []
    shard_bytes = [0] * num_shards
    for i in range(num_shards):
        shard_name = f"c4-{split_name}.array_record-{i:05d}-of-{num_shards:05d}"
        writers.append(ArrayRecordWriter(os.path.join(dst_dir, shard_name), "group_size:1"))

    shard_idx = 0
    count = 0
    for raw_example in tf.compat.v1.io.tf_record_iterator(src_path):
        rec_len = len(raw_example)
        if shard_bytes[shard_idx] + rec_len > max_shard_bytes:
            shard_idx = (shard_idx + 1) % num_shards
            if shard_bytes[shard_idx] + rec_len > max_shard_bytes:
                break
        writers[shard_idx].write(raw_example)
        shard_bytes[shard_idx] += rec_len
        count += 1
        shard_idx = (shard_idx + 1) % num_shards
        if count >= max_total_records:
            break

    for w in writers:
        w.close()

    print(f"[{split_name}] Wrote {count} records across {num_shards} shards; "
          f"per-shard sizes: {[round(b/1024/1024, 2) for b in shard_bytes]} MiB")
    return count, shard_bytes

def compute_dir_size_bytes(dir_path: str, patterns: List[str]) -> int:
    """Sum file sizes for all files matching provided glob patterns in dir_path."""
    total = 0
    for patt in patterns:
        for p in glob.glob(os.path.join(dir_path, patt)):
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create minimal c4/en dataset shards from MinIO Instance"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite all existing dataset files without prompting"
    )
    args = parser.parse_args()

    # Use TF v1-style record iterator
    tf.compat.v1.disable_eager_execution()

    # Check which versions already exist
    existing_versions = []
    for ver in VERSIONS:
        local_version_dir = os.path.join(LOCAL_BASE, ver)
        # Check if directory exists and has ArrayRecord files
        if os.path.exists(local_version_dir):
            shard_files = glob.glob(os.path.join(local_version_dir, "c4-*.array_record-*"))
            if shard_files:
                existing_versions.append(ver)

    if existing_versions:
        if args.force:
            print(f"Force mode: Will overwrite existing versions: {existing_versions}")
        else:
            print(f"Found existing versions: {existing_versions}")
            # Check if all versions exist to avoid MinIO connection
            if set(existing_versions) == set(VERSIONS):
                print("All versions already exist. Nothing to do.")
                print("Use --force to regenerate all versions.")
                sys.exit(0)
            print(f"Will skip these and only generate missing versions.")
            print(f"Use --force to overwrite all versions.\n")

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

    if not client.bucket_exists(BUCKET):
        print(f"Bucket '{BUCKET}' does not exist.")
        sys.exit(1)

    print("Bucket exists. Starting minimal dataset creation...")
    ensure_dir(LOCAL_BASE)


    print("Listing c4/en top-level entries (non-recursive):")
    for obj in client.list_objects(BUCKET, prefix="c4/en", recursive=False):
        print("-", obj.object_name)

    print("\nListing recursively (first 200 entries):")
    i = 0
    for obj in client.list_objects(BUCKET, prefix="c4/en", recursive=True):
        print(obj.object_name)
        i += 1
        if i >= 200: break

    for ver in VERSIONS:
        local_version_dir = os.path.join(LOCAL_BASE, ver)
        
        # Skip existing versions unless force mode is enabled
        if not args.force and ver in existing_versions:
            print(f"\nSkipping existing version {ver}")
            continue
            
        print(f"\nProcessing c4/en:{ver}")
        ensure_dir(local_version_dir)

        # Find source shards for train/validation (prefer ArrayRecord)
        train_src_objs = list_matching(client, BUCKET, ARRAY_RECORD_TRAIN_PREFIX.format(ver=ver))
        val_src_objs   = list_matching(client, BUCKET, ARRAY_RECORD_VAL_PREFIX.format(ver=ver))

        train_is_arrayrec = True
        val_is_arrayrec = True

        if not train_src_objs:
            train_src_objs = list_matching(client, BUCKET, TFRECORD_TRAIN_PREFIX.format(ver=ver))
            train_is_arrayrec = False
        if not val_src_objs:
            val_src_objs = list_matching(client, BUCKET, TFRECORD_VAL_PREFIX.format(ver=ver))
            val_is_arrayrec = False

        if not train_src_objs:
            print(f"Warning: No train shards found for {ver}. Skipping this version.")
            continue
        if not val_src_objs:
            print(f"Warning: No validation shards found for {ver}. Skipping validation for this version.")
            continue

        # Pick the smallest shard per split to minimize download
        smallest_train = pick_smallest(train_src_objs)
        smallest_val   = pick_smallest(val_src_objs)

        # Download one shard per split to a temp folder
        tmp_dir = os.path.join(local_version_dir, "_tmp_download")
        try:
            train_src_local = download_shard_with_optional_range(
                client, BUCKET, smallest_train, tmp_dir,
                allow_range=(not train_is_arrayrec),  # only range for TFRecord
                max_bytes=MAX_TEMP_DOWNLOAD_BYTES
            )
            val_src_local = download_shard_with_optional_range(
                client, BUCKET, smallest_val, tmp_dir,
                allow_range=(not val_is_arrayrec),
                max_bytes=MAX_TEMP_DOWNLOAD_BYTES
            )
        except S3Error as e:
            print(f"Download error for version {ver}: {e}")
            # Clean up and skip
            try:
                if os.path.exists(tmp_dir):
                    for f in os.listdir(tmp_dir):
                        os.remove(os.path.join(tmp_dir, f))
                    os.rmdir(tmp_dir)
            except Exception:
                pass
            continue

        # Write minimal multi-shard ArrayRecord files with per-shard caps
        try:
            if train_is_arrayrec:
                write_sharded_with_byte_caps_from_arrayrecord(
                    train_src_local, local_version_dir, "train",
                    NUM_SHARDS_TRAIN, EXACT_TRAIN_RECORDS, MAX_OUTPUT_SHARD_BYTES
                )
            else:
                write_sharded_with_byte_caps_from_tfrecord(
                    train_src_local, local_version_dir, "train",
                    NUM_SHARDS_TRAIN, EXACT_TRAIN_RECORDS, MAX_OUTPUT_SHARD_BYTES
                )

            if val_is_arrayrec:
                write_sharded_with_byte_caps_from_arrayrecord(
                    val_src_local, local_version_dir, "validation",
                    NUM_SHARDS_VAL, EXACT_VAL_RECORDS, MAX_OUTPUT_SHARD_BYTES
                )
            else:
                write_sharded_with_byte_caps_from_tfrecord(
                    val_src_local, local_version_dir, "validation",
                    NUM_SHARDS_VAL, EXACT_VAL_RECORDS, MAX_OUTPUT_SHARD_BYTES
                )

            # Post-write size check
            total_bytes = compute_dir_size_bytes(
                local_version_dir,
                patterns=["c4-train.array_record-*", "c4-validation.array_record-*"]
            )
            mb = total_bytes / (1024 * 1024)
            print(f"Total size for {ver}: {mb:.2f} MiB")
            if total_bytes > MAX_VERSION_BYTES:
                print(f"Note: {ver} exceeds {MAX_VERSION_BYTES/(1024*1024):.0f} MiB. "
                      f"Consider reducing records or MAX_OUTPUT_SHARD_BYTES.")
        finally:
            # Clean up temp downloads
            try:
                if os.path.exists(train_src_local):
                    os.remove(train_src_local)
                if os.path.exists(val_src_local):
                    os.remove(val_src_local)
                if os.path.isdir(tmp_dir):
                    for f in os.listdir(tmp_dir):
                        os.remove(os.path.join(tmp_dir, f))
                    os.rmdir(tmp_dir)
            except Exception as cleanup_err:
                print(f"Cleanup warning: {cleanup_err}")

    print("\nDone. Verify local directories:")
    for ver in VERSIONS:
        print(f"- {os.path.join(LOCAL_BASE, ver)}")
        for p in sorted(glob.glob(os.path.join(LOCAL_BASE, ver, "c4-*.array_record-*"))):
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()

