#!/usr/bin/env python3
"""Minimal C4/en TFRecord -> Parquet converter.

Fetch the first train & validation TFRecord 00000-of shard for a version and
sample rows into two tiny parquet files with fixed output names for the usage
in tests/grain_data_processing_test.py, tests/hf_data_processing_test.py,
tests/train_tests.py:
    c4-train-00000-of-01637.parquet
    c4-validation-00000-of-01637.parquet
"""

import argparse
import os
from pathlib import Path

from minio import Minio
import pyarrow as pa
import pyarrow.parquet as pq

import tensorflow as tf

# ---------------- Environment / Defaults ----------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio-frameworks.amd.com")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "hidden")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "hidden")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() == "true"
BUCKET = os.environ.get("MINIO_C4_BUCKET", "datasets.dl")
SCRIPT_DIR = Path(__file__).parent

def download_object(client: Minio, obj, dest_path: Path):
    data = client.get_object(BUCKET, obj.object_name)
    try:
        with dest_path.open("wb") as f:
            for chunk in data.stream(128 * 1024):
                f.write(chunk)
    finally:
        data.close(); data.release_conn()
    return dest_path

def write_parquet(path: Path, rows: list[str], force: bool = False):
    if not force and path.exists():
        print(f"[skip] {path} exists")
        return
    # If force is set and path exists, remove it first
    if force and path.exists():
        path.unlink()
    # Normalize & drop empties again defensively.
    rows = [r.strip() for r in rows if isinstance(r, str) and r.strip()]
    table = pa.Table.from_pydict({"text": rows})
    pq.write_table(table, path, compression="ZSTD")
    print(f"[write] {path} rows={len(rows)} size_kib={path.stat().st_size/1024:.1f}")

def sample_tfrecord(path: Path, cap: int) -> list[str]:
    """Sample up to `cap` records from a TFRecord, extracting only the 'text' feature.
    Assumes each record is a serialized tf.train.Example with a bytes feature named 'text'.
    Fails fast (raises) if parsing or feature access fails.
    """
    feature_spec = {"text": tf.io.FixedLenFeature([], tf.string)}
    rows: list[str] = []
    for raw in tf.data.TFRecordDataset(str(path)).take(cap):
        parsed = tf.io.parse_single_example(raw, feature_spec)
        txt = parsed["text"].numpy().decode("utf-8", "ignore").strip()
        if txt:
            rows.append(txt)
    return rows


def main():
    p = argparse.ArgumentParser(description="Minimal C4 TFRecord -> parquet generator")
    p.add_argument("--version", default="3.0.1")
    p.add_argument("--train-rows", type=int, default=800)
    p.add_argument("--val-rows", type=int, default=160)
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "c4_en_dataset_minimal" / "hf" / "c4"))
    p.add_argument("--force", action="store_true", default=False,
                   help="Force overwrite existing parquet files")
    a = p.parse_args()

    # Resolve output paths first so we can early stop.
    out_dir = Path(a.output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "c4-train-00000-of-01637.parquet"
    val_out = out_dir / "c4-validation-00000-of-01637.parquet"
    
    if not a.force and train_out.exists() and val_out.exists():
        print("Both output parquet files already exist; skipping (no MinIO connection needed).")
        print("Use --force to regenerate the files.")
        return

    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
    if not client.bucket_exists(BUCKET):
        print("Bucket missing; abort.")
        return

    ver = a.version
    # List to find the first train 00000-of shard.
    train_prefix = f"c4/en/{ver}/c4-train.tfrecord-00000-of-"
    train_obj = None
    for o in client.list_objects(BUCKET, prefix=train_prefix, recursive=False):
        train_obj = o
        break
    if not train_obj:
        print("Train 00000-of shard not found; abort.")
        return

    val_prefix = f"c4/en/{ver}/c4-validation.tfrecord-00000-of-"
    val_obj = None
    for o in client.list_objects(BUCKET, prefix=val_prefix, recursive=False):
        val_obj = o
        break
    if not val_obj:
        print("Validation 00000-of shard not found; abort.")
        return
    print(f"Using train object {train_obj.object_name} and validation object {val_obj.object_name}.")

    rows_train: list[str]
    rows_val: list[str]

    tmp_train = out_dir.parent / "_tmp_train"
    download_object(client, train_obj, tmp_train)
    rows_train = sample_tfrecord(tmp_train, a.train_rows)
    try: tmp_train.unlink()
    except Exception: pass

    tmp_val = out_dir.parent / "_tmp_val"
    download_object(client, val_obj, tmp_val)
    rows_val = sample_tfrecord(tmp_val, a.val_rows)
    try: tmp_val.unlink()
    except Exception: pass

    print(f"Rows: train={len(rows_train)} val={len(rows_val)}")
    write_parquet(train_out, rows_train, force=a.force)
    write_parquet(val_out, rows_val, force=a.force)

if __name__ == "__main__":
    main()

