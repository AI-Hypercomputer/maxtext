#!/usr/bin/env python3
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

"""Build an OLMo-style numpy mix index from a mix file.

Reads each ``.npy`` file's header (small range read from GCS, no download of
the array data) and writes a JSON index suitable for
``maxtext.input_pipeline.olmo_data.load_index``.

Usage:

    python tools/data_generation/build_olmo_npy_index.py \\
        --mix-file /home/.../OLMo-mix-0925-official.txt \\
        --gcs-base gs://my-bucket/dataset/ \\
        --tokenizer allenai/dolma3-tokenizer \\
        --sequence-length 8192 \\
        --output /tmp/olmo_index.json \\
        --workers 32

The mix file format is the same as AI2's OLMo-core data mix files:

    label,relative/path/{TOKENIZER}/...000000.npy

``{TOKENIZER}`` is substituted with the value of ``--tokenizer``.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Allow running from the repo root with src/ on the path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))

from maxtext.input_pipeline.olmo_data import (  # noqa: E402
    build_index,
    has_npy_magic,
    parse_npy_header,
    read_raw_metadata_from_path,
)
import numpy as np  # noqa: E402


def parse_mix_file(mix_path: str, tokenizer: str) -> List[Tuple[str, str]]:
  """Parse an OLMo data-mix .txt file. Returns list of (label, rel_path)."""
  entries: List[Tuple[str, str]] = []
  with open(mix_path, encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split(",", maxsplit=1)
      if len(parts) != 2:
        print(
            f"WARNING: {mix_path}:{line_num}: skipping malformed line: {line!r}",
            file=sys.stderr,
        )
        continue
      label, rel_path = parts
      entries.append((label, rel_path.replace("{TOKENIZER}", tokenizer)))
  return entries


def _split_gs_uri(uri: str) -> Tuple[str, str]:
  if not uri.startswith("gs://"):
    raise ValueError(f"Not a gs:// URI: {uri}")
  path = uri[len("gs://") :]
  bucket, _, key = path.partition("/")
  if not bucket or not key:
    raise ValueError(f"Malformed gs:// URI: {uri}")
  return bucket, key


def _read_gcs_prefix(client, uri: str, n_bytes: int = 4096) -> bytes:
  """Range-read the first ``n_bytes`` of a GCS object. ~1 GCS roundtrip."""
  bucket_name, blob_name = _split_gs_uri(uri)
  blob = client.bucket(bucket_name).blob(blob_name)
  # download_as_bytes(start=, end=) does a Range request. ``end`` is inclusive.
  return blob.download_as_bytes(start=0, end=n_bytes - 1)


def _gcs_blob_size(client, uri: str) -> int:
  """Return blob size in bytes (single metadata roundtrip, no data read)."""
  bucket_name, blob_name = _split_gs_uri(uri)
  blob = client.bucket(bucket_name).blob(blob_name)
  blob.reload()
  if blob.size is None:
    raise RuntimeError(f"GCS blob {uri} has unknown size")
  return int(blob.size)


def make_header_reader(dtype_for_raw: str):
  """Return a header_reader(path) -> (dtype, shape) accepting gs:// or local.

  Auto-detects whether each file is a real ``.npy`` (has the magic bytes and
  a parseable header) or AI2's headerless raw binary (the OLMo `.npy` files).

  In raw mode we trust ``dtype_for_raw`` and compute ``n_tokens`` from the
  blob size — no bytes need to be downloaded for the array data.
  """
  client_holder = {"client": None}

  def _get_client():
    if client_holder["client"] is None:
      # Lazy import: this script can run in dry-run / local-only mode without
      # google-cloud-storage installed.
      from google.cloud import storage  # pylint: disable=import-outside-toplevel

      client_holder["client"] = storage.Client()
    return client_holder["client"]

  def _gcs_reader(uri: str):
    client = _get_client()
    head = _read_gcs_prefix(client, uri, n_bytes=8)
    if has_npy_magic(head):
      # Real .npy — fetch enough bytes to cover the header.
      header_bytes = _read_gcs_prefix(client, uri, n_bytes=4096)
      return parse_npy_header(io.BytesIO(header_bytes))
    # Raw binary: derive shape from blob size + dtype itemsize.
    size = _gcs_blob_size(client, uri)
    itemsize = np.dtype(dtype_for_raw).itemsize
    if size % itemsize != 0:
      raise ValueError(
          f"GCS blob {uri} size {size} is not a multiple of dtype " f"{dtype_for_raw} itemsize ({itemsize})."
      )
    return dtype_for_raw, (size // itemsize,)

  def _local_reader(path: str):
    with open(path, "rb") as fh:
      head = fh.read(8)
    if has_npy_magic(head):
      return parse_npy_header(io.BytesIO(open(path, "rb").read()))
    return read_raw_metadata_from_path(path, dtype_for_raw)

  def _reader(path: str):
    if path.startswith("gs://"):
      return _gcs_reader(path)
    return _local_reader(path)

  return _reader


def _scan_one(reader, idx: int, label: str, path: str):
  dtype, shape = reader(path)
  return idx, label, path, dtype, shape


def scan_headers_parallel(
    paths_and_labels: List[Tuple[str, str]],
    *,
    dtype_for_raw: str,
    workers: int = 32,
    progress_every: int = 50,
) -> List[Tuple[str, str, str, Tuple[int, ...]]]:
  """Read .npy headers for all entries in parallel; preserve input order.

  Returns a list of (label, path, dtype, shape) tuples in the same order as
  the input.
  """
  reader = make_header_reader(dtype_for_raw=dtype_for_raw)
  results: List[Tuple[int, str, str, str, Tuple[int, ...]]] = [None] * len(paths_and_labels)  # type: ignore[list-item]
  start = time.time()
  with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(_scan_one, reader, i, label, path): i for i, (label, path) in enumerate(paths_and_labels)}
    done = 0
    for fut in as_completed(futures):
      idx, label, path, dtype, shape = fut.result()
      results[idx] = (idx, label, path, dtype, shape)
      done += 1
      if done % progress_every == 0 or done == len(paths_and_labels):
        elapsed = time.time() - start
        print(
            f"  scanned {done}/{len(paths_and_labels)} headers ({elapsed:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
  # Drop the index helper column.
  return [(label, path, dtype, shape) for (_, label, path, dtype, shape) in results]


def parse_args():
  """Parse CLI args for the index builder."""
  p = argparse.ArgumentParser(description="Build an OLMo-style numpy mix index by scanning .npy headers.")
  p.add_argument("--mix-file", required=True, help="Path to the mix .txt file.")
  p.add_argument(
      "--gcs-base",
      required=True,
      help=(
          "Base prefix for resolved file paths, e.g. gs://my-bucket/dataset/."
          " Mix-file relative paths are joined to this."
      ),
  )
  p.add_argument(
      "--tokenizer",
      default="allenai/dolma3-tokenizer",
      help="Substituted for {TOKENIZER} in mix paths. Also stored in the index.",
  )
  p.add_argument(
      "--sequence-length",
      type=int,
      required=True,
      help="Tokens per training instance (e.g. 8192).",
  )
  p.add_argument("--output", required=True, help="Output JSON path.")
  p.add_argument(
      "--dtype",
      default="uint32",
      help=(
          "Numpy dtype for files lacking a .npy header (the AI2 'pseudo-.npy'"
          " files are headerless uint32 streams). Default: uint32."
      ),
  )
  p.add_argument("--workers", type=int, default=32, help="Parallel header-scan threads.")
  return p.parse_args()


def main():
  args = parse_args()

  print(f"Parsing mix file: {args.mix_file}", file=sys.stderr)
  entries = parse_mix_file(args.mix_file, args.tokenizer)
  print(f"  {len(entries)} entries", file=sys.stderr)

  base = args.gcs_base.rstrip("/") + "/"
  resolved = [(label, base + rel.lstrip("/")) for label, rel in entries]

  print(
      f"Scanning {len(resolved)} .npy headers ({args.workers} threads, " f"raw dtype={args.dtype})...",
      file=sys.stderr,
  )
  headers = scan_headers_parallel(resolved, dtype_for_raw=args.dtype, workers=args.workers)

  # Cache the (dtype, shape) we already read so build_index doesn't re-scan.
  header_cache = {path: (dtype, shape) for (_, path, dtype, shape) in headers}

  def _cached_reader(path: str):
    return header_cache[path]

  paths_for_build = [(path, label) for (label, path, _, _) in headers]
  index = build_index(
      paths_for_build,
      sequence_length=args.sequence_length,
      tokenizer=args.tokenizer,
      header_reader=_cached_reader,
  )

  # Sanity: total token count for human inspection.
  total_t = index.total_tokens
  total_i = index.total_instances
  print(
      f"Total tokens: {total_t:,}  |  instances at SEQ={args.sequence_length}: "
      f"{total_i:,}  |  fingerprint: {index.fingerprint}",
      file=sys.stderr,
  )

  index.save(args.output)
  print(f"Wrote index to {args.output}", file=sys.stderr)


if __name__ == "__main__":
  main()
