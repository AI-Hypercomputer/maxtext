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

"""Download OLMo .npy dataset files from HTTP to GCS with HTTP-Range resume.

Each HTTP transfer is staged to local disk and resumes via the HTTP
``Range: bytes=N-`` header on any RequestException — a 240 GB file whose
upstream connection drops at 140 GB resumes from there instead of
restarting at 0. After the local file is complete (and matches
Content-Length), it's uploaded to GCS in one shot, verified against
``blob.size``, and the local copy removed.

Local disk usage at peak is bounded by ``--workers * largest_file_size``.

Usage:
    python download_olmo_data_to_gcs.py \\
        --mix-file /path/to/OLMo-mix-0925-official.txt \\
        --gcs-dest gs://my-bucket/olmo/ \\
        --staging-dir /mnt/local-ssd/olmo-staging \\
        --workers 2

Skip-existing is on by default, so re-running against the full mix file
finishes only the entries missing from GCS. A partial local stage file is
detected via os.path.getsize() and resumed.
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests
from requests.exceptions import RequestException
from google.cloud import storage

from maxtext.utils.gcs_utils import gcs_path_exists, parse_gcs_bucket_and_prefix


def parse_mix_file(mix_path: str, tokenizer: str) -> list[tuple[str, str]]:
  """Parse an OLMo data-mix .txt file. Returns list of (label, rel_path)."""
  entries = []
  with open(mix_path, encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split(",", maxsplit=1)
      if len(parts) != 2:
        print(f"WARNING: {mix_path}:{line_num}: skipping malformed line: {line!r}")
        continue
      label, rel_path = parts
      entries.append((label, rel_path.replace("{TOKENIZER}", tokenizer)))
  return entries


def _total_size_from_response(resp, bytes_have):
  """Extract total file size from a response, preferring Content-Range."""
  cr = resp.headers.get("Content-Range")
  if cr and "/" in cr:
    tail = cr.split("/")[-1].strip()
    if tail.isdigit():
      return int(tail)
  cl = resp.headers.get("Content-Length")
  if cl is not None:
    # 200 OK ⇒ Content-Length is the full size; 206 ⇒ remaining size.
    return int(cl) + (bytes_have if resp.status_code == 206 else 0)
  return None


def http_resumable_to_local(
    url: str,
    local_path: str,
    max_retries: int = 20,
    chunk_size: int = 8 * 1024 * 1024,
    timeout: int = 300,
    progress_every_s: float = 30.0,
) -> tuple[int, int | None]:
  """Download ``url`` to ``local_path`` with Range-based resume.

  Returns (bytes_written, total_size). total_size may be None if the server
  refuses to disclose it on every attempt.
  """
  os.makedirs(os.path.dirname(local_path), exist_ok=True)
  session = requests.Session()
  total_size = None
  bytes_have = os.path.getsize(local_path) if os.path.exists(local_path) else 0
  if bytes_have:
    print(f"    resuming {os.path.basename(local_path)} from {bytes_have / 1e9:.2f} GB", flush=True)

  for attempt in range(1, max_retries + 1):
    try:
      headers = {}
      if bytes_have > 0:
        headers["Range"] = f"bytes={bytes_have}-"

      with session.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        # If we asked for a Range and the server returned 200 (full body),
        # discard our partial file — server doesn't honor ranges and is
        # restarting from byte 0.
        if bytes_have > 0 and resp.status_code == 200:
          print(
              f"    server returned 200 to Range request; restarting " f"{os.path.basename(local_path)} from 0",
              flush=True,
          )
          bytes_have = 0
          # Truncate; we'll reopen "wb" below.
          if os.path.exists(local_path):
            os.remove(local_path)

        new_total = _total_size_from_response(resp, bytes_have)
        if new_total is not None:
          if total_size is None:
            total_size = new_total
          elif new_total != total_size:
            print(
                f"    WARNING: total size changed mid-download " f"({total_size} → {new_total}); using new value",
                flush=True,
            )
            total_size = new_total

        mode = "ab" if bytes_have > 0 else "wb"
        last_log = time.time()
        attempt_start = time.time()
        attempt_start_bytes = bytes_have
        with open(local_path, mode) as fp:
          for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
              continue
            fp.write(chunk)
            bytes_have += len(chunk)
            now = time.time()
            if now - last_log >= progress_every_s:
              dt = max(now - attempt_start, 1e-6)
              dbytes = bytes_have - attempt_start_bytes
              rate_mb = (dbytes / 1024 / 1024) / dt
              if total_size:
                pct = 100.0 * bytes_have / total_size
                eta_s = (total_size - bytes_have) / max(dbytes / dt, 1)
                print(
                    f"    [{os.path.basename(local_path)}] "
                    f"{bytes_have/1e9:.2f}/{total_size/1e9:.2f} GB "
                    f"({pct:.1f}%) @ {rate_mb:.0f} MB/s; ETA {eta_s/60:.0f} min",
                    flush=True,
                )
              else:
                print(
                    f"    [{os.path.basename(local_path)}] " f"{bytes_have/1e9:.2f} GB @ {rate_mb:.0f} MB/s",
                    flush=True,
                )
              last_log = now

      # Clean exit. Validate full size if known.
      if total_size is not None and bytes_have != total_size:
        raise RuntimeError(f"truncated: have {bytes_have} bytes, expected {total_size}")
      return bytes_have, total_size

    except (RequestException, RuntimeError, ConnectionError) as exc:
      wait = min(2**attempt, 60)
      print(
          f"    [retry {attempt}/{max_retries}] @{bytes_have/1e9:.2f} GB: "
          f"{type(exc).__name__}: {str(exc)[:200]}; sleeping {wait}s",
          flush=True,
      )
      time.sleep(wait)
      # Refresh bytes_have from disk in case the partial write was flushed.
      bytes_have = os.path.getsize(local_path) if os.path.exists(local_path) else 0

  raise RuntimeError(
      f"exhausted {max_retries} retries; got {bytes_have} bytes" f"{'/' + str(total_size) if total_size else ''}"
  )


def download_one(
    rel_path: str,
    http_base_url: str,
    gcs_dest_prefix: str,
    staging_dir: str,
    skip_existing: bool,
    max_retries: int,
) -> tuple[str, str, int]:
  """Download one mix-file entry: HTTP → local stage → GCS upload."""
  if not http_base_url.endswith("/"):
    http_base_url += "/"
  http_url = urljoin(http_base_url, rel_path)
  gcs_dest = gcs_dest_prefix.rstrip("/") + "/" + rel_path.lstrip("/")

  if skip_existing and gcs_path_exists(gcs_dest):
    return rel_path, "skipped", 0

  local_path = os.path.join(staging_dir, rel_path)

  try:
    bytes_written, _ = http_resumable_to_local(http_url, local_path, max_retries=max_retries)

    bucket_name, blob_name = parse_gcs_bucket_and_prefix(gcs_dest)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    blob.reload()
    if blob.size != bytes_written:
      return (
          rel_path,
          f"error:size_mismatch local={bytes_written} gcs={blob.size}",
          bytes_written,
      )

    os.remove(local_path)
    return rel_path, "ok", bytes_written

  except Exception:  # pylint: disable=broad-except
    return rel_path, f"error:{traceback.format_exc()[-800:]}", 0


def parse_args():
  """Parse CLI args for the HTTP → GCS downloader."""
  parser = argparse.ArgumentParser(
      description="Resumable HTTP→GCS downloader for OLMo .npy mix files.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument("--mix-file", required=True)
  parser.add_argument("--gcs-dest", required=True)
  parser.add_argument("--tokenizer", default="allenai/dolma3-tokenizer")
  parser.add_argument("--http-base-url", default="http://olmo-data.org/")
  parser.add_argument("--workers", type=int, default=2)
  parser.add_argument(
      "--staging-dir",
      default="/tmp/olmo-staging",
      help="Local scratch dir for partial downloads.",
  )
  parser.add_argument("--max-retries", type=int, default=20)
  parser.add_argument("--no-skip-existing", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args()


def main():
  args = parse_args()

  print(f"Parsing mix file: {args.mix_file}")
  entries = parse_mix_file(args.mix_file, args.tokenizer)
  print(f"  {len(entries)} files found  (tokenizer={args.tokenizer!r})")

  gcs_dest_prefix = args.gcs_dest.rstrip("/")
  http_base = args.http_base_url.rstrip("/") + "/"

  if args.dry_run:
    for label, rel_path in entries:
      url = urljoin(http_base, rel_path)
      gcs = gcs_dest_prefix + "/" + rel_path.lstrip("/")
      print(f"  [{label}]  {url}  →  {gcs}")
    print(f"\nDry run complete. {len(entries)} files listed.")
    return

  os.makedirs(args.staging_dir, exist_ok=True)
  skip_existing = not args.no_skip_existing
  if skip_existing:
    print("Existing GCS files will be skipped (use --no-skip-existing to re-download).")
  print(f"Staging to: {args.staging_dir}")
  print(f"Workers: {args.workers}  Max retries per file: {args.max_retries}")

  rel_paths = [r for _, r in entries]
  n_ok = n_skipped = n_error = 0
  total_bytes = 0
  errors = []
  start = time.time()

  with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {
        pool.submit(
            download_one,
            rel_path,
            args.http_base_url,
            gcs_dest_prefix,
            args.staging_dir,
            skip_existing,
            args.max_retries,
        ): rel_path
        for rel_path in rel_paths
    }
    for i, future in enumerate(as_completed(futures), start=1):
      rel_path, status, nbytes = future.result()
      total_bytes += nbytes
      if status == "ok":
        n_ok += 1
      elif status == "skipped":
        n_skipped += 1
      else:
        n_error += 1
        errors.append((rel_path, status))

      elapsed = time.time() - start
      tag = status.split(":", 1)[0]
      print(
          f"  [{i}/{len(rel_paths)}] ok={n_ok} skipped={n_skipped} err={n_error}"
          f" | {total_bytes/1e9:.2f} GB | {elapsed:.0f}s | {tag} {rel_path}",
          flush=True,
      )

  elapsed = time.time() - start
  print(
      f"\nDone in {elapsed:.1f}s. ok={n_ok} skipped={n_skipped} errors={n_error}"
      f" | {total_bytes/1e9:.2f} GB transferred"
  )

  if errors:
    print(f"\nFailed files ({len(errors)}):")
    for rel_path, status in errors:
      print(f"  {rel_path}\n    {status}")
    sys.exit(1)


if __name__ == "__main__":
  main()
