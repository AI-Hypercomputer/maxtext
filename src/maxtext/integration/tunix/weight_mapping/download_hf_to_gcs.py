"""Download HuggingFace model files directly to GCS without requiring local disk storage.

Each file is streamed from HuggingFace and uploaded directly to GCS in chunks,
avoiding the need for ~700GB of local disk for large models like DeepSeek-V3.

Usage:
  HF_TOKEN=<token> python download_hf_to_gcs.py \\
    --repo_id=deepseek-ai/DeepSeek-V3 \\
    --gcs_path=gs://maxtext-model-checkpoints/deepseek-v3/hf-weights/ \\
    --workers=8

  # Dry run to list files without downloading:
  python download_hf_to_gcs.py --gcs_path=gs://... --dry_run

  # Resume an interrupted download (skips already-uploaded files by default):
  HF_TOKEN=<token> python download_hf_to_gcs.py --gcs_path=gs://... --skip_existing
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from absl import app, flags
from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url, list_repo_files
from tqdm import tqdm

FLAGS = flags.FLAGS
_REPO_ID = flags.DEFINE_string("repo_id", "deepseek-ai/DeepSeek-V3", "HuggingFace repo ID.")
_GCS_PATH = flags.DEFINE_string("gcs_path", None, "Target GCS path (e.g. gs://bucket/prefix/).")
_WORKERS = flags.DEFINE_integer("workers", 4, "Number of parallel download/upload threads.")
_SKIP_EXISTING = flags.DEFINE_boolean("skip_existing", True, "Skip files already present in GCS.")
_REVISION = flags.DEFINE_string("revision", "main", "HuggingFace revision/branch/tag.")
_DRY_RUN = flags.DEFINE_boolean("dry_run", False, "List files only; do not download or upload.")

flags.mark_flag_as_required("gcs_path")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Non-model files to exclude from the download
_SKIP_FILES = frozenset({".gitattributes", "README.md", "LICENSE", ".git"})

# GCS chunk size for resumable uploads (must be a multiple of 256KB).
# 64MB gives good throughput for large safetensors shards.
_GCS_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB

# Timeout for individual HTTP requests (seconds). HF CDN can be slow for large files.
_HTTP_TIMEOUT_S = 600

# Timeout for GCS upload operations (seconds). Large shards (~4GB) need generous time.
_GCS_UPLOAD_TIMEOUT_S = 7200


def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
  """Parse 'gs://bucket/some/prefix/' into ('bucket', 'some/prefix/')."""
  path = gcs_path.removeprefix("gs://")
  bucket_name, _, prefix = path.partition("/")
  if prefix and not prefix.endswith("/"):
    prefix += "/"
  return bucket_name, prefix


def _stream_hf_file_to_gcs(
    filename: str,
    repo_id: str,
    revision: str,
    token: str,
    bucket: storage.Bucket,
    gcs_prefix: str,
    skip_existing: bool,
) -> tuple[str, bool, str]:
  """Stream a single HuggingFace file directly to GCS.

  Returns:
    (filename, success, message) where message describes the outcome.
  """
  blob_name = f"{gcs_prefix}{filename}"
  blob = bucket.blob(blob_name)
  blob.chunk_size = _GCS_CHUNK_SIZE

  if skip_existing and blob.exists():
    return filename, True, "skipped (already in GCS)"

  url = hf_hub_url(repo_id, filename, revision=revision)
  headers = {"Authorization": f"Bearer {token}"} if token else {}

  try:
    t0 = time.time()
    is_large_binary = filename.endswith(".safetensors")
    with requests.get(url, stream=is_large_binary, headers=headers, timeout=_HTTP_TIMEOUT_S) as resp:
      resp.raise_for_status()
      if is_large_binary:
        # Safetensors are uncompressed — stream directly to GCS to avoid buffering GBs.
        # Content-Length is accurate so GCS resumable upload works correctly.
        resp.raw.decode_content = True
        content_length = int(resp.headers.get("Content-Length", 0)) or None
        blob.upload_from_file(
            resp.raw,
            content_type="application/octet-stream",
            size=content_length,
            timeout=_GCS_UPLOAD_TIMEOUT_S,
        )
      else:
        # Small text/JSON/Python files: HF CDN serves them gzip-compressed but
        # urllib3 decompresses transparently, making Content-Length wrong.
        # Read fully into memory so GCS gets the correct actual byte count.
        data = resp.content
        content_length = len(data)
        blob.upload_from_string(
            data,
            content_type="application/octet-stream",
            timeout=_GCS_UPLOAD_TIMEOUT_S,
        )
    elapsed = time.time() - t0
    if content_length:
      size_mb = content_length / 1024 / 1024
      speed_mb = size_mb / elapsed if elapsed > 0 else 0
      return filename, True, f"{size_mb:.1f} MB in {elapsed:.1f}s ({speed_mb:.1f} MB/s)"
    return filename, True, f"done in {elapsed:.1f}s"
  except Exception as e:  # pylint: disable=broad-except
    return filename, False, str(e)


def _list_repo_files_with_retry(repo_id: str, revision: str, token: str | None) -> list[str]:
  """List all files in an HF repo, with retry and non-recursive fallback for large repos."""
  api = HfApi(token=token)

  # Try the fast recursive listing first (single API call), with retries.
  for attempt in range(3):
    try:
      files = list(list_repo_files(repo_id, revision=revision, token=token))
      return files
    except Exception as e:  # pylint: disable=broad-except
      if attempt < 2:
        wait = 10 * (attempt + 1)
        log.warning("list_repo_files attempt %d failed (%s), retrying in %ds...", attempt + 1, e, wait)
        time.sleep(wait)
      else:
        log.warning("Recursive listing failed 3 times, falling back to non-recursive traversal: %s", e)

  # Fallback: traverse directory tree non-recursively.
  log.info("Traversing repo tree non-recursively...")
  all_files = []
  dirs_to_visit = [""]  # start at repo root
  while dirs_to_visit:
    current_dir = dirs_to_visit.pop()
    try:
      entries = list(api.list_repo_tree(repo_id, path_in_repo=current_dir, revision=revision, recursive=False))
    except Exception as e:  # pylint: disable=broad-except
      log.error("Failed to list directory '%s': %s", current_dir, e)
      continue
    for entry in entries:
      if entry.rfilename if hasattr(entry, "rfilename") else False:
        all_files.append(entry.rfilename)
      elif hasattr(entry, "path"):
        # Determine if it's a file or directory by checking for blob-specific attrs
        if hasattr(entry, "size"):  # RepoFile has size; RepoFolder does not
          all_files.append(entry.path)
        else:
          dirs_to_visit.append(entry.path)
  return all_files


def main(argv):
  del argv

  token = os.environ.get("HF_TOKEN", "")
  if not token:
    log.warning("HF_TOKEN not set; downloading without auth (public model, may be rate-limited)")

  repo_id = _REPO_ID.value
  gcs_path = _GCS_PATH.value
  bucket_name, gcs_prefix = _parse_gcs_path(gcs_path)

  log.info("Listing files in %s @ %s ...", repo_id, _REVISION.value)
  all_files = _list_repo_files_with_retry(repo_id, _REVISION.value, token or None)
  files = [f for f in all_files if f not in _SKIP_FILES]
  skipped_meta = len(all_files) - len(files)
  log.info("Found %d files to download (%d metadata files excluded)", len(files), skipped_meta)

  if _DRY_RUN.value:
    print(f"\nDry run — {len(files)} files that would be uploaded to gs://{bucket_name}/{gcs_prefix}:")
    for f in files:
      print(f"  {f}")
    return

  gcs_client = storage.Client()
  bucket = gcs_client.bucket(bucket_name)

  log.info(
      "Target: gs://%s/%s | workers=%d | skip_existing=%s",
      bucket_name, gcs_prefix, _WORKERS.value, _SKIP_EXISTING.value,
  )

  success_count = 0
  skip_count = 0
  failed_files = []

  with ThreadPoolExecutor(max_workers=_WORKERS.value) as executor:
    futures = {
        executor.submit(
            _stream_hf_file_to_gcs,
            f, repo_id, _REVISION.value, token,
            bucket, gcs_prefix, _SKIP_EXISTING.value,
        ): f
        for f in files
    }
    with tqdm(total=len(files), unit="file", dynamic_ncols=True) as pbar:
      for future in as_completed(futures):
        filename, success, msg = future.result()
        if success:
          if "skipped" in msg:
            skip_count += 1
          else:
            success_count += 1
          pbar.set_postfix_str(f"{os.path.basename(filename)}: {msg}")
        else:
          failed_files.append(filename)
          log.error("FAILED: %s — %s", filename, msg)
        pbar.update(1)

  print(f"\n{'='*60}")
  print(f"Uploaded: {success_count}  Skipped: {skip_count}  Failed: {len(failed_files)}")
  print(f"GCS destination: gs://{bucket_name}/{gcs_prefix}")
  if failed_files:
    print("Failed files:")
    for f in failed_files:
      print(f"  {f}")
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
