# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone cold Step 1 (HF -> GCS) and DeepSeek Step 1b (FP8/FP4 -> BF16) runner."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import datetime
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_DIR not in sys.path:
  sys.path.insert(0, _BENCH_DIR)

from bench import metadata  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position

_CHUNK_BYTES = 16 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 32 * 1024 * 1024
_DEFAULT_MAX_WORKERS = 16
_MAX_RETRIES = 8
_GATED_PREFIXES = ("meta-llama/", "google/")
_EXCLUDED_DIR_PREFIXES = ("original/", "metal/", ".git/")


def split_gcs_uri(uri: str) -> tuple[str, str]:
  """Splits `gs://bucket/prefix` into `(bucket, prefix_without_leading_slash)`."""
  if not uri.startswith("gs://"):
    raise ValueError(f"Expected gs:// URI, got {uri!r}")
  bucket, _, prefix = uri.removeprefix("gs://").partition("/")
  if not bucket:
    raise ValueError(f"Missing GCS bucket in {uri!r}")
  return bucket, prefix.strip("/")


def _utc_now_iso() -> str:
  return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_storage_client():
  from google.cloud import storage  # pylint: disable=import-outside-toplevel

  return storage.Client()


def _default_hf_token(explicit: str | None = None) -> str | None:
  if explicit:
    return explicit.strip() or None
  for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    val = os.environ.get(key, "").strip()
    if val:
      return val
  return None


def _retry_backoff(attempt: int, base_s: float = 12.0, cap_s: float = 60.0) -> float:
  return min(base_s * (attempt + 1), cap_s) + random.uniform(0.5, 3.0)


def _list_blobs_under_prefix(storage_client: Any, gcs_dir: str) -> list[Any]:
  bucket_name, prefix = split_gcs_uri(gcs_dir)
  bucket = storage_client.bucket(bucket_name)
  norm_prefix = f"{prefix}/" if prefix else ""
  return list(storage_client.list_blobs(bucket, prefix=norm_prefix))


def ensure_empty_gcs_prefix(
    gcs_dir: str,
    *,
    storage_client: Any | None = None,
    clean_target: bool = False,
) -> int:
  """Verifies `gcs_dir` is empty before a cold download (optionally wiping stale partial blobs first).

  Raises RuntimeError if non-empty and `clean_target` is False, preventing warm-cache
  folders from masquerading as cold Step 1 measurements.
  """
  client = storage_client or _default_storage_client()
  blobs = [b for b in _list_blobs_under_prefix(client, gcs_dir) if not b.name.endswith("/")]
  if not blobs:
    return 0
  if not clean_target:
    raise RuntimeError(
        f"Refusing cold Step 1 download: target GCS folder {gcs_dir} is non-empty ({len(blobs)} objects). "
        "Pass --clean-target only when wiping an incomplete failed attempt to measure from 0 files."
    )
  for blob in blobs:
    blob.delete()
  remaining = [b for b in _list_blobs_under_prefix(client, gcs_dir) if not b.name.endswith("/")]
  if remaining:
    raise RuntimeError(f"Failed to clean {gcs_dir}: {len(remaining)} objects still remain")
  return 0


class _CountingReader:
  """Wraps a file-like binary stream to count exact bytes read via `read()`."""

  def __init__(self, raw: Any):
    self._raw = raw
    self.bytes_read = 0

  def read(self, size: int = -1) -> bytes:
    chunk = self._raw.read(size)
    if chunk:
      self.bytes_read += len(chunk)
    return chunk


def _fetch_hf_url_bytes(url: str, headers: dict[str, str], timeout: tuple[int, int] = (30, 120)) -> bytes:
  """Fetches small metadata/index bytes from HuggingFace Hub with 429 retry handling."""
  import requests  # pylint: disable=import-outside-toplevel

  for attempt in range(_MAX_RETRIES):
    try:
      with requests.get(url, headers=headers, timeout=timeout) as resp:
        if resp.status_code == 429:
          retry_after = float(resp.headers.get("Retry-After", 20 * (attempt + 1)))
          time.sleep(min(retry_after, 90.0) + random.uniform(0.5, 2.5))
          continue
        resp.raise_for_status()
        return resp.content
    except Exception:  # pylint: disable=broad-exception-caught
      if attempt == _MAX_RETRIES - 1:
        raise
      time.sleep(_retry_backoff(attempt))
  return b""


def _hf_headers(repo_id: str, token: str | None, attempt: int) -> dict[str, str]:
  is_gated = repo_id.startswith(_GATED_PREFIXES)
  if token and (is_gated or attempt % 2 == 0):
    return {"Authorization": f"Bearer {token}"}
  return {}


def resolve_hf_repo_files(
    repo_id: str,
    token: str | None = None,
    *,
    hf_api_factory: Callable[[str | None], Any] | None = None,
    http_get_bytes: Callable[[str, dict[str, str]], bytes] | None = None,
    hf_url_fn: Callable[[str, str], str] | None = None,
) -> tuple[list[str], list[str], float]:
  """Lists `.safetensors` shard filenames and `.json` config/index filenames in `repo_id`."""
  import huggingface_hub  # pylint: disable=import-outside-toplevel

  t0 = time.perf_counter()
  api_factory = hf_api_factory or (lambda tok: huggingface_hub.HfApi(token=tok))
  url_fn = hf_url_fn or (lambda r, fn: huggingface_hub.hf_hub_url(repo_id=r, filename=fn))
  get_bytes = http_get_bytes or _fetch_hf_url_bytes

  all_files: list[str] = []
  for attempt in range(_MAX_RETRIES):
    try:
      use_tok = token if (repo_id.startswith(_GATED_PREFIXES) or attempt % 2 == 0) else None
      all_files = list(api_factory(use_tok).list_repo_files(repo_id=repo_id))
      break
    except Exception:  # pylint: disable=broad-exception-caught
      if attempt == _MAX_RETRIES - 1:
        raise
      time.sleep(_retry_backoff(attempt, base_s=15.0))

  filtered = [fn for fn in all_files if not fn.startswith(_EXCLUDED_DIR_PREFIXES)]
  json_files = sorted(fn for fn in filtered if fn.endswith(".json"))
  safetensors_files = sorted(fn for fn in filtered if fn.endswith(".safetensors"))

  # When an index JSON exists, prefer its exact weight_map shards so consolidated duplicates are excluded.
  if "model.safetensors.index.json" in json_files:
    try:
      raw_idx = get_bytes(url_fn(repo_id, "model.safetensors.index.json"), _hf_headers(repo_id, token, 0))
      weight_map = json.loads(raw_idx.decode("utf-8")).get("weight_map") or {}
      indexed = sorted({str(v) for v in weight_map.values() if str(v).endswith(".safetensors")})
      if indexed:
        safetensors_files = indexed
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  if not safetensors_files:
    raise RuntimeError(f"No .safetensors files found in HF repository {repo_id!r}")
  list_s = time.perf_counter() - t0
  return safetensors_files, json_files, list_s


@dataclasses.dataclass(frozen=True)
class FileTransferStats:
  """Measured single-file download timing and size."""

  filename: str
  bytes_transferred: int
  elapsed_s: float


def stream_hf_file_to_gcs(
    repo_id: str,
    filename: str,
    gcs_cache_dir: str,
    token: str | None,
    *,
    storage_client: Any | None = None,
    hf_url_fn: Callable[[str, str], str] | None = None,
    requests_get: Callable[..., Any] | None = None,
) -> FileTransferStats:
  """Streams one HF Hub file into `<gcs_cache_dir>/<filename>` in a single GET request."""
  import huggingface_hub  # pylint: disable=import-outside-toplevel
  import requests  # pylint: disable=import-outside-toplevel

  bucket_name, prefix = split_gcs_uri(gcs_cache_dir)
  blob_name = f"{prefix}/{filename}" if prefix else filename
  url = (hf_url_fn or (lambda r, fn: huggingface_hub.hf_hub_url(repo_id=r, filename=fn)))(repo_id, filename)
  get_fn = requests_get or requests.get
  client = storage_client or _default_storage_client()

  t0 = time.perf_counter()
  for attempt in range(_MAX_RETRIES):
    try:
      blob = client.bucket(bucket_name).blob(blob_name)
      headers = _hf_headers(repo_id, token, attempt)
      with get_fn(url, headers=headers, stream=True, timeout=(30, 300)) as resp:
        if getattr(resp, "status_code", 200) == 429:
          retry_after = float((getattr(resp, "headers", None) or {}).get("Retry-After", 20 * (attempt + 1)))
          time.sleep(min(retry_after, 90.0) + random.uniform(1.0, 4.0))
          continue
        resp.raise_for_status()
        if hasattr(resp.raw, "decode_content"):
          resp.raw.decode_content = True
        reader = _CountingReader(resp.raw)
        with blob.open("wb", chunk_size=_UPLOAD_CHUNK_BYTES) as gcs_out:
          shutil.copyfileobj(reader, gcs_out, length=_CHUNK_BYTES)
      expected_len = (getattr(resp, "headers", None) or {}).get("Content-Length")
      if expected_len is not None and int(expected_len) > 0 and reader.bytes_read != int(expected_len):
        raise IOError(f"Short read on {filename}: got {reader.bytes_read} bytes, expected {expected_len}")
      return FileTransferStats(
          filename=filename,
          bytes_transferred=reader.bytes_read,
          elapsed_s=time.perf_counter() - t0,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      if attempt == _MAX_RETRIES - 1:
        raise
      time.sleep(_retry_backoff(attempt))
  raise RuntimeError(f"Exhausted retries downloading {filename} from {repo_id}")


def _write_json_local_and_gcs(
    payload: dict[str, Any],
    *,
    local_path: str | None = None,
    gcs_uri: str | None = None,
    storage_client: Any | None = None,
) -> None:
  """Writes JSON payload to local path and/or `gs://` destination URI."""
  encoded = json.dumps(payload, indent=2, sort_keys=False)
  if local_path:
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
      f.write(encoded)
  if gcs_uri:
    client = storage_client or _default_storage_client()
    bucket_name, blob_name = split_gcs_uri(gcs_uri)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_string(encoded, content_type="application/json")


def _build_base_step1_record(
    *,
    platform: str,
    region: str,
    model_name: str,
    hf_repo_id: str,
    run_id: str,
    gcs_cache_dir: str,
    machine_type: str,
    workers: int,
    started_utc: str,
    result_json_gcs_uri: str | None,
    step1_log_url: str | None,
) -> dict[str, Any]:
  """Constructs an initialized Step 1 result dictionary matching schema_v2."""
  return {
      "schema_version": schema_v2.SCHEMA_VERSION,
      "phase_name": "step1_hf_to_gcs",
      "tpu_platform": platform,
      "step1_gcs_bucket_region": region,
      "model_name": model_name,
      "step1_hf_repo_id": hf_repo_id,
      "step1_list_hf_repo_files_seconds": 0.0,
      "step1_files_already_in_gcs_count": 0,
      "step1_download_hf_to_gcs_seconds": 0.0,
      "step1_download_hf_to_gcs_gb": 0.0,
      "step1_download_hf_to_gcs_gb_per_second": 0.0,
      "step1_number_of_files": 0,
      "step1_slowest_file_download_seconds": 0.0,
      "step1_slowest_file_name": "",
      "step1_download_workers_count": workers,
      "step1_host_machine_type": machine_type,
      "step1_total_seconds": 0.0,
      "step1_value_source": "cold_download",
      "step1_measured_by_run_id": run_id,
      "run_id": run_id,
      "run_status": "OK",
      "failure_category": "",
      "error_summary": "",
      "gcs_cache_dir": gcs_cache_dir,
      "phase_started_at_utc": started_utc,
      "phase_finished_at_utc": started_utc,
      "step1_log_url": step1_log_url or "",
      "result_json_url": result_json_gcs_uri or "",
      "host_hardware": metadata.collect_host_hardware_info(),
      "env_cluster": metadata.collect_env_cluster_metadata(),
      "code_version": metadata.load_code_version_metadata(),
  }


def run_cold_hf_to_gcs_step1(
    *,
    platform: str,
    model_name: str,
    hf_repo_id: str,
    gcs_cache_dir: str,
    run_id: str,
    region: str | None = None,
    token: str | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    clean_target: bool = False,
    local_result_path: str | None = None,
    result_gcs_uri: str | None = None,
    step1_log_url: str | None = None,
    storage_client: Any | None = None,
    hf_api_factory: Callable[[str | None], Any] | None = None,
    http_get_bytes: Callable[[str, dict[str, str]], bytes] | None = None,
    hf_url_fn: Callable[[str, str], str] | None = None,
    requests_get: Callable[..., Any] | None = None,
    header_inspector: Callable[[Sequence[str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
  """Executes a verified-cold Step 1 download from HuggingFace Hub into a GCS prefix."""
  profile = platforms.get_profile(platform)
  resolved_region = region or profile.gcp_region
  client = storage_client or _default_storage_client()
  resolved_token = _default_hf_token(token)
  started_utc = _utc_now_iso()
  env_meta = metadata.collect_env_cluster_metadata()
  hw_info = metadata.collect_host_hardware_info()
  machine_type = str(env_meta.get("host_machine_type") or os.environ.get("STEP1_HOST_MACHINE_TYPE", "") or "cpu-worker")

  record = _build_base_step1_record(
      platform=platform,
      region=resolved_region,
      model_name=model_name,
      hf_repo_id=hf_repo_id,
      run_id=run_id,
      gcs_cache_dir=gcs_cache_dir,
      machine_type=machine_type,
      workers=max_workers,
      started_utc=started_utc,
      result_json_gcs_uri=result_gcs_uri,
      step1_log_url=step1_log_url,
  )
  record["host_hardware"] = hw_info
  record["env_cluster"] = env_meta

  try:
    t_total0 = time.perf_counter()
    ensure_empty_gcs_prefix(gcs_cache_dir, storage_client=client, clean_target=clean_target)
    st_files, json_files, list_s = resolve_hf_repo_files(
        hf_repo_id,
        resolved_token,
        hf_api_factory=hf_api_factory,
        http_get_bytes=http_get_bytes,
        hf_url_fn=hf_url_fn,
    )
    all_targets = list(json_files) + list(st_files)
    worker_count = min(max_workers, max(1, len(all_targets)))
    record["step1_download_workers_count"] = worker_count
    record["step1_list_hf_repo_files_seconds"] = round(list_s, 4)

    t_dl0 = time.perf_counter()
    stats: list[FileTransferStats] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
      futs = [
          pool.submit(
              stream_hf_file_to_gcs,
              hf_repo_id,
              fn,
              gcs_cache_dir,
              resolved_token,
              storage_client=client,
              hf_url_fn=hf_url_fn,
              requests_get=requests_get,
          )
          for fn in all_targets
      ]
      for fut in concurrent.futures.as_completed(futs):
        stats.append(fut.result())
    dl_s = time.perf_counter() - t_dl0

    st_stats = [s for s in stats if s.filename.endswith(".safetensors")]
    total_bytes = sum(s.bytes_transferred for s in stats)
    dl_gb = total_bytes / 1e9
    slowest = max(stats, key=lambda s: s.elapsed_s)
    record.update(
        {
            "step1_download_hf_to_gcs_seconds": round(dl_s, 4),
            "step1_download_hf_to_gcs_gb": round(dl_gb, 4),
            "step1_download_hf_to_gcs_gb_per_second": round(dl_gb / dl_s, 4) if dl_s > 0 else 0.0,
            "step1_number_of_files": len(st_stats),
            "step1_slowest_file_download_seconds": round(slowest.elapsed_s, 4),
            "step1_slowest_file_name": slowest.filename,
            "step1_total_seconds": round(time.perf_counter() - t_total0, 4),
        }
    )

    inspector = header_inspector or metadata.inspect_safetensors_headers
    try:
      hdr_info = inspector([f"{gcs_cache_dir.rstrip('/')}/{fn}" for fn in st_files])
      record.update(
          {
              "hf_checkpoint_dtype": hdr_info.get("hf_checkpoint_dtype", ""),
              "checkpoint_number_of_files": hdr_info.get("checkpoint_number_of_files", len(st_stats)),
              "checkpoint_number_of_tensors": hdr_info.get("checkpoint_number_of_tensors", 0),
              "checkpoint_size_gb": hdr_info.get("checkpoint_size_gb", round(dl_gb, 4)),
          }
      )
    except Exception:  # pylint: disable=broad-exception-caught
      record["checkpoint_number_of_files"] = len(st_stats)
      record["checkpoint_size_gb"] = round(dl_gb, 4)

    record["phase_finished_at_utc"] = _utc_now_iso()
    validation_errors = schema_v2.validate_step1_record(record)
    if validation_errors:
      raise ValueError(f"Step 1 record failed schema_v2 validation: {validation_errors}")
    _write_json_local_and_gcs(
        record,
        local_path=local_result_path,
        gcs_uri=result_gcs_uri,
        storage_client=client,
    )
    return record
  except Exception as exc:
    record["run_status"] = "FAILED"
    record["failure_category"] = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
    record["error_summary"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else repr(exc)}"
    record["phase_finished_at_utc"] = _utc_now_iso()
    _write_json_local_and_gcs(
        record,
        local_path=local_result_path,
        gcs_uri=result_gcs_uri,
        storage_client=client,
    )
    raise


def _download_gcs_prefix_to_local(
    storage_client: Any,
    gcs_dir: str,
    local_dir: str,
    *,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> int:
  """Downloads all `.safetensors` and `.json` files from `gcs_dir` to `local_dir`; returns bytes copied."""
  os.makedirs(local_dir, exist_ok=True)
  blobs = [
      b
      for b in _list_blobs_under_prefix(storage_client, gcs_dir)
      if not b.name.endswith("/") and b.name.endswith((".safetensors", ".json"))
  ]
  if not blobs:
    raise RuntimeError(f"No .safetensors/.json objects found under {gcs_dir}")

  def _fetch_one(blob: Any) -> int:
    fname = blob.name.rsplit("/", 1)[-1]
    dst = os.path.join(local_dir, fname)
    with blob.open("rb") as src, open(dst, "wb") as out:
      shutil.copyfileobj(src, out, length=_CHUNK_BYTES)
    return int(getattr(blob, "size", 0) or os.path.getsize(dst))

  with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(blobs))) as pool:
    return sum(pool.map(_fetch_one, blobs))


def _upload_local_dir_to_gcs(
    storage_client: Any,
    local_dir: str,
    gcs_dir: str,
    *,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> int:
  """Uploads all `.safetensors` and `.json` files in `local_dir` to `gcs_dir`; returns bytes uploaded."""
  bucket_name, prefix = split_gcs_uri(gcs_dir)
  names = sorted(fn for fn in os.listdir(local_dir) if fn.endswith((".safetensors", ".json")))
  if not names:
    raise RuntimeError(f"No converted .safetensors/.json files found in {local_dir}")

  def _push_one(fname: str) -> int:
    src = os.path.join(local_dir, fname)
    blob_name = f"{prefix}/{fname}" if prefix else fname
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    with open(src, "rb") as f_in, blob.open("wb", chunk_size=_UPLOAD_CHUNK_BYTES) as gcs_out:
      shutil.copyfileobj(f_in, gcs_out, length=_CHUNK_BYTES)
    return os.path.getsize(src)

  with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
    return sum(pool.map(_push_one, names))


def _default_dequantize_runner(input_dir: str, output_dir: str, cache_size: int = 8) -> None:
  """Runs MaxText's standalone DeepSeek FP8/FP4 -> BF16 conversion script."""
  repo_root = os.path.dirname(_BENCH_DIR)
  script = os.path.join(
      repo_root, "src", "maxtext", "checkpoint_conversion", "standalone_scripts", "deepseek_dequantize.py"
  )
  if not os.path.exists(script):
    raise FileNotFoundError(f"MaxText dequantize script not found at {script}")
  env = dict(os.environ)
  src_dir = os.path.join(repo_root, "src")
  env["PYTHONPATH"] = f"{src_dir}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_dir
  subprocess.run(
      [
          sys.executable,
          script,
          "--input-path",
          input_dir,
          "--output-path",
          output_dir,
          "--cache-size",
          str(cache_size),
      ],
      env=env,
      check=True,
  )


def run_deepseek_step1b_dequantize(
    *,
    platform: str,
    model_name: str,
    hf_repo_id: str,
    source_gcs_dir: str,
    dest_gcs_dir: str,
    scratch_dir: str,
    run_id: str,
    region: str | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    cache_size: int = 8,
    clean_target: bool = False,
    cleanup_scratch: bool = True,
    local_result_path: str | None = None,
    result_gcs_uri: str | None = None,
    step1b_log_url: str | None = None,
    storage_client: Any | None = None,
    dequantize_fn: Callable[[str, str, int], None] | None = None,
) -> dict[str, Any]:
  """Converts DeepSeek FP8/FP4 weights from `source_gcs_dir` to BF16 in `dest_gcs_dir`."""
  profile = platforms.get_profile(platform)
  resolved_region = region or profile.gcp_region
  client = storage_client or _default_storage_client()
  env_meta = metadata.collect_env_cluster_metadata()
  machine_type = str(env_meta.get("host_machine_type") or os.environ.get("STEP1B_HOST_MACHINE_TYPE", "") or "cpu-worker")
  started_utc = _utc_now_iso()
  in_dir = os.path.join(scratch_dir, "input_fp8")
  out_dir = os.path.join(scratch_dir, "output_bf16")

  record: dict[str, Any] = {
      "schema_version": schema_v2.SCHEMA_VERSION,
      "phase_name": "step1b_deepseek_dequantize",
      "tpu_platform": platform,
      "step1_gcs_bucket_region": resolved_region,
      "model_name": model_name,
      "step1_hf_repo_id": hf_repo_id,
      "step1b_deepseek_convert_fp8_fp4_to_bf16_seconds": 0.0,
      "step1b_deepseek_convert_input_gb": 0.0,
      "step1b_deepseek_convert_output_gb": 0.0,
      "step1b_deepseek_upload_bf16_to_gcs_seconds": 0.0,
      "step1b_deepseek_copy_bf16_to_region_seconds": 0.0,
      "step1b_deepseek_host_machine_type": machine_type,
      "step1b_deepseek_bf16_source": "converted_by_this_benchmark",
      "step1b_deepseek_total_seconds": 0.0,
      "step1_measured_by_run_id": run_id,
      "run_id": run_id,
      "run_status": "OK",
      "failure_category": "",
      "error_summary": "",
      "source_gcs_dir": source_gcs_dir,
      "dest_gcs_dir": dest_gcs_dir,
      "phase_started_at_utc": started_utc,
      "phase_finished_at_utc": started_utc,
      "step1b_log_url": step1b_log_url or "",
      "result_json_url": result_gcs_uri or "",
      "host_hardware": metadata.collect_host_hardware_info(),
      "env_cluster": env_meta,
      "code_version": metadata.load_code_version_metadata(),
  }

  try:
    t0 = time.perf_counter()
    ensure_empty_gcs_prefix(dest_gcs_dir, storage_client=client, clean_target=clean_target)
    shutil.rmtree(in_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    t_conv0 = time.perf_counter()
    in_bytes = _download_gcs_prefix_to_local(client, source_gcs_dir, in_dir, max_workers=max_workers)
    (dequantize_fn or _default_dequantize_runner)(in_dir, out_dir, cache_size)
    conv_s = time.perf_counter() - t_conv0

    out_st = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".safetensors")]
    hdr_info = metadata.inspect_safetensors_headers(out_st) if out_st else {}

    t_up0 = time.perf_counter()
    out_bytes = _upload_local_dir_to_gcs(client, out_dir, dest_gcs_dir, max_workers=max_workers)
    up_s = time.perf_counter() - t_up0

    record.update(
        {
            "step1b_deepseek_convert_fp8_fp4_to_bf16_seconds": round(conv_s, 4),
            "step1b_deepseek_convert_input_gb": round(in_bytes / 1e9, 4),
            "step1b_deepseek_convert_output_gb": round(out_bytes / 1e9, 4),
            "step1b_deepseek_upload_bf16_to_gcs_seconds": round(up_s, 4),
            "step1b_deepseek_total_seconds": round(time.perf_counter() - t0, 4),
            "loaded_checkpoint_dtype": hdr_info.get("hf_checkpoint_dtype", "BF16"),
            "checkpoint_number_of_files": hdr_info.get("checkpoint_number_of_files", len(out_st)),
            "checkpoint_number_of_tensors": hdr_info.get("checkpoint_number_of_tensors", 0),
            "checkpoint_size_gb": hdr_info.get("checkpoint_size_gb", round(out_bytes / 1e9, 4)),
        }
    )
    record["phase_finished_at_utc"] = _utc_now_iso()
    _write_json_local_and_gcs(record, local_path=local_result_path, gcs_uri=result_gcs_uri, storage_client=client)
    return record
  except Exception as exc:
    record["run_status"] = "FAILED"
    record["failure_category"] = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
    record["error_summary"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else repr(exc)}"
    record["phase_finished_at_utc"] = _utc_now_iso()
    _write_json_local_and_gcs(record, local_path=local_result_path, gcs_uri=result_gcs_uri, storage_client=client)
    raise
  finally:
    if cleanup_scratch:
      shutil.rmtree(in_dir, ignore_errors=True)
      shutil.rmtree(out_dir, ignore_errors=True)


def copy_bf16_gcs_to_region(
    *,
    platform: str,
    dest_region: str,
    model_name: str,
    hf_repo_id: str,
    source_gcs_dir: str,
    dest_gcs_dir: str,
    run_id: str,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    clean_target: bool = False,
    bf16_source: str = "converted_by_this_benchmark",
    local_result_path: str | None = None,
    result_gcs_uri: str | None = None,
    step1b_log_url: str | None = None,
    storage_client: Any | None = None,
) -> dict[str, Any]:
  """Copies pre-converted BF16 `.safetensors`/`.json` files across GCS buckets/regions."""
  if bf16_source not in schema_v2.DEEPSEEK_BF16_SOURCES:
    raise ValueError(f"Invalid bf16_source {bf16_source!r}; expected one of {schema_v2.DEEPSEEK_BF16_SOURCES}")
  client = storage_client or _default_storage_client()
  env_meta = metadata.collect_env_cluster_metadata()
  started_utc = _utc_now_iso()

  ensure_empty_gcs_prefix(dest_gcs_dir, storage_client=client, clean_target=clean_target)
  dst_bucket_name, dst_prefix = split_gcs_uri(dest_gcs_dir)
  dst_bucket = client.bucket(dst_bucket_name)
  src_bucket_name, _ = split_gcs_uri(source_gcs_dir)
  src_bucket = client.bucket(src_bucket_name)

  src_blobs = [
      b
      for b in _list_blobs_under_prefix(client, source_gcs_dir)
      if not b.name.endswith("/") and b.name.endswith((".safetensors", ".json"))
  ]
  if not src_blobs:
    raise RuntimeError(f"No .safetensors/.json objects found under {source_gcs_dir}")

  t0 = time.perf_counter()

  def _copy_one(blob: Any) -> int:
    fname = blob.name.rsplit("/", 1)[-1]
    dst_blob_name = f"{dst_prefix}/{fname}" if dst_prefix else fname
    dst_blob = dst_bucket.blob(dst_blob_name)
    if hasattr(dst_blob, "rewrite"):
      token = None
      while True:
        token, _, _ = dst_blob.rewrite(blob, token=token)
        if token is None:
          break
    else:
      src_bucket.copy_blob(blob, dst_bucket, dst_blob_name)
    return int(getattr(blob, "size", 0) or 0)

  with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(src_blobs))) as pool:
    total_bytes = sum(pool.map(_copy_one, src_blobs))
  copy_s = time.perf_counter() - t0

  st_count = sum(1 for b in src_blobs if b.name.endswith(".safetensors"))
  record: dict[str, Any] = {
      "schema_version": schema_v2.SCHEMA_VERSION,
      "phase_name": "step1b_deepseek_copy_region",
      "tpu_platform": platform,
      "step1_gcs_bucket_region": dest_region,
      "model_name": model_name,
      "step1_hf_repo_id": hf_repo_id,
      "step1b_deepseek_copy_bf16_to_region_seconds": round(copy_s, 4),
      "step1b_deepseek_convert_output_gb": round(total_bytes / 1e9, 4),
      "step1b_deepseek_bf16_source": bf16_source,
      "step1b_deepseek_total_seconds": round(copy_s, 4),
      "checkpoint_number_of_files": st_count,
      "checkpoint_size_gb": round(total_bytes / 1e9, 4),
      "run_id": run_id,
      "run_status": "OK",
      "source_gcs_dir": source_gcs_dir,
      "dest_gcs_dir": dest_gcs_dir,
      "phase_started_at_utc": started_utc,
      "phase_finished_at_utc": _utc_now_iso(),
      "step1b_log_url": step1b_log_url or "",
      "result_json_url": result_gcs_uri or "",
      "host_hardware": metadata.collect_host_hardware_info(),
      "env_cluster": env_meta,
      "code_version": metadata.load_code_version_metadata(),
  }
  _write_json_local_and_gcs(record, local_path=local_result_path, gcs_uri=result_gcs_uri, storage_client=client)
  return record


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI flags for Step 1 / Step 1b download, dequantize, and copy subcommands."""
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  dl = sub.add_parser("download", help="Run a cold HF -> GCS Step 1 download")
  dl.add_argument("--platform", required=True, choices=schema_v2.PLATFORMS)
  dl.add_argument("--region", default="")
  dl.add_argument("--model", required=True)
  dl.add_argument("--hf-repo-id", required=True)
  dl.add_argument("--gcs-cache-dir", required=True)
  dl.add_argument("--run-id", required=True)
  dl.add_argument("--workers", type=int, default=_DEFAULT_MAX_WORKERS)
  dl.add_argument("--clean-target", action="store_true")
  dl.add_argument("--local-result-json", default="")
  dl.add_argument("--result-gcs-uri", default="")
  dl.add_argument("--step1-log-url", default="")

  dq = sub.add_parser("deepseek-convert", help="Convert DeepSeek FP8/FP4 -> BF16 and upload to GCS")
  dq.add_argument("--platform", required=True, choices=schema_v2.PLATFORMS)
  dq.add_argument("--region", default="")
  dq.add_argument("--model", required=True)
  dq.add_argument("--hf-repo-id", required=True)
  dq.add_argument("--source-gcs-dir", required=True)
  dq.add_argument("--dest-gcs-dir", required=True)
  dq.add_argument("--scratch-dir", required=True)
  dq.add_argument("--run-id", required=True)
  dq.add_argument("--workers", type=int, default=_DEFAULT_MAX_WORKERS)
  dq.add_argument("--cache-size", type=int, default=8)
  dq.add_argument("--clean-target", action="store_true")
  dq.add_argument("--keep-scratch", action="store_true")
  dq.add_argument("--local-result-json", default="")
  dq.add_argument("--result-gcs-uri", default="")
  dq.add_argument("--step1b-log-url", default="")

  cp = sub.add_parser("copy-region", help="Copy converted DeepSeek BF16 weights to another regional bucket")
  cp.add_argument("--platform", required=True, choices=schema_v2.PLATFORMS)
  cp.add_argument("--dest-region", required=True)
  cp.add_argument("--model", required=True)
  cp.add_argument("--hf-repo-id", required=True)
  cp.add_argument("--source-gcs-dir", required=True)
  cp.add_argument("--dest-gcs-dir", required=True)
  cp.add_argument("--run-id", required=True)
  cp.add_argument("--workers", type=int, default=_DEFAULT_MAX_WORKERS)
  cp.add_argument("--clean-target", action="store_true")
  cp.add_argument("--bf16-source", default="converted_by_this_benchmark", choices=schema_v2.DEEPSEEK_BF16_SOURCES)
  cp.add_argument("--local-result-json", default="")
  cp.add_argument("--result-gcs-uri", default="")
  cp.add_argument("--step1b-log-url", default="")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  if args.command == "download":
    rec = run_cold_hf_to_gcs_step1(
        platform=args.platform,
        region=args.region or None,
        model_name=args.model,
        hf_repo_id=args.hf_repo_id,
        gcs_cache_dir=args.gcs_cache_dir,
        run_id=args.run_id,
        max_workers=args.workers,
        clean_target=args.clean_target,
        local_result_path=args.local_result_json or None,
        result_gcs_uri=args.result_gcs_uri or None,
        step1_log_url=args.step1_log_url or None,
    )
  elif args.command == "deepseek-convert":
    rec = run_deepseek_step1b_dequantize(
        platform=args.platform,
        region=args.region or None,
        model_name=args.model,
        hf_repo_id=args.hf_repo_id,
        source_gcs_dir=args.source_gcs_dir,
        dest_gcs_dir=args.dest_gcs_dir,
        scratch_dir=args.scratch_dir,
        run_id=args.run_id,
        max_workers=args.workers,
        cache_size=args.cache_size,
        clean_target=args.clean_target,
        cleanup_scratch=not args.keep_scratch,
        local_result_path=args.local_result_json or None,
        result_gcs_uri=args.result_gcs_uri or None,
        step1b_log_url=args.step1b_log_url or None,
    )
  else:
    rec = copy_bf16_gcs_to_region(
        platform=args.platform,
        dest_region=args.dest_region,
        model_name=args.model,
        hf_repo_id=args.hf_repo_id,
        source_gcs_dir=args.source_gcs_dir,
        dest_gcs_dir=args.dest_gcs_dir,
        run_id=args.run_id,
        max_workers=args.workers,
        clean_target=args.clean_target,
        bf16_source=args.bf16_source,
        local_result_path=args.local_result_json or None,
        result_gcs_uri=args.result_gcs_uri or None,
        step1b_log_url=args.step1b_log_url or None,
    )
  print(json.dumps(rec, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
