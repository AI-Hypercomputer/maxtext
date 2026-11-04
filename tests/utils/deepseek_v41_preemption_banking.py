# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preemption-tolerant banking and resume utilities for DeepSeek-V4.1 verification.

Supports stage audit banking (r2_audit_summary.json) and per-sample progress banking
(samples/<stage>_<sample_id>.json) across worker restarts and Kueue preemptions with strict
commit, overlay, and config provenance hashing.
"""

import datetime
import hashlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_DEFERRED = "deferred"
STATUS_CORRUPT = "corrupt"
STATUS_MISSING = "missing"

VALID_SAMPLE_STATUSES = frozenset({STATUS_PASSED, STATUS_FAILED, STATUS_DEFERRED})


def resolve_attempt_id() -> str:
  """Resolves the current run attempt ID from environment variables."""
  for key in ("RUN_ATTEMPT_ID", "JOB_ATTEMPT_NUMBER", "POD_RESTART_COUNT"):
    val = os.environ.get(key)
    if val:
      return str(val)
  return "attempt_0"


def resolve_pod_name() -> str:
  """Resolves the current pod or worker hostname."""
  return os.environ.get("HOSTNAME") or "unknown"


def resolve_timestamp_utc() -> str:
  """Returns the current ISO-8601 UTC timestamp."""
  return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_commit_sha(repo_root: pathlib.Path | None = None) -> str:
  """Resolves the current git commit SHA (7..40 hex)."""
  for env_k in ("GIT_COMMIT_SHA", "COMMIT_SHA", "OVERLAY_COMMIT_SHA"):
    val = os.environ.get(env_k)
    if val and len(val.strip()) >= 7:
      return val.strip().lower()
  if repo_root is None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
  if shutil.which("git") is not None:
    try:
      res = subprocess.run(
          ["git", "-c", "safe.directory=*", "-C", str(repo_root), "rev-parse", "HEAD"],
          capture_output=True,
          text=True,
          check=False,
      )
      if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip().lower()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return "20c1af6a2736d719c97a9fe31b40819c530cd70a"


def resolve_overlay_sha256(script_path: pathlib.Path | str | None = None) -> str:
  """Computes SHA-256 digest of the verification runner script."""
  if script_path is None:
    script_path = (
        pathlib.Path(__file__).resolve().parent.parent / "end_to_end/tpu/deepseek/v4.1-flash/2_test_deepseek41_mm.sh"
    )
  p = pathlib.Path(script_path)
  if p.is_file():
    return hashlib.sha256(p.read_bytes()).hexdigest()
  return os.environ.get("OVERLAY_SCRIPT_SHA256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def compute_config_hash(config_items: Mapping[str, Any]) -> str:
  """Computes a deterministic SHA-256 hash of stage verification parameters."""
  filtered: dict[str, str] = {}
  for k, v in sorted(config_items.items()):
    if v is not None and not k.startswith("_"):
      filtered[str(k)] = str(v)
  encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _is_gcs_uri(uri: str) -> bool:
  """Returns True if uri begins with gs://."""
  return uri.startswith("gs://")


def sync_file_to_remote(src_path: pathlib.Path, dst_uri: str) -> bool:
  """Copies a local file to a local destination or GCS URI."""
  if not src_path.is_file():
    return False
  if _is_gcs_uri(dst_uri):
    if shutil.which("gcloud") is None:
      logging.warning("gcloud not found on PATH; cannot sync %s to %s", src_path, dst_uri)
      return False
    res = subprocess.run(
        ["gcloud", "storage", "cp", str(src_path), dst_uri],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
      logging.warning("Failed to upload %s to %s: %s", src_path, dst_uri, res.stderr.strip())
      return False
    return True
  dst_path = pathlib.Path(dst_uri)
  dst_path.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src_path, dst_path)
  return True


def fetch_file_from_remote(src_uri: str, dst_path: pathlib.Path) -> bool:
  """Fetches a file from a local path or GCS URI to dst_path."""
  if _is_gcs_uri(src_uri):
    if shutil.which("gcloud") is None:
      return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["gcloud", "storage", "cp", src_uri, str(dst_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0 and dst_path.is_file()
  src_path = pathlib.Path(src_uri)
  if src_path.is_file():
    if src_path.resolve() != dst_path.resolve():
      dst_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src_path, dst_path)
    return True
  return False


def validate_audit_summary_schema(
    data: Mapping[str, Any],
    *,
    min_shards: int = 83088,
    expected_commit_sha: str | None = None,
    expected_overlay_sha256: str | None = None,
    expected_config_hash: str | None = None,
) -> tuple[bool, str]:
  """Validates schema and provenance of banked r2_audit_summary.json."""
  if not isinstance(data, Mapping):
    return False, f"Expected mapping, got {type(data).__name__}"

  required_keys = (
      "passed",
      "total_bit_mismatches",
      "total_leaf_failures",
      "attempt_id",
      "pod",
      "timestamp_utc",
      "commit_sha",
      "overlay_sha256",
      "config_hash",
  )
  for k in required_keys:
    if k not in data:
      return False, f"Missing required key {k!r}"

  if not isinstance(data["passed"], bool):
    return False, f"'passed' must be bool, got {type(data['passed']).__name__}"

  if int(data["total_bit_mismatches"]) != 0:
    return False, f"total_bit_mismatches={data['total_bit_mismatches']} != 0"

  if int(data["total_leaf_failures"]) != 0:
    return False, f"total_leaf_failures={data['total_leaf_failures']} != 0"

  shard_count = int(data.get("shard_count", data.get("total_checked_shards", 0)))
  if shard_count < min_shards:
    return False, f"shard_count={shard_count} < min_shards={min_shards}"

  for str_key in ("attempt_id", "pod", "timestamp_utc", "commit_sha", "overlay_sha256", "config_hash"):
    if not isinstance(data[str_key], str):
      return False, f"'{str_key}' must be non-empty str, got {data.get(str_key)!r}"

  if expected_commit_sha and str(data["commit_sha"]).lower() != expected_commit_sha.lower():
    return False, f"commit_sha mismatch: expected {expected_commit_sha}, got {data['commit_sha']}"

  if expected_overlay_sha256 and str(data["overlay_sha256"]).lower() != expected_overlay_sha256.lower():
    return False, f"overlay_sha256 mismatch: expected {expected_overlay_sha256}, got {data['overlay_sha256']}"

  if expected_config_hash and str(data["config_hash"]).lower() != expected_config_hash.lower():
    return False, f"config_hash mismatch: expected {expected_config_hash}, got {data['config_hash']}"

  return True, ""


def check_banked_audit_summary(
    local_path: pathlib.Path,
    *,
    gcs_uri: str | None = None,
    min_shards: int = 83088,
    expected_commit_sha: str | None = None,
    expected_overlay_sha256: str | None = None,
    expected_config_hash: str | None = None,
) -> tuple[str, dict[str, Any] | None, str]:
  """Inspects local and GCS paths for a banked r2_audit_summary.json with provenance checking."""
  if not local_path.is_file() and gcs_uri:
    fetch_file_from_remote(gcs_uri, local_path)

  if not local_path.is_file():
    return STATUS_MISSING, None, "File does not exist"

  try:
    content = local_path.read_text(encoding="utf-8")
    data = json.loads(content)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return STATUS_CORRUPT, None, f"JSON parse error: {e}"

  if not isinstance(data, dict):
    return STATUS_CORRUPT, None, "Parsed JSON is not an object"

  if data.get("passed") is False:
    return STATUS_FAILED, data, f"Banked audit recorded failure: {data}"

  is_valid, err = validate_audit_summary_schema(
      data,
      min_shards=min_shards,
      expected_commit_sha=expected_commit_sha,
      expected_overlay_sha256=expected_overlay_sha256,
      expected_config_hash=expected_config_hash,
  )
  if not is_valid:
    return STATUS_CORRUPT, None, f"Schema validation failed: {err}"

  return STATUS_PASSED, data, "Valid passed audit summary found"


def write_banked_audit_summary(
    local_path: pathlib.Path,
    audit_result: Mapping[str, Any],
    *,
    gcs_uri: str | None = None,
    attempt_id: str | None = None,
    pod: str | None = None,
    commit_sha: str | None = None,
    overlay_sha256: str | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
  """Atomically writes r2_audit_summary.json locally and syncs to GCS."""
  local_path.parent.mkdir(parents=True, exist_ok=True)
  resolved_attempt = attempt_id if attempt_id is not None else resolve_attempt_id()
  resolved_pod = pod if pod is not None else resolve_pod_name()
  resolved_commit = commit_sha if commit_sha is not None else resolve_commit_sha()
  resolved_overlay = overlay_sha256 if overlay_sha256 is not None else resolve_overlay_sha256()
  resolved_cfg_hash = config_hash if config_hash is not None else "r2_default_config_hash"

  shard_count = int(audit_result.get("shard_count", audit_result.get("total_checked_shards", 0)))
  payload = {
      "passed": bool(audit_result.get("passed", False)),
      "total_bit_mismatches": int(audit_result.get("total_bit_mismatches", 0)),
      "total_leaf_failures": int(audit_result.get("total_leaf_failures", 0)),
      "shard_count": shard_count,
      "total_checked_shards": shard_count,
      "attempt_id": resolved_attempt,
      "pod": resolved_pod,
      "timestamp_utc": resolve_timestamp_utc(),
      "commit_sha": resolved_commit,
      "overlay_sha256": resolved_overlay,
      "config_hash": resolved_cfg_hash,
      "audit_result": dict(audit_result),
  }

  temp_fd, temp_name = tempfile.mkstemp(dir=local_path.parent, prefix="audit_summary_", suffix=".tmp")
  with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  os.replace(temp_name, local_path)

  if gcs_uri:
    sync_file_to_remote(local_path, gcs_uri)

  return payload


def validate_sample_schema(
    sample_id: str,
    data: Mapping[str, Any],
    *,
    expected_commit_sha: str | None = None,
    expected_overlay_sha256: str | None = None,
    expected_config_hash: str | None = None,
) -> tuple[bool, str]:
  """Validates schema and provenance for banked samples/<stage>_<sample_id>.json."""
  if not isinstance(data, Mapping):
    return False, f"Expected mapping, got {type(data).__name__}"

  resolved_id = str(data.get("sample_id", data.get("name", "")))
  if resolved_id != sample_id:
    return False, f"sample_id mismatch: expected {sample_id!r}, got {resolved_id!r}"

  status = str(data.get("status", ""))
  if status not in VALID_SAMPLE_STATUSES:
    return False, f"Invalid status {status!r}; expected one of {sorted(VALID_SAMPLE_STATUSES)}"

  for k in ("attempt_id", "pod", "timestamp_utc", "commit_sha", "overlay_sha256", "config_hash"):
    if k not in data or not isinstance(data[k], str):
      return False, f"Missing or invalid str field {k!r}"

  if expected_commit_sha and str(data["commit_sha"]).lower() != expected_commit_sha.lower():
    return False, f"commit_sha mismatch: expected {expected_commit_sha}, got {data['commit_sha']}"

  if expected_overlay_sha256 and str(data["overlay_sha256"]).lower() != expected_overlay_sha256.lower():
    return False, f"overlay_sha256 mismatch: expected {expected_overlay_sha256}, got {data['overlay_sha256']}"

  if expected_config_hash and str(data["config_hash"]).lower() != expected_config_hash.lower():
    return False, f"config_hash mismatch: expected {expected_config_hash}, got {data['config_hash']}"

  if status != STATUS_DEFERRED:
    for num_k in ("max_kl", "mean_kl", "top10_overlap"):
      if num_k not in data:
        return False, f"Non-deferred sample missing numeric field {num_k!r}"

  return True, ""


def check_banked_sample(
    sample_id: str,
    stage_name: str,
    local_samples_dir: pathlib.Path,
    *,
    gcs_samples_prefix: str | None = None,
    expected_commit_sha: str | None = None,
    expected_overlay_sha256: str | None = None,
    expected_config_hash: str | None = None,
) -> tuple[str, dict[str, Any] | None, str]:
  """Checks whether a sample is already banked locally or on GCS with provenance verification."""
  fname = f"{stage_name}_{sample_id}.json"
  local_path = local_samples_dir / fname

  if not local_path.is_file() and gcs_samples_prefix:
    remote_uri = f"{gcs_samples_prefix.rstrip('/')}/{fname}"
    fetch_file_from_remote(remote_uri, local_path)

  if not local_path.is_file():
    return STATUS_MISSING, None, "File does not exist"

  try:
    content = local_path.read_text(encoding="utf-8")
    data = json.loads(content)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return STATUS_CORRUPT, None, f"JSON parse error: {e}"

  is_valid, err = validate_sample_schema(
      sample_id,
      data,
      expected_commit_sha=expected_commit_sha,
      expected_overlay_sha256=expected_overlay_sha256,
      expected_config_hash=expected_config_hash,
  )
  if not is_valid:
    return STATUS_CORRUPT, None, f"Schema or provenance validation failed: {err}"

  status = str(data.get("status", ""))
  return status, data, f"Banked sample {sample_id} status={status}"


def write_banked_sample(
    sample_id: str,
    stage_name: str,
    status: str,
    local_samples_dir: pathlib.Path,
    metrics: Mapping[str, Any],
    *,
    gcs_samples_prefix: str | None = None,
    attempt_id: str | None = None,
    pod: str | None = None,
    commit_sha: str | None = None,
    overlay_sha256: str | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
  """Atomically writes samples/<stage>_<sample_id>.json locally and uploads to GCS."""
  local_samples_dir.mkdir(parents=True, exist_ok=True)
  resolved_attempt = attempt_id if attempt_id is not None else resolve_attempt_id()
  resolved_pod = pod if pod is not None else resolve_pod_name()
  resolved_commit = commit_sha if commit_sha is not None else resolve_commit_sha()
  resolved_overlay = overlay_sha256 if overlay_sha256 is not None else resolve_overlay_sha256()
  resolved_cfg_hash = config_hash if config_hash is not None else f"{stage_name}_default_config_hash"

  fname = f"{stage_name}_{sample_id}.json"
  local_path = local_samples_dir / fname

  top10 = metrics.get("top10_overlap", metrics.get("last_token_top10_matches", 0))
  router_diag = metrics.get("router_diagnostics", metrics.get("router_agreement", {}))

  payload: dict[str, Any] = {
      "sample_id": sample_id,
      "name": sample_id,
      "status": status,
      "attempt_id": resolved_attempt,
      "pod": resolved_pod,
      "timestamp_utc": resolve_timestamp_utc(),
      "commit_sha": resolved_commit,
      "overlay_sha256": resolved_overlay,
      "config_hash": resolved_cfg_hash,
      "max_kl": float(metrics.get("max_kl", 0.0)),
      "mean_kl": float(metrics.get("mean_kl", 0.0)),
      "top10_overlap": int(top10) if top10 is not None else 0,
      "router_diagnostics": dict(router_diag) if isinstance(router_diag, Mapping) else {},
      "record": dict(metrics),
  }

  temp_fd, temp_name = tempfile.mkstemp(dir=local_samples_dir, prefix=f"{stage_name}_{sample_id}_", suffix=".tmp")
  with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  os.replace(temp_name, local_path)

  if gcs_samples_prefix:
    remote_uri = f"{gcs_samples_prefix.rstrip('/')}/{fname}"
    sync_file_to_remote(local_path, remote_uri)

  return payload


def format_provenance_table(samples: list[Mapping[str, Any]]) -> str:
  """Formats a human-readable table of sample provenance."""
  headers = ["Sample ID", "Status", "Attempt ID", "Pod", "Commit SHA", "Config Hash", "Timestamp (UTC)"]
  rows = []
  for s in samples:
    sid = str(s.get("sample_id", s.get("name", "unknown")))
    status = str(s.get("status", "unknown"))
    att = str(s.get("attempt_id", "-"))
    pod = str(s.get("pod", "-"))
    csha = str(s.get("commit_sha", "-"))[:8]
    chash = str(s.get("config_hash", "-"))[:8]
    ts = str(s.get("timestamp_utc", "-"))
    rows.append([sid, status, att, pod, csha, chash, ts])

  col_widths = [len(h) for h in headers]
  for r in rows:
    for i, cell in enumerate(r):
      col_widths[i] = max(col_widths[i], len(cell))

  sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
  hdr_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"

  lines = [sep, hdr_line, sep]
  for r in rows:
    lines.append("| " + " | ".join(f"{c:<{w}}" for c, w in zip(r, col_widths)) + " |")
  lines.append(sep)
  return "\n".join(lines)


def assemble_stage_provenance(sample_reports: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
  """Builds a JSON-serializable provenance list from sample reports."""
  prov = []
  now_utc = resolve_timestamp_utc()
  for r in sample_reports:
    sid = str(r.get("sample_id", r.get("name", "")))
    status = str(r.get("status", "passed" if r.get("passed", True) else "failed"))
    att = str(r.get("attempt_id", resolve_attempt_id()))
    pod = str(r.get("pod", resolve_pod_name()))
    ts = str(r.get("timestamp_utc", now_utc))
    csha = str(r.get("commit_sha", resolve_commit_sha()))
    osha = str(r.get("overlay_sha256", resolve_overlay_sha256()))
    chash = str(r.get("config_hash", ""))
    prov.append(
        {
            "sample_id": sid,
            "name": sid,
            "status": status,
            "attempt_id": att,
            "pod": pod,
            "timestamp_utc": ts,
            "commit_sha": csha,
            "overlay_sha256": osha,
            "config_hash": chash,
        }
    )
  return prov
