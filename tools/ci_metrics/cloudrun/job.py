# Copyright 2026 Google LLC
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

"""One collector tick over a store kept in a GCS bucket: lock, download, tick, upload, unlock."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "AI-Hypercomputer/maxtext"
LOCK_NAME = ".lock"

# The only object names (under the prefix) this job reads, writes or deletes. They are exactly the
# files the collector's store and views are made of; anything else in the bucket is left alone.
MANAGED_NAMES = (
    re.compile(r"data/[a-z]+-\d{4}-\d{2}(?:-part\d+)?\.ndjson"),
    re.compile(r"data/state\.json"),
    re.compile(r"views/meta\.json"),
    re.compile(r"views/[a-z]+-\d{4}-\d{2}\.json"),
    re.compile(r"views/pr/\d+\.json"),
)
META_NAME = "views/meta.json"
STATE_NAME = "data/state.json"

# The tick's own exit codes (collector/tick.py), and what this job adds to them.
EXIT_OK = 0
EXIT_DATA_LOST = 1
EXIT_USAGE = 2
EXIT_SYNC_FAILED = 3
EXIT_TICK_TIMEOUT = 4
EXIT_TICK_CRASHED = 5
EXIT_LOCKED = 75  # EX_TEMPFAIL: another tick holds the lock. Not normal, so the execution fails.

# A deletion is refused when it would remove more than half of the stored files, and more than this
# many. Pruning one month of a full store removes far less; losing the store would remove it all.
MAX_DELETE_FRACTION = 0.5
MAX_DELETE_FREE = 20

# collector.tick.print_report pads every label of its report to this width.
REPORT_LABEL_WIDTH = 26


class ConfigError(ValueError):
  """The environment does not describe a job that can run."""


class SyncError(RuntimeError):
  """The bucket and the local store could not be brought in line."""


def log(severity: str, message: str, **fields: Any) -> None:
  """Prints one structured log line; Cloud Logging reads the severity and the fields from it."""
  print(json.dumps({"severity": severity, "message": message, **fields}, sort_keys=True, default=str), flush=True)


def is_managed(name: str) -> bool:
  """True for an object name (relative to the prefix) that belongs to the store or its views."""
  return any(pattern.fullmatch(name) for pattern in MANAGED_NAMES)


def md5_base64(path: Path) -> str:
  """Returns a file's MD5 the way GCS reports `md5Hash`."""
  digest = hashlib.md5()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return base64.b64encode(digest.digest()).decode("ascii")


def upload_phase(name: str) -> int:
  """Orders uploads: rows, then the index, then the month views, then meta.json last."""
  if name == META_NAME:
    return 3
  if name.startswith("views/"):
    return 2
  if name == STATE_NAME:
    return 1
  return 0


def content_type(name: str) -> str:
  """Returns the content type an object is stored with."""
  return "application/x-ndjson" if name.endswith(".ndjson") else "application/json"


@dataclass(frozen=True)
class ObjectInfo:
  """What the job needs to know about one stored object."""

  name: str
  md5: str | None
  generation: int
  size: int
  created: datetime | None = None


class ObjectStore(Protocol):
  """The part of a bucket this job uses. `GcsObjectStore` is the real one; tests pass a fake."""

  def list(self, prefix: str) -> dict[str, ObjectInfo]:
    """Lists the objects whose names start with the prefix, by full name."""

  def download(self, name: str, generation: int, path: Path) -> None:
    """Writes exactly that generation of the object to the path."""

  def upload(self, path: Path, name: str, if_generation_match: int, content_type_value: str) -> None:
    """Writes the file only if the object is still at that generation (0: only if it does not exist)."""

  def create_if_absent(self, name: str, data: bytes) -> int | None:
    """Creates the object if no object has the name, and returns its generation; None if one did."""

  def info(self, name: str) -> ObjectInfo | None:
    """Returns the object's details, or None when there is no such object."""

  def delete(self, name: str, if_generation_match: int) -> bool:
    """Deletes the object if it is still at that generation; False if it was not."""


class GcsObjectStore:
  """`ObjectStore` over a real bucket, through google-cloud-storage."""

  def __init__(self, bucket_name: str, client: Any = None) -> None:
    """Opens the bucket with the ambient credentials (the job's service account on Cloud Run)."""
    # Imported here so the logic and its tests run without the library installed.
    from google.api_core import exceptions  # pylint: disable=import-outside-toplevel
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    self._exceptions = exceptions
    self._client = client if client is not None else storage.Client()
    self._bucket = self._client.bucket(bucket_name)

  @staticmethod
  def _info(blob: Any) -> ObjectInfo:
    """Turns a blob into the details the job keeps."""
    return ObjectInfo(
        name=blob.name,
        md5=blob.md5_hash,
        generation=int(blob.generation),
        size=int(blob.size or 0),
        created=blob.time_created,
    )

  def list(self, prefix: str) -> dict[str, ObjectInfo]:
    """Lists the objects under a prefix."""
    return {blob.name: self._info(blob) for blob in self._client.list_blobs(self._bucket, prefix=prefix)}

  def download(self, name: str, generation: int, path: Path) -> None:
    """Downloads one generation of an object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    self._bucket.blob(name, generation=generation).download_to_filename(str(path))

  def upload(self, path: Path, name: str, if_generation_match: int, content_type_value: str) -> None:
    """Uploads a file with a generation precondition."""
    try:
      self._bucket.blob(name).upload_from_filename(
          str(path), content_type=content_type_value, if_generation_match=if_generation_match
      )
    except self._exceptions.PreconditionFailed as error:
      raise SyncError(f"{name} changed in the bucket while this tick ran; nothing more is uploaded") from error

  def create_if_absent(self, name: str, data: bytes) -> int | None:
    """Creates an object only if the name is free."""
    blob = self._bucket.blob(name)
    try:
      blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
    except self._exceptions.PreconditionFailed:
      return None
    return int(blob.generation)

  def info(self, name: str) -> ObjectInfo | None:
    """Reads one object's details."""
    blob = self._bucket.get_blob(name)
    return None if blob is None else self._info(blob)

  def delete(self, name: str, if_generation_match: int) -> bool:
    """Deletes one object with a generation precondition."""
    try:
      self._bucket.blob(name).delete(if_generation_match=if_generation_match)
    except (self._exceptions.PreconditionFailed, self._exceptions.NotFound):
      return False
    return True


@dataclass(frozen=True)
class Config:
  """The job's settings, all from the environment."""

  bucket: str
  prefix: str = ""
  repo: str = DEFAULT_REPO
  tick_args: tuple[str, ...] = ()
  lock_stale_seconds: int = 1800
  tick_timeout_seconds: int = 600
  workdir: str | None = None
  no_upload: bool = False
  keep_workdir: bool = False

  @property
  def lock_name(self) -> str:
    """The lock object's full name."""
    return f"{self.prefix}{LOCK_NAME}"

  @classmethod
  def from_env(cls, env: Mapping[str, str]) -> Config:
    """Reads and checks the settings."""
    bucket = env.get("CI_PULSE_BUCKET", "").strip()
    if not bucket:
      raise ConfigError("CI_PULSE_BUCKET is not set")
    prefix = env.get("CI_PULSE_PREFIX", "").strip()
    if prefix and (prefix.startswith("/") or not prefix.endswith("/") or ".." in prefix.split("/")):
      raise ConfigError(f"CI_PULSE_PREFIX must be empty or a relative path ending in '/', got {prefix!r}")
    tick_args = tuple(shlex.split(env.get("CI_PULSE_TICK_ARGS", "")))
    for reserved in ("--out", "--repo", "--dry-run"):
      if any(arg == reserved or arg.startswith(f"{reserved}=") for arg in tick_args):
        raise ConfigError(f"CI_PULSE_TICK_ARGS may not set {reserved}; the job sets it")
    return cls(
        bucket=bucket,
        prefix=prefix,
        repo=env.get("CI_PULSE_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO,
        tick_args=tick_args,
        lock_stale_seconds=_positive_int(env, "CI_PULSE_LOCK_STALE_SECONDS", 1800),
        tick_timeout_seconds=_positive_int(env, "CI_PULSE_TICK_TIMEOUT_SECONDS", 600),
        workdir=env.get("CI_PULSE_WORKDIR") or None,
        no_upload=env.get("CI_PULSE_NO_UPLOAD", "") == "1",
        keep_workdir=env.get("CI_PULSE_KEEP_WORKDIR", "") == "1",
    )


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
  """Reads a positive whole number from the environment."""
  raw = env.get(name, "").strip()
  if not raw:
    return default
  try:
    value = int(raw)
  except ValueError as error:
    raise ConfigError(f"{name} must be a whole number, got {raw!r}") from error
  if value < 1:
    raise ConfigError(f"{name} must be at least 1, got {value}")
  return value


class Lock:
  """A lock object in the bucket, so two ticks never write the store at the same time."""

  def __init__(self, objects: ObjectStore, name: str, stale_seconds: int, owner: str, clock: Any = None) -> None:
    """Describes the lock; nothing is taken yet."""
    self.objects = objects
    self.name = name
    self.stale_seconds = stale_seconds
    self.owner = owner
    self.clock = clock or (lambda: datetime.now(timezone.utc))
    self.generation: int | None = None
    self.took_over: str | None = None

  def _payload(self) -> bytes:
    """What the lock object holds: who took it, and when."""
    return json.dumps({"owner": self.owner, "taken_at": self.clock().isoformat()}).encode("utf-8")

  def acquire(self) -> bool:
    """Takes the lock, taking over one left behind by a tick that died; False if another tick holds it."""
    self.generation = self.objects.create_if_absent(self.name, self._payload())
    if self.generation is not None:
      return True
    held = self.objects.info(self.name)
    if held is None:
      # Released between the two calls: try once more.
      self.generation = self.objects.create_if_absent(self.name, self._payload())
      return self.generation is not None
    age = (self.clock() - held.created).total_seconds() if held.created else 0.0
    if age < self.stale_seconds:
      return False
    if not self.objects.delete(self.name, held.generation):
      return False
    self.generation = self.objects.create_if_absent(self.name, self._payload())
    if self.generation is not None:
      self.took_over = f"a lock {age:.0f}s old (generation {held.generation}) was taken over"
    return self.generation is not None

  def release(self) -> bool:
    """Removes the lock if it is still ours."""
    if self.generation is None:
      return False
    released = self.objects.delete(self.name, self.generation)
    self.generation = None
    return released


@dataclass
class SyncReport:
  """What a download or an upload did."""

  files: int = 0
  bytes: int = 0
  unchanged: int = 0
  deleted: int = 0
  skipped_local: list[str] = field(default_factory=list)


def download_store(objects: ObjectStore, prefix: str, store_dir: Path) -> tuple[dict[str, ObjectInfo], SyncReport]:
  """Copies every managed object under the prefix into the store directory."""
  remote: dict[str, ObjectInfo] = {}
  report = SyncReport()
  for full_name, info in sorted(objects.list(prefix).items()):
    name = full_name[len(prefix) :]
    if not is_managed(name):
      continue
    target = store_dir / name
    objects.download(full_name, info.generation, target)
    actual = md5_base64(target)
    if info.md5 is not None and actual != info.md5:
      raise SyncError(f"{full_name} arrived with MD5 {actual}, but the bucket lists {info.md5}")
    remote[name] = info
    report.files += 1
    report.bytes += target.stat().st_size
  return remote, report


def local_files(store_dir: Path) -> tuple[dict[str, Path], list[str]]:
  """Lists the managed files of the local store, and the other files it holds."""
  managed: dict[str, Path] = {}
  others: list[str] = []
  if not store_dir.is_dir():
    return managed, others
  for path in sorted(store_dir.rglob("*")):
    if not path.is_file() or path.is_symlink():
      continue
    name = path.relative_to(store_dir).as_posix()
    if is_managed(name):
      managed[name] = path
    else:
      others.append(name)
  return managed, others


def upload_store(objects: ObjectStore, prefix: str, store_dir: Path, remote: Mapping[str, ObjectInfo]) -> SyncReport:
  """Uploads what changed, meta.json last, then deletes what the tick pruned."""
  managed, others = local_files(store_dir)
  report = SyncReport(skipped_local=others)

  gone = sorted(set(remote) - set(managed))
  if len(gone) > MAX_DELETE_FREE and len(gone) > MAX_DELETE_FRACTION * len(remote):
    raise SyncError(
        f"the tick left {len(managed)} of {len(remote)} stored files and would delete {len(gone)}; "
        "refusing, because that looks like a lost store rather than pruning"
    )

  changed = []
  for name, path in managed.items():
    known = remote.get(name)
    if known is not None and known.md5 is not None and known.md5 == md5_base64(path):
      report.unchanged += 1
      continue
    changed.append(name)
  for name in sorted(changed, key=lambda item: (upload_phase(item), item)):
    known = remote.get(name)
    objects.upload(managed[name], f"{prefix}{name}", known.generation if known else 0, content_type(name))
    report.files += 1
    report.bytes += managed[name].stat().st_size

  # Deleted last: the meta.json just uploaded no longer names these files.
  for name in gone:
    if objects.delete(f"{prefix}{name}", remote[name].generation):
      report.deleted += 1
    else:
      raise SyncError(f"{prefix}{name} changed in the bucket while this tick ran; it was not deleted")
  return report


def report_values(lines: Sequence[str], label: str) -> list[str]:
  """Returns the values of the tick report's lines with exactly that label."""
  # collector.tick.print_report prints `  {label:<26} {value}`; matching the padded label means
  # "API" can never match "API budget" or "API requests spent".
  prefix = f"{label:<{REPORT_LABEL_WIDTH}} "
  values = []
  for line in lines:
    text = line.lstrip()
    if text.startswith(prefix):
      values.append(text[len(prefix) :].strip())
  return values


@dataclass
class TickResult:
  """How the tick subprocess ended, and what it printed."""

  exit_code: int | None
  summary: str | None
  timed_out: bool = False
  lines: list[str] = field(default_factory=list)


def tick_command(config: Config, store_dir: Path) -> list[str]:
  """The collector command line for one tick over the local store."""
  command = [sys.executable, "-B", "-m", "collector.tick", "--out", str(store_dir), "--repo", config.repo]
  return command + list(config.tick_args)


def run_tick(config: Config, store_dir: Path, env: Mapping[str, str]) -> TickResult:
  """Runs `python -m collector.tick` on the local store, echoing its output, with a deadline."""
  command = tick_command(config, store_dir)
  log("INFO", "ci-pulse: tick starting", ci_pulse={"command": command[3:]})
  return run_command(command, config.tick_timeout_seconds, env, cwd=APP_DIR)


def run_command(
    command: Sequence[str], timeout_seconds: float, env: Mapping[str, str], cwd: Path = APP_DIR
) -> TickResult:
  """Runs a command, echoing and keeping its output and its last summary line; stops it at the deadline."""
  with subprocess.Popen(
      list(command),
      cwd=str(cwd),
      env=dict(env),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
  ) as process:
    summary: list[str] = []
    lines: list[str] = []

    def echo() -> None:
      for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
        if line.startswith("ci-metrics "):
          summary[:] = [line.strip()]

    reader = threading.Thread(target=echo, daemon=True)
    reader.start()
    try:
      code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
      process.terminate()
      try:
        process.wait(timeout=15)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
      reader.join(timeout=5)
      return TickResult(exit_code=None, summary=summary[0] if summary else None, timed_out=True, lines=lines)
    reader.join(timeout=5)
  return TickResult(exit_code=code, summary=summary[0] if summary else None, lines=lines)


def run(config: Config, objects: ObjectStore, env: Mapping[str, str]) -> int:
  """Runs one locked tick and returns the job's exit code."""
  started = time.monotonic()
  owner = env.get("CLOUD_RUN_EXECUTION") or f"local-{os.getpid()}"
  facts: dict[str, Any] = {"bucket": config.bucket, "prefix": config.prefix, "execution": owner}
  lock = Lock(objects, config.lock_name, config.lock_stale_seconds, owner)

  try:
    acquired = lock.acquire()
  except Exception as error:  # pylint: disable=broad-exception-caught
    facts["error"] = f"lock: {type(error).__name__}: {error}"
    log("ERROR", f"ci-pulse: exit {EXIT_SYNC_FAILED}: {facts['error']}", ci_pulse=facts)
    return EXIT_SYNC_FAILED
  if not acquired:
    held = objects.info(config.lock_name)
    facts["lock_created"] = held.created if held else None
    log("WARNING", "ci-pulse: another tick holds the lock; this one did nothing", ci_pulse=facts)
    return EXIT_LOCKED
  if lock.took_over:
    facts["lock_note"] = lock.took_over
    log("WARNING", f"ci-pulse: {lock.took_over}", ci_pulse=facts)

  base: Path | None = None
  code = EXIT_OK
  stage = "download"
  try:
    base = Path(tempfile.mkdtemp(prefix="ci-pulse-", dir=config.workdir))
    store_dir = base / "store"
    store_dir.mkdir()
    remote, downloaded = download_store(objects, config.prefix, store_dir)
    facts["downloaded"] = {"files": downloaded.files, "bytes": downloaded.bytes}
    facts["github_token"] = bool(env.get("GITHUB_TOKEN"))

    stage = "tick"
    result = run_tick(config, store_dir, env)
    facts["tick_exit_code"] = result.exit_code
    facts["tick_summary"] = result.summary
    facts["tick_timed_out"] = result.timed_out
    # With no token the budget is per IP address, and on Cloud Run each connection can leave from a
    # different one, so only the tick's own readings describe the budget it spent.
    facts["tick_budget"] = next(iter(report_values(result.lines, "API budget")), None)
    facts["tick_stopped_early"] = next(iter(report_values(result.lines, "Stopped early")), None)
    facts["tick_problems"] = report_values(result.lines, "Problem")

    if result.timed_out:
      code = EXIT_TICK_TIMEOUT
    elif result.exit_code in (EXIT_OK, EXIT_DATA_LOST):
      code = result.exit_code
    elif result.exit_code is None or result.exit_code < 0:
      # Killed by a signal (out of memory, for one).
      code = EXIT_TICK_CRASHED
    else:
      # A usage error writes nothing, and any other code is not the collector's: neither is saved.
      code = result.exit_code

    stage = "upload"
    if config.no_upload:
      facts["uploaded"] = "skipped (CI_PULSE_NO_UPLOAD=1)"
    elif code in (EXIT_OK, EXIT_DATA_LOST, EXIT_TICK_TIMEOUT):
      # Every store file is written whole (temp file, then rename), so even a tick that lost data
      # or was stopped leaves files worth keeping; the next tick re-reads what is missing.
      uploaded = upload_store(objects, config.prefix, store_dir, remote)
      facts["uploaded"] = {
          "files": uploaded.files,
          "bytes": uploaded.bytes,
          "unchanged": uploaded.unchanged,
          "deleted": uploaded.deleted,
          "not_managed": uploaded.skipped_local,
      }
    else:
      facts["uploaded"] = "skipped (the tick did not finish)"
  except Exception as error:  # pylint: disable=broad-exception-caught
    code = EXIT_SYNC_FAILED if stage != "tick" else EXIT_DATA_LOST
    facts["error"] = f"{stage}: {type(error).__name__}: {error}"
  finally:
    try:
      facts["lock_released"] = lock.release()
    except Exception as error:  # pylint: disable=broad-exception-caught
      # Left behind, the lock goes stale and the tick after next takes it over.
      facts["lock_released"] = False
      facts["lock_error"] = f"{type(error).__name__}: {error}"
    if base is not None:
      if config.keep_workdir:
        facts["workdir"] = str(base)
      else:
        shutil.rmtree(base, ignore_errors=True)
  if not facts["lock_released"] and code == EXIT_OK:
    code = EXIT_SYNC_FAILED

  facts["exit_code"] = code
  facts["seconds"] = round(time.monotonic() - started, 1)
  outcome = (facts.get("tick_summary") or "no summary").split("|")[0].strip()
  severity = "INFO" if code == EXIT_OK else "ERROR"
  log(severity, f"ci-pulse: exit {code}: {facts.get('error') or outcome}", ci_pulse=facts)
  return code


def main(argv: Sequence[str] | None = None) -> int:
  """Reads the settings, opens the bucket and runs one tick."""
  del argv
  try:
    config = Config.from_env(os.environ)
  except ConfigError as error:
    log("ERROR", f"ci-pulse: bad settings: {error}")
    return EXIT_USAGE
  objects = GcsObjectStore(config.bucket)
  return run(config, objects, os.environ)


if __name__ == "__main__":
  sys.exit(main())
