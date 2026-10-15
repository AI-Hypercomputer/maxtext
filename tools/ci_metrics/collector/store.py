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

"""The append-only store: the collector's own copy of everything it has already read."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from collector import rows

# Schema version of `state.json`. Bump it when a field changes meaning. A state file written
# by a newer collector is refused rather than read wrong; an older one is read as it is.
STATE_VERSION = 1

# Names inside the output directory. The dashboard's loader hard-codes these paths, so they
# are constants here rather than arguments.
DATA_DIRNAME = "data"
VIEWS_DIRNAME = "views"
PR_VIEWS_DIRNAME = "pr"
STATE_FILENAME = "state.json"
NDJSON_SUFFIX = ".ndjson"
JSON_SUFFIX = ".json"

# Prefix for the temporary file an atomic write lands in first. It is created in the target
# directory, because `os.replace` is only atomic within one filesystem.
TEMP_PREFIX = ".tmp-"

# The kinds that can be stored, in the order a tick writes them.
ROW_KINDS = (rows.KIND_RUN, rows.KIND_JOB, rows.KIND_SUITE, rows.KIND_TEST, rows.KIND_RESCUE)

# The kinds whose rows all belong to one run attempt, so `state.json` can answer for them
# without the file being read. A rescue spans attempts by definition and is never in here.
ATTEMPT_KINDS = (rows.KIND_RUN, rows.KIND_JOB, rows.KIND_SUITE, rows.KIND_TEST)

# The kinds whose key stays the same while their content is still being decided. A rescue is
# keyed by run, job name and the attempt that failed - deliberately, so the failure has one
# identity - but its answer changes: the tick that sees only attempt 1 writes "failed, never
# re-run", and the tick that sees the re-run writes "rescued". Skipping the second row on its
# key alone would freeze every rescue at its first, wrong answer, so these kinds are deduped
# on their content instead: an identical row is skipped, a changed one is written as the
# correction it is.
MUTABLE_KINDS = (rows.KIND_RESCUE,)

# The field that differs between two writes of the same unchanged row, and so must be left out
# when their content is compared.
STAMP_FIELD = "collected_at"

# The payload fields that make up each kind's key, in key order. This mirrors the `key()`
# methods in `rows.py` deliberately: the store must be able to key a line that a later schema
# version would refuse to rebuild, so it reads the fields instead of constructing the row.
# Keep it in step with `rows.py` - the two are checked against each other by the tests.
KEY_FIELDS: dict[str, tuple[str, ...]] = {
    rows.KIND_RUN: ("run_id", "attempt"),
    rows.KIND_JOB: ("run_id", "attempt", "job_id"),
    rows.KIND_SUITE: ("run_id", "attempt", "suite_id"),
    rows.KIND_TEST: ("run_id", "attempt", "suite_id", "worker", "classname", "name"),
    rows.KIND_RESCUE: ("run_id", "job_name", "failed_attempt"),
}

# The timestamp a row can name its own month from, when the caller does not pass one.
#
# Only two kinds are in here. A job row is not, even though it has a `created_at`: that is the
# JOB's creation, and a re-run started days later would file the same run's rows in two months
# - which is exactly what rule 5 forbids and what `month_for_run` exists to prevent. A suite
# and a test row carry no timestamp at all. All three take `month=store.month_for_run(run)`
# from the caller. A rescue row's `failed_created_at` is the failed attempt's creation, which
# is always inside the run's own month, so it can name its month itself.
MONTH_FIELD: dict[str, str] = {
    rows.KIND_RUN: "created_at",
    rows.KIND_RESCUE: "failed_created_at",
}

# "YYYY-MM". These strings sort chronologically, which is why months are handled as text.
MONTH_FORMAT = "%Y-%m"
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# What follows "<kind>-" in a data file name. A month's first part keeps the plain name,
# `job-2026-09.ndjson`, so a store written before parts existed reads as it is. Later parts
# add a number: `job-2026-09-part2.ndjson`, `-part3`, and so on.
PART_NAME_PATTERN = re.compile(r"^(?P<month>\d{4}-(0[1-9]|1[0-2]))(-part(?P<part>[2-9]|[1-9]\d+))?$")

# A run attempt reports this status once GitHub is finished with it. Anything else means the
# attempt was still moving when it was written.
COMPLETED_STATUS = "completed"

# How long an attempt may sit in the in-flight list before a tick writes what it knows and
# stops waiting, so one stuck run cannot hold the store open forever.
PENDING_MAX_AGE_HOURS = 24

# How far a rebuilt state rewinds its watermark. The in-flight list cannot be rebuilt from an
# append-only store - nothing was written for those attempts - so the tick after a rebuild
# re-asks the last day rather than trusting a watermark that may step over them.
REBUILD_REWIND_HOURS = 24

# How old a leftover temporary file has to be before it is swept. A killed process leaves one
# behind - the bytes are safe, the rename simply never happened - and each one is a full copy
# of that month's file, in the directory the collector commits. The age guard is what makes
# the sweep safe if two processes ever write the same store at once.
TEMP_MAX_AGE_HOURS = 6

# How many run ids `state.json` keeps. Run ids rise with time, so the newest N are the recent
# ones. Upstream files roughly 2,600 pipeline runs a month, so 20,000 covers well over the
# 90-day view window; anything older can never be listed again anyway, because the watermark
# has passed it. Dropping an id is safe rather than lossy: `append` then falls back to the
# month file's own keys, which is the exact answer.
MAX_INDEXED_RUNS = 20000

# The biggest a data file may grow. GitHub rejects a push that holds any file over 100 MB and
# warns at 50 MB, and one busy month of job rows is about 180 MB. So a month's rows are split
# into parts: when the current part is full, the next rows go into a new part file.
MAX_PART_BYTES = 40 * 1024 * 1024

# Read sizes. The copy chunk bounds an append's memory; the tail window is how far back the
# last newline of a file is looked for.
COPY_CHUNK_BYTES = 1 << 20
TAIL_WINDOW_BYTES = 1 << 16


class StoreError(RuntimeError):
  """Raised when the store cannot be read, written or addressed."""


def _warn(message: str) -> None:
  """Prints a warning to stderr so it never lands in piped collector output."""
  print(message, file=sys.stderr, flush=True)


def parse_timestamp(value: Any) -> datetime | None:
  """Reads one of GitHub's ISO-8601 timestamps."""
  if isinstance(value, datetime):
    return as_utc(value)
  if not isinstance(value, str) or not value.strip():
    return None
  text = value.strip()
  if text.endswith("Z"):
    text = f"{text[:-1]}+00:00"
  try:
    return as_utc(datetime.fromisoformat(text))
  except ValueError:
    return None


def as_utc(moment: datetime) -> datetime:
  """Puts a datetime in UTC, treating a naive one as UTC already."""
  if moment.tzinfo is None:
    return moment.replace(tzinfo=timezone.utc)
  return moment.astimezone(timezone.utc)


def utc_now() -> datetime:
  """Returns the current instant in UTC."""
  return datetime.now(timezone.utc)


def month_key(value: Any) -> str:
  """Turns a timestamp into the month string its rows are filed under."""
  moment = parse_timestamp(value)
  if moment is None:
    raise StoreError(f"{value!r} is not a timestamp, so it names no month.")
  return moment.strftime(MONTH_FORMAT)


def month_for_run(run: Mapping[str, Any]) -> str:
  """Returns the month every row of a run belongs in."""
  created = run.get("created_at")
  if created is None:
    raise StoreError(f"run {run.get('id', run.get('run_id'))!r} has no 'created_at', so it names no month.")
  return month_key(created)


def _key_part(value: object) -> str:
  """Encodes one part of a key, exactly as `rows.py` does."""
  if value is None:
    return ""
  return quote(str(value), safe="")


def row_key(payload: Mapping[str, Any]) -> str:
  """Builds the store key of a stored JSON row."""
  kind = str(payload.get("kind", ""))
  fields = KEY_FIELDS.get(kind)
  if fields is None:
    raise StoreError(f"row kind {payload.get('kind')!r} is not one of {sorted(KEY_FIELDS)}.")
  if payload.get(fields[0]) is None:
    raise StoreError(f"{kind} row has no {fields[0]!r}, so it cannot be keyed.")
  return rows.KEY_SEPARATOR.join([kind, *(_key_part(payload.get(name)) for name in fields)])


def check_kind(kind: str) -> str:
  """Checks that a kind is one this store files."""
  if kind not in KEY_FIELDS:
    raise StoreError(f"{kind!r} is not a stored row kind; it is one of {list(ROW_KINDS)}.")
  return kind


def check_month(month: str) -> str:
  """Checks that a month string is a real "YYYY-MM"."""
  if not isinstance(month, str) or not MONTH_PATTERN.match(month):
    raise StoreError(f"{month!r} is not a month in the form YYYY-MM.")
  return month


def write_bytes_atomic(path: Path, data: bytes) -> None:
  """Writes a file so that a killed process cannot leave it half written."""
  _ensure_dir(path.parent)
  handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=TEMP_PREFIX, suffix=path.suffix)
  temp_path = Path(temp_name)
  try:
    with os.fdopen(handle, "wb") as out:
      out.write(data)
      out.flush()
      os.fsync(out.fileno())
    os.replace(temp_path, path)
  except OSError as error:
    temp_path.unlink(missing_ok=True)
    raise StoreError(f"{path} could not be written: {error}") from error
  except BaseException:
    temp_path.unlink(missing_ok=True)
    raise
  _fsync_dir(path.parent)


def write_text_atomic(path: Path, text: str) -> None:
  """Writes text atomically, UTF-8 encoded."""
  write_bytes_atomic(path, text.encode("utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
  """Writes one JSON document atomically."""
  try:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  except (TypeError, ValueError) as error:
    raise StoreError(f"{path} could not be serialised: {error}") from error
  write_text_atomic(path, f"{text}\n")


def _ensure_dir(path: Path) -> None:
  """Creates a directory and its parents if they are not there yet."""
  try:
    path.mkdir(parents=True, exist_ok=True)
  except OSError as error:
    raise StoreError(f"{path} could not be created: {error}") from error


def _fsync_dir(path: Path) -> None:
  """Flushes a directory entry so a rename survives a power cut."""
  try:
    handle = os.open(str(path), os.O_RDONLY)
  except OSError:
    return
  try:
    os.fsync(handle)
  except OSError:
    # Accepted on purpose, and the only swallowed error in this module: the rename that was
    # just made is already atomic, so a directory that will not flush costs durability across
    # a power cut and nothing else. There is no partial state to report.
    pass
  finally:
    os.close(handle)


def _sweep_temp(directory: Path, now: float | None = None) -> int:
  """Deletes temporary files an earlier process left behind when it was killed."""
  if not directory.is_dir():
    return 0
  moment = now if now is not None else datetime.now(tz=timezone.utc).timestamp()
  cutoff = moment - TEMP_MAX_AGE_HOURS * 3600
  swept = 0
  try:
    leftovers = sorted(directory.glob(f"{TEMP_PREFIX}*"))
  except OSError as error:
    _warn(f"{directory} could not be listed for leftover temporary files ({error}); carrying on.")
    return 0
  for path in leftovers:
    try:
      if path.stat().st_mtime > cutoff:
        continue
      path.unlink()
    except OSError as error:
      _warn(f"{path.name} is a leftover temporary file that could not be removed ({error}); carrying on.")
      continue
    swept += 1
  if swept:
    _warn(f"removed {swept} leftover temporary file(s) from {directory} - an earlier run was killed mid-write.")
  return swept


def _complete_length(path: Path) -> int:
  """Returns how many bytes of a file end in a complete line."""
  size = path.stat().st_size
  if size == 0:
    return 0
  with path.open("rb") as handle:
    position = size
    while position > 0:
      start = max(0, position - TAIL_WINDOW_BYTES)
      handle.seek(start)
      chunk = handle.read(position - start)
      index = chunk.rfind(b"\n")
      if index != -1:
        return start + index + 1
      position = start
  return 0


def _iter_payloads(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
  """Reads one NDJSON file, line by line."""
  if not path.exists():
    return
  try:
    with path.open("r", encoding="utf-8") as handle:
      for index, raw in enumerate(handle):
        text = raw.strip()
        if not text:
          continue
        try:
          payload = json.loads(text)
        except json.JSONDecodeError as error:
          if not raw.endswith("\n"):
            _warn(f"{path.name} line {index + 1} is a torn write and was ignored: {error}")
            continue
          raise StoreError(f"{path} line {index + 1} is not JSON: {error}") from error
        if not isinstance(payload, dict):
          raise StoreError(f"{path} line {index + 1} holds a {type(payload).__name__}, not a row object.")
        yield index, payload
  except OSError as error:
    raise StoreError(f"{path} could not be read: {error}") from error


@dataclass
class PendingAttempt:
  """A run attempt the collector has seen but cannot store yet, because it is still running."""

  run_id: int
  attempt: int
  first_seen_at: str
  created_at: str | None = None
  status: str | None = None

  @property
  def identity(self) -> tuple[int, int]:
    """Returns the (run id, attempt) pair this entry is filed under."""
    return (self.run_id, self.attempt)

  def to_json(self) -> dict[str, Any]:
    """Returns the entry as it is stored in `state.json`."""
    return {
        "run_id": self.run_id,
        "attempt": self.attempt,
        "first_seen_at": self.first_seen_at,
        "created_at": self.created_at,
        "status": self.status,
    }

  @classmethod
  def from_json(cls, payload: Mapping[str, Any]) -> PendingAttempt:
    """Rebuilds an entry from `state.json`."""
    run_id = _as_int(payload.get("run_id"))
    attempt = _as_int(payload.get("attempt"))
    if run_id is None or attempt is None:
      raise StoreError(f"pending entry {dict(payload)!r} has no run id and attempt.")
    first_seen = payload.get("first_seen_at")
    return cls(
        run_id=run_id,
        attempt=attempt,
        first_seen_at=str(first_seen) if first_seen else rows.utc_now_iso(),
        created_at=_as_str(payload.get("created_at")),
        status=_as_str(payload.get("status")),
    )


def _as_int(value: Any) -> int | None:
  """Reads a value as an int, or None when it is not one."""
  if isinstance(value, bool) or value is None:
    return None
  if isinstance(value, int):
    return value
  try:
    return int(str(value))
  except (TypeError, ValueError):
    return None


def _as_str(value: Any) -> str | None:
  """Reads a value as a string, keeping None as None."""
  return None if value is None else str(value)


@dataclass
class State:
  """The index of what the store already holds, and what it is still waiting for."""

  collected: dict[int, set[int]] = field(default_factory=dict)
  incomplete: dict[int, set[int]] = field(default_factory=dict)
  pending: dict[tuple[int, int], PendingAttempt] = field(default_factory=dict)
  watermark_run_id: int | None = None
  watermark_created_at: str | None = None
  rebuilt: bool = False
  updated_at: str | None = None
  v: int = STATE_VERSION

  def has_attempt(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt's rows are in the store already, final or not."""
    return self.is_collected(run_id, attempt) or self.is_incomplete(run_id, attempt)

  def is_collected(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt was stored from a run GitHub had finished."""
    return attempt in self.collected.get(run_id, ())

  def is_incomplete(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt was stored while it was still running."""
    return attempt in self.incomplete.get(run_id, ())

  def mark_collected(self, run_id: int, attempt: int, created_at: str | None = None) -> None:
    """Records an attempt as stored in full."""
    self.collected.setdefault(run_id, set()).add(attempt)
    self.incomplete.get(run_id, set()).discard(attempt)
    self._drop_empty(run_id)
    self.drop_pending(run_id, attempt)
    self.note_run(run_id, created_at)

  def mark_incomplete(self, run_id: int, attempt: int, created_at: str | None = None) -> None:
    """Records an attempt as written while it was still running."""
    self.incomplete.setdefault(run_id, set()).add(attempt)
    self.collected.get(run_id, set()).discard(attempt)
    self._drop_empty(run_id)
    self.drop_pending(run_id, attempt)
    self.note_run(run_id, created_at)

  def mark_pending(
      self,
      run_id: int,
      attempt: int,
      created_at: str | None = None,
      status: str | None = None,
      first_seen_at: str | None = None,
  ) -> bool:
    """Puts an attempt on the in-flight list, or leaves it there."""
    if self.has_attempt(run_id, attempt):
      return False
    identity = (run_id, attempt)
    existing = self.pending.get(identity)
    if existing is not None:
      existing.created_at = created_at or existing.created_at
      existing.status = status or existing.status
    else:
      self.pending[identity] = PendingAttempt(
          run_id=run_id,
          attempt=attempt,
          first_seen_at=first_seen_at or rows.utc_now_iso(),
          created_at=created_at,
          status=status,
      )
    self.note_run(run_id, created_at)
    return True

  def drop_pending(self, run_id: int, attempt: int) -> bool:
    """Takes an attempt off the in-flight list."""
    return self.pending.pop((run_id, attempt), None) is not None

  def forget_attempt(self, run_id: int, attempt: int) -> bool:
    """Un-records an attempt so its rows can be harvested again."""
    was_known = self.has_attempt(run_id, attempt)
    self.collected.get(run_id, set()).discard(attempt)
    self.incomplete.get(run_id, set()).discard(attempt)
    self._drop_empty(run_id)
    return was_known

  def note_run(self, run_id: int, created_at: str | None = None) -> None:
    """Moves the watermark forward if this run is newer than everything seen so far."""
    if self.watermark_run_id is None or run_id > self.watermark_run_id:
      self.watermark_run_id = run_id
      if created_at:
        self.watermark_created_at = created_at
      return
    if created_at and (self.watermark_created_at is None or created_at > self.watermark_created_at):
      self.watermark_created_at = created_at

  def expired_pending(
      self, now: datetime | None = None, max_age_hours: int = PENDING_MAX_AGE_HOURS
  ) -> list[PendingAttempt]:
    """Returns the in-flight attempts that have waited too long."""
    moment = as_utc(now) if now is not None else utc_now()
    cutoff = moment - timedelta(hours=max_age_hours)
    expired: list[tuple[str, PendingAttempt]] = []
    for entry in self.pending.values():
      seen = parse_timestamp(entry.first_seen_at)
      if seen is None:
        _warn(
            f"pending run {entry.run_id} attempt {entry.attempt} has an unreadable first_seen_at; treating it as expired."
        )
        expired.append(("", entry))
      elif seen <= cutoff:
        expired.append((entry.first_seen_at, entry))
    expired.sort(key=lambda item: (item[0], item[1].run_id, item[1].attempt))
    return [entry for _, entry in expired]

  @property
  def attempt_count(self) -> int:
    """Returns how many run attempts the store holds, incomplete ones included."""
    return sum(len(attempts) for attempts in self.collected.values()) + sum(
        len(attempts) for attempts in self.incomplete.values()
    )

  @property
  def pending_count(self) -> int:
    """Returns how many attempts are in flight - meta.json's uncollected count."""
    return len(self.pending)

  def prune(self, keep: int = MAX_INDEXED_RUNS) -> int:
    """Drops the oldest run ids from the index so `state.json` cannot grow forever."""
    known = set(self.collected) | set(self.incomplete)
    if len(known) <= keep:
      return 0
    protected = {run_id for run_id, _ in self.pending}
    droppable = sorted(known - protected)
    surplus = len(known) - keep
    dropped = 0
    for run_id in droppable:
      if dropped >= surplus:
        break
      self.collected.pop(run_id, None)
      self.incomplete.pop(run_id, None)
      dropped += 1
    return dropped

  def _drop_empty(self, run_id: int) -> None:
    """Removes a run id whose attempt set has emptied, so the file does not grow husks."""
    for index in (self.collected, self.incomplete):
      if run_id in index and not index[run_id]:
        del index[run_id]

  def to_json(self) -> dict[str, Any]:
    """Returns the state as it is stored."""
    return {
        "v": self.v,
        "updated_at": self.updated_at,
        "watermark_run_id": self.watermark_run_id,
        "watermark_created_at": self.watermark_created_at,
        "rebuilt": self.rebuilt,
        "collected": {str(run_id): sorted(attempts) for run_id, attempts in sorted(self.collected.items()) if attempts},
        "incomplete": {str(run_id): sorted(attempts) for run_id, attempts in sorted(self.incomplete.items()) if attempts},
        "pending": [entry.to_json() for _, entry in sorted(self.pending.items())],
    }

  @classmethod
  def from_json(cls, payload: Mapping[str, Any]) -> State:
    """Rebuilds the state from `state.json`."""
    version = _as_int(payload.get("v"))
    if version is None:
      raise StoreError(f"state has a non-integer v={payload.get('v')!r}.")
    if version > STATE_VERSION:
      raise StoreError(f"state was written with schema version {version}; this collector understands {STATE_VERSION}.")
    return cls(
        collected=_attempt_index(payload.get("collected"), "collected"),
        incomplete=_attempt_index(payload.get("incomplete"), "incomplete"),
        pending={entry.identity: entry for entry in _pending_list(payload.get("pending"))},
        watermark_run_id=_as_int(payload.get("watermark_run_id")),
        watermark_created_at=_as_str(payload.get("watermark_created_at")),
        rebuilt=bool(payload.get("rebuilt", False)),
        updated_at=_as_str(payload.get("updated_at")),
        v=version,
    )


def _attempt_index(payload: Any, what: str) -> dict[int, set[int]]:
  """Reads one of the state file's run id -> attempts maps."""
  if payload is None:
    return {}
  if not isinstance(payload, dict):
    raise StoreError(f"state's {what!r} is a {type(payload).__name__}, not a map of run id to attempts.")
  index: dict[int, set[int]] = {}
  for raw_run_id, raw_attempts in payload.items():
    run_id = _as_int(raw_run_id)
    if run_id is None:
      raise StoreError(f"state's {what!r} has a non-numeric run id {raw_run_id!r}.")
    if not isinstance(raw_attempts, (list, tuple)):
      raise StoreError(f"state's {what!r} run {run_id} holds a {type(raw_attempts).__name__}, not a list of attempts.")
    attempts = {value for value in (_as_int(item) for item in raw_attempts) if value is not None}
    if attempts:
      index[run_id] = attempts
  return index


def _pending_list(payload: Any) -> list[PendingAttempt]:
  """Reads the state file's in-flight list."""
  if payload is None:
    return []
  if not isinstance(payload, (list, tuple)):
    raise StoreError(f"state's 'pending' is a {type(payload).__name__}, not a list.")
  entries: list[PendingAttempt] = []
  for item in payload:
    if not isinstance(item, dict):
      raise StoreError(f"state's 'pending' holds a {type(item).__name__}, not an entry.")
    entries.append(PendingAttempt.from_json(item))
  return entries


class Store:
  """The append-only row store under one output directory."""

  def __init__(self, out_dir: Path | str) -> None:
    """Opens the store at a directory."""
    if out_dir is None or (isinstance(out_dir, str) and not out_dir.strip()):
      raise StoreError("the store needs an output directory; there is no default.")
    resolved = Path(out_dir).expanduser().resolve()
    if resolved == resolved.parent:
      raise StoreError(f"{resolved} is the filesystem root; give the store a directory of its own.")
    if resolved == Path(os.path.expanduser("~")).resolve():
      raise StoreError(f"{resolved} is your home directory; give the store a directory of its own.")
    self._out_dir = resolved
    if (resolved / ".git").exists() and not self._looks_like_store():
      raise StoreError(
          f"{resolved} is a git checkout, not a data store. Point --out at a directory of its own, "
          "such as a ci-metrics folder inside a checkout of the store branch."
      )
    self._state: State | None = None
    self._keys: dict[tuple[str, str], set[str]] = {}
    self._bodies: dict[tuple[str, str], dict[str, str]] = {}

  @property
  def out_dir(self) -> Path:
    """Returns the root of the store, resolved to an absolute path."""
    return self._out_dir

  @property
  def data_dir(self) -> Path:
    """Returns the directory the NDJSON files and `state.json` live in."""
    return self._out_dir / DATA_DIRNAME

  @property
  def views_dir(self) -> Path:
    """Returns the directory the browser's view JSON files live in."""
    return self._out_dir / VIEWS_DIRNAME

  @property
  def pr_views_dir(self) -> Path:
    """Returns the directory the per-pull-request view files live in."""
    return self.views_dir / PR_VIEWS_DIRNAME

  @property
  def state_path(self) -> Path:
    """Returns the path of `state.json`."""
    return self.data_dir / STATE_FILENAME

  def path_for(self, kind: str, month: str) -> Path:
    """Returns the file one kind's rows for one month are stored in."""
    return self.data_dir / f"{check_kind(kind)}-{check_month(month)}{NDJSON_SUFFIX}"

  def months(self, kind: str | None = None) -> list[str]:
    """Lists the months the store has files for."""
    if kind is not None:
      return self._months_of(check_kind(kind))
    found: set[str] = set()
    for name in ROW_KINDS:
      found.update(self._months_of(name))
    return sorted(found)

  def read_month(self, month: str, kinds: Sequence[str] | None = None) -> Iterator[rows.Row]:
    """Yields one month's rows as row objects, in the order the lines were written."""
    check_month(month)
    wanted = set(ROW_KINDS) if kinds is None else {check_kind(item) for item in kinds}
    for kind in ROW_KINDS:
      if kind not in wanted:
        continue
      for path in self._parts(kind, month):
        for _, payload in _iter_payloads(path):
          yield rows.from_json(payload)

  def pending_run_ids(self) -> list[int]:
    """Returns the runs seen but not yet stored, from the index."""
    return sorted({entry.run_id for entry in self.load_state().pending.values()})

  def _months_of(self, kind: str) -> list[str]:
    """Lists the months one kind has a file for."""
    if not self.data_dir.is_dir():
      return []
    found: set[str] = set()
    for path in self.data_dir.glob(f"{kind}-*{NDJSON_SUFFIX}"):
      match = PART_NAME_PATTERN.match(path.name[len(kind) + 1 : -len(NDJSON_SUFFIX)])
      if match:
        found.add(match.group("month"))
      else:
        _warn(f"{path.name} is not named <kind>-YYYY-MM[-partN]{NDJSON_SUFFIX} and was ignored.")
    return sorted(found)

  def _numbered_parts(self, kind: str, month: str) -> list[tuple[int, Path]]:
    """Lists the files one month of one kind is stored in, with their part numbers."""
    if not self.data_dir.is_dir():
      return []
    numbered: list[tuple[int, Path]] = []
    for path in self.data_dir.glob(f"{kind}-{month}*{NDJSON_SUFFIX}"):
      match = PART_NAME_PATTERN.match(path.name[len(kind) + 1 : -len(NDJSON_SUFFIX)])
      if match and match.group("month") == month:
        numbered.append((int(match.group("part") or 1), path))
    return sorted(numbered)

  def _parts(self, kind: str, month: str) -> list[Path]:
    """Lists the files one month of one kind is stored in, part 1 first."""
    return [path for _, path in self._numbered_parts(kind, month)]

  def _part_path(self, kind: str, month: str, number: int) -> Path:
    """Returns the path of one part of a month, whether or not it exists yet."""
    if number == 1:
      return self.path_for(kind, month)
    return self.data_dir / f"{check_kind(kind)}-{check_month(month)}-part{number}{NDJSON_SUFFIX}"

  def refresh(self) -> None:
    """Drops the cached state and the cached per-file key sets."""
    self._state = None
    self._keys.clear()
    self._bodies.clear()

  def load_state(self) -> State:
    """Returns the store's index, rebuilding it from the NDJSON when the file is gone."""
    if self._state is not None:
      return self._state
    path = self.state_path
    if not path.exists():
      self._state = self.rebuild_state()
      return self._state
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      _warn(f"{path} could not be read ({error}); rebuilding the index from the stored rows.")
      self._state = self.rebuild_state()
      return self._state
    if not isinstance(payload, dict):
      _warn(f"{path} holds a {type(payload).__name__}, not a state object; rebuilding the index from the stored rows.")
      self._state = self.rebuild_state()
      return self._state
    self._state = State.from_json(payload)
    return self._state

  def save_state(self, state: State) -> None:
    """Writes the index, atomically, and keeps it as this store's cached state."""
    state.prune()
    state.rebuilt = False
    state.updated_at = rows.utc_now_iso()
    write_json_atomic(self.state_path, state.to_json())
    self._state = state

  def rebuild_state(self) -> State:
    """Rebuilds the index by scanning the stored rows."""
    state = State()
    scanned = 0
    newest_created: str | None = None
    with_jobs = self._attempts_with_jobs()
    for payload in self.read(rows.KIND_RUN):
      run_id = _as_int(payload.get("run_id"))
      attempt = _as_int(payload.get("attempt"))
      if run_id is None or attempt is None:
        _warn(f"a stored run row has no run id and attempt ({dict(list(payload.items())[:4])!r}); skipping it.")
        continue
      created = _as_str(payload.get("created_at"))
      if payload.get("status") != COMPLETED_STATUS:
        state.mark_incomplete(run_id, attempt, created)
      elif (run_id, attempt) in with_jobs:
        state.mark_collected(run_id, attempt, created)
      else:
        state.note_run(run_id, created)
      scanned += 1
      if created and (newest_created is None or created > newest_created):
        newest_created = created
    state.rebuilt = scanned > 0
    if newest_created is not None:
      rewound = parse_timestamp(newest_created)
      if rewound is not None:
        state.watermark_created_at = (rewound - timedelta(hours=REBUILD_REWIND_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return state

  def _attempts_with_jobs(self) -> set[tuple[int, int]]:
    """Returns every (run id, attempt) that has at least one stored job row."""
    found: set[tuple[int, int]] = set()
    for month in self._months_of(rows.KIND_JOB):
      for path in self._parts(rows.KIND_JOB, month):
        for _, payload in _iter_payloads(path):
          run_id = _as_int(payload.get("run_id"))
          attempt = _as_int(payload.get("attempt"))
          if run_id is not None and attempt is not None:
            found.add((run_id, attempt))
    return found

  def has_attempt(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt's rows are already stored."""
    return self.load_state().has_attempt(run_id, attempt)

  def append(
      self,
      kind: str,
      records: Sequence[Any],
      month: str | None = None,
      correction: bool = False,
  ) -> int:
    """Appends rows to their month's file, skipping the ones already stored."""
    check_kind(kind)
    if not records:
      return 0
    batches: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for record in records:
      payload = self._as_payload(record, kind)
      target = check_month(month) if month is not None else self._month_of(payload, kind)
      batches.setdefault(target, []).append((row_key(payload), payload))
    written = 0
    for target in sorted(batches):
      written += self._append_month(kind, target, batches[target], correction)
    return written

  def read(self, kind: str, months: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Reads one kind back, with corrections applied."""
    check_kind(kind)
    wanted = self.months(kind) if months is None else sorted(check_month(item) for item in months)
    payloads, _ = self._winners(kind, [path for item in wanted for path in self._parts(kind, item)])
    return payloads

  def read_rows(self, kind: str, months: Sequence[str] | None = None) -> list[Any]:
    """Reads one kind back as row objects."""
    return [rows.from_json(payload) for payload in self.read(kind, months)]

  def prune_months(self, before: str, dry_run: bool = False) -> list[Path]:
    """Deletes every data file of the months before `before`: every kind, every part."""
    check_month(before)
    old_months = {(kind, month) for kind in ROW_KINDS for month in self._months_of(kind) if month < before}
    doomed = sorted(path for kind, month in old_months for path in self._parts(kind, month))
    if dry_run:
      return doomed
    for path in doomed:
      try:
        path.unlink()
      except OSError as error:
        raise StoreError(f"{path} could not be deleted: {error}") from error
    for key in old_months:
      self._keys.pop(key, None)
      self._bodies.pop(key, None)
    return doomed

  def compact_month(
      self,
      kind: str,
      month: str,
      merged_run_ids: Iterable[int],
      allow_open_month: bool = False,
      allow_dropping_all: bool = False,
  ) -> int:
    """Rewrites one closed month's file down to what has to be kept."""
    check_kind(kind)
    check_month(month)
    if not allow_open_month and month >= month_key(utc_now()):
      raise StoreError(
          f"{month} has not closed yet, so compacting it could race a tick that is appending to it. "
          "Pass allow_open_month=True only when nothing is writing."
      )
    numbered = self._numbered_parts(kind, month)
    if not numbered:
      return 0
    paths = [path for _, path in numbered]
    survivors, total = self._winners(kind, paths)
    if kind == rows.KIND_TEST:
      keep = {value for value in (_as_int(item) for item in merged_run_ids) if value is not None}
      if not keep and survivors and not allow_dropping_all:
        raise StoreError(
            f"compacting {kind}-{month} with an empty keep list would delete all {len(survivors)} test rows, and their "
            "artifacts are long gone. Pass the merged pull request runs and the daily scheduled main runs, or "
            "allow_dropping_all=True if the month really holds nothing worth keeping."
        )
      survivors = [payload for payload in survivors if _as_int(payload.get("run_id")) in keep]
    removed = total - len(survivors)
    if removed == 0 and all(_complete_length(path) == path.stat().st_size for path in paths):
      return 0
    # Write the new parts first and delete the leftovers last. A crash in between then leaves
    # rows twice, which `read` already handles, and never loses one.
    groups = _split_into_parts([f"{_encode(payload)}\n" for payload in survivors], MAX_PART_BYTES)
    for number, group in enumerate(groups, start=1):
      write_text_atomic(self._part_path(kind, month, number), "".join(group))
    for number, path in numbered:
      if number > len(groups):
        path.unlink()
    self._keys.pop((kind, month), None)
    self._bodies.pop((kind, month), None)
    return removed

  def write_view(self, name: str, payload: Any) -> Path:
    """Writes one view file under `views/`, atomically."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
      raise StoreError(f"view name {name!r} has to stay inside {VIEWS_DIRNAME}/.")
    if relative.suffix != JSON_SUFFIX:
      raise StoreError(f"view name {name!r} has to end in {JSON_SUFFIX}.")
    path = self.views_dir / relative
    write_json_atomic(path, payload)
    return path

  def _looks_like_store(self) -> bool:
    """Says whether this directory already holds a store, so the checkout guard can stand down."""
    return self.data_dir.is_dir() or self.state_path.is_file()

  def _as_payload(self, record: Any, kind: str) -> dict[str, Any]:
    """Turns one appended record into the JSON object that gets written."""
    if isinstance(record, dict):
      given = record.get("kind")
      if given is None:
        raise StoreError(f"a {kind} payload carries no 'kind'; append rows built by rows.to_json.")
      if given != kind:
        raise StoreError(f"a {given!r} row was appended to {kind!r}; one call writes one kind.")
      return dict(record)
    try:
      given = rows.row_kind(record)
    except rows.RowError as error:
      raise StoreError(f"{type(record).__name__} is not a stored row: {error}") from error
    if given != kind:
      raise StoreError(f"a {given!r} row was appended to {kind!r}; one call writes one kind.")
    return rows.to_json(record)

  def _month_of(self, payload: Mapping[str, Any], kind: str) -> str:
    """Works out which month a row belongs to when the caller passed none."""
    field_name = MONTH_FIELD.get(kind)
    if field_name is None:
      raise StoreError(
          f"a {kind} row carries no timestamp of its own, so pass month=<the run's month> to append. "
          "Use store.month_for_run(run) to get it."
      )
    value = payload.get(field_name)
    if value is None:
      raise StoreError(f"a {kind} row has no {field_name!r}, so pass month= to append.")
    return month_key(value)

  def _append_month(
      self,
      kind: str,
      month: str,
      items: Sequence[tuple[str, dict[str, Any]]],
      correction: bool,
  ) -> int:
    """Appends one kind's rows to one month's file."""
    mutable = kind in MUTABLE_KINDS
    fresh: list[tuple[str, dict[str, Any]]] = []
    batch: dict[str, str] = {}
    for key, payload in items:
      body = _body(payload) if mutable else ""
      if not correction:
        if key in batch and batch[key] == body:
          continue
        if self._state_has_row(kind, payload):
          continue
        if mutable:
          if self._known_bodies(kind, month).get(key) == body:
            continue
        elif key in self._known_keys(kind, month):
          continue
      batch[key] = body
      fresh.append((key, payload))
    if not fresh:
      return 0
    self._append_lines(kind, month, [f"{_encode(payload)}\n" for _, payload in fresh])
    known = self._keys.get((kind, month))
    if known is not None:
      known.update(key for key, _ in fresh)
    bodies = self._bodies.get((kind, month))
    if bodies is not None:
      bodies.update((key, _body(payload)) for key, payload in fresh)
    return len(fresh)

  def _state_has_row(self, kind: str, payload: Mapping[str, Any]) -> bool:
    """Says whether the index already accounts for this row, without reading any file."""
    if kind not in ATTEMPT_KINDS:
      return False
    run_id = _as_int(payload.get("run_id"))
    attempt = _as_int(payload.get("attempt"))
    if run_id is None or attempt is None:
      return False
    return self.load_state().has_attempt(run_id, attempt)

  def _known_keys(self, kind: str, month: str) -> set[str]:
    """Returns every key already in one month's file, reading it at most once per process."""
    cached = self._keys.get((kind, month))
    if cached is not None:
      return cached
    keys = {row_key(payload) for path in self._parts(kind, month) for _, payload in _iter_payloads(path)}
    self._keys[(kind, month)] = keys
    return keys

  def _known_bodies(self, kind: str, month: str) -> dict[str, str]:
    """Returns the current content of every key in a mutable kind's month file."""
    cached = self._bodies.get((kind, month))
    if cached is not None:
      return cached
    bodies: dict[str, str] = {}
    for path in self._parts(kind, month):
      for _, payload in _iter_payloads(path):
        bodies[row_key(payload)] = _body(payload)
    self._bodies[(kind, month)] = bodies
    return bodies

  def sweep_temp(self) -> int:
    """Removes temporary files a killed process left in the data directory."""
    return _sweep_temp(self.data_dir)

  def _append_lines(self, kind: str, month: str, lines: Sequence[str]) -> None:
    """Adds lines to the end of a month, starting a new part whenever the current one is full."""
    numbered = self._numbered_parts(kind, month)
    number, path = numbered[-1] if numbered else (1, self.path_for(kind, month))
    used = _complete_length(path) if path.exists() else 0
    for offset, group in enumerate(_split_into_parts(lines, MAX_PART_BYTES, used)):
      if group:
        self._append_text(self._part_path(kind, month, number + offset), "".join(group))

  def _append_text(self, path: Path, block: str) -> None:
    """Adds a block of complete lines to a file, atomically."""
    _ensure_dir(path.parent)
    carry_over = 0
    if path.exists():
      carry_over = _complete_length(path)
      size = path.stat().st_size
      if carry_over < size:
        _warn(f"{path.name} ended in a torn line of {size - carry_over} bytes; it was dropped before appending.")
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=TEMP_PREFIX, suffix=NDJSON_SUFFIX)
    temp_path = Path(temp_name)
    try:
      with os.fdopen(handle, "wb") as out:
        if carry_over:
          with path.open("rb") as current:
            remaining = carry_over
            while remaining > 0:
              chunk = current.read(min(COPY_CHUNK_BYTES, remaining))
              if not chunk:
                break
              out.write(chunk)
              remaining -= len(chunk)
        out.write(block.encode("utf-8"))
        out.flush()
        os.fsync(out.fileno())
      os.replace(temp_path, path)
    except OSError as error:
      temp_path.unlink(missing_ok=True)
      raise StoreError(f"{path} could not be appended to: {error}") from error
    except BaseException:
      temp_path.unlink(missing_ok=True)
      raise
    _fsync_dir(path.parent)

  def _winners(self, kind: str, paths: Sequence[Path]) -> tuple[list[dict[str, Any]], int]:
    """Applies the correction rule across a set of files."""
    best: dict[str, tuple[tuple[str, int, int], dict[str, Any]]] = {}
    total = 0
    for file_index, path in enumerate(paths):
      for line_index, payload in _iter_payloads(path):
        total += 1
        if payload.get("kind") != kind:
          raise StoreError(f"{path} line {line_index + 1} holds a {payload.get('kind')!r} row, not {kind!r}.")
        key = row_key(payload)
        order = (str(payload.get("collected_at") or ""), file_index, line_index)
        current = best.get(key)
        if current is None or order > current[0]:
          best[key] = (order, payload)
    ranked = sorted(best.values(), key=lambda item: (item[0][1], item[0][2]))
    return [payload for _, payload in ranked], total


def _split_into_parts(lines: Sequence[str], limit: int, used: int = 0) -> list[list[str]]:
  """Groups lines into parts of at most `limit` bytes, never splitting a line."""
  groups: list[list[str]] = [[]]
  for line in lines:
    size = len(line.encode("utf-8"))
    if used and used + size > limit:
      groups.append([])
      used = 0
    groups[-1].append(line)
    used += size
  return groups


def _body(payload: Mapping[str, Any]) -> str:
  """Returns a row's content without the stamp that changes on every tick."""
  return json.dumps(
      {name: value for name, value in payload.items() if name != STAMP_FIELD},
      sort_keys=True,
      ensure_ascii=False,
      separators=(",", ":"),
  )


def _encode(payload: Mapping[str, Any]) -> str:
  """Turns one row into the line that is written."""
  try:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  except (TypeError, ValueError) as error:
    raise StoreError(f"a {payload.get('kind')!r} row could not be serialised: {error}") from error
