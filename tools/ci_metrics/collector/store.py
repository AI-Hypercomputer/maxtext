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

"""The append-only store: the collector's own copy of everything it has already read.

This is layer 3's foundation. `rows.py` says what a record looks like; this module decides
where it lives, whether it is already there, and how a reader gets the current version of it
back. It touches no network and computes no metric.

The store lives under one directory, given to `Store` and never defaulted:

    <out>/data/<kind>-YYYY-MM.ndjson   one JSON object a line, appended, never edited
    <out>/data/state.json              which attempts are stored, which are in flight
    <out>/views/<name>-YYYY-MM.json    what the browser loads, rebuilt each tick
    <out>/views/pr/<n>.json            one merged pull request in full

Five rules hold the whole design up.

1. **Append-only, with corrections.** A line is never edited and never deleted in place. When
   a number turns out to be wrong - an artifact that was unreadable at 04:00 and readable at
   08:00, a run that gained an attempt - the fix is a second line with the same key and a
   later `collected_at`. Readers take the last row per key, ties broken by file order and
   then line order. That is what `read` does, so nothing downstream has to remember the rule.
   One kind corrects itself: a rescue is keyed by the failure and not by the outcome, so a
   row whose answer has changed is written without the caller asking (see `MUTABLE_KINDS`).

2. **Writes are atomic per file.** Every write goes to a temporary file in the same
   directory, is flushed and fsynced, and is then moved into place with `os.replace`. A
   process killed mid-tick - the Actions job cancelled, the runner pulled - leaves either the
   old file or the new one, never a half-written line and never a truncated `state.json`. It
   can leave the temporary file itself behind, which `Store.sweep_temp` clears on a later run.
   An append copies the file it is extending, so it costs one pass over that month's file:
   pass a tick's rows in as few `append` calls as you can rather than one call per run.

3. **`state.json` is an index, not the truth.** It records run attempts, never individual
   rows: indexing every test row would reach a hundred megabytes and be rewritten six times a
   day. It keeps the newest `MAX_INDEXED_RUNS` run ids and drops the rest on save, so the
   file the collector commits stops growing. If it is lost or corrupt, `load_state` rebuilds
   it by scanning the NDJSON, and says so with `State.rebuilt`. The rebuild cannot recover the in-flight list, because an
   attempt that was still running was never written - so a rebuilt state rewinds its
   watermark by `REBUILD_REWIND_HOURS` and the next tick re-asks that window. Re-asking is
   cheap: `append` skips what is already stored.

4. **Missing is None, never zero.** The store passes values through exactly as `rows.py`
   built them. It never fills a gap, never coerces a None to 0, and never drops a null field.

5. **A month closes once.** A tick appends only to the month of the run it is reading, so a
   past month stops changing. `compact_month` is then run once against the closed month: it
   keeps the last row per key, and for test rows it drops the detail of runs whose pull
   request never merged, whose per-flavor totals survive in the suite rows. Running it a
   second time changes nothing and rewrites nothing.

The store never imports `github.py`, `runs.py` or `junit.py`, so it can be exercised with no
network module present and no `requests` installed. Nothing here calls an AI service: a row
goes in the way `rows.py` built it and comes back out the same.
"""

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

# Read sizes. The copy chunk bounds an append's memory; the tail window is how far back the
# last newline of a file is looked for.
COPY_CHUNK_BYTES = 1 << 20
TAIL_WINDOW_BYTES = 1 << 16


class StoreError(RuntimeError):
  """Raised when the store cannot be read, written or addressed.

  The message always names the file, the key or the month at fault, so a collector tick that
  dies says which part of the store broke it. A row that cannot be shaped stays
  `rows.RowError`; a request that failed stays `github.GitHubError`.
  """


def _warn(message: str) -> None:
  """Prints a warning to stderr so it never lands in piped collector output.

  Args:
    message: The line to print.
  """
  print(message, file=sys.stderr, flush=True)


def parse_timestamp(value: Any) -> datetime | None:
  """Reads one of GitHub's ISO-8601 timestamps.

  A third copy of this parser, next to the ones in `runs.py` and `derive.py`. It is repeated
  rather than imported so that the store depends on `rows.py` alone and stays importable with
  no network module and no `requests` installed.

  Args:
    value: An ISO-8601 string, a datetime, or anything else.

  Returns:
    A timezone-aware UTC datetime, or None when the value is not a timestamp.
  """
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
  """Puts a datetime in UTC, treating a naive one as UTC already.

  Args:
    moment: Any datetime.

  Returns:
    The same instant, timezone-aware, in UTC.
  """
  if moment.tzinfo is None:
    return moment.replace(tzinfo=timezone.utc)
  return moment.astimezone(timezone.utc)


def utc_now() -> datetime:
  """Returns the current instant in UTC.

  Returns:
    A timezone-aware UTC datetime.
  """
  return datetime.now(timezone.utc)


def month_key(value: Any) -> str:
  """Turns a timestamp into the month string its rows are filed under.

  The month is always taken in UTC, so a run created at 01:00 on the first of the month in a
  +03:00 zone still files under the previous month, which is when GitHub says it happened.

  Args:
    value: An ISO-8601 timestamp or a datetime.

  Returns:
    The month, e.g. "2026-09".

  Raises:
    StoreError: The value is not a timestamp.
  """
  moment = parse_timestamp(value)
  if moment is None:
    raise StoreError(f"{value!r} is not a timestamp, so it names no month.")
  return moment.strftime(MONTH_FORMAT)


def month_for_run(run: Mapping[str, Any]) -> str:
  """Returns the month every row of a run belongs in.

  A run's rows are filed together under the run's own `created_at`, not under each row's
  timestamp: a re-run started three days later must not scatter one run across two files, or
  a closed month would keep changing.

  Args:
    run: A run payload from the API, or a stored run row as JSON.

  Returns:
    The month, e.g. "2026-09".

  Raises:
    StoreError: The payload carries no readable `created_at`.
  """
  created = run.get("created_at")
  if created is None:
    raise StoreError(f"run {run.get('id', run.get('run_id'))!r} has no 'created_at', so it names no month.")
  return month_key(created)


def _key_part(value: object) -> str:
  """Encodes one part of a key, exactly as `rows.py` does.

  Args:
    value: The part. None becomes an empty part, which is what an unknown worker number is.

  Returns:
    The value percent-encoded, leaving only unreserved ASCII characters as they were.
  """
  if value is None:
    return ""
  return quote(str(value), safe="")


def row_key(payload: Mapping[str, Any]) -> str:
  """Builds the store key of a stored JSON row.

  Read straight off the payload's fields rather than by rebuilding the row, so that a line
  written before a field was added is still addressable. The result is identical to the row
  object's own `key()`.

  Args:
    payload: A row as `rows.to_json` wrote it, carrying "kind".

  Returns:
    The key, e.g. "job|33468578834|1|99733460534".

  Raises:
    StoreError: The payload has no known "kind", or no value in its leading key field.
  """
  kind = str(payload.get("kind", ""))
  fields = KEY_FIELDS.get(kind)
  if fields is None:
    raise StoreError(f"row kind {payload.get('kind')!r} is not one of {sorted(KEY_FIELDS)}.")
  if payload.get(fields[0]) is None:
    raise StoreError(f"{kind} row has no {fields[0]!r}, so it cannot be keyed.")
  return rows.KEY_SEPARATOR.join([kind, *(_key_part(payload.get(name)) for name in fields)])


def check_kind(kind: str) -> str:
  """Checks that a kind is one this store files.

  Args:
    kind: One of the `rows.KIND_*` constants.

  Returns:
    The kind, unchanged.

  Raises:
    StoreError: The kind is not stored.
  """
  if kind not in KEY_FIELDS:
    raise StoreError(f"{kind!r} is not a stored row kind; it is one of {list(ROW_KINDS)}.")
  return kind


def check_month(month: str) -> str:
  """Checks that a month string is a real "YYYY-MM".

  Args:
    month: The month to check.

  Returns:
    The month, unchanged.

  Raises:
    StoreError: The string is not a month.
  """
  if not isinstance(month, str) or not MONTH_PATTERN.match(month):
    raise StoreError(f"{month!r} is not a month in the form YYYY-MM.")
  return month


def write_bytes_atomic(path: Path, data: bytes) -> None:
  """Writes a file so that a killed process cannot leave it half written.

  The bytes land in a temporary file beside the target, are flushed and fsynced, and are then
  moved onto the target with `os.replace`, which is atomic within one filesystem. A reader
  therefore sees either the whole previous file or the whole new one.

  Args:
    path: The file to write.
    data: Its entire new contents.

  Raises:
    StoreError: The directory could not be created or the file could not be written.
  """
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
  """Writes text atomically, UTF-8 encoded.

  Args:
    path: The file to write.
    text: Its entire new contents.

  Raises:
    StoreError: The file could not be written.
  """
  write_bytes_atomic(path, text.encode("utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
  """Writes one JSON document atomically.

  Args:
    path: The file to write.
    payload: Anything `json.dumps` accepts.

  Raises:
    StoreError: The payload does not serialise, or the file could not be written.
  """
  try:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  except (TypeError, ValueError) as error:
    raise StoreError(f"{path} could not be serialised: {error}") from error
  write_text_atomic(path, f"{text}\n")


def _ensure_dir(path: Path) -> None:
  """Creates a directory and its parents if they are not there yet.

  Args:
    path: The directory.

  Raises:
    StoreError: The directory could not be created.
  """
  try:
    path.mkdir(parents=True, exist_ok=True)
  except OSError as error:
    raise StoreError(f"{path} could not be created: {error}") from error


def _fsync_dir(path: Path) -> None:
  """Flushes a directory entry so a rename survives a power cut.

  Best effort. Platforms that will not open a directory - Windows - are left alone, because
  the rename itself is already atomic there.

  Args:
    path: The directory to flush.
  """
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
  """Deletes temporary files an earlier process left behind when it was killed.

  An atomic write lands in `<dir>/.tmp-*` and is then renamed. A process killed between the
  two leaves the temporary file in place: harmless to read - `months` and `read` only look at
  the real suffixes - but it is a full copy of that month's rows, in the directory the
  collector commits, and nothing else ever removes it.

  Only files older than `TEMP_MAX_AGE_HOURS` are touched, so a write that is happening right
  now in another process is never pulled out from under it.

  Args:
    directory: The directory to sweep. Missing is not an error.
    now: The current time as a POSIX timestamp, for tests.

  Returns:
    How many files were deleted.
  """
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
  """Returns how many bytes of a file end in a complete line.

  Every write here is atomic, so a torn last line should be impossible. It is measured
  anyway, because a file truncated by something outside the collector - a full disk, a killed
  copy - must not be able to glue its broken tail onto the next row.

  Args:
    path: The file to measure.

  Returns:
    The offset just past the last newline, or 0 when the file holds no newline at all.
  """
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
  """Reads one NDJSON file, line by line.

  Args:
    path: The file. A file that does not exist yields nothing, because a month with no rows
      and a month that was never written are the same fact.

  Yields:
    (line index counted from 0, the decoded object).

  Raises:
    StoreError: A line in the middle of the file is not a JSON object. A broken last line
      that has no newline after it is a torn write: it is reported and skipped, because the
      rest of the file is still good.
  """
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
  """A run attempt the collector has seen but cannot store yet, because it is still running.

  Attributes:
    run_id: The run.
    attempt: The attempt number, from the payload's `run_attempt`.
    first_seen_at: When the collector first put it on this list, ISO-8601 UTC. The 24-hour
      rule is measured from here, not from the run's own start, so a run discovered late is
      still given a full day.
    created_at: The run's `created_at`, kept so a tick can file it in the right month without
      re-reading the run.
    status: The status GitHub last reported, e.g. "in_progress" or "queued".
  """

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
    """Rebuilds an entry from `state.json`.

    Args:
      payload: One element of the file's "pending" list.

    Returns:
      The entry.

    Raises:
      StoreError: The entry carries no usable run id or attempt number.
    """
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
  """Reads a value as an int, or None when it is not one.

  Args:
    value: Anything.

  Returns:
    The int, or None. A bool is not an int here: True is not run 1.
  """
  if isinstance(value, bool) or value is None:
    return None
  if isinstance(value, int):
    return value
  try:
    return int(str(value))
  except (TypeError, ValueError):
    return None


def _as_str(value: Any) -> str | None:
  """Reads a value as a string, keeping None as None.

  Args:
    value: Anything.

  Returns:
    The string, or None.
  """
  return None if value is None else str(value)


@dataclass
class State:
  """The index of what the store already holds, and what it is still waiting for.

  It records run attempts and nothing finer. A collected attempt implies its jobs, suites and
  tests: indexing every test row would reach a hundred megabytes and be rewritten six times a
  day for no benefit, since rows of one attempt are always written together.

  Attributes:
    collected: run id -> the attempts stored in full, from a run GitHub had finished.
    incomplete: run id -> the attempts written while they were still running, because they sat
      in `pending` longer than a day. They are stored, so they are never fetched again by
      accident, but they are not final.
    pending: (run id, attempt) -> the attempt, still in flight, nothing written for it yet.
    watermark_run_id: The highest run id seen. The tick's "everything after this" mark.
    watermark_created_at: That run's `created_at`. The runs endpoint filters by creation time,
      not by id, so this is the value a tick actually queries with.
    rebuilt: True when this state was reconstructed by scanning the NDJSON because
      `state.json` was missing or unreadable, and the scan found rows. The in-flight list
      cannot be reconstructed, so a caller seeing this should widen its window instead of
      trusting the watermark exactly. A brand new store is not flagged: it has nothing to
      recover and no watermark at all.
    updated_at: When the state was last saved, ISO-8601 UTC.
    v: Schema version of the state file.
  """

  collected: dict[int, set[int]] = field(default_factory=dict)
  incomplete: dict[int, set[int]] = field(default_factory=dict)
  pending: dict[tuple[int, int], PendingAttempt] = field(default_factory=dict)
  watermark_run_id: int | None = None
  watermark_created_at: str | None = None
  rebuilt: bool = False
  updated_at: str | None = None
  v: int = STATE_VERSION

  def has_attempt(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt's rows are in the store already, final or not.

    Args:
      run_id: The run.
      attempt: The attempt number.

    Returns:
      True when the attempt was written, whether completed or written incomplete.
    """
    return self.is_collected(run_id, attempt) or self.is_incomplete(run_id, attempt)

  def is_collected(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt was stored from a run GitHub had finished.

    Args:
      run_id: The run.
      attempt: The attempt number.

    Returns:
      True when the attempt is in `collected`.
    """
    return attempt in self.collected.get(run_id, ())

  def is_incomplete(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt was stored while it was still running.

    Args:
      run_id: The run.
      attempt: The attempt number.

    Returns:
      True when the attempt is in `incomplete`.
    """
    return attempt in self.incomplete.get(run_id, ())

  def mark_collected(self, run_id: int, attempt: int, created_at: str | None = None) -> None:
    """Records an attempt as stored in full.

    Call this after every row of the attempt has been appended, never before: an attempt
    marked collected is skipped by `Store.append`, so marking it early loses rows.

    Args:
      run_id: The run.
      attempt: The attempt number.
      created_at: The run's `created_at`, which also moves the watermark.
    """
    self.collected.setdefault(run_id, set()).add(attempt)
    self.incomplete.get(run_id, set()).discard(attempt)
    self._drop_empty(run_id)
    self.drop_pending(run_id, attempt)
    self.note_run(run_id, created_at)

  def mark_incomplete(self, run_id: int, attempt: int, created_at: str | None = None) -> None:
    """Records an attempt as written while it was still running.

    The two indexes are exclusive, so this clears the attempt from `collected` exactly as
    `mark_collected` clears it from `incomplete`. An attempt in both would be counted twice
    by `attempt_count` and would contradict itself in `state.json`.

    Args:
      run_id: The run.
      attempt: The attempt number.
      created_at: The run's `created_at`, which also moves the watermark.
    """
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
    """Puts an attempt on the in-flight list, or leaves it there.

    An attempt already in the store is not added: it has an answer, so it is not waiting for
    one. An attempt already pending keeps its original `first_seen_at`, so the 24-hour clock
    is not reset by every tick that sees it again.

    Args:
      run_id: The run.
      attempt: The attempt number.
      created_at: The run's `created_at`.
      status: The status GitHub last reported.
      first_seen_at: When it was first seen; defaults to now.

    Returns:
      True when the entry is on the list after this call.
    """
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
    """Takes an attempt off the in-flight list.

    Args:
      run_id: The run.
      attempt: The attempt number.

    Returns:
      True when something was removed.
    """
    return self.pending.pop((run_id, attempt), None) is not None

  def forget_attempt(self, run_id: int, attempt: int) -> bool:
    """Un-records an attempt so its rows can be harvested again.

    The rows already in the store stay there; this only clears the index entry that would
    make `Store.append` skip them. Use it to re-read an attempt whose rows were wrong. To add
    a corrected version alongside the old one instead, append with `correction=True`.

    Args:
      run_id: The run.
      attempt: The attempt number.

    Returns:
      True when the attempt had been recorded.
    """
    was_known = self.has_attempt(run_id, attempt)
    self.collected.get(run_id, set()).discard(attempt)
    self.incomplete.get(run_id, set()).discard(attempt)
    self._drop_empty(run_id)
    return was_known

  def note_run(self, run_id: int, created_at: str | None = None) -> None:
    """Moves the watermark forward if this run is newer than everything seen so far.

    Args:
      run_id: The run.
      created_at: That run's `created_at`, if it is known.
    """
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
    """Returns the in-flight attempts that have waited too long.

    A tick writes what it knows for each of these once, marks it incomplete and stops waiting,
    so one stuck run cannot block the store forever.

    Args:
      now: The moment to measure from; defaults to the current time.
      max_age_hours: How long an attempt may stay in flight.

    Returns:
      The expired entries, oldest first. An entry whose `first_seen_at` cannot be read counts
      as expired, because a timestamp nobody can compare would keep it on the list for ever.
    """
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
    """Drops the oldest run ids from the index so `state.json` cannot grow forever.

    The file is rewritten and committed on every tick, six times a day, so an index that only
    ever grows adds to the committed history for the life of the project. Run ids rise with
    time, so keeping the highest `keep` of them keeps everything recent. Dropping an id costs
    nothing: the watermark means it is never listed again, and if it ever is, `append` falls
    back to the month file's own keys, which is the exact answer rather than a cached one.

    In-flight attempts are never pruned - they are the one thing a scan cannot rebuild.

    Args:
      keep: How many run ids to keep.

    Returns:
      How many run ids were dropped.
    """
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
    """Removes a run id whose attempt set has emptied, so the file does not grow husks.

    Args:
      run_id: The run to check.
    """
    for index in (self.collected, self.incomplete):
      if run_id in index and not index[run_id]:
        del index[run_id]

  def to_json(self) -> dict[str, Any]:
    """Returns the state as it is stored.

    Attempt sets become sorted lists under string run ids, because JSON has neither sets nor
    integer keys, and sorting keeps the committed file's diff small.

    Returns:
      A JSON-safe dict.
    """
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
    """Rebuilds the state from `state.json`.

    Args:
      payload: The parsed file.

    Returns:
      The state.

    Raises:
      StoreError: The file was written by a newer collector, or its shape is not readable. A
        newer version is refused rather than read wrong, the same rule `rows.from_json` uses.
    """
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
  """Reads one of the state file's run id -> attempts maps.

  Args:
    payload: The stored map.
    what: Which map it is, for the error message.

  Returns:
    run id -> the set of attempt numbers.

  Raises:
    StoreError: The map is not a map, or holds something that is not a run id and attempts.
  """
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
  """Reads the state file's in-flight list.

  Args:
    payload: The stored list.

  Returns:
    The entries.

  Raises:
    StoreError: The list is not a list, or an entry cannot be read.
  """
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
  """The append-only row store under one output directory.

  One instance per output directory per process. It caches the state and, per file it has had
  to read, the set of keys that file already holds, so a tick parses each month's file at most
  once. `refresh` drops both caches.

  Nothing is written until something is appended: constructing a Store for reading creates no
  directories.
  """

  def __init__(self, out_dir: Path | str) -> None:
    """Opens the store at a directory.

    There is no default and no fallback to the working directory. A path is refused outright
    when it is empty, when it is the filesystem root or the user's home directory, or when it
    holds a `.git` directory and is not already a store - writing a data store into somebody's
    checkout by accident has to be impossible, and those are the three ways it happens.

    Args:
      out_dir: The output directory. It does not have to exist yet.

    Raises:
      StoreError: The path is missing, empty or one of the refused places.
    """
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
    """Returns the file one kind's rows for one month are stored in.

    Args:
      kind: One of the `rows.KIND_*` constants.
      month: "YYYY-MM".

    Returns:
      The path, e.g. `<out>/data/test-2026-09.ndjson`.

    Raises:
      StoreError: The kind is not stored, or the month is not a month.
    """
    return self.data_dir / f"{check_kind(kind)}-{check_month(month)}{NDJSON_SUFFIX}"

  def months(self, kind: str | None = None) -> list[str]:
    """Lists the months the store has files for.

    Args:
      kind: One of the `rows.KIND_*` constants, or None for every month any kind has a file
        for. `months()` with no argument is what a view builder wants: it asks which months
        exist at all.

    Returns:
      The months, oldest first. Empty when nothing has been written.

    Raises:
      StoreError: The kind is not stored.
    """
    if kind is not None:
      return self._months_of(check_kind(kind))
    found: set[str] = set()
    for name in ROW_KINDS:
      found.update(self._months_of(name))
    return sorted(found)

  def read_month(self, month: str, kinds: Sequence[str] | None = None) -> Iterator[rows.Row]:
    """Yields one month's rows as row objects, in the order the lines were written.

    Nothing is deduplicated here: every line comes out, corrections included, so a caller that
    wants to apply the correction rule itself can. `read` is the deduplicated view of the same
    data. Calling this twice for one month is fine - each call opens the files again, which is
    what a two-pass view builder needs.

    Args:
      month: "YYYY-MM".
      kinds: The kinds to yield, from `rows.KIND_*`. None yields every kind. Whatever order
        they are given in, the kinds come out in `ROW_KINDS` order, and each kind's rows come
        out in line order.

    Yields:
      The stored rows.

    Raises:
      StoreError: The month or a kind is malformed, or a file could not be read.
      rows.RowError: A stored line does not match the current row schema.
    """
    check_month(month)
    wanted = set(ROW_KINDS) if kinds is None else {check_kind(item) for item in kinds}
    for kind in ROW_KINDS:
      if kind not in wanted:
        continue
      for _, payload in _iter_payloads(self.path_for(kind, month)):
        yield rows.from_json(payload)

  def pending_run_ids(self) -> list[int]:
    """Returns the runs seen but not yet stored, from the index.

    This is meta.json's uncollected count, as run ids rather than a number.

    Returns:
      The run ids, ascending, each once however many attempts of it are in flight.

    Raises:
      StoreError: The index could not be loaded.
    """
    return sorted({entry.run_id for entry in self.load_state().pending.values()})

  def _months_of(self, kind: str) -> list[str]:
    """Lists the months one kind has a file for.

    Args:
      kind: One of the `rows.KIND_*` constants, already checked.

    Returns:
      The months, oldest first.
    """
    if not self.data_dir.is_dir():
      return []
    found: list[str] = []
    for path in self.data_dir.glob(f"{kind}-*{NDJSON_SUFFIX}"):
      month = path.name[len(kind) + 1 : -len(NDJSON_SUFFIX)]
      if MONTH_PATTERN.match(month):
        found.append(month)
      else:
        _warn(f"{path.name} is not named <kind>-YYYY-MM{NDJSON_SUFFIX} and was ignored.")
    return sorted(found)

  def refresh(self) -> None:
    """Drops the cached state and the cached per-file key sets.

    Call it when something outside this process has written to the store, and in tests
    between cases that share a directory.
    """
    self._state = None
    self._keys.clear()
    self._bodies.clear()

  def load_state(self) -> State:
    """Returns the store's index, rebuilding it from the NDJSON when the file is gone.

    The state is cached: the same object comes back on every call, so a caller can mark
    attempts on it and save once at the end of a tick. A `state.json` written by a newer
    collector is an error; a missing or unreadable one is not, because it is an index and the
    NDJSON is the truth.

    Returns:
      The state. `State.rebuilt` is True when it came from a scan rather than from the file.

    Raises:
      StoreError: `state.json` was written by a newer schema version.
    """
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
    """Writes the index, atomically, and keeps it as this store's cached state.

    Two things happen on the way out. The oldest run ids beyond `MAX_INDEXED_RUNS` are
    dropped, so the committed file stops growing once the project is a year old. And
    `rebuilt` is cleared: the flag says "this index came from a scan, so widen your window",
    and once the scan's result has been written back, the file is the index again and the
    next tick has nothing to distrust.

    Args:
      state: The state to write. Its `updated_at` is stamped here.

    Raises:
      StoreError: The file could not be written.
    """
    state.prune()
    state.rebuilt = False
    state.updated_at = rows.utc_now_iso()
    write_json_atomic(self.state_path, state.to_json())
    self._state = state

  def rebuild_state(self) -> State:
    """Rebuilds the index by scanning the stored rows.

    Every stored attempt has a run row, so the run files say which attempts exist. They do
    not say whether the attempt was finished: a tick writes its run row first, so a tick that
    died straight afterwards leaves a run row with no jobs behind it. Marking that attempt
    "collected" would make the next tick skip it, and its jobs, suites and tests would never
    be stored - the rows would be lost with nothing said. So the job files are scanned too,
    and an attempt is only collected when its jobs are there as well. An attempt that really
    held no job is re-read once and then settles; re-reading costs requests, not rows.

    An attempt whose run row reports a status other than "completed" was written while it was
    still running and comes back as incomplete, whether or not it has jobs.

    Two things a scan cannot recover, both of them stated in the result rather than guessed:
    the in-flight list, because nothing was ever written for those attempts, and an exact
    watermark, because a run that was in flight can have an id below the newest stored one. So
    the rebuilt watermark timestamp is rewound by `REBUILD_REWIND_HOURS` and `rebuilt` is set,
    and the next tick re-asks that window. Re-asking costs API calls and nothing else: `append`
    skips every row it already has.

    Returns:
      The reconstructed state. `rebuilt` is set when the scan actually found rows; a store
      with nothing in it comes back empty and unflagged, because a first tick has nothing to
      recover and no watermark to distrust.

    Raises:
      StoreError: A run file could not be read.
    """
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
    """Returns every (run id, attempt) that has at least one stored job row.

    Read line by line rather than through `read`, because a rebuild happens over the whole
    store and the job files are the biggest thing in it after the tests.

    Returns:
      The attempts whose jobs are on disk.

    Raises:
      StoreError: A job file could not be read.
    """
    found: set[tuple[int, int]] = set()
    for month in self._months_of(rows.KIND_JOB):
      for _, payload in _iter_payloads(self.path_for(rows.KIND_JOB, month)):
        run_id = _as_int(payload.get("run_id"))
        attempt = _as_int(payload.get("attempt"))
        if run_id is not None and attempt is not None:
          found.add((run_id, attempt))
    return found

  def has_attempt(self, run_id: int, attempt: int) -> bool:
    """Says whether an attempt's rows are already stored.

    Args:
      run_id: The run.
      attempt: The attempt number.

    Returns:
      True when the index has it, whether it was completed or written incomplete.

    Raises:
      StoreError: The index could not be loaded.
    """
    return self.load_state().has_attempt(run_id, attempt)

  def append(
      self,
      kind: str,
      records: Sequence[Any],
      month: str | None = None,
      correction: bool = False,
  ) -> int:
    """Appends rows to their month's file, skipping the ones already stored.

    A row is skipped when the index says its attempt is stored, or when its key is already in
    the target file. Neither check applies with `correction=True`, which is how a fixed row is
    added: it is appended with its own later `collected_at`, the old line stays where it is,
    and `read` returns the new one.

    A rescue row is the one exception to the key rule, because its key names the failure and
    not the outcome. It is compared on its content instead: an identical row is skipped, and
    one whose answer has changed - "never re-run" becoming "rescued" - is written without the
    caller having to ask for a correction.

    An append copies the file it extends, so pass a batch: one call per kind per tick costs one
    pass over that month's file, one call per run costs one pass per run.

    Args:
      kind: One of the `rows.KIND_*` constants. Every record must be of this kind.
      records: Row objects from `rows.py`, or the JSON dicts `rows.to_json` produces.
      month: The month to file them under, "YYYY-MM". Leave it out only for run and rescue
        rows, which carry a timestamp that is inside their run's own month. Job, suite and
        test rows need it passed: `store.month_for_run(run)`. Passing it explicitly is what
        keeps a run's rows in one file when a re-run lands in the next month.
      correction: Write every row even if its key is stored already.

    Returns:
      How many rows were actually written.

    Raises:
      StoreError: The kind is unknown, a record is of another kind, a record cannot be keyed,
        a month cannot be worked out, or the file could not be written.
    """
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
    """Reads one kind back, with corrections applied.

    The correction rule lives here so nothing downstream has to remember it: of the lines
    sharing a key, the one with the greatest `collected_at` wins, and when those tie the later
    line in the later file wins.

    Args:
      kind: One of the `rows.KIND_*` constants.
      months: The months to read, or None for every month in the store. A month with no file
        contributes nothing rather than failing.

    Returns:
      The winning row of each key, as the JSON dicts they were stored as, in the order the
      winning lines appear in the store. Nothing is coerced: a null field stays null.

    Raises:
      StoreError: The kind is unknown, a month is malformed, or a file is unreadable.
    """
    check_kind(kind)
    wanted = self.months(kind) if months is None else sorted(check_month(item) for item in months)
    payloads, _ = self._winners(kind, [self.path_for(kind, item) for item in wanted])
    return payloads

  def read_rows(self, kind: str, months: Sequence[str] | None = None) -> list[Any]:
    """Reads one kind back as row objects.

    The typed view of `read`. It goes through `rows.from_json`, which insists a stored line
    carries exactly the fields its dataclass declares, so this raises where `read` would hand
    back a row written under a different field set.

    Args:
      kind: One of the `rows.KIND_*` constants.
      months: The months to read, or None for every month.

    Returns:
      The winning rows, in the same order as `read`.

    Raises:
      StoreError: The kind is unknown or a file is unreadable.
      rows.RowError: A stored line does not match the current row schema.
    """
    return [rows.from_json(payload) for payload in self.read(kind, months)]

  def compact_month(
      self,
      kind: str,
      month: str,
      merged_run_ids: Iterable[int],
      allow_open_month: bool = False,
      allow_dropping_all: bool = False,
  ) -> int:
    """Rewrites one closed month's file down to what has to be kept.

    Two things are dropped, and nothing else:

      * superseded lines - of the rows sharing a key, only the last one survives;
      * for test rows, every row of a run that is not in `merged_run_ids`. Their per-flavor
        totals are not lost: those live in the suite rows, which compaction never touches.
        This is the rule that keeps the store inside its size budget.

    **`merged_run_ids` is the keep list, and it is wider than its name.** It has to hold the
    merged pull request runs AND the scheduled main runs that were kept at full resolution -
    one a day, the daily history the dashboard's long-range charts are drawn from. A
    scheduled run has no pull request at all, so leaving those ids out quietly deletes a
    month of daily history that cannot be fetched again: the JUnit artifacts expired the day
    after the run.

    It is idempotent by construction. A second run finds no duplicate keys and no droppable
    test rows, removes nothing, and leaves the file untouched - it does not even rewrite it,
    so the month's file stops changing in the committed history.

    Args:
      kind: One of the `rows.KIND_*` constants.
      month: The month to compact, "YYYY-MM".
      merged_run_ids: The runs whose per-test detail is kept. Only consulted for test rows;
        for every other kind this call only drops superseded lines.
      allow_open_month: Compact a month that has not closed yet. Off by default, because a
        tick may be appending to the current month while this runs.
      allow_dropping_all: Let an empty keep list wipe a month of test rows. Off by default,
        because an empty set is almost always a caller that failed to load its merged runs,
        and the rows it deletes cannot be fetched again.

    Returns:
      How many lines were removed. Zero means there was nothing to do.

    Raises:
      StoreError: The kind or month is malformed, the month is still open, the keep list is
        empty for a month that holds test rows, or the file could not be read or written.
    """
    check_kind(kind)
    check_month(month)
    if not allow_open_month and month >= month_key(utc_now()):
      raise StoreError(
          f"{month} has not closed yet, so compacting it could race a tick that is appending to it. "
          "Pass allow_open_month=True only when nothing is writing."
      )
    path = self.path_for(kind, month)
    if not path.exists():
      return 0
    survivors, total = self._winners(kind, [path])
    if kind == rows.KIND_TEST:
      keep = {value for value in (_as_int(item) for item in merged_run_ids) if value is not None}
      if not keep and survivors and not allow_dropping_all:
        raise StoreError(
            f"compacting {path.name} with an empty keep list would delete all {len(survivors)} test rows, and their "
            "artifacts are long gone. Pass the merged pull request runs and the daily scheduled main runs, or "
            "allow_dropping_all=True if the month really holds nothing worth keeping."
        )
      survivors = [payload for payload in survivors if _as_int(payload.get("run_id")) in keep]
    removed = total - len(survivors)
    if removed == 0 and _complete_length(path) == path.stat().st_size:
      return 0
    body = "".join(f"{_encode(payload)}\n" for payload in survivors)
    write_text_atomic(path, body)
    self._keys.pop((kind, month), None)
    self._bodies.pop((kind, month), None)
    return removed

  def write_view(self, name: str, payload: Any) -> Path:
    """Writes one view file under `views/`, atomically.

    The browser fetches these, so a half-written one would break the dashboard rather than the
    collector. The name may carry the `pr/` sub-directory and nothing else: no absolute path,
    no walking upwards.

    Args:
      name: The file name relative to `views/`, e.g. "runs-2026-09.json" or "pr/5070.json".
      payload: Anything `json.dumps` accepts.

    Returns:
      The path written.

    Raises:
      StoreError: The name escapes `views/`, does not end in ".json", or the write failed.
    """
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
      raise StoreError(f"view name {name!r} has to stay inside {VIEWS_DIRNAME}/.")
    if relative.suffix != JSON_SUFFIX:
      raise StoreError(f"view name {name!r} has to end in {JSON_SUFFIX}.")
    path = self.views_dir / relative
    write_json_atomic(path, payload)
    return path

  def _looks_like_store(self) -> bool:
    """Says whether this directory already holds a store, so the checkout guard can stand down.

    Returns:
      True when `data/` or `data/state.json` is already there.
    """
    return self.data_dir.is_dir() or self.state_path.is_file()

  def _as_payload(self, record: Any, kind: str) -> dict[str, Any]:
    """Turns one appended record into the JSON object that gets written.

    Args:
      record: A row object from `rows.py`, or a JSON dict it produced.
      kind: The kind the caller said it is appending.

    Returns:
      The payload, carrying "kind".

    Raises:
      StoreError: The record is of another kind, or is not a row at all.
    """
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
    """Works out which month a row belongs to when the caller passed none.

    Args:
      payload: The row.
      kind: Its kind.

    Returns:
      The month, "YYYY-MM".

    Raises:
      StoreError: The kind carries no timestamp of its own, or its timestamp is missing.
    """
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
    """Appends one kind's rows to one month's file.

    Args:
      kind: The kind.
      month: The month.
      items: (key, payload) pairs, in the order they should be written.
      correction: Write every row even if its key is stored already.

    Returns:
      How many rows were written.

    Raises:
      StoreError: The file could not be read or written.
    """
    path = self.path_for(kind, month)
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
    self._append_text(path, "".join(f"{_encode(payload)}\n" for _, payload in fresh))
    known = self._keys.get((kind, month))
    if known is not None:
      known.update(key for key, _ in fresh)
    bodies = self._bodies.get((kind, month))
    if bodies is not None:
      bodies.update((key, _body(payload)) for key, payload in fresh)
    return len(fresh)

  def _state_has_row(self, kind: str, payload: Mapping[str, Any]) -> bool:
    """Says whether the index already accounts for this row, without reading any file.

    The cheap half of the dedup. It answers only for the kinds that belong to one attempt: a
    rescue spans attempts, so it always falls through to the file's keys.

    Args:
      kind: The row's kind.
      payload: The row.

    Returns:
      True when the row's attempt is recorded as stored.
    """
    if kind not in ATTEMPT_KINDS:
      return False
    run_id = _as_int(payload.get("run_id"))
    attempt = _as_int(payload.get("attempt"))
    if run_id is None or attempt is None:
      return False
    return self.load_state().has_attempt(run_id, attempt)

  def _known_keys(self, kind: str, month: str) -> set[str]:
    """Returns every key already in one month's file, reading it at most once per process.

    Args:
      kind: The kind.
      month: The month.

    Returns:
      The keys. Empty when the file does not exist.

    Raises:
      StoreError: The file could not be read.
    """
    cached = self._keys.get((kind, month))
    if cached is not None:
      return cached
    path = self.path_for(kind, month)
    keys = {row_key(payload) for _, payload in _iter_payloads(path)}
    self._keys[(kind, month)] = keys
    return keys

  def _known_bodies(self, kind: str, month: str) -> dict[str, str]:
    """Returns the current content of every key in a mutable kind's month file.

    Only the kinds in `MUTABLE_KINDS` need this. The value is the row with its `collected_at`
    left out, so two writes of an unchanged rescue compare equal while a rescue that has
    since been re-run does not. Later lines win, which is the same correction rule `read`
    applies.

    Args:
      kind: The kind.
      month: The month.

    Returns:
      key -> that key's latest stored content. Empty when the file does not exist.

    Raises:
      StoreError: The file could not be read.
    """
    cached = self._bodies.get((kind, month))
    if cached is not None:
      return cached
    path = self.path_for(kind, month)
    bodies: dict[str, str] = {}
    for _, payload in _iter_payloads(path):
      bodies[row_key(payload)] = _body(payload)
    self._bodies[(kind, month)] = bodies
    return bodies

  def sweep_temp(self) -> int:
    """Removes temporary files a killed process left in the data directory.

    Called once when a tick opens the store. See `_sweep_temp` for why the age guard is there.

    Returns:
      How many files were removed.
    """
    return _sweep_temp(self.data_dir)

  def _append_text(self, path: Path, block: str) -> None:
    """Adds a block of complete lines to a file, atomically.

    The existing bytes are copied into a temporary file, the block goes on the end, and the
    temporary file replaces the target. A killed process therefore leaves the file exactly as
    it was. If the existing file ends in a torn line - which this writer cannot produce, but
    a full disk or an outside truncation can - that line is dropped and reported rather than
    having the new rows glued onto it.

    Args:
      path: The NDJSON file.
      block: The lines to add, each already ending in a newline.

    Raises:
      StoreError: The file could not be read or written.
    """
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
    """Applies the correction rule across a set of files.

    Args:
      kind: The kind being read, for the error messages.
      paths: The files, in the order they should be ranked - oldest month first.

    Returns:
      (the winning payloads, in the order their winning lines appear; how many lines were read
      in total, which is what compaction subtracts from).

    Raises:
      StoreError: A file could not be read, or a line could not be keyed.
    """
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


def _body(payload: Mapping[str, Any]) -> str:
  """Returns a row's content without the stamp that changes on every tick.

  Two ticks that re-derive the same unchanged rescue produce rows that differ only in
  `collected_at`. Comparing on this instead of on the whole payload is what lets the store
  skip the repeat while still writing the row whose answer has changed.

  Args:
    payload: The row.

  Returns:
    A canonical string of every field except `collected_at`.
  """
  return json.dumps(
      {name: value for name, value in payload.items() if name != STAMP_FIELD},
      sort_keys=True,
      ensure_ascii=False,
      separators=(",", ":"),
  )


def _encode(payload: Mapping[str, Any]) -> str:
  """Turns one row into the line that is written.

  Field order is left exactly as the caller built it - `rows.to_json` emits "kind" first and
  then the dataclass's own order - so the committed file diffs line by line instead of
  reshuffling on every write.

  Args:
    payload: The row.

  Returns:
    One line of JSON, with no trailing newline.

  Raises:
    StoreError: The row does not serialise.
  """
  try:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  except (TypeError, ValueError) as error:
    raise StoreError(f"a {payload.get('kind')!r} row could not be serialised: {error}") from error
