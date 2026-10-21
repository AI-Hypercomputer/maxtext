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

"""Exact GCS HTTP QPS and POSIX/GCSFuse `pread64` tracer for MaxText pipelines.

Provides:
1. `extract_arrayrecord_footer_offset`:
   Parses the trailing 64 KiB `RiegeliPostscript` block of any `.array_record`
   shard to determine the exact byte boundary `footer_offset` separating user
   data chunks (`[0, footer_offset)`) from the `ArrayRecordFooter` index chunk
   and `RiegeliPostscript` (`[footer_offset, file_size)`).
2. `InstrumentedGCSServer`:
   A multi-threaded HTTP/1.1 server compatible with
   `CLOUD_STORAGE_EMULATOR_ENDPOINT` (C++ `google::cloud::storage::Client` /
   `riegeli::gcs::GcsReader`) and `STORAGE_EMULATOR_HOST` (Python
   `google-cloud-storage`). Records every HTTP request immediately upon arrival,
   tracks live `transferred_bytes`, and classifies every GET deterministically
   by exact byte offset into:
   - `list_or_meta`
   - `init_probe_shard_open` (`ArrayRecordDataSource.__init__` probe)
   - `lazy_worker_shard_open` (`_ensure_reader_exists` per-worker reader open)
   - `record_read` (`[0, footer_offset)` data chunk read)
3. `PosixGCSFuseTracer`:
   Attaches `strace -f -qq -s 0 -ttt -y -e trace=pread64` across the process
   tree (including all PyGrain `mp_prefetch` worker processes) to capture the
   exact POSIX `pread64(fd<path>, count, offset)` syscalls issued by C++
   `riegeli::FdReader` on `.array_record` files, and replays them through a
   stateful GCSFuse streaming read-ahead and local file-cache state machine.
"""

import ctypes
import dataclasses
import fnmatch
from http import server as http_server
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Optional
from urllib import parse as urllib_parse

_RIEGELI_BLOCK_SIZE = 65536
_ARRAY_RECORD_MAGIC = 0x71930E704FDAE05E


def _encode_varint(val: int) -> bytes:
  out = bytearray()
  while val > 0x7F:
    out.append((val & 0x7F) | 0x80)
    val >>= 7
  out.append(val)
  return bytes(out)


_MAGIC_PROTO_TAG_BYTES = b"\x10" + _encode_varint(_ARRAY_RECORD_MAGIC)
_FOOTER_OFFSET_CACHE: dict[str, int] = {}
_FOOTER_CACHE_LOCK = threading.Lock()


def extract_arrayrecord_footer_offset(file_path: str) -> int:
  """Returns the exact byte offset where ArrayRecordFooter starts in `file_path`."""
  abs_path = os.path.abspath(file_path)
  with _FOOTER_CACHE_LOCK:
    if abs_path in _FOOTER_OFFSET_CACHE:
      return _FOOTER_OFFSET_CACHE[abs_path]

  try:
    size = os.path.getsize(abs_path)
    if size < _RIEGELI_BLOCK_SIZE:
      return size
    postscript_offset = size - _RIEGELI_BLOCK_SIZE
    with open(abs_path, "rb") as f:
      f.seek(postscript_offset)
      tail = f.read(_RIEGELI_BLOCK_SIZE)
    idx = tail.find(_MAGIC_PROTO_TAG_BYTES)
    if idx != -1:
      # In RiegeliPostscript proto, field 1 (0x08, footer_offset) precedes magic
      sub = tail[max(0, idx - 12) : idx]
      t_idx = sub.rfind(b"\x08")
      if t_idx != -1:
        val = 0
        shift = 0
        for b in sub[t_idx + 1 :]:
          val |= (b & 0x7F) << shift
          if not (b & 0x80):
            break
          shift += 7
        if 0 < val < size:
          with _FOOTER_CACHE_LOCK:
            _FOOTER_OFFSET_CACHE[abs_path] = val
          return val
    with _FOOTER_CACHE_LOCK:
      _FOOTER_OFFSET_CACHE[abs_path] = postscript_offset
    return postscript_offset
  except OSError:
    return 0


def _compute_peak_window_gets(
    timestamps: list[float], window_s: float = 1.0
) -> int:
  """Returns the maximum number of requests within any `window_s` interval."""
  if not timestamps:
    return 0
  sorted_ts = sorted(timestamps)
  max_count = 0
  left = 0
  for right, ts in enumerate(sorted_ts):
    while ts - sorted_ts[left] > window_s:
      left += 1
    count = right - left + 1
    if count > max_count:
      max_count = count
  return max_count


@dataclasses.dataclass
class GCSRequestRecord:
  """Metadata for a single HTTP request received by the GCS endpoint."""

  timestamp: float
  phase: str  # "init", "first_batch", "warmup", "steady"
  step_idx: int
  method: str
  object_name: str
  range_header: Optional[str]
  range_type: str  # "list_or_meta", "full_get", "unbounded_range", etc.
  request_category: str
  start_offset: int
  end_offset: int
  footer_offset: int
  file_size: int
  requested_bytes: int
  transferred_bytes: int


class InstrumentedGCSServer:
  """Local multi-threaded GCS HTTP server for exact C++ GcsReader QPS tracing."""

  def __init__(self, root_dir: str, bucket_name: str = "qps-bench"):
    self.root_dir = os.path.abspath(root_dir)
    self.bucket_name = bucket_name
    self._lock = threading.Lock()
    self._phase = "init"
    self._step_idx = -1
    self.requests: list[GCSRequestRecord] = []
    self._server: Optional[http_server.ThreadingHTTPServer] = None
    self._thread: Optional[threading.Thread] = None

  def set_phase(self, phase: str, step_idx: int = -1) -> None:
    """Updates the active execution phase and step index for request tagging."""
    with self._lock:
      self._phase = phase
      self._step_idx = step_idx

  def reset_stats(self) -> None:
    """Clears all recorded HTTP requests and resets phase to `init`."""
    with self._lock:
      self.requests.clear()
      self._phase = "init"
      self._step_idx = -1

  def _record(self, rec: GCSRequestRecord) -> None:
    with self._lock:
      self.requests.append(rec)

  def _add_transferred(self, rec: GCSRequestRecord, nbytes: int) -> None:
    with self._lock:
      rec.transferred_bytes += nbytes

  def _resolve_local_path(self, object_name: str) -> str:
    clean_rel = os.path.normpath(object_name).lstrip(os.sep)
    return os.path.join(self.root_dir, clean_rel)

  def _list_matching_objects(
      self, prefix: str = "", match_glob: str = ""
  ) -> list[dict[str, Any]]:
    """Lists local files under `root_dir` matching `prefix` and `match_glob`."""
    items = []
    for dirpath, _, filenames in os.walk(self.root_dir):
      for fname in sorted(filenames):
        full_path = os.path.join(dirpath, fname)
        rel_path = os.path.relpath(full_path, self.root_dir).replace(
            os.sep, "/"
        )
        if prefix and not rel_path.startswith(prefix):
          continue
        if match_glob and not fnmatch.fnmatch(rel_path, match_glob):
          continue
        size = os.path.getsize(full_path)
        items.append({
            "kind": "storage#object",
            "id": f"{self.bucket_name}/{rel_path}/1",
            "name": rel_path,
            "bucket": self.bucket_name,
            "generation": "1",
            "size": str(size),
        })
    return items

  def start(self) -> str:
    """Starts the background HTTP server and configures GCS emulator env vars."""
    outer = self

    class QuietThreadingHTTPServer(http_server.ThreadingHTTPServer):
      """ThreadingHTTPServer that suppresses BrokenPipeError from aborted GETs."""

      daemon_threads = True

      def handle_error(self, request, client_address):
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
          return
        super().handle_error(request, client_address)

    class Handler(http_server.BaseHTTPRequestHandler):
      """HTTP handler emulating the GCS JSON API with byte-offset classification."""

      protocol_version = "HTTP/1.1"

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

      def handle(self):
        try:
          super().handle()
        except (BrokenPipeError, ConnectionResetError):
          pass

      def do_GET(self):  # pylint: disable=invalid-name
        """Handles GCS list, object metadata, and media Range GET requests."""
        parsed = urllib_parse.urlparse(self.path)
        query = urllib_parse.parse_qs(parsed.query)
        path = parsed.path
        now = time.perf_counter()
        with outer._lock:  # pylint: disable=protected-access
          phase = outer._phase  # pylint: disable=protected-access
          step_idx = outer._step_idx  # pylint: disable=protected-access

        # 1. List objects endpoint: /storage/v1/b/<bucket>/o
        if path.rstrip("/") == f"/storage/v1/b/{outer.bucket_name}/o":
          prefix = query.get("prefix", [""])[0]
          match_glob = query.get("matchGlob", [""])[0]
          items = outer._list_matching_objects(  # pylint: disable=protected-access
              prefix=prefix, match_glob=match_glob
          )
          body = json.dumps({"kind": "storage#objects", "items": items}).encode(
              "utf-8"
          )
          outer._record(  # pylint: disable=protected-access
              GCSRequestRecord(
                  timestamp=now,
                  phase=phase,
                  step_idx=step_idx,
                  method="GET",
                  object_name="__list__",
                  range_header=None,
                  range_type="list_or_meta",
                  request_category="list_or_meta",
                  start_offset=0,
                  end_offset=0,
                  footer_offset=0,
                  file_size=0,
                  requested_bytes=0,
                  transferred_bytes=len(body),
              )
          )
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
          return

        marker = f"/b/{outer.bucket_name}/o/"
        xml_marker = f"/{outer.bucket_name}/"
        if marker in path:
          object_name = urllib_parse.unquote(path.split(marker, 1)[1])
        elif path.startswith(xml_marker):
          object_name = urllib_parse.unquote(path[len(xml_marker) :])
        else:
          self.send_error(404, f"Unknown path: {path}")
          return
        local_path = outer._resolve_local_path(object_name)  # pylint: disable=protected-access
        if not os.path.isfile(local_path):
          self.send_error(404, f"Object not found: {object_name}")
          return

        file_size = os.path.getsize(local_path)
        footer_offset = extract_arrayrecord_footer_offset(local_path)
        alt = query.get("alt", [""])[0]

        # 2. Object metadata JSON request
        if alt != "media" and "/download/" not in path and marker in path:
          meta = {
              "kind": "storage#object",
              "id": f"{outer.bucket_name}/{object_name}/1",
              "name": object_name,
              "bucket": outer.bucket_name,
              "generation": "1",
              "metageneration": "1",
              "size": str(file_size),
              "contentType": "application/octet-stream",
          }
          body = json.dumps(meta).encode("utf-8")
          outer._record(  # pylint: disable=protected-access
              GCSRequestRecord(
                  timestamp=now,
                  phase=phase,
                  step_idx=step_idx,
                  method="GET",
                  object_name=object_name,
                  range_header=None,
                  range_type="list_or_meta",
                  request_category="list_or_meta",
                  start_offset=0,
                  end_offset=0,
                  footer_offset=footer_offset,
                  file_size=file_size,
                  requested_bytes=0,
                  transferred_bytes=len(body),
              )
          )
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
          return

        # 3. Media download request (`?alt=media`)
        rng = self.headers.get("Range") or self.headers.get("range")
        start = 0
        end = file_size - 1
        status = 200
        range_type = "full_get"

        if rng and rng.startswith("bytes="):
          spec = rng.split("=", 1)[1].strip()
          p0, p1 = spec.split("-", 1)
          if not p0:
            suffix_len = int(p1)
            start = max(0, file_size - suffix_len)
            end = file_size - 1
            range_type = "suffix_range"
          elif not p1:
            start = min(int(p0), max(0, file_size - 1))
            end = file_size - 1
            range_type = "unbounded_range"
          else:
            start = min(int(p0), max(0, file_size - 1))
            end = min(int(p1), max(0, file_size - 1))
            range_type = "bounded_range"
          status = 206

        is_tail_footer_read = start >= footer_offset or (
            footer_offset > 0
            and end >= file_size - 1
            and start >= max(0, footer_offset - 65536)
        )
        if (
            phase == "init"
            or range_type in ("full_get", "suffix_range")
            or is_tail_footer_read
        ):
          request_category = (
              "init_probe_shard_open"
              if phase == "init"
              else "lazy_worker_shard_open"
          )
        else:
          request_category = "record_read"

        requested_bytes = max(0, end - start + 1)
        rec = GCSRequestRecord(
            timestamp=now,
            phase=phase,
            step_idx=step_idx,
            method="GET",
            object_name=object_name,
            range_header=rng,
            range_type=range_type,
            request_category=request_category,
            start_offset=start,
            end_offset=end,
            footer_offset=footer_offset,
            file_size=file_size,
            requested_bytes=requested_bytes,
            transferred_bytes=0,
        )
        outer._record(rec)  # pylint: disable=protected-access

        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(requested_bytes))
        self.send_header("x-goog-generation", "1")
        self.send_header("x-goog-metageneration", "1")
        self.send_header("x-goog-stored-content-length", str(file_size))
        if status == 206:
          self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        chunk_size = 64 * 1024
        try:
          with open(local_path, "rb") as f:
            f.seek(start)
            remaining = requested_bytes
            while remaining > 0:
              buf = f.read(min(chunk_size, remaining))
              if not buf:
                break
              self.wfile.write(buf)
              outer._add_transferred(rec, len(buf))  # pylint: disable=protected-access
              remaining -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
          self.close_connection = True

    self._server = QuietThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self._thread = threading.Thread(
        target=self._server.serve_forever, daemon=True
    )
    self._thread.start()
    endpoint = f"http://127.0.0.1:{self._server.server_port}"
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = endpoint
    os.environ["STORAGE_EMULATOR_HOST"] = endpoint
    return endpoint

  def stop(self) -> None:
    """Stops the background HTTP server."""
    if self._server is not None:
      self._server.shutdown()
      self._server.server_close()
      self._server = None

  def summarize(self) -> dict[str, Any]:
    """Aggregates request counts and transferred bytes by phase and category."""
    with self._lock:
      reqs = list(self.requests)

    media_reqs = [r for r in reqs if r.request_category != "list_or_meta"]
    meta_reqs = [r for r in reqs if r.request_category == "list_or_meta"]

    init_probe_reqs = [
        r for r in media_reqs if r.request_category == "init_probe_shard_open"
    ]
    lazy_open_reqs = [
        r for r in media_reqs if r.request_category == "lazy_worker_shard_open"
    ]
    shard_open_reqs = init_probe_reqs + lazy_open_reqs
    record_read_reqs = [
        r for r in media_reqs if r.request_category == "record_read"
    ]

    init_only_media = [r for r in media_reqs if r.phase == "init"]
    first_batch_media = [r for r in media_reqs if r.phase == "first_batch"]
    init_and_first_media = [
        r for r in media_reqs if r.phase in ("init", "first_batch")
    ]
    warmup_media = [r for r in media_reqs if r.phase == "warmup"]
    steady_media = [r for r in media_reqs if r.phase == "steady"]
    active_iter_media = [r for r in media_reqs if r.phase != "init"]
    steady_record_reads = [
        r for r in steady_media if r.request_category == "record_read"
    ]
    steady_lazy_opens = [
        r
        for r in steady_media
        if r.request_category == "lazy_worker_shard_open"
    ]

    step_counts: dict[tuple[str, int], int] = {}
    steady_step_counts: dict[int, int] = {}
    for r in active_iter_media:
      key = (r.phase, r.step_idx)
      step_counts[key] = step_counts.get(key, 0) + 1
      if r.phase == "steady":
        steady_step_counts[r.step_idx] = (
            steady_step_counts.get(r.step_idx, 0) + 1
        )

    peak_step_gets = max(step_counts.values()) if step_counts else 0
    steady_peak_step_gets = (
        max(steady_step_counts.values()) if steady_step_counts else 0
    )
    peak_1s_gets = _compute_peak_window_gets(
        [r.timestamp for r in media_reqs], window_s=1.0
    )
    peak_1s_active_gets = _compute_peak_window_gets(
        [r.timestamp for r in active_iter_media], window_s=1.0
    )

    unique_shards = sorted({r.object_name for r in media_reqs})
    by_type: dict[str, int] = {}
    for r in media_reqs:
      by_type[r.range_type] = by_type.get(r.range_type, 0) + 1

    by_category: dict[str, int] = {}
    for r in media_reqs:
      by_category[r.request_category] = (
          by_category.get(r.request_category, 0) + 1
      )

    return {
        "total_http_requests": len(reqs),
        "metadata_or_list_requests": len(meta_reqs),
        "total_media_get_requests": len(media_reqs),
        "init_probe_shard_open_gets": len(init_probe_reqs),
        "lazy_worker_shard_open_gets": len(lazy_open_reqs),
        "exact_shard_open_gets": len(shard_open_reqs),
        "exact_record_read_gets": len(record_read_reqs),
        "init_only_gets": len(init_only_media),
        "first_batch_gets": len(first_batch_media),
        "init_and_first_batch_gets": len(init_and_first_media),
        "warmup_gets": len(warmup_media),
        "steady_total_gets": len(steady_media),
        "steady_record_read_gets": len(steady_record_reads),
        "steady_lazy_shard_open_gets": len(steady_lazy_opens),
        "peak_step_gets": peak_step_gets,
        "steady_peak_step_gets": steady_peak_step_gets,
        "peak_1s_gets": peak_1s_gets,
        "peak_1s_active_gets": peak_1s_active_gets,
        "unique_shards_touched": len(unique_shards),
        "range_type_breakdown": by_type,
        "request_category_breakdown": by_category,
        "total_requested_bytes": sum(r.requested_bytes for r in media_reqs),
        "total_transferred_bytes": sum(r.transferred_bytes for r in media_reqs),
        "record_read_requested_bytes": sum(
            r.requested_bytes for r in record_read_reqs
        ),
        "record_read_transferred_bytes": sum(
            r.transferred_bytes for r in record_read_reqs
        ),
    }


@dataclasses.dataclass
class PreadEvent:
  """A single POSIX `pread64` syscall on an `.array_record` shard."""

  pid: int
  timestamp: float
  fd: int
  file_path: str
  offset: int
  requested_bytes: int
  returned_bytes: int
  footer_offset: int
  file_size: int
  category: str  # "init_probe_shard_open", "lazy_worker_shard_open", etc.
  phase: str  # "init", "first_batch", "warmup", "steady"
  step_idx: int = -1


_PREAD_LINE_RE = re.compile(
    r"^\s*(?:(?:\[pid\s+)?(\d+)\]?\s+)?(\d+\.\d+)\s+"
    r"pread64\((\d+)<([^>]+)>,\s*.*,\s*(\d+),\s*(\d+)\s*\)\s*=\s*(\d+)"
)
_PREAD_UNFINISHED_RE = re.compile(
    r"^(\s*(?:(?:\[pid\s+)?(\d+)\]?\s+)?\d+\.\d+\s+pread64\(.*?)<unfinished"
    r" \.\.\.>"
)
_PREAD_RESUMED_RE = re.compile(
    r"^\s*(?:(?:\[pid\s+)?(\d+)\]?\s+)?\d+\.\d+\s+<\.\.\.\s*pread64"
    r" resumed>(.*)$"
)


class PosixGCSFuseTracer:
  """Traces C++ `FdReader` `pread64` syscalls and replays GCSFuse read-ahead."""

  def __init__(self, target_dir: str, read_ahead_kb: int = 1024):
    self.target_dir = os.path.abspath(target_dir)
    self.read_ahead_bytes = read_ahead_kb * 1024
    self._proc: Optional[subprocess.Popen[bytes]] = None
    self._trace_file: Optional[Any] = None
    self.phase_timestamps: list[tuple[float, str, int]] = []

  def mark_phase(self, phase: str, step_idx: int = -1) -> None:
    """Records the wall-clock timestamp when a new benchmark phase begins."""
    self.phase_timestamps.append((time.time(), phase, step_idx))

  def start(self) -> None:
    """Unlocks ptrace_scope and attaches `strace -f -e trace=pread64`."""
    try:
      ctypes.CDLL(None).prctl(0x59616D61, -1, 0, 0, 0)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    self._trace_file = tempfile.NamedTemporaryFile(
        mode="r", suffix=".strace", delete=False
    )
    my_pid = os.getpid()
    self._proc = subprocess.Popen(
        [
            "strace",
            "-f",
            "-qq",
            "-s",
            "0",
            "-ttt",
            "-y",
            "-e",
            "trace=pread64",
            "-o",
            self._trace_file.name,
            "-p",
            str(my_pid),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 2.0
    while time.time() < deadline:
      try:
        with open(f"/proc/{my_pid}/status", "r", encoding="utf-8") as sf:
          status_text = sf.read()
        if "\nTracerPid:\t0\n" not in status_text:
          break
      except OSError:
        break
      time.sleep(0.02)
    time.sleep(0.05)
    self.phase_timestamps = [(time.time(), "init", -1)]

  def _resolve_phase(self, ts: float) -> tuple[str, int]:
    phase = "init"
    step_idx = -1
    for p_ts, p_name, s_idx in self.phase_timestamps:
      if ts >= p_ts:
        phase = p_name
        step_idx = s_idx
      else:
        break
    return phase, step_idx

  def stop_and_summarize(self) -> dict[str, Any]:
    """Stops `strace` and replays `pread64` events through GCSFuse state."""
    if self._proc is not None:
      time.sleep(0.05)
      self._proc.terminate()
      try:
        self._proc.wait(timeout=3.0)
      except subprocess.TimeoutExpired:
        self._proc.kill()
      self._proc = None

    events: list[PreadEvent] = []
    if self._trace_file is not None and os.path.exists(self._trace_file.name):
      pending_by_pid: dict[int, str] = {}
      with open(
          self._trace_file.name, "r", encoding="utf-8", errors="replace"
      ) as f:
        for raw_line in f:
          um = _PREAD_UNFINISHED_RE.match(raw_line)
          if um:
            u_pid = int(um.group(2)) if um.group(2) else os.getpid()
            pending_by_pid[u_pid] = um.group(1)
            continue
          rm = _PREAD_RESUMED_RE.match(raw_line)
          if rm:
            r_pid = int(rm.group(1)) if rm.group(1) else os.getpid()
            prefix = pending_by_pid.pop(r_pid, None)
            if prefix is None:
              continue
            line = prefix + rm.group(2)
          else:
            line = raw_line

          if ".array_record" not in line:
            continue
          m = _PREAD_LINE_RE.match(line)
          if not m:
            continue
          (
              pid_str,
              ts_str,
              fd_str,
              path_str,
              req_str,
              off_str,
              ret_str,
          ) = m.groups()
          abs_path = os.path.abspath(path_str)
          if not abs_path.startswith(self.target_dir):
            continue
          pid = int(pid_str) if pid_str else os.getpid()
          ts = float(ts_str)
          fd = int(fd_str)
          req_b = int(req_str)
          off = int(off_str)
          ret_b = int(ret_str)
          f_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
          footer_off = extract_arrayrecord_footer_offset(abs_path)
          phase, step_idx = self._resolve_phase(ts)
          is_tail_footer_read = off >= footer_off or (
              footer_off > 0
              and (off + ret_b) >= f_size
              and off >= max(0, footer_off - 65536)
          )
          if phase == "init" or is_tail_footer_read:
            cat = (
                "init_probe_shard_open"
                if phase == "init"
                else "lazy_worker_shard_open"
            )
          else:
            cat = "record_read"
          events.append(
              PreadEvent(
                  pid=pid,
                  timestamp=ts,
                  fd=fd,
                  file_path=abs_path,
                  offset=off,
                  requested_bytes=req_b,
                  returned_bytes=ret_b,
                  footer_offset=footer_off,
                  file_size=f_size,
                  category=cat,
                  phase=phase,
                  step_idx=step_idx,
              )
          )
      try:
        os.unlink(self._trace_file.name)
      except OSError:
        pass
      self._trace_file = None

    ra_windows: dict[tuple[int, int, str], tuple[int, int]] = {}
    gcsfuse_cold_gets: list[PreadEvent] = []
    gcsfuse_cold_bytes = 0
    readahead_hits = 0

    for ev in events:
      key = (ev.pid, ev.fd, ev.file_path)
      win = ra_windows.get(key)
      if (
          win is not None
          and ev.offset >= win[0]
          and (ev.offset + ev.returned_bytes) <= win[1]
      ):
        readahead_hits += 1
      else:
        fetch_len = min(
            max(0, ev.file_size - ev.offset),
            max(ev.requested_bytes, self.read_ahead_bytes),
        )
        ra_windows[key] = (ev.offset, ev.offset + fetch_len)
        gcsfuse_cold_gets.append(ev)
        gcsfuse_cold_bytes += fetch_len

    init_probe_gets = [
        e for e in gcsfuse_cold_gets if e.category == "init_probe_shard_open"
    ]
    lazy_open_gets = [
        e for e in gcsfuse_cold_gets if e.category == "lazy_worker_shard_open"
    ]
    shard_open_gets = init_probe_gets + lazy_open_gets
    record_read_gets = [
        e for e in gcsfuse_cold_gets if e.category == "record_read"
    ]

    init_only_gets = [e for e in gcsfuse_cold_gets if e.phase == "init"]
    first_batch_gets = [
        e for e in gcsfuse_cold_gets if e.phase == "first_batch"
    ]
    init_and_first_gets = [
        e for e in gcsfuse_cold_gets if e.phase in ("init", "first_batch")
    ]
    warmup_gets = [e for e in gcsfuse_cold_gets if e.phase == "warmup"]
    steady_gets = [e for e in gcsfuse_cold_gets if e.phase == "steady"]
    active_iter_gets = [e for e in gcsfuse_cold_gets if e.phase != "init"]
    steady_record_gets = [e for e in steady_gets if e.category == "record_read"]
    steady_lazy_open_gets = [
        e for e in steady_gets if e.category == "lazy_worker_shard_open"
    ]

    step_counts: dict[tuple[str, int], int] = {}
    steady_step_counts: dict[int, int] = {}
    for e in active_iter_gets:
      key = (e.phase, e.step_idx)
      step_counts[key] = step_counts.get(key, 0) + 1
      if e.phase == "steady":
        steady_step_counts[e.step_idx] = (
            steady_step_counts.get(e.step_idx, 0) + 1
        )

    peak_step_gets = max(step_counts.values()) if step_counts else 0
    steady_peak_step_gets = (
        max(steady_step_counts.values()) if steady_step_counts else 0
    )
    peak_1s_gets = _compute_peak_window_gets(
        [e.timestamp for e in gcsfuse_cold_gets], window_s=1.0
    )
    peak_1s_active_gets = _compute_peak_window_gets(
        [e.timestamp for e in active_iter_gets], window_s=1.0
    )

    unique_shards = sorted({e.file_path for e in events})

    return {
        "total_pread64_syscalls": len(events),
        "pread64_readahead_hits": readahead_hits,
        "total_media_get_requests": len(gcsfuse_cold_gets),
        "init_probe_shard_open_gets": len(init_probe_gets),
        "lazy_worker_shard_open_gets": len(lazy_open_gets),
        "exact_shard_open_gets": len(shard_open_gets),
        "exact_record_read_gets": len(record_read_gets),
        "init_only_gets": len(init_only_gets),
        "first_batch_gets": len(first_batch_gets),
        "init_and_first_batch_gets": len(init_and_first_gets),
        "warmup_gets": len(warmup_gets),
        "steady_total_gets": len(steady_gets),
        "steady_record_read_gets": len(steady_record_gets),
        "steady_lazy_shard_open_gets": len(steady_lazy_open_gets),
        "peak_step_gets": peak_step_gets,
        "steady_peak_step_gets": steady_peak_step_gets,
        "peak_1s_gets": peak_1s_gets,
        "peak_1s_active_gets": peak_1s_active_gets,
        "unique_shards_touched": len(unique_shards),
        "total_requested_bytes": gcsfuse_cold_bytes,
        "total_transferred_bytes": gcsfuse_cold_bytes,
        "posix_pread_bytes": sum(e.returned_bytes for e in events),
        "range_type_breakdown": {
            "bounded_range (gcsfuse_readahead_miss)": len(gcsfuse_cold_gets),
            "gcsfuse_readahead_hit (0_gcs_gets)": readahead_hits,
        },
        "request_category_breakdown": {
            "init_probe_shard_open": len(init_probe_gets),
            "lazy_worker_shard_open": len(lazy_open_gets),
            "record_read": len(record_read_gets),
        },
    }


def read_process_tree_io(root_pid: Optional[int] = None) -> dict[str, int]:
  """Sums `/proc/<pid>/io` counters (`syscr`, `read_bytes`, `rchar`)."""
  if root_pid is None:
    root_pid = os.getpid()

  pids = {root_pid}
  frontier = [root_pid]
  while frontier:
    curr = frontier.pop()
    children_file = f"/proc/{curr}/task/{curr}/children"
    try:
      with open(children_file, "r", encoding="utf-8") as f:
        for tok in f.read().split():
          cpid = int(tok)
          if cpid not in pids:
            pids.add(cpid)
            frontier.append(cpid)
    except OSError:
      pass

  totals = {"syscr": 0, "read_bytes": 0, "rchar": 0}
  for pid in pids:
    try:
      with open(f"/proc/{pid}/io", "r", encoding="utf-8") as f:
        for line in f:
          parts = line.strip().split(":")
          if len(parts) == 2 and parts[0] in totals:
            totals[parts[0]] += int(parts[1].strip())
    except OSError:
      pass
  return totals
