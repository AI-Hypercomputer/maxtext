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

"""Local GCS JSON-API emulator that records every request it serves.

Each simulated host gets its own `InstrumentedGCSServer` on its own port. The
host process points the C++ client at it through
`CLOUD_STORAGE_EMULATOR_ENDPOINT` and the Python client through
`STORAGE_EMULATOR_HOST`. Requests are recorded as `gcs_trace.GcsRequest` events
with Unix timestamps, so they are summarized exactly like live client traces.
The emulator is for hermetic CI and self-tests. It does not model GCS latency
or throughput.
"""

from __future__ import annotations

import fnmatch
from http import server as http_server
import json
import os
import sys
import threading
import time
from typing import Any
from urllib import parse as urllib_parse

from benchmarks.input_pipeline import gcs_trace

_RIEGELI_BLOCK_SIZE = 1 << 16
_ARRAY_RECORD_MAGIC = 0x71930E704FDAE05E


def _encode_varint(val: int) -> bytes:
  out = bytearray()
  while val > 0x7F:
    out.append((val & 0x7F) | 0x80)
    val >>= 7
  out.append(val)
  return bytes(out)


def _decode_varint(data: bytes, pos: int) -> tuple[int | None, int]:
  val, shift = 0, 0
  while pos < len(data):
    b = data[pos]
    pos += 1
    val |= (b & 0x7F) << shift
    if not b & 0x80:
      return val, pos
    shift += 7
  return None, pos


_MAGIC_TAG = b"\x10" + _encode_varint(_ARRAY_RECORD_MAGIC)
_FOOTER_CACHE: dict[str, int] = {}
_FOOTER_LOCK = threading.Lock()


def extract_arrayrecord_footer_offset(path: str) -> int:
  """Returns the byte offset where the ArrayRecord footer starts in `path`.

  The Riegeli postscript in the last 64 KiB block stores the footer offset as
  field 1 (varint), next to the ArrayRecord magic in field 2. If no postscript
  is found, the offset of the last block is returned.

  Args:
    path: Local ArrayRecord file.

  Returns:
    The footer offset in bytes, or 0 if the file is unreadable.
  """
  real = os.path.realpath(path)
  with _FOOTER_LOCK:
    if real in _FOOTER_CACHE:
      return _FOOTER_CACHE[real]
  try:
    size = os.path.getsize(real)
    if size < _RIEGELI_BLOCK_SIZE:
      return size
    tail_off = size - _RIEGELI_BLOCK_SIZE
    with open(real, "rb") as f:
      f.seek(tail_off)
      tail = f.read()
  except OSError:
    return 0
  result = tail_off
  pos = tail.find(_MAGIC_TAG)
  while pos != -1:
    before = tail[max(0, pos - 11) : pos]
    for i in range(len(before) - 1):
      if before[i] == 0x08:
        val, end = _decode_varint(before, i + 1)
        if end == len(before) and val is not None and 0 < val < size:
          result = val
          break
    else:
      pos = tail.find(_MAGIC_TAG, pos + 1)
      continue
    break
  with _FOOTER_LOCK:
    _FOOTER_CACHE[real] = result
  return result


class InstrumentedGCSServer:
  """Threaded GCS JSON-API emulator serving files from a local directory.

  Supports object listing (`prefix`, `matchGlob`), object metadata, and media
  GETs with `Range: bytes=a-b`, `bytes=a-`, or `bytes=-n`. Optionally injects
  deterministic failures (every Nth media GET returns `fail_status`) to
  exercise client retry accounting.
  """

  def __init__(
      self,
      root_dir: str,
      bucket_name: str = "qps-bench",
      host_index: int = 0,
      fail_every_n: int = 0,
      fail_status: int = 429,
  ):
    self.root_dir = os.path.abspath(root_dir)
    self.bucket_name = bucket_name
    self.host_index = host_index
    self.fail_every_n = fail_every_n
    self.fail_status = fail_status
    self._lock = threading.Lock()
    self._media_count = 0
    self.requests: list[gcs_trace.GcsRequest] = []
    self._server: http_server.ThreadingHTTPServer | None = None
    self._thread: threading.Thread | None = None
    self.endpoint = ""

  def reset(self) -> None:
    """Clears recorded requests and the fault-injection counter."""
    with self._lock:
      self.requests = []
      self._media_count = 0

  def snapshot(self) -> list[gcs_trace.GcsRequest]:
    with self._lock:
      return list(self.requests)

  def record(self, req: gcs_trace.GcsRequest) -> gcs_trace.GcsRequest:
    req.host = self.host_index
    req.source = "emulator"
    with self._lock:
      self.requests.append(req)
    return req

  def next_media_should_fail(self) -> bool:
    with self._lock:
      self._media_count += 1
      return self.fail_every_n > 0 and self._media_count % self.fail_every_n == 0

  def list_objects(self, prefix: str, match_glob: str) -> list[dict[str, Any]]:
    """Returns object metadata for files under `root_dir` matching `prefix` and `matchGlob`."""
    items = []
    for dirpath, _, filenames in os.walk(self.root_dir):
      for fname in sorted(filenames):
        full = os.path.join(dirpath, fname)
        rel = os.path.relpath(full, self.root_dir).replace(os.sep, "/")
        if prefix and not rel.startswith(prefix):
          continue
        if match_glob and not fnmatch.fnmatchcase(rel, match_glob):
          continue
        items.append(self.object_metadata(rel, os.path.getsize(full)))
    items.sort(key=lambda x: x["name"])
    return items

  def object_metadata(self, name: str, size: int) -> dict[str, Any]:
    return {
        "kind": "storage#object",
        "id": f"{self.bucket_name}/{name}/1",
        "name": name,
        "bucket": self.bucket_name,
        "generation": "1",
        "metageneration": "1",
        "size": str(size),
        "contentType": "application/octet-stream",
    }

  def start(self) -> str:
    """Starts serving on an ephemeral localhost port and returns the endpoint."""
    outer = self

    class _Server(http_server.ThreadingHTTPServer):
      """Threading server that ignores clients aborting a streamed body."""

      daemon_threads = True

      def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
          return
        super().handle_error(request, client_address)

    class _Handler(http_server.BaseHTTPRequestHandler):
      """GCS JSON API subset: list, metadata, and ranged media GETs."""

      protocol_version = "HTTP/1.1"

      def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        del format, args

      def _send(self, status: int, body: bytes, ctype: str = "application/json", extra=()):
        """Sends a complete response with an in-memory body."""
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
          self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

      def do_GET(self):  # pylint: disable=invalid-name
        """Serves one request and records it."""
        now = time.time()
        parsed = urllib_parse.urlparse(self.path)
        query = urllib_parse.parse_qs(parsed.query)
        method, bucket, obj = gcs_trace.http_object_from_path(self.path)
        if bucket != outer.bucket_name:
          self._send(404, b'{"error": {"code": 404, "message": "bucket not found"}}')
          return
        if method == "ListObjects":
          items = outer.list_objects(query.get("prefix", [""])[0], query.get("matchGlob", [""])[0])
          body = json.dumps({"kind": "storage#objects", "items": items}).encode("utf-8")
          outer.record(gcs_trace.GcsRequest(ts=now, method=method, bucket=bucket, range_kind="list", status="200"))
          self._send(200, body)
          return
        local = os.path.join(outer.root_dir, os.path.normpath(obj).lstrip(os.sep))
        if not os.path.isfile(local):
          self._send(404, b'{"error": {"code": 404, "message": "object not found"}}')
          return
        size = os.path.getsize(local)
        if method == "GetObjectMetadata":
          outer.record(
              gcs_trace.GcsRequest(ts=now, method=method, bucket=bucket, object_name=obj, range_kind="meta", status="200")
          )
          self._send(200, json.dumps(outer.object_metadata(obj, size)).encode("utf-8"))
          return

        rng = self.headers.get("Range")
        kind, start, end, last_n = (
            gcs_trace.range_from_header(rng[len("bytes=") :]) if rng else ("full", None, None, None)
        )
        req = outer.record(
            gcs_trace.GcsRequest(
                ts=now,
                method=method,
                bucket=bucket,
                object_name=obj,
                range_kind=kind,
                start=start,
                end=end,
                last_n=last_n,
            )
        )
        if outer.next_media_should_fail():
          req.status = str(outer.fail_status)
          req.outcome = gcs_trace.outcome_for_status(req.status)
          body = json.dumps({"error": {"code": outer.fail_status, "message": "injected"}}).encode("utf-8")
          self._send(outer.fail_status, body)
          return
        if kind == "last":
          first, last = max(0, size - (last_n or 0)), size - 1
        elif kind in ("offset", "range"):
          first = start or 0
          last = size - 1 if end is None else min(end - 1, size - 1)
        else:
          first, last = 0, size - 1
        if size > 0 and (first > last or first >= size):
          req.status, req.outcome = "416", "error"
          self._send(416, b"", extra=(("Content-Range", f"bytes */{size}"),))
          return
        footer = extract_arrayrecord_footer_offset(local)
        req.category = "shard_open" if (kind in ("full", "last") or first >= footer) else "record_read"
        length = max(0, last - first + 1)
        req.content_length = length
        status = 206 if rng else 200
        req.status = str(status)
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("x-goog-generation", "1")
        self.send_header("x-goog-metageneration", "1")
        self.send_header("x-goog-stored-content-length", str(size))
        if rng:
          self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        try:
          with open(local, "rb") as f:
            f.seek(first)
            remaining = length
            while remaining > 0:
              buf = f.read(min(1 << 16, remaining))
              if not buf:
                break
              self.wfile.write(buf)
              req.bytes_received += len(buf)
              remaining -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
          self.close_connection = True
        if req.bytes_received < length:
          req.outcome = "cancelled"

    self._server = _Server(("127.0.0.1", 0), _Handler)
    self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
    self._thread.start()
    self.endpoint = f"http://127.0.0.1:{self._server.server_port}"
    return self.endpoint

  def stop(self) -> None:
    if self._server is not None:
      self._server.shutdown()
      self._server.server_close()
      self._server = None


def emulator_env(endpoint: str) -> dict[str, str]:
  """Env vars pointing the C++ and Python GCS clients at an emulator endpoint."""
  return {
      "CLOUD_STORAGE_EMULATOR_ENDPOINT": endpoint,
      "STORAGE_EMULATOR_HOST": endpoint,
  }
