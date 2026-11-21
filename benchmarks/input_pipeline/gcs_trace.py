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

"""Measured GCS request accounting for the MaxText input pipeline benchmark.

Every number produced by this module is derived from observed events. Nothing
is estimated from per-record byte constants or read-amplification formulas.

Event sources:

1. `google-cloud-cpp` client logs (live `gs://` runs). The ArrayRecord /
   Riegeli `GcsReader` uses `google::cloud::storage::Client`, which logs to
   `std::clog` (stderr) when these environment variables are set (see
   `client_trace_env()`):

   - `GOOGLE_CLOUD_CPP_ENABLE_CLOG=yes` installs the `std::clog` backend. Each
     record is formatted as
     `<YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ> [<SEVERITY>] <thread-id> <message> (<file>:<line>)`
     and `<message>` may span several lines.
   - `CLOUD_STORAGE_ENABLE_TRACING=raw-client` wraps the REST transport stub
     in `LoggingStub` (`generic_stub_factory.cc`). That stub logs exactly one
     `ReadObject() << ReadObjectRangeRequest={bucket_name=B, object_name=O, ...}`
     line per transport-level read. This includes retries, resumes, and hedged
     reads, because they all go through the stub. A failed open adds
     `ReadObject() >> status={CODE: ...}`.
   - `CLOUD_STORAGE_ENABLE_TRACING=http` (REST only) logs the curl traffic:
     `>> curl(Send Header): GET /storage/v1/b/B/o/O?alt=media HTTP/1.1`, the
     request headers (including `Range: bytes=...`),
     `<< curl(Recv Header): HTTP/1.1 206 Partial Content`, and
     `<< curl(Recv Data): size=N ...`.
   - `CLOUD_STORAGE_ENABLE_TRACING=rpc,rpc-streams` (gRPC only) wraps the gRPC
     `StorageStub` in `StorageLogging` (`storage_stub_factory.cc`; the storage
     client reads its logging components only from `CLOUD_STORAGE_ENABLE_TRACING`).
     It logs
     `ReadObject() << bucket: "projects/_/buckets/B" object: "O" read_offset: N read_limit: M`,
     then per-stream `Read(<id>)() >> <status>` and `Cancel(<id>)() << (void)`.

2. `InstrumentedGCSServer` (hermetic runs). This local HTTP emulator records one
   event per request it serves (see `gcs_emulator.py`).

The module also provides `/proc` accounting (NIC RX bytes from `/proc/net/dev`,
plus per-process-tree RSS, open FDs, and I/O counters), an optional Cloud
Monitoring cross-check, and a per-host SHA-256 sample-order digest.
"""

from __future__ import annotations

import bisect
import calendar
import dataclasses
import glob
import hashlib
import os
import re
import time
from typing import Any, Iterable, Mapping, Sequence

PHASES = ("init", "first_batch", "warmup", "steady")


def client_trace_env(http: bool = True, grpc: bool = True) -> dict[str, str]:
  """Returns the `google-cloud-cpp` env vars that enable request tracing.

  Args:
    http: Also log raw curl traffic (status codes and bytes received). This is
      verbose, so it is only enabled for the traced pass.
    grpc: Also enable gRPC RPC and stream logging (status and cancellation).

  Returns:
    Environment variables to set before the first GCS client is created in a
    process. Child processes inherit them.
  """
  components = ["raw-client"] + (["http"] if http else []) + (["rpc", "rpc-streams"] if grpc else [])
  env = {
      "GOOGLE_CLOUD_CPP_ENABLE_CLOG": "yes",
      # `storage::Client` (REST and gRPC) reads logging components only from here.
      "CLOUD_STORAGE_ENABLE_TRACING": ",".join(components),
  }
  if grpc:
    # Common gRPC tracing switch used by other google-cloud-cpp libraries; harmless for storage.
    env["GOOGLE_CLOUD_CPP_ENABLE_TRACING"] = "rpc,rpc-streams"
  return env


# Keys `client_trace_env()` may set. Strip them from the environment of an
# untraced timing pass.
CLIENT_TRACE_ENV_KEYS = (
    "GOOGLE_CLOUD_CPP_ENABLE_CLOG",
    "CLOUD_STORAGE_ENABLE_TRACING",
    "GOOGLE_CLOUD_CPP_ENABLE_TRACING",
)


@dataclasses.dataclass
class GcsRequest:
  """One transport-level GCS request (an HTTP GET or a gRPC stream)."""

  ts: float  # Unix epoch seconds when the request was issued.
  host: int = 0
  pid: int = 0
  thread: str = ""
  source: str = ""  # "raw-client", "http", "grpc", or "emulator".
  method: str = "ReadObject"  # ReadObject, ListObjects, GetObjectMetadata, ...
  bucket: str = ""
  object_name: str = ""
  range_kind: str = "full"  # full, range, offset, last, list, meta.
  start: int | None = None  # First requested byte (inclusive).
  end: int | None = None  # Last requested byte, exclusive; None means EOF.
  last_n: int | None = None  # For `ReadLast(N)` / `Range: bytes=-N`.
  status: str = ""  # HTTP code ("206", "429") or gRPC code ("UNAVAILABLE").
  outcome: str = "ok"  # ok, throttled, unavailable, cancelled, error.
  bytes_received: int = 0
  content_length: int | None = None
  category: str = ""  # Optional emulator classification (shard_open, record_read).

  @property
  def requested_bytes(self) -> int | None:
    if self.range_kind == "range" and self.start is not None and self.end is not None:
      return max(0, self.end - self.start)
    if self.range_kind == "last" and self.last_n is not None:
      return self.last_n
    return self.content_length

  @property
  def is_media(self) -> bool:
    return self.method == "ReadObject"


# A clog record starts at the beginning of a line.
_CLOG_RECORD_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d+))?Z \[(?P<sev>[A-Z]+)\] (?P<tid>\S+) (?P<msg>.*)$"
)
_TRAILER_RE = re.compile(r" \([^()\s]+:\d+\)\s*$")

_RAW_REQ_RE = re.compile(
    r"(?P<method>ReadObject|ListObjects|GetObjectMetadata)\(\) << "
    r"(?P<rtype>\w+)=\{bucket_name=(?P<bucket>[^,}]*)(?:, object_name=(?P<object>[^,}]*))?(?P<rest>.*)\}"
)
_RAW_FAIL_RE = re.compile(r"(?P<method>ReadObject|ListObjects|GetObjectMetadata)\(\) >> status=\{(?P<code>[A-Z_]+)")
_OPT_RANGE_RE = re.compile(r"read-range=ReadRangeData=\{begin=(?P<b>-?\d+), end=(?P<e>-?\d+)\}")
_OPT_OFFSET_RE = re.compile(r"read-offset=(?P<o>-?\d+)")
_OPT_LAST_RE = re.compile(r"read-last=(?P<n>-?\d+)")

_CURL_EVENT_RE = re.compile(
    r">> curl\(Send Header\): [A-Z]+ (?P<path>\S+) HTTP/[\d.]+"
    r"|<< curl\(Recv Header\): HTTP/[\d.]+ (?P<code>\d{3})"
    r"|<< curl\(Recv Header\): [Cc]ontent-[Ll]ength:\s*(?P<content_len>\d+)"
    r"|<< curl\(Recv Data\): size=(?P<size>\d+)"
)
_HTTP_DATA_RE = re.compile(r"<< curl\(Recv Data\): size=(?P<n>\d+)")
_HTTP_RANGE_RE = re.compile(r"^\s*range:\s*bytes=(?P<spec>[^\s,]+)", re.IGNORECASE | re.MULTILINE)
_GRPC_REQ_RE = re.compile(
    r"ReadObject\([^)]*\) << .*?bucket: \"(?P<bucket>[^\"]*)\".*?object: \"(?P<object>[^\"]*)\"(?P<rest>.*)"
)
_GRPC_READ_OFFSET_RE = re.compile(r"read_offset: (?P<n>-?\d+)")
_GRPC_READ_LIMIT_RE = re.compile(r"read_limit: (?P<n>-?\d+)")
_GRPC_STREAM_RE = re.compile(r"(?P<op>Read|Cancel)\((?P<rid>[^)]*)\)\(\) (?P<dir><<|>>) (?P<payload>.*)")
_GRPC_CODE_RE = re.compile(
    r"^(?P<code>OK|CANCELLED|UNKNOWN|INVALID_ARGUMENT|DEADLINE_EXCEEDED|NOT_FOUND|ALREADY_EXISTS|"
    r"PERMISSION_DENIED|RESOURCE_EXHAUSTED|FAILED_PRECONDITION|ABORTED|OUT_OF_RANGE|UNIMPLEMENTED|"
    r"INTERNAL|UNAVAILABLE|DATA_LOSS|UNAUTHENTICATED)\b"
)


def _parse_clog_ts(ts: str, frac: str | None) -> float:
  secs = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
  return secs + (float("0." + frac) if frac else 0.0)


def outcome_for_status(status: str) -> str:
  """Maps an HTTP status code or a gRPC status code name to an outcome bucket."""
  if not status:
    return "ok"
  if status.isdigit():
    code = int(status)
    if code < 400:
      return "ok"
    if code == 429:
      return "throttled"
    if code == 503:
      return "unavailable"
    if code == 499:
      return "cancelled"
    return "error"
  return {
      "OK": "ok",
      "RESOURCE_EXHAUSTED": "throttled",
      "UNAVAILABLE": "unavailable",
      "CANCELLED": "cancelled",
  }.get(status, "error")


def range_from_header(spec: str) -> tuple[str, int | None, int | None, int | None]:
  """Parses an HTTP `Range: bytes=` spec into (kind, start, end_exclusive, last_n)."""
  spec = spec.split(",", 1)[0].strip()
  first, _, second = spec.partition("-")
  if not first:
    return "last", None, None, int(second) if second else None
  if not second:
    return "offset", int(first), None, None
  return "range", int(first), int(second) + 1, None


def http_object_from_path(path: str) -> tuple[str, str, str]:
  """Extracts (method, bucket, object) from a JSON API request path."""
  from urllib import parse as urllib_parse  # pylint: disable=import-outside-toplevel

  parsed = urllib_parse.urlparse(path)
  m = re.match(r"^/(?:download/)?storage/v1/b/(?P<bucket>[^/]+)/o(?:/(?P<object>.*))?$", parsed.path)
  if not m:
    return "Other", "", ""
  if m.group("object") is None:
    return "ListObjects", m.group("bucket"), ""
  query = urllib_parse.parse_qs(parsed.query)
  method = "ReadObject" if query.get("alt", [""])[0] == "media" else "GetObjectMetadata"
  return method, m.group("bucket"), urllib_parse.unquote(m.group("object"))


def iter_clog_records(lines: Iterable[str]) -> Iterable[tuple[float, str, str]]:
  """Groups raw log lines into (ts, thread, message) clog records.

  Lines that do not start with a clog timestamp are appended to the previous
  record. A multi-line curl trace is one record. Lines before the first record
  (for example, Python output on the same stderr) are skipped.

  Args:
    lines: Raw lines of a trace file.

  Yields:
    One (unix_ts, thread_id, message) tuple per clog record.
  """
  cur: list[Any] | None = None
  secs_cache: dict[str, float] = {}  # Whole-second prefixes repeat across many records.
  for raw in lines:
    line = raw.rstrip("\n")
    m = _CLOG_RECORD_RE.match(line) if len(line) > 30 and line[4] == "-" and line[10] == "T" else None
    if m:
      if cur is not None:
        yield cur[0], cur[1], _TRAILER_RE.sub("", "\n".join(cur[2]))
      whole = m.group("ts")
      secs = secs_cache.get(whole)
      if secs is None:
        secs = secs_cache[whole] = _parse_clog_ts(whole, None)
      frac = m.group("frac")
      cur = [secs + (float("0." + frac) if frac else 0.0), m.group("tid"), [m.group("msg")]]
    elif cur is not None:
      cur[2].append(line)
  if cur is not None:
    yield cur[0], cur[1], _TRAILER_RE.sub("", "\n".join(cur[2]))


@dataclasses.dataclass
class ClogParseResult:
  """Requests and auxiliary counters parsed from one or more clog files."""

  requests: list[GcsRequest]
  http_bytes_timeline: list[tuple[float, int, int]]  # (ts, host, bytes) per curl Recv Data flush.
  records: int = 0
  http_requests_seen: int = 0
  grpc_requests_seen: int = 0
  unmatched_http_requests: int = 0


def file_has_raw_client_lines(lines: Iterable[str]) -> bool:
  """Cheap substring pre-scan: does a trace contain REST `raw-client` request lines?"""
  return any("<< ReadObjectRangeRequest=" in line or "<< ListObjectsRequest=" in line for line in lines)


def parse_clog_lines(lines: Iterable[str], host: int = 0, pid: int = 0, has_raw: bool | None = None) -> ClogParseResult:
  """Parses `google-cloud-cpp` clog output into `GcsRequest` events.

  Request identity comes from REST `raw-client` `ReadObject() <<` lines when
  they are present. If a file has none (for example a gRPC client, whose
  `StorageLogging` decorator emits `ReadObject() << bucket:` lines instead, or a
  REST trace with only `http`), gRPC request lines or curl `Send Header` GETs
  define the requests. Statuses, bytes, and cancellations from
  the HTTP and gRPC traces are attached to the matching request on the same
  thread. The client issues these calls synchronously, so the stub log line
  and its transport traffic share a thread.

  Args:
    lines: Raw lines of a trace file.
    host: Simulated host index to tag requests with.
    pid: OS process id that wrote the file.
    has_raw: Whether the input has `raw-client` lines (see
      `file_has_raw_client_lines`). If None, `lines` is buffered and scanned.
      Pass it for large files so records are streamed.

  Returns:
    A ClogParseResult. Curl receive bytes are aggregated into 10 ms buckets.
  """
  if has_raw is None:
    lines = list(lines)
    has_raw = file_has_raw_client_lines(lines)
  requests: list[GcsRequest] = []
  bytes_timeline: list[tuple[float, int, int]] = []
  bytes_buckets: dict[float, int] = {}
  # Per thread: requests awaiting an HTTP Send Header, the request currently
  # receiving HTTP data, and gRPC requests awaiting a stream id.
  awaiting_http: dict[str, list[GcsRequest]] = {}
  current_http: dict[str, GcsRequest | None] = {}
  awaiting_grpc: dict[str, list[GcsRequest]] = {}
  grpc_by_rid: dict[str, GcsRequest] = {}
  result = ClogParseResult(requests=requests, http_bytes_timeline=bytes_timeline)

  def _add(req: GcsRequest) -> GcsRequest:
    req.host, req.pid = host, pid
    requests.append(req)
    return req

  def _handle_curl(ts: float, tid: str, msg: str) -> None:
    """Walks curl events in order; one flush can end one body and start the next request."""
    for ev in _CURL_EVENT_RE.finditer(msg):
      if ev.group("path"):
        result.http_requests_seen += 1
        method, bucket, obj = http_object_from_path(ev.group("path"))
        nxt = msg.find("curl(", ev.end())
        mr = _HTTP_RANGE_RE.search(msg[ev.end() : nxt if nxt >= 0 else len(msg)])
        kind, start, end, last_n = range_from_header(mr.group("spec")) if mr else ("full", None, None, None)
        req = None
        if has_raw:
          queue = awaiting_http.get(tid, [])
          for i, cand in enumerate(queue):
            if cand.object_name == obj and cand.method == method:
              req = queue.pop(i)
              break
          if req is None:
            result.unmatched_http_requests += 1
        else:
          kind = {"ListObjects": "list", "GetObjectMetadata": "meta"}.get(method, kind)
          req = _add(
              GcsRequest(
                  ts=ts,
                  thread=tid,
                  source="http",
                  method=method,
                  bucket=bucket,
                  object_name=obj,
                  range_kind=kind,
                  start=start,
                  end=end,
                  last_n=last_n,
              )
          )
        current_http[tid] = req
        continue
      req = current_http.get(tid)
      if ev.group("code"):
        if req is not None and not req.status:
          req.status = ev.group("code")
          req.outcome = outcome_for_status(req.status)
      elif ev.group("content_len"):
        if req is not None and req.content_length is None:
          req.content_length = int(ev.group("content_len"))
      elif ev.group("size"):
        nbytes = int(ev.group("size"))
        bucket = round(ts, 2)
        bytes_buckets[bucket] = bytes_buckets.get(bucket, 0) + nbytes
        if req is not None:
          req.bytes_received += nbytes

  for ts, tid, msg in iter_clog_records(lines):
    result.records += 1
    if " recv_count=" in msg and "curl(" not in msg:
      continue  # CurlHandle::FlushDebug counter summary; carries no request data.
    if "curl(" in msg:
      if "Send Header" not in msg and "Recv Header" not in msg:
        # Fast path for the dominant body-only flushes.
        nbytes = sum(int(md.group("n")) for md in _HTTP_DATA_RE.finditer(msg))
        if nbytes:
          bucket = round(ts, 2)
          bytes_buckets[bucket] = bytes_buckets.get(bucket, 0) + nbytes
          req = current_http.get(tid)
          if req is not None:
            req.bytes_received += nbytes
        continue
      _handle_curl(ts, tid, msg)
      continue
    m = _RAW_REQ_RE.search(msg)
    if m:
      rest = m.group("rest") or ""
      method = m.group("method")
      req = GcsRequest(
          ts=ts,
          thread=tid,
          source="raw-client",
          method=method,
          bucket=m.group("bucket"),
          object_name=m.group("object") or "",
      )
      if method == "ListObjects":
        req.range_kind = "list"
      elif method == "GetObjectMetadata":
        req.range_kind = "meta"
      elif mr := _OPT_RANGE_RE.search(rest):
        req.range_kind, req.start, req.end = "range", int(mr.group("b")), int(mr.group("e"))
      elif ml := _OPT_LAST_RE.search(rest):
        req.range_kind, req.last_n = "last", int(ml.group("n"))
      elif mo := _OPT_OFFSET_RE.search(rest):
        req.range_kind, req.start = "offset", int(mo.group("o"))
      _add(req)
      awaiting_http.setdefault(tid, []).append(req)
      awaiting_grpc.setdefault(tid, []).append(req)
      continue

    m = _RAW_FAIL_RE.search(msg)
    if m:
      for req in reversed(requests):
        if req.thread == tid and req.method == m.group("method") and req.source == "raw-client":
          if not req.status or req.status.isdigit() and int(req.status) < 400:
            req.status = m.group("code")
            req.outcome = outcome_for_status(req.status)
          break
      continue

    m = _GRPC_REQ_RE.search(msg)
    if m and "ReadObjectRangeRequest" not in msg:
      result.grpc_requests_seen += 1
      if has_raw:
        continue  # Already counted by the raw-client line on this thread.
      rest = m.group("rest")
      bucket = m.group("bucket").rsplit("/", 1)[-1]
      off = _GRPC_READ_OFFSET_RE.search(rest)
      lim = _GRPC_READ_LIMIT_RE.search(rest)
      req = GcsRequest(ts=ts, thread=tid, source="grpc", bucket=bucket, object_name=m.group("object"))
      start = int(off.group("n")) if off else 0
      if start < 0:
        req.range_kind, req.last_n = "last", -start
      elif lim and int(lim.group("n")) > 0:
        req.range_kind, req.start, req.end = "range", start, start + int(lim.group("n"))
      elif start > 0:
        req.range_kind, req.start = "offset", start
      _add(req)
      awaiting_grpc.setdefault(tid, []).append(req)
      continue

    m = _GRPC_STREAM_RE.search(msg)
    if m:
      rid = m.group("rid")
      req = grpc_by_rid.get(rid)
      if req is None:
        queue = awaiting_grpc.get(tid, [])
        if queue:
          req = queue.pop()
          queue.clear()  # Older unbound requests on this thread never streamed.
          grpc_by_rid[rid] = req
      if req is None:
        continue
      if m.group("op") == "Cancel":
        req.outcome, req.status = "cancelled", req.status or "CANCELLED"
      elif m.group("dir") == ">>":
        mc = _GRPC_CODE_RE.match(m.group("payload"))
        if mc and mc.group("code") != "OK" and req.outcome == "ok":
          req.status = mc.group("code")
          req.outcome = outcome_for_status(req.status)
      continue

  bytes_timeline.extend((b, host, n) for b, n in sorted(bytes_buckets.items()))
  # REST cancellation: fewer body bytes than the server announced.
  for req in requests:
    if (
        req.outcome == "ok"
        and req.content_length is not None
        and req.bytes_received < req.content_length
        and req.source in ("raw-client", "http")
        and req.status
    ):
      req.outcome = "cancelled"
  return result


def _parse_trace_file(path: str) -> ClogParseResult:
  """Parses one `host<H>.pid<P>.clog` file in two streaming passes (pre-scan, then parse)."""
  m = re.search(r"host(?P<h>\d+)\.pid(?P<p>\d+)\.clog$", path)
  with open(path, "r", encoding="utf-8", errors="replace") as f:
    has_raw = file_has_raw_client_lines(f)
  with open(path, "r", encoding="utf-8", errors="replace") as f:
    return parse_clog_lines(f, host=int(m.group("h")), pid=int(m.group("p")), has_raw=has_raw)


def parse_trace_dir(trace_dir: str, host: int | None = None, max_workers: int = 0, mp_context=None) -> ClogParseResult:
  """Parses all `host<H>.pid<P>.clog` files under `trace_dir`.

  Files are independent (one per process), so they are parsed in parallel
  worker processes when `max_workers` > 1 (0 means one per CPU, capped at the
  file count).

  Args:
    trace_dir: Directory of per-process trace files.
    host: Only parse this simulated host's files.
    max_workers: Parallel parser processes; 1 parses in-process.
    mp_context: Optional multiprocessing context for the worker pool.

  Returns:
    Merged ClogParseResult with requests sorted by time.
  """
  paths = []
  for path in sorted(glob.glob(os.path.join(trace_dir, "host*.pid*.clog"))):
    m = re.search(r"host(?P<h>\d+)\.pid(?P<p>\d+)\.clog$", path)
    if m and (host is None or int(m.group("h")) == host):
      paths.append(path)
  workers = min(len(paths), max_workers or os.cpu_count() or 1)
  results: list[ClogParseResult] = []
  if workers > 1:
    from concurrent import futures  # pylint: disable=import-outside-toplevel

    try:
      with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as ex:
        results = list(ex.map(_parse_trace_file, paths))
    except Exception:  # pylint: disable=broad-exception-caught
      results = []  # Fall back to in-process parsing (e.g. no spawn support).
  if not results:
    results = [_parse_trace_file(p) for p in paths]
  merged = ClogParseResult(requests=[], http_bytes_timeline=[])
  for res in results:
    merged.requests.extend(res.requests)
    merged.http_bytes_timeline.extend(res.http_bytes_timeline)
    merged.records += res.records
    merged.http_requests_seen += res.http_requests_seen
    merged.grpc_requests_seen += res.grpc_requests_seen
    merged.unmatched_http_requests += res.unmatched_http_requests
  merged.requests.sort(key=lambda r: r.ts)
  return merged


def peak_window_count(timestamps: Sequence[float], window_s: float = 1.0) -> int:
  """Returns the maximum number of events in any half-open `window_s` window."""
  ts = sorted(timestamps)
  best = 0
  left = 0
  for right, t in enumerate(ts):
    while t - ts[left] >= window_s:
      left += 1
    best = max(best, right - left + 1)
  return best


@dataclasses.dataclass
class PhaseMarks:
  """Phase boundaries reported by one simulated host (Unix epoch seconds)."""

  host: int
  init_start: float
  init_end: float
  first_batch_end: float
  warmup_end: float
  steady_end: float
  step_starts: list[float] = dataclasses.field(default_factory=list)  # Steady-step start times.

  def phase_of(self, ts: float) -> str:
    if ts < self.init_end:
      return "init"
    if ts < self.first_batch_end:
      return "first_batch"
    if ts < self.warmup_end:
      return "warmup"
    if ts <= self.steady_end:
      return "steady"
    return "teardown"

  def steady_step_of(self, ts: float) -> int | None:
    if not self.step_starts or ts < self.step_starts[0] or ts > self.steady_end:
      return None
    return bisect.bisect_right(self.step_starts, ts) - 1

  def duration(self, phase: str) -> float:
    bounds = {
        "init": (self.init_start, self.init_end),
        "first_batch": (self.init_end, self.first_batch_end),
        "warmup": (self.first_batch_end, self.warmup_end),
        "steady": (self.warmup_end, self.steady_end),
    }[phase]
    return max(0.0, bounds[1] - bounds[0])


def _count_by(items: Iterable[GcsRequest], key) -> dict[str, int]:
  out: dict[str, int] = {}
  for r in items:
    k = key(r)
    out[k] = out.get(k, 0) + 1
  return dict(sorted(out.items()))


def summarize_requests(
    requests: Sequence[GcsRequest],
    marks: Mapping[int, PhaseMarks],
    http_bytes_timeline: Sequence[tuple[float, int, int]] = (),
) -> dict[str, Any]:
  """Aggregates measured requests by phase, host, outcome, and range kind.

  Args:
    requests: Measured requests from client traces or the emulator.
    marks: Phase boundaries for each simulated host. A request is assigned to
      the phase of the host that issued it.
    http_bytes_timeline: Optional (ts, host, bytes) curl receive events, used
      for per-phase received bytes independent of per-request attribution.

  Returns:
    A JSON-serializable summary.
  """
  phase_of: dict[int, str] = {}
  for i, r in enumerate(requests):
    pm = marks.get(r.host)
    phase_of[i] = pm.phase_of(r.ts) if pm else "unknown"

  media = [i for i, r in enumerate(requests) if r.is_media]
  meta = [i for i, r in enumerate(requests) if not r.is_media]

  per_phase: dict[str, Any] = {}
  for phase in PHASES + ("teardown",):
    idx = [i for i in media if phase_of[i] == phase]
    reqs = [requests[i] for i in idx]
    hosts_dur = [marks[h].duration(phase) for h in marks] if phase in PHASES and marks else []
    dur = max(hosts_dur) if hosts_dur else 0.0
    per_phase[phase] = {
        "gets": len(reqs),
        "metadata_requests": sum(1 for i in meta if phase_of[i] == phase),
        "wall_s": round(dur, 4),
        "avg_qps": round(len(reqs) / dur, 2) if dur > 0 else None,
        "peak_1s_qps": peak_window_count([r.ts for r in reqs]),
        "by_outcome": _count_by(reqs, lambda r: r.outcome),
        "by_range_kind": _count_by(reqs, lambda r: r.range_kind),
        "bytes_received": sum(r.bytes_received for r in reqs),
        "requested_bytes_known": sum(r.requested_bytes or 0 for r in reqs),
        "unique_objects": len({r.object_name for r in reqs}),
    }
    if phase == "teardown" and not reqs:
      per_phase.pop("teardown")

  steady_step_counts: dict[tuple[int, int], int] = {}
  for i in media:
    r = requests[i]
    pm = marks.get(r.host)
    if pm is None:
      continue
    step = pm.steady_step_of(r.ts)
    if step is not None:
      steady_step_counts[(r.host, step)] = steady_step_counts.get((r.host, step), 0) + 1
  n_steps = max((len(pm.step_starts) for pm in marks.values()), default=0)
  cluster_step_gets = [sum(steady_step_counts.get((h, s), 0) for h in marks) for s in range(n_steps)]

  per_host: dict[str, Any] = {}
  for h in sorted(marks):
    reqs = [requests[i] for i in media if requests[i].host == h]
    per_host[str(h)] = {
        "gets": len(reqs),
        "by_phase": {p: sum(1 for i in media if requests[i].host == h and phase_of[i] == p) for p in PHASES},
        "unique_objects": len({r.object_name for r in reqs}),
        "pids": sorted({r.pid for r in reqs}),
    }

  bytes_by_phase: dict[str, int] = {}
  for ts, h, n in http_bytes_timeline:
    pm = marks.get(h)
    p = pm.phase_of(ts) if pm else "unknown"
    bytes_by_phase[p] = bytes_by_phase.get(p, 0) + n

  all_media = [requests[i] for i in media]
  first_ts = min((r.ts for r in all_media), default=None)
  return {
      "total_gets": len(all_media),
      "total_metadata_requests": len(meta),
      "metadata_by_method": _count_by((requests[i] for i in meta), lambda r: r.method),
      "by_outcome": _count_by(all_media, lambda r: r.outcome),
      "by_status": _count_by(all_media, lambda r: r.status or "unknown"),
      "by_range_kind": _count_by(all_media, lambda r: r.range_kind),
      "by_category": _count_by(all_media, lambda r: r.category or "uncategorized"),
      "by_source": _count_by(all_media, lambda r: r.source),
      "peak_1s_qps": peak_window_count([r.ts for r in all_media]),
      "startup_peak_1s_qps": peak_window_count([requests[i].ts for i in media if phase_of[i] in ("init", "first_batch")]),
      "bytes_received": sum(r.bytes_received for r in all_media),
      "http_bytes_received_by_phase": bytes_by_phase,
      "unique_objects": len({r.object_name for r in all_media}),
      "first_request_ts": first_ts,
      "steady_gets_per_step": {
          "per_step_cluster": cluster_step_gets,
          "mean": round(sum(cluster_step_gets) / len(cluster_step_gets), 3) if cluster_step_gets else 0.0,
          "max": max(cluster_step_gets, default=0),
      },
      "per_phase": per_phase,
      "per_host": per_host,
  }


# ----------------------------------------------------------------------------
# /proc accounting.
# ----------------------------------------------------------------------------


def parse_proc_net_dev(text: str) -> dict[str, dict[str, int]]:
  """Parses `/proc/net/dev` into {iface: {rx_bytes, rx_packets, tx_bytes, tx_packets}}."""
  out: dict[str, dict[str, int]] = {}
  for line in text.splitlines():
    if ":" not in line:
      continue
    iface, _, data = line.partition(":")
    fields = data.split()
    if len(fields) < 16 or not fields[0].isdigit():
      continue
    out[iface.strip()] = {
        "rx_bytes": int(fields[0]),
        "rx_packets": int(fields[1]),
        "tx_bytes": int(fields[8]),
        "tx_packets": int(fields[9]),
    }
  return out


def read_net_dev(path: str = "/proc/net/dev") -> dict[str, dict[str, int]]:
  """Reads `/proc/net/dev`; returns {} where unavailable."""
  try:
    with open(path, "r", encoding="utf-8") as f:
      return parse_proc_net_dev(f.read())
  except OSError:
    return {}


def net_dev_delta(
    before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]], include_loopback: bool = False
) -> dict[str, Any]:
  """Returns per-interface and total RX/TX deltas between two snapshots."""
  per_iface = {}
  for iface, a in after.items():
    if iface == "lo" and not include_loopback:
      continue
    b = before.get(iface)
    if b is None:
      continue
    per_iface[iface] = {k: a[k] - b[k] for k in a}
  return {
      "rx_bytes": sum(d["rx_bytes"] for d in per_iface.values()),
      "tx_bytes": sum(d["tx_bytes"] for d in per_iface.values()),
      "rx_packets": sum(d["rx_packets"] for d in per_iface.values()),
      "per_interface": per_iface,
  }


def _children_map(proc_root: str = "/proc") -> dict[int, list[int]]:
  """Maps each PID to its child PIDs from `/proc/<pid>/stat`."""
  children: dict[int, list[int]] = {}
  for entry in os.listdir(proc_root):
    if not entry.isdigit():
      continue
    try:
      with open(os.path.join(proc_root, entry, "stat"), "r", encoding="utf-8") as f:
        stat = f.read()
      ppid = int(stat[stat.rfind(")") + 2 :].split()[1])
    except (OSError, ValueError, IndexError):
      continue
    children.setdefault(ppid, []).append(int(entry))
  return children


def process_tree_pids(root_pid: int | None = None, proc_root: str = "/proc") -> list[int]:
  """Returns `root_pid` and all its descendants."""
  root = root_pid or os.getpid()
  children = _children_map(proc_root)
  out, stack = [], [root]
  while stack:
    pid = stack.pop()
    out.append(pid)
    stack.extend(children.get(pid, []))
  return out


def read_process_tree_stats(root_pid: int | None = None, proc_root: str = "/proc") -> dict[str, int]:
  """Sums RSS, open FDs, threads, and `/proc/<pid>/io` counters over a process tree."""
  totals = {"processes": 0, "rss_bytes": 0, "open_fds": 0, "threads": 0, "rchar": 0, "syscr": 0, "read_bytes": 0}
  page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
  for pid in process_tree_pids(root_pid, proc_root):
    base = os.path.join(proc_root, str(pid))
    try:
      with open(os.path.join(base, "statm"), "r", encoding="utf-8") as f:
        totals["rss_bytes"] += int(f.read().split()[1]) * page
      totals["processes"] += 1
    except (OSError, ValueError, IndexError):
      continue
    try:
      totals["open_fds"] += len(os.listdir(os.path.join(base, "fd")))
    except OSError:
      pass
    try:
      with open(os.path.join(base, "status"), "r", encoding="utf-8") as f:
        for line in f:
          if line.startswith("Threads:"):
            totals["threads"] += int(line.split()[1])
    except (OSError, ValueError):
      pass
    try:
      with open(os.path.join(base, "io"), "r", encoding="utf-8") as f:
        for line in f:
          k, _, v = line.partition(":")
          if k in ("rchar", "syscr", "read_bytes"):
            totals[k] += int(v.strip())
    except (OSError, ValueError):
      pass
  return totals


def parse_prometheus_counter(text: str, metric: str) -> dict[str, float]:
  """Sums samples of `metric` from Prometheus text, keyed by the label string."""
  out: dict[str, float] = {}
  pat = re.compile(r"^" + re.escape(metric) + r"(?P<labels>\{[^}]*\})?\s+(?P<v>[-+0-9.eE]+)")
  for line in text.splitlines():
    m = pat.match(line)
    if m:
      key = m.group("labels") or ""
      out[key] = out.get(key, 0.0) + float(m.group("v"))
  return out


# ----------------------------------------------------------------------------
# Sample-order parity digest.
# ----------------------------------------------------------------------------


class RecordOrderDigest:
  """Incremental SHA-256 over the exact sequence of batches a host yields.

  Each batch is hashed key by key in sorted order. Each array contributes its
  key, dtype, shape, and raw C-order bytes, so any reordering, duplication, or
  content change in the sample stream changes the digest.
  """

  def __init__(self, keys: Sequence[str] | None = None):
    self._h = hashlib.sha256()
    self._keys = tuple(keys) if keys else None
    self.batches = 0

  def update(self, batch: Mapping[str, Any]) -> None:
    """Hashes one yielded batch."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    keys = self._keys or tuple(sorted(batch))
    for k in keys:
      if k not in batch:
        continue
      arr = np.ascontiguousarray(np.asarray(batch[k]))
      self._h.update(f"{k}|{arr.dtype.str}|{arr.shape}|".encode("utf-8"))
      self._h.update(arr.tobytes())
    self._h.update(b"\x00batch\x00")
    self.batches += 1

  def hexdigest(self) -> str:
    """Hex SHA-256 of everything hashed so far."""
    return self._h.hexdigest()


# ----------------------------------------------------------------------------
# Optional Cloud Monitoring cross-check.
# ----------------------------------------------------------------------------


def query_cloud_monitoring_request_count(
    project: str, bucket: str, start_ts: float, end_ts: float, alignment_s: int = 60
) -> dict[str, Any]:
  """Sums `storage.googleapis.com/api/request_count` for a bucket and time window.

  The metric is sampled every 60 s and can lag by several minutes, so only use
  it to sanity-check minute-scale totals, never per-step counts. Requires the
  `google-cloud-monitoring` package.

  Args:
    project: GCP project that owns the bucket.
    bucket: Bucket name without the `gs://` prefix.
    start_ts: Window start (Unix epoch seconds).
    end_ts: Window end (Unix epoch seconds).
    alignment_s: Aligner period.

  Returns:
    {"available": bool, "total": int, "by_method_and_code": {...}}.
  """
  try:
    from google.cloud import monitoring_v3  # pylint: disable=import-outside-toplevel
  except ImportError as e:
    return {"available": False, "reason": f"google-cloud-monitoring not installed: {e}"}
  try:
    client = monitoring_v3.MetricServiceClient()
    start = int(start_ts) - (int(start_ts) % alignment_s)
    end = int(end_ts) + alignment_s - (int(end_ts) % alignment_s)
    interval = monitoring_v3.TimeInterval({"start_time": {"seconds": start}, "end_time": {"seconds": end}})
    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": alignment_s},
            "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,
            "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
            "group_by_fields": ["metric.label.method", "metric.label.response_code"],
        }
    )
    series = client.list_time_series(
        request={
            "name": f"projects/{project}",
            "filter": (
                'metric.type="storage.googleapis.com/api/request_count" AND '
                f'resource.type="gcs_bucket" AND resource.label.bucket_name="{bucket}"'
            ),
            "interval": interval,
            "aggregation": aggregation,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    )
    by_key: dict[str, int] = {}
    for ts in series:
      key = f"{ts.metric.labels.get('method', '?')}/{ts.metric.labels.get('response_code', '?')}"
      by_key[key] = by_key.get(key, 0) + sum(int(p.value.int64_value) for p in ts.points)
    return {
        "available": True,
        "window_start": start,
        "window_end": end,
        "total": sum(by_key.values()),
        "by_method_and_code": dict(sorted(by_key.items())),
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {"available": False, "reason": f"{type(e).__name__}: {e}"}
