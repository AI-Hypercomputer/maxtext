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

"""Tests for live GCS trace parsing, request summaries, /proc parsing, digests, and the emulator."""

import os
import tempfile
import unittest
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from benchmarks.input_pipeline import gcs_emulator
from benchmarks.input_pipeline import gcs_trace

# Unix time of 2026-11-19T02:00:00Z.
T0 = 1795053600.0
LOC_STUB = " (google/cloud/storage/internal/logging_stub.cc:192)"
LOC_CURL = " (google/cloud/internal/curl_handle.cc:250)"
LOC_RPC = " (google/cloud/internal/log_wrapper.cc:24)"


def _ts(offset_s: float) -> str:
  whole = int(offset_s)
  nanos = int(round((offset_s - whole) * 1e9))
  return f"2026-11-19T02:00:{whole:02d}.{nanos:09d}Z"


def _rec(offset_s: float, sev: str, tid: str, msg: str, loc: str = LOC_STUB) -> str:
  return f"{_ts(offset_s)} [{sev}] <{tid}> {msg}{loc}\n"


# Format matches google-cloud-cpp LoggingStub (raw-client) + CurlHandle::FlushDebug (http).
REST_TRACE = "".join(
    [
        "I1119 02:00:00.000000 python noise that is not a clog record\n",
        _rec(
            0.10,
            "INFO",
            "11",
            "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=c4/s-00000, read-last=1048576}",
        ),
        _rec(
            0.11,
            "DEBUG",
            "11",
            "PerformWorkUntil == curl(Info): Connected to storage.googleapis.com\n"
            ">> curl(Send Header): GET /storage/v1/b/bkt/o/c4%2Fs-00000?alt=media HTTP/1.1\r\n"
            "Host: storage.googleapis.com\r\n"
            "Range: bytes=-1048576\r\n"
            "authorization: Bearer <redacted>\r\n\n"
            "<< curl(Recv Header): HTTP/1.1 206 Partial Content\r\n\n"
            "<< curl(Recv Header): Content-Length: 1048576\r\n",
            LOC_CURL,
        ),
        _rec(0.12, "DEBUG", "11", "WriteCallback << curl(Recv Data): size=524288 data=......\n", LOC_CURL),
        _rec(0.13, "DEBUG", "11", "WriteCallback << curl(Recv Data): size=524288 data=......\n", LOC_CURL),
        # Bounded range read on another thread.
        _rec(
            0.50,
            "INFO",
            "12",
            "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=c4/s-00001, generation=7, "
            "read-range=ReadRangeData={begin=4096, end=69632}}",
        ),
        _rec(
            0.51,
            "DEBUG",
            "12",
            ">> curl(Send Header): GET /storage/v1/b/bkt/o/c4%2Fs-00001?alt=media&generation=7 HTTP/1.1\r\n"
            "Range: bytes=4096-69631\r\n\n"
            "<< curl(Recv Header): HTTP/2 206 \r\n\n"
            "<< curl(Recv Header): content-length: 65536\r\n\n"
            "<< curl(Recv Data): size=65536 data=....\n",
            LOC_CURL,
        ),
        # Throttled open (429) followed by a retry on the same thread.
        _rec(
            0.90,
            "INFO",
            "11",
            "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=c4/s-00002, read-offset=100}",
        ),
        _rec(
            0.91,
            "DEBUG",
            "11",
            ">> curl(Send Header): GET /storage/v1/b/bkt/o/c4%2Fs-00002?alt=media HTTP/1.1\r\n"
            "Range: bytes=100-\r\n\n"
            "<< curl(Recv Header): HTTP/1.1 429 Too Many Requests\r\n",
            LOC_CURL,
        ),
        _rec(0.92, "INFO", "11", "ReadObject() >> status={RESOURCE_EXHAUSTED: rate limit exceeded error_info={...}}"),
        _rec(
            1.95,
            "INFO",
            "11",
            "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=c4/s-00002, read-offset=100}",
        ),
        _rec(
            1.96,
            "DEBUG",
            "11",
            ">> curl(Send Header): GET /storage/v1/b/bkt/o/c4%2Fs-00002?alt=media HTTP/1.1\r\n"
            "Range: bytes=100-\r\n\n"
            "<< curl(Recv Header): HTTP/1.1 206 Partial Content\r\n\n"
            "<< curl(Recv Header): content-length: 900000\r\n\n"
            "<< curl(Recv Data): size=16384 data=....\n",
            LOC_CURL,
        ),
        # 503 on a fourth object.
        _rec(2.40, "INFO", "13", "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=c4/s-00003}"),
        _rec(
            2.41,
            "DEBUG",
            "13",
            ">> curl(Send Header): GET /storage/v1/b/bkt/o/c4%2Fs-00003?alt=media HTTP/1.1\r\n\n"
            "<< curl(Recv Header): HTTP/1.1 503 Service Unavailable\r\n",
            LOC_CURL,
        ),
        _rec(2.42, "INFO", "13", "ReadObject() >> status={UNAVAILABLE: backend error}"),
        _rec(2.50, "INFO", "13", "ListObjects() << ListObjectsRequest={bucket_name=bkt, prefix=c4/}"),
    ]
)

GRPC_TRACE = "".join(
    [
        _rec(
            5.00,
            "DEBUG",
            "21",
            'ReadObject() << bucket: "projects/_/buckets/bkt" object: "c4/s-00009" read_offset: -1048576',
            LOC_RPC,
        ),
        _rec(5.01, "DEBUG", "21", "ReadObject() >> not null", LOC_RPC),
        _rec(5.02, "DEBUG", "21", "Read(r1)() << (void)", LOC_RPC),
        _rec(5.03, "DEBUG", "21", 'Read(r1)() >> checksummed_data { content: "abc" }', LOC_RPC),
        _rec(5.04, "DEBUG", "21", "Read(r1)() >> OK", LOC_RPC),
        _rec(
            5.10,
            "DEBUG",
            "22",
            'ReadObject() << bucket: "projects/_/buckets/bkt" object: "c4/s-00010" read_offset: 4096 read_limit: 8192',
            LOC_RPC,
        ),
        _rec(5.11, "DEBUG", "22", "Read(r2)() << (void)", LOC_RPC),
        _rec(5.12, "DEBUG", "22", "Read(r2)() >> RESOURCE_EXHAUSTED: Quota exceeded", LOC_RPC),
        _rec(
            5.20,
            "DEBUG",
            "23",
            'ReadObject() << bucket: "projects/_/buckets/bkt" object: "c4/s-00011" read_offset: 64',
            LOC_RPC,
        ),
        _rec(5.21, "DEBUG", "23", "Read(r3)() << (void)", LOC_RPC),
        _rec(5.22, "DEBUG", "23", "Cancel(r3)() << (void)", LOC_RPC),
        _rec(5.30, "DEBUG", "24", 'ReadObject() << bucket: "projects/_/buckets/bkt" object: "c4/s-00012"', LOC_RPC),
        _rec(5.31, "DEBUG", "24", "Read(r4)() >> UNAVAILABLE: connection reset", LOC_RPC),
    ]
)


class ClogParserTest(unittest.TestCase):

  def test_rest_raw_client_with_http_trace(self):
    res = gcs_trace.parse_clog_lines(REST_TRACE.splitlines(keepends=True), host=3, pid=99)
    media = [r for r in res.requests if r.is_media]
    self.assertEqual(len(media), 5)
    self.assertEqual(res.unmatched_http_requests, 0)
    self.assertEqual(res.http_requests_seen, 5)
    r0, r1, r2, r3, r4 = media
    self.assertEqual(
        (r0.object_name, r0.range_kind, r0.last_n, r0.status, r0.outcome), ("c4/s-00000", "last", 1048576, "206", "ok")
    )
    self.assertEqual(r0.bytes_received, 1048576)
    self.assertEqual(r0.content_length, 1048576)
    self.assertEqual((r1.range_kind, r1.start, r1.end, r1.status), ("range", 4096, 69632, "206"))
    self.assertEqual(r1.requested_bytes, 65536)
    self.assertEqual(r1.bytes_received, 65536)
    self.assertEqual((r2.range_kind, r2.start, r2.end, r2.status, r2.outcome), ("offset", 100, None, "429", "throttled"))
    # Retry is a separate measured request; it stopped early (16 KiB of 900000), so it counts as cancelled.
    self.assertEqual((r3.status, r3.outcome, r3.bytes_received), ("206", "cancelled", 16384))
    self.assertEqual((r4.range_kind, r4.status, r4.outcome), ("full", "503", "unavailable"))
    self.assertTrue(all(r.host == 3 and r.pid == 99 for r in res.requests))
    self.assertAlmostEqual(r0.ts, T0 + 0.10, places=6)
    meta = [r for r in res.requests if not r.is_media]
    self.assertEqual([(m.method, m.range_kind) for m in meta], [("ListObjects", "list")])
    self.assertEqual(sum(n for _, _, n in res.http_bytes_timeline), 1048576 + 65536 + 16384)

  def test_http_only_trace_defines_requests(self):
    lines = [
        l for l in REST_TRACE.splitlines(keepends=True) if "ReadObjectRangeRequest" not in l and "ListObjects" not in l
    ]
    res = gcs_trace.parse_clog_lines(lines)
    media = [r for r in res.requests if r.is_media]
    self.assertEqual(len(media), 5)
    self.assertEqual([r.source for r in media], ["http"] * 5)
    self.assertEqual([r.status for r in media], ["206", "206", "429", "206", "503"])
    self.assertEqual([r.range_kind for r in media], ["last", "range", "offset", "offset", "full"])
    self.assertEqual((media[1].start, media[1].end), (4096, 69632))

  def test_grpc_trace(self):
    res = gcs_trace.parse_clog_lines(GRPC_TRACE.splitlines(keepends=True), host=1)
    self.assertEqual(len(res.requests), 4)
    a, b, c, d = res.requests
    self.assertEqual((a.source, a.range_kind, a.last_n, a.outcome), ("grpc", "last", 1048576, "ok"))
    self.assertEqual(
        (b.range_kind, b.start, b.end, b.status, b.outcome), ("range", 4096, 12288, "RESOURCE_EXHAUSTED", "throttled")
    )
    self.assertEqual((c.range_kind, c.start, c.outcome), ("offset", 64, "cancelled"))
    self.assertEqual((d.range_kind, d.outcome), ("full", "unavailable"))
    self.assertEqual(a.bucket, "bkt")

  def test_grpc_lines_not_double_counted_with_raw_client(self):
    lines = (
        _rec(1.0, "INFO", "31", "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=o, read-last=100}")
        + _rec(
            1.0, "DEBUG", "31", 'ReadObject() << bucket: "projects/_/buckets/bkt" object: "o" read_offset: -100', LOC_RPC
        )
        + _rec(1.1, "DEBUG", "31", "Read(z)() >> RESOURCE_EXHAUSTED: slow down", LOC_RPC)
    )
    res = gcs_trace.parse_clog_lines(lines.splitlines(keepends=True))
    self.assertEqual(len(res.requests), 1)
    self.assertEqual(res.grpc_requests_seen, 1)
    self.assertEqual((res.requests[0].source, res.requests[0].outcome), ("raw-client", "throttled"))

  def test_parse_trace_dir_tags_host_and_pid(self):
    with tempfile.TemporaryDirectory() as d:
      with open(os.path.join(d, "host2.pid1234.clog"), "w", encoding="utf-8") as f:
        f.write(REST_TRACE)
      with open(os.path.join(d, "host0.pid55.clog"), "w", encoding="utf-8") as f:
        f.write(GRPC_TRACE)
      with open(os.path.join(d, "unrelated.log"), "w", encoding="utf-8") as f:
        f.write(REST_TRACE)
      res = gcs_trace.parse_trace_dir(d)
    self.assertEqual(sum(1 for r in res.requests if r.is_media), 9)
    self.assertEqual({(r.host, r.pid) for r in res.requests}, {(2, 1234), (0, 55)})
    self.assertEqual([r.ts for r in res.requests], sorted(r.ts for r in res.requests))

  def test_streaming_parse_skips_flush_summaries_and_sums_body_flushes(self):
    lines = [
        _rec(0.10, "INFO", "11", "ReadObject() << ReadObjectRangeRequest={bucket_name=bkt, object_name=o, read-last=64}")
    ]
    lines.append(
        _rec(
            0.11,
            "DEBUG",
            "11",
            ">> curl(Send Header): GET /storage/v1/b/bkt/o/o?alt=media HTTP/1.1\r\nRange: bytes=-64\r\n\n"
            "<< curl(Recv Header): HTTP/1.1 206 Partial Content\r\n",
            LOC_CURL,
        )
    )
    for i in range(2000):
      t = 0.2 + i * 1e-4
      lines.append(
          _rec(t, "DEBUG", "11", "WriteCallback recv_count=1 (0 with no data), send_count=0 (0 with no data).", LOC_CURL)
      )
      lines.append(_rec(t, "DEBUG", "11", "WriteCallback << curl(Recv Data): size=8 data=........\n", LOC_CURL))
    text = "".join(lines).splitlines(keepends=True)
    res = gcs_trace.parse_clog_lines(iter(text), has_raw=gcs_trace.file_has_raw_client_lines(text))
    self.assertEqual(res.records, 2 + 4000)
    self.assertEqual(len(res.requests), 1)
    self.assertEqual((res.requests[0].status, res.requests[0].bytes_received), ("206", 16000))
    self.assertEqual(sum(n for _, _, n in res.http_bytes_timeline), 16000)
    self.assertLess(len(res.http_bytes_timeline), 2000)  # Aggregated into 10 ms buckets.

  def test_parse_trace_dir_parallel_matches_serial(self):
    with tempfile.TemporaryDirectory() as d:
      for h in range(3):
        with open(os.path.join(d, f"host{h}.pid{100 + h}.clog"), "w", encoding="utf-8") as f:
          f.write(REST_TRACE)
      serial = gcs_trace.parse_trace_dir(d, max_workers=1)
      parallel = gcs_trace.parse_trace_dir(d, max_workers=3)

    def key(res):
      return [(r.host, r.pid, r.ts, r.object_name, r.status, r.bytes_received) for r in res.requests]

    self.assertEqual(key(serial), key(parallel))
    self.assertEqual(sum(1 for r in serial.requests if r.is_media), 15)

  def test_client_trace_env(self):
    env = gcs_trace.client_trace_env()
    self.assertEqual(env["GOOGLE_CLOUD_CPP_ENABLE_CLOG"], "yes")
    self.assertEqual(env["CLOUD_STORAGE_ENABLE_TRACING"], "raw-client,http,rpc,rpc-streams")
    self.assertEqual(env["GOOGLE_CLOUD_CPP_ENABLE_TRACING"], "rpc,rpc-streams")
    self.assertEqual(set(env), set(gcs_trace.CLIENT_TRACE_ENV_KEYS))
    self.assertEqual(gcs_trace.client_trace_env(http=False, grpc=False)["CLOUD_STORAGE_ENABLE_TRACING"], "raw-client")
    self.assertNotIn("GOOGLE_CLOUD_CPP_ENABLE_TRACING", gcs_trace.client_trace_env(grpc=False))


class SummaryTest(unittest.TestCase):

  def test_peak_window(self):
    self.assertEqual(gcs_trace.peak_window_count([]), 0)
    self.assertEqual(gcs_trace.peak_window_count([0.0, 0.5, 0.99, 1.0, 1.5, 3.0]), 3)  # Half-open [t, t + 1).
    self.assertEqual(gcs_trace.peak_window_count([0.0, 0.5, 0.99, 1.0, 1.4, 3.0]), 4)
    self.assertEqual(gcs_trace.peak_window_count([0.0, 1.0, 2.0]), 1)
    self.assertEqual(gcs_trace.peak_window_count([5.0] * 7 + [9.0]), 7)

  def test_phase_assignment_and_peaks(self):
    marks = {
        0: gcs_trace.PhaseMarks(0, T0, T0 + 1, T0 + 2, T0 + 3, T0 + 5, step_starts=[T0 + 3, T0 + 4]),
        1: gcs_trace.PhaseMarks(1, T0, T0 + 1.5, T0 + 2.5, T0 + 3.5, T0 + 5.5, step_starts=[T0 + 3.5, T0 + 4.5]),
    }

    def req(ts, host, outcome="ok", kind="range"):
      return gcs_trace.GcsRequest(ts=T0 + ts, host=host, object_name=f"o{host}", outcome=outcome, range_kind=kind)

    reqs = [req(0.1, 0, kind="last")] * 3 + [req(0.2, 1, kind="last")] * 4  # init: 7 GETs within 1s.
    reqs += [req(0.9, 1, kind="last")]  # Host 1 init.
    reqs += [req(1.05, 0)] * 2  # Host 0 is already in first_batch at t=1.05.
    reqs += [req(3.2, 0), req(4.2, 0, "throttled"), req(4.3, 0, "cancelled"), req(3.7, 1)]
    reqs += [gcs_trace.GcsRequest(ts=T0 + 0.05, host=0, method="ListObjects", range_kind="list")]
    s = gcs_trace.summarize_requests(reqs, marks)
    self.assertEqual(s["total_gets"], 14)
    self.assertEqual(s["total_metadata_requests"], 1)
    self.assertEqual(s["per_phase"]["init"]["gets"], 8)
    self.assertEqual(s["per_phase"]["first_batch"]["gets"], 2)
    self.assertEqual(s["per_phase"]["steady"]["gets"], 4)
    self.assertEqual(s["per_phase"]["init"]["peak_1s_qps"], 8)
    self.assertEqual(s["startup_peak_1s_qps"], 10)
    self.assertEqual(s["by_outcome"], {"cancelled": 1, "ok": 12, "throttled": 1})
    self.assertEqual(s["steady_gets_per_step"]["per_step_cluster"], [2, 2])
    self.assertEqual(s["per_host"]["1"]["by_phase"]["init"], 5)
    self.assertEqual(s["per_phase"]["init"]["wall_s"], 1.5)

  def test_no_marks(self):
    s = gcs_trace.summarize_requests([gcs_trace.GcsRequest(ts=T0)], {})
    self.assertEqual(s["total_gets"], 1)


class ProcTest(unittest.TestCase):

  NET_DEV_1 = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000      10    0    0    0     0          0         0     1000      10    0    0    0     0       0          0
  ens4: 5000000   4000    0    0    0     0          0         0   200000    1500    0    0    0     0       0          0
"""
  NET_DEV_2 = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 9000      90    0    0    0     0          0         0     9000      90    0    0    0     0       0          0
  ens4: 7500000   6000    0    0    0     0          0         0   260000    1900    0    0    0     0       0          0
"""

  def test_parse_and_delta(self):
    a = gcs_trace.parse_proc_net_dev(self.NET_DEV_1)
    b = gcs_trace.parse_proc_net_dev(self.NET_DEV_2)
    self.assertEqual(a["ens4"], {"rx_bytes": 5000000, "rx_packets": 4000, "tx_bytes": 200000, "tx_packets": 1500})
    d = gcs_trace.net_dev_delta(a, b)
    self.assertEqual(d["rx_bytes"], 2500000)
    self.assertEqual(d["tx_bytes"], 60000)
    self.assertNotIn("lo", d["per_interface"])
    self.assertEqual(gcs_trace.net_dev_delta(a, b, include_loopback=True)["rx_bytes"], 2508000)

  def test_live_proc(self):
    self.assertIn("lo", gcs_trace.read_net_dev())
    stats = gcs_trace.read_process_tree_stats()
    self.assertGreaterEqual(stats["processes"], 1)
    self.assertGreater(stats["rss_bytes"], 0)
    self.assertGreater(stats["open_fds"], 0)
    self.assertIn(os.getpid(), gcs_trace.process_tree_pids())

  def test_prometheus(self):
    text = (
        "# HELP gcs_request_count x\n"
        'gcs_request_count{gcs_method="NewReader"} 12\n'
        'gcs_request_count{gcs_method="StatObject"} 3\n'
        "gcs_request_count_total 99\n"
    )
    out = gcs_trace.parse_prometheus_counter(text, "gcs_request_count")
    self.assertEqual(out, {'{gcs_method="NewReader"}': 12.0, '{gcs_method="StatObject"}': 3.0})


class DigestTest(unittest.TestCase):

  def _batches(self, seed):
    rng = np.random.default_rng(seed)
    return [
        {"inputs": rng.integers(0, 100, (2, 8), dtype=np.int32), "inputs_segmentation": np.ones((2, 8), np.int32)}
        for _ in range(4)
    ]

  def _digest(self, batches):
    d = gcs_trace.RecordOrderDigest()
    for b in batches:
      d.update(b)
    return d.hexdigest()

  def test_deterministic_and_order_sensitive(self):
    self.assertEqual(self._digest(self._batches(0)), self._digest(self._batches(0)))
    self.assertNotEqual(self._digest(self._batches(0)), self._digest(self._batches(1)))
    b = self._batches(0)
    self.assertNotEqual(self._digest(b), self._digest([b[1], b[0]] + b[2:]))
    # Key insertion order must not matter.
    self.assertEqual(self._digest([dict(reversed(list(x.items()))) for x in b]), self._digest(b))
    # dtype and shape are part of the digest.
    self.assertNotEqual(
        self._digest([{"inputs": np.zeros(4, np.int32)}]), self._digest([{"inputs": np.zeros(4, np.int64)}])
    )
    self.assertNotEqual(self._digest([{"inputs": np.zeros((2, 2))}]), self._digest([{"inputs": np.zeros(4)}]))


class EmulatorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    os.makedirs(os.path.join(self.root, "d"))
    self.payload = bytes(range(256)) * 400  # 102400 bytes.
    with open(os.path.join(self.root, "d", "obj-0"), "wb") as f:
      f.write(self.payload)

  def _get(self, endpoint, path, rng=None):
    req = urllib_request.Request(endpoint + path, headers={"Range": rng} if rng else {})
    try:
      with urllib_request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()
    except urllib_error.HTTPError as e:
      return e.code, e.read()

  def test_ranges_list_and_fault_injection(self):
    srv = gcs_emulator.InstrumentedGCSServer(self.root, "bkt", host_index=4, fail_every_n=4, fail_status=503)
    ep = srv.start()
    try:
      self.assertEqual(gcs_emulator.emulator_env(ep)["CLOUD_STORAGE_EMULATOR_ENDPOINT"], ep)
      media = "/storage/v1/b/bkt/o/d%2Fobj-0?alt=media"
      self.assertEqual(self._get(ep, media, "bytes=-1000"), (206, self.payload[-1000:]))
      self.assertEqual(self._get(ep, media, "bytes=10-19"), (206, self.payload[10:20]))
      self.assertEqual(self._get(ep, media, "bytes=102000-"), (206, self.payload[102000:]))
      self.assertEqual(self._get(ep, media)[0], 503)  # 4th media GET is injected.
      self.assertEqual(self._get(ep, media), (200, self.payload))
      code, body = self._get(ep, "/storage/v1/b/bkt/o?prefix=d/&matchGlob=d/obj-*")
      self.assertEqual(code, 200)
      self.assertIn(b'"name": "d/obj-0"', body)
      self.assertEqual(self._get(ep, "/storage/v1/b/bkt/o/d%2Fmissing?alt=media")[0], 404)
    finally:
      srv.stop()
    reqs = srv.snapshot()
    media_reqs = [r for r in reqs if r.is_media]
    self.assertEqual([r.range_kind for r in media_reqs], ["last", "range", "offset", "full", "full"])
    self.assertEqual([r.status for r in media_reqs], ["206", "206", "206", "503", "200"])
    self.assertEqual([r.outcome for r in media_reqs], ["ok", "ok", "ok", "unavailable", "ok"])
    self.assertEqual([r.bytes_received for r in media_reqs], [1000, 10, 400, 0, 102400])
    self.assertTrue(all(r.host == 4 and r.source == "emulator" for r in reqs))
    self.assertEqual(sum(1 for r in reqs if r.method == "ListObjects"), 1)

  def test_footer_offset_without_postscript(self):
    path = os.path.join(self.root, "d", "obj-0")
    self.assertEqual(gcs_emulator.extract_arrayrecord_footer_offset(path), len(self.payload) - 65536)


if __name__ == "__main__":
  unittest.main()
