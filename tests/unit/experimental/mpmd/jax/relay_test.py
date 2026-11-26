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

"""transfer.py's host relay (`--transport relay`): RelayChannel / RelayContext.

Three layers, cheapest first:
  * in ONE process on 8 CPU devices: RelayContexts on loopback that a fake
    key-value store introduces, each playing one process (the routes
    `relay_routes` would give each are patched in) -- round trips, receives
    claimed before and after their data, chunks spread over every stream,
    two senders feeding one receiver (each shard on its device as it
    completes), sent shards let go once written, the timing samples, what a
    channel refuses, how a receive fails (a timeout naming the channel, the
    sequence number and the missing peer; a sender's error; a peer that
    closed, or whose streams did not all open; avals that differ between the
    two sides; the context closing under it), a listener outliving a failed
    accept, a failed receive's late shards drained, and ds_ep_pipeline's act
    / cotangents storage resolving a receive when it is popped;
  * two JAX processes (jax.distributed over localhost, `relay_check`):
    RelayChannels both ways, leaves of five dtypes and shardings, 120 sends
    each way, every value checked, then a 64 MiB boundary 16 times for a
    throughput number;
  * `ds_ep_pipeline --distributed --transport relay` against the same
    pipeline in one process (multihost_test's `_compare`): the first loss bit
    for bit where the other transports' is, the rest to multihost_test's
    tolerances -- 2 to 4 processes, stages within a process and across two,
    circular and V placements, interleaved 1F1B's F&B pairs, with and without overlap.

The first two run on every PR (seconds, no model); the pipelines, several JAX
processes each, are marked scheduled_only (MaxText's scheduled CI runs them).

    export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8
    python3 -m pytest -q tests/unit/experimental/mpmd/jax/relay_test.py
    python3 -m pytest -q -m 'not scheduled_only' tests/unit/experimental/mpmd/jax/relay_test.py   # the first two
    # MiB/s between 2 processes:
    python3 -m tests.unit.experimental.mpmd.jax.relay_test bench --mib 256 --sends 16 --streams 8
"""

from __future__ import annotations

import argparse
import collections
import errno
import gc
import os
import re
import socket
import sys
import threading
import time
import unittest
import warnings
import weakref
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

from tests.unit.experimental.mpmd.jax import multihost_test as mh
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer
# the channel tests' stages and fake hosts: AXES / ACT / AVAL, mesh(), FakeDevice, fake_mesh()
from tests.unit.experimental.mpmd.jax.transfer_channels_test import ACT, AVAL, AXES, FakeDevice, fake_mesh, mesh
from maxtext.experimental.mpmd.jax.transfer import RelayError, RelayTimeoutError, ShardingMismatchError

# The two-process check's leaves: (shape, dtype, spec over its stage's one axis 'x').
LEAVES = (
    ((8, 6, 4), "bfloat16", P("x")),
    ((), "float32", P()),
    ((5, 3), "int32", P()),
    ((4, 2), "bool", P("x")),
    ((6, 10), "float16", P(None, "x")),
)


class FakeKV:
  """The two calls RelayContext makes of jax.distributed's client."""

  def __init__(self):
    self.d, self.cv = {}, threading.Condition()

  def key_value_set(self, k, v, allow_overwrite=False):
    with self.cv:
      assert allow_overwrite or k not in self.d, k
      self.d[k] = v
      self.cv.notify_all()

  def blocking_key_value_get(self, k, timeout_ms):
    with self.cv:
      if not self.cv.wait_for(lambda: k in self.d, timeout_ms / 1000):
        raise TimeoutError(k)
      return self.d[k]


def close_all(*ctxs):
  """Close the contexts at once, as every process does at the end of a run
  (each waits for the others' streams into it to end)."""
  ts = [threading.Thread(target=c.close) for c in ctxs]
  for t in ts:
    t.start()
  for t in ts:
    t.join()


def relay_threads():
  return sorted(t.name for t in threading.enumerate() if t.name.startswith("relay-"))


def wait_for_threads(test, *prefixes, timeout=5):
  """Wait for the relay threads named `prefixes`... to end: fail the test
  if some are still running after `timeout` seconds."""
  deadline = time.monotonic() + timeout
  left = lambda: [n for n in relay_threads() if n.startswith(prefixes)]
  while left() and time.monotonic() < deadline:
    time.sleep(0.02)
  test.assertEqual(left(), [])


def value(shape, dtype, *seed):
  rng = np.random.default_rng(seed)
  if dtype == "bool":
    return rng.random(shape) < 0.5
  if dtype == "int32":
    return rng.integers(-1000, 1000, shape, dtype=np.int32)
  return np.asarray(rng.standard_normal(shape), np.float32).astype(jnp.dtype(dtype))


@testing.needs_devices(8, "two stages of 2x2 devices: the routes and shard counts are written for four shards a stage")
class Loopback(unittest.TestCase):
  """Two stages of 2x2 devices of this process, a on devices 0-3 and b on
  4-7, and two RelayContexts on loopback standing in for process 0 (holding
  a) and process 1 (holding b)."""

  def setUp(self):
    self.devs = jax.devices()[:8]
    self.a, self.b = mesh(self.devs[:4]), mesh(self.devs[4:])
    self.src, self.dst = NamedSharding(self.a, ACT), NamedSharding(self.b, ACT)
    self.kv = FakeKV()
    self.ctxs = []

  def tearDown(self):
    with warnings.catch_warnings():  # what a failure test left behind: it checks its own
      warnings.simplefilter("ignore", RuntimeWarning)
      close_all(*self.ctxs)
    self.assertEqual(relay_threads(), [])  # close() leaves no thread behind

  def context(self, pid, **kw):
    kw = dict(dict(streams=2, timeout_s=60, key=f"test/{self.id()}"), **kw)
    c = transfer.RelayContext(kv=self.kv, pid=pid, ip="127.0.0.1", **kw)
    self.ctxs.append(c)
    return c

  def pair(self, avals=AVAL, src=None, dst=None, idx=3, kw0=None, kw1=None, avals1=None, c1=None):
    """(process 0's channel, process 1's channel) of one boundary a -> b
    (process 1's context `c1` if given)."""
    c0, c1 = self.context(0, **(kw0 or {})), c1 or self.context(1, **(kw1 or {}))
    src, dst = src or self.src, dst or self.dst
    with mock.patch.object(transfer, "relay_routes", lambda *_: ({1: [0, 1, 2, 3]}, {})):
      s = transfer.RelayChannel(avals, src, dst, c0, idx, name="s0->s1")
    with mock.patch.object(transfer, "relay_routes", lambda *_: ({}, {0: [0, 1, 2, 3]})):
      r = transfer.RelayChannel(avals if avals1 is None else avals1, src, dst, c1, idx, name="s0->s1")
    return s, r

  def two_senders(self, idx=4):
    """(process 0's, process 2's, process 1's channel) of one boundary a -> b
    whose source positions 0-1 are on process 0 and 2-3 on process 2, the
    whole destination on process 1 (a pairing XferChannel refuses)."""
    return self.boundary(idx, (0, ({1: [0, 1]}, {})), (2, ({1: [2, 3]}, {})), (1, ({}, {0: [0, 1], 2: [2, 3]})))

  def two_receivers(self, idx=6):
    """(process 0's, process 1's, process 2's channel) of one boundary a -> b
    whose whole source is on process 0, destination positions 0-1 on process
    1 and 2-3 on process 2."""
    return self.boundary(idx, (0, ({1: [0, 1], 2: [2, 3]}, {})), (1, ({}, {0: [0, 1]})), (2, ({}, {0: [2, 3]})))

  def hung_peer(self, pid, streams=2):
    """A listener published as process `pid`'s that accepts `streams`
    connections and never reads them, its receive buffers 64 KiB -> (the
    listening socket, the list the accepted (connection, address) pairs go
    into).  Both are closed when the test ends."""
    hung = socket.socket()
    hung.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)
    hung.bind(("127.0.0.1", 0))
    hung.listen(8)
    hung.settimeout(30)
    held = []
    threading.Thread(target=lambda: [held.append(hung.accept()) for _ in range(streams)], daemon=True).start()
    self.kv.key_value_set(f"test/{self.id()}/addr/{pid}", f"127.0.0.1:{hung.getsockname()[1]} {bytes(16).hex()}")

    def close():
      for conn, _ in held:
        conn.close()
      try:
        hung.shutdown(socket.SHUT_RDWR)  # wakes a thread still in accept()
      except OSError:
        pass
      hung.close()

    self.addCleanup(close)
    return hung, held

  def raw_stream(self, ctx, pid=0, streams=1):
    """A connection into `ctx`, opened as process `pid`'s stream is (its
    HELLO, with the token `ctx` published), for a test to write messages by
    hand.  Closed when the test ends."""
    addr, token = self.kv.d[f"{ctx.key}/addr/{ctx.pid}"].split()
    host, port = addr.rsplit(":", 1)
    s = socket.create_connection((host, int(port)))
    self.addCleanup(s.close)
    s.sendall(
        transfer._RELAY_HDR.pack(transfer._RELAY_MAGIC, transfer._HELLO, pid, 0, 0, 0, 0, 16, streams, ctx._key_sig)
        + bytes.fromhex(token)
    )
    return s

  def boundary(self, idx, *procs):
    """Each (process, its relay_routes) of one boundary a -> b -> its channel
    there, on a context of its own."""
    chans = []
    for pid, routes in procs:
      c = self.context(pid)
      with mock.patch.object(transfer, "relay_routes", lambda *_, routes=routes: routes):
        chans.append(transfer.RelayChannel(AVAL, self.src, self.dst, c, idx, name="s0->s1"))
    return chans


class RoundTripTest(Loopback):

  def test_round_trip(self):
    """Every leaf of a pytree -- a bf16 activation, an f32 scalar, an int32
    matrix, replicated or sharded -- arrives exact on its destination
    sharding, send after send, whether the receive is claimed before its
    data comes or after; the source arrays are left as they were, and both
    sides end with the same number of sends and nothing left over."""
    avals = {
        "y": jax.ShapeDtypeStruct((8, 16, 8), jnp.bfloat16),
        "sq": jax.ShapeDtypeStruct((), jnp.float32),
        "ids": jax.ShapeDtypeStruct((4, 3), jnp.int32),
    }
    src = {"y": self.src, "sq": NamedSharding(self.a, P()), "ids": NamedSharding(self.a, P(AXES, None))}
    dst = {"y": self.dst, "sq": NamedSharding(self.b, P()), "ids": NamedSharding(self.b, P(AXES, None))}
    s, r = self.pair(avals, src, dst)
    self.assertEqual((s.owns_src, s.owns_dst, r.owns_src, r.owns_dst), (True, False, False, True))
    none = {k: None for k in avals}
    for i in range(30):
      host = {"y": value((8, 16, 8), "bfloat16", i), "sq": np.float32(i - 0.25), "ids": value((4, 3), "int32", i)}
      x = jax.device_put(host, src)
      if i % 2:  # claimed first: the future completes later
        out = r.send(none)
        self.assertEqual(s.send(x), none)
      else:
        self.assertEqual(s.send(x), none)
        out = r.send(none)
      self.assertTrue(all(type(f) is transfer.RelayFuture for f in out.values()), out)
      for k in host:
        y = transfer.resolve(out[k])
        self.assertEqual(y.sharding, dst[k], k)
        np.testing.assert_array_equal(np.asarray(y), host[k], k)
        np.testing.assert_array_equal(np.asarray(x[k]), host[k], k)
    self.assertEqual((s.seq, r.seq), (30, 30))
    st = r.ctx.stats()
    self.assertEqual((st["unclaimed"], st["waiting"]), ([], []))
    self.assertEqual(st["received"], s.ctx.stats()["sent"])
    self.assertEqual(transfer.resolve(5), 5)  # anything else passes through

  def test_an_equivalent_source_sharding(self):
    """A source array on a sharding equivalent to the declared one -- the
    same devices holding the same slices, spelled P(AXES) for the declared
    P(AXES, None, None), which jax 0.11.1 does not count equal -- is the
    declared array: its shards are taken by device.  (It used to be refused,
    as another sharding.)"""
    s, r = self.pair()
    want = value(AVAL.shape, "float32", 15)
    s.send_flat([jax.device_put(want, NamedSharding(self.a, P(AXES)))])
    np.testing.assert_array_equal(np.asarray(transfer.resolve(r.send_flat([None])[0])), want)

  def test_claimed_after_it_completed(self):
    """The order of a consumer that is behind: a receive's data all come in
    first, so it completes unclaimed (stats say so), and its claim finds it
    done -- retires it at once, the future done before it is asked, the
    consumer's wait nothing -- send after send, the numbers retired in order
    and nothing left over.  (The round-trip tests claim before or while the
    data come: this path had no loopback test.)"""
    s, r = self.pair()
    for i in range(4):
      want = value(AVAL.shape, "float32", 16, i)
      s.send_flat([jax.device_put(want, self.src)])
      deadline = time.monotonic() + 30
      while (r.idx, i) not in r.ctx._recvs or not r.ctx._recvs[(r.idx, i)].event.is_set():
        self.assertLess(time.monotonic(), deadline)
        time.sleep(0.005)
      self.assertEqual(r.ctx.stats()["unclaimed"], [f"s0->s1 (relay channel #3) seq {i}"])
      f = r.send_flat([None])[0]
      self.assertTrue(f.done())
      self.assertEqual(((r.idx, i) in r.ctx._recvs, r.ctx._floor[r.idx]), (False, i + 1))  # retired at its claim
      np.testing.assert_array_equal(np.asarray(f.result(timeout=0)), want)
    t = r.ctx.timing()["channels"]["s0->s1"]
    self.assertEqual([t[m]["n"] for m in ("recv_wire", "recv_dispatch", "recv_wait")], [4, 4, 4])
    self.assertLess(t["recv_wait"]["max_ms"], 100)
    st = r.ctx.stats()
    self.assertEqual((st["unclaimed"], st["waiting"], r.ctx._retired[r.idx]), ([], [], set()))

  def test_chunks_over_every_stream(self):
    """With 1 KiB chunks over 4 streams a 1 MiB boundary is 1024 chunks
    arriving in no particular order across the streams, each written
    straight into its shard at its offset."""
    aval = jax.ShapeDtypeStruct((64, 64, 64), jnp.float32)
    s, r = self.pair(aval, kw0=dict(streams=4, chunk_mb=1 / 1024))
    for i in range(3):
      want = value((64, 64, 64), "float32", 7, i)
      s.send_flat([jax.device_put(want, self.src)])
      np.testing.assert_array_equal(np.asarray(transfer.resolve(r.send_flat([None])[0])), want)
    self.assertEqual(len(s.ctx._peers[1].socks), 4)

  def test_receives_in_any_order(self):
    """A receive takes its own sequence number's data whatever completes
    first: five claimed ahead of their sends, then three sent ahead of their
    claims, resolved newest first."""
    s, r = self.pair()
    want = [value(AVAL.shape, "float32", 3, i) for i in range(8)]
    futs = [r.send_flat([None])[0] for _ in range(5)]
    for w in want[:5]:
      s.send_flat([jax.device_put(w, self.src)])
    for w in want[5:]:
      s.send_flat([jax.device_put(w, self.src)])
    futs += [r.send_flat([None])[0] for _ in range(3)]
    for f, w in reversed(list(zip(futs, want))):
      np.testing.assert_array_equal(np.asarray(f.result()), w)

  def test_two_senders_one_receiver(self):
    """Two source processes, one destination (`two_senders`): a receive is
    complete with both peers' shards, and until then says whose are missing
    -- while each shard that is in is on its device already."""
    s0, s2, r = self.two_senders()
    want = value(AVAL.shape, "float32", 5)
    x = jax.device_put(want, self.src)
    f = r.send_flat([None])[0]
    s0.send_flat([x])
    # process 0's two shards in -- and each on its device already, before the receive is complete
    deadline = time.monotonic() + 30
    while len(f._r.placed) < 2 and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertEqual(sorted(d.id for y in f._r.placed.values() for d in y.devices()), [d.id for d in self.devs[4:6]])
    with self.assertRaisesRegex(
        RelayTimeoutError, r"2 of 4 shards in after 0\.5 s, waiting on process 2 " r"\(2 shards\)$"
    ):
      f.result(timeout=0.5)
    s2.send_flat([x])
    np.testing.assert_array_equal(np.asarray(f.result()), want)

  def test_sent_shards_are_let_go(self):
    """Nothing of a written shard stays with the relay's threads: once its
    chunks are written and the sender drops the array, its shards go (a
    writer used to hold the last shard it wrote until its next chunk came:
    on TPU an HBM buffer and its host copy per writer, between bursts)."""
    s, r = self.pair(kw0=dict(streams=4))
    refs = []
    for i in range(4):
      want = value(AVAL.shape, "float32", 9, i)
      x = jax.device_put(want, self.src)
      refs += [weakref.ref(sh.data) for sh in x.addressable_shards]
      s.send_flat([x])
      del x
      np.testing.assert_array_equal(np.asarray(transfer.resolve(r.send_flat([None])[0])), want)
    self.assertTrue(s.ctx._peers[1].flush(10))
    deadline = time.monotonic() + 10
    while any(w() is not None for w in refs) and time.monotonic() < deadline:
      gc.collect()  # on CPU a shard and the host view it caches are a cycle
      time.sleep(0.01)
    self.assertEqual([w() for w in refs if w() is not None], [])

  def test_receive_buffers_are_reused_once_free(self):
    """A shard's host buffer goes back to a later receive of that shard once
    nothing holds it, and never while anything does.  Once a shard is
    placed, what holds its buffer is jax: on TPU until the host-to-device
    copy is done; on CPU, if jax put a view of the buffer on the device, for
    as long as the placed array lives, and if it copied the buffer, until
    the next garbage collection (jax drops what its threads let go of in a
    gc callback).  jax 0.11.1 to 0.12.0.1 view a buffer only if its address
    is a multiple of 64 bytes, wherever the allocator happened to put it.
    So the test holds each buffer itself once it is placed from, as such an
    array would (that a viewing array holds its buffer is checked first):
    six receives each get buffers of their own, the pool filling to
    _RELAY_POOL a position, every value exact; let go, three more take
    pooled buffers only.  (Every shard used to get a new buffer, faulted in
    and zeroed.)"""
    s, r = self.pair()
    n = r._nbytes[0]
    for off in (0, 16):  # bytes past a 64-byte boundary: jax views the first and copies the second
      owner = np.zeros(n + 128, np.uint8)
      lo = -owner.ctypes.data % 64 + off
      refs = sys.getrefcount(owner)
      y = r._place((0, r._recv_pos[0]), owner[lo : lo + n])
      y.block_until_ready()
      owner[lo : lo + n] = 1  # the array shows this only if it views the buffer
      if np.asarray(y).any():
        self.assertGreater(sys.getrefcount(owner), refs, off)
      del y
      gc.collect()
      self.assertEqual(sys.getrefcount(owner), refs, off)
    used, real = [], transfer.RelayChannel._place
    # `used` holds every buffer placed from until cleared
    with mock.patch.object(transfer.RelayChannel, "_place", lambda ch, key, buf: used.append(buf) or real(ch, key, buf)):
      held = []
      for i in range(6):
        want = value(AVAL.shape, "float32", 17, i)
        s.send_flat([jax.device_put(want, self.src)])
        held.append((transfer.resolve(r.send_flat([None])[0]), want))
      for y, want in held:
        np.testing.assert_array_equal(np.asarray(y), want)
      self.assertEqual(len({b.ctypes.data for b in used}), 24)  # 6 receives x 4 shards, a buffer each
      pooled = {b.ctypes.data for p in r.ctx._pool.values() for b in p}
      self.assertEqual((len(r.ctx._pool), len(pooled)), (4, 4 * transfer._RELAY_POOL))  # 4 positions, full
      del held, y
      used.clear()
      gc.collect()  # on CPU a shard and the host view it caches are a cycle, and jax lets go of copied buffers
      for i in range(3):
        want = value(AVAL.shape, "float32", 18, i)
        s.send_flat([jax.device_put(want, self.src)])
        np.testing.assert_array_equal(np.asarray(transfer.resolve(r.send_flat([None])[0])), want)
    self.assertEqual(len({b.ctypes.data for b in used}), 12)  # 3 receives x 4 shards, held again
    self.assertLessEqual({b.ctypes.data for b in used}, pooled)  # none new
    self.assertEqual({b.ctypes.data for p in r.ctx._pool.values() for b in p}, pooled)

  def test_timing(self):
    """Where the relay's time went: a sample per shard sent (on the host,
    written), per receive (wire, dispatch) and per receive resolved (its
    consumer's wait), per channel and in all, each counted before a flush
    or the consumer can see it done; reset forgets them."""
    s, r = self.pair()
    for i in range(5):
      s.send_flat([jax.device_put(value(AVAL.shape, "float32", 4, i), self.src)])
      transfer.resolve(r.send_flat([None])[0])
    self.assertTrue(s.ctx._peers[1].flush(10))
    ts, tr = s.ctx.timing(), r.ctx.timing()
    self.assertEqual({m: x["n"] for m, x in ts["all"].items()}, {"send_host": 20, "send_wire": 20})
    self.assertEqual({m: x["n"] for m, x in tr["all"].items()}, {"recv_wire": 5, "recv_dispatch": 5, "recv_wait": 5})
    self.assertEqual((list(ts["channels"]), list(tr["channels"])), (["s0->s1"], ["s0->s1"]))
    self.assertLessEqual(set(ts["all"]) | set(tr["all"]), set(transfer.RELAY_MEASURES))
    for x in [*ts["all"].values(), *tr["all"].values()]:
      self.assertTrue(0 <= x["p50_ms"] <= x["p99_ms"] <= x["max_ms"] <= x["total_s"] * 1e3 + 1e-6, x)
    self.assertRegex(
        transfer.relay_timing_line(ts),
        r"^sent 20 shards \(on the host p50 [\d.]+ / p99 [\d.]+ ms, " r"written p50 [\d.]+ / p99 [\d.]+ ms\)$",
    )
    self.assertRegex(
        transfer.relay_timing_line(tr),
        r"^received 5 \(wire p50 [\d.]+ / p99 [\d.]+ ms, dispatched to "
        r"the devices p50 [\d.]+ / p99 [\d.]+ ms\); consumers waited p50 [\d.]+ / p99 [\d.]+ ms, "
        r"[\d.]+ s in all \(s0->s1 [\d.]+ s\)$",
    )
    self.assertEqual(r.ctx.timing(reset=True), tr)
    self.assertEqual(r.ctx.timing(), {"all": {}, "channels": {}})
    self.assertEqual(transfer.relay_timing_line(r.ctx.timing()), "no transfers")
    self.assertRegex(
        r.ctx.describe(),
        r"^relay test/\S+ p1 at 127\.0\.0\.1:\d+: 2 streams per peer, 4 MiB chunks, "
        r"socket buffers left to the kernel's autotuning, timeout 60 s; this host: thp: .*, tcp_rmem: ",
    )

  def test_timing_counts_every_sample(self):
    """Counts, totals and maxima cover every sample since the last reset;
    only the percentiles come from the newest _RELAY_SAMPLES kept per
    channel (here 4), and `kept` says how many those are.  (The counts and
    totals used to be the kept samples', worded as totals: past 16384 shards
    a channel's "sent N shards" and "s in all" stopped growing.)"""
    c = self.context(0)
    with mock.patch.object(transfer, "_RELAY_SAMPLES", 4):
      for i in range(10):
        c._note("send_host", "a", 0.001 * (i + 1))
        c._note("send_wire", "a", 0.002)
      c._note("send_host", "b", 0.5)
    t = c.timing()
    a, every = t["channels"]["a"]["send_host"], t["all"]["send_host"]
    self.assertEqual((a["n"], a["kept"], every["n"], every["kept"]), (10, 4, 11, 5))
    np.testing.assert_allclose(
        [a["total_s"], a["max_ms"], a["p50_ms"], every["total_s"], every["max_ms"]], [0.055, 10, 8.5, 0.555, 500]
    )  # p50 of the kept 7-10 ms
    self.assertRegex(transfer.relay_timing_line(t), r"^sent 11 shards ")

  def test_trace_annotations(self):
    """A profile shows the relay's three data-moving steps by name, on the
    host plane: a `relay_send` per chunk written, a `relay_recv` per chunk
    read, a `relay_place` per shard put on its device -- what a trace reader
    measures the boundary off, ahead of the Python tracer's frames, which
    this capture also has.  The names are fixed: such readers match them."""
    import glob
    import tempfile
    from maxtext.experimental.mpmd.jax import xprof_align

    self.assertEqual(transfer.RELAY_TRACE, {"send": "relay_send", "recv": "relay_recv", "place": "relay_place"})
    s, r = self.pair(kw0={"chunk_mb": 1 / 4096}, kw1={"chunk_mb": 1 / 4096})  # 512 B shards: 2 chunks each
    x = jax.device_put(value(AVAL.shape, "float32", 5), self.src)
    with tempfile.TemporaryDirectory() as d:
      with jax.profiler.trace(d):
        for _ in range(3):
          s.send_flat([x])
          y = transfer.resolve(r.send_flat([None])[0])
        jax.block_until_ready(y)
        self.assertTrue(s.ctx._peers[1].flush(10))
      (path,) = glob.glob(os.path.join(d, "**", "*.xplane.pb"), recursive=True)
      (host,) = [pl for pl in xprof_align._load(path).planes if pl.name == "/host:CPU"]
      events = collections.Counter(
          host.event_metadata[e.metadata_id].name.split("#")[0] for ln in host.lines for e in ln.events
      )
    self.assertEqual(
        {k: events[v] for k, v in transfer.RELAY_TRACE.items()}, {"send": 3 * 4 * 2, "recv": 3 * 4 * 2, "place": 3 * 4}
    )
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x))

  def test_bystander(self):
    """A process holding neither stage takes no part: None back at once, no
    stream opened, no number taken."""
    c = self.context(5)
    with mock.patch.object(transfer, "relay_routes", lambda *_: ({}, {})):
      ch = transfer.RelayChannel(AVAL, self.src, self.dst, c, 0)
    self.assertFalse(ch.owns_src or ch.owns_dst)
    self.assertEqual((ch.send_flat([None]), ch.seq, c._peers), ([None], 0, {}))


class FailureTest(Loopback):

  def test_timeout_names_channel_seq_and_peer(self):
    """A receive whose data never comes: after MPMD_RELAY_TIMEOUT (here the
    call's own timeout) RelayTimeoutError says which channel, which number,
    and whose shards are missing."""
    _, r = self.pair()
    f = r.send_flat([None])[0]
    self.assertFalse(f.done())
    with self.assertRaisesRegex(
        RelayTimeoutError,
        r"s0->s1 \(relay channel #3\) seq 0: 0 of 4 shards in " r"after 0\.2 s, waiting on process 0 \(4 shards\)",
    ):
      f.result(timeout=0.2)
    self.assertIsInstance(RelayTimeoutError("x"), TimeoutError)

  def test_sender_error_fails_the_receive(self):
    """A shard whose copy to host fails (its program did) reaches the peer as
    an error: that receive raises it at once, instead of timing out, and the
    sender's next send raises too."""
    s, r = self.pair()

    class Broken:

      def __array__(self, *a, **k):
        raise RuntimeError("the producing program failed")

    p = s.ctx._peer(1)
    for k in range(4):  # what send_flat queues, but with shards that cannot reach the host
      p.put((s.idx, s.seq, 0, k, Broken(), s._nbytes[0], s._sigs[0], s.name, time.monotonic()))
    s.seq += 1
    f = r.send_flat([None])[0]
    with self.assertRaisesRegex(
        RelayError, "seq 0: process 0 could not send it: RuntimeError: the producing " "program failed"
    ):
      f.result(timeout=30)
    with self.assertRaisesRegex(RelayError, "could not send to process 1"):
      s.send_flat([jax.device_put(np.zeros(AVAL.shape, np.float32), self.src)])
    # that send took its number all the same, and told the peer: its receive fails at once too
    self.assertEqual(s.seq, 2)
    with self.assertRaisesRegex(
        RelayError, r"seq 1: process 0 could not send it: an earlier send failed \(.*the " "producing program failed"
    ):
      r.send_flat([None])[0].result(timeout=10)

  def test_a_broken_stream_fails_every_destination(self):
    """One sender, two destination processes, and its streams to process 2
    break: its next send raises -- and still takes its number and tells every
    destination it cannot come, so the healthy process 1's receive of it
    fails at once, with the reason, instead of waiting out MPMD_RELAY_TIMEOUT
    (process 2 sees its streams end).  The numbers stay in step: the send
    after it fails the same way.  (In one process a destination holding half
    of a stage cannot assemble its array, so no data moves here.)"""
    s, r1, r2 = self.two_receivers()
    x = jax.device_put(value(AVAL.shape, "float32", 12), self.src)
    p2 = s.ctx._peer(2)  # its streams open, with nothing sent
    deadline = time.monotonic() + 30
    while len(p2.writers) < 2 and time.monotonic() < deadline:
      time.sleep(0.01)
    p2._fail(ConnectionResetError(errno.ECONNRESET, "the stream broke (by the test)"))
    for seq in (0, 1):
      f1, f2 = r1.send_flat([None])[0], r2.send_flat([None])[0]
      t0 = time.monotonic()
      with self.assertRaisesRegex(RelayError, "relay to process 2: ConnectionResetError"):
        s.send_flat([x])
      with self.assertRaisesRegex(
          RelayError,
          rf"seq {seq}: process 0 could not send it: an earlier send failed "
          r"\(relay to process 2: ConnectionResetError",
      ):
        f1.result(timeout=10)
      with self.assertRaisesRegex(
          RelayError, rf"seq {seq}: process(\(es\))? \[?0\]? closed (its|their) relay " "streams"
      ):
        f2.result(timeout=10)
      self.assertLess(time.monotonic() - t0, 10)
    self.assertEqual((s.seq, r1.seq, r2.seq), (2, 2, 2))

  def test_avals_differ_between_the_sides(self):
    """Two processes that built the channel with different avals: the
    receiver's checks catch the first chunk (size and dtype/shape CRC) and
    fail the receive, and every one after it."""
    s, r = self.pair(avals1=jax.ShapeDtypeStruct(AVAL.shape, jnp.bfloat16))
    s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])
    f = r.send_flat([None])[0]
    with self.assertRaisesRegex(RelayError, "built the channel with different avals"):
      f.result(timeout=30)
    with self.assertRaisesRegex(RelayError, "different avals"):
      r.send_flat([None])[0].result(timeout=1)

  def test_peer_that_closed(self):
    """Once every stream of a peer has ended, a receive still missing its
    shards fails at once -- the one waiting, and one claimed after."""
    s, r = self.pair()
    s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])
    transfer.resolve(r.send_flat([None])[0])  # the streams exist
    f = r.send_flat([None])[0]
    s.ctx.close()
    t0 = time.monotonic()
    with self.assertRaisesRegex(
        RelayError, r"seq 1: process 0 closed its relay streams with 4 shard\(s\) " "of it unsent"
    ):
      f.result(timeout=30)
    self.assertLess(time.monotonic() - t0, 20)
    with self.assertRaisesRegex(RelayError, r"seq 2: process\(es\) \[0\] closed their relay streams"):
      r.send_flat([None])[0].result(timeout=1)
    with self.assertRaisesRegex(RelayError, "is closed"):
      s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])

  def test_peer_that_stopped_reading(self):
    """A peer that accepts the streams but never drains them (hung): a
    writer gives up after MPMD_RELAY_TIMEOUT without progress (SO_SNDTIMEO;
    here 1 s, and 64 KiB socket buffers so that a 4 MiB send cannot hide in
    them), and the next send says why."""
    self.hung_peer(1)
    c = self.context(0, timeout_s=1, sockbuf_mb=1 / 16)
    aval = jax.ShapeDtypeStruct((64, 128, 128), jnp.float32)
    with mock.patch.object(transfer, "relay_routes", lambda *_: ({1: [0, 1, 2, 3]}, {})):
      s = transfer.RelayChannel(aval, self.src, self.dst, c, 3, name="s0->s1")
    x = jax.device_put(np.zeros(aval.shape, np.float32), self.src)
    s.send_flat([x])
    deadline = time.monotonic() + 30
    while c._send_err is None and time.monotonic() < deadline:
      time.sleep(0.05)
    with self.assertRaisesRegex(
        RelayError, r"relay to process 1: no progress in 1 s \(MPMD_RELAY_TIMEOUT\): " "the peer stopped reading"
    ):
      s.send_flat([x])

  def test_stray_connection_is_refused(self):
    """A connection that does not open with this context's HELLO is
    dropped with a warning; the relay carries on."""
    s, r = self.pair()
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      host, port = r.ctx.address.rsplit(":", 1)
      with socket.create_connection((host, int(port))) as c:
        c.sendall(b"GET / HTTP/1.0\r\n\r\n" + bytes(64))
      deadline = time.monotonic() + 10
      while not any("refused a connection" in str(x.message) for x in w) and time.monotonic() < deadline:
        time.sleep(0.01)
    self.assertTrue(any("refused a connection" in str(x.message) for x in w), [str(x.message) for x in w])
    want = value(AVAL.shape, "float32", 1)
    s.send_flat([jax.device_put(want, self.src)])
    np.testing.assert_array_equal(np.asarray(transfer.resolve(r.send_flat([None])[0])), want)

  def test_a_stream_needs_the_token(self):
    """A connection that opens like a stream of this context -- its magic, a
    HELLO, its key's CRC, which anyone can compute -- but without the random
    token the context published with its address is refused, and so is one
    that sends nothing (here for 0.5 s); the relay carries on.  The listener
    is on the published address only."""
    s, r = self.pair()
    host, port = r.ctx.address.rsplit(":", 1)
    self.assertEqual(self.kv.d[f"{r.ctx.key}/addr/1"], f"{r.ctx.address} {r.ctx._token.hex()}")
    self.assertEqual(r.ctx._lsock.getsockname(), (host, int(port)))
    want = ("a HELLO as process 0 without this context's token", "no HELLO: TimeoutError")
    with warnings.catch_warnings(record=True) as w, mock.patch.object(transfer, "_RELAY_HELLO_S", 0.5):
      warnings.simplefilter("always")
      forged = socket.create_connection((host, int(port)))
      forged.sendall(
          transfer._RELAY_HDR.pack(transfer._RELAY_MAGIC, transfer._HELLO, 0, 0, 0, 0, 0, 16, 1, r.ctx._key_sig)
          + bytes(16)
      )
      mute = socket.create_connection((host, int(port)))
      deadline = time.monotonic() + 10
      seen = lambda: [x for x in want if any(x in str(m.message) for m in w)]
      while seen() != list(want) and time.monotonic() < deadline:
        time.sleep(0.01)
    self.assertEqual(seen(), list(want), [str(m.message) for m in w])
    for c in (forged, mute):  # dropped by the relay
      c.settimeout(10)
      self.assertEqual(c.recv(1), b"")
      c.close()
    want = value(AVAL.shape, "float32", 13)
    s.send_flat([jax.device_put(want, self.src)])
    np.testing.assert_array_equal(np.asarray(transfer.resolve(r.send_flat([None])[0])), want)

  def test_listener_survives_accept_errors(self):
    """A failed accept() (EMFILE here, once) is no reason to stop listening:
    a peer connecting after it is still read.  (The listener used to end at
    the first error, its peers' streams then never read.)"""

    class Flaky(socket.socket):
      fails = [1]

      def accept(self):
        if self.fails[0]:
          self.fails[0] -= 1
          raise OSError(errno.EMFILE, "Too many open files")
        return super().accept()

    real = transfer._relay_socket

    def listener(sockbuf):
      s = real(sockbuf)
      return Flaky(s.family, s.type, s.proto, fileno=s.detach())

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      with mock.patch.object(transfer, "_relay_socket", listener):
        c1 = self.context(1)
      deadline = time.monotonic() + 10
      while not any("accept failed" in str(x.message) for x in w) and time.monotonic() < deadline:
        time.sleep(0.01)
    self.assertTrue(
        any(
            re.search(r"accept failed \(OSError: \[Errno 24\] Too many open files\); retrying", str(x.message)) for x in w
        ),
        [str(x.message) for x in w],
    )
    s, r = self.pair(c1=c1)
    want = value(AVAL.shape, "float32", 2)
    s.send_flat([jax.device_put(want, self.src)])
    np.testing.assert_array_equal(np.asarray(r.send_flat([None])[0].result(timeout=20)), want)

  def hand_written(self, pos0, seed=14):
    """Receive seq 0 of a boundary like `pair`'s (on contexts of their own)
    from a stream written by hand as process 0's: shards 1-3 in one chunk
    each, shard 0 as the byte ranges `pos0` -> (its future, the array sent)."""
    key = f"test/{self.id()}/{pos0}"
    _, r = self.pair(kw0=dict(key=key), kw1=dict(key=key))
    f = r.send_flat([None])[0]
    total, sig, _ = r.ctx._channels[r.idx]._recv_spec[(0, 0)]
    want = value(AVAL.shape, "float32", seed)
    msgs = []
    for pos in range(4):  # shard k of ACT is row k
      b = want[pos : pos + 1].tobytes()
      for lo, hi in pos0 if pos == 0 else ((0, total),):
        msgs.append(
            transfer._RELAY_HDR.pack(transfer._RELAY_MAGIC, transfer._DATA, r.idx, 0, 0, pos, lo, hi - lo, total, sig)
            + b[lo:hi]
        )
    s = self.raw_stream(r.ctx)
    s.sendall(b"".join(msgs))
    try:
      s.shutdown(socket.SHUT_WR)  # the stream ends after them, as a sender's does
    except OSError as e:
      # a receiver that refuses a chunk may close the stream before this: ENOTCONN (seen once in 5 runs)
      if e.errno != errno.ENOTCONN:
        raise
    return f, want

  def test_a_shard_is_every_byte_once(self):
    """A shard is complete when each of its bytes came, once, in whatever
    order: a chunk repeated, or two that overlap, fail the receive -- also
    where the bytes add up to the shard's size, as here (the shard used to
    be placed then, with the bytes that never came left as they were).  The
    relay's own sender never does either: streams written by hand."""
    f, want = self.hand_written(((256, 512), (0, 256)))
    np.testing.assert_array_equal(np.asarray(f.result(timeout=10)), want)
    for pos0, what in ((((0, 256), (0, 256)), "repeated"), (((0, 256), (128, 384)), "overlapping")):
      with self.subTest(what):
        f, _ = self.hand_written(pos0)
        with self.assertRaisesRegex(
            RelayError,
            rf"leaf 0 position 0 from process 0: bytes \[{pos0[1][0]}, "
            rf"{pos0[1][1]}\) of a 512-byte shard overlap bytes already in: duplicate or "
            "overlong data",
        ):
          f.result(timeout=10)

  def test_close_fails_waiting_receives(self):
    """A receive claimed but not complete when its context closes fails at
    once -- nothing can complete it once the streams are gone -- and the
    warning names it.  (It used to wait out MPMD_RELAY_TIMEOUT.)"""
    _, r = self.pair()
    f = r.send_flat([None])[0]
    with testing.assert_warns_regex(
        self,
        RuntimeWarning,
        r"1 claimed receive\(s\) not complete \(failed now\), e\.g\. " r"s0->s1 \(relay channel #3\) seq 0",
    ):
      r.ctx.close()
    t0 = time.monotonic()
    with self.assertRaisesRegex(
        RelayError, r"seq 0: relay context \S+ \(process 1\) is closed and the receive " "was not complete"
    ):
      f.result(timeout=30)
    self.assertLess(time.monotonic() - t0, 5)

  def test_streams_that_do_not_all_open(self):
    """A sender whose third stream of four cannot connect says so on the two
    that did (ABORT) before they end: the peer fails the receive at once,
    with the reason, instead of waiting for streams that never come (it
    counts a sender gone once all the streams its HELLO announced have
    ended); a receive claimed later fails at once too, and the sender's next
    send raises."""
    s, r = self.pair(kw0=dict(streams=4))
    real, calls = transfer._relay_connect, []

    def connect(*a):
      calls.append(a)
      if len(calls) == 3:
        raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused (by the test)")
      return real(*a)

    f = r.send_flat([None])[0]
    x = jax.device_put(np.ones(AVAL.shape, np.float32), self.src)
    with mock.patch.object(transfer, "_relay_connect", connect):
      s.send_flat([x])
      t0 = time.monotonic()
      with self.assertRaisesRegex(
          RelayError,
          r"seq 0: process 0 gave up on its relay streams to this process "
          r"with 4 shard\(s\) of it unsent: relay to process 1: connecting failed: "
          "ConnectionRefusedError",
      ):
        f.result(timeout=30)
      self.assertLess(time.monotonic() - t0, 20)
    self.assertEqual(len(calls), 3)
    with self.assertRaisesRegex(
        RelayError,
        r"seq 1: process\(es\) \[0\] closed their relay streams with shards "
        "of it unsent; process 0: relay to process 1: connecting failed",
    ):
      r.send_flat([None])[0].result(timeout=1)
    with self.assertRaisesRegex(RelayError, "relay to process 1: connecting failed"):
      s.send_flat([x])

  def test_late_shards_of_a_failed_receive(self):
    """A receive failed for one sender's sake -- process 0 closed with its
    shards of it unsent -- while its other sender's shards still come: those
    are drained.  (They used to be taken for a receive completed twice,
    which failed everything after it with a misleading cause.)"""
    s0, s2, r = self.two_senders()
    want = value(AVAL.shape, "float32", 6)
    x = jax.device_put(want, self.src)
    f = r.send_flat([None])[0]  # one whole round first: every stream is open
    s0.send_flat([x])
    s2.send_flat([x])
    np.testing.assert_array_equal(np.asarray(f.result(timeout=30)), want)
    f = r.send_flat([None])[0]
    s0.ctx.close()
    with self.assertRaisesRegex(
        RelayError, r"seq 1: process 0 closed its relay streams with 2 shard\(s\) of it " "unsent"
    ):
      f.result(timeout=30)
    s2.send_flat([x])  # process 2's half of seq 1, late
    self.assertTrue(s2.ctx._peers[1].flush(10))
    deadline = time.monotonic() + 10
    while r.ctx.stats()["received"] < 1.5 * want.nbytes and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertEqual(r.ctx.stats()["received"], 1.5 * want.nbytes)
    self.assertIsNone(r.ctx._fatal)
    with self.assertRaisesRegex(RelayError, r"seq 2: process\(es\) \[0\] closed their relay streams"):
      r.send_flat([None])[0].result(timeout=1)

  def test_close_while_a_stream_is_accepted(self):
    """close() while the listener is starting the reader of a stream it just
    accepted: the reader is started before close() can find it.  (It used to
    be registered first and started after the lock was let go, so close()
    could try to join a thread not yet started -- it raised, and left the
    listener running.)"""
    s, r = self.pair()
    real_start, starting = threading.Thread.start, threading.Event()

    def slow_start(t):
      if t.name == "relay-read":
        starting.set()
        time.sleep(0.5)
      return real_start(t)

    with mock.patch.object(threading.Thread, "start", slow_start), warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)  # data nobody claimed, or none at all
      s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])
      self.assertTrue(starting.wait(10))
      sender = threading.Thread(target=s.ctx.close)  # its streams end: r's close is quick
      sender.start()
      r.ctx.close()
      sender.join()
    self.assertEqual(relay_threads(), [])

  def test_close_while_a_writer_starts(self):
    """close() while a peer's stager is starting the writers of the streams
    it just opened: close() warns and never raises.  (A writer used to be
    listed before it was started, so close() could try to join a thread not
    yet started -- it raised, and left the listener running.)"""
    s, r = self.pair()
    real_start, starting = threading.Thread.start, threading.Event()

    def slow_start(t):
      if t.name.startswith("relay-write-"):
        starting.set()
        time.sleep(0.5)
      return real_start(t)

    with mock.patch.object(threading.Thread, "start", slow_start):
      s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])
      self.assertTrue(starting.wait(10))
      with testing.assert_warns_regex(self, RuntimeWarning, r"sends to process\(es\) \[1\] not all written"):
        s.ctx.close(timeout_s=0.1)
    wait_for_threads(self, "relay-stage-p1", "relay-write-p1-")
    self.assertFalse(s.ctx._listener.is_alive())

  def test_close_shuts_the_streams_of_a_peer_that_stopped_reading(self):
    """A peer that accepted the streams but stopped reading: close() gives up
    on it at its deadline and shuts its streams, so that the stager and the
    writers end at once and the peer sees the streams end -- not after
    MPMD_RELAY_TIMEOUT (SO_SNDTIMEO), here 60 s, with their buffers held."""
    _, held = self.hung_peer(1)
    c = self.context(0, sockbuf_mb=1 / 16)
    aval = jax.ShapeDtypeStruct((64, 128, 128), jnp.float32)
    with mock.patch.object(transfer, "relay_routes", lambda *_: ({1: [0, 1, 2, 3]}, {})):
      s = transfer.RelayChannel(aval, self.src, self.dst, c, 3, name="s0->s1")
    s.send_flat([jax.device_put(np.zeros(aval.shape, np.float32), self.src)])
    deadline = time.monotonic() + 30
    while len(held) < 2 and time.monotonic() < deadline:
      time.sleep(0.05)
    t0 = time.monotonic()
    with testing.assert_warns_regex(self, RuntimeWarning, r"sends to process\(es\) \[1\] not all written"):
      c.close(timeout_s=1)
    self.assertLess(time.monotonic() - t0, 10)
    wait_for_threads(self, "relay-stage-p1", "relay-write-p1-")
    for conn, _ in held:  # the peer sees both streams end
      conn.settimeout(10)
      try:
        while conn.recv(1 << 20):
          pass
      except ConnectionResetError:
        pass

  def test_close(self):
    """close() is idempotent, warns about data nobody claimed, stops every
    thread and the listener; afterwards a send is an error, and so is a
    receive -- but one whose data was in."""
    s, r = self.pair()
    s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])
    deadline = time.monotonic() + 10
    while not r.ctx.stats()["unclaimed"] and time.monotonic() < deadline:
      time.sleep(0.01)
    s.ctx.close()
    with testing.assert_warns_regex(
        self,
        RuntimeWarning,
        r"1 receive\(s\) arrived that no send_flat claimed, e\.g\. " r"s0->s1 \(relay channel #3\) seq 0",
    ):
      r.ctx.close()
    r.ctx.close()
    self.assertEqual(relay_threads(), [])
    host, port = r.ctx.address.rsplit(":", 1)
    with self.assertRaises(OSError):
      socket.create_connection((host, int(port)), timeout=5).close()
    with self.assertRaisesRegex(RelayError, "is closed"):
      s.send_flat([jax.device_put(np.ones(AVAL.shape, np.float32), self.src)])
    np.testing.assert_array_equal(np.asarray(r.send_flat([None])[0].result(timeout=1)), 1)  # it was in
    with self.assertRaisesRegex(RelayError, r"seq 1: relay context .* is closed"):
      r.send_flat([None])[0].result(timeout=1)


class BuildTest(Loopback):

  def test_build_rejections(self):
    """Identical stage meshes and specs, so that shard k is the same slice
    on both sides; no extended dtypes; a process on one side only; one
    channel per id."""
    c = self.context(0)
    renamed = mesh(self.devs[4:], axes=("data", "expert"))
    for dst, why in (
        (NamedSharding(mesh(self.devs[4:], (4, 1)), ACT), "shape"),
        (NamedSharding(renamed, P(("data", "expert"), None, None)), "axis names"),
    ):
      with self.subTest(why), self.assertRaisesRegex(ShardingMismatchError, "the stage meshes differ"):
        transfer.RelayChannel(AVAL, self.src, dst, c, 0)
    with self.assertRaisesRegex(ShardingMismatchError, "the same spec on both"):
      transfer.RelayChannel(AVAL, self.src, NamedSharding(self.b, P("fsdp", None, None)), c, 0)
    key = jax.ShapeDtypeStruct((4,), jax.random.key(0).dtype)
    with self.assertRaisesRegex(ShardingMismatchError, "extended dtype"):
      transfer.RelayChannel(key, NamedSharding(self.a, P()), NamedSharding(self.b, P()), c, 0)
    real = transfer._global_index_values
    skew = lambda sh, shape: real(sh, shape)[::-1] if sh.mesh == self.b else real(sh, shape)
    with (
        mock.patch.object(transfer, "_global_index_values", skew),
        self.assertRaisesRegex(ShardingMismatchError, r"s0->s1\[leaf 0\]: per-device shard indices differ"),
    ):
      transfer.RelayChannel(AVAL, self.src, self.dst, c, 0, name="s0->s1")
    # in one process every device is process 0's: both sides
    with self.assertRaisesRegex(ValueError, "s0->s1: process 0 holds devices of both stages"):
      transfer.RelayChannel(AVAL, self.src, self.dst, c, 0, name="s0->s1")
    with self.assertRaisesRegex(ValueError, "the two stages share devices"):
      transfer.RelayChannel(AVAL, self.src, NamedSharding(mesh(self.devs[2:6]), ACT), c, 0)
    with mock.patch.object(transfer, "relay_routes", lambda *_: ({1: [0, 1, 2, 3]}, {})):
      transfer.RelayChannel(AVAL, self.src, self.dst, c, 9, name="first")
      with self.assertRaisesRegex(ValueError, "relay channel id 9 is taken by first"):
        transfer.RelayChannel(AVAL, self.src, self.dst, c, 9)
      with self.assertRaisesRegex(ValueError, "not a uint32"):
        transfer.RelayChannel(AVAL, self.src, self.dst, c, 2**32)

  def test_send_rejections(self):
    """The source side must hand over the declared array -- not None, not
    another dtype or sharding -- and takes no number when it does not."""
    s, _ = self.pair()
    x = np.zeros(AVAL.shape, np.float32)
    for bad, why in (
        (None, "None"),
        (jax.device_put(x.astype(jnp.bfloat16), self.src), "dtype"),
        (jax.device_put(x, NamedSharding(self.a, P(None, AXES, None))), "sharding"),
    ):
      with self.subTest(why), self.assertRaisesRegex(ShardingMismatchError, "must pass the declared float32"):
        s.send_flat([bad])
    self.assertEqual((s.seq, s.ctx._peers), (0, {}))

  def test_settings(self):
    """A setting means the same given as an argument or in the environment,
    0 included: an argument wins, an unset or blank variable leaves the
    default, and a value the relay cannot run with -- no streams, chunks of
    no bytes, no time to wait -- is refused before anything is bound.  (0
    used to mean "unset" as an argument but be taken as given from the
    environment: MPMD_RELAY_CHUNK_MB=0 sent 1-byte chunks, and
    MPMD_RELAY_TIMEOUT=0 timed every wait out at once.)"""
    env = dict(MPMD_RELAY_STREAMS="3", MPMD_RELAY_CHUNK_MB="2", MPMD_RELAY_TIMEOUT=" ", MPMD_RELAY_SOCKBUF_MB="")
    with mock.patch.dict(os.environ, env):
      c = self.context(0, streams=None, timeout_s=None)
      self.assertEqual((c.streams, c.chunk, c.timeout_s, c.sockbuf), (3, 2 << 20, 1800, 0))
      c = self.context(1, streams=1, chunk_mb=0.5, timeout_s=7, sockbuf_mb=0)
      self.assertEqual((c.streams, c.chunk, c.timeout_s, c.sockbuf), (1, 1 << 19, 7, 0))
    refused = (
        (dict(streams=0), {}, r"0 streams per peer \(MPMD_RELAY_STREAMS\): want at least 1"),
        (dict(streams=None), dict(MPMD_RELAY_STREAMS="0"), "0 streams per peer"),
        (dict(chunk_mb=0), {}, r"0 MiB chunks \(MPMD_RELAY_CHUNK_MB\): want at least a byte"),
        ({}, dict(MPMD_RELAY_CHUNK_MB="0"), "0 MiB chunks"),
        (dict(timeout_s=0), {}, r"a 0 s timeout \(MPMD_RELAY_TIMEOUT\): want more than 0"),
        (dict(timeout_s=None), dict(MPMD_RELAY_TIMEOUT="0"), "a 0 s timeout"),
        (dict(sockbuf_mb=-1), {}, r"-1 MiB socket buffers \(MPMD_RELAY_SOCKBUF_MB\): want 0"),
    )
    for kw, env, msg in refused:
      with (
          self.subTest(kw=kw, env=env),
          mock.patch.dict(os.environ, env),
          self.assertRaisesRegex(ValueError, "^RelayContext: " + msg),
      ):
        self.context(2, **kw)
    self.assertEqual(relay_threads(), [n for c in self.ctxs for n in ["relay-listen"]])

  def test_a_failed_publish_takes_no_number(self):
    """A context whose address the key-value store refused once: its
    listener is shut, and its number is not used up -- the next context
    takes it, and so meets the other processes' n-th context, where theirs
    succeeded the first time.  (The number used to be taken first: the
    process stayed a context ahead of its peers, each side waiting under a
    key the other never used, and the listener ran on with nobody to close
    it.)"""

    class Flaky(FakeKV):
      fails = 1

      def key_value_set(self, k, v, allow_overwrite=False):
        if self.fails:
          self.fails -= 1
          raise RuntimeError("the coordination service is busy (by the test)")
        return super().key_value_set(k, v, allow_overwrite)

    kv = Flaky()
    with mock.patch.object(transfer, "_relay_contexts", 7):
      with self.assertRaisesRegex(RuntimeError, "busy"):
        transfer.RelayContext(kv=kv, pid=0, ip="127.0.0.1")
      self.assertEqual(relay_threads(), [])
      for n in (7, 8):
        self.ctxs.append(transfer.RelayContext(kv=kv, pid=0, ip="127.0.0.1"))
        self.assertEqual(self.ctxs[-1].key, f"mpmd/relay/{n}")
      self.ctxs.append(transfer.RelayContext(kv=kv, pid=0, key="mine", ip="127.0.0.1"))  # its own key: no number
      self.assertEqual(transfer._relay_contexts, 9)
    self.assertEqual(sorted(kv.d), ["mine/addr/0", "mpmd/relay/7/addr/0", "mpmd/relay/8/addr/0"])

  def test_context_needs_a_key_value_store(self):
    """Without jax.distributed (and no `kv`) there is nowhere to publish the
    address."""
    with self.assertRaisesRegex(RuntimeError, "jax.distributed is not initialized"):
      transfer.RelayContext(ip="127.0.0.1")


class PipelineStorageTest(Loopback):
  """What ds_ep_pipeline does with a relay receive: files it where the
  consuming op pops it, and only that pop waits for the bytes."""

  def test_pop_resolves(self):
    """The step loop's act / cotangents storage under multi-controller JAX
    (_RelayBox) hands the consumer the array a RelayFuture stands for
    (anything else as it was filed), and the receive's error if it failed.
    One controller's storage (_Box) is a plain dict: no relay there."""
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp

    self.assertIs(dsp._Box.pop, dict.pop)
    s, r = self.pair()
    box = dsp._RelayBox()
    want = value(AVAL.shape, "float32", 11)
    box[(1, 0)] = r.send_flat([None])[0]  # filed before its data is sent
    box[(1, 1)] = "not a future"
    s.send_flat([jax.device_put(want, self.src)])
    np.testing.assert_array_equal(np.asarray(box.pop((1, 0))), want)
    self.assertEqual(box.pop((1, 1)), "not a future")
    box = dsp._RelayBox()
    box["late"] = r.send_flat([None])[0]
    s.ctx.close()  # its data never comes
    with self.assertRaisesRegex(RelayError, "closed its relay streams"):
      box.pop("late")

  def test_needs_distributed(self):
    """Like 'xfer', the relay moves data between processes: one process
    refuses it when the pipeline is built."""
    from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp

    args = mh.TINY + "--stages 2 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 2 --transport relay".split()
    with self.assertRaisesRegex(ValueError, "transport 'relay' .* needs one process per host"):
      dsp.build(dsp.make_parser().parse_args(args))


class RoutesTest(unittest.TestCase):
  """RelayChannel's routes: shard k from the host of source position k to
  the host of destination position k, any pairing of hosts."""

  def test_routes(self):
    cases = (
        # host-aligned: positions 0-1 on host 0 go to host 2, 2-3 on host 1 to host 3; host 4 idle
        (
            ([0, 0, 1, 1], [2, 2, 3, 3]),
            {0: ({2: [0, 1]}, {}), 1: ({3: [2, 3]}, {}), 2: ({}, {0: [0, 1]}), 3: ({}, {1: [2, 3]}), 4: ({}, {})},
        ),
        # one destination host fed by two source hosts, and the other way round (XferChannel refuses both)
        (([0, 0, 1, 1], [2, 2, 2, 2]), {0: ({2: [0, 1]}, {}), 2: ({}, {0: [0, 1], 1: [2, 3]})}),
        (([0, 0, 0, 0], [1, 1, 2, 2]), {0: ({1: [0, 1], 2: [2, 3]}, {}), 2: ({}, {0: [2, 3]})}),
        # interleaved positions
        (([0, 1, 0, 1], [2, 2, 3, 3]), {1: ({2: [1], 3: [3]}, {}), 3: ({}, {0: [2], 1: [3]})}),
    )
    for (ha, hb), want in cases:
      a, b = fake_mesh(ha), fake_mesh(hb, first_id=4)
      for pid, w in want.items():
        self.assertEqual(transfer.relay_routes(a, b, pid), w, (ha, hb, pid))
    with self.assertRaisesRegex(ValueError, "process 1 holds devices of both stages"):
      transfer.relay_routes(fake_mesh([0, 0, 1, 1]), fake_mesh([1, 1, 2, 2], 4), 1)
    a = fake_mesh([0, 0, 1, 1])
    with self.assertRaisesRegex(ValueError, "the two stages share devices"):
      transfer.relay_routes(a, mesh(list(a.devices.flat)[2:] + [FakeDevice(8, 2), FakeDevice(9, 2)]), 2)


# --------------------------------------------------------- two processes
def relay_check(bench_mib=64, bench_sends=16):
  """One process of `test_two_processes` (two processes of two CPU devices,
  jax.distributed over localhost): RelayChannels both ways between process
  0's stage and process 1's, carrying LEAVES (sharded over the stage,
  replicated, a scalar; bf16, f32, int32, bool, f16), 120 sends each way
  interleaved, each process sending on one channel while it receives on the
  other, up to 4 receives outstanding; every value checked on its
  destination sharding.  Then a `bench_mib` boundary `bench_sends` times
  0 -> 1, which process 1 times from a common barrier to its last array:
  a throughput smoke number (CPU device copies included)."""
  from jax._src import distributed
  from maxtext.experimental.mpmd.jax import ds_ep_pipeline

  ds_ep_pipeline.init_distributed()
  pid, client = jax.process_index(), distributed.global_state.client
  own = [[d for d in jax.devices() if d.process_index == q] for q in range(2)]
  assert [len(x) for x in own] == [2, 2], own
  stages = [Mesh(np.array(own[q]), ("x",)) for q in range(2)]
  ctx = transfer.RelayContext()
  avals = [jax.ShapeDtypeStruct(shape, jnp.dtype(dt)) for shape, dt, _ in LEAVES]
  chans = []
  for a, b in ((0, 1), (1, 0)):
    src = [NamedSharding(stages[a], spec) for _, _, spec in LEAVES]
    dst = [NamedSharding(stages[b], spec) for _, _, spec in LEAVES]
    chans.append((a, b, transfer.RelayChannel(avals, src, dst, ctx, a, name=f"s{a}->s{b}"), src, dst))
  pending, n_checked = collections.deque(), 0

  def check(what, futs, vals, dst):
    for i, (f, v, d) in enumerate(zip(futs, vals, dst)):
      y = transfer.resolve(f)
      assert y.sharding == d, (what, i, y.sharding)
      np.testing.assert_array_equal(np.asarray(y).astype(np.float32), v.astype(np.float32), err_msg=f"{what} leaf {i}")

  rounds = 120
  for r in range(rounds):
    for a, _, ch, src, dst in chans:
      vals = [value(shape, dt, a, r, i) for i, (shape, dt, _) in enumerate(LEAVES)]
      if pid == a:
        assert ch.send_flat([jax.device_put(v, s) for v, s in zip(vals, src)]) == [None] * len(LEAVES)
      else:
        pending.append((f"{ch.name} round {r}", ch.send_flat([None] * len(LEAVES)), vals, dst))
    while len(pending) > 4:
      check(*pending.popleft())
      n_checked += 1
  while pending:
    check(*pending.popleft())
    n_checked += 1
  assert n_checked == rounds, n_checked
  print(f"relay: {rounds} sends each way, {n_checked} receives of {len(LEAVES)} leaves exact", flush=True)

  rows = bench_mib * 2**20 // (4 * 4096)
  big = jax.ShapeDtypeStruct((rows, 4096), jnp.float32)
  bch = transfer.RelayChannel(
      big, NamedSharding(stages[0], P("x")), NamedSharding(stages[1], P("x")), ctx, 9, name="bench s0->s1"
  )
  want = np.arange(rows * 4096, dtype=np.float32).reshape(rows, 4096)
  x = jax.device_put(want, NamedSharding(stages[0], P("x"))) if pid == 0 else None
  if pid == 1:  # the first one, not timed
    transfer.resolve(bch.send_flat([None])[0])
  else:
    bch.send_flat([x])
  client.wait_at_barrier("relay_bench_start", 300_000)
  t0 = time.perf_counter()
  if pid == 0:
    for _ in range(bench_sends):
      bch.send_flat([x])
  else:
    ys = [bch.send_flat([None])[0] for _ in range(bench_sends)]
    ys = jax.block_until_ready([transfer.resolve(y) for y in ys])
    dt = time.perf_counter() - t0
    np.testing.assert_array_equal(np.asarray(ys[-1]), want)
    print(
        f"relay throughput s0->s1: {bench_sends * rows * 4096 * 4 / 2**20 / dt:.0f} MiB/s "
        f"({bench_sends} x {bench_mib} MiB in {dt:.3f} s; {ctx.streams} streams, "
        f"{ctx.chunk / 2**20:g} MiB chunks)",
        flush=True,
    )
  client.wait_at_barrier("relay_check_done", 300_000)
  ctx.close()
  st = ctx.stats()
  assert not st["unclaimed"] and not st["waiting"], st
  print(f'relay ok: sent {st["sent"]} B, received {st["received"]} B', flush=True)


class TwoProcessTest(unittest.TestCase):

  def setUp(self):
    testing.require_multihost()  # starts two JAX processes

  def test_two_processes(self):
    """`relay_check` on two processes of two devices: no model, seconds."""
    outs = mh.run_multi(
        None,
        procs=2,
        devs=2,
        timeout=300,
        cmd=[sys.executable, "-c", "from tests.unit.experimental.mpmd.jax import relay_test as m; m.relay_check()"],
    )
    mh.MultiHostTest._check_ok(self, outs)
    for i, (_, o) in enumerate(outs):
      self.assertIn("relay ok", o, f"process {i}:\n{o[-3000:]}")
      self.assertIn("120 receives of 5 leaves exact", o, f"process {i}:\n{o[-3000:]}")
    rate = re.search(r"relay throughput s0->s1: (\d+) MiB/s.*", outs[1][1])
    self.assertTrue(rate, outs[1][1][-3000:])
    print(f"\n  {rate[0]}", file=sys.stderr)


class PipelineTest(unittest.TestCase):
  """`ds_ep_pipeline --distributed --transport relay` against the same
  pipeline in one process (multihost_test's checks)."""

  _check_ok = mh.MultiHostTest._check_ok
  _records = mh.MultiHostTest._records
  _compare = mh.MultiHostTest._compare
  _check_one_session = mh.MultiHostTest._check_one_session

  def setUp(self):
    super().setUp()
    # every test here starts several JAX processes: skipped up front, not after its one-process reference run
    testing.require_multihost()

  @pytest.mark.scheduled_only
  def test_relay_transport(self):
    """PP=2 x FSDP=2 on 2 processes of 2 devices: every boundary copy (and
    gradient-norm hop) is two shards over the relay, both ways (circular
    V=2: s0 -> s1 -> s0 -> s1).  Within a stage the FSDP collectives are
    collective-permutes in one process, so the first step is the
    single-process one bit for bit (`_compare`).  Each stage's leader logs
    the relay's settings, and each process where its time went."""
    args = (
        mh.TINY
        + ("--stages 2 --vv 2 --fsdp 2 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search " "--transport relay").split()
    )
    outs, _ = self._compare(args, procs=2, devs=2)
    for i, (_, o) in enumerate(outs):
      self.assertRegex(
          o,
          rf"\[s{i} p{i}\] relay mpmd/relay/0 p{i} at 127\.0\.0\.1:\d+: 8 streams per peer, "
          r"4 MiB chunks, .*; this host: thp: ",
      )
      self.assertRegex(
          o,
          rf"\[s{i} p{i}\] relay timing, warmup steps: sent \d+ shards \(on the host p50 .*; "
          r"received \d+ \(wire p50 .*; consumers waited p50 .* s in all \(.*sumsq s",
      )

  @pytest.mark.scheduled_only
  def test_relay_stage_spans_two_processes(self):
    """PP=2 x FSDP=2 on 4 processes of one device: each stage spans two
    hosts, as in production (a stage per multi-host slice), so every
    boundary is two host pairs and each destination process assembles only
    its own shard.  The FSDP collectives cross processes too, over gloo,
    which does not even repeat itself: not `exact`."""
    args = mh.TINY + "--stages 2 --vv 2 --fsdp 2 --ep 1 --gbs 8 --mb 4 --layers 4 --transport relay".split()
    self._compare(args, procs=4, devs=1, collectives="gloo", exact=False)

  @pytest.mark.scheduled_only
  def test_relay_four_stages(self):
    """PP=4, one device per process, under --layout physical_origin_last
    (stage s on process s + 1 mod 4): 8 activation channels and 12
    gradient-norm hops, every hop issued before any update waits on one."""
    args = (
        mh.TINY
        + (
            "--stages 4 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 8 --schedule search "
            "--layout physical_origin_last --transport relay"
        ).split()
    )
    self._compare(args, procs=4, devs=1)

  @pytest.mark.scheduled_only
  def test_relay_interleaved_pairs(self):
    """Interleaved 1F1B's F&B pairs over the relay: a remote pair issues its F's
    copy, then its B's (`_do_remote`), each a receive of its own."""
    args = (
        mh.TINY
        + (
            "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 "
            "--schedule interleaved --pair-stages all --transport relay"
        ).split()
    )
    self._compare(args, procs=2, devs=1)

  @pytest.mark.scheduled_only
  def test_relay_no_overlap(self):
    """--no-overlap: a step's weights gathered up front, its gradients
    reduced after the loop and its update apart from them -- the other path
    through `step` to the gradient-norm hops."""
    args = (
        mh.TINY
        + (
            "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --schedule search "
            "--no-overlap --transport relay"
        ).split()
    )
    self._compare(args, procs=2, devs=1)

  @pytest.mark.scheduled_only
  def test_relay_v_stash_idle_process(self):
    """V placement with stash modes, PP=2, and a third process holding no
    stage: it opens no stream, and the run is the single-process one on the
    stages' 2 devices."""
    args = (
        mh.TINY
        + (
            "--stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --placement v "
            "--schedule stash --stash lite --stash-budget-gib 0.01 --transport relay"
        ).split()
    )
    outs, _ = self._compare(args, procs=3, devs=1, ref_devs=2)
    self.assertIn("[idle p2] this process holds no stage", outs[2][1])


def main():
  ap = argparse.ArgumentParser(description="the relay between two local processes: MiB/s")
  ap.add_argument("what", choices=["bench"])
  ap.add_argument("--mib", type=int, default=64, help="boundary size")
  ap.add_argument("--sends", type=int, default=16)
  ap.add_argument("--streams", type=int, default=0, help="MPMD_RELAY_STREAMS (0: its default)")
  ap.add_argument("--chunk-mb", type=float, default=0, help="MPMD_RELAY_CHUNK_MB (0: its default)")
  a = ap.parse_args()
  env = {k: str(v) for k, v in (("MPMD_RELAY_STREAMS", a.streams), ("MPMD_RELAY_CHUNK_MB", a.chunk_mb)) if v}
  code = f"from tests.unit.experimental.mpmd.jax import relay_test as m; m.relay_check({a.mib}, {a.sends})"
  with mock.patch.dict(os.environ, env):
    outs = mh.run_multi(None, procs=2, devs=2, timeout=900, cmd=[sys.executable, "-c", code])
  for i, (rc, o) in enumerate(outs):
    print(f"===== process {i}: exit {rc}")
    print("\n".join(ln for ln in o.splitlines() if ln.startswith("relay")) if rc == 0 else o[-3000:])


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "bench":
    main()
  else:
    unittest.main()
