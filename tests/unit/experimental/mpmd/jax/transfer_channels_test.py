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

"""transfer.py's channels between stages of different processes, checked in
ONE process on 4 CPU devices (or 8): PermuteChannel end to end (this process
holds both stages), what each channel refuses when it is built or handed the
wrong array, a bystander's part, and XferChannel's host pairing (`xfer_peers`)
on fake devices of several hosts.  The copies across processes themselves
need several processes: multihost_test.py.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/transfer_channels_test.py -q
"""

import dataclasses
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer
from maxtext.experimental.mpmd.jax.transfer import ShardingMismatchError

AXES = ("fsdp", "expert")
ACT = P(AXES, None, None)  # an activation's spec in the pipeline: batch over both axes
AVAL = jax.ShapeDtypeStruct((4, 16, 8), jnp.float32)


def setUpModule():
  testing.require_cpu()


def mesh(devs, shape=(2, 2), axes=AXES):
  return Mesh(np.array(devs, dtype=object).reshape(shape), axes)


@testing.needs_devices(4)
class Stages(unittest.TestCase):
  """Two stages of n devices each, fsdp x expert meshes as the pipeline's on the devices there are
  (testing.stage_axes): a on devices 0 to n-1, b on n to 2n-1 -- 2x2 meshes on 8 devices, 1x2 on 4."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    axes = testing.stage_axes()
    cls.shape = (axes["fsdp"], axes["ep"])
    cls.n = n = axes["fsdp"] * axes["ep"]
    cls.devs = jax.devices()[: 2 * n]
    cls.a, cls.b = mesh(cls.devs[:n], cls.shape), mesh(cls.devs[n:], cls.shape)
    cls.src, cls.dst = NamedSharding(cls.a, ACT), NamedSharding(cls.b, ACT)

  def meshes_that_differ(self):
    """(destination sharding, why) for b's devices in a mesh unlike a's."""
    renamed = mesh(self.devs[self.n :], self.shape, axes=("data", "expert"))
    return (
        (NamedSharding(mesh(self.devs[self.n :], (self.n, 1)), ACT), "shape"),
        (NamedSharding(renamed, P(("data", "expert"), None, None)), "axis names"),
    )

  def overlapping(self):
    """A mesh like b's on devices half of which are a's."""
    return mesh(self.devs[self.n // 2 : self.n // 2 + self.n], self.shape)


class PermuteChannelTest(Stages):
  """A boundary as a collective-permute program over both stages' devices,
  which here are all this process's: a send runs end to end."""

  def test_round_trip(self):
    """Every leaf arrives exact, on its destination sharding, send after
    send -- the destination's half of the program's input is zeros it never
    writes -- and the source arrays are left as they were."""
    avals = {"y": jax.ShapeDtypeStruct((4, 16, 8), jnp.bfloat16), "sq": jax.ShapeDtypeStruct((), jnp.float32)}
    src = {"y": self.src, "sq": NamedSharding(self.a, P())}
    dst = {"y": self.dst, "sq": NamedSharding(self.b, P())}
    ch = transfer.PermuteChannel(avals, src, dst, name="s0->s1")
    self.assertTrue(ch.owns_src and ch.owns_dst)
    self.assertEqual(dict(ch.union.shape), {"_xfer": 2, "fsdp": self.shape[0], "expert": self.shape[1]})
    self.assertEqual(
        [(a.shape, a.sharding) for a in ch.lower_args()],  # leaves in tree order: sq, y
        [((), ch.u_sh[0]), ((4, 16, 8), ch.u_sh[1])],
    )
    for i in range(3):
      host = {
          "y": (np.arange(4 * 16 * 8).reshape(4, 16, 8) * (i + 1) - 300).astype(jnp.bfloat16),
          "sq": np.float32(i - 0.25),
      }
      x = jax.device_put(host, src)
      out = ch.send(x)
      for k in host:
        self.assertEqual(out[k].sharding, dst[k], k)
        np.testing.assert_array_equal(np.asarray(out[k]), host[k], k)
        np.testing.assert_array_equal(np.asarray(x[k]), host[k], k)

  def test_destination_zeros_are_shared(self):
    """Every channel into a stage passes the same zero buffers (one per
    device, shard shape and dtype), whichever stage it comes from; a channel
    into another stage has its own."""
    w = self.n // 2  # three stages of w devices
    s0, s1, s2 = (mesh(self.devs[k : k + w], (1, w)) for k in (0, w, 2 * w))
    into = lambda a, b: transfer.PermuteChannel(AVAL, NamedSharding(a, ACT), NamedSharding(b, ACT))._dummy()[0]
    z02, z12, z20 = into(s0, s2), into(s1, s2), into(s2, s0)
    self.assertEqual([z.devices() for z in z02], [{d} for d in self.devs[2 * w : 3 * w]])
    self.assertTrue(all(x is y for x, y in zip(z02, z12)))
    self.assertEqual([z.devices() for z in z20], [{d} for d in self.devs[:w]])
    self.assertFalse(np.any([np.asarray(z) for z in z02]))

  def test_build_rejections(self):
    """Same-shape meshes with the same axis names on disjoint devices, and
    the same spec on both sides: shard i of one stage is device i's of the
    other."""
    for dst, why in self.meshes_that_differ():
      with self.subTest(why), self.assertRaisesRegex(ShardingMismatchError, "the stage meshes differ"):
        transfer.PermuteChannel(AVAL, self.src, dst)
    for m, why in ((self.a, "the same devices"), (self.overlapping(), "half of them")):
      with self.subTest(why), self.assertRaisesRegex(ValueError, "the two stages share devices"):
        transfer.PermuteChannel(AVAL, self.src, NamedSharding(m, ACT))
    with self.assertRaisesRegex(ShardingMismatchError, "the same spec on both"):
      transfer.PermuteChannel(AVAL, self.src, NamedSharding(self.b, P("fsdp", None, None)))

  def test_send_rejections(self):
    """The source side must hand over the declared array -- its shape, dtype
    and sharding (the program would reinterpret another layout) -- not None."""
    ch = transfer.PermuteChannel(AVAL, self.src, self.dst)
    x = np.zeros(AVAL.shape, np.float32)
    for bad, why in (
        (None, "None"),
        (jax.device_put(x.astype(jnp.bfloat16), self.src), "dtype"),
        (jax.device_put(np.zeros((8, 16, 8), np.float32), self.src), "shape"),
        (jax.device_put(x, NamedSharding(self.a, P(None, AXES, None))), "sharding"),
    ):
      with self.subTest(why), self.assertRaisesRegex(ShardingMismatchError, "must pass the declared float32"):
        ch.send_flat([bad])

  def test_bystander(self):
    """A process holding neither stage takes no part: it passes nothing and
    gets None back at once, without running the program.  A process of a
    multi-controller run (process 3 of 4); in one controller (this process,
    or a Pathways client) both sides are its own, and no process is asked."""
    with mock.patch.object(jax, "process_index", lambda: 3), mock.patch.object(jax, "process_count", lambda: 4):
      ch = transfer.PermuteChannel(AVAL, self.src, self.dst)
    self.assertFalse(ch.owns_src or ch.owns_dst)
    with mock.patch.object(ch, "prog", None):  # nothing to call
      self.assertEqual(ch.send_flat([None]), [None])
    with mock.patch.object(jax, "process_index", mock.Mock(side_effect=AssertionError("asked"))):
      one = transfer.PermuteChannel(AVAL, self.src, self.dst)
      self.assertEqual((one.multi, one.owns_src, one.owns_dst), (False, True, True))


class LoopbackXfer:
  """An XferContext's server and connections within one process: an offer
  (`await_pull`) waits under its uuid until a pull of that uuid takes it,
  each array copied to the device the puller's aval names."""

  def __init__(self):
    self.srv, self.offers = self, {}

  def await_pull(self, uuid, arrays):
    assert uuid not in self.offers, uuid
    self.offers[uuid] = list(arrays)

  def conn(self, peer):
    return self

  def pull(self, uuid, avals):
    arrays = self.offers.pop(uuid)
    assert [(a.shape, a.dtype) for a in arrays] == [(v.shape, v.dtype) for v in avals], (arrays, avals)
    return [jax.device_put(a, v.sharding) for a, v in zip(arrays, avals)]


class XferChannelTest(Stages):
  """XferChannel: what it checks when it is built (before it needs a
  transfer server, so `ctx` None) -- identical stage meshes and specs, so
  that shard k is the same slice on both sides, and a host pairing
  `xfer_peers` takes -- and the two hosts' halves of a send, given their
  roles, over a loopback server."""

  def channel(self, peers, ctx, src=None):
    """The channel as a process of the role `peers` (what xfer_peers
    returns for it) builds it."""
    with mock.patch.object(transfer, "xfer_peers", lambda *_: peers):
      return transfer.XferChannel(AVAL, src or self.src, self.dst, ctx, 5, name="s0->s1")

  def test_loopback(self):
    """The source host offers its shards in mesh-position order under the
    channel's uuid -- its index and a sequence number both hosts advance on
    every send -- and the destination host pulls that uuid and reassembles
    its stage array, send after send.  The source mesh's positions run
    against device-id order, so an offer in id order would arrive permuted."""
    a = mesh(self.devs[: self.n][::-1], self.shape)
    xfer = LoopbackXfer()
    src = self.channel((list(range(self.n)), [], None, 1), xfer, NamedSharding(a, ACT))
    dst = self.channel(([], list(self.b.devices.flat), 0, None), xfer, NamedSharding(a, ACT))
    self.assertEqual([(c.owns_src, c.owns_dst, c.peer) for c in (src, dst)], [(True, False, None), (False, True, 0)])
    for i in range(3):
      x = jax.device_put(np.arange(512, dtype=np.float32).reshape(AVAL.shape) - 1000 * i, NamedSharding(a, ACT))
      self.assertEqual(src.send_flat([x]), [None])
      self.assertEqual(list(xfer.offers), [5 << 40 | i])
      self.assertEqual([s.devices() for s in xfer.offers[5 << 40 | i]], [{d} for d in a.devices.flat])
      (y,) = dst.send_flat([None])
      self.assertEqual(y.sharding, self.dst)
      np.testing.assert_array_equal(np.asarray(y), np.asarray(x))
    self.assertEqual((src.seq, dst.seq, xfer.offers), (3, 3, {}))

  def test_equivalent_source_sharding(self):
    """The source host may pass its array on a sharding equivalent to the
    declared one -- P(AXES) for P(AXES, None, None): the same devices hold
    the same slices, and its shards are taken by device."""
    xfer = LoopbackXfer()
    src = self.channel((list(range(self.n)), [], None, 1), xfer)
    dst = self.channel(([], list(self.b.devices.flat), 0, None), xfer)
    x = jax.device_put(np.arange(512, dtype=np.float32).reshape(AVAL.shape), NamedSharding(self.a, P(AXES)))
    self.assertEqual(src.send_flat([x]), [None])
    (y,) = dst.send_flat([None])
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x))

  def test_send_rejections_and_bystander(self):
    """A source host must offer the declared array -- not None, not another
    dtype -- and offers nothing otherwise; a process holding neither stage
    returns None without touching the transfer server."""
    xfer = LoopbackXfer()
    src = self.channel((list(range(self.n)), [], None, 1), xfer)
    for bad in (None, jax.device_put(np.zeros(AVAL.shape, jnp.bfloat16), self.src)):
      with self.assertRaisesRegex(ShardingMismatchError, "must pass the declared float32"):
        src.send_flat([bad])
    self.assertEqual(xfer.offers, {})
    with mock.patch.object(jax, "process_index", lambda: 5):
      ch = transfer.XferChannel(AVAL, self.src, self.dst, None, 3)
    self.assertFalse(ch.owns_src or ch.owns_dst)
    self.assertEqual((ch.peer, ch.send_flat([None])), (None, [None]))

  def test_build_rejections(self):
    for dst, why in self.meshes_that_differ():
      with self.subTest(why), self.assertRaisesRegex(ShardingMismatchError, "the stage meshes differ"):
        transfer.XferChannel(AVAL, self.src, dst, None, 0)
    with self.assertRaisesRegex(ShardingMismatchError, "the same spec on both"):
      transfer.XferChannel(AVAL, self.src, NamedSharding(self.b, P("fsdp", None, None)), None, 0)
    # shard k must be the same slice on both sides.  NamedShardings that pass the checks above always
    # agree, so skew the destination's per-device index map by hand.
    real = transfer._global_index_values
    skew = lambda sh, shape: real(sh, shape)[::-1] if sh.mesh == self.b else real(sh, shape)
    with (
        mock.patch.object(transfer, "_global_index_values", skew),
        self.assertRaisesRegex(ShardingMismatchError, r"s0->s1\[leaf 0\]: per-device shard indices differ"),
    ):
      transfer.XferChannel(AVAL, self.src, self.dst, None, 0, name="s0->s1")
    with self.assertRaisesRegex(ValueError, "s0->s1: process 0 holds devices of both stages"):
      transfer.XferChannel(AVAL, self.src, self.dst, None, 0, name="s0->s1")
    with self.assertRaisesRegex(ValueError, "the two stages share devices"):
      transfer.XferChannel(AVAL, self.src, NamedSharding(self.overlapping(), ACT), None, 0)


@dataclasses.dataclass(frozen=True)
class FakeDevice:
  """A device of some host: all `xfer_peers` and `relay_routes` read of it
  (relay_test imports this harness too)."""

  id: int
  process_index: int


def fake_mesh(hosts, first_id=0):
  """A 2x2 mesh of fake devices, mesh position k on host hosts[k]."""
  return mesh([FakeDevice(first_id + k, h) for k, h in enumerate(hosts)])


class XferPeersTest(unittest.TestCase):
  """XferChannel's host pairing: a function of the two stage meshes and a
  process id, so ds_ep_aot checks it ahead of time for every process of a
  topology.  Each host gives its shards to one peer host and each takes
  them from one."""

  def test_host_aligned(self):
    """Mesh positions 0-1 on host 0 go to host 2, positions 2-3 on host 1 to
    host 3; host 4 holds neither stage."""
    a, b = fake_mesh([0, 0, 1, 1]), fake_mesh([2, 2, 3, 3], first_id=4)
    fb = list(b.devices.flat)
    want = {
        0: ([0, 1], [], None, 2),
        1: ([2, 3], [], None, 3),
        2: ([], fb[:2], 0, None),
        3: ([], fb[2:], 1, None),
        4: ([], [], None, None),
    }
    for pid, w in want.items():
      self.assertEqual(transfer.xfer_peers(a, b, pid), w, pid)

  def test_rejected_pairings(self):
    cases = (
        # one destination host fed by two source hosts
        (([0, 0, 1, 1], [2, 2, 2, 2]), 2, r"process 2 would receive from processes \[0, 1\]"),
        # one source host feeding two destination hosts
        (([0, 0, 0, 0], [1, 1, 2, 2]), 0, r"process 0 would send to processes \[1, 2\]"),
        # positions interleaved across the hosts: both at once
        (([0, 1, 0, 1], [2, 2, 3, 3]), 2, r"process 2 would receive from processes \[0, 1\]"),
        (([0, 1, 0, 1], [2, 2, 3, 3]), 1, r"process 1 would send to processes \[2, 3\]"),
        # a host on both sides
        (([0, 0, 1, 1], [1, 1, 2, 2]), 1, "process 1 holds devices of both stages"),
    )
    for (ha, hb), pid, msg in cases:
      with self.subTest(src=ha, dst=hb, process=pid), self.assertRaisesRegex(ValueError, msg):
        transfer.xfer_peers(fake_mesh(ha), fake_mesh(hb, first_id=4), pid)
    # the other side of the first two: a lone destination host, a source host with a lone peer
    self.assertEqual(transfer.xfer_peers(fake_mesh([0, 0, 1, 1]), fake_mesh([2, 2, 2, 2], 4), 0)[3], 2)
    self.assertEqual(transfer.xfer_peers(fake_mesh([0, 0, 0, 0]), fake_mesh([1, 1, 2, 2], 4), 2)[2], 0)
    a = fake_mesh([0, 0, 1, 1])
    shared = mesh(list(a.devices.flat)[2:] + [FakeDevice(8, 2), FakeDevice(9, 2)])
    with self.assertRaisesRegex(ValueError, "the two stages share devices"):
      transfer.xfer_peers(a, shared, 2)


class OneProcess(Stages):
  """CrossProcessChannel and XferContext in a single process."""

  def test_cross_process_channel_needs_two_processes(self):
    """jax's cross-host copy is for stages on different processes; in one
    process a boundary is a Channel's, and is_cross_process says so."""
    self.assertEqual((transfer.process_indices(self.src), transfer.process_indices(self.dst)), ({0}, {0}))
    self.assertFalse(transfer.is_cross_process(self.src, self.dst))
    with self.assertRaisesRegex(ValueError, r"s0->s1\[leaf 0\]: source and destination are on the same processes"):
      transfer.CrossProcessChannel(AVAL, self.src, self.dst, name="s0->s1")

  def test_one_xfer_context_per_process(self):
    """A process publishes its transfer server's address once: xfer_context()
    hands back the context it has, and a second XferContext is an error
    (before it starts a server)."""
    ctx = object()
    with mock.patch.object(transfer, "_XFER_CONTEXT", ctx):
      self.assertIs(transfer.xfer_context(), ctx)
      with self.assertRaisesRegex(RuntimeError, "already has an XferContext"):
        transfer.XferContext()


if __name__ == "__main__":
  unittest.main()
