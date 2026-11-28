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

"""Tests for the one layer of `xprof` that needs a real xplane: bytes -> Trace.

Separate from `xprof_test.py` on purpose.  That file builds its traces with
`Trace.__new__` against synthetic timelines, so it needs neither protobuf nor
a capture; nothing above the parse needs either.  What is left uncovered by
that arrangement is exactly this file: the line-timestamp arithmetic that puts
host and device events on comparable clocks, the line filter that keeps the
parse from materialising 43,641 counter events per device, the two ways a path
can fail to be a trace, and the messages themselves.  `xprof` builds them from
xplane.proto's field numbers and never imports tensorflow, whose xplane_pb2
they must match field for field (MessagesTest, scheduled_only: where tensorflow
is installed, importing it takes 10 s).
"""

import importlib.util
import tempfile
import unittest
import pytest

from maxtext.experimental.mpmd.jax import xprof

try:  # the messages `xprof` builds from their descriptor: protobuf, no tensorflow
  xprof._xplane_pb2()
  HAVE_PB = True
except ImportError:  # pragma: no cover
  HAVE_PB = False


@unittest.skipUnless(HAVE_PB, "needs protobuf for the xplane protobuf")
class TraceParseTest(unittest.TestCase):

  def space(self):
    """One TPU plane and one host plane, small enough to reason about."""
    pb = xprof._xplane_pb2()
    space = pb.XSpace()

    pl = space.planes.add(name="/device:TPU:0")
    pl.event_metadata[1].name = "jit__bwda(1)"
    pl.event_metadata[2].name = "while.1"
    pl.event_metadata[3].name = "fusion.2"
    pl.event_metadata[4].name = "step"
    pl.stat_metadata[1].name = "peak_teraflops_per_second"
    pl.stat_metadata[2].name = "peak_hbm_bw_gigabytes_per_second"
    pl.stats.add(metadata_id=1).double_value = 946.7
    pl.stats.add(metadata_id=2).double_value = 1637.993152512
    # The line's own timestamp is the origin every offset on it is relative to.
    mods = pl.lines.add(name="XLA Modules", timestamp_ns=1000)
    mods.events.add(metadata_id=1, offset_ps=5000, duration_ps=100000)
    # Contained op added BEFORE its container, so the sort has work to do.
    ops = pl.lines.add(name="XLA Ops", timestamp_ns=1000)
    ops.events.add(metadata_id=3, offset_ps=5000, duration_ps=40000)
    ops.events.add(metadata_id=2, offset_ps=5000, duration_ps=100000)
    steps = pl.lines.add(name="Steps", timestamp_ns=1000)
    for i in range(2):
      steps.events.add(metadata_id=4, offset_ps=i, duration_ps=1)
    counters = pl.lines.add(name="counters_0", timestamp_ns=1000)
    counters.events.add(metadata_id=4, offset_ps=0, duration_ps=0)

    hp = space.planes.add(name="/host:CPU")
    hp.event_metadata[1].name = "PjitFunction(step)"
    hp.event_metadata[2].name = "something else"
    hl = hp.lines.add(name="python3", timestamp_ns=0)
    hl.events.add(metadata_id=1, offset_ps=0, duration_ps=1000)
    other = hp.lines.add(name="tsl::thread", timestamp_ns=0)
    other.events.add(metadata_id=2, offset_ps=0, duration_ps=1000)
    return space

  def trace(self, space=None):
    with tempfile.NamedTemporaryFile(suffix=".xplane.pb") as f:
      f.write((space if space is not None else self.space()).SerializeToString())
      f.flush()
      return xprof.Trace(f.name)

  def test_an_event_starts_at_its_line_timestamp_plus_its_offset(self):
    # Offsets are per line and the lines have different origins, so dropping
    # the timestamp puts the host plane and the device planes on clocks that
    # silently disagree -- which is precisely the comparison `driver_busy`
    # makes. On the mbs-256 gpipe capture the host line starts 4,408,980 ns in
    # while every device line starts at 0, so this is the only place it shows.
    tr = self.trace()
    self.assertEqual(tr.modules(0)[0].start, 1000 * 1000 + 5000)
    self.assertEqual(tr.modules(0)[0].dur, 100000)
    self.assertEqual(tr.modules(0)[0].name, "jit__bwda(1)")

  def test_events_come_out_container_first(self):
    # `children` walks the list greedily and relies on a container arriving
    # before anything it contains; here both start at the same picosecond.
    self.assertEqual([o.base for o in self.trace().ops(0)], ["while", "fusion"])

  def test_only_the_lines_a_subcommand_reads_are_materialised(self):
    # `counters_0` alone is 43,641 zero-duration events per device, 95% of the
    # file and about a second of parse time that is then thrown away.
    tr = self.trace()
    self.assertEqual(sorted(tr.devices[0]), ["Steps", "XLA Modules", "XLA Ops"])
    self.assertEqual(tr.n_steps_line[0], 2)  # module executions, not steps

  def test_peaks_come_from_the_planes_own_stats(self):
    self.assertEqual(self.trace().peaks[0], (946.7e12, 1637.993152512e9))

  def test_the_host_plane_keeps_only_the_interpreter_lines(self):
    tr = self.trace()
    self.assertEqual(list(tr.host), ["python3"])
    self.assertEqual(tr.driver_busy(0, 10**15), 1000)

  def test_a_plane_with_no_peaks_parses_to_none_rather_than_raising(self):
    space = self.space()
    del space.planes[0].stats[:]
    self.assertEqual(self.trace(space).peaks[0], (None, None))

  def test_the_messages_are_built_once_a_process(self):
    self.assertIs(xprof._xplane_pb2(), xprof._xplane_pb2())

  def test_a_file_that_is_not_a_trace_is_a_message_not_a_traceback(self):
    # Easy to hit: the `RUN.txt` and the `*.trace.json.gz` that sit beside
    # every capture are both plausible things to name on the command line.
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
      f.write(
          b"python3 -m maxtext.experimental.mpmd.jax.ds_ep_pipeline --schedule gpipe --profile /tmp/xprof/mpmd-gpipe\n"
      )
      f.flush()
      with self.assertRaises(SystemExit) as cm:
        xprof.Trace(f.name)
    self.assertIn("not a serialized XSpace", str(cm.exception))

  def test_sparsecore_planes_are_set_aside_not_parsed_as_devices(self):
    # v6e captures of anything with a MoE carry `/device:TPU:0 SparseCore 0`
    # planes; `int('0 SparseCore 0')` used to take the whole parse down.
    space = self.space()
    sc = space.planes.add(name="/device:TPU:0 SparseCore 0")
    sc.event_metadata[1].name = "sc.gather"
    sc.lines.add(name="Sparse Core Modules", timestamp_ns=0).events.add(metadata_id=1, offset_ps=0, duration_ps=10)
    tr = self.trace(space)
    self.assertEqual(list(tr.devices), [0])
    self.assertEqual(tr.sparsecore, ["/device:TPU:0 SparseCore 0"])

  def test_a_trace_with_no_tpu_planes_is_a_message(self):
    space = self.space()
    del space.planes[0]
    with self.assertRaises(SystemExit) as cm:
      self.trace(space)
    self.assertIn("no /device:TPU:* planes", str(cm.exception))

  def test_an_ops_tf_op_comes_from_its_metadata(self):
    # The named scope is a stat of the op's *metadata*, not of the event.
    space = self.space()
    pl = space.planes[0]
    pl.stat_metadata[3].name = "tf_op"
    pl.event_metadata[3].stats.add(metadata_id=3).str_value = "jit(bwd_c0)/shard_map/fsdp_reduce_scatter/ppermute:"
    tr = self.trace(space)
    fusion = [o for o in tr.ops(0) if o.base == "fusion"][0]
    self.assertEqual(xprof.scope_of(fusion), "fsdp_reduce_scatter")
    self.assertNotIn("tf_op", [o for o in tr.ops(0) if o.base == "while"][0].stats)
    self.assertNotIn("tf_op", tr.modules(0)[0].stats)

  def test_an_ops_hlo_category_and_the_planes_megacore_flag(self):
    # XLA's op class rides on the metadata like the name stack; it is what
    # says a TPU `fusion.N` is a fused all-reduce-scatter.
    space = self.space()
    pl = space.planes[0]
    pl.stat_metadata[3].name = "hlo_category"
    pl.event_metadata[3].stats.add(metadata_id=3).str_value = "all-reduce-scatter fusion"
    pl.stat_metadata[4].name = "has_megacore"
    pl.stats.add(metadata_id=4).int64_value = 1
    tr = self.trace(space)
    fusion = [o for o in tr.ops(0) if o.base == "fusion"][0]
    self.assertEqual(fusion.stats["hlo_category"], "all-reduce-scatter fusion")
    self.assertNotIn("tf_op", fusion.stats)
    self.assertEqual(tr.plane_stats[0]["has_megacore"], 1)
    self.assertAlmostEqual(tr.plane_stats[0]["peak_teraflops_per_second"], 946.7)


@pytest.mark.scheduled_only
@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "needs tensorflow (its xplane_pb2)")
class MessagesTest(unittest.TestCase):

  def test_the_messages_write_a_capture_back_as_tensorflows_do(self):
    """`xprof` parses with messages built from their descriptor, not tensorflow's xplane_pb2: the two agree on every
    field, and an XSpace with every field set comes back from either byte for byte."""
    from google.protobuf import descriptor_pb2
    from tensorflow.tsl.profiler.protobuf import xplane_pb2 as tf

    def fields(file_descriptor):
      fdp = descriptor_pb2.FileDescriptorProto()
      file_descriptor.CopyToProto(fdp)
      out = {"syntax": fdp.syntax}

      def walk(msgs, prefix):
        for m in msgs:
          out[prefix + m.name] = m.options.map_entry, [o.name for o in m.oneof_decl]
          for f in m.field:
            oneof = f.oneof_index if f.HasField("oneof_index") else None
            out[f"{prefix}{m.name}.{f.name}"] = f.number, f.type, f.label, f.type_name, oneof, f.options.packed
          walk(m.nested_type, f"{prefix}{m.name}.")

      walk(fdp.message_type, "")
      return out

    ours = xprof._xplane_pb2()
    self.assertEqual(fields(ours.XSpace.DESCRIPTOR.file), fields(tf.DESCRIPTOR))
    sp = tf.XSpace(errors=["e"], warnings=["w"], hostnames=["h0", "h1"])
    pl = sp.planes.add(id=-7, name="/device:TPU:0")
    for key in (9, -3, 2**40):
      pl.event_metadata[key].MergeFrom(tf.XEventMetadata(id=key, name="n", display_name="d", metadata=b"\0\xff"))
      pl.event_metadata[key].stats.add(metadata_id=1, str_value="\u00b5s")
      pl.event_metadata[key].child_id.extend([5, -1, 2**62])
    pl.stat_metadata[1].MergeFrom(tf.XStatMetadata(id=1, name="s", description="d"))
    ln = pl.lines.add(id=3, display_id=10, name="XLA Ops", display_name="o", timestamp_ns=-5, duration_ps=2**63 - 1)
    ev = ln.events.add(metadata_id=9, offset_ps=123, duration_ps=4)
    for value in ({"double_value": -0.0}, {"uint64_value": 2**64 - 1}, {"int64_value": -(2**63)}, {"int64_value": 0}):
      ev.stats.add(metadata_id=1, **value)
    ev.stats.add(metadata_id=1, bytes_value=b"\0")
    ev.stats.add(metadata_id=1, ref_value=1)
    ln.events.add(metadata_id=9, num_occurrences=7)
    pl.stats.add(metadata_id=1, double_value=float("nan"))
    data = sp.SerializeToString() + b"\x98\x06\x2a"  # and a field neither knows (99), which both keep
    self.assertEqual(ours.XSpace.FromString(data).SerializeToString(), tf.XSpace.FromString(data).SerializeToString())


if __name__ == "__main__":
  unittest.main()
