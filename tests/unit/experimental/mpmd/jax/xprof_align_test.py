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

"""Tests for xprof_align: many hosts' xplanes in one XProf session, on one clock.

Synthetic XSpaces from `xprof`'s descriptor-built messages (no tensorflow, no
jax): a 'Task Environment' plane holding profile_start_time, and a device
plane whose lines the rebase moves.  gs:// paths go to an in-memory stand-in
for google.cloud.storage, put in sys.modules for the test.
"""

import concurrent.futures as cf
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import xprof_align as xa

try:  # the descriptor-built messages xprof_align parses with: protobuf, no tensorflow
  xa._xplane_pb2()
  HAVE_PB = True
except ImportError:  # pragma: no cover
  HAVE_PB = False

S = 1_000_000_000  # ns


def space(start_ns, line_ts=(0, 5_000), stat="uint64_value"):
  """A host's capture: its profile start, and two device lines at `line_ts`."""
  sp = xa._xplane_pb2().XSpace()
  env = sp.planes.add(name="Task Environment", id=1)
  env.stat_metadata[7].id = 7
  env.stat_metadata[7].name = xa.START_STAT
  setattr(env.stats.add(metadata_id=7), stat, start_ns)
  dev = sp.planes.add(name="/device:TPU:0", id=2)
  dev.event_metadata[1].name = "fusion.1"
  for i, t in enumerate(line_ts):
    ln = dev.lines.add(id=i, timestamp_ns=t)
    ln.events.add(metadata_id=1, offset_ps=1_000 * (i + 1), duration_ps=500)
  return sp


def lines(sp):
  return [ln.timestamp_ns for pl in sp.planes for ln in pl.lines]


def offsets(sp):
  return [ev.offset_ps for pl in sp.planes for ln in pl.lines for ev in ln.events]


def write(path, sp):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    f.write(sp.SerializeToString())
  return path


@unittest.skipUnless(HAVE_PB, "needs protobuf for the xplane protobuf")
class RebaseTest(unittest.TestCase):

  def test_lines_move_events_stay(self):
    sp = space(1_000, (0, 5))
    self.assertEqual(xa.rebase(sp, 400), 600)
    self.assertEqual(lines(sp), [600, 605])
    self.assertEqual(offsets(sp), [1_000, 2_000])  # relative to their line: untouched
    self.assertEqual(xa.start_time_ns(sp), 400)
    self.assertEqual(xa._start_stat(sp)[1].WhichOneof("value"), "uint64_value")

  def test_an_int64_start_stays_int64(self):
    sp = space(1_000, stat="int64_value")
    xa.rebase(sp, 250)
    self.assertEqual(xa._start_stat(sp)[1].WhichOneof("value"), "int64_value")
    self.assertEqual(xa.start_time_ns(sp), 250)

  def test_an_origin_after_the_start_raises_and_changes_nothing(self):
    sp = space(1_000, (0, 5))
    with self.assertRaisesRegex(ValueError, "after this host"):
      xa.rebase(sp, 1_001)
    self.assertEqual((lines(sp), xa.start_time_ns(sp)), ([0, 5], 1_000))

  def test_no_start_stat_raises(self):
    with self.assertRaisesRegex(ValueError, xa.START_STAT):
      xa.rebase(xa._xplane_pb2().XSpace(), 0)


@unittest.skipUnless(HAVE_PB, "needs protobuf for the xplane protobuf")
class AlignTest(unittest.TestCase):

  def setUp(self):
    self.d = self.enterContext(tempfile.TemporaryDirectory())

  def session(self, name="sess"):
    return f"{self.d}/out/plugins/profile/{name}"

  def test_every_host_onto_the_earliest_start(self):
    """The v6e-256 leaders' case: starts 26.7 and 83.1 ms after the first, in two spawned workers."""
    starts = {"hostA": S, "hostB": S + 26_700_000, "hostC": S + 83_100_000}
    for h, s in starts.items():
      write(f"{self.d}/raw/{h}/plugins/profile/2026_10_05/{h}.xplane.pb", space(s))
    with mock.patch.object(cf, "ProcessPoolExecutor", wraps=cf.ProcessPoolExecutor) as pool:
      shifts = xa.align(
          xa.find_xplanes(f"{self.d}/raw"), f"{self.d}/out", "sess", workers=2, names={"hostB": "s1-p1-hostB"}
      )
    self.assertEqual(pool.call_args.args[0], 2)
    self.assertEqual(pool.call_args.kwargs["mp_context"].get_start_method(), "spawn")  # fork can deadlock jax
    self.assertEqual(shifts, {"hostA.xplane.pb": 0.0, "s1-p1-hostB.xplane.pb": 26.7, "hostC.xplane.pb": 83.1})
    self.assertEqual(sorted(os.listdir(self.session())), sorted(shifts))
    for f, ms in shifts.items():
      sp = xa._load(f"{self.session()}/{f}")
      self.assertEqual(xa.start_time_ns(sp), S)
      self.assertEqual(lines(sp), [round(ms * 1e6), round(ms * 1e6) + 5_000])

  def test_names_by_path_and_by_directory(self):
    a = write(f"{self.d}/runA/plugins/profile/t1/h.xplane.pb", space(S))
    b = write(f"{self.d}/runB/plugins/profile/t2/h.xplane.pb", space(S + 1_000_000))
    c = write(f"{self.d}/runC/plugins/profile/t3/g.xplane.pb", space(S + 2_000_000))
    names = {a: "s0-h", f"{self.d}/runB/": "outer", f"{self.d}/runB/plugins": "s1-h", "g": "s2-g"}
    shifts = xa.align([a, b, c], f"{self.d}/out", "sess", workers=1, names=names)
    self.assertEqual(shifts, {"s0-h.xplane.pb": 0.0, "s1-h.xplane.pb": 1.0, "s2-g.xplane.pb": 2.0})
    self.assertEqual(xa._labels([a, b], {f"{self.d}/runB": "s1-h"}), ["h", "s1-h"])  # the nearest directory wins

  def test_two_captures_of_one_host_get_distinct_labels(self):
    """Two processes of one machine, or two runs: the capture directory, then a count."""
    a = write(f"{self.d}/runA/plugins/profile/2026_01/h.xplane.pb", space(S))
    b = write(f"{self.d}/runB/plugins/profile/2026_02/h.xplane.pb", space(S + 1_000_000))
    shifts = xa.align([a, b], f"{self.d}/out", "sess", workers=2)
    self.assertEqual(shifts, {"h-2026_01.xplane.pb": 0.0, "h-2026_02.xplane.pb": 1.0})
    same_dir = [f"{self.d}/{r}/plugins/profile/t/h.xplane.pb" for r in ("x", "y", "z")]
    self.assertEqual(xa._labels(same_dir, {}), ["h-t", "h-t-1", "h-t-2"])
    self.assertEqual(xa._labels([a, b], {"h": "s0"}), ["s0-2026_01", "s0-2026_02"])

  def test_rewriting_a_session_drops_its_xprof_caches(self):
    """XProf would keep drawing a rewritten host from <host>.SSTABLE & co."""
    sess = self.session()
    os.makedirs(sess)
    stale = ["hostA.SSTABLE", "hostA.metadata.SSTABLE", "hostA.trie.SSTABLE", "cache_version.txt"]
    kept = ["hostB.SSTABLE", "hostAB.SSTABLE", "hostB.xplane.pb", "hostA.b.xplane.pb"]
    for f in stale + kept:
      open(f"{sess}/{f}", "w").close()
    p = write(f"{self.d}/raw/hostA.xplane.pb", space(S))
    xa.align([p], f"{self.d}/out", "sess", workers=1)
    self.assertEqual(sorted(os.listdir(sess)), sorted(kept + ["hostA.xplane.pb"]))

  def test_cli(self):
    for h, s in (("hostA", S), ("hostB", S + 3_000_000)):
      write(f"{self.d}/raw/{h}.xplane.pb", space(s))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      xa.main([f"{self.d}/raw", "--out", f"{self.d}/out", "--session", "s", "--name", "hostB=s1-hostB", "--workers", "2"])
    self.assertIn("s1-hostB.xplane.pb: +3.000 ms", out.getvalue())
    self.assertEqual(sorted(os.listdir(self.session("s"))), ["hostA.xplane.pb", "s1-hostB.xplane.pb"])
    for bad in ("hostB", "hostB=", "=s1"):
      err = io.StringIO()
      with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
        xa.main([f"{self.d}/raw", "--out", f"{self.d}/out", "--session", "s", "--name", bad])
      self.assertEqual(e.exception.code, 2)
      self.assertIn("want HOST=LABEL", err.getvalue())

  @unittest.skipUnless(os.path.exists("/proc/meminfo"), "reads MemAvailable")
  def test_workers_fit_in_memory(self):
    paths = [write(f"{self.d}/raw/h{i}.xplane.pb", space(S)) for i in range(3)]
    self.assertEqual(xa._pool_size(paths, 8), 3)
    with mock.patch.object(xa, "_size", return_value=10**15):  # a petabyte capture: one at a time
      self.assertEqual(xa._pool_size(paths, 8), 1)

  def test_one_worker_runs_in_this_process(self):
    """workers=1, one path, or room for one parsed capture: no process to start, and no __main__ guard to need."""
    paths = [write(f"{self.d}/raw/h{i}.xplane.pb", space(S + i * 1_000_000)) for i in range(2)]
    both = {"h0.xplane.pb": 0.0, "h1.xplane.pb": 1.0}
    with mock.patch.object(cf, "ProcessPoolExecutor", side_effect=AssertionError("a process pool")):
      self.assertEqual(xa.align(paths, f"{self.d}/out", "a", workers=1), both)
      self.assertEqual(xa.align(paths[1:], f"{self.d}/out", "b"), {"h1.xplane.pb": 0.0})
      with mock.patch.object(xa, "_pool_size", return_value=1):
        self.assertEqual(xa.align(paths, f"{self.d}/out", "c"), both)
    self.assertEqual(xa.start_time_ns(xa._load(self.session("c") + "/h1.xplane.pb")), S)

  def test_a_worker_imports_no_tensorflow(self):
    """What a spawned worker runs -- read, rebase, write -- in a fresh interpreter.  Importing tensorflow, where it is
    installed, would add some 10 s and 0.7 GB to every worker.  The worker's first import finder records every import
    of tensorflow and refuses it with a RuntimeError, which no `except ImportError` fallback swallows: so an attempt
    fails the test whether tensorflow is installed or not (CI installs none)."""
    p = write(f"{self.d}/raw/h.xplane.pb", space(S))
    code = (
        "import sys\n"
        "class Refuse:\n"
        "  tried = []\n"
        "  def find_spec(self, name, path=None, target=None):\n"
        "    if name.split('.')[0] == 'tensorflow':\n"
        "      Refuse.tried.append(name)\n"
        '      raise RuntimeError(f"a worker imports {name}")\n'
        "    return None\n"
        "sys.meta_path.insert(0, Refuse())\n"
        "from maxtext.experimental.mpmd.jax import xprof_align as xa\n"
        "xa._rebase_into(sys.argv[1], xa._start_of(sys.argv[1]) - 1_000, sys.argv[2])\n"
        "print(Refuse.tried, 'tensorflow' in sys.modules)\n"
    )
    env = dict(os.environ, PYTHONPATH=testing.child_pythonpath(__name__))
    r = subprocess.run(
        [sys.executable, "-c", code, p, f"{self.d}/h.xplane.pb"], env=env, capture_output=True, text=True, timeout=300
    )
    self.assertEqual((r.returncode, r.stdout.strip()), (0, "[] False"), r.stderr)
    self.assertEqual(lines(xa._load(f"{self.d}/h.xplane.pb")), [1_000, 6_000])


@unittest.skipUnless(HAVE_PB, "needs protobuf for the xplane protobuf")
class ExportTest(unittest.TestCase):

  def setUp(self):
    self.d = self.enterContext(tempfile.TemporaryDirectory())
    write(f"{self.d}/local/plugins/profile/2026_10_05/raw-host.xplane.pb", space(1_000, (0, 5)))

  def test_rebased_into_the_session_under_its_name(self):
    sess = f"{self.d}/out/plugins/profile/sess"
    os.makedirs(sess)
    for f in ("s0-p0-h.SSTABLE", "cache_version.txt", "s1-p1-h.SSTABLE"):
      open(f"{sess}/{f}", "w").close()
    dst = xa.export_into_session(f"{self.d}/local", f"{self.d}/out/", "sess", 400, host="s0-p0-h")
    self.assertEqual(dst, f"{sess}/s0-p0-h.xplane.pb")
    sp = xa._load(dst)
    self.assertEqual((xa.start_time_ns(sp), lines(sp)), (400, [600, 605]))
    self.assertEqual(sorted(os.listdir(sess)), ["s0-p0-h.xplane.pb", "s1-p1-h.SSTABLE"])

  def test_a_clock_past_the_origin_is_written_unaligned_with_a_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      dst = xa.export_into_session(f"{self.d}/local", f"{self.d}/out", "sess", 1_500)
    self.assertTrue(dst.endswith("/sess/raw-host.xplane.pb"), dst)
    self.assertEqual([x.category for x in w], [RuntimeWarning])
    self.assertIn("written unaligned", str(w[0].message))
    sp = xa._load(dst)
    self.assertEqual((xa.start_time_ns(sp), lines(sp)), (1_000, [0, 5]))

  def test_gs_session(self):
    """The gs:// branch: written through the client, caches listed and deleted
    (one another host deleted first is no error)."""
    store = {
        ("bkt", "x/plugins/profile/sess/h.SSTABLE"): b"",
        ("bkt", "x/plugins/profile/sess/cache_version.txt"): b"",
        ("bkt", "x/plugins/profile/sess/g.SSTABLE"): b"",
        ("bkt", "x/plugins/profile/sess/sub/h.SSTABLE"): b"",
    }
    with fake_gcs(store, gone={"x/plugins/profile/sess/cache_version.txt"}):
      dst = xa.export_into_session(f"{self.d}/local", "gs://bkt/x", "sess", 400, host="h")
    self.assertEqual(dst, "gs://bkt/x/plugins/profile/sess/h.xplane.pb")
    self.assertEqual(
        sorted(n for _, n in store),
        [
            "x/plugins/profile/sess/g.SSTABLE",
            "x/plugins/profile/sess/h.xplane.pb",
            "x/plugins/profile/sess/sub/h.SSTABLE",
        ],
    )
    sp = xa._xplane_pb2().XSpace()
    sp.ParseFromString(store[("bkt", "x/plugins/profile/sess/h.xplane.pb")])
    self.assertEqual(xa.start_time_ns(sp), 400)


@contextlib.contextmanager
def fake_gcs(store, gone=()):
  """google.cloud.storage over `store` ({(bucket, name): bytes}); deleting a
  name in `gone` raises NotFound, as when another host deleted it first."""

  class NotFound(Exception):
    pass

  class Blob:

    def __init__(self, bucket, name):
      self.bucket, self.name = bucket, name

    def download_as_bytes(self):
      return store[(self.bucket, self.name)]

    def upload_from_string(self, data):
      store[(self.bucket, self.name)] = bytes(data)

    def delete(self):
      store.pop((self.bucket, self.name))
      if self.name in gone:
        raise NotFound(self.name)

  class Client:

    def bucket(self, b):
      return types.SimpleNamespace(blob=lambda n: Blob(b, n))

    def list_blobs(self, b, prefix="", delimiter=None):
      return [
          Blob(bb, n)
          for bb, n in sorted(store)
          if bb == b and n.startswith(prefix) and not (delimiter and delimiter in n[len(prefix) :])
      ]

  cloud, storage = types.ModuleType("google.cloud"), types.ModuleType("google.cloud.storage")
  cloud.storage, storage.Client = storage, Client
  exceptions = types.ModuleType("google.api_core.exceptions")
  exceptions.NotFound = NotFound
  with mock.patch.dict(
      sys.modules, {"google.cloud": cloud, "google.cloud.storage": storage, "google.api_core.exceptions": exceptions}
  ):
    yield


if __name__ == "__main__":
  unittest.main()
