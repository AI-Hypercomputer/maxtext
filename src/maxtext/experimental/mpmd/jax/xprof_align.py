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

"""Every host of a multi-host profile in ONE XProf session, on one clock.

Each host's `jax.profiler.trace` writes `<logdir>/plugins/profile/<its own
timestamp>/<host>.xplane.pb`, and every timeline in it -- TPU devices,
SparseCores, host threads -- counts from that host's own
`profile_start_time` (a stat of the 'Task Environment' plane; the device
lines carry timestamp_ns = 0).  XProf's trace viewer merges the hosts of one
session (`hosts=a,b,...`) but takes each file's times as they are, so hosts
that started tracing at different moments land shifted against each other.
On the v6e-256 runs the four stage leaders started one stage drain apart --
+26.7, +53.4, +83.1 ms -- which in an MPMD timeline puts a stage's forward
before the copy that feeds it.

This rebases every host onto one origin -- each line's timestamp_ns moves by
(that host's profile_start_time - origin), and profile_start_time becomes the
origin -- and writes the hosts side by side in one session:

    python3 -m maxtext.experimental.mpmd.jax.xprof_align SRC [SRC ...] --out LOGDIR --session NAME

SRC: xplane files, or directories (local or gs://) holding `*.xplane.pb` at
any depth; LOGDIR/plugins/profile/NAME/ gets one `<host>.xplane.pb` each.
`xprof --logdir LOGDIR` then lists session NAME, and its trace viewer takes
every host at once.  Checked on the v6e-256 ppermute run: the boundary
collective-permutes, which end at the same instant on both stages, line up
across hosts to within 50 us after the shift (26.7 ms apart before it).

`ds_ep_pipeline --distributed --profile` does this as it writes
(`export_into_session`): every traced host joins one session, rebased to an
origin process 0 picks.

XProf caches what it converts from each host's xplane beside it (<host>.SSTABLE,
...) and serves that while the session's cache_version.txt is current.  Both
writers therefore remove the caches of every host they write, and
cache_version.txt, so re-aligning into a session XProf has shown takes effect.

What a traced host captures, and the name it is listed under (in stage order),
are profiling.py's.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import glob
import multiprocessing
import os
import sys
import tempfile
import warnings

START_STAT = "profile_start_time"


def _xplane_pb2():
  """`xprof`'s xplane messages, built from their descriptor: never tensorflow's, whose import would cost each process
  that parses a capture -- every worker `align` spawns, a pipeline host exporting its trace -- some 10 s and 0.7 GB."""
  from maxtext.experimental.mpmd.jax import xprof

  return xprof._xplane_pb2()


# ------------------------------------------------------------------- I/O
def _blob(url):
  from google.cloud import storage

  bucket, _, name = url[5:].partition("/")
  return storage.Client().bucket(bucket).blob(name)


def read_bytes(path):
  if path.startswith("gs://"):
    return _blob(path).download_as_bytes()
  with open(path, "rb") as f:
    return f.read()


def write_bytes(path, data):
  if path.startswith("gs://"):
    _blob(path).upload_from_string(data)
    return
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    f.write(data)


def delete(path):
  """Remove the file `path` (local or gs://)."""
  if path.startswith("gs://"):
    _blob(path).delete()
    return
  os.remove(path)


def find_xplanes(src):
  """Every `*.xplane.pb` at or below `src` (a file or a directory, local or gs://)."""
  if src.endswith(".xplane.pb"):
    return [src]
  if src.startswith("gs://"):
    from google.cloud import storage

    bucket, _, prefix = src[5:].partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    return sorted(
        f"gs://{bucket}/{b.name}"
        for b in storage.Client().list_blobs(bucket, prefix=prefix)
        if b.name.endswith(".xplane.pb")
    )
  return sorted(glob.glob(os.path.join(src, "**", "*.xplane.pb"), recursive=True))


# ---------------------------------------------------------------- the shift
def _start_stat(space):
  """(plane, stat) holding the host's profile_start_time, or (None, None)."""
  for pl in space.planes:
    names = {k: v.name for k, v in pl.stat_metadata.items()}
    for s in pl.stats:
      if names.get(s.metadata_id) == START_STAT:
        return pl, s
  return None, None


def start_time_ns(space):
  """The host's absolute profile start (ns since the epoch)."""
  _, s = _start_stat(space)
  if s is None:
    raise ValueError(f"no {START_STAT} stat in this xplane: cannot place it on a common clock")
  return int(s.int64_value or s.uint64_value)


def rebase(space, origin_ns):
  """Move every timeline of `space` from its own profile start onto
  `origin_ns` (ns since the epoch, at or before the start) -> the shift (ns)."""
  start = start_time_ns(space)
  shift = start - origin_ns
  if shift < 0:
    raise ValueError(f"origin {origin_ns} is after this host's profile start {start}")
  for pl in space.planes:
    for ln in pl.lines:
      ln.timestamp_ns += shift
  _, s = _start_stat(space)
  if s.HasField("uint64_value"):
    s.uint64_value = origin_ns
  else:
    s.int64_value = origin_ns
  return shift


def _load(path):
  space = _xplane_pb2().XSpace()
  space.ParseFromString(read_bytes(path))
  return space


def _start_of(path):
  return start_time_ns(_load(path))


def _size(path):
  if path.startswith("gs://"):
    b = _blob(path)
    b.reload()
    return b.size or 0
  return os.path.getsize(path)


def _pool_size(paths, workers):
  """`workers`, fewer for big xplanes: each worker holds one parsed XSpace, some
  8x its file (a 1.93 GB TPU capture: 11.5 GB parsed, 15.7 GB while rebased)."""
  try:
    with open("/proc/meminfo") as f:
      avail = next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith("MemAvailable:"))
  except (OSError, StopIteration, ValueError):
    return min(workers, len(paths))
  return max(1, min(workers, len(paths), avail // (8 * max(map(_size, paths)) + 1)))


def _rebase_into(path, origin_ns, dst):
  space = _load(path)
  shift = rebase(space, origin_ns)
  write_bytes(dst, space.SerializeToString())
  return shift


def _labels(paths, names):
  """The host XProf lists each xplane under (its file's name before
  `.xplane.pb`): names[its path, or the nearest directory above it in names],
  else names[the host jax wrote], else that host.  Captures that would share
  one get their directory's name appended (jax's capture timestamp), then a
  count: a session holds one file per host."""
  norm = lambda p: p.rstrip("/") if p.startswith("gs://") else os.path.abspath(p)
  host = lambda p: os.path.basename(p)[: -len(".xplane.pb")]

  def label(p):
    under = [(len(norm(k)), v) for k, v in names.items() if (norm(p) + "/").startswith(norm(k) + "/")]
    return max(under)[1] if under else names.get(host(p), host(p))

  labels = [label(p) for p in paths]
  shared = collections.Counter(labels)
  labels = [f"{l}-{os.path.basename(os.path.dirname(p))}" if shared[l] > 1 else l for p, l in zip(paths, labels)]
  seen = set()
  for i, l in enumerate(labels):
    k = 1
    while labels[i] in seen:
      labels[i] = f"{l}-{k}"
      k += 1
    seen.add(labels[i])
  return labels


def _drop_xprof_caches(session_dir, labels):
  """Remove XProf's caches of the hosts `labels` in session_dir (every
  `<label>.*` but an xplane) and the session's cache_version.txt, without which
  XProf converts afresh.  Run after writing their xplanes: a view while they
  are written would cache the old ones again."""
  heads = tuple(f"{l}." for l in labels)
  stale = lambda n: n == "cache_version.txt" or (n.startswith(heads) and not n.endswith(".xplane.pb"))
  if session_dir.startswith("gs://"):
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    bucket, _, prefix = session_dir[5:].partition("/")
    for b in storage.Client().list_blobs(bucket, prefix=f"{prefix}/", delimiter="/"):
      if stale(b.name[len(prefix) + 1 :]):
        with contextlib.suppress(NotFound):  # another host of the session was first
          b.delete()
  elif os.path.isdir(session_dir):
    for n in os.listdir(session_dir):
      if stale(n):
        with contextlib.suppress(FileNotFoundError):
          os.remove(os.path.join(session_dir, n))


def align(paths, out_logdir, session, workers=8, names=None):
  """Rebase the xplanes `paths` (one per host) onto their earliest profile
  start and write them to out_logdir/plugins/profile/session/ -> {host file:
  shift in ms}.  `names`: {host, or an xplane's path or a directory above it:
  label} -- XProf takes a file's name before `.xplane.pb` as its host, so a
  label such as `s0-p0-<host>` lists the hosts by stage.  Two captures of one
  host (two processes of one machine, or two runs) get distinct labels
  (`_labels`).

  Two or more workers are processes it spawns, and a spawned process imports
  the caller's main module afresh: call `align` from a script under
  `if __name__ == '__main__':`.  One worker (`workers=1`, one path, or
  captures too big for two in memory) runs in this process."""
  labels = _labels(paths, names or {})
  dst_dir = f'{out_logdir.rstrip("/")}/plugins/profile/{session}'
  pool = _pool_size(paths, workers)
  with contextlib.ExitStack() as stack:
    run = map  # one worker: a pool of one would only add a process start
    if pool > 1:
      # spawn, not fork: the caller may be multithreaded -- a jax process, a pytest-xdist worker -- and fork can
      # deadlock one; the workers (_start_of, _rebase_into) are module-level and pickle.
      run = stack.enter_context(cf.ProcessPoolExecutor(pool, mp_context=multiprocessing.get_context("spawn"))).map
    starts = list(run(_start_of, paths))
    origin = min(starts)
    shifts = list(run(_rebase_into, paths, [origin] * len(paths), [f"{dst_dir}/{l}.xplane.pb" for l in labels]))
  _drop_xprof_caches(dst_dir, labels)
  return {f"{l}.xplane.pb": s / 1e6 for l, s in zip(labels, shifts)}


# ------------------------------------------------ inside a multi-host run
def export_into_session(local_logdir, out_logdir, session, origin_ns, host=None):
  """After `jax.profiler.stop_trace()` into `local_logdir` on one host: its
  xplane, rebased to `origin_ns` (one every traced host shares), ->
  out_logdir/plugins/profile/session/<host>.xplane.pb (`host`: the name XProf
  lists it under; default the hostname jax wrote).  -> the destination."""
  found = find_xplanes(local_logdir)
  if len(found) != 1:
    raise RuntimeError(f"expected one xplane under {local_logdir}, found {found}")
  space = _load(found[0])
  try:
    rebase(space, origin_ns)
  except ValueError as e:  # a clock far off the origin's: keep the data, unaligned
    warnings.warn(f"{found[0]}: written unaligned ({e})", RuntimeWarning)
  name = f"{host}.xplane.pb" if host else os.path.basename(found[0])
  dst_dir = f'{out_logdir.rstrip("/")}/plugins/profile/{session}'
  write_bytes(f"{dst_dir}/{name}", space.SerializeToString())
  try:  # a fresh session has no caches: the export stands without this
    _drop_xprof_caches(dst_dir, [name[: -len(".xplane.pb")]])
  except Exception as e:  # noqa: BLE001
    warnings.warn(f"{dst_dir}: XProf caches not removed ({type(e).__name__}: {e})", RuntimeWarning)
  return f"{dst_dir}/{name}"


def local_trace_dir():
  return tempfile.mkdtemp(prefix="mpmd_xprof_")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("src", nargs="+", help="xplane files or directories (local or gs://)")
  ap.add_argument("--out", required=True, help="the XProf logdir to write (local or gs://)")
  ap.add_argument("--session", required=True, help="the session (run) name inside it")
  ap.add_argument(
      "--name",
      action="append",
      default=[],
      metavar="HOST=LABEL",
      help="list HOST under LABEL in XProf (e.g. a stage prefix: s0-p0-HOST); HOST may "
      "also be an xplane or a directory above it; repeatable",
  )
  ap.add_argument("--workers", type=int, default=8)
  a = ap.parse_args(argv)
  names = {}
  for n in a.name:
    k, _, v = n.partition("=")
    if not (k and v):
      ap.error(f"--name {n!r}: want HOST=LABEL")
    names[k] = v
  paths = [p for s in a.src for p in find_xplanes(s)]
  if not paths:
    sys.exit(f"no *.xplane.pb under {a.src}")
  shifts = align(paths, a.out, a.session, a.workers, names=names)
  for n, ms in shifts.items():
    print(f"  {n}: +{ms:.3f} ms")
  print(f'{len(paths)} hosts -> {a.out.rstrip("/")}/plugins/profile/{a.session}/')


if __name__ == "__main__":
  main()
