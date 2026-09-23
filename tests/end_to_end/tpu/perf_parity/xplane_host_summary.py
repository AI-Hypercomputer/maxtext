# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregates host-side XPlane events by name, straight off the wire format.

Neither `xprof` nor `tensorflow.core.profiler.protobuf` is importable in this
venv, and `tensorboard_plugin_profile`'s generated protos are too old for the
installed protobuf runtime. XSpace is a small enough schema to read directly,
and only three message types are needed to answer "what did the host spend its
time on, per event name".

Usage: python xplane_host_summary.py <a.xplane.pb> [<b.xplane.pb> ...]
"""

import collections
import sys


def _fields(buf, start=0, end=None):
  """Yields `(field_number, wire_type, value)` for one protobuf message.

  `value` is an int for varint/fixed fields and a `memoryview` slice for
  length-delimited ones, so nested messages are walked without copying.
  """
  end = len(buf) if end is None else end
  i = start
  while i < end:
    key, i = _varint(buf, i)
    field, wire = key >> 3, key & 7
    if wire == 0:
      val, i = _varint(buf, i)
    elif wire == 1:
      val, i = int.from_bytes(buf[i : i + 8], "little"), i + 8
    elif wire == 2:
      ln, i = _varint(buf, i)
      val, i = buf[i : i + ln], i + ln
    elif wire == 5:
      val, i = int.from_bytes(buf[i : i + 4], "little"), i + 4
    else:
      raise ValueError(f"unsupported wire type {wire} at {i}")
    yield field, wire, val


def _varint(buf, i):
  shift = result = 0
  while True:
    b = buf[i]
    i += 1
    result |= (b & 0x7F) << shift
    if not b & 0x80:
      return result, i
    shift += 7


def _one(msg, field, default=None):
  for f, _, v in _fields(msg):
    if f == field:
      return v
  return default


def _str(msg, field):
  v = _one(msg, field)
  return bytes(v).decode("utf-8", "replace") if v is not None else ""


def planes(space):
  """XSpace.planes = 1."""
  for f, _, v in _fields(space):
    if f == 1:
      yield v


def event_metadata(plane):
  """XPlane.event_metadata = 4, a map<int64, XEventMetadata>.

  Proto3 map entries are messages with key=1 and value=2; the value's own
  `name` is field 2.
  """
  names = {}
  for f, _, entry in _fields(plane):
    if f != 4:
      continue
    key = _one(entry, 1, 0)
    meta = _one(entry, 2)
    if meta is not None:
      names[key] = _str(meta, 2)
  return names


def lines(plane):
  """XPlane.lines = 3."""
  for f, _, v in _fields(plane):
    if f == 3:
      yield v


def events(line):
  """XLine.events = 4; XEvent.metadata_id = 1, duration_ps = 3."""
  for f, _, ev in _fields(line):
    if f == 4:
      yield _one(ev, 1, 0), _one(ev, 3, 0)


def summarize(path, top=25):
  """Prints the `top` costliest host-side event names in one xplane.pb."""
  with open(path, "rb") as fh:
    space = memoryview(fh.read())

  print(f"\n===== {path}")
  for plane in planes(space):
    plane_name = _str(plane, 2)
    # Device planes are XLA ops, not the Python overhead this is looking for.
    if "TPU" in plane_name and "Python" not in plane_name:
      continue
    names = event_metadata(plane)
    per_event = collections.defaultdict(lambda: [0, 0])
    for line in lines(plane):
      for meta_id, dur_ps in events(line):
        agg = per_event[names.get(meta_id, f"<{meta_id}>")]
        agg[0] += 1
        agg[1] += dur_ps
    if not per_event:
      continue
    total_ms = sum(v[1] for v in per_event.values()) / 1e9
    print(f"\n-- plane {plane_name!r}: {len(per_event)} distinct events, {total_ms:.1f} ms summed")
    ranked = sorted(per_event.items(), key=lambda kv: -kv[1][1])[:top]
    for name, (count, dur_ps) in ranked:
      print(f"   {dur_ps / 1e9:9.2f} ms  n={count:<6d} {name[:110]}")


if __name__ == "__main__":
  for p in sys.argv[1:]:
    summarize(p)
