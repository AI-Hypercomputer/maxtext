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

"""Pytest configuration and fixtures for post-training unit tests."""

import faulthandler
import os
import sys
import threading

import pytest


_DUMP_PATH = (
    os.environ.get("HANG_DUMP_FILE")
    or os.environ.get("GITHUB_STEP_SUMMARY")
    or "/tmp/hang_watchdog_dump.txt"
)
_DUMP_FH = open(_DUMP_PATH, "a", buffering=1)
os.environ["HANG_DUMP_FILE"] = _DUMP_PATH
_DUMP_AFTER_SECS = float(os.environ.get("HANG_DUMP_AFTER_SECS", "300"))
_EXIT_AFTER_SECS = float(os.environ.get("HANG_EXIT_AFTER_SECS", "900"))
faulthandler.enable(file=_DUMP_FH, all_threads=True)


def _dump(header):
  for sink in (_DUMP_FH, sys.__stderr__):
    try:
      sink.write("\n" + header + "\n")
      sink.flush()
      faulthandler.dump_traceback(file=sink, all_threads=True)
      sink.flush()
    except Exception:
      pass


@pytest.fixture(autouse=True)
def _hang_watchdog(request):
  """Watchdog fixture to detect and dump stack traces for hanging tests."""
  node = request.node.nodeid
  stop = threading.Event()

  def _watch():
    waited = 0.0
    while not stop.wait(_DUMP_AFTER_SECS):
      waited += _DUMP_AFTER_SECS
      _dump(
          f"===== HANG WATCHDOG: {node!r} still running after {int(waited)}s;"
          " all threads: ====="
      )
      if waited >= _EXIT_AFTER_SECS:
        _dump("===== HANG WATCHDOG: aborting process for CI =====")
        try:
          os.fsync(_DUMP_FH.fileno())
        except Exception:
          pass
        os._exit(99)

  t = threading.Thread(target=_watch, name="hang-watchdog", daemon=True)
  t.start()
  try:
    yield
  finally:
    stop.set()
