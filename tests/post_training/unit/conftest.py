# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# periodic, HANG_WATCHDOG watchdog for pytest execution.
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest conftest with hang watchdog for post-training unit tests."""

import faulthandler
import os
import threading
import pytest

_REAL_STDERR = os.fdopen(os.dup(2), "w", buffering=1)          # dup BEFORE pytest captures fd 2
os.environ["HANG_REAL_STDERR_FD"] = str(_REAL_STDERR.fileno())
_DUMP_AFTER_SECS = float(os.environ.get("HANG_DUMP_AFTER_SECS", "300"))
_EXIT_AFTER_SECS = float(os.environ.get("HANG_EXIT_AFTER_SECS", "900"))
faulthandler.enable(file=_REAL_STDERR, all_threads=True)


def _dump(header):
  _REAL_STDERR.write("\n" + header + "\n")
  _REAL_STDERR.flush()
  faulthandler.dump_traceback(file=_REAL_STDERR, all_threads=True)
  _REAL_STDERR.flush()


@pytest.fixture(autouse=True)
def _hang_watchdog(request):
  node, stop = request.node.nodeid, threading.Event()

  def _watch():
    waited = 0.0
    while not stop.wait(_DUMP_AFTER_SECS):
      waited += _DUMP_AFTER_SECS
      _dump(f"===== HANG WATCHDOG: {node!r} still running after {int(waited)}s; all threads: =====")
      if waited >= _EXIT_AFTER_SECS:
        _dump("===== HANG WATCHDOG: aborting process for CI =====")
        os._exit(99)

  t = threading.Thread(target=_watch, name="hang-watchdog", daemon=True)
  t.start()
  try:
    yield
  finally:
    stop.set()
