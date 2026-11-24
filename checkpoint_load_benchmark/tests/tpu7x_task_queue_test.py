# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests scripts/tpu7x_task_queue.py runs tasks directly from the tarball tree without syncing from GCS."""

import importlib.util
import os
import shutil
import tempfile
import types
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QUEUE_PATH = os.path.join(_REPO_ROOT, "checkpoint_load_benchmark", "scripts", "tpu7x_task_queue.py")

_spec = importlib.util.spec_from_file_location("tpu7x_task_queue", _QUEUE_PATH)
task_queue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task_queue)


class _FakeQueue:
  """Minimal in-memory Queue double for testing run_task without GCS."""

  def __init__(self):
    self.done = []
    self.released = []

  def heartbeat(self, task_id: str, worker: str, gen: int) -> int:
    del task_id, worker
    return gen + 1

  def release(self, task_id: str, gen: int) -> None:
    self.released.append((task_id, gen))

  def mark_done(self, task_id: str, record: dict) -> None:
    self.done.append((task_id, record))


class TaskQueueNoSyncTest(unittest.TestCase):
  """Tasks run the harness inside the extracted tarball rather than overwriting it from GCS."""

  def setUp(self):
    super().setUp()
    self.tree = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tree)
    harness = os.path.join(self.tree, task_queue.HARNESS_REL)
    os.makedirs(os.path.dirname(harness), exist_ok=True)
    self.marker = os.path.join(self.tree, "ran.txt")
    with open(harness, "w", encoding="utf-8") as f:
      f.write(
          "import json, sys\n"
          f"with open({self.marker!r}, 'w', encoding='utf-8') as out:\n"
          "  json.dump(sys.argv[1:], out)\n"
      )

  def test_has_no_sync_harness_helper(self):
    self.assertFalse(hasattr(task_queue, "_sync_harness"))

  def test_run_task_executes_tarball_harness_verbatim(self):
    q = _FakeQueue()
    args = types.SimpleNamespace(
        worker="w1",
        heartbeat_s=60,
        max_attempts=3,
        results="gs://unused/results",
        remote_slice_addrs=None,
    )
    task = {"id": "t1", "slices": 1, "tree": self.tree, "args": ["--models", "qwen3-0.6b"]}
    rc = task_queue.run_task(q, task, gen=1, args=args)
    self.assertEqual(rc, 0)
    self.assertTrue(os.path.isfile(self.marker))
    self.assertEqual(len(q.done), 1)

  def test_hosts_per_slice_daemon_addressing_and_filtering(self):
    q = _FakeQueue()
    args = types.SimpleNamespace(
        worker="w_v6e_4x4",
        heartbeat_s=60,
        max_attempts=3,
        results="gs://unused/results",
        remote_slice_addrs="h1:55555,h2:55555,h3:55555,h4:55555,h5:55555,h6:55555,h7:55555",
        hosts_per_slice=4,
        max_slices=2,
        min_slices=1,
        platform="v6e",
        topology="4x4",
    )
    self.assertTrue(
        task_queue._runnable_by_me(  # pylint: disable=protected-access
            {"id": "ok", "slices": 1, "hosts_per_slice": 4, "platform": "v6e", "topology": "4x4", "tree": self.tree},
            args,
            max_slices=2,
        )
    )
    self.assertFalse(
        task_queue._runnable_by_me(  # pylint: disable=protected-access
            {"id": "wrong_hosts", "slices": 1, "hosts_per_slice": 1, "tree": self.tree},
            args,
            max_slices=2,
        )
    )
    self.assertFalse(
        task_queue._runnable_by_me(  # pylint: disable=protected-access
            {"id": "wrong_plat", "slices": 1, "hosts_per_slice": 4, "platform": "v7x", "tree": self.tree},
            args,
            max_slices=2,
        )
    )

    task_4h = {
        "id": "t_4h",
        "slices": 1,
        "hosts_per_slice": 4,
        "tree": self.tree,
        "args": ["--models", "qwen3-0.6b"],
    }
    orig_reach = task_queue._remote_reachable  # pylint: disable=protected-access
    try:
      task_queue._remote_reachable = lambda _a, needed=None: needed <= 3  # pylint: disable=protected-access
      self.assertEqual(task_queue._usable_slices(args), 1)  # pylint: disable=protected-access
      rc = task_queue.run_task(q, task_4h, gen=1, args=args)
    finally:
      task_queue._remote_reachable = orig_reach  # pylint: disable=protected-access
    self.assertEqual(rc, 0)
    import json  # pylint: disable=import-outside-toplevel

    with open(self.marker, "r", encoding="utf-8") as f:
      argv_seen = json.load(f)
    idx = argv_seen.index("--remote-slice-addrs")
    self.assertEqual(argv_seen[idx + 1 : idx + 4], ["h1:55555", "h2:55555", "h3:55555"])


if __name__ == "__main__":
  unittest.main()
