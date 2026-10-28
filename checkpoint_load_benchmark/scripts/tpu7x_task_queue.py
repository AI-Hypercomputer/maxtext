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

"""Preemption-tolerant task queue in GCS for finishing tpu7x benchmark runs on several workers.

Each task is one invocation of `benchmark_ckpt_load_time.py`. The task list is a JSON file in the queue
directory (`<queue>/tasks.json`, re-read on every iteration so tasks can be appended while workers run).
A worker claims a task by creating `<queue>/locks/<id>.json` with `if_generation_match=0` (atomic), refreshes
a heartbeat in it while the harness runs, and writes `<queue>/done/<id>.json` when the harness exits.
A lock whose heartbeat is older than `--stale-after-s` belongs to a preempted worker and is taken over; the
harness's `--skip-completed` then resumes the task from its last finished phase.

Workers run on pods built by `bootstrap_tpu7x.py`, so the code trees named by tasks (`/tmp/maxtext`,
`/tmp/maxtext_origin_main`, `/tmp/maxtext_hbmfix`, `/tmp/maxtext_stagefix`) exist on both slices of a pair. A task
is only claimed by a worker with enough slices (`--max-slices`) whose pod has the task's tree. Before each task, the
worker copies the latest harness from `--code-dir` into the task's tree. Worker processes started before this sync
existed keep the harness from pod start; tasks that need a newer harness name a tree only newer pods have
(`/tmp/maxtext_stagefix`).

Task fields: `id`, `slices` (1 or 2), `tree`, `args` (harness CLI args), optional `env`, `depends_on`
(ids that must be done first) and `csv` (name of the CSV uploaded to `<results>/`).

Exit codes: 0 = no task left that this worker can ever run; 2 = ran `--max-tasks` tasks; 3 = nothing claimable
right now (dependencies pending or tasks locked by live workers) and `--no-wait` was given.

Usage:
  # On the slice-0 pod of a 2-slice pair (REMOTE_SLICE1_ADDR is exported by bootstrap_tpu7x.py):
  python3 tpu7x_task_queue.py --queue gs://mesa-maxtext/mesakh_tpu7x_bench/queue/p1 --worker g --max-slices 2
  # Slice 0 of a 4-slice group (REMOTE_SLICE1_ADDR lists the 3 daemons), only taking 4-slice tasks:
  python3 tpu7x_task_queue.py --queue gs://... --worker m --max-slices 4 --min-slices 4
  # One task per call on a single slice (e.g. sent to a daemon pod by tpu7x_daemon_controller.py):
  python3 tpu7x_task_queue.py --queue gs://... --worker f1 --max-slices 1 --max-tasks 1 --no-wait
"""

import argparse
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

from google.api_core import exceptions as gexc
from google.cloud import storage

HARNESS_REL = "checkpoint_load_benchmark/benchmark_ckpt_load_time.py"
DEFAULT_RESULTS = "gs://mesa-maxtext/mesakh_tpu7x_bench/results"
DEFAULT_CODE_DIR = "gs://mesa-maxtext/mesakh_tpu7x_bench/code"


def _now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
  print(f"[task_queue {_now()}] {msg}", flush=True)


class Queue:
  """GCS-backed task list, locks and done markers."""

  def __init__(self, queue_uri: str):
    bucket, _, prefix = queue_uri.removeprefix("gs://").partition("/")
    self.client = storage.Client()
    self.bucket = self.client.bucket(bucket)
    self.prefix = prefix.rstrip("/")

  def _blob(self, name: str):
    return self.bucket.blob(f"{self.prefix}/{name}")

  def tasks(self) -> list[dict]:
    return json.loads(self._blob("tasks.json").download_as_text())["tasks"]

  def done_ids(self) -> set[str]:
    return {
        b.name.rsplit("/", 1)[1].removesuffix(".json")
        for b in self.client.list_blobs(self.bucket, prefix=f"{self.prefix}/done/")
    }

  def lock(self, task_id: str):
    """Returns (lock dict, generation) or (None, None)."""
    blob = self._blob(f"locks/{task_id}.json")
    try:
      blob.reload()
      return json.loads(blob.download_as_text(if_generation_match=blob.generation)), blob.generation
    except (gexc.NotFound, gexc.PreconditionFailed):
      return None, None

  def try_claim(self, task_id: str, worker: str, stale_gen: int | None) -> int | None:
    """Atomically creates (or takes over a stale) lock; returns its generation, or None if another worker won."""
    blob = self._blob(f"locks/{task_id}.json")
    body = json.dumps({"worker": worker, "host": socket.gethostname(), "claimed": _now(), "heartbeat": time.time()})
    try:
      if stale_gen is not None:
        blob.delete(if_generation_match=stale_gen)
      blob.upload_from_string(body, content_type="application/json", if_generation_match=0)
      blob.reload()
      return blob.generation
    except (gexc.PreconditionFailed, gexc.NotFound):
      return None

  def heartbeat(self, task_id: str, worker: str, gen: int) -> int | None:
    blob = self._blob(f"locks/{task_id}.json")
    body = json.dumps({"worker": worker, "host": socket.gethostname(), "heartbeat": time.time(), "updated": _now()})
    try:
      blob.upload_from_string(body, content_type="application/json", if_generation_match=gen)
      blob.reload()
      return blob.generation
    except gexc.PreconditionFailed:
      return None

  def release(self, task_id: str, gen: int | None) -> None:
    try:
      self._blob(f"locks/{task_id}.json").delete(if_generation_match=gen)
    except (gexc.NotFound, gexc.PreconditionFailed):
      pass

  def mark_done(self, task_id: str, record: dict) -> None:
    self._blob(f"done/{task_id}.json").upload_from_string(json.dumps(record, indent=2), content_type="application/json")

  def bump_attempts(self, task_id: str) -> int:
    """Counts interrupted runs of a task (not atomic; only the lock holder calls it)."""
    blob = self._blob(f"attempts/{task_id}.json")
    try:
      n = json.loads(blob.download_as_text()).get("attempts", 0) + 1
    except gexc.NotFound:
      n = 1
    blob.upload_from_string(json.dumps({"attempts": n, "updated": _now()}), content_type="application/json")
    return n

  def upload_file(self, local_path: str, gcs_uri: str) -> None:
    bucket, _, name = gcs_uri.removeprefix("gs://").partition("/")
    self.client.bucket(bucket).blob(name).upload_from_filename(local_path)


def _remote_addrs(args) -> list[str]:
  """host:port of the slice daemons (slice 1, 2, ...), from --remote-slice-addrs or $REMOTE_SLICE1_ADDR."""
  raw = args.remote_slice_addrs or os.environ.get("REMOTE_SLICE1_ADDR", "")
  return [a.strip() for a in raw.split(",") if a.strip()]


def _remote_reachable(args, needed: int | None = None) -> bool:
  """True if the first `needed` slice daemons (default: all) accept TCP connections.

  A daemon may have been preempted or may not be admitted yet.
  """
  addrs = _remote_addrs(args)
  if needed is not None:
    if len(addrs) < needed:
      return False
    addrs = addrs[:needed]
  if not addrs:
    return False
  for addr in addrs:
    host, port = addr.rsplit(":", 1)
    try:
      with socket.create_connection((host, int(port)), timeout=5):
        pass
    except OSError:
      return False
  return True


def _usable_slices(args) -> int:
  """Largest slice count this worker can drive right now: 1 + the number of leading reachable daemons."""
  usable = 1
  for n in range(1, min(args.max_slices, len(_remote_addrs(args)) + 1)):
    if not _remote_reachable(args, needed=n):
      break
    usable = n + 1
  return usable


def _runnable_by_me(task: dict, args, max_slices: int) -> bool:
  slices = task.get("slices", 1)
  if slices > max_slices or slices < args.min_slices:
    return False
  return os.path.isfile(os.path.join(task["tree"], HARNESS_REL))


def _sync_harness(tree: str, code_dir: str) -> None:
  """Copies the latest harness from `code_dir` into `tree`, so harness fixes reach running workers on their next task.

  A failed download keeps the copy from pod start. 2-slice runs ship this file to the daemons, so all slices match.
  """
  if not code_dir:
    return
  bucket, _, blob = f"{code_dir.rstrip('/')}/{os.path.basename(HARNESS_REL)}".removeprefix("gs://").partition("/")
  try:
    data = storage.Client().bucket(bucket).blob(blob).download_as_bytes()
    with open(os.path.join(tree, HARNESS_REL), "wb") as f:
      f.write(data)
    _log(f"Synced the harness into {tree} from {code_dir}")
  except Exception as ex:  # pylint: disable=broad-exception-caught
    _log(f"WARN: harness sync from {code_dir} failed ({ex}); using the copy from pod start")


def run_task(q: Queue, task: dict, gen: int, args) -> int:
  """Runs the harness for `task` while refreshing the lock heartbeat; returns the harness exit code."""
  tree = task["tree"]
  _sync_harness(tree, args.code_dir)
  # sys.executable is empty when the interpreter cannot resolve its own path (e.g. under the slice daemon).
  python = sys.executable or shutil.which("python3") or "python3"
  cmd = [python, "-u", os.path.join(tree, HARNESS_REL), *task["args"]]
  csv_local = f"/tmp/task_{task['id']}.csv"
  cmd += ["--csv-report", csv_local]
  if task.get("slices", 1) > 1:
    # One host:port per remote slice; the harness maps slice i to the (i-1)-th address.
    cmd += ["--remote-slice-addrs", *_remote_addrs(args)[: task["slices"] - 1]]
  env = os.environ.copy()
  env.update({k: str(v) for k, v in task.get("env", {}).items()})
  env["PYTHONPATH"] = f"{tree}/src:{tree}"
  _log(f"START {task['id']} on {args.worker}: {' '.join(cmd)}  env={task.get('env', {})}")

  try:
    proc = subprocess.Popen(cmd, env=env, cwd=tree)
  except Exception:
    q.release(task["id"], gen)
    raise
  state = {"gen": gen, "lost": False}

  def _beat():
    while proc.poll() is None:
      time.sleep(args.heartbeat_s)
      if proc.poll() is not None:
        break
      try:
        new_gen = q.heartbeat(task["id"], args.worker, state["gen"])
      except Exception as ex:  # pylint: disable=broad-exception-caught
        # A transient GCS or auth error (e.g. a failed credential refresh) must not end the heartbeat thread: the
        # lock would then go stale and another worker would start a duplicate run of this task.
        _log(f"WARN: heartbeat for {task['id']} failed ({type(ex).__name__}: {ex}); retrying")
        continue
      if new_gen is None:
        # Another worker took the lock over (it thought we were preempted): stop to avoid two runs at once.
        state["lost"] = True
        _log(f"LOST LOCK for {task['id']}; terminating the harness to avoid a duplicate run")
        proc.kill()
        break
      state["gen"] = new_gen

  t = threading.Thread(target=_beat, daemon=True)
  t.start()
  rc = proc.wait()
  t.join(timeout=5)
  if state["lost"]:
    return rc
  if os.path.exists(csv_local) and task.get("csv"):
    try:
      q.upload_file(csv_local, f"{args.results.rstrip('/')}/{task['csv']}")
    except Exception as ex:  # pylint: disable=broad-exception-caught
      _log(f"WARN: CSV upload failed for {task['id']}: {ex}")
  # The harness exits 0 even when phases fail (--allow-step-failure records them per phase). A non-zero exit or a
  # slice-1 daemon that disappeared during the run means the run itself was interrupted (preemption, eviction), so
  # the task is released for another attempt instead of being marked done.
  interrupted = rc != 0 or (task.get("slices", 1) > 1 and not _remote_reachable(args, needed=task["slices"] - 1))
  attempts = q.bump_attempts(task["id"]) if interrupted else 0
  if interrupted and attempts < args.max_attempts:
    q.release(task["id"], state["gen"])
    _log(f"INTERRUPTED {task['id']} on {args.worker} (exit code {rc}, attempt {attempts}/{args.max_attempts}); released")
    return rc
  q.mark_done(
      task["id"],
      {"worker": args.worker, "host": socket.gethostname(), "exit_code": rc, "attempts": attempts, "finished": _now()},
  )
  q.release(task["id"], state["gen"])
  _log(f"DONE {task['id']} on {args.worker} (exit code {rc})")
  return rc


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--queue", required=True, help="gs:// directory holding tasks.json, locks/ and done/")
  p.add_argument("--worker", required=True, help="Worker name recorded in locks and done markers")
  p.add_argument("--max-slices", type=int, default=1, help="Slices this worker can drive (N needs N-1 remote daemons)")
  p.add_argument("--min-slices", type=int, default=1, help="Only claim tasks needing at least this many slices")
  p.add_argument(
      "--remote-slice-addrs",
      default=None,
      help="Comma-separated host:port of slice daemons 1..N-1 (default: $REMOTE_SLICE1_ADDR)",
  )
  p.add_argument("--results", default=DEFAULT_RESULTS, help="gs:// directory for per-task CSVs")
  p.add_argument(
      "--code-dir",
      default=DEFAULT_CODE_DIR,
      help="gs:// directory whose benchmark_ckpt_load_time.py is copied into the task's tree before each task "
      "(empty: keep the copy from pod start)",
  )
  p.add_argument("--max-tasks", type=int, default=0, help="Exit with code 2 after this many tasks (0 = no limit)")
  p.add_argument("--no-wait", action="store_true", help="Exit with code 3 instead of polling when nothing is claimable")
  p.add_argument("--prefer-max-slices", action="store_true", help="Claim tasks needing --max-slices before smaller ones")
  p.add_argument("--poll-s", type=int, default=120)
  p.add_argument("--heartbeat-s", type=int, default=60)
  p.add_argument("--stale-after-s", type=int, default=1200)
  p.add_argument("--max-attempts", type=int, default=3, help="Interrupted runs of a task before it is marked done")
  args = p.parse_args()

  q = Queue(args.queue)
  ran = 0
  while True:
    tasks = q.tasks()
    done = q.done_ids()
    pending = [t for t in tasks if t["id"] not in done]
    if not [t for t in pending if _runnable_by_me(t, args, args.max_slices)]:
      _log(f"No remaining task that {args.worker} can run ({len(pending)} pending for other workers). Exiting.")
      return 0
    # 2-slice tasks need a live slice-1 daemon right now; otherwise only single-slice tasks are candidates.
    max_slices = _usable_slices(args) if args.max_slices > 1 else 1
    mine = [t for t in pending if _runnable_by_me(t, args, max_slices)]
    if args.prefer_max_slices:
      mine.sort(key=lambda t: 0 if t.get("slices", 1) == max_slices else 1)  # stable: keeps file order otherwise

    claimed = None
    for task in mine:
      if any(dep not in done for dep in task.get("depends_on", [])):
        continue
      lock, lock_gen = q.lock(task["id"])
      stale_gen = None
      if lock is not None:
        age = time.time() - float(lock.get("heartbeat", 0))
        if age < args.stale_after_s:
          continue
        _log(f"Taking over stale lock of {task['id']} from {lock.get('worker')} (heartbeat {age:.0f}s old)")
        stale_gen = lock_gen
      gen = q.try_claim(task["id"], args.worker, stale_gen)
      if gen is not None:
        claimed = (task, gen)
        break

    if claimed is None:
      if args.no_wait:
        _log("Nothing claimable right now (dependencies pending or tasks locked by live workers).")
        return 3
      time.sleep(args.poll_s)
      continue

    run_task(q, claimed[0], claimed[1], args)
    ran += 1
    if args.max_tasks and ran >= args.max_tasks:
      return 2


if __name__ == "__main__":
  sys.exit(main())
