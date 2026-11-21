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

"""Tests for the multi-process host runner, sharded-count exchange, sweep expansion, and parity checks."""

import json
import os
import signal
import subprocess
import tempfile
import time
import unittest

from absl.testing import absltest

import numpy as np

from benchmarks.input_pipeline import gcs_trace
from benchmarks.input_pipeline import qps_runner


def _fake_host(task, conn, barrier):
  """Picklable host target: exchanges shard counts and hashes a per-slice synthetic stream."""
  try:
    barrier.wait(timeout=120)
    combined = None
    if task["mode"] == "orphan" and task["host"] == 0:
      # Mimic an OOM-killed host that leaves a Grain worker behind.
      child = subprocess.Popen(["/bin/sleep", "300"])  # pylint: disable=consider-using-with
      with open(task["pid_file"], "w", encoding="utf-8") as f:
        f.write(str(child.pid))
      os.kill(os.getpid(), signal.SIGKILL)
    if task["mode"] == "slow_teardown":
      conn.send(("result", {"host": task["host"], "pid": os.getpid(), "digest": "", "combined": None}))
      time.sleep(300)  # Mimics an mp_prefetch iterator close() that never returns.
    if task["mode"] == "fail":
      if task["host"] == 1:
        raise ValueError("boom from host 1")
      barrier.wait(timeout=120)  # Broken by host 1's abort; must not hang.
    if task["mode"] == "exchange":
      local = [1000 + i for i in range(task["host"], task["num_files"], task["host_count"])]
      conn.send(("shard_counts", {"num_files": task["num_files"], "digest": 42, "counts": local}))
      kind, combined = conn.recv()
      if kind != "all_shard_counts":
        raise RuntimeError(combined)
    digest = gcs_trace.RecordOrderDigest()
    rng = np.random.default_rng(task["slice"])
    for _ in range(3):
      digest.update({"inputs": rng.integers(0, 50, (2, 4))})
    conn.send(("result", {"host": task["host"], "pid": os.getpid(), "digest": digest.hexdigest(), "combined": combined}))
  except Exception as e:  # pylint: disable=broad-exception-caught
    try:
      barrier.abort()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    conn.send(("error", f"host {task['host']}: {e!r}"))
  finally:
    conn.close()


def _is_running(pid: int) -> bool:
  """True if `pid` exists and is not a zombie."""
  try:
    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
      stat = f.read()
  except OSError:
    return False
  return stat[stat.rfind(")") + 2] != "Z"


class RunHostsTest(unittest.TestCase):

  def test_separate_processes_and_sharded_count_exchange(self):
    tasks = [{"host": h, "host_count": 4, "num_files": 10, "mode": "exchange", "slice": h} for h in range(4)]
    results = qps_runner.run_hosts(_fake_host, tasks, timeout_s=300)
    pids = [r["pid"] for r in results]
    self.assertEqual(len(set(pids)), 4)
    self.assertNotIn(os.getpid(), pids)
    for r in results:
      self.assertEqual(r["combined"], [1000 + i for i in range(10)])
    self.assertEqual(len({r["digest"] for r in results}), 4)

  def test_host_error_propagates_without_hanging(self):
    tasks = [{"host": h, "host_count": 3, "num_files": 0, "mode": "fail", "slice": h} for h in range(3)]
    start = time.time()
    with self.assertRaisesRegex(RuntimeError, "boom from host 1"):
      qps_runner.run_hosts(_fake_host, tasks, timeout_s=300)
    self.assertLess(time.time() - start, 120)

  def test_orphaned_grandchild_is_reaped_after_host_dies(self):
    with tempfile.TemporaryDirectory() as d:
      pid_file = os.path.join(d, "grandchild.pid")
      tasks = [
          {"host": h, "host_count": 2, "num_files": 0, "mode": "orphan", "slice": h, "pid_file": pid_file}
          for h in range(2)
      ]
      with self.assertRaisesRegex(RuntimeError, "host 0: exited without a result"):
        qps_runner.run_hosts(_fake_host, tasks, timeout_s=300)
      with open(pid_file, "r", encoding="utf-8") as f:
        grandchild = int(f.read())
    deadline = time.time() + 10
    while _is_running(grandchild) and time.time() < deadline:
      time.sleep(0.1)
    self.assertFalse(_is_running(grandchild), msg=f"orphan {grandchild} survived run_hosts teardown")

  def test_results_returned_without_waiting_for_slow_teardown(self):
    tasks = [{"host": h, "host_count": 2, "num_files": 0, "mode": "slow_teardown", "slice": h} for h in range(2)]
    start = time.time()
    results = qps_runner.run_hosts(_fake_host, tasks, timeout_s=300)
    self.assertLess(time.time() - start, 90)
    for r in results:
      deadline = time.time() + 10
      while _is_running(r["pid"]) and time.time() < deadline:
        time.sleep(0.1)
      self.assertFalse(_is_running(r["pid"]))

  def test_digest_parity_is_deterministic_across_runs(self):
    run = qps_runner.RunConfig(host_count=4, data_sharding="per_replica", hosts_per_replica=2)
    tasks = [
        {"host": h, "host_count": 4, "num_files": 0, "mode": "plain", "slice": qps_runner.dataloading_shard(run, h)[0]}
        for h in range(4)
    ]
    first = qps_runner.run_hosts(_fake_host, tasks, timeout_s=300)
    second = qps_runner.run_hosts(_fake_host, tasks, timeout_s=300)
    self.assertEqual([r["digest"] for r in first], [r["digest"] for r in second])
    parity = qps_runner._digest_parity_within_run(run, first)  # pylint: disable=protected-access
    self.assertEqual(parity["distinct_streams"], 2)
    self.assertTrue(parity["hosts_sharing_a_slice_identical"])


class HelpersTest(unittest.TestCase):

  def test_dataloading_shard(self):
    run = qps_runner.RunConfig(host_count=16, hosts_per_replica=8)
    self.assertEqual([qps_runner.dataloading_shard(run, h) for h in (0, 7, 15)], [(0, 16), (7, 16), (15, 16)])
    rep = qps_runner.RunConfig(host_count=16, hosts_per_replica=8, data_sharding="per_replica")
    self.assertEqual([qps_runner.dataloading_shard(rep, h) for h in (0, 7, 8, 15)], [(0, 2), (0, 2), (1, 2), (1, 2)])
    with self.assertRaises(ValueError):
      qps_runner.dataloading_shard(qps_runner.RunConfig(host_count=4, data_sharding="per_replica"), 0)

  def test_startup_jitter_is_deterministic_and_bounded(self):
    self.assertEqual(qps_runner.startup_jitter_s(0.0, 0, 3), 0.0)
    vals = [qps_runner.startup_jitter_s(10.0, 1234, h) for h in range(64)]
    self.assertEqual(vals, [qps_runner.startup_jitter_s(10.0, 1234, h) for h in range(64)])
    self.assertTrue(all(0.0 <= v < 10.0 for v in vals))
    self.assertGreater(len(set(vals)), 60)
    self.assertNotEqual(vals, [qps_runner.startup_jitter_s(10.0, 99, h) for h in range(64)])

  def test_parse_reader_options(self):
    self.assertIsNone(qps_runner.parse_reader_options(""))
    self.assertEqual(
        qps_runner.parse_reader_options("footer_prefetch_bytes:8M, readahead_buffer_size:0"),
        {"footer_prefetch_bytes": "8M", "readahead_buffer_size": "0"},
    )
    with self.assertRaises(ValueError):
      qps_runner.parse_reader_options("footer_prefetch_bytes")

  def test_combine_shard_counts(self):
    # Host h counted files[h::3] of 5 sorted shards.
    msgs = [{"num_files": 5, "digest": 1, "counts": c} for c in ([10, 13], [11, 14], [12])]
    self.assertEqual(qps_runner.combine_shard_counts(msgs), [10, 11, 12, 13, 14])
    with self.assertRaisesRegex(RuntimeError, "mismatch"):
      qps_runner.combine_shard_counts([msgs[0], dict(msgs[1], digest=2), msgs[2]])
    with self.assertRaisesRegex(RuntimeError, "expected"):
      qps_runner.combine_shard_counts([msgs[0], dict(msgs[1], counts=[1]), msgs[2]])

  def test_expand_sweep_and_labels(self):
    args = qps_runner.create_parser().parse_args(
        [
            "--backend=gcs",
            "--train_files=gs://a/x-*|/mnt/fuse/x-*",
            "--host_counts=1,8",
            "--grain_worker_count=0,4",
            "--grain_prefetch_buffer_size=16,500",
            "--startup_counting=every_host,manifest",
            "--grain_reader_options=|footer_prefetch_bytes:8M",
            "--data_sharding=per_host,per_replica",
        ]
    )
    runs = qps_runner.expand_sweep(args)
    self.assertEqual(len(runs), 2 * 2 * 2 * 2 * 2 * 2 * 2)
    self.assertEqual(len({r.label() for r in runs}), len(runs) // 2)  # Labels omit the storage path.
    self.assertIn("footer_prefetch_bytes:8M", {r.grain_reader_options for r in runs})
    # Storage tier and worker count are not part of the parity key.
    keys = {r.parity_key() for r in runs}
    self.assertEqual(len(keys), 2 * 2 * 2)  # host_count x data_sharding x grain_worker_count
    with self.assertRaises(ValueError):
      qps_runner.expand_sweep(qps_runner.create_parser().parse_args(["--startup_counting=block_shuffle"]))

  def test_per_replica_full_batch_keeps_global_batch(self):
    per_host = qps_runner.RunConfig(host_count=16, per_host_batch_size=4)
    rep = qps_runner.RunConfig(host_count=16, per_host_batch_size=4, data_sharding="per_replica", hosts_per_replica=8)
    self.assertEqual(per_host.host_batch_size, 4)
    self.assertEqual(rep.host_batch_size, 32)
    replicas = qps_runner.dataloading_shard(rep, 0)[1]
    # Same unique examples per step, 8x the loaded (read) examples.
    self.assertEqual(replicas * rep.host_batch_size, per_host.host_count * per_host.host_batch_size)
    self.assertEqual(rep.host_count * rep.host_batch_size, 8 * per_host.host_count * per_host.host_batch_size)
    self.assertEqual(qps_runner.RunConfig(data_sharding="per_replica", per_replica_full_batch=False).host_batch_size, 4)
    self.assertIn("B32", rep.label())
    self.assertEqual(qps_runner.unique_examples_per_step(rep), qps_runner.unique_examples_per_step(per_host))
    self.assertEqual(qps_runner.unique_examples_per_step(per_host), 64)
    self.assertNotEqual(
        qps_runner.RunConfig(grain_worker_count=0).parity_key(), qps_runner.RunConfig(grain_worker_count=2).parity_key()
    )

  def test_passes_for(self):
    p = qps_runner.create_parser()
    self.assertEqual(qps_runner.passes_for(qps_runner.RunConfig(backend="gcs"), p.parse_args([])), ("timing", "trace"))
    self.assertEqual(
        qps_runner.passes_for(qps_runner.RunConfig(backend="gcs"), p.parse_args(["--measure_mode=trace"])), ("trace",)
    )
    self.assertEqual(qps_runner.passes_for(qps_runner.RunConfig(backend="emulator"), p.parse_args([])), ("measure",))
    self.assertEqual(
        qps_runner.passes_for(qps_runner.RunConfig(backend="emulator"), p.parse_args(["--emulator_client_trace"])),
        ("timing", "trace"),
    )
    self.assertEqual(qps_runner.passes_for(qps_runner.RunConfig(backend="posix"), p.parse_args([])), ("measure",))

  def test_check_parity_and_markdown(self):
    def res(label, digests, key=("k",)):
      return {
          "label": label,
          "parity_key": key,
          "digests": digests,
          "distinct_streams": len(set(digests)),
          "timing": {
              "init_s_max": 1.0,
              "first_batch_s_max": 0.5,
              "step_p50_s": 0.01,
              "step_p90_s": 0.02,
              "step_p99_s": 0.03,
              "examples_per_s_per_host": 100.0,
          },
          "requests": {},
          "nic_rx_bytes": 0,
      }

    results = [res("a", ["x", "y"]), res("b", ["x", "y"]), res("c", ["x", "z"]), res("d", ["q"], key=("other",))]
    results.append({"label": "e", "parity_key": ("k",), "error": "failed"})
    qps_runner.check_parity(results)
    self.assertEqual(
        [r.get("parity_vs_baseline", {}).get("status") for r in results],
        ["BASELINE", "MATCH", "MISMATCH", "BASELINE", None],
    )
    self.assertEqual(results[2]["parity_vs_baseline"]["mismatched_hosts"], [1])
    md = qps_runner.markdown_report(results)
    self.assertIn("| c | a | MISMATCH |", md)
    self.assertIn("| e | ERROR |", md)

  def test_summarize_timing(self):
    run = qps_runner.RunConfig(host_count=2, steps=4, per_host_batch_size=8)
    hosts = []
    for h in range(2):
      hosts.append(
          {
              "step_latencies_s": [0.1 * (h + 1)] * 4,
              "marks": {
                  "init_start": 0.0,
                  "init_end": 2.0 + h,
                  "first_batch_end": 3.0 + h,
                  "warmup_end": 4.0,
                  "steady_end": 5.0,
              },
              "packed_docs_steady": 40,
          }
      )
    t = qps_runner.summarize_timing(run, hosts)
    self.assertEqual(t["init_s_max"], 3.0)
    self.assertEqual(t["first_batch_s_max"], 1.0)
    self.assertEqual(t["cluster_step_p50_s"], 0.2)
    self.assertAlmostEqual(t["data_wait_examples_per_s_per_host"], (32 / 0.4 + 32 / 0.8) / 2, places=1)
    self.assertAlmostEqual(t["examples_per_s_per_host"], 32 / 1.0, places=1)  # steady wall = 5.0 - 4.0 s.
    self.assertEqual(t["packed_docs_per_step_per_host"], 10.0)

  def test_make_trace_dir_creates_missing_nested_root(self):
    with tempfile.TemporaryDirectory() as d:
      root = os.path.join(d, "does", "not", "exist")
      first, second = qps_runner.make_trace_dir(root), qps_runner.make_trace_dir(root)
      self.assertTrue(os.path.isdir(first))
      self.assertNotEqual(first, second)
      self.assertEqual(os.path.dirname(first), root)
    self.assertTrue(os.path.isdir(tmp := qps_runner.make_trace_dir("")))
    os.rmdir(tmp)

  def test_manifest_file_instructions(self):
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "m.json")
      with open(path, "w", encoding="utf-8") as f:
        json.dump({"gs://b/t-00001": 7, "gs://b/t-00000": 5, "gs://b/v-00000": 9, "gs://b/t-x/deeper": 3}, f)
      fi = qps_runner.load_manifest_file_instructions(path, "gs://b/t-*")
    ((pattern, instructions),) = fi.items()
    self.assertEqual(pattern, "gs://b/t-*")
    self.assertEqual(
        [(i.filename, i.skip, i.take, i.examples_in_shard) for i in instructions],
        [("gs://b/t-00000", 0, 5, 5), ("gs://b/t-00001", 0, 7, 7)],
    )


if __name__ == "__main__":
  absltest.main()
