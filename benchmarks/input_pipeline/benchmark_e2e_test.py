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

"""Hermetic end-to-end smoke test of the benchmark against `InstrumentedGCSServer`.

It runs the real MaxText Grain pipeline in separate host processes against
per-host GCS emulators, with `google-cloud-cpp` client tracing enabled. It then
checks that:

- Request counts parsed from client traces equal the emulator's ground truth.
- BP-1 startup counting modes reduce measured init GETs.
- All modes yield bitwise-identical per-host sample order (SHA-256 parity).
- No hardcoded QPS or byte-amplification constants remain in the harness.
"""

import os
import pickle
import re
import shutil
import tempfile
import unittest

from absl.testing import absltest

from benchmarks.input_pipeline import qps_runner

_HARNESS_DIR = os.path.dirname(os.path.abspath(qps_runner.__file__))
_COMMON = [
    "--backend=emulator",
    "--num_shards=8",
    "--records_per_shard=64",
    "--host_counts=2",
    "--grain_prefetch_buffer_size=16",
    "--grain_num_threads=2",
    "--per_host_batch_size=2",
    "--max_target_length=256",
    "--warmup_steps=1",
    "--steps=3",
    "--host_timeout_s=600",
]


class BenchmarkEndToEndTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    cls.data_dir = tempfile.mkdtemp(prefix="qps_e2e_data_")

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.data_dir, ignore_errors=True)
    super().tearDownClass()

  def _run(self, extra):
    args = qps_runner.create_parser().parse_args(_COMMON + [f"--data_dir={self.data_dir}"] + extra)
    report = qps_runner.run_benchmark(args)
    for r in report["runs"]:
      self.assertNotIn("error", r, msg=r.get("error"))
    return report["runs"]

  def test_startup_counting_modes_trace_matches_emulator_and_parity(self):
    runs = self._run(
        [
            "--startup_counting=every_host,sharded,manifest",
            "--grain_worker_count=0",
            "--emulator_client_trace",
            "--measure_mode=trace",
        ]
    )
    by_mode = {r["config"]["startup_counting"]: r for r in runs}
    self.assertEqual(set(by_mode), {"every_host", "sharded", "manifest"})
    for r in runs:
      emu, trace = r["requests"]["emulator"], r["requests"]["client_trace"]
      # Every GET the server saw was also observed by the per-process client trace parser.
      self.assertGreater(emu["total_gets"], 0)
      self.assertEqual(trace["total_gets"], emu["total_gets"], msg=r["label"])
      self.assertEqual(trace["by_range_kind"], emu["by_range_kind"], msg=r["label"])
      self.assertEqual(trace["by_source"], {"raw-client": emu["total_gets"]})
      self.assertIn("206", trace["by_status"])
      self.assertEqual(len({h["pid"] for h in r["passes"]["trace"]["hosts"]}), 2)
      self.assertEqual(r["passes"]["trace"]["client_trace_stats"]["unmatched_http_requests"], 0)
    init = {m: by_mode[m]["requests"]["emulator"]["per_phase"]["init"]["gets"] for m in by_mode}
    # every_host: each of 2 hosts opens all 8 shards to count records.
    self.assertGreaterEqual(init["every_host"], 2 * 8)
    self.assertLess(init["sharded"], init["every_host"])
    self.assertLessEqual(init["manifest"], init["sharded"])
    statuses = [r["parity_vs_baseline"]["status"] for r in runs]
    self.assertEqual(statuses, ["BASELINE", "MATCH", "MATCH"])
    self.assertEqual(runs[0]["distinct_streams"], 2)

  def test_injected_503_counted_by_client_trace_and_emulator(self):
    runs = self._run(
        [
            "--host_counts=1",
            "--emulator_fail_every_n=5",
            "--emulator_fail_status=503",
            "--emulator_client_trace",
            "--measure_mode=trace",
        ]
    )
    emu, trace = runs[0]["requests"]["emulator"], runs[0]["requests"]["client_trace"]
    self.assertGreater(emu["by_outcome"].get("unavailable", 0), 0)
    self.assertEqual(trace["by_outcome"].get("unavailable", 0), emu["by_outcome"]["unavailable"])
    self.assertEqual(trace["total_gets"], emu["total_gets"])

  def test_mp_prefetch_workers_and_per_replica_duplicates(self):
    from maxtext.input_pipeline import tokenizer  # pylint: disable=import-outside-toplevel

    tok_path = qps_runner._default_tokenizer("tokenizer.default")  # pylint: disable=protected-access
    tok = tokenizer.SentencePieceTokenizer(tok_path, True, False)
    try:
      pickle.loads(pickle.dumps(tok)).encode("probe")
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"grain_worker_count>0 needs a picklable SentencePieceTokenizer (MaxText change): {e!r}")
    runs = self._run(
        [
            "--grain_worker_count=1",
            "--data_sharding=per_host,per_replica",
            "--hosts_per_replica=2",
            "--emulator_client_trace",
            "--measure_mode=paired",
        ]
    )
    self.assertEqual(len(runs), 2)
    per_host, per_replica = runs[0], runs[1]
    trace_pass = per_host["passes"]["trace"]
    # Host processes plus their mp_prefetch worker processes each wrote their own trace file.
    self.assertGreaterEqual(trace_pass["client_trace_stats"]["files"], 3)
    self.assertEqual(per_host["requests"]["client_trace"]["total_gets"], per_host["requests"]["emulator"]["total_gets"])
    self.assertTrue(per_host["digests_identical_across_passes"])
    self.assertEqual(per_host["distinct_streams"], 2)
    # Same unique examples per step; per_replica loads the full replica batch on both hosts.
    self.assertEqual(per_replica["unique_examples_per_step"], per_host["unique_examples_per_step"])
    self.assertIn("steady_gets_per_unique_example", per_replica["requests"]["emulator"])
    # Both hosts in the replica read the same slice: identical streams.
    self.assertEqual(per_replica["distinct_streams"], 1)
    self.assertTrue(per_replica["passes"]["timing"]["parity"]["hosts_sharing_a_slice_identical"])
    self.assertFalse(per_replica["timing_from_traced_pass"])

  def test_no_hardcoded_qps_or_byte_formulas(self):
    banned = [r"491\.7", r"113\.8", r"\b17\.5\b", r"\b3891\b", r"\b6656\b", r"\b52224\b", r"\b526336\b", r"read_amp\b"]
    banned += ["BlockShuffle", "GCSBlockCached", "gcs_block_cache", "compute_file_sharding"]
    for name in os.listdir(_HARNESS_DIR):
      if name.endswith(".py") and not name.endswith("_test.py"):
        with open(os.path.join(_HARNESS_DIR, name), "r", encoding="utf-8") as f:
          src = f.read()
        for pat in banned:
          self.assertIsNone(re.search(pat, src), msg=f"{name} contains banned pattern {pat}")


if __name__ == "__main__":
  absltest.main()
