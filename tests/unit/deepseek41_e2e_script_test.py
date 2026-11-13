# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for preemption-tolerant DeepSeek-V4.1 E2E verification harness.

Verifies:
1. Resume skips banked passed/deferred samples and runs only unbanked samples.
2. Resume immediately fails when encountering a banked failed sample or audit (no rerun).
3. Stale, partial, or corrupt JSON files are rejected and re-run cleanly.
4. Assembled stage summary gates correctly and records per-sample provenance.
5. Banked sample with mismatched commit_sha, overlay_sha256, or config_hash is rejected on resume.
6. Per-attempt GCS startup barrier and any-worker rank-0 summary upload.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils import deepseek_v41_frozen_topk
from tests.utils import deepseek_v41_preemption_banking as banking
from tests.utils import forward_pass_logit_checker
from tests.utils import masked_prefill_logits_checker

pytestmark = pytest.mark.cpu_only

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/2_test_deepseek41_mm.sh"


class DeepSeek41E2EPreemptionHarnessTest(unittest.TestCase):
  """Unit tests for preemption-tolerant banking and E2E script contracts."""

  def setUp(self):
    super().setUp()
    self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix="ds41_e2e_test_")  # pylint: disable=consider-using-with
    self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)
    self.samples_dir = self.tmp_dir / "samples"
    self.samples_dir.mkdir(parents=True, exist_ok=True)
    self.audit_json = self.tmp_dir / "r2_audit_summary.json"

  def tearDown(self):
    self.tmp_dir_obj.cleanup()
    super().tearDown()

  def test_1_resume_skips_banked_passed_and_deferred_samples(self):
    """Test 1: Resume skips banked passed (sample_0, 1) and deferred (sample_3) samples."""
    banking.write_banked_sample(
        sample_id="sample_0",
        stage_name="r2",
        status=banking.STATUS_PASSED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.001, "mean_kl": 0.0001, "top10_overlap": 10},
        attempt_id="attempt_1",
        pod="pod-1",
    )
    banking.write_banked_sample(
        sample_id="sample_1",
        stage_name="r2",
        status=banking.STATUS_PASSED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.002, "mean_kl": 0.0002, "top10_overlap": 10},
        attempt_id="attempt_1",
        pod="pod-1",
    )
    banking.write_banked_sample(
        sample_id="sample_3",
        stage_name="r2",
        status=banking.STATUS_DEFERRED,
        local_samples_dir=self.samples_dir,
        metrics={},
        attempt_id="attempt_1",
        pod="pod-1",
    )

    sample_list = ["sample_0", "sample_1", "sample_2", "sample_3", "L1"]
    executed_samples = []

    def mock_forward_eval(sid: str) -> dict[str, float]:
      executed_samples.append(sid)
      return {"max_kl": 0.005, "mean_kl": 0.0005, "top10_overlap": 10}

    results = []
    for sid in sample_list:
      status, banked, _ = banking.check_banked_sample(sid, "r2", self.samples_dir)
      if status in (banking.STATUS_PASSED, banking.STATUS_DEFERRED) and banked is not None:
        results.append(banked)
        continue
      metrics = mock_forward_eval(sid)
      saved = banking.write_banked_sample(
          sample_id=sid,
          stage_name="r2",
          status=banking.STATUS_PASSED,
          local_samples_dir=self.samples_dir,
          metrics=metrics,
          attempt_id="attempt_2",
          pod="pod-2",
      )
      results.append(saved)

    # sample_0, sample_1, and sample_3 MUST be skipped; only sample_2 and L1 run
    self.assertEqual(executed_samples, ["sample_2", "L1"])
    self.assertEqual(len(results), 5)
    self.assertEqual([r["status"] for r in results], ["passed", "passed", "passed", "deferred", "passed"])
    self.assertEqual(
        [r["attempt_id"] for r in results], ["attempt_1", "attempt_1", "attempt_2", "attempt_1", "attempt_2"]
    )

  def test_2_resume_fails_immediately_on_banked_failed_sample_or_audit(self):
    """Test 2: Resume NEVER re-runs a failed sample; continues remaining samples and fails at end."""
    banking.write_banked_sample(
        sample_id="sample_1",
        stage_name="r2",
        status=banking.STATUS_FAILED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.5, "mean_kl": 0.2, "top10_overlap": 2},
        attempt_id="attempt_1",
        pod="pod-fail",
    )

    mock_eval = mock.MagicMock()
    failed_samples = []
    for sid in ["sample_0", "sample_1", "sample_2"]:
      status, _, _ = banking.check_banked_sample(sid, "r2", self.samples_dir)
      if status == banking.STATUS_FAILED:
        failed_samples.append(sid)
        continue
      mock_eval(sid)
    self.assertEqual(failed_samples, ["sample_1"])
    mock_eval.assert_has_calls([mock.call("sample_0"), mock.call("sample_2")])

    # Banked failed audit
    banking.write_banked_audit_summary(
        local_path=self.audit_json,
        audit_result={"passed": False, "total_bit_mismatches": 5, "total_leaf_failures": 1, "shard_count": 83088},
        attempt_id="attempt_audit_fail",
        pod="pod-audit",
    )
    status, audit_data, _ = banking.check_banked_audit_summary(self.audit_json, min_shards=83088)
    self.assertEqual(status, banking.STATUS_FAILED)
    self.assertFalse(audit_data["passed"])

  def test_3_stale_partial_corrupt_json_rejected_and_rerun(self):
    """Test 3: Corrupt or partial JSON files are rejected and re-run cleanly."""
    corrupt_sample_file = self.samples_dir / "r2_sample_0.json"
    corrupt_sample_file.write_text('{"sample_id": "sample_0", "status": "pass', encoding="utf-8")

    status, banked, reason = banking.check_banked_sample("sample_0", "r2", self.samples_dir)
    self.assertEqual(status, banking.STATUS_CORRUPT)
    self.assertIsNone(banked)
    self.assertIn("JSON parse error", reason)

    # Re-running and writing over corrupt file succeeds
    banking.write_banked_sample(
        sample_id="sample_0",
        stage_name="r2",
        status=banking.STATUS_PASSED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.001, "mean_kl": 0.0001, "top10_overlap": 10},
    )
    status, banked, _ = banking.check_banked_sample("sample_0", "r2", self.samples_dir)
    self.assertEqual(status, banking.STATUS_PASSED)
    self.assertIsNotNone(banked)

    # Missing required schema fields in audit summary
    self.audit_json.write_text(json.dumps({"passed": True, "shard_count": 100}), encoding="utf-8")
    status, audit_data, reason = banking.check_banked_audit_summary(self.audit_json, min_shards=83088)
    self.assertEqual(status, banking.STATUS_CORRUPT)
    self.assertIsNone(audit_data)
    self.assertIn("Missing required key", reason)

  def test_4_assembled_stage_summary_gating_and_provenance(self):
    """Test 4: Assembled stage summary gates correctly and produces provenance table."""
    sample_reports = [
        {
            "name": "sample_0",
            "status": "passed",
            "attempt_id": "att_1",
            "pod": "pod-1",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T01:00:00Z",
            "passed": True,
        },
        {
            "name": "sample_1",
            "status": "passed",
            "attempt_id": "att_1",
            "pod": "pod-1",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T01:02:00Z",
            "passed": True,
        },
        {
            "name": "sample_2",
            "status": "passed",
            "attempt_id": "att_2",
            "pod": "pod-2",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T02:00:00Z",
            "passed": True,
        },
        {
            "name": "sample_3",
            "status": "deferred",
            "attempt_id": "att_1",
            "pod": "pod-1",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T01:04:00Z",
        },
        {
            "name": "L1",
            "status": "passed",
            "attempt_id": "att_2",
            "pod": "pod-2",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T02:05:00Z",
            "passed": True,
        },
        {
            "name": "L2",
            "status": "passed",
            "attempt_id": "att_2",
            "pod": "pod-2",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T02:10:00Z",
            "passed": True,
        },
        {
            "name": "L3",
            "status": "passed",
            "attempt_id": "att_3",
            "pod": "pod-3",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T03:00:00Z",
            "passed": True,
        },
        {
            "name": "L4",
            "status": "passed",
            "attempt_id": "att_3",
            "pod": "pod-3",
            "commit_sha": "20c1af6a",
            "overlay_sha256": "ovl1",
            "config_hash": "cfg1",
            "timestamp_utc": "2026-11-06T03:05:00Z",
            "passed": True,
        },
    ]

    prov_list = banking.assemble_stage_provenance(sample_reports)
    self.assertEqual(len(prov_list), 8)
    self.assertEqual(prov_list[0]["sample_id"], "sample_0")
    self.assertEqual(prov_list[3]["status"], "deferred")

    table_str = banking.format_provenance_table(prov_list)
    self.assertIn("Sample ID", table_str)
    self.assertIn("Commit SHA", table_str)
    self.assertIn("Config Hash", table_str)
    self.assertIn("sample_0", table_str)
    self.assertIn("att_3", table_str)

    # Verification of gating: 7/7 gated samples pass with sample_3 deferred
    gated_samples = [s for s in sample_reports if s.get("status") != "deferred"]
    all_gated_passed = all(s.get("passed", False) for s in gated_samples)
    self.assertTrue(all_gated_passed)

  def test_5_banked_sample_provenance_mismatch_rejected_and_rerun(self):
    """Test 5 (Controller Condition 3): Mismatched commit_sha, overlay_sha256, or config_hash is rejected."""
    # Write a banked sample with old commit_sha
    banking.write_banked_sample(
        sample_id="sample_0",
        stage_name="r2",
        status=banking.STATUS_PASSED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.001, "mean_kl": 0.0001, "top10_overlap": 10},
        attempt_id="att_old",
        pod="pod-1",
        commit_sha="old_commit_1111",
        overlay_sha256="overlay_hash_aaa",
        config_hash="cfg_hash_xxx",
    )

    # Check with matching expected provenance -> PASS
    status, _, _ = banking.check_banked_sample(
        "sample_0",
        "r2",
        self.samples_dir,
        expected_commit_sha="old_commit_1111",
        expected_overlay_sha256="overlay_hash_aaa",
        expected_config_hash="cfg_hash_xxx",
    )
    self.assertEqual(status, banking.STATUS_PASSED)

    # Check with mismatched commit_sha -> rejected (STATUS_CORRUPT / re-run)
    status_mismatch_commit, _, reason_commit = banking.check_banked_sample(
        "sample_0",
        "r2",
        self.samples_dir,
        expected_commit_sha="new_commit_2222",
        expected_overlay_sha256="overlay_hash_aaa",
        expected_config_hash="cfg_hash_xxx",
    )
    self.assertEqual(status_mismatch_commit, banking.STATUS_CORRUPT)
    self.assertIn("commit_sha mismatch", reason_commit)

    # Check with mismatched overlay_sha256 -> rejected
    status_mismatch_ovl, _, reason_ovl = banking.check_banked_sample(
        "sample_0",
        "r2",
        self.samples_dir,
        expected_commit_sha="old_commit_1111",
        expected_overlay_sha256="overlay_hash_bbb",
        expected_config_hash="cfg_hash_xxx",
    )
    self.assertEqual(status_mismatch_ovl, banking.STATUS_CORRUPT)
    self.assertIn("overlay_sha256 mismatch", reason_ovl)

    # Check with mismatched config_hash -> rejected
    status_mismatch_cfg, _, reason_cfg = banking.check_banked_sample(
        "sample_0",
        "r2",
        self.samples_dir,
        expected_commit_sha="old_commit_1111",
        expected_overlay_sha256="overlay_hash_aaa",
        expected_config_hash="cfg_hash_yyy",
    )
    self.assertEqual(status_mismatch_cfg, banking.STATUS_CORRUPT)
    self.assertIn("config_hash mismatch", reason_cfg)

  def test_6_script_startup_barrier_and_any_worker_rank0_upload(self):
    """Test 6: Existing tests for any-worker rank-0 upload and per-attempt GCS startup barrier pass."""
    ready_dir = self.tmp_dir / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    now_epoch = int(time.time())

    def run_barrier(  # pylint: disable=too-many-positional-arguments
        num_workers: int = 4,
        worker_id: int = 0,
        run_attempt_id: str = "",
        barrier_start_epoch: int | None = None,
        max_skew_sec: int = 180,
        timeout_sec: int = 2,
        poll_sec: int = 1,
    ) -> subprocess.CompletedProcess[str]:
      if barrier_start_epoch is None:
        barrier_start_epoch = int(time.time())
      env = os.environ.copy()
      env["PYTHON_BIN"] = sys.executable
      return subprocess.run(
          [
              "bash",
              "-c",
              f'source "{SCRIPT_PATH.as_posix()}" && wait_for_workers_ready "$@"',
              "--",
              ready_dir.as_posix(),
              str(num_workers),
              str(worker_id),
              run_attempt_id,
              str(barrier_start_epoch),
              str(max_skew_sec),
              str(timeout_sec),
              str(poll_sec),
          ],
          capture_output=True,
          text=True,
          env=env,
          check=False,
      )

    # 1. Stale attempt markers from attempt_old block attempt_new
    for w in range(4):
      (ready_dir / f"worker_{w}.ready").write_text(
          json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_old", "epoch": now_epoch}),
          encoding="utf-8",
      )
    for w in range(3):
      (ready_dir / f"worker_{w}.ready").write_text(
          json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_new", "epoch": now_epoch}),
          encoding="utf-8",
      )
    res1 = run_barrier(num_workers=4, worker_id=0, run_attempt_id="attempt_new", timeout_sec=2, poll_sec=1)
    self.assertNotEqual(res1.returncode, 0, f"Expected barrier failure with stale marker, got 0. stderr={res1.stderr}")

    # 2. When all workers update to attempt_new, barrier passes
    (ready_dir / "worker_3.ready").write_text(
        json.dumps({"worker_id": 3, "pod": "pod-3", "attempt_id": "attempt_new", "epoch": now_epoch}),
        encoding="utf-8",
    )
    res2 = run_barrier(num_workers=4, worker_id=0, run_attempt_id="attempt_new", timeout_sec=5, poll_sec=1)
    self.assertEqual(res2.returncode, 0, f"Expected barrier success, got {res2.returncode}. stderr={res2.stderr}")

  def test_forward_pass_and_prefill_checkers_banking_helpers(self):
    """Exercises --samples_dir and --audit_summary_json helper paths in both checkers."""

    # 1. Test forward_pass_logit_checker audit check helper with None / empty
    status, data, reason = forward_pass_logit_checker.check_banked_audit_summary_for_test(None)
    self.assertEqual(status, banking.STATUS_MISSING)
    self.assertIsNone(data)
    self.assertIn("not provided", reason)

    status, data, reason = forward_pass_logit_checker.check_banked_audit_summary_for_test("")
    self.assertEqual(status, banking.STATUS_MISSING)

    # 2. Test forward_pass_logit_checker audit check helper with real / mock files
    status, data, reason = forward_pass_logit_checker.check_banked_audit_summary_for_test(self.audit_json)
    self.assertEqual(status, banking.STATUS_MISSING)

    # Valid audit
    banking.write_banked_audit_summary(
        local_path=self.audit_json,
        audit_result={"passed": True, "total_bit_mismatches": 0, "total_leaf_failures": 0, "shard_count": 83088},
        attempt_id="att1",
        pod="pod-0",
    )
    status, data, reason = forward_pass_logit_checker.check_banked_audit_summary_for_test(str(self.audit_json))
    self.assertEqual(status, banking.STATUS_PASSED)
    self.assertTrue(data["passed"])

    # 3. Test forward_pass_logit_checker sample check helper with None / empty / valid
    status, data, reason = forward_pass_logit_checker.check_banked_sample_for_test(None, "sample_0")
    self.assertEqual(status, banking.STATUS_MISSING)

    status, data, reason = forward_pass_logit_checker.check_banked_sample_for_test(str(self.samples_dir), "sample_0")
    self.assertEqual(status, banking.STATUS_MISSING)

    # Banked sample
    banking.write_banked_sample(
        sample_id="sample_0",
        stage_name="r2",
        status=banking.STATUS_PASSED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.001, "mean_kl": 0.0001, "top10_overlap": 10},
        attempt_id="att1",
        pod="pod-0",
    )
    status, data, reason = forward_pass_logit_checker.check_banked_sample_for_test(self.samples_dir, "sample_0")
    self.assertEqual(status, banking.STATUS_PASSED)
    self.assertIsNotNone(data)

    # 4. Test masked_prefill_logits_checker sample check helper with None / empty / valid
    status, data, reason = masked_prefill_logits_checker.check_banked_sample_for_test(None, "sample_0")
    self.assertEqual(status, banking.STATUS_MISSING)

    status, data, reason = masked_prefill_logits_checker.check_banked_sample_for_test(self.samples_dir, "sample_0")
    # r3 stage is not yet banked in samples_dir
    self.assertEqual(status, banking.STATUS_MISSING)

    # Bank for r3
    banking.write_banked_sample(
        sample_id="sample_0",
        stage_name="r3",
        status=banking.STATUS_PASSED,
        local_samples_dir=self.samples_dir,
        metrics={"max_kl": 0.001, "mean_kl": 0.0001, "top10_overlap": 10},
        attempt_id="att1",
        pod="pod-0",
    )
    status, data, reason = masked_prefill_logits_checker.check_banked_sample_for_test(str(self.samples_dir), "sample_0")
    self.assertEqual(status, banking.STATUS_PASSED)
    self.assertIsNotNone(data)

  def test_compile_once_per_shape_and_bit_identical_logits(self):
    """Verifies single JIT compilation per static shape and bit-identical logits."""
    # pylint: disable=protected-access

    class DummyMLP(nnx.Module):
      """Dummy MLP module."""

      def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(d_in, d_out, rngs=rngs)
        self._frozen_topk_indices = nnx.Variable(jnp.array([0.0], dtype=jnp.float32))

      def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
        self.sow(nnx.Intermediate, "moe_lb_loss", jnp.sum(x))
        out = self.dense(x)
        if self._frozen_topk_indices is not None:
          out = out + jnp.sum(self._frozen_topk_indices[...])
        return out

    class DummyDecoderLayer(nnx.Module):
      """Dummy Decoder Layer module."""

      def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.mlp = DummyMLP(d_model, d_model, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
        return self.mlp(x)

    class DummyDecoder(nnx.Module):
      """Dummy Decoder module."""

      def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.layers_0 = DummyDecoderLayer(d_model, rngs=rngs)

      def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
        return self.layers_0(x)

    class DummyE2EModel(nnx.Module):
      """Dummy E2E Model for unit testing eval_step_nnx."""

      def __init__(self, vocab_size: int, d_model: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.decoder = DummyDecoder(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

      def __call__(  # pylint: disable=arguments-differ,too-many-positional-arguments,too-many-arguments
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        x = self.embed(decoder_input_tokens)
        x = self.decoder(x)
        return self.head(x)

    rngs = nnx.Rngs(0)
    model = DummyE2EModel(vocab_size=32, d_model=16, rngs=rngs)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)), ("data",))

    # (a) make_eval_step_nnx test vs eager model execution
    eval_step, nnx_state = forward_pass_logit_checker.make_eval_step_nnx(model, mesh)

    ids_1 = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    pos_1 = jnp.arange(8, dtype=jnp.int32)[None, :]
    seg_1 = jnp.ones((1, 8), dtype=jnp.int32)

    eager_1 = model(decoder_input_tokens=ids_1, decoder_positions=pos_1, decoder_segment_ids=seg_1, enable_dropout=False)
    jitted_1 = eval_step(nnx_state, ids_1, pos_1, seg_1)
    self.assertTrue(np.array_equal(np.asarray(jitted_1), np.asarray(eager_1)))
    self.assertEqual(float(np.max(np.abs(jitted_1 - eager_1))), 0.0)
    self.assertEqual(eval_step._cache_size(), 1)

    # Distinct tokens, identical shape (1, 8) -> compile count stays 1
    ids_2 = jnp.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=jnp.int32)
    eager_2 = model(decoder_input_tokens=ids_2, decoder_positions=pos_1, decoder_segment_ids=seg_1, enable_dropout=False)
    jitted_2 = eval_step(nnx_state, ids_2, pos_1, seg_1)
    self.assertTrue(np.array_equal(np.asarray(jitted_2), np.asarray(eager_2)))
    self.assertEqual(float(np.max(np.abs(jitted_2 - eager_2))), 0.0)
    self.assertEqual(eval_step._cache_size(), 1)

    # Different shape (1, 12) -> triggers second compilation
    ids_3 = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=jnp.int32)
    pos_3 = jnp.arange(12, dtype=jnp.int32)[None, :]
    seg_3 = jnp.ones((1, 12), dtype=jnp.int32)
    jitted_3 = eval_step(nnx_state, ids_3, pos_3, seg_3)
    self.assertEqual(jitted_3.shape, (1, 12, 32))
    self.assertEqual(eval_step._cache_size(), 2)

    # (b) deepseek_v41_frozen_topk memoized step functions test with nnx.Intermediate sow & update
    deepseek_v41_frozen_topk.clear_frozen_topk_step_caches()
    init_leaf_count = len(jax.tree_util.tree_leaves(nnx.state(model)))

    # Cycle 1: Sample 1 with _frozen_topk_indices = 1.0
    model.decoder.layers_0.mlp._frozen_topk_indices[...] = jnp.array([1.0], dtype=jnp.float32)
    diag_gdef_1, diag_state_1 = nnx.split(model)
    diag_step_1 = deepseek_v41_frozen_topk.get_frozen_topk_step_fn(model, mesh, None, diag_gdef_1)
    out_diag_1, updated_non_params_1 = diag_step_1(diag_state_1, (ids_1, pos_1, seg_1, None, None))
    clean_non_params_1 = nnx.filter_state(updated_non_params_1, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params_1)
    self.assertEqual(len(jax.tree_util.tree_leaves(nnx.state(model))), init_leaf_count)
    self.assertEqual(diag_step_1._cache_size(), 1)

    # Cycle 2: Sample 2 (same shape (1, 8)) with distinct _frozen_topk_indices = 2.0
    model.decoder.layers_0.mlp._frozen_topk_indices[...] = jnp.array([2.0], dtype=jnp.float32)
    diag_gdef_2, diag_state_2 = nnx.split(model)
    diag_step_2 = deepseek_v41_frozen_topk.get_frozen_topk_step_fn(model, mesh, None, diag_gdef_2)
    self.assertIs(diag_step_1, diag_step_2)
    out_diag_2, updated_non_params_2 = diag_step_2(diag_state_2, (ids_2, pos_1, seg_1, None, None))
    clean_non_params_2 = nnx.filter_state(updated_non_params_2, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params_2)
    self.assertEqual(len(jax.tree_util.tree_leaves(nnx.state(model))), init_leaf_count)
    self.assertEqual(diag_step_2._cache_size(), 1)
    diff_val = float(np.max(np.abs(np.asarray(out_diag_2) - np.asarray(out_diag_1))))
    self.assertGreater(diff_val, 0.0)

    # Different shape (1, 12) -> triggers second compilation
    model.decoder.layers_0.mlp._frozen_topk_indices[...] = jnp.array([3.0], dtype=jnp.float32)
    diag_gdef_3, diag_state_3 = nnx.split(model)
    diag_step_3 = deepseek_v41_frozen_topk.get_frozen_topk_step_fn(model, mesh, None, diag_gdef_3)
    out_diag_3, updated_non_params_3 = diag_step_3(diag_state_3, (ids_3, pos_3, seg_3, None, None))
    clean_non_params_3 = nnx.filter_state(updated_non_params_3, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params_3)
    self.assertEqual(out_diag_3.shape, (1, 12, 32))
    self.assertEqual(diag_step_3._cache_size(), 2)

    # Natural routing step cache test
    trace_gdef_1, trace_state_1 = nnx.split(model)
    trace_step_1 = deepseek_v41_frozen_topk.get_natural_routing_step_fn(model, mesh, None, trace_gdef_1)
    out_trace_1, updated_trace_1 = trace_step_1(trace_state_1, (ids_1, pos_1, seg_1, None, None))
    clean_trace_1 = nnx.filter_state(updated_trace_1, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_trace_1)
    self.assertEqual(out_trace_1.shape, (1, 8, 32))
    self.assertEqual(trace_step_1._cache_size(), 1)

    trace_gdef_2, trace_state_2 = nnx.split(model)
    trace_step_2 = deepseek_v41_frozen_topk.get_natural_routing_step_fn(model, mesh, None, trace_gdef_2)
    self.assertIs(trace_step_1, trace_step_2)
    out_trace_2, updated_trace_2 = trace_step_2(trace_state_2, (ids_2, pos_1, seg_1, None, None))
    clean_trace_2 = nnx.filter_state(updated_trace_2, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_trace_2)
    self.assertEqual(out_trace_2.shape, (1, 8, 32))
    self.assertEqual(trace_step_2._cache_size(), 1)

    trace_gdef_3, trace_state_3 = nnx.split(model)
    trace_step_3 = deepseek_v41_frozen_topk.get_natural_routing_step_fn(model, mesh, None, trace_gdef_3)
    out_trace_3, updated_trace_3 = trace_step_3(trace_state_3, (ids_3, pos_3, seg_3, None, None))
    clean_trace_3 = nnx.filter_state(updated_trace_3, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_trace_3)
    self.assertEqual(out_trace_3.shape, (1, 12, 32))
    self.assertEqual(trace_step_3._cache_size(), 2)


if __name__ == "__main__":
  unittest.main()
