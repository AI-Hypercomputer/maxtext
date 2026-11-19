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

import contextlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import jsonlines
from maxtext.common.common_types import MODEL_MODE_TRAIN
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
          encoder_audios: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, encoder_audios, enable_dropout
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

  def test_get_data_with_masks_threads_audio_values_without_pixel_values(self):
    """Verifies audio-only multimodal goldens thread tiled audio_values without requiring pixel_values."""
    raw_audio = np.arange(1 * 16 * 80, dtype=np.float32).reshape((1, 16, 80)) / 100.0
    golden_audio_only = {
        "prompt": "Transcribe this clip",
        "tokens": [1, 2, 3, 4, 5],
        "logits": np.zeros((5, 32), dtype=np.float32),
        "audio_values": raw_audio,
    }
    for model_name in ("ming-3.0-audio-tiny", "qwen3-omni-30b-a3b"):
      cfg_audio = types.SimpleNamespace(
          model_name=model_name,
          global_batch_size_to_train_on=2,
          max_target_length=8,
          use_multimodal=True,
          use_audio=True,
          use_mrope=False,
      )
      unpacked = forward_pass_logit_checker.get_data_with_masks(golden_audio_only, cfg_audio)
      self.assertEqual(len(unpacked), 8)
      ids, seg_ids, pos, logits, seq_len, pixel_values, image_masks, audio_values = unpacked
      self.assertEqual(ids.shape, (2, 8))
      self.assertEqual(seg_ids.shape, (2, 8))
      self.assertEqual(pos.shape, (2, 8))
      self.assertEqual(logits.shape, (5, 32))
      self.assertEqual(seq_len, 5)
      self.assertIsNone(pixel_values)
      self.assertIsNone(image_masks)
      self.assertIsNotNone(audio_values)
      assert audio_values is not None
      self.assertEqual(audio_values.shape, (2, 16, 80))
      np.testing.assert_array_equal(audio_values, np.concatenate([raw_audio, raw_audio], axis=0))

      legacy_unpacked = forward_pass_logit_checker.get_data(golden_audio_only, cfg_audio)
      self.assertEqual(len(legacy_unpacked), 7)
      self.assertIsNone(legacy_unpacked[5])
      np.testing.assert_array_equal(legacy_unpacked[6], audio_values)

    cfg_vision_missing_img = types.SimpleNamespace(
        model_name="gemma3-4b",
        global_batch_size_to_train_on=2,
        max_target_length=8,
        use_multimodal=True,
        use_audio=False,
        use_mrope=False,
    )
    with self.assertRaisesRegex(AssertionError, "no image found in golden data while use_multimodal=True"):
      forward_pass_logit_checker.get_data_with_masks(golden_audio_only, cfg_vision_missing_img)

    cfg_text_only = types.SimpleNamespace(
        model_name="llama3.1-8b",
        global_batch_size_to_train_on=2,
        max_target_length=8,
        use_multimodal=False,
        use_audio=False,
        use_mrope=False,
    )
    *_, audio_disabled = forward_pass_logit_checker.get_data_with_masks(golden_audio_only, cfg_text_only)
    self.assertIsNone(audio_disabled)

  def test_make_eval_step_nnx_and_main_thread_encoder_audios(self):
    """Verifies make_eval_step_nnx and main thread encoder_audios end-to-end across pkl and npz goldens."""

    class DummyAudioModel(nnx.Module):
      """Tiny NNX model whose output logits depend strongly on encoder_audios."""

      def __init__(self, vocab_size: int, d_model: int, n_mels: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.audio_proj = nnx.Linear(n_mels, d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

      def __call__(  # pylint: disable=arguments-differ,too-many-positional-arguments,too-many-arguments
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          encoder_audios: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, enable_dropout
        x = self.embed(decoder_input_tokens)
        if encoder_audios is not None:
          pooled_audio = jnp.mean(encoder_audios, axis=1, keepdims=True)
          x = x + 8.0 * self.audio_proj(pooled_audio)
        return self.head(x)

    audio_model = DummyAudioModel(vocab_size=32, d_model=16, n_mels=8, rngs=nnx.Rngs(7))
    single_dev = np.array(jax.devices()[:1]).reshape((1,))
    mesh = jax.sharding.Mesh(single_dev, ("data",))

    ids = jnp.array([[2, 4, 6, 8, 10, 0, 0, 0]], dtype=jnp.int32)
    pos = jnp.arange(8, dtype=jnp.int32)[None, :]
    seg = jnp.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32)
    raw_audio = np.linspace(0.5, 2.5, num=1 * 12 * 8, dtype=np.float32).reshape((1, 12, 8))
    audios_jax = jnp.asarray(raw_audio)

    eager_with_audio = np.asarray(
        audio_model(
            decoder_input_tokens=ids,
            decoder_positions=pos,
            decoder_segment_ids=seg,
            encoder_audios=audios_jax,
            enable_dropout=False,
        ),
        dtype=np.float32,
    )
    eager_without_audio = np.asarray(
        audio_model(
            decoder_input_tokens=ids,
            decoder_positions=pos,
            decoder_segment_ids=seg,
            encoder_audios=None,
            enable_dropout=False,
        ),
        dtype=np.float32,
    )
    self.assertGreater(float(np.max(np.abs(eager_with_audio - eager_without_audio))), 1.0)

    pkl_path = self.tmp_dir / "audio_golden.pkl"
    with open(pkl_path, "wb") as f:
      pickle.dump(
          [
              {
                  "prompt": "audio_sample",
                  "tokens": [2, 4, 6, 8, 10],
                  "logits": eager_with_audio[0, :5, :],
                  "audio_values": raw_audio,
              }
          ],
          f,
      )

    npz_path = self.tmp_dir / "audio_golden.npz"
    np.savez(
        npz_path,
        sample_id=np.asarray("audio_npz"),
        prompt=np.asarray("audio_npz_prompt"),
        tokens=np.asarray([2, 4, 6, 8, 10], dtype=np.int32),
        logits=eager_with_audio[0, :5, :],
        audio_values=raw_audio,
    )

    def _make_cfg(use_multimodal: bool):
      return types.SimpleNamespace(
          init_weights_seed=0,
          mesh_axes=("data",),
          hf_access_token="",
          tokenizer_path="",
          model_name="ming-3.0-audio-tiny",
          load_parameters_path="",
          scan_layers=False,
          use_multimodal=use_multimodal,
          use_audio=True,
          use_mrope=False,
          global_batch_size_to_train_on=1,
          max_target_length=8,
          vocab_size=32,
      )

    def _make_args(golden_path: pathlib.Path):
      return types.SimpleNamespace(
          hf_model_path="",
          trust_remote_code=True,
          run_ds41_mm_audit=False,
          hf_checkpoint_dir_for_audit="",
          export_orbax_dir="",
          mm_golden_logits_path="",
          thresholds_json="",
          max_images_per_sample=4,
          summary_json="",
          enable_tt2_gate=False,
          require_frozen_topk_gate=False,
          golden_logits_path=str(golden_path),
          output_logits_path="",
          gcs_output_logits_path="",
          run_hf_model=False,
          token_size=None,
          skip_first_token=False,
          atol=1e-5,
          rtol=1e-5,
          max_kl_div=1e-6,
          clip_logits_epsilon=None,
      )

    with (
        mock.patch.object(forward_pass_logit_checker.maxtext_utils, "create_device_mesh", return_value=single_dev),
        mock.patch.object(forward_pass_logit_checker.AutoTokenizer, "from_pretrained", return_value=None),
    ):
      # Check both use_multimodal=False (isolates encoder_audios threading) and use_multimodal=True (Ming/Qwen-Omni).
      summary_audio_only = forward_pass_logit_checker.main(
          _make_cfg(use_multimodal=False), _make_args(pkl_path), preloaded_model=audio_model
      )
      self.assertTrue(summary_audio_only["passed"])
      self.assertLess(summary_audio_only["max_kl"], 1e-6)

      summary_mm_audio = forward_pass_logit_checker.main(
          _make_cfg(use_multimodal=True), _make_args(pkl_path), preloaded_model=audio_model
      )
      self.assertTrue(summary_mm_audio["passed"])
      self.assertLess(summary_mm_audio["max_kl"], 1e-6)

      summary_npz_audio = forward_pass_logit_checker.main(
          _make_cfg(use_multimodal=True), _make_args(npz_path), preloaded_model=audio_model
      )
      self.assertTrue(summary_npz_audio["passed"])

      # Negative control: perturbed audio_values fails logit gates when encoder_audios is genuinely wired.
      bad_pkl_path = self.tmp_dir / "audio_golden_perturbed.pkl"
      with open(bad_pkl_path, "wb") as f:
        pickle.dump(
            [
                {
                    "prompt": "audio_sample_bad",
                    "tokens": [2, 4, 6, 8, 10],
                    "logits": eager_with_audio[0, :5, :],
                    "audio_values": -raw_audio,
                }
            ],
            f,
        )
      with self.assertRaisesRegex(AssertionError, "forward logit samples failed gates"):
        forward_pass_logit_checker.main(
            _make_cfg(use_multimodal=True), _make_args(bad_pkl_path), preloaded_model=audio_model
        )

    eval_step, nnx_state = forward_pass_logit_checker.make_eval_step_nnx(audio_model, mesh)
    jitted_with_audio = np.asarray(eval_step(nnx_state, ids, pos, seg, None, None, audios_jax), dtype=np.float32)
    jitted_without_audio = np.asarray(eval_step(nnx_state, ids, pos, seg), dtype=np.float32)
    np.testing.assert_allclose(jitted_with_audio, eager_with_audio, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(jitted_without_audio, eager_without_audio, atol=1e-6, rtol=1e-6)

  def test_main_run_hf_model_initializes_model_and_eval_step_nnx(self):
    """Verifies --run_hf_model=true initializes model and eval_step_nnx without UnboundLocalError (torch-free)."""

    class _NumpyTensorStub:
      """Minimal numpy-backed tensor stub exercising check_kl_divergence without requiring PyTorch."""

      def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr, dtype=np.float32)

      @property
      def shape(self) -> tuple[int, ...]:
        return self._arr.shape

      def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._arr, dtype=dtype)

      def __getitem__(self, idx) -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr[idx])

      def reshape(self, *shape: int) -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr.reshape(shape))

      def unsqueeze(self, dim: int) -> "_NumpyTensorStub":
        return _NumpyTensorStub(np.expand_dims(self._arr, axis=dim))

      def sum(self, dim: int = -1, keepdim: bool = False) -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr.sum(axis=dim, keepdims=keepdim))

      def max(self) -> "_NumpyTensorStub":
        return _NumpyTensorStub(np.asarray(self._arr.max(), dtype=np.float32))

      def item(self) -> float:
        return float(self._arr.item())

      def tolist(self):
        return self._arr.tolist()

      def __truediv__(self, other: "_NumpyTensorStub") -> "_NumpyTensorStub":
        return _NumpyTensorStub(self._arr / np.asarray(other))

      def __lt__(self, other: float) -> bool:
        return bool(self._arr < other)

    def _fake_softmax(x: _NumpyTensorStub, dim: int = -1) -> _NumpyTensorStub:
      arr = np.asarray(x, dtype=np.float64)
      shifted = arr - np.max(arr, axis=dim, keepdims=True)
      exp = np.exp(shifted)
      return _NumpyTensorStub((exp / np.sum(exp, axis=dim, keepdims=True)).astype(np.float32))

    def _fake_log_softmax(x: _NumpyTensorStub, dim: int = -1) -> _NumpyTensorStub:
      arr = np.asarray(x, dtype=np.float64)
      shifted = arr - np.max(arr, axis=dim, keepdims=True)
      log_z = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
      return _NumpyTensorStub((shifted - log_z).astype(np.float32))

    def _fake_kl_div(
        input: _NumpyTensorStub,  # pylint: disable=redefined-builtin
        target: _NumpyTensorStub,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> _NumpyTensorStub:
      del log_target
      log_q = np.asarray(input, dtype=np.float64)
      p = np.asarray(target, dtype=np.float64)
      elem = np.where(p > 0.0, p * (np.log(np.maximum(p, 1e-30)) - log_q), 0.0)
      if reduction == "none":
        return _NumpyTensorStub(elem.astype(np.float32))
      if reduction == "batchmean":
        return _NumpyTensorStub(np.asarray(elem.sum() / p.shape[0], dtype=np.float32))
      return _NumpyTensorStub(np.asarray(elem.mean(), dtype=np.float32))

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        float16="float16",
        no_grad=contextlib.nullcontext,
        clamp=lambda x, min=None: _NumpyTensorStub(np.clip(np.asarray(x), a_min=min, a_max=None)),
        log=lambda x: _NumpyTensorStub(np.log(np.asarray(x))),
    )
    fake_f = types.SimpleNamespace(
        softmax=_fake_softmax,
        log_softmax=_fake_log_softmax,
        kl_div=_fake_kl_div,
    )

    class DummyRunHFModel(nnx.Module):
      """Tiny NNX model for on-the-fly --run_hf_model parity tests."""

      def __init__(self, vocab_size: int, d_model: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

      def __call__(  # pylint: disable=arguments-differ,too-many-positional-arguments,too-many-arguments
          self,
          decoder_input_tokens: jax.Array,
          decoder_positions: jax.Array | None = None,
          decoder_segment_ids: jax.Array | None = None,
          encoder_images: jax.Array | None = None,
          encoder_image_masks: jax.Array | None = None,
          encoder_audios: jax.Array | None = None,
          enable_dropout: bool = False,
      ) -> jax.Array:
        del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, encoder_audios, enable_dropout
        return self.head(self.embed(decoder_input_tokens))

    tiny_nnx_model = DummyRunHFModel(vocab_size=32, d_model=16, rngs=nnx.Rngs(11))
    single_dev = np.array(jax.devices()[:1]).reshape((1,))

    class _FakeTokenizer:
      """Deterministic word-length tokenizer stub."""

      eos_token = "<eos>"
      pad_token = "<pad>"

      def __call__(
          self,
          text: str,
          *,
          return_tensors: str = "pt",
          padding: bool = True,
          max_length: int = 16,
          truncation: bool = True,
      ):
        del return_tensors, padding, truncation
        toks = [((ord(ch) % 28) + 2) for ch in text.split()[:max_length] for ch in ch[:1]][:max_length]
        if not toks:
          toks = [2, 3, 4]
        return {"input_ids": np.asarray([toks], dtype=np.int32)}

      def decode(self, tok_id: int) -> str:
        return f"tok_{int(tok_id)}"

    logit_perturbation = {"offset": 0.0}

    class _FakeHFModel:
      """HF model stub emitting exact (or perturbed) logits matching tiny_nnx_model."""

      def __call__(self, **inputs):
        ids_np = np.asarray(inputs["input_ids"], dtype=np.int32)
        logits_jax = tiny_nnx_model(decoder_input_tokens=jnp.asarray(ids_np))
        logits_np = np.asarray(logits_jax, dtype=np.float32)
        if logit_perturbation["offset"] != 0.0:
          logits_np = logits_np[:, :, ::-1] + logit_perturbation["offset"]
        return types.SimpleNamespace(logits=_NumpyTensorStub(logits_np))

    hf_dir = self.tmp_dir / "fake_hf_dir"
    hf_dir.mkdir(parents=True, exist_ok=True)
    out_logits_file = self.tmp_dir / "out" / "run_hf_logits.jsonl"

    cfg = types.SimpleNamespace(
        init_weights_seed=0,
        mesh_axes=("data",),
        hf_access_token="",
        tokenizer_path="",
        model_name="qwen3-0.6b",
        dtype="float32",
        hf_lora_adapter_path="",
        global_batch_size_to_train_on=1,
        max_target_length=16,
        vocab_size=32,
        lora=types.SimpleNamespace(enable_lora=True, lora_restore_path="/tmp/fake_lora"),
    )
    test_args = types.SimpleNamespace(
        hf_model_path=str(hf_dir),
        trust_remote_code=True,
        run_ds41_mm_audit=False,
        hf_checkpoint_dir_for_audit="",
        export_orbax_dir="",
        mm_golden_logits_path="",
        thresholds_json="",
        max_images_per_sample=4,
        summary_json="",
        enable_tt2_gate=False,
        require_frozen_topk_gate=False,
        golden_logits_path="",
        output_logits_path=str(out_logits_file),
        gcs_output_logits_path="",
        run_hf_model=True,
        reference_model="",
        token_size=None,
        skip_first_token=False,
        atol=None,
        rtol=1e-5,
        max_kl_div=1e-5,
        clip_logits_epsilon=None,
    )

    with (
        mock.patch.object(forward_pass_logit_checker.maxtext_utils, "create_device_mesh", return_value=single_dev),
        mock.patch.object(forward_pass_logit_checker, "torch", fake_torch),
        mock.patch.object(forward_pass_logit_checker, "F", fake_f),
        mock.patch.object(
            forward_pass_logit_checker,
            "convert_jax_weight_to_torch",
            side_effect=lambda arr: _NumpyTensorStub(np.asarray(arr, dtype=np.float32)),
        ),
        mock.patch.object(
            forward_pass_logit_checker.AutoTokenizer,
            "from_pretrained",
            return_value=_FakeTokenizer(),
        ),
        mock.patch.object(
            forward_pass_logit_checker,
            "AutoModelForCausalLM",
            types.SimpleNamespace(from_pretrained=mock.Mock(return_value=_FakeHFModel())),
        ),
        mock.patch.object(
            forward_pass_logit_checker.model_creation_utils,
            "from_pretrained",
            return_value=tiny_nnx_model,
        ) as mock_mt_from_pretrained,
        mock.patch.object(
            forward_pass_logit_checker.lora_utils,
            "apply_lora_to_model",
            side_effect=lambda model, mesh, config: model,
        ) as mock_apply_lora,
        mock.patch.object(
            forward_pass_logit_checker.lora_utils,
            "restore_lora_from_path",
        ) as mock_restore_lora,
    ):
      # (0) Missing hf_model_path still fails fast before loading MaxText weights.
      missing_hf_args = types.SimpleNamespace(**{**vars(test_args), "hf_model_path": ""})
      with self.assertRaisesRegex(ValueError, "run_hf_model requires hf_model_path"):
        forward_pass_logit_checker.main(cfg, missing_hf_args, preloaded_model=None)
      mock_mt_from_pretrained.assert_not_called()

      # (i) preloaded_model=None initializes MaxText model (+ LoRA) and runs @jax.jit eval_step_nnx across prompts.
      result_none = forward_pass_logit_checker.main(cfg, test_args, preloaded_model=None)
      self.assertIsNone(result_none)
      mock_mt_from_pretrained.assert_called_once()
      self.assertEqual(mock_mt_from_pretrained.call_args.kwargs["model_mode"], MODEL_MODE_TRAIN)
      mock_apply_lora.assert_called_once()
      mock_restore_lora.assert_called_once()
      self.assertTrue(out_logits_file.is_file())
      with jsonlines.open(out_logits_file, "r") as reader:
        saved_batches = list(reader)
      self.assertEqual(len(saved_batches), 1)
      self.assertEqual(len(saved_batches[0]), 4)

      # (ii) preloaded_model=tiny_nnx_model reuses preloaded model without calling from_pretrained again.
      mock_mt_from_pretrained.reset_mock()
      no_save_args = types.SimpleNamespace(**{**vars(test_args), "output_logits_path": ""})
      result_preloaded = forward_pass_logit_checker.main(cfg, no_save_args, preloaded_model=tiny_nnx_model)
      self.assertIsNone(result_preloaded)
      mock_mt_from_pretrained.assert_not_called()

      # (iii) Negative control: mismatched HF logits fail check_kl_divergence.
      logit_perturbation["offset"] = 3.0
      with self.assertRaisesRegex(AssertionError, "KL divergence values .* exceed the threshold"):
        forward_pass_logit_checker.main(cfg, no_save_args, preloaded_model=tiny_nnx_model)


if __name__ == "__main__":
  unittest.main()
