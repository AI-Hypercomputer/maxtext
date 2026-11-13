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

"""Unit and regression tests for masked_prefill_logits_checker.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import yaml

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from tests.utils import deepseek_v41_frozen_topk
from tests.utils import deepseek_v41_preemption_banking as banking
from tests.utils import masked_prefill_logits_checker as checker
import transformers

try:
  import torch
  from safetensors.torch import save_file
  from tests.utils import deepseek_v41_mm_testing as mm_testing
  from tests.utils.deepseek_v41_reference import multimodal

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  save_file = None  # type: ignore[assignment]
  mm_testing = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  HAS_TORCH = False


class MaskedPrefillLogitsCheckerUnitTest(unittest.TestCase):
  """Unit tests for rerun determinism, batch drift, and sample gating logic."""

  def test_negative_control_a_rerun_single_element_perturbation_fails(self):
    """Negative control (a): perturbing run2 in one element fails rerun determinism."""
    run1 = np.zeros((4, 16, 2048), dtype=np.float32)
    run1[:, :, 5] = 3.0
    run2 = run1.copy()
    run2[1, 3, 7] = np.nextafter(run2[1, 3, 7], np.float32(1.0))

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=16,
        min_vocab_size=2048,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
    )
    self.assertFalse(res["rerun_bitwise_identical"])
    self.assertGreater(res["rerun_max_abs_diff"], 0.0)
    self.assertTrue(res["rerun_argmax_identical"])
    self.assertFalse(res["passed"])

    # Also verify argmax flip when the perturbed element overtakes top-1.
    run2_flip = run1.copy()
    run2_flip[2, 9, 11] = 4.0
    res_flip = checker.check_batch_slot_determinism(
        run1,
        seq_len=16,
        min_vocab_size=2048,
        rows_per_batch_shard=2,
        rerun_train_logits=run2_flip,
    )
    self.assertFalse(res_flip["rerun_bitwise_identical"])
    self.assertFalse(res_flip["rerun_argmax_identical"])
    self.assertFalse(res_flip["passed"])

  def test_negative_control_b_row_mean_drift_exceeds_threshold_fails(self):
    """Negative control (b): row mean drift > 1e-4 fails even when reruns match bitwise."""
    run1 = np.zeros((4, 16, 2048), dtype=np.float32)
    run1[1:] += 3e-4
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=16,
        min_vocab_size=2048,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
    )
    self.assertTrue(res["rerun_bitwise_identical"])
    self.assertEqual(res["rerun_max_abs_diff"], 0.0)
    self.assertGreater(res["mean_batch_diff"], 1e-4)
    self.assertFalse(res["passed"])

  def test_negative_control_c_single_element_row_spike_passes_with_diagnostics(self):
    """Control (c): row-to-row single-element spike with identical reruns and mean <= 1e-4 passes."""
    run1 = np.zeros((4, 16, 2048), dtype=np.float32)
    run1[:, :, 0] = 10.0
    run1[1, 0, 42] = 0.644418
    run1[2, 5, 99] = 0.518848
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=16,
        min_vocab_size=2048,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
    )
    self.assertTrue(res["rerun_bitwise_identical"])
    self.assertEqual(res["rerun_max_abs_diff"], 0.0)
    self.assertTrue(res["rerun_argmax_identical"])
    self.assertLessEqual(res["mean_batch_diff"], 1e-4)
    self.assertTrue(res["passed"])
    # Spike remains visible in the logged diagnostics.
    self.assertTrue(res["intra_shard_checked"])
    self.assertAlmostEqual(res["intra_shard_max_diff"], 0.644418, places=5)
    self.assertAlmostEqual(res["cross_shard_max_diff"], 0.518848, places=5)
    self.assertAlmostEqual(res["max_batch_diff"], 0.644418, places=5)

  def test_evaluate_sample_gates_uses_rerun_determinism_definition(self):
    """Verifies evaluate_sample_gates gates on rerun_bitwise_identical and mean_batch_diff <= 1e-4."""
    effective_min_cos = 0.999
    effective_max_kl = 0.08
    bucket_gates = {"mean_kl": 1e-3, "top1_agreement": 0.98, "p95_kl": 5e-3, "p99_kl": 2e-3}
    clean_parity = {
        "mean_cos_sim": 0.99998,
        "max_kl": 0.001,
        "mean_kl": 2e-6,
        "top1_agreement": 1.0,
        "p95_kl": 2e-7,
        "p99_kl": 2e-7,
    }

    # 1. Identical reruns + mean_batch_diff <= 1e-4 passes despite large intra/cross/max spikes.
    batch_spike_pass = {
        "rerun_bitwise_identical": True,
        "rerun_argmax_identical": True,
        "rerun_max_abs_diff": 0.0,
        "mean_batch_diff": 8.96e-5,
        "intra_shard_max_diff": 0.644418,
        "cross_shard_max_diff": 0.644416,
        "max_batch_diff": 0.644418,
        "mean_diff_threshold": 1e-4,
    }
    gates_spike = checker.evaluate_sample_gates(
        parity_metrics=clean_parity,
        batch_det=batch_spike_pass,
        bucket_gates=bucket_gates,
        effective_max_kl=effective_max_kl,
        effective_min_cos=effective_min_cos,
    )
    self.assertTrue(gates_spike["batch_determinism_pass"])
    self.assertTrue(all(gates_spike.values()))

    # 2. Non-identical rerun fails even when mean_batch_diff == 0.0.
    batch_rerun_fail = {
        **batch_spike_pass,
        "rerun_bitwise_identical": False,
        "rerun_max_abs_diff": 1e-6,
        "mean_batch_diff": 0.0,
        "intra_shard_max_diff": 0.0,
    }
    gates_rerun_fail = checker.evaluate_sample_gates(
        parity_metrics=clean_parity,
        batch_det=batch_rerun_fail,
        bucket_gates=bucket_gates,
        effective_max_kl=effective_max_kl,
        effective_min_cos=effective_min_cos,
    )
    self.assertFalse(gates_rerun_fail["batch_determinism_pass"])
    self.assertFalse(all(gates_rerun_fail.values()))

    # 3. Mean batch drift > 1e-4 fails even when rerun_bitwise_identical is True.
    batch_mean_fail = {
        **batch_spike_pass,
        "mean_batch_diff": 1.5e-4,
        "intra_shard_max_diff": 1.5e-4,
    }
    gates_mean_fail = checker.evaluate_sample_gates(
        parity_metrics=clean_parity,
        batch_det=batch_mean_fail,
        bucket_gates=bucket_gates,
        effective_max_kl=effective_max_kl,
        effective_min_cos=effective_min_cos,
    )
    self.assertFalse(gates_mean_fail["batch_determinism_pass"])
    self.assertFalse(all(gates_mean_fail.values()))

  def test_cos_sim_diagnostic_only_gate_behavior(self):
    """Verifies cos_sim_diagnostic_only behavior in evaluate_sample_gates."""
    effective_min_cos = 0.999
    effective_max_kl = 0.08
    bucket_gates = {"mean_kl": 1e-3, "top1_agreement": 0.98, "p95_kl": 5e-3, "p99_kl": 2e-3}
    batch_det_pass = {
        "rerun_bitwise_identical": True,
        "rerun_argmax_identical": True,
        "mean_batch_diff": 1e-6,
        "mean_diff_threshold": 1e-4,
    }
    parity_low_cos = {
        "mean_cos_sim": 0.9983,
        "max_kl": 0.01,
        "mean_kl": 1e-4,
        "top1_agreement": 0.99,
        "p95_kl": 1e-4,
        "p99_kl": 1e-4,
    }

    gates_strict = checker.evaluate_sample_gates(
        parity_metrics=parity_low_cos,
        batch_det=batch_det_pass,
        bucket_gates=bucket_gates,
        effective_max_kl=effective_max_kl,
        effective_min_cos=effective_min_cos,
        cos_sim_diagnostic_only=False,
    )
    self.assertFalse(gates_strict["mean_cos_sim_pass"])

    gates_diag = checker.evaluate_sample_gates(
        parity_metrics=parity_low_cos,
        batch_det=batch_det_pass,
        bucket_gates=bucket_gates,
        effective_max_kl=effective_max_kl,
        effective_min_cos=effective_min_cos,
        cos_sim_diagnostic_only=True,
    )
    self.assertTrue(gates_diag["mean_cos_sim_pass"])
    self.assertTrue(all(gates_diag.values()))

  def test_positive_control_tie_flip_envelope_passes_single_token_tie(self):
    """Positive control: 1 token with kl_primary=0.085 > 0.08 and kl_tie_flip~1e-4 <= 0.08 passes with p99_kl <= 2e-3."""
    seq_len, vocab = 200, 64
    golden = np.full((seq_len, vocab), -20.0, dtype=np.float32)
    golden[:, 0] = 12.0
    golden[42, 0] = 10.0
    golden[42, 1] = 10.0

    # At pos 42, P_golden = (0.5, 0.5) and D_KL((0.5, 0.5) || (e^d, e^-d)/(2 cosh d)) = log(cosh(d)) = 0.085.
    d_085 = float(np.arccosh(np.exp(0.085)))
    model = golden.copy()
    model[42, 0] = 10.0 + d_085
    model[42, 1] = 10.0 - d_085

    # tie_flip_logits matches the alternate tie branch at pos 42 up to KL ~ 1e-4.
    tie_flip = golden.copy()
    tie_flip[42, 0] = 10.0 + d_085 + 0.0155
    tie_flip[42, 1] = 10.0 - d_085 - 0.0155

    parity = checker.check_mathematical_parity(
        model_logits=model,
        golden_logits=golden,
        min_cos_sim=None,
        max_kl=None,
        tokens=np.arange(seq_len, dtype=np.int32),
        tie_flip_golden_logits=tie_flip,
    )
    self.assertTrue(parity["tie_flip_envelope_used"])
    self.assertAlmostEqual(parity["raw_primary_max_kl"], 0.085, places=4)
    self.assertEqual(parity["raw_primary_argmax_kl_pos"], 42)
    self.assertLessEqual(parity["max_kl"], 2e-4)
    self.assertGreater(parity["max_kl"], 0.0)
    self.assertLessEqual(parity["p99_kl"], 2e-3)

    bucket_gates = checker.get_length_bucket_thresholds(1185)
    self.assertEqual(bucket_gates["p99_kl"], 0.002)
    self.assertEqual(checker.get_length_bucket_thresholds(2500)["p99_kl"], 0.005)
    batch_det_pass = {
        "rerun_bitwise_identical": True,
        "rerun_argmax_identical": True,
        "mean_batch_diff": 0.0,
        "mean_diff_threshold": 1e-4,
    }
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=batch_det_pass,
        bucket_gates=bucket_gates,
        effective_max_kl=float(bucket_gates["max_kl"]),
        effective_min_cos=float(bucket_gates["mean_cos_sim"]),
    )
    self.assertTrue(gates["max_kl_pass"])
    self.assertTrue(gates["p99_kl_pass"])
    self.assertTrue(all(gates.values()))

  def test_negative_control_1_genuine_single_token_bug_fails_both_branches(self):
    """Negative control 1: 1 token corrupted so both kl_primary > 0.08 and kl_tie_flip > 0.08 fails max_kl_pass."""
    seq_len, vocab = 100, 64
    golden = np.full((seq_len, vocab), -20.0, dtype=np.float32)
    golden[:, 0] = 12.0
    golden[42, 0] = 10.0
    golden[42, 1] = 10.0

    tie_flip = golden.copy()
    tie_flip[42, 0] = 10.2
    tie_flip[42, 1] = 9.8

    # Corrupt pos 42 onto a wrong token index so both primary and tie_flip exceed 0.08 KL.
    model = golden.copy()
    model[42, 0] = 8.5
    model[42, 1] = 8.5
    model[42, 2] = 11.0

    parity = checker.check_mathematical_parity(
        model_logits=model,
        golden_logits=golden,
        min_cos_sim=None,
        max_kl=None,
        tie_flip_golden_logits=tie_flip,
    )
    self.assertTrue(parity["tie_flip_envelope_used"])
    self.assertGreater(parity["raw_primary_max_kl"], 0.08)
    self.assertGreater(parity["max_kl"], 0.08)

    bucket_gates = checker.get_length_bucket_thresholds(1185)
    batch_det_pass = {
        "rerun_bitwise_identical": True,
        "rerun_argmax_identical": True,
        "mean_batch_diff": 0.0,
        "mean_diff_threshold": 1e-4,
    }
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=batch_det_pass,
        bucket_gates=bucket_gates,
        effective_max_kl=float(bucket_gates["max_kl"]),
        effective_min_cos=float(bucket_gates["mean_cos_sim"]),
    )
    self.assertFalse(gates["max_kl_pass"])
    self.assertFalse(all(gates.values()))

  def test_negative_control_2_p99_kl_breach_fails_despite_tie_flip_envelope(self):
    """Negative control 2: p99_kl > 2e-3 against primary logits fails p99_kl_pass even when envelope max_kl <= 0.08."""
    seq_len, vocab = 100, 64
    golden = np.full((seq_len, vocab), -20.0, dtype=np.float32)
    golden[:, 0] = 12.0
    d_004 = float(np.arccosh(np.exp(0.004)))
    model = golden.copy()
    for pos_i in (97, 98, 99):
      golden[pos_i, 0] = 10.0
      golden[pos_i, 1] = 10.0
      model[pos_i, 0] = 10.0 + d_004
      model[pos_i, 1] = 10.0 - d_004

    # tie_flip matches model everywhere so envelope max_kl == 0.0 <= 0.08, yet primary p99_kl == 0.004 > 2e-3.
    tie_flip = model.copy()
    parity = checker.check_mathematical_parity(
        model_logits=model,
        golden_logits=golden,
        min_cos_sim=None,
        max_kl=None,
        tie_flip_golden_logits=tie_flip,
    )
    self.assertTrue(parity["tie_flip_envelope_used"])
    self.assertLessEqual(parity["max_kl"], 0.08)
    self.assertLessEqual(parity["p95_kl"], 5e-3)
    self.assertGreater(parity["p99_kl"], 2e-3)

    bucket_gates = checker.get_length_bucket_thresholds(1185)
    batch_det_pass = {
        "rerun_bitwise_identical": True,
        "rerun_argmax_identical": True,
        "mean_batch_diff": 0.0,
        "mean_diff_threshold": 1e-4,
    }
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=batch_det_pass,
        bucket_gates=bucket_gates,
        effective_max_kl=float(bucket_gates["max_kl"]),
        effective_min_cos=float(bucket_gates["mean_cos_sim"]),
    )
    self.assertTrue(gates["max_kl_pass"])
    self.assertTrue(gates["p95_kl_pass"])
    self.assertFalse(gates["p99_kl_pass"])
    self.assertFalse(all(gates.values()))

  def test_negative_control_3_tie_flip_shape_or_nonfinite_raises_value_error(self):
    """Negative control 3: mismatched shape or non-finite tie_flip_logits raises ValueError."""
    golden = np.zeros((16, 64), dtype=np.float32)
    golden[:, 0] = 5.0
    model = golden.copy()

    with self.assertRaisesRegex(ValueError, "shape"):
      checker.check_mathematical_parity(
          model_logits=model,
          golden_logits=golden,
          min_cos_sim=None,
          max_kl=None,
          tie_flip_golden_logits=np.zeros((15, 64), dtype=np.float32),
      )

    bad_nan = golden.copy()
    bad_nan[3, 7] = np.nan
    with self.assertRaisesRegex(ValueError, "non-finite"):
      checker.check_mathematical_parity(
          model_logits=model,
          golden_logits=golden,
          min_cos_sim=None,
          max_kl=None,
          tie_flip_golden_logits=bad_nan,
      )

    bad_inf = golden.copy()
    bad_inf[0, 0] = np.inf
    with self.assertRaisesRegex(ValueError, "non-finite"):
      checker._extract_tie_flip_slice({"tie_flip_logits": bad_inf}, golden, 0, 16, 64)  # pylint: disable=protected-access

  def test_tie_flip_envelope_and_primary_p99_kl_gates(self):
    """Verifies Option 1 tie_flip_logits max_kl envelope, primary p99_kl <= 2e-3 gate, and loader validation."""
    for seq_len_probe in (128, 768, 1500):
      bg = checker.get_length_bucket_thresholds(seq_len_probe)
      self.assertEqual(bg["p99_kl"], 0.002)

    batch_det_ok = {
        "rerun_bitwise_identical": True,
        "rerun_argmax_identical": True,
        "mean_batch_diff": 0.0,
        "mean_diff_threshold": 1e-4,
    }

    # (i) Positive control: 1 near-tie token in 1025-2048 bucket (seq_len=1200) exceeds max_kl=0.08 on primary
    #     logits, matches tie_flip_logits, and keeps primary p99_kl == 0.0 <= 0.002.
    model_1200 = np.zeros((1200, 32), dtype=np.float32)
    model_1200[:, 0] = 6.0
    golden_1200 = model_1200.copy()
    golden_1200[500, 1] = 3.6  # Primary KL at token 500 > 0.08.
    tie_flip_1200 = golden_1200.copy()
    tie_flip_1200[500] = model_1200[500]

    bucket_1200 = checker.get_length_bucket_thresholds(1200)
    res_no_env = checker.check_mathematical_parity(model_1200, golden_1200)
    self.assertFalse(res_no_env["tie_flip_envelope_used"])
    self.assertGreater(res_no_env["max_kl"], 0.08)
    self.assertEqual(res_no_env["p99_kl"], 0.0)
    gates_no_env = checker.evaluate_sample_gates(
        parity_metrics=res_no_env,
        batch_det=batch_det_ok,
        bucket_gates=bucket_1200,
        effective_max_kl=float(bucket_1200["max_kl"]),
        effective_min_cos=float(bucket_1200["mean_cos_sim"]),
    )
    self.assertFalse(gates_no_env["max_kl_pass"])
    self.assertTrue(gates_no_env["p99_kl_pass"])

    res_env = checker.check_mathematical_parity(model_1200, golden_1200, tie_flip_golden_logits=tie_flip_1200)
    self.assertTrue(res_env["tie_flip_envelope_used"])
    self.assertGreater(res_env["raw_primary_max_kl"], 0.08)
    self.assertEqual(res_env["raw_primary_argmax_kl_pos"], 500)
    self.assertEqual(res_env["max_kl"], 0.0)
    self.assertEqual(res_env["p99_kl"], 0.0)
    # Aggregate/percentile stats MUST match primary unperturbed logits identically.
    self.assertEqual(res_env["mean_kl"], res_no_env["mean_kl"])
    self.assertEqual(res_env["p95_kl"], res_no_env["p95_kl"])
    self.assertEqual(res_env["p99_kl"], res_no_env["p99_kl"])
    self.assertEqual(res_env["top1_agreement"], res_no_env["top1_agreement"])
    self.assertEqual(res_env["mean_cos_sim"], res_no_env["mean_cos_sim"])

    gates_env = checker.evaluate_sample_gates(
        parity_metrics=res_env,
        batch_det=batch_det_ok,
        bucket_gates=bucket_1200,
        effective_max_kl=float(bucket_1200["max_kl"]),
        effective_min_cos=float(bucket_1200["mean_cos_sim"]),
    )
    self.assertTrue(gates_env["max_kl_pass"])
    self.assertTrue(gates_env["p99_kl_pass"])
    self.assertTrue(all(gates_env.values()))

    # (ii) Negative control 1: genuine single-token corruption exceeds max_kl against BOTH primary and tie_flip_logits.
    corrupt_model_1200 = model_1200.copy()
    corrupt_model_1200[777, 5] = 12.0
    res_corrupt = checker.check_mathematical_parity(corrupt_model_1200, golden_1200, tie_flip_golden_logits=tie_flip_1200)
    self.assertGreater(res_corrupt["max_kl"], 0.08)
    gates_corrupt = checker.evaluate_sample_gates(
        parity_metrics=res_corrupt,
        batch_det=batch_det_ok,
        bucket_gates=bucket_1200,
        effective_max_kl=float(bucket_1200["max_kl"]),
        effective_min_cos=float(bucket_1200["mean_cos_sim"]),
    )
    self.assertFalse(gates_corrupt["max_kl_pass"])
    self.assertFalse(all(gates_corrupt.values()))

    # (iii) Negative control 2: primary p99_kl > 0.002 fails p99_kl_pass even when tie_flip_logits == model_logits.
    model_200 = np.zeros((200, 32), dtype=np.float32)
    model_200[:, 0] = 4.0
    golden_200 = model_200.copy()
    golden_200[:4, 1] = 0.8  # 4/200 tokens (2%) have KL ~ 0.0056: > 0.002 (p99), <= 0.025 (max_kl), p95 == 0.0.
    bucket_200 = checker.get_length_bucket_thresholds(200)
    res_p99_fail = checker.check_mathematical_parity(
        model_200,
        golden_200,
        tie_flip_golden_logits=model_200.copy(),
    )
    self.assertEqual(res_p99_fail["max_kl"], 0.0)
    self.assertEqual(res_p99_fail["p95_kl"], 0.0)
    self.assertGreater(res_p99_fail["p99_kl"], 0.002)
    self.assertLessEqual(res_p99_fail["raw_primary_max_kl"], 0.025)
    gates_p99_fail = checker.evaluate_sample_gates(
        parity_metrics=res_p99_fail,
        batch_det=batch_det_ok,
        bucket_gates=bucket_200,
        effective_max_kl=float(bucket_200["max_kl"]),
        effective_min_cos=float(bucket_200["mean_cos_sim"]),
    )
    self.assertTrue(gates_p99_fail["max_kl_pass"])
    self.assertTrue(gates_p99_fail["mean_kl_pass"])
    self.assertTrue(gates_p99_fail["p95_kl_pass"])
    self.assertTrue(gates_p99_fail["top1_agreement_pass"])
    self.assertTrue(gates_p99_fail["mean_cos_sim_pass"])
    self.assertFalse(gates_p99_fail["p99_kl_pass"])
    self.assertFalse(all(gates_p99_fail.values()))

    # (iv) Validation negative controls: wrong-shape or NaN/Inf tie_flip_logits raise ValueError.
    with self.assertRaises(ValueError):
      checker.check_mathematical_parity(model_200, golden_200, tie_flip_golden_logits=model_200[:, :-1])
    for non_finite in (np.nan, np.inf, -np.inf):
      bad_tf = model_200.copy()
      bad_tf[0, 0] = non_finite
      with self.assertRaises(ValueError):
        checker.check_mathematical_parity(model_200, golden_200, tie_flip_golden_logits=bad_tf)

    tmp_loader_dir = tempfile.mkdtemp(prefix="tie_flip_loader_")
    try:
      good_npz = os.path.join(tmp_loader_dir, "good_b1.npz")
      np.savez(
          good_npz,
          sample_id=np.array("S0"),
          tokens=np.arange(8, dtype=np.int32),
          logits=np.zeros((1, 8, 16), dtype=np.float32),
          tie_flip_logits=np.ones((1, 8, 16), dtype=np.float32),
      )
      loaded_good = checker.load_golden_dataset(good_npz)
      self.assertEqual(loaded_good[0]["logits"].shape, (8, 16))
      self.assertEqual(loaded_good[0]["tie_flip_logits"].shape, (8, 16))

      bad_shape_npz = os.path.join(tmp_loader_dir, "bad_shape.npz")
      np.savez(
          bad_shape_npz,
          sample_id=np.array("S_bad_shape"),
          tokens=np.arange(8, dtype=np.int32),
          logits=np.zeros((8, 16), dtype=np.float32),
          tie_flip_logits=np.zeros((8, 15), dtype=np.float32),
      )
      with self.assertRaises(ValueError):
        checker.load_golden_dataset(bad_shape_npz)

      bad_nan_npz = os.path.join(tmp_loader_dir, "bad_nan.npz")
      nan_arr = np.zeros((8, 16), dtype=np.float32)
      nan_arr[2, 3] = np.nan
      np.savez(
          bad_nan_npz,
          sample_id=np.array("S_bad_nan"),
          tokens=np.arange(8, dtype=np.int32),
          logits=np.zeros((8, 16), dtype=np.float32),
          tie_flip_logits=nan_arr,
      )
      with self.assertRaises(ValueError):
        checker.load_golden_dataset(bad_nan_npz)

      bad_pkl = os.path.join(tmp_loader_dir, "bad_inf.pkl")
      inf_arr = np.zeros((8, 16), dtype=np.float32)
      inf_arr[1, 1] = np.inf
      with open(bad_pkl, "wb") as f:
        pickle.dump(
            [
                {
                    "name": "P_inf",
                    "tokens": [1] * 8,
                    "logits": np.zeros((8, 16), dtype=np.float32),
                    "tie_flip_logits": inf_arr,
                }
            ],
            f,
        )
      with self.assertRaises(ValueError):
        checker.load_golden_dataset(bad_pkl)
    finally:
      shutil.rmtree(tmp_loader_dir, ignore_errors=True)

  def test_compute_per_token_kl_float64_129280_vocab_precision_and_negative_control(self):
    """Verifies host float64 _compute_per_token_kl on 129,280-vocab float32 logits + 1e-6 and 1.5e-3 negative controls."""
    vocab_size = 129280
    rng = np.random.default_rng(20261026)
    row0 = rng.normal(loc=9.55, scale=0.65, size=(vocab_size,)).astype(np.float32)
    row0[0] = np.float32(13.1)
    row1 = np.full((vocab_size,), 1.33, dtype=np.float32)
    row1[0] = np.float32(13.1)
    row2 = rng.normal(loc=9.55, scale=0.65, size=(vocab_size,)).astype(np.float32)
    row2[0] = np.float32(13.1)
    golden = np.stack([row0, row1, row2], axis=0)

    delta0 = 8.0e-5
    train = golden.copy()
    train[0, 1:] = (golden[0, 1:].astype(np.float64) + delta0).astype(np.float32)
    train[1, 1:] = (golden[1, 1:].astype(np.float64) - delta0).astype(np.float32)
    train[2, 1:] = (golden[2, 1:].astype(np.float64) - 0.5 * delta0).astype(np.float32)

    l_inf = float(np.max(np.abs(train - golden)))
    self.assertLessEqual(l_inf, 8.5e-5)
    theoretical_kl_bound = 0.5 * (8.5e-5) ** 2 + 1e-13
    self.assertLess(theoretical_kl_bound, 2.0e-7)

    g64 = golden.astype(np.float64)
    t64 = train.astype(np.float64)
    g_shift = g64 - np.max(g64, axis=-1, keepdims=True)
    t_shift = t64 - np.max(t64, axis=-1, keepdims=True)
    log_Z_row0 = float(np.max(g64[0]) + np.log(np.sum(np.exp(g_shift[0]))))
    self.assertAlmostEqual(log_Z_row0, 21.538, delta=0.1)
    p_g64 = np.exp(g_shift - np.log(np.sum(np.exp(g_shift), axis=-1, keepdims=True)))
    p_t64 = np.exp(t_shift - np.log(np.sum(np.exp(t_shift), axis=-1, keepdims=True)))
    kl_scipy = np.maximum(np.sum(scipy.special.kl_div(p_g64, p_t64), axis=-1, dtype=np.float64), 0.0)

    kl_got = checker._compute_per_token_kl(golden, train)  # pylint: disable=protected-access
    self.assertEqual(kl_got.dtype, np.float64)
    self.assertLess(float(np.max(np.abs(kl_got - kl_scipy))), 1e-14)
    self.assertGreaterEqual(float(np.min(kl_got)), 0.0)
    self.assertGreaterEqual(float(np.mean(kl_got)), 0.0)
    self.assertLessEqual(float(np.max(kl_got)), theoretical_kl_bound)
    self.assertLess(float(np.max(kl_got)), 2.0e-7)

    kl_clipped = checker._compute_per_token_kl(golden, train, clip_logits_epsilon=1e-12)  # pylint: disable=protected-access
    self.assertGreaterEqual(float(np.min(kl_clipped)), 0.0)
    self.assertLessEqual(float(np.max(kl_clipped)), theoretical_kl_bound)

    parity = checker.check_mathematical_parity(train, golden, max_kl=2.0e-7)
    self.assertGreaterEqual(float(parity["max_kl"]), 0.0)
    self.assertLessEqual(float(parity["max_kl"]), theoretical_kl_bound)
    self.assertGreaterEqual(float(parity["mean_kl"]), 0.0)
    for k_base in ("max_kl", "raw_primary_max_kl", "mean_kl", "median_kl", "p95_kl", "p99_kl"):
      self.assertIn(k_base, parity)
      self.assertIn(f"{k_base}_legacy_fp32", parity)
    self.assertGreater(float(parity["max_kl_legacy_fp32"]), 9.0e-7)
    legacy_rows = checker._compute_per_token_kl_legacy_fp32(golden, train)  # pylint: disable=protected-access
    self.assertLess(float(np.min(legacy_rows)), 0.0)

    # Negative Control A (Condition (c)): 1e-6-scale KL perturbation (delta = 2.828e-3 on peak logit)
    # produces float64 max_kl ~ 1.0e-6 (0.5e-6 <= max_kl <= 1.5e-6) and FAILS the 2.0e-7 ceiling.
    p0_row1 = float(p_g64[1, 0])
    delta_1e6 = 2.828e-3
    train_bad_1e6 = golden.copy()
    train_bad_1e6[1, 0] = np.float32(float(golden[1, 0]) + delta_1e6)
    eff_delta_1e6 = float(train_bad_1e6[1, 0]) - float(golden[1, 0])
    expected_kl_1e6 = 0.5 * p0_row1 * (1.0 - p0_row1) * (eff_delta_1e6**2)
    self.assertTrue(0.5e-6 <= expected_kl_1e6 <= 1.5e-6)
    kl_bad_1e6 = checker._compute_per_token_kl(golden, train_bad_1e6)  # pylint: disable=protected-access
    self.assertGreaterEqual(float(np.max(kl_bad_1e6)), 0.5e-6)
    self.assertLessEqual(float(np.max(kl_bad_1e6)), 1.5e-6)
    self.assertAlmostEqual(float(np.max(kl_bad_1e6)), expected_kl_1e6, delta=5e-9)
    with self.assertRaisesRegex(AssertionError, "exceeds threshold"):
      checker.check_mathematical_parity(train_bad_1e6, golden, max_kl=2.0e-7)

    # Negative Control B: Perturbing a single high-probability logit by 1.5e-3 produces max_kl > 2.0e-7 and fails.
    train_bad_15e4 = golden.copy()
    train_bad_15e4[1, 0] = np.float32(float(golden[1, 0]) + 1.5e-3)
    kl_bad_15e4 = checker._compute_per_token_kl(golden, train_bad_15e4)  # pylint: disable=protected-access
    self.assertGreater(float(np.max(kl_bad_15e4)), 2.0e-7)
    with self.assertRaisesRegex(AssertionError, "exceeds threshold"):
      checker.check_mathematical_parity(train_bad_15e4, golden, max_kl=2.0e-7)


def _build_mini_e2e_config_and_args(golden_path: str, ckpt_dir: str, out_dir: str):
  """Builds hermetic pyconfig and CLI args for DeepSeek-V4.1-Flash mini E2E tests."""
  num_devices = len(jax.devices())
  cfg = mm_testing.mini_vision_maxtext_config(
      use_multimodal=False,
      load_parameters_path=ckpt_dir,
      source_checkpoint_layout="safetensors_dynamic",
      scan_layers=True,
      remat_policy="none",
      max_target_length=64,
      per_device_batch_size=2.0,
      ici_fsdp_parallelism=-1,
      ici_expert_parallelism=2 if num_devices >= 2 else 1,
      ici_tensor_parallelism=1,
      dtype="float32",
      weight_dtype="bfloat16",
      matmul_precision="highest",
      sparse_matmul=False,
      capacity_factor=-1,
      ds41_activation_quant=False,
      enable_checkpointing=False,
      base_output_directory=os.path.join(out_dir, "out"),
  )
  test_args = argparse.Namespace(
      atol=None,
      max_kl_div=None,
      min_cos_sim=None,
      golden_logits_path=golden_path,
      mm_golden_dir="",
      max_images_per_sample=4,
      output_json="",
      hf_model_path="",
      run_hf_model=False,
      reference_model="",
      output_logits_path=os.path.join(out_dir, "mini_test_reports.jsonl"),
      gcs_output_logits_path="",
      clip_logits_epsilon=None,
      enforce_length_bucket_gates=True,
      skip_first_token=False,
      ckpt_type="linen",
      trust_remote_code=True,
      token_size=None,
  )
  return cfg, test_args


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "torch not installed")
class MaskedPrefillLogitsCheckerE2ETest(unittest.TestCase):
  """Hermetic positive and negative E2E tests on a DeepSeek-V4.1-Flash mini text checkpoint."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.tmp_dir = tempfile.mkdtemp(prefix="ds41_mini_e2e_")
    cls.ckpt_dir = os.path.join(cls.tmp_dir, "ds41_mini_ckpt")
    os.makedirs(cls.ckpt_dir, exist_ok=True)
    cls.golden_path = os.path.join(cls.tmp_dir, "mini_goldens_3samples.pkl")

    tokenizer = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
    ref_args = multimodal.mini_model_args(vision_n_layers=0)
    ref_model = multimodal.build_multimodal_model(ref_args, tokenizer=tokenizer)
    multimodal.init_mini_reference(ref_model, seed=42)

    hf_tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
      for name, param in ref_model.named_parameters():
        if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
          param.copy_(param.to(torch.float32))
          hf_tensors[name] = param.detach().cpu().to(torch.float32).contiguous()
        else:
          param.copy_(param.to(torch.bfloat16).to(torch.float32))
          hf_tensors[name] = param.detach().cpu().to(torch.bfloat16).contiguous()

    shard_name = "model-00001-of-00001.safetensors"
    save_file(hf_tensors, os.path.join(cls.ckpt_dir, shard_name), metadata={"format": "pt"})
    index_payload = {
        "metadata": {"total_size": sum(int(t.numel() * t.element_size()) for t in hf_tensors.values())},
        "weight_map": {k: shard_name for k in sorted(hf_tensors.keys())},
    }
    with open(os.path.join(cls.ckpt_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
      json.dump(index_payload, f, indent=2)
    with open(os.path.join(cls.ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
      json.dump({"model_type": "deepseek_v41"}, f, indent=2)

    sample_specs = [
        ("sample_0_short", [10, 11, 12, 13, 14, 15, 16, 17], False),
        ("sample_1_medium", [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], False),
        ("sample_6_verbatim_repeat_stress", [30, 31, 32, 33] * 4, True),
    ]
    golden_entries = []
    with torch.no_grad():
      for sample_name, token_ids, is_diag in sample_specs:
        ids_t = torch.tensor([token_ids], dtype=torch.long)
        logits_np = ref_model(ids_t, full_logits=True)[0].detach().cpu().to(torch.float32).numpy()
        golden_entries.append(
            {
                "name": sample_name,
                "prompt": sample_name,
                "tokens": token_ids,
                "logits": logits_np,
                "cos_sim_diagnostic_only": is_diag,
            }
        )
    with open(cls.golden_path, "wb") as f:
      pickle.dump(golden_entries, f)

  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(cls.tmp_dir, ignore_errors=True)

  def test_positive_e2e_ds41_mini(self):
    """Positive E2E check across all 3 mini samples with two compiled runs per sample."""
    cfg, test_args = _build_mini_e2e_config_and_args(self.golden_path, self.ckpt_dir, self.tmp_dir)
    summary_json_path = os.path.join(self.tmp_dir, "mini_summary.json")
    test_args.output_json = summary_json_path
    reports = checker.main(cfg, test_args)
    self.assertIsNotNone(reports)
    self.assertEqual(len(reports), 3)
    self.assertTrue(os.path.exists(test_args.output_logits_path))
    self.assertTrue(os.path.exists(summary_json_path))
    with open(summary_json_path, "r", encoding="utf-8") as f:
      summary_payload = json.load(f)
    for top_kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl", "max_kl_div", "mean_kl_div"):
      self.assertIn(top_kl_key, summary_payload)
      self.assertIn(f"{top_kl_key}_legacy_fp32", summary_payload)

    for report in reports:
      self.assertTrue(report["passed"], f"Sample {report['name']} failed gates: {report['gates']}")
      metrics = report["metrics"]
      for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
        self.assertIn(kl_key, report)
        self.assertIn(f"{kl_key}_legacy_fp32", report)
        self.assertIn(kl_key, metrics)
        self.assertIn(f"{kl_key}_legacy_fp32", metrics)
      self.assertTrue(metrics["rerun_bitwise_identical"])
      self.assertEqual(metrics["rerun_max_abs_diff"], 0.0)
      self.assertTrue(metrics["rerun_argmax_identical"])
      self.assertEqual(metrics["rows_per_batch_shard"], 2)
      self.assertTrue(metrics["intra_shard_checked"])
      self.assertLessEqual(metrics["mean_batch_diff"], 1e-4)

    s6_report = next(r for r in reports if r["name"] == "sample_6_verbatim_repeat_stress")
    self.assertTrue(s6_report["cos_sim_diagnostic_only"])
    self.assertTrue(s6_report["gates"]["mean_cos_sim_pass"])

  def test_e2e_negative_control_run2_perturbation_fails(self):
    """Negative E2E check: perturbing one element on Run 2 causes checker.main to fail."""
    with open(self.golden_path, "rb") as f:
      one_sample = [pickle.load(f)[0]]

    one_sample_path = os.path.join(self.tmp_dir, "one_sample.pkl")
    with open(one_sample_path, "wb") as f:
      pickle.dump(one_sample, f)

    try:
      cfg, test_args = _build_mini_e2e_config_and_args(one_sample_path, self.ckpt_dir, self.tmp_dir)
      orig_gather = checker._gather_and_free_logits  # pylint: disable=protected-access
      call_counter = {"n": 0}

      def _perturbed_gather(raw_logits, seq_len, vocab_size):
        host_logits, rows_per_shard = orig_gather(raw_logits, seq_len, vocab_size)
        call_counter["n"] += 1
        if call_counter["n"] == 2:
          host_logits[0, 0, 0] = np.nextafter(host_logits[0, 0, 0], np.float32(1e6))
        return host_logits, rows_per_shard

      with mock.patch.object(checker, "_gather_and_free_logits", side_effect=_perturbed_gather):
        with self.assertRaisesRegex(AssertionError, "failed masked-prefill gates"):
          checker.main(cfg, test_args)
      self.assertEqual(call_counter["n"], 2)
    finally:
      if os.path.exists(one_sample_path):
        os.remove(one_sample_path)

  def test_multimodal_npz_record_loading_segment_mask_and_ref_topk_validation(self):
    """Tests multimodal .npz loading (`token_types`, `segment_ids`, `ref_topk_experts`) and negative controls."""
    mm_dir = os.path.join(self.tmp_dir, "mm_loader_unit")
    os.makedirs(mm_dir, exist_ok=True)
    ref_topk = np.zeros((14, 8, 6), dtype=np.uint16)
    for layer_idx in range(1, 14):
      ref_topk[layer_idx] = np.broadcast_to(np.arange(6, dtype=np.uint16)[None, :], (8, 6))
    token_types = np.array([-1, 0, 1, 2, 3, -1, -1, -1], dtype=np.int32)
    segment_ids = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int32)
    patches = np.zeros((1, 35, 588), dtype=np.float32)
    image_masks = np.array([1, 0, 0, 0], dtype=np.int32)
    m1_path = os.path.join(mm_dir, "M1.npz")
    np.savez(
        m1_path,
        sample_id=np.array("M1"),
        tokens=np.array([10, 200, 200, 200, 200, 11, 0, 0], dtype=np.int32),
        logits=np.zeros((8, 16), dtype=np.float32),
        token_types=token_types,
        segment_ids=segment_ids,
        patches=patches,
        image_masks=image_masks,
        num_images=np.array(1, dtype=np.int32),
        ref_topk_experts=ref_topk,
    )
    loaded_mm = checker.load_golden_dataset(mm_dir)
    self.assertEqual(len(loaded_mm), 1)
    self.assertEqual(loaded_mm[0]["name"], "M1")
    self.assertEqual(loaded_mm[0]["ref_topk_experts"].dtype, np.uint16)
    self.assertEqual(loaded_mm[0]["ref_topk_experts"].shape, (14, 8, 6))
    self.assertTrue(np.array_equal(loaded_mm[0]["segment_ids"], segment_ids))

    padded = deepseek_v41_frozen_topk.prepare_padded_ref_topk_experts(
        loaded_mm[0]["ref_topk_experts"],
        sample_name="M1",
        num_layers=14,
        seq_len=8,
        padded_seq_len=12,
        batch_size=2,
        top_k=6,
        num_experts=16,
        moe_layer_ids=list(range(1, 14)),
    )
    self.assertEqual(padded.shape, (14, 2, 12, 6))
    self.assertTrue(np.array_equal(padded[1, 0, 8:, :], np.broadcast_to(np.arange(6, dtype=np.int32), (4, 6))))

    # Negative control: wrong seq_len in ref_topk_experts raises ValueError on load.
    bad_dir = os.path.join(self.tmp_dir, "mm_loader_bad_topk")
    os.makedirs(bad_dir, exist_ok=True)
    np.savez(
        os.path.join(bad_dir, "M1.npz"),
        sample_id=np.array("M1"),
        tokens=np.arange(8, dtype=np.int32),
        logits=np.zeros((8, 16), dtype=np.float32),
        token_types=token_types,
        patches=patches,
        image_masks=image_masks,
        ref_topk_experts=np.zeros((14, 7, 6), dtype=np.uint16),
    )
    with self.assertRaises(ValueError):
      checker.load_golden_dataset(bad_dir)

    # Option C (`mm_v3`) loader validation: all 7 fields loaded + fail-closed when require_option_c=True and missing.
    with self.assertRaises(ValueError):
      checker.load_golden_dataset(mm_dir, require_option_c=True)

    mm_v3_dir = os.path.join(self.tmp_dir, "mm_v3_loader_unit")
    os.makedirs(mm_v3_dir, exist_ok=True)
    ref_gate_margins = np.full((14, 8), 2.0e-5, dtype=np.float32)
    ref_gate_margins[0] = 0.0
    ref_gate_ulp = np.full((14, 8), 1.0e-7, dtype=np.float32)
    ref_gate_ulp[0] = 0.0
    ref_idx_layer_ids = np.array([2, 8], dtype=np.int32)
    tpu_idx_topk = np.broadcast_to(np.arange(4, dtype=np.int32)[None, None, :], (2, 8, 4)).copy()
    ref_idx_topk = tpu_idx_topk + 8
    ref_idx_margins = np.full((2, 8), 2.0e-4, dtype=np.float32)
    ref_idx_ulp = np.full((2, 8), 1.0e-6, dtype=np.float32)
    np.savez(
        os.path.join(mm_v3_dir, "M1.npz"),
        sample_id=np.array("M1"),
        tokens=np.array([10, 200, 200, 200, 200, 11, 0, 0], dtype=np.int32),
        logits=np.zeros((8, 16), dtype=np.float32),
        token_types=token_types,
        segment_ids=segment_ids,
        patches=patches,
        image_masks=image_masks,
        num_images=np.array(1, dtype=np.int32),
        ref_topk_experts=ref_topk,
        ref_gate_margins=ref_gate_margins,
        ref_gate_boundary_ulp=ref_gate_ulp,
        ref_indexer_layer_ids=ref_idx_layer_ids,
        ref_indexer_topk=ref_idx_topk,
        ref_indexer_margins=ref_idx_margins,
        ref_indexer_boundary_ulp=ref_idx_ulp,
    )
    loaded_v3 = checker.load_golden_dataset(mm_v3_dir, require_option_c=True)
    self.assertEqual(len(loaded_v3), 1)
    self.assertTrue(deepseek_v41_frozen_topk.has_option_c_routing_fields(loaded_v3[0]))
    ra_pass = deepseek_v41_frozen_topk.check_router_agreement(
        sample_name="M1",
        seq_len=8,
        token_types=token_types,
        tpu_moe_topk=ref_topk,
        ref_topk_experts=ref_topk,
        ref_gate_margins=ref_gate_margins,
        ref_gate_boundary_ulp=ref_gate_ulp,
        tpu_indexer_topk=tpu_idx_topk,
        ref_indexer_layer_ids=ref_idx_layer_ids,
        ref_indexer_topk=ref_idx_topk,
        ref_indexer_margins=ref_idx_margins,
        ref_indexer_boundary_ulp=ref_idx_ulp,
        ulp_multiplier=8.0,
        raise_on_failure=False,
    )
    self.assertTrue(ra_pass["passed"])


class OptionAPrimeTieBranchTest(unittest.TestCase):
  """Unit tests for Option A-prime reference tie branch matching in masked_prefill_logits_checker."""

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()
    self.seq_len = 16
    self.vocab_size = 256
    self.bucket_gates = {
        "max_kl": 0.05,
        "mean_kl": 0.005,
        "top1_agreement": 0.999,
        "p95_kl": 0.01,
        "p99_kl": 0.02,
        "mean_cos_sim": 0.9999,
    }
    self.effective_max_kl = 0.05
    self.effective_min_cos = 0.9999

    np.random.seed(42)
    self.primary_logits = np.random.randn(self.seq_len, self.vocab_size).astype(np.float32)

    # Valid tie branch: diverges from primary by enough to fail primary gate (> 0.05 max_kl)
    self.valid_branch_logits = self.primary_logits.copy()
    self.valid_branch_logits[5, 10:20] += 2.0

    self.valid_branch_npz = os.path.join(self.tmp_dir, "s1_L06_pos562.npz")
    np.savez(
        self.valid_branch_npz,
        tie_flip_logits=self.valid_branch_logits,
        tie_flip_sites=np.array([[6, 562, 158, 107]], dtype=np.int32),
    )

    with open(self.valid_branch_npz, "rb") as fp:
      branch_sha = hashlib.sha256(fp.read()).hexdigest()

    self.manifest = {
        "s1_L06_pos562": {
            "branch_id": "s1_L06_pos562",
            "site_tag": "s1_L06_pos562",
            "sample": "sample_1_prose",
            "layer": 6,
            "pos": 562,
            "file": "s1_L06_pos562.npz",
            "sha256": branch_sha,
            "generator_commit": "deadbeef1234",
            "num_swapped_sites": 1,
            "experts": [158, 107],
            "ref_rank_k_expert": 158,
            "ref_rank_k_plus_1_expert": 107,
            "swapped_out_expert": 158,
            "swapped_in_expert": 107,
            "primary_experts": [158, 1, 2, 3, 4, 5],
            "swapped_experts": [107, 1, 2, 3, 4, 5],
            "gap_ulp": 2.0,
        }
    }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(self.manifest, fp)

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)

  def test_copy_matching_primary_passes(self):
    """B=4 batch where copies match primary; asserts pass and report records matched == 'primary'."""
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run2 = run1.copy()
    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(res["passed"])
    self.assertTrue(res["copy0_primary_pass"])
    self.assertEqual(len(res["per_copy_tie_branch_report"]), 4)
    for report in res["per_copy_tie_branch_report"]:
      self.assertEqual(report["matched"], "primary")
      self.assertTrue(report["passed"])
      self.assertIn("primary_l2", report)
      self.assertIn("primary_l_inf", report)
      self.assertAlmostEqual(report["primary_l2"], 0.0, places=5)
      self.assertAlmostEqual(report["primary_l_inf"], 0.0, places=5)

    parity = checker.check_mathematical_parity(run1[0], self.primary_logits, min_cos_sim=None)
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=res,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(gates["batch_determinism_pass"])
    self.assertTrue(all(gates.values()))

  def test_copy_matching_le_8ulp_branch_passes_and_is_named(self):
    """B=4 batch where copy 1 matches a valid <= 8.0 ULP branch; asserts pass and report names site."""
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1[1] = self.valid_branch_logits
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(res["passed"])
    self.assertTrue(res["copy0_primary_pass"])
    rep1 = res["per_copy_tie_branch_report"][1]
    self.assertEqual(rep1["branch_id"], "s1_L06_pos562")
    self.assertEqual(rep1["matched"], "tie_branch")
    self.assertEqual(rep1["site"], "s1_L06_pos562")
    self.assertEqual(rep1["layer"], 6)
    self.assertEqual(rep1["pos"], 562)
    self.assertEqual(rep1["experts"], [158, 107])
    self.assertEqual(rep1["gap_ulp"], 2.0)
    self.assertEqual(len(rep1["sha256"]), 64)
    self.assertEqual(rep1["generator_commit"], "deadbeef1234")
    self.assertTrue(rep1["passed"])

  def test_copy_matching_neither_fails(self):
    """Copy 1 has arbitrary noise; asserts failure and batch_determinism_pass == False."""
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1[1] = self.primary_logits + 10.0
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res["passed"])
    self.assertEqual(res["per_copy_tie_branch_report"][1]["matched"], "neither")
    self.assertFalse(res["per_copy_tie_branch_report"][1]["passed"])

    parity = checker.check_mathematical_parity(run1[0], self.primary_logits, min_cos_sim=None)
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=res,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(gates["batch_determinism_pass"])

  def test_branch_from_gt_8ulp_site_is_rejected(self):
    """Manifest has branch with gap_ulp = 12.0; asserts branch is rejected at load and copy matching it fails."""
    gt_branch_npz = os.path.join(self.tmp_dir, "s1_L10_pos100.npz")
    gt_branch_logits = self.primary_logits.copy()
    gt_branch_logits[3, 5:15] += 2.0
    np.savez(gt_branch_npz, tie_flip_logits=gt_branch_logits)

    with open(gt_branch_npz, "rb") as fp:
      gt_sha = hashlib.sha256(fp.read()).hexdigest()

    manifest_gt = dict(self.manifest)
    manifest_gt["s1_L10_pos100"] = {
        "site_tag": "s1_L10_pos100",
        "sample": "sample_1_prose",
        "layer": 10,
        "pos": 100,
        "file": "s1_L10_pos100.npz",
        "sha256": gt_sha,
        "generator_commit": "deadbeef1234",
        "num_swapped_sites": 1,
        "experts": [10, 20],
        "gap_ulp": 12.0,  # > 8.0 ULP
    }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(manifest_gt, fp)

    # load_and_validate_tie_branches must reject it
    loaded = checker.load_and_validate_tie_branches(self.tmp_dir, sample_name="sample_1_prose", max_ulp=8.0)
    self.assertNotIn("s1_L10_pos100", loaded)
    self.assertIn("s1_L06_pos562", loaded)

    # A batch with copy 1 matching the gt_8ulp branch must fail
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1[1] = gt_branch_logits
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        tie_branch_max_ulp=8.0,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res["passed"])
    self.assertEqual(res["per_copy_tie_branch_report"][1]["matched"], "neither")

  def test_copy0_on_branch_not_primary_fails(self):
    """All 4 copies on tie branch with 0 on primary reference; asserts failure under any_copy_primary_pass=False."""
    run1 = np.broadcast_to(self.valid_branch_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res["copy0_primary_pass"])
    self.assertFalse(res["any_copy_primary_pass"])
    self.assertTrue(res["all_copies_valid"])
    self.assertFalse(res["tie_branch_batch_gate_passed"])
    self.assertEqual(res["primary_matched_copies"], [])
    self.assertFalse(res["passed"])

    parity = checker.check_mathematical_parity(run1[0], self.primary_logits, min_cos_sim=None)
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=res,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(gates["batch_determinism_pass"])

  def test_copy0_on_branch_with_other_copies_on_primary_passes(self):
    """Sample-1 like B=16 batch: copy 0 & 10 copies on branch, 5 on primary -> PASS."""
    run1 = np.broadcast_to(self.valid_branch_logits[None, :, :], (16, self.seq_len, self.vocab_size)).copy()
    pri_indices = [3, 8, 11, 12, 14]
    for idx in pri_indices:
      run1[idx] = self.primary_logits
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(res["passed"])
    self.assertTrue(res["tie_branch_batch_gate_passed"])
    self.assertTrue(res["any_copy_primary_pass"])
    self.assertTrue(res["all_copies_valid"])
    self.assertFalse(res["copy0_primary_pass"])
    self.assertEqual(res["primary_passed_copies"], pri_indices)
    self.assertEqual(res["primary_matched_copies"], pri_indices)
    self.assertEqual(res["per_copy_tie_branch_report"][0]["matched"], "tie_branch")
    self.assertTrue(res["per_copy_tie_branch_report"][0]["passed"])

    parity = checker.check_mathematical_parity(run1[3], self.primary_logits, min_cos_sim=None)
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=res,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(gates["batch_determinism_pass"])
    self.assertTrue(all(gates.values()))

  def test_all_16_copies_on_branch_none_on_primary_fails(self):
    """All 16 copies on tie branch with primary_passed=False everywhere fails."""
    run1 = np.broadcast_to(self.valid_branch_logits[None, :, :], (16, self.seq_len, self.vocab_size)).copy()
    run2 = run1.copy()

    res_a = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res_a["passed"])
    self.assertFalse(res_a["any_copy_primary_pass"])
    self.assertTrue(res_a["all_copies_valid"])
    self.assertFalse(res_a["tie_branch_batch_gate_passed"])
    self.assertEqual(res_a["primary_passed_copies"], [])
    self.assertEqual(res_a["primary_matched_copies"], [])

    gates_a = checker.evaluate_sample_gates(
        parity_metrics=checker.check_mathematical_parity(run1[0], self.primary_logits, min_cos_sim=None),
        batch_det=res_a,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(gates_a["batch_determinism_pass"])

  def test_sample_2_all_copies_primary_pass_matched_to_tie_branch_passes(self):
    """Sample-2 like B=16 batch: all 16 copies have primary_passed=True, but match 0 ULP tie branch closer.

    Asserts matched="tie_branch" for all 16, primary_passed_copies=[0..15], primary_matched_copies=[],
    any_copy_primary_pass=True, all_copies_valid=True, passed=True.
    """
    small_branch_logits = self.primary_logits.copy()
    small_branch_logits[2, 0:5] += 0.02
    small_branch_npz = os.path.join(self.tmp_dir, "s1_small_branch.npz")
    np.savez(small_branch_npz, tie_flip_logits=small_branch_logits)
    with open(small_branch_npz, "rb") as fp:
      small_sha = hashlib.sha256(fp.read()).hexdigest()

    manifest_small = dict(self.manifest)
    manifest_small["s1_small_branch"] = {
        "site_tag": "s1_small_branch",
        "sample": "sample_1_prose",
        "layer": 6,
        "pos": 2,
        "file": "s1_small_branch.npz",
        "sha256": small_sha,
        "generator_commit": "deadbeef1234",
        "num_swapped_sites": 1,
        "experts": [10, 20],
        "gap_ulp": 0.5,
    }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(manifest_small, fp)

    run1_small = np.broadcast_to(small_branch_logits[None, :, :], (16, self.seq_len, self.vocab_size)).copy()
    run2_small = run1_small.copy()

    res_b = checker.check_batch_slot_determinism(
        run1_small,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2_small,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(res_b["passed"])
    self.assertTrue(res_b["any_copy_primary_pass"])
    self.assertTrue(res_b["all_copies_valid"])
    self.assertTrue(res_b["tie_branch_batch_gate_passed"])
    self.assertEqual(res_b["primary_passed_copies"], list(range(16)))
    self.assertEqual(res_b["primary_matched_copies"], [])
    self.assertEqual(len(res_b["tie_branch_matched_copies"]), 16)

  def test_any_copy_matching_neither_fails(self):
    """Copies 0..14 match primary, copy 15 matches neither -> FAIL."""
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (16, self.seq_len, self.vocab_size)).copy()
    run1[15] = self.primary_logits + 10.0
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res["passed"])
    self.assertTrue(res["any_copy_primary_pass"])
    self.assertFalse(res["all_copies_valid"])
    self.assertFalse(res["tie_branch_batch_gate_passed"])
    self.assertEqual(res["unmatched_copies"], [15])
    self.assertEqual(res["per_copy_tie_branch_report"][15]["matched"], "neither")
    self.assertFalse(res["per_copy_tie_branch_report"][15]["passed"])

  def test_two_flip_copy_fails(self):
    """Compound two-flip copy fails against single-site branches, and manifest num_swapped_sites==2 is rejected."""
    # Part 1: Manifest with num_swapped_sites == 2 is rejected at load time.
    two_flip_npz = os.path.join(self.tmp_dir, "s1_two_flip.npz")
    two_flip_logits = self.primary_logits.copy()
    two_flip_logits[7, :10] += 2.0
    np.savez(two_flip_npz, tie_flip_logits=two_flip_logits)

    with open(two_flip_npz, "rb") as fp:
      two_flip_sha = hashlib.sha256(fp.read()).hexdigest()

    manifest_two_flip = dict(self.manifest)
    manifest_two_flip["s1_two_flip"] = {
        "site_tag": "s1_two_flip",
        "sample": "sample_1_prose",
        "layer": 6,
        "pos": 562,
        "file": "s1_two_flip.npz",
        "sha256": two_flip_sha,
        "generator_commit": "deadbeef1234",
        "num_swapped_sites": 2,  # > 1 flip
        "experts": [158, 107],
        "gap_ulp": 1.5,
    }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(manifest_two_flip, fp)

    loaded = checker.load_and_validate_tie_branches(self.tmp_dir, sample_name="sample_1_prose", max_ulp=8.0)
    self.assertNotIn("s1_two_flip", loaded)

    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1[1] = two_flip_logits
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res["passed"])
    self.assertEqual(res["per_copy_tie_branch_report"][1]["matched"], "neither")

    # Part 2: Two valid single-site branches (A and B) in manifest; a copy with simultaneous (A + B) flips fails.
    branch_b_logits = self.primary_logits.copy()
    branch_b_logits[9, 30:40] += 2.0  # site B perturbation
    branch_b_npz = os.path.join(self.tmp_dir, "s1_L08_pos32.npz")
    np.savez(
        branch_b_npz,
        tie_flip_logits=branch_b_logits,
        tie_flip_sites=np.array([[8, 32, 48, 372]], dtype=np.int32),
    )
    with open(branch_b_npz, "rb") as fp:
      b_sha = hashlib.sha256(fp.read()).hexdigest()

    manifest_both_valid = {
        "s1_L06_pos562": self.manifest["s1_L06_pos562"],
        "s1_L08_pos32": {
            "branch_id": "s1_L08_pos32",
            "site_tag": "s1_L08_pos32",
            "sample": "sample_1_prose",
            "layer": 8,
            "pos": 32,
            "file": "s1_L08_pos32.npz",
            "sha256": b_sha,
            "generator_commit": "deadbeef1234",
            "num_swapped_sites": 1,
            "experts": [48, 372],
            "ref_rank_k_expert": 48,
            "ref_rank_k_plus_1_expert": 372,
            "swapped_out_expert": 48,
            "swapped_in_expert": 372,
            "primary_experts": [48, 1, 2, 3, 4, 5],
            "swapped_experts": [372, 1, 2, 3, 4, 5],
            "gap_ulp": 1.0,
        },
    }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(manifest_both_valid, fp)

    loaded_both = checker.load_and_validate_tie_branches(self.tmp_dir, sample_name="sample_1_prose", max_ulp=8.0)
    self.assertIn("s1_L06_pos562", loaded_both)
    self.assertIn("s1_L08_pos32", loaded_both)

    compound_logits = (
        self.primary_logits + (self.valid_branch_logits - self.primary_logits) + (branch_b_logits - self.primary_logits)
    )
    run1_compound = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1_compound[1] = compound_logits
    run2_compound = run1_compound.copy()

    res_compound = checker.check_batch_slot_determinism(
        run1_compound,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2_compound,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res_compound["passed"])
    self.assertEqual(res_compound["per_copy_tie_branch_report"][1]["matched"], "neither")
    self.assertFalse(res_compound["per_copy_tie_branch_report"][1]["passed"])

  def test_non_opted_in_model_keeps_mean_batch_diff_gate(self):
    """With allow_reference_tie_branches=False, copy on branch with mean_batch_diff > 1e-4 fails."""
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1[1] = self.valid_branch_logits
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=False,
    )
    self.assertFalse(res["passed"])
    self.assertFalse(res["allow_reference_tie_branches"])
    self.assertGreater(res["mean_batch_diff"], 1e-4)

    parity = checker.check_mathematical_parity(run1[0], self.primary_logits, min_cos_sim=None)
    gates = checker.evaluate_sample_gates(
        parity_metrics=parity,
        batch_det=res,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(gates["batch_determinism_pass"])

  def test_tie_branch_alignment_with_non_prefix_pos_mask(self):
    """Verifies check_batch_slot_determinism aligns br['logits'] correctly when non-prefix pos_mask is provided."""
    seq_len = 20
    vocab = self.vocab_size
    rng = np.random.default_rng(42)
    full_primary = rng.normal(size=(seq_len, vocab)).astype(np.float32)

    # valid_mask: True on text tokens [9:16]
    pos_mask = np.zeros(seq_len, dtype=bool)
    pos_mask[9:16] = True

    # Branch logits with perturbation at pos 12 (inside text mask)
    br_logits = full_primary.copy()
    br_logits[12] += 0.3

    with tempfile.TemporaryDirectory() as tmp_dir:
      npz_file = os.path.join(tmp_dir, "s1_L06_pos12.npz")
      np.savez(
          npz_file,
          tie_flip_logits=br_logits,
          tie_flip_sites=np.array([[6, 12, 10, 20]], dtype=np.int32),
      )
      with open(npz_file, "rb") as fp:
        sha = hashlib.sha256(fp.read()).hexdigest()
      manifest = {
          "s1_L06_pos12": {
              "branch_id": "s1_L06_pos12",
              "site_tag": "s1_L06_pos12",
              "sample": "sample_1_prose",
              "layer": 6,
              "pos": 12,
              "primary_exp": 10,
              "alt_exp": 20,
              "ref_rank_k_expert": 10,
              "ref_rank_k_plus_1_expert": 20,
              "swapped_out_expert": 10,
              "swapped_in_expert": 20,
              "primary_experts": [10, 1, 2, 3, 4, 5],
              "swapped_experts": [20, 1, 2, 3, 4, 5],
              "experts": [10, 20],
              "num_swapped_sites": 1,
              "gap_ulp": 1.0,
              "file": "s1_L06_pos12.npz",
              "sha256": sha,
              "generator_commit": "deadbeef1234",
          }
      }
      with open(os.path.join(tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp)

      masked_primary = full_primary[pos_mask]
      masked_branch = br_logits[pos_mask]

      run1 = np.broadcast_to(masked_primary[None, :, :], (2, int(np.sum(pos_mask)), vocab)).copy()
      run1[1] = masked_branch
      run2 = run1.copy()

      bucket = checker.get_length_bucket_thresholds(int(np.sum(pos_mask)))
      res = checker.check_batch_slot_determinism(
          run1,
          seq_len=int(np.sum(pos_mask)),
          min_vocab_size=vocab,
          rows_per_batch_shard=1,
          rerun_train_logits=run2,
          allow_reference_tie_branches=True,
          tie_branch_dir=tmp_dir,
          sample_name="sample_1_prose",
          primary_golden_logits=masked_primary,
          bucket_gates=bucket,
          effective_max_kl=float(bucket["max_kl"]),
          effective_min_cos=float(bucket["mean_cos_sim"]),
          start_index=0,
          token_size=seq_len,
          pos_mask=pos_mask,
      )
      self.assertTrue(res["passed"])
      self.assertEqual(res["per_copy_tie_branch_report"][1]["matched"], "tie_branch")
      self.assertEqual(res["per_copy_tie_branch_report"][1]["branch_id"], "s1_L06_pos12")

  def test_allow_reference_tie_branches_without_gates_raises(self):
    """When allow_reference_tie_branches=True, missing golden logits or gates raises ValueError."""
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (2, self.seq_len, self.vocab_size)).copy()
    run2 = run1.copy()
    with self.assertRaises(ValueError):
      checker.check_batch_slot_determinism(
          run1,
          seq_len=self.seq_len,
          min_vocab_size=self.vocab_size,
          rerun_train_logits=run2,
          allow_reference_tie_branches=True,
          primary_golden_logits=None,
      )

  def test_r23_real_s1_s5_s6_tie_branches(self):
    """Tests using real .npz branch files if DS41_TIE_BRANCH_DIR is set, and always tests hermetic 3-sample fixture."""
    real_dir = os.environ.get("DS41_TIE_BRANCH_DIR", "")
    if real_dir and os.path.exists(os.path.join(real_dir, "tie_branches_manifest.json")):
      loaded_s1 = checker.load_and_validate_tie_branches(real_dir, sample_name="sample_1_prose", max_ulp=8.0)
      self.assertGreater(len(loaded_s1), 0, "Expected at least 1 validated branch for sample_1_prose")
      for br in loaded_s1.values():
        self.assertLessEqual(br["gap_ulp"], 8.0)
        self.assertEqual(len(br["experts"]), 2)
        self.assertTrue(bool(br["generator_commit"]))
        self.assertEqual(len(br["sha256"]), 64)
        self.assertEqual(br["logits"].ndim, 2)

    with tempfile.TemporaryDirectory() as hermetic_dir:
      synth_samples = [
          ("sample_1_prose", "s1_L06_pos562", 6, 562, [257, 173], 1.0),
          ("sample_5_chat_template", "s5_L36_pos560", 36, 560, [287, 196], 0.0),
          ("sample_6_verbatim_repeat_stress", "s6_L32_pos1054", 32, 1054, [112, 105], 1.0),
      ]
      manifest = {}
      for sample_name, site_tag, l, p, exps, gap in synth_samples:
        br_logits = self.primary_logits.copy()
        br_logits[p % self.seq_len, :10] += 0.5
        npz_file = os.path.join(hermetic_dir, f"{site_tag}.npz")
        np.savez(
            npz_file,
            tie_flip_logits=br_logits,
            tie_flip_sites=np.array([[l, p, exps[0], exps[1]]], dtype=np.int32),
        )
        with open(npz_file, "rb") as fp:
          sha = hashlib.sha256(fp.read()).hexdigest()
        manifest[site_tag] = {
            "branch_id": site_tag,
            "site_tag": site_tag,
            "sample": sample_name,
            "layer": l,
            "pos": p,
            "primary_exp": exps[0],
            "alt_exp": exps[1],
            "ref_rank_k_expert": exps[0],
            "ref_rank_k_plus_1_expert": exps[1],
            "swapped_out_expert": exps[0],
            "swapped_in_expert": exps[1],
            "primary_experts": [exps[0], 99, 98, 97, 96, 95],
            "swapped_experts": [exps[1], 99, 98, 97, 96, 95],
            "experts": exps,
            "num_swapped_sites": 1,
            "gap_ulp": gap,
            "file": f"{site_tag}.npz",
            "sha256": sha,
            "generator_commit": "deadbeef1234",
        }
      with open(os.path.join(hermetic_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp)

      for sample_name, site_tag, l, p, exps, gap in synth_samples:
        loaded = checker.load_and_validate_tie_branches(hermetic_dir, sample_name=sample_name, max_ulp=8.0)
        self.assertEqual(len(loaded), 1)
        self.assertIn(site_tag, loaded)
        br = loaded[site_tag]
        self.assertEqual(br["branch_id"], site_tag)
        self.assertEqual(br["experts"], exps)
        self.assertEqual(br["gap_ulp"], gap)
        self.assertEqual(len(br["sha256"]), 64)

        run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
        run1[1] = br["logits"]
        run2 = run1.copy()
        res = checker.check_batch_slot_determinism(
            run1,
            seq_len=self.seq_len,
            min_vocab_size=self.vocab_size,
            rows_per_batch_shard=2,
            rerun_train_logits=run2,
            allow_reference_tie_branches=True,
            tie_branch_dir=hermetic_dir,
            sample_name=sample_name,
            primary_golden_logits=self.primary_logits,
            bucket_gates=self.bucket_gates,
            effective_max_kl=self.effective_max_kl,
            effective_min_cos=self.effective_min_cos,
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["per_copy_tie_branch_report"][1]["matched"], "tie_branch")
        self.assertEqual(res["per_copy_tie_branch_report"][1]["branch_id"], site_tag)

  def test_deepseek41_flash_logical_axis_rules_hashable(self):
    """Verifies base.yml & deepseek4.1-flash.yml nested rules hash cleanly via _freeze_hashable."""

    base_path = os.path.join(os.path.dirname(__file__), "../../src/maxtext/configs/base.yml")
    flash_path = os.path.join(os.path.dirname(__file__), "../../src/maxtext/configs/models/deepseek4.1-flash.yml")
    with open(base_path, "r", encoding="utf-8") as f:
      base_cfg = yaml.safe_load(f)
    with open(flash_path, "r", encoding="utf-8") as f:
      flash_cfg = yaml.safe_load(f)

    base_rules = base_cfg.get("logical_axis_rules", [])
    flash_rules = flash_cfg.get("logical_axis_rules", [])
    # Prove that nested lists exist in the raw rules
    nested_in_base = [r for r in base_rules if isinstance(r, list) and any(isinstance(x, list) for x in r)]
    nested_in_flash = [r for r in flash_rules if isinstance(r, list) and any(isinstance(x, list) for x in r)]
    self.assertGreater(len(nested_in_base) + len(nested_in_flash), 0, "Expected nested lists in logical_axis_rules")

    # Prove raw tuple(rules) fails hash() due to nested list
    with self.assertRaises(TypeError):
      hash(tuple(flash_rules))

    # Prove _freeze_hashable successfully allows hash() computation
    # pylint: disable=protected-access
    frozen_base = deepseek_v41_frozen_topk._freeze_hashable(base_rules)
    frozen_flash = deepseek_v41_frozen_topk._freeze_hashable(flash_rules)
    # pylint: enable=protected-access
    self.assertIsInstance(hash(frozen_base), int)
    self.assertIsInstance(hash(frozen_flash), int)

  def test_main_unhandled_type_error_exits_code_42(self):
    """Verifies that an unhandled TypeError inside checker.main logs [DETERMINISTIC_CODE_ERROR] and exits with code 42."""
    # pylint: disable=protected-access
    with mock.patch.object(checker, "_main_impl", side_effect=TypeError("synthetic nested list unhashable type")):
      # pylint: enable=protected-access
      with self.assertRaises(SystemExit) as cm:
        checker.main(mock.MagicMock(), mock.MagicMock())
      self.assertEqual(cm.exception.code, 42)

  def test_closest_reference_attribution_both_pass_tie_branch_closer(self):
    """Copy 1 passes both primary (max_kl <= 0.05) and tie branch (max_kl=0.0 <= 0.05).

    Verifies matched == 'tie_branch', and both primary and best_branch metrics are recorded.
    """
    closer_branch_npz = os.path.join(self.tmp_dir, "s1_L06_closer.npz")
    closer_branch_logits = self.primary_logits.copy()
    closer_branch_logits[5, 10:20] += 0.5
    np.savez(
        closer_branch_npz,
        tie_flip_logits=closer_branch_logits,
        tie_flip_sites=np.array([[6, 562, 158, 107]], dtype=np.int32),
    )
    with open(closer_branch_npz, "rb") as fp:
      c_sha = hashlib.sha256(fp.read()).hexdigest()

    manifest_closer = dict(self.manifest)
    manifest_closer["s1_L06_closer"] = {
        "branch_id": "s1_L06_closer",
        "site_tag": "s1_L06_closer",
        "sample": "sample_1_prose",
        "layer": 6,
        "pos": 562,
        "file": "s1_L06_closer.npz",
        "sha256": c_sha,
        "generator_commit": "deadbeef1234",
        "num_swapped_sites": 1,
        "experts": [158, 107],
        "ref_rank_k_expert": 158,
        "ref_rank_k_plus_1_expert": 107,
        "swapped_out_expert": 158,
        "swapped_in_expert": 107,
        "primary_experts": [158, 1, 2, 3, 4, 5],
        "swapped_experts": [107, 1, 2, 3, 4, 5],
        "gap_ulp": 2.0,
    }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(manifest_closer, fp)

    run1 = np.broadcast_to(self.primary_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run1[1] = closer_branch_logits.copy()
    run2 = run1.copy()

    gates = dict(self.bucket_gates)
    gates["top1_agreement"] = 0.90
    gates["mean_cos_sim"] = 0.99
    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=0.99,
    )
    self.assertTrue(res["passed"])
    self.assertTrue(res["copy0_primary_pass"])
    rep1 = res["per_copy_tie_branch_report"][1]
    self.assertTrue(rep1["passed"])
    self.assertEqual(rep1["matched"], "tie_branch")
    self.assertTrue(rep1["primary_passed"])
    self.assertTrue(rep1["best_branch_passed"])
    self.assertIn("primary_max_kl", rep1)
    self.assertIn("primary_mean_kl", rep1)
    self.assertIn("primary_l2", rep1)
    self.assertIn("primary_l_inf", rep1)
    self.assertIn("primary_metrics", rep1)
    self.assertIn("best_branch_max_kl", rep1)
    self.assertIn("best_branch_mean_kl", rep1)
    self.assertIn("best_branch_l2", rep1)
    self.assertIn("best_branch_l_inf", rep1)
    self.assertIn("best_branch_metrics", rep1)
    self.assertLess(rep1["best_branch_max_kl"], rep1["primary_max_kl"])

  def test_closest_reference_attribution_copy0_fails_primary_matches_branch(self):
    """Copy 0 fails primary (max_kl > 0.08) but matches tie branch (max_kl < 1e-7).

    Verifies copy0_primary_pass is False, res['passed'] is False, but report truthfully attributes
    matched == 'tie_branch', primary_passed == False, best_branch_passed == True, and records both distances.
    """
    run1 = np.broadcast_to(self.valid_branch_logits[None, :, :], (4, self.seq_len, self.vocab_size)).copy()
    run2 = run1.copy()

    strict_max_kl = 1e-4

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=strict_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertFalse(res["passed"])
    self.assertFalse(res["copy0_primary_pass"])
    self.assertFalse(res["any_copy_primary_pass"])
    self.assertFalse(res["tie_branch_batch_gate_passed"])
    rep0 = res["per_copy_tie_branch_report"][0]
    self.assertTrue(rep0["passed"])
    self.assertFalse(rep0["primary_passed"])
    self.assertTrue(rep0["best_branch_passed"])
    self.assertEqual(rep0["matched"], "tie_branch")
    self.assertGreater(rep0["primary_max_kl"], strict_max_kl)
    self.assertLess(rep0["best_branch_max_kl"], strict_max_kl)
    self.assertIn("primary_metrics", rep0)
    self.assertIn("best_branch_metrics", rep0)
    self.assertIn("primary_l2", rep0)
    self.assertIn("best_branch_l2", rep0)

  def test_fast_screening_equivalence_with_exhaustive_evaluation(self):
    """Tests equivalence between fast screening path and exhaustive evaluate_against_ref."""
    # Create multiple tie branches
    manifest_multi = dict(self.manifest)
    created_branch_logits = {}
    for b_idx in range(5):
      br_logits = self.primary_logits.copy()
      br_logits[b_idx, 0:5] += 0.8
      br_npz = os.path.join(self.tmp_dir, f"br_{b_idx}.npz")
      np.savez(br_npz, tie_flip_logits=br_logits)
      with open(br_npz, "rb") as fp:
        sha = hashlib.sha256(fp.read()).hexdigest()
      b_tag = f"br_{b_idx}"
      created_branch_logits[b_tag] = br_logits
      manifest_multi[b_tag] = {
          "branch_id": b_tag,
          "site_tag": b_tag,
          "sample": "sample_1_prose",
          "layer": 2 + b_idx,
          "pos": b_idx,
          "file": f"br_{b_idx}.npz",
          "sha256": sha,
          "generator_commit": "deadbeef1234",
          "num_swapped_sites": 1,
          "experts": [10, 20],
          "gap_ulp": 1.0 + b_idx,
      }
    with open(os.path.join(self.tmp_dir, "tie_branches_manifest.json"), "w", encoding="utf-8") as fp:
      json.dump(manifest_multi, fp)

    checker._TIE_BRANCH_CACHE.clear()  # pylint: disable=protected-access

    # Construct batch with mixed distinct copies:
    # 0, 1 on primary, 2, 3 on br_0, 4, 5 on br_1, 6, 7 on primary
    run1 = np.broadcast_to(self.primary_logits[None, :, :], (8, self.seq_len, self.vocab_size)).copy()
    run1[2] = created_branch_logits["br_0"]
    run1[3] = created_branch_logits["br_0"]
    run1[4] = created_branch_logits["br_1"]
    run1[5] = created_branch_logits["br_1"]
    run2 = run1.copy()

    res = checker.check_batch_slot_determinism(
        run1,
        seq_len=self.seq_len,
        min_vocab_size=self.vocab_size,
        rows_per_batch_shard=2,
        rerun_train_logits=run2,
        allow_reference_tie_branches=True,
        tie_branch_dir=self.tmp_dir,
        sample_name="sample_1_prose",
        primary_golden_logits=self.primary_logits,
        bucket_gates=self.bucket_gates,
        effective_max_kl=self.effective_max_kl,
        effective_min_cos=self.effective_min_cos,
    )
    self.assertTrue(res["passed"])
    self.assertTrue(res["tie_branch_batch_gate_passed"])
    self.assertIn("primary_passed_copies", res)

    # Exhaustive verification on each copy
    v_branches = checker.load_and_validate_tie_branches(self.tmp_dir, sample_name="sample_1_prose", max_ulp=8.0)
    for c in range(8):
      cand = run1[c]
      pri_pass, pri_metrics = checker.evaluate_against_ref(
          cand,
          self.primary_logits,
          tokens=None,
          bucket_gates=self.bucket_gates,
          effective_max_kl=self.effective_max_kl,
          effective_min_cos=self.effective_min_cos,
      )
      best_exhaustive_key = (True, float("inf"), float("inf"))
      best_exhaustive_id = None
      for br in v_branches.values():
        br_pass, br_metrics = checker.evaluate_against_ref(
            cand,
            br["logits"],
            tokens=None,
            bucket_gates=self.bucket_gates,
            effective_max_kl=self.effective_max_kl,
            effective_min_cos=self.effective_min_cos,
        )
        cur_key = (not bool(br_pass), float(br_metrics["max_kl"]), float(br_metrics.get("l_inf", 0.0)))
        if cur_key < best_exhaustive_key:
          best_exhaustive_key = cur_key
          best_exhaustive_id = br["branch_id"]

      rep = res["per_copy_tie_branch_report"][c]
      self.assertEqual(rep["primary_passed"], bool(pri_pass))
      self.assertAlmostEqual(rep["primary_max_kl"], float(pri_metrics["max_kl"]), places=6)
      if rep["matched"] == "tie_branch":
        self.assertEqual(rep["best_branch_id"], best_exhaustive_id)

  def test_memoization_across_bitwise_identical_copies(self):
    """B=16 with 2 unique trajectories; evaluate_against_ref called only 2*(1+num_branches) times."""
    # 8 copies identical to primary (matching copy 0)
    # 8 copies identical to valid_branch (matching copy 1)
    run1 = np.empty((16, self.seq_len, self.vocab_size), dtype=np.float32)
    for c in range(16):
      if c % 2 == 0:
        run1[c] = self.primary_logits
      else:
        run1[c] = self.valid_branch_logits
    run2 = run1.copy()

    # Track calls to evaluate_against_ref
    original_eval = checker.evaluate_against_ref
    with mock.patch.object(checker, "evaluate_against_ref", wraps=original_eval) as mock_eval:
      res = checker.check_batch_slot_determinism(
          run1,
          seq_len=self.seq_len,
          min_vocab_size=self.vocab_size,
          rows_per_batch_shard=2,
          rerun_train_logits=run2,
          allow_reference_tie_branches=True,
          tie_branch_dir=self.tmp_dir,
          sample_name="sample_1_prose",
          primary_golden_logits=self.primary_logits,
          bucket_gates=self.bucket_gates,
          effective_max_kl=self.effective_max_kl,
          effective_min_cos=self.effective_min_cos,
      )

      # 1 primary + 1 branch = 2 references.
      # 2 unique copy trajectories across B=16 => 2 * 2 = 4 calls.
      expected_calls = 2 * (1 + res["tie_branches_evaluated_count"])
      self.assertEqual(mock_eval.call_count, expected_calls)

    self.assertTrue(res["passed"])
    self.assertEqual(len(res["per_copy_tie_branch_report"]), 16)
    for c, rep in enumerate(res["per_copy_tie_branch_report"]):
      if c % 2 == 0:
        self.assertEqual(rep["matched"], "primary")
        self.assertTrue(rep["passed"])
      else:
        self.assertEqual(rep["matched"], "tie_branch")
        self.assertEqual(rep["branch_id"], "s1_L06_pos562")
        self.assertTrue(rep["passed"])
      self.assertIn("primary_l2", rep)
      self.assertIn("best_branch_l2", rep)


class PreemptionResumeHarnessTest(unittest.TestCase):
  """Unit tests for preemption resume semantics in masked_prefill_logits_checker."""

  def setUp(self):
    super().setUp()
    checker._TIE_BRANCH_CACHE.clear()  # pylint: disable=protected-access

  def test_resume_banked_failed_sample_evaluates_multimodal_and_exits_42(self):
    """Resume with banked failed sample evaluates unbanked multimodal samples and exits 42."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      samples_dir = Path(tmp_dir) / "samples"
      samples_dir.mkdir(parents=True, exist_ok=True)
      summary_json = os.path.join(tmp_dir, "summary.json")

      banking.write_banked_sample(
          sample_id="sample_1",
          stage_name="r3",
          status=banking.STATUS_FAILED,
          local_samples_dir=samples_dir,
          metrics={"name": "sample_1", "max_kl": 0.5, "passed": False},
          attempt_id="attempt_0",
          pod="pod-prior-attempt",
      )

      rng = np.random.RandomState(42)
      m1_logits = rng.randn(4, 32).astype(np.float32)

      sample_1_point = {
          "name": "sample_1",
          "input_tokens": np.array([1, 2, 3, 4], dtype=np.int32),
          "golden_logits": np.ones((4, 32), dtype=np.float32),
      }
      m1_point = {
          "name": "M1",
          "input_tokens": np.array([1, 2, 3, 4], dtype=np.int32),
          "golden_logits": m1_logits,
          "token_types": np.array([-1, 0, 0, 0], dtype=np.int32),
          "encoder_images": np.zeros((1, 4, 32), dtype=np.float32),
      }

      test_args = argparse.Namespace(
          atol=None,
          max_kl_div=0.08,
          min_cos_sim=0.999,
          token_size=4,
          golden_logits_path="",
          mm_golden_dir="",
          max_images_per_sample=4,
          output_json=summary_json,
          samples_dir=str(samples_dir),
          stage_name="r3",
          gcs_results_path="",
          hf_model_path="",
          run_hf_model=False,
          reference_model="",
          output_logits_path="",
          gcs_output_logits_path="",
          clip_logits_epsilon=None,
          enforce_length_bucket_gates=True,
          skip_first_token=False,
          ckpt_type="linen",
          trust_remote_code=True,
          allow_reference_tie_branches=False,
          tie_branch_dir="",
          tie_branch_max_ulp=8.0,
          require_frozen_topk_gate=False,
          router_agreement_ulp_multiplier=1.0,
      )

      cfg = mock.MagicMock()
      cfg.model_name = "deepseek4.1-flash"
      cfg.use_multimodal = False
      cfg.logical_axis_rules = []
      cfg.dtype = "bfloat16"
      cfg.init_weights_seed = 0
      cfg.hf_access_token = None
      cfg.tokenizer_path = ""
      cfg.mesh_axes = ("data", "model")
      cfg.vocab_size = 32
      cfg.lora.enable_lora = False

      evaluated_samples = []

      def fake_get_data(golden_data_point, config, max_images_per_sample=4):
        del config, max_images_per_sample
        sname = golden_data_point["name"]
        evaluated_samples.append(sname)
        return (
            np.zeros((1, 4), dtype=np.int32),
            np.ones((1, 4), dtype=np.int32),
            np.arange(4, dtype=np.int32)[None, :],
            golden_data_point["golden_logits"],
            4,
            None,
            None,
        )

      with (
          mock.patch.object(checker, "load_golden_dataset", return_value=[sample_1_point, m1_point]),
          mock.patch.object(checker, "get_data", side_effect=fake_get_data),
          mock.patch("maxtext.utils.maxtext_utils.create_device_mesh", return_value=np.array([[jax.devices()[0]]])),
          mock.patch("jax.sharding.Mesh", return_value=mock.MagicMock()),
          mock.patch("maxtext.utils.model_creation_utils.from_pretrained", return_value=mock.MagicMock()),
          mock.patch("flax.nnx.split", return_value=(mock.MagicMock(), mock.MagicMock())),
          mock.patch(
              "flax.nnx.merge",
              return_value=mock.MagicMock(
                  side_effect=lambda *args, **kwargs: jnp.asarray(m1_logits[None, :, :], dtype=jnp.float32)
              ),
          ),
          mock.patch("jax.jit", side_effect=lambda f, *args, **kwargs: f),
      ):
        with self.assertRaises(SystemExit) as cm:
          try:
            checker.main(cfg, test_args)
          except AssertionError as e:
            if "failed masked-prefill gates" in str(e):
              sys.exit(42)
            raise
        self.assertEqual(cm.exception.code, 42)

      self.assertNotIn("sample_1", evaluated_samples)
      self.assertIn("M1", evaluated_samples)
      m1_banked_file = samples_dir / "r3_M1.json"
      self.assertTrue(m1_banked_file.exists(), f"Expected {m1_banked_file} to be banked on disk")
      with open(m1_banked_file, "r", encoding="utf-8") as f:
        m1_data = json.load(f)
      self.assertEqual(m1_data["status"], "passed")
      self.assertEqual(m1_data["sample_id"], "M1")

      self.assertTrue(os.path.exists(summary_json), "Expected summary.json to be written")
      with open(summary_json, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
      sample_names_in_summary = [s["name"] for s in summary_data["samples"]]
      self.assertIn("sample_1", sample_names_in_summary)
      self.assertIn("M1", sample_names_in_summary)
      self.assertEqual(len(summary_data["samples"]), 2)

  def test_check_banked_sample_provenance_rejection(self):
    """Verifies check_banked_sample rejects mismatched commit_sha or overlay_sha256 as corrupt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      samples_dir = Path(tmp_dir) / "samples"
      samples_dir.mkdir(parents=True, exist_ok=True)

      banking.write_banked_sample(
          sample_id="sample_test",
          stage_name="r3",
          status=banking.STATUS_PASSED,
          local_samples_dir=samples_dir,
          metrics={"name": "sample_test", "max_kl": 0.0001, "passed": True},
          commit_sha="74507c0e2deadbeef",
          overlay_sha256="0197f499f8b3d43c79d64e23471ad236ec2a6407a2d54a43ed9d5f5bac1e62b8",
          config_hash="r3_config_hash",
      )

      # Mismatched commit
      status, data, msg = banking.check_banked_sample(
          sample_id="sample_test",
          stage_name="r3",
          local_samples_dir=samples_dir,
          expected_commit_sha="different_commit_sha",
          expected_overlay_sha256="0197f499f8b3d43c79d64e23471ad236ec2a6407a2d54a43ed9d5f5bac1e62b8",
      )
      self.assertEqual(status, banking.STATUS_CORRUPT)
      self.assertIsNone(data)
      self.assertIn("commit_sha mismatch", msg)

      # Mismatched overlay
      status, data, msg = banking.check_banked_sample(
          sample_id="sample_test",
          stage_name="r3",
          local_samples_dir=samples_dir,
          expected_commit_sha="74507c0e2deadbeef",
          expected_overlay_sha256="different_overlay_sha",
      )
      self.assertEqual(status, banking.STATUS_CORRUPT)
      self.assertIsNone(data)
      self.assertIn("overlay_sha256 mismatch", msg)

      # Staged real file test (if present)
      staged_sample_2 = Path("/tmp/r23_aprime_b_stage/samples/r3_sample_2_code_json_math.json")
      if staged_sample_2.exists():
        staged_dir = staged_sample_2.parent
        status, data, msg = banking.check_banked_sample(
            sample_id="sample_2_code_json_math",
            stage_name="r3",
            local_samples_dir=staged_dir,
            expected_commit_sha="new_candidate_commit",
            expected_overlay_sha256="new_candidate_overlay",
        )
        self.assertEqual(status, banking.STATUS_CORRUPT)
        self.assertIsNone(data)
        self.assertIn("mismatch", msg)


if __name__ == "__main__":
  unittest.main()
