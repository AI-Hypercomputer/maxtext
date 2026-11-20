# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the torch-free m3 parity, negative-control, and round-trip helpers."""

import hashlib
import unittest
import warnings

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from tests.unit.m3.testing import ckpt_roundtrip_check
from tests.unit.m3.testing import e2e_check
from tests.unit.m3.testing import layer_parity
from tests.unit.m3.testing import negctl

# Bit fingerprint of `hashed_normal("probe", (1024,), mean=0, std=1)`; changes here would silently invalidate every
# committed golden archive whose inputs are rebuilt from hashed keys.
_PROBE_SHA256 = "83de743ba5b5f8a992cc63ffac7dab02e707dc5484cf849af90230588bb9259a"
_TIGHT = layer_parity.Tolerance(atol=1e-6, rtol=1e-6, frob_rel=1e-6)
_EXACT_DTYPES = ("bfloat16", "float32", "int32", "int64")


def _linear_fn(params, inputs):
  return {"y": inputs["x"] @ params["w"] + params["b"]}


def _linear_reference(w: np.ndarray, b: np.ndarray, x: np.ndarray, cot: np.ndarray):
  """Closed-form outputs and gradients of y = x @ w + b under loss sum(y * cot)."""
  y = x @ w + b
  return {"y": y}, {"w": x.T @ cot, "b": cot.sum(axis=0), "x": cot @ w.T}


class HashedNormalTest(unittest.TestCase):
  """Verifies cross-platform bit stability and moments of `hashed_normal`."""

  def test_bits_are_pinned_and_deterministic(self):
    a = layer_parity.hashed_normal("probe", (1024,), mean=0.0, std=1.0)
    b = layer_parity.hashed_normal("probe", (1024,), mean=0.0, std=1.0)
    self.assertEqual(a.dtype, np.float32)
    np.testing.assert_array_equal(a, b)
    self.assertEqual(hashlib.sha256(a.tobytes()).hexdigest(), _PROBE_SHA256)

  def test_keys_shape_and_moments(self):
    a = layer_parity.hashed_normal("k1", (64, 512), mean=0.0, std=1.0)
    b = layer_parity.hashed_normal("k2", (64, 512), mean=0.0, std=1.0)
    self.assertEqual(a.shape, (64, 512))
    self.assertGreater(float(np.mean(np.abs(a - b))), 0.5)
    self.assertLess(abs(float(a.mean())), 0.02)
    self.assertLess(abs(float(a.std()) - 1.0), 0.02)
    c = layer_parity.hashed_normal("k1", (64, 512), mean=1.0, std=0.1)
    np.testing.assert_allclose(c, 1.0 + 0.1 * a, rtol=1e-6, atol=1e-6)


class CompareTest(unittest.TestCase):
  """Verifies `layer_parity.compare` thresholds, shape, non-finite, and empty guards."""

  def test_pass_fail_shape_empty_and_nonfinite(self):
    ref = np.linspace(-2.0, 2.0, 32).reshape(4, 8)
    ok, metrics = layer_parity.compare(ref + 1e-8, ref, _TIGHT)
    self.assertTrue(ok)
    self.assertLess(metrics.max_abs, 1e-7)
    ok, _ = layer_parity.compare(ref + 1e-3, ref, _TIGHT)
    self.assertFalse(ok)
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      layer_parity.compare(ref[:2], ref, _TIGHT)
    with self.assertRaisesRegex(ValueError, "zero-size"):
      layer_parity.compare(np.zeros((2, 0)), np.zeros((2, 0)), _TIGHT)
    bad = ref.copy()
    bad[0, 0] = np.nan
    ok, metrics = layer_parity.compare(bad, ref, _TIGHT)
    self.assertFalse(ok)
    self.assertFalse(metrics.finite)


class NegctlTest(unittest.TestCase):
  """Verifies single-leaf perturbation and band/dtype/zero-underflow guards."""

  def test_perturbation_touches_one_leaf_at_requested_scale(self):
    params = {"a": jnp.ones((4, 4)), "b": jnp.arange(8.0)}
    out = negctl.perturb_one_leaf(params, target_path="b", rel_scale=1e-2, seed=0)
    np.testing.assert_array_equal(out["a"], params["a"])
    rel = float(jnp.linalg.norm(out["b"] - params["b"]) / jnp.linalg.norm(params["b"]))
    self.assertAlmostEqual(rel, 1e-2, places=4)

  def test_rejects_invalid_controls(self):
    params = {"a": jnp.ones((4,)), "z": jnp.zeros((4,)), "i": jnp.arange(4)}
    with self.assertRaisesRegex(ValueError, "outside"):
      negctl.perturb_one_leaf(params, target_path="a", rel_scale=0.5, seed=0)
    with self.assertRaisesRegex(ValueError, "matched 0"):
      negctl.perturb_one_leaf(params, target_path="missing", rel_scale=1e-2, seed=0)
    with self.assertRaisesRegex(ValueError, "all zeros"):
      negctl.perturb_one_leaf(params, target_path="z", rel_scale=1e-2, seed=0)
    with self.assertRaisesRegex(ValueError, "non-float"):
      negctl.perturb_one_leaf(params, target_path="i", rel_scale=1e-2, seed=0)
    # The smallest positive float16 subnormal has 0 mantissa bits (ULP == value), so a 0.5% relative step rounds back
    # to the original bit pattern and must be rejected as a silent no-op control.
    tiny_f16 = np.array([np.finfo(np.float16).smallest_subnormal], dtype=np.float16)
    with self.assertRaisesRegex(ValueError, "rounds away"):
      negctl.scaled_perturbation(tiny_f16, name="f16", rel_scale=5e-3, seed=0)

  def test_leaf_path_renders_nested_keys(self):
    tree = {"layers": [{"w": jnp.ones(2)}]}
    paths = [negctl.leaf_path(p) for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]
    self.assertEqual(paths, ["layers/0/w"])


class CheckForwardAndGradsTest(unittest.TestCase):
  """Verifies `check_forward_and_grads` and `negative_control_fires` across fp32, fp64, and failure modes."""

  def setUp(self):
    super().setUp()
    self.w = layer_parity.hashed_normal("lin/w", (8, 3), mean=0.0, std=1.0)
    self.b = layer_parity.hashed_normal("lin/b", (3,), mean=0.0, std=1.0)
    self.x = layer_parity.hashed_normal("lin/x", (5, 8), mean=0.0, std=1.0)
    self.cot = {"y": layer_parity.hashed_normal("lin/cot", (5, 3), mean=0.0, std=1.0)}
    self.ref_out, self.ref_grads = _linear_reference(
        self.w.astype(np.float64), self.b.astype(np.float64), self.x.astype(np.float64), self.cot["y"]
    )
    self.params = {"w": jnp.asarray(self.w), "b": jnp.asarray(self.b)}
    self.inputs = {"x": jnp.asarray(self.x)}
    self.grad_map = {"w": "params/w", "b": "params/b", "x": "inputs/x"}

  def _check(self, ref_grads, grad_map=None):
    return layer_parity.check_forward_and_grads(
        cand_fn=_linear_fn,
        cand_params=self.params,
        cand_inputs=self.inputs,
        ref_outputs=self.ref_out,
        cotangents=self.cot,
        ref_grads=ref_grads,
        grad_map=self.grad_map if grad_map is None else grad_map,
        fwd_tol=_TIGHT,
        grad_tol=_TIGHT,
    )

  def test_passes_and_catches_wrong_grad_and_incomplete_grad_map(self):
    report = self._check(self.ref_grads)
    self.assertTrue(report.passed, report.failures)
    wrong = dict(self.ref_grads, b=self.ref_grads["b"] + 1e-2)
    report = self._check(wrong)
    self.assertFalse(report.passed)
    self.assertTrue(any("grad b" in f for f in report.failures))
    transposed_ref = dict(self.ref_grads, w=self.ref_grads["w"].T)
    report = self._check(transposed_ref, {"w": "params/w:T", "b": "params/b", "x": "inputs/x"})
    self.assertTrue(report.passed, report.failures)
    with self.assertRaisesRegex(ValueError, "cover every gradient"):
      self._check(self.ref_grads, {"w": "params/w", "b": "params/b"})

  def test_preserves_float64_cotangent_precision(self):
    with jax.enable_x64(True):
      w64 = layer_parity.hashed_normal("lin64/w", (8, 3), mean=0.0, std=1.0, dtype=np.float64)
      b64 = layer_parity.hashed_normal("lin64/b", (3,), mean=0.0, std=1.0, dtype=np.float64)
      x64 = layer_parity.hashed_normal("lin64/x", (5, 8), mean=0.0, std=1.0, dtype=np.float64)
      cot64 = {"y": layer_parity.hashed_normal("lin64/cot", (5, 3), mean=0.0, std=1.0, dtype=np.float64)}
      ref_out, ref_grads = _linear_reference(w64, b64, x64, cot64["y"])
      tol64 = layer_parity.Tolerance(atol=1e-12, rtol=1e-12, frob_rel=1e-12)
      report = layer_parity.check_forward_and_grads(
          cand_fn=_linear_fn,
          cand_params={"w": jnp.asarray(w64), "b": jnp.asarray(b64)},
          cand_inputs={"x": jnp.asarray(x64)},
          ref_outputs=ref_out,
          cotangents=cot64,
          ref_grads=ref_grads,
          grad_map=self.grad_map,
          fwd_tol=tol64,
          grad_tol=tol64,
      )
      self.assertTrue(report.passed, report.failures)

  def test_rejects_numpy_candidate_leaves_and_empty_or_mismatched_outputs(self):
    with self.assertRaisesRegex(TypeError, "not jax.Array"):
      layer_parity.check_forward_and_grads(
          cand_fn=_linear_fn,
          cand_params={"w": self.w, "b": self.b},
          cand_inputs=self.inputs,
          ref_outputs=self.ref_out,
          cotangents=self.cot,
          ref_grads=self.ref_grads,
          grad_map=self.grad_map,
          fwd_tol=_TIGHT,
          grad_tol=_TIGHT,
      )
    with self.assertRaisesRegex(ValueError, "ref_outputs is empty"):
      layer_parity.check_forward_and_grads(
          cand_fn=lambda p, inp: {},
          cand_params={},
          cand_inputs={},
          ref_outputs={},
          cotangents={},
          ref_grads={},
          grad_map={},
          fwd_tol=_TIGHT,
          grad_tol=_TIGHT,
      )
    with self.assertRaisesRegex(ValueError, "ref_outputs is empty"):
      layer_parity.negative_control_fires(
          cand_fn=_linear_fn,
          cand_params=self.params,
          cand_inputs=self.inputs,
          ref_outputs={},
          fwd_tol=_TIGHT,
          target_path="w",
          rel_scale=5e-3,
          seed=1,
      )
    with self.assertRaisesRegex(TypeError, "not jax.Array"):
      layer_parity.negative_control_fires(
          cand_fn=_linear_fn,
          cand_params={"w": self.w, "b": self.b},
          cand_inputs=self.inputs,
          ref_outputs=self.ref_out,
          fwd_tol=_TIGHT,
          target_path="w",
          rel_scale=5e-3,
          seed=1,
      )
    with self.assertRaisesRegex(ValueError, "Output names differ"):
      layer_parity.negative_control_fires(
          cand_fn=lambda p, inp: {"y": _linear_fn(p, inp)["y"], "extra": jnp.zeros((1,))},
          cand_params=self.params,
          cand_inputs=self.inputs,
          ref_outputs=self.ref_out,
          fwd_tol=_TIGHT,
          target_path="w",
          rel_scale=5e-3,
          seed=1,
      )

  def test_negative_control_fires_and_rejects_meaningless_controls(self):
    fired, metrics = layer_parity.negative_control_fires(
        cand_fn=_linear_fn,
        cand_params=self.params,
        cand_inputs=self.inputs,
        ref_outputs=self.ref_out,
        fwd_tol=_TIGHT,
        target_path="w",
        rel_scale=5e-3,
        seed=1,
    )
    self.assertTrue(fired)
    self.assertGreater(metrics["y"].max_abs, 1e-4)

    with self.assertRaisesRegex(ValueError, "Unperturbed candidate already fails"):
      layer_parity.negative_control_fires(
          cand_fn=_linear_fn,
          cand_params={"w": self.params["w"] + 1.0, "b": self.params["b"]},
          cand_inputs=self.inputs,
          ref_outputs=self.ref_out,
          fwd_tol=_TIGHT,
          target_path="w",
          rel_scale=5e-3,
          seed=1,
      )

    with self.assertRaisesRegex(ValueError, "unchanged"):
      layer_parity.negative_control_fires(
          cand_fn=lambda p, inp: {"y": inp["x"] @ p["w"] + self.params["b"]},
          cand_params={"w": self.params["w"], "unused": jnp.ones((4,))},
          cand_inputs=self.inputs,
          ref_outputs=self.ref_out,
          fwd_tol=_TIGHT,
          target_path="unused",
          rel_scale=5e-3,
          seed=1,
      )

    def _nan_on_perturbation(p, inp):
      moved = jnp.max(jnp.abs(p["w"] - self.params["w"])) > 0
      y = inp["x"] @ p["w"] + p["b"]
      return {"y": jnp.where(moved, jnp.nan, y)}

    with self.assertRaisesRegex(ValueError, "not finite"):
      layer_parity.negative_control_fires(
          cand_fn=_nan_on_perturbation,
          cand_params=self.params,
          cand_inputs=self.inputs,
          ref_outputs=self.ref_out,
          fwd_tol=_TIGHT,
          target_path="w",
          rel_scale=5e-3,
          seed=1,
      )


class E2ECheckTest(unittest.TestCase):
  """Verifies logits, token, allclose, negative-control, and router-agreement checks."""

  def setUp(self):
    super().setUp()
    self.golden = {"logits": layer_parity.hashed_normal("e2e/logits", (2, 6, 32), mean=0.0, std=3.0)}
    self.tol = e2e_check.LogitsTol(kl_mean_max=1e-6, top1_min=1.0)

  def test_logits_and_negative_run_status(self):
    primary = {"logits": self.golden["logits"] + 1e-6}
    ok, metrics, _ = e2e_check.check_outputs(self.golden, primary, ["logits"], self.tol)
    self.assertTrue(ok)
    self.assertLess(metrics["logits"]["kl_mean"], 1e-9)
    perturbed = {"logits": self.golden["logits"] * 1.05}
    self.assertEqual(e2e_check.negative_run_status(self.golden, primary, perturbed, ["logits"], self.tol)[0], "FIRED")
    tiny = {"logits": self.golden["logits"] + 2e-6}
    self.assertEqual(e2e_check.negative_run_status(self.golden, primary, tiny, ["logits"], self.tol)[0], "MISSED")
    self.assertEqual(e2e_check.negative_run_status(self.golden, primary, primary, ["logits"], self.tol)[0], "ERROR")
    bad_primary = {"logits": self.golden["logits"] * -1.0}
    self.assertEqual(e2e_check.negative_run_status(self.golden, bad_primary, perturbed, ["logits"], self.tol)[0], "ERROR")
    nan_perturbed = {"logits": np.full_like(self.golden["logits"], np.nan)}
    self.assertEqual(e2e_check.negative_run_status(self.golden, primary, nan_perturbed, ["logits"], self.tol)[0], "ERROR")
    # bfloat16 (`ml_dtypes.bfloat16` / `jnp.bfloat16`) is not a subtype of `np.floating`; verify NaN/Inf still return ERROR.
    bf16_golden = {"logits": jnp.asarray(self.golden["logits"], dtype=jnp.bfloat16)}
    bf16_tol = e2e_check.LogitsTol(kl_mean_max=1e-2, top1_min=0.9)
    for nonfinite in (jnp.nan, jnp.inf):
      bf16_bad = {"logits": jnp.full_like(bf16_golden["logits"], nonfinite)}
      self.assertEqual(
          e2e_check.negative_run_status(bf16_golden, bf16_golden, bf16_bad, ["logits"], bf16_tol)[0], "ERROR"
      )
      self.assertEqual(
          e2e_check.negative_run_status(
              bf16_golden, bf16_golden, bf16_bad, ["logits"], layer_parity.Tolerance(atol=1e-2, rtol=1e-2, frob_rel=1e-2)
          )[0],
          "ERROR",
      )
    with self.assertRaises(KeyError):
      e2e_check.check_outputs(self.golden, primary, ["missing"], self.tol)

  def test_jax_array_inputs_preserve_float64_precision_without_x64(self):
    with jax.enable_x64(False), warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      jax_cand = jnp.asarray(self.golden["logits"], dtype=jnp.float32) + jnp.float32(1e-7)
      ok, metrics = e2e_check.compare_one(self.golden["logits"], jax_cand, self.tol)
      self.assertTrue(ok)
      self.assertGreaterEqual(metrics["kl_mean"], 0.0)
      _, np_metrics = e2e_check.compare_one(self.golden["logits"], np.asarray(jax_cand), self.tol)
      self.assertAlmostEqual(metrics["kl_mean"], np_metrics["kl_mean"], places=15)
      router_g = np.array([[2.0, 1.0, 0.0, -1.0], [3.0, 1.0, 0.0, -2.0]], dtype=np.float32)
      router_c = jnp.asarray(router_g) + jnp.float32(1e-7)
      r_tol = e2e_check.RouterAgreementTol(top_k=1, min_agreement=1.0, tie_margin_ulp=8.0)
      r_ok, r_metrics = e2e_check.check_router_agreement(router_g, router_c, r_tol)
      self.assertTrue(r_ok)
      self.assertGreater(r_metrics["max_abs_diff"], 0.0)
      x64_warnings = [w for w in caught if "truncated to dtype float32" in str(w.message)]
      self.assertEqual(x64_warnings, [])

  def test_rejects_empty_arrays_and_invalid_router_tolerances(self):
    with self.assertRaisesRegex(ValueError, "zero-size"):
      e2e_check.compare_one(np.zeros((0, 8), dtype=np.float32), np.zeros((0, 8), dtype=np.float32), self.tol)
    with self.assertRaisesRegex(ValueError, "zero-size"):
      e2e_check.compare_one(
          np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), e2e_check.TokensTol(match_min=1.0)
      )
    with self.assertRaisesRegex(ValueError, "at least 1 dimension"):
      e2e_check.compare_one(np.float32(1.0), np.float32(1.0), self.tol)
    scores = np.ones((2, 4), dtype=np.float32)
    for bad_k in (0, -1):
      with self.assertRaisesRegex(ValueError, "top_k must be >= 1"):
        e2e_check.check_router_agreement(
            scores, scores, e2e_check.RouterAgreementTol(top_k=bad_k, min_agreement=1.0, tie_margin_ulp=8.0)
        )
    with self.assertRaisesRegex(ValueError, "tie_margin_ulp must be >= 0"):
      e2e_check.check_router_agreement(
          scores, scores, e2e_check.RouterAgreementTol(top_k=1, min_agreement=1.0, tie_margin_ulp=-1.0)
      )
    with self.assertRaisesRegex(ValueError, "zero-size"):
      e2e_check.check_router_agreement(
          np.zeros((0, 4), dtype=np.float32),
          np.zeros((0, 4), dtype=np.float32),
          e2e_check.RouterAgreementTol(top_k=1, min_agreement=1.0, tie_margin_ulp=8.0),
      )

  def test_tokens_and_allclose_modes(self):
    tok_golden = np.array([[1, 2, 3, 4]], dtype=np.int32)
    tok_cand = np.array([[1, 2, 3, 9]], dtype=np.int32)
    ok, m = e2e_check.compare_one(tok_golden, tok_cand, e2e_check.TokensTol(match_min=0.75))
    self.assertTrue(ok)
    self.assertEqual(m["match"], 0.75)
    ok, _ = e2e_check.compare_one(tok_golden, tok_cand, e2e_check.TokensTol(match_min=1.0))
    self.assertFalse(ok)
    with self.assertRaisesRegex(TypeError, "integer"):
      e2e_check.compare_one(tok_golden.astype(np.float32), tok_cand.astype(np.float32), e2e_check.TokensTol(1.0))

    lat_g = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    ok, m = e2e_check.compare_one(lat_g, lat_g + 1e-7, _TIGHT)
    self.assertTrue(ok)
    self.assertLess(m["max_abs"], 1e-6)
    ok, _ = e2e_check.compare_one(lat_g, lat_g + 1e-2, _TIGHT)
    self.assertFalse(ok)

  def test_router_agreement_ignores_near_ties_and_rejects_all_ties(self):
    # Sub-unit scores (1e-3 vs 1e-3 - 5e-7) are ~6800 ULPs apart at 1e-3 (spacing ~ 5.96e-11), so a real bug on small
    # scores is evaluated rather than masked by a float32-eps floor.
    golden = np.array([[1e-3, 1e-3 - 5e-7, 0.0, -1.0], [2.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    cand_tie_flip = np.array([[1e-3, 1e-3 - 5e-7, 0.0, -1.0], [2.0, 2.0, 0.0, 1.0]], dtype=np.float32)
    tol = e2e_check.RouterAgreementTol(top_k=1, min_agreement=1.0, tie_margin_ulp=8.0)
    ok, metrics = e2e_check.check_router_agreement(golden, cand_tie_flip, tol)
    self.assertTrue(ok)
    self.assertEqual(metrics["non_tie_fraction"], 0.5)
    cand_sub_unit_flip = np.array([[1e-3 - 5e-7, 1e-3, 0.0, -1.0], [2.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    ok, _ = e2e_check.check_router_agreement(golden, cand_sub_unit_flip, tol)
    self.assertFalse(ok)

    all_ties = np.zeros((3, 4), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "non_tie_fraction=0.0"):
      e2e_check.check_router_agreement(all_ties, all_ties, tol)


class CkptRoundTripCheckTest(unittest.TestCase):
  """Verifies exact/lossy checkpoint round-trips, dtype maps, unexported keys, and negative controls."""

  def setUp(self):
    super().setUp()
    self.source = {
        "w": layer_parity.hashed_normal("rt/w", (4, 8), mean=0.0, std=1.0),
        "bias": layer_parity.hashed_normal("rt/bias", (8,), mean=0.0, std=1.0),
        "ids": np.arange(12, dtype=np.int64).reshape(4, 3),
    }

  def test_exact_roundtrip_and_failures(self):
    tol = ckpt_roundtrip_check.RoundTripTol.exact(require_exact_dtypes=_EXACT_DTYPES)
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(self.source, dict(self.source), tol=tol)
    self.assertTrue(ok, failures)
    downcast = dict(self.source, bias=self.source["bias"].astype(np.float16))
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(self.source, downcast, tol=tol)
    self.assertFalse(ok)
    self.assertTrue(any("dtype mismatch" in f for f in failures))
    float_ids = dict(self.source, ids=self.source["ids"].astype(np.float32))
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(self.source, float_ids, tol=tol)
    self.assertFalse(ok)
    missing = {k: v for k, v in self.source.items() if k != "w"}
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(self.source, missing, tol=tol)
    self.assertFalse(ok)
    self.assertTrue(any("missing exported keys" in f for f in failures))
    with self.assertRaisesRegex(ValueError, "zero elements"):
      empty = {"e": np.zeros((0,), dtype=np.float32)}
      ckpt_roundtrip_check.check_hf_roundtrip(empty, empty, tol=tol)

  def test_dtype_maps_unexported_keys_and_lossy_tolerance(self):
    src_with_scale = dict(self.source, w_scale=np.ones((4,), dtype=np.float32))
    src_dtypes = {"w": "bfloat16", "bias": "float32", "ids": "int64", "w_scale": "float32"}
    exp_dtypes = {"w": "bfloat16", "bias": "float32", "ids": "int64"}
    lossy_tol = ckpt_roundtrip_check.RoundTripTol(
        atol=1e-2, rtol=1e-2, frob_rel=1e-2, require_exact_dtypes=("float32", "int64")
    )
    noisy_export = dict(self.source, w=self.source["w"] + 1e-3)
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(
        src_with_scale,
        noisy_export,
        tol=lossy_tol,
        unexported_source_keys=("w_scale",),
        source_dtypes=src_dtypes,
        exported_dtypes=exp_dtypes,
    )
    self.assertTrue(ok, failures)

    bad_exp_dtypes = dict(exp_dtypes, bias="bfloat16")
    ok, _, failures = ckpt_roundtrip_check.check_hf_roundtrip(
        src_with_scale,
        noisy_export,
        tol=lossy_tol,
        unexported_source_keys=("w_scale",),
        source_dtypes=src_dtypes,
        exported_dtypes=bad_exp_dtypes,
    )
    self.assertFalse(ok)
    self.assertTrue(any("dtype mismatch" in f for f in failures))

    with self.assertRaisesRegex(ValueError, "Pass both"):
      ckpt_roundtrip_check.check_hf_roundtrip(self.source, dict(self.source), tol=lossy_tol, source_dtypes=exp_dtypes)
    with self.assertRaises(KeyError):
      ckpt_roundtrip_check.check_hf_roundtrip(
          self.source, dict(self.source), tol=lossy_tol, source_dtypes={"w": "f32"}, exported_dtypes=exp_dtypes
      )
    with self.assertRaises(KeyError):
      ckpt_roundtrip_check.check_hf_roundtrip(
          self.source, dict(self.source), tol=lossy_tol, unexported_source_keys=("absent",)
      )

  def test_negative_roundtrip_status(self):
    tol = ckpt_roundtrip_check.RoundTripTol.exact(require_exact_dtypes=_EXACT_DTYPES)
    perturbed = ckpt_roundtrip_check.perturb_one_tensor(self.source, target_key="w", rel_scale=1e-2, seed=0)
    status, reasons = ckpt_roundtrip_check.negative_roundtrip_status(self.source, dict(self.source), perturbed, tol=tol)
    self.assertEqual(status, "FIRED", reasons)
    loose = ckpt_roundtrip_check.RoundTripTol(atol=1.0, rtol=1.0, frob_rel=1.0, require_exact_dtypes=_EXACT_DTYPES)
    status, _ = ckpt_roundtrip_check.negative_roundtrip_status(self.source, dict(self.source), perturbed, tol=loose)
    self.assertEqual(status, "MISSED")
    status, _ = ckpt_roundtrip_check.negative_roundtrip_status(self.source, dict(self.source), dict(self.source), tol=tol)
    self.assertEqual(status, "ERROR")
    bad_primary = dict(self.source, w=self.source["w"] + 1.0)
    status, _ = ckpt_roundtrip_check.negative_roundtrip_status(self.source, bad_primary, perturbed, tol=tol)
    self.assertEqual(status, "ERROR")
    nan_export = dict(self.source, w=np.full_like(self.source["w"], np.nan))
    status, _ = ckpt_roundtrip_check.negative_roundtrip_status(self.source, dict(self.source), nan_export, tol=tol)
    self.assertEqual(status, "ERROR")
    keyset_changed = {k: v for k, v in perturbed.items() if k != "bias"}
    status, _ = ckpt_roundtrip_check.negative_roundtrip_status(self.source, dict(self.source), keyset_changed, tol=tol)
    self.assertEqual(status, "ERROR")
    with self.assertRaisesRegex(ValueError, "non-float"):
      ckpt_roundtrip_check.perturb_one_tensor(self.source, target_key="ids", rel_scale=1e-2, seed=0)


if __name__ == "__main__":
  absltest.main()
