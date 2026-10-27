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

"""Unit tests for the standalone parity, negative-control, and e2e check helpers."""

from __future__ import annotations

import dataclasses
from typing import Any
import unittest

import jax
import jax.numpy as jnp
import numpy as np

torch: Any
F: Any
try:
  import torch  # pylint: disable=g-import-not-at-top
  import torch.nn.functional as F  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  F = None
  HAS_TORCH = False

import ml_dtypes
from tests.utils import parity
from tests.utils.parity import ckpt_roundtrip_check
from tests.utils.parity import e2e_check
from tests.utils.parity import layer_parity
from tests.utils.parity import negctl


@dataclasses.dataclass
class _NamedContainer:
  weight: jax.Array


jax.tree_util.register_dataclass(_NamedContainer, data_fields=["weight"], meta_fields=[])


def _toy_five_class_ref(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
  """Toy reference module exercising all five gradient classes."""
  scores = x @ params["w_disc"].T + params["b_state"]
  top_idx = torch.argmax(scores, dim=-1)
  embed_rows = params["w_pad"][top_idx % 3]
  frozen_proj = x @ params["w_froz"].T
  return torch.tanh(x @ params["w_comp"].T) + embed_rows + frozen_proj


def _toy_five_class_cand(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
  """Toy JAX candidate matching `_toy_five_class_ref` with padded rows and stop_gradient."""
  b_state = jnp.array([0.1, -0.2, 0.05, 0.3], dtype=x.dtype)
  scores = x @ params["w_disc"] + b_state
  top_idx = jnp.argmax(scores, axis=-1)
  embed_rows = params["w_pad"][top_idx % 3]
  frozen_proj = x @ jax.lax.stop_gradient(params["w_froz"])
  return jnp.tanh(x @ params["w_comp"]) + embed_rows + frozen_proj


def _build_five_class_fixture() -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, jax.Array],
    jax.Array,
    dict[str, str],
    layer_parity.GradientClasses,
]:
  """Creates paired reference and candidate parameters for the five-class toy module."""
  rng = np.random.default_rng(7)
  w_comp_np = rng.standard_normal((4, 4)).astype(np.float32) * 0.5
  b_state_np = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float32)
  w_disc_np = rng.standard_normal((4, 4)).astype(np.float32) * 0.4
  w_froz_np = rng.standard_normal((4, 4)).astype(np.float32) * 0.3
  w_pad_active_np = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
  w_pad_full_np = np.zeros((6, 4), dtype=np.float32)
  w_pad_full_np[:3] = w_pad_active_np
  x_np = rng.standard_normal((5, 4)).astype(np.float32)

  ref_params = {
      "w_comp": torch.tensor(w_comp_np, dtype=torch.float32),
      "b_state": torch.tensor(b_state_np, dtype=torch.float32),
      "w_disc": torch.tensor(w_disc_np, dtype=torch.float32),
      "w_froz": torch.tensor(w_froz_np, dtype=torch.float32),
      "w_pad": torch.tensor(w_pad_active_np, dtype=torch.float32),
  }
  ref_input = torch.tensor(x_np, dtype=torch.float32)
  cand_params = {
      "w_comp": jnp.asarray(w_comp_np.T, dtype=jnp.float32),
      "w_disc": jnp.asarray(w_disc_np.T, dtype=jnp.float32),
      "w_froz": jnp.asarray(w_froz_np.T, dtype=jnp.float32),
      "w_pad": jnp.asarray(w_pad_full_np, dtype=jnp.float32),
  }
  cand_input = jnp.asarray(x_np, dtype=jnp.float32)
  weight_map = {
      "w_comp": "w_comp:T",
      "w_disc": "w_disc:T",
      "w_froz": "w_froz:T",
      "w_pad": "w_pad",
  }
  grad_classes = layer_parity.GradientClasses(
      compared={"w_comp"},
      non_gradient={"b_state"},
      discrete_only={"w_disc"},
      frozen={"w_froz"},
      padded={"w_pad"},
  )
  return ref_params, ref_input, cand_params, cand_input, weight_map, grad_classes


class ParityUtilsTest(unittest.TestCase):

  def test_package_reexports_modules_and_constants(self):
    self.assertIs(parity.ckpt_roundtrip_check, ckpt_roundtrip_check)
    self.assertIs(parity.layer_parity, layer_parity)
    self.assertIs(parity.negctl, negctl)
    self.assertIs(parity.e2e_check, e2e_check)
    self.assertEqual(layer_parity.FP32_ATOL_CEILING, 1e-5)
    self.assertEqual(layer_parity.FP32_RTOL_CEILING, 1e-5)
    self.assertEqual(layer_parity.FP32_FROB_REL_CEILING, 1e-5)
    self.assertEqual(layer_parity.FP64_FROB_REL_CEILING, 1e-10)
    self.assertEqual(negctl.REL_NORM_BAND, (5e-3, 2e-2))

  def test_compare_floors_protect_small_and_scale_large_references(self):
    tol = layer_parity.FP32_TOLERANCE
    # Small-norm reference (|ref|_inf = 1e-4 < 1, ||ref||_2 < 1):
    # diff of 8e-6 on entry 0 passes ONLY when max(1, |ref|_inf) and max(1, ||ref||_2)
    # floor the denominator/scale at 1.0 (without the floor, 8e-6 > 1e-5 * 1e-4).
    ref_small = np.array([1e-4, -1e-4], dtype=np.float64)
    cand_small = ref_small + np.array([8e-6, 0.0], dtype=np.float64)
    ok_small, metrics_small = layer_parity.compare(cand_small, ref_small, tol)
    self.assertTrue(ok_small)
    self.assertTrue(metrics_small.finite)
    self.assertAlmostEqual(metrics_small.max_abs, 8e-6, places=12)
    self.assertAlmostEqual(metrics_small.ref_inf, 1e-4, places=12)
    self.assertAlmostEqual(metrics_small.frob_rel, 8e-6, places=12)

    # Zero reference also passes with a perturbation below 1e-5.
    ref_zero = np.zeros((4,), dtype=np.float64)
    cand_zero = np.full((4,), 4e-6, dtype=np.float64)
    ok_zero, metrics_zero = layer_parity.compare(cand_zero, ref_zero, tol)
    self.assertTrue(ok_zero)
    self.assertAlmostEqual(metrics_zero.frob_rel, 8e-6, places=12)

    # Large-norm reference (|ref|_inf = 100 > 1, ||ref||_2 = 100 > 1) with a 0 entry:
    # diff of 5e-4 at entry 1 passes ONLY because max(1, 100) scales atol and frob_rel by 100.
    ref_large = np.array([100.0, 0.0], dtype=np.float64)
    cand_large = np.array([100.0, 5e-4], dtype=np.float64)
    ok_large, metrics_large = layer_parity.compare(cand_large, ref_large, tol)
    self.assertTrue(ok_large)
    self.assertAlmostEqual(metrics_large.frob_rel, 5e-6, places=12)

    # Perturbation exceeding the 1e-5 floor on small reference must fail.
    cand_fail = ref_small + np.array([2e-5, 0.0], dtype=np.float64)
    ok_fail, _ = layer_parity.compare(cand_fail, ref_small, tol)
    self.assertFalse(ok_fail)

    # Non-finite input returns finite=False and ok=False; shape mismatch raises.
    ok_nan, metrics_nan = layer_parity.compare(np.array([np.nan, 0.0]), ref_small, tol)
    self.assertFalse(ok_nan)
    self.assertFalse(metrics_nan.finite)
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      layer_parity.compare(np.zeros((3,)), ref_small, tol)

  def test_fp64_ceiling_tolerance(self):
    ref = np.array([2.0, -3.0, 1.5], dtype=np.float64)
    cand_ok = ref + np.array([1e-11, -1e-11, 0.0], dtype=np.float64)
    ok, metrics = layer_parity.compare(cand_ok, ref, layer_parity.FP64_TOLERANCE)
    self.assertTrue(ok)
    self.assertLessEqual(metrics.frob_rel, layer_parity.FP64_FROB_REL_CEILING)

    cand_bad = ref + np.array([1e-9, 0.0, 0.0], dtype=np.float64)
    ok_bad, _ = layer_parity.compare(cand_bad, ref, layer_parity.FP64_TOLERANCE)
    self.assertFalse(ok_bad)

  def test_perturb_one_leaf_enforces_rel_norm_band_and_paths(self):
    tree = {
        "layers": [_NamedContainer(weight=jnp.array([[3.0, 4.0], [0.0, 0.0]], dtype=jnp.float32))],
        "other": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "indices": jnp.array([1, 2], dtype=jnp.int32),
        "zeros": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    # Positive match on the same tree first: valid scale in [5e-3, 2e-2] perturbs
    # only the targeted leaf by rel_scale * ||leaf||_2.
    perturbed = negctl.perturb_one_leaf(tree, target_path="layers/0/weight", rel_scale=1e-2, seed=42)
    delta = np.asarray(
        perturbed["layers"][0].weight - tree["layers"][0].weight,
        dtype=np.float64,
    )
    orig_norm = float(np.linalg.norm(np.asarray(tree["layers"][0].weight, dtype=np.float64)))
    self.assertAlmostEqual(float(np.linalg.norm(delta)) / orig_norm, 1e-2, places=6)
    np.testing.assert_array_equal(perturbed["other"], tree["other"])

    # Scales outside REL_NORM_BAND = (5e-3, 2e-2) must raise ValueError.
    with self.assertRaisesRegex(ValueError, "outside"):
      negctl.perturb_one_leaf(tree, target_path="layers/0/weight", rel_scale=4.9e-3, seed=42)
    with self.assertRaisesRegex(ValueError, "outside"):
      negctl.perturb_one_leaf(tree, target_path="layers/0/weight", rel_scale=2.1e-2, seed=42)

    # Unknown path, non-float leaf, and all-zero leaf must raise ValueError.
    with self.assertRaisesRegex(ValueError, "matched 0 leaves"):
      negctl.perturb_one_leaf(tree, target_path="layers/0/missing", rel_scale=1e-2, seed=42)
    with self.assertRaisesRegex(ValueError, "non-float"):
      negctl.perturb_one_leaf(tree, target_path="indices", rel_scale=1e-2, seed=42)
    with self.assertRaisesRegex(ValueError, "all zeros"):
      negctl.perturb_one_leaf(tree, target_path="zeros", rel_scale=1e-2, seed=42)
    with self.assertRaisesRegex(TypeError, "Unsupported key type"):
      negctl.leaf_path(("not_a_jax_key",))

  def test_padded_row_slicing_helper(self):
    ref = np.arange(12, dtype=np.float64).reshape(3, 4) * 0.25
    cand = np.zeros((6, 4), dtype=np.float64)
    cand[:3] = ref + 1e-7
    ok, metrics = layer_parity.compare_padded_rows(cand, ref, num_rows=3, tol=layer_parity.FP32_TOLERANCE)
    self.assertTrue(ok)
    self.assertTrue(metrics.finite)

    # Any non-zero entry in padding rows [3:] must fail even when [:3] matches.
    cand_bad_pad = cand.copy()
    cand_bad_pad[4, 2] = 1e-8
    ok_pad, _ = layer_parity.compare_padded_rows(cand_bad_pad, ref, num_rows=3, tol=layer_parity.FP32_TOLERANCE)
    self.assertFalse(ok_pad)

    # Active rows outside tolerance must fail even when padding rows are zero.
    cand_bad_active = cand.copy()
    cand_bad_active[1, 0] += 1e-2
    ok_active, _ = layer_parity.compare_padded_rows(cand_bad_active, ref, num_rows=3, tol=layer_parity.FP32_TOLERANCE)
    self.assertFalse(ok_active)

    with self.assertRaisesRegex(ValueError, "Invalid num_rows"):
      layer_parity.compare_padded_rows(cand, ref, num_rows=7, tol=layer_parity.FP32_TOLERANCE)
    with self.assertRaisesRegex(ValueError, "Padded shape mismatch"):
      layer_parity.compare_padded_rows(cand, np.zeros((2, 4)), num_rows=3, tol=layer_parity.FP32_TOLERANCE)

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_gradient_classes_must_partition_parameters(self):
    (
        ref_params,
        ref_input,
        cand_params,
        cand_input,
        weight_map,
        grad_classes,
    ) = _build_five_class_fixture()

    # Positive check: valid partition across all 5 classes passes check_parity.
    report = layer_parity.check_parity(
        ref_fn=_toy_five_class_ref,
        ref_params=ref_params,
        ref_input=ref_input,
        cand_fn=_toy_five_class_cand,
        cand_params=cand_params,
        cand_input=cand_input,
        weight_map=weight_map,
        grad_classes=grad_classes,
        fwd_tol=layer_parity.FP32_TOLERANCE,
        grad_tol=layer_parity.FP32_TOLERANCE,
        cotangent_seed=0,
    )
    self.assertTrue(report.passed, report.failures)
    self.assertEqual(set(report.grads.keys()), {"w_comp", "w_disc", "w_froz", "w_pad"})
    self.assertEqual(grad_classes.expected_grad_none, {"b_state", "w_disc", "w_froz"})

    # Missing parameter from GradientClasses (while weight_map still maps w_comp)
    # must fail the partition check.
    missing_classes = layer_parity.GradientClasses(
        compared=set(),
        non_gradient={"b_state"},
        discrete_only={"w_disc"},
        frozen={"w_froz"},
        padded={"w_pad"},
    )
    with self.assertRaisesRegex(ValueError, "missing=\\['w_comp'\\]"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params=cand_params,
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=missing_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

    # Overlapping classes (b_state in both non_gradient and discrete_only, where
    # expected_grad_none is unchanged) must fail the partition check.
    overlap_classes = layer_parity.GradientClasses(
        compared={"w_comp"},
        non_gradient={"b_state"},
        discrete_only={"w_disc", "b_state"},
        frozen={"w_froz"},
        padded={"w_pad"},
    )
    with self.assertRaisesRegex(ValueError, "overlaps="):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params=cand_params,
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=overlap_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

    # Unknown parameter in GradientClasses must fail the partition check.
    unknown_classes = layer_parity.GradientClasses(
        compared={"w_comp", "extra_param"},
        non_gradient={"b_state"},
        discrete_only={"w_disc"},
        frozen={"w_froz"},
        padded={"w_pad"},
    )
    with self.assertRaisesRegex(ValueError, "unknown=\\['extra_param'\\]"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params=cand_params,
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=unknown_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_gradient_classes_grad_none_mismatch_and_candidate_zero_checks(self):
    (
        ref_params,
        ref_input,
        cand_params,
        cand_input,
        weight_map,
        grad_classes,
    ) = _build_five_class_fixture()

    # Declaring a grad=None parameter (w_disc) in compared must raise ValueError.
    wrong_compared = layer_parity.GradientClasses(
        compared={"w_comp", "w_disc"},
        non_gradient={"b_state"},
        discrete_only=set(),
        frozen={"w_froz"},
        padded={"w_pad"},
    )
    with self.assertRaisesRegex(ValueError, "unexpected_grad_none=\\['w_disc'\\]"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params=cand_params,
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=wrong_compared,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

    # Declaring a differentiable parameter (w_comp) in discrete_only must raise ValueError.
    wrong_discrete = layer_parity.GradientClasses(
        compared=set(),
        non_gradient={"b_state"},
        discrete_only={"w_disc", "w_comp"},
        frozen={"w_froz"},
        padded={"w_pad"},
    )
    with self.assertRaisesRegex(ValueError, "expected_grad_none_but_present=\\['w_comp'\\]"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params=cand_params,
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=wrong_discrete,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

    # If the candidate fails to stop_gradient on a frozen parameter, check_parity fails.
    def _unfrozen_cand(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
      b_state = jnp.array([0.1, -0.2, 0.05, 0.3], dtype=x.dtype)
      scores = x @ params["w_disc"] + b_state
      top_idx = jnp.argmax(scores, axis=-1)
      return jnp.tanh(x @ params["w_comp"]) + params["w_pad"][top_idx % 3] + x @ params["w_froz"]

    report_unfrozen = layer_parity.check_parity(
        ref_fn=_toy_five_class_ref,
        ref_params=ref_params,
        ref_input=ref_input,
        cand_fn=_unfrozen_cand,
        cand_params=cand_params,
        cand_input=cand_input,
        weight_map=weight_map,
        grad_classes=grad_classes,
        fwd_tol=layer_parity.FP32_TOLERANCE,
        grad_tol=layer_parity.FP32_TOLERANCE,
        cotangent_seed=0,
    )
    self.assertFalse(report_unfrozen.passed)
    self.assertTrue(
        any("frozen grad w_froz" in f for f in report_unfrozen.failures),
        report_unfrozen.failures,
    )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_toy_mlp_fp32_fp64_noise_floor_and_negative_control(self):
    rng = np.random.default_rng(11)
    w1_np = rng.standard_normal((6, 4)).astype(np.float64) * 0.3
    b1_np = rng.standard_normal((6,)).astype(np.float64) * 0.1
    w2_np = rng.standard_normal((3, 6)).astype(np.float64) * 0.3
    x_np = rng.standard_normal((4, 4)).astype(np.float64)

    def ref_mlp(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
      h = F.silu(x @ params["w1"].T + params["b1"])
      return h @ params["w2"].T

    def cand_mlp(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
      h = jax.nn.silu(x @ params["w1"] + params["b1"])
      return h @ params["w2"]

    weight_map = {"w1": "w1:T", "b1": "b1", "w2": "w2:T"}
    grad_classes = layer_parity.GradientClasses(compared={"w1", "b1", "w2"})

    ref_params_32 = {
        "w1": torch.tensor(w1_np, dtype=torch.float32),
        "b1": torch.tensor(b1_np, dtype=torch.float32),
        "w2": torch.tensor(w2_np, dtype=torch.float32),
    }
    ref_input_32 = torch.tensor(x_np, dtype=torch.float32)
    cand_params_32 = {
        "w1": jnp.asarray(w1_np.T, dtype=jnp.float32),
        "b1": jnp.asarray(b1_np, dtype=jnp.float32),
        "w2": jnp.asarray(w2_np.T, dtype=jnp.float32),
    }
    cand_input_32 = jnp.asarray(x_np, dtype=jnp.float32)

    noise = layer_parity.reference_noise_floor(ref_mlp, ref_params_32, ref_input_32)
    self.assertTrue(noise.finite)
    self.assertLess(noise.frob_rel, 1e-5)

    report_32 = layer_parity.check_parity(
        ref_fn=ref_mlp,
        ref_params=ref_params_32,
        ref_input=ref_input_32,
        cand_fn=cand_mlp,
        cand_params=cand_params_32,
        cand_input=cand_input_32,
        weight_map=weight_map,
        grad_classes=grad_classes,
        fwd_tol=layer_parity.FP32_TOLERANCE,
        grad_tol=layer_parity.FP32_TOLERANCE,
        cotangent_seed=3,
    )
    self.assertTrue(report_32.passed, report_32.failures)

    fired, neg_metrics = layer_parity.negative_control_fires(
        ref_out=report_32.ref_out,
        cand_fn=cand_mlp,
        cand_params=cand_params_32,
        cand_input=cand_input_32,
        fwd_tol=layer_parity.FP32_TOLERANCE,
        target_path="w1",
        rel_scale=1e-2,
        seed=0,
    )
    self.assertTrue(fired, f"negative control failed to fire: {neg_metrics}")

    with jax.enable_x64(True):
      ref_params_64 = {
          "w1": torch.tensor(w1_np, dtype=torch.float64),
          "b1": torch.tensor(b1_np, dtype=torch.float64),
          "w2": torch.tensor(w2_np, dtype=torch.float64),
      }
      ref_input_64 = torch.tensor(x_np, dtype=torch.float64)
      cand_params_64 = {
          "w1": jnp.asarray(w1_np.T, dtype=jnp.float64),
          "b1": jnp.asarray(b1_np, dtype=jnp.float64),
          "w2": jnp.asarray(w2_np.T, dtype=jnp.float64),
      }
      cand_input_64 = jnp.asarray(x_np, dtype=jnp.float64)
      report_64 = layer_parity.check_parity(
          ref_fn=ref_mlp,
          ref_params=ref_params_64,
          ref_input=ref_input_64,
          cand_fn=cand_mlp,
          cand_params=cand_params_64,
          cand_input=cand_input_64,
          weight_map=weight_map,
          grad_classes=grad_classes,
          fwd_tol=layer_parity.FP64_TOLERANCE,
          grad_tol=layer_parity.FP64_TOLERANCE,
          cotangent_seed=3,
      )
      self.assertTrue(report_64.passed, report_64.failures)

  def test_e2e_check_logits_tokens_allclose_and_negative_run_status(self):
    golden_logits = {"out": np.array([[4.0, 1.0, -1.0], [0.5, 3.5, -0.5]], dtype=np.float64)}
    primary_logits = {"out": golden_logits["out"] + np.array([[1e-4, -1e-4, 0.0], [0.0, 1e-4, -1e-4]])}
    perturbed_logits = {"out": np.array([[1.0, 4.0, -1.0], [3.5, 0.5, -0.5]], dtype=np.float64)}
    logits_tol = e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=1e-6, top1_min=1.0)

    ok_logits, metrics_logits, failures_logits = e2e_check.check_outputs(
        golden_logits, primary_logits, ["out"], logits_tol
    )
    self.assertTrue(ok_logits, failures_logits)
    self.assertLessEqual(metrics_logits["out"]["kl_max"], 1e-6)
    self.assertEqual(metrics_logits["out"]["top1"], 1.0)

    status_fired, reasons_fired = e2e_check.negative_run_status(
        golden_logits, primary_logits, perturbed_logits, ["out"], logits_tol
    )
    self.assertEqual(status_fired, "FIRED", reasons_fired)

    # Perturbed output still passing tolerance -> MISSED (which fails a parity test).
    tiny_perturbed = {"out": golden_logits["out"] + np.array([[2e-4, -1e-4, 0.0], [0.0, 2e-4, -1e-4]])}
    status_missed, _ = e2e_check.negative_run_status(golden_logits, primary_logits, tiny_perturbed, ["out"], logits_tol)
    self.assertEqual(status_missed, "MISSED")

    # Unperturbed run failing, non-finite perturbed output, and identical perturbed
    # output must all return ERROR.
    status_err_primary, _ = e2e_check.negative_run_status(
        golden_logits, perturbed_logits, primary_logits, ["out"], logits_tol
    )
    self.assertEqual(status_err_primary, "ERROR")
    status_err_nan, _ = e2e_check.negative_run_status(
        golden_logits,
        primary_logits,
        {"out": np.full((2, 3), np.nan)},
        ["out"],
        logits_tol,
    )
    self.assertEqual(status_err_nan, "ERROR")
    status_err_same, _ = e2e_check.negative_run_status(golden_logits, primary_logits, primary_logits, ["out"], logits_tol)
    self.assertEqual(status_err_same, "ERROR")

    # TokensTol: exact integer match, flipped token FIRED, and fallback to underlying
    # logits when a small logit perturbation does not flip argmax tokens.
    tokens_tol = e2e_check.TokensTol(match_min=1.0)
    golden_tokens = {"tok": np.argmax(golden_logits["out"], axis=-1).astype(np.int32)}
    primary_tokens = {"tok": np.argmax(primary_logits["out"], axis=-1).astype(np.int32)}
    flipped_tokens = {"tok": np.argmax(perturbed_logits["out"], axis=-1).astype(np.int32)}
    ok_tok, _, _ = e2e_check.check_outputs(golden_tokens, primary_tokens, ["tok"], tokens_tol)
    self.assertTrue(ok_tok)
    status_tok_fired, _ = e2e_check.negative_run_status(
        golden_tokens, primary_tokens, flipped_tokens, ["tok"], tokens_tol
    )
    self.assertEqual(status_tok_fired, "FIRED")
    with self.assertRaisesRegex(TypeError, "integer token ids"):
      e2e_check.compare_one(golden_logits["out"], primary_logits["out"], tokens_tol)

    # Allclose mode (layer_parity.Tolerance):
    ok_allclose, _, _ = e2e_check.check_outputs(
        golden_logits,
        {"out": golden_logits["out"] + 1e-6},
        ["out"],
        layer_parity.FP32_TOLERANCE,
    )
    self.assertTrue(ok_allclose)
    status_allclose_fired, _ = e2e_check.negative_run_status(
        golden_logits,
        {"out": golden_logits["out"] + 1e-6},
        perturbed_logits,
        ["out"],
        layer_parity.FP32_TOLERANCE,
    )
    self.assertEqual(status_allclose_fired, "FIRED")

  def test_e2e_logits_tol_enforces_kl_max_ceiling(self):
    # 80 positions where 79 match identically (KL = 0) and 1 delimiter position
    # spikes kl_max > 1e-6 while kl_mean <= 1e-6 and top1 == 1.0.
    base_row = np.array([3.0, 1.0, 0.0, -1.0], dtype=np.float64)
    golden = {"logits": np.tile(base_row, (80, 1))}
    primary = {"logits": golden["logits"].copy()}
    spiked = golden["logits"].copy()
    spiked[0] = np.array([3.0, 1.02, 0.0, -1.0], dtype=np.float64)
    cand_spiked = {"logits": spiked}
    tol = e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=1e-6, top1_min=1.0)

    ok, metrics, failures = e2e_check.check_outputs(golden, cand_spiked, ["logits"], tol)
    self.assertLessEqual(metrics["logits"]["kl_mean"], 1e-6)
    self.assertGreater(metrics["logits"]["kl_max"], 1e-6)
    self.assertEqual(metrics["logits"]["top1"], 1.0)
    self.assertFalse(ok, f"Expected kl_max spike to fail check_outputs: {metrics}")
    self.assertEqual(len(failures), 1)

    status, reasons = e2e_check.negative_run_status(golden, primary, cand_spiked, ["logits"], tol)
    self.assertEqual(status, "FIRED", reasons)

    with self.assertRaisesRegex(ValueError, "LogitsTol"):
      e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=-1e-6, top1_min=1.0)
    with self.assertRaisesRegex(ValueError, "LogitsTol"):
      e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=1e-6, top1_min=1.5)
    with self.assertRaisesRegex(ValueError, "TokensTol"):
      e2e_check.TokensTol(match_min=-0.1)

  def test_negative_run_status_rejects_bf16_nan_and_missing_perturbed_keys(self):
    golden = {"out": np.array([[4.0, 1.0, -1.0], [0.5, 3.5, -0.5]], dtype=np.float64)}
    primary = {"out": golden["out"] + np.array([[1e-6, -1e-6, 0.0], [0.0, 1e-6, -1e-6]])}

    # bfloat16 NaN and Inf perturbed outputs must return ERROR, never FIRED.
    bf16_nan = {"out": jnp.full((2, 3), jnp.nan, dtype=jnp.bfloat16)}
    status_nan_allclose, reasons_nan_allclose = e2e_check.negative_run_status(
        golden, primary, bf16_nan, ["out"], layer_parity.FP32_TOLERANCE
    )
    self.assertEqual(status_nan_allclose, "ERROR")
    self.assertTrue(any("not finite" in r for r in reasons_nan_allclose), reasons_nan_allclose)

    tol = e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=1e-6, top1_min=1.0)
    status_nan, reasons_nan = e2e_check.negative_run_status(golden, primary, bf16_nan, ["out"], tol)
    self.assertEqual(status_nan, "ERROR")
    self.assertTrue(any("not finite" in r for r in reasons_nan), reasons_nan)

    bf16_inf = {"out": jnp.full((2, 3), jnp.inf, dtype=jnp.bfloat16)}
    status_inf, reasons_inf = e2e_check.negative_run_status(golden, primary, bf16_inf, ["out"], tol)
    self.assertEqual(status_inf, "ERROR")
    self.assertTrue(any("not finite" in r for r in reasons_inf), reasons_inf)

    # Missing key in perturbed must raise KeyError upfront even when primary would fail.
    bad_primary = {"out": np.array([[1.0, 4.0, -1.0], [3.5, 0.5, -0.5]], dtype=np.float64)}
    with self.assertRaisesRegex(KeyError, "Keys missing"):
      e2e_check.negative_run_status(golden, bad_primary, {}, ["out"], layer_parity.FP32_TOLERANCE)

  def test_perturb_one_leaf_rejects_nonfinite_and_out_of_band_realized_norm(self):
    nan_tree = {"w": jnp.array([1.0, jnp.nan], dtype=jnp.float32)}
    with self.assertRaisesRegex(ValueError, "non-finite"):
      negctl.perturb_one_leaf(nan_tree, target_path="w", rel_scale=1e-2, seed=0)

    inf_tree = {"w": jnp.array([1.0, jnp.inf], dtype=jnp.float32)}
    with self.assertRaisesRegex(ValueError, "non-finite"):
      negctl.perturb_one_leaf(inf_tree, target_path="w", rel_scale=1e-2, seed=0)

    # At 1.875 in bfloat16, 1 ULP = 2^-7 = 0.0078125; a 5e-3 perturbation
    # (0.009375 = 1.2 ULP) rounds to 1 ULP, giving realized_rel = 1/240 = 4.167e-3 < 5e-3.
    bf16_under_tree = {"w": jnp.array([1.875], dtype=jnp.bfloat16)}
    with self.assertRaisesRegex(ValueError, "outside"):
      negctl.perturb_one_leaf(bf16_under_tree, target_path="w", rel_scale=5e-3, seed=0)

  def test_negative_control_fires_returns_false_on_nonfinite_output(self):
    cand_params = {"w": jnp.array([[1.0, 0.5], [-0.5, 1.0]], dtype=jnp.float32)}
    cand_input = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    ref_out = np.asarray(cand_input @ cand_params["w"], dtype=np.float64)

    def nan_cand_fn(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
      return jnp.full_like(x @ params["w"], jnp.nan)

    fired_nan, metrics_nan = layer_parity.negative_control_fires(
        ref_out=ref_out,
        cand_fn=nan_cand_fn,
        cand_params=cand_params,
        cand_input=cand_input,
        fwd_tol=layer_parity.FP32_TOLERANCE,
        target_path="w",
        rel_scale=1e-2,
        seed=0,
    )
    self.assertFalse(fired_nan)
    self.assertFalse(metrics_nan.finite)

    def numpy_cand_fn(params: dict[str, jax.Array], x: jax.Array) -> np.ndarray:
      return np.asarray(x @ params["w"])

    with self.assertRaisesRegex(TypeError, "not jax.Array"):
      layer_parity.negative_control_fires(
          ref_out=ref_out,
          cand_fn=numpy_cand_fn,  # pytype: disable=wrong-arg-types
          cand_params=cand_params,
          cand_input=cand_input,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          target_path="w",
          rel_scale=1e-2,
          seed=0,
      )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_layer_parity_rejects_bypass_vectors(self):
    (
        ref_params,
        ref_input,
        cand_params,
        cand_input,
        weight_map,
        grad_classes,
    ) = _build_five_class_fixture()

    # 1. Mapping a non_gradient reference parameter in weight_map to a candidate
    # leaf must raise ValueError instead of swallowing that candidate leaf.
    swallowed_cand = dict(cand_params)
    swallowed_cand["b_state"] = jnp.ones((4,), dtype=jnp.float32)
    swallowed_map = dict(weight_map)
    swallowed_map["b_state"] = "b_state"
    with self.assertRaisesRegex(ValueError, "non_grad_mapped=\\['b_state'\\]"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params=swallowed_cand,
          cand_input=cand_input,
          weight_map=swallowed_map,
          grad_classes=grad_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

    # 2. Direct check_gradients_by_class must reject None, torch.Tensor, and
    # non-floating arrays in cand_grads.
    ref_grads: dict[str, torch.Tensor | None] = {
        "w_comp": torch.zeros_like(ref_params["w_comp"]),
        "b_state": None,
        "w_disc": None,
        "w_froz": None,
        "w_pad": torch.zeros_like(ref_params["w_pad"]),
    }
    valid_cand_grads = {k: np.zeros(v.shape, dtype=np.float32) for k, v in cand_params.items()}
    for bad_grad in (None, torch.zeros((4, 4)), np.zeros((4, 4), dtype=np.int32)):
      bad_cand_grads = dict(valid_cand_grads)
      bad_cand_grads["w_comp"] = bad_grad  # type: ignore[assignment]
      with self.assertRaises(TypeError):
        layer_parity.check_gradients_by_class(
            ref_params=ref_params,
            ref_grads=ref_grads,
            cand_grads=bad_cand_grads,
            weight_map=weight_map,
            grad_classes=grad_classes,
            grad_tol=layer_parity.FP32_TOLERANCE,
        )

    # 3. Non-zero candidate gradient on a discrete_only parameter must fail.
    nonzero_disc_grads = dict(valid_cand_grads)
    nonzero_disc_grads["w_disc"] = np.full((4, 4), 1e-4, dtype=np.float32)
    _, disc_failures = layer_parity.check_gradients_by_class(
        ref_params=ref_params,
        ref_grads=ref_grads,
        cand_grads=nonzero_disc_grads,
        weight_map=weight_map,
        grad_classes=grad_classes,
        grad_tol=layer_parity.FP32_TOLERANCE,
    )
    self.assertTrue(any("discrete_only grad w_disc" in f for f in disc_failures), disc_failures)

    # 4. validate_gradient_classes must reject empty param_names and partitions
    # with empty compared | padded.
    with self.assertRaisesRegex(ValueError, "non-empty"):
      layer_parity.validate_gradient_classes(layer_parity.GradientClasses(), set(), {})
    with self.assertRaisesRegex(ValueError, "at least one compared or padded"):
      layer_parity.validate_gradient_classes(
          layer_parity.GradientClasses(non_gradient={"b_state"}),
          {"b_state"},
          {"b_state": None},
      )

    # 5. compare_padded_rows must reject cand.shape[0] == num_rows (zero padding rows).
    ref_rows = np.ones((3, 4), dtype=np.float64)
    with self.assertRaisesRegex(ValueError, "Invalid num_rows"):
      layer_parity.compare_padded_rows(ref_rows.copy(), ref_rows, num_rows=3, tol=layer_parity.FP32_TOLERANCE)

    # 6. _assert_jax_tree / check_parity must reject empty pytrees and non-jax.Array leaves.
    with self.assertRaisesRegex(ValueError, "no leaves"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params={},
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=grad_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )
    with self.assertRaisesRegex(TypeError, "not jax.Array"):
      layer_parity.check_parity(
          ref_fn=_toy_five_class_ref,
          ref_params=ref_params,
          ref_input=ref_input,
          cand_fn=_toy_five_class_cand,
          cand_params={"w_comp": np.zeros((4, 4), dtype=np.float32)},
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=grad_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=0,
      )

    # 7. reference_noise_floor handles requires_grad=True / bfloat16 and raises on NaN.
    bf16_params = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16, requires_grad=True)}
    int_input = torch.tensor([0, 1], dtype=torch.int64)
    noise_bf16 = layer_parity.reference_noise_floor(
        lambda p, idx: (p["w"][idx] * 0.5).to(torch.bfloat16),
        bf16_params,
        int_input,
    )
    self.assertTrue(noise_bf16.finite)
    with self.assertRaisesRegex(ValueError, "non-finite"):
      layer_parity.reference_noise_floor(
          lambda p, idx: torch.full((2, 2), float("nan"), dtype=p["w"].dtype),
          bf16_params,
          int_input,
      )

  def test_tolerance_and_logits_tol_enforce_approved_ceilings(self):
    # Bypass (a): looser-than-ceiling Tolerance (> 1e-5) or LogitsTol (> 1e-6 or top1 < 1.0)
    # must raise ValueError at construction; tighter-than-ceiling (e.g. 1e-12 for R8) works.
    tight_tol = layer_parity.Tolerance(atol=1e-12, rtol=1e-12, frob_rel=1e-12)
    self.assertEqual(tight_tol.frob_rel, 1e-12)
    with self.assertRaisesRegex(ValueError, "exceeds approved fp32 ceilings"):
      layer_parity.Tolerance(atol=1e-2, rtol=1e-5, frob_rel=1e-5)
    with self.assertRaisesRegex(ValueError, "exceeds approved fp32 ceilings"):
      layer_parity.Tolerance(atol=1e-5, rtol=1e-2, frob_rel=1e-5)
    with self.assertRaisesRegex(ValueError, "exceeds approved fp32 ceilings"):
      layer_parity.Tolerance(atol=1e-5, rtol=1e-5, frob_rel=1e-2)

    self.assertEqual(e2e_check.LOGITS_KL_CEILING, 1e-6)
    self.assertEqual(e2e_check.LOGITS_TOP1_FLOOR, 1.0)
    self.assertEqual(e2e_check.logits_tol_for_row("T13"), e2e_check.CPU_LOGITS_TOL)
    with self.assertRaisesRegex(ValueError, "exceeds approved CPU logits ceilings"):
      e2e_check.LogitsTol(kl_mean_max=0.1, kl_max=0.5, top1_min=0.5)
    with self.assertRaisesRegex(ValueError, "exceeds approved CPU logits ceilings"):
      e2e_check.LogitsTol(kl_mean_max=2e-6, kl_max=1e-6, top1_min=1.0)
    with self.assertRaisesRegex(ValueError, "exceeds approved CPU logits ceilings"):
      e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=2e-6, top1_min=1.0)
    with self.assertRaisesRegex(ValueError, "exceeds approved CPU logits ceilings"):
      e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=1e-6, top1_min=0.99)
    with self.assertRaisesRegex(ValueError, r"Unregistered logits tolerance row.*registered rows: \[T13\]"):
      e2e_check.logits_tol_for_row("TT3a")
    with self.assertRaisesRegex(ValueError, r"Unregistered logits tolerance row.*registered rows: \[T13\]"):
      e2e_check.logits_tol_for_row("UNKNOWN_ROW")

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_fp64_parity_and_gradients_reject_fp32_tolerance(self):
    # Bypass (b): check_parity, check_gradients_by_class, and negative_control_fires
    # on float64 tensors must reject FP32_TOLERANCE (1e-5 > 1e-10).
    with jax.enable_x64(True):
      ref_params_64 = {"w": torch.tensor([[1.0, 0.5], [-0.5, 1.0]], dtype=torch.float64)}
      ref_input_64 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
      cand_params_64 = {"w": jnp.array([[1.0, -0.5], [0.5, 1.0]], dtype=jnp.float64)}
      cand_input_64 = jnp.array([[1.0, 2.0]], dtype=jnp.float64)
      weight_map = {"w": "w:T"}
      grad_classes = layer_parity.GradientClasses(compared={"w"})

      with self.assertRaisesRegex(ValueError, "FP64_FROB_REL_CEILING"):
        layer_parity.check_parity(
            ref_fn=lambda p, x: x @ p["w"].T,
            ref_params=ref_params_64,
            ref_input=ref_input_64,
            cand_fn=lambda p, x: x @ p["w"],
            cand_params=cand_params_64,
            cand_input=cand_input_64,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP64_TOLERANCE,
            cotangent_seed=0,
        )
      with self.assertRaisesRegex(ValueError, "FP64_FROB_REL_CEILING"):
        layer_parity.check_parity(
            ref_fn=lambda p, x: x @ p["w"].T,
            ref_params=ref_params_64,
            ref_input=ref_input_64,
            cand_fn=lambda p, x: x @ p["w"],
            cand_params=cand_params_64,
            cand_input=cand_input_64,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP64_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=0,
        )
      with self.assertRaisesRegex(ValueError, "FP64_FROB_REL_CEILING"):
        layer_parity.check_gradients_by_class(
            ref_params=ref_params_64,
            ref_grads={"w": torch.zeros((2, 2), dtype=torch.float64)},
            cand_grads={"w": np.zeros((2, 2), dtype=np.float64)},
            weight_map=weight_map,
            grad_classes=grad_classes,
            grad_tol=layer_parity.FP32_TOLERANCE,
        )
      with self.assertRaisesRegex(ValueError, "FP64_FROB_REL_CEILING"):
        layer_parity.negative_control_fires(
            ref_out=np.zeros((1, 2), dtype=np.float64),
            cand_fn=lambda p, x: x @ p["w"],
            cand_params=cand_params_64,
            cand_input=cand_input_64,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="w",
            rel_scale=1e-2,
            seed=0,
        )
      with self.assertRaisesRegex(ValueError, "Reference gradient shape mismatch"):
        layer_parity.check_gradients_by_class(
            ref_params=ref_params_64,
            ref_grads={"w": torch.zeros((2, 3), dtype=torch.float64)},
            cand_grads={"w": np.zeros((2, 2), dtype=np.float64)},
            weight_map=weight_map,
            grad_classes=grad_classes,
            grad_tol=layer_parity.FP64_TOLERANCE,
        )

  def test_logits_compare_one_rejects_degenerate_vocab_axis(self):
    # Bypass (c): shape[-1] < 2 (where _log_softmax == 0 and argmax == 0) must raise ValueError.
    tol = e2e_check.LogitsTol(kl_mean_max=1e-6, kl_max=1e-6, top1_min=1.0)
    with self.assertRaisesRegex(ValueError, "shape\\[-1\\] >= 2"):
      e2e_check.compare_one(
          np.zeros((4, 1), dtype=np.float64),
          np.ones((4, 1), dtype=np.float64),
          tol,
      )
    with self.assertRaisesRegex(ValueError, "shape\\[-1\\] >= 2"):
      e2e_check.compare_one(
          np.array(1.0, dtype=np.float64),
          np.array(2.0, dtype=np.float64),
          tol,
      )

  def test_ckpt_roundtrip_tol_validation(self):
    tol = ckpt_roundtrip_check.RoundTripTol.exact()
    self.assertTrue(tol.is_exact)
    self.assertEqual((tol.atol, tol.rtol, tol.frob_rel), (0.0, 0.0, 0.0))
    non_exact = ckpt_roundtrip_check.RoundTripTol(atol=1e-5, rtol=1e-5, frob_rel=1e-5)
    self.assertFalse(non_exact.is_exact)

  def test_ckpt_roundtrip_check_pass_and_failure_modes(self):
    tol = ckpt_roundtrip_check.RoundTripTol.exact()
    src = {
        "w_bf16": np.array([1.0, -2.5, 0.125], dtype=ml_dtypes.bfloat16),
        "w_f32": np.array([0.5, -1.25], dtype=np.float32),
        "w_i64": np.array([1, 2, 3], dtype=np.int64),
        "unexported": np.array([9.0], dtype=np.float32),
    }
    exp = {
        "w_bf16": src["w_bf16"].copy(),
        "w_f32": src["w_f32"].copy(),
        "w_i64": src["w_i64"].copy(),
    }
    src_dtypes = {k: str(v.dtype) for k, v in src.items()}
    exp_dtypes = {k: str(v.dtype) for k, v in exp.items()}
    ok, metrics, failures = ckpt_roundtrip_check.check_hf_roundtrip(
        src,
        exp,
        tol=tol,
        unexported_source_keys=frozenset({"unexported"}),
        source_dtypes=src_dtypes,
        exported_dtypes=exp_dtypes,
    )
    self.assertTrue(ok, msg=f"Unexpected failures: {failures}")
    self.assertEqual(failures, [])
    self.assertEqual(metrics["w_bf16"]["max_abs"], 0.0)

    # Missing and extra key failures.
    ok_miss, _, fail_miss = ckpt_roundtrip_check.check_hf_roundtrip(src, exp, tol=tol)
    self.assertFalse(ok_miss)
    self.assertTrue(any("missing exported keys" in f and "unexported" in f for f in fail_miss))

    exp_extra = dict(exp, extra_key=np.array([1.0], dtype=np.float32))
    ok_extra, _, fail_extra = ckpt_roundtrip_check.check_hf_roundtrip(
        src, exp_extra, tol=tol, unexported_source_keys=frozenset({"unexported"})
    )
    self.assertFalse(ok_extra)
    self.assertTrue(any("unexpected exported keys" in f and "extra_key" in f for f in fail_extra))

    # Shape and dtype mismatch failures.
    exp_shape = dict(exp, w_f32=np.zeros((3,), dtype=np.float32))
    ok_shape, _, fail_shape = ckpt_roundtrip_check.check_hf_roundtrip(
        src, exp_shape, tol=tol, unexported_source_keys=frozenset({"unexported"})
    )
    self.assertFalse(ok_shape)
    self.assertTrue(any("w_f32: shape mismatch" in f for f in fail_shape))

    exp_bad_dtypes = dict(exp_dtypes, w_f32="bfloat16")
    ok_dt, _, fail_dt = ckpt_roundtrip_check.check_hf_roundtrip(
        src,
        exp,
        tol=tol,
        unexported_source_keys=frozenset({"unexported"}),
        source_dtypes=src_dtypes,
        exported_dtypes=exp_bad_dtypes,
    )
    self.assertFalse(ok_dt)
    self.assertTrue(any("w_f32: dtype mismatch" in f for f in fail_dt))

    # 1-ULP bfloat16 perturbation failure.
    w_bf16_u16 = src["w_bf16"].view(np.uint16).copy()
    w_bf16_u16[0] = np.uint16(w_bf16_u16[0] + 1)
    exp_ulp = dict(exp, w_bf16=w_bf16_u16.view(ml_dtypes.bfloat16))
    ok_ulp, _, fail_ulp = ckpt_roundtrip_check.check_hf_roundtrip(
        src, exp_ulp, tol=tol, unexported_source_keys=frozenset({"unexported"})
    )
    self.assertFalse(ok_ulp)
    self.assertTrue(any("w_bf16: max_abs=" in f for f in fail_ulp))

  def test_ckpt_roundtrip_perturb_and_negative_status(self):
    tol = ckpt_roundtrip_check.RoundTripTol.exact()
    src = {
        "w_bf16": np.linspace(-1.0, 1.0, 16, dtype=np.float32).astype(ml_dtypes.bfloat16),
        "w_f32": np.linspace(0.5, 2.5, 8, dtype=np.float32),
    }
    primary = {k: v.copy() for k, v in src.items()}
    perturbed = ckpt_roundtrip_check.perturb_one_tensor(primary, target_key="w_bf16", rel_scale=1e-2, seed=0)
    self.assertTrue(np.array_equal(perturbed["w_f32"], primary["w_f32"]))
    self.assertFalse(np.array_equal(perturbed["w_bf16"].view(np.uint16), primary["w_bf16"].view(np.uint16)))

    status, reasons = ckpt_roundtrip_check.negative_roundtrip_status(src, primary, perturbed, tol=tol)
    self.assertEqual((status, reasons), ("FIRED", []))

    status_ident, _ = ckpt_roundtrip_check.negative_roundtrip_status(src, primary, primary, tol=tol)
    self.assertEqual(status_ident, "ERROR")

    loose_tol = ckpt_roundtrip_check.RoundTripTol(atol=1.0, rtol=1.0, frob_rel=1.0)
    status_missed, _ = ckpt_roundtrip_check.negative_roundtrip_status(src, primary, perturbed, tol=loose_tol)
    self.assertEqual(status_missed, "MISSED")

    status_pfail, _ = ckpt_roundtrip_check.negative_roundtrip_status(src, perturbed, perturbed, tol=tol)
    self.assertEqual(status_pfail, "ERROR")

    with self.assertRaises(KeyError):
      ckpt_roundtrip_check.perturb_one_tensor(primary, target_key="missing", rel_scale=1e-2, seed=0)
    with self.assertRaises(ValueError):
      ckpt_roundtrip_check.perturb_one_tensor(primary, target_key="w_f32", rel_scale=0.0, seed=0)


if __name__ == "__main__":
  unittest.main()
