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

"""Parity test template: one m3 component vs. a pinned PyTorch reference.

Copy to `tests/unit/m3/<model>_<component>_parity_test.py` and replace every
`_UNFILLED` sentinel via `_require_filled(...)`. Uses standard `unittest`
(collected by `pytest`) with method-level `@unittest.skipUnless(HAS_TORCH, ...)`
so torch-free MaxText CPU test suites never fail at module collection time.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any
import unittest

import jax
import jax.numpy as jnp

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _load_layer_parity() -> Any:
  """Loads `.agents/skills/model-bringup/scripts/layer_parity.py` without torch."""
  qual = "_maxtext_bringup_layer_parity"
  if qual in sys.modules:
    return sys.modules[qual]
  here = pathlib.Path(__file__).resolve()
  for parent in here.parents:
    cand = parent / ".agents/skills/model-bringup/scripts/layer_parity.py"
    if cand.is_file():
      spec = importlib.util.spec_from_file_location(qual, cand)
      if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qual] = mod
        spec.loader.exec_module(mod)
        return mod
  raise ImportError("Could not locate .agents/skills/model-bringup/scripts/layer_parity.py")


layer_parity = _load_layer_parity()

_UNFILLED = object()

# Replace each `_UNFILLED` with explicit rows from the design tolerance table.
FWD_TOL: Any = _UNFILLED  # layer_parity.Tolerance(atol=..., rtol=..., frob_rel=...)
GRAD_TOL: Any = _UNFILLED  # layer_parity.Tolerance(atol=..., rtol=..., frob_rel=...)
FP64_TOL: Any = _UNFILLED  # layer_parity.Tolerance(atol=..., rtol=..., frob_rel=1e-10)
WEIGHT_MAP: Any = _UNFILLED  # dict[str, str] covering every ref & candidate leaf once
NEGCTL_LEAF: Any = _UNFILLED  # candidate leaf path string for perturbation
NEGCTL_REL_SCALE = 1e-2
NEGCTL_SEED = 0
COTANGENT_SEED = 0


def _require_filled(val: Any, what: str) -> Any:
  if val is _UNFILLED:
    raise NotImplementedError(f"Parity template not filled in: {what}")
  return val


def make_reference(dtype: Any) -> tuple[Any, dict[str, Any], Any]:
  """Returns `(ref_fn, ref_params, ref_input)` from the pinned PyTorch reference."""
  if dtype is not _UNFILLED:
    raise NotImplementedError("Parity template not filled in: reference factory (ref_fn, ref_params, ref_input)")
  return (lambda p, x: x), {}, None


def make_candidate(ref_params: dict[str, Any], ref_input: Any, dtype: jnp.dtype) -> tuple[Any, Any, Any]:
  """Returns `(cand_fn, cand_params, cand_input)` mapped via production weight hooks."""
  del ref_params, ref_input
  if dtype is not _UNFILLED:
    raise NotImplementedError("Parity template not filled in: candidate factory (cand_fn, cand_params, cand_input)")
  return (lambda p, x: x), {}, None


def _run_parity(
    torch_dtype: Any, jax_dtype: jnp.dtype, *, fwd_tol: Any, grad_tol: Any
) -> tuple[Any, tuple[Any, Any, Any], int]:
  """Runs forward and all-leaf gradient parity between reference and candidate."""
  ref_fn, ref_params, ref_input = make_reference(torch_dtype)
  cand_fn, cand_params, cand_input = make_candidate(ref_params, ref_input, jax_dtype)
  report = layer_parity.check_parity(
      ref_fn=ref_fn,
      ref_params=ref_params,
      ref_input=ref_input,
      cand_fn=cand_fn,
      cand_params=cand_params,
      cand_input=cand_input,
      weight_map=_require_filled(WEIGHT_MAP, "WEIGHT_MAP"),
      fwd_tol=_require_filled(fwd_tol, "fwd_tol"),
      grad_tol=_require_filled(grad_tol, "grad_tol"),
      cotangent_seed=COTANGENT_SEED,
  )
  return report, (cand_fn, cand_params, cand_input), len(ref_params)


class ComponentParityTest(unittest.TestCase):
  """Template parity tests with method-level torch guards and negative controls."""

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference package")
  def test_fp32_forward_and_every_gradient_leaf_match_reference(self) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    report, _, num_params = _run_parity(torch.float32, jnp.float32, fwd_tol=FWD_TOL, grad_tol=GRAD_TOL)
    self.assertEqual(report.failures, [])
    self.assertEqual(len(report.grads), num_params)

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference package")
  def test_negative_control_one_perturbed_leaf_breaks_parity(self) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    report, (cand_fn, cand_params, cand_input), _ = _run_parity(
        torch.float32, jnp.float32, fwd_tol=FWD_TOL, grad_tol=GRAD_TOL
    )
    self.assertEqual(report.failures, [], "positive case must pass on the same inputs")
    leaf = _require_filled(NEGCTL_LEAF, "NEGCTL_LEAF")
    fired, metrics = layer_parity.negative_control_fires(
        ref_out=report.ref_out,
        cand_fn=cand_fn,
        cand_params=cand_params,
        cand_input=cand_input,
        fwd_tol=_require_filled(FWD_TOL, "FWD_TOL"),
        target_path=leaf,
        rel_scale=NEGCTL_REL_SCALE,
        seed=NEGCTL_SEED,
    )
    self.assertTrue(fired, f"perturbing {leaf} left the output within tolerance: {metrics}")

  @unittest.skipUnless(HAS_TORCH, "Requires PyTorch reference package")
  def test_fp64_frobenius_matches_reference(self) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    with jax.enable_x64(True):
      report, _, num_params = _run_parity(torch.float64, jnp.float64, fwd_tol=FP64_TOL, grad_tol=FP64_TOL)
      self.assertTrue(jax.config.jax_enable_x64)
    self.assertEqual(report.failures, [])
    self.assertEqual(len(report.grads), num_params)


if __name__ == "__main__":
  unittest.main()
