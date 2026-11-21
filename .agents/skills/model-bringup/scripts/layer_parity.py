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

"""Forward and all-leaf gradient parity between a PyTorch reference and a JAX candidate.

Used by parity tests built from `.agents/skills/model-bringup-m3/templates/parity_test.py`. Tolerances are
explicit arguments taken from the approved design tolerance table; nothing here
has a default tolerance.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import importlib.util
import pathlib
import sys
from typing import Any, Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
  import torch


def _load_sibling(name: str) -> Any:
  """Loads a sibling helper module by filename without requiring sys.path edits."""
  qual = f"_maxtext_bringup_{name}"
  if qual in sys.modules:
    return sys.modules[qual]
  path = pathlib.Path(__file__).resolve().with_name(f"{name}.py")
  spec = importlib.util.spec_from_file_location(qual, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load sibling helper {path}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[qual] = mod
  spec.loader.exec_module(mod)
  return mod


def _require_torch() -> Any:
  """Imports torch lazily so pure-JAX/NumPy callers work in torch-free venvs."""
  try:
    import torch as _torch  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError("torch is required for PyTorch-to-JAX reference parity checks") from exc
  return _torch


negctl = _load_sibling("negctl")


@dataclasses.dataclass(frozen=True)
class Tolerance:
  """One row of the design tolerance table.

  Passing requires, elementwise, |c - r| <= atol * max(1, |r|_inf) + rtol * |r|,
  and ||c - r||_2 <= frob_rel * max(1, ||r||_2).
  """

  atol: float
  rtol: float
  frob_rel: float


@dataclasses.dataclass(frozen=True)
class Metrics:
  max_abs: float
  ref_inf: float
  frob_rel: float
  finite: bool


def compare(cand: np.ndarray, ref: np.ndarray, tol: Tolerance) -> tuple[bool, Metrics]:
  """Compares candidate with reference in float64; raises on a shape mismatch."""
  if cand.shape != ref.shape:
    raise ValueError(f"Shape mismatch: candidate {cand.shape} vs reference {ref.shape}")
  c = np.asarray(cand, dtype=np.float64)
  r = np.asarray(ref, dtype=np.float64)
  if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
    return False, Metrics(np.nan, np.nan, np.nan, finite=False)
  diff = np.abs(c - r)
  ref_inf = float(np.max(np.abs(r)))
  ref_l2 = float(np.linalg.norm(r))
  frob_rel = float(np.linalg.norm(diff)) / max(1.0, ref_l2)
  metrics = Metrics(float(np.max(diff)), ref_inf, frob_rel, finite=True)
  elementwise_ok = bool(np.all(diff <= tol.atol * max(1.0, ref_inf) + tol.rtol * np.abs(r)))
  return elementwise_ok and frob_rel <= tol.frob_rel, metrics


def _assert_jax_tree(tree: Any, label: str) -> None:
  """Blocks self-comparison: every array leaf of the candidate must be a jax.Array."""
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    if not isinstance(leaf, jax.Array):
      raise TypeError(f"{label} leaf {negctl.leaf_path(path)!r} is {type(leaf).__name__}, not jax.Array")


@dataclasses.dataclass
class ParityReport:
  forward: Metrics
  grads: dict[str, Metrics]
  failures: list[str]
  ref_out: np.ndarray

  @property
  def passed(self) -> bool:
    return not self.failures


def check_parity(
    *,
    ref_fn: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    ref_params: dict[str, torch.Tensor],
    ref_input: torch.Tensor,
    cand_fn: Callable[[Any, Any], jax.Array],
    cand_params: Any,
    cand_input: Any,
    weight_map: dict[str, str],
    fwd_tol: Tolerance,
    grad_tol: Tolerance,
    cotangent_seed: int,
) -> ParityReport:
  """Checks the forward output and the gradient of every parameter leaf.

  Both sides are differentiated through the same random projection of the
  output, so no gradient term cancels by symmetry.

  Args:
    weight_map: Reference parameter name -> candidate leaf path, covering every
      reference parameter and every candidate leaf exactly once. A `:T` suffix
      on the candidate path compares the transposed candidate gradient.
  """
  torch = _require_torch()
  _assert_jax_tree(cand_params, "Candidate params")
  _assert_jax_tree(cand_input, "Candidate input")
  cand_leaves = {negctl.leaf_path(p): leaf for p, leaf in jax.tree_util.tree_flatten_with_path(cand_params)[0]}
  mapped = {spec.removesuffix(":T") for spec in weight_map.values()}
  if set(weight_map) != set(ref_params) or mapped != set(cand_leaves):
    raise ValueError(
        "weight_map must cover every parameter exactly once: "
        f"unmapped ref={sorted(set(ref_params) - set(weight_map))}, "
        f"unknown ref={sorted(set(weight_map) - set(ref_params))}, "
        f"unmapped cand={sorted(set(cand_leaves) - mapped)}, "
        f"unknown cand={sorted(mapped - set(cand_leaves))}"
    )

  ref_params = {k: v.detach().clone().requires_grad_(True) for k, v in ref_params.items()}
  ref_out = ref_fn(ref_params, ref_input)
  cand_out = cand_fn(cand_params, cand_input)
  _assert_jax_tree(cand_out, "Candidate output")
  ref_out_np = ref_out.detach().cpu().to(torch.float64).numpy()
  failures = []
  fwd_ok, fwd_metrics = compare(np.asarray(cand_out), ref_out_np, fwd_tol)
  if not fwd_ok:
    failures.append(f"forward: {fwd_metrics}")

  cotangent = np.random.default_rng(cotangent_seed).standard_normal(ref_out_np.shape)
  ref_loss = (ref_out * torch.as_tensor(cotangent, dtype=ref_out.dtype)).sum()
  ref_grads = torch.autograd.grad(ref_loss, list(ref_params.values()))
  cand_ct = jnp.asarray(cotangent, dtype=cand_out.dtype)
  cand_grad_tree = jax.grad(lambda p: jnp.sum(cand_fn(p, cand_input) * cand_ct))(cand_params)
  cand_grads = {negctl.leaf_path(p): np.asarray(g) for p, g in jax.tree_util.tree_flatten_with_path(cand_grad_tree)[0]}

  grad_metrics = {}
  for name, ref_g in zip(ref_params, ref_grads):
    spec = weight_map[name]
    cand_g = cand_grads[spec.removesuffix(":T")]
    if spec.endswith(":T"):
      cand_g = cand_g.T
    ok, metrics = compare(cand_g, ref_g.detach().cpu().to(torch.float64).numpy(), grad_tol)
    grad_metrics[name] = metrics
    if not ok:
      failures.append(f"grad {name}: {metrics}")
  return ParityReport(fwd_metrics, grad_metrics, failures, ref_out_np)


def negative_control_fires(
    *,
    ref_out: np.ndarray,
    cand_fn: Callable[[Any, Any], jax.Array],
    cand_params: Any,
    cand_input: Any,
    fwd_tol: Tolerance,
    target_path: str,
    rel_scale: float,
    seed: int,
) -> tuple[bool, Metrics]:
  """Perturbs one candidate leaf and returns (comparison failed, metrics).

  Uses the same predicate and tolerance as the positive forward check. A True
  result means the parity test can detect a wrong weight of this size.
  """
  perturbed = negctl.perturb_one_leaf(cand_params, target_path=target_path, rel_scale=rel_scale, seed=seed)
  ok, metrics = compare(np.asarray(cand_fn(perturbed, cand_input)), ref_out, fwd_tol)
  return not ok, metrics


def reference_noise_floor(
    ref_fn: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    ref_params: dict[str, torch.Tensor],
    ref_input: torch.Tensor,
) -> Metrics:
  """Measures the reference's own float32-vs-float64 difference (tolerance calibration)."""

  torch = _require_torch()

  def run(dtype: Any) -> Any:
    params = {k: v.to(dtype) for k, v in ref_params.items()}
    inputs = ref_input.to(dtype) if ref_input.is_floating_point() else ref_input
    with torch.no_grad():
      return ref_fn(params, inputs)

  zero_tol = Tolerance(atol=0.0, rtol=0.0, frob_rel=0.0)
  return compare(run(torch.float32).numpy(), run(torch.float64).numpy(), zero_tol)[1]


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for layer_parity tolerance comparison self-check."""
  parser = argparse.ArgumentParser(
      description="Compare candidate and reference activations/gradients with negative controls."
  )
  parser.add_argument(
      "--self-test",
      action="store_true",
      help="Run a pure-NumPy/JAX self-check without requiring torch.",
  )
  args = parser.parse_args(argv)
  if args.self_test:
    tol = Tolerance(atol=1e-5, rtol=1e-5, frob_rel=1e-5)
    ref = np.ones((2, 4), dtype=np.float32)
    ok, metrics = compare(ref, ref, tol)
    if not ok or not metrics.finite:
      raise AssertionError(f"Self-test compare failed: {metrics}")
    print(f"LAYER_PARITY_SELF_TEST PASS max_abs={metrics.max_abs:.3e}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
