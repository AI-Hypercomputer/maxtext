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

Tolerances are explicit arguments taken from the approved design tolerance
table; nothing here has a default tolerance. Parameter gradients are partitioned
into five explicit classes (`compared`, `non_gradient`, `discrete_only`,
`frozen`, `padded`) so every reference parameter and every candidate leaf is
accounted for without silent skips.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set as AbstractSet
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

torch: Any
try:
  import torch  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  HAS_TORCH = False

from tests.utils.parity import negctl

# Numeric parity ceilings from jax_ml_parity.md §5.
FP32_ATOL_CEILING = 1e-5
FP32_RTOL_CEILING = 1e-5
FP32_FROB_REL_CEILING = 1e-5
FP64_FROB_REL_CEILING = 1e-10


@dataclasses.dataclass(frozen=True)
class Tolerance:
  """One row of the design tolerance table.

  Passing requires, elementwise, |c - r| <= atol * max(1, |r|_inf) + rtol * |r|,
  and ||c - r||_2 <= frob_rel * max(1, ||r||_2).
  """

  atol: float
  rtol: float
  frob_rel: float

  def __post_init__(self) -> None:
    if not (np.isfinite(self.atol) and np.isfinite(self.rtol) and np.isfinite(self.frob_rel)):
      raise ValueError(f"Tolerance values must be finite, got {self}")
    if self.atol < 0.0 or self.rtol < 0.0 or self.frob_rel < 0.0:
      raise ValueError(f"Tolerance values must be >= 0, got {self}")
    if self.atol > FP32_ATOL_CEILING or self.rtol > FP32_RTOL_CEILING or self.frob_rel > FP32_FROB_REL_CEILING:
      raise ValueError(
          "Tolerance exceeds approved fp32 ceilings "
          f"({FP32_ATOL_CEILING}, {FP32_RTOL_CEILING}, {FP32_FROB_REL_CEILING}): {self}"
      )


FP32_TOLERANCE = Tolerance(
    atol=FP32_ATOL_CEILING,
    rtol=FP32_RTOL_CEILING,
    frob_rel=FP32_FROB_REL_CEILING,
)
FP64_TOLERANCE = Tolerance(
    atol=FP64_FROB_REL_CEILING,
    rtol=FP64_FROB_REL_CEILING,
    frob_rel=FP64_FROB_REL_CEILING,
)


def _enforce_fp64_ceiling(tol: Tolerance, label: str) -> None:
  """Enforces the 1e-10 fp64 ceiling when comparing float64 tensors."""
  if tol.atol > FP64_FROB_REL_CEILING or tol.rtol > FP64_FROB_REL_CEILING or tol.frob_rel > FP64_FROB_REL_CEILING:
    raise ValueError(f"{label} on float64 tensors exceeds FP64_FROB_REL_CEILING ({FP64_FROB_REL_CEILING}): {tol}")


@dataclasses.dataclass(frozen=True)
class Metrics:
  """Summary metrics for a single array comparison."""

  max_abs: float
  ref_inf: float
  frob_rel: float
  finite: bool


@dataclasses.dataclass(frozen=True)
class GradientClasses:
  """Explicit five-class partition of reference parameter names.

  Every reference parameter must belong to exactly one of the five sets:
    compared: Parameters with non-None reference gradients compared within
      tolerance against their candidate counterparts.
    non_gradient: Reference nn.Parameter state tensors (e.g. routing biases)
      whose reference autograd gradient is None and that are excluded from
      candidate gradient comparison.
    discrete_only: Parameters used only on discrete index-selection paths
      whose reference autograd gradient is None and whose candidate gradient
      must be identically zero.
    frozen: Parameters frozen during training (reference requires_grad=False /
      grad=None, candidate gradient identically zero).
    padded: Parameters whose candidate gradient has trailing zero-padded rows
      along axis 0; active rows [:n] are compared within tolerance and trailing
      rows [n:] must be identically zero.
  """

  compared: AbstractSet[str] = frozenset()
  non_gradient: AbstractSet[str] = frozenset()
  discrete_only: AbstractSet[str] = frozenset()
  frozen: AbstractSet[str] = frozenset()
  padded: AbstractSet[str] = frozenset()

  def __post_init__(self) -> None:
    object.__setattr__(self, "compared", frozenset(self.compared))
    object.__setattr__(self, "non_gradient", frozenset(self.non_gradient))
    object.__setattr__(self, "discrete_only", frozenset(self.discrete_only))
    object.__setattr__(self, "frozen", frozenset(self.frozen))
    object.__setattr__(self, "padded", frozenset(self.padded))

  @property
  def expected_grad_none(self) -> frozenset[str]:
    """Returns the set of parameters expected to have grad=None in PyTorch."""
    return frozenset(self.non_gradient) | frozenset(self.discrete_only) | frozenset(self.frozen)


def compare(cand: np.ndarray, ref: np.ndarray, tol: Tolerance) -> tuple[bool, Metrics]:
  """Compares candidate with reference in float64; raises on a shape or dtype mismatch."""
  c_raw = np.asarray(cand)
  r_raw = np.asarray(ref)
  if not (jnp.issubdtype(c_raw.dtype, jnp.floating) and jnp.issubdtype(r_raw.dtype, jnp.floating)):
    raise TypeError(f"compare requires floating-point arrays, got cand {c_raw.dtype} and ref {r_raw.dtype}")
  if c_raw.shape != r_raw.shape:
    raise ValueError(f"Shape mismatch: candidate {c_raw.shape} vs reference {r_raw.shape}")
  if c_raw.size == 0:
    raise ValueError(f"compare requires non-empty arrays, got shape {c_raw.shape}")
  c = np.asarray(c_raw, dtype=np.float64)
  r = np.asarray(r_raw, dtype=np.float64)
  if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
    return False, Metrics(np.nan, np.nan, np.nan, finite=False)
  diff = np.abs(c - r)
  ref_inf = float(np.max(np.abs(r)))
  ref_l2 = float(np.linalg.norm(r))
  frob_rel = float(np.linalg.norm(diff)) / max(1.0, ref_l2)
  metrics = Metrics(float(np.max(diff)), ref_inf, frob_rel, finite=True)
  elementwise_ok = bool(np.all(diff <= tol.atol * max(1.0, ref_inf) + tol.rtol * np.abs(r)))
  return elementwise_ok and frob_rel <= tol.frob_rel, metrics


def compare_padded_rows(cand: np.ndarray, ref: np.ndarray, num_rows: int, tol: Tolerance) -> tuple[bool, Metrics]:
  """Compares active rows `cand[:num_rows]` with `ref` and asserts `cand[num_rows:] == 0`.

  Args:
    cand: Candidate array or gradient with shape `(total_rows, ...)`, where
      `total_rows > num_rows`.
    ref: Reference array or gradient with shape `(num_rows, ...)`.
    num_rows: Number of valid unpadded rows along axis 0.
    tol: Tolerance for the active slice `cand[:num_rows]`.

  Returns:
    `(passed, metrics)` where `passed` requires `compare(cand[:num_rows], ref, tol)`
    to pass and every element of `cand[num_rows:]` to be finite and identically 0.

  Raises:
    TypeError: If `cand` or `ref` is not a floating-point array.
    ValueError: If array ranks are < 1, arrays are empty, `num_rows <= 0`,
      `cand.shape[0] <= num_rows` (no padding rows), `ref.shape[0] != num_rows`,
      or trailing dimensions mismatch.
  """
  c = np.asarray(cand)
  r = np.asarray(ref)
  if not (jnp.issubdtype(c.dtype, jnp.floating) and jnp.issubdtype(r.dtype, jnp.floating)):
    raise TypeError(f"compare_padded_rows requires floating-point arrays, got cand {c.dtype} and ref {r.dtype}")
  if c.ndim < 1 or r.ndim < 1 or c.size == 0 or r.size == 0:
    raise ValueError(f"compare_padded_rows requires non-empty rank >= 1 arrays, got cand {c.shape} and ref {r.shape}")
  if num_rows <= 0 or c.shape[0] <= num_rows:
    raise ValueError(
        f"Invalid num_rows={num_rows} for padded candidate shape {c.shape}; requires 0 < num_rows < cand.shape[0]"
    )
  if r.shape != (num_rows, *c.shape[1:]):
    raise ValueError(f"Padded shape mismatch: candidate {c.shape} with num_rows={num_rows} vs reference {r.shape}")
  active_ok, metrics = compare(c[:num_rows], r, tol)
  padding = np.asarray(c[num_rows:], dtype=np.float64)
  if not np.all(np.isfinite(padding)):
    return False, Metrics(np.nan, np.nan, np.nan, finite=False)
  padding_zero_ok = bool(np.all(padding == 0.0))
  return active_ok and padding_zero_ok, metrics


def validate_gradient_classes(
    classes: GradientClasses,
    param_names: AbstractSet[str],
    ref_grads: Mapping[str, torch.Tensor | None],
) -> None:
  """Verifies that `classes` partitions `param_names` and matches the `grad=None` set.

  Args:
    classes: Declared five-class parameter partition.
    param_names: Complete set of reference parameter names.
    ref_grads: Mapping from reference parameter name to its autograd gradient
      (or `None` when unused / frozen).

  Raises:
    ValueError: If `param_names` is empty, if the five classes overlap, omit or
      invent parameters, if `compared | padded` is empty, or if the set of
      parameters with `ref_grads[name] is None` differs from
      `classes.expected_grad_none`.
  """
  expected = frozenset(param_names)
  if not expected:
    raise ValueError("param_names must be non-empty")
  named_sets: list[tuple[str, frozenset[str]]] = [
      ("compared", frozenset(classes.compared)),
      ("non_gradient", frozenset(classes.non_gradient)),
      ("discrete_only", frozenset(classes.discrete_only)),
      ("frozen", frozenset(classes.frozen)),
      ("padded", frozenset(classes.padded)),
  ]
  overlaps: list[str] = []
  for idx, (name_a, set_a) in enumerate(named_sets):
    for name_b, set_b in named_sets[idx + 1 :]:
      common = sorted(set_a & set_b)
      if common:
        overlaps.append(f"{name_a}&{name_b}={common}")
  declared_union = frozenset().union(*(s for _, s in named_sets))
  missing = sorted(expected - declared_union)
  unknown = sorted(declared_union - expected)
  if overlaps or missing or unknown:
    raise ValueError(
        f"GradientClasses must partition the parameter set: overlaps={overlaps}, missing={missing}, unknown={unknown}"
    )
  if not (classes.compared or classes.padded):
    raise ValueError("GradientClasses must include at least one compared or padded parameter")
  if frozenset(ref_grads.keys()) != expected:
    raise ValueError(
        "ref_grads keys must equal param_names: "
        f"missing={sorted(expected - frozenset(ref_grads.keys()))}, "
        f"extra={sorted(frozenset(ref_grads.keys()) - expected)}"
    )
  actual_grad_none = frozenset(name for name, g in ref_grads.items() if g is None)
  expected_grad_none = classes.expected_grad_none
  if actual_grad_none != expected_grad_none:
    unexpected_none = sorted(actual_grad_none - expected_grad_none)
    unexpected_present = sorted(expected_grad_none - actual_grad_none)
    raise ValueError(
        "Reference grad=None set does not match declared non_gradient | discrete_only | frozen: "
        f"unexpected_grad_none={unexpected_none}, "
        f"expected_grad_none_but_present={unexpected_present}"
    )


def check_gradients_by_class(
    *,
    ref_params: Mapping[str, torch.Tensor],
    ref_grads: Mapping[str, torch.Tensor | None],
    cand_grads: Mapping[str, np.ndarray],
    weight_map: Mapping[str, str],
    grad_classes: GradientClasses,
    grad_tol: Tolerance,
) -> tuple[dict[str, Metrics], list[str]]:
  """Checks parameter gradients across all five gradient classes.

  Args:
    ref_params: Reference parameter tensors keyed by reference parameter name.
    ref_grads: Reference gradients keyed by reference parameter name (`None` for
      unused or frozen parameters).
    cand_grads: Flattened candidate gradient leaves keyed by `negctl.leaf_path`.
    weight_map: Reference parameter name -> candidate leaf path (with optional
      `:T` suffix to transpose 2D gradients). Must cover every reference
      parameter outside `grad_classes.non_gradient` and every candidate leaf,
      and must not include `non_gradient` keys.
    grad_classes: Explicit five-class partition of `ref_params`.
    grad_tol: Numerical tolerance for `compared` and `padded` gradients.

  Returns:
    `(grad_metrics, failures)` where `grad_metrics` holds a `Metrics` entry for
    every parameter in `compared | padded | discrete_only | frozen`.
  """
  for cand_key, raw_cand_g in cand_grads.items():
    if raw_cand_g is None or isinstance(raw_cand_g, torch.Tensor):
      raise TypeError(f"Candidate gradient {cand_key!r} is {type(raw_cand_g).__name__}; expected JAX/NumPy float array")
    arr_cand_g = np.asarray(raw_cand_g)
    if not jnp.issubdtype(arr_cand_g.dtype, jnp.floating) or arr_cand_g.size == 0:
      raise TypeError(
          f"Candidate gradient {cand_key!r} must be a non-empty floating-point array, "
          f"got dtype={arr_cand_g.dtype}, shape={arr_cand_g.shape}"
      )
  mapped_list = [spec.removesuffix(":T") for spec in weight_map.values()]
  mapped = set(mapped_list)
  required_ref = set(ref_params) - set(grad_classes.non_gradient)
  unmapped_ref = sorted(required_ref - set(weight_map))
  unknown_ref = sorted(set(weight_map) - set(ref_params))
  non_grad_mapped = sorted(set(weight_map) & set(grad_classes.non_gradient))
  unmapped_cand = sorted(set(cand_grads) - mapped)
  unknown_cand = sorted(mapped - set(cand_grads))
  has_dup_cand = len(mapped_list) != len(mapped)
  if unmapped_ref or unknown_ref or non_grad_mapped or unmapped_cand or unknown_cand or has_dup_cand:
    raise ValueError(
        "weight_map must cover required reference parameters and every candidate leaf once: "
        f"unmapped_ref={unmapped_ref}, unknown_ref={unknown_ref}, "
        f"non_grad_mapped={non_grad_mapped}, unmapped_cand={unmapped_cand}, "
        f"unknown_cand={unknown_cand}, duplicate_cand={has_dup_cand}"
    )
  validate_gradient_classes(grad_classes, set(ref_params), ref_grads)
  is_fp64 = (
      any(v.dtype == torch.float64 for v in ref_params.values())
      or any(g is not None and g.dtype == torch.float64 for g in ref_grads.values())
      or any(np.asarray(g).dtype == np.float64 for g in cand_grads.values())
  )
  if is_fp64:
    _enforce_fp64_ceiling(grad_tol, "grad_tol")

  grad_metrics: dict[str, Metrics] = {}
  failures: list[str] = []
  for name, ref_param in ref_params.items():
    if name in grad_classes.non_gradient:
      continue
    spec = weight_map[name]
    cand_g = np.asarray(cand_grads[spec.removesuffix(":T")])
    if spec.endswith(":T"):
      cand_g = cand_g.T
    if name in grad_classes.compared:
      ref_g = ref_grads[name]
      if ref_g is None:
        raise ValueError(f"Compared parameter {name!r} has ref_grad=None")
      if tuple(ref_g.shape) != tuple(ref_param.shape):
        raise ValueError(
            f"Reference gradient shape mismatch for compared parameter {name!r}: "
            f"grad {tuple(ref_g.shape)} vs param {tuple(ref_param.shape)}"
        )
      ok, metrics = compare(cand_g, ref_g.detach().cpu().to(torch.float64).numpy(), grad_tol)
      grad_metrics[name] = metrics
      if not ok:
        failures.append(f"grad {name}: {metrics}")
    elif name in grad_classes.padded:
      ref_g = ref_grads[name]
      if ref_g is None:
        raise ValueError(f"Padded parameter {name!r} has ref_grad=None")
      if tuple(ref_g.shape) != tuple(ref_param.shape):
        raise ValueError(
            f"Reference gradient shape mismatch for padded parameter {name!r}: "
            f"grad {tuple(ref_g.shape)} vs param {tuple(ref_param.shape)}"
        )
      ref_g_np = ref_g.detach().cpu().to(torch.float64).numpy()
      ok, metrics = compare_padded_rows(cand_g, ref_g_np, ref_g_np.shape[0], grad_tol)
      grad_metrics[name] = metrics
      if not ok:
        failures.append(f"padded grad {name}: {metrics}")
    elif name in grad_classes.discrete_only or name in grad_classes.frozen:
      cls_label = "discrete_only" if name in grad_classes.discrete_only else "frozen"
      ref_shape = tuple(ref_param.shape)
      if cand_g.shape != ref_shape:
        raise ValueError(
            f"Shape mismatch on {cls_label} parameter {name!r}: candidate {cand_g.shape} vs reference {ref_shape}"
        )
      cand_g64 = np.asarray(cand_g, dtype=np.float64)
      finite = bool(np.all(np.isfinite(cand_g64)))
      max_abs = float(np.max(np.abs(cand_g64))) if finite else np.nan
      frob_rel = float(np.linalg.norm(cand_g64)) if finite else np.nan
      metrics = Metrics(max_abs=max_abs, ref_inf=0.0, frob_rel=frob_rel, finite=finite)
      grad_metrics[name] = metrics
      if not (finite and max_abs == 0.0):
        failures.append(f"{cls_label} grad {name} not identically zero: {metrics}")
    else:
      failures.append(f"unclassified parameter {name!r}")
  return grad_metrics, failures


def _assert_jax_tree(tree: Any, label: str) -> None:
  """Blocks self-comparison and vacuous empty trees: every leaf must be a non-empty jax.Array."""
  leaves_with_path = jax.tree_util.tree_flatten_with_path(tree)[0]
  if not leaves_with_path:
    raise ValueError(f"{label} pytree has no leaves")
  for path, leaf in leaves_with_path:
    if not isinstance(leaf, jax.Array):
      raise TypeError(f"{label} leaf {negctl.leaf_path(path)!r} is {type(leaf).__name__}, not jax.Array")
    if leaf.size == 0:
      raise ValueError(f"{label} leaf {negctl.leaf_path(path)!r} is empty (shape={leaf.shape})")


@dataclasses.dataclass
class ParityReport:
  """Result of a forward and all-leaf gradient parity check."""

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
    weight_map: Mapping[str, str],
    grad_classes: GradientClasses,
    fwd_tol: Tolerance,
    grad_tol: Tolerance,
    cotangent_seed: int,
) -> ParityReport:
  """Checks the forward output and the gradient of every parameter leaf by class.

  Both sides are differentiated through the same random projection of the
  output, so no gradient term cancels by symmetry.

  Args:
    ref_fn: Reference forward function `(ref_params, ref_input) -> torch.Tensor`.
    ref_params: Reference parameter dict.
    ref_input: Reference input tensor.
    cand_fn: Candidate forward function `(cand_params, cand_input) -> jax.Array`.
    cand_params: Candidate parameter pytree.
    cand_input: Candidate input pytree.
    weight_map: Reference parameter name -> candidate leaf path. A `:T` suffix
      on the candidate path compares the transposed candidate gradient.
    grad_classes: Explicit five-class partition of `ref_params`.
    fwd_tol: Tolerance for the forward output.
    grad_tol: Tolerance for `compared` and `padded` parameter gradients.
    cotangent_seed: Seed for the random output cotangent.

  Returns:
    A `ParityReport` with forward metrics, per-parameter gradient metrics,
    failure descriptions, and the float64 reference output.
  """
  _assert_jax_tree(cand_params, "Candidate params")
  _assert_jax_tree(cand_input, "Candidate input")

  cloned_ref_params = {k: v.detach().clone().requires_grad_(k not in grad_classes.frozen) for k, v in ref_params.items()}
  ref_out = ref_fn(cloned_ref_params, ref_input)
  cand_out = cand_fn(cand_params, cand_input)
  _assert_jax_tree(cand_out, "Candidate output")
  if (
      cand_out.dtype == jnp.float64
      or ref_out.dtype == torch.float64
      or any(v.dtype == torch.float64 for v in cloned_ref_params.values())
      or any(leaf.dtype == jnp.float64 for _, leaf in jax.tree_util.tree_flatten_with_path(cand_params)[0])
  ):
    _enforce_fp64_ceiling(fwd_tol, "fwd_tol")
    _enforce_fp64_ceiling(grad_tol, "grad_tol")
  ref_out_np = ref_out.detach().cpu().to(torch.float64).numpy()
  failures: list[str] = []
  fwd_ok, fwd_metrics = compare(np.asarray(cand_out), ref_out_np, fwd_tol)
  if not fwd_ok:
    failures.append(f"forward: {fwd_metrics}")

  cotangent = np.random.default_rng(cotangent_seed).standard_normal(ref_out_np.shape)
  ref_loss = (ref_out * torch.as_tensor(cotangent, dtype=ref_out.dtype)).sum()
  active_names = [k for k, v in cloned_ref_params.items() if v.requires_grad]
  active_tensors = [cloned_ref_params[k] for k in active_names]
  active_grads = torch.autograd.grad(ref_loss, active_tensors, allow_unused=True) if active_tensors else ()
  ref_grad_map: dict[str, torch.Tensor | None] = {k: None for k in cloned_ref_params}
  for k, g in zip(active_names, active_grads):
    ref_grad_map[k] = g

  cand_ct = jnp.asarray(cotangent, dtype=cand_out.dtype)
  cand_grad_tree = jax.grad(lambda p: jnp.sum(cand_fn(p, cand_input) * cand_ct))(cand_params)
  _assert_jax_tree(cand_grad_tree, "Candidate gradients")
  cand_grads = {negctl.leaf_path(p): np.asarray(g) for p, g in jax.tree_util.tree_flatten_with_path(cand_grad_tree)[0]}

  grad_metrics, grad_failures = check_gradients_by_class(
      ref_params=cloned_ref_params,
      ref_grads=ref_grad_map,
      cand_grads=cand_grads,
      weight_map=weight_map,
      grad_classes=grad_classes,
      grad_tol=grad_tol,
  )
  failures.extend(grad_failures)
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
  """Perturbs one candidate leaf and returns (comparison failed with finite output, metrics).

  Uses the same predicate and tolerance as the positive forward check. A True
  result means the parity test can detect a wrong weight of this size with a
  finite output; non-finite outputs return False with `metrics.finite=False`.
  """
  _assert_jax_tree(cand_params, "Candidate params")
  _assert_jax_tree(cand_input, "Candidate input")
  perturbed = negctl.perturb_one_leaf(cand_params, target_path=target_path, rel_scale=rel_scale, seed=seed)
  perturbed_out = cand_fn(perturbed, cand_input)
  _assert_jax_tree(perturbed_out, "Perturbed candidate output")
  if perturbed_out.dtype == jnp.float64 or any(
      leaf.dtype == jnp.float64 for _, leaf in jax.tree_util.tree_flatten_with_path(cand_params)[0]
  ):
    _enforce_fp64_ceiling(fwd_tol, "fwd_tol")
  ok, metrics = compare(np.asarray(perturbed_out), ref_out, fwd_tol)
  return (not ok) and metrics.finite, metrics


def reference_noise_floor(
    ref_fn: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    ref_params: dict[str, torch.Tensor],
    ref_input: torch.Tensor,
) -> Metrics:
  """Measures the reference's own float32-vs-float64 difference (tolerance calibration)."""

  def run(dtype: torch.dtype) -> np.ndarray:
    params = {k: v.detach().to(dtype) if v.is_floating_point() else v.detach() for k, v in ref_params.items()}
    inputs = ref_input.detach().to(dtype) if ref_input.is_floating_point() else ref_input.detach()
    with torch.no_grad():
      out = ref_fn(params, inputs)
    return out.detach().cpu().to(torch.float64).numpy()

  zero_tol = Tolerance(atol=0.0, rtol=0.0, frob_rel=0.0)
  _, metrics = compare(run(torch.float32), run(torch.float64), zero_tol)
  if not metrics.finite:
    raise ValueError(f"Reference noise floor is non-finite: {metrics}")
  return metrics
