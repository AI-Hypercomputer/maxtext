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

"""Model-agnostic forward and all-leaf gradient parity checks in JAX/NumPy.

Tolerances are explicit arguments taken from each model's approved design
tolerance table; nothing in this shared package hardcodes model-specific
ceilings. Supports both torch-free golden-archive checks (`hashed_normal` +
`check_forward_and_grads`) and live PyTorch reference parity (`check_parity` +
`GradientClasses` + `reference_noise_floor`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
import dataclasses
import hashlib
import math
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

_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
# Irwin-Hall sum of four uniforms has variance 4 / 12; this scale gives unit variance.
_IRWIN_HALL_TERMS = 4
_IRWIN_HALL_SCALE = math.sqrt(3.0)


@dataclasses.dataclass(frozen=True)
class Tolerance:
  """One row of a design tolerance table.

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
  """Compares candidate with reference in float64; raises on a shape, dtype, or zero-size error."""
  c_raw = np.asarray(cand)
  r_raw = np.asarray(ref)
  if not (negctl.is_floating_dtype(c_raw.dtype) and negctl.is_floating_dtype(r_raw.dtype)):
    raise TypeError(f"compare requires floating-point arrays, got cand {c_raw.dtype} and ref {r_raw.dtype}")
  if c_raw.shape != r_raw.shape:
    raise ValueError(f"Shape mismatch: candidate {c_raw.shape} vs reference {r_raw.shape}")
  if c_raw.size == 0:
    raise ValueError(f"Cannot compare zero-size arrays of shape {c_raw.shape}")
  c = np.asarray(c_raw, dtype=np.float64)
  r = np.asarray(r_raw, dtype=np.float64)
  if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
    return False, Metrics(np.nan, np.nan, np.nan, finite=False)
  diff = np.abs(c - r)
  ref_inf = float(np.max(np.abs(r)))
  max_abs = float(np.max(diff))
  frob_rel = float(np.linalg.norm(diff)) / max(1.0, float(np.linalg.norm(r)))
  elementwise_ok = bool(np.all(diff <= tol.atol * max(1.0, ref_inf) + tol.rtol * np.abs(r)))
  metrics = Metrics(max_abs=max_abs, ref_inf=ref_inf, frob_rel=frob_rel, finite=True)
  return elementwise_ok and frob_rel <= tol.frob_rel, metrics


def _splitmix64(counter: np.ndarray) -> np.ndarray:
  """Vectorized SplitMix64 finalizer on uint64 counters (wrapping arithmetic)."""
  z = counter + _SPLITMIX_GAMMA
  z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_MUL1
  z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_MUL2
  return z ^ (z >> np.uint64(31))


def hashed_normal(
    key: str,
    shape: Sequence[int],
    *,
    mean: float,
    std: float,
    dtype: np.typing.DTypeLike = np.float32,
) -> np.ndarray:
  """Returns a platform-independent approximately standard-normal tensor derived from `key`.

  Uses only integer hashing, additions, and multiplications (no libm transcendental functions),
  so both golden generators and offline JAX tests reproduce identical bits across platforms.
  """
  size = math.prod(shape)
  seed = np.uint64(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little"))
  counters = np.arange(size * _IRWIN_HALL_TERMS, dtype=np.uint64) + seed
  bits = _splitmix64(counters)
  uniforms = (bits >> np.uint64(11)).astype(np.float64) * (2.0**-53)
  sums = uniforms.reshape(size, _IRWIN_HALL_TERMS).sum(axis=1)
  normal = (sums - _IRWIN_HALL_TERMS / 2.0) * _IRWIN_HALL_SCALE
  return (normal * std + mean).astype(dtype).reshape(tuple(shape))


def compare_padded_rows(cand: np.ndarray, ref: np.ndarray, num_rows: int, tol: Tolerance) -> tuple[bool, Metrics]:
  """Compares active rows `cand[:num_rows]` with `ref` and asserts `cand[num_rows:] == 0`."""
  c = np.asarray(cand)
  r = np.asarray(ref)
  if not (negctl.is_floating_dtype(c.dtype) and negctl.is_floating_dtype(r.dtype)):
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
    ref_grads: Mapping[str, Any],
) -> None:
  """Verifies that `classes` partitions `param_names` and matches the `grad=None` set."""
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
    ref_params: Mapping[str, Any],
    ref_grads: Mapping[str, Any],
    cand_grads: Mapping[str, np.ndarray],
    weight_map: Mapping[str, str],
    grad_classes: GradientClasses,
    grad_tol: Tolerance,
    fp64_tol_validator: Callable[[Tolerance, str], None] | None = None,
) -> tuple[dict[str, Metrics], list[str]]:
  """Checks parameter gradients across all five gradient classes."""
  for cand_key, raw_cand_g in cand_grads.items():
    if raw_cand_g is None or (HAS_TORCH and isinstance(raw_cand_g, torch.Tensor)):
      raise TypeError(f"Candidate gradient {cand_key!r} is {type(raw_cand_g).__name__}; expected JAX/NumPy float array")
    arr_cand_g = np.asarray(raw_cand_g)
    if not negctl.is_floating_dtype(arr_cand_g.dtype) or arr_cand_g.size == 0:
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
  if fp64_tol_validator is not None:
    both_fp64 = False
    for name in grad_classes.compared | grad_classes.padded:
      ref_g = ref_grads[name]
      cand_key = weight_map[name].removesuffix(":T")
      cand_g = np.asarray(cand_grads[cand_key])
      ref_is_fp64 = (HAS_TORCH and ref_params[name].dtype == torch.float64) or (
          HAS_TORCH and ref_g is not None and ref_g.dtype == torch.float64
      )
      if ref_is_fp64 and cand_g.dtype == np.float64:
        both_fp64 = True
        break
    if both_fp64:
      fp64_tol_validator(grad_tol, "grad_tol")

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
      ref_g_np = ref_g.detach().cpu().to(torch.float64).numpy() if HAS_TORCH else np.asarray(ref_g, dtype=np.float64)
      ok, metrics = compare(cand_g, ref_g_np, grad_tol)
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
      ref_g_np = ref_g.detach().cpu().to(torch.float64).numpy() if HAS_TORCH else np.asarray(ref_g, dtype=np.float64)
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

  forward: Metrics | dict[str, Metrics]
  grads: dict[str, Metrics]
  failures: list[str]
  ref_out: np.ndarray | None = None

  @property
  def passed(self) -> bool:
    return not self.failures


def check_forward_and_grads(
    *,
    cand_fn: Callable[[Any, Any], Mapping[str, jax.Array]],
    cand_params: Any,
    cand_inputs: Any,
    ref_outputs: Mapping[str, np.ndarray],
    cotangents: Mapping[str, np.ndarray],
    ref_grads: Mapping[str, np.ndarray],
    grad_map: Mapping[str, str],
    fwd_tol: Tolerance,
    grad_tol: Tolerance,
) -> ParityReport:
  """Checks every candidate output and the gradient of every parameter and input leaf (golden-archive API)."""
  if not ref_outputs:
    raise ValueError("ref_outputs is empty; name at least one compared output")
  _assert_jax_tree(cand_params, "Candidate params")
  _assert_jax_tree(cand_inputs, "Candidate inputs")
  cand_leaves = {
      f"{prefix}/{negctl.leaf_path(p)}"
      for prefix, tree in (("params", cand_params), ("inputs", cand_inputs))
      for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]
  }
  mapped = {spec.removesuffix(":T") for spec in grad_map.values()}
  if set(grad_map) != set(ref_grads) or mapped != cand_leaves or len(mapped) != len(grad_map):
    raise ValueError(
        "grad_map must cover every gradient exactly once: "
        f"unmapped ref={sorted(set(ref_grads) - set(grad_map))}, "
        f"unknown ref={sorted(set(grad_map) - set(ref_grads))}, "
        f"unmapped cand={sorted(cand_leaves - mapped)}, "
        f"unknown cand={sorted(mapped - cand_leaves)}"
    )

  cand_out = jax.jit(cand_fn)(cand_params, cand_inputs)
  _assert_jax_tree(cand_out, "Candidate output")
  if set(cand_out) != set(ref_outputs) or set(cotangents) != set(ref_outputs):
    raise ValueError(
        f"Output names differ: candidate={sorted(cand_out)}, reference={sorted(ref_outputs)}, "
        f"cotangents={sorted(cotangents)}"
    )
  failures = []
  fwd_metrics = {}
  for name in sorted(ref_outputs):
    ok, metrics = compare(np.asarray(cand_out[name]), ref_outputs[name], fwd_tol)
    fwd_metrics[name] = metrics
    if not ok:
      failures.append(f"forward {name}: {metrics}")

  def loss_fn(params: Any, inputs: Any) -> jax.Array:
    outs = cand_fn(params, inputs)
    total: Any = 0.0
    for k in sorted(outs):
      dtype = jnp.promote_types(jnp.promote_types(outs[k].dtype, cotangents[k].dtype), jnp.float32)
      total = total + jnp.sum(outs[k].astype(dtype) * jnp.asarray(cotangents[k], dtype))
    return total

  param_grads, input_grads = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(cand_params, cand_inputs)
  cand_grads = {}
  for prefix, tree in (("params", param_grads), ("inputs", input_grads)):
    for p, g in jax.tree_util.tree_flatten_with_path(tree)[0]:
      cand_grads[f"{prefix}/{negctl.leaf_path(p)}"] = np.asarray(g)

  grad_metrics = {}
  for name in sorted(ref_grads):
    spec = grad_map[name]
    cand_g = cand_grads[spec.removesuffix(":T")]
    if spec.endswith(":T"):
      cand_g = cand_g.T
    ok, metrics = compare(cand_g, ref_grads[name], grad_tol)
    grad_metrics[name] = metrics
    if not ok:
      failures.append(f"grad {name}: {metrics}")
  return ParityReport(fwd_metrics, grad_metrics, failures)


def check_parity(
    *,
    ref_fn: Callable[[dict[str, Any], Any], Any],
    ref_params: dict[str, Any],
    ref_input: Any,
    cand_fn: Callable[[Any, Any], jax.Array],
    cand_params: Any,
    cand_input: Any,
    weight_map: Mapping[str, str],
    grad_classes: GradientClasses,
    fwd_tol: Tolerance,
    grad_tol: Tolerance,
    cotangent_seed: int,
    fp64_tol_validator: Callable[[Tolerance, str], None] | None = None,
) -> ParityReport:
  """Checks the forward output and the gradient of every parameter leaf by class (live PyTorch API)."""
  if not HAS_TORCH:
    raise RuntimeError("check_parity requires PyTorch to be installed")
  _assert_jax_tree(cand_params, "Candidate params")
  _assert_jax_tree(cand_input, "Candidate input")

  cloned_ref_params = {k: v.detach().clone().requires_grad_(k not in grad_classes.frozen) for k, v in ref_params.items()}
  ref_out = ref_fn(cloned_ref_params, ref_input)
  cand_out = cand_fn(cand_params, cand_input)
  _assert_jax_tree(cand_out, "Candidate output")
  if fp64_tol_validator is not None:
    if cand_out.dtype == jnp.float64 and ref_out.dtype == torch.float64:
      fp64_tol_validator(fwd_tol, "fwd_tol")
    cand_leaf_map = {negctl.leaf_path(p): leaf for p, leaf in jax.tree_util.tree_flatten_with_path(cand_params)[0]}
    grad_both_fp64 = False
    for name in grad_classes.compared | grad_classes.padded:
      spec = weight_map.get(name)
      if spec is None:
        continue
      cand_leaf = cand_leaf_map.get(spec.removesuffix(":T"))
      ref_param = cloned_ref_params.get(name)
      if (
          ref_param is not None
          and ref_param.dtype == torch.float64
          and cand_leaf is not None
          and cand_leaf.dtype == jnp.float64
      ):
        grad_both_fp64 = True
        break
    if grad_both_fp64:
      fp64_tol_validator(grad_tol, "grad_tol")

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
  ref_grad_map: dict[str, Any] = {k: None for k in cloned_ref_params}
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
      fp64_tol_validator=fp64_tol_validator,
  )
  failures.extend(grad_failures)
  return ParityReport(fwd_metrics, grad_metrics, failures, ref_out_np)


def negative_control_fires(
    *,
    cand_fn: Callable[[Any, Any], Any],
    cand_params: Any,
    fwd_tol: Tolerance,
    target_path: str,
    rel_scale: float,
    seed: int,
    cand_inputs: Any = None,
    ref_outputs: Mapping[str, np.ndarray] | None = None,
    cand_input: Any = None,
    ref_out: np.ndarray | None = None,
    fp64_tol_validator: Callable[[Tolerance, str], None] | None = None,
) -> tuple[bool, Any]:
  """Perturbs one candidate leaf and verifies the forward comparison fires.

  Supports both the golden-archive dict API (`cand_inputs`, `ref_outputs`) and
  the single-array API (`cand_input`, `ref_out`).
  """
  use_dict = cand_inputs is not None or ref_outputs is not None
  use_single = cand_input is not None or ref_out is not None
  if use_dict and use_single:
    raise ValueError("Pass either (cand_inputs, ref_outputs) or (cand_input, ref_out), not both")
  if not use_dict and not use_single:
    raise ValueError("Pass either (cand_inputs, ref_outputs) or (cand_input, ref_out)")

  _assert_jax_tree(cand_params, "Candidate params")
  if use_dict:
    if ref_outputs is None or cand_inputs is None:
      raise ValueError("Both cand_inputs and ref_outputs must be provided together")
    if not ref_outputs:
      raise ValueError("ref_outputs is empty; name at least one compared output")
    _assert_jax_tree(cand_inputs, "Candidate inputs")
    run = jax.jit(cand_fn)
    primary = run(cand_params, cand_inputs)
    _assert_jax_tree(primary, "Candidate output")
    if set(primary) != set(ref_outputs):
      raise ValueError(f"Output names differ: candidate={sorted(primary)}, reference={sorted(ref_outputs)}")
    primary_failures = []
    for name in sorted(ref_outputs):
      ok, m = compare(np.asarray(primary[name]), ref_outputs[name], fwd_tol)
      if not ok:
        primary_failures.append(f"{name}: {m}")
    if primary_failures:
      raise ValueError(f"Unperturbed candidate already fails, so the control is meaningless: {primary_failures}")
    perturbed = negctl.perturb_one_leaf(cand_params, target_path=target_path, rel_scale=rel_scale, seed=seed)
    outs = run(perturbed, cand_inputs)
    _assert_jax_tree(outs, "Perturbed candidate output")
    if set(outs) != set(ref_outputs):
      raise ValueError(f"Perturbed output names differ: candidate={sorted(outs)}, reference={sorted(ref_outputs)}")
    non_finite = [
        name for name in sorted(ref_outputs) if not np.all(np.isfinite(np.asarray(outs[name], dtype=np.float64)))
    ]
    if non_finite:
      raise ValueError(f"Perturbed outputs {non_finite} are not finite; a crash-like result is not a fired control")
    if all(np.array_equal(np.asarray(outs[name]), np.asarray(primary[name])) for name in ref_outputs):
      raise ValueError(f"Perturbing {target_path!r} left every compared output unchanged")
    metrics = {}
    fired = False
    for name in sorted(ref_outputs):
      ok, metrics[name] = compare(np.asarray(outs[name]), ref_outputs[name], fwd_tol)
      fired = fired or not ok
    return fired, metrics

  if ref_out is None or cand_input is None:
    raise ValueError("Both cand_input and ref_out must be provided together")
  _assert_jax_tree(cand_input, "Candidate input")
  perturbed = negctl.perturb_one_leaf(cand_params, target_path=target_path, rel_scale=rel_scale, seed=seed)
  perturbed_out = cand_fn(perturbed, cand_input)
  _assert_jax_tree(perturbed_out, "Perturbed candidate output")
  if fp64_tol_validator is not None and (
      perturbed_out.dtype == jnp.float64
      or any(leaf.dtype == jnp.float64 for _, leaf in jax.tree_util.tree_flatten_with_path(cand_params)[0])
  ):
    fp64_tol_validator(fwd_tol, "fwd_tol")
  ok, metrics = compare(np.asarray(perturbed_out), ref_out, fwd_tol)
  return (not ok) and metrics.finite, metrics


def reference_noise_floor(
    ref_fn: Callable[[dict[str, Any], Any], Any],
    ref_params: dict[str, Any],
    ref_input: Any,
) -> Metrics:
  """Measures the reference'\''s own float32-vs-float64 difference (tolerance calibration)."""
  if not HAS_TORCH:
    raise RuntimeError("reference_noise_floor requires PyTorch to be installed")

  def run(dtype: Any) -> np.ndarray:
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
