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

"""Forward and all-leaf gradient parity of a JAX candidate against stored reference arrays.

Reference outputs and gradients come from a PyTorch oracle, either live or from a committed golden archive, so this
module never imports PyTorch. Tolerances are explicit arguments taken from the approved design tolerance table; nothing
here has a default tolerance.

`hashed_normal` produces the deterministic parameters, inputs, and cotangents that both the golden generator and the
JAX tests rebuild from a string key, so golden archives store only reference outputs and gradients.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tests.unit.m3.testing import negctl

_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
# Irwin-Hall sum of four uniforms has variance 4 / 12; this scale gives unit variance.
_IRWIN_HALL_TERMS = 4
_IRWIN_HALL_SCALE = math.sqrt(3.0)


@dataclasses.dataclass(frozen=True)
class Tolerance:
  """One row of the design tolerance table.

  Passing requires, elementwise, |c - r| <= atol * max(1, |r|_inf) + rtol * |r|, and
  ||c - r||_2 <= frob_rel * max(1, ||r||_2).
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
  """Compares candidate with reference in float64; raises on a shape mismatch or zero-size arrays."""
  if cand.shape != ref.shape:
    raise ValueError(f"Shape mismatch: candidate {cand.shape} vs reference {ref.shape}")
  if ref.size == 0:
    raise ValueError(f"Cannot compare zero-size arrays of shape {ref.shape}")
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


def _splitmix64(counter: np.ndarray) -> np.ndarray:
  """Vectorized SplitMix64 finalizer on uint64 counters (wrapping arithmetic)."""
  z = counter + _SPLITMIX_GAMMA
  z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_MUL1
  z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_MUL2
  return z ^ (z >> np.uint64(31))


def hashed_normal(
    key: str, shape: Sequence[int], *, mean: float, std: float, dtype: np.typing.DTypeLike = np.float32
) -> np.ndarray:
  """Returns a platform-independent approximately standard-normal tensor derived from `key`.

  Uses only integer hashing, additions, and multiplications (no libm transcendental functions), so the generator
  and every consumer reproduce identical bits on any platform and NumPy version.

  Args:
    key: Unique string naming the tensor, e.g. "attn_csa/param/q_a_proj.weight".
    shape: Output shape.
    mean: Mean of the returned tensor.
    std: Standard deviation of the returned tensor.
    dtype: Output dtype.

  Returns:
    An array of `shape` and `dtype`.
  """
  size = math.prod(shape)
  seed = np.uint64(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little"))
  counters = np.arange(size * _IRWIN_HALL_TERMS, dtype=np.uint64) + seed
  bits = _splitmix64(counters)
  uniforms = (bits >> np.uint64(11)).astype(np.float64) * (2.0**-53)
  sums = uniforms.reshape(size, _IRWIN_HALL_TERMS).sum(axis=1)
  normal = (sums - _IRWIN_HALL_TERMS / 2.0) * _IRWIN_HALL_SCALE
  return (normal * std + mean).astype(dtype).reshape(tuple(shape))


def _assert_jax_tree(tree: Any, label: str) -> None:
  """Blocks self-comparison: every array leaf of the candidate must be a jax.Array."""
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    if not isinstance(leaf, jax.Array):
      raise TypeError(f"{label} leaf {negctl.leaf_path(path)!r} is {type(leaf).__name__}, not jax.Array")


@dataclasses.dataclass
class ParityReport:
  forward: dict[str, Metrics]
  grads: dict[str, Metrics]
  failures: list[str]

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
  """Checks every candidate output and the gradient of every parameter and input leaf.

  The scalar loss is sum_k <output_k, cotangent_k>, the same projection the reference used, so no gradient term
  cancels by symmetry.

  Args:
    cand_fn: Maps (params, inputs) to a dict of named outputs.
    cand_params: Candidate parameter pytree (jax.Array leaves).
    cand_inputs: Candidate differentiable input pytree (jax.Array leaves).
    ref_outputs: Reference output per name; must cover every candidate output exactly.
    cotangents: Cotangent per output name used by the reference loss.
    ref_grads: Reference gradient per reference name.
    grad_map: Reference gradient name -> candidate leaf spec, covering every reference gradient and every candidate
      parameter and input leaf exactly once. Specs are `params/<leaf_path>` or `inputs/<leaf_path>`; a `:T` suffix
      compares the transposed candidate gradient.
    fwd_tol: Forward tolerance row.
    grad_tol: Gradient tolerance row.

  Returns:
    A ParityReport whose `failures` lists every failing comparison.
  """
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

  def loss_fn(params, inputs):
    outs = cand_fn(params, inputs)
    total = 0.0
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


def negative_control_fires(
    *,
    cand_fn: Callable[[Any, Any], Mapping[str, jax.Array]],
    cand_params: Any,
    cand_inputs: Any,
    ref_outputs: Mapping[str, np.ndarray],
    fwd_tol: Tolerance,
    target_path: str,
    rel_scale: float,
    seed: int,
) -> tuple[bool, dict[str, Metrics]]:
  """Perturbs one candidate parameter leaf and returns (some output comparison failed, perturbed metrics).

  Uses the same predicate and tolerance as the positive forward check. A True result means the parity test can
  detect a wrong weight of this size.

  Raises:
    ValueError: If the control proves nothing: the unperturbed candidate already fails, the perturbed outputs are
      not finite, or they equal the unperturbed outputs (the leaf is not on the compared path).
  """
  if not ref_outputs:
    raise ValueError("ref_outputs is empty; name at least one compared output")
  _assert_jax_tree(cand_params, "Candidate params")
  _assert_jax_tree(cand_inputs, "Candidate inputs")
  run = jax.jit(cand_fn)
  primary = run(cand_params, cand_inputs)
  _assert_jax_tree(primary, "Candidate output")
  if set(primary) != set(ref_outputs):
    raise ValueError(f"Output names differ: candidate={sorted(primary)}, reference={sorted(ref_outputs)}")
  primary_failures = []
  for name in sorted(ref_outputs):
    ok, metrics = compare(np.asarray(primary[name]), ref_outputs[name], fwd_tol)
    if not ok:
      primary_failures.append(f"{name}: {metrics}")
  if primary_failures:
    raise ValueError(f"Unperturbed candidate already fails, so the control is meaningless: {primary_failures}")
  perturbed = negctl.perturb_one_leaf(cand_params, target_path=target_path, rel_scale=rel_scale, seed=seed)
  outs = run(perturbed, cand_inputs)
  _assert_jax_tree(outs, "Perturbed candidate output")
  if set(outs) != set(ref_outputs):
    raise ValueError(f"Perturbed output names differ: candidate={sorted(outs)}, reference={sorted(ref_outputs)}")
  non_finite = [name for name in sorted(ref_outputs) if not np.all(np.isfinite(np.asarray(outs[name], dtype=np.float64)))]
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
