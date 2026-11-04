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

"""Model-agnostic implementations of MaxText m3 conformance Checks 1-4."""

from collections.abc import Callable, Mapping
import copy
import fnmatch
import re
from typing import Any, cast

from flax import nnx
import jax
import jax.numpy as jnp
from tests.unit.m3.conformance import spec
import numpy as np

# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 1e-5.
_FP32_TOL = 1e-5
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 1e-7.
_AUX_FLOOR = 1e-7
_OUTPUT_KIND_NDIM = {
    spec.OutputKind.LOGITS: 3,
    spec.OutputKind.DIFFUSION: 3,
    spec.OutputKind.EMBEDDINGS: 2,
}


def _unwrap_leaf(node: Any) -> Any:
  return node.get_value() if isinstance(node, nnx.Variable) else node


def _key_str(path: Any) -> str:
  return jax.tree_util.keystr(path, simple=True, separator=".")


def _flatten_vars(tree: Any) -> list[tuple[str, Any]]:
  flat, _ = jax.tree_util.tree_flatten_with_path(
      tree,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
  return [(_key_str(p), v) for p, v in flat]


def _is_inexact(val: Any) -> bool:
  if not isinstance(val, (jax.Array, np.ndarray)):
    return False
  return jax.dtypes.issubdtype(val.dtype, jnp.inexact)


def _leaf_meta(val: Any) -> tuple[Any, ...]:
  if isinstance(val, (jax.Array, np.ndarray)):
    return (isinstance(val, jax.Array), val.shape, val.dtype)
  return ("py", type(val))


def _var_meta(path: str, var: Any) -> tuple[Any, ...]:
  return (path, type(var), *_leaf_meta(_unwrap_leaf(var)))


def _validate_forward_output(
    spec_obj: spec.ConformanceSpec,
    out: spec.ForwardOutput,
) -> None:
  """Validates `out.primary` rank and inexact `jax.Array` dtype."""
  name, kind = spec_obj.model_name, spec_obj.output_kind
  if not isinstance(out.aux_losses, Mapping):
    raise AssertionError(f"{name}: out.aux_losses must be Mapping")
  if not isinstance(kind, spec.OutputKind) or kind not in _OUTPUT_KIND_NDIM:
    raise AssertionError(f"{name}: unsupported output_kind {kind!r}")
  arr = out.primary
  if not isinstance(arr, jax.Array):
    got = type(arr).__name__
    raise AssertionError(f"{name}: out.primary must be jax.Array, got {got}")
  if not _is_inexact(arr) or arr.ndim < _OUTPUT_KIND_NDIM[kind]:
    raise AssertionError(f"{name}: invalid {kind} {arr.shape} {arr.dtype}")


def _assert_fp32_parity(actual: jax.Array, ref: jax.Array, tag: str) -> None:
  """Asserts elementwise and floored Frobenius fp32 parity against `ref`."""
  actual_np = np.asarray(actual, dtype=np.float32)
  ref_np = np.asarray(ref, dtype=np.float32)
  atol = _FP32_TOL * max(1.0, float(np.max(np.abs(ref_np))))
  err = f"Elementwise mismatch at {tag}"
  np.testing.assert_allclose(
      actual_np,
      ref_np,
      atol=atol,
      rtol=_FP32_TOL,
      equal_nan=False,
      err_msg=err,
  )
  diff_norm = float(np.linalg.norm(actual_np - ref_np))
  limit = _FP32_TOL * max(1.0, float(np.linalg.norm(ref_np)))
  if not np.isfinite(diff_norm) or diff_norm > limit:
    raise AssertionError(f"Frobenius mismatch at {tag}: {diff_norm} > {limit}")


def run_check_or_skip(
    spec_obj: spec.ConformanceSpec,
    check_id: spec.CheckId,
    check_fn: Callable[[spec.ConformanceSpec, jax.sharding.Mesh], None],
    mesh: jax.sharding.Mesh,
) -> bool:
  """Runs `check_fn` or validates an explicit `NOT_APPLICABLE` skip."""
  spec_obj.validate_spec_invariants()
  if check_id in spec_obj.skips:
    decl = spec_obj.skips[check_id]
    if decl.status != spec.SkipStatus.NOT_APPLICABLE or not decl.reason.strip():
      raise AssertionError(
          f"{spec_obj.model_name}: invalid or empty skip {check_id}",
      )
    return False
  check_fn(spec_obj, mesh)
  return True


def _eval_loss(
    module: nnx.Module,
    spec_obj: spec.ConformanceSpec,
    batch: Mapping[str, jax.Array],
    weights: Mapping[str, float],
) -> tuple[jax.Array, tuple[jax.Array, dict[str, jax.Array]]]:
  """Runs `forward_fn`, validates output, and returns `(loss, aux_pair)`."""
  out = spec_obj.forward_fn(module, batch)
  _validate_forward_output(spec_obj, out)
  loss = spec_obj.loss_fn(out, batch, weights)
  return loss, (out.primary, {**out.aux_losses})


def check_scan_remat_equivalence(
    spec_obj: spec.ConformanceSpec,
    mesh: jax.sharding.Mesh,
) -> None:
  """Check 1: Enforces Invariant (3) (scan/remat path & fp32 parity)."""
  name, variants = spec_obj.model_name, spec_obj.scan_remat_variants
  has_ref = (False, "none") in variants and len(variants) >= 2
  if not has_ref or len(set(variants)) != len(variants):
    raise ValueError(f"{name}: invalid scan_remat_variants {variants!r}")
  weights = {k: 1.0 for k in spec_obj.aux_losses}
  with jax.sharding.set_mesh(mesh):
    model_ref = spec_obj.build_model(mesh, nnx.Rngs(0), False, "none")
    state_ref = nnx.clone(nnx.state(model_ref))
    meta_ref = [_var_meta(p, v) for p, v in _flatten_vars(state_ref)]
    batch = spec_obj.build_batch(jax.random.PRNGKey(42), mesh)
    diff_param = nnx.DiffState(0, nnx.Param)
    vg_fn = nnx.value_and_grad(_eval_loss, argnums=diff_param, has_aux=True)
    ref_out, grads_ref = vg_fn(model_ref, spec_obj, batch, weights)
    loss_ref, (primary_ref, aux_ref) = ref_out
    grads_ref_map = {p: _unwrap_leaf(v) for p, v in _flatten_vars(grads_ref)}
    for scan, remat in variants:
      if (scan, remat) == (False, "none"):
        continue
      tag = f"{name}[scan={scan},remat={remat}]"
      model_var = spec_obj.build_model(mesh, nnx.Rngs(0), scan, remat)
      state_var = nnx.state(model_var)
      meta_var = [_var_meta(p, v) for p, v in _flatten_vars(state_var)]
      if meta_var != meta_ref:
        diff = set(meta_var) ^ set(meta_ref)
        detail = diff if diff else (meta_var, meta_ref)
        raise AssertionError(
            f"{tag}: state paths/shapes/dtypes differ: {detail}",
        )
      nnx.update(model_var, nnx.clone(state_ref))
      var_out, grads_var = vg_fn(model_var, spec_obj, batch, weights)
      loss_var, (primary_var, aux_var) = var_out
      _assert_fp32_parity(primary_var, primary_ref, f"{tag}.primary")
      if set(aux_var.keys()) != set(aux_ref.keys()):
        keys = sorted(aux_var)
        raise AssertionError(f"{tag}: aux_losses keys differ: {keys}")
      for aux_key in sorted(aux_ref):
        _assert_fp32_parity(
            aux_var[aux_key],
            aux_ref[aux_key],
            f"{tag}.aux.{aux_key}",
        )
      _assert_fp32_parity(loss_var, loss_ref, f"{tag}.loss")
      for path, grad_var in _flatten_vars(grads_var):
        _assert_fp32_parity(
            _unwrap_leaf(grad_var),
            grads_ref_map[path],
            f"{tag}.grad.{path}",
        )


def check_param_and_grad_health(
    spec_obj: spec.ConformanceSpec,
    mesh: jax.sharding.Mesh,
) -> None:
  """Check 2: Enforces Invariant (1) (`nnx.Param` is the only trainable var)."""
  name = spec_obj.model_name
  with jax.sharding.set_mesh(mesh):
    try:
      spec_obj.build_model(mesh, cast(Any, None), True, "none")
      raise AssertionError(f"{name}: accepted rngs=None")
    except (TypeError, ValueError, AttributeError):
      pass
    model = spec_obj.build_model(mesh, nnx.Rngs(0), True, "none")
    param_vars = _flatten_vars(nnx.state(model, nnx.Param))
    if not param_vars:
      raise AssertionError(f"{name}: model has no nnx.Param")
    for path, param_leaf in param_vars:
      raw = _unwrap_leaf(param_leaf)
      if not isinstance(raw, jax.Array) or not _is_inexact(raw):
        is_jax = isinstance(raw, jax.Array)
        dt = str(raw.dtype) if is_jax else type(raw).__name__
        raise AssertionError(f"{name}: non-inexact nnx.Param {path} dtype={dt}")
    for pat in spec_obj.frozen_param_patterns:
      if not any(re.search(pat, p) for p, _ in param_vars):
        raise AssertionError(f"{name}: unmatched frozen pattern {pat!r}")
    all_vars = _flatten_vars(nnx.state(model))
    batch = spec_obj.build_batch(jax.random.PRNGKey(0), mesh)
    weights = {k: 1.0 for k in spec_obj.aux_losses}
    diff_state = nnx.DiffState(0, lambda _, n: _is_inexact(_unwrap_leaf(n)))
    vg_fn = nnx.value_and_grad(_eval_loss, argnums=diff_state, has_aux=True)
    (loss_val, (primary_val, _)), grads = vg_fn(model, spec_obj, batch, weights)
    primary_ok = bool(jnp.all(jnp.isfinite(primary_val)))
    if not bool(jnp.isfinite(loss_val)) or not primary_ok:
      raise AssertionError(f"{name}: non-finite loss ({loss_val}) or output")
    grad_map = {p: _unwrap_leaf(v) for p, v in _flatten_vars(grads)}
    issues: list[str] = []
    for path, leaf in all_vars:
      grad = grad_map[path] if path in grad_map else None
      nz = grad is not None and bool(jnp.any(grad != 0))
      bad_g = grad is not None and not bool(jnp.all(jnp.isfinite(grad)))
      if isinstance(leaf, nnx.Param):
        if not bool(jnp.all(jnp.isfinite(_unwrap_leaf(leaf)))):
          issues.append(f"nan_param:{path}")
        frz = any(re.search(p, path) for p in spec_obj.frozen_param_patterns)
        if bad_g:
          issues.append(f"non_finite:{path}")
        elif frz == nz:
          issues.append(f"{'unfrozen' if frz else 'dead'}:{path}")
      elif bad_g or nz:
        issues.append(f"leaky_non_param:{path}")
    if issues:
      raise AssertionError(f"{name}: grad health issues {issues}")


def check_non_param_state_stability(
    spec_obj: spec.ConformanceSpec,
    mesh: jax.sharding.Mesh,
) -> None:
  """Check 3: Enforces Invariant (2) (no `sow`; stable non-`Param` state)."""
  name, mut_pats = spec_obj.model_name, spec_obj.mutable_non_param_patterns

  def _inspect_rest(
      mod: nnx.Module,
      step_tag: str,
  ) -> tuple[Any, dict[str, tuple[Any, Any]]]:
    snaps: dict[str, tuple[Any, Any]] = {}
    for v_path, v_node in _flatten_vars(nnx.state(mod)):
      if isinstance(v_node, nnx.Intermediate):
        raise AssertionError(
            f"{name}: {step_tag} sowed nnx.Intermediate at {v_path}",
        )
      if not isinstance(v_node, nnx.Param):
        v_arr = _unwrap_leaf(v_node)
        if _is_inexact(v_arr) and not bool(jnp.all(jnp.isfinite(v_arr))):
          raise AssertionError(
              f"{name}: {step_tag} non-finite non-Param state at {v_path}",
          )
        is_jax = isinstance(v_arr, jax.Array)
        snap = v_arr if is_jax else copy.deepcopy(v_arr)
        snaps[v_path] = (v_node, snap)
    _, _, rest_state = nnx.split(mod, nnx.Param, ...)
    return jax.tree.structure(rest_state), snaps

  weights = {k: 1.0 for k in spec_obj.aux_losses}
  diff_param = nnx.DiffState(0, nnx.Param)
  vg_fn = nnx.value_and_grad(_eval_loss, argnums=diff_param, has_aux=True)
  with jax.sharding.set_mesh(mesh):
    model = spec_obj.build_model(mesh, nnx.Rngs(0), True, "none")
    batch = spec_obj.build_batch(jax.random.PRNGKey(0), mesh)
    treedef_ref, ref_map = _inspect_rest(model, "init")
    for pat in mut_pats:
      if not any(fnmatch.fnmatchcase(p_ref, pat) for p_ref in ref_map):
        raise AssertionError(f"{name}: unmatched mutable pattern {pat!r}")
    for label, run_grad in (("step 1", False), ("step 2", True)):
      if run_grad:
        _ = vg_fn(model, spec_obj, batch, weights)
      else:
        _validate_forward_output(spec_obj, spec_obj.forward_fn(model, batch))
      treedef_cur, cur_map = _inspect_rest(model, label)
      if list(cur_map) != list(ref_map):
        add = sorted(set(cur_map) - set(ref_map))
        rem = sorted(set(ref_map) - set(cur_map))
        raise AssertionError(f"{name} {label}: +{add} -{rem} {list(cur_map)}")
      pairs = zip(ref_map.items(), cur_map.items(), strict=True)
      for (path, (v_ref, a_ref)), (_, (v_cur, a_cur)) in pairs:
        same_type = type(v_ref) is type(v_cur)
        same_meta = same_type and _leaf_meta(a_ref) == _leaf_meta(a_cur)
        is_mut = any(fnmatch.fnmatchcase(path, p) for p in mut_pats)
        exempt = isinstance(v_ref, nnx.RngCount) or is_mut
        is_arr = isinstance(a_ref, (jax.Array, np.ndarray))
        eq_arr = bool(jnp.array_equal(a_ref, a_cur)) if is_arr else False
        same_val = eq_arr if is_arr else a_ref == a_cur
        if not same_meta or (not exempt and not same_val):
          raise AssertionError(
              f"{name}: {label} non-Param state mutated at {path}",
          )
      if treedef_cur != treedef_ref:
        raise AssertionError(f"{name}: {label} non-Param state mutated")


def check_aux_loss_grad_coupling(
    spec_obj: spec.ConformanceSpec,
    mesh: jax.sharding.Mesh,
) -> None:
  """Check 4: Enforces Invariant (4) (aux losses fold into primary loss)."""
  name = spec_obj.model_name
  if not spec_obj.aux_losses:
    raise AssertionError(f"{name}: spec.aux_losses is empty")
  with jax.sharding.set_mesh(mesh):
    model = spec_obj.build_model(mesh, nnx.Rngs(0), True, "none")
    init_full_state = nnx.clone(nnx.state(model))
    batch = spec_obj.build_batch(jax.random.PRNGKey(7), mesh)
    out = spec_obj.forward_fn(model, batch)
    _validate_forward_output(spec_obj, out)
    extra = sorted(set(out.aux_losses) - set(spec_obj.aux_losses))
    if extra:
      raise AssertionError(f"{name}: Undeclared aux_losses keys {extra}")
    for aux_key, pats in spec_obj.aux_losses.items():
      if not pats or aux_key not in out.aux_losses:
        msg = "empty pattern tuple for" if not pats else "missing aux_loss"
        raise AssertionError(f"{name}: {msg} {aux_key!r}")
      aux_val = out.aux_losses[aux_key]
      is_scalar = isinstance(aux_val, jax.Array) and aux_val.ndim == 0
      ok_aux = is_scalar and _is_inexact(aux_val)
      ok_val = ok_aux and bool(jnp.isfinite(aux_val))
      if not ok_val or float(jnp.abs(aux_val)) <= 0.0:
        raise AssertionError(
            f"{name}: invalid/zero aux_loss {aux_key!r}: {aux_val!r}",
        )
    diff_param = nnx.DiffState(0, nnx.Param)
    grad_fn = nnx.grad(_eval_loss, argnums=diff_param, has_aux=True)

    def _grads_for(w_map: dict[str, float]) -> dict[str, jax.Array]:
      nnx.update(model, nnx.clone(init_full_state))
      grad_tree, _ = grad_fn(model, spec_obj, batch, w_map)
      return {k: _unwrap_leaf(v) for k, v in _flatten_vars(grad_tree)}

    grads_zero = _grads_for({k: 0.0 for k in spec_obj.aux_losses})
    for aux_key, pats in spec_obj.aux_losses.items():
      matched: list[str] = []
      for pat in pats:
        hits = [p for p in grads_zero if re.search(pat, p)]
        if not hits:
          raise AssertionError(
              f"{name}: {aux_key!r} pattern {pat!r} matched no nnx.Param",
          )
        matched.extend(hits)
      w_one = {k: 1.0 if k == aux_key else 0.0 for k in spec_obj.aux_losses}
      grads_one = _grads_for(w_one)
      for path in dict.fromkeys(matched):
        delta = grads_one[path] - grads_zero[path]
        finite = bool(jnp.all(jnp.isfinite(delta)))
        if not finite or float(jnp.max(jnp.abs(delta))) <= _AUX_FLOOR:
          raise AssertionError(
              f"{name}: {aux_key} bad/zero grad delta on {path}",
          )
