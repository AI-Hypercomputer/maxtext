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

"""Model-agnostic implementations of MaxText m3 conformance Checks 1-8."""

from collections.abc import Callable, Mapping, Sequence
import copy
import fnmatch
import math
import re
from typing import Any, cast

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.core import sharding as m3_sharding
from tests.unit.m3.conformance import spec
import numpy as np

# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 1e-5.
_FP32_TOL = 1e-5
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting
# to 1e-7.
_AUX_FLOOR = 1e-7
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 4.
_DRIVER_MESH_DEVICES = 4
_DRIVER_MESH_ICI = {"data": 1, "fsdp": 2, "tensor": 2, "expert": 1}
_VALID_MESH_AXES = frozenset(m3_sharding.MESH_AXIS_NAMES)
_PARAM_TAINT_UNARY_PRIMITIVES = frozenset(
    {
        "convert_element_type",
        "bitcast_convert_type",
        "reshape",
        "squeeze",
        "rev",
        "transpose",
        "broadcast_in_dim",
        "slice",
        "dynamic_slice",
        "copy",
        "Copy",
        "device_put",
        "sharding_constraint",
        "reshard",
    }
)
_CALL_HOP_PRIMITIVES = frozenset(
    {
        "jit",
        "pjit",
        "remat2",
        "remat",
        "checkpoint",
        "shard_map",
        "custom_jvp_call",
        "custom_vjp_call",
        "eval_jaxpr",
        "closed_call",
        "call",
        "composite",
    }
)
_REDUCTION_JAXPR_PRIMITIVES = frozenset({"reduce", "reduce_window"})
_ALL_GATHER_OPCODE_RE = re.compile(r"\ball-gather(?:-start|-done)?\(")
_HLO_TENSOR_SHAPE_RE = re.compile(r"\b[a-z][a-z0-9]*\[([0-9,\s]*)\]")
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


def _should_skip(spec_obj: spec.ConformanceSpec, check_id: spec.CheckId) -> bool:
  """Validates spec invariants and returns True when `check_id` is declared `NOT_APPLICABLE`."""
  spec_obj.validate_spec_invariants()
  if check_id not in spec_obj.skips:
    return False
  decl = spec_obj.skips[check_id]
  if decl.status != spec.SkipStatus.NOT_APPLICABLE or not decl.reason.strip():
    raise AssertionError(f"{spec_obj.model_name}: invalid or empty skip {check_id}")
  return True


def run_check_or_skip(
    spec_obj: spec.ConformanceSpec,
    check_id: spec.CheckId,
    check_fn: Callable[[spec.ConformanceSpec, jax.sharding.Mesh], None],
    mesh: jax.sharding.Mesh,
) -> bool:
  """Runs `check_fn` or validates an explicit `NOT_APPLICABLE` skip."""
  if _should_skip(spec_obj, check_id):
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
  with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
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
  with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
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
  with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
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
  with jax.sharding.set_mesh(mesh), nnx.use_eager_sharding(True):
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


def create_conformance_driver_mesh() -> jax.sharding.Mesh:
  """Builds the canonical 4-device m3 `(data=1, fsdp=2, tensor=2, expert=1)` driver mesh."""
  devices = jax.devices()
  if len(devices) < _DRIVER_MESH_DEVICES:
    raise AssertionError(f"Checks require >= {_DRIVER_MESH_DEVICES} JAX devices, got {len(devices)}.")
  return m3_sharding.create_mesh(ici=_DRIVER_MESH_ICI, devices=devices[:_DRIVER_MESH_DEVICES])


def _create_single_device_mesh() -> jax.sharding.Mesh:
  """Builds a 1-device canonical m3 mesh for single-device Checks 1-4."""
  devices = jax.devices()
  if not devices:
    raise AssertionError("Conformance checks require >= 1 JAX device, got 0.")
  return m3_sharding.create_mesh(devices=devices[:1])


def _extract_spec_axes(spec_like: Any, context: str) -> list[str]:
  """Extracts physical mesh axis strings from a PartitionSpec, NamedSharding, tuple, or str."""
  if spec_like is None:
    return []
  if isinstance(spec_like, str):
    return [spec_like]
  if isinstance(spec_like, jax.sharding.NamedSharding):
    return _extract_spec_axes(spec_like.spec, context)
  if isinstance(spec_like, (jax.sharding.PartitionSpec, tuple)):
    axes: list[str] = []
    for entry in spec_like:
      if entry is not None and not isinstance(entry, (str, tuple)):
        raise AssertionError(f"{context}: invalid PartitionSpec entry {entry!r}")
      axes.extend(_extract_spec_axes(entry, context))
    return axes
  raise AssertionError(f"{context}: unsupported sharding spec {spec_like!r}")


def _assert_valid_axes(axes: Sequence[str], context: str) -> None:
  """Asserts every physical axis name belongs to `m3_sharding.MESH_AXIS_NAMES`."""
  unknown = [axis for axis in axes if axis not in _VALID_MESH_AXES]
  if unknown:
    raise AssertionError(f"{context}: non-m3 mesh axes {unknown}; got!= {m3_sharding.MESH_AXIS_NAMES}")


def _validate_stacked_patterns(spec_obj: spec.ConformanceSpec, param_vars: Sequence[tuple[str, Any]]) -> None:
  """Validates `stacked_param_path_patterns` and `num_scanned_blocks` (`m3/README.md` Invariant (3))."""
  if not spec_obj.stacked_param_path_patterns:
    return
  n_scan = spec_obj.num_scanned_blocks
  if n_scan <= 1:
    raise AssertionError(f"{spec_obj.model_name}: requires num_scanned_blocks > 1, got {n_scan}")
  for pat in spec_obj.stacked_param_path_patterns:
    if not any(bool(re.search(pat, path_str)) for path_str, _ in param_vars):
      raise AssertionError(f"{spec_obj.model_name}: unmatched stacked pattern {pat!r}")


def _resolve_unstacked_shape(
    spec_obj: spec.ConformanceSpec, path_str: str, full_shape: tuple[int, ...]
) -> tuple[bool, tuple[int, ...]]:
  """Returns `(is_stacked, unstacked_shape)` after verifying any leading scan dimension (`m3/README.md` (3))."""
  is_stacked = any(bool(re.search(pat, path_str)) for pat in spec_obj.stacked_param_path_patterns)
  if not is_stacked:
    return False, full_shape
  n_scan = spec_obj.num_scanned_blocks
  if not full_shape or full_shape[0] != n_scan:
    raise AssertionError(f"{spec_obj.model_name}: {path_str} {full_shape} leading axis != num_scanned_blocks={n_scan}")
  return True, full_shape[1:]


def check_driver_mesh_and_axis_names(spec_obj: spec.ConformanceSpec) -> None:
  """Check 5: Enforces `m3/README.md` Invariants (5) & (6) (driver `Mesh` identity & `MESH_AXIS_NAMES`)."""
  name = spec_obj.model_name
  driver_mesh = create_conformance_driver_mesh()
  with jax.sharding.set_mesh(driver_mesh), nnx.use_eager_sharding(True):
    model = spec_obj.build_model(driver_mesh, nnx.Rngs(0), True, "none")
  if not hasattr(model, "mesh") or model.mesh is not driver_mesh:
    raise AssertionError(f"{name}: model.mesh must be identical (`is`) to driver_mesh")
  for graph_path, graph_node in nnx.iter_graph(model, graph=True):
    node_label = ".".join(map(str, graph_path)) if graph_path else "<root>"
    if isinstance(graph_node, jax.sharding.Mesh) and graph_node is not driver_mesh:
      raise AssertionError(f"{name}: {node_label} holds private Mesh != driver_mesh")
    if hasattr(graph_node, "mesh") and graph_node.mesh is not None and graph_node.mesh is not driver_mesh:
      raise AssertionError(f"{name}: {node_label}.mesh is not driver_mesh")
  array_count = 0
  for path_str, leaf_node in _flatten_vars(nnx.state(model)):
    if isinstance(leaf_node, nnx.Variable):
      meta = leaf_node.get_metadata()
      for meta_key in ("out_sharding", "sharding"):
        if meta_key in meta and meta[meta_key] is not None:
          meta_val = meta[meta_key]
          ctx = f"{name}:{path_str}.{meta_key}"
          if isinstance(meta_val, jax.sharding.NamedSharding) and meta_val.mesh is not driver_mesh:
            raise AssertionError(f"{ctx}: NamedSharding.mesh is not driver_mesh")
          _assert_valid_axes(_extract_spec_axes(meta_val, ctx), ctx)
      val = leaf_node.get_value()
    else:
      val = leaf_node
    if not isinstance(val, jax.Array):
      continue
    array_count += 1
    if not isinstance(val.sharding, jax.sharding.NamedSharding):
      raise AssertionError(f"{name}: {path_str} sharding must be NamedSharding, got {type(val.sharding).__name__}")
    if val.sharding.mesh is not driver_mesh:
      raise AssertionError(f"{name}: {path_str} sharding.mesh is not driver_mesh")
    _assert_valid_axes(_extract_spec_axes(val.sharding.spec, f"{name}:{path_str}"), f"{name}:{path_str}")
  if array_count < 1:
    raise AssertionError(f"{name}: expected >= 1 jax.Array leaf in nnx.state(model)")


def _bind_outvar_taints(
    outvars: Sequence[Any], out_taints: Sequence[str | None], local_taint: dict[int, str], prim: str
) -> None:
  """Verifies output arity and records non-`None` parameter taint on `outvars` (`m3/README.md` (7))."""
  if len(outvars) != len(out_taints):
    raise AssertionError(f"{prim}: outvars arity mismatch ({len(outvars)} != {len(out_taints)})")
  for out_v, src in zip(outvars, out_taints, strict=True):
    if src is not None and type(out_v).__name__ != "DropVar" and id(out_v) not in local_taint:
      local_taint[id(out_v)] = src


def _walk_carry_fixed_point(
    sub_j: Any,
    prefix_taints: Sequence[str | None],
    init_carry_taints: Sequence[str | None],
    suffix_taints: Sequence[str | None],
    violations: list[tuple[str, str]],
) -> tuple[list[str | None], list[str | None]]:
  """Iterates `_walk_jaxpr_param_taint` until loop-carry taint stabilizes (`m3/README.md` (7))."""
  carry = list(init_carry_taints)
  n_carry = len(carry)
  while True:
    iter_violations: list[tuple[str, str]] = []
    sub_outs = _walk_jaxpr_param_taint(sub_j, [*prefix_taints, *carry, *suffix_taints], iter_violations)
    if len(sub_outs) < n_carry:
      raise AssertionError(f"carry jaxpr outvars ({len(sub_outs)}) < num_carry ({n_carry})")
    promoted = False
    for idx in range(n_carry):
      src = sub_outs[idx]
      if carry[idx] is None and src is not None:
        carry[idx] = src
        promoted = True
    if not promoted:
      violations.extend(iter_violations)
      return carry, sub_outs[n_carry:]


def _walk_jaxpr_param_taint(
    jaxpr_like: Any, in_taints: Sequence[str | None], violations: list[tuple[str, str]]
) -> list[str | None]:
  """Propagates parameter taint through `jaxpr_like` and records sharding constraints (`m3/README.md` (7))."""
  jaxpr_obj = jaxpr_like.jaxpr if hasattr(jaxpr_like, "jaxpr") else jaxpr_like
  n_sub_in, n_in = len(jaxpr_obj.invars), len(in_taints)
  if n_sub_in != n_in:
    raise AssertionError(f"jaxpr invars arity mismatch ({n_sub_in} != {n_in})")
  local_taint: dict[int, str] = {}
  for sub_in, src in zip(jaxpr_obj.invars, in_taints, strict=True):
    if src is not None and type(sub_in).__name__ != "DropVar":
      local_taint[id(sub_in)] = src

  for eqn in jaxpr_obj.eqns:
    prim = eqn.primitive.name
    eqn_in = [local_taint[id(v)] if id(v) in local_taint else None for v in eqn.invars]
    in0 = eqn_in[0] if eqn_in else None
    if prim in ("sharding_constraint", "reshard", "device_put") and in0 is not None:
      violations.append((in0, prim))

    tainted_src: str | None = None
    if prim in _PARAM_TAINT_UNARY_PRIMITIVES and in0 is not None:
      tainted_src = in0
    elif prim == "concatenate" and in0 is not None and all(t is not None for t in eqn_in):
      tainted_src = in0
    if tainted_src is not None:
      _bind_outvar_taints(eqn.outvars, [tainted_src] * len(eqn.outvars), local_taint, prim)
      continue

    if prim == "cond":
      if "branches" not in eqn.params or not eqn.params["branches"] or not eqn_in:
        raise AssertionError(f"cond: invalid params/arity invars={len(eqn_in)}")
      for branch in eqn.params["branches"]:
        b_outs = _walk_jaxpr_param_taint(branch, eqn_in[1:], violations)
        _bind_outvar_taints(eqn.outvars, b_outs, local_taint, prim)
    elif prim == "while":
      for req in ("cond_jaxpr", "cond_nconsts", "body_jaxpr", "body_nconsts"):
        if req not in eqn.params:
          raise AssertionError(f"while: missing {req!r} in eqn.params")
      c_nc, b_nc = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
      if c_nc < 0 or b_nc < 0 or c_nc + b_nc > len(eqn_in):
        raise AssertionError(f"while: bad consts ({c_nc}, {b_nc}) vs {len(eqn_in)}")
      cond_j = eqn.params["cond_jaxpr"].jaxpr
      body_j = eqn.params["body_jaxpr"].jaxpr
      init_carry = eqn_in[c_nc + b_nc :]
      if len(body_j.outvars) != len(init_carry):
        raise AssertionError(f"while: body outvars ({len(body_j.outvars)}) != {len(init_carry)}")
      carry_out, _ = _walk_carry_fixed_point(body_j, eqn_in[c_nc : c_nc + b_nc], init_carry, (), violations)
      _walk_jaxpr_param_taint(cond_j, [*eqn_in[:c_nc], *carry_out], violations)
      _bind_outvar_taints(eqn.outvars, carry_out, local_taint, prim)
    elif prim == "scan":
      if "jaxpr" not in eqn.params or "ft_in" not in eqn.params:
        raise AssertionError(f"scan: missing jaxpr/ft_in in {sorted(eqn.params)}")
      consts_ft, carry_ft, xs_ft = eqn.params["ft_in"].unpack()
      n_c, n_k, n_x = len(consts_ft), len(carry_ft), len(xs_ft)
      if len(eqn_in) != n_c + n_k + n_x:
        raise AssertionError(f"scan: invars ({len(eqn_in)}) != ft_in ({n_c} + {n_k} + {n_x})")
      carry_out, ys_out = _walk_carry_fixed_point(
          eqn.params["jaxpr"], eqn_in[:n_c], eqn_in[n_c : n_c + n_k], eqn_in[n_c + n_k :], violations
      )
      _bind_outvar_taints(eqn.outvars, [*carry_out, *ys_out], local_taint, prim)
    elif prim in _CALL_HOP_PRIMITIVES or (
        prim not in _REDUCTION_JAXPR_PRIMITIVES and ("jaxpr" in eqn.params or "call_jaxpr" in eqn.params)
    ):
      if "jaxpr" not in eqn.params and "call_jaxpr" not in eqn.params:
        raise AssertionError(f"{prim}: missing jaxpr/call_jaxpr in eqn.params")
      sub_ref = eqn.params["jaxpr"] if "jaxpr" in eqn.params else eqn.params["call_jaxpr"]
      sub_outs = _walk_jaxpr_param_taint(sub_ref, eqn_in, violations)
      _bind_outvar_taints(eqn.outvars, sub_outs, local_taint, prim)

  return [local_taint[id(v)] if id(v) in local_taint else None for v in jaxpr_obj.outvars]


def check_no_post_init_param_constraints(spec_obj: spec.ConformanceSpec) -> None:
  """Check 6: Enforces `m3/README.md` Invariant (7) (no parameter sharding constraints in `__call__`)."""
  name = spec_obj.model_name
  driver_mesh = create_conformance_driver_mesh()
  with jax.sharding.set_mesh(driver_mesh), nnx.use_eager_sharding(True):
    model = spec_obj.build_model(driver_mesh, nnx.Rngs(0), True, "none")
    batch = spec_obj.build_batch(jax.random.PRNGKey(0), driver_mesh)
    graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
    flat_params, _ = jax.tree_util.tree_flatten_with_path(params_state)
    if not flat_params:
      raise AssertionError(f"{name}: model has no nnx.Param leaves")

    def _pure_forward(
        p_state: Any, r_state: Any, b_inputs: Mapping[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
      merged = nnx.merge(graphdef, p_state, r_state)
      fwd_out = spec_obj.forward_fn(merged, b_inputs)
      _validate_forward_output(spec_obj, fwd_out)
      return fwd_out.primary, {**fwd_out.aux_losses}

    closed_jaxpr = jax.make_jaxpr(_pure_forward)(params_state, rest_state, batch)
    top_jaxpr = closed_jaxpr.jaxpr
    n_invars, n_params = len(top_jaxpr.invars), len(flat_params)
    if n_invars < n_params:
      raise AssertionError(f"{name}: jaxpr invars ({n_invars}) < param leaves ({n_params})")
    root_in_taints: list[str | None] = [None] * n_invars
    for idx, (key_path, leaf_val) in enumerate(flat_params):
      if not isinstance(leaf_val, jax.Array):
        raise AssertionError(f"{name}: non-jax.Array nnx.Param leaf at {key_path}")
      raw_name = jax.tree_util.keystr(key_path, simple=True, separator=".")
      root_in_taints[idx] = raw_name.removesuffix(".value").removesuffix(".raw_value")
    violations: list[tuple[str, str]] = []
    _walk_jaxpr_param_taint(top_jaxpr, root_in_taints, violations)
    if violations:
      deduped = list(dict.fromkeys(violations))
      raise AssertionError(f"{name}: post-init parameter sharding constraints in forward: {deduped}")


def check_weight_sharding_coverage(spec_obj: spec.ConformanceSpec) -> None:
  """Check 7: Enforces `m3/README.md` Invariant (6) & comparison-doc 2-D+ weight sharding coverage."""
  name = spec_obj.model_name
  driver_mesh = create_conformance_driver_mesh()
  with jax.sharding.set_mesh(driver_mesh), nnx.use_eager_sharding(True):
    model = spec_obj.build_model(driver_mesh, nnx.Rngs(0), True, "none")
  param_vars = _flatten_vars(nnx.state(model, nnx.Param))
  if not param_vars:
    raise AssertionError(f"{name}: model has no nnx.Param")
  _validate_stacked_patterns(spec_obj, param_vars)
  eligible_count = 0
  seen_axes: set[str] = set()
  min_elems = spec_obj.min_sharded_param_elements
  for path_str, param_node in param_vars:
    arr = _unwrap_leaf(param_node)
    if not isinstance(arr, jax.Array):
      raise AssertionError(f"{name}: non-jax.Array nnx.Param at {path_str}")
    _, unstacked_shape = _resolve_unstacked_shape(spec_obj, path_str, arr.shape)
    if len(unstacked_shape) < 2 or math.prod(unstacked_shape) < min_elems:
      continue
    eligible_count += 1
    if not isinstance(arr.sharding, jax.sharding.NamedSharding) or arr.sharding.mesh is not driver_mesh:
      raise AssertionError(f"{name}: {path_str} must have NamedSharding on driver_mesh")
    if arr.sharding.spec == jax.sharding.PartitionSpec():
      raise AssertionError(f"{name}: {path_str} {arr.shape} has unsharded PartitionSpec()")
    axes = _extract_spec_axes(arr.sharding.spec, f"{name}:{path_str}")
    _assert_valid_axes(axes, f"{name}:{path_str}")
    if not axes or not any(driver_mesh.shape[axis] > 1 for axis in axes):
      raise AssertionError(f"{name}: {path_str} uses size-1 mesh axes {arr.sharding.spec}")
    if not arr.addressable_shards or math.prod(arr.addressable_shards[0].data.shape) >= math.prod(arr.shape):
      raise AssertionError(f"{name}: {path_str} shard not smaller than {arr.shape}")
    seen_axes.update(axes)
  if eligible_count < 1:
    raise AssertionError(f"{name}: expected >= 1 eligible 2-D+ weight >= {min_elems} elements")
  if "fsdp" not in seen_axes or "tensor" not in seen_axes:
    raise AssertionError(f"{name}: weights must shard across both 'fsdp' and 'tensor': {sorted(seen_axes)}")


def check_no_expert_weight_all_gather(spec_obj: spec.ConformanceSpec) -> None:
  """Check 8: Enforces comparison-doc expert `all-gather` OOM invariant in AOT-compiled 4-device HLO."""
  name = spec_obj.model_name
  if not spec_obj.expert_weight_param_patterns:
    raise AssertionError(f"{name}: spec.expert_weight_param_patterns is empty")
  driver_mesh = create_conformance_driver_mesh()
  with jax.sharding.set_mesh(driver_mesh), nnx.use_eager_sharding(True):
    model = spec_obj.build_model(driver_mesh, nnx.Rngs(0), True, "none")
    batch = spec_obj.build_batch(jax.random.PRNGKey(0), driver_mesh)
    param_vars = _flatten_vars(nnx.state(model, nnx.Param))
    _validate_stacked_patterns(spec_obj, param_vars)
    forbidden_shapes: dict[tuple[int, ...], str] = {}
    for pat in spec_obj.expert_weight_param_patterns:
      hits = [(p_str, p_node) for p_str, p_node in param_vars if bool(re.search(pat, p_str))]
      if not hits:
        raise AssertionError(f"{name}: unmatched expert_weight_param_patterns entry {pat!r}")
      for path_str, param_node in hits:
        arr = _unwrap_leaf(param_node)
        if not isinstance(arr, jax.Array) or not isinstance(arr.sharding, jax.sharding.NamedSharding):
          raise AssertionError(f"{name}: expert param {path_str} must be NamedSharding jax.Array")
        if arr.sharding.mesh is not driver_mesh or not arr.addressable_shards:
          raise AssertionError(f"{name}: expert param {path_str} not placed on driver_mesh")
        full_shape = arr.shape
        shard_shape = arr.addressable_shards[0].data.shape
        is_stacked, unstacked_shape = _resolve_unstacked_shape(spec_obj, path_str, full_shape)
        if len(unstacked_shape) < 3:
          raise AssertionError(f"{name}: {path_str} unstacked {unstacked_shape} must be >= 3-D")
        if math.prod(shard_shape) >= math.prod(full_shape):
          raise AssertionError(f"{name}: {path_str} {full_shape} is unsharded (shard={shard_shape})")
        forbidden_shapes[full_shape] = path_str
        forbidden_shapes[unstacked_shape] = path_str
        if len(unstacked_shape) == 4:
          d0, d1, d2, d3 = unstacked_shape
          for s3 in ((d0, d1, d2 * d3), (d0, d1 * d2, d3)):
            forbidden_shapes[s3] = path_str
            if is_stacked:
              forbidden_shapes[(spec_obj.num_scanned_blocks, *s3)] = path_str

    graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
    weights = {k: 1.0 for k in spec_obj.aux_losses}

    @jax.jit
    def _compiled_step(p_state: Any, r_state: Any, b_inputs: Mapping[str, jax.Array]) -> Any:
      def _step_loss(p_inner: Any) -> jax.Array:
        merged = nnx.merge(graphdef, p_inner, r_state)
        fwd_out = spec_obj.forward_fn(merged, b_inputs)
        _validate_forward_output(spec_obj, fwd_out)
        return spec_obj.loss_fn(fwd_out, b_inputs, weights)

      return jax.value_and_grad(_step_loss)(p_state)

    hlo_text = _compiled_step.lower(params_state, rest_state, batch).compile().as_text()
    if hlo_text is None:
      raise AssertionError(f"{name}: compiled step as_text() returned None")

  for raw_line in hlo_text.splitlines():
    stripped = raw_line.strip()
    if stripped.startswith("//"):
      continue
    ag_match = _ALL_GATHER_OPCODE_RE.search(stripped)
    if ag_match is None:
      continue
    lhs_prefix = stripped[: ag_match.start()]
    for dims_str in _HLO_TENSOR_SHAPE_RE.findall(lhs_prefix):
      ag_shape = tuple(int(dim) for dim in dims_str.split(",") if dim.strip())
      if ag_shape in forbidden_shapes:
        raise AssertionError(
            f"{name}: full expert weight all-gather of {forbidden_shapes[ag_shape]} {ag_shape}: {stripped}"
        )


def _bind_single_device(
    check_fn: Callable[[spec.ConformanceSpec, jax.sharding.Mesh], None],
) -> Callable[[spec.ConformanceSpec], None]:
  """Adapts a single-device check `(spec, mesh) -> None` to `(spec) -> None`."""

  def _runner(spec_obj: spec.ConformanceSpec) -> None:
    check_fn(spec_obj, _create_single_device_mesh())

  return _runner


_CHECK_RUNNERS: Mapping[spec.CheckId, Callable[[spec.ConformanceSpec], None]] = {
    spec.CheckId.SCAN_REMAT_EQUIVALENCE: _bind_single_device(check_scan_remat_equivalence),
    spec.CheckId.PARAM_AND_GRAD_HEALTH: _bind_single_device(check_param_and_grad_health),
    spec.CheckId.NON_PARAM_STATE_STABILITY: _bind_single_device(check_non_param_state_stability),
    spec.CheckId.AUX_LOSS_GRAD_COUPLING: _bind_single_device(check_aux_loss_grad_coupling),
    spec.CheckId.DRIVER_MESH_AND_AXIS_NAMES: check_driver_mesh_and_axis_names,
    spec.CheckId.NO_POST_INIT_PARAM_CONSTRAINTS: check_no_post_init_param_constraints,
    spec.CheckId.WEIGHT_SHARDING_COVERAGE: check_weight_sharding_coverage,
    spec.CheckId.NO_EXPERT_WEIGHT_ALL_GATHER: check_no_expert_weight_all_gather,
}


def run_conformance_check(spec_obj: spec.ConformanceSpec, check_id: spec.CheckId) -> bool:
  """Dispatches `check_id` across Checks 1-8 (`m3/README.md` Invariants (1)-(7)), respecting `skips`."""
  if not isinstance(check_id, spec.CheckId) or check_id not in _CHECK_RUNNERS:
    raise ValueError(f"{spec_obj.model_name}: unsupported check_id {check_id!r}")
  if _should_skip(spec_obj, check_id):
    return False
  _CHECK_RUNNERS[check_id](spec_obj)
  return True
