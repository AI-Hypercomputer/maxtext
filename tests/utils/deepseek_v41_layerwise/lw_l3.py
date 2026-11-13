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

"""Gate L3: MaxText-vs-MaxText invariances of the model code on real weights.

  * scan vs unscan: prefix + >= 2 scanned repeats + suffix through the production scan applier, against the same
    layers unscanned; forward and every gradient leaf within `scan_vs_unscan_max_abs`.
  * remat on vs off: the unscanned chain through the production remat applier; forward within
    `remat_forward_max_abs`, every gradient leaf within `remat_backward_max_abs`.
  * EP=1 vs EP=<all devices>: one layer, dense MoE, replayed routing, at the tier-a tolerance.
  * MoE implementations: dense vs sparse megablox vs sparse ragged_dot, natural routing, at the tier-a tolerance.

Each comparison has a negative control: one leaf of one side is moved by `negative_control_rel_scale` (relative
norm) and the same comparison must fail.

Device memory: only one program's gradients are on device at a time; the reference side's gradients are kept on the
host and compared leaf by leaf on device.
"""

import dataclasses
import functools
import gc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.layers import moe
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config, lw_model, lw_programs
from tests.utils.parity import negctl

# The tier each L3 comparison runs in (the AOT checker compiles the same table).
L3_CHAIN_TIER = lw_config.TIER_PRODUCTION
L3_MOE_TIER = lw_config.TIER_CODE_PARITY

# Compiled L3 chain programs (`lw_programs.chain_program_name`), each forward + grads.
L3_CHAIN_MODES = ("unscanned", "unscanned_remat", "scanned")


@dataclasses.dataclass(frozen=True)
class MoeVariant:
  """One compiled L3 MoE program: a single-layer forward."""

  expert_parallelism: int
  sparse_matmul: bool
  megablox: bool
  replayed_routing: bool
  activation_quant_off: bool

  def program_name(self) -> str:
    return (
        f"L3.moe[ep={self.expert_parallelism},sparse={self.sparse_matmul},megablox={self.megablox},"
        f"replay={self.replayed_routing},activation_quant_off={self.activation_quant_off}]"
    )

  def config_overrides(self) -> dict[str, Any]:
    """Per-program config flags of this variant (the AOT checker and the harness build the same config)."""
    flags = {"scan_layers": False, "remat_policy": "none", "sparse_matmul": self.sparse_matmul, "megablox": self.megablox}
    if self.activation_quant_off:
      flags[lw_config.L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE] = False
    return flags


def moe_variants(num_devices: int) -> dict[str, MoeVariant]:
  """The L3 MoE programs: EP=1 / EP=`num_devices` with replayed routing (tier activation quant); dense / megablox /
  ragged_dot with natural routing and activation quant off (`lw_config.L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE`)."""
  return {
      "dense_ep1": MoeVariant(1, False, False, True, False),
      "dense_epn": MoeVariant(num_devices, False, False, True, False),
      "dense_natural": MoeVariant(1, False, False, False, True),
      "sparse_megablox": MoeVariant(1, True, True, False, True),
      "sparse_ragged": MoeVariant(1, True, False, False, True),
  }


@dataclasses.dataclass(frozen=True)
class ChainInputs:
  """Inputs of a chain starting at its first layer (host arrays)."""

  carry: Mapping[str, np.ndarray]
  positions: np.ndarray
  tokens: np.ndarray
  image_mask: np.ndarray
  cotangent_h: np.ndarray
  decoder_segment_ids: np.ndarray | None = None


def _flat(state: nnx.State) -> dict[str, Any]:
  """Leaves of a module State keyed by their pure-dict path ("mlp/MoeBlock_0/wi_0")."""
  pure = nnx.to_pure_dict(state)
  return {negctl.leaf_path(path): leaf for path, leaf in jax.tree_util.tree_flatten_with_path(pure)[0]}


def _diff_metrics_on_device(a: jax.Array, b: Any) -> dict[str, float | None]:
  """max |a - b| and ||a - b|| / ||b|| (None when ||b|| < `lw_checks.COSINE_NORM_FLOOR`), reduced on device."""
  b_dev = jax.device_put(jnp.asarray(b, a.dtype), a.sharding)
  diff = a.astype(jnp.float32) - b_dev.astype(jnp.float32)
  max_abs = float(jnp.max(jnp.abs(diff)))
  diff_norm = float(jnp.sqrt(jnp.sum(jnp.square(diff))))
  ref_norm = float(jnp.sqrt(jnp.sum(jnp.square(b_dev.astype(jnp.float32)))))
  b_dev.delete()
  rel_l2 = diff_norm / ref_norm if ref_norm >= lw_checks.COSINE_NORM_FLOOR else None
  return {"max_abs": max_abs, "rel_l2": rel_l2, "ref_l2_norm": ref_norm}


def compare_leaves(
    label: str, cand: Mapping[str, jax.Array], ref_host: Mapping[str, np.ndarray], max_abs: float
) -> list[lw_checks.CheckResult]:
  """Per-leaf `max |cand - ref| <= max_abs` (rel_l2 is reported, not gated); the leaf sets must be identical."""
  if set(cand) != set(ref_host):
    raise ValueError(
        f"{label}: leaf sets differ; only candidate {sorted(set(cand) - set(ref_host))[:5]}, "
        f"only reference {sorted(set(ref_host) - set(cand))[:5]}"
    )
  results = []
  for key in sorted(cand):
    metrics = _diff_metrics_on_device(cand[key], ref_host[key])
    finite = bool(np.isfinite(metrics["max_abs"]))
    results.append(
        lw_checks.CheckResult(
            label=f"{label}/{key}",
            passed=finite and metrics["max_abs"] <= max_abs,
            metrics=metrics,
            bound={"max_abs": max_abs},
        )
    )
  return results


def unstack_scanned_grads(
    geometry: lw_programs.ScanGeometry, config: Any, grads: Mapping[str, Any]
) -> dict[str, jax.Array]:
  """Scanned-program gradients keyed like the unscanned chain's: "<layer index in chain>/<leaf path>".

  The repeat axis of each stacked leaf follows `lw_programs.stack_axis`, keyed on the leaf's variable type
  (`moe.MoEBiasVar` vs `nnx.Param`), as `stack_group_weights` stacks them.
  """
  out: dict[str, jax.Array] = {}
  for i, w in enumerate(grads["prefix"]):
    out.update({f"{i}/{k}": v for k, v in _flat(w).items()})
  offset = len(geometry.prefix)
  block = len(geometry.repeats[0])
  for is_bias, part in (
      (False, nnx.filter_state(grads["stack"], nnx.Param)),
      (True, nnx.filter_state(grads["stack"], moe.MoEBiasVar)),
  ):
    for key, leaf in _flat(part).items():
      slot, _, sub = key.partition("/")
      j = int(slot.removeprefix("layers_"))
      axis = lw_programs.stack_axis(config, is_bias=is_bias, stacked_ndim=leaf.ndim)
      for r in range(len(geometry.repeats)):
        out[f"{offset + r * block + j}/{sub}"] = jnp.take(leaf, r, axis=axis)
  offset += block * len(geometry.repeats)
  for i, w in enumerate(grads["suffix"]):
    out.update({f"{offset + i}/{k}": v for k, v in _flat(w).items()})
  return out


def unscanned_grad_leaves(grads: Sequence[Any]) -> dict[str, jax.Array]:
  return {f"{i}/{k}": v for i, w in enumerate(grads) for k, v in _flat(w).items()}


def _to_host(tree: Mapping[str, jax.Array]) -> dict[str, np.ndarray]:
  return {k: np.asarray(jax.device_get(v)) for k, v in tree.items()}


def _device_inputs(inputs: ChainInputs, config: Any, mesh: Any) -> dict[str, Any]:
  rep = lw_programs.replicated(mesh)
  carry = {k: jax.device_put(jnp.asarray(v, config.dtype if k == "h" else v.dtype), rep) for k, v in inputs.carry.items()}
  decoder_segment_ids = (
      jax.device_put(jnp.asarray(inputs.decoder_segment_ids, jnp.int32), rep)
      if inputs.decoder_segment_ids is not None
      else None
  )
  return {
      "carry": carry,
      "positions": jax.device_put(jnp.asarray(inputs.positions, jnp.int32), rep),
      "tokens": jax.device_put(jnp.asarray(inputs.tokens, jnp.int32), rep),
      "decoder_segment_ids": decoder_segment_ids,
      "image_mask": jax.device_put(jnp.asarray(inputs.image_mask, jnp.bool_), rep),
      "cotangent_h": jax.device_put(jnp.asarray(inputs.cotangent_h, jnp.float32), rep),
  }


def _status(results: Sequence[lw_checks.CheckResult]) -> str:
  return lw_checks.STATUS_PASS if results and all(r.passed for r in results) else lw_checks.STATUS_FAIL


def _negative_control(
    label: str, clean: Sequence[lw_checks.CheckResult], results: Sequence[lw_checks.CheckResult]
) -> dict[str, Any]:
  """A negative control FIRES when a comparison that passed on the clean side fails under the perturbation
  (`lw_checks.control_fired`, matched by label)."""
  fired = lw_checks.control_fired([r.to_json() for r in clean], [r.to_json() for r in results])
  return {
      "label": label,
      "status": "FIRED" if fired else "DID_NOT_FIRE",
      "checks": [r.to_json() for r in results if not r.passed][:5],
  }


@dataclasses.dataclass
class LoadedChain:
  """Real weights of the chain's layers, loaded once (unscanned layout)."""

  config: Any
  mesh: Any
  layer_ids: tuple[int, ...]
  parts: list[lw_model.ModuleParts]


def load_chain(
    config: Any, mesh: Any, layer_ids: Sequence[int], load_fn: Callable[[Any, Any, int], lw_model.ModuleParts]
) -> LoadedChain:
  return LoadedChain(config, mesh, tuple(layer_ids), [load_fn(config, mesh, i) for i in layer_ids])


def run_chain_invariance(  # pylint: disable=too-many-positional-arguments
    chain: LoadedChain,
    remat_config: Any,
    scan_config: Any,
    geometry: lw_programs.ScanGeometry,
    inputs: ChainInputs,
    tol: lw_checks.Tolerances,
    nc_leaf: str,
) -> dict[str, Any]:
  """scan-vs-unscan and remat-on-vs-off on the loaded chain, each with its negative control.

  Args:
    chain: Loaded layers of `geometry.layer_ids` under the unscanned, remat-off config.
    remat_config: Same as `chain.config` but `remat_policy=full`.
    scan_config: Same as `chain.config` but `scan_layers=True`.
    geometry: The scanned layout of the same layers.
    inputs: Chain inputs and the output cotangent.
    tol: Harness tolerances.
    nc_leaf: Leaf path (inside one layer's weights) the negative controls perturb, e.g. "mlp/shared_experts/wo/kernel".
  """
  if chain.layer_ids != geometry.layer_ids:
    raise ValueError(f"Chain layers {chain.layer_ids} != geometry layers {geometry.layer_ids}")
  geometry.validate(chain.config)
  mesh = chain.mesh
  dev = _device_inputs(inputs, chain.config, mesh)
  args = (dev["carry"], dev["positions"], dev["tokens"], dev["image_mask"], dev["cotangent_h"])
  weights = [p.weights for p in chain.parts]
  rest = [p.rest for p in chain.parts]
  graphdefs = tuple(p.graphdef for p in chain.parts)

  def unscanned_entry(config: Any, remat: bool) -> Callable[..., Any]:
    program = lw_programs.GroupProgram(config, mesh, chain.layer_ids, graphdefs, (None,) * len(graphdefs))
    return jax.jit(lw_programs.chain_entry(functools.partial(lw_programs.chain_unscanned, program, remat=remat), config))

  # Op-for-op execution: the gate compares the production scan / remat appliers algebraically against the unscanned
  # no-remat baseline without whole-graph XLA cross-layer fusion differences between compiled scan / remat variants.
  report: dict[str, Any] = {}
  off = unscanned_entry(chain.config, False)
  with jax.disable_jit():
    out_ref, grads_ref = off(weights, rest, *args)
  ref_out_host = _to_host(out_ref)
  ref_grads_host = _to_host({**unscanned_grad_leaves(grads_ref[0]), "d_h_in": grads_ref[1], "d_pre_mix_in": grads_ref[2]})
  lw_model.release({"out": out_ref, "grads": grads_ref})
  del out_ref, grads_ref, off
  jax.clear_caches()
  gc.collect()

  on = unscanned_entry(remat_config, True)
  with jax.disable_jit():
    out_remat, grads_remat = on(weights, rest, *args)
  remat_checks = [
      lw_checks.check_max_abs("remat.forward.h", out_remat["h"], ref_out_host["h"], tol.remat_forward_max_abs),
      lw_checks.check_max_abs(
          "remat.forward.pre_mix", out_remat["pre_mix"], ref_out_host["pre_mix"], tol.remat_forward_max_abs
      ),
  ]
  remat_grads = {**unscanned_grad_leaves(grads_remat[0]), "d_h_in": grads_remat[1], "d_pre_mix_in": grads_remat[2]}
  remat_checks += compare_leaves("remat.grad", remat_grads, ref_grads_host, tol.remat_backward_max_abs)
  lw_model.release({"out": out_remat, "grads": grads_remat})
  del out_remat, grads_remat, remat_grads
  perturbed = list(weights)
  perturbed[0] = _perturb(weights[0], nc_leaf, tol.negative_control_rel_scale)
  with jax.disable_jit():
    out_nc, grads_nc = on(perturbed, rest, *args)
  nc_remat_chk = lw_checks.check_max_abs("remat.forward.h", out_nc["h"], ref_out_host["h"], tol.remat_forward_max_abs)
  lw_model.release({"out": out_nc, "grads": grads_nc})
  report["remat"] = {
      "status": _status(remat_checks),
      "checks": [r.to_json() for r in remat_checks],
      "negative_control": _negative_control(
          f"remat.forward.h with layer {chain.layer_ids[0]} {nc_leaf} perturbed",
          remat_checks,
          [nc_remat_chk],
      ),
  }
  lw_model.release(perturbed[0], keep=weights[0])
  del out_nc, grads_nc, perturbed, on
  jax.clear_caches()
  gc.collect()

  scanned_report = _run_scanned(
      chain, scan_config, geometry, weights, rest, args, ref_out_host, ref_grads_host, tol, nc_leaf
  )
  report["scan_vs_unscan"] = scanned_report
  return report


def _perturb(weights: nnx.State, leaf: str, rel_scale: float) -> nnx.State:
  return lw_model.perturb_leaf(weights, leaf, rel_scale, seed=0)


def _run_scanned(chain, scan_config, geometry, weights, rest, args, ref_out_host, ref_grads_host, tol, nc_leaf):  # pylint: disable=too-many-positional-arguments
  """Builds the scanned stack from the loaded repeat layers and compares the scanned chain with the unscanned one."""
  mesh = chain.mesh
  n_prefix = len(geometry.prefix)
  n_stacked = sum(len(g) for g in geometry.repeats)
  block = len(geometry.repeats[0])
  stack_build = lw_programs.scanned_block_build_fn(scan_config, mesh, geometry.repeats)
  stack_abstract = lw_model.abstract_parts(stack_build, scan_config, mesh)
  per_repeat = [
      nnx.State({f"layers_{j}": weights[n_prefix + r * block + j] for j in range(block)})
      for r in range(len(geometry.repeats))
  ]
  stack_weights = lw_programs.stack_group_weights(scan_config, per_repeat, stack_abstract.weights)
  stack_rest = lw_model.materialize_rest(stack_build, scan_config, mesh)
  for w in weights[n_prefix : n_prefix + n_stacked]:
    lw_model.release(w)
  graphdefs = [p.graphdef for p in chain.parts]
  program = lw_programs.ScannedProgram(
      config=scan_config,
      geometry=geometry,
      prefix=lw_programs.GroupProgram(
          scan_config, mesh, geometry.prefix, tuple(graphdefs[:n_prefix]), (None,) * n_prefix
      ),
      stack_graphdef=stack_abstract.graphdef,
      suffix=lw_programs.GroupProgram(
          scan_config, mesh, geometry.suffix, tuple(graphdefs[n_prefix + n_stacked :]), (None,) * len(geometry.suffix)
      ),
  )
  entry = jax.jit(lw_programs.chain_entry(functools.partial(lw_programs.chain_scanned, program), scan_config))
  s_weights = {"prefix": weights[:n_prefix], "stack": stack_weights, "suffix": weights[n_prefix + n_stacked :]}
  s_rest = {"prefix": rest[:n_prefix], "stack": stack_rest, "suffix": rest[n_prefix + n_stacked :]}
  with jax.disable_jit():
    out_scan, grads_scan = entry(s_weights, s_rest, *args)
  checks = [
      lw_checks.check_max_abs("scan.forward.h", out_scan["h"], ref_out_host["h"], tol.scan_vs_unscan_max_abs),
      lw_checks.check_max_abs(
          "scan.forward.pre_mix", out_scan["pre_mix"], ref_out_host["pre_mix"], tol.scan_vs_unscan_max_abs
      ),
  ]
  scan_grads = {
      **unstack_scanned_grads(geometry, scan_config, grads_scan[0]),
      "d_h_in": grads_scan[1],
      "d_pre_mix_in": grads_scan[2],
  }
  checks += compare_leaves("scan.grad", scan_grads, ref_grads_host, tol.scan_vs_unscan_max_abs)
  lw_model.release({"out": out_scan, "grads": grads_scan})
  del out_scan, grads_scan, scan_grads
  nc_weights = dict(s_weights)
  nc_weights["stack"] = _perturb(stack_weights, f"layers_0/{nc_leaf}", tol.negative_control_rel_scale)
  with jax.disable_jit():
    out_nc, grads_nc = entry(nc_weights, s_rest, *args)
  nc_scan_chk = lw_checks.check_max_abs("scan.forward.h", out_nc["h"], ref_out_host["h"], tol.scan_vs_unscan_max_abs)
  lw_model.release({"out": out_nc, "grads": grads_nc})
  lw_model.release(nc_weights["stack"], keep=stack_weights)
  lw_model.release(stack_weights)
  del out_nc, grads_nc, nc_weights, stack_weights, entry
  jax.clear_caches()
  gc.collect()
  return {
      "status": _status(checks),
      "geometry": dataclasses.asdict(geometry),
      "checks": [r.to_json() for r in checks],
      "negative_control": _negative_control(
          f"scan.forward.h with stacked {nc_leaf} perturbed",
          checks,
          [nc_scan_chk],
      ),
  }


def reshard_parts(
    parts: lw_model.ModuleParts, config: Any, mesh: Any, build_fn: Callable[[], Any]
) -> lw_model.ModuleParts:
  """`parts`' weights moved on device to the shardings `build_fn` has under (`config`, `mesh`), with a new graph."""
  target = lw_model.abstract_parts(build_fn, config, mesh)
  weights = jax.tree.map(lambda x, a: jax.device_put(x, a.sharding), parts.weights, target.weights)
  return lw_model.ModuleParts(
      graphdef=target.graphdef, weights=weights, rest=lw_model.materialize_rest(build_fn, config, mesh)
  )


def _forward_and_grads(out: Mapping[str, Any], grads: Any) -> dict[str, Any]:
  """One MoE-family side as flat leaves: forward h / pre_mix, every weight gradient, d_h_in, d_pre_mix_in."""
  return {
      "forward/h": out["h"],
      "forward/pre_mix": out["pre_mix"],
      **{f"grad/{k}": v for k, v in unscanned_grad_leaves(grads[0]).items()},
      "grad/d_h_in": grads[1],
      "grad/d_pre_mix_in": grads[2],
  }


def _code_parity_leaves(
    label: str, cand: Mapping[str, Any], ref_host: Mapping[str, np.ndarray], tol: lw_checks.Tolerances
) -> list[lw_checks.CheckResult]:
  """`check_code_parity` on every leaf (the tier-a predicate); the leaf sets must be identical."""
  if set(cand) != set(ref_host):
    raise ValueError(
        f"{label}: leaf sets differ; only candidate {sorted(set(cand) - set(ref_host))[:5]}, "
        f"only reference {sorted(set(ref_host) - set(cand))[:5]}"
    )
  return [
      lw_checks.check_code_parity(f"{label}/{k}", np.asarray(jax.device_get(cand[k])), ref_host[k], tol.code_parity)
      for k in sorted(cand)
  ]


def run_moe_invariance(  # pylint: disable=too-many-positional-arguments
    layer_id: int,
    base: lw_model.ModuleParts,
    configs: Mapping[str, tuple[Any, Any]],
    inputs: ChainInputs,
    replay: Mapping[str, np.ndarray],
    tol: lw_checks.Tolerances,
    nc_leaf: str,
) -> dict[str, Any]:
  """EP=1 vs EP=all (replayed routing) and dense vs sparse megablox / ragged_dot (natural routing) on one layer.

  Each family compares the forward h / pre_mix and every gradient leaf, d_h_in and d_pre_mix_in of
  d<h_out, cotangent_h> at the tier-a predicate; its negative control moves `nc_leaf` on the variant side and the
  forward h comparison must fail.

  Args:
    layer_id: The real layer.
    base: Its weights loaded under `configs["dense_ep1"]`.
    configs: name -> (config, mesh) for every `moe_variants` name.
    inputs: The layer's carry, inputs and output cotangent.
    replay: The replay dict of the layer for the EP comparison.
    tol: Harness tolerances (`code_parity` bounds every comparison; L3 MoE runs in tier a).
    nc_leaf: Leaf path the negative controls perturb.
  """
  report: dict[str, Any] = {}
  config1, mesh1 = configs["dense_ep1"]
  dev = _device_inputs(inputs, config1, mesh1)
  io = (dev["carry"], dev["positions"], dev["tokens"], dev["image_mask"])
  rep_dev = [{k: jax.device_put(jnp.asarray(v), lw_programs.replicated(mesh1)) for k, v in replay.items()}]

  def replayed(parts: lw_model.ModuleParts, config: Any, mesh: Any) -> Callable[..., Any]:
    program = lw_programs.GroupProgram(config, mesh, (layer_id,), (parts.graphdef,), (None,))
    entry = jax.jit(lw_programs.replayed_chain_entry(program))
    return lambda w: entry([w], [parts.rest], *io, rep_dev, dev["cotangent_h"])

  run_1 = replayed(base, config1, mesh1)
  out1, grads1 = run_1(base.weights)
  ref_ep = _to_host(_forward_and_grads(out1, grads1))
  lw_model.release({"out": out1, "grads": grads1})
  del out1, grads1, run_1
  jax.clear_caches()
  gc.collect()

  config_n, mesh_n = configs["dense_epn"]
  parts_n = reshard_parts(base, config_n, mesh_n, lw_programs.layer_build_fn(config_n, mesh_n, layer_id))
  run_n = replayed(parts_n, config_n, mesh_n)
  out_n, grads_n = run_n(parts_n.weights)
  ep_checks = _code_parity_leaves("ep", _forward_and_grads(out_n, grads_n), ref_ep, tol)
  lw_model.release({"out": out_n, "grads": grads_n})
  del out_n, grads_n
  nc_weights = _perturb(parts_n.weights, nc_leaf, tol.negative_control_rel_scale)
  out_nc, grads_nc = run_n(nc_weights)
  nc_ep_chk = lw_checks.check_code_parity("ep/forward/h", np.asarray(out_nc["h"]), ref_ep["forward/h"], tol.code_parity)
  lw_model.release({"out": out_nc, "grads": grads_nc})
  report["ep1_vs_epn"] = {
      "status": _status(ep_checks),
      "expert_parallelism": int(config_n.ici_expert_parallelism),
      "checks": [r.to_json() for r in ep_checks],
      "negative_control": _negative_control(
          f"ep forward/h with {nc_leaf} perturbed on the EP={config_n.ici_expert_parallelism} side",
          ep_checks,
          [nc_ep_chk],
      ),
  }
  lw_model.release(nc_weights, keep=parts_n.weights)
  lw_model.release(parts_n.weights, keep=base.weights)
  lw_model.release(rep_dev)
  del out_nc, grads_nc, nc_weights, parts_n, run_n, ref_ep, rep_dev
  jax.clear_caches()
  gc.collect()

  def natural(config: Any, mesh: Any) -> tuple[Callable[..., Any], lw_model.ModuleParts]:
    parts = reshard_parts(base, config, mesh, lw_programs.layer_build_fn(config, mesh, layer_id))
    program = lw_programs.GroupProgram(config, mesh, (layer_id,), (parts.graphdef,), (None,))
    entry = jax.jit(lw_programs.chain_entry(functools.partial(lw_programs.chain_unscanned, program, remat=False), config))
    return (lambda w: entry([w], [parts.rest], *io, dev["cotangent_h"])), parts

  dense_fn, dense_parts = natural(*configs["dense_natural"])
  dense_out, dense_grads = dense_fn(dense_parts.weights)
  ref_dense = _to_host(_forward_and_grads(dense_out, dense_grads))
  lw_model.release({"out": dense_out, "grads": dense_grads})
  lw_model.release(dense_parts.weights, keep=base.weights)
  del dense_out, dense_grads, dense_parts, dense_fn
  jax.clear_caches()
  gc.collect()

  impl = {}
  for name in ("sparse_megablox", "sparse_ragged"):
    config, mesh = configs[name]
    if name == "sparse_megablox" and jax.default_backend() != "tpu":
      impl[name] = {
          "status": lw_checks.STATUS_NOT_RUN_ON_BACKEND,
          "reason": f"megablox is a TPU Pallas kernel; backend is {jax.default_backend()}",
      }
      continue
    fn, parts = natural(config, mesh)
    out, grads = fn(parts.weights)
    checks = _code_parity_leaves(name, _forward_and_grads(out, grads), ref_dense, tol)
    lw_model.release({"out": out, "grads": grads})
    del out, grads
    nc = _perturb(parts.weights, nc_leaf, tol.negative_control_rel_scale)
    nc_out, nc_grads = fn(nc)
    nc_sp_chk = lw_checks.check_code_parity(
        f"{name}/forward/h", np.asarray(nc_out["h"]), ref_dense["forward/h"], tol.code_parity
    )
    lw_model.release({"out": nc_out, "grads": nc_grads})
    impl[name] = {
        "status": _status(checks),
        "checks": [r.to_json() for r in checks],
        "negative_control": _negative_control(
            f"{name} forward/h with {nc_leaf} perturbed",
            checks,
            [nc_sp_chk],
        ),
    }
    lw_model.release(nc, keep=parts.weights)
    lw_model.release(parts.weights, keep=base.weights)
    del nc_out, nc_grads, nc, parts, fn
    jax.clear_caches()
    gc.collect()
  report["moe_implementations_vs_dense"] = impl
  report["activation_quant"] = {name: bool(config.ds41_activation_quant) for name, (config, _) in configs.items()}
  return report
