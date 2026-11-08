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

"""Gate L4: per-layer VJP of real layers against the reference's autograd, leaf by leaf.

Candidate gradients are converted to the reference's HF tensor layout with the canonical
`param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(saving_to_hf=True)`; MaxText and HF names are related only
through `DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING`. Leaf completeness is checked both ways: every candidate weight
leaf must map to a reference gradient or to the declared non-gradient set (the routing biases, which only select
experts), and every reference gradient must come from a candidate leaf.

Routed experts: full gradients of the reference's sampled experts, plus per-expert Frobenius norm, max-abs and the
reference's sketch S_left @ dW_e @ S_right for all experts. Engram table: gradients of the active rows; the control
rows, the poison row and the padding rows must be exactly zero.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.checkpoint_conversion.utils import param_mapping
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config, lw_l2, lw_model, lw_programs
from tests.utils.parity import negctl

NON_GRADIENT_SUFFIXES = ("ffn.gate.bias", "ffn.gate.bias_vl")
EXPERT_WEIGHTS = {"wi_0": "w1", "wi_1": "w3", "wo": "w2"}


@dataclasses.dataclass(frozen=True)
class ReferenceVJP:
  """Reference VJP of one layer, keyed by HF tensor names (built by the bundle adapter).

  dense: HF name -> gradient for every non-expert, non-engram-table leaf.
  expert_frob / expert_max_abs / expert_sketch: per MaxText expert weight ("wi_0", "wi_1", "wo"), over all experts.
  sketch_left / sketch_right: the reference's sketch matrices per expert weight, in HF layout.
  expert_samples: (expert id, MaxText expert weight) -> full HF-layout gradient.
  engram_active_rows / engram_active_grads: row ids and their gradients; None exactly for non-engram layers.
  """

  layer_id: int
  carry_in: Mapping[str, np.ndarray]
  replay: Mapping[str, np.ndarray]
  cotangents: Mapping[str, np.ndarray]
  d_h_in: lw_l2.ReferenceTensor
  d_pre_mix_in: lw_l2.ReferenceTensor
  dense: Mapping[str, lw_l2.ReferenceTensor]
  expert_frob: Mapping[str, lw_l2.ReferenceTensor]
  expert_max_abs: Mapping[str, lw_l2.ReferenceTensor]
  expert_sketch: Mapping[str, lw_l2.ReferenceTensor]
  sketch_left: Mapping[str, np.ndarray]
  sketch_right: Mapping[str, np.ndarray]
  expert_samples: Mapping[tuple[int, str], lw_l2.ReferenceTensor]
  engram_active_rows: np.ndarray | None
  engram_active_grads: lw_l2.ReferenceTensor | None


def hf_layer_mapping(config: Any, layer_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
  """MaxText leaf path in the layer ("self_attention/wq_a/kernel") -> (HF key or per-expert key list, saving hook)."""
  quant = {"quantization_config": False}
  param_map = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(quant, config, scan_layers=False)
  hooks = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(quant, config, scan_layers=False, saving_to_hf=True)
  names, hook_by_path = {}, {}
  for mt_key, hf in param_map.items():
    for collection in ("params-", "MoEBiasVar-"):
      prefix = f"{collection}decoder-layers_{layer_id}-"
      if mt_key.startswith(prefix):
        path = mt_key[len(prefix) :].replace("-", "/")
        names[path] = hf
        hook_by_path[path] = hooks[mt_key] if mt_key in hooks else None
  return names, hook_by_path


def to_hf(grad: np.ndarray, hook: Any, hf_shape: Sequence[int]) -> np.ndarray:
  return np.asarray(grad) if hook is None else np.asarray(hook(np.asarray(grad), tuple(hf_shape)))


def _check(
    label: str, cand: np.ndarray, ref: lw_l2.ReferenceTensor, bounds: lw_l2.GateBounds, bound_key: str
) -> dict[str, Any]:
  return lw_l2.compare_tensor(label, cand, ref, None, bounds, bound_key)


def leaf_checks(  # pylint: disable=too-many-positional-arguments
    config: Any,
    layer_id: int,
    grads: Mapping[str, np.ndarray],
    ref: ReferenceVJP,
    bounds: lw_l2.GateBounds,
    engram_plan: lw_model.EngramSlice | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """All leaf-gradient checks of one layer, and the completeness accounting."""
  names, hooks = hf_layer_mapping(config, layer_id)
  checks: list[dict[str, Any]] = []
  candidate_paths = set(grads)
  unmapped = sorted(candidate_paths - set(names))
  if unmapped:
    raise ValueError(f"Layer {layer_id}: candidate leaves with no HF mapping: {unmapped}")
  covered_ref: set[str] = set()
  for path in sorted(candidate_paths):
    hf = names[path]
    grad = grads[path]
    if path.endswith("engram/embed"):
      checks += _engram_checks(layer_id, grad, ref, bounds, engram_plan)
      covered_ref.add(hf)
      continue
    if isinstance(hf, list):
      checks += _expert_checks(layer_id, path, grad, hf, hooks[path], ref, bounds)
      continue
    if hf.endswith(NON_GRADIENT_SUFFIXES):
      worst = float(np.max(np.abs(grad)))
      checks.append(
          {
              "label": f"layer_{layer_id}.{hf}.non_gradient",
              "status": lw_checks.STATUS_PASS if worst == 0.0 else lw_checks.STATUS_FAIL,
              "metrics": {"max_abs": worst},
              "bound": {"max_abs": 0.0},
          }
      )
      continue
    ref_t = lw_checks.require(ref.dense, hf, f"reference gradients of layer {layer_id}")
    covered_ref.add(hf)
    checks.append(_check(f"layer_{layer_id}.{hf}", to_hf(grad, hooks[path], ref_t.fp32.shape), ref_t, bounds, "dW_dense"))
  extra_ref = sorted(set(ref.dense) - covered_ref)
  if extra_ref:
    raise ValueError(f"Layer {layer_id}: reference gradients with no candidate leaf: {extra_ref}")
  return checks, {"candidate_leaves": len(candidate_paths), "reference_dense_leaves": len(ref.dense)}


def _expert_checks(layer_id, path, grad, hf_keys, hook, ref, bounds) -> list[dict[str, Any]]:  # pylint: disable=too-many-positional-arguments
  """Per-expert Frobenius norms, max |g|, sketches over all experts, and the sampled experts' full gradients."""
  name = path.rsplit("/", 1)[-1]
  w = EXPERT_WEIGHTS[name]
  hf_shape = tuple(grad.shape[1:][::-1])  # torch Linear [out, in] of one expert
  per_expert = [to_hf(grad[e], hook, hf_shape) for e in range(grad.shape[0])]
  frob = np.asarray([np.linalg.norm(x.astype(np.float64)) for x in per_expert])
  max_abs = np.asarray([np.max(np.abs(x)) for x in per_expert], np.float64)
  sketch = np.stack(
      [
          ref.sketch_left[name].astype(np.float64) @ x.astype(np.float64) @ ref.sketch_right[name].astype(np.float64)
          for x in per_expert
      ]
  )
  if len(hf_keys) != grad.shape[0]:
    raise ValueError(f"Layer {layer_id} {path}: {grad.shape[0]} experts but {len(hf_keys)} HF keys")
  checks = [
      _check(f"layer_{layer_id}.experts.{w}.frob", frob, ref.expert_frob[name], bounds, "dW_expert_frob"),
      _check(f"layer_{layer_id}.experts.{w}.max_abs", max_abs, ref.expert_max_abs[name], bounds, "dW_expert_max_abs"),
      _check(f"layer_{layer_id}.experts.{w}.sketch", sketch, ref.expert_sketch[name], bounds, "dW_expert_sketch"),
  ]
  for (eid, weight_name), ref_t in sorted(ref.expert_samples.items()):
    if weight_name == name:
      checks.append(_check(f"layer_{layer_id}.experts.{eid}.{w}", per_expert[eid], ref_t, bounds, "dW_expert_sample"))
  return checks


def _engram_checks(layer_id, grad, ref, bounds, plan) -> list[dict[str, Any]]:
  """Engram table gradient: the reference's active rows, and exactly 0 on every other slice row."""
  if plan is None or ref.engram_active_rows is None:
    raise ValueError(f"Layer {layer_id} has an engram table but no slice plan / reference active rows")
  grad = np.asarray(grad)
  pos = np.searchsorted(plan.row_ids, ref.engram_active_rows)
  if not np.array_equal(plan.row_ids[pos], ref.engram_active_rows):
    raise ValueError(f"Layer {layer_id}: reference active rows are not all in the slice")
  checks = [
      _check(f"layer_{layer_id}.engram.embed.active", grad[pos], ref.engram_active_grads, bounds, "dW_engram_active")
  ]
  untouched = np.ones(grad.shape[0], bool)
  untouched[pos] = False
  worst = float(np.max(np.abs(grad[untouched])))
  checks.append(
      {
          "label": f"layer_{layer_id}.engram.embed.untouched_rows",
          "status": lw_checks.STATUS_PASS if worst == 0.0 else lw_checks.STATUS_FAIL,
          "metrics": {"max_abs": worst, "rows": int(untouched.sum())},
          "bound": {"max_abs": 0.0},
      }
  )
  return checks


def grads_by_path(state: nnx.State) -> dict[str, np.ndarray]:
  pure = nnx.to_pure_dict(state)
  return {negctl.leaf_path(p): np.asarray(jax.device_get(v)) for p, v in jax.tree_util.tree_flatten_with_path(pure)[0]}


def _packing_control(  # pylint: disable=too-many-positional-arguments
    vjp_fn: Callable[..., Any],
    weights: nnx.State,
    rest: nnx.State,
    carry: Mapping[str, jax.Array],
    inputs: lw_l2.SampleInputs,
    replay: Mapping[str, jax.Array],
    cotangents: Mapping[str, jax.Array],
    layer_id: int,
) -> dict[str, Any]:
  """NC_L4_pack: 2-doc packed layout with unaligned Doc A boundary.

  Asserts ||dL_B / d(h_in_DocA)||_F == 0.0 under proper segment IDs,
  and > 1e-3 under injected decoder_segment_ids = jnp.ones_like(tokens).
  """
  b, s = inputs.tokens.shape[0], inputs.tokens.shape[1]
  if s < 16:
    return {"status": lw_checks.STATUS_NOT_APPLICABLE, "reason": "seq_len < 16 too short for 2-doc packing"}

  seg_proper, seg_ones, img_mask, s_a, s_b = lw_model.packing_control_segment_ids(inputs.tokens)

  dev_seg_proper = jnp.asarray(seg_proper)
  dev_seg_ones = jnp.asarray(seg_ones)

  pos_proper = np.zeros((b, s), dtype=np.int32)
  pos_proper[:, :s_a] = np.arange(s_a)
  pos_proper[:, s_a : s_a + s_b] = np.arange(s_b)
  dev_pos = jnp.asarray(pos_proper)

  dev_img = jnp.asarray(img_mask)
  dev_tokens = jnp.asarray(inputs.tokens)

  cot_b = {k: jnp.zeros_like(v) for k, v in cotangents.items()}
  cot_b["h"] = cot_b["h"].at[:, s_a : s_a + s_b].set(1.0)

  _, d_h_clean, _ = vjp_fn(
      weights, rest, carry, dev_pos, dev_tokens, dev_img, replay, cot_b, decoder_segment_ids=dev_seg_proper
  )
  frob_clean = float(jnp.linalg.norm(d_h_clean[:, :s_a]))

  _, d_h_ones, _ = vjp_fn(
      weights, rest, carry, dev_pos, dev_tokens, dev_img, replay, cot_b, decoder_segment_ids=dev_seg_ones
  )
  frob_ones = float(jnp.linalg.norm(d_h_ones[:, :s_a]))

  fired = frob_clean == 0.0 and frob_ones > 1e-3
  return {
      "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE,
      "grad_frob_clean": frob_clean,
      "grad_frob_ones": frob_ones,
      "layer": layer_id,
  }


def run_layer(
    program: lw_programs.GroupProgram,
    parts: lw_model.ModuleParts,
    ref: ReferenceVJP,
    inputs: lw_l2.SampleInputs,
    bounds: lw_l2.GateBounds,
    *,
    nc_leaf: str,
) -> dict[str, Any]:
  """Gate L4 on one layer, with NC_L4 (one weight leaf moved, VJP re-run, a gradient check must fail)."""
  rep = lw_programs.replicated(program.mesh)
  dev = lambda x, dtype=None: jax.device_put(jnp.asarray(x, dtype), rep)  # pylint: disable=unnecessary-lambda-assignment
  carry = {
      k: dev(v, program.config.dtype if k in ("h", "compress_kv", "index_k") else None) for k, v in ref.carry_in.items()
  }
  io = (dev(inputs.positions, jnp.int32), dev(inputs.tokens, jnp.int32), dev(inputs.image_mask, jnp.bool_))
  replay = {k: dev(v) for k, v in ref.replay.items()}
  cotangents = {k: dev(v, jnp.float32) for k, v in ref.cotangents.items()}
  seg_dev = dev(inputs.decoder_segment_ids, jnp.int32) if inputs.decoder_segment_ids is not None else None
  vjp = jax.jit(lw_programs.layer_vjp_entry(program))

  def evaluate(
      weights: nnx.State,
  ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    d_w, d_h, d_pre = vjp(weights, parts.rest, carry, *io, replay, cotangents, decoder_segment_ids=seg_dev)
    d_h_arr = np.asarray(d_h)
    d_pre_arr = np.asarray(d_pre)
    if inputs.decoder_segment_ids is not None:
      valid_h_np = (
          (inputs.decoder_segment_ids != 0)[:, :, None, None]
          if d_h_arr.ndim == 4
          else (inputs.decoder_segment_ids != 0)[..., None]
      )
      valid_pre_np = (inputs.decoder_segment_ids != 0)[..., None]
      d_h_arr = np.where(valid_h_np, d_h_arr, 0.0)
      d_pre_arr = np.where(valid_pre_np, d_pre_arr, 0.0)
    checks = [
        _check(f"layer_{ref.layer_id}.d_h_in", d_h_arr, ref.d_h_in, bounds, "dx"),
        _check(f"layer_{ref.layer_id}.d_pre_mix_in", d_pre_arr, ref.d_pre_mix_in, bounds, "d_pre_mix_in"),
    ]
    leaves = grads_by_path(d_w)
    lw_model.release([d_w, d_h, d_pre])
    leaf, accounting = leaf_checks(program.config, ref.layer_id, leaves, ref, bounds, program.engram_slices[0])
    return checks + leaf, accounting, leaves, d_h_arr, d_pre_arr

  checks, accounting, clean_leaves, clean_d_h, clean_d_pre = evaluate(parts.weights)
  moved = lw_model.perturb_leaf(parts.weights, nc_leaf, bounds.tol.negative_control_rel_scale, seed=0)
  nc_checks, _, nc_leaves, nc_d_h, nc_d_pre = evaluate(moved)
  lw_model.release(moved, keep=parts.weights)
  fired = lw_checks.control_fired(checks, nc_checks)
  if not fired and bounds.tier != lw_config.TIER_CODE_PARITY:
    nc_tol = bounds.tol.code_parity
    cand_vs_clean = [
        lw_checks.check_code_parity(f"layer_{ref.layer_id}.d_h_in", nc_d_h, clean_d_h, nc_tol),
        lw_checks.check_code_parity(f"layer_{ref.layer_id}.d_pre_mix_in", nc_d_pre, clean_d_pre, nc_tol),
    ]
    for k in clean_leaves:
      if k in nc_leaves:
        cand_vs_clean.append(
            lw_checks.check_code_parity(f"layer_{ref.layer_id}.{k}", nc_leaves[k], clean_leaves[k], nc_tol)
        )
    fired = any(c.status == lw_checks.STATUS_FAIL for c in cand_vs_clean)
  return {
      "status": lw_checks.worst_status([c["status"] for c in checks]),
      "checks": checks,
      "leaf_accounting": accounting,
      "negative_control": (
          {"NC_L4": {"leaf": nc_leaf, "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE}}
          if ref.layer_id not in (1, 2, 14, 20)
          else {
              "NC_L4": {"leaf": nc_leaf, "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE},
              "NC_L4_pack": _packing_control(
                  vjp, parts.weights, parts.rest, carry, inputs, replay, cotangents, ref.layer_id
              ),
          }
      ),
      "tier": bounds.tier,
      "tier_is_code_parity": bounds.tier == lw_config.TIER_CODE_PARITY,
  }


@dataclasses.dataclass(frozen=True)
class ReferenceHeadVJP:
  """Reference VJP of decoder_norm and logits_dense."""

  d_h_40: lw_l2.ReferenceTensor
  norm_scale: lw_l2.ReferenceTensor
  logits_row_frob: lw_l2.ReferenceTensor
  logits_row_max_abs: lw_l2.ReferenceTensor
  logits_sketch: lw_l2.ReferenceTensor
  logits_samples: Mapping[int, lw_l2.ReferenceTensor]


def run_head(  # pylint: disable=too-many-positional-arguments
    norm_weights: nnx.State,
    logits_weights: nnx.State,
    h_in: jax.Array,
    cotangent_logits: jax.Array,
    ref: ReferenceHeadVJP,
    bounds: lw_l2.GateBounds,
    sketch_matrix: np.ndarray,
    sample_rows: Sequence[int],
    *,
    nc_leaf: str = "scale",
) -> dict[str, Any]:
  """Gate L4 on decoder head: decoder_norm.scale + logits_dense.kernel (row norms, sketch, sampled rows, d_h_40)."""

  def loss(w_norm, w_logits, x):
    norm_val = w_norm["scale"]
    kernel_val = w_logits["kernel"]
    h_norm = x * norm_val
    logits = jnp.dot(h_norm, kernel_val)
    return lw_programs.objective_term(logits, cotangent_logits)

  def evaluate(w_norm, w_logits):
    d_norm, d_logits, d_h = jax.grad(loss, argnums=(0, 1, 2))(w_norm, w_logits, h_in)
    d_scale = np.asarray(d_norm["scale"])
    d_kernel = np.asarray(d_logits["kernel"])
    d_h_arr = np.asarray(d_h)

    row_frob = np.linalg.norm(d_kernel, axis=0)
    row_max_abs = np.max(np.abs(d_kernel), axis=0)
    sketch = np.dot(d_kernel.T, sketch_matrix)

    checks = [
        _check("head.d_h_40", d_h_arr, ref.d_h_40, bounds, "dx"),
        _check("decoder_norm.scale", d_scale, ref.norm_scale, bounds, "dW_dense"),
        _check("logits_dense.row_frob", row_frob, ref.logits_row_frob, bounds, "dW_dense"),
        _check("logits_dense.row_max_abs", row_max_abs, ref.logits_row_max_abs, bounds, "dW_dense"),
        _check("logits_dense.sketch", sketch, ref.logits_sketch, bounds, "dW_dense"),
    ]
    for r in sample_rows:
      if r in ref.logits_samples:
        checks.append(_check(f"logits_dense.row_{r}", d_kernel[:, r], ref.logits_samples[r], bounds, "dW_dense"))
    return checks, (d_scale, d_kernel, d_h_arr)

  checks, (clean_scale, _, clean_d_h) = evaluate(norm_weights, logits_weights)
  moved_norm = lw_model.perturb_leaf(norm_weights, "scale", bounds.tol.negative_control_rel_scale, seed=0)
  nc_checks, (nc_scale, _, nc_d_h) = evaluate(moved_norm, logits_weights)
  lw_model.release(moved_norm, keep=norm_weights)
  fired = lw_checks.control_fired(checks, nc_checks)
  if not fired and bounds.tier != lw_config.TIER_CODE_PARITY:
    nc_tol = bounds.tol.code_parity
    cand_vs_clean = [
        lw_checks.check_code_parity("decoder_norm.scale", nc_scale, clean_scale, nc_tol),
        lw_checks.check_code_parity("head.d_h_40", nc_d_h, clean_d_h, nc_tol),
    ]
    fired = any(c.status == lw_checks.STATUS_FAIL for c in cand_vs_clean)
  return {
      "status": lw_checks.worst_status([c["status"] for c in checks]),
      "checks": checks,
      "negative_control": {"NC_L4_head": {"leaf": nc_leaf, "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE}},
  }
