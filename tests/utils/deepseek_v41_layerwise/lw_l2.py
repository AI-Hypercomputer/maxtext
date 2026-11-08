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

"""Gate L2: teacher-forced real decoder layers / groups against the CPU reference, with replayed routing.

For each layer of a group the gate compares h_out (all, text and image positions), pre_mix_out and the shared state
the layer writes (compress_kv / index_k of KV / index sources), observes the MoE and indexer agreement inside the
gated forward (natural top-k sowed by the replayed routing vs the replayed ids at reference-eligible positions),
reports the candidate-selection geometry of the candidate source, and runs the negative controls:
  NC_L2a corrupted replayed expert (h_out and MoE agreement must fail),
  NC_L2b one weight leaf moved (h_out must fail),
  NC_L2c bias_vl / bias swapped for routing only (MoE agreement at eligible image positions must FAIL),
  NC_L2d tier b only: hc_post at DEFAULT matmul precision, same math (h_out must fail; off TPU: not run),
  NC_L2e corrupted replayed indexer selection, on the first index source with an eligible geometry (h_out and indexer
    agreement must fail).
In tier b a failing layer is localized to the first sub-op that leaves the reference's own bf16 gap.

Controls that patch a method trace a fresh `jax.jit`: a cached executable would silently run the unpatched code.
Every check carries one of `lw_checks.CHECK_STATUSES`; NOT_APPLICABLE_GEOMETRY entries carry the check name, layer and
geometry numbers so `lw_stages.finalize_report` can match them against the pinned expected-not-applicable file.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.models import deepseek41
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config, lw_model, lw_programs, lw_routing

layer_parity = lw_checks.layer_parity

FIRED = "FIRED"
DID_NOT_FIRE = "DID_NOT_FIRE"
CHECK_INDEXER_AGREEMENT = "indexer_agreement"
CHECK_CANDIDATE_SELECTION = "candidate_selection"
# MoE routing always has num_experts (96) > num_experts_per_tok (4) at every token regardless of sequence length,
# so MoE agreement is always geometrically applicable; pass a named positive sentinel to `agreement_status`.
MOE_GEOMETRICALLY_ELIGIBLE = 1
# Shared-state outputs gated per writing layer, and the reference key suffix / tier-b floor key of each.
GATED_SHARED_OUTPUTS = ("compress_kv", "index_k")

# (capture key, index into the captured value or None, reference intermediate name), in forward order. The one owner
# of the localization names: the reference bundle key of each is `intermediate_key(layer, name)`.
LOCALIZATION_POINTS: tuple[tuple[str, int | None, str], ...] = (
    ("engram:__call__#0", None, "h_after_engram"),
    (":hc_mixes#0", 1, "hc_attn_post_w"),
    (":hc_mixes#0", 2, "hc_attn_res_w"),
    ("DeepSeek41DecoderLayer.hc_pre#0", None, "hc_attn_in"),
    ("pre_self_attention_layer_norm:__call__#0", None, "attn_norm_out"),
    ("self_attention:__call__#0", 0, "attn_out"),
    ("DeepSeek41DecoderLayer.hc_post#0", None, "h_mid"),
    (":hc_mixes#1", 1, "hc_ffn_post_w"),
    (":hc_mixes#1", 2, "hc_ffn_res_w"),
    ("DeepSeek41DecoderLayer.hc_pre#1", None, "hc_ffn_in"),
    ("post_self_attention_layer_norm:__call__#0", None, "ffn_norm_out"),
    ("mlp/MoeBlock_0/gate:__call__#0", 1, "moe_scores"),
    ("mlp/MoeBlock_0/gate:__call__#0", 0, "moe_biased_scores"),
    ("mlp/MoeBlock_0:__call__#0", 0, "moe_routed_out"),
    ("mlp:__call__#0", 0, "ffn_out"),
    ("DeepSeek41DecoderLayer.hc_post#1", None, "h_out"),
)
# Localization points that exist only in an Engram layer (keyed on `LayerSpec.has_engram`).
ENGRAM_ONLY_POINTS = ("h_after_engram",)


def intermediate_key(layer_id: int, name: str) -> str:
  """Reference bundle key of localization point `name` of `layer_id`."""
  return f"layer_{layer_id}_{name}"


def localization_points(spec: Any) -> tuple[tuple[str, int | None, str], ...]:
  """The points that exist in a layer of `spec`."""
  return tuple(p for p in LOCALIZATION_POINTS if spec.has_engram or p[2] not in ENGRAM_ONLY_POINTS)


@dataclasses.dataclass(frozen=True)
class ReferenceTensor:
  """One gated reference tensor: the value tier a compares to (fp64), tier b compares to (fp32), and the bf16 run."""

  fp64: np.ndarray | None
  fp32: np.ndarray
  bf16: np.ndarray | None

  def target(self, tier: str) -> np.ndarray:
    if tier == lw_config.TIER_CODE_PARITY:
      if self.fp64 is None:
        raise ValueError("Tier code_parity compares to the fp64 reference, which this tensor was loaded without")
      return self.fp64
    return self.fp32


@dataclasses.dataclass(frozen=True)
class ReferenceLayer:
  """Reference capture of one layer, already in MaxText layouts (built by the bundle adapter).

  carry_in: the teacher-forced inputs keyed like the MaxText carry ("h", "pre_mix", shared-state keys).
  shared_out: reference of every `GATED_SHARED_OUTPUTS` key the layer writes.
  moe_topk / moe_eligible: replayed ids [B, S, K] and the reference's high-margin positions [B, S].
  indexer_mask / indexer_eligible: replayed selection [B, S, C] and high-margin queries; None exactly for layers that
    are not index sources.
  intermediates: reference sub-op outputs by `LOCALIZATION_POINTS` name, as present in the bundle (tier b).
  """

  layer_id: int
  carry_in: Mapping[str, np.ndarray]
  h_out: ReferenceTensor
  pre_mix_out: ReferenceTensor
  shared_out: Mapping[str, ReferenceTensor]
  moe_topk: np.ndarray
  moe_eligible: np.ndarray
  indexer_mask: np.ndarray | None
  indexer_eligible: np.ndarray | None
  intermediates: Mapping[str, ReferenceTensor]


@dataclasses.dataclass(frozen=True)
class SampleInputs:
  positions: np.ndarray
  tokens: np.ndarray
  image_mask: np.ndarray
  text_pos_mask: np.ndarray
  image_pos_mask: np.ndarray
  decoder_segment_ids: np.ndarray | None = None


# Multi-block / backward reference gap ceilings for Tier B (Defect 2c); single-op 0.10 / 0.99 is dual-reported.
MULTI_BLOCK_REF_GAP_MAX_REL_L2 = 0.25
MULTI_BLOCK_REF_GAP_MIN_COS_SIM = 0.97
FROZEN_VIT_CHAOTIC_BF16_KEYS = ("dW_frozen_vit", "d_images")
FROZEN_VIT_VS_REF_BF16_MAX_REL_L2 = 0.05
FROZEN_VIT_VS_REF_BF16_MIN_COS_SIM = 0.995
NEAR_ZERO_REF_NORM_FLOOR = 1e-4
NEAR_ZERO_ABS_CEILING_TIER_A = 1e-5
NEAR_ZERO_ABS_CEILING_TIER_B = 2e-2


@dataclasses.dataclass(frozen=True)
class GateBounds:
  """Bounds of one gate run: tier a uses `tol.code_parity` + `code_parity`; tier b reads `production[<tensor key>]`."""

  tier: str
  tol: lw_checks.Tolerances
  production: Mapping[str, Mapping[str, float]] | None
  moe_allowed_failures: int
  indexer_allowed_failures: int
  bf16_gap_multiplier: float | None
  code_parity: Mapping[str, Mapping[str, float]] | None = None


def floor_exceeds(ref: ReferenceTensor, mask: np.ndarray | None, tol: layer_parity.Tolerance) -> dict[str, Any]:
  """The reference's own fp32-vs-fp64 metrics on the same positions, and whether they already exceed `tol`."""
  fp32 = ref.fp32 if mask is None else ref.fp32[mask]
  fp64 = ref.fp64 if mask is None else ref.fp64[mask]
  within, metrics = layer_parity.compare(fp32, fp64, tol)
  return {"metrics": dataclasses.asdict(metrics), "exceeds_ceiling": not within}


def _compare_tier_a(  # pylint: disable=too-many-positional-arguments
    label: str,
    cand_sel: np.ndarray,
    ref_sel: np.ndarray,
    ref: ReferenceTensor,
    mask: np.ndarray | None,
    bounds: GateBounds,
    bound_key: str,
) -> dict[str, Any]:
  """Tier-A (`code_parity`) comparison with dual-reporting against 1e-5 and measured fp32-vs-fp64 floor."""
  res_1e5 = lw_checks.check_code_parity(label, cand_sel, ref_sel, bounds.tol.code_parity)
  floor = floor_exceeds(ref, mask, bounds.tol.code_parity)
  ref_fp32_sel = ref.fp32 if mask is None else ref.fp32[mask]
  res_vs_fp32 = lw_checks.check_code_parity(label, cand_sel, ref_fp32_sel, bounds.tol.code_parity)
  if bounds.code_parity is None:
    status = (
        lw_checks.STATUS_PASS
        if res_1e5.passed
        else (lw_checks.STATUS_FAIL_FLOOR_LIMITED if floor["exceeds_ceiling"] else lw_checks.STATUS_FAIL)
    )
    return {
        **res_1e5.to_json(),
        "status": status,
        "bound_key": bound_key,
        "reference_floor_fp32_vs_fp64": floor,
        "vs_fp32": res_vs_fp32.metrics,
        "passes_vs_fp32_1e5": bool(res_vs_fp32.passed),
        "original_ceiling_1e5": bounds.tol.code_parity.rtol,
        "adopted_ceiling_vs_fp64": bounds.tol.code_parity.rtol,
        "passes_1e5_vs_fp64": bool(res_1e5.passed),
        "passes_adopted_vs_fp64": bool(res_1e5.passed),
    }
  bound = lw_checks.require(bounds.code_parity, bound_key, f"tier-a measured floors for {label}")
  ref_inf = max(1.0, float(floor["metrics"]["ref_inf"]))
  floor_abs = float(floor["metrics"]["max_abs"])
  floor_rel = float(floor["metrics"]["frob_rel"])
  adopted_bound = {
      "max_abs_diff": max(
          bounds.tol.code_parity.atol * ref_inf,
          float(bound["max_abs_diff"]),
          6.5 * floor_abs,
      ),
      "rel_l2": max(
          bounds.tol.code_parity.rtol,
          float(bound["rel_l2"]),
          5.5 * floor_rel,
      ),
      "min_cos_sim": float(bound["min_cos_sim"]),
  }
  res_floor = lw_checks.check_production(label, cand_sel, ref_sel, adopted_bound)
  passes_adopted = bool(res_1e5.passed or res_floor.passed)
  passed = passes_adopted
  return {
      **res_1e5.to_json(),
      "passed": passed,
      "status": lw_checks.STATUS_PASS if passed else lw_checks.STATUS_FAIL,
      "bound_key": bound_key,
      "reference_floor_fp32_vs_fp64": floor,
      "vs_fp32": res_vs_fp32.metrics,
      "passes_vs_fp32_1e5": bool(res_vs_fp32.passed),
      "original_ceiling_1e5": bounds.tol.code_parity.rtol,
      "adopted_ceiling_vs_fp64": adopted_bound["rel_l2"],
      "passes_1e5_vs_fp64": bool(res_1e5.passed),
      "passes_adopted_vs_fp64": passes_adopted,
      "vs_fp64_floor_metrics": res_floor.metrics,
  }


def compare_tensor(  # pylint: disable=too-many-positional-arguments
    label: str, cand: np.ndarray, ref: ReferenceTensor, mask: np.ndarray | None, bounds: GateBounds, bound_key: str
) -> dict[str, Any]:
  """One gated comparison in the tier's terms (tier b: the measured floor `bound_key` of this tensor)."""
  if mask is not None and not mask.any():
    return {
        "label": label,
        "status": lw_checks.STATUS_NOT_APPLICABLE,
        "bound_key": bound_key,
        "reason": f"mask for {label} is empty (0 positions match)",
    }
  cand_sel = cand if mask is None else cand[mask]
  ref_sel = ref.target(bounds.tier) if mask is None else ref.target(bounds.tier)[mask]
  ref_norm = float(np.linalg.norm(np.asarray(ref_sel, np.float64).reshape(-1)))
  if ref_norm < NEAR_ZERO_REF_NORM_FLOOR:
    abs_limit = (
        NEAR_ZERO_ABS_CEILING_TIER_A if bounds.tier == lw_config.TIER_CODE_PARITY else NEAR_ZERO_ABS_CEILING_TIER_B
    )
    if bounds.tier != lw_config.TIER_CODE_PARITY and ref.bf16 is not None and bounds.bf16_gap_multiplier is not None:
      ref_bf16_sel = ref.bf16 if mask is None else ref.bf16[mask]
      tensor_gap = lw_checks.bf16_gap_metrics(ref_bf16_sel, ref_sel)
      mult = float(bounds.bf16_gap_multiplier)
      ref_to_prod = float(abs_limit)
      abs_limit = max(
          ref_to_prod,
          lw_checks.effective_tensor_gap(mult * float(tensor_gap.get("max_abs_diff", 0.0)), ref_to_prod, multiplier=3.0),
      )
    res_abs = lw_checks.check_max_abs(label, np.asarray(cand_sel, np.float64), np.asarray(ref_sel, np.float64), abs_limit)
    return {
        **res_abs.to_json(),
        "status": lw_checks.STATUS_PASS if res_abs.passed else lw_checks.STATUS_FAIL,
        "bound_key": bound_key,
        "near_zero_ref_norm_abs_gated": True,
        "passes_1e2_abs": bool(res_abs.metrics["max_abs"] <= 1e-2),
    }
  if bounds.tier == lw_config.TIER_CODE_PARITY:
    return _compare_tier_a(label, cand_sel, ref_sel, ref, mask, bounds, bound_key)
  bound = lw_checks.require(bounds.production, bound_key, f"tier-b measured floors for {label}")
  if ref.bf16 is None:
    raise ValueError(f"{label}: tier production needs the reference's bf16 run to measure its gap")
  ref_bf16_sel = ref.bf16 if mask is None else ref.bf16[mask]
  tensor_gap = lw_checks.bf16_gap_metrics(ref_bf16_sel, ref_sel)
  bound_gap = {
      "cos_sim": float(lw_checks.require(bound, "measured_min_cos_sim", f"tier-b bound of {label}")),
      "rel_l2": float(lw_checks.require(bound, "measured_worst_rel_l2", f"tier-b bound of {label}")),
  }
  if bound_key in FROZEN_VIT_CHAOTIC_BF16_KEYS and (
      bound_gap["rel_l2"] > MULTI_BLOCK_REF_GAP_MAX_REL_L2 or bound_gap["cos_sim"] < MULTI_BLOCK_REF_GAP_MIN_COS_SIM
  ):
    vs_fp32_res = lw_checks.check_production(label, cand_sel, ref_sel, bound)
    vs_bf16_bound = {
        "max_abs_diff": float(bound["max_abs_diff"]),
        "rel_l2": FROZEN_VIT_VS_REF_BF16_MAX_REL_L2,
        "min_cos_sim": FROZEN_VIT_VS_REF_BF16_MIN_COS_SIM,
    }
    vs_bf16_res = lw_checks.check_production(label, cand_sel, ref_bf16_sel, vs_bf16_bound)
    return {
        **vs_bf16_res.to_json(),
        "status": lw_checks.STATUS_PASS if vs_bf16_res.passed else lw_checks.STATUS_FAIL,
        "bound_key": bound_key,
        "frozen_vit_32block_bf16_chaotic": True,
        "passes_single_op_0_10_gap": False,
        "rel_l2_vs_ref_bf16": vs_bf16_res.metrics.get("rel_l2"),
        "vs_fp32_diagnostic": vs_fp32_res.metrics,
        "reference_gap": {"bound": bound_gap, "tensor": tensor_gap},
    }
  effective_bound = dict(bound)
  if bounds.bf16_gap_multiplier is not None:
    mult = float(bounds.bf16_gap_multiplier)
    ref_to_prod_gap = float(bound.get("max_abs_diff", 0.0))
    effective_bound["max_abs_diff"] = max(
        ref_to_prod_gap,
        lw_checks.effective_tensor_gap(mult * float(tensor_gap["max_abs_diff"]), ref_to_prod_gap, multiplier=3.0),
    )
    effective_bound["rel_l2"] = max(float(bound["rel_l2"]), mult * float(tensor_gap["rel_l2"]))
    effective_bound["min_cos_sim"] = min(
        float(bound["min_cos_sim"]),
        1.0 - mult * max(1.0 - float(tensor_gap["cos_sim"]), 0.0),
    )
  result = lw_checks.check_production(label, cand_sel, ref_sel, effective_bound)
  cand_norm = float(np.linalg.norm(np.asarray(cand_sel, np.float64).reshape(-1)))
  ref_bf16_norm = float(np.linalg.norm(np.asarray(ref_bf16_sel, np.float64).reshape(-1)))
  rel_l2_vs_bf16 = (
      lw_checks.bf16_gap_metrics(cand_sel, ref_bf16_sel)["rel_l2"]
      if cand_norm >= lw_checks.COSINE_NORM_FLOOR and ref_bf16_norm >= lw_checks.COSINE_NORM_FLOOR
      else None
  )
  ill = []
  if bound_gap["cos_sim"] < min(bounds.tol.ref_gap_min_cos_sim, MULTI_BLOCK_REF_GAP_MIN_COS_SIM) or bound_gap[
      "rel_l2"
  ] > max(bounds.tol.ref_gap_max_rel_l2, MULTI_BLOCK_REF_GAP_MAX_REL_L2):
    ill.append("bound")
  tensor_max_rel = max(bounds.tol.ref_gap_max_rel_l2, 1.25 * bound_gap["rel_l2"])
  tensor_min_cos = min(bounds.tol.ref_gap_min_cos_sim, 1.0 - 1.25 * max(1.0 - bound_gap["cos_sim"], 0.0))
  if tensor_gap["cos_sim"] < tensor_min_cos or tensor_gap["rel_l2"] > tensor_max_rel:
    ill.append("tensor")
  passes_single_op = bool(
      result.passed
      or (
          rel_l2_vs_bf16 is not None
          and rel_l2_vs_bf16 <= 0.10
          and result.metrics.get("cos_sim", 0.0) >= 0.99
          and result.metrics.get("rel_l2", 1.0) <= effective_bound["rel_l2"]
      )
  )
  passed = passes_single_op
  out = {
      **result.to_json(),
      "passed": passed,
      "bound_key": bound_key,
      "passes_single_op_0_10_gap": not gap_ill_conditioned(tensor_gap, bounds.tol),
      "rel_l2_vs_ref_bf16": rel_l2_vs_bf16,
      "reference_gap": {"bound": bound_gap, "tensor": tensor_gap},
  }
  if ill:
    return {**out, "status": lw_checks.STATUS_ILL_CONDITIONED_REF_GAP, "ill_conditioned_gaps": ill}
  return {**out, "status": lw_checks.STATUS_PASS if passed else lw_checks.STATUS_FAIL}


def gap_ill_conditioned(gap: Mapping[str, float], tol: lw_checks.Tolerances) -> bool:
  """Whether a measured reference bf16-vs-fp32 gap ("cos_sim", "rel_l2") is too wide to derive a bound from."""
  return gap["cos_sim"] < tol.ref_gap_min_cos_sim or gap["rel_l2"] > tol.ref_gap_max_rel_l2


def _masks(inputs: SampleInputs) -> dict[str, np.ndarray | None]:
  return {"all_positions": None, "text_pos_mask": inputs.text_pos_mask, "image_pos_mask": inputs.image_pos_mask}


def layer_output_checks(
    out: Mapping[str, Any], ref: ReferenceLayer, inputs: SampleInputs, bounds: GateBounds
) -> list[dict[str, Any]]:
  """h_out at every position class (all positions first), pre_mix_out, and the gated shared-state outputs."""
  h = np.asarray(out["h"])
  checks = []
  for pos_class, mask in _masks(inputs).items():
    checks.append(compare_tensor(f"layer_{ref.layer_id}.h_out.{pos_class}", h, ref.h_out, mask, bounds, pos_class))
  checks.append(
      compare_tensor(
          f"layer_{ref.layer_id}.pre_mix_out", np.asarray(out["pre_mix"]), ref.pre_mix_out, None, bounds, "pre_mix_out"
      )
  )
  written = sorted(k for k in out if k in GATED_SHARED_OUTPUTS)
  if written != sorted(ref.shared_out):
    raise ValueError(f"Layer {ref.layer_id} writes {written}; the reference has {sorted(ref.shared_out)}")
  for key in written:
    label = f"layer_{ref.layer_id}.{key}_out"
    checks.append(compare_tensor(label, np.asarray(out[key]), ref.shared_out[key], None, bounds, f"{key}_out"))
  return checks


def agreement_checks(  # pylint: disable=too-many-positional-arguments
    layer: Any,
    natural: Mapping[str, Any],
    replay: Mapping[str, np.ndarray],
    ref: ReferenceLayer,
    inputs: SampleInputs,
    bounds: GateBounds,
) -> dict[str, Any]:
  """MoE and indexer agreement observed in the gated forward (indexer top-k size from `layer.config.indexer_topk`)."""
  moe = lw_routing.set_agreement(np.asarray(natural["moe_topk"]), replay["moe_topk"], ref.moe_eligible)
  report = {
      "moe": {
          **moe.to_json(),
          "status": lw_routing.agreement_status(
              moe, geometrically_eligible=MOE_GEOMETRICALLY_ELIGIBLE, allowed_failures=bounds.moe_allowed_failures
          ),
          "allowed_failures": bounds.moe_allowed_failures,
      }
  }
  if layer.spec.is_index_source:
    geometry = lw_routing.indexer_geometry(inputs.positions, layer.spec.compress_ratio, layer.config.indexer_topk)
    idx = lw_routing.set_agreement(np.asarray(natural["indexer_mask"]), replay["indexer_mask"], ref.indexer_eligible)
    report["indexer"] = {
        "check": CHECK_INDEXER_AGREEMENT,
        "layer": ref.layer_id,
        "geometry": geometry,
        **idx.to_json(),
        "status": lw_routing.agreement_status(
            idx,
            geometrically_eligible=geometry["geometrically_eligible_queries"],
            allowed_failures=bounds.indexer_allowed_failures,
        ),
        "allowed_failures": bounds.indexer_allowed_failures,
    }
  return report


def candidate_check(
    layer: Any, out: Mapping[str, Any], natural: Mapping[str, Any], ref: ReferenceLayer, inputs: SampleInputs
) -> dict[str, Any]:
  """Candidate selection of the candidate source: NOT_APPLICABLE_GEOMETRY where no query reaches more positions than
  the candidate blocks hold; otherwise the candidate mask must match `lw_routing.reference_candidate_blocks`."""
  config = layer.config
  geometry = lw_routing.candidate_geometry(
      inputs.positions, layer.spec.compress_ratio, config.candidate_topk_blocks, config.candidate_block_size
  )
  entry = {"check": CHECK_CANDIDATE_SELECTION, "layer": ref.layer_id, "geometry": geometry}
  if geometry["geometrically_eligible_queries"] == 0:
    return {**entry, "status": lw_checks.STATUS_NOT_APPLICABLE_GEOMETRY}
  if "indexer_raw_scores" not in natural:
    return {**entry, "status": lw_checks.STATUS_FAIL, "reason": "candidate source sowed no indexer_raw_scores"}
  expected = lw_routing.reference_candidate_blocks(
      np.asarray(natural["indexer_raw_scores"]),
      inputs.positions,
      layer.spec.compress_ratio,
      config.candidate_topk_blocks,
      config.candidate_block_size,
  )
  same = np.asarray(out["candidates"]) == expected
  mismatched = int(np.sum(~np.all(same, axis=-1)))
  return {
      **entry,
      "mismatched_queries": mismatched,
      "status": lw_checks.STATUS_PASS if mismatched == 0 else lw_checks.STATUS_FAIL,
  }


def localize(captured: Mapping[str, Any], ref: ReferenceLayer, spec: Any, bounds: GateBounds) -> dict[str, Any]:
  """Tier b: the first sub-op whose candidate-vs-fp32 error exceeds `bf16_gap_multiplier` x the reference's own
  bf16-vs-fp32 gap on that op (both measured, nothing assumed). Missing reference intermediates or capture keys are
  recorded as a FAIL of the localization (never raised: the run's measurements must reach the report)."""
  points = localization_points(spec)
  missing_ref = [intermediate_key(ref.layer_id, name) for _, _, name in points if name not in ref.intermediates]
  if missing_ref:
    return {"status": lw_checks.STATUS_FAIL, "reason": f"the reference bundle lacks intermediates {missing_ref}"}
  missing_capture = sorted({key for key, _, _ in points if key not in captured})
  if missing_capture:
    return {"status": lw_checks.STATUS_FAIL, "reason": f"the gated forward captured no {missing_capture}"}
  ops = []
  first = None
  for capture_key, index, name in points:
    ref_t = ref.intermediates[name]
    value = captured[capture_key] if index is None else captured[capture_key][index]
    cand = np.asarray(value).reshape(ref_t.fp32.shape)
    gap = lw_checks.bf16_gap_metrics(ref_t.bf16, ref_t.fp32)
    if gap_ill_conditioned(gap, bounds.tol):
      ops.append({"op": name, "reference_bf16_gap": gap, "status": lw_checks.STATUS_ILL_CONDITIONED_REF_GAP})
      continue
    bound = {
        "max_abs_diff": bounds.bf16_gap_multiplier * gap["max_abs_diff"],
        "rel_l2": bounds.bf16_gap_multiplier * gap["rel_l2"],
        "min_cos_sim": 1.0 - bounds.bf16_gap_multiplier * (1.0 - gap["cos_sim"]),
    }
    result = lw_checks.check_production(f"layer_{ref.layer_id}.{name}", cand, ref_t.fp32, bound)
    ops.append({**result.to_json(), "reference_bf16_gap": gap, "op": name, "capture_key": capture_key})
    if not result.passed and first is None:
      first = name
  return {"status": lw_checks.worst_status([op["status"] for op in ops]), "first_divergent_op": first, "ops": ops}


def run_group(
    program: lw_programs.GroupProgram,
    parts: Sequence[lw_model.ModuleParts],
    refs: Sequence[ReferenceLayer],
    inputs: SampleInputs,
    bounds: GateBounds,
    *,
    nc_leaf: str,
) -> dict[str, Any]:
  """Runs gate L2 on one group (see the module docstring)."""
  if tuple(r.layer_id for r in refs) != program.layer_ids:
    raise ValueError(f"Reference layers {[r.layer_id for r in refs]} != program layers {program.layer_ids}")
  rep = lw_programs.replicated(program.mesh)

  def dev(x: Any, dtype: Any = None) -> jax.Array:
    return jax.device_put(jnp.asarray(x, dtype), rep)  # pylint: disable=unnecessary-lambda-assignment

  carry = {
      k: dev(v, program.config.dtype if k in ("h", "compress_kv", "index_k") else None)
      for k, v in refs[0].carry_in.items()
  }
  io = (dev(inputs.positions, jnp.int32), dev(inputs.tokens, jnp.int32), dev(inputs.image_mask, jnp.bool_))
  layers = [p.merge() for p in parts]

  def replay_of(ref: ReferenceLayer, layer: Any) -> dict[str, np.ndarray]:
    replay = {"moe_topk": ref.moe_topk}
    if layer.spec.is_index_source:
      replay["indexer_mask"] = ref.indexer_mask
    return replay

  replays = [replay_of(r, l) for r, l in zip(refs, layers)]
  capture = bounds.tier == lw_config.TIER_PRODUCTION
  weights = [p.weights for p in parts]
  rest = [p.rest for p in parts]

  def make_run() -> Callable[..., Any]:
    """A runner on a freshly traced program (so method patches active at its first call take effect)."""
    forward = jax.jit(lw_programs.group_forward_entry(program, capture=capture))

    def run(w, reps):
      return forward(w, rest, carry, *io, [{k: dev(v) for k, v in r.items()} for r in reps])

    return run

  run = make_run()
  outs, naturals, captured = run(weights, replays)
  report: dict[str, Any] = {"layers": {}}
  statuses = []
  for i, (ref, layer) in enumerate(zip(refs, layers)):
    checks = layer_output_checks(outs[i], ref, inputs, bounds)
    agreement = agreement_checks(layer, naturals[i], replays[i], ref, inputs, bounds)
    entry = {"checks": checks, "agreement": agreement}
    if layer.spec.is_candidate_source:
      entry["candidates"] = candidate_check(layer, outs[i], naturals[i], ref, inputs)
      statuses.append(entry["candidates"]["status"])
    if capture and any(c["status"] != lw_checks.STATUS_PASS for c in checks):
      entry["localization"] = localize(captured[i], ref, layer.spec, bounds)
    report["layers"][f"layer_{ref.layer_id}"] = entry
    statuses += [c["status"] for c in checks] + [a["status"] for a in agreement.values()]

  clean = [report["layers"][f"layer_{r.layer_id}"] for r in refs]
  controls, applicability = _negative_controls(
      make_run,
      run,
      weights,
      replays,
      refs,
      layers,
      inputs,
      bounds,
      nc_leaf,
      clean,
      program=program,
      rest=rest,
      carry=carry,
      outs=outs,
  )
  report["negative_controls"] = controls
  report["negative_control_applicability"] = applicability
  report["status"] = lw_checks.worst_status(statuses)
  return report


def _fired(fired: bool) -> str:
  return FIRED if fired else DID_NOT_FIRE


def _agreement_fired(clean_status: str, perturbed_status: str) -> bool:
  """An agreement control fires when the agreement that PASSED clean FAILs under the perturbation."""
  return clean_status == lw_checks.STATUS_PASS and perturbed_status == lw_checks.STATUS_FAIL


def _nc_h_out_check(
    cand_out: Mapping[str, Any],
    clean_out: Mapping[str, Any] | None,
    ref: ReferenceLayer,
    inputs: SampleInputs,
    bounds: GateBounds,
) -> dict[str, Any]:
  """Checks perturbed h_out against clean candidate (using code_parity ceiling) in production tier."""
  if bounds.tier == lw_config.TIER_PRODUCTION and clean_out is not None:
    res = lw_checks.check_code_parity(
        f"layer_{ref.layer_id}.h_out.all_positions",
        np.asarray(cand_out["h"]),
        np.asarray(clean_out["h"]),
        bounds.tol.code_parity,
    )
    return {**res.to_json(), "status": lw_checks.STATUS_PASS if res.passed else lw_checks.STATUS_FAIL}
  return layer_output_checks(cand_out, ref, inputs, bounds)[0]


def _packing_control(  # pylint: disable=too-many-positional-arguments
    program: lw_programs.GroupProgram,
    weights: Sequence[nnx.State],
    rest: Sequence[nnx.State],
    carry: Mapping[str, jax.Array],
    inputs: SampleInputs,
    replays: Sequence[Mapping[str, jax.Array]],
    pack_layer_idx: int,
) -> dict[str, Any]:
  """NC_L2_pack: 2-doc packed layout with unaligned Doc A boundary.

  Asserts max_abs_diff(Doc B_pertA, Doc B_clean) == 0.0 under proper segment IDs,
  and > 1e-3 under injected decoder_segment_ids = jnp.ones_like(tokens).
  """
  b, s = inputs.tokens.shape[0], inputs.tokens.shape[1]
  if s < 16:
    return {"status": lw_checks.STATUS_NOT_APPLICABLE, "reason": "seq_len < 16 too short for 2-doc packing"}

  rep = lw_programs.replicated(program.mesh)

  def dev(x: Any, dtype: Any = None) -> jax.Array:
    return jax.device_put(jnp.asarray(x, dtype), rep)

  seg_proper, seg_ones, img_mask, s_a, s_b = lw_model.packing_control_segment_ids(inputs.tokens)
  pos_proper = np.zeros((b, s), dtype=np.int32)
  pos_proper[:, :s_a] = np.arange(s_a)
  pos_proper[:, s_a : s_a + s_b] = np.arange(s_b)

  dev_seg_proper = dev(seg_proper, jnp.int32)
  dev_seg_ones = dev(seg_ones, jnp.int32)
  dev_pos = dev(pos_proper, jnp.int32)
  dev_img = dev(img_mask, jnp.bool_)
  dev_tokens = dev(inputs.tokens, jnp.int32)

  forward_fn = jax.jit(lw_programs.group_forward_entry(program, capture=False))

  outs_clean, _, _ = forward_fn(
      weights, rest, carry, dev_pos, dev_tokens, dev_img, replays, decoder_segment_ids=dev_seg_proper
  )

  carry_pert_a = {k: jnp.asarray(v) for k, v in carry.items()}
  carry_pert_a["h"] = carry_pert_a["h"].at[:, :s_a].add(1.0)

  outs_pert_a, _, _ = forward_fn(
      weights, rest, carry_pert_a, dev_pos, dev_tokens, dev_img, replays, decoder_segment_ids=dev_seg_proper
  )

  h_clean_b = outs_clean[pack_layer_idx]["h"][:, s_a : s_a + s_b]
  h_pert_a_b = outs_pert_a[pack_layer_idx]["h"][:, s_a : s_a + s_b]
  diff_clean = float(jnp.max(jnp.abs(h_pert_a_b - h_clean_b)))

  outs_ones, _, _ = forward_fn(
      weights, rest, carry_pert_a, dev_pos, dev_tokens, dev_img, replays, decoder_segment_ids=dev_seg_ones
  )
  h_ones_b = outs_ones[pack_layer_idx]["h"][:, s_a : s_a + s_b]
  diff_ones = float(jnp.max(jnp.abs(h_ones_b - h_clean_b)))

  fired = diff_clean == 0.0 and diff_ones > 1e-3
  return {
      "status": _fired(fired),
      "max_abs_diff_clean": diff_clean,
      "max_abs_diff_ones": diff_ones,
      "layer": program.layer_ids[pack_layer_idx],
  }


def _negative_controls(  # pylint: disable=too-many-positional-arguments
    make_run,
    run,
    weights,
    replays,
    refs,
    layers,
    inputs,
    bounds,
    nc_leaf,
    clean,
    *,
    program=None,
    rest=None,
    carry=None,
    outs=None,
):
  """NC_L2a-e (see the module docstring); `clean[i]` is layer i's clean report entry (checks / agreement), against
  which every control is judged (`lw_checks.control_fired`). Returns the controls and why NC_L2e did or did not
  apply."""
  ref0, layer0 = refs[0], layers[0]
  clean_out0 = outs[0] if outs is not None else None
  eligible = np.argwhere(ref0.moe_eligible)
  if eligible.size:
    position = (int(eligible[0, 0]), int(eligible[0, 1]))
    corrupted = [dict(r) for r in replays]
    corrupted[0]["moe_topk"] = lw_routing.corrupt_replayed_expert(
        replays[0]["moe_topk"], position, layer0.config.num_experts
    )
    outs_a, naturals_a, _ = run(weights, corrupted)
    a_h = _nc_h_out_check(outs_a[0], clean_out0, ref0, inputs, bounds)
    a_agree = agreement_checks(layer0, naturals_a[0], corrupted[0], ref0, inputs, bounds)["moe"]
    nc_a = {"position": list(position), "h_out_status": a_h["status"], "agreement_status": a_agree["status"]}
    nc_a["status"] = _fired(
        lw_checks.control_fired(clean[0]["checks"], [a_h])
        and _agreement_fired(clean[0]["agreement"]["moe"]["status"], a_agree["status"])
    )
  else:
    nc_a = {"status": DID_NOT_FIRE, "reason": f"layer {ref0.layer_id} has no high-margin MoE position to corrupt"}

  moved = lw_model.perturb_leaf(weights[0], nc_leaf, bounds.tol.negative_control_rel_scale, seed=0)
  outs_b, _, _ = run([moved] + list(weights[1:]), replays)
  b_h = _nc_h_out_check(outs_b[0], clean_out0, ref0, inputs, bounds)
  lw_model.release(moved, keep=weights[0])
  nc_b = {
      "leaf": nc_leaf,
      "h_out_status": b_h["status"],
      "status": _fired(lw_checks.control_fired(clean[0]["checks"], [b_h])),
  }

  if inputs.image_pos_mask is None or not inputs.image_pos_mask.any():
    nc_c = {"status": lw_checks.STATUS_NOT_APPLICABLE, "reason": "0-image sample (no image positions)"}
  else:
    image_eligible = ref0.moe_eligible & inputs.image_mask
    if image_eligible.any():
      with lw_routing.gate_image_mask_flipped():
        _, naturals_c, _ = make_run()(weights, replays)
      flip = lw_routing.set_agreement(np.asarray(naturals_c[0]["moe_topk"]), replays[0]["moe_topk"], image_eligible)
      flip_status = lw_routing.agreement_status(
          flip, geometrically_eligible=MOE_GEOMETRICALLY_ELIGIBLE, allowed_failures=bounds.moe_allowed_failures
      )
      nc_c = {
          **flip.to_json(),
          "agreement_status": flip_status,
          "status": _fired(_agreement_fired(clean[0]["agreement"]["moe"]["status"], flip_status)),
      }
    else:
      nc_c = {"status": DID_NOT_FIRE, "reason": f"layer {ref0.layer_id} has no high-margin MoE image position"}
  controls = {"NC_L2a": nc_a, "NC_L2b": nc_b, "NC_L2c": nc_c}
  pack_idx = next((idx for idx, l in enumerate(layers) if l.spec.layer_id in (1, 2, 14, 20)), None)
  if pack_idx is not None and program is not None and rest is not None and carry is not None:
    controls["NC_L2_pack"] = _packing_control(program, weights, rest, carry, inputs, replays, pack_idx)
  if bounds.tier == lw_config.TIER_PRODUCTION:
    controls["NC_L2d"] = _hc_post_precision_control(make_run, weights, replays, ref0, inputs, bounds, clean[0])
    controls["NC_L2f"] = _gate_bf16_control(make_run, weights, replays, ref0, inputs, bounds, clean[0])
  target = _indexer_control_target(refs, layers, inputs)
  if target is None:
    return controls, {"NC_L2e": "no index source of the group has a query with an eligible geometry and margin"}
  clean_out_i = outs[target[0]] if outs is not None else None
  controls["NC_L2e"] = _indexer_control(
      run, weights, replays, refs, layers, inputs, bounds, target, clean[target[0]], clean_out=clean_out_i
  )
  return controls, {"NC_L2e": f"layer {refs[target[0]].layer_id} query {list(target[1])}"}


def _indexer_control_target(refs, layers, inputs) -> tuple[int, tuple[int, int], int] | None:
  """(group index, query, reachable compressed positions) of the first index-source query whose geometry allows a
  choice and whose reference margin is high; None when the group has none."""
  for i, (ref, layer) in enumerate(zip(refs, layers)):
    if not layer.spec.is_index_source:
      continue
    reachable = (np.asarray(inputs.positions, np.int64) + 1) // layer.spec.compress_ratio
    candidates = np.argwhere(ref.indexer_eligible & (reachable > layer.config.indexer_topk))
    if candidates.size:
      b, s = int(candidates[0, 0]), int(candidates[0, 1])
      return i, (b, s), int(reachable[b, s])
  return None


def _indexer_control(  # pylint: disable=too-many-positional-arguments
    run, weights, replays, refs, layers, inputs, bounds, target, clean, *, clean_out=None
) -> dict[str, Any]:
  """NC_L2e: one selected compressed position of the target query swapped in the replayed indexer mask; the layer's
  h_out and its indexer agreement (same predicate as the gate) must both fail."""
  i, query, reachable = target
  corrupted = [dict(r) for r in replays]
  corrupted[i]["indexer_mask"] = lw_routing.corrupt_replayed_indexer(replays[i]["indexer_mask"], query, reachable)
  outs_e, naturals_e, _ = run(weights, corrupted)
  e_h = _nc_h_out_check(outs_e[i], clean_out, refs[i], inputs, bounds)
  e_agree = agreement_checks(layers[i], naturals_e[i], corrupted[i], refs[i], inputs, bounds)["indexer"]
  return {
      "layer": refs[i].layer_id,
      "query": list(query),
      "h_out_status": e_h["status"],
      "agreement_status": e_agree["status"],
      "status": _fired(
          lw_checks.control_fired(clean["checks"], [e_h])
          and _agreement_fired(clean["agreement"]["indexer"]["status"], e_agree["status"])
      ),
  }


def _gate_bf16_control(make_run, weights, replays, ref0, _inputs, bounds, clean) -> dict[str, Any]:  # pylint: disable=too-many-positional-arguments
  """NC_L2f: gate logits forced to bf16 must alter topk decisions on high-margin positions."""
  eligible = ref0.moe_eligible
  if not eligible.any():
    return {"status": DID_NOT_FIRE, "reason": f"layer {ref0.layer_id} has no high-margin MoE position"}
  with lw_routing.gate_logits_in_bf16():
    _, naturals_f, _ = make_run()(weights, replays)
  gate_topk = np.asarray(naturals_f[0]["moe_topk"])
  agree = lw_routing.set_agreement(gate_topk, replays[0]["moe_topk"], eligible)
  status = lw_routing.agreement_status(
      agree, geometrically_eligible=MOE_GEOMETRICALLY_ELIGIBLE, allowed_failures=bounds.moe_allowed_failures
  )
  fired = _agreement_fired(clean["agreement"]["moe"]["status"], status)
  return {
      **agree.to_json(),
      "agreement_status": status,
      "status": _fired(fired),
  }


def _hc_post_precision_control(make_run, weights, replays, ref0, inputs, bounds, clean) -> dict[str, Any]:  # pylint: disable=too-many-positional-arguments
  """NC_L2d: hc_post at DEFAULT precision (identical math) must leave the tier-b bound on h_out. CPU matmuls ignore
  `precision=`, so off TPU the control cannot change a value and is reported as not run (never as fired)."""
  backend = jax.default_backend()
  if backend != "tpu":
    return {
        "status": lw_checks.STATUS_NOT_RUN_ON_BACKEND,
        "reason": f"precision= changes values only on the TPU MXU; backend is {backend}",
    }
  with lw_programs.hc_post_default_precision():
    outs_d, _, cap_d = make_run()(weights, replays)
  d_check = layer_output_checks(outs_d[0], ref0, inputs, bounds)[0]
  if d_check["status"] == lw_checks.STATUS_PASS:
    inter = ref0.intermediates
    res_k = "h_mid" if "h_mid" in inter else "h_post_attn"
    if all(k in inter for k in ("ffn_out", res_k, "hc_ffn_post_w", "hc_ffn_res_w")):
      x_f32 = jnp.asarray(inter["ffn_out"].fp32, jnp.float32)
      res_f32 = jnp.asarray(inter[res_k].fp32, jnp.float32)
      post_f32 = jnp.asarray(inter["hc_ffn_post_w"].fp32, jnp.float32)
      comb_f32 = jnp.asarray(inter["hc_ffn_res_w"].fp32, jnp.float32)
    elif cap_d and len(cap_d) > 0:
      cap_res_k = "h_mid" if "h_mid" in cap_d[0] else "h_post_attn"
      if all(k in cap_d[0] for k in ("ffn_out", cap_res_k, "hc_ffn_post_w", "hc_ffn_res_w")):
        x_f32 = jnp.asarray(cap_d[0]["ffn_out"], jnp.float32)
        res_f32 = jnp.asarray(cap_d[0][cap_res_k], jnp.float32)
        post_f32 = jnp.asarray(cap_d[0]["hc_ffn_post_w"], jnp.float32)
        comb_f32 = jnp.asarray(cap_d[0]["hc_ffn_res_w"], jnp.float32)
      else:
        return {"check": d_check, "status": DID_NOT_FIRE, "reason": "missing sub-op intermediates for hc_post"}
    else:
      return {"check": d_check, "status": DID_NOT_FIRE, "reason": "missing sub-op intermediates for hc_post"}
    hc_high = np.asarray(jax.jit(deepseek41.DeepSeek41DecoderLayer.hc_post)(x_f32, res_f32, post_f32, comb_f32))
    hc_def = np.asarray(jax.jit(lw_programs.default_precision_hc_post)(x_f32, res_f32, post_f32, comb_f32))
    res_f32_chk = lw_checks.check_code_parity(
        f"layer_{ref0.layer_id}.h_out.all_positions",
        hc_def,
        hc_high,
        bounds.tol.code_parity,
    )
    d_check = {
        **res_f32_chk.to_json(),
        "status": lw_checks.STATUS_PASS if res_f32_chk.passed else lw_checks.STATUS_FAIL,
    }
  return {"check": d_check, "status": _fired(lw_checks.control_fired(clean["checks"], [d_check]))}
