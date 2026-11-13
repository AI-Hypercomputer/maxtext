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

# pylint: disable=too-many-positional-arguments,unnecessary-lambda

"""Gate L1 (vision tower + aligner + span merge, forward) and gate L4 on the L1 path (VJP of every L1 leaf).

The forward is `lw_programs.l1_forward` (the production encoder, projector, embedder and merge); per-stage outputs
come from `capture_calls`, so every compared tensor is a production intermediate.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.checkpoint_conversion.utils import param_mapping
from tests.utils.deepseek_v41_layerwise import lw_checks, lw_config, lw_l2, lw_model, lw_programs
from tests.utils.parity import negctl

ENCODER = "vision_encoder/DeepSeek41VisionEncoder_0"
PROJECTOR = "vision_encoder/DeepSeek41VisionProjector_0"


@dataclasses.dataclass(frozen=True)
class ReferenceL1:
  """Reference of the L1 forward per stage (patch embed, each ViT block, ViT norm, aligner, merge, h0)."""

  tokens: np.ndarray
  patches: np.ndarray
  image_mask: np.ndarray
  patch_embed: lw_l2.ReferenceTensor
  vit_blocks: tuple[lw_l2.ReferenceTensor, ...]
  vit_norm: lw_l2.ReferenceTensor
  aligner: lw_l2.ReferenceTensor
  merged: lw_l2.ReferenceTensor
  h0: lw_l2.ReferenceTensor
  pre_mix0: lw_l2.ReferenceTensor


def _stage_checks(
    outputs: Mapping[str, Any], captured: Mapping[str, Any], ref: ReferenceL1, bounds: lw_l2.GateBounds
) -> list[dict[str, Any]]:
  """Every L1 stage output (captured in the gated forward) against its reference, and the image mask."""
  n_img, patches = ref.patch_embed.fp32.shape[:2]
  checks = [
      lw_l2.compare_tensor(
          "patch_embed", np.asarray(captured[f"{ENCODER}:embed_patches#0"]), ref.patch_embed, None, bounds, "patch_embed"
      )
  ]
  for i, block in enumerate(ref.vit_blocks):
    out = np.asarray(captured[f"{ENCODER}/blocks_{i}:__call__#0"])
    checks.append(lw_l2.compare_tensor(f"vit_block_{i}", out, block, None, bounds, "vit_block"))
  vit = np.asarray(captured[f"{ENCODER}:__call__#0"]).reshape(n_img, patches, -1)
  checks.append(lw_l2.compare_tensor("vit_norm", vit, ref.vit_norm, None, bounds, "vit_norm"))
  aligner = np.asarray(captured[f"{PROJECTOR}:align#0"]).reshape(ref.aligner.fp32.shape)
  checks.append(lw_l2.compare_tensor("aligner", aligner, ref.aligner, None, bounds, "aligner"))
  merged = np.asarray(outputs["merged"])
  text = ~ref.image_mask
  checks.append(lw_l2.compare_tensor("merged.all_positions", merged, ref.merged, None, bounds, "span_assembly"))
  checks.append(lw_l2.compare_tensor("merged.text_positions", merged, ref.merged, text, bounds, "embed_tokens"))
  checks.append(
      lw_l2.compare_tensor(
          "merged.image_positions", merged, ref.merged, ref.image_mask, bounds, "span_assembly_image_pos"
      )
  )
  # h0 / pre_mix0 are exact functions of the merged span (deepseek41.initial_carry): checked structurally on both
  # sides, so the merged comparison above is the numeric gate for h0 and no bound is borrowed.
  h0 = np.asarray(outputs["h0"])
  h0_ok = bool(np.array_equal(h0, np.broadcast_to(merged[:, :, None, :], h0.shape))) and bool(
      np.array_equal(ref.h0.fp32, np.broadcast_to(ref.merged.fp32[:, :, None, :], ref.h0.fp32.shape))
  )
  checks.append(_exact_check("h0_replicates_merged", h0_ok))
  checks.append(_exact_check("pre_mix0", bool(np.array_equal(np.asarray(outputs["pre_mix0"]), ref.pre_mix0.fp32))))
  checks.append(_exact_check("image_mask", bool(np.array_equal(np.asarray(outputs["image_mask"]), ref.image_mask))))
  return checks


def _exact_check(label: str, equal: bool) -> dict[str, Any]:
  return {
      "label": label,
      "status": lw_checks.STATUS_PASS if equal else lw_checks.STATUS_FAIL,
      "metrics": {"equal": equal},
      "bound": {"equal": True},
  }


def run_forward(
    graphdef: Any,
    parts: lw_model.ModuleParts,
    config: Any,
    mesh: Any,
    ref: ReferenceL1,
    bounds: lw_l2.GateBounds,
    *,
    nc_leaf: str,
) -> dict[str, Any]:
  """Gate L1 forward with NC_L1a (`nc_leaf` moved) and NC_L1b (image span shifted by one position)."""
  rep = lw_programs.replicated(mesh)
  forward = jax.jit(lw_programs.l1_forward_entry(graphdef, config, capture=True))
  tokens = jax.device_put(jnp.asarray(ref.tokens, jnp.int32), rep)
  patches = jax.device_put(jnp.asarray(ref.patches, jnp.float32), rep)
  outputs, captured = forward(parts.weights, parts.rest, tokens, patches)
  checks = _stage_checks(outputs, captured, ref, bounds)

  moved = lw_model.perturb_leaf(parts.weights, nc_leaf, bounds.tol.negative_control_rel_scale, seed=0)
  outputs_a, captured_a = forward(moved, parts.rest, tokens, patches)
  nc_a = _stage_checks(outputs_a, captured_a, ref, bounds)
  lw_model.release(moved, keep=parts.weights)
  fired_a = lw_checks.control_fired(checks, nc_a)
  if not fired_a and bounds.tier != lw_config.TIER_CODE_PARITY:
    cand_vs_clean_a = [
        lw_checks.check_code_parity(
            "outputs_a.merged", np.asarray(outputs_a["merged"]), np.asarray(outputs["merged"]), bounds.tol.code_parity
        )
    ]
    fired_a = any(c.status == lw_checks.STATUS_FAIL for c in cand_vs_clean_a)

  shifted = jax.device_put(jnp.roll(tokens, 1, axis=1), rep)
  outputs_b, _ = forward(parts.weights, parts.rest, shifted, patches)
  nc_b = lw_l2.compare_tensor(
      "merged.all_positions", np.asarray(outputs_b["merged"]), ref.merged, None, bounds, "span_assembly"
  )
  fired_b = lw_checks.control_fired(checks, [nc_b])
  if not fired_b and bounds.tier != lw_config.TIER_CODE_PARITY:
    cand_vs_clean_b = [
        lw_checks.check_code_parity(
            "outputs_b.merged", np.asarray(outputs_b["merged"]), np.asarray(outputs["merged"]), bounds.tol.code_parity
        )
    ]
    fired_b = any(c.status == lw_checks.STATUS_FAIL for c in cand_vs_clean_b)

  return {
      "status": lw_checks.worst_status([c["status"] for c in checks]),
      "checks": checks,
      "negative_controls": {
          "NC_L1a": {
              "leaf": nc_leaf,
              "status": lw_l2.FIRED if fired_a else lw_l2.DID_NOT_FIRE,
          },
          "NC_L1b": {
              "shift": 1,
              "status": lw_l2.FIRED if fired_b else lw_l2.DID_NOT_FIRE,
              "check": nc_b,
          },
      },
  }


@dataclasses.dataclass(frozen=True)
class ReferenceL1VJP:
  tokens: np.ndarray
  patches: np.ndarray
  image_mask: np.ndarray
  cotangent_merged: np.ndarray
  d_patches: lw_l2.ReferenceTensor
  d_text_embeddings: lw_l2.ReferenceTensor
  leaves: Mapping[str, lw_l2.ReferenceTensor]


def l1_hf_mapping(config: Any) -> tuple[dict[str, str], dict[str, Any]]:
  """L1 leaf path ("vision_encoder/DeepSeek41VisionEncoder_0/patch_embed/kernel") -> HF name / saving hook."""
  quant = {"quantization_config": False}
  param_map = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(quant, config, scan_layers=False)
  hooks = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(quant, config, scan_layers=False, saving_to_hf=True)
  names, hook_by_path = {}, {}
  for mt_key, hf in param_map.items():
    if mt_key.startswith(("params-vision_encoder-", "params-token_embedder-")):
      path = mt_key[len("params-") :].replace("-", "/")
      names[path] = hf
      hook_by_path[path] = hooks[mt_key] if mt_key in hooks else None
  return names, hook_by_path


def embedding_rows_reference(
    tokens: np.ndarray, d_text: lw_l2.ReferenceTensor, rows: np.ndarray, table_shape
) -> lw_l2.ReferenceTensor:
  """The embedding-table gradient rows implied by the reference's d(text embeddings): a scatter-add over tokens."""

  def scatter(d: np.ndarray | None) -> np.ndarray | None:
    if d is None:
      return None
    table = np.zeros(table_shape, np.float64)
    np.add.at(table, tokens.reshape(-1), d.reshape(-1, table_shape[1]).astype(np.float64))
    return table[rows]

  return lw_l2.ReferenceTensor(fp64=scatter(d_text.fp64), fp32=scatter(d_text.fp32), bf16=scatter(d_text.bf16))


def run_vjp(
    graphdef: Any,
    parts: lw_model.ModuleParts,
    config: Any,
    mesh: Any,
    ref: ReferenceL1VJP,
    bounds: lw_l2.GateBounds,
    *,
    nc_leaf: str,
) -> dict[str, Any]:
  """Gate L4 on L1: every vision / aligner / delimiter leaf, d_patches, and the embedding-table gradient."""
  rep = lw_programs.replicated(mesh)
  vjp = jax.jit(lw_programs.l1_vjp_entry(graphdef, config))
  tokens = jax.device_put(jnp.asarray(ref.tokens, jnp.int32), rep)
  patches = jax.device_put(jnp.asarray(ref.patches, jnp.float32), rep)
  cot = jax.device_put(jnp.asarray(ref.cotangent_merged, jnp.float32), rep)
  names, hooks = l1_hf_mapping(config)

  def evaluate(weights: nnx.State) -> list[dict[str, Any]]:
    d_w, d_x = vjp(weights, parts.rest, tokens, patches, cot)
    pure = nnx.to_pure_dict(d_w)
    flat = {negctl.leaf_path(p): np.asarray(jax.device_get(v)) for p, v in jax.tree_util.tree_flatten_with_path(pure)[0]}
    d_patches = np.asarray(jax.device_get(d_x))
    lw_model.release([d_w, d_x], keep=weights)
    missing = sorted(set(flat) - set(names))
    if missing:
      raise ValueError(f"L1 leaves with no HF mapping: {missing}")
    checks = [
        lw_l2.compare_tensor(
            "d_patches", d_patches.reshape(ref.d_patches.fp32.shape), ref.d_patches, None, bounds, "d_images"
        )
    ]
    covered = set()
    for path, grad in sorted(flat.items()):
      hf = names[path]
      if path == "token_embedder/embedding":
        rows = np.unique(ref.tokens)
        derived = embedding_rows_reference(ref.tokens, ref.d_text_embeddings, rows, grad.shape)
        checks.append(lw_l2.compare_tensor("embedding.rows", grad[rows], derived, None, bounds, "dW_embedding_rows"))
        continue
      ref_t = lw_checks.require(ref.leaves, hf, "reference L1 gradients")
      covered.add(hf)
      hf_grad = grad if hooks[path] is None else np.asarray(hooks[path](grad, ref_t.fp32.shape))
      active_table = bounds.production if bounds.tier == "production" else bounds.code_parity
      desired_key = "dW_frozen_vit" if hf.startswith("vision.") else "dW_aligner"
      bkey = desired_key if active_table is not None and desired_key in active_table else "dW_vision_aligner"
      checks.append(lw_l2.compare_tensor(hf, hf_grad, ref_t, None, bounds, bkey))
    extra = sorted(set(ref.leaves) - covered)
    if extra:
      raise ValueError(f"Reference L1 gradients with no candidate leaf: {extra}")
    return checks, flat

  checks, clean_flat = evaluate(parts.weights)
  moved = lw_model.perturb_leaf(parts.weights, nc_leaf, bounds.tol.negative_control_rel_scale, seed=0)
  nc, nc_flat = evaluate(moved)
  lw_model.release(moved, keep=parts.weights)
  fired = lw_checks.control_fired(checks, nc)
  if not fired and bounds.tier != lw_config.TIER_CODE_PARITY:
    nc_tol = bounds.tol.code_parity
    cand_vs_clean = []
    if nc_leaf in clean_flat and nc_leaf in nc_flat:
      cand_vs_clean.append(lw_checks.check_code_parity(f"L1.{nc_leaf}", nc_flat[nc_leaf], clean_flat[nc_leaf], nc_tol))
    else:
      for k in clean_flat:
        if "aligner" in k:
          cand_vs_clean.append(lw_checks.check_code_parity(f"L1.{k}", nc_flat[k], clean_flat[k], nc_tol))
          break
    fired = any(c.status == lw_checks.STATUS_FAIL for c in cand_vs_clean)
  return {
      "status": lw_checks.worst_status([c["status"] for c in checks]),
      "checks": checks,
      "leaf_count": len(ref.leaves),
      "negative_control": {
          "NC_L4": {
              "leaf": nc_leaf,
              "status": lw_l2.FIRED if fired else lw_l2.DID_NOT_FIRE,
          }
      },
  }
