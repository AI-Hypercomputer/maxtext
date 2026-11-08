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

"""Chained 40-layer forward, hidden-norm consistency, and Step-1 R4 parity protocol
for DeepSeek-V4.1 multimodal on TPU v5p-8.

Includes:
  1. Dual-mode `lw_chain_forward`:
     - Golden Parity Forward Mode (R1–R3): fp32 compute, highest precision, ds41_activation_quant=False.
     - Production bf16 Forward Mode (R4): bf16 compute, default precision, ds41_activation_quant=True.
     Supports both multimodal samples (L1–L4, M1–M8) and 0-image text samples (T0–T3).
  2. `lw_chain_hidden_norm_consistency`:
     - Chained 40-layer forward pass evaluating scaled final hidden norm consistency metric and MoE load balance loss.
  3. `lw_chain_step1_r4_parity`:
     - Controller-approved Option A (design:155) chained Step-1 training verification against pinned TPU7x R4 (20c1af6a).
     - Full 40-layer streaming forward + backward VJP (1 layer in HBM at a time), evaluating lm_loss, moe_lb_loss,
       and raw_grad_norm under the measured noise floor gate and mandatory negative control (NC_L5_step1_perturb).
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import pathlib
import pickle
import resource
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax import nnx
from flax.core.spmd import logical_axis_rules


def _ensure_pure_dict(x: Any) -> Any:
  if isinstance(x, nnx.State):
    return nnx.to_pure_dict(x)
  return x


from maxtext.layers import engram_deepseek41, linears, normalizations
from maxtext.models import deepseek41
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import maxtext_utils
from tests.utils.deepseek_v41_layerwise import lw_bundle, lw_checks, lw_config, lw_l2, lw_model, lw_programs, lw_stages

_DEFAULT_TOLERANCES = lw_checks.load_tolerances(lw_checks.TOLERANCES_PATH)
_STEP1_PARITY_SPEC = _DEFAULT_TOLERANCES.step1_r4_parity or {}

# Pinned TPU7x R4 (ds41mm-w1-r4-opt1-nap, 20c1af6a, r4_metrics.jsonl:1) reference metrics.
TPU7X_R4_STEP1_REF = dict(_STEP1_PARITY_SPEC["tpu7x_ref"])

# Pre-registered hardware tolerance boundaries tau_hw matching design:155.
TAU_HW = dict(_STEP1_PARITY_SPEC["tau_hw"])
MEASURED_FLOOR_MULTIPLIER = float(_STEP1_PARITY_SPEC["measured_floor_multiplier"])
DEFAULT_NC_PERTURB_LEAF = str(_STEP1_PARITY_SPEC["negative_control_leaf"])


@dataclasses.dataclass(frozen=True)
class ChainForwardReport:
  mode: str
  sample_id: str
  status: str
  metrics: dict[str, Any]
  elapsed_seconds: float


class HeadModules(nnx.Module):
  """Decoder RMSNorm and DenseGeneral lm_head matching nnx_decoders.py:605-635."""

  def __init__(self, config: Any, mesh: Any, *, rngs: nnx.Rngs):
    del mesh
    self.decoder_norm = normalizations.RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.logits_dense = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=config.vocab_size,
        weight_dtype=config.weight_dtype,
        dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
        kernel_axes=("embed", "vocab"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:  # pylint: disable=arguments-differ
    if x.ndim == 4:
      x = jnp.sum(x, axis=2)
    normed = self.decoder_norm(x)
    return self.logits_dense(normed)


def load_head(config: Any, mesh: Any, hf_dir: pathlib.Path) -> lw_model.ModuleParts:
  """Loads the real decoder_norm and logits_dense weights from the HF checkpoint."""

  def build():
    return HeadModules(config, mesh, rngs=nnx.Rngs(params=0))

  parts = lw_model.abstract_parts(build, config, mesh)
  target = {
      "decoder": {
          "decoder_norm": nnx.to_pure_dict(parts.weights["decoder_norm"]),
          "logits_dense": nnx.to_pure_dict(parts.weights["logits_dense"]),
      }
  }
  loaded = lw_model.load_weights(target, config, hf_dir)["decoder"]
  filled = lw_model.fill_weights(parts, {"decoder_norm": loaded["decoder_norm"], "logits_dense": loaded["logits_dense"]})
  return lw_model.ModuleParts(filled.graphdef, filled.weights, lw_model.materialize_rest(build, config, mesh))


def generate_exact_step0_synthetic_batch() -> dict[str, np.ndarray]:
  """Generates the exact 16-row step-0 synthetic batch matching synthetic_data_processing.py:80-155."""
  global_batch_size_to_load = 128
  max_target_length = 512
  vocab_size = 129280
  image_token_id = 129264
  num_images = 1
  n_vit_h, n_vit_w = 39, 39
  ds = 3
  span_len = ((n_vit_h + (-n_vit_h) % ds) // ds) * (((n_vit_w + (-n_vit_w) % ds) // ds) + 1) + 2
  total_span_len = num_images * span_len

  tokens = jax.random.randint(
      jax.random.PRNGKey(0),
      (global_batch_size_to_load, max_target_length + 1),
      0,
      vocab_size,
      dtype=jnp.int32,
  )
  sequence_positions = jnp.arange(0, max_target_length + 1, dtype=jnp.int32).reshape(1, -1)
  batch_positions = jnp.broadcast_to(sequence_positions, (global_batch_size_to_load, max_target_length + 1))

  tokens = jnp.where(tokens == image_token_id, (image_token_id + 1) % vocab_size, tokens)
  tokens = tokens.at[:, 8 : 8 + total_span_len].set(image_token_id)
  seg_shape = (global_batch_size_to_load, max_target_length)
  segmentation = jnp.ones(seg_shape, dtype=jnp.int32)
  targets_segmentation = jnp.ones(seg_shape, dtype=jnp.int32).at[:, 7 : 7 + total_span_len].set(0)

  patch_dim = 3 * (14**2)
  images = jax.random.normal(
      jax.random.PRNGKey(1),
      (global_batch_size_to_load, num_images, n_vit_h * n_vit_w, patch_dim),
      dtype=jnp.float32,
  )

  return {
      "inputs": np.asarray(tokens[:16, :max_target_length]),
      "targets": np.asarray(tokens[:16, 1:]),
      "inputs_position": np.asarray(batch_positions[:16, :max_target_length]),
      "inputs_segmentation": np.asarray(segmentation[:16]),
      "targets_segmentation": np.asarray(targets_segmentation[:16]),
      "images": np.asarray(images[:16]),
  }


def lw_chain_forward(
    config: Any,
    mesh: Any,
    hf_dir: pathlib.Path,
    bundle_index: lw_bundle.BundleIndex,
    sample_id: str,
    *,
    mode: str = "golden_parity",
    floors: lw_bundle.NoiseFloors | None = None,
    out_state: dict[str, np.ndarray] | None = None,
) -> ChainForwardReport:
  """Runs chained 40-layer forward in golden parity (R1-R3) or production bf16 (R4) mode."""
  if mode not in ("golden_parity", "production_bf16"):
    raise ValueError(f"Unknown forward mode {mode!r}; expected 'golden_parity' or 'production_bf16'")

  t0 = time.perf_counter()
  sample_geom = bundle_index.sample(sample_id)
  rep = lw_programs.replicated(mesh)

  # Unit 0: L1 Modules
  l1_parts = lw_stages.load_l1(config, mesh, hf_dir)
  l1_mod = lw_model.merge_fresh(l1_parts.graphdef, l1_parts.weights, l1_parts.rest)

  if sample_geom.num_images > 0:
    l1_ref = bundle_index.open(lw_bundle.l1_path(sample_id))
    tokens = jax.device_put(jnp.asarray(l1_ref.get("tokens"), jnp.int32), rep)
    pixels = jax.device_put(jnp.asarray(l1_ref.get("pixel_values"), jnp.float32), rep)

    img_emb, _ = l1_mod.vision_encoder(input_images=pixels, deterministic=True)
    img_mask = jax.device_put(jnp.asarray(l1_ref.get("image_token_mask"), jnp.bool_), rep)
    text_emb = l1_mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
    # Merge
    merged = mm_utils.merge_mm_embeddings(
        text_embeddings=text_emb, multimodal_embeddings=img_emb, mask=img_mask, token_masks=None
    )
  else:
    l2_0 = bundle_index.open(lw_bundle.l2_path(sample_id, 0))
    tokens = jax.device_put(jnp.asarray(l2_0.get("tokens"), jnp.int32), rep)
    img_mask = jax.device_put(jnp.zeros_like(tokens, dtype=jnp.bool_), rep)
    merged = l1_mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)

  carry = deepseek41.initial_carry(merged, config.mhc_expansion_rate)
  lw_model.release(l1_parts.weights)
  del l1_mod, l1_parts
  gc.collect()

  positions = jax.device_put(jnp.arange(sample_geom.seq_len, dtype=jnp.int32)[None, :], rep)

  # Units 1-40: Decoder Layers
  specs = deepseek41.build_layer_specs(config)
  if not specs:
    raise ValueError("lw_chain_forward requires at least 1 decoder layer in build_layer_specs(config)")
  state = dict(carry)
  for spec in specs:
    i = spec.layer_id
    if spec.has_engram:
      active, controls = lw_bundle.engram_rows(bundle_index, lw_bundle.l2_path(sample_id, i), f"layer_{i}_")
      engram_rows = lw_stages.EngramRows(
          active, controls, np.asarray(jax.device_get(tokens)), np.asarray(jax.device_get(img_mask))
      )
    else:
      engram_rows = None
    layer_parts, plan = lw_stages.load_layer(config, mesh, i, hf_dir, engram_rows)
    layer = lw_model.merge_fresh(layer_parts.graphdef, layer_parts.weights, layer_parts.rest)
    if plan is not None:
      lw_model.install_sliced_hasher(layer, plan)

    # Run layer
    state = layer(state, positions, tokens, image_mask=img_mask)
    lw_model.release(layer_parts.weights)
    del layer, layer_parts
    gc.collect()

  # Unit 41: Final Decoder Layer Output Gate
  h_state = state["h"]
  h_state_np = np.asarray(jax.device_get(h_state))
  h_final = jnp.sum(h_state, axis=2) if h_state.ndim == 4 else h_state
  h_final_np = np.asarray(jax.device_get(h_final))
  if out_state is not None:
    out_state["h_final"] = h_final_np
    out_state["h_state"] = h_state_np

  if not (np.all(np.isfinite(h_final_np)) and np.all(np.isfinite(h_state_np))):
    return ChainForwardReport(
        mode=mode,
        sample_id=sample_id,
        status=lw_checks.STATUS_FAIL,
        metrics={"final_h_norm": float("nan"), "finite": False},
        elapsed_seconds=time.perf_counter() - t0,
    )

  final_h_norm = float(np.linalg.norm(h_final_np.astype(np.float64)))
  last_layer_id = specs[-1].layer_id
  tier = lw_config.TIER_CODE_PARITY if mode == "golden_parity" else lw_config.TIER_PRODUCTION
  l2_last = bundle_index.open(lw_bundle.l2_path(sample_id, last_layer_id))
  ref_key = f"layer_{last_layer_id}_h_out"
  ref_h = l2_last.tensor(ref_key, tier)

  active_floors = floors
  if active_floors is None and isinstance(getattr(bundle_index, "root", None), pathlib.Path):
    floors_file = bundle_index.root / "measured_noise_floors.json"
    if tier == lw_config.TIER_PRODUCTION or floors_file.is_file():
      active_floors = lw_bundle.load_noise_floors(bundle_index.root)

  if tier == lw_config.TIER_PRODUCTION:
    if active_floors is None:
      raise FileNotFoundError("lw_chain_forward production_bf16 mode requires measured_noise_floors.json")
    bounds = lw_l2.GateBounds(
        tier=tier,
        tol=_DEFAULT_TOLERANCES,
        production=active_floors.production_bounds("L2_decoder_groups"),
        moe_allowed_failures=active_floors.allowed_moe_failures(),
        indexer_allowed_failures=active_floors.allowed_indexer_failures(),
        bf16_gap_multiplier=active_floors.bf16_gap_multiplier(),
    )
  else:
    has_cp = active_floors is not None and "code_parity" in active_floors.data.get("recommended_tolerances", {}).get(
        "L2_decoder_groups", {}
    )
    bounds = lw_l2.GateBounds(
        tier=tier,
        tol=_DEFAULT_TOLERANCES,
        production=None,
        moe_allowed_failures=active_floors.allowed_moe_failures() if active_floors is not None else 0,
        indexer_allowed_failures=active_floors.allowed_indexer_failures() if active_floors is not None else 0,
        bf16_gap_multiplier=None,
        code_parity=active_floors.code_parity_bounds("L2_decoder_groups") if has_cp else None,
    )

  cand_for_cmp = h_state_np if h_state_np.shape == ref_h.fp32.shape else h_final_np
  parity_res = lw_l2.compare_tensor(f"layer_{last_layer_id}.h_out", cand_for_cmp, ref_h, None, bounds, "all_positions")
  return ChainForwardReport(
      mode=mode,
      sample_id=sample_id,
      status=parity_res["status"],
      metrics={
          "final_h_norm": final_h_norm,
          "finite": True,
          "last_layer_id": last_layer_id,
          "parity_check": parity_res,
      },
      elapsed_seconds=time.perf_counter() - t0,
  )


def lw_chain_hidden_norm_consistency(
    config: Any,
    mesh: Any,
    hf_dir: pathlib.Path,
    pinned_batch_path: pathlib.Path,
    *,
    out_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
  """Runs chained 40-layer forward sweep evaluating scaled final hidden norm consistency metric."""
  del out_dir
  if config.gradient_accumulation_steps != 1:
    raise ValueError(f"gradient_accumulation_steps must be 1, got {config.gradient_accumulation_steps}")

  with np.load(pinned_batch_path) as batch_npz:
    batch_np = dict(batch_npz)
  rep = lw_programs.replicated(mesh)

  def dev(x: Any, dtype: Any = None) -> jax.Array:
    return jax.device_put(jnp.asarray(x, dtype), rep)

  tokens = dev(batch_np["inputs"], jnp.int32)
  positions = dev(batch_np["inputs_position"], jnp.int32)
  segment_ids = dev(batch_np["inputs_segmentation"], jnp.int32) if "inputs_segmentation" in batch_np else None

  # Unit 0: L1 Projector + Token Embedder
  l1_parts = lw_stages.load_l1(config, mesh, hf_dir)
  l1_mod = lw_model.merge_fresh(l1_parts.graphdef, l1_parts.weights, l1_parts.rest)

  img_key = (
      "images"
      if ("images" in batch_np and batch_np["images"] is not None)
      else ("pixel_values" if ("pixel_values" in batch_np and batch_np["pixel_values"] is not None) else None)
  )
  if img_key is not None:
    pixels = dev(batch_np[img_key], jnp.float32)
    if "image_token_mask" in batch_np and batch_np["image_token_mask"] is not None:
      img_mask = dev(batch_np["image_token_mask"], jnp.bool_)
    elif "image_mask" in batch_np and batch_np["image_mask"] is not None:
      img_mask = dev(batch_np["image_mask"], jnp.bool_)
    else:
      img_mask = dev(tokens == int(config.ds41_image_token_id), jnp.bool_)
    img_emb, _ = l1_mod.vision_encoder(input_images=pixels, deterministic=True)
    text_emb = l1_mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
    merged = mm_utils.merge_mm_embeddings(
        text_embeddings=text_emb, multimodal_embeddings=img_emb, mask=img_mask, token_masks=None
    )
  else:
    img_mask = dev(jnp.zeros_like(tokens, dtype=jnp.bool_))
    merged = l1_mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)

  carry = deepseek41.initial_carry(merged, config.mhc_expansion_rate)
  lw_model.release(l1_parts.weights)
  del l1_mod, l1_parts
  gc.collect()

  specs = deepseek41.build_layer_specs(config)
  state = dict(carry)
  aux_lb_losses = []
  per_layer_h_norms: list[float] = []
  np_tokens = np.asarray(jax.device_get(tokens))
  np_mask = np.asarray(jax.device_get(img_mask))
  np_seg = np.asarray(jax.device_get(segment_ids)) if segment_ids is not None else None

  for spec in specs:
    i = spec.layer_id
    if spec.has_engram:
      token_map, compressed_vocab_size = deepseek41.compressed_token_map(config)
      hasher = engram_deepseek41.make_hasher(config, i, token_map, compressed_vocab_size)
      hash_ids = np.asarray(hasher(input_ids=np_tokens, decoder_segment_ids=np_seg, image_mask=np_mask))
      active = np.unique(hash_ids[hash_ids >= 0])
      controls = np.array([], dtype=np.int64)
      engram_rows = lw_stages.EngramRows(
          active=active,
          controls=controls,
          tokens=np_tokens,
          image_mask=np_mask,
          include_packing_rows=False,
          decoder_segment_ids=np_seg,
      )
    else:
      engram_rows = None

    layer_parts, plan = lw_stages.load_layer(config, mesh, i, hf_dir, engram_rows)
    layer = lw_model.merge_fresh(layer_parts.graphdef, layer_parts.weights, layer_parts.rest)
    if plan is not None:
      lw_model.install_sliced_hasher(layer, plan)

    state = layer(state, positions, tokens, decoder_segment_ids=segment_ids, image_mask=img_mask)
    layer_h_np = np.asarray(jax.device_get(state["h"]))
    if np.all(np.isfinite(layer_h_np)):
      per_layer_h_norms.append(float(np.linalg.norm(layer_h_np.astype(np.float64))))
    else:
      per_layer_h_norms.append(float("nan"))

    if isinstance(layer, nnx.Module):
      inter = nnx.pop(layer, nnx.Intermediate)
      inter_dict = inter.to_pure_dict()
      lb_list = maxtext_utils.collect_intermediates_by_suffix(inter_dict, "moe_lb_loss")
      if lb_list:
        lb_loss_k = jnp.mean(jnp.concatenate(lb_list))
      elif getattr(config, "num_experts", 1) > 1 and getattr(config, "load_balance_loss_weight", 0.0) > 0.0:
        raise RuntimeError(
            f"Missing moe_lb_loss intermediate in layer {i} "
            f"when load_balance_loss_weight={config.load_balance_loss_weight} > 0"
        )
      else:
        lb_loss_k = jnp.zeros((), dtype=jnp.float32)
    else:
      lb_loss_k = jnp.zeros((), dtype=jnp.float32)
    aux_lb_losses.append(lb_loss_k)

    lw_model.release(layer_parts.weights)
    del layer, layer_parts
    gc.collect()

  total_moe_lb_loss = jnp.mean(jnp.stack(aux_lb_losses)) if aux_lb_losses else jnp.asarray(0.0)
  total_moe_lb_val = float(jax.device_get(total_moe_lb_loss))

  h_final = state["h"]
  if h_final.ndim == 4:
    h_final = jnp.sum(h_final, axis=2)
  h_final_np = np.asarray(jax.device_get(h_final))

  finite_ok = (
      bool(np.all(np.isfinite(h_final_np)))
      and bool(np.isfinite(total_moe_lb_val))
      and all(bool(np.isfinite(v)) for v in per_layer_h_norms)
  )
  final_hidden_norm_scaled = float(np.linalg.norm(h_final_np.astype(np.float64))) * 0.001 if finite_ok else float("nan")
  positive_ok = finite_ok and final_hidden_norm_scaled > 0.0 and all(v > 0.0 for v in per_layer_h_norms)
  total_consistency_objective = final_hidden_norm_scaled + total_moe_lb_val if finite_ok else float("nan")
  passed = finite_ok and positive_ok

  return {
      "status": lw_checks.STATUS_PASS if passed else lw_checks.STATUS_FAIL,
      "passed": passed,
      "finite": finite_ok,
      "final_hidden_norm_scaled": final_hidden_norm_scaled,
      "moe_lb_loss": total_moe_lb_val,
      "total_consistency_objective": total_consistency_objective,
      "per_layer_h_norms": per_layer_h_norms,
      "num_layers_consistency": len(specs),
  }


def run_layer_fwd_microbatched(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    graphdef: Any,
    weights: Any,
    rest: Any,
    plan: Any,
    carry: dict[str, jax.Array],
    positions: jax.Array,
    tokens: jax.Array,
    segment_ids: jax.Array | None,
    image_mask: jax.Array | None,
    b_micro: int,
    config: Any = None,
) -> tuple[dict[str, jax.Array], jax.Array]:
  """Runs one decoder layer forward across microbatches under jax.jit."""
  batch_size = tokens.shape[0]
  b_micro = min(b_micro, batch_size)
  num_m = (batch_size + b_micro - 1) // b_micro
  rules = config.logical_axis_rules if config is not None and hasattr(config, "logical_axis_rules") else None

  def fwd_fn(w, c_m, p_m, t_m, s_m, m_m):
    if rules:
      with logical_axis_rules(rules):
        mod = lw_model.merge_fresh(graphdef, w, rest)
        if plan is not None:
          lw_model.install_sliced_hasher(mod, plan)
        out_c = mod(c_m, p_m, t_m, decoder_segment_ids=s_m, image_mask=m_m)
        inter = nnx.pop(mod, nnx.Intermediate)
        inter_dict = inter.to_pure_dict()
    else:
      mod = lw_model.merge_fresh(graphdef, w, rest)
      if plan is not None:
        lw_model.install_sliced_hasher(mod, plan)
      out_c = mod(c_m, p_m, t_m, decoder_segment_ids=s_m, image_mask=m_m)
      inter = nnx.pop(mod, nnx.Intermediate)
      inter_dict = inter.to_pure_dict()
    lb_list = maxtext_utils.collect_intermediates_by_suffix(inter_dict, "moe_lb_loss")
    if lb_list:
      lb = jnp.mean(jnp.concatenate(lb_list))
    elif (
        config is not None
        and getattr(config, "num_experts", 1) > 1
        and getattr(config, "load_balance_loss_weight", 0.0) > 0.0
    ):
      raise RuntimeError("Missing moe_lb_loss intermediate when load_balance_loss_weight > 0")
    else:
      lb = jnp.zeros((), dtype=jnp.float32)
    return out_c, lb

  fwd_jit = jax.jit(fwd_fn)

  out_carries = []
  lb_losses = []
  for m in range(num_m):
    sl = slice(m * b_micro, min((m + 1) * b_micro, batch_size))
    c_m = {k: v[sl] for k, v in carry.items()}
    p_m = positions[sl]
    t_m = tokens[sl]
    s_m = segment_ids[sl] if segment_ids is not None else None
    m_m = image_mask[sl] if image_mask is not None else None

    out_c_m, lb_m = fwd_jit(weights, c_m, p_m, t_m, s_m, m_m)
    out_carries.append(out_c_m)
    lb_losses.append(lb_m)

  combined_carry = {}
  for k in out_carries[0].keys():
    combined_carry[k] = jnp.concatenate([c[k] for c in out_carries], axis=0)

  del out_carries
  gc.collect()

  combined_lb = jnp.mean(jnp.stack(lb_losses))
  return combined_carry, combined_lb


def run_layer_vjp_microbatched(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    graphdef: Any,
    weights: Any,
    rest: Any,
    plan: Any,
    carry_in: dict[str, jax.Array],
    positions: jax.Array,
    tokens: jax.Array,
    segment_ids: jax.Array | None,
    image_mask: jax.Array | None,
    d_carry_out: dict[str, jax.Array],
    cotangent_lb: float,
    b_micro: int,
    config: Any = None,
) -> tuple[Any, dict[str, jax.Array]]:
  """Runs one decoder layer VJP backward across microbatches under jax.jit."""
  batch_size = tokens.shape[0]
  b_micro = min(b_micro, batch_size)
  num_m = (batch_size + b_micro - 1) // b_micro
  rules = config.logical_axis_rules if config is not None and hasattr(config, "logical_axis_rules") else None

  def fwd_fn(w, c_m, p_m, t_m, s_m, m_m):
    if rules:
      with logical_axis_rules(rules):
        mod = lw_model.merge_fresh(graphdef, w, rest)
        if plan is not None:
          lw_model.install_sliced_hasher(mod, plan)
        out_c = mod(c_m, p_m, t_m, decoder_segment_ids=s_m, image_mask=m_m)
        inter = nnx.pop(mod, nnx.Intermediate)
        inter_dict = inter.to_pure_dict()
    else:
      mod = lw_model.merge_fresh(graphdef, w, rest)
      if plan is not None:
        lw_model.install_sliced_hasher(mod, plan)
      out_c = mod(c_m, p_m, t_m, decoder_segment_ids=s_m, image_mask=m_m)
      inter = nnx.pop(mod, nnx.Intermediate)
      inter_dict = inter.to_pure_dict()
    lb_list = maxtext_utils.collect_intermediates_by_suffix(inter_dict, "moe_lb_loss")
    if lb_list:
      lb = jnp.mean(jnp.concatenate(lb_list))
    elif (
        config is not None
        and getattr(config, "num_experts", 1) > 1
        and getattr(config, "load_balance_loss_weight", 0.0) > 0.0
    ):
      raise RuntimeError("Missing moe_lb_loss intermediate when load_balance_loss_weight > 0")
    else:
      lb = jnp.zeros((), dtype=jnp.float32)
    return out_c, lb

  def vjp_fn(w, c_m, p_m, t_m, s_m, m_m, d_out_c_m, cot_lb_m):
    def f(w_inner, c_inner):
      return fwd_fn(w_inner, c_inner, p_m, t_m, s_m, m_m)

    _, vjp_step = jax.vjp(f, w, c_m)
    d_w_m, d_c_m = vjp_step((d_out_c_m, cot_lb_m))
    return d_w_m, d_c_m

  vjp_jit = jax.jit(vjp_fn)

  accum_d_weights = None
  d_carries_in = []
  cot_lb_m = jnp.asarray(cotangent_lb / float(num_m), dtype=jnp.float32)

  for m in range(num_m):
    sl = slice(m * b_micro, min((m + 1) * b_micro, batch_size))
    c_m = {k: v[sl] for k, v in carry_in.items()}
    p_m = positions[sl]
    t_m = tokens[sl]
    s_m = segment_ids[sl] if segment_ids is not None else None
    m_m = image_mask[sl] if image_mask is not None else None
    d_out_m = {k: d_carry_out[k][sl] for k in d_carry_out.keys()}

    d_w_m, d_c_m = vjp_jit(weights, c_m, p_m, t_m, s_m, m_m, d_out_m, cot_lb_m)
    d_carries_in.append(d_c_m)
    if accum_d_weights is None:
      accum_d_weights = d_w_m
    else:
      accum_d_weights = jax.tree.map(lambda a, b: a + b, accum_d_weights, d_w_m)

  combined_d_carry_in = {}
  for k in d_carries_in[0].keys():
    combined_d_carry_in[k] = jnp.concatenate([c[k] for c in d_carries_in], axis=0)

  del d_carries_in
  gc.collect()

  return accum_d_weights, combined_d_carry_in


def _tree_l2_sq_f32(tree: Any) -> float:
  """Computes L2 norm squared of a pytree in float32."""
  total_sq = 0.0
  for leaf in jax.tree_util.tree_leaves(tree):
    if leaf is None:
      continue
    arr = jnp.asarray(leaf, dtype=jnp.float32)
    total_sq += float(jnp.sum(jnp.square(arr)))
  return total_sq


def build_layer_skeleton(
    config: Any,
    mesh: Any,
    layer_id: int,
    engram_rows: lw_stages.EngramRows | None = None,
) -> tuple[Any, Any, Any, Any]:
  """Constructs Flax NNX (graphdef, rest, plan, abstract_weights) for layer_id without reading safetensors weights."""
  build = lw_programs.layer_build_fn(config, mesh, layer_id)
  parts = lw_model.abstract_parts(build, config, mesh)
  plan = None
  if engram_rows is not None:
    target = nnx.to_pure_dict(parts.weights)
    table_abstract = target["engram"]["embed"]
    active_rows = engram_rows.active
    control_rows = engram_rows.controls
    if engram_rows.include_packing_rows and engram_rows.tokens.shape[1] >= 16:
      temp_mod = lw_model.merge_fresh(parts.graphdef, parts.weights, parts.rest)
      pack_rows = lw_model.engram_packing_control_rows(temp_mod.engram_hasher.val, engram_rows.tokens)
      active_rows = np.unique(np.concatenate([active_rows, pack_rows]))
      control_rows = np.setdiff1d(control_rows, active_rows)
    plan = lw_model.plan_engram_slice(
        active_rows, control_rows, num_table_rows=table_abstract.shape[0], row_multiple=mesh.devices.size
    )
    parts.weights["engram"]["embed"].set_value(
        jax.ShapeDtypeStruct(
            (plan.num_slice_rows, table_abstract.shape[1]),
            table_abstract.dtype,
            sharding=table_abstract.sharding,
        )
    )
  rest = lw_model.materialize_rest(build, config, mesh)
  abstract_weights = _ensure_pure_dict(parts.weights)
  return parts.graphdef, rest, plan, abstract_weights


def _pad_engram_weights_if_needed(cpu_w: Any, abstract_weights: Any) -> Any:
  """Pads engram embedding table along axis 0 up to abstract_weights if cur_rows < target_rows."""
  if (
      isinstance(cpu_w, dict)
      and "engram" in cpu_w
      and isinstance(cpu_w["engram"], dict)
      and "embed" in cpu_w["engram"]
      and isinstance(abstract_weights, dict)
      and "engram" in abstract_weights
      and isinstance(abstract_weights["engram"], dict)
      and "embed" in abstract_weights["engram"]
  ):
    cur_rows = cpu_w["engram"]["embed"].shape[0]
    target_rows = abstract_weights["engram"]["embed"].shape[0]
    if cur_rows < target_rows:
      pad_shape = (target_rows - cur_rows, *cpu_w["engram"]["embed"].shape[1:])
      zero_pad = np.zeros(pad_shape, dtype=cpu_w["engram"]["embed"].dtype)
      cpu_w = dict(cpu_w)
      cpu_w["engram"] = dict(cpu_w["engram"])
      cpu_w["engram"]["embed"] = np.concatenate([np.asarray(cpu_w["engram"]["embed"]), zero_pad], axis=0)
  return cpu_w


def verify_weight_cache_dir(weight_cache_dir: str | pathlib.Path, num_layers: int = 40) -> None:
  """Verifies existing layer weight cache files against .sha256 sidecars before step-1 execution."""
  cache_dir = pathlib.Path(weight_cache_dir)
  if not cache_dir.is_dir():
    raise ValueError(f"Weight cache directory does not exist: {cache_dir}")

  for sha_file in sorted(cache_dir.glob("layer_*.pkl.sha256")):
    c_file = sha_file.with_name(sha_file.name[:-7])
    if not c_file.exists():
      raise ValueError(f"Weight cache file missing for existing sidecar: {sha_file}")

  for layer_idx in range(num_layers):
    cache_file = cache_dir / f"layer_{layer_idx:02d}.pkl"
    sha_file = cache_dir / f"layer_{layer_idx:02d}.pkl.sha256"

    if not cache_file.exists():
      continue

    if cache_file.stat().st_size == 0:
      raise ValueError(f"Weight cache file is empty (0 bytes): {cache_file}")

    with open(cache_file, "rb") as f:
      hdr = f.read(2)
      if len(hdr) < 2 or hdr[0] != 0x80:
        raise ValueError(f"Weight cache file {cache_file} has invalid or corrupted pickle header ({hdr!r}).")

    if not sha_file.exists():
      raise ValueError(
          f"Weight cache file {cache_file} exists but missing sha256 sidecar {sha_file}. "
          "Corrupted or partial cache write detected."
      )

    expected_sha = sha_file.read_text(encoding="utf-8").strip().split()[0]
    if len(expected_sha) != 64:
      raise ValueError(f"Corrupt sha256 sidecar {sha_file}: '{expected_sha}' (expected 64-char hex string)")

    hasher = hashlib.sha256()
    with open(cache_file, "rb") as f:
      while chunk := f.read(64 * 1024 * 1024):
        hasher.update(chunk)
    actual_sha = hasher.hexdigest()
    if actual_sha != expected_sha:
      raise ValueError(
          f"Weight cache checksum mismatch for {cache_file}: "
          f"expected {expected_sha}, computed {actual_sha}. Corrupted weight cache!"
      )


def _get_or_cache_layer_cpu_weights(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    layer_idx: int,
    config: Any,
    mesh: Any,
    hf_dir: pathlib.Path,
    engram_rows: lw_stages.EngramRows | None,
    weight_cache_dir: str | pathlib.Path | None,
    skeletons: dict[tuple[int, int, int], tuple[Any, Any, Any, Any]] | None = None,
    negative_control: bool = False,
    perturbation: str | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
  """Loads or caches CPU weights for layer_idx on disk, reusing in-memory skeletons."""
  cache_file = None
  if weight_cache_dir is not None:
    cache_dir = pathlib.Path(weight_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"layer_{layer_idx:02d}.pkl"

  mesh_key = (id(mesh), id(config), layer_idx)
  is_layer0_perturbed = negative_control and perturbation == "layer0_wq_a" and layer_idx == 0

  # 1. If weight cache file exists on disk, load weights without reading safetensors (skip dynamic engram layers):
  if engram_rows is None and cache_file is not None and cache_file.is_file() and cache_file.stat().st_size > 0:
    if skeletons is not None and mesh_key in skeletons:
      graphdef, rest, plan, abstract_weights = skeletons[mesh_key]
    else:
      graphdef, rest, plan, abstract_weights = build_layer_skeleton(config, mesh, layer_idx, engram_rows)
      if skeletons is not None:
        skeletons[mesh_key] = (graphdef, rest, plan, abstract_weights)
    with open(cache_file, "rb") as f:
      data = pickle.load(f)
    cpu_w = data[0] if isinstance(data, tuple) else data
    cpu_w = _ensure_pure_dict(cpu_w)
    cpu_w = _pad_engram_weights_if_needed(cpu_w, abstract_weights)
    if is_layer0_perturbed:
      cpu_w = dict(cpu_w)
      cpu_w["self_attention"] = dict(cpu_w["self_attention"])
      cpu_w["self_attention"]["wq_a"] = dict(cpu_w["self_attention"]["wq_a"])
      cpu_w["self_attention"]["wq_a"]["kernel"] = -cpu_w["self_attention"]["wq_a"]["kernel"]
    return cpu_w, graphdef, rest, plan, abstract_weights

  # 2. Otherwise load layer from checkpoint, saving weights to disk cache if configured:
  l_parts, plan = lw_stages.load_layer(config, mesh, layer_idx, hf_dir, engram_rows)
  graphdef = l_parts.graphdef
  rest = l_parts.rest
  cpu_w = _ensure_pure_dict(jax.device_get(l_parts.weights))
  abstract_weights = _ensure_pure_dict(
      jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), l_parts.weights)
  )
  if skeletons is not None:
    skeletons[mesh_key] = (graphdef, rest, plan, abstract_weights)
  lw_model.release(l_parts.weights)
  del l_parts
  gc.collect()
  cpu_w = _pad_engram_weights_if_needed(cpu_w, abstract_weights)
  if cache_file is not None and not is_layer0_perturbed and engram_rows is None:
    tmp_file = cache_file.with_suffix(".tmp")
    tmp_sha = cache_file.with_name(f"{cache_file.name}.sha256.tmp")
    with open(tmp_file, "wb") as f:
      pickle.dump(cpu_w, f, protocol=pickle.HIGHEST_PROTOCOL)
    hasher = hashlib.sha256()
    with open(tmp_file, "rb") as f:
      while chunk := f.read(64 * 1024 * 1024):
        hasher.update(chunk)
    digest = hasher.hexdigest()
    with open(tmp_sha, "w", encoding="utf-8") as sf:
      sf.write(f"{digest}  {cache_file.name}\n")
    os.replace(tmp_file, cache_file)
    os.replace(tmp_sha, cache_file.with_name(f"{cache_file.name}.sha256"))
  if is_layer0_perturbed:
    cpu_w = dict(cpu_w)
    cpu_w["self_attention"] = dict(cpu_w["self_attention"])
    cpu_w["self_attention"]["wq_a"] = dict(cpu_w["self_attention"]["wq_a"])
    cpu_w["self_attention"]["wq_a"]["kernel"] = -cpu_w["self_attention"]["wq_a"]["kernel"]
  return cpu_w, graphdef, rest, plan, abstract_weights


def run_chained_step1_loss_and_grad(  # pylint: disable=too-many-locals,too-many-statements,too-many-arguments,too-many-positional-arguments
    config: Any,
    mesh: Any,
    hf_dir: pathlib.Path,
    batch_np: dict[str, Any],
    *,
    b_micro: int = 1,
    negative_control: bool = False,
    nc_perturb_leaf: str | None = DEFAULT_NC_PERTURB_LEAF,
    perturbation: str | None = None,
    retain_layer_grads: bool = True,
    weight_cache_dir: str | pathlib.Path | None = None,
    cpu_weights_list: list[Any] | None = None,
    skeletons: dict[tuple[int, int], tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
  """Executes forward and backward passes of chained step-1 on TPU/CPU streaming layers."""
  if weight_cache_dir is not None:
    pathlib.Path(weight_cache_dir).mkdir(parents=True, exist_ok=True)
    num_decoder_layers = int(getattr(config, "num_decoder_layers", 40))
    verify_weight_cache_dir(weight_cache_dir, num_layers=num_decoder_layers)

  rep = lw_programs.replicated(mesh)

  def dev(x: Any, dtype: Any = None) -> jax.Array:
    return jax.device_put(jnp.asarray(x, dtype), rep)

  tokens_in = np.asarray(batch_np["inputs"])
  if negative_control and perturbation == "corrupt_tokens":
    tokens_in = np.roll(tokens_in, 5, axis=-1)

  targets_seg_np = np.asarray(batch_np["targets_segmentation"])
  if negative_control and perturbation == "zero_targets_segmentation":
    targets_seg_np = np.zeros_like(targets_seg_np)

  total_weights = max(1, int(np.sum(targets_seg_np != 0)))

  tokens = dev(tokens_in, jnp.int32)
  positions = dev(batch_np["inputs_position"], jnp.int32)
  segment_ids = dev(batch_np["inputs_segmentation"], jnp.int32)
  targets = dev(batch_np["targets"], jnp.int32)
  targets_seg = dev(targets_seg_np, jnp.int32)

  # Unit 0: L1 Projector + Token Embedder Forward
  l1_parts = lw_stages.load_l1(config, mesh, hf_dir)
  l1_graphdef = l1_parts.graphdef
  l1_rest = l1_parts.rest
  cpu_w_l1 = jax.device_get(l1_parts.weights)
  lw_model.release(l1_parts.weights)
  del l1_parts
  gc.collect()

  has_images = False
  pixels = None
  if "images" in batch_np and batch_np["images"] is not None:
    has_images = True
    pixels = dev(batch_np["images"], jnp.float32)
    img_mask = dev(tokens == int(config.ds41_image_token_id), jnp.bool_)
  elif "pixel_values" in batch_np and batch_np["pixel_values"] is not None:
    has_images = True
    pixels = dev(batch_np["pixel_values"], jnp.float32)
    img_mask = dev(batch_np["image_token_mask"], jnp.bool_)
  else:
    img_mask = dev(jnp.zeros_like(tokens, dtype=jnp.bool_))

  def u0_forward(w_l1):
    mod = lw_model.merge_fresh(l1_graphdef, w_l1, l1_rest)
    if has_images:
      img_emb, _ = mod.vision_encoder(input_images=pixels, deterministic=True)
      text_emb = mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
      merged = mm_utils.merge_mm_embeddings(
          text_embeddings=text_emb, multimodal_embeddings=img_emb, mask=img_mask, token_masks=None
      )
    else:
      merged = mod.token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
    return deepseek41.initial_carry(merged, config.mhc_expansion_rate)

  w_l1_device = jax.tree.map(lambda x: jax.device_put(x, rep), cpu_w_l1)
  carry_0 = u0_forward(w_l1_device)
  lw_model.release(w_l1_device)
  del w_l1_device
  gc.collect()

  np_tokens = np.asarray(jax.device_get(tokens))
  np_mask = np.asarray(jax.device_get(img_mask))
  np_seg = np.asarray(jax.device_get(segment_ids)) if segment_ids is not None else None

  # Forward Sweep over Decoder Layers with Exact Mathematical Microbatching
  specs = deepseek41.build_layer_specs(config)
  layer_indices = [spec.layer_id for spec in specs]
  saved_states = [jax.device_get(carry_0)]
  aux_lb_losses = []
  engram_rows_by_layer = {}
  layer_skeletons = skeletons if skeletons is not None else {}

  curr_state = dict(carry_0)
  for pos, spec in enumerate(specs):
    i = spec.layer_id
    if spec.has_engram:
      token_map, compressed_vocab_size = deepseek41.compressed_token_map(config)
      hasher = engram_deepseek41.make_hasher(config, i, token_map, compressed_vocab_size)
      hash_ids = np.asarray(hasher(input_ids=np_tokens, decoder_segment_ids=np_seg, image_mask=np_mask))
      active = np.unique(hash_ids[hash_ids >= 0])
      controls = np.array([], dtype=np.int64)
      engram_rows = lw_stages.EngramRows(
          active=active,
          controls=controls,
          tokens=np_tokens,
          image_mask=np_mask,
          include_packing_rows=False,
          decoder_segment_ids=np_seg,
      )
      engram_rows_by_layer[i] = engram_rows
    else:
      engram_rows = None

    if cpu_weights_list is not None and pos < len(cpu_weights_list) and cpu_weights_list[pos] is not None:
      cpu_w = _ensure_pure_dict(cpu_weights_list[pos])
      mesh_key = (id(mesh), id(config), i)
      if mesh_key in layer_skeletons:
        graphdef, rest, plan, abstract_weights = layer_skeletons[mesh_key]
      else:
        graphdef, rest, plan, abstract_weights = build_layer_skeleton(
            config, mesh, i, engram_rows=engram_rows_by_layer.get(i)
        )
        layer_skeletons[mesh_key] = (graphdef, rest, plan, abstract_weights)
    else:
      cpu_w, graphdef, rest, plan, abstract_weights = _get_or_cache_layer_cpu_weights(
          i,
          config,
          mesh,
          hf_dir,
          engram_rows,
          weight_cache_dir,
          skeletons=layer_skeletons,
          negative_control=negative_control,
          perturbation=perturbation,
      )

    cpu_w = _pad_engram_weights_if_needed(_ensure_pure_dict(cpu_w), _ensure_pure_dict(abstract_weights))
    w_device = jax.tree.map(
        lambda x, a: jax.device_put(x, a.sharding), _ensure_pure_dict(cpu_w), _ensure_pure_dict(abstract_weights)
    )
    old_state = curr_state
    curr_state, lb_k = run_layer_fwd_microbatched(
        graphdef,
        w_device,
        rest,
        plan,
        curr_state,
        positions,
        tokens,
        segment_ids,
        img_mask,
        b_micro=b_micro,
        config=config,
    )
    saved_states.append(jax.device_get(curr_state))
    aux_lb_losses.append(lb_k)

    del old_state

    lw_model.release(w_device)
    del w_device
    del cpu_w
    gc.collect()
    if len(specs) >= 8:
      jax.clear_caches()

    if len(layer_indices) >= 8:
      rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
      print(
          f"[lw_chain] Fwd layer {i} ({pos + 1}/{len(layer_indices)}) done, lb={float(lb_k):.6f}, max_rss={rss_gib:.2f} GiB",
          flush=True,
      )

  # Unit 41: Decoder Norm + Head
  head_parts = load_head(config, mesh, hf_dir)
  if negative_control and (
      perturbation == "logits_dense" or (perturbation is None and nc_perturb_leaf == "logits_dense")
  ):
    head_parts.weights["logits_dense"]["kernel"] = head_parts.weights["logits_dense"]["kernel"] * 1.5

  def head_loss_fn(w_head, h_in):
    rules = config.logical_axis_rules if config is not None and hasattr(config, "logical_axis_rules") else None
    if rules:
      with logical_axis_rules(rules):
        h_mod = lw_model.merge_fresh(head_parts.graphdef, w_head, head_parts.rest)
        logits = h_mod(h_in)
        one_hot = jax.nn.one_hot(targets, config.vocab_size)
        xent = optax.softmax_cross_entropy(logits, one_hot)
        return jnp.sum(xent * (targets_seg != 0)) / float(total_weights)
    else:
      h_mod = lw_model.merge_fresh(head_parts.graphdef, w_head, head_parts.rest)
      logits = h_mod(h_in)
      one_hot = jax.nn.one_hot(targets, config.vocab_size)
      xent = optax.softmax_cross_entropy(logits, one_hot)
      return jnp.sum(xent * (targets_seg != 0)) / float(total_weights)

  h_final_in = dev(saved_states[-1]["h"])
  lm_loss, head_vjp = jax.vjp(head_loss_fn, head_parts.weights, h_final_in)
  moe_lb_loss = jnp.mean(jnp.stack(aux_lb_losses))
  total_loss = lm_loss + moe_lb_loss

  # Backward Pass: Head VJP
  d_w_head, d_h_final = head_vjp(jnp.asarray(1.0, dtype=jnp.float32))
  total_grad_sq = _tree_l2_sq_f32(d_w_head)

  if retain_layer_grads:
    head_grads = jax.device_get(d_w_head)
  else:
    head_grads = {}

  lw_model.release(d_w_head)
  del d_w_head
  lw_model.release(head_parts.weights)
  del head_parts
  del head_vjp
  del h_final_in
  del curr_state
  gc.collect()
  if len(specs) >= 8:
    jax.clear_caches()

  d_state = {k: dev(jnp.zeros_like(v)) for k, v in saved_states[-1].items()}
  d_state["h"] = d_h_final
  saved_states[-1] = {}

  # Backward Pass: Reverse Layer Streaming with Microbatching
  per_layer_grad_norms: dict[int, float] = {}
  layer_grads: dict[int, Any] = {}
  cotangent_lb = 1.0 / float(len(specs))

  for pos, i in reversed(list(enumerate(layer_indices))):
    if cpu_weights_list is not None and pos < len(cpu_weights_list) and cpu_weights_list[pos] is not None:
      cpu_w = _ensure_pure_dict(cpu_weights_list[pos])
      mesh_key = (id(mesh), id(config), i)
      if mesh_key in layer_skeletons:
        graphdef, rest, plan, abstract_weights = layer_skeletons[mesh_key]
      else:
        graphdef, rest, plan, abstract_weights = build_layer_skeleton(
            config, mesh, i, engram_rows=engram_rows_by_layer.get(i)
        )
        layer_skeletons[mesh_key] = (graphdef, rest, plan, abstract_weights)
    else:
      cpu_w, graphdef, rest, plan, abstract_weights = _get_or_cache_layer_cpu_weights(
          i,
          config,
          mesh,
          hf_dir,
          engram_rows_by_layer.get(i),
          weight_cache_dir,
          skeletons=layer_skeletons,
          negative_control=negative_control,
          perturbation=perturbation,
      )

    cpu_w = _pad_engram_weights_if_needed(_ensure_pure_dict(cpu_w), _ensure_pure_dict(abstract_weights))
    w_device = jax.tree.map(
        lambda x, a: jax.device_put(x, a.sharding), _ensure_pure_dict(cpu_w), _ensure_pure_dict(abstract_weights)
    )
    st_in_device = {k: dev(v) for k, v in saved_states[pos].items()}

    d_w_i, d_st_in = run_layer_vjp_microbatched(
        graphdef,
        w_device,
        rest,
        plan,
        st_in_device,
        positions,
        tokens,
        segment_ids,
        img_mask,
        d_state,
        cotangent_lb=cotangent_lb,
        b_micro=b_micro,
        config=config,
    )

    layer_sq = _tree_l2_sq_f32(d_w_i)
    total_grad_sq += layer_sq
    per_layer_grad_norms[i] = float(np.sqrt(layer_sq))

    if retain_layer_grads:
      d_w_i_host = jax.device_get(d_w_i)
      if (
          isinstance(d_w_i_host, dict)
          and "engram" in d_w_i_host
          and isinstance(d_w_i_host["engram"], dict)
          and "embed" in d_w_i_host["engram"]
          and plan is not None
      ):
        num_valid = len(plan.row_ids)
        embed_grad = np.asarray(d_w_i_host["engram"]["embed"])
        if embed_grad.shape[0] > num_valid:
          assert np.all(
              embed_grad[num_valid:] == 0
          ), f"Padded engram gradient rows are non-zero: {embed_grad[num_valid:]}"
      layer_grads[i] = d_w_i_host

    lw_model.release(d_w_i)
    del d_w_i
    del d_state
    d_state = d_st_in

    saved_states[pos] = {}

    lw_model.release(w_device)
    del w_device
    del st_in_device
    del cpu_w
    gc.collect()
    if len(specs) >= 8:
      jax.clear_caches()

    if len(layer_indices) >= 8:
      rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
      print(
          f"[lw_chain] Bwd layer {i} (pos {pos}) done, grad_norm={per_layer_grad_norms[i]:.4f}, max_rss={rss_gib:.2f} GiB",
          flush=True,
      )

  # Unit 0 Backward: trainable aligner, delimiters, token_embedder
  w_l1_device = jax.tree.map(lambda x: jax.device_put(x, rep), cpu_w_l1)
  _, u0_vjp = jax.vjp(u0_forward, w_l1_device)
  (d_w_l1,) = u0_vjp(d_state)
  total_grad_sq += _tree_l2_sq_f32(d_w_l1)

  if retain_layer_grads:
    l1_grads = jax.device_get(d_w_l1)
    if isinstance(l1_grads, dict) and "token_embedder" in l1_grads:
      emb_grads = l1_grads["token_embedder"]
      vision_proj_grads = l1_grads.get("vision_encoder", {})
    else:
      emb_grads = l1_grads
      vision_proj_grads = {}
  else:
    vision_proj_grads = {}
    emb_grads = {}

  lw_model.release(d_w_l1)
  del d_w_l1
  lw_model.release(w_l1_device)
  del w_l1_device
  del cpu_w_l1
  lw_model.release(d_state)
  del d_state
  gc.collect()

  raw_grad_norm = float(np.sqrt(total_grad_sq))

  return {
      "lm_loss": float(lm_loss),
      "moe_lb_loss": float(moe_lb_loss),
      "total_loss": float(total_loss),
      "raw_grad_norm": raw_grad_norm,
      "per_layer_grad_norms": per_layer_grad_norms,
      "layer_grads": layer_grads,
      "head_grads": head_grads,
      "emb_grads": emb_grads,
      "vision_proj_grads": vision_proj_grads,
  }


def lw_chain_step1_r4_parity(  # pylint: disable=too-many-locals,too-many-statements
    config: Any,
    mesh: Any,
    hf_dir: pathlib.Path,
    pinned_batch_path: pathlib.Path | None = None,
    *,
    batch: dict[str, Any] | None = None,
    measured_floor: dict[str, float] | None = None,
    ref_metrics: dict[str, float] | None = None,
    clean_metrics: dict[str, float] | None = None,
    negative_control: bool = False,
    nc_perturb_leaf: str | None = DEFAULT_NC_PERTURB_LEAF,
    perturbation: str | None = None,
    out_dir: pathlib.Path | None = None,
    b_micro: int = 1,
    weight_cache_dir: str | pathlib.Path | None = None,
    cpu_weights_list: list[Any] | None = None,
    skeletons: dict[tuple[int, int], tuple[Any, Any, Any]] | None = None,
) -> dict[str, Any]:
  """Runs chained Step-1 training verification on TPU v5p-8 matching design:155 and TPU7x R4 (20c1af6a)."""
  t0 = time.time()
  if config.gradient_accumulation_steps != 1:
    raise ValueError(f"gradient_accumulation_steps must be 1, got {config.gradient_accumulation_steps}")

  # 1. Batch Preparation & Row Selection Contract
  # src/maxtext/trainers/pre_train/train.py:168-171:
  #   if is_train:
  #     for k, v in data.items():
  #       data[k] = v[: config.micro_batch_size_to_train_on, :]
  # On 128-device TPU7x step (16 hosts, per_device_batch_size=0.125), micro_batch_size_to_train_on = 16.
  # Rows [:16] of the 128 generated rows are trained on.
  batch_row_selection_source = "src/maxtext/trainers/pre_train/train.py:168-171"
  grad_norm_scope_source = "src/maxtext/trainers/pre_train/train.py:586,638,801"

  if batch is not None:
    batch_np = dict(batch)
  elif pinned_batch_path is not None and pathlib.Path(pinned_batch_path).is_file():
    with np.load(pinned_batch_path) as b_npz:
      batch_np = dict(b_npz)
  else:
    batch_np = generate_exact_step0_synthetic_batch()

  # Decimate to the 16 trained rows if full loaded batch was provided
  for k in list(batch_np.keys()):
    if isinstance(batch_np[k], np.ndarray) and batch_np[k].ndim > 0 and batch_np[k].shape[0] > 16:
      batch_np[k] = batch_np[k][:16]

  # Verification of total non-zero target weights
  targets_segmentation_np = np.asarray(batch_np["targets_segmentation"])
  total_weights = int(np.sum(targets_segmentation_np != 0))
  if int(config.max_target_length) == 512 and total_weights != 5248:
    raise ValueError(f"Expected exact total_weights == 5248, got {total_weights}")

  # Compute exact SHA-256 for all arrays
  batch_sha256 = {}
  for k in (
      "inputs",
      "targets",
      "inputs_segmentation",
      "targets_segmentation",
      "inputs_position",
      "images",
      "image_masks",
  ):
    if k in batch_np and batch_np[k] is not None:
      arr = np.ascontiguousarray(batch_np[k])
      batch_sha256[k] = hashlib.sha256(arr.tobytes()).hexdigest()

  specs = deepseek41.build_layer_specs(config)
  num_layers = len(specs)

  if weight_cache_dir is None:
    if out_dir is not None:
      weight_cache_dir = pathlib.Path(out_dir) / "layer_weight_cache"
    elif pathlib.Path("/mnt/disks/data").is_dir() and num_layers >= 8:
      weight_cache_dir = pathlib.Path("/mnt/disks/data/ds41_lw_step1_weight_cache")

  layer_skeletons = skeletons if skeletons is not None else {}

  # Execute chained step 1 loss and grad streaming without retaining host layer gradients
  res = run_chained_step1_loss_and_grad(
      config,
      mesh,
      hf_dir,
      batch_np,
      b_micro=b_micro,
      negative_control=negative_control,
      nc_perturb_leaf=nc_perturb_leaf,
      perturbation=perturbation,
      retain_layer_grads=False,
      weight_cache_dir=weight_cache_dir,
      cpu_weights_list=cpu_weights_list,
      skeletons=layer_skeletons,
  )
  gc.collect()
  if len(specs) >= 8:
    jax.clear_caches()

  # Evaluation against Pinned References & Noise Floor Gate
  eval_ref = dict(TPU7X_R4_STEP1_REF if ref_metrics is None else ref_metrics)
  if measured_floor is None:
    raise ValueError(
        "measured_floor must be provided as a dict with keys ('lm_loss', 'moe_lb_loss', 'raw_grad_norm'). "
        "Use run_full_step1_r4_parity_suite to measure the floor across fsdp=4 vs ep=4 and evaluate parity."
    )
  floor = measured_floor
  eval_metrics = {
      "lm_loss": res["lm_loss"],
      "moe_lb_loss": res["moe_lb_loss"],
      "total_loss": res["total_loss"],
      "raw_grad_norm": res["raw_grad_norm"],
  }

  gates = {}
  deltas = {}
  all_passed = True

  for k in ("lm_loss", "moe_lb_loss", "raw_grad_norm"):
    delta = abs(eval_metrics[k] - eval_ref[k])
    deltas[k] = delta
    tau = TAU_HW[k]
    capped_f = min(float(floor.get(k, 0.0)), tau)
    tol = max(MEASURED_FLOOR_MULTIPLIER * capped_f, tau)
    g_pass = bool(np.isfinite(delta) and delta <= tol)
    gates[k] = {
        "candidate": eval_metrics[k],
        "ref": eval_ref[k],
        "delta": delta,
        "floor": floor.get(k, 0.0),
        "tau_hw": tau,
        "effective_tolerance": tol,
        "passed": g_pass,
    }
    if not g_pass:
      all_passed = False

  if negative_control:
    clean_ok = True
    if clean_metrics is not None:
      clean_ok = all(
          np.isfinite(float(clean_metrics[k]))
          and abs(float(clean_metrics[k]) - float(eval_ref[k])) <= gates[k]["effective_tolerance"]
          for k in ("lm_loss", "moe_lb_loss", "raw_grad_norm")
      )
    nc_fired = bool(clean_ok and (not all_passed))
  else:
    nc_fired = False
  final_passed = nc_fired if negative_control else all_passed

  elapsed = time.time() - t0
  report = {
      "status": lw_checks.STATUS_PASS if final_passed else lw_checks.STATUS_FAIL,
      "passed": final_passed,
      "negative_control": {
          "active": negative_control,
          "nc_fired": nc_fired,
          "neg_all_exceed_gate": nc_fired,
          "perturb_leaf": nc_perturb_leaf,
          "perturbation": perturbation,
      },
      "metrics": eval_metrics,
      "per_layer_grad_norms": res["per_layer_grad_norms"],
      "tpu7x_r4_ref": eval_ref,
      "tau_hw": TAU_HW,
      "measured_floor_method": "abs_diff_v5p8_fsdp4_ep1_vs_fsdp1_ep4",
      "measured_floor": floor,
      "deltas": deltas,
      "gates": gates,
      "batch_row_selection_source": batch_row_selection_source,
      "grad_norm_scope_source": grad_norm_scope_source,
      "batch_sha256": batch_sha256,
      "total_weights": total_weights,
      "elapsed_seconds": elapsed,
  }

  if out_dir is not None:
    p = pathlib.Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fname = "chain_step1_r4_parity_nc.json" if negative_control else "chain_step1_r4_parity.json"
    (p / fname).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

  return report


run_chained_step1_r4_parity = lw_chain_step1_r4_parity


def run_full_step1_r4_parity_suite(  # pylint: disable=too-many-locals,too-many-statements
    config_base: Any,
    devices: list[Any],
    hf_dir: pathlib.Path,
    pinned_batch_path: pathlib.Path | None = None,
    *,
    batch: dict[str, Any] | None = None,
    ref_metrics: dict[str, float] | None = None,
    out_dir: pathlib.Path | None = None,
    b_micro: int = 1,
    weight_cache_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
  """Runs full design:155 Step-1 R4 parity suite: fsdp=4, ep=4 floor measurement, gate check, and negative control."""
  t0 = time.time()
  num_devices = len(devices)
  if weight_cache_dir is None:
    if out_dir is not None:
      weight_cache_dir = pathlib.Path(out_dir) / "layer_weight_cache"
    elif pathlib.Path("/mnt/disks/data").is_dir():
      weight_cache_dir = pathlib.Path("/mnt/disks/data/ds41_lw_step1_weight_cache")

  shared_skeletons: dict[tuple[int, int, int], tuple[Any, Any, Any, Any]] = {}

  if batch is None:
    if pinned_batch_path is not None and pinned_batch_path.is_file():
      with np.load(pinned_batch_path) as b_npz:
        batch = dict(b_npz)
    else:
      batch = generate_exact_step0_synthetic_batch()

  batch = {
      k: (v[:16] if hasattr(v, "__getitem__") and (hasattr(v, "ndim") and v.ndim >= 1) and v.shape[0] > 16 else v)
      for k, v in batch.items()
  }
  total_loss_tokens = int(np.sum(np.asarray(batch["targets_segmentation"]) != 0))
  if int(config_base.max_target_length) == 512 and total_loss_tokens != 5248:
    raise ValueError(
        f"Step 1 R4 parity suite requires 5248 loss tokens (got {total_loss_tokens} from targets_segmentation)"
    )

  # 1. Pass 1: fsdp=4, ep=1
  cfg_fsdp4 = config_base
  mesh_fsdp4 = lw_config.build_mesh(cfg_fsdp4, devices)
  print("[step1_suite] Running Pass 1 (fsdp=4, ep=1)...", flush=True)
  res_fsdp4 = run_chained_step1_loss_and_grad(
      cfg_fsdp4,
      mesh_fsdp4,
      hf_dir,
      batch_np=dict(batch),
      b_micro=b_micro,
      retain_layer_grads=False,
      weight_cache_dir=weight_cache_dir,
      skeletons=shared_skeletons,
  )
  m_fsdp4 = {
      "lm_loss": float(res_fsdp4["lm_loss"]),
      "moe_lb_loss": float(res_fsdp4["moe_lb_loss"]),
      "raw_grad_norm": float(res_fsdp4["raw_grad_norm"]),
  }
  print(f"[step1_suite] Pass 1 metrics: {m_fsdp4}", flush=True)

  # 2. Pass 2: fsdp=1, ep=4
  print("[step1_suite] Running Pass 2 (fsdp=1, ep=4)...", flush=True)
  batch_size = int(batch["inputs"].shape[0])
  cfg_ep4 = lw_config.build_config(
      lw_config.TIER_PRODUCTION,
      reference_activation_quant=True,
      tokenizer_path=str(cfg_fsdp4.tokenizer_path),
      seq_len=cfg_fsdp4.max_target_length,
      batch_size=batch_size,
      num_devices=num_devices,
      expert_parallelism=min(4, num_devices),
      extra={
          "gradient_accumulation_steps": 1,
          "scan_layers": False,
          "remat_policy": "none",
          "freeze_vision_encoder_params": True,
          "load_balance_loss_weight": cfg_fsdp4.load_balance_loss_weight,
      },
  )
  mesh_ep4 = lw_config.build_mesh(cfg_ep4, devices)
  res_ep4 = run_chained_step1_loss_and_grad(
      cfg_ep4,
      mesh_ep4,
      hf_dir,
      batch_np=dict(batch),
      b_micro=b_micro,
      retain_layer_grads=False,
      weight_cache_dir=weight_cache_dir,
      skeletons=shared_skeletons,
  )
  m_ep4 = {
      "lm_loss": float(res_ep4["lm_loss"]),
      "moe_lb_loss": float(res_ep4["moe_lb_loss"]),
      "raw_grad_norm": float(res_ep4["raw_grad_norm"]),
  }
  print(f"[step1_suite] Pass 2 metrics: {m_ep4}", flush=True)

  # 3. Compute measured noise floor (capped so diverging ep=4 never widens effective_gate) & gate both meshes
  eval_ref = dict(TPU7X_R4_STEP1_REF if ref_metrics is None else ref_metrics)
  metric_keys = ("lm_loss", "moe_lb_loss", "raw_grad_norm")
  measured_floor = {k: abs(m_fsdp4[k] - m_ep4[k]) for k in metric_keys}
  capped_floor = {
      k: min(measured_floor[k], TAU_HW[k]) if np.isfinite(measured_floor[k]) else TAU_HW[k] for k in metric_keys
  }
  effective_gate = {k: max(MEASURED_FLOOR_MULTIPLIER * capped_floor[k], tau_val) for k, tau_val in TAU_HW.items()}
  deltas_vs_r4 = {k: abs(m_fsdp4[k] - eval_ref[k]) for k in metric_keys}
  ep4_deltas_vs_r4 = {k: abs(m_ep4[k] - eval_ref[k]) for k in metric_keys}
  fsdp4_passed = all(np.isfinite(deltas_vs_r4[k]) and deltas_vs_r4[k] <= effective_gate[k] for k in metric_keys)
  ep4_passed = all(np.isfinite(ep4_deltas_vs_r4[k]) and ep4_deltas_vs_r4[k] <= effective_gate[k] for k in metric_keys)
  primary_passed = fsdp4_passed and ep4_passed

  # 4. Pass 3: Negative Control (perturbation="layer0_wq_a" only, never double-perturbing logits_dense)
  print("[step1_suite] Running Pass 3 (Negative Control layer0_wq_a)...", flush=True)
  res_nc = run_chained_step1_loss_and_grad(
      cfg_fsdp4,
      mesh_fsdp4,
      hf_dir,
      batch_np=dict(batch),
      b_micro=b_micro,
      negative_control=True,
      nc_perturb_leaf="",
      perturbation="layer0_wq_a",
      retain_layer_grads=False,
      weight_cache_dir=weight_cache_dir,
      skeletons=shared_skeletons,
  )
  m_nc = {
      "lm_loss": float(res_nc["lm_loss"]),
      "moe_lb_loss": float(res_nc["moe_lb_loss"]),
      "raw_grad_norm": float(res_nc["raw_grad_norm"]),
  }
  neg_control_deltas_vs_r4 = {k: abs(m_nc[k] - eval_ref[k]) for k in metric_keys}
  neg_control_triggered = primary_passed and any(
      (not np.isfinite(neg_control_deltas_vs_r4[k])) or neg_control_deltas_vs_r4[k] > effective_gate[k]
      for k in metric_keys
  )
  suite_passed = primary_passed and neg_control_triggered

  # Compute batch sha256
  batch_used = dict(batch) if batch is not None else generate_exact_step0_synthetic_batch()
  for k in list(batch_used.keys()):
    if isinstance(batch_used[k], np.ndarray) and batch_used[k].ndim > 0 and batch_used[k].shape[0] > 16:
      batch_used[k] = batch_used[k][:16]
  batch_sha256 = {}
  for k in (
      "inputs",
      "targets",
      "inputs_segmentation",
      "targets_segmentation",
      "inputs_position",
      "images",
      "image_masks",
  ):
    if k in batch_used and batch_used[k] is not None:
      arr = np.ascontiguousarray(batch_used[k])
      batch_sha256[k] = hashlib.sha256(arr.tobytes()).hexdigest()

  summary = {
      "status": lw_checks.STATUS_PASS if suite_passed else lw_checks.STATUS_FAIL,
      "passed": suite_passed,
      "primary_passed": primary_passed,
      "fsdp4_passed": fsdp4_passed,
      "ep4_passed": ep4_passed,
      "neg_control_triggered": neg_control_triggered,
      "v5p8_fsdp4_metrics": m_fsdp4,
      "v5p8_ep4_metrics": m_ep4,
      "measured_floor": measured_floor,
      "effective_gate": effective_gate,
      "deltas_vs_r4": deltas_vs_r4,
      "ep4_deltas_vs_r4": ep4_deltas_vs_r4,
      "neg_control_metrics": m_nc,
      "neg_control_deltas_vs_r4": neg_control_deltas_vs_r4,
      "tpu7x_r4_ref": eval_ref,
      "tau_hw": TAU_HW,
      "batch_sha256": batch_sha256,
      "total_loss_tokens": int(np.sum(np.asarray(batch_used["targets_segmentation"]) != 0)),
      "duration_sec": time.time() - t0,
  }

  if out_dir is not None:
    p = pathlib.Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "step1_r4_parity_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

  return summary


def flags_parser(argv: list[str]) -> argparse.Namespace:
  """Parses CLI arguments and marks absl flags parsed."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--hf-ckpt-dir", required=True)
  parser.add_argument("--bundle-dir", required=False, default="")
  parser.add_argument("--pinned-batch", required=False, default="")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--tokenizer-path", required=False, default="")
  parser.add_argument("--sample-id", required=False, default="M4")
  parser.add_argument("--negative-control", action="store_true", default=False)
  parser.add_argument("--b-micro", type=int, default=2)
  parser.add_argument("--weight-cache-dir", required=False, default="")
  args = parser.parse_args(argv[1:])
  flags.FLAGS.mark_as_parsed()
  return args


def _chain_forward_ep4_agrees(
    fwd_fsdp: ChainForwardReport,
    fwd_ep: ChainForwardReport,
    h_fsdp: dict[str, np.ndarray],
    h_ep: dict[str, np.ndarray],
    *,
    is_golden: bool,
) -> tuple[bool, dict[str, Any]]:
  """Checks that ep=4 chained forward passed and agrees with fsdp=4 outputs."""
  if fwd_fsdp.status != lw_checks.STATUS_PASS or fwd_ep.status != lw_checks.STATUS_PASS:
    return False, {"reason": f"status mismatch fsdp={fwd_fsdp.status}, ep={fwd_ep.status}"}
  n_fsdp = float(fwd_fsdp.metrics["final_h_norm"])
  n_ep = float(fwd_ep.metrics["final_h_norm"])
  if not (np.isfinite(n_fsdp) and np.isfinite(n_ep)):
    return False, {"reason": "non-finite final_h_norm"}
  norm_tol = 1e-5 * max(1.0, abs(n_fsdp)) if is_golden else max(1e-5, 1e-2 * abs(n_fsdp))
  norm_diff = abs(n_ep - n_fsdp)
  if norm_diff > norm_tol:
    return False, {"final_h_norm_diff": norm_diff, "norm_tol": norm_tol}
  if "h_final" in h_fsdp and "h_final" in h_ep:
    arr_fsdp = h_fsdp["h_final"]
    arr_ep = h_ep["h_final"]
    if is_golden:
      chk = lw_checks.check_code_parity("chain_forward.ep4_vs_fsdp4", arr_ep, arr_fsdp, _DEFAULT_TOLERANCES.code_parity)
      return chk.passed, dict(chk.metrics)
    arr_f64 = np.asarray(arr_fsdp, np.float64).reshape(-1)
    arr_e64 = np.asarray(arr_ep, np.float64).reshape(-1)
    if not (np.all(np.isfinite(arr_f64)) and np.all(np.isfinite(arr_e64))):
      return False, {"finite": False}
    norm_f = float(np.linalg.norm(arr_f64))
    norm_e = float(np.linalg.norm(arr_e64))
    if norm_f >= lw_checks.COSINE_NORM_FLOOR and norm_e >= lw_checks.COSINE_NORM_FLOOR:
      gap = lw_checks.bf16_gap_metrics(arr_e64, arr_f64)
      ok = (
          gap["rel_l2"] <= _DEFAULT_TOLERANCES.ref_gap_max_rel_l2
          and gap["cos_sim"] >= _DEFAULT_TOLERANCES.ref_gap_min_cos_sim
      )
      return ok, gap
    max_abs = float(np.max(np.abs(arr_e64 - arr_f64)))
    return max_abs <= 1e-2, {"max_abs_diff": max_abs}
  return True, {"final_h_norm_diff": norm_diff, "norm_tol": norm_tol}


def _hidden_norm_ep4_agrees(hn_fsdp: dict[str, Any], hn_ep: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
  """Checks that ep=4 hidden-norm consistency passed and matches fsdp=4 per-layer and final norms."""
  if hn_fsdp.get("status") != lw_checks.STATUS_PASS or hn_ep.get("status") != lw_checks.STATUS_PASS:
    return False, {"reason": f"status mismatch fsdp={hn_fsdp.get('status')}, ep={hn_ep.get('status')}"}
  n_fsdp = float(hn_fsdp["final_hidden_norm_scaled"])
  n_ep = float(hn_ep["final_hidden_norm_scaled"])
  if not (np.isfinite(n_fsdp) and np.isfinite(n_ep)) or n_fsdp <= 0.0 or n_ep <= 0.0:
    return False, {"reason": f"invalid final_hidden_norm_scaled fsdp={n_fsdp}, ep={n_ep}"}
  final_tol = max(1e-5, 1e-2 * abs(n_fsdp))
  final_diff = abs(n_ep - n_fsdp)
  if final_diff > final_tol:
    return False, {"final_hidden_norm_diff": final_diff, "final_tol": final_tol}
  layers_fsdp = hn_fsdp.get("per_layer_h_norms")
  layers_ep = hn_ep.get("per_layer_h_norms")
  if layers_fsdp is not None and layers_ep is not None:
    if len(layers_fsdp) != len(layers_ep):
      return False, {"reason": f"layer count mismatch {len(layers_fsdp)} vs {len(layers_ep)}"}
    max_rel = 0.0
    for idx, (vf, ve) in enumerate(zip(layers_fsdp, layers_ep)):
      if not (np.isfinite(vf) and np.isfinite(ve)) or vf <= 0.0 or ve <= 0.0:
        return False, {"reason": f"invalid layer {idx} norm fsdp={vf}, ep={ve}"}
      tol = max(1e-5, 1e-2 * abs(vf))
      diff = abs(ve - vf)
      if diff > tol:
        return False, {"divergent_layer": idx, "fsdp_norm": vf, "ep_norm": ve, "tol": tol}
      max_rel = max(max_rel, diff / max(abs(vf), 1e-12))
    return True, {"final_hidden_norm_diff": final_diff, "max_layer_rel_diff": max_rel}
  return True, {"final_hidden_norm_diff": final_diff}


def main(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals,too-many-statements
  devices = jax.devices()
  num_devices = len(devices)
  hf_dir = pathlib.Path(args.hf_ckpt_dir)
  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  pinned_batch = pathlib.Path(args.pinned_batch) if args.pinned_batch else None
  tok_path = args.tokenizer_path or str(hf_dir)
  w_cache = pathlib.Path(args.weight_cache_dir) if args.weight_cache_dir else None

  # Common configs for fsdp=4 (ep=1) and ep=4 (ep=min(4, num_devices))
  cfg_prod_fsdp4 = lw_config.build_config(
      lw_config.TIER_PRODUCTION,
      reference_activation_quant=True,
      tokenizer_path=tok_path,
      seq_len=512,
      batch_size=16,
      num_devices=num_devices,
      expert_parallelism=1,
      extra={
          "gradient_accumulation_steps": 1,
          "scan_layers": False,
          "remat_policy": "none",
          "freeze_vision_encoder_params": True,
          "load_balance_loss_weight": 0.0001,
      },
  )
  mesh_prod_fsdp4 = lw_config.build_mesh(cfg_prod_fsdp4, devices)

  cfg_prod_ep4 = lw_config.build_config(
      lw_config.TIER_PRODUCTION,
      reference_activation_quant=True,
      tokenizer_path=tok_path,
      seq_len=512,
      batch_size=16,
      num_devices=num_devices,
      expert_parallelism=min(4, num_devices),
      extra={
          "gradient_accumulation_steps": 1,
          "scan_layers": False,
          "remat_policy": "none",
          "freeze_vision_encoder_params": True,
          "load_balance_loss_weight": 0.0001,
      },
  )
  mesh_prod_ep4 = lw_config.build_mesh(cfg_prod_ep4, devices)

  chain_results: dict[str, Any] = {}

  if args.bundle_dir and pathlib.Path(args.bundle_dir).is_dir():
    b_dir = pathlib.Path(args.bundle_dir)
    b_index = lw_bundle.load_index(b_dir)

    cfg_parity_fsdp4 = lw_config.build_config(
        lw_config.TIER_CODE_PARITY,
        reference_activation_quant=True,
        tokenizer_path=tok_path,
        seq_len=512,
        batch_size=16,
        num_devices=num_devices,
        expert_parallelism=1,
        extra={
            "gradient_accumulation_steps": 1,
            "scan_layers": False,
            "remat_policy": "none",
            "freeze_vision_encoder_params": True,
            "load_balance_loss_weight": 0.0001,
        },
    )
    mesh_parity_fsdp4 = lw_config.build_mesh(cfg_parity_fsdp4, devices)

    cfg_parity_ep4 = lw_config.build_config(
        lw_config.TIER_CODE_PARITY,
        reference_activation_quant=True,
        tokenizer_path=tok_path,
        seq_len=512,
        batch_size=16,
        num_devices=num_devices,
        expert_parallelism=min(4, num_devices),
        extra={
            "gradient_accumulation_steps": 1,
            "scan_layers": False,
            "remat_policy": "none",
            "freeze_vision_encoder_params": True,
            "load_balance_loss_weight": 0.0001,
        },
    )
    mesh_parity_ep4 = lw_config.build_mesh(cfg_parity_ep4, devices)

    try:
      # 1. Four forward runs
      h_parity_fsdp4: dict[str, np.ndarray] = {}
      h_parity_ep4: dict[str, np.ndarray] = {}
      h_prod_fsdp4: dict[str, np.ndarray] = {}
      h_prod_ep4: dict[str, np.ndarray] = {}

      print(f"[lw_chain] Running chained forward (golden_parity, fsdp=4) on sample {args.sample_id}...")
      fwd_parity_fsdp4 = lw_chain_forward(
          cfg_parity_fsdp4,
          mesh_parity_fsdp4,
          hf_dir,
          b_index,
          args.sample_id,
          mode="golden_parity",
          out_state=h_parity_fsdp4,
      )
      print(f"[lw_chain] Running chained forward (golden_parity, ep=4) on sample {args.sample_id}...")
      fwd_parity_ep4 = lw_chain_forward(
          cfg_parity_ep4,
          mesh_parity_ep4,
          hf_dir,
          b_index,
          args.sample_id,
          mode="golden_parity",
          out_state=h_parity_ep4,
      )
      print(f"[lw_chain] Running chained forward (production_bf16, fsdp=4) on sample {args.sample_id}...")
      fwd_prod_fsdp4 = lw_chain_forward(
          cfg_prod_fsdp4,
          mesh_prod_fsdp4,
          hf_dir,
          b_index,
          args.sample_id,
          mode="production_bf16",
          out_state=h_prod_fsdp4,
      )
      print(f"[lw_chain] Running chained forward (production_bf16, ep=4) on sample {args.sample_id}...")
      fwd_prod_ep4 = lw_chain_forward(
          cfg_prod_ep4,
          mesh_prod_ep4,
          hf_dir,
          b_index,
          args.sample_id,
          mode="production_bf16",
          out_state=h_prod_ep4,
      )

      parity_ep4_ok, parity_ep4_cmp = _chain_forward_ep4_agrees(
          fwd_parity_fsdp4, fwd_parity_ep4, h_parity_fsdp4, h_parity_ep4, is_golden=True
      )
      prod_ep4_ok, prod_ep4_cmp = _chain_forward_ep4_agrees(
          fwd_prod_fsdp4, fwd_prod_ep4, h_prod_fsdp4, h_prod_ep4, is_golden=False
      )
      h_parity_fsdp4.clear()
      h_parity_ep4.clear()
      h_prod_fsdp4.clear()
      h_prod_ep4.clear()

      # 2. Hidden norm consistency (fsdp=4 and ep=4)
      if pinned_batch is None or not pinned_batch.is_file():
        syn_path = out_dir / "synthetic_pinned_batch.npz"
        if not syn_path.is_file():
          syn_batch = generate_exact_step0_synthetic_batch()
          np.savez(syn_path, **syn_batch)
        pinned_batch = syn_path

      print("[lw_chain] Running chained hidden norm consistency (fsdp=4)...")
      hn_fsdp4 = lw_chain_hidden_norm_consistency(cfg_prod_fsdp4, mesh_prod_fsdp4, hf_dir, pinned_batch)
      print("[lw_chain] Running chained hidden norm consistency (ep=4)...")
      hn_ep4 = lw_chain_hidden_norm_consistency(cfg_prod_ep4, mesh_prod_ep4, hf_dir, pinned_batch)
      hn_ep4_ok, hn_ep4_cmp = _hidden_norm_ep4_agrees(hn_fsdp4, hn_ep4)

      # 3. Step 1 R4 parity suite
      print("[lw_chain] Running Step 1 R4 parity suite...")
      suite_res = run_full_step1_r4_parity_suite(
          cfg_prod_fsdp4,
          devices,
          hf_dir,
          pinned_batch,
          out_dir=out_dir,
          b_micro=args.b_micro,
          weight_cache_dir=w_cache,
      )
    finally:
      b_index.close()

    fsdp4_step1_passed = bool(suite_res.get("fsdp4_passed", suite_res.get("primary_passed", False)))
    ep4_step1_passed = bool(suite_res.get("ep4_passed", suite_res.get("primary_passed", False)))

    step5_checks = {
        "chain_forward:golden_parity:fsdp=4": {
            "status": fwd_parity_fsdp4.status,
            "passed": fwd_parity_fsdp4.status == lw_checks.STATUS_PASS,
            "report": dataclasses.asdict(fwd_parity_fsdp4),
        },
        "chain_forward:golden_parity:ep=4": {
            "status": lw_checks.STATUS_PASS if parity_ep4_ok else lw_checks.STATUS_FAIL,
            "passed": parity_ep4_ok,
            "ep4_vs_fsdp4": parity_ep4_cmp,
            "report": dataclasses.asdict(fwd_parity_ep4),
        },
        "chain_forward:production_bf16:fsdp=4": {
            "status": fwd_prod_fsdp4.status,
            "passed": fwd_prod_fsdp4.status == lw_checks.STATUS_PASS,
            "report": dataclasses.asdict(fwd_prod_fsdp4),
        },
        "chain_forward:production_bf16:ep=4": {
            "status": lw_checks.STATUS_PASS if prod_ep4_ok else lw_checks.STATUS_FAIL,
            "passed": prod_ep4_ok,
            "ep4_vs_fsdp4": prod_ep4_cmp,
            "report": dataclasses.asdict(fwd_prod_ep4),
        },
        "chain_hidden_norm_consistency:fsdp=4": {
            "status": hn_fsdp4.get("status", lw_checks.STATUS_FAIL),
            "passed": hn_fsdp4.get("status") == lw_checks.STATUS_PASS,
            "report": hn_fsdp4,
        },
        "chain_hidden_norm_consistency:ep=4": {
            "status": lw_checks.STATUS_PASS if hn_ep4_ok else lw_checks.STATUS_FAIL,
            "passed": hn_ep4_ok,
            "ep4_vs_fsdp4": hn_ep4_cmp,
            "report": hn_ep4,
        },
        "chain_step1_r4_parity:fsdp=4": {
            "status": lw_checks.STATUS_PASS if fsdp4_step1_passed else lw_checks.STATUS_FAIL,
            "passed": fsdp4_step1_passed,
            "metrics": suite_res.get("v5p8_fsdp4_metrics"),
            "deltas_vs_r4": suite_res.get("deltas_vs_r4"),
        },
        "chain_step1_r4_parity:ep=4": {
            "status": lw_checks.STATUS_PASS if ep4_step1_passed else lw_checks.STATUS_FAIL,
            "passed": ep4_step1_passed,
            "metrics": suite_res.get("v5p8_ep4_metrics"),
            "ep4_deltas_vs_r4": suite_res.get("ep4_deltas_vs_r4"),
            "measured_floor": suite_res.get("measured_floor"),
        },
        "chain_step1_r4_parity:negative_control": {
            "status": (lw_checks.STATUS_PASS if suite_res.get("neg_control_triggered", False) else lw_checks.STATUS_FAIL),
            "passed": bool(suite_res.get("neg_control_triggered", False)),
            "metrics": suite_res.get("neg_control_metrics"),
            "neg_control_deltas_vs_r4": suite_res.get("neg_control_deltas_vs_r4"),
        },
    }

    all_passed = all(c["passed"] for c in step5_checks.values()) and bool(suite_res.get("passed", False))
    if args.negative_control:
      nc_triggered = bool(suite_res.get("neg_control_triggered", False))
      chain_results["negative_control_passed"] = nc_triggered
      if not nc_triggered:
        print("[lw_chain] Negative control requested but not triggered!", file=sys.stderr)
        all_passed = False

    overall_status = lw_checks.STATUS_PASS if all_passed else lw_checks.STATUS_FAIL
    chain_results["step5_checks"] = step5_checks
    chain_results["overall_status"] = overall_status
    chain_results["passed"] = all_passed
    chain_results["step1_r4_parity"] = suite_res

    summary_path = out_dir / "step5_chain_summary.json"
    summary_path.write_text(json.dumps(chain_results, indent=2, default=str), encoding="utf-8")
    print(f"[lw_chain] Saved Step 5 chain summary to {summary_path}")
    print(json.dumps({"overall_status": overall_status, "passed": all_passed}, indent=2))
    if not all_passed:
      raise SystemExit(1)
  else:
    if pinned_batch is not None and pinned_batch.is_file():
      print("[lw_chain] Running chained hidden norm consistency (fsdp=4)...")
      hn_res = lw_chain_hidden_norm_consistency(cfg_prod_fsdp4, mesh_prod_fsdp4, hf_dir, pinned_batch)
      chain_results["hidden_norm_consistency"] = hn_res
      if hn_res.get("status") != lw_checks.STATUS_PASS:
        summary_path = out_dir / "step5_chain_summary.json"
        summary_path.write_text(json.dumps(chain_results, indent=2, default=str), encoding="utf-8")
        raise SystemExit(1)

    print("[lw_chain] Running Step 1 R4 parity suite...")
    res = run_full_step1_r4_parity_suite(
        cfg_prod_fsdp4,
        devices,
        hf_dir,
        pinned_batch,
        out_dir=out_dir,
        b_micro=args.b_micro,
        weight_cache_dir=w_cache,
    )
    chain_results["step1_r4_parity"] = res
    if args.negative_control:
      nc_triggered = bool(res.get("neg_control_triggered", False))
      chain_results["negative_control_passed"] = nc_triggered
      if not nc_triggered:
        print("[lw_chain] Negative control requested but not triggered!", file=sys.stderr)
        res["passed"] = False

    summary_path = out_dir / "step5_chain_summary.json"
    summary_path.write_text(json.dumps(chain_results, indent=2, default=str), encoding="utf-8")
    print(f"[lw_chain] Saved Step 5 chain summary to {summary_path}")
    print(json.dumps(res, indent=2, default=str))
    if not res["passed"]:
      raise SystemExit(1)


if __name__ == "__main__":
  app.run(main, flags_parser=flags_parser)
