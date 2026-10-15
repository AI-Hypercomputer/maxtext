# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint loading and conversion for GLM-5.3-Flash."""

from collections.abc import Mapping
import glob
import json
import math
import os
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.glm5_3_flash import create_mini_slice
from maxtext.m3.models.glm5_3_flash import glm5_config
from maxtext.m3.models.glm5_3_flash import glm5_moe
from maxtext.m3.models.glm5_3_flash import glm5_scan
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch


def _is_fp8(val: Any) -> bool:
  return "float8" in str(getattr(val, "dtype", ""))


def dequantize_fp8_block128(
    weight: Any, scale_inv: Any, *, block_size: tuple[int, int] = (128, 128), target_dtype: Any
) -> jax.Array:
  """Dequantizes 2D or 3D FP8 weights with 128x128 block-wise inverse scales."""
  if not _is_fp8(weight):
    raise ValueError(f"Weight must be float8 dtype, got {getattr(weight, 'dtype', type(weight))}.")
  to_np_dt = np.float64 if target_dtype in (jnp.float64, np.float64) else np.float32

  def to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
      return (x.to(torch.float32) if _is_fp8(x) else x.detach()).cpu().numpy().astype(to_np_dt)
    return np.asarray(x, dtype=to_np_dt)

  w_np, s_np = to_np(weight), to_np(scale_inv)
  if len(w_np.shape) not in (2, 3):
    raise ValueError(f"Weight must be 2D or 3D, got shape {w_np.shape}.")
  m, n = w_np.shape[-2], w_np.shape[-1]
  bm, bn = block_size
  sm, sn = math.ceil(m / bm), math.ceil(n / bn)
  exp_scale = (sm, sn) if len(w_np.shape) == 2 else (w_np.shape[0], sm, sn)
  if s_np.shape != exp_scale:
    raise ValueError(f"scale_inv shape {s_np.shape} != expected {exp_scale} for weight shape {w_np.shape}.")
  scale = np.repeat(np.repeat(s_np, bm, axis=-2), bn, axis=-1)[..., :m, :n]
  return jnp.asarray(w_np * scale, dtype=target_dtype)


def _normalize_key(key: str) -> str:
  prefix = "model.language_model."
  if key.startswith(prefix):
    return key[len(prefix) :]
  if key == "lm_head.weight" or key.startswith("lm_head."):
    return key
  raise ValueError(f"Unknown HF key prefix: {key!r}")


def load_hf_state_into_model(
    model: modeling_glm5_3_flash.Glm5Model,
    hf_tensors: Mapping[str, Any],
    *,
    ignore_mtp_layer: bool,
) -> None:
  """Loads HF safetensors parameters into a Glm5Model instance."""
  orig_key_map: dict[str, str] = {}
  for orig_k in hf_tensors.keys():
    norm_k = _normalize_key(orig_k)
    if norm_k in orig_key_map:
      raise ValueError(f"Duplicate normalized key found: {norm_k!r}")
    orig_key_map[norm_k] = orig_k

  consumed: set[str] = set()

  def _get(norm_k: str) -> jax.Array:
    if norm_k not in orig_key_map:
      raise ValueError(f"Required parameter missing from HF state: {norm_k!r}")
    orig_k = orig_key_map[norm_k]
    consumed.add(orig_k)
    val = hf_tensors[orig_k]
    scale_k = norm_k[:-7] + ".weight_scale_inv" if norm_k.endswith(".weight") else norm_k + "_scale_inv"
    is_fp8 = _is_fp8(val)
    has_scale = scale_k in orig_key_map
    if is_fp8 and not has_scale:
      raise ValueError(f"FP8 weight {norm_k!r} is missing scale inverse {scale_k!r}.")
    if has_scale and not is_fp8:
      raise ValueError(
          f"Scale inverse {scale_k!r} present but weight {norm_k!r} is not float8 (got {getattr(val, 'dtype', type(val))})."
      )
    if has_scale:
      consumed.add(orig_key_map[scale_k])
      val = dequantize_fp8_block128(
          val, hf_tensors[orig_key_map[scale_k]], target_dtype=getattr(jnp, model.config.weight_dtype)
      )
    if isinstance(val, torch.Tensor):
      dt = torch.float64 if getattr(jnp, model.config.weight_dtype) == jnp.float64 else torch.float32
      val = val.detach().to(dt).cpu().numpy()
    return jnp.asarray(val)

  def _get_t(k: str) -> jax.Array:
    return _get(k).T

  num_prefix = min(model.config.num_hidden_layers, model.config.first_k_dense_replace)
  num_sb = (model.config.num_hidden_layers - num_prefix) // 4
  assigned_slots: set[tuple[int, int]] = set()

  def _assign(var: nnx.Variable, val: jax.Array, b_idx: int = 0, is_stacked: bool = False) -> None:
    assigned_slots.add((id(var), b_idx))
    v = val.astype(var.value.dtype)
    var.value = var.value.at[b_idx].set(v) if is_stacked else v

  _assign(model.embed_tokens.embedding, _get("embed_tokens.weight"))
  _assign(model.norm.weight, _get("norm.weight"))
  _assign(model.lm_head.kernel, _get_t("lm_head.weight"))

  for l in range(model.config.num_hidden_layers):
    sec, b_idx, s_idx = glm5_scan.layer_to_block_slot(l, model.config.first_k_dense_replace, num_sb)
    is_stacked = sec == "super_block"
    target_l = getattr(model, f"{sec}_layers")[b_idx] if not is_stacked else getattr(model.super_block, f"slot_{s_idx}")

    def _set(var: nnx.Variable, val: jax.Array, b: int = b_idx, stacked: bool = is_stacked) -> None:
      _assign(var, val, b, stacked)

    lp = f"layers.{l}."

    for pre in ("attn", "ffn"):
      hc = getattr(target_l, f"{pre}_hc")
      _set(hc.fn, _get_t(f"{lp}hc_{pre}_fn"))
      _set(hc.base, _get(f"{lp}hc_{pre}_base"))
      _set(hc.scale, _get(f"{lp}hc_{pre}_scale"))

    for n in ("input_layernorm", "post_attention_layernorm"):
      _set(getattr(target_l, n).weight, _get(f"{lp}{n}.weight"))

    attn = target_l.self_attn
    if target_l.block_type == "linear_attention":
      for p in ("q", "k", "v", "b", "o", "g_a", "g_b", "f_a", "f_b"):
        _set(getattr(attn, f"{p}_proj").kernel, _get_t(f"{lp}self_attn.{p}_proj.weight"))
      _set(attn.A_log, _get(f"{lp}self_attn.A_log"))
      _set(attn.dt_bias, _get(f"{lp}self_attn.dt_bias"))
      _set(attn.o_norm.weight, _get(f"{lp}self_attn.o_norm.weight"))
      for c in ("q", "k", "v"):
        _set(getattr(attn, f"{c}_conv1d").weight, _get(f"{lp}self_attn.{c}_conv1d.weight"))
    else:
      for p in ("q_a", "q_b", "kv_b", "o"):
        _set(getattr(attn, f"{p}_proj").kernel, _get_t(f"{lp}self_attn.{p}_proj.weight"))
      _set(attn.kv_a_proj_with_mqa.kernel, _get_t(f"{lp}self_attn.kv_a_proj_with_mqa.weight"))
      for p in ("q_a", "kv_a"):
        _set(getattr(attn, f"{p}_layernorm").weight, _get(f"{lp}self_attn.{p}_layernorm.weight"))
      idx = attn.indexer
      _set(idx.wq_b.kernel, _get_t(f"{lp}self_attn.indexer.wq_b.weight"))
      for p in ("wk", "weights_proj"):
        _set(getattr(idx, p).kernel, _get_t(f"{lp}self_attn.indexer.{p}.weight"))
      for p in ("weight", "bias"):
        _set(getattr(idx.k_norm, p), _get(f"{lp}self_attn.indexer.k_norm.{p}"))
      for p in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
        _set(getattr(idx, p), _get(f"{lp}self_attn.indexer.{p}"))

    mlp_tgt = target_l.mlp if not target_l.is_moe else target_l.mlp.shared_experts
    for p in ("gate", "up", "down"):
      _set(
          getattr(mlp_tgt, f"{p}_proj").kernel,
          _get_t(f"{lp}mlp.{p}_proj.weight" if not target_l.is_moe else f"{lp}mlp.shared_experts.{p}_proj.weight"),
      )
    if target_l.is_moe:
      _set(target_l.mlp.gate.weight, _get_t(f"{lp}mlp.gate.weight"))
      _set(target_l.mlp.gate.e_score_correction_bias, _get(f"{lp}mlp.gate.e_score_correction_bias"))
      ne = model.config.n_routed_experts
      gus = [
          jnp.concatenate(
              [_get_t(f"{lp}mlp.experts.{e}.gate_proj.weight"), _get_t(f"{lp}mlp.experts.{e}.up_proj.weight")],
              axis=-1,
          )
          for e in range(ne)
      ]
      _set(target_l.mlp.experts.gate_up_proj, jnp.stack(gus, axis=0))
      _set(
          target_l.mlp.experts.down_proj,
          jnp.stack([_get_t(f"{lp}mlp.experts.{e}.down_proj.weight") for e in range(ne)], axis=0),
      )

  if not ignore_mtp_layer:
    if model.mtp is None:
      raise ValueError("ignore_mtp_layer=False requires model.mtp to be initialized (num_nextn_predict_layers > 0).")
    lp = f"layers.{glm5_config.MTP_LAYER_INDEX}."
    _assign(model.mtp.enorm.weight, _get(f"{lp}enorm.weight"))
    _assign(model.mtp.hnorm.weight, _get(f"{lp}hnorm.weight"))
    _assign(model.mtp.eh_proj.kernel, _get_t(f"{lp}eh_proj.weight"))
    _assign(model.mtp.shared_head_norm.weight, _get(f"{lp}shared_head.norm.weight"))
    _assign(model.mtp.input_layernorm.weight, _get(f"{lp}input_layernorm.weight"))
    _assign(model.mtp.post_attention_layernorm.weight, _get(f"{lp}post_attention_layernorm.weight"))

    attn = model.mtp.self_attn
    for p in ("q_a", "q_b", "kv_b", "o"):
      _assign(getattr(attn, f"{p}_proj").kernel, _get_t(f"{lp}self_attn.{p}_proj.weight"))
    _assign(attn.kv_a_proj_with_mqa.kernel, _get_t(f"{lp}self_attn.kv_a_proj_with_mqa.weight"))
    for p in ("q_a", "kv_a"):
      _assign(getattr(attn, f"{p}_layernorm").weight, _get(f"{lp}self_attn.{p}_layernorm.weight"))
    idx = attn.indexer
    _assign(idx.wq_b.kernel, _get_t(f"{lp}self_attn.indexer.wq_b.weight"))
    for p in ("wk", "weights_proj"):
      _assign(getattr(idx, p).kernel, _get_t(f"{lp}self_attn.indexer.{p}.weight"))
    for p in ("weight", "bias"):
      _assign(getattr(idx.k_norm, p), _get(f"{lp}self_attn.indexer.k_norm.{p}"))
    for p in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
      _assign(getattr(idx, p), _get(f"{lp}self_attn.indexer.{p}"))

    mlp_tgt = model.mtp.mlp.shared_experts
    for p in ("gate", "up", "down"):
      _assign(getattr(mlp_tgt, f"{p}_proj").kernel, _get_t(f"{lp}mlp.shared_experts.{p}_proj.weight"))
    _assign(model.mtp.mlp.gate.weight, _get_t(f"{lp}mlp.gate.weight"))
    _assign(model.mtp.mlp.gate.e_score_correction_bias, _get(f"{lp}mlp.gate.e_score_correction_bias"))
    ne = model.config.n_routed_experts
    gus = [
        jnp.concatenate(
            [_get_t(f"{lp}mlp.experts.{e}.gate_proj.weight"), _get_t(f"{lp}mlp.experts.{e}.up_proj.weight")],
            axis=-1,
        )
        for e in range(ne)
    ]
    _assign(model.mtp.mlp.experts.gate_up_proj, jnp.stack(gus, axis=0))
    _assign(
        model.mtp.mlp.experts.down_proj,
        jnp.stack([_get_t(f"{lp}mlp.experts.{e}.down_proj.weight") for e in range(ne)], axis=0),
    )

  for orig_k in hf_tensors.keys():
    if orig_k not in consumed:
      clean = _normalize_key(orig_k)
      if ignore_mtp_layer and clean.startswith(f"layers.{glm5_config.MTP_LAYER_INDEX}."):
        continue
      raise ValueError(f"Unmapped key in hf_tensors: {orig_k!r}")

  mtp_leaves = set()
  if model.mtp is not None and ignore_mtp_layer:
    for v in jax.tree.leaves(nnx.state(model.mtp)):
      mtp_leaves.add(id(v))

  sb_leaves = set()
  if model.super_block is not None:
    for s in range(4):
      sl = getattr(model.super_block, f"slot_{s}")
      for v in jax.tree.leaves(nnx.state(sl)):
        sb_leaves.add(id(v))
  for p in jax.tree.leaves(nnx.state(model, (nnx.Param, glm5_moe.Glm5RouterBias))):
    if isinstance(p, nnx.Variable):
      if id(p) in mtp_leaves:
        continue
      for b in range(num_sb) if id(p) in sb_leaves else (0,):
        if (id(p), b) not in assigned_slots:
          raise ValueError(f"Unpopulated model parameter slot ({b}): {p}")


def save_orbax_checkpoint(model: modeling_glm5_3_flash.Glm5Model, orbax_dir: str) -> None:
  """Saves a Glm5Model checkpoint in Orbax format."""
  checkpointer = ocp.StandardCheckpointer()
  checkpointer.save(orbax_dir, nnx.to_pure_dict(nnx.state(model)))
  checkpointer.wait_until_finished()


def restore_orbax_checkpoint(model: modeling_glm5_3_flash.Glm5Model, orbax_dir: str) -> None:
  """Restores Orbax checkpoint state into an existing Glm5Model instance."""
  nnx.update(model, nnx.restore_int_paths(ocp.StandardCheckpointer().restore(orbax_dir)))


def convert_hf_checkpoint_to_orbax(
    hf_dir: str, orbax_dir: str, config: glm5_config.Glm5Config, *, ignore_mtp_layer: bool
) -> None:
  """Converts HuggingFace FP8/BF16 safetensors checkpoint to an Orbax checkpoint."""
  index_path = os.path.join(hf_dir, "model.safetensors.index.json")
  all_tensors: dict[str, Any] = {}
  if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
      weight_map = json.load(f)["weight_map"]
    needed = create_mini_slice.extract_mini_slice_keys(
        weight_map,
        config.num_hidden_layers,
        include_mtp_layer=None if ignore_mtp_layer else glm5_config.MTP_LAYER_INDEX,
    )
    shard_paths = [os.path.join(hf_dir, s) for s in sorted(set(needed.values()))]
  else:
    shard_paths = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    if not shard_paths:
      raise ValueError(f"No safetensors files found in {hf_dir}")
    needed = None
  for sp in shard_paths:
    with safetensors.safe_open(sp, framework="pt") as f:
      for k in f.keys():
        if needed is None or k in needed:
          all_tensors[k] = f.get_tensor(k)

  model = modeling_glm5_3_flash.Glm5Model(config, rngs=nnx.Rngs(0))
  load_hf_state_into_model(model, all_tensors, ignore_mtp_layer=ignore_mtp_layer)
  save_orbax_checkpoint(model, orbax_dir)
