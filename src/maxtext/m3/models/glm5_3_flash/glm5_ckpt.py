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

from collections.abc import Callable, Mapping
import dataclasses
import functools
import glob
import json
import math
import os
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.models.glm5_3_flash import create_mini_slice
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

MAX_IN_MEMORY_CONVERSION_LAYERS = 6
MAX_IN_MEMORY_CONVERSION_PARAMS = 25_000_000_000


def _is_fp8(val: Any) -> bool:
  """Checks whether a tensor or array has exact float8_e4m3fn dtype."""
  if not hasattr(val, "dtype") or val.dtype is None:
    raise TypeError("Expected array or tensor with a .dtype attribute, got" f" {type(val).__name__}.")
  return val.dtype in (
      torch.float8_e4m3fn,
      jnp.float8_e4m3fn,
      ml_dtypes.float8_e4m3fn,
      np.dtype(ml_dtypes.float8_e4m3fn),
  )


_TARGET_TO_INTERMEDIATE_DTYPE: dict[np.dtype, tuple[torch.dtype, np.dtype]] = {
    np.dtype(np.float64): (torch.float64, np.dtype(np.float64)),
    np.dtype(np.float32): (torch.float32, np.dtype(np.float32)),
    np.dtype(ml_dtypes.bfloat16): (torch.float32, np.dtype(np.float32)),
}


def _to_target_np_array(val: Any, target_dtype: Any) -> np.ndarray:
  """Converts a tensor or array to an owned NumPy array of target_dtype."""
  target_np_dt = np.dtype(target_dtype)
  if target_np_dt not in _TARGET_TO_INTERMEDIATE_DTYPE:
    raise ValueError(
        f"Unsupported target_dtype {target_dtype!r} (resolved to"
        f" {target_np_dt}); expected one of"
        f" {tuple(_TARGET_TO_INTERMEDIATE_DTYPE.keys())}."
    )
  if isinstance(val, torch.Tensor):
    inter_torch_dt, _ = _TARGET_TO_INTERMEDIATE_DTYPE[target_np_dt]
    val = val.detach().to(inter_torch_dt).cpu().numpy()
  return np.array(val, dtype=target_np_dt, copy=True)


def dequantize_fp8_block128(
    weight: Any,
    scale_inv: Any,
    *,
    block_size: tuple[int, int] = (128, 128),
    target_dtype: Any,
) -> np.ndarray:
  """Dequantizes 2D or 3D FP8 weights with 128x128 block-wise inverse scales.

  Returns a np.ndarray of np.dtype(target_dtype) and performs intermediate
  block-128 multiplication in
  _TARGET_TO_INTERMEDIATE_DTYPE[np.dtype(target_dtype)] (float64 for float64
  targets and float32 for float32/bfloat16 targets), operating purely in
  NumPy/PyTorch on host so float64 targets preserve exact 64-bit precision even
  when jax_enable_x64 is False.

  Args:
    weight: 2D or 3D FP8 weight tensor or array.
    scale_inv: Block-wise inverse scale tensor or array.
    block_size: 2D block size (bm, bn), default (128, 128).
    target_dtype: Target dtype (e.g. float32, bfloat16, float64).

  Returns:
    Dequantized weight as a NumPy array of target_dtype.
  """
  if not _is_fp8(weight):
    raise ValueError(f"Expected float8_e4m3fn weight, got {weight.dtype}.")
  target_np_dt = np.dtype(target_dtype)
  if target_np_dt not in _TARGET_TO_INTERMEDIATE_DTYPE:
    raise ValueError(
        f"Unsupported target_dtype {target_dtype!r} (resolved to"
        f" {target_np_dt}); expected one of"
        f" {tuple(_TARGET_TO_INTERMEDIATE_DTYPE.keys())}."
    )
  _, compute_np_dt = _TARGET_TO_INTERMEDIATE_DTYPE[target_np_dt]
  w_np = _to_target_np_array(weight, compute_np_dt)
  s_np = _to_target_np_array(scale_inv, compute_np_dt)
  if len(w_np.shape) not in (2, 3):
    raise ValueError(f"Weight must be 2D or 3D, got shape {w_np.shape}.")
  m, n = w_np.shape[-2], w_np.shape[-1]
  bm, bn = block_size
  sm, sn = math.ceil(m / bm), math.ceil(n / bn)
  exp_scale = (sm, sn) if len(w_np.shape) == 2 else (w_np.shape[0], sm, sn)
  if s_np.shape != exp_scale:
    raise ValueError(f"scale_inv shape {s_np.shape} != expected {exp_scale} for weight shape" f" {w_np.shape}.")
  scale = np.repeat(np.repeat(s_np, bm, axis=-2), bn, axis=-1)[..., :m, :n]
  return np.asarray(w_np * scale, dtype=target_np_dt)


def _normalize_key(key: str) -> str:
  prefix = "model.language_model."
  if key.startswith(prefix):
    return key[len(prefix) :]
  if key.startswith("lm_head."):
    return key
  raise ValueError(f"Unknown HF key prefix: {key!r}")


def _fetch_and_convert_tensor(
    hub_k: str,
    target_dtype: Any,
    orig_key_map: Mapping[str, str],
    consumed: set[str],
    load_raw_fn: Callable[[str], Any],
) -> np.ndarray:
  """Fetches a raw tensor from storage, dequantizing or converting to target_dtype."""
  if hub_k not in orig_key_map:
    raise ValueError(f"Required parameter missing from HF state: {hub_k!r}")
  orig_k = orig_key_map[hub_k]
  consumed.add(orig_k)
  val = load_raw_fn(orig_k)
  norm_k = _normalize_key(orig_k)
  scale_norm_k = norm_k[:-7] + ".weight_scale_inv" if norm_k.endswith(".weight") else norm_k + "_scale_inv"
  scale_hub_k = scale_norm_k if scale_norm_k.startswith("lm_head.") else f"model.language_model.{scale_norm_k}"
  is_fp8 = _is_fp8(val)
  has_scale = scale_hub_k in orig_key_map
  if is_fp8 and not has_scale:
    raise ValueError(f"FP8 weight {norm_k!r} is missing scale inverse {scale_norm_k!r}.")
  if has_scale and not is_fp8:
    raise ValueError(f"Scale inverse {scale_norm_k!r} present but weight {norm_k!r} is not" f" float8 (got {val.dtype}).")
  if has_scale:
    orig_scale_k = orig_key_map[scale_hub_k]
    consumed.add(orig_scale_k)
    scale_val = load_raw_fn(orig_scale_k)
    return dequantize_fp8_block128(val, scale_val, target_dtype=target_dtype)
  return _to_target_np_array(val, target_dtype)


@dataclasses.dataclass(frozen=True)
class ConversionSpec:
  """Specification for mapping HuggingFace tensor(s) to a MaxText model parameter."""

  hf_keys: tuple[str, ...]
  target_path: tuple[str | int, ...]
  stack_index: int | None
  num_super_blocks: int | None
  kind: str  # "direct" | "transpose_2d" | "moe_gate_up" | "moe_down"
  target_dtype: Any  # jnp.dtype


def iter_conversion_specs(
    config: glm5_config.Glm5Config,
    *,
    include_mtp: bool,
) -> list[ConversionSpec]:
  """Generates all HuggingFace-to-MaxText conversion specs for a Glm5Config.

  Single owner of the HuggingFace safetensors key mapping.
  Target paths correspond to attribute paths on Glm5Model.

  Args:
    config: Glm5Config defining the model architecture.
    include_mtp: Whether to include MTP layer specifications.

  Returns:
    List of ConversionSpec objects in deterministic order.
  """
  if include_mtp and config.num_nextn_predict_layers == 0:
    raise ValueError("include_mtp=True requires config.num_nextn_predict_layers > 0.")
  specs: list[ConversionSpec] = []
  p_dt = getattr(jnp, config.weight_dtype)
  r_dt = getattr(jnp, config.router_param_dtype)

  def _add(
      hf_keys: tuple[str, ...],
      target_path: tuple[str | int, ...],
      kind: str,
      *,
      dtype: Any,
      stack_idx: int | None = None,
      num_sbs: int | None = None,
  ) -> None:
    specs.append(
        ConversionSpec(
            hf_keys=hf_keys,
            target_path=target_path,
            stack_index=stack_idx,
            num_super_blocks=num_sbs,
            kind=kind,
            target_dtype=dtype,
        )
    )

  def _emit_mla(
      lp: str,
      tgt_base: tuple[str | int, ...],
      stack_idx: int | None,
      num_sbs: int | None,
  ) -> None:
    def _add_l(
        hf_suffix: str,
        tgt_suffix: tuple[str | int, ...],
        kind: str,
        *,
        dtype: Any,
    ) -> None:
      _add(
          (f"{lp}{hf_suffix}",),
          (*tgt_base, *tgt_suffix),
          kind,
          dtype=dtype,
          stack_idx=stack_idx,
          num_sbs=num_sbs,
      )

    for p in ("q_a", "q_b", "kv_b", "o"):
      _add_l(
          f"self_attn.{p}_proj.weight",
          ("self_attn", f"{p}_proj", "kernel"),
          "transpose_2d",
          dtype=p_dt,
      )
    _add_l(
        "self_attn.kv_a_proj_with_mqa.weight",
        ("self_attn", "kv_a_proj_with_mqa", "kernel"),
        "transpose_2d",
        dtype=p_dt,
    )
    for p in ("q_a", "kv_a"):
      _add_l(
          f"self_attn.{p}_layernorm.weight",
          ("self_attn", f"{p}_layernorm", "weight"),
          "direct",
          dtype=p_dt,
      )
    if config.use_indexer:
      _add_l(
          "self_attn.indexer.wq_b.weight",
          ("self_attn", "indexer", "wq_b", "kernel"),
          "transpose_2d",
          dtype=p_dt,
      )
      for p in ("wk", "weights_proj"):
        _add_l(
            f"self_attn.indexer.{p}.weight",
            ("self_attn", "indexer", p, "kernel"),
            "transpose_2d",
            dtype=p_dt,
        )
      for p in ("weight", "bias"):
        _add_l(
            f"self_attn.indexer.k_norm.{p}",
            ("self_attn", "indexer", "k_norm", p),
            "direct",
            dtype=p_dt,
        )
      for p in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
        _add_l(
            f"self_attn.indexer.{p}",
            ("self_attn", "indexer", p),
            "direct",
            dtype=p_dt,
        )

  def _emit_moe(
      lp: str,
      tgt_base: tuple[str | int, ...],
      stack_idx: int | None,
      num_sbs: int | None,
  ) -> None:

    def _add_l(
        hf_suffix: str,
        tgt_suffix: tuple[str | int, ...],
        kind: str,
        *,
        dtype: Any,
    ) -> None:
      _add(
          (f"{lp}{hf_suffix}",),
          (*tgt_base, *tgt_suffix),
          kind,
          dtype=dtype,
          stack_idx=stack_idx,
          num_sbs=num_sbs,
      )

    for p in ("gate", "up", "down"):
      _add_l(
          f"mlp.shared_experts.{p}_proj.weight",
          ("mlp", "shared_experts", f"{p}_proj", "kernel"),
          "transpose_2d",
          dtype=p_dt,
      )
    _add_l("mlp.gate.weight", ("mlp", "gate", "weight"), "transpose_2d", dtype=r_dt)
    _add_l(
        "mlp.gate.e_score_correction_bias",
        ("mlp", "gate", "e_score_correction_bias"),
        "direct",
        dtype=jnp.float32,
    )
    ne = config.n_routed_experts
    gate_up_keys = tuple(
        k
        for e in range(ne)
        for k in (
            f"{lp}mlp.experts.{e}.gate_proj.weight",
            f"{lp}mlp.experts.{e}.up_proj.weight",
        )
    )
    _add(
        gate_up_keys,
        (*tgt_base, "mlp", "experts", "gate_up_proj"),
        "moe_gate_up",
        dtype=p_dt,
        stack_idx=stack_idx,
        num_sbs=num_sbs,
    )
    down_keys = tuple(f"{lp}mlp.experts.{e}.down_proj.weight" for e in range(ne))
    _add(
        down_keys,
        (*tgt_base, "mlp", "experts", "down_proj"),
        "moe_down",
        dtype=p_dt,
        stack_idx=stack_idx,
        num_sbs=num_sbs,
    )

  _add(
      ("model.language_model.embed_tokens.weight",),
      ("embed_tokens", "embedding"),
      "direct",
      dtype=p_dt,
  )
  _add(
      ("model.language_model.norm.weight",),
      ("norm", "weight"),
      "direct",
      dtype=p_dt,
  )
  _add(
      ("lm_head.weight",),
      ("lm_head", "kernel"),
      "transpose_2d",
      dtype=p_dt,
  )

  def _add_layer_spec(
      layer_prefix: str,
      target_base: tuple[str | int, ...],
      s_idx_val: int | None,
      n_sbs_val: int | None,
      hf_suffix: str,
      tgt_suffix: tuple[str | int, ...],
      kind: str,
      *,
      dtype: Any,
  ) -> None:
    _add(
        (f"{layer_prefix}{hf_suffix}",),
        target_base + tgt_suffix,
        kind,
        dtype=dtype,
        stack_idx=s_idx_val,
        num_sbs=n_sbs_val,
    )

  num_prefix = min(config.num_hidden_layers, config.first_k_dense_replace)
  num_sb = (config.num_hidden_layers - num_prefix) // 4
  layer_types = config.layer_types
  if layer_types is None:
    raise ValueError("config.layer_types must not be None.")

  for l in range(config.num_hidden_layers):
    sec, b_idx, s_idx = modeling_glm5_3_flash.layer_to_block_slot(
        l, config.first_k_dense_replace, config.num_hidden_layers
    )
    is_stacked = sec == "super_block"
    stack_idx = b_idx if is_stacked else None
    num_sbs = num_sb if is_stacked else None
    tgt_base: tuple[str | int, ...] = ("super_block", f"slot_{s_idx}") if is_stacked else (f"{sec}_layers", b_idx)
    lp = f"model.language_model.layers.{l}."

    _add_l = functools.partial(_add_layer_spec, lp, tgt_base, stack_idx, num_sbs)

    for pre in ("attn", "ffn"):
      _add_l(f"hc_{pre}_fn", (f"{pre}_hc", "fn"), "transpose_2d", dtype=p_dt)
      _add_l(f"hc_{pre}_base", (f"{pre}_hc", "base"), "direct", dtype=p_dt)
      _add_l(f"hc_{pre}_scale", (f"{pre}_hc", "scale"), "direct", dtype=p_dt)

    for n in ("input_layernorm", "post_attention_layernorm"):
      _add_l(f"{n}.weight", (n, "weight"), "direct", dtype=p_dt)

    if layer_types[l] == "linear_attention":
      for p in ("q", "k", "v", "b", "o", "g_a", "g_b", "f_a", "f_b"):
        _add_l(
            f"self_attn.{p}_proj.weight",
            ("self_attn", f"{p}_proj", "kernel"),
            "transpose_2d",
            dtype=p_dt,
        )
      _add_l("self_attn.A_log", ("self_attn", "A_log"), "direct", dtype=jnp.float32)
      _add_l(
          "self_attn.dt_bias",
          ("self_attn", "dt_bias"),
          "direct",
          dtype=jnp.float32,
      )
      _add_l(
          "self_attn.o_norm.weight",
          ("self_attn", "o_norm", "weight"),
          "direct",
          dtype=p_dt,
      )
      for c in ("q", "k", "v"):
        _add_l(
            f"self_attn.{c}_conv1d.weight",
            ("self_attn", f"{c}_conv1d", "weight"),
            "direct",
            dtype=jnp.float32,
        )
    else:
      _emit_mla(lp, tgt_base, stack_idx, num_sbs)

    if l < config.first_k_dense_replace:
      for p in ("gate", "up", "down"):
        _add_l(
            f"mlp.{p}_proj.weight",
            ("mlp", f"{p}_proj", "kernel"),
            "transpose_2d",
            dtype=p_dt,
        )
    else:
      _emit_moe(lp, tgt_base, stack_idx, num_sbs)

  if include_mtp:
    lp = f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}."
    tgt_base = ("mtp",)
    for pre in (
        "enorm",
        "hnorm",
        "input_layernorm",
        "post_attention_layernorm",
    ):
      _add(
          (f"{lp}{pre}.weight",),
          (*tgt_base, pre, "weight"),
          "direct",
          dtype=p_dt,
      )
    _add(
        (f"{lp}eh_proj.weight",),
        (*tgt_base, "eh_proj", "kernel"),
        "transpose_2d",
        dtype=p_dt,
    )
    _add(
        (f"{lp}shared_head.norm.weight",),
        (*tgt_base, "shared_head_norm", "weight"),
        "direct",
        dtype=p_dt,
    )
    _emit_mla(lp, tgt_base, None, None)
    _emit_moe(lp, tgt_base, None, None)

  return specs


def resolve_target_var(root: Any, path: tuple[str | int, ...]) -> nnx.Variable:
  """Resolves a nested path of attribute names and list indices to an nnx.Variable."""
  curr = root
  for part in path:
    if isinstance(part, int):
      curr = curr[part]
    else:
      curr = getattr(curr, part)
  if not isinstance(curr, nnx.Variable):
    raise TypeError(f"Target path {path} resolved to non-Variable {type(curr)}: {curr}")
  return curr


def _transform_for_spec(spec: ConversionSpec, get_tensor_fn: Callable[[str, Any], jax.Array]) -> jax.Array:
  """Applies the conversion specification transformation kind to retrieved tensor(s)."""
  dt = spec.target_dtype
  if spec.kind == "direct":
    return get_tensor_fn(spec.hf_keys[0], dt)
  if spec.kind == "transpose_2d":
    return get_tensor_fn(spec.hf_keys[0], dt).T
  if spec.kind == "moe_gate_up":
    ne = len(spec.hf_keys) // 2
    gus = [
        jnp.stack(
            [
                get_tensor_fn(spec.hf_keys[2 * e], dt).T,
                get_tensor_fn(spec.hf_keys[2 * e + 1], dt).T,
            ],
            axis=1,
        )
        for e in range(ne)
    ]
    return jnp.stack(gus, axis=0)
  if spec.kind == "moe_down":
    ne = len(spec.hf_keys)
    return jnp.stack([get_tensor_fn(spec.hf_keys[e], dt).T for e in range(ne)], axis=0)
  raise ValueError(f"Unknown conversion spec kind: {spec.kind!r}")


def load_hf_state_into_model(
    model: modeling_glm5_3_flash.Glm5Model,
    hf_tensors: Mapping[str, Any],
    *,
    ignore_mtp_layer: bool,
) -> None:
  """Loads HF safetensors parameters into a Glm5Model instance."""
  if ignore_mtp_layer and hasattr(model, "mtp") and model.mtp is not None:
    raise ValueError(
        "ignore_mtp_layer=True cannot be used when model.mtp is initialized"
        " (config.num_nextn_predict_layers > 0); set num_nextn_predict_layers=0"
        " on the config instead."
    )
  if not ignore_mtp_layer and (model.mtp is None or model.config.num_nextn_predict_layers == 0):
    raise ValueError("ignore_mtp_layer=False requires model.mtp to be initialized" " (num_nextn_predict_layers > 0).")
  orig_key_map: dict[str, str] = {}
  for orig_k in hf_tensors.keys():
    if orig_k.startswith("model.visual.") or orig_k.startswith("visual."):
      continue
    norm_k = _normalize_key(orig_k)
    hub_k = norm_k if norm_k.startswith("lm_head.") else f"model.language_model.{norm_k}"
    if hub_k in orig_key_map:
      raise ValueError(f"Duplicate normalized key found: {norm_k!r}")
    orig_key_map[hub_k] = orig_k

  consumed: set[str] = set()

  def _get(hub_k: str, target_dtype: Any) -> jax.Array:
    return jnp.asarray(
        _fetch_and_convert_tensor(
            hub_k,
            target_dtype,
            orig_key_map,
            consumed,
            lambda k: hf_tensors[k],
        ),
        dtype=target_dtype,
    )

  assigned_slots: set[tuple[int, int]] = set()

  def _assign(
      var: nnx.Variable,
      val: jax.Array,
      *,
      b_idx: int,
      is_stacked: bool,
      key_name: str,
  ) -> None:
    expected_shape = var.value.shape[1:] if is_stacked else var.value.shape
    if val.shape != expected_shape:
      raise ValueError(f"Shape mismatch for {key_name}: got {val.shape}, expected" f" {expected_shape}")
    assigned_slots.add((id(var), b_idx))
    if val.dtype != var.value.dtype:
      raise ValueError(
          f"Dtype mismatch for {key_name}: spec produced {val.dtype}, model" f" variable expects {var.value.dtype}"
      )
    if is_stacked:
      var.value = var.value.at[b_idx].set(val)
    else:
      var.value = val

  specs = iter_conversion_specs(model.config, include_mtp=not ignore_mtp_layer)

  for spec in specs:
    target_var = resolve_target_var(model, spec.target_path)
    val = _transform_for_spec(spec, _get)
    is_stacked = spec.stack_index is not None
    b_idx = 0 if spec.stack_index is None else spec.stack_index
    _assign(
        target_var,
        val,
        b_idx=b_idx,
        is_stacked=is_stacked,
        key_name=spec.hf_keys[0],
    )

  for orig_k in hf_tensors.keys():
    if orig_k.startswith("model.visual.") or orig_k.startswith("visual."):
      continue
    if orig_k not in consumed:
      clean = _normalize_key(orig_k)
      if ignore_mtp_layer and clean.startswith(f"layers.{glm5_config.MTP_LAYER_INDEX}."):
        continue
      raise ValueError(f"Unmapped key in hf_tensors: {orig_k!r}")

  def _is_var(x: Any) -> bool:
    return isinstance(x, nnx.Variable)

  num_prefix = min(model.config.num_hidden_layers, model.config.first_k_dense_replace)
  num_sb = (model.config.num_hidden_layers - num_prefix) // 4
  sb_leaves = set()
  if model.super_block is not None:
    for s in range(4):
      sl = getattr(model.super_block, f"slot_{s}")
      for v in jax.tree.leaves(nnx.state(sl), is_leaf=_is_var):
        sb_leaves.add(id(v))
  for p in jax.tree.leaves(
      nnx.state(model, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)),
      is_leaf=_is_var,
  ):
    if isinstance(p, nnx.Variable):
      for b in range(num_sb) if id(p) in sb_leaves else (0,):
        if (id(p), b) not in assigned_slots:
          raise ValueError(f"Unpopulated model parameter slot ({b}): {p}")


def save_orbax_checkpoint(model: modeling_glm5_3_flash.Glm5Model, orbax_dir: str) -> None:
  """Saves a Glm5Model checkpoint in Orbax format."""
  with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(orbax_dir, nnx.to_pure_dict(nnx.state(model)))
    checkpointer.wait_until_finished()


def restore_orbax_checkpoint(model: modeling_glm5_3_flash.Glm5Model, orbax_dir: str) -> None:
  """Restores Orbax checkpoint state into an existing Glm5Model instance."""
  target = nnx.to_pure_dict(nnx.state(model))
  with ocp.StandardCheckpointer() as checkpointer:
    restored = checkpointer.restore(orbax_dir, target)
  nnx.update(
      model,
      nnx.restore_int_paths(restored),
  )


def convert_hf_checkpoint_to_orbax(
    hf_checkpoint_dir: str,
    orbax_output_dir: str,
    config: glm5_config.Glm5Config,
    *,
    ignore_mtp_layer: bool = False,
    slice_to_config_layers: bool = False,
) -> None:
  """Converts HuggingFace FP8/BF16 safetensors checkpoint to an Orbax checkpoint.

  convert_hf_checkpoint_to_orbax is an in-memory converter scoped strictly to
  mini-slice checkpoints (num_hidden_layers <= MAX_IN_MEMORY_CONVERSION_LAYERS
  (6) and <= MAX_IN_MEMORY_CONVERSION_PARAMS (25B) parameters). One MoE layer
  is ~7.25B params, so a 5-layer real slice is ~16.9B params and fits under
  25B, whereas >= 7 layers exceeds 25B and requires
  convert_hf_checkpoint_to_orbax_streaming.
  """
  if config.num_hidden_layers > MAX_IN_MEMORY_CONVERSION_LAYERS:
    raise ValueError(
        "convert_hf_checkpoint_to_orbax is scoped to mini-slice checkpoints"
        f" with at most {MAX_IN_MEMORY_CONVERSION_LAYERS} layers (got"
        f" num_hidden_layers={config.num_hidden_layers}); full-model streaming"
        " conversion is tracked under F33."
    )
  with jax.default_device(jax.devices("cpu")[0]):
    hf_dir = hf_checkpoint_dir
    orbax_dir = orbax_output_dir
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    all_tensors: dict[str, Any] = {}
    needed: dict[str, str] | None = None

    if os.path.exists(index_path):
      with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
      if slice_to_config_layers:
        needed = create_mini_slice.extract_mini_slice_keys(
            weight_map,
            config.num_hidden_layers,
            include_mtp_layer=None if ignore_mtp_layer else glm5_config.MTP_LAYER_INDEX,
        )
        shard_paths = [os.path.join(hf_dir, s) for s in sorted(set(needed.values()))]
      else:
        shard_paths = [os.path.join(hf_dir, s) for s in sorted(set(weight_map.values()))]
    else:
      shard_paths = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
      if not shard_paths:
        raise ValueError(f"No safetensors files found in {hf_dir}")
      if slice_to_config_layers:
        temp_map: dict[str, str] = {}
        for sp in shard_paths:
          with safetensors.safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
              temp_map[k] = sp
        needed = create_mini_slice.extract_mini_slice_keys(
            temp_map,
            config.num_hidden_layers,
            include_mtp_layer=None if ignore_mtp_layer else glm5_config.MTP_LAYER_INDEX,
        )

    specs = iter_conversion_specs(config, include_mtp=not ignore_mtp_layer)
    valid_hf_keys: set[str] = set()
    for spec in specs:
      for hk in spec.hf_keys:
        valid_hf_keys.add(hk)
        valid_hf_keys.add(f"{hk[:-7]}.weight_scale_inv" if hk.endswith(".weight") else f"{hk}_scale_inv")

    # First pass: check headers and keys before materializing tensors
    total_params = 0
    for sp in shard_paths:
      with safetensors.safe_open(sp, framework="pt", device="cpu") as f:
        for k in f.keys():
          if k.startswith("model.visual.") or k.startswith("visual."):
            continue
          if needed is not None and k not in needed:
            continue
          if ignore_mtp_layer and k.startswith(f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}."):
            continue
          if not (k.startswith("model.language_model.") or k in ("lm_head.weight", "lm_head.weight_scale_inv")):
            raise ValueError(f"Unknown HF key prefix: {k!r}")
          if k not in valid_hf_keys:
            raise ValueError(f"Unmapped key in hf_tensors: {k!r}")
          header_shape = f.get_slice(k).get_shape()
          total_params += int(np.prod(header_shape))
          if total_params > MAX_IN_MEMORY_CONVERSION_PARAMS:
            raise ValueError(
                "convert_hf_checkpoint_to_orbax is scoped to mini-slice"
                f" checkpoints with at most {MAX_IN_MEMORY_CONVERSION_PARAMS}"
                f" parameters (got total_params={total_params}); full-model"
                " streaming conversion is tracked under F33."
            )

    # Second pass: materialize tensors
    for sp in shard_paths:
      with safetensors.safe_open(sp, framework="pt", device="cpu") as f:
        for k in f.keys():
          if k.startswith("model.visual.") or k.startswith("visual."):
            continue
          if needed is not None and k not in needed:
            continue
          if ignore_mtp_layer and k.startswith(f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}."):
            continue
          all_tensors[k] = f.get_tensor(k)

    model = modeling_glm5_3_flash.Glm5Model(
        config,
        rngs=nnx.Rngs(0),
        place_on_devices=False,
    )
    load_hf_state_into_model(model, all_tensors, ignore_mtp_layer=ignore_mtp_layer)
    save_orbax_checkpoint(model, orbax_dir)
