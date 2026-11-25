# Copyright 2026 Google LLC
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

"""Centralized `to_maxtext` / `to_huggingface` parameter mappings, hooks, and shapes for GraniteMoe (`m3`)."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.granitemoe import modeling_granitemoe
import ml_dtypes
import numpy as np

# `transformers` v5 in-memory module state_dict suffixes -> Hub safetensors suffixes.
_V5_TO_HUB_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".block_sparse_moe.router.weight", ".block_sparse_moe.router.layer.weight"),
    (".block_sparse_moe.experts.gate_up_proj", ".block_sparse_moe.input_linear.weight"),
    (".block_sparse_moe.experts.down_proj", ".block_sparse_moe.output_linear.weight"),
)


def _as_hf_config_dict(config: Any) -> dict[str, Any]:
  """Normalizes an HF `PreTrainedConfig`, `GraniteMoeConfig`, or mapping into a plain dict."""
  if config is None:
    return {}
  if isinstance(config, Mapping):
    raw = dict(config)
  elif dataclasses.is_dataclass(config) and not isinstance(config, type):
    raw = dataclasses.asdict(config)
  elif hasattr(config, "to_dict") and callable(config.to_dict):
    raw = dict(config.to_dict())
  else:
    raw = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
  if isinstance(raw.get("text_config"), Mapping):
    merged = dict(raw)
    merged.update(raw["text_config"])
    return merged
  return raw


def _cfg_lookup(
    maxtext_config: Any, hf_cfg: Mapping[str, Any], mt_keys: tuple[str, ...], hf_keys: tuple[str, ...], default: Any
) -> Any:
  """Resolves an architectural setting from `maxtext_config` first, falling back to `hf_cfg` and `default`."""
  if maxtext_config is not None:
    for key in mt_keys:
      if isinstance(maxtext_config, Mapping):
        if key in maxtext_config and maxtext_config[key] is not None:
          return maxtext_config[key]
      elif hasattr(maxtext_config, key):
        val = getattr(maxtext_config, key)
        if val is not None:
          return val
  for key in hf_keys:
    if key in hf_cfg and hf_cfg[key] is not None:
      return hf_cfg[key]
  return default


# pylint: disable=invalid-name
def GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING(
    config: Any,
    maxtext_config: Any = None,
    scan_layers: bool = False,
) -> dict[str, Any]:
  """Returns the MaxText-to-HuggingFace parameter mapping for `GraniteMoeModel`.

  `GraniteMoeModel` builds `self.layers` via `nnx.vmap` with leading `num_hidden_layers` axis `0`
  (`param_scan_axis=0`) identically across `scan_layers=True` and `scan_layers=False`. Each stacked
  layer parameter therefore maps to the 1-D list of per-layer HuggingFace safetensors keys.
  """
  del scan_layers
  hf_cfg = _as_hf_config_dict(config)
  num_layers = int(
      _cfg_lookup(
          maxtext_config,
          hf_cfg,
          ("num_hidden_layers", "num_decoder_layers", "base_num_decoder_layers"),
          ("num_hidden_layers",),
          24,
      )
  )
  tie_word_embeddings = bool(
      _cfg_lookup(
          maxtext_config,
          hf_cfg,
          ("tie_word_embeddings", "logits_via_embedding"),
          ("tie_word_embeddings",),
          True,
      )
  )
  attention_bias = bool(_cfg_lookup(maxtext_config, hf_cfg, ("attention_bias",), ("attention_bias",), False))

  mapping: dict[str, Any] = {
      "params-embed_tokens-embedding": "model.embed_tokens.weight",
      "params-norm-weight": "model.norm.weight",
      "params-layers-input_layernorm-weight": [f"model.layers.{i}.input_layernorm.weight" for i in range(num_layers)],
      "params-layers-post_attention_layernorm-weight": [
          f"model.layers.{i}.post_attention_layernorm.weight" for i in range(num_layers)
      ],
      "params-layers-self_attn-q_proj-kernel": [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(num_layers)],
      "params-layers-self_attn-k_proj-kernel": [f"model.layers.{i}.self_attn.k_proj.weight" for i in range(num_layers)],
      "params-layers-self_attn-v_proj-kernel": [f"model.layers.{i}.self_attn.v_proj.weight" for i in range(num_layers)],
      "params-layers-self_attn-o_proj-kernel": [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(num_layers)],
      "params-layers-block_sparse_moe-router-weight": [
          f"model.layers.{i}.block_sparse_moe.router.layer.weight" for i in range(num_layers)
      ],
      "params-layers-block_sparse_moe-experts-gate_up_proj": [
          f"model.layers.{i}.block_sparse_moe.input_linear.weight" for i in range(num_layers)
      ],
      "params-layers-block_sparse_moe-experts-down_proj": [
          f"model.layers.{i}.block_sparse_moe.output_linear.weight" for i in range(num_layers)
      ],
  }
  if attention_bias:
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
      mapping[f"params-layers-self_attn-{proj}-bias"] = [
          f"model.layers.{i}.self_attn.{proj}.bias" for i in range(num_layers)
      ]
  if not tie_word_embeddings:
    mapping["params-lm_head-kernel"] = "lm_head.weight"
  return mapping


def GRANITEMOE_MAXTEXT_TO_HF_PARAM_HOOK_FN(
    config: Any,
    maxtext_config: Any = None,
    scan_layers: bool = False,
    saving_to_hf: bool = False,
) -> dict[str, Any]:
  """Returns bidirectional tensor transformation hooks for `GraniteMoeModel`."""
  del scan_layers
  hf_cfg = _as_hf_config_dict(config)
  tie_word_embeddings = bool(
      _cfg_lookup(
          maxtext_config,
          hf_cfg,
          ("tie_word_embeddings", "logits_via_embedding"),
          ("tie_word_embeddings",),
          True,
      )
  )

  def transpose_2d(input_tensor: Any, target_shape: Any = None) -> Any:
    transposed = input_tensor.T
    return transposed.reshape(target_shape) if target_shape is not None else transposed

  def reshape_gate_up(input_tensor: Any, target_shape: Any = None) -> Any:
    del target_shape
    if saving_to_hf:
      # M3 [E, H, 2, I] -> [E, H, 2*I] -> HF [E, 2*I, H]
      n_exp, h_dim, two, i_dim = input_tensor.shape
      return input_tensor.reshape(n_exp, h_dim, two * i_dim).transpose(0, 2, 1)
    # HF [E, 2*I, H] -> [E, H, 2*I] -> M3 [E, H, 2, I] (0=gate, 1=up)
    n_exp, two_i, h_dim = input_tensor.shape
    return input_tensor.transpose(0, 2, 1).reshape(n_exp, h_dim, 2, two_i // 2)

  def transpose_expert_down(input_tensor: Any, target_shape: Any = None) -> Any:
    del target_shape
    # HF [E, H, I] <-> M3 [E, I, H]
    return input_tensor.transpose(0, 2, 1)

  hooks: dict[str, Any] = {
      "params-layers-self_attn-q_proj-kernel": transpose_2d,
      "params-layers-self_attn-k_proj-kernel": transpose_2d,
      "params-layers-self_attn-v_proj-kernel": transpose_2d,
      "params-layers-self_attn-o_proj-kernel": transpose_2d,
      "params-layers-block_sparse_moe-router-weight": transpose_2d,
      "params-layers-block_sparse_moe-experts-gate_up_proj": reshape_gate_up,
      "params-layers-block_sparse_moe-experts-down_proj": transpose_expert_down,
  }
  if not tie_word_embeddings:
    hooks["params-lm_head-kernel"] = transpose_2d
  return hooks


def GRANITEMOE_HF_WEIGHTS_TO_SHAPE(config: Any) -> dict[str, list[int]]:
  """Returns `{hf_param_key: shape_list}` for HuggingFace `GraniteMoeForCausalLM`."""
  hf_cfg = _as_hf_config_dict(config)
  vocab_size = int(hf_cfg["vocab_size"])
  hidden_size = int(hf_cfg["hidden_size"])
  intermediate_size = int(hf_cfg["intermediate_size"])
  num_layers = int(hf_cfg["num_hidden_layers"])
  num_q_heads = int(hf_cfg["num_attention_heads"])
  num_kv_heads = int(hf_cfg["num_key_value_heads"])
  head_dim = int(hf_cfg.get("head_dim") or (hidden_size // num_q_heads))
  num_experts = int(hf_cfg.get("num_local_experts", hf_cfg.get("num_experts", 32)))
  attention_bias = bool(hf_cfg.get("attention_bias", False))
  tie_word_embeddings = bool(hf_cfg.get("tie_word_embeddings", True))

  q_dim = num_q_heads * head_dim
  kv_dim = num_kv_heads * head_dim

  shapes: dict[str, list[int]] = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
  }
  if not tie_word_embeddings:
    shapes["lm_head.weight"] = [vocab_size, hidden_size]

  for i in range(num_layers):
    pfx = f"model.layers.{i}"
    shapes[f"{pfx}.input_layernorm.weight"] = [hidden_size]
    shapes[f"{pfx}.post_attention_layernorm.weight"] = [hidden_size]
    shapes[f"{pfx}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    shapes[f"{pfx}.self_attn.k_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{pfx}.self_attn.v_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{pfx}.self_attn.o_proj.weight"] = [hidden_size, q_dim]
    if attention_bias:
      shapes[f"{pfx}.self_attn.q_proj.bias"] = [q_dim]
      shapes[f"{pfx}.self_attn.k_proj.bias"] = [kv_dim]
      shapes[f"{pfx}.self_attn.v_proj.bias"] = [kv_dim]
      shapes[f"{pfx}.self_attn.o_proj.bias"] = [hidden_size]
    shapes[f"{pfx}.block_sparse_moe.router.layer.weight"] = [num_experts, hidden_size]
    shapes[f"{pfx}.block_sparse_moe.input_linear.weight"] = [num_experts, 2 * intermediate_size, hidden_size]
    shapes[f"{pfx}.block_sparse_moe.output_linear.weight"] = [num_experts, hidden_size, intermediate_size]
  return shapes


# pylint: enable=invalid-name


# ==============================================================================
# Table-Driven In-Memory Weight Helpers (Shared by Unit & Parity Tests)
# ==============================================================================


def _tensor_to_numpy(val: Any, target_dtype: np.dtype | None = None) -> np.ndarray:
  """Converts a `jax.Array`, `np.ndarray`, or PyTorch `Tensor` (including `bf16`) to `np.ndarray`."""
  if hasattr(val, "detach") and hasattr(val, "cpu"):
    t_cpu = val.detach().cpu().contiguous()
    if "bfloat16" in str(t_cpu.dtype):
      arr = t_cpu.float().numpy().astype(ml_dtypes.bfloat16)
    else:
      arr = t_cpu.numpy()
  else:
    arr = np.asarray(val)

  if target_dtype is not None and arr.dtype != target_dtype:
    return arr.astype(target_dtype)
  return np.array(arr, copy=True)


def normalize_granitemoe_hf_keys(hf_tensors: Mapping[str, Any]) -> dict[str, Any]:
  """Maps `transformers` v5 in-memory `GraniteMoe` parameter names to Hub safetensors names."""
  normalized: dict[str, Any] = {}
  for key, value in hf_tensors.items():
    mapped_key = key
    for v5_sfx, hub_sfx in _V5_TO_HUB_SUFFIXES:
      if key.endswith(v5_sfx):
        mapped_key = key[: -len(v5_sfx)] + hub_sfx
        break
    if mapped_key in normalized:
      raise ValueError(f"Duplicate GraniteMoe HF key {mapped_key!r} after alias normalization.")
    normalized[mapped_key] = value
  return normalized


def _flatten_pure_dict(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
  """Flattens a nested pure dict into `'a.b.c': array` entries."""
  flat: dict[str, Any] = {}
  for k, v in tree.items():
    path = f"{prefix}.{k}" if prefix else str(k)
    if isinstance(v, Mapping):
      flat.update(_flatten_pure_dict(v, path))
    else:
      flat[path] = v
  return flat


def hf_weights_to_m3_pure_dict(
    hf_tensors: Mapping[str, Any],
    config: Any,
    *,
    target_dtype: Any = None,
) -> dict[str, Any]:
  """Applies `GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING` and hooks in memory to build an M3 pure param dict."""
  gm_cfg = modeling_granitemoe.GraniteMoeConfig.from_config(config)
  tgt_dt = (
      np.dtype(modeling_granitemoe._resolve_jnp_dtype(target_dtype))  # pylint: disable=protected-access
      if target_dtype is not None
      else None
  )
  normalized_hf = normalize_granitemoe_hf_keys(hf_tensors)
  pool: dict[str, np.ndarray] = {k: _tensor_to_numpy(v, tgt_dt) for k, v in normalized_hf.items()}

  hf_cfg_dict = dataclasses.asdict(gm_cfg)
  mapping = GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, gm_cfg, scan_layers=gm_cfg.scan_layers)
  hooks = GRANITEMOE_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, gm_cfg, scan_layers=gm_cfg.scan_layers, saving_to_hf=False)
  hf_shapes = GRANITEMOE_HF_WEIGHTS_TO_SHAPE(hf_cfg_dict)

  if gm_cfg.tie_word_embeddings and "lm_head.weight" in pool and "model.embed_tokens.weight" in pool:
    tied_lm = pool.pop("lm_head.weight")
    embed_ref = pool["model.embed_tokens.weight"]
    if not np.array_equal(tied_lm.astype(np.float32), embed_ref.astype(np.float32)):
      raise ValueError("tie_word_embeddings=True but HF lm_head.weight differs from model.embed_tokens.weight.")

  def _pop_and_convert(hf_key: str, hook_fn: Any) -> np.ndarray:
    if hf_key not in pool:
      raise KeyError(f"Missing required HF checkpoint key {hf_key!r}.")
    arr = pool.pop(hf_key)
    expected = tuple(hf_shapes[hf_key])
    if arr.shape != expected:
      raise ValueError(f"HF tensor {hf_key!r} shape mismatch: got {arr.shape}, expected {expected}.")
    return np.ascontiguousarray(hook_fn(arr, None) if hook_fn is not None else arr)

  pure_dict: dict[str, Any] = {}
  for mt_key, hf_val in mapping.items():
    hook_fn = hooks.get(mt_key)
    if isinstance(hf_val, list):
      converted = np.stack([_pop_and_convert(k, hook_fn) for k in hf_val], axis=0)
    else:
      converted = _pop_and_convert(hf_val, hook_fn)

    parts = mt_key.removeprefix("params-").split("-")
    cursor = pure_dict
    for part in parts[:-1]:
      cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = converted

  if pool:
    unmapped = sorted(pool.keys())
    raise ValueError(f"Unmapped HF checkpoint keys ({len(unmapped)}): {unmapped}")
  return pure_dict


def m3_pure_dict_to_hf_weights(
    pure_dict: Mapping[str, Any],
    config: Any,
    *,
    naming: str = "hub",
) -> dict[str, np.ndarray]:
  """Applies `GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING` and `saving_to_hf=True` hooks in memory."""
  if naming not in ("hub", "v5"):
    raise ValueError(f"naming must be 'hub' or 'v5', got {naming!r}.")
  gm_cfg = modeling_granitemoe.GraniteMoeConfig.from_config(config)
  hf_cfg_dict = dataclasses.asdict(gm_cfg)
  mapping = GRANITEMOE_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, gm_cfg, scan_layers=gm_cfg.scan_layers)
  hooks = GRANITEMOE_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, gm_cfg, scan_layers=gm_cfg.scan_layers, saving_to_hf=True)
  hf_shapes = GRANITEMOE_HF_WEIGHTS_TO_SHAPE(hf_cfg_dict)
  hub_to_v5 = {hub_sfx: v5_sfx for v5_sfx, hub_sfx in _V5_TO_HUB_SUFFIXES}

  def _maybe_alias(hf_key: str) -> str:
    if naming == "hub":
      return hf_key
    for hub_sfx, v5_sfx in hub_to_v5.items():
      if hf_key.endswith(hub_sfx):
        return hf_key[: -len(hub_sfx)] + v5_sfx
    return hf_key

  hf_out: dict[str, np.ndarray] = {}
  for mt_key, hf_val in mapping.items():
    parts = mt_key.removeprefix("params-").split("-")
    leaf: Any = pure_dict
    for part in parts:
      leaf = leaf[part]
    arr = _tensor_to_numpy(leaf)
    hook_fn = hooks.get(mt_key)

    if isinstance(hf_val, list):
      for idx, hf_k in enumerate(hf_val):
        tgt_shape = tuple(hf_shapes[hf_k])
        transformed = hook_fn(arr[idx], tgt_shape) if hook_fn is not None else arr[idx]
        hf_out[_maybe_alias(hf_k)] = np.ascontiguousarray(transformed)
    else:
      tgt_shape = tuple(hf_shapes[hf_val])
      transformed = hook_fn(arr, tgt_shape) if hook_fn is not None else arr
      hf_out[_maybe_alias(hf_val)] = np.ascontiguousarray(transformed)
  return hf_out


def load_pure_dict_into_model(model: modeling_granitemoe.GraniteMoeModel, pure_dict: Mapping[str, Any]) -> None:
  """Populates every `nnx.Param` leaf of `model` from `pure_dict`, preserving leaf sharding."""
  param_state = nnx.state(model, nnx.Param)
  expected_flat = _flatten_pure_dict(nnx.to_pure_dict(param_state))
  provided_flat = _flatten_pure_dict(pure_dict)

  missing = sorted(set(expected_flat) - set(provided_flat))
  extra = sorted(set(provided_flat) - set(expected_flat))
  if missing or extra:
    raise ValueError(f"M3 Param leaf mismatch: missing={missing}, extra={extra}.")

  def _assign_leaf(path: tuple[Any, ...], var: Any) -> Any:
    if not isinstance(var, nnx.Variable):
      return var
    key = ".".join(str(getattr(p, "key", p)) for p in path)
    src_np = provided_flat[key]
    cur_arr = var.get_raw_value()
    if src_np.shape != cur_arr.shape:
      raise ValueError(f"Leaf {key!r} shape mismatch: got {src_np.shape}, expected {cur_arr.shape}.")
    placed = jnp.asarray(src_np, dtype=cur_arr.dtype)
    if isinstance(cur_arr, jax.Array) and isinstance(cur_arr.sharding, jax.sharding.NamedSharding):
      placed = jax.device_put(placed, cur_arr.sharding)
    return var.replace(placed)

  updated = jax.tree_util.tree_map_with_path(
      _assign_leaf,
      param_state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
  nnx.update(model, updated)


def load_hf_weights_into_model(
    model: modeling_granitemoe.GraniteMoeModel,
    hf_tensors: Mapping[str, Any],
) -> int:
  """Loads an in-memory HF `state_dict` into `model` via the centralized GraniteMoe conversion tables."""
  pure_dict = hf_weights_to_m3_pure_dict(hf_tensors, model.config, target_dtype=model.config.weight_dtype)
  load_pure_dict_into_model(model, pure_dict)
  return len(hf_tensors)


def export_model_to_hf_weights(
    model: modeling_granitemoe.GraniteMoeModel,
    *,
    naming: str = "hub",
) -> dict[str, np.ndarray]:
  """Exports `model` parameters to an in-memory HF dictionary via the centralized GraniteMoe conversion tables."""
  pure_dict = nnx.to_pure_dict(nnx.state(model, nnx.Param))
  return m3_pure_dict_to_hf_weights(pure_dict, model.config, naming=naming)
