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

import argparse
import collections
from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import dataclasses
import functools
import json
import math
import os
import sys
import threading
import time
from typing import Any
from etils import epath
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
import tensorstore as ts
import torch

MAX_IN_MEMORY_CONVERSION_LAYERS = 6
MAX_IN_MEMORY_CONVERSION_PARAMS = 25_000_000_000
_MAX_CONCURRENT_OPEN_SHARDS = 2
# math.gcd(_MAX_MOE_EXPERT_CHUNK_SIZE, num_experts) ensures expert_chunk evenly
# divides num_experts so TensorStore writes full Zarr chunks without
# partial-chunk read-modify-write.
_MAX_MOE_EXPERT_CHUNK_SIZE = 16
_STANDARD_CHECKPOINT_HANDLER = "orbax.checkpoint._src.handlers.standard_checkpoint_handler" ".StandardCheckpointHandler"
_BYTE_SIZE_SUFFIXES = {
    "GiB": 1024**3,
    "GB": 1000**3,
    "MiB": 1024**2,
    "MB": 1000**2,
    "KiB": 1024,
    "KB": 1000,
    "B": 1,
}


def _parse_byte_size(val: str, flag_name: str) -> int:
  """Parses a byte size string with optional suffixes into an integer."""
  if not isinstance(val, str):
    raise ValueError(f"Invalid {flag_name} value: expected string, got {type(val)}")
  val = val.strip()
  for suffix, multiplier in sorted(_BYTE_SIZE_SUFFIXES.items(), key=lambda kv: len(kv[0]), reverse=True):
    if val.endswith(suffix):
      num_str = val[: -len(suffix)].strip()
      if suffix == "B" and any(c.isalpha() for c in num_str):
        raise ValueError(f"Invalid format for {flag_name} (unknown unit suffix): {val!r}")
      try:
        total_bytes = int(num_str) * multiplier
      except ValueError as e:
        raise ValueError(f"Invalid numeric part in {flag_name}: {num_str}") from e
      if total_bytes < 1:
        raise ValueError(f"{flag_name} must be >= 1, got {total_bytes}")
      return total_bytes
  if any(c.isalpha() for c in val):
    raise ValueError(f"Invalid format for {flag_name} (unknown unit suffix): {val!r}")
  try:
    total_bytes = int(val)
  except ValueError as e:
    raise ValueError(f"Invalid format for {flag_name}: {val}") from e
  if total_bytes < 1:
    raise ValueError(f"{flag_name} must be >= 1, got {total_bytes}")
  return total_bytes


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

_SUPPORTED_CLI_DTYPES: frozenset[str] = frozenset(dt.name for dt in _TARGET_TO_INTERMEDIATE_DTYPE)
_CLI_OWNED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "num_hidden_layers",
        "num_nextn_predict_layers",
        "weight_dtype",
    }
)
_ALLOWED_CONFIG_JSON_KEYS: frozenset[str] = (
    frozenset(f.name for f in dataclasses.fields(glm5_config.Glm5Config)) - _CLI_OWNED_CONFIG_KEYS
)
_TUPLE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "layer_types",
        "indexer_types",
    }
)


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
  w_np *= scale
  return np.asarray(w_np, dtype=target_np_dt)


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


class _MemoryBudgetSemaphore:
  """Thread-safe memory budget semaphore tracking active inflight bytes."""

  def __init__(self, max_bytes: int):
    if max_bytes <= 0:
      raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    self._max_bytes = max_bytes
    self._current_bytes = 0
    self._peak_bytes = 0
    self._active_acquisitions = 0
    self._max_concurrent_acquisitions = 0
    self._cond = threading.Condition()

  def acquire(self, required_bytes: int) -> None:
    """Acquires required_bytes from the memory budget, blocking until available."""
    if required_bytes <= 0:
      raise ValueError(f"required_bytes must be > 0, got {required_bytes}")
    if required_bytes > self._max_bytes:
      raise ValueError(
          f"Required work unit memory {required_bytes} B exceeds total memory" f" budget {self._max_bytes} B."
      )
    with self._cond:
      while self._current_bytes + required_bytes > self._max_bytes:
        self._cond.wait()
      self._current_bytes += required_bytes
      self._peak_bytes = max(self._peak_bytes, self._current_bytes)
      self._active_acquisitions += 1
      self._max_concurrent_acquisitions = max(self._max_concurrent_acquisitions, self._active_acquisitions)

  def release(self, freed_bytes: int) -> None:
    with self._cond:
      self._current_bytes -= freed_bytes
      self._active_acquisitions -= 1
      self._cond.notify_all()

  @property
  def current_bytes(self) -> int:
    with self._cond:
      return self._current_bytes

  @property
  def peak_bytes(self) -> int:
    with self._cond:
      return self._peak_bytes

  @property
  def max_concurrent_acquisitions(self) -> int:
    with self._cond:
      return self._max_concurrent_acquisitions


class _ThreadSafeShardLRU:
  """Thread-safe LRU handle cache for safetensors shards with reference leases."""

  def __init__(self, max_open: int):
    if max_open <= 0:
      raise ValueError(f"max_open must be positive, got {max_open}")
    self._max_open = max_open
    self._handles: collections.OrderedDict[str, Any] = collections.OrderedDict()
    self._leases: collections.Counter[str] = collections.Counter()
    self._opening: set[str] = set()
    self._cond = threading.Condition()

  def _acquire_shard(self, shard_path: str) -> Any:
    """Acquires a shard handle under reference lease."""
    while True:
      with self._cond:
        if shard_path in self._handles:
          self._handles.move_to_end(shard_path)
          self._leases[shard_path] += 1
          return self._handles[shard_path]
        if shard_path in self._opening:
          self._cond.wait()
          continue
        if len(self._handles) + len(self._opening) < self._max_open:
          self._opening.add(shard_path)
          break
        evicted = False
        for sp in list(self._handles.keys()):
          if self._leases[sp] == 0:
            h = self._handles.pop(sp)
            # pylint: disable=unnecessary-dunder-call
            h.__exit__(None, None, None)
            del self._leases[sp]
            evicted = True
            break
        if evicted:
          self._opening.add(shard_path)
          break
        self._cond.wait()

    try:
      handle = safetensors.safe_open(shard_path, framework="pt", device="cpu")
      # pylint: disable=unnecessary-dunder-call
      handle.__enter__()
    except BaseException:
      with self._cond:
        self._opening.discard(shard_path)
        self._cond.notify_all()
      raise

    with self._cond:
      self._opening.discard(shard_path)
      self._handles[shard_path] = handle
      self._leases[shard_path] = 1
      self._cond.notify_all()
      return handle

  def _release_shard(self, shard_path: str) -> None:
    with self._cond:
      self._leases[shard_path] -= 1
      if self._leases[shard_path] <= 0:
        self._leases[shard_path] = 0
        self._cond.notify_all()

  def load_tensor(self, shard_path: str, orig_k: str, shard_file: str) -> Any:
    """Loads a tensor from a safetensors shard under concurrency-bounded LRU lease."""
    handle = self._acquire_shard(shard_path)
    try:
      return handle.get_tensor(orig_k)
    except (safetensors.SafetensorError, KeyError) as e:
      raise ValueError(f"Tensor {orig_k!r} listed in index for shard {shard_file!r} is" " absent from shard file.") from e
    finally:
      self._release_shard(shard_path)

  def get_slice_shape(self, shard_path: str, orig_k: str, shard_file: str) -> tuple[int, ...]:
    """Inspects a tensor slice shape from a safetensors shard header under LRU lease."""
    handle = self._acquire_shard(shard_path)
    try:
      s = handle.get_slice(orig_k)
      return tuple(int(x) for x in s.get_shape())
    except (safetensors.SafetensorError, KeyError) as e:
      raise ValueError(f"Tensor {orig_k!r} listed in index for shard {shard_file!r} is" " absent from shard file.") from e
    finally:
      self._release_shard(shard_path)

  def close_all(self) -> None:
    """Closes all active and cached shard handles."""
    with self._cond:
      while self._handles:
        _, h = self._handles.popitem(last=False)
        # pylint: disable=unnecessary-dunder-call
        h.__exit__(None, None, None)
      self._leases.clear()
      self._opening.clear()
      self._cond.notify_all()


@dataclasses.dataclass(frozen=True)
class ConversionSpec:
  """Specification for mapping HuggingFace tensor(s) to a MaxText model parameter."""

  hf_keys: tuple[str, ...]
  target_path: tuple[str | int, ...]
  stack_index: int | None
  num_super_blocks: int | None
  kind: str  # "direct" | "transpose_2d" | "moe_gate_up" | "moe_down"
  target_dtype: Any  # jnp.dtype


@dataclasses.dataclass(frozen=True)
class _ConversionWorkUnit:
  """Metadata for a parallel conversion work unit."""

  sort_key: tuple[str, int, int]
  spec: ConversionSpec
  slot: int
  e_start: int
  e_end: int
  byte_weight: int


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
      stack_idx: int | None,
      num_sbs: int | None,
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
      stack_idx=None,
      num_sbs=None,
  )
  _add(
      ("model.language_model.norm.weight",),
      ("norm", "weight"),
      "direct",
      dtype=p_dt,
      stack_idx=None,
      num_sbs=None,
  )
  _add(
      ("lm_head.weight",),
      ("lm_head", "kernel"),
      "transpose_2d",
      dtype=p_dt,
      stack_idx=None,
      num_sbs=None,
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
          stack_idx=None,
          num_sbs=None,
      )
    _add(
        (f"{lp}eh_proj.weight",),
        (*tgt_base, "eh_proj", "kernel"),
        "transpose_2d",
        dtype=p_dt,
        stack_idx=None,
        num_sbs=None,
    )
    _add(
        (f"{lp}shared_head.norm.weight",),
        (*tgt_base, "shared_head_norm", "weight"),
        "direct",
        dtype=p_dt,
        stack_idx=None,
        num_sbs=None,
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
      if not 0 <= b_idx < var.value.shape[0]:
        raise ValueError(f"b_idx {b_idx} out of bounds [0, {var.value.shape[0]}) for stacked" f" {key_name}")
      var.value = var.value.at[b_idx].set(val)
    else:
      if b_idx != 0:
        raise ValueError(f"b_idx must be 0 when is_stacked=False for {key_name}, got {b_idx}")
      var.value = val

  specs = iter_conversion_specs(model.config, include_mtp=not ignore_mtp_layer)

  all_vars: list[tuple[tuple[Any, ...], nnx.Variable]] = []
  sb_leaves: set[int] = set()
  unstacked_leaves: set[int] = set()
  for path, node in nnx.iter_graph(model):
    if isinstance(node, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias)):
      var_id = id(node)
      all_vars.append((path, node))
      if path and path[0] == "super_block":
        sb_leaves.add(var_id)
      else:
        unstacked_leaves.add(var_id)

  if not sb_leaves.isdisjoint(unstacked_leaves) or (len(sb_leaves) + len(unstacked_leaves) != len(all_vars)):
    raise ValueError("Model variable leaf classification mismatch between stacked and" " unstacked sets.")

  for spec in specs:
    arr = _transform_for_spec(spec, _get)
    target_var = resolve_target_var(model, spec.target_path)
    var_id = id(target_var)
    if (var_id in sb_leaves) == (var_id in unstacked_leaves):
      raise ValueError(
          f"Target variable {spec.target_path} was not classified into" " exactly one of stacked or unstacked leaves."
      )
    is_stacked = var_id in sb_leaves
    if is_stacked and spec.stack_index is None:
      raise ValueError(
          f"Stacked parameter {'.'.join(str(p) for p in spec.target_path)}" " requires a non-None stack_index"
      )
    b_idx = 0 if spec.stack_index is None else spec.stack_index
    _assign(
        target_var,
        arr,
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

  num_prefix = min(model.config.num_hidden_layers, model.config.first_k_dense_replace)
  num_sb = (model.config.num_hidden_layers - num_prefix) // 4
  for path, p in all_vars:
    for b in range(num_sb) if id(p) in sb_leaves else (0,):
      if (id(p), b) not in assigned_slots:
        raise ValueError(f"Unpopulated model parameter slot ({b}) at" f" {'.'.join(str(k) for k in path)}: {p}")


def save_orbax_checkpoint(model: modeling_glm5_3_flash.Glm5Model, orbax_dir: str) -> None:
  """Saves a Glm5Model checkpoint in Orbax format."""
  with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(orbax_dir, nnx.to_pure_dict(nnx.state(model)))
    checkpointer.wait_until_finished()


def _to_str_paths(tree: Any) -> Any:
  """Converts dict keys to strings so target PyTree matches Orbax on-disk keys."""
  if isinstance(tree, dict):
    return {str(k): _to_str_paths(v) for k, v in tree.items()}
  return tree


def restore_orbax_checkpoint(
    model: Any,
    orbax_dir: str,
    *,
    mesh: jax.sharding.Mesh | None,
) -> None:
  """Restores Orbax checkpoint state directly into sharded model variables.

  Args:
    model: Glm5Model instance (or nnx.eval_shape abstract model) to restore
      into.
    orbax_dir: Directory containing the Orbax checkpoint.
    mesh: Must be None. Passing a mesh to restore_orbax_checkpoint is rejected;
      pass mesh to Glm5Model(..., mesh=mesh) at construction time so
      model.sharding is populated.

  Raises:
    ValueError: If mesh is not None, if model.sharding is set but model.mesh is
      None or empty, if checkpoint metadata is missing, or if leaf dtype/shape
      mismatches.
    TypeError: If variable value or sharding metadata has an unexpected type.
  """
  var_state = nnx.state(model)
  pure_target = nnx.to_pure_dict(var_state)
  str_pure_target = _to_str_paths(pure_target)

  with ocp.StandardCheckpointer() as checkpointer:
    step_metadata = checkpointer.metadata(orbax_dir)
    if step_metadata.item_metadata is None:
      raise ValueError(f"Missing item_metadata in Orbax checkpoint at {orbax_dir!r}.")
    meta_tree = step_metadata.item_metadata.tree

    def _validate_meta_leaf(meta_leaf: Any, target_leaf: Any) -> None:
      if meta_leaf is None or not isinstance(meta_leaf, ocp.metadata.value.ArrayMetadata):
        raise ValueError(f"Expected Orbax ArrayMetadata leaf, got {type(meta_leaf)}")
      saved_dt = meta_leaf.dtype
      probe = jnp.asarray(np.empty(0, dtype=saved_dt))
      if np.dtype(probe.dtype) != np.dtype(saved_dt):
        raise ValueError(f"Restored leaf dtype changed during conversion: saved {saved_dt}," f" got {probe.dtype}")
      if np.dtype(probe.dtype) != np.dtype(target_leaf.dtype):
        raise ValueError(f"Restored leaf dtype {probe.dtype} does not match model variable" f" dtype {target_leaf.dtype}")
      if tuple(meta_leaf.shape) != tuple(target_leaf.shape):
        raise ValueError(
            f"Restored leaf shape {tuple(meta_leaf.shape)} does not match model"
            f" variable shape {tuple(target_leaf.shape)}"
        )

    jax.tree.map(_validate_meta_leaf, meta_tree, str_pure_target)

    if mesh is not None:
      raise ValueError(
          "Passing mesh to restore_orbax_checkpoint is not supported; pass"
          " mesh to Glm5Model(..., mesh=mesh) at construction time so"
          " model.sharding is populated."
      )

    if model.sharding is not None:
      model_mesh = model.mesh
      if model_mesh is None or model_mesh.empty:
        raise ValueError("model.sharding is set, but model.mesh is None or empty.")

      def _sharding_from_model_metadata(var: nnx.Variable) -> nnx.Variable:
        if not var.has_metadata("out_sharding") or (var.get_metadata("out_sharding") is None):
          raise ValueError(f"Missing out_sharding metadata on variable {var!r} when" " model.sharding is set.")
        out_sh = var.get_metadata("out_sharding")
        if isinstance(out_sh, jax.sharding.NamedSharding):
          spec = out_sh.spec
        elif isinstance(out_sh, jax.sharding.PartitionSpec):
          spec = out_sh
        else:
          raise TypeError(f"Unsupported out_sharding metadata type: {type(out_sh)}")
        return var.replace(jax.sharding.NamedSharding(model_mesh, spec))

      named_sharding_state = jax.tree.map(
          _sharding_from_model_metadata,
          var_state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
    else:
      for leaf in jax.tree.leaves(pure_target):
        if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and isinstance(leaf.sharding, jax.sharding.NamedSharding):
          raise ValueError(
              "mesh=None with model.sharding=None cannot be used when model" " leaves already carry NamedSharding."
          )
      single_sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])
      named_sharding_state = jax.tree.map(
          lambda var: var.replace(single_sharding),
          var_state,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )

    sharding_pure = _to_str_paths(nnx.to_pure_dict(named_sharding_state))
    target_struct_tree = jax.tree.map(
        lambda leaf, sh: jax.ShapeDtypeStruct(shape=leaf.shape, dtype=leaf.dtype, sharding=sh),
        str_pure_target,
        sharding_pure,
    )
    restored = checkpointer.restore(orbax_dir, target=target_struct_tree)
  nnx.update(model, nnx.restore_int_paths(restored))


def _discover_and_validate_checkpoint(
    hf_checkpoint_dir: str,
    config: glm5_config.Glm5Config,
    *,
    slice_to_config_layers: bool,
    ignore_mtp_layer: bool,
) -> tuple[dict[str, str], list[str]]:
  """Discovers and validates safetensors checkpoint files via intent-keyed discovery.

  Args:
    hf_checkpoint_dir: Directory containing index or single-shard safetensors.
    config: Model configuration.
    slice_to_config_layers: Whether to filter keys to config.num_hidden_layers.
    ignore_mtp_layer: Whether to exclude MTP layer 46 keys.

  Returns:
    Tuple of (active_weight_map, sorted_shard_paths).
  Raises:
    FileNotFoundError: If checkpoint files or referenced shards are missing.
    ValueError: If model.safetensors.index.json is missing 'weight_map'.
  """
  idx_p = os.path.join(hf_checkpoint_dir, "model.safetensors.index.json")
  single_p = os.path.join(hf_checkpoint_dir, "model.safetensors")
  if os.path.isfile(idx_p):
    with open(idx_p, "r", encoding="utf-8") as f:
      idx_data = json.load(f)
    if not (isinstance(idx_data, Mapping) and "weight_map" in idx_data and isinstance(idx_data["weight_map"], Mapping)):
      raise ValueError(f"Invalid safetensors index in {idx_p!r}: missing 'weight_map'" " mapping.")
    weight_map: dict[str, str] = {str(k): str(v) for k, v in idx_data["weight_map"].items()}
  elif os.path.isfile(single_p):
    st_files = [f for f in os.listdir(hf_checkpoint_dir) if f.endswith(".safetensors")]
    if len(st_files) > 1:
      raise FileNotFoundError(
          "Multiple safetensors files found without"
          f" model.safetensors.index.json in {hf_checkpoint_dir!r}:"
          f" {sorted(st_files)}"
      )
    with safetensors.safe_open(single_p, framework="pt", device="cpu") as f:
      weight_map = {k: "model.safetensors" for k in f.keys()}
  else:
    contents = sorted(os.listdir(hf_checkpoint_dir)) if os.path.isdir(hf_checkpoint_dir) else "directory does not exist"
    raise FileNotFoundError(
        "Expected model.safetensors.index.json or single-file"
        f" model.safetensors in {hf_checkpoint_dir!r}; found: {contents}"
    )

  mtp_idx = None if ignore_mtp_layer else glm5_config.MTP_LAYER_INDEX
  active_map: dict[str, str] = (
      create_mini_slice.extract_mini_slice_keys(weight_map, config.num_hidden_layers, include_mtp_layer=mtp_idx)
      if slice_to_config_layers
      else weight_map
  )
  unique_shards = sorted(set(active_map.values()))
  missing = [s for s in unique_shards if not os.path.isfile(os.path.join(hf_checkpoint_dir, s))]
  if missing:
    raise FileNotFoundError(f"Missing referenced safetensors shards in {hf_checkpoint_dir!r}:" f" {missing}")
  return active_map, [os.path.join(hf_checkpoint_dir, s) for s in unique_shards]


def _extract_valid_hf_keys(specs: list[ConversionSpec]) -> set[str]:
  return {
      k
      for s in specs
      for hk in s.hf_keys
      for k in (
          hk,
          f"{hk[:-7]}.weight_scale_inv" if hk.endswith(".weight") else f"{hk}_scale_inv",
      )
  }


def _fill_moe_expert_chunk(
    spec: ConversionSpec,
    e_start: int,
    e_end: int,
    per_expert_shape: tuple[int, ...],
    dt: np.dtype,
    get_tensor_fn: Callable[[str, Any], np.ndarray],
) -> np.ndarray:
  """Fills an MoE expert chunk buffer for gate_up or down projection."""
  chunk_buf = np.empty((e_end - e_start, *per_expert_shape), dtype=dt)
  if spec.kind == "moe_gate_up":
    expected_proj_shape = (per_expert_shape[0], per_expert_shape[2])
    for e in range(e_start, e_end):
      gate_t = get_tensor_fn(spec.hf_keys[2 * e], spec.target_dtype).T
      up_t = get_tensor_fn(spec.hf_keys[2 * e + 1], spec.target_dtype).T
      if gate_t.shape != expected_proj_shape or up_t.shape != expected_proj_shape:
        raise ValueError(
            f"Shape mismatch for {spec.target_path} expert {e}:"
            f" expected {expected_proj_shape}, got gate={gate_t.shape},"
            f" up={up_t.shape}."
        )
      chunk_buf[e - e_start, :, 0, :] = gate_t
      chunk_buf[e - e_start, :, 1, :] = up_t
      del gate_t, up_t
  elif spec.kind == "moe_down":
    for e in range(e_start, e_end):
      down_t = get_tensor_fn(spec.hf_keys[e], spec.target_dtype).T
      if down_t.shape != per_expert_shape:
        raise ValueError(
            f"Shape mismatch for {spec.target_path} expert {e}:" f" expected {per_expert_shape}, got {down_t.shape}."
        )
      chunk_buf[e - e_start] = down_t
      del down_t
  else:
    raise ValueError(f"Unknown conversion spec kind: {spec.kind!r}")
  return chunk_buf


def _convert_hf_checkpoint_to_orbax_in_memory(
    hf_checkpoint_dir: str,
    orbax_output_dir: str,
    config: glm5_config.Glm5Config,
    *,
    ignore_mtp_layer: bool,
    slice_to_config_layers: bool,
) -> None:
  """In-memory converter reference oracle scoped strictly to mini-slice checkpoints."""
  if config.num_hidden_layers > MAX_IN_MEMORY_CONVERSION_LAYERS:
    raise ValueError(
        "_convert_hf_checkpoint_to_orbax_in_memory is scoped to mini-slice"
        f" checkpoints with at most {MAX_IN_MEMORY_CONVERSION_LAYERS} layers"
        f" (got num_hidden_layers={config.num_hidden_layers}); use"
        " convert_hf_checkpoint_to_orbax for streaming conversion."
    )
  with jax.default_device(jax.devices("cpu")[0]):
    active_weight_map, shard_paths = _discover_and_validate_checkpoint(
        hf_checkpoint_dir,
        config,
        slice_to_config_layers=slice_to_config_layers,
        ignore_mtp_layer=ignore_mtp_layer,
    )
    specs = iter_conversion_specs(config, include_mtp=not ignore_mtp_layer)
    valid_hf_keys = _extract_valid_hf_keys(specs)

    # First pass: check headers and keys before materializing tensors
    total_params = 0
    all_tensors: dict[str, Any] = {}
    for sp in shard_paths:
      with safetensors.safe_open(sp, framework="pt", device="cpu") as f:
        for k in f.keys():
          if (
              k.startswith(("model.visual.", "visual."))
              or k not in active_weight_map
              or (ignore_mtp_layer and k.startswith(f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}."))
          ):
            continue
          if not (k.startswith("model.language_model.") or k in ("lm_head.weight", "lm_head.weight_scale_inv")):
            raise ValueError(f"Unknown HF key prefix: {k!r}")
          if k not in valid_hf_keys:
            raise ValueError(f"Unmapped key in hf_tensors: {k!r}")
          total_params += int(np.prod(f.get_slice(k).get_shape()))
          if total_params > MAX_IN_MEMORY_CONVERSION_PARAMS:
            raise ValueError(
                "_convert_hf_checkpoint_to_orbax_in_memory is scoped to"
                " mini-slice checkpoints with at most"
                f" {MAX_IN_MEMORY_CONVERSION_PARAMS} parameters (got"
                f" total_params={total_params}); use"
                " convert_hf_checkpoint_to_orbax for streaming conversion."
            )

    # Second pass: materialize tensors
    for sp in shard_paths:
      with safetensors.safe_open(sp, framework="pt", device="cpu") as f:
        for k in f.keys():
          if (
              k.startswith(("model.visual.", "visual."))
              or k not in active_weight_map
              or (ignore_mtp_layer and k.startswith(f"model.language_model.layers.{glm5_config.MTP_LAYER_INDEX}."))
          ):
            continue
          all_tensors[k] = f.get_tensor(k)

    model = modeling_glm5_3_flash.Glm5Model(config, rngs=nnx.Rngs(0), place_on_devices=False)
    load_hf_state_into_model(model, all_tensors, ignore_mtp_layer=ignore_mtp_layer)
    save_orbax_checkpoint(model, orbax_output_dir)


def convert_hf_checkpoint_to_orbax(
    hf_checkpoint_dir: str,
    orbax_output_dir: str,
    config: glm5_config.Glm5Config,
    *,
    ignore_mtp_layer: bool,
    slice_to_config_layers: bool,
    num_workers: int,
    max_inflight_bytes: int,
) -> None:
  """Streams a multi-shard HF safetensors checkpoint to an Orbax checkpoint.

  Bounds host RAM by writing each parameter slice directly to TensorStore/OCDBT
  on disk rather than accumulating the full parameter tree in memory, using a
  bounded-inflight thread pool.

  Args:
    hf_checkpoint_dir: Directory containing HuggingFace safetensors checkpoint.
    orbax_output_dir: Destination directory for the saved Orbax checkpoint.
    config: Glm5Config model configuration.
    ignore_mtp_layer: Whether to omit MTP layer 46 weights.
    slice_to_config_layers: Whether to slice checkpoint keys to config layers.
    num_workers: Number of concurrent worker threads (must be >= 1).
    max_inflight_bytes: Memory budget for concurrent work units.

  Returns:
    None.
  Raises:
    FileNotFoundError: If checkpoint files or referenced shards are missing.
    ValueError: If config, keys, dtypes, or parameter slots are invalid.
    FileExistsError: If orbax_output_dir exists and is non-empty.
  """
  if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
    raise ValueError(f"num_workers must be an integer >= 1, got {num_workers!r}")
  if isinstance(max_inflight_bytes, bool) or not isinstance(max_inflight_bytes, int) or max_inflight_bytes < 1:
    raise ValueError("max_inflight_bytes must be an integer >= 1, got" f" {max_inflight_bytes!r}")
  if ignore_mtp_layer and config.num_nextn_predict_layers > 0:
    raise ValueError(
        "ignore_mtp_layer=True cannot be used when"
        " config.num_nextn_predict_layers > 0; set num_nextn_predict_layers=0"
        " on the config instead."
    )
  if not ignore_mtp_layer and config.num_nextn_predict_layers == 0:
    raise ValueError("ignore_mtp_layer=False requires config.num_nextn_predict_layers > 0.")

  active_weight_map, _ = _discover_and_validate_checkpoint(
      hf_checkpoint_dir,
      config,
      slice_to_config_layers=slice_to_config_layers,
      ignore_mtp_layer=ignore_mtp_layer,
  )

  specs = iter_conversion_specs(
      config,
      include_mtp=not ignore_mtp_layer and config.num_nextn_predict_layers > 0,
  )
  valid_hf_keys = _extract_valid_hf_keys(specs)

  orig_key_map: dict[str, str] = {}
  for orig_k in active_weight_map.keys():
    if orig_k.startswith("model.visual.") or orig_k.startswith("visual."):
      continue
    if slice_to_config_layers and orig_k not in valid_hf_keys:
      continue
    norm_k = _normalize_key(orig_k)
    hub_k = norm_k if norm_k.startswith("lm_head.") else f"model.language_model.{norm_k}"
    if hub_k in orig_key_map:
      raise ValueError(f"Duplicate normalized key found: {norm_k!r}")
    orig_key_map[hub_k] = orig_k

  for spec in specs:
    for hk in spec.hf_keys:
      if hk not in orig_key_map:
        raise ValueError(f"Required parameter missing from HF state: {hk!r}")

  final_path = epath.Path(orbax_output_dir)
  if final_path.exists() and (not final_path.is_dir() or any(final_path.iterdir())):
    raise FileExistsError("Refusing to write into existing non-empty orbax_output_dir:" f" {orbax_output_dir!r}")

  shard_lru = _ThreadSafeShardLRU(_MAX_CONCURRENT_OPEN_SHARDS)
  write_dir: epath.Path | None = None
  ts_arrays: dict[tuple[str | int, ...], tuple[Any, tuple[int, ...]]] = {}
  try:
    init_timestamp_nsecs = time.time_ns()
    tmp_cls = ocp.path.atomicity_defaults.get_default_temporary_path_class(final_path)
    tmp_path = tmp_cls.from_final(final_path)
    write_dir = ocp.checkpoint_utils.asyncio_utils.run_sync(tmp_path.create())
    ts_context = ocp.serialization.tensorstore_utils.get_ts_context(use_ocdbt=True)

    abs_model = nnx.eval_shape(lambda: modeling_glm5_3_flash.Glm5Model(config, rngs=nnx.Rngs(0), place_on_devices=False))

    # Phase 1: Planning and work unit generation
    work_units: list[_ConversionWorkUnit] = []
    moe_populated_experts: dict[tuple[tuple[str | int, ...], int], set[int]] = collections.defaultdict(set)
    stacked_shapes: dict[tuple[str | int, ...], int] = collections.defaultdict(int)

    for spec_idx, spec in enumerate(specs):
      if spec.stack_index is not None and spec.num_super_blocks is not None:
        stacked_shapes[spec.target_path] = max(stacked_shapes[spec.target_path], spec.num_super_blocks)
      dt = np.dtype(spec.target_dtype)
      target_itemsize = dt.itemsize
      compute_itemsize = _TARGET_TO_INTERMEDIATE_DTYPE[dt][1].itemsize
      slot = 0 if spec.stack_index is None else spec.stack_index
      target_var = resolve_target_var(abs_model, spec.target_path)
      expected_shape = target_var.shape
      expected_slot_shape = expected_shape[1:] if spec.stack_index is not None else expected_shape

      if spec.kind in ("direct", "transpose_2d"):
        orig_k = orig_key_map[spec.hf_keys[0]]
        shard_file = active_weight_map[orig_k]
        primary_shard = shard_file
        slice_shape = shard_lru.get_slice_shape(os.path.join(hf_checkpoint_dir, shard_file), orig_k, shard_file)
        in_shape = slice_shape if spec.kind == "direct" else (slice_shape[1], slice_shape[0])
        if in_shape != expected_slot_shape:
          raise ValueError(
              f"Shape mismatch for {spec.target_path} (slot {slot}): expected" f" {expected_slot_shape}, got {in_shape}."
          )
        elems = int(np.prod(expected_slot_shape))
        output_slice_bytes = elems * target_itemsize
        norm_k = _normalize_key(orig_k)
        scale_norm_k = norm_k[:-7] + ".weight_scale_inv" if norm_k.endswith(".weight") else norm_k + "_scale_inv"
        scale_hub_k = scale_norm_k if scale_norm_k.startswith("lm_head.") else f"model.language_model.{scale_norm_k}"
        if scale_hub_k in orig_key_map:
          byte_weight = 3 * output_slice_bytes + 2 * elems * compute_itemsize
        elif dt == np.dtype(ml_dtypes.bfloat16):
          byte_weight = 3 * output_slice_bytes + elems * compute_itemsize
        else:
          byte_weight = 3 * output_slice_bytes
        work_units.append(
            _ConversionWorkUnit(
                sort_key=(primary_shard, spec_idx, 0),
                spec=spec,
                slot=slot,
                e_start=0,
                e_end=0,
                byte_weight=byte_weight,
            )
        )
      elif spec.kind in ("moe_gate_up", "moe_down"):
        per_expert_shape = expected_shape[1:] if spec.stack_index is None else expected_shape[2:]
        num_experts = len(spec.hf_keys) // 2 if spec.kind == "moe_gate_up" else len(spec.hf_keys)
        expert_chunk = math.gcd(_MAX_MOE_EXPERT_CHUNK_SIZE, num_experts) if num_experts > 0 else 1
        single_proj_elems = config.hidden_size * config.moe_intermediate_size
        for e_start in range(0, num_experts, expert_chunk):
          e_end = min(e_start + expert_chunk, num_experts)
          first_hk = spec.hf_keys[2 * e_start] if spec.kind == "moe_gate_up" else spec.hf_keys[e_start]
          primary_shard = active_weight_map[orig_key_map[first_hk]]
          output_slice_bytes = (e_end - e_start) * int(np.prod(per_expert_shape)) * target_itemsize
          byte_weight = 3 * output_slice_bytes + 2 * single_proj_elems * compute_itemsize
          work_units.append(
              _ConversionWorkUnit(
                  sort_key=(primary_shard, spec_idx, e_start),
                  spec=spec,
                  slot=slot,
                  e_start=e_start,
                  e_end=e_end,
                  byte_weight=byte_weight,
              )
          )
      else:
        raise ValueError(f"Unknown conversion spec kind: {spec.kind!r}")

    if not work_units:
      raise ValueError("No conversion work units generated.")
    largest_work_unit_bytes = max(u.byte_weight for u in work_units)
    if max_inflight_bytes < largest_work_unit_bytes:
      raise ValueError(
          f"max_inflight_bytes ({max_inflight_bytes} B) is smaller than the"
          f" largest single work unit ({largest_work_unit_bytes} B);"
          " deadlock would occur."
      )

    work_units.sort(key=lambda u: u.sort_key)

    # Phase 2: Open TensorStore arrays helper
    tree_metadata_entries: dict[str, Any] = {}

    def _open_ts_array(
        target_path: tuple[str | int, ...],
        global_shape: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dt: np.dtype,
    ) -> Any:
      param_name = ".".join(str(k) for k in target_path)
      write_spec = ocp._src.serialization.tensorstore_utils.ArrayWriteSpec(  # pylint: disable=protected-access
          str(write_dir),
          param_name,
          global_shape=global_shape,
          write_shape=global_shape,
          dtype=dt,
          use_compression=True,
          use_zarr3=False,
          use_ocdbt=True,
          process_id=0,
      )
      tspec = dict(write_spec.json)
      tspec["metadata"] = dict(tspec["metadata"])
      tspec["metadata"]["chunks"] = [int(d) for d in chunk_shape]
      if "compressor" not in tspec["metadata"] or tspec["metadata"]["compressor"] not in (
          {"id": "zstd"},
          {"id": "zstd", "level": 1},
      ):
        raise ValueError("Expected default Orbax zstd level 1 compressor in metadata:" f" {tspec['metadata']!r}")
      tspec["metadata"]["compressor"] = {"id": "zstd", "level": 1}
      store = ts.open(
          ts.Spec(tspec),
          create=True,
          open=True,
          context=ts_context,
      ).result()
      tuple_key_str = str(tuple(str(k) for k in target_path))
      tree_metadata_entries[tuple_key_str] = {
          "key_metadata": [{"key": str(k), "key_type": 2} for k in target_path],
          "value_metadata": {
              "value_type": "np.ndarray",
              "skip_deserialize": False,
          },
      }
      return store

    # Phase 3: Parallel worker execution
    semaphore = _MemoryBudgetSemaphore(max_inflight_bytes)
    consumed_lock = threading.Lock()
    consumed: set[str] = set()
    populated_slots_lock = threading.Lock()
    populated_slots: set[tuple[tuple[str | int, ...], int]] = set()
    ts_arrays_lock = threading.Lock()
    abort_event = threading.Event()

    def _load_from_shard(orig_k: str) -> Any:
      shard_file = active_weight_map[orig_k]
      shard_path = os.path.join(hf_checkpoint_dir, shard_file)
      return shard_lru.load_tensor(shard_path, orig_k, shard_file)

    def _get_tensor(hub_k: str, target_dtype: Any) -> np.ndarray:
      local_consumed: set[str] = set()
      val = _fetch_and_convert_tensor(
          hub_k,
          target_dtype,
          orig_key_map,
          local_consumed,
          _load_from_shard,
      )
      with consumed_lock:
        consumed.update(local_consumed)
      return val

    def _run_work_unit(unit: _ConversionWorkUnit) -> None:
      if abort_event.is_set():
        return
      semaphore.acquire(unit.byte_weight)
      try:
        if abort_event.is_set():
          return
        spec = unit.spec
        dt = np.dtype(spec.target_dtype)
        slot = unit.slot
        slot_key = (spec.target_path, slot)
        target_var = resolve_target_var(abs_model, spec.target_path)
        expected_shape = target_var.shape
        expected_slot_shape = expected_shape[1:] if spec.stack_index is not None else expected_shape

        if spec.kind in ("direct", "transpose_2d"):
          raw_v = _get_tensor(spec.hf_keys[0], spec.target_dtype)
          slot_val = np.ascontiguousarray(raw_v if spec.kind == "direct" else raw_v.T)
          del raw_v
          if slot_val.shape != expected_slot_shape:
            raise ValueError(
                f"Shape mismatch for {spec.target_path} (slot {slot}): expected"
                f" {expected_slot_shape}, got {slot_val.shape}."
            )
          if spec.stack_index is None:
            store = _open_ts_array(spec.target_path, expected_shape, slot_val.shape, dt)
            store.write(slot_val).result()
            del store
          else:
            with ts_arrays_lock:
              if spec.target_path not in ts_arrays:
                c_shape = (1, *slot_val.shape)
                ts_arrays[spec.target_path] = (
                    _open_ts_array(spec.target_path, expected_shape, c_shape, dt),
                    expected_shape,
                )
                stacked_shapes[spec.target_path] = max(stacked_shapes[spec.target_path], expected_shape[0])
              store, _ = ts_arrays[spec.target_path]
            store[slot].write(slot_val).result()
          del slot_val
          with populated_slots_lock:
            if slot_key in populated_slots:
              raise ValueError(f"Duplicate write to parameter slot {slot_key}.")
            populated_slots.add(slot_key)
        elif spec.kind in ("moe_gate_up", "moe_down"):
          num_experts = len(spec.hf_keys) // 2 if spec.kind == "moe_gate_up" else len(spec.hf_keys)
          expert_chunk = math.gcd(_MAX_MOE_EXPERT_CHUNK_SIZE, num_experts) if num_experts > 0 else 1
          g_shape = expected_shape
          per_expert_shape = g_shape[1:] if spec.stack_index is None else g_shape[2:]
          chunk_buf = _fill_moe_expert_chunk(
              spec,
              unit.e_start,
              unit.e_end,
              per_expert_shape,
              dt,
              _get_tensor,
          )
          if spec.stack_index is None:
            c_shape = (expert_chunk, *g_shape[1:])
            store = _open_ts_array(spec.target_path, g_shape, c_shape, dt)
            store[unit.e_start : unit.e_end].write(chunk_buf).result()
            del store
          else:
            with ts_arrays_lock:
              if spec.target_path not in ts_arrays:
                c_shape = (1, expert_chunk, *g_shape[2:])
                ts_arrays[spec.target_path] = (
                    _open_ts_array(spec.target_path, g_shape, c_shape, dt),
                    g_shape,
                )
                stacked_shapes[spec.target_path] = max(stacked_shapes[spec.target_path], expected_shape[0])
              store, _ = ts_arrays[spec.target_path]
            store[slot, unit.e_start : unit.e_end].write(chunk_buf).result()
          del chunk_buf
          chunk_experts = set(range(unit.e_start, unit.e_end))
          with populated_slots_lock:
            if not moe_populated_experts[slot_key].isdisjoint(chunk_experts):
              raise ValueError(f"Duplicate write to parameter slot {slot_key}.")
            moe_populated_experts[slot_key].update(chunk_experts)
            if len(moe_populated_experts[slot_key]) == num_experts:
              if slot_key in populated_slots:
                raise ValueError(f"Duplicate write to parameter slot {slot_key}.")
              populated_slots.add(slot_key)
        else:
          raise ValueError(f"Unknown conversion spec kind: {spec.kind!r}")
      except BaseException:
        abort_event.set()
        raise
      finally:
        semaphore.release(unit.byte_weight)

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
      unit_futures = [executor.submit(_run_work_unit, u) for u in work_units]
      try:
        for fut in futures.as_completed(unit_futures):
          fut.result()
      except BaseException:
        abort_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise

    unfilled = [
        (k, s) for k, num_slots in stacked_shapes.items() for s in range(num_slots) if (k, s) not in populated_slots
    ]
    if unfilled:
      raise ValueError(f"Unpopulated model parameter slot ({unfilled[0][1]}) for" f" {unfilled[0][0]}.")

    unconsumed = [
        k
        for k in orig_key_map.values()
        if k not in consumed
        and not (ignore_mtp_layer and _normalize_key(k).startswith(f"layers.{glm5_config.MTP_LAYER_INDEX}."))
    ]
    if unconsumed:
      raise ValueError(f"Unmapped key in hf_tensors: {unconsumed[0]!r}")

    ts_arrays.clear()
    shard_lru.close_all()
    ocp.checkpoint_utils.asyncio_utils.run_sync(
        ocp.type_handlers.merge_ocdbt_per_process_files(
            write_dir,
            ts_context=ts_context,
            use_zarr3=False,
            enable_validation=True,
        )
    )
    metadata_content = {
        "tree_metadata": tree_metadata_entries,
        "use_ocdbt": True,
        "use_zarr3": False,
        "store_array_data_equal_to_fill_value": True,
        "custom_metadata": None,
    }
    (write_dir / "_METADATA").write_text(json.dumps(metadata_content))
    ckpt_metadata_content = {
        "item_handlers": _STANDARD_CHECKPOINT_HANDLER,
        "metrics": {},
        "performance_metrics": {},
        "init_timestamp_nsecs": init_timestamp_nsecs,
        "commit_timestamp_nsecs": time.time_ns(),
        "custom_metadata": {},
    }
    (write_dir / "_CHECKPOINT_METADATA").write_text(json.dumps(ckpt_metadata_content))
    if write_dir != final_path and final_path.exists():
      final_path.rmdir()
    ocp.checkpoint_utils.asyncio_utils.run_sync(tmp_path.finalize())
  except BaseException:
    shard_lru.close_all()
    if write_dir is not None and write_dir.exists():
      write_dir.rmtree(missing_ok=True)
    raise
  finally:
    shard_lru.close_all()


def _parse_bool(val: str, flag_name: str) -> bool:
  """Parses explicit boolean flag values without silent defaults."""
  if not isinstance(val, str):
    raise ValueError(f"Invalid boolean value {val!r} for --{flag_name}; expected 'true' or" " 'false'.")
  lower = val.strip().lower()
  if lower == "true":
    return True
  elif lower == "false":
    return False
  else:
    raise ValueError(f"Invalid boolean value {val!r} for --{flag_name}; expected 'true' or" " 'false'.")


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the argument parser for GLM-5.3-Flash HF-to-Orbax conversion."""
  parser = argparse.ArgumentParser(description="GLM-5.3-Flash HF-to-Orbax Conversion CLI.")
  parser.add_argument(
      "--hf_checkpoint_dir",
      type=str,
      required=True,
      help=("Path to local directory containing HuggingFace Safetensors" " checkpoint shards."),
  )
  parser.add_argument(
      "--orbax_output_dir",
      type=str,
      required=True,
      help="Destination path or gs:// URI for output Orbax checkpoint.",
  )
  parser.add_argument(
      "--weight_dtype",
      type=str,
      required=True,
      help=("Target dtype for dequantized weights (e.g. bfloat16, float32," " float64)."),
  )
  parser.add_argument(
      "--ignore_mtp_layer",
      type=str,
      required=True,
      help=("Explicit boolean ('true'/'false'): whether to ignore MTP NextN" " prediction layer."),
  )
  parser.add_argument(
      "--slice_to_config_layers",
      type=str,
      required=True,
      help=("Explicit boolean ('true'/'false'): whether to slice hidden layers to" " num_hidden_layers."),
  )
  parser.add_argument(
      "--num_hidden_layers",
      type=int,
      required=True,
      help="Number of hidden decoder layers to convert.",
  )
  parser.add_argument(
      "--num_nextn_predict_layers",
      type=int,
      required=True,
      help="Number of NextN predict layers (0 or 1).",
  )
  parser.add_argument(
      "--config_json",
      type=str,
      required=True,
      help=(
          "Path to JSON file containing Glm5Config field overrides (rather than"
          " an HF config.json), or 'none' to use default config fields."
      ),
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      required=True,
      help=("Number of concurrent worker threads for streaming conversion (must" " be >= 1)."),
  )
  parser.add_argument(
      "--max_inflight_bytes",
      type=str,
      required=True,
      help=("Memory budget for concurrent work units; accepts positive byte count" " or unit suffix."),
  )
  return parser


def create_glm5_config_from_cli_args(
    args: argparse.Namespace,
) -> glm5_config.Glm5Config:
  """Validates parsed CLI arguments and constructs a Glm5Config instance."""
  if args.num_hidden_layers < 1:
    raise ValueError(f"--num_hidden_layers must be >= 1, got {args.num_hidden_layers}")
  if args.num_nextn_predict_layers not in (0, 1):
    raise ValueError("--num_nextn_predict_layers must be 0 or 1, got" f" {args.num_nextn_predict_layers}")
  if args.weight_dtype not in _SUPPORTED_CLI_DTYPES:
    raise ValueError(
        f"Unsupported --weight_dtype {args.weight_dtype!r}; expected one of" f" {sorted(_SUPPORTED_CLI_DTYPES)}"
    )

  if args.config_json == "none":
    return glm5_config.Glm5Config(
        num_hidden_layers=args.num_hidden_layers,
        num_nextn_predict_layers=args.num_nextn_predict_layers,
        weight_dtype=args.weight_dtype,
    )

  if "://" in args.config_json:
    raise ValueError(
        f"Unsupported URI scheme in --config_json: {args.config_json!r};" " expected a local filesystem path or 'none'."
    )
  if not os.path.isfile(args.config_json):
    raise FileNotFoundError(f"config_json file not found: {args.config_json!r}")

  with open(args.config_json, "r", encoding="utf-8") as f:
    raw_overrides = json.load(f)

  if not isinstance(raw_overrides, dict) or len(raw_overrides) == 0:
    raise ValueError("config_json must contain a non-empty JSON object.")

  overlap = set(raw_overrides.keys()) & _CLI_OWNED_CONFIG_KEYS
  if overlap:
    raise ValueError(f"config_json must not specify CLI-owned keys {sorted(overlap)};" " pass them via CLI flags only.")
  unknown = set(raw_overrides.keys()) - _ALLOWED_CONFIG_JSON_KEYS
  if unknown:
    raise ValueError(f"Unknown Glm5Config keys in config_json: {sorted(unknown)}")

  overrides = dict(raw_overrides)
  for tk in _TUPLE_CONFIG_KEYS:
    if tk in overrides:
      if not isinstance(overrides[tk], list):
        raise TypeError(f"Expected JSON list for {tk!r}, got {type(overrides[tk])}")
      overrides[tk] = tuple(overrides[tk])

  return glm5_config.Glm5Config(
      **overrides,
      num_hidden_layers=args.num_hidden_layers,
      num_nextn_predict_layers=args.num_nextn_predict_layers,
      weight_dtype=args.weight_dtype,
  )


def run_cli(argv: Sequence[str]) -> None:
  """Parses explicit CLI arguments and executes streaming HF-to-Orbax conversion."""
  parser = build_arg_parser()
  args = parser.parse_args(list(argv))
  hf_dir = args.hf_checkpoint_dir.strip()
  if not hf_dir:
    raise ValueError("--hf_checkpoint_dir must be a non-empty path.")
  if "://" in hf_dir:
    raise ValueError(
        "Unsupported URI scheme in --hf_checkpoint_dir:" f" {args.hf_checkpoint_dir!r}; expected a local filesystem path."
    )
  if not os.path.isdir(hf_dir):
    raise FileNotFoundError(f"HF checkpoint directory not found: {hf_dir!r}")
  orbax_out = args.orbax_output_dir.strip()
  if not orbax_out:
    raise ValueError("--orbax_output_dir must be a non-empty path.")
  if "://" in orbax_out and not orbax_out.startswith("gs://"):
    raise ValueError(
        "Unsupported URI scheme in --orbax_output_dir:"
        f" {args.orbax_output_dir!r}; expected a local path or 'gs://' URI."
    )
  out_path = epath.Path(orbax_out)
  if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
    raise FileExistsError("Refusing to write into existing non-empty --orbax_output_dir:" f" {orbax_out!r}")
  ignore_mtp_layer = _parse_bool(args.ignore_mtp_layer, "ignore_mtp_layer")
  slice_to_config_layers = _parse_bool(args.slice_to_config_layers, "slice_to_config_layers")
  config = create_glm5_config_from_cli_args(args)
  if ignore_mtp_layer and config.num_nextn_predict_layers > 0:
    raise ValueError("ignore_mtp_layer=True cannot be used when" " config.num_nextn_predict_layers > 0")
  if not ignore_mtp_layer and config.num_nextn_predict_layers == 0:
    raise ValueError("ignore_mtp_layer=False requires config.num_nextn_predict_layers > 0")
  num_workers = int(args.num_workers)
  if num_workers < 1:
    raise ValueError(f"--num_workers must be >= 1, got {args.num_workers}")
  max_inflight_bytes = _parse_byte_size(args.max_inflight_bytes, "max_inflight_bytes")
  convert_hf_checkpoint_to_orbax(
      hf_dir,
      orbax_out,
      config,
      ignore_mtp_layer=ignore_mtp_layer,
      slice_to_config_layers=slice_to_config_layers,
      num_workers=num_workers,
      max_inflight_bytes=max_inflight_bytes,
  )


def main() -> None:
  """CLI entrypoint reading sys.argv[1:] with zero parameter defaults."""
  run_cli(sys.argv[1:])
