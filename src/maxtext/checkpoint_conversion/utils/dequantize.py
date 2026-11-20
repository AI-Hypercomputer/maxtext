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

"""On-the-fly JAX dequantization for quantized HuggingFace SafeTensors checkpoints.

Supports block-scaled FP8 (`F8_E4M3` / `F8_E5M2`) and packed MXFP4 (`I8`, two
`E2M1` nibbles per byte) weights paired with `*_scale_inv` (e.g. DeepSeek-V3,
DeepSeek-V3.2, Kimi-K2) or `*.scale` (e.g. DeepSeek-V4) block-scale grids in
`F32`, `BF16`, `F16`, `F8_E8M0`, or `U8` (`E8M0`).

When a quantized weight is sharded along axis 0 across TPU hosts and its tiny
block-scale grid is replicated on the same mesh, `dequantize_weight` runs per
shard inside `jax.shard_map` with zero cross-host communication and produces the
exact `bfloat16` tensor produced by `standalone_scripts/deepseek_dequantize.py`.

This checkpoint-loading helper is kept separate from the runtime
`maxtext.layers.quantizations.dequantize_weight` utility because checkpoint
streaming requires packed `MXFP4` nibble unpacking + `E8M0` scales, edge-padded
`FP8` block grids, per-shard `jax.shard_map` row-offset slicing without
cross-host collectives, and lightweight `DECOUPLE_GCLOUD` imports free of
model-layer dependencies.
"""

from collections.abc import Mapping, Sequence, Set as AbstractSet
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils import tensor_handling
import numpy as np

# Lookup table for E2M1 FP4 (two 4-bit nibbles packed per int8/uint8 byte: low nibble first).
_FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_FP8_DTYPES = (np.dtype(jnp.float8_e4m3fn), np.dtype(jnp.float8_e5m2))
_PACKED_FP4_DTYPES = (np.dtype(np.int8),)
_QUANTIZED_WEIGHT_DTYPES = _FP8_DTYPES + _PACKED_FP4_DTYPES

DEFAULT_FP8_BLOCK_SIZE = 128
_WEIGHT_SUFFIX = ".weight"
_SCALE_INV_SUFFIX = "_scale_inv"
_DOT_SCALE_SUFFIX = ".scale"


def is_quantized_weight_dtype(dtype: Any) -> bool:
  """Returns True if `dtype` is a supported quantized checkpoint weight dtype (`FP8` or packed `MXFP4`)."""
  return dtype is not None and np.dtype(dtype) in _QUANTIZED_WEIGHT_DTYPES


def _is_float8_dtype(dtype: Any) -> bool:
  return dtype is not None and np.dtype(dtype).name.startswith("float8")


def companion_weight_key(key: str) -> str | None:
  """Returns the base weight key if `key` ends with `*_scale_inv` or `*.scale`, else None."""
  if key.endswith(_SCALE_INV_SUFFIX):
    return key[: -len(_SCALE_INV_SUFFIX)]
  if key.endswith(_DOT_SCALE_SUFFIX):
    return f"{key[:-len(_DOT_SCALE_SUFFIX)]}{_WEIGHT_SUFFIX}"
  return None


def find_scale_key(weight_key: str, hf_metadata: Mapping[str, Any]) -> str | None:
  """Returns the companion block-scale tensor key in `hf_metadata` for `weight_key`, or None."""
  scale_inv_key = f"{weight_key}{_SCALE_INV_SUFFIX}"
  if scale_inv_key in hf_metadata:
    return scale_inv_key
  if weight_key.endswith(_WEIGHT_SUFFIX):
    dot_scale_key = f"{weight_key[:-len(_WEIGHT_SUFFIX)]}{_DOT_SCALE_SUFFIX}"
    if dot_scale_key in hf_metadata:
      return dot_scale_key
  return None


def is_companion_scale_key(key: str, hf_metadata: Mapping[str, Any]) -> bool:
  """Returns True if `key` is a `*_scale_inv` or `*.scale` companion of a quantized weight in `hf_metadata`."""
  weight_key = companion_weight_key(key)
  if weight_key is None or weight_key not in hf_metadata:
    return False
  return is_quantized_weight_dtype(getattr(hf_metadata[weight_key], "dtype", None))


def needs_dequantization(
    weight_key: str,
    target_dtype: Any,
    hf_metadata: Mapping[str, Any],
    mapped_hf_keys: AbstractSet[str] = frozenset(),
) -> str | None:
  """Returns the companion scale key if `weight_key` should be dequantized on the fly, else None.

  Dequantization runs only when:
  1. `weight_key` is stored as FP8 (`float8_e4m3fn`/`float8_e5m2`) or packed MXFP4 (`int8`),
  2. the target MaxText leaf expects a floating-point dtype of at least 2 bytes (`bf16`/`fp16`/`fp32`),
  3. a companion `*_scale_inv` or `*.scale` tensor exists in `hf_metadata` and is not itself
     explicitly mapped to a MaxText `kernel_scale` leaf.
  """
  if weight_key not in hf_metadata:
    return None
  raw_dtype = getattr(hf_metadata[weight_key], "dtype", None)
  if not is_quantized_weight_dtype(raw_dtype):
    return None
  if not (jnp.issubdtype(target_dtype, jnp.floating) and np.dtype(target_dtype).itemsize >= 2):
    return None
  scale_key = find_scale_key(weight_key, hf_metadata)
  if scale_key is None or scale_key in mapped_hf_keys:
    return None
  return scale_key


def check_unscaled_fp8(
    weight_key: str,
    target_dtype: Any,
    hf_metadata: Mapping[str, Any],
    mapped_hf_keys: AbstractSet[str] = frozenset(),
    mt_key: Any = "",
) -> None:
  """Raises ValueError if `weight_key` is an FP8 tensor targeting a wider weight with no companion scale."""
  if weight_key not in hf_metadata:
    return
  raw_dtype = getattr(hf_metadata[weight_key], "dtype", None)
  if not _is_float8_dtype(raw_dtype) or _is_float8_dtype(target_dtype):
    return
  scale_key = find_scale_key(weight_key, hf_metadata)
  if scale_key is not None and scale_key in mapped_hf_keys:
    return
  if scale_key is None:
    target_label = f" (for {mt_key})" if mt_key else ""
    raise ValueError(
        f"FP8 HF tensors would be cast into wider MaxText weights without their scales:"
        f" {weight_key!r}{target_label} has dtype {np.dtype(raw_dtype)} targeting {np.dtype(target_dtype)},"
        " and no companion '*_scale_inv' or '*.scale' tensor was found."
    )


def resolve_source_scales(
    hf_source: str | tuple[str, ...] | None,
    target_dtypes: Sequence[Any],
    hf_metadata: Mapping[str, Any],
    mapped_hf_keys: AbstractSet[str] = frozenset(),
    mt_key: Any = "",
) -> dict[str, str]:
  """Resolves `{weight_key: scale_key}` for a `str` or `tuple` HF source and guards against unscaled FP8 casts.

  A tuple HF source that already includes a scale tensor (e.g. `("w", "w_scale")`) passes its
  scale directly to the hook and is not flagged when no separate companion scale exists.
  """
  if hf_source is None:
    return {}
  keys = hf_source if isinstance(hf_source, tuple) else (hf_source,)
  tuple_carries_scale = isinstance(hf_source, tuple) and any("scale" in k for k in keys)
  first_dtype = target_dtypes[0]
  scales: dict[str, str] = {}
  for key in keys:
    scale_key = needs_dequantization(key, first_dtype, hf_metadata, mapped_hf_keys)
    if scale_key is not None:
      scales[key] = scale_key
      continue
    if tuple_carries_scale:
      continue
    for t_dtype in target_dtypes:
      check_unscaled_fp8(key, t_dtype, hf_metadata, mapped_hf_keys, mt_key)
  return scales


def _normalize_block_size(block_size: int | Sequence[int] | None) -> tuple[int, int]:
  """Normalizes an `int`, `[block_m, block_n]` sequence, or `None` into `(block_m, block_n)`."""
  if block_size is None:
    return DEFAULT_FP8_BLOCK_SIZE, DEFAULT_FP8_BLOCK_SIZE
  if isinstance(block_size, int):
    if block_size <= 0:
      raise ValueError(f"block_size must be positive, got {block_size}.")
    return block_size, block_size
  dims = tuple(block_size)
  if len(dims) == 1:
    return _normalize_block_size(dims[0])
  if len(dims) != 2 or dims[0] <= 0 or dims[1] <= 0:
    raise ValueError(f"block_size must be a positive int or 2-element (block_m, block_n), got {block_size!r}.")
  return int(dims[0]), int(dims[1])


def resolve_fp8_block_size(
    maxtext_config: Any = None,
    hf_config: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
  """Resolves `(block_m, block_n)` from `maxtext_config.weight_block_size` or `hf_config.quantization_config`."""
  raw = getattr(maxtext_config, "weight_block_size", None)
  if isinstance(raw, bool) or not isinstance(raw, (int, list, tuple)):
    raw = None
  model_name = getattr(maxtext_config, "model_name", None)
  if raw is None and hf_config is None and isinstance(model_name, str) and model_name:
    from maxtext.checkpoint_conversion.utils import hf_model_configs  # pylint: disable=import-outside-toplevel

    model_key = model_name.replace("-Instruct", "")
    if model_key in hf_model_configs.HF_MODEL_CONFIGS:
      hf_config = hf_model_configs.HF_MODEL_CONFIGS[model_key].to_dict()
  if raw is None and isinstance(hf_config, Mapping):
    quant_cfg = hf_config.get("quantization_config")
    if isinstance(quant_cfg, Mapping):
      raw = quant_cfg.get("weight_block_size")
  return _normalize_block_size(raw)


def _block_dims(
    rows: int,
    cols: int,
    scale_rows: int,
    scale_cols: int,
    is_fp4: bool,
    default_block_size: int | Sequence[int] | None = DEFAULT_FP8_BLOCK_SIZE,
) -> tuple[int, int]:
  """Infers `(block_m, block_n)` from unpacked `(rows, cols)` and `(scale_rows, scale_cols)`."""
  if rows <= 0 or cols <= 0 or scale_rows <= 0 or scale_cols <= 0:
    raise ValueError(f"Weight ({rows}, {cols}) and scale ({scale_rows}, {scale_cols}) dimensions must be positive.")
  if is_fp4:
    if rows % scale_rows != 0 or cols % scale_cols != 0:
      raise ValueError(f"MXFP4 unpacked shape ({rows}, {cols}) not divisible by scale ({scale_rows}, {scale_cols}).")
    return rows // scale_rows, cols // scale_cols
  default_m, default_n = _normalize_block_size(default_block_size)
  # Check `ceil(dim / default_dim)` before exact division for FP8: an edge-padded tensor
  # (e.g. 36 rows with block_size=16 -> 3 scale rows) can accidentally divide `scale_rows` (36 / 3 = 12).
  if math.ceil(rows / default_m) == scale_rows:
    block_m = default_m
  elif rows % scale_rows == 0:
    block_m = rows // scale_rows
  else:
    raise ValueError(f"Weight rows {rows} are incompatible with scale rows {scale_rows}.")
  if math.ceil(cols / default_n) == scale_cols:
    block_n = default_n
  elif cols % scale_cols == 0:
    block_n = cols // scale_cols
  else:
    raise ValueError(f"Weight cols {cols} are incompatible with scale cols {scale_cols}.")
  return block_m, block_n


def _unpack_to_fp32(weight: jax.Array, is_fp4: bool) -> jax.Array:
  """Converts packed MXFP4 (`int8`) or FP8 weights to `float32`."""
  if not is_fp4:
    return weight.astype(jnp.float32)
  lut = jnp.asarray(_FP4_E2M1_LUT, dtype=jnp.float32)
  u8 = jax.lax.bitcast_convert_type(weight, jnp.uint8)
  unpacked = jnp.stack([lut[u8 & 0xF], lut[(u8 >> 4) & 0xF]], axis=-1)
  return unpacked.reshape(*weight.shape[:-1], 2 * weight.shape[-1])


def _scale_to_fp32(scale: jax.Array) -> jax.Array:
  """Converts float or E8M0 (`float8_e8m0fnu` / `uint8`) block scales to `float32`."""
  if scale.dtype == jnp.uint8:
    scale = jax.lax.bitcast_convert_type(scale, jnp.float8_e8m0fnu)
  return scale.astype(jnp.float32)


def _dequantize_2d_slice(
    w_slice: jax.Array,
    scale: jax.Array,
    row_offset: jax.Array | int,
    block_m: int,
    block_n: int,
    is_fp4: bool,
) -> jax.Array:
  """Dequantizes a contiguous 2D row slice `w_slice` starting at global row `row_offset`."""
  q_f32 = _unpack_to_fp32(w_slice, is_fp4)
  s_f32 = _scale_to_fp32(scale)
  local_m, cols = q_f32.shape
  scale_cols = s_f32.shape[1]
  s_rows = s_f32[(jnp.arange(local_m) + row_offset) // block_m]
  if cols == scale_cols * block_n:
    scaled = q_f32.reshape(local_m, scale_cols, block_n) * s_rows[:, :, None]
    return scaled.astype(jnp.bfloat16).reshape(local_m, cols)
  return (q_f32 * s_rows[:, jnp.arange(cols) // block_n]).astype(jnp.bfloat16)


def _dequantize_nd_local(
    weight: jax.Array,
    scale: jax.Array,
    block_m: int,
    block_n: int,
    is_fp4: bool,
) -> jax.Array:
  """Dequantizes an array of rank >= 2 where the trailing `(rows, cols)` matrix is unsharded."""
  q_f32 = _unpack_to_fp32(weight, is_fp4)
  s_f32 = _scale_to_fp32(scale)
  batch_shape = q_f32.shape[:-2]
  rows, cols = q_f32.shape[-2:]
  scale_rows, scale_cols = s_f32.shape[-2:]
  if rows == scale_rows * block_m and cols == scale_cols * block_n:
    q_blocks = q_f32.reshape(*batch_shape, scale_rows, block_m, scale_cols, block_n)
    s_blocks = s_f32.reshape(*batch_shape, scale_rows, 1, scale_cols, 1)
    return (q_blocks * s_blocks).astype(jnp.bfloat16).reshape(*batch_shape, rows, cols)
  row_idx = jnp.arange(rows) // block_m
  col_idx = jnp.arange(cols) // block_n
  s_expanded = s_f32[..., row_idx, :][..., :, col_idx]
  return (q_f32 * s_expanded).astype(jnp.bfloat16)


@functools.lru_cache(maxsize=None)
def _sharded_dequant_fn(
    mesh: jax.sharding.Mesh,
    row_axis: Any,
    ndim: int,
    block_m: int,
    block_n: int,
    is_fp4: bool,
):
  """Returns a cached jitted `shard_map` dequantizer for axis-0 sharded weights + replicated scales."""
  w_spec = jax.sharding.PartitionSpec(row_axis) if row_axis is not None else jax.sharding.PartitionSpec()
  s_spec = jax.sharding.PartitionSpec()
  out_sharding = jax.sharding.NamedSharding(mesh, w_spec)

  @functools.partial(jax.jit, out_shardings=out_sharding)
  @functools.partial(jax.shard_map, mesh=mesh, in_specs=(w_spec, s_spec), out_specs=w_spec, check_vma=False)
  def fn(w_shard: jax.Array, s_full: jax.Array) -> jax.Array:
    offset = jax.lax.axis_index(row_axis) * w_shard.shape[0] if row_axis is not None else 0
    if ndim == 2:
      return _dequantize_2d_slice(w_shard, s_full, offset, block_m, block_n, is_fp4)
    s_local = jax.lax.dynamic_slice_in_dim(s_full, offset, w_shard.shape[0], axis=0) if row_axis is not None else s_full
    return _dequantize_nd_local(w_shard, s_local, block_m, block_n, is_fp4)

  return fn


@functools.lru_cache(maxsize=None)
def _general_dequant_fn(block_m: int, block_n: int, is_fp4: bool):
  """Fallback jitted dequantizer for host arrays or single-host non-axis-0 shardings."""
  return jax.jit(lambda w, s: _dequantize_nd_local(w, s, block_m, block_n, is_fp4))


def dequantize_weight(
    weight: jax.Array | np.ndarray,
    scale: jax.Array | np.ndarray,
    block_size: int | Sequence[int] | None = DEFAULT_FP8_BLOCK_SIZE,
) -> jax.Array:
  """Dequantizes an FP8 or packed MXFP4 `weight` using its block `scale` tensor into `bfloat16`.

  Args:
    weight: Quantized weight tensor (`float8_e4m3fn`, `float8_e5m2`, or packed `int8` MXFP4) of
      rank >= 2. Packed MXFP4 holds two E2M1 values per byte along the last axis.
    scale: Block-scale tensor (`float32`, `bfloat16`, `float16`, `float8_e8m0fnu`, or `uint8` E8M0)
      of the same rank and leading batch dimensions as `weight`.
    block_size: Block size (`int` or `(block_m, block_n)`) for edge-padded FP8 tensors whose
      dimensions are not multiples of `scale.shape[-2:]` (defaults to 128).

  Returns:
    A `bfloat16` `jax.Array` of shape `(*weight.shape[:-1], 2 * weight.shape[-1])` for MXFP4 or
    `weight.shape` for FP8, preserving axis-0 `NamedSharding` without cross-host communication.
  """
  w_dtype = np.dtype(weight.dtype)
  if w_dtype not in _QUANTIZED_WEIGHT_DTYPES:
    raise ValueError(f"Unsupported quantized weight dtype {weight.dtype}; expected one of {_QUANTIZED_WEIGHT_DTYPES}.")
  if weight.ndim < 2 or scale.ndim != weight.ndim or tuple(weight.shape[:-2]) != tuple(scale.shape[:-2]):
    raise ValueError(f"Incompatible quantized weight shape {weight.shape} and scale shape {scale.shape}.")

  is_fp4 = w_dtype in _PACKED_FP4_DTYPES
  rows = weight.shape[-2]
  cols = weight.shape[-1] * 2 if is_fp4 else weight.shape[-1]
  scale_rows, scale_cols = scale.shape[-2:]
  block_m, block_n = _block_dims(rows, cols, scale_rows, scale_cols, is_fp4, block_size)

  if isinstance(weight, jax.Array) and isinstance(weight.sharding, jax.sharding.NamedSharding):
    spec = weight.sharding.spec
    if all(axis is None for axis in spec[1:]):
      mesh = weight.sharding.mesh
      row_axis = spec[0] if len(spec) >= 1 else None
      rep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      if not isinstance(scale, jax.Array) or scale.sharding != rep_sharding:
        scale = tensor_handling._device_put(scale, rep_sharding)  # pylint: disable=protected-access
      return _sharded_dequant_fn(mesh, row_axis, weight.ndim, block_m, block_n, is_fp4)(weight, scale)

  return _general_dequant_fn(block_m, block_n, is_fp4)(jnp.asarray(weight), jnp.asarray(scale))
