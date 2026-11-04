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

"""Option C FrozenTopK teacher-forced routing gate and >= 8 ULP router-agreement check for DeepSeek-V4.1."""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from maxtext.layers import attention_deepseek41
from maxtext.layers import moe
from maxtext.models import deepseek41

_FROZEN_TOPK_ATTR = "_frozen_topk_indices"
_FROZEN_INDEXER_TOPK_ATTR = "_frozen_indexer_topk"
_CAPTURED_TOPK_ATTR = "_captured_topk_indices"
_CAPTURED_MOE_TOPK_ATTR = _CAPTURED_TOPK_ATTR
_CAPTURED_INDEXER_TOPK_ATTR = "_captured_indexer_topk"
_UINT16_MAX = int(np.iinfo(np.uint16).max)
DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER = 8.0


class CapturedRoutingTopK(NamedTuple):
  """Captured pre-override natural MoE and Sparse-Attention Indexer top-k selections."""

  tpu_moe_topk: np.ndarray
  tpu_indexer_topk: dict[int, np.ndarray]

  def __getitem__(self, key: int | slice | str) -> Any:
    if isinstance(key, str):
      if key in ("tpu_moe_topk", "moe_topk"):
        return self.tpu_moe_topk
      if key in ("tpu_indexer_topk", "indexer_topk"):
        return self.tpu_indexer_topk
      raise KeyError(key)
    return tuple.__getitem__(self, key)

  def __contains__(self, key: object) -> bool:
    if isinstance(key, str):
      return key in ("tpu_moe_topk", "moe_topk", "tpu_indexer_topk", "indexer_topk")
    return tuple.__contains__(self, key)

  def get(self, key: str, default: Any = None) -> Any:
    try:
      return self[key]
    except KeyError:
      return default

  def keys(self) -> tuple[str, str]:
    return ("tpu_moe_topk", "tpu_indexer_topk")


class NaturalRoutingForwardResult(NamedTuple):
  """Outputs of a natural-routing forward pass with captured MoE and Indexer top-k selections."""

  logits: jax.Array
  tpu_moe_topk: np.ndarray
  tpu_indexer_topk: dict[int, np.ndarray]


def _safe_gather_and_numpy(var: Any, dtype=np.int32) -> np.ndarray:
  """Extracts a NumPy array from a JAX array or NNX Variable across single- or multi-host meshes."""
  val = var[...] if hasattr(var, "__getitem__") else var
  if hasattr(val, "value"):
    val = val.value
  if isinstance(val, jax.Array) and not val.is_fully_addressable:
    val = multihost_utils.process_allgather(val, tiled=True)
  return np.asarray(val, dtype=dtype)


def _resolve_mesh(model: Any, explicit_mesh: jax.sharding.Mesh | None = None) -> jax.sharding.Mesh | None:
  """Resolves a non-empty `jax.sharding.Mesh` from `explicit_mesh`, `model.mesh`, or `model.decoder.mesh`."""
  for candidate in (
      explicit_mesh,
      getattr(model, "mesh", None),
      getattr(getattr(model, "decoder", None), "mesh", None),
  ):
    if isinstance(candidate, jax.sharding.Mesh) and not getattr(candidate, "empty", False):
      return candidate
  return None


def _maybe_put_replicated(
    arr: jax.Array,
    model: Any,
    *,
    mesh: jax.sharding.Mesh | None = None,
) -> jax.Array:
  """Places `arr` on a fully replicated `NamedSharding(mesh, PartitionSpec())` when a non-empty mesh is present."""
  resolved_mesh = _resolve_mesh(model, mesh)
  if resolved_mesh is not None:
    sharding = getattr(arr, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh == resolved_mesh:
      return arr
    return jax.device_put(arr, jax.sharding.NamedSharding(resolved_mesh, jax.sharding.PartitionSpec()))
  return arr


def validate_raw_ref_topk_experts(
    raw_experts: Any,
    *,
    sample_name: str,
    expected_seq_len: int | None = None,
) -> np.ndarray:
  """Validates a raw `[num_layers, seq_len, top_k]` integer `ref_topk_experts` array.

  Args:
    raw_experts: Array-like per-layer reference top-k expert indices.
    sample_name: Sample identifier included in error messages.
    expected_seq_len: Optional sequence length that axis 1 must match.

  Returns:
    Validated `np.uint16` array of shape `(num_layers, seq_len, top_k)`.

  Raises:
    ValueError: If `raw_experts` has non-integer/boolean dtype, rank != 3, empty axes,
      out-of-range values, or mismatches `expected_seq_len`.
  """
  arr = np.asarray(raw_experts)
  if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
    raise ValueError(f"Sample {sample_name!r}: ref_topk_experts must have integer dtype, got dtype={arr.dtype}")
  if arr.ndim != 3:
    raise ValueError(
        f"Sample {sample_name!r}: ref_topk_experts must be 3D [num_layers, seq_len, top_k], got shape={arr.shape}"
    )
  if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1:
    raise ValueError(
        f"Sample {sample_name!r}: ref_topk_experts must have strictly positive dimensions, got shape={arr.shape}"
    )
  if expected_seq_len is not None and int(arr.shape[1]) != int(expected_seq_len):
    raise ValueError(
        f"Sample {sample_name!r}: ref_topk_experts seq_len {arr.shape[1]} != tokens seq_len {int(expected_seq_len)}"
    )
  if bool(np.any(arr < 0)) or int(np.max(arr)) > _UINT16_MAX:
    raise ValueError(
        f"Sample {sample_name!r}: ref_topk_experts contains out-of-bounds uint16 values "
        f"[min={int(np.min(arr))}, max={int(np.max(arr))}]"
    )
  return arr.astype(np.uint16, copy=False)


def validate_raw_ref_gate_margins_and_ulp(
    raw_margins: Any,
    raw_ulp: Any,
    *,
    sample_name: str,
    expected_num_layers: int | None = None,
    expected_seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Validates `ref_gate_margins` and `ref_gate_boundary_ulp` arrays (`float32 [num_layers, seq_len]`)."""
  if raw_margins is None or raw_ulp is None:
    raise ValueError(
        f"Sample {sample_name!r}: both ref_gate_margins and ref_gate_boundary_ulp are required when validating "
        "MoE router margin metadata."
    )
  margins_arr = np.asarray(raw_margins)
  ulp_arr = np.asarray(raw_ulp)
  for field_name, arr in (("ref_gate_margins", margins_arr), ("ref_gate_boundary_ulp", ulp_arr)):
    if np.issubdtype(arr.dtype, np.bool_) or not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ):
      raise ValueError(f"Sample {sample_name!r}: {field_name} must be numeric, got dtype={arr.dtype}")
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
      raise ValueError(
          f"Sample {sample_name!r}: {field_name} must be non-empty 2D [num_layers, seq_len], got shape={arr.shape}"
      )
  if margins_arr.shape != ulp_arr.shape:
    raise ValueError(
        f"Sample {sample_name!r}: ref_gate_margins shape {margins_arr.shape} != "
        f"ref_gate_boundary_ulp shape {ulp_arr.shape}"
    )
  if expected_num_layers is not None and int(margins_arr.shape[0]) != int(expected_num_layers):
    raise ValueError(
        f"Sample {sample_name!r}: ref_gate_margins num_layers {margins_arr.shape[0]} != "
        f"expected {int(expected_num_layers)}"
    )
  if expected_seq_len is not None and int(margins_arr.shape[1]) != int(expected_seq_len):
    raise ValueError(
        f"Sample {sample_name!r}: ref_gate_margins seq_len {margins_arr.shape[1]} != "
        f"tokens seq_len {int(expected_seq_len)}"
    )
  margins_f32 = margins_arr.astype(np.float32, copy=False)
  ulp_f32 = ulp_arr.astype(np.float32, copy=False)
  if not bool(np.all(np.isfinite(margins_f32))) or bool(np.any(margins_f32 < 0.0)):
    raise ValueError(f"Sample {sample_name!r}: ref_gate_margins must be finite and >= 0.0 everywhere.")
  if not bool(np.all(np.isfinite(ulp_f32))) or bool(np.any(ulp_f32 < 0.0)):
    raise ValueError(f"Sample {sample_name!r}: ref_gate_boundary_ulp must be finite and >= 0.0 everywhere.")
  moe_ulp_slice = ulp_f32[1:] if ulp_f32.shape[0] > 1 else ulp_f32
  if bool(np.any(moe_ulp_slice <= 0.0)) or bool(np.any((margins_f32 > 0.0) & (ulp_f32 <= 0.0))):
    raise ValueError(f"Sample {sample_name!r}: ref_gate_boundary_ulp must be finite and > 0.0 on all MoE layers.")
  return margins_f32, ulp_f32


def validate_raw_ref_indexer_routing(
    raw_layer_ids: Any,
    raw_indexer_topk: Any,
    raw_indexer_margins: Any | None = None,
    raw_indexer_ulp: Any | None = None,
    *,
    sample_name: str,
    expected_seq_len: int | None = None,
    require_margins_and_ulp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
  """Validates `ref_indexer_layer_ids`, `ref_indexer_topk`, `ref_indexer_margins`, and `ref_indexer_boundary_ulp`."""
  if raw_layer_ids is None or raw_indexer_topk is None:
    raise ValueError(f"Sample {sample_name!r}: both ref_indexer_layer_ids and ref_indexer_topk must be provided.")
  layer_ids_arr = np.asarray(raw_layer_ids)
  topk_arr = np.asarray(raw_indexer_topk)
  if np.issubdtype(layer_ids_arr.dtype, np.bool_) or not np.issubdtype(layer_ids_arr.dtype, np.integer):
    raise ValueError(f"Sample {sample_name!r}: ref_indexer_layer_ids must have integer dtype, got {layer_ids_arr.dtype}")
  if layer_ids_arr.ndim != 1 or layer_ids_arr.shape[0] < 1:
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_layer_ids must be non-empty 1D [n_idx_layers], got {layer_ids_arr.shape}"
    )
  if bool(np.any(layer_ids_arr < 0)) or (layer_ids_arr.shape[0] > 1 and not bool(np.all(np.diff(layer_ids_arr) > 0))):
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_layer_ids must be strictly increasing and >= 0, got {layer_ids_arr.tolist()}"
    )

  if np.issubdtype(topk_arr.dtype, np.bool_) or not np.issubdtype(topk_arr.dtype, np.integer):
    raise ValueError(f"Sample {sample_name!r}: ref_indexer_topk must have integer dtype, got {topk_arr.dtype}")
  if topk_arr.ndim != 3 or topk_arr.shape[0] != layer_ids_arr.shape[0] or topk_arr.shape[1] < 1 or topk_arr.shape[2] < 1:
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_topk must be 3D [n_idx_layers={layer_ids_arr.shape[0]}, seq_len, index_topk], "
        f"got shape={topk_arr.shape}"
    )
  if expected_seq_len is not None and int(topk_arr.shape[1]) != int(expected_seq_len):
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_topk seq_len {topk_arr.shape[1]} != tokens seq_len {int(expected_seq_len)}"
    )
  if bool(np.any(topk_arr < -1)):
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_topk contains invalid index < -1 (min={int(np.min(topk_arr))})"
    )

  n_idx_layers = int(layer_ids_arr.shape[0])
  seq_len = int(topk_arr.shape[1])
  if require_margins_and_ulp and (raw_indexer_margins is None or raw_indexer_ulp is None):
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_margins and ref_indexer_boundary_ulp are required when "
        "ref_indexer_topk is provided."
    )
  if (raw_indexer_margins is None) != (raw_indexer_ulp is None):
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_margins and ref_indexer_boundary_ulp must both be present or both None."
    )

  margins_f32: np.ndarray | None = None
  ulp_f32: np.ndarray | None = None
  if raw_indexer_margins is not None and raw_indexer_ulp is not None:
    m_arr = np.asarray(raw_indexer_margins)
    u_arr = np.asarray(raw_indexer_ulp)
    for field_name, arr in (("ref_indexer_margins", m_arr), ("ref_indexer_boundary_ulp", u_arr)):
      if np.issubdtype(arr.dtype, np.bool_) or not (
          np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
      ):
        raise ValueError(f"Sample {sample_name!r}: {field_name} must be numeric, got dtype={arr.dtype}")
      if arr.shape != (n_idx_layers, seq_len):
        raise ValueError(f"Sample {sample_name!r}: {field_name} shape {arr.shape} != expected {(n_idx_layers, seq_len)}")
    margins_f32 = m_arr.astype(np.float32, copy=False)
    ulp_f32 = u_arr.astype(np.float32, copy=False)
    nan_m = np.isnan(margins_f32)
    nan_u = np.isnan(ulp_f32)
    if not bool(np.array_equal(nan_m, nan_u)):
      raise ValueError(
          f"Sample {sample_name!r}: ref_indexer_margins and ref_indexer_boundary_ulp NaN masks do not match."
      )
    if bool(np.any(np.isinf(margins_f32))) or bool(np.any(np.isinf(ulp_f32))):
      raise ValueError(
          f"Sample {sample_name!r}: ref_indexer_margins and ref_indexer_boundary_ulp must not contain +/-inf."
      )
    finite_mask = ~nan_m
    if bool(np.any(margins_f32[finite_mask] < 0.0)):
      raise ValueError(f"Sample {sample_name!r}: finite ref_indexer_margins entries must be >= 0.0.")
    if bool(np.any(ulp_f32[finite_mask] <= 0.0)):
      raise ValueError(f"Sample {sample_name!r}: finite ref_indexer_boundary_ulp entries must be > 0.0.")

  return (
      layer_ids_arr.astype(np.int32, copy=False),
      topk_arr.astype(np.int32, copy=False),
      margins_f32,
      ulp_f32,
  )


def prepare_padded_ref_topk_experts(
    raw_experts: Any,
    *,
    sample_name: str,
    num_layers: int,
    seq_len: int,
    padded_seq_len: int,
    batch_size: int,
    top_k: int,
    num_experts: int,
    moe_layer_ids: Sequence[int],
) -> np.ndarray:
  """Validates `ref_topk_experts` against model config and pads to `[num_layers, B, S_pad, top_k]`.

  Args:
    raw_experts: Reference top-k indices `[num_layers, seq_len, top_k]`.
    sample_name: Sample identifier for error messages.
    num_layers: Expected `config.num_decoder_layers`.
    seq_len: Unpadded sequence length of the sample.
    padded_seq_len: Compiled sequence length (`ids.shape[1]`).
    batch_size: Global batch size (`ids.shape[0]`).
    top_k: Expected `config.num_experts_per_tok`.
    num_experts: Total routed experts `config.num_experts`.
    moe_layer_ids: Global decoder layer indices that carry a `RoutedMoE` block.

  Returns:
    `np.int32` array of shape `(num_layers, batch_size, padded_seq_len, top_k)`.

  Raises:
    ValueError: On shape, expert-bound, or duplicate-expert-per-token violations.
  """
  arr = validate_raw_ref_topk_experts(raw_experts, sample_name=sample_name, expected_seq_len=seq_len)
  expected_shape = (int(num_layers), int(seq_len), int(top_k))
  if arr.shape != expected_shape:
    raise ValueError(f"Sample {sample_name!r}: ref_topk_experts shape {arr.shape} != expected {expected_shape}")
  if int(top_k) > int(num_experts):
    raise ValueError(f"Sample {sample_name!r}: top_k={top_k} exceeds num_experts={num_experts}")
  if int(padded_seq_len) < int(seq_len):
    raise ValueError(f"Sample {sample_name!r}: padded_seq_len={padded_seq_len} < seq_len={seq_len}")
  if int(batch_size) < 1:
    raise ValueError(f"Sample {sample_name!r}: batch_size must be >= 1, got {batch_size}")
  if bool(np.any(arr >= int(num_experts))):
    raise ValueError(
        f"Sample {sample_name!r}: ref_topk_experts max index {int(np.max(arr))} >= num_experts={num_experts}"
    )
  if int(top_k) > 1:
    for layer_id in moe_layer_ids:
      sorted_layer = np.sort(arr[int(layer_id)], axis=-1)
      if not bool(np.all(np.diff(sorted_layer, axis=-1) > 0)):
        raise ValueError(
            f"Sample {sample_name!r}: ref_topk_experts has duplicate expert indices within top_k at MoE layer {layer_id}"
        )

  padded_1b = np.empty((int(num_layers), int(padded_seq_len), int(top_k)), dtype=np.int32)
  padded_1b[:, : int(seq_len), :] = arr.astype(np.int32)
  if int(padded_seq_len) > int(seq_len):
    padded_1b[:, int(seq_len) :, :] = np.arange(int(top_k), dtype=np.int32)[None, None, :]
  return np.broadcast_to(
      padded_1b[:, None, :, :],
      (int(num_layers), int(batch_size), int(padded_seq_len), int(top_k)),
  ).copy()


def _normalize_ref_indexer_topk_slice(
    layer_slice: np.ndarray,
    seq_len: int,
    *,
    layer_id: int | None = None,
) -> np.ndarray:
  """Normalizes PyTorch reference indexer top-k indices by subtracting the `seq_len` window-KV offset.

  In the PyTorch reference (`model.py:430/509`), window KV (`seq_len` tokens) and compressed KV
  (`num_compressed <= seq_len` blocks) are concatenated along axis 1, so all valid (`>= 0`) compressed
  KV indices are stored in reference offset form `[seq_len, seq_len + num_compressed)`. Per layer, all
  valid entries (`>= 0`) must be `>= seq_len` (and at least one valid entry must exist when `seq_len > 0`);
  any non-offset entry in `[0, seq_len)` or all-invalid layer raises `ValueError` naming the layer and count.
  """
  arr = np.asarray(layer_slice, dtype=np.int32).copy()
  s_len = int(seq_len)
  if s_len <= 0:
    return arr
  if arr.ndim == 3:
    for l_idx in range(int(arr.shape[0])):
      sub = arr[l_idx]
      valid = sub >= 0
      bad_mask = valid & (sub < s_len)
      bad_count = int(np.sum(bad_mask))
      if bad_count > 0:
        raise ValueError(
            f"Expected reference offset form (>= seq_len={s_len}) in ref_indexer_topk at layer {l_idx}, "
            f"but found count={bad_count} valid entries < {s_len}"
        )
      if not bool(np.any(valid)):
        raise ValueError(
            f"Expected valid entries (>= 0) in ref_indexer_topk at layer {l_idx}, " "but found count=0 valid entries"
        )
    arr[arr >= 0] -= s_len
    return arr

  layer_label = 0 if layer_id is None else int(layer_id)
  valid = arr >= 0
  bad_mask = valid & (arr < s_len)
  bad_count = int(np.sum(bad_mask))
  if bad_count > 0:
    raise ValueError(
        f"Expected reference offset form (>= seq_len={s_len}) in ref_indexer_topk at layer {layer_label}, "
        f"but found count={bad_count} valid entries < {s_len}"
    )
  if not bool(np.any(valid)):
    raise ValueError(
        f"Expected valid entries (>= 0) in ref_indexer_topk at layer {layer_label}, " "but found count=0 valid entries"
    )
  arr[valid] -= s_len
  return arr


def prepare_padded_ref_indexer_topk(
    raw_indexer_topk: Any,
    raw_indexer_layer_ids: Any,
    *,
    sample_name: str,
    seq_len: int,
    padded_seq_len: int,
    batch_size: int,
    expected_layer_ids: Sequence[int] | None = None,
    index_topk: int | None = None,
) -> dict[int, np.ndarray]:
  """Validates `ref_indexer_topk` and pads along sequence/batch axes to `dict[layer_id, int32 [B, S_pad, index_topk]]`."""
  layer_ids, topk_arr, _, _ = validate_raw_ref_indexer_routing(
      raw_indexer_layer_ids,
      raw_indexer_topk,
      sample_name=sample_name,
      expected_seq_len=seq_len,
      require_margins_and_ulp=False,
  )
  if int(padded_seq_len) < int(seq_len):
    raise ValueError(f"Sample {sample_name!r}: padded_seq_len={padded_seq_len} < seq_len={seq_len}")
  if int(batch_size) < 1:
    raise ValueError(f"Sample {sample_name!r}: batch_size must be >= 1, got {batch_size}")
  if expected_layer_ids is not None:
    got_ids = [int(x) for x in layer_ids.tolist()]
    want_ids = [int(x) for x in expected_layer_ids]
    if got_ids != want_ids:
      raise ValueError(f"Sample {sample_name!r}: ref_indexer_layer_ids {got_ids} != model index_source_layers {want_ids}")
  if index_topk is not None and int(topk_arr.shape[2]) != int(index_topk):
    raise ValueError(
        f"Sample {sample_name!r}: ref_indexer_topk trailing axis {topk_arr.shape[2]} "
        f"!= expected index_topk={int(index_topk)}"
    )

  k_dim = int(topk_arr.shape[2])
  out: dict[int, np.ndarray] = {}
  for idx_i, layer_id_raw in enumerate(layer_ids.tolist()):
    layer_id = int(layer_id_raw)
    layer_slice = _normalize_ref_indexer_topk_slice(topk_arr[idx_i], int(seq_len), layer_id=layer_id)
    for pos in range(int(seq_len)):
      row = layer_slice[pos]
      valid_vals = row[row >= 0]
      if valid_vals.size > 1 and np.unique(valid_vals).size != valid_vals.size:
        raise ValueError(
            f"Sample {sample_name!r}: ref_indexer_topk has duplicate compressed indices at layer={layer_id}, pos={pos}"
        )
    padded_1b = np.full((int(padded_seq_len), k_dim), -1, dtype=np.int32)
    padded_1b[: int(seq_len), :] = layer_slice.astype(np.int32, copy=False)
    out[layer_id] = np.broadcast_to(
        padded_1b[None, :, :],
        (int(batch_size), int(padded_seq_len), k_dim),
    ).copy()
  return out


def _collect_decoder_moe_and_indexer_targets(
    decoder: Any,
    config: Any,
) -> tuple[
    list[tuple[int, moe.RoutedMoE]],
    list[tuple[Any, int, moe.RoutedMoE]],
    list[int],
    list[tuple[int, attention_deepseek41.DeepSeek41Indexer]],
    list[tuple[Any, int, attention_deepseek41.DeepSeek41Indexer]],
    list[int],
]:
  """Collects unscanned or scanned `RoutedMoE` and `DeepSeek41Indexer` module targets on `model.decoder`."""
  num_layers = int(config.num_decoder_layers)
  unscanned_moe: list[tuple[int, moe.RoutedMoE]] = []
  scanned_moe: list[tuple[Any, int, moe.RoutedMoE]] = []
  moe_layer_ids: list[int] = []

  unscanned_idx: list[tuple[int, attention_deepseek41.DeepSeek41Indexer]] = []
  scanned_idx: list[tuple[Any, int, attention_deepseek41.DeepSeek41Indexer]] = []
  indexer_layer_ids: list[int] = []

  if bool(config.scan_layers):
    segments = deepseek41.build_segments(config)
    for seg_idx, segment in enumerate(segments):
      stack = getattr(decoder, f"segments_{seg_idx}")
      for sub_idx in range(segment.block_size):
        sub_layer = getattr(stack, f"layers_{sub_idx}")
        if hasattr(sub_layer, "mlp") and hasattr(sub_layer.mlp, "MoeBlock_0"):
          moe_block = sub_layer.mlp.MoeBlock_0
          scanned_moe.append((segment, sub_idx, moe_block))
          for rep_idx in range(segment.repeats):
            moe_layer_ids.append(int(segment.groups[rep_idx][sub_idx]))
        if hasattr(sub_layer, "self_attention") and getattr(sub_layer.self_attention, "indexer", None) is not None:
          idx_mod = sub_layer.self_attention.indexer
          scanned_idx.append((segment, sub_idx, idx_mod))
          for rep_idx in range(segment.repeats):
            indexer_layer_ids.append(int(segment.groups[rep_idx][sub_idx]))
  else:
    for layer_idx in range(num_layers):
      layer = getattr(decoder, f"layers_{layer_idx}")
      if hasattr(layer, "mlp") and hasattr(layer.mlp, "MoeBlock_0"):
        moe_block = layer.mlp.MoeBlock_0
        unscanned_moe.append((layer_idx, moe_block))
        moe_layer_ids.append(layer_idx)
      if hasattr(layer, "self_attention") and getattr(layer.self_attention, "indexer", None) is not None:
        idx_mod = layer.self_attention.indexer
        unscanned_idx.append((layer_idx, idx_mod))
        indexer_layer_ids.append(layer_idx)

  return unscanned_moe, scanned_moe, moe_layer_ids, unscanned_idx, scanned_idx, indexer_layer_ids


def _compute_indexer_scores_and_reachability(
    self_idx: attention_deepseek41.DeepSeek41Indexer,
    x: jax.Array,
    qr: jax.Array,
    positions: jax.Array,
    shared: dict[str, Any],
    *,
    decoder_segment_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, int, dict[str, Any]]:
  """Runs the score, causal reachability, and candidate-block prefix of `DeepSeek41Indexer.__call__`."""
  updates: dict[str, Any] = {}
  index_k = shared["index_k"]
  num_compressed = int(index_k.shape[1])
  q = self_idx.wq_b(attention_deepseek41.deepseek41_quant.fp8_qdq(qr) if self_idx.act_quant else qr)
  q = self_idx.rotary_emb(q, positions, unsqueeze_dim=-2)
  if self_idx.act_quant:
    q = attention_deepseek41.deepseek41_quant.fp4_qdq(q)
  weights = self_idx.weights_proj(x).astype(jnp.float32) * (self_idx.head_dim**-0.5 * self_idx.n_heads**-0.5)
  scores = jnp.einsum(
      "bshd,btd->bsht",
      q.astype(jnp.float32),
      index_k.astype(jnp.float32),
      precision=jax.lax.Precision(self_idx.config.matmul_precision),
  )
  scores = jnp.sum(jax.nn.relu(scores) * weights[..., None], axis=2)

  ratio = self_idx.compress_ratio
  if decoder_segment_ids is None:
    compress_lens = attention_deepseek41.compressed_visible_len(positions, ratio)
    reachable = jnp.arange(num_compressed)[None, None, :] < compress_lens[..., None]
  else:
    block_start_seg = decoder_segment_ids[:, : num_compressed * ratio : ratio]
    block_end_seg = decoder_segment_ids[:, ratio - 1 : num_compressed * ratio : ratio]
    block_end_pos = positions[:, ratio - 1 : num_compressed * ratio : ratio]
    reachable = (
        (block_start_seg[:, None, :] == decoder_segment_ids[:, :, None])
        & (block_end_seg[:, None, :] == decoder_segment_ids[:, :, None])
        & (decoder_segment_ids[:, :, None] != 0)
        & (block_end_pos[:, None, :] <= positions[:, :, None])
    )
    compress_lens = jnp.max(
        jnp.where(reachable, jnp.arange(num_compressed)[None, None, :] + 1, 0),
        axis=-1,
    )
  scores = jnp.where(reachable, scores, -jnp.inf)

  if self_idx.is_candidate_source:
    updates["candidates"] = attention_deepseek41.select_candidate_blocks(
        scores, compress_lens, self_idx.config.candidate_topk_blocks, self_idx.config.candidate_block_size
    )
  elif self_idx.uses_candidates:
    scores = jnp.where(shared["candidates"], scores, -jnp.inf)

  return scores, reachable, compress_lens, num_compressed, updates


@contextlib.contextmanager
def frozen_topk_routing_context(
    model: Any,
    config: Any,
    raw_experts: Any,
    *,
    sample_name: str,
    seq_len: int,
    batch_size: int,
    padded_seq_len: int,
    raw_indexer_topk: Any | None = None,
    raw_indexer_layer_ids: Any | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> Iterator[int]:
  """Temporarily attaches teacher-forced `ref_topk_experts` (and optional `ref_indexer_topk`) onto `model.decoder`.

  Works identically under `jax.jit` for both `scan_layers=True` (stacking per-segment repeat axes on
  each `DeepSeek41Block` sub-layer's `nnx.Variable` state) and `scan_layers=False` (attaching per-layer
  `nnx.Variable` state), while gathering and normalizing MaxText's own unbiased router scores and
  replaying reference Sparse-Attention Indexer top-k compressed KV selections when provided.

  Args:
    model: Loaded NNX MaxText model with `model.decoder`.
    config: MaxText config object.
    raw_experts: Reference top-k expert indices `[num_layers, seq_len, top_k]`.
    sample_name: Sample identifier for error messages.
    seq_len: Sample unpadded sequence length.
    batch_size: Global batch size (`ids.shape[0]`).
    padded_seq_len: Compiled sequence length (`ids.shape[1]`).
    raw_indexer_topk: Optional reference indexer top-k indices `[n_idx_layers, seq_len, index_topk]`.
    raw_indexer_layer_ids: Optional reference indexer source layer IDs `[n_idx_layers]`.

  Yields:
    Number of MoE decoder layers configured with teacher-forced routing.
  """
  if (raw_indexer_topk is None) != (raw_indexer_layer_ids is None):
    raise ValueError(
        f"Sample {sample_name!r}: raw_indexer_topk and raw_indexer_layer_ids must both be provided or both be None."
    )

  decoder = model.decoder
  num_layers = int(config.num_decoder_layers)
  top_k = int(config.num_experts_per_tok)
  num_experts = int(config.num_experts)

  (
      unscanned_targets,
      scanned_targets,
      moe_layer_ids,
      unscanned_idx_targets,
      scanned_idx_targets,
      indexer_layer_ids,
  ) = _collect_decoder_moe_and_indexer_targets(decoder, config)

  if not moe_layer_ids:
    raise RuntimeError(f"Sample {sample_name!r}: FrozenTopK found zero RoutedMoE (MoeBlock_0) modules on model.decoder")

  padded_indices = prepare_padded_ref_topk_experts(
      raw_experts,
      sample_name=sample_name,
      num_layers=num_layers,
      seq_len=seq_len,
      padded_seq_len=padded_seq_len,
      batch_size=batch_size,
      top_k=top_k,
      num_experts=num_experts,
      moe_layer_ids=sorted(moe_layer_ids),
  )

  padded_indexer_map: dict[int, np.ndarray] | None = None
  if raw_indexer_topk is not None and raw_indexer_layer_ids is not None:
    if not indexer_layer_ids:
      raise RuntimeError(
          f"Sample {sample_name!r}: FrozenTopK received raw_indexer_topk but found zero DeepSeek41Indexer modules."
      )
    padded_indexer_map = prepare_padded_ref_indexer_topk(
        raw_indexer_topk,
        raw_indexer_layer_ids,
        sample_name=sample_name,
        seq_len=seq_len,
        padded_seq_len=padded_seq_len,
        batch_size=batch_size,
        expected_layer_ids=sorted(indexer_layer_ids),
        index_topk=int(config.indexer_topk),
    )

  index_topk = int(getattr(config, "indexer_topk", 0) or 0)
  if index_topk <= 0 and padded_indexer_map:
    first_idx_arr = list(padded_indexer_map.values())[0]
    index_topk = int(first_idx_arr.shape[-1])

  patched_blocks: list[moe.RoutedMoE] = []
  patched_indexers: list[attention_deepseek41.DeepSeek41Indexer] = []
  orig_deepseek_routing = moe.RoutedMoE.deepseek_routing
  orig_indexer_call = attention_deepseek41.DeepSeek41Indexer.__call__

  def _frozen_deepseek_routing(
      self_moe: moe.RoutedMoE,
      gate_logits: jax.Array,
      pre_bias_logits: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    has_frozen = hasattr(self_moe, _FROZEN_TOPK_ATTR) and getattr(self_moe, _FROZEN_TOPK_ATTR) is not None
    has_captured = hasattr(self_moe, _CAPTURED_TOPK_ATTR) and getattr(self_moe, _CAPTURED_TOPK_ATTR) is not None
    if not has_frozen:
      top_k_weights, top_k_indices = orig_deepseek_routing(self_moe, gate_logits, pre_bias_logits)
      if has_captured:
        getattr(self_moe, _CAPTURED_TOPK_ATTR).set_value(jnp.asarray(top_k_indices, dtype=jnp.int32))
      return top_k_weights, top_k_indices
    if has_captured:
      _, nat_top_k_indices = orig_deepseek_routing(self_moe, gate_logits, pre_bias_logits)
      getattr(self_moe, _CAPTURED_TOPK_ATTR).set_value(jnp.asarray(nat_top_k_indices, dtype=jnp.int32))
    frozen_var = getattr(self_moe, _FROZEN_TOPK_ATTR)
    top_k_indices = jnp.asarray(frozen_var[...], dtype=jnp.int32)
    expected_idx_shape = pre_bias_logits.shape[:-1] + (self_moe.num_experts_per_tok,)
    if top_k_indices.shape != expected_idx_shape:
      raise ValueError(f"FrozenTopK indices shape {top_k_indices.shape} != expected {expected_idx_shape}")
    if top_k_indices.ndim == 3:
      top_k_indices = self_moe._maybe_shard_with_logical(  # pylint: disable=protected-access
          top_k_indices,
          ("activation_batch_moe", "activation_length_moe", None),
      )
    tag_routing_fn = moe.routing_tag_fn(self_moe.config)
    if tag_routing_fn is not None:
      top_k_indices = tag_routing_fn(top_k_indices)
    use_dense_vjp = bool(self_moe.config.router_topk_matmul_vjp)
    if use_dense_vjp:
      top_k_weights = moe.take_along_last_axis_dense_vjp(pre_bias_logits, top_k_indices, pre_bias_logits.shape[-1])
    else:
      top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def _frozen_indexer_call(  # pylint: disable=too-many-positional-arguments
      self_idx: attention_deepseek41.DeepSeek41Indexer,
      x: jax.Array,
      qr: jax.Array,
      positions: jax.Array,
      shared: dict[str, Any],
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, Any]]:
    has_frozen = hasattr(self_idx, _FROZEN_INDEXER_TOPK_ATTR) and getattr(self_idx, _FROZEN_INDEXER_TOPK_ATTR) is not None
    has_captured = (
        hasattr(self_idx, _CAPTURED_INDEXER_TOPK_ATTR) and getattr(self_idx, _CAPTURED_INDEXER_TOPK_ATTR) is not None
    )
    if not has_frozen and not has_captured:
      return orig_indexer_call(self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids)
    scores, reachable, compress_lens, num_compressed, updates = _compute_indexer_scores_and_reachability(
        self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids
    )
    k = min(int(self_idx.index_topk), num_compressed)
    if has_captured or not has_frozen:
      nat_top_idx = jax.lax.top_k(scores, k)[1]
      nat_keep = (
          nat_top_idx < compress_lens[..., None]
          if decoder_segment_ids is None
          else jnp.take_along_axis(reachable, nat_top_idx, axis=-1)
      )
      if has_captured:
        nat_valid_idx = jnp.where(nat_keep, nat_top_idx.astype(jnp.int32), jnp.int32(-1))
        if k < int(self_idx.index_topk):
          pad_width = ((0, 0), (0, 0), (0, int(self_idx.index_topk) - int(k)))
          nat_valid_idx = jnp.pad(nat_valid_idx, pad_width, constant_values=-1)
        else:
          nat_valid_idx = nat_valid_idx[..., : int(self_idx.index_topk)]
        getattr(self_idx, _CAPTURED_INDEXER_TOPK_ATTR).set_value(nat_valid_idx)
      if not has_frozen:
        return attention_deepseek41._scatter_bool(nat_top_idx, nat_keep, num_compressed), updates  # pylint: disable=protected-access
    frozen_var = getattr(self_idx, _FROZEN_INDEXER_TOPK_ATTR)
    ref_idx = jnp.asarray(frozen_var[...], dtype=jnp.int32)[..., :k]
    expected_idx_shape = (x.shape[0], x.shape[1], k)
    if ref_idx.shape != expected_idx_shape:
      raise ValueError(f"FrozenTopK indexer shape {ref_idx.shape} != expected {expected_idx_shape}")
    valid_ref = (ref_idx >= 0) & (ref_idx < num_compressed)
    safe_idx = jnp.where(valid_ref, ref_idx, 0)
    keep = valid_ref & (
        safe_idx < compress_lens[..., None]
        if decoder_segment_ids is None
        else jnp.take_along_axis(reachable, safe_idx, axis=-1)
    )

    def _scatter_frozen_row(idx: jax.Array, val: jax.Array) -> jax.Array:
      return jnp.zeros((num_compressed,), dtype=jnp.bool_).at[idx].max(val)

    return jax.vmap(jax.vmap(_scatter_frozen_row))(safe_idx, keep), updates

  try:
    if bool(config.scan_layers):
      for segment, sub_idx, moe_block in scanned_targets:
        stacked = np.stack(
            [padded_indices[int(segment.groups[r][sub_idx])] for r in range(segment.repeats)],
            axis=0,
        )
        init_moe = _maybe_put_replicated(jnp.asarray(stacked, dtype=jnp.int32), model, mesh=mesh)
        setattr(moe_block, _FROZEN_TOPK_ATTR, nnx.Variable(init_moe))
        init_cap_moe = _maybe_put_replicated(
            jnp.zeros((segment.repeats, int(batch_size), int(padded_seq_len), top_k), dtype=jnp.int32),
            model,
            mesh=mesh,
        )
        setattr(moe_block, _CAPTURED_TOPK_ATTR, nnx.Variable(init_cap_moe))
        patched_blocks.append(moe_block)
      for segment, sub_idx, idx_mod in scanned_idx_targets:
        if padded_indexer_map is not None:
          stacked_idx = np.stack(
              [padded_indexer_map[int(segment.groups[r][sub_idx])] for r in range(segment.repeats)],
              axis=0,
          )
          init_idx = _maybe_put_replicated(jnp.asarray(stacked_idx, dtype=jnp.int32), model, mesh=mesh)
          setattr(idx_mod, _FROZEN_INDEXER_TOPK_ATTR, nnx.Variable(init_idx))
        if index_topk > 0:
          init_cap_idx = _maybe_put_replicated(
              jnp.full((segment.repeats, int(batch_size), int(padded_seq_len), index_topk), -1, dtype=jnp.int32),
              model,
              mesh=mesh,
          )
          setattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR, nnx.Variable(init_cap_idx))
        if padded_indexer_map is not None or index_topk > 0:
          patched_indexers.append(idx_mod)
    else:
      for layer_idx, moe_block in unscanned_targets:
        init_moe = _maybe_put_replicated(jnp.asarray(padded_indices[layer_idx], dtype=jnp.int32), model, mesh=mesh)
        setattr(
            moe_block,
            _FROZEN_TOPK_ATTR,
            nnx.Variable(init_moe),
        )
        init_cap_moe = _maybe_put_replicated(
            jnp.zeros((int(batch_size), int(padded_seq_len), top_k), dtype=jnp.int32),
            model,
            mesh=mesh,
        )
        setattr(moe_block, _CAPTURED_TOPK_ATTR, nnx.Variable(init_cap_moe))
        patched_blocks.append(moe_block)
      for layer_idx, idx_mod in unscanned_idx_targets:
        if padded_indexer_map is not None:
          init_idx = _maybe_put_replicated(jnp.asarray(padded_indexer_map[layer_idx], dtype=jnp.int32), model, mesh=mesh)
          setattr(
              idx_mod,
              _FROZEN_INDEXER_TOPK_ATTR,
              nnx.Variable(init_idx),
          )
        if index_topk > 0:
          init_cap_idx = _maybe_put_replicated(
              jnp.full((int(batch_size), int(padded_seq_len), index_topk), -1, dtype=jnp.int32),
              model,
              mesh=mesh,
          )
          setattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR, nnx.Variable(init_cap_idx))
        if padded_indexer_map is not None or index_topk > 0:
          patched_indexers.append(idx_mod)
    moe.RoutedMoE.deepseek_routing = _frozen_deepseek_routing  # type: ignore[method-assign]
    if padded_indexer_map is not None or patched_indexers:
      attention_deepseek41.DeepSeek41Indexer.__call__ = _frozen_indexer_call  # type: ignore[method-assign]
    yield len(moe_layer_ids)
  finally:
    moe.RoutedMoE.deepseek_routing = orig_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = orig_indexer_call  # type: ignore[method-assign]
    for moe_block in patched_blocks:
      if hasattr(moe_block, _FROZEN_TOPK_ATTR):
        delattr(moe_block, _FROZEN_TOPK_ATTR)
      if hasattr(moe_block, _CAPTURED_TOPK_ATTR):
        delattr(moe_block, _CAPTURED_TOPK_ATTR)
    for idx_mod in patched_indexers:
      if hasattr(idx_mod, _FROZEN_INDEXER_TOPK_ATTR):
        delattr(idx_mod, _FROZEN_INDEXER_TOPK_ATTR)
      if hasattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR):
        delattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR)


def _extract_captured(
    model: Any,
    config: Any,
    *,
    batch_size: int,
    padded_seq_len: int,
) -> CapturedRoutingTopK:
  """Extracts captured MoE and Indexer top-k arrays from `model.decoder` across scanned or unscanned layers."""
  decoder = model.decoder
  num_layers = int(config.num_decoder_layers)
  top_k = int(config.num_experts_per_tok)
  (
      unscanned_targets,
      scanned_targets,
      _,
      unscanned_idx_targets,
      scanned_idx_targets,
      _,
  ) = _collect_decoder_moe_and_indexer_targets(decoder, config)
  moe_out = np.zeros((num_layers, int(batch_size), int(padded_seq_len), top_k), dtype=np.int32)
  idx_out: dict[int, np.ndarray] = {}
  if bool(config.scan_layers):
    for segment, sub_idx, moe_block in scanned_targets:
      if hasattr(moe_block, _CAPTURED_TOPK_ATTR) and getattr(moe_block, _CAPTURED_TOPK_ATTR) is not None:
        arr = _safe_gather_and_numpy(getattr(moe_block, _CAPTURED_TOPK_ATTR), dtype=np.int32)
        for rep_idx in range(segment.repeats):
          moe_out[int(segment.groups[rep_idx][sub_idx])] = arr[rep_idx]
    for segment, sub_idx, idx_mod in scanned_idx_targets:
      if hasattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR) and getattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR) is not None:
        arr_idx = _safe_gather_and_numpy(getattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR), dtype=np.int32)
        for rep_idx in range(segment.repeats):
          idx_out[int(segment.groups[rep_idx][sub_idx])] = arr_idx[rep_idx]
  else:
    for layer_idx, moe_block in unscanned_targets:
      if hasattr(moe_block, _CAPTURED_TOPK_ATTR) and getattr(moe_block, _CAPTURED_TOPK_ATTR) is not None:
        moe_out[layer_idx] = _safe_gather_and_numpy(getattr(moe_block, _CAPTURED_TOPK_ATTR), dtype=np.int32)
    for layer_idx, idx_mod in unscanned_idx_targets:
      if hasattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR) and getattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR) is not None:
        idx_out[layer_idx] = _safe_gather_and_numpy(getattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR), dtype=np.int32)
  return CapturedRoutingTopK(tpu_moe_topk=moe_out, tpu_indexer_topk=idx_out)


_FROZEN_TOPK_STEP_CACHE: dict[tuple[int, int, tuple[Any, ...], Any], Any] = {}
_NATURAL_ROUTING_STEP_CACHE: dict[tuple[int, int, tuple[Any, ...], Any], Any] = {}


def clear_frozen_topk_step_caches() -> None:
  """Clears memoized JIT step caches for testing and benchmarking."""
  _FROZEN_TOPK_STEP_CACHE.clear()
  _NATURAL_ROUTING_STEP_CACHE.clear()


def _safe_graphdef_key(gdef: Any) -> Any:
  """Returns a hashable cache key representation for a Flax NNX GraphDef."""
  try:
    hash(gdef)
    return gdef
  except TypeError:
    pass

  def _make_hashable(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
      return tuple(_make_hashable(x) for x in val)
    if hasattr(val, "value"):
      return ("Static", _make_hashable(val.value))
    try:
      hash(val)
      return val
    except TypeError:
      return str(val)

  node_types = tuple(type(n).__name__ for n in gdef.nodes)
  attr_keys = tuple(
      (k, type(v).__name__, _make_hashable(v.value) if hasattr(v, "value") else str(v)) for k, v in gdef.attributes
  )
  return (gdef.num_leaves, len(gdef.nodes), node_types, hash(attr_keys))


def _freeze_hashable(obj: Any) -> Any:
  """Recursively converts unhashable containers (lists, dicts, sets) into hashable equivalents."""
  if isinstance(obj, (list, tuple)):
    return tuple(_freeze_hashable(x) for x in obj)
  if isinstance(obj, dict):
    return tuple(sorted((k, _freeze_hashable(v)) for k, v in obj.items()))
  if isinstance(obj, (set, frozenset)):
    return frozenset(_freeze_hashable(x) for x in obj)
  return obj


def get_frozen_topk_step_fn(
    model: Any,
    mesh: jax.sharding.Mesh,
    logical_axis_rules: Sequence[tuple[str, Any]] | None,
    graphdef: Any,
) -> Any:
  """Gets or creates the memoized @jax.jit step function for frozen topk forward."""
  rules_key = _freeze_hashable(logical_axis_rules) if logical_axis_rules is not None else ()
  cache_key = (id(model), id(mesh), rules_key, _safe_graphdef_key(graphdef))
  rules_for_axis = logical_axis_rules if logical_axis_rules is not None else ()
  if cache_key not in _FROZEN_TOPK_STEP_CACHE:

    @jax.jit
    def _step_fn(step_state, step_inputs):
      step_ids, step_pos, step_seg, step_imgs, step_masks = step_inputs
      with mesh, nn_partitioning.axis_rules(rules_for_axis):
        diag_model = nnx.merge(graphdef, step_state)
        logits = diag_model(
            decoder_input_tokens=step_ids,
            decoder_positions=step_pos,
            decoder_segment_ids=step_seg,
            encoder_images=step_imgs,
            encoder_image_masks=step_masks,
            enable_dropout=False,
        )
        _, _, captured_non_params = nnx.split(diag_model, (nnx.Param, nnx.RngState, nnx.Intermediate), ...)
        return logits, captured_non_params

    _FROZEN_TOPK_STEP_CACHE[cache_key] = _step_fn
  return _FROZEN_TOPK_STEP_CACHE[cache_key]


def get_natural_routing_step_fn(
    model: Any,
    mesh: jax.sharding.Mesh,
    logical_axis_rules: Sequence[tuple[str, Any]] | None,
    graphdef: Any,
) -> Any:
  """Gets or creates the memoized @jax.jit step function for natural routing trace."""
  rules_key = _freeze_hashable(logical_axis_rules) if logical_axis_rules is not None else ()
  cache_key = (id(model), id(mesh), rules_key, _safe_graphdef_key(graphdef))
  rules_for_axis = logical_axis_rules if logical_axis_rules is not None else ()
  if cache_key not in _NATURAL_ROUTING_STEP_CACHE:

    @jax.jit
    def _step_fn(step_state, step_inputs):
      step_ids, step_pos, step_seg, step_imgs, step_masks = step_inputs
      with mesh, nn_partitioning.axis_rules(rules_for_axis):
        trace_model = nnx.merge(graphdef, step_state)
        logits = trace_model(
            decoder_input_tokens=step_ids,
            decoder_positions=step_pos,
            decoder_segment_ids=step_seg,
            encoder_images=step_imgs,
            encoder_image_masks=step_masks,
            enable_dropout=False,
        )
        _, _, captured_non_params = nnx.split(trace_model, (nnx.Param, nnx.RngState, nnx.Intermediate), ...)
        return logits, captured_non_params

    _NATURAL_ROUTING_STEP_CACHE[cache_key] = _step_fn
  return _NATURAL_ROUTING_STEP_CACHE[cache_key]


def run_frozen_topk_forward_nnx(
    model: Any,
    config: Any,
    mesh: jax.sharding.Mesh,
    raw_experts: Any,
    *,
    sample_name: str,
    ids: Any,
    decoder_positions: Any,
    decoder_segment_ids: Any,
    images: Any,
    image_masks: Any,
    seq_len: int,
    raw_indexer_topk: Any | None = None,
    raw_indexer_layer_ids: Any | None = None,
    return_rerun: bool = False,
    return_captured: bool = False,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, CapturedRoutingTopK]
    | tuple[jax.Array, jax.Array, CapturedRoutingTopK]
):
  """Runs a `jax.jit`-compiled teacher-forced `FrozenTopK` forward pass on an NNX model."""
  ids_arr = jnp.asarray(ids, dtype=jnp.int32)
  pos_arr = jnp.asarray(decoder_positions, dtype=jnp.int32)
  seg_arr = jnp.asarray(decoder_segment_ids, dtype=jnp.int32)
  imgs_arr = jnp.asarray(images, dtype=jnp.float32) if images is not None else None
  masks_arr = jnp.asarray(image_masks, dtype=jnp.int32) if image_masks is not None else None
  batch_size = int(ids_arr.shape[0])
  padded_seq_len = int(ids_arr.shape[1])

  with frozen_topk_routing_context(
      model,
      config,
      raw_experts,
      sample_name=sample_name,
      seq_len=int(seq_len),
      batch_size=batch_size,
      padded_seq_len=padded_seq_len,
      raw_indexer_topk=raw_indexer_topk,
      raw_indexer_layer_ids=raw_indexer_layer_ids,
      mesh=mesh,
  ):
    diag_graphdef, diag_state = nnx.split(model)
    _diag_eval_step = get_frozen_topk_step_fn(model, mesh, config.logical_axis_rules, diag_graphdef)
    step_args = (ids_arr, pos_arr, seg_arr, imgs_arr, masks_arr)
    run1_logits, updated_non_params = _diag_eval_step(diag_state, step_args)
    run2_logits: jax.Array | None = None
    if return_rerun:
      run2_logits, _ = _diag_eval_step(diag_state, step_args)
    clean_non_params = nnx.filter_state(updated_non_params, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params)
    if return_captured:
      captured = _extract_captured(
          model,
          config,
          batch_size=batch_size,
          padded_seq_len=padded_seq_len,
      )
      if return_rerun:
        assert run2_logits is not None
        return run1_logits, run2_logits, captured
      return run1_logits, captured
    if return_rerun:
      assert run2_logits is not None
      return run1_logits, run2_logits
    return run1_logits


@contextlib.contextmanager
def natural_routing_trace_context(
    model: Any,
    config: Any,
    *,
    sample_name: str,
    batch_size: int,
    padded_seq_len: int,
    mesh: jax.sharding.Mesh | None = None,
) -> Iterator[tuple[Callable[[], tuple[np.ndarray, dict[int, np.ndarray]]], list[int], list[int]]]:
  """Temporarily attaches mutable `nnx.Variable` trace buffers to capture natural MoE and Indexer top-k selections."""
  decoder = model.decoder
  top_k = int(config.num_experts_per_tok)
  index_topk = int(config.indexer_topk)

  (
      unscanned_targets,
      scanned_targets,
      moe_layer_ids,
      unscanned_idx_targets,
      scanned_idx_targets,
      indexer_layer_ids,
  ) = _collect_decoder_moe_and_indexer_targets(decoder, config)

  if not moe_layer_ids:
    raise RuntimeError(
        f"Sample {sample_name!r}: natural_routing_trace_context found zero RoutedMoE modules on model.decoder"
    )

  patched_blocks: list[moe.RoutedMoE] = []
  patched_indexers: list[attention_deepseek41.DeepSeek41Indexer] = []
  orig_deepseek_routing = moe.RoutedMoE.deepseek_routing
  orig_indexer_call = attention_deepseek41.DeepSeek41Indexer.__call__

  def _trace_deepseek_routing(
      self_moe: moe.RoutedMoE,
      gate_logits: jax.Array,
      pre_bias_logits: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    top_k_weights, top_k_indices = orig_deepseek_routing(self_moe, gate_logits, pre_bias_logits)
    if hasattr(self_moe, _CAPTURED_TOPK_ATTR) and getattr(self_moe, _CAPTURED_TOPK_ATTR) is not None:
      getattr(self_moe, _CAPTURED_TOPK_ATTR).set_value(jnp.asarray(top_k_indices, dtype=jnp.int32))
    return top_k_weights, top_k_indices

  def _trace_indexer_call(  # pylint: disable=too-many-positional-arguments
      self_idx: attention_deepseek41.DeepSeek41Indexer,
      x: jax.Array,
      qr: jax.Array,
      positions: jax.Array,
      shared: dict[str, Any],
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, Any]]:
    if not hasattr(self_idx, _CAPTURED_INDEXER_TOPK_ATTR) or getattr(self_idx, _CAPTURED_INDEXER_TOPK_ATTR) is None:
      return orig_indexer_call(self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids)
    scores, reachable, compress_lens, num_compressed, updates = _compute_indexer_scores_and_reachability(
        self_idx, x, qr, positions, shared, decoder_segment_ids=decoder_segment_ids
    )
    k = min(int(self_idx.index_topk), num_compressed)
    top_idx = jax.lax.top_k(scores, k)[1]
    keep = (
        top_idx < compress_lens[..., None]
        if decoder_segment_ids is None
        else jnp.take_along_axis(reachable, top_idx, axis=-1)
    )
    valid_idx = jnp.where(keep, top_idx.astype(jnp.int32), jnp.int32(-1))
    if k < int(self_idx.index_topk):
      pad_width = ((0, 0), (0, 0), (0, int(self_idx.index_topk) - int(k)))
      valid_idx = jnp.pad(valid_idx, pad_width, constant_values=-1)
    else:
      valid_idx = valid_idx[..., : int(self_idx.index_topk)]
    getattr(self_idx, _CAPTURED_INDEXER_TOPK_ATTR).set_value(valid_idx)
    return attention_deepseek41._scatter_bool(top_idx, keep, num_compressed), updates  # pylint: disable=protected-access

  def _extract_captured_fn() -> CapturedRoutingTopK:
    return _extract_captured(
        model,
        config,
        batch_size=int(batch_size),
        padded_seq_len=int(padded_seq_len),
    )

  try:
    if bool(config.scan_layers):
      for segment, _, moe_block in scanned_targets:
        init_moe = _maybe_put_replicated(
            jnp.zeros((segment.repeats, int(batch_size), int(padded_seq_len), top_k), dtype=jnp.int32),
            model,
            mesh=mesh,
        )
        setattr(moe_block, _CAPTURED_TOPK_ATTR, nnx.Variable(init_moe))
        patched_blocks.append(moe_block)
      for segment, _, idx_mod in scanned_idx_targets:
        init_idx = _maybe_put_replicated(
            jnp.full((segment.repeats, int(batch_size), int(padded_seq_len), index_topk), -1, dtype=jnp.int32),
            model,
            mesh=mesh,
        )
        setattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR, nnx.Variable(init_idx))
        patched_indexers.append(idx_mod)
    else:
      for _, moe_block in unscanned_targets:
        init_moe = _maybe_put_replicated(
            jnp.zeros((int(batch_size), int(padded_seq_len), top_k), dtype=jnp.int32),
            model,
            mesh=mesh,
        )
        setattr(moe_block, _CAPTURED_TOPK_ATTR, nnx.Variable(init_moe))
        patched_blocks.append(moe_block)
      for _, idx_mod in unscanned_idx_targets:
        init_idx = _maybe_put_replicated(
            jnp.full((int(batch_size), int(padded_seq_len), index_topk), -1, dtype=jnp.int32),
            model,
            mesh=mesh,
        )
        setattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR, nnx.Variable(init_idx))
        patched_indexers.append(idx_mod)

    moe.RoutedMoE.deepseek_routing = _trace_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = _trace_indexer_call  # type: ignore[method-assign]
    yield _extract_captured_fn, sorted(moe_layer_ids), sorted(indexer_layer_ids)
  finally:
    moe.RoutedMoE.deepseek_routing = orig_deepseek_routing  # type: ignore[method-assign]
    attention_deepseek41.DeepSeek41Indexer.__call__ = orig_indexer_call  # type: ignore[method-assign]
    for moe_block in patched_blocks:
      if hasattr(moe_block, _CAPTURED_TOPK_ATTR):
        delattr(moe_block, _CAPTURED_TOPK_ATTR)
    for idx_mod in patched_indexers:
      if hasattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR):
        delattr(idx_mod, _CAPTURED_INDEXER_TOPK_ATTR)


def run_natural_routing_forward_with_trace_nnx(
    model: Any,
    config: Any,
    mesh: jax.sharding.Mesh,
    *,
    sample_name: str,
    ids: Any,
    decoder_positions: Any,
    decoder_segment_ids: Any,
    images: Any,
    image_masks: Any,
) -> NaturalRoutingForwardResult:
  """Runs a `jax.jit`-compiled natural-routing forward pass while capturing MoE and Indexer top-k indices."""
  ids_arr = jnp.asarray(ids, dtype=jnp.int32)
  pos_arr = jnp.asarray(decoder_positions, dtype=jnp.int32)
  seg_arr = jnp.asarray(decoder_segment_ids, dtype=jnp.int32)
  imgs_arr = jnp.asarray(images, dtype=jnp.float32) if images is not None else None
  masks_arr = jnp.asarray(image_masks, dtype=jnp.int32) if image_masks is not None else None

  with natural_routing_trace_context(
      model,
      config,
      sample_name=sample_name,
      batch_size=int(ids_arr.shape[0]),
      padded_seq_len=int(ids_arr.shape[1]),
      mesh=mesh,
  ) as (extract_fn, _, _):
    trace_graphdef, trace_state = nnx.split(model)
    _trace_eval_step = get_natural_routing_step_fn(model, mesh, config.logical_axis_rules, trace_graphdef)
    logits, updated_non_params = _trace_eval_step(
        trace_state,
        (ids_arr, pos_arr, seg_arr, imgs_arr, masks_arr),
    )
    clean_non_params = nnx.filter_state(updated_non_params, nnx.All(nnx.Not(nnx.RngState), nnx.Not(nnx.Intermediate)))
    nnx.update(model, clean_non_params)
    tpu_moe_topk, tpu_indexer_topk = extract_fn()
    return NaturalRoutingForwardResult(
        logits=logits,
        tpu_moe_topk=tpu_moe_topk,
        tpu_indexer_topk=tpu_indexer_topk,
    )


def check_router_agreement(
    *,
    sample_name: str,
    seq_len: int,
    token_types: np.ndarray | None,
    tpu_moe_topk: Any,
    ref_topk_experts: Any,
    ref_gate_margins: Any,
    ref_gate_boundary_ulp: Any,
    tpu_indexer_topk: Mapping[int, Any] | np.ndarray | None = None,
    ref_indexer_layer_ids: Any | None = None,
    ref_indexer_topk: Any | None = None,
    ref_indexer_margins: Any | None = None,
    ref_indexer_boundary_ulp: Any | None = None,
    moe_layer_ids: Sequence[int] | None = None,
    ulp_multiplier: float = DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
  """Enforces Option C Pass Gate 2 (`>= 8 ULP` hard set equality on MoE router and Sparse-Attention Indexer).

  Args:
    sample_name: Sample identifier for error messages.
    seq_len: Unpadded sequence length `S`.
    token_types: Optional 1D `int32 [S]` array (`-1` for text, `>= 0` for image slots).
    tpu_moe_topk: Captured TPU MoE top-k indices `[num_layers, B, S_pad, top_k]` (or `[num_layers, S_pad, top_k]`).
    ref_topk_experts: Reference MoE top-k indices `[num_layers, S, top_k]`.
    ref_gate_margins: Reference MoE top-k boundary gap $s_{(k)} - s_{(k+1)}$ (`float32 [num_layers, S]`).
    ref_gate_boundary_ulp: Reference MoE IEEE-754 fp32 ULP at $s_{(k)}$ (`float32 [num_layers, S]`).
    tpu_indexer_topk: Optional captured TPU Indexer top-k indices (`dict[layer_id, [B, S_pad, k]]` or 4D array).
    ref_indexer_layer_ids: Optional `int32 [n_idx_layers]` of index-source layer IDs.
    ref_indexer_topk: Optional `int32 [n_idx_layers, S, index_topk]` of reference compressed KV selections.
    ref_indexer_margins: Optional `float32 [n_idx_layers, S]` of reference indexer boundary gaps (`NaN` when unpruned).
    ref_indexer_boundary_ulp: Optional `float32 [n_idx_layers, S]` of reference indexer boundary ULPs (`NaN` when unpruned).
    moe_layer_ids: Optional active MoE layer IDs (defaults to all `0 .. num_layers - 1`).
    ulp_multiplier: Boundary ULP multiplier threshold (default `8.0`).
    raise_on_failure: When True, raises `AssertionError` if any high-margin MoE or Indexer mismatch occurs.

  Returns:
    Structured router-agreement summary dictionary.
  """
  if float(ulp_multiplier) <= 0.0 or not np.isfinite(float(ulp_multiplier)):
    raise ValueError(f"Sample {sample_name!r}: ulp_multiplier must be finite and > 0, got {ulp_multiplier}")

  ref_moe = validate_raw_ref_topk_experts(
      ref_topk_experts,
      sample_name=sample_name,
      expected_seq_len=int(seq_len),
  )
  num_layers, s_len, top_k = (int(ref_moe.shape[0]), int(ref_moe.shape[1]), int(ref_moe.shape[2]))
  margins_f32, ulp_f32 = validate_raw_ref_gate_margins_and_ulp(
      ref_gate_margins,
      ref_gate_boundary_ulp,
      sample_name=sample_name,
      expected_num_layers=num_layers,
      expected_seq_len=s_len,
  )

  tpu_moe_arr = _safe_gather_and_numpy(tpu_moe_topk, dtype=np.int32)
  if tpu_moe_arr.ndim == 3:
    tpu_moe_arr = tpu_moe_arr[:, None, :, :]
  if tpu_moe_arr.ndim != 4 or int(tpu_moe_arr.shape[0]) != num_layers or int(tpu_moe_arr.shape[3]) != top_k:
    raise ValueError(
        f"Sample {sample_name!r}: tpu_moe_topk shape {tpu_moe_arr.shape} incompatible with "
        f"ref_topk_experts shape {ref_moe.shape}"
    )
  if int(tpu_moe_arr.shape[2]) < s_len or int(tpu_moe_arr.shape[1]) < 1:
    raise ValueError(
        f"Sample {sample_name!r}: tpu_moe_topk shape {tpu_moe_arr.shape} has S_pad < seq_len={s_len} or B < 1"
    )

  tok_types_1d: np.ndarray | None = None
  if token_types is not None:
    tok_types_1d = np.asarray(token_types, dtype=np.int32).reshape(-1)
    if int(tok_types_1d.shape[0]) != s_len:
      raise ValueError(f"Sample {sample_name!r}: token_types length {tok_types_1d.shape[0]} != seq_len={s_len}")

  active_moe_layers = list(range(num_layers)) if moe_layer_ids is None else [int(l) for l in moe_layer_ids]
  batch_size = int(tpu_moe_arr.shape[1])
  mult = float(ulp_multiplier)

  high_margin_moe_failures: list[dict[str, Any]] = []
  low_margin_moe_flips: list[dict[str, Any]] = []

  for layer_id in active_moe_layers:
    tpu_layer_arr = tpu_moe_arr[layer_id]
    active_moe_slots = [
        b
        for b in range(batch_size)
        if b == 0 or not bool(np.array_equal(tpu_layer_arr[b, :s_len], tpu_layer_arr[0, :s_len]))
    ]
    for pos in range(s_len):
      ref_set = {int(x) for x in ref_moe[layer_id, pos].tolist()}
      margin = float(margins_f32[layer_id, pos])
      ulp = float(ulp_f32[layer_id, pos])
      margin_ulps = float(margin / ulp) if ulp > 0.0 else float("inf")
      is_high = bool(margin >= mult * ulp)
      tok_t = int(tok_types_1d[pos]) if tok_types_1d is not None else -1
      for b in active_moe_slots:
        tpu_set = {int(x) for x in tpu_layer_arr[b, pos].tolist()}
        if tpu_set != ref_set:
          entry = {
              "layer": int(layer_id),
              "batch_slot": int(b),
              "pos": int(pos),
              "token_type": tok_t,
              "margin": margin,
              "ulp": ulp,
              "margin_ulps": margin_ulps,
              "ref_experts": sorted(ref_set),
              "tpu_experts": sorted(tpu_set),
              "ref_only": sorted(ref_set - tpu_set),
              "tpu_only": sorted(tpu_set - ref_set),
          }
          if is_high:
            high_margin_moe_failures.append(entry)
          else:
            low_margin_moe_flips.append(entry)

  high_margin_indexer_failures: list[dict[str, Any]] = []
  low_margin_indexer_flips: list[dict[str, Any]] = []

  any_indexer_arg = any(
      x is not None
      for x in (
          tpu_indexer_topk,
          ref_indexer_layer_ids,
          ref_indexer_topk,
          ref_indexer_margins,
          ref_indexer_boundary_ulp,
      )
  )
  if any_indexer_arg:  # pylint: disable=too-many-nested-blocks
    if tpu_indexer_topk is None:
      raise ValueError(
          f"Sample {sample_name!r}: tpu_indexer_topk is required when reference indexer routing arrays are provided."
      )
    idx_layer_ids, idx_ref_topk, idx_margins, idx_ulp = validate_raw_ref_indexer_routing(
        ref_indexer_layer_ids,
        ref_indexer_topk,
        ref_indexer_margins,
        ref_indexer_boundary_ulp,
        sample_name=sample_name,
        expected_seq_len=s_len,
        require_margins_and_ulp=True,
    )
    assert idx_margins is not None and idx_ulp is not None

    tpu_idx_map: dict[int, np.ndarray] = {}
    if isinstance(tpu_indexer_topk, Mapping):
      for k_id, v_arr in tpu_indexer_topk.items():
        arr_v = _safe_gather_and_numpy(v_arr, dtype=np.int32)
        if arr_v.ndim == 2:
          arr_v = arr_v[None, :, :]
        tpu_idx_map[int(k_id)] = arr_v
    else:
      arr_all = _safe_gather_and_numpy(tpu_indexer_topk, dtype=np.int32)
      if arr_all.ndim == 3:
        arr_all = arr_all[:, None, :, :]
      if arr_all.ndim != 4 or int(arr_all.shape[0]) != int(idx_layer_ids.shape[0]):
        raise ValueError(
            f"Sample {sample_name!r}: tpu_indexer_topk array shape {arr_all.shape} incompatible with "
            f"ref_indexer_layer_ids {idx_layer_ids.tolist()}"
        )
      for idx_i, layer_id_raw in enumerate(idx_layer_ids.tolist()):
        tpu_idx_map[int(layer_id_raw)] = arr_all[idx_i]

    for idx_i, layer_id_raw in enumerate(idx_layer_ids.tolist()):
      layer_id = int(layer_id_raw)
      if layer_id not in tpu_idx_map:
        raise ValueError(
            f"Sample {sample_name!r}: missing layer {layer_id} in tpu_indexer_topk (keys={sorted(tpu_idx_map.keys())})"
        )
      tpu_layer_arr = tpu_idx_map[layer_id]
      if tpu_layer_arr.ndim != 3 or int(tpu_layer_arr.shape[1]) < s_len or int(tpu_layer_arr.shape[0]) < 1:
        raise ValueError(
            f"Sample {sample_name!r}: tpu_indexer_topk[{layer_id}] shape {tpu_layer_arr.shape} invalid for seq_len={s_len}"
        )
      idx_batch_size = int(tpu_layer_arr.shape[0])
      active_idx_slots = [
          b
          for b in range(idx_batch_size)
          if b == 0 or not bool(np.array_equal(tpu_layer_arr[b, :s_len], tpu_layer_arr[0, :s_len]))
      ]
      ref_layer_slice = _normalize_ref_indexer_topk_slice(idx_ref_topk[idx_i], s_len, layer_id=layer_id)
      for pos in range(s_len):
        ref_row = ref_layer_slice[pos]
        ref_set = {int(x) for x in ref_row.tolist() if int(x) >= 0}
        margin_val = float(idx_margins[idx_i, pos])
        ulp_val = float(idx_ulp[idx_i, pos])
        is_unpruned = bool(np.isnan(margin_val))
        tok_t = int(tok_types_1d[pos]) if tok_types_1d is not None else -1
        for b in active_idx_slots:
          tpu_row = tpu_layer_arr[b, pos]
          tpu_set = {int(x) for x in tpu_row.tolist() if int(x) >= 0}
          if tpu_set != ref_set:
            if is_unpruned:
              high_margin_indexer_failures.append(
                  {
                      "layer": layer_id,
                      "batch_slot": int(b),
                      "pos": int(pos),
                      "token_type": tok_t,
                      "reason": "unpruned_reachable_mismatch",
                      "margin": None,
                      "ulp": None,
                      "margin_ulps": None,
                      "ref_count": len(ref_set),
                      "tpu_count": len(tpu_set),
                      "ref_only": sorted(ref_set - tpu_set)[:16],
                      "tpu_only": sorted(tpu_set - ref_set)[:16],
                  }
              )
            else:
              margin_ulps = float(margin_val / ulp_val) if ulp_val > 0.0 else float("inf")
              entry_idx = {
                  "layer": layer_id,
                  "batch_slot": int(b),
                  "pos": int(pos),
                  "token_type": tok_t,
                  "reason": "high_margin_mismatch" if margin_val >= mult * ulp_val else "low_margin_flip",
                  "margin": margin_val,
                  "ulp": ulp_val,
                  "margin_ulps": margin_ulps,
                  "ref_count": len(ref_set),
                  "tpu_count": len(tpu_set),
                  "ref_only": sorted(ref_set - tpu_set)[:16],
                  "tpu_only": sorted(tpu_set - ref_set)[:16],
              }
              if margin_val >= mult * ulp_val:
                high_margin_indexer_failures.append(entry_idx)
              else:
                low_margin_indexer_flips.append(entry_idx)

  passed = len(high_margin_moe_failures) == 0 and len(high_margin_indexer_failures) == 0
  summary: dict[str, Any] = {
      "passed": bool(passed),
      "ulp_multiplier": mult,
      "high_margin_moe_failure_count": len(high_margin_moe_failures),
      "high_margin_moe_failures": high_margin_moe_failures,
      "low_margin_moe_flip_count": len(low_margin_moe_flips),
      "low_margin_moe_flips": low_margin_moe_flips,
      "high_margin_indexer_failure_count": len(high_margin_indexer_failures),
      "high_margin_indexer_failures": high_margin_indexer_failures,
      "low_margin_indexer_flip_count": len(low_margin_indexer_flips),
      "low_margin_indexer_flips": low_margin_indexer_flips,
      "moe_router": {
          "passed": len(high_margin_moe_failures) == 0,
          "mismatches_above_threshold": len(high_margin_moe_failures),
          "mismatches_below_threshold": len(low_margin_moe_flips),
      },
      "sparse_attention_indexer": {
          "passed": len(high_margin_indexer_failures) == 0,
          "mismatches_above_threshold": len(high_margin_indexer_failures),
          "mismatches_below_threshold": len(low_margin_indexer_flips),
      },
  }
  if not passed and raise_on_failure:
    details: list[str] = []
    if high_margin_moe_failures:
      first_moe = high_margin_moe_failures[0]
      details.append(
          f"high_margin_moe_failures={len(high_margin_moe_failures)} "
          f"(first: layer={first_moe['layer']}, slot={first_moe['batch_slot']}, pos={first_moe['pos']}, "
          f"token_type={first_moe['token_type']}, margin={first_moe['margin']:.3e}, "
          f"ulp={first_moe['ulp']:.3e}, margin_ulps={first_moe['margin_ulps']:.2f}, "
          f"ref_only={first_moe['ref_only']}, tpu_only={first_moe['tpu_only']})"
      )
    if high_margin_indexer_failures:
      first_idx = high_margin_indexer_failures[0]
      details.append(
          f"high_margin_indexer_failures={len(high_margin_indexer_failures)} "
          f"(first: layer={first_idx['layer']}, slot={first_idx['batch_slot']}, pos={first_idx['pos']}, "
          f"token_type={first_idx['token_type']}, reason={first_idx['reason']}, "
          f"margin={first_idx['margin']}, ulp={first_idx['ulp']}, margin_ulps={first_idx['margin_ulps']}, "
          f"ref_only={first_idx['ref_only']}, tpu_only={first_idx['tpu_only']})"
      )
    err = AssertionError(
        f"Sample {sample_name!r} FAILED Option C Router-Agreement Gate (>= {mult:.1f} ULP): " + "; ".join(details)
    )
    setattr(err, "router_agreement_summary", summary)
    raise err
  return summary


def format_router_agreement_failure_reasons(router_agreement: Mapping[str, Any]) -> list[str]:
  """Formats compact failure strings for failed Option C router-agreement checks."""
  reasons: list[str] = []
  mult = float(router_agreement.get("ulp_multiplier", DEFAULT_ROUTER_AGREEMENT_ULP_MULTIPLIER))
  moe_fails = router_agreement.get("high_margin_moe_failures") or []
  if moe_fails:
    first_moe = moe_fails[0]
    reasons.append(
        f"router_agreement.moe_high_margin_failures={len(moe_fails)} (>= {mult:.1f} ULP; "
        f"first: layer={first_moe['layer']}, slot={first_moe['batch_slot']}, pos={first_moe['pos']}, "
        f"token_type={first_moe['token_type']}, margin_ulps={first_moe['margin_ulps']:.2f}, "
        f"ref_only={first_moe['ref_only']}, tpu_only={first_moe['tpu_only']})"
    )
  idx_fails = router_agreement.get("high_margin_indexer_failures") or []
  if idx_fails:
    first_idx = idx_fails[0]
    reasons.append(
        f"router_agreement.indexer_high_margin_failures={len(idx_fails)} (>= {mult:.1f} ULP; "
        f"first: layer={first_idx['layer']}, slot={first_idx['batch_slot']}, pos={first_idx['pos']}, "
        f"token_type={first_idx['token_type']}, reason={first_idx['reason']}, "
        f"margin_ulps={first_idx['margin_ulps']}, ref_only={first_idx['ref_only']}, tpu_only={first_idx['tpu_only']})"
    )
  return reasons


def is_mm_v3_golden_path(path_str: str | os.PathLike[str] | None) -> bool:
  """Returns True when `path_str` points to an `mm_v3` golden file or directory."""
  if not path_str:
    return False
  p = Path(str(path_str))
  if "mm_v3" in p.parts:
    return True
  cand_manifests: list[Path] = []
  if p.is_dir():
    cand_manifests.append(p / "manifest.json")
    npz_files = sorted(p.glob("*.npz"))
    if npz_files:
      cand_manifests.append(npz_files[0].resolve().parent / "manifest.json")
  else:
    cand_manifests.append(p.parent / "manifest.json")
    if p.exists():
      cand_manifests.append(p.resolve().parent / "manifest.json")
  for mpath in cand_manifests:
    if mpath.is_file():
      with contextlib.suppress(OSError, ValueError, TypeError):
        with open(mpath, "r", encoding="utf-8") as f:
          mdata = json.load(f)
        if isinstance(mdata, Mapping) and (
            mdata.get("golden_version") == "mm_v3" or mdata.get("schema_version") == "ds41_mm_golden_v3"
        ):
          return True
  return False


OPTION_C_REQUIRED_ROUTING_KEYS: tuple[str, ...] = (
    "ref_topk_experts",
    "ref_gate_margins",
    "ref_gate_boundary_ulp",
    "ref_indexer_layer_ids",
    "ref_indexer_topk",
    "ref_indexer_margins",
    "ref_indexer_boundary_ulp",
)


def has_option_c_routing_fields(golden_data_point: Mapping[str, Any]) -> bool:
  """Returns True when `golden_data_point` carries all 7 Option C MoE & Indexer routing & margin/ULP arrays."""
  return all(k in golden_data_point and golden_data_point[k] is not None for k in OPTION_C_REQUIRED_ROUTING_KEYS)


def extract_and_validate_option_c_fields(
    raw_source: Mapping[str, Any],
    *,
    sample_name: str,
    expected_seq_len: int,
    is_multimodal: bool,
    require_option_c: bool = False,
) -> dict[str, Any]:
  """Extracts and validates `ref_topk_experts` and Option C margin/ULP/indexer fields from an `.npz` or dict record."""
  out: dict[str, Any] = {}
  has_moe_topk = "ref_topk_experts" in raw_source and raw_source["ref_topk_experts"] is not None
  has_moe_margins = "ref_gate_margins" in raw_source and raw_source["ref_gate_margins"] is not None
  has_moe_ulp = "ref_gate_boundary_ulp" in raw_source and raw_source["ref_gate_boundary_ulp"] is not None
  has_idx_ids = "ref_indexer_layer_ids" in raw_source and raw_source["ref_indexer_layer_ids"] is not None
  has_idx_topk = "ref_indexer_topk" in raw_source and raw_source["ref_indexer_topk"] is not None
  has_idx_margins = "ref_indexer_margins" in raw_source and raw_source["ref_indexer_margins"] is not None
  has_idx_ulp = "ref_indexer_boundary_ulp" in raw_source and raw_source["ref_indexer_boundary_ulp"] is not None

  any_option_c_extra = any((has_moe_margins, has_moe_ulp, has_idx_ids, has_idx_topk, has_idx_margins, has_idx_ulp))

  if has_moe_topk:
    out["ref_topk_experts"] = validate_raw_ref_topk_experts(
        raw_source["ref_topk_experts"],
        sample_name=sample_name,
        expected_seq_len=int(expected_seq_len),
    )

  if is_multimodal and (require_option_c or any_option_c_extra):
    missing_moe = [
        k
        for k, present in (
            ("ref_topk_experts", has_moe_topk),
            ("ref_gate_margins", has_moe_margins),
            ("ref_gate_boundary_ulp", has_moe_ulp),
        )
        if not present
    ]
    if missing_moe:
      raise ValueError(
          f"Sample {sample_name!r}: Option C multimodal golden requires ref_topk_experts, ref_gate_margins, "
          f"and ref_gate_boundary_ulp; missing {missing_moe}"
      )
    num_layers = int(out["ref_topk_experts"].shape[0])
    margins_f32, ulp_f32 = validate_raw_ref_gate_margins_and_ulp(
        raw_source["ref_gate_margins"],
        raw_source["ref_gate_boundary_ulp"],
        sample_name=sample_name,
        expected_num_layers=num_layers,
        expected_seq_len=int(expected_seq_len),
    )
    out["ref_gate_margins"] = margins_f32
    out["ref_gate_boundary_ulp"] = ulp_f32

    missing_idx = [
        k
        for k, present in (
            ("ref_indexer_layer_ids", has_idx_ids),
            ("ref_indexer_topk", has_idx_topk),
            ("ref_indexer_margins", has_idx_margins),
            ("ref_indexer_boundary_ulp", has_idx_ulp),
        )
        if not present
    ]
    if missing_idx:
      raise ValueError(
          f"Sample {sample_name!r}: Option C multimodal golden requires complete indexer routing arrays "
          f"(ref_indexer_layer_ids, ref_indexer_topk, ref_indexer_margins, ref_indexer_boundary_ulp); "
          f"missing {missing_idx}"
      )
    idx_ids, idx_topk, idx_margins, idx_ulp = validate_raw_ref_indexer_routing(
        raw_source["ref_indexer_layer_ids"],
        raw_source["ref_indexer_topk"],
        raw_source["ref_indexer_margins"],
        raw_source["ref_indexer_boundary_ulp"],
        sample_name=sample_name,
        expected_seq_len=int(expected_seq_len),
        require_margins_and_ulp=True,
    )
    out["ref_indexer_layer_ids"] = idx_ids
    out["ref_indexer_topk"] = idx_topk
    out["ref_indexer_margins"] = idx_margins
    out["ref_indexer_boundary_ulp"] = idx_ulp

  return out
