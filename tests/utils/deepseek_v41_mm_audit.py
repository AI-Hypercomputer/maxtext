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

"""Streaming HuggingFace-to-MaxText weight, dtype, and sharding audit."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
from typing import Any

from etils import epath
from flax import nnx
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from maxtext.common import checkpointing
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.utils import model_creation_utils

# pylint: disable=possibly-used-before-assignment
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
  import torch  # pylint: disable=g-import-not-at-top

  from tests.utils.deepseek_v41_reference import runner as ref_runner  # pylint: disable=g-import-not-at-top
else:
  torch = None
  ref_runner = None


@dataclasses.dataclass(frozen=True)
class LeafAuditRule:
  """Independent audit rule for a single MaxText parameter or MoEBiasVar leaf."""

  maxtext_path: str
  category: str  # "multimodal" or "text"
  kind: str  # "single", "expert_stack", "scanned_single", "scanned_expert_stack"
  hf_keys: tuple[Any, ...]
  transform: str
  expected_mt_dtype: str
  expected_hf_header_dtype: str | None = None
  forbid_scale: bool = False
  engram_layer_ids: tuple[int, ...] = ()


_SAFETENSORS_DTYPES: dict[str, tuple[Any, int]] = (
    {
        "BF16": (torch.bfloat16, 2),
        "F32": (torch.float32, 4),
        "F16": (torch.float16, 2),
        "F64": (torch.float64, 8),
        "I8": (torch.int8, 1),
        "U8": (torch.uint8, 1),
        "I16": (torch.int16, 2),
        "U16": (torch.uint16, 2),
        "I32": (torch.int32, 4),
        "I64": (torch.int64, 8),
        "BOOL": (torch.bool, 1),
        "F8_E4M3": (torch.float8_e4m3fn, 1),
        "F8_E5M2": (torch.float8_e5m2, 1),
        "F8_E8M0": (torch.float8_e8m0fnu, 1),
    }
    if HAS_TORCH
    else {
        "BF16": (ml_dtypes.bfloat16, 2),
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "F64": (np.float64, 8),
        "I8": (np.int8, 1),
        "U8": (np.uint8, 1),
        "I16": (np.int16, 2),
        "U16": (np.uint16, 2),
        "I32": (np.int32, 4),
        "I64": (np.int64, 8),
        "BOOL": (bool, 1),
        "F8_E4M3": (ml_dtypes.float8_e4m3fn, 1),
        "F8_E5M2": (ml_dtypes.float8_e5m2, 1),
        "F8_E8M0": (np.uint8, 1),
    }
)


def _sha256_file(path: os.PathLike[str] | str) -> str:
  h = hashlib.sha256()
  with epath.Path(str(path)).open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _normalize_orbax_root(export_orbax_dir: str | os.PathLike[str]) -> str:
  """Normalizes an Orbax export root directory, preserving `gs://` URIs and stripping `/0/items`."""
  export_path = epath.Path(str(export_orbax_dir))
  if export_path.name == "items" and export_path.parent.name == "0":
    mgr_root = export_path.parent.parent
  else:
    mgr_root = export_path
  return mgr_root.as_posix().rstrip("/")


_AUDIT_IO_MAX_WORKERS = 32
_AUDIT_SINGLE_MAX_WORKERS = 4
_EXPERT_AUDIT_CHUNK_SIZE = 64


class _ThreadLocalHandles(threading.local):
  """Per-thread dictionary mapping normalized shard path to `(exit_stack, entered_stream)`."""

  def __init__(self) -> None:
    super().__init__()
    self.handles: dict[str, tuple[contextlib.ExitStack, Any]] = {}


def _close_cm_quietly(stack: contextlib.ExitStack) -> None:
  """Closes an `ExitStack` holding an open shard binary stream, ignoring close-time I/O errors."""
  with contextlib.suppress(OSError, ValueError):
    stack.close()


def _read_exact_from_stream(handle: Any, abs_offset: int, nbytes: int) -> bytearray:
  """Seeks `handle` to `abs_offset` and reads up to `nbytes`, looping across partial reads until EOF."""
  handle.seek(abs_offset)
  first = handle.read(nbytes)
  if not first:
    return bytearray()
  if len(first) == nbytes:
    return bytearray(first)
  buf = bytearray(first)
  remaining = nbytes - len(first)
  while remaining > 0:
    chunk = handle.read(remaining)
    if not chunk:
      break
    buf.extend(chunk)
    remaining -= len(chunk)
  return buf


def _read_safetensors_header(
    shard_path: epath.Path,
) -> tuple[epath.Path, int, dict[str, dict[str, Any]]]:
  """Reads the JSON header and binary data start offset of a safetensors shard."""
  with shard_path.open("rb") as f:
    raw_len = _read_exact_from_stream(f, 0, 8)
    if len(raw_len) != 8:
      raise ValueError(f"Truncated safetensors header in {shard_path}")
    header_len = int.from_bytes(raw_len, "little")
    raw_hdr_bytes = _read_exact_from_stream(f, 8, header_len)
    if len(raw_hdr_bytes) != header_len:
      raise ValueError(
          f"Truncated safetensors header body in {shard_path}: got {len(raw_hdr_bytes)} bytes, expected {header_len}"
      )
    raw_hdr = json.loads(bytes(raw_hdr_bytes))
  hdr = {k: v for k, v in raw_hdr.items() if k != "__metadata__"}
  return shard_path, 8 + header_len, hdr


class _SafetensorsShardReader:
  """Caches safetensors header metadata across local or `gs://` shards and loads tensors or row slices."""

  def __init__(
      self,
      checkpoint_dir: str | os.PathLike[str],
      *,
      key_filter_re: re.Pattern[str] | None = None,
  ):
    self.checkpoint_dir = epath.Path(str(checkpoint_dir))
    index_path = self.checkpoint_dir / "model.safetensors.index.json"
    single_path = self.checkpoint_dir / "model.safetensors"
    self.weight_map: dict[str, str] = {}
    self._headers: dict[str, dict[str, dict[str, Any]]] = {}
    self._shard_info: dict[str, tuple[epath.Path, int, dict[str, dict[str, Any]]]] = {}
    self._tensor_cache: dict[str, np.ndarray] = {}
    self._thread_local = _ThreadLocalHandles()
    self._handles_lock = threading.Lock()
    self._open_handles: list[contextlib.ExitStack] = []

    if index_path.exists():
      with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
      full_weight_map = dict(data["weight_map"])
      if key_filter_re is not None:
        self.weight_map = {k: v for k, v in full_weight_map.items() if key_filter_re.match(k) is not None}
      else:
        self.weight_map = full_weight_map
      shard_files = sorted(set(self.weight_map.values()))
    elif single_path.exists():
      shard_files = ["model.safetensors"]
    else:
      shard_files = sorted(p.name for p in self.checkpoint_dir.glob("*.safetensors"))
      if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {self.checkpoint_dir}")

    max_workers = min(16, max(1, len(shard_files)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
      future_to_name = {pool.submit(_read_safetensors_header, self.checkpoint_dir / name): name for name in shard_files}
      for fut in concurrent.futures.as_completed(future_to_name):
        name = future_to_name[fut]
        shard_path, data_start, hdr = fut.result()
        self._shard_info[name] = (shard_path, data_start, hdr)
        self._headers[name] = hdr

    if not self.weight_map:
      for name in shard_files:
        for k in self._headers[name]:
          if key_filter_re is None or key_filter_re.match(k) is not None:
            self.weight_map[k] = name

  def __enter__(self) -> "_SafetensorsShardReader":
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.close()

  def close(self) -> None:
    """Closes all open per-thread shard file context managers and clears cached tensors."""
    with self._handles_lock:
      open_cms = list(self._open_handles)
      self._open_handles.clear()
      self._thread_local.handles.clear()
      self._thread_local = _ThreadLocalHandles()
    for cm in open_cms:
      _close_cm_quietly(cm)
    self._tensor_cache.clear()

  def clear_cache(self) -> None:
    self._tensor_cache.clear()

  def all_keys(self) -> set[str]:
    return set(self.weight_map.keys())

  def _open_and_register_handle(self, shard_path: epath.Path) -> tuple[contextlib.ExitStack, Any]:
    stack = contextlib.ExitStack()
    try:
      stream = stack.enter_context(shard_path.open("rb"))
    except BaseException:
      _close_cm_quietly(stack)
      raise
    with self._handles_lock:
      self._open_handles.append(stack)
    return stack, stream

  def _read_shard_bytes(self, shard_path: epath.Path, abs_offset: int, nbytes: int) -> bytearray:
    """Reads exact `nbytes` from `shard_path` at `abs_offset` using a persistent per-thread stream."""
    if nbytes == 0:
      return bytearray()
    shard_key = shard_path.as_posix()
    handles = self._thread_local.handles
    if shard_key in handles:
      old_cm, stream = handles[shard_key]
      try:
        data = _read_exact_from_stream(stream, abs_offset, nbytes)
      except OSError:
        del handles[shard_key]
        with self._handles_lock:
          if old_cm in self._open_handles:
            self._open_handles.remove(old_cm)
        _close_cm_quietly(old_cm)
        reopened_cm, reopened_stream = self._open_and_register_handle(shard_path)
        handles[shard_key] = (reopened_cm, reopened_stream)
        data = _read_exact_from_stream(reopened_stream, abs_offset, nbytes)
    else:
      opened_cm, opened_stream = self._open_and_register_handle(shard_path)
      handles[shard_key] = (opened_cm, opened_stream)
      data = _read_exact_from_stream(opened_stream, abs_offset, nbytes)
    if len(data) != nbytes:
      raise ValueError(f"Short read from {shard_path} at offset {abs_offset}: got {len(data)} bytes, expected {nbytes}")
    return data

  def _get_shard_and_entry(
      self, key: str
  ) -> tuple[epath.Path, int, dict[str, Any], torch.dtype, int, tuple[int, ...], int]:
    """Returns validated shard path, data offset, header entry, dtype, itemsize, shape, and relative offset."""
    if key not in self.weight_map:
      raise KeyError(f"HF key {key!r} missing from safetensors weight_map")
    shard_name = self.weight_map[key]
    if shard_name not in self._shard_info:
      with self._handles_lock:
        if shard_name not in self._shard_info:
          shard_path, data_start, hdr = _read_safetensors_header(self.checkpoint_dir / shard_name)
          self._shard_info[shard_name] = (shard_path, data_start, hdr)
          self._headers[shard_name] = hdr
    shard_path, data_start, hdr_map = self._shard_info[shard_name]
    if key not in hdr_map:
      raise KeyError(f"HF key {key!r} missing from shard header {shard_name!r}")
    entry = hdr_map[key]
    dtype_code = str(entry["dtype"])
    if dtype_code not in _SAFETENSORS_DTYPES:
      raise ValueError(f"Unsupported safetensors dtype {dtype_code!r} for {key!r}")
    torch_dtype, itemsize = _SAFETENSORS_DTYPES[dtype_code]
    shape = tuple(int(d) for d in entry["shape"])
    begin, end = int(entry["data_offsets"][0]), int(entry["data_offsets"][1])
    expected_nbytes = math.prod(shape) * itemsize
    if end - begin != expected_nbytes:
      raise ValueError(
          f"Safetensors byte length mismatch for {key!r}: end-begin={end - begin}, expected={expected_nbytes}"
      )
    return shard_path, data_start, entry, torch_dtype, itemsize, shape, begin

  def get_header_entry(self, key: str) -> dict[str, Any]:
    """Returns the parsed safetensors header entry for the given tensor key."""
    _, _, entry, _, _, _, _ = self._get_shard_and_entry(key)
    return entry

  def load_raw_tensor(self, key: str) -> Any:
    """Loads a raw tensor by key from safetensors as torch tensor or numpy array."""
    shard_path, data_start, _, torch_dtype, itemsize, shape, begin = self._get_shard_and_entry(key)
    nbytes = math.prod(shape) * itemsize
    if HAS_TORCH and torch is not None:
      if nbytes == 0:
        return torch.empty(shape, dtype=torch_dtype, device="cpu")
      data = self._read_shard_bytes(shard_path, data_start + begin, nbytes)
      return torch.frombuffer(data, dtype=torch_dtype).reshape(shape)
    if nbytes == 0:
      return np.empty(shape, dtype=torch_dtype)
    data = self._read_shard_bytes(shard_path, data_start + begin, nbytes)
    return np.frombuffer(data, dtype=torch_dtype).reshape(shape)

  def load_raw_row_slice(self, key: str, row_start: int, row_stop: int) -> torch.Tensor:
    """Reads raw rows `[row_start:row_stop]` of tensor `key` without loading the full tensor."""
    shard_path, data_start, _, torch_dtype, itemsize, shape, begin = self._get_shard_and_entry(key)
    if not shape:
      raise ValueError(f"Cannot row-slice scalar tensor {key!r}")
    if row_start < 0 or row_stop < row_start or row_stop > shape[0]:
      raise ValueError(f"Invalid row slice [{row_start}:{row_stop}] for {key!r} with shape {shape}")
    row_stride_bytes = math.prod(shape[1:]) * itemsize
    rel_offset = row_start * row_stride_bytes
    nbytes = (row_stop - row_start) * row_stride_bytes
    out_shape = (row_stop - row_start,) + shape[1:]
    if HAS_TORCH and torch is not None:
      if nbytes == 0:
        return torch.empty(out_shape, dtype=torch_dtype, device="cpu")
      data = self._read_shard_bytes(shard_path, data_start + begin + rel_offset, nbytes)
      return torch.frombuffer(data, dtype=torch_dtype).reshape(out_shape)
    if nbytes == 0:
      return np.empty(out_shape, dtype=torch_dtype)
    data = self._read_shard_bytes(shard_path, data_start + begin + rel_offset, nbytes)
    return np.frombuffer(data, dtype=torch_dtype).reshape(out_shape)

  def load_dequantized_tensor(
      self,
      key: str,
      target_dtype_str: str,
      *,
      use_cache: bool = True,
  ) -> np.ndarray:
    """Loads `key` from safetensors, dequantizing with sibling `.scale` if present."""
    cache_key = f"{key}:{target_dtype_str}"
    if use_cache and cache_key in self._tensor_cache:
      return self._tensor_cache[cache_key]
    raw = self.load_raw_tensor(key)
    for scale_key in _candidate_scale_keys(key):
      if scale_key in self.weight_map:
        if not HAS_TORCH or torch is None or ref_runner is None:
          raise ImportError(
              f"PyTorch is required to dequantize scaled safetensors key {key!r} (found scale {scale_key!r})."
          )
        scale = self.load_raw_tensor(scale_key)
        out_cols = raw.shape[1] * 2 if raw.dtype == torch.int8 else raw.shape[1]
        out = torch.empty((raw.shape[0], out_cols), dtype=torch.bfloat16, device="cpu")
        ref_runner.dequantize_into(out, raw, scale)
        raw = out
        break
    arr = _torch_to_numpy_exact(raw, target_dtype_str)
    if use_cache:
      self._tensor_cache[cache_key] = arr
    return arr

  def load_dequantized_row_slice(
      self,
      key: str,
      target_dtype_str: str,
      row_start: int,
      row_stop: int,
  ) -> np.ndarray:
    """Loads and dequantizes only rows `[row_start:row_stop]` of a 2D row-major tensor `key`."""
    _, _, _, torch_dtype, _, shape, _ = self._get_shard_and_entry(key)
    if len(shape) != 2:
      raise ValueError(f"load_dequantized_row_slice requires a 2D tensor, got {key!r} with shape {shape}")
    raw_rows, raw_cols = shape
    scale_key_found: str | None = None
    for cand in _candidate_scale_keys(key):
      if cand in self.weight_map:
        scale_key_found = cand
        break
    if scale_key_found is not None and (not HAS_TORCH or torch is None or ref_runner is None):
      raise ImportError(
          f"PyTorch is required to dequantize scaled safetensors key {key!r} (found scale {scale_key_found!r})."
      )
    cols = raw_cols * 2 if (scale_key_found is not None and torch_dtype == torch.int8) else raw_cols
    target_np_dtype = ml_dtypes.bfloat16 if target_dtype_str == "bfloat16" else np.float32
    f0 = min(max(0, int(row_start)), raw_rows)
    f1 = min(max(f0, int(row_stop)), raw_rows)
    if f0 == f1:
      return np.zeros((0, cols), dtype=target_np_dtype)

    cache_key = f"{key}:{target_dtype_str}"
    if cache_key in self._tensor_cache:
      return self._tensor_cache[cache_key][f0:f1]

    if scale_key_found is not None:
      _, _, _, _, _, s_shape, _ = self._get_shard_and_entry(scale_key_found)
      s_rows = int(s_shape[0])
      block_rows = -(-raw_rows // s_rows)
      b0 = (f0 // block_rows) * block_rows
      b1 = min(raw_rows, -(-f1 // block_rows) * block_rows)
      sb0 = b0 // block_rows
      sb1 = -(-b1 // block_rows)
      raw_rows_tensor = self.load_raw_row_slice(key, b0, b1)
      scale_rows_tensor = self.load_raw_row_slice(scale_key_found, sb0, sb1)
      if (b1 - b0) != (sb1 - sb0) * block_rows:
        scale_rows_tensor = scale_rows_tensor.repeat_interleave(block_rows, dim=0)[: b1 - b0]
      out = torch.empty((b1 - b0, cols), dtype=torch.bfloat16, device="cpu")
      ref_runner.dequantize_into(out, raw_rows_tensor, scale_rows_tensor)
      deq_np = _torch_to_numpy_exact(out, target_dtype_str)
      return deq_np[f0 - b0 : f1 - b0]

    raw_slice = self.load_raw_row_slice(key, f0, f1)
    return _torch_to_numpy_exact(raw_slice, target_dtype_str)


def _candidate_scale_keys(hf_k: str) -> tuple[str, ...]:
  """Returns possible `.scale` sibling key names for `hf_k` in HF safetensors."""
  if hf_k.endswith(".weight"):
    return (f"{hf_k[:-len('.weight')]}.scale", f"{hf_k}.scale")
  return (f"{hf_k}.scale",)


def _torch_to_numpy_exact(tensor: Any, target_dtype_str: str) -> np.ndarray:
  """Converts a torch tensor or numpy array to target numpy array dtype without loss."""
  if isinstance(tensor, np.ndarray):
    if target_dtype_str == "bfloat16":
      return tensor.astype(ml_dtypes.bfloat16)
    if target_dtype_str == "float32":
      return tensor.astype(np.float32)
    raise ValueError(f"Unsupported target_dtype_str: {target_dtype_str!r}")
  t = tensor.detach().cpu().contiguous()
  if target_dtype_str == "bfloat16":
    t_bf16 = t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)
    return t_bf16.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
  if target_dtype_str == "float32":
    t_f32 = t if t.dtype == torch.float32 else t.to(torch.float32)
    return t_f32.numpy()
  raise ValueError(f"Unsupported target_dtype_str: {target_dtype_str!r}")


def _layer_single_specs(
    layer_idx: int,
    spec: deepseek41.LayerSpec,
    use_multimodal: bool,
    weight_dtype_str: str,
) -> list[tuple[str, str, str, str, str, str | None, bool]]:
  """Returns `(mt_subpath, hf_key, category, transform, mt_dtype, hf_hdr_dtype, forbid_scale)` for 1-to-1 leaves."""
  pfx = f"layers.{layer_idx}"
  items: list[tuple[str, str, str, str, str, str | None, bool]] = [
      (
          "pre_self_attention_layer_norm/scale",
          f"{pfx}.attn_norm.weight",
          "text",
          "identity",
          weight_dtype_str,
          None,
          True,
      ),
      (
          "post_self_attention_layer_norm/scale",
          f"{pfx}.ffn_norm.weight",
          "text",
          "identity",
          weight_dtype_str,
          None,
          True,
      ),
      ("hc_attn_fn", f"{pfx}.hc_attn_fn", "text", "transpose_2d", "float32", None, True),
      ("hc_ffn_fn", f"{pfx}.hc_ffn_fn", "text", "transpose_2d", "float32", None, True),
      ("hc_attn_base", f"{pfx}.hc_attn_base", "text", "identity", "float32", None, True),
      ("hc_ffn_base", f"{pfx}.hc_ffn_base", "text", "identity", "float32", None, True),
      ("hc_attn_scale", f"{pfx}.hc_attn_scale", "text", "identity", "float32", None, True),
      ("hc_ffn_scale", f"{pfx}.hc_ffn_scale", "text", "identity", "float32", None, True),
      ("self_attention/sinks", f"{pfx}.attn.attn_sink", "text", "identity", "float32", None, True),
      (
          "self_attention/wq_a/kernel",
          f"{pfx}.attn.wq_a.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "self_attention/q_norm/scale",
          f"{pfx}.attn.q_norm.weight",
          "text",
          "identity",
          weight_dtype_str,
          None,
          True,
      ),
      (
          "self_attention/wq_b/kernel",
          f"{pfx}.attn.wq_b.weight",
          "text",
          "wq_b_reshape",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "self_attention/wkv/kernel",
          f"{pfx}.attn.wkv.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "self_attention/kv_norm/scale",
          f"{pfx}.attn.kv_norm.weight",
          "text",
          "identity",
          weight_dtype_str,
          None,
          True,
      ),
      (
          "self_attention/o_a_proj/kernel",
          f"{pfx}.attn.wo_a.weight",
          "text",
          "wo_a_reshape",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "self_attention/o_b_proj/kernel",
          f"{pfx}.attn.wo_b.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "mlp/MoeBlock_0/gate/kernel",
          f"{pfx}.ffn.gate.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          True,
      ),
      ("mlp/MoeBlock_0/gate/bias", f"{pfx}.ffn.gate.bias", "text", "identity", "float32", "F32", True),
      (
          "mlp/shared_experts/wi_0/kernel",
          f"{pfx}.ffn.shared_experts.w1.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "mlp/shared_experts/wi_1/kernel",
          f"{pfx}.ffn.shared_experts.w3.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          False,
      ),
      (
          "mlp/shared_experts/wo/kernel",
          f"{pfx}.ffn.shared_experts.w2.weight",
          "text",
          "transpose_2d",
          weight_dtype_str,
          None,
          False,
      ),
  ]
  if use_multimodal:
    items.append(
        (
            "mlp/MoeBlock_0/gate/bias_vl",
            f"{pfx}.ffn.gate.bias_vl",
            "multimodal",
            "identity",
            "float32",
            "F32",
            True,
        )
    )
  if spec.is_kv_source:
    items.append(
        (
            "self_attention/compressor/norm/scale",
            f"{pfx}.attn.compressor.norm.weight",
            "text",
            "identity",
            weight_dtype_str,
            None,
            True,
        )
    )
    items.append(
        (
            "self_attention/compressor/wkv/kernel",
            f"{pfx}.attn.compressor.wkv.weight",
            "text",
            "transpose_2d",
            weight_dtype_str,
            None,
            False,
        )
    )
    if spec.compress_ratio > 1:
      items.append(
          (
              "self_attention/compressor/wgate/kernel",
              f"{pfx}.attn.compressor.wgate.weight",
              "text",
              "transpose_2d",
              weight_dtype_str,
              None,
              False,
          )
      )
  if spec.is_index_source:
    items.append(
        (
            "self_attention/indexer/wq_b/kernel",
            f"{pfx}.attn.indexer.wq_b.weight",
            "text",
            "indexer_wq_b_reshape",
            weight_dtype_str,
            None,
            False,
        )
    )
    items.append(
        (
            "self_attention/indexer/weights_proj/kernel",
            f"{pfx}.attn.indexer.weights_proj.weight",
            "text",
            "transpose_2d",
            weight_dtype_str,
            None,
            True,
        )
    )
    if spec.is_kv_source:
      items.append(
          (
              "self_attention/indexer/wk/kernel",
              f"{pfx}.attn.indexer.wk.weight",
              "text",
              "transpose_2d",
              weight_dtype_str,
              None,
              False,
          )
      )
      items.append(
          (
              "self_attention/indexer/k_norm/scale",
              f"{pfx}.attn.indexer.k_norm.weight",
              "text",
              "identity",
              weight_dtype_str,
              None,
              True,
          )
      )
  if spec.has_engram:
    items.append(("engram/q_weight", f"{pfx}.engram.q_weight", "text", "identity", weight_dtype_str, None, True))
    items.append(("engram/k_weight", f"{pfx}.engram.k_weight", "text", "identity", weight_dtype_str, None, True))
    items.append(("engram/wkv/kernel", f"{pfx}.engram.wkv.weight", "text", "transpose_2d", weight_dtype_str, None, False))
    items.append(("engram/embed", f"{pfx}.engram.embed.weight", "text", "engram_pad", weight_dtype_str, None, False))
  return items


def build_independent_audit_rules(config: Any) -> list[LeafAuditRule]:
  """Builds the complete list of independent HF-to-MaxText leaf audit rules for `config`."""
  weight_dtype_str = "float32" if str(config.weight_dtype) in ("float32", "jax.numpy.float32") else "bfloat16"
  use_mm = bool(config.use_multimodal)
  rules: list[LeafAuditRule] = [
      LeafAuditRule(
          "token_embedder/embedding", "text", "single", ("embed.weight",), "identity", weight_dtype_str, None, True
      ),
      LeafAuditRule(
          "decoder/decoder_norm/scale",
          "text",
          "single",
          ("norm.weight",),
          "identity",
          weight_dtype_str,
          None,
          True,
      ),
      LeafAuditRule(
          "decoder/logits_dense/kernel",
          "text",
          "single",
          ("head.weight",),
          "transpose_2d",
          weight_dtype_str,
          None,
          True,
      ),
  ]

  if use_mm:
    vp = "vision_encoder/DeepSeek41VisionProjector_0"
    ve = "vision_encoder/DeepSeek41VisionEncoder_0"
    mm_top = [
        (f"{vp}/image_start", "image_start", "identity"),
        (f"{vp}/image_end", "image_end", "identity"),
        (f"{vp}/image_newline", "image_newline", "identity"),
        (f"{vp}/w1/kernel", "aligner.w1.weight", "transpose_2d"),
        (f"{vp}/w1/bias", "aligner.w1.bias", "identity"),
        (f"{vp}/w2/kernel", "aligner.w2.weight", "transpose_2d"),
        (f"{vp}/w2/bias", "aligner.w2.bias", "identity"),
        (f"{ve}/patch_embed/kernel", "vision.patch_embed.proj.weight", "transpose_2d"),
        (f"{ve}/patch_embed/bias", "vision.patch_embed.proj.bias", "identity"),
        (f"{ve}/norm/scale", "vision.norm.weight", "identity"),
    ]
    for mt_p, hf_k, tfm in mm_top:
      rules.append(LeafAuditRule(mt_p, "multimodal", "single", (hf_k,), tfm, weight_dtype_str, "BF16", True))
    for b_idx in range(int(config.num_hidden_layers_for_vit)):
      bp = f"{ve}/blocks_{b_idx}"
      hb = f"vision.blocks.{b_idx}"
      for sub_mt, sub_hf, tfm in (
          ("norm1/scale", "norm1.weight", "identity"),
          ("norm2/scale", "norm2.weight", "identity"),
          ("attn/wqkv/kernel", "attn.wqkv.weight", "transpose_2d"),
          ("attn/wqkv/bias", "attn.wqkv.bias", "identity"),
          ("attn/wo/kernel", "attn.wo.weight", "transpose_2d"),
          ("attn/wo/bias", "attn.wo.bias", "identity"),
          ("mlp/w1/kernel", "mlp.w1.weight", "transpose_2d"),
          ("mlp/w2/kernel", "mlp.w2.weight", "transpose_2d"),
      ):
        rules.append(
            LeafAuditRule(
                f"{bp}/{sub_mt}", "multimodal", "single", (f"{hb}.{sub_hf}",), tfm, weight_dtype_str, "BF16", True
            )
        )

  specs = deepseek41.build_layer_specs(config)
  num_experts = int(config.num_experts)

  if not config.scan_layers:
    for layer_idx, spec in enumerate(specs):
      lp = f"decoder/layers_{layer_idx}"
      for sub_mt, hf_k, cat, tfm, dt, hdr_dt, no_scale in _layer_single_specs(layer_idx, spec, use_mm, weight_dtype_str):
        rules.append(
            LeafAuditRule(
                f"{lp}/{sub_mt}",
                cat,
                "single",
                (hf_k,),
                tfm,
                dt,
                hdr_dt,
                no_scale,
                (layer_idx,) if tfm == "engram_pad" else (),
            )
        )
      for mt_w, hf_w in (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2")):
        exp_keys = tuple(f"layers.{layer_idx}.ffn.experts.{e}.{hf_w}.weight" for e in range(num_experts))
        rules.append(
            LeafAuditRule(
                f"{lp}/mlp/MoeBlock_0/{mt_w}",
                "text",
                "expert_stack",
                exp_keys,
                "transpose_2d",
                weight_dtype_str,
                None,
                False,
            )
        )
  else:
    segments = deepseek41.build_segments(config)
    for seg_idx, segment in enumerate(segments):
      num_slots = len(segment.groups[0])
      for slot_idx in range(num_slots):
        layer_ids = tuple(group[slot_idx] for group in segment.groups)
        first_spec = specs[layer_ids[0]]
        first_items = _layer_single_specs(layer_ids[0], first_spec, use_mm, weight_dtype_str)
        per_rep_items = [
            {item[0]: item for item in _layer_single_specs(lid, specs[lid], use_mm, weight_dtype_str)}
            for lid in layer_ids
        ]
        sp = f"decoder/segments_{seg_idx}/layers_{slot_idx}"
        for sub_mt, _, cat, tfm, dt, hdr_dt, no_scale in first_items:
          hf_keys = tuple(per_rep_items[r][sub_mt][1] for r in range(len(layer_ids)))
          rules.append(
              LeafAuditRule(
                  f"{sp}/{sub_mt}",
                  cat,
                  "scanned_single",
                  hf_keys,
                  tfm,
                  dt,
                  hdr_dt,
                  no_scale,
                  layer_ids if tfm == "engram_pad" else (),
              )
          )
        for mt_w, hf_w in (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2")):
          exp_grid = tuple(
              tuple(f"layers.{lid}.ffn.experts.{e}.{hf_w}.weight" for e in range(num_experts)) for lid in layer_ids
          )
          rules.append(
              LeafAuditRule(
                  f"{sp}/mlp/MoeBlock_0/{mt_w}",
                  "text",
                  "scanned_expert_stack",
                  exp_grid,
                  "transpose_2d",
                  weight_dtype_str,
                  None,
                  False,
              )
          )
  return rules


def _transform_single_hf_tensor(
    arr: np.ndarray,
    transform: str,
    config: Any,
    padded_engram_rows: int | None = None,
) -> np.ndarray:
  """Applies the independent HF-to-MaxText layout transformation to a single 2D/1D HF tensor."""
  if transform == "identity":
    return arr
  if transform == "transpose_2d":
    return np.ascontiguousarray(arr.T)
  if transform == "wq_b_reshape":
    return np.ascontiguousarray(
        arr.T.reshape(int(config.q_lora_rank), int(config.base_num_query_heads), int(config.head_dim))
    )
  if transform == "indexer_wq_b_reshape":
    return np.ascontiguousarray(
        arr.T.reshape(int(config.q_lora_rank), int(config.indexer_n_heads), int(config.indexer_head_dim))
    )
  if transform == "wo_a_reshape":
    reshaped = arr.reshape(int(config.o_groups), int(config.o_lora_rank), -1)
    return np.ascontiguousarray(np.transpose(reshaped, (0, 2, 1)))
  if transform == "engram_pad":
    if padded_engram_rows is None or padded_engram_rows < arr.shape[0]:
      raise ValueError(f"Invalid padded_engram_rows={padded_engram_rows} for raw shape {arr.shape}")
    if padded_engram_rows == arr.shape[0]:
      return arr
    out = np.zeros((padded_engram_rows, arr.shape[1]), dtype=arr.dtype)
    out[: arr.shape[0], :] = arr
    return out
  raise ValueError(f"Unknown transform: {transform!r}")


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> tuple[bool, int]:
  """Compares two NumPy arrays of identical shape and dtype at the bit pattern level."""
  if a.shape != b.shape or a.dtype != b.dtype:
    return False, int(max(a.size, b.size))
  if a.dtype == ml_dtypes.bfloat16 or a.dtype.name == "bfloat16":
    va = np.ascontiguousarray(a).view(np.uint16)
    vb = np.ascontiguousarray(b).view(np.uint16)
  elif a.dtype == np.float32:
    va = np.ascontiguousarray(a).view(np.uint32)
    vb = np.ascontiguousarray(b).view(np.uint32)
  else:
    va = np.ascontiguousarray(a)
    vb = np.ascontiguousarray(b)
  diff = va != vb
  mismatch_count = int(np.count_nonzero(diff))
  return mismatch_count == 0, mismatch_count


def flatten_state_arrays(tree: Mapping[str, Any], prefix: str = "") -> dict[str, jax.Array]:
  """Flattens a nested state array dictionary into dot-separated paths."""
  out: dict[str, jax.Array] = {}
  for k, v in tree.items():
    path = f"{prefix}/{k}" if prefix else str(k)
    if isinstance(v, Mapping):
      out.update(flatten_state_arrays(v, path))
    else:
      out[path] = v
  return out


def audit_hf_index_accounting(
    index_json_path: str | os.PathLike[str],
    config: Any,
) -> dict[str, Any]:
  """Verifies 100% HF safetensors index key accounting against `build_independent_audit_rules(config)`."""
  with epath.Path(str(index_json_path)).open("r", encoding="utf-8") as f:
    data = json.load(f)
  all_hf_keys = set(data["weight_map"].keys())
  rules = build_independent_audit_rules(config)

  consumed_hf_keys: set[str] = set()
  duplicate_hf_keys: set[str] = set()
  missing_hf_keys: set[str] = set()
  forbidden_scale_violations: list[str] = []

  def _record_key(hf_k: str, forbid_scale: bool) -> None:
    if hf_k in consumed_hf_keys:
      duplicate_hf_keys.add(hf_k)
    consumed_hf_keys.add(hf_k)
    if hf_k not in all_hf_keys:
      missing_hf_keys.add(hf_k)
    for scale_k in _candidate_scale_keys(hf_k):
      if scale_k in all_hf_keys:
        if forbid_scale:
          forbidden_scale_violations.append(scale_k)
        if scale_k in consumed_hf_keys:
          duplicate_hf_keys.add(scale_k)
        consumed_hf_keys.add(scale_k)

  for rule in rules:
    if rule.kind in ("single", "expert_stack", "scanned_single"):
      for hf_k in rule.hf_keys:
        _record_key(str(hf_k), rule.forbid_scale)
    elif rule.kind == "scanned_expert_stack":
      for row in rule.hf_keys:
        for hf_k in row:
          _record_key(str(hf_k), rule.forbid_scale)

  unloaded_hf_keys = all_hf_keys - consumed_hf_keys
  expected_unloaded = {k for k in all_hf_keys if k.startswith("mtp.")}
  unexpected_unloaded = sorted(unloaded_hf_keys - expected_unloaded)
  passed = (
      not missing_hf_keys
      and not duplicate_hf_keys
      and not forbidden_scale_violations
      and unloaded_hf_keys == expected_unloaded
  )
  return {
      "passed": passed,
      "total_hf_keys": len(all_hf_keys),
      "consumed_hf_keys": len(consumed_hf_keys),
      "unloaded_mtp_keys": len(unloaded_hf_keys & expected_unloaded),
      "missing_hf_keys": sorted(missing_hf_keys),
      "duplicate_hf_keys": sorted(duplicate_hf_keys),
      "unexpected_unloaded_hf_keys": unexpected_unloaded,
      "forbidden_scale_violations": sorted(forbidden_scale_violations),
  }


def audit_loaded_model_against_hf(
    model: nnx.Module,
    config: Any,
    hf_checkpoint_dir: str | os.PathLike[str],
    mesh: jax.sharding.Mesh,
    *,
    verify_all_text_leaves: bool = True,
    export_orbax_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
  """Audits every loaded MaxText model leaf against the HF safetensors checkpoint on disk."""
  reader = _SafetensorsShardReader(hf_checkpoint_dir)
  try:
    return _audit_loaded_model_with_reader(
        model=model,
        config=config,
        reader=reader,
        mesh=mesh,
        verify_all_text_leaves=verify_all_text_leaves,
        export_orbax_dir=export_orbax_dir,
    )
  finally:
    reader.close()


def _audit_loaded_model_with_reader(
    *,
    model: nnx.Module,
    config: Any,
    reader: _SafetensorsShardReader,
    mesh: jax.sharding.Mesh,
    verify_all_text_leaves: bool,
    export_orbax_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
  """Runs HF-to-MaxText key, dtype, sharding, and bitwise shard verification using `reader`."""
  all_hf_keys = reader.all_keys()
  rules = build_independent_audit_rules(config)
  rule_by_path = {r.maxtext_path: r for r in rules}

  _, params_state, bias_state, _ = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
  merged_state = nnx.merge_state(params_state, bias_state)
  pure_weights = nnx.to_pure_dict(merged_state)
  model_leaves = flatten_state_arrays(pure_weights)

  _, abs_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
  _, abs_params_state, abs_bias_state, _ = nnx.split(abs_model, nnx.Param, moe.MoEBiasVar, ...)
  abs_merged_state = nnx.merge_state(abs_params_state, abs_bias_state)
  expected_leaf_avals = flatten_state_arrays(nnx.to_pure_dict(abs_merged_state))

  missing_mt_leaves = sorted(set(rule_by_path.keys()) - set(model_leaves.keys()))
  unexpected_mt_leaves = sorted(set(model_leaves.keys()) - set(rule_by_path.keys()))

  consumed_hf_keys: set[str] = set()
  duplicate_hf_keys: set[str] = set()
  missing_hf_keys: set[str] = set()
  header_dtype_mismatches: list[str] = []
  forbidden_scale_violations: list[str] = []

  def _account_hf_key(hf_k: str, rule: LeafAuditRule) -> None:
    if hf_k in consumed_hf_keys:
      duplicate_hf_keys.add(hf_k)
    consumed_hf_keys.add(hf_k)
    if hf_k not in all_hf_keys:
      missing_hf_keys.add(hf_k)
      return
    hdr = reader.get_header_entry(hf_k)
    if rule.expected_hf_header_dtype is not None and hdr["dtype"] != rule.expected_hf_header_dtype:
      header_dtype_mismatches.append(
          f"{hf_k}: safetensors dtype={hdr['dtype']!r}, expected={rule.expected_hf_header_dtype!r}"
      )
    for scale_k in _candidate_scale_keys(hf_k):
      if scale_k in all_hf_keys:
        if rule.forbid_scale:
          forbidden_scale_violations.append(scale_k)
        if scale_k in consumed_hf_keys:
          duplicate_hf_keys.add(scale_k)
        consumed_hf_keys.add(scale_k)

  for rule in rules:
    if rule.kind in ("single", "expert_stack", "scanned_single"):
      for hf_k in rule.hf_keys:
        _account_hf_key(str(hf_k), rule)
    elif rule.kind == "scanned_expert_stack":
      for row in rule.hf_keys:
        for hf_k in row:
          _account_hf_key(str(hf_k), rule)

  unloaded_hf_keys = all_hf_keys - consumed_hf_keys
  expected_unloaded = {k for k in all_hf_keys if k.startswith("mtp.")}
  unexpected_unloaded = sorted(unloaded_hf_keys - expected_unloaded)
  if not config.use_multimodal:
    unexpected_unloaded = [
        k
        for k in unexpected_unloaded
        if not (
            k.startswith("vision.")
            or k.startswith("aligner.")
            or k in ("image_start", "image_end", "image_newline")
            or k.endswith(".ffn.gate.bias_vl")
        )
    ]

  local_checked_leaves = 0
  local_checked_shards = 0
  local_bit_mismatches = 0
  leaf_failures: list[str] = []
  scan_axis = int(config.param_scan_axis)

  sharding_mismatches: list[str] = []
  leaf_tasks: list[tuple[str, LeafAuditRule, tuple[int, ...], list[Any]]] = []
  for mt_path, rule in rules_to_check(rules, verify_all_text_leaves):
    if mt_path not in model_leaves or missing_hf_keys:
      continue
    mt_arr = model_leaves[mt_path]
    if mt_path not in expected_leaf_avals or not isinstance(
        expected_leaf_avals[mt_path].sharding, jax.sharding.NamedSharding
    ):
      expected_sharding = expected_leaf_avals[mt_path].sharding if mt_path in expected_leaf_avals else None
      msg = f"{mt_path}: missing or invalid abstract NamedSharding in create_nnx_abstract_model: {expected_sharding!r}"
      sharding_mismatches.append(msg)
      leaf_failures.append(msg)
    else:
      expected_sharding = expected_leaf_avals[mt_path].sharding
      actual_sharding = getattr(mt_arr, "sharding", None)
      if not isinstance(actual_sharding, jax.sharding.Sharding) or not actual_sharding.is_equivalent_to(
          expected_sharding, mt_arr.ndim
      ):
        msg = f"{mt_path}: sharding={actual_sharding!r}, expected {expected_sharding!r}"
        sharding_mismatches.append(msg)
        leaf_failures.append(msg)

    actual_dtype_name = jnp.dtype(mt_arr.dtype).name
    if actual_dtype_name != rule.expected_mt_dtype:
      leaf_failures.append(f"{mt_path}: MaxText dtype={actual_dtype_name!r}, expected={rule.expected_mt_dtype!r}")
      continue

    rep0_shards = [s for s in mt_arr.addressable_shards if s.replica_id == 0]
    if not rep0_shards:
      rep0_shards = list(mt_arr.addressable_shards[:1])
    if not rep0_shards:
      continue

    local_checked_leaves += 1
    leaf_tasks.append((mt_path, rule, mt_arr.shape, rep0_shards))

  with concurrent.futures.ThreadPoolExecutor(max_workers=_AUDIT_IO_MAX_WORKERS) as pool:
    single_tasks = [t for t in leaf_tasks if t[1].kind == "single"]
    other_tasks = [t for t in leaf_tasks if t[1].kind != "single"]

    def _run_single_leaf_task(
        task: tuple[str, LeafAuditRule, tuple[int, ...], list[Any]],
    ) -> tuple[int, int, list[str]]:
      mt_path_t, rule_t, shape_t, shards_t = task
      return _audit_leaf_rep0_shards(
          mt_path=mt_path_t,
          rule=rule_t,
          global_shape=shape_t,
          rep0_shards=shards_t,
          scan_axis=scan_axis,
          reader=reader,
          config=config,
          pool=None,
      )

    engram_single_tasks = [t for t in single_tasks if t[1].transform == "engram_pad"]
    regular_single_tasks = [t for t in single_tasks if t[1].transform != "engram_pad"]
    single_results = [_run_single_leaf_task(t) for t in engram_single_tasks]
    single_workers = min(_AUDIT_SINGLE_MAX_WORKERS, _AUDIT_IO_MAX_WORKERS, max(1, len(regular_single_tasks)))
    if len(regular_single_tasks) > 1 and single_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(max_workers=single_workers) as single_pool:
        single_results.extend(single_pool.map(_run_single_leaf_task, regular_single_tasks))
    else:
      single_results.extend(_run_single_leaf_task(t) for t in regular_single_tasks)

    for checked_shards, bit_mismatches, shard_failures in single_results:
      local_checked_shards += checked_shards
      local_bit_mismatches += bit_mismatches
      leaf_failures.extend(shard_failures)

    expert_stack_tasks = [t for t in other_tasks if t[1].kind == "expert_stack"]
    scanned_tasks = [t for t in other_tasks if t[1].kind != "expert_stack"]

    if len(expert_stack_tasks) > 1 and _AUDIT_IO_MAX_WORKERS > 1:
      rule_workers = min(8, len(expert_stack_tasks))

      def _run_expert_stack_task(
          task: tuple[str, LeafAuditRule, tuple[int, ...], list[Any]],
      ) -> tuple[int, int, list[str]]:
        mt_path_t, rule_t, shape_t, shards_t = task
        return _audit_leaf_rep0_shards(
            mt_path=mt_path_t,
            rule=rule_t,
            global_shape=shape_t,
            rep0_shards=shards_t,
            scan_axis=scan_axis,
            reader=reader,
            config=config,
            pool=pool,
        )

      with concurrent.futures.ThreadPoolExecutor(max_workers=rule_workers) as rule_pool:
        es_results = list(rule_pool.map(_run_expert_stack_task, expert_stack_tasks))
    else:
      es_results = [
          _audit_leaf_rep0_shards(
              mt_path=mt_path_t,
              rule=rule_t,
              global_shape=shape_t,
              rep0_shards=shards_t,
              scan_axis=scan_axis,
              reader=reader,
              config=config,
              pool=pool,
          )
          for mt_path_t, rule_t, shape_t, shards_t in expert_stack_tasks
      ]

    for checked_shards, bit_mismatches, shard_failures in es_results:
      local_checked_shards += checked_shards
      local_bit_mismatches += bit_mismatches
      leaf_failures.extend(shard_failures)

    for mt_path_t, rule_t, shape_t, shards_t in scanned_tasks:
      checked_shards, bit_mismatches, shard_failures = _audit_leaf_rep0_shards(
          mt_path=mt_path_t,
          rule=rule_t,
          global_shape=shape_t,
          rep0_shards=shards_t,
          scan_axis=scan_axis,
          reader=reader,
          config=config,
          pool=pool,
      )
      local_checked_shards += checked_shards
      local_bit_mismatches += bit_mismatches
      leaf_failures.extend(shard_failures)

  gathered_stats = multihost_utils.process_allgather(
      np.array(
          [local_checked_leaves, local_checked_shards, local_bit_mismatches, len(leaf_failures)],
          dtype=np.int64,
      )
  )
  stats_arr = np.asarray(gathered_stats).reshape(-1, 4)
  total_checked_shards = int(np.sum(stats_arr[:, 1]))
  total_bit_mismatches = int(np.sum(stats_arr[:, 2]))
  total_leaf_failures = int(np.sum(stats_arr[:, 3]))

  passed = (
      not missing_mt_leaves
      and not unexpected_mt_leaves
      and not missing_hf_keys
      and not duplicate_hf_keys
      and not unexpected_unloaded
      and not header_dtype_mismatches
      and not forbidden_scale_violations
      and not sharding_mismatches
      and total_checked_shards > 0
      and total_bit_mismatches == 0
      and total_leaf_failures == 0
  )

  result: dict[str, Any] = {
      "passed": passed,
      "total_hf_keys": len(all_hf_keys),
      "consumed_hf_keys": len(consumed_hf_keys),
      "unloaded_mtp_keys": len(unloaded_hf_keys & expected_unloaded),
      "missing_hf_keys": sorted(missing_hf_keys),
      "duplicate_hf_keys": sorted(duplicate_hf_keys),
      "unexpected_unloaded_hf_keys": unexpected_unloaded,
      "header_dtype_mismatches": header_dtype_mismatches,
      "forbidden_scale_violations": sorted(forbidden_scale_violations),
      "sharding_mismatches": sharding_mismatches,
      "missing_maxtext_leaves": missing_mt_leaves,
      "unexpected_maxtext_leaves": unexpected_mt_leaves,
      "local_checked_leaves": local_checked_leaves,
      "total_checked_shards": total_checked_shards,
      "total_bit_mismatches": total_bit_mismatches,
      "total_leaf_failures": total_leaf_failures,
      "leaf_failures": leaf_failures[:20],
      "exported_orbax_items_path": None,
  }

  if export_orbax_dir is not None:
    if not passed:
      raise RuntimeError(
          f"Refusing to export Orbax checkpoint to {export_orbax_dir}: HF-to-MaxText audit failed: {result}"
      )
    mgr_root_str = _normalize_orbax_root(export_orbax_dir)
    ckpt_mgr = checkpointing.create_orbax_checkpoint_manager(
        mgr_root_str,
        True,
        False,
        1,
        use_ocdbt=bool(config.checkpoint_storage_use_ocdbt),
        use_zarr3=bool(config.checkpoint_storage_use_zarr3),
    )
    st = train_state.TrainState(
        step=0,
        apply_fn=None,
        params={"params": pure_weights},
        tx=None,
        opt_state={},
    )
    checkpointing.save_checkpoint(ckpt_mgr, 0, st)
    checkpointing.wait_until_finished(ckpt_mgr)
    result["exported_orbax_items_path"] = f"{mgr_root_str.rstrip('/')}/0/items"

  return result


def rules_to_check(
    rules: list[LeafAuditRule],
    verify_all_text_leaves: bool,
) -> list[tuple[str, LeafAuditRule]]:
  out: list[tuple[str, LeafAuditRule]] = []
  for r in rules:
    if r.category == "multimodal" or verify_all_text_leaves:
      out.append((r.maxtext_path, r))
  return out


def _load_engram_padded_shard_2d(
    *,
    reader: _SafetensorsShardReader,
    hf_key: str,
    target_dtype_str: str,
    padded_rows: int,
    row_slice: slice,
    col_slice: slice,
) -> np.ndarray:
  """Loads a 2D shard `[row_slice, col_slice]` of an `engram_pad` tensor using row-sliced IO."""
  hdr = reader.get_header_entry(hf_key)
  raw_shape = tuple(int(d) for d in hdr["shape"])
  if len(raw_shape) != 2:
    raise ValueError(f"engram_pad expects 2D HF tensor for {hf_key!r}, got shape {raw_shape}")
  raw_rows, raw_cols = raw_shape
  if padded_rows < raw_rows:
    raise ValueError(f"Invalid padded_rows={padded_rows} < raw_rows={raw_rows} for {hf_key!r}")
  r0, r1, r_step = row_slice.indices(padded_rows)
  if r_step != 1:
    full = _transform_single_hf_tensor(
        reader.load_dequantized_tensor(hf_key, target_dtype_str, use_cache=False),
        "engram_pad",
        None,
        padded_engram_rows=padded_rows,
    )
    return full[row_slice, col_slice]
  f0 = min(max(0, r0), raw_rows)
  f1 = min(max(f0, r1), raw_rows)
  part = reader.load_dequantized_row_slice(hf_key, target_dtype_str, f0, f1)
  part_cols = part[:, col_slice]
  pad_needed = (r1 - r0) - (f1 - f0)
  if pad_needed <= 0:
    return np.ascontiguousarray(part_cols)
  c0, c1, c_step = col_slice.indices(raw_cols)
  sliced_cols = len(range(c0, c1, c_step))
  zeros = np.zeros((pad_needed, sliced_cols), dtype=part_cols.dtype)
  if part_cols.shape[0] == 0:
    return zeros
  return np.concatenate([part_cols, zeros], axis=0)


def _target_numpy_dtype(target_dtype_str: str) -> np.dtype:
  if target_dtype_str == "bfloat16":
    return np.dtype(ml_dtypes.bfloat16)
  if target_dtype_str == "float32":
    return np.dtype(np.float32)
  raise ValueError(f"Unsupported target_dtype_str: {target_dtype_str!r}")


def _finalize_leaf_shard_results(
    *,
    mt_path: str,
    rep0_shards: list[Any],
    shard_shapes: list[tuple[int, ...]],
    shard_dtypes: list[np.dtype],
    mismatches_by_shard: list[int],
) -> tuple[int, int, list[str]]:
  """Formats per-shard mismatch counts into the canonical `(checked_shards, bit_mismatches, failures)` tuple."""
  local_bit_mismatches = 0
  shard_failures: list[str] = []
  for k, shard in enumerate(rep0_shards):
    n_bad = mismatches_by_shard[k]
    if n_bad > 0:
      local_bit_mismatches += n_bad
      shard_failures.append(
          f"{mt_path} shard {shard.index}: {n_bad} bitwise element mismatches "
          f"(shape={shard_shapes[k]}, dtype={shard_dtypes[k]})"
      )
  return len(rep0_shards), local_bit_mismatches, shard_failures


def _audit_leaf_rep0_shards(
    *,
    mt_path: str,
    rule: LeafAuditRule,
    global_shape: tuple[int, ...],
    rep0_shards: list[Any],
    scan_axis: int,
    reader: _SafetensorsShardReader,
    config: Any,
    pool: concurrent.futures.Executor | None = None,
) -> tuple[int, int, list[str]]:
  """Audits all local replica-0 shards of a MaxText leaf without caching or stacking full unsharded planes."""
  num_shards = len(rep0_shards)
  mismatches_by_shard = [0] * num_shards
  shard_shapes: list[tuple[int, ...]] = []
  shard_dtypes: list[np.dtype] = []

  key_to_u: dict[tuple[tuple[int | None, int | None, int | None], ...], int] = {}
  unique_indices: list[tuple[slice, ...]] = []
  unique_groups: list[list[tuple[int, np.ndarray]]] = []

  for k, shard in enumerate(rep0_shards):
    shard_np = np.asarray(shard.data)
    shard_shapes.append(shard_np.shape)
    shard_dtypes.append(shard_np.dtype)
    idx = tuple(shard.index)
    idx_key = tuple((s.start, s.stop, s.step) for s in idx)
    if idx_key not in key_to_u:
      key_to_u[idx_key] = len(unique_indices)
      unique_indices.append(idx)
      unique_groups.append([])
    unique_groups[key_to_u[idx_key]].append((k, shard_np))

  num_unique = len(unique_indices)
  expected_dtype = _target_numpy_dtype(rule.expected_mt_dtype)
  norm_indices = [idx + (slice(None),) * max(0, len(global_shape) - len(idx)) for idx in unique_indices]

  if rule.kind == "scanned_expert_stack" and len(global_shape) >= 3:
    rep_axis = scan_axis if len(global_shape) > scan_axis else 0
    exp_axis = 1 if rep_axis == 0 else 0
    if rep_axis in (0, 1) and all(
        n_idx[rep_axis].step in (None, 1) and n_idx[exp_axis].step in (None, 1) for n_idx in norm_indices
    ):
      rep_map: list[dict[int, int]] = []
      exp_map: list[dict[int, int]] = []
      tail_idx: list[tuple[slice, ...]] = []
      valid_groups: list[list[tuple[int, np.ndarray]]] = []

      for u, n_idx in enumerate(norm_indices):
        expected_shape = tuple(len(range(*n_idx[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
        rep_map.append({r: r_loc for r_loc, r in enumerate(range(*n_idx[rep_axis].indices(global_shape[rep_axis])))})
        exp_map.append({e: e_loc for e_loc, e in enumerate(range(*n_idx[exp_axis].indices(global_shape[exp_axis])))})
        tail_idx.append(tuple(n_idx[2:]))
        v_group: list[tuple[int, np.ndarray]] = []
        for k, shard_np in unique_groups[u]:
          if shard_np.shape != expected_shape or shard_np.dtype != expected_dtype:
            mismatches_by_shard[k] = int(max(shard_np.size, math.prod(expected_shape)))
          else:
            v_group.append((k, shard_np))
        valid_groups.append(v_group)

      needed_reps = sorted(set().union(*(rep_map[u].keys() for u in range(num_unique) if valid_groups[u])))
      for r in needed_reps:
        active_u = [u for u in range(num_unique) if valid_groups[u] and r in rep_map[u]]
        needed_exps = sorted(set().union(*(exp_map[u].keys() for u in active_u)))
        row_keys = rule.hf_keys[r]

        def _load_rep_expert(exp_id: int, keys_row: tuple[Any, ...] = row_keys) -> tuple[int, np.ndarray]:
          plane_2d = _transform_single_hf_tensor(
              reader.load_dequantized_tensor(str(keys_row[exp_id]), rule.expected_mt_dtype, use_cache=False),
              rule.transform,
              config,
          )
          return exp_id, plane_2d

        for start in range(0, len(needed_exps), _EXPERT_AUDIT_CHUNK_SIZE):
          chunk_exps = needed_exps[start : start + _EXPERT_AUDIT_CHUNK_SIZE]
          mapped_rep = pool.map(_load_rep_expert, chunk_exps) if pool is not None else map(_load_rep_expert, chunk_exps)
          for e, plane_2d in mapped_rep:
            for u in active_u:
              if e not in exp_map[u]:
                continue
              expected_sub = plane_2d[tail_idx[u]]
              r_loc = rep_map[u][r]
              e_loc = exp_map[u][e]
              for k, shard_np in valid_groups[u]:
                actual_sub = shard_np[r_loc, e_loc] if rep_axis == 0 else shard_np[e_loc, r_loc]
                ok, n_bad = _bitwise_equal(actual_sub, expected_sub)
                if not ok:
                  mismatches_by_shard[k] += n_bad

      return _finalize_leaf_shard_results(
          mt_path=mt_path,
          rep0_shards=rep0_shards,
          shard_shapes=shard_shapes,
          shard_dtypes=shard_dtypes,
          mismatches_by_shard=mismatches_by_shard,
      )

  if rule.kind == "expert_stack" and len(global_shape) >= 2 and all(n_idx[0].step in (None, 1) for n_idx in norm_indices):
    exp_map_es: list[dict[int, int]] = []
    tail_idx_es: list[tuple[slice, ...]] = []
    valid_groups_es: list[list[tuple[int, np.ndarray]]] = []

    for u, n_idx in enumerate(norm_indices):
      expected_shape = tuple(len(range(*n_idx[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
      exp_map_es.append({e: e_loc for e_loc, e in enumerate(range(*n_idx[0].indices(global_shape[0])))})
      tail_idx_es.append(tuple(n_idx[1:]))
      v_group_es: list[tuple[int, np.ndarray]] = []
      for k, shard_np in unique_groups[u]:
        if shard_np.shape != expected_shape or shard_np.dtype != expected_dtype:
          mismatches_by_shard[k] = int(max(shard_np.size, math.prod(expected_shape)))
        else:
          v_group_es.append((k, shard_np))
      valid_groups_es.append(v_group_es)

    active_u_es = [u for u in range(num_unique) if valid_groups_es[u]]
    needed_exps = sorted(set().union(*(exp_map_es[u].keys() for u in active_u_es)))

    def _load_expert(exp_id: int) -> tuple[int, np.ndarray]:
      plane_2d = _transform_single_hf_tensor(
          reader.load_dequantized_tensor(str(rule.hf_keys[exp_id]), rule.expected_mt_dtype, use_cache=False),
          rule.transform,
          config,
      )
      return exp_id, plane_2d

    for start in range(0, len(needed_exps), _EXPERT_AUDIT_CHUNK_SIZE):
      chunk_exps = needed_exps[start : start + _EXPERT_AUDIT_CHUNK_SIZE]
      mapped_es = pool.map(_load_expert, chunk_exps) if pool is not None else map(_load_expert, chunk_exps)
      for e, plane_2d in mapped_es:
        for u in active_u_es:
          if e not in exp_map_es[u]:
            continue
          expected_sub = plane_2d[tail_idx_es[u]]
          e_loc = exp_map_es[u][e]
          for k, shard_np in valid_groups_es[u]:
            actual_sub = shard_np[e_loc]
            ok, n_bad = _bitwise_equal(actual_sub, expected_sub)
            if not ok:
              mismatches_by_shard[k] += n_bad

    return _finalize_leaf_shard_results(
        mt_path=mt_path,
        rep0_shards=rep0_shards,
        shard_shapes=shard_shapes,
        shard_dtypes=shard_dtypes,
        mismatches_by_shard=mismatches_by_shard,
    )

  if rule.kind == "single" and num_unique > 1 and rule.transform != "engram_pad":
    valid_groups_s: list[list[tuple[int, np.ndarray]]] = []
    for u, n_idx in enumerate(norm_indices):
      expected_shape = tuple(len(range(*n_idx[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
      v_group_s: list[tuple[int, np.ndarray]] = []
      for k, shard_np in unique_groups[u]:
        if shard_np.shape != expected_shape or shard_np.dtype != expected_dtype:
          mismatches_by_shard[k] = int(max(shard_np.size, math.prod(expected_shape)))
        else:
          v_group_s.append((k, shard_np))
      valid_groups_s.append(v_group_s)

    active_u_s = [u for u in range(num_unique) if valid_groups_s[u]]
    if active_u_s:
      hf_k = str(rule.hf_keys[0])
      full = _transform_single_hf_tensor(
          reader.load_dequantized_tensor(hf_k, rule.expected_mt_dtype, use_cache=False),
          rule.transform,
          config,
          padded_engram_rows=None,
      )
      for u in active_u_s:
        expected_slice = np.ascontiguousarray(full[norm_indices[u]])
        for k, shard_np in valid_groups_s[u]:
          ok, n_bad = _bitwise_equal(shard_np, expected_slice)
          if not ok:
            mismatches_by_shard[k] = n_bad
        del expected_slice
      del full
    return _finalize_leaf_shard_results(
        mt_path=mt_path,
        rep0_shards=rep0_shards,
        shard_shapes=shard_shapes,
        shard_dtypes=shard_dtypes,
        mismatches_by_shard=mismatches_by_shard,
    )

  for u, idx in enumerate(unique_indices):
    expected_slice = _build_expected_shard_slice(
        rule=rule,
        global_shape=global_shape,
        shard_index=idx,
        scan_axis=scan_axis,
        reader=reader,
        config=config,
        pool=pool,
    )
    for k, shard_np in unique_groups[u]:
      ok, n_bad = _bitwise_equal(shard_np, expected_slice)
      if not ok:
        mismatches_by_shard[k] = n_bad
    del expected_slice

  return _finalize_leaf_shard_results(
      mt_path=mt_path,
      rep0_shards=rep0_shards,
      shard_shapes=shard_shapes,
      shard_dtypes=shard_dtypes,
      mismatches_by_shard=mismatches_by_shard,
  )


def _build_expected_shard_slice(
    *,
    rule: LeafAuditRule,
    global_shape: tuple[int, ...],
    shard_index: tuple[slice, ...],
    scan_axis: int,
    reader: _SafetensorsShardReader,
    config: Any,
    pool: concurrent.futures.Executor | None = None,
) -> np.ndarray:
  """Constructs the expected NumPy slice for a single replica-0 shard from HF safetensors."""
  norm_index = tuple(shard_index) + (slice(None),) * max(0, len(global_shape) - len(shard_index))
  expected_dtype = _target_numpy_dtype(rule.expected_mt_dtype)

  if rule.kind == "single":
    hf_k = str(rule.hf_keys[0])
    if rule.transform == "engram_pad" and len(global_shape) == 2:
      row_slice = norm_index[0]
      col_slice = norm_index[1]
      if row_slice.step in (None, 1):
        return _load_engram_padded_shard_2d(
            reader=reader,
            hf_key=hf_k,
            target_dtype_str=rule.expected_mt_dtype,
            padded_rows=global_shape[0],
            row_slice=row_slice,
            col_slice=col_slice,
        )
    if rule.transform == "identity" and len(global_shape) == 2:
      row_slice = norm_index[0]
      col_slice = norm_index[1]
      if row_slice.step in (None, 1):
        r0, r1, _ = row_slice.indices(global_shape[0])
        part = reader.load_dequantized_row_slice(hf_k, rule.expected_mt_dtype, r0, r1)
        return np.ascontiguousarray(part[:, col_slice])
    padded_rows = global_shape[0] if rule.transform == "engram_pad" else None
    full = _transform_single_hf_tensor(
        reader.load_dequantized_tensor(hf_k, rule.expected_mt_dtype, use_cache=False),
        rule.transform,
        config,
        padded_engram_rows=padded_rows,
    )
    return full[norm_index]

  if rule.kind == "expert_stack":
    exp_slice = norm_index[0] if norm_index else slice(None)
    exp_indices = list(range(*exp_slice.indices(global_shape[0])))
    if not exp_indices:
      empty_shape = tuple(len(range(*norm_index[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
      return np.zeros(empty_shape, dtype=expected_dtype)
    tail_idx = tuple(norm_index[1:])

    def _load_sliced_expert(exp_id: int) -> np.ndarray:
      plane_2d = _transform_single_hf_tensor(
          reader.load_dequantized_tensor(str(rule.hf_keys[exp_id]), rule.expected_mt_dtype, use_cache=False),
          rule.transform,
          config,
      )
      return np.ascontiguousarray(plane_2d[tail_idx])

    if pool is not None and len(exp_indices) > 1:
      planes: list[np.ndarray] = []
      for start in range(0, len(exp_indices), _EXPERT_AUDIT_CHUNK_SIZE):
        chunk_ids = exp_indices[start : start + _EXPERT_AUDIT_CHUNK_SIZE]
        planes.extend(pool.map(_load_sliced_expert, chunk_ids))
    else:
      planes = [_load_sliced_expert(e) for e in exp_indices]
    return np.stack(planes, axis=0)

  if rule.kind == "scanned_single":
    if rule.transform == "engram_pad" and len(global_shape) == 3:
      layer_axis = scan_axis if scan_axis < 3 else 0
      row_axis = 1 if layer_axis == 0 else 0
      col_axis = 3 - layer_axis - row_axis
      layer_slice = norm_index[layer_axis]
      row_slice = norm_index[row_axis]
      col_slice = norm_index[col_axis]
      if layer_slice.step in (None, 1) and row_slice.step in (None, 1):
        layer_indices = list(range(*layer_slice.indices(global_shape[layer_axis])))
        if not layer_indices:
          empty_shape = tuple(len(range(*norm_index[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
          return np.zeros(empty_shape, dtype=expected_dtype)

        def _load_engram_layer(l_idx: int) -> np.ndarray:
          return _load_engram_padded_shard_2d(
              reader=reader,
              hf_key=str(rule.hf_keys[l_idx]),
              target_dtype_str=rule.expected_mt_dtype,
              padded_rows=global_shape[row_axis],
              row_slice=row_slice,
              col_slice=col_slice,
          )

        if pool is not None and len(layer_indices) > 1:
          planes = list(pool.map(_load_engram_layer, layer_indices))
        else:
          planes = [_load_engram_layer(l_idx) for l_idx in layer_indices]
        return np.stack(planes, axis=layer_axis)

    is_moe_bias = rule.maxtext_path.endswith("/gate/bias") or rule.maxtext_path.endswith("/gate/bias_vl")
    eff_scan_axis = 0 if (is_moe_bias or len(global_shape) <= scan_axis) else scan_axis
    layer_slice = norm_index[eff_scan_axis]
    other_idx = tuple(s for ax, s in enumerate(norm_index) if ax != eff_scan_axis)
    layer_indices = list(range(*layer_slice.indices(len(rule.hf_keys))))
    if not layer_indices:
      empty_shape = tuple(len(range(*norm_index[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
      return np.zeros(empty_shape, dtype=expected_dtype)
    padded_rows = None
    if rule.transform == "engram_pad":
      row_axis = 1 if scan_axis == 0 else 0
      padded_rows = global_shape[row_axis]

    def _load_sliced_layer(l_idx: int) -> np.ndarray:
      plane = _transform_single_hf_tensor(
          reader.load_dequantized_tensor(str(rule.hf_keys[l_idx]), rule.expected_mt_dtype, use_cache=False),
          rule.transform,
          config,
          padded_engram_rows=padded_rows,
      )
      return np.ascontiguousarray(plane[other_idx])

    if pool is not None and len(layer_indices) > 1:
      planes = list(pool.map(_load_sliced_layer, layer_indices))
    else:
      planes = [_load_sliced_layer(l_idx) for l_idx in layer_indices]
    return np.stack(planes, axis=eff_scan_axis)

  if rule.kind == "scanned_expert_stack":
    rep_axis = scan_axis if len(global_shape) > scan_axis else 0
    exp_axis = 1 if rep_axis == 0 else 0
    rep_slice = norm_index[rep_axis]
    exp_slice = norm_index[exp_axis]
    rep_indices = list(range(*rep_slice.indices(global_shape[rep_axis])))
    exp_indices = list(range(*exp_slice.indices(global_shape[exp_axis])))
    if not rep_indices or not exp_indices:
      empty_shape = tuple(len(range(*norm_index[ax].indices(global_shape[ax]))) for ax in range(len(global_shape)))
      return np.zeros(empty_shape, dtype=expected_dtype)
    tail_idx = tuple(s for ax, s in enumerate(norm_index) if ax not in (rep_axis, exp_axis))
    rep_blocks: list[np.ndarray] = []
    for r in rep_indices:
      row_keys = rule.hf_keys[r]

      def _load_sliced_rep_expert(exp_id: int, keys_row: tuple[Any, ...] = row_keys) -> np.ndarray:
        plane_2d = _transform_single_hf_tensor(
            reader.load_dequantized_tensor(str(keys_row[exp_id]), rule.expected_mt_dtype, use_cache=False),
            rule.transform,
            config,
        )
        return np.ascontiguousarray(plane_2d[tail_idx])

      if pool is not None and len(exp_indices) > 1:
        exp_planes: list[np.ndarray] = []
        for start in range(0, len(exp_indices), _EXPERT_AUDIT_CHUNK_SIZE):
          chunk_ids = exp_indices[start : start + _EXPERT_AUDIT_CHUNK_SIZE]
          exp_planes.extend(pool.map(_load_sliced_rep_expert, chunk_ids))
      else:
        exp_planes = [_load_sliced_rep_expert(e) for e in exp_indices]
      rep_blocks.append(np.stack(exp_planes, axis=0))
    return np.stack(rep_blocks, axis=rep_axis)

  raise ValueError(f"Unsupported rule.kind: {rule.kind!r}")


def _cli_main(argv: list[str] | None = None) -> dict[str, Any]:
  """CLI entrypoint for streaming HF-to-MaxText weight audit and optional Orbax export."""
  parser = argparse.ArgumentParser(description="DeepSeek-V4.1 multimodal streaming weight audit.")
  parser.add_argument("--hf_checkpoint_dir", "--hf_ckpt_path", dest="hf_checkpoint_dir", type=str, default="")
  parser.add_argument("--export_orbax_dir", type=str, default="")
  parser.add_argument("--summary_json", type=str, default="")
  args, remaining_args = parser.parse_known_args(argv)

  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel

  cfg = pyconfig.initialize([sys.argv[0]] + remaining_args)
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
  model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  hf_dir = args.hf_checkpoint_dir if args.hf_checkpoint_dir else str(cfg.load_parameters_path)
  export_dir = args.export_orbax_dir if args.export_orbax_dir else None
  result = audit_loaded_model_against_hf(
      model=model,
      config=cfg,
      hf_checkpoint_dir=hf_dir,
      mesh=mesh,
      verify_all_text_leaves=True,
      export_orbax_dir=export_dir,
  )
  if args.summary_json and jax.process_index() == 0:
    summary_path = epath.Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2))
  print(f"DS41_AUDIT_RESULT: {json.dumps(result)}", flush=True)
  if not result["passed"]:
    raise AssertionError(f"Model weight audit against HF failed: {result['leaf_failures']}")
  return result


if __name__ == "__main__":
  _cli_main()
