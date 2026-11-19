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

"""Independent HF-to-MaxText weight/dtype/sharding audit and golden manifest pin verifier."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import logging as std_logging
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from absl import flags
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from flax.training import train_state
from jax.experimental import multihost_utils

from maxtext.common import checkpointing
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.utils import model_creation_utils

# pylint: disable=possibly-used-before-assignment
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
  import torch  # pylint: disable=g-import-not-at-top

  from tests.utils.deepseek_v41_reference import runner as ref_runner  # pylint: disable=g-import-not-at-top

EXPECTED_HF_CHECKPOINT_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
EXPECTED_HF_REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
EXIT_DETERMINISTIC_GATE_FAIL = 42

EXPECTED_TOOL_FILE_SHA256S: dict[str, str] = {
    "tests/utils/deepseek_v41_reference/vision.py": ("0f14ad58a28f774fd26313907d050ecea1670f30b23ccaf2115046c8babb4fe8"),
    "tests/utils/deepseek_v41_reference/image_processor.py": (
        "584eace12c981aae13dd5a9f4fe0124359b0ac21d3ab18c88a5ae28ce4a95488"
    ),
    "tests/utils/deepseek_v41_mm_golden/__init__.py": (
        "59c54bf4a4bbe4015b6fbde934282c90032a97f3f084099612bee4b3857f2d49"
    ),
    "tests/utils/deepseek_v41_mm_golden/reference_loader.py": (
        "f73fe242e115774ceb97ab7709259274ceff526630875ffbef8481f1ddee544c"
    ),
    "tests/utils/deepseek_v41_mm_golden/generate_mm_golden_logits.py": (
        "138884268064ef6990ee3f194919709bce48b5155751b2d42ca57e1d3ac3023d"
    ),
}

_MM_V1_SCHEMA_VERSION = "ds41_mm_golden_v1"
_MM_V2_SCHEMA_VERSION = "ds41_mm_golden_v2"
_MM_V3_SCHEMA_VERSION = "ds41_mm_golden_v3"
_MM_V3_OPTION_C_SPEC_VERSION = "mm_v3_option_c_frozen_topk_v1"
_MM_V3_OPTION_C_SPEC_SHA256 = "70974d6fbb40df2482397f1f438485af7bfbf0b6a6490324160333c63f3095b0"
_MM_V3_EXPECTED_INDEXER_LAYER_IDS = [2, 8, 14, 20, 24, 28, 32, 36]
_MM_V3_EXPECTED_INDEXER_TOPK = 512
_MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS = 26
EXPECTED_MM_V3_ROUTING_KEYS: tuple[str, ...] = (
    "ref_topk_experts",
    "ref_gate_margins",
    "ref_gate_boundary_ulp",
    "ref_indexer_topk",
    "ref_indexer_margins",
    "ref_indexer_boundary_ulp",
    "ref_indexer_layer_ids",
)
EXPECTED_MM_V3_DISTINCT_NON_L2_RANKS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
)
EXPECTED_MM_V3_DISTINCT_CROP_SOURCE_IDS: tuple[str, ...] = (
    "crop_544:B0:anchor_0:mirror_0",
    "crop_544:B1:anchor_0:mirror_0",
    "crop_544:B2:anchor_0:mirror_0",
    "crop_544:B3:anchor_0:mirror_0",
    "crop_544:B0:anchor_1:mirror_0",
    "crop_544:B1:anchor_1:mirror_0",
    "crop_544:B2:anchor_1:mirror_0",
    "crop_544:B3:anchor_1:mirror_0",
    "crop_544:B0:anchor_2:mirror_0",
    "crop_544:B1:anchor_2:mirror_0",
    "crop_544:B2:anchor_2:mirror_0",
    "crop_544:B3:anchor_2:mirror_0",
    "crop_544:B0:anchor_0:mirror_1",
    "crop_544:B1:anchor_0:mirror_1",
    "crop_544:B2:anchor_0:mirror_1",
    "crop_544:B3:anchor_0:mirror_1",
    "crop_544:B0:anchor_1:mirror_1",
    "crop_544:B1:anchor_1:mirror_1",
    "crop_544:B2:anchor_1:mirror_1",
    "crop_544:B3:anchor_1:mirror_1",
    "crop_544:B0:anchor_2:mirror_1",
    "crop_544:B1:anchor_2:mirror_1",
    "crop_544:B2:anchor_2:mirror_1",
    "crop_544:B3:anchor_2:mirror_1",
)
_MM_V2_PRESCREEN_SPEC_VERSION = "option_b1_blind_v1"
_MM_V2_PRESCREEN_SPEC_SHA256 = "6f70fbe7f5b9696f0160fb2258279647470f1aaa34da78d3ff4b296e12dfeff5"
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to (96, 64).
_MM_V2_PRESCREEN_THREAD_PAIR = (96, 64)
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to Option B-1 criteria.
_MM_V2_EXPECTED_PRESCREEN_CRITERIA: dict[str, float] = {
    "tau_gate_l0_19_image_tokens": 2.0e-6,
    "min_gate_l0_39_image_tokens_gt": 0.0,
    "min_indexer_l0_19_image_tokens_gt": 0.0,
    "perturbation_max_kl_le": 0.008,
    "perturbation_mean_kl_le": 1.0e-4,
    "perturbation_p95_kl_le": 5.0e-4,
    "perturbation_top1_agreement_ge": 0.995,
}
_MM_V2_EARLY_ROUTER_LAYER_LIMIT = 20
_MM_V2_INITIAL_REAL_SAMPLE_IDS: frozenset[str] = frozenset({"L2", "L3", "M2"})
_MM_V2_FIXED_TEST_IMAGE_SLOT0_SAMPLES: frozenset[str] = frozenset({"L4", "M4", "M6", "M8"})

REQUIRED_GOLDEN_CASES: dict[str, dict[str, Any]] = {
    "L1": {"category": "logit_forward", "num_images": 1},
    "L2": {"category": "logit_forward", "num_images": 2},
    "L3": {"category": "logit_forward", "num_images": 1},
    "L4": {"category": "logit_forward", "num_images": 4},
    "M1": {"category": "masked_prefill", "num_images": 1},
    "M2": {"category": "masked_prefill", "num_images": 1},
    "M3": {"category": "masked_prefill", "num_images": 2},
    "M4": {"category": "masked_prefill", "num_images": 2},
    "M5": {"category": "masked_prefill", "num_images": 3},
    "M6": {"category": "masked_prefill", "num_images": 3},
    "M7": {"category": "masked_prefill", "num_images": 2},
    "M8": {"category": "masked_prefill", "num_images": 4},
}

EXPECTED_MM_V1_OUTPUT_FILES: frozenset[str] = frozenset(
    {*(f"L{i}.npz" for i in range(1, 5)), *(f"M{i}.npz" for i in range(1, 9)), "mm_logit.pkl"}
)
EXPECTED_MM_V1_ARTIFACTS: tuple[str, ...] = (
    "L1.npz",
    "L2.npz",
    "L3.npz",
    "L4.npz",
    "M1.npz",
    "M2.npz",
    "M3.npz",
    "M4.npz",
    "M5.npz",
    "M6.npz",
    "M7.npz",
    "M8.npz",
    "mm_logit.pkl",
    "manifest.json",
)


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
      expected_spec = expected_leaf_avals[mt_path].sharding.spec
      if (
          not isinstance(mt_arr.sharding, jax.sharding.NamedSharding)
          or mt_arr.sharding.mesh != mesh
          or mt_arr.sharding.spec != expected_spec
      ):
        msg = f"{mt_path}: sharding={mt_arr.sharding!r}, expected NamedSharding(mesh, {expected_spec!r})"
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
          for e, plane_2d in pool.map(_load_rep_expert, chunk_exps):
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
      for e, plane_2d in pool.map(_load_expert, chunk_exps):
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


def _is_non_bool_int(val: Any) -> bool:
  return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_real(val: Any) -> bool:
  return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _expected_mm_v2_screened_slots(sample_id: str, test_image_failed_c1: bool) -> list[int]:
  num_images = int(REQUIRED_GOLDEN_CASES[sample_id]["num_images"])
  if sample_id in _MM_V2_FIXED_TEST_IMAGE_SLOT0_SAMPLES and not test_image_failed_c1:
    return list(range(1, num_images))
  return list(range(num_images))


def _verify_mm_v2_perturbation_dict(
    pert: Any,
    label: str,
    failures: list[str],
) -> None:
  """Verifies one Option B-1 `C2` perturbation KL dictionary on an accepted candidate."""
  if not isinstance(pert, Mapping):
    failures.append(f"{label} must be a mapping on accepted candidate")
    return
  req_keys = ("max_kl", "mean_kl", "p95_kl", "top1_agreement")
  if any(k not in pert or not _is_finite_real(pert[k]) for k in req_keys):
    failures.append(f"{label} must carry finite numeric {req_keys}")
    return
  crit = _MM_V2_EXPECTED_PRESCREEN_CRITERIA
  if float(pert["max_kl"]) > crit["perturbation_max_kl_le"]:
    failures.append(f"{label}.max_kl={pert['max_kl']} > {crit['perturbation_max_kl_le']}")
  if float(pert["mean_kl"]) > crit["perturbation_mean_kl_le"]:
    failures.append(f"{label}.mean_kl={pert['mean_kl']} > {crit['perturbation_mean_kl_le']}")
  if float(pert["p95_kl"]) > crit["perturbation_p95_kl_le"]:
    failures.append(f"{label}.p95_kl={pert['p95_kl']} > {crit['perturbation_p95_kl_le']}")
  if float(pert["top1_agreement"]) < crit["perturbation_top1_agreement_ge"]:
    failures.append(f"{label}.top1_agreement={pert['top1_agreement']} < {crit['perturbation_top1_agreement_ge']}")


def _verify_mm_v2_accepted_c1_and_c2(
    cand: Mapping[str, Any],
    label: str,
    failures: list[str],
) -> None:
  """Verifies Option B-1 `C1` and `C2` numeric bounds on an `accepted=True` candidate entry."""
  crit = _MM_V2_EXPECTED_PRESCREEN_CRITERIA
  worst_gate = cand["worst_gate_margin_l0_19"] if "worst_gate_margin_l0_19" in cand else None
  if not isinstance(worst_gate, Mapping) or "margin" not in worst_gate or not _is_finite_real(worst_gate["margin"]):
    failures.append(f"{label}.worst_gate_margin_l0_19 must be a mapping with finite 'margin'")
  elif float(worst_gate["margin"]) < crit["tau_gate_l0_19_image_tokens"]:
    failures.append(
        f"{label}.worst_gate_margin_l0_19.margin={worst_gate['margin']} < {crit['tau_gate_l0_19_image_tokens']}"
    )

  gate_by_layer = (
      cand["min_gate_margin_by_layer_image_tokens"] if "min_gate_margin_by_layer_image_tokens" in cand else None
  )
  if not isinstance(gate_by_layer, list) or not gate_by_layer or not all(_is_finite_real(v) for v in gate_by_layer):
    failures.append(f"{label}.min_gate_margin_by_layer_image_tokens must be a non-empty list of finite numbers")
  else:
    if min(float(v) for v in gate_by_layer) <= crit["min_gate_l0_39_image_tokens_gt"]:
      failures.append(f"{label}.min_gate_margin_by_layer_image_tokens breaches min_gate_l0_39_image_tokens_gt")
    early_gate = [float(v) for v in gate_by_layer[:_MM_V2_EARLY_ROUTER_LAYER_LIMIT]]
    if min(early_gate) < crit["tau_gate_l0_19_image_tokens"]:
      failures.append(f"{label}.min_gate_margin_by_layer_image_tokens[:20] breaches tau_gate_l0_19_image_tokens")

  idx_by_layer = (
      cand["min_indexer_margin_by_layer_image_tokens"] if "min_indexer_margin_by_layer_image_tokens" in cand else None
  )
  if (
      not isinstance(idx_by_layer, list)
      or (isinstance(gate_by_layer, list) and len(idx_by_layer) != len(gate_by_layer))
      or not all(v is None or _is_finite_real(v) for v in idx_by_layer)
  ):
    failures.append(f"{label}.min_indexer_margin_by_layer_image_tokens must match layer count with finite/None values")
  else:
    early_idx = [float(v) for v in idx_by_layer[:_MM_V2_EARLY_ROUTER_LAYER_LIMIT] if v is not None]
    if early_idx and min(early_idx) <= crit["min_indexer_l0_19_image_tokens_gt"]:
      failures.append(f"{label}.min_indexer_margin_by_layer_image_tokens[:20] breaches min_indexer_l0_19_image_tokens_gt")

  pert_64 = cand["perturbation_64t_vs_96t"] if "perturbation_64t_vs_96t" in cand else None
  pert_ulp = cand["perturbation_one_ulp_vs_96t"] if "perturbation_one_ulp_vs_96t" in cand else None
  _verify_mm_v2_perturbation_dict(pert_64, f"{label}.perturbation_64t_vs_96t", failures)
  _verify_mm_v2_perturbation_dict(pert_ulp, f"{label}.perturbation_one_ulp_vs_96t", failures)


def _verify_mm_v2_candidate_record(
    cand: Any,
    idx: int,
    *,
    test_image_failed_c1: bool,
    by_sample: dict[str, list[Mapping[str, Any]]],
    all_ranks: list[int],
    failures: list[str],
) -> None:
  """Verifies per-entry schema, slot coverage, and `C1`/`C2` invariants for one `evaluated_candidates` entry."""
  label = f"evaluated_candidates[{idx}]"
  req_keys = (
      "sample_id",
      "attempt_index",
      "queue_ranks",
      "screened_slots",
      "slot_source_ids",
      "accepted",
      "failed_criteria",
  )
  if not isinstance(cand, Mapping) or any(k not in cand for k in req_keys):
    failures.append(f"{label} must be a mapping containing {req_keys}")
    return

  sid = cand["sample_id"]
  attempt_idx = cand["attempt_index"]
  q_ranks = cand["queue_ranks"]
  scr_slots = cand["screened_slots"]
  slot_srcs = cand["slot_source_ids"]
  accepted = cand["accepted"]
  failed_crit = cand["failed_criteria"]

  if not isinstance(sid, str) or sid not in REQUIRED_GOLDEN_CASES:
    failures.append(f"{label}.sample_id={sid!r} not in {sorted(REQUIRED_GOLDEN_CASES)}")
    return
  if not _is_non_bool_int(attempt_idx) or attempt_idx < 0:
    failures.append(f"{label}.attempt_index={attempt_idx!r} must be a non-negative int")
    return
  if not isinstance(accepted, bool):
    failures.append(f"{label}.accepted={accepted!r} must be a bool")
    return
  if not isinstance(failed_crit, list) or not all(isinstance(x, str) and bool(x) for x in failed_crit):
    failures.append(f"{label}.failed_criteria must be a list of non-empty strings")
    return

  num_images = int(REQUIRED_GOLDEN_CASES[sid]["num_images"])
  exp_slots = _expected_mm_v2_screened_slots(sid, test_image_failed_c1)
  if not isinstance(scr_slots, list) or not all(_is_non_bool_int(s) for s in scr_slots) or scr_slots != exp_slots:
    failures.append(f"{label}.screened_slots={scr_slots!r} != expected {exp_slots} for {sid}")
  if (
      not isinstance(slot_srcs, list)
      or len(slot_srcs) != num_images
      or not all(isinstance(s, str) and bool(s) for s in slot_srcs)
  ):
    failures.append(f"{label}.slot_source_ids must list {num_images} non-empty source strings for {sid}")
  elif (
      sid in _MM_V2_FIXED_TEST_IMAGE_SLOT0_SAMPLES and not test_image_failed_c1 and slot_srcs[0] != "asset:test_image.jpg"
  ):
    failures.append(f"{label}.slot_source_ids[0]={slot_srcs[0]!r} != 'asset:test_image.jpg' for {sid}")

  if not isinstance(q_ranks, list) or not all(_is_non_bool_int(r) and r >= 0 for r in q_ranks):
    failures.append(f"{label}.queue_ranks={q_ranks!r} must be a list of non-negative ints")
  else:
    if sid in _MM_V2_INITIAL_REAL_SAMPLE_IDS and attempt_idx == 0:
      if q_ranks:
        failures.append(f"{label}.queue_ranks must be [] on initial real-image screen of {sid}")
    elif not 1 <= len(q_ranks) <= len(exp_slots):
      failures.append(f"{label}.queue_ranks length {len(q_ranks)} outside 1..{len(exp_slots)} for {sid}")
    all_ranks.extend(int(r) for r in q_ranks)

  if accepted:
    if failed_crit:
      failures.append(f"{label} accepted=True must have failed_criteria==[], got {failed_crit}")
    if isinstance(slot_srcs, list) and any(isinstance(s, str) and s.startswith("synth:") for s in slot_srcs):
      failures.append(f"{label} accepted=True contains synthetic slot_source_ids: {slot_srcs}")
    _verify_mm_v2_accepted_c1_and_c2(cand, label, failures)
  elif not failed_crit:
    failures.append(f"{label} accepted=False must have non-empty failed_criteria")

  by_sample.setdefault(sid, []).append(cand)


def _verify_mm_v2_candidate_ledger(
    evaluated_candidates: list[Any],
    test_image_failed_c1: bool,
    failures: list[str],
) -> None:
  """Verifies Option B-1 Spec Section 4 suite-wide invariants across `prescreen_audit.evaluated_candidates`."""
  by_sample: dict[str, list[Mapping[str, Any]]] = {}
  all_ranks: list[int] = []
  for idx, cand in enumerate(evaluated_candidates):
    _verify_mm_v2_candidate_record(
        cand,
        idx,
        test_image_failed_c1=test_image_failed_c1,
        by_sample=by_sample,
        all_ranks=all_ranks,
        failures=failures,
    )

  missing_samples = sorted(set(REQUIRED_GOLDEN_CASES) - set(by_sample))
  if missing_samples:
    failures.append(f"prescreen_audit.evaluated_candidates missing required samples: {missing_samples}")

  for sid in sorted(by_sample):
    entries = by_sample[sid]
    attempts = [int(e["attempt_index"]) for e in entries]
    if attempts != list(range(len(entries))):
      failures.append(
          f"prescreen_audit.evaluated_candidates[{sid}] attempt_index={attempts} != {list(range(len(entries)))}"
      )
    accepted_flags = [bool(e["accepted"]) for e in entries]
    if sum(accepted_flags) != 1 or not accepted_flags[-1]:
      failures.append(
          f"prescreen_audit.evaluated_candidates[{sid}] must end in unique accepted=True entry, got {accepted_flags}"
      )

  if len(all_ranks) != len(set(all_ranks)):
    failures.append(f"prescreen_audit.evaluated_candidates has duplicate queue_ranks: {all_ranks}")
  elif sorted(all_ranks) != list(range(len(all_ranks))):
    failures.append(
        f"prescreen_audit.evaluated_candidates queue_ranks {sorted(all_ranks)} "
        f"do not form contiguous 0..{len(all_ranks) - 1}"
    )


def _verify_mm_v2_prescreen_audit(
    manifest: Mapping[str, Any],
    failures: list[str],
) -> None:
  """Verifies Option B-1 `prescreen_audit` and `mm_v2` provenance invariants inside `manifest.json`."""
  if "golden_version" not in manifest or manifest["golden_version"] != "mm_v2":
    got_gv = manifest["golden_version"] if "golden_version" in manifest else None
    failures.append(f"manifest.json golden_version={got_gv!r} != 'mm_v2'")
  if "mini_mode" not in manifest or not isinstance(manifest["mini_mode"], bool) or manifest["mini_mode"] is not False:
    got_mm = manifest["mini_mode"] if "mini_mode" in manifest else None
    failures.append(f"manifest.json mini_mode={got_mm!r} must be False for production ds41_mm_golden_v2")
  if (
      "num_threads" not in manifest
      or not _is_non_bool_int(manifest["num_threads"])
      or int(manifest["num_threads"]) != _MM_V2_PRESCREEN_THREAD_PAIR[0]
  ):
    got_nt = manifest["num_threads"] if "num_threads" in manifest else None
    failures.append(f"manifest.json num_threads={got_nt!r} != {_MM_V2_PRESCREEN_THREAD_PAIR[0]}")
  if "ref_topk_experts_dtype" not in manifest or manifest["ref_topk_experts_dtype"] != "uint16":
    got_dt = manifest["ref_topk_experts_dtype"] if "ref_topk_experts_dtype" in manifest else None
    failures.append(f"manifest.json ref_topk_experts_dtype={got_dt!r} != 'uint16'")
  if "prescreen_spec_sha256" not in manifest or manifest["prescreen_spec_sha256"] != _MM_V2_PRESCREEN_SPEC_SHA256:
    got_sha = manifest["prescreen_spec_sha256"] if "prescreen_spec_sha256" in manifest else None
    failures.append(f"manifest.json prescreen_spec_sha256={got_sha!r} != {_MM_V2_PRESCREEN_SPEC_SHA256!r}")

  if "prescreen_audit" not in manifest or not isinstance(manifest["prescreen_audit"], Mapping):
    failures.append("manifest.json prescreen_audit must be a mapping for ds41_mm_golden_v2")
    return

  pa = manifest["prescreen_audit"]
  if "prescreen_spec_version" not in pa or pa["prescreen_spec_version"] != _MM_V2_PRESCREEN_SPEC_VERSION:
    got_ver = pa["prescreen_spec_version"] if "prescreen_spec_version" in pa else None
    failures.append(f"prescreen_audit.prescreen_spec_version={got_ver!r} != {_MM_V2_PRESCREEN_SPEC_VERSION!r}")
  if "prescreen_spec_sha256" not in pa or pa["prescreen_spec_sha256"] != _MM_V2_PRESCREEN_SPEC_SHA256:
    got_pa_sha = pa["prescreen_spec_sha256"] if "prescreen_spec_sha256" in pa else None
    failures.append(f"prescreen_audit.prescreen_spec_sha256={got_pa_sha!r} != {_MM_V2_PRESCREEN_SPEC_SHA256!r}")

  if (
      "primary_threads" not in pa
      or "perturbation_threads" not in pa
      or not _is_non_bool_int(pa["primary_threads"])
      or not _is_non_bool_int(pa["perturbation_threads"])
  ):
    failures.append("prescreen_audit missing integer 'primary_threads' or 'perturbation_threads'")
  else:
    got_pair = (int(pa["primary_threads"]), int(pa["perturbation_threads"]))
    if got_pair != _MM_V2_PRESCREEN_THREAD_PAIR:
      failures.append(
          f"prescreen_audit thread pair (primary_threads, perturbation_threads)={got_pair} "
          f"!= {_MM_V2_PRESCREEN_THREAD_PAIR}"
      )

  test_img_failed_ok = "test_image_failed_initial_c1" in pa and isinstance(pa["test_image_failed_initial_c1"], bool)
  if not test_img_failed_ok:
    failures.append("prescreen_audit.test_image_failed_initial_c1 must be a bool")

  if "criteria" not in pa or not isinstance(pa["criteria"], Mapping):
    failures.append("prescreen_audit.criteria must be a mapping")
  else:
    crit = pa["criteria"]
    if set(crit.keys()) != set(_MM_V2_EXPECTED_PRESCREEN_CRITERIA.keys()):
      failures.append(
          f"prescreen_audit.criteria keys {sorted(crit.keys())} != {sorted(_MM_V2_EXPECTED_PRESCREEN_CRITERIA.keys())}"
      )
    else:
      for c_key, exp_val in _MM_V2_EXPECTED_PRESCREEN_CRITERIA.items():
        if not _is_finite_real(crit[c_key]) or float(crit[c_key]) != exp_val:
          failures.append(f"prescreen_audit.criteria[{c_key!r}]={crit[c_key]!r} != {exp_val!r}")

  if (
      "evaluated_candidates" not in pa
      or not isinstance(pa["evaluated_candidates"], list)
      or not pa["evaluated_candidates"]
  ):
    failures.append("prescreen_audit.evaluated_candidates must be a non-empty list")
  elif test_img_failed_ok:
    _verify_mm_v2_candidate_ledger(pa["evaluated_candidates"], bool(pa["test_image_failed_initial_c1"]), failures)


def _verify_mm_v3_manifest(
    manifest: Mapping[str, Any],
    failures: list[str],
) -> None:
  """Verifies Option C `mm_v3` provenance, routing dtypes/shapes, and 26-slot distinct image assignment invariants."""
  if "golden_version" not in manifest or manifest["golden_version"] != "mm_v3":
    got_gv = manifest["golden_version"] if "golden_version" in manifest else None
    failures.append(f"manifest.json golden_version={got_gv!r} != 'mm_v3'")
  if "mini_mode" not in manifest or not isinstance(manifest["mini_mode"], bool) or manifest["mini_mode"] is not False:
    got_mm = manifest["mini_mode"] if "mini_mode" in manifest else None
    failures.append(f"manifest.json mini_mode={got_mm!r} must be False for production ds41_mm_golden_v3")
  if (
      "num_threads" not in manifest
      or not _is_non_bool_int(manifest["num_threads"])
      or int(manifest["num_threads"]) != _MM_V2_PRESCREEN_THREAD_PAIR[0]
  ):
    got_nt = manifest["num_threads"] if "num_threads" in manifest else None
    failures.append(f"manifest.json num_threads={got_nt!r} != {_MM_V2_PRESCREEN_THREAD_PAIR[0]}")
  if "option_c_spec_version" not in manifest or manifest["option_c_spec_version"] != _MM_V3_OPTION_C_SPEC_VERSION:
    got_ver = manifest["option_c_spec_version"] if "option_c_spec_version" in manifest else None
    failures.append(f"manifest.json option_c_spec_version={got_ver!r} != {_MM_V3_OPTION_C_SPEC_VERSION!r}")
  if "option_c_spec_sha256" not in manifest or manifest["option_c_spec_sha256"] != _MM_V3_OPTION_C_SPEC_SHA256:
    got_sha = manifest["option_c_spec_sha256"] if "option_c_spec_sha256" in manifest else None
    failures.append(f"manifest.json option_c_spec_sha256={got_sha!r} != {_MM_V3_OPTION_C_SPEC_SHA256!r}")
  for dt_key, exp_dt in (
      ("ref_topk_experts_dtype", "uint16"),
      ("ref_gate_margins_dtype", "float32"),
      ("ref_gate_boundary_ulp_dtype", "float32"),
      ("ref_indexer_topk_dtype", "int32"),
      ("ref_indexer_margins_dtype", "float32"),
      ("ref_indexer_boundary_ulp_dtype", "float32"),
  ):
    if dt_key not in manifest or manifest[dt_key] != exp_dt:
      got_dt = manifest[dt_key] if dt_key in manifest else None
      failures.append(f"manifest.json {dt_key}={got_dt!r} != {exp_dt!r}")
  if (
      "ref_indexer_layer_ids" not in manifest
      or not isinstance(manifest["ref_indexer_layer_ids"], list)
      or manifest["ref_indexer_layer_ids"] != _MM_V3_EXPECTED_INDEXER_LAYER_IDS
  ):
    got_ids = manifest["ref_indexer_layer_ids"] if "ref_indexer_layer_ids" in manifest else None
    failures.append(f"manifest.json ref_indexer_layer_ids={got_ids!r} != {_MM_V3_EXPECTED_INDEXER_LAYER_IDS}")
  if (
      "ref_indexer_topk" not in manifest
      or not _is_non_bool_int(manifest["ref_indexer_topk"])
      or int(manifest["ref_indexer_topk"]) != _MM_V3_EXPECTED_INDEXER_TOPK
  ):
    got_topk = manifest["ref_indexer_topk"] if "ref_indexer_topk" in manifest else None
    failures.append(f"manifest.json ref_indexer_topk={got_topk!r} != {_MM_V3_EXPECTED_INDEXER_TOPK}")

  if "distinct_slot_image_assignment" not in manifest or not isinstance(
      manifest["distinct_slot_image_assignment"], Mapping
  ):
    failures.append("manifest.json distinct_slot_image_assignment must be a mapping for ds41_mm_golden_v3")
    return
  dsa = manifest["distinct_slot_image_assignment"]
  if "total_image_slots" not in dsa or dsa["total_image_slots"] != _MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS:
    failures.append(
        f"distinct_slot_image_assignment.total_image_slots={dsa.get('total_image_slots')!r} "
        f"!= {_MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS}"
    )
  if "unique_slot_sha256_count" not in dsa or dsa["unique_slot_sha256_count"] != _MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS:
    failures.append(
        f"distinct_slot_image_assignment.unique_slot_sha256_count={dsa.get('unique_slot_sha256_count')!r} "
        f"!= {_MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS}"
    )
  if "all_slots_distinct" not in dsa or dsa["all_slots_distinct"] is not True:
    failures.append(f"distinct_slot_image_assignment.all_slots_distinct={dsa.get('all_slots_distinct')!r} != True")
  slot_recs = dsa["slot_records"] if "slot_records" in dsa else None
  if not isinstance(slot_recs, list) or len(slot_recs) != _MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS:
    failures.append(
        f"distinct_slot_image_assignment.slot_records must be a list of {_MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS} entries"
    )
  else:
    seen_shas: set[str] = set()
    non_l2_ranks: list[int] = []
    for idx, rec in enumerate(slot_recs):
      if not isinstance(rec, Mapping):
        failures.append(f"distinct_slot_image_assignment.slot_records[{idx}] must be a mapping")
        continue
      rec_sha = rec.get("image_sha256") or rec.get("sha256")
      rec_src = rec.get("image_source_id") or rec.get("source_id")
      if not rec_sha or not rec_src:
        failures.append(
            f"distinct_slot_image_assignment.slot_records[{idx}] missing "
            "'image_sha256'/'sha256' or 'image_source_id'/'source_id'"
        )
      else:
        seen_shas.add(str(rec_sha))
      if "rank" in rec and rec["rank"] is not None:
        if _is_non_bool_int(rec["rank"]):
          non_l2_ranks.append(int(rec["rank"]))
        else:
          failures.append(f"distinct_slot_image_assignment.slot_records[{idx}].rank={rec['rank']!r} must be int/None")
    if len(seen_shas) != _MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS:
      failures.append(
          f"distinct_slot_image_assignment.slot_records has {len(seen_shas)} unique sha256 values, "
          f"expected {_MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS}"
      )
    if non_l2_ranks and tuple(non_l2_ranks) != EXPECTED_MM_V3_DISTINCT_NON_L2_RANKS:
      failures.append(
          f"distinct_slot_image_assignment.slot_records non-L2 ranks={non_l2_ranks} "
          f"!= {list(EXPECTED_MM_V3_DISTINCT_NON_L2_RANKS)}"
      )

  samples = manifest["samples"] if "samples" in manifest else None
  expected_sample_ids = list(REQUIRED_GOLDEN_CASES.keys())
  if not isinstance(samples, list) or len(samples) != len(expected_sample_ids):
    failures.append(f"manifest.json samples must be a list of {len(expected_sample_ids)} sample records")
    return
  got_sample_ids = [s.get("sample_id") if isinstance(s, Mapping) else None for s in samples]
  if got_sample_ids != expected_sample_ids:
    failures.append(f"manifest.json samples order {got_sample_ids} != {expected_sample_ids}")
    return

  sample_slot_srcs: list[str] = []
  sample_slot_shas: list[str] = []
  non_l2_slot_srcs: list[str] = []
  for s_entry in samples:
    sid = str(s_entry["sample_id"])
    exp_n_imgs = int(REQUIRED_GOLDEN_CASES[sid]["num_images"])
    src_ids = s_entry.get("image_source_ids")
    src_shas = s_entry.get("image_source_sha256s")
    if (
        not isinstance(src_ids, list)
        or not isinstance(src_shas, list)
        or len(src_ids) != exp_n_imgs
        or len(src_shas) != exp_n_imgs
    ):
      failures.append(f"manifest.json samples[{sid}] must have {exp_n_imgs} image_source_ids and image_source_sha256s")
      continue
    if sid == "L2" and src_ids != ["release:carrots.jpeg", "release:corn.jpeg"]:
      failures.append(
          f"manifest.json samples['L2'].image_source_ids={src_ids} != ['release:carrots.jpeg', 'release:corn.jpeg']"
      )
    elif sid != "L2":
      non_l2_slot_srcs.extend(str(x) for x in src_ids)
    sample_slot_srcs.extend(str(x) for x in src_ids)
    sample_slot_shas.extend(str(x) for x in src_shas)

  if len(sample_slot_srcs) != _MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS:
    failures.append(f"manifest.json samples total image slots={len(sample_slot_srcs)} != 26")
  if len(set(sample_slot_shas)) != _MM_V3_EXPECTED_TOTAL_IMAGE_SLOTS:
    failures.append(
        f"manifest.json samples has {len(set(sample_slot_shas))} unique image_source_sha256s across 26 slots, "
        "expected 26 (0 duplicates)"
    )
  if tuple(non_l2_slot_srcs) != EXPECTED_MM_V3_DISTINCT_CROP_SOURCE_IDS:
    failures.append(
        f"manifest.json samples non-L2 image_source_ids={non_l2_slot_srcs} "
        f"!= {list(EXPECTED_MM_V3_DISTINCT_CROP_SOURCE_IDS)}"
    )


def _verify_mm_v3_on_disk_routing_keys(
    art_path: Path,
    fname: str,
    failures: list[str],
) -> None:
  """Verifies that an on-disk `mm_v3` `.npz` or `mm_logit.pkl` artifact contains all 7 Option C routing keys."""
  if fname.endswith(".npz"):
    try:
      with np.load(art_path, allow_pickle=False) as npz:
        present = set(npz.files)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append(f"Failed to load mm_v3 golden artifact {fname} as .npz: {exc}")
      return
    missing = [k for k in EXPECTED_MM_V3_ROUTING_KEYS if k not in present]
    if missing:
      failures.append(f"mm_v3 golden artifact {fname} missing required Option C routing keys: {missing}")
  elif fname == "mm_logit.pkl":
    try:
      with art_path.open("rb") as f:
        records = pickle.load(f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failures.append(f"Failed to load mm_v3 golden artifact {fname} as pickle: {exc}")
      return
    if not isinstance(records, list) or len(records) not in (4, len(REQUIRED_GOLDEN_CASES)):
      failures.append(f"mm_v3 mm_logit.pkl must be a list of 4 or {len(REQUIRED_GOLDEN_CASES)} sample dicts")
      return
    for idx, rec in enumerate(records):
      if not isinstance(rec, Mapping):
        failures.append(f"mm_v3 mm_logit.pkl[{idx}] must be a mapping")
        continue
      missing = [k for k in EXPECTED_MM_V3_ROUTING_KEYS if k not in rec]
      if missing:
        sid = rec.get("sample_id", idx)
        failures.append(f"mm_v3 mm_logit.pkl[{sid}] missing required Option C routing keys: {missing}")


def _verify_golden_output_files_on_disk(
    manifest_dir: Path,
    *,
    out_sha_map: Any,
    out_keys_ok: bool,
    pinned_outputs: Any,
    pinned_outputs_ok: bool,
    failures: list[str],
    schema_ver: str = _MM_V1_SCHEMA_VERSION,
) -> None:
  """Hashes all 13 on-disk `.npz` / `mm_logit.pkl` golden files against `manifest.json` and `pin`."""
  for fname in sorted(EXPECTED_MM_V1_OUTPUT_FILES):
    art_path = manifest_dir / fname
    if not art_path.is_file():
      failures.append(f"Missing golden output artifact on disk: {art_path}")
      continue
    disk_sha = _sha256_file(art_path)
    if out_keys_ok and disk_sha != out_sha_map[fname]:
      failures.append(f"Disk sha256 mismatch for golden artifact {fname}: disk={disk_sha}, manifest={out_sha_map[fname]}")
    if pinned_outputs_ok and disk_sha != pinned_outputs[fname]:
      failures.append(
          f"Disk sha256 mismatch against pin for golden artifact {fname}: disk={disk_sha}, pin={pinned_outputs[fname]}"
      )
    if schema_ver == _MM_V3_SCHEMA_VERSION:
      _verify_mm_v3_on_disk_routing_keys(art_path, fname, failures)


def _verify_manifest_against_pin(
    manifest: Mapping[str, Any],
    pin: Mapping[str, Any],
    failures: list[str],
    *,
    manifest_dir: Path | None = None,
    require_disk_files: bool = True,
) -> None:
  """Verifies a `ds41_mm_golden_v3`, `ds41_mm_golden_v2`, or `ds41_mm_golden_v1` `manifest.json` against `pin`."""
  required_manifest_keys = (
      "schema_version",
      "release_revision",
      "u2_commit_sha",
      "tool_files_sha256",
      "output_files_sha256",
  )
  missing_mkeys = [k for k in required_manifest_keys if k not in manifest]
  if missing_mkeys:
    failures.append(f"manifest.json missing required keys: {missing_mkeys}")
    return

  schema_ver = manifest["schema_version"]
  if schema_ver not in (_MM_V3_SCHEMA_VERSION, _MM_V2_SCHEMA_VERSION, _MM_V1_SCHEMA_VERSION):
    failures.append(
        f"manifest.json schema_version={schema_ver!r} not in "
        f"({_MM_V3_SCHEMA_VERSION!r}, {_MM_V2_SCHEMA_VERSION!r}, {_MM_V1_SCHEMA_VERSION!r})"
    )
    return

  if manifest["release_revision"] != pin["hf_revision"]:
    failures.append(f"manifest.json release_revision={manifest['release_revision']!r} != {pin['hf_revision']!r}")
  if manifest["u2_commit_sha"] != pin["u2_commit"]:
    failures.append(f"manifest.json u2_commit_sha={manifest['u2_commit_sha']!r} != {pin['u2_commit']!r}")

  if schema_ver == _MM_V3_SCHEMA_VERSION:
    expected_tools = pin["pinned_tool_files"]
  elif schema_ver == _MM_V2_SCHEMA_VERSION:
    expected_tools = pin["mm_v2_pinned_tool_files"] if "mm_v2_pinned_tool_files" in pin else pin["pinned_tool_files"]
  elif "mm_v1_pinned_tool_files" in pin:
    expected_tools = pin["mm_v1_pinned_tool_files"]
  else:
    expected_tools = pin["pinned_tool_files"]

  tool_sha_map = manifest["tool_files_sha256"]
  if not isinstance(tool_sha_map, Mapping) or not tool_sha_map:
    failures.append("manifest.json tool_files_sha256 must be a non-empty mapping")
  elif not isinstance(expected_tools, Mapping):
    failures.append("golden_manifest_pin.json expected tool SHA-256 table must be a mapping")
  else:
    for t_key, t_sha in tool_sha_map.items():
      if t_key not in expected_tools:
        failures.append(f"manifest.json tool_files_sha256 has unpinned tool key {t_key!r}")
      elif expected_tools[t_key] != t_sha:
        failures.append(f"manifest.json tool_files_sha256[{t_key!r}]={t_sha!r} != {expected_tools[t_key]!r}")
    for req_tool_k in expected_tools:
      if req_tool_k.startswith("tests/utils/deepseek_v41_mm_golden/") and req_tool_k not in tool_sha_map:
        failures.append(f"manifest.json tool_files_sha256 missing required golden tool key {req_tool_k!r}")

  out_sha_map = manifest["output_files_sha256"]
  out_keys_ok = isinstance(out_sha_map, Mapping) and set(out_sha_map.keys()) == set(EXPECTED_MM_V1_OUTPUT_FILES)
  if not out_keys_ok:
    got_keys = sorted(out_sha_map.keys()) if isinstance(out_sha_map, Mapping) else type(out_sha_map)
    failures.append(f"manifest.json output_files_sha256 keys {got_keys} != {sorted(EXPECTED_MM_V1_OUTPUT_FILES)}")

  if schema_ver == _MM_V3_SCHEMA_VERSION:
    _verify_mm_v3_manifest(manifest, failures)
    pinned_outputs = pin["mm_v3_outputs_sha256"] if "mm_v3_outputs_sha256" in pin else None
    pinned_keys_set = set(pinned_outputs.keys()) if isinstance(pinned_outputs, Mapping) else set()
    pinned_outputs_ok = pinned_keys_set in (
        set(EXPECTED_MM_V1_OUTPUT_FILES),
        set(EXPECTED_MM_V1_ARTIFACTS),
    )
    if not pinned_outputs_ok:
      got_n = len(pinned_outputs) if isinstance(pinned_outputs, Mapping) else 0
      failures.append(
          f"golden_manifest_pin.json mm_v3_outputs_sha256 is unpopulated or incomplete "
          f"(got {got_n} keys, expected {len(EXPECTED_MM_V1_OUTPUT_FILES)} or {len(EXPECTED_MM_V1_ARTIFACTS)})"
      )
    else:
      pinned_binary_outputs = {k: pinned_outputs[k] for k in EXPECTED_MM_V1_OUTPUT_FILES}
      if out_keys_ok and dict(out_sha_map) != pinned_binary_outputs:
        failures.append("manifest.json output_files_sha256 does not match pin['mm_v3_outputs_sha256']")
  elif schema_ver == _MM_V2_SCHEMA_VERSION:
    _verify_mm_v2_prescreen_audit(manifest, failures)
    pinned_outputs = pin["mm_v2_outputs_sha256"] if "mm_v2_outputs_sha256" in pin else None
    pinned_outputs_ok = isinstance(pinned_outputs, Mapping) and set(pinned_outputs.keys()) == set(
        EXPECTED_MM_V1_OUTPUT_FILES
    )
    if not pinned_outputs_ok:
      got_n = len(pinned_outputs) if isinstance(pinned_outputs, Mapping) else 0
      failures.append(
          f"golden_manifest_pin.json mm_v2_outputs_sha256 is unpopulated or incomplete "
          f"(got {got_n} keys, expected {len(EXPECTED_MM_V1_OUTPUT_FILES)})"
      )
    elif out_keys_ok and dict(out_sha_map) != dict(pinned_outputs):
      failures.append("manifest.json output_files_sha256 does not match pin['mm_v2_outputs_sha256']")
  else:
    pinned_outputs = pin["mm_v1_outputs_sha256"] if "mm_v1_outputs_sha256" in pin else None
    pinned_outputs_ok = isinstance(pinned_outputs, Mapping) and set(pinned_outputs.keys()) == set(
        EXPECTED_MM_V1_OUTPUT_FILES
    )
    if not pinned_outputs_ok:
      failures.append("golden_manifest_pin.json mm_v1_outputs_sha256 must map all 13 mm_v1 output artifacts")
    elif out_keys_ok and dict(out_sha_map) != dict(pinned_outputs):
      failures.append("manifest.json output_files_sha256 does not match pin['mm_v1_outputs_sha256']")

  if require_disk_files and manifest_dir is not None:
    _verify_golden_output_files_on_disk(
        manifest_dir,
        out_sha_map=out_sha_map,
        out_keys_ok=out_keys_ok,
        pinned_outputs=pinned_outputs,
        pinned_outputs_ok=pinned_outputs_ok,
        failures=failures,
        schema_ver=str(schema_ver),
    )


def verify_golden_manifest_pin(
    pin_path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] | None = None,
    tool_root: str | os.PathLike[str] | None = None,
    manifest_json_path: str | os.PathLike[str] | None = None,
    require_disk_files: bool = True,
) -> dict[str, Any]:
  """Verifies `golden_manifest_pin.json` against repo provenance, tool sha256s, and optional generated `manifest.json`."""
  pin_file = Path(pin_path)
  with open(pin_file, "r", encoding="utf-8") as f:
    pin = json.load(f)

  root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
  t_root = Path(tool_root) if tool_root is not None else root
  failures: list[str] = []

  if pin["hf_checkpoint_id"] != EXPECTED_HF_CHECKPOINT_ID:
    failures.append(f"hf_checkpoint_id={pin['hf_checkpoint_id']!r} != {EXPECTED_HF_CHECKPOINT_ID!r}")
  if pin["hf_revision"] != EXPECTED_HF_REVISION:
    failures.append(f"hf_revision={pin['hf_revision']!r} != {EXPECTED_HF_REVISION!r}")

  prov_path = root / pin["provenance_json_path"]
  if not prov_path.exists():
    failures.append(f"Missing provenance.json at {prov_path}")
  else:
    with open(prov_path, "r", encoding="utf-8") as f:
      prov = json.load(f)
    prov_repo_id = prov["repo_id"] if "repo_id" in prov else prov["hf_checkpoint_id"]
    prov_rev = prov["revision"] if "revision" in prov else prov["hf_revision"]
    if prov_repo_id != pin["hf_checkpoint_id"]:
      failures.append("provenance.json hf_checkpoint_id mismatch")
    if prov_rev != pin["hf_revision"]:
      failures.append("provenance.json hf_revision mismatch")
    if "vendored_files" in prov:
      for fname, meta in prov["vendored_files"].items():
        fpath = root / meta["vendored_path"]
        if not fpath.exists():
          failures.append(f"Missing vendored reference file {fpath} ({fname})")
    elif "files" in prov:
      ref_dir = prov_path.parent
      for fname, meta in prov["files"].items():
        fpath = ref_dir / fname
        if not fpath.exists():
          failures.append(f"Missing reference file {fpath}")
        else:
          actual_sha = _sha256_file(fpath)
          if actual_sha != meta["sha256"]:
            failures.append(f"Reference file {fname} sha256={actual_sha} != {meta['sha256']}")

  pinned_tools = pin["pinned_tool_files"]
  if set(pinned_tools.keys()) != set(EXPECTED_TOOL_FILE_SHA256S.keys()):
    failures.append(
        f"pinned_tool_files keys {sorted(pinned_tools.keys())} != {sorted(EXPECTED_TOOL_FILE_SHA256S.keys())}"
    )
  for rel_path, expected_sha in EXPECTED_TOOL_FILE_SHA256S.items():
    if rel_path not in pinned_tools:
      continue
    if pinned_tools[rel_path] != expected_sha:
      failures.append(f"Pin mismatch for {rel_path}: pin has {pinned_tools[rel_path]}, expected {expected_sha}")
    cand_path = t_root / rel_path
    if not cand_path.exists():
      cand_path = root / rel_path
    if cand_path.exists():
      disk_sha = _sha256_file(cand_path)
      if disk_sha != pinned_tools[rel_path]:
        failures.append(f"Disk sha256 mismatch for {rel_path}: disk={disk_sha}, pin={pinned_tools[rel_path]}")
    elif require_disk_files:
      failures.append(f"Missing pinned tool file on disk: {rel_path}")

  cases = pin["expected_golden_cases"]
  if set(cases.keys()) != set(REQUIRED_GOLDEN_CASES.keys()):
    failures.append(f"expected_golden_cases keys {sorted(cases.keys())} != {sorted(REQUIRED_GOLDEN_CASES.keys())}")
  for cid, req in REQUIRED_GOLDEN_CASES.items():
    if cid not in cases:
      continue
    entry = cases[cid]
    if entry["category"] != req["category"] or int(entry["num_images"]) != int(req["num_images"]):
      failures.append(f"Case {cid} metadata mismatch: {entry} vs {req}")

  if "mm_v1_expected_artifacts" not in pin:
    failures.append("golden_manifest_pin.json missing 'mm_v1_expected_artifacts'")
  elif tuple(pin["mm_v1_expected_artifacts"]) != EXPECTED_MM_V1_ARTIFACTS:
    failures.append(f"mm_v1_expected_artifacts={pin['mm_v1_expected_artifacts']} != {list(EXPECTED_MM_V1_ARTIFACTS)}")

  if "mm_v2_expected_artifacts" not in pin:
    failures.append("golden_manifest_pin.json missing 'mm_v2_expected_artifacts'")
  elif tuple(pin["mm_v2_expected_artifacts"]) != EXPECTED_MM_V1_ARTIFACTS:
    failures.append(f"mm_v2_expected_artifacts={pin['mm_v2_expected_artifacts']} != {list(EXPECTED_MM_V1_ARTIFACTS)}")

  if "mm_v3_expected_artifacts" in pin and tuple(pin["mm_v3_expected_artifacts"]) != EXPECTED_MM_V1_ARTIFACTS:
    failures.append(f"mm_v3_expected_artifacts={pin['mm_v3_expected_artifacts']} != {list(EXPECTED_MM_V1_ARTIFACTS)}")

  if "golden_version" in pin and pin["golden_version"] != "mm_v3":
    failures.append(f"golden_manifest_pin.json golden_version={pin['golden_version']!r} != 'mm_v3'")

  if (
      "mm_v3_expected_distinct_non_l2_ranks" in pin
      and tuple(pin["mm_v3_expected_distinct_non_l2_ranks"]) != EXPECTED_MM_V3_DISTINCT_NON_L2_RANKS
  ):
    failures.append(
        f"mm_v3_expected_distinct_non_l2_ranks={pin['mm_v3_expected_distinct_non_l2_ranks']} "
        f"!= {list(EXPECTED_MM_V3_DISTINCT_NON_L2_RANKS)}"
    )

  if "mm_v1_pinned_tool_files" in pin:
    v1_pinned = pin["mm_v1_pinned_tool_files"]
    if not isinstance(v1_pinned, Mapping) or set(v1_pinned.keys()) != set(EXPECTED_TOOL_FILE_SHA256S.keys()):
      failures.append(
          f"mm_v1_pinned_tool_files must map the exact 5 tool keys {sorted(EXPECTED_TOOL_FILE_SHA256S.keys())}"
      )

  if "mm_v2_pinned_tool_files" in pin:
    v2_pinned = pin["mm_v2_pinned_tool_files"]
    if not isinstance(v2_pinned, Mapping) or set(v2_pinned.keys()) != set(EXPECTED_TOOL_FILE_SHA256S.keys()):
      failures.append(
          f"mm_v2_pinned_tool_files must map the exact 5 tool keys {sorted(EXPECTED_TOOL_FILE_SHA256S.keys())}"
      )

  if manifest_json_path is not None:
    mpath = Path(manifest_json_path)
    with open(mpath, "r", encoding="utf-8") as f:
      manifest = json.load(f)
    _verify_manifest_against_pin(
        manifest,
        pin,
        failures,
        manifest_dir=mpath.parent,
        require_disk_files=require_disk_files,
    )
    if (
        require_disk_files
        and mpath.name == "manifest.json"
        and isinstance(manifest, Mapping)
        and manifest.get("schema_version") == _MM_V3_SCHEMA_VERSION
        and isinstance(pin.get("mm_v3_outputs_sha256"), Mapping)
        and "manifest.json" in pin["mm_v3_outputs_sha256"]
    ):
      disk_m_sha = _sha256_file(mpath)
      expected_m_sha = pin["mm_v3_outputs_sha256"]["manifest.json"]
      if disk_m_sha != expected_m_sha:
        failures.append(f"Disk sha256 mismatch against pin for manifest.json: disk={disk_m_sha}, pin={expected_m_sha}")

  return {
      "passed": len(failures) == 0,
      "failures": failures,
      "checked_tool_files": len(pinned_tools),
      "checked_cases": len(cases),
  }


_COMPLETED_STEP_RE = re.compile(
    r"completed step:\s*(\d+)\b.*?loss:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r".*?moe_lb_loss:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
_DS41_MM_NORMS_RE = re.compile(
    r"DS41_MM_NORMS\s+step=(\d+)\s+vit_norm=(\S+)\s+aligner_norm=(\S+)"
    r"\s+delim_norm=(\S+)\s+bias_vl_norm=\[([^\]]*)\](.*)"
)
_PEAK_HBM_KV_RE = re.compile(r"\bpeak_hbm_gb=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_BYTES_IN_USE_KV_RE = re.compile(r"\bbytes_in_use_gib=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_LAYER_BIAS_VL_KEY_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias_vl$")


def _hex_to_f32_bits(hex_str: str) -> int:
  """Converts a float hex string into its IEEE-754 float32 uint32 bit pattern."""
  val = np.float32(float.fromhex(hex_str.strip()))
  return int(val.view(np.uint32))


def parse_r4_log_text(log_text: str) -> dict[str, Any]:
  """Parses `completed step` and `DS41_MM_NORMS` lines (`peak_hbm_gb` and `bytes_in_use_gib`) from R4 log text."""
  completed_steps: dict[int, dict[str, float]] = {}
  mm_norms: dict[int, dict[str, Any]] = {}
  max_peak_hbm_gb: float | None = None
  max_step_bytes_in_use_gib: float | None = None
  for line in log_text.splitlines():
    m_step = _COMPLETED_STEP_RE.search(line)
    if m_step is not None:
      step_idx = int(m_step.group(1))
      completed_steps[step_idx] = {
          "loss": float(m_step.group(2)),
          "moe_lb_loss": float(m_step.group(3)),
      }
    m_norm = _DS41_MM_NORMS_RE.search(line)
    if m_norm is not None:
      step_idx = int(m_norm.group(1))
      bvl_list = [tok.strip() for tok in m_norm.group(5).split(",") if tok.strip()]
      entry: dict[str, Any] = {
          "vit_norm": m_norm.group(2).strip(),
          "aligner_norm": m_norm.group(3).strip(),
          "delim_norm": m_norm.group(4).strip(),
          "bias_vl_norm": bvl_list,
      }
      tail = m_norm.group(6) if m_norm.group(6) is not None else ""
      m_peak = _PEAK_HBM_KV_RE.search(tail)
      if m_peak is not None:
        step_peak = float(m_peak.group(1))
        entry["peak_hbm_gb"] = step_peak
        max_peak_hbm_gb = step_peak if max_peak_hbm_gb is None else max(max_peak_hbm_gb, step_peak)
      m_in_use = _BYTES_IN_USE_KV_RE.search(tail)
      if m_in_use is not None:
        step_in_use = float(m_in_use.group(1))
        entry["bytes_in_use_gib"] = step_in_use
        max_step_bytes_in_use_gib = (
            step_in_use if max_step_bytes_in_use_gib is None else max(max_step_bytes_in_use_gib, step_in_use)
        )
      mm_norms[step_idx] = entry
  return {
      "completed_steps": completed_steps,
      "mm_norms": mm_norms,
      "peak_hbm_gb": max_peak_hbm_gb,
      "step_peak_bytes_in_use_gib": max_step_bytes_in_use_gib,
  }


def parse_r4_log(log_path: str | os.PathLike[str]) -> dict[str, Any]:
  """Reads an R4 training log file from local disk or GCS (`epath.Path`) and parses its lines."""
  log_text = epath.Path(str(log_path)).read_text(encoding="utf-8")
  return parse_r4_log_text(log_text)


def parse_r4_metrics_text(metrics_text: str) -> dict[int, dict[str, Any]]:
  """Parses MaxText JSONL `metrics_file` text indexed by integer training step."""
  by_step: dict[int, dict[str, Any]] = {}
  for raw_line in metrics_text.splitlines():
    stripped = raw_line.strip()
    if not stripped:
      continue
    record = json.loads(stripped)
    step_idx = int(round(float(record["step"])))
    by_step[step_idx] = record
  return by_step


def parse_r4_metrics_file(metrics_path: str | os.PathLike[str]) -> dict[int, dict[str, Any]]:
  """Reads a MaxText JSONL `metrics_file` from local disk or GCS (`epath.Path`) and indexes records by step."""
  metrics_text = epath.Path(str(metrics_path)).read_text(encoding="utf-8")
  return parse_r4_metrics_text(metrics_text)


def load_hf_bias_vl_norms_hex(hf_checkpoint_dir: str | os.PathLike[str]) -> list[str]:
  """Computes per-layer float32 L2 norm hex strings for `layers.<idx>.ffn.gate.bias_vl` on CPU from safetensors."""
  with _SafetensorsShardReader(hf_checkpoint_dir, key_filter_re=_LAYER_BIAS_VL_KEY_RE) as reader:
    layer_keys: dict[int, str] = {}
    for key in reader.all_keys():
      m = _LAYER_BIAS_VL_KEY_RE.match(key)
      if m is not None:
        layer_keys[int(m.group(1))] = key
    if not layer_keys:
      raise KeyError(f"No layers.<idx>.ffn.gate.bias_vl tensors found in {hf_checkpoint_dir}")
    expected_indices = list(range(len(layer_keys)))
    if sorted(layer_keys) != expected_indices:
      raise ValueError(
          f"Non-contiguous bias_vl layer indices in {hf_checkpoint_dir}: "
          f"got {sorted(layer_keys)}, expected {expected_indices}"
      )
    norms_hex: list[str] = []
    for layer_idx in expected_indices:
      arr = reader.load_dequantized_tensor(layer_keys[layer_idx], "float32", use_cache=False).astype(np.float32)
      l2_f32 = np.float32(np.sqrt(np.sum(np.square(arr), dtype=np.float32), dtype=np.float32))
      norms_hex.append(float(l2_f32).hex())
    return norms_hex


def load_hf_bias_vl_tensors(
    hf_checkpoint_dir: str | os.PathLike[str],
    *,
    expected_num_layers: int | None = 40,
    expected_shape: tuple[int, ...] | None = (384,),
) -> dict[int, np.ndarray]:
  """Loads all `layers.<idx>.ffn.gate.bias_vl` float32 tensors from local or `gs://` safetensors."""
  with _SafetensorsShardReader(hf_checkpoint_dir, key_filter_re=_LAYER_BIAS_VL_KEY_RE) as reader:
    layer_keys: dict[int, str] = {}
    for key in reader.all_keys():
      m = _LAYER_BIAS_VL_KEY_RE.match(key)
      if m is not None:
        layer_keys[int(m.group(1))] = key
    if expected_num_layers is None:
      if not layer_keys:
        raise KeyError(f"No layers.<idx>.ffn.gate.bias_vl tensors found in {hf_checkpoint_dir}")
      expected_indices = list(range(len(layer_keys)))
    else:
      expected_indices = list(range(int(expected_num_layers)))
    if sorted(layer_keys) != expected_indices:
      raise ValueError(
          f"Expected {len(expected_indices)} contiguous bias_vl layers 0..{len(expected_indices) - 1} "
          f"in {hf_checkpoint_dir}, got {sorted(layer_keys)}"
      )

    def _read_layer_tensor(layer_idx: int) -> tuple[int, str, np.ndarray]:
      key = layer_keys[layer_idx]
      entry = reader.get_header_entry(key)
      if "dtype" not in entry or str(entry["dtype"]) != "F32":
        header_dtype = entry["dtype"] if "dtype" in entry else None
        raise ValueError(f"HF tensor {key} has header dtype {header_dtype!r}, expected 'F32'")
      raw_t = reader.load_raw_tensor(key)
      return layer_idx, key, np.array(raw_t.numpy() if hasattr(raw_t, "numpy") else raw_t, copy=True)

    max_workers = min(16, max(1, len(expected_indices)))
    loaded_by_idx: dict[int, tuple[str, np.ndarray]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
      for layer_idx, key, arr in pool.map(_read_layer_tensor, expected_indices):
        loaded_by_idx[layer_idx] = (key, arr)

    tensors: dict[int, np.ndarray] = {}
    for layer_idx in expected_indices:
      key, arr = loaded_by_idx[layer_idx]
      target_shape = (
          tuple(expected_shape) if expected_shape is not None else (tensors[0].shape if 0 in tensors else arr.shape)
      )
      if arr.dtype != np.float32 or arr.ndim != 1 or arr.shape != target_shape:
        raise ValueError(
            f"HF tensor {key} has dtype={arr.dtype}, shape={arr.shape}, expected dtype=float32, shape={target_shape}"
        )
      tensors[layer_idx] = arr
    return tensors


def _materialize_host_f32_array(node: Any) -> np.ndarray:
  """Unwraps an NNX variable or array and materializes a full host float32 numpy array."""
  arr = node[...] if isinstance(node, nnx.Variable) else node
  if isinstance(arr, jax.Array):
    if not arr.is_fully_addressable:
      arr = multihost_utils.process_allgather(arr, tiled=True)
    arr = jax.device_get(arr)
  return np.array(arr, dtype=np.float32, copy=True)


def extract_model_bias_vl_tensors(config: Any, decoder_state: Any) -> dict[int, np.ndarray]:
  """Extracts all decoder `bias_vl` tensors (`0..num_decoder_layers-1`) from scanned or unscanned state."""
  if isinstance(decoder_state, nnx.Module):
    decoder_state = nnx.state(decoder_state)
  if (
      isinstance(decoder_state, Mapping)
      and "decoder" in decoder_state
      and "segments_0" not in decoder_state
      and "layers_0" not in decoder_state
  ):
    decoder_state = decoder_state["decoder"]
  num_layers = int(config.num_decoder_layers)
  by_layer: dict[int, np.ndarray] = {}
  if bool(config.scan_layers):
    for seg_idx, segment in enumerate(deepseek41.build_segments(config)):
      seg_state = decoder_state[f"segments_{seg_idx}"]
      for slot_idx in range(len(segment.groups[0])):
        stacked_node = seg_state[f"layers_{slot_idx}"]["mlp"]["MoeBlock_0"]["gate"]["bias_vl"]
        stacked = _materialize_host_f32_array(stacked_node)
        if stacked.ndim != 2 or stacked.shape[0] != len(segment.groups):
          raise ValueError(
              f"segments_{seg_idx}/layers_{slot_idx} bias_vl shape {stacked.shape} "
              f"does not match {len(segment.groups)} scan groups"
          )
        for rep_idx, group in enumerate(segment.groups):
          by_layer[int(group[slot_idx])] = np.array(stacked[rep_idx], dtype=np.float32, copy=True)
  else:
    for layer_idx in range(num_layers):
      node = decoder_state[f"layers_{layer_idx}"]["mlp"]["MoeBlock_0"]["gate"]["bias_vl"]
      by_layer[layer_idx] = _materialize_host_f32_array(node)
  expected_indices = list(range(num_layers))
  if sorted(by_layer) != expected_indices:
    raise ValueError(f"Extracted bias_vl layer indices {sorted(by_layer)} != {expected_indices}")
  return {idx: by_layer[idx] for idx in expected_indices}


def _configure_r4_info_logging() -> None:
  """Configures `absl.logging` and root Python logging so `max_logging.log` emits `INFO` lines to `sys.stdout`."""
  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()
  absl_logging.use_absl_handler()
  absl_logging.set_verbosity(absl_logging.INFO)
  absl_logging.set_stderrthreshold(absl_logging.INFO)
  std_logging.getLogger().setLevel(std_logging.INFO)
  absl_handler = absl_logging.get_absl_handler()
  if absl_handler is not None and absl_handler.python_handler is not None:
    absl_handler.python_handler.stream = sys.stdout
    absl_handler.python_handler.setLevel(std_logging.INFO)


def _flush_r4_info_logging() -> None:
  """Flushes `absl` Python logging handler plus `sys.stdout` and `sys.stderr`."""
  absl_handler = absl_logging.get_absl_handler()
  if absl_handler is not None and absl_handler.python_handler is not None:
    absl_handler.python_handler.flush()
  sys.stdout.flush()
  sys.stderr.flush()


def run_train_with_bias_vl_dump(
    train_argv: Sequence[str],
    bias_vl_npz_path: str | os.PathLike[str],
) -> dict[str, np.ndarray]:
  """Runs `maxtext.trainers.pre_train.train.main` while dumping step 0 and last-step `bias_vl` outside JIT."""
  from maxtext.trainers.pre_train import train as maxtext_train  # pylint: disable=import-outside-toplevel

  norm_argv = list(train_argv)
  if norm_argv and norm_argv[0] == "--":
    norm_argv = norm_argv[1:]
  if not norm_argv or not norm_argv[0].endswith(".py"):
    norm_argv = [maxtext_train.__file__] + norm_argv

  _configure_r4_info_logging()
  orig_iteration = maxtext_train.training_loop_iteration
  captured_by_step: dict[int, dict[int, np.ndarray]] = {}
  observed_total_steps: list[int] = []

  def _wrapped_training_loop_iteration(
      jax_device_state: dict[str, Any],
      python_vars: dict[str, Any],
      immutable_data: dict[str, Any],
      *args: Any,
      **kwargs: Any,
  ) -> Any:
    _configure_r4_info_logging()
    current_step = int(python_vars["step"])
    total_steps = int(immutable_data["steps"])
    if not observed_total_steps:
      observed_total_steps.append(total_steps)
    last_step = total_steps - 1
    metrics = orig_iteration(jax_device_state, python_vars, immutable_data, *args, **kwargs)
    _flush_r4_info_logging()
    if current_step in (0, last_step):
      cfg = immutable_data["config"]
      dec_state = nnx.state(jax_device_state["state"].model)["decoder"]
      captured_by_step[current_step] = extract_model_bias_vl_tensors(cfg, dec_state)
    return metrics

  maxtext_train.training_loop_iteration = _wrapped_training_loop_iteration
  try:
    maxtext_train.main(norm_argv)
  finally:
    maxtext_train.training_loop_iteration = orig_iteration
    _flush_r4_info_logging()

  if not observed_total_steps:
    raise RuntimeError("training_loop_iteration was never invoked during run_train_with_bias_vl_dump")
  last_step = observed_total_steps[0] - 1
  if 0 not in captured_by_step or last_step not in captured_by_step:
    raise RuntimeError(
        f"Expected step 0 and last_step {last_step} bias_vl captures, got steps {sorted(captured_by_step)}"
    )
  num_layers = len(captured_by_step[0])
  npz_payload: dict[str, np.ndarray] = {
      "steps": np.asarray([0, last_step], dtype=np.int32),
      "layer_order": np.arange(num_layers, dtype=np.int32),
  }
  for layer_idx in range(num_layers):
    npz_payload[f"step_0_layer_{layer_idx}"] = np.asarray(captured_by_step[0][layer_idx], dtype=np.float32)
    npz_payload[f"step_{last_step}_layer_{layer_idx}"] = np.asarray(
        captured_by_step[last_step][layer_idx], dtype=np.float32
    )
  if not str(bias_vl_npz_path).startswith("gs://") or jax.process_index() == 0:
    buf = io.BytesIO()
    np.savez(buf, **npz_payload)
    dest = epath.Path(str(bias_vl_npz_path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.getvalue())
  return npz_payload


_HEX_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_HEX40_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 85.0.
DEFAULT_R4_AOT_MAX_HBM_GIB = 85.0
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 85.0.
DEFAULT_R4_INIT_PEAK_MAX_HBM_GIB = 85.0


def resolve_repo_head_commit_sha(repo_root: str | os.PathLike[str] | None = None) -> str:
  """Resolves the 40-char lowercase hex git HEAD commit SHA for `repo_root`, failing closed on error."""
  target_dir = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
  proc = subprocess.run(
      ["git", "-c", "safe.directory=*", "-C", os.fspath(target_dir), "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(
        f"git rev-parse HEAD failed in {os.fspath(target_dir)!r} (rc={proc.returncode}): {proc.stderr.strip()}"
    )
  sha = proc.stdout.strip().lower()
  if not _HEX40_COMMIT_SHA_RE.match(sha):
    raise RuntimeError(f"git rev-parse HEAD in {os.fspath(target_dir)!r} returned non-40-hex SHA: {proc.stdout!r}")
  return sha


def _resolve_default_expected_aot_commit_sha(expected_commit_sha: str | None = None) -> str:
  """Resolves expected AOT commit SHA from explicit argument, environment, or repo `HEAD`."""
  if expected_commit_sha is not None:
    return expected_commit_sha
  if "EXPECTED_AOT_COMMIT_SHA" in os.environ and os.environ["EXPECTED_AOT_COMMIT_SHA"].strip():
    return os.environ["EXPECTED_AOT_COMMIT_SHA"].strip()
  if "OVERLAY_COMMIT_SHA" in os.environ and os.environ["OVERLAY_COMMIT_SHA"].strip():
    return os.environ["OVERLAY_COMMIT_SHA"].strip()
  if "EXPECTED_OVERLAY_COMMIT_SHA" in os.environ and os.environ["EXPECTED_OVERLAY_COMMIT_SHA"].strip():
    return os.environ["EXPECTED_OVERLAY_COMMIT_SHA"].strip()
  return resolve_repo_head_commit_sha()


# TPU PJRT allocator split and 3-part like-for-like TT6/TT7 HBM verification:
# - Runtime `device.memory_stats()` (`BestFitAllocator::Allocate`) tracks ONLY PJRT heap allocations (model
#   parameters, optimizer state, batch inputs, host staging), excluding compiled XLA HLO temporaries
#   (`TpuSegmentType::kNobits` reserved via `BestFitAllocator::ReserveBottomOfMemory` into `bytes_reserved`).
# - Under `safetensors_dynamic`, `setup_initial_state` holds streamed `raw_params` and newly initialized
#   `TrainStateNNX` simultaneously before step 0, making process-lifetime `peak_bytes_in_use` (`peak_hbm_gb`) an
#   init-phase watermark distinct from the step-phase heap (`bytes_in_use_gib` sampled after each step finishes).
# - Therefore, TT6/TT7 enforces three separate fail-closed checks and logs all three in the gate JSON:
#   (a) Step-phase heap: `max(bytes_in_use_gib across steps 0..N-1) <= 1.10 * aot_heap_hbm_gib` (`arg + out - alias`).
#   (b) Init-phase heap: process-lifetime `peak_bytes_in_use` (`peak_hbm_gb`) `<= 85.0` GiB (`< 94.7` GB).
#   (c) Compile-time total budget: AOT `naive_sum_hbm_gib <= 85.0` GiB for the exact 40-hex `git_commit_sha`.
def validate_and_extract_r4_aot_hbm_result(
    aot_payload: Mapping[str, Any],
    expected_commit_sha: str,
    *,
    max_hbm_gib_ceiling: float = DEFAULT_R4_AOT_MAX_HBM_GIB,
    source_label: str = "<aot_json>",
) -> dict[str, Any]:
  """Validates A6 AOT JSON commit provenance and HBM gate status (fail-closed) and extracts A6 HBM telemetry."""
  if not isinstance(aot_payload, Mapping):
    raise ValueError(f"AOT summary payload from {source_label} must be a JSON object, got {type(aot_payload).__name__}")
  if not isinstance(expected_commit_sha, str) or not expected_commit_sha.strip():
    raise ValueError(
        f"expected_commit_sha must be a non-empty 7..40 hex commit SHA for {source_label}, got {expected_commit_sha!r}"
    )
  expected_sha = expected_commit_sha.strip().lower()
  if not _HEX_COMMIT_SHA_RE.match(expected_sha):
    raise ValueError(
        f"expected_commit_sha must be a 7..40 lowercase hex commit SHA for {source_label}, got {expected_commit_sha!r}"
    )

  if "results" in aot_payload:
    results_map = aot_payload["results"]
    if not isinstance(results_map, Mapping) or "A6" not in results_map:
      raise KeyError(f"AOT summary JSON {source_label} missing results['A6']")
    a6_entry = results_map["A6"]
    if not isinstance(a6_entry, Mapping):
      raise ValueError(f"AOT summary JSON {source_label} results['A6'] must be a mapping")
    if "passed" in aot_payload and aot_payload["passed"] is not True:
      raise ValueError(f"AOT summary JSON {source_label} top-level passed must be True, got {aot_payload['passed']!r}")
  else:
    a6_entry = aot_payload

  top_raw_sha = aot_payload["git_commit_sha"] if "git_commit_sha" in aot_payload else None
  a6_raw_sha = a6_entry["git_commit_sha"] if "git_commit_sha" in a6_entry else None
  if top_raw_sha is not None and a6_raw_sha is not None:
    if not isinstance(top_raw_sha, str) or not isinstance(a6_raw_sha, str):
      raise ValueError(f"AOT summary JSON {source_label} git_commit_sha must be a string")
    if top_raw_sha.strip().lower() != a6_raw_sha.strip().lower():
      raise ValueError(
          f"AOT summary JSON {source_label} has conflicting top-level git_commit_sha={top_raw_sha!r} "
          f"vs A6 git_commit_sha={a6_raw_sha!r}"
      )

  actual_raw_sha = a6_raw_sha if a6_raw_sha is not None else top_raw_sha
  if actual_raw_sha is None or not isinstance(actual_raw_sha, str) or not actual_raw_sha.strip():
    raise ValueError(
        f"AOT summary JSON {source_label} is missing required 'git_commit_sha' "
        f"(expected_commit_sha={expected_sha!r}, actual_commit_sha={actual_raw_sha!r})"
    )
  actual_sha = actual_raw_sha.strip().lower()
  if not _HEX_COMMIT_SHA_RE.match(actual_sha):
    raise ValueError(
        f"AOT summary JSON {source_label} has invalid 'git_commit_sha': "
        f"expected_commit_sha={expected_sha!r}, actual_commit_sha={actual_raw_sha!r}"
    )

  commit_matches = actual_sha == expected_sha or (len(expected_sha) < 40 and actual_sha.startswith(expected_sha))
  if not commit_matches:
    raise ValueError(
        f"AOT summary JSON {source_label} commit provenance mismatch: "
        f"expected_commit_sha={expected_sha!r}, actual_commit_sha={actual_sha!r}"
    )

  if "compile_succeeded" not in a6_entry or a6_entry["compile_succeeded"] is not True:
    actual_comp = a6_entry["compile_succeeded"] if "compile_succeeded" in a6_entry else None
    raise ValueError(f"AOT summary JSON {source_label} A6 compile_succeeded must be True, got {actual_comp!r}")
  if "passed" not in a6_entry or a6_entry["passed"] is not True:
    actual_passed = a6_entry["passed"] if "passed" in a6_entry else None
    raise ValueError(f"AOT summary JSON {source_label} A6 passed must be True, got {actual_passed!r}")

  max_hbm_gib = float(a6_entry["max_hbm_gib"]) if "max_hbm_gib" in a6_entry else float(max_hbm_gib_ceiling)
  if max_hbm_gib <= 0.0 or not math.isfinite(max_hbm_gib) or max_hbm_gib > float(max_hbm_gib_ceiling):
    raise ValueError(
        f"AOT summary JSON {source_label} A6 max_hbm_gib={max_hbm_gib} invalid or exceeds ceiling {max_hbm_gib_ceiling}"
    )

  if "naive_sum_hbm_gib" in a6_entry and a6_entry["naive_sum_hbm_gib"] is not None:
    naive_sum_val = float(a6_entry["naive_sum_hbm_gib"])
  elif "peak_hbm_gib" in a6_entry and a6_entry["peak_hbm_gib"] is not None:
    naive_sum_val = float(a6_entry["peak_hbm_gib"])
  else:
    raise KeyError(f"AOT summary JSON {source_label} missing 'naive_sum_hbm_gib' / 'peak_hbm_gib' in A6 entry")

  if "peak_hbm_gib" in a6_entry and a6_entry["peak_hbm_gib"] is not None:
    peak_val = float(a6_entry["peak_hbm_gib"])
  else:
    peak_val = naive_sum_val

  if naive_sum_val <= 0.0 or not math.isfinite(naive_sum_val):
    raise ValueError(f"Invalid A6 AOT naive_sum_hbm_gib={naive_sum_val} in {source_label}")
  if peak_val <= 0.0 or not math.isfinite(peak_val):
    raise ValueError(f"Invalid A6 AOT peak_hbm_gib={peak_val} in {source_label}")
  if naive_sum_val > max_hbm_gib:
    raise ValueError(
        f"AOT summary JSON {source_label} A6 naive_sum_hbm_gib={naive_sum_val:.6f} exceeds max_hbm_gib={max_hbm_gib:.6f}"
    )

  byte_keys = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")
  present_byte_keys = [k for k in byte_keys if k in a6_entry and a6_entry[k] is not None]
  has_non_aliased_gib = "non_aliased_io_hbm_gib" in a6_entry and a6_entry["non_aliased_io_hbm_gib"] is not None
  if present_byte_keys:
    if len(present_byte_keys) != len(byte_keys):
      missing_byte_keys = [k for k in byte_keys if k not in present_byte_keys]
      raise ValueError(f"AOT summary JSON {source_label} A6 has partial byte fields; missing {missing_byte_keys}")
    arg_bytes = float(a6_entry["argument_size_in_bytes"])
    out_bytes = float(a6_entry["output_size_in_bytes"])
    alias_bytes = float(a6_entry["alias_size_in_bytes"])
    if not (math.isfinite(arg_bytes) and math.isfinite(out_bytes) and math.isfinite(alias_bytes)):
      raise ValueError(f"AOT summary JSON {source_label} A6 has non-finite argument/output/alias byte sizes")
    byte_heap_gib = (arg_bytes + out_bytes - alias_bytes) / float(1024**3)
    if has_non_aliased_gib:
      reported_heap_gib = float(a6_entry["non_aliased_io_hbm_gib"])
      if not math.isfinite(reported_heap_gib) or abs(reported_heap_gib - byte_heap_gib) > 1e-4:
        raise ValueError(
            f"AOT summary JSON {source_label} A6 non_aliased_io_hbm_gib={reported_heap_gib} disagrees with "
            f"(argument_size_in_bytes + output_size_in_bytes - alias_size_in_bytes)/2^30={byte_heap_gib:.6f}"
        )
    aot_heap_hbm_gib = byte_heap_gib
  elif has_non_aliased_gib:
    aot_heap_hbm_gib = float(a6_entry["non_aliased_io_hbm_gib"])
  else:
    raise ValueError(
        f"AOT summary JSON {source_label} A6 missing required heap telemetry "
        "('argument_size_in_bytes'/'output_size_in_bytes'/'alias_size_in_bytes' or 'non_aliased_io_hbm_gib')"
    )

  if aot_heap_hbm_gib <= 0.0 or not math.isfinite(aot_heap_hbm_gib):
    raise ValueError(f"Invalid A6 AOT aot_heap_hbm_gib={aot_heap_hbm_gib} in {source_label}")

  return {
      "git_commit_sha": actual_sha,
      "expected_commit_sha": expected_sha,
      "compile_succeeded": True,
      "passed": True,
      "aot_heap_hbm_gib": aot_heap_hbm_gib,
      "naive_sum_hbm_gib": naive_sum_val,
      "peak_hbm_gib": peak_val,
      "aot_hbm_gb": aot_heap_hbm_gib,
      "max_hbm_gib": max_hbm_gib,
  }


def load_a6_aot_hbm_gb(
    aot_summary_json: str | os.PathLike[str],
    *,
    expected_commit_sha: str | None = None,
) -> float:
  """Loads and validates A6 like-for-like heap HBM (`aot_heap_hbm_gib`) from an A6 or `W1_ALL` summary JSON."""
  payload = json.loads(epath.Path(str(aot_summary_json)).read_text(encoding="utf-8"))
  resolved_expected_sha = _resolve_default_expected_aot_commit_sha(expected_commit_sha)
  validated = validate_and_extract_r4_aot_hbm_result(
      payload,
      resolved_expected_sha,
      source_label=str(aot_summary_json),
  )
  return float(validated["aot_hbm_gb"])


def _check_r4_norm_invariants(
    mm_norms: Mapping[int, Mapping[str, Any]],
    expected_num_layers: int,
    tt5a: Mapping[str, Any],
    tt6: Mapping[str, Any],
    *,
    expected_steps: int,
    failures: list[str],
) -> tuple[bool, bool]:
  """Checks TT5a(7) step-to-step parameter norm invariants across steps 0..expected_steps-1."""
  vit_0_hex = mm_norms[0]["vit_norm"]
  vit_0_bits = _hex_to_f32_bits(vit_0_hex)
  delim_0_hex = mm_norms[0]["delim_norm"]
  delim_0_bits = _hex_to_f32_bits(delim_0_hex)
  bvl_0_hex_list = mm_norms[0]["bias_vl_norm"]
  if not bvl_0_hex_list:
    failures.append("TT5a(7)/TT6: step 0 bias_vl_norm list is empty")
  bvl_0_bits = [_hex_to_f32_bits(h) for h in bvl_0_hex_list]

  if len(bvl_0_hex_list) != int(expected_num_layers):
    failures.append(
        f"TT5a(7): step 0 bias_vl_norm layer count {len(bvl_0_hex_list)} != "
        f"HF checkpoint layer count {expected_num_layers}"
    )

  check_vit = bool(tt5a["vit_norm_bit_identical"])
  check_bvl = bool(tt5a["bias_vl_norm_bit_identical"]) or bool(tt6["bias_vl_norm_bit_identical"])
  for k in range(1, expected_steps):
    if check_vit:
      vit_k_hex = mm_norms[k]["vit_norm"]
      if vit_k_hex != vit_0_hex or _hex_to_f32_bits(vit_k_hex) != vit_0_bits:
        failures.append(f"TT5a(7): step {k} vit_norm={vit_k_hex} != step 0 vit_norm={vit_0_hex}")
    if check_bvl:
      bvl_k_hex_list = mm_norms[k]["bias_vl_norm"]
      if len(bvl_k_hex_list) != len(bvl_0_hex_list):
        failures.append(f"TT5a(7)/TT6: step {k} bias_vl_norm length {len(bvl_k_hex_list)} != {len(bvl_0_hex_list)}")
        continue
      for layer_idx, (hk, h0, b0) in enumerate(zip(bvl_k_hex_list, bvl_0_hex_list, bvl_0_bits)):
        if hk != h0 or _hex_to_f32_bits(hk) != b0:
          failures.append(f"TT5a(7)/TT6: step {k} layer {layer_idx} bias_vl_norm={hk} != step 0 {h0}")

  last_s = expected_steps - 1
  align_0_hex = mm_norms[0]["aligner_norm"]
  align_last_hex = mm_norms[last_s]["aligner_norm"]
  align_last_f32 = float.fromhex(align_last_hex)
  aligner_changed = (
      expected_steps >= 2
      and not math.isnan(align_last_f32)
      and align_last_hex != align_0_hex
      and _hex_to_f32_bits(align_last_hex) != _hex_to_f32_bits(align_0_hex)
  )
  if bool(tt5a["aligner_norm_must_change"]) and expected_steps >= 2 and not aligner_changed:
    failures.append(f"TT5a(7): aligner_norm({last_s})={align_last_hex} did not change from step 0 ({align_0_hex})")

  delim_last_hex = mm_norms[last_s]["delim_norm"]
  delim_last_f32 = float.fromhex(delim_last_hex)
  delim_changed = (
      expected_steps >= 2
      and not math.isnan(delim_last_f32)
      and delim_last_hex != delim_0_hex
      and _hex_to_f32_bits(delim_last_hex) != delim_0_bits
  )
  if bool(tt5a.get("delim_norm_must_change", True)) and expected_steps >= 2 and not delim_changed:
    failures.append(
        f"TT5a(7): delim_norm did not change from step 0 ({delim_0_hex}) to last step {last_s} ({delim_last_hex})"
    )
  return aligner_changed, delim_changed


def _check_r4_tt6_bias_vl_bit_exact(
    bias_vl_npz: str | os.PathLike[str] | None,
    hf_tensors: Mapping[int, np.ndarray],
    *,
    expected_steps: int,
    expected_num_layers: int = 40,
    expected_shape: tuple[int, ...] = (384,),
    failures: list[str],
) -> tuple[bool, bool, list[dict[str, Any]]]:
  """Verifies bit-for-bit (`uint32`) equality of all 40 `bias_vl` tensors at step 0 and `expected_steps - 1`."""
  last_step = int(expected_steps) - 1
  expected_order = list(range(int(expected_num_layers)))
  if sorted(hf_tensors) != expected_order:
    failures.append(
        f"TT6: HF checkpoint bias_vl layer indices {sorted(hf_tensors)} != expected 0..{expected_num_layers - 1}"
    )
    return False, False, []
  if bias_vl_npz is None or not str(bias_vl_npz).strip():
    failures.append("TT6: missing required --bias_vl_npz path for bit-exact bias_vl verification")
    return False, False, []
  npz_path = epath.Path(str(bias_vl_npz))
  if not npz_path.exists():
    failures.append(f"TT6: bias_vl_npz file does not exist: {bias_vl_npz}")
    return False, False, []

  with npz_path.open("rb") as f, np.load(f, allow_pickle=False) as npz_data:
    file_keys = set(npz_data.files)
    meta_ok = True
    if "steps" not in file_keys:
      failures.append("TT6: bias_vl_npz missing 'steps' key")
      meta_ok = False
    else:
      steps_arr = npz_data["steps"]
      expected_steps_list = [0] if last_step == 0 else [0, last_step]
      if steps_arr.ndim != 1 or steps_arr.tolist() != expected_steps_list:
        failures.append(f"TT6: bias_vl_npz steps={steps_arr.tolist()} != {expected_steps_list}")
        meta_ok = False

    if "layer_order" not in file_keys:
      failures.append("TT6: bias_vl_npz missing 'layer_order' key")
      meta_ok = False
    else:
      order_arr = npz_data["layer_order"]
      if order_arr.ndim != 1 or order_arr.tolist() != expected_order:
        failures.append(f"TT6: bias_vl_npz layer_order={order_arr.tolist()} != {expected_order}")
        meta_ok = False

    req_step0_keys = {f"step_0_layer_{i}" for i in expected_order}
    req_last_keys = {f"step_{last_step}_layer_{i}" for i in expected_order}
    allowed_keys = {"steps", "layer_order"} | req_step0_keys | req_last_keys
    missing_step0 = sorted(req_step0_keys - file_keys)
    missing_last = sorted(req_last_keys - file_keys)
    extra_keys = sorted(file_keys - allowed_keys)
    step0_all_ok = meta_ok and not missing_step0 and not extra_keys
    last_all_ok = meta_ok and not missing_last and not extra_keys
    if missing_step0:
      failures.append(f"TT6: bias_vl_npz missing step_0 keys: {missing_step0}")
    if missing_last:
      failures.append(f"TT6: bias_vl_npz missing step_{last_step} keys: {missing_last}")
    if extra_keys:
      failures.append(f"TT6: bias_vl_npz contains unexpected extra keys: {extra_keys}")

    table: list[dict[str, Any]] = []
    for layer_idx in expected_order:
      hf_arr = hf_tensors[layer_idx]
      if hf_arr.dtype != np.float32 or hf_arr.shape != tuple(expected_shape):
        failures.append(
            f"TT6: HF layer {layer_idx} bias_vl has dtype={hf_arr.dtype}, shape={hf_arr.shape}, "
            f"expected float32{tuple(expected_shape)}"
        )
        return False, False, table
      hf_u32 = np.ascontiguousarray(hf_arr, dtype=np.float32).view(np.uint32)
      hf_sha16 = hashlib.sha256(hf_u32.tobytes()).hexdigest()[:16]

      k0 = f"step_0_layer_{layer_idx}"
      kl = f"step_{last_step}_layer_{layer_idx}"
      if k0 not in file_keys or kl not in file_keys:
        step0_all_ok = False
        last_all_ok = False
        continue

      arr0 = npz_data[k0]
      arrl = npz_data[kl]
      s0_valid = arr0.dtype == np.float32 and arr0.shape == tuple(expected_shape)
      sl_valid = arrl.dtype == np.float32 and arrl.shape == tuple(expected_shape)
      if not s0_valid:
        failures.append(
            f"TT6: {k0} has dtype={arr0.dtype}, shape={arr0.shape}, "
            f"expected dtype=float32, shape={tuple(expected_shape)}"
        )
        step0_all_ok = False
      if not sl_valid:
        failures.append(
            f"TT6: {kl} has dtype={arrl.dtype}, shape={arrl.shape}, "
            f"expected dtype=float32, shape={tuple(expected_shape)}"
        )
        last_all_ok = False

      if s0_valid:
        u0 = np.ascontiguousarray(arr0).view(np.uint32)
        s0_sha16 = hashlib.sha256(u0.tobytes()).hexdigest()[:16]
        s0_eq = bool(np.array_equal(u0, hf_u32))
        diff0 = u0 != hf_u32
        s0_diff_elems = int(np.count_nonzero(diff0))
        s0_max_ulp = int(np.max(np.abs(u0.astype(np.int64) - hf_u32.astype(np.int64)))) if s0_diff_elems > 0 else 0
        if not s0_eq:
          step0_all_ok = False
          failures.append(
              f"TT6: step 0 layer {layer_idx} bias_vl is not bit-exact with HF "
              f"(diff_elements={s0_diff_elems}, max_ulp_diff={s0_max_ulp})"
          )
      else:
        s0_sha16 = ""
        s0_eq = False
        s0_diff_elems = int(expected_shape[0])
        s0_max_ulp = -1

      if sl_valid:
        ul = np.ascontiguousarray(arrl).view(np.uint32)
        sl_sha16 = hashlib.sha256(ul.tobytes()).hexdigest()[:16]
        sl_eq = bool(np.array_equal(ul, hf_u32))
        diffl = ul != hf_u32
        sl_diff_elems = int(np.count_nonzero(diffl))
        sl_max_ulp = int(np.max(np.abs(ul.astype(np.int64) - hf_u32.astype(np.int64)))) if sl_diff_elems > 0 else 0
        if not sl_eq:
          last_all_ok = False
          failures.append(
              f"TT6: step {last_step} layer {layer_idx} bias_vl is not bit-exact with HF "
              f"(diff_elements={sl_diff_elems}, max_ulp_diff={sl_max_ulp})"
          )
      else:
        sl_sha16 = ""
        sl_eq = False
        sl_diff_elems = int(expected_shape[0])
        sl_max_ulp = -1

      table.append(
          {
              "layer_idx": int(layer_idx),
              "shape": list(arr0.shape) if not s0_valid else list(arrl.shape),
              "dtype": str(arr0.dtype) if not s0_valid else str(arrl.dtype),
              "step0_shape": list(arr0.shape),
              "last_step_shape": list(arrl.shape),
              "step0_dtype": str(arr0.dtype),
              "last_step_dtype": str(arrl.dtype),
              "hf_sha256_16": hf_sha16,
              "step0_sha256_16": s0_sha16,
              "last_step_sha256_16": sl_sha16,
              "step0_uint32_equal": s0_eq,
              "last_step_uint32_equal": sl_eq,
              "step0_diff_elements": s0_diff_elems,
              "last_step_diff_elements": sl_diff_elems,
              "step0_max_ulp_diff": s0_max_ulp,
              "last_step_max_ulp_diff": sl_max_ulp,
          }
      )

    if len(table) != int(expected_num_layers):
      step0_all_ok = False
      last_all_ok = False
    return step0_all_ok, last_all_ok, table


# `_check_r4_tt6_hbm` enforces Gate (a) on step-phase `bytes_in_use_gib` (`<= 1.10 * aot_heap_hbm_gib`) and
# Gate (b) on init-phase `peak_bytes_in_use` (`peak_hbm_gb <= 85.0` GiB and `< 94.7` GB), while
# `validate_and_extract_r4_aot_hbm_result` enforces Gate (c) (`naive_sum_hbm_gib <= 85.0` GiB).
def _check_r4_tt6_hbm(
    mm_norms: Mapping[int, Mapping[str, Any]],
    expected_steps: int,
    *,
    peak_hbm_gb: float | None,
    aot_hbm_gb: float,
    tt6: Mapping[str, Any],
    failures: list[str],
    init_peak_max_hbm_gib: float = DEFAULT_R4_INIT_PEAK_MAX_HBM_GIB,
) -> tuple[float | None, float | None]:
  """Validates Gate (a) step-phase `bytes_in_use_gib` vs AOT heap and Gate (b) init-phase `peak_hbm_gb` ceiling."""
  peak_lt = float(tt6["peak_hbm_gb_lt"])
  max_aot_ratio = float(tt6["peak_hbm_max_aot_ratio"])
  max_allowed_by_aot = max_aot_ratio * aot_hbm_gb

  missing_in_use_steps: list[int] = []
  invalid_in_use = False
  step_in_use_vals: list[float] = []
  for s in range(int(expected_steps)):
    if s not in mm_norms or "bytes_in_use_gib" not in mm_norms[s] or mm_norms[s]["bytes_in_use_gib"] is None:
      missing_in_use_steps.append(s)
      continue
    val = float(mm_norms[s]["bytes_in_use_gib"])
    if val <= 0.0 or not math.isfinite(val):
      failures.append(f"TT6(a): step {s} invalid non-positive or non-finite bytes_in_use_gib={val}")
      invalid_in_use = True
    else:
      step_in_use_vals.append(val)

  step_peak_bytes_in_use_gib: float | None = None
  step_peak_ratio: float | None = None
  if missing_in_use_steps:
    failures.append(
        f"TT6(a): missing bytes_in_use_gib telemetry in DS41_MM_NORMS log lines for steps: {missing_in_use_steps}"
    )
  elif not invalid_in_use and step_in_use_vals:
    step_peak_bytes_in_use_gib = max(step_in_use_vals)
    if not step_peak_bytes_in_use_gib < peak_lt:
      failures.append(f"TT6(a): step_peak_bytes_in_use_gib={step_peak_bytes_in_use_gib} is not < {peak_lt} GB")
    if step_peak_bytes_in_use_gib > max_allowed_by_aot:
      failures.append(
          f"TT6(a): step_peak_bytes_in_use_gib={step_peak_bytes_in_use_gib} > "
          f"{max_aot_ratio} * aot_hbm_gb ({max_allowed_by_aot:.6f} GB)"
      )
    step_peak_ratio = step_peak_bytes_in_use_gib / aot_hbm_gb

  if peak_hbm_gb is None:
    failures.append("TT6(b): missing peak_hbm_gb telemetry in DS41_MM_NORMS log lines")
  elif peak_hbm_gb <= 0.0 or not math.isfinite(peak_hbm_gb):
    failures.append(f"TT6(b): invalid non-positive or non-finite peak_hbm_gb={peak_hbm_gb}")
  else:
    if not peak_hbm_gb < peak_lt:
      failures.append(f"TT6(b): peak_hbm_gb={peak_hbm_gb} is not < {peak_lt} GB")
    if peak_hbm_gb > float(init_peak_max_hbm_gib):
      failures.append(f"TT6(b): init_peak_bytes_in_use_gib={peak_hbm_gb} > {float(init_peak_max_hbm_gib)} GiB")

  return step_peak_bytes_in_use_gib, step_peak_ratio


def verify_r4_tt5a_tt6(
    log_path: str | os.PathLike[str],
    metrics_path: str | os.PathLike[str],
    thresholds_path: str | os.PathLike[str],
    *,
    hf_checkpoint_dir: str | os.PathLike[str],
    aot_hbm_gb: float | None = None,
    aot_summary_json: str | os.PathLike[str] | None = None,
    expected_aot_commit_sha: str | None = None,
    expected_steps: int = 20,
    bias_vl_npz: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
  """Verifies all TT5a and TT6 conditions on an R4 training stdout log file, `metrics_file`, and `bias_vl_npz`."""
  has_aot_json = aot_summary_json is not None and str(aot_summary_json).strip() != ""
  has_aot_val = aot_hbm_gb is not None
  if has_aot_json == has_aot_val:
    raise ValueError("verify_r4_tt5a_tt6 requires exactly one of aot_hbm_gb or aot_summary_json")
  aot_git_commit_sha: str | None = None
  resolved_expected_aot_sha: str | None = None
  aot_naive_sum_hbm_gib: float | None = None
  aot_max_hbm_gib = float(DEFAULT_R4_AOT_MAX_HBM_GIB)
  if has_aot_json:
    aot_payload = json.loads(epath.Path(str(aot_summary_json)).read_text(encoding="utf-8"))
    resolved_expected_aot_sha = _resolve_default_expected_aot_commit_sha(expected_aot_commit_sha)
    validated_aot = validate_and_extract_r4_aot_hbm_result(
        aot_payload,
        resolved_expected_aot_sha,
        source_label=str(aot_summary_json),
    )
    aot_val = float(validated_aot["aot_hbm_gb"])
    aot_naive_sum_hbm_gib = float(validated_aot["naive_sum_hbm_gib"])
    aot_max_hbm_gib = float(validated_aot["max_hbm_gib"])
    aot_git_commit_sha = str(validated_aot["git_commit_sha"])
    resolved_expected_aot_sha = str(validated_aot["expected_commit_sha"])
  else:
    assert aot_hbm_gb is not None
    aot_val = float(aot_hbm_gb)
    if aot_val <= 0.0 or not math.isfinite(aot_val):
      raise ValueError(f"aot_hbm_gb must be positive and finite, got {aot_hbm_gb!r}")

  thresholds_dict = json.loads(epath.Path(str(thresholds_path)).read_text(encoding="utf-8"))
  tt5a = thresholds_dict["TT5a"]
  tt6 = thresholds_dict["TT6"]
  parsed_log = parse_r4_log(log_path)
  completed_steps = parsed_log["completed_steps"]
  mm_norms = parsed_log["mm_norms"]
  peak_hbm_gb = parsed_log["peak_hbm_gb"]
  metrics_by_step = parse_r4_metrics_file(metrics_path)

  failures: list[str] = []
  try:
    hf_bias_vl_tensors = load_hf_bias_vl_tensors(hf_checkpoint_dir, expected_num_layers=None, expected_shape=None)
  except (KeyError, ValueError, OSError) as exc:
    failures.append(f"TT6: failed to load HF bias_vl tensors from {hf_checkpoint_dir}: {exc}")
    hf_bias_vl_tensors = {}

  required_step_ids = list(range(int(expected_steps)))
  missing_log_steps = [s for s in required_step_ids if s not in completed_steps]
  missing_norm_steps = [s for s in required_step_ids if s not in mm_norms]
  missing_metric_steps = [s for s in required_step_ids if s not in metrics_by_step]
  if missing_log_steps:
    failures.append(f"Missing completed step lines for steps: {missing_log_steps}")
  if missing_norm_steps:
    failures.append(f"Missing DS41_MM_NORMS lines for steps: {missing_norm_steps}")
  if missing_metric_steps:
    failures.append(f"Missing metrics_file JSONL entries for steps: {missing_metric_steps}")
  if missing_log_steps or missing_norm_steps or missing_metric_steps:
    if hf_bias_vl_tensors:
      step0_bit_exact, last_step_bit_exact, bit_equality_table = _check_r4_tt6_bias_vl_bit_exact(
          bias_vl_npz,
          hf_bias_vl_tensors,
          expected_steps=expected_steps,
          expected_num_layers=len(hf_bias_vl_tensors),
          expected_shape=hf_bias_vl_tensors[0].shape,
          failures=failures,
      )
    else:
      step0_bit_exact = False
      last_step_bit_exact = False
      bit_equality_table = []
    return {
        "passed": False,
        "failures": failures,
        "steps_checked": int(expected_steps),
        "step_peak_bytes_in_use_gib": parsed_log["step_peak_bytes_in_use_gib"],
        "aot_heap_hbm_gib": aot_val,
        "step_peak_to_aot_heap_ratio": None,
        "step_peak_max_aot_ratio": float(tt6["peak_hbm_max_aot_ratio"]),
        "init_peak_bytes_in_use_gib": peak_hbm_gb,
        "init_peak_max_hbm_gib": float(DEFAULT_R4_INIT_PEAK_MAX_HBM_GIB),
        "aot_naive_sum_hbm_gib": aot_naive_sum_hbm_gib,
        "aot_naive_sum_max_hbm_gib": aot_max_hbm_gib,
        "peak_hbm_gb": peak_hbm_gb,
        "aot_hbm_gb": aot_val,
        "aot_git_commit_sha": aot_git_commit_sha,
        "expected_aot_commit_sha": resolved_expected_aot_sha,
        "peak_to_aot_ratio": None,
        "aligner_norm_changed": False,
        "delim_norm_changed": False,
        "bias_vl_step0_bit_exact_hf": step0_bit_exact,
        "bias_vl_last_step_bit_exact_hf": last_step_bit_exact,
        "bias_vl_bit_equality_table": bit_equality_table,
    }

  loss_0 = completed_steps[0]["loss"]
  loss_0_min = float(tt5a["loss_0_min"])
  loss_0_max = float(tt5a["loss_0_max"])
  if not loss_0_min <= loss_0 <= loss_0_max:
    failures.append(f"TT5a(1): loss(0)={loss_0} outside [{loss_0_min}, {loss_0_max}]")

  if expected_steps > 4:
    loss_4 = completed_steps[4]["loss"]
    max_loss_4 = loss_0 + float(tt5a["loss_4_max_delta_from_0"])
    if loss_4 > max_loss_4:
      failures.append(f"TT5a(2): loss(4)={loss_4} > loss(0) + {tt5a['loss_4_max_delta_from_0']} ({max_loss_4})")

  last_s = expected_steps - 1
  loss_last = completed_steps[last_s]["loss"]
  if expected_steps >= 20:
    max_loss_19 = loss_0 + float(tt5a["loss_19_max_delta_from_0"])
    if loss_last > max_loss_19:
      failures.append(
          f"TT5a(3): loss({last_s})={loss_last} > loss(0) + {tt5a['loss_19_max_delta_from_0']} ({max_loss_19})"
      )
    if bool(tt5a["require_loss_19_lt_loss_4"]) and loss_last >= completed_steps[4]["loss"]:
      failures.append(f"TT5a(3): loss({last_s})={loss_last} is not < loss(4)={completed_steps[4]['loss']}")
  elif expected_steps >= 2 and loss_last >= loss_0:
    failures.append(f"TT5a(3): loss({last_s})={loss_last} is not < loss(0)={loss_0}")

  max_inc = float(tt5a["step_to_step_max_increase"])
  for k in range(expected_steps - 1):
    lk = completed_steps[k]["loss"]
    lk1 = completed_steps[k + 1]["loss"]
    if lk1 > lk + max_inc:
      failures.append(f"TT5a(4): loss({k + 1})={lk1} > loss({k})={lk} + {max_inc}")

  lb_min = float(tt5a["moe_lb_loss_min"])
  lb_max = float(tt5a["moe_lb_loss_max"])
  for k in required_step_ids:
    lb_k = completed_steps[k]["moe_lb_loss"]
    if not lb_min <= lb_k <= lb_max:
      failures.append(f"TT5a(5): step {k} moe_lb_loss={lb_k} outside [{lb_min}, {lb_max}]")

  if "learning/raw_grad_norm" not in metrics_by_step[0]:
    failures.append("TT5a(6): step 0 metrics_file entry missing 'learning/raw_grad_norm'")
  else:
    rgn_0 = float(metrics_by_step[0]["learning/raw_grad_norm"])
    max_ratio = float(tt5a["raw_grad_norm_max_ratio_to_step0"])
    if rgn_0 <= 0.0 or not math.isfinite(rgn_0):
      failures.append(f"TT5a(6): step 0 raw_grad_norm={rgn_0} is not positive finite")
    for k in required_step_ids:
      if "learning/raw_grad_norm" not in metrics_by_step[k]:
        failures.append(f"TT5a(6): step {k} metrics_file entry missing 'learning/raw_grad_norm'")
        continue
      rgn_k = float(metrics_by_step[k]["learning/raw_grad_norm"])
      if rgn_k <= 0.0 or not math.isfinite(rgn_k):
        failures.append(f"TT5a(6): step {k} raw_grad_norm={rgn_k} is not positive finite")
      elif rgn_0 > 0.0 and rgn_k > max_ratio * rgn_0:
        failures.append(f"TT5a(6): step {k} raw_grad_norm={rgn_k} > {max_ratio} * raw_grad_norm(0)={rgn_0}")

  expected_num_layers = len(hf_bias_vl_tensors) if hf_bias_vl_tensors else 40
  aligner_changed, delim_changed = _check_r4_norm_invariants(
      mm_norms, expected_num_layers, tt5a, tt6, expected_steps=expected_steps, failures=failures
  )
  if hf_bias_vl_tensors:
    step0_bit_exact, last_step_bit_exact, bit_equality_table = _check_r4_tt6_bias_vl_bit_exact(
        bias_vl_npz,
        hf_bias_vl_tensors,
        expected_steps=expected_steps,
        expected_num_layers=len(hf_bias_vl_tensors),
        expected_shape=hf_bias_vl_tensors[0].shape,
        failures=failures,
    )
  else:
    step0_bit_exact = False
    last_step_bit_exact = False
    bit_equality_table = []
  step_peak_bytes_in_use_gib, peak_to_aot_ratio = _check_r4_tt6_hbm(
      mm_norms,
      expected_steps,
      peak_hbm_gb=peak_hbm_gb,
      aot_hbm_gb=aot_val,
      tt6=tt6,
      failures=failures,
  )

  return {
      "passed": len(failures) == 0,
      "failures": failures,
      "steps_checked": int(expected_steps),
      "step_peak_bytes_in_use_gib": step_peak_bytes_in_use_gib,
      "aot_heap_hbm_gib": aot_val,
      "step_peak_to_aot_heap_ratio": peak_to_aot_ratio,
      "step_peak_max_aot_ratio": float(tt6["peak_hbm_max_aot_ratio"]),
      "init_peak_bytes_in_use_gib": peak_hbm_gb,
      "init_peak_max_hbm_gib": float(DEFAULT_R4_INIT_PEAK_MAX_HBM_GIB),
      "aot_naive_sum_hbm_gib": aot_naive_sum_hbm_gib,
      "aot_naive_sum_max_hbm_gib": aot_max_hbm_gib,
      "peak_hbm_gb": peak_hbm_gb,
      "aot_hbm_gb": aot_val,
      "aot_git_commit_sha": aot_git_commit_sha,
      "expected_aot_commit_sha": resolved_expected_aot_sha,
      "peak_to_aot_ratio": peak_to_aot_ratio,
      "aligner_norm_changed": bool(aligner_changed),
      "delim_norm_changed": bool(delim_changed),
      "bias_vl_step0_bit_exact_hf": step0_bit_exact,
      "bias_vl_last_step_bit_exact_hf": last_step_bit_exact,
      "bias_vl_bit_equality_table": bit_equality_table,
  }


verify_r4_gates = verify_r4_tt5a_tt6

SUMMARY_LOG_TAGS: tuple[str, ...] = (
    "DS41_FORWARD_CHECKER_SUMMARY:",
    "MASKED_PREFILL_SUMMARY:",
    "DS41_MM_PREFILL_SUMMARY:",
    "DS41_R4_GATES_RESULT:",
    "DS41_PIN_VERIFY_RESULT:",
    "DS41_AUDIT_RESULT:",
    "DS41_MM_AUDIT_RESULT:",
)


def _extract_json_from_line(line: str, tag: str) -> dict[str, Any] | None:
  """Extracts JSON dict payload from a line if tag matches."""
  if tag not in line:
    return None
  idx = line.find(tag)
  payload = line[idx + len(tag) :].strip()
  if not payload:
    return None
  try:
    data = json.loads(payload)
    if isinstance(data, dict):
      return data
  except json.JSONDecodeError:
    return None
  return None


def extract_summary_from_log(
    log_path: str | os.PathLike[str],
    *,
    preferred_tags: Sequence[str] | None = None,
) -> dict[str, Any]:
  """Extracts the most recent summary JSON payload from a log file."""
  path = Path(log_path)
  if not path.exists():
    raise FileNotFoundError(f"Log file not found: {log_path}")
  content = path.read_text(encoding="utf-8")
  lines = content.splitlines()
  tags_to_check = tuple(preferred_tags) if preferred_tags else SUMMARY_LOG_TAGS

  for line in reversed(lines):
    for tag in tags_to_check:
      data = _extract_json_from_line(line, tag)
      if data is not None:
        return data
  raise ValueError(f"No valid summary JSON line matching {tags_to_check} found in log file {log_path}")


def _recover_from_gcs(summary_p: Path, fname: str, gcs_prefix: str) -> dict[str, Any] | None:
  """Recovers summary artifact from remote GCS prefix via epath or gcloud storage."""
  clean_prefix = str(gcs_prefix).rstrip("/")
  candidate_uris = [f"{clean_prefix}/{fname}"]
  try:
    prefix_ep = epath.Path(clean_prefix)
    if prefix_ep.exists():
      candidate_uris.extend(match.as_posix() for match in prefix_ep.glob(f"worker_*_{fname}"))
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  for uri in candidate_uris:
    try:
      remote_ep = epath.Path(uri)
      if remote_ep.exists():
        parsed = json.loads(remote_ep.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
          return parsed
    except Exception:  # pylint: disable=broad-exception-caught
      continue

  if not shutil.which("gcloud"):
    return None

  summary_p.parent.mkdir(parents=True, exist_ok=True)
  for uri in candidate_uris:
    try:
      proc = subprocess.run(
          ["gcloud", "storage", "cp", uri, str(summary_p)],
          check=False,
          capture_output=True,
          text=True,
      )
      if proc.returncode == 0 and summary_p.exists():
        parsed = json.loads(summary_p.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
          return parsed
    except Exception:  # pylint: disable=broad-exception-caught
      continue
  return None


def _recover_from_log_fallback(
    log_path: str | os.PathLike[str],
    fname: str,
    allowed_log_tags: Sequence[str] | None,
) -> dict[str, Any] | None:
  """Extracts summary dict payload from a log file."""
  log_p = Path(log_path)
  if not log_p.exists():
    return None
  try:
    preferred = allowed_log_tags
    if not preferred:
      tag_map = {
          "r1_pin_summary.json": ("DS41_PIN_VERIFY_RESULT:",),
          "r2_pin_summary.json": ("DS41_PIN_VERIFY_RESULT:",),
          "r3_pin_summary.json": ("DS41_PIN_VERIFY_RESULT:",),
          "r1_logit_summary.json": ("DS41_FORWARD_CHECKER_SUMMARY:",),
          "r1_summary.json": ("DS41_FORWARD_CHECKER_SUMMARY:",),
          "r2_logit_summary.json": ("DS41_FORWARD_CHECKER_SUMMARY:",),
          "r2_summary.json": ("DS41_FORWARD_CHECKER_SUMMARY:",),
          "r3_prefill_summary.json": ("MASKED_PREFILL_SUMMARY:", "DS41_MM_PREFILL_SUMMARY:"),
          "r4_gates_summary.json": ("DS41_R4_GATES_RESULT:",),
          "r1_audit_summary.json": ("DS41_MM_AUDIT_RESULT:", "DS41_AUDIT_RESULT:"),
      }
      preferred = tag_map.get(fname, SUMMARY_LOG_TAGS)
    return extract_summary_from_log(log_p, preferred_tags=preferred)
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def verify_stage_summary_gate(
    summary_path: str | os.PathLike[str],
    *,
    log_path: str | os.PathLike[str] | None = None,
    gcs_prefix: str | None = None,
    allowed_log_tags: Sequence[str] | None = None,
) -> dict[str, Any]:
  """Enforces the hard stage completion gate for a stage summary JSON artifact.

  Verifies that summary_path exists and is valid JSON where `passed is True`
  (or `all_passed is True`). Recovers missing artifacts via GCS pull or log extraction fallback.
  """
  summary_p = Path(summary_path)
  fname = summary_p.name
  data: dict[str, Any] | None = None
  source: str | None = None

  # 1. Primary path: local file exists
  if summary_p.exists():
    try:
      raw = summary_p.read_text(encoding="utf-8")
      parsed = json.loads(raw)
      if isinstance(parsed, dict):
        data = parsed
        source = data.get("source", "local_process0")
    except Exception:  # pylint: disable=broad-exception-caught
      data = None

  # 2. Remote GCS recovery if not found locally
  if data is None and gcs_prefix and str(gcs_prefix).strip():
    data = _recover_from_gcs(summary_p, fname, str(gcs_prefix))
    if data is not None:
      source = "process0_worker_gcs"

  # 3. Fallback path: extract from log
  if data is None and log_path and str(log_path).strip():
    data = _recover_from_log_fallback(log_path, fname, allowed_log_tags)
    if data is not None:
      source = "log_extracted"

  if data is None:
    raise FileNotFoundError(
        f"Stage summary gate failed: summary file {summary_path} does not exist and could not be recovered from GCS or logs."
    )

  data["source"] = source
  summary_p.parent.mkdir(parents=True, exist_ok=True)
  summary_p.write_text(json.dumps(data, indent=2), encoding="utf-8")

  # Materialize embedded r1_audit_summary.json if present
  if fname in ("r1_logit_summary.json", "r1_summary.json"):
    audit_summary_path = summary_p.parent / "r1_audit_summary.json"
    if not audit_summary_path.exists() and "audit_result" in data and isinstance(data["audit_result"], dict):
      audit_data = dict(data["audit_result"])
      audit_data["source"] = source
      audit_summary_path.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")

  # Hard gate assertion: passed == True or all_passed == True
  is_passed = (data.get("passed") is True) or (data.get("all_passed") is True)
  if not is_passed:
    raise AssertionError(
        f"Stage summary gate verification failed for {summary_path}: "
        f"passed={data.get('passed')!r}, all_passed={data.get('all_passed')!r}, source={source!r}"
    )

  return data


def _cli_main(argv: list[str] | None = None) -> dict[str, Any]:
  """CLI entrypoint for R1/R2 weight audit + Orbax export and R4 TT5a/TT6 gate verification."""
  parser = argparse.ArgumentParser(description="DeepSeek-V4.1 multimodal audit and R4 gate verifier.")
  parser.add_argument(
      "--mode",
      type=str,
      default="audit",
      choices=("audit", "verify_r4_gates", "verify_pin", "run_r4_train", "verify_stage_summary_gate"),
  )
  parser.add_argument("--hf_checkpoint_dir", "--hf_ckpt_path", dest="hf_checkpoint_dir", type=str, default="")
  parser.add_argument("--export_orbax_dir", type=str, default="")
  parser.add_argument("--pin_json_path", "--pin-path", "--pin_path", dest="pin_json_path", type=str, default="")
  parser.add_argument(
      "--manifest_json_path",
      "--manifest-json-path",
      dest="manifest_json_path",
      type=str,
      default="",
  )
  parser.add_argument("--golden_dir", "--golden-dir", dest="golden_dir", type=str, default="")
  parser.add_argument("--log_file", type=str, default="")
  parser.add_argument("--metrics_file", type=str, default="")
  parser.add_argument("--thresholds_json", type=str, default="")
  parser.add_argument("--aot_summary_json", type=str, default="")
  parser.add_argument(
      "--expected_aot_commit_sha",
      "--expected-aot-commit-sha",
      dest="expected_aot_commit_sha",
      type=str,
      default="",
  )
  parser.add_argument("--aot_hbm_gb", type=float, default=None)
  parser.add_argument("--expected_steps", type=int, default=20)
  parser.add_argument("--bias_vl_npz", type=str, default="")
  parser.add_argument("--run_r4_train_dump_npz", type=str, default="")
  parser.add_argument("--summary_json", type=str, default="")
  parser.add_argument("--gcs_prefix", type=str, default="")

  args, remaining_args = parser.parse_known_args(argv)

  if args.mode == "verify_stage_summary_gate":
    if not args.summary_json:
      raise ValueError("--mode=verify_stage_summary_gate requires --summary_json")
    result = verify_stage_summary_gate(
        summary_path=args.summary_json,
        log_path=args.log_file if args.log_file else None,
        gcs_prefix=args.gcs_prefix if args.gcs_prefix else None,
    )
    print(f"STAGE_SUMMARY_GATE_PASSED: {args.summary_json} (source={result.get('source')})", flush=True)
    return result

  if args.mode == "run_r4_train" or args.run_r4_train_dump_npz:
    npz_dest = (
        args.bias_vl_npz
        or args.run_r4_train_dump_npz
        or (os.environ["DS41_DUMP_BIAS_VL_NPZ"] if "DS41_DUMP_BIAS_VL_NPZ" in os.environ else "")
    )
    if not npz_dest:
      raise ValueError("--mode=run_r4_train requires --bias_vl_npz, --run_r4_train_dump_npz, or DS41_DUMP_BIAS_VL_NPZ")
    if args.aot_summary_json:
      cli_expected_sha = args.expected_aot_commit_sha if args.expected_aot_commit_sha else None
      load_a6_aot_hbm_gb(args.aot_summary_json, expected_commit_sha=cli_expected_sha)
    run_train_with_bias_vl_dump(remaining_args, npz_dest)
    return {"passed": True, "bias_vl_npz": str(npz_dest)}

  resolved_manifest_path = args.manifest_json_path
  if not resolved_manifest_path and args.golden_dir:
    resolved_manifest_path = str(Path(args.golden_dir) / "manifest.json")

  if args.mode == "verify_r4_gates":
    if not args.log_file or not args.metrics_file or not args.thresholds_json or not args.hf_checkpoint_dir:
      raise ValueError(
          "--mode=verify_r4_gates requires --log_file, --metrics_file, --thresholds_json, and --hf_checkpoint_dir"
      )
    has_aot_json = bool(args.aot_summary_json)
    has_aot_val = args.aot_hbm_gb is not None
    if has_aot_json == has_aot_val:
      raise ValueError("--mode=verify_r4_gates requires exactly one of --aot_summary_json or --aot_hbm_gb")
    cli_expected_sha = args.expected_aot_commit_sha if args.expected_aot_commit_sha else None
    npz_arg = (
        args.bias_vl_npz
        if args.bias_vl_npz
        else (os.environ["DS41_DUMP_BIAS_VL_NPZ"] if "DS41_DUMP_BIAS_VL_NPZ" in os.environ else None)
    )
    result = verify_r4_tt5a_tt6(
        log_path=args.log_file,
        metrics_path=args.metrics_file,
        thresholds_path=args.thresholds_json,
        hf_checkpoint_dir=args.hf_checkpoint_dir,
        aot_hbm_gb=None if has_aot_json else float(args.aot_hbm_gb),
        aot_summary_json=args.aot_summary_json if has_aot_json else None,
        expected_aot_commit_sha=cli_expected_sha,
        expected_steps=args.expected_steps,
        bias_vl_npz=npz_arg,
    )
    if args.summary_json:
      summary_path = epath.Path(args.summary_json)
      summary_path.parent.mkdir(parents=True, exist_ok=True)
      summary_path.write_text(json.dumps(result, indent=2))
    print(f"DS41_R4_GATES_RESULT: {json.dumps(result)}", flush=True)
    if not result["passed"]:
      raise AssertionError(f"R4 TT5a/TT6 gate verification failed: {result['failures']}")
    return result

  if args.mode == "verify_pin":
    if not args.pin_json_path:
      raise ValueError("--mode=verify_pin requires --pin_json_path")
    manifest_arg = resolved_manifest_path if resolved_manifest_path else None
    result = verify_golden_manifest_pin(args.pin_json_path, manifest_json_path=manifest_arg)
    if args.summary_json:
      summary_path = epath.Path(args.summary_json)
      summary_path.parent.mkdir(parents=True, exist_ok=True)
      summary_path.write_text(json.dumps(result, indent=2))
    print(f"DS41_PIN_VERIFY_RESULT: {json.dumps(result)}", flush=True)
    if not result["passed"]:
      sys.stderr.write(f"Golden manifest pin verification failed: {result['failures']}\n")
      sys.exit(EXIT_DETERMINISTIC_GATE_FAIL)
    return result

  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel

  if args.pin_json_path:
    manifest_arg = resolved_manifest_path if resolved_manifest_path else None
    pin_res = verify_golden_manifest_pin(args.pin_json_path, manifest_json_path=manifest_arg)
    if not pin_res["passed"]:
      sys.stderr.write(f"Golden manifest pin verification failed before audit: {pin_res['failures']}\n")
      sys.exit(EXIT_DETERMINISTIC_GATE_FAIL)

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
    raise AssertionError(f"Model weight audit against HF failed: {result['failures']}")
  return result


if __name__ == "__main__":
  _cli_main()
