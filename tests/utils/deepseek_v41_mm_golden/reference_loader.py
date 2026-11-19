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

"""Loads the DeepSeek-V4.1-Flash multimodal PyTorch reference from HF safetensors shards.

Supports both eager bfloat16 weight materialization (`lazy_dequant=False`) and
compressed FP8/FP4 residency with on-the-fly per-module dequantization
(`lazy_dequant=True`, deviation R10) that produces bit-identical output logits
while keeping 749B resident weight memory near ~0.51 TB instead of ~1.50 TB.
"""

from collections.abc import Iterator
import contextlib
import dataclasses
import glob
import itertools
import json
import os
import struct
from typing import Any
from unittest import mock

import numpy as np
from safetensors import safe_open
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from tests.utils.deepseek_v41_reference import image_processor
from tests.utils.deepseek_v41_reference import model as ref_model
from tests.utils.deepseek_v41_reference import multimodal
from tests.utils.deepseek_v41_reference import runner

_MULTIMODAL_PREFIXES = ("vision.", "aligner.")
_MULTIMODAL_DELIMITERS = frozenset({"image_start", "image_end", "image_newline"})
_MULTIMODAL_BIAS_VL_SUFFIX = ".bias_vl"
_MTP_PREFIX = "mtp."
_SCALE_SUFFIX = ".scale"
_WEIGHT_SUFFIX = ".weight"
_SLICE_CHUNK_BYTES = 256 * 1024 * 1024

_VISION_CONFIG_KEY_MAP = (
    ("num_hidden_layers", "vision_n_layers"),
    ("hidden_size", "vision_dim"),
    ("num_attention_heads", "vision_n_heads"),
    ("intermediate_size", "vision_inter_dim"),
    ("patch_size", "vision_patch_size"),
    ("rope_theta", "vision_rope_theta"),
    ("downsample_ratio", "vision_downsample_ratio"),
    ("max_image_tokens", "vision_max_n_token"),
    ("min_pixels", "vision_min_pixels"),
    ("max_wh_ratio", "vision_max_wh_ratio"),
)

DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "R9",
        "summary": (
            "Lossless bfloat16-to-float32 cast of image_processor patches to match"
            " vision.patch_embed.proj.weight.dtype before MultimodalTransformer.forward"
        ),
        "release_lines": "inference/image_processor.py:126-127; inference/vision.py:42",
        "reason": (
            "Under fp32_compute=True, vision nn.Linear weights are float32 holding exact"
            " bfloat16 checkpoint values while load_image returns bfloat16 patches"
        ),
        "numerics": "none (lossless bfloat16 -> float32 widening)",
        "how_checked": "R9 patch cast unit test and revert patch 02_drop_r9_patch_cast.patch",
    },
    {
        "id": "R10",
        "summary": (
            "Lazy on-the-fly FP8/FP4 block dequantization from resident compressed"
            " safetensors buffers into transient bfloat16 module weights per forward call"
        ),
        "release_lines": "tests/utils/deepseek_v41_reference/runner.py:121-170",
        "reason": (
            "Keep 749B reference resident memory within the 1.2 TB scope cap (~0.51 TB"
            " compressed weights vs ~1.50 TB eager bfloat16)"
        ),
        "numerics": "none (bit-identical to eager runner.dequantize_into)",
        "how_checked": "torch.equal(logits_lazy, logits_eager) on multimodal and text-only inputs",
    },
)


def multimodal_model_args_from_hf_config(
    hf_config: dict[str, Any],
    **overrides: Any,
) -> ref_model.ModelArgs:
  """Maps an HF `config.json` dictionary (including `vision_config`) onto `ModelArgs`.

  Args:
    hf_config: Parsed `config.json` dictionary containing `text_config`,
      `vision_config`, and `image_token_id`.
    **overrides: Explicit `ModelArgs` field overrides applied last.

  Returns:
    Populated `ref_model.ModelArgs` with `vision_enabled == True`.

  Raises:
    ValueError: If `vision_config`, `image_token_id`, or any required vision field
      is missing, or if `vision_n_layers <= 0`.
  """
  if "vision_config" not in hf_config or not isinstance(hf_config["vision_config"], dict):
    raise ValueError(f"HF config is missing required dict 'vision_config'; keys={sorted(hf_config)}")
  if "image_token_id" not in hf_config:
    raise ValueError(f"HF config is missing required 'image_token_id'; keys={sorted(hf_config)}")

  vision_cfg = hf_config["vision_config"]
  mm_fields: dict[str, Any] = {"image_token_id": int(hf_config["image_token_id"])}
  for hf_key, arg_name in _VISION_CONFIG_KEY_MAP:
    if hf_key not in vision_cfg:
      raise ValueError(f"HF vision_config is missing required key {hf_key!r}; keys={sorted(vision_cfg)}")
    mm_fields[arg_name] = vision_cfg[hf_key]

  base_args = runner.model_args_from_hf_config(hf_config)
  merged_updates = {**mm_fields, **overrides}
  args = dataclasses.replace(base_args, **merged_updates)
  if args.vision_n_layers <= 0:
    raise ValueError(f"Expected vision_n_layers > 0 for multimodal model, got {args.vision_n_layers}")
  return args


def is_multimodal_param_key(name: str) -> bool:
  """Returns True if `name` is one of the non-MTP multimodal parameter keys."""
  if name.startswith(_MTP_PREFIX) or name.endswith(_SCALE_SUFFIX):
    return False
  return (
      name.startswith(_MULTIMODAL_PREFIXES) or name in _MULTIMODAL_DELIMITERS or name.endswith(_MULTIMODAL_BIAS_VL_SUFFIX)
  )


def is_multimodal_reference_key(name: str) -> bool:
  """Returns True if `name` is a text or multimodal parameter key loaded by the reference."""
  return runner.is_reference_key(name) or is_multimodal_param_key(name)


def read_safetensors_headers(ckpt_dir: str) -> dict[str, dict[str, dict[str, Any]]]:
  """Reads binary safetensors JSON headers across all shards in `ckpt_dir`.

  If `model.safetensors.index.json` is present, verifies 1:1 key-to-shard agreement
  against the binary shard headers.

  Args:
    ckpt_dir: Directory containing `*.safetensors` shard files.

  Returns:
    Mapping `{shard_path: {tensor_name: {"dtype": ..., "shape": ..., "data_offsets": ...}}}`.

  Raises:
    ValueError: If no shards exist, a key is duplicated across shards, or
      `model.safetensors.index.json` disagrees with the shard headers.
  """
  shard_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
  if not shard_paths:
    raise ValueError(f"No .safetensors files found in checkpoint directory {ckpt_dir!r}")

  headers_by_shard: dict[str, dict[str, dict[str, Any]]] = {}
  key_to_shard_basename: dict[str, str] = {}
  for shard_path in shard_paths:
    basename = os.path.basename(shard_path)
    with open(shard_path, "rb") as f:
      raw_len = f.read(8)
      if len(raw_len) != 8:
        raise ValueError(f"Truncated safetensors header in {shard_path!r}")
      header_len = struct.unpack("<Q", raw_len)[0]
      raw_header = json.loads(f.read(header_len).decode("utf-8"))
    tensor_headers: dict[str, dict[str, Any]] = {}
    for key, meta in raw_header.items():
      if key == "__metadata__":
        continue
      if key in key_to_shard_basename:
        raise ValueError(f"Duplicate tensor key {key!r} in {basename!r} and {key_to_shard_basename[key]!r}")
      key_to_shard_basename[key] = basename
      tensor_headers[key] = meta
    headers_by_shard[shard_path] = tensor_headers

  index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
  if os.path.exists(index_path):
    with open(index_path, encoding="utf-8") as f:
      index_data = json.load(f)
    if "weight_map" not in index_data or not isinstance(index_data["weight_map"], dict):
      raise ValueError(f"Invalid weight_map in {index_path!r}")
    weight_map: dict[str, str] = index_data["weight_map"]
    if weight_map != key_to_shard_basename:
      only_index = sorted(set(weight_map) - set(key_to_shard_basename))
      only_shards = sorted(set(key_to_shard_basename) - set(weight_map))
      mismatched = sorted(
          k for k in set(weight_map) & set(key_to_shard_basename) if weight_map[k] != key_to_shard_basename[k]
      )
      raise ValueError(
          f"model.safetensors.index.json mismatch vs shard headers in {ckpt_dir!r}: "
          f"only_in_index={only_index[:5]}, only_in_shards={only_shards[:5]}, mismatched={mismatched[:5]}"
      )

  return headers_by_shard


def _validate_multimodal_header_dtypes(
    headers_by_shard: dict[str, dict[str, dict[str, Any]]],
) -> None:
  """Verifies BF16/F32 safetensors header dtypes on all non-MTP multimodal tensors."""
  for shard_path, shard_header in headers_by_shard.items():
    for name, meta in shard_header.items():
      if name.startswith(_MTP_PREFIX) or name.endswith(_SCALE_SUFFIX):
        continue
      dtype_code = meta["dtype"]
      if name.startswith(_MULTIMODAL_PREFIXES) or name in _MULTIMODAL_DELIMITERS:
        if dtype_code != "BF16":
          raise ValueError(
              f"Expected safetensors header dtype 'BF16' for multimodal tensor {name!r} "
              f"in {os.path.basename(shard_path)!r}, got {dtype_code!r}"
          )
      elif name.endswith(_MULTIMODAL_BIAS_VL_SUFFIX):
        if dtype_code != "F32":
          raise ValueError(
              f"Expected safetensors header dtype 'F32' for routing bias tensor {name!r} "
              f"in {os.path.basename(shard_path)!r}, got {dtype_code!r}"
          )


def _drop_file_page_cache(path: str) -> None:
  """Advises the kernel to evict clean page-cache pages for `path` after reading."""
  if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
    return
  fd = os.open(path, os.O_RDONLY)
  try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
  finally:
    os.close(fd)


def _read_owned_tensor(sf: Any, name: str) -> torch.Tensor:
  """Reads `name` from an open safetensors handle into owned CPU memory in chunks."""
  tensor_slice = sf.get_slice(name)
  shape = tuple(tensor_slice.get_shape())
  if not shape or shape[0] == 0:
    return sf.get_tensor(name).clone()
  probe = tensor_slice[:0]
  out = torch.empty(shape, dtype=probe.dtype, device="cpu")
  elems_per_row = max(1, out.numel() // shape[0])
  bytes_per_row = max(1, elems_per_row * out.element_size())
  rows_per_chunk = max(1, _SLICE_CHUNK_BYTES // bytes_per_row)
  for start in range(0, shape[0], rows_per_chunk):
    stop = min(shape[0], start + rows_per_chunk)
    out[start:stop].copy_(tensor_slice[start:stop])
  return out


def _copy_param_from_safetensors(dest: torch.Tensor, sf: Any, name: str) -> None:
  """Copies unquantized tensor `name` from `sf` into `dest` in row chunks."""
  tensor_slice = sf.get_slice(name)
  shape = tuple(tensor_slice.get_shape())
  if shape != tuple(dest.shape):
    raise ValueError(f"Shape mismatch for {name!r}: checkpoint {shape} vs model {tuple(dest.shape)}")
  if not shape or shape[0] == 0:
    dest.copy_(sf.get_tensor(name))
    return
  elems_per_row = max(1, dest.numel() // shape[0])
  bytes_per_row = max(1, elems_per_row * dest.element_size())
  rows_per_chunk = max(1, _SLICE_CHUNK_BYTES // bytes_per_row)
  for start in range(0, shape[0], rows_per_chunk):
    stop = min(shape[0], start + rows_per_chunk)
    dest[start:stop].copy_(tensor_slice[start:stop])


_BYTE_ARANGE = torch.arange(256, dtype=torch.long)
_FP4_BYTE_LUT = torch.stack(
    [
        runner._FP4_E2M1_VALUES[_BYTE_ARANGE & 0xF],  # pylint: disable=protected-access
        runner._FP4_E2M1_VALUES[_BYTE_ARANGE >> 4],  # pylint: disable=protected-access
    ],
    dim=-1,
).contiguous()
_DEQUANT_FAST_VERIFIED = False


def verify_dequantize_into_fast() -> None:
  """Verifies `dequantize_into_fast` is bitwise equal (`torch.equal`) to `runner.dequantize_into`."""
  gen = torch.Generator().manual_seed(42)
  qw_fp4 = torch.randint(-128, 128, (64, 2560), generator=gen, dtype=torch.int8)
  qs_fp4 = torch.randint(110, 135, (64, 160), generator=gen, dtype=torch.uint8).view(torch.float8_e8m0fnu)
  qw_fp8_2d = torch.randint(0, 120, (128, 128), generator=gen, dtype=torch.uint8).view(torch.float8_e4m3fn)
  qs_fp8_2d = torch.randint(110, 135, (4, 4), generator=gen, dtype=torch.uint8).view(torch.float8_e8m0fnu)
  qw_fp8_row = torch.randint(0, 120, (37, 64), generator=gen, dtype=torch.uint8).view(torch.float8_e4m3fn)
  qs_fp8_row = torch.randint(110, 135, (37, 2), generator=gen, dtype=torch.uint8).view(torch.float8_e8m0fnu)

  cases = (
      ("fp4_1x32", qw_fp4, qs_fp4, (64, 5120)),
      ("fp8_32x32", qw_fp8_2d, qs_fp8_2d, (128, 128)),
      ("fp8_1x32", qw_fp8_row, qs_fp8_row, (37, 64)),
  )
  for label, qw, qs, out_shape in cases:
    for dt in (torch.float32, torch.bfloat16):
      out_ref = torch.empty(out_shape, dtype=dt)
      out_fast = torch.empty(out_shape, dtype=dt)
      runner.dequantize_into(out_ref, qw, qs)
      _dequantize_into_fast_impl(out_fast, qw, qs)
      if not torch.equal(out_ref, out_fast):
        raise AssertionError(f"dequantize_into_fast bitwise mismatch vs runner.dequantize_into on {label} ({dt})")


def _dequantize_into_fast_impl(
    out: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    rows_per_chunk: int = 1 << 20,
) -> None:
  """Vectorized FP4-E2M1 / FP8-E4M3 block-broadcast dequantization matching `runner.dequantize_into`."""
  rows = int(weight.shape[0])
  cols = int(out.shape[1])
  if rows == 0 or cols == 0:
    runner.dequantize_into(out, weight, scale, rows_per_chunk=rows_per_chunk)
    return
  s_rows = int(scale.shape[0])
  s_cols = int(scale.shape[1])
  block_rows = -(-rows // s_rows)
  block_cols = -(-cols // s_cols)

  if (
      weight.dtype == torch.int8
      and block_rows == 1
      and s_rows == rows
      and s_cols * block_cols == cols
      and out.dtype in (torch.float32, torch.bfloat16)
      and weight.device.type == "cpu"
  ):
    w_3d = F.embedding(weight.view(torch.uint8).int(), _FP4_BYTE_LUT).view(rows, s_cols, block_cols)
    s_3d = scale.float().unsqueeze(-1)
    if out.dtype == torch.float32:
      torch.mul(w_3d, s_3d, out=out.view(rows, s_cols, block_cols))
    else:
      out.view(rows, s_cols, block_cols).copy_((w_3d * s_3d).to(out.dtype))
    return

  if (
      weight.dtype != torch.int8
      and rows == s_rows * block_rows
      and cols == s_cols * block_cols
      and out.dtype in (torch.float32, torch.bfloat16)
  ):
    w_4d = weight.float().view(s_rows, block_rows, s_cols, block_cols)
    s_4d = scale.float().view(s_rows, 1, s_cols, 1)
    if out.dtype == torch.float32:
      torch.mul(w_4d, s_4d, out=out.view(s_rows, block_rows, s_cols, block_cols))
    else:
      out.view(s_rows, block_rows, s_cols, block_cols).copy_((w_4d * s_4d).to(out.dtype))
    return

  runner.dequantize_into(out, weight, scale, rows_per_chunk=rows_per_chunk)


def dequantize_into_fast(
    out: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    rows_per_chunk: int = 1 << 20,
) -> None:
  """Dequantizes `weight` * `scale` into `out` after a one-time startup bit-identity self-test."""
  global _DEQUANT_FAST_VERIFIED  # pylint: disable=global-statement
  if not _DEQUANT_FAST_VERIFIED:
    verify_dequantize_into_fast()
    _DEQUANT_FAST_VERIFIED = True
  _dequantize_into_fast_impl(out, weight, scale, rows_per_chunk=rows_per_chunk)


def _dequantize_lazy_weight(
    qweight: torch.Tensor,
    qscale: torch.Tensor,
    out_shape: tuple[int, int],
    storage_dtype: torch.dtype,
) -> torch.Tensor:
  """Dequantizes a compressed FP8/FP4 weight into `storage_dtype` matching eager load."""
  out = torch.empty(out_shape, dtype=storage_dtype, device=qweight.device)
  dequantize_into_fast(out, qweight, qscale)
  return out


def _attach_lazy_linear(
    mod: ref_model.Linear,
    qweight: torch.Tensor,
    qscale: torch.Tensor,
) -> None:
  """Replaces `mod.weight` with compressed FP8/FP4 buffers and an on-the-fly forward wrapper."""
  orig_shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
  orig_dtype = mod.weight.dtype
  empty_param = nn.Parameter(torch.empty(0, dtype=orig_dtype), requires_grad=False)
  mod.weight = empty_param
  mod.register_buffer("_lazy_qweight", qweight, persistent=False)
  mod.register_buffer("_lazy_qscale", qscale, persistent=False)
  mod._lazy_weight_shape = orig_shape  # pylint: disable=protected-access
  mod._lazy_weight_dtype = orig_dtype  # pylint: disable=protected-access

  orig_forward = mod.forward

  def _lazy_linear_forward(x: torch.Tensor) -> torch.Tensor:
    tmp_weight = _dequantize_lazy_weight(
        mod._lazy_qweight,  # pylint: disable=protected-access
        mod._lazy_qscale,  # pylint: disable=protected-access
        mod._lazy_weight_shape,  # pylint: disable=protected-access
        mod._lazy_weight_dtype,  # pylint: disable=protected-access
    )
    mod._parameters["weight"] = nn.Parameter(tmp_weight, requires_grad=False)  # pylint: disable=protected-access
    try:
      return orig_forward(x)
    finally:
      mod._parameters["weight"] = empty_param  # pylint: disable=protected-access

  mod.forward = _lazy_linear_forward  # type: ignore[method-assign]


def _attach_lazy_engram_embedding(
    mod: ref_model.ParallelEngramEmbedding,
    name: str,
    qweight: torch.Tensor,
    qscale: torch.Tensor,
) -> None:
  """Replaces `ParallelEngramEmbedding.weight` with row-gathered lazy FP8 dequantization."""
  orig_shape = (int(mod.weight.shape[0]), int(mod.weight.shape[1]))
  orig_dtype = mod.weight.dtype
  if qweight.shape[0] != orig_shape[0] or qscale.shape[0] != orig_shape[0]:
    raise ValueError(
        f"Lazy ParallelEngramEmbedding requires 1xblock_cols row-independent scales for {name!r}; "
        f"got weight={tuple(qweight.shape)}, scale={tuple(qscale.shape)}, expected_rows={orig_shape[0]}"
    )
  empty_param = nn.Parameter(torch.empty(0, dtype=orig_dtype), requires_grad=False)
  mod.weight = empty_param
  mod.register_buffer("_lazy_qweight", qweight, persistent=False)
  mod.register_buffer("_lazy_qscale", qscale, persistent=False)
  mod._lazy_weight_shape = orig_shape  # pylint: disable=protected-access
  mod._lazy_weight_dtype = orig_dtype  # pylint: disable=protected-access

  def _lazy_engram_forward(indices: torch.Tensor) -> torch.Tensor:
    mask = (indices < mod.vocab_start_idx) | (indices >= mod.vocab_end_idx)
    local_indices = (indices - mod.vocab_start_idx).masked_fill(mask, 0)
    flat_indices = local_indices.reshape(-1)
    if flat_indices.numel() == 0:
      values = torch.empty(
          *indices.shape,
          mod.dim,
          dtype=mod._lazy_weight_dtype,  # pylint: disable=protected-access
          device=indices.device,
      )
    else:
      sub_qw = mod._lazy_qweight.index_select(0, flat_indices)  # pylint: disable=protected-access
      sub_qs = mod._lazy_qscale.index_select(0, flat_indices)  # pylint: disable=protected-access
      sub_out = _dequantize_lazy_weight(
          sub_qw,
          sub_qs,
          (int(flat_indices.numel()), mod._lazy_weight_shape[1]),  # pylint: disable=protected-access
          mod._lazy_weight_dtype,  # pylint: disable=protected-access
      )
      values = sub_out.view(*indices.shape, mod.dim)
    values = values.masked_fill(mask.unsqueeze(-1), 0)
    if ref_model.world_size > 1:
      dist.all_reduce(values)
    return values

  mod.forward = _lazy_engram_forward  # type: ignore[method-assign]


def _attach_lazy_quantized_weight(
    model: multimodal.MultimodalTransformer,
    param_name: str,
    qweight: torch.Tensor,
    qscale: torch.Tensor,
) -> None:
  """Attaches compressed weight + scale buffers onto the module owning `param_name`."""
  module_path, sep, attr_name = param_name.rpartition(".")
  if not sep or attr_name != "weight":
    raise ValueError(f"Expected quantized parameter ending in '.weight', got {param_name!r}")
  submod = model.get_submodule(module_path)
  if isinstance(submod, ref_model.Linear):
    _attach_lazy_linear(submod, qweight, qscale)
  elif isinstance(submod, ref_model.ParallelEngramEmbedding):
    _attach_lazy_engram_embedding(submod, param_name, qweight, qscale)
  else:
    raise ValueError(f"Unsupported module type {type(submod).__name__} for lazy-dequant parameter {param_name!r}")


def _install_lazy_attention_wo_a_wrappers(model: multimodal.MultimodalTransformer) -> None:
  """Wraps `Attention.forward` so direct `self.wo_a.weight` reads materialize transiently."""
  for submod in model.modules():
    if not isinstance(submod, ref_model.Attention):
      continue
    wo_a_mod = submod.wo_a
    if not hasattr(wo_a_mod, "_lazy_qweight"):
      continue
    empty_param = wo_a_mod.weight
    orig_attn_forward = submod.forward

    def _make_attn_forward(
        attn_fn: Any,
        linear_mod: ref_model.Linear,
        placeholder: nn.Parameter,
    ) -> Any:
      def _lazy_attn_forward(*attn_args: Any, **attn_kwargs: Any) -> torch.Tensor:
        tmp_wo_a = _dequantize_lazy_weight(
            linear_mod._lazy_qweight,  # pylint: disable=protected-access
            linear_mod._lazy_qscale,  # pylint: disable=protected-access
            linear_mod._lazy_weight_shape,  # pylint: disable=protected-access
            linear_mod._lazy_weight_dtype,  # pylint: disable=protected-access
        )
        linear_mod._parameters["weight"] = nn.Parameter(tmp_wo_a, requires_grad=False)  # pylint: disable=protected-access
        try:
          return attn_fn(*attn_args, **attn_kwargs)
        finally:
          linear_mod._parameters["weight"] = placeholder  # pylint: disable=protected-access

      return _lazy_attn_forward

    submod.forward = _make_attn_forward(orig_attn_forward, wo_a_mod, empty_param)  # type: ignore[method-assign]


def load_multimodal_checkpoint_into(
    model: multimodal.MultimodalTransformer,
    ckpt_dir: str,
    *,
    lazy_dequant: bool = True,
) -> None:
  """Loads all text and multimodal tensors from `ckpt_dir` into `model` with strict audits.

  Verifies:
    1. Every non-MTP `vision.*`, `aligner.*`, and `image_*` tensor has header dtype `BF16`,
       and every non-MTP `*.bias_vl` tensor has header dtype `F32`.
    2. Every non-MTP checkpoint tensor (including `.scale` siblings) is consumed
       exactly once.
    3. Every parameter of `model` is populated, and skipped checkpoint keys consist
       exclusively of `mtp.*` keys.

  Args:
    model: Initialized `MultimodalTransformer` instance.
    ckpt_dir: Directory holding `*.safetensors` shards.
    lazy_dequant: When True (deviation R10), keeps block-quantized FP8/FP4 weights
      compressed in module buffers and dequantizes per module forward call.

  Raises:
    ValueError: On any dtype mismatch, unexpected non-MTP key, duplicate key, or
      missing model parameter.
  """
  headers_by_shard = read_safetensors_headers(ckpt_dir)
  _validate_multimodal_header_dtypes(headers_by_shard)

  key_to_shard_path: dict[str, str] = {}
  for shard_path, shard_header in headers_by_shard.items():
    for key in shard_header:
      key_to_shard_path[key] = shard_path

  expected_params = dict(model.named_parameters())
  consumed_ckpt_keys: set[str] = set()
  populated_params: set[str] = set()
  all_ckpt_keys: set[str] = set(key_to_shard_path)

  with torch.no_grad():
    for shard_path in sorted(headers_by_shard):
      shard_keys = set(headers_by_shard[shard_path])
      with safe_open(shard_path, framework="pt", device="cpu") as sf:
        for name in sorted(shard_keys):
          if not is_multimodal_reference_key(name):
            continue
          if name not in expected_params:
            raise ValueError(f"Unexpected checkpoint tensor {name!r} not present in MultimodalTransformer")
          if name in consumed_ckpt_keys:
            raise ValueError(f"Checkpoint tensor {name!r} consumed more than once")

          scale_name = f"{name[:-len(_WEIGHT_SUFFIX)]}{_SCALE_SUFFIX}" if name.endswith(_WEIGHT_SUFFIX) else None
          has_scale = scale_name is not None and scale_name in key_to_shard_path
          if has_scale:
            assert scale_name is not None
            if scale_name in consumed_ckpt_keys:
              raise ValueError(f"Checkpoint scale tensor {scale_name!r} consumed more than once")
            consumed_ckpt_keys.add(name)
            consumed_ckpt_keys.add(scale_name)
            populated_params.add(name)

            scale_shard = key_to_shard_path[scale_name]
            if lazy_dequant:
              qweight = _read_owned_tensor(sf, name)
              if scale_shard == shard_path:
                qscale = _read_owned_tensor(sf, scale_name)
              else:
                with safe_open(scale_shard, framework="pt", device="cpu") as scale_sf:
                  qscale = _read_owned_tensor(scale_sf, scale_name)
              _attach_lazy_quantized_weight(model, name, qweight, qscale)
            else:
              qweight = _read_owned_tensor(sf, name)
              if scale_shard == shard_path:
                qscale = _read_owned_tensor(sf, scale_name)
              else:
                with safe_open(scale_shard, framework="pt", device="cpu") as scale_sf:
                  qscale = _read_owned_tensor(scale_sf, scale_name)
              dequantize_into_fast(expected_params[name].data, qweight, qscale)
          else:
            consumed_ckpt_keys.add(name)
            populated_params.add(name)
            _copy_param_from_safetensors(expected_params[name].data, sf, name)
      _drop_file_page_cache(shard_path)

  if lazy_dequant:
    _install_lazy_attention_wo_a_wrappers(model)

  skipped_keys = sorted(all_ckpt_keys - consumed_ckpt_keys)
  non_mtp_skipped = [k for k in skipped_keys if not k.startswith(_MTP_PREFIX)]
  if non_mtp_skipped:
    raise ValueError(f"Non-MTP checkpoint tensors were skipped ({len(non_mtp_skipped)}): {non_mtp_skipped[:20]}")

  missing_params = sorted(set(expected_params) - populated_params)
  if missing_params:
    raise ValueError(f"Multimodal checkpoint is missing {len(missing_params)} model parameters: {missing_params[:20]}")


def load_multimodal_reference_model(
    ckpt_dir: str,
    *,
    dtype: str = "bf16",
    max_seq_len: int = 2048,
    tokenizer: Any = None,
    fp32_compute: bool = True,
    lazy_dequant: bool = True,
    **overrides: Any,
) -> tuple[multimodal.MultimodalTransformer, ref_model.ModelArgs]:
  """Builds and loads `MultimodalTransformer` from an HF-layout checkpoint directory.

  Args:
    ckpt_dir: Local checkpoint directory containing `config.json` and `*.safetensors`.
    dtype: Weight storage dtype (`"bf16"` or `"fp32"`).
    max_seq_len: Maximum sequence length sizing RoPE tables and attention caches.
    tokenizer: Optional HF tokenizer for Engram hashing; loaded from `ckpt_dir` if None.
    fp32_compute: Whether matmuls and vision/norm layers run in float32.
    lazy_dequant: Whether to keep FP8/FP4 weights compressed in memory (deviation R10).
    **overrides: Additional `ModelArgs` field overrides.

  Returns:
    Tuple `(model, args)` ready for evaluation.
  """
  hf_config = runner.load_hf_config(ckpt_dir)
  args = multimodal_model_args_from_hf_config(
      hf_config,
      dtype=dtype,
      max_seq_len=max_seq_len,
      max_batch_size=1,
      fp32_compute=fp32_compute,
      **overrides,
  )
  if tokenizer is None and args.engram_layer_ids:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
  model = multimodal.build_multimodal_model(args, tokenizer=tokenizer)
  load_multimodal_checkpoint_into(model, ckpt_dir, lazy_dequant=lazy_dequant)
  return model, args


def forward_multimodal_logits(
    model: multimodal.MultimodalTransformer,
    input_ids: torch.Tensor,
    *,
    images: list[list[image_processor.ImageInput] | None] | None = None,
    token_types: torch.Tensor | None = None,
) -> torch.Tensor:
  """Runs a full-sequence multimodal reference forward pass (deviation R9).

  Casts each `ImageInput.patches` tensor to `model.vision.patch_embed.proj.weight.dtype`
  without mutating caller inputs or editing vendored `vision.py`.

  Args:
    model: Loaded `MultimodalTransformer`.
    input_ids: Integer token IDs `[B, S]`.
    images: Per-batch list of `ImageInput` lists (or `None` for text-only).
    token_types: Token type tensor `[B, S]` (`-1` for text, `0..3` for image spans).

  Returns:
    Float32 full-sequence logits `[B, S, vocab_size]`.
  """
  cast_images: list[list[image_processor.ImageInput] | None] | None = None
  if images is not None:
    if model.vision is None:
      raise RuntimeError("forward_multimodal_logits received images, but model.vision is None")
    patch_dtype = model.vision.patch_embed.proj.weight.dtype
    cast_images = []
    for sample_imgs in images:
      if sample_imgs is None:
        cast_images.append(None)
      else:
        cast_images.append([dataclasses.replace(img, patches=img.patches.to(dtype=patch_dtype)) for img in sample_imgs])

  with torch.no_grad():
    logits = model(
        input_ids,
        start_pos=0,
        images=cast_images,
        token_types=token_types,
        full_logits=True,
    )
    assert isinstance(logits, torch.Tensor)
    return logits.float()


def resident_model_bytes(model: nn.Module) -> int:
  """Returns total unique storage bytes across all parameters and buffers of `model`."""
  seen_ptrs: set[int] = set()
  total_bytes = 0
  for tensor in itertools.chain(model.parameters(), model.buffers()):
    if tensor.numel() == 0:
      continue
    ptr = int(tensor.untyped_storage().data_ptr())
    if ptr in seen_ptrs:
      continue
    seen_ptrs.add(ptr)
    total_bytes += int(tensor.numel() * tensor.element_size())
  return total_bytes


def _collect_default_dtype_tensors(
    model: multimodal.MultimodalTransformer,
) -> tuple[list[nn.Parameter], list[tuple[nn.Module, str, torch.Tensor]]]:
  """Finds parameters and cache buffers whose dtype follows torch's default_dtype."""
  params_to_cast: list[nn.Parameter] = []
  seen_param_ids: set[int] = set()

  def _add_param(p: nn.Parameter | None) -> None:
    if p is not None and p.numel() > 0 and p.dtype == torch.float32 and id(p) not in seen_param_ids:
      seen_param_ids.add(id(p))
      params_to_cast.append(p)

  _add_param(model.embed.weight)
  if model.vision_enabled:
    _add_param(model.image_start)
    _add_param(model.image_end)
    _add_param(model.image_newline)
    for root in (model.vision, model.aligner):
      if root is None:
        continue
      for sub in root.modules():
        if isinstance(sub, nn.Linear):
          _add_param(sub.weight)
          _add_param(sub.bias)

  buffers_to_cast: list[tuple[nn.Module, str, torch.Tensor]] = []
  for sub in model.modules():
    if isinstance(sub, ref_model.RMSNorm):
      _add_param(sub.weight)
    elif isinstance(sub, ref_model.Engram):
      _add_param(sub.q_weight)
      _add_param(sub.k_weight)
    elif isinstance(sub, ref_model.Gate):
      _add_param(sub.weight)
    elif isinstance(sub, ref_model.Attention):
      for buf_name in ("window_kv_cache", "compress_kv_cache"):
        if hasattr(sub, buf_name):
          buf = getattr(sub, buf_name)
          if isinstance(buf, torch.Tensor) and buf.dtype == torch.float32:
            buffers_to_cast.append((sub, buf_name, buf))
    elif isinstance(sub, ref_model.Indexer) and sub.owns_k:
      if hasattr(sub, "k_cache") and sub.k_cache.dtype == torch.float32:
        buffers_to_cast.append((sub, "k_cache", sub.k_cache))

  return params_to_cast, buffers_to_cast


@contextlib.contextmanager
def bf16_compute_context(model: multimodal.MultimodalTransformer) -> Iterator[None]:
  """Diagnostic-scoped context running an `fp32_compute=True` model in `bf16` compute mode.

  Temporarily disables `ref_model.compute_fp32` and casts parameters/buffers created
  under default float32 dtype back to `torch.bfloat16`, restoring every parameter,
  buffer, and module global bit-identically on exit without reloading checkpoint shards.
  """
  prev_compute_fp32 = ref_model.compute_fp32
  params_to_cast, buffers_to_cast = _collect_default_dtype_tensors(model)
  saved_params = [(p, p.data.clone()) for p in params_to_cast]
  saved_buffers = [(owner, name, buf.clone()) for owner, name, buf in buffers_to_cast]
  try:
    with torch.no_grad():
      ref_model.compute_fp32 = False
      for p, _ in saved_params:
        p.data = p.data.to(torch.bfloat16)
      for owner, name, buf in saved_buffers:
        owner._buffers[name] = buf.to(torch.bfloat16)  # pylint: disable=protected-access
    yield
  finally:
    with torch.no_grad():
      ref_model.compute_fp32 = prev_compute_fp32
      for p, orig_data in saved_params:
        p.data = orig_data
      for owner, name, orig_buf in saved_buffers:
        owner._buffers[name] = orig_buf  # pylint: disable=protected-access


def _perturb_scores_alternating_one_ulp(scores: torch.Tensor) -> torch.Tensor:
  """Steps finite entries by +1 ULP on even last-dim indices and -1 ULP on odd indices."""
  scores_fp32 = scores.float()
  last_dim = scores_fp32.size(-1)
  targets = torch.full((last_dim,), -torch.inf, dtype=scores_fp32.dtype, device=scores_fp32.device)
  targets[0::2] = torch.inf
  perturbed = torch.nextafter(scores_fp32, targets)
  return torch.where(torch.isfinite(scores_fp32), perturbed, scores_fp32)


def _gate_forward_one_ulp(
    self: ref_model.Gate,
    x: torch.Tensor,
    image_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Gate forward applying alternating ±1-ULP tie-flip perturbation to pre-topk selection scores."""
  scores = ref_model.linear(x.float(), self.weight.float()) / self.gate_temp
  if self.score_func == "softmax":
    scores = scores.softmax(dim=-1)
  elif self.score_func == "sigmoid":
    scores = scores.sigmoid()
  else:
    scores = F.softplus(scores).sqrt()
  bias = self.bias
  if image_mask is not None and self.bias_vl is not None:
    bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
  selection_scores = _perturb_scores_alternating_one_ulp(scores + bias)
  indices = selection_scores.topk(self.topk, dim=-1)[1]
  weights = scores.gather(1, indices)
  if self.norm_topk_prob and self.topk > 1:
    weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
  weights *= self.route_scale
  return weights, indices


def _indexer_forward_one_ulp(
    self: ref_model.Indexer,
    x: torch.Tensor,
    qr: torch.Tensor,
    latent: torch.Tensor | None,
    start_pos: int,
    offset: int,
) -> torch.Tensor:
  """Indexer forward applying alternating ±1-ULP tie-flip perturbation to `index_score` before selection."""
  assert self.freqs_cis is not None
  bsz, seqlen, _ = x.size()
  ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

  if self.owns_k and latent is not None:
    freqs = (
        self.freqs_cis[: seqlen - seqlen % ratio : ratio]
        if start_pos == 0
        else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
    )
    k = self.k_norm(self.wk(latent))
    ref_model.apply_rotary_emb(k[..., -rd:], freqs)
    ref_model.fp4_act_quant(k, ref_model.fp4_block_size, True)
    self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
  if self.owns_k:
    ref_model.shared_attn.index_k = self.k_cache

  q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
  ref_model.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
  ref_model.fp4_act_quant(q, ref_model.fp4_block_size, True)

  assert ref_model.shared_attn.index_k is not None
  index_k = ref_model.shared_attn.index_k[:bsz, : end_pos // ratio]
  weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
  index_score = torch.einsum("bshd,btd->bsht", q, index_k)
  index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
  if ref_model.world_size > 1:
    dist.all_reduce(index_score)
  index_score = _perturb_scores_alternating_one_ulp(index_score)

  compress_lens: torch.Tensor | int
  if start_pos == 0:
    compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
    index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
  else:
    compress_lens = end_pos // ratio

  if self.is_candidate_source:
    ref_model.shared_attn.candidates = ref_model.select_candidate_blocks(
        index_score,
        compress_lens,
        self.candidate_topk_blocks,
        self.candidate_block_size,
    )
  elif self.uses_candidates:
    assert ref_model.shared_attn.candidates is not None
    index_score = index_score.masked_fill(~ref_model.shared_attn.candidates, -torch.inf)

  topk = min(self.index_topk, end_pos // ratio)
  idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
  return torch.where(idxs < compress_lens, idxs + offset, -1).int()


@contextlib.contextmanager
def one_ulp_routing_context() -> Iterator[None]:
  """Diagnostic-scoped context applying alternating ±1-ULP tie-flip perturbations to Gate and Indexer selection scores."""
  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.object(ref_model.Gate, "forward", _gate_forward_one_ulp))
    stack.enter_context(mock.patch.object(ref_model.Indexer, "forward", _indexer_forward_one_ulp))
    yield


@dataclasses.dataclass(frozen=True)
class RoutingTrace:
  """Per-layer MoE and Sparse-Attention Indexer routing indices, margins, and ULPs from one pass.

  Attributes:
    ref_topk_experts: `uint16` array `[n_layers, seq_len, top_k]` of selected MoE expert IDs.
    gate_margins: `float32` array `[n_layers, seq_len]` holding `s_(k) - s_(k+1)` of `scores + bias`.
    gate_boundary_experts: `int32` array `[n_layers, seq_len, 2]` holding `(top_k_expert, top_{k+1}_expert)`.
    indexer_margins: Length-`n_layers` tuple where index-source layers with sparse selection hold a
      `float32` array `[seq_len]` (`np.nan` at positions where reachable candidates `<= index_topk`)
      and all other layers hold `None`.
    ref_gate_margins: `float32` array `[n_layers, seq_len]` holding `s_(k) - s_(k+1)` of `scores + bias`.
    ref_gate_boundary_ulp: `float32` array `[n_layers, seq_len]` holding `abs(spacing(s_(k)))`.
    ref_indexer_layer_ids: `int32` array `[n_idx_layers]` of index-source layer indices.
    ref_indexer_topk: `int32` array `[n_idx_layers, seq_len, index_topk]` of selected compressed KV indices
      (padded with `-1` where `topk < index_topk` or causally unreachable).
    ref_indexer_margins: `float32` array `[n_idx_layers, seq_len]` holding `z_(index_topk) - z_(index_topk+1)`
      where finite reachable compressed positions `> index_topk`, else `np.nan`.
    ref_indexer_boundary_ulp: `float32` array `[n_idx_layers, seq_len]` holding `abs(spacing(z_(index_topk)))`
      where finite reachable compressed positions `> index_topk`, else `np.nan`.
  """

  ref_topk_experts: np.ndarray
  gate_margins: np.ndarray
  gate_boundary_experts: np.ndarray
  indexer_margins: tuple[np.ndarray | None, ...]
  ref_gate_margins: np.ndarray
  ref_gate_boundary_ulp: np.ndarray
  ref_indexer_layer_ids: np.ndarray
  ref_indexer_topk: np.ndarray
  ref_indexer_margins: np.ndarray
  ref_indexer_boundary_ulp: np.ndarray


def forward_multimodal_logits_with_routing_trace(
    model: multimodal.MultimodalTransformer,
    input_ids: torch.Tensor,
    *,
    images: list[list[image_processor.ImageInput] | None] | None = None,
    token_types: torch.Tensor | None = None,
) -> tuple[torch.Tensor, RoutingTrace]:
  """Runs `forward_multimodal_logits` while capturing `ref_topk_experts` and `C1` margins with zero extra GEMMs.

  Produces logits bitwise identical (`torch.equal`) to `forward_multimodal_logits` on batch size 1.

  Args:
    model: Loaded `MultimodalTransformer`.
    input_ids: Integer token IDs `[1, S]`.
    images: Per-batch list of `ImageInput` lists (or `None` for text-only).
    token_types: Token type tensor `[1, S]` (`-1` for text, `0..3` for image spans).

  Returns:
    Tuple `(logits_3d, routing_trace)` where `logits_3d` has shape `[1, S, vocab_size]`.

  Raises:
    ValueError: If `input_ids` does not have batch size 1 or `n_routed_experts <= topk`.
    RuntimeError: If any decoder layer fails to execute its `Gate`.
  """
  if input_ids.ndim != 2 or int(input_ids.size(0)) != 1:
    raise ValueError(
        f"forward_multimodal_logits_with_routing_trace requires [1, S] input_ids, got {tuple(input_ids.shape)}"
    )
  n_layers = len(model.layers)
  seq_len = int(input_ids.size(1))

  gate_id_to_layer: dict[int, int] = {}
  indexer_id_to_layer: dict[int, int] = {}
  indexer_id_to_slot: dict[int, int] = {}
  indexer_layer_ids: list[int] = []
  indexer_topk_widths: list[int] = []
  for layer_id, layer in enumerate(model.layers):
    gate_id_to_layer[id(layer.ffn.gate)] = layer_id
    if layer.attn.indexer is not None:
      indexer_id_to_layer[id(layer.attn.indexer)] = layer_id
      indexer_id_to_slot[id(layer.attn.indexer)] = len(indexer_layer_ids)
      indexer_layer_ids.append(layer_id)
      indexer_topk_widths.append(int(layer.attn.indexer.index_topk))

  n_idx_layers = len(indexer_layer_ids)
  captured_topk: list[np.ndarray | None] = [None] * n_layers
  captured_gate_margins: list[np.ndarray | None] = [None] * n_layers
  captured_gate_ulp: list[np.ndarray | None] = [None] * n_layers
  captured_gate_pairs: list[np.ndarray | None] = [None] * n_layers
  captured_indexer_margins: list[np.ndarray | None] = [None] * n_layers
  captured_ref_indexer_topk: list[np.ndarray | None] = [None] * n_idx_layers
  captured_ref_indexer_margins: list[np.ndarray | None] = [None] * n_idx_layers
  captured_ref_indexer_ulp: list[np.ndarray | None] = [None] * n_idx_layers

  def _gate_forward_traced(
      self: ref_model.Gate,
      x: torch.Tensor,
      image_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    scores = ref_model.linear(x.float(), self.weight.float()) / self.gate_temp
    if self.score_func == "softmax":
      scores = scores.softmax(dim=-1)
    elif self.score_func == "sigmoid":
      scores = scores.sigmoid()
    else:
      scores = F.softplus(scores).sqrt()
    bias = self.bias
    if image_mask is not None and self.bias_vl is not None:
      bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
    selection_scores = scores + bias
    indices = selection_scores.topk(self.topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if self.norm_topk_prob and self.topk > 1:
      weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    weights *= self.route_scale

    if id(self) in gate_id_to_layer:
      layer_idx = gate_id_to_layer[id(self)]
      n_experts = int(selection_scores.size(-1))
      if n_experts <= self.topk:
        raise ValueError(f"Gate layer {layer_idx} n_routed_experts={n_experts} must exceed topk={self.topk}")
      if n_experts > 65535:
        raise ValueError(f"Gate layer {layer_idx} n_routed_experts={n_experts} exceeds uint16 range")
      captured_topk[layer_idx] = indices.detach().cpu().numpy().astype(np.uint16, copy=True)
      top_vals, top_idx = selection_scores.float().topk(self.topk + 1, dim=-1, largest=True, sorted=True)
      s_k = top_vals[:, self.topk - 1].detach().cpu().numpy().astype(np.float32, copy=True)
      s_kp1 = top_vals[:, self.topk].detach().cpu().numpy().astype(np.float32, copy=True)
      captured_gate_margins[layer_idx] = (s_k - s_kp1).astype(np.float32, copy=False)
      captured_gate_ulp[layer_idx] = np.abs(np.spacing(s_k)).astype(np.float32, copy=False)
      captured_gate_pairs[layer_idx] = (
          top_idx[:, self.topk - 1 : self.topk + 1].detach().cpu().numpy().astype(np.int32, copy=True)
      )
    return weights, indices

  def _indexer_forward_traced(
      self: ref_model.Indexer,
      x: torch.Tensor,
      qr: torch.Tensor,
      latent: torch.Tensor | None,
      start_pos: int,
      offset: int,
  ) -> torch.Tensor:
    assert self.freqs_cis is not None
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

    if self.owns_k and latent is not None:
      freqs = (
          self.freqs_cis[: seqlen - seqlen % ratio : ratio]
          if start_pos == 0
          else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
      )
      k = self.k_norm(self.wk(latent))
      ref_model.apply_rotary_emb(k[..., -rd:], freqs)
      ref_model.fp4_act_quant(k, ref_model.fp4_block_size, True)
      self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
    if self.owns_k:
      ref_model.shared_attn.index_k = self.k_cache

    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    ref_model.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    ref_model.fp4_act_quant(q, ref_model.fp4_block_size, True)

    assert ref_model.shared_attn.index_k is not None
    index_k = ref_model.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if ref_model.world_size > 1:
      dist.all_reduce(index_score)

    compress_lens: torch.Tensor | int
    if start_pos == 0:
      compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
      index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
    else:
      compress_lens = end_pos // ratio

    if self.is_candidate_source:
      ref_model.shared_attn.candidates = ref_model.select_candidate_blocks(
          index_score,
          compress_lens,
          self.candidate_topk_blocks,
          self.candidate_block_size,
      )
    elif self.uses_candidates:
      assert ref_model.shared_attn.candidates is not None
      index_score = index_score.masked_fill(~ref_model.shared_attn.candidates, -torch.inf)

    topk = min(self.index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_idxs = torch.where(idxs < compress_lens, idxs + offset, -1).int()

    if id(self) in indexer_id_to_slot:
      layer_idx = indexer_id_to_layer[id(self)]
      slot_idx = indexer_id_to_slot[id(self)]
      padded_topk = np.full((seqlen, int(self.index_topk)), -1, dtype=np.int32)
      if topk > 0:
        padded_topk[:, :topk] = selected_idxs[0].detach().cpu().numpy().astype(np.int32, copy=True)
      captured_ref_indexer_topk[slot_idx] = padded_topk

      margin_arr = np.full((seqlen,), np.nan, dtype=np.float32)
      ulp_arr = np.full((seqlen,), np.nan, dtype=np.float32)
      if int(index_score.size(-1)) > self.index_topk:
        scores_2d = index_score[0].float()
        finite_counts = torch.isfinite(scores_2d).sum(dim=-1)
        active_mask = finite_counts > self.index_topk
        if bool(active_mask.any().item()):
          top_vals = scores_2d[active_mask].topk(self.index_topk + 1, dim=-1, largest=True, sorted=True).values
          z_k = top_vals[:, self.index_topk - 1].detach().cpu().numpy().astype(np.float32, copy=True)
          z_kp1 = top_vals[:, self.index_topk].detach().cpu().numpy().astype(np.float32, copy=True)
          active_np = active_mask.cpu().numpy()
          margin_arr[active_np] = (z_k - z_kp1).astype(np.float32, copy=False)
          ulp_arr[active_np] = np.abs(np.spacing(z_k)).astype(np.float32, copy=False)
        captured_indexer_margins[layer_idx] = margin_arr
      captured_ref_indexer_margins[slot_idx] = margin_arr
      captured_ref_indexer_ulp[slot_idx] = ulp_arr

    return selected_idxs

  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.object(ref_model.Gate, "forward", _gate_forward_traced))
    stack.enter_context(mock.patch.object(ref_model.Indexer, "forward", _indexer_forward_traced))
    logits_3d = forward_multimodal_logits(
        model,
        input_ids,
        images=images,
        token_types=token_types,
    )

  missing_layers = [
      idx
      for idx in range(n_layers)
      if (
          captured_topk[idx] is None
          or captured_gate_margins[idx] is None
          or captured_gate_ulp[idx] is None
          or captured_gate_pairs[idx] is None
      )
  ]
  if missing_layers:
    raise RuntimeError(f"Routing trace missed Gate execution on decoder layers: {missing_layers}")

  missing_idx_slots = [
      indexer_layer_ids[slot_i]
      for slot_i in range(n_idx_layers)
      if (
          captured_ref_indexer_topk[slot_i] is None
          or captured_ref_indexer_margins[slot_i] is None
          or captured_ref_indexer_ulp[slot_i] is None
      )
  ]
  if missing_idx_slots:
    raise RuntimeError(f"Routing trace missed Indexer execution on index-source layers: {missing_idx_slots}")

  topk_arrays = [arr for arr in captured_topk if arr is not None]
  margin_arrays = [arr for arr in captured_gate_margins if arr is not None]
  ulp_arrays = [arr for arr in captured_gate_ulp if arr is not None]
  pair_arrays = [arr for arr in captured_gate_pairs if arr is not None]

  ref_topk_np = np.stack(topk_arrays, axis=0)
  gate_margins_np = np.stack(margin_arrays, axis=0)
  gate_ulp_np = np.stack(ulp_arrays, axis=0)
  gate_pairs_np = np.stack(pair_arrays, axis=0)
  if ref_topk_np.shape[:2] != (n_layers, seq_len):
    raise RuntimeError(f"Unexpected ref_topk_experts shape {ref_topk_np.shape}; expected ({n_layers}, {seq_len}, top_k)")

  ref_indexer_layer_ids_np = np.asarray(indexer_layer_ids, dtype=np.int32)
  if n_idx_layers > 0:
    idx_topk_arrays = [arr for arr in captured_ref_indexer_topk if arr is not None]
    idx_margin_arrays = [arr for arr in captured_ref_indexer_margins if arr is not None]
    idx_ulp_arrays = [arr for arr in captured_ref_indexer_ulp if arr is not None]
    ref_indexer_topk_np = np.stack(idx_topk_arrays, axis=0).astype(np.int32, copy=False)
    ref_indexer_margins_np = np.stack(idx_margin_arrays, axis=0).astype(np.float32, copy=False)
    ref_indexer_ulp_np = np.stack(idx_ulp_arrays, axis=0).astype(np.float32, copy=False)
  else:
    ref_indexer_topk_np = np.empty((0, seq_len, 0), dtype=np.int32)
    ref_indexer_margins_np = np.empty((0, seq_len), dtype=np.float32)
    ref_indexer_ulp_np = np.empty((0, seq_len), dtype=np.float32)

  return logits_3d, RoutingTrace(
      ref_topk_experts=ref_topk_np,
      gate_margins=gate_margins_np,
      gate_boundary_experts=gate_pairs_np,
      indexer_margins=tuple(captured_indexer_margins),
      ref_gate_margins=gate_margins_np,
      ref_gate_boundary_ulp=gate_ulp_np,
      ref_indexer_layer_ids=ref_indexer_layer_ids_np,
      ref_indexer_topk=ref_indexer_topk_np,
      ref_indexer_margins=ref_indexer_margins_np,
      ref_indexer_boundary_ulp=ref_indexer_ulp_np,
  )
