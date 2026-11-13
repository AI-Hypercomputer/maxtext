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

"""Parallel CPU per-layer reference h_in/h_out and shared-attention cache capture across all 12 MM samples.

Captures teacher-forced layer boundary activations (`h_in`, `h_out`, `pre_mix_in`, `pre_mix_out`),
shared-attention source caches (`compress_kv`, `index_k`, `candidates`), and router/indexer scores
for all 40 decoder layers across all 12 multimodal golden samples (`L1-L4`, `M1-M8`), verifying
zero mismatches against `mm_v3` `ref_topk_experts` and `ref_indexer_topk` and `max_abs <= 1e-5`
against `mm_v3` `logits`.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import resource
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from typing import Any
from unittest import mock

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None and sys.modules.get("torch", -1) is not None

if HAS_TORCH:
  import torch
  import torch.nn.functional as F
  from safetensors import safe_open
  from torch import nn
  from transformers import AutoTokenizer

  from tests.utils.deepseek_v41_mm_golden import reference_loader
  from tests.utils.deepseek_v41_reference import image_processor
  from tests.utils.deepseek_v41_reference import model as ref_model
  from tests.utils.deepseek_v41_reference import multimodal, runner, vision
else:
  safe_open = None
  torch = None
  nn = None
  F = None
  AutoTokenizer = None
  reference_loader = None
  image_processor = None
  ref_model = None
  multimodal = None
  runner = None
  vision = None

ALL_40_LAYERS: tuple[int, ...] = tuple(range(40))
INDEXER_LAYERS: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
KV_SOURCE_LAYERS: tuple[int, ...] = (2, 8, 14, 20)
ENGRAM_LAYERS: tuple[int, ...] = (1, 14)

ALL_12_SAMPLES: tuple[str, ...] = (
    "L1",
    "L2",
    "L3",
    "L4",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
)

EXPECTED_MM_V3_SHA256: dict[str, str] = {
    "L1": "87c946317058429a6ce443b35ef042b0d3694fb7fbdda44f82d96c81233e8d50",
    "L2": "f61498ee831f3834d0598f2dd31f368f5a42c66257f51c167b98879f0b5e8f3f",
    "L3": "9b4aaef76cfee29e1f131a3beeb4fb7e513e4e885677a3f4d4946c145a901a8b",
    "L4": "d4218a5a0d27bd96584e2619c01f5bb7a60b48556bd33540034b3f6c13ab401a",
    "M1": "bdbe116d8c87887b23378c5fac3d2644021a85a83d68bfdfbf7010ed01ef73a7",
    "M2": "d6516fb91eab177b2e3264500353cfe3491b1e3537710681bd6e4219831ab6e8",
    "M3": "1b75164099338e15bd06eb7280e2309455e0b3a9780419c89a53c66ce319fc5d",
    "M4": "98b8ca6a309b481b7274663606d3a6c4315d116b4ff91a06969176d8261c2208",
    "M5": "9df1f6cff38752eadc995c1aa319a862310d04ad1fbe0d31110d9a0fffca89a9",
    "M6": "41ffb6de1ea06c67791a254c89cbc976291caa408514ce3c5f776c1c836cb5c6",
    "M7": "1cbcd36dab43a189647850a1ce36e5b704f73b84976bb354b9ab029932376f58",
    "M8": "01d7f853e97abcb5148e5deb63202637b8b8eb4f2f4a650442e37ea9ddebab18",
}

PARALLEL_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("48-75", ("M8", "M3")),
    ("76-95,160-167", ("M7", "L4")),
    ("168-195", ("M6", "M4", "L1")),
    ("196-223", ("M5", "M2", "M1", "L2", "L3")),
)

# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 1e-5.
LOGIT_SELF_CHECK_ATOL: float = 1e-5


@dataclasses.dataclass(frozen=True)
class LoadedGoldenSample:
  """Golden sample loaded and SHA256-verified from `mm_v3` `.npz`."""

  sample_id: str
  npz_path: pathlib.Path
  npz_sha256: str
  seq_len: int
  num_images: int
  grid_hw: tuple[int, int]
  span_starts: tuple[int, ...]
  tokens_1d: np.ndarray
  token_types_1d: np.ndarray
  positions_1d: np.ndarray
  patches_padded: np.ndarray
  image_masks: np.ndarray
  image_inputs: list[image_processor.ImageInput]
  golden_logits: np.ndarray
  ref_topk_experts_full: np.ndarray
  ref_gate_margins_full: np.ndarray
  ref_gate_boundary_ulp_full: np.ndarray
  ref_indexer_layer_ids_full: np.ndarray
  ref_indexer_topk_full: np.ndarray
  ref_indexer_margins_full: np.ndarray
  ref_indexer_boundary_ulp_full: np.ndarray


def _rss_gib() -> float:
  """Returns current process peak RSS in GiB."""
  ru = resource.getrusage(resource.RUSAGE_SELF)
  return float(ru.ru_maxrss) / (1024.0 * 1024.0)


def sha256_file(path: pathlib.Path) -> str:
  """Computes streaming hex SHA-256 digest of a file."""
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while True:
      chunk = f.read(1 << 20)
      if not chunk:
        break
      h.update(chunk)
  return h.hexdigest()


@contextlib.contextmanager
def _torch_threads_context(num_threads: int) -> Iterator[None]:
  """Temporarily sets `torch.set_num_threads(num_threads)` and restores on exit."""
  prev = torch.get_num_threads()
  torch.set_num_threads(num_threads)
  try:
    yield
  finally:
    torch.set_num_threads(prev)


def load_mm_v3_sample(npz_dir: pathlib.Path, sample_id: str) -> LoadedGoldenSample:
  """Loads and SHA256-verifies a sample `.npz` from `npz_dir`."""
  if sample_id not in EXPECTED_MM_V3_SHA256:
    raise ValueError(f"Unsupported sample_id {sample_id!r}; expected one of {sorted(EXPECTED_MM_V3_SHA256)}")
  npz_path = npz_dir / f"{sample_id}.npz"
  if not npz_path.exists():
    raise FileNotFoundError(f"Missing mm_v3 golden sample at {npz_path}")
  actual_sha = sha256_file(npz_path)
  expected_sha = EXPECTED_MM_V3_SHA256[sample_id]
  if actual_sha != expected_sha:
    raise ValueError(f"SHA256 mismatch for {npz_path}: expected {expected_sha}, got {actual_sha}")

  with np.load(npz_path, allow_pickle=False) as data:
    tokens_1d = np.asarray(data["tokens"], dtype=np.int32)
    token_types_1d = np.asarray(data["token_types"], dtype=np.int32)
    positions_1d = np.asarray(data["positions"], dtype=np.int32)
    patches_padded = np.asarray(data["patches"], dtype=np.float32)
    image_masks = np.asarray(data["image_masks"], dtype=np.int32)
    golden_logits = np.asarray(data["logits"], dtype=np.float32)
    seq_len = int(data["seq_len"])
    num_images = int(data["num_images"])
    grid_hw_arr = np.asarray(data["grid_hw"], dtype=np.int32)
    grid_hw = (int(grid_hw_arr[0]), int(grid_hw_arr[1]))
    span_starts = tuple(int(x) for x in np.asarray(data["span_starts"], dtype=np.int32).tolist())
    ref_topk_experts_full = np.asarray(data["ref_topk_experts"], dtype=np.int32)
    ref_gate_margins_full = np.asarray(data["ref_gate_margins"], dtype=np.float32)
    ref_gate_boundary_ulp_full = np.asarray(data["ref_gate_boundary_ulp"], dtype=np.float32)
    ref_indexer_layer_ids_full = np.asarray(data["ref_indexer_layer_ids"], dtype=np.int32)
    ref_indexer_topk_full = np.asarray(data["ref_indexer_topk"], dtype=np.int32)
    ref_indexer_margins_full = np.asarray(data["ref_indexer_margins"], dtype=np.float32)
    ref_indexer_boundary_ulp_full = np.asarray(data["ref_indexer_boundary_ulp"], dtype=np.float32)

  n_vit_h, n_vit_w = grid_hw
  n_llm_h = (n_vit_h + (-n_vit_h) % 3) // 3
  n_llm_w = (n_vit_w + (-n_vit_w) % 3) // 3
  span_len = image_processor.num_image_tokens(n_llm_h, n_llm_w)
  image_inputs: list[image_processor.ImageInput] = []
  for slot_idx in range(num_images):
    start = span_starts[slot_idx]
    patches_bf16 = (
        torch.from_numpy(patches_padded[slot_idx]).to(torch.bfloat16).view(n_vit_h * n_vit_w, 3, 14, 14).contiguous()
    )
    types_pt = torch.from_numpy(token_types_1d[start : start + span_len]).to(torch.int64)
    image_inputs.append(
        image_processor.ImageInput(
            start=start,
            patches=patches_bf16,
            n_vit_h=n_vit_h,
            n_vit_w=n_vit_w,
            types=types_pt,
        )
    )

  return LoadedGoldenSample(
      sample_id=sample_id,
      npz_path=npz_path,
      npz_sha256=actual_sha,
      seq_len=seq_len,
      num_images=num_images,
      grid_hw=grid_hw,
      span_starts=span_starts,
      tokens_1d=tokens_1d,
      token_types_1d=token_types_1d,
      positions_1d=positions_1d,
      patches_padded=patches_padded,
      image_masks=image_masks,
      image_inputs=image_inputs,
      golden_logits=golden_logits,
      ref_topk_experts_full=ref_topk_experts_full,
      ref_gate_margins_full=ref_gate_margins_full,
      ref_gate_boundary_ulp_full=ref_gate_boundary_ulp_full,
      ref_indexer_layer_ids_full=ref_indexer_layer_ids_full,
      ref_indexer_topk_full=ref_indexer_topk_full,
      ref_indexer_margins_full=ref_indexer_margins_full,
      ref_indexer_boundary_ulp_full=ref_indexer_boundary_ulp_full,
  )


if HAS_TORCH:
  _BYTE_ARANGE = torch.arange(256, dtype=torch.long)
  _FP4_BYTE_LUT = torch.stack(
      [
          runner._FP4_E2M1_VALUES[_BYTE_ARANGE & 0xF],  # pylint: disable=protected-access
          runner._FP4_E2M1_VALUES[_BYTE_ARANGE >> 4],  # pylint: disable=protected-access
      ],
      dim=-1,
  ).contiguous()
else:
  _BYTE_ARANGE = None
  _FP4_BYTE_LUT = None
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
    w_3d = F.embedding(weight.view(torch.uint8).int(), _FP4_BYTE_LUT).view(rows, s_cols, block_cols)  # pylint: disable=not-callable
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


def _dequantize_lazy_weight_fast(
    qweight: torch.Tensor,
    qscale: torch.Tensor,
    out_shape: tuple[int, ...],
    storage_dtype: torch.dtype,
) -> torch.Tensor:
  """Dequantizes a compressed FP8/FP4 weight into `storage_dtype` using `dequantize_into_fast`."""
  out = torch.empty(out_shape, dtype=storage_dtype, device=qweight.device)
  dequantize_into_fast(out, qweight, qscale)
  return out


if HAS_TORCH and reference_loader is not None:
  reference_loader._dequantize_lazy_weight = _dequantize_lazy_weight_fast  # pylint: disable=protected-access


def _is_full_model_ckpt_key(name: str) -> bool:
  """Returns True if checkpoint key `name` belongs to the 40-layer multimodal model (including head.weight)."""
  if name.startswith("mtp."):
    return False
  if name.startswith(("vision.", "aligner.")) or name in (
      "image_start",
      "image_end",
      "image_newline",
      "embed.weight",
      "norm.weight",
      "head.weight",
  ):
    return True
  if name.startswith("layers."):
    parts = name.split(".", 2)
    if len(parts) >= 3 and parts[1].isdigit():
      return int(parts[1]) in ALL_40_LAYERS
  return False


def _attach_sparse_engram_embedding(
    mod: ref_model.ParallelEngramEmbedding,
    row_ids: np.ndarray,
    qweight: torch.Tensor,
    qscale: torch.Tensor,
    orig_dtype: torch.dtype,
) -> None:
  """Attaches sparse row-cached FP8 weights/scales to `ParallelEngramEmbedding` for active rows."""
  empty_param = nn.Parameter(torch.empty(0, dtype=orig_dtype), requires_grad=False)
  mod.weight = empty_param
  row_ids_pt = torch.from_numpy(np.asarray(row_ids, dtype=np.int64))
  mod.register_buffer("_sparse_row_ids", row_ids_pt, persistent=False)
  mod.register_buffer("_sparse_qweight", qweight, persistent=False)
  mod.register_buffer("_sparse_qscale", qscale, persistent=False)
  mod._lazy_weight_dtype = orig_dtype  # pylint: disable=protected-access

  def _sparse_engram_forward(indices: torch.Tensor) -> torch.Tensor:
    if mod.weight.numel() > 0:
      return F.embedding(indices, mod.weight)
    flat_ids = indices.reshape(-1).long()
    pos = torch.searchsorted(mod._sparse_row_ids, flat_ids)  # pylint: disable=protected-access
    pos_clamped = pos.clamp(max=int(mod._sparse_row_ids.numel()) - 1)  # pylint: disable=protected-access
    if not bool(torch.equal(mod._sparse_row_ids[pos_clamped], flat_ids)):  # pylint: disable=protected-access
      raise KeyError("Engram lookup encountered a row ID not present in sparse Engram cache")
    sub_qw = mod._sparse_qweight.index_select(0, pos_clamped)  # pylint: disable=protected-access
    sub_qs = mod._sparse_qscale.index_select(0, pos_clamped)  # pylint: disable=protected-access
    flat_out = reference_loader._dequantize_lazy_weight(  # pylint: disable=protected-access
        sub_qw,
        sub_qs,
        (int(flat_ids.numel()), mod.dim),
        mod._lazy_weight_dtype,  # pylint: disable=protected-access
    )
    return flat_out.view(*indices.shape, mod.dim)

  mod.forward = _sparse_engram_forward  # type: ignore[method-assign]


def _read_sparse_engram_rows_from_safetensors(
    sf: Any,
    scale_sf: Any,
    weight_name: str,
    scale_name: str,
    row_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reads only the sorted `row_ids` from `weight_name` and `scale_name` via `sf.get_slice`."""
  w_slice = sf.get_slice(weight_name)
  s_slice = scale_sf.get_slice(scale_name)
  w_shape = tuple(w_slice.get_shape())
  s_shape = tuple(s_slice.get_shape())
  w_dtype = w_slice[:0].dtype
  s_dtype = s_slice[:0].dtype
  k_rows = int(row_ids.size)
  qweight = torch.empty((k_rows, w_shape[1]), dtype=w_dtype, device="cpu")
  qscale = torch.empty((k_rows, s_shape[1]), dtype=s_dtype, device="cpu")
  for idx, row_id in enumerate(row_ids.tolist()):
    r = int(row_id)
    qweight[idx].copy_(w_slice[r : r + 1].squeeze(0))
    qscale[idx].copy_(s_slice[r : r + 1].squeeze(0))
  return qweight, qscale


def load_full_40layer_reference_model(
    ckpt_dir: pathlib.Path,
    *,
    samples: Sequence[LoadedGoldenSample],
    max_seq_len: int = 2048,
) -> multimodal.MultimodalTransformer:
  """Builds and loads the 40-layer `MultimodalTransformer` with sparse Engram rows and real `model.head`."""
  hf_config = runner.load_hf_config(str(ckpt_dir))
  args = reference_loader.multimodal_model_args_from_hf_config(
      hf_config,
      dtype="bf16",
      max_seq_len=max_seq_len,
      max_batch_size=1,
      fp32_compute=True,
      dspark_block_size=0,
      n_mtp_layers=0,
  )
  tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

  def _lightweight_engram_embed_init(
      self_emb: ref_model.ParallelEngramEmbedding,
      num_embeddings: int,
      dim: int,
  ) -> None:
    nn.Module.__init__(self_emb)  # pylint: disable=unnecessary-dunder-call
    self_emb.num_embeddings = num_embeddings
    self_emb.dim = dim
    self_emb.part_num_embeddings = num_embeddings
    self_emb.vocab_start_idx = 0
    self_emb.vocab_end_idx = num_embeddings
    self_emb.weight = nn.Parameter(torch.empty((1, dim), dtype=ref_model.default_dtype))

  prev_default = torch.get_default_dtype()
  torch.set_default_dtype(torch.float32)
  try:
    with mock.patch.object(ref_model.ParallelEngramEmbedding, "__init__", _lightweight_engram_embed_init):
      text_args = dataclasses.replace(args, vision_n_layers=0)
      model = multimodal.MultimodalTransformer.__new__(multimodal.MultimodalTransformer)  # pylint: disable=no-value-for-parameter
      ref_model.Transformer.__init__(model, text_args, tokenizer=tokenizer)
      model.vision_enabled = args.vision_enabled
      model.vision = vision.ViT(args)
      model.aligner = vision.Aligner(args)
      model.image_start = nn.Parameter(torch.empty(args.dim))
      model.image_end = nn.Parameter(torch.empty(args.dim))
      model.image_newline = nn.Parameter(torch.empty(args.dim))
      for layer_id in ALL_40_LAYERS:
        layer = model.layers[layer_id]
        n_routed_experts, _ = args.get_moe_config(layer_id)
        layer.ffn.gate.bias_vl = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
  finally:
    torch.set_default_dtype(prev_default)
  model.eval()

  assert model.engram_hash is not None
  engram_required_rows: dict[int, np.ndarray] = {}
  with torch.no_grad():
    for lid in ENGRAM_LAYERS:
      blk_mod = model.layers[lid]
      assert isinstance(blk_mod, ref_model.Block) and blk_mod.engram is not None
      h_idx = blk_mod.engram.layer_hash_index
      row_union: set[int] = set(range(256))
      for s in samples:
        inp_pt = torch.from_numpy(s.tokens_1d).to(torch.int64).unsqueeze(0)
        eng_mask_pt = ~(torch.from_numpy(s.token_types_1d).unsqueeze(0) >= 0)
        hashes_pt = model.engram_hash(inp_pt, 0, eng_mask_pt)
        act_ids = np.unique(hashes_pt[:, :, h_idx, :].cpu().numpy().astype(np.int64))
        row_union.update(int(x) for x in act_ids.tolist())
      engram_required_rows[lid] = np.asarray(sorted(row_union), dtype=np.int64)

  headers_by_shard = reference_loader.read_safetensors_headers(str(ckpt_dir))
  key_to_shard_path: dict[str, str] = {}
  for shard_path, shard_header in headers_by_shard.items():
    for key in shard_header:
      key_to_shard_path[key] = shard_path

  expected_params = dict(model.named_parameters())
  populated_params: set[str] = set()

  with torch.no_grad(), _torch_threads_context(1):
    for shard_path in sorted(headers_by_shard):
      shard_keys = [k for k in sorted(headers_by_shard[shard_path]) if _is_full_model_ckpt_key(k)]
      if not shard_keys:
        continue
      is_engram_shard = any(k.endswith(".engram.embed.weight") for k in shard_keys)
      if not is_engram_shard and hasattr(os, "posix_fadvise"):
        fd = os.open(shard_path, os.O_RDONLY)
        try:
          os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
          os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
        finally:
          os.close(fd)
      with safe_open(shard_path, framework="pt", device="cpu") as sf:
        for name in shard_keys:
          if not reference_loader.is_multimodal_reference_key(name):
            continue
          if name not in expected_params:
            raise ValueError(f"Unexpected checkpoint tensor {name!r} not in model")
          scale_name = f"{name[:-len('.weight')]}.scale" if name.endswith(".weight") else None
          has_scale = scale_name is not None and scale_name in key_to_shard_path
          if has_scale:
            assert scale_name is not None
            populated_params.add(name)
            scale_shard = key_to_shard_path[scale_name]
            if name.endswith(".engram.embed.weight"):
              lid_val = int(name.split(".")[1])
              req_rows = engram_required_rows[lid_val]
              emb_mod = model.layers[lid_val].engram.embed
              orig_dtype = emb_mod.weight.dtype
              if scale_shard == shard_path:
                qw, qs = _read_sparse_engram_rows_from_safetensors(sf, sf, name, scale_name, req_rows)
              else:
                with safe_open(scale_shard, framework="pt", device="cpu") as scale_sf:
                  qw, qs = _read_sparse_engram_rows_from_safetensors(sf, scale_sf, name, scale_name, req_rows)
              _attach_sparse_engram_embedding(emb_mod, req_rows, qw, qs, orig_dtype)
              continue
            qweight = reference_loader._read_owned_tensor(sf, name)  # pylint: disable=protected-access
            if scale_shard == shard_path:
              qscale = reference_loader._read_owned_tensor(sf, scale_name)  # pylint: disable=protected-access
            else:
              with safe_open(scale_shard, framework="pt", device="cpu") as scale_sf:
                qscale = reference_loader._read_owned_tensor(scale_sf, scale_name)  # pylint: disable=protected-access
            reference_loader._attach_lazy_quantized_weight(model, name, qweight, qscale)  # pylint: disable=protected-access
          else:
            populated_params.add(name)
            reference_loader._copy_param_from_safetensors(expected_params[name].data, sf, name)  # pylint: disable=protected-access

  reference_loader._install_lazy_attention_wo_a_wrappers(model)  # pylint: disable=protected-access

  missing_params = sorted(set(expected_params) - populated_params)
  if missing_params:
    raise ValueError(f"Checkpoint load missed {len(missing_params)} parameters: {missing_params[:20]}")

  return model


def capture_sample_layerwise(
    model: multimodal.MultimodalTransformer,
    sample: LoadedGoldenSample,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """Runs 40-layer forward pass for `sample`, captures all layer boundaries, and runs self-checks."""
  input_ids = torch.from_numpy(sample.tokens_1d).to(torch.int64).unsqueeze(0)
  token_types = torch.from_numpy(sample.token_types_1d).to(torch.int64).unsqueeze(0)
  image_mask = token_types >= 0
  seqlen = sample.seq_len

  npz_dict: dict[str, np.ndarray] = {
      "tokens": sample.tokens_1d[None, :].astype(np.int32),
      "token_types": sample.token_types_1d[None, :].astype(np.int32),
      "positions": sample.positions_1d[None, :].astype(np.int32),
      "image_mask": image_mask.cpu().numpy().astype(np.bool_),
      "image_masks": sample.image_masks.astype(np.int32),
      "seq_len": np.int32(seqlen),
  }

  gate_to_lid = {id(model.layers[lid].ffn.gate): lid for lid in ALL_40_LAYERS}
  indexer_to_lid = {id(model.layers[lid].attn.indexer): lid for lid in INDEXER_LAYERS}

  all_topk_experts = np.zeros((40, seqlen, 6), dtype=np.int32)
  all_gate_margins = np.zeros((40, seqlen), dtype=np.float32)
  all_gate_ulp = np.zeros((40, seqlen), dtype=np.float32)
  all_router_logits = np.zeros((40, seqlen, 384), dtype=np.float32)
  all_router_scores_unbiased = np.zeros((40, seqlen, 384), dtype=np.float32)

  idx_topk_list: dict[int, np.ndarray] = {}

  def _gate_traced(
      self_gate: ref_model.Gate,
      x: torch.Tensor,
      img_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    lid = gate_to_lid[id(self_gate)]
    scores = ref_model.linear(x.float(), self_gate.weight.float()) / self_gate.gate_temp
    if self_gate.score_func == "softmax":
      scores = scores.softmax(dim=-1)
    elif self_gate.score_func == "sigmoid":
      scores = scores.sigmoid()
    else:
      scores = F.softplus(scores).sqrt()  # pylint: disable=not-callable
    bias = self_gate.bias
    if img_mask is not None and self_gate.bias_vl is not None:
      bias = torch.where(img_mask.unsqueeze(-1), self_gate.bias_vl, bias)
    selection_scores = scores + bias
    indices = selection_scores.topk(self_gate.topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if self_gate.norm_topk_prob and self_gate.topk > 1:
      weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * self_gate.route_scale

    top_vals = selection_scores.float().topk(self_gate.topk + 1, dim=-1, largest=True, sorted=True).values
    s_k = top_vals[:, self_gate.topk - 1].detach().cpu().numpy().astype(np.float32, copy=True)
    s_kp1 = top_vals[:, self_gate.topk].detach().cpu().numpy().astype(np.float32, copy=True)

    all_topk_experts[lid] = indices.detach().cpu().numpy().astype(np.int32).reshape(seqlen, self_gate.topk)
    all_gate_margins[lid] = (s_k - s_kp1).astype(np.float32).reshape(seqlen)
    all_gate_ulp[lid] = np.abs(np.spacing(s_k)).astype(np.float32).reshape(seqlen)
    all_router_logits[lid] = selection_scores.detach().float().cpu().numpy().reshape(seqlen, -1)
    all_router_scores_unbiased[lid] = scores.detach().float().cpu().numpy().reshape(seqlen, -1)
    return weights, indices

  def _indexer_traced(  # pylint: disable=too-many-positional-arguments
      self_idx: ref_model.Indexer,
      x: torch.Tensor,
      qr: torch.Tensor,
      latent: torch.Tensor | None,
      start_pos: int,
      offset: int,
  ) -> torch.Tensor:
    lid = indexer_to_lid[id(self_idx)]
    pfx = f"l{lid}"
    assert self_idx.freqs_cis is not None
    bsz, seq_l, _ = x.size()
    ratio, rd, end_pos = self_idx.compress_ratio, self_idx.rope_head_dim, start_pos + seq_l

    if self_idx.owns_k and latent is not None:
      freqs = (
          self_idx.freqs_cis[: seq_l - seq_l % ratio : ratio]
          if start_pos == 0
          else self_idx.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
      )
      k = self_idx.k_norm(self_idx.wk(latent))
      ref_model.apply_rotary_emb(k[..., -rd:], freqs)
      ref_model.fp4_act_quant(k, ref_model.fp4_block_size, True)
      self_idx.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
    if self_idx.owns_k:
      ref_model.shared_attn.index_k = self_idx.k_cache

    q = self_idx.wq_b(qr).unflatten(-1, (self_idx.n_local_heads, self_idx.index_head_dim))
    ref_model.apply_rotary_emb(q[..., -rd:], self_idx.freqs_cis[start_pos:end_pos])
    ref_model.fp4_act_quant(q, ref_model.fp4_block_size, True)

    assert ref_model.shared_attn.index_k is not None
    index_k = ref_model.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self_idx.weights_proj(x) * (self_idx.softmax_scale * self_idx.n_heads**-0.5)
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    raw_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
    index_score = raw_score.clone()

    npz_dict[f"{pfx}_raw_index_score"] = raw_score.detach().float().cpu().numpy()

    compress_lens = (torch.arange(1, seq_l + 1, device=x.device) // ratio).unsqueeze(-1)
    index_score.masked_fill_(torch.arange(seq_l // ratio, device=x.device) >= compress_lens, -torch.inf)

    if self_idx.is_candidate_source:
      ref_model.shared_attn.candidates = ref_model.select_candidate_blocks(
          index_score,
          compress_lens,
          self_idx.candidate_topk_blocks,
          self_idx.candidate_block_size,
      )
    elif self_idx.uses_candidates:
      assert ref_model.shared_attn.candidates is not None
      index_score = index_score.masked_fill(~ref_model.shared_attn.candidates, -torch.inf)

    topk = min(self_idx.index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    unoffset_selected = torch.where(idxs < compress_lens, idxs, -1).int()
    selected_idxs = torch.where(idxs < compress_lens, idxs + offset, -1).int()

    padded_unoffset = np.full((1, seq_l, int(self_idx.index_topk)), -1, dtype=np.int32)
    padded_offset = np.full((1, seq_l, int(self_idx.index_topk)), -1, dtype=np.int32)
    if topk > 0:
      padded_unoffset[:, :, :topk] = unoffset_selected.detach().cpu().numpy().astype(np.int32, copy=True)
      padded_offset[:, :, :topk] = selected_idxs.detach().cpu().numpy().astype(np.int32, copy=True)
    npz_dict[f"{pfx}_ref_indexer_topk"] = padded_unoffset
    npz_dict[f"{pfx}_ref_indexer_topk_offset"] = padded_offset

    margin_arr = np.full((seq_l,), np.nan, dtype=np.float32)
    ulp_arr = np.full((seq_l,), np.nan, dtype=np.float32)
    if int(index_score.size(-1)) > self_idx.index_topk:
      scores_2d = index_score[0].float()
      finite_counts = torch.isfinite(scores_2d).sum(dim=-1)
      active_mask = finite_counts > self_idx.index_topk
      if bool(active_mask.any().item()):
        top_vals = scores_2d[active_mask].topk(self_idx.index_topk + 1, dim=-1, largest=True, sorted=True).values
        z_k = top_vals[:, self_idx.index_topk - 1].detach().cpu().numpy().astype(np.float32, copy=True)
        z_kp1 = top_vals[:, self_idx.index_topk].detach().cpu().numpy().astype(np.float32, copy=True)
        active_np = active_mask.cpu().numpy()
        margin_arr[active_np] = (z_k - z_kp1).astype(np.float32)
        ulp_arr[active_np] = np.abs(np.spacing(z_k)).astype(np.float32)
    npz_dict[f"{pfx}_ref_indexer_margins"] = margin_arr.reshape(1, seq_l)
    npz_dict[f"{pfx}_ref_indexer_boundary_ulp"] = ulp_arr.reshape(1, seq_l)
    idx_topk_list[lid] = padded_offset[0]
    return selected_idxs

  handles = []
  for lid in ALL_40_LAYERS:
    blk = model.layers[lid]
    assert isinstance(blk, ref_model.Block)
    pfx = f"l{lid}"
    if blk.engram is not None:

      def _make_engram_pre_hook(p: str):
        def _engram_pre_hook(_m: nn.Module, inp: tuple[Any, ...]) -> None:
          npz_dict[f"{p}_h_in"] = inp[0].detach().float().cpu().numpy()

        return _engram_pre_hook

      handles.append(blk.engram.register_forward_pre_hook(_make_engram_pre_hook(pfx)))

    def _make_blk_pre_hook(p: str):
      def _blk_pre_hook(_m: nn.Module, inp: tuple[Any, ...]) -> None:
        if f"{p}_h_in" not in npz_dict:
          npz_dict[f"{p}_h_in"] = inp[0].detach().float().cpu().numpy()
        npz_dict[f"{p}_pre_mix_in"] = inp[2].detach().float().cpu().numpy()

      return _blk_pre_hook

    def _make_blk_post_hook(l_id: int, p: str):
      def _blk_post_hook(
          _m: nn.Module,
          _inp: tuple[Any, ...],
          out: tuple[torch.Tensor, torch.Tensor],
      ) -> None:
        npz_dict[f"{p}_h_out"] = out[0].detach().float().cpu().numpy()
        npz_dict[f"{p}_pre_mix_out"] = out[1].detach().float().cpu().numpy()
        blk_mod = model.layers[l_id]
        ratio = blk_mod.attn.compress_ratio
        if ratio > 0:
          comp_len = seqlen // ratio
          if blk_mod.attn.is_kv_source:
            assert ref_model.shared_attn.compress_kv is not None
            npz_dict[f"{p}_compress_kv_out"] = (
                ref_model.shared_attn.compress_kv[:1, :comp_len].detach().float().cpu().numpy().copy()
            )
          if blk_mod.attn.is_index_source and blk_mod.attn.indexer.owns_k:
            assert ref_model.shared_attn.index_k is not None
            npz_dict[f"{p}_index_k_out"] = (
                ref_model.shared_attn.index_k[:1, :comp_len].detach().float().cpu().numpy().copy()
            )
          if blk_mod.attn.is_index_source and blk_mod.attn.indexer.is_candidate_source:
            assert ref_model.shared_attn.candidates is not None
            npz_dict[f"{p}_candidates_mask_out"] = (
                ref_model.shared_attn.candidates.detach().cpu().numpy().astype(np.bool_, copy=True)
            )

      return _blk_post_hook

    handles.append(blk.register_forward_pre_hook(_make_blk_pre_hook(pfx)))
    handles.append(blk.register_forward_hook(_make_blk_post_hook(lid, pfx)))

  assert model.vision is not None
  patch_dtype = model.vision.patch_embed.proj.weight.dtype
  cast_imgs = [dataclasses.replace(img, patches=img.patches.to(dtype=patch_dtype)) for img in sample.image_inputs]

  try:
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(ref_model.Gate, "forward", _gate_traced))
      stack.enter_context(mock.patch.object(ref_model.Indexer, "forward", _indexer_traced))
      with torch.no_grad():
        logits_pt = model(input_ids, start_pos=0, images=[cast_imgs], token_types=token_types, full_logits=True)
  finally:
    for h_hook in handles:
      h_hook.remove()

  logits_np = logits_pt.detach().float().cpu().numpy()
  npz_dict["all_topk_experts"] = all_topk_experts
  npz_dict["all_gate_margins"] = all_gate_margins
  npz_dict["all_gate_ulp"] = all_gate_ulp
  npz_dict["all_router_logits"] = all_router_logits
  npz_dict["all_router_scores_unbiased"] = all_router_scores_unbiased

  for lid in range(39):
    if not np.array_equal(npz_dict[f"l{lid}_h_out"], npz_dict[f"l{lid + 1}_h_in"]):
      raise RuntimeError(f"{sample.sample_id}: l{lid}_h_out != l{lid + 1}_h_in")
    if not np.array_equal(npz_dict[f"l{lid}_pre_mix_out"], npz_dict[f"l{lid + 1}_pre_mix_in"]):
      raise RuntimeError(f"{sample.sample_id}: l{lid}_pre_mix_out != l{lid + 1}_pre_mix_in")

  golden_topk = sample.ref_topk_experts_full.astype(np.int32)
  moe_mismatches = int(np.sum(all_topk_experts != golden_topk))
  moe_margin_max_diff = float(np.max(np.abs(all_gate_margins - sample.ref_gate_margins_full)))
  per_layer_moe_mismatches = {
      int(lid): int(np.sum(all_topk_experts[lid] != golden_topk[lid]))
      for lid in range(40)
      if int(np.sum(all_topk_experts[lid] != golden_topk[lid])) > 0
  }

  idx_topk_stacked = np.stack([idx_topk_list[lid] for lid in INDEXER_LAYERS], axis=0)
  golden_idx_topk = sample.ref_indexer_topk_full.astype(np.int32)
  idx_mismatches = int(np.sum(idx_topk_stacked != golden_idx_topk))
  per_layer_idx_mismatches = {
      int(INDEXER_LAYERS[i]): int(np.sum(idx_topk_stacked[i] != golden_idx_topk[i]))
      for i in range(len(INDEXER_LAYERS))
      if int(np.sum(idx_topk_stacked[i] != golden_idx_topk[i])) > 0
  }

  logits_max_abs_diff = float(np.max(np.abs(logits_np[0].astype(np.float64) - sample.golden_logits.astype(np.float64))))

  verification = {
      "sample_id": sample.sample_id,
      "seq_len": seqlen,
      "num_images": sample.num_images,
      "moe_topk_exact_match": moe_mismatches == 0,
      "moe_topk_mismatch_count": moe_mismatches,
      "per_layer_moe_mismatches": per_layer_moe_mismatches,
      "per_layer_idx_mismatches": per_layer_idx_mismatches,
      "moe_margin_max_abs_diff": moe_margin_max_diff,
      "indexer_topk_exact_match": idx_mismatches == 0,
      "indexer_topk_mismatch_count": idx_mismatches,
      "logits_max_abs_diff_vs_golden": logits_max_abs_diff,
      "logits_within_1e5": logits_max_abs_diff <= LOGIT_SELF_CHECK_ATOL,
      "layer_boundary_continuity_verified": True,
  }
  print(f"[Verify {sample.sample_id}] {json.dumps(verification)}", flush=True)
  if moe_mismatches != 0 or idx_mismatches != 0 or logits_max_abs_diff > LOGIT_SELF_CHECK_ATOL:
    raise RuntimeError(f"Self-check failed on {sample.sample_id}: {verification}")

  return npz_dict, verification


def run_worker(  # pylint: disable=too-many-positional-arguments
    ckpt_dir: pathlib.Path,
    mm_v3_dir: pathlib.Path,
    out_dir: pathlib.Path,
    sample_ids: Sequence[str],
    num_threads: int,
    worker_id: int,
    gcs_prefix: str | None,
) -> None:
  """Executes a single worker process capturing `sample_ids`."""
  t_worker_start = time.time()
  out_dir.mkdir(parents=True, exist_ok=True)
  samples = [load_mm_v3_sample(mm_v3_dir, sid) for sid in sample_ids]
  print(
      f"[Worker {worker_id}] Loaded {len(samples)} samples: "
      f"{[(s.sample_id, s.seq_len) for s in samples]} (threads={num_threads})",
      flush=True,
  )

  t_load = time.time()
  model = load_full_40layer_reference_model(ckpt_dir, samples=samples, max_seq_len=2048)
  model_load_s = round(time.time() - t_load, 2)
  print(
      f"[Worker {worker_id}] Loaded 40-layer model in {model_load_s}s (RSS={_rss_gib():.1f} GiB)",
      flush=True,
  )

  worker_summary: dict[str, Any] = {
      "worker_id": worker_id,
      "sample_ids": list(sample_ids),
      "num_threads": num_threads,
      "model_load_sec": model_load_s,
      "samples": {},
  }

  with _torch_threads_context(num_threads):
    for sample in samples:
      t_s = time.time()
      npz_dict, ver = capture_sample_layerwise(model, sample)
      fwd_s = round(time.time() - t_s, 2)
      out_npz = out_dir / f"{sample.sample_id}_ref_layerwise.npz"
      t_save = time.time()
      np.savez(out_npz, **npz_dict)
      save_s = round(time.time() - t_save, 2)
      sha = sha256_file(out_npz)
      size_bytes = out_npz.stat().st_size
      del npz_dict

      if gcs_prefix:
        gcs_dst = f"{gcs_prefix.rstrip("/")}/{out_npz.name}"
        subprocess.run(["gcloud", "storage", "cp", str(out_npz), gcs_dst], check=True)

      sample_record = {
          **ver,
          "npz_filename": out_npz.name,
          "sha256": sha,
          "size_bytes": size_bytes,
          "forward_sec": fwd_s,
          "save_sec": save_s,
          "rss_gib": round(_rss_gib(), 2),
      }
      worker_summary["samples"][sample.sample_id] = sample_record
      print(
          f"[Worker {worker_id}] Finished {sample.sample_id}: {out_npz.name} "
          f"({size_bytes / (1024**2):.1f} MiB, sha256={sha[:16]}..., fwd={fwd_s}s, save={save_s}s)",
          flush=True,
      )

  worker_summary["total_worker_sec"] = round(time.time() - t_worker_start, 2)
  worker_json = out_dir / f"worker_{worker_id}_manifest.json"
  worker_json.write_text(json.dumps(worker_summary, indent=2) + "\n", encoding="utf-8")
  print(f"[Worker {worker_id}] Wrote {worker_json} in {worker_summary["total_worker_sec"]}s", flush=True)


def launch_parallel_workers(
    ckpt_dir: pathlib.Path,
    mm_v3_dir: pathlib.Path,
    out_dir: pathlib.Path,
    num_threads: int,
    gcs_prefix: str | None,
) -> None:
  """Launches 4 taskset-pinned worker processes across cores 48-95,160-223 and writes capture_manifest.json."""
  t0 = time.time()
  out_dir.mkdir(parents=True, exist_ok=True)
  procs: list[tuple[int, str, tuple[str, ...], subprocess.Popen[str], pathlib.Path]] = []
  script_path = pathlib.Path(__file__).resolve()

  for wid, (cpu_list, bucket_samples) in enumerate(PARALLEL_BUCKETS):
    log_path = out_dir / f"worker_{wid}.log"
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["MKL_NUM_THREADS"] = str(num_threads)
    cmd = [
        "taskset",
        "-c",
        cpu_list,
        sys.executable,
        "-u",
        str(script_path),
        f"--ckpt_dir={ckpt_dir}",
        f"--mm_v3_dir={mm_v3_dir}",
        f"--out_dir={out_dir}",
        f"--samples={",".join(bucket_samples)}",
        f"--num_threads={num_threads}",
        f"--worker_id={wid}",
    ]
    if gcs_prefix:
      cmd.append(f"--gcs_prefix={gcs_prefix}")
    log_file = open(log_path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)  # pylint: disable=consider-using-with
    procs.append((wid, cpu_list, bucket_samples, proc, log_path))
    print(f"[Launcher] Started worker {wid} (PID={proc.pid}, CPUs={cpu_list}, samples={bucket_samples})", flush=True)

  failed = []
  for wid, cpu_list, bucket_samples, proc, log_path in procs:
    rc = proc.wait()
    print(f"[Launcher] Worker {wid} exited with code {rc} (log={log_path})", flush=True)
    if rc != 0:
      failed.append((wid, rc, str(log_path)))

  if failed:
    raise RuntimeError(f"One or more capture workers failed: {failed}")

  manifest: dict[str, Any] = {
      "schema_version": 1,
      "total_wall_sec": round(time.time() - t0, 2),
      "num_workers": len(PARALLEL_BUCKETS),
      "threads_per_worker": num_threads,
      "workers": {},
      "samples": {},
  }
  for wid in range(len(PARALLEL_BUCKETS)):
    w_path = out_dir / f"worker_{wid}_manifest.json"
    w_data = json.loads(w_path.read_text(encoding="utf-8"))
    manifest["workers"][str(wid)] = w_data
    for sid, s_rec in w_data["samples"].items():
      manifest["samples"][sid] = s_rec

  if sorted(manifest["samples"].keys()) != sorted(ALL_12_SAMPLES):
    raise RuntimeError(f"Missing samples in manifest: got {sorted(manifest["samples"].keys())}")

  manifest["total_tokens"] = sum(int(v["seq_len"]) for v in manifest["samples"].values())
  manifest["total_size_bytes"] = sum(int(v["size_bytes"]) for v in manifest["samples"].values())

  manifest_path = out_dir / "capture_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if gcs_prefix:
    subprocess.run(
        ["gcloud", "storage", "cp", str(manifest_path), f"{gcs_prefix.rstrip("/")}/capture_manifest.json"],
        check=True,
    )
  print(
      f"[Launcher] All 12 samples captured & verified in {manifest["total_wall_sec"]}s -> {manifest_path}",
      flush=True,
  )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments."""
  parser = argparse.ArgumentParser(description="Capture 12-sample per-layer reference h_in/h_out bundles.")
  parser.add_argument("--ckpt_dir", type=pathlib.Path, required=True, help="Path to HF checkpoint directory.")
  parser.add_argument("--mm_v3_dir", type=pathlib.Path, required=True, help="Path to mm_v3 golden .npz directory.")
  parser.add_argument("--out_dir", type=pathlib.Path, required=True, help="Output directory for .npz bundles.")
  parser.add_argument("--samples", type=str, default="all", help="Comma-separated sample IDs or all.")
  parser.add_argument("--num_threads", type=int, default=28, help="PyTorch threads per worker.")
  parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for single-worker invocation.")
  parser.add_argument("--launch_parallel_4", action="store_true", help="Launch 4 taskset-pinned workers in parallel.")
  parser.add_argument("--gcs_prefix", type=str, default="", help="Optional gs:// prefix to upload each .npz + manifest.")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  """Main entry point."""
  args = parse_args(argv)
  gcs_prefix = args.gcs_prefix if args.gcs_prefix else None
  if args.launch_parallel_4:
    launch_parallel_workers(
        ckpt_dir=args.ckpt_dir,
        mm_v3_dir=args.mm_v3_dir,
        out_dir=args.out_dir,
        num_threads=args.num_threads,
        gcs_prefix=gcs_prefix,
    )
    return

  if args.samples == "all":
    sample_ids = list(ALL_12_SAMPLES)
  else:
    sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]
  run_worker(
      ckpt_dir=args.ckpt_dir,
      mm_v3_dir=args.mm_v3_dir,
      out_dir=args.out_dir,
      sample_ids=sample_ids,
      num_threads=args.num_threads,
      worker_id=args.worker_id,
      gcs_prefix=gcs_prefix,
  )


if __name__ == "__main__":
  main()
