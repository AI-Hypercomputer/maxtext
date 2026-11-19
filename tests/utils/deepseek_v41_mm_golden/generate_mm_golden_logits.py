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

"""Generates deterministic multimodal golden logits for DeepSeek-V4.1-Flash.

Supports both the full 749B FP8/FP4 release checkpoint (`--checkpoint-dir`) and a
self-contained quantized mini-twin mode (`--mini`) exercising the identical
safetensors header audit, lazy/eager FP8/FP4 loader, R9 patch dtype cast, 12-sample
builder (`L1..L4`, `M1..M8`), `mm_logit.pkl` export, `manifest.json` provenance
recording, and optional `bf16` / `1-ULP` diagnostics.
"""

import argparse
import ast
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import inspect
import io
import json
import os
import pathlib
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageOps
import pyarrow.parquet as pq
from safetensors.torch import save_file
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tests.utils.deepseek_v41_mm_golden import reference_loader
from tests.utils.deepseek_v41_reference import image_processor
from tests.utils.deepseek_v41_reference import model as ref_model
from tests.utils.deepseek_v41_reference import multimodal
from tests.utils.deepseek_v41_reference import quantize_checkpoint

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_REF_DIR = _REPO_ROOT / "tests" / "utils" / "deepseek_v41_reference"
_TOOL_DIR = pathlib.Path(__file__).resolve().parent
_PROVENANCE_PATH = _REF_DIR / "provenance.json"
_TEST_IMAGE_PATH = _REPO_ROOT / "tests" / "assets" / "test_image.jpg"
_TEST_IMAGE_REASONING_PATH = _REPO_ROOT / "tests" / "assets" / "test_image_reasoning.jpg"
_C4_VAL_PARQUET_PATH = (
    _REPO_ROOT
    / "tests"
    / "assets"
    / "local_datasets"
    / "c4_en_dataset_minimal"
    / "hf"
    / "c4"
    / "c4-validation-00000-of-01637.parquet"
)

_SKIP_AST_FIELDS = frozenset({"ctx", "type_comment"})
_IMAGE_TAG_RE = re.compile(r"<image>(.*?)</image>", re.DOTALL)
_DEFAULT_HF_TOKENIZER_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
_MAX_IMAGE_SLOTS = 4
_FULL_GRID_HW = (39, 39)
_FULL_SPAN_TOKENS = 184
_FULL_MIN_PIXELS_SIDE = 544
_MINI_GRID_HW = (6, 6)
_MINI_SPAN_TOKENS = 8
_MIN_C4_POOL_TOKENS = 6144
_ANTI_COPY_NGRAM_WINDOW = 16

GOLDEN_VERSIONS = ("mm_v1", "mm_v2", "mm_v3")
PRESCREEN_SPEC_VERSION = "option_b1_blind_v1"
PRESCREEN_SPEC_SHA256 = "6f70fbe7f5b9696f0160fb2258279647470f1aaa34da78d3ff4b296e12dfeff5"
OPTION_C_SPEC_VERSION = "mm_v3_option_c_frozen_topk_v1"
OPTION_C_SPEC_SHA256 = "70974d6fbb40df2482397f1f438485af7bfbf0b6a6490324160333c63f3095b0"
MM_V3_DISTINCT_CROP_SOURCE_IDS: tuple[str, ...] = (
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
MM_V3_EXPECTED_DISTINCT_NON_L2_RANKS: tuple[int, ...] = (
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
_DEFAULT_BASE_ASSET_SIZES: dict[str, tuple[int, int]] = {
    "B0": (512, 341),
    "B1": (1024, 701),
    "B2": (450, 308),
    "B3": (640, 536),
}
REAL_INITIAL_SCREEN_SAMPLE_IDS = ("L2", "L3", "M2")

B0_SHA256 = "6686b937bdb28e2c9804c435bd63a7ab56ac1531ffc9b482f2b8a75aed8770f5"
B1_SHA256 = "5df896a4a07e127281c60fc957f8b3d73f4735b3258a0bf762b4383557f8fa9a"
B2_SHA256 = "251fe7ad58f44db139a6af5ed489e027eca93ae60878b83f3e709c7d6f0112a1"
B3_SHA256 = "399f4f5879cbba9533741dc589056e51f93a19317bb68a8cd6ac3aaf1fe0f420"
EXPECTED_BASE_ASSET_SHA256 = {
    "B0": B0_SHA256,
    "B1": B1_SHA256,
    "B2": B2_SHA256,
    "B3": B3_SHA256,
}
CANDIDATE_BASE_IDS = ("B0", "B1", "B2", "B3")
CANDIDATE_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.0, 0.5),
    (1.0, 0.5),
)
_CANDIDATE_CROP_SIDE = 544
_DUPLICATE_TEST_IMAGE_CROP_ID = "crop_544:B3:anchor_0:mirror_0"
_EARLY_ROUTER_LAYER_LIMIT = 20
_DEFAULT_MM_V2_PRIMARY_THREADS = 96
_DEFAULT_MM_V2_PERTURBATION_THREADS = 64
_DEFAULT_MINI_PRIMARY_THREADS = 2
_DEFAULT_MINI_PERTURBATION_THREADS = 1

# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 2.0e-6.
PRESCREEN_TAU_GATE_L0_19 = 2.0e-6
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 0.0.
PRESCREEN_MIN_GATE_ALL_GT = 0.0
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 0.0.
PRESCREEN_MIN_INDEXER_L0_19_GT = 0.0
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 0.008.
PRESCREEN_MAX_KL_LE = 0.008
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 1.0e-4.
PRESCREEN_MEAN_KL_LE = 1.0e-4
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 5.0e-4.
PRESCREEN_P95_KL_LE = 5.0e-4
# TODO(yaoyuchen): Make threshold configurable via flag/user input defaulting to 0.995.
PRESCREEN_TOP1_AGREEMENT_GE = 0.995

LOGIT_SAMPLE_IDS = ("L1", "L2", "L3", "L4")
PREFILL_SAMPLE_IDS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
ALL_SAMPLE_IDS = LOGIT_SAMPLE_IDS + PREFILL_SAMPLE_IDS
DIAGNOSTIC_SAMPLE_IDS = ("L4", "M8")

_TOOL_FILENAMES = (
    "__init__.py",
    "reference_loader.py",
    "generate_mm_golden_logits.py",
)

_ARG_TO_HF_TEXT_KEY = {
    "dim": "hidden_size",
    "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "n_activated_experts": "num_experts_per_tok",
    "score_func": "scoring_func",
    "route_scale": "routed_scaling_factor",
    "rope_head_dim": "qk_rope_head_dim",
    "norm_eps": "rms_norm_eps",
    "window_size": "sliding_window",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "engram_pad_id": "engram_pad_token_id",
}

_MINI_TWO_IMAGE_PROMPT = (
    "Compare first image<image>synth:201</image>and second image"
    "<image>synth:202</image>and identify each ingredient and edible part."
)


@dataclasses.dataclass(frozen=True)
class SampleSpec:
  """Specification for one deterministic multimodal golden sample."""

  sample_id: str
  target_seq_len: int | None
  num_images: int
  placement: str
  image_sources: tuple[str, ...]
  c4_offset: int


@dataclasses.dataclass(frozen=True)
class PreparedImage:
  """Preprocessed ViT patch tensor and provenance metadata for one image."""

  source_id: str
  source_sha256: str
  patches_bf16: torch.Tensor
  patches_flat_fp32: np.ndarray
  n_vit_h: int
  n_vit_w: int
  token_types: torch.Tensor
  prepared_png_sha256: str
  base_asset_sha256: str


@dataclasses.dataclass(frozen=True)
class CandidateBaseAsset:
  """Loaded real photographic base image (`B0..B3`) for the `mm_v2` 72-crop pool."""

  base_id: str
  path: pathlib.Path
  raw_bytes: bytes
  sha256: str
  rgb_image: Image.Image


@dataclasses.dataclass(frozen=True)
class CandidateCropSpec:
  """Deterministic crop specification in the fixed 72-candidate `mm_v2` queue."""

  rank: int
  source_id: str
  base_id: str
  anchor_idx: int
  centering: tuple[float, float]
  mirror: int
  base_asset_sha256: str
  crop_box: tuple[int, int, int, int] = (0, 0, _CANDIDATE_CROP_SIDE, _CANDIDATE_CROP_SIDE)


@dataclasses.dataclass(frozen=True)
class PrescreenCriteria:
  """Frozen Option B-1 `C1` + `C2` thresholds recorded in `manifest.json`."""

  tau_gate_l0_19_image_tokens: float = PRESCREEN_TAU_GATE_L0_19
  min_gate_l0_39_image_tokens_gt: float = PRESCREEN_MIN_GATE_ALL_GT
  min_indexer_l0_19_image_tokens_gt: float = PRESCREEN_MIN_INDEXER_L0_19_GT
  perturbation_max_kl_le: float = PRESCREEN_MAX_KL_LE
  perturbation_mean_kl_le: float = PRESCREEN_MEAN_KL_LE
  perturbation_p95_kl_le: float = PRESCREEN_P95_KL_LE
  perturbation_top1_agreement_ge: float = PRESCREEN_TOP1_AGREEMENT_GE

  def to_manifest_dict(self) -> dict[str, float]:
    """Serializes criteria matching the Option B-1 `prescreen_audit.criteria` schema."""
    return {
        "tau_gate_l0_19_image_tokens": float(self.tau_gate_l0_19_image_tokens),
        "min_gate_l0_39_image_tokens_gt": float(self.min_gate_l0_39_image_tokens_gt),
        "min_indexer_l0_19_image_tokens_gt": float(self.min_indexer_l0_19_image_tokens_gt),
        "perturbation_max_kl_le": float(self.perturbation_max_kl_le),
        "perturbation_mean_kl_le": float(self.perturbation_mean_kl_le),
        "perturbation_p95_kl_le": float(self.perturbation_p95_kl_le),
        "perturbation_top1_agreement_ge": float(self.perturbation_top1_agreement_ge),
    }


@dataclasses.dataclass(frozen=True)
class BuiltSampleInputs:
  """Tokenized sequence and padded image arrays ready for reference evaluation."""

  sample_id: str
  prompt: str
  tokens: np.ndarray
  token_types: np.ndarray
  positions: np.ndarray
  patches_padded: np.ndarray
  image_masks: np.ndarray
  image_inputs: list[image_processor.ImageInput]
  grid_hw: tuple[int, int]
  span_starts: tuple[int, ...]
  image_source_ids: tuple[str, ...]
  image_source_sha256s: tuple[str, ...]
  prepared_png_sha256s: tuple[str, ...]
  base_asset_sha256s: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class GoldenSampleResult:
  """Evaluated golden sample metadata recorded in `manifest.json`."""

  sample_id: str
  seq_len: int
  num_images: int
  grid_hw: tuple[int, int]
  span_starts: tuple[int, ...]
  image_source_ids: tuple[str, ...]
  image_source_sha256s: tuple[str, ...]
  npz_filename: str
  npz_sha256: str


def _sha256_bytes(data: bytes) -> str:
  """Returns hex SHA-256 digest of `data`."""
  return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path | str) -> str:
  """Returns streaming hex SHA-256 digest of a file."""
  hasher = hashlib.sha256()
  with open(path, "rb") as f:
    while True:
      chunk = f.read(1 << 20)
      if not chunk:
        break
      hasher.update(chunk)
  return hasher.hexdigest()


def _norm_docstrings(tree: ast.AST) -> ast.AST:
  """Normalizes AST docstrings matching U2 `canonical_ast_v2` serialization."""
  stack = [tree]
  while stack:
    node = stack.pop()
    stack.extend(ast.iter_child_nodes(node))
    body = getattr(node, "body") if hasattr(node, "body") else None
    if (
        isinstance(body, list)
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
      doc = inspect.cleandoc(body[0].value.value)
      body[0].value.value = "\n".join(line.rstrip() for line in doc.splitlines())
  return tree


def _canon_ast(node: object) -> str:
  """Serializes an AST node into U2's Python-version-stable canonical string."""
  if isinstance(node, ast.AST):
    parts: list[str] = []
    for field in node._fields:
      if field in _SKIP_AST_FIELDS:
        continue
      value = getattr(node, field)
      if value is None or (isinstance(value, list) and not value):
        continue
      parts.append(f"{field}={_canon_ast(value)}")
    return f"{type(node).__name__}({','.join(parts)})"
  if isinstance(node, list):
    return "[" + ",".join(_canon_ast(v) for v in node) + "]"
  return repr(node)


def _canon_sha256(source_text: str) -> tuple[str, dict[str, str]]:
  """Computes `(whole_module_sha256, {symbol: prefix16})` under `canonical_ast_v2`."""
  tree = _norm_docstrings(ast.parse(source_text))
  assert isinstance(tree, ast.Module)
  whole = _sha256_bytes(_canon_ast(tree).encode("utf-8"))
  symbols = {
      node.name: _sha256_bytes(_canon_ast(node).encode("utf-8"))[:16]
      for node in tree.body
      if isinstance(node, (ast.ClassDef, ast.FunctionDef))
  }
  return whole, symbols


def verify_u2_provenance(provenance_path: pathlib.Path = _PROVENANCE_PATH) -> dict[str, Any]:
  """Verifies U2 `provenance.json` canonical AST v2 hashes at runtime.

  Args:
    provenance_path: Path to `tests/utils/deepseek_v41_reference/provenance.json`.

  Returns:
    Dictionary summarizing the verified revision, vendored file hashes,symbol
    hashes, and U2 deviations `R1..R8`.

  Raises:
    ValueError: If `provenance.json` is missing or any canonical AST hash mismatches.
  """
  if not provenance_path.exists():
    raise ValueError(f"Missing U2 provenance file: {provenance_path}")
  prov = json.loads(provenance_path.read_text(encoding="utf-8"))
  if prov["serializer"] != "canonical_ast_v2":
    raise ValueError(f"Unsupported provenance serializer {prov['serializer']!r}")

  verified_vendored: dict[str, str] = {}
  for key, entry in prov["vendored_files"].items():
    rel_path = entry["vendored_path"]
    full_path = _REPO_ROOT / rel_path
    if not full_path.exists():
      raise ValueError(f"Missing vendored reference file {full_path}")
    whole_hash, sym_hashes = _canon_sha256(full_path.read_text(encoding="utf-8"))
    if whole_hash != entry["canonical_ast_sha256"]:
      raise ValueError(
          f"Canonical AST SHA-256 mismatch for {key}: expected {entry['canonical_ast_sha256']}, got {whole_hash}"
      )
    for sym_name, expected_prefix in entry["symbols"].items():
      if sym_name not in sym_hashes or sym_hashes[sym_name] != expected_prefix:
        raise ValueError(
            f"Canonical AST symbol hash mismatch for {key}:{sym_name}: expected {expected_prefix!r}, "
            f"got {sym_hashes[sym_name] if sym_name in sym_hashes else None!r}"
        )
    verified_vendored[rel_path] = whole_hash

  verified_symbols: dict[str, str] = {}
  cached_symbol_maps: dict[str, dict[str, str]] = {}
  for sym_name, entry in prov["pinned_symbols"].items():
    rel_path = entry["file"]
    if rel_path not in cached_symbol_maps:
      full_path = _REPO_ROOT / rel_path
      if not full_path.exists():
        raise ValueError(f"Missing pinned symbol file {full_path}")
      _, cached_symbol_maps[rel_path] = _canon_sha256(full_path.read_text(encoding="utf-8"))
    file_syms = cached_symbol_maps[rel_path]
    expected_prefix = entry["canonical_ast_sha256_prefix16"]
    if sym_name not in file_syms or file_syms[sym_name] != expected_prefix:
      raise ValueError(
          f"Pinned symbol AST mismatch for {rel_path}:{sym_name}: expected {expected_prefix!r}, "
          f"got {file_syms[sym_name] if sym_name in file_syms else None!r}"
      )
    verified_symbols[sym_name] = expected_prefix

  return {
      "repo_id": prov["repo_id"],
      "revision": prov["revision"],
      "serializer": prov["serializer"],
      "provenance_json_sha256": _sha256_file(provenance_path),
      "verified_vendored_canonical_ast_sha256": verified_vendored,
      "verified_pinned_symbols_prefix16": verified_symbols,
      "deviations": prov["deviations"],
  }


def _resolve_u2_commit_sha(explicit_sha: str) -> str:
  """Resolves the 40-char git commit SHA from `explicit_sha` or `git rev-parse HEAD`."""
  if explicit_sha:
    cleaned = explicit_sha.strip()
    if len(cleaned) != 40:
      raise ValueError(f"Expected 40-char hex git SHA for --u2-commit-sha, got {explicit_sha!r}")
    return cleaned
  git_marker = _REPO_ROOT / ".git"
  if not git_marker.exists():
    raise ValueError(
        "Repository .git marker not found and --u2-commit-sha was not provided; "
        "pass --u2-commit-sha explicitly when running from an exported archive."
    )
  proc = subprocess.run(
      ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
      check=True,
      capture_output=True,
      text=True,
  )
  sha = proc.stdout.strip()
  if len(sha) != 40:
    raise ValueError(f"Unexpected git rev-parse HEAD output: {sha!r}")
  return sha


def _resolve_tokenizer_source(checkpoint_dir: str, explicit_tokenizer_path: str, *, mini: bool) -> str:
  """Resolves the tokenizer directory or HF repo ID, failing loudly when `mini=False`."""
  if explicit_tokenizer_path:
    return explicit_tokenizer_path
  if not mini:
    if not checkpoint_dir:
      raise FileNotFoundError("Missing checkpoint_dir for tokenizer resolution when mini=False")
    tok_json = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tok_json):
      raise FileNotFoundError(f"Missing required tokenizer.json in checkpoint directory: {tok_json}")
    return checkpoint_dir
  if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "tokenizer.json")):
    return checkpoint_dir
  if "DEEPSEEK_V41_TOKENIZER" in os.environ:
    return os.environ["DEEPSEEK_V41_TOKENIZER"]
  return _DEFAULT_HF_TOKENIZER_ID


def build_quantized_mini_checkpoint(
    output_dir: str | pathlib.Path,
    *,
    tokenizer_path: str = "",
    seed: int = 1,
) -> pathlib.Path:
  """Builds a deterministic release-layout FP8/FP4 multimodal mini checkpoint.

  Writes:
    - `BF16` `vision.*`, `aligner.*`, `image_{start,end,newline}` tensors.
    - `F32` `layers.<i>.ffn.gate.bias_vl` tensors.
    - Block-quantized FP8 (`32x32` linear, `1x32` engram) and packed FP4 (`1x32`
      expert) text weights + `ue8m0` `.scale` tensors across 2 safetensors shards.
    - Dummy `mtp.0.*` tensors verifying MTP skip handling.
    - `config.json`, `model.safetensors.index.json`, and tokenizer files.

  Args:
    output_dir: Target directory for the generated mini checkpoint.
    tokenizer_path: Optional tokenizer path/repo ID.
    seed: Deterministic weight initialization seed.

  Returns:
    Resolved `pathlib.Path` to `output_dir`.
  """
  out_path = pathlib.Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  tok_src = _resolve_tokenizer_source("", tokenizer_path, mini=True)
  tokenizer = AutoTokenizer.from_pretrained(tok_src)
  tokenizer.save_pretrained(str(out_path))

  args = multimodal.mini_model_args(
      dtype="bf16",
      fp32_compute=False,
      n_routed_experts=32,
      moe_inter_dim=512,
      q_lora_rank=32,
      o_lora_rank=16,
      engram_vocab_size=10000,
      engram_head_dim=32,
  )
  model = multimodal.build_multimodal_model(args, tokenizer=tokenizer)
  multimodal.init_mini_reference(model, seed=seed)

  raw_tensors: dict[str, torch.Tensor] = {}
  for name, param in model.named_parameters():
    tensor = param.detach().cpu().contiguous()
    if name.startswith(("vision.", "aligner.")) or name in ("image_start", "image_end", "image_newline"):
      raw_tensors[name] = tensor.to(torch.bfloat16)
    elif name.endswith(".bias_vl"):
      raw_tensors[name] = tensor.to(torch.float32)
    else:
      raw_tensors[name] = tensor.clone()

  # Include representative MTP tensors (BF16, F32 bias_vl, and quantized FP8 linear)
  # to exercise strict MTP-only skip filtering.
  g_mtp = torch.Generator().manual_seed(seed + 999)
  raw_tensors["mtp.0.norm.weight"] = torch.ones(args.dim, dtype=torch.bfloat16)
  raw_tensors["mtp.0.ffn.gate.bias_vl"] = torch.zeros(args.n_routed_experts, dtype=torch.float32)
  mtp_wq = torch.randn((32, args.dim), generator=g_mtp, dtype=torch.float32) * 0.05
  raw_tensors["mtp.0.attn.wq_a.weight"] = mtp_wq.to(torch.bfloat16)

  quantized_shards: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] = ({}, {})
  sorted_names = sorted(raw_tensors)
  midpoint = len(sorted_names) // 2
  for idx, name in enumerate(sorted_names):
    target_shard = quantized_shards[0 if idx < midpoint else 1]
    t = raw_tensors[name]
    scale_key = f"{name[:-len('weight')]}scale" if name.endswith("weight") else ""
    if quantize_checkpoint._FP4_EXPERTS.search(name):  # pylint: disable=protected-access
      qw, qs = quantize_checkpoint.quantize_fp4(t)
      target_shard[name] = qw
      target_shard[scale_key] = qs
    elif quantize_checkpoint._FP8_LINEARS.search(name):  # pylint: disable=protected-access
      qw, qs = quantize_checkpoint.quantize_fp8(t, 32, 32)
      target_shard[name] = qw
      target_shard[scale_key] = qs
    elif quantize_checkpoint._ENGRAM_TABLE.search(name):  # pylint: disable=protected-access
      qw, qs = quantize_checkpoint.quantize_fp8(t, 1, 32)
      target_shard[name] = qw
      target_shard[scale_key] = qs
    else:
      target_shard[name] = t

  weight_map: dict[str, str] = {}
  for shard_idx, shard_dict in enumerate(quantized_shards, start=1):
    shard_name = f"model-{shard_idx:05d}-of-00002.safetensors"
    save_file(shard_dict, str(out_path / shard_name), metadata={"format": "pt"})
    for key in shard_dict:
      weight_map[key] = shard_name

  index_payload = {"metadata": {}, "weight_map": weight_map}
  (out_path / "model.safetensors.index.json").write_text(
      json.dumps(index_payload, indent=2, sort_keys=True),
      encoding="utf-8",
  )

  text_cfg: dict[str, Any] = {
      "model_type": "deepseek_v41_text",
      "rope_scaling": {
          "rope_type": "yarn",
          "factor": args.rope_factor,
          "beta_fast": args.beta_fast,
          "beta_slow": args.beta_slow,
          "original_max_position_embeddings": args.original_seq_len,
      },
  }
  excluded_fields = {
      "max_batch_size",
      "max_seq_len",
      "temperature",
      "dtype",
      "fp32_compute",
      "expert_dtype",
      "original_seq_len",
      "rope_factor",
      "beta_fast",
      "beta_slow",
      "n_mtp_layers",
      "dspark_block_size",
      "dspark_noise_token_id",
      "dspark_target_layer_ids",
      "dspark_markov_rank",
      "dspark_n_routed_experts",
      "dspark_n_activated_experts",
      "vision_n_layers",
      "vision_dim",
      "vision_n_heads",
      "vision_inter_dim",
      "vision_patch_size",
      "vision_rope_theta",
      "vision_downsample_ratio",
      "vision_max_n_token",
      "vision_min_pixels",
      "vision_max_wh_ratio",
      "image_token_id",
  }
  for field_name in ref_model.ModelArgs.__dataclass_fields__:
    if field_name in excluded_fields:
      continue
    val = getattr(args, field_name)
    hf_key = _ARG_TO_HF_TEXT_KEY[field_name] if field_name in _ARG_TO_HF_TEXT_KEY else field_name
    text_cfg[hf_key] = list(val) if isinstance(val, tuple) else val

  hf_config = {
      "architectures": ["DeepseekV41ForCausalLM"],
      "model_type": "deepseek_v41",
      "dtype": "bfloat16",
      "bos_token_id": 0,
      "eos_token_id": 1,
      "pad_token_id": 2,
      "image_token_id": args.image_token_id,
      "quantization_config": {
          "quant_method": "fp8",
          "activation_scheme": "dynamic",
          "weight_block_size": [32, 32],
          "scale_fmt": "ue8m0",
          "expert_dtype": "fp4",
      },
      "text_config": text_cfg,
      "vision_config": {
          "model_type": "deepseek_v41_vision",
          "num_hidden_layers": args.vision_n_layers,
          "hidden_size": args.vision_dim,
          "num_attention_heads": args.vision_n_heads,
          "intermediate_size": args.vision_inter_dim,
          "patch_size": args.vision_patch_size,
          "rope_theta": args.vision_rope_theta,
          "downsample_ratio": args.vision_downsample_ratio,
          "max_image_tokens": args.vision_max_n_token,
          "min_pixels": args.vision_min_pixels,
          "max_wh_ratio": args.vision_max_wh_ratio,
      },
  }
  (out_path / "config.json").write_text(json.dumps(hf_config, indent=2, sort_keys=True), encoding="utf-8")
  return out_path


def _make_seeded_synthetic_rgb(seed: int, height: int = 384, width: int = 512) -> tuple[Image.Image, bytes]:
  """Creates a deterministic patterned + noise RGB image and its raw PNG bytes."""
  rng = np.random.default_rng(seed)
  ys = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
  xs = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
  r = np.clip(0.6 * ys + 0.4 * xs + rng.normal(0.0, 18.0, size=(height, width)), 0, 255)
  g = np.clip(0.5 * (255.0 - ys) + 0.5 * xs + rng.normal(0.0, 18.0, size=(height, width)), 0, 255)
  b = np.clip(128.0 + 60.0 * np.sin(ys / 19.0) * np.cos(xs / 23.0) + rng.normal(0.0, 18.0, size=(height, width)), 0, 255)
  rgb_arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
  img = Image.fromarray(rgb_arr, mode="RGB")
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return img, buf.getvalue()


def prepare_letterboxed_image(
    source_id: str,
    rgb_image: Image.Image,
    raw_source_bytes: bytes,
    args: ref_model.ModelArgs,
    *,
    target_grid_hw: tuple[int, int] = _FULL_GRID_HW,
    base_asset_sha256: str = "",
) -> PreparedImage:
  """Letterboxes `rgb_image` and runs `image_processor.load_image` asserting `target_grid_hw`.

  For the full `(39, 39)` grid (`min_pixels = 544 * 544`), letterboxing to a
  `544 x 544` square guarantees `plan_image_grid` plans `(13, 13, 546, 546)` ->
  `39 x 39 = 1521` patches and `184` image-span tokens.

  Args:
    source_id: Logical identifier recorded in sample/manifest provenance.
    rgb_image: Source PIL image.
    raw_source_bytes: Raw source bytes hashed for provenance.
    args: Reference `ModelArgs` providing vision grid parameters.
    target_grid_hw: Expected `(n_vit_h, n_vit_w)` patch grid.
    base_asset_sha256: Optional SHA-256 of the underlying base file when `rgb_image`
      is a deterministic crop (`crop_544:...`); defaults to `SHA256(raw_source_bytes)`.

  Returns:
    `PreparedImage` carrying both `bfloat16` patches for `load_image` fidelity and
    flattened `float32 [P, 588]` patches for `.npz` / `mm_logit.pkl` storage.

  Raises:
    ValueError: If `plan_image_grid` or `load_image` disagrees with `target_grid_hw`.
  """
  if target_grid_hw == _FULL_GRID_HW and args.vision_min_pixels == _FULL_MIN_PIXELS_SIDE * _FULL_MIN_PIXELS_SIDE:
    box_h, box_w = _FULL_MIN_PIXELS_SIDE, _FULL_MIN_PIXELS_SIDE
  else:
    box_h = target_grid_hw[0] * args.vision_patch_size
    box_w = target_grid_hw[1] * args.vision_patch_size

  letterboxed = ImageOps.pad(rgb_image.convert("RGB"), (box_w, box_h), color=(127, 127, 127))
  png_buf = io.BytesIO()
  letterboxed.save(png_buf, format="PNG")
  png_bytes = png_buf.getvalue()

  patches_bf16, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor.load_image({"data": png_bytes}, args)
  if (n_vit_h, n_vit_w) != target_grid_hw:
    raise ValueError(f"Expected ViT grid {target_grid_hw} for {source_id!r}, got {(n_vit_h, n_vit_w)}")
  expected_llm_hw = image_processor.llm_grid(
      n_vit_h * args.vision_patch_size,
      n_vit_w * args.vision_patch_size,
      args.vision_patch_size,
      args.vision_downsample_ratio,
  )
  if (n_llm_h, n_llm_w) != expected_llm_hw:
    raise ValueError(f"Expected LLM grid {expected_llm_hw} for {source_id!r}, got {(n_llm_h, n_llm_w)}")
  if patches_bf16.dtype != torch.bfloat16:
    raise ValueError(f"Expected bfloat16 patches from load_image, got {patches_bf16.dtype}")

  token_types = image_processor.image_token_types(n_llm_h, n_llm_w)
  expected_span = image_processor.num_image_tokens(n_llm_h, n_llm_w)
  if int(token_types.numel()) != expected_span:
    raise ValueError(
        f"Span length mismatch for {source_id!r}: token_types={token_types.numel()} vs expected={expected_span}"
    )
  if target_grid_hw == _FULL_GRID_HW and expected_span != _FULL_SPAN_TOKENS:
    raise ValueError(f"Expected {_FULL_SPAN_TOKENS} span tokens for 39x39 grid, got {expected_span}")

  patches_flat_fp32 = patches_bf16.float().flatten(1).cpu().numpy().astype(np.float32)
  source_sha = _sha256_bytes(raw_source_bytes)
  png_sha = _sha256_bytes(png_bytes)
  resolved_base_sha = base_asset_sha256 if base_asset_sha256 else source_sha
  return PreparedImage(
      source_id=source_id,
      source_sha256=source_sha,
      patches_bf16=patches_bf16,
      patches_flat_fp32=patches_flat_fp32,
      n_vit_h=n_vit_h,
      n_vit_w=n_vit_w,
      token_types=token_types,
      prepared_png_sha256=png_sha,
      base_asset_sha256=resolved_base_sha,
  )


def _resolve_release_images_dir(ckpt_dir: pathlib.Path, release_images_dir: str) -> pathlib.Path:
  """Resolves the directory containing `carrots.jpeg` and `corn.jpeg`."""
  if release_images_dir:
    return pathlib.Path(release_images_dir)
  return ckpt_dir / "inference" / "examples" / "images"


def load_mm_v2_base_assets(
    ckpt_dir: pathlib.Path,
    *,
    release_images_dir: str = "",
    verify_release_sha256: bool = True,
) -> dict[str, CandidateBaseAsset]:
  """Loads and SHA-256-verifies the 4 real photographic base assets (`B0..B3`).

  Args:
    ckpt_dir: Checkpoint root directory containing `inference/examples/images/` unless
      overridden by `release_images_dir`.
    release_images_dir: Optional explicit override directory for `B1`/`B2` (`carrots.jpeg`,
      `corn.jpeg`) in hermetic `--mini` tests.
    verify_release_sha256: When `True`, enforces pinned SHA-256s on `B1` and `B2` in
      addition to repository assets `B0` and `B3` (which are always verified).

  Returns:
    Mapping from `base_id` (`"B0".."B3"`) to `CandidateBaseAsset`.

  Raises:
    FileNotFoundError: If any base image file does not exist.
    ValueError: If any verified SHA-256 digest mismatches `EXPECTED_BASE_ASSET_SHA256`.
  """
  images_dir = _resolve_release_images_dir(ckpt_dir, release_images_dir)
  base_paths = {
      "B0": _TEST_IMAGE_REASONING_PATH,
      "B1": images_dir / "carrots.jpeg",
      "B2": images_dir / "corn.jpeg",
      "B3": _TEST_IMAGE_PATH,
  }
  loaded: dict[str, CandidateBaseAsset] = {}
  for base_id in CANDIDATE_BASE_IDS:
    path = base_paths[base_id]
    if not path.exists():
      raise FileNotFoundError(f"Missing required mm_v2 base asset {base_id} at {path}")
    raw_bytes = path.read_bytes()
    actual_sha = _sha256_bytes(raw_bytes)
    must_verify = base_id in ("B0", "B3") or verify_release_sha256
    if must_verify:
      expected_sha = EXPECTED_BASE_ASSET_SHA256[base_id]
      if actual_sha != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for mm_v2 base asset {base_id} ({path}): " f"expected {expected_sha}, got {actual_sha}"
        )
    with Image.open(io.BytesIO(raw_bytes)) as im:
      rgb_img = im.convert("RGB")
      rgb_img.load()
    loaded[base_id] = CandidateBaseAsset(
        base_id=base_id,
        path=path,
        raw_bytes=raw_bytes,
        sha256=actual_sha,
        rgb_image=rgb_img,
    )
  return loaded


def _compute_square_crop_box(
    size: tuple[int, int],
    centering: tuple[float, float],
) -> tuple[int, int, int, int]:
  """Computes the `(x0, y0, x1, y1)` square source crop box used by `ImageOps.fit(..., (544, 544))`."""
  width, height = int(size[0]), int(size[1])
  side = min(width, height)
  left = int(round((width - side) * float(centering[0])))
  top = int(round((height - side) * float(centering[1])))
  return (left, top, left + side, top + side)


def build_mm_v2_candidate_queue(base_assets: dict[str, CandidateBaseAsset]) -> tuple[CandidateCropSpec, ...]:
  """Builds the deterministic 72-candidate real-photographic crop queue (`Spec §2`).

  Interleaves `base_id in ("B0", "B1", "B2", "B3")` inside `anchor_idx in 0..8` and
  `mirror in (0, 1)` so consecutive slots draw from distinct base photographs first.

  Args:
    base_assets: Loaded `B0..B3` `CandidateBaseAsset` mapping.

  Returns:
    Tuple of 72 `CandidateCropSpec` entries (`rank = 0..71`).
  """
  missing = [bid for bid in CANDIDATE_BASE_IDS if bid not in base_assets]
  if missing:
    raise ValueError(f"Missing base_assets keys for mm_v2 queue: {missing}")

  queue: list[CandidateCropSpec] = []
  rank = 0
  for mirror in (0, 1):
    for anchor_idx, centering in enumerate(CANDIDATE_ANCHORS):
      for base_id in CANDIDATE_BASE_IDS:
        source_id = f"crop_{_CANDIDATE_CROP_SIDE}:{base_id}:anchor_{anchor_idx}:mirror_{mirror}"
        cbox = _compute_square_crop_box(base_assets[base_id].rgb_image.size, centering)
        queue.append(
            CandidateCropSpec(
                rank=rank,
                source_id=source_id,
                base_id=base_id,
                anchor_idx=anchor_idx,
                centering=centering,
                mirror=mirror,
                base_asset_sha256=base_assets[base_id].sha256,
                crop_box=cbox,
            )
        )
        rank += 1
  return tuple(queue)


def materialize_candidate_crop_rgb(
    crop_spec: CandidateCropSpec,
    base_assets: dict[str, CandidateBaseAsset],
) -> tuple[Image.Image, bytes]:
  """Renders a `544x544` BICUBIC crop (`Spec §2`) and deterministic PNG bytes."""
  src_rgb = base_assets[crop_spec.base_id].rgb_image
  oriented = ImageOps.mirror(src_rgb) if crop_spec.mirror == 1 else src_rgb
  crop_rgb = ImageOps.fit(
      oriented,
      (_CANDIDATE_CROP_SIDE, _CANDIDATE_CROP_SIDE),
      method=Image.Resampling.BICUBIC,
      centering=crop_spec.centering,
  )
  png_buf = io.BytesIO()
  crop_rgb.save(png_buf, format="PNG")
  return crop_rgb, png_buf.getvalue()


def build_mm_v3_distinct_crop_queue(
    base_assets: dict[str, CandidateBaseAsset],
) -> tuple[tuple[CandidateCropSpec, str], ...]:
  """Builds the Option C `mm_v3` deduplicated crop queue `(CandidateCropSpec, png_sha256)`.

  Walks `build_mm_v2_candidate_queue(base_assets)` in rank order `0..71`, materializes each
  `544x544` crop's PNG bytes via `materialize_candidate_crop_rgb`, and retains only the first
  occurrence of each distinct PNG SHA-256 (`24` distinct crops when `B0..B3` are distinct).
  """
  candidate_queue = build_mm_v2_candidate_queue(base_assets)
  seen_png_sha256s: set[str] = set()
  distinct_queue: list[tuple[CandidateCropSpec, str]] = []
  for spec in candidate_queue:
    _, png_bytes = materialize_candidate_crop_rgb(spec, base_assets)
    png_sha = _sha256_bytes(png_bytes)
    if png_sha in seen_png_sha256s:
      continue
    seen_png_sha256s.add(png_sha)
    distinct_queue.append((spec, png_sha))
  return tuple(distinct_queue)


def _parse_crop_source_key(source_key: str) -> tuple[str, int, int]:
  """Parses `crop_544:<base_id>:anchor_<a_idx>:mirror_<m>` into `(base_id, a_idx, m)`."""
  parts = source_key.split(":")
  if (
      len(parts) != 4
      or parts[0] != f"crop_{_CANDIDATE_CROP_SIDE}"
      or parts[1] not in CANDIDATE_BASE_IDS
      or not parts[2].startswith("anchor_")
      or not parts[3].startswith("mirror_")
  ):
    raise ValueError(f"Malformed mm_v2 crop source key: {source_key!r}")
  anchor_idx = int(parts[2].split("_", 1)[1])
  mirror = int(parts[3].split("_", 1)[1])
  if anchor_idx < 0 or anchor_idx >= len(CANDIDATE_ANCHORS) or mirror not in (0, 1):
    raise ValueError(f"Out-of-range anchor/mirror in mm_v2 crop source key: {source_key!r}")
  return parts[1], anchor_idx, mirror


def _load_prepared_image_by_key(
    source_key: str,
    ckpt_dir: pathlib.Path,
    args: ref_model.ModelArgs,
    target_grid_hw: tuple[int, int],
    cache: dict[str, PreparedImage],
    *,
    release_images_dir: str = "",
    base_assets: dict[str, CandidateBaseAsset] | None = None,
) -> PreparedImage:
  """Resolves `source_key` to a cached `PreparedImage`."""
  if source_key in cache:
    return cache[source_key]

  if source_key == "asset:test_image.jpg":
    if not _TEST_IMAGE_PATH.exists():
      raise ValueError(f"Missing required repository test image {_TEST_IMAGE_PATH}")
    raw_bytes = _TEST_IMAGE_PATH.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as im:
      rgb = im.convert("RGB")
    prepared = prepare_letterboxed_image(source_key, rgb, raw_bytes, args, target_grid_hw=target_grid_hw)
  elif source_key in ("release:carrots.jpeg", "release:corn.jpeg"):
    rel_name = source_key.split(":", 1)[1]
    images_dir = _resolve_release_images_dir(ckpt_dir, release_images_dir)
    candidate = images_dir / rel_name
    if not candidate.exists():
      raise FileNotFoundError(f"Missing required release example image: {candidate}")
    raw_bytes = candidate.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as im:
      rgb = im.convert("RGB")
    prepared = prepare_letterboxed_image(source_key, rgb, raw_bytes, args, target_grid_hw=target_grid_hw)
  elif source_key.startswith(f"crop_{_CANDIDATE_CROP_SIDE}:"):
    base_id, anchor_idx, mirror = _parse_crop_source_key(source_key)
    resolved_assets = (
        base_assets
        if base_assets is not None
        else load_mm_v2_base_assets(
            ckpt_dir,
            release_images_dir=release_images_dir,
            verify_release_sha256=not bool(release_images_dir),
        )
    )
    crop_spec = CandidateCropSpec(
        rank=-1,
        source_id=source_key,
        base_id=base_id,
        anchor_idx=anchor_idx,
        centering=CANDIDATE_ANCHORS[anchor_idx],
        mirror=mirror,
        base_asset_sha256=resolved_assets[base_id].sha256,
    )
    crop_rgb, crop_png_bytes = materialize_candidate_crop_rgb(crop_spec, resolved_assets)
    prepared = prepare_letterboxed_image(
        source_key,
        crop_rgb,
        crop_png_bytes,
        args,
        target_grid_hw=target_grid_hw,
        base_asset_sha256=resolved_assets[base_id].sha256,
    )
  elif source_key.startswith("synth:"):
    seed = int(source_key.split(":", 1)[1])
    rgb, raw_bytes = _make_seeded_synthetic_rgb(seed=seed)
    prepared = prepare_letterboxed_image(source_key, rgb, raw_bytes, args, target_grid_hw=target_grid_hw)
  else:
    raise ValueError(f"Unknown image source key {source_key!r}")

  cache[source_key] = prepared
  return prepared


def load_c4_token_pool(tokenizer: Any, image_token_id: int) -> list[int]:
  """Loads deterministic English text tokens from the local C4 validation parquet."""
  if not _C4_VAL_PARQUET_PATH.exists():
    raise ValueError(f"Missing required local C4 validation parquet {_C4_VAL_PARQUET_PATH}")
  table = pq.read_table(str(_C4_VAL_PARQUET_PATH), columns=["text"])
  passages: list[str] = table["text"].to_pylist()
  joined = "\n\n".join(p.strip() for p in passages if isinstance(p, str) and p.strip())
  encoded: list[int] = tokenizer.encode(joined, add_special_tokens=False)
  filtered = [int(tok) for tok in encoded if int(tok) != image_token_id]
  if len(filtered) < _MIN_C4_POOL_TOKENS:
    raise ValueError(f"C4 token pool too short ({len(filtered)} tokens < {_MIN_C4_POOL_TOKENS})")
  return filtered


def build_sample_suite(*, mini: bool, golden_version: str = "mm_v1") -> dict[str, SampleSpec]:
  """Returns the canonical 12-sample specification table (`L1..L4`, `M1..M8`)."""
  if golden_version not in GOLDEN_VERSIONS:
    raise ValueError(f"Unsupported golden_version={golden_version!r}; valid={GOLDEN_VERSIONS}")
  mini_l2_sources = (
      ("release:carrots.jpeg", "release:corn.jpeg")
      if golden_version in ("mm_v2", "mm_v3")
      else ("synth:201", "synth:202")
  )
  if mini:
    specs = (
        SampleSpec("L1", 32, 1, "middle", ("synth:101",), 0),
        SampleSpec("L2", None, 2, "release_l2", mini_l2_sources, 64),
        SampleSpec("L3", 32, 1, "middle", ("asset:test_image.jpg",), 128),
        SampleSpec("L4", 64, 4, "four_interior", ("asset:test_image.jpg", "synth:102", "synth:103", "synth:104"), 192),
        SampleSpec("M1", 48, 1, "start", ("synth:105",), 256),
        SampleSpec("M2", 56, 1, "middle", ("asset:test_image.jpg",), 320),
        SampleSpec("M3", 64, 2, "start_end", ("synth:106", "synth:107"), 384),
        SampleSpec("M4", 72, 2, "middle", ("asset:test_image.jpg", "synth:108"), 448),
        SampleSpec("M5", 72, 3, "start_mid_end", ("synth:109", "synth:110", "synth:111"), 512),
        SampleSpec("M6", 80, 3, "middle", ("asset:test_image.jpg", "synth:112", "synth:113"), 576),
        SampleSpec("M7", 96, 2, "start_mid", ("synth:114", "synth:115"), 640),
        SampleSpec(
            "M8", 96, 4, "start_two_mid_end", ("asset:test_image.jpg", "synth:116", "synth:117", "synth:118"), 704
        ),
    )
  else:
    specs = (
        SampleSpec("L1", 256, 1, "middle", ("synth:101",), 0),
        SampleSpec("L2", None, 2, "release_l2", ("release:carrots.jpeg", "release:corn.jpeg"), 128),
        SampleSpec("L3", 256, 1, "middle", ("asset:test_image.jpg",), 256),
        SampleSpec("L4", 896, 4, "four_interior", ("asset:test_image.jpg", "synth:102", "synth:103", "synth:104"), 384),
        SampleSpec("M1", 512, 1, "start", ("synth:105",), 640),
        SampleSpec("M2", 768, 1, "middle", ("asset:test_image.jpg",), 960),
        SampleSpec("M3", 896, 2, "start_end", ("synth:106", "synth:107"), 1280),
        SampleSpec("M4", 1024, 2, "middle", ("asset:test_image.jpg", "synth:108"), 1600),
        SampleSpec("M5", 1024, 3, "start_mid_end", ("synth:109", "synth:110", "synth:111"), 1920),
        SampleSpec("M6", 1536, 3, "middle", ("asset:test_image.jpg", "synth:112", "synth:113"), 2240),
        SampleSpec("M7", 2048, 2, "start_mid", ("synth:114", "synth:115"), 2688),
        SampleSpec(
            "M8", 2048, 4, "start_two_mid_end", ("asset:test_image.jpg", "synth:116", "synth:117", "synth:118"), 3200
        ),
    )
  if golden_version == "mm_v3":
    cursor = 0
    mm_v3_specs: list[SampleSpec] = []
    for spec in specs:
      if spec.sample_id == "L2":
        mm_v3_specs.append(spec)
      else:
        assigned = MM_V3_DISTINCT_CROP_SOURCE_IDS[cursor : cursor + spec.num_images]
        if len(assigned) != spec.num_images:
          raise ValueError(f"Insufficient MM_V3_DISTINCT_CROP_SOURCE_IDS for {spec.sample_id}")
        cursor += spec.num_images
        mm_v3_specs.append(dataclasses.replace(spec, image_sources=assigned))
    return {s.sample_id: s for s in mm_v3_specs}
  return {s.sample_id: s for s in specs}


def _partition_text_gaps(text_budget: int, num_images: int, placement: str) -> list[int]:
  """Splits `text_budget` across `num_images + 1` text gaps around the image spans."""
  if text_budget < 0:
    raise ValueError(f"Negative text_budget={text_budget} for placement={placement!r}")
  n_gaps = num_images + 1
  if placement == "start":
    if num_images != 1:
      raise ValueError(f"placement='start' requires 1 image, got {num_images}")
    return [0, text_budget]
  if placement == "end":
    if num_images != 1:
      raise ValueError(f"placement='end' requires 1 image, got {num_images}")
    return [text_budget, 0]
  if placement == "start_end":
    if num_images != 2:
      raise ValueError(f"placement='start_end' requires 2 images, got {num_images}")
    return [0, text_budget, 0]
  if placement == "start_mid":
    if num_images != 2:
      raise ValueError(f"placement='start_mid' requires 2 images, got {num_images}")
    g1 = text_budget // 2
    return [0, g1, text_budget - g1]
  if placement == "start_mid_end":
    if num_images != 3:
      raise ValueError(f"placement='start_mid_end' requires 3 images, got {num_images}")
    g1 = text_budget // 2
    return [0, g1, text_budget - g1, 0]
  if placement == "start_two_mid_end":
    if num_images != 4:
      raise ValueError(f"placement='start_two_mid_end' requires 4 images, got {num_images}")
    g1 = text_budget // 3
    g2 = (text_budget - g1) // 2
    return [0, g1, g2, text_budget - g1 - g2, 0]
  if placement in ("middle", "four_interior"):
    base = text_budget // n_gaps
    rem = text_budget % n_gaps
    return [base + (1 if i < rem else 0) for i in range(n_gaps)]
  raise ValueError(f"Unsupported placement {placement!r}")


def _slice_token_pool(pool: Sequence[int], offset: int, count: int) -> list[int]:
  """Slices `count` contiguous tokens deterministically from `pool` without wraparound."""
  if count < 0:
    raise ValueError(f"Negative token slice count={count}")
  if count == 0:
    return []
  if offset < 0 or offset + count > len(pool):
    raise ValueError(f"Token pool slice [{offset}, {offset + count}) exceeds pool length {len(pool)}")
  return [int(pool[offset + i]) for i in range(count)]


def validate_no_duplicate_text_ngrams(
    tokens: np.ndarray,
    token_types: np.ndarray,
    *,
    sample_id: str,
    window_size: int = _ANTI_COPY_NGRAM_WINDOW,
) -> None:
  """Verifies that all `window_size`-token non-image text windows in a sample are unique.

  Args:
    tokens: Token ID array `[seq_len]`.
    token_types: Token type array `[seq_len]` (`image_processor.TEXT` on text tokens).
    sample_id: Sample identifier for error messages.
    window_size: Contiguous non-image text n-gram window size (default: 16).

  Raises:
    ValueError: If `window_size <= 0`, array shapes mismatch, or any `window_size`-token
      window of non-image text tokens repeats within the sample.
  """
  if window_size <= 0:
    raise ValueError(f"window_size must be positive, got {window_size}")
  if tokens.shape != token_types.shape:
    raise ValueError(f"tokens.shape={tokens.shape} != token_types.shape={token_types.shape} on sample {sample_id!r}")
  text_tokens = [
      int(tok)
      for tok, tok_type in zip(tokens.tolist(), token_types.tolist(), strict=True)
      if int(tok_type) == image_processor.TEXT
  ]
  seen_windows: dict[tuple[int, ...], int] = {}
  for idx in range(len(text_tokens) - window_size + 1):
    window = tuple(text_tokens[idx : idx + window_size])
    if window in seen_windows:
      first_idx = seen_windows[window]
      raise ValueError(
          f"Duplicate {window_size}-token text window in sample {sample_id!r} "
          f"at text offsets {first_idx} and {idx}: {window}"
      )
    seen_windows[window] = idx


def _extract_release_l2_tagged_prompt(ckpt_dir: pathlib.Path, *, mini: bool) -> str:
  """Returns the two-image prompt (`_MINI_TWO_IMAGE_PROMPT` for `mini=True`, `example.txt` for `mini=False`)."""
  if mini:
    return _MINI_TWO_IMAGE_PROMPT
  example_file = ckpt_dir / "inference" / "examples" / "example.txt"
  if not example_file.exists():
    raise FileNotFoundError(f"Missing required release prompt file: {example_file}")
  blocks = [b.strip() for b in example_file.read_text(encoding="utf-8").split("\n\n") if b.strip()]
  for block in blocks:
    if len(_IMAGE_TAG_RE.findall(block)) == 2:
      return block
  raise ValueError(f"No two-image <image>...</image> prompt found in {example_file}")


def _build_release_l2_sample(
    ckpt_dir: pathlib.Path,
    tokenizer: Any,
    args: ref_model.ModelArgs,
    prepared_imgs: Sequence[PreparedImage],
    *,
    mini: bool,
) -> tuple[str, list[int], list[int], list[image_processor.ImageInput]]:
  """Tokenizes L2 matching release `encoding.encode_messages(..., thinking_mode='chat')`."""
  tagged = _extract_release_l2_tagged_prompt(ckpt_dir, mini=mini)
  parts = _IMAGE_TAG_RE.split(tagged)
  if len(parts) != 5:
    raise ValueError(f"Expected 2 <image> tags (5 split parts) in L2 prompt, got {len(parts)}: {tagged!r}")
  # re.split with 1 capture group yields [text0, tag0, text1, tag1, text2]
  text_pieces = [parts[0], parts[2], parts[4]]
  wrapped_text0 = f"<｜User｜>{text_pieces[0]}\n\n"
  wrapped_text1 = f"\n\n{text_pieces[1]}\n\n"
  wrapped_text2 = f"\n\n{text_pieces[2]}<｜Assistant｜></think>"
  seg_texts = [wrapped_text0, wrapped_text1, wrapped_text2]

  tokens: list[int] = []
  token_types: list[int] = []
  image_inputs: list[image_processor.ImageInput] = []

  if tokenizer.bos_token_id is not None and int(tokenizer.bos_token_id) != args.image_token_id:
    tokens.append(int(tokenizer.bos_token_id))
    token_types.append(image_processor.TEXT)

  for idx, seg_text in enumerate(seg_texts):
    seg_ids = [int(t) for t in tokenizer.encode(seg_text, add_special_tokens=False) if int(t) != args.image_token_id]
    tokens.extend(seg_ids)
    token_types.extend([image_processor.TEXT] * len(seg_ids))
    if idx < len(prepared_imgs):
      pimg = prepared_imgs[idx]
      span_len = int(pimg.token_types.numel())
      image_inputs.append(
          image_processor.ImageInput(
              start=len(tokens),
              patches=pimg.patches_bf16,
              n_vit_h=pimg.n_vit_h,
              n_vit_w=pimg.n_vit_w,
              types=pimg.token_types,
          )
      )
      tokens.extend([args.image_token_id] * span_len)
      token_types.extend(int(t) for t in pimg.token_types.tolist())

  return tagged, tokens, token_types, image_inputs


def build_sample_inputs(
    spec: SampleSpec,
    ckpt_dir: pathlib.Path,
    tokenizer: Any,
    args: ref_model.ModelArgs,
    c4_pool: Sequence[int],
    image_cache: dict[str, PreparedImage],
    *,
    mini: bool,
    release_images_dir: str = "",
    base_assets: dict[str, CandidateBaseAsset] | None = None,
) -> BuiltSampleInputs:
  """Constructs and validates one multimodal golden sample's tokens and image tensors."""
  target_grid_hw = _MINI_GRID_HW if mini else _FULL_GRID_HW
  expected_span = _MINI_SPAN_TOKENS if mini else _FULL_SPAN_TOKENS
  if len(spec.image_sources) != spec.num_images:
    raise ValueError(f"Sample {spec.sample_id} image_sources mismatch vs num_images={spec.num_images}")
  if spec.num_images < 1 or spec.num_images > _MAX_IMAGE_SLOTS:
    raise ValueError(f"Sample {spec.sample_id} num_images={spec.num_images} outside [1, {_MAX_IMAGE_SLOTS}]")

  prepared_imgs = [
      _load_prepared_image_by_key(
          src_key,
          ckpt_dir,
          args,
          target_grid_hw,
          image_cache,
          release_images_dir=release_images_dir,
          base_assets=base_assets,
      )
      for src_key in spec.image_sources
  ]

  if spec.placement == "release_l2":
    prompt_str, tok_list, type_list, image_inputs = _build_release_l2_sample(
        ckpt_dir,
        tokenizer,
        args,
        prepared_imgs,
        mini=mini,
    )
  else:
    assert spec.target_seq_len is not None
    total_img_tokens = spec.num_images * expected_span
    text_budget = spec.target_seq_len - total_img_tokens
    gaps = _partition_text_gaps(text_budget, spec.num_images, spec.placement)
    tok_list = []
    type_list = []
    image_inputs = []
    prompt_parts: list[str] = []
    pool_cursor = spec.c4_offset

    for gap_idx, gap_len in enumerate(gaps):
      gap_tokens = _slice_token_pool(c4_pool, pool_cursor, gap_len)
      pool_cursor += gap_len
      if gap_idx == 0 and gap_len > 0 and tokenizer.bos_token_id is not None:
        bos_id = int(tokenizer.bos_token_id)
        if bos_id != args.image_token_id:
          gap_tokens[0] = bos_id
      tok_list.extend(gap_tokens)
      type_list.extend([image_processor.TEXT] * len(gap_tokens))
      if gap_tokens:
        prompt_parts.append(tokenizer.decode(gap_tokens, skip_special_tokens=False))
      if gap_idx < spec.num_images:
        pimg = prepared_imgs[gap_idx]
        span_len = int(pimg.token_types.numel())
        image_inputs.append(
            image_processor.ImageInput(
                start=len(tok_list),
                patches=pimg.patches_bf16,
                n_vit_h=pimg.n_vit_h,
                n_vit_w=pimg.n_vit_w,
                types=pimg.token_types,
            )
        )
        tok_list.extend([args.image_token_id] * span_len)
        type_list.extend(int(t) for t in pimg.token_types.tolist())
        prompt_parts.append(f"<image>{pimg.source_id}</image>")

    prompt_str = "".join(prompt_parts)
    if len(tok_list) != spec.target_seq_len:
      raise ValueError(f"Sample {spec.sample_id} length {len(tok_list)} != target_seq_len {spec.target_seq_len}")

  seq_len = len(tok_list)
  if seq_len > args.max_seq_len:
    raise ValueError(f"Sample {spec.sample_id} seq_len={seq_len} exceeds model max_seq_len={args.max_seq_len}")

  tokens_np = np.asarray(tok_list, dtype=np.int32)
  token_types_np = np.asarray(type_list, dtype=np.int32)
  positions_np = np.arange(seq_len, dtype=np.int32)

  # Strict alignment check between tokens, token_types, and ImageInput spans.
  is_img_tok = tokens_np == args.image_token_id
  is_img_type = token_types_np >= 0
  if not np.array_equal(is_img_tok, is_img_type):
    raise ValueError(f"Token/type alignment failure on sample {spec.sample_id}")

  if spec.sample_id in PREFILL_SAMPLE_IDS:
    validate_no_duplicate_text_ngrams(tokens_np, token_types_np, sample_id=spec.sample_id)

  span_starts: list[int] = []
  num_patches = target_grid_hw[0] * target_grid_hw[1]
  patch_dim = 3 * args.vision_patch_size * args.vision_patch_size
  patches_padded = np.zeros((_MAX_IMAGE_SLOTS, num_patches, patch_dim), dtype=np.float32)
  image_masks = np.zeros((_MAX_IMAGE_SLOTS,), dtype=np.int32)

  prev_end = 0
  for slot_idx, (img_in, pimg) in enumerate(zip(image_inputs, prepared_imgs, strict=True)):
    span_len = int(img_in.types.numel())
    if span_len != expected_span:
      raise ValueError(f"Sample {spec.sample_id} slot {slot_idx} span_len={span_len} != {expected_span}")
    if img_in.start < prev_end or img_in.start + span_len > seq_len:
      raise ValueError(f"Overlapping or out-of-bounds image span in {spec.sample_id}")
    prev_end = img_in.start + span_len
    if not np.array_equal(token_types_np[img_in.start : prev_end], img_in.types.numpy()):
      raise ValueError(f"Span token_types mismatch in {spec.sample_id} slot {slot_idx}")
    span_starts.append(int(img_in.start))
    patches_padded[slot_idx] = pimg.patches_flat_fp32
    image_masks[slot_idx] = 1

  return BuiltSampleInputs(
      sample_id=spec.sample_id,
      prompt=prompt_str,
      tokens=tokens_np,
      token_types=token_types_np,
      positions=positions_np,
      patches_padded=patches_padded,
      image_masks=image_masks,
      image_inputs=image_inputs,
      grid_hw=target_grid_hw,
      span_starts=tuple(span_starts),
      image_source_ids=tuple(p.source_id for p in prepared_imgs),
      image_source_sha256s=tuple(p.source_sha256 for p in prepared_imgs),
      prepared_png_sha256s=tuple(p.prepared_png_sha256 for p in prepared_imgs),
      base_asset_sha256s=tuple(p.base_asset_sha256 for p in prepared_imgs),
  )


def _compute_kl_stats(golden_fp32_logits: torch.Tensor, diag_logits: torch.Tensor) -> dict[str, float]:
  """Computes per-token `D_KL(P_golden || Q_diag)` max and mean in float64."""
  p_golden = F.softmax(golden_fp32_logits.double(), dim=-1)
  log_p_golden = F.log_softmax(golden_fp32_logits.double(), dim=-1)
  log_q_diag = F.log_softmax(diag_logits.double(), dim=-1)
  kl_per_token = (p_golden * (log_p_golden - log_q_diag)).sum(dim=-1)
  kl_per_token = kl_per_token.clamp_min(0.0)
  return {
      "max_kl": float(kl_per_token.max().item()),
      "mean_kl": float(kl_per_token.mean().item()),
  }


def compute_prescreen_kl_metrics(
    primary_fp32_logits: torch.Tensor,
    perturbed_fp32_logits: torch.Tensor,
) -> dict[str, float]:
  """Computes Option B-1 `C2` `float64` `KL(P_primary || Q_pert)` and top-1 agreement."""
  if primary_fp32_logits.shape != perturbed_fp32_logits.shape:
    raise ValueError(
        f"Logit shape mismatch in compute_prescreen_kl_metrics: "
        f"{tuple(primary_fp32_logits.shape)} vs {tuple(perturbed_fp32_logits.shape)}"
    )
  p_ref = F.softmax(primary_fp32_logits.double(), dim=-1)
  log_p_ref = F.log_softmax(primary_fp32_logits.double(), dim=-1)
  log_q_pert = F.log_softmax(perturbed_fp32_logits.double(), dim=-1)
  kl_per_token = (p_ref * (log_p_ref - log_q_pert)).sum(dim=-1).clamp_min(0.0)
  kl_np = kl_per_token.detach().cpu().numpy().astype(np.float64)
  ref_top1 = primary_fp32_logits.argmax(dim=-1)
  pert_top1 = perturbed_fp32_logits.argmax(dim=-1)
  top1_acc = float((ref_top1 == pert_top1).double().mean().item())
  return {
      "max_kl": float(kl_np.max()),
      "mean_kl": float(kl_np.mean()),
      "p95_kl": float(np.percentile(kl_np, 95)),
      "top1_agreement": top1_acc,
  }


def _image_positions_for_slots(built: BuiltSampleInputs, slots: Sequence[int]) -> np.ndarray:
  """Returns sorted `IMAGE` patch-grid token positions (`token_type == 1`) within `slots`."""
  if not slots:
    raise ValueError(f"Expected non-empty screened slots for sample {built.sample_id!r}")
  positions: list[int] = []
  for slot_idx in slots:
    if slot_idx < 0 or slot_idx >= len(built.span_starts):
      raise ValueError(f"Slot index {slot_idx} out of bounds for {built.sample_id!r} ({len(built.span_starts)} slots)")
    start = int(built.span_starts[slot_idx])
    span_len = int(built.image_inputs[slot_idx].types.numel())
    for pos in range(start, start + span_len):
      if int(built.token_types[pos]) == image_processor.IMAGE:
        positions.append(pos)
  if not positions:
    raise ValueError(f"No IMAGE (token_type == 1) positions found in slots {list(slots)} of {built.sample_id!r}")
  return np.asarray(sorted(positions), dtype=np.int64)


def evaluate_c1_routing_margins(
    trace: reference_loader.RoutingTrace,
    built: BuiltSampleInputs,
    screened_slots: Sequence[int],
    criteria: PrescreenCriteria,
) -> dict[str, Any]:
  """Evaluates Option B-1 `C1` routing margins over `IMAGE` tokens in `screened_slots`.

  Args:
    trace: `RoutingTrace` captured during the primary `fp32_compute=True` pass.
    built: `BuiltSampleInputs` describing sequence token types and span boundaries.
    screened_slots: Image slot indices (`0..num_images-1`) being screened.
    criteria: `PrescreenCriteria` thresholds.

  Returns:
    Dictionary containing `"passed"`, `"failed_criteria"`, `"worst_gate_margin_l0_19"`,
    `"min_gate_margin_by_layer_image_tokens"`, and `"min_indexer_margin_by_layer_image_tokens"`.
  """
  img_pos = _image_positions_for_slots(built, screened_slots)
  n_layers = int(trace.gate_margins.shape[0])
  early_end = min(_EARLY_ROUTER_LAYER_LIMIT, n_layers)
  if n_layers <= 0:
    raise ValueError("RoutingTrace must contain at least 1 layer")

  gate_sub = trace.gate_margins[:, img_pos]
  min_gate_by_layer = [float(gate_sub[layer_idx].min()) for layer_idx in range(n_layers)]

  early_sub = gate_sub[:early_end]
  flat_worst_idx = int(np.argmin(early_sub))
  worst_layer = int(flat_worst_idx // len(img_pos))
  worst_pos = int(img_pos[flat_worst_idx % len(img_pos)])
  worst_margin = float(trace.gate_margins[worst_layer, worst_pos])
  top6_expert = int(trace.gate_boundary_experts[worst_layer, worst_pos, 0])
  top7_expert = int(trace.gate_boundary_experts[worst_layer, worst_pos, 1])

  min_indexer_by_layer: list[float | None] = []
  early_active_indexer_mins: list[float] = []
  for layer_idx in range(n_layers):
    layer_idx_arr = trace.indexer_margins[layer_idx]
    if layer_idx_arr is None:
      min_indexer_by_layer.append(None)
      continue
    vals = layer_idx_arr[img_pos]
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
      min_indexer_by_layer.append(None)
      continue
    layer_min = float(finite_vals.min())
    min_indexer_by_layer.append(layer_min)
    if layer_idx < early_end:
      early_active_indexer_mins.append(layer_min)

  failed: list[str] = []
  if worst_margin < criteria.tau_gate_l0_19_image_tokens:
    failed.append("C1_gate_margin_l0_19")
  if min(min_gate_by_layer) <= criteria.min_gate_l0_39_image_tokens_gt:
    failed.append("C1_gate_margin_all_layers_gt0")
  if early_active_indexer_mins and min(early_active_indexer_mins) <= criteria.min_indexer_l0_19_image_tokens_gt:
    failed.append("C1_indexer_margin_l0_19_gt0")

  return {
      "passed": not failed,
      "failed_criteria": failed,
      "worst_gate_margin_l0_19": {
          "margin": worst_margin,
          "layer": worst_layer,
          "pos": worst_pos,
          "token_type": int(built.token_types[worst_pos]),
          "top6_expert": top6_expert,
          "top7_expert": top7_expert,
      },
      "min_gate_margin_by_layer_image_tokens": min_gate_by_layer,
      "min_indexer_margin_by_layer_image_tokens": min_indexer_by_layer,
  }


def evaluate_c2_perturbation_metrics(
    metrics: dict[str, float],
    *,
    tag_prefix: str,
    criteria: PrescreenCriteria,
) -> list[str]:
  """Checks one `C2` perturbation KL dictionary against `criteria`, returning failed tags."""
  failed: list[str] = []
  if metrics["max_kl"] > criteria.perturbation_max_kl_le:
    failed.append(f"{tag_prefix}_max_kl")
  if metrics["mean_kl"] > criteria.perturbation_mean_kl_le:
    failed.append(f"{tag_prefix}_mean_kl")
  if metrics["p95_kl"] > criteria.perturbation_p95_kl_le:
    failed.append(f"{tag_prefix}_p95_kl")
  if metrics["top1_agreement"] < criteria.perturbation_top1_agreement_ge:
    failed.append(f"{tag_prefix}_top1_agreement")
  return failed


def evaluate_sample_prescreen(
    model: multimodal.MultimodalTransformer,
    built: BuiltSampleInputs,
    *,
    screened_slots: Sequence[int],
    attempt_index: int,
    queue_ranks: Sequence[int],
    primary_threads: int,
    perturbation_threads: int,
    criteria: PrescreenCriteria,
) -> tuple[bool, torch.Tensor, reference_loader.RoutingTrace, torch.Tensor | None, dict[str, Any]]:
  """Runs Option B-1 `C1` (with short-circuit) and `C2` (`64t` + `one_ulp`) on `built`.

  Args:
    model: Loaded reference `MultimodalTransformer` (`fp32_compute=True`).
    built: Candidate `BuiltSampleInputs`.
    screened_slots: Slot indices (`0..num_images-1`) whose `IMAGE` tokens are checked by `C1`.
    attempt_index: Zero-based attempt counter for `built.sample_id`.
    queue_ranks: Queue ranks newly drawn for this attempt (`[]` for initial `L2/L3/M2` screen).
    primary_threads: PyTorch intra-op thread count for primary (`96t`) and `C2b` (`one_ulp`) passes.
    perturbation_threads: PyTorch intra-op thread count for `C2a` (`64t`) pass.
    criteria: `PrescreenCriteria` thresholds.

  Returns:
    `(accepted, primary_logits_2d, primary_trace, ulp_logits_2d_or_none, audit_record)`.
  """
  if primary_threads < 1 or perturbation_threads < 1:
    raise ValueError(
        f"Expected positive thread counts, got primary_threads={primary_threads}, "
        f"perturbation_threads={perturbation_threads}"
    )
  input_ids_pt = torch.from_numpy(built.tokens).to(torch.int64).unsqueeze(0)
  token_types_pt = torch.from_numpy(built.token_types).to(torch.int64).unsqueeze(0)

  torch.set_num_threads(primary_threads)
  logits_3d, trace = reference_loader.forward_multimodal_logits_with_routing_trace(
      model,
      input_ids_pt,
      images=[built.image_inputs],
      token_types=token_types_pt,
  )
  primary_logits_2d = logits_3d[0].cpu()
  if not torch.isfinite(primary_logits_2d).all():
    raise ValueError(f"Non-finite primary golden logits produced on sample {built.sample_id}")

  c1_eval = evaluate_c1_routing_margins(trace, built, screened_slots, criteria)
  failed_criteria = list(c1_eval["failed_criteria"])
  pert_64t_metrics: dict[str, float] | None = None
  pert_ulp_metrics: dict[str, float] | None = None
  ulp_logits_2d: torch.Tensor | None = None

  # C1 short-circuit: skip C2a (64t) and C2b (one_ulp) whenever C1 fails on primary pass.
  if c1_eval["passed"]:
    try:
      torch.set_num_threads(perturbation_threads)
      pert_64t_logits_2d = reference_loader.forward_multimodal_logits(
          model,
          input_ids_pt,
          images=[built.image_inputs],
          token_types=token_types_pt,
      )[0].cpu()
    finally:
      torch.set_num_threads(primary_threads)
    if not torch.isfinite(pert_64t_logits_2d).all():
      raise ValueError(f"Non-finite C2a perturbation logits produced on sample {built.sample_id}")
    pert_64t_metrics = compute_prescreen_kl_metrics(primary_logits_2d, pert_64t_logits_2d)
    failed_criteria.extend(evaluate_c2_perturbation_metrics(pert_64t_metrics, tag_prefix="C2a_64t", criteria=criteria))

    with reference_loader.one_ulp_routing_context():
      ulp_logits_2d = reference_loader.forward_multimodal_logits(
          model,
          input_ids_pt,
          images=[built.image_inputs],
          token_types=token_types_pt,
      )[0].cpu()
    if not torch.isfinite(ulp_logits_2d).all():
      raise ValueError(f"Non-finite C2b 1-ULP perturbation logits produced on sample {built.sample_id}")
    pert_ulp_metrics = compute_prescreen_kl_metrics(primary_logits_2d, ulp_logits_2d)
    failed_criteria.extend(
        evaluate_c2_perturbation_metrics(pert_ulp_metrics, tag_prefix="C2b_one_ulp", criteria=criteria)
    )

  accepted = not failed_criteria
  audit_record: dict[str, Any] = {
      "sample_id": built.sample_id,
      "attempt_index": int(attempt_index),
      "queue_ranks": [int(r) for r in queue_ranks],
      "screened_slots": [int(s) for s in screened_slots],
      "slot_source_ids": list(built.image_source_ids),
      "candidate_source_ids": [built.image_source_ids[s] for s in screened_slots],
      "base_asset_sha256s": [built.base_asset_sha256s[s] for s in screened_slots],
      "prepared_png_sha256s": [built.prepared_png_sha256s[s] for s in screened_slots],
      "accepted": accepted,
      "failed_criteria": failed_criteria,
      "worst_gate_margin_l0_19": c1_eval["worst_gate_margin_l0_19"],
      "min_gate_margin_by_layer_image_tokens": c1_eval["min_gate_margin_by_layer_image_tokens"],
      "min_indexer_margin_by_layer_image_tokens": c1_eval["min_indexer_margin_by_layer_image_tokens"],
      "perturbation_64t_vs_96t": pert_64t_metrics,
      "perturbation_one_ulp_vs_96t": pert_ulp_metrics,
  }
  return accepted, primary_logits_2d, trace, ulp_logits_2d, audit_record


def _draw_next_candidate_rank(
    remaining_ranks: list[int],
    candidate_queue: Sequence[CandidateCropSpec],
    *,
    forbid_duplicate_test_image: bool,
    sample_id: str,
) -> CandidateCropSpec:
  """Pops the lowest available rank from `remaining_ranks` (deferring `B3:anchor_0:mirror_0` when required)."""
  for list_idx, rank in enumerate(remaining_ranks):
    spec = candidate_queue[rank]
    if forbid_duplicate_test_image and spec.source_id == _DUPLICATE_TEST_IMAGE_CROP_ID:
      continue
    remaining_ranks.pop(list_idx)
    return spec
  raise RuntimeError(
      f"Exhausted 72-candidate mm_v2 queue while screening sample {sample_id!r} " f"(remaining_ranks={remaining_ranks})"
  )


def _save_sample_outputs(
    out_path: pathlib.Path,
    built: BuiltSampleInputs,
    logits_np: np.ndarray,
    *,
    ref_topk_experts: np.ndarray | None,
    routing_trace: reference_loader.RoutingTrace | None,
) -> tuple[str, str, dict[str, Any], dict[str, Any] | None]:
  """Writes `<sample_id>.npz` and builds manifest + optional `mm_logit.pkl` entries."""
  npz_name = f"{built.sample_id}.npz"
  npz_path = out_path / npz_name
  npz_kwargs: dict[str, Any] = {
      "tokens": built.tokens,
      "token_types": built.token_types,
      "patches": built.patches_padded,
      "image_masks": built.image_masks,
      "positions": built.positions,
      "logits": logits_np,
      "prompt": np.array(built.prompt),
      "sample_id": np.array(built.sample_id),
      "seq_len": np.int32(len(built.tokens)),
      "num_images": np.int32(len(built.image_inputs)),
      "grid_hw": np.asarray(built.grid_hw, dtype=np.int32),
      "span_starts": np.asarray(built.span_starts, dtype=np.int32),
      "image_source_ids": np.asarray(built.image_source_ids),
      "image_source_sha256s": np.asarray(built.image_source_sha256s),
  }
  if ref_topk_experts is not None:
    if ref_topk_experts.dtype != np.uint16:
      raise ValueError(f"Expected uint16 ref_topk_experts, got {ref_topk_experts.dtype}")
    npz_kwargs["ref_topk_experts"] = ref_topk_experts
  if routing_trace is not None:
    npz_kwargs["ref_gate_margins"] = np.asarray(routing_trace.ref_gate_margins, dtype=np.float32)
    npz_kwargs["ref_gate_boundary_ulp"] = np.asarray(routing_trace.ref_gate_boundary_ulp, dtype=np.float32)
    npz_kwargs["ref_indexer_layer_ids"] = np.asarray(routing_trace.ref_indexer_layer_ids, dtype=np.int32)
    npz_kwargs["ref_indexer_topk"] = np.asarray(routing_trace.ref_indexer_topk, dtype=np.int32)
    npz_kwargs["ref_indexer_margins"] = np.asarray(routing_trace.ref_indexer_margins, dtype=np.float32)
    npz_kwargs["ref_indexer_boundary_ulp"] = np.asarray(routing_trace.ref_indexer_boundary_ulp, dtype=np.float32)
  np.savez(npz_path, **npz_kwargs)
  npz_sha = _sha256_file(npz_path)

  sample_entry = dataclasses.asdict(
      GoldenSampleResult(
          sample_id=built.sample_id,
          seq_len=int(len(built.tokens)),
          num_images=int(len(built.image_inputs)),
          grid_hw=built.grid_hw,
          span_starts=built.span_starts,
          image_source_ids=built.image_source_ids,
          image_source_sha256s=built.image_source_sha256s,
          npz_filename=npz_name,
          npz_sha256=npz_sha,
      )
  )
  if ref_topk_experts is not None:
    sample_entry["ref_topk_experts_shape"] = [int(d) for d in ref_topk_experts.shape]
    sample_entry["ref_topk_experts_dtype"] = str(ref_topk_experts.dtype)
  if routing_trace is not None:
    sample_entry["ref_gate_margins_shape"] = [int(d) for d in routing_trace.ref_gate_margins.shape]
    sample_entry["ref_gate_boundary_ulp_shape"] = [int(d) for d in routing_trace.ref_gate_boundary_ulp.shape]
    sample_entry["ref_indexer_layer_ids"] = [int(x) for x in routing_trace.ref_indexer_layer_ids.tolist()]
    sample_entry["ref_indexer_topk_shape"] = [int(d) for d in routing_trace.ref_indexer_topk.shape]
    sample_entry["ref_indexer_margins_shape"] = [int(d) for d in routing_trace.ref_indexer_margins.shape]
    sample_entry["ref_indexer_boundary_ulp_shape"] = [int(d) for d in routing_trace.ref_indexer_boundary_ulp.shape]

  pkl_entry: dict[str, Any] | None = None
  if built.sample_id in LOGIT_SAMPLE_IDS:
    pkl_entry = {
        "sample_id": built.sample_id,
        "prompt": built.prompt,
        "tokens": built.tokens,
        "logits": logits_np,
        "token_types": built.token_types,
        "encoder_images": built.patches_padded,
        "encoder_image_masks": built.image_masks,
        "pixel_values": built.patches_padded,
        "grid_hw": np.asarray(built.grid_hw, dtype=np.int32),
        "num_images": int(len(built.image_inputs)),
        "image_source_ids": list(built.image_source_ids),
        "image_source_sha256s": list(built.image_source_sha256s),
    }
    if ref_topk_experts is not None:
      pkl_entry["ref_topk_experts"] = ref_topk_experts
    if routing_trace is not None:
      pkl_entry["ref_gate_margins"] = np.asarray(routing_trace.ref_gate_margins, dtype=np.float32)
      pkl_entry["ref_gate_boundary_ulp"] = np.asarray(routing_trace.ref_gate_boundary_ulp, dtype=np.float32)
      pkl_entry["ref_indexer_layer_ids"] = np.asarray(routing_trace.ref_indexer_layer_ids, dtype=np.int32)
      pkl_entry["ref_indexer_topk"] = np.asarray(routing_trace.ref_indexer_topk, dtype=np.int32)
      pkl_entry["ref_indexer_margins"] = np.asarray(routing_trace.ref_indexer_margins, dtype=np.float32)
      pkl_entry["ref_indexer_boundary_ulp"] = np.asarray(routing_trace.ref_indexer_boundary_ulp, dtype=np.float32)
  return npz_name, npz_sha, sample_entry, pkl_entry


def _screen_sample_until_accepted(
    *,
    base_spec: SampleSpec,
    initial_sources: list[str],
    screened_slots: tuple[int, ...],
    initial_ranks: list[int],
    model: multimodal.MultimodalTransformer,
    ckpt_path: pathlib.Path,
    tokenizer: Any,
    margs: ref_model.ModelArgs,
    c4_pool: Sequence[int],
    image_cache: dict[str, PreparedImage],
    mini: bool,
    release_images_dir: str,
    base_assets: dict[str, CandidateBaseAsset],
    candidate_queue: Sequence[CandidateCropSpec],
    remaining_ranks: list[int],
    primary_threads: int,
    pert_threads: int,
    criteria: PrescreenCriteria,
) -> tuple[
    BuiltSampleInputs,
    torch.Tensor,
    reference_loader.RoutingTrace,
    torch.Tensor | None,
    list[dict[str, Any]],
    bool,
]:
  """Iterates candidate replacements on `base_spec` until `C1 + C2` pass."""
  current_sources = list(initial_sources)
  pending_ranks = list(initial_ranks)
  attempts: list[dict[str, Any]] = []
  initial_c1_failed = False
  attempt_idx = 0

  while True:
    cand_spec = dataclasses.replace(base_spec, image_sources=tuple(current_sources))
    built = build_sample_inputs(
        cand_spec,
        ckpt_path,
        tokenizer,
        margs,
        c4_pool,
        image_cache,
        mini=mini,
        release_images_dir=release_images_dir,
        base_assets=base_assets,
    )
    accepted, logits_2d, trace, ulp_logits_2d, audit_rec = evaluate_sample_prescreen(
        model,
        built,
        screened_slots=screened_slots,
        attempt_index=attempt_idx,
        queue_ranks=pending_ranks,
        primary_threads=primary_threads,
        perturbation_threads=pert_threads,
        criteria=criteria,
    )
    attempts.append(audit_rec)
    if attempt_idx == 0 and any(tag.startswith("C1_") for tag in audit_rec["failed_criteria"]):
      initial_c1_failed = True
    if accepted:
      return built, logits_2d, trace, ulp_logits_2d, attempts, initial_c1_failed

    if any(tag.startswith("C1_") for tag in audit_rec["failed_criteria"]):
      slots_to_replace = [
          slot_i
          for slot_i in screened_slots
          if not evaluate_c1_routing_margins(trace, built, (slot_i,), criteria)["passed"]
      ]
    else:
      slots_to_replace = list(screened_slots)

    pending_ranks = []
    for slot_i in slots_to_replace:
      has_fixed_test_img = any(
          idx != slot_i and current_sources[idx] == "asset:test_image.jpg" for idx in range(base_spec.num_images)
      )
      crop = _draw_next_candidate_rank(
          remaining_ranks,
          candidate_queue,
          forbid_duplicate_test_image=has_fixed_test_img,
          sample_id=base_spec.sample_id,
      )
      current_sources[slot_i] = crop.source_id
      pending_ranks.append(crop.rank)
    attempt_idx += 1


def _parse_sample_ids(samples_arg: str) -> tuple[str, ...]:
  """Parses `--samples` (`all` or comma-separated IDs) in canonical suite order."""
  cleaned = samples_arg.strip()
  if cleaned == "all":
    return ALL_SAMPLE_IDS
  requested = [item.strip() for item in cleaned.split(",") if item.strip()]
  if not requested:
    raise ValueError("--samples must not be empty")
  unknown = [sid for sid in requested if sid not in ALL_SAMPLE_IDS]
  if unknown:
    raise ValueError(f"Unknown sample IDs in --samples: {unknown}; valid={ALL_SAMPLE_IDS}")
  requested_set = set(requested)
  return tuple(sid for sid in ALL_SAMPLE_IDS if sid in requested_set)


def populate_mm_v3_manifest_metadata(
    manifest: dict[str, Any],
    ordered_sample_entries: Sequence[Mapping[str, Any]],
    *,
    ref_indexer_layer_ids: Sequence[int] = (2, 8, 14, 20, 24, 28, 32, 36),
    ref_indexer_topk: int = 512,
) -> dict[str, Any]:
  """Populates Option C (`mm_v3`) provenance, routing dtypes/shapes, and 26-slot image assignment metadata."""
  slot_records: list[dict[str, Any]] = []
  slot_index = 0
  for sample_entry in ordered_sample_entries:
    sid = str(sample_entry["sample_id"])
    src_ids = list(sample_entry["image_source_ids"])
    src_shas = list(sample_entry["image_source_sha256s"])
    for sample_image_index, (src_id_raw, sha_raw) in enumerate(zip(src_ids, src_shas, strict=True)):
      src_id = str(src_id_raw)
      img_sha = str(sha_raw)
      rank_val: int | None = None
      crop_box_val: list[int] | None = None
      if src_id.startswith(f"crop_{_CANDIDATE_CROP_SIDE}:"):
        base_id, anchor_idx, mirror = _parse_crop_source_key(src_id)
        base_idx = CANDIDATE_BASE_IDS.index(base_id)
        rank_val = int(
            mirror * (len(CANDIDATE_ANCHORS) * len(CANDIDATE_BASE_IDS)) + anchor_idx * len(CANDIDATE_BASE_IDS) + base_idx
        )
        cbox = _compute_square_crop_box(
            _DEFAULT_BASE_ASSET_SIZES[base_id],
            CANDIDATE_ANCHORS[anchor_idx],
        )
        crop_box_val = [int(c) for c in cbox]
      slot_records.append(
          {
              "slot_index": int(slot_index),
              "sample_id": sid,
              "sample_image_index": int(sample_image_index),
              "image_source_id": src_id,
              "source_id": src_id,
              "image_sha256": img_sha,
              "sha256": img_sha,
              "rank": rank_val,
              "crop_box": crop_box_val,
          }
      )
      slot_index += 1

  unique_shas = {r["image_sha256"] for r in slot_records}
  manifest["option_c_spec_version"] = OPTION_C_SPEC_VERSION
  manifest["option_c_spec_sha256"] = OPTION_C_SPEC_SHA256
  manifest["ref_topk_experts_dtype"] = "uint16"
  manifest["ref_gate_margins_dtype"] = "float32"
  manifest["ref_gate_boundary_ulp_dtype"] = "float32"
  manifest["ref_indexer_topk_dtype"] = "int32"
  manifest["ref_indexer_margins_dtype"] = "float32"
  manifest["ref_indexer_boundary_ulp_dtype"] = "float32"
  manifest["ref_indexer_layer_ids"] = [int(x) for x in ref_indexer_layer_ids]
  manifest["ref_indexer_topk"] = int(ref_indexer_topk)
  manifest["distinct_slot_image_assignment"] = {
      "total_image_slots": len(slot_records),
      "unique_slot_sha256_count": len(unique_shas),
      "all_slots_distinct": len(slot_records) == 26 and len(unique_shas) == 26,
      "slot_records": slot_records,
  }
  return manifest


def run_generation(
    *,
    output_dir: str | pathlib.Path,
    checkpoint_dir: str = "",
    mini: bool = False,
    mini_ckpt_dir: str = "",
    tokenizer_path: str = "",
    lazy_dequant: bool = True,
    samples: str = "all",
    num_threads: int = 0,
    perturbation_threads: int = 0,
    golden_version: str = "mm_v1",
    release_images_dir: str = "",
    prescreen_criteria: PrescreenCriteria | None = None,
    diagnostics: bool = True,
    u2_commit_sha: str = "",
    cli_argv: Sequence[str] = (),
) -> dict[str, Any]:
  """Executes multimodal golden-logit generation and writes `.npz`, `.pkl`, and `manifest.json`."""
  t_start = time.monotonic()
  if golden_version not in GOLDEN_VERSIONS:
    raise ValueError(f"Unsupported golden_version={golden_version!r}; valid={GOLDEN_VERSIONS}")
  if num_threads > 0:
    torch.set_num_threads(num_threads)

  selected_ids = _parse_sample_ids(samples)
  u2_prov = verify_u2_provenance()
  resolved_u2_sha = _resolve_u2_commit_sha(u2_commit_sha)

  out_path = pathlib.Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  temp_ckpt_dir: str | None = None
  try:
    if mini:
      if checkpoint_dir:
        ckpt_path = pathlib.Path(checkpoint_dir)
      elif mini_ckpt_dir:
        ckpt_path = build_quantized_mini_checkpoint(mini_ckpt_dir, tokenizer_path=tokenizer_path)
      else:
        temp_ckpt_dir = tempfile.mkdtemp(prefix="ds41_mm_mini_ckpt_")
        ckpt_path = build_quantized_mini_checkpoint(temp_ckpt_dir, tokenizer_path=tokenizer_path)
      max_seq_len = 128
    else:
      if not checkpoint_dir:
        raise ValueError("--checkpoint-dir is required unless --mini is set")
      ckpt_path = pathlib.Path(checkpoint_dir)
      max_seq_len = 2048

    index_file = ckpt_path / "model.safetensors.index.json"
    config_file = ckpt_path / "config.json"
    if not index_file.exists():
      raise ValueError(f"Missing model.safetensors.index.json in {ckpt_path}")
    if not config_file.exists():
      raise ValueError(f"Missing config.json in {ckpt_path}")

    tok_src = _resolve_tokenizer_source(str(ckpt_path), tokenizer_path, mini=mini)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    model, margs = reference_loader.load_multimodal_reference_model(
        str(ckpt_path),
        dtype="bf16",
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        fp32_compute=True,
        lazy_dequant=lazy_dequant,
    )
    model_bytes = reference_loader.resident_model_bytes(model)

    suite = build_sample_suite(mini=mini, golden_version=golden_version)
    c4_pool = load_c4_token_pool(tokenizer, margs.image_token_id)
    image_cache: dict[str, PreparedImage] = {}

    built_samples: dict[str, BuiltSampleInputs] = {}
    golden_logits_by_id: dict[str, torch.Tensor] = {}
    accepted_ulp_logits_by_id: dict[str, torch.Tensor] = {}
    sample_entries_by_id: dict[str, dict[str, Any]] = {}
    pkl_entries_by_id: dict[str, dict[str, Any]] = {}
    output_files_sha256: dict[str, str] = {}
    prescreen_audit: dict[str, Any] | None = None

    if golden_version == "mm_v1":
      for sid in selected_ids:
        built = build_sample_inputs(
            suite[sid],
            ckpt_path,
            tokenizer,
            margs,
            c4_pool,
            image_cache,
            mini=mini,
            release_images_dir=release_images_dir,
        )
        built_samples[sid] = built
        input_ids_pt = torch.from_numpy(built.tokens).to(torch.int64).unsqueeze(0)
        token_types_pt = torch.from_numpy(built.token_types).to(torch.int64).unsqueeze(0)
        logits_3d = reference_loader.forward_multimodal_logits(
            model,
            input_ids_pt,
            images=[built.image_inputs],
            token_types=token_types_pt,
        )
        logits_2d = logits_3d[0].cpu()
        if not torch.isfinite(logits_2d).all():
          raise ValueError(f"Non-finite golden logits produced on sample {sid}")
        golden_logits_by_id[sid] = logits_2d
        logits_np = logits_2d.numpy().astype(np.float32)

        npz_name, npz_sha, sample_entry, pkl_entry = _save_sample_outputs(
            out_path,
            built,
            logits_np,
            ref_topk_experts=None,
            routing_trace=None,
        )
        output_files_sha256[npz_name] = npz_sha
        sample_entries_by_id[sid] = sample_entry
        if pkl_entry is not None:
          pkl_entries_by_id[sid] = pkl_entry
    elif golden_version == "mm_v3":
      primary_threads = (
          num_threads if num_threads > 0 else (_DEFAULT_MINI_PRIMARY_THREADS if mini else _DEFAULT_MM_V2_PRIMARY_THREADS)
      )
      torch.set_num_threads(primary_threads)
      verify_release_sha = (not mini) or (not bool(release_images_dir))
      base_assets = load_mm_v2_base_assets(
          ckpt_path,
          release_images_dir=release_images_dir,
          verify_release_sha256=verify_release_sha,
      )
      distinct_queue = build_mm_v3_distinct_crop_queue(base_assets)
      if len(distinct_queue) < len(MM_V3_DISTINCT_CROP_SOURCE_IDS):
        raise ValueError(
            f"mm_v3 requires {len(MM_V3_DISTINCT_CROP_SOURCE_IDS)} distinct candidate crops, "
            f"got {len(distinct_queue)}"
        )
      expected_distinct_ids = tuple(spec.source_id for spec, _ in distinct_queue[: len(MM_V3_DISTINCT_CROP_SOURCE_IDS)])
      if expected_distinct_ids != MM_V3_DISTINCT_CROP_SOURCE_IDS:
        raise ValueError(
            f"mm_v3 distinct crop queue order mismatch: expected {MM_V3_DISTINCT_CROP_SOURCE_IDS}, "
            f"got {expected_distinct_ids}"
        )

      seen_slot_sha256s: dict[str, tuple[str, int, str]] = {}
      for sid in selected_ids:
        built = build_sample_inputs(
            suite[sid],
            ckpt_path,
            tokenizer,
            margs,
            c4_pool,
            image_cache,
            mini=mini,
            release_images_dir=release_images_dir,
            base_assets=base_assets,
        )
        for slot_i, (src_id, slot_sha) in enumerate(zip(built.image_source_ids, built.image_source_sha256s, strict=True)):
          if slot_sha in seen_slot_sha256s:
            prev_sid, prev_slot, prev_src = seen_slot_sha256s[slot_sha]
            raise ValueError(
                f"Duplicate image_source_sha256 {slot_sha} in mm_v3 between "
                f"{prev_sid}[{prev_slot}] ({prev_src}) and {sid}[{slot_i}] ({src_id})"
            )
          seen_slot_sha256s[slot_sha] = (sid, slot_i, src_id)
        built_samples[sid] = built

        input_ids_pt = torch.from_numpy(built.tokens).to(torch.int64).unsqueeze(0)
        token_types_pt = torch.from_numpy(built.token_types).to(torch.int64).unsqueeze(0)
        t_sample = time.monotonic()
        logits_3d, trace = reference_loader.forward_multimodal_logits_with_routing_trace(
            model,
            input_ids_pt,
            images=[built.image_inputs],
            token_types=token_types_pt,
        )
        elapsed_sample = time.monotonic() - t_sample
        print(
            f"[MM_V3_PRIMARY] sample={sid} seq_len={len(built.tokens)} "
            f"threads={int(torch.get_num_threads())} elapsed_sec={elapsed_sample:.2f}s",
            flush=True,
        )
        logits_2d = logits_3d[0].cpu()
        if not torch.isfinite(logits_2d).all():
          raise ValueError(f"Non-finite primary golden logits produced on mm_v3 sample {sid}")
        golden_logits_by_id[sid] = logits_2d
        logits_np = logits_2d.numpy().astype(np.float32)

        npz_name, npz_sha, sample_entry, pkl_entry = _save_sample_outputs(
            out_path,
            built,
            logits_np,
            ref_topk_experts=trace.ref_topk_experts,
            routing_trace=trace,
        )
        output_files_sha256[npz_name] = npz_sha
        sample_entries_by_id[sid] = sample_entry
        if pkl_entry is not None:
          pkl_entries_by_id[sid] = pkl_entry
    else:
      criteria = prescreen_criteria if prescreen_criteria is not None else PrescreenCriteria()
      primary_threads = (
          num_threads if num_threads > 0 else (_DEFAULT_MINI_PRIMARY_THREADS if mini else _DEFAULT_MM_V2_PRIMARY_THREADS)
      )
      pert_threads = (
          perturbation_threads
          if perturbation_threads > 0
          else (_DEFAULT_MINI_PERTURBATION_THREADS if mini else _DEFAULT_MM_V2_PERTURBATION_THREADS)
      )
      torch.set_num_threads(primary_threads)
      verify_release_sha = (not mini) or (not bool(release_images_dir))
      base_assets = load_mm_v2_base_assets(
          ckpt_path,
          release_images_dir=release_images_dir,
          verify_release_sha256=verify_release_sha,
      )
      candidate_queue = build_mm_v2_candidate_queue(base_assets)
      remaining_ranks = list(range(len(candidate_queue)))
      evaluated_candidates: list[dict[str, Any]] = []
      test_image_failed_c1 = False

      # Phase A: Screen existing all-real-image samples (L2, L3, M2) FIRST on all slots.
      initial_real_ids = [sid for sid in REAL_INITIAL_SCREEN_SAMPLE_IDS if sid in selected_ids]
      queued_ids = [sid for sid in selected_ids if sid not in REAL_INITIAL_SCREEN_SAMPLE_IDS]

      for sid in initial_real_ids:
        base_spec = suite[sid]
        built, logits_2d, trace, ulp_logits_2d, attempts, init_c1_fail = _screen_sample_until_accepted(
            base_spec=base_spec,
            initial_sources=list(base_spec.image_sources),
            screened_slots=tuple(range(base_spec.num_images)),
            initial_ranks=[],
            model=model,
            ckpt_path=ckpt_path,
            tokenizer=tokenizer,
            margs=margs,
            c4_pool=c4_pool,
            image_cache=image_cache,
            mini=mini,
            release_images_dir=release_images_dir,
            base_assets=base_assets,
            candidate_queue=candidate_queue,
            remaining_ranks=remaining_ranks,
            primary_threads=primary_threads,
            pert_threads=pert_threads,
            criteria=criteria,
        )
        evaluated_candidates.extend(attempts)
        if sid in ("L3", "M2") and init_c1_fail:
          test_image_failed_c1 = True
        built_samples[sid] = built
        golden_logits_by_id[sid] = logits_2d
        if ulp_logits_2d is not None:
          accepted_ulp_logits_by_id[sid] = ulp_logits_2d
        npz_name, npz_sha, sample_entry, pkl_entry = _save_sample_outputs(
            out_path,
            built,
            logits_2d.numpy().astype(np.float32),
            ref_topk_experts=trace.ref_topk_experts,
            routing_trace=None,
        )
        output_files_sha256[npz_name] = npz_sha
        sample_entries_by_id[sid] = sample_entry
        if pkl_entry is not None:
          pkl_entries_by_id[sid] = pkl_entry

      # Phase B: Screen remaining samples in canonical synth:101..118 order.
      for sid in queued_ids:
        base_spec = suite[sid]
        current_sources = list(base_spec.image_sources)
        screened_slots = tuple(
            slot_i
            for slot_i, orig_src in enumerate(current_sources)
            if orig_src.startswith("synth:") or (orig_src == "asset:test_image.jpg" and test_image_failed_c1)
        )
        if not screened_slots:
          raise ValueError(f"Expected at least one screened slot in mm_v2 sample {sid!r}")

        has_fixed_test_img = any(
            slot_i not in screened_slots and current_sources[slot_i] == "asset:test_image.jpg"
            for slot_i in range(base_spec.num_images)
        )
        initial_ranks: list[int] = []
        for slot_i in screened_slots:
          crop = _draw_next_candidate_rank(
              remaining_ranks,
              candidate_queue,
              forbid_duplicate_test_image=has_fixed_test_img,
              sample_id=sid,
          )
          current_sources[slot_i] = crop.source_id
          initial_ranks.append(crop.rank)

        built, logits_2d, trace, ulp_logits_2d, attempts, _ = _screen_sample_until_accepted(
            base_spec=base_spec,
            initial_sources=current_sources,
            screened_slots=screened_slots,
            initial_ranks=initial_ranks,
            model=model,
            ckpt_path=ckpt_path,
            tokenizer=tokenizer,
            margs=margs,
            c4_pool=c4_pool,
            image_cache=image_cache,
            mini=mini,
            release_images_dir=release_images_dir,
            base_assets=base_assets,
            candidate_queue=candidate_queue,
            remaining_ranks=remaining_ranks,
            primary_threads=primary_threads,
            pert_threads=pert_threads,
            criteria=criteria,
        )
        evaluated_candidates.extend(attempts)
        built_samples[sid] = built
        golden_logits_by_id[sid] = logits_2d
        if ulp_logits_2d is not None:
          accepted_ulp_logits_by_id[sid] = ulp_logits_2d
        npz_name, npz_sha, sample_entry, pkl_entry = _save_sample_outputs(
            out_path,
            built,
            logits_2d.numpy().astype(np.float32),
            ref_topk_experts=trace.ref_topk_experts,
            routing_trace=None,
        )
        output_files_sha256[npz_name] = npz_sha
        sample_entries_by_id[sid] = sample_entry
        if pkl_entry is not None:
          pkl_entries_by_id[sid] = pkl_entry

      prescreen_audit = {
          "prescreen_spec_version": PRESCREEN_SPEC_VERSION,
          "prescreen_spec_sha256": PRESCREEN_SPEC_SHA256,
          "primary_threads": int(primary_threads),
          "perturbation_threads": int(pert_threads),
          "test_image_failed_initial_c1": bool(test_image_failed_c1),
          "criteria": criteria.to_manifest_dict(),
          "evaluated_candidates": evaluated_candidates,
      }

    ordered_sample_entries = [sample_entries_by_id[sid] for sid in selected_ids]
    pkl_entries = [pkl_entries_by_id[sid] for sid in selected_ids if sid in pkl_entries_by_id]
    if pkl_entries:
      pkl_path = out_path / "mm_logit.pkl"
      with open(pkl_path, "wb") as f:
        pickle.dump(pkl_entries, f, protocol=4)
      output_files_sha256["mm_logit.pkl"] = _sha256_file(pkl_path)

    diagnostics_payload: dict[str, dict[str, dict[str, float]]] = {
        "bf16_compute": {},
        "one_ulp_routing": {},
    }
    if diagnostics:
      diag_targets = [sid for sid in DIAGNOSTIC_SAMPLE_IDS if sid in built_samples]
      if diag_targets:
        with reference_loader.bf16_compute_context(model):
          for sid in diag_targets:
            built = built_samples[sid]
            input_ids_pt = torch.from_numpy(built.tokens).to(torch.int64).unsqueeze(0)
            token_types_pt = torch.from_numpy(built.token_types).to(torch.int64).unsqueeze(0)
            bf16_logits = reference_loader.forward_multimodal_logits(
                model,
                input_ids_pt,
                images=[built.image_inputs],
                token_types=token_types_pt,
            )[0].cpu()
            if not torch.isfinite(bf16_logits).all():
              raise ValueError(f"Non-finite bf16 diagnostic logits on sample {sid}")
            diagnostics_payload["bf16_compute"][sid] = _compute_kl_stats(golden_logits_by_id[sid], bf16_logits)

        if golden_version != "mm_v3":
          ulp_missing = [sid for sid in diag_targets if sid not in accepted_ulp_logits_by_id]
          for sid in diag_targets:
            if sid in accepted_ulp_logits_by_id:
              diagnostics_payload["one_ulp_routing"][sid] = _compute_kl_stats(
                  golden_logits_by_id[sid],
                  accepted_ulp_logits_by_id[sid],
              )
          if ulp_missing:
            with reference_loader.one_ulp_routing_context():
              for sid in ulp_missing:
                built = built_samples[sid]
                input_ids_pt = torch.from_numpy(built.tokens).to(torch.int64).unsqueeze(0)
                token_types_pt = torch.from_numpy(built.token_types).to(torch.int64).unsqueeze(0)
                ulp_logits = reference_loader.forward_multimodal_logits(
                    model,
                    input_ids_pt,
                    images=[built.image_inputs],
                    token_types=token_types_pt,
                )[0].cpu()
                if not torch.isfinite(ulp_logits).all():
                  raise ValueError(f"Non-finite 1-ULP diagnostic logits on sample {sid}")
                diagnostics_payload["one_ulp_routing"][sid] = _compute_kl_stats(golden_logits_by_id[sid], ulp_logits)

    tool_sha256: dict[str, str] = {}
    for fname in _TOOL_FILENAMES:
      rel_key = f"tests/utils/deepseek_v41_mm_golden/{fname}"
      tool_sha256[rel_key] = _sha256_file(_TOOL_DIR / fname)

    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    wall_time_sec = float(time.monotonic() - t_start)

    cli_args_dict: dict[str, Any] = {
        "mini": mini,
        "lazy_dequant": lazy_dequant,
        "samples": list(selected_ids),
        "num_threads": int(torch.get_num_threads()),
        "diagnostics": diagnostics,
        "argv": list(cli_argv),
    }
    if golden_version == "mm_v2":
      cli_args_dict["golden_version"] = golden_version
      cli_args_dict["perturbation_threads"] = int(prescreen_audit["perturbation_threads"])  # type: ignore[index]
    elif golden_version == "mm_v3":
      cli_args_dict["golden_version"] = golden_version

    schema_version_by_golden = {
        "mm_v1": "ds41_mm_golden_v1",
        "mm_v2": "ds41_mm_golden_v2",
        "mm_v3": "ds41_mm_golden_v3",
    }
    manifest: dict[str, Any] = {
        "schema_version": schema_version_by_golden[golden_version],
        "golden_version": golden_version,
        "mini_mode": mini,
        "u2_commit_sha": resolved_u2_sha,
        "release_revision": u2_prov["revision"],
        "u2_provenance": u2_prov,
        "tool_files_sha256": tool_sha256,
        "checkpoint_index_sha256": _sha256_file(index_file),
        "checkpoint_config_sha256": _sha256_file(config_file),
        "cli_args": cli_args_dict,
        "lazy_dequant": lazy_dequant,
        "deviations": list(u2_prov["deviations"]) + list(reference_loader.DEVIATIONS),
        "torch_version": str(torch.__version__),
        "num_threads": int(torch.get_num_threads()),
        "resident_model_bytes": model_bytes,
        "peak_rss_kb": peak_rss_kb,
        "wall_time_sec": wall_time_sec,
        "output_files_sha256": output_files_sha256,
        "samples": ordered_sample_entries,
        "diagnostics": diagnostics_payload,
    }
    if prescreen_audit is not None:
      manifest["prescreen_spec_sha256"] = PRESCREEN_SPEC_SHA256
      manifest["ref_topk_experts_dtype"] = "uint16"
      manifest["prescreen_audit"] = prescreen_audit
    elif golden_version == "mm_v3":
      populate_mm_v3_manifest_metadata(
          manifest,
          ordered_sample_entries,
          ref_indexer_layer_ids=tuple(int(x) for x in margs.index_source_layers),
          ref_indexer_topk=int(margs.index_topk),
      )
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
  finally:
    if temp_ckpt_dir is not None:
      shutil.rmtree(temp_ckpt_dir, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI flags for `generate_mm_golden_logits`."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint-dir", default="", help="Path to HF DeepSeek-V4.1-Flash checkpoint directory.")
  parser.add_argument("--output-dir", required=True, help="Output directory for .npz, mm_logit.pkl, and manifest.json.")
  parser.add_argument("--mini", action="store_true", help="Build and run on a deterministic quantized mini checkpoint.")
  parser.add_argument("--mini-ckpt-dir", default="", help="Optional directory to store the generated --mini checkpoint.")
  parser.add_argument("--tokenizer-path", default="", help="Optional HF tokenizer directory or repo ID.")
  parser.add_argument(
      "--golden_version",
      "--golden-version",
      dest="golden_version",
      choices=GOLDEN_VERSIONS,
      default="mm_v3",
      help="Golden suite version ('mm_v3' Option C default, 'mm_v2' Option B-1, or 'mm_v1' legacy).",
  )
  parser.add_argument(
      "--release-images-dir",
      "--release_images_dir",
      dest="release_images_dir",
      default="",
      help="Optional directory overriding <checkpoint-dir>/inference/examples/images in --mini tests.",
  )
  parser.add_argument(
      "--lazy-dequant",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Keep FP8/FP4 weights compressed in resident buffers and dequantize on the fly (default: True).",
  )
  parser.add_argument(
      "--samples",
      default="all",
      help="Comma-separated sample IDs from L1..L4,M1..M8 or 'all' (default: all).",
  )
  parser.add_argument("--num-threads", type=int, default=0, help="PyTorch intra-op CPU thread count (0 = default).")
  parser.add_argument(
      "--perturbation-threads",
      "--perturbation_threads",
      dest="perturbation_threads",
      type=int,
      default=0,
      help="PyTorch intra-op CPU thread count for mm_v2 C2a perturbation pass (0 = 64 full / 1 mini).",
  )
  parser.add_argument(
      "--diagnostics",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Run bf16-compute and 1-ULP routing score diagnostics on L4 and M8 (default: True).",
  )
  parser.add_argument(
      "--u2-commit-sha",
      default="",
      help="Optional explicit 40-char U2 git SHA when running outside a git working tree.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint."""
  effective_argv = list(sys.argv[1:] if argv is None else argv)
  parsed = parse_args(effective_argv)
  # Strip filesystem-specific paths from recorded argv so manifest.json remains portable/Leakr-clean.
  sanitized_argv: list[str] = []
  skip_next = False
  path_flags = {
      "--checkpoint-dir",
      "--output-dir",
      "--mini-ckpt-dir",
      "--tokenizer-path",
      "--release-images-dir",
      "--release_images_dir",
  }
  for token in effective_argv:
    if skip_next:
      sanitized_argv.append("<redacted-path>")
      skip_next = False
      continue
    if token in path_flags:
      sanitized_argv.append(token)
      skip_next = True
    elif any(token.startswith(f"{flag}=") for flag in path_flags):
      flag_name = token.split("=", 1)[0]
      sanitized_argv.append(f"{flag_name}=<redacted-path>")
    else:
      sanitized_argv.append(token)

  run_generation(
      output_dir=parsed.output_dir,
      checkpoint_dir=parsed.checkpoint_dir,
      mini=parsed.mini,
      mini_ckpt_dir=parsed.mini_ckpt_dir,
      tokenizer_path=parsed.tokenizer_path,
      lazy_dequant=parsed.lazy_dequant,
      samples=parsed.samples,
      num_threads=parsed.num_threads,
      perturbation_threads=parsed.perturbation_threads,
      golden_version=parsed.golden_version,
      release_images_dir=parsed.release_images_dir,
      diagnostics=parsed.diagnostics,
      u2_commit_sha=parsed.u2_commit_sha,
      cli_argv=sanitized_argv,
  )


if __name__ == "__main__":
  main()
