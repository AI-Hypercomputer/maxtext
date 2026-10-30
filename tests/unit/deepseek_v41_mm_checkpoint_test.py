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

"""T11 and T12: DeepSeek-V4.1 multimodal streaming checkpoint round-trip and HF key accounting."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import shutil
import sys
import tempfile
import types
from typing import Any
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.flax import save_file as save_flax_file

from maxtext.checkpoint_conversion import to_huggingface
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_streaming_load
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import utils as ckpt_utils
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import deepseek41
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.parity import ckpt_roundtrip_check
from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.parity import negctl

torch: Any
ref_text_model: Any
multimodal: Any
try:
  import torch  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import model as ref_text_model  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  ref_text_model = None
  multimodal = None
  HAS_TORCH = False


def _saved_tensor_to_jax(tensor: torch.Tensor) -> tuple[jax.Array, jnp.dtype]:
  """Converts a saved CPU SafeTensors tensor to `(jax_array, expected_jax_dtype)`."""
  t = tensor.detach().cpu()
  if t.dtype == torch.bfloat16:
    return jnp.asarray(t.to(torch.float32).numpy(), dtype=jnp.bfloat16), jnp.bfloat16
  if t.dtype == torch.float32:
    return jnp.asarray(t.numpy(), dtype=jnp.float32), jnp.float32
  raise TypeError(f"Unsupported checkpoint tensor dtype in test: {t.dtype}")


def _unstack_decoder_layer_params(model: Any, config: Any) -> dict[str, jax.Array]:
  """Extracts all `nnx.Param` leaves, unstacking `decoder/segments_*` to `decoder/layers_{i}/*`."""
  pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
  flat: dict[str, jax.Array] = {}

  def _walk(tree: Mapping[str, Any], prefix: str = "") -> None:
    for k, v in tree.items():
      path = f"{prefix}/{k}" if prefix else str(k)
      if isinstance(v, Mapping):
        _walk(v, path)
      else:
        flat[path] = v

  _walk(pure)
  if not config.scan_layers:
    return flat

  out: dict[str, jax.Array] = {}
  segments = deepseek41.build_segments(config)
  scan_axis = config.param_scan_axis
  for path, arr in flat.items():
    if not path.startswith("decoder/segments_"):
      out[path] = arr
      continue
    parts = path.split("/")
    seg_idx = int(parts[1].removeprefix("segments_"))
    slot_idx = int(parts[2].removeprefix("layers_"))
    subpath = "/".join(parts[3:])
    groups = segments[seg_idx].groups
    moved = jnp.moveaxis(arr, scan_axis, 0) if arr.ndim > scan_axis else arr
    if moved.shape[0] != len(groups):
      raise ValueError(f"Scan repeat mismatch at {path}: {moved.shape} vs {len(groups)}")
    for rep_idx, group in enumerate(groups):
      out[f"decoder/layers_{group[slot_idx]}/{subpath}"] = moved[rep_idx]
  return out


def _extract_moe_biases_by_layer(model: Any, config: Any, attr_name: str) -> list[jax.Array]:
  """Extracts per-layer `MoEBiasVar` (`bias` or `bias_vl`) across unscanned or scanned layouts."""
  if not config.scan_layers:
    return [
        getattr(getattr(model.decoder, f"layers_{i}").mlp.MoeBlock_0.gate, attr_name)[...]
        for i in range(config.num_decoder_layers)
    ]

  by_layer: dict[int, jax.Array] = {}
  for k, seg in enumerate(deepseek41.build_segments(config)):
    seg_mod = getattr(model.decoder, f"segments_{k}")
    for j in range(seg.block_size):
      stacked = getattr(getattr(seg_mod, f"layers_{j}").mlp.MoeBlock_0.gate, attr_name)[...]
      if stacked.shape != (seg.repeats, config.num_experts):
        raise ValueError(f"Expected scanned {attr_name} shape {(seg.repeats, config.num_experts)}, got {stacked.shape}")
      for r, group in enumerate(seg.groups):
        by_layer[group[j]] = stacked[r]
  return [by_layer[i] for i in range(config.num_decoder_layers)]


def _verify_t11_round_trip(
    model: Any,
    config: Any,
    saved_tensors: Mapping[str, torch.Tensor],
) -> tuple[list[str], int, bool]:
  """Checks bitwise equality and exact dtypes on all multimodal leaves + representative text leaves.

  Returns `(failures, checked_mm_leaf_count, all_finite)`.
  """
  failures: list[str] = []
  unstacked_params = _unstack_decoder_layer_params(model, config)
  bias_vl_by_layer = _extract_moe_biases_by_layer(model, config, "bias_vl")
  bias_by_layer = _extract_moe_biases_by_layer(model, config, "bias")

  all_finite = True
  for arr in list(unstacked_params.values()) + bias_vl_by_layer + bias_by_layer:
    arr64 = np.asarray(arr.astype(jnp.float32), dtype=np.float64)
    if not np.all(np.isfinite(arr64)):
      all_finite = False

  def _check_leaf(cand: jax.Array, hf_key: str, *, transpose: bool = False, label: str = "") -> None:
    expected, exp_dtype = _saved_tensor_to_jax(saved_tensors[hf_key])
    if transpose:
      expected = expected.T
    tag = label if label else hf_key
    if cand.dtype != exp_dtype:
      failures.append(f"{tag}: dtype {cand.dtype} != expected {exp_dtype}")
      return
    if cand.shape != expected.shape:
      failures.append(f"{tag}: shape {cand.shape} != expected {expected.shape}")
      return
    if not bool(jnp.array_equal(cand, expected)):
      max_diff = float(jnp.max(jnp.abs(cand.astype(jnp.float32) - expected.astype(jnp.float32))))
      failures.append(f"{tag}: bitwise mismatch (max_abs={max_diff:.3e})")

  # 1. Every vision_encoder nnx.Param leaf (19 ViT + 7 Projector = 26 leaves in mini-twin).
  mm_specs: list[tuple[str, str, bool]] = [
      ("vision_encoder/DeepSeek41VisionEncoder_0/patch_embed/kernel", "vision.patch_embed.proj.weight", True),
      ("vision_encoder/DeepSeek41VisionEncoder_0/patch_embed/bias", "vision.patch_embed.proj.bias", False),
      ("vision_encoder/DeepSeek41VisionEncoder_0/norm/scale", "vision.norm.weight", False),
      ("vision_encoder/DeepSeek41VisionProjector_0/w1/kernel", "aligner.w1.weight", True),
      ("vision_encoder/DeepSeek41VisionProjector_0/w1/bias", "aligner.w1.bias", False),
      ("vision_encoder/DeepSeek41VisionProjector_0/w2/kernel", "aligner.w2.weight", True),
      ("vision_encoder/DeepSeek41VisionProjector_0/w2/bias", "aligner.w2.bias", False),
      ("vision_encoder/DeepSeek41VisionProjector_0/image_start", "image_start", False),
      ("vision_encoder/DeepSeek41VisionProjector_0/image_end", "image_end", False),
      ("vision_encoder/DeepSeek41VisionProjector_0/image_newline", "image_newline", False),
  ]
  for blk_id in range(config.num_hidden_layers_for_vit):
    c_blk = f"vision_encoder/DeepSeek41VisionEncoder_0/blocks_{blk_id}"
    h_blk = f"vision.blocks.{blk_id}"
    mm_specs.extend(
        [
            (f"{c_blk}/norm1/scale", f"{h_blk}.norm1.weight", False),
            (f"{c_blk}/norm2/scale", f"{h_blk}.norm2.weight", False),
            (f"{c_blk}/attn/wqkv/kernel", f"{h_blk}.attn.wqkv.weight", True),
            (f"{c_blk}/attn/wqkv/bias", f"{h_blk}.attn.wqkv.bias", False),
            (f"{c_blk}/attn/wo/kernel", f"{h_blk}.attn.wo.weight", True),
            (f"{c_blk}/attn/wo/bias", f"{h_blk}.attn.wo.bias", False),
            (f"{c_blk}/mlp/w1/kernel", f"{h_blk}.mlp.w1.weight", True),
            (f"{c_blk}/mlp/w2/kernel", f"{h_blk}.mlp.w2.weight", True),
        ]
    )

  actual_vision_paths = {p for p in unstacked_params if p.startswith("vision_encoder/")}
  declared_vision_paths = {p for p, _, _ in mm_specs}
  if actual_vision_paths != declared_vision_paths:
    failures.append(
        f"Vision leaf set mismatch: missing={sorted(actual_vision_paths - declared_vision_paths)}, "
        f"extra={sorted(declared_vision_paths - actual_vision_paths)}"
    )

  for cand_path, hf_key, transpose in mm_specs:
    cand_arr = unstacked_params[cand_path]
    if cand_arr.dtype != jnp.bfloat16:
      failures.append(f"{cand_path}: expected bfloat16, got {cand_arr.dtype}")
    _check_leaf(cand_arr, hf_key, transpose=transpose, label=cand_path)

  # 2. Every decoder layer's MoEBiasVar bias_vl (14 layers, float32).
  for layer_id, cand_bvl in enumerate(bias_vl_by_layer):
    hf_key = f"layers.{layer_id}.ffn.gate.bias_vl"
    if cand_bvl.dtype != jnp.float32:
      failures.append(f"layer {layer_id} bias_vl: expected float32, got {cand_bvl.dtype}")
    _check_leaf(cand_bvl, hf_key, transpose=False, label=f"layers_{layer_id}/bias_vl")

  # 3. Representative text backbone leaves across top-level and all 14 layers.
  _check_leaf(unstacked_params["token_embedder/embedding"], "embed.weight", transpose=False)
  _check_leaf(unstacked_params["decoder/decoder_norm/scale"], "norm.weight", transpose=False)
  _check_leaf(unstacked_params["decoder/logits_dense/kernel"], "head.weight", transpose=True)
  for layer_id in range(config.num_decoder_layers):
    lpfx = f"decoder/layers_{layer_id}"
    _check_leaf(
        unstacked_params[f"{lpfx}/self_attention/wq_a/kernel"],
        f"layers.{layer_id}.attn.wq_a.weight",
        transpose=True,
    )
    _check_leaf(unstacked_params[f"{lpfx}/hc_attn_fn"], f"layers.{layer_id}.hc_attn_fn", transpose=True)
    _check_leaf(unstacked_params[f"{lpfx}/self_attention/sinks"], f"layers.{layer_id}.attn.attn_sink", transpose=False)
    _check_leaf(bias_by_layer[layer_id], f"layers.{layer_id}.ffn.gate.bias", transpose=False)

    exp_w1 = jnp.stack(
        [
            _saved_tensor_to_jax(saved_tensors[f"layers.{layer_id}.ffn.experts.{e}.w1.weight"])[0].T
            for e in range(config.num_experts)
        ],
        axis=0,
    )
    cand_wi0 = unstacked_params[f"{lpfx}/mlp/MoeBlock_0/wi_0"]
    if cand_wi0.dtype != jnp.bfloat16 or not bool(jnp.array_equal(cand_wi0, exp_w1)):
      failures.append(f"layers_{layer_id}/wi_0: dtype/value mismatch")

    if layer_id in config.ds41_engram_layers:
      eng_ref, eng_dtype = _saved_tensor_to_jax(saved_tensors[f"layers.{layer_id}.engram.embed.weight"])
      cand_eng = unstacked_params[f"{lpfx}/engram/embed"]
      n_rows = eng_ref.shape[0]
      if cand_eng.dtype != eng_dtype or cand_eng.shape[0] <= n_rows:
        failures.append(f"layers_{layer_id}/engram/embed: bad dtype/shape {cand_eng.dtype} {cand_eng.shape}")
      else:
        if not bool(jnp.array_equal(cand_eng[:n_rows], eng_ref)):
          failures.append(f"layers_{layer_id}/engram/embed[:{n_rows}] mismatch")
        if not bool(jnp.all(cand_eng[n_rows:] == 0)):
          failures.append(f"layers_{layer_id}/engram/embed[{n_rows}:] non-zero padding")

  checked_mm_count = len(mm_specs) + len(bias_vl_by_layer)
  return failures, checked_mm_count, all_finite


def _load_safetensors_dir(ckpt_dir: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
  """Loads all `.safetensors` shards in `ckpt_dir` as `(float32_numpy_tensors, exact_dtype_names)`."""
  tensors: dict[str, np.ndarray] = {}
  dtypes: dict[str, str] = {}
  shard_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
  if not shard_files:
    raise FileNotFoundError(f"No .safetensors files found in {ckpt_dir}")
  for fname in shard_files:
    fpath = os.path.join(ckpt_dir, fname)
    with safe_open(fpath, framework="flax", device="cpu") as f:
      for key in f.keys():
        arr = f.get_tensor(key)
        dtypes[key] = str(arr.dtype)
        if arr.dtype == jnp.bfloat16:
          tensors[key] = np.asarray(arr.astype(jnp.float32), dtype=np.float32)
        else:
          tensors[key] = np.asarray(arr)
  return tensors, dtypes


def _stage_offline_mini_hf_dir(base_dir: str) -> tuple[str, int, int]:
  """Prepares an offline mini multimodal HF checkpoint dir with BF16 weights, F32 routing/mHC/sinks, and tokenizer."""
  hf_dir = os.path.join(base_dir, "roundtrip_hf_src")
  if os.path.exists(hf_dir):
    shutil.rmtree(hf_dir)
  os.makedirs(hf_dir, exist_ok=True)
  _, np_tensors = mm_testing.write_numpy_mini_mm_checkpoint(hf_dir, seed=17, use_multimodal=True)
  flax_tensors: dict[str, jax.Array] = {}
  for key, arr in np_tensors.items():
    target_dtype = jnp.float32 if key.endswith(mm_testing.F32_CHECKPOINT_SUFFIXES) else jnp.bfloat16
    flax_tensors[key] = jnp.asarray(arr, dtype=target_dtype)
  save_flax_file(flax_tensors, os.path.join(hf_dir, "model-00001-of-00001.safetensors"))
  for fn in os.listdir(mm_testing.OFFLINE_TOKENIZER_PATH):
    src_f = os.path.join(mm_testing.OFFLINE_TOKENIZER_PATH, fn)
    if os.path.isfile(src_f):
      shutil.copy2(src_f, os.path.join(hf_dir, fn))
  return hf_dir, mm_testing.MINI_VOCAB_SIZE, mm_testing.MINI_IMAGE_TOKEN_ID


def _mini_conversion_cli_args(
    *,
    run_name: str,
    base_output_directory: str,
    hf_model_path: str,
    scan_layers: bool,
    use_multimodal: bool,
    vocab_size: int,
    image_token_id: int,
    load_parameters_path: str = "",
    for_to_huggingface: bool = False,
) -> list[str]:
  """Builds CLI arguments for `to_maxtext.main` or `to_huggingface.main` on the 14-layer mini model."""
  args = [
      sys.argv[0],
      "src/maxtext/configs/base.yml",
      "model_name=deepseek4.1-flash-mini",
      f"run_name={run_name}",
      f"base_output_directory={base_output_directory}",
      f"hf_model_path={hf_model_path}",
      "override_model_config=True",
      f"use_multimodal={use_multimodal}",
      "vision_encoder_block=deepseek41",
      "freeze_vision_encoder_params=False",
      f"scan_layers={scan_layers}",
      "hardware=cpu",
      "skip_jax_distributed_system=True",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
      "dtype_mm=bfloat16",
      "base_emb_dim=64",
      "base_num_query_heads=4",
      "head_dim=32",
      "qk_rope_head_dim=8",
      "q_lora_rank=16",
      "o_groups=2",
      "o_lora_rank=8",
      "base_mlp_dim=32",
      "base_moe_mlp_dim=32",
      "num_experts=4",
      "num_experts_per_tok=2",
      "sparse_matmul=False",
      "capacity_factor=-1",
      "sliding_window_size=8",
      f"base_num_decoder_layers={len(mm_testing.MINI_COMPRESS_RATIOS)}",
      f"compress_ratios={list(mm_testing.MINI_COMPRESS_RATIOS)}",
      f"kv_source_layers={list(mm_testing.MINI_KV_SOURCE_LAYERS)}",
      f"index_source_layers={list(mm_testing.MINI_INDEX_SOURCE_LAYERS)}",
      "indexer_n_heads=16",
      "indexer_head_dim=16",
      "indexer_topk=3",
      f"candidate_source_layer={mm_testing.MINI_CANDIDATE_SOURCE_LAYER}",
      "candidate_topk_blocks=2",
      "candidate_block_size=4",
      f"ds41_engram_layers={list(mm_testing.MINI_ENGRAM_LAYER_IDS)}",
      "ds41_engram_vocab_size=50",
      "ds41_engram_max_ngram_size=3",
      "ds41_engram_n_heads=2",
      "ds41_engram_head_dim=8",
      "ds41_engram_compressed_vocab_size=0",
      f"vocab_size={vocab_size}",
      f"ds41_image_token_id={image_token_id}",
      "num_hidden_layers_for_vit=2",
      "hidden_size_for_vit=128",
      "num_attention_heads_for_vit=2",
      "intermediate_size_for_vit=352",
      "patch_size_for_vit=14",
      "rope_theta_for_vit=10000.0",
      "num_channels_for_vit=3",
      "ds41_vision_downsample_ratio=3",
      f"ds41_vision_grid={list(mm_testing.MINI_GRID_PADDED)}",
      f"tokenizer_path={mm_testing.OFFLINE_TOKENIZER_PATH}",
  ]
  if for_to_huggingface:
    args.extend(
        [
            f"load_parameters_path={load_parameters_path}",
            "override_model_architecture=True",
        ]
    )
  else:
    args.extend(
        [
            "lazy_load_tensors=False",
            "save_dtype=bfloat16",
            "simulated_cpu_devices_count=1",
        ]
    )
  return args


@pytest.mark.cpu_only
class DeepSeek41MultimodalCheckpointTest(unittest.TestCase):
  """T11 and T12: streaming SafeTensors round-trip, dtype preservation, text-only guard, and key accounting."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.base_dir = tempfile.mkdtemp(prefix="ds41_mm_u5_ckpt_")
    if not HAS_TORCH:
      return
    args = multimodal.mini_model_args()
    cls.ref_mm = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
    multimodal.init_mini_reference(cls.ref_mm, seed=17)
    cls.ckpt_dir = os.path.join(cls.base_dir, "mini_mm_bf16")
    cls.saved_tensors = mm_testing.write_mini_safetensors_checkpoint(
        cls.ckpt_dir,
        cls.ref_mm,
        param_dtype=torch.bfloat16,
        include_mtp=False,
    )
    cls.ckpt_mtp_dir = os.path.join(cls.base_dir, "mini_mm_bf16_with_mtp")
    cls.saved_mtp_tensors = mm_testing.write_mini_safetensors_checkpoint(
        cls.ckpt_mtp_dir,
        cls.ref_mm,
        param_dtype=torch.bfloat16,
        include_mtp=True,
    )

  @classmethod
  def tearDownClass(cls) -> None:
    shutil.rmtree(cls.base_dir, ignore_errors=True)

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_t11_streaming_round_trip_bitwise_and_dtypes(self) -> None:
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        model, cfg, _ = mm_testing.load_mini_maxtext_from_safetensors(
            self.ckpt_dir,
            scan_layers=scan_layers,
            dtype="bfloat16",
            weight_dtype="bfloat16",
            dtype_mm="bfloat16",
        )
        failures, checked_mm_count, all_finite = _verify_t11_round_trip(model, cfg, self.saved_tensors)
        self.assertTrue(all_finite)
        self.assertEqual(checked_mm_count, 40)
        self.assertEqual(failures, [], msg=f"T11 failures (scan_layers={scan_layers}): {failures}")

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_t11_negative_controls_corrupted_safetensors_leaf(self) -> None:
    for target_hf_key, seed in (("aligner.w1.weight", 71), ("layers.2.ffn.gate.bias_vl", 72)):
      orig_jax, _ = _saved_tensor_to_jax(self.saved_tensors[target_hf_key])
      pert_tree = negctl.perturb_one_leaf(
          {"leaf": orig_jax},
          target_path="leaf",
          rel_scale=1e-2,
          seed=seed,
      )
      pert_jax = pert_tree["leaf"]
      if pert_jax.dtype == jnp.bfloat16:
        pert_torch = torch.from_numpy(np.asarray(pert_jax.astype(jnp.float32))).to(torch.bfloat16).contiguous()
      else:
        pert_torch = torch.from_numpy(np.asarray(pert_jax, dtype=np.float32)).contiguous()

      corrupt_dir = os.path.join(self.base_dir, f"corrupt_{target_hf_key.replace('.', '_')}")
      os.makedirs(corrupt_dir, exist_ok=True)
      corrupt_state = dict(self.saved_tensors)
      corrupt_state[target_hf_key] = pert_torch
      from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

      shard_name = "model-00001-of-00001.safetensors"
      save_file(corrupt_state, os.path.join(corrupt_dir, shard_name), metadata={"format": "pt"})
      with open(os.path.join(corrupt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "deepseek_v41"}, f)

      for scan_layers in (False, True):
        with self.subTest(target_hf_key=target_hf_key, scan_layers=scan_layers):
          model, cfg, _ = mm_testing.load_mini_maxtext_from_safetensors(
              corrupt_dir,
              scan_layers=scan_layers,
              dtype="bfloat16",
              weight_dtype="bfloat16",
              dtype_mm="bfloat16",
          )
          failures, _, all_finite = _verify_t11_round_trip(model, cfg, self.saved_tensors)
          self.assertTrue(all_finite, msg="Corrupted checkpoint produced non-finite loaded weights")
          self.assertGreater(
              len(failures),
              0,
              msg=f"T11 negative control MISSED on {target_hf_key} (scan_layers={scan_layers})",
          )

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_t11_text_only_mapping_and_load_unchanged(self) -> None:
    text_args = multimodal.mini_model_args(vision_n_layers=0)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
      ref_text = ref_text_model.Transformer(text_args, tokenizer=mm_testing.get_reference_tokenizer()).eval()
    finally:
      torch.set_default_dtype(prev_dtype)
    multimodal.init_mini_reference(ref_text, seed=29)
    text_ckpt_dir = os.path.join(self.base_dir, "mini_text_only_bf16")
    saved_text = mm_testing.write_mini_safetensors_checkpoint(
        text_ckpt_dir,
        ref_text,
        param_dtype=torch.bfloat16,
        include_mtp=False,
    )
    self.assertFalse(any(k.startswith(("vision.", "aligner.", "image_")) or "bias_vl" in k for k in saved_text))

    for scan_layers in (False, True):
      cfg_text = mm_testing.mini_vision_maxtext_config(
          use_multimodal=False,
          scan_layers=scan_layers,
          dtype="bfloat16",
          weight_dtype="bfloat16",
          load_parameters_path=text_ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
      )
      cfg_mm = mm_testing.mini_vision_maxtext_config(
          use_multimodal=True,
          scan_layers=scan_layers,
          dtype="bfloat16",
          weight_dtype="bfloat16",
      )
      for quantized in (False, True):
        hf_cfg = {"quantization_config": {"quant_method": "fp8"}} if quantized else {}
        text_map = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, cfg_text, scan_layers=scan_layers)
        mm_map = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, cfg_mm, scan_layers=scan_layers)
        self.assertFalse(any("vision_encoder" in k or "bias_vl" in k for k in text_map))
        mm_text_subset = {k: v for k, v in mm_map.items() if "vision_encoder" not in k and "bias_vl" not in k}
        self.assertEqual(text_map, mm_text_subset)

        text_hooks = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, cfg_text, scan_layers=scan_layers)
        mm_hooks = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, cfg_mm, scan_layers=scan_layers)
        self.assertFalse(any("vision_encoder" in k or "bias_vl" in k for k in text_hooks))
        self.assertEqual(set(text_hooks), {k for k in mm_hooks if "vision_encoder" not in k and "bias_vl" not in k})

      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_text), cfg_text.mesh_axes)
      with mesh:
        mt_text = model_creation_utils.from_pretrained(cfg_text, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      self.assertIsNone(mt_text.vision_encoder)
      exp_emb, _ = _saved_tensor_to_jax(saved_text["embed.weight"])
      self.assertTrue(bool(jnp.array_equal(mt_text.token_embedder.embedding[...], exp_emb)))

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_t12_unloaded_hf_keys_equal_mtp_only_and_negctl(self) -> None:
    release_hf_keys = mm_testing.build_release_40l_hf_keys(include_mtp=True)
    expected_release_mtp = {k for k in release_hf_keys if k.startswith("mtp.")}
    self.assertEqual(len(release_hf_keys), 96085)
    self.assertEqual(len(expected_release_mtp), 2401)

    mini_hf_keys = set(self.saved_mtp_tensors.keys())
    expected_mini_mtp = {k for k in mini_hf_keys if k.startswith("mtp.")}
    self.assertGreater(len(expected_mini_mtp), 0)

    index_hf_keys: set[str] | None = None
    if "DS41_HF_INDEX_JSON" in os.environ:
      index_path = os.environ["DS41_HF_INDEX_JSON"]
      with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)
      index_hf_keys = set(index_data["weight_map"].keys())
      self.assertEqual(index_hf_keys, release_hf_keys)

    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        # (a) 40-layer quantized release key accounting.
        cfg_40l = pyconfig.initialize(
            [sys.argv[0], "src/maxtext/configs/base.yml"],
            model_name="deepseek4.1-flash",
            run_name="ds41_t12_40l",
            base_output_directory=os.path.join(self.base_dir, "out_40l"),
            enable_checkpointing=False,
            skip_jax_distributed_system=True,
            use_multimodal=True,
            scan_layers=scan_layers,
        )
        map_40l = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(
            {"quantization_config": {"quant_method": "fp8"}},
            cfg_40l,
            scan_layers=scan_layers,
        )
        mapped_40l = mm_testing.flatten_mapped_hf_keys(map_40l)
        self.assertEqual(mapped_40l - release_hf_keys, set())
        self.assertEqual(release_hf_keys - mapped_40l, expected_release_mtp)

        if index_hf_keys is not None:
          self.assertEqual(mapped_40l - index_hf_keys, set())
          self.assertEqual(index_hf_keys - mapped_40l, {k for k in index_hf_keys if k.startswith("mtp.")})

        # (b) 14-layer mini-twin key accounting + live LoadPlan unmatched_mt_keys == [].
        model_mini, cfg_mini, _ = mm_testing.load_mini_maxtext_from_safetensors(
            self.ckpt_mtp_dir,
            scan_layers=scan_layers,
            dtype="bfloat16",
            weight_dtype="bfloat16",
            dtype_mm="bfloat16",
        )
        map_mini = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING({}, cfg_mini, scan_layers=scan_layers)
        hooks_mini = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_HOOK_FN({}, cfg_mini, scan_layers=scan_layers)
        mapped_mini = mm_testing.flatten_mapped_hf_keys(map_mini)
        self.assertEqual(mapped_mini - mini_hf_keys, set())
        self.assertEqual(mini_hf_keys - mapped_mini, expected_mini_mtp)

        param_state = nnx.state(model_mini).filter(
            lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
        )
        abstract_target = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
            param_state.to_pure_dict(),
        )
        plan = hf_streaming_load.build_plan(map_mini, hooks_mini, abstract_target, cfg_mini)
        self.assertEqual(plan.unmatched_mt_keys, [])
        plan_needed = {k for grp in plan.groups for w in grp for k in w.hf_keys}
        self.assertEqual(plan_needed, mapped_mini)
        self.assertEqual(mini_hf_keys - plan_needed, expected_mini_mtp)

        # (c) Negative controls: dropping a bias_vl entry or a vision.* entry surfaces non-mtp.* unloaded keys.
        bias_vl_mt_keys = [k for k in map_40l if k.endswith("-bias_vl")]
        self.assertGreater(len(bias_vl_mt_keys), 0)
        map_missing_bvl = {k: v for k, v in map_40l.items() if k != bias_vl_mt_keys[0]}
        unloaded_bvl = release_hf_keys - mm_testing.flatten_mapped_hf_keys(map_missing_bvl)
        self.assertNotEqual(unloaded_bvl, expected_release_mtp)
        self.assertTrue(any(k.endswith(".ffn.gate.bias_vl") and not k.startswith("mtp.") for k in unloaded_bvl))

        vis_mt_key = "params-vision_encoder-DeepSeek41VisionEncoder_0-patch_embed-kernel"
        map_missing_vis = {k: v for k, v in map_40l.items() if k != vis_mt_key}
        unloaded_vis = release_hf_keys - mm_testing.flatten_mapped_hf_keys(map_missing_vis)
        self.assertNotEqual(unloaded_vis, expected_release_mtp)
        self.assertIn("vision.patch_embed.proj.weight", unloaded_vis - expected_release_mtp)

  def test_t12_release_40l_key_accounting_and_numpy_safetensors_without_torch(self) -> None:
    release_hf_keys = mm_testing.build_release_40l_hf_keys(include_mtp=True)
    expected_release_mtp = {k for k in release_hf_keys if k.startswith("mtp.")}
    self.assertEqual(len(release_hf_keys), 96085)
    self.assertEqual(len(expected_release_mtp), 2401)

    np_ckpt_dir = os.path.join(self.base_dir, "np_mini_mm_f32")
    _, hf_tensors = mm_testing.write_numpy_mini_mm_checkpoint(np_ckpt_dir, seed=13, use_multimodal=True)
    for scan_layers in (False, True):
      cfg_40l = pyconfig.initialize(
          [sys.argv[0], "src/maxtext/configs/base.yml"],
          model_name="deepseek4.1-flash",
          run_name="ds41_t12_40l_notorch",
          base_output_directory=os.path.join(self.base_dir, "out_40l_notorch"),
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          use_multimodal=True,
          scan_layers=scan_layers,
      )
      map_40l = param_mapping.DEEPSEEK_V41_MAXTEXT_TO_HF_PARAM_MAPPING(
          {"quantization_config": {"quant_method": "fp8"}},
          cfg_40l,
          scan_layers=scan_layers,
      )
      mapped_40l = mm_testing.flatten_mapped_hf_keys(map_40l)
      self.assertEqual(mapped_40l - release_hf_keys, set())
      self.assertEqual(release_hf_keys - mapped_40l, expected_release_mtp)

      model_np, cfg_np, _ = mm_testing.load_mini_maxtext_from_safetensors(
          np_ckpt_dir,
          scan_layers=scan_layers,
          vocab_size=mm_testing.MINI_VOCAB_SIZE,
          ds41_image_token_id=mm_testing.MINI_IMAGE_TOKEN_ID,
          ds41_engram_compressed_vocab_size=0,
          tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
      )
      enc = model_np.vision_encoder.DeepSeek41VisionEncoder_0
      proj = model_np.vision_encoder.DeepSeek41VisionProjector_0
      np.testing.assert_array_equal(
          np.asarray(enc.patch_embed.kernel[...]),
          hf_tensors["vision.patch_embed.proj.weight"].T,
      )
      np.testing.assert_array_equal(
          np.asarray(proj.w1.kernel[...]),
          hf_tensors["aligner.w1.weight"].T,
      )
      np.testing.assert_array_equal(np.asarray(proj.image_start[...]), hf_tensors["image_start"])
      bvl_by_layer = _extract_moe_biases_by_layer(model_np, cfg_np, "bias_vl")
      for lid in range(cfg_np.num_decoder_layers):
        np.testing.assert_array_equal(np.asarray(bvl_by_layer[lid]), hf_tensors[f"layers.{lid}.ffn.gate.bias_vl"])

  def test_t11_hf_to_maxtext_to_hf_roundtrip_and_negctl(self) -> None:
    hf_src_dir, vocab_size, image_token_id = _stage_offline_mini_hf_dir(self.base_dir)
    source_tensors, source_dtypes = _load_safetensors_dir(hf_src_dir)
    self.assertEqual(len(source_tensors), 544)

    mm_only_keys = {
        k for k in source_tensors if k.startswith(("vision.", "aligner.", "image_")) or k.endswith(".ffn.gate.bias_vl")
    }
    self.assertEqual(len(mm_only_keys), 40)
    for lid in range(len(mm_testing.MINI_COMPRESS_RATIOS)):
      self.assertEqual(source_dtypes[f"layers.{lid}.ffn.gate.bias_vl"], "float32")
      self.assertEqual(source_dtypes[f"layers.{lid}.ffn.gate.bias"], "float32")
    for k in mm_only_keys:
      if not k.endswith(".ffn.gate.bias_vl"):
        self.assertEqual(source_dtypes[k], "bfloat16", msg=f"Expected bfloat16 on {k}")

    tol = ckpt_roundtrip_check.RoundTripTol.exact()
    cached_orbax: dict[bool, Any] = {}
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers), mock.patch("gc.collect"):
        tag = "scanned" if scan_layers else "unscanned"
        mt_out_dir = os.path.join(self.base_dir, f"rt_mt_{tag}")
        hf_out_dir = os.path.join(self.base_dir, f"rt_hf_{tag}")
        to_maxtext.main(
            _mini_conversion_cli_args(
                run_name=f"rt_mt_{tag}",
                base_output_directory=mt_out_dir,
                hf_model_path=hf_src_dir,
                scan_layers=scan_layers,
                use_multimodal=True,
                vocab_size=vocab_size,
                image_token_id=image_token_id,
                for_to_huggingface=False,
            )
        )
        orbax_items_path = os.path.join(mt_out_dir, "0", "items")

        def _memoized_load_orbax(cfg: Any, _scan: bool = scan_layers) -> dict[str, Any]:
          if _scan not in cached_orbax:
            cached_orbax[_scan] = ckpt_utils.load_orbax_checkpoint(cfg)
          return cached_orbax[_scan]

        with mock.patch.object(to_huggingface, "load_orbax_checkpoint", side_effect=_memoized_load_orbax):
          to_huggingface.main(
              _mini_conversion_cli_args(
                  run_name=f"rt_hf_{tag}",
                  base_output_directory=hf_out_dir,
                  hf_model_path=hf_src_dir,
                  scan_layers=scan_layers,
                  use_multimodal=True,
                  vocab_size=vocab_size,
                  image_token_id=image_token_id,
                  load_parameters_path=orbax_items_path,
                  for_to_huggingface=True,
              )
          )

        primary_exported, exported_dtypes = _load_safetensors_dir(hf_out_dir)
        for lid in range(len(mm_testing.MINI_COMPRESS_RATIOS)):
          self.assertEqual(exported_dtypes[f"layers.{lid}.ffn.gate.bias_vl"], "float32")
          self.assertEqual(exported_dtypes[f"layers.{lid}.ffn.gate.bias"], "float32")
        for k in mm_only_keys:
          if not k.endswith(".ffn.gate.bias_vl"):
            self.assertEqual(exported_dtypes[k], "bfloat16", msg=f"Expected bfloat16 on exported {k}")

        ok, metrics, failures = ckpt_roundtrip_check.check_hf_roundtrip(
            source_tensors,
            primary_exported,
            tol=tol,
            source_dtypes=source_dtypes,
            exported_dtypes=exported_dtypes,
        )
        self.assertTrue(ok, msg=f"Round-trip failed (scan_layers={scan_layers}): {failures}")
        self.assertEqual(failures, [])
        self.assertEqual(len(metrics), len(source_tensors))

        # Negative controls: perturb a BF16 vision leaf and an FP32 bias_vl leaf in MaxText state before export.
        bias_vl_mt_key = (
            "MoEBiasVar-decoder-segments_1-layers_0-mlp-MoeBlock_0-gate-bias_vl"
            if scan_layers
            else "MoEBiasVar-decoder-layers_2-mlp-MoeBlock_0-gate-bias_vl"
        )
        neg_targets = (
            ("params-vision_encoder-DeepSeek41VisionProjector_0-w1-kernel", 81),
            (bias_vl_mt_key, 82),
        )
        for target_mt_key, seed in neg_targets:
          with self.subTest(scan_layers=scan_layers, neg_target=target_mt_key):
            hf_pert_dir = os.path.join(self.base_dir, f"rt_hf_pert_{tag}_{seed}")

            def _extract_perturbed_state(
                ckpt_dict: dict[str, Any], _key: str = target_mt_key, _seed: int = seed
            ) -> dict[str, Any]:
              state_dict = dict(ckpt_utils.detect_and_extract_checkpoint(ckpt_dict))
              orig_leaf = state_dict[_key]
              pert_np = ckpt_roundtrip_check.perturb_one_tensor(
                  {_key: np.asarray(orig_leaf)},
                  target_key=_key,
                  rel_scale=1e-2,
                  seed=_seed,
              )[_key]
              state_dict[_key] = jnp.asarray(pert_np, dtype=orig_leaf.dtype)
              return state_dict

            with (
                mock.patch.object(to_huggingface, "load_orbax_checkpoint", side_effect=_memoized_load_orbax),
                mock.patch.object(to_huggingface, "detect_and_extract_checkpoint", side_effect=_extract_perturbed_state),
            ):
              to_huggingface.main(
                  _mini_conversion_cli_args(
                      run_name=f"rt_hf_pert_{tag}_{seed}",
                      base_output_directory=hf_pert_dir,
                      hf_model_path=hf_src_dir,
                      scan_layers=scan_layers,
                      use_multimodal=True,
                      vocab_size=vocab_size,
                      image_token_id=image_token_id,
                      load_parameters_path=orbax_items_path,
                      for_to_huggingface=True,
                  )
              )
            perturbed_exported, perturbed_dtypes = _load_safetensors_dir(hf_pert_dir)
            status, reasons = ckpt_roundtrip_check.negative_roundtrip_status(
                source_tensors,
                primary_exported,
                perturbed_exported,
                tol=tol,
                source_dtypes=source_dtypes,
                exported_dtypes=perturbed_dtypes,
            )
            self.assertEqual(
                (status, reasons),
                ("FIRED", []),
                msg=f"Negative control did not fire for {target_mt_key} (scan_layers={scan_layers}): {status} {reasons}",
            )

        if not scan_layers:
          hf_text_out_dir = os.path.join(self.base_dir, "rt_hf_text_only_unscanned")
          with mock.patch.object(to_huggingface, "load_orbax_checkpoint", side_effect=_memoized_load_orbax):
            to_huggingface.main(
                _mini_conversion_cli_args(
                    run_name="rt_hf_text_only_unscanned",
                    base_output_directory=hf_text_out_dir,
                    hf_model_path=hf_src_dir,
                    scan_layers=False,
                    use_multimodal=False,
                    vocab_size=vocab_size,
                    image_token_id=image_token_id,
                    load_parameters_path=orbax_items_path,
                    for_to_huggingface=True,
                )
            )
          text_exported, text_dtypes = _load_safetensors_dir(hf_text_out_dir)
          self.assertTrue(mm_only_keys.isdisjoint(text_exported.keys()))
          text_ok, _, text_failures = ckpt_roundtrip_check.check_hf_roundtrip(
              source_tensors,
              text_exported,
              tol=tol,
              unexported_source_keys=mm_only_keys,
              source_dtypes=source_dtypes,
              exported_dtypes=text_dtypes,
          )
          self.assertTrue(text_ok, msg=f"Text-only export failed: {text_failures}")

  def test_t12_non_ds41_conversion_helpers_noop(self):
    """Proves non-deepseek4.1 models preserve legacy numpy and full-precision helper behavior."""
    # pylint: disable=import-outside-toplevel,protected-access
    import ml_dtypes

    arr_f32 = np.array([1.25, -2.5], dtype=np.float32)
    out_legacy = to_maxtext._convert_tensor_to_numpy(arr_f32, save_dtype="bfloat16", is_ds41=False)
    self.assertIs(out_legacy, arr_f32)
    self.assertEqual(out_legacy.dtype, np.float32)

    out_ds41 = to_maxtext._convert_tensor_to_numpy(arr_f32, save_dtype="bfloat16", is_ds41=True)
    self.assertEqual(out_ds41.dtype, ml_dtypes.bfloat16)

    for bf16_dtype in ("bfloat16", jnp.bfloat16, np.dtype("bfloat16"), jnp.dtype("bfloat16")):
      for fn in (to_maxtext._param_save_dtype, ckpt_utils._param_save_dtype):
        self.assertEqual(fn("params-decoder-layers_0-hc_attn_fn", bf16_dtype, is_ds41=False), bf16_dtype)
        self.assertEqual(fn("params-decoder-layers_0-self_attention-sinks", bf16_dtype, is_ds41=False), bf16_dtype)
        self.assertEqual(fn("params-decoder-layers_0-hc_attn_fn", bf16_dtype, is_ds41=True), "float32")
        self.assertEqual(fn("params-decoder-layers_0-self_attention-sinks", bf16_dtype, is_ds41=True), "float32")
        self.assertEqual(
            fn("MoEBiasVar-decoder-layers_0-mlp-MoeBlock_0-gate-bias", bf16_dtype, is_ds41=True),
            "float32",
        )
      self.assertEqual(
          to_maxtext._param_save_dtype("MoEBiasVar-decoder-layers_0-mlp-MoeBlock_0-gate-bias", bf16_dtype, is_ds41=False),
          "float32",
      )
      self.assertEqual(
          ckpt_utils._param_save_dtype("MoEBiasVar-decoder-layers_0-mlp-MoeBlock_0-gate-bias", bf16_dtype, is_ds41=False),
          bf16_dtype,
      )

    # Verify config.json with `"text_config": null` sets hf_config_dict["quantization_config"] = None without TypeError.
    null_text_cfg_dir = os.path.join(self.base_dir, "null_text_cfg")
    os.makedirs(null_text_cfg_dir, exist_ok=True)
    with open(os.path.join(null_text_cfg_dir, "config.json"), "w", encoding="utf-8") as f:
      json.dump({"model_type": "deepseek_v41", "text_config": None}, f)
    captured_hf_cfg: dict[str, Any] = {}

    def _capture_param_map(hf_cfg, _mt_cfg, _scan):
      captured_hf_cfg.update(hf_cfg)
      return {}

    fake_cfg = types.SimpleNamespace(
        model_name="deepseek4.1-flash",
        base_output_directory=os.path.join(self.base_dir, "null_text_out"),
        run_name="null_text_run",
        hf_access_token="",
        hf_lora_adapter_path="",
        load_parameters_path="",
        use_multimodal=False,
        scan_layers=False,
        checkpoint_storage_use_ocdbt=False,
        checkpoint_storage_use_zarr3=False,
    )
    with (
        mock.patch.object(to_maxtext.pyconfig, "initialize", return_value=fake_cfg),
        mock.patch.object(to_maxtext.max_utils, "print_system_information"),
        mock.patch.object(
            to_maxtext,
            "load_hf_dict_from_safetensors",
            return_value={"w": np.zeros((1,), dtype=np.float32)},
        ),
        mock.patch.dict(to_maxtext.PARAM_MAPPING, {"deepseek4.1-flash": _capture_param_map}),
        mock.patch.dict(to_maxtext.HOOK_FNS, {"deepseek4.1-flash": lambda *_a, **_kw: {}}),
        mock.patch.object(to_maxtext, "get_maxtext_model_info", return_value=({}, types.SimpleNamespace(num_leaves=0))),
        mock.patch.object(to_maxtext, "validate_and_filter_param_map_keys", return_value=[]),
        mock.patch.object(to_maxtext.jax.tree_util, "tree_unflatten", return_value={}),
        mock.patch.object(to_maxtext, "save_weights_to_checkpoint"),
    ):
      to_maxtext.main(
          [
              "to_maxtext.py",
              "src/maxtext/configs/base.yml",
              "model_name=deepseek4.1-flash",
              f"--hf_model_path={null_text_cfg_dir}",
              "--lazy_load_tensors=False",
          ]
      )
    self.assertIn("quantization_config", captured_hf_cfg)
    self.assertIsNone(captured_hf_cfg["quantization_config"])


if __name__ == "__main__":
  unittest.main()
