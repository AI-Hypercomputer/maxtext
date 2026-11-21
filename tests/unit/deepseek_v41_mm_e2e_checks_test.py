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

"""CPU unit tests and negative controls for DeepSeek-V4.1-Flash multimodal E2E verification tools."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import jsonlines
import numpy as np
import pytest
import scipy.special
import transformers
from flax import nnx

from tests.utils import (  # isort: skip
    deepseek_v41_frozen_topk,
    deepseek_v41_mm_audit,
    forward_pass_logit_checker,
)
from tests.utils.deepseek_v41 import logit_runner, masked_prefill_runner
from maxtext.common import checkpointing
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.utils import maxtext_utils, model_creation_utils


class _ArrayWrapper(np.ndarray):

  def numpy(self):
    return np.asarray(self)


def _wrap(arr):
  return arr.view(_ArrayWrapper)


class _ImageInput:
  """Minimal image input container for pure JAX test harness."""

  def __init__(self, start, patches, n_vit_h, n_vit_w, img_types):
    self.start = start
    self.patches = patches
    self.n_vit_h = n_vit_h
    self.n_vit_w = n_vit_w
    self.types = img_types


class _MiniVLBatch:

  def __init__(self, input_ids, token_types, patches_bnpd, images):
    self.input_ids = _wrap(input_ids)
    self.token_types = _wrap(token_types)
    self.patches_bnpd = _wrap(patches_bnpd)
    self.images = images


class _MultimodalShim:
  """Pure-JAX compatibility shim for multimodal reference utilities when PyTorch is absent."""

  MINI_GRID_PADDED = (7, 5)
  MINI_SEQ_LEN = 40
  MultimodalTransformer = object

  @staticmethod
  def mini_model_args(**kwargs):
    """Constructs minimal model configuration arguments for testing."""
    d = {
        "dtype": "fp32",
        "max_batch_size": 2,
        "max_seq_len": 64,
        "vocab_size": 129280,
        "dim": 64,
        "moe_inter_dim": 32,
        "n_layers": 14,
        "n_mtp_layers": 0,
        "n_heads": 4,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "image_token_id": 129264,
    }
    d.update(kwargs)
    return types.SimpleNamespace(**d)

  @staticmethod
  def build_mini_vl_batch(
      args=None,
      *,
      grid=(7, 5),
      batch_size=1,
      seq_len=40,
      span_starts=((2,),),
      seed=0,
      **kw,
  ):
    """Builds a synthetic mini VL batch with image tokens and patch embeddings."""
    del kw
    n_vit_h, n_vit_w = grid
    n_llm_h = math.ceil(n_vit_h / 3)
    n_llm_w = math.ceil(n_vit_w / 3)

    types_list = [0]
    for _ in range(n_llm_h):
      types_list.extend([1] * n_llm_w)
      types_list.append(2)
    types_list.append(3)
    span_len = len(types_list)

    vocab_size = getattr(args, "vocab_size", 129280)
    image_token_id = getattr(args, "image_token_id", 129264)

    rng = np.random.default_rng(seed)
    input_ids = rng.integers(3, vocab_size - 1, size=(batch_size, seq_len), dtype=np.int64)
    input_ids = np.where(input_ids >= image_token_id, input_ids + 1, input_ids)
    token_types = np.full((batch_size, seq_len), -1, dtype=np.int64)

    num_images_per_sample = len(span_starts[0])
    num_patches = n_vit_h * n_vit_w
    patches_bnpd = rng.standard_normal((batch_size, num_images_per_sample, num_patches, 588)).astype(np.float32)

    images = []
    for b in range(batch_size):
      sample_imgs = []
      for img_idx, start in enumerate(span_starts[b]):
        end = start + span_len
        input_ids[b, start:end] = image_token_id
        token_types[b, start:end] = types_list
        sample_imgs.append(
            _ImageInput(
                start=start,
                patches=_wrap(patches_bnpd[b, img_idx]),
                n_vit_h=n_vit_h,
                n_vit_w=n_vit_w,
                img_types=_wrap(np.array(types_list, dtype=np.int64)),
            )
        )
      images.append(sample_imgs)

    return _MiniVLBatch(input_ids, token_types, patches_bnpd, images)


try:
  import torch
  from safetensors.torch import save_file

  from tests.utils.deepseek_v41_reference import multimodal

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  save_file = None  # type: ignore[assignment]
  multimodal = _MultimodalShim  # type: ignore[assignment]
  HAS_TORCH = False

from tests.utils import deepseek_v41_mm_testing as mm_testing

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS_JSON_PATH = _REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/thresholds.json"
_AOT_CHECK_PY_PATH = _REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py"
_STAGE_SCRIPT_PATH = _REPO_ROOT / "tests/end_to_end/tpu/deepseek/v4.1-flash/2_test_deepseek41_mm.sh"


def _load_aot_check_module() -> Any:
  """Loads `tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py` as a module."""
  spec = importlib.util.spec_from_file_location("ds41_aot_check", _AOT_CHECK_PY_PATH)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module spec from {_AOT_CHECK_PY_PATH}")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _save_safetensors_dir(
    out_dir: str | os.PathLike[str],
    tensors: Mapping[str, torch.Tensor],
) -> None:
  """Writes a single-shard safetensors directory with `model.safetensors.index.json`."""
  out_path = Path(out_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  shard_name = "model-00001-of-00001.safetensors"
  save_file(dict(tensors), (out_path / shard_name).as_posix(), metadata={"format": "pt"})
  index_payload = {
      "metadata": {"total_size": sum(int(t.numel() * t.element_size()) for t in tensors.values())},
      "weight_map": {k: shard_name for k in sorted(tensors.keys())},
  }
  with open(out_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
    json.dump(index_payload, f, indent=2)
  with open(out_path / "config.json", "w", encoding="utf-8") as f:
    json.dump({"model_type": "deepseek_v41"}, f, indent=2)


def _build_mini_hf_tensors_from_ref(
    ref_mm: multimodal.MultimodalTransformer,
    ref_args: Any,
    *,
    include_mtp: bool = True,
) -> dict[str, torch.Tensor]:
  """Extracts a complete HF safetensors dictionary (`BF16` weights + `F32` `bias`/`bias_vl`) from `ref_mm`."""
  out: dict[str, torch.Tensor] = {}
  for name, tensor in ref_mm.named_parameters():
    t_cpu = tensor.detach().cpu().contiguous()
    if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
      out[name] = t_cpu.to(torch.float32).contiguous()
    else:
      out[name] = t_cpu.to(torch.bfloat16).contiguous()
  for name, tensor in ref_mm.named_buffers():
    if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
      out[name] = tensor.detach().cpu().to(torch.float32).contiguous()
  if include_mtp:
    out["mtp.0.norm.weight"] = torch.ones((ref_args.dim,), dtype=torch.bfloat16)
    out["mtp.0.head.weight"] = torch.zeros((ref_args.vocab_size, ref_args.dim), dtype=torch.bfloat16)
  return out


def _flip_one_ulp_torch(tensor: torch.Tensor) -> torch.Tensor:
  """Flips the lowest bit (1 ULP) of element 0 in a `bfloat16` or `float32` Torch tensor."""
  clone = tensor.detach().cpu().clone().contiguous()
  flat = clone.view(-1)
  if clone.dtype == torch.bfloat16:
    u16 = flat.view(torch.uint16)
    u16[0] = torch.tensor(int(u16[0].item()) ^ 1, dtype=torch.uint16)
    return clone
  if clone.dtype == torch.float32:
    i32 = flat.view(torch.int32)
    i32[0] = torch.tensor(int(i32[0].item()) ^ 1, dtype=torch.int32)
    return clone
  raise TypeError(f"Unsupported tensor dtype for 1-ULP flip: {clone.dtype}")


def _reshard_model_to_abstract_mesh(model: Any, config: Any, mesh: jax.sharding.Mesh) -> None:
  """Re-shards all `nnx.Param` and `moe.MoEBiasVar` leaves onto `create_nnx_abstract_model(config, mesh=mesh)`."""
  _, abs_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
  _, p_state, b_state, rest_state = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
  _, abs_p_state, abs_b_state, _ = nnx.split(abs_model, nnx.Param, moe.MoEBiasVar, ...)

  def _put_leaf(arr: Any, aval: Any) -> Any:
    return jax.device_put(arr, aval.sharding)

  p_sharded = jax.tree.map(_put_leaf, p_state, abs_p_state)
  b_sharded = jax.tree.map(_put_leaf, b_state, abs_b_state)
  nnx.update(model, p_sharded, b_sharded, rest_state)


def _perturb_row_to_target_kl(row_logits: np.ndarray, target_kl: float) -> np.ndarray:
  """Perturbs argmax logit in a copy of 1D `row_logits` via bisection to reach `KL(perturbed || orig) ~= target_kl`."""
  orig = np.asarray(row_logits, dtype=np.float32)
  argmax_idx = int(np.argmax(orig))
  orig_j = jnp.asarray(orig[None, :], dtype=jnp.float32)
  lo, hi = 0.0, 25.0
  best = orig.copy()
  for _ in range(32):
    mid = 0.5 * (lo + hi)
    cand = orig.copy()
    cand[argmax_idx] = orig[argmax_idx] + np.float32(mid)
    kl_val = float(
        masked_prefill_runner._compute_per_token_kl(  # pylint: disable=protected-access
            jnp.asarray(cand[None, :], dtype=jnp.float32), orig_j
        )[0]
    )
    best = cand
    if kl_val < target_kl:
      lo = mid
    else:
      hi = mid
  return best


def _copy_unscanned_to_scanned(mt_scanned: Any, mt_unscanned: Any, config: Any) -> None:
  """Copies weights from an unscanned MaxText model into a scanned MaxText model."""
  if config.use_multimodal:
    _, v_p, v_b, _ = nnx.split(mt_unscanned.vision_encoder, nnx.Param, moe.MoEBiasVar, ...)
    nnx.update(mt_scanned.vision_encoder, v_p, v_b)
  mt_scanned.token_embedder.embedding[...] = mt_unscanned.token_embedder.embedding[...]
  mt_scanned.decoder.decoder_norm.scale[...] = mt_unscanned.decoder.decoder_norm.scale[...]
  mt_scanned.decoder.logits_dense.kernel[...] = mt_unscanned.decoder.logits_dense.kernel[...]

  specs = deepseek41.build_layer_specs(config)
  scan_axis = config.param_scan_axis
  for k, segment in enumerate(deepseek41.build_segments(config)):
    group_params = []
    group_biases = []
    for group in segment.groups:
      tmp_block = deepseek41.DeepSeek41Block(
          config=config,
          mesh=mt_scanned.mesh,
          model_mode=mt_scanned.decoder.model_mode,
          specs=tuple(specs[idx] for idx in group),
          quant=mt_scanned.quant,
          rngs=nnx.Rngs(params=0),
      )
      for j, layer_idx in enumerate(group):
        unscanned_layer = getattr(mt_unscanned.decoder, f"layers_{layer_idx}")
        _, p_state, b_state, _ = nnx.split(unscanned_layer, nnx.Param, moe.MoEBiasVar, ...)
        nnx.update(getattr(tmp_block, f"layers_{j}"), p_state, b_state)
      _, p_state, b_state, _ = nnx.split(tmp_block, nnx.Param, moe.MoEBiasVar, ...)
      group_params.append(nnx.to_pure_dict(p_state))
      group_biases.append(nnx.to_pure_dict(b_state))

    def _stack_param_leaves(*leaves: jax.Array) -> jax.Array:
      stacked = jnp.stack(leaves, axis=0)
      return jnp.moveaxis(stacked, 0, scan_axis) if stacked.ndim > scan_axis else stacked

    def _stack_bias_leaves(*leaves: jax.Array) -> jax.Array:
      return jnp.stack(leaves, axis=0)

    stacked_params = jax.tree.map(_stack_param_leaves, group_params[0], *group_params[1:])
    stacked_biases = jax.tree.map(_stack_bias_leaves, group_biases[0], *group_biases[1:])
    stack = getattr(mt_scanned.decoder, f"segments_{k}")
    graphdef, s_params, s_biases, s_other = nnx.split(stack, nnx.Param, moe.MoEBiasVar, ...)
    nnx.replace_by_pure_dict(s_params, stacked_params)
    nnx.replace_by_pure_dict(s_biases, stacked_biases)
    setattr(mt_scanned.decoder, f"segments_{k}", nnx.merge(graphdef, s_params, s_biases, s_other))


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class DeepSeekV41MultimodalE2EChecksTest(unittest.TestCase):
  """Verifies DeepSeek-V4.1 multimodal audit, Orbax export, logit/prefill runners, and AOT checks."""

  @classmethod
  def setUpClass(cls) -> None:
    cls._orig_matmul_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    cls.addClassCleanup(jax.config.update, "jax_default_matmul_precision", cls._orig_matmul_precision)
    cls.tmp_root = Path(tempfile.mkdtemp(prefix="ds41_mm_unit_"))
    cls.addClassCleanup(shutil.rmtree, cls.tmp_root, ignore_errors=True)
    if HAS_TORCH:
      cls.tokenizer = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
      cls.ref_args = multimodal.mini_model_args()
      cls.ref_mm = multimodal.build_multimodal_model(cls.ref_args, tokenizer=cls.tokenizer)
      multimodal.init_mini_reference(cls.ref_mm, seed=42)
      for layer in cls.ref_mm.layers:
        with torch.no_grad():
          layer.ffn.gate.bias.copy_(torch.linspace(-0.05, 0.05, cls.ref_args.n_routed_experts, dtype=torch.float32))
          if layer.ffn.gate.bias_vl is not None:
            layer.ffn.gate.bias_vl.copy_(torch.linspace(0.1, -0.1, cls.ref_args.n_routed_experts, dtype=torch.float32))
      with torch.no_grad():
        for name, param in cls.ref_mm.named_parameters():
          if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
            param.copy_(param.to(torch.float32))
          else:
            param.copy_(param.to(torch.bfloat16).to(torch.float32))

      cls.hf_tensors = _build_mini_hf_tensors_from_ref(cls.ref_mm, cls.ref_args, include_mtp=True)
      cls.hf_ckpt_dir = cls.tmp_root / "hf_clean"
      _save_safetensors_dir(cls.hf_ckpt_dir, cls.hf_tensors)
    else:
      cls.tokenizer = None
      cls.ref_args = _MultimodalShim.mini_model_args()
      cls.ref_mm = None
      cls.hf_tensors = {}
      cls.hf_ckpt_dir = cls.tmp_root / "hf_clean"

    num_devices = len(jax.devices())
    fsdp_par = 2 if num_devices >= 2 else 1
    ep_par = 2 if num_devices >= 4 else 1
    max_target_len = multimodal.MINI_SEQ_LEN if multimodal is not None else 40
    tok_kw = (
        {}
        if HAS_TORCH
        else {
            "tokenizer_path": mm_testing.OFFLINE_TOKENIZER_PATH,
            "ds41_engram_compressed_vocab_size": 0,
        }
    )
    cls.cfg_unscanned = mm_testing.mini_vision_maxtext_config(
        scan_layers=False,
        dtype="float32",
        weight_dtype="bfloat16",
        dtype_mm="float32",
        matmul_precision="highest",
        remat_policy="none",
        ds41_activation_quant=False,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=fsdp_par,
        ici_expert_parallelism=ep_par,
        max_target_length=max_target_len,
        **tok_kw,
    )
    cls.mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cls.cfg_unscanned), cls.cfg_unscanned.mesh_axes)
    cls.mt_unscanned = model_creation_utils.from_config(
        cls.cfg_unscanned,
        mesh=cls.mesh,
        rngs=nnx.Rngs(params=0),
        model_mode=MODEL_MODE_TRAIN,
    )
    if HAS_TORCH:
      mm_testing.copy_full_multimodal_model(cls.mt_unscanned, cls.ref_mm, cls.cfg_unscanned, jnp.bfloat16)
    _reshard_model_to_abstract_mesh(cls.mt_unscanned, cls.cfg_unscanned, cls.mesh)

    cls.cfg_scanned = mm_testing.mini_vision_maxtext_config(
        scan_layers=True,
        dtype="float32",
        weight_dtype="bfloat16",
        dtype_mm="float32",
        matmul_precision="highest",
        remat_policy="none",
        ds41_activation_quant=False,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=fsdp_par,
        ici_expert_parallelism=ep_par,
        max_target_length=max_target_len,
        **tok_kw,
    )
    cls.mt_scanned = model_creation_utils.from_config(
        cls.cfg_scanned,
        mesh=cls.mesh,
        rngs=nnx.Rngs(params=0),
        model_mode=MODEL_MODE_TRAIN,
    )
    if HAS_TORCH:
      mm_testing.copy_full_multimodal_model(cls.mt_scanned, cls.ref_mm, cls.cfg_scanned, jnp.bfloat16)
    else:
      _copy_unscanned_to_scanned(cls.mt_scanned, cls.mt_unscanned, cls.cfg_scanned)
    _reshard_model_to_abstract_mesh(cls.mt_scanned, cls.cfg_scanned, cls.mesh)

  @classmethod
  def tearDownClass(cls) -> None:
    jax.config.update("jax_default_matmul_precision", cls._orig_matmul_precision)
    shutil.rmtree(cls.tmp_root, ignore_errors=True)

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_audit_and_orbax_export_positive_and_negative_controls(self) -> None:
    """Tests `audit_loaded_model_against_hf`, Orbax export roundtrip, and 7 negative controls."""
    for label, model, cfg in (
        ("unscanned", self.mt_unscanned, self.cfg_unscanned),
        ("scanned", self.mt_scanned, self.cfg_scanned),
    ):
      with self.subTest(layout=label):
        orbax_dir = self.tmp_root / f"orbax_export_{label}"
        res = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
            model=model,
            config=cfg,
            hf_checkpoint_dir=self.hf_ckpt_dir,
            mesh=self.mesh,
            verify_all_text_leaves=True,
            export_orbax_dir=orbax_dir,
        )
        self.assertTrue(res["passed"], msg=f"Clean audit failed for {label}: {res}")
        self.assertEqual(res["total_bit_mismatches"], 0)
        self.assertEqual(res["total_leaf_failures"], 0)
        self.assertEqual(res["unloaded_mtp_keys"], 2)
        self.assertEqual(res["unexpected_unloaded_hf_keys"], [])
        self.assertEqual(res["sharding_mismatches"], [])
        self.assertEqual(res["missing_maxtext_leaves"], [])
        self.assertEqual(res["unexpected_maxtext_leaves"], [])

        items_path = Path(res["exported_orbax_items_path"])
        self.assertTrue(items_path.exists())

        _, p_orig, b_orig, _ = nnx.split(model, nnx.Param, moe.MoEBiasVar, ...)
        orig_leaves = deepseek_v41_mm_audit.flatten_state_arrays(nnx.to_pure_dict(nnx.merge_state(p_orig, b_orig)))
        _, abs_model = model_creation_utils.create_nnx_abstract_model(cfg, mesh=self.mesh)
        _, abs_p, abs_b, _ = nnx.split(abs_model, nnx.Param, moe.MoEBiasVar, ...)
        abs_params = nnx.to_pure_dict(nnx.merge_state(abs_p, abs_b))
        abs_train_state = checkpointing.train_state.TrainState(
            step=0, apply_fn=None, params={"params": abs_params}, tx=None, opt_state={}
        )
        restored_params = checkpointing.load_params_from_path(
            items_path.as_posix(),
            abs_train_state.params,
            cfg.checkpoint_storage_concurrent_gb,
            use_ocdbt=bool(cfg.checkpoint_storage_use_ocdbt),
            use_zarr3=bool(cfg.checkpoint_storage_use_zarr3),
        )
        restored_leaves = deepseek_v41_mm_audit.flatten_state_arrays(restored_params["params"])
        self.assertEqual(set(orig_leaves.keys()), set(restored_leaves.keys()))
        for k, orig_arr in orig_leaves.items():
          ok, n_bad = deepseek_v41_mm_audit._bitwise_equal(  # pylint: disable=protected-access
              np.asarray(orig_arr), np.asarray(restored_leaves[k])
          )
          self.assertTrue(ok, msg=f"Orbax roundtrip mismatch on {k} ({n_bad} elements)")

    # Negative controls 1..4: 1-ULP bit flips on vision, aligner, bias_vl, and engram.
    for target_key in (
        "vision.blocks.0.attn.wqkv.weight",
        "aligner.w2.bias",
        "layers.0.ffn.gate.bias_vl",
        "layers.1.engram.embed.weight",
    ):
      with self.subTest(negctl_1ulp=target_key):
        bad_tensors = dict(self.hf_tensors)
        bad_tensors[target_key] = _flip_one_ulp_torch(self.hf_tensors[target_key])
        bad_dir = self.tmp_root / f"neg_1ulp_{target_key.replace('.', '_')}"
        _save_safetensors_dir(bad_dir, bad_tensors)

        res_bad = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
            model=self.mt_unscanned,
            config=self.cfg_unscanned,
            hf_checkpoint_dir=bad_dir,
            mesh=self.mesh,
            verify_all_text_leaves=True,
        )
        self.assertFalse(res_bad["passed"], msg=f"1-ULP flip on {target_key} was not caught!")
        self.assertGreater(res_bad["total_bit_mismatches"], 0)

        blocked_orbax = self.tmp_root / f"blocked_orbax_{target_key.replace('.', '_')}"
        with self.assertRaises(RuntimeError):
          deepseek_v41_mm_audit.audit_loaded_model_against_hf(
              model=self.mt_unscanned,
              config=self.cfg_unscanned,
              hf_checkpoint_dir=bad_dir,
              mesh=self.mesh,
              verify_all_text_leaves=True,
              export_orbax_dir=blocked_orbax,
          )
        self.assertFalse((blocked_orbax / "0" / "items").exists())

    # Negative control 5: missing bias_vl key.
    missing_bvl_tensors = {k: v for k, v in self.hf_tensors.items() if k != "layers.0.ffn.gate.bias_vl"}
    missing_bvl_dir = self.tmp_root / "neg_missing_bias_vl"
    _save_safetensors_dir(missing_bvl_dir, missing_bvl_tensors)
    res_missing = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
        model=self.mt_unscanned,
        config=self.cfg_unscanned,
        hf_checkpoint_dir=missing_bvl_dir,
        mesh=self.mesh,
    )
    self.assertFalse(res_missing["passed"])
    self.assertIn("layers.0.ffn.gate.bias_vl", res_missing["missing_hf_keys"])

    # Negative control 6: wrong F32 header dtype on vision weight (expected BF16).
    wrong_hdr_tensors = dict(self.hf_tensors)
    wrong_hdr_tensors["vision.blocks.0.attn.wqkv.weight"] = (
        self.hf_tensors["vision.blocks.0.attn.wqkv.weight"].to(torch.float32).contiguous()
    )
    wrong_hdr_dir = self.tmp_root / "neg_wrong_hdr_dtype"
    _save_safetensors_dir(wrong_hdr_dir, wrong_hdr_tensors)
    res_hdr = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
        model=self.mt_unscanned,
        config=self.cfg_unscanned,
        hf_checkpoint_dir=wrong_hdr_dir,
        mesh=self.mesh,
    )
    self.assertFalse(res_hdr["passed"])
    self.assertGreater(len(res_hdr["header_dtype_mismatches"]), 0)

    # Negative control 7: forbidden .scale on vision weight.
    forbid_scale_tensors = dict(self.hf_tensors)
    forbid_scale_tensors["vision.blocks.0.attn.wqkv.weight.scale"] = torch.ones((1, 1), dtype=torch.float32)
    forbid_scale_dir = self.tmp_root / "neg_forbid_scale"
    _save_safetensors_dir(forbid_scale_dir, forbid_scale_tensors)
    res_scale = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
        model=self.mt_unscanned,
        config=self.cfg_unscanned,
        hf_checkpoint_dir=forbid_scale_dir,
        mesh=self.mesh,
    )
    self.assertFalse(res_scale["passed"])
    self.assertIn("vision.blocks.0.attn.wqkv.weight.scale", res_scale["forbidden_scale_violations"])

    # Negative control 8: bitwise-identical leaf placed on wrong NamedSharding(self.mesh, PartitionSpec()).
    orig_emb = self.mt_unscanned.token_embedder.embedding[...]
    self.assertIsInstance(orig_emb.sharding, jax.sharding.NamedSharding)
    self.assertNotEqual(orig_emb.sharding.spec, jax.sharding.PartitionSpec())
    try:
      self.mt_unscanned.token_embedder.embedding.set_value(
          jax.device_put(
              orig_emb,
              jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec()),
          )
      )
      self.assertEqual(
          self.mt_unscanned.token_embedder.embedding[...].sharding.spec,
          jax.sharding.PartitionSpec(),
      )
      res_shard = deepseek_v41_mm_audit.audit_loaded_model_against_hf(
          model=self.mt_unscanned,
          config=self.cfg_unscanned,
          hf_checkpoint_dir=self.hf_ckpt_dir,
          mesh=self.mesh,
          verify_all_text_leaves=True,
      )
      self.assertFalse(res_shard["passed"])
      self.assertEqual(res_shard["total_bit_mismatches"], 0)
      self.assertGreaterEqual(len(res_shard["sharding_mismatches"]), 1)
      self.assertGreaterEqual(res_shard["total_leaf_failures"], 1)
      self.assertTrue(any("token_embedder/embedding" in m for m in res_shard["sharding_mismatches"]))

      blocked_shard_orbax = self.tmp_root / "blocked_orbax_wrong_sharding"
      with self.assertRaises(RuntimeError):
        deepseek_v41_mm_audit.audit_loaded_model_against_hf(
            model=self.mt_unscanned,
            config=self.cfg_unscanned,
            hf_checkpoint_dir=self.hf_ckpt_dir,
            mesh=self.mesh,
            verify_all_text_leaves=True,
            export_orbax_dir=blocked_shard_orbax,
        )
      self.assertFalse((blocked_shard_orbax / "0" / "items").exists())
    finally:
      self.mt_unscanned.token_embedder.embedding.set_value(orig_emb)

    # Index accounting positive + negative control on mini-twin index.json (and optional 749B DS41_HF_INDEX_JSON).
    idx_res = deepseek_v41_mm_audit.audit_hf_index_accounting(
        self.hf_ckpt_dir / "model.safetensors.index.json",
        self.cfg_unscanned,
    )
    self.assertTrue(idx_res["passed"], msg=f"Index accounting failed: {idx_res}")
    self.assertEqual(idx_res["unloaded_mtp_keys"], 2)
    self.assertEqual(idx_res["unexpected_unloaded_hf_keys"], [])

    idx_bad = deepseek_v41_mm_audit.audit_hf_index_accounting(
        forbid_scale_dir / "model.safetensors.index.json",
        self.cfg_unscanned,
    )
    self.assertFalse(idx_bad["passed"])
    self.assertIn("vision.blocks.0.attn.wqkv.weight.scale", idx_bad["forbidden_scale_violations"])

    if "DS41_HF_INDEX_JSON" in os.environ and os.path.exists(os.environ["DS41_HF_INDEX_JSON"]):
      aot_mod = _load_aot_check_module()
      cfg_749b = aot_mod._build_config_from_overrides(  # pylint: disable=protected-access
          aot_mod.get_preset_spec("tpu7x_64c_mm_scanned_fwd")["overrides"],
          cpu_mini=False,
      )
      res_749b = deepseek_v41_mm_audit.audit_hf_index_accounting(
          os.environ["DS41_HF_INDEX_JSON"],
          cfg_749b,
      )
      self.assertTrue(res_749b["passed"], msg=f"749B index accounting failed: {res_749b}")
      self.assertEqual(res_749b["total_hf_keys"], 96085)
      self.assertEqual(res_749b["consumed_hf_keys"], 93684)
      self.assertEqual(res_749b["unloaded_mtp_keys"], 2401)

  def test_logit_runner_mm_and_image_mask_negative_control(self) -> None:
    """Tests merged text+MM verification in `logit_runner` and `encoder_image_masks` routing."""
    batch = multimodal.build_mini_vl_batch(
        self.ref_args,
        grid=multimodal.MINI_GRID_PADDED,
        batch_size=1,
        span_starts=((2,),),
        seed=99,
    )
    tokens_np = batch.input_ids.numpy().astype(np.int32)[0]
    valid_patch_0 = batch.patches_bnpd.numpy().astype(np.float32)[0]  # [1, 35, 588]
    img_tok_id = int(self.cfg_unscanned.ds41_image_token_id)
    token_types = batch.token_types.numpy().astype(np.int32)[0]
    self.assertEqual(set(np.unique(token_types).tolist()), {-1, 0, 1, 2, 3})
    self.assertTrue(np.array_equal(token_types >= 0, tokens_np == img_tok_id))

    rng = np.random.default_rng(1234)
    fwd_dir = self.tmp_root / "fwd_checker_case"
    fwd_dir.mkdir(parents=True, exist_ok=True)

    # 1. Explicit test of padded-slot multimodal `mm_logit.pkl` entry (`encoder_images` [4, P, 588],
    #    `encoder_image_masks=[1, 0, 0, 0]`, `num_images=1`, `sample_id="L1"`, no `patches` or `name` key).
    u6a_pkl_patches = np.zeros((4, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32)
    u6a_pkl_patches[0] = valid_patch_0[0]
    u6a_pkl_patches[1:] = (
        rng.standard_normal((3, valid_patch_0.shape[1], valid_patch_0.shape[2])).astype(np.float32) * 25.0
    )
    u6a_pkl_mask = np.array([1, 0, 0, 0], dtype=np.int32)

    ids_j = jnp.asarray(tokens_np[None, :], dtype=jnp.int32)
    pos_j = jnp.arange(tokens_np.shape[0], dtype=jnp.int32)[None, :]
    seg_j = jnp.ones_like(ids_j, dtype=jnp.int32)
    pkl_logits = np.asarray(
        self.mt_unscanned(
            decoder_input_tokens=ids_j,
            decoder_positions=pos_j,
            decoder_segment_ids=seg_j,
            encoder_images=jnp.asarray(u6a_pkl_patches[None, ...], dtype=jnp.float32),
            encoder_image_masks=jnp.asarray(u6a_pkl_mask[None, :], dtype=jnp.int32),
            enable_dropout=False,
        )[0],
        dtype=np.float32,
    )
    u6a_pkl_entry = {
        "sample_id": "L1",
        "prompt": "L1 prompt",
        "tokens": tokens_np.tolist(),
        "logits": pkl_logits,
        "token_types": token_types,
        "encoder_images": u6a_pkl_patches,
        "encoder_image_masks": u6a_pkl_mask,
        "pixel_values": u6a_pkl_patches,
        "grid_hw": np.array([[6, 6]], dtype=np.int32),
        "num_images": 1,
    }
    _, _, _, _, _, pkl_imgs, pkl_masks, _ = logit_runner.get_data_with_masks(
        u6a_pkl_entry, self.cfg_unscanned, max_images_per_sample=4
    )
    self.assertIsNotNone(pkl_imgs)
    self.assertIsNotNone(pkl_masks)
    self.assertTrue(np.all(np.asarray(pkl_masks)[:, 0] == 1))
    self.assertTrue(np.all(np.asarray(pkl_masks)[:, 1:] == 0))

    mm_logit_pkl_path = fwd_dir / "mm_logit.pkl"
    with open(mm_logit_pkl_path, "wb") as f:
      pickle.dump([u6a_pkl_entry], f)

    # 2. Build L1 .npz with 4 patch slots where slot 0 has large dummy noise (masked out by image_masks=[0, 1, 0, 0])
    #    and slot 1 holds the 1 valid image (`sum(image_masks) == num_images == 1`). If forward_pass_logit_checker
    #    drops encoder_image_masks, slot 0 noise is merged into the <|image|> span instead of slot 1, causing a
    #    massive KL divergence failure.
    padded_4_slots = np.zeros((4, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32)
    padded_4_slots[0] = rng.standard_normal(valid_patch_0[0].shape).astype(np.float32) * 50.0
    padded_4_slots[1] = valid_patch_0[0]
    image_mask_slot1 = np.array([0, 1, 0, 0], dtype=np.int32)

    imgs_j = jnp.asarray(padded_4_slots[None, ...], dtype=jnp.float32)
    masks_j = jnp.asarray(image_mask_slot1[None, :], dtype=jnp.int32)

    mm_logits = np.asarray(
        self.mt_unscanned(
            decoder_input_tokens=ids_j,
            decoder_positions=pos_j,
            decoder_segment_ids=seg_j,
            encoder_images=imgs_j,
            encoder_image_masks=masks_j,
            enable_dropout=False,
        )[0],
        dtype=np.float32,
    )

    text_tokens = np.where(tokens_np[:12] == img_tok_id, 5, tokens_np[:12]).astype(np.int32)
    t_ids, t_seg, t_pos, _, t_seq_len, _, _, _ = logit_runner.get_data_with_masks(
        {
            "prompt": "short_text_prompts",
            "tokens": text_tokens.tolist(),
            "logits": np.zeros((int(text_tokens.shape[0]), int(mm_logits.shape[-1])), dtype=np.float32),
        },
        self.cfg_unscanned,
    )
    text_logits = np.asarray(
        self.mt_unscanned(
            decoder_input_tokens=jnp.asarray(t_ids, dtype=jnp.int32),
            decoder_positions=jnp.asarray(t_pos, dtype=jnp.int32),
            decoder_segment_ids=jnp.asarray(t_seg, dtype=jnp.int32),
            encoder_images=None,
            encoder_image_masks=None,
            enable_dropout=False,
        )[0, :t_seq_len, :],
        dtype=np.float32,
    )

    text_jsonl = fwd_dir / "text_goldens.jsonl"
    with jsonlines.open(text_jsonl, "w") as writer:
      writer.write(
          {
              "name": "short_text_prompts",
              "prompt": "short_text_prompts",
              "tokens": text_tokens.tolist(),
              "logits": text_logits.tolist(),
          }
      )

    mm_npz_dir = fwd_dir / "mm_npz"
    mm_npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        mm_npz_dir / "L1.npz",
        sample_id=np.array("L1"),
        tokens=tokens_np,
        logits=mm_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )

    summary_json = fwd_dir / "summary.json"
    test_args = argparse.Namespace(
        atol=None,
        rtol=1e-5,
        token_size=None,
        max_kl_div=5e-3,
        golden_logits_path=text_jsonl.as_posix(),
        mm_golden_logits_path=mm_npz_dir.as_posix(),
        thresholds_json=_THRESHOLDS_JSON_PATH.as_posix(),
        enable_long_text_gate=False,
        max_images_per_sample=4,
        run_ds41_mm_audit=False,
        hf_checkpoint_dir_for_audit="",
        export_orbax_dir="",
        summary_json=summary_json.as_posix(),
        hf_model_path="",
        run_hf_model=False,
        output_logits_path="",
        gcs_output_logits_path="",
        clip_logits_epsilon=None,
        skip_first_token=False,
        ckpt_type="nnx",
        trust_remote_code=True,
        reference_model="",
    )

    summary = logit_runner.main(
        self.cfg_unscanned,
        test_args,
        preloaded_model=self.mt_unscanned,
    )
    expected_img_tokens = int(np.sum(token_types >= 0))
    expected_txt_tokens = int(np.sum(token_types == -1))
    self.assertGreater(expected_img_tokens, int(np.sum(token_types == 1)))
    self.assertIsNotNone(summary)
    self.assertTrue(summary["passed"])
    self.assertTrue(summary["all_passed"])
    self.assertTrue(summary["text_only_summary"]["all_passed"])
    self.assertTrue(summary["multimodal_summary"]["all_passed"])
    self.assertEqual(summary["multimodal_summary"]["total_image_tokens"], expected_img_tokens)
    txt_rec = summary["samples"][0]
    self.assertEqual(txt_rec["image_token_count"], 0)
    self.assertEqual(txt_rec["text_token_count"], int(text_tokens.shape[0]))
    mm_rec = summary["samples"][1]
    self.assertEqual(mm_rec["name"], "L1")
    self.assertEqual(mm_rec["tier"], "mm_short_prompts")
    self.assertTrue(mm_rec["is_multimodal"])
    self.assertEqual(mm_rec["image_token_count"], expected_img_tokens)
    self.assertEqual(mm_rec["text_token_count"], expected_txt_tokens)
    self.assertEqual(mm_rec["all_positions_metrics"]["seq_len"], int(tokens_np.shape[0]))
    self.assertLessEqual(mm_rec["all_positions_metrics"]["max_kl"], 1e-6)
    self.assertEqual(mm_rec["all_positions_metrics"]["last_token_top10_matches"], 10)
    self.assertTrue(mm_rec["all_positions_gates"]["last_token_top10_matches_pass"])
    self.assertIsNotNone(mm_rec["text_positions_metrics"])
    self.assertEqual(mm_rec["text_positions_metrics"]["seq_len"], expected_txt_tokens)
    self.assertLessEqual(mm_rec["text_positions_metrics"]["max_kl"], 1e-6)
    self.assertIsNotNone(mm_rec["image_positions_metrics"])
    self.assertEqual(mm_rec["image_positions_metrics"]["seq_len"], expected_img_tokens)
    self.assertLessEqual(mm_rec["image_positions_metrics"]["max_kl"], 1e-6)
    self.assertEqual(
        mm_rec["frozen_topk_diagnostic"],
        {"ran": False, "reason": "text_positions_passed"},
    )

    # Verify padded-slot multimodal `mm_logit.pkl` end-to-end through `logit_runner.main`.
    pkl_args = copy.copy(test_args)
    pkl_args.golden_logits_path = ""
    pkl_args.mm_golden_logits_path = mm_logit_pkl_path.as_posix()
    summary_pkl = logit_runner.main(
        self.cfg_unscanned,
        pkl_args,
        preloaded_model=self.mt_unscanned,
    )
    self.assertIsNotNone(summary_pkl)
    self.assertTrue(summary_pkl["all_passed"])
    self.assertEqual(summary_pkl["samples"][0]["name"], "L1")
    self.assertEqual(summary_pkl["samples"][0]["tier"], "mm_short_prompts")

    # Verify short (<= 16) vs long (> 16) threshold tier routing.
    with open(_THRESHOLDS_JSON_PATH, "r", encoding="utf-8") as f:
      thresholds_cfg = json.load(f)
    t_short, spec_short = logit_runner.select_forward_threshold_spec(
        thresholds_cfg, sample_name="short_prompt", seq_len=12, is_mm_sample=False
    )
    self.assertEqual(t_short, "short_text_prompts")
    self.assertEqual(spec_short["max_kl"], 2e-7)
    t_long, spec_long = logit_runner.select_forward_threshold_spec(
        thresholds_cfg, sample_name="long_prompt", seq_len=462, is_mm_sample=False
    )
    self.assertEqual(t_long, "long_text_prompts")
    self.assertEqual(spec_long["max_kl"], 0.0212)

    # Negative control 1: corrupted golden logits must fail KL gate.
    corrupt_npz_dir = fwd_dir / "mm_npz_corrupt"
    corrupt_npz_dir.mkdir(parents=True, exist_ok=True)
    bad_logits = mm_logits.copy()
    bad_logits[:, 0] += 25.0
    np.savez(
        corrupt_npz_dir / "L1.npz",
        sample_id=np.array("L1"),
        tokens=tokens_np,
        logits=bad_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    bad_args = copy.copy(test_args)
    bad_args.mm_golden_logits_path = corrupt_npz_dir.as_posix()
    with self.assertRaises(AssertionError):
      logit_runner.main(
          self.cfg_unscanned,
          bad_args,
          preloaded_model=self.mt_unscanned,
      )

    # Negative control 2: mismatched token_types length raises ValueError.
    bad_tt_dir = fwd_dir / "mm_npz_bad_tt"
    bad_tt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        bad_tt_dir / "L1.npz",
        sample_id=np.array("L1"),
        tokens=tokens_np,
        logits=mm_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types[:-3],
        num_images=np.array(1, dtype=np.int32),
    )
    bad_tt_args = copy.copy(test_args)
    bad_tt_args.mm_golden_logits_path = bad_tt_dir.as_posix()
    with self.assertRaises((ValueError, SystemExit)) as cm:
      logit_runner.main(
          self.cfg_unscanned,
          bad_tt_args,
          preloaded_model=self.mt_unscanned,
      )
    if isinstance(cm.exception, SystemExit):
      self.assertEqual(cm.exception.code, 42)

    # Negative control 3: sum(image_masks) != num_images raises ValueError.
    with self.assertRaises(ValueError):
      logit_runner.prepare_ds41_mm_inputs(
          {
              "sample_id": "L1",
              "patches": padded_4_slots,
              "image_masks": np.array([1, 1, 0, 0], dtype=np.int32),
              "num_images": 1,
          },
          4,
          max_images_per_sample=4,
      )

    # Negative controls on logit_runner:
    def _run_fwd_single_npz(
        npz_subdir: str,
        *,
        max_kl_div_override: float | None = 0.005,
        **npz_kwargs: Any,
    ) -> Any:
      d = fwd_dir / npz_subdir
      d.mkdir(parents=True, exist_ok=True)
      np.savez(d / "case.npz", **npz_kwargs)
      local_args = copy.copy(test_args)
      local_args.max_kl_div = max_kl_div_override
      local_args.golden_logits_path = ""
      local_args.mm_golden_logits_path = d.as_posix()
      local_args.summary_json = (d / "summary.json").as_posix()
      return logit_runner.main(
          self.cfg_unscanned,
          local_args,
          preloaded_model=self.mt_unscanned,
      )

    # Multimodal sample with missing token_types raises ValueError.
    with self.assertRaises(ValueError):
      _run_fwd_single_npz(
          "neg_missing_tt",
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          num_images=np.array(1, dtype=np.int32),
      )

    # Out-of-range (4, -2) and non-integer (float32, bool) token_types raise ValueError naming {-1, 0, 1, 2, 3}.
    for bad_label, bad_tt_arr in (
        ("above_max_4", np.where(np.arange(token_types.shape[0]) == 0, 4, token_types).astype(np.int32)),
        ("below_min_neg2", np.where(np.arange(token_types.shape[0]) == 0, -2, token_types).astype(np.int32)),
        ("float32_dtype", token_types.astype(np.float32)),
        ("bool_dtype", (token_types >= 0)),
    ):
      with self.subTest(fwd_bad_tt=bad_label):
        with self.assertRaisesRegex(ValueError, r"\{-1, 0, 1, 2, 3\}"):
          _run_fwd_single_npz(
              f"neg_invalid_tt_{bad_label}",
              sample_id=np.array("L1"),
              tokens=tokens_np,
              logits=mm_logits,
              patches=padded_4_slots,
              image_masks=image_mask_slot1,
              token_types=bad_tt_arr,
              num_images=np.array(1, dtype=np.int32),
          )

    # Mismatch between token_types >= 0 and tokens == ds41_image_token_id raises ValueError.
    first_txt_pos = int(np.flatnonzero(token_types == -1)[0])
    first_img_pos = int(np.flatnonzero(token_types >= 0)[0])
    tt_txt_marked_img = token_types.copy()
    tt_txt_marked_img[first_txt_pos] = 0
    tt_img_marked_txt = token_types.copy()
    tt_img_marked_txt[first_img_pos] = -1
    for mismatch_name, bad_align_tt in (
        ("text_marked_img_start", tt_txt_marked_img),
        ("img_marked_text", tt_img_marked_txt),
    ):
      with self.subTest(fwd_align_mismatch=mismatch_name):
        with self.assertRaisesRegex(ValueError, "ds41_image_token_id"):
          _run_fwd_single_npz(
              f"neg_align_mismatch_{mismatch_name}",
              sample_id=np.array("L1"),
              tokens=tokens_np,
              logits=mm_logits,
              patches=padded_4_slots,
              image_masks=image_mask_slot1,
              token_types=bad_align_tt,
              num_images=np.array(1, dtype=np.int32),
          )

    # Multimodal sample with all-1 token_types (no -1 text positions) raises ValueError.
    with self.assertRaises(ValueError):
      _run_fwd_single_npz(
          "neg_all_one_tt",
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=np.ones_like(token_types, dtype=np.int32),
          num_images=np.array(1, dtype=np.int32),
      )

    # Text-only sample (_has_ds41_mm_images == False) declaring token_types >= 0 (0 or 1) raises ValueError.
    for bad_txt_label in (0, 1):
      text_bad_tt = np.full((int(text_tokens.shape[0]),), -1, dtype=np.int32)
      text_bad_tt[0] = bad_txt_label
      with self.assertRaises(ValueError):
        _run_fwd_single_npz(
            f"neg_text_with_img_tt_{bad_txt_label}",
            sample_id=np.array("short_text_prompts"),
            tokens=text_tokens,
            logits=text_logits,
            token_types=text_bad_tt,
        )

    # Qwen3-style multimodal gate scope on logit_runner (thresholds_json mode, max_kl_div=None):
    # Corrupting ONLY image tokens ({0, 2, 3} frame tokens OR all {0, 1, 2, 3} image tokens) PASSES overall because
    # multimodal gating is strictly on text_positions (-1), while surfacing spikes in image_positions_metrics and
    # all_positions_metrics and NOT triggering FrozenTopK; corrupting TEXT (-1) FAILS.
    fwd_frame_idx = np.flatnonzero(np.isin(token_types, [0, 2, 3]))
    self.assertGreater(len(fwd_frame_idx), 0)
    fwd_bad_frame_logits = mm_logits.copy()
    fwd_bad_frame_logits[fwd_frame_idx, 0] += 25.0
    fwd_frame_summary = _run_fwd_single_npz(
        "scope_corrupt_frame_tokens",
        max_kl_div_override=None,
        sample_id=np.array("L1"),
        tokens=tokens_np,
        logits=fwd_bad_frame_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    self.assertTrue(fwd_frame_summary["passed"])
    self.assertTrue(fwd_frame_summary["all_passed"])
    fwd_frame_rec = fwd_frame_summary["samples"][0]
    self.assertTrue(fwd_frame_rec["passed"])
    self.assertEqual(fwd_frame_rec["failure_reasons"], [])
    self.assertFalse(fwd_frame_rec["all_positions_gates"]["max_kl_pass"])
    self.assertGreater(fwd_frame_rec["all_positions_metrics"]["max_kl"], 1.0)
    self.assertGreater(fwd_frame_rec["image_positions_metrics"]["max_kl"], 1.0)
    self.assertTrue(all(fwd_frame_rec["text_positions_gates"].values()))
    self.assertLessEqual(fwd_frame_rec["text_positions_metrics"]["max_kl"], 1e-6)
    self.assertLessEqual(fwd_frame_rec["max_kl"], 1e-6)
    self.assertEqual(
        fwd_frame_rec["frozen_topk_diagnostic"],
        {"ran": False, "reason": "text_positions_passed"},
    )

    fwd_all_img_idx = np.flatnonzero(token_types != -1)
    self.assertGreater(len(fwd_all_img_idx), len(fwd_frame_idx))
    fwd_bad_all_img_logits = mm_logits.copy()
    fwd_bad_all_img_logits[fwd_all_img_idx, 0] += 25.0
    fwd_all_img_summary = _run_fwd_single_npz(
        "scope_corrupt_all_image_tokens",
        max_kl_div_override=None,
        sample_id=np.array("L1"),
        tokens=tokens_np,
        logits=fwd_bad_all_img_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    self.assertTrue(fwd_all_img_summary["passed"])
    self.assertTrue(fwd_all_img_summary["samples"][0]["passed"])
    self.assertGreater(fwd_all_img_summary["samples"][0]["image_positions_metrics"]["max_kl"], 1.0)
    self.assertGreater(fwd_all_img_summary["samples"][0]["all_positions_metrics"]["max_kl"], 1.0)
    self.assertLessEqual(fwd_all_img_summary["samples"][0]["text_positions_metrics"]["max_kl"], 1e-6)
    self.assertFalse(fwd_all_img_summary["samples"][0]["frozen_topk_diagnostic"]["ran"])

    fwd_bad_text_logits = mm_logits.copy()
    fwd_bad_text_logits[first_txt_pos, 0] += 25.0
    with self.assertRaises(AssertionError):
      _run_fwd_single_npz(
          "scope_corrupt_text_tokens",
          max_kl_div_override=None,
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=fwd_bad_text_logits,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(fwd_dir / "scope_corrupt_text_tokens" / "summary.json", "r", encoding="utf-8") as f:
      fwd_txt_summary = json.load(f)
    self.assertFalse(fwd_txt_summary["passed"])
    self.assertFalse(fwd_txt_summary["all_passed"])
    self.assertFalse(fwd_txt_summary["samples"][0]["text_positions_gates"]["max_kl_pass"])
    self.assertFalse(fwd_txt_summary["samples"][0]["all_positions_gates"]["max_kl_pass"])
    self.assertEqual(
        fwd_txt_summary["samples"][0]["frozen_topk_diagnostic"],
        {"ran": False, "reason": "missing_ref_topk_experts"},
    )

    # Verify last_token_top10_matches is gated on text_positions for multimodal mm_short_prompts samples:
    last_txt_pos = int(np.flatnonzero(token_types == -1)[-1])
    fwd_bad_top10_logits = mm_logits.copy()
    fwd_bad_top10_logits[last_txt_pos, :15] += np.linspace(50.0, 40.0, 15, dtype=np.float32)
    with self.assertRaises(AssertionError):
      _run_fwd_single_npz(
          "scope_corrupt_last_text_top10",
          max_kl_div_override=None,
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=fwd_bad_top10_logits,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(fwd_dir / "scope_corrupt_last_text_top10" / "summary.json", "r", encoding="utf-8") as f:
      fwd_top10_summary = json.load(f)
    self.assertFalse(fwd_top10_summary["samples"][0]["text_positions_gates"]["last_token_top10_matches_pass"])

    # Non-binary image_masks ([-1, 2, 0, 0] with num_images=1) raises ValueError.
    with self.assertRaises(ValueError):
      logit_runner.prepare_ds41_mm_inputs(
          {
              "sample_id": "L1",
              "patches": padded_4_slots,
              "image_masks": np.array([-1, 2, 0, 0], dtype=np.int32),
              "num_images": 1,
          },
          4,
          max_images_per_sample=4,
      )
    with self.assertRaises(ValueError):
      _run_fwd_single_npz(
          "neg_nonbinary_mask",
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_4_slots,
          image_masks=np.array([-1, 2, 0, 0], dtype=np.int32),
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )

    # Padded-slot image_masks ([0, 1, 0, 0] on n_img=1 patches [1, P, 588], num_images=1) raises ValueError.
    with self.assertRaises(ValueError):
      logit_runner.prepare_ds41_mm_inputs(
          {
              "sample_id": "L1",
              "patches": valid_patch_0,
              "image_masks": np.array([0, 1, 0, 0], dtype=np.int32),
              "num_images": 1,
          },
          4,
          max_images_per_sample=4,
      )
    with self.assertRaises(ValueError):
      _run_fwd_single_npz(
          "neg_padded_slot_mask",
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=valid_patch_0,
          image_masks=np.array([0, 1, 0, 0], dtype=np.int32),
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )

    # All-zero image_masks ([0, 0, 0, 0] with num_images=None) and n_img=0 raise ValueError.
    with self.assertRaises(ValueError):
      logit_runner.prepare_ds41_mm_inputs(
          {
              "sample_id": "L1",
              "patches": padded_4_slots,
              "image_masks": np.array([0, 0, 0, 0], dtype=np.int32),
          },
          4,
          max_images_per_sample=4,
      )
    with self.assertRaises(ValueError):
      logit_runner.prepare_ds41_mm_inputs(
          {
              "sample_id": "L1",
              "patches": np.zeros((0, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32),
          },
          4,
          max_images_per_sample=4,
      )
    with self.assertRaises(ValueError):
      _run_fwd_single_npz(
          "neg_all_zero_mask",
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_4_slots,
          image_masks=np.array([0, 0, 0, 0], dtype=np.int32),
          token_types=token_types,
      )

    # Tie-flip envelope on logit_runner: near-tie spike > max_kl_div=3e-3 passes with tie_flip_logits.
    fwd_text_idx = int(np.flatnonzero(token_types == -1)[0])
    fwd_spiked_logits = mm_logits.copy()
    fwd_spiked_logits[fwd_text_idx] = _perturb_row_to_target_kl(mm_logits[fwd_text_idx], 0.0045)
    with self.assertRaises(AssertionError):
      _run_fwd_single_npz(
          "env_fwd_no_tie_flip",
          max_kl_div_override=0.003,
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=fwd_spiked_logits,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    fwd_tf_summary = _run_fwd_single_npz(
        "env_fwd_with_tie_flip",
        max_kl_div_override=0.003,
        sample_id=np.array("L1"),
        tokens=tokens_np,
        logits=fwd_spiked_logits,
        tie_flip_logits=mm_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    self.assertTrue(fwd_tf_summary["all_passed"])
    fwd_tf_rec = fwd_tf_summary["samples"][0]
    # On multimodal samples, tie_flip_logits applies ONLY to text_positions max_kl;
    # all_positions and image_positions remain raw primary-golden diagnostics.
    self.assertFalse(fwd_tf_rec["all_positions_metrics"]["tie_flip_envelope_used"])
    self.assertGreater(fwd_tf_rec["all_positions_metrics"]["max_kl"], 0.003)
    self.assertFalse(fwd_tf_rec["image_positions_metrics"]["tie_flip_envelope_used"])
    self.assertTrue(fwd_tf_rec["text_positions_metrics"]["tie_flip_envelope_used"])
    self.assertGreater(fwd_tf_rec["text_positions_metrics"]["raw_primary_max_kl"], 0.003)
    self.assertLessEqual(fwd_tf_rec["text_positions_metrics"]["max_kl"], 1e-6)

    # Negative control on logit_runner (mm_short_prompts max_kl_limit from thresholds.json):
    # (a) Single-token +1.5 * max_kl_limit spike on a text_positions index MUST fail
    #     text_positions max_kl (passed == False) even when tie_flip_logits envelope is present!
    with open(_THRESHOLDS_JSON_PATH, "r", encoding="utf-8") as f:
      mm_short_max_kl_limit = float(
          json.load(f)["forward_logit_checker"]["multimodal_cases"]["mm_short_prompts"]["max_kl"]
      )
    fwd_env_iii_spike = mm_logits.copy()
    fwd_env_iii_spike[fwd_text_idx] = _perturb_row_to_target_kl(mm_logits[fwd_text_idx], 1.5 * mm_short_max_kl_limit)
    fwd_env_iii_env = mm_logits.copy()
    fwd_env_iii_env[fwd_text_idx] = fwd_env_iii_spike[fwd_text_idx]
    with self.assertRaises(AssertionError):
      _run_fwd_single_npz(
          "env_iii_fwd_spike_fails_max_kl",
          max_kl_div_override=None,
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=fwd_env_iii_spike,
          tie_flip_logits=fwd_env_iii_env,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(fwd_dir / "env_iii_fwd_spike_fails_max_kl" / "summary.json", "r", encoding="utf-8") as f:
      fwd_env_iii_summary = json.load(f)
    self.assertFalse(fwd_env_iii_summary["passed"])
    self.assertFalse(fwd_env_iii_summary["samples"][0]["passed"])
    self.assertFalse(fwd_env_iii_summary["samples"][0]["text_positions_gates"]["max_kl_pass"])
    self.assertGreater(fwd_env_iii_summary["samples"][0]["text_positions_metrics"]["max_kl"], mm_short_max_kl_limit)

    # (b) Aggregate-metric no-envelope invariant: even if tie_flip_logits == mm_logits masks max_kl to 0,
    #     a text-token spike pushing primary mean_kl above mm_short_prompts mean_kl=5e-4 still fails mean_kl_pass!
    fwd_mean_bad = mm_logits.copy()
    fwd_mean_bad[fwd_text_idx] = _perturb_row_to_target_kl(mm_logits[fwd_text_idx], 0.05)
    with self.assertRaises(AssertionError):
      _run_fwd_single_npz(
          "env_fwd_mean_kl_no_envelope",
          max_kl_div_override=None,
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=fwd_mean_bad,
          tie_flip_logits=mm_logits,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(fwd_dir / "env_fwd_mean_kl_no_envelope" / "summary.json", "r", encoding="utf-8") as f:
      fwd_mean_bad_summary = json.load(f)
    self.assertFalse(fwd_mean_bad_summary["samples"][0]["passed"])
    self.assertFalse(fwd_mean_bad_summary["samples"][0]["text_positions_gates"]["mean_kl_pass"])

    # (c) Multimodal text_positions failure index must report offending text token index (NOT higher-KL image token)!
    img_spike_idx = int(np.flatnonzero(token_types >= 0)[0])
    self.assertEqual(img_spike_idx, 2)
    txt_spike_idx = int(np.flatnonzero(token_types == -1)[-2])
    self.assertNotEqual(txt_spike_idx, img_spike_idx)
    fwd_idx_check = mm_logits.copy()
    fwd_idx_check[img_spike_idx] = _perturb_row_to_target_kl(mm_logits[img_spike_idx], 2.5)
    fwd_idx_check[txt_spike_idx] = _perturb_row_to_target_kl(mm_logits[txt_spike_idx], 0.25)
    with self.assertRaises(AssertionError):
      _run_fwd_single_npz(
          "mm_text_max_kl_index_reports_text_not_image",
          max_kl_div_override=None,
          sample_id=np.array("L1"),
          tokens=tokens_np,
          logits=fwd_idx_check,
          patches=padded_4_slots,
          image_masks=image_mask_slot1,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(fwd_dir / "mm_text_max_kl_index_reports_text_not_image" / "summary.json", "r", encoding="utf-8") as f:
      fwd_idx_summary = json.load(f)
    fwd_idx_rec = fwd_idx_summary["samples"][0]
    self.assertEqual(fwd_idx_rec["max_kl_index"], txt_spike_idx)
    self.assertTrue(
        any("text_positions.max_kl=" in r and f"at index {txt_spike_idx}" in r for r in fwd_idx_rec["failure_reasons"]),
        msg=f"Expected text_positions failure at index {txt_spike_idx}, got {fwd_idx_rec['failure_reasons']}",
    )
    self.assertFalse(
        any(f"at index {img_spike_idx}" in r for r in fwd_idx_rec["failure_reasons"]),
        msg=f"Unexpected image index {img_spike_idx} in text_positions failure_reasons: {fwd_idx_rec['failure_reasons']}",
    )

  def test_logit_runner_text_gates_fail_closed_and_long_text_deferral(self) -> None:
    """Tests fail-closed short/long text gates from `logit_runner` and deferred long-text replay."""
    self.assertEqual(logit_runner.SHORT_TEXT_PROMPT_MAX_TOKENS, 16)

    num_devices = len(jax.devices())
    fsdp_par = 2 if num_devices >= 2 else 1
    ep_par = 2 if num_devices >= 4 else 1
    tok_kw = (
        {}
        if HAS_TORCH
        else {
            "tokenizer_path": mm_testing.OFFLINE_TOKENIZER_PATH,
            "ds41_engram_compressed_vocab_size": 0,
        }
    )
    cfg_512 = mm_testing.mini_vision_maxtext_config(
        scan_layers=False,
        dtype="float32",
        weight_dtype="bfloat16",
        dtype_mm="float32",
        matmul_precision="highest",
        remat_policy="none",
        ds41_activation_quant=False,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=fsdp_par,
        ici_expert_parallelism=ep_par,
        max_target_length=512,
        **tok_kw,
    )
    mt_512 = model_creation_utils.from_config(
        cfg_512,
        mesh=self.mesh,
        rngs=nnx.Rngs(params=0),
        model_mode=MODEL_MODE_TRAIN,
    )
    if HAS_TORCH:
      mm_testing.copy_full_multimodal_model(mt_512, self.ref_mm, cfg_512, jnp.bfloat16)
    _reshard_model_to_abstract_mesh(mt_512, cfg_512, self.mesh)

    img_tok_id = int(cfg_512.ds41_image_token_id)
    vocab_size = int(cfg_512.vocab_size)
    rng = np.random.default_rng(20261025)

    def _make_text_tokens(seq_len: int) -> np.ndarray:
      raw = rng.integers(3, min(vocab_size, 2000), size=(seq_len,), dtype=np.int32)
      return np.where(raw == img_tok_id, 5, raw).astype(np.int32)

    def _forward_text_logits(tokens_1d: np.ndarray) -> np.ndarray:
      ids, seg, pos, _, s_len, _, _, _ = logit_runner.get_data_with_masks(
          {
              "prompt": "text_case",
              "tokens": tokens_1d.tolist(),
              "logits": np.zeros((int(tokens_1d.shape[0]), vocab_size), dtype=np.float32),
          },
          cfg_512,
      )
      return np.asarray(
          mt_512(
              decoder_input_tokens=jnp.asarray(ids, dtype=jnp.int32),
              decoder_positions=jnp.asarray(pos, dtype=jnp.int32),
              decoder_segment_ids=jnp.asarray(seg, dtype=jnp.int32),
              encoder_images=None,
              encoder_image_masks=None,
              enable_dropout=False,
          )[0, :s_len, :],
          dtype=np.float32,
      )

    def _perturb_row_exact_kl(train_row: np.ndarray, target_kl: float) -> np.ndarray:
      orig = np.asarray(train_row, dtype=np.float32)
      if target_kl <= 0.0:
        return orig.copy()
      argmin_idx = int(np.argmin(orig))
      orig_j = jnp.asarray(orig[None, :], dtype=jnp.float32)
      lo, hi = 0.0, 45.0
      best = orig.copy()
      for _ in range(60):
        mid = 0.5 * (lo + hi)
        cand = orig.copy()
        cand[argmin_idx] = np.float32(float(orig[argmin_idx]) + mid)
        kl_val = float(
            forward_pass_logit_checker._per_token_kl_2d(  # pylint: disable=protected-access
                jnp.asarray(cand[None, :], dtype=jnp.float32),
                orig_j,
            )[0]
        )
        best = cand
        if kl_val < target_kl:
          lo = mid
        else:
          hi = mid
      return best

    def _swap_last_token_rank10_min_kl(train_row: np.ndarray) -> np.ndarray:
      cand = np.asarray(train_row, dtype=np.float32).copy()
      order = np.argsort(-cand, kind="stable")
      i9 = int(order[9])
      i10 = int(order[10])
      mid = np.float32(0.5 * (float(cand[i9]) + float(cand[i10])))
      cand[i10] = np.nextafter(mid, np.float32(np.inf))
      cand[i9] = np.nextafter(mid, np.float32(-np.inf))
      return cand

    gates_root = self.tmp_root / "text_gates"
    gates_root.mkdir(parents=True, exist_ok=True)

    with open(_THRESHOLDS_JSON_PATH, "r", encoding="utf-8") as f:
      in_tree_thresholds = json.load(f)

    # Also stage a 1e-5 short_text_prompts thresholds config to verify the 1.5e-5 > 1e-5 boundary directly alongside
    # in-tree 2e-7.
    short_1e5_thresholds = copy.deepcopy(in_tree_thresholds)
    short_1e5_thresholds["forward_logit_checker"]["short_text_prompts"]["max_kl"] = 1e-5
    short_1e5_path = gates_root / "thresholds_short_1e5.json"
    with open(short_1e5_path, "w", encoding="utf-8") as f:
      json.dump(short_1e5_thresholds, f, indent=2)

    def _run_text_pkl(
        case_name: str,
        entries: list[dict[str, Any]],
        *,
        thresholds_path: Path = _THRESHOLDS_JSON_PATH,
        mm_golden_path: str = "",
        enable_long_text_gate: bool = True,
    ) -> tuple[Path, dict[str, Any]]:
      case_dir = gates_root / case_name
      case_dir.mkdir(parents=True, exist_ok=True)
      pkl_path = case_dir / "text.pkl"
      with open(pkl_path, "wb") as pf:
        pickle.dump(entries, pf)
      summary_path = case_dir / "summary.json"
      args_ns = argparse.Namespace(
          atol=None,
          rtol=1e-5,
          token_size=None,
          max_kl_div=None,
          golden_logits_path=pkl_path.as_posix(),
          mm_golden_logits_path=mm_golden_path,
          thresholds_json=thresholds_path.as_posix(),
          enable_long_text_gate=enable_long_text_gate,
          max_images_per_sample=4,
          run_ds41_mm_audit=False,
          hf_checkpoint_dir_for_audit="",
          export_orbax_dir="",
          summary_json=summary_path.as_posix(),
          hf_model_path="",
          run_hf_model=False,
          output_logits_path="",
          gcs_output_logits_path="",
          clip_logits_epsilon=None,
          skip_first_token=False,
          ckpt_type="nnx",
          trust_remote_code=True,
          reference_model="",
      )
      res = logit_runner.main(cfg_512, args_ns, preloaded_model=mt_512)
      return summary_path, res

    # 1. short_text_prompts (`len <= 16` against unmodified in-tree `thresholds.json`: `len = 8` and `len = 16`),
    # boundary `len = 17` -> `long_text_prompts`, and custom `"short_text_max_seq_len": 32` (`len = 32` ->
    # `short_text_prompts`, `33` -> `long_text_prompts`).
    tok_8 = _make_text_tokens(8)
    logits_8 = _forward_text_logits(tok_8)
    tok_16 = _make_text_tokens(16)
    logits_16 = _forward_text_logits(tok_16)
    tok_17 = _make_text_tokens(17)
    logits_17 = _forward_text_logits(tok_17)

    short_pass_8 = logits_8.copy()
    short_pass_8[3] = _perturb_row_exact_kl(logits_8[3], 6.9e-8)
    short_pass_16 = logits_16.copy()
    short_pass_16[10] = _perturb_row_exact_kl(logits_16[10], 6.9e-8)
    long_pass_17 = logits_17.copy()
    long_pass_17[5] = _perturb_row_exact_kl(logits_17[5], 1.5e-5)
    _, short_ok_summary = _run_text_pkl(
        "short_pass_and_17_boundary",
        [
            {"name": "sample_0", "prompt": "short 8", "tokens": tok_8.tolist(), "logits": short_pass_8},
            {"name": "sample_1", "prompt": "short 16", "tokens": tok_16.tolist(), "logits": short_pass_16},
            {"name": "sample_2", "prompt": "boundary 17", "tokens": tok_17.tolist(), "logits": long_pass_17},
        ],
    )
    self.assertTrue(short_ok_summary["passed"])
    self.assertTrue(short_ok_summary["text_gates_passed"])
    self.assertEqual(len(short_ok_summary["text_samples"]), 3)
    for idx_s, exp_len in ((0, 8), (1, 16)):
      t_entry = short_ok_summary["text_samples"][idx_s]
      self.assertEqual(t_entry["gate_id"], "short_text_prompts")
      self.assertEqual(t_entry["true_length"], exp_len)
      self.assertAlmostEqual(t_entry["max_kl"], 6.9e-8, delta=5e-9)
      self.assertEqual(t_entry["last_token_top10_matches"], 10)
      self.assertEqual(t_entry["required_top10_matches"], 10)
      self.assertTrue(t_entry["passed"])
    b17_entry = short_ok_summary["text_samples"][2]
    self.assertEqual(b17_entry["gate_id"], "long_text_prompts")
    self.assertEqual(b17_entry["true_length"], 17)
    self.assertEqual(b17_entry["max_kl_limit"], 0.0212)
    self.assertTrue(b17_entry["passed"])

    # Verify custom `"short_text_max_seq_len": 32` routes `len = 32` to `short_text_prompts` and `len = 33` to
    # `long_text_prompts`,
    # and omitting `"short_text_max_seq_len"` defaults to `SHORT_TEXT_PROMPT_MAX_TOKENS` (`16`).
    custom_32_cfg = copy.deepcopy(in_tree_thresholds)
    custom_32_cfg["forward_logit_checker"]["short_text_max_seq_len"] = 32
    custom_32_path = gates_root / "thresholds_short_max_32.json"
    with open(custom_32_path, "w", encoding="utf-8") as f:
      json.dump(custom_32_cfg, f, indent=2)
    tok_32 = _make_text_tokens(32)
    logits_32 = _forward_text_logits(tok_32)
    tok_33 = _make_text_tokens(33)
    logits_33 = _forward_text_logits(tok_33)
    _, custom_32_summary = _run_text_pkl(
        "custom_short_max_32",
        [
            {"name": "s32", "prompt": "len 32", "tokens": tok_32.tolist(), "logits": logits_32},
            {"name": "s33", "prompt": "len 33", "tokens": tok_33.tolist(), "logits": logits_33},
        ],
        thresholds_path=custom_32_path,
    )
    self.assertTrue(custom_32_summary["passed"])
    self.assertEqual(custom_32_summary["text_samples"][0]["gate_id"], "short_text_prompts")
    self.assertEqual(custom_32_summary["text_samples"][0]["true_length"], 32)
    self.assertEqual(custom_32_summary["text_samples"][1]["gate_id"], "long_text_prompts")
    self.assertEqual(custom_32_summary["text_samples"][1]["true_length"], 33)

    default_max_cfg = copy.deepcopy(in_tree_thresholds)
    del default_max_cfg["forward_logit_checker"]["short_text_max_seq_len"]
    g16, _ = logit_runner.select_forward_threshold_spec(
        default_max_cfg, sample_name="s16", seq_len=16, is_mm_sample=False
    )
    g17, _ = logit_runner.select_forward_threshold_spec(
        default_max_cfg, sample_name="s17", seq_len=17, is_mm_sample=False
    )
    self.assertEqual(g16, "short_text_prompts")
    self.assertEqual(g17, "long_text_prompts")

    # short_text_prompts fail closed on max_kl = 1.5e-5 (with top10 = 10) against both in-tree 2e-7 and 1e-5 configs.
    short_bad_kl = logits_8.copy()
    short_bad_kl[4] = _perturb_row_exact_kl(logits_8[4], 1.5e-5)
    for cfg_label, cfg_path in (("in_tree_2e7", _THRESHOLDS_JSON_PATH), ("short_1e5", short_1e5_path)):
      with self.subTest(short_kl_fail=cfg_label):
        s_path = gates_root / f"short_fail_kl_{cfg_label}" / "summary.json"
        with self.assertRaises(AssertionError):
          _run_text_pkl(
              f"short_fail_kl_{cfg_label}",
              [{"name": "sample_0", "prompt": "short 8", "tokens": tok_8.tolist(), "logits": short_bad_kl}],
              thresholds_path=cfg_path,
          )
        with open(s_path, "r", encoding="utf-8") as f:
          failed_short_kl = json.load(f)
        self.assertFalse(failed_short_kl["passed"])
        self.assertFalse(failed_short_kl["text_gates_passed"])
        self.assertEqual(failed_short_kl["text_samples"][0]["gate_id"], "short_text_prompts")
        self.assertEqual(failed_short_kl["text_samples"][0]["last_token_top10_matches"], 10)
        self.assertFalse(failed_short_kl["text_samples"][0]["passed"])
        self.assertFalse(failed_short_kl["samples"][0]["all_positions_gates"]["max_kl_pass"])
        self.assertTrue(failed_short_kl["samples"][0]["all_positions_gates"]["last_token_top10_matches_pass"])

    # short_text_prompts fail closed on top10 = 9 with exact max_kl == 0.0 (against in-tree 2e-7) and min-KL swap
    # (against 1e-5).
    eval_zero_kl_top9 = logit_runner.evaluate_thresholds(
        {
            "max_kl": 0.0,
            "mean_kl": 0.0,
            "last_token_top10_matches": 9,
            "last_token_top_k": 10,
        },
        in_tree_thresholds["forward_logit_checker"]["short_text_prompts"],
        check_last_token_top10=True,
    )
    self.assertTrue(eval_zero_kl_top9["max_kl_pass"])
    self.assertFalse(eval_zero_kl_top9["last_token_top10_matches_pass"])

    short_bad_top10 = logits_8.copy()
    short_bad_top10[-1] = _swap_last_token_rank10_min_kl(logits_8[-1])
    with self.assertRaisesRegex(AssertionError, "short_text_prompts"):
      _run_text_pkl(
          "short_fail_top10_zero_kl",
          [
              {
                  "name": "sample_0",
                  "prompt": "short 8",
                  "tokens": tok_8.tolist(),
                  "logits": short_bad_top10,
                  "tie_flip_logits": logits_8,
              }
          ],
          thresholds_path=_THRESHOLDS_JSON_PATH,
      )
    with open(gates_root / "short_fail_top10_zero_kl" / "summary.json", "r", encoding="utf-8") as f:
      failed_short_zero_kl = json.load(f)
    self.assertFalse(failed_short_zero_kl["passed"])
    self.assertEqual(failed_short_zero_kl["text_samples"][0]["gate_id"], "short_text_prompts")
    self.assertAlmostEqual(failed_short_zero_kl["text_samples"][0]["max_kl"], 0.0, delta=1e-11)
    self.assertEqual(failed_short_zero_kl["text_samples"][0]["last_token_top10_matches"], 9)
    self.assertTrue(failed_short_zero_kl["samples"][0]["all_positions_gates"]["max_kl_pass"])
    self.assertFalse(failed_short_zero_kl["samples"][0]["all_positions_gates"]["last_token_top10_matches_pass"])

    with self.assertRaisesRegex(AssertionError, "short_text_prompts"):
      _run_text_pkl(
          "short_fail_top10",
          [{"name": "sample_0", "prompt": "short 8", "tokens": tok_8.tolist(), "logits": short_bad_top10}],
          thresholds_path=short_1e5_path,
      )
    with open(gates_root / "short_fail_top10" / "summary.json", "r", encoding="utf-8") as f:
      failed_short_top10 = json.load(f)
    self.assertFalse(failed_short_top10["passed"])
    self.assertFalse(failed_short_top10["text_gates_passed"])
    self.assertEqual(failed_short_top10["text_samples"][0]["gate_id"], "short_text_prompts")
    self.assertEqual(failed_short_top10["text_samples"][0]["last_token_top10_matches"], 9)
    self.assertTrue(failed_short_top10["samples"][0]["all_positions_gates"]["max_kl_pass"])
    self.assertFalse(failed_short_top10["samples"][0]["all_positions_gates"]["last_token_top10_matches_pass"])

    # 2. long_text_prompts (`len = 462 > 16`) pass + exact 4-prompt replay + independent failures.
    tok_462 = _make_text_tokens(462)
    logits_462 = _forward_text_logits(tok_462)

    # (2a) long_text_prompts pass at `max_kl = 0.02115` at index 448, `top10 = 10`.
    long_pass_462 = logits_462.copy()
    long_pass_462[448] = _perturb_row_exact_kl(logits_462[448], 0.02115)
    _, long_ok_summary = _run_text_pkl(
        "long_pass_462",
        [{"name": "sample_3", "prompt": "Lyon 462", "tokens": tok_462.tolist(), "logits": long_pass_462}],
    )
    self.assertTrue(long_ok_summary["passed"])
    self.assertTrue(long_ok_summary["text_gates_passed"])
    long_ok_rec = long_ok_summary["text_samples"][0]
    self.assertEqual(long_ok_rec["gate_id"], "long_text_prompts")
    self.assertEqual(long_ok_rec["true_length"], 462)
    self.assertEqual(long_ok_rec["max_kl_index"], 448)
    self.assertAlmostEqual(long_ok_rec["max_kl"], 0.02115, delta=1e-5)
    self.assertEqual(long_ok_rec["max_kl_limit"], 0.0212)
    self.assertEqual(long_ok_rec["last_token_top10_matches"], 10)
    self.assertEqual(long_ok_rec["required_top10_matches"], 10)
    self.assertTrue(long_ok_rec["passed"])

    # (2b) Exact 4-prompt replay (`len = 462`, `max_kl = 0.02196408063173294` at index 448, `top10 = 9`),
    # combined with a passing `short_text_prompts` sample AND a passing multimodal `L1` (`mm_short_prompts`) sample to
    # prove ALL
    #      samples are evaluated and written to `summary.json` before `main()` raises `AssertionError`.
    mm_batch = multimodal.build_mini_vl_batch(
        self.ref_args,
        grid=multimodal.MINI_GRID_PADDED,
        batch_size=1,
        span_starts=((2,),),
        seed=88,
    )
    mm_tokens = mm_batch.input_ids.numpy().astype(np.int32)[0]
    mm_valid_patch = mm_batch.patches_bnpd.numpy().astype(np.float32)[0]
    mm_tt = mm_batch.token_types.numpy().astype(np.int32)[0]
    mm_padded_patches = np.zeros((4, mm_valid_patch.shape[1], mm_valid_patch.shape[2]), dtype=np.float32)
    mm_padded_patches[0] = mm_valid_patch[0]
    mm_mask = np.array([1, 0, 0, 0], dtype=np.int32)
    mm_ids, mm_seg, mm_pos, _, mm_seq_len, mm_imgs, mm_img_masks, _ = logit_runner.get_data_with_masks(
        {
            "sample_id": "L1",
            "tokens": mm_tokens.tolist(),
            "logits": np.zeros((int(mm_tokens.shape[0]), vocab_size), dtype=np.float32),
            "patches": mm_padded_patches,
            "image_masks": mm_mask,
            "token_types": mm_tt,
            "num_images": 1,
        },
        cfg_512,
        max_images_per_sample=4,
    )
    mm_logits_clean = np.asarray(
        mt_512(
            decoder_input_tokens=jnp.asarray(mm_ids, dtype=jnp.int32),
            decoder_positions=jnp.asarray(mm_pos, dtype=jnp.int32),
            decoder_segment_ids=jnp.asarray(mm_seg, dtype=jnp.int32),
            encoder_images=jnp.asarray(mm_imgs, dtype=jnp.float32),
            encoder_image_masks=jnp.asarray(mm_img_masks, dtype=jnp.int32),
            enable_dropout=False,
        )[0, :mm_seq_len, :],
        dtype=np.float32,
    )
    mm_pkl_path = gates_root / "l1_clean.pkl"
    with open(mm_pkl_path, "wb") as pf:
      pickle.dump(
          [
              {
                  "sample_id": "L1",
                  "prompt": "L1 clean",
                  "tokens": mm_tokens.tolist(),
                  "logits": mm_logits_clean,
                  "token_types": mm_tt,
                  "encoder_images": mm_padded_patches,
                  "encoder_image_masks": mm_mask,
                  "num_images": 1,
              }
          ],
          pf,
      )

    replay_sample3 = logits_462.copy()
    replay_sample3[448] = _perturb_row_exact_kl(logits_462[448], 0.02196408063173294)
    replay_sample3[-1] = _swap_last_token_rank10_min_kl(logits_462[-1])
    with self.assertRaisesRegex(AssertionError, "long_text_prompts"):
      _run_text_pkl(
          "replay",
          [
              {"name": "sample_0", "prompt": "I love to", "tokens": tok_8.tolist(), "logits": short_pass_8},
              {"name": "sample_3", "prompt": "Lyon 462", "tokens": tok_462.tolist(), "logits": replay_sample3},
          ],
          mm_golden_path=mm_pkl_path.as_posix(),
      )
    with open(gates_root / "replay" / "summary.json", "r", encoding="utf-8") as f:
      replay_summary = json.load(f)
    self.assertFalse(replay_summary["passed"])
    self.assertFalse(replay_summary["all_passed"])
    self.assertFalse(replay_summary["text_gates_passed"])
    self.assertTrue(replay_summary["mm_gates_passed"])
    self.assertEqual(len(replay_summary["samples"]), 3)
    self.assertTrue(replay_summary["samples"][0]["passed"])
    self.assertFalse(replay_summary["samples"][1]["passed"])
    self.assertTrue(replay_summary["samples"][2]["passed"])
    s3_rec = replay_summary["text_samples"][1]
    self.assertEqual(s3_rec["name"], "sample_3")
    self.assertEqual(s3_rec["gate_id"], "long_text_prompts")
    self.assertEqual(s3_rec["true_length"], 462)
    self.assertEqual(s3_rec["max_kl_index"], 448)
    self.assertAlmostEqual(s3_rec["max_kl"], 0.02196408063173294, delta=1e-5)
    self.assertEqual(s3_rec["last_token_top10_matches"], 9)
    self.assertFalse(s3_rec["passed"])

    # (2c) Independent long_text_prompts failure on `max_kl = 0.02196, top10 = 10`.
    long_bad_kl_only = logits_462.copy()
    long_bad_kl_only[448] = _perturb_row_exact_kl(logits_462[448], 0.02196)
    with self.assertRaisesRegex(AssertionError, "long_text_prompts"):
      _run_text_pkl(
          "long_fail_kl_only",
          [{"name": "sample_3", "prompt": "Lyon 462", "tokens": tok_462.tolist(), "logits": long_bad_kl_only}],
      )
    with open(gates_root / "long_fail_kl_only" / "summary.json", "r", encoding="utf-8") as f:
      s_kl_only = json.load(f)
    self.assertFalse(s_kl_only["passed"])
    self.assertFalse(s_kl_only["text_gates_passed"])
    self.assertEqual(s_kl_only["text_samples"][0]["gate_id"], "long_text_prompts")
    self.assertEqual(s_kl_only["text_samples"][0]["last_token_top10_matches"], 10)
    self.assertFalse(s_kl_only["samples"][0]["all_positions_gates"]["max_kl_pass"])
    self.assertTrue(s_kl_only["samples"][0]["all_positions_gates"]["last_token_top10_matches_pass"])

    # (2d) Independent long_text_prompts failure on `max_kl = 0.010, top10 = 9`.
    long_bad_top10_only = logits_462.copy()
    long_bad_top10_only[448] = _perturb_row_exact_kl(logits_462[448], 0.010)
    long_bad_top10_only[-1] = _swap_last_token_rank10_min_kl(logits_462[-1])
    with self.assertRaisesRegex(AssertionError, "long_text_prompts"):
      _run_text_pkl(
          "long_fail_top10_only",
          [{"name": "sample_3", "prompt": "Lyon 462", "tokens": tok_462.tolist(), "logits": long_bad_top10_only}],
      )
    with open(gates_root / "long_fail_top10_only" / "summary.json", "r", encoding="utf-8") as f:
      s_top10_only = json.load(f)
    self.assertFalse(s_top10_only["passed"])
    self.assertFalse(s_top10_only["text_gates_passed"])
    self.assertEqual(s_top10_only["text_samples"][0]["gate_id"], "long_text_prompts")
    self.assertEqual(s_top10_only["text_samples"][0]["last_token_top10_matches"], 9)
    self.assertTrue(s_top10_only["samples"][0]["all_positions_gates"]["max_kl_pass"])
    self.assertFalse(s_top10_only["samples"][0]["all_positions_gates"]["last_token_top10_matches_pass"])

    # (2e) Independent long_text_prompts failure on `mean_kl > 0.0001` (`max_kl = 0.015 <= 0.0212`, `top10 = 10`).
    long_bad_mean_kl = logits_462.copy()
    perturbed_row_015 = _perturb_row_exact_kl(logits_462[0], 0.015)
    long_bad_mean_kl[0] = perturbed_row_015
    for idx_m in range(1, 8):
      long_bad_mean_kl[idx_m] = _perturb_row_exact_kl(logits_462[idx_m], 0.015)
    with self.assertRaisesRegex(AssertionError, "long_text_prompts"):
      _run_text_pkl(
          "long_fail_mean_kl_only",
          [{"name": "sample_3", "prompt": "Lyon 462", "tokens": tok_462.tolist(), "logits": long_bad_mean_kl}],
      )
    with open(gates_root / "long_fail_mean_kl_only" / "summary.json", "r", encoding="utf-8") as f:
      s_mean_only = json.load(f)
    self.assertFalse(s_mean_only["passed"])
    self.assertTrue(s_mean_only["samples"][0]["all_positions_gates"]["max_kl_pass"])
    self.assertFalse(s_mean_only["samples"][0]["all_positions_gates"]["mean_kl_pass"])
    self.assertTrue(s_mean_only["samples"][0]["all_positions_gates"]["last_token_top10_matches_pass"])

    # 3. Missing / malformed `forward_logit_checker` in `thresholds_json` when text samples are present -> ValueError.
    malformed_cases: list[tuple[str, dict[str, Any], str]] = []

    no_flc = copy.deepcopy(in_tree_thresholds)
    del no_flc["forward_logit_checker"]
    malformed_cases.append(("missing_flc", no_flc, "forward_logit_checker"))

    no_short = copy.deepcopy(in_tree_thresholds)
    del no_short["forward_logit_checker"]["short_text_prompts"]
    malformed_cases.append(("missing_short", no_short, "short_text_prompts"))

    no_long = copy.deepcopy(in_tree_thresholds)
    del no_long["forward_logit_checker"]["long_text_prompts"]
    malformed_cases.append(("missing_long", no_long, "long_text_prompts"))

    zero_kl = copy.deepcopy(in_tree_thresholds)
    zero_kl["forward_logit_checker"]["short_text_prompts"]["max_kl"] = 0.0
    malformed_cases.append(("zero_max_kl", zero_kl, "max_kl"))

    neg_kl = copy.deepcopy(in_tree_thresholds)
    neg_kl["forward_logit_checker"]["long_text_prompts"]["max_kl"] = -0.01
    malformed_cases.append(("neg_max_kl", neg_kl, "max_kl"))

    bool_kl = copy.deepcopy(in_tree_thresholds)
    bool_kl["forward_logit_checker"]["short_text_prompts"]["max_kl"] = True
    malformed_cases.append(("bool_max_kl", bool_kl, "max_kl"))

    wrong_top10 = copy.deepcopy(in_tree_thresholds)
    wrong_top10["forward_logit_checker"]["long_text_prompts"]["last_token_top10_matches"] = 9
    malformed_cases.append(("top10_eq_9", wrong_top10, "last_token_top10_matches"))

    bool_top10 = copy.deepcopy(in_tree_thresholds)
    bool_top10["forward_logit_checker"]["short_text_prompts"]["last_token_top10_matches"] = True
    malformed_cases.append(("bool_top10", bool_top10, "last_token_top10_matches"))

    for bad_label, bad_short_max in (
        ("short_max_zero", 0),
        ("short_max_neg", -1),
        ("short_max_bool", True),
        ("short_max_float", 16.5),
        ("short_max_str", "16"),
    ):
      bad_sm_cfg = copy.deepcopy(in_tree_thresholds)
      bad_sm_cfg["forward_logit_checker"]["short_text_max_seq_len"] = bad_short_max
      malformed_cases.append((bad_label, bad_sm_cfg, "short_text_max_seq_len"))

    for bad_name, bad_cfg_dict, err_regex in malformed_cases:
      with self.subTest(malformed_thresholds=bad_name):
        bad_cfg_path = gates_root / f"bad_{bad_name}.json"
        with open(bad_cfg_path, "w", encoding="utf-8") as f:
          json.dump(bad_cfg_dict, f)
        with self.assertRaisesRegex(ValueError, err_regex):
          _run_text_pkl(
              f"malformed_{bad_name}",
              [{"name": "sample_0", "prompt": "short 8", "tokens": tok_8.tolist(), "logits": short_pass_8}],
              thresholds_path=bad_cfg_path,
          )

    # NaN `max_kl` tested directly on `_validate_forward_logit_checker_thresholds` (standard JSON has no NaN).
    nan_kl = copy.deepcopy(in_tree_thresholds)
    nan_kl["forward_logit_checker"]["long_text_prompts"]["max_kl"] = float("nan")
    with self.assertRaisesRegex(ValueError, "max_kl"):
      logit_runner.validate_forward_logit_checker_thresholds(nan_kl)

    # 4. Single-flag `long_text_prompts` deferral (`enable_long_text_gate=False` by default,
    # `enable_long_text_gate=True` re-enables).
    expected_deferred_status = logit_runner.DEFERRED_LONG_TEXT_STATUS

    tok_5 = _make_text_tokens(5)
    logits_5 = _forward_text_logits(tok_5)
    tok_12 = _make_text_tokens(12)
    logits_12 = _forward_text_logits(tok_12)
    short_pass_5 = logits_5.copy()
    short_pass_5[2] = _perturb_row_exact_kl(logits_5[2], 6.9e-8)
    short_pass_12 = logits_12.copy()
    short_pass_12[6] = _perturb_row_exact_kl(logits_12[6], 6.9e-8)

    replay_4samples = [
        {"name": "sample_0", "prompt": "I love to", "tokens": tok_5.tolist(), "logits": short_pass_5},
        {"name": "sample_1", "prompt": "Today is a", "tokens": tok_12.tolist(), "logits": short_pass_12},
        {"name": "sample_2", "prompt": "What is the", "tokens": tok_16.tolist(), "logits": short_pass_16},
        {"name": "sample_3", "prompt": "Lyon 462", "tokens": tok_462.tolist(), "logits": replay_sample3},
    ]
    replay_4pkl = gates_root / "replay_4samples.pkl"
    with open(replay_4pkl, "wb") as pf:
      pickle.dump(replay_4samples, pf)

    # (4a) Default `enable_long_text_gate=False` on 4-sample 4-prompt replay succeeds, defers `sample_3` without
    #      PASS/FAIL, and still gates `samples 0..2` (`short_text_prompts`) + multimodal `L1` (`mm_short_prompts`).
    with self.subTest(defer_mode="default_defer_replay"):
      defer_summary_path = gates_root / "defer_default_defer_summary.json"
      cli_default_args, _ = logit_runner.build_arg_parser().parse_known_args(
          [
              f"--golden_logits_path={replay_4pkl.as_posix()}",
              f"--mm_golden_logits_path={mm_pkl_path.as_posix()}",
              f"--thresholds_json={_THRESHOLDS_JSON_PATH.as_posix()}",
              f"--summary_json={defer_summary_path.as_posix()}",
              "--ckpt_type=nnx",
          ]
      )
      self.assertFalse(cli_default_args.enable_long_text_gate)
      stdout_buf = io.StringIO()
      with contextlib.redirect_stdout(stdout_buf):
        defer_summary = logit_runner.main(cfg_512, cli_default_args, preloaded_model=mt_512)
      stdout_txt = stdout_buf.getvalue()

      self.assertIn(expected_deferred_status, stdout_txt)
      self.assertIn("DS41_TEXT_GATE_SUMMARY:", stdout_txt)
      for forbidden in ("Test 3 Passed", "Test 3 Failed", "[long_text_prompts] PASS", "[long_text_prompts] FAIL"):
        self.assertNotIn(forbidden, stdout_txt)

      with open(defer_summary_path, "r", encoding="utf-8") as f:
        defer_disk = json.load(f)
      self.assertEqual(defer_summary, defer_disk)
      self.assertTrue(defer_disk["passed"])
      self.assertTrue(defer_disk["all_passed"])
      self.assertFalse(defer_disk["enable_long_text_gate"])
      self.assertEqual(defer_disk["long_text_gate_status"], expected_deferred_status)
      self.assertEqual(defer_disk["long_text_sample_3_status"], expected_deferred_status)
      self.assertEqual(defer_disk["num_samples"], 5)
      self.assertEqual(defer_disk["num_gated_samples"], 4)
      self.assertEqual(defer_disk["num_deferred_samples"], 1)
      self.assertEqual(defer_disk["num_passed"], 4)
      for idx_short in (0, 1, 2):
        short_rec = defer_disk["text_samples"][idx_short]
        self.assertEqual(short_rec["gate"], "short_text_prompts")
        self.assertEqual(short_rec["gate_id"], "short_text_prompts")
        self.assertTrue(short_rec["passed"])
        for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
          self.assertIn(kl_key, short_rec)
          self.assertIn(f"{kl_key}_legacy_fp32", short_rec)
      for top_kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl", "max_kl_div", "mean_kl_div"):
        self.assertIn(top_kl_key, defer_disk)
        self.assertIn(f"{top_kl_key}_legacy_fp32", defer_disk)
      for s_rep in defer_disk["samples"]:
        if "passed" in s_rep:
          for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
            self.assertIn(kl_key, s_rep)
            self.assertIn(f"{kl_key}_legacy_fp32", s_rep)
            self.assertIn(kl_key, s_rep["all_positions_metrics"])
            self.assertIn(f"{kl_key}_legacy_fp32", s_rep["all_positions_metrics"])
          if s_rep["text_positions_metrics"] is not None:
            for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
              self.assertIn(kl_key, s_rep["text_positions_metrics"])
              self.assertIn(f"{kl_key}_legacy_fp32", s_rep["text_positions_metrics"])
      for s3_container in (defer_disk["text_samples"][3], defer_disk["samples"][3]):
        self.assertEqual(s3_container["name"], "sample_3")
        self.assertEqual(s3_container["gate"], "long_text_prompts")
        self.assertEqual(s3_container["gate_id"], "long_text_prompts")
        self.assertEqual(s3_container["status"], expected_deferred_status)
        self.assertNotIn("passed", s3_container)
      self.assertEqual(defer_disk["samples"][4]["name"], "L1")
      self.assertTrue(defer_disk["samples"][4]["passed"])

    # (4b) Under default `enable_long_text_gate=False`, corrupting `short_text_prompts` (`samples 0..2`) still fails
    # closed (`AssertionError`).
    with self.subTest(defer_mode="default_tt1_still_fails_closed"):
      short_bad_25e8 = logits_5.copy()
      short_bad_25e8[1] = _perturb_row_exact_kl(logits_5[1], 2.5e-7)
      short_bad_15e6 = logits_12.copy()
      short_bad_15e6[3] = _perturb_row_exact_kl(logits_12[3], 1.5e-5)
      short_bad_top9_s2 = logits_16.copy()
      short_bad_top9_s2[-1] = _swap_last_token_rank10_min_kl(logits_16[-1])

      for bad_short_name, bad_idx, bad_entry, cfg_p in (
          (
              "s0_kl_25e8",
              0,
              {"name": "sample_0", "prompt": "I love to", "tokens": tok_5.tolist(), "logits": short_bad_25e8},
              _THRESHOLDS_JSON_PATH,
          ),
          (
              "s1_kl_15e6",
              1,
              {"name": "sample_1", "prompt": "Today is a", "tokens": tok_12.tolist(), "logits": short_bad_15e6},
              short_1e5_path,
          ),
          (
              "s2_top10_9",
              2,
              {
                  "name": "sample_2",
                  "prompt": "What is the",
                  "tokens": tok_16.tolist(),
                  "logits": short_bad_top9_s2,
                  "tie_flip_logits": logits_16,
              },
              _THRESHOLDS_JSON_PATH,
          ),
      ):
        corrupted_4 = copy.deepcopy(replay_4samples)
        corrupted_4[bad_idx] = bad_entry
        with self.assertRaisesRegex(AssertionError, "short_text_prompts"):
          _run_text_pkl(
              f"defer_default_short_fail_{bad_short_name}",
              corrupted_4,
              thresholds_path=cfg_p,
              enable_long_text_gate=False,
          )
        with open(gates_root / f"defer_default_short_fail_{bad_short_name}" / "summary.json", "r", encoding="utf-8") as f:
          bad_short_disk = json.load(f)
        self.assertFalse(bad_short_disk["passed"])
        self.assertEqual(bad_short_disk["long_text_gate_status"], expected_deferred_status)
        self.assertNotIn("passed", bad_short_disk["text_samples"][3])

    # (4c) Under default `enable_long_text_gate=False`, corrupting multimodal `L1..L4`
    # (`mm_short_prompts`/`mm_long_prompts` KL, Top-1, image_masks)
    #      still fails closed.
    with self.subTest(defer_mode="default_mm_l1_l4_still_fails_closed"):

      def _corrupt_top1_low_kl(clean_2d: np.ndarray, num_flips: int) -> np.ndarray:
        cand = clean_2d.copy()
        sorted_vals = -np.sort(-cand, axis=-1)
        gaps = sorted_vals[:, 0] - sorted_vals[:, 1]
        flip_positions = np.argsort(gaps)[:num_flips]
        for pos_i in flip_positions:
          row_order = np.argsort(-cand[pos_i], kind="stable")
          c0, c1 = int(row_order[0]), int(row_order[1])
          mid_v = np.float32(0.5 * (float(cand[pos_i, c0]) + float(cand[pos_i, c1])))
          cand[pos_i, c1] = np.nextafter(mid_v, np.float32(np.inf))
          cand[pos_i, c0] = np.nextafter(mid_v, np.float32(-np.inf))
        return cand

      mm_bad_kl_l1 = mm_logits_clean.copy()
      mm_bad_kl_l1[0, 0] += 25.0
      mm_bad_kl_l4 = mm_logits_clean.copy()
      mm_bad_kl_l4[0, 0] += 25.0
      num_flips_needed = max(2, int(mm_logits_clean.shape[0] * 0.05) + 1)
      mm_bad_top1 = _corrupt_top1_low_kl(mm_logits_clean, num_flips_needed)

      for mm_case_label, sample_id_str, exp_tier, bad_mm_logits, tie_flip_mm in (
          ("l1_kl", "L1", "mm_short_prompts", mm_bad_kl_l1, None),
          ("l2_top1", "L2", "mm_short_prompts", mm_bad_top1, mm_logits_clean),
          ("l3_kl", "L3", "mm_short_prompts", mm_bad_kl_l1, None),
          ("l4_kl", "L4", "mm_long_prompts", mm_bad_kl_l4, None),
          ("l4_top1", "L4", "mm_long_prompts", mm_bad_top1, mm_logits_clean),
      ):
        bad_mm_pkl = gates_root / f"defer_mm_bad_{mm_case_label}.pkl"
        mm_rec_payload: dict[str, Any] = {
            "sample_id": sample_id_str,
            "prompt": f"{sample_id_str} corrupt",
            "tokens": mm_tokens.tolist(),
            "logits": bad_mm_logits,
            "token_types": mm_tt,
            "encoder_images": mm_padded_patches,
            "encoder_image_masks": mm_mask,
            "num_images": 1,
        }
        if tie_flip_mm is not None:
          mm_rec_payload["tie_flip_logits"] = tie_flip_mm
        with open(bad_mm_pkl, "wb") as pf:
          pickle.dump([mm_rec_payload], pf)
        with self.assertRaisesRegex(AssertionError, exp_tier):
          _run_text_pkl(
              f"defer_default_mm_fail_{mm_case_label}",
              replay_4samples,
              mm_golden_path=bad_mm_pkl.as_posix(),
              enable_long_text_gate=False,
          )

      bad_mask_pkl = gates_root / "defer_mm_bad_mask.pkl"
      with open(bad_mask_pkl, "wb") as pf:
        pickle.dump(
            [
                {
                    "sample_id": "L1",
                    "prompt": "L1 bad mask",
                    "tokens": mm_tokens.tolist(),
                    "logits": mm_logits_clean,
                    "token_types": mm_tt,
                    "encoder_images": mm_valid_patch,
                    "encoder_image_masks": np.array([0, 1, 0, 0], dtype=np.int32),
                    "num_images": 1,
                }
            ],
            pf,
        )
      with self.assertRaisesRegex(ValueError, "image_masks"):
        _run_text_pkl(
            "defer_default_mm_fail_mask",
            replay_4samples,
            mm_golden_path=bad_mask_pkl.as_posix(),
            enable_long_text_gate=False,
        )

    # (4d) Flipping `--enable_long_text_gate=true` immediately re-enables fail-closed `long_text_prompts` enforcement
    # on 4-prompt replay.
    with self.subTest(defer_mode="flip_enable_long_text_gate_true_fails_closed"):
      enabled_summary_path = gates_root / "defer_enabled_w1_t3_summary.json"
      cli_enabled_args, _ = logit_runner.build_arg_parser().parse_known_args(
          [
              f"--golden_logits_path={replay_4pkl.as_posix()}",
              f"--mm_golden_logits_path={mm_pkl_path.as_posix()}",
              f"--thresholds_json={_THRESHOLDS_JSON_PATH.as_posix()}",
              "--enable_long_text_gate=true",
              f"--summary_json={enabled_summary_path.as_posix()}",
              "--ckpt_type=nnx",
          ]
      )
      self.assertTrue(cli_enabled_args.enable_long_text_gate)
      with self.assertRaisesRegex(AssertionError, "long_text_prompts"):
        logit_runner.main(cfg_512, cli_enabled_args, preloaded_model=mt_512)
      with open(enabled_summary_path, "r", encoding="utf-8") as f:
        enabled_disk = json.load(f)
      self.assertFalse(enabled_disk["passed"])
      self.assertTrue(enabled_disk["enable_long_text_gate"])
      self.assertEqual(enabled_disk["long_text_gate_status"], "failed")
      self.assertEqual(enabled_disk["long_text_sample_3_status"], "failed")
      s3_enabled = enabled_disk["text_samples"][3]
      self.assertEqual(s3_enabled["gate"], "long_text_prompts")
      self.assertAlmostEqual(s3_enabled["max_kl"], 0.02196408063173294, delta=1e-5)
      self.assertEqual(s3_enabled["last_token_top10_matches"], 9)
      self.assertFalse(s3_enabled["passed"])

    # (4e) Verify `2_test_deepseek41_mm.sh` (`STAGES=scanned_logits`) wires `ENABLE_LONG_TEXT_GATE` and writes
    # `scanned_logits_summary.json`.
    with self.subTest(defer_mode="stage_script_scanned_logits_and_flag_wiring"):
      sh_dir = gates_root / "stage_script_defer"
      shim_bin = sh_dir / "bin"
      shim_bin.mkdir(parents=True, exist_ok=True)
      calls_jsonl = sh_dir / "calls.jsonl"
      shim_py = shim_bin / "python3"
      shim_code = f"""#!{sys.executable}
import json
from pathlib import Path
import sys

with open({calls_jsonl.as_posix()!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

if any("logit_runner" in a for a in sys.argv[1:]):
  enable_flag = any(a == "--enable_long_text_gate=true" for a in sys.argv[1:])
  status_val = "passed" if enable_flag else {expected_deferred_status!r}
  for arg in sys.argv[2:]:
    if arg.startswith("--summary_json="):
      out_p = Path(arg.split("=", 1)[1])
      out_p.parent.mkdir(parents=True, exist_ok=True)
      out_p.write_text(
          json.dumps(
              {{
                  "passed": True,
                  "all_passed": True,
                  "enable_long_text_gate": enable_flag,
                  "long_text_gate_status": status_val,
                  "long_text_sample_3_status": status_val,
              }}
          ),
          encoding="utf-8",
      )
sys.exit(0)
"""
      shim_py.write_text(shim_code, encoding="utf-8")
      os.chmod(shim_py, stat.S_IRWXU)

      mm_stub = sh_dir / "mm_stub"
      mm_stub.mkdir(parents=True, exist_ok=True)
      out_default = sh_dir / "out_default"
      env_sh = os.environ.copy()
      if "ENABLE_LONG_TEXT_GATE" in env_sh:
        del env_sh["ENABLE_LONG_TEXT_GATE"]
      env_sh["PYTHON_BIN"] = shim_py.as_posix()
      env_sh["HF_CKPT_PATH"] = sh_dir.as_posix()
      env_sh["OUTPUT_DIR"] = out_default.as_posix()
      env_sh["GOLDEN_TEXT_PKL"] = replay_4pkl.as_posix()
      env_sh["GOLDEN_MM_DIR"] = mm_stub.as_posix()
      env_sh["STAGES"] = "scanned_logits"

      proc_def = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()], env=env_sh, capture_output=True, text=True, check=False
      )
      self.assertEqual(proc_def.returncode, 0, msg=f"stderr={proc_def.stderr}")
      sh_calls = [json.loads(line) for line in calls_jsonl.read_text(encoding="utf-8").splitlines()]
      checker_call = next(c for c in sh_calls if any("logit_runner" in a for a in c))
      self.assertIn("--enable_long_text_gate=false", checker_call)
      self.assertIn("max_target_length=1024", checker_call)
      scanned_summary_data = json.loads((out_default / "scanned_logits_summary.json").read_text(encoding="utf-8"))
      self.assertEqual(scanned_summary_data["long_text_sample_3_status"], expected_deferred_status)
      self.assertEqual(scanned_summary_data["long_text_gate_status"], expected_deferred_status)

      calls_jsonl.unlink()
      env_on = dict(env_sh)
      env_on["ENABLE_LONG_TEXT_GATE"] = "1"
      env_on["OUTPUT_DIR"] = (sh_dir / "out_on").as_posix()
      proc_on = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()], env=env_on, capture_output=True, text=True, check=False
      )
      self.assertEqual(proc_on.returncode, 0, msg=f"stderr={proc_on.stderr}")
      sh_calls_on = [json.loads(line) for line in calls_jsonl.read_text(encoding="utf-8").splitlines()]
      checker_call_on = next(c for c in sh_calls_on if any("logit_runner" in a for a in c))
      self.assertIn("--enable_long_text_gate=true", checker_call_on)

      env_bad = dict(env_sh)
      env_bad["ENABLE_LONG_TEXT_GATE"] = "2"
      proc_bad = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()], env=env_bad, capture_output=True, text=True, check=False
      )
      self.assertNotEqual(proc_bad.returncode, 0)
      self.assertIn("ENABLE_LONG_TEXT_GATE", proc_bad.stderr)

  def test_masked_prefill_runner_mm_and_cossim_negative_control(self) -> None:
    """Tests `masked_prefill_runner` multimodal `.npz`/`mm_logit.pkl` verification and negative controls."""
    batch = multimodal.build_mini_vl_batch(
        self.ref_args,
        grid=multimodal.MINI_GRID_PADDED,
        batch_size=1,
        span_starts=((2,),),
        seed=55,
    )
    tokens_np = batch.input_ids.numpy().astype(np.int32)[0]
    valid_patch_0 = batch.patches_bnpd.numpy().astype(np.float32)[0]  # [1, 35, 588]
    img_tok_id = int(self.cfg_unscanned.ds41_image_token_id)
    token_types = batch.token_types.numpy().astype(np.int32)[0]
    self.assertEqual(set(np.unique(token_types).tolist()), {-1, 0, 1, 2, 3})
    self.assertTrue(np.array_equal(token_types >= 0, tokens_np == img_tok_id))

    rng = np.random.default_rng(77)
    padded_patches = np.zeros((4, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32)
    padded_patches[0] = valid_patch_0[0]
    padded_patches[1:] = (
        rng.standard_normal((3, valid_patch_0.shape[1], valid_patch_0.shape[2])).astype(np.float32) * 20.0
    )
    mm_image_masks = np.array([1, 0, 0, 0], dtype=np.int32)

    ids_j = jnp.asarray(tokens_np[None, :], dtype=jnp.int32)
    pos_j = jnp.arange(tokens_np.shape[0], dtype=jnp.int32)[None, :]
    seg_j = jnp.ones_like(ids_j, dtype=jnp.int32)
    imgs_j = jnp.asarray(padded_patches[None, ...], dtype=jnp.float32)
    masks_j = jnp.asarray(mm_image_masks[None, :], dtype=jnp.int32)

    mm_logits = np.asarray(
        self.mt_unscanned(
            decoder_input_tokens=ids_j,
            decoder_positions=pos_j,
            decoder_segment_ids=seg_j,
            encoder_images=imgs_j,
            encoder_image_masks=masks_j,
            enable_dropout=False,
        )[0],
        dtype=np.float32,
    )

    mp_dir = self.tmp_root / "masked_prefill_mm"
    mp_dir.mkdir(parents=True, exist_ok=True)
    # Save padded-slot multimodal `.npz`: `patches` already padded to [4, P, 588], `image_masks=[1, 0, 0, 0]`,
    # `num_images=1`, `sample_id="M1"`, `grid_hw=[[6, 6]]`.
    np.savez(
        mp_dir / "M1.npz",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mm_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )

    loaded_samples = masked_prefill_runner.load_golden_dataset(mp_dir.as_posix())
    self.assertEqual(len(loaded_samples), 1)
    self.assertEqual(loaded_samples[0]["name"], "M1")
    self.assertFalse(bool(loaded_samples[0]["cos_sim_diagnostic_only"]))

    _, _, _, _, _, mp_imgs, mp_masks = masked_prefill_runner.get_data(
        loaded_samples[0], self.cfg_unscanned, max_images_per_sample=4
    )
    self.assertIsNotNone(mp_imgs)
    self.assertIsNotNone(mp_masks)
    self.assertTrue(np.all(np.asarray(mp_masks)[:, 0] == 1))
    self.assertTrue(np.all(np.asarray(mp_masks)[:, 1:] == 0))

    # Also verify padded-slot multimodal `mm_logit.pkl` entry (`encoder_images` [4, P, 588], `encoder_image_masks=[1,
    # 0, 0, 0]`,
    # `num_images=1`, `sample_id="L1"`) through `masked_prefill_runner.load_golden_dataset` & `get_data`.
    mp_pkl_path = self.tmp_root / "mp_mm_logit.pkl"
    with open(mp_pkl_path, "wb") as f:
      pickle.dump(
          [
              {
                  "sample_id": "L1",
                  "prompt": "L1 prompt",
                  "tokens": tokens_np.tolist(),
                  "logits": mm_logits,
                  "token_types": token_types,
                  "encoder_images": padded_patches,
                  "encoder_image_masks": mm_image_masks,
                  "pixel_values": padded_patches,
                  "grid_hw": np.array([[6, 6]], dtype=np.int32),
                  "num_images": 1,
              }
          ],
          f,
      )
    pkl_samples = masked_prefill_runner.load_golden_dataset(mp_pkl_path.as_posix())
    self.assertEqual(len(pkl_samples), 1)
    self.assertEqual(pkl_samples[0]["name"], "L1")
    _, _, _, _, _, pkl_mp_imgs, pkl_mp_masks = masked_prefill_runner.get_data(
        pkl_samples[0], self.cfg_unscanned, max_images_per_sample=4
    )
    self.assertIsNotNone(pkl_mp_imgs)
    self.assertIsNotNone(pkl_mp_masks)
    self.assertTrue(np.all(np.asarray(pkl_mp_masks)[:, 0] == 1))
    self.assertTrue(np.all(np.asarray(pkl_mp_masks)[:, 1:] == 0))

    mp_summary_json = mp_dir / "mp_summary.json"
    mp_args = argparse.Namespace(
        atol=None,
        min_cos_sim=0.999,
        token_size=None,
        max_kl_div=0.08,
        golden_logits_path="",
        mm_golden_dir=mp_dir.as_posix(),
        max_images_per_sample=4,
        output_json=mp_summary_json.as_posix(),
        hf_model_path="",
        run_hf_model=False,
        reference_model="",
        output_logits_path="",
        gcs_output_logits_path="",
        clip_logits_epsilon=None,
        enforce_length_bucket_gates=True,
        skip_first_token=False,
        ckpt_type="nnx",
        trust_remote_code=True,
    )
    mp_reports = masked_prefill_runner.main(
        self.cfg_unscanned,
        mp_args,
        preloaded_model=self.mt_unscanned,
    )
    expected_mp_img_tokens = int(np.sum(token_types >= 0))
    expected_mp_txt_tokens = int(np.sum(token_types == -1))
    self.assertGreater(expected_mp_img_tokens, int(np.sum(token_types == 1)))
    self.assertEqual(len(mp_reports), 1)
    self.assertTrue(mp_reports[0]["passed"])
    self.assertTrue(mp_reports[0]["is_multimodal"])
    self.assertEqual(mp_reports[0]["image_token_count"], expected_mp_img_tokens)
    self.assertEqual(mp_reports[0]["text_token_count"], expected_mp_txt_tokens)
    self.assertTrue(mp_reports[0]["metrics"]["rerun_bitwise_identical"])
    self.assertEqual(mp_reports[0]["metrics"]["rerun_max_abs_diff"], 0.0)
    self.assertTrue(mp_reports[0]["metrics"]["rerun_argmax_identical"])
    self.assertLessEqual(mp_reports[0]["metrics"]["mean_batch_diff"], 1e-4)
    self.assertIsNotNone(mp_reports[0]["text_positions_metrics"])
    self.assertEqual(mp_reports[0]["text_positions_metrics"]["seq_len"], expected_mp_txt_tokens)
    self.assertTrue(all(mp_reports[0]["text_positions_gates"].values()))
    self.assertIsNotNone(mp_reports[0]["image_positions_metrics"])
    self.assertEqual(mp_reports[0]["image_positions_metrics"]["seq_len"], expected_mp_img_tokens)
    self.assertIsNotNone(mp_reports[0]["all_positions_metrics"])
    self.assertEqual(
        mp_reports[0]["frozen_topk_diagnostic"],
        {"ran": False, "reason": "text_positions_passed"},
    )
    with open(mp_summary_json, "r", encoding="utf-8") as f:
      mp_summary_payload = json.load(f)
    self.assertTrue(mp_summary_payload["passed"])
    self.assertTrue(mp_summary_payload["all_passed"])
    self.assertEqual(mp_summary_payload["multimodal_summary"]["total_image_tokens"], expected_mp_img_tokens)
    for top_kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl", "max_kl_div", "mean_kl_div"):
      self.assertIn(top_kl_key, mp_summary_payload)
      self.assertIn(f"{top_kl_key}_legacy_fp32", mp_summary_payload)
    for kl_key in ("max_kl", "mean_kl", "p95_kl", "p99_kl"):
      self.assertIn(kl_key, mp_summary_payload["samples"][0])
      self.assertIn(f"{kl_key}_legacy_fp32", mp_summary_payload["samples"][0])
      self.assertIn(kl_key, mp_summary_payload["samples"][0]["metrics"])
      self.assertIn(f"{kl_key}_legacy_fp32", mp_summary_payload["samples"][0]["metrics"])
      self.assertIn(kl_key, mp_summary_payload["samples"][0]["text_positions_metrics"])
      self.assertIn(f"{kl_key}_legacy_fp32", mp_summary_payload["samples"][0]["text_positions_metrics"])

    with open(_THRESHOLDS_JSON_PATH, "r", encoding="utf-8") as f:
      mp_cfg = json.load(f)["masked_prefill"]
    self.assertTrue(mp_cfg["rerun_bitwise_identical"])
    self.assertTrue(mp_cfg["rerun_argmax_identical"])
    self.assertEqual(mp_cfg["mean_batch_diff"], 1e-4)
    self.assertNotIn("intra_shard_max_diff", mp_cfg)
    self.assertIn("intra_shard_max_diff", mp_cfg["diagnostic_only_fields"])
    self.assertIn("cross_shard_max_diff", mp_cfg["diagnostic_only_fields"])
    self.assertIn("max_batch_diff", mp_cfg["diagnostic_only_fields"])
    self.assertEqual(mp_cfg["p99_kl"], 0.002)
    self.assertEqual(
        set(mp_cfg["length_buckets"].keys()),
        set(masked_prefill_runner.LENGTH_BUCKET_THRESHOLDS.keys()),
    )
    for b_name, b_json in mp_cfg["length_buckets"].items():
      b_code = masked_prefill_runner.LENGTH_BUCKET_THRESHOLDS[b_name]
      self.assertEqual(b_json["max_seq_len"], b_code["max_seq_len"])
      self.assertEqual(b_json["max_kl"], b_code["max_kl"])
      self.assertEqual(b_json["mean_kl"], b_code["mean_kl"])
      expected_p99 = 0.005 if b_name == "2049 - 4096" else 0.002
      self.assertEqual(b_json["p99_kl"], expected_p99)
      self.assertEqual(b_json["p99_kl"], b_code["p99_kl"])
      self.assertEqual(b_json["top1_agreement"], b_code["top1_agreement"])
      self.assertEqual(b_code["p95_kl"], mp_cfg["p95_kl"])
      self.assertEqual(b_code["mean_cos_sim"], mp_cfg["mean_cos_sim"])

    # Negative control 1: cos_sim_diagnostic_only=True on a multimodal sample must raise ValueError.
    with self.assertRaises(ValueError):
      masked_prefill_runner.evaluate_sample_gates(
          parity_metrics=mp_reports[0]["metrics"],
          batch_det={
              "rerun_bitwise_identical": True,
              "rerun_argmax_identical": True,
              "mean_batch_diff": 0.0,
              "mean_diff_threshold": 1e-4,
          },
          bucket_gates=masked_prefill_runner.get_length_bucket_thresholds(int(tokens_np.shape[0])),
          effective_max_kl=0.025,
          effective_min_cos=0.999,
          cos_sim_diagnostic_only=True,
          is_multimodal=True,
      )

    # Also test via .npz with cos_sim_diagnostic_only=True in main().
    bad_diag_dir = self.tmp_root / "masked_prefill_mm_bad_diag"
    bad_diag_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        bad_diag_dir / "M1.npz",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mm_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
        cos_sim_diagnostic_only=np.array(True),
    )
    bad_diag_args = copy.copy(mp_args)
    bad_diag_args.mm_golden_dir = bad_diag_dir.as_posix()
    with self.assertRaises((ValueError, SystemExit)) as cm:
      masked_prefill_runner.main(
          self.cfg_unscanned,
          bad_diag_args,
          preloaded_model=self.mt_unscanned,
      )
    if isinstance(cm.exception, SystemExit):
      self.assertEqual(cm.exception.code, 42)

    # Negative control 2: rerun determinism & PR #4447 mean batch drift + cross-row spike diagnostic pass.
    clean_batch = np.broadcast_to(mm_logits[None, ...], (4, mm_logits.shape[0], mm_logits.shape[1])).copy()
    # (2a) 1-ULP perturbation on Run 2 fails rerun_bitwise_identical and passed.
    run2_ulp = clean_batch.copy()
    run2_ulp[1, 2, 3] = np.nextafter(run2_ulp[1, 2, 3], np.float32(1e6))
    det_ulp = masked_prefill_runner.check_batch_slot_determinism(
        clean_batch,
        int(tokens_np.shape[0]),
        int(mm_logits.shape[-1]),
        rows_per_batch_shard=2,
        rerun_train_logits=run2_ulp,
    )
    self.assertFalse(det_ulp["rerun_bitwise_identical"])
    self.assertFalse(det_ulp["passed"])

    # (2b) Row mean drift > 1e-4 fails even when reruns match bitwise.
    drifted_batch = clean_batch.copy()
    drifted_batch[1:] += 3e-4
    det_drift = masked_prefill_runner.check_batch_slot_determinism(
        drifted_batch,
        int(tokens_np.shape[0]),
        int(mm_logits.shape[-1]),
        rows_per_batch_shard=2,
        rerun_train_logits=drifted_batch.copy(),
    )
    self.assertTrue(det_drift["rerun_bitwise_identical"])
    self.assertGreater(det_drift["mean_batch_diff"], 1e-4)
    self.assertFalse(det_drift["passed"])

    # (2c) Single-element row spike with mean_batch_diff <= 1e-4 and bitwise-identical rerun passes.
    spiked_batch = clean_batch.copy()
    spiked_batch[1, 0, 0] += np.float32(0.4)
    det_spike_pass = masked_prefill_runner.check_batch_slot_determinism(
        spiked_batch,
        int(tokens_np.shape[0]),
        int(mm_logits.shape[-1]),
        rows_per_batch_shard=2,
        rerun_train_logits=spiked_batch.copy(),
    )
    self.assertTrue(det_spike_pass["rerun_bitwise_identical"])
    self.assertLessEqual(det_spike_pass["mean_batch_diff"], 1e-4)
    self.assertTrue(det_spike_pass["passed"])
    self.assertAlmostEqual(det_spike_pass["intra_shard_max_diff"], 0.4, places=5)

    # Negative control 3: sum(image_masks) != num_images raises ValueError.
    with self.assertRaises(ValueError):
      masked_prefill_runner.prepare_ds41_mm_batch_inputs(
          padded_patches,
          4,
          max_images_per_sample=4,
          image_masks=np.array([1, 1, 0, 0], dtype=np.int32),
          num_images=1,
      )
    with self.assertRaises(ValueError):
      masked_prefill_runner.get_data(
          {
              "sample_id": "M1",
              "tokens": tokens_np.tolist(),
              "logits": mm_logits,
              "patches": padded_patches,
              "image_masks": np.array([1, 1, 0, 0], dtype=np.int32),
              "num_images": 1,
          },
          self.cfg_unscanned,
          max_images_per_sample=4,
      )

    # Negative control 4: mismatched token_types length or missing image-span entries raises ValueError in main().
    bad_mp_tt_dir = self.tmp_root / "masked_prefill_mm_bad_tt"
    bad_mp_tt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        bad_mp_tt_dir / "M1.npz",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mm_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types[:-3],
        num_images=np.array(1, dtype=np.int32),
    )
    bad_mp_tt_args = copy.copy(mp_args)
    bad_mp_tt_args.mm_golden_dir = bad_mp_tt_dir.as_posix()
    with self.assertRaises((ValueError, SystemExit)) as cm:
      masked_prefill_runner.main(
          self.cfg_unscanned,
          bad_mp_tt_args,
          preloaded_model=self.mt_unscanned,
      )
    if isinstance(cm.exception, SystemExit):
      self.assertEqual(cm.exception.code, 42)

    no_img_tt_dir = self.tmp_root / "masked_prefill_mm_no_img_tt"
    no_img_tt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        no_img_tt_dir / "M1.npz",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mm_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=np.full_like(token_types, -1),
        num_images=np.array(1, dtype=np.int32),
    )
    no_img_tt_args = copy.copy(mp_args)
    no_img_tt_args.mm_golden_dir = no_img_tt_dir.as_posix()
    with self.assertRaises((ValueError, SystemExit)) as cm:
      masked_prefill_runner.main(
          self.cfg_unscanned,
          no_img_tt_args,
          preloaded_model=self.mt_unscanned,
      )
    if isinstance(cm.exception, SystemExit):
      self.assertEqual(cm.exception.code, 42)

    def _run_mp_single_npz(
        npz_subdir: str,
        *,
        use_text_path: bool = False,
        enforce_buckets: bool = True,
        max_kl_override: float = 0.08,
        **npz_kwargs: Any,
    ) -> Any:
      d = self.tmp_root / npz_subdir
      d.mkdir(parents=True, exist_ok=True)
      f_path = d / "case.npz"
      np.savez(f_path, **npz_kwargs)
      local_args = copy.copy(mp_args)
      local_args.enforce_length_bucket_gates = enforce_buckets
      local_args.max_kl_div = max_kl_override
      local_args.output_json = (d / "summary.json").as_posix()
      if use_text_path:
        local_args.golden_logits_path = f_path.as_posix()
        local_args.mm_golden_dir = ""
      else:
        local_args.golden_logits_path = ""
        local_args.mm_golden_dir = d.as_posix()
      return masked_prefill_runner._main_impl(  # pylint: disable=protected-access
          self.cfg_unscanned,
          local_args,
          preloaded_model=self.mt_unscanned,
      )

    # Negative controls on masked_prefill_runner:
    # Multimodal sample missing token_types raises ValueError.
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_neg_missing_tt",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_patches,
          image_masks=mm_image_masks,
          num_images=np.array(1, dtype=np.int32),
      )

    # Out-of-range (4, -2) and non-integer (float32, bool) token_types raise ValueError naming {-1, 0, 1, 2, 3}.
    for mp_bad_label, mp_bad_tt_arr in (
        ("above_max_4", np.where(np.arange(token_types.shape[0]) == 0, 4, token_types).astype(np.int32)),
        ("below_min_neg2", np.where(np.arange(token_types.shape[0]) == 0, -2, token_types).astype(np.int32)),
        ("float32_dtype", token_types.astype(np.float32)),
        ("bool_dtype", (token_types >= 0)),
    ):
      with self.subTest(mp_bad_tt=mp_bad_label):
        with self.assertRaisesRegex(ValueError, r"\{-1, 0, 1, 2, 3\}"):
          _run_mp_single_npz(
              f"mp_neg_invalid_tt_{mp_bad_label}",
              sample_id=np.array("M1"),
              tokens=tokens_np,
              logits=mm_logits,
              patches=padded_patches,
              image_masks=mm_image_masks,
              token_types=mp_bad_tt_arr,
              num_images=np.array(1, dtype=np.int32),
          )

    # Mismatch between token_types >= 0 and tokens == ds41_image_token_id raises ValueError.
    mp_first_txt_pos = int(np.flatnonzero(token_types == -1)[0])
    mp_first_img_pos = int(np.flatnonzero(token_types >= 0)[0])
    mp_tt_txt_marked_img = token_types.copy()
    mp_tt_txt_marked_img[mp_first_txt_pos] = 0
    mp_tt_img_marked_txt = token_types.copy()
    mp_tt_img_marked_txt[mp_first_img_pos] = -1
    for mp_mismatch_name, mp_bad_align_tt in (
        ("text_marked_img_start", mp_tt_txt_marked_img),
        ("img_marked_text", mp_tt_img_marked_txt),
    ):
      with self.subTest(mp_align_mismatch=mp_mismatch_name):
        with self.assertRaisesRegex(ValueError, "ds41_image_token_id"):
          _run_mp_single_npz(
              f"mp_neg_align_mismatch_{mp_mismatch_name}",
              sample_id=np.array("M1"),
              tokens=tokens_np,
              logits=mm_logits,
              patches=padded_patches,
              image_masks=mm_image_masks,
              token_types=mp_bad_align_tt,
              num_images=np.array(1, dtype=np.int32),
          )

    # Multimodal sample with all-1 token_types (no -1 text positions) raises ValueError.
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_neg_all_one_tt",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=np.ones_like(token_types, dtype=np.int32),
          num_images=np.array(1, dtype=np.int32),
      )

    # Text-only sample (_has_ds41_mm_images == False) declaring token_types >= 0 (0 or 1) raises ValueError.
    mp_text_tokens = np.where(tokens_np[:12] == img_tok_id, 5, tokens_np[:12]).astype(np.int32)
    for mp_bad_txt_label in (0, 1):
      mp_text_bad_tt = np.full((int(mp_text_tokens.shape[0]),), -1, dtype=np.int32)
      mp_text_bad_tt[0] = mp_bad_txt_label
      with self.assertRaises(ValueError):
        _run_mp_single_npz(
            f"mp_neg_text_with_img_tt_{mp_bad_txt_label}",
            use_text_path=True,
            sample_id=np.array("S0"),
            tokens=mp_text_tokens,
            logits=mm_logits[: int(mp_text_tokens.shape[0])],
            token_types=mp_text_bad_tt,
        )

    # Qwen3-style multimodal gate scope on masked_prefill_runner:
    # Corrupting ONLY image tokens ({0, 2, 3} frame tokens OR all {0, 1, 2, 3} image tokens) PASSES overall because
    # multimodal gating (including rerun/batch determinism gate) is strictly on text_positions (-1), while surfacing
    # spikes in image_positions_metrics and all_positions_metrics and NOT triggering FrozenTopK; corrupting TEXT (-1)
    # FAILS.
    mp_frame_idx = np.flatnonzero(np.isin(token_types, [0, 2, 3]))
    self.assertGreater(len(mp_frame_idx), 0)
    mp_bad_frame_logits = mm_logits.copy()
    mp_bad_frame_logits[mp_frame_idx, 0] += 25.0
    mp_frame_reports = _run_mp_single_npz(
        "mp_scope_corrupt_frame_tokens",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mp_bad_frame_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    with open(self.tmp_root / "mp_scope_corrupt_frame_tokens" / "summary.json", "r", encoding="utf-8") as f:
      mp_frame_summary = json.load(f)
    self.assertTrue(mp_frame_summary["passed"])
    self.assertTrue(mp_frame_summary["all_passed"])
    self.assertLessEqual(mp_frame_summary["max_kl"], 1e-6)
    self.assertTrue(mp_frame_reports[0]["passed"])
    self.assertEqual(mp_frame_reports[0]["failure_reasons"], [])
    self.assertFalse(mp_frame_reports[0]["gates"]["max_kl_pass"])
    self.assertGreater(mp_frame_reports[0]["all_positions_metrics"]["max_kl"], 1.0)
    self.assertGreater(mp_frame_reports[0]["image_positions_metrics"]["max_kl"], 1.0)
    self.assertTrue(all(mp_frame_reports[0]["text_positions_gates"].values()))
    self.assertLessEqual(mp_frame_reports[0]["text_positions_metrics"]["max_kl"], 1e-6)
    self.assertLessEqual(mp_frame_reports[0]["max_kl"], 1e-6)
    self.assertEqual(
        mp_frame_reports[0]["frozen_topk_diagnostic"],
        {"ran": False, "reason": "text_positions_passed"},
    )

    mp_all_img_idx = np.flatnonzero(token_types != -1)
    mp_bad_all_img_logits = mm_logits.copy()
    mp_bad_all_img_logits[mp_all_img_idx, 0] += 25.0
    mp_all_img_reports = _run_mp_single_npz(
        "mp_scope_corrupt_all_image_tokens",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mp_bad_all_img_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    self.assertTrue(mp_all_img_reports[0]["passed"])
    self.assertGreater(mp_all_img_reports[0]["image_positions_metrics"]["max_kl"], 1.0)
    self.assertGreater(mp_all_img_reports[0]["all_positions_metrics"]["max_kl"], 1.0)
    self.assertLessEqual(mp_all_img_reports[0]["text_positions_metrics"]["max_kl"], 1e-6)
    self.assertFalse(mp_all_img_reports[0]["frozen_topk_diagnostic"]["ran"])

    # Rerun determinism scoping on multimodal samples: image-only rerun perturbation PASSES text_positions_gates,
    # whereas text-position rerun perturbation FAILS text_positions_gates!
    orig_mp_gather = masked_prefill_runner._gather_and_free_logits  # pylint: disable=protected-access
    img_rerun_counter = {"n": 0}

    def _perturb_img_rerun(raw_logits: Any, s_len: int, v_size: int) -> tuple[np.ndarray, int]:
      host_arr, r_per_shard = orig_mp_gather(raw_logits, s_len, v_size)
      img_rerun_counter["n"] += 1
      if img_rerun_counter["n"] == 2:
        host_arr[0, mp_first_img_pos, 0] = np.nextafter(host_arr[0, mp_first_img_pos, 0], np.float32(1e6))
      return host_arr, r_per_shard

    with mock.patch.object(masked_prefill_runner, "_gather_and_free_logits", side_effect=_perturb_img_rerun):
      mp_img_rerun_reports = _run_mp_single_npz(
          "mp_scope_img_rerun_passes",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    self.assertTrue(mp_img_rerun_reports[0]["passed"])
    self.assertTrue(mp_img_rerun_reports[0]["text_positions_metrics"]["rerun_bitwise_identical"])
    self.assertFalse(mp_img_rerun_reports[0]["image_positions_metrics"]["rerun_bitwise_identical"])
    self.assertFalse(mp_img_rerun_reports[0]["all_positions_metrics"]["rerun_bitwise_identical"])

    txt_rerun_counter = {"n": 0}

    def _perturb_txt_rerun(raw_logits: Any, s_len: int, v_size: int) -> tuple[np.ndarray, int]:
      host_arr, r_per_shard = orig_mp_gather(raw_logits, s_len, v_size)
      txt_rerun_counter["n"] += 1
      if txt_rerun_counter["n"] == 2:
        host_arr[0, mp_first_txt_pos, 0] = np.nextafter(host_arr[0, mp_first_txt_pos, 0], np.float32(1e6))
      return host_arr, r_per_shard

    with mock.patch.object(masked_prefill_runner, "_gather_and_free_logits", side_effect=_perturb_txt_rerun):
      with self.assertRaises(AssertionError):
        _run_mp_single_npz(
            "mp_scope_txt_rerun_fails",
            sample_id=np.array("M1"),
            tokens=tokens_np,
            logits=mm_logits,
            patches=padded_patches,
            image_masks=mm_image_masks,
            token_types=token_types,
            num_images=np.array(1, dtype=np.int32),
        )
    with open(self.tmp_root / "mp_scope_txt_rerun_fails" / "summary.json", "r", encoding="utf-8") as f:
      mp_txt_rerun_summary = json.load(f)
    self.assertFalse(mp_txt_rerun_summary["passed"])
    self.assertFalse(mp_txt_rerun_summary["samples"][0]["text_positions_gates"]["batch_determinism_pass"])
    self.assertFalse(mp_txt_rerun_summary["samples"][0]["text_positions_metrics"]["rerun_bitwise_identical"])

    mp_bad_text_logits = mm_logits.copy()
    mp_bad_text_logits[mp_first_txt_pos, 0] += 25.0
    with self.assertRaises(AssertionError):
      _run_mp_single_npz(
          "mp_scope_corrupt_text_tokens",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mp_bad_text_logits,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(self.tmp_root / "mp_scope_corrupt_text_tokens" / "summary.json", "r", encoding="utf-8") as f:
      mp_txt_summary = json.load(f)
    self.assertFalse(mp_txt_summary["passed"])
    self.assertFalse(mp_txt_summary["all_passed"])
    self.assertFalse(mp_txt_summary["samples"][0]["text_positions_gates"]["max_kl_pass"])
    self.assertFalse(mp_txt_summary["samples"][0]["gates"]["max_kl_pass"])
    self.assertEqual(
        mp_txt_summary["samples"][0]["frozen_topk_diagnostic"],
        {"ran": False, "reason": "missing_ref_topk_experts"},
    )

    # Non-binary image_masks ([-1, 2, 0, 0] with num_images=1) raises ValueError.
    with self.assertRaises(ValueError):
      masked_prefill_runner.prepare_ds41_mm_batch_inputs(
          padded_patches,
          4,
          max_images_per_sample=4,
          image_masks=np.array([-1, 2, 0, 0], dtype=np.int32),
          num_images=1,
      )
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_neg_nonbinary_mask",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_patches,
          image_masks=np.array([-1, 2, 0, 0], dtype=np.int32),
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )

    # Padded-slot image_masks ([0, 1, 0, 0] on n_img=1 patches [1, P, 588], num_images=1) raises ValueError.
    with self.assertRaises(ValueError):
      masked_prefill_runner.prepare_ds41_mm_batch_inputs(
          valid_patch_0,
          4,
          max_images_per_sample=4,
          image_masks=np.array([0, 1, 0, 0], dtype=np.int32),
          num_images=1,
      )
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_neg_padded_slot_mask",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=valid_patch_0,
          image_masks=np.array([0, 1, 0, 0], dtype=np.int32),
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )

    # All-zero image_masks ([0, 0, 0, 0] with num_images=None) and n_img=0 raise ValueError.
    with self.assertRaises(ValueError):
      masked_prefill_runner.prepare_ds41_mm_batch_inputs(
          padded_patches,
          4,
          max_images_per_sample=4,
          image_masks=np.array([0, 0, 0, 0], dtype=np.int32),
          num_images=None,
      )
    with self.assertRaises(ValueError):
      masked_prefill_runner.prepare_ds41_mm_batch_inputs(
          np.zeros((0, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32),
          4,
          max_images_per_sample=4,
      )
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_neg_all_zero_mask",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          patches=padded_patches,
          image_masks=np.array([0, 0, 0, 0], dtype=np.int32),
          token_types=token_types,
      )

    # Tie-flip envelope E2E multimodal .npz controls (exercising both all_positions and text_positions slicing):
    mp_text_idx = int(np.flatnonzero(token_types == -1)[0])

    # (i) Positive near-tie envelope pass: 1 text token with primary KL ~ 1.4e-3 > max_kl_div=1e-3,
    # primary p99_kl <= 2e-3 on both all_positions and text_positions, resolved by tie_flip_logits.
    mp_neartie_logits = mm_logits.copy()
    mp_neartie_logits[mp_text_idx] = _perturb_row_to_target_kl(mm_logits[mp_text_idx], 1.4e-3)
    with self.assertRaises(AssertionError):
      _run_mp_single_npz(
          "mp_env_neartie_without_env",
          enforce_buckets=False,
          max_kl_override=1e-3,
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mp_neartie_logits,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    mp_env_reports = _run_mp_single_npz(
        "mp_env_neartie_with_env",
        enforce_buckets=False,
        max_kl_override=1e-3,
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=mp_neartie_logits,
        tie_flip_logits=mm_logits,
        patches=padded_patches,
        image_masks=mm_image_masks,
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
    )
    self.assertTrue(mp_env_reports[0]["passed"])
    # On multimodal samples, tie_flip_logits applies ONLY to text_positions max_kl.
    self.assertFalse(mp_env_reports[0]["metrics"]["tie_flip_envelope_used"])
    self.assertGreater(mp_env_reports[0]["metrics"]["max_kl"], 1e-3)
    self.assertFalse(mp_env_reports[0]["image_positions_metrics"]["tie_flip_envelope_used"])
    self.assertTrue(mp_env_reports[0]["text_positions_metrics"]["tie_flip_envelope_used"])
    self.assertGreater(mp_env_reports[0]["text_positions_metrics"]["raw_primary_max_kl"], 1e-3)
    self.assertLessEqual(mp_env_reports[0]["text_positions_metrics"]["max_kl"], 1e-6)
    self.assertTrue(mp_env_reports[0]["text_positions_gates"]["p99_kl_pass"])

    # (ii) Single-token +1.5 * max_kl_limit spike on text_positions with envelope present:
    # Bucket max_kl_limit for <= 256 tokens is 0.025, so +1.5 * 0.025 = 0.0375 MUST fail text_positions max_kl_pass!
    mp_corrupt_both = mm_logits.copy()
    mp_corrupt_both[mp_text_idx] = _perturb_row_to_target_kl(mm_logits[mp_text_idx], 1.5 * 0.025)
    with self.assertRaises(AssertionError):
      _run_mp_single_npz(
          "mp_env_corrupt_both",
          enforce_buckets=True,
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mp_corrupt_both,
          tie_flip_logits=mp_corrupt_both,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    with open(self.tmp_root / "mp_env_corrupt_both" / "summary.json", "r", encoding="utf-8") as f:
      mp_corrupt_both_summary = json.load(f)
    self.assertFalse(mp_corrupt_both_summary["passed"])
    self.assertFalse(mp_corrupt_both_summary["samples"][0]["passed"])
    self.assertFalse(mp_corrupt_both_summary["samples"][0]["text_positions_gates"]["max_kl_pass"])
    self.assertGreater(mp_corrupt_both_summary["samples"][0]["text_positions_metrics"]["max_kl"], 0.025)

    # (iii) Primary p99_kl > 0.002 fails p99_kl_pass even when tie_flip_logits == mm_logits (max_kl == 0).
    mp_p99_bad = mm_logits.copy()
    mp_p99_bad[mp_text_idx] = _perturb_row_to_target_kl(mm_logits[mp_text_idx], 4.0e-3)
    with self.assertRaises(AssertionError):
      _run_mp_single_npz(
          "mp_env_p99_fail",
          enforce_buckets=True,
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mp_p99_bad,
          tie_flip_logits=mm_logits,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )

    # (iv) Wrong-shape or NaN tie_flip_logits in .npz raises ValueError.
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_env_bad_shape",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          tie_flip_logits=mm_logits[:, :-1],
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )
    mp_nan_tf = mm_logits.copy()
    mp_nan_tf[0, 0] = np.nan
    with self.assertRaises(ValueError):
      _run_mp_single_npz(
          "mp_env_bad_nan",
          sample_id=np.array("M1"),
          tokens=tokens_np,
          logits=mm_logits,
          tie_flip_logits=mp_nan_tf,
          patches=padded_patches,
          image_masks=mm_image_masks,
          token_types=token_types,
          num_images=np.array(1, dtype=np.int32),
      )

  def test_aot_check_cpu_mini_with_and_without_images(self) -> None:
    """Tests `aot_check.py` forward compilation, full `[B, S, V]` output shape, vision HLO markers, and HBM."""
    aot_mod = _load_aot_check_module()

    # Check E2E and benchmark preset specifications.
    for p_name in (
        "tpu7x_64c_text_scanned_fwd",
        "tpu7x_64c_text_unscanned_fwd",
        "tpu7x_64c_mm_scanned_fwd",
        "tpu7x_64c_mm_unscanned_fwd",
        "tpu7x_64c_mm_prefill_2048",
        "tpu7x_64c_text_prefill_2048",
    ):
      spec = aot_mod.get_preset_spec(p_name)
      self.assertEqual(spec["mode"], "forward")
      self.assertFalse(spec["overrides"]["ds41_activation_quant"])
      self.assertEqual(spec["overrides"]["dtype"], "float32")
      self.assertEqual(spec["overrides"]["weight_dtype"], "bfloat16")
      self.assertEqual(spec["overrides"]["remat_policy"], "none")
      self.assertEqual(spec["max_hbm_gib"], 85.0)
    spec_mm_train = aot_mod.get_preset_spec("tpu7x_64c_mm_train_adamw")
    self.assertEqual(spec_mm_train["mode"], "train")
    self.assertTrue(spec_mm_train["overrides"]["ds41_activation_quant"])
    self.assertTrue(spec_mm_train["overrides"]["freeze_vision_encoder_params"])
    self.assertEqual(spec_mm_train["max_hbm_gib"], 85.0)

    for p_name in ("tpu7x_64c_text_train_sgd_b1", "tpu7x_64c_text_train_sgd_b0125"):
      spec = aot_mod.get_preset_spec(p_name)
      self.assertEqual(spec["mode"], "train")
      self.assertFalse(spec["with_images"])
      self.assertFalse(spec["expect_vision_in_hlo"])
      self.assertEqual(spec["max_hbm_gib"], 85.0)
      self.assertTrue(spec["overrides"]["use_multimodal"])
      self.assertTrue(spec["overrides"]["freeze_vision_encoder_params"])
      self.assertEqual(spec["overrides"]["max_target_length"], 1024)
      self.assertEqual(spec["overrides"]["opt_type"], "sgd")

    for p_name in ("tpu7x_64c_mm_bench_min_remat", "tpu7x_64c_mm_bench_full_remat"):
      spec = aot_mod.get_preset_spec(p_name)
      self.assertEqual(spec["mode"], "train")
      self.assertTrue(spec["with_images"])
      self.assertTrue(spec["expect_vision_in_hlo"])
      self.assertEqual(spec["max_hbm_gib"], aot_mod.BENCH_MAX_HBM_GIB)
      self.assertEqual(spec["overrides"]["compile_topology"], "tpu7x-128")
      self.assertTrue(spec["overrides"]["use_multimodal"])
      self.assertTrue(spec["overrides"]["freeze_vision_encoder_params"])
      self.assertEqual(spec["overrides"]["max_num_images_per_example"], 4)
      self.assertEqual(spec["overrides"]["max_target_length"], 4096)

    spec_64c_txt_min = aot_mod.get_preset_spec("tpu7x_64c_text_bench_min_remat")
    self.assertEqual(spec_64c_txt_min["mode"], "train")
    self.assertFalse(spec_64c_txt_min["with_images"])
    self.assertFalse(spec_64c_txt_min["expect_vision_in_hlo"])
    self.assertEqual(spec_64c_txt_min["max_hbm_gib"], aot_mod.BENCH_MAX_HBM_GIB)
    self.assertEqual(spec_64c_txt_min["overrides"]["compile_topology"], "tpu7x-128")
    self.assertFalse(spec_64c_txt_min["overrides"]["use_multimodal"])
    self.assertFalse(spec_64c_txt_min["overrides"]["freeze_vision_encoder_params"])
    self.assertEqual(spec_64c_txt_min["overrides"]["max_target_length"], 4096)

    for p_name in ("tpu7x_128c_mm_bench_min_remat", "tpu7x_128c_mm_bench_full_remat"):
      spec = aot_mod.get_preset_spec(p_name)
      self.assertEqual(spec["mode"], "train")
      self.assertTrue(spec["with_images"])
      self.assertTrue(spec["expect_vision_in_hlo"])
      self.assertEqual(spec["max_hbm_gib"], aot_mod.BENCH_MAX_HBM_GIB)
      self.assertEqual(spec["overrides"]["compile_topology"], "tpu7x-256")
      self.assertTrue(spec["overrides"]["use_multimodal"])
      self.assertTrue(spec["overrides"]["freeze_vision_encoder_params"])
      self.assertEqual(spec["overrides"]["max_num_images_per_example"], 4)
      self.assertEqual(spec["overrides"]["max_target_length"], 4096)

    spec_128c_txt_min = aot_mod.get_preset_spec("tpu7x_128c_text_bench_min_remat")
    self.assertEqual(spec_128c_txt_min["mode"], "train")
    self.assertFalse(spec_128c_txt_min["with_images"])
    self.assertFalse(spec_128c_txt_min["expect_vision_in_hlo"])
    self.assertEqual(spec_128c_txt_min["max_hbm_gib"], aot_mod.BENCH_MAX_HBM_GIB)
    self.assertEqual(spec_128c_txt_min["overrides"]["compile_topology"], "tpu7x-256")
    self.assertFalse(spec_128c_txt_min["overrides"]["use_multimodal"])
    self.assertFalse(spec_128c_txt_min["overrides"]["freeze_vision_encoder_params"])
    self.assertEqual(spec_128c_txt_min["overrides"]["max_target_length"], 4096)

    # Positive control 1: tpu7x_64c_mm_scanned_fwd (with_images=True) on CPU mini-twin.
    res_mm_scan = aot_mod.run_preset("tpu7x_64c_mm_scanned_fwd", cpu_mini=True, max_target_length_override=36)
    self.assertEqual(res_mm_scan["output_shape"], [int(self.cfg_scanned.global_batch_size_to_load), 36, 256])
    self.assertTrue(all(res_mm_scan["vision_hlo_markers"].values()))
    self.assertLessEqual(res_mm_scan["peak_hbm_gib"], 85.0)

    # Positive control 2: tpu7x_64c_text_scanned_fwd (with_images=False) on CPU mini-twin.
    res_txt_scan = aot_mod.run_preset("tpu7x_64c_text_scanned_fwd", cpu_mini=True, max_target_length_override=36)
    self.assertEqual(res_txt_scan["output_shape"], [int(self.cfg_scanned.global_batch_size_to_load), 36, 256])
    self.assertFalse(any(res_txt_scan["vision_hlo_markers"].values()))
    self.assertLessEqual(res_txt_scan["peak_hbm_gib"], 85.0)

    # Negative control 1: wrong expect_vision_in_hlo raises AssertionError.
    with self.assertRaises(AssertionError):
      aot_mod.run_preset(
          "tpu7x_64c_mm_scanned_fwd",
          cpu_mini=True,
          max_target_length_override=36,
          expect_vision_in_hlo_override=False,
      )
    with self.assertRaises(AssertionError):
      aot_mod.run_preset(
          "tpu7x_64c_text_scanned_fwd",
          cpu_mini=True,
          max_target_length_override=36,
          expect_vision_in_hlo_override=True,
      )

    # Negative control 2: impossible HBM ceiling raises AssertionError.
    with self.assertRaises(AssertionError):
      aot_mod.run_preset(
          "tpu7x_64c_mm_scanned_fwd",
          cpu_mini=True,
          max_target_length_override=36,
          max_hbm_gib_override=1e-9,
      )

    # Verify FrozenTopK and double-pass AOT presets and routing_mode coverage.
    self.assertEqual(
        aot_mod.E2E_PRESET_ORDER,
        (
            "tpu7x_64c_text_scanned_fwd",
            "tpu7x_64c_text_unscanned_fwd",
            "tpu7x_64c_mm_scanned_fwd",
            "tpu7x_64c_mm_scanned_fwd_frozen",
            "tpu7x_64c_mm_scanned_fwd_double",
            "tpu7x_64c_mm_scanned_fwd_frozen_b128",
            "tpu7x_64c_mm_scanned_fwd_double_b128",
            "tpu7x_64c_mm_unscanned_fwd",
            "tpu7x_64c_mm_unscanned_fwd_frozen",
            "tpu7x_64c_mm_unscanned_fwd_double",
            "tpu7x_64c_mm_unscanned_fwd_frozen_b128",
            "tpu7x_64c_mm_unscanned_fwd_double_b128",
            "tpu7x_64c_mm_prefill_2048",
            "tpu7x_64c_mm_prefill_2048_frozen",
            "tpu7x_64c_mm_prefill_2048_double",
            "tpu7x_64c_mm_prefill_2048_frozen_b128",
            "tpu7x_64c_mm_prefill_2048_double_b128",
            "tpu7x_64c_text_prefill_2048",
            "tpu7x_64c_mm_train_adamw",
        ),
    )
    for base_p, want_scan, want_seq in (
        ("tpu7x_64c_mm_scanned_fwd", True, 1024),
        ("tpu7x_64c_mm_unscanned_fwd", False, 1024),
        ("tpu7x_64c_mm_prefill_2048", True, 2048),
    ):
      for suffix, want_rmode, want_pdbs in (
          ("_frozen", "frozen_topk", 0.125),
          ("_double", "double_pass", 0.125),
          ("_frozen_b128", "frozen_topk", 1.0),
          ("_double_b128", "double_pass", 1.0),
      ):
        p_spec = aot_mod.get_preset_spec(f"{base_p}{suffix}")
        self.assertEqual(p_spec["mode"], "forward")
        self.assertEqual(p_spec["routing_mode"], want_rmode)
        self.assertTrue(p_spec["with_images"])
        self.assertEqual(p_spec["num_images"], 4)
        self.assertTrue(p_spec["expect_vision_in_hlo"])
        self.assertEqual(p_spec["max_hbm_gib"], 85.0)
        self.assertEqual(p_spec["overrides"]["scan_layers"], want_scan)
        self.assertEqual(p_spec["overrides"]["max_target_length"], want_seq)
        self.assertEqual(p_spec["overrides"]["per_device_batch_size"], want_pdbs)

    res_mm_scan_frozen = aot_mod.run_preset(
        "tpu7x_64c_mm_scanned_fwd_frozen", cpu_mini=True, max_target_length_override=36
    )
    self.assertEqual(res_mm_scan_frozen["routing_mode"], "frozen_topk")
    self.assertTrue(res_mm_scan_frozen["frozen_topk_hlo_verified"])
    self.assertEqual(res_mm_scan_frozen["output_shape"], [int(self.cfg_scanned.global_batch_size_to_load), 36, 256])
    self.assertTrue(all(res_mm_scan_frozen["vision_hlo_markers"].values()))
    self.assertTrue(res_mm_scan_frozen["passed"])

    res_a4_frozen = aot_mod.run_preset("tpu7x_64c_mm_unscanned_fwd_frozen", cpu_mini=True, max_target_length_override=36)
    self.assertEqual(res_a4_frozen["routing_mode"], "frozen_topk")
    self.assertFalse(res_a4_frozen["scan_layers"])
    self.assertTrue(res_a4_frozen["frozen_topk_hlo_verified"])
    self.assertTrue(res_a4_frozen["passed"])

    res_a5_frozen = aot_mod.run_preset("tpu7x_64c_mm_prefill_2048_frozen", cpu_mini=True, max_target_length_override=36)
    self.assertEqual(res_a5_frozen["routing_mode"], "frozen_topk")
    self.assertTrue(res_a5_frozen["frozen_topk_hlo_verified"])
    self.assertTrue(res_a5_frozen["passed"])

    res_nat_trace = aot_mod.run_forward_aot_check(
        self.cfg_scanned,
        with_images=True,
        num_images=4,
        expect_vision_in_hlo=True,
        max_hbm_gib=85.0,
        preset_label="mm_scanned_fwd_natural_trace",
        use_cpu_mesh=True,
        routing_mode="natural_trace",
    )
    self.assertEqual(res_nat_trace["routing_mode"], "natural_trace")
    self.assertTrue(res_nat_trace["natural_trace_hlo_verified"])
    self.assertEqual(
        res_nat_trace["output_shape"],
        [
            int(self.cfg_scanned.global_batch_size_to_load),
            int(self.cfg_scanned.max_target_length),
            int(self.cfg_scanned.vocab_size),
        ],
    )
    self.assertTrue(res_nat_trace["passed"])

    res_mm_scan_double = aot_mod.run_preset(
        "tpu7x_64c_mm_scanned_fwd_double", cpu_mini=True, max_target_length_override=36
    )
    self.assertEqual(res_mm_scan_double["routing_mode"], "double_pass")
    self.assertTrue(res_mm_scan_double["passed"])
    self.assertEqual(res_mm_scan_double["frozen_pass"]["routing_mode"], "frozen_topk")
    self.assertEqual(res_mm_scan_double["natural_trace_pass"]["routing_mode"], "natural_trace")
    self.assertAlmostEqual(
        res_mm_scan_double["unshared_sum_double_pass_hbm_gib"],
        res_mm_scan_double["frozen_peak_hbm_gib"] + res_mm_scan_double["natural_trace_peak_hbm_gib"],
        places=9,
    )
    self.assertEqual(res_mm_scan_double["peak_hbm_gib"], res_mm_scan_double["shared_weights_double_pass_peak_hbm_gib"])

    res_mm_scan_frozen_b128 = aot_mod.run_preset(
        "tpu7x_64c_mm_scanned_fwd_frozen_b128", cpu_mini=True, max_target_length_override=36
    )
    self.assertTrue(res_mm_scan_frozen_b128["passed"])
    res_mm_scan_double_b128 = aot_mod.run_preset(
        "tpu7x_64c_mm_scanned_fwd_double_b128", cpu_mini=True, max_target_length_override=36
    )
    self.assertTrue(res_mm_scan_double_b128["passed"])

    with self.assertRaises(AssertionError):
      aot_mod.run_preset(
          "tpu7x_64c_mm_scanned_fwd_double",
          cpu_mini=True,
          max_target_length_override=36,
          max_hbm_gib_override=1e-9,
      )
    with self.assertRaises(ValueError):
      aot_mod.run_forward_aot_check(
          self.cfg_scanned,
          with_images=True,
          use_cpu_mesh=True,
          routing_mode="invalid_mode",
      )

  def test_aot_train_subbatch_and_vision_hlo_semantics(self) -> None:
    """Tests `_check_vision_hlo` marker semantics and `run_train_aot_check` with `per_device_batch_size < 1.0`."""
    aot_mod = _load_aot_check_module()
    check_vision_hlo = aot_mod._check_vision_hlo  # pylint: disable=protected-access
    self.assertEqual(aot_mod.VISION_MODULE_MARKERS, ("vision_encoder",))
    self.assertEqual(aot_mod.VISION_COMPUTE_MARKERS, ("embed_patches", "merge_mm_embeddings"))
    self.assertEqual(
        aot_mod.VISION_HLO_MARKERS,
        ("vision_encoder", "embed_patches", "merge_mm_embeddings"),
    )

    # 1a. expect_vision_in_hlo=True: requires all 3 markers.
    full_hlo = "HloModule m %p = vision_encoder %c1 = embed_patches %c2 = merge_mm_embeddings"
    res_true = check_vision_hlo(full_hlo, expect_vision_in_hlo=True, use_multimodal=True)
    self.assertEqual(
        res_true,
        {"vision_encoder": True, "embed_patches": True, "merge_mm_embeddings": True},
    )
    for missing_marker in aot_mod.VISION_HLO_MARKERS:
      partial_hlo = full_hlo.replace(missing_marker, "other_symbol")
      with self.assertRaisesRegex(AssertionError, missing_marker):
        check_vision_hlo(partial_hlo, expect_vision_in_hlo=True, use_multimodal=True)

    # 1b. expect_vision_in_hlo=False, use_multimodal=True (text-only SFT SGD case):
    # vision_encoder parameter marker is allowed when compute markers are absent.
    a7_hlo = "HloModule train_step %arg_state_params_vision_encoder_w = f32[16,16] parameter(0)"
    res_a7 = check_vision_hlo(a7_hlo, expect_vision_in_hlo=False, use_multimodal=True)
    self.assertEqual(
        res_a7,
        {"vision_encoder": True, "embed_patches": False, "merge_mm_embeddings": False},
    )
    for leaked_compute in aot_mod.VISION_COMPUTE_MARKERS:
      with self.assertRaisesRegex(AssertionError, leaked_compute):
        check_vision_hlo(
            f"{a7_hlo} {leaked_compute}",
            expect_vision_in_hlo=False,
            use_multimodal=True,
        )

    # 1c. expect_vision_in_hlo=False, use_multimodal=False (or default):
    # vision_encoder marker must be absent.
    res_text_only = check_vision_hlo(
        "HloModule text_only %arg0 = f32[16,16] parameter(0)",
        expect_vision_in_hlo=False,
        use_multimodal=False,
    )
    self.assertFalse(any(res_text_only.values()))
    with self.assertRaisesRegex(AssertionError, "use_multimodal=False"):
      check_vision_hlo(a7_hlo, expect_vision_in_hlo=False, use_multimodal=False)
    with self.assertRaisesRegex(AssertionError, "use_multimodal=False"):
      check_vision_hlo(a7_hlo, expect_vision_in_hlo=False)

    # 2. run_train_aot_check on 4-device CPU mini-twin with per_device_batch_size=0.5
    # (global_batch_size_to_load=4, micro_batch_size_to_train_on=2, partitioned across 4 devices).
    if jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 host CPU devices for sub-batch AOT check, got {jax.device_count()}.")
    train_cfg = mm_testing.mini_vision_maxtext_config(
        vocab_size=256,
        ds41_image_token_id=200,
        use_multimodal=True,
        scan_layers=True,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        dtype_mm="bfloat16",
        matmul_precision="default",
        remat_policy="full",
        ds41_activation_quant=False,
        per_device_batch_size=0.5,
        max_target_length=36,
        freeze_vision_encoder_params=True,
        max_num_images_per_example=1,
        opt_type="sgd",
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        ds41_engram_compressed_vocab_size=0,
    )
    self.assertEqual(int(train_cfg.global_batch_size_to_load), 4)
    self.assertEqual(int(train_cfg.micro_batch_size_to_train_on), 2)

    # 2a. Multimodal train step with images (multimodal train sub-batch case).
    res_train_mm = aot_mod.run_train_aot_check(
        train_cfg,
        with_images=True,
        expect_vision_in_hlo=True,
        max_hbm_gib=85.0,
        use_cpu_mesh=True,
    )
    self.assertEqual(res_train_mm["mode"], "train")
    self.assertTrue(res_train_mm["with_images"])
    self.assertTrue(res_train_mm["use_multimodal"])
    self.assertEqual(
        res_train_mm["vision_hlo_markers"],
        {"vision_encoder": True, "embed_patches": True, "merge_mm_embeddings": True},
    )
    self.assertLessEqual(res_train_mm["peak_hbm_gib"], 85.0)
    with self.assertRaisesRegex(AssertionError, "compute_present"):
      aot_mod.run_train_aot_check(
          train_cfg,
          with_images=True,
          expect_vision_in_hlo=False,
          max_hbm_gib=85.0,
          use_cpu_mesh=True,
      )

    # 2b. Multimodal train step without images (text-only SFT SGD case).
    res_train_text = aot_mod.run_train_aot_check(
        train_cfg,
        with_images=False,
        expect_vision_in_hlo=False,
        max_hbm_gib=85.0,
        use_cpu_mesh=True,
    )
    self.assertEqual(res_train_text["mode"], "train")
    self.assertFalse(res_train_text["with_images"])
    self.assertTrue(res_train_text["use_multimodal"])
    self.assertTrue(res_train_text["vision_hlo_markers"]["vision_encoder"])
    self.assertFalse(res_train_text["vision_hlo_markers"]["embed_patches"])
    self.assertFalse(res_train_text["vision_hlo_markers"]["merge_mm_embeddings"])
    self.assertLessEqual(res_train_text["peak_hbm_gib"], 85.0)
    with self.assertRaisesRegex(AssertionError, "missing"):
      aot_mod.run_train_aot_check(
          train_cfg,
          with_images=False,
          expect_vision_in_hlo=True,
          max_hbm_gib=85.0,
          use_cpu_mesh=True,
      )

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_row_sliced_safetensors_dequant_engram_pad_and_gs_orbax_uri(self) -> None:
    """Tests `load_dequantized_row_slice`, row-sliced `engram_pad`, and `gs://` `_normalize_orbax_root`."""
    # 1. Verify _normalize_orbax_root preserves double-slash gs:// URIs and strips /0/items.
    gs_root = deepseek_v41_mm_audit._normalize_orbax_root(  # pylint: disable=protected-access
        "gs://example-bucket/prefix/scanned"
    )
    self.assertEqual(gs_root, "gs://example-bucket/prefix/scanned")
    self.assertTrue(gs_root.startswith("gs://"))

    gs_items_root = deepseek_v41_mm_audit._normalize_orbax_root(  # pylint: disable=protected-access
        "gs://example-bucket/prefix/scanned/0/items"
    )
    self.assertEqual(gs_items_root, "gs://example-bucket/prefix/scanned")

    local_items_root = deepseek_v41_mm_audit._normalize_orbax_root(  # pylint: disable=protected-access
        self.tmp_root / "some_orbax" / "0" / "items"
    )
    self.assertEqual(local_items_root, (self.tmp_root / "some_orbax").as_posix())

    # 2. Build a synthetic safetensors checkpoint containing:
    #    - 32x32 block-scaled FP8 linear (`layers.0.attn.wq_a.weight` + `.scale`)
    #    - 1x32 Engram-style FP8 table (`layers.1.engram.embed.weight` + `.scale`, raw_rows=100, cols=64)
    #    - unquantized BF16 (`embed.weight`) and F32 (`layers.0.hc_attn_base`) tensors
    torch.manual_seed(2026)
    raw_rows = 100
    cols = 64
    fp8_32x32_w = torch.randint(-60, 60, (128, cols), dtype=torch.int8).view(torch.float8_e4m3fn)
    fp8_32x32_s = torch.randint(124, 131, (4, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)

    engram_fp8_w = torch.randint(-60, 60, (raw_rows, cols), dtype=torch.int8).view(torch.float8_e4m3fn)
    engram_fp8_s = torch.randint(124, 131, (raw_rows, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)

    bf16_w = torch.randn((raw_rows, cols), dtype=torch.bfloat16)
    f32_w = torch.randn((raw_rows, cols), dtype=torch.float32)

    synth_dir = self.tmp_root / "synth_row_slice_ckpt"
    _save_safetensors_dir(
        synth_dir,
        {
            "layers.0.attn.wq_a.weight": fp8_32x32_w,
            "layers.0.attn.wq_a.scale": fp8_32x32_s,
            "layers.1.engram.embed.weight": engram_fp8_w,
            "layers.1.engram.embed.scale": engram_fp8_s,
            "embed.weight": bf16_w,
            "layers.0.hc_attn_base": f32_w,
        },
    )

    reader = deepseek_v41_mm_audit._SafetensorsShardReader(synth_dir)  # pylint: disable=protected-access
    for key, dt_str, n_rows in (
        ("layers.0.attn.wq_a.weight", "bfloat16", 128),
        ("layers.1.engram.embed.weight", "bfloat16", raw_rows),
        ("embed.weight", "bfloat16", raw_rows),
        ("layers.0.hc_attn_base", "float32", raw_rows),
    ):
      full_arr = reader.load_dequantized_tensor(key, dt_str)
      reader.clear_cache()
      for r0, r1 in ((0, 37), (19, 83), (n_rows - 15, n_rows), (40, 40)):
        sliced_arr = reader.load_dequantized_row_slice(key, dt_str, r0, r1)
        ok, n_bad = deepseek_v41_mm_audit._bitwise_equal(sliced_arr, full_arr[r0:r1])  # pylint: disable=protected-access
        self.assertTrue(
            ok,
            msg=f"Row slice [{r0}:{r1}] mismatch on {key} ({dt_str}): {n_bad} bad elements",
        )

    # 3. Verify `engram_pad` row-sliced shard construction matches full-tensor construction bit-for-bit
    #    across interior, boundary-straddling, and pure-padding tail slices (both `single` and `scanned_single`).
    padded_rows = 128
    full_engram_padded = deepseek_v41_mm_audit._transform_single_hf_tensor(  # pylint: disable=protected-access
        reader.load_dequantized_tensor("layers.1.engram.embed.weight", "bfloat16"),
        "engram_pad",
        None,
        padded_engram_rows=padded_rows,
    )
    reader.clear_cache()

    single_rule = deepseek_v41_mm_audit.LeafAuditRule(
        maxtext_path="decoder/layers_1/engram/engram_table/embedding",
        category="text",
        kind="single",
        hf_keys=("layers.1.engram.embed.weight",),
        transform="engram_pad",
        expected_mt_dtype="bfloat16",
    )
    scanned_rule = deepseek_v41_mm_audit.LeafAuditRule(
        maxtext_path="decoder/segments_0/engram/engram_table/embedding",
        category="text",
        kind="scanned_single",
        hf_keys=("layers.1.engram.embed.weight", "layers.1.engram.embed.weight"),
        transform="engram_pad",
        expected_mt_dtype="bfloat16",
    )

    for row_sl, col_sl in (
        (slice(0, 37), slice(None)),
        (slice(19, 83), slice(16, 48)),
        (slice(85, 120), slice(0, 32)),
        (slice(100, 128), slice(None)),
    ):
      got_single = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
          rule=single_rule,
          global_shape=(padded_rows, cols),
          shard_index=(row_sl, col_sl),
          scan_axis=0,
          reader=reader,
          config=self.cfg_unscanned,
      )
      want_single = full_engram_padded[row_sl, col_sl]
      ok_s, n_bad_s = deepseek_v41_mm_audit._bitwise_equal(got_single, want_single)  # pylint: disable=protected-access
      self.assertTrue(ok_s, msg=f"single engram_pad mismatch for {row_sl}, {col_sl}: {n_bad_s}")

      for scan_axis in (0, 1):
        global_3d = (2, padded_rows, cols) if scan_axis == 0 else (padded_rows, 2, cols)
        idx_3d = (slice(0, 2), row_sl, col_sl) if scan_axis == 0 else (row_sl, slice(0, 2), col_sl)
        got_scanned = deepseek_v41_mm_audit._build_expected_shard_slice(  # pylint: disable=protected-access
            rule=scanned_rule,
            global_shape=global_3d,
            shard_index=idx_3d,
            scan_axis=scan_axis,
            reader=reader,
            config=self.cfg_scanned,
        )
        want_scanned = np.stack([want_single, want_single], axis=scan_axis)
        ok_sc, n_bad_sc = deepseek_v41_mm_audit._bitwise_equal(  # pylint: disable=protected-access
            got_scanned, want_scanned
        )
        self.assertTrue(
            ok_sc,
            msg=f"scanned_single engram_pad mismatch (scan_axis={scan_axis}) for {row_sl}, {col_sl}: {n_bad_sc}",
        )

  def test_per_token_kl_float64_129280_vocab_precision_and_negative_control(self) -> None:
    """Verifies host float64 per-token KL on 129,280-vocab float32 logits plus 1e-6 and 1.5e-3 negative controls."""
    _assert_129280_vocab_float64_kl_precision_and_negative_controls(self)

  def test_frozen_topk_teacher_forced_diagnostic_scanned_and_unscanned(self) -> None:
    """Tests failure-only FrozenTopK diagnostic under jax.jit for BOTH scan_layers=True and scan_layers=False."""
    short_seq_len = 36
    self.assertLess(short_seq_len, int(self.cfg_unscanned.max_target_length))
    batch = multimodal.build_mini_vl_batch(
        self.ref_args,
        grid=multimodal.MINI_GRID_PADDED,
        batch_size=1,
        seq_len=short_seq_len,
        span_starts=((2,),),
        seed=2026,
    )
    tokens_np = batch.input_ids.numpy().astype(np.int32)[0]
    token_types = batch.token_types.numpy().astype(np.int32)[0]
    valid_patch_0 = batch.patches_bnpd.numpy().astype(np.float32)[0]
    padded_4_slots = np.zeros((4, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32)
    padded_4_slots[0] = valid_patch_0[0]
    image_mask_slot1 = np.array([1, 0, 0, 0], dtype=np.int32)

    num_layers = int(self.cfg_unscanned.num_decoder_layers)
    top_k = int(self.cfg_unscanned.num_experts_per_tok)
    num_experts = int(self.cfg_unscanned.num_experts)

    # Record clean logits and exact per-layer top-k selected expert indices (uint16 [num_layers, S, top_k])
    # on unperturbed mt_unscanned with S < max_target_length (exercising sequence padding to 64).
    ids_pad = np.zeros((1, int(self.cfg_unscanned.max_target_length)), dtype=np.int32)
    ids_pad[0, :short_seq_len] = tokens_np
    pos_pad = np.broadcast_to(
        np.arange(int(self.cfg_unscanned.max_target_length), dtype=np.int32)[None, :],
        ids_pad.shape,
    ).copy()
    seg_pad = np.zeros_like(ids_pad, dtype=np.int32)
    seg_pad[0, :short_seq_len] = 1

    captured_layer_topk: list[np.ndarray] = []
    orig_routing = moe.RoutedMoE.deepseek_routing

    def _capture_routing(
        self_moe: moe.RoutedMoE,
        gate_logits: jax.Array,
        pre_bias_logits: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      weights, indices = orig_routing(self_moe, gate_logits, pre_bias_logits)
      captured_layer_topk.append(np.asarray(indices[0, :short_seq_len, :], dtype=np.uint16))
      return weights, indices

    moe.RoutedMoE.deepseek_routing = _capture_routing  # type: ignore[method-assign]
    try:
      clean_full_logits = np.asarray(
          self.mt_unscanned(
              decoder_input_tokens=jnp.asarray(ids_pad, dtype=jnp.int32),
              decoder_positions=jnp.asarray(pos_pad, dtype=jnp.int32),
              decoder_segment_ids=jnp.asarray(seg_pad, dtype=jnp.int32),
              encoder_images=jnp.asarray(padded_4_slots[None, ...], dtype=jnp.float32),
              encoder_image_masks=jnp.asarray(image_mask_slot1[None, :], dtype=jnp.int32),
              enable_dropout=False,
          )[0, :short_seq_len, :],
          dtype=np.float32,
      )
    finally:
      moe.RoutedMoE.deepseek_routing = orig_routing  # type: ignore[method-assign]

    first_dense = int(self.cfg_unscanned.first_num_dense_layers)
    self.assertEqual(len(captured_layer_topk), num_layers - first_dense)
    ref_topk_experts = np.zeros((num_layers, short_seq_len, top_k), dtype=np.uint16)
    for moe_idx, layer_arr in enumerate(captured_layer_topk, start=first_dense):
      ref_topk_experts[moe_idx] = layer_arr

    # Verify zero-filled dense prefix layers (e.g., layer 0 on 40L 749B where moe_layer_ids=1..num_layers-1)
    # pass prepare_padded_ref_topk_experts cleanly while duplicate indices in an MoE layer raise ValueError.
    dense_prefix_experts = ref_topk_experts.copy()
    dense_prefix_experts[0] = 0
    padded_dense_prefix = deepseek_v41_frozen_topk.prepare_padded_ref_topk_experts(
        dense_prefix_experts,
        sample_name="L1",
        num_layers=num_layers,
        seq_len=short_seq_len,
        padded_seq_len=int(self.cfg_unscanned.max_target_length),
        batch_size=4,
        top_k=top_k,
        num_experts=num_experts,
        moe_layer_ids=list(range(1, num_layers)),
    )
    self.assertEqual(
        padded_dense_prefix.shape,
        (num_layers, 4, int(self.cfg_unscanned.max_target_length), top_k),
    )

    # Validate negative controls on validate_raw_ref_topk_experts and prepare_padded_ref_topk_experts.
    with self.assertRaises(ValueError):
      deepseek_v41_frozen_topk.validate_raw_ref_topk_experts(
          ref_topk_experts.astype(np.float32), sample_name="L1", expected_seq_len=short_seq_len
      )
    with self.assertRaises(ValueError):
      deepseek_v41_frozen_topk.validate_raw_ref_topk_experts(
          ref_topk_experts[:, :-1, :], sample_name="L1", expected_seq_len=short_seq_len
      )
    bad_oob = ref_topk_experts.copy()
    bad_oob[1, 0, 0] = num_experts
    with self.assertRaises(ValueError):
      deepseek_v41_frozen_topk.prepare_padded_ref_topk_experts(
          bad_oob,
          sample_name="L1",
          num_layers=num_layers,
          seq_len=short_seq_len,
          padded_seq_len=int(self.cfg_unscanned.max_target_length),
          batch_size=4,
          top_k=top_k,
          num_experts=num_experts,
          moe_layer_ids=list(range(first_dense, num_layers)),
      )
    bad_dup = ref_topk_experts.copy()
    bad_dup[1, 0, 1] = bad_dup[1, 0, 0]
    with self.assertRaises(ValueError):
      deepseek_v41_frozen_topk.prepare_padded_ref_topk_experts(
          bad_dup,
          sample_name="L1",
          num_layers=num_layers,
          seq_len=short_seq_len,
          padded_seq_len=int(self.cfg_unscanned.max_target_length),
          batch_size=4,
          top_k=top_k,
          num_experts=num_experts,
          moe_layer_ids=list(range(first_dense, num_layers)),
      )

    # Test BOTH scan_layers=False (unscanned) and scan_layers=True (scanned) end-to-end under jax.jit
    # across BOTH forward_pass_logit_checker and masked_prefill_runner!
    diag_root = self.tmp_root / "frozen_topk_e2e"
    diag_root.mkdir(parents=True, exist_ok=True)
    clean_npz_dir = diag_root / "clean_mm_npz"
    clean_npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        clean_npz_dir / "M1.npz",
        sample_id=np.array("M1"),
        tokens=tokens_np,
        logits=clean_full_logits,
        patches=padded_4_slots,
        image_masks=image_mask_slot1,
        grid_hw=np.array([[6, 6]], dtype=np.int32),
        token_types=token_types,
        num_images=np.array(1, dtype=np.int32),
        ref_topk_experts=ref_topk_experts,
    )

    bias_ramp = jnp.linspace(-30.0, 30.0, num_experts, dtype=jnp.float32)

    for layout_label, model, cfg in (
        ("unscanned", self.mt_unscanned, self.cfg_unscanned),
        ("scanned", self.mt_scanned, self.cfg_scanned),
    ):
      with self.subTest(layout=layout_label):
        fwd_pass_summary = diag_root / f"fwd_pass_{layout_label}.json"
        fwd_args = argparse.Namespace(
            atol=None,
            rtol=1e-5,
            token_size=None,
            max_kl_div=None,
            golden_logits_path="",
            mm_golden_logits_path=clean_npz_dir.as_posix(),
            thresholds_json=_THRESHOLDS_JSON_PATH.as_posix(),
            enable_long_text_gate=False,
            max_images_per_sample=4,
            run_ds41_mm_audit=False,
            hf_checkpoint_dir_for_audit="",
            export_orbax_dir="",
            summary_json=fwd_pass_summary.as_posix(),
            hf_model_path="",
            run_hf_model=False,
            output_logits_path="",
            gcs_output_logits_path="",
            clip_logits_epsilon=None,
            skip_first_token=False,
            ckpt_type="nnx",
            trust_remote_code=True,
            reference_model="",
        )
        mp_pass_summary = diag_root / f"mp_pass_{layout_label}.json"
        mp_args = argparse.Namespace(
            atol=None,
            max_kl_div=0.08,
            min_cos_sim=0.999,
            golden_logits_path="",
            mm_golden_dir=clean_npz_dir.as_posix(),
            max_images_per_sample=4,
            output_json=mp_pass_summary.as_posix(),
            hf_model_path="",
            run_hf_model=False,
            reference_model="",
            output_logits_path=(diag_root / f"mp_pass_{layout_label}.jsonl").as_posix(),
            gcs_output_logits_path="",
            clip_logits_epsilon=None,
            enforce_length_bucket_gates=True,
            skip_first_token=False,
            ckpt_type="nnx",
            trust_remote_code=True,
            token_size=None,
        )

        # 1. Passing gate: FrozenTopK MUST NOT run (zero extra compile / forward pass).
        with mock.patch.object(
            deepseek_v41_frozen_topk,
            "run_frozen_topk_forward_nnx",
            wraps=deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx,
        ) as spy_frozen:
          fwd_ok = logit_runner.main(cfg, fwd_args, preloaded_model=model)
          mp_ok = masked_prefill_runner.main(cfg, mp_args, preloaded_model=model)
          self.assertEqual(spy_frozen.call_count, 0)
        self.assertTrue(fwd_ok["passed"])
        self.assertEqual(
            fwd_ok["samples"][0]["frozen_topk_diagnostic"],
            {"ran": False, "reason": "text_positions_passed"},
        )
        self.assertTrue(mp_ok[0]["passed"])
        self.assertEqual(
            mp_ok[0]["frozen_topk_diagnostic"],
            {"ran": False, "reason": "text_positions_passed"},
        )

        # 2. Perturb ONLY selection biases (gate.bias and gate.bias_vl) so unforced top-k routing flips
        #    and fails text_positions gates, while unbiased pre_bias_logits stays 100% unchanged!
        saved_biases: list[tuple[Any, Any, jax.Array, jax.Array]] = []
        if bool(cfg.scan_layers):
          for seg_idx in range(10):
            seg_attr = f"segments_{seg_idx}"
            if not hasattr(model.decoder, seg_attr):
              break
            stack = getattr(model.decoder, seg_attr)
            for sub_idx in range(10):
              layer_attr = f"layers_{sub_idx}"
              if not hasattr(stack, layer_attr):
                break
              sub_layer = getattr(stack, layer_attr)
              if hasattr(sub_layer, "mlp") and hasattr(sub_layer.mlp, "MoeBlock_0"):
                moe_mod = sub_layer.mlp.MoeBlock_0
                gate_mod = moe_mod.gate
                orig_b = gate_mod.bias[...]
                orig_bvl = gate_mod.bias_vl[...]
                saved_biases.append((moe_mod, gate_mod, orig_b, orig_bvl))
                b_sharding = orig_b.sharding
                bvl_sharding = orig_bvl.sharding
                delta = jnp.broadcast_to(bias_ramp, orig_b.shape)
                gate_mod.bias.set_value(jax.device_put(orig_b + delta, b_sharding))
                gate_mod.bias_vl.set_value(jax.device_put(orig_bvl - delta, bvl_sharding))
        else:
          for layer_idx in range(num_layers):
            layer = getattr(model.decoder, f"layers_{layer_idx}")
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "MoeBlock_0"):
              moe_mod = layer.mlp.MoeBlock_0
              gate_mod = moe_mod.gate
              orig_b = gate_mod.bias[...]
              orig_bvl = gate_mod.bias_vl[...]
              saved_biases.append((moe_mod, gate_mod, orig_b, orig_bvl))
              gate_mod.bias.set_value(jax.device_put(orig_b + bias_ramp, orig_b.sharding))
              gate_mod.bias_vl.set_value(jax.device_put(orig_bvl - bias_ramp, orig_bvl.sharding))

        self.assertGreater(len(saved_biases), 0)
        try:
          fwd_fail_args = copy.copy(fwd_args)
          fwd_fail_summary_path = diag_root / f"fwd_fail_{layout_label}.json"
          fwd_fail_args.summary_json = fwd_fail_summary_path.as_posix()

          mp_fail_args = copy.copy(mp_args)
          mp_fail_summary_path = diag_root / f"mp_fail_{layout_label}.json"
          mp_fail_args.output_json = mp_fail_summary_path.as_posix()
          mp_fail_args.output_logits_path = (diag_root / f"mp_fail_{layout_label}.jsonl").as_posix()

          with mock.patch.object(
              deepseek_v41_frozen_topk,
              "run_frozen_topk_forward_nnx",
              wraps=deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx,
          ) as spy_fail:
            with self.assertRaises(AssertionError):
              logit_runner.main(cfg, fwd_fail_args, preloaded_model=model)
            self.assertEqual(spy_fail.call_count, 1)

            with self.assertRaises(AssertionError):
              masked_prefill_runner.main(cfg, mp_fail_args, preloaded_model=model)
            self.assertEqual(spy_fail.call_count, 2)

          for summary_path in (fwd_fail_summary_path, mp_fail_summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
              fail_payload = json.load(f)
            # FrozenTopK NEVER flips passed from False to True!
            self.assertFalse(fail_payload["passed"])
            self.assertFalse(fail_payload["all_passed"])
            sample_out = fail_payload["samples"][0]
            self.assertFalse(sample_out["passed"])
            self.assertGreater(sample_out["text_positions_metrics"]["max_kl"], 0.05)
            diag_out = sample_out["frozen_topk_diagnostic"]
            self.assertIsNotNone(diag_out)
            self.assertTrue(diag_out["ran"])
            # Teacher-forced ref_topk_experts collapses KL divergence back to ~0 across all scopes!
            self.assertLessEqual(diag_out["text_positions_metrics"]["max_kl"], 1e-5)
            self.assertLessEqual(diag_out["image_positions_metrics"]["max_kl"], 1e-5)
            self.assertLessEqual(diag_out["all_positions_metrics"]["max_kl"], 1e-5)
        finally:
          for _, gate_mod, orig_b, orig_bvl in saved_biases:
            gate_mod.bias.set_value(orig_b)
            gate_mod.bias_vl.set_value(orig_bvl)

        # Verify model state and class method were restored cleanly.
        self.assertIs(moe.RoutedMoE.deepseek_routing, orig_routing)
        for moe_mod, _, _, _ in saved_biases:
          self.assertFalse(hasattr(moe_mod, "_frozen_topk_indices"))

  def _stage_clean_frozen_topk_sample_dict(self, sample_id: str = "M1", short_seq_len: int = 36) -> dict[str, Any]:
    """Runs `run_natural_routing_forward_with_trace_nnx` on `mt_unscanned` and builds a clean `mm_v3` sample."""
    batch = multimodal.build_mini_vl_batch(
        self.ref_args,
        grid=multimodal.MINI_GRID_PADDED,
        batch_size=1,
        seq_len=short_seq_len,
        span_starts=((2,),),
        seed=2026,
    )
    tokens_np = batch.input_ids.numpy().astype(np.int32)[0]
    token_types = batch.token_types.numpy().astype(np.int32)[0]
    valid_patch_0 = batch.patches_bnpd.numpy().astype(np.float32)[0]
    padded_4_slots = np.zeros((4, valid_patch_0.shape[1], valid_patch_0.shape[2]), dtype=np.float32)
    padded_4_slots[0] = valid_patch_0[0]
    image_mask_slot1 = np.array([1, 0, 0, 0], dtype=np.int32)

    max_len = int(self.cfg_unscanned.max_target_length)
    gbs = int(self.cfg_unscanned.global_batch_size_to_train_on)
    ids_1 = np.zeros((1, max_len), dtype=np.int32)
    ids_1[0, :short_seq_len] = tokens_np
    ids_b = jnp.tile(jnp.asarray(ids_1, dtype=jnp.int32), (gbs, 1))
    pos_b = jnp.broadcast_to(jnp.arange(max_len, dtype=jnp.int32)[None, :], (gbs, max_len))
    seg_1 = np.zeros((1, max_len), dtype=np.int32)
    seg_1[0, :short_seq_len] = 1
    seg_b = jnp.tile(jnp.asarray(seg_1, dtype=jnp.int32), (gbs, 1))
    imgs_b = jnp.tile(jnp.asarray(padded_4_slots[None, ...], dtype=jnp.float32), (gbs, 1, 1, 1))
    masks_b = jnp.tile(jnp.asarray(image_mask_slot1[None, :], dtype=jnp.int32), (gbs, 1))

    nat_res = deepseek_v41_frozen_topk.run_natural_routing_forward_with_trace_nnx(
        self.mt_unscanned,
        self.cfg_unscanned,
        self.mesh,
        sample_name=sample_id,
        ids=ids_b,
        decoder_positions=pos_b,
        decoder_segment_ids=seg_b,
        images=imgs_b,
        image_masks=masks_b,
    )
    clean_logits = np.asarray(nat_res.logits[0, :short_seq_len, :], dtype=np.float32)
    ref_topk_experts = np.asarray(nat_res.tpu_moe_topk[:, 0, :short_seq_len, :], dtype=np.uint16)
    num_layers = int(ref_topk_experts.shape[0])
    ref_gate_boundary_ulp = np.full((num_layers, short_seq_len), 1.0e-7, dtype=np.float32)
    ref_gate_margins = np.full((num_layers, short_seq_len), 20.0 * 1.0e-7, dtype=np.float32)
    ref_gate_margins[0, :] = 0.0
    ref_gate_boundary_ulp[0, :] = 0.0

    sorted_idx_layers = sorted(nat_res.tpu_indexer_topk.keys())
    ref_indexer_layer_ids = np.asarray(sorted_idx_layers, dtype=np.int32)
    tpu_indexer_topk_0based = np.stack(
        [np.asarray(nat_res.tpu_indexer_topk[lid][0, :short_seq_len, :], dtype=np.int32) for lid in sorted_idx_layers],
        axis=0,
    )
    ref_indexer_topk = np.where(
        tpu_indexer_topk_0based >= 0,
        tpu_indexer_topk_0based + int(short_seq_len),
        -1,
    ).astype(np.int32)
    n_idx_layers = int(ref_indexer_layer_ids.shape[0])
    ref_indexer_boundary_ulp = np.full((n_idx_layers, short_seq_len), 1.0e-6, dtype=np.float32)
    ref_indexer_margins = np.full((n_idx_layers, short_seq_len), 20.0 * 1.0e-6, dtype=np.float32)

    return {
        "sample_id": np.array(sample_id),
        "tokens": tokens_np,
        "logits": clean_logits,
        "patches": padded_4_slots,
        "image_masks": image_mask_slot1,
        "grid_hw": np.array([[6, 6]], dtype=np.int32),
        "token_types": token_types,
        "num_images": np.array(1, dtype=np.int32),
        "ref_topk_experts": ref_topk_experts,
        "ref_gate_margins": ref_gate_margins,
        "ref_gate_boundary_ulp": ref_gate_boundary_ulp,
        "ref_indexer_layer_ids": ref_indexer_layer_ids,
        "ref_indexer_topk": ref_indexer_topk,
        "ref_indexer_margins": ref_indexer_margins,
        "ref_indexer_boundary_ulp": ref_indexer_boundary_ulp,
    }

  def test_router_agreement_fails_on_high_margin_expert_or_indexer_corruption(self) -> None:
    """Negative control: high-margin (>= 8 ULP) MoE or Indexer routing flip fails closed even when FrozenTopK KL is 0."""
    base_sample = self._stage_clean_frozen_topk_sample_dict("M1", short_seq_len=36)
    nc1_root = self.tmp_root / "frozen_topk_nc1"
    clean_dir = nc1_root / "mm_v3_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    np.savez(clean_dir / "M1.npz", **base_sample)

    low_margin_dir = nc1_root / "mm_v3_low_margin_flip"
    low_margin_dir.mkdir(parents=True, exist_ok=True)
    low_sample = copy.deepcopy(base_sample)
    num_experts = int(self.cfg_unscanned.num_experts)
    cur_set = set(int(x) for x in low_sample["ref_topk_experts"][1, 5, :])
    alt_expert = next(e for e in range(num_experts) if e not in cur_set)
    low_sample["ref_topk_experts"][1, 5, -1] = np.uint16(alt_expert)
    # Margin 2.0 * boundary_ulp < 8.0 * boundary_ulp -> low-margin boundary flip is tolerated by router agreement
    # while FrozenTopK teacher-forces the uncorrupted reference routing in both runs when logits match!
    # To keep teacher-forced KL at 0 for the low-margin unit check on check_router_agreement:
    tpu_idx_topk_0based = np.where(
        base_sample["ref_indexer_topk"] >= 0,
        base_sample["ref_indexer_topk"] - 36,
        -1,
    ).astype(np.int32)
    res_low = deepseek_v41_frozen_topk.check_router_agreement(
        sample_name="M1",
        seq_len=36,
        token_types=base_sample["token_types"],
        tpu_moe_topk=base_sample["ref_topk_experts"],
        ref_topk_experts=low_sample["ref_topk_experts"],
        ref_gate_margins=np.where(
            np.arange(int(base_sample["ref_topk_experts"].shape[0]))[:, None] == 1,
            2.0 * base_sample["ref_gate_boundary_ulp"],
            base_sample["ref_gate_margins"],
        ).astype(np.float32),
        ref_gate_boundary_ulp=base_sample["ref_gate_boundary_ulp"],
        tpu_indexer_topk=tpu_idx_topk_0based,
        ref_indexer_layer_ids=base_sample["ref_indexer_layer_ids"],
        ref_indexer_topk=base_sample["ref_indexer_topk"],
        ref_indexer_margins=base_sample["ref_indexer_margins"],
        ref_indexer_boundary_ulp=base_sample["ref_indexer_boundary_ulp"],
        ulp_multiplier=8.0,
        raise_on_failure=False,
    )
    self.assertTrue(res_low["passed"])
    self.assertEqual(res_low["moe_router"]["mismatches_above_threshold"], 0)
    self.assertEqual(res_low["moe_router"]["mismatches_below_threshold"], 1)

    # Unconditional 7-key requirement negative control:
    # Omitting any of the 4 indexer keys (`ref_indexer_topk`) returns False from `has_frozen_topk_routing_fields`
    # and raises `ValueError` naming the missing key in `extract_and_validate_frozen_topk_fields`.
    self.assertTrue(deepseek_v41_frozen_topk.has_frozen_topk_routing_fields(base_sample))
    missing_idx_sample = copy.deepcopy(base_sample)
    del missing_idx_sample["ref_indexer_topk"]
    self.assertFalse(deepseek_v41_frozen_topk.has_frozen_topk_routing_fields(missing_idx_sample))
    with self.assertRaisesRegex(ValueError, "ref_indexer_topk"):
      deepseek_v41_frozen_topk.extract_and_validate_frozen_topk_fields(
          missing_idx_sample,
          sample_name="M1",
          expected_seq_len=36,
          is_multimodal=True,
          require_frozen_topk=True,
      )

    # High-margin indexer corruption (margin = 20 * boundary_ulp >= 8 * boundary_ulp):
    # Note that at position s=35 (last token), changing an earlier or later indexer key or MoE expert at a
    # position that does not alter logits (or corrupting ref_indexer_topk at position s=35 while keeping
    # logits fixed to the forced run) must still fail check_router_agreement!
    bad_idx_sample = copy.deepcopy(base_sample)
    # Pick a valid reference-offset key index in [36, 72) not currently in ref_indexer_topk[0, 35, :]
    cur_idx_set = set(int(x) for x in bad_idx_sample["ref_indexer_topk"][0, 35, :] if int(x) >= 0)
    replacement_key = next(k for k in range(36, 72) if k not in cur_idx_set)
    bad_idx_sample["ref_indexer_topk"][0, 35, -1] = np.int32(replacement_key)
    res_bad_idx = deepseek_v41_frozen_topk.check_router_agreement(
        sample_name="M1",
        seq_len=36,
        token_types=base_sample["token_types"],
        tpu_moe_topk=base_sample["ref_topk_experts"],
        ref_topk_experts=base_sample["ref_topk_experts"],
        ref_gate_margins=base_sample["ref_gate_margins"],
        ref_gate_boundary_ulp=base_sample["ref_gate_boundary_ulp"],
        tpu_indexer_topk=tpu_idx_topk_0based,
        ref_indexer_layer_ids=base_sample["ref_indexer_layer_ids"],
        ref_indexer_topk=bad_idx_sample["ref_indexer_topk"],
        ref_indexer_margins=base_sample["ref_indexer_margins"],
        ref_indexer_boundary_ulp=base_sample["ref_indexer_boundary_ulp"],
        ulp_multiplier=8.0,
        raise_on_failure=False,
    )
    self.assertFalse(res_bad_idx["passed"])
    self.assertEqual(res_bad_idx["sparse_attention_indexer"]["mismatches_above_threshold"], 1)

    # End-to-end checker verification on both unscanned and scanned models:
    # Perturb MoE gate.bias on the model so natural routing flips at high-margin positions (>= 8 ULP),
    # while FrozenTopK teacher-forced logits remain 100% identical to golden (max_kl == 0)!
    num_layers = int(self.cfg_unscanned.num_decoder_layers)
    bias_ramp = jnp.linspace(-30.0, 30.0, num_experts, dtype=jnp.float32)

    for layout_label, model, cfg in (
        ("unscanned", self.mt_unscanned, self.cfg_unscanned),
        ("scanned", self.mt_scanned, self.cfg_scanned),
    ):
      with self.subTest(layout=layout_label):
        fwd_ok_summary = nc1_root / f"fwd_ok_{layout_label}.json"
        fwd_args = argparse.Namespace(
            atol=None,
            rtol=1e-5,
            token_size=None,
            max_kl_div=None,
            golden_logits_path="",
            mm_golden_logits_path=clean_dir.as_posix(),
            thresholds_json=_THRESHOLDS_JSON_PATH.as_posix(),
            enable_long_text_gate=False,
            max_images_per_sample=4,
            run_ds41_mm_audit=False,
            hf_checkpoint_dir_for_audit="",
            export_orbax_dir="",
            summary_json=fwd_ok_summary.as_posix(),
            hf_model_path="",
            run_hf_model=False,
            output_logits_path="",
            gcs_output_logits_path="",
            clip_logits_epsilon=None,
            skip_first_token=False,
            ckpt_type="nnx",
            trust_remote_code=True,
            reference_model="",
            require_frozen_topk_gate=True,
            router_agreement_ulp_multiplier=8.0,
        )
        mp_ok_summary = nc1_root / f"mp_ok_{layout_label}.json"
        mp_args = argparse.Namespace(
            atol=None,
            max_kl_div=0.08,
            min_cos_sim=0.999,
            golden_logits_path="",
            mm_golden_dir=clean_dir.as_posix(),
            max_images_per_sample=4,
            output_json=mp_ok_summary.as_posix(),
            hf_model_path="",
            run_hf_model=False,
            reference_model="",
            output_logits_path="",
            gcs_output_logits_path="",
            clip_logits_epsilon=None,
            enforce_length_bucket_gates=True,
            skip_first_token=False,
            ckpt_type="nnx",
            trust_remote_code=True,
            token_size=None,
            require_frozen_topk_gate=True,
            router_agreement_ulp_multiplier=8.0,
        )
        fwd_res = logit_runner.main(cfg, fwd_args, preloaded_model=model)
        self.assertTrue(fwd_res["passed"])
        self.assertEqual(fwd_res["samples"][0]["frozen_topk_diagnostic"]["mode"], "frozen_topk_primary_gate")
        self.assertTrue(fwd_res["samples"][0]["router_agreement"]["passed"])
        self.assertTrue(fwd_res["samples"][0]["router_agreement"]["diagnostic_only"])
        self.assertTrue(fwd_res["samples"][0]["unforced_router_agreement_summary"]["diagnostic_only"])

        mp_res = masked_prefill_runner.main(cfg, mp_args, preloaded_model=model)
        self.assertTrue(mp_res[0]["passed"])
        self.assertEqual(mp_res[0]["frozen_topk_diagnostic"]["mode"], "frozen_topk_primary_gate")
        self.assertTrue(mp_res[0]["router_agreement"]["passed"])
        self.assertTrue(mp_res[0]["router_agreement"]["diagnostic_only"])
        self.assertTrue(mp_res[0]["unforced_router_agreement_summary"]["diagnostic_only"])

        # Now perturb gate.bias so natural routing flips at high-margin positions while FrozenTopK KL stays 0!
        # In 40-layer E2E FrozenTopK runs, forced-routing text logit gates remain strict PASS/FAIL
        # while E2E router_agreement is computed, logged, and persisted in summary JSON as a diagnostic.
        saved_biases: list[tuple[Any, jax.Array, jax.Array]] = []
        if bool(cfg.scan_layers):
          for seg_idx in range(10):
            seg_attr = f"segments_{seg_idx}"
            if not hasattr(model.decoder, seg_attr):
              break
            stack = getattr(model.decoder, seg_attr)
            for sub_idx in range(10):
              layer_attr = f"layers_{sub_idx}"
              if not hasattr(stack, layer_attr):
                break
              sub_layer = getattr(stack, layer_attr)
              if hasattr(sub_layer, "mlp") and hasattr(sub_layer.mlp, "MoeBlock_0"):
                gate_mod = sub_layer.mlp.MoeBlock_0.gate
                orig_b = gate_mod.bias[...]
                orig_bvl = gate_mod.bias_vl[...]
                saved_biases.append((gate_mod, orig_b, orig_bvl))
                delta = jnp.broadcast_to(bias_ramp, orig_b.shape)
                gate_mod.bias.set_value(jax.device_put(orig_b + delta, orig_b.sharding))
                gate_mod.bias_vl.set_value(jax.device_put(orig_bvl - delta, orig_bvl.sharding))
        else:
          for layer_idx in range(num_layers):
            layer = getattr(model.decoder, f"layers_{layer_idx}")
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "MoeBlock_0"):
              gate_mod = layer.mlp.MoeBlock_0.gate
              orig_b = gate_mod.bias[...]
              orig_bvl = gate_mod.bias_vl[...]
              saved_biases.append((gate_mod, orig_b, orig_bvl))
              gate_mod.bias.set_value(jax.device_put(orig_b + bias_ramp, orig_b.sharding))
              gate_mod.bias_vl.set_value(jax.device_put(orig_bvl - bias_ramp, orig_bvl.sharding))
        try:
          fwd_diag_args = copy.copy(fwd_args)
          fwd_diag_summary = nc1_root / f"fwd_diag_{layout_label}.json"
          fwd_diag_args.summary_json = fwd_diag_summary.as_posix()
          fwd_diag_res = logit_runner.main(cfg, fwd_diag_args, preloaded_model=model)
          self.assertTrue(fwd_diag_res["passed"])
          with open(fwd_diag_summary, "r", encoding="utf-8") as f:
            fwd_diag_payload = json.load(f)
          self.assertTrue(fwd_diag_payload["passed"])
          fwd_sample_rec = fwd_diag_payload["samples"][0]
          self.assertLessEqual(fwd_sample_rec["text_positions_metrics"]["max_kl"], 1e-5)
          self.assertEqual(fwd_sample_rec["failure_reasons"], [])
          self.assertFalse(fwd_sample_rec["router_agreement"]["passed"])
          self.assertTrue(fwd_sample_rec["router_agreement"]["diagnostic_only"])
          self.assertGreater(fwd_sample_rec["router_agreement"]["high_margin_moe_failure_count"], 0)
          self.assertGreater(len(fwd_sample_rec["router_agreement"]["high_margin_moe_failures"]), 0)
          self.assertFalse(fwd_sample_rec["router_agreement"]["moe_router"]["passed"])
          self.assertFalse(fwd_sample_rec["unforced_router_agreement_summary"]["passed"])
          self.assertTrue(fwd_sample_rec["unforced_router_agreement_summary"]["diagnostic_only"])

          mp_diag_args = copy.copy(mp_args)
          mp_diag_summary = nc1_root / f"mp_diag_{layout_label}.json"
          mp_diag_args.output_json = mp_diag_summary.as_posix()
          mp_diag_res = masked_prefill_runner.main(cfg, mp_diag_args, preloaded_model=model)
          self.assertTrue(mp_diag_res[0]["passed"])
          with open(mp_diag_summary, "r", encoding="utf-8") as f:
            mp_diag_payload = json.load(f)
          self.assertTrue(mp_diag_payload["passed"])
          mp_sample_rec = mp_diag_payload["samples"][0]
          self.assertLessEqual(mp_sample_rec["text_positions_metrics"]["max_kl"], 1e-5)
          self.assertEqual(mp_sample_rec["failure_reasons"], [])
          self.assertFalse(mp_sample_rec["router_agreement"]["passed"])
          self.assertTrue(mp_sample_rec["router_agreement"]["diagnostic_only"])
          self.assertGreater(mp_sample_rec["router_agreement"]["high_margin_moe_failure_count"], 0)
          self.assertGreater(len(mp_sample_rec["router_agreement"]["high_margin_moe_failures"]), 0)
          self.assertFalse(mp_sample_rec["router_agreement"]["moe_router"]["passed"])
          self.assertFalse(mp_sample_rec["unforced_router_agreement_summary"]["passed"])
          self.assertTrue(mp_sample_rec["unforced_router_agreement_summary"]["diagnostic_only"])
        finally:
          for gate_mod, orig_b, orig_bvl in saved_biases:
            gate_mod.bias.set_value(orig_b)
            gate_mod.bias_vl.set_value(orig_bvl)

  def test_e2e_router_agreement_diagnostic_and_nondrop_guard(self) -> None:
    """Verifies E2E router/indexer diagnostic persistence and non-drop guard."""
    base_sample = self._stage_clean_frozen_topk_sample_dict("M1", short_seq_len=36)
    router_diag_root = self.tmp_root / "e2e_router_diag"
    clean_dir = router_diag_root / "mm_v3_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    np.savez(clean_dir / "M1.npz", **base_sample)

    fwd_summary = router_diag_root / "fwd_diag_moe_idx.json"
    fwd_args = argparse.Namespace(
        atol=None,
        rtol=1e-5,
        token_size=None,
        max_kl_div=None,
        golden_logits_path="",
        mm_golden_logits_path=clean_dir.as_posix(),
        thresholds_json=_THRESHOLDS_JSON_PATH.as_posix(),
        enable_long_text_gate=False,
        max_images_per_sample=4,
        run_ds41_mm_audit=False,
        hf_checkpoint_dir_for_audit="",
        export_orbax_dir="",
        summary_json=fwd_summary.as_posix(),
        hf_model_path="",
        run_hf_model=False,
        output_logits_path="",
        gcs_output_logits_path="",
        clip_logits_epsilon=None,
        skip_first_token=False,
        ckpt_type="nnx",
        trust_remote_code=True,
        reference_model="",
        require_frozen_topk_gate=True,
        router_agreement_ulp_multiplier=8.0,
    )
    mp_summary = router_diag_root / "mp_diag_moe_idx.json"
    mp_args = argparse.Namespace(
        atol=None,
        max_kl_div=0.08,
        min_cos_sim=0.999,
        golden_logits_path="",
        mm_golden_dir=clean_dir.as_posix(),
        max_images_per_sample=4,
        output_json=mp_summary.as_posix(),
        hf_model_path="",
        run_hf_model=False,
        reference_model="",
        output_logits_path="",
        gcs_output_logits_path="",
        clip_logits_epsilon=None,
        enforce_length_bucket_gates=True,
        skip_first_token=False,
        ckpt_type="nnx",
        trust_remote_code=True,
        token_size=None,
        require_frozen_topk_gate=True,
        router_agreement_ulp_multiplier=8.0,
    )

    num_experts = int(self.cfg_unscanned.num_experts)
    corrupted_moe = base_sample["ref_topk_experts"].copy()
    cur_moe_set = set(int(x) for x in corrupted_moe[1, 5, :])
    alt_expert = next(e for e in range(num_experts) if e not in cur_moe_set)
    corrupted_moe[1, 5, -1] = np.uint16(alt_expert)

    tpu_idx_0based = np.where(
        base_sample["ref_indexer_topk"] >= 0,
        base_sample["ref_indexer_topk"] - 36,
        -1,
    ).astype(np.int32)
    corrupted_idx = tpu_idx_0based.copy()
    cur_idx_set = set(int(x) for x in corrupted_idx[0, 35, :] if int(x) >= 0)
    alt_key = next(k for k in range(36) if k not in cur_idx_set)
    corrupted_idx[0, 35, -1] = np.int32(alt_key)

    orig_run_frozen = deepseek_v41_frozen_topk.run_frozen_topk_forward_nnx

    def _corrupted_capture_wrapper(*args: Any, **kwargs: Any) -> Any:
      res = orig_run_frozen(*args, **kwargs)
      bad_cap = deepseek_v41_frozen_topk.CapturedRoutingTopK(
          tpu_moe_topk=corrupted_moe[:, None, ...],
          tpu_indexer_topk=corrupted_idx[:, None, ...],
      )
      if "return_rerun" in kwargs and bool(kwargs["return_rerun"]):
        logits_1, logits_2, _ = res
        return logits_1, logits_2, bad_cap
      logits_1, _ = res
      return logits_1, bad_cap

    # (a) Diagnostic persistence when both >=8-ULP MoE and >=8-ULP Indexer disagreements occur:
    with mock.patch.object(
        deepseek_v41_frozen_topk,
        "run_frozen_topk_forward_nnx",
        side_effect=_corrupted_capture_wrapper,
    ):
      fwd_out = io.StringIO()
      with contextlib.redirect_stdout(fwd_out):
        fwd_res = logit_runner.main(self.cfg_unscanned, fwd_args, preloaded_model=self.mt_unscanned)
      self.assertTrue(fwd_res["passed"])
      self.assertIn("[DIAGNOSTIC] sample=M1 E2E router_agreement:", fwd_out.getvalue())

      mp_out = io.StringIO()
      with contextlib.redirect_stdout(mp_out):
        mp_res = masked_prefill_runner.main(self.cfg_unscanned, mp_args, preloaded_model=self.mt_unscanned)
      self.assertTrue(mp_res[0]["passed"])
      self.assertIn("[DIAGNOSTIC] sample=M1 E2E router_agreement:", mp_out.getvalue())

    for summary_file in (fwd_summary, mp_summary):
      with open(summary_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
      self.assertTrue(payload["passed"])
      sample_rec = payload["samples"][0]
      self.assertTrue(sample_rec["passed"])
      self.assertEqual(sample_rec["failure_reasons"], [])
      for ra in (
          sample_rec["router_agreement"],
          sample_rec["frozen_topk_diagnostic"]["router_agreement"],
      ):
        self.assertIsNotNone(ra)
        self.assertFalse(ra["passed"])
        self.assertTrue(ra["diagnostic_only"])
        self.assertEqual(ra["high_margin_moe_failure_count"], 1)
        self.assertEqual(len(ra["high_margin_moe_failures"]), 1)
        self.assertFalse(ra["moe_router"]["passed"])
        self.assertEqual(ra["high_margin_indexer_failure_count"], 1)
        self.assertEqual(len(ra["high_margin_indexer_failures"]), 1)
        self.assertFalse(ra["sparse_attention_indexer"]["passed"])
      for unforced_ra in (
          sample_rec["unforced_router_agreement_summary"],
          sample_rec["frozen_topk_diagnostic"]["unforced_router_agreement_summary"],
      ):
        self.assertIsNotNone(unforced_ra)
        self.assertTrue(unforced_ra["diagnostic_only"])

    # (b) Non-drop guard: when router_agreement_ulp_multiplier > 0, missing captured routing raises ValueError.
    def _missing_capture_wrapper(*args: Any, **kwargs: Any) -> Any:
      res = orig_run_frozen(*args, **kwargs)
      if "return_rerun" in kwargs and bool(kwargs["return_rerun"]):
        logits_1, logits_2, _ = res
        return logits_1, logits_2, None
      logits_1, _ = res
      return logits_1, None

    with mock.patch.object(
        deepseek_v41_frozen_topk,
        "run_frozen_topk_forward_nnx",
        side_effect=_missing_capture_wrapper,
    ):
      with self.assertRaises((ValueError, SystemExit)) as cm_fwd:
        logit_runner.main(self.cfg_unscanned, fwd_args, preloaded_model=self.mt_unscanned)
      if isinstance(cm_fwd.exception, SystemExit):
        self.assertEqual(cm_fwd.exception.code, 42)
      with self.assertRaises((ValueError, SystemExit)) as cm_mp:
        masked_prefill_runner.main(self.cfg_unscanned, mp_args, preloaded_model=self.mt_unscanned)
      if isinstance(cm_mp.exception, SystemExit):
        self.assertEqual(cm_mp.exception.code, 42)

  def test_frozen_topk_gate_fails_on_zeroed_or_scaled_vision_embedding(self) -> None:
    """Negative control: zeroing or scaling vision embeddings fails FrozenTopK primary gate even with frozen routing."""
    base_sample = self._stage_clean_frozen_topk_sample_dict("M1", short_seq_len=36)
    nc2_root = self.tmp_root / "frozen_topk_nc2"
    clean_dir = nc2_root / "mm_v3_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    np.savez(clean_dir / "M1.npz", **base_sample)

    for scale_label, scale_factor in (("zeroed", 0.0), ("scaled_2x", 2.0)):
      for layout_label, model, cfg in (
          ("unscanned", self.mt_unscanned, self.cfg_unscanned),
          ("scanned", self.mt_scanned, self.cfg_scanned),
      ):
        with self.subTest(corruption=scale_label, layout=layout_label):
          proj = model.vision_encoder.DeepSeek41VisionProjector_0
          saved_proj_vars = [
              (proj.w2.kernel, proj.w2.kernel[...]),
              (proj.w2.bias, proj.w2.bias[...]),
              (proj.image_start, proj.image_start[...]),
              (proj.image_end, proj.image_end[...]),
              (proj.image_newline, proj.image_newline[...]),
          ]
          for var_obj, orig_val in saved_proj_vars:
            var_obj.set_value(
                jax.device_put((orig_val * jnp.float32(scale_factor)).astype(orig_val.dtype), orig_val.sharding)
            )
          try:
            fwd_summary = nc2_root / f"fwd_{scale_label}_{layout_label}.json"
            fwd_args = argparse.Namespace(
                atol=None,
                rtol=1e-5,
                token_size=None,
                max_kl_div=None,
                golden_logits_path="",
                mm_golden_logits_path=clean_dir.as_posix(),
                thresholds_json=_THRESHOLDS_JSON_PATH.as_posix(),
                enable_long_text_gate=False,
                max_images_per_sample=4,
                run_ds41_mm_audit=False,
                hf_checkpoint_dir_for_audit="",
                export_orbax_dir="",
                summary_json=fwd_summary.as_posix(),
                hf_model_path="",
                run_hf_model=False,
                output_logits_path="",
                gcs_output_logits_path="",
                clip_logits_epsilon=None,
                skip_first_token=False,
                ckpt_type="nnx",
                trust_remote_code=True,
                reference_model="",
                require_frozen_topk_gate=True,
                router_agreement_ulp_multiplier=8.0,
            )
            with self.assertRaises(AssertionError):
              logit_runner.main(cfg, fwd_args, preloaded_model=model)
            with open(fwd_summary, "r", encoding="utf-8") as f:
              fwd_payload = json.load(f)
            self.assertFalse(fwd_payload["passed"])
            self.assertGreater(fwd_payload["samples"][0]["text_positions_metrics"]["max_kl"], 0.01)

            mp_summary = nc2_root / f"mp_{scale_label}_{layout_label}.json"
            mp_args = argparse.Namespace(
                atol=None,
                max_kl_div=0.08,
                min_cos_sim=0.999,
                golden_logits_path="",
                mm_golden_dir=clean_dir.as_posix(),
                max_images_per_sample=4,
                output_json=mp_summary.as_posix(),
                hf_model_path="",
                run_hf_model=False,
                reference_model="",
                output_logits_path="",
                gcs_output_logits_path="",
                clip_logits_epsilon=None,
                enforce_length_bucket_gates=True,
                skip_first_token=False,
                ckpt_type="nnx",
                trust_remote_code=True,
                token_size=None,
                require_frozen_topk_gate=True,
                router_agreement_ulp_multiplier=8.0,
            )
            with self.assertRaises(AssertionError):
              masked_prefill_runner.main(cfg, mp_args, preloaded_model=model)
            with open(mp_summary, "r", encoding="utf-8") as f:
              mp_payload = json.load(f)
            self.assertFalse(mp_payload["passed"])
            self.assertGreater(mp_payload["samples"][0]["text_positions_metrics"]["max_kl"], 0.01)
          finally:
            for var_obj, orig_val in saved_proj_vars:
              var_obj.set_value(orig_val)

  def test_frozen_topk_gate_fails_on_shifted_image_token_positions(self) -> None:
    """Negative control: shifting `<|image|>` token positions by +1 fails FrozenTopK primary gate."""
    base_sample = self._stage_clean_frozen_topk_sample_dict("M1", short_seq_len=36)
    shifted_sample = copy.deepcopy(base_sample)
    # Shift tokens and token_types by +1 starting at pos 1 so the image span [2..15] moves to [3..16].
    shifted_tokens = base_sample["tokens"].copy()
    shifted_tt = base_sample["token_types"].copy()
    shifted_tokens[2:] = base_sample["tokens"][1:-1]
    shifted_tokens[1] = base_sample["tokens"][-1]
    shifted_tt[2:] = base_sample["token_types"][1:-1]
    shifted_tt[1] = base_sample["token_types"][-1]
    shifted_sample["tokens"] = shifted_tokens
    shifted_sample["token_types"] = shifted_tt

    nc3_root = self.tmp_root / "frozen_topk_nc3"
    shifted_dir = nc3_root / "mm_v3_shifted"
    shifted_dir.mkdir(parents=True, exist_ok=True)
    np.savez(shifted_dir / "M1.npz", **shifted_sample)

    for layout_label, model, cfg in (
        ("unscanned", self.mt_unscanned, self.cfg_unscanned),
        ("scanned", self.mt_scanned, self.cfg_scanned),
    ):
      with self.subTest(layout=layout_label):
        fwd_summary = nc3_root / f"fwd_shifted_{layout_label}.json"
        fwd_args = argparse.Namespace(
            atol=None,
            rtol=1e-5,
            token_size=None,
            max_kl_div=None,
            golden_logits_path="",
            mm_golden_logits_path=shifted_dir.as_posix(),
            thresholds_json=_THRESHOLDS_JSON_PATH.as_posix(),
            enable_long_text_gate=False,
            max_images_per_sample=4,
            run_ds41_mm_audit=False,
            hf_checkpoint_dir_for_audit="",
            export_orbax_dir="",
            summary_json=fwd_summary.as_posix(),
            hf_model_path="",
            run_hf_model=False,
            output_logits_path="",
            gcs_output_logits_path="",
            clip_logits_epsilon=None,
            skip_first_token=False,
            ckpt_type="nnx",
            trust_remote_code=True,
            reference_model="",
            require_frozen_topk_gate=True,
            router_agreement_ulp_multiplier=8.0,
        )
        with self.assertRaises(AssertionError):
          logit_runner.main(cfg, fwd_args, preloaded_model=model)
        with open(fwd_summary, "r", encoding="utf-8") as f:
          fwd_payload = json.load(f)
        self.assertFalse(fwd_payload["passed"])

        mp_summary = nc3_root / f"mp_shifted_{layout_label}.json"
        mp_args = argparse.Namespace(
            atol=None,
            max_kl_div=0.08,
            min_cos_sim=0.999,
            golden_logits_path="",
            mm_golden_dir=shifted_dir.as_posix(),
            max_images_per_sample=4,
            output_json=mp_summary.as_posix(),
            hf_model_path="",
            run_hf_model=False,
            reference_model="",
            output_logits_path="",
            gcs_output_logits_path="",
            clip_logits_epsilon=None,
            enforce_length_bucket_gates=True,
            skip_first_token=False,
            ckpt_type="nnx",
            trust_remote_code=True,
            token_size=None,
            require_frozen_topk_gate=True,
            router_agreement_ulp_multiplier=8.0,
        )
        with self.assertRaises(AssertionError):
          masked_prefill_runner.main(cfg, mp_args, preloaded_model=model)
        with open(mp_summary, "r", encoding="utf-8") as f:
          mp_payload = json.load(f)
        self.assertFalse(mp_payload["passed"])

  def test_stage_script_default_mm_golden_gcs_prefix_and_override(self) -> None:
    """Verifies `2_test_deepseek41_mm.sh` GCS golden staging, override, local dir, and resume skipping."""
    with tempfile.TemporaryDirectory(prefix="ds41_mm_v3_stage_") as tmp_dir:
      tmp_root = Path(tmp_dir)
      shim_bin = tmp_root / "bin"
      shim_bin.mkdir(parents=True, exist_ok=True)
      gcloud_log = tmp_root / "gcloud_calls.jsonl"
      py_log = tmp_root / "python_calls.jsonl"

      gcs_src_dir = tmp_root / "gcs_mm_v3_src"
      gcs_src_dir.mkdir(parents=True, exist_ok=True)
      for sid in ("L1", "M1"):
        np.savez(
            gcs_src_dir / f"{sid}.npz",
            sample_id=np.array(sid),
            ref_topk_experts=np.zeros((2, 4, 2), dtype=np.uint16),
        )
      with (gcs_src_dir / "mm_logit.pkl").open("wb") as f:
        pickle.dump([{"sample_id": "L1"}], f)
      (gcs_src_dir / "manifest.json").write_text(
          json.dumps({"golden_version": "mm_v3", "schema_version": "ds41_mm_golden_v3"}),
          encoding="utf-8",
      )

      shim_py = shim_bin / "python3"
      shim_py.write_text(
          f"""#!{sys.executable}
import json
from pathlib import Path
import sys
if len(sys.argv) > 1 and sys.argv[1] == "-":
  import os
  os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
with open({py_log.as_posix()!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)
if len(sys.argv) > 1 and "deepseek_v41_mm_audit.py" in sys.argv[1]:
  import os
  os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
for arg in sys.argv[1:]:
  if arg.startswith("--summary_json=") or arg.startswith("--output_json="):
    p = Path(arg.split("=", 1)[1])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({{"passed": True, "all_passed": True, "num_passed": 1, "total_samples": 1}}),
        encoding="utf-8",
    )
sys.exit(0)
""",
          encoding="utf-8",
      )
      shim_py.chmod(stat.S_IRWXU)

      shim_gcloud = shim_bin / "gcloud"
      shim_gcloud.write_text(
          f"""#!{sys.executable}
import json
from pathlib import Path
import shutil
import sys
with open({gcloud_log.as_posix()!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)
if len(sys.argv) >= 4 and sys.argv[1] == "storage" and sys.argv[2] == "cp":
  dst = Path(sys.argv[-1])
  if not str(dst).startswith("gs://"):
    dst.mkdir(parents=True, exist_ok=True)
    for item in Path({gcs_src_dir.as_posix()!r}).iterdir():
      shutil.copy2(item, dst / item.name)
sys.exit(0)
""",
          encoding="utf-8",
      )
      shim_gcloud.chmod(stat.S_IRWXU)

      base_env = {
          "PATH": f"{shim_bin.as_posix()}:{os.environ['PATH']}",
          "JAX_PLATFORMS": "cpu",
          "HF_CKPT_PATH": self.hf_ckpt_dir.as_posix(),
          "GOLDEN_TEXT_PKL": (tmp_root / "text.pkl").as_posix(),
          "PYTHON_BIN": shim_py.as_posix(),
      }

      def _read_py_calls() -> list[list[str]]:
        if not py_log.exists():
          return []
        return [json.loads(line) for line in py_log.read_text(encoding="utf-8").splitlines()]

      # 1. Default MM_GOLDEN_GCS_PREFIX stages once across scanned_logits, unscanned_logits, masked_prefill.
      out_default = tmp_root / "out_pos_all"
      env_pos = dict(base_env)
      env_pos["OUTPUT_DIR"] = out_default.as_posix()
      env_pos["STAGES"] = "scanned_logits,unscanned_logits,masked_prefill"
      proc_def = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()],
          env=env_pos,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_def.returncode, 0, msg=f"stderr={proc_def.stderr}\nstdout={proc_def.stdout}")
      gcloud_calls = [json.loads(line) for line in gcloud_log.read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(gcloud_calls), 1)
      self.assertEqual(
          gcloud_calls[0][:3],
          ["storage", "cp", "gs://maxtext-deepseek/deepseek-v4.1-flash/golden/mm_v3/*"],
      )
      mp_mm_dir = out_default / "masked_prefill_mm_m_only"
      self.assertTrue(deepseek_v41_frozen_topk.is_mm_v3_golden_path(mp_mm_dir))
      self.assertTrue((mp_mm_dir / "M1.npz").exists())
      self.assertFalse((mp_mm_dir / "L1.npz").exists())

      py_calls = _read_py_calls()
      self.assertEqual(len(py_calls), 3, msg=f"Expected 3 stage calls, got {py_calls}")
      self.assertEqual(py_calls[0][:2], ["-m", "tests.utils.deepseek_v41.logit_runner"])
      self.assertEqual(py_calls[1][:2], ["-m", "tests.utils.deepseek_v41.logit_runner"])
      self.assertEqual(py_calls[2][:2], ["-m", "tests.utils.deepseek_v41.masked_prefill_runner"])
      for call in py_calls:
        self.assertIn("--require_frozen_topk_gate=true", call)

      # 2. Custom MM_GOLDEN_GCS_PREFIX override on STAGES=masked_prefill.
      gcloud_log.unlink()
      py_log.unlink()
      env_ovr = dict(base_env)
      env_ovr["OUTPUT_DIR"] = (tmp_root / "out_pos_ovr").as_posix()
      env_ovr["STAGES"] = "masked_prefill"
      env_ovr["MM_GOLDEN_GCS_PREFIX"] = "gs://custom-dev-bucket/deepseek-v4.1-flash/golden/mm_v3/"
      proc_ovr = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()],
          env=env_ovr,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_ovr.returncode, 0, msg=f"stderr={proc_ovr.stderr}\nstdout={proc_ovr.stdout}")
      gcloud_calls_ovr = [json.loads(line) for line in gcloud_log.read_text(encoding="utf-8").splitlines()]
      self.assertEqual(
          gcloud_calls_ovr[0][:3],
          ["storage", "cp", "gs://custom-dev-bucket/deepseek-v4.1-flash/golden/mm_v3/*"],
      )
      self.assertEqual(len(_read_py_calls()), 1)

      # 3. Explicit local GOLDEN_MM_DIR skips gcloud cp cleanly.
      gcloud_log.unlink()
      py_log.unlink()
      env_local_mp = dict(base_env)
      env_local_mp["OUTPUT_DIR"] = (tmp_root / "out_pos_local_mp").as_posix()
      env_local_mp["STAGES"] = "masked_prefill"
      env_local_mp["GOLDEN_MM_DIR"] = gcs_src_dir.as_posix()
      proc_local_mp = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()],
          env=env_local_mp,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_local_mp.returncode, 0, msg=f"stderr={proc_local_mp.stderr}\nstdout={proc_local_mp.stdout}")
      self.assertFalse(gcloud_log.exists())
      self.assertTrue(
          deepseek_v41_frozen_topk.is_mm_v3_golden_path(tmp_root / "out_pos_local_mp" / "masked_prefill_mm_m_only")
      )
      self.assertEqual(len(_read_py_calls()), 1)

      # 4. RESUME_PASSED_STAGES=1 skips passed scanned_logits/unscanned_logits and runs masked_prefill.
      py_log.unlink()
      resume_dir = tmp_root / "resume_passed_logits"
      resume_dir.mkdir(parents=True, exist_ok=True)
      (resume_dir / "scanned_logits_summary.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
      (resume_dir / "unscanned_logits_summary.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
      env_resume = dict(base_env)
      env_resume["OUTPUT_DIR"] = (tmp_root / "out_resume").as_posix()
      env_resume["STAGES"] = "scanned_logits,unscanned_logits,masked_prefill"
      env_resume["GOLDEN_MM_DIR"] = gcs_src_dir.as_posix()
      env_resume["RESUME_PASSED_STAGES"] = "1"
      env_resume["GCS_RESUME_SUMMARY_PREFIX"] = resume_dir.as_posix()
      proc_resume = subprocess.run(
          ["bash", _STAGE_SCRIPT_PATH.as_posix()],
          env=env_resume,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_resume.returncode, 0, msg=f"stderr={proc_resume.stderr}\nstdout={proc_resume.stdout}")
      resume_calls = _read_py_calls()
      self.assertEqual(len(resume_calls), 1, msg=f"Expected only masked_prefill call, got {resume_calls}")
      self.assertEqual(resume_calls[0][:2], ["-m", "tests.utils.deepseek_v41.masked_prefill_runner"])

  def test_gcs_startup_barrier_per_attempt_and_epoch_freshness(self) -> None:
    """Verifies wait_for_workers_ready in 2_test_deepseek41_mm.sh rejects stale attempt markers.

    Ensures the barrier passes only when all workers write current attempt markers.
    """
    with tempfile.TemporaryDirectory(prefix="ds41_mm_barrier_") as tmp_dir:
      ready_dir = Path(tmp_dir) / "ready"
      ready_dir.mkdir(parents=True, exist_ok=True)
      now_epoch = int(time.time())

      def run_barrier(  # pylint: disable=too-many-positional-arguments
          num_workers: int = 4,
          worker_id: int = 0,
          run_attempt_id: str = "",
          barrier_start_epoch: int | None = None,
          max_skew_sec: int = 180,
          timeout_sec: int = 2,
          poll_sec: int = 1,
      ) -> subprocess.CompletedProcess[str]:
        if barrier_start_epoch is None:
          barrier_start_epoch = int(time.time())
        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        return subprocess.run(
            [
                "bash",
                "-c",
                f'source "{_STAGE_SCRIPT_PATH.as_posix()}" && wait_for_workers_ready "$@"',
                "--",
                ready_dir.as_posix(),
                str(num_workers),
                str(worker_id),
                run_attempt_id,
                str(barrier_start_epoch),
                str(max_skew_sec),
                str(timeout_sec),
                str(poll_sec),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

      # 1. Test 1 (stale_attempt_id_blocks):
      # Pre-populate workers 0..3 with attempt_old.
      # Workers 0..2 update with attempt_new, worker 3 remains attempt_old.
      for w in range(4):
        (ready_dir / f"worker_{w}.ready").write_text(
            json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_old", "epoch": now_epoch}),
            encoding="utf-8",
        )
      for w in range(3):
        (ready_dir / f"worker_{w}.ready").write_text(
            json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_new", "epoch": now_epoch}),
            encoding="utf-8",
        )
      res1 = run_barrier(num_workers=4, worker_id=0, run_attempt_id="attempt_new", timeout_sec=2, poll_sec=1)
      self.assertNotEqual(
          res1.returncode, 0, msg=f"Expected barrier failure with stale attempt_id, got 0. stderr={res1.stderr}"
      )
      self.assertIn("Timed out waiting for 4 workers", res1.stderr)

      # 2. Test 2 (stale_epoch_blocks_without_attempt_id):
      # Pre-populate workers 0..3 with stale epoch = now - 3600 and no attempt_id.
      # Workers 0..2 update with fresh epoch; worker 3 remains stale.
      stale_epoch = now_epoch - 3600
      for w in range(4):
        (ready_dir / f"worker_{w}.ready").write_text(
            json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "", "epoch": stale_epoch}),
            encoding="utf-8",
        )
      fresh_epoch = int(time.time())
      for w in range(3):
        (ready_dir / f"worker_{w}.ready").write_text(
            json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "", "epoch": fresh_epoch}),
            encoding="utf-8",
        )
      res2 = run_barrier(
          num_workers=4,
          worker_id=0,
          run_attempt_id="",
          barrier_start_epoch=fresh_epoch,
          max_skew_sec=180,
          timeout_sec=2,
          poll_sec=1,
      )
      self.assertNotEqual(
          res2.returncode, 0, msg=f"Expected barrier failure with stale epoch, got 0. stderr={res2.stderr}"
      )
      self.assertIn("Timed out waiting for 4 workers", res2.stderr)

      # 3. Test 3 (current_attempt_all_workers_pass):
      # Worker 3 also writes its current attempt marker with fresh epoch.
      current_epoch = int(time.time())
      for w in range(4):
        (ready_dir / f"worker_{w}.ready").write_text(
            json.dumps({"worker_id": w, "pod": f"pod-{w}", "attempt_id": "attempt_new", "epoch": current_epoch}),
            encoding="utf-8",
        )
      res3 = run_barrier(
          num_workers=4,
          worker_id=0,
          run_attempt_id="attempt_new",
          barrier_start_epoch=current_epoch,
          max_skew_sec=180,
          timeout_sec=5,
          poll_sec=1,
      )
      self.assertEqual(res3.returncode, 0, msg=f"Expected barrier pass, got {res3.returncode}. stderr={res3.stderr}")
      self.assertIn("All 4 workers ready", res3.stdout)


def _assert_129280_vocab_float64_kl_precision_and_negative_controls(test_case: unittest.TestCase) -> None:
  """Asserts 129,280-vocab float64 KL precision, legacy_fp32 dual-reporting, and 1e-6 / 1.5e-3 negative controls."""
  vocab_size = 129280
  rng = np.random.default_rng(20261026)
  # Construct 129,280-vocab float32 logits in the exact sample_1 / sample_2 regime:
  # peak logit ~13.1, log-partition ~21.5, l_inf <= 8.5e-5, and 10/10 top-10 agreement at the last token.
  row0 = rng.normal(loc=9.55, scale=0.65, size=(vocab_size,)).astype(np.float32)
  row0[0] = np.float32(13.1)
  row1 = np.full((vocab_size,), 1.33, dtype=np.float32)
  row1[0] = np.float32(13.1)
  row2 = rng.normal(loc=9.55, scale=0.65, size=(vocab_size,)).astype(np.float32)
  row2[0] = np.float32(13.1)
  golden = np.stack([row0, row1, row2], axis=0)

  delta0 = 8.0e-5
  train = golden.copy()
  train[0, 1:] = (golden[0, 1:].astype(np.float64) + delta0).astype(np.float32)
  train[1, 1:] = (golden[1, 1:].astype(np.float64) - delta0).astype(np.float32)
  train[2, 1:] = (golden[2, 1:].astype(np.float64) - 0.5 * delta0).astype(np.float32)

  l_inf = float(np.max(np.abs(train - golden)))
  test_case.assertLessEqual(l_inf, 8.5e-5)
  theoretical_kl_bound = 0.5 * (8.5e-5) ** 2 + 1e-13
  test_case.assertLess(theoretical_kl_bound, 2.0e-7)

  # Reference float64 distribution and scipy.special.kl_div ground truth.
  g64 = golden.astype(np.float64)
  t64 = train.astype(np.float64)
  g_shift = g64 - np.max(g64, axis=-1, keepdims=True)
  t_shift = t64 - np.max(t64, axis=-1, keepdims=True)
  log_Z_row0 = float(np.max(g64[0]) + np.log(np.sum(np.exp(g_shift[0]))))
  test_case.assertAlmostEqual(log_Z_row0, 21.538, delta=0.1)
  p_g64 = np.exp(g_shift - np.log(np.sum(np.exp(g_shift), axis=-1, keepdims=True)))
  p_t64 = np.exp(t_shift - np.log(np.sum(np.exp(t_shift), axis=-1, keepdims=True)))
  kl_scipy = np.maximum(np.sum(scipy.special.kl_div(p_g64, p_t64), axis=-1, dtype=np.float64), 0.0)

  # 1. Both checkers' per-token KL functions match scipy float64 to < 1e-14 and stay in [0, 0.5*(8.5e-5)^2 + 1e-13].
  kl_fplc = forward_pass_logit_checker._per_token_kl_2d(golden, train)  # pylint: disable=protected-access
  kl_mplc = masked_prefill_runner._compute_per_token_kl(golden, train)  # pylint: disable=protected-access
  test_case.assertEqual(kl_fplc.dtype, np.float64)
  test_case.assertEqual(kl_mplc.dtype, np.float64)
  np.testing.assert_allclose(kl_fplc, kl_mplc, rtol=0.0, atol=1e-15)
  test_case.assertLess(float(np.max(np.abs(kl_fplc - kl_scipy))), 1e-14)
  test_case.assertLess(float(np.max(np.abs(kl_mplc - kl_scipy))), 1e-14)
  test_case.assertGreaterEqual(float(np.min(kl_fplc)), 0.0)
  test_case.assertGreaterEqual(float(np.mean(kl_fplc)), 0.0)
  test_case.assertLessEqual(float(np.max(kl_fplc)), theoretical_kl_bound)
  test_case.assertLess(float(np.max(kl_fplc)), 2.0e-7)

  # Verify clip_logits_epsilon branch in float64 as well.
  for kl_clipped in (
      forward_pass_logit_checker._per_token_kl_2d(golden, train, clip_logits_epsilon=1e-12),  # pylint: disable=protected-access
      masked_prefill_runner._compute_per_token_kl(golden, train, clip_logits_epsilon=1e-12),  # pylint: disable=protected-access
  ):
    test_case.assertGreaterEqual(float(np.min(kl_clipped)), 0.0)
    test_case.assertLessEqual(float(np.max(kl_clipped)), theoretical_kl_bound)

  # 2. Verify scope/parity metrics pass short_text_prompts (max_kl <= 2.0e-7, last_token_top10_matches == 10) and emit
  # *_legacy_fp32.
  short_spec = {"id": "short_text_prompts", "max_kl": 2.0e-7, "last_token_top10_matches": 10}
  scope_m = logit_runner.compute_scope_metrics(train, golden)
  parity_m = masked_prefill_runner.check_mathematical_parity(train, golden, max_kl=2.0e-7)
  for m_dict in (scope_m, parity_m):
    test_case.assertGreaterEqual(float(m_dict["max_kl"]), 0.0)
    test_case.assertLessEqual(float(m_dict["max_kl"]), theoretical_kl_bound)
    test_case.assertGreaterEqual(float(m_dict["mean_kl"]), 0.0)
    for k_base in ("max_kl", "raw_primary_max_kl", "mean_kl", "p95_kl", "p99_kl"):
      test_case.assertIn(k_base, m_dict)
      test_case.assertIn(f"{k_base}_legacy_fp32", m_dict)
    # Legacy fp32 log_softmax exhibits > 9e-7 false positive spike on row 0, while float64 gate passes cleanly.
    test_case.assertGreater(float(m_dict["max_kl_legacy_fp32"]), 9.0e-7)
  legacy_rows = forward_pass_logit_checker._per_token_kl_2d_legacy_fp32(golden, train)  # pylint: disable=protected-access
  test_case.assertLess(float(np.min(legacy_rows)), 0.0)

  gates_clean = logit_runner.evaluate_thresholds(scope_m, short_spec)
  test_case.assertTrue(
      all(gates_clean.values()), msg=f"Expected clean short_text_prompts pass, got {gates_clean} ({scope_m})"
  )

  # 3. Negative Control A (Condition (c)): 1e-6-scale KL perturbation (delta = 2.828e-3 on peak logit)
  #    produces float64 max_kl ~ 1.0e-6 (0.5e-6 <= max_kl <= 1.5e-6) and FAILS short_text_prompts (2.0e-7).
  p0_row1 = float(p_g64[1, 0])
  delta_1e6 = 2.828e-3
  train_bad_1e6 = golden.copy()
  train_bad_1e6[1, 0] = np.float32(float(golden[1, 0]) + delta_1e6)
  eff_delta_1e6 = float(train_bad_1e6[1, 0]) - float(golden[1, 0])
  expected_kl_1e6 = 0.5 * p0_row1 * (1.0 - p0_row1) * (eff_delta_1e6**2)
  test_case.assertTrue(0.5e-6 <= expected_kl_1e6 <= 1.5e-6)

  scope_bad_1e6 = logit_runner.compute_scope_metrics(train_bad_1e6, golden)
  test_case.assertGreaterEqual(float(scope_bad_1e6["max_kl"]), 0.5e-6)
  test_case.assertLessEqual(float(scope_bad_1e6["max_kl"]), 1.5e-6)
  test_case.assertAlmostEqual(float(scope_bad_1e6["max_kl"]), expected_kl_1e6, delta=5e-9)
  gates_bad_1e6 = logit_runner.evaluate_thresholds(scope_bad_1e6, short_spec)
  test_case.assertFalse(gates_bad_1e6["max_kl_pass"])
  with test_case.assertRaisesRegex(AssertionError, "exceeds threshold"):
    masked_prefill_runner.check_mathematical_parity(train_bad_1e6, golden, max_kl=2.0e-7)

  # 4. Negative Control B: Perturbing a single high-probability logit by 1.5e-3 produces max_kl > 2.0e-7 and FAILS
  # short_text_prompts.
  train_bad_15e4 = golden.copy()
  train_bad_15e4[1, 0] = np.float32(float(golden[1, 0]) + 1.5e-3)
  scope_bad_15e4 = logit_runner.compute_scope_metrics(train_bad_15e4, golden)
  test_case.assertGreater(float(scope_bad_15e4["max_kl"]), 2.0e-7)
  gates_bad_15e4 = logit_runner.evaluate_thresholds(scope_bad_15e4, short_spec)
  test_case.assertFalse(gates_bad_15e4["max_kl_pass"])
  with test_case.assertRaisesRegex(AssertionError, "exceeds threshold"):
    masked_prefill_runner.check_mathematical_parity(train_bad_15e4, golden, max_kl=2.0e-7)


@pytest.mark.cpu_only
class Float64PerTokenKLPrecisionTest(unittest.TestCase):
  """Fast standalone unit test for 129,280-vocab float64 per-token KL precision and negative controls."""

  def test_per_token_kl_float64_129280_vocab_precision_and_negative_control(self) -> None:
    _assert_129280_vocab_float64_kl_precision_and_negative_controls(self)


@pytest.mark.cpu_only
class LivePreForwardAuditFailureExit42Test(unittest.TestCase):
  """Verifies live --run_ds41_mm_audit=true failure writes summary_json and exits with code 42."""

  def test_live_ds41_mm_audit_failure_writes_summary_and_exits_42(self) -> None:
    with tempfile.TemporaryDirectory(prefix="ds41_audit_fail42_") as tmp_dir:
      summary_path = os.path.join(tmp_dir, "unscanned_summary.json")
      fake_cfg = types.SimpleNamespace(
          init_weights_seed=0,
          mesh_axes=("data",),
          hf_access_token=None,
          tokenizer_path="",
          model_name="default",
          use_multimodal=True,
          load_parameters_path=tmp_dir,
      )
      parser = logit_runner.build_arg_parser()
      test_args = parser.parse_args(
          [
              "--run_ds41_mm_audit=true",
              f"--hf_checkpoint_dir_for_audit={tmp_dir}",
              f"--summary_json={summary_path}",
              "--max_kl_div=1e-4",
          ]
      )
      failed_audit = {"passed": False, "num_mismatches": 3, "reason": "injected_weight_mismatch"}
      with (
          mock.patch.object(maxtext_utils, "create_device_mesh", return_value=np.array([jax.devices()[0]])),
          mock.patch.object(deepseek_v41_mm_audit, "audit_loaded_model_against_hf", return_value=failed_audit),
      ):
        with self.assertRaises(SystemExit) as cm:
          logit_runner.run_checker(fake_cfg, test_args, preloaded_model=mock.MagicMock())
        self.assertEqual(cm.exception.code, 42)
      self.assertTrue(os.path.exists(summary_path))
      written = json.loads(Path(summary_path).read_text(encoding="utf-8"))
      self.assertFalse(written["passed"])
      self.assertFalse(written["all_passed"])
      self.assertEqual(written["audit_result"], failed_audit)

      summary_export = os.path.join(tmp_dir, "unscanned_export_summary.json")
      args_export = parser.parse_args(
          [
              "--run_ds41_mm_audit=true",
              f"--hf_checkpoint_dir_for_audit={tmp_dir}",
              f"--export_orbax_dir={os.path.join(tmp_dir, 'orbax_out')}",
              f"--summary_json={summary_export}",
              "--max_kl_div=1e-4",
          ]
      )
      with (
          mock.patch.object(maxtext_utils, "create_device_mesh", return_value=np.array([jax.devices()[0]])),
          mock.patch.object(
              deepseek_v41_mm_audit,
              "audit_loaded_model_against_hf",
              side_effect=RuntimeError("Refusing to export Orbax checkpoint: HF-to-MaxText audit failed: mismatch"),
          ),
      ):
        with self.assertRaises(SystemExit) as cm_exp:
          logit_runner.run_checker(fake_cfg, args_export, preloaded_model=mock.MagicMock())
        self.assertEqual(cm_exp.exception.code, 42)
      written_exp = json.loads(Path(summary_export).read_text(encoding="utf-8"))
      self.assertFalse(written_exp["passed"])
      self.assertFalse(written_exp["audit_result"]["passed"])


if __name__ == "__main__":
  unittest.main()
