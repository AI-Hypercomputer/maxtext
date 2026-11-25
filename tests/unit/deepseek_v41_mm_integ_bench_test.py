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

"""Unit tests and negative controls for DeepSeek-V4.1 multimodal TFLOPs, parity, and benchmark harness."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.layers import attention_deepseek41
from maxtext.trainers.pre_train import train
from maxtext.utils import max_utils, maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils.deepseek_v41.parity import e2e_check, layer_parity, negctl

try:
  import torch
  import torch.nn.functional as F

  from tests.utils.deepseek_v41_reference import multimodal

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  F = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  HAS_TORCH = False

from tests.utils import deepseek_v41_mm_testing as mm_testing

_REPO_ROOT = Path(__file__).resolve().parents[2]
_E2E_DIR = _REPO_ROOT / "tests" / "end_to_end" / "tpu" / "deepseek" / "v4.1-flash"
_BASE_CONFIG_YML = (_REPO_ROOT / "src" / "maxtext" / "configs" / "base.yml").as_posix()
_AOT_CHECK_PY = (_E2E_DIR / "aot_check.py").as_posix()
_BENCH_METRICS_PY = (_E2E_DIR / "bench_metrics.py").as_posix()
_BENCH_SCRIPT_SH = (_E2E_DIR / "3_bench_deepseek41_mm.sh").as_posix()
_SFT_EOS_PROBE_PY = (_REPO_ROOT / "tests" / "end_to_end" / "tpu" / "deepseek" / "sft_eos_probe.py").as_posix()

_VALID_SEQ_LEN = 36
# Exact 749B vision encoder TFLOPs invariant at per_device_batch_size=1.0, N=4, grid=[39, 39], freeze_vision=True.
_EXPECTED_VISION_TFLOPS = 6.5195040768
_ORIG_MATMUL_PRECISION = None


def setUpModule() -> None:
  """Enforces IEEE-754 float32 matmul precision for parity checks."""
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")


def tearDownModule() -> None:
  """Restores default matmul precision to prevent cross-test contamination."""
  jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


def _load_module(module_name: str, file_path: str) -> Any:
  """Loads a Python script under `tests/end_to_end/` by file path."""
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load module spec from {file_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _build_multimodal_train_batch(
    batch: multimodal.MiniVLBatch,
    *,
    image_token_id: int,
) -> tuple[dict[str, jax.Array], np.ndarray, np.ndarray]:
  """Builds a MaxText multimodal training batch with `targets_segmentation == 0` on image spans and padding."""
  inputs_np = batch.input_ids.numpy().astype(np.int32)
  batch_size, seq_len = inputs_np.shape
  targets_np = np.roll(inputs_np, -1, axis=1)

  inputs_seg_np = np.ones((batch_size, seq_len), dtype=np.int32)
  inputs_seg_np[:, _VALID_SEQ_LEN:] = 0

  img_mask_np = batch.image_mask.numpy()
  targets_seg_bool = (~img_mask_np) & (targets_np != image_token_id) & (inputs_seg_np != 0)
  targets_np[:, _VALID_SEQ_LEN - 1 :] = 3
  targets_seg_bool[:, _VALID_SEQ_LEN - 1 :] = False
  targets_seg_np = targets_seg_bool.astype(np.int32)

  positions_j = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
  num_images = batch.patches_bnpd.shape[1]
  mt_batch = {
      "inputs": jnp.asarray(inputs_np, dtype=jnp.int32),
      "inputs_position": positions_j,
      "inputs_segmentation": jnp.asarray(inputs_seg_np, dtype=jnp.int32),
      "targets": jnp.asarray(targets_np, dtype=jnp.int32),
      "targets_position": positions_j,
      "targets_segmentation": jnp.asarray(targets_seg_np, dtype=jnp.int32),
      "images": mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float32),
      "image_masks": jnp.ones((batch_size, num_images), dtype=jnp.int32),
  }
  return mt_batch, targets_np, targets_seg_np


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class DeepSeekV41MmIntegBenchTest(unittest.TestCase):
  """Verifies DeepSeek-V4.1 multimodal TFLOPs, parity, and benchmark harness gates."""

  def test_1_exact_749b_tflops_zero_double_counting_and_negctl(self) -> None:
    """Test 1: Dynamic 749B text + 6.5195040768 vision == multimodal TFLOPs additivity with zero double-counting."""
    common_kw: dict[str, Any] = {
        "model_name": "deepseek4.1-flash",
        "override_model_config": True,
        "run_name": "ds41_integ_tflops_test",
        "enable_checkpointing": False,
        "skip_jax_distributed_system": True,
        "compile_topology": "tpu7x-128",
        "compile_topology_num_slices": 1,
        "ici_fsdp_parallelism": 8,
        "ici_expert_parallelism": 16,
        "optimizer_memory_host_offload": True,
        "grad_dtype": "bfloat16",
        "num_vocab_tiling": 8,
        "quantization_local_shard_count": 8,
        "sparse_matmul": True,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "allow_split_physical_axes": True,
        "ragged_buffer_factor": 2.0,
        "ds41_activation_quant": False,
        "max_target_length": 4096,
        "per_device_batch_size": 1.0,
        "vision_encoder_block": "deepseek41",
        "ds41_vision_grid": [39, 39],
        "max_num_images_per_example": 4,
        "log_period": 1,
    }
    cfg_text = pyconfig.initialize(
        [sys.argv[0], _BASE_CONFIG_YML],
        **common_kw,
        use_multimodal=False,
        freeze_vision_encoder_params=False,
    )
    cfg_mm = pyconfig.initialize(
        [sys.argv[0], _BASE_CONFIG_YML],
        **common_kw,
        use_multimodal=True,
        freeze_vision_encoder_params=True,
    )

    tflops_text, lw_text, attn_text = maxtext_utils.calculate_tflops_training_per_device(cfg_text, log=False)
    tflops_mm, lw_mm, attn_mm = maxtext_utils.calculate_tflops_training_per_device(cfg_mm, log=False)
    vision_tot, vision_lw, vision_attn = maxtext_utils.calculate_vision_encoder_tflops(cfg_mm)

    self.assertGreater(tflops_text, 0.0)
    self.assertAlmostEqual(vision_tot, _EXPECTED_VISION_TFLOPS, places=10)

    # Assert exact float64 additivity (tflops_mm == float(tflops_text + vision_tot) with 0.0 float diff).
    self.assertEqual(tflops_mm, float(tflops_text + vision_tot))
    self.assertEqual(lw_mm, float(lw_text + vision_lw))
    self.assertEqual(attn_mm, float(attn_text + vision_attn))
    np.testing.assert_allclose(tflops_mm - tflops_text, vision_tot, rtol=0.0, atol=1e-12)

    # Negative control: simulated double-counting (+2 * vision_tot) must fail both exact MM check and diff check.
    double_counted_mm = float(tflops_text + 2.0 * vision_tot)
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(double_counted_mm - tflops_text, vision_tot, rtol=0.0, atol=1e-6)

    # Verify bench_metrics.py expected TFLOPs helper matches dynamic text + fixed vision additivity across 64c/128c.
    bench_mod = _load_module("ds41_bench_metrics_tflops_check", _BENCH_METRICS_PY)
    for topo_mode in ("64c", "128c"):
      txt_res = bench_mod.compute_expected_per_device_tflops(
          use_multimodal=False,
          topology_mode=topo_mode,
          remat_policy="full",
      )
      mm_res = bench_mod.compute_expected_per_device_tflops(
          use_multimodal=True,
          topology_mode=topo_mode,
          remat_policy="full",
      )
      self.assertAlmostEqual(txt_res["total_per_device_tflops"], tflops_text, places=10)
      self.assertAlmostEqual(mm_res["vision_per_device_tflops"], _EXPECTED_VISION_TFLOPS, places=10)
      self.assertEqual(
          mm_res["total_per_device_tflops"],
          float(txt_res["total_per_device_tflops"] + mm_res["vision_per_device_tflops"]),
      )
      self.assertAlmostEqual(mm_res["total_per_device_tflops"], tflops_mm, places=10)

    if Path(_SFT_EOS_PROBE_PY).exists():
      sft_probe_mod = _load_module("ds41_sft_eos_probe", _SFT_EOS_PROBE_PY)
      n_eos, n_weighted = sft_probe_mod.count_eos_targets(
          {
              "targets": np.array([[10, 1, 0, 0]], dtype=np.int32),
              "targets_segmentation": np.array([[1, 1, 0, 0]], dtype=np.int32),
          },
          eos_id=1,
      )
      self.assertEqual((n_eos, n_weighted), (1, 1))

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_2_scatter_bool_and_multimodal_forward_backward_parity_smoke(self) -> None:
    """Test 2: #5812 `_scatter_bool` + chunked attention + per-layer remat preserve multimodal parity."""
    rng = np.random.default_rng(20261025)
    bsz, qlen, width, topk = 2, 19, 32, 6
    indices_np = np.stack(
        [np.stack([rng.choice(width, size=topk, replace=False) for _ in range(qlen)]) for _ in range(bsz)],
        axis=0,
    ).astype(np.int32)
    values_np = rng.choice([False, True], size=(bsz, qlen, topk), p=[0.25, 0.75])

    cand_mask = np.asarray(
        attention_deepseek41._scatter_bool(  # pylint: disable=protected-access
            jnp.asarray(indices_np),
            jnp.asarray(values_np),
            width,
        )
    )
    bi = np.arange(bsz)[:, None, None]
    si = np.arange(qlen)[None, :, None]
    ref_mask = np.zeros((bsz, qlen, width), dtype=bool)
    np.maximum.at(ref_mask, (bi, si, indices_np), values_np)
    np.testing.assert_array_equal(cand_mask, ref_mask)

    # Negative control for _scatter_bool: flipping an active True entry changes the scattered mask.
    mutated_values = values_np.copy()
    mutated_values[0, 0, 0] = not mutated_values[0, 0, 0]
    mutated_mask = np.asarray(
        attention_deepseek41._scatter_bool(  # pylint: disable=protected-access
            jnp.asarray(indices_np),
            jnp.asarray(mutated_values),
            width,
        )
    )
    self.assertFalse(np.array_equal(mutated_mask, ref_mask))

    # Full 14-layer multimodal mini-twin forward + all-leaf backward parity with #5812 training features active.
    grid = multimodal.MINI_GRID_PADDED
    args = multimodal.mini_model_args()
    ref_mm = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
    multimodal.init_mini_reference(ref_mm, seed=1)
    batch = multimodal.build_mini_vl_batch(args, grid=grid, seed=1001, dtype=torch.float32)
    mt_batch, targets_np, targets_seg_np = _build_multimodal_train_batch(
        batch,
        image_token_id=args.image_token_id,
    )

    ref_params_detached = {k: v.detach().clone() for k, v in ref_mm.named_parameters()}
    ref_mm_run = copy.deepcopy(ref_mm)
    grad_classes = multimodal.reference_gradient_classes(ref_mm_run, freeze_vision=True)
    for name, p in ref_mm_run.named_parameters():
      p.requires_grad_(name not in grad_classes.frozen)

    targets_t = torch.as_tensor(targets_np, dtype=torch.long)
    mask_t = torch.as_tensor(targets_seg_np, dtype=torch.float32)
    logits_t = ref_mm_run(batch.input_ids, images=batch.images, token_types=batch.token_types, full_logits=True)
    log_probs_t = F.log_softmax(logits_t, dim=-1)
    nll_t = -torch.gather(log_probs_t, -1, targets_t.unsqueeze(-1)).squeeze(-1)
    loss_t = (nll_t * mask_t).sum() / mask_t.sum()
    ref_loss_val = float(loss_t.detach().cpu().item())

    active_names = [k for k, p in ref_mm_run.named_parameters() if p.requires_grad]
    active_params = [dict(ref_mm_run.named_parameters())[k] for k in active_names]
    raw_grads = torch.autograd.grad(loss_t, active_params, allow_unused=True)
    ref_grads: dict[str, torch.Tensor | None] = {k: None for k in ref_params_detached}
    for k, g in zip(active_names, raw_grads):
      ref_grads[k] = None if g is None else g.detach().clone()
    layer_parity.validate_gradient_classes(grad_classes, set(ref_params_detached), ref_grads)
    del ref_mm_run

    with tempfile.TemporaryDirectory(prefix="ds41_integ_mm_grad_") as ckpt_dir:
      mm_testing.write_mini_safetensors_checkpoint(
          ckpt_dir,
          ref_mm,
          param_dtype=torch.float32,
          include_mtp=False,
      )
      del ref_mm
      gc.collect()

      per_dev_bs = float(multimodal.MINI_BATCH_SIZE) / jax.device_count()
      for scan_layers, attn_chunk in ((True, 16), (False, 16)):
        with self.subTest(scan_layers=scan_layers, attn_chunk=attn_chunk):
          mt_mod, cfg, mesh = mm_testing.load_mini_maxtext_from_safetensors(
              ckpt_dir,
              grid=grid,
              scan_layers=scan_layers,
              freeze_vision_encoder_params=True,
              per_device_batch_size=per_dev_bs,
              z_loss_multiplier=0.0,
              load_balance_loss_weight=0.0,
              routed_bias=True,
              routed_bias_update_rate=0.0,
              remat_policy="custom",
              dtype="float32",
              weight_dtype="float32",
              dtype_mm="float32",
              matmul_precision="highest",
          )
          graphdef, param_state, param_pure, rest_state = mm_testing.extract_pure_params(mt_mod)

          def _loss_and_aux(
              p_dict: dict[str, Any],
              cfg_run: Any = cfg,
              gdef: Any = graphdef,
              pstate: Any = param_state,
              rstate: Any = rest_state,
          ) -> tuple[jax.Array, dict[str, Any]]:
            m = mm_testing.apply_with_pure_params(gdef, pstate, p_dict, rstate)
            return train.loss_fn(m, cfg_run, dict(mt_batch), None, None, is_train=True)

          with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
            (loss_j, _), grad_pure = jax.value_and_grad(_loss_and_aux, has_aux=True)(param_pure)
          cand_loss = float(loss_j)

          ok_loss, loss_m = layer_parity.compare(
              np.asarray([cand_loss], dtype=np.float64),
              np.asarray([ref_loss_val], dtype=np.float64),
              layer_parity.FP32_TOLERANCE,
          )
          self.assertTrue(ok_loss, msg=f"Loss mismatch (scan={scan_layers}): {loss_m}")

          _, grad_failures, _, counts = mm_testing.check_full_model_gradients(
              ref_params=ref_params_detached,
              ref_grads=ref_grads,
              cand_grad_pure=grad_pure,
              config=cfg,
              grad_classes=grad_classes,
              grad_tol=layer_parity.FP32_TOLERANCE,
          )
          self.assertEqual(grad_failures, [], msg=f"Grad failures (scan={scan_layers}): {grad_failures[:5]}")
          self.assertEqual(counts["compared"], 475)

          if scan_layers:
            pert_params = negctl.perturb_one_leaf(
                param_pure,
                target_path="vision_encoder/DeepSeek41VisionProjector_0/w1/kernel",
                rel_scale=1e-2,
                seed=701,
            )
            with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
              (pert_loss_j, _), pert_grad_pure = jax.value_and_grad(_loss_and_aux, has_aux=True)(pert_params)
            status_loss, reasons = e2e_check.negative_run_status(
                {"loss": np.asarray([ref_loss_val], dtype=np.float64)},
                {"loss": np.asarray([cand_loss], dtype=np.float64)},
                {"loss": np.asarray([float(pert_loss_j)], dtype=np.float64)},
                ["loss"],
                layer_parity.FP32_TOLERANCE,
            )
            self.assertEqual(status_loss, "FIRED", msg=f"Loss negctl {status_loss}: {reasons}")
            _, pert_failures, _, _ = mm_testing.check_full_model_gradients(
                ref_params=ref_params_detached,
                ref_grads=ref_grads,
                cand_grad_pure=pert_grad_pure,
                config=cfg,
                grad_classes=grad_classes,
                grad_tol=layer_parity.FP32_TOLERANCE,
            )
            self.assertGreater(len(pert_failures), 0)

  def test_3_aot_presets_metrics_parser_and_bench_script_shims_with_negctl(self) -> None:
    """Test 3: Bench AOT specs/cpu_mini + `bench_metrics.py` + `3_bench_deepseek41_mm.sh` bash -x shim + negctl."""
    aot_mod = _load_module("ds41_aot_check_integ", _AOT_CHECK_PY)
    bench_mod = _load_module("ds41_bench_metrics_integ", _BENCH_METRICS_PY)

    # 3a. Verify 64c/128c bench preset specifications (including PR #5812 flags & 94.74 GiB ceiling) + CPU mini-twin.
    self.assertEqual(aot_mod.BENCH_MAX_HBM_GIB, 94.74)
    for preset_name, topo, fsdp, offload, remat, use_mm in (
        ("tpu7x_64c_mm_bench_min_remat", "tpu7x-128", 8, True, "minimal", True),
        ("tpu7x_64c_text_bench_min_remat", "tpu7x-128", 8, True, "minimal", False),
        ("tpu7x_64c_mm_bench_full_remat", "tpu7x-128", 8, True, "full", True),
        ("tpu7x_64c_text_bench_full_remat", "tpu7x-128", 8, True, "full", False),
        ("tpu7x_128c_mm_bench_min_remat", "tpu7x-256", 16, True, "minimal", True),
        ("tpu7x_128c_text_bench_min_remat", "tpu7x-256", 16, True, "minimal", False),
        ("tpu7x_128c_mm_bench_full_remat", "tpu7x-256", 16, True, "full", True),
        ("tpu7x_128c_text_bench_full_remat", "tpu7x-256", 16, True, "full", False),
    ):
      spec = aot_mod.get_preset_spec(preset_name)
      self.assertEqual(spec["max_hbm_gib"], 94.74)
      ov = spec["overrides"]
      self.assertEqual(ov["compile_topology"], topo)
      self.assertEqual(ov["ici_fsdp_parallelism"], fsdp)
      self.assertEqual(ov["ici_expert_parallelism"], 16)
      self.assertEqual(ov["optimizer_memory_host_offload"], offload)
      self.assertEqual(ov["remat_policy"], remat)
      self.assertEqual(ov["use_multimodal"], use_mm)
      self.assertEqual(ov["grad_dtype"], "bfloat16")
      self.assertEqual(ov["num_vocab_tiling"], 8)
      self.assertEqual(ov["quantization_local_shard_count"], 8)
      self.assertTrue(ov["sparse_matmul"])
      self.assertTrue(ov["megablox"])
      self.assertTrue(ov["use_tokamax_gmm"])
      self.assertTrue(ov["use_gmm_v2"])
      self.assertTrue(ov["allow_split_physical_axes"])
      self.assertEqual(ov["ragged_buffer_factor"], 2.0)
      self.assertFalse(ov["ds41_activation_quant"])
      self.assertEqual(ov["max_target_length"], 4096)
      self.assertEqual(ov["per_device_batch_size"], 1.0)
      self.assertTrue(ov["enable_checkpointing"])
      self.assertFalse(ov["async_checkpointing"])
      self.assertFalse(ov["save_checkpoint_on_start"])
      self.assertFalse(ov["save_checkpoint_on_completion"])
      self.assertEqual(ov["checkpoint_period"], 100000)
      self.assertEqual(ov["log_period"], 1)

    # End-to-end config loader reachability on full 749B TPU7x bench full-remat overrides (64c & 128c).
    for full_preset, expected_topo, expected_fsdp in (
        ("tpu7x_64c_mm_bench_full_remat", "tpu7x-128", 8),
        ("tpu7x_128c_mm_bench_full_remat", "tpu7x-256", 16),
    ):
      full_cfg = pyconfig.initialize(
          [sys.argv[0], aot_mod.BASE_CONFIG_PATH],
          **aot_mod.get_preset_spec(full_preset)["overrides"],
      )
      self.assertEqual(full_cfg.compile_topology, expected_topo)
      self.assertEqual(full_cfg.ici_fsdp_parallelism, expected_fsdp)
      self.assertEqual(full_cfg.ici_expert_parallelism, 16)
      self.assertEqual(full_cfg.grad_dtype, "bfloat16")
      self.assertEqual(full_cfg.num_vocab_tiling, 8)
      self.assertEqual(full_cfg.quantization_local_shard_count, 8)
      self.assertTrue(full_cfg.use_tokamax_gmm)
      self.assertTrue(full_cfg.use_gmm_v2)
      self.assertTrue(full_cfg.allow_split_physical_axes)
      self.assertEqual(full_cfg.ragged_buffer_factor, 2.0)
      self.assertEqual(full_cfg.remat_policy, "full")
      self.assertTrue(full_cfg.enable_checkpointing)
      self.assertFalse(full_cfg.async_checkpointing)
      self.assertFalse(full_cfg.save_checkpoint_on_start)
      self.assertFalse(full_cfg.save_checkpoint_on_completion)
      self.assertEqual(full_cfg.checkpoint_period, 100000)

    res_a8_mini = aot_mod.run_preset("tpu7x_64c_mm_bench_min_remat", cpu_mini=True, max_target_length_override=64)
    self.assertTrue(res_a8_mini["compile_succeeded"])
    self.assertTrue(res_a8_mini["passed"])
    self.assertEqual(res_a8_mini["gate_mode"], "tpu7x_heap_compile")
    self.assertEqual(res_a8_mini["tpu7x_heap_limit_gib"], 94.74)
    self.assertNotIn("peak_hbm_gib", res_a8_mini)
    self.assertNotIn("effective_peak_hbm_gib", res_a8_mini)
    self.assertGreater(res_a8_mini["temp_hbm_gib"], 0.0)
    self.assertGreaterEqual(res_a8_mini["non_aliased_io_hbm_gib"], 0.0)
    self.assertGreater(res_a8_mini["naive_sum_hbm_gib"], 0.0)
    self.assertEqual(
        res_a8_mini["clamped_heap_cap_gib"],
        round(min(94.74, res_a8_mini["naive_sum_hbm_gib"]), 6),
    )
    self.assertTrue(all(res_a8_mini["vision_hlo_markers"].values()))
    self.assertEqual(res_a8_mini["recommended_remat_policy"], "minimal")
    res_a8t_mini = aot_mod.run_preset("tpu7x_64c_text_bench_min_remat", cpu_mini=True, max_target_length_override=64)
    self.assertTrue(res_a8t_mini["compile_succeeded"])
    self.assertTrue(res_a8t_mini["passed"])
    self.assertEqual(res_a8t_mini["gate_mode"], "tpu7x_heap_compile")
    self.assertNotIn("peak_hbm_gib", res_a8t_mini)
    self.assertNotIn("effective_peak_hbm_gib", res_a8t_mini)
    self.assertFalse(any(res_a8t_mini["vision_hlo_markers"].values()))

    # Verify E2E presets remain strictly on <= 85.0 GiB unclamped `naive_sum_hbm_gib` (`peak_hbm_gib` kept).
    for remat_preset in (
        "tpu7x_64c_text_scanned_fwd",
        "tpu7x_64c_text_unscanned_fwd",
        "tpu7x_64c_mm_scanned_fwd",
        "tpu7x_64c_mm_unscanned_fwd",
        "tpu7x_64c_mm_prefill_2048",
        "tpu7x_64c_text_prefill_2048",
        "tpu7x_64c_mm_train_adamw",
        "tpu7x_64c_text_train_sgd_b1",
    ):
      remat_spec = aot_mod.get_preset_spec(remat_preset)
      self.assertEqual(remat_spec["max_hbm_gib"], 85.0)
    res_a3_mini = aot_mod.run_preset("tpu7x_64c_mm_scanned_fwd", cpu_mini=True, max_target_length_override=36)
    self.assertEqual(res_a3_mini["gate_mode"], "naive_sum_budget")
    self.assertIn("peak_hbm_gib", res_a3_mini)
    self.assertNotIn("effective_peak_hbm_gib", res_a3_mini)
    self.assertEqual(res_a3_mini["peak_hbm_gib"], res_a3_mini["naive_sum_hbm_gib"])

    # Verify recommended_remat_policy flips to 'full' when threshold is set below naive_sum_hbm_gib.
    res_fb = aot_mod.run_preset(
        "tpu7x_64c_mm_bench_min_remat",
        cpu_mini=True,
        max_target_length_override=64,
        remat_fallback_hbm_gib_override=max(1e-9, res_a8_mini["naive_sum_hbm_gib"] * 0.5),
    )
    self.assertEqual(res_fb["recommended_remat_policy"], "full")

    # 3a-ii. Verify unclamped load-time HBM reporting vs 85.0 GiB naive_sum gate + standardized log line.
    gib_bytes = 1024**3

    def _fake_compiled(temp_gib: float, io_gib: float) -> Any:
      mem_ns = types.SimpleNamespace(
          temp_size_in_bytes=int(round(temp_gib * gib_bytes)),
          argument_size_in_bytes=int(round(io_gib * gib_bytes)),
          output_size_in_bytes=int(round(io_gib * gib_bytes)),
          alias_size_in_bytes=int(round(io_gib * gib_bytes)),
      )
      return types.SimpleNamespace(memory_analysis=lambda: mem_ns)

    # Sample 128c full-remat profile: temp=84.723 GiB, non_aliased_io=19.107 GiB, naive_sum=103.830 GiB.
    stats_ok = aot_mod._extract_memory_stats(  # pylint: disable=protected-access
        _fake_compiled(84.723, 19.107),
        max_hbm_gib=94.74,
        use_heap_compile_gate=True,
    )
    self.assertTrue(stats_ok["compile_succeeded"])
    self.assertTrue(stats_ok["passed"])
    self.assertEqual(stats_ok["gate_mode"], "tpu7x_heap_compile")
    self.assertEqual(stats_ok["tpu7x_heap_limit_gib"], 94.74)
    self.assertAlmostEqual(stats_ok["temp_hbm_gib"], 84.723, places=6)
    self.assertAlmostEqual(stats_ok["non_aliased_io_hbm_gib"], 19.107, places=6)
    self.assertAlmostEqual(stats_ok["naive_sum_hbm_gib"], 103.830, places=6)
    self.assertAlmostEqual(stats_ok["clamped_heap_cap_gib"], 94.74, places=6)
    self.assertNotIn("peak_hbm_gib", stats_ok)
    self.assertNotIn("effective_peak_hbm_gib", stats_ok)
    log_line_a9b = aot_mod._format_aot_log_line("tpu7x_128c_mm_bench_full_remat", stats_ok)  # pylint: disable=protected-access
    self.assertEqual(
        log_line_a9b,
        "[aot_check] preset=tpu7x_128c_mm_bench_full_remat compile_succeeded=True passed=True "
        "temp_hbm_gib=84.723000 non_aliased_io_hbm_gib=19.107000 "
        "naive_sum_hbm_gib=103.830000 clamped_heap_cap_gib=94.740000 "
        "tpu7x_heap_limit_gib=94.740000",
    )

    # Negative controls: temp > 94.74 (96.568), non_aliased_io > 94.74, or naive_sum 90.0 > 85.0 MUST fail!
    stats_bad_temp = aot_mod._extract_memory_stats(  # pylint: disable=protected-access
        _fake_compiled(96.568, 16.382),
        max_hbm_gib=94.74,
        use_heap_compile_gate=True,
    )
    self.assertFalse(stats_bad_temp["passed"])
    self.assertNotIn("peak_hbm_gib", stats_bad_temp)
    stats_bad_io = aot_mod._extract_memory_stats(  # pylint: disable=protected-access
        _fake_compiled(50.0, 95.10),
        max_hbm_gib=94.74,
        use_heap_compile_gate=True,
    )
    self.assertFalse(stats_bad_io["passed"])
    stats_strict_85 = aot_mod._extract_memory_stats(  # pylint: disable=protected-access
        _fake_compiled(50.0, 40.0),
        max_hbm_gib=85.0,
        use_heap_compile_gate=False,
    )
    self.assertFalse(stats_strict_85["passed"])
    self.assertAlmostEqual(stats_strict_85["peak_hbm_gib"], 90.0, places=6)

    # Verify `_check_vision_hlo` when XLA:TPU omits >500-leaf parameter names in `compiled` while `lowered` retains them.
    tpu_compiled_hlo = "HloModule m %param.1 = bf16[5120] parameter(1) %c1 = embed_patches %c2 = merge_mm_embeddings"
    tpu_lowered_hlo = "module @jit_train_step { // state['model']['vision_encoder']['DeepSeek41VisionEncoder_0'] }"
    markers_749b = aot_mod._check_vision_hlo(  # pylint: disable=protected-access
        tpu_compiled_hlo,
        expect_vision_in_hlo=True,
        use_multimodal=True,
        lowered_hlo_text=tpu_lowered_hlo,
    )
    self.assertEqual(
        markers_749b,
        {"vision_encoder": True, "embed_patches": True, "merge_mm_embeddings": True},
    )
    # Negative control: compute markers in `lowered_hlo_text` NEVER count if DCE'd from `compiled_hlo_text`.
    with self.assertRaisesRegex(AssertionError, "embed_patches"):
      aot_mod._check_vision_hlo(  # pylint: disable=protected-access
          "HloModule m %param.1 = bf16[5120] parameter(1) %c2 = merge_mm_embeddings",
          expect_vision_in_hlo=True,
          use_multimodal=True,
          lowered_hlo_text=f"{tpu_lowered_hlo} embed_patches",
      )
    # Negative control: `vision_encoder` in `lowered_hlo_text` fails when `use_multimodal=False`.
    with self.assertRaisesRegex(AssertionError, "use_multimodal=False"):
      aot_mod._check_vision_hlo(  # pylint: disable=protected-access
          "HloModule m %param.1 = bf16[5120] parameter(1)",
          expect_vision_in_hlo=False,
          use_multimodal=False,
          lowered_hlo_text=tpu_lowered_hlo,
      )

    # 3a-iii. Verify 64c and 128c bench batch presets catch minimal RESOURCE_EXHAUSTED, run full, and exit 0 (+ negctl).
    def _mock_run_preset_single(preset_name: str, **_kwargs: Any) -> dict[str, Any]:
      if preset_name in (
          "tpu7x_64c_mm_bench_min_remat",
          "tpu7x_64c_text_bench_min_remat",
          "tpu7x_128c_mm_bench_min_remat",
          "tpu7x_128c_text_bench_min_remat",
      ):
        raise RuntimeError(f"RESOURCE_EXHAUSTED: {preset_name} exceeded 94.74G HBM in minimal remat")
      return {
          "preset": preset_name,
          "mode": "train",
          "compile_succeeded": True,
          "compile_error": "",
          "passed": True,
          "gate_mode": "tpu7x_heap_compile",
          "remat_policy": "full",
          "tpu7x_heap_limit_gib": 94.74,
          "clamped_heap_cap_gib": 94.74,
          "temp_hbm_gib": 84.723,
          "non_aliased_io_hbm_gib": 19.107,
          "naive_sum_hbm_gib": 103.830,
      }

    with tempfile.TemporaryDirectory(prefix="ds41_aot_batch_") as aot_tmp:
      aot_tmp_path = Path(aot_tmp)
      for batch_name, min_mm, min_txt, full_mm, full_txt in (
          (
              "all_64c_bench_presets",
              "tpu7x_64c_mm_bench_min_remat",
              "tpu7x_64c_text_bench_min_remat",
              "tpu7x_64c_mm_bench_full_remat",
              "tpu7x_64c_text_bench_full_remat",
          ),
          (
              "all_128c_bench_presets",
              "tpu7x_128c_mm_bench_min_remat",
              "tpu7x_128c_text_bench_min_remat",
              "tpu7x_128c_mm_bench_full_remat",
              "tpu7x_128c_text_bench_full_remat",
          ),
      ):
        out_json = aot_tmp_path / f"{batch_name}.json"
        orig_run_preset = aot_mod.run_preset

        def _dispatch_preset(
            p: str,
            _orig: Any = orig_run_preset,
            _batch: str = batch_name,
            **kw: Any,
        ) -> dict[str, Any]:
          if p == _batch:
            return _orig(p, **kw)
          return _mock_run_preset_single(p, **kw)

        with mock.patch.object(aot_mod, "run_preset", side_effect=_dispatch_preset):
          rc = aot_mod.main([f"--preset={batch_name}", f"--output_json={out_json.as_posix()}"])
        self.assertEqual(rc, 0)
        batch_payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertTrue(batch_payload["passed"])
        self.assertEqual(batch_payload["selected_remat_policy"], "full")
        self.assertFalse(batch_payload["results"][min_mm]["compile_succeeded"])
        self.assertFalse(batch_payload["results"][min_mm]["passed"])
        self.assertIn("RESOURCE_EXHAUSTED", batch_payload["results"][min_mm]["compile_error"])
        self.assertFalse(batch_payload["results"][min_txt]["passed"])
        self.assertTrue(batch_payload["results"][full_mm]["passed"])
        self.assertTrue(batch_payload["results"][full_txt]["passed"])
        self.assertNotIn("peak_hbm_gib", batch_payload["results"][full_mm])
        self.assertNotIn("effective_peak_hbm_gib", batch_payload["results"][full_mm])

      # Negative control: when tpu7x_64c_mm_bench_full_remat fails HBM too, main() writes JSON and raises!
      fail_all_json = aot_tmp_path / "bench_fail_all.json"
      orig_run_preset = aot_mod.run_preset

      def _dispatch_all_fail(p: str, _orig: Any = orig_run_preset, **kw: Any) -> dict[str, Any]:
        if p == "all_64c_bench_presets":
          return _orig(p, **kw)
        if p in ("tpu7x_64c_mm_bench_min_remat", "tpu7x_64c_text_bench_min_remat", "tpu7x_64c_mm_bench_full_remat"):
          raise RuntimeError(f"RESOURCE_EXHAUSTED: {p} failed XLA HBM allocation")
        return _mock_run_preset_single(p, **kw)

      with mock.patch.object(aot_mod, "run_preset", side_effect=_dispatch_all_fail):
        with self.assertRaises(AssertionError):
          aot_mod.main(["--preset=all_64c_bench_presets", f"--output_json={fail_all_json.as_posix()}"])
      fail_payload = json.loads(fail_all_json.read_text(encoding="utf-8"))
      self.assertFalse(fail_payload["passed"])
      self.assertIsNone(fail_payload["selected_remat_policy"])
      self.assertFalse(fail_payload["results"]["tpu7x_64c_mm_bench_full_remat"]["passed"])

    # 3b. Run `bash -x 3_bench_deepseek41_mm.sh` with a PATH/PYTHON_BIN shim across 64c (minimal) and 128c (full fallback).
    with tempfile.TemporaryDirectory(prefix="ds41_bench_shim_") as tmpdir:
      tmp_path = Path(tmpdir)
      shim_bin_dir = tmp_path / "bin"
      shim_bin_dir.mkdir(parents=True, exist_ok=True)
      real_python = sys.executable
      expected_text_tflops = bench_mod.compute_expected_per_device_tflops(
          use_multimodal=False,
          topology_mode="64c",
          remat_policy="minimal",
      )["total_per_device_tflops"]
      expected_mm_tflops = bench_mod.compute_expected_per_device_tflops(
          use_multimodal=True,
          topology_mode="64c",
          remat_policy="minimal",
      )["total_per_device_tflops"]
      self.assertAlmostEqual(expected_mm_tflops - expected_text_tflops, _EXPECTED_VISION_TFLOPS, places=10)

      python_shim = shim_bin_dir / "python_shim.py"
      python_shim.write_text(
          f"""#!{real_python}
import json
import os
from pathlib import Path
import sys

REAL_PYTHON = {real_python!r}
MM_TFLOPS = {expected_mm_tflops!r}
TEXT_TFLOPS = {expected_text_tflops!r}

if len(sys.argv) >= 2 and sys.argv[1].endswith("train.py"):
  kv = {{}}
  for arg in sys.argv[3:]:
    if "=" in arg:
      k, v = arg.split("=", 1)
      kv[k] = v
  kv["_flag_log_moe_dropped_frac"] = "1" if kv.get("log_moe_dropped_frac", "") == "true" else ""
  invocation_log = os.environ["SHIM_INVOCATION_JSONL"] if "SHIM_INVOCATION_JSONL" in os.environ else ""
  if invocation_log:
    with open(invocation_log, "a", encoding="utf-8") as inv_f:
      inv_f.write(json.dumps(kv, sort_keys=True) + "\\n")

  if "SHIM_CAPTURE_ONLY" in os.environ and os.environ["SHIM_CAPTURE_ONLY"] == "1":
    sys.exit(0)
  steps = int(kv["steps"])
  use_mm = kv["use_multimodal"].lower() == "true"
  tflops = MM_TFLOPS if use_mm else TEXT_TFLOPS
  default_dt = "6.0" if use_mm else "5.85"
  base_step_s = float(os.environ["SHIM_STEP_TIME_S"] if "SHIM_STEP_TIME_S" in os.environ else default_dt)
  inject_nan_step = int(os.environ["SHIM_NAN_STEP"] if "SHIM_NAN_STEP" in os.environ else "-1")
  if kv.get("profiler") == "xplane":
    tb_trace = Path(kv["base_output_directory"]) / kv["run_name"] / "tensorboard" / "plugins" / "profile" / "t.xplane.pb"
    tb_trace.parent.mkdir(parents=True, exist_ok=True)
    tb_trace.write_bytes(b"xplane_stub")
  metrics_path = Path(kv["metrics_file"])
  metrics_path.parent.mkdir(parents=True, exist_ok=True)
  with metrics_path.open("w", encoding="utf-8") as mf:
    for s in range(steps):
      # Distort trailing XProf steps >= 30 on purpose to prove steady window [10..29] ignores them.
      if s < 10:
        dt = base_step_s + 1.5
      elif s < 30:
        dt = base_step_s + (0.02 if (s % 2 == 1) else -0.02)
      else:
        dt = 45.0
      loss = float("nan") if s == inject_nan_step else (12.2 - 0.01 * s)
      rec = {{
          "step": float(s),
          "run_name": kv["run_name"],
          "learning/loss": loss,
          "perf/step_time_seconds": dt,
          "perf/per_device_tflops": tflops,
          "perf/per_device_tflops_per_sec": tflops / dt,
          "perf/per_device_tokens_per_sec": 4096.0 / dt,
      }}
      mf.write(json.dumps(rec) + "\\n")
  omit_moe_dropped = "SHIM_OMIT_MOE_DROPPED" in os.environ and os.environ["SHIM_OMIT_MOE_DROPPED"] == "1"
  if not omit_moe_dropped and kv["_flag_log_moe_dropped_frac"] == "1":
    print(
        "MOE_DROPPED_FRAC step=0 ragged_buffer_factor=2.0 mean=0.001250 max=0.005000 "
        "per_layer=[0.000000,0.002500,0.005000,0.000000]",
        flush=True,
    )
  sys.exit(0)

if (
    len(sys.argv) >= 2
    and sys.argv[1].endswith("bench_metrics.py")
    and "--mode=summarize_bench" in sys.argv
    and "SHIM_CAPTURE_ONLY" in os.environ
    and os.environ["SHIM_CAPTURE_ONLY"] == "1"
):
  sys.exit(0)
os.execv(REAL_PYTHON, [REAL_PYTHON] + sys.argv[1:])
""",
          encoding="utf-8",
      )
      python_shim.chmod(python_shim.stat().st_mode | stat.S_IEXEC)

      gcloud_log = tmp_path / "gcloud_calls.log"
      gcloud_shim = shim_bin_dir / "gcloud"
      gcloud_shim.write_text(
          f"""#!/usr/bin/env bash
set -euo pipefail
echo "$@" >> "{gcloud_log.as_posix()}"
""",
          encoding="utf-8",
      )
      gcloud_shim.chmod(gcloud_shim.stat().st_mode | stat.S_IEXEC)

      base_env = os.environ.copy()
      orig_path = base_env["PATH"] if "PATH" in base_env else ""
      base_env["PATH"] = f"{shim_bin_dir.as_posix()}:{orig_path}"
      base_env["PYTHONPATH"] = f"{(_REPO_ROOT / 'src').as_posix()}:{_REPO_ROOT.as_posix()}"
      base_env["PYTHON_BIN"] = python_shim.as_posix()

      # Positive Case A: 64c, primary AOT = 82.0 GiB (<= 85.0 -> selects remat_policy='minimal'), Orbax load.
      out_64c = tmp_path / "out_64c"
      inv_64c = tmp_path / "inv_64c.jsonl"
      env_64c = {
          **base_env,
          "TOPOLOGY_MODE": "64c",
          "OUTPUT_DIR": out_64c.as_posix(),
          "LOAD_PARAMETERS_PATH": "gs://maxtext-deepseek/deepseek-v4.1-flash/orbax/scanned/0/items",
          "MM_AOT_HBM_GIB": "82.0",
          "TEXT_AOT_HBM_GIB": "80.0",
          "GCS_LOG_PREFIX": "gs://maxtext-deepseek/deepseek-v4.1-flash/logs/bench",
          "SHIM_INVOCATION_JSONL": inv_64c.as_posix(),
      }
      proc_64c = subprocess.run(
          ["bash", "-x", _BENCH_SCRIPT_SH],
          env=env_64c,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_64c.returncode, 0, msg=f"64c shim failed:\n{proc_64c.stdout}\n{proc_64c.stderr}")

      invocations_64c = [json.loads(line) for line in inv_64c.read_text(encoding="utf-8").splitlines() if line.strip()]
      self.assertEqual(len(invocations_64c), 2)
      self.assertNotIn("ds41_routed_bias_vl_update_rate", invocations_64c[0])
      self.assertEqual(invocations_64c[0]["use_multimodal"], "true")
      self.assertEqual(invocations_64c[0]["remat_policy"], "minimal")
      self.assertEqual(invocations_64c[0]["grad_dtype"], "bfloat16")
      self.assertEqual(invocations_64c[0]["num_vocab_tiling"], "8")
      self.assertEqual(invocations_64c[0]["quantization_local_shard_count"], "8")
      self.assertEqual(invocations_64c[0]["use_tokamax_gmm"], "true")
      self.assertEqual(invocations_64c[0]["use_gmm_v2"], "true")
      self.assertEqual(invocations_64c[0]["allow_split_physical_axes"], "true")
      self.assertEqual(invocations_64c[0]["ragged_buffer_factor"], "2.0")
      self.assertEqual(invocations_64c[0]["use_random_routing"], "false")
      self.assertEqual(invocations_64c[0]["log_period"], "1")
      self.assertEqual(invocations_64c[0]["enable_tpu_profiling_options"], "true")
      self.assertEqual(invocations_64c[0]["ds41_activation_quant"], "false")
      self.assertEqual(invocations_64c[0]["ici_fsdp_parallelism"], "8")
      self.assertEqual(invocations_64c[0]["optimizer_memory_host_offload"], "true")
      self.assertEqual(invocations_64c[0]["source_checkpoint_layout"], "orbax")
      self.assertEqual(invocations_64c[0]["enable_checkpointing"], "true")
      self.assertEqual(invocations_64c[0]["async_checkpointing"], "false")
      self.assertEqual(invocations_64c[0]["save_checkpoint_on_start"], "false")
      self.assertEqual(invocations_64c[0]["save_checkpoint_on_completion"], "false")
      self.assertEqual(invocations_64c[0]["checkpoint_period"], "100000")
      self.assertGreater(int(invocations_64c[0]["checkpoint_period"]), int(invocations_64c[0]["steps"]))
      self.assertEqual(invocations_64c[0]["steps"], "33")
      self.assertEqual(invocations_64c[0]["skip_first_n_steps_for_profiler"], "30")
      self.assertEqual(invocations_64c[0]["profiler_steps"], "3")
      self.assertNotIn("tensorboard_dir", invocations_64c[0])
      self.assertIn(
          "gs://maxtext-deepseek/deepseek-v4.1-flash/logs/bench/xprof_64c_mm/",
          gcloud_log.read_text(encoding="utf-8"),
      )
      self.assertEqual(invocations_64c[1]["use_multimodal"], "false")
      self.assertEqual(invocations_64c[1]["enable_checkpointing"], "true")
      self.assertEqual(invocations_64c[1]["async_checkpointing"], "false")
      self.assertEqual(invocations_64c[1]["save_checkpoint_on_start"], "false")
      self.assertEqual(invocations_64c[1]["save_checkpoint_on_completion"], "false")
      self.assertEqual(invocations_64c[1]["checkpoint_period"], "100000")

      sum_64c_mm = json.loads((out_64c / "bench_64c_mm_summary.json").read_text(encoding="utf-8"))
      sum_64c_txt = json.loads((out_64c / "bench_64c_text_summary.json").read_text(encoding="utf-8"))
      self.assertTrue(sum_64c_mm["passed"])
      self.assertTrue(sum_64c_txt["passed"])
      self.assertNotIn("effective_peak_hbm_gib", sum_64c_mm)
      self.assertNotIn("live_peak_hbm_gib", sum_64c_mm)
      self.assertAlmostEqual(sum_64c_mm["steady_median_step_time_s"], 6.0, places=9)
      self.assertAlmostEqual(sum_64c_mm["steady_mean_step_time_s"], 6.0, places=9)
      self.assertAlmostEqual(sum_64c_mm["per_device_tflops"], expected_mm_tflops, places=9)
      self.assertAlmostEqual(
          sum_64c_mm["mfu_pct"],
          (expected_mm_tflops / 6.0) / 1153.5 * 100.0,
          places=9,
      )
      self.assertAlmostEqual(sum_64c_mm["tokens_per_sec_per_device"], 4096.0 / 6.0, places=9)
      self.assertAlmostEqual(sum_64c_mm["tokens_per_sec_per_chip"], 2.0 * 4096.0 / 6.0, places=9)
      self.assertTrue(sum_64c_mm["loss_finite_all_bench_steps"])
      self.assertAlmostEqual(sum_64c_mm["peak_hbm_max_aot_ratio"], 1.1, places=6)
      self.assertAlmostEqual(sum_64c_mm["max_allowed_hbm_gb"], min(94.74, 1.1 * 82.0), places=6)
      self.assertEqual(
          sum_64c_mm["xprof_trace_gcs_uri"],
          "gs://maxtext-deepseek/deepseek-v4.1-flash/logs/bench/xprof_64c_mm",
      )
      self.assertEqual(sum_64c_mm["moe_routing_mode"], "token_dropping")
      self.assertEqual(sum_64c_mm["ragged_buffer_factor"], 2.0)
      self.assertAlmostEqual(sum_64c_mm["step0_moe_dropped_frac"]["mean"], 0.00125, places=6)
      self.assertAlmostEqual(sum_64c_mm["step0_moe_dropped_frac"]["max"], 0.005, places=6)
      self.assertEqual(len(sum_64c_mm["step0_moe_dropped_frac"]["per_layer"]), 4)
      table_64c_mm = bench_mod.format_stage_summary_table(sum_64c_mm)
      self.assertIn("moe_routing_mode", table_64c_mm)
      self.assertIn("token_dropping", table_64c_mm)
      self.assertIn("ragged_buffer_factor", table_64c_mm)
      self.assertIn("2.0", table_64c_mm)
      self.assertIn("mean=0.001250 max=0.005000", table_64c_mm)

      # Positive Case B: 128c AOT (text min-remat temp > 94.74 -> fallback to full),
      # safetensors_dynamic HF_CKPT_PATH, and RESUME_PASSED_STAGES=1 skipping already-passed MM stage.
      out_128c = tmp_path / "out_128c"
      resume_dir = tmp_path / "resume_128c"
      resume_dir.mkdir(parents=True, exist_ok=True)
      (resume_dir / "bench_128c_mm_summary.json").write_text(
          json.dumps({"passed": True, "stage": "MM"}),
          encoding="utf-8",
      )
      aot_batch_path = tmp_path / "bench_128c_aot.json"
      aot_batch_path.write_text(
          json.dumps(
              {
                  "preset": "all_128c_bench_presets",
                  "passed": True,
                  "selected_remat_policy": "full",
                  "results": {
                      "tpu7x_128c_mm_bench_min_remat": {
                          "compile_succeeded": True,
                          "passed": True,
                          "gate_mode": "tpu7x_heap_compile",
                          "tpu7x_heap_limit_gib": 94.74,
                          "clamped_heap_cap_gib": 94.74,
                          "temp_hbm_gib": 94.014,
                          "non_aliased_io_hbm_gib": 19.107,
                          "naive_sum_hbm_gib": 113.121,
                      },
                      "tpu7x_128c_text_bench_min_remat": {
                          "compile_succeeded": True,
                          "passed": False,
                          "gate_mode": "tpu7x_heap_compile",
                          "tpu7x_heap_limit_gib": 94.74,
                          "clamped_heap_cap_gib": 94.74,
                          "temp_hbm_gib": 96.568,
                          "non_aliased_io_hbm_gib": 16.382,
                          "naive_sum_hbm_gib": 112.951,
                      },
                      "tpu7x_128c_mm_bench_full_remat": {
                          "compile_succeeded": True,
                          "passed": True,
                          "gate_mode": "tpu7x_heap_compile",
                          "tpu7x_heap_limit_gib": 94.74,
                          "clamped_heap_cap_gib": 94.74,
                          "temp_hbm_gib": 84.723,
                          "non_aliased_io_hbm_gib": 19.107,
                          "naive_sum_hbm_gib": 103.830,
                      },
                      "tpu7x_128c_text_bench_full_remat": {
                          "compile_succeeded": True,
                          "passed": True,
                          "gate_mode": "tpu7x_heap_compile",
                          "tpu7x_heap_limit_gib": 94.74,
                          "clamped_heap_cap_gib": 94.74,
                          "temp_hbm_gib": 89.993,
                          "non_aliased_io_hbm_gib": 16.382,
                          "naive_sum_hbm_gib": 106.375,
                      },
                  },
              }
          ),
          encoding="utf-8",
      )
      inv_128c = tmp_path / "inv_128c.jsonl"
      env_128c = {
          **base_env,
          "TOPOLOGY_MODE": "128c",
          "OUTPUT_DIR": out_128c.as_posix(),
          "HF_CKPT_PATH": "gs://maxtext-deepseek/deepseek-v4.1-flash/hf",
          "AOT_BATCH_JSON": aot_batch_path.as_posix(),
          "RESUME_PASSED_STAGES": "1",
          "GCS_RESUME_SUMMARY_PREFIX": resume_dir.as_posix(),
          "SHIM_INVOCATION_JSONL": inv_128c.as_posix(),
      }
      proc_128c = subprocess.run(
          ["bash", "-x", _BENCH_SCRIPT_SH],
          env=env_128c,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_128c.returncode, 0, msg=f"128c shim failed:\n{proc_128c.stdout}\n{proc_128c.stderr}")
      self.assertIn("| 128c/text | full |", proc_128c.stdout)
      self.assertIn("aot_temp_hbm_gib", proc_128c.stdout)
      self.assertIn("aot_clamped_heap_cap_gib", proc_128c.stdout)
      self.assertIn("moe_routing_mode", proc_128c.stdout)
      self.assertIn("| token_dropping | 2.0 | mean=0.001250 max=0.005000 |", proc_128c.stdout)
      invocations_128c = [json.loads(line) for line in inv_128c.read_text(encoding="utf-8").splitlines() if line.strip()]
      # MM was skipped via RESUME_PASSED_STAGES=1; only TEXT executed, with auto-selected remat_policy='full'!
      self.assertEqual(len(invocations_128c), 1)
      self.assertEqual(invocations_128c[0]["use_multimodal"], "false")
      self.assertEqual(invocations_128c[0]["remat_policy"], "full")
      self.assertEqual(invocations_128c[0]["grad_dtype"], "bfloat16")
      self.assertEqual(invocations_128c[0]["num_vocab_tiling"], "8")
      self.assertEqual(invocations_128c[0]["quantization_local_shard_count"], "8")
      self.assertEqual(invocations_128c[0]["use_tokamax_gmm"], "true")
      self.assertEqual(invocations_128c[0]["use_gmm_v2"], "true")
      self.assertEqual(invocations_128c[0]["allow_split_physical_axes"], "true")
      self.assertEqual(invocations_128c[0]["ragged_buffer_factor"], "2.0")
      self.assertEqual(invocations_128c[0]["ici_fsdp_parallelism"], "16")
      self.assertEqual(invocations_128c[0]["optimizer_memory_host_offload"], "true")
      self.assertEqual(invocations_128c[0]["_flag_log_moe_dropped_frac"], "1")
      self.assertEqual(invocations_128c[0]["source_checkpoint_layout"], "safetensors_dynamic")
      self.assertEqual(invocations_128c[0]["enable_checkpointing"], "true")
      self.assertEqual(invocations_128c[0]["async_checkpointing"], "false")
      self.assertEqual(invocations_128c[0]["save_checkpoint_on_start"], "false")
      self.assertEqual(invocations_128c[0]["save_checkpoint_on_completion"], "false")
      self.assertEqual(invocations_128c[0]["checkpoint_period"], "100000")
      self.assertGreater(int(invocations_128c[0]["checkpoint_period"]), int(invocations_128c[0]["steps"]))
      sel_128c = json.loads((out_128c / "bench_128c_aot_selection.json").read_text(encoding="utf-8"))
      self.assertEqual(sel_128c["selection_source"], "primary_aot_failed_fallback")
      sum_128c_txt = json.loads((out_128c / "bench_128c_text_summary.json").read_text(encoding="utf-8"))
      self.assertTrue(sum_128c_txt["passed"])
      self.assertNotIn("effective_peak_hbm_gib", sum_128c_txt)
      self.assertTrue(sum_128c_txt["aot_compile_succeeded"])
      self.assertAlmostEqual(sum_128c_txt["aot_temp_hbm_gib"], 89.993, places=6)
      self.assertAlmostEqual(sum_128c_txt["aot_non_aliased_io_hbm_gib"], 16.382, places=6)
      self.assertAlmostEqual(sum_128c_txt["aot_naive_sum_hbm_gib"], 106.375, places=6)
      self.assertAlmostEqual(sum_128c_txt["aot_clamped_heap_cap_gib"], 94.74, places=6)
      self.assertEqual(sum_128c_txt["moe_routing_mode"], "token_dropping")
      self.assertEqual(sum_128c_txt["ragged_buffer_factor"], 2.0)
      self.assertAlmostEqual(sum_128c_txt["step0_moe_dropped_frac"]["mean"], 0.00125, places=6)
      self.assertAlmostEqual(sum_128c_txt["step0_moe_dropped_frac"]["max"], 0.005, places=6)
      self.assertEqual(len(sum_128c_txt["step0_moe_dropped_frac"]["per_layer"]), 4)
      self.assertAlmostEqual(sum_128c_txt["step0_moe_dropped_frac_mean"], 0.00125, places=6)
      self.assertAlmostEqual(sum_128c_txt["step0_moe_dropped_frac_max"], 0.005, places=6)
      self.assertEqual(len(sum_128c_txt["step0_moe_dropped_frac_per_layer"]), 4)
      table_128c_txt = bench_mod.format_stage_summary_table(sum_128c_txt)
      self.assertIn("| token_dropping | 2.0 | mean=0.001250 max=0.005000 |", table_128c_txt)
      self.assertAlmostEqual(sum_128c_txt["max_allowed_hbm_gb"], 94.74, places=6)

      # Positive Case C: Dynamic checkpoint_period = max(100000, total_steps + 1000) when BENCH_STEPS=200000.
      inv_dyn = tmp_path / "inv_dyn.jsonl"
      proc_dyn = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH],
          env={
              **env_64c,
              "OUTPUT_DIR": (tmp_path / "out_dyn").as_posix(),
              "STAGES": "MM",
              "BENCH_STEPS": "200000",
              "XPROF_STEPS": "3",
              "SHIM_CAPTURE_ONLY": "1",
              "SHIM_INVOCATION_JSONL": inv_dyn.as_posix(),
          },
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_dyn.returncode, 0, msg=f"Dynamic ckpt_period shim failed:\n{proc_dyn.stderr}")
      invocations_dyn = [json.loads(line) for line in inv_dyn.read_text(encoding="utf-8").splitlines() if line.strip()]
      self.assertEqual(len(invocations_dyn), 1)
      self.assertEqual(invocations_dyn[0]["enable_checkpointing"], "true")
      self.assertEqual(invocations_dyn[0]["async_checkpointing"], "false")
      self.assertEqual(invocations_dyn[0]["save_checkpoint_on_start"], "false")
      self.assertEqual(invocations_dyn[0]["save_checkpoint_on_completion"], "false")
      self.assertEqual(invocations_dyn[0]["steps"], "200003")
      self.assertEqual(invocations_dyn[0]["checkpoint_period"], "201003")
      self.assertGreater(int(invocations_dyn[0]["checkpoint_period"]), int(invocations_dyn[0]["steps"]))

      # Positive Case D: XPROF_STEPS=0 omits profiler arguments and does NOT pass profiler=None.
      inv_xprof0 = tmp_path / "inv_xprof0.jsonl"
      proc_xprof0 = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH],
          env={
              **env_64c,
              "OUTPUT_DIR": (tmp_path / "out_xprof0").as_posix(),
              "STAGES": "MM",
              "BENCH_STEPS": "20",
              "XPROF_STEPS": "0",
              "SHIM_CAPTURE_ONLY": "1",
              "SHIM_INVOCATION_JSONL": inv_xprof0.as_posix(),
          },
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_xprof0.returncode, 0, msg=f"XPROF_STEPS=0 shim failed:\n{proc_xprof0.stderr}")
      invocations_xprof0 = [
          json.loads(line) for line in inv_xprof0.read_text(encoding="utf-8").splitlines() if line.strip()
      ]
      self.assertEqual(len(invocations_xprof0), 1)
      self.assertNotIn("profiler", invocations_xprof0[0])

      # Negative Control 0: When fallback full preset tpu7x_128c_text_bench_full_remat ALSO failed compilation, script fails!
      aot_all_fail_path = tmp_path / "bench_128c_aot_all_fail.json"
      aot_all_fail_path.write_text(
          json.dumps(
              {
                  "preset": "all_128c_bench_presets",
                  "passed": False,
                  "selected_remat_policy": None,
                  "results": {
                      "tpu7x_128c_mm_bench_min_remat": {
                          "compile_succeeded": False,
                          "passed": False,
                          "compile_error": "RESOURCE_EXHAUSTED mm_min",
                      },
                      "tpu7x_128c_text_bench_min_remat": {
                          "compile_succeeded": False,
                          "passed": False,
                          "compile_error": "RESOURCE_EXHAUSTED txt_min",
                      },
                      "tpu7x_128c_mm_bench_full_remat": {
                          "compile_succeeded": True,
                          "passed": True,
                          "temp_hbm_gib": 84.723,
                          "non_aliased_io_hbm_gib": 19.107,
                          "naive_sum_hbm_gib": 103.830,
                          "clamped_heap_cap_gib": 94.74,
                      },
                      "tpu7x_128c_text_bench_full_remat": {
                          "compile_succeeded": False,
                          "passed": False,
                          "compile_error": "RESOURCE_EXHAUSTED txt_full",
                      },
                  },
              }
          ),
          encoding="utf-8",
      )
      proc_all_fail = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH],
          env={
              **env_128c,
              "OUTPUT_DIR": (tmp_path / "out_128c_all_fail").as_posix(),
              "AOT_BATCH_JSON": aot_all_fail_path.as_posix(),
              "SHIM_INVOCATION_JSONL": "",
          },
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_all_fail.returncode, 0)
      self.assertIn("RESOURCE_EXHAUSTED txt_full", proc_all_fail.stderr)

      # Negative Control 1: Missing step-0 `MOE_DROPPED_FRAC` telemetry MUST fail closed when required.
      out_neg_missing_moe = tmp_path / "out_neg_missing_moe"
      proc_neg_missing_moe = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH],
          env={
              **env_128c,
              "OUTPUT_DIR": out_neg_missing_moe.as_posix(),
              "SHIM_OMIT_MOE_DROPPED": "1",
              "SHIM_INVOCATION_JSONL": "",
          },
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_neg_missing_moe.returncode, 0)
      neg_moe_summary = json.loads((out_neg_missing_moe / "bench_128c_text_summary.json").read_text(encoding="utf-8"))
      self.assertFalse(neg_moe_summary["passed"])
      self.assertIsNone(neg_moe_summary["moe_routing_mode"])
      self.assertIsNone(neg_moe_summary["ragged_buffer_factor"])
      self.assertIsNone(neg_moe_summary["step0_moe_dropped_frac"])
      self.assertIn("missing_step0_moe_dropped_frac_telemetry", neg_moe_summary["failure_reasons"])

      valid_txt_metrics = tmp_path / "moe_neg_metrics.jsonl"
      with valid_txt_metrics.open("w", encoding="utf-8") as mf:
        for s in range(30):
          mf.write(
              json.dumps(
                  {
                      "step": float(s),
                      "learning/loss": 11.5,
                      "perf/step_time_seconds": 6.0,
                      "perf/per_device_tflops": expected_text_tflops,
                      "perf/per_device_tflops_per_sec": expected_text_tflops / 6.0,
                      "perf/per_device_tokens_per_sec": 4096.0 / 6.0,
                  }
              )
              + "\n"
          )
      moe_pos_log = tmp_path / "moe_pos_only.log"
      moe_pos_log.write_text(
          "MOE_DROPPED_FRAC step=0 ragged_buffer_factor=2.0 mean=0.001250 max=0.005000 "
          "per_layer=[0.000000,0.002500,0.005000,0.000000]\n",
          encoding="utf-8",
      )
      moe_neg_log = tmp_path / "moe_neg_only.log"
      moe_neg_log.write_text("STAGE_START_UTC=128c_TEXT 2026-11-21T00:00:00Z\n", encoding="utf-8")
      moe_aot_rec = {
          "preset": "tpu7x_128c_text_bench_full_remat",
          "aot_compile_succeeded": True,
          "aot_temp_hbm_gib": 89.993,
          "aot_non_aliased_io_hbm_gib": 16.382,
          "aot_naive_sum_hbm_gib": 106.375,
          "aot_clamped_heap_cap_gib": 94.74,
          "aot_tpu7x_heap_limit_gib": 94.74,
      }
      moe_pos_summary = bench_mod.summarize_benchmark_stage(
          metrics_path=valid_txt_metrics,
          log_path=moe_pos_log,
          summary_json_path=tmp_path / "moe_kw_pos_summary.json",
          topology_mode="128c",
          modality="text",
          remat_policy="full",
          aot_record=moe_aot_rec,
          require_step0_moe_dropped_frac=True,
          bench_steps=30,
          xprof_steps=0,
      )
      self.assertTrue(moe_pos_summary["passed"])
      self.assertEqual(moe_pos_summary["moe_routing_mode"], "token_dropping")

      moe_kw_summary_path = tmp_path / "moe_kw_neg_summary.json"
      with self.assertRaisesRegex(AssertionError, "missing_step0_moe_dropped_frac_telemetry"):
        bench_mod.summarize_benchmark_stage(
            metrics_path=valid_txt_metrics,
            log_path=moe_neg_log,
            summary_json_path=moe_kw_summary_path,
            topology_mode="128c",
            modality="text",
            remat_policy="full",
            aot_record=moe_aot_rec,
            require_step0_moe_dropped_frac=True,
            bench_steps=30,
            xprof_steps=0,
        )
      moe_kw_summary = json.loads(moe_kw_summary_path.read_text(encoding="utf-8"))
      self.assertFalse(moe_kw_summary["passed"])
      self.assertIn("missing_step0_moe_dropped_frac_telemetry", moe_kw_summary["failure_reasons"])

      moe_env_summary_path = tmp_path / "moe_env_neg_summary.json"
      with mock.patch.dict(os.environ, {"DS41_LOG_STEP0_MOE_DROPPED": "1"}, clear=False):
        with self.assertRaisesRegex(AssertionError, "missing_step0_moe_dropped_frac_telemetry"):
          bench_mod.summarize_benchmark_stage(
              metrics_path=valid_txt_metrics,
              log_path=moe_neg_log,
              summary_json_path=moe_env_summary_path,
              topology_mode="128c",
              modality="text",
              remat_policy="full",
              aot_record=moe_aot_rec,
              bench_steps=30,
              xprof_steps=0,
          )
      moe_env_summary = json.loads(moe_env_summary_path.read_text(encoding="utf-8"))
      self.assertFalse(moe_env_summary["passed"])
      self.assertIn("missing_step0_moe_dropped_frac_telemetry", moe_env_summary["failure_reasons"])

      # Negative Control 2: NaN loss at step 15 MUST exit non-zero and mark loss_finite_all_bench_steps=False.
      out_neg_nan = tmp_path / "out_neg_nan"
      env_neg_nan = {
          **env_64c,
          "OUTPUT_DIR": out_neg_nan.as_posix(),
          "STAGES": "MM",
          "MM_AOT_HBM_GIB": "80.0",
          "SHIM_NAN_STEP": "15",
          "SHIM_INVOCATION_JSONL": "",
      }
      proc_neg_nan = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH],
          env=env_neg_nan,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_neg_nan.returncode, 0)
      neg_nan_summary = json.loads((out_neg_nan / "bench_64c_mm_summary.json").read_text(encoding="utf-8"))
      self.assertFalse(neg_nan_summary["passed"])
      self.assertFalse(neg_nan_summary["loss_finite_all_bench_steps"])
      self.assertIn(15, neg_nan_summary["nan_or_inf_steps"])

      # Negative Control 3: Wrong logged per_device_tflops (e.g. missing vision FLOPs on MM) fails bench_metrics.py.
      bad_metrics = tmp_path / "bad_tflops_metrics.jsonl"
      bad_log = tmp_path / "bad_tflops.log"
      bad_summary = tmp_path / "bad_tflops_summary.json"
      with bad_metrics.open("w", encoding="utf-8") as bf:
        for s in range(30):
          bf.write(
              json.dumps(
                  {
                      "step": float(s),
                      "learning/loss": 11.5,
                      "perf/step_time_seconds": 6.0,
                      "perf/per_device_tflops": expected_text_tflops,  # Wrong for modality='mm'!
                      "perf/per_device_tflops_per_sec": expected_text_tflops / 6.0,
                      "perf/per_device_tokens_per_sec": 4096.0 / 6.0,
                  }
              )
              + "\n"
          )
      bad_log.write_text("STAGE_START_UTC=64c_MM 2026-11-21T00:00:00Z\n", encoding="utf-8")
      with self.assertRaises(AssertionError):
        bench_mod.summarize_benchmark_stage(
            metrics_path=bad_metrics,
            log_path=bad_log,
            summary_json_path=bad_summary,
            topology_mode="64c",
            modality="mm",
            remat_policy="minimal",
            aot_record={
                "preset": "tpu7x_64c_mm_bench_min_remat",
                "aot_compile_succeeded": True,
                "aot_temp_hbm_gib": 82.0,
                "aot_non_aliased_io_hbm_gib": 10.0,
                "aot_naive_sum_hbm_gib": 92.0,
                "aot_clamped_heap_cap_gib": 92.0,
                "aot_tpu7x_heap_limit_gib": 94.74,
            },
            bench_steps=30,
            xprof_steps=0,
        )

      # Regression: ALLOW_UNINITIALIZED_BENCH=1 keeps use_random_routing=false across MM and TEXT,
      # explicit USE_RANDOM_ROUTING=true with MM fails fast, and incomplete steady window writes JSON before raising.
      inv_uninit = tmp_path / "inv_uninit.jsonl"
      out_uninit = tmp_path / "out_uninit"
      uninit_env = {
          **base_env,
          "TOPOLOGY_MODE": "64c",
          "OUTPUT_DIR": out_uninit.as_posix(),
          "ALLOW_UNINITIALIZED_BENCH": "1",
          "MM_AOT_HBM_GIB": "82.0",
          "TEXT_AOT_HBM_GIB": "80.0",
          "STAGES": "MM,TEXT",
          "SHIM_INVOCATION_JSONL": inv_uninit.as_posix(),
      }
      uninit_env.pop("USE_RANDOM_ROUTING", None)
      proc_uninit = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH], env=uninit_env, capture_output=True, text=True, check=False
      )
      self.assertEqual(proc_uninit.returncode, 0, msg=f"{proc_uninit.stdout}\n{proc_uninit.stderr}")
      inv_uninit_recs = [json.loads(ln) for ln in inv_uninit.read_text(encoding="utf-8").splitlines() if ln.strip()]
      self.assertEqual(len(inv_uninit_recs), 2)
      self.assertEqual([r["use_random_routing"] for r in inv_uninit_recs], ["false", "false"])

      proc_bad_rr = subprocess.run(
          ["bash", _BENCH_SCRIPT_SH],
          env={
              **uninit_env,
              "STAGES": "MM",
              "USE_RANDOM_ROUTING": "true",
              "OUTPUT_DIR": (tmp_path / "out_bad_rr").as_posix(),
          },
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_bad_rr.returncode, 0)
      self.assertIn("incompatible with stage MM", proc_bad_rr.stderr)

      inc_metrics = tmp_path / "inc_steady_metrics.jsonl"
      inc_metrics.write_text(
          "\n".join(
              json.dumps(
                  {
                      "step": float(s),
                      "learning/loss": 11.5,
                      "perf/step_time_seconds": 6.0,
                      "perf/per_device_tflops": expected_mm_tflops,
                      "perf/per_device_tflops_per_sec": expected_mm_tflops / 6.0,
                      "perf/per_device_tokens_per_sec": 4096.0 / 6.0,
                  }
              )
              for s in range(10)
          )
          + "\n",
          encoding="utf-8",
      )
      inc_summary = tmp_path / "inc_steady_summary.json"
      with self.assertRaises(AssertionError):
        bench_mod.summarize_benchmark_stage(
            metrics_path=inc_metrics,
            log_path=bad_log,
            summary_json_path=inc_summary,
            topology_mode="64c",
            modality="mm",
            remat_policy="minimal",
            aot_record={
                "preset": "tpu7x_64c_mm_bench_min_remat",
                "aot_compile_succeeded": True,
                "aot_temp_hbm_gib": 82.0,
                "aot_non_aliased_io_hbm_gib": 10.0,
                "aot_naive_sum_hbm_gib": 92.0,
                "aot_clamped_heap_cap_gib": 92.0,
                "aot_tpu7x_heap_limit_gib": 94.74,
            },
            require_step0_moe_dropped_frac=False,
            bench_steps=30,
            xprof_steps=0,
        )
      inc_payload = json.loads(inc_summary.read_text(encoding="utf-8"))
      self.assertFalse(inc_payload["passed"])
      self.assertTrue(any("incomplete_steady_window" in r for r in inc_payload["failure_reasons"]))

  def test_4_heap_compile_hbm_gate_failure_modes(self) -> None:
    """Test 4: Unclamped load-time HBM gate and heap compile OOM/non-HBM negative controls."""
    aot_mod = _load_module("ds41_aot_check_heap_gate", _AOT_CHECK_PY)
    bench_mod = _load_module("ds41_bench_metrics_heap_gate", _BENCH_METRICS_PY)
    gib_bytes = 1024**3

    def _make_compiled_hbm(temp_gib: float, io_gib: float, hlo_text: str) -> Any:
      mem_ns = types.SimpleNamespace(
          temp_size_in_bytes=int(round(temp_gib * gib_bytes)),
          argument_size_in_bytes=int(round(io_gib * gib_bytes)),
          output_size_in_bytes=int(round(io_gib * gib_bytes)),
          alias_size_in_bytes=int(round(io_gib * gib_bytes)),
      )
      return types.SimpleNamespace(memory_analysis=lambda: mem_ns, as_text=lambda: hlo_text)

    with tempfile.TemporaryDirectory(prefix="ds41_heap_gate_") as tmpdir:
      tmp_path = Path(tmpdir)

      # --- Failure Mode (2a): `temp_hbm_gib = 96.568 > 94.74` -> passed=False & AssertionError ---
      min_fail_2a_json = tmp_path / "min_fail_2a.json"
      fake_lowered_a9t = types.SimpleNamespace(
          as_text=lambda: "module @jit_train_step {}",
          compile=lambda compiler_options=None: _make_compiled_hbm(
              96.568, 16.382, "HloModule a9t_txt %p = bf16[5120] parameter(0)"
          ),
      )
      with mock.patch.object(jax, "jit", return_value=types.SimpleNamespace(lower=lambda *a, **kw: fake_lowered_a9t)):
        with mock.patch.object(maxtext_utils, "maybe_dump_jaxpr", return_value=None):
          with self.assertRaises(AssertionError) as ctx_2a:
            aot_mod.main(
                [
                    "--preset=tpu7x_128c_text_bench_min_remat",
                    "--cpu_mini=true",
                    "--max_target_length=64",
                    f"--output_json={min_fail_2a_json.as_posix()}",
                ]
            )
      self.assertIsInstance(ctx_2a.exception, aot_mod.AotHbmGateError)
      rec_2a = json.loads(min_fail_2a_json.read_text(encoding="utf-8"))
      self.assertTrue(rec_2a["compile_succeeded"])
      self.assertFalse(rec_2a["passed"])
      self.assertEqual(rec_2a["gate_mode"], "tpu7x_heap_compile")
      self.assertAlmostEqual(rec_2a["temp_hbm_gib"], 96.568, places=6)
      self.assertAlmostEqual(rec_2a["non_aliased_io_hbm_gib"], 16.382, places=6)
      self.assertAlmostEqual(rec_2a["naive_sum_hbm_gib"], 112.950, places=3)
      self.assertAlmostEqual(rec_2a["clamped_heap_cap_gib"], 94.74, places=6)
      self.assertNotIn("peak_hbm_gib", rec_2a)
      self.assertNotIn("effective_peak_hbm_gib", rec_2a)
      with self.assertRaises(AssertionError):
        bench_mod.extract_aot_record(rec_2a, preset_key="tpu7x_128c_text_bench_min_remat")

      # --- Failure Mode (2b): XLA HBM `RESOURCE_EXHAUSTED` compile exception -> compile_succeeded=False & raises ---
      min_fail_2b_json = tmp_path / "min_fail_2b.json"

      def _raise_xla_oom(compiler_options: Any = None) -> Any:
        del compiler_options
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating 103.8G on TPU7x")

      fake_lowered_oom = types.SimpleNamespace(
          as_text=lambda: "module @jit_train_step {}",
          compile=_raise_xla_oom,
      )
      with mock.patch.object(jax, "jit", return_value=types.SimpleNamespace(lower=lambda *a, **kw: fake_lowered_oom)):
        with mock.patch.object(maxtext_utils, "maybe_dump_jaxpr", return_value=None):
          with self.assertRaises(RuntimeError) as ctx_2b:
            aot_mod.main(
                [
                    "--preset=tpu7x_128c_text_bench_min_remat",
                    "--cpu_mini=true",
                    "--max_target_length=64",
                    f"--output_json={min_fail_2b_json.as_posix()}",
                ]
            )
      self.assertIsInstance(ctx_2b.exception, aot_mod.AotCompileOomError)
      rec_2b = json.loads(min_fail_2b_json.read_text(encoding="utf-8"))
      self.assertFalse(rec_2b["compile_succeeded"])
      self.assertFalse(rec_2b["passed"])
      self.assertIn("RESOURCE_EXHAUSTED", rec_2b["compile_error"])
      self.assertIsNone(rec_2b["temp_hbm_gib"])
      self.assertNotIn("peak_hbm_gib", rec_2b)
      self.assertNotIn("effective_peak_hbm_gib", rec_2b)

      # Verify select_remat_and_aot_hbm raises AssertionError if BOTH minimal (2b) and full (2b/2a) fail!
      with self.assertRaises(AssertionError):
        bench_mod.select_remat_and_aot_hbm(
            topology_mode="128c",
            enabled_stages=["TEXT"],
            text_aot_json=min_fail_2b_json.as_posix(),
            text_fallback_aot_json=min_fail_2a_json.as_posix(),
        )

      # --- Failure Mode (2c): Non-HBM compile exceptions MUST propagate immediately WITHOUT falling back to `full`! ---
      orig_run_preset = aot_mod.run_preset

      def _make_dispatch_2c(target_batch: str, err_obj: BaseException, calls_out: list[str]) -> Any:
        def _dispatch(p: str, **kw: Any) -> dict[str, Any]:
          if p == target_batch:
            return orig_run_preset(p, **kw)
          calls_out.append(p)
          raise err_obj

        return _dispatch

      for batch_name, first_min_preset, non_hbm_exc, expected_exc_cls in (
          (
              "all_64c_bench_presets",
              "tpu7x_64c_mm_bench_min_remat",
              ValueError("shape mismatch in attention_deepseek41"),
              ValueError,
          ),
          (
              "all_128c_bench_presets",
              "tpu7x_128c_mm_bench_min_remat",
              jax.errors.JaxRuntimeError("INTERNAL: unexpected HLO opcode"),
              jax.errors.JaxRuntimeError,
          ),
          (
              "all_64c_bench_presets",
              "tpu7x_64c_mm_bench_min_remat",
              AssertionError("Expected all vision HLO markers in compiled HLO"),
              AssertionError,
          ),
      ):
        called_presets: list[str] = []
        with mock.patch.object(
            aot_mod,
            "run_preset",
            side_effect=_make_dispatch_2c(batch_name, non_hbm_exc, called_presets),
        ):
          with self.assertRaises(expected_exc_cls):
            aot_mod.run_preset(batch_name)
        # Prove execution aborted on the first minimal preset and never touched full-remat fallbacks!
        self.assertEqual(called_presets, [first_min_preset])

      # Also verify `_compile_lowered_or_raise` directly re-raises non-OOM `JaxRuntimeError` and `ValueError`.
      for raw_non_oom in (
          ValueError("invalid sharding spec"),
          jax.errors.JaxRuntimeError("INVALID_ARGUMENT: failed to legalize mhlo"),
      ):
        bad_lowered = types.SimpleNamespace(
            compile=mock.Mock(side_effect=raw_non_oom),
        )
        with self.assertRaises(type(raw_non_oom)):
          aot_mod._compile_lowered_or_raise(  # pylint: disable=protected-access
              bad_lowered,
              compiler_options={},
              preset_label="tpu7x_64c_mm_bench_full_remat",
              max_hbm_gib=94.74,
              use_heap_compile_gate=True,
              base_failure_fields={"preset": "tpu7x_64c_mm_bench_full_remat"},
          )

  def test_5_tpu_orbax_restore_checkpointing_flags_and_zero_save_invariants(self) -> None:
    """Test 5: Multi-host TPU `jax.distributed` init via `enable_checkpointing=True` + zero-save invariants & negctls."""
    aot_mod = _load_module("ds41_aot_check_orbax", _AOT_CHECK_PY)
    bench_ov = dict(aot_mod.get_preset_spec("tpu7x_64c_mm_bench_full_remat")["overrides"])
    # Runtime benchmark execution on TPU clears AOT compile_topology and sets steps=33 (30 bench + 3 xprof).
    bench_ov["compile_topology"] = ""
    bench_ov["compile_topology_num_slices"] = -1
    bench_ov["steps"] = 33
    bench_ov["learning_rate_schedule_steps"] = 33
    bench_ov["load_parameters_path"] = "gs://maxtext-deepseek/deepseek-v4.1-flash/orbax/scanned/0/items"
    bench_ov["source_checkpoint_layout"] = "orbax"

    cfg = pyconfig.initialize([sys.argv[0], _BASE_CONFIG_YML], **bench_ov)
    self.assertTrue(cfg.enable_checkpointing)
    self.assertFalse(cfg.async_checkpointing)
    self.assertFalse(cfg.save_checkpoint_on_start)
    self.assertFalse(cfg.save_checkpoint_on_completion)
    self.assertEqual(cfg.checkpoint_period, 100000)
    self.assertGreater(cfg.checkpoint_period, cfg.steps)

    # 1. Verify `3_bench_deepseek41_mm.sh` CLI flags contract (default 33 steps and BENCH_STEPS=200000).
    with tempfile.TemporaryDirectory(prefix="ds41_orbax_cli_") as tmpdir:
      tmp_path = Path(tmpdir)
      shim_py = tmp_path / "python_capture_shim.py"
      shim_py.write_text(
          f"""#!{sys.executable}
import json
import os
import sys

if len(sys.argv) >= 2 and sys.argv[1].endswith("train.py"):
  kv = {{}}
  for arg in sys.argv[3:]:
    if "=" in arg:
      k, v = arg.split("=", 1)
      kv[k] = v
  with open(os.environ["SHIM_INVOCATION_JSONL"], "a", encoding="utf-8") as f:
    f.write(json.dumps(kv, sort_keys=True) + "\\n")
  sys.exit(0)
if len(sys.argv) >= 2 and sys.argv[1].endswith("bench_metrics.py") and "--mode=summarize_bench" in sys.argv:
  sys.exit(0)
os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])
""",
          encoding="utf-8",
      )
      shim_py.chmod(shim_py.stat().st_mode | stat.S_IEXEC)
      for bench_steps_str, expected_steps, expected_period in (
          ("30", "33", "100000"),
          ("200000", "200003", "201003"),
      ):
        inv_file = tmp_path / f"inv_{bench_steps_str}.jsonl"
        env_run = {
            **os.environ,
            "PYTHONPATH": f"{(_REPO_ROOT / 'src').as_posix()}:{_REPO_ROOT.as_posix()}",
            "PYTHON_BIN": shim_py.as_posix(),
            "TOPOLOGY_MODE": "64c",
            "STAGES": "MM",
            "BENCH_STEPS": bench_steps_str,
            "XPROF_STEPS": "3",
            "OUTPUT_DIR": (tmp_path / f"out_{bench_steps_str}").as_posix(),
            "LOAD_PARAMETERS_PATH": "gs://maxtext-deepseek/deepseek-v4.1-flash/orbax/scanned/0/items",
            "MM_AOT_HBM_GIB": "82.0",
            "TEXT_AOT_HBM_GIB": "80.0",
            "SHIM_INVOCATION_JSONL": inv_file.as_posix(),
        }
        proc_cli = subprocess.run(["bash", _BENCH_SCRIPT_SH], env=env_run, capture_output=True, text=True, check=False)
        self.assertEqual(proc_cli.returncode, 0, msg=f"3_bench shim failed:\n{proc_cli.stderr}")
        recs = [json.loads(line) for line in inv_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["enable_checkpointing"], "true")
        self.assertEqual(recs[0]["async_checkpointing"], "false")
        self.assertEqual(recs[0]["save_checkpoint_on_start"], "false")
        self.assertEqual(recs[0]["save_checkpoint_on_completion"], "false")
        self.assertEqual(recs[0]["steps"], expected_steps)
        self.assertEqual(recs[0]["checkpoint_period"], expected_period)
        self.assertGreater(int(recs[0]["checkpoint_period"]), int(recs[0]["steps"]))

    # 2. Verify `max_utils.maybe_initialize_jax_distributed_system` initializes `jax.distributed` on multi-host TPU
    # with the benchmark config (`enable_checkpointing=True`), whereas `enable_checkpointing=False` (negative control)
    # skips initialization and causes multi-host Orbax restore to fail with ThreadSafeKeyValueSignalingClient error.
    raw_keys = dict(cfg.get_keys())
    with (
        mock.patch.object(jax.distributed, "is_initialized", return_value=False),
        mock.patch.object(max_utils, "is_gpu_backend", return_value=False),
        mock.patch.object(max_utils, "is_cpu_backend", return_value=False),
        mock.patch.object(jax.distributed, "initialize") as mock_dist_init,
    ):
      max_utils.maybe_initialize_jax_distributed_system(raw_keys)
      mock_dist_init.assert_called_once_with(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])

      mock_dist_init.reset_mock()
      raw_disabled = dict(
          raw_keys,
          enable_checkpointing=False,
          load_parameters_path="",
          load_full_state_path="",
          source_checkpoint_layout="orbax",
      )
      max_utils.maybe_initialize_jax_distributed_system(raw_disabled)
      mock_dist_init.assert_not_called()

    # 3. Zero-save invariant across all benchmark steps 0..32 and completion (`save_checkpoint_on_completion=False`).
    ckpt_mgr = mock.MagicMock()
    ckpt_mgr.latest = None
    ckpt_mgr.checkpoints = []
    dummy_state = types.SimpleNamespace(step=32)
    with mock.patch.object(checkpointing, "save_checkpoint", return_value=True) as mock_save_ckpt:
      for step in range(0, int(cfg.steps)):
        self.assertFalse(
            checkpointing._should_save_checkpoint_at_step(ckpt_mgr, step, cfg, False),  # pylint: disable=protected-access
            msg=f"_should_save_checkpoint_at_step must be False at step={step}",
        )
        checkpointing.maybe_save_checkpoint(ckpt_mgr, dummy_state, cfg, data_iterator=None, step=step)
      # Mirror `train.py:1594-1595`: completion save runs ONLY when `save_checkpoint_on_completion` is True.
      if cfg.save_checkpoint_on_completion:
        checkpointing.maybe_save_checkpoint(ckpt_mgr, dummy_state, cfg, data_iterator=None)
      mock_save_ckpt.assert_not_called()
      ckpt_mgr.save.assert_not_called()

    # 4. Negative Control A: Omitting `save_checkpoint_on_start=False` (`base.yml` defaults to `True`) triggers a
    # 578 GB checkpoint write at step 0 (`0 % 100000 == 0`)!
    cfg_bad_start = pyconfig.initialize(
        [sys.argv[0], _BASE_CONFIG_YML],
        **{**bench_ov, "save_checkpoint_on_start": True},
    )
    self.assertTrue(
        checkpointing._should_save_checkpoint_at_step(  # pylint: disable=protected-access
            ckpt_mgr, 0, cfg_bad_start, False
        )
    )
    with (
        mock.patch.object(checkpointing, "save_checkpoint", return_value=True) as mock_save_bad_start,
        mock.patch.object(checkpointing, "print_save_message"),
    ):
      checkpointing.maybe_save_checkpoint(ckpt_mgr, dummy_state, cfg_bad_start, data_iterator=None, step=0)
      mock_save_bad_start.assert_called_once_with(ckpt_mgr, 0, dummy_state, cfg_bad_start, None, False)

    # 5. Negative Control B: Setting `checkpoint_period=10 <= steps=33` triggers periodic saves at steps 10, 20, 30!
    cfg_bad_period = pyconfig.initialize(
        [sys.argv[0], _BASE_CONFIG_YML],
        **{**bench_ov, "checkpoint_period": 10},
    )
    triggered_steps = [
        s
        for s in range(0, int(cfg_bad_period.steps))
        if checkpointing._should_save_checkpoint_at_step(  # pylint: disable=protected-access
            ckpt_mgr, s, cfg_bad_period, False
        )
    ]
    self.assertEqual(triggered_steps, [10, 20, 30])

  def test_6_ds41_pinned_batch_npz_export_and_load_contract(self) -> None:
    """Proves multi-host PinnedBatchIterator export and load contract at per_device_batch_size = 0.5."""
    from tests.utils import deepseek_v41_pinned_batch  # pylint: disable=import-outside-toplevel

    mesh = mm_testing.single_device_mesh()
    with tempfile.TemporaryDirectory() as tmp_dir:
      export_npz = os.path.join(tmp_dir, "test_pinned_batch.npz")
      cfg = pyconfig.initialize(
          [sys.argv[0], _BASE_CONFIG_YML],
          per_device_batch_size=0.5,
          max_target_length=200,
          use_multimodal=True,
          vision_encoder_block="deepseek41",
          ds41_image_token_id=100,
          vocab_size=1000,
          ds41_vision_grid=[39, 39],
          max_num_images_per_example=1,
          ds41_vision_downsample_ratio=3,
          num_channels_for_vit=3,
          patch_size_for_vit=14,
      )
      object.__setattr__(cfg, "global_batch_size_to_load", 16)
      object.__setattr__(cfg, "micro_batch_size_to_train_on", 16)
      # 1. Export path
      syn_it = synthetic_data_processing.SyntheticDataIterator(cfg, mesh)
      syn_batch = next(syn_it)
      deepseek_v41_pinned_batch.export_pinned_batch(syn_batch, export_npz)
      self.assertTrue(os.path.exists(export_npz))
      with np.load(export_npz) as data:
        self.assertIn("tokens", data)
        self.assertIn("batch_positions", data)
        self.assertIn("segmentation", data)
        self.assertIn("targets_segmentation", data)
        self.assertIn("images", data)
        self.assertEqual(data["tokens"].shape[0], 16)
        self.assertEqual(data["images"].shape[0], 16)

      with open(export_npz, "rb") as f:
        valid_sha = hashlib.sha256(f.read()).hexdigest()

      # 2. Load path with valid SHA256
      it_load = deepseek_v41_pinned_batch.PinnedBatchIterator(syn_it, mesh, export_npz, valid_sha)
      batch = next(it_load)
      self.assertIn("inputs", batch)
      self.assertIn("images", batch)
      self.assertEqual(batch["inputs"].shape[0], 16)
      self.assertEqual(batch["images"].shape[0], 16)
      np.testing.assert_array_equal(np.asarray(batch["inputs"]), np.asarray(syn_batch["inputs"]))
      np.testing.assert_array_equal(np.asarray(batch["targets"]), np.asarray(syn_batch["targets"]))
      np.testing.assert_array_equal(np.asarray(batch["images"]), np.asarray(syn_batch["images"]))

      # 3. Negative Control A: Corrupted SHA-256 raises ValueError fail-closed
      corrupted_sha = "0" * 64 if valid_sha != "0" * 64 else "1" * 64
      with self.assertRaisesRegex(ValueError, "fail-closed mismatch"):
        deepseek_v41_pinned_batch.PinnedBatchIterator(cfg, mesh, export_npz, corrupted_sha)

      # Negative Control: a pinned batch without an expected SHA-256 fails closed.
      with self.assertRaisesRegex(ValueError, "requires.*sha256"):
        deepseek_v41_pinned_batch.PinnedBatchIterator(cfg, mesh, export_npz, "")

      # 4. Negative Control B: micro_batch_size_to_train_on != 16 raises AssertionError
      object.__setattr__(cfg, "micro_batch_size_to_train_on", 8)
      with self.assertRaisesRegex(AssertionError, "micro_batch_size_to_train_on == 16"):
        deepseek_v41_pinned_batch.PinnedBatchIterator(cfg, mesh, export_npz, valid_sha)

  def test_7_base_config_path_and_non_offload_load_time_hbm_gate(self) -> None:
    """Verifies shared BASE_CONFIG_PATH wiring and evaluate_load_time_hbm_gate offload rules."""
    aot_mod = _load_module("ds41_aot_check_gate_fix", _AOT_CHECK_PY)
    bench_mod = _load_module("ds41_bench_metrics_gate_fix", _BENCH_METRICS_PY)
    expected_base = str(Path(MAXTEXT_CONFIGS_DIR) / "base.yml")
    self.assertEqual(aot_mod.BASE_CONFIG_PATH, expected_base)
    self.assertEqual(bench_mod.BASE_CONFIG_PATH, expected_base)

    # optimizer_memory_host_offload=False allows opt_hbm_gib > 0; True rejects it.
    res_no_offload = aot_mod.evaluate_load_time_hbm_gate(
        preset_name="tpu7x_64c_mm_train_adamw",
        topology="tpu7x-128",
        remat_policy="full",
        optimizer_memory_host_offload=False,
        compile_succeeded=True,
        opt_hbm_gib=5.0,
        opt_pinned_host_gib=0.0,
        temp_hbm_gib=10.0,
        non_aliased_io_hbm_gib=1.0,
        generated_code_hbm_gib=0.1,
        max_hbm_gib=94.74,
    )
    self.assertTrue(res_no_offload["gate_passed"], msg=str(res_no_offload))
    res_offload_bad = aot_mod.evaluate_load_time_hbm_gate(
        preset_name="tpu7x_64c_mm_bench_min_remat",
        topology="tpu7x-128",
        remat_policy="minimal",
        optimizer_memory_host_offload=True,
        compile_succeeded=True,
        opt_hbm_gib=5.0,
        opt_pinned_host_gib=20.0,
        temp_hbm_gib=10.0,
        non_aliased_io_hbm_gib=1.0,
        generated_code_hbm_gib=0.1,
        max_hbm_gib=94.74,
    )
    self.assertFalse(res_offload_bad["gate_passed"])


if __name__ == "__main__":
  unittest.main()
