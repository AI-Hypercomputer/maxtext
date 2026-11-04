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

"""Unit tests for DeepSeek-V4.1 multimodal training E2E support and staged TPU script (U6c)."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.multimodal import processor as mm_processor
from maxtext.trainers.pre_train import train
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
import numpy as np

from tests.utils import deepseek_v41_mm_audit

try:
  import torch
  from tests.utils.deepseek_v41_reference import multimodal

  HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
  torch = None  # type: ignore[assignment]
  multimodal = None  # type: ignore[assignment]
  HAS_TORCH = False

from tests.utils import deepseek_v41_mm_testing as mm_testing

_REPO_ROOT = Path(__file__).resolve().parents[2]
_E2E_DIR = _REPO_ROOT / "tests" / "end_to_end" / "tpu" / "deepseek" / "v4.1-flash"
_THRESHOLDS_JSON = (_E2E_DIR / "thresholds.json").as_posix()
_PIN_JSON = (_E2E_DIR / "golden_manifest_pin.json").as_posix()
_STAGE_SCRIPT = (_E2E_DIR / "2_test_deepseek41_mm.sh").as_posix()
_AOT_CHECK_PY = (_E2E_DIR / "aot_check.py").as_posix()
_AUDIT_PY = (_REPO_ROOT / "tests" / "utils" / "deepseek_v41_mm_audit.py").as_posix()


def _load_aot_check_module() -> Any:
  """Loads `tests/end_to_end/tpu/deepseek/v4.1-flash/aot_check.py` as a module."""
  spec = importlib.util.spec_from_file_location("ds41_aot_check_u6c", _AOT_CHECK_PY)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_AOT_CHECK_PY}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _init_mini_mm_config(*, grid: tuple[int, int] | None = None, **overrides: Any) -> Any:
  """Initializes a 14-layer DeepSeek-V4.1-Flash mini multimodal config."""
  if grid is None:
    grid = multimodal.MINI_GRID_PADDED if multimodal is not None else (7, 5)
  defaults: dict[str, Any] = {
      "run_name": "ds41_mm_u6c_unit",
      "save_checkpoint_on_completion": False,
      "dataset_type": "synthetic",
      "per_device_batch_size": 1.0,
      "max_target_length": 64,
      "ici_fsdp_parallelism": 1,
      "ici_data_parallelism": -1,
      "ici_expert_parallelism": 1,
      "ici_tensor_parallelism": 1,
  }
  defaults.update(overrides)
  return mm_testing.mini_vision_maxtext_config(grid=grid, **defaults)


def _cpu_group_f32_l2_hex(
    reader: deepseek_v41_mm_audit._SafetensorsShardReader,  # pylint: disable=protected-access
    keys: list[str],
) -> str:
  """Computes exact float32 L2 norm hex string across `keys` matching `train._f32_l2_norm`."""
  sq_sum = np.float32(0.0)
  for key in keys:
    arr = reader.load_dequantized_tensor(key, "float32").astype(np.float32)
    sq_sum = np.float32(sq_sum + np.sum(np.square(arr), dtype=np.float32))
  return float(np.float32(np.sqrt(sq_sum, dtype=np.float32))).hex()


def _build_synthetic_r4_trace(
    bias_vl_hex_list: list[str],
    *,
    num_steps: int = 20,
    peak_hbm_gb: float | None = 76.25,
    bytes_in_use_gib: float | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
  """Builds a 20-step R4 stdout log and `metrics_file` record sequence."""
  log_lines: list[str] = []
  metric_records: list[dict[str, Any]] = []
  vit_hex = float(np.float32(12.5)).hex()
  delim_hex = float(np.float32(1.75)).hex()
  bvl_str = ",".join(bias_vl_hex_list)
  eff_in_use = bytes_in_use_gib if bytes_in_use_gib is not None else peak_hbm_gb
  hbm_suffix = ""
  if peak_hbm_gb is not None:
    hbm_suffix += f" peak_hbm_gb={peak_hbm_gb:.6f} live_peak_hbm_gib={peak_hbm_gb:.6f}"
  if eff_in_use is not None:
    hbm_suffix += f" bytes_in_use_gib={eff_in_use:.6f}"

  for step in range(num_steps):
    loss_val = 13.5 - 0.30 * min(step, 4) - 0.06 * max(0, step - 4)
    moe_lb_val = 1.5e-4
    align_val = np.float32(4.0 + 0.02 * step)
    align_hex = float(align_val).hex()
    delim_hex = float(np.float32(1.75 + 0.015 * step)).hex()
    log_lines.append(
        f"completed step: {step}, seconds: 1.000, TFLOP/s/device: 100.000, "
        f"Tokens/s/device: 512.000, total_weights: 5248, loss: {loss_val:.3f}, "
        f"lm_loss: {loss_val:.3f}, perplexity: 100.000, moe_lb_loss: {moe_lb_val:.6f}"
    )
    log_lines.append(
        f"DS41_MM_NORMS step={step} vit_norm={vit_hex} "
        f"aligner_norm={align_hex} delim_norm={delim_hex} bias_vl_norm=[{bvl_str}]{hbm_suffix}"
    )
    metric_records.append(
        {
            "step": float(step),
            "run_name": "ds41_mm_r4_train",
            "learning/loss": loss_val,
            "learning/lm_loss": loss_val,
            "learning/moe_lb_loss": moe_lb_val,
            "learning/raw_grad_norm": 12.0 + 0.5 * step,
            "learning/grad_norm": 1.0,
        }
    )
  return log_lines, metric_records


class DeepSeekV41MMTrainE2ETest(unittest.TestCase):
  """CPU unit tests for U6c training support, R4 gate verifier, and stage runner script."""

  tmp_dir: str = ""
  ckpt_dir: str = ""
  ckpt_vit_norm_hex: str = ""
  ckpt_aligner_norm_hex: str = ""
  ckpt_delim_norm_hex: str = ""
  ckpt_bias_vl_norms_hex: list[str] = []

  @classmethod
  def setUpClass(cls) -> None:
    """Initializes the shared 14-layer mini multimodal safetensors checkpoint and CPU oracle norms."""
    super().setUpClass()
    cls.tmp_dir = tempfile.mkdtemp(prefix="ds41_u6c_train_")
    if HAS_TORCH:
      cls.ckpt_dir = os.path.join(cls.tmp_dir, "mini_safetensors")
      args = multimodal.mini_model_args()
      ref_mm = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
      multimodal.init_mini_reference(ref_mm, seed=91)
      offset = torch.tensor([-0.35, 0.35, -0.20, 0.20], dtype=torch.float32)
      with torch.no_grad():
        for layer in ref_mm.layers:
          layer.ffn.gate.bias_vl.copy_(layer.ffn.gate.bias + offset)
      mm_testing.write_mini_safetensors_checkpoint(cls.ckpt_dir, ref_mm, param_dtype=torch.float32)
      del ref_mm
      gc.collect()

      reader = deepseek_v41_mm_audit._SafetensorsShardReader(cls.ckpt_dir)  # pylint: disable=protected-access
      all_keys = sorted(reader.all_keys())
      vit_keys = [k for k in all_keys if k.startswith("vision.")]
      aligner_keys = ["aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias"]
      delim_keys = ["image_start", "image_end", "image_newline"]
      cls.ckpt_vit_norm_hex = _cpu_group_f32_l2_hex(reader, vit_keys)
      cls.ckpt_aligner_norm_hex = _cpu_group_f32_l2_hex(reader, aligner_keys)
      cls.ckpt_delim_norm_hex = _cpu_group_f32_l2_hex(reader, delim_keys)
      cls.ckpt_bias_vl_norms_hex = deepseek_v41_mm_audit.load_hf_bias_vl_norms_hex(cls.ckpt_dir)
    else:
      cls.ckpt_dir = ""
      cls.ckpt_vit_norm_hex = ""
      cls.ckpt_aligner_norm_hex = ""
      cls.ckpt_delim_norm_hex = ""
      cls.ckpt_bias_vl_norms_hex = []

  @classmethod
  def tearDownClass(cls) -> None:
    """Cleans up the shared temporary directory."""
    if cls.tmp_dir:
      shutil.rmtree(cls.tmp_dir, ignore_errors=True)
      cls.tmp_dir = ""
    super().tearDownClass()

  def _run_two_step_cpu_train_loop(self, *, scan_layers: bool) -> None:
    """Runs a 2-step CPU training loop and checks checkpoint norms, loss decrease, and frozen-ViT invariants."""
    assert self.tmp_dir
    run_out_dir = os.path.join(self.tmp_dir, f"out_scan_{int(scan_layers)}")
    metrics_path = os.path.join(run_out_dir, "metrics.jsonl")
    log_path = os.path.join(run_out_dir, "train.log")
    os.makedirs(run_out_dir, exist_ok=True)

    if self.ckpt_dir:
      cfg = _init_mini_mm_config(
          grid=multimodal.MINI_GRID_PADDED,
          base_output_directory=run_out_dir,
          load_parameters_path=self.ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
          scan_layers=scan_layers,
          freeze_vision_encoder_params=True,
          log_moe_bias_norms=True,
          routed_bias=True,
          routed_bias_update_rate=1e-3,
          dataset_type="synthetic",
          max_num_images_per_example=1,
          max_target_length=32,
          per_device_batch_size=1.0,
          steps=2,
          learning_rate_schedule_steps=2,
          learning_rate=1e-3,
          warmup_steps_fraction=0.0,
          adam_weight_decay=0.1,
          metrics_file=metrics_path,
      )

      # Unconditional pre-step check: freshly loaded model matches all 4 deterministic CPU checkpoint norms
      # identically under both scan_layers=False and scan_layers=True (even when run in isolation with -k).
      devices_array = maxtext_utils.create_device_mesh(cfg)
      mesh = Mesh(devices_array, cfg.mesh_axes)
      pre_model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      pre_vec = np.asarray(
          jax.device_get(train._compute_ds41_mm_norms_vector(cfg, pre_model)),  # pylint: disable=protected-access
          dtype=np.float32,
      )
      pre_hex = [float(np.float32(v)).hex() for v in pre_vec]
      self.assertEqual(pre_hex[0], self.ckpt_vit_norm_hex)
      self.assertEqual(pre_hex[1], self.ckpt_aligner_norm_hex)
      self.assertEqual(pre_hex[2], self.ckpt_delim_norm_hex)
      self.assertEqual(pre_hex[3:], self.ckpt_bias_vl_norms_hex)

      # Verify live TPU HBM formatting on _maybe_log_ds41_mm_norms when memory_stats() reports peak_bytes_in_use.
      fake_dev = mock.Mock()
      fake_dev.memory_stats.return_value = {"peak_bytes_in_use": int(75.5 * (1024**3))}
      with mock.patch.object(jax, "local_devices", return_value=[fake_dev]):
        hbm_line = train._maybe_log_ds41_mm_norms(cfg, 0, {"ds41_mm_norms": pre_vec})  # pylint: disable=protected-access
      self.assertIsNotNone(hbm_line)
      assert hbm_line is not None
      self.assertIn("peak_hbm_gb=75.500000", hbm_line)
      parsed_hbm = deepseek_v41_mm_audit.parse_r4_log_text(hbm_line)
      self.assertEqual(parsed_hbm["peak_hbm_gb"], 75.5)
      del pre_model
    else:
      cfg = _init_mini_mm_config(
          grid=(7, 5),
          base_output_directory=run_out_dir,
          scan_layers=scan_layers,
          freeze_vision_encoder_params=True,
          log_moe_bias_norms=True,
          routed_bias=True,
          routed_bias_update_rate=1e-3,
          dataset_type="synthetic",
          max_num_images_per_example=1,
          max_target_length=32,
          per_device_batch_size=1.0,
          steps=2,
          learning_rate_schedule_steps=2,
          learning_rate=1e-3,
          warmup_steps_fraction=0.0,
          adam_weight_decay=0.1,
          metrics_file=metrics_path,
      )

    captured_logs: list[str] = []
    orig_log = max_logging.log

    def _capture_log(msg: str, *a: Any, **kw: Any) -> None:
      captured_logs.append(str(msg))
      orig_log(msg, *a, **kw)

    with mock.patch.object(max_logging, "log", side_effect=_capture_log):
      train.train_loop(cfg, recorder=None)

    Path(log_path).write_text("\n".join(captured_logs), encoding="utf-8")
    parsed_log = deepseek_v41_mm_audit.parse_r4_log(log_path)
    mm_norms = parsed_log["mm_norms"]
    self.assertIn(0, mm_norms)
    self.assertIn(1, mm_norms)

    metrics_by_step = deepseek_v41_mm_audit.parse_r4_metrics_file(metrics_path)
    self.assertIn(0, metrics_by_step)
    self.assertIn(1, metrics_by_step)
    loss_0 = float(metrics_by_step[0]["learning/loss"])
    loss_1 = float(metrics_by_step[1]["learning/loss"])
    self.assertTrue(np.isfinite(loss_0) and np.isfinite(loss_1))
    self.assertLess(loss_1, loss_0, msg=f"Expected loss(1)={loss_1} < loss(0)={loss_0}")

    if self.ckpt_dir:
      # Frozen ViT and static MoEBiasVar bias_vl must match the deterministic CPU checkpoint norms at steps 0 and 1.
      self.assertEqual(len(self.ckpt_bias_vl_norms_hex), 14)
      self.assertEqual(mm_norms[0]["vit_norm"], self.ckpt_vit_norm_hex)
      self.assertEqual(mm_norms[1]["vit_norm"], self.ckpt_vit_norm_hex)
      self.assertEqual(mm_norms[0]["bias_vl_norm"], self.ckpt_bias_vl_norms_hex)
      self.assertEqual(mm_norms[1]["bias_vl_norm"], self.ckpt_bias_vl_norms_hex)

      # Trainable projector/delimiter weights update after apply_gradients at step 0 and again at step 1.
      self.assertNotEqual(mm_norms[0]["aligner_norm"], self.ckpt_aligner_norm_hex)
      self.assertNotEqual(mm_norms[1]["aligner_norm"], mm_norms[0]["aligner_norm"])
      self.assertNotEqual(mm_norms[0]["delim_norm"], self.ckpt_delim_norm_hex)
    else:
      self.assertIn("vit_norm", mm_norms[0])
      self.assertIn("vit_norm", mm_norms[1])
      self.assertEqual(mm_norms[0]["vit_norm"], mm_norms[1]["vit_norm"])
      self.assertIn("aligner_norm", mm_norms[0])
      self.assertIn("aligner_norm", mm_norms[1])
      self.assertNotEqual(mm_norms[0]["aligner_norm"], mm_norms[1]["aligner_norm"])

  def test_dummy_image_shape_and_synthetic_iterator_n1_and_n4(self) -> None:
    """Verifies N=1/N=4 shapes, synthetic spans, target masking, and packing/vocab bounds."""
    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices), 1), devices), ("data", "fsdp"))

    cases = [
        ((39, 39), -1, 1, 1521, 184, 512),
        ((39, 39), 1, 1, 1521, 184, 512),
        ((39, 39), 4, 4, 1521, 184, 1024),
        ((7, 5), 1, 1, 35, 11, 64),
        ((7, 5), 4, 4, 35, 11, 64),
        ((6, 9), 1, 1, 54, 10, 64),
        ((6, 9), 4, 4, 54, 10, 64),
    ]
    for grid, cfg_max_images, expected_n, expected_patches, expected_span_len, seq_len in cases:
      with self.subTest(grid=grid, cfg_max_images=cfg_max_images):
        cfg = _init_mini_mm_config(
            grid=grid,
            max_num_images_per_example=cfg_max_images,
            max_target_length=seq_len,
            per_device_batch_size=1.0,
            packing=True,
            max_segments_per_seq=1,
        )
        batch_size = int(cfg.global_batch_size_to_load)
        expected_img_shape = (batch_size, expected_n, expected_patches, 588)

        dummy_shape = mm_processor.get_dummy_image_shape_for_init(
            cfg.model_name,
            batch_size=batch_size,
            config=cfg,
        )
        self.assertEqual(dummy_shape, expected_img_shape)

        override_shape = mm_processor.get_dummy_image_shape_for_init(
            cfg.model_name,
            batch_size=batch_size,
            num_image_per_sequence=3,
            config=cfg,
        )
        self.assertEqual(override_shape, (batch_size, 3, expected_patches, 588))

        shaped = maxtext_utils.get_shaped_batch(cfg)
        self.assertIn("images", shaped)
        self.assertEqual(shaped["images"].shape, expected_img_shape)
        self.assertEqual(shaped["images"].dtype, jnp.float32)
        self.assertNotIn("image_masks", shaped)

        iterator = SyntheticDataIterator(cfg, mesh)
        batch = next(iterator)
        self.assertIn("images", batch)
        self.assertEqual(batch["images"].shape, expected_img_shape)
        self.assertEqual(batch["images"].dtype, jnp.float32)

        inputs = np.asarray(batch["inputs"])
        targets = np.asarray(batch["targets"])
        inputs_seg = np.asarray(batch["inputs_segmentation"])
        targets_seg = np.asarray(batch["targets_segmentation"])
        image_token_id = int(cfg.ds41_image_token_id)
        total_span = expected_n * expected_span_len

        self.assertEqual(inputs_seg.shape, (batch_size, seq_len))
        self.assertEqual(targets_seg.shape, (batch_size, seq_len))
        self.assertTrue(np.all(inputs[:, :8] != image_token_id))
        self.assertTrue(np.all(inputs[:, 8 : 8 + total_span] == image_token_id))
        self.assertTrue(np.all(inputs[:, 8 + total_span :] != image_token_id))
        self.assertTrue(np.all(inputs_seg == 1))

        self.assertTrue(np.all(targets[:, :7] != image_token_id))
        self.assertTrue(np.all(targets[:, 7 : 7 + total_span] == image_token_id))
        self.assertTrue(np.all(targets[:, 7 + total_span :] != image_token_id))
        self.assertTrue(np.all(targets_seg[:, :7] == 1))
        self.assertTrue(np.all(targets_seg[:, 7 : 7 + total_span] == 0))
        self.assertTrue(np.all(targets_seg[:, 7 + total_span :] == 1))
        np.testing.assert_array_equal(targets_seg == 0, targets == image_token_id)

    # Non-synthetic dataset_type preserves int32 shaped_batch["images"] while sizing axis 0 to global_batch_size.
    cfg_tfds = _init_mini_mm_config(
        grid=(7, 5),
        dataset_type="tfds",
        per_device_batch_size=0.5,
        max_target_length=64,
    )
    shaped_tfds = maxtext_utils.get_shaped_batch(cfg_tfds)
    self.assertEqual(shaped_tfds["images"].shape, (int(cfg_tfds.global_batch_size_to_load), 1, 35, 588))
    self.assertEqual(shaped_tfds["images"].dtype, jnp.int32)

    # Negative control 1: max_target_length <= 8 + N * span_len raises ValueError.
    for bad_seq_len in (8 + 4 * 11, 8 + 4 * 11 - 1):
      cfg_bad_len = _init_mini_mm_config(
          grid=(7, 5),
          max_num_images_per_example=4,
          max_target_length=bad_seq_len,
      )
      with self.assertRaisesRegex(ValueError, "max_target_length"):
        SyntheticDataIterator(cfg_bad_len, mesh)

    # Negative control 2: packing=True with max_segments_per_seq >= 2 raises ValueError (no fractured span).
    cfg_bad_pack = _init_mini_mm_config(
        grid=(7, 5),
        max_num_images_per_example=1,
        max_target_length=64,
        packing=True,
        max_segments_per_seq=2,
    )
    with self.assertRaisesRegex(ValueError, "max_segments_per_seq"):
      SyntheticDataIterator(cfg_bad_pack, mesh)

    # Negative control 3: out-of-range ds41_image_token_id (<= 0 or >= vocab_size=256) raises ValueError.
    cfg_tok = _init_mini_mm_config(
        grid=(7, 5),
        max_num_images_per_example=1,
        max_target_length=64,
        vocab_size=256,
        ds41_image_token_id=200,
    )
    ok_batch = next(SyntheticDataIterator(cfg_tok, mesh))
    self.assertTrue(bool(np.all(np.asarray(ok_batch["inputs"])[:, 8:19] == 200)))
    for bad_tok_id in (-1, 0, 256, 300):
      cfg_tok.get_keys()["ds41_image_token_id"] = bad_tok_id
      with self.assertRaisesRegex(ValueError, "ds41_image_token_id"):
        SyntheticDataIterator(cfg_tok, mesh)
    cfg_tok.get_keys()["ds41_image_token_id"] = 200

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_subbatch_per_device_0_5_no_indivisible_error_and_single_trace(self) -> None:
    """Verifies per_device_batch_size=0.5 across 4 CPU devices runs without IndivisibleError and traces train_step once."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices (e.g. XLA_FLAGS='--xla_force_host_platform_device_count=4')")
    run_out_dir = os.path.join(self.tmp_dir, "out_subbatch_0_5")
    metrics_path = os.path.join(run_out_dir, "metrics.jsonl")
    os.makedirs(run_out_dir, exist_ok=True)

    cfg = _init_mini_mm_config(
        grid=multimodal.MINI_GRID_PADDED,
        base_output_directory=run_out_dir,
        load_parameters_path=self.ckpt_dir,
        source_checkpoint_layout="safetensors_dynamic",
        scan_layers=True,
        freeze_vision_encoder_params=True,
        log_moe_bias_norms=True,
        dataset_type="synthetic",
        max_num_images_per_example=1,
        max_target_length=32,
        per_device_batch_size=0.5,
        ici_data_parallelism=1,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
        ici_tensor_parallelism=1,
        steps=2,
        learning_rate_schedule_steps=2,
        learning_rate=1e-5,
        warmup_steps_fraction=0.0,
        adam_weight_decay=0.1,
        gradient_clipping_threshold=1.0,
        metrics_file=metrics_path,
    )
    self.assertEqual(int(cfg.global_batch_size_to_load), 4)
    self.assertEqual(int(cfg.micro_batch_size_to_train_on), 2)

    shaped = maxtext_utils.get_shaped_batch(cfg)
    self.assertEqual(shaped["images"].shape, (4, 1, 35, 588))
    self.assertEqual(shaped["images"].dtype, jnp.float32)
    self.assertNotIn("image_masks", shaped)

    # Verify A6 AOT preset carries #5937 schedule flags and run_train_aot_check compiles the exact synthetic signature.
    aot_mod = _load_aot_check_module()
    spec_a6 = aot_mod.get_preset_spec("A6")
    self.assertEqual(spec_a6["overrides"]["learning_rate"], 1e-5)
    self.assertEqual(spec_a6["overrides"]["warmup_steps_fraction"], 0.0)
    self.assertEqual(spec_a6["overrides"]["adam_weight_decay"], 0.1)
    self.assertEqual(spec_a6["overrides"]["gradient_clipping_threshold"], 1.0)
    self.assertEqual(spec_a6["overrides"]["load_balance_loss_weight"], 0.0001)
    self.assertEqual(spec_a6["overrides"]["routed_bias_update_rate"], 0.001)
    self.assertEqual(spec_a6["overrides"]["steps"], 20)
    self.assertEqual(spec_a6["overrides"]["learning_rate_schedule_steps"], 20)

    res_aot = aot_mod.run_train_aot_check(
        cfg,
        with_images=True,
        expect_vision_in_hlo=True,
        max_hbm_gib=85.0,
        use_cpu_mesh=True,
    )
    self.assertTrue(all(res_aot["vision_hlo_markers"].values()))
    self.assertGreater(res_aot["peak_hbm_gib"], 0.0)

    orig_train_step = train.train_step
    trace_count = 0

    def _counting_train_step(*args: Any, **kwargs: Any) -> Any:
      nonlocal trace_count
      trace_count += 1
      return orig_train_step(*args, **kwargs)

    captured_logs: list[str] = []
    orig_log = max_logging.log

    def _capture_log(msg: str, *a: Any, **kw: Any) -> None:
      captured_logs.append(str(msg))
      orig_log(msg, *a, **kw)

    with (
        mock.patch.object(train, "train_step", side_effect=_counting_train_step),
        mock.patch.object(max_logging, "log", side_effect=_capture_log),
    ):
      train.train_loop(cfg, recorder=None)

    # Exactly 1 XLA trace during AOT .lower(...).compile(); zero step-0 or step-1 cache-miss recompilations.
    self.assertEqual(trace_count, 1)
    parsed_log = deepseek_v41_mm_audit.parse_r4_log_text("\n".join(captured_logs))
    self.assertEqual(parsed_log["mm_norms"][0]["vit_norm"], self.ckpt_vit_norm_hex)
    self.assertEqual(parsed_log["mm_norms"][1]["vit_norm"], self.ckpt_vit_norm_hex)
    self.assertEqual(parsed_log["mm_norms"][0]["bias_vl_norm"], self.ckpt_bias_vl_norms_hex)

  def test_two_step_cpu_train_loop_norms_and_loss_unscanned(self) -> None:
    """Verifies 2-step CPU train_loop loss reduction and DS41_MM_NORMS invariants with scan_layers=False."""
    self._run_two_step_cpu_train_loop(scan_layers=False)

  def test_two_step_cpu_train_loop_norms_and_loss_scanned(self) -> None:
    """Verifies 2-step CPU train_loop loss reduction and DS41_MM_NORMS invariants with scan_layers=True."""
    self._run_two_step_cpu_train_loop(scan_layers=True)

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_r4_tt5a_tt6_gate_verifier_and_negative_controls(self) -> None:
    """Verifies verify_r4_tt5a_tt6, child-process CLI execution, and all TT5a/TT6/file negative controls."""
    with tempfile.TemporaryDirectory(prefix="ds41_u6c_gates_") as gate_dir:
      aot_val = 74.3736
      log_lines, metric_records = _build_synthetic_r4_trace(
          self.ckpt_bias_vl_norms_hex,
          num_steps=20,
          peak_hbm_gb=76.25,
      )
      log_path = os.path.join(gate_dir, "r4_train.log")
      metrics_path = os.path.join(gate_dir, "r4_metrics.jsonl")
      a6_json_path = os.path.join(gate_dir, "a6_aot.json")
      w1_json_path = os.path.join(gate_dir, "w1_all_aot.json")
      summary_out = os.path.join(gate_dir, "r4_summary.json")

      def _write_trace(lines: list[str], records: list[dict[str, Any]], tag: str) -> tuple[str, str]:
        lp = os.path.join(gate_dir, f"{tag}.log")
        mp = os.path.join(gate_dir, f"{tag}.jsonl")
        Path(lp).write_text("\n".join(lines), encoding="utf-8")
        Path(mp).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return lp, mp

      Path(log_path).write_text("\n".join(log_lines), encoding="utf-8")
      Path(metrics_path).write_text("\n".join(json.dumps(r) for r in metric_records), encoding="utf-8")
      head_sha = deepseek_v41_mm_audit.resolve_repo_head_commit_sha()
      gib_bytes = 1024**3
      a6_entry_payload = {
          "preset": "A6",
          "git_commit_sha": head_sha,
          "compile_succeeded": True,
          "passed": True,
          "argument_size_in_bytes": int(aot_val * gib_bytes),
          "output_size_in_bytes": int(aot_val * gib_bytes),
          "alias_size_in_bytes": int(aot_val * gib_bytes),
          "non_aliased_io_hbm_gib": aot_val,
          "temp_hbm_gib": 5.0,
          "naive_sum_hbm_gib": 80.0,
          "peak_hbm_gib": 80.0,
          "max_hbm_gib": 85.0,
      }
      Path(a6_json_path).write_text(json.dumps(a6_entry_payload), encoding="utf-8")
      Path(w1_json_path).write_text(
          json.dumps(
              {
                  "preset": "W1_ALL",
                  "git_commit_sha": head_sha,
                  "passed": True,
                  "results": {"A6": a6_entry_payload},
              }
          ),
          encoding="utf-8",
      )

      self.assertAlmostEqual(deepseek_v41_mm_audit.load_a6_aot_hbm_gb(a6_json_path), aot_val)
      self.assertAlmostEqual(deepseek_v41_mm_audit.load_a6_aot_hbm_gb(w1_json_path), aot_val)

      hf_bvl_tensors = deepseek_v41_mm_audit.load_hf_bias_vl_tensors(
          self.ckpt_dir, expected_num_layers=None, expected_shape=None
      )
      bias_vl_npz_path = os.path.join(gate_dir, "r4_bias_vl_steps.npz")
      clean_bvl_payload: dict[str, Any] = {
          "steps": np.asarray([0, 19], dtype=np.int32),
          "layer_order": np.arange(len(hf_bvl_tensors), dtype=np.int32),
      }
      for l_idx, l_arr in hf_bvl_tensors.items():
        clean_bvl_payload[f"step_0_layer_{l_idx}"] = l_arr.copy()
        clean_bvl_payload[f"step_19_layer_{l_idx}"] = l_arr.copy()
      np.savez(bias_vl_npz_path, **clean_bvl_payload)

      res_pass = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          log_path,
          metrics_path,
          _THRESHOLDS_JSON,
          hf_checkpoint_dir=self.ckpt_dir,
          aot_hbm_gb=aot_val,
          expected_steps=20,
          bias_vl_npz=bias_vl_npz_path,
      )
      self.assertTrue(res_pass["passed"], msg=f"Unexpected failures: {res_pass['failures']}")
      self.assertEqual(res_pass["peak_hbm_gb"], 76.25)
      self.assertEqual(res_pass["aot_hbm_gb"], aot_val)
      self.assertAlmostEqual(res_pass["peak_to_aot_ratio"], 76.25 / aot_val)

      # Fresh child-process subprocess.run smoke test on deepseek_v41_mm_audit.py --mode=verify_r4_gates.
      child_env = os.environ.copy()
      child_env["JAX_PLATFORMS"] = "cpu"
      child_env["PYTHONPATH"] = f"{_REPO_ROOT.as_posix()}:{(_REPO_ROOT / 'src').as_posix()}"
      proc_cli = subprocess.run(
          [
              sys.executable,
              _AUDIT_PY,
              "--mode=verify_r4_gates",
              f"--log_file={log_path}",
              f"--metrics_file={metrics_path}",
              f"--thresholds_json={_THRESHOLDS_JSON}",
              f"--hf_checkpoint_dir={self.ckpt_dir}",
              f"--aot_summary_json={w1_json_path}",
              "--expected_steps=20",
              f"--bias_vl_npz={bias_vl_npz_path}",
              f"--summary_json={summary_out}",
          ],
          env=child_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_cli.returncode, 0, msg=f"stderr={proc_cli.stderr} stdout={proc_cli.stdout}")
      self.assertIn("DS41_R4_GATES_RESULT:", proc_cli.stdout)
      summary_payload = json.loads(Path(summary_out).read_text(encoding="utf-8"))
      self.assertTrue(summary_payload["passed"])

      # Real child-process CLI test on --mode=verify_pin against repo golden_manifest_pin.json with NO --repo_root.
      pin_summary_out = os.path.join(gate_dir, "pin_verify_summary.json")
      proc_pin_cli = subprocess.run(
          [
              sys.executable,
              _AUDIT_PY,
              "--mode=verify_pin",
              f"--pin_json_path={_PIN_JSON}",
              f"--summary_json={pin_summary_out}",
          ],
          env=child_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_pin_cli.returncode, 0, msg=f"stderr={proc_pin_cli.stderr} stdout={proc_pin_cli.stdout}")
      self.assertIn("DS41_PIN_VERIFY_RESULT:", proc_pin_cli.stdout)
      pin_summary = json.loads(Path(pin_summary_out).read_text(encoding="utf-8"))
      self.assertTrue(pin_summary["passed"], msg=f"pin_summary={pin_summary}")

      # Negative control on --mode=verify_pin CLI without --repo_root: tampered hf_revision fails with non-zero rc.
      bad_pin_path = os.path.join(gate_dir, "bad_golden_manifest_pin.json")
      bad_pin_obj = json.loads(Path(_PIN_JSON).read_text(encoding="utf-8"))
      bad_pin_obj["hf_revision"] = "0" * 40
      Path(bad_pin_path).write_text(json.dumps(bad_pin_obj), encoding="utf-8")
      proc_bad_pin = subprocess.run(
          [sys.executable, _AUDIT_PY, "--mode=verify_pin", f"--pin_json_path={bad_pin_path}"],
          env=child_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_bad_pin.returncode, 0)
      self.assertIn("hf_revision", proc_bad_pin.stderr)

      # Negative control 1: perturbed bias_vl_norm at step 12 -> fails step-to-step TT5a(7)/TT6.
      bad_bvl_lines = list(log_lines)
      bad_bvl_lines[2 * 12 + 1] = bad_bvl_lines[2 * 12 + 1].replace(
          self.ckpt_bias_vl_norms_hex[0],
          float(np.float32(9.125)).hex(),
          1,
      )
      lp, mp = _write_trace(bad_bvl_lines, metric_records, "nc1_bvl_step12")
      res_bad_bvl = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp,
          mp,
          _THRESHOLDS_JSON,
          hf_checkpoint_dir=self.ckpt_dir,
          aot_hbm_gb=aot_val,
          expected_steps=20,
          bias_vl_npz=bias_vl_npz_path,
      )
      self.assertFalse(res_bad_bvl["passed"])
      self.assertTrue(any("bias_vl_norm" in f for f in res_bad_bvl["failures"]))

      # Negative control 2: 1-ULP perturbed bias_vl_npz at step 19 layer 3 -> fails TT6 bit-exact HF check.
      bad_bvl_payload = {k: v.copy() for k, v in clean_bvl_payload.items()}
      bad_bvl_payload["step_19_layer_3"].view(np.uint32)[0] ^= np.uint32(1)
      bad_bvl_npz = os.path.join(gate_dir, "nc2_bvl_ckpt_mismatch.npz")
      np.savez(bad_bvl_npz, **bad_bvl_payload)
      res_nc2 = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          log_path,
          metrics_path,
          _THRESHOLDS_JSON,
          hf_checkpoint_dir=self.ckpt_dir,
          aot_hbm_gb=aot_val,
          expected_steps=20,
          bias_vl_npz=bad_bvl_npz,
      )
      self.assertFalse(res_nc2["passed"])
      self.assertTrue(any("not bit-exact with HF" in f for f in res_nc2["failures"]))

      # Negative control 3: drifting vit_norm at step 5 -> fails TT5a(7).
      bad_vit_lines = list(log_lines)
      bad_vit_lines[2 * 5 + 1] = bad_vit_lines[2 * 5 + 1].replace(
          f"vit_norm={float(np.float32(12.5)).hex()}",
          f"vit_norm={float(np.float32(12.499)).hex()}",
      )
      lp, mp = _write_trace(bad_vit_lines, metric_records, "nc3_vit")
      res_bad_vit = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=aot_val, expected_steps=20
      )
      self.assertFalse(res_bad_vit["passed"])
      self.assertTrue(any("vit_norm" in f for f in res_bad_vit["failures"]))

      # Negative control 4: unchanged aligner_norm at step 19 -> fails TT5a(7).
      bad_align_lines = list(log_lines)
      align_0_hex = float(np.float32(4.0)).hex()
      align_19_hex = float(np.float32(4.0 + 0.02 * 19)).hex()
      bad_align_lines[2 * 19 + 1] = bad_align_lines[2 * 19 + 1].replace(
          f"aligner_norm={align_19_hex}",
          f"aligner_norm={align_0_hex}",
      )
      lp, mp = _write_trace(bad_align_lines, metric_records, "nc4_align")
      res_bad_align = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=aot_val, expected_steps=20
      )
      self.assertFalse(res_bad_align["passed"])
      self.assertTrue(any("aligner_norm" in f for f in res_bad_align["failures"]))

      # Negative control 5: loss(k+1) > loss(k) + 0.05 (zero +1e-12 slack) -> fails TT5a(4).
      bad_spike_lines = list(log_lines)
      # Step 7 has loss 12.120; 12.170001 exceeds 12.120 + 0.05 by 1e-6.
      bad_spike_lines[2 * 8] = bad_spike_lines[2 * 8].replace("loss: 12.060", "loss: 12.170001")
      lp, mp = _write_trace(bad_spike_lines, metric_records, "nc5_spike")
      res_bad_spike = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=aot_val, expected_steps=20
      )
      self.assertFalse(res_bad_spike["passed"])
      self.assertTrue(any("TT5a(4)" in f for f in res_bad_spike["failures"]))

      # Negative control 6: moe_lb_loss out of [5e-5, 1e-3] -> fails TT5a(5).
      bad_lb_lines = list(log_lines)
      bad_lb_lines[2 * 3] = bad_lb_lines[2 * 3].replace("moe_lb_loss: 0.000150", "moe_lb_loss: 0.002500")
      lp, mp = _write_trace(bad_lb_lines, metric_records, "nc6_lb")
      res_bad_lb = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=aot_val, expected_steps=20
      )
      self.assertFalse(res_bad_lb["passed"])
      self.assertTrue(any("TT5a(5)" in f for f in res_bad_lb["failures"]))

      # Negative control 7: raw_grad_norm > 10 * raw_grad_norm(0) -> fails TT5a(6).
      bad_metrics = [dict(r) for r in metric_records]
      bad_metrics[7]["learning/raw_grad_norm"] = 10.5 * float(bad_metrics[0]["learning/raw_grad_norm"])
      lp, mp = _write_trace(log_lines, bad_metrics, "nc7_rgn")
      res_bad_rgn = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=aot_val, expected_steps=20
      )
      self.assertFalse(res_bad_rgn["passed"])
      self.assertTrue(any("TT5a(6)" in f for f in res_bad_rgn["failures"]))

      # Negative control 8: peak_hbm_gb >= 94.7 GB -> fails TT6 absolute HBM ceiling.
      nc8_lines, _ = _build_synthetic_r4_trace(self.ckpt_bias_vl_norms_hex, num_steps=20, peak_hbm_gb=94.7)
      lp, mp = _write_trace(nc8_lines, metric_records, "nc8_hbm_abs")
      res_nc8 = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=90.0, expected_steps=20
      )
      self.assertFalse(res_nc8["passed"])
      self.assertTrue(any("is not < 94.7 GB" in f for f in res_nc8["failures"]))

      # Negative control 9: peak_hbm_gb > 1.10 * aot_hbm_gb (82.6 > 1.10 * 75.0 = 82.5) -> fails TT6 AOT ratio.
      nc9_lines, _ = _build_synthetic_r4_trace(self.ckpt_bias_vl_norms_hex, num_steps=20, peak_hbm_gb=82.6)
      lp, mp = _write_trace(nc9_lines, metric_records, "nc9_hbm_ratio")
      res_nc9 = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=75.0, expected_steps=20
      )
      self.assertFalse(res_nc9["passed"])
      self.assertTrue(any("1.1 * aot_hbm_gb" in f for f in res_nc9["failures"]))

      # Negative control 10: missing peak_hbm_gb in DS41_MM_NORMS lines -> fails TT6 loudly.
      nc10_lines, _ = _build_synthetic_r4_trace(self.ckpt_bias_vl_norms_hex, num_steps=20, peak_hbm_gb=None)
      lp, mp = _write_trace(nc10_lines, metric_records, "nc10_hbm_missing")
      res_nc10 = deepseek_v41_mm_audit.verify_r4_tt5a_tt6(
          lp, mp, _THRESHOLDS_JSON, hf_checkpoint_dir=self.ckpt_dir, aot_hbm_gb=aot_val, expected_steps=20
      )
      self.assertFalse(res_nc10["passed"])
      self.assertTrue(any("missing peak_hbm_gb" in f for f in res_nc10["failures"]))

      # Negative control 11: missing --log_file or --metrics_file raises FileNotFoundError (no .exists() fallback).
      missing_log = os.path.join(gate_dir, "does_not_exist.log")
      missing_metrics = os.path.join(gate_dir, "does_not_exist.jsonl")
      with self.assertRaises(FileNotFoundError):
        deepseek_v41_mm_audit.parse_r4_log(missing_log)
      with self.assertRaises(FileNotFoundError):
        deepseek_v41_mm_audit.parse_r4_metrics_file(missing_metrics)

      proc_missing_log = subprocess.run(
          [
              sys.executable,
              _AUDIT_PY,
              "--mode=verify_r4_gates",
              f"--log_file={missing_log}",
              f"--metrics_file={metrics_path}",
              f"--thresholds_json={_THRESHOLDS_JSON}",
              f"--hf_checkpoint_dir={self.ckpt_dir}",
              f"--aot_hbm_gb={aot_val}",
          ],
          env=child_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_missing_log.returncode, 0)
      self.assertIn("FileNotFoundError", proc_missing_log.stderr)

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_stage_script_bash_x_path_shim_and_failure_continuation(self) -> None:
    """Verifies 2_test_deepseek41_mm.sh flags, PYTHONPATH child imports, GCS error propagation, and stage continuation."""
    self.assertTrue(os.path.isfile(_STAGE_SCRIPT))
    self.assertTrue(os.access(_STAGE_SCRIPT, os.X_OK))

    with tempfile.TemporaryDirectory(prefix="ds41_u6c_shim_") as tmp_dir:
      shim_bin = os.path.join(tmp_dir, "bin")
      os.makedirs(shim_bin, exist_ok=True)
      calls_log = os.path.join(tmp_dir, "python_calls.jsonl")
      gcloud_log = os.path.join(tmp_dir, "gcloud_calls.jsonl")
      a6_json = os.path.join(tmp_dir, "a6_aot.json")
      head_sha = deepseek_v41_mm_audit.resolve_repo_head_commit_sha()
      Path(a6_json).write_text(
          json.dumps(
              {
                  "preset": "A6",
                  "git_commit_sha": head_sha,
                  "compile_succeeded": True,
                  "passed": True,
                  "argument_size_in_bytes": 38067465024,
                  "output_size_in_bytes": 38063870976,
                  "alias_size_in_bytes": 38063832064,
                  "non_aliased_io_hbm_gib": 38067503936 / float(1024**3),
                  "temp_hbm_gib": 27.148749,
                  "naive_sum_hbm_gib": 62.601875,
                  "peak_hbm_gib": 62.601875,
                  "max_hbm_gib": 85.0,
              }
          ),
          encoding="utf-8",
      )

      shim_py = os.path.join(shim_bin, "python3")
      shim_code = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

with open({calls_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

out_dir = os.environ["OUTPUT_DIR"] if "OUTPUT_DIR" in os.environ else ""
if out_dir:
  Path(out_dir).mkdir(parents=True, exist_ok=True)
  (Path(out_dir) / "shim_marker.json").write_text("{{}}", encoding="utf-8")

fail_match = os.environ["SHIM_FAIL_SUBSTR"] if "SHIM_FAIL_SUBSTR" in os.environ else ""
joined = " ".join(sys.argv[1:])
if fail_match and fail_match in joined:
  sys.exit(7)
sys.exit(0)
"""
      Path(shim_py).write_text(shim_code, encoding="utf-8")
      os.chmod(shim_py, stat.S_IRWXU)

      shim_gcloud = os.path.join(shim_bin, "gcloud")
      gcloud_code = f"""#!{sys.executable}
import json
import os
import sys

with open({gcloud_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

if "SHIM_GCLOUD_FAIL" in os.environ and os.environ["SHIM_GCLOUD_FAIL"] == "1":
  sys.exit(9)
sys.exit(0)
"""
      Path(shim_gcloud).write_text(gcloud_code, encoding="utf-8")
      os.chmod(shim_gcloud, stat.S_IRWXU)

      mm_v1_dir = os.path.join(tmp_dir, "mm_v1")
      os.makedirs(mm_v1_dir, exist_ok=True)
      Path(mm_v1_dir, "L1.npz").write_bytes(b"mock_L1")
      Path(mm_v1_dir, "M1.npz").write_bytes(b"mock_M1")

      base_env = os.environ.copy()
      base_env["PATH"] = f"{shim_bin}:{base_env['PATH']}"
      base_env["HF_CKPT_PATH"] = self.ckpt_dir
      base_env["OUTPUT_DIR"] = os.path.join(tmp_dir, "out")
      base_env["GOLDEN_TEXT_PKL"] = os.path.join(tmp_dir, "golden_full.pkl")
      base_env["GOLDEN_MM_DIR"] = mm_v1_dir
      base_env["GOLDEN_PREFILL_TEXT_PKL"] = os.path.join(tmp_dir, "prefill_v2.pkl")
      base_env["ORBAX_EXPORT_PREFIX"] = os.path.join(tmp_dir, "orbax")
      base_env["A6_AOT_JSON"] = a6_json
      base_env["RUN_TEXT_PREFILL"] = "1"
      base_env["STAGES"] = "R1,R2,R3,R4"

      proc = subprocess.run(
          ["bash", "-x", _STAGE_SCRIPT],
          env=base_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr} stdout={proc.stdout}")
      for stg in ("R1", "R2", "R3", "R4"):
        self.assertIn(f"=== STAGE {stg} START ", proc.stdout)
        self.assertIn(f"=== STAGE {stg} END ", proc.stdout)
        self.assertIn("rc=0 ===", proc.stdout)

      calls = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(calls), 17)
      main_calls = [call for call in calls if not any("--mode=verify_stage_summary_gate" in a for a in call)]
      self.assertEqual(len(main_calls), 8)
      calls = main_calls
      hf_ckpt_arg = f"--hf_checkpoint_dir_for_audit={base_env['HF_CKPT_PATH']}"

      # R1: verify_pin + single forward_pass_logit_checker
      self.assertTrue(any("deepseek_v41_mm_audit.py" in a for a in calls[0]))
      self.assertIn("--mode=verify_pin", calls[0])
      self.assertTrue(any("forward_pass_logit_checker.py" in a for a in calls[1]))
      for req_r1 in (
          "scan_layers=true",
          "override_model_config=true",
          "remat_policy=none",
          "ds41_activation_quant=false",
          "vision_encoder_block=deepseek41",
          "--run_ds41_mm_audit=true",
          hf_ckpt_arg,
      ):
        self.assertIn(req_r1, calls[1])
      self.assertTrue(any(a.startswith("--export_orbax_dir=") and a.endswith("/scanned") for a in calls[1]))

      # R2: verify_pin + single forward_pass_logit_checker
      self.assertTrue(any("deepseek_v41_mm_audit.py" in a for a in calls[2]))
      self.assertIn("--mode=verify_pin", calls[2])
      self.assertTrue(any("forward_pass_logit_checker.py" in a for a in calls[3]))
      for req_r2 in (
          "scan_layers=false",
          "override_model_config=true",
          "remat_policy=none",
          "ds41_activation_quant=false",
          "vision_encoder_block=deepseek41",
          "--run_ds41_mm_audit=true",
          hf_ckpt_arg,
      ):
        self.assertIn(req_r2, calls[3])
      self.assertTrue(any(a.startswith("--export_orbax_dir=") and a.endswith("/unscanned") for a in calls[3]))

      # R3: unconditional verify_pin + masked_prefill_logits_checker
      self.assertTrue(any("deepseek_v41_mm_audit.py" in a for a in calls[4]))
      self.assertIn("--mode=verify_pin", calls[4])
      self.assertIn(f"--summary_json={os.path.join(base_env['OUTPUT_DIR'], 'r3_pin_summary.json')}", calls[4])
      self.assertTrue(any("masked_prefill_logits_checker.py" in a for a in calls[5]))
      for req_r3 in (
          "scan_layers=true",
          "max_target_length=2048",
          "override_model_config=true",
          "remat_policy=none",
          "ds41_activation_quant=false",
          "vision_encoder_block=deepseek41",
          "--max_kl_div=0.08",
          "--min_cos_sim=0.999",
          "--enforce_length_bucket_gates=true",
      ):
        self.assertIn(req_r3, calls[5])
      expected_r3_mm_dir = os.path.join(base_env["OUTPUT_DIR"], "r3_mm_m_only")
      self.assertIn(f"--mm_golden_dir={expected_r3_mm_dir}", calls[5])
      self.assertNotIn(f"--mm_golden_dir={mm_v1_dir}", calls[5])
      self.assertEqual(sorted(os.listdir(expected_r3_mm_dir)), ["M1.npz"])
      self.assertNotIn("L1.npz", os.listdir(expected_r3_mm_dir))
      self.assertEqual(
          Path(expected_r3_mm_dir, "M1.npz").resolve(),
          Path(mm_v1_dir, "M1.npz").resolve(),
      )
      self.assertTrue(any(a.startswith("--golden_logits_path=") for a in calls[5]))

      # R4: run_r4_train (--mode=run_r4_train + --bias_vl_npz + train.py flags) + verify_r4_gates
      self.assertIn("--mode=run_r4_train", calls[6])
      self.assertTrue(any(a.startswith("--bias_vl_npz=") for a in calls[6]))
      self.assertTrue(any("train.py" in a for a in calls[6]))
      for required_flag in (
          "override_model_config=true",
          "dtype_mm=bfloat16",
          "ds41_activation_quant=true",
          "indexer_sparse_training=true",
          "indexer_loss_scaling_factor=0.0",
          "use_multimodal=true",
          "vision_encoder_block=deepseek41",
          "freeze_vision_encoder_params=true",
          "dataset_type=synthetic",
          "max_num_images_per_example=1",
          "max_target_length=512",
          "per_device_batch_size=0.125",
          "log_moe_bias_norms=true",
          "steps=20",
          "learning_rate_schedule_steps=20",
          "learning_rate=1e-5",
          "warmup_steps_fraction=0.0",
          "adam_weight_decay=0.1",
          "gradient_clipping_threshold=1.0",
          "load_balance_loss_weight=0.0001",
          "routed_bias_update_rate=0.001",
      ):
        self.assertIn(required_flag, calls[6])
      self.assertIn("--mode=verify_r4_gates", calls[7])
      self.assertIn(f"--hf_checkpoint_dir={self.ckpt_dir}", calls[7])
      self.assertIn(f"--aot_summary_json={a6_json}", calls[7])
      self.assertTrue(any(a.startswith("--bias_vl_npz=") for a in calls[7]))

      # Real child-process PYTHONPATH test on 2_test_deepseek41_mm.sh (STAGES=R4) with caller PYTHONPATH=src only:
      # --mode=run_r4_train is shimmed to emit a 20-step trace + clean bias_vl_npz; verify_r4_gates execs real python.
      real_bin = os.path.join(tmp_dir, "real_r4_bin")
      os.makedirs(real_bin, exist_ok=True)
      trace_lines, trace_records = _build_synthetic_r4_trace(
          self.ckpt_bias_vl_norms_hex,
          num_steps=20,
          peak_hbm_gb=76.25,
      )
      trace_payload = os.path.join(tmp_dir, "r4_trace_payload.json")
      Path(trace_payload).write_text(
          json.dumps({"lines": trace_lines, "records": trace_records}),
          encoding="utf-8",
      )
      hf_r4_bvl = deepseek_v41_mm_audit.load_hf_bias_vl_tensors(
          self.ckpt_dir, expected_num_layers=None, expected_shape=None
      )
      clean_r4_bvl_npz = os.path.join(tmp_dir, "clean_r4_bvl.npz")
      r4_bvl_payload: dict[str, Any] = {
          "steps": np.asarray([0, 19], dtype=np.int32),
          "layer_order": np.arange(len(hf_r4_bvl), dtype=np.int32),
      }
      for l_idx, l_arr in hf_r4_bvl.items():
        r4_bvl_payload[f"step_0_layer_{l_idx}"] = l_arr.copy()
        r4_bvl_payload[f"step_19_layer_{l_idx}"] = l_arr.copy()
      np.savez(clean_r4_bvl_npz, **r4_bvl_payload)

      real_r4_py = os.path.join(real_bin, "python3")
      real_r4_code = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

if "--mode=run_r4_train" in sys.argv:
  payload = json.loads(Path({trace_payload!r}).read_text(encoding="utf-8"))
  metrics_dest = ""
  bvl_dest = ""
  for arg in sys.argv[1:]:
    if arg.startswith("metrics_file="):
      metrics_dest = arg.split("=", 1)[1]
    elif arg.startswith("--bias_vl_npz="):
      bvl_dest = arg.split("=", 1)[1]
  if not metrics_dest or not bvl_dest:
    raise RuntimeError("Missing metrics_file= or --bias_vl_npz= argument")
  Path(metrics_dest).parent.mkdir(parents=True, exist_ok=True)
  Path(metrics_dest).write_text("\\n".join(json.dumps(r) for r in payload["records"]), encoding="utf-8")
  Path(bvl_dest).parent.mkdir(parents=True, exist_ok=True)
  Path(bvl_dest).write_bytes(Path({clean_r4_bvl_npz!r}).read_bytes())
  for line in payload["lines"]:
    print(line, flush=True)
  sys.exit(0)

os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])
"""
      Path(real_r4_py).write_text(real_r4_code, encoding="utf-8")
      os.chmod(real_r4_py, stat.S_IRWXU)

      real_r4_env = dict(base_env)
      real_r4_env["PATH"] = f"{real_bin}:{os.environ['PATH']}"
      real_r4_env["PYTHONPATH"] = (_REPO_ROOT / "src").as_posix()
      real_r4_env["JAX_PLATFORMS"] = "cpu"
      real_r4_env["STAGES"] = "R4"
      real_r4_env["OUTPUT_DIR"] = os.path.join(tmp_dir, "out_real_r4")
      del real_r4_env["A6_AOT_JSON"]
      real_r4_env["A6_AOT_HBM_GIB"] = "74.3736"
      proc_real_r4 = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=real_r4_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_real_r4.returncode, 0, msg=f"stderr={proc_real_r4.stderr} stdout={proc_real_r4.stdout}")
      self.assertIn("DS41_R4_GATES_RESULT:", proc_real_r4.stdout)
      real_summary = json.loads((Path(real_r4_env["OUTPUT_DIR"]) / "r4_gates_summary.json").read_text(encoding="utf-8"))
      self.assertTrue(real_summary["passed"])

      # R4 fail-fast when neither A6_AOT_JSON nor A6_AOT_HBM_GIB is set, and when both are set.
      no_aot_env = dict(base_env)
      del no_aot_env["A6_AOT_JSON"]
      no_aot_env["STAGES"] = "R4"
      proc_no_aot = subprocess.run(["bash", _STAGE_SCRIPT], env=no_aot_env, capture_output=True, text=True, check=False)
      self.assertNotEqual(proc_no_aot.returncode, 0)
      self.assertIn("A6_AOT_JSON or A6_AOT_HBM_GIB", proc_no_aot.stderr)
      self.assertNotIn("=== STAGE R4 START ", proc_no_aot.stdout)

      both_aot_env = dict(base_env)
      both_aot_env["A6_AOT_HBM_GIB"] = "74.3736"
      both_aot_env["STAGES"] = "R4"
      proc_both_aot = subprocess.run(
          ["bash", _STAGE_SCRIPT], env=both_aot_env, capture_output=True, text=True, check=False
      )
      self.assertNotEqual(proc_both_aot.returncode, 0)
      self.assertIn("not both", proc_both_aot.stderr)

      # GCS_LOG_PREFIX positive & negative controls (failed gcloud and missing gcloud on PATH must fail loudly).
      gcs_ok_env = dict(base_env)
      gcs_ok_env["STAGES"] = "R1"
      gcs_ok_env["GCS_LOG_PREFIX"] = "gs://example-bucket/u6c_logs"
      proc_gcs_ok = subprocess.run(["bash", _STAGE_SCRIPT], env=gcs_ok_env, capture_output=True, text=True, check=False)
      self.assertEqual(proc_gcs_ok.returncode, 0, msg=f"stderr={proc_gcs_ok.stderr}")
      self.assertTrue(os.path.isfile(gcloud_log))

      gcs_fail_env = dict(gcs_ok_env)
      gcs_fail_env["SHIM_GCLOUD_FAIL"] = "1"
      proc_gcs_fail = subprocess.run(
          ["bash", _STAGE_SCRIPT], env=gcs_fail_env, capture_output=True, text=True, check=False
      )
      self.assertNotEqual(proc_gcs_fail.returncode, 0)

      no_gcloud_bin = os.path.join(tmp_dir, "no_gcloud_bin")
      os.makedirs(no_gcloud_bin, exist_ok=True)
      os.symlink(shim_py, os.path.join(no_gcloud_bin, "python3"))
      for tool_name in ("dirname", "mkdir", "date", "tee"):
        resolved = shutil.which(tool_name)
        if resolved:
          os.symlink(resolved, os.path.join(no_gcloud_bin, tool_name))
      gcs_missing_bin_env = dict(gcs_ok_env)
      gcs_missing_bin_env["PATH"] = no_gcloud_bin
      proc_no_gcloud = subprocess.run(
          ["/bin/bash", _STAGE_SCRIPT], env=gcs_missing_bin_env, capture_output=True, text=True, check=False
      )
      self.assertNotEqual(proc_no_gcloud.returncode, 0)
      self.assertIn("gcloud was not found", proc_no_gcloud.stderr)

      # Setting TOKENIZER_PATH threads tokenizer_type=huggingface and tokenizer_path=... into R1..R4.
      Path(calls_log).unlink()
      tok_env = dict(base_env)
      tok_path = os.path.join(tmp_dir, "custom_tokenizer")
      tok_env["TOKENIZER_PATH"] = tok_path
      proc_tok = subprocess.run(
          ["bash", "-x", _STAGE_SCRIPT],
          env=tok_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_tok.returncode, 0, msg=f"stderr={proc_tok.stderr} stdout={proc_tok.stdout}")
      tok_calls = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(tok_calls), 17)
      main_tok_calls = [call for call in tok_calls if not any("--mode=verify_stage_summary_gate" in a for a in call)]
      self.assertEqual(len(main_tok_calls), 8)
      tok_calls = main_tok_calls
      for call_idx in (1, 3, 5, 6):
        self.assertIn("tokenizer_type=huggingface", tok_calls[call_idx])
        self.assertIn(f"tokenizer_path={tok_path}", tok_calls[call_idx])

      # Negative control: _cli_main with nonexistent --manifest_json_path raises FileNotFoundError loudly.
      with self.assertRaises(FileNotFoundError):
        deepseek_v41_mm_audit._cli_main(  # pylint: disable=protected-access
            [
                "--mode=verify_pin",
                f"--pin_json_path={_PIN_JSON}",
                "--manifest_json_path=/nonexistent/missing_manifest.json",
            ]
        )
      with self.assertRaises(FileNotFoundError):
        deepseek_v41_mm_audit._cli_main(  # pylint: disable=protected-access
            [
                "--mode=audit",
                f"--pin_json_path={_PIN_JSON}",
                "--manifest_json_path=/nonexistent/missing_manifest.json",
            ]
        )

      # Subset STAGES="R1,R2,R4" skips R3; stage failure in R2 still runs R4 and exits non-zero.
      Path(calls_log).unlink()
      sub_env = dict(base_env)
      sub_env["STAGES"] = "R1,R2,R4"
      sub_env["SHIM_FAIL_SUBSTR"] = "run_name=ds41_mm_r2_logits"
      proc_fail = subprocess.run(
          ["bash", "-x", _STAGE_SCRIPT],
          env=sub_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_fail.returncode, 0)
      self.assertNotIn("=== STAGE R3 START ", proc_fail.stdout)
      self.assertIn("=== STAGE R1 END ", proc_fail.stdout)
      self.assertIn("=== STAGE R2 END ", proc_fail.stdout)
      self.assertIn("rc=7 ===", proc_fail.stdout)
      self.assertIn("=== STAGE R4 START ", proc_fail.stdout)
      self.assertIn("=== STAGE R4 END ", proc_fail.stdout)

      # Negative control: STAGES=R3 exits non-zero when GOLDEN_MM_DIR contains zero M*.npz files (e.g. L1.npz only).
      mm_l_only_dir = os.path.join(tmp_dir, "mm_v1_l_only")
      os.makedirs(mm_l_only_dir, exist_ok=True)
      Path(mm_l_only_dir, "L1.npz").write_bytes(b"mock_L1")
      r3_no_m_env = dict(base_env)
      r3_no_m_env["STAGES"] = "R3"
      r3_no_m_env["GOLDEN_MM_DIR"] = mm_l_only_dir
      r3_no_m_env["OUTPUT_DIR"] = os.path.join(tmp_dir, "out_r3_no_m")
      proc_r3_no_m = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=r3_no_m_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_r3_no_m.returncode, 0)
      self.assertIn("ERROR: Stage R3 found zero M*.npz", proc_r3_no_m.stderr)

      # Missing required env var fails fast before starting any stage.
      bad_env = dict(base_env)
      del bad_env["GOLDEN_TEXT_PKL"]
      proc_missing = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=bad_env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertNotEqual(proc_missing.returncode, 0)
      self.assertNotIn("=== STAGE R1 START ", proc_missing.stdout)

  @unittest.skipUnless(HAS_TORCH, "torch not installed")
  def test_stage_runner_resume_passed_stages_and_per_stage_gcs_sync(self) -> None:
    """Verifies `RESUME_PASSED_STAGES=1` skips passed stages, re-runs failed stages, and checks R4 Orbax items."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      stub_bin = os.path.join(tmp_dir, "stub_bin")
      os.makedirs(stub_bin, exist_ok=True)
      calls_log = os.path.join(tmp_dir, "resume_calls.jsonl")
      gcloud_log = os.path.join(tmp_dir, "resume_gcloud.jsonl")
      a6_json = os.path.join(tmp_dir, "a6_aot.json")
      Path(a6_json).write_text(json.dumps({"preset": "A6", "peak_hbm_gib": 74.3736}), encoding="utf-8")

      # Stub python3 delegates inline `-` scripts (`stage_already_passed`) to real sys.executable so etils.epath runs,
      # while recording checker/train/audit script invocations and writing requested --summary_json files.
      stub_py = os.path.join(stub_bin, "python3")
      stub_py_code = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

if len(sys.argv) > 1 and sys.argv[1] == "-":
  os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])

with open({calls_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)

for arg in sys.argv[2:]:
  if arg.startswith("--summary_json="):
    summary_dest = Path(arg.split("=", 1)[1])
    summary_dest.parent.mkdir(parents=True, exist_ok=True)
    summary_dest.write_text(json.dumps({{"passed": True, "written_by": "stub"}}), encoding="utf-8")
sys.exit(0)
"""
      Path(stub_py).write_text(stub_py_code, encoding="utf-8")
      os.chmod(stub_py, stat.S_IRWXU)

      stub_gcloud = os.path.join(stub_bin, "gcloud")
      stub_gcloud_code = f"""#!{sys.executable}
import json
import sys

with open({gcloud_log!r}, "a", encoding="utf-8") as f:
  print(json.dumps(sys.argv[1:]), file=f)
sys.exit(0)
"""
      Path(stub_gcloud).write_text(stub_gcloud_code, encoding="utf-8")
      os.chmod(stub_gcloud, stat.S_IRWXU)

      mm_v1_dir = os.path.join(tmp_dir, "mm_v1")
      os.makedirs(mm_v1_dir, exist_ok=True)
      Path(mm_v1_dir, "L1.npz").write_bytes(b"mock_L1")
      Path(mm_v1_dir, "M1.npz").write_bytes(b"mock_M1")

      resume_dir = Path(tmp_dir, "gcs_resume_summaries")
      resume_dir.mkdir(parents=True, exist_ok=True)
      (resume_dir / "r1_logit_summary.json").write_text(
          json.dumps({"passed": True, "origin": "cached_r1"}), encoding="utf-8"
      )
      (resume_dir / "r2_logit_summary.json").write_text(
          json.dumps({"passed": False, "origin": "failed_r2"}), encoding="utf-8"
      )

      out_r12 = Path(tmp_dir, "out_r12")
      env_r12 = os.environ.copy()
      env_r12["PATH"] = f"{stub_bin}:{env_r12['PATH']}"
      env_r12["PYTHONPATH"] = f"{_REPO_ROOT.as_posix()}:{(_REPO_ROOT / 'src').as_posix()}"
      env_r12["HF_CKPT_PATH"] = self.ckpt_dir
      env_r12["OUTPUT_DIR"] = out_r12.as_posix()
      env_r12["GOLDEN_TEXT_PKL"] = os.path.join(tmp_dir, "golden_full.pkl")
      env_r12["GOLDEN_MM_DIR"] = mm_v1_dir
      env_r12["STAGES"] = "R1,R2"
      env_r12["RESUME_PASSED_STAGES"] = "1"
      env_r12["GCS_RESUME_SUMMARY_PREFIX"] = resume_dir.as_posix()
      env_r12["GCS_LOG_PREFIX"] = "gs://example-bucket/w1_resume_logs"

      proc_r12 = subprocess.run(
          ["bash", _STAGE_SCRIPT],
          env=env_r12,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_r12.returncode, 0, msg=f"stderr={proc_r12.stderr} stdout={proc_r12.stdout}")
      self.assertIn(
          f"[2_test_deepseek41_mm] Stage R1 already passed in {resume_dir.as_posix()}; skipping (RESUME_PASSED_STAGES=1).",
          proc_r12.stdout,
      )
      self.assertNotIn("=== STAGE R1 START ", proc_r12.stdout)
      self.assertIn("=== STAGE R2 START ", proc_r12.stdout)
      self.assertIn("=== STAGE R2 END ", proc_r12.stdout)

      copied_r1 = json.loads((out_r12 / "r1_logit_summary.json").read_text(encoding="utf-8"))
      self.assertEqual(copied_r1, {"passed": True, "origin": "cached_r1"})
      written_r2 = json.loads((out_r12 / "r2_logit_summary.json").read_text(encoding="utf-8"))
      self.assertEqual(written_r2, {"passed": True, "written_by": "stub"})

      calls_r12 = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      self.assertEqual(len(calls_r12), 5)
      main_calls_r12 = [call for call in calls_r12 if not any("--mode=verify_stage_summary_gate" in a for a in call)]
      self.assertEqual(len(main_calls_r12), 2, msg=f"Unexpected calls: {calls_r12}")
      calls_r12 = main_calls_r12
      self.assertFalse(any("ds41_mm_r1_logits" in " ".join(c) for c in calls_r12))
      self.assertTrue(any("deepseek_v41_mm_audit.py" in a for a in calls_r12[0]))
      self.assertTrue(any("forward_pass_logit_checker.py" in a for a in calls_r12[1]))
      self.assertIn("run_name=ds41_mm_r2_logits", calls_r12[1])

      # Per-stage sync_outputs_to_gcs ran after R2 (rc==0) AND at script exit.
      gcloud_calls = [json.loads(line) for line in Path(gcloud_log).read_text(encoding="utf-8").splitlines()]
      self.assertGreaterEqual(len(gcloud_calls), 2)

      # R4 with EXPORT_ORBAX=1 + ORBAX_EXPORT_PREFIX requires BOTH scanned/0/items and unscanned/0/items to skip.
      (resume_dir / "r4_gates_summary.json").write_text(
          json.dumps({"passed": True, "origin": "cached_r4"}), encoding="utf-8"
      )
      orbax_dir = Path(tmp_dir, "orbax_exports")
      (orbax_dir / "scanned" / "0" / "items").mkdir(parents=True, exist_ok=True)

      Path(calls_log).unlink()
      env_r4_incomplete_orbax = dict(env_r12)
      env_r4_incomplete_orbax["STAGES"] = "R4"
      env_r4_incomplete_orbax["A6_AOT_JSON"] = a6_json
      env_r4_incomplete_orbax["EXPORT_ORBAX"] = "1"
      env_r4_incomplete_orbax["ORBAX_EXPORT_PREFIX"] = orbax_dir.as_posix()
      env_r4_incomplete_orbax["OUTPUT_DIR"] = os.path.join(tmp_dir, "out_r4_rerun")
      proc_r4_rerun = subprocess.run(
          ["bash", _STAGE_SCRIPT], env=env_r4_incomplete_orbax, capture_output=True, text=True, check=False
      )
      self.assertEqual(proc_r4_rerun.returncode, 0, msg=f"stderr={proc_r4_rerun.stderr} stdout={proc_r4_rerun.stdout}")
      self.assertIn("=== STAGE R4 START ", proc_r4_rerun.stdout)
      calls_r4_rerun = [json.loads(line) for line in Path(calls_log).read_text(encoding="utf-8").splitlines()]
      main_calls_r4 = [call for call in calls_r4_rerun if not any("--mode=verify_stage_summary_gate" in a for a in call)]
      self.assertEqual(len(main_calls_r4), 2)
      calls_r4_rerun = main_calls_r4
      self.assertTrue(any("train.py" in a for a in calls_r4_rerun[0]))

      # Once unscanned/0/items also exists, R4 is skipped cleanly and summary copied to OUTPUT_DIR.
      (orbax_dir / "unscanned" / "0" / "items").mkdir(parents=True, exist_ok=True)
      Path(calls_log).unlink()
      out_r4_skip = Path(tmp_dir, "out_r4_skip")
      env_r4_skip = dict(env_r4_incomplete_orbax)
      env_r4_skip["OUTPUT_DIR"] = out_r4_skip.as_posix()
      proc_r4_skip = subprocess.run(["bash", _STAGE_SCRIPT], env=env_r4_skip, capture_output=True, text=True, check=False)
      self.assertEqual(proc_r4_skip.returncode, 0, msg=f"stderr={proc_r4_skip.stderr} stdout={proc_r4_skip.stdout}")
      self.assertIn("Stage R4 already passed", proc_r4_skip.stdout)
      self.assertNotIn("=== STAGE R4 START ", proc_r4_skip.stdout)
      self.assertFalse(Path(calls_log).exists())
      self.assertEqual(
          json.loads((out_r4_skip / "r4_gates_summary.json").read_text(encoding="utf-8")),
          {"passed": True, "origin": "cached_r4"},
      )


if __name__ == "__main__":
  unittest.main()
