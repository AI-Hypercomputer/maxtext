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

"""Torch-free unit tests for DeepSeek-V4.1 dynamic-resolution processor, Grain/HF pipeline, and mixed-grid training."""

from __future__ import annotations

import base64
import functools
import io
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

import datasets
import flax
from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
import transformers

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import processor_deepseek41 as ds41_proc
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

_BASE_YML = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
_OFFLINE_QWEN3_TOKENIZER = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer")
_QWEN3_IMAGE_PAD_STR = "<|image_pad|>"
_QWEN3_IMAGE_PAD_ID = 151655


def _merge_pure_params(
    graphdef: nnx.GraphDef,
    params_state: nnx.State,
    pure_params: dict[str, Any],
    other_state: nnx.State,
) -> Any:
  state_copy = jax.tree.map(lambda arr: arr, params_state)
  other_copy = jax.tree.map(lambda arr: arr, other_state)
  nnx.replace_by_pure_dict(state_copy, pure_params)
  return nnx.merge(graphdef, state_copy, other_copy)


def _make_rgb_array(height: int, width: int, *, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _tiny_ds41_mm_config(**overrides: Any) -> Any:
  """Builds an offline tiny DeepSeek-V4.1 multimodal config using local `qwen3-tokenizer`."""
  args: dict[str, Any] = {
      "run_name": "ds41_processor_unit_test",
      "base_output_directory": os.path.join(tempfile.gettempdir(), "ds41_proc_test_out"),
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "deepseek4.1-flash-mini",
      "override_model_config": True,
      "use_multimodal": True,
      "vision_encoder_block": "deepseek41",
      "freeze_vision_encoder_params": True,
      "per_device_batch_size": 1.0,
      "max_target_length": 64,
      "max_num_images_per_example": 2,
      "vocab_size": 152064,
      "ds41_image_token_id": _QWEN3_IMAGE_PAD_ID,
      "image_placeholder": _QWEN3_IMAGE_PAD_STR,
      "ds41_engram_compressed_vocab_size": 0,
      "dtype": "float32",
      "weight_dtype": "float32",
      "dtype_mm": "float32",
      "matmul_precision": "highest",
      "base_emb_dim": 64,
      "base_num_query_heads": 4,
      "head_dim": 32,
      "qk_rope_head_dim": 8,
      "q_lora_rank": 16,
      "o_groups": 2,
      "o_lora_rank": 8,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "sparse_matmul": False,
      "capacity_factor": -1,
      "sliding_window_size": 8,
      "base_num_decoder_layers": 2,
      "compress_ratios": [0, 1],
      "ds41_kv_source_layers": [1],
      "ds41_index_source_layers": [1],
      "indexer_n_heads": 4,
      "indexer_head_dim": 16,
      "indexer_topk": 2,
      "ds41_candidate_source_layer": -1,
      "ds41_candidate_topk_blocks": 0,
      "ds41_candidate_block_size": 0,
      "ds41_engram_layers": [1],
      "ds41_engram_vocab_size": 32,
      "ds41_engram_max_ngram_size": 3,
      "ds41_engram_n_heads": 2,
      "ds41_engram_head_dim": 8,
      "num_hidden_layers_for_vit": 2,
      "hidden_size_for_vit": 64,
      "num_attention_heads_for_vit": 2,
      "intermediate_size_for_vit": 128,
      "patch_size_for_vit": 14,
      "rope_theta_for_vit": 10000.0,
      "num_channels_for_vit": 3,
      "ds41_vision_downsample_ratio": 3,
      "ds41_vision_grid": [6, 9],
      "ds41_vision_min_pixels": 0,
      "ds41_vision_max_n_token": 32,
      "tokenizer_type": "huggingface",
      "tokenizer_path": _OFFLINE_QWEN3_TOKENIZER,
  }
  args.update(overrides)
  return pyconfig.initialize([sys.argv[0], _BASE_YML], **args)


def _build_heterogeneous_grain_batch(cfg: Any) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
  """Builds a 2-sample heterogeneous `(5, 8) + (6, 9)` and `(7, 5) + [0, 0]` batch via Grain transforms."""
  img_5x8 = _make_rgb_array(70, 112, seed=101)
  img_6x9 = _make_rgb_array(84, 126, seed=102)
  img_7x5 = _make_rgb_array(98, 70, seed=103)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      _OFFLINE_QWEN3_TOKENIZER, add_bos_token=False, add_eos_token=False
  )
  pad_id = int(tokenizer.pad_token_id)
  raw_examples = [
      {
          "prompt": f"Compare {_QWEN3_IMAGE_PAD_STR} and {_QWEN3_IMAGE_PAD_STR}",
          "completion": "same",
          "images": [img_5x8, img_6x9],
      },
      {"prompt": f"Describe {_QWEN3_IMAGE_PAD_STR}", "completion": "chart", "images": [img_7x5]},
  ]
  processed_rows = []
  for ex in raw_examples:
    row = dict(ex)
    row = input_pipeline_utils.reformat_prompt(row, "prompt", cfg.image_placeholder, cfg.model_name)
    row = input_pipeline_utils.reformat_response(row, "completion", cfg.model_name)
    row = input_pipeline_utils.pre_process_image_sft(row, "images", cfg)
    row["prompt"] = np.asarray(tokenizer.encode(row["prompt"], add_special_tokens=False), dtype=np.int64)
    row["completion"] = np.asarray(tokenizer.encode(row["completion"], add_special_tokens=False), dtype=np.int64)
    row = input_pipeline_utils.prepare_text_for_image_fusion(row, "prompt", cfg)
    processed_rows.append(row)

  grain_ops = [
      input_pipeline_utils.SFTPromptMaskingVision(
          query_column="prompt",
          response_column="completion",
          max_target_length=cfg.max_target_length,
          pad_id=pad_id,
      ),
      input_pipeline_utils.PadOrTrimToMaxLength(
          cfg.max_target_length,
          pad_id,
          config=cfg,
          max_num_images_per_example=cfg.max_num_images_per_example,
      ),
      input_pipeline_utils.ExtractImagesAndMasks(),
      grain.Batch(batch_size=2, drop_remainder=True),
      input_pipeline_utils.FoldImagesIntoBatch(model_name=cfg.model_name, config=cfg),
      input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=1),
  ]
  sampler = grain.IndexSampler(
      num_records=len(processed_rows),
      num_epochs=1,
      shard_options=grain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False),
      shuffle=False,
      seed=0,
  )
  grain_batch = next(
      iter(grain.DataLoader(data_source=processed_rows, operations=grain_ops, sampler=sampler, worker_count=0))
  )

  def _png_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(b, format="PNG")
    return b.getvalue()

  sft_samples = [
      {
          "prompt": f"Compare {_QWEN3_IMAGE_PAD_STR} and {_QWEN3_IMAGE_PAD_STR}",
          "completion": "same",
          "images": [{"bytes": _png_bytes(img_5x8)}, {"bytes": _png_bytes(img_6x9)}],
      },
      {
          "prompt": f"Describe {_QWEN3_IMAGE_PAD_STR}",
          "completion": "chart",
          "images": [{"bytes": _png_bytes(img_7x5)}],
      },
  ]
  return grain_batch, sft_samples


@pytest.mark.cpu_only
class DeepSeek41ProcessorUnitTest(unittest.TestCase):
  """Pure NumPy/PIL/JAX unit tests for `processor_deepseek41.py` and `multimodal/processor.py`."""

  def test_plan_image_grid_buckets_and_input_formats(self) -> None:
    """Verifies dynamic grid planning across small, portrait, landscape, capped, and extreme aspect ratios."""
    # 1. Small image (< 544x544) upscales to at least vision_min_pixels while preserving aspect ratio.
    n_llm_h, n_llm_w, best_h, best_w = ds41_proc.plan_image_grid(200, 100)
    self.assertEqual((best_h % 14, best_w % 14), (0, 0))
    self.assertGreater(best_w, best_h)
    self.assertLessEqual(ds41_proc.num_image_tokens(n_llm_h, n_llm_w), 1024)

    # 2. High-resolution image shrinks so total LLM span tokens <= 1024.
    n_llm_h_big, n_llm_w_big, best_h_big, best_w_big = ds41_proc.plan_image_grid(3200, 2400)
    self.assertLessEqual(ds41_proc.num_image_tokens(n_llm_h_big, n_llm_w_big), 1024)
    self.assertGreater(best_w_big, best_h_big)

    # 3. Extreme tall and extreme wide collapse cleanly within max_n_token.
    tall_h, tall_w, _, _ = ds41_proc.plan_image_grid(32, 8000, max_n_token=64)
    self.assertEqual(tall_w, 1)
    self.assertLessEqual(ds41_proc.num_image_tokens(tall_h, tall_w), 64)
    wide_h, wide_w, _, _ = ds41_proc.plan_image_grid(8000, 32, max_n_token=64)
    self.assertEqual(wide_h, 1)
    self.assertLessEqual(ds41_proc.num_image_tokens(wide_h, wide_w), 64)

    # 4. Identical patch extraction across PIL Image, uint8 ndarray, PNG bytes dict, data URL, and file path.
    arr = _make_rgb_array(70, 112, seed=7)
    pil_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

    p_pil, vh, vw, lh, lw = ds41_proc.extract_image_patches_deepseek41(pil_img, min_pixels=0, max_n_token=64)
    self.assertEqual((vh, vw, lh, lw), (5, 8, 2, 3))
    self.assertEqual(p_pil.shape, (40, 588))
    self.assertEqual(p_pil.dtype, np.float32)

    p_arr, *_ = ds41_proc.extract_image_patches_deepseek41(arr, min_pixels=0, max_n_token=64)
    p_bytes, *_ = ds41_proc.extract_image_patches_deepseek41({"bytes": png_bytes}, min_pixels=0, max_n_token=64)
    p_url, *_ = ds41_proc.extract_image_patches_deepseek41({"url": data_url}, min_pixels=0, max_n_token=64)
    np.testing.assert_array_equal(p_pil, p_arr)
    np.testing.assert_array_equal(p_pil, p_bytes)
    np.testing.assert_array_equal(p_pil, p_url)

    with tempfile.TemporaryDirectory() as tmpdir:
      img_a = Path(tmpdir) / "a.png"
      img_b = Path(tmpdir) / "b.png"
      pil_img.save(img_a)
      Image.fromarray(_make_rgb_array(84, 126, seed=8), mode="RGB").save(img_b)
      cfg = _tiny_ds41_mm_config(image_path=f"{img_a},{img_b}", max_num_images_per_example=3)
      mm_out = mm_processor.preprocess_mm_data(cfg)
      p_max = ds41_proc.max_patches_for_config(cfg)
      self.assertEqual(mm_out.num_images, 2)
      assert mm_out.pixel_values is not None
      assert mm_out.pixel_mask is not None
      self.assertEqual(mm_out.pixel_values.shape, (3, p_max, 588))
      np.testing.assert_array_equal(mm_out.pixel_mask, np.array([1, 1, 0], dtype=np.int32))
      np.testing.assert_array_equal(
          getattr(mm_out, "image_grids"),
          np.array([[5, 8], [6, 9], [0, 0]], dtype=np.int32),
      )
      np.testing.assert_array_equal(mm_out.pixel_values[0, :40], p_pil)
      np.testing.assert_array_equal(mm_out.pixel_values[2], 0.0)
      # Span lengths: (5, 8) -> 2*(3+1)+2 = 10 (+9 offset); (6, 9) -> 2*(3+1)+2 = 10 (+9 offset).
      self.assertEqual(mm_processor.get_image_offsets(cfg, mm_out), 18)

      prompt_tokens = np.array([10, _QWEN3_IMAGE_PAD_ID, 20, _QWEN3_IMAGE_PAD_ID, 30], dtype=np.int64)
      fused = mm_processor.prepare_text_for_image_fusion(prompt_tokens, cfg, processor_output=mm_out)
      self.assertEqual(fused.shape, (3 + 10 + 10,))
      self.assertEqual(int(np.sum(fused == _QWEN3_IMAGE_PAD_ID)), 20)
      with self.assertRaisesRegex(ValueError, "placeholders in prompt tokens"):
        mm_processor.prepare_text_for_image_fusion(np.array([10, _QWEN3_IMAGE_PAD_ID], dtype=np.int64), cfg, mm_out)

    # Verify dummy init shape toggles cleanly between static ds41_vision_grid and dynamic P_max.
    cfg_static = _tiny_ds41_mm_config(dataset_type="synthetic", ds41_dynamic_vision_grid=False)
    cfg_dyn = _tiny_ds41_mm_config(dataset_type="synthetic", ds41_dynamic_vision_grid=True)
    self.assertEqual(
        mm_processor.get_dummy_image_shape_for_init(cfg_static.model_name, batch_size=2, config=cfg_static),
        (2, 2, 54, 588),
    )
    self.assertEqual(
        mm_processor.get_dummy_image_shape_for_init(cfg_dyn.model_name, batch_size=2, config=cfg_dyn),
        (2, 2, ds41_proc.max_patches_for_config(cfg_dyn), 588),
    )

  @pytest.mark.scheduled_only
  def test_grain_and_hf_pipeline_heterogeneous_grids_and_train_step_invariance(self) -> None:
    """Exercises Grain transforms + HF `vision_sft_preprocessing_pipeline` + CPU `loss_fn` VJP without torch."""
    cfg = _tiny_ds41_mm_config(
        dataset_type="hf",
        hf_path="parquet",
        use_sft=True,
        sft_train_on_completion_only=True,
        per_device_batch_size=2.0,
        max_num_images_per_example=2,
        max_target_length=64,
        enable_data_shuffling=False,
    )
    grain_batch, sft_samples = _build_heterogeneous_grain_batch(cfg)
    p_max = ds41_proc.max_patches_for_config(cfg)
    self.assertEqual(grain_batch["images"].shape, (2, 2, p_max, 588))
    np.testing.assert_array_equal(grain_batch["image_masks"], np.array([[1, 1], [1, 0]], dtype=np.int32))
    np.testing.assert_array_equal(
        grain_batch["image_grids"],
        np.array([[[5, 8], [6, 9]], [[7, 5], [0, 0]]], dtype=np.int32),
    )
    # Exact expanded image-span counts in `inputs`: sample 0 -> 10 + 10 = 20; sample 1 -> 11 + 0 = 11.
    self.assertEqual(int(np.sum(grain_batch["inputs"][0] == _QWEN3_IMAGE_PAD_ID)), 20)
    self.assertEqual(int(np.sum(grain_batch["inputs"][1] == _QWEN3_IMAGE_PAD_ID)), 11)

    # 2. End-to-end HF & Grain `vision_sft_preprocessing_pipeline` checks on identical raw samples.
    hf_ds = datasets.Dataset.from_list(sft_samples).to_iterable_dataset()
    mesh1 = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    hf_loader = hf_data_processing.vision_sft_preprocessing_pipeline(
        dataset=hf_ds,
        config=cfg,
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh=mesh1,
        text_columns=["prompt", "completion"],
        image_column="images",
        global_batch_size=2,
    )
    hf_batch = next(iter(hf_loader))
    np.testing.assert_array_equal(hf_batch["image_grids"], grain_batch["image_grids"])
    np.testing.assert_array_equal(hf_batch["image_masks"], grain_batch["image_masks"])
    np.testing.assert_array_equal(hf_batch["inputs"], grain_batch["inputs"])

    cfg_grain = _tiny_ds41_mm_config(
        dataset_type="grain",
        grain_file_type="parquet",
        grain_train_files="/tmp/unused.parquet",
        use_sft=True,
        sft_train_on_completion_only=True,
        per_device_batch_size=2.0,
        max_num_images_per_example=2,
        max_target_length=64,
        enable_data_shuffling=False,
        grain_worker_count=0,
    )
    grain_source = [dict(row) for row in sft_samples * max(1, int(cfg_grain.global_batch_size_to_load) // 2)]
    grain_sft_ds = grain_data_processing.vision_sft_preprocessing_pipeline(
        dataset=grain.MapDataset.source(grain_source),
        config=cfg_grain,
        data_columns=["prompt", "completion"],
        image_column="images",
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )
    grain_sft_batch = next(iter(grain_sft_ds))
    np.testing.assert_array_equal(grain_sft_batch["image_grids"][:2], grain_batch["image_grids"])
    np.testing.assert_array_equal(grain_sft_batch["image_masks"][:2], grain_batch["image_masks"])
    np.testing.assert_array_equal(grain_sft_batch["inputs"][:2], grain_batch["inputs"])

    # 3. CPU `jax.jit` forward + `pre_train.loss_fn` VJP smoke & +100.0 padded-patch invariance + frozen ViT check.
    mesh_1d = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(cfg.mesh_axes)), cfg.mesh_axes)
    jax_batch = {k: jnp.asarray(v) for k, v in grain_batch.items()}
    model = model_creation_utils.from_pretrained(cfg, mesh=mesh_1d, model_mode=MODEL_MODE_TRAIN)
    with jax.set_mesh(mesh_1d), flax.core.spmd.logical_axis_rules(cfg.logical_axis_rules):
      graphdef, params_state, other_state = nnx.split(model, nnx.Param, ...)
      pure_params = nnx.to_pure_dict(params_state)

      @jax.jit
      def run_loss_and_grad(p_dict: dict[str, Any], batch_in: dict[str, jax.Array]) -> tuple[jax.Array, dict[str, Any]]:
        def _loss(p: dict[str, Any]) -> jax.Array:
          st = _merge_pure_params(graphdef, params_state, p, other_state)
          loss_val, _ = pre_train.loss_fn(st, cfg, batch_in, dropout_rng=None, params=None, is_train=True)
          return loss_val

        return jax.value_and_grad(_loss)(p_dict)

      loss_clean, grads_clean = run_loss_and_grad(pure_params, jax_batch)
      self.assertTrue(bool(jnp.isfinite(loss_clean)))
      for leaf in jax.tree.leaves(grads_clean):
        self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

      # Negative control: +100.0 on padded patches (p >= n_vit_h * n_vit_w) and [0, 0] slot -> exact 0.0 diff.
      noisy_imgs = np.array(grain_batch["images"], copy=True)
      noisy_imgs[0, 0, 40:, :] += 100.0
      noisy_imgs[0, 1, 54:, :] += 100.0
      noisy_imgs[1, 0, 35:, :] += 100.0
      noisy_imgs[1, 1, :, :] += 100.0
      noisy_batch = {**jax_batch, "images": jnp.asarray(noisy_imgs)}
      loss_noisy, grads_noisy = run_loss_and_grad(pure_params, noisy_batch)
      self.assertEqual(float(jnp.max(jnp.abs(loss_clean - loss_noisy))), 0.0)
      for g_a, g_b in zip(jax.tree.leaves(grads_clean), jax.tree.leaves(grads_noisy)):
        self.assertEqual(float(jnp.max(jnp.abs(g_a - g_b))), 0.0)

      # Positive control: +100.0 on valid patch 0 of image (0, 0) changes loss and aligner gradients.
      valid_perturbed = np.array(grain_batch["images"], copy=True)
      valid_perturbed[0, 0, 0, :] += 100.0
      loss_pert, _ = run_loss_and_grad(pure_params, {**jax_batch, "images": jnp.asarray(valid_perturbed)})
      self.assertGreater(float(jnp.abs(loss_clean - loss_pert)), 1e-5)

      # Optimizer step check: frozen ViT params remain bit-identical while trainable projector updates.
      tx = optimizers.get_optimizer(cfg, learning_rate_schedule=lambda _: 1e-2, model=model)
      opt_state = tx.init(params_state)
      grads_state = jax.tree.map(lambda arr: arr, params_state)
      nnx.replace_by_pure_dict(grads_state, grads_clean)
      updates, _ = tx.update(grads_state, opt_state, params_state)
      vit_updates = nnx.to_pure_dict(updates)["vision_encoder"]["DeepSeek41VisionEncoder_0"]
      for leaf in jax.tree.leaves(vit_updates):
        self.assertEqual(float(jnp.max(jnp.abs(leaf))), 0.0)

  @pytest.mark.scheduled_only
  def test_heterogeneous_grids_fsdp4_matches_single_device(self) -> None:
    """Verifies 4-device FSDP forward logits match single-device forward on heterogeneous dynamic grids."""
    if jax.device_count() < 4:
      self.skipTest("Requires >= 4 JAX devices for fsdp=4 parity")
    cfg = _tiny_ds41_mm_config(
        dataset_type="hf",
        hf_path="parquet",
        use_sft=True,
        sft_train_on_completion_only=True,
        per_device_batch_size=2.0,
        max_num_images_per_example=2,
        max_target_length=64,
        enable_data_shuffling=False,
    )
    grain_batch, _ = _build_heterogeneous_grain_batch(cfg)
    jax_batch = {k: jnp.asarray(v) for k, v in grain_batch.items()}

    mesh_1d = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(cfg.mesh_axes)), cfg.mesh_axes)
    model_1d = model_creation_utils.from_pretrained(cfg, mesh=mesh_1d, model_mode=MODEL_MODE_TRAIN)
    graphdef, params_state, other_state = nnx.split(model_1d, nnx.Param, ...)
    pure_params = nnx.to_pure_dict(params_state)

    fsdp_shape = [1] * len(cfg.mesh_axes)
    fsdp_shape[list(cfg.mesh_axes).index("fsdp")] = 4
    mesh_4d = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(tuple(fsdp_shape)), cfg.mesh_axes)
    model_4d = model_creation_utils.from_pretrained(cfg, mesh=mesh_4d, model_mode=MODEL_MODE_TRAIN)
    gdef_4d, pstate_4d, ostate_4d = nnx.split(model_4d, nnx.Param, ...)
    p_shd_4d = nnx.to_pure_dict(sharding.nnx_construct_named_sharding(pstate_4d, mesh_4d))
    params_4d = jax.tree.map(jax.device_put, pure_params, p_shd_4d)
    rep_4d = jax.sharding.NamedSharding(mesh_4d, jax.sharding.PartitionSpec())
    batch_4d = {k: jax.device_put(v, rep_4d) for k, v in jax_batch.items()}

    with jax.set_mesh(mesh_4d), flax.core.spmd.logical_axis_rules(cfg.logical_axis_rules):

      @functools.partial(jax.jit)
      def fwd_4d(p_dict: dict[str, Any], batch_in: dict[str, jax.Array]) -> jax.Array:
        m = _merge_pure_params(gdef_4d, pstate_4d, p_dict, ostate_4d)
        return m(
            batch_in["inputs"],
            batch_in["inputs_position"],
            decoder_segment_ids=batch_in["inputs_segmentation"],
            encoder_images=batch_in["images"],
            encoder_image_masks=batch_in["image_masks"],
            encoder_image_grids=batch_in["image_grids"],
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )

      logits_4d = np.asarray(fwd_4d(params_4d, batch_4d))

    with jax.set_mesh(mesh_1d), flax.core.spmd.logical_axis_rules(cfg.logical_axis_rules):

      @functools.partial(jax.jit)
      def fwd_1d(p_dict: dict[str, Any], batch_in: dict[str, jax.Array]) -> jax.Array:
        m = _merge_pure_params(graphdef, params_state, p_dict, other_state)
        return m(
            batch_in["inputs"],
            batch_in["inputs_position"],
            decoder_segment_ids=batch_in["inputs_segmentation"],
            encoder_images=batch_in["images"],
            encoder_image_masks=batch_in["image_masks"],
            encoder_image_grids=batch_in["image_grids"],
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )

      logits_1d = np.asarray(fwd_1d(pure_params, jax_batch))

    np.testing.assert_allclose(logits_4d, logits_1d, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
