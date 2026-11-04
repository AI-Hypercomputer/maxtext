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

"""T13: Full-model DeepSeek-V4.1 multimodal logits parity loaded via `safetensors_dynamic`."""

from __future__ import annotations

import gc
import tempfile
from typing import Any
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.parity import e2e_check
from tests.utils.parity import layer_parity
from tests.utils.parity import negctl

torch: Any
image_processor: Any
multimodal: Any
try:
  import torch  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import image_processor  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  image_processor = None
  multimodal = None
  HAS_TORCH = False

_VALID_FULL_LEN = 36
_DOC1_LEN = 16
_DOC2_LEN = 16


def setUpModule() -> None:
  """Enforces IEEE-754 float32 matmul precision for parity tests."""
  jax.config.update("jax_default_matmul_precision", "highest")


def _measure_multimodal_logits_noise_floor(
    ref_mm: multimodal.MultimodalTransformer,
    batch: multimodal.MiniVLBatch,
) -> layer_parity.Metrics:
  """Measures end-to-end multimodal fp32-vs-fp64 propagation to full logits via `reference_noise_floor`."""
  all_params = {k: v.detach() for k, v in ref_mm.named_parameters()}
  mm_params = {
      k: v for k, v in all_params.items() if k.startswith(("vision.", "aligner.", "image_")) or k.endswith(".bias_vl")
  }
  patch_size = ref_mm.vision.patch_embed.proj.in_features // 3  # type: ignore[union-attr]
  p_side = int(round(patch_size**0.5))

  def _ref_fn(cast_mm_params: dict[str, torch.Tensor], cast_patches_bnpd: torch.Tensor) -> torch.Tensor:
    cast_images: list[list[image_processor.ImageInput]] = []
    for b_idx, sample in enumerate(batch.images):
      row_imgs: list[image_processor.ImageInput] = []
      for img_idx, orig_img in enumerate(sample):
        patches_4d = cast_patches_bnpd[b_idx, img_idx].reshape(-1, 3, p_side, p_side)
        row_imgs.append(
            image_processor.ImageInput(
                start=orig_img.start,
                patches=patches_4d,
                n_vit_h=orig_img.n_vit_h,
                n_vit_w=orig_img.n_vit_w,
                types=orig_img.types,
            )
        )
      cast_images.append(row_imgs)
    merged_params = {**all_params, **cast_mm_params}
    logits = torch.func.functional_call(
        ref_mm,
        merged_params,
        (batch.input_ids,),
        {"images": cast_images, "token_types": batch.token_types, "full_logits": True},
    )
    return logits[:, :_VALID_FULL_LEN, :]

  with multimodal.fp64_reference_context():
    return layer_parity.reference_noise_floor(_ref_fn, mm_params, batch.patches_bnpd)


def _eval_candidate_outputs(
    model_graphdef: Any,
    param_state: nnx.State,
    param_pure: dict[str, Any],
    rest_state: nnx.State,
    mesh: jax.sharding.Mesh,
    *,
    full_tokens: jax.Array,
    full_pos: jax.Array,
    full_seg: jax.Array,
    full_patches: jax.Array,
    packed_tokens: jax.Array,
    packed_pos: jax.Array,
    packed_seg: jax.Array,
    packed_patches: jax.Array,
) -> dict[str, np.ndarray]:
  """Evaluates full valid logits and packed `[seg=1, seg=2, pad=0]` document logits."""
  local_model = mm_testing.apply_with_pure_params(model_graphdef, param_state, param_pure, rest_state)
  with mesh:
    logits_full = local_model(
        decoder_input_tokens=full_tokens,
        decoder_positions=full_pos,
        decoder_segment_ids=full_seg,
        encoder_images=full_patches,
        enable_dropout=False,
    )
    logits_packed = local_model(
        decoder_input_tokens=packed_tokens,
        decoder_positions=packed_pos,
        decoder_segment_ids=packed_seg,
        encoder_images=packed_patches,
        enable_dropout=False,
    )
  return {
      "full_valid": np.asarray(logits_full[:, :_VALID_FULL_LEN, :], dtype=np.float64),
      "packed_doc1": np.asarray(logits_packed[:, :_DOC1_LEN, :], dtype=np.float64),
      "packed_doc2": np.asarray(logits_packed[:, _DOC1_LEN : _DOC1_LEN + _DOC2_LEN, :], dtype=np.float64),
  }


@pytest.mark.cpu_only
class DeepSeek41MultimodalLogitsVsReferenceTest(unittest.TestCase):
  """T13: End-to-end multimodal logits parity via `hf_streaming_load` across layouts, grids, and packing."""

  def _run_grid_case(self, grid: tuple[int, int], *, seed: int) -> None:
    """Runs T13 logits parity, packing, padding isolation, scan agreement, and negative controls on `grid`."""
    args = multimodal.mini_model_args()
    ref_mm = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
    multimodal.init_mini_reference(ref_mm, seed=seed)

    batch_full = multimodal.build_mini_vl_batch(
        args,
        grid=grid,
        batch_size=multimodal.MINI_BATCH_SIZE,
        seq_len=multimodal.MINI_SEQ_LEN,
        span_starts=multimodal.MINI_DEFAULT_SPAN_STARTS,
        seed=seed + 10,
        dtype=torch.float32,
    )
    doc1 = multimodal.build_mini_vl_batch(
        args,
        grid=grid,
        batch_size=multimodal.MINI_BATCH_SIZE,
        seq_len=_DOC1_LEN,
        span_starts=((2,), (1,)),
        seed=seed + 20,
        dtype=torch.float32,
    )
    doc2 = multimodal.build_mini_vl_batch(
        args,
        grid=grid,
        batch_size=multimodal.MINI_BATCH_SIZE,
        seq_len=_DOC2_LEN,
        span_starts=((3,), (2,)),
        seed=seed + 30,
        dtype=torch.float32,
    )

    noise_metrics = _measure_multimodal_logits_noise_floor(ref_mm, batch_full)
    self.assertTrue(noise_metrics.finite)
    self.assertLessEqual(
        noise_metrics.max_abs,
        layer_parity.FP32_ATOL_CEILING * max(1.0, noise_metrics.ref_inf),
        msg=f"Noise floor max_abs exceeds FP32_TOLERANCE on {grid}: {noise_metrics}",
    )
    self.assertLessEqual(
        noise_metrics.frob_rel,
        layer_parity.FP32_FROB_REL_CEILING,
        msg=f"Noise floor frob_rel exceeds FP32_TOLERANCE on {grid}: {noise_metrics}",
    )

    with torch.no_grad():
      ref_full = (
          ref_mm(
              batch_full.input_ids,
              images=batch_full.images,
              token_types=batch_full.token_types,
              full_logits=True,
          )[:, :_VALID_FULL_LEN, :]
          .detach()
          .cpu()
          .to(torch.float64)
          .numpy()
      )
      ref_doc1 = (
          ref_mm(doc1.input_ids, images=doc1.images, token_types=doc1.token_types, full_logits=True)
          .detach()
          .cpu()
          .to(torch.float64)
          .numpy()
      )
      ref_doc2 = (
          ref_mm(doc2.input_ids, images=doc2.images, token_types=doc2.token_types, full_logits=True)
          .detach()
          .cpu()
          .to(torch.float64)
          .numpy()
      )

    golden = {
        "full_valid": ref_full,
        "packed_doc1": ref_doc1,
        "packed_doc2": ref_doc2,
    }
    output_keys = ["full_valid", "packed_doc1", "packed_doc2"]

    with tempfile.TemporaryDirectory(prefix=f"ds41_t13_{grid[0]}x{grid[1]}_") as ckpt_dir:
      mm_testing.write_mini_safetensors_checkpoint(
          ckpt_dir,
          ref_mm,
          param_dtype=torch.float32,
          include_mtp=False,
      )
      # Free PyTorch reference before instantiating MaxText models.
      del ref_mm
      gc.collect()

      full_tokens_np = batch_full.input_ids.numpy().astype(np.int32)
      # Ensure trailing pad [36:40] has non-image tokens so placeholder count matches 2 images/sample.
      self.assertFalse(np.any(full_tokens_np[:, _VALID_FULL_LEN:] == args.image_token_id))
      full_tokens = jnp.asarray(full_tokens_np, dtype=jnp.int32)
      full_pos = jnp.broadcast_to(jnp.arange(multimodal.MINI_SEQ_LEN, dtype=jnp.int32), full_tokens.shape)
      full_seg_np = np.ones_like(full_tokens_np, dtype=np.int32)
      full_seg_np[:, _VALID_FULL_LEN:] = 0
      full_seg = jnp.asarray(full_seg_np, dtype=jnp.int32)
      full_patches = mm_testing.torch_to_jax(batch_full.patches_bnpd, jnp.float32)

      pad_len = multimodal.MINI_SEQ_LEN - _DOC1_LEN - _DOC2_LEN
      pad_tokens_a = np.full((multimodal.MINI_BATCH_SIZE, pad_len), 11, dtype=np.int32)
      pad_tokens_b = np.full((multimodal.MINI_BATCH_SIZE, pad_len), 29, dtype=np.int32)
      packed_tokens = jnp.asarray(
          np.concatenate([doc1.input_ids.numpy(), doc2.input_ids.numpy(), pad_tokens_a], axis=1),
          dtype=jnp.int32,
      )
      packed_tokens_alt = jnp.asarray(
          np.concatenate([doc1.input_ids.numpy(), doc2.input_ids.numpy(), pad_tokens_b], axis=1),
          dtype=jnp.int32,
      )
      pos_1d = np.concatenate(
          [np.arange(_DOC1_LEN), np.arange(_DOC2_LEN), np.arange(pad_len)],
          axis=0,
      ).astype(np.int32)
      seg_1d = np.concatenate(
          [
              np.ones(_DOC1_LEN, dtype=np.int32),
              np.full(_DOC2_LEN, 2, dtype=np.int32),
              np.zeros(pad_len, dtype=np.int32),
          ],
          axis=0,
      )
      packed_pos = jnp.broadcast_to(jnp.asarray(pos_1d, dtype=jnp.int32), packed_tokens.shape)
      packed_seg = jnp.broadcast_to(jnp.asarray(seg_1d, dtype=jnp.int32), packed_tokens.shape)
      packed_patches_t = torch.cat([doc1.patches_bnpd, doc2.patches_bnpd], dim=1)
      packed_patches = mm_testing.torch_to_jax(packed_patches_t, jnp.float32)

      logits_tol = e2e_check.logits_tol_for_row("T13")
      scan_agree_tol = layer_parity.Tolerance(atol=1e-6, rtol=1e-6, frob_rel=1e-6)
      outputs_by_scan: dict[bool, dict[str, np.ndarray]] = {}

      for scan_layers in (False, True):
        with self.subTest(grid=grid, scan_layers=scan_layers):
          model, _, mesh = mm_testing.load_mini_maxtext_from_safetensors(
              ckpt_dir,
              grid=grid,
              scan_layers=scan_layers,
              dtype="float32",
              weight_dtype="float32",
              dtype_mm="float32",
              matmul_precision="highest",
          )
          graphdef, param_state, param_pure, rest_state = mm_testing.extract_pure_params(model)

          primary = _eval_candidate_outputs(
              graphdef,
              param_state,
              param_pure,
              rest_state,
              mesh,
              full_tokens=full_tokens,
              full_pos=full_pos,
              full_seg=full_seg,
              full_patches=full_patches,
              packed_tokens=packed_tokens,
              packed_pos=packed_pos,
              packed_seg=packed_seg,
              packed_patches=packed_patches,
          )
          outputs_by_scan[scan_layers] = primary

          # 1. Elementwise + Frobenius parity under FP32_TOLERANCE (1e-5).
          ok_fp32, _, fp32_failures = e2e_check.check_outputs(golden, primary, output_keys, layer_parity.FP32_TOLERANCE)
          self.assertTrue(
              ok_fp32,
              msg=f"T13 FP32_TOLERANCE failed (grid={grid}, scan_layers={scan_layers}): {fp32_failures}",
          )

          # 2. Distribution parity under approved T13 LogitsTol (KL <= 1e-6, top-1 == 1.0).
          ok_kl, _, kl_failures = e2e_check.check_outputs(golden, primary, output_keys, logits_tol)
          self.assertTrue(
              ok_kl,
              msg=f"T13 LogitsTol failed (grid={grid}, scan_layers={scan_layers}): {kl_failures}",
          )

          # 3. Padding token isolation on packed [1, 2, 0] sequence.
          alt_pad_out = _eval_candidate_outputs(
              graphdef,
              param_state,
              param_pure,
              rest_state,
              mesh,
              full_tokens=full_tokens,
              full_pos=full_pos,
              full_seg=full_seg,
              full_patches=full_patches,
              packed_tokens=packed_tokens_alt,
              packed_pos=packed_pos,
              packed_seg=packed_seg,
              packed_patches=packed_patches,
          )
          for doc_key in ("packed_doc1", "packed_doc2"):
            ok_pad, pad_m = layer_parity.compare(alt_pad_out[doc_key], primary[doc_key], scan_agree_tol)
            self.assertTrue(ok_pad, msg=f"Padding leaked into {doc_key}: {pad_m}")

          # 4. Negative controls on both projector w2/kernel and layer-0 attention wq_a/kernel.
          wqa_path = (
              "decoder/segments_0/layers_0/self_attention/wq_a/kernel"
              if scan_layers
              else "decoder/layers_0/self_attention/wq_a/kernel"
          )
          for target_path, nc_seed in (
              ("vision_encoder/DeepSeek41VisionProjector_0/w2/kernel", 401),
              (wqa_path, 402),
          ):
            pert_params = negctl.perturb_one_leaf(
                param_pure,
                target_path=target_path,
                rel_scale=1e-2,
                seed=nc_seed,
            )
            perturbed = _eval_candidate_outputs(
                graphdef,
                param_state,
                pert_params,
                rest_state,
                mesh,
                full_tokens=full_tokens,
                full_pos=full_pos,
                full_seg=full_seg,
                full_patches=full_patches,
                packed_tokens=packed_tokens,
                packed_pos=packed_pos,
                packed_seg=packed_seg,
                packed_patches=packed_patches,
            )
            status_kl, reasons_kl = e2e_check.negative_run_status(golden, primary, perturbed, output_keys, logits_tol)
            self.assertEqual(
                status_kl,
                "FIRED",
                msg=f"T13 LogitsTol negctl {status_kl} on {target_path} (scan={scan_layers}): {reasons_kl}",
            )
            status_fp32, reasons_fp32 = e2e_check.negative_run_status(
                golden, primary, perturbed, output_keys, layer_parity.FP32_TOLERANCE
            )
            self.assertEqual(
                status_fp32,
                "FIRED",
                msg=f"T13 FP32 negctl {status_fp32} on {target_path} (scan={scan_layers}): {reasons_fp32}",
            )

      # 5. Scanned vs unscanned logit agreement (<= 1e-6).
      ok_scan, _, scan_failures = e2e_check.check_outputs(
          outputs_by_scan[False],
          outputs_by_scan[True],
          output_keys,
          scan_agree_tol,
      )
      self.assertTrue(ok_scan, msg=f"Scanned vs unscanned mismatch on {grid}: {scan_failures}")

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_t13_padded_grid_7x5_unscanned_and_scanned(self) -> None:
    self._run_grid_case(multimodal.MINI_GRID_PADDED, seed=51)

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_t13_exact_grid_6x9_unscanned_and_scanned(self) -> None:
    self._run_grid_case(multimodal.MINI_GRID_EXACT, seed=61)

  def test_jax_multimodal_forward_and_packed_segment_isolation_without_torch(self) -> None:
    with tempfile.TemporaryDirectory(prefix="ds41_t13_notorch_") as ckpt_dir:
      mm_testing.write_numpy_mini_mm_checkpoint(ckpt_dir, seed=41, use_multimodal=True)
      rng = np.random.default_rng(42)
      full_tokens_np = rng.integers(3, 500, size=(2, mm_testing.MINI_SEQ_LEN), dtype=np.int32)
      for b, starts in enumerate(mm_testing.MINI_DEFAULT_SPAN_STARTS):
        for s in starts:
          full_tokens_np[b, s : s + 11] = mm_testing.MINI_IMAGE_TOKEN_ID
      full_tokens = jnp.asarray(full_tokens_np, dtype=jnp.int32)
      full_pos = jnp.broadcast_to(jnp.arange(mm_testing.MINI_SEQ_LEN, dtype=jnp.int32), (2, mm_testing.MINI_SEQ_LEN))
      full_seg = jnp.asarray(
          (np.arange(mm_testing.MINI_SEQ_LEN)[None, :] < _VALID_FULL_LEN).astype(np.int32)
          * np.ones((2, 1), dtype=np.int32)
      )
      full_patches = jnp.asarray(rng.standard_normal((2, 2, 35, 588), dtype=np.float32))

      packed_tokens_np = rng.integers(3, 500, size=(2, mm_testing.MINI_SEQ_LEN), dtype=np.int32)
      packed_tokens_np[:, 2:13] = mm_testing.MINI_IMAGE_TOKEN_ID
      packed_tokens_np[:, _DOC1_LEN + 2 : _DOC1_LEN + 13] = mm_testing.MINI_IMAGE_TOKEN_ID
      packed_tokens = jnp.asarray(packed_tokens_np, dtype=jnp.int32)
      pos_row = np.zeros((mm_testing.MINI_SEQ_LEN,), dtype=np.int32)
      pos_row[:_DOC1_LEN] = np.arange(_DOC1_LEN, dtype=np.int32)
      pos_row[_DOC1_LEN : _DOC1_LEN + _DOC2_LEN] = np.arange(_DOC2_LEN, dtype=np.int32)
      packed_pos = jnp.broadcast_to(jnp.asarray(pos_row, dtype=jnp.int32), (2, mm_testing.MINI_SEQ_LEN))
      seg_row = np.zeros((mm_testing.MINI_SEQ_LEN,), dtype=np.int32)
      seg_row[:_DOC1_LEN] = 1
      seg_row[_DOC1_LEN : _DOC1_LEN + _DOC2_LEN] = 2
      packed_seg = jnp.broadcast_to(jnp.asarray(seg_row, dtype=jnp.int32), (2, mm_testing.MINI_SEQ_LEN))
      packed_patches = full_patches

      outputs_by_scan: dict[bool, dict[str, np.ndarray]] = {}
      output_keys = ["full_valid", "packed_doc1", "packed_doc2"]
      for scan_layers in (False, True):
        model, _, mesh = mm_testing.load_mini_maxtext_from_safetensors(
            ckpt_dir,
            scan_layers=scan_layers,
            vocab_size=mm_testing.MINI_VOCAB_SIZE,
            ds41_image_token_id=mm_testing.MINI_IMAGE_TOKEN_ID,
            ds41_engram_compressed_vocab_size=0,
            tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        )
        gdef, pstate, ppure, rstate = mm_testing.extract_pure_params(model)
        primary = _eval_candidate_outputs(
            gdef,
            pstate,
            ppure,
            rstate,
            mesh,
            full_tokens=full_tokens,
            full_pos=full_pos,
            full_seg=full_seg,
            full_patches=full_patches,
            packed_tokens=packed_tokens,
            packed_pos=packed_pos,
            packed_seg=packed_seg,
            packed_patches=packed_patches,
        )
        outputs_by_scan[scan_layers] = primary
        if not scan_layers:
          pert_pure = negctl.perturb_one_leaf(
              ppure,
              target_path="vision_encoder/DeepSeek41VisionProjector_0/w1/kernel",
              rel_scale=1e-2,
              seed=91,
          )
          perturbed = _eval_candidate_outputs(
              gdef,
              pstate,
              pert_pure,
              rstate,
              mesh,
              full_tokens=full_tokens,
              full_pos=full_pos,
              full_seg=full_seg,
              full_patches=full_patches,
              packed_tokens=packed_tokens,
              packed_pos=packed_pos,
              packed_seg=packed_seg,
              packed_patches=packed_patches,
          )
          status, reasons = e2e_check.negative_run_status(
              primary,
              primary,
              perturbed,
              output_keys,
              layer_parity.FP32_TOLERANCE,
          )
          self.assertEqual(status, "FIRED", msg=f"Torch-free negctl {status}: {reasons}")

      ok_scan, _, scan_failures = e2e_check.check_outputs(
          outputs_by_scan[False],
          outputs_by_scan[True],
          output_keys,
          layer_parity.FP32_TOLERANCE,
      )
      self.assertTrue(ok_scan, msg=f"Scanned vs unscanned mismatch: {scan_failures}")


if __name__ == "__main__":
  unittest.main()
