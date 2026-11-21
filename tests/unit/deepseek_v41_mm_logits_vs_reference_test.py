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

"""PyTorch-reference full-model DeepSeek-V4.1 multimodal logits parity tests."""

from __future__ import annotations

import copy
import gc
import tempfile
from typing import Any
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils import model_creation_utils
from tests.unit import deepseek_v41_mm_jax_test as mm_jax_test
from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.deepseek_v41.parity import e2e_check
from tests.utils.deepseek_v41.parity import layer_parity
from tests.utils.deepseek_v41.parity import negctl

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


_ORIG_MATMUL_PRECISION = None


def setUpModule() -> None:
  """Enforces IEEE-754 float32 matmul precision for parity tests."""
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")


def tearDownModule() -> None:
  jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


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


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41MultimodalLogitsVsReferenceTest(unittest.TestCase):
  """PyTorch-reference end-to-end multimodal logits parity across layouts, grids, and packing."""

  def _run_grid_case(
      self,
      grid: tuple[int, int],
      *,
      seed: int,
      check_fp32_on_single_device_only: bool = False,
  ) -> None:
    """Runs end-to-end logits parity, packing, padding isolation, scan agreement, and negative controls on `grid`."""
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
      ref_vision_spans64: np.ndarray | None = None
      if check_fp32_on_single_device_only:
        assert ref_mm.vision is not None and ref_mm.aligner is not None
        with multimodal.fp64_reference_context():
          vit64 = copy.deepcopy(ref_mm.vision).to(torch.float64)
          aligner64 = copy.deepcopy(ref_mm.aligner).to(torch.float64)
          start64 = ref_mm.image_start.detach().clone().to(torch.float64)
          end64 = ref_mm.image_end.detach().clone().to(torch.float64)
          nl64 = ref_mm.image_newline.detach().clone().to(torch.float64)
          patches64 = batch_full.patches_bnpd.detach().to(torch.float64)
          r = aligner64.downsample_ratio
          n_llm_h = (grid[0] + (-grid[0]) % r) // r
          n_llm_w = (grid[1] + (-grid[1]) % r) // r
          types64 = image_processor.image_token_types(n_llm_h, n_llm_w)
          span_rows = []
          with torch.no_grad():
            for b_i in range(patches64.shape[0]):
              img_spans = []
              for n_i in range(patches64.shape[1]):
                emb = aligner64(vit64(patches64[b_i, n_i], grid[0], grid[1]), grid[0], grid[1])
                span = torch.empty((types64.numel(), args.dim), dtype=torch.float64)
                span[types64 == image_processor.IMAGE_START] = start64
                span[types64 == image_processor.IMAGE_END] = end64
                span[types64 == image_processor.IMAGE_NEW_LINE] = nl64
                span[types64 == image_processor.IMAGE] = emb
                img_spans.append(span)
              span_rows.append(torch.stack(img_spans, dim=0))
          ref_vision_spans64 = torch.stack(span_rows, dim=0).cpu().numpy()
          del vit64, aligner64
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

          primary = mm_jax_test.eval_multimodal_packed_and_full_logits(
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

          # 1. Elementwise + Frobenius parity under FP32_TOLERANCE (1e-5) on fsdp=4 (unless delegated to 1-device).
          if not check_fp32_on_single_device_only:
            ok_fp32, _, fp32_failures = e2e_check.check_outputs(golden, primary, output_keys, layer_parity.FP32_TOLERANCE)
            self.assertTrue(
                ok_fp32,
                msg=f"End-to-end FP32_TOLERANCE failed (grid={grid}, scan_layers={scan_layers}): {fp32_failures}",
            )

          # 2. Distribution parity under approved LogitsTol (KL <= 1e-6, top-1 == 1.0).
          ok_kl, _, kl_failures = e2e_check.check_outputs(golden, primary, output_keys, logits_tol)
          self.assertTrue(
              ok_kl,
              msg=f"End-to-end LogitsTol failed (grid={grid}, scan_layers={scan_layers}): {kl_failures}",
          )

          # 3. Padding token isolation on packed [1, 2, 0] sequence.
          alt_pad_out = mm_jax_test.eval_multimodal_packed_and_full_logits(
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
            perturbed = mm_jax_test.eval_multimodal_packed_and_full_logits(
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
                msg=f"End-to-end LogitsTol negctl {status_kl} on {target_path} (scan={scan_layers}): {reasons_kl}",
            )
            if not check_fp32_on_single_device_only:
              status_fp32, reasons_fp32 = e2e_check.negative_run_status(
                  golden, primary, perturbed, output_keys, layer_parity.FP32_TOLERANCE
              )
              self.assertEqual(
                  status_fp32,
                  "FIRED",
                  msg=f"End-to-end FP32 negctl {status_fp32} on {target_path} (scan={scan_layers}): {reasons_fp32}",
              )

          if check_fp32_on_single_device_only:
            # 4b. Sharded fsdp=4 float64 vision parity (<= 1e-12) when scan_layers=False.
            if not scan_layers:
              assert ref_vision_spans64 is not None
              cfg64 = mm_testing.mini_vision_maxtext_config(
                  grid=grid,
                  load_parameters_path=ckpt_dir,
                  source_checkpoint_layout="safetensors_dynamic",
                  dtype="float32",
                  weight_dtype="float32",
                  dtype_mm="float32",
                  matmul_precision="highest",
              )
              with mm_testing.fp64_maxtext_context(cfg64):
                with mesh:
                  model64 = model_creation_utils.from_pretrained(cfg64, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
                vgdef, vpstate, vpure, vostate = mm_testing.extract_pure_params(model64.vision_encoder)
                with mesh:
                  cand_spans64, _ = mm_testing.apply_with_pure_params(vgdef, vpstate, vpure, vostate)(
                      mm_testing.torch_to_jax(batch_full.patches_bnpd, jnp.float64), deterministic=True
                  )
              tol_1e12 = layer_parity.Tolerance(atol=1e-12, rtol=1e-12, frob_rel=1e-12)
              ok_v64, m_v64 = layer_parity.compare(
                  np.asarray(cand_spans64, dtype=np.float64), ref_vision_spans64, tol_1e12
              )
              self.assertTrue(ok_v64, msg=f"Sharded fsdp=4 float64 vision parity failed on {grid}: {m_v64}")

            # 4c. Elementwise FP32_TOLERANCE (1e-5) + FP32 negctl on single_device_mesh() for both scan modes.
            single_mesh = mm_testing.single_device_mesh()
            m1, _, _ = mm_testing.load_mini_maxtext_from_safetensors(
                ckpt_dir,
                grid=grid,
                scan_layers=scan_layers,
                mesh=single_mesh,
                ici_fsdp_parallelism=1,
                dtype="float32",
                weight_dtype="float32",
                dtype_mm="float32",
                matmul_precision="highest",
            )
            g1, p1, pur1, r1 = mm_testing.extract_pure_params(m1)
            prim1 = mm_jax_test.eval_multimodal_packed_and_full_logits(
                g1,
                p1,
                pur1,
                r1,
                single_mesh,
                full_tokens=full_tokens,
                full_pos=full_pos,
                full_seg=full_seg,
                full_patches=full_patches,
                packed_tokens=packed_tokens,
                packed_pos=packed_pos,
                packed_seg=packed_seg,
                packed_patches=packed_patches,
            )
            ok_1dev, _, f1dev = e2e_check.check_outputs(golden, prim1, output_keys, layer_parity.FP32_TOLERANCE)
            self.assertTrue(
                ok_1dev,
                msg=f"Single-device FP32_TOLERANCE failed (grid={grid}, scan_layers={scan_layers}): {f1dev}",
            )
            pert1 = negctl.perturb_one_leaf(
                pur1,
                target_path="vision_encoder/DeepSeek41VisionProjector_0/w2/kernel",
                rel_scale=1e-2,
                seed=401,
            )
            pert1_out = mm_jax_test.eval_multimodal_packed_and_full_logits(
                g1,
                p1,
                pert1,
                r1,
                single_mesh,
                full_tokens=full_tokens,
                full_pos=full_pos,
                full_seg=full_seg,
                full_patches=full_patches,
                packed_tokens=packed_tokens,
                packed_pos=packed_pos,
                packed_seg=packed_seg,
                packed_patches=packed_patches,
            )
            st1, rsn1 = e2e_check.negative_run_status(golden, prim1, pert1_out, output_keys, layer_parity.FP32_TOLERANCE)
            self.assertEqual(st1, "FIRED", msg=f"Single-device FP32 negctl {st1} (scan={scan_layers}): {rsn1}")

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
    self._run_grid_case(multimodal.MINI_GRID_EXACT, seed=61, check_fp32_on_single_device_only=True)


if __name__ == "__main__":
  unittest.main()
