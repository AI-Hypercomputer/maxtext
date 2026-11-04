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

"""Decoder integration parity tests (T6-T10) for DeepSeek-V4.1 multimodal MaxText."""

from __future__ import annotations

import copy
import functools
import os
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import transformers

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.common.common_types import MultimodalInput
from maxtext.configs import pyconfig
from maxtext.layers import engram_deepseek41
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.parity import layer_parity

torch: Any
functional_call: Any
F: Any
image_processor: Any
ref_model: Any
multimodal: Any
try:
  import torch  # pylint: disable=g-import-not-at-top
  from torch.func import functional_call  # pylint: disable=g-import-not-at-top
  import torch.nn.functional as F  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import image_processor  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import model as ref_model  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  functional_call = None
  F = None
  image_processor = None
  ref_model = None
  multimodal = None
  HAS_TORCH = False

_NEGCTL_REL_SCALE = 1e-2
_TOKENIZER_CACHE: Any = None


def setUpModule() -> None:
  jax.config.update("jax_default_matmul_precision", "highest")


def _get_tokenizer() -> Any:
  global _TOKENIZER_CACHE  # pylint: disable=global-statement
  if _TOKENIZER_CACHE is None:
    _TOKENIZER_CACHE = transformers.AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)
  return _TOKENIZER_CACHE


def _build_reference_pair(
    grid: tuple[int, int] = mm_testing.MINI_GRID_PADDED,
    *,
    seed: int = 1,
    dtype: Any = None,
    span_starts: tuple[tuple[int, ...], ...] = mm_testing.MINI_DEFAULT_SPAN_STARTS,
) -> tuple[multimodal.MultimodalTransformer, multimodal.MiniVLBatch]:
  """Builds a seeded mini-twin `MultimodalTransformer` and synthetic `MiniVLBatch`."""
  if dtype is None:
    dtype = torch.float32
  args = multimodal.mini_model_args()
  ref_mm = multimodal.build_multimodal_model(args, tokenizer=_get_tokenizer())
  multimodal.init_mini_reference(ref_mm, seed=seed)
  batch = multimodal.build_mini_vl_batch(
      args,
      grid=grid,
      span_starts=span_starts,
      seed=seed + 1000,
      dtype=dtype,
  )
  return ref_mm, batch


def _reference_span_embeddings(
    ref_mm: multimodal.MultimodalTransformer,
    batch: multimodal.MiniVLBatch,
) -> torch.Tensor:
  """Materializes `[B, N, K, D]` reference span embeddings matching `VisionEncoder` output."""
  emb_dim = int(ref_mm.image_start.shape[0])
  rows = []
  for sample in batch.images:
    sample_spans = []
    for img in sample:
      types_t = img.types
      span = torch.empty((int(types_t.numel()), emb_dim), dtype=img.patches.dtype)
      span[types_t == image_processor.IMAGE_START] = ref_mm.image_start.to(span.dtype)
      span[types_t == image_processor.IMAGE_END] = ref_mm.image_end.to(span.dtype)
      span[types_t == image_processor.IMAGE_NEW_LINE] = ref_mm.image_newline.to(span.dtype)
      span[types_t == image_processor.IMAGE] = ref_mm.encode_image(img.patches, img.n_vit_h, img.n_vit_w).to(span.dtype)
      sample_spans.append(span)
    rows.append(torch.stack(sample_spans, dim=0))
  return torch.stack(rows, dim=0)


@pytest.mark.cpu_only
class DeepSeek41EmbeddingMergeParityTest(unittest.TestCase):
  """T6: Config reachability, `get_bidirectional_mask_vision`, and `NNXDecoder._apply_embedding` merge."""

  def test_config_reachability_and_validation(self) -> None:
    out_dir = os.path.join(tempfile.gettempdir(), "maxtext_mm_test_out")
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg = pyconfig.initialize(
          [sys.argv[0], "src/maxtext/configs/base.yml"],
          run_name="ds41_cfg_reachability",
          base_output_directory=out_dir,
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          model_name=model_name,
          use_multimodal=True,
          tokenizer_path=mm_testing.TOKENIZER_PATH,
      )
      self.assertEqual(cfg.ds41_image_token_id, 129264)

    custom_cfg = mm_testing.mini_vision_maxtext_config(ds41_image_token_id=129265)
    sample_tokens = jnp.array([[10, 129265, 129264, 129265]], dtype=jnp.int32)
    mask = mm_processor.get_bidirectional_mask_vision(custom_cfg, sample_tokens)
    np.testing.assert_array_equal(np.asarray(mask), np.array([[False, True, False, True]]))

    for bad_token_id in (0, -5, custom_cfg.vocab_size, custom_cfg.vocab_size + 7):
      with self.assertRaisesRegex(ValueError, "0 < ds41_image_token_id < vocab_size"):
        mm_testing.mini_vision_maxtext_config(ds41_image_token_id=bad_token_id)

    with self.assertRaisesRegex(ValueError, "video"):
      mm_processor.get_bidirectional_mask_vision(custom_cfg, sample_tokens, is_video=True)

    valid_unmm_cfg = mm_testing.mini_vision_maxtext_config(use_multimodal=False, ds41_image_token_id=129265)
    unmm_mask = mm_processor.get_bidirectional_mask_vision(valid_unmm_cfg, sample_tokens)
    np.testing.assert_array_equal(np.asarray(unmm_mask), np.array([[False, True, False, True]]))
    for bad_rt_id in (0, custom_cfg.vocab_size):
      bad_rt_cfg = mm_testing.mini_vision_maxtext_config(use_multimodal=False, ds41_image_token_id=bad_rt_id)
      with self.assertRaisesRegex(ValueError, "0 < ds41_image_token_id < vocab_size"):
        mm_processor.get_bidirectional_mask_vision(bad_rt_cfg, sample_tokens)

  def test_jax_embedding_merge_and_bias_vl_routing_without_torch(self) -> None:
    mesh = mm_testing.single_device_mesh()
    cfg = mm_testing.mini_vision_maxtext_config(
        vocab_size=mm_testing.MINI_VOCAB_SIZE,
        ds41_image_token_id=mm_testing.MINI_IMAGE_TOKEN_ID,
        ds41_engram_compressed_vocab_size=0,
        tokenizer_path=mm_testing.OFFLINE_TOKENIZER_PATH,
        scan_layers=False,
    )
    with mesh:
      mt_model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      tokens_j = jnp.full((2, 16), 10, dtype=jnp.int32).at[:, 2:5].set(mm_testing.MINI_IMAGE_TOKEN_ID)
      positions_j = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
      bidir_mask = mm_processor.get_bidirectional_mask_vision(cfg, tokens_j)
      spans = jnp.full((2, 1, 3, cfg.emb_dim), 0.75, dtype=jnp.float32)
      mm_in = MultimodalInput(image_embeddings=spans, bidirectional_mask=bidir_mask)
      merged = mt_model.decoder._apply_embedding(  # pylint: disable=protected-access
          mt_model.token_embedder,
          tokens_j,
          positions_j,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          multimodal_input=mm_in,
      )
      np.testing.assert_allclose(np.asarray(merged[:, 2:5, :]), np.asarray(spans[:, 0, :, :]), rtol=0.0, atol=0.0)

      gate = mt_model.decoder.layers_0.mlp.MoeBlock_0.gate
      gate.bias[...] = jnp.array([0.5, -0.5, 0.0, 0.0], dtype=jnp.float32)
      gate.bias_vl[...] = jnp.array([-0.5, 0.5, 0.0, 0.0], dtype=jnp.float32)
      x_in = jnp.zeros((2, 16, cfg.emb_dim), dtype=jnp.float32)
      gl_txt, _ = gate(x_in, image_mask=None)
      gl_vl, _ = gate(x_in, image_mask=bidir_mask)
      self.assertFalse(bool(jnp.array_equal(gl_txt[:, 2:5, :], gl_vl[:, 2:5, :])))
      self.assertTrue(bool(jnp.array_equal(gl_txt[:, :2, :], gl_vl[:, :2, :])))

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_decoder_embedding_merge_bitwise_partial_masks_and_negative_control(self) -> None:
    mesh = mm_testing.single_device_mesh()
    for grid in (multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT):
      ref_mm, batch = _build_reference_pair(grid, seed=21, dtype=torch.float32)
      cfg = mm_testing.mini_vision_maxtext_config(grid=grid, scan_layers=False)
      with mesh:
        mt_model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
        mm_testing.copy_full_multimodal_model(mt_model, ref_mm, cfg, jnp.float32)

        with torch.no_grad():
          ref_merged = ref_mm.merge_image_embeddings(batch.images, ref_mm.embed(batch.input_ids).clone())
          ref_spans = _reference_span_embeddings(ref_mm, batch)

        tokens_j = jnp.asarray(batch.input_ids.numpy(), dtype=jnp.int32)
        positions_j = jnp.broadcast_to(jnp.arange(tokens_j.shape[1], dtype=jnp.int32), tokens_j.shape)
        bidir_mask = mm_processor.get_bidirectional_mask_vision(cfg, tokens_j)
        np.testing.assert_array_equal(np.asarray(bidir_mask), batch.image_mask.numpy())

        # Bitwise fp32 merge check when given exact reference span embeddings.
        mm_in_exact = MultimodalInput(
            image_embeddings=mm_testing.torch_to_jax(ref_spans, jnp.float32),
            bidirectional_mask=bidir_mask,
        )
        merged_exact = mt_model.decoder._apply_embedding(  # pylint: disable=protected-access
            mt_model.token_embedder,
            tokens_j,
            positions_j,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
            multimodal_input=mm_in_exact,
        )
        self.assertTrue(
            np.array_equal(np.asarray(merged_exact), ref_merged.numpy()),
            msg=f"T6 bitwise merge mismatch for grid={grid}",
        )

        # Partial image validity masks (`[[1, 0], [0, 1]]` and `[[1, 0], [1, 0]]`): bitwise equal to reference.
        for valid_pairs in (((1, 0), (0, 1)), ((1, 0), (1, 0))):
          partial_ids = batch.input_ids.clone()
          partial_images = []
          for b_idx in range(batch.input_ids.shape[0]):
            row_imgs = []
            for img_idx, keep in enumerate(valid_pairs[b_idx]):
              img = batch.images[b_idx][img_idx]
              if keep:
                row_imgs.append(img)
              else:
                span_end = img.start + int(img.types.numel())
                partial_ids[b_idx, img.start : span_end] = 7 + b_idx + img_idx
            partial_images.append(row_imgs)

          with torch.no_grad():
            ref_partial = ref_mm.merge_image_embeddings(partial_images, ref_mm.embed(partial_ids).clone())
          partial_tokens_j = jnp.asarray(partial_ids.numpy(), dtype=jnp.int32)
          partial_bidir = mm_processor.get_bidirectional_mask_vision(cfg, partial_tokens_j)
          image_masks_j = jnp.asarray(valid_pairs, dtype=jnp.int32)
          mm_in_partial = MultimodalInput(
              image_embeddings=mm_testing.torch_to_jax(ref_spans, jnp.float32),
              image_masks=image_masks_j,
              bidirectional_mask=partial_bidir,
          )
          merged_partial = mt_model.decoder._apply_embedding(  # pylint: disable=protected-access
              mt_model.token_embedder,
              partial_tokens_j,
              positions_j,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
              multimodal_input=mm_in_partial,
          )
          self.assertTrue(
              np.array_equal(np.asarray(merged_partial), ref_partial.numpy()),
              msg=f"T6 partial mask {valid_pairs} bitwise merge mismatch for grid={grid}",
          )

        # End-to-end VisionEncoder + _apply_embedding check and negative control on `projector/image_start`.
        enc_mod = getattr(mt_model.vision_encoder, mt_model.vision_encoder.encoder_name)
        proj_mod = getattr(mt_model.vision_encoder, mt_model.vision_encoder.projector_name)
        enc_gdef, enc_pstate, enc_pure, enc_ostate = mm_testing.extract_pure_params(enc_mod)
        proj_gdef, proj_pstate, proj_pure, proj_ostate = mm_testing.extract_pure_params(proj_mod)
        emb_gdef, emb_pstate, emb_pure, emb_ostate = mm_testing.extract_pure_params(mt_model.token_embedder)
        cand_params = {
            "encoder": enc_pure,
            "projector": proj_pure,
            "embedder": emb_pure,
        }
        cand_patches = mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float32)

        def cand_fn(
            params: dict[str, Any],
            patches_in: jax.Array,
            enc_gdef=enc_gdef,
            enc_pstate=enc_pstate,
            enc_ostate=enc_ostate,
            proj_gdef=proj_gdef,
            proj_pstate=proj_pstate,
            proj_ostate=proj_ostate,
            emb_gdef=emb_gdef,
            emb_pstate=emb_pstate,
            emb_ostate=emb_ostate,
            decoder=mt_model.decoder,
            tokens=tokens_j,
            positions=positions_j,
            bmask=bidir_mask,
        ) -> jax.Array:
          enc = mm_testing.apply_with_pure_params(enc_gdef, enc_pstate, params["encoder"], enc_ostate)
          proj = mm_testing.apply_with_pure_params(proj_gdef, proj_pstate, params["projector"], proj_ostate)
          emb = mm_testing.apply_with_pure_params(emb_gdef, emb_pstate, params["embedder"], emb_ostate)
          spans = proj(enc(patches_in, deterministic=True))
          mm_in = MultimodalInput(image_embeddings=spans, bidirectional_mask=bmask)
          return decoder._apply_embedding(  # pylint: disable=protected-access
              emb,
              tokens,
              positions,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
              multimodal_input=mm_in,
          )

        e2e_merged = np.asarray(cand_fn(cand_params, cand_patches))
        ref_merged_64 = ref_merged.to(torch.float64).numpy()
        ok_e2e, m_e2e = layer_parity.compare(e2e_merged, ref_merged_64, layer_parity.FP32_TOLERANCE)
        self.assertTrue(ok_e2e, msg=f"T6 end-to-end merge failed for grid={grid}: {m_e2e}")

        fired, neg_m = layer_parity.negative_control_fires(
            ref_out=ref_merged_64,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_patches,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="projector/image_start",
            rel_scale=_NEGCTL_REL_SCALE,
            seed=601,
        )
        self.assertTrue(fired, msg=f"T6 negative control on projector/image_start did not fire: {neg_m}")


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41EngramHasherMultimodalTest(unittest.TestCase):
  """T7: `DeepSeek41NgramHasher` DEAD barrier on image spans (start/middle/end), segments, and gather bounds."""

  def test_hash_ids_dead_barrier_bounds_segments_and_discriminating_control(self) -> None:
    args = multimodal.mini_model_args()
    ref_mm = multimodal.build_multimodal_model(args, tokenizer=_get_tokenizer())
    cfg = mm_testing.mini_vision_maxtext_config(grid=multimodal.MINI_GRID_PADDED)

    token_map, compressed_vocab_size = deepseek41.compressed_token_map(cfg)
    hashers = [
        engram_deepseek41.make_hasher(cfg, layer_id, token_map, compressed_vocab_size)
        for layer_id in args.engram_layer_ids
    ]

    # Spans starting at position 0, middle, and ending at seq_len (29 + 11 == 40).
    placement_cases = (
        ((0, 15), (0, 20)),
        ((4, 20), (6, 22)),
        ((5, 29), (3, 29)),
    )
    for case_idx, span_starts in enumerate(placement_cases):
      batch = multimodal.build_mini_vl_batch(
          args,
          grid=multimodal.MINI_GRID_PADDED,
          span_starts=span_starts,
          seed=300 + case_idx,
      )
      ref_hashes = ref_mm.engram_hash(batch.input_ids, 0, ~batch.image_mask).numpy()
      tokens_j = jnp.asarray(batch.input_ids.numpy(), dtype=jnp.int32)
      img_mask_j = jnp.asarray(batch.image_mask.numpy(), dtype=jnp.bool_)

      for li, hasher in enumerate(hashers):
        per_shift = hasher._gather_ngram_tokens(tokens_j, image_mask=img_mask_j)  # pylint: disable=protected-access
        self.assertEqual(len(per_shift), cfg.ds41_engram_max_ngram_size)
        for shift, shift_tokens in enumerate(per_shift):
          arr = np.asarray(shift_tokens)
          self.assertGreaterEqual(int(arr.min()), 0, msg=f"Negative gather index at shift={shift}")
          self.assertLess(int(arr.max()), args.engram_compressed_vocab_size)

        cand_hashes = np.asarray(hasher(tokens_j, image_mask=img_mask_j))
        np.testing.assert_array_equal(
            cand_hashes,
            ref_hashes[:, :, li, :],
            err_msg=f"T7 hash mismatch on span_starts={span_starts}, layer_index={li}",
        )

        # Discriminating control: omitting image_mask changes hashes both on image spans and on post-span lookback.
        unmasked_hashes = np.asarray(hasher(tokens_j, image_mask=None))
        img_mask_np = batch.image_mask.numpy()
        self.assertFalse(
            np.array_equal(cand_hashes[img_mask_np], unmasked_hashes[img_mask_np]),
            msg="T7 control failed: hashes on image positions did not change when image_mask was dropped.",
        )
        # First span in row 0 ends before a text token (`0 + 11 = 11` or `4 + 11 = 15` or `5 + 11 = 16`).
        post_span_pos = span_starts[0][0] + 11
        self.assertFalse(
            np.array_equal(cand_hashes[0, post_span_pos], unmasked_hashes[0, post_span_pos]),
            msg=f"T7 control failed: lookback hash at post-span text token {post_span_pos} did not change.",
        )

    # Multi-segment + padding (`decoder_segment_ids` with segments 1, 2, and padding 0) combined with image spans.
    seg_batch = multimodal.build_mini_vl_batch(
        args,
        grid=multimodal.MINI_GRID_PADDED,
        span_starts=((2, 20), (0, 18)),
        seed=350,
    )
    # Segment 1: [0, 16), Segment 2: [16, 35), Padding 0: [35, 40).
    seg_ids_np = np.zeros((2, 40), dtype=np.int32)
    seg_ids_np[:, :16] = 1
    seg_ids_np[:, 16:35] = 2
    seg_ids_j = jnp.asarray(seg_ids_np, dtype=jnp.int32)
    tokens_j = jnp.asarray(seg_batch.input_ids.numpy(), dtype=jnp.int32)
    img_mask_j = jnp.asarray(seg_batch.image_mask.numpy(), dtype=jnp.bool_)

    # Reference hashes computed independently on each active segment, and pure pad_id on padding segment 0.
    seg1_ref = ref_mm.engram_hash(seg_batch.input_ids[:, :16], 0, ~seg_batch.image_mask[:, :16]).numpy()
    seg2_ref = ref_mm.engram_hash(seg_batch.input_ids[:, 16:35], 0, ~seg_batch.image_mask[:, 16:35]).numpy()
    pad_ids_t = torch.full((2, 5), args.engram_pad_id, dtype=torch.int64)
    pad_mask_t = torch.zeros((2, 5), dtype=torch.bool)
    pad_ref = ref_mm.engram_hash(pad_ids_t, 0, pad_mask_t).numpy()
    expected_seg_hashes = np.concatenate([seg1_ref, seg2_ref, pad_ref], axis=1)

    for li, hasher in enumerate(hashers):
      per_shift_seg = hasher._gather_ngram_tokens(  # pylint: disable=protected-access
          tokens_j, decoder_segment_ids=seg_ids_j, image_mask=img_mask_j
      )
      for shift_tokens in per_shift_seg:
        self.assertGreaterEqual(int(np.asarray(shift_tokens).min()), 0)
      cand_seg = np.asarray(hasher(tokens_j, decoder_segment_ids=seg_ids_j, image_mask=img_mask_j))
      np.testing.assert_array_equal(cand_seg, expected_seg_hashes[:, :, li, :])

    # Shape validation paired with valid call above.
    bad_mask = jnp.ones((2, 39), dtype=jnp.bool_)
    with self.assertRaisesRegex(ValueError, "image_mask shape"):
      hashers[0](tokens_j, image_mask=bad_mask)


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41EngramLayerMultimodalTest(unittest.TestCase):
  """T8: `DeepSeek41Engram` forward + all-leaf/input gradients in fp32 and fp64, gate zeroing, and negctl."""

  def setUp(self) -> None:
    super().setUp()
    # Pin PyTorch CPU OpenMP to 1 thread to prevent multi-threaded fp32 partial-sum noise near sqrt(|dot|) clamp.
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    self.addCleanup(torch.set_num_threads, orig_threads)

  def test_engram_layer_fp32_fp64_parity_and_negative_control(self) -> None:
    grid = multimodal.MINI_GRID_PADDED
    ref_mm, batch = _build_reference_pair(grid, seed=1, dtype=torch.float32)
    args = multimodal.mini_model_args()
    engram_token_mask_t = ~batch.image_mask
    engram_hashes_all = ref_mm.engram_hash(batch.input_ids, 0, engram_token_mask_t)

    # Capture the actual multimodal hidden stream `h` entering every engram layer (L1 and L6).
    engram_inputs: dict[int, torch.Tensor] = {}
    with torch.no_grad():
      h_emb = ref_mm.merge_image_embeddings(batch.images, ref_mm.embed(batch.input_ids).clone())
      h_cur = h_emb.unsqueeze(2).repeat(1, 1, ref_mm.hc_mult, 1)
      pre_cur = ref_model.make_identity_pre_mix(h_cur, ref_mm.hc_mult)
      for lid, layer in enumerate(ref_mm.layers):
        if layer.engram is not None:
          engram_inputs[lid] = h_cur.detach().clone()
          h_cur = layer.engram(h_cur, engram_hashes_all[:, :, layer.engram.layer_hash_index, :], engram_token_mask_t)
        h_cur, pre_cur = layer(h_cur, 0, pre_cur, batch.image_mask)

    self.assertEqual(tuple(engram_inputs.keys()), args.engram_layer_ids)
    grad_classes = layer_parity.GradientClasses(
        compared=frozenset({"wkv.weight", "q_weight", "k_weight"}),
        padded=frozenset({"embed.weight"}),
    )
    weight_map = mm_testing.engram_weight_map(cand_root="engram")
    cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
    mesh = mm_testing.single_device_mesh()
    img_mask_j = jnp.asarray(batch.image_mask.numpy(), dtype=jnp.bool_)

    for engram_layer_id in args.engram_layer_ids:
      ref_engram_mod = ref_mm.layers[engram_layer_id].engram
      assert ref_engram_mod is not None
      ref_h32 = engram_inputs[engram_layer_id]
      hash_ids_t = engram_hashes_all[:, :, ref_engram_mod.layer_hash_index, :].clone()
      hash_ids_j = jnp.asarray(hash_ids_t.numpy(), dtype=jnp.int32)
      num_embeddings = args.engram_num_embeddings[ref_engram_mod.layer_hash_index]
      ref_params32 = dict(ref_engram_mod.named_parameters())

      def make_ref_fn(mod: Any, h_ids: torch.Tensor, tok_mask: torch.Tensor):
        def _fn(params: dict[str, torch.Tensor], h_in: torch.Tensor) -> torch.Tensor:
          return functional_call(mod, params, (h_in, h_ids, tok_mask))

        return _fn

      ref_fn32 = make_ref_fn(ref_engram_mod, hash_ids_t, engram_token_mask_t)
      with multimodal.fp64_reference_context():
        noise_floor = layer_parity.reference_noise_floor(ref_fn32, ref_params32, ref_h32)
      self.assertTrue(noise_floor.finite)
      self.assertLessEqual(noise_floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

      with mesh:
        mt_engram32 = engram_deepseek41.DeepSeek41Engram(cfg, num_embeddings=num_embeddings, rngs=nnx.Rngs(params=0))
        mm_testing.copy_engram(mt_engram32, ref_engram_mod, jnp.float32)
        gdef32, pstate32, pure32, ostate32 = mm_testing.extract_pure_params(mt_engram32)
        cand_params32 = {"engram": pure32}
        cand_h32 = mm_testing.torch_to_jax(ref_h32, jnp.float32)

        def cand_fn32(
            params: dict[str, Any],
            h_in: jax.Array,
            gd=gdef32,
            ps=pstate32,
            os_=ostate32,
            hids=hash_ids_j,
        ) -> jax.Array:
          mod = mm_testing.apply_with_pure_params(gd, ps, params["engram"], os_)
          return mod(h_in, hids, image_mask=img_mask_j)

        report32 = layer_parity.check_parity(
            ref_fn=ref_fn32,
            ref_params=ref_params32,
            ref_input=ref_h32,
            cand_fn=cand_fn32,
            cand_params=cand_params32,
            cand_input=cand_h32,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=801 + engram_layer_id,
        )
        self.assertTrue(report32.passed, msg=f"T8 L{engram_layer_id} fp32 failures: {report32.failures}")

        in_ok32, in_m32 = mm_testing.check_input_gradient(
            ref_fn=lambda h_in, f=ref_fn32, p=ref_params32: f(p, h_in),
            ref_input=ref_h32,
            cand_fn=lambda h_in, f=cand_fn32, p=cand_params32: f(p, h_in),
            cand_input=cand_h32,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=811 + engram_layer_id,
        )
        self.assertTrue(in_ok32, msg=f"T8 L{engram_layer_id} fp32 input gradient failed: {in_m32}")

        # Image-span positions must pass `h` through unchanged (`gate == 0` exact), whereas `image_mask=None` alters them.
        out32_np = np.asarray(cand_fn32(cand_params32, cand_h32))
        img_mask_np = batch.image_mask.numpy()
        self.assertTrue(np.array_equal(out32_np[img_mask_np], np.asarray(cand_h32)[img_mask_np]))
        out32_unmasked = np.asarray(mt_engram32(cand_h32, hash_ids_j, image_mask=None))
        self.assertFalse(np.array_equal(out32_np[img_mask_np], out32_unmasked[img_mask_np]))

        # Shape mismatch error path paired with valid call above.
        bad_mask = jnp.ones((2, 39), dtype=jnp.bool_)
        with self.assertRaisesRegex(ValueError, "image_mask shape"):
          mt_engram32(cand_h32, hash_ids_j, image_mask=bad_mask)

        fired, neg_m = layer_parity.negative_control_fires(
            ref_out=report32.ref_out,
            cand_fn=cand_fn32,
            cand_params=cand_params32,
            cand_input=cand_h32,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="engram/wkv/kernel",
            rel_scale=_NEGCTL_REL_SCALE,
            seed=821 + engram_layer_id,
        )
        self.assertTrue(fired, msg=f"T8 L{engram_layer_id} negative control on engram/wkv/kernel did not fire: {neg_m}")

      # Float64 parity (`frob_rel <= 1e-10`) under upcast-only reference and JAX contexts.
      with multimodal.fp64_reference_context(), mm_testing.fp64_maxtext_context(cfg), mesh:
        ref_engram64 = copy.deepcopy(ref_engram_mod).to(torch.float64)
        ref_params64 = dict(ref_engram64.named_parameters())
        ref_h64 = ref_h32.to(torch.float64)
        ref_fn64 = make_ref_fn(ref_engram64, hash_ids_t, engram_token_mask_t)

        mt_engram64 = engram_deepseek41.DeepSeek41Engram(cfg, num_embeddings=num_embeddings, rngs=nnx.Rngs(params=0))
        mm_testing.copy_engram(mt_engram64, ref_engram64, jnp.float64)
        gdef64, pstate64, pure64, ostate64 = mm_testing.extract_pure_params(mt_engram64)
        cand_params64 = {"engram": pure64}
        cand_h64 = mm_testing.torch_to_jax(ref_h64, jnp.float64)

        def cand_fn64(
            params: dict[str, Any],
            h_in: jax.Array,
            gd=gdef64,
            ps=pstate64,
            os_=ostate64,
            hids=hash_ids_j,
        ) -> jax.Array:
          mod = mm_testing.apply_with_pure_params(gd, ps, params["engram"], os_)
          return mod(h_in, hids, image_mask=img_mask_j)

        report64 = layer_parity.check_parity(
            ref_fn=ref_fn64,
            ref_params=ref_params64,
            ref_input=ref_h64,
            cand_fn=cand_fn64,
            cand_params=cand_params64,
            cand_input=cand_h64,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP64_TOLERANCE,
            grad_tol=layer_parity.FP64_TOLERANCE,
            cotangent_seed=831 + engram_layer_id,
        )
        self.assertTrue(report64.passed, msg=f"T8 L{engram_layer_id} fp64 failures: {report64.failures}")

        in_ok64, in_m64 = mm_testing.check_input_gradient(
            ref_fn=lambda h_in, f=ref_fn64, p=ref_params64: f(p, h_in),
            ref_input=ref_h64,
            cand_fn=lambda h_in, f=cand_fn64, p=cand_params64: f(p, h_in),
            cand_input=cand_h64,
            grad_tol=layer_parity.FP64_TOLERANCE,
            cotangent_seed=841 + engram_layer_id,
        )
        self.assertTrue(in_ok64, msg=f"T8 L{engram_layer_id} fp64 input gradient failed: {in_m64}")


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41MoEBiasVlParityTest(unittest.TestCase):
  """T9: `GateLogit` and `RoutedAndSharedMoE` with `bias_vl` parity, fp64, TP/EP HLO sharding, and guards."""

  def test_gate_and_moe_bias_vl_fp32_fp64_discriminating_and_negctl(self) -> None:
    grid = multimodal.MINI_GRID_PADDED
    ref_mm, batch = _build_reference_pair(grid, seed=51, dtype=torch.float32)
    args = multimodal.mini_model_args()
    total_flips = multimodal.count_bias_vl_routing_changes(ref_mm, batch)
    self.assertGreaterEqual(total_flips, 1)

    # Pick a decoder layer and construct input `x` with realistic variance where `bias_vl` flips >= 1 image token.
    rng_x = torch.Generator().manual_seed(510)
    ref_x32 = torch.randn(
        (batch.input_ids.shape[0], batch.input_ids.shape[1], args.dim),
        generator=rng_x,
        dtype=torch.float32,
    )
    img_mask_flat_t = batch.image_mask.reshape(-1)
    chosen_layer_id: int | None = None
    for lid, blk in enumerate(ref_mm.layers):
      with torch.no_grad():
        _, idx_vl = blk.ffn.gate(ref_x32.reshape(-1, args.dim), img_mask_flat_t)
        _, idx_txt = blk.ffn.gate(ref_x32.reshape(-1, args.dim), None)
      vl_sorted = torch.sort(idx_vl[img_mask_flat_t], dim=-1).values
      txt_sorted = torch.sort(idx_txt[img_mask_flat_t], dim=-1).values
      if not torch.equal(vl_sorted, txt_sorted):
        chosen_layer_id = lid
        break
    self.assertIsNotNone(chosen_layer_id)
    assert chosen_layer_id is not None

    ref_ffn32 = ref_mm.layers[chosen_layer_id].ffn
    ref_gate32 = ref_ffn32.gate
    bsz, seqlen = batch.input_ids.shape
    topk = args.n_activated_experts

    ref_gate_params32 = dict(ref_gate32.named_parameters())
    gate_grad_classes = layer_parity.GradientClasses(
        compared=frozenset({"weight"}),
        non_gradient=frozenset({"bias", "bias_vl"}),
    )
    gate_weight_map = {"weight": "gate/kernel:T"}

    def ref_gate_weights_fn32(params: dict[str, torch.Tensor], x_in: torch.Tensor) -> torch.Tensor:
      w, _ = functional_call(ref_gate32, params, (x_in.reshape(-1, args.dim), img_mask_flat_t))
      return w.reshape(bsz, seqlen, topk)

    with multimodal.fp64_reference_context():
      noise_floor = layer_parity.reference_noise_floor(ref_gate_weights_fn32, ref_gate_params32, ref_x32)
    self.assertTrue(noise_floor.finite)

    cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
    mesh = mm_testing.single_device_mesh()
    img_mask_j = jnp.asarray(batch.image_mask.numpy(), dtype=jnp.bool_)
    cand_x32 = mm_testing.torch_to_jax(ref_x32, jnp.float32)

    with mesh:
      mt_mlp32 = deepseek41.DeepSeek41DecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          spec=deepseek41.build_layer_specs(cfg)[chosen_layer_id],
          quant=None,
          rngs=nnx.Rngs(params=0),
      ).mlp
      mm_testing.copy_moe(mt_mlp32, ref_ffn32, jnp.float32)
      moe_block32 = mt_mlp32.MoeBlock_0

      # 1. Exact top-k expert indices + discriminating `bias_vl` check + un-biased post-topk weight gather check.
      with torch.no_grad():
        _, ref_idx_vl = ref_gate32(ref_x32.reshape(-1, args.dim), img_mask_flat_t)
      gate_logits_vl, pre_bias_vl = moe_block32.gate(cand_x32, image_mask=img_mask_j)
      cand_w_vl, cand_idx_vl = moe_block32.get_topk(gate_logits_vl, pre_bias_vl, moe_block32.rngs)
      np.testing.assert_array_equal(
          np.asarray(cand_idx_vl),
          ref_idx_vl.reshape(bsz, seqlen, topk).numpy(),
      )

      gate_logits_txt, pre_bias_txt = moe_block32.gate(cand_x32, image_mask=None)
      cand_w_txt, cand_idx_txt = moe_block32.get_topk(gate_logits_txt, pre_bias_txt, moe_block32.rngs)
      img_mask_np = batch.image_mask.numpy()
      text_mask_np = ~img_mask_np
      # Discriminating: >= 1 image token changes routed expert set, while text tokens are bitwise identical.
      self.assertFalse(
          np.array_equal(
              np.sort(np.asarray(cand_idx_vl)[img_mask_np], axis=-1),
              np.sort(np.asarray(cand_idx_txt)[img_mask_np], axis=-1),
          )
      )
      self.assertTrue(np.array_equal(np.asarray(cand_idx_vl)[text_mask_np], np.asarray(cand_idx_txt)[text_mask_np]))
      self.assertTrue(np.array_equal(np.asarray(cand_w_vl)[text_mask_np], np.asarray(cand_w_txt)[text_mask_np]))

      out_moe_vl, _, _ = mt_mlp32(cand_x32, image_mask=img_mask_j)
      out_moe_txt, _, _ = mt_mlp32(cand_x32, image_mask=None)
      self.assertTrue(np.array_equal(np.asarray(out_moe_vl)[text_mask_np], np.asarray(out_moe_txt)[text_mask_np]))
      self.assertFalse(np.array_equal(np.asarray(out_moe_vl)[img_mask_np], np.asarray(out_moe_txt)[img_mask_np]))

      # Verify post-topk expert weights gather from unbiased `pre_bias_logits`: shifting `bias_vl` uniformly by +0.25
      # preserves argmax/top-k indices and must leave `cand_w_vl` and `out_moe_vl` bitwise identical.
      orig_bias_vl = jnp.array(moe_block32.gate.bias_vl[...])
      moe_block32.gate.bias_vl[...] = orig_bias_vl + jnp.float32(0.25)
      shifted_logits, shifted_pre = moe_block32.gate(cand_x32, image_mask=img_mask_j)
      shifted_w, shifted_idx = moe_block32.get_topk(shifted_logits, shifted_pre, moe_block32.rngs)
      out_moe_shifted, _, _ = mt_mlp32(cand_x32, image_mask=img_mask_j)
      moe_block32.gate.bias_vl[...] = orig_bias_vl
      np.testing.assert_array_equal(np.asarray(shifted_idx), np.asarray(cand_idx_vl))
      self.assertTrue(np.array_equal(np.asarray(shifted_w), np.asarray(cand_w_vl)))
      self.assertTrue(np.array_equal(np.asarray(out_moe_shifted), np.asarray(out_moe_vl)))

      # 2. Gate routing weights + grads (`gate/kernel` and input `x`) in fp32 + negative control on `gate/kernel`.
      g_gdef32, g_pstate32, g_pure32, g_ostate32 = mm_testing.extract_pure_params(moe_block32.gate)
      cand_gate_params32 = {"gate": g_pure32}

      def cand_gate_fn32(params: dict[str, Any], x_in: jax.Array) -> jax.Array:
        gate_mod = mm_testing.apply_with_pure_params(g_gdef32, g_pstate32, params["gate"], g_ostate32)
        g_logits, pre_b = gate_mod(x_in, image_mask=img_mask_j)
        weights, _ = moe_block32.get_topk(g_logits, pre_b, moe_block32.rngs)
        return weights

      gate_report32 = layer_parity.check_parity(
          ref_fn=ref_gate_weights_fn32,
          ref_params=ref_gate_params32,
          ref_input=ref_x32,
          cand_fn=cand_gate_fn32,
          cand_params=cand_gate_params32,
          cand_input=cand_x32,
          weight_map=gate_weight_map,
          grad_classes=gate_grad_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=901,
      )
      self.assertTrue(gate_report32.passed, msg=f"T9 gate fp32 failures: {gate_report32.failures}")

      gate_in_ok32, gate_in_m32 = mm_testing.check_input_gradient(
          ref_fn=lambda x_in: ref_gate_weights_fn32(ref_gate_params32, x_in),
          ref_input=ref_x32,
          cand_fn=lambda x_in: cand_gate_fn32(cand_gate_params32, x_in),
          cand_input=cand_x32,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=902,
      )
      self.assertTrue(gate_in_ok32, msg=f"T9 gate fp32 input gradient failed: {gate_in_m32}")

      fired_gate, neg_gate_m = layer_parity.negative_control_fires(
          ref_out=gate_report32.ref_out,
          cand_fn=cand_gate_fn32,
          cand_params=cand_gate_params32,
          cand_input=cand_x32,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          target_path="gate/kernel",
          rel_scale=_NEGCTL_REL_SCALE,
          seed=903,
      )
      self.assertTrue(fired_gate, msg=f"T9 negative control on gate/kernel did not fire: {neg_gate_m}")

      # 3. Full `RoutedAndSharedMoE` forward + all-leaf grads (gate, all routed experts, shared expert, input) in fp32.
      ref_moe_params32 = dict(ref_ffn32.named_parameters())
      moe_grad_classes = layer_parity.GradientClasses(
          compared=frozenset(k for k in ref_moe_params32 if k not in ("gate.bias", "gate.bias_vl")),
          non_gradient=frozenset({"gate.bias", "gate.bias_vl"}),
      )
      num_experts = args.n_routed_experts
      moe_weight_map = {
          "gate.weight": "MoeBlock_0/gate/kernel:T",
          "shared_experts.w1.weight": "shared_experts/wi_0/kernel:T",
          "shared_experts.w3.weight": "shared_experts/wi_1/kernel:T",
          "shared_experts.w2.weight": "shared_experts/wo/kernel:T",
      }
      for e in range(num_experts):
        moe_weight_map[f"experts.{e}.w1.weight"] = f"MoeBlock_0/wi_0/{e}:T"
        moe_weight_map[f"experts.{e}.w3.weight"] = f"MoeBlock_0/wi_1/{e}:T"
        moe_weight_map[f"experts.{e}.w2.weight"] = f"MoeBlock_0/wo/{e}:T"

      m_gdef, m_pstate, m_pure_raw, m_ostate = mm_testing.extract_pure_params(mt_mlp32)
      cand_moe_params = {
          "MoeBlock_0": {
              "gate": m_pure_raw["MoeBlock_0"]["gate"],
              "wi_0": {str(e): m_pure_raw["MoeBlock_0"]["wi_0"][e] for e in range(num_experts)},
              "wi_1": {str(e): m_pure_raw["MoeBlock_0"]["wi_1"][e] for e in range(num_experts)},
              "wo": {str(e): m_pure_raw["MoeBlock_0"]["wo"][e] for e in range(num_experts)},
          },
          "shared_experts": m_pure_raw["shared_experts"],
      }

      def cand_moe_fn(params: dict[str, Any], x_in: jax.Array) -> jax.Array:
        restored = {
            "MoeBlock_0": {
                "gate": params["MoeBlock_0"]["gate"],
                "wi_0": jnp.stack([params["MoeBlock_0"]["wi_0"][str(e)] for e in range(num_experts)], axis=0),
                "wi_1": jnp.stack([params["MoeBlock_0"]["wi_1"][str(e)] for e in range(num_experts)], axis=0),
                "wo": jnp.stack([params["MoeBlock_0"]["wo"][str(e)] for e in range(num_experts)], axis=0),
            },
            "shared_experts": params["shared_experts"],
        }
        mod = mm_testing.apply_with_pure_params(m_gdef, m_pstate, restored, m_ostate)
        out, _, _ = mod(x_in, image_mask=img_mask_j)
        return out

      def ref_moe_fn(params: dict[str, torch.Tensor], x_in: torch.Tensor) -> torch.Tensor:
        return functional_call(ref_ffn32, params, (x_in, batch.image_mask))

      moe_report32 = layer_parity.check_parity(
          ref_fn=ref_moe_fn,
          ref_params=ref_moe_params32,
          ref_input=ref_x32,
          cand_fn=cand_moe_fn,
          cand_params=cand_moe_params,
          cand_input=cand_x32,
          weight_map=moe_weight_map,
          grad_classes=moe_grad_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=904,
      )
      self.assertTrue(moe_report32.passed, msg=f"T9 full MoE fp32 failures: {moe_report32.failures}")

      moe_in_ok32, moe_in_m32 = mm_testing.check_input_gradient(
          ref_fn=lambda x_in: ref_moe_fn(ref_moe_params32, x_in),
          ref_input=ref_x32,
          cand_fn=lambda x_in: cand_moe_fn(cand_moe_params, x_in),
          cand_input=cand_x32,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=905,
      )
      self.assertTrue(moe_in_ok32, msg=f"T9 full MoE fp32 input gradient failed: {moe_in_m32}")

      fired_moe, neg_moe_m = layer_parity.negative_control_fires(
          ref_out=moe_report32.ref_out,
          cand_fn=cand_moe_fn,
          cand_params=cand_moe_params,
          cand_input=cand_x32,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          target_path="MoeBlock_0/gate/kernel",
          rel_scale=_NEGCTL_REL_SCALE,
          seed=906,
      )
      self.assertTrue(fired_moe, msg=f"T9 negative control on MoeBlock_0/gate/kernel did not fire: {neg_moe_m}")

    # 4. Float64 gate parity (`frob_rel <= 1e-10`) on indices, routing weights, `gate/kernel` grad, and input grad.
    with multimodal.fp64_reference_context(), mm_testing.fp64_maxtext_context(cfg), mesh:
      ref_gate64 = ref_mm.to(torch.float64).layers[chosen_layer_id].ffn.gate
      # Keep routing biases as exact float64 promotions of their float32 state so both sides add identical biases.
      ref_gate_params64 = dict(ref_gate64.named_parameters())
      ref_x64 = ref_x32.to(torch.float64)

      def ref_gate_weights_fn64(params: dict[str, torch.Tensor], x_in: torch.Tensor) -> torch.Tensor:
        w, _ = functional_call(ref_gate64, params, (x_in.reshape(-1, args.dim), img_mask_flat_t))
        return w.reshape(bsz, seqlen, topk)

      mt_layer64 = deepseek41.DeepSeek41DecoderLayer(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          spec=deepseek41.build_layer_specs(cfg)[chosen_layer_id],
          quant=None,
          rngs=nnx.Rngs(params=0),
      )
      mm_testing.copy_moe(mt_layer64.mlp, ref_mm.layers[chosen_layer_id].ffn, jnp.float64)
      moe_block64 = mt_layer64.mlp.MoeBlock_0
      cand_x64 = mm_testing.torch_to_jax(ref_x64, jnp.float64)

      g_logits64, pre_b64 = moe_block64.gate(cand_x64, image_mask=img_mask_j)
      _, cand_idx64 = moe_block64.get_topk(g_logits64, pre_b64, moe_block64.rngs)
      with torch.no_grad():
        raw_lin64 = torch.nn.functional.linear(ref_x64.reshape(-1, args.dim), ref_gate64.weight) / ref_gate64.gate_temp
        if ref_gate64.score_func == "softmax":
          ref_raw64 = raw_lin64.softmax(dim=-1)
        elif ref_gate64.score_func == "sigmoid":
          ref_raw64 = raw_lin64.sigmoid()
        else:
          ref_raw64 = torch.nn.functional.softplus(raw_lin64).sqrt()
        ref_bias64 = torch.where(img_mask_flat_t.unsqueeze(-1), ref_gate64.bias_vl, ref_gate64.bias)
        ref_biased_scores64 = (ref_raw64 + ref_bias64).reshape(bsz, seqlen, -1)
        _, ref_idx64 = ref_gate64(ref_x64.reshape(-1, args.dim), img_mask_flat_t)
      biased_ok64, biased_m64 = layer_parity.compare(
          np.asarray(g_logits64),
          ref_biased_scores64.cpu().to(torch.float64).numpy(),
          layer_parity.FP64_TOLERANCE,
      )
      self.assertTrue(biased_ok64, msg=f"T9 gate fp64 biased scores failed: {biased_m64}")
      self.assertEqual(g_logits64.dtype, jnp.float64)
      self.assertEqual(pre_b64.dtype, jnp.float64)
      np.testing.assert_array_equal(np.asarray(cand_idx64), ref_idx64.reshape(bsz, seqlen, topk).numpy())

      g_gdef64, g_pstate64, g_pure64, g_ostate64 = mm_testing.extract_pure_params(moe_block64.gate)
      cand_gate_params64 = {"gate": g_pure64}

      def cand_gate_fn64(params: dict[str, Any], x_in: jax.Array) -> jax.Array:
        gate_mod = mm_testing.apply_with_pure_params(g_gdef64, g_pstate64, params["gate"], g_ostate64)
        gl, pb = gate_mod(x_in, image_mask=img_mask_j)
        w, _ = moe_block64.get_topk(gl, pb, moe_block64.rngs)
        return w

      gate_report64 = layer_parity.check_parity(
          ref_fn=ref_gate_weights_fn64,
          ref_params=ref_gate_params64,
          ref_input=ref_x64,
          cand_fn=cand_gate_fn64,
          cand_params=cand_gate_params64,
          cand_input=cand_x64,
          weight_map=gate_weight_map,
          grad_classes=gate_grad_classes,
          fwd_tol=layer_parity.FP64_TOLERANCE,
          grad_tol=layer_parity.FP64_TOLERANCE,
          cotangent_seed=907,
      )
      self.assertTrue(gate_report64.passed, msg=f"T9 gate fp64 failures: {gate_report64.failures}")

      gate_in_ok64, gate_in_m64 = mm_testing.check_input_gradient(
          ref_fn=lambda x_in: ref_gate_weights_fn64(ref_gate_params64, x_in),
          ref_input=ref_x64,
          cand_fn=lambda x_in: cand_gate_fn64(cand_gate_params64, x_in),
          cand_input=cand_x64,
          grad_tol=layer_parity.FP64_TOLERANCE,
          cotangent_seed=908,
      )
      self.assertTrue(gate_in_ok64, msg=f"T9 gate fp64 input gradient failed: {gate_in_m64}")

  def test_gate_tp_ep_sharding_and_error_guards(self) -> None:
    self.assertGreaterEqual(len(jax.devices()), 4)
    tp_ep_mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("tensor", "expert"))
    ref_mm, batch = _build_reference_pair(multimodal.MINI_GRID_PADDED, seed=55, dtype=torch.float32)
    args = multimodal.mini_model_args()
    rng_x = torch.Generator().manual_seed(550)
    ref_x = torch.randn((batch.input_ids.shape[0], batch.input_ids.shape[1], args.dim), generator=rng_x)
    cand_x = mm_testing.torch_to_jax(ref_x, jnp.float32)
    img_mask_j = jnp.asarray(batch.image_mask.numpy(), dtype=jnp.bool_)

    cfg_tpep = mm_testing.mini_vision_maxtext_config()
    with tp_ep_mesh, flax.core.spmd.logical_axis_rules(cfg_tpep.logical_axis_rules):
      layer_tpep = deepseek41.DeepSeek41DecoderLayer(
          config=cfg_tpep,
          mesh=tp_ep_mesh,
          model_mode=MODEL_MODE_TRAIN,
          spec=deepseek41.build_layer_specs(cfg_tpep)[0],
          quant=None,
          rngs=nnx.Rngs(params=0),
      )
      mm_testing.copy_moe(layer_tpep.mlp, ref_mm.layers[0].ffn, jnp.float32)
      gate_tpep = layer_tpep.mlp.MoeBlock_0.gate

      t_gdef, t_vstate, t_ostate = nnx.split(gate_tpep, (nnx.Param, moe.MoEBiasVar), ...)
      t_pure = nnx.to_pure_dict(t_vstate)
      self.assertEqual(set(t_pure.keys()), {"kernel", "bias", "bias_vl"})
      named_shardings = nnx.to_pure_dict(sharding.nnx_construct_named_sharding(t_vstate, tp_ep_mesh))
      for key_name, ns in named_shardings.items():
        self.assertIsInstance(ns, jax.sharding.NamedSharding)
        self.assertTrue(
            all(axis is None for axis in ns.spec),
            msg=f"Expected replicated PartitionSpec for GateLogit.{key_name}, got {ns.spec}",
        )

      rep_sharding = jax.sharding.NamedSharding(tp_ep_mesh, jax.sharding.PartitionSpec())

      @functools.partial(jax.jit, out_shardings=rep_sharding)
      def jit_gate(var_dict: dict[str, Any], x_in: jax.Array, mask_in: jax.Array) -> jax.Array:
        vstate_copy = jax.tree.map(lambda arr: arr, t_vstate)
        ostate_copy = jax.tree.map(lambda arr: arr, t_ostate)
        nnx.replace_by_pure_dict(vstate_copy, var_dict)
        mod = nnx.merge(t_gdef, vstate_copy, ostate_copy)
        g_logits, _ = mod(x_in, image_mask=mask_in)
        return g_logits

      sharded_vars = jax.tree.map(jax.device_put, t_pure, named_shardings)
      sharded_x = jax.device_put(cand_x, rep_sharding)
      sharded_mask = jax.device_put(img_mask_j, rep_sharding)
      hlo_text = jit_gate.lower(sharded_vars, sharded_x, sharded_mask).compile().as_text()
      self.assertNotIn("all-gather", hlo_text)

      # Verify sharded GateLogit output matches unsharded computation within FP32_TOLERANCE.
      ref_scores = torch.nn.functional.softplus(
          torch.nn.functional.linear(ref_x, ref_mm.layers[0].ffn.gate.weight)
      ).sqrt()
      ref_bias = torch.where(
          batch.image_mask.unsqueeze(-1),
          ref_mm.layers[0].ffn.gate.bias_vl,
          ref_mm.layers[0].ffn.gate.bias,
      )
      expected_gate_logits = (ref_scores + ref_bias).detach().to(torch.float64).numpy()
      ok_tpep, m_tpep = layer_parity.compare(
          np.asarray(jit_gate(sharded_vars, sharded_x, sharded_mask)),
          expected_gate_logits,
          layer_parity.FP32_TOLERANCE,
      )
      self.assertTrue(ok_tpep, msg=f"T9 TP>1/EP>1 GateLogit output mismatch: {m_tpep}")

      # Negative controls: sharding `kernel` on "tensor" or `bias_vl` on "expert" must emit `all-gather`.
      neg_kernel_shardings = dict(named_shardings)
      neg_kernel_shardings["kernel"] = jax.sharding.NamedSharding(tp_ep_mesh, jax.sharding.PartitionSpec("tensor", None))
      neg_kernel_vars = jax.tree.map(jax.device_put, t_pure, neg_kernel_shardings)
      self.assertIn("all-gather", jit_gate.lower(neg_kernel_vars, sharded_x, sharded_mask).compile().as_text())

      neg_bias_shardings = dict(named_shardings)
      neg_bias_shardings["bias_vl"] = jax.sharding.NamedSharding(tp_ep_mesh, jax.sharding.PartitionSpec("expert"))
      neg_bias_vars = jax.tree.map(jax.device_put, t_pure, neg_bias_shardings)
      self.assertIn("all-gather", jit_gate.lower(neg_bias_vars, sharded_x, sharded_mask).compile().as_text())

    # Error-path and zero-state checks on a 1-device mesh (setup outside assertRaisesRegex, liveness-paired).
    mesh1 = mm_testing.single_device_mesh()
    cfg_text = mm_testing.mini_vision_maxtext_config(use_multimodal=False)
    with mesh1:
      layer_text = deepseek41.DeepSeek41DecoderLayer(
          config=cfg_text,
          mesh=mesh1,
          model_mode=MODEL_MODE_TRAIN,
          spec=deepseek41.build_layer_specs(cfg_text)[0],
          quant=None,
          rngs=nnx.Rngs(params=0),
      )
      gate_text = layer_text.mlp.MoeBlock_0.gate
      self.assertIsNone(gate_text.bias_vl)
      bias_state_dict = nnx.to_pure_dict(nnx.state(layer_text, moe.MoEBiasVar))
      self.assertNotIn("bias_vl", bias_state_dict["mlp"]["MoeBlock_0"]["gate"])

      # Liveness: image_mask=None succeeds; passing image_mask to use_bias_vl=False raises ValueError.
      text_logits, _ = gate_text(cand_x, image_mask=None)
      self.assertEqual(text_logits.shape, (2, 40, args.n_routed_experts))
      with self.assertRaisesRegex(ValueError, "use_bias_vl"):
        gate_text(cand_x, image_mask=img_mask_j)

      with self.assertRaisesRegex(ValueError, "use_bias_vl=True requires use_bias=True"):
        moe.GateLogit(
            in_features_shape=args.dim,
            out_features_shape=args.n_routed_experts,
            model_name=cfg_tpep.model_name,
            mesh=mesh1,
            use_bias=False,
            use_bias_vl=True,
            rngs=nnx.Rngs(params=0),
        )

      gate_mm = moe.GateLogit(
          in_features_shape=args.dim,
          out_features_shape=args.n_routed_experts,
          model_name=cfg_tpep.model_name,
          mesh=mesh1,
          use_bias=True,
          use_bias_vl=True,
          score_func="sqrtsoftplus",
          rngs=nnx.Rngs(params=0),
      )
      valid_mm_logits, _ = gate_mm(cand_x, image_mask=img_mask_j)
      self.assertEqual(valid_mm_logits.shape, (2, 40, args.n_routed_experts))
      bad_mask = jnp.ones((2, 39), dtype=jnp.bool_)
      with self.assertRaisesRegex(ValueError, "image_mask shape"):
        gate_mm(cand_x, image_mask=bad_mask)

      nd_kernel_init = layer_text.mlp.MoeBlock_0.kernel_init
      with self.assertRaisesRegex(ValueError, "is_hash_routing"):
        moe.RoutedMoE(
            config=cfg_tpep,
            num_experts=args.n_routed_experts,
            num_experts_per_tok=args.n_activated_experts,
            mesh=mesh1,
            kernel_init=nd_kernel_init,
            kernel_axes=("embed_router", None),
            intermediate_dim=args.moe_inter_dim,
            is_hash_routing=True,
            use_bias_vl=True,
            rngs=nnx.Rngs(params=0),
        )

      hash_routed = moe.RoutedMoE(
          config=cfg_text,
          num_experts=args.n_routed_experts,
          num_experts_per_tok=args.n_activated_experts,
          mesh=mesh1,
          kernel_init=nd_kernel_init,
          kernel_axes=("embed_router", None),
          intermediate_dim=args.moe_inter_dim,
          is_hash_routing=True,
          use_bias_vl=False,
          rngs=nnx.Rngs(params=0),
      )
      sample_ids_j = jnp.ones((2, 40), dtype=jnp.int32)
      hash_pos_out, _, _ = hash_routed(cand_x, input_ids=sample_ids_j, image_mask=None)
      self.assertEqual(hash_pos_out.shape, cand_x.shape)
      with self.assertRaisesRegex(ValueError, "is_hash_routing=True does not support multimodal image_mask"):
        hash_routed(cand_x, input_ids=sample_ids_j, image_mask=img_mask_j)

    with tp_ep_mesh:
      routed_mm = layer_tpep.mlp.MoeBlock_0
      mm_pos_out, _, _ = routed_mm(cand_x, image_mask=img_mask_j)
      self.assertEqual(mm_pos_out.shape, cand_x.shape)
      with mock.patch.dict(
          routed_mm.config._flat_config,  # pylint: disable=protected-access
          {"te_moe_block": True},
      ):
        with self.assertRaisesRegex(ValueError, "te_moe_block=True does not support multimodal image_mask"):
          routed_mm(cand_x, image_mask=img_mask_j)
      with mock.patch.dict(
          routed_mm.config._flat_config,  # pylint: disable=protected-access
          {"attention": "vllm_rpa"},
      ):
        with self.assertRaisesRegex(ValueError, "vllm_rpa fused_moe_matmul does not support multimodal image_mask"):
          routed_mm(cand_x, image_mask=img_mask_j)


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41MultimodalDecoderStackTest(unittest.TestCase):
  """T10: Full 14-layer mini-twin multimodal decoder stack (per-layer, unscanned/scanned/remat fwd + all-leaf grads)."""

  def test_per_layer_and_full_stack_unscanned_scanned_parity_and_negctl(self) -> None:
    grid = multimodal.MINI_GRID_PADDED
    ref_mm, batch = _build_reference_pair(grid, seed=63, dtype=torch.float32)
    ref_params = dict(ref_mm.named_parameters())
    grad_classes = multimodal.reference_gradient_classes(ref_mm, freeze_vision=False)
    self.assertEqual(len(grad_classes.compared), 494)
    self.assertEqual(len(grad_classes.padded), 2)
    self.assertEqual(len(grad_classes.discrete_only), 20)
    self.assertEqual(len(grad_classes.non_gradient), 28)
    self.assertEqual(len(grad_classes.frozen), 0)

    with torch.no_grad():
      image_mask_t = batch.token_types >= 0
      engram_mask_t = ~image_mask_t
      engram_hashes_t = ref_mm.engram_hash(batch.input_ids, 0, engram_mask_t)
      h_ref = ref_mm.merge_image_embeddings(batch.images, ref_mm.embed(batch.input_ids).clone())
      h_cur = h_ref.unsqueeze(2).repeat(1, 1, ref_mm.hc_mult, 1)
      pre_cur = ref_model.make_identity_pre_mix(h_cur, ref_mm.hc_mult)
      ref_per_layer: list[tuple[np.ndarray, np.ndarray]] = []
      for layer in ref_mm.layers:
        if layer.engram is not None:
          h_cur = layer.engram(h_cur, engram_hashes_t[:, :, layer.engram.layer_hash_index, :], engram_mask_t)
        h_cur, pre_cur = layer(h_cur, 0, pre_cur, image_mask_t)
        ref_per_layer.append((h_cur.to(torch.float64).numpy(), pre_cur.to(torch.float64).numpy()))

    targets_np = np.roll(batch.input_ids.numpy(), -1, axis=1)
    targets_t = torch.as_tensor(targets_np, dtype=torch.long)
    valid_mask_np = (batch.token_types.numpy() < 0).astype(np.float32)
    valid_mask_t = torch.as_tensor(valid_mask_np, dtype=torch.float32)

    ref_logits_t = ref_mm(
        batch.input_ids,
        images=batch.images,
        token_types=batch.token_types,
        full_logits=True,
    )
    ref_log_probs_t = F.log_softmax(ref_logits_t, dim=-1)
    ref_nll_t = -torch.gather(ref_log_probs_t, -1, targets_t.unsqueeze(-1)).squeeze(-1)
    ref_loss_t = (ref_nll_t * valid_mask_t).sum() / valid_mask_t.sum()
    raw_ref_grads = torch.autograd.grad(ref_loss_t, list(ref_params.values()), allow_unused=True)
    ref_grads = dict(zip(ref_params.keys(), raw_ref_grads))
    ref_logits_64 = ref_logits_t.detach().cpu().to(torch.float64).numpy()
    ref_loss_val = float(ref_loss_t.detach().cpu().item())

    tokens_j = jnp.asarray(batch.input_ids.numpy(), dtype=jnp.int32)
    positions_j = jnp.broadcast_to(jnp.arange(tokens_j.shape[1], dtype=jnp.int32), tokens_j.shape)
    patches_j = mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float32)
    img_mask_j = jnp.asarray(batch.image_mask.numpy(), dtype=jnp.bool_)
    targets_j = jnp.asarray(targets_np, dtype=jnp.int32)
    valid_mask_j = jnp.asarray(valid_mask_np, dtype=jnp.float32)

    cfg_base = mm_testing.mini_vision_maxtext_config(grid=grid, scan_layers=False, remat_policy="none")
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_base), cfg_base.mesh_axes)
    logits_by_mode: dict[tuple[bool, str], np.ndarray] = {}
    mt_unscanned_ref: Any = None
    mt_scanned_ref: Any = None

    # 1. Forward + full 14-layer all-parameter leaf-gradient parity across scan_layers x remat_policy.
    for scan_layers, remat_policy in ((False, "none"), (False, "full"), (True, "none"), (True, "full")):
      cfg_mode = mm_testing.mini_vision_maxtext_config(
          grid=grid,
          scan_layers=scan_layers,
          remat_policy=remat_policy,
      )
      with mesh:
        mt_mod = model_creation_utils.from_pretrained(cfg_mode, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
        mm_testing.copy_full_multimodal_model(mt_mod, ref_mm, cfg_mode, jnp.float32)

        if not scan_layers and remat_policy == "none":
          mt_unscanned_ref = mt_mod
          spans_j, _ = mt_mod.vision_encoder(patches_j, deterministic=True)
          mm_in = MultimodalInput(image_embeddings=spans_j, bidirectional_mask=img_mask_j)
          y_emb = mt_mod.decoder._apply_embedding(  # pylint: disable=protected-access
              mt_mod.token_embedder,
              tokens_j,
              positions_j,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
              multimodal_input=mm_in,
          )
          carry = deepseek41.initial_carry(y_emb, cfg_mode.mhc_expansion_rate)
          for i in range(cfg_mode.num_decoder_layers):
            layer_mod = getattr(mt_mod.decoder, f"layers_{i}")
            carry = layer_mod(carry, positions_j, tokens_j, image_mask=img_mask_j)
            ref_h_i, ref_pre_i = ref_per_layer[i]
            ok_h, m_h = layer_parity.compare(np.asarray(carry["h"]), ref_h_i, layer_parity.FP32_TOLERANCE)
            ok_p, m_p = layer_parity.compare(np.asarray(carry["pre_mix"]), ref_pre_i, layer_parity.FP32_TOLERANCE)
            self.assertTrue(ok_h, msg=f"T10 layer {i} h mismatch: {m_h}")
            self.assertTrue(ok_p, msg=f"T10 layer {i} pre_mix mismatch: {m_p}")
        elif scan_layers and remat_policy == "none":
          mt_scanned_ref = mt_mod

        logits_mode = np.asarray(
            mt_mod(
                tokens_j,
                positions_j,
                encoder_images=patches_j,
                enable_dropout=False,
                model_mode=MODEL_MODE_TRAIN,
            )
        )
        logits_by_mode[(scan_layers, remat_policy)] = logits_mode
        ok_fwd, m_fwd = layer_parity.compare(logits_mode, ref_logits_64, layer_parity.FP32_TOLERANCE)
        self.assertTrue(
            ok_fwd,
            msg=f"T10 full-stack logits mismatch (scan_layers={scan_layers}, remat_policy={remat_policy}): {m_fwd}",
        )

        m_gdef, m_pstate, m_pure, m_ostate = mm_testing.extract_pure_params(mt_mod)

        def masked_ce_loss(
            param_dict: dict[str, Any],
            m_gdef=m_gdef,
            m_pstate=m_pstate,
            m_ostate=m_ostate,
        ) -> jax.Array:
          model_inst = mm_testing.apply_with_pure_params(m_gdef, m_pstate, param_dict, m_ostate)
          logits_out = model_inst(
              tokens_j,
              positions_j,
              encoder_images=patches_j,
              enable_dropout=False,
              model_mode=MODEL_MODE_TRAIN,
          )
          log_probs = jax.nn.log_softmax(logits_out, axis=-1)
          tok_nll = -jnp.take_along_axis(log_probs, targets_j[..., None], axis=-1).squeeze(-1)
          return jnp.sum(tok_nll * valid_mask_j) / jnp.sum(valid_mask_j)

        cand_loss_j, cand_grad_pure = jax.value_and_grad(masked_ce_loss)(m_pure)
        loss_diff = abs(float(cand_loss_j) - ref_loss_val)
        self.assertLessEqual(
            loss_diff,
            layer_parity.FP32_ATOL_CEILING * max(1.0, abs(ref_loss_val)),
            msg=f"T10 scalar loss mismatch (scan={scan_layers}, remat={remat_policy}): diff={loss_diff}",
        )
        _, grad_failures, consumed_keys, class_counts = mm_testing.check_full_model_gradients(
            ref_params=ref_params,
            ref_grads=ref_grads,
            cand_grad_pure=cand_grad_pure,
            config=cfg_mode,
            grad_classes=grad_classes,
            grad_tol=layer_parity.FP32_TOLERANCE,
        )
        self.assertEqual(
            grad_failures,
            [],
            msg=f"T10 leaf-gradient failures (scan={scan_layers}, remat={remat_policy}): {grad_failures[:5]}",
        )
        self.assertEqual(class_counts["compared"], len(grad_classes.compared))
        self.assertEqual(class_counts["padded"], len(grad_classes.padded))
        self.assertEqual(class_counts["discrete_only"], len(grad_classes.discrete_only))
        self.assertEqual(class_counts["non_gradient"], len(grad_classes.non_gradient))
        self.assertEqual(len(consumed_keys), 390)

    # 2. Scan-vs-unscanned agreement: FP32_TOLERANCE + frob_rel <= 1e-6 on 4-device FSDP mesh,
    #    and exact <= 1e-6 max_abs agreement on single-device mesh under matched remat lowering.
    for remat_policy in ("none", "full"):
      ok_scan, m_scan = layer_parity.compare(
          logits_by_mode[(True, remat_policy)],
          logits_by_mode[(False, remat_policy)].astype(np.float64),
          layer_parity.FP32_TOLERANCE,
      )
      self.assertTrue(ok_scan, msg=f"T10 4-dev scan vs unscanned mismatch (remat={remat_policy}): {m_scan}")
      self.assertLessEqual(m_scan.frob_rel, 1e-6)

    mesh1 = mm_testing.single_device_mesh()
    cfg_u_1d = mm_testing.mini_vision_maxtext_config(grid=grid, scan_layers=False, remat_policy="full")
    cfg_s_1d = mm_testing.mini_vision_maxtext_config(grid=grid, scan_layers=True, remat_policy="full")
    with mesh1:
      mt_u_1d = model_creation_utils.from_pretrained(cfg_u_1d, mesh=mesh1, model_mode=MODEL_MODE_TRAIN)
      mm_testing.copy_full_multimodal_model(mt_u_1d, ref_mm, cfg_u_1d, jnp.float32)
      mt_s_1d = model_creation_utils.from_pretrained(cfg_s_1d, mesh=mesh1, model_mode=MODEL_MODE_TRAIN)
      mm_testing.copy_full_multimodal_model(mt_s_1d, ref_mm, cfg_s_1d, jnp.float32)
      unscanned_1d = np.asarray(
          mt_u_1d(tokens_j, positions_j, encoder_images=patches_j, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
      )
      scanned_1d = np.asarray(
          mt_s_1d(tokens_j, positions_j, encoder_images=patches_j, enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
      )
    self.assertLessEqual(float(np.max(np.abs(scanned_1d - unscanned_1d))), 1e-6)
    ok_1d, m_1d = layer_parity.compare(scanned_1d, unscanned_1d.astype(np.float64), layer_parity.FP32_TOLERANCE)
    self.assertTrue(ok_1d, msg=f"T10 1-dev scan vs unscanned mismatch: {m_1d}")

    # 3. Packed multi-segment + tail padding (binarized `seg_ids != 0` valid mask) + negative controls.
    with mesh:
      seg_ids = jnp.asarray(
          [[1] * 16 + [2] * 19 + [0] * 5, [1] * 18 + [2] * 17 + [0] * 5],
          dtype=jnp.int32,
      )
      seg_pos = jnp.asarray(
          [list(range(16)) + list(range(19)) + [0] * 5, list(range(18)) + list(range(17)) + [0] * 5],
          dtype=jnp.int32,
      )
      unscanned_seg_logits = np.asarray(
          mt_unscanned_ref(
              tokens_j,
              seg_pos,
              decoder_segment_ids=seg_ids,
              encoder_images=patches_j,
              enable_dropout=False,
              model_mode=MODEL_MODE_TRAIN,
          )
      )
      scanned_seg_logits = np.asarray(
          mt_scanned_ref(
              tokens_j,
              seg_pos,
              decoder_segment_ids=seg_ids,
              encoder_images=patches_j,
              enable_dropout=False,
              model_mode=MODEL_MODE_TRAIN,
          )
      )
      ok_seg, m_seg = layer_parity.compare(
          scanned_seg_logits,
          unscanned_seg_logits.astype(np.float64),
          layer_parity.FP32_TOLERANCE,
      )
      self.assertTrue(ok_seg, msg=f"T10 packed multi-segment scanned vs unscanned mismatch: {m_seg}")

      tokens_mut = tokens_j.at[0, 30:35].set(42)
      patches_mut = patches_j.at[0, 1].set(-patches_j[0, 1])
      scanned_seg_mut = np.asarray(
          mt_scanned_ref(
              tokens_mut,
              seg_pos,
              decoder_segment_ids=seg_ids,
              encoder_images=patches_mut,
              enable_dropout=False,
              model_mode=MODEL_MODE_TRAIN,
          )
      )
      self.assertTrue(np.array_equal(scanned_seg_logits[0, :16], scanned_seg_mut[0, :16]))
      self.assertFalse(np.array_equal(scanned_seg_logits[0, 16:35], scanned_seg_mut[0, 16:35]))

      # Binarized `seg_ids != 0` gradient check: padded tail tokens (positions 35:40) contribute 0 gradient.
      binarized_valid_j = ((seg_ids != 0) & ~img_mask_j).astype(jnp.float32)
      s_gdef, s_pstate, s_pure, s_ostate = mm_testing.extract_pure_params(mt_scanned_ref)

      def packed_loss_for_tokens(tok_in: jax.Array, param_dict: dict[str, Any]) -> jax.Array:
        model_inst = mm_testing.apply_with_pure_params(s_gdef, s_pstate, param_dict, s_ostate)
        lg = model_inst(
            tok_in,
            seg_pos,
            decoder_segment_ids=seg_ids,
            encoder_images=patches_j,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )
        lp = jax.nn.log_softmax(lg, axis=-1)
        nll = -jnp.take_along_axis(lp, targets_j[..., None], axis=-1).squeeze(-1)
        return jnp.sum(nll * binarized_valid_j) / jnp.sum(binarized_valid_j)

      loss_pad_a, grad_pad_a = jax.value_and_grad(functools.partial(packed_loss_for_tokens, tokens_j))(s_pure)
      tokens_pad_mut = tokens_j.at[:, 35:40].set(99)
      loss_pad_b, grad_pad_b = jax.value_and_grad(functools.partial(packed_loss_for_tokens, tokens_pad_mut))(s_pure)
      self.assertEqual(float(loss_pad_a), float(loss_pad_b))
      self.assertTrue(
          np.array_equal(
              np.asarray(grad_pad_a["token_embedder"]["embedding"]),
              np.asarray(grad_pad_b["token_embedder"]["embedding"]),
          )
      )

      # Negative controls on `vision_encoder` (`DeepSeek41VisionProjector_0/w2/kernel`),
      # unscanned decoder (`layers_2/mlp/MoeBlock_0/gate/kernel`), and scanned decoder (`segments_2/...`).
      u_gdef, u_pstate, u_pure, u_ostate = mm_testing.extract_pure_params(mt_unscanned_ref)

      def cand_full_unscanned_fn(model_params: dict[str, Any], patches_in: jax.Array) -> jax.Array:
        model_inst = mm_testing.apply_with_pure_params(u_gdef, u_pstate, model_params, u_ostate)
        return model_inst(
            tokens_j,
            positions_j,
            encoder_images=patches_in,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )

      for neg_seed, neg_path in (
          (1000, "vision_encoder/DeepSeek41VisionProjector_0/w2/kernel"),
          (1001, "decoder/layers_2/mlp/MoeBlock_0/gate/kernel"),
      ):
        fired_u, neg_u_m = layer_parity.negative_control_fires(
            ref_out=ref_logits_64,
            cand_fn=cand_full_unscanned_fn,
            cand_params=u_pure,
            cand_input=patches_j,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path=neg_path,
            rel_scale=_NEGCTL_REL_SCALE,
            seed=neg_seed,
        )
        self.assertTrue(fired_u, msg=f"T10 unscanned negative control on {neg_path} did not fire: {neg_u_m}")

      def cand_full_scanned_fn(model_params: dict[str, Any], patches_in: jax.Array) -> jax.Array:
        model_inst = mm_testing.apply_with_pure_params(s_gdef, s_pstate, model_params, s_ostate)
        return model_inst(
            tokens_j,
            positions_j,
            encoder_images=patches_in,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )

      fired_s, neg_s_m = layer_parity.negative_control_fires(
          ref_out=ref_logits_64,
          cand_fn=cand_full_scanned_fn,
          cand_params=s_pure,
          cand_input=patches_j,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          target_path="decoder/segments_2/layers_0/mlp/MoeBlock_0/gate/kernel",
          rel_scale=_NEGCTL_REL_SCALE,
          seed=1002,
      )
      self.assertTrue(fired_s, msg=f"T10 scanned negative control did not fire: {neg_s_m}")

  def test_use_multimodal_false_text_bit_identity(self) -> None:
    ref_mm, _ = _build_reference_pair(multimodal.MINI_GRID_PADDED, seed=71, dtype=torch.float32)
    rng = np.random.default_rng(710)
    text_tokens_np = rng.integers(3, 1000, size=(2, multimodal.MINI_SEQ_LEN), dtype=np.int32)
    tokens_j = jnp.asarray(text_tokens_np, dtype=jnp.int32)
    positions_j = jnp.broadcast_to(jnp.arange(tokens_j.shape[1], dtype=jnp.int32), tokens_j.shape)

    for scan_layers in (False, True):
      cfg_off = mm_testing.mini_vision_maxtext_config(use_multimodal=False, scan_layers=scan_layers)
      cfg_on = mm_testing.mini_vision_maxtext_config(use_multimodal=True, scan_layers=scan_layers)
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_off), cfg_off.mesh_axes)
      with mesh:
        mt_off = model_creation_utils.from_pretrained(cfg_off, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
        mt_on = model_creation_utils.from_pretrained(cfg_on, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
        mm_testing.copy_full_multimodal_model(mt_off, ref_mm, cfg_off, jnp.float32)
        mm_testing.copy_full_multimodal_model(mt_on, ref_mm, cfg_on, jnp.float32)

        logits_off = mt_off(
            tokens_j,
            positions_j,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )
        logits_on_text_only = mt_on(
            tokens_j,
            positions_j,
            encoder_images=None,
            enable_dropout=False,
            model_mode=MODEL_MODE_TRAIN,
        )
        self.assertTrue(
            bool(jnp.array_equal(logits_off, logits_on_text_only)),
            msg=f"T10 text-only bit-identity failed for scan_layers={scan_layers}",
        )


if __name__ == "__main__":
  unittest.main()
