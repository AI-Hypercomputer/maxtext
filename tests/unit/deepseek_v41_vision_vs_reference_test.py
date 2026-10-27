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

"""Forward, all-leaf gradient, fp64, negative-control, and init/config tests for DeepSeek-V4.1 vision (U3: T1-T5b)."""

from __future__ import annotations

import functools
import sys
from typing import Any
import unittest

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoTokenizer

from maxtext.common.common_types import MODEL_MODE_TRAIN, VisionEncoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers import encoders
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.models import deepseek41_vision
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.parity import layer_parity

torch: Any
nn: Any
functional_call: Any
ref_image_processor: Any
multimodal: Any
ref_vision: Any
try:
  import torch  # pylint: disable=g-import-not-at-top
  from torch import nn  # pylint: disable=g-import-not-at-top
  from torch.func import functional_call  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import image_processor as ref_image_processor  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import multimodal  # pylint: disable=g-import-not-at-top
  from tests.utils.deepseek_v41_reference import vision as ref_vision  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
except ImportError:
  torch = None
  nn = None
  functional_call = None
  ref_image_processor = None
  multimodal = None
  ref_vision = None
  HAS_TORCH = False

_TorchModuleBase: Any = nn.Module if HAS_TORCH else object
_NEGCTL_REL_SCALE = 1e-2


def setUpModule() -> None:
  jax.config.update("jax_default_matmul_precision", "highest")


class _BatchedPatchEmbedAdapter(_TorchModuleBase):
  """Applies `vision.PatchEmbed` across `[B, N, P, 3*p*p]` inputs for `functional_call`."""

  def __init__(self, patch_embed: ref_vision.PatchEmbed) -> None:
    super().__init__()
    self.proj = patch_embed.proj

  def forward(self, patches_bnpd: torch.Tensor) -> torch.Tensor:
    b, n, p, d = patches_bnpd.shape
    out = self.proj(patches_bnpd.reshape(b * n, p, d).reshape(b * n * p, d))
    return out.reshape(b, n, p, -1)


class _BatchedBlockAdapter(_TorchModuleBase):
  """Applies one `vision.Block` across `[B*N, P, C]` inputs for `functional_call`."""

  def __init__(self, block: ref_vision.Block, cos: torch.Tensor, sin: torch.Tensor) -> None:
    super().__init__()
    self.block = block
    self._cos = cos
    self._sin = sin

  def forward(self, x_bpc: torch.Tensor) -> torch.Tensor:
    cos = self._cos.to(x_bpc.dtype)
    sin = self._sin.to(x_bpc.dtype)
    return torch.stack([self.block(img, cos, sin) for img in x_bpc], dim=0)


class _BatchedViTAdapter(_TorchModuleBase):
  """Applies `vision.ViT` across `[B, N, P, 3*p*p]` inputs for `functional_call`."""

  def __init__(self, vit: ref_vision.ViT, n_vit_h: int, n_vit_w: int) -> None:
    super().__init__()
    self.vit = vit
    self.n_vit_h = n_vit_h
    self.n_vit_w = n_vit_w

  def forward(self, patches_bnpd: torch.Tensor) -> torch.Tensor:
    b, n, p, _ = patches_bnpd.shape
    per_img = [self.vit(patches_bnpd[bi, ni], self.n_vit_h, self.n_vit_w) for bi in range(b) for ni in range(n)]
    return torch.stack(per_img, dim=0).reshape(b, n, p, -1)


class _BatchedAlignerAdapter(_TorchModuleBase):
  """Applies `vision.Aligner` across `[B, N, P, C]` inputs for `functional_call`."""

  def __init__(self, aligner: ref_vision.Aligner, n_vit_h: int, n_vit_w: int) -> None:
    super().__init__()
    self.aligner = aligner
    self.n_vit_h = n_vit_h
    self.n_vit_w = n_vit_w

  def forward(self, vit_bnpc: torch.Tensor) -> torch.Tensor:
    b, n, _, _ = vit_bnpc.shape
    per_img = [self.aligner(vit_bnpc[bi, ni], self.n_vit_h, self.n_vit_w) for bi in range(b) for ni in range(n)]
    return torch.stack(per_img, dim=0).reshape(b, n, -1, per_img[0].shape[-1])


class _SpanMergeAdapter(_TorchModuleBase):
  """Invokes `MultimodalTransformer.merge_image_embeddings` on zero `h` for `functional_call`."""

  def __init__(
      self,
      ref_mm: multimodal.MultimodalTransformer,
      n_vit_h: int,
      n_vit_w: int,
      *,
      include_vit: bool,
  ) -> None:
    super().__init__()
    if ref_mm.vision is None or ref_mm.aligner is None:
      raise ValueError("Reference MultimodalTransformer has no vision modules.")
    self.include_vit = include_vit
    if include_vit:
      self.vision = ref_mm.vision
    self.aligner = ref_mm.aligner
    self.image_start = ref_mm.image_start
    self.image_end = ref_mm.image_end
    self.image_newline = ref_mm.image_newline
    self.n_vit_h = n_vit_h
    self.n_vit_w = n_vit_w
    r = ref_mm.aligner.downsample_ratio
    self.n_llm_h = (n_vit_h + (-n_vit_h) % r) // r
    self.n_llm_w = (n_vit_w + (-n_vit_w) % r) // r
    self._types = ref_image_processor.image_token_types(self.n_llm_h, self.n_llm_w)

  def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
    """Encodes one image patch grid into `[N_llm, D]` aligned embeddings."""
    feats = self.vision(patches, n_vit_h, n_vit_w) if self.include_vit else patches
    return self.aligner(feats, n_vit_h, n_vit_w)

  def forward(self, inputs_bnpd: torch.Tensor) -> torch.Tensor:
    """Assembles `[B, N, K, D]` image spans via `MultimodalTransformer.merge_image_embeddings`."""
    batch_size, num_images = inputs_bnpd.shape[:2]
    span_len = int(self._types.numel())
    images: list[list[ref_image_processor.ImageInput]] = []
    for bi in range(batch_size):
      sample: list[ref_image_processor.ImageInput] = []
      for ni in range(num_images):
        sample.append(
            ref_image_processor.ImageInput(
                start=ni * span_len,
                patches=inputs_bnpd[bi, ni],
                n_vit_h=self.n_vit_h,
                n_vit_w=self.n_vit_w,
                types=self._types,
            )
        )
      images.append(sample)
    zero_h = torch.zeros(
        (batch_size, num_images * span_len, self.image_start.shape[0]),
        dtype=inputs_bnpd.dtype,
    )
    merged = multimodal.MultimodalTransformer.merge_image_embeddings(self, images, zero_h)  # type: ignore[arg-type]
    return merged.reshape(batch_size, num_images, span_len, -1)


@functools.lru_cache(maxsize=1)
def _get_tokenizer() -> Any:
  """Loads and caches the DeepSeek-V4.1 tokenizer once per test process."""
  return AutoTokenizer.from_pretrained(mm_testing.TOKENIZER_PATH)


def _build_reference_pair(
    grid: tuple[int, int],
    *,
    seed: int,
    dtype: Any = None,
) -> tuple[multimodal.MultimodalTransformer, multimodal.MiniVLBatch]:
  """Builds a deterministic mini reference transformer and synthetic VL batch."""
  if dtype is None:
    dtype = torch.float32
  args = multimodal.mini_model_args()
  ref_mm = multimodal.build_multimodal_model(args, tokenizer=_get_tokenizer())
  multimodal.init_mini_reference(ref_mm, seed=seed)
  if dtype == torch.float64:
    ref_mm = ref_mm.to(torch.float64)
  batch = multimodal.build_mini_vl_batch(args, grid=grid, seed=seed + 1000, dtype=dtype)
  return ref_mm, batch


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41PatchEmbedParityTest(unittest.TestCase):
  """T1: PatchEmbed forward, all-leaf + input gradients, fp64, int32 cast, and negative control."""

  def test_patch_embed_fp32_and_fp64_parity_and_negative_control(self) -> None:
    grid = multimodal.MINI_GRID_PADDED
    ref_mm, batch = _build_reference_pair(grid, seed=11, dtype=torch.float32)
    assert ref_mm.vision is not None
    adapter = _BatchedPatchEmbedAdapter(ref_mm.vision.patch_embed)
    ref_params = dict(adapter.named_parameters())
    grad_classes = layer_parity.GradientClasses(compared=frozenset(ref_params))
    weight_map = {
        "proj.weight": "patch_embed/kernel:T",
        "proj.bias": "patch_embed/bias",
    }

    cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
    mesh = mm_testing.single_device_mesh()
    with mesh:
      mt_enc = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg, rngs=nnx.Rngs(params=0))
      mm_testing.copy_vision_encoder(mt_enc, ref_mm.vision, jnp.float32)
      pe_gdef, pe_pstate, pe_pure, pe_ostate = mm_testing.extract_pure_params(mt_enc.patch_embed)
      cand_params = {"patch_embed": pe_pure}
      cand_input = mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float32)

      def cand_fn(params: dict[str, Any], x: jax.Array) -> jax.Array:
        pe = mm_testing.apply_with_pure_params(pe_gdef, pe_pstate, params["patch_embed"], pe_ostate)
        return pe(x.astype(cfg.dtype_mm))

      def ref_fn(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        return functional_call(adapter, params, (x,))

      np.testing.assert_array_equal(
          np.asarray(mt_enc.embed_patches(cand_input)),
          np.asarray(cand_fn(cand_params, cand_input)),
      )
      report32 = layer_parity.check_parity(
          ref_fn=ref_fn,
          ref_params=ref_params,
          ref_input=batch.patches_bnpd,
          cand_fn=cand_fn,
          cand_params=cand_params,
          cand_input=cand_input,
          weight_map=weight_map,
          grad_classes=grad_classes,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=1001,
      )
      self.assertTrue(report32.passed, msg=f"T1 fp32 failures: {report32.failures}")
      in_ok32, in_metrics32 = mm_testing.check_input_gradient(
          ref_fn=lambda x: ref_fn(ref_params, x),
          ref_input=batch.patches_bnpd,
          cand_fn=lambda x: cand_fn(cand_params, x),
          cand_input=cand_input,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=1001,
      )
      self.assertTrue(in_ok32, msg=f"T1 fp32 input grad failed: {in_metrics32}")

      fired, neg_metrics = layer_parity.negative_control_fires(
          ref_out=report32.ref_out,
          cand_fn=cand_fn,
          cand_params=cand_params,
          cand_input=cand_input,
          fwd_tol=layer_parity.FP32_TOLERANCE,
          target_path="patch_embed/kernel",
          rel_scale=_NEGCTL_REL_SCALE,
          seed=101,
      )
      self.assertTrue(fired, msg=f"T1 negative control did not fire: {neg_metrics}")

      with multimodal.fp64_reference_context():
        floor = layer_parity.reference_noise_floor(ref_fn, ref_params, batch.patches_bnpd)
        self.assertTrue(floor.finite)
        self.assertLessEqual(floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

        ref_mm64, batch64 = _build_reference_pair(grid, seed=11, dtype=torch.float64)
        assert ref_mm64.vision is not None
        adapter64 = _BatchedPatchEmbedAdapter(ref_mm64.vision.patch_embed)
        ref_params64 = dict(adapter64.named_parameters())
        with mm_testing.fp64_maxtext_context(cfg):
          mt_enc64 = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg, rngs=nnx.Rngs(params=0))
          mm_testing.copy_vision_encoder(mt_enc64, ref_mm64.vision, jnp.float64)
          pe64_gdef, pe64_pstate, pe64_pure, pe64_ostate = mm_testing.extract_pure_params(mt_enc64.patch_embed)
          cand_params64 = {"patch_embed": pe64_pure}
          cand_input64 = mm_testing.torch_to_jax(batch64.patches_bnpd, jnp.float64)

          def cand_fn64(params: dict[str, Any], x: jax.Array) -> jax.Array:
            pe64 = mm_testing.apply_with_pure_params(pe64_gdef, pe64_pstate, params["patch_embed"], pe64_ostate)
            return pe64(x.astype(cfg.dtype_mm))

          def ref_fn64(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            return functional_call(adapter64, params, (x,))

          report64 = layer_parity.check_parity(
              ref_fn=ref_fn64,
              ref_params=ref_params64,
              ref_input=batch64.patches_bnpd,
              cand_fn=cand_fn64,
              cand_params=cand_params64,
              cand_input=cand_input64,
              weight_map=weight_map,
              grad_classes=grad_classes,
              fwd_tol=layer_parity.FP64_TOLERANCE,
              grad_tol=layer_parity.FP64_TOLERANCE,
              cotangent_seed=1001,
          )
          self.assertTrue(report64.passed, msg=f"T1 fp64 failures: {report64.failures}")
          in_ok64, in_metrics64 = mm_testing.check_input_gradient(
              ref_fn=lambda x: ref_fn64(ref_params64, x),
              ref_input=batch64.patches_bnpd,
              cand_fn=lambda x: cand_fn64(cand_params64, x),
              cand_input=cand_input64,
              grad_tol=layer_parity.FP64_TOLERANCE,
              cotangent_seed=1001,
          )
          self.assertTrue(in_ok64, msg=f"T1 fp64 input grad failed: {in_metrics64}")

  def test_patch_embed_casts_int32_patches_to_dtype_mm(self) -> None:
    grid = multimodal.MINI_GRID_PADDED
    ref_mm, _ = _build_reference_pair(grid, seed=12, dtype=torch.float32)
    assert ref_mm.vision is not None
    raw_int = np.random.default_rng(77).integers(-2, 3, size=(2, 1, grid[0] * grid[1], 588), dtype=np.int32)
    int_patches_jax = jnp.asarray(raw_int, dtype=jnp.int32)
    float_patches_pt = torch.as_tensor(raw_int.astype(np.float32))
    adapter = _BatchedPatchEmbedAdapter(ref_mm.vision.patch_embed)
    with torch.no_grad():
      ref_out = adapter(float_patches_pt).numpy()

    mesh = mm_testing.single_device_mesh()
    for dtype_mm_name, expected_dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
      cfg = mm_testing.mini_vision_maxtext_config(grid=grid, dtype_mm=dtype_mm_name)
      with mesh:
        mt_enc = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg, rngs=nnx.Rngs(params=0))
        mm_testing.copy_vision_encoder(mt_enc, ref_mm.vision, jnp.float32)
        out = jax.jit(mt_enc.embed_patches)(int_patches_jax)
        self.assertEqual(out.dtype, expected_dtype)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
        if expected_dtype == jnp.float32:
          ok, metrics = layer_parity.compare(np.asarray(out), ref_out, layer_parity.FP32_TOLERANCE)
          self.assertTrue(ok, msg=f"int32->float32 PatchEmbed mismatch: {metrics}")


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41VisionRotaryParityTest(unittest.TestCase):
  """T2a: 2D vision RoPE table, forward + input gradient parity, fp64, and swapped-halves control."""

  def test_vision_cos_sin_and_apply_rotary_fp32_and_fp64(self) -> None:
    head_dim = 64
    rope_dim = head_dim // 2
    theta = 10000.0
    for grid_idx, (n_h, n_w) in enumerate((multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT)):
      num_patches = n_h * n_w
      rng = np.random.default_rng(200 + grid_idx)
      x_np = rng.standard_normal((2, num_patches, 2, head_dim))

      ref_vision.get_vision_cos_sin.cache_clear()
      ref_cos32, ref_sin32 = ref_vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
      cand_cos32, cand_sin32 = deepseek41_vision.vision_cos_sin(n_h, n_w, rope_dim, theta, dtype=jnp.float32)
      ok_c32, m_c32 = layer_parity.compare(np.asarray(cand_cos32), ref_cos32.numpy(), layer_parity.FP32_TOLERANCE)
      ok_s32, m_s32 = layer_parity.compare(np.asarray(cand_sin32), ref_sin32.numpy(), layer_parity.FP32_TOLERANCE)
      self.assertTrue(ok_c32, msg=f"cos fp32 {n_h}x{n_w}: {m_c32}")
      self.assertTrue(ok_s32, msg=f"sin fp32 {n_h}x{n_w}: {m_s32}")

      x_pt32 = torch.tensor(x_np, dtype=torch.float32)
      x_jax32 = jnp.asarray(x_np.astype(np.float32), dtype=jnp.float32)
      ref_rot32 = ref_vision.apply_rotary(x_pt32, ref_cos32, ref_sin32)
      cand_rot32 = deepseek41_vision.apply_vision_rotary(x_jax32, cand_cos32, cand_sin32)
      ok_r32, m_r32 = layer_parity.compare(np.asarray(cand_rot32), ref_rot32.numpy(), layer_parity.FP32_TOLERANCE)
      self.assertTrue(ok_r32, msg=f"apply_vision_rotary fp32 {n_h}x{n_w}: {m_r32}")
      in_ok32, in_m32 = mm_testing.check_input_gradient(
          ref_fn=lambda x, c=ref_cos32, s=ref_sin32: ref_vision.apply_rotary(x, c, s),
          ref_input=x_pt32,
          cand_fn=lambda x, c=cand_cos32, s=cand_sin32: deepseek41_vision.apply_vision_rotary(x, c, s),
          cand_input=x_jax32,
          grad_tol=layer_parity.FP32_TOLERANCE,
          cotangent_seed=2100 + grid_idx,
      )
      self.assertTrue(in_ok32, msg=f"apply_vision_rotary fp32 input grad {n_h}x{n_w}: {in_m32}")

      with multimodal.fp64_reference_context(), jax.enable_x64(True):
        floor = layer_parity.reference_noise_floor(
            lambda _p, x, nh=n_h, nw=n_w: ref_vision.apply_rotary(
                x, *ref_vision.get_vision_cos_sin(nh, nw, rope_dim, theta)
            ),
            {},
            x_pt32,
        )
        self.assertTrue(floor.finite)
        self.assertLessEqual(floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

        ref_cos64, ref_sin64 = ref_vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
        cand_cos64, cand_sin64 = deepseek41_vision.vision_cos_sin(n_h, n_w, rope_dim, theta, dtype=jnp.float64)
        ok_c64, m_c64 = layer_parity.compare(np.asarray(cand_cos64), ref_cos64.numpy(), layer_parity.FP64_TOLERANCE)
        ok_s64, m_s64 = layer_parity.compare(np.asarray(cand_sin64), ref_sin64.numpy(), layer_parity.FP64_TOLERANCE)
        self.assertTrue(ok_c64, msg=f"cos fp64 {n_h}x{n_w}: {m_c64}")
        self.assertTrue(ok_s64, msg=f"sin fp64 {n_h}x{n_w}: {m_s64}")

        x_pt64 = torch.tensor(x_np, dtype=torch.float64)
        x_jax64 = jnp.asarray(x_np, dtype=jnp.float64)
        ref_rot64 = ref_vision.apply_rotary(x_pt64, ref_cos64, ref_sin64)
        cand_rot64 = deepseek41_vision.apply_vision_rotary(x_jax64, cand_cos64, cand_sin64)
        ok_r64, m_r64 = layer_parity.compare(np.asarray(cand_rot64), ref_rot64.numpy(), layer_parity.FP64_TOLERANCE)
        self.assertTrue(ok_r64, msg=f"apply_vision_rotary fp64 {n_h}x{n_w}: {m_r64}")
        in_ok64, in_m64 = mm_testing.check_input_gradient(
            ref_fn=lambda x, c=ref_cos64, s=ref_sin64: ref_vision.apply_rotary(x, c, s),
            ref_input=x_pt64,
            cand_fn=lambda x, c=cand_cos64, s=cand_sin64: deepseek41_vision.apply_vision_rotary(x, c, s),
            cand_input=x_jax64,
            grad_tol=layer_parity.FP64_TOLERANCE,
            cotangent_seed=2100 + grid_idx,
        )
        self.assertTrue(in_ok64, msg=f"apply_vision_rotary fp64 input grad {n_h}x{n_w}: {in_m64}")

  def test_vision_rotary_swapped_row_col_halves_control_fails(self) -> None:
    head_dim = 64
    rope_dim = head_dim // 2
    half = rope_dim // 2
    theta = 10000.0
    for grid_idx, (n_h, n_w) in enumerate((multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT)):
      x_np = np.random.default_rng(250 + grid_idx).standard_normal((2, n_h * n_w, 2, head_dim)).astype(np.float32)
      ref_vision.get_vision_cos_sin.cache_clear()
      ref_cos, ref_sin = ref_vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
      ref_out = ref_vision.apply_rotary(torch.tensor(x_np), ref_cos, ref_sin).numpy()

      cand_cos, cand_sin = deepseek41_vision.vision_cos_sin(n_h, n_w, rope_dim, theta, dtype=jnp.float32)
      x_jax = jnp.asarray(x_np)
      pos_ok, _ = layer_parity.compare(
          np.asarray(deepseek41_vision.apply_vision_rotary(x_jax, cand_cos, cand_sin)),
          ref_out,
          layer_parity.FP32_TOLERANCE,
      )
      self.assertTrue(pos_ok)

      # Swap the [row, col] halves of the 2D RoPE frequency table.
      swapped_cos = jnp.concatenate([cand_cos[..., half:], cand_cos[..., :half]], axis=-1)
      swapped_sin = jnp.concatenate([cand_sin[..., half:], cand_sin[..., :half]], axis=-1)
      neg_out = deepseek41_vision.apply_vision_rotary(x_jax, swapped_cos, swapped_sin)
      neg_ok, neg_metrics = layer_parity.compare(np.asarray(neg_out), ref_out, layer_parity.FP32_TOLERANCE)
      self.assertFalse(neg_ok, msg=f"Swapped [row, col] halves unexpectedly passed on {n_h}x{n_w}")
      self.assertTrue(neg_metrics.finite)


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41VisionBlockParityTest(unittest.TestCase):
  """T2: Single ViT block (2D wqkv + 2D RoPE + SwiGLU MLP) on 7x5 and 6x9 grids, fp32 + fp64."""

  def test_vision_block_fp32_and_fp64_both_grids(self) -> None:
    weight_map = mm_testing.block_weight_map("blocks_0")
    for grid_idx, grid in enumerate((multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT)):
      n_h, n_w = grid
      num_patches = n_h * n_w
      ref_mm, _ = _build_reference_pair(grid, seed=21 + grid_idx, dtype=torch.float32)
      assert ref_mm.vision is not None
      rope_dim = ref_mm.vision.rope_dim
      theta = ref_mm.vision.rope_theta
      ref_vision.get_vision_cos_sin.cache_clear()
      ref_cos32, ref_sin32 = ref_vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
      adapter = _BatchedBlockAdapter(ref_mm.vision.blocks[0], ref_cos32, ref_sin32)
      ref_params = {k.removeprefix("block."): v for k, v in adapter.named_parameters()}
      grad_classes = layer_parity.GradientClasses(compared=frozenset(ref_params))

      x_np = np.random.default_rng(310 + grid_idx).standard_normal((2, num_patches, 128))
      x_pt32 = torch.tensor(x_np, dtype=torch.float32)

      cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
      mesh = mm_testing.single_device_mesh()
      with mesh:
        mt_block = deepseek41_vision.DeepSeek41VisionBlock(config=cfg, rngs=nnx.Rngs(params=0))
        mm_testing.copy_vision_block(mt_block, ref_mm.vision.blocks[0], jnp.float32)
        b_gdef, b_pstate, b_pure, b_ostate = mm_testing.extract_pure_params(mt_block)
        cand_params = {"blocks_0": b_pure}
        cand_input = jnp.asarray(x_np.astype(np.float32), dtype=jnp.float32)
        cand_cos32, cand_sin32 = deepseek41_vision.vision_cos_sin(n_h, n_w, rope_dim, theta, dtype=jnp.float32)

        def cand_fn(
            params: dict[str, Any],
            x: jax.Array,
            gdef: Any = b_gdef,
            pstate: Any = b_pstate,
            ostate: Any = b_ostate,
            c: jax.Array = cand_cos32,
            s: jax.Array = cand_sin32,
        ) -> jax.Array:
          blk = mm_testing.apply_with_pure_params(gdef, pstate, params["blocks_0"], ostate)
          return blk(x, c, s)

        def ref_fn(
            params: dict[str, torch.Tensor],
            x: torch.Tensor,
            adp: _BatchedBlockAdapter = adapter,
        ) -> torch.Tensor:
          return functional_call(adp, {f"block.{k}": v for k, v in params.items()}, (x,))

        report32 = layer_parity.check_parity(
            ref_fn=ref_fn,
            ref_params=ref_params,
            ref_input=x_pt32,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=2200 + grid_idx,
        )
        self.assertTrue(report32.passed, msg=f"T2 fp32 {grid} failures: {report32.failures}")
        in_ok32, in_m32 = mm_testing.check_input_gradient(
            ref_fn=lambda x, rf=ref_fn, rp=ref_params: rf(rp, x),
            ref_input=x_pt32,
            cand_fn=lambda x, cf=cand_fn, cp=cand_params: cf(cp, x),
            cand_input=cand_input,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=2200 + grid_idx,
        )
        self.assertTrue(in_ok32, msg=f"T2 fp32 input grad {grid}: {in_m32}")

        fired, neg_metrics = layer_parity.negative_control_fires(
            ref_out=report32.ref_out,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="blocks_0/attn/wqkv/kernel",
            rel_scale=_NEGCTL_REL_SCALE,
            seed=202 + grid_idx,
        )
        self.assertTrue(fired, msg=f"T2 negative control {grid} did not fire: {neg_metrics}")

        with multimodal.fp64_reference_context():
          ref_mm64, _ = _build_reference_pair(grid, seed=21 + grid_idx, dtype=torch.float64)
          assert ref_mm64.vision is not None
          ref_cos64, ref_sin64 = ref_vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
          adapter64 = _BatchedBlockAdapter(ref_mm64.vision.blocks[0], ref_cos64, ref_sin64)
          ref_params64 = {k.removeprefix("block."): v for k, v in adapter64.named_parameters()}
          x_pt64 = torch.tensor(x_np, dtype=torch.float64)

          def ref_fn64(
              params: dict[str, torch.Tensor],
              x: torch.Tensor,
              nh: int = n_h,
              nw: int = n_w,
              r_dim: int = rope_dim,
              r_theta: float = theta,
              blk64: ref_vision.Block = ref_mm64.vision.blocks[0],
          ) -> torch.Tensor:
            c_dyn, s_dyn = ref_vision.get_vision_cos_sin(nh, nw, r_dim, r_theta, dtype=x.dtype)
            return torch.stack(
                [functional_call(blk64, params, (img, c_dyn, s_dyn)) for img in x],
                dim=0,
            )

          floor = layer_parity.reference_noise_floor(ref_fn64, ref_params64, x_pt64)
          self.assertTrue(floor.finite)
          self.assertLessEqual(floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

          with mm_testing.fp64_maxtext_context(cfg):
            mt_block64 = deepseek41_vision.DeepSeek41VisionBlock(config=cfg, rngs=nnx.Rngs(params=0))
            mm_testing.copy_vision_block(mt_block64, ref_mm64.vision.blocks[0], jnp.float64)
            b64_gdef, b64_pstate, b64_pure, b64_ostate = mm_testing.extract_pure_params(mt_block64)
            cand_params64 = {"blocks_0": b64_pure}
            cand_input64 = jnp.asarray(x_np, dtype=jnp.float64)
            cand_cos64, cand_sin64 = deepseek41_vision.vision_cos_sin(n_h, n_w, rope_dim, theta, dtype=jnp.float64)

            def cand_fn64(
                params: dict[str, Any],
                x: jax.Array,
                gdef: Any = b64_gdef,
                pstate: Any = b64_pstate,
                ostate: Any = b64_ostate,
                c: jax.Array = cand_cos64,
                s: jax.Array = cand_sin64,
            ) -> jax.Array:
              blk = mm_testing.apply_with_pure_params(gdef, pstate, params["blocks_0"], ostate)
              return blk(x, c, s)

            report64 = layer_parity.check_parity(
                ref_fn=ref_fn64,
                ref_params=ref_params64,
                ref_input=x_pt64,
                cand_fn=cand_fn64,
                cand_params=cand_params64,
                cand_input=cand_input64,
                weight_map=weight_map,
                grad_classes=grad_classes,
                fwd_tol=layer_parity.FP64_TOLERANCE,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=2200 + grid_idx,
            )
            self.assertTrue(report64.passed, msg=f"T2 fp64 {grid} failures: {report64.failures}")
            in_ok64, in_m64 = mm_testing.check_input_gradient(
                ref_fn=lambda x, rf=ref_fn64, rp=ref_params64: rf(rp, x),
                ref_input=x_pt64,
                cand_fn=lambda x, cf=cand_fn64, cp=cand_params64: cf(cp, x),
                cand_input=cand_input64,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=2200 + grid_idx,
            )
            self.assertTrue(in_ok64, msg=f"T2 fp64 input grad {grid}: {in_m64}")


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41VisionEncoderParityTest(unittest.TestCase):
  """T3: Full 2-block DeepSeek41VisionEncoder `[B,N,P,588] -> [B,N,P,128]`, fp32 + fp64."""

  def test_full_vision_encoder_fp32_and_fp64_both_grids(self) -> None:
    for grid_idx, grid in enumerate((multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT)):
      n_h, n_w = grid
      ref_mm, batch = _build_reference_pair(grid, seed=31 + grid_idx, dtype=torch.float32)
      assert ref_mm.vision is not None
      adapter = _BatchedViTAdapter(ref_mm.vision, n_h, n_w)
      ref_params = {k.removeprefix("vit."): v for k, v in adapter.named_parameters()}
      grad_classes = layer_parity.GradientClasses(compared=frozenset(ref_params))

      cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
      weight_map = mm_testing.vit_weight_map(cfg.num_hidden_layers_for_vit)
      mesh = mm_testing.single_device_mesh()
      with mesh:
        mt_enc = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg, rngs=nnx.Rngs(params=0))
        mm_testing.copy_vision_encoder(mt_enc, ref_mm.vision, jnp.float32)
        e_gdef, e_pstate, cand_params, e_ostate = mm_testing.extract_pure_params(mt_enc)
        cand_input = mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float32)

        def cand_fn(
            params: dict[str, Any],
            x: jax.Array,
            gdef: Any = e_gdef,
            pstate: Any = e_pstate,
            ostate: Any = e_ostate,
        ) -> jax.Array:
          enc = mm_testing.apply_with_pure_params(gdef, pstate, params, ostate)
          return enc(x, deterministic=True)

        def ref_fn(
            params: dict[str, torch.Tensor],
            x: torch.Tensor,
            adp: _BatchedViTAdapter = adapter,
        ) -> torch.Tensor:
          return functional_call(adp, {f"vit.{k}": v for k, v in params.items()}, (x,))

        report32 = layer_parity.check_parity(
            ref_fn=ref_fn,
            ref_params=ref_params,
            ref_input=batch.patches_bnpd,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=3300 + grid_idx,
        )
        self.assertTrue(report32.passed, msg=f"T3 fp32 {grid} failures: {report32.failures}")
        in_ok32, in_m32 = mm_testing.check_input_gradient(
            ref_fn=lambda x, rf=ref_fn, rp=ref_params: rf(rp, x),
            ref_input=batch.patches_bnpd,
            cand_fn=lambda x, cf=cand_fn, cp=cand_params: cf(cp, x),
            cand_input=cand_input,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=3300 + grid_idx,
        )
        self.assertTrue(in_ok32, msg=f"T3 fp32 input grad {grid}: {in_m32}")

        fired, neg_metrics = layer_parity.negative_control_fires(
            ref_out=report32.ref_out,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="blocks_1/mlp/w1/kernel",
            rel_scale=_NEGCTL_REL_SCALE,
            seed=303 + grid_idx,
        )
        self.assertTrue(fired, msg=f"T3 negative control {grid} did not fire: {neg_metrics}")

        with multimodal.fp64_reference_context():
          ref_mm64, batch64 = _build_reference_pair(grid, seed=31 + grid_idx, dtype=torch.float64)
          assert ref_mm64.vision is not None
          adapter64 = _BatchedViTAdapter(ref_mm64.vision, n_h, n_w)
          ref_params64 = {k.removeprefix("vit."): v for k, v in adapter64.named_parameters()}

          def ref_fn64(
              params: dict[str, torch.Tensor],
              x: torch.Tensor,
              adp: _BatchedViTAdapter = adapter64,
          ) -> torch.Tensor:
            return functional_call(adp, {f"vit.{k}": v for k, v in params.items()}, (x,))

          floor = layer_parity.reference_noise_floor(ref_fn64, ref_params64, batch64.patches_bnpd)
          self.assertTrue(floor.finite)
          self.assertLessEqual(floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

          with mm_testing.fp64_maxtext_context(cfg):
            mt_enc64 = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg, rngs=nnx.Rngs(params=0))
            mm_testing.copy_vision_encoder(mt_enc64, ref_mm64.vision, jnp.float64)
            e64_gdef, e64_pstate, cand_params64, e64_ostate = mm_testing.extract_pure_params(mt_enc64)
            cand_input64 = mm_testing.torch_to_jax(batch64.patches_bnpd, jnp.float64)

            def cand_fn64(
                params: dict[str, Any],
                x: jax.Array,
                gdef: Any = e64_gdef,
                pstate: Any = e64_pstate,
                ostate: Any = e64_ostate,
            ) -> jax.Array:
              enc = mm_testing.apply_with_pure_params(gdef, pstate, params, ostate)
              return enc(x, deterministic=True)

            report64 = layer_parity.check_parity(
                ref_fn=ref_fn64,
                ref_params=ref_params64,
                ref_input=batch64.patches_bnpd,
                cand_fn=cand_fn64,
                cand_params=cand_params64,
                cand_input=cand_input64,
                weight_map=weight_map,
                grad_classes=grad_classes,
                fwd_tol=layer_parity.FP64_TOLERANCE,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=3300 + grid_idx,
            )
            self.assertTrue(report64.passed, msg=f"T3 fp64 {grid} failures: {report64.failures}")
            in_ok64, in_m64 = mm_testing.check_input_gradient(
                ref_fn=lambda x, rf=ref_fn64, rp=ref_params64: rf(rp, x),
                ref_input=batch64.patches_bnpd,
                cand_fn=lambda x, cf=cand_fn64, cp=cand_params64: cf(cp, x),
                cand_input=cand_input64,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=3300 + grid_idx,
            )
            self.assertTrue(in_ok64, msg=f"T3 fp64 input grad {grid}: {in_m64}")


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41VisionAlignerParityTest(unittest.TestCase):
  """T4: 3x3 spatial aligner + exact GELU on padded 7x5 and exact 6x9 grids, fp32 + fp64."""

  def test_aligner_padded_and_exact_grids_fp32_and_fp64(self) -> None:
    weight_map = mm_testing.projector_weight_map(cand_root="projector", include_delimiters=False)
    for grid_idx, grid in enumerate((multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT)):
      n_h, n_w = grid
      num_patches = n_h * n_w
      ref_mm, _ = _build_reference_pair(grid, seed=41 + grid_idx, dtype=torch.float32)
      assert ref_mm.aligner is not None
      adapter = _BatchedAlignerAdapter(ref_mm.aligner, n_h, n_w)
      ref_params = dict(adapter.named_parameters())
      grad_classes = layer_parity.GradientClasses(compared=frozenset(ref_params))

      vit_np = np.random.default_rng(410 + grid_idx).standard_normal((2, 2, num_patches, 128))
      vit_pt32 = torch.tensor(vit_np, dtype=torch.float32)

      cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
      mesh = mm_testing.single_device_mesh()
      with mesh:
        mt_proj = deepseek41_vision.DeepSeek41VisionProjector(config=cfg, rngs=nnx.Rngs(params=0))
        mm_testing.copy_vision_projector(
            mt_proj,
            ref_mm.aligner,
            ref_mm.image_start,
            ref_mm.image_end,
            ref_mm.image_newline,
            jnp.float32,
        )
        p_gdef, p_pstate, p_pure, p_ostate = mm_testing.extract_pure_params(mt_proj)
        cand_params = {"projector": {"w1": p_pure["w1"], "w2": p_pure["w2"]}}
        fixed_delims = (p_pure["image_start"], p_pure["image_end"], p_pure["image_newline"])
        cand_input = jnp.asarray(vit_np.astype(np.float32), dtype=jnp.float32)

        def cand_fn(
            params: dict[str, Any],
            x: jax.Array,
            gdef: Any = p_gdef,
            pstate: Any = p_pstate,
            ostate: Any = p_ostate,
            delims: tuple[Any, Any, Any] = fixed_delims,
            nh: int = n_h,
            nw: int = n_w,
        ) -> jax.Array:
          s_tok, e_tok, nl_tok = delims
          full_pure = {
              **params["projector"],
              "image_start": s_tok,
              "image_end": e_tok,
              "image_newline": nl_tok,
          }
          proj = mm_testing.apply_with_pure_params(gdef, pstate, full_pure, ostate)
          return proj.align(x, nh, nw)

        def ref_fn(
            params: dict[str, torch.Tensor],
            x: torch.Tensor,
            adp: _BatchedAlignerAdapter = adapter,
        ) -> torch.Tensor:
          return functional_call(adp, params, (x,))

        report32 = layer_parity.check_parity(
            ref_fn=ref_fn,
            ref_params=ref_params,
            ref_input=vit_pt32,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=4400 + grid_idx,
        )
        self.assertTrue(report32.passed, msg=f"T4 fp32 {grid} failures: {report32.failures}")
        in_ok32, in_m32 = mm_testing.check_input_gradient(
            ref_fn=lambda x, rf=ref_fn, rp=ref_params: rf(rp, x),
            ref_input=vit_pt32,
            cand_fn=lambda x, cf=cand_fn, cp=cand_params: cf(cp, x),
            cand_input=cand_input,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=4400 + grid_idx,
        )
        self.assertTrue(in_ok32, msg=f"T4 fp32 input grad {grid}: {in_m32}")

        fired, neg_metrics = layer_parity.negative_control_fires(
            ref_out=report32.ref_out,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="projector/w1/kernel",
            rel_scale=_NEGCTL_REL_SCALE,
            seed=404 + grid_idx,
        )
        self.assertTrue(fired, msg=f"T4 negative control {grid} did not fire: {neg_metrics}")

        with multimodal.fp64_reference_context():
          ref_mm64, _ = _build_reference_pair(grid, seed=41 + grid_idx, dtype=torch.float64)
          assert ref_mm64.aligner is not None
          adapter64 = _BatchedAlignerAdapter(ref_mm64.aligner, n_h, n_w)
          ref_params64 = dict(adapter64.named_parameters())
          vit_pt64 = torch.tensor(vit_np, dtype=torch.float64)

          def ref_fn64(
              params: dict[str, torch.Tensor],
              x: torch.Tensor,
              adp: _BatchedAlignerAdapter = adapter64,
          ) -> torch.Tensor:
            return functional_call(adp, params, (x,))

          floor = layer_parity.reference_noise_floor(ref_fn64, ref_params64, vit_pt64)
          self.assertTrue(floor.finite)
          self.assertLessEqual(floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

          with mm_testing.fp64_maxtext_context(cfg):
            mt_proj64 = deepseek41_vision.DeepSeek41VisionProjector(config=cfg, rngs=nnx.Rngs(params=0))
            mm_testing.copy_vision_projector(
                mt_proj64,
                ref_mm64.aligner,
                ref_mm64.image_start,
                ref_mm64.image_end,
                ref_mm64.image_newline,
                jnp.float64,
            )
            p64_gdef, p64_pstate, p64_pure, p64_ostate = mm_testing.extract_pure_params(mt_proj64)
            cand_params64 = {"projector": {"w1": p64_pure["w1"], "w2": p64_pure["w2"]}}
            fixed_delims64 = (p64_pure["image_start"], p64_pure["image_end"], p64_pure["image_newline"])
            cand_input64 = jnp.asarray(vit_np, dtype=jnp.float64)

            def cand_fn64(
                params: dict[str, Any],
                x: jax.Array,
                gdef: Any = p64_gdef,
                pstate: Any = p64_pstate,
                ostate: Any = p64_ostate,
                delims: tuple[Any, Any, Any] = fixed_delims64,
                nh: int = n_h,
                nw: int = n_w,
            ) -> jax.Array:
              s_tok, e_tok, nl_tok = delims
              full_pure = {
                  **params["projector"],
                  "image_start": s_tok,
                  "image_end": e_tok,
                  "image_newline": nl_tok,
              }
              proj = mm_testing.apply_with_pure_params(gdef, pstate, full_pure, ostate)
              return proj.align(x, nh, nw)

            report64 = layer_parity.check_parity(
                ref_fn=ref_fn64,
                ref_params=ref_params64,
                ref_input=vit_pt64,
                cand_fn=cand_fn64,
                cand_params=cand_params64,
                cand_input=cand_input64,
                weight_map=weight_map,
                grad_classes=grad_classes,
                fwd_tol=layer_parity.FP64_TOLERANCE,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=4400 + grid_idx,
            )
            self.assertTrue(report64.passed, msg=f"T4 fp64 {grid} failures: {report64.failures}")
            in_ok64, in_m64 = mm_testing.check_input_gradient(
                ref_fn=lambda x, rf=ref_fn64, rp=ref_params64: rf(rp, x),
                ref_input=vit_pt64,
                cand_fn=lambda x, cf=cand_fn64, cp=cand_params64: cf(cp, x),
                cand_input=cand_input64,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=4400 + grid_idx,
            )
            self.assertTrue(in_ok64, msg=f"T4 fp64 input grad {grid}: {in_m64}")


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41SpanAssemblyParityTest(unittest.TestCase):
  """T5: Span assembly `[START] + (row + NEWLINE)*H + [END]` and `encoders.VisionEncoder` parity."""

  def test_projector_span_assembly_fp32_bitwise_delimiters_and_fp64(self) -> None:
    weight_map = mm_testing.projector_weight_map(cand_root="projector", include_delimiters=True)
    for grid_idx, grid in enumerate((multimodal.MINI_GRID_PADDED, multimodal.MINI_GRID_EXACT)):
      n_h, n_w = grid
      num_patches = n_h * n_w
      ref_mm, _ = _build_reference_pair(grid, seed=51 + grid_idx, dtype=torch.float32)
      assert ref_mm.aligner is not None
      adapter = _SpanMergeAdapter(ref_mm, n_h, n_w, include_vit=False)
      ref_params = dict(adapter.named_parameters())
      grad_classes = layer_parity.GradientClasses(compared=frozenset(ref_params))

      vit_np = np.random.default_rng(510 + grid_idx).standard_normal((2, 2, num_patches, 128))
      vit_pt32 = torch.tensor(vit_np, dtype=torch.float32)

      cfg = mm_testing.mini_vision_maxtext_config(grid=grid)
      mesh = mm_testing.single_device_mesh()
      with mesh:
        mt_proj = deepseek41_vision.DeepSeek41VisionProjector(config=cfg, rngs=nnx.Rngs(params=0))
        mm_testing.copy_vision_projector(
            mt_proj,
            ref_mm.aligner,
            ref_mm.image_start,
            ref_mm.image_end,
            ref_mm.image_newline,
            jnp.float32,
        )
        p_gdef, p_pstate, p_pure, p_ostate = mm_testing.extract_pure_params(mt_proj)
        cand_params = {"projector": p_pure}
        cand_input = jnp.asarray(vit_np.astype(np.float32), dtype=jnp.float32)

        def cand_fn(
            params: dict[str, Any],
            x: jax.Array,
            gdef: Any = p_gdef,
            pstate: Any = p_pstate,
            ostate: Any = p_ostate,
        ) -> jax.Array:
          proj = mm_testing.apply_with_pure_params(gdef, pstate, params["projector"], ostate)
          return proj(x)

        def ref_fn(
            params: dict[str, torch.Tensor],
            x: torch.Tensor,
            adp: _SpanMergeAdapter = adapter,
        ) -> torch.Tensor:
          return functional_call(adp, params, (x,))

        # Verify bitwise delimiter equality and IMAGE row parity separately, then full-span parity.
        cand_span32 = np.asarray(mt_proj(cand_input))
        with torch.no_grad():
          ref_span32 = adapter(vit_pt32).numpy()
        types_np = ref_image_processor.image_token_types(adapter.n_llm_h, adapter.n_llm_w).numpy()
        for delim_code, delim_name in (
            (ref_image_processor.IMAGE_START, "IMAGE_START"),
            (ref_image_processor.IMAGE_NEW_LINE, "IMAGE_NEW_LINE"),
            (ref_image_processor.IMAGE_END, "IMAGE_END"),
        ):
          mask = types_np == delim_code
          self.assertGreater(int(np.sum(mask)), 0)
          np.testing.assert_array_equal(
              cand_span32[:, :, mask, :],
              ref_span32[:, :, mask, :],
              err_msg=f"Delimiter {delim_name} not bitwise equal on {grid}",
          )
        img_mask = types_np == ref_image_processor.IMAGE
        img_ok, img_m = layer_parity.compare(
            cand_span32[:, :, img_mask, :],
            ref_span32[:, :, img_mask, :],
            layer_parity.FP32_TOLERANCE,
        )
        self.assertTrue(img_ok, msg=f"T5 IMAGE rows {grid}: {img_m}")

        report32 = layer_parity.check_parity(
            ref_fn=ref_fn,
            ref_params=ref_params,
            ref_input=vit_pt32,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=5500 + grid_idx,
        )
        self.assertTrue(report32.passed, msg=f"T5 fp32 {grid} failures: {report32.failures}")
        in_ok32, in_m32 = mm_testing.check_input_gradient(
            ref_fn=lambda x, rf=ref_fn, rp=ref_params: rf(rp, x),
            ref_input=vit_pt32,
            cand_fn=lambda x, cf=cand_fn, cp=cand_params: cf(cp, x),
            cand_input=cand_input,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=5500 + grid_idx,
        )
        self.assertTrue(in_ok32, msg=f"T5 fp32 input grad {grid}: {in_m32}")

        fired, neg_metrics = layer_parity.negative_control_fires(
            ref_out=report32.ref_out,
            cand_fn=cand_fn,
            cand_params=cand_params,
            cand_input=cand_input,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            target_path="projector/image_newline",
            rel_scale=_NEGCTL_REL_SCALE,
            seed=505 + grid_idx,
        )
        self.assertTrue(fired, msg=f"T5 negative control {grid} did not fire: {neg_metrics}")

        with multimodal.fp64_reference_context():
          ref_mm64, _ = _build_reference_pair(grid, seed=51 + grid_idx, dtype=torch.float64)
          adapter64 = _SpanMergeAdapter(ref_mm64, n_h, n_w, include_vit=False)
          ref_params64 = dict(adapter64.named_parameters())
          vit_pt64 = torch.tensor(vit_np, dtype=torch.float64)

          def ref_fn64(
              params: dict[str, torch.Tensor],
              x: torch.Tensor,
              adp: _SpanMergeAdapter = adapter64,
          ) -> torch.Tensor:
            return functional_call(adp, params, (x,))

          floor = layer_parity.reference_noise_floor(ref_fn64, ref_params64, vit_pt64)
          self.assertTrue(floor.finite)
          self.assertLessEqual(floor.frob_rel, layer_parity.FP32_FROB_REL_CEILING)

          with mm_testing.fp64_maxtext_context(cfg):
            mt_proj64 = deepseek41_vision.DeepSeek41VisionProjector(config=cfg, rngs=nnx.Rngs(params=0))
            assert ref_mm64.aligner is not None
            mm_testing.copy_vision_projector(
                mt_proj64,
                ref_mm64.aligner,
                ref_mm64.image_start,
                ref_mm64.image_end,
                ref_mm64.image_newline,
                jnp.float64,
            )
            p64_gdef, p64_pstate, p64_pure, p64_ostate = mm_testing.extract_pure_params(mt_proj64)
            cand_params64 = {"projector": p64_pure}
            cand_input64 = jnp.asarray(vit_np, dtype=jnp.float64)

            def cand_fn64(
                params: dict[str, Any],
                x: jax.Array,
                gdef: Any = p64_gdef,
                pstate: Any = p64_pstate,
                ostate: Any = p64_ostate,
            ) -> jax.Array:
              proj = mm_testing.apply_with_pure_params(gdef, pstate, params["projector"], ostate)
              return proj(x)

            report64 = layer_parity.check_parity(
                ref_fn=ref_fn64,
                ref_params=ref_params64,
                ref_input=vit_pt64,
                cand_fn=cand_fn64,
                cand_params=cand_params64,
                cand_input=cand_input64,
                weight_map=weight_map,
                grad_classes=grad_classes,
                fwd_tol=layer_parity.FP64_TOLERANCE,
                grad_tol=layer_parity.FP64_TOLERANCE,
                cotangent_seed=5500 + grid_idx,
            )
            self.assertTrue(report64.passed, msg=f"T5 fp64 {grid} failures: {report64.failures}")

  def test_vision_encoder_wrapper_freeze_modes_and_multiaxis_sharding(self) -> None:
    grid = multimodal.MINI_GRID_PADDED
    n_h, n_w = grid
    ref_mm, batch = _build_reference_pair(grid, seed=61, dtype=torch.float32)
    adapter = _SpanMergeAdapter(ref_mm, n_h, n_w, include_vit=True)
    ref_params = dict(adapter.named_parameters())

    # Cross-check that _SpanMergeAdapter matches ref_mm.merge_image_embeddings bitwise.
    span_len = int(batch.images[0][0].types.numel())
    with torch.no_grad():
      direct_zero_h = torch.zeros((batch.patches_bnpd.shape[0], 2 * span_len, ref_mm.image_start.shape[0]))
      direct_images = [
          [
              ref_image_processor.ImageInput(
                  start=ni * span_len,
                  patches=batch.patches_bnpd[bi, ni],
                  n_vit_h=n_h,
                  n_vit_w=n_w,
                  types=batch.images[bi][ni].types,
              )
              for ni in range(2)
          ]
          for bi in range(batch.patches_bnpd.shape[0])
      ]
      direct_spans = ref_mm.merge_image_embeddings(direct_images, direct_zero_h).reshape(2, 2, span_len, -1)
      np.testing.assert_array_equal(adapter(batch.patches_bnpd).numpy(), direct_spans.numpy())

    weight_map = mm_testing.vision_wrapper_weight_map(num_layers=2)
    mesh = mm_testing.single_device_mesh()
    cand_input = mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float32)

    for freeze_vit in (False, True):
      cfg = mm_testing.mini_vision_maxtext_config(grid=grid, freeze_vision_encoder_params=freeze_vit)
      frozen_keys = frozenset(k for k in ref_params if freeze_vit and k.startswith("vision."))
      compared_keys = frozenset(ref_params) - frozen_keys
      grad_classes = layer_parity.GradientClasses(compared=compared_keys, frozen=frozen_keys)
      with mesh:
        wrapper = encoders.VisionEncoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0))
        mm_testing.copy_full_vision_encoder(wrapper, ref_mm, jnp.float32)
        w_gdef, w_pstate, w_pure, w_ostate = mm_testing.extract_pure_params(wrapper)

        def cand_fn(
            params: dict[str, Any],
            x: jax.Array,
            gdef: Any = w_gdef,
            pstate: Any = w_pstate,
            ostate: Any = w_ostate,
        ) -> jax.Array:
          mod = mm_testing.apply_with_pure_params(gdef, pstate, params, ostate)
          spans, deep_feats = mod(x, deterministic=True)
          assert deep_feats is None
          return spans

        def ref_fn(
            params: dict[str, torch.Tensor],
            x: torch.Tensor,
            adp: _SpanMergeAdapter = adapter,
        ) -> torch.Tensor:
          return functional_call(adp, params, (x,))

        report = layer_parity.check_parity(
            ref_fn=ref_fn,
            ref_params=ref_params,
            ref_input=batch.patches_bnpd,
            cand_fn=cand_fn,
            cand_params=w_pure,
            cand_input=cand_input,
            weight_map=weight_map,
            grad_classes=grad_classes,
            fwd_tol=layer_parity.FP32_TOLERANCE,
            grad_tol=layer_parity.FP32_TOLERANCE,
            cotangent_seed=6100,
        )
        self.assertTrue(report.passed, msg=f"VisionEncoder freeze={freeze_vit} failures: {report.failures}")

    # Multi-axis TP=2, EP=2 HLO check across 4 CPU devices: resolve parameter NamedShardings from
    # `t_pstate` via MaxText's sharding resolver and verify replicated vision weights emit no all-gather.
    self.assertGreaterEqual(len(jax.devices()), 4)
    tp_ep_mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("tensor", "expert"))
    cfg_tpep = mm_testing.mini_vision_maxtext_config(grid=grid)
    with tp_ep_mesh, flax.core.spmd.logical_axis_rules(cfg_tpep.logical_axis_rules):
      wrapper_tpep = encoders.VisionEncoder(config=cfg_tpep, mesh=tp_ep_mesh, rngs=nnx.Rngs(params=0))
      mm_testing.copy_full_vision_encoder(wrapper_tpep, ref_mm, jnp.float32)
      t_gdef, t_pstate, t_pure, t_ostate = mm_testing.extract_pure_params(wrapper_tpep)
      named_shardings = nnx.to_pure_dict(sharding.nnx_construct_named_sharding(t_pstate, tp_ep_mesh))
      for path, ns in jax.tree_util.tree_leaves_with_path(named_shardings):
        self.assertIsInstance(ns, jax.sharding.NamedSharding)
        self.assertTrue(
            all(axis is None for axis in ns.spec),
            msg=f"Expected replicated PartitionSpec for {jax.tree_util.keystr(path)}, got {ns.spec}",
        )

      @jax.jit
      def jit_fwd(params: dict[str, Any], x: jax.Array) -> jax.Array:
        mod = mm_testing.apply_with_pure_params(t_gdef, t_pstate, params, t_ostate)
        spans, _ = mod(x, deterministic=True)
        return spans

      replicated_input_sharding = jax.sharding.NamedSharding(tp_ep_mesh, jax.sharding.PartitionSpec())
      sharded_params = jax.tree.map(jax.device_put, t_pure, named_shardings)
      sharded_input = jax.device_put(cand_input, replicated_input_sharding)
      hlo_text = jit_fwd.lower(sharded_params, sharded_input).compile().as_text()
      self.assertNotIn("all-gather", hlo_text)
      out_tpep = np.asarray(jit_fwd(sharded_params, sharded_input))
      ok_tpep, m_tpep = layer_parity.compare(out_tpep, direct_spans.numpy(), layer_parity.FP32_TOLERANCE)
      self.assertTrue(ok_tpep, msg=f"TP>1/EP>1 output mismatch: {m_tpep}")

      # Negative control: sharding mlp/w1/kernel on "tensor" must emit all-gather in compiled HLO.
      neg_shardings = jax.tree.map(lambda s: s, named_shardings)
      neg_shardings["DeepSeek41VisionEncoder_0"]["blocks_0"]["mlp"]["w1"]["kernel"] = jax.sharding.NamedSharding(
          tp_ep_mesh, jax.sharding.PartitionSpec(None, "tensor")
      )
      neg_params = jax.tree.map(jax.device_put, t_pure, neg_shardings)
      neg_hlo_text = jit_fwd.lower(neg_params, sharded_input).compile().as_text()
      self.assertIn("all-gather", neg_hlo_text)


@pytest.mark.cpu_only
class DeepSeek41InitAndConfigTest(unittest.TestCase):
  """T5b: `get_shaped_batch`, int32 dummy init/forward, YAML config validation, and decoder boundary."""

  def test_get_shaped_batch_and_int32_dummy_init_and_forward(self) -> None:
    mesh = mm_testing.single_device_mesh()
    for grid, expected_patches, expected_span_len in (
        (mm_testing.MINI_GRID_PADDED, 35, 11),
        (mm_testing.MINI_GRID_EXACT, 54, 10),
        ((39, 39), 1521, 13 * (13 + 1) + 2),
    ):
      cfg = mm_testing.mini_vision_maxtext_config(grid=grid, dtype_mm="bfloat16")
      shaped = maxtext_utils.get_shaped_batch(cfg)
      batch_size = int(cfg.micro_batch_size_to_train_on)
      self.assertIn("images", shaped)
      self.assertNotIn("image_masks", shaped)
      self.assertEqual(shaped["images"].shape, (batch_size, 1, expected_patches, 588))
      self.assertEqual(shaped["images"].dtype, jnp.int32)

      if grid != (39, 39):
        dummy_images = jnp.ones(shaped["images"].shape, dtype=jnp.int32)
        with mesh:
          wrapper = encoders.VisionEncoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0))
          spans_nnx, deep_feats_nnx = jax.jit(lambda imgs, w=wrapper: w(imgs, deterministic=True))(dummy_images)
          self.assertIsNone(deep_feats_nnx)
          self.assertEqual(spans_nnx.shape, (batch_size, 1, expected_span_len, cfg.emb_dim))
          self.assertEqual(spans_nnx.dtype, jnp.bfloat16)
          self.assertTrue(bool(jnp.all(jnp.isfinite(spans_nnx))))

          linen_enc = nnx_wrappers.ToLinen(
              encoders.VisionEncoder,
              args=(),
              kwargs={"config": cfg, "mesh": mesh},
              metadata_fn=initializers.variable_to_logically_partitioned,
          )
          variables = jax.jit(linen_enc.init)(jax.random.PRNGKey(0), dummy_images, deterministic=True)
          spans_linen, deep_feats_linen = jax.jit(linen_enc.apply)(variables, dummy_images, deterministic=True)
          self.assertIsNone(deep_feats_linen)
          self.assertEqual(spans_linen.shape, (batch_size, 1, expected_span_len, cfg.emb_dim))
          self.assertEqual(spans_linen.dtype, jnp.bfloat16)
          self.assertTrue(bool(jnp.all(jnp.isfinite(spans_linen))))

    # Positive match with config passed, paired with negative assertion when config is omitted.
    cfg_pos = mm_testing.mini_vision_maxtext_config(grid=mm_testing.MINI_GRID_PADDED)
    self.assertEqual(
        mm_processor.get_dummy_image_shape_for_init("deepseek4.1-flash", batch_size=2, config=cfg_pos),
        (2, 1, 35, 588),
    )
    with self.assertRaisesRegex(ValueError, "ds41_vision_grid"):
      mm_processor.get_dummy_image_shape_for_init("deepseek4.1-flash", batch_size=2, config=None)

  def test_yaml_configs_and_validation_and_decoder_boundary(self) -> None:
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg_text = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              f"model_name={model_name}",
              "run_name=ds41_yaml_text_check",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertFalse(cfg_text.use_multimodal)
      self.assertEqual(cfg_text.vision_encoder_block, VisionEncoderBlockType.DEEPSEEK41)
      self.assertEqual(cfg_text.num_hidden_layers_for_vit, 32)
      self.assertEqual(cfg_text.hidden_size_for_vit, 1024)
      self.assertEqual(cfg_text.num_attention_heads_for_vit, 16)
      self.assertEqual(cfg_text.intermediate_size_for_vit, 2816)
      self.assertEqual(cfg_text.patch_size_for_vit, 14)
      self.assertEqual(cfg_text.rope_theta_for_vit, 10000)
      self.assertEqual(cfg_text.num_channels_for_vit, 3)
      self.assertEqual(cfg_text.ds41_vision_downsample_ratio, 3)
      self.assertEqual(list(cfg_text.ds41_vision_grid), [39, 39])
      self.assertEqual(cfg_text.dtype_mm, "bfloat16")
      self.assertTrue(cfg_text.freeze_vision_encoder_params)

      cfg_mm = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              f"model_name={model_name}",
              "run_name=ds41_yaml_mm_check",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
              "use_multimodal=true",
          ]
      )
      self.assertTrue(cfg_mm.use_multimodal)
      self.assertEqual(cfg_mm.vision_encoder_block, VisionEncoderBlockType.DEEPSEEK41)

    # Production 1024-wide PatchEmbed kernel shape check: [588, 1024].
    mesh = mm_testing.single_device_mesh()
    cfg_prod_vit = mm_testing.mini_vision_maxtext_config(
        num_hidden_layers_for_vit=1,
        hidden_size_for_vit=1024,
        num_attention_heads_for_vit=16,
        intermediate_size_for_vit=2816,
    )
    with mesh:
      prod_vit = deepseek41_vision.DeepSeek41VisionEncoder(config=cfg_prod_vit, rngs=nnx.Rngs(params=0))
      self.assertEqual(prod_vit.patch_embed.kernel[...].shape, (588, 1024))

    # Validation: multimodal DeepSeek-V4.1 rejects vision_encoder_block != 'deepseek41' (both explicit and default 'none').
    with self.assertRaisesRegex(ValueError, "vision_encoder_block='deepseek41'"):
      mm_testing.mini_vision_maxtext_config(use_multimodal=True, vision_encoder_block="gemma3")
    with self.assertRaisesRegex(ValueError, "vision_encoder_block='deepseek41'"):
      pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              "model_name=default",
              "decoder_block=deepseek41",
              "run_name=ds41_default_missing_vision_block",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
              "use_multimodal=true",
          ]
      )

    # Validation: ds41_vision_grid must be two positive integers.
    for bad_grid in ([39], [0, 39], [39, -1]):
      with self.assertRaisesRegex(ValueError, "ds41_vision_grid"):
        mm_testing.mini_vision_maxtext_config(ds41_vision_grid=bad_grid)

    # Text-only full Transformer has no vision_encoder params; multimodal Transformer executes end-to-end.
    cfg_mini_text = mm_testing.mini_vision_maxtext_config(use_multimodal=False)
    model_mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg_mini_text), cfg_mini_text.mesh_axes)
    model_text = model_creation_utils.from_pretrained(cfg_mini_text, mesh=model_mesh, model_mode=MODEL_MODE_TRAIN)
    self.assertFalse(hasattr(model_text, "vision_encoder") and model_text.vision_encoder is not None)

    cfg_mini_mm = mm_testing.mini_vision_maxtext_config(use_multimodal=True)
    model_mm = model_creation_utils.from_pretrained(cfg_mini_mm, mesh=model_mesh, model_mode=MODEL_MODE_TRAIN)
    self.assertIsNotNone(model_mm.vision_encoder)
    batch_size = int(cfg_mini_mm.micro_batch_size_to_train_on)
    seq_len = int(cfg_mini_mm.max_target_length)
    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    tokens = tokens.at[:, 2:13].set(cfg_mini_mm.ds41_image_token_id)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
    dummy_imgs = jnp.ones((batch_size, 1, 35, 588), dtype=jnp.float32)
    with model_mesh:
      logits = model_mm(tokens, positions, encoder_images=dummy_imgs, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(logits.shape, (batch_size, seq_len, cfg_mini_mm.vocab_size))
    self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))


if __name__ == "__main__":
  unittest.main()
