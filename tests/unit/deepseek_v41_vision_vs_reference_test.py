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

"""PyTorch-reference forward, all-leaf gradient, fp64, and negative-control parity tests for DeepSeek-V4.1 vision."""

from __future__ import annotations

import functools
import io
from typing import Any
import unittest

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest
from transformers import AutoTokenizer

from maxtext.layers import encoders
from maxtext.models import deepseek41_vision
from maxtext.multimodal import processor_deepseek41 as ds41_proc
from maxtext.utils import sharding
from tests.utils import deepseek_v41_mm_testing as mm_testing
from tests.utils.deepseek_v41.parity import layer_parity

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


_ORIG_MATMUL_PRECISION = None


def setUpModule() -> None:
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")


def tearDownModule() -> None:
  jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


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
    # `t_pstate` via MaxText's sharding resolver and verify sharded ViT/aligner kernels emit all-gather and match.
    if len(jax.devices()) < 4:
      self.skipTest("Requires >= 4 JAX devices")
    tp_ep_mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("tensor", "expert"))
    cfg_tpep = mm_testing.mini_vision_maxtext_config(grid=grid)
    with tp_ep_mesh, flax.core.spmd.logical_axis_rules(cfg_tpep.logical_axis_rules):
      wrapper_tpep = encoders.VisionEncoder(config=cfg_tpep, mesh=tp_ep_mesh, rngs=nnx.Rngs(params=0))
      mm_testing.copy_full_vision_encoder(wrapper_tpep, ref_mm, jnp.float32)
      t_gdef, t_pstate, t_pure, t_ostate = mm_testing.extract_pure_params(wrapper_tpep)
      named_shardings = nnx.to_pure_dict(sharding.nnx_construct_named_sharding(t_pstate, tp_ep_mesh))
      pspec = jax.sharding.PartitionSpec
      enc_s = named_shardings["DeepSeek41VisionEncoder_0"]
      proj_s = named_shardings["DeepSeek41VisionProjector_0"]
      self.assertEqual(enc_s["patch_embed"]["kernel"].spec, pspec(None, "expert"))
      self.assertEqual(enc_s["patch_embed"]["bias"].spec, pspec("expert"))
      for blk_idx in range(2):
        blk_s = enc_s[f"blocks_{blk_idx}"]
        self.assertEqual(blk_s["attn"]["wqkv"]["kernel"].spec, pspec("expert", "tensor"))
        self.assertEqual(blk_s["attn"]["wqkv"]["bias"].spec, pspec("tensor"))
        self.assertEqual(blk_s["attn"]["wo"]["kernel"].spec, pspec("tensor", "expert"))
        self.assertEqual(blk_s["attn"]["wo"]["bias"].spec, pspec("expert"))
        self.assertEqual(blk_s["mlp"]["w1"]["kernel"].spec, pspec("expert", "tensor"))
        self.assertEqual(blk_s["mlp"]["w2"]["kernel"].spec, pspec("tensor", "expert"))
        self.assertEqual(blk_s["norm1"]["scale"].spec, pspec(None))
        self.assertEqual(blk_s["norm2"]["scale"].spec, pspec(None))
      self.assertEqual(enc_s["norm"]["scale"].spec, pspec(None))
      self.assertEqual(proj_s["w1"]["kernel"].spec, pspec("expert", "tensor"))
      self.assertEqual(proj_s["w1"]["bias"].spec, pspec("tensor"))
      self.assertEqual(proj_s["w2"]["kernel"].spec, pspec("tensor", "expert"))
      self.assertEqual(proj_s["w2"]["bias"].spec, pspec("expert"))
      self.assertEqual(proj_s["image_start"].spec, pspec(None))
      self.assertEqual(proj_s["image_end"].spec, pspec(None))
      self.assertEqual(proj_s["image_newline"].spec, pspec(None))

      @jax.jit
      def jit_fwd(params: dict[str, Any], x: jax.Array) -> jax.Array:
        mod = mm_testing.apply_with_pure_params(t_gdef, t_pstate, params, t_ostate)
        spans, _ = mod(x, deterministic=True)
        return spans

      replicated_input_sharding = jax.sharding.NamedSharding(tp_ep_mesh, jax.sharding.PartitionSpec())
      sharded_params = jax.tree.map(jax.device_put, t_pure, named_shardings)
      sharded_input = jax.device_put(cand_input, replicated_input_sharding)
      hlo_text = jit_fwd.lower(sharded_params, sharded_input).compile().as_text()
      self.assertIn("all-gather", hlo_text)
      out_tpep = np.asarray(jit_fwd(sharded_params, sharded_input))
      ok_tpep, m_tpep = layer_parity.compare(out_tpep, direct_spans.numpy(), layer_parity.FP32_TOLERANCE)
      self.assertTrue(ok_tpep, msg=f"TP>1/EP>1 output mismatch: {m_tpep}")

      # Negative control: overriding all parameter shardings to replicated removes all-gather from compiled HLO.
      rep_shardings = jax.tree.map(lambda _: replicated_input_sharding, named_shardings)
      rep_params = jax.tree.map(jax.device_put, t_pure, rep_shardings)
      rep_hlo_text = jit_fwd.lower(rep_params, sharded_input).compile().as_text()
      self.assertNotIn("all-gather", rep_hlo_text)


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "Requires torch")
class DeepSeek41DynamicVisionParityTest(unittest.TestCase):
  """PyTorch-reference parity and padded-patch negative controls for `processor_deepseek41` and dynamic `image_grids`."""

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_processor_deepseek41_vs_reference_image_processor(self) -> None:
    """Verifies `processor_deepseek41` matches `image_processor.py` across portrait, landscape, small, and large images."""
    args_full = multimodal.mini_model_args(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_min_pixels=544 * 544,
        vision_max_n_token=1024,
        vision_max_wh_ratio=None,
    )
    sizes = [
        (380, 910),  # portrait
        (1150, 430),  # landscape
        (120, 180),  # small (< 544x544, exercises vision_min_pixels upscale)
        (546, 546),  # exact 39x39 default bucket
        (2800, 2200),  # large (hits vision_max_n_token=1024 cap)
        (40, 4200),  # extreme tall (single-column collapse)
        (4200, 40),  # extreme wide (single-row collapse)
    ]
    for idx, (width, height) in enumerate(sizes):
      raw = np.random.default_rng(900 + idx).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
      pil_img = Image.fromarray(raw, mode="RGB")
      buf = io.BytesIO()
      pil_img.save(buf, format="PNG")
      png_bytes = buf.getvalue()

      ref_patches_bf16, ref_vh, ref_vw, ref_lh, ref_lw = ref_image_processor.load_image({"data": png_bytes}, args_full)
      cand_patches_f32, cand_vh, cand_vw, cand_lh, cand_lw = ds41_proc.extract_image_patches_deepseek41(
          {"bytes": png_bytes},
          patch_size=args_full.vision_patch_size,
          downsample_ratio=args_full.vision_downsample_ratio,
          min_pixels=args_full.vision_min_pixels,
          max_n_token=args_full.vision_max_n_token,
          max_wh_ratio=args_full.vision_max_wh_ratio,
      )
      self.assertEqual((cand_vh, cand_vw), (ref_vh, ref_vw), msg=f"ViT grid mismatch for {(width, height)}")
      self.assertEqual((cand_lh, cand_lw), (ref_lh, ref_lw), msg=f"LLM grid mismatch for {(width, height)}")
      ref_span_count = int(ref_image_processor.image_token_types(ref_lh, ref_lw).numel())
      self.assertEqual(ds41_proc.num_image_tokens(cand_lh, cand_lw), ref_span_count)

      # Bit-exact bfloat16 patch equality with `image_processor.load_image`.
      cand_as_bf16 = torch.from_numpy(cand_patches_f32).to(torch.bfloat16)
      self.assertTrue(
          torch.equal(cand_as_bf16, ref_patches_bf16.flatten(1)),
          msg=f"bfloat16 patch mismatch for {(width, height)}",
      )

      # Float32 patch comparison against exact float32 reference formula (no bf16 truncation).
      if args_full.vision_max_wh_ratio is not None and pil_img.width >= args_full.vision_max_wh_ratio * pil_img.height:
        ref_pil = pil_img.resize((ref_vw * 14, ref_vh * 14))
      else:
        from PIL import ImageOps  # pylint: disable=import-outside-toplevel

        ref_pil = ImageOps.pad(pil_img, (ref_vw * 14, ref_vh * 14), color=(127, 127, 127))
      x_ref = torch.from_numpy(np.asarray(ref_pil, dtype=np.float32)).permute(2, 0, 1) / 255.0
      x_ref = (x_ref - 0.5) / 0.5
      ref_f32 = x_ref.reshape(3, ref_vh, 14, ref_vw, 14).permute(1, 3, 0, 2, 4).reshape(ref_vh * ref_vw, 588).numpy()
      self.assertLessEqual(float(np.max(np.abs(cand_patches_f32 - ref_f32))), 1e-6)

    # Also check `vision_max_wh_ratio` resize branch.
    args_wh = multimodal.mini_model_args(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_min_pixels=544 * 544,
        vision_max_n_token=1024,
        vision_max_wh_ratio=3,
    )
    raw_wh = np.random.default_rng(977).integers(0, 256, size=(200, 1000, 3), dtype=np.uint8)
    buf_wh = io.BytesIO()
    Image.fromarray(raw_wh, mode="RGB").save(buf_wh, format="PNG")
    ref_wh_p, ref_wh_vh, ref_wh_vw, ref_wh_lh, ref_wh_lw = ref_image_processor.load_image(
        {"data": buf_wh.getvalue()}, args_wh
    )
    cand_wh_p, cand_wh_vh, cand_wh_vw, cand_wh_lh, cand_wh_lw = ds41_proc.extract_image_patches_deepseek41(
        {"bytes": buf_wh.getvalue()},
        patch_size=14,
        downsample_ratio=3,
        min_pixels=544 * 544,
        max_n_token=1024,
        max_wh_ratio=3.0,
    )
    self.assertEqual((cand_wh_vh, cand_wh_vw, cand_wh_lh, cand_wh_lw), (ref_wh_vh, ref_wh_vw, ref_wh_lh, ref_wh_lw))
    self.assertTrue(torch.equal(torch.from_numpy(cand_wh_p).to(torch.bfloat16), ref_wh_p.flatten(1)))

  @unittest.skipUnless(HAS_TORCH, "Requires torch")
  def test_mixed_grid_vision_encoder_projector_parity_fp32_fp64_and_negative_controls(self) -> None:
    """Verifies heterogeneous grids `(5, 8) + (6, 9) + (7, 4) + [0, 0]` match per-image PyTorch fwd+VJP in fp32 & fp64."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    self.addCleanup(torch.set_num_threads, prev_threads)

    grid_layout = (((5, 8), (6, 9)), ((7, 4), (0, 0)))
    p_buf = 64
    patch_dim = 3 * 14 * 14
    args = multimodal.mini_model_args()
    r = args.vision_downsample_ratio
    t_span_max = deepseek41_vision.max_span_tokens_for_buffer(p_buf, r, 1024)

    def _build_mixed_inputs(dtype_torch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], Any]]:
      patches_np = np.zeros((2, 2, p_buf, patch_dim), dtype=np.float64 if dtype_torch == torch.float64 else np.float32)
      grids_np = np.zeros((2, 2, 2), dtype=np.int32)
      masks_np = np.zeros((2, 2), dtype=np.int32)
      per_slot: dict[tuple[int, int], Any] = {}
      for bi in range(2):
        for ni in range(2):
          nh, nw = grid_layout[bi][ni]
          grids_np[bi, ni] = [nh, nw]
          if nh > 0 and nw > 0:
            masks_np[bi, ni] = 1
            p_t, lh, lw = multimodal.make_synthetic_image_patches(
                args, nh, nw, seed=700 + bi * 10 + ni, dtype=dtype_torch
            )
            flat_np = p_t.flatten(1).numpy()
            patches_np[bi, ni, : nh * nw, :] = flat_np
            per_slot[(bi, ni)] = (nh, nw, lh, lw, ref_image_processor.image_token_types(lh, lw))
      return patches_np, grids_np, masks_np, per_slot

    def _run_ref_mixed_loss_and_grads(
        ref_mm_inst: multimodal.MultimodalTransformer,
        patches_np: np.ndarray,
        per_slot: dict[tuple[int, int], Any],
        cotangents_np: np.ndarray,
        dtype_torch: Any,
        *,
        freeze_vit: bool,
    ) -> tuple[np.ndarray, float, dict[str, np.ndarray], np.ndarray]:
      emb_dim = int(ref_mm_inst.image_start.shape[0])
      np_dtype = np.float64 if dtype_torch == torch.float64 else np.float32
      ref_mm_inst.zero_grad(set_to_none=True)
      for name, param in ref_mm_inst.named_parameters():
        if freeze_vit and name.startswith("vision."):
          param.requires_grad_(False)
        else:
          param.requires_grad_(True)

      slot_inputs: dict[tuple[int, int], Any] = {}
      ref_spans_np = np.zeros((2, 2, t_span_max, emb_dim), dtype=np_dtype)
      total_loss = torch.zeros((), dtype=dtype_torch)
      for (bi, ni), (nh, nw, _, _, types_t) in per_slot.items():
        p_var = torch.tensor(
            patches_np[bi, ni, : nh * nw, :].reshape(nh * nw, 3, 14, 14), dtype=dtype_torch, requires_grad=True
        )
        slot_inputs[(bi, ni)] = p_var
        zero_h = torch.zeros((1, int(types_t.numel()), emb_dim), dtype=dtype_torch)
        img_in = ref_image_processor.ImageInput(start=0, patches=p_var, n_vit_h=nh, n_vit_w=nw, types=types_t)
        merged = ref_mm_inst.merge_image_embeddings([[img_in]], zero_h)[0]
        span_len = int(types_t.numel())
        ref_spans_np[bi, ni, :span_len, :] = merged.detach().cpu().numpy()
        cot_t = torch.tensor(cotangents_np[bi, ni, :span_len, :], dtype=dtype_torch)
        total_loss = total_loss + (merged * cot_t).sum()

      total_loss.backward()
      ref_grads: dict[str, np.ndarray] = {}
      for name, param in ref_mm_inst.named_parameters():
        if name.startswith(("vision.", "aligner.")) or name in ("image_start", "image_end", "image_newline"):
          if param.grad is not None:
            ref_grads[name] = param.grad.detach().cpu().numpy()
      ref_in_grad = np.zeros_like(patches_np)
      for (bi, ni), p_var in slot_inputs.items():
        nh, nw = grid_layout[bi][ni]
        assert p_var.grad is not None
        ref_in_grad[bi, ni, : nh * nw, :] = p_var.grad.detach().reshape(nh * nw, patch_dim).cpu().numpy()
      return ref_spans_np, float(total_loss.detach().item()), ref_grads, ref_in_grad

    # 1. FP32 parity on single_device_mesh() (both freeze_vit=False and freeze_vit=True) + +100.0 negative control.
    ref_mm32 = multimodal.build_multimodal_model(args, tokenizer=_get_tokenizer())
    multimodal.init_mini_reference(ref_mm32, seed=81)
    patches32_np, grids_np, masks_np, per_slot32 = _build_mixed_inputs(torch.float32)
    cot32_np = np.random.default_rng(8101).standard_normal((2, 2, t_span_max, args.dim)).astype(np.float32)
    weight_map = mm_testing.vision_wrapper_weight_map(num_layers=2)
    mesh1 = mm_testing.single_device_mesh()

    def _extract_cand_leaf(pure_dict: dict[str, Any], spec: str) -> np.ndarray:
      transposed = spec.endswith(":T")
      cur: Any = pure_dict
      for part in spec.removesuffix(":T").split("/"):
        cur = cur[part]
      arr = np.asarray(cur)
      return arr.T if transposed else arr

    for freeze_vit in (False, True):
      ref_spans32, ref_loss32, ref_grads32, ref_ingrad32 = _run_ref_mixed_loss_and_grads(
          ref_mm32, patches32_np, per_slot32, cot32_np, torch.float32, freeze_vit=freeze_vit
      )
      cfg32 = mm_testing.mini_vision_maxtext_config(grid=(6, 9), freeze_vision_encoder_params=freeze_vit)
      with mesh1:
        wrapper32 = encoders.VisionEncoder(config=cfg32, mesh=mesh1, rngs=nnx.Rngs(params=0))
        mm_testing.copy_full_vision_encoder(wrapper32, ref_mm32, jnp.float32)
        w_gdef, w_pstate, w_pure, w_ostate = mm_testing.extract_pure_params(wrapper32)
        patches_j = jnp.asarray(patches32_np, dtype=jnp.float32)
        grids_j = jnp.asarray(grids_np, dtype=jnp.int32)
        masks_j = jnp.asarray(masks_np, dtype=jnp.int32)
        cot_j = jnp.asarray(cot32_np, dtype=jnp.float32)
        span_mask_j = deepseek41_vision.compute_image_span_mask(
            grids_j, span_len_max=t_span_max, downsample_ratio=r, image_masks=masks_j
        )

        @jax.jit
        def eval_fwd_and_vjp32(
            params: dict[str, Any],
            x_in: jax.Array,
            gdef: Any = w_gdef,
            pstate: Any = w_pstate,
            ostate: Any = w_ostate,
            smask: jax.Array = span_mask_j,
            cot: jax.Array = cot_j,
            masks_arg: jax.Array = masks_j,
            grids_arg: jax.Array = grids_j,
        ) -> tuple[jax.Array, jax.Array, dict[str, Any], jax.Array]:
          def _loss(p: dict[str, Any], inp: jax.Array) -> tuple[jax.Array, jax.Array]:
            mod = mm_testing.apply_with_pure_params(gdef, pstate, p, ostate)
            spans_out, deep = mod(inp, input_masks=masks_arg, deterministic=True, image_grids=grids_arg)
            assert deep is None
            masked_cot = cot * smask[..., None].astype(spans_out.dtype)
            return jnp.sum(spans_out * masked_cot), spans_out

          (loss_v, spans_v), (p_grads, x_grad) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(params, x_in)
          return loss_v, spans_v, p_grads, x_grad

        cand_loss32, cand_spans32, cand_pgrads32, cand_xgrad32 = eval_fwd_and_vjp32(w_pure, patches_j)
        self.assertTrue(bool(jnp.all(jnp.isfinite(cand_spans32))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(cand_xgrad32))))
        for leaf in jax.tree.leaves(cand_pgrads32):
          self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

        # Unused [0, 0] slot (1, 1) and padded span positions are exact zeros.
        np.testing.assert_array_equal(np.asarray(cand_spans32)[1, 1], 0.0)
        ok_fwd32, m_fwd32 = layer_parity.compare(
            np.asarray(cand_spans32), ref_spans32.astype(np.float64), layer_parity.FP32_TOLERANCE
        )
        self.assertTrue(ok_fwd32, msg=f"Mixed-grid VisionEncoder fp32 fwd (freeze={freeze_vit}): {m_fwd32}")
        self.assertLessEqual(abs(float(cand_loss32) - ref_loss32), 1e-5 * max(1.0, abs(ref_loss32)))

        for ref_name, spec in weight_map.items():
          cand_leaf = _extract_cand_leaf(cand_pgrads32, spec)
          if freeze_vit and ref_name.startswith("vision."):
            self.assertEqual(float(np.max(np.abs(cand_leaf))), 0.0, msg=f"Non-zero frozen ViT grad on {ref_name}")
          else:
            ok_g, m_g = layer_parity.compare(
                cand_leaf, ref_grads32[ref_name].astype(np.float64), layer_parity.FP32_TOLERANCE
            )
            self.assertTrue(ok_g, msg=f"Mixed-grid fp32 grad {ref_name} (freeze={freeze_vit}): {m_g}")

        if not freeze_vit:
          ok_in, m_in = layer_parity.compare(
              np.asarray(cand_xgrad32), ref_ingrad32.astype(np.float64), layer_parity.FP32_TOLERANCE
          )
          self.assertTrue(ok_in, msg=f"Mixed-grid fp32 input grad: {m_in}")
          # Padded patches and unused [0, 0] slot must have exact 0.0 input gradient.
          np.testing.assert_array_equal(np.asarray(cand_xgrad32)[0, 0, 40:], 0.0)
          np.testing.assert_array_equal(np.asarray(cand_xgrad32)[0, 1, 54:], 0.0)
          np.testing.assert_array_equal(np.asarray(cand_xgrad32)[1, 0, 28:], 0.0)
          np.testing.assert_array_equal(np.asarray(cand_xgrad32)[1, 1, :], 0.0)

          # Negative control: +100.0 noise on all padded patches and unused [0, 0] slot -> exact 0.0 diff.
          noisy_np = np.array(patches32_np, copy=True)
          noisy_np[0, 0, 40:, :] += 100.0
          noisy_np[0, 1, 54:, :] += 100.0
          noisy_np[1, 0, 28:, :] += 100.0
          noisy_np[1, 1, :, :] += 100.0
          n_loss, n_spans, n_pgrads, n_xgrad = eval_fwd_and_vjp32(w_pure, jnp.asarray(noisy_np))
          self.assertEqual(float(jnp.max(jnp.abs(cand_spans32 - n_spans))), 0.0)
          self.assertEqual(float(jnp.abs(cand_loss32 - n_loss)), 0.0)
          self.assertEqual(float(jnp.max(jnp.abs(cand_xgrad32 - n_xgrad))), 0.0)
          for ga, gb in zip(jax.tree.leaves(cand_pgrads32), jax.tree.leaves(n_pgrads)):
            self.assertEqual(float(jnp.max(jnp.abs(ga - gb))), 0.0)

          # Positive control: +100.0 on valid patch (0, 0, 0) changes valid span embeddings and gradients.
          pert_np = np.array(patches32_np, copy=True)
          pert_np[0, 0, 0, :] += 100.0
          _, pert_spans, _, _ = eval_fwd_and_vjp32(w_pure, jnp.asarray(pert_np))
          self.assertGreater(float(jnp.max(jnp.abs(cand_spans32[0, 0, :10] - pert_spans[0, 0, :10]))), 1e-2)

    # 2. Float64 parity (`1e-10` FP64_TOLERANCE) on 4-device `fsdp=4` mesh.
    if len(jax.devices()) < 4:
      self.skipTest("Requires >= 4 JAX devices for fsdp=4 float64 parity")
    fsdp_mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]), ("fsdp",))
    with multimodal.fp64_reference_context():
      args64 = multimodal.mini_model_args()
      prev_dt = torch.get_default_dtype()
      torch.set_default_dtype(torch.float64)
      try:
        ref_mm64 = multimodal.MultimodalTransformer(args64, tokenizer=_get_tokenizer()).eval()
      finally:
        torch.set_default_dtype(prev_dt)
      multimodal.init_mini_reference(ref_mm64, seed=82)
      for p in ref_mm64.parameters():
        if p.is_floating_point():
          p.data = p.data.to(torch.float64)
      patches64_np, _, _, per_slot64 = _build_mixed_inputs(torch.float64)
      cot64_np = np.random.default_rng(8202).standard_normal((2, 2, t_span_max, args64.dim)).astype(np.float64)
      ref_spans64, ref_loss64, ref_grads64, ref_ingrad64 = _run_ref_mixed_loss_and_grads(
          ref_mm64, patches64_np, per_slot64, cot64_np, torch.float64, freeze_vit=False
      )

      cfg64 = mm_testing.mini_vision_maxtext_config(grid=(6, 9), freeze_vision_encoder_params=False)
      with mm_testing.fp64_maxtext_context(cfg64), fsdp_mesh, flax.core.spmd.logical_axis_rules(cfg64.logical_axis_rules):
        wrapper64 = encoders.VisionEncoder(config=cfg64, mesh=fsdp_mesh, rngs=nnx.Rngs(params=0))
        mm_testing.copy_full_vision_encoder(wrapper64, ref_mm64, jnp.float64)
        w64_gdef, w64_pstate, w64_pure, w64_ostate = mm_testing.extract_pure_params(wrapper64)
        p64_shd = nnx.to_pure_dict(sharding.nnx_construct_named_sharding(w64_pstate, fsdp_mesh))
        sharded_p64 = jax.tree.map(jax.device_put, w64_pure, p64_shd)
        batch_shd = jax.sharding.NamedSharding(fsdp_mesh, jax.sharding.PartitionSpec("fsdp"))
        # Replicate small B=2 across 4 FSDP devices cleanly so sharded kernel_axes run on fsdp=4.
        rep_shd = jax.sharding.NamedSharding(fsdp_mesh, jax.sharding.PartitionSpec())
        del batch_shd
        patches64_j = jax.device_put(jnp.asarray(patches64_np, dtype=jnp.float64), rep_shd)
        grids64_j = jax.device_put(jnp.asarray(grids_np, dtype=jnp.int32), rep_shd)
        masks64_j = jax.device_put(jnp.asarray(masks_np, dtype=jnp.int32), rep_shd)
        cot64_j = jax.device_put(jnp.asarray(cot64_np, dtype=jnp.float64), rep_shd)
        smask64_j = deepseek41_vision.compute_image_span_mask(
            grids64_j, span_len_max=t_span_max, downsample_ratio=r, image_masks=masks64_j
        )

        @jax.jit
        def eval_fwd_and_vjp64(
            params: dict[str, Any], x_in: jax.Array
        ) -> tuple[jax.Array, jax.Array, dict[str, Any], jax.Array]:
          def _loss(p: dict[str, Any], inp: jax.Array) -> tuple[jax.Array, jax.Array]:
            mod = mm_testing.apply_with_pure_params(w64_gdef, w64_pstate, p, w64_ostate)
            spans_out, _ = mod(inp, input_masks=masks64_j, deterministic=True, image_grids=grids64_j)
            return jnp.sum(spans_out * (cot64_j * smask64_j[..., None].astype(spans_out.dtype))), spans_out

          (loss_v, spans_v), (p_grads, x_grad) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(params, x_in)
          return loss_v, spans_v, p_grads, x_grad

        c_loss64, c_spans64, c_pgrads64, c_xgrad64 = eval_fwd_and_vjp64(sharded_p64, patches64_j)
        ok_fwd64, m_fwd64 = layer_parity.compare(np.asarray(c_spans64), ref_spans64, layer_parity.FP64_TOLERANCE)
        self.assertTrue(ok_fwd64, msg=f"Mixed-grid fsdp=4 fp64 fwd: {m_fwd64}")
        self.assertLessEqual(abs(float(c_loss64) - ref_loss64), 1e-10 * max(1.0, abs(ref_loss64)))
        for ref_name, spec in weight_map.items():
          cand_leaf64 = _extract_cand_leaf(c_pgrads64, spec)
          ok_g64, m_g64 = layer_parity.compare(cand_leaf64, ref_grads64[ref_name], layer_parity.FP64_TOLERANCE)
          self.assertTrue(ok_g64, msg=f"Mixed-grid fsdp=4 fp64 grad {ref_name}: {m_g64}")
        ok_in64, m_in64 = layer_parity.compare(np.asarray(c_xgrad64), ref_ingrad64, layer_parity.FP64_TOLERANCE)
        self.assertTrue(ok_in64, msg=f"Mixed-grid fsdp=4 fp64 input grad: {m_in64}")


if __name__ == "__main__":
  unittest.main()
