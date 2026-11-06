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

"""Tests for OpenVLA vision layers: MaxText implementation vs. the timm / PyTorch reference.

The HF OpenVLA model (openvla/openvla-7b) builds its fused vision backbone from timm ViTs and takes
the patch tokens of the second-to-last block of each tower; the projector is a plain 3-layer GELU MLP.
"""

import os
import unittest

import pytest

try:
  import timm
  import torch
  from torch import nn

  from tests.utils.multimodal_test_utils import (
      assert_all_close_jax_torch,
      copy_conv2d_weights,
      copy_layernorm_weights,
      create_random_jax_torch,
  )

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="torch or timm not available"),
]

from flax import nnx
import jax.numpy as jnp
import numpy as np
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_REPO_ROOT, MAXTEXT_TEST_ASSETS_ROOT
from maxtext.models.openvla_vision import (
    DINOV2_SPEC,
    SIGLIP_SPEC,
    OpenVLAProjector,
    OpenVLAViT,
    OpenVLAVisionEncoder,
    ViTBlock,
)
from maxtext.multimodal import processor_openvla

base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="openvla-7b",
    use_multimodal=True,
    override_model_config=True,
    base_num_decoder_layers=1,
    attention="dot_product",
    matmul_precision="highest",
    dtype="float32",
    dtype_mm="float32",
    weight_dtype="float32",
    skip_jax_distributed_system=True,
)

TIMM_IDS = {
    "dino": ("vit_large_patch14_reg4_dinov2.lvd142m", DINOV2_SPEC),
    "siglip": ("vit_so400m_patch14_siglip_224", SIGLIP_SPEC),
}

if HAS_TORCH:
  torch.set_grad_enabled(False)


def _np(t):
  return t.detach().cpu().numpy()


def copy_block_weights(torch_block, jax_block: ViTBlock, spec):
  """timm Block -> MaxText ViTBlock (same transforms as the OpenVLA checkpoint conversion hooks)."""
  h, d = spec.num_heads, spec.hidden_size // spec.num_heads
  copy_layernorm_weights(torch_block.norm1, jax_block.norm1)
  copy_layernorm_weights(torch_block.norm2, jax_block.norm2)
  jax_block.attn.qkv.kernel.value = jnp.array(_np(torch_block.attn.qkv.weight).T.reshape(spec.hidden_size, 3, h, d))
  jax_block.attn.qkv.bias.value = jnp.array(_np(torch_block.attn.qkv.bias).reshape(3, h, d))
  jax_block.attn.proj.kernel.value = jnp.array(_np(torch_block.attn.proj.weight).T.reshape(h, d, spec.hidden_size))
  jax_block.attn.proj.bias.value = jnp.array(_np(torch_block.attn.proj.bias))
  for name in ("fc1", "fc2"):
    t, j = getattr(torch_block.mlp, name), getattr(jax_block.mlp, name)
    j.kernel.value = jnp.array(_np(t.weight).T)
    j.bias.value = jnp.array(_np(t.bias))
  if spec.use_layer_scale:
    jax_block.ls1.value = jnp.array(_np(torch_block.ls1.gamma))
    jax_block.ls2.value = jnp.array(_np(torch_block.ls2.gamma))


def copy_vit_weights(torch_vit, jax_vit: OpenVLAViT, spec):
  """timm VisionTransformer -> MaxText OpenVLAViT (first `spec.num_layers` blocks)."""
  copy_conv2d_weights(torch_vit.patch_embed.proj, jax_vit.patch_embed)
  jax_vit.patch_embed.bias.value = jnp.array(_np(torch_vit.patch_embed.proj.bias))
  jax_vit.pos_embed.value = jnp.array(_np(torch_vit.pos_embed))
  if spec.use_cls_token:
    jax_vit.cls_token.value = jnp.array(_np(torch_vit.cls_token))
  if spec.num_reg_tokens:
    jax_vit.reg_token.value = jnp.array(_np(torch_vit.reg_token))
  for i in range(spec.num_layers):
    copy_block_weights(torch_vit.blocks[i], getattr(jax_vit, f"blocks_{i}"), spec)


def make_timm_vit(name):
  timm_id, _ = TIMM_IDS[name]
  model = timm.create_model(timm_id, pretrained=False, num_classes=0, img_size=224).eval()
  # Random LayerScale (timm initializes gamma to 1e-5, which would hide bugs in the residual branches).
  for module in model.modules():
    if hasattr(module, "gamma"):
      module.gamma.data.normal_(0.0, 0.5)
  return model


def timm_second_to_last(model, images_nchw):
  """Prismatic featurizer forward: patch tokens of the second-to-last block, no final norm."""
  return model.get_intermediate_layers(images_nchw, n={len(model.blocks) - 2})[0]


class OpenVLAViTBlockTest(unittest.TestCase):
  """Single transformer block parity (DINOv2 block with LayerScale, SigLIP block without)."""

  def _check(self, name):
    np.random.seed(0)
    torch.manual_seed(0)
    timm_model = make_timm_vit(name)
    _, spec = TIMM_IDS[name]
    jax_block = ViTBlock(jax_config, spec, rngs=nnx.Rngs(0))
    copy_block_weights(timm_model.blocks[0], jax_block, spec)
    x_jax, x_torch = create_random_jax_torch(2, 12, spec.hidden_size)
    assert_all_close_jax_torch(jax_block(x_jax), timm_model.blocks[0](x_torch), rtol=1e-4, atol=1e-4, error_msg=name)

  def test_dino_block(self):
    self._check("dino")

  def test_siglip_block(self):
    self._check("siglip")


class OpenVLAViTTowerTest(unittest.TestCase):
  """Full tower parity: patch embed + pos embed + prefix tokens + blocks, second-to-last block output."""

  def _check(self, name):
    """Compares the full JAX tower against timm (second-to-last block output) on random images."""
    np.random.seed(0)
    torch.manual_seed(0)
    timm_model = make_timm_vit(name)
    _, spec = TIMM_IDS[name]
    jax_vit = OpenVLAViT(jax_config, spec, rngs=nnx.Rngs(0))
    copy_vit_weights(timm_model, jax_vit, spec)
    images = np.random.randn(2, 224, 224, 3).astype(np.float32)
    expected = timm_second_to_last(timm_model, torch.from_numpy(images).permute(0, 3, 1, 2))
    actual = jax_vit(jnp.array(images))
    self.assertEqual(actual.shape, (2, spec.num_patches, spec.hidden_size))
    assert_all_close_jax_torch(actual, expected, rtol=1e-3, atol=1e-3, error_msg=name)

  def test_dino_tower(self):
    self._check("dino")

  def test_siglip_tower(self):
    self._check("siglip")

  def test_fused_encoder_and_projector(self):
    """Channel-stacked [DINOv2 | SigLIP] input -> concatenated patches -> fused GELU MLP projector."""
    np.random.seed(0)
    torch.manual_seed(0)
    dino, siglip = make_timm_vit("dino"), make_timm_vit("siglip")
    encoder = OpenVLAVisionEncoder(jax_config, mesh=None, rngs=nnx.Rngs(0))
    copy_vit_weights(dino, encoder.featurizer, DINOV2_SPEC)
    copy_vit_weights(siglip, encoder.fused_featurizer, SIGLIP_SPEC)

    vision_dim = DINOV2_SPEC.hidden_size + SIGLIP_SPEC.hidden_size
    torch_proj = nn.Sequential(
        nn.Linear(vision_dim, 4 * vision_dim),
        nn.GELU(),
        nn.Linear(4 * vision_dim, jax_config.emb_dim),
        nn.GELU(),
        nn.Linear(jax_config.emb_dim, jax_config.emb_dim),
    ).eval()
    projector = OpenVLAProjector(jax_config, mesh=None, rngs=nnx.Rngs(0))
    for i, idx in enumerate((0, 2, 4)):
      fc = getattr(projector, f"fc{i + 1}")
      fc.kernel.value = jnp.array(_np(torch_proj[idx].weight).T)
      fc.bias.value = jnp.array(_np(torch_proj[idx].bias))

    pixels = np.random.randn(1, 224, 224, 6).astype(np.float32)
    pt = torch.from_numpy(pixels).permute(0, 3, 1, 2)
    patches = torch.cat([timm_second_to_last(dino, pt[:, :3]), timm_second_to_last(siglip, pt[:, 3:])], dim=2)
    expected = torch_proj(patches)
    actual = projector(encoder(jnp.array(pixels)))  # [B, N=1, 256, 4096]
    self.assertEqual(actual.shape, (1, 1, 256, jax_config.emb_dim))
    assert_all_close_jax_torch(actual[:, 0], expected, rtol=2e-3, atol=2e-3)


class OpenVLAProcessorTest(unittest.TestCase):
  """Text/image preprocessing helpers."""

  def test_token_fusion(self):
    pad = processor_openvla.OPENVLA_PAD_TOKEN
    tokens = np.array([1, 512, 29901, 3744, 29901, pad, pad])
    out = processor_openvla.add_extra_tokens_for_images_openvla(tokens)
    n = processor_openvla.OPENVLA_NUM_IMAGE_TOKENS
    self.assertEqual(out.shape[0], tokens.shape[0] + n + 1)
    self.assertEqual(out[0], 1)
    self.assertTrue(np.all(out[1 : 1 + n] == processor_openvla.OPENVLA_IMAGE_PLACEHOLDER))
    np.testing.assert_array_equal(out[1 + n : 1 + n + 4], [512, 29901, 3744, 29901])
    self.assertEqual(out[1 + n + 4], processor_openvla.OPENVLA_EMPTY_TOKEN)
    self.assertTrue(np.all(out[1 + n + 5 :] == pad))
    self.assertEqual(
        processor_openvla.get_image_offsets_openvla(processor_openvla.OpenVLAPreprocessorOutput(num_images=1)), n + 1
    )

  def test_prompt_format(self):
    self.assertEqual(
        processor_openvla.reformat_prompt_openvla("pick up the cup "),
        "In: What action should the robot take to pick up the cup?\nOut:",
    )
    full = "In: What action should the robot take to open the drawer?\nOut:"
    self.assertEqual(processor_openvla.reformat_prompt_openvla(full + " "), full)

  def test_image_preprocessing_matches_hf_golden(self):
    """Compares against `pixel_values` saved from the HF `PrismaticProcessor` (if available)."""
    golden = os.environ.get("OPENVLA_GOLDEN_PKL", "")
    if not golden or not os.path.exists(golden):
      self.skipTest("set OPENVLA_GOLDEN_PKL to a golden file produced by the HF OpenVLA processor")
    import pickle  # pylint: disable=import-outside-toplevel

    with open(golden, "rb") as f:
      data = pickle.load(f)[0]
    image_path = data.get("image_path") or os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "test_image.jpg")
    from maxtext.multimodal import utils as mm_utils  # pylint: disable=import-outside-toplevel

    out = processor_openvla.preprocess_mm_data_openvla([mm_utils.load_image_from_path(image_path)])
    np.testing.assert_allclose(out.pixel_values[0], np.transpose(data["pixel_values"], (1, 2, 0)), rtol=0, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
