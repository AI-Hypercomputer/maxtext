# Copyright 2026 Google LLC
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

"""Tests for the Gemma 4 Unified (12B) encoder-free vision embedder.

These cover the parts that do not need a PyTorch reference: patch/position layout,
output shape, and the parameter names and shapes the checkpoint converter relies on.
Numerical parity against Hugging Face lives in ``gemma4_unified_vision_layers_test.py``.
"""

import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN, AttentionType, VisionEncoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers import encoders
from maxtext.models import gemma4
from maxtext.models.gemma4_vision import Gemma4UnifiedVisionEmbedder, factorized_posemb, patchify
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


def _make_config(**overrides):
  """Initializes a float32 gemma4 config (12b unless overridden) for deterministic comparisons."""
  model_name = overrides.pop("model_name", "gemma4-12b")
  return pyconfig.initialize(
      ["", BASE_CONFIG_PATH],
      model_name=model_name,
      use_multimodal=True,
      matmul_precision="highest",
      dropout_rate=0.0,
      dtype="float32",
      dtype_mm="float32",
      weight_dtype="float32",
      **overrides,
  )


def _make_mesh(config):
  devices = np.array(jax.devices()).reshape([1] * len(config.mesh_axes))
  return jax.sharding.Mesh(devices, config.mesh_axes)


class Gemma4UnifiedPatchifyTest(unittest.TestCase):
  """The embedder relies on `patchify` reproducing HF's 16px-patch + 3x3-merge layout."""

  def test_patchify_at_model_patch_size_matches_teacher_patch_merge(self):
    # HF cuts 16px "teacher" patches then merges them in 3x3 groups. Cutting 48px
    # patches directly must produce the same elements in the same order.
    patch_size, pooling_kernel = 16, 3
    model_patch_size = patch_size * pooling_kernel
    height, width = 96, 144  # 2x3 model patches
    image = np.arange(height * width * 3, dtype=np.float32).reshape(1, height, width, 3)

    merged, positions = patchify(jnp.asarray(image), model_patch_size)

    # NumPy reference for HF's convert_image_to_patches + patches_merge on a full grid.
    num_h, num_w = height // model_patch_size, width // model_patch_size
    reference = image.reshape(1, num_h, model_patch_size, num_w, model_patch_size, 3)
    reference = reference.transpose(0, 1, 3, 2, 4, 5).reshape(1, num_h * num_w, -1)

    np.testing.assert_array_equal(np.asarray(merged), reference)
    self.assertEqual(merged.shape, (1, num_h * num_w, model_patch_size * model_patch_size * 3))
    # Merged positions are the teacher positions floor-divided by the pooling kernel,
    # i.e. a plain (x, y) grid over the model-patch layout.
    expected_xy = np.stack(np.meshgrid(np.arange(num_w), np.arange(num_h)), axis=-1).reshape(-1, 2)
    np.testing.assert_array_equal(np.asarray(positions)[0], expected_xy)

  def test_gemma4_12b_image_size_yields_the_configured_soft_token_count(self):
    config = _make_config()
    height, width = config.image_size_for_vit
    patches, _ = patchify(jnp.zeros((1, height, width, 3)), config.patch_size_for_vit)
    self.assertEqual(patches.shape[1], config.vision_output_length)


class Gemma4UnifiedFactorizedPosembTest(unittest.TestCase):
  """The factorized table is shared with the ViT variants but keeps HF's (N, 2, D) layout."""

  def test_sums_the_x_and_y_rows(self):
    posemb = jnp.asarray(np.random.RandomState(0).randn(5, 2, 4).astype(np.float32))
    positions = jnp.asarray([[[1, 3], [4, 0]]])

    result = factorized_posemb(posemb, positions, jax.lax.Precision.HIGHEST)

    np.testing.assert_allclose(np.asarray(result[0, 0]), np.asarray(posemb[1, 0] + posemb[3, 1]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result[0, 1]), np.asarray(posemb[4, 0] + posemb[0, 1]), rtol=1e-6)

  def test_padding_positions_contribute_nothing(self):
    posemb = jnp.asarray(np.random.RandomState(0).randn(5, 2, 4).astype(np.float32))
    positions = jnp.asarray([[[-1, -1]]])

    result = factorized_posemb(posemb, positions, jax.lax.Precision.HIGHEST)

    np.testing.assert_array_equal(np.asarray(result[0, 0]), np.zeros(4, dtype=np.float32))


class Gemma4UnifiedVisionEmbedderTest(unittest.TestCase):
  """Structure of the embedder itself."""

  def test_config_selects_the_encoder_free_embedder(self):
    config = _make_config()
    self.assertEqual(config.vision_encoder_block, VisionEncoderBlockType.GEMMA4_UNIFIED)
    encoder = encoders.VisionEncoder(config=config, mesh=_make_mesh(config), rngs=nnx.Rngs(params=0))
    self.assertEqual(encoder.encoder_name, "Gemma4UnifiedVisionEmbedder_0")
    # The multimodal head (RMSNorm -> Dense) is shared with the ViT variants.
    self.assertEqual(encoder.projector_name, "Gemma4VisionProjector_0")

  def test_projects_images_into_the_text_embedding_space(self):
    config = _make_config()
    encoder = encoders.VisionEncoder(config=config, mesh=_make_mesh(config), rngs=nnx.Rngs(params=0))
    height, width = config.image_size_for_vit
    images = jnp.asarray(np.random.RandomState(0).rand(2, 1, height, width, 3).astype(np.float32))

    embeddings, deep_features = encoder(images)

    self.assertEqual(embeddings.shape, (2, 1, config.vision_output_length, config.emb_dim))
    self.assertIsNone(deep_features)
    self.assertTrue(np.all(np.isfinite(np.asarray(embeddings))))

  def test_parameter_shapes_match_the_hf_checkpoint_layout(self):
    config = _make_config()
    embedder = Gemma4UnifiedVisionEmbedder(config=config, mesh=_make_mesh(config), rngs=nnx.Rngs(params=0))
    patch_dim = config.patch_size_for_vit**2 * config.num_channels_for_vit

    self.assertEqual(embedder.patch_ln1.scale.value.shape, (patch_dim,))
    self.assertEqual(embedder.patch_ln1.bias.value.shape, (patch_dim,))
    # DenseGeneral stores (in, out); HF stores the transpose, handled by the conversion hook.
    self.assertEqual(embedder.patch_dense.kernel.value.shape, (patch_dim, config.hidden_size_for_vit))
    self.assertEqual(embedder.patch_dense.bias.value.shape, (config.hidden_size_for_vit,))
    self.assertEqual(embedder.patch_ln2.scale.value.shape, (config.hidden_size_for_vit,))
    self.assertEqual(
        embedder.pos_emb_param.value.shape,
        (config.num_position_embeddings_for_vit, 2, config.hidden_size_for_vit),
    )
    self.assertEqual(embedder.pos_norm.scale.value.shape, (config.hidden_size_for_vit,))


class Gemma4BidirectionalMaskTest(unittest.TestCase):
  """Where the bidirectional image block applies differs by Gemma 4 architecture.

  `gemma4` (26b/31b): `Gemma4Model.forward` routes through `create_masks_for_vision_model`,
  which keeps the overlay in sliding layers and leaves global layers causal-only.

  `gemma4_unified` (12b): `Gemma4UnifiedModel.forward` has no such branch — it hands
  `block_sequence_ids` to the generic `create_masks_for_generate`, which ORs the overlay into
  *both* mask types. We follow the reference implementation per architecture, so 12b keeps the
  overlay on global layers. Flipping 12b to the sliding-only rule costs KL ~0.75 and ~50% of
  next-token argmaxes against a stock Hugging Face forward pass.

  These tests pin both behaviours, so a change to either fails CI.
  """

  def _mask_reaching_attention(self, attention_type, mask, model_name="gemma4-12b"):
    """Runs a decoder layer far enough to see what mask self-attention is handed."""
    # Shrunk dims: the gating under test is independent of layer size, and a full-size
    # layer is far too large to build in a unit test.
    config = _make_config(
        model_name=model_name,
        override_model_config=True,
        base_emb_dim=128,
        base_mlp_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=1,
        head_dim=16,
        global_head_dim=32,
        global_num_kv_heads=1,
        base_num_decoder_layers=6,
        run_name="gemma4_bidirectional_mask",
    )
    layer = gemma4.Gemma4DecoderLayer(
        config=config,
        mesh=_make_mesh(config),
        model_mode=MODEL_MODE_TRAIN,
        attention_type=attention_type,
        rngs=nnx.Rngs(params=0),
    )

    seen = {}

    class _Stop(Exception):
      """Ends the forward pass once the mask has been observed."""

    def record(*_args, bidirectional_mask=None, **_kwargs):
      seen["mask"] = bidirectional_mask
      raise _Stop()

    layer.self_attention = record

    seq_len = mask.shape[1]
    inputs = jnp.zeros((1, seq_len, config.emb_dim), dtype=jnp.float32)
    with self.assertRaises(_Stop):
      layer(inputs, None, jnp.arange(seq_len)[None], True, MODEL_MODE_TRAIN, bidirectional_mask=mask)
    return seen["mask"]

  def test_sliding_layers_keep_the_bidirectional_mask(self):
    mask = jnp.ones((1, 4), dtype=jnp.bool_)
    self.assertIs(self._mask_reaching_attention(AttentionType.LOCAL_SLIDING, mask), mask)

  def test_gemma4_unified_global_layers_keep_the_bidirectional_mask(self):
    # 12b follows `Gemma4UnifiedModel.forward`, which overlays both mask types.
    mask = jnp.ones((1, 4), dtype=jnp.bool_)
    self.assertIs(self._mask_reaching_attention(AttentionType.GLOBAL, mask), mask)

  def test_non_unified_global_layers_drop_the_bidirectional_mask(self):
    # 26b/31b follow `create_masks_for_vision_model`: global layers stay causal-only.
    # Guards against the 12b carve-out leaking into the other Gemma 4 variants.
    mask = jnp.ones((1, 4), dtype=jnp.bool_)
    self.assertIsNone(self._mask_reaching_attention(AttentionType.GLOBAL, mask, model_name="gemma4-31b"))


if __name__ == "__main__":
  unittest.main()
