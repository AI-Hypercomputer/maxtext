# Copyright 2025 Google LLC
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

"""Tests for Qwen3 Omni Moe Vision Encoder layers."""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as TorchQwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionMLP as TorchQwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed as TorchQwen3OmniMoeVisionPatchEmbed,
    Qwen3OmniMoeVisionPatchMerger as TorchQwen3OmniMoeVisionPatchMerger,
    apply_rotary_pos_emb_vision,
)

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_REPO_ROOT
from MaxText.layers.embeddings import (
    Qwen3OmniMoeVisionPosEmbedInterpolate as JaxQwen3OmniMoeVisionPosEmbedInterpolate,
    Qwen3OmniMoeVisionRotaryEmbedding as JaxQwen3OmniMoeVisionRotaryEmbedding,
)
from MaxText.layers.qwen3 import (
    Qwen3OmniMoeVisionAttention as JaxQwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionEncoder as JaxQwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionMLP as JaxQwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed as JaxQwen3OmniMoeVisionPatchEmbed,
    Qwen3OmniMoeVisionPatchMerger as JaxQwen3OmniMoeVisionPatchMerger,
    Qwen3OmniMoeVisionProjector as JaxQwen3OmniMoeVisionProjector,
)
from MaxText.multimodal import preprocessor
from tests.multimodal_test_utils import (
    assert_all_close_jax_torch,
    copy_attention_weights_to_maxtext,
    copy_mlp_weights,
    copy_patch_embed_weights,
    copy_patch_merger_weights,
    copy_vision_encoder_weights,
    create_random_jax_torch,
    split_into_patches,
)


base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "configs", "base.yml")
jax_vision_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen3-omni-30b-a3b",
    attention="dot_product",
    attention_type="full",
    matmul_precision="highest",
    dtype="float32",
    dtype_mm="float32",
    weight_dtype="float32",
    float32_logits=True,
    float32_qk_product=True,
)

torch_vision_config = Qwen3OmniMoeVisionEncoderConfig(
    hidden_size=jax_vision_config.hidden_size_for_vit,
    num_heads=jax_vision_config.num_attention_heads_for_vit,
    intermediate_size=jax_vision_config.intermediate_size_for_vit,
    spatial_merge_size=jax_vision_config.spatial_merge_size_for_vit,
    depth=jax_vision_config.num_hidden_layers_for_vit,
    rope_theta=jax_vision_config.rope_theta_for_vit,
    patch_size=jax_vision_config.patch_size_for_vit,
    temporal_patch_size=jax_vision_config.temporal_patch_size_for_vit,
    in_channels=jax_vision_config.num_channels_for_vit,
    num_position_embeddings=jax_vision_config.num_position_embeddings_for_vit,
    out_hidden_size=jax_vision_config.out_hidden_size_for_vit,
    deepstack_visual_indexes=list(jax_vision_config.deepstack_visual_indexes_for_vit),
    hidden_act="gelu_pytorch_tanh",
)
torch_vision_config._attn_implementation = "eager"  # pylint: disable=protected-access

torch.set_grad_enabled(False)


def create_torch_encoder():
  """Create and configure PyTorch vision encoder."""
  encoder = TorchQwen3OmniMoeVisionEncoder(torch_vision_config)
  encoder.eval()
  return encoder


def setup_test_seeds():
  """Set random seeds for reproducibility."""
  np.random.seed(42)
  torch.manual_seed(42)


class BaseVisionTestCase(unittest.TestCase):
  """Base class for vision tests with common setup."""

  def setUp(self):
    self.config = jax_vision_config
    setup_test_seeds()


class BaseVisionTestCaseWithMesh(BaseVisionTestCase):
  """Base class for vision tests that require mesh setup."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))


class TestQwen3OmniMoeVisionAttention(BaseVisionTestCaseWithMesh):
  """Test cases for Qwen3 Omni Moe Vision Attention layer."""

  def setUp(self):
    super().setUp()
    self.seq_length = 16
    self.hidden_size = self.config.hidden_size_for_vit
    self.num_heads = self.config.num_attention_heads_for_vit

  def test_attention_output_matches_torch(self):
    """Test that JAX vision attention output matches PyTorch implementation."""
    torch_encoder = create_torch_encoder()
    torch_model = torch_encoder.blocks[0].attn

    jax_model = JaxQwen3OmniMoeVisionAttention(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42))

    copy_attention_weights_to_maxtext(torch_model, jax_model.attn, fused_qkv=True)

    jax_hidden_states_2d, torch_hidden_states = create_random_jax_torch(self.seq_length, self.hidden_size)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32)

    cu_seqlens = torch.tensor([0, self.seq_length], dtype=torch.int32)

    # Compute rotary position embeddings for PyTorch
    rotary_pos_emb = torch_encoder.rot_pos_emb(grid_thw)
    rotary_pos_emb = rotary_pos_emb.reshape(self.seq_length, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    torch_output = torch_model(
        torch_hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )

    jax_hidden_states_3d = jax_hidden_states_2d[jnp.newaxis, :, :]
    jax_output = jax_model(
        jax_hidden_states_3d,  # Shape: (1, seq_len, hidden_size)
        num_frames=1,
        height=4,
        width=4,
        deterministic=True,
    )
    jax_output = jax_output[0]

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-2,
        atol=1e-2,
        error_msg="Vision attention outputs differ",
    )

  def test_attention_is_jittable(self):
    """Test that attention is JIT-compilable."""
    model = JaxQwen3OmniMoeVisionAttention(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42))
    hidden_states = jnp.ones((1, 16, self.hidden_size))

    @nnx.jit
    def forward(model, hidden_states):
      return model(hidden_states, num_frames=1, height=4, width=4, deterministic=True)

    _ = forward(model, hidden_states)


class TestQwen3OmniMoeVisionPatchMerger(BaseVisionTestCase):
  """Test cases for Qwen3 Omni Moe Vision Patch Merger layer."""

  def _test_patch_merger_with_postshuffle(self, use_postshuffle_norm):
    """Helper method to test patch merger with/without postshuffle_norm."""
    torch_model = TorchQwen3OmniMoeVisionPatchMerger(torch_vision_config, use_postshuffle_norm=use_postshuffle_norm)
    torch_model.eval()

    jax_model = JaxQwen3OmniMoeVisionPatchMerger(
        config=self.config,
        use_postshuffle_norm=use_postshuffle_norm,
        rngs=nnx.Rngs(42),
    )

    copy_patch_merger_weights(torch_model, jax_model)

    batch_size = 2
    seq_len = 64
    jax_hidden_states, torch_hidden_states = create_random_jax_torch(
        batch_size * seq_len, self.config.hidden_size_for_vit
    )

    jax_hidden_states = jax_hidden_states.reshape(batch_size, seq_len, self.config.hidden_size_for_vit)
    torch_output = torch_model(torch_hidden_states)
    jax_output = jax_model(jax_hidden_states)
    jax_output = jax_output.reshape(-1, jax_output.shape[-1])

    assert_all_close_jax_torch(jax_output, torch_output, rtol=1e-3, atol=3e-3)

  def test_patch_merger_output_matches_torch_without_postshuffle(self):
    """Test patch merger without postshuffle_norm matches PyTorch."""
    self._test_patch_merger_with_postshuffle(use_postshuffle_norm=False)

  def test_patch_merger_output_matches_torch_with_postshuffle(self):
    """Test patch merger with postshuffle_norm matches PyTorch."""
    self._test_patch_merger_with_postshuffle(use_postshuffle_norm=True)

  def test_patch_merger_is_jittable(self):
    """Test that patch merger is JIT-compilable."""
    model = JaxQwen3OmniMoeVisionPatchMerger(config=self.config, use_postshuffle_norm=False, rngs=nnx.Rngs(42))

    @nnx.jit
    def forward(model, hidden_states):
      return model(hidden_states)

    batch_size = 2
    seq_len = 64
    hidden_states = jnp.ones((batch_size, seq_len, self.config.hidden_size_for_vit))
    forward(model, hidden_states)


class TestQwen3OmniMoeVisionMLP(BaseVisionTestCase):
  """Test cases for Qwen3 Omni Moe Vision MLP layer."""

  def setUp(self):
    super().setUp()
    self.torch_model = TorchQwen3OmniMoeVisionMLP(torch_vision_config)
    self.torch_model.eval()
    self.jax_model = JaxQwen3OmniMoeVisionMLP(config=self.config, rngs=nnx.Rngs(42))
    copy_mlp_weights(self.torch_model, self.jax_model)

  def test_mlp_output_matches_torch(self):
    """Test that JAX MLP output matches PyTorch implementation."""
    # Create test input
    seq_len = 16
    jax_hidden_states, torch_hidden_states = create_random_jax_torch(seq_len, self.config.hidden_size_for_vit)

    # Forward pass
    torch_output = self.torch_model(torch_hidden_states)
    jax_output = self.jax_model(jax_hidden_states)

    # Compare outputs
    assert_all_close_jax_torch(jax_output, torch_output, rtol=1e-4, atol=3e-3)

  def test_mlp_is_jittable(self):
    """Test that MLP is JIT-compilable."""

    @nnx.jit
    def forward(model, hidden_states):
      return model(hidden_states)

    hidden_states = jnp.ones((16, self.config.hidden_size_for_vit))
    output = forward(self.jax_model, hidden_states)

    self.assertEqual(output.shape, (16, self.config.hidden_size_for_vit))


class TestQwen3OmniMoeVisionPatchEmbed(BaseVisionTestCase):
  """Test cases for Qwen3 Omni Moe Vision Patch Embed layer."""

  def setUp(self):
    super().setUp()
    self.torch_model = TorchQwen3OmniMoeVisionPatchEmbed(torch_vision_config)
    self.torch_model.eval()
    self.jax_model = JaxQwen3OmniMoeVisionPatchEmbed(config=self.config, rngs=nnx.Rngs(42))
    copy_patch_embed_weights(self.torch_model, self.jax_model)

  def test_patch_embed_output_matches_torch(self):
    """Test that JAX patch embed output matches PyTorch implementation."""
    batch_size = 2
    total_elements = (
        batch_size
        * self.config.num_channels_for_vit
        * self.config.temporal_patch_size_for_vit
        * self.config.patch_size_for_vit
        * self.config.patch_size_for_vit
    )
    jax_hidden_states, torch_hidden_states = create_random_jax_torch(total_elements)

    # Reshape JAX input to proper 5D shape: (batch, in_channels, temporal, height, width)
    jax_hidden_states = jax_hidden_states.reshape(
        batch_size,
        self.config.num_channels_for_vit,
        self.config.temporal_patch_size_for_vit,
        self.config.patch_size_for_vit,
        self.config.patch_size_for_vit,
    )

    torch_output = self.torch_model(torch_hidden_states)
    jax_output = self.jax_model(jax_hidden_states)

    torch_output_squeezed = torch_output.squeeze(1)
    jax_output_squeezed = jax_output.squeeze(1)

    assert_all_close_jax_torch(jax_output_squeezed, torch_output_squeezed, rtol=1e-3, atol=5e-3)

  def test_patch_embed_is_jittable(self):
    """Test that patch embed is JIT-compilable."""

    @nnx.jit
    def forward(model, hidden_states):
      return model(hidden_states)

    batch_size = 2

    # Patch embed expects 5D input: (batch, in_channels, temporal, height, width)
    hidden_states = jnp.ones(
        (
            batch_size,
            self.config.num_channels_for_vit,
            self.config.temporal_patch_size_for_vit,
            self.config.patch_size_for_vit,
            self.config.patch_size_for_vit,
        )
    )
    forward(self.jax_model, hidden_states)


class TestQwen3OmniMoeVisionRotaryEmbedding(BaseVisionTestCase):
  """Test the grid-based rotary embedding from embeddings.py against PyTorch."""

  def setUp(self):
    super().setUp()
    self.jax_model = JaxQwen3OmniMoeVisionRotaryEmbedding(
        hidden_size=self.config.hidden_size_for_vit,
        num_attention_heads=self.config.num_attention_heads_for_vit,
        spatial_merge_size=self.config.spatial_merge_size_for_vit,
        rope_theta=self.config.rope_theta_for_vit,
        cast_as_fprop_dtype=False,
        fprop_dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )
    self.torch_encoder = create_torch_encoder()

  def _create_jax_rotary_model(self):
    """Helper to create JAX rotary embedding model."""
    return JaxQwen3OmniMoeVisionRotaryEmbedding(
        hidden_size=self.config.hidden_size_for_vit,
        num_attention_heads=self.config.num_attention_heads_for_vit,
        spatial_merge_size=self.config.spatial_merge_size_for_vit,
        rope_theta=self.config.rope_theta_for_vit,
        cast_as_fprop_dtype=False,
        fprop_dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )

  def test_grid_based_embedding_matches_torch(self):
    """Test that JAX grid-based rotary embedding matches PyTorch implementation."""
    num_frames, height, width = 1, 8, 8
    grid_thw_np = np.array([[num_frames, height, width]], dtype=np.int64)
    grid_thw_torch = torch.from_numpy(grid_thw_np)

    cos_emb_jax, sin_emb_jax = self.jax_model.compute_cos_sin(num_frames, height, width)

    rotary_pos_emb = self.torch_encoder.rot_pos_emb(grid_thw_torch)
    embeddings = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
    cos_emb_torch = embeddings.cos()
    sin_emb_torch = embeddings.sin()

    assert_all_close_jax_torch(cos_emb_jax, cos_emb_torch, rtol=1e-5, atol=1e-5)
    assert_all_close_jax_torch(sin_emb_jax, sin_emb_torch, rtol=1e-5, atol=1e-5)

  def test_rotation_application_matches_torch(self):
    """Test that applying rotary embedding to Q/K tensors matches PyTorch."""
    head_dim = self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit

    num_frames, height, width = 1, 8, 8
    grid_thw_np = np.array([[num_frames, height, width]], dtype=np.int64)
    grid_thw_torch = torch.from_numpy(grid_thw_np)

    seq_len = 64
    q_jax, q_torch = create_random_jax_torch(seq_len, self.config.num_attention_heads_for_vit, head_dim)
    k_jax, k_torch = create_random_jax_torch(seq_len, self.config.num_attention_heads_for_vit, head_dim)

    q_rotated_jax = self.jax_model(q_jax, num_frames, height, width)
    k_rotated_jax = self.jax_model(k_jax, num_frames, height, width)

    rotary_pos_emb = self.torch_encoder.rot_pos_emb(grid_thw_torch)
    embeddings = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
    cos = embeddings.cos()  # [seq_len, head_dim]
    sin = embeddings.sin()  # [seq_len, head_dim]

    q_rotated_torch, k_rotated_torch = apply_rotary_pos_emb_vision(q_torch, k_torch, cos, sin)

    assert_all_close_jax_torch(
        q_rotated_jax,
        q_rotated_torch,
        rtol=1e-3,
        atol=1e-4,
        error_msg="Q rotation mismatch",
    )
    assert_all_close_jax_torch(
        k_rotated_jax,
        k_rotated_torch,
        rtol=1e-3,
        atol=1e-4,
        error_msg="K rotation mismatch",
    )


class TestQwen3OmniMoeVisionPosEmbedInterpolate(BaseVisionTestCase):
  """Test bilinear position embedding interpolation from embeddings.py."""

  def setUp(self):
    super().setUp()
    self.jax_model = JaxQwen3OmniMoeVisionPosEmbedInterpolate(
        num_position_embeddings=self.config.num_position_embeddings_for_vit,
        hidden_size=self.config.hidden_size_for_vit,
        spatial_merge_size=self.config.spatial_merge_size_for_vit,
        dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )
    self.torch_encoder = create_torch_encoder()
    torch_pos_embed_weight = self.torch_encoder.pos_embed.weight.detach().cpu().numpy()
    self.jax_model.pos_embed.value = jnp.array(torch_pos_embed_weight)

  def _create_jax_pos_embed_model(self):
    """Helper to create JAX position embedding model."""
    return JaxQwen3OmniMoeVisionPosEmbedInterpolate(
        num_position_embeddings=self.config.num_position_embeddings_for_vit,
        hidden_size=self.config.hidden_size_for_vit,
        spatial_merge_size=self.config.spatial_merge_size_for_vit,
        dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )

  def _copy_weights_and_test(self, num_frames, height, width):
    """Helper to copy weights and test position embedding interpolation."""
    grid_thw_np = np.array([[num_frames, height, width]], dtype=np.int64)
    grid_thw_torch = torch.from_numpy(grid_thw_np)

    pos_embed_jax = self.jax_model(num_frames, height, width)
    pos_embed_torch = self.torch_encoder.fast_pos_embed_interpolate(grid_thw_torch)

    assert_all_close_jax_torch(pos_embed_jax, pos_embed_torch, rtol=1e-2, atol=1e-2)

  def test_pos_embed_interpolate_matches_torch(self):
    """Test that JAX position embedding interpolation matches PyTorch encoder."""
    self._copy_weights_and_test(num_frames=1, height=16, width=16)

  def test_pos_embed_interpolate_multiple_images(self):
    """Test position embedding interpolation with multiple images/videos."""
    self._copy_weights_and_test(num_frames=1, height=8, width=8)


class TestQwen3OmniMoeVisionEncoderEndToEnd(BaseVisionTestCaseWithMesh):
  """End-to-end test for the full vision encoder."""

  def test_vision_encoder_single_image(self):
    """Test full vision encoder with single image matches PyTorch."""
    torch_encoder = create_torch_encoder()

    jax_encoder = JaxQwen3OmniMoeVisionEncoder(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(42))
    jax_projector = JaxQwen3OmniMoeVisionProjector(config=self.config, rngs=nnx.Rngs(43))

    copy_vision_encoder_weights(torch_encoder, jax_encoder)
    copy_patch_merger_weights(torch_encoder.merger, jax_projector.merger)

    patch_size = self.config.patch_size_for_vit
    temporal_patch_size = self.config.temporal_patch_size_for_vit
    in_channels = self.config.num_channels_for_vit
    h, w = 8, 8  # 8x8 patches

    total_elements = 1 * in_channels * temporal_patch_size * (h * patch_size) * (w * patch_size)
    jax_hidden_states, _ = create_random_jax_torch(total_elements)

    jax_hidden_states = jax_hidden_states.reshape(1, in_channels, temporal_patch_size, h * patch_size, w * patch_size)

    torch_hidden_states = split_into_patches(
        torch.from_numpy(np.array(jax_hidden_states)),
        temporal_patch_size,
        patch_size,
    )

    grid_thw = np.array([[1, h, w]], dtype=np.int64)
    grid_thw_torch = torch.from_numpy(grid_thw)

    torch_output, torch_deep_feats = torch_encoder(torch_hidden_states, grid_thw_torch)
    jax_encoder_output, jax_deep_feats = jax_encoder(jax_hidden_states)
    jax_output = jax_projector(jax_encoder_output)

    jax_output = jax_output[0]
    jax_deep_feats = [feat[0] for feat in jax_deep_feats]

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-2,
        atol=1e-2,
        error_msg="Vision encoder final output differs",
    )

    # Compare deep features
    self.assertEqual(
        len(jax_deep_feats),
        len(torch_deep_feats),
        "Number of deep features should match",
    )
    for i, (jax_feat, torch_feat) in enumerate(zip(jax_deep_feats, torch_deep_feats)):
      assert_all_close_jax_torch(
          jax_feat,
          torch_feat,
          rtol=1e-2,
          atol=1e-2,
          error_msg=f"Deep feature {i} differs",
      )


class TextQwen3OmniPreprocessing(unittest.TestCase):
  """Test MaxText Qwen3 Omni preprocessor against HuggingFace reference."""

  def setUp(self):
    self.base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "configs", "base.yml")
    self.image_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "test_assets", "test_image.jpg")
    self.video_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "test_assets", "test_video.mp4")
    self.maxtext_config = pyconfig.initialize(
        ["", self.base_config_path],
        model_name="qwen3-omni-30b-a3b",
        use_multimodal=True,
        image_path=self.image_path,
        video_path=self.video_path,
        use_audio_in_video=True,
    )

  def test_preprocess_mm_data(self):
    # MaxText preprocessor
    mt_processor_outputs = preprocessor.preprocess_mm_data(self.maxtext_config)

    # HuggingFace preprocessor
    from transformers import Qwen3OmniMoeProcessor  # pylint: disable=import-outside-toplevel
    from qwen_omni_utils import process_mm_info  # pylint: disable=import-outside-toplevel

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": self.image_path},
                {"type": "video", "video": self.video_path},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ]
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    hf_processor_outputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )

    # Add assertions to check the output
    self.assertIsNotNone(mt_processor_outputs)
    assert np.allclose(
        mt_processor_outputs.pixel_values,
        np.array(hf_processor_outputs["pixel_values"]).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert np.allclose(
        mt_processor_outputs.video_values,
        np.array(hf_processor_outputs["pixel_values_videos"]).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert np.allclose(
        mt_processor_outputs.audio_values,
        np.array(hf_processor_outputs["input_features"]).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  unittest.main()
