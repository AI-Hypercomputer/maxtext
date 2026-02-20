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

"""Tests for Qwen3 Omni layers comparing MaxText implementation against PyTorch reference.

This module tests both vision and audio encoder components.
"""

import math
import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText import common_types
from MaxText import pyconfig
from MaxText.globals import MAXTEXT_REPO_ROOT
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import (
    PositionalEmbedding,
    Qwen3OmniMoeVisionPosEmbedInterpolate as JaxQwen3OmniMoeVisionPosEmbedInterpolate,
    Qwen3OmniMoeVisionRotaryEmbedding as JaxQwen3OmniMoeVisionRotaryEmbedding,
)
from maxtext.layers.encoders import AudioEncoder
from maxtext.models.qwen3 import (
    Qwen3OmniAudioEncoder,
    Qwen3OmniAudioEncoderLayer,
    Qwen3OmniMoeVisionAttention as JaxQwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionEncoder as JaxQwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionMLP as JaxQwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed as JaxQwen3OmniMoeVisionPatchEmbed,
    Qwen3OmniMoeVisionPatchMerger as JaxQwen3OmniMoeVisionPatchMerger,
    Qwen3OmniMoeVisionProjector as JaxQwen3OmniMoeVisionProjector,
)
from maxtext.multimodal import processor as mm_processor
from tests.utils.multimodal_test_utils import (
    assert_all_close_jax_torch,
    copy_attention_weights_to_maxtext,
    copy_audio_projector_weights,
    copy_maxtext_audio_encoder_weights,
    copy_maxtext_encoder_layer_weights,
    copy_mlp_weights,
    copy_patch_embed_weights,
    copy_patch_merger_weights,
    copy_vision_encoder_weights,
    create_block_diagonal_attention_mask,
    create_random_jax_torch,
    split_into_patches,
)
import numpy as np
import torch
import torch.nn.functional as F
# Vision encoder imports from transformers
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention as TorchQwen3OmniMoeAudioAttention,
    Qwen3OmniMoeAudioEncoderLayer as TorchQwen3OmniMoeAudioEncoderLayer,
    Qwen3OmniMoeAudioEncoder as TorchQwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder as TorchQwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionMLP as TorchQwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed as TorchQwen3OmniMoeVisionPatchEmbed,
    Qwen3OmniMoeVisionPatchMerger as TorchQwen3OmniMoeVisionPatchMerger,
    SinusoidsPositionEmbedding as TorchSinusoidsPositionEmbedding,
    apply_rotary_pos_emb_vision,
)

# Initialize config once for all tests
base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
jax_config = pyconfig.initialize(
    ["", base_config_path],
    model_name="qwen3-omni-30b-a3b",
    attention="dot_product",
    attention_type="full",
    matmul_precision="highest",
    dropout_rate=0.0,
    dtype="float32",
    dtype_mm="float32",
    weight_dtype="float32",
    float32_logits=True,
    float32_qk_product=True,
)

# PyTorch vision encoder config
torch_vision_config = Qwen3OmniMoeVisionEncoderConfig(
    hidden_size=jax_config.hidden_size_for_vit,
    num_heads=jax_config.num_attention_heads_for_vit,
    intermediate_size=jax_config.intermediate_size_for_vit,
    spatial_merge_size=jax_config.spatial_merge_size_for_vit,
    depth=jax_config.num_hidden_layers_for_vit,
    rope_theta=jax_config.rope_theta_for_vit,
    patch_size=jax_config.patch_size_for_vit,
    temporal_patch_size=jax_config.temporal_patch_size_for_vit,
    in_channels=jax_config.num_channels_for_vit,
    num_position_embeddings=jax_config.num_position_embeddings_for_vit,
    out_hidden_size=jax_config.out_hidden_size_for_vit,
    deepstack_visual_indexes=list(jax_config.deepstack_visual_indexes_for_vit),
    hidden_act="gelu_pytorch_tanh",
)
torch_vision_config._attn_implementation = "eager"  # pylint: disable=protected-access

# PyTorch audio encoder config
torch_audio_encoder_config = Qwen3OmniMoeAudioEncoderConfig(
    d_model=jax_config.d_model_for_audio,
    encoder_attention_heads=jax_config.encoder_attention_heads_for_audio,
    encoder_ffn_dim=jax_config.encoder_ffn_dim_for_audio,
    encoder_layers=jax_config.encoder_layers_for_audio,
    attention_dropout=jax_config.attention_dropout_for_audio,
    dropout=0.0,
    activation_dropout=0.0,
    activation_function="gelu",
    num_mel_bins=jax_config.num_mel_bins_for_audio,
    max_source_positions=jax_config.max_source_positions_for_audio,
    scale_embedding=True,
    n_window=jax_config.n_window_for_audio,
    n_window_infer=jax_config.n_window_infer_for_audio,
    conv_chunksize=jax_config.conv_chunksize_for_audio,
    downsample_hidden_size=jax_config.downsample_hidden_size_for_audio,
    output_dim=jax_config.output_dim_for_audio,
    torch_dtype=torch.float32,
    weight_dtype=torch.float32,
)
torch_audio_encoder_config._attn_implementation = "eager"  # pylint: disable=protected-access

torch.set_grad_enabled(False)


def create_torch_vision_encoder():
  """Create and configure PyTorch vision encoder."""
  encoder = TorchQwen3OmniMoeVisionEncoder(torch_vision_config)
  encoder.eval()
  return encoder


def setup_test_seeds():
  """Set random seeds for reproducibility."""
  np.random.seed(42)
  torch.manual_seed(42)


# =============================================================================
# Vision Encoder Tests
# =============================================================================


class BaseVisionTestCase(unittest.TestCase):
  """Base class for vision tests with common setup."""

  def setUp(self):
    self.config = jax_config
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
    torch_encoder = create_torch_vision_encoder()
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
    self.torch_encoder = create_torch_vision_encoder()

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
    self.torch_encoder = create_torch_vision_encoder()
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
    torch_encoder = create_torch_vision_encoder()

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


class TestQwen3OmniPreprocessing(unittest.TestCase):
  """Test MaxText Qwen3 Omni preprocessor against HuggingFace reference."""

  def setUp(self):
    self.base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    self.image_path = os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets", "test_image.jpg")
    self.video_path = os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets", "test_video.mp4")
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
    mt_processor_outputs = mm_processor.preprocess_mm_data(self.maxtext_config)

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
        rtol=5e-2,
        atol=5e-2,
    )
    assert np.allclose(
        mt_processor_outputs.audio_values,
        np.array(hf_processor_outputs["input_features"]).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


# =============================================================================
# Audio Encoder Tests
# =============================================================================


class TestMaxTextAudioAttentionVsPyTorch(unittest.TestCase):
  """Test that MaxText's Attention module matches PyTorch's audio attention implementation."""

  def setUp(self):
    self.batch_size = 1
    self.seq_length = 16
    self.config = jax_config
    self.embed_dim = self.config.d_model_for_audio
    self.num_heads = self.config.encoder_attention_heads_for_audio
    self.head_dim = self.embed_dim // self.num_heads
    np.random.seed(42)
    torch.manual_seed(42)
    self.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))

  def test_attention_output_matches_torch(self):
    """Test that MaxText Attention produces same output as PyTorch attention."""
    torch_config = torch_audio_encoder_config
    torch_model = TorchQwen3OmniMoeAudioAttention(torch_config)
    torch_model.eval()

    # Create input - PyTorch expects (seq_length, channels), MaxText expects (batch, seq, channels)
    jax_hidden_states_2d, torch_hidden_states = create_random_jax_torch(self.seq_length, self.embed_dim)
    jax_hidden_states = jax_hidden_states_2d[jnp.newaxis, :, :]  # Add batch dimension for MaxText

    # Create cu_seqlens for PyTorch (cumulative sequence lengths)
    cu_seqlens = torch.tensor([0, self.seq_length], dtype=torch.long)

    jax_attn = Attention(
        config=self.config,
        num_query_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        head_dim=self.head_dim,
        max_target_length=self.config.max_source_positions_for_audio,
        attention_kernel="dot_product",
        inputs_q_shape=(
            self.config.per_device_batch_size,
            self.seq_length,
            self.embed_dim,
        ),
        inputs_kv_shape=(
            self.config.per_device_batch_size,
            self.seq_length,
            self.embed_dim,
        ),
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=self.mesh,
        dropout_rate=0.0,
        name="test_attention",
        attention_type=common_types.AttentionType.FULL,
        is_nope_layer=True,
        use_bias_in_projections=True,
        use_qk_norm=False,
        query_pre_attn_scalar=1 / math.sqrt(self.head_dim),
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(42),
    )

    copy_attention_weights_to_maxtext(torch_model, jax_attn)
    torch_output = torch_model(torch_hidden_states, cu_seqlens=cu_seqlens)

    jax_output, _ = jax_attn(inputs_q=jax_hidden_states, inputs_kv=jax_hidden_states, deterministic=True)

    # Both should be (seq, embed) after removing batch dimensions
    jax_output_2d = jax_output[0]  # (batch, seq, embed) -> (seq, embed)
    # PyTorch returns (batch, seq, embed), squeeze to remove batch dimension
    torch_output_2d = torch_output.squeeze(0)  # (1, seq, embed) -> (seq, embed)

    assert_all_close_jax_torch(
        jax_output_2d,
        torch_output_2d,
        rtol=1e-5,
        atol=5e-3,
        error_msg="Attention outputs differ",
    )


class TestAudioEncoderLayer(unittest.TestCase):
  """Test MaxText AudioEncoderLayer against PyTorch implementation."""

  def setUp(self):
    self.config = jax_config
    self.torch_config = torch_audio_encoder_config
    np.random.seed(42)
    torch.manual_seed(42)

    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

  def _test_encoder_layer_with_batch_size(self, batch_size):
    """Helper function to test encoder layer with a given batch size."""

    torch_layer = TorchQwen3OmniMoeAudioEncoderLayer(self.torch_config)
    torch_layer.eval()

    maxtext_layer = Qwen3OmniAudioEncoderLayer(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))

    # Copy weights from PyTorch to MaxText
    copy_maxtext_encoder_layer_weights(torch_layer, maxtext_layer)

    # Create test input
    seq_len = 12  # After conv layers
    hidden_size = self.config.d_model_for_audio

    jax_input, torch_input_3d = create_random_jax_torch(batch_size, seq_len, hidden_size)

    # PyTorch forward pass - expects 2D input (total_seq_len, hidden_dim) with cu_seqlens
    torch_input_2d = torch_input_3d.reshape(-1, hidden_size)

    # Create cu_seqlens for PyTorch (cumulative sequence lengths for each batch)
    # For batch_size=2, seq_len=12: [0, 12, 24] indicates two sequences of length 12 each
    cu_seqlens = torch.tensor([i * seq_len for i in range(batch_size + 1)], dtype=torch.int32)

    attention_mask = create_block_diagonal_attention_mask(cu_seqlens, torch_input_2d.dtype)

    torch_output_1d = torch_layer(torch_input_2d, cu_seqlens=cu_seqlens, attention_mask=attention_mask)[0]
    torch_output = torch_output_1d.reshape(batch_size, seq_len, hidden_size)

    jax_output = maxtext_layer(jax_input, deterministic=True)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-5,
        atol=5e-3,
        error_msg="AudioEncoderLayer outputs differ",
    )

  def test_encoder_layer_matches_torch_batch_1(self):
    """Test that MaxText AudioEncoderLayer matches PyTorch with batch_size=1."""
    self._test_encoder_layer_with_batch_size(batch_size=1)

  def test_encoder_layer_matches_torch_batch_2(self):
    """Test that MaxText AudioEncoderLayer matches PyTorch with batch_size=2."""
    self._test_encoder_layer_with_batch_size(batch_size=2)

  def test_encoder_layer_is_jittable(self):
    """Test that encoder layer can be JIT compiled."""
    with self.mesh:
      jax_layer = Qwen3OmniAudioEncoderLayer(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))

    @nnx.jit
    def forward(layer, x):
      return layer(x, deterministic=True)

    batch_size = 2
    seq_len = 12
    hidden_size = self.config.d_model_for_audio

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size))
    output = forward(jax_layer, hidden_states)

    self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))


class TestPositionalEmbedding(unittest.TestCase):
  """Tests for PositionalEmbedding implementation."""

  def setUp(self):
    self.length = 100
    self.channels = 512
    self.max_timescale = 10000.0
    np.random.seed(42)
    torch.manual_seed(42)

  def test_positional_embedding_matches_torch(self):
    torch_model = TorchSinusoidsPositionEmbedding(self.length, self.channels, self.max_timescale)
    jax_model = PositionalEmbedding(
        embedding_dims=self.channels, max_wavelength=self.max_timescale, cast_as_fprop_dtype=False
    )

    # Test full sequence
    torch_output = torch_model(self.length)
    jax_output = jax_model(self.length)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-5,
        atol=3e-4,
        error_msg="Positional embedding outputs differ",
    )

  def test_positional_embedding_is_jittable(self):
    model = PositionalEmbedding(embedding_dims=self.channels, max_wavelength=self.max_timescale)

    @nnx.jit(static_argnames=["seqlen"])
    def forward(model, seqlen):
      return model(seqlen)

    output = forward(model, seqlen=self.length)
    self.assertEqual(output.shape, (self.length, self.channels))


class TestAudioEncoder(unittest.TestCase):
  """Test AudioEncoder (convs + transformer, no projector) against PyTorch implementation."""

  def setUp(self):
    self.config = jax_config
    self.torch_config = torch_audio_encoder_config
    np.random.seed(42)
    torch.manual_seed(42)

    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

  def test_audio_encoder_matches_torch(self):
    """Test that MaxText AudioEncoder matches PyTorch encoder (convs + transformer + layernorm, before projector)."""
    torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
    torch_model.eval()

    maxtext_encoder = Qwen3OmniAudioEncoder(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))

    copy_maxtext_audio_encoder_weights(torch_model, maxtext_encoder, self.config)

    batch_size = 1
    num_mel_bins = self.config.num_mel_bins_for_audio
    audio_length = 200  # n_window=50, chunk_size=100, gives 2 chunks

    jax_audio_features, torch_audio_features_3d = create_random_jax_torch(batch_size, num_mel_bins, audio_length)

    # PyTorch forward (manually run convs + transformer encoder without projector)
    torch_audio_features = torch_audio_features_3d[0]

    # Run through PyTorch convs + positional + encoder
    chunk_size = self.torch_config.n_window * 2
    num_chunks = audio_length // chunk_size
    chunk_lengths = torch.tensor([chunk_size] * num_chunks, dtype=torch.long)
    chunk_list = torch_audio_features.T.split(chunk_lengths.tolist(), dim=0)
    torch_padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    torch_padded_feature = torch_padded_feature.unsqueeze(1)

    torch_conv1 = F.gelu(torch_model.conv2d1(torch_padded_feature))
    torch_conv2 = F.gelu(torch_model.conv2d2(torch_conv1))
    torch_conv3 = F.gelu(torch_model.conv2d3(torch_conv2))

    b, c, f, t = torch_conv3.size()
    torch_conv_out = torch_model.conv_out(torch_conv3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    torch_pos_emb = (
        torch_model.positional_embedding.positional_embedding[: torch_conv_out.shape[1], :]
        .unsqueeze(0)
        .to(torch_conv_out.dtype)
    )
    torch_after_pos = torch_conv_out + torch_pos_emb

    # Run through encoder layers + layernorm (but not projector)
    # Process all chunks together
    seq_len_per_chunk = torch_after_pos.shape[1]
    cu_seqlens = torch.tensor([i * seq_len_per_chunk for i in range(num_chunks + 1)], dtype=torch.int32)
    attention_mask = create_block_diagonal_attention_mask(cu_seqlens, torch_after_pos.dtype)

    # Flatten: (num_chunks, seq_len_per_chunk, hidden) -> (num_chunks*seq_len_per_chunk, hidden)
    hidden_state = torch_after_pos.reshape(-1, torch_after_pos.shape[-1])
    for layer in torch_model.layers:
      hidden_state = layer(hidden_state, cu_seqlens=cu_seqlens, attention_mask=attention_mask)[0]
    hidden_state = torch_model.ln_post(hidden_state)

    # Reshape back: (num_chunks*seq_len_per_chunk, hidden) -> (batch=1, num_chunks*seq_len_per_chunk, hidden)
    torch_output = hidden_state.reshape(1, num_chunks * seq_len_per_chunk, -1)

    # MaxText forward
    jax_output = maxtext_encoder(jax_audio_features, deterministic=True)

    assert_all_close_jax_torch(
        jax_output,
        torch_output,
        rtol=1e-3,
        atol=0.1,
        error_msg="AudioEncoder outputs differ",
    )


class TestAudioModel(unittest.TestCase):
  """Test full AudioModel end-to-end against PyTorch implementation."""

  def setUp(self):
    self.config = jax_config
    self.torch_config = torch_audio_encoder_config
    np.random.seed(42)
    torch.manual_seed(42)

    devices = jax.devices()
    self.mesh = Mesh(np.array(devices[:1]), axis_names=("data",))

  def test_audio_model_end_to_end(self):
    """Test full AudioModel pipeline against PyTorch."""
    torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
    torch_model.eval()

    maxtext_model = AudioEncoder(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))
    encoder = getattr(maxtext_model, maxtext_model.encoder_name)
    projector = getattr(maxtext_model, maxtext_model.projector_name)
    copy_maxtext_audio_encoder_weights(torch_model, encoder, self.config)
    copy_audio_projector_weights(torch_model, projector)

    batch_size = 1
    num_mel_bins = self.config.num_mel_bins_for_audio
    audio_length = 200  # With n_window=50, chunk_size=100, gives 2 chunks

    jax_audio_features, torch_audio_features_3d = create_random_jax_torch(batch_size, num_mel_bins, audio_length)
    audio_lengths_np = np.array([audio_length], dtype=np.int64)

    torch_audio_features = torch_audio_features_3d[0]
    torch_audio_lengths = torch.from_numpy(audio_lengths_np)

    torch_output = torch_model(input_features=torch_audio_features, feature_lens=torch_audio_lengths)
    torch_output_tensor = torch_output.last_hidden_state

    jax_output = maxtext_model(jax_audio_features, deterministic=True)

    assert_all_close_jax_torch(
        jax_output[0],
        torch_output_tensor,
        rtol=1e-3,
        atol=0.02,
        error_msg="AudioModel outputs differ",
    )

  def test_audio_model_intermediates(self):
    """Debug intermediate outputs to verify each stage matches PyTorch."""
    torch_model = TorchQwen3OmniMoeAudioEncoder(self.torch_config)
    torch_model.eval()

    audio_encoder = Qwen3OmniAudioEncoder(config=self.config, mesh=self.mesh, rngs=nnx.Rngs(0))
    copy_maxtext_audio_encoder_weights(torch_model, audio_encoder, self.config)

    batch_size = 1
    num_mel_bins = self.config.num_mel_bins_for_audio
    audio_length = 100

    jax_audio_features, torch_audio_features_3d = create_random_jax_torch(batch_size, num_mel_bins, audio_length)
    torch_audio_features = torch_audio_features_3d[0]

    # PyTorch forward
    chunk_size = self.torch_config.n_window * 2
    num_chunks = audio_length // chunk_size
    chunk_lengths = torch.tensor([chunk_size] * num_chunks, dtype=torch.long)
    chunk_list = torch_audio_features.T.split(chunk_lengths.tolist(), dim=0)
    torch_padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    torch_padded_feature = torch_padded_feature.unsqueeze(1)

    torch_conv1 = F.gelu(torch_model.conv2d1(torch_padded_feature))
    torch_conv2 = F.gelu(torch_model.conv2d2(torch_conv1))
    torch_conv3 = F.gelu(torch_model.conv2d3(torch_conv2))

    b, c, f, t = torch_conv3.size()
    torch_conv_out = torch_model.conv_out(torch_conv3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    torch_pos_emb = (
        torch_model.positional_embedding.positional_embedding[: torch_conv_out.shape[1], :]
        .unsqueeze(0)
        .to(torch_conv_out.dtype)
    )
    torch_after_pos = torch_conv_out + torch_pos_emb

    # JAX forward
    jax_audio_chunks = jax_audio_features.reshape(batch_size, num_mel_bins, num_chunks, chunk_size)
    jax_audio_chunks = jax_audio_chunks.transpose(0, 2, 1, 3).reshape(batch_size * num_chunks, num_mel_bins, chunk_size)
    jax_hidden = jax_audio_chunks[:, :, :, jnp.newaxis]

    jax_conv1 = jax.nn.gelu(audio_encoder.conv2d1(jax_hidden))
    jax_conv2 = jax.nn.gelu(audio_encoder.conv2d2(jax_conv1))
    jax_conv3 = jax.nn.gelu(audio_encoder.conv2d3(jax_conv2))

    bc, f_jax, t_jax, c_jax = jax_conv3.shape
    jax_conv_out = audio_encoder.conv_out(jax_conv3.transpose(0, 2, 3, 1).reshape(bc, t_jax, c_jax * f_jax))

    seq_len_per_chunk = jax_conv_out.shape[1]
    jax_pos_emb = audio_encoder.positional_embedding(seq_len_per_chunk)
    jax_pos_emb = jnp.broadcast_to(
        jax_pos_emb[None, :, :], (batch_size * num_chunks, seq_len_per_chunk, self.config.d_model_for_audio)
    )
    jax_after_pos = jax_conv_out + jax_pos_emb

    # Verify all stages match
    assert_all_close_jax_torch(
        jax_conv1[0], torch_conv1.permute(0, 2, 3, 1)[0], rtol=1e-4, atol=1e-3, error_msg="Conv1 differs"
    )
    assert_all_close_jax_torch(
        jax_conv2[0], torch_conv2.permute(0, 2, 3, 1)[0], rtol=1e-4, atol=1e-3, error_msg="Conv2 differs"
    )
    assert_all_close_jax_torch(
        jax_conv3[0], torch_conv3.permute(0, 2, 3, 1)[0], rtol=1e-4, atol=1e-3, error_msg="Conv3 differs"
    )
    assert_all_close_jax_torch(jax_conv_out[0], torch_conv_out[0], rtol=1e-4, atol=1e-3, error_msg="Conv out differs")
    assert_all_close_jax_torch(
        jax_after_pos[0], torch_after_pos[0], rtol=1e-4, atol=1e-3, error_msg="After pos emb differs"
    )


if __name__ == "__main__":
  unittest.main()
