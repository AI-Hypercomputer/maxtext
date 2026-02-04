# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Qwen3-Omni MRoPE position ID computation.

This test suite verifies the get_rope_index() function by comparing
outputs with the PyTorch reference implementation from modeling_qwen3_omni_moe.py.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRotaryEmbedding as PyTorchMRoPE,
    apply_rotary_pos_emb,
)

from MaxText import multimodal_utils
from MaxText.input_pipeline._input_pipeline_utils import ComputeQwen3OmniPositions
from MaxText.layers.embeddings import Qwen3OmniMoeThinkerTextRotaryEmbedding as JaxMRoPE


# Qwen3-Omni special token IDs
VISION_START = multimodal_utils.QWEN3_OMNI_VISION_START_TOKEN
VISION_END = multimodal_utils.QWEN3_OMNI_VISION_END_TOKEN
AUDIO_START = multimodal_utils.QWEN3_OMNI_AUDIO_START_TOKEN
AUDIO_END = multimodal_utils.QWEN3_OMNI_AUDIO_END_TOKEN
IMAGE_TOKEN = multimodal_utils.QWEN3_OMNI_IMAGE_TOKEN
VIDEO_TOKEN = multimodal_utils.QWEN3_OMNI_VIDEO_TOKEN
AUDIO_TOKEN = multimodal_utils.QWEN3_OMNI_AUDIO_TOKEN


def create_pytorch_config(head_dim=128, mrope_section=(24, 20, 20), rope_max_timescale=1_000_000):
  """Create a unified mock config for PyTorch models.

  This config supports both MRoPE embedding and get_rope_index functionality.
  """

  class MockConfig:
    """Mock configuration for testing."""

    def __init__(self):
      # MRoPE-specific attributes
      self.head_dim = head_dim
      self.hidden_size = head_dim
      self.num_attention_heads = 1
      self.max_position_embeddings = 65536
      self.rope_theta = rope_max_timescale
      self.mrope_section = mrope_section
      self.rope_scaling = {"mrope_section": list(mrope_section)}
      self.attention_scaling = 1.0
      self.partial_rotary_factor = 1.0

      # Token ID attributes for get_rope_index
      self.image_token_id = IMAGE_TOKEN
      self.video_token_id = VIDEO_TOKEN
      self.audio_token_id = AUDIO_TOKEN
      self.vision_start_token_id = VISION_START
      self.audio_start_token_id = AUDIO_START
      self.position_id_per_seconds = 25

  return MockConfig()


def create_pytorch_model():
  """Create PyTorch model instance with get_rope_index method."""

  class MockModel(Qwen3OmniMoePreTrainedModelForConditionalGeneration):

    def __init__(self):
      self.config = create_pytorch_config()
      self.spatial_merge_size = 2

  return MockModel()


def create_audio_in_video_sequence(
    video_grid_thw,
    audio_lengths,
    second_per_grids,
    spatial_merge_size=2,
    position_id_per_seconds=25,
):
  """Create interleaved audio-in-video token sequence.

  Args:
    video_grid_thw: Video dimensions (temporal, height, width). Shape: (num_videos, 3).
    audio_lengths: Raw audio sequence lengths. Shape: (num_audios,).
    second_per_grids: Time interval per temporal grid. Shape: (num_videos,).
    spatial_merge_size: Number of patches merged spatially.
    position_id_per_seconds: Temporal granularity.

  Returns:
    np.ndarray of interleaved token IDs for audio-in-video.
  """
  # Compute token counts
  expected_audio_tokens = int(multimodal_utils._get_feat_extract_output_lengths(jnp.array(audio_lengths[0])).item())  # pylint: disable=protected-access

  # Video tokens
  video_tokens_per_frame = (video_grid_thw[0, 1] // spatial_merge_size) * (video_grid_thw[0, 2] // spatial_merge_size)
  num_frames = video_grid_thw[0, 0]

  # Compute temporal positions for video tokens
  video_temporal_positions = []
  for frame_idx in range(num_frames):
    frame_time = frame_idx * second_per_grids[0] * position_id_per_seconds
    video_temporal_positions.extend([frame_time] * video_tokens_per_frame)

  # Audio tokens have sequential positions (0, 1, 2, ...)
  audio_temporal_positions = list(range(expected_audio_tokens))

  # Interleave tokens based on temporal order
  interleaved_tokens = []
  video_idx = 0
  audio_idx = 0

  while video_idx < len(video_temporal_positions) and audio_idx < len(audio_temporal_positions):
    if video_temporal_positions[video_idx] <= audio_temporal_positions[audio_idx]:
      interleaved_tokens.append(VIDEO_TOKEN)
      video_idx += 1
    else:
      interleaved_tokens.append(AUDIO_TOKEN)
      audio_idx += 1

  # Append remaining tokens
  interleaved_tokens.extend([VIDEO_TOKEN] * (len(video_temporal_positions) - video_idx))
  interleaved_tokens.extend([AUDIO_TOKEN] * (len(audio_temporal_positions) - audio_idx))

  # Build full sequence with proper token structure
  return np.array(
      [
          VISION_START,
          AUDIO_START,
          *interleaved_tokens,
          AUDIO_END,
          VISION_END,
      ],
      dtype=np.int32,
  )


def assert_mrope_matches_pytorch(
    query_states,
    position_ids,
    err_msg,
    mrope_section=(24, 20, 20),
    rope_max_timescale=1_000_000,
    head_dim=128,
    rtol=1e-4,
    atol=1e-4,
):
  """Compare JAX MRoPE with PyTorch reference and assert they match.

  Args:
    query_states: Query tensor. Shape: (batch, seq_len, num_heads, head_dim)
    position_ids: 3D position IDs. Shape: (3, batch, seq_len)
    err_msg: Error message for assertion failure
    mrope_section: Dimensions for temporal, height, width
    rope_max_timescale: Max timescale for RoPE
    head_dim: Dimension of each attention head
    rtol: Relative tolerance for comparison
    atol: Absolute tolerance for comparison
  """
  # JAX version
  rngs = nnx.Rngs(0)
  jax_mrope = JaxMRoPE(
      min_timescale=1,
      max_timescale=rope_max_timescale,
      embedding_dims=head_dim,
      cast_as_fprop_dtype=False,
      fprop_dtype=jnp.float32,
      mrope_section=mrope_section,
      rngs=rngs,
  )

  jax_query = jnp.array(query_states)
  jax_position_ids = jnp.array(position_ids)
  jax_output = jax_mrope(jax_query, jax_position_ids)

  # PyTorch version
  torch_config = create_pytorch_config(head_dim, mrope_section, rope_max_timescale)
  torch_mrope = PyTorchMRoPE(torch_config)

  torch_query = torch.from_numpy(np.array(query_states)).float()
  torch_position_ids = torch.from_numpy(np.array(position_ids))

  # PyTorch expects (batch, num_heads, seq_len, head_dim)
  # We have (batch, seq_len, num_heads, head_dim), so transpose
  torch_query = torch_query.transpose(1, 2)

  torch_cos, torch_sin = torch_mrope(torch_query, torch_position_ids)

  # Apply rotation in PyTorch using the reference implementation
  # apply_rotary_pos_emb expects (q, k, cos, sin) and returns (q_embed, k_embed)
  # We only need q_embed, so pass torch_query twice and take the first result
  # unsqueeze_dim=1 because query is (batch, num_heads, seq_len, head_dim)
  # and cos/sin are (batch, seq_len, head_dim), so unsqueeze at dim=1 gives (batch, 1, seq_len, head_dim)
  torch_output, _ = apply_rotary_pos_emb(torch_query, torch_query, torch_cos, torch_sin, unsqueeze_dim=1)

  # Transpose back to (batch, seq_len, num_heads, head_dim)
  torch_output = torch_output.transpose(1, 2)

  # Assert outputs match
  np.testing.assert_allclose(np.array(jax_output), torch_output.cpu().numpy(), rtol=rtol, atol=atol, err_msg=err_msg)


class GetRopeIndexComparisonTest(unittest.TestCase):
  """Test get_rope_index() against PyTorch reference implementation."""

  @classmethod
  def setUpClass(cls):
    """Set up PyTorch reference model once for all tests."""
    cls.pytorch_model = create_pytorch_model()

  def compare_with_pytorch(
      self,
      input_ids,
      image_grid_thw=None,
      video_grid_thw=None,
      attention_mask=None,
      use_audio_in_video=False,
      audio_lengths=None,
      second_per_grids=None,
      spatial_merge_size=2,
      position_id_per_seconds=25,
  ):
    """Compare JAX and PyTorch implementations.

    Args:
      input_ids: Token IDs as numpy array (batch, seq_len)
      image_grid_thw: Optional (num_images, 3)
      video_grid_thw: Optional (num_videos, 3)
      attention_mask: Optional (batch, seq_len)
      use_audio_in_video: Whether to interleave audio with video
      audio_lengths: Optional (num_audios,)
      second_per_grids: Optional (num_videos,)
      spatial_merge_size: Spatial merge size
      position_id_per_seconds: Temporal granularity

    Returns:
      Tuple of (jax_position_ids, pytorch_position_ids, match_status)
    """
    jax_position_ids_np, jax_deltas_np = multimodal_utils.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        use_audio_in_video=use_audio_in_video,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        spatial_merge_size=spatial_merge_size,
        position_id_per_seconds=position_id_per_seconds,
    )

    # PyTorch version
    torch_input_ids = torch.from_numpy(input_ids).long()
    # PyTorch get_rope_index requires attention_mask - create one if not provided
    if attention_mask is None:
      torch_attention_mask = torch.ones_like(torch_input_ids)
    else:
      torch_attention_mask = torch.from_numpy(attention_mask)
    torch_image_grid_thw = torch.from_numpy(image_grid_thw).long() if image_grid_thw is not None else None
    torch_video_grid_thw = torch.from_numpy(video_grid_thw).long() if video_grid_thw is not None else None
    torch_audio_lengths = torch.from_numpy(audio_lengths).long() if audio_lengths is not None else None
    torch_second_per_grids = torch.from_numpy(second_per_grids).float() if second_per_grids is not None else None

    torch_position_ids, torch_deltas = self.pytorch_model.get_rope_index(
        input_ids=torch_input_ids,
        image_grid_thw=torch_image_grid_thw,
        video_grid_thw=torch_video_grid_thw,
        attention_mask=torch_attention_mask,
        use_audio_in_video=use_audio_in_video,
        audio_seqlens=torch_audio_lengths,
        second_per_grids=torch_second_per_grids,
    )

    # Convert to numpy for comparison
    torch_position_ids_np = torch_position_ids.cpu().numpy()
    torch_deltas_np = torch_deltas.cpu().numpy()

    return jax_position_ids_np, torch_position_ids_np, jax_deltas_np, torch_deltas_np

  def test_text_only(self):
    """Test text-only sequences (single, with padding, and batched) against PyTorch."""
    # Test 1: Simple single sequence
    input_ids = np.array([[1, 2, 3, 4, 5]])
    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids)
    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5, err_msg="Single sequence positions don't match PyTorch")
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5, err_msg="Single sequence deltas don't match PyTorch")

    # Test 2: With padding
    input_ids = np.array([[1, 2, 3, 0, 0]])
    attention_mask = np.array([[1, 1, 1, 0, 0]])
    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids, attention_mask=attention_mask)
    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5, err_msg="Padded sequence positions don't match PyTorch")
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5, err_msg="Padded sequence deltas don't match PyTorch")

    # Test 3: Batched
    input_ids = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 0, 0],
        ]
    )
    attention_mask = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ]
    )
    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids, attention_mask=attention_mask)
    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5, err_msg="Batched sequence positions don't match PyTorch")
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5, err_msg="Batched sequence deltas don't match PyTorch")

  def test_single_image(self):
    """Test single image sequence against PyTorch."""
    # Sequence: <|vision_start|> <|image_pad|> x4 <|vision_end|>
    input_ids = np.array([[VISION_START, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, VISION_END]])

    # Image: 1 frame, 4x4 patches, spatial_merge_size=2 -> 2x2 = 4 tokens
    image_grid_thw = np.array([[1, 4, 4]])

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids, image_grid_thw=image_grid_thw)

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_image_with_text_before_after(self):
    """Test image with text before and after against PyTorch."""
    # "Describe" <img> "please"
    input_ids = np.array(
        [
            [
                100,
                101,  # text before
                VISION_START,
                IMAGE_TOKEN,
                IMAGE_TOKEN,
                IMAGE_TOKEN,
                IMAGE_TOKEN,
                VISION_END,
                200,
                201,  # text after
            ]
        ]
    )

    image_grid_thw = np.array([[1, 4, 4]])

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids, image_grid_thw=image_grid_thw)

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_multiple_images(self):
    """Test multiple images in sequence against PyTorch."""
    input_ids = np.array(
        [
            [
                VISION_START,
                IMAGE_TOKEN,
                IMAGE_TOKEN,
                VISION_END,
                100,  # text token
                VISION_START,
                IMAGE_TOKEN,
                IMAGE_TOKEN,
                VISION_END,
            ]
        ]
    )

    # Two images: each 1 frame, 2x2 patches, merge to 1x1 = 1 token
    image_grid_thw = np.array(
        [
            [1, 2, 2],
            [1, 2, 2],
        ]
    )

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids, image_grid_thw=image_grid_thw)

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_single_video_temporal_spacing(self):
    """Test video with temporal spacing against PyTorch."""
    # Video: 2 frames, 4x4 patches, merge to 2x2 = 4 tokens per frame = 8 total
    input_ids = np.array(
        [
            [
                VISION_START,
                VIDEO_TOKEN,
                VIDEO_TOKEN,
                VIDEO_TOKEN,
                VIDEO_TOKEN,  # frame 1
                VIDEO_TOKEN,
                VIDEO_TOKEN,
                VIDEO_TOKEN,
                VIDEO_TOKEN,  # frame 2
                VISION_END,
            ]
        ]
    )

    video_grid_thw = np.array([[2, 4, 4]])
    second_per_grids = np.array([2.0])  # 2 seconds per frame

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(
        input_ids,
        video_grid_thw=video_grid_thw,
        second_per_grids=second_per_grids,
    )

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_single_audio(self):
    """Test audio sequence against PyTorch."""
    # Compute expected audio tokens from raw length
    audio_lengths = np.array([1600])
    # pylint: disable=protected-access
    expected_tokens = int(multimodal_utils._get_feat_extract_output_lengths(jnp.array(1600)).item())

    audio_tokens = [AUDIO_TOKEN] * expected_tokens
    input_ids = np.array([[AUDIO_START, *audio_tokens, AUDIO_END]])

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(input_ids, audio_lengths=audio_lengths)

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_image_and_text_batch(self):
    """Test batch with mixed text-only and image sequences against PyTorch."""
    # Batch: sequence 0 is text-only, sequence 1 has image
    input_ids = np.array(
        [
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],  # text only + padding
            [100, VISION_START, IMAGE_TOKEN, IMAGE_TOKEN, VISION_END, 200, 0, 0, 0, 0],  # image + padding
        ]
    )
    attention_mask = np.array(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )

    # Only second sequence has image
    image_grid_thw = np.array([[1, 2, 2]])

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_video_with_different_fps(self):
    """Test video with different frame rates against PyTorch."""
    # Single video, 3 frames
    num_tokens = 3 * 4  # 3 frames * 4 tokens per frame (2x2 grid)
    input_ids = np.array([[VISION_START, *([VIDEO_TOKEN] * num_tokens), VISION_END]])

    video_grid_thw = np.array([[3, 4, 4]])
    second_per_grids = np.array([1.5])  # 1.5 seconds per frame

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(
        input_ids,
        video_grid_thw=video_grid_thw,
        second_per_grids=second_per_grids,
    )

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5)
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5)

  def test_audio_in_video(self):
    """Test audio-in-video interleaving against PyTorch.

    The HuggingFace processor interleaves audio and video tokens based on temporal
    ordering. We use the helper function to create realistic test inputs.
    """
    # Video: 2 frames, 4x4 patches, merge to 2x2 = 4 tokens per frame = 8 total
    video_grid_thw = np.array([[2, 4, 4]])
    audio_lengths = np.array([800])
    second_per_grids = np.array([1.0])  # 1 second per frame
    spatial_merge_size = 2
    position_id_per_seconds = 25

    # Create interleaved sequence
    token_sequence = create_audio_in_video_sequence(
        video_grid_thw, audio_lengths, second_per_grids, spatial_merge_size, position_id_per_seconds
    )
    input_ids = token_sequence.reshape(1, -1)

    jax_pos, torch_pos, jax_deltas, torch_deltas = self.compare_with_pytorch(
        input_ids,
        video_grid_thw=video_grid_thw,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        use_audio_in_video=True,
    )

    np.testing.assert_allclose(jax_pos, torch_pos, rtol=1e-5, err_msg="Audio-in-video positions don't match PyTorch")
    np.testing.assert_allclose(jax_deltas, torch_deltas, rtol=1e-5, err_msg="Audio-in-video deltas don't match PyTorch")


class MRoPEComparisonTest(unittest.TestCase):
  """Test MRoPE (Multi-dimensional RoPE) against PyTorch reference."""

  def test_mrope_text_only_1d(self):
    """Test MRoPE with text-only (1D) position IDs."""
    batch, seq_len, num_heads, head_dim = 1, 5, 4, 128

    # Query states
    query_states = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)

    # 1D position IDs (text-only): same value in all 3 dimensions
    position_ids_1d = np.arange(seq_len).reshape(1, seq_len)
    position_ids_3d = np.broadcast_to(position_ids_1d[np.newaxis, :, :], (3, batch, seq_len))

    assert_mrope_matches_pytorch(query_states, position_ids_3d, err_msg="MRoPE text-only output doesn't match PyTorch")

  def test_mrope_vision_3d(self):
    """Test MRoPE with vision (3D) position IDs."""
    batch, seq_len, num_heads, head_dim = 1, 8, 4, 128

    # Query states
    query_states = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)

    # 3D position IDs for vision: temporal=0, height=[0,0,1,1,...], width=[0,1,0,1,...]
    position_ids_3d = np.array(
        [
            [[0, 0, 0, 0, 25, 25, 25, 25]],  # temporal
            [[0, 0, 1, 1, 0, 0, 1, 1]],  # height
            [[0, 1, 0, 1, 0, 1, 0, 1]],  # width
        ],
        dtype=np.float32,
    )

    assert_mrope_matches_pytorch(query_states, position_ids_3d, err_msg="MRoPE vision 3D output doesn't match PyTorch")

  def test_mrope_mixed_sequence(self):
    """Test MRoPE with mixed text and vision tokens."""
    batch, seq_len, num_heads, head_dim = 1, 10, 4, 128

    # Query states
    query_states = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)

    # Mixed: text tokens [0,1], vision tokens (4 tokens), text tokens [5,6,7]
    position_ids_3d = np.array(
        [
            [[0, 1, 2, 2, 2, 2, 6, 7, 8, 9]],  # temporal
            [[0, 1, 2, 2, 3, 3, 6, 7, 8, 9]],  # height (vision different)
            [[0, 1, 2, 3, 2, 3, 6, 7, 8, 9]],  # width (vision different)
        ],
        dtype=np.float32,
    )

    assert_mrope_matches_pytorch(
        query_states, position_ids_3d, err_msg="MRoPE mixed sequence output doesn't match PyTorch"
    )

  def test_mrope_different_mrope_sections(self):
    """Test MRoPE with different mrope_section values."""
    batch, seq_len, num_heads, head_dim = 1, 5, 4, 128

    # Query states
    query_states = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)

    # 3D position IDs
    position_ids_3d = np.array(
        [
            [[0, 0, 25, 25, 50]],  # temporal
            [[0, 1, 0, 1, 0]],  # height
            [[0, 0, 1, 1, 2]],  # width
        ],
        dtype=np.float32,
    )

    # Test different mrope_section
    for mrope_section in [(16, 28, 20), (32, 16, 16), (24, 20, 20)]:
      with self.subTest(mrope_section=mrope_section):
        assert_mrope_matches_pytorch(
            query_states,
            position_ids_3d,
            mrope_section=mrope_section,
            err_msg=f"MRoPE with {mrope_section} doesn't match PyTorch",
        )

  def test_mrope_batch(self):
    """Test MRoPE with batched inputs."""
    batch, seq_len, num_heads, head_dim = 4, 6, 4, 128

    # Query states
    query_states = np.random.randn(batch, seq_len, num_heads, head_dim).astype(np.float32)

    # Different position IDs for each sequence in batch
    position_ids_3d = np.random.randint(0, 100, size=(3, batch, seq_len)).astype(np.float32)

    # Batch test may have slightly larger numerical differences due to accumulation
    assert_mrope_matches_pytorch(
        query_states, position_ids_3d, rtol=1e-3, atol=1e-3, err_msg="MRoPE batched output doesn't match PyTorch"
    )


class ComputeQwen3OmniPositionsTest(unittest.TestCase):
  """Test ComputeQwen3OmniPositions Grain transform wrapper."""

  def test_transform_wrapper(self):
    """Test that the Grain transform wrapper correctly calls get_rope_index."""
    # Test with image to verify multimodal handling
    spatial_merge_size = 2
    transform = ComputeQwen3OmniPositions(data_column="inputs", spatial_merge_size=spatial_merge_size)

    element = {
        "inputs": np.array(
            [[VISION_START, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, VISION_END, 100]], dtype=np.int32
        ),
        "inputs_segmentation": np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.int32),
        "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
    }

    result = transform.map(element)

    # Verify transform adds position fields
    self.assertIn("inputs_position", result)
    self.assertIn("inputs_mrope_deltas", result)

    # Verify it matches direct get_rope_index call
    expected_pos, expected_deltas = multimodal_utils.get_rope_index(
        input_ids=jnp.array(element["inputs"]),
        image_grid_thw=jnp.array(element["image_grid_thw"]),
        video_grid_thw=None,
        attention_mask=jnp.array(element["inputs_segmentation"]),
        use_audio_in_video=False,
        audio_lengths=None,
        second_per_grids=None,
        spatial_merge_size=spatial_merge_size,
        position_id_per_seconds=25,
    )

    np.testing.assert_array_equal(result["inputs_position"], expected_pos)
    np.testing.assert_array_equal(result["inputs_mrope_deltas"], expected_deltas)


if __name__ == "__main__":
  unittest.main()
