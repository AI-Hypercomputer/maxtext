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

"""Model Garden config for the Qwen3-VL multimodal model family."""

import math
from typing import Any, ClassVar

from maxtext.configs.components import DType
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from pydantic import Field, model_validator


class Qwen3VLConfig(Qwen3Config):
  """Frozen architecture configuration for the Qwen3-VL multimodal model family."""

  model_type: str = Field("qwen3_vl", description="HuggingFace model_type identifier.")
  rope_theta: float = Field(5_000_000.0, gt=0.0, description="RoPE base period / max timescale.")

  # M-RoPE settings
  use_mrope: bool = Field(True, description="Enable interleaved 3D M-RoPE for multimodal positions.")
  mrope_section: tuple[int, int, int] = Field(
      (24, 20, 20),
      description="Temporal/height/width rotary dimension splits summing to `head_dim // 2`.",
  )

  # Vision encoder settings (native `vision_config` naming)
  use_multimodal: bool = Field(True, description="Enable vision encoder and multimodal DeepStack fusion.")
  vision_dtype: DType = Field(DType.FLOAT32, description="Compute dtype for the vision encoder tower.")
  vision_image_size: int = Field(768, gt=0, description="Default ViT input image resolution.")
  vision_hidden_size: int = Field(1024, gt=0, description="Vision transformer hidden dimension.")
  vision_intermediate_size: int = Field(4096, gt=0, description="Vision transformer MLP dimension.")
  vision_num_heads: int = Field(16, gt=0, description="Vision transformer attention head count.")
  vision_depth: int = Field(24, gt=0, description="Number of vision transformer encoder blocks.")
  vision_in_channels: int = Field(3, gt=0, description="Input image/video channel count.")
  vision_patch_size: int = Field(16, gt=0, description="Spatial ViT patch size.")
  vision_temporal_patch_size: int = Field(2, gt=0, description="Temporal patch size for video inputs.")
  vision_spatial_merge_size: int = Field(2, gt=0, description="Spatial patch-merger downsampling factor.")
  vision_out_hidden_size: int = Field(..., gt=0, description="Projected vision feature dimension matching text.")
  vision_num_position_embeddings: int = Field(2304, gt=0, description="Learned 2D position embedding count (N^2).")
  vision_rope_theta: float = Field(10000.0, gt=0.0, description="2D RoPE base period for the vision encoder.")
  deepstack_visual_indexes: tuple[int, ...] = Field(
      (5, 11, 17),
      description="Vision encoder block indices injected via DeepStack.",
  )

  @model_validator(mode="after")
  def _validate_vision_and_mrope(self) -> "Qwen3VLConfig":
    """Validates M-RoPE splits, ViT head geometry, position embeddings, and DeepStack indices."""
    if self.use_mrope and (any(s < 0 for s in self.mrope_section) or sum(self.mrope_section) != self.head_dim // 2):
      raise ValueError(
          f"mrope_section ({self.mrope_section}) entries must be non-negative and sum to "
          f"head_dim // 2 ({self.head_dim // 2})."
      )
    if self.vision_hidden_size % self.vision_num_heads != 0:
      raise ValueError(
          f"vision_hidden_size ({self.vision_hidden_size}) must be divisible by "
          f"vision_num_heads ({self.vision_num_heads})."
      )
    side = math.isqrt(self.vision_num_position_embeddings)
    if side * side != self.vision_num_position_embeddings:
      raise ValueError(
          f"vision_num_position_embeddings ({self.vision_num_position_embeddings}) must be a perfect square."
      )
    for idx in self.deepstack_visual_indexes:
      if not 0 <= idx < self.vision_depth:
        raise ValueError(f"deepstack_visual_indexes entry {idx} out of range [0, {self.vision_depth}).")
    if self.use_multimodal and self.vision_out_hidden_size != self.hidden_size:
      raise ValueError(
          f"vision_out_hidden_size ({self.vision_out_hidden_size}) must equal hidden_size ({self.hidden_size})."
      )
    return self

  def supported_mesh_axes(self) -> frozenset[str]:
    """Mesh axes Qwen3-VL implements (`MODEL_SUPPORTED_MESH_AXES`); narrower than `Qwen3Config`."""
    return frozenset({"data", "fsdp", "tensor", "expert"})

  PYCONFIG_CONSTANTS: ClassVar[dict[str, Any]] = {
      **{k: v for k, v in Qwen3Config.PYCONFIG_CONSTANTS.items() if k != "tokenizer_type"},
      "vision_encoder_block": "qwen3_vl",
  }
  PYCONFIG_KEYS: ClassVar[dict[str, str]] = {
      **Qwen3Config.PYCONFIG_KEYS,
      "use_multimodal": "use_multimodal",
      "dtype_mm": "vision_dtype",
      "image_size_for_vit": "vision_image_size",
      "hidden_size_for_vit": "vision_hidden_size",
      "intermediate_size_for_vit": "vision_intermediate_size",
      "num_attention_heads_for_vit": "vision_num_heads",
      "num_hidden_layers_for_vit": "vision_depth",
      "num_channels_for_vit": "vision_in_channels",
      "patch_size_for_vit": "vision_patch_size",
      "temporal_patch_size_for_vit": "vision_temporal_patch_size",
      "spatial_merge_size_for_vit": "vision_spatial_merge_size",
      "out_hidden_size_for_vit": "vision_out_hidden_size",
      "num_position_embeddings_for_vit": "vision_num_position_embeddings",
      "rope_theta_for_vit": "vision_rope_theta",
      "deepstack_visual_indexes_for_vit": "deepstack_visual_indexes",
      "use_mrope": "use_mrope",
      "mrope_section": "mrope_section",
  }

  @classmethod
  def qwen3_vl_2b(cls) -> "Qwen3VLConfig":
    """Returns the `qwen3-vl-2b` architecture configuration."""
    return cls(
        model_name="qwen3-vl-2b",
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        tie_word_embeddings=True,
        vision_out_hidden_size=2048,
    )
