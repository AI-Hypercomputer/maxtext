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
from typing import Any

from maxtext.configs.types import DType
from maxtext.m3.models.model_config import coerce_dtype, extract_config_attrs
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config, extract_qwen3_kwargs
from pydantic import Field, model_validator

_QWEN3_VL_ATTR_MAP: dict[str, tuple[str, ...]] = {
    "use_mrope": ("use_mrope",),
    "mrope_section": ("mrope_section",),
    "use_multimodal": ("use_multimodal",),
    "vision_dtype": ("vision_dtype", "dtype_mm"),
    "vision_image_size": ("vision_image_size", "image_size_for_vit"),
    "vision_hidden_size": ("vision_hidden_size", "hidden_size_for_vit"),
    "vision_intermediate_size": ("vision_intermediate_size", "intermediate_size_for_vit"),
    "vision_num_heads": ("vision_num_heads", "num_attention_heads_for_vit"),
    "vision_depth": ("vision_depth", "num_hidden_layers_for_vit"),
    "vision_in_channels": ("vision_in_channels", "num_channels_for_vit"),
    "vision_patch_size": ("vision_patch_size", "patch_size_for_vit"),
    "vision_temporal_patch_size": ("vision_temporal_patch_size", "temporal_patch_size_for_vit"),
    "vision_spatial_merge_size": ("vision_spatial_merge_size", "spatial_merge_size_for_vit"),
    "vision_out_hidden_size": ("vision_out_hidden_size", "out_hidden_size_for_vit"),
    "vision_num_position_embeddings": ("vision_num_position_embeddings", "num_position_embeddings_for_vit"),
    "vision_rope_theta": ("vision_rope_theta", "rope_theta_for_vit"),
    "deepstack_visual_indexes": ("deepstack_visual_indexes", "deepstack_visual_indexes_for_vit"),
}


class Qwen3VLConfig(Qwen3Config):
  """Frozen architecture configuration for the Qwen3-VL multimodal model family."""

  model_type: str = Field("qwen3_vl", description="HuggingFace model_type identifier.")
  model_name: str = Field("qwen3-vl-2b", description="Canonical Qwen3-VL model identifier.")
  hidden_size: int = Field(2048, gt=0, description="Text decoder hidden / embedding dimension.")
  intermediate_size: int = Field(6144, gt=0, description="Text decoder feed-forward intermediate dimension.")
  num_hidden_layers: int = Field(28, gt=0, description="Number of text decoder layers.")
  num_attention_heads: int = Field(16, gt=0, description="Number of attention query heads.")
  num_key_value_heads: int = Field(8, gt=0, description="Number of grouped-query KV heads.")
  head_dim: int = Field(128, gt=0, description="Per-head attention dimension.")
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
  vision_out_hidden_size: int = Field(2048, gt=0, description="Projected vision feature dimension matching text.")
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

  def to_pyconfig_dict(self) -> dict[str, Any]:
    """Returns `MaxTextConfig` field defaults matching `qwen3-vl-2b.yml`."""
    out = super().to_pyconfig_dict()
    out.pop("tokenizer_type", None)
    out.update(
        {
            "vision_encoder_block": "qwen3_vl",
            "use_multimodal": self.use_multimodal,
            "image_size_for_vit": self.vision_image_size,
            "hidden_size_for_vit": self.vision_hidden_size,
            "intermediate_size_for_vit": self.vision_intermediate_size,
            "num_attention_heads_for_vit": self.vision_num_heads,
            "num_hidden_layers_for_vit": self.vision_depth,
            "num_channels_for_vit": self.vision_in_channels,
            "patch_size_for_vit": self.vision_patch_size,
            "temporal_patch_size_for_vit": self.vision_temporal_patch_size,
            "spatial_merge_size_for_vit": self.vision_spatial_merge_size,
            "out_hidden_size_for_vit": self.vision_out_hidden_size,
            "num_position_embeddings_for_vit": self.vision_num_position_embeddings,
            "deepstack_visual_indexes_for_vit": list(self.deepstack_visual_indexes),
            "use_mrope": self.use_mrope,
            "mrope_section": list(self.mrope_section),
        }
    )
    return out

  @classmethod
  def from_config(cls, config: Any) -> "Qwen3VLConfig":
    """Returns a `Qwen3VLConfig` from either a `Qwen3VLConfig` or a legacy `MaxTextConfig` / namespace."""
    if isinstance(config, cls):
      return config
    kwargs = extract_qwen3_kwargs(config)
    kwargs.update(extract_config_attrs(config, _QWEN3_VL_ATTR_MAP))
    if "vision_dtype" in kwargs:
      kwargs["vision_dtype"] = coerce_dtype(kwargs["vision_dtype"])
    if "mrope_section" in kwargs:
      kwargs["mrope_section"] = tuple(int(v) for v in kwargs["mrope_section"])
    if "deepstack_visual_indexes" in kwargs:
      kwargs["deepstack_visual_indexes"] = tuple(int(v) for v in kwargs["deepstack_visual_indexes"])
    return cls(**kwargs)

  @classmethod
  def qwen3_vl_2b(cls, **overrides: Any) -> "Qwen3VLConfig":
    """Returns the Qwen3-VL-2B architecture configuration."""
    return cls(**overrides)
