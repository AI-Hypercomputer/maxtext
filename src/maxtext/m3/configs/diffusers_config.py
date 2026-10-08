# Copyright 2023–2025 Google LLC
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

"""Diffusion model configuration schema and hyperparameter definitions."""

from typing import Any, Dict, List, Optional, Tuple
from maxtext.m3.configs.config import (
    BaseConfig,
    CheckpointConfig,
    DataConfig,
    DeviceConfig,
    TrainerConfig,
    TransformerConfig,
)
from pydantic import BaseModel, ConfigDict, Field


class AutoencoderConfig(BaseModel):
  """Autoencoder (VAE) configurations for spatial and temporal image/video compression."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  vae_type: str = "autoencoder_kl"
  vae_weights_dtype: Any = "float32"
  vae_dtype: Any = "float32"
  replicate_vae: bool = False
  vae_decode_chunk: int = 1
  vae_encode_chunk: int = 4
  vae_spatial: int = -1
  vae_logical_axis_rules: tuple = ()


class SchedulerConfig(BaseModel):
  """Diffusion noise scheduling, flow matching, and ODE solver configurations."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  scheduler_type: str = "unipc_multistep"
  scheduler_dtype: Any = "float32"
  use_experimental_scheduler: bool = True
  flow_shift: float = 5.0
  guidance_scale: float = 5.0
  guidance_rescale: float = 0.0
  num_inference_steps: int = 50
  diffusion_scheduler_config: dict = Field(default_factory=dict)
  timestep_bias: dict = Field(default_factory=dict)


class TextEncoderConfig(BaseModel):
  """Text conditioning and tokenizer configurations."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  text_encoder_type: str = "umt5"
  text_encoder_dtype: Any = "float32"
  compile_text_encoder: bool = False
  max_sequence_length: int = 512


class DiffusionTransformerConfig(TransformerConfig):
  """Diffusion Transformer (DiT) architecture configurations extending base TransformerConfig."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  patch_size: tuple[int, int, int] = (1, 2, 2)
  in_channels: int = 16
  out_channels: int = 16
  text_dim: int = 4096
  freq_dim: int = 256
  qk_norm: str = "rms_norm"
  cross_attn_norm: bool = True
  flash_block_sizes: Optional[Any] = None
  flash_min_seq_length: int = 4096
  ulysses_shards: int = -1
  ulysses_attention_chunks: int = 1
  remat_policy: str = "NONE"
  names_which_can_be_saved: list[str] = Field(default_factory=list)
  names_which_can_be_offloaded: list[str] = Field(default_factory=list)
  enable_jax_named_scopes: bool = False


class DiffusionPipelineConfig(BaseModel):
  """End-to-end diffusion pipeline, sampling, and conditioning hyperparameters."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  pretrained_model_name_or_path: str = ""
  model_name: str = ""
  model_type: str = "T2V"
  prompt: str = ""
  prompt_2: str = ""
  negative_prompt: str = ""
  prompt_file: str = ""
  do_classifier_free_guidance: bool = True
  height: int = 720
  width: int = 1280
  num_frames: int = 81
  fps: int = 16
  seed: int = 0

  # Cache optimizations
  use_cfg_cache: bool = False
  use_batched_text_encoder: bool = False
  use_kv_cache: bool = False
  use_magcache: bool = False
  magcache_thresh: float = 0.12
  magcache_K: int = 2
  retention_ratio: float = 0.2
  mag_ratios_base: list = Field(default_factory=list)

  # Adaptors and control
  enable_lora: bool = False
  lora_config: dict = Field(default_factory=dict)
  controlnet_model_name_or_path: str = ""
  controlnet_from_pt: bool = True
  controlnet_conditioning_scale: float = 0.5
  controlnet_image: str = ""
  quantization: str = ""
  use_qwix_quantization: bool = False
  qwix_module_path: str = ""
  weight_quantization_calibration_method: str = "absmax"
  act_quantization_calibration_method: str = "absmax"
  bwd_quantization_calibration_method: str = "absmax"


class DiffusionConfig(BaseConfig):
  """Top-level diffusion model configuration orchestrating core and modular diffusers components."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
  scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
  text_encoder: TextEncoderConfig = Field(default_factory=TextEncoderConfig)
  transformer: DiffusionTransformerConfig = Field(
      default_factory=DiffusionTransformerConfig
  )
  pipeline: DiffusionPipelineConfig = Field(
      default_factory=DiffusionPipelineConfig
  )

  @classmethod
  def from_dict(cls, raw: dict[str, Any]) -> "DiffusionConfig":
    device = DeviceConfig.model_validate(raw)
    checkpoint = CheckpointConfig.model_validate(raw)
    trainer = TrainerConfig.model_validate(raw)
    data = DataConfig.model_validate(raw)
    autoencoder = AutoencoderConfig.model_validate(raw)
    scheduler = SchedulerConfig.model_validate(raw)
    text_encoder = TextEncoderConfig.model_validate(raw)
    transformer = DiffusionTransformerConfig.model_validate(raw)
    pipeline = DiffusionPipelineConfig.model_validate(raw)

    return cls(
        device=device,
        checkpoint=checkpoint,
        trainer=trainer,
        data=data,
        autoencoder=autoencoder,
        scheduler=scheduler,
        text_encoder=text_encoder,
        transformer=transformer,
        pipeline=pipeline,
        **raw,
    )
