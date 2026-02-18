# Copyright 2023–2025 Google LLC

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

"""Pydantic-based configuration system for MaxText, organized into modular classes."""

# pylint: disable=too-many-lines

import datetime
import enum
from enum import Enum
import logging
import math
from math import prod
import os
from tempfile import gettempdir
from typing import Any, Literal, NewType, Optional

import jax
from maxtext.common.common_types import AttentionType, DecoderBlockType, ShardMode
from maxtext.utils import gcs_utils
from maxtext.utils import max_utils
from MaxText import accelerator_to_spec_map
from MaxText.globals import MAXTEXT_ASSETS_ROOT
from pydantic.config import ConfigDict
from pydantic.fields import Field
from pydantic.functional_validators import field_validator, model_validator
from pydantic.main import BaseModel
from pydantic.types import NonNegativeFloat, NonNegativeInt, PositiveInt


class XProfTPUPowerTraceMode(enum.IntEnum):  # pylint: disable=invalid-name
  """Enum for XProfTPUPowerTraceMode."""

  POWER_TRACE_NONE = 0
  POWER_TRACE_NORMAL = 1
  POWER_TRACE_SPI = 2


logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Reusable Enums and Type Aliases
# ----------------------------------------------------------------------------

PathStr = str
AxisNames = NewType("AxisNames", str)


class DType(str, Enum):
  """Supported data types for weights and activations."""

  BFLOAT16 = "bfloat16"
  FLOAT32 = "float32"
  FLOAT16 = "float16"


class MatmulPrecision(str, Enum):
  """Precision levels for matrix multiplications."""

  DEFAULT = "default"
  HIGH = "high"
  HIGHEST = "highest"
  # same as default
  BFLOAT16 = "bfloat16"
  # same as highest
  FLOAT32 = "float32"


class QuantizationType(str, Enum):
  """Supported quantization schemes."""

  NONE = ""
  INT8 = "int8"
  INTMP = "intmp"
  FP8 = "fp8"
  NANOO_FP8 = "nanoo_fp8"
  FP8_NANO_V2 = "fp8_nanoo"
  FP8_GPU = "fp8_gpu"
  FP8_FULL = "fp8_full"
  TE_FP8_DS = "te_fp8_delayedscaling"
  TE_FP8_CS = "te_fp8_currentscaling"
  TE_MXFP8 = "te_mxfp8"
  TE_NVFP4 = "te_nvfp4"
  TE_NVFP4_NO_RHT = "te_nvfp4_no_rht"


class KvQuantAxis(str, Enum):
  """Axes to quantize over for the Key-Value cache."""

  NONE = ""
  DKV = "dkv"
  HEADS_AND_DKV = "heads_and_dkv"


class RematPolicy(str, Enum):
  """Available rematerialization (gradient checkpointing) policies."""

  FULL = "full"
  MINIMAL = "minimal"
  SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP = "save_dot_with_context_except_mlp"
  SAVE_DOT_EXCEPT_MLPWI = "save_dot_except_mlpwi"
  SAVE_DOT_EXCEPT_MLP = "save_dot_except_mlp"
  SAVE_QKV_PROJ = "save_qkv_proj"
  QKV_PROJ_OFFLOADED = "qkv_proj_offloaded"
  CUSTOM = "custom"
  MINIMAL_OFFLOADED = "minimal_offloaded"
  SAVE_OUT_PROJ = "save_out_proj"


class RematLocation(str, Enum):
  """Specifies where to store activations for rematerialization."""

  REMAT = "remat"
  DEVICE = "device"
  OFFLOAD = "offload"


class OptimizerType(str, Enum):
  """Supported optimizer algorithms."""

  ADAMW = "adamw"
  ADAM_PAX = "adam_pax"
  SGD = "sgd"
  MUON = "muon"


class LearningRateScheduleType(str, Enum):
  """Supported learning rate schedule types."""

  COSINE = "cosine"
  WSD = "wsd"


class WsdDecayStyle(str, Enum):
  """Supported decay styles for WSD schedule."""

  LINEAR = "linear"
  COSINE = "cosine"


class RopeType(str, Enum):
  """Supported Rotary Positional Embedding (RoPE) implementations."""

  DEFAULT = "default"
  LLAMA3_1 = "llama3.1"
  YARN = "yarn"


class TokenizerType(str, Enum):
  """Supported tokenizer libraries."""

  SENTENCEPIECE = "sentencepiece"
  HUGGINGFACE = "huggingface"
  TIKTOKEN = "tiktoken"


class DatasetType(str, Enum):
  """Supported data loading pipelines."""

  SYNTHETIC = "synthetic"
  HF = "hf"
  GRAIN = "grain"
  TFDS = "tfds"
  C4MLPERF = "c4_mlperf"


class SamplingStrategy(str, Enum):
  """Supported decoding and sampling strategies."""

  GREEDY = "greedy"
  WEIGHTED = "weighted"
  NUCLEUS = "nucleus"
  TOPK = "topk"
  COMPOSITE = "composite"


class ProfilerType(str, Enum):
  """Supported performance profilers."""

  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"


# ----------------------------------------------------------------------------
# Pydantic models for configuration
# ----------------------------------------------------------------------------

ModelName = Literal[
    "default",
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
    "llama3-8b",
    "llama3-70b",
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    "llama3.3-70b",
    "mistral-7b",
    "mixtral-8x7b",
    "mixtral-8x22b",
    "deepseek2-16b",
    "deepseek2-236b",
    "deepseek3-671b",
    "deepseek3-671b-2dfsdp",
    "deepseek3-test",
    "deepseek3-tiny",
    "deepseek3.2-671b",
    "deepseek-custom",
    "kimi-k2-1t",
    "gemma-7b",
    "gemma-2b",
    "gemma2-2b",
    "gemma2-9b",
    "gemma2-27b",
    "gemma3-4b",
    "gemma3-12b",
    "gemma3-27b",
    "qwen3-0.6b",
    "qwen3-4b",
    "qwen3-4b-thinking-2507",
    "qwen3-8b",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-480b-a35b",
    "qwen3-next-80b-a3b",
    "qwen3-omni-30b-a3b",
    "gpt3-175b",
    "gpt3-22b",
    "gpt3-6b",
    "gpt3-52k",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "llama4-17b-16e",
    "llama4-17b-128e",
    "olmo3-7b",
    "olmo3-7b-pt",
    "olmo3-32b",
]


class RunInfo(BaseModel):
  """Configuration for the overall run, model identity, and logging."""

  base_config: None | str = Field(
      None,
      description="Base config to inherit from. This is a meta-field and is consumed by the config loading system.",
  )
  run_name: str = Field(
      "",
      description="The name of the run. Checkpoints will be stored under this name.",
  )
  model_name: ModelName = Field("default", description="The name of the model configuration to use.")
  override_model_config: bool = Field(False, description="If True, allows overriding model parameters via CLI.")
  override_logical_axis_rules: bool = Field(
      False,
      description="If True, logical_axis_rules will be overridden instead of merged.",
  )
  log_config: bool = Field(
      True,
      description="If True, prints the final configuration after initialization.",
  )
  debug_sharding: bool = Field(False, description="If True, print model weight sharding details.")
  base_output_directory: PathStr = Field("", description="Base directory for all outputs, typically a GCS path.")
  sharding_strategy: None | Literal["experimental"] = Field(
      None,
      description="Experimental sharding strategy used for some inference configs.",
  )


class Checkpointing(BaseModel):
  """Core configuration for checkpointing and run restoration."""

  load_parameters_path: PathStr = Field("", description="Loads only model parameters from a specific checkpoint path.")
  lora_input_adapters_path: PathStr = Field("", description="Input GCS path for LoRA adapters.")
  load_full_state_path: PathStr = Field("", description="Loads the complete training state from a checkpoint path.")
  enable_checkpointing: bool = Field(True, description="If True, enables saving checkpoints during training.")
  load_checkpoint_only_once: bool = Field(False, description="If True, deep copy the reference model to the actor model.")
  async_checkpointing: bool = Field(True, description="If True, uses an asynchronous checkpointer for performance.")
  checkpoint_period: int = Field(10_000, description="The frequency (in steps) at which to save checkpoints.")
  max_num_checkpoints_to_keep: int | None = Field(None, description="Maximum number of checkpoints to keep.")
  enable_single_replica_ckpt_restoring: bool = Field(
      False, description="One replica reads and broadcasts the checkpoint."
  )
  force_unroll: bool = Field(
      False,
      description="During param-only checkpoint generation, whether to unroll the loop.",
  )
  checkpoint_is_quantized: bool = Field(
      False,
      description="Set to True if reading from a saved AQT quantized checkpoint.",
  )
  save_quantized_params_path: PathStr = Field("", description="Path to save params quantized on the fly.")
  enable_orbax_v1: bool = Field(False, description="Bool flag for enabling Orbax v1.")
  checkpoint_conversion_fn: None | str = Field(None, description="Function for processing loaded checkpoint dict.")
  source_checkpoint_layout: Literal["orbax", "safetensors"] = Field(
      "orbax", description="The layout of the source checkpoint to load."
  )
  save_checkpoint_on_completion: bool = Field(
      True, description="If True, saves a final checkpoint upon training completion."
  )
  enable_continuous_checkpointing: bool = Field(False, description="If True, enables continuous checkpointing.")


class OrbaxStorage(BaseModel):
  """Configuration for Orbax checkpoint storage options."""

  checkpoint_storage_target_data_file_size_bytes: int = Field(
      2147483648, description="Target file size for chunking large arrays in Orbax."
  )
  checkpoint_storage_use_ocdbt: bool = Field(True, description="Whether to use the OCDbT storage format for checkpoints.")
  checkpoint_storage_use_zarr3: bool = Field(
      True, description="Whether to use Zarr3 with OCDbT. Requires use_ocdbt=True."
  )
  checkpoint_storage_concurrent_gb: int = Field(96, description="Concurrent GB for I/O operations during checkpointing.")


class EmergencyCheckpointing(BaseModel):
  """Configuration for emergency (local) checkpointing."""

  enable_multi_tier_checkpointing: bool = Field(
      False, description="Enables multi-tier checkpointing (local and persistent)."
  )
  local_checkpoint_directory: PathStr = Field("", description="Local directory for emergency checkpoints.")
  local_checkpoint_period: NonNegativeInt = Field(0, description="Frequency (in steps) for local emergency checkpoints.")
  multi_tier_checkpointing_backup_interval_minutes: NonNegativeInt = Field(
      0,
      description="Interval in minutes to back up local checkpoints to persistent storage.",
  )
  mtc_data_parallelism: int = Field(
      0,
      description="Number of identical pipelines in the job for multi-tier checkpointing. 0 defaults to num_slices.",
  )
  enable_emergency_checkpoint: bool = Field(
      False,
      description="Legacy flag for enabling emergency checkpointing. Prefer `enable_multi_tier_checkpointing`.",
  )
  use_replicator_service: bool = Field(
      False,
      description="Whether to use emergency checkpointing with the replicator service.",
  )
  replicator_backup_interval_minutes: NonNegativeInt = Field(
      0, description="Interval in minutes to back up local checkpoints."
  )


class DataTypes(BaseModel):
  """Configuration for data types and precision."""

  dtype: DType = Field(DType.BFLOAT16, description="The data type for activations.")
  grad_dtype: DType = Field(DType.FLOAT32, description="The data type for gradients.")
  weight_dtype: DType = Field(DType.FLOAT32, description="The data type for model weights.")
  matmul_precision: MatmulPrecision = Field(
      MatmulPrecision.DEFAULT,
      description="Precision level for matrix multiplications.",
  )
  activations_in_float32: bool = Field(
      False,
      description="If True, sets activations to float32 before the nonlinearity.",
  )
  dtype_mm: str = Field("float32", description="Data type for multimodal model's vision encoder")


class Quantization(BaseModel):
  """Configuration for model quantization."""

  quantization: None | QuantizationType = Field(
      QuantizationType.NONE,
      description="Activates quantization for transformer layers.",
  )
  replicate_quant_scale: bool = Field(
      False,
      description="Replicates quantization scale to avoid inefficient XLA fusion.",
  )
  quant_cfg_path: PathStr = Field("", description="Path to the configuration file for 'intmp' quantization.")
  quantize_kvcache: bool = Field(False, description="If True, quantizes the Key-Value cache.")
  kv_quant_axis: KvQuantAxis = Field(KvQuantAxis.HEADS_AND_DKV, description="Axes to quantize over for the KV cache.")
  kv_quant_dtype: Literal["int8", "int4"] = Field("int8", description="Data type for KV cache quantization.")
  quantization_local_shard_count: int = Field(-1, description="Shards the range finding operation for quantization.")
  use_qwix_quantization: bool = Field(False, description="Whether to use qwix for quantization.")
  weight_quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for weights.",
  )
  act_quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for activations.",
  )
  bwd_quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for gradients.",
  )


class ModelArchitecture(BaseModel):
  """Core model architecture parameters."""

  decoder_block: DecoderBlockType = Field(
      "llama2",
      description="The style of DecoderBlock to use (e.g., 'llama2', 'gemma').",
  )
  global_parameter_scale: int = Field(1, description="A global scaling factor for model dimensions.")
  base_emb_dim: int = Field(2048, description="Base embedding dimension.")
  base_num_query_heads: int = Field(16, description="Base number of query heads.")
  base_num_kv_heads: int = Field(16, description="Base number of key/value heads.")
  base_mlp_dim: int = Field(7168, description="Base dimension of the MLP layer.")
  base_num_decoder_layers: int = Field(16, description="Base number of decoder layers.")
  head_dim: int = Field(128, description="Dimension of each attention head.")
  mlp_activations: list[str] = Field(["silu", "linear"], description="Activation functions in the MLP layer.")
  mlp_activations_limit: float = Field(
      -1.0,
      description="Upper bound to clip the MLP activation values. -1.0 means no clipping.",
  )
  normalization_layer_epsilon: float = Field(1.0e-05, description="Epsilon value for normalization layers.")
  fused_qkv: bool = Field(False, description="If supported, fuse the Q, K, and V projections.")
  attention_bias: bool = Field(
      False,
      description="If True, adds a learnable bias to the query, key, and value projections.",
  )
  fused_mlp: bool = Field(False, description="If supported, fuse the MLP layers.")


class MTP(BaseModel):
  """Multi-Token Prediction Configs."""

  mtp_num_layers: NonNegativeInt = Field(0, description="The number of auxiliary prediction layers to use for MTP.")
  mtp_loss_scaling_factor: NonNegativeFloat = Field(
      0.1,
      description="The scaling factor (lambda) for the MTP auxiliary loss.",
  )
  mtp_eval_target_module: NonNegativeInt = Field(
      0,
      description="Specifies which MTP layer is used to calculate metrics.",
  )


class Logits(BaseModel):
  """Configuration for the final logits computation."""

  logits_via_embedding: bool = Field(False, description="If True, tie the embedding and unembedding matrices.")
  normalize_embedding_logits: bool = Field(
      True,
      description="If logits_via_embedding is true, normalize pre-softmax logits.",
  )
  logits_dot_in_fp32: bool = Field(False, description="Use fp32 for the logits dot product for stability.")
  cast_logits_to_fp32: bool = Field(True, description="Whether to cast the final logits to fp32.")
  final_logits_soft_cap: None | NonNegativeFloat = Field(
      None,
      description="Soft-cap value for the final logits. None or 0.0 means no cap.",
  )


class Attention(BaseModel):
  """General configuration for the attention mechanism."""

  attention: str = Field(
      "autoselected",
      description="The attention algorithm to use (dot_product, flash, etc).",
  )
  attention_type: Literal["global", "local_sliding", "chunk", "mla", "full"] = Field(
      "global", description="The variant of attention to use."
  )
  attention_sink: bool = Field(False, description="If True, enables attention sinks.")
  float32_qk_product: bool = Field(False, description="In dot-product attention, cast query-key product to fp32.")
  float32_logits: bool = Field(
      False,
      description="In dot-product attention, cast logits to fp32 before softmax.",
  )
  sliding_window_size: NonNegativeInt = Field(0, description="The size of the sliding window for local attention.")
  chunk_attn_window_size: NonNegativeInt = Field(0, description="The window size for chunked attention.")
  attn_logits_soft_cap: None | NonNegativeFloat = Field(
      None, description="Soft-cap value for attention logits. None means no cap."
  )
  use_post_attn_norm: bool = Field(False, description="Apply LayerNorm after the attention block.")
  use_post_ffw_norm: bool = Field(False, description="Apply LayerNorm after the feed-forward block.")
  use_ragged_attention: bool = Field(False, description="Whether to use ragged attention kernels.")
  use_tokamax_gmm: bool = Field(
      False,
      description="Whether to use the Tokamax library for GMM kernel implementation.",
  )
  ragged_block_size: int = Field(256, description="Block size for ragged attention.")
  enable_padding_causal_mask: bool = Field(True, description="Temporary flag for TE padding.")
  use_tokamax_splash: bool = Field(False, description="Whether to use tokamax splash attention.")
  use_jax_splash: bool = Field(False, description="Whether to use jax splash attention.")
  force_q_layout: bool = Field(False, description="Force the Q layout")
  use_qk_clip: bool = Field(False, description="Whether to use QK-Clip (MuonClip) for training stability.")
  qk_clip_threshold: float = Field(100.0, description="Threshold for QK-Clip (tau).")


class MoBa(BaseModel):
  """Configuration for Mixture of Block Attention (MoBA)."""

  moba: bool = Field(False, description="If True, enables Mixture of Block Attention.")
  moba_chunk_size: int = Field(1024, description="The chunk size for MoBA.")
  moba_topk: int = Field(8, description="The number of top-k chunks to select in MoBA.")


class MlaAttention(BaseModel):
  """Configuration for Multi-Layer Attention (MLA)."""

  mla_naive_kvcache: bool = Field(True, description="Whether to use naive kvcache for MLA attention.")
  q_lora_rank: NonNegativeInt = Field(0, description="Query LoRA rank for MLA.")
  kv_lora_rank: NonNegativeInt = Field(512, description="Key/Value LoRA rank for MLA.")
  qk_nope_head_dim: NonNegativeInt = Field(128, description="Dimension for non-RoPE part of QK heads in MLA.")
  qk_rope_head_dim: NonNegativeInt = Field(64, description="Dimension for RoPE part of QK heads in MLA.")
  v_head_dim: NonNegativeInt = Field(128, description="Dimension of V heads in MLA.")


class AttentionIndexer(BaseModel):
  """Configuration for DeepSeek Sparse Attention (DSA): DeepSeek3.2-style MLA with indexer."""

  use_sparse_indexer: bool = Field(False, description="Whether to use sparse indexer for MLA.")
  index_head_dim: NonNegativeInt = Field(128, description="Head dim for indexer query and key.")
  index_n_heads: NonNegativeInt = Field(64, description="Number of query heads in indexer.")
  index_topk: NonNegativeInt = Field(2048, description="Number of tokens selected by the query token in indexer.")


class Llama4Attention(BaseModel):
  """Configuration specific to Llama4-style models."""

  use_qk_norm: bool = Field(
      False,
      description="Whether to apply L2 normalization to Query/Key vectors after RoPE.",
  )
  temperature_tuning: bool = Field(
      False,
      description="Dynamically scale attention temperature based on sequence length.",
  )


class SplashAttention(BaseModel):
  """Tunable block sizes for Splash Attention kernels."""

  sa_block_q: int = Field(512, description="Block size for Q in splash attention.")
  sa_block_kv: int = Field(512, description="Block size for KV in splash attention.")
  sa_block_kv_compute: int = Field(512, description="Block size for KV compute in splash attention.")
  sa_block_q_dkv: int = Field(512, description="Block size for Q_dkv in splash attention.")
  sa_block_kv_dkv: int = Field(512, description="Block size for KV_dkv in splash attention.")
  sa_block_kv_dkv_compute: int = Field(512, description="Block size for KV_dkv compute in splash attention.")
  sa_block_q_dq: int = Field(512, description="Block size for Q_dq in splash attention.")
  sa_block_kv_dq: int = Field(512, description="Block size for KV_dq in splash attention.")
  sa_use_fused_bwd_kernel: bool = Field(False, description="Use fused backward kernel in splash attention.")
  sa_q_layout: str = Field("HEAD_DIM_MINOR", description="Layout for Q in splash attention.")
  sa_k_layout: str = Field("HEAD_DIM_MINOR", description="Layout for K in splash attention.")
  sa_v_layout: str = Field("HEAD_DIM_MINOR", description="Layout for V in splash attention.")
  use_max_logit_estimate: int = Field(
      -1,
      description="-1 means no estimate, any > 0 value will be used as max logit estimate",
  )
  cost_estimate_flops_fwd: int = Field(
      -1,
      description="-1 means using splash default cost estimation, any >= 0 value will be used as cost estimation for splash "
      "to overlap for communication (forward)",
  )
  cost_estimate_flops_bwd: int = Field(
      -1,
      description="-1 means using splash default cost estimation, any >= 0 value will be used as cost estimation for splash "
      "to overlap for communication (backward)",
  )
  dq_reduction_steps: int = Field(
      0,
      description="the number of reduction steps. For now, only 3 or all the kv steps are supported.",
  )
  use_splash_scheduler: bool = Field(False, description="Use experimental splash attention scheduler.")


class PagedAttention(BaseModel):
  """Tunable parameters for Paged Attention kernels."""

  pagedattn_num_pages: int = Field(64, description="Total number of pages to allocate for paged attention.")
  pagedattn_tokens_per_page: int = Field(32, description="Number of tokens each page can hold.")
  pagedattn_pages_per_compute_block: int = Field(4, description="Number of pages processed together in pallas kernels.")
  pagedattn_max_pages_per_group: int = Field(-1, description="Max pages per request; -1 defaults to max_target_length.")
  # Alignment of head_dim to the nearest multiple of this value, set to 0 to disable alignment. On
  # TPUs, the head_dim is padded to the nearest multiple of 128.
  pagedattn_head_dim_alignment: int = Field(128, description="Alignment of head_dim to the nearest multiple.")


class MoEGeneral(BaseModel):
  """General configuration for Mixture of Experts (MoE) layers."""

  num_experts: PositiveInt = Field(1, description="The total number of experts in each MoE layer.")
  num_experts_per_tok: PositiveInt = Field(1, description="The number of experts to route each token to.")
  capacity_factor: float = Field(-1.0, description="Expert capacity factor. If < 0, no token dropping.")
  load_balance_loss_weight: NonNegativeFloat = Field(0.0, description="Weight for the load balancing auxiliary loss.")
  use_custom_sort_vjp: bool = Field(
      True, description="Whether to use a custom VJP sort for efficient backward pass processing in sparse matmul."
  )
  use_ring_of_experts: bool = Field(
      False,
      description="Whether to use Ring of Experts for sparse matmul expert parallelism.",
  )
  use_random_routing: bool = Field(False, description="Whether to use random routing for debugging.")
  interleave_moe_layer_step: int = Field(1, description="Frequency of MoE layers, e.g., 2 means every 2nd layer is MoE.")
  expert_shard_attention_option: Literal["fsdp", "context"] = Field(
      "fsdp",
      description="How the expert axis is used to shard attention weights and activations.",
  )
  moe_fsdp_use_two_stage_all_gather: bool = Field(
      False,
      description="Use two separate All-Gather calls for MoE weights sharded on both FSDP and FSDP-transpose.",
  )
  shard_exp_on_fsdp: bool = Field(
      False,
      description="Shard the expert dimension of the MLP weights on the FSDP axis, "
      "and recommended only when num_experts is a multiple of fsdp_parallelism",
  )
  use_2d_fsdp_sharding: bool = Field(
      False,
      description="Use `fsdp` and `fsdp_transpose` axes for 2D FSDP sharding.",
  )
  norm_topk_prob: bool = Field(
      False,
      description="Enable top-k probability normalization for router weights (Qwen3-specific).",
  )
  float32_weight_sum: bool = Field(
      True,
      description="Whether to use full fp32 precision to sum expert weights for numerical stability.",
  )


class MoEKernels(BaseModel):
  """Configuration for MoE-specific kernels like Megablox."""

  megablox: bool = Field(True, description="Whether to use Megablox kernels for MoE.")
  sparse_matmul: bool = Field(True, description="Whether to use sparse matmul kernels for MoE.")
  wi_tile_fwd_batch_seq: int = Field(
      512,
      description="forward pass tiling dimension for batch/sequence in GMM for wi.",
  )
  wi_tile_fwd_embed_dim: int = Field(1024, description="forward pass tiling dimension for embedding in GMM for wi.")
  wi_tile_fwd_mlp_dim: int = Field(1024, description="forward pass tiling dimension for MLP in GMM for wi.")
  wi_tile_dlhs_batch_seq: int = Field(
      512,
      description="bwd pass dlhs tiling dimension for batch/sequence in GMM for wi.",
  )
  wi_tile_dlhs_embed_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for embedding in GMM for wi.")
  wi_tile_dlhs_mlp_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for MLP in GMM for wi.")
  wi_tile_drhs_batch_seq: int = Field(
      512,
      description="bwd pass drhs tiling dimension for batch/sequence in GMM for wi.",
  )
  wi_tile_drhs_embed_dim: int = Field(1024, description="bwd pass drhs tiling dimension for embedding in GMM for wi.")
  wi_tile_drhs_mlp_dim: int = Field(1024, description="bwd pass drhs tiling dimension for MLP in GMM for wi.")
  wo_tile_fwd_batch_seq: int = Field(
      512,
      description="forward pass tiling dimension for batch/sequence in GMM for wo.",
  )
  wo_tile_fwd_embed_dim: int = Field(1024, description="forward pass tiling dimension for embedding in GMM for wo.")
  wo_tile_fwd_mlp_dim: int = Field(1024, description="forward pass tiling dimension for MLP in GMM for wo.")
  wo_tile_dlhs_batch_seq: int = Field(
      512,
      description="bwd pass dlhs tiling dimension for batch/sequence in GMM for wo.",
  )
  wo_tile_dlhs_embed_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for embedding in GMM for wo.")
  wo_tile_dlhs_mlp_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for MLP in GMM for wo.")
  wo_tile_drhs_batch_seq: int = Field(
      512,
      description="bwd pass drhs tiling dimension for batch/sequence in GMM for wo.",
  )
  wo_tile_drhs_embed_dim: int = Field(1024, description="bwd pass drhs tiling dimension for embedding in GMM for wo.")
  wo_tile_drhs_mlp_dim: int = Field(1024, description="bwd pass drhs tiling dimension for MLP in GMM for wo.")

  wi_tile_fwd_buffer_count: int = Field(2, description="forward pass tiling buffer count in GMM for wi.")
  wi_tile_dlhs_buffer_count: int = Field(2, description="bwd pass dlhs tiling buffer count in GMM for wi.")
  wi_tile_drhs_buffer_count: int = Field(2, description="bwd pass drhs tiling buffer count in GMM for wi.")
  wo_tile_fwd_buffer_count: int = Field(2, description="forward pass tiling buffer count in GMM for wo.")
  wo_tile_dlhs_buffer_count: int = Field(2, description="bwd pass dlhs tiling buffer count in GMM for wo.")
  wo_tile_drhs_buffer_count: int = Field(2, description="bwd pass drhs tiling buffer count in GMM for wo.")

  wi_combine_scopes: bool = Field(False, description="whether to use combine_scopes features for tgmm for wi.")
  wo_combine_scopes: bool = Field(False, description="whether to use combine_scopes features for tgmm for wo.")


class DeepSeekMoE(BaseModel):
  """Configuration specific to DeepSeek-style MoE layers."""

  base_moe_mlp_dim: int = Field(7168, description="Intermediate dimension at MoE layer (DeepSeek style).")
  first_num_dense_layers: NonNegativeInt = Field(0, description="Number of initial dense layers in the model.")
  shared_experts: PositiveInt = Field(1, description="Number of shared experts.")
  routed_scaling_factor: float = Field(1.0, description="Scaling factor for routing scores.")
  routed_score_func: str = Field("", description="Scoring function for routing (e.g., 'softmax', 'sigmoid').")
  routed_bias: bool = Field(False, description="Whether to add a bias term for routing.")
  routed_bias_update_rate: float = Field(0.0, description="Update rate applied to the router bias term.")
  mlp_bias: bool = Field(
      False,
      description="Whether to add a learnable bias for MLP matmul, "
      "and originally implemented to support the GPT-OSS model architecture",
  )
  n_routing_groups: int = Field(-1, description="Number of groups for routing, disabled by default.")
  topk_routing_group: int = Field(-1, description="Number of top groups to route inputs to.")
  use_batch_split_schedule: bool = Field(
      False,
      description="Whether to split batch into micro-batches to hide communications that yields performance benefits.",
  )
  batch_split_factor: int = Field(
      1,
      description="Factor by which to split the batch into micro-batches. Only used if use_batch_split_schedule is True.",
  )


class Qwen3Next(BaseModel):
  """Configuration specific to Qwen3-Next models with Gated Delta Net."""

  gdn_conv_kernel_dim: int = Field(4, description="Kernel size for the 1D convolution in the Gated Delta Net.")
  gdn_key_head_dim: int = Field(128, description="Head dimension for the key/query in the Gated Delta Net.")
  gdn_value_head_dim: int = Field(128, description="Head dimension for the value in the Gated Delta Net.")
  gdn_num_key_heads: int = Field(16, description="Number of key/query heads in the Gated Delta Net.")
  gdn_num_value_heads: int = Field(32, description="Number of value heads in the Gated Delta Net.")
  gdn_chunk_size: int = Field(
      64,
      description="Chunk size for the parallel scan algorithm in the Gated Delta Net.",
  )
  use_qk_norm_in_gdn: bool = Field(
      True,
      description="Whether to apply L2 normalization to query and key tensors inside the Gated Delta Rule kernel.",
  )
  partial_rotary_factor: float = Field(1.0, description="The ratio of dimension to apply ROPE on")


class HardwareAndMesh(BaseModel):
  """Configuration for hardware and parallelism mesh."""

  hardware: Literal["tpu", "gpu", "gpu_multiprocess", "cpu"] = Field("tpu", description="The type of hardware to run on.")
  num_slices: int = Field(-1, description="Number of TPU slices. Automatically determined.")
  mesh_axes: list[str] = Field(
      [
          "data",
          "stage",
          "fsdp",
          "fsdp_transpose",
          "sequence",
          "context",
          "context_autoregressive",
          "tensor",
          "tensor_transpose",
          "tensor_sequence",
          "expert",
          "autoregressive",
      ],
      description="The names of the axes in the logical device mesh.",
  )
  shard_mode: ShardMode = Field("auto", description="can be either auto or explicit")
  inhomogeneous_layer_cycle_interval: int = Field(1, description="The interval of repeated inhomogeneous layer patterns.")
  scan_layers: bool = Field(True, description="Whether to use jax.lax.scan over layers.")
  param_scan_axis: int = Field(1, description="Axis to scan over for parameters.")
  context_parallel_load_balance: bool = Field(True, description="Whether to use load balancing for context parallelism.")
  context_parallel_strategy: str = Field(
      "all_gather",
      description="Strategy for context parallelism ('all_gather' or 'ring').",
  )
  custom_mesh: str = Field("", description="Available options: ['hybrid_ring_64x4', 'hybrid_ring_32x8']")
  allow_split_physical_axes: bool = Field(False, description="Allow splitting physical axes for device mesh creation.")
  enable_nnx: bool = Field(False, description="Whether to use NNX for model definition.")
  optimize_mesh_for_tpu_v6e: bool = Field(False, description="Apply transformations to the mesh for TPU v6e.")
  shardy: bool = Field(True, description="Whether to use shardy XLA backend.")


class LayoutAndSharding(BaseModel):
  """Configuration for data and model sharding rules."""

  logical_axis_rules: Any = Field([], description="Rules for mapping logical axes to physical mesh axes.")
  data_sharding: Any = Field([], description="Sharding for input data.")
  input_data_sharding_logical_axes: list[str] = Field(
      ["activation_embed_and_logits_batch", "activation_norm_length"],
      description="Logical axes for sharding input data.",
  )
  sharding_tolerance: float = Field(
      0.02,
      ge=0.0,
      le=1.0,
      description="Allowed percentage of non-sharded parameters.",
  )
  shard_optimizer_over_data: bool = Field(False, description="Enable ZeRO-1 optimizer sharding over the data axis.")


class DcnParallelism(BaseModel):
  """Parallelism dimensions across the DCN (Data Center Network)."""

  dcn_diloco_parallelism: int = Field(1, description="DCN axis for Diloco parallelism.")
  dcn_data_parallelism: int = Field(-1, description="DCN axis for data parallelism.")
  dcn_fsdp_parallelism: int = Field(1, description="DCN axis for FSDP.")
  dcn_fsdp_transpose_parallelism: int = Field(1, description="DCN axis for FSDP transpose.")
  dcn_sequence_parallelism: int = Field(1, description="DCN axis for sequence parallelism (not recommended).")
  dcn_context_parallelism: int = Field(1, description="DCN axis for context parallelism.")
  dcn_context_autoregressive_parallelism: int = Field(1, description="DCN axis for context autoregressive parallelism.")
  dcn_tensor_parallelism: int = Field(1, description="DCN axis for tensor parallelism (not recommended).")
  dcn_tensor_transpose_parallelism: int = Field(1, description="DCN axis for tensor transpose parallelism.")
  dcn_tensor_sequence_parallelism: int = Field(
      1, description="DCN axis for tensor sequence parallelism (not recommended)."
  )
  dcn_pipeline_parallelism: int = Field(1, description="DCN axis for pipeline parallelism.")
  dcn_expert_parallelism: int = Field(1, description="DCN axis for expert parallelism.")
  dcn_autoregressive_parallelism: int = Field(1, description="DCN axis for autoregressive parallelism (not recommended).")


class IciParallelism(BaseModel):
  """Parallelism dimensions within the ICI (Inter-Chip Interconnect)."""

  ici_diloco_parallelism: int = Field(1, description="ICI axis for Diloco parallelism.")
  ici_data_parallelism: int = Field(1, description="ICI axis for data parallelism.")
  ici_fsdp_parallelism: int = Field(-1, description="ICI axis for FSDP.")
  ici_fsdp_transpose_parallelism: int = Field(1, description="ICI axis for FSDP transpose.")
  ici_sequence_parallelism: int = Field(1, description="ICI axis for sequence parallelism.")
  ici_context_parallelism: int = Field(1, description="ICI axis for context parallelism.")
  ici_context_autoregressive_parallelism: int = Field(1, description="ICI axis for context autoregressive parallelism.")
  ici_tensor_parallelism: int = Field(1, description="ICI axis for tensor parallelism.")
  ici_tensor_transpose_parallelism: int = Field(1, description="ICI axis for tensor transpose parallelism.")
  ici_tensor_sequence_parallelism: int = Field(1, description="ICI axis for tensor sequence parallelism.")
  ici_autoregressive_parallelism: int = Field(1, description="ICI axis for autoregressive parallelism.")
  ici_pipeline_parallelism: int = Field(1, description="ICI axis for pipeline parallelism.")
  ici_expert_parallelism: int = Field(1, description="ICI axis for expert parallelism.")


class PipelineParallelism(BaseModel):
  """Configuration for pipeline parallelism."""

  num_layers_per_pipeline_stage: int = Field(1, description="Number of layers to place on each pipeline stage.")
  num_pipeline_repeats: int = Field(
      -1,
      description="Number of pipeline repeats. Calculated from other params if -1.",
  )
  pipeline_parallel_layers: int = Field(-1, description="Number of layers to pipeline. -1 pipelines all decoder layers.")
  num_pipeline_microbatches: int = Field(
      -1,
      description="Number of microbatches for the pipeline. -1 defaults to num_stages.",
  )
  pipeline_delay_activation_forwarding: bool = Field(
      False, description="Delays activation forwarding to aid XLA optimization."
  )
  pipeline_fsdp_ag_once: bool = Field(False, description="If True, all-gather FSDP weights once per pipeline repeat.")
  scan_pipeline_iterations: bool = Field(True, description="Use jax.lax.scan over pipeline iterations.")
  scan_layers_per_stage: bool = Field(False, description="Use jax.lax.scan over layers within a stage.")
  set_remat_policy_on_pipeline_iterations: bool = Field(True, description="Set remat policy on the pipeline scan.")
  set_remat_policy_on_layers_per_stage: bool = Field(False, description="Set remat policy on the inner layer scan.")


class RematAndOffload(BaseModel):
  """Configuration for gradient checkpointing (rematerialization) and offloading."""

  remat_policy: str = Field(
      RematPolicy.FULL.value,
      description="The rematerialization policy, trading off speed and memory.",
  )
  remat_policy_for_vit: str = Field("minimal", description="Remat policy for multimodal model's vision encoder.")
  decoder_layer_input: RematLocation = Field(
      RematLocation.DEVICE, description="Remat policy for the decoder layer's input."
  )
  context: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the attention context.")
  mlpwi: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the first MLP layer's intermediate output.",
  )
  mlpwi_0: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the first part of a gated MLP's output.",
  )
  mlpwi_1: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the second part of a gated MLP's output.",
  )
  mlpwo: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the second MLP layer's output.",
  )
  query_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the query projection.")
  key_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the key projection.")
  value_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the value projection.")
  qkv_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for fused QKV projection.")
  out_proj: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the attention output projection.",
  )
  mla_q: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the mla's query projectiont.",
  )
  mla_kv: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the mla's key and value projection.",
  )
  attention_out: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the attention output.",
  )
  engram: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the engram output.")

  optimizer_memory_host_offload: bool = Field(False, description="Offload optimizer state to host memory.")
  parameter_memory_host_offload: bool = Field(False, description="Offload parameters to host memory.")


class Tokenizer(BaseModel):
  """Configuration for the tokenizer."""

  vocab_size: int = Field(32_000, description="The size of the vocabulary.")
  tokenizer_path: PathStr = Field(
      os.path.join("assets", "tokenizers", "tokenizer.llama2"),
      description="Path to the tokenizer model file.",
  )
  tokenizer_type: TokenizerType = Field(TokenizerType.SENTENCEPIECE, description="The type of tokenizer.")
  use_chat_template: bool = Field(False, description="Whether to use the chat template for tokenization.")
  chat_template_path: str = Field("", description="Path to chat template json file.")
  tokenize_train_data: bool = Field(True, description="If False, assumes the training dataset is pre-tokenized.")
  tokenize_eval_data: bool = Field(True, description="If False, assumes the evaluation dataset is pre-tokenized.")
  add_bos: bool = Field(True, description="Whether to add a beginning-of-sentence token.")
  add_eos: bool = Field(True, description="Whether to add an end-of-sentence token.")
  use_truncation: bool = Field(
      True,
      description="If False, use chunking for long sequences instead of truncation.",
  )
  num_vocab_tiling: int = Field(
      1,
      description="Enables memory-saving optimization by tiling cross-entropy loss computation. >1 to enable.",
  )


class DatasetGeneral(BaseModel):
  """General configuration for dataset and data loading."""

  dataset_type: DatasetType = Field(DatasetType.TFDS, description="The type of the data loading pipeline.")
  per_device_batch_size: int | float = Field(12, description="The batch size per device.")
  eval_per_device_batch_size: int | float = Field(
      0.0,
      description="The batch size per device for evaluation. Defaults to per_device_batch_size.",
  )
  max_corpus_chars: int = Field(10_000_000, description="Maximum number of characters to use from the corpus.")
  train_data_columns: list[str] = Field(["text"], description="Column(s) to use from the training data.")
  train_image_column: str | list[str] = Field("image", description="Column name(s) for images in the training data.")
  eval_data_columns: list[str] = Field(["text"], description="Column(s) to use from the evaluation data.")
  eval_image_column: str | list[str] = Field("image", description="Column name(s) for images in evaluation data.")
  packing: bool = Field(
      True,
      description="Whether to pack multiple short examples into a single sequence.",
  )
  grain_packing_type: Literal["first_fit", "best_fit", "concat_then_split"] = Field(
      "first_fit",
      description="Packing type when using Grain pipeline. 'first_fit', 'best_fit' or 'concat_then_split'.",
  )
  max_segments_per_seq: int = Field(
      -1,
      description="Maximum number of segments that can be packed into a single sequence. -1 or None for no limit.",
  )
  num_epoch: int = Field(1, description="Number of epochs to train for.")
  expansion_factor_real_data: float = Field(-1.0, description="Factor for partial data loading on hosts.")
  reuse_example_batch: int = Field(0, description="For performance testing, repeatedly uses the same batch.")
  generate_padding_batch_train: bool = Field(
      False,
      description="Whether to generate a padding batch for training to ensure divisibility.",
  )
  generate_padding_batch_eval: bool = Field(
      False,
      description="Whether to generate a padding batch for evaluation to ensure divisibility.",
  )
  enable_rampup_batch_size: bool = Field(False, description="Enable rampup batch size.")
  per_device_batch_size_start: float = Field(4.0, description="Start per device batch size for rampup.")
  per_device_batch_size_increment: float = Field(2.0, description="Increment for per device batch size for rampup.")
  global_rampup_samples: int = Field(500, description="Target number of training samples for rampup.")
  colocated_python_data_input: bool = Field(False, description="Experimental feature for Pathways.")


class TfdsDataset(BaseModel):
  """Configuration specific to TFDS datasets."""

  dataset_path: PathStr = Field("", description="Path to the TFDS dataset.")
  dataset_name: str = Field("c4/en:3.0.1", description="Name of the TFDS dataset.")
  eval_dataset_name: str = Field("c4/en:3.0.1", description="Name of the TFDS eval dataset.")
  train_split: str = Field("train", description="Dataset split for training.")
  eval_split: str = Field("validation", description="Dataset split for evaluation.")


class HfDataset(BaseModel):
  """Configuration specific to HuggingFace datasets."""

  hf_path: str = Field("", description="Path of the Hugging Face dataset.")
  hf_name: str = Field("", description="Name of the Hugging Face dataset.")
  hf_data_dir: PathStr = Field("", description="Data directory for the HF dataset.")
  hf_train_files: Optional[str] = Field(None, description="Files for the HF training split.")
  hf_eval_split: str = Field("", description="Name of the HF evaluation split.")
  hf_eval_files: Optional[str] = Field(None, description="Files for the HF evaluation split.")
  hf_access_token: None | str = Field(None, description="Hugging Face API access token.")


class GrainDataset(BaseModel):
  """Configuration specific to Grain datasets."""

  grain_train_files: PathStr = Field("", description="Path to Grain training files.")
  grain_eval_files: PathStr = Field("", description="Path to Grain evaluation files.")
  grain_train_mixture_config_path: PathStr = Field(
      "", description="Path to a JSON file specifying the mixture weights for Grain training data."
  )
  grain_file_type: str = Field("arrayrecord", description="File type for Grain data.")
  grain_worker_count: int = Field(1, description="Number of workers for Grain data loading.")
  grain_per_worker_buffer_size: int = Field(
      1,
      description="Buffer size for each worker for Grain data loading during training.",
  )
  grain_worker_count_eval: int = Field(1, description="Number of workers for Grain eval data loading.")
  grain_per_worker_buffer_size_eval: int = Field(
      1,
      description="Buffer size for each worker for Grain data loading during evaluation.",
  )
  grain_ram_budget_mb: int = Field(1024, description="RAM budget (MB) for auto-tuning worker count.")
  grain_num_threads: int = Field(16, description="Number of threads for Grain ReadOptions during training.")
  grain_prefetch_buffer_size: int = Field(500, description="Prefetch buffer size for Grain ReadOptions during training.")
  grain_num_threads_eval: int = Field(16, description="Number of threads for Grain ReadOptions during evaluation.")
  grain_prefetch_buffer_size_eval: int = Field(
      500, description="Prefetch buffer size for Grain ReadOptions during evaluation."
  )
  grain_data_source_max_workers: int = Field(
      16,
      description="Max workers for ThreadPoolExecutor when mixing multiple Grain data sources.",
  )


class FineTuning(BaseModel):
  """Configuration for fine-tuning methods like DPO, SFT, and GRPO."""

  use_dpo: bool = Field(False, description="If True, enables Direct Preference Optimization training.")
  dpo_label_smoothing: float = Field(0.0, ge=0.0, le=1.0, description="Label smoothing for DPO.")
  dpo_beta: float = Field(0.1, description="Beta parameter for DPO.")
  use_sft: bool = Field(False, description="If True, enables Supervised Fine-Tuning.")
  sft_train_on_completion_only: bool = Field(
      False, description="If True, trains only on the completion part of the text."
  )
  use_grpo: None | bool = Field(None, description="If True, enables Group Relative Policy Optimization.")


class Distillation(BaseModel):
  """Configuration for Knowledge Distillation."""

  # --- Overrides ---
  # These dictionaries allow flexible configuration injection for Student/Teacher
  # without needing to duplicate the entire MaxText schema here.
  student_overrides: dict[str, Any] = Field(
      default_factory=dict, description="Overrides specific to the Student model (e.g., {'num_query_heads': 16})."
  )
  teacher_overrides: dict[str, Any] = Field(
      default_factory=dict, description="Overrides specific to the Teacher model (e.g., {'num_query_heads': 64})."
  )

  # --- Loss Params ---
  distill_alpha: float = Field(0.5, description="Weight for the distillation loss component.")
  distill_temperature: float = Field(1.0, description="Temperature for distillation softening.")


class TrainingLoop(BaseModel):
  """Configuration for the main training loop, evaluation, and reproducibility."""

  steps: int = Field(
      150_001,
      ge=-1,
      description="Total number of training steps. -1 defaults to learning_rate_schedule_steps.",
  )
  log_period: int = Field(100, description="Frequency (in steps) to log metrics and flush Tensorboard.")
  eval_interval: int = Field(
      -1,
      description="Run evaluation every N training steps. -1 disables interval-based evaluation.",
  )
  eval_steps: int = Field(
      -1,
      description="Number of steps to run for each evaluation. -1 runs on entire eval split.",
  )
  target_eval_loss: float = Field(
      0.0,
      description="If set, training will stop early when this evaluation loss is reached.",
  )
  enable_dropout: bool = Field(True, description="Enables dropout in the model.")
  dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="The dropout rate.")
  enable_data_shuffling: bool = Field(True, description="Enables shuffling of the training data.")
  data_shuffle_seed: int = Field(0, description="Seed for data shuffling.")
  init_weights_seed: int = Field(0, description="Seed for model weight initialization.")


class ManifoldConstrainedHyperConnections(BaseModel):
  """Configuration for DeepSeek Manifold-Constrained Hyper Connections (mHC)."""

  mhc_expansion_rate: PositiveInt = Field(1, description="The number of parallel streams in Hyper Connection.")
  sinkhorn_iterations: PositiveInt = Field(20, description="The number of iterations for the Sinkhorn-Knopp algorithm.")


class DilocoParams(BaseModel):
  """Diloco Hyperparameters"""

  enable_diloco: bool = Field(False, description="Enable Diloco parallelism")
  diloco_sync_period: int = Field(36, description="Diloco sync period.")
  diloco_outer_lr: float = Field(0.3, description="learning rate for outer optimizer.")
  diloco_outer_momentum: float = Field(0.9, description="momentum for outer optimizer.")


class Optimizer(BaseModel):
  """Configuration for the optimizer and learning rate schedule."""

  opt_type: OptimizerType = Field(OptimizerType.ADAMW, description="The type of optimizer to use.")
  gradient_accumulation_steps: PositiveInt = Field(
      1, description="Number of steps to accumulate gradients before updating."
  )
  use_tunix_gradient_accumulation: bool = Field(
      False,
      description="Whether to use the Tunix implementation for gradient accumulation.",
  )
  gradient_clipping_threshold: NonNegativeFloat = Field(
      1.0, description="The threshold for gradient clipping. 0 disables clipping."
  )
  learning_rate: NonNegativeFloat = Field(3.0e-5, description="The peak learning rate.")
  lr_schedule_type: LearningRateScheduleType = Field(
      LearningRateScheduleType.COSINE, description="The type of learning rate schedule to use."
  )
  learning_rate_final_fraction: float = Field(
      0.1, description="Final LR as a fraction of peak LR (applies to both cosine and WSD schedules)."
  )
  wsd_decay_steps_fraction: float = Field(
      0.1, ge=0.0, le=1.0, description="Fraction of total steps for decay phase in WSD schedule."
  )
  wsd_decay_style: WsdDecayStyle = Field(
      WsdDecayStyle.LINEAR, description="The decay style for WSD schedule ('linear' or 'cosine')."
  )
  warmup_steps_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of total steps for LR warmup.")
  learning_rate_schedule_steps: int = Field(
      -1,
      ge=-1,
      description="Total steps for the LR schedule. -1 defaults to `steps`.",
  )


class AdamW(BaseModel):
  """Configuration specific to the AdamW optimizer."""

  adam_b1: float = Field(
      0.9,
      description="Exponential decay rate for the first moment of past gradients (beta1).",
  )
  adam_b2: float = Field(
      0.95,
      description="Exponential decay rate for the second moment of past gradients (beta2).",
  )
  adam_eps: float = Field(
      1.0e-8,
      description="A small constant for numerical stability (epsilon), applied outside of the square root.",
  )
  adam_eps_root: float = Field(
      0.0,
      description="A small constant for numerical stability (epsilon), applied inside of the square root.",
  )
  adam_weight_decay: float = Field(0.1, description="Weight decay regularization.")
  mu_dtype: str = Field(
      "",
      description="Data type for 'mu' (first moment) in AdamW. Inherits from weight_dtype if empty.",
  )


class Muon(BaseModel):
  """Configuration specific to the Muon optimizer."""

  muon_beta: float = Field(0.95, description="Decay rate for the exponentially weighted average of grads.")
  muon_weight_decay: float = Field(
      0, description="Strength of the weight decay regularization. This is multiplied with the learning rate."
  )
  muon_consistent_rms: None | float = Field(
      None, description="If None, apply width scaling to updates. If float, apply consistent rms scaling (recommend 0.2)."
  )


class PositionalEmbedding(BaseModel):
  """General configuration for positional embeddings."""

  use_iota_embed: bool = Field(
      False,
      description="Use iota operator in Embed, an efficient way to represent positions.",
  )
  use_untrainable_positional_embedding: bool = Field(
      False, description="Use untrainable sinusoidal positional embeddings."
  )
  trainable_position_size: int = Field(
      -1,
      description="Enables GPT-3 style trainable positional embeddings if positive.",
  )
  nope_layer_interval: int = Field(-1, description="If positive, every N-th layer will NOT use RoPE (Llama4).")


class Rope(BaseModel):
  """Configuration for Rotary Positional Embedding (RoPE)."""

  rope_type: RopeType = Field(RopeType.DEFAULT, description="The type of RoPE to use.")
  rope_use_scale: bool = Field(True, description="Apply RoPE scaling for Llama3.1 style.")
  rope_min_timescale: int = Field(1, description="The minimum timescale for RoPE.")
  rope_max_timescale: int = Field(10_000, description="The maximum timescale for global attention RoPE.")
  rope_linear_scaling_factor: float = Field(1.0, description="Linear scaling factor for 'default' RoPE implementation.")
  local_rope_max_timescale: int = Field(-1, description="If positive, used for local window attention RoPE.")


class YarnRope(BaseModel):
  """Configuration specific to YaRN (Yet another RoPE) scaling."""

  max_position_embeddings: int = Field(163840, description="The maximum position embeddings for YaRN scaling.")
  original_max_position_embeddings: int = Field(4096, description="The original max position embeddings before scaling.")
  rope_factor: int = Field(40, description="The scaling factor for YaRN.")
  beta_fast: int = Field(32, description="The 'beta_fast' parameter for YaRN.")
  beta_slow: int = Field(1, description="The 'beta_slow' parameter for YaRN.")
  mscale: float = Field(1.0, description="The 'mscale' parameter for YaRN.")
  rope_interleave: bool = Field(True, description="Whether RoPE sin/cos are interleaved vs concatenated.")
  rope_truncate: bool = Field(True, description="Whether to floor/ceil the correction range for YaRN.")
  rope_attention_scaling: bool = Field(
      False,
      description="Scale the rotary embedding output. Used by some models like gpt-oss.",
  )


class InferenceGeneral(BaseModel):
  """General configuration for inference."""

  max_target_length: int = Field(2048, description="Maximum sequence length for the model.")
  max_prefill_predict_length: int = Field(64, description="Maximum length for the prefill stage in decoding.")
  prompt: str = Field("I love to", description="The default prompt for sampling.")
  load_from_prefill_dir: bool = Field(False, description="Reads prefill cache from directory instead of computing it.")
  prefill_cache_dir: PathStr = Field("", description="Directory for the prefill cache.")
  autoregressive_decode_assert: str = Field(
      "",
      description="Value to assert against during autoregressive decoding, for testing.",
  )
  model_call_mode: str = Field("", description="Mode for model call, e.g., 'inference'.")
  use_chunked_prefill: bool = Field(False, description="Use chunked prefilling for long sequences.")
  prefill_chunk_size: int = Field(256, description="The chunk size for chunked prefilling.")
  enable_model_warmup: bool = Field(False, description="Run a warmup cycle before starting the server.")
  enable_llm_inference_pool: bool = Field(False, description="Launch inference server for llm_inference_gateway.")
  multi_sampling: bool = Field(False, description="Enable multiple sampling configurations.")
  return_log_prob: bool = Field(False, description="Return log probabilities during inference.")


class Decoding(BaseModel):
  """Configuration for decoding and sampling strategies."""

  decode_sampling_strategy: SamplingStrategy = Field(SamplingStrategy.GREEDY, description="The strategy for decoding.")
  decode_sampling_nucleus_p: int | float = Field(-1.0, description="Nucleus (top-p) sampling probability. -1 to disable.")
  decode_sampling_top_k: int = Field(0, description="Top-k sampling value. 0 to disable.")
  decode_sampling_temperature: float = Field(1.0, description="Sampling temperature.")


class InferenceLayout(BaseModel):
  """Configuration for KV cache and compute layouts during inference."""

  stack_prefill_result_cache: bool = Field(False, description="Stack prefill cache across layers to reduce latency.")
  prefill_cache_axis_order: str = Field("1,2,0,3", description="Axis order for the prefill KV cache.")
  ar_cache_axis_order: str = Field("1,2,0,3", description="Axis order for the autoregressive KV cache.")
  compute_axis_order: str = Field("0,1,2,3", description="Axis order for compute operations.")
  reshape_q: bool = Field(False, description="Reshape Q tensor in attention.")


class InferenceServer(BaseModel):
  """Configuration for running as an inference server."""

  inference_server: str = Field("MaxtextInterleavedServer", description="Inference server to start.")
  prefill_slice: str = Field("v5e-16", description="Slice to use for prefill in disaggregation mode.")
  generate_slice: str = Field("v5e-16", description="Slice to use for generatation in disaggregation mode.")


class InferenceBenchmark(BaseModel):
  """Configuration for running inference microbenchmarks."""

  inference_microbenchmark_prefill_lengths: str = Field(
      "64,128,256,512,1024", description="Prefill lengths to benchmark."
  )
  inference_microbenchmark_stages: str = Field("prefill,generate", description="Stages to benchmark.")
  inference_microbenchmark_loop_iters: int = Field(10, description="Number of iterations for the benchmark loop.")
  inference_microbenchmark_log_file_path: PathStr = Field("", description="Path to log benchmark results.")
  inference_microbenchmark_num_samples: list[int] = Field([1, 2, 3, 4, 5], description="Number of samples to benchmark.")
  inference_metadata_file: PathStr = Field("", description="Path to a JSON file with inference metadata.")
  inference_benchmark_test: bool = Field(False, description="Flag to indicate a benchmark test run.")


class PrefixCaching(BaseModel):
  """Configuration for Prefix Caching in JetStream."""

  enable_prefix_caching: bool = Field(False, description="Enable prefix caching.")
  prefix_caching_hbm_byte: int = Field(10_000_000_000, description="HBM memory allocation for prefix caching in bytes.")
  prefix_caching_dram_byte: int = Field(
      100_000_000_000,
      description="DRAM memory allocation for prefix caching in bytes.",
  )


class AOT(BaseModel):
  """Ahead of Time (AOT) Compilation settings."""

  compiled_trainstep_file: PathStr = Field("", description="Name of saved serialized compiled train_step.")
  compile_topology: str = Field("", description="Target hardware version, e.g. 'v5e-256'.")
  compile_topology_num_slices: int = Field(-1, description="Number of target slices.")


class DevelopmentAndDebugging(BaseModel):
  """General settings for development and debugging."""

  constant_bound_config: list = Field([], description="Legacy configuration for constant bounds.")
  jax_cache_dir: PathStr | None = Field(
      os.path.join(os.path.expanduser("~"), "jax_cache"),
      description="Directory for JAX compilation cache.",
  )
  jax_distributed_initialization_timeout: int = Field(300, description="Timeout for jax.distributed.initialize.")
  jax_debug_log_modules: str = Field("", description="Set to 'jax' for verbose JAX logging.")
  skip_jax_distributed_system: bool = Field(False, description="If True, do not initialize the jax distributed system.")
  enable_single_controller: bool = Field(False, description="Enable single-controller mode (Pathways).")
  subslice_shape: str = Field("", description="Subslice shape in the form of 'x,y,z' for Pathways.")
  max_checkify: bool = Field(
      False,
      description="If True, perform extra checks using jax.checkify, affecting performance.",
  )

  @classmethod
  def _clean_empty_string_for_list(cls, v: Any) -> Any:
    """Coerces an empty string from YAML into an empty list before validation."""
    if v == "":
      return []
    elif isinstance(v, str):
      return list(map(float, v.split(",")))
    return v

  # Manually apply the field_validator decorator outside of the class definition to avoid pytype issues
  _validate_config = field_validator("constant_bound_config", mode="before")(_clean_empty_string_for_list)


class Profiling(BaseModel):
  """Configuration for performance profiling."""

  profiler: ProfilerType = Field(ProfilerType.NONE, description="Profiler to use ('xplane', 'nsys').")
  upload_all_profiler_results: bool = Field(False, description="Upload profiler results from all hosts.")
  skip_first_n_steps_for_profiler: int = Field(1, description="Number of initial steps to skip for profiling.")
  profiler_steps: int = Field(5, description="Number of steps to profile.")
  profile_cleanly: bool = Field(True, description="Add block_until_ready to align profile for each step.")
  profile_periodically_period: int = Field(-1, description="If positive, profile every N steps.")
  hide_profiler_step_metric: bool = Field(False, description="Whether to enable profiler step metric.")
  enable_jax_profiler: bool = Field(False, description="Enable the JAX live profiler.")
  jax_profiler_port: int = Field(9999, description="Port for the JAX profiler.")
  xprof_tpu_power_trace_level: XProfTPUPowerTraceMode = Field(
      XProfTPUPowerTraceMode.POWER_TRACE_NONE,
      description=(
          "TPU power trace level. The value should be 0 (POWER_TRACE_NONE), 1"
          " (POWER_TRACE_NORMAL), or 2 (POWER_TRACE_SPI)"
      ),
  )
  xprof_e2e_enable_fw_throttle_event: bool = Field(False, description="Enable FW throttle event.")
  xprof_e2e_enable_fw_power_level_event: bool = Field(False, description="Enable FW power level event.")
  xprof_e2e_enable_fw_thermal_event: bool = Field(False, description="Enable FW thermal event.")


class HloDump(BaseModel):
  """Configuration for dumping HLO modules for debugging."""

  dump_hlo: bool = Field(False, description="Enable HLO dumping.")
  dump_step: int = Field(-1, description="Dump HLO at a specific step. -1 disables step-specific dump.")
  dump_hlo_local_dir: PathStr = Field(
      os.path.join(gettempdir(), "xla_dump", ""),
      description="Local directory to dump HLO.",
  )
  dump_hlo_delete_local_after: bool = Field(True, description="Delete local HLO dump after uploading to GCS.")
  dump_hlo_gcs_dir: PathStr = Field("", description="GCS directory to upload HLO dumps.")
  dump_hlo_module_name: str = Field("jit_train_step", description="Filter modules to dump by this name.")
  dump_hlo_local_module_name: str = Field("jit_train_step", description="Filter modules to save locally by this name.")
  dump_hlo_xla_flags: str = Field("", description="Pass custom XLA flags for HLO dumping.")
  dump_hlo_upload_all: bool = Field(False, description="Upload HLO from all hosts.")
  dump_jaxpr: bool = Field(False, description="Enable jaxpr dumping.")
  dump_jaxpr_local_dir: PathStr = Field(
      os.path.join(gettempdir(), "jaxpr_dump", ""),
      description="Local directory to dump jaxpr.",
  )
  dump_jaxpr_delete_local_after: bool = Field(True, description="Delete local jaxpr dump after uploading to GCS.")
  dump_jaxpr_gcs_dir: PathStr = Field("", description="GCS directory to upload jaxpr dumps.")


class StackTrace(BaseModel):
  """Configuration for collecting and logging stack traces."""

  collect_stack_trace: bool = Field(False, description="Enable periodic stack trace collection.")
  stack_trace_to_cloud: bool = Field(False, description="Upload stack traces to cloud logging instead of console.")
  stack_trace_interval_seconds: int = Field(600, description="Frequency of stack trace collection in seconds.")


class Metrics(BaseModel):
  """General configuration for metrics and monitoring."""

  metrics_file: None | PathStr = Field(None, description="Local file to store scalar metrics for testing.")
  gcs_metrics: bool = Field(False, description="If True, save metrics to GCS.")
  save_config_to_gcs: bool = Field(False, description="If True, save config to GCS.")
  record_internal_nn_metrics: int = Field(0, description="Record internal neural network metrics.")
  prometheus_port: int = Field(0, description="Port for Prometheus metrics server. 0 disables it.")
  enable_checkpoint_cloud_logger: bool = Field(False, description="Enables structured logging for checkpointing.")
  enable_tunix_perf_metrics: bool = Field(
      False,
      description="Whether to enable Tunix-managed metrics measurement. The metrics will be uploaded to tensorboard.",
  )


class ManagedMLDiagnostics(BaseModel):
  """Configuration for managed mldiagnostics."""

  managed_mldiagnostics: bool = Field(False, description="Enable managed mldiagnostics.")
  managed_mldiagnostics_run_group: str = Field("", description="Name used to group multiple runs.")


class Goodput(BaseModel):
  """Configuration for goodput monitoring."""

  enable_goodput_recording: bool = Field(False, description="Enable goodput recording.")
  monitor_goodput: bool = Field(False, description="Monitor goodput.")
  goodput_upload_interval_seconds: int = Field(30, description="Interval to upload goodput metrics.")
  enable_pathways_goodput: bool = Field(False, description="Enable goodput monitoring for Pathways.")
  monitor_step_time_deviation: bool = Field(True, description="Monitor step time deviation.")
  step_deviation_interval_seconds: int = Field(30, description="Interval to check step time deviation.")
  enable_gcp_goodput_metrics: bool = Field(True, description="Enable GCP goodput metrics.")
  enable_gcp_step_deviation_metrics: bool = Field(True, description="Enable GCP step deviation metrics.")


class GcpMonitoring(BaseModel):
  """Configuration for GCP-specific workload monitoring."""

  report_heartbeat_metric_for_gcp_monitoring: bool = Field(
      False, description="Report heartbeat metric for GCP monitoring."
  )
  heartbeat_reporting_interval_in_seconds: int = Field(5, description="Interval for heartbeat metric.")
  report_performance_metric_for_gcp_monitoring: bool = Field(
      False, description="Report performance metric for GCP monitoring."
  )


class Tensorboard(BaseModel):
  """Configuration for Tensorboard logging."""

  enable_tensorboard: bool = Field(True, description="Enable Tensorboard logging.")
  use_vertex_tensorboard: bool = Field(False, description="Set to True for GCE, False if running via XPK.")
  vertex_tensorboard_project: Optional[str] = Field("", description="GCP project for Vertex AI Tensorboard.")
  vertex_tensorboard_region: Optional[str] = Field("", description="Region for Vertex AI Tensorboard.")


class MultimodalGeneral(BaseModel):
  """General configuration for Multimodal models."""

  use_multimodal: bool = Field(False, description="Enable multimodal capabilities.")
  freeze_vision_encoder_params: bool = Field(True, description="Freeze the parameters of the vision encoder.")
  freeze_audio_encoder_params: bool = Field(True, description="Freeze the parameters of the audio encoder.")
  use_audio: bool = Field(False, description="Enable audio encoder for multimodal models.")
  image_size_for_vit: int = Field(896, description="Input image size for the Vision Transformer.")
  image_path: PathStr = Field("", description="Path to an image for decoding.")
  image_placeholder: str = Field("<|image|>", description="Placeholder string for images in text prompts.")
  posemb_type_for_vit: str = Field("learn", description="Positional embedding type for the vision encoder.")
  max_num_images_per_example: int = Field(
      -1,
      description="Maximum number of images per example for training with image lists. -1 means no limit.",
  )
  video_path: PathStr = Field("", description="Path to a video for decoding.")
  audio_path: PathStr = Field("", description="Path to an audio file for decoding.")
  video_placeholder: str = Field("<|video|>", description="Placeholder string for video in text prompts.")
  audio_placeholder: str = Field("<|audio|>", description="Placeholder string for audio in text prompts.")
  use_audio_in_video: bool = Field(False, description="Extract and use audio from video files.")
  use_mrope: bool = Field(False, description="Enable Multi-dimensional RoPE for Qwen3-Omni models.")
  mrope_section: list[int] = Field([24, 20, 20], description="Dimensions for temporal, height, width in MRoPE.")
  position_id_per_seconds: int = Field(25, description="Temporal granularity for MRoPE (tokens per second).")


class VisionTower(BaseModel):
  """Configuration for the Vision Tower (Encoder) in a multimodal model."""

  hidden_size_for_vit: int = Field(1408, description="Hidden size for the Vision Transformer.")
  intermediate_size_for_vit: int = Field(5632, description="Intermediate size for the Vision Transformer's MLP.")
  num_attention_heads_for_vit: int = Field(16, description="Number of attention heads in the Vision Transformer.")
  num_channels_for_vit: int = Field(
      3,
      description="Number of input channels for the Vision Transformer (e.g., 3 for RGB).",
  )
  tile_size_for_vit: int = Field(336, description="Tile size for the Vision Transformer.")
  patch_size_for_vit: int = Field(14, description="Patch size for the Vision Transformer.")
  conv_stride_for_vit: int = Field(
      14,
      description="Convolutional stride for the Vision Transformer's patch embedding.",
  )
  num_hidden_layers_for_vit: int = Field(34, description="Number of hidden layers in the Vision Transformer.")
  rope_theta_for_vit: int = Field(10000, description="RoPE theta value for the Vision Transformer.")
  vision_output_dim_for_vit: int = Field(4096, description="Final output dimension of the vision-to-language projection.")
  spatial_merge_size_for_vit: int = Field(2, description="Spatial merge factor for vision patches.")
  out_hidden_size_for_vit: int = Field(512, description="Output dimension of ViT.")
  temporal_patch_size_for_vit: int = Field(2, description="Temporal patch size for video inputs.")
  num_position_embeddings_for_vit: int = Field(1024, description="Number of position embeddings for ViT.")
  deepstack_visual_indexes_for_vit: list[int] = Field([], description="Layer indices to extract deep visual features.")


class VisionProjector(BaseModel):
  """Configuration for the Vision Projector in a multimodal model."""

  projector_input_dim_for_vit: int = Field(4096, description="Input dimension for the vision projector.")
  projector_output_dim_for_vit: int = Field(4096, description="Output dimension for the vision projector.")
  pixel_shuffle_ratio_for_vit: float = Field(0.5, description="Pixel shuffle ratio for the Vision Transformer.")
  projector_dropout_for_vit: float = Field(0.0, description="Dropout rate for the vision projector.")


class AudioEncoder(BaseModel):
  """Configuration for the Audio Encoder in a multimodal model."""

  d_model_for_audio: int = Field(256, description="Model dimension for the audio encoder.")
  encoder_attention_heads_for_audio: int = Field(4, description="Number of attention heads in the audio encoder.")
  encoder_ffn_dim_for_audio: int = Field(512, description="Feed-forward network dimension for the audio encoder.")
  encoder_layers_for_audio: int = Field(2, description="Number of encoder layers for audio.")
  attention_dropout_for_audio: float = Field(0.0, description="Attention dropout rate for audio encoder.")
  activation_dropout_for_audio: float = Field(0.0, description="Activation dropout rate for audio encoder.")
  activation_function_for_audio: str = Field("gelu", description="Activation function for audio encoder.")
  num_mel_bins_for_audio: int = Field(128, description="Number of mel-frequency bins for audio input.")
  max_source_positions_for_audio: int = Field(1500, description="Maximum source positions for audio encoder.")
  scale_embedding_for_audio: bool = Field(True, description="Whether to scale embeddings in audio encoder.")
  n_window_for_audio: int = Field(50, description="Window size for audio processing.")
  n_window_infer_for_audio: int = Field(800, description="Window size for audio inference.")
  conv_chunksize_for_audio: int = Field(500, description="Chunk size for convolutional layers in audio encoder.")
  downsample_hidden_size_for_audio: int = Field(256, description="Hidden size for downsampling in audio encoder.")
  output_dim_for_audio: int = Field(512, description="Output dimension for audio encoder.")
  num_conv_layers_for_audio: int = Field(3, description="Number of convolutional layers in audio encoder.")
  max_timescale_for_audio: float = Field(10000.0, description="Maximum timescale for audio positional encoding.")
  max_sample_len_for_audio: int = Field(10000, description="Maximum sample length for audio input.")


class Debug(BaseModel):
  """Configuration for debugging options."""

  rl: bool = Field(False, description="RL-specific debugging")


class RLHardware(BaseModel):
  """Hardware settings specific to RL training."""

  trainer_devices_fraction: float = Field(0.5, description="Fraction of devices to use for the trainer.")
  sampler_devices_fraction: float = Field(0.5, description="Fraction of devices to use for the sampler.")
  chips_per_vm: int = Field(4, description="Number of accelerator chips per VM.")
  use_pathways: bool = Field(True, description="Whether to use Pathways for multihost orchestration.")
  num_trainer_slices: int = Field(-1, description="Number of slices for the trainer.")
  num_samplers_slices: int = Field(-1, description="Number of slices for the samplers.")
  rollout_data_parallelism: int = Field(
      -1,
      description="Total model replicas for rollout. It should only be specified when you would like to use more "
      "than one model replica in rollout.",
  )
  rollout_tensor_parallelism: int = Field(
      -1, description="Tensor parallelism per replica for rollout. If not specified, it will be auto-determined."
  )


class VLLM(BaseModel):
  """vLLM-specific configuration for rollouts."""

  kv_cache_buffer: int = Field(256, description="Buffer for KV cache.")
  hbm_utilization_vllm: float = Field(0.72, description="Target HBM utilization for vLLM.")
  swap_space_vllm_gb: int = Field(2, description="Swap space in GB for vLLM.")
  enable_dp_attention: bool = Field(False, description="Enable the attn_dp mesh axis in vLLM.")
  max_num_batched_tokens: Optional[int] = Field(None, description="Max number of batched tokens in vLLM.")
  max_num_seqs: Optional[int] = Field(None, description="Max number of sequences in vLLM.")
  vllm_additional_config: dict[str, Any] = Field(default_factory=dict, description="Additional vLLM config options.")
  vllm_hf_overrides: dict[str, Any] = Field(
      default_factory=dict, description="Overrides for HuggingFace model config for MaxText model."
  )
  vllm_hf_config_path: str = Field("", description="Path to HuggingFace model config for MaxText model.")


class RL(BaseModel):
  """Configuration for RL algorithms like Group Relative Policy Optimization (GRPO) among others."""

  num_generations: int = Field(2, description="Number of responses to generate per prompt (G in GRPO paper).")
  num_iterations: int = Field(1, description="Number of iterations per batch (μ in GRPO paper).")
  grpo_beta: float = Field(0.08, description="Coefficient for the KL divergence penalty (β).")
  grpo_epsilon: float = Field(0.2, description="Epsilon value for clipping in the GRPO loss.")
  loss_algo: Literal["grpo", "gspo-token"] = Field("grpo", description="Loss algorithm, i.e., 'grpo' or 'gspo-token'.")


class RLDataset(BaseModel):
  """Dataset settings for RL training."""

  batch_size: int = Field(1, description="Global batch size for the dataset loader in RL.")
  num_batches: int = Field(4, description="Number of batches for RL training.")
  num_test_batches: int = Field(5, description="Number of batches for RL evaluation.")
  train_fraction: float = Field(1.0, description="Fraction of the dataset to be used for training.")
  micro_batch_size: int = Field(-1, description="Micro batch size for rollout and training.")


class RLEvaluation(BaseModel):
  """Settings for RL evaluation."""

  eval_sampling_strategy: str = Field("greedy", description="Sampling strategy for evaluation.")
  generation_configs: dict[str, Any] = Field(
      default_factory=dict,
      description="Configurations for different generation strategies.",
  )
  num_eval_passes: int = Field(1, description="Number of generation passes during evaluation.")
  eval_corr_lst: bool = Field(
      False,
      description="If True, only include correct responses in the list during evaluation.",
  )
  eval_make_lst: bool = Field(
      False,
      description="If True, return a list of (question, answer, responses) during evaluation.",
  )


class Reward(BaseModel):
  """Configuration for the reward/penalty model in RL."""

  reward_exact_format_match: float = Field(3.0, description="Reward for an exact format match.")
  reward_white_space_format_match: float = Field(1.5, description="Reward for a format match ignoring whitespace.")
  reward_partial_format_match: float = Field(0.5, description="Reward for a partial format match.")
  reward_ratio_guess_to_answer_high: float = Field(
      0.5, description="Reward when guess/answer ratio is within a high threshold."
  )
  reward_ratio_guess_to_answer_low: float = Field(
      0.25, description="Reward when guess/answer ratio is within a low threshold."
  )
  penalty_incorrect_format: float = Field(-0.5, description="Penalty for an incorrect format.")
  penalty_incorrect_answer: float = Field(-1.0, description="Penalty for an incorrect answer.")


class SpecialTokens(BaseModel):
  """Special tokens used for formatting prompts and responses in RL."""

  reasoning_start_token: str = Field("<reasoning>", description="Token to mark the beginning of a reasoning section.")
  reasoning_end_token: str = Field("</reasoning>", description="Token to mark the end of a reasoning section.")
  solution_start_token: str = Field("<answer>", description="Token to mark the beginning of a solution section.")
  solution_end_token: str = Field("</answer>", description="Token to mark the end of a solution section.")


class Engram(BaseModel):
  """Configuration for DeepSeek Engram (https://www.arxiv.org/pdf/2601.07372)."""

  engram_layers: list[int] = Field(
      default_factory=list,
      description="Indices of transformer layers where Engram are integrated.",
  )
  engram_num_heads: int = Field(8, description="Number of heads dedicated to the Engram.")
  engram_head_dim: int = Field(1280, description="Head dimension for heads.")
  engram_vocab_bases: list[int] = Field(
      default_factory=list, description="List of minimum head vocab sizes for each n-gram order."
  )
  engram_max_ngram_size: int = Field(3, description="The max 'n' in N-gram.")
  engram_kernel_size: int = Field(4, description="Temporal window size for Engram convolution.")
  engram_seed: int = Field(0, description="The seed for Engram hash mapping.")


class DerivedValues(BaseModel):
  """Holds all fields that are derived from other config values for perfect legacy compatibility."""

  emb_dim: None | int = Field(
      None,
      description="Effective embedding dimension, scaled by `global_parameter_scale`.",
  )
  mlp_dim: None | int = Field(None, description="Effective MLP dimension, scaled by `global_parameter_scale`.")
  moe_mlp_dim: None | int = Field(
      None,
      description="Effective MLP dimension for MoE layers, scaled by `global_parameter_scale`.",
  )
  num_decoder_layers: None | int = Field(
      None,
      description="Effective number of decoder layers, scaled by `global_parameter_scale`.",
  )
  num_kv_heads: None | int = Field(
      None,
      description="Effective number of key/value heads, scaled by `global_parameter_scale`.",
  )
  num_query_heads: None | int = Field(
      None,
      description="Effective number of query heads, scaled by `global_parameter_scale`.",
  )

  num_diloco_replicas: None | int = Field(
      None,
      description="The number of diloco replicas, derived from ICI and DCN values.",
  )

  ici_parallelism: None | list[int] = Field(
      None,
      description="Aggregated list of all ICI parallelism values for legacy compatibility.",
  )
  dcn_parallelism: None | list[int] = Field(
      None,
      description="Aggregated list of all DCN parallelism values for legacy compatibility.",
  )

  using_pipeline_parallelism: None | bool = Field(
      None,
      description="Boolean flag indicating if pipeline parallelism is active across ICI or DCN.",
  )

  context_parallel_size: None | int = Field(
      None,
      description="The total size of context parallelism, derived from ICI and DCN values.",
  )

  num_target_devices: None | int = Field(
      None,
      description="The number of devices computed from topology in train_compile or jax.devices() in train",
  )

  global_batch_size_to_train_on: None | int = Field(
      None,
      description="The total batch size for training across all devices. Derived from `per_device_batch_size` and data"
      "parallelism.",
  )
  global_batch_size_to_eval_on: None | int = Field(
      None,
      description="The total batch size for evaluation across all devices. Derived from `eval_per_device_batch_size` and"
      " data parallelism.",
  )
  global_batch_size_to_load: None | int = Field(
      None,
      description="The global batch size for the training dataloader, potentially scaled by `expansion_factor_real_data`.",
  )
  global_batch_size_to_load_eval: None | int = Field(
      None,
      description="The global batch size for the evaluation dataloader, potentially scaled by `expansion_factor_real_data`.",
  )
  micro_batch_size_to_train_on: None | int = Field(
      None,
      description="The size of each micro-batch for training, used in pipeline parallelism. Derived from "
      "`global_batch_size_to_train_on`.",
  )
  micro_batch_size_to_eval_on: None | int = Field(
      None,
      description="The size of each micro-batch for evaluation, used in pipeline parallelism. Derived from "
      "`global_batch_size_to_eval_on`.",
  )

  checkpoint_dir: None | str = Field(
      None,
      description="The full path to the checkpoint directory, derived from `run_name`.",
  )
  metrics_dir: None | str = Field(
      None,
      description="The full path to the metrics directory, derived from `run_name`.",
  )
  tensorboard_dir: None | str = Field(
      None,
      description="The full path to the tensorboard directory, derived from `run_name`.",
  )
  managed_mldiagnostics_dir: None | str = Field(
      None,
      description="The full path to the managed mldiagnostics directory, derived from `run_name`.",
  )

  rampup_end_step: None | int = Field(None, description="The step at which the batch size ramp-up phase concludes.")
  tensors_on_device: None | list[str] = Field(
      None, description="List of tensors to keep on device memory for custom remat."
  )
  tensors_to_offload: None | list[str] = Field(
      None, description="List of tensors to offload to host memory for custom remat."
  )
  global_batch_size_to_load_start: None | int = Field(None, description="Starting global batch size for rampup.")
  global_batch_size_to_load_increment: None | int = Field(
      None, description="Increment for global batch size during rampup."
  )
  rampup_samples_per_increment_to_load: None | float = Field(None, description="Samples per increment for rampup.")


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def get_individual_scales(scale: int) -> tuple[int, int, int, int]:
  """Choose appropriate scales for individual dimensions based on global scale."""
  if scale == 0:
    raise ValueError("Global parameter scale cannot be zero.")
  log_2_scale = math.floor(math.log2(scale))
  if 2**log_2_scale != scale:
    raise ValueError(
        "Global parameter scale should be a power of two. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads, "
        "base_mlp_dim, base_num_decoder_layers and/or head_dim."
    )
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


# ----------------------------------------------------------------------------
# Main Config Class
# ----------------------------------------------------------------------------


class MaxTextConfig(
    # Run and Checkpointing
    RunInfo,
    Checkpointing,
    OrbaxStorage,
    EmergencyCheckpointing,
    # Data Types and Quantization
    DataTypes,
    Quantization,
    # Core Model Architecture
    ModelArchitecture,
    Engram,
    MTP,
    Logits,
    # Attention Mechanisms
    Attention,
    MlaAttention,
    MoBa,
    AttentionIndexer,
    Llama4Attention,
    SplashAttention,
    PagedAttention,
    # Mixture of Experts
    MoEGeneral,
    MoEKernels,
    DeepSeekMoE,
    Qwen3Next,
    # Parallelism and Layout
    HardwareAndMesh,
    LayoutAndSharding,
    DcnParallelism,
    IciParallelism,
    PipelineParallelism,
    # Training, Optimization, and Fine-Tuning
    RematAndOffload,
    TrainingLoop,
    ManifoldConstrainedHyperConnections,
    DilocoParams,
    Optimizer,
    AdamW,
    Muon,
    FineTuning,
    Distillation,
    # Reinforcement Learning
    RLHardware,
    VLLM,
    RL,
    RLDataset,
    RLEvaluation,
    Reward,
    SpecialTokens,
    # Positional Embeddings
    PositionalEmbedding,
    Rope,
    YarnRope,
    # Dataset Loading and Tokenization
    DatasetGeneral,
    TfdsDataset,
    HfDataset,
    GrainDataset,
    Tokenizer,
    # Inference
    InferenceGeneral,
    Decoding,
    InferenceLayout,
    InferenceServer,
    InferenceBenchmark,
    PrefixCaching,
    # Development and Debugging
    AOT,
    DevelopmentAndDebugging,
    Profiling,
    HloDump,
    StackTrace,
    # Metrics and Monitoring
    Metrics,
    Goodput,
    GcpMonitoring,
    Tensorboard,
    ManagedMLDiagnostics,
    # Multimodal
    MultimodalGeneral,
    VisionTower,
    VisionProjector,
    AudioEncoder,
    # Derived
    DerivedValues,
):
  """
  The main configuration object for MaxText.

  This class aggregates all configuration options from modular `BaseModel` classes
  into a single, validated object. It is populated by the `initialize` function.
  Every field is explicitly defined to prevent misconfigurations (`extra='forbid'`).
  """

  debug: Debug = Field(default_factory=Debug, description="Configuration for debugging options.")
  rl: RL = Field(
      default_factory=RL, description="Configuration for RL algorithms like Group Relative Policy Optimization (GRPO)."
  )
  model_config = ConfigDict(extra="forbid", protected_namespaces=())

  @model_validator(mode="before")
  @classmethod
  def load_model_specific_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
    """This method is a no-op because `pyconfig` handles model-specific config loading."""
    return values

  @model_validator(mode="after")
  def set_derived_and_validate_values(self) -> "MaxTextConfig":
    """
    Computes all derived values and runs all cross-field validations after initial parsing.
    This logic is ported from the legacy pyconfig_deprecated.py system and adapted for Pydantic.
    """
    # A. SET RUN NAME AND PATHS
    # If run_name is not set, generate one from the JOBSET_NAME environment variable (if available)
    # or create one from the model name and a timestamp.
    if not self.run_name:
      if os.environ.get("JOBSET_NAME"):
        self.run_name = os.environ.get("JOBSET_NAME")
      else:
        self.run_name = f'{self.model_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}'

    # Construct full paths for output directories based on the base directory and run name.
    if self.run_name and self.base_output_directory:
      output_dir = os.path.join(self.base_output_directory, self.run_name)
      self.checkpoint_dir = os.path.join(output_dir, "checkpoints", "")
      self.metrics_dir = os.path.join(output_dir, "metrics", "")
      self.tensorboard_dir = os.path.join(output_dir, "tensorboard", "")
      # To work around SDK bug b/454725283, remove the trailing back slash from the managed_mldiagnostics_dir.
      self.managed_mldiagnostics_dir = os.path.join(output_dir, "managed-mldiagnostics")
    else:
      self.checkpoint_dir, self.metrics_dir, self.tensorboard_dir = (
          None,
          None,
          None,
      )

    # B. RESOLVE TOKENIZER PATH
    # If the tokenizer path is a relative name without a directory, resolve it against the assets root.
    # This maintains backward compatibility for configs that just specify e.g., "tokenizer.llama2".
    tokenizer_path = getattr(self, "tokenizer_path", "")
    if tokenizer_path and not os.path.exists(tokenizer_path) and not tokenizer_path.startswith("gs://"):
      tokenizer_path = next(
          filter(
              os.path.exists,
              (
                  os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", os.path.basename(tokenizer_path)),
                  os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", tokenizer_path),
              ),
          ),
          tokenizer_path,
      )
      self.tokenizer_path = tokenizer_path

    # C. SET PRIMARY DEPENDENCIES & DEFAULTS
    # If learning_rate_schedule_steps is -1, it defaults to the total number of training steps.
    if self.learning_rate_schedule_steps == -1:
      self.learning_rate_schedule_steps = self.steps
    # If steps is -1, it defaults to the length of the learning rate schedule.
    if self.steps == -1:
      self.steps = self.learning_rate_schedule_steps

    # Validate deepstack + scan_layers incompatibility
    if self.deepstack_visual_indexes_for_vit and self.scan_layers:
      raise ValueError(
          "Deepstack visual embedding injection requires scan_layers=False. "
          "Set scan_layers=False in your config to use deepstack features."
      )

    # Validate WSD learning rate schedule fractions
    if self.lr_schedule_type == LearningRateScheduleType.WSD:
      total_fraction = self.warmup_steps_fraction + self.wsd_decay_steps_fraction
      if total_fraction > 1.0:
        raise ValueError(
            f"Invalid WSD schedule: warmup_steps_fraction ({self.warmup_steps_fraction}) + "
            f"wsd_decay_steps_fraction ({self.wsd_decay_steps_fraction}) must not exceed 1.0. "
            f"Current sum: {total_fraction}"
        )

    # If eval_per_device_batch_size is not set, it defaults to the training per_device_batch_size.
    if getattr(self, "eval_per_device_batch_size", 0.0) == 0.0:
      self.eval_per_device_batch_size = self.per_device_batch_size
    # The mu_dtype for the AdamW optimizer defaults to the weight_dtype if not specified.
    if not self.mu_dtype:
      self.mu_dtype = self.weight_dtype
    # A soft-cap value of 0.0 is treated as disabling the cap.
    if self.attn_logits_soft_cap == 0.0:
      self.attn_logits_soft_cap = None
    if self.final_logits_soft_cap == 0.0:
      self.final_logits_soft_cap = None

    # This must be invoked before initializing the backend
    # pylint: disable=access-member-before-definition
    def validate_and_set_hlo_dump_defaults():
      if os.environ.get("XLA_FLAGS") and self.dump_hlo_xla_flags:
        raise ValueError("You must set either XLA_FLAGS or dump_hlo_xla_flags to dump HLO, but not both.")
      if not os.environ.get("XLA_FLAGS") and not self.dump_hlo_xla_flags:
        self.dump_hlo_xla_flags = f"--xla_dump_to={self.dump_hlo_local_dir} --xla_dump_large_constants"
        if self.dump_hlo_local_module_name:
          self.dump_hlo_xla_flags = (
              f"{self.dump_hlo_xla_flags} --xla_dump_hlo_module_re={self.dump_hlo_local_module_name}"
          )
      if not self.dump_hlo_gcs_dir:
        self.dump_hlo_gcs_dir = os.path.join(self.base_output_directory, self.run_name, "xla_dump")
      else:
        self.dump_hlo_gcs_dir = gcs_utils.add_trailing_slash(self.dump_hlo_gcs_dir)
      if not self.dump_jaxpr_gcs_dir:
        self.dump_jaxpr_gcs_dir = os.path.join(self.base_output_directory, self.run_name, "jaxpr_dump")
      else:
        self.dump_jaxpr_gcs_dir = gcs_utils.add_trailing_slash(self.dump_jaxpr_gcs_dir)
      if not os.environ.get("XLA_FLAGS"):
        os.environ["XLA_FLAGS"] = self.dump_hlo_xla_flags

    # pylint: enable=access-member-before-definition

    # Validate and initiate hlo dump related configs
    validate_and_set_hlo_dump_defaults()

    # D. CALCULATE MODEL DIMENSIONS from global_parameter_scale
    # This allows scaling the model size up or down easily with a single power-of-two factor.
    emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(self.global_parameter_scale)
    self.emb_dim = (2**emb_scale) * self.base_emb_dim
    self.num_query_heads = (2**num_head_scale) * self.base_num_query_heads
    self.num_kv_heads = (2**num_head_scale) * self.base_num_kv_heads
    self.mlp_dim = (2**mlp_dim_scale) * self.base_mlp_dim
    self.moe_mlp_dim = (2**mlp_dim_scale) * self.base_moe_mlp_dim
    self.num_decoder_layers = (2**layer_scale) * self.base_num_decoder_layers

    # E. HARDWARE-DEPENDENT CALCULATIONS
    def get_num_target_devices():
      """Get the number of devices for the target topology, handling AOT compilation and single-controller modes."""
      if self.compile_topology:
        spec = accelerator_to_spec_map.get_system_characteristics(self.compile_topology)
        return int(spec.devices_per_slice * self.compile_topology_num_slices)
      elif self.subslice_shape and self.enable_single_controller:
        shape_tuple = tuple(int(x) for x in self.subslice_shape.split(","))
        return prod(shape_tuple)
      else:
        return len(jax.devices())

    self.num_target_devices = 1  # Default for validation when JAX is not initialized
    try:
      self.num_target_devices = get_num_target_devices()
    except (RuntimeError, IndexError):
      logger.warning("JAX device system not available for config validation. Assuming 1 device.")

    # Automatically determine number of slices if not specified.
    raw_keys_for_num_slices = {
        "num_slices": self.num_slices,
        "hardware": self.hardware,
        "compile_topology_num_slices": self.compile_topology_num_slices,
    }
    self.num_slices = max_utils.get_num_slices(raw_keys_for_num_slices)

    # Default quantization sharding count to number of local devices if not set.
    if self.quantization_local_shard_count == -1:
      try:
        self.quantization_local_shard_count = jax.local_device_count()
      except RuntimeError:
        self.quantization_local_shard_count = 1

    # F. CALCULATE BATCH SIZES
    def calculate_global_batch_sizes(per_device_batch_size, expansion_factor, num_devices, grad_accum_steps):
      """Helper to calculate global and micro batch sizes for training and loading."""
      if per_device_batch_size < 1.0:
        micro_batch_to_load = num_devices * (expansion_factor if expansion_factor > 0 else 1)
      else:
        micro_batch_to_load = int(num_devices * per_device_batch_size * (expansion_factor if expansion_factor > 0 else 1))
      micro_batch_to_train = int(num_devices * per_device_batch_size)
      global_batch_to_load = int(micro_batch_to_load * grad_accum_steps)
      global_batch_to_train = int(micro_batch_to_train * grad_accum_steps)
      return global_batch_to_load, global_batch_to_train, micro_batch_to_train

    # Calculate final training batch sizes.
    (
        self.global_batch_size_to_load,
        self.global_batch_size_to_train_on,
        self.micro_batch_size_to_train_on,
    ) = calculate_global_batch_sizes(
        self.per_device_batch_size,
        self.expansion_factor_real_data,
        self.num_target_devices,
        self.gradient_accumulation_steps,
    )

    # Calculate final evaluation batch sizes.
    (
        self.global_batch_size_to_load_eval,
        self.global_batch_size_to_eval_on,
        self.micro_batch_size_to_eval_on,
    ) = calculate_global_batch_sizes(
        self.eval_per_device_batch_size, self.expansion_factor_real_data, self.num_target_devices, 1
    )

    # Calculate ramp-up batch size parameters if enabled.
    if self.enable_rampup_batch_size:
      (
          self.global_batch_size_to_load_start,
          _,
          _,
      ) = calculate_global_batch_sizes(
          self.per_device_batch_size_start,
          self.expansion_factor_real_data,
          self.num_target_devices,
          self.gradient_accumulation_steps,
      )
      (
          self.global_batch_size_to_load_increment,
          _,
          _,
      ) = calculate_global_batch_sizes(
          self.per_device_batch_size_increment,
          self.expansion_factor_real_data,
          self.num_target_devices,
          self.gradient_accumulation_steps,
      )
      diff_batch_size = self.global_batch_size_to_load - self.global_batch_size_to_load_start
      if self.global_batch_size_to_load_increment > 0:
        num_increments = diff_batch_size // self.global_batch_size_to_load_increment
        if num_increments > 0:
          rampup_samples_per_increment = self.global_rampup_samples / num_increments
          self.rampup_samples_per_increment_to_load = rampup_samples_per_increment
          total_rampup_steps = 0
          current_batch_size = self.global_batch_size_to_load_start
          for _ in range(int(num_increments)):
            steps_for_this_stage = (
                math.ceil(rampup_samples_per_increment / current_batch_size) if current_batch_size > 0 else 0
            )
            total_rampup_steps += steps_for_this_stage
            current_batch_size += self.global_batch_size_to_load_increment
          self.rampup_end_step = total_rampup_steps
        else:
          self.rampup_end_step = 0
      else:
        self.rampup_end_step = 0
    else:
      self.rampup_end_step = 0

    # G. CALCULATE/SET OTHER DERIVED VALUES, E.G. PIPELINE CONFIG
    if self.remat_policy == "custom":
      tensors = [
          "decoder_layer_input",
          "context",
          "mlpwi",
          "mlpwi_0",
          "mlpwi_1",
          "mlpwo",
          "query_proj",
          "key_proj",
          "value_proj",
          "mla_kv",
          "mla_q",
          "qkv_proj",
          "attention_out",
          "out_proj",
      ]
      self.tensors_on_device = [t for t in tensors if getattr(self, t) == "device"]
      self.tensors_to_offload = [t for t in tensors if getattr(self, t) == "offload"]

    cp_size = self.ici_context_parallelism * self.dcn_context_parallelism
    if self.expert_shard_attention_option == "context":
      cp_size *= self.ici_expert_parallelism * self.dcn_expert_parallelism
    self.context_parallel_size = cp_size
    if self.pipeline_parallel_layers == -1:
      if self.decoder_block == DecoderBlockType.DEEPSEEK:
        moe_layers = self.num_decoder_layers - self.first_num_dense_layers
        self.pipeline_parallel_layers = moe_layers
      else:
        self.pipeline_parallel_layers = self.num_decoder_layers

    self.using_pipeline_parallelism = self.ici_pipeline_parallelism > 1 or self.dcn_pipeline_parallelism > 1
    if self.using_pipeline_parallelism:
      num_stages = int(self.ici_pipeline_parallelism * self.dcn_pipeline_parallelism)
      if self.num_pipeline_repeats == -1:
        num_pipeline_repeats, remainder = divmod(
            self.pipeline_parallel_layers,
            num_stages * self.num_layers_per_pipeline_stage,
        )
        assert not remainder, (
            f"The number of layers per stage ({self.num_layers_per_pipeline_stage}) times the number of stages "
            f"({num_stages}) must divide the number of pipeline_parallel_layers which defaults to decoder layers "
            f"({self.pipeline_parallel_layers}) "
        )
        self.num_pipeline_repeats = num_pipeline_repeats

      assert (num_stages * self.num_pipeline_repeats * self.num_layers_per_pipeline_stage) == (
          self.pipeline_parallel_layers
      ), (
          f"The product of pipeline stages ({num_stages}), repeats ({self.num_pipeline_repeats}), and layers "
          f"per stage ({self.num_layers_per_pipeline_stage}) must be equal to pipeline_parallel_layers "
          f"which defaults to decoder layers ({self.pipeline_parallel_layers})"
      )
      if self.num_pipeline_microbatches == -1:
        if self.pipeline_delay_activation_forwarding:
          self.num_pipeline_microbatches = 2 * num_stages
        else:
          self.num_pipeline_microbatches = num_stages

      assert self.num_pipeline_microbatches > 0, "num_pipeline_microbatches must be positive"
      assert self.num_pipeline_microbatches % num_stages == 0, (
          f"The number of microbatches ({self.num_pipeline_microbatches}) must be divisible by the number of "
          f"stages ({num_stages})"
      )
      if self.micro_batch_size_to_train_on > 0:
        assert self.micro_batch_size_to_train_on % self.num_pipeline_microbatches == 0, (
            f"The batch size for a single forward pass ({self.micro_batch_size_to_train_on}) must be divisible "
            f"by the number of microbatches ({self.num_pipeline_microbatches})"
        )
      if self.pipeline_delay_activation_forwarding:
        assert self.num_pipeline_microbatches >= 2 * num_stages, (
            f"Delayed activation forwarding requires at least 2 * num_stages microbatches, but {num_stages} stages "
            f"are used with {self.num_pipeline_microbatches} microbatches"
        )

      # For AOT compilation and correctness, always prioritize the 'stage' axis for sharding when pipelining.
      for rule in self.logical_axis_rules:
        if rule and rule[0] == "activation_embed_and_logits_batch":
          rule[1] = ["stage", "data", "fsdp", "fsdp_transpose", "expert"]
          break

      if "stage" in self.mesh_axes:
        stage_idx = self.mesh_axes.index("stage")
        data_idx = self.mesh_axes.index("data")
        if stage_idx > data_idx:  # Ensure 'stage' comes before 'data' for correct sharding logic.
          self.mesh_axes[stage_idx], self.mesh_axes[data_idx] = (
              self.mesh_axes[data_idx],
              self.mesh_axes[stage_idx],
          )

      # Adjust data_sharding to also prioritize 'stage'.
      if (
          isinstance(self.data_sharding, list)
          and len(self.data_sharding) > 0
          and self.data_sharding[0]
          and "stage" in self.data_sharding[0]
      ):
        self.data_sharding[0].remove("stage")
        self.data_sharding[0].insert(0, "stage")

      # Add sharding for FP8 amax history when using pipeline parallelism.
      if self.quantization and self.quantization in (
          "fp8",
          "nanoo_fp8",
          "fp8_gpu",
          "te_fp8_delayedscaling",
      ):
        self.logical_axis_rules.append(["aqt_amax_history", ("stage",)])

    # H. RUN ALL CROSS-FIELD VALIDATIONS
    if self.load_parameters_path and self.load_full_state_path:
      raise ValueError("At most one of `load_parameters_path` or `load_full_state_path` should be set.")
    if (self.load_parameters_path or self.load_full_state_path) and not self.enable_checkpointing:
      raise ValueError("You must set enable_checkpointing=True to load a checkpoint.")
    if self.enable_multi_tier_checkpointing:
      if not self.local_checkpoint_directory:
        raise ValueError("`local_checkpoint_directory` must be set for multi-tier checkpointing.")
      if self.local_checkpoint_period <= 0:
        raise ValueError("`local_checkpoint_period` must be > 0 for multi-tier checkpointing.")
      if self.multi_tier_checkpointing_backup_interval_minutes <= 0:
        raise ValueError("`multi_tier_checkpointing_backup_interval_minutes` must be > 0.")
    if self.enable_emergency_checkpoint:
      if not self.local_checkpoint_directory:
        raise ValueError("`local_checkpoint_directory` must be set for emergency checkpointing.")
      if self.local_checkpoint_period <= 0:
        raise ValueError("`local_checkpoint_period` must be > 0 for emergency checkpointing.")
    if self.moba and self.attention not in ("dot_product"):
      raise ValueError("MoBA is only supported with dot_product attention.")
    if self.use_sparse_indexer:
      if self.q_lora_rank == 0:
        raise NotImplementedError("Sparse indexer has not implemented for q_lora_rank = 0.")
      supports_dot_product = self.attention == "dot_product"
      supports_flash_splash = self.attention == "flash" and self.use_tokamax_splash
      if not (supports_dot_product or supports_flash_splash):
        raise NotImplementedError(
            "Sparse indexer is only supported dot_product attention or flash attention with tokamax splash."
        )
    if self.attention_type == AttentionType.CHUNK.value and (
        not isinstance(self.chunk_attn_window_size, int) or self.chunk_attn_window_size <= 0
    ):
      raise ValueError("`chunk_attn_window_size` must be an integer > 0 for 'chunk' attention.")
    if self.attention_type == AttentionType.LOCAL_SLIDING.value and (
        not isinstance(self.sliding_window_size, int) or self.sliding_window_size <= 0
    ):
      raise ValueError("`sliding_window_size` must be an integer > 0 for 'local_sliding' attention.")
    if self.quantize_kvcache and not self.kv_quant_axis:
      raise ValueError("`kv_quant_axis` cannot be empty when quantize_kvcache is True.")
    if (
        self.quantization in ("fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling")
        and self.gradient_accumulation_steps > 1
    ):
      raise ValueError("FP8 quantization is not compatible with gradient accumulation.")
    if self.engram_layers:
      if not self.hf_access_token or not self.tokenizer_path:
        raise ValueError(
            "Engram requires both 'hf_access_token' and 'tokenizer_path' " "to load the Hugging Face tokenizer."
        )
      if self.scan_layers:
        raise NotImplementedError("Currently Engram only supports unscanned version. Please set scan_layers=False.")
      if len(self.engram_vocab_bases) != (self.engram_max_ngram_size - 1):
        raise ValueError(
            f"Engram vocab size mismatch: expected {self.engram_max_ngram_size - 1} (max_ngram_size - 1), "
            f"but got {self.engram_vocab_bases}."
        )
    if self.num_experts > 1:
      is_fully_moe = (
          self.interleave_moe_layer_step == 1
          and self.first_num_dense_layers == 0
          and self.inhomogeneous_layer_cycle_interval == 1
      )
      if is_fully_moe and self.base_mlp_dim != self.base_moe_mlp_dim:
        if self.decoder_block == DecoderBlockType.QWEN3_MOE:
          self.base_mlp_dim = self.base_moe_mlp_dim
          _, _, mlp_dim_scale, _ = get_individual_scales(self.global_parameter_scale)
          self.mlp_dim = (2**mlp_dim_scale) * self.base_mlp_dim
        else:
          raise ValueError(
              "For a fully MoE model, base_mlp_dim must equal base_moe_mlp_dim. "
              f"Got base_mlp_dim={self.base_mlp_dim}, base_moe_mlp_dim={self.base_moe_mlp_dim}."
          )
      if self.decoder_block == DecoderBlockType.GPT_OSS and not self.sparse_matmul and self.capacity_factor != -1:
        raise ValueError("GPT-OSS MoE only supports dropless (capacity_factor=-1) with dense matmul.")
      if self.routed_bias and self.routed_bias_update_rate > 0.0 and self.decoder_block != DecoderBlockType.DEEPSEEK:
        raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")
    if self.use_multimodal:
      valid_mm_models = (
          "gemma3-4b",
          "gemma3-12b",
          "gemma3-27b",
          "llama4-17b-16e",
          "llama4-17b-128e",
          "qwen3-omni-30b-a3b",
      )
      if self.model_name not in valid_mm_models and self.model_name != "default":
        raise ValueError(f"Multimodal is only supported for {valid_mm_models}, not {self.model_name}")
      if self.use_sft:
        if not self.sft_train_on_completion_only:
          raise ValueError("For multimodal SFT, `sft_train_on_completion_only` must be True.")
        if self.packing:
          raise ValueError("For multimodal SFT, `packing` is not yet supported.")
    if self.shard_mode == ShardMode.EXPLICIT:
      supported_decoders = {"simple", "simple_mlp", "llama2", "deepseek"}
      if self.decoder_block.value not in supported_decoders:
        raise ValueError(
            f"Decoder '{self.decoder_block.value}' is not supported with 'explicit' sharding. "
            f"Supported options are: {list(supported_decoders)}."
        )
      if self.quantization:
        raise ValueError("Quantization is not supported with 'explicit' sharding.")
    if (
        self.per_device_batch_size > 0
        and (self.per_device_batch_size * self.max_target_length) % self.num_vocab_tiling != 0
    ):
      raise ValueError("Per device batch size times sequence length should be divisible by the number of vocab tiles.")
    if self.num_vocab_tiling > 1 and self.enable_nnx:
      raise ValueError("We currently don't support vocab tiling on NNX module.")
    if self.context_parallel_size > 1 and self.context_parallel_strategy.lower() == "ring":
      if "gpu" not in self.hardware:
        raise ValueError(
            "Ring context parallelism strategy (context_parallel_strategy='ring') is only supported on GPUs."
        )
    if self.hardware == "gpu" and self.packing and self.attention == "cudnn_flash_te" and self.max_segments_per_seq <= 0:
      raise ValueError("max_segments_per_seq must be set when using TransformerEngine attention and packing")
    dcn_product = (
        self.dcn_data_parallelism
        * self.dcn_pipeline_parallelism
        * self.dcn_fsdp_parallelism
        * self.dcn_fsdp_transpose_parallelism
        * self.dcn_sequence_parallelism
        * self.dcn_context_parallelism
        * self.dcn_tensor_parallelism
        * self.dcn_tensor_sequence_parallelism
        * self.dcn_expert_parallelism
        * self.dcn_context_autoregressive_parallelism
        * self.dcn_autoregressive_parallelism
    )
    if abs(dcn_product) > 1:
      if not self.num_slices > 1:
        raise ValueError("DCN parallelism requested but only one slice available.")
    if self.decoder_block == DecoderBlockType.LLAMA4:
      if self.capacity_factor >= 0:
        raise ValueError(
            "Llama4 decoder has not been tested with capacity_factor >= 0 -- please set that value to -1 for now!"
        )
      if self.num_experts_per_tok > 1:
        raise ValueError("Only top-1 routing is supported for Llama4 for now!")
      if self.base_num_decoder_layers % self.interleave_moe_layer_step != 0:
        raise ValueError(
            f"The number of decoder layers ({self.base_num_decoder_layers}) must be divisible by interleave moe layer step "
            f"({self.interleave_moe_layer_step})"
        )
    if self.decoder_block == DecoderBlockType.QWEN3_NEXT:
      if int(self.gdn_num_value_heads) % int(self.gdn_num_key_heads) != 0:
        raise ValueError("gdn_num_value_heads must be divisible by gdn_num_key_heads")
      rotary_dim = int(self.head_dim * self.partial_rotary_factor)
      if rotary_dim % 2 != 0:
        raise ValueError(f"Calculated rotary dimension ({rotary_dim}) must be a multiple of 2.")
    else:
      if self.partial_rotary_factor is not None and self.partial_rotary_factor != 1.0:
        raise ValueError("`partial_rotary_factor` is only effective when `decoder_block` is set to 'qwen3_next'.")

    tokenizer_path = getattr(self, "tokenizer_path", None)
    if (
        tokenizer_path
        and "tokenizer_llama3.tiktoken" in tokenizer_path
        and getattr(self, "tokenizer_type", None) != "tiktoken"
    ):
      logger.warning("`tokenizer_type` is not 'tiktoken' when using llama3 tokenizer. Overriding to 'tiktoken'.")
      self.tokenizer_type = TokenizerType.TIKTOKEN
    # Data input validations
    if self.dataset_type == DatasetType.HF:
      if not self.hf_path:
        raise ValueError("hf_path can't be empty when dataset_type=hf")
      if self.hf_eval_files:
        self.hf_eval_split = "train"
      if self.eval_interval > 0 and not self.hf_eval_split:
        raise ValueError("Please specify hf_eval_split or set eval_interval to <=0.")
    elif self.dataset_type == DatasetType.GRAIN:
      if not self.grain_train_files and not self.grain_train_mixture_config_path:
        raise ValueError("When dataset_type=grain, please set grain_train_files or grain_train_mixture_config_path")
      if self.eval_interval > 0 and not self.grain_eval_files:
        raise ValueError("Please specify grain_eval_files or set eval_interval to <=0.")
      if self.tokenizer_type not in (TokenizerType.SENTENCEPIECE, TokenizerType.HUGGINGFACE):
        raise ValueError(
            f"grain pipeline only supports tokenizer_type: sentencepiece, huggingface, but got {self.tokenizer_type}"
        )
    elif self.dataset_type == DatasetType.TFDS:
      if not self.dataset_name:
        raise ValueError("dataset_name can't be empty when dataset_type=tfds")
      if self.eval_interval > 0 and not self.eval_split:
        raise ValueError("Please specify eval_split or set eval_interval to <=0.")

    if self.sharding_tolerance > 1.0 or self.sharding_tolerance < 0.0:
      logger.warning("'sharding_tolerance: allowed percentage of non-sharded parameters' should be between 0.0 and 1.0")

    if self.eval_interval > 0 >= self.eval_steps and self.generate_padding_batch_eval:
      raise ValueError("`eval_steps` must be > 0 when `generate_padding_batch_eval` is True.")
    if self.rl.loss_algo == "grpo":
      self.use_grpo = True
    else:
      self.use_grpo = False
    if self.opt_type == "muon" and self.decoder_block not in [
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.QWEN3,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.LLAMA2,
    ]:
      raise ValueError(
          "Muon dimension numbers haven't been tested for this model. Run this command first: "
          f"`python3 -m MaxText.muon_utils {self.model_name} True`"
      )
    if self.force_q_layout and not self.use_jax_splash:
      raise ValueError("`force_q_layout` can only be true if `use_jax_splash` is also true.")

    if self.use_qk_clip and self.attention_type != "mla":
      raise ValueError(
          f"QK-Clip is only supported when attention_type='mla', but found attention_type='{self.attention_type}'."
      )

    if self.use_qk_clip and self.attn_logits_soft_cap is not None:
      raise ValueError(
          "QK-Clip monitors raw dot products, but attn_logits_soft_cap is enabled. "
          "Recording pre-cap max_logits is not fully supported yet. "
          "Please disable attn_logits_soft_cap when using use_qk_clip."
      )

    # I. FINAL TYPE CONVERSIONS AND DERIVED LISTS
    # Create the ici_parallelism and dcn_parallelism lists for legacy compatibility.
    if self.using_pipeline_parallelism and self.mesh_axes and self.mesh_axes[0] == "stage":
      self.ici_parallelism = [
          self.ici_diloco_parallelism,
          self.ici_pipeline_parallelism,
          self.ici_data_parallelism,
          self.ici_fsdp_parallelism,
          self.ici_fsdp_transpose_parallelism,
          self.ici_sequence_parallelism,
          self.ici_context_parallelism,
          self.ici_context_autoregressive_parallelism,
          self.ici_tensor_parallelism,
          self.ici_tensor_transpose_parallelism,
          self.ici_tensor_sequence_parallelism,
          self.ici_expert_parallelism,
          self.ici_autoregressive_parallelism,
      ]
      self.dcn_parallelism = [
          self.dcn_diloco_parallelism,
          self.dcn_pipeline_parallelism,
          self.dcn_data_parallelism,
          self.dcn_fsdp_parallelism,
          self.dcn_fsdp_transpose_parallelism,
          self.dcn_sequence_parallelism,
          self.dcn_context_parallelism,
          self.dcn_context_autoregressive_parallelism,
          self.dcn_tensor_parallelism,
          self.dcn_tensor_transpose_parallelism,
          self.dcn_tensor_sequence_parallelism,
          self.dcn_expert_parallelism,
          self.dcn_autoregressive_parallelism,
      ]
    else:
      ici_map = {
          "diloco": self.ici_diloco_parallelism,
          "data": self.ici_data_parallelism,
          "stage": self.ici_pipeline_parallelism,
          "fsdp": self.ici_fsdp_parallelism,
          "fsdp_transpose": self.ici_fsdp_transpose_parallelism,
          "sequence": self.ici_sequence_parallelism,
          "context": self.ici_context_parallelism,
          "context_autoregressive": self.ici_context_autoregressive_parallelism,
          "tensor": self.ici_tensor_parallelism,
          "tensor_transpose": self.ici_tensor_transpose_parallelism,
          "tensor_sequence": self.ici_tensor_sequence_parallelism,
          "model": self.ici_tensor_parallelism,
          "expert": self.ici_expert_parallelism,
          "autoregressive": self.ici_autoregressive_parallelism,
          "attn_dp": 1,  # initialized to 1, vLLM will auto calculate this value based on TP and num_kv_heads
      }
      self.ici_parallelism = [ici_map[axis] for axis in self.mesh_axes]

      dcn_map = {
          "diloco": self.dcn_diloco_parallelism,
          "data": self.dcn_data_parallelism,
          "stage": self.dcn_pipeline_parallelism,
          "fsdp": self.dcn_fsdp_parallelism,
          "fsdp_transpose": self.dcn_fsdp_transpose_parallelism,
          "sequence": self.dcn_sequence_parallelism,
          "context": self.dcn_context_parallelism,
          "context_autoregressive": self.dcn_context_autoregressive_parallelism,
          "tensor": self.dcn_tensor_parallelism,
          "tensor_transpose": self.dcn_tensor_transpose_parallelism,
          "tensor_sequence": self.dcn_tensor_sequence_parallelism,
          "model": self.dcn_tensor_parallelism,
          "expert": self.dcn_expert_parallelism,
          "autoregressive": self.dcn_autoregressive_parallelism,
          "attn_dp": 1,  # initialized to 1, vLLM will auto calculate this value based on TP and num_kv_heads
      }
      self.dcn_parallelism = [dcn_map[axis] for axis in self.mesh_axes]

    # Diloco params
    self.num_diloco_replicas = int(self.ici_diloco_parallelism * self.dcn_diloco_parallelism)

    # Final string-to-enum conversions if they haven't been coerced by pydantic yet.
    if isinstance(self.decoder_block, str):
      self.decoder_block = DecoderBlockType(self.decoder_block.lower())
    if isinstance(self.shard_mode, str):
      self.shard_mode = ShardMode(self.shard_mode.lower())

    constant_bound_config = getattr(self, "constant_bound_config", None)
    if isinstance(constant_bound_config, str):
      if constant_bound_config:
        self.constant_bound_config = [float(v.strip()) for v in constant_bound_config.split(",")]
      else:
        self.constant_bound_config = []

    return self
