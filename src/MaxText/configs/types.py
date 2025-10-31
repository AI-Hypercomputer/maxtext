# Copyright 2023–2025 Google LLC
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

"""Pydantic-based configuration system for MaxText, organized into modular classes."""

# pylint: disable=too-many-lines

import os
import sys
from enum import Enum
from tempfile import gettempdir
from typing import Any, NewType, Literal
import math
from math import prod

import jax

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
    field_validator,
)
from MaxText import accelerator_to_spec_map
from MaxText.common_types import DecoderBlockType

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

type ModelName = Literal[
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
    "deepseek3-test",
    "deepseek3-tiny",
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
    "gpt3-175b",
    "gpt3-22b",
    "gpt3-6b",
    "gpt3-52k",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "llama4-17b-16e",
    "llama4-17b-128e",
]


class RunInfo(BaseModel):
  """Configuration for the overall run, model identity, and logging."""

  run_name: str = Field(
      "",
      description="The name of the run. Checkpoints will be stored under this name.",
  )
  model_name: ModelName = Field("default", description="The name of the model configuration to use.")
  override_model_config: bool = Field(False, description="If True, allows overriding model parameters via CLI.")
  log_config: bool = Field(
      True,
      description="If True, prints the final configuration after initialization.",
  )
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
  async_checkpointing: bool = Field(True, description="If True, uses an asynchronous checkpointer for performance.")
  checkpoint_period: int = Field(10_000, description="The frequency (in steps) at which to save checkpoints.")
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
  enable_checkpoint_cloud_logger: bool = Field(False, description="Enables structured logging for checkpointing.")


class EmergencyCheckpointing(BaseModel):
  """Configuration for emergency (local) checkpointing."""

  enable_multi_tier_checkpointing: bool = Field(
      False, description="Enables multi-tier checkpointing (local and persistent)."
  )
  local_checkpoint_directory: PathStr = Field("", description="Local directory for emergency checkpoints.")
  local_checkpoint_period: NonNegativeInt = Field(0, description="Frequency (in steps) for local emergency checkpoints.")
  multi_tier_checkpointing_backup_interval_minutes: NonNegativeInt = Field(
      0, description="Interval in minutes to back up local checkpoints to persistent storage."
  )
  mtc_data_parallelism: int = Field(
      0, description="Number of identical pipelines in the job for multi-tier checkpointing. 0 defaults to num_slices."
  )
  enable_emergency_checkpoint: bool = Field(
      False, description="Legacy flag for enabling emergency checkpointing. Prefer `enable_multi_tier_checkpointing`."
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
  quantization_calibration_method: str = Field(
      "absmax",
      description="Quantization calibration method used for weights and activations.",
  )


class ModelArchitecture(BaseModel):
  """Core model architecture parameters."""

  decoder_block: str = Field(
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
      -1.0, description="Upper bound to clip the MLP activation values. -1.0 means no clipping."
  )
  normalization_layer_epsilon: float = Field(1.0e-05, description="Epsilon value for normalization layers.")
  fused_qkv: bool = Field(False, description="If supported, fuse the Q, K, and V projections.")
  attention_bias: bool = Field(
      False, description="If True, adds a learnable bias to the query, key, and value projections."
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
  attention_type: Literal["global", "local_sliding", "chunk", "mla"] = Field(
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
  use_tokamax_gmm: bool = Field(False, description="Whether to use the Tokamax library for GMM kernel implementation.")
  ragged_block_size: int = Field(256, description="Block size for ragged attention.")
  enable_padding_causal_mask: bool = Field(True, description="Temporary flag for TE padding.")


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


class PagedAttention(BaseModel):
  """Tunable parameters for Paged Attention kernels."""

  pagedattn_num_pages: int = Field(64, description="Total number of pages to allocate for paged attention.")
  pagedattn_tokens_per_page: int = Field(32, description="Number of tokens each page can hold.")
  pagedattn_pages_per_compute_block: int = Field(4, description="Number of pages processed together in pallas kernels.")
  pagedattn_max_pages_per_group: int = Field(-1, description="Max pages per request; -1 defaults to max_target_length.")
  pagedattn_head_dim_alignment: int = Field(128, description="Alignment of head_dim to the nearest multiple.")


class MoEGeneral(BaseModel):
  """General configuration for Mixture of Experts (MoE) layers."""

  num_experts: PositiveInt = Field(1, description="The total number of experts in each MoE layer.")
  num_experts_per_tok: PositiveInt = Field(1, description="The number of experts to route each token to.")
  capacity_factor: float = Field(-1.0, description="Expert capacity factor. If < 0, no token dropping.")
  load_balance_loss_weight: NonNegativeFloat = Field(0.01, description="Weight for the load balancing auxiliary loss.")
  use_custom_sort_vjp: bool = Field(True, description="Whether to use a custom sort VJP for sparse matmul ops.")
  use_ring_of_experts: bool = Field(
      False, description="Whether to use Ring of Experts for sparse matmul expert parallelism."
  )
  use_random_routing: bool = Field(False, description="Whether to use random routing for debugging.")
  interleave_moe_layer_step: int = Field(1, description="Frequency of MoE layers, e.g., 2 means every 2nd layer is MoE.")
  expert_shard_attention_option: Literal["fsdp", "context"] = Field(
      "fsdp", description="How the expert axis is used to shard attention weights and activations."
  )
  moe_fsdp_use_two_stage_all_gather: bool = Field(
      False, description="Use two separate All-Gather calls for MoE weights sharded on both FSDP and FSDP-transpose."
  )
  fsdp_shard_on_exp: bool = Field(
      False,
      description="Shard the MoE weights on the num_expert dimension. Can be performant when "
      "num_experts % fsdp_parallelism != 0.",
  )
  norm_topk_prob: bool = Field(
      False, description="Enable top-k probability normalization for router weights (Qwen3-specific)."
  )


class MoEKernels(BaseModel):
  """Configuration for MoE-specific kernels like Megablox."""

  megablox: bool = Field(True, description="Whether to use Megablox kernels for MoE.")
  sparse_matmul: bool = Field(True, description="Whether to use sparse matmul kernels for MoE.")
  tile_batch_seq: int = Field(512, description="Tunable tiling dimension for batch/sequence in Megablox.")
  tile_embed_dim: int = Field(1024, description="Tunable tiling dimension for embedding in Megablox.")
  tile_mlp_dim: int = Field(1024, description="Tunable tiling dimension for MLP in Megablox.")


class DeepSeekMoE(BaseModel):
  """Configuration specific to DeepSeek-style MoE layers."""

  base_moe_mlp_dim: int = Field(7168, description="Intermediate dimension at MoE layer (DeepSeek style).")
  first_num_dense_layers: NonNegativeInt = Field(0, description="Number of initial dense layers in the model.")
  shared_experts: PositiveInt = Field(1, description="Number of shared experts.")
  routed_scaling_factor: float = Field(1.0, description="Scaling factor for routing scores.")
  routed_score_func: str = Field("", description="Scoring function for routing (e.g., 'softmax', 'sigmoid').")
  routed_bias: bool = Field(False, description="Whether to add a bias term for routing.")
  mlp_bias: bool = Field(False, description="Whether to add a learnable bias for MLP matmul.")
  n_routing_groups: int = Field(-1, description="Number of groups for routing, disabled by default.")
  topk_routing_group: int = Field(-1, description="Number of top groups to route inputs to.")
  use_batch_split_schedule: bool = Field(
      False,
      description="Splits the batch to allow for better scheduling when using expert parallelism by overlapping all-to-all "
      "with compute.",
  )


class Qwen3Next(BaseModel):
  """Configuration specific to Qwen3-Next models with Gated Delta Net."""

  gdn_conv_kernel_dim: int = Field(4, description="Kernel size for the 1D convolution in the Gated Delta Net.")
  gdn_key_head_dim: int = Field(128, description="Head dimension for the key/query in the Gated Delta Net.")
  gdn_value_head_dim: int = Field(128, description="Head dimension for the value in the Gated Delta Net.")
  gdn_num_key_heads: int = Field(16, description="Number of key/query heads in the Gated Delta Net.")
  gdn_num_value_heads: int = Field(32, description="Number of value heads in the Gated Delta Net.")
  gdn_chunk_size: int = Field(64, description="Chunk size for the parallel scan algorithm in the Gated Delta Net.")
  use_qk_norm_in_gdn: bool = Field(
      True,
      description="Whether to apply L2 normalization to query and key tensors inside the Gated Delta Rule kernel.",
  )


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
      ]
  )
  inhomogeneous_layer_cycle_interval: int = Field(1, description="The interval of repeated inhomogeneous layer patterns.")
  scan_layers: bool = Field(True, description="Whether to use jax.lax.scan over layers.")
  param_scan_axis: int = Field(1, description="Axis to scan over for parameters.")
  context_parallel_load_balance: bool = Field(True, description="Whether to use load balancing for context parallelism.")
  custom_mesh: str = Field("", description="Available options: ['hybrid_ring_64x4', 'hybrid_ring_32x8']")
  allow_split_physical_axes: bool = Field(False, description="Allow splitting physical axes for device mesh creation.")
  enable_nnx: bool = Field(False, description="Whether to use NNX for model definition.")
  optimize_mesh_for_tpu_v6e: bool = Field(False, description="Apply transformations to the mesh for TPU v6e.")
  shardy: bool = Field(True, description="Whether to use shardy XLA backend.")


class LayoutAndSharding(BaseModel):
  """Configuration for data and model sharding rules."""

  logical_axis_rules: list[list[Any]] = Field([], description="Rules for mapping logical axes to physical mesh axes.")
  data_sharding: list[list[str]] = Field([], description="Sharding for input data.")
  input_data_sharding_logical_axes: list[str] = Field(["activation_embed_and_logits_batch", "activation_norm_length"])
  sharding_tolerance: float = Field(
      0.02,
      ge=0.0,
      le=1.0,
      description="Allowed percentage of non-sharded parameters.",
  )
  shard_optimizer_over_data: bool = Field(False, description="Enable ZeRO-1 optimizer sharding over the data axis.")


class DcnParallelism(BaseModel):
  """Parallelism dimensions across the DCN (Data Center Network)."""

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
  optimizer_memory_host_offload: bool = Field(False, description="Offload optimizer state to host memory.")
  parameter_memory_host_offload: bool = Field(False, description="Offload parameters to host memory.")


class Tokenizer(BaseModel):
  """Configuration for the tokenizer."""

  vocab_size: int = Field(32_000, description="The size of the vocabulary.")
  tokenizer_path: PathStr = Field(
      os.path.join("assets", "tokenizer.llama2"),
      description="Path to the tokenizer model file.",
  )
  tokenizer_type: TokenizerType = Field(TokenizerType.SENTENCEPIECE, description="The type of tokenizer.")
  use_chat_template: bool = Field(False, description="Whether to use the chat template for tokenization.")
  tokenize_train_data: bool = Field(True, description="If False, assumes the training dataset is pre-tokenized.")
  tokenize_eval_data: bool = Field(True, description="If False, assumes the evaluation dataset is pre-tokenized.")
  add_bos: bool = Field(True, description="Whether to add a beginning-of-sentence token.")
  add_eos: bool = Field(True, description="Whether to add an end-of-sentence token.")
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
  num_epoch: int = Field(1, description="Number of epochs to train for.")
  expansion_factor_real_data: int = Field(-1, description="Factor for partial data loading on hosts.")
  reuse_example_batch: int = Field(0, description="For performance testing, repeatedly uses the same batch.")
  generate_padding_batch_train: bool = Field(
      False, description="Whether to generate a padding batch for training to ensure divisibility."
  )
  generate_padding_batch_eval: bool = Field(
      False, description="Whether to generate a padding batch for evaluation to ensure divisibility."
  )
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

  hf_path: str = Field("", description="Path or name of the Hugging Face dataset.")
  hf_data_dir: PathStr = Field("", description="Data directory for the HF dataset.")
  hf_train_files: str = Field("", description="Files for the HF training split.")
  hf_eval_split: str = Field("", description="Name of the HF evaluation split.")
  hf_eval_files: str = Field("", description="Files for the HF evaluation split.")
  hf_access_token: None | str = Field(None, description="Hugging Face API access token.")


class GrainDataset(BaseModel):
  """Configuration specific to Grain datasets."""

  grain_train_files: PathStr = Field("", description="Path to Grain training files.")
  grain_eval_files: PathStr = Field("", description="Path to Grain evaluation files.")
  grain_file_type: str = Field("arrayrecord", description="File type for Grain data.")
  grain_worker_count: int = Field(1, description="Number of workers for Grain data loading.")
  grain_worker_count_eval: int = Field(1, description="Number of workers for Grain eval data loading.")


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
  grpo_beta: None | float = Field(None, description="Beta parameter for GRPO.")
  num_generations: None | int = Field(None, description="Number of generations for GRPO.")


class TrainingLoop(BaseModel):
  """Configuration for the main training loop, evaluation, and reproducibility."""

  steps: int = Field(
      150_001,
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


class Optimizer(BaseModel):
  """Configuration for the optimizer and learning rate schedule."""

  opt_type: OptimizerType = Field(OptimizerType.ADAMW, description="The type of optimizer to use.")
  gradient_accumulation_steps: PositiveInt = Field(
      1, description="Number of steps to accumulate gradients before updating."
  )
  gradient_clipping_threshold: NonNegativeFloat = Field(
      1.0, description="The threshold for gradient clipping. 0 disables clipping."
  )
  learning_rate: NonNegativeFloat = Field(3.0e-5, description="The peak learning rate.")
  cosine_learning_rate_final_fraction: float = Field(
      0.1, description="Final LR as a fraction of peak LR in cosine decay."
  )
  warmup_steps_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of total steps for LR warmup.")
  learning_rate_schedule_steps: int = Field(-1, description="Total steps for the LR schedule. -1 defaults to `steps`.")


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
      False, description="Scale the rotary embedding output. Used by some models like gpt-oss."
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
  prefill_slice: str = Field("v5e-16", description="Slice for prefill in disaggregation mode.")
  generate_slice: str = Field("v5e-16", description="Slice for generation in disaggregation mode.")


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
  jax_cache_dir: PathStr = Field(
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

  @field_validator("constant_bound_config", mode="before")
  @classmethod
  def _clean_empty_string_for_list(cls, v: Any) -> Any:
    """Coerces an empty string from YAML into an empty list before validation."""
    if v == "":
      return []
    if isinstance(v, str):
      return list(map(float, v.split(",")))
    return v


class Profiling(BaseModel):
  """Configuration for performance profiling."""

  profiler: ProfilerType = Field(ProfilerType.NONE, description="Profiler to use ('xplane', 'nsys').")
  upload_all_profiler_results: bool = Field(False, description="Upload profiler results from all hosts.")
  skip_first_n_steps_for_profiler: int = Field(1, description="Number of initial steps to skip for profiling.")
  profiler_steps: int = Field(5, description="Number of steps to profile.")
  profile_cleanly: bool = Field(True, description="Add block_until_ready to align profile for each step.")
  profile_periodically_period: int = Field(-1, description="If positive, profile every N steps.")
  enable_jax_profiler: bool = Field(False, description="Enable the JAX live profiler.")
  jax_profiler_port: int = Field(9999, description="Port for the JAX profiler.")


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


class StackTrace(BaseModel):
  """Configuration for collecting and logging stack traces."""

  collect_stack_trace: bool = Field(False, description="Enable periodic stack trace collection.")
  stack_trace_to_cloud: bool = Field(False, description="Upload stack traces to cloud logging instead of console.")
  stack_trace_interval_seconds: int = Field(600, description="Frequency of stack trace collection in seconds.")


class Metrics(BaseModel):
  """General configuration for metrics and monitoring."""

  metrics_file: PathStr = Field("", description="Local file to store scalar metrics for testing.")
  gcs_metrics: bool = Field(False, description="If True, save metrics to GCS.")
  save_config_to_gcs: bool = Field(False, description="If True, save config to GCS.")
  record_internal_nn_metrics: int = Field(0, description="Record internal neural network metrics.")
  prometheus_port: int = Field(0, description="Port for Prometheus metrics server. 0 disables it.")


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
  vertex_tensorboard_project: str = Field("", description="GCP project for Vertex AI Tensorboard.")
  vertex_tensorboard_region: str = Field("", description="Region for Vertex AI Tensorboard.")


class MultimodalGeneral(BaseModel):
  """General configuration for Multimodal models."""

  use_multimodal: bool = Field(False, description="Enable multimodal capabilities.")
  freeze_vision_encoder_params: bool = Field(True, description="Freeze the parameters of the vision encoder.")
  image_size_for_vit: int = Field(896, description="Input image size for the Vision Transformer.")
  image_path: PathStr = Field("", description="Path to an image for decoding.")
  image_placeholder: str = Field("<|image|>", description="Placeholder string for images in text prompts.")
  remat_policy_for_vit: str = Field("minimal", description="Rematerialization policy for the vision encoder.")
  posemb_type_for_vit: str = Field("learn", description="Positional embedding type for the vision encoder.")
  max_num_images_per_example: int = Field(
      -1,
      description="Maximum number of images per example for training with image lists. -1 means no limit.",
  )


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
  conv_stride_for_vit: int = Field(14, description="Convolutional stride for the Vision Transformer's patch embedding.")
  num_hidden_layers_for_vit: int = Field(34, description="Number of hidden layers in the Vision Transformer.")
  rope_theta_for_vit: int = Field(10000, description="RoPE theta value for the Vision Transformer.")


class VisionProjector(BaseModel):
  """Configuration for the Vision Projector in a multimodal model."""

  projector_input_dim_for_vit: int = Field(4096, description="Input dimension for the vision projector.")
  projector_output_dim_for_vit: int = Field(4096, description="Output dimension for the vision projector.")
  vision_output_dim_for_vit: int = Field(4096, description="Final output dimension of the vision-to-language projection.")
  pixel_shuffle_ratio_for_vit: float = Field(0.5, description="Pixel shuffle ratio for the Vision Transformer.")
  projector_dropout_for_vit: float = Field(0.0, description="Dropout rate for the vision projector.")


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
  model_fsdp_ag_once: bool = Field(
      False,
      description="An alias for `pipeline_fsdp_ag_once` for backward compatibility.",
  )

  context_parallel_size: None | int = Field(
      None,
      description="The total size of context parallelism, derived from ICI and DCN values.",
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


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def get_individual_scales(scale):
  """Choose appropriate scales for individual dimensions based on global scale."""
  log_2_scale = math.floor(math.log2(scale))
  if 2**log_2_scale != scale:
    raise ValueError(
        "Global parameter scale should be a power of 2. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_heads, base_mlp_dim, base_num_decoder_layers and/or head_dim."
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
    MTP,
    Logits,
    # Attention Mechanisms
    Attention,
    MlaAttention,
    MoBa,
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
    Optimizer,
    AdamW,
    FineTuning,
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
    # Multimodal
    MultimodalGeneral,
    VisionTower,
    VisionProjector,
    # Derived
    DerivedValues,
):
  """
  The main configuration object for MaxText.

  This class aggregates all configuration options from modular `BaseModel` classes
  into a single, validated object. It is populated by the `initialize` function.
  Every field is explicitly defined to prevent misconfigurations (`extra='forbid'`).
  """

  class Config:
    extra = "forbid"
    protected_namespaces = ()

  @model_validator(mode="after")
  def set_derived_values(self) -> "MaxTextConfig":
    """
    Computes derived values and aliases after initial validation to perfectly
    replicate the post-processing logic of the legacy configuration system.
    """
    # A. Set primary dependencies first
    if self.learning_rate_schedule_steps == -1:
      self.learning_rate_schedule_steps = self.steps
    if self.eval_per_device_batch_size == 0.0:
      self.eval_per_device_batch_size = self.per_device_batch_size
    if not self.mu_dtype:
      self.mu_dtype = self.weight_dtype

    # B. Calculate model dimensions based on global_parameter_scale
    emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(self.global_parameter_scale)
    self.emb_dim = (2**emb_scale) * self.base_emb_dim
    self.num_query_heads = (2**num_head_scale) * self.base_num_query_heads
    self.num_kv_heads = (2**num_head_scale) * self.base_num_kv_heads
    self.mlp_dim = (2**mlp_dim_scale) * self.base_mlp_dim
    self.moe_mlp_dim = (2**mlp_dim_scale) * self.base_moe_mlp_dim
    self.num_decoder_layers = (2**layer_scale) * self.base_num_decoder_layers

    # C. Get number of devices, replicating legacy logic
    num_devices = 0
    try:
      if self.compile_topology:
        compile_topology_spec = accelerator_to_spec_map.get_system_characteristics(self.compile_topology)
        devices_per_slice = compile_topology_spec.devices_per_slice
        num_devices = int(devices_per_slice * self.compile_topology_num_slices)
      elif self.subslice_shape and self.enable_single_controller:
        subslice_shape_tuple = tuple(int(x) for x in self.subslice_shape.split(","))
        num_devices = prod(subslice_shape_tuple)
      else:
        num_devices = len(jax.devices())
    except (
        RuntimeError,
        IndexError,
    ):  # Handle cases where jax device system is not initialized
      print(
          "Warning: JAX device system not available for config validation. Assuming 1 device.",
          file=sys.stderr,
      )
      num_devices = 1

    # D. Calculate batch sizes, replicating legacy logic
    # Train
    if self.per_device_batch_size < 1.0:
      micro_batch_size_to_load = num_devices * (
          self.expansion_factor_real_data if self.expansion_factor_real_data > 0 else 1
      )
    else:
      micro_batch_size_to_load = int(
          num_devices
          * self.per_device_batch_size
          * (self.expansion_factor_real_data if self.expansion_factor_real_data > 0 else 1)
      )
    micro_batch_size_to_train_on = int(num_devices * self.per_device_batch_size)
    self.global_batch_size_to_load = int(micro_batch_size_to_load * self.gradient_accumulation_steps)
    self.global_batch_size_to_train_on = int(micro_batch_size_to_train_on * self.gradient_accumulation_steps)
    self.micro_batch_size_to_train_on = micro_batch_size_to_train_on
    # Eval
    if self.eval_per_device_batch_size < 1.0:
      micro_batch_size_to_load_eval = num_devices * (
          self.expansion_factor_real_data if self.expansion_factor_real_data > 0 else 1
      )
    else:
      micro_batch_size_to_load_eval = int(
          num_devices
          * self.eval_per_device_batch_size
          * (self.expansion_factor_real_data if self.expansion_factor_real_data > 0 else 1)
      )
    micro_batch_size_to_eval_on = int(num_devices * self.eval_per_device_batch_size)
    self.global_batch_size_to_load_eval = micro_batch_size_to_load_eval
    self.global_batch_size_to_eval_on = micro_batch_size_to_eval_on
    self.micro_batch_size_to_eval_on = micro_batch_size_to_eval_on

    # E. Set other derived fields
    if self.quantization_local_shard_count == -1:
      try:
        self.quantization_local_shard_count = jax.local_device_count()
      except RuntimeError:
        self.quantization_local_shard_count = 1

    try:
      if self.num_slices == -1:
        self.num_slices = len(jax.devices()) // jax.local_device_count()
    except (RuntimeError, ZeroDivisionError):
      self.num_slices = 1

    self.ici_parallelism = [
        self.ici_data_parallelism,
        self.ici_pipeline_parallelism,
        self.ici_fsdp_parallelism,
        self.ici_tensor_parallelism,
        self.ici_sequence_parallelism,
        self.ici_context_parallelism,
        self.ici_autoregressive_parallelism,
        self.ici_expert_parallelism,
        self.ici_fsdp_transpose_parallelism,
        self.ici_tensor_transpose_parallelism,
        self.ici_tensor_sequence_parallelism,
        self.ici_context_autoregressive_parallelism,
    ]
    self.dcn_parallelism = [
        self.dcn_data_parallelism,
        self.dcn_pipeline_parallelism,
        self.dcn_fsdp_parallelism,
        self.dcn_tensor_parallelism,
        self.dcn_sequence_parallelism,
        self.dcn_context_parallelism,
        self.dcn_autoregressive_parallelism,
        self.dcn_expert_parallelism,
        self.dcn_fsdp_transpose_parallelism,
        self.dcn_tensor_transpose_parallelism,
        self.dcn_tensor_sequence_parallelism,
        self.dcn_context_autoregressive_parallelism,
    ]
    self.using_pipeline_parallelism = self.ici_pipeline_parallelism > 1 or self.dcn_pipeline_parallelism > 1
    self.model_fsdp_ag_once = self.pipeline_fsdp_ag_once

    # This calculation is from pyconfig_deprecated.py's get_context_parallel_size
    self.context_parallel_size = self.ici_context_parallelism * self.dcn_context_parallelism

    if self.run_name:
      output_dir = os.path.join(self.base_output_directory, self.run_name)
      self.checkpoint_dir = os.path.join(output_dir, "checkpoints") + "/"
      self.metrics_dir = os.path.join(output_dir, "metrics") + "/"
      self.tensorboard_dir = os.path.join(output_dir, "tensorboard") + "/"
    else:
      self.checkpoint_dir, self.metrics_dir, self.tensorboard_dir = (
          None,
          None,
          None,
      )

    if self.pagedattn_max_pages_per_group == -1:
      self.pagedattn_max_pages_per_group = self.max_target_length // self.pagedattn_tokens_per_page

    # F. Final cosmetic fixes for perfect legacy serialization
    if self.decoder_block.islower():
      try:
        self.decoder_block = str(DecoderBlockType[self.decoder_block.upper()])
      except KeyError:
        # If not in enum, keep as is.
        pass

    if self.final_logits_soft_cap == 0.0:
      self.final_logits_soft_cap = None
    if getattr(self, "attn_logits_soft_cap", 0.0) == 0.0:
      self.attn_logits_soft_cap = None
    if getattr(self, "hf_access_token", "") == "":
      self.hf_access_token = None
    if getattr(self, "decode_sampling_nucleus_p", -1.0) == -1.0:
      self.decode_sampling_nucleus_p = -1

    if self.per_device_batch_size is not None and self.per_device_batch_size == int(self.per_device_batch_size):
      self.per_device_batch_size = int(self.per_device_batch_size)
    if self.eval_per_device_batch_size is not None and self.eval_per_device_batch_size == int(
        self.eval_per_device_batch_size
    ):
      self.eval_per_device_batch_size = int(self.eval_per_device_batch_size)

    return self
