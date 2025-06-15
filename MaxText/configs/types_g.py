"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt, NonNegativeFloat, validator, RootModel

# TODO: Merge both `.types` into one
from MaxText.configs.types import ParallelismConfig


class DecoderBlockType(Enum):
  DEFAULT = "default"
  LLAMA2 = "llama2"
  MISTRAL = "mistral"
  MIXTRAL = "mixtral"
  DEEPSEEK = "deepseek"
  GEMMA = "gemma"
  GEMMA2 = "gemma2"
  GEMMA3 = "gemma3"
  GPT3 = "gpt3"
  SIMPLE = "simple"
  SIMPLE_MLP = "simple_mlp"
  LLAMA4 = "llama4"


class AttentionType(Enum):
  GLOBAL = "global"
  LOCAL_SLIDING = "local_sliding"
  CHUNK = "chunk"
  MLA = "mla"
  FULL = "full"


class OptimizerType(Enum):
  ADAMW = "adamw"
  ADAM_PAX = "adam_pax"
  SGD = "sgd"


class MatMulPrecision(Enum):
  DEFAULT = "default"
  HIGH = "high"
  HIGHEST = "highest"


class DatasetType(Enum):
  SYNTHETIC = "synthetic"
  HF = "hf"
  GRAIN = "grain"
  TFDS = "tfds"
  C4_MLPERF = "c4_mlperf"


class GrainFileType(Enum):
  ARRAYRECORD = "arrayrecord"
  PARQUET = "parquet"


class HardwareType(Enum):
  TPU = "tpu"
  GPU = "gpu"
  GPU_MULTIPROCESS = "gpu_multiprocess"
  CPU = "cpu"


class ProfilerType(Enum):
  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"


class AttentionKernel(Enum):
  AUTOSELECTED = "autoselected"
  DOT_PRODUCT = "dot_product"
  FLASH = "flash"
  CUDNN_FLASH_TE = "cudnn_flash_te"
  CUDNN_FLASH_JAX = "cudnn_flash_jax"
  PAGED = "paged"


class RematPolicy(Enum):
  MINIMAL = "minimal"
  SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP = "save_dot_with_context_except_mlp"
  SAVE_DOT_EXCEPT_MLPWI = "save_dot_except_mlpwi"
  SAVE_DOT_EXCEPT_MLP = "save_dot_except_mlp"
  SAVE_QKV_PROJ = "save_qkv_proj"
  QKV_PROJ_OFFLOADED = "qkv_proj_offloaded"
  CUSTOM = "custom"
  MINIMAL_OFFLOADED = "minimal_offloaded"
  SAVE_OUT_PROJ = "save_out_proj"
  FULL = "full"
  MINIMAL_FLASH = "minimal_flash"


class RematTensorConfigValue(Enum):
  REMAT = "remat"
  DEVICE = "device"
  OFFLOAD = "offload"


class ModelCallMode(Enum):
  TRAIN = ""
  INFERENCE = "inference"


class SamplingStrategy(Enum):
  GREEDY = "greedy"
  WEIGHTED = "weighted"
  NUCLEUS = "nucleus"
  TOPK = "topk"


class RoPEType(Enum):
  DEFAULT = "default"
  LLAMA3_1 = "llama3.1"
  YARN = "yarn"


class TokenizerTypeEnum(Enum):
  SENTENCEPIECE = "sentencepiece"
  TIKTOKEN = "tiktoken"
  HUGGINGFACE = "huggingface"


class InferenceServerType(Enum):
  MAXTEXT_INTERLEAVED = "MaxtextInterleavedServer"
  EXPERIMENTAL_MAXTEXT_DISAGGREGATED = "ExperimentalMaxtextDisaggregatedServer"


# Pydantic Models for Configuration Structure


class RunConfig(BaseModel):
  """Configuration related to a single run/experiment."""

  run_name: str = Field(default="", description="Name of the run. Auto-populated if empty.")
  base_output_directory: str = Field(description="Base directory for all outputs.")
  metrics_file: Optional[str] = Field(default="", description="Local file for scalar metrics; empty means no local file.")
  gcs_metrics: bool = Field(default=False, description="Save metrics to GCS.")
  save_config_to_gcs: bool = Field(default=False, description="Save config to GCS.")
  log_period: PositiveInt = Field(default=100, description="Frequency of TensorBoard flushes.")
  steps: int = Field(default=150_001, description="Total training steps. -1 to use learning_rate_schedule_steps.")
  enable_tensorboard: bool = Field(default=True, description="Enable TensorBoard logging.")
  use_vertex_tensorboard: bool = Field(default=False, description="Use Vertex AI TensorBoard.")
  vertex_tensorboard_project: Optional[str] = Field(default="", description="GCP project for Vertex AI TensorBoard.")
  vertex_tensorboard_region: Optional[str] = Field(default="", description="Region for Vertex AI TensorBoard.")
  log_config: bool = Field(default=True, description="Print the final configuration.")


class CheckpointLoadingConfig(BaseModel):
  """Configuration for loading checkpoints."""

  load_parameters_path: Optional[str] = Field(default="", description="Path to load parameters-only checkpoint.")
  lora_input_adapters_path: Optional[str] = Field(default="", description="GCS path for LoRA adapters directory.")
  load_full_state_path: Optional[str] = Field(default="", description="Path to load full training state checkpoint.")
  checkpoint_is_quantized: bool = Field(default=False, description="True if loading a quantized checkpoint.")


class CheckpointSavingConfig(BaseModel):
  """Configuration for saving checkpoints."""

  enable_checkpointing: bool = Field(default=True, description="Enable checkpointing.")
  async_checkpointing: bool = Field(default=True, description="Use asynchronous checkpointing.")
  checkpoint_period: NonNegativeInt = Field(default=10_000, description="Checkpoint saving frequency in steps.")
  force_unroll: bool = Field(default=False, description="Force unroll loop for generate_param_only_checkpoint.")
  save_quantized_params_path: Optional[str] = Field(default="", description="Path to save on-the-fly quantized params.")


class CheckpointStorageConfig(BaseModel):
  """Configuration for checkpoint storage backend."""

  checkpoint_storage_target_data_file_size_bytes: int = Field(
      default=2147483648, description="Target file size for Orbax checkpoint sharding."
  )
  checkpoint_storage_use_ocdbt: bool = Field(default=True, description="Use OCDBT for checkpointing.")
  checkpoint_storage_use_zarr3: bool = Field(default=True, description="Use Zarr3 for checkpointing.")
  checkpoint_storage_concurrent_gb: int = Field(default=96, description="Concurrent GB for checkpoint I/O.")


class EmergencyCheckpointConfig(BaseModel):
  """Configuration for emergency (local) checkpointing."""

  enable_emergency_checkpoint: bool = Field(default=False, description="Enable Orbax emergency checkpointing.")
  local_checkpoint_directory: Optional[str] = Field(default="", description="Local directory for emergency checkpoints.")
  local_checkpoint_period: NonNegativeInt = Field(default=0, description="Frequency for local emergency checkpoints.")
  use_replicator_service: bool = Field(default=False, description="Use emergency checkpoint with replicator service.")
  replicator_backup_interval_minutes: NonNegativeInt = Field(
      default=0, description="Interval for backing up local checkpoints."
  )


class CheckpointLoggingMiscConfig(BaseModel):
  """Miscellaneous checkpointing and logging configurations."""

  enable_single_replica_ckpt_restoring: bool = Field(
      default=False, description="Enable single replica checkpoint restoring."
  )
  enable_checkpoint_cloud_logger: bool = Field(default=False, description="Enable checkpoint cloud logger.")


class CheckpointConfig(BaseModel):
  """Container for all checkpoint related configurations."""

  loading: CheckpointLoadingConfig = Field(default_factory=CheckpointLoadingConfig)
  saving: CheckpointSavingConfig = Field(default_factory=CheckpointSavingConfig)
  storage: CheckpointStorageConfig = Field(default_factory=CheckpointStorageConfig)
  emergency: EmergencyCheckpointConfig = Field(default_factory=EmergencyCheckpointConfig)
  logging_misc: CheckpointLoggingMiscConfig = Field(default_factory=CheckpointLoggingMiscConfig)


class ModelIdentityConfig(BaseModel):
  """Model identity configurations."""

  model_name: str = Field(default="default", description="Name of the model configuration to use.")
  override_model_config: bool = Field(default=False, description="Allow overriding model params via CLI.")


class ModelCoreConfig(BaseModel):
  """Core model configurations like decoder type and scaling."""

  decoder_block: DecoderBlockType = Field(default=DecoderBlockType.LLAMA2, description="Type of decoder block to use.")
  global_parameter_scale: int = Field(default=1, description="Global parameter scale (power of 2).")
  weight_dtype: str = Field(default="float32", description="Data type for model weights.")
  normalization_layer_epsilon: float = Field(default=1.0e-05, description="Epsilon for normalization layers.")
  model_call_mode: ModelCallMode = Field(
      default=ModelCallMode.TRAIN, description="Mode for model execution ('train', 'inference')."
  )
  param_scan_axis: int = Field(default=1, description="Axis for parameter scanning if scan_layers is true.")
  inhomogeneous_layer_cycle_interval: int = Field(
      default=1, description="Cycle interval for inhomogeneous layers (e.g., Llama4)."
  )


class ModelArchitectureConfig(BaseModel):
  """Base model architecture parameters."""

  base_emb_dim: PositiveInt = Field(default=2048, description="Base embedding dimension.")
  base_num_query_heads: PositiveInt = Field(default=16, description="Base number of query heads.")
  base_num_kv_heads: PositiveInt = Field(default=16, description="Base number of key/value heads.")
  base_mlp_dim: PositiveInt = Field(default=7168, description="Base MLP dimension.")
  base_num_decoder_layers: PositiveInt = Field(default=16, description="Base number of decoder layers.")
  head_dim: Optional[PositiveInt] = Field(default=128, description="Dimension of each attention head.")


class ModelActivationConfig(BaseModel):
  """Configurations for activations, dropout, and logits behavior."""

  mlp_activations: List[str] = Field(default_factory=lambda: ["silu", "linear"], description="MLP activation functions.")
  dropout_rate: NonNegativeFloat = Field(default=0.0, description="Dropout rate.")
  logits_via_embedding: bool = Field(default=False, description="Compute logits via embedding layer transpose.")
  normalize_embedding_logits: bool = Field(
      default=True, description="Normalize pre-softmax logits if logits_via_embedding is true."
  )
  logits_dot_in_fp32: bool = Field(default=False, description="Use fp32 for logits dot product for stability.")
  cast_logits_to_fp32: bool = Field(default=True, description="Cast final logits to fp32.")
  float32_qk_product: bool = Field(default=False, description="Use fp32 for QK product in attention.")
  float32_logits: bool = Field(
      default=False, description="Use fp32 for attention logits before softmax."
  )  # Renamed from float32_logits_attn to avoid conflict
  activations_in_float32: bool = Field(default=False, description="Cast activations to float32 before nonlinearity.")


class ModelMiscBehaviorConfig(BaseModel):
  """Miscellaneous model behavior configurations."""

  record_internal_nn_metrics: NonNegativeInt = Field(default=0, description="Log internal NN metrics if > 0.")
  use_iota_embed: bool = Field(default=False, description="Use iota operator in Embed layer.")
  use_untrainable_positional_embedding: bool = Field(default=False, description="Use untrainable positional embeddings.")
  trainable_position_size: int = Field(default=-1, description="Enable GPT3-style trainable positional embeddings if > 0.")


class QuantizationConfig(BaseModel):
  """Quantization configurations."""

  dtype: str = Field(default="bfloat16", description="Data type for activations.")
  quantization: Optional[str] = Field(default="", description="Quantization type (e.g., 'int8', 'fp8').")
  matmul_precision: MatMulPrecision = Field(default=MatMulPrecision.DEFAULT, description="Precision for matmul operations.")
  replicate_quant_scale: bool = Field(default=False, description="Replicate quantization scale for 2D sharding.")
  quant_cfg_path: Optional[str] = Field(default="", description="Path to quantization config for 'intmp'.")
  quantize_kvcache: bool = Field(default=False, description="Quantize KV Cache values.")
  kv_quant_axis: str = Field(default="heads_and_dkv", description="Axis for KV cache quantization.")
  kv_quant_dtype: str = Field(default="int8", description="Data type for KV cache quantization.")
  quantization_local_shard_count: int = Field(default=-1, description="Local shard count for quantization range finding.")

  @validator("kv_quant_axis")
  def validate_kv_axis(cls, v, values):
    if values.get("quantize_kvcache") and v == "":
      raise ValueError("kv_quant_axis cannot be empty if quantize_kvcache is True")
    return v


class MoEConfig(BaseModel):
  """Mixture of Experts configurations."""

  num_experts: PositiveInt = Field(default=1, description="Number of experts.")
  num_experts_per_tok: PositiveInt = Field(default=1, description="Number of experts per token.")
  megablox: bool = Field(default=True, description="Use Megablox for MoE.")
  sparse_matmul: bool = Field(default=True, description="Use sparse matmul for MoE.")
  capacity_factor: float = Field(default=-1.0, description="Expert capacity factor for token dropping.")
  load_balance_loss_weight: NonNegativeFloat = Field(default=0.01, description="Weight for load balance loss.")
  use_random_routing: bool = Field(default=False, description="Use random routing for debug/test.")
  tile_batch_seq: Optional[PositiveInt] = Field(default=512, description="Tunable tiling dimension for Megablox.")
  tile_activation_dim: Optional[PositiveInt] = Field(default=1024, description="Tunable tiling dimension for Megablox.")
  tile_weight_dim: Optional[PositiveInt] = Field(default=1024, description="Tunable tiling dimension for Megablox.")


class DeepSeekMoEConfig(BaseModel):
  """DeepSeek-specific MoE configurations."""

  base_moe_mlp_dim: PositiveInt = Field(default=7168, description="Intermediate dimension at MoE layer for DeepSeek.")
  first_num_dense_layers: NonNegativeInt = Field(default=0, description="Number of initial dense layers for DeepSeek.")
  shared_experts: PositiveInt = Field(default=1, description="Number of shared experts for DeepSeek.")
  routed_scaling_factor: float = Field(default=1.0, description="Scaling factor for routing scores for DeepSeek.")
  routed_score_func: Optional[str] = Field(default="", description="Scoring function for routing for DeepSeek.")
  routed_bias: bool = Field(default=False, description="Add bias term for routing for DeepSeek.")
  n_routing_groups: int = Field(default=-1, description="Number of groups for routing for DeepSeek.")
  topk_routing_group: int = Field(default=-1, description="Number of top groups to route inputs for DeepSeek.")


class PipelineParallelConfig(BaseModel):
  """Pipeline parallelism configurations."""

  num_layers_per_pipeline_stage: PositiveInt = Field(default=1)
  num_pipeline_repeats: int = Field(default=-1, description="Auto-computed if -1.")
  pipeline_parallel_layers: int = Field(default=-1, description="All layers if -1.")
  num_pipeline_microbatches: int = Field(default=-1, description="Auto-computed if -1.")
  pipeline_delay_activation_forwarding: bool = Field(default=False)
  pipeline_fsdp_ag_once: bool = Field(default=False)
  scan_pipeline_iterations: bool = Field(default=True)
  scan_layers_per_stage: bool = Field(default=False)
  set_remat_policy_on_pipeline_iterations: bool = Field(default=True)
  set_remat_policy_on_layers_per_stage: bool = Field(default=False)


class RematConfig(BaseModel):
  """Rematerialization (checkpointing) policy configurations."""

  remat_policy: RematPolicy = Field(default=RematPolicy.FULL)
  decoder_layer_input: RematTensorConfigValue = Field(default=RematTensorConfigValue.DEVICE)
  context: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwi: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwi_0: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwi_1: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  mlpwo: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  query_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  key_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  value_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  qkv_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)
  out_proj: RematTensorConfigValue = Field(default=RematTensorConfigValue.REMAT)


class AttentionMechanismConfig(BaseModel):
  """Configuration for attention mechanism types and fusion."""

  attention: AttentionKernel = Field(default=AttentionKernel.AUTOSELECTED)
  attention_type: AttentionType = Field(default=AttentionType.GLOBAL)
  sliding_window_size: NonNegativeInt = Field(default=0)
  chunk_attn_window_size: NonNegativeInt = Field(default=0)
  fused_qkv: bool = Field(default=False)
  fused_mlp: bool = Field(default=False)


class AttentionBehaviorConfig(BaseModel):
  """Configuration for attention behavior like soft capping and norms."""

  attn_logits_soft_cap: NonNegativeFloat = Field(default=0.0)
  final_logits_soft_cap: NonNegativeFloat = Field(default=0.0)
  use_post_attn_norm: bool = Field(default=False)
  use_post_ffw_norm: bool = Field(default=False)
  stack_prefill_result_cache: bool = Field(default=False)
  enable_padding_causal_mask: bool = Field(default=True)
  use_ragged_attention: bool = Field(default=False)
  ragged_block_size: PositiveInt = Field(default=256)


class MLAConfig(BaseModel):
  """Multi-Head Latent Attention (MLA) configurations."""

  q_lora_rank: NonNegativeInt = Field(default=0)
  kv_lora_rank: NonNegativeInt = Field(default=512)
  qk_nope_head_dim: PositiveInt = Field(default=128)
  qk_rope_head_dim: PositiveInt = Field(default=64)
  v_head_dim: PositiveInt = Field(default=128)


class HardwareConfig(BaseModel):
  """Hardware and JAX distributed system configurations."""

  hardware: HardwareType = Field(default=HardwareType.TPU)
  num_slices: int = Field(default=-1, description="Auto-determined if -1.")
  jax_cache_dir: str = Field(default="~/jax_cache")
  jax_distributed_initialization_timeout: PositiveInt = Field(default=300)
  jax_debug_log_modules: Optional[str] = Field(default="")
  skip_jax_distributed_system: bool = Field(default=False)
  enable_single_controller: bool = Field(default=False)
  compiled_trainstep_file: Optional[str] = Field(default="")
  compile_topology: Optional[str] = Field(default="")
  compile_topology_num_slices: int = Field(default=-1)


class MeshConfig(BaseModel):
  """Mesh and sharding rule configurations."""

  mesh_axes: List[str] = Field(
      default_factory=lambda: [
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
  logical_axis_rules: List[List[Any]] = Field(
      default_factory=lambda: [["activation_batch", ["data", "fsdp", "fsdp_transpose", "expert"]]]
  )  # Simplified default
  data_sharding: List[List[str]] = Field(
      default_factory=lambda: [
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
      ]
  )
  input_data_sharding_logical_axes: List[str] = Field(
      default_factory=lambda: ["activation_embed_and_logits_batch", "activation_norm_length"]
  )
  sharding_tolerance: float = Field(default=0.02, ge=0.0, le=1.0)
  custom_mesh: Optional[str] = Field(default="")
  allow_split_physical_axes: bool = Field(default=False)
  optimize_mesh_for_tpu_v6e: bool = Field(default=False)
  context_parallel_load_balance: bool = Field(default=True)


class DCNParallelismConfig(BaseModel):
  """Data Center Network (inter-slice) parallelism configurations."""

  dcn_data_parallelism: int = Field(default=-1)
  dcn_fsdp_parallelism: int = Field(default=1)
  dcn_fsdp_transpose_parallelism: int = Field(default=1)
  dcn_sequence_parallelism: int = Field(default=1)
  dcn_context_parallelism: int = Field(default=1)
  dcn_context_autoregressive_parallelism: int = Field(default=1)
  dcn_tensor_parallelism: int = Field(default=1)
  dcn_tensor_transpose_parallelism: int = Field(default=1)
  dcn_tensor_sequence_parallelism: int = Field(default=1)
  dcn_pipeline_parallelism: int = Field(default=1)
  dcn_expert_parallelism: int = Field(default=1)
  dcn_autoregressive_parallelism: int = Field(default=1)


class ICIParallelismConfig(BaseModel):
  """Inter-Chip Interconnect (intra-slice) parallelism configurations."""

  ici_data_parallelism: int = Field(default=1)
  ici_fsdp_parallelism: int = Field(default=-1)
  ici_fsdp_transpose_parallelism: int = Field(default=1)
  ici_sequence_parallelism: int = Field(default=1)
  ici_context_parallelism: int = Field(default=1)
  ici_context_autoregressive_parallelism: int = Field(default=1)
  ici_tensor_parallelism: int = Field(default=1)
  ici_tensor_transpose_parallelism: int = Field(default=1)
  ici_tensor_sequence_parallelism: int = Field(default=1)
  ici_autoregressive_parallelism: int = Field(default=1)
  ici_pipeline_parallelism: int = Field(default=1)
  ici_expert_parallelism: int = Field(default=1)


class TokenizerConfig(BaseModel):
  """Tokenizer configurations."""

  vocab_size: PositiveInt = Field(default=32_000)
  tokenizer_path: str = Field(default="assets/tokenizer.llama2")
  tokenizer_type: TokenizerTypeEnum = Field(default=TokenizerTypeEnum.SENTENCEPIECE)
  use_chat_template: bool = Field(default=False)
  tokenize_train_data: bool = Field(default=True)
  tokenize_eval_data: bool = Field(default=True)
  add_bos: bool = Field(default=True)
  add_eos: bool = Field(default=True)


class BaseDatasetConfig(BaseModel):
  """Base dataset configurations, common across types."""

  per_device_batch_size: float = Field(default=12.0, gt=0.0)  # Can be < 1.0
  expansion_factor_real_data: int = Field(default=-1)
  eval_per_device_batch_size: NonNegativeFloat = Field(default=0.0)
  max_corpus_chars: Optional[PositiveInt] = Field(default=10_000_000)
  train_data_columns: List[str] = Field(default_factory=lambda: ["text"])
  eval_data_columns: List[str] = Field(default_factory=lambda: ["text"])
  packing: bool = Field(default=True)
  num_epoch: PositiveInt = Field(default=1)
  dataset_type: DatasetType = Field(default=DatasetType.TFDS)
  colocated_python_data_input: bool = Field(default=False)


class TFDSDatasetConfig(BaseModel):
  """TFDS-specific dataset configurations."""

  dataset_path: Optional[str] = Field(default="")
  dataset_name: str = Field(default="c4/en:3.0.1")
  eval_dataset_name: str = Field(default="c4/en:3.0.1")
  train_split: str = Field(default="train")
  eval_split: str = Field(default="validation")


class HFDatasetConfig(BaseModel):
  """HuggingFace dataset configurations."""

  hf_path: Optional[str] = Field(default="")
  hf_data_dir: Optional[str] = Field(default="")
  hf_train_files: Optional[str] = Field(default="")
  hf_eval_split: Optional[str] = Field(default="")
  hf_eval_files: Optional[str] = Field(default="")
  hf_access_token: Optional[str] = Field(default="")


class GrainDatasetConfig(BaseModel):
  """Grain dataset configurations."""

  grain_train_files: Optional[str] = Field(default="")
  grain_eval_files: Optional[str] = Field(default="")
  grain_file_type: GrainFileType = Field(default=GrainFileType.ARRAYRECORD)
  grain_worker_count: NonNegativeInt = Field(default=1)
  grain_worker_count_eval: NonNegativeInt = Field(default=1)


class DatasetNestingConfig(BaseModel):
  """Container for all dataset configurations."""

  base: BaseDatasetConfig = Field(default_factory=BaseDatasetConfig)
  tfds: Optional[TFDSDatasetConfig] = None
  hf: Optional[HFDatasetConfig] = None
  grain: Optional[GrainDatasetConfig] = None


class BasicTrainingConfig(BaseModel):
  """Basic training loop configurations."""

  reuse_example_batch: NonNegativeInt = Field(default=0)
  max_target_length: PositiveInt = Field(default=2048)
  max_prefill_predict_length: PositiveInt = Field(default=64)
  enable_dropout: bool = Field(default=True)
  enable_data_shuffling: bool = Field(default=True)
  data_shuffle_seed: NonNegativeInt = Field(default=0)
  init_weights_seed: NonNegativeInt = Field(default=0)
  gradient_clipping_threshold: NonNegativeFloat = Field(default=1.0)
  gradient_accumulation_steps: PositiveInt = Field(default=1)
  optimizer_memory_host_offload: bool = Field(default=False)
  parameter_memory_host_offload: bool = Field(default=False)
  scan_layers: bool = Field(default=True)


class LearningRateConfig(BaseModel):
  """Learning rate schedule configurations."""

  learning_rate: NonNegativeFloat = Field(default=3.0e-5)
  cosine_learning_rate_final_fraction: NonNegativeFloat = Field(default=0.1)
  warmup_steps_fraction: NonNegativeFloat = Field(default=0.1)
  learning_rate_schedule_steps: int = Field(default=-1, description="Auto-computed if -1.")


class PromptConfig(BaseModel):
  """Prompt configurations for generation/decode."""

  prompt: str = Field(default="I love to")
  load_from_prefill_dir: bool = Field(default=False)
  prefill_cache_dir: Optional[str] = Field(default="")
  autoregressive_decode_assert: Optional[str] = Field(default="")


class OptimizerConfig(BaseModel):
  """Optimizer configurations (primarily AdamW)."""

  opt_type: OptimizerType = Field(default=OptimizerType.ADAMW)
  adam_b1: float = Field(default=0.9)
  adam_b2: float = Field(default=0.95)
  adam_eps: float = Field(default=1.0e-8)
  adam_eps_root: NonNegativeFloat = Field(default=0.0)
  adam_weight_decay: NonNegativeFloat = Field(default=0.1)
  mu_dtype: Optional[str] = Field(default="", description="Data type for AdamW 'mu'. Inherits from weight_dtype if unset.")


class RoPEConfig(BaseModel):
  """Rotary Position Embedding configurations."""

  rope_type: RoPEType = Field(default=RoPEType.DEFAULT)
  rope_use_scale: bool = Field(default=True)
  rope_min_timescale: PositiveInt = Field(default=1)
  rope_max_timescale: PositiveInt = Field(default=10_000)
  local_rope_max_timescale: int = Field(default=-1, description="Use rope_max_timescale if -1.")


class YarnRoPEConfig(BaseModel):
  """Yarn RoPE specific configurations."""

  max_position_embeddings: PositiveInt = Field(default=163840)
  original_max_position_embeddings: PositiveInt = Field(default=4096)
  rope_factor: PositiveInt = Field(default=40)
  beta_fast: PositiveInt = Field(default=32)
  beta_slow: PositiveInt = Field(default=1)
  mscale: NonNegativeFloat = Field(default=1.0)


class DecodeAlgoConfig(BaseModel):
  """Configurations for decoding algorithms."""

  decode_sampling_strategy: SamplingStrategy = Field(default=SamplingStrategy.GREEDY)
  decode_sampling_nucleus_p: float = Field(default=-1.0, ge=-1.0, le=1.0, description="Allow -1 for 'not set'.")
  decode_sampling_top_k: NonNegativeInt = Field(default=0)
  decode_sampling_temperature: NonNegativeFloat = Field(default=1.0)


class EvalRunConfig(BaseModel):
  """Evaluation run configurations."""

  eval_interval: int = Field(default=-1)
  eval_steps: int = Field(default=-1)
  target_eval_loss: NonNegativeFloat = Field(default=0.0)


class ProfilerRunConfig(BaseModel):
  """Profiler configurations."""

  profiler: ProfilerType = Field(default=ProfilerType.NONE)
  upload_all_profiler_results: bool = Field(default=False)
  skip_first_n_steps_for_profiler: NonNegativeInt = Field(default=1)
  profiler_steps: PositiveInt = Field(default=5)
  profile_cleanly: bool = Field(default=True)
  profile_periodically_period: int = Field(default=-1)


class HloDumpRunConfig(BaseModel):
  """HLO dump configurations."""

  dump_hlo: bool = Field(default=False)
  dump_step: int = Field(default=-1)
  dump_hlo_local_dir: str = Field(default="/tmp/xla_dump/")
  dump_hlo_delete_local_after: bool = Field(default=True)
  dump_hlo_gcs_dir: Optional[str] = Field(default="")
  dump_hlo_module_name: str = Field(default="jit_train_step")
  dump_hlo_xla_flags: Optional[str] = Field(default="")
  dump_hlo_upload_all: bool = Field(default=False)


class KVLayoutRunConfig(BaseModel):
  """KV Cache and compute layout configurations."""

  prefill_cache_axis_order: str = Field(default="1,2,0,3")
  ar_cache_axis_order: str = Field(default="1,2,0,3")
  compute_axis_order: str = Field(default="0,1,2,3")
  reshape_q: bool = Field(default=False)

  @validator("compute_axis_order")
  def validate_compute_layout(cls, v):
    if v not in ("0,1,2,3", "0,2,1,3"):
      raise ValueError("compute_axis_order must be '0,1,2,3' or '0,2,1,3'")
    return v


class MaxEngineRunConfig(BaseModel):
  """MaxEngine server specific configurations."""

  prometheus_port: NonNegativeInt = Field(default=0)
  enable_jax_profiler: bool = Field(default=False)
  jax_profiler_port: PositiveInt = Field(default=9999)
  inference_server: InferenceServerType = Field(default=InferenceServerType.MAXTEXT_INTERLEAVED)
  prefill_slice: Optional[str] = Field(default="v5e-16", description="Slice for prefill in disaggregated server.")
  generate_slice: Optional[str] = Field(default="v5e-16", description="Slice for generation in disaggregated server.")


class SplashAttentionRunConfig(BaseModel):
  """Splash attention block size configurations."""

  sa_block_q: PositiveInt = Field(default=512)
  sa_block_kv: PositiveInt = Field(default=512)
  sa_block_kv_compute: PositiveInt = Field(default=512)
  sa_block_q_dkv: PositiveInt = Field(default=512)
  sa_block_kv_dkv: PositiveInt = Field(default=512)
  sa_block_kv_dkv_compute: PositiveInt = Field(default=512)
  sa_block_q_dq: PositiveInt = Field(default=512)
  sa_block_kv_dq: PositiveInt = Field(default=512)
  sa_use_fused_bwd_kernel: bool = Field(default=False)
  sa_q_layout: str = Field(default="HEAD_DIM_MINOR")
  sa_k_layout: str = Field(default="HEAD_DIM_MINOR")
  sa_v_layout: str = Field(default="HEAD_DIM_MINOR")


class PagedAttentionRunConfig(BaseModel):
  """Paged attention configurations."""

  pagedattn_num_pages: PositiveInt = Field(default=64)
  pagedattn_tokens_per_page: PositiveInt = Field(default=32)
  pagedattn_pages_per_compute_block: PositiveInt = Field(default=4)
  pagedattn_max_pages_per_group: int = Field(default=-1, description="Auto-computed if -1.")


class ChunkedPrefillRunConfig(BaseModel):
  """Chunked prefill configurations."""

  prefill_chunk_size: PositiveInt = Field(default=256)
  use_chunked_prefill: bool = Field(default=False)


class PrefixCachingRunConfig(BaseModel):
  """Prefix caching configurations for JetStream."""

  enable_prefix_caching: bool = Field(default=False)
  prefix_caching_hbm_byte: PositiveInt = Field(default=10_000_000_000)  # 10 GB
  prefix_caching_dram_byte: PositiveInt = Field(default=100_000_000_000)  # 100 GB


class Llama4SpecificConfig(BaseModel):
  """Llama4-specific configurations from base.yml."""

  use_qk_norm: bool = Field(default=False)
  nope_layer_interval: int = Field(default=-1)
  interleave_moe_layer_step: PositiveInt = Field(default=1)
  temperature_tuning: bool = Field(default=False)


class MultimodalRunConfig(BaseModel):
  """Multimodal configurations."""

  use_multimodal: bool = Field(default=False)
  freeze_vision_encoder_params: bool = Field(default=True)
  dtype_mm: str = Field(default="float32", description="Data type for ViT.")
  remat_policy_for_vit: RematPolicy = Field(default=RematPolicy.MINIMAL)
  image_size_for_vit: PositiveInt = Field(default=896, description="Default for Gemma3.")
  image_path: Optional[str] = Field(default="")


class Llama4VitRunConfig(BaseModel):
  """Llama4-specific Vision Transformer configurations from base.yml."""

  hidden_size_for_vit: PositiveInt = Field(default=1408)
  intermediate_size_for_vit: PositiveInt = Field(default=5632)
  num_attention_heads_for_vit: PositiveInt = Field(default=16)
  num_channels_for_vit: PositiveInt = Field(default=3)
  patch_size_for_vit: PositiveInt = Field(default=14)
  num_hidden_layers_for_vit: PositiveInt = Field(default=34)
  projector_input_dim_for_vit: PositiveInt = Field(default=4096)
  projector_output_dim_for_vit: PositiveInt = Field(default=4096)
  rope_theta_for_vit: PositiveInt = Field(default=10000)
  vision_output_dim_for_vit: PositiveInt = Field(default=4096)
  pixel_shuffle_ratio_for_vit: float = Field(default=0.5, gt=0.0, lt=1.0)
  projector_dropout_for_vit: NonNegativeFloat = Field(default=0.0)


class DPOSpecificConfig(BaseModel):
  """DPO-specific configurations."""

  use_dpo: bool = Field(default=False)
  dpo_label_smoothing: NonNegativeFloat = Field(default=0.0)
  dpo_beta: NonNegativeFloat = Field(default=0.1)


class SFTSpecificConfig(BaseModel):
  """SFT-specific configurations."""

  use_sft: bool = Field(default=False)
  sft_train_on_completion_only: bool = Field(default=False)


class StackTraceConfig(BaseModel):
  """Stack trace collection configurations."""

  collect_stack_trace: bool = Field(default=False)
  stack_trace_to_cloud: bool = Field(default=False)
  stack_trace_interval_seconds: PositiveInt = Field(default=600)


class GCPWorkloadMonitorConfig(BaseModel):
  """GCP workload monitoring configurations."""

  report_heartbeat_metric_for_gcp_monitoring: bool = Field(default=False)
  heartbeat_reporting_interval_in_seconds: PositiveInt = Field(default=5)
  report_performance_metric_for_gcp_monitoring: bool = Field(default=False)


class InferenceMicrobenchmarkConfig(BaseModel):
  """Inference microbenchmark configurations."""

  inference_microbenchmark_prefill_lengths: str = Field(default="64,128,256,512,1024")
  inference_microbenchmark_stages: str = Field(default="prefill,generate")
  inference_microbenchmark_loop_iters: PositiveInt = Field(default=10)
  inference_microbenchmark_log_file_path: Optional[str] = Field(default="")
  inference_microbenchmark_num_samples: List[PositiveInt] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
  inference_metadata_file: Optional[str] = Field(default="")  # Path to JSON, not in base.yml
  inference_benchmark_test: bool = Field(default=False)
  enable_model_warmup: bool = Field(default=False)
  enable_llm_inference_pool: bool = Field(default=False)
  multi_sampling: bool = Field(default=False)
  return_log_prob: bool = Field(default=False)


class MaxTextConfig(BaseModel):
  """Top-level configuration model for MaxText, derived from YAML files."""

  run_config: RunConfig = Field(default_factory=RunConfig)
  checkpoint_config: CheckpointConfig = Field(default_factory=CheckpointConfig)

  model_identity_config: ModelIdentityConfig = Field(default_factory=ModelIdentityConfig)
  model_core_config: ModelCoreConfig = Field(default_factory=ModelCoreConfig)
  model_architecture_config: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)
  model_activation_config: ModelActivationConfig = Field(default_factory=ModelActivationConfig)
  model_misc_behavior_config: ModelMiscBehaviorConfig = Field(default_factory=ModelMiscBehaviorConfig)

  quantization_config: QuantizationConfig = Field(default_factory=QuantizationConfig)
  moe_config: MoEConfig = Field(default_factory=MoEConfig)
  deepseek_moe_config: Optional[DeepSeekMoEConfig] = None
  pipeline_parallel_config: PipelineParallelConfig = Field(default_factory=PipelineParallelConfig)
  remat_config: RematConfig = Field(default_factory=RematConfig)

  attention_mechanism_config: AttentionMechanismConfig = Field(default_factory=AttentionMechanismConfig)
  attention_behavior_config: AttentionBehaviorConfig = Field(default_factory=AttentionBehaviorConfig)

  mla_config: Optional[MLAConfig] = None
  hardware_config: HardwareConfig = Field(default_factory=HardwareConfig)
  parallelism_config: ParallelismConfig = Field(default_factory=ParallelismConfig)
  tokenizer_config: TokenizerConfig = Field(default_factory=TokenizerConfig)
  dataset_nesting_config: DatasetNestingConfig = Field(default_factory=DatasetNestingConfig)

  basic_training_config: BasicTrainingConfig = Field(default_factory=BasicTrainingConfig)
  learning_rate_config: LearningRateConfig = Field(default_factory=LearningRateConfig)
  prompt_config: PromptConfig = Field(default_factory=PromptConfig)

  optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
  rope_config: RoPEConfig = Field(default_factory=RoPEConfig)
  yarn_rope_config: Optional[YarnRoPEConfig] = None
  decode_algo_config: DecodeAlgoConfig = Field(default_factory=DecodeAlgoConfig)
  eval_run_config: EvalRunConfig = Field(default_factory=EvalRunConfig)
  profiler_run_config: ProfilerRunConfig = Field(default_factory=ProfilerRunConfig)
  hlo_dump_run_config: HloDumpRunConfig = Field(default_factory=HloDumpRunConfig)
  kv_layout_run_config: KVLayoutRunConfig = Field(default_factory=KVLayoutRunConfig)
  max_engine_run_config: MaxEngineRunConfig = Field(default_factory=MaxEngineRunConfig)
  splash_attention_run_config: SplashAttentionRunConfig = Field(default_factory=SplashAttentionRunConfig)
  paged_attention_run_config: PagedAttentionRunConfig = Field(default_factory=PagedAttentionRunConfig)
  chunked_prefill_run_config: ChunkedPrefillRunConfig = Field(default_factory=ChunkedPrefillRunConfig)
  prefix_caching_run_config: PrefixCachingRunConfig = Field(default_factory=PrefixCachingRunConfig)
  llama4_specific_config: Optional[Llama4SpecificConfig] = None
  multimodal_run_config: MultimodalRunConfig = Field(default_factory=MultimodalRunConfig)
  llama4_vit_run_config: Optional[Llama4VitRunConfig] = None
  dpo_specific_config: Optional[DPOSpecificConfig] = None
  sft_specific_config: Optional[SFTSpecificConfig] = None
  stack_trace_config: StackTraceConfig = Field(default_factory=StackTraceConfig)
  gcp_workload_monitor_config: GCPWorkloadMonitorConfig = Field(default_factory=GCPWorkloadMonitorConfig)
  inference_microbenchmark_config: InferenceMicrobenchmarkConfig = Field(default_factory=InferenceMicrobenchmarkConfig)

  reuse_example_batch: NonNegativeInt = Field(default=0, description="For testing TPU performance, repeat the same batch.")
  max_checkify: bool = Field(default=False, description="Enable extra checks using jax.checkify (affects performance).")

  # Fields from `dpo.yml` etc. are typically booleans or simple types already covered,
  # or they override values in the sub-configs. For example, `dpo.yml` has `use_dpo: true`.
  # This would be loaded by setting `maxtext_config.dpo_specific_config.use_dpo = True`.
  # Model specific parameters from `configs/models/*.yml` would populate the `model_architecture_config`, etc.

  class Config:
    extra = "forbid"
    validate_assignment = True
