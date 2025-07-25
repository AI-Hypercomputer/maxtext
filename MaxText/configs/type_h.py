# MaxText/configs/type_h.py
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

This module defines the Pydantic models for MaxText configuration.
It uses a two-step process:
1. An internal, flat `_FlatConfig` validates the flat base.yml.
2. A conversion function `build_config` creates the user-facing, nested `MaxTextConfig`
   for improved code readability and maintainability.
"""

from enum import Enum
from typing import List, Optional, Any
import os

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    NonNegativeFloat,
    computed_field,
    ConfigDict,
)


# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------
class DecoderBlockType(str, Enum):
    DEFAULT, LLAMA2, MISTRAL, MIXTRAL = "default", "llama2", "mistral", "mixtral"
    DEEPSEEK, GEMMA, GEMMA2, GEMMA3 = "deepseek", "gemma", "gemma2", "gemma3"
    GPT3, SIMPLE, SIMPLE_MLP, LLAMA4 = "gpt3", "simple", "simple_mlp", "llama4"


class AttentionType(str, Enum):
    GLOBAL, LOCAL_SLIDING, CHUNK, MLA, FULL = (
        "global",
        "local_sliding",
        "chunk",
        "mla",
        "full",
    )


class OptimizerType(str, Enum):
    ADAMW, ADAM_PAX, SGD = "adamw", "adam_pax", "sgd"


class MatMulPrecision(str, Enum):
    DEFAULT, HIGH, HIGHEST = "default", "high", "highest"


class DatasetType(str, Enum):
    SYNTHETIC, HF, GRAIN, TFDS, C4_MLPERF = (
        "synthetic",
        "hf",
        "grain",
        "tfds",
        "c4_mlperf",
    )


class GrainFileType(str, Enum):
    ARRAYRECORD, PARQUET = "arrayrecord", "parquet"


class HardwareType(str, Enum):
    TPU, GPU, GPU_MULTIPROCESS, CPU = "tpu", "gpu", "gpu_multiprocess", "cpu"


class ProfilerType(str, Enum):
    NONE, XPLANE, NSYS = "", "xplane", "nsys"


class AttentionKernel(str, Enum):
    AUTOSELECTED, DOT_PRODUCT, FLASH = "autoselected", "dot_product", "flash"
    CUDNN_FLASH_TE, CUDNN_FLASH_JAX, PAGED = (
        "cudnn_flash_te",
        "cudnn_flash_jax",
        "paged",
    )


class RematPolicy(str, Enum):
    MINIMAL, SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP = (
        "minimal",
        "save_dot_with_context_except_mlp",
    )
    SAVE_DOT_EXCEPT_MLPWI, SAVE_DOT_EXCEPT_MLP = (
        "save_dot_except_mlpwi",
        "save_dot_except_mlp",
    )
    SAVE_QKV_PROJ, QKV_PROJ_OFFLOADED = "save_qkv_proj", "qkv_proj_offloaded"
    CUSTOM, MINIMAL_OFFLOADED, SAVE_OUT_PROJ, FULL, MINIMAL_FLASH = (
        "custom",
        "minimal_offloaded",
        "save_out_proj",
        "full",
        "minimal_flash",
    )


class RematTensorConfigValue(str, Enum):
    REMAT, DEVICE, OFFLOAD = "remat", "device", "offload"


class ModelCallMode(str, Enum):
    TRAIN, INFERENCE, AUTOREGRESSIVE, PREFILL = (
        "train",
        "inference",
        "autoregressive",
        "prefill",
    )


class SamplingStrategy(str, Enum):
    GREEDY, WEIGHTED, NUCLEUS, TOPK = "greedy", "weighted", "nucleus", "topk"


class RoPEType(str, Enum):
    DEFAULT, LLAMA3_1, YARN = "default", "llama3.1", "yarn"


class TokenizerTypeEnum(str, Enum):
    SENTENCEPIECE, TIKTOKEN, HUGGINGFACE = "sentencepiece", "tiktoken", "huggingface"


class InferenceServerType(str, Enum):
    MAXTEXT_INTERLEAVED, EXPERIMENTAL_MAXTEXT_DISAGGREGATED = (
        "MaxtextInterleavedServer",
        "ExperimentalMaxtextDisaggregatedServer",
    )


# -----------------------------------------------------------------------------
# Nested, Readable Configuration Models (User-Facing)
# -----------------------------------------------------------------------------


class PathConfig(BaseModel):
    """Configuration for various important file system paths."""

    base_output_directory: str = Field(
        description="Base directory for experiment outputs."
    )
    run_name: str = Field(description="User-defined name for the run, used in paths.")

    @computed_field()
    @property
    def checkpoint_dir(self) -> str:
        path = (
            os.path.join(self.base_output_directory, self.run_name, "checkpoints/")
            if self.base_output_directory and self.run_name
            else "default_checkpoint_dir/"
        )
        if self.run_name == "test" and self.base_output_directory == "":
            path = "test/checkpoints/"
        return path

    @computed_field()
    @property
    def metrics_dir(self) -> str:
        path = (
            os.path.join(self.base_output_directory, self.run_name, "metrics/")
            if self.base_output_directory and self.run_name
            else "default_metrics_dir/"
        )
        if self.run_name == "test" and self.base_output_directory == "":
            path = "test/metrics/"
        return path

    @computed_field()
    @property
    def tensorboard_dir(self) -> str:
        path = (
            os.path.join(self.base_output_directory, self.run_name, "tensorboard/")
            if self.base_output_directory and self.run_name
            else "default_tensorboard_dir/"
        )
        if self.run_name == "test" and self.base_output_directory == "":
            path = "test/tensorboard/"
        return path


class GeneralRunSetting(BaseModel):
    """General settings for the execution of a training or evaluation run."""

    log_period: PositiveInt = Field(
        description="Frequency (steps) for TensorBoard/metrics logging."
    )
    steps: int = Field(
        description="Total training steps. -1 uses learning_rate_schedule_steps."
    )
    log_config: bool = Field(description="Print the final configuration at startup.")
    enable_tensorboard: bool = Field(description="Enable TensorBoard logging.")
    metrics_file: Optional[str] = Field(
        description="Path to local file for scalar metrics. Empty disables local logging."
    )
    gcs_metrics: bool = Field(description="Save scalar metrics (loss, TFLOPS) to GCS.")
    save_config_to_gcs: bool = Field(description="Save final config file to GCS.")
    max_checkify: bool = Field(
        description="Enable jax.checkify for debugging; affects performance."
    )
    rep: NonNegativeInt = Field(
        description="For TPU perf testing, repeats execution of the same batch N times."
    )


class CheckpointSetting(BaseModel):
    """Configuration for loading and saving model checkpoints."""

    load_parameters_path: Optional[str] = Field(
        description="Path to load parameters-only checkpoint from."
    )
    lora_input_adapters_path: Optional[str] = Field(
        description="GCS path to parent directory of LoRA adapters."
    )
    load_full_state_path: Optional[str] = Field(
        description="Path to load full training state from."
    )
    checkpoint_is_quantized: bool = Field(
        description="Indicates if loading a quantized (AQT) checkpoint."
    )
    enable_checkpointing: bool = Field(description="Enable checkpoint saving.")
    async_checkpointing: bool = Field(
        description="Use asynchronous checkpointing if enabled."
    )
    checkpoint_period: NonNegativeInt = Field(
        description="Frequency (steps) for saving checkpoints."
    )
    save_quantized_params_path: Optional[str] = Field(
        description="Path to save on-the-fly quantized model params (AQT)."
    )
    force_unroll: bool = Field(
        description="Force unroll loop for param-only checkpoint generation."
    )


class ModelArchitecture(BaseModel):
    """Core architectural parameters defining the model's size and structure."""

    model_name: str = Field(
        description="Identifier for model architecture (e.g., 'llama2-7b')."
    )
    decoder_block: DecoderBlockType = Field(description="Type of decoder block.")
    emb_dim: PositiveInt = Field(description="Core embedding dimension.")
    mlp_dim: PositiveInt = Field(description="Intermediate dimension of MLP layers.")
    num_decoder_layers: PositiveInt = Field(
        description="Total number of decoder layers."
    )
    num_query_heads: PositiveInt = Field(
        description="Number of attention heads for queries."
    )
    num_kv_heads: PositiveInt = Field(
        description="Number of heads for keys/values (for GQA/MQA)."
    )
    head_dim: Optional[PositiveInt] = Field(
        description="Dimension of each attention head."
    )
    global_parameter_scale: int = Field(
        description="Global scaling factor for model parameters."
    )


class AttentionSetting(BaseModel):
    """Configuration for the attention mechanism."""

    attention: AttentionKernel = Field(
        description="Specific attention kernel algorithm to use."
    )
    attention_type: AttentionType = Field(description="Variant of attention mechanism.")
    sliding_window_size: NonNegativeInt = Field(
        description="Window size for local sliding window attention."
    )
    chunk_attn_window_size: NonNegativeInt = Field(
        description="Window size for chunked attention."
    )
    fused_qkv: bool = Field(
        description="Fuse Query, Key, and Value projection matmuls into a single operation."
    )
    fused_mlp: bool = Field(
        description="Fuse MLP layers if applicable by the decoder block."
    )
    attn_logits_soft_cap: Optional[NonNegativeFloat] = Field(
        description="Soft cap value for attention logits."
    )
    final_logits_soft_cap: Optional[NonNegativeFloat] = Field(
        description="Soft cap value for final model output logits."
    )
    use_post_attn_norm: bool = Field(
        description="Apply a normalization layer after the attention block."
    )
    use_post_ffw_norm: bool = Field(
        description="Apply a normalization layer after the feed-forward/MLP block."
    )


class MlaSetting(BaseModel):
    """Multi-Head Latent Attention (MLA) architectural parameters."""

    q_lora_rank: NonNegativeInt = Field(
        description="Rank for LoRA applied to query projections in MLA."
    )
    kv_lora_rank: NonNegativeInt = Field(
        description="Rank for LoRA applied to key/value projections in MLA."
    )
    qk_nope_head_dim: NonNegativeInt = Field(
        description="Dimension for the NoPE part of Query/Key projections."
    )
    qk_rope_head_dim: PositiveInt = Field(
        description="Dimension for the RoPE part of Query/Key projections."
    )
    v_head_dim: PositiveInt = Field(
        description="Dimension for value projections per head in MLA."
    )
    mla_naive_kvcache: bool = Field(
        description="Use a naive (simpler) KV cache implementation for MLA."
    )


class HardwareAndParallelismSetting(BaseModel):
    """Configurations for hardware, parallelism, and device mesh."""

    hardware: HardwareType = Field(description="Target hardware (tpu, gpu, cpu).")
    num_slices: int = Field(description="Number of TPU slices. -1 for auto.")
    ici_fsdp_parallelism: int = Field(description="FSDP parallelism within an ICI.")
    dcn_data_parallelism: int = Field(description="Data parallelism across the DCN.")
    mesh_axes: List[str] = Field(description="Names of axes in the device mesh.")
    logical_axis_rules: List[List[Any]] = Field(
        description="Rules for sharding tensors."
    )


class TrainingSetting(BaseModel):
    """Configurations for the training loop process, optimization, and data."""

    per_device_batch_size: float = Field(
        description="Batch size per device for training."
    )
    eval_per_device_batch_size: NonNegativeFloat = Field(
        description="Batch size per device for evaluation."
    )
    max_target_length: PositiveInt = Field(
        description="Maximum sequence length for model processing."
    )
    max_prefill_predict_length: PositiveInt = Field(
        description="Max length for prefill stage in autoregression."
    )
    learning_rate: NonNegativeFloat = Field(
        description="Peak learning rate after warmup."
    )
    learning_rate_schedule_steps: int = Field(description="Total steps in LR schedule.")
    warmup_steps_fraction: NonNegativeFloat = Field(
        le=1.0, description="Fraction of schedule for warmup."
    )
    opt_type: OptimizerType = Field(description="Optimizer algorithm to use.")
    adam_b1: float = Field(gt=0.0, lt=1.0, description="Beta1 for AdamW optimizer.")
    adam_b2: float = Field(gt=0.0, lt=1.0, description="Beta2 for AdamW optimizer.")
    adam_weight_decay: NonNegativeFloat = Field(
        description="Weight decay for AdamW optimizer."
    )
    mu_dtype: Optional[str] = Field(
        description="Data type for AdamW 'mu' (1st moment)."
    )


class _FlatConfig(BaseModel):
    """An internal, flat Pydantic model for validating the flat `base.yml` file."""

    # All fields from base.yml must be here. Aliases are used for old keys.
    run_name: str
    log_period: PositiveInt
    steps: int
    log_config: bool
    enable_tensorboard: bool
    metrics_file: Optional[str]
    gcs_metrics: bool
    save_config_to_gcs: bool
    max_checkify: bool
    rep: NonNegativeInt
    base_output_directory: str
    tokenizer_path: str
    prefill_cache_dir: Optional[str]
    compiled_trainstep_file: Optional[str]
    quant_cfg_path: Optional[str]
    use_vertex_tensorboard: bool
    vertex_tensorboard_project: Optional[str]
    vertex_tensorboard_region: Optional[str]
    load_parameters_path: Optional[str]
    lora_input_adapters_path: Optional[str]
    load_full_state_path: Optional[str]
    checkpoint_is_quantized: bool
    enable_checkpointing: bool
    async_checkpointing: bool
    checkpoint_period: NonNegativeInt
    force_unroll: bool
    save_quantized_params_path: Optional[str]
    checkpoint_storage_target_data_file_size_bytes: int
    checkpoint_storage_use_ocdbt: bool
    checkpoint_storage_use_zarr3: bool
    checkpoint_storage_concurrent_gb: int
    enable_emergency_checkpoint: bool
    local_checkpoint_directory: Optional[str]
    local_checkpoint_period: NonNegativeInt
    use_replicator_service: bool
    replicator_backup_interval_minutes: NonNegativeInt
    enable_single_replica_ckpt_restoring: bool
    enable_checkpoint_cloud_logger: bool
    model_name: str
    override_model_config: bool
    decoder_block: DecoderBlockType
    emb_dim: PositiveInt = Field(alias="base_emb_dim")
    mlp_dim: PositiveInt = Field(alias="base_mlp_dim")
    num_decoder_layers: PositiveInt = Field(alias="base_num_decoder_layers")
    num_query_heads: PositiveInt = Field(alias="base_num_query_heads")
    num_kv_heads: PositiveInt = Field(alias="base_num_kv_heads")
    head_dim: Optional[PositiveInt]
    global_parameter_scale: int
    base_moe_mlp_dim: Optional[PositiveInt]
    weight_dtype: str
    normalization_layer_epsilon: float
    model_call_mode: ModelCallMode
    param_scan_axis: int
    inhomogeneous_layer_cycle_interval: int
    use_iota_embed: bool
    use_untrainable_positional_embedding: bool
    trainable_position_size: int
    mlp_activations: List[str]
    dropout_rate: NonNegativeFloat
    logits_via_embedding: bool
    normalize_embedding_logits: bool
    logits_dot_in_fp32: bool
    cast_logits_to_fp32: bool
    float32_qk_product: bool
    float32_logits: bool
    activations_in_float32: bool
    dtype: str
    quantization: Optional[str]
    matmul_precision: MatMulPrecision
    replicate_quant_scale: bool
    quantize_kvcache: bool
    kv_quant_axis: str
    kv_quant_dtype: str
    quantization_local_shard_count: int
    num_experts: PositiveInt
    num_experts_per_tok: PositiveInt
    megablox: bool
    sparse_matmul: bool
    capacity_factor: float
    load_balance_loss_weight: NonNegativeFloat
    use_random_routing: bool
    moe_mlp_dim: Optional[PositiveInt]
    tile_batch_seq: Optional[PositiveInt]
    tile_activation_dim: Optional[PositiveInt]
    tile_weight_dim: Optional[PositiveInt]
    first_num_dense_layers: NonNegativeInt
    shared_experts: PositiveInt
    routed_scaling_factor: float
    routed_score_func: Optional[str]
    routed_bias: bool
    n_routing_groups: int
    topk_routing_group: int
    num_layers_per_pipeline_stage: PositiveInt
    num_pipeline_repeats: int
    pipeline_parallel_layers: int
    num_pipeline_microbatches: int
    pipeline_delay_activation_forwarding: bool
    pipeline_fsdp_ag_once: bool
    scan_pipeline_iterations: bool
    scan_layers_per_stage: bool
    set_remat_policy_on_pipeline_iterations: bool
    set_remat_policy_on_layers_per_stage: bool
    using_pipeline_parallelism: bool
    remat_policy: RematPolicy
    decoder_layer_input: RematTensorConfigValue
    context: RematTensorConfigValue
    mlpwi: RematTensorConfigValue
    mlpwi_0: RematTensorConfigValue
    mlpwi_1: RematTensorConfigValue
    mlpwo: RematTensorConfigValue
    query_proj: RematTensorConfigValue
    key_proj: RematTensorConfigValue
    value_proj: RematTensorConfigValue
    qkv_proj: RematTensorConfigValue
    out_proj: RematTensorConfigValue
    attention: AttentionKernel
    attention_type: AttentionType
    sliding_window_size: NonNegativeInt
    chunk_attn_window_size: NonNegativeInt
    mla_naive_kvcache: bool
    fused_qkv: bool
    fused_mlp: bool
    attn_logits_soft_cap: Optional[NonNegativeFloat]
    final_logits_soft_cap: Optional[NonNegativeFloat]
    use_post_attn_norm: bool
    use_post_ffw_norm: bool
    stack_prefill_result_cache: bool
    enable_padding_causal_mask: bool
    use_ragged_attention: bool
    ragged_block_size: PositiveInt
    q_lora_rank: NonNegativeInt
    kv_lora_rank: NonNegativeInt
    qk_nope_head_dim: PositiveInt
    qk_rope_head_dim: PositiveInt
    v_head_dim: PositiveInt
    hardware: HardwareType
    num_slices: int
    jax_cache_dir: str
    jax_distributed_initialization_timeout: PositiveInt
    jax_debug_log_modules: Optional[str]
    skip_jax_distributed_system: bool
    enable_single_controller: bool
    compiled_trainstep_file: Optional[str]
    compile_topology: Optional[str]
    compile_topology_num_slices: int
    mesh_axes: List[str]
    logical_axis_rules: List[List[Any]]
    data_sharding: List[List[str]]
    input_data_sharding_logical_axes: List[str]
    sharding_tolerance: float
    custom_mesh: Optional[str]
    allow_split_physical_axes: bool
    optimize_mesh_for_tpu_v6e: bool
    context_parallel_load_balance: bool
    dcn_data_parallelism: int
    dcn_fsdp_parallelism: int
    dcn_fsdp_transpose_parallelism: int
    dcn_sequence_parallelism: int
    dcn_context_parallelism: int
    dcn_context_autoregressive_parallelism: int
    dcn_tensor_parallelism: int
    dcn_tensor_transpose_parallelism: int
    dcn_tensor_sequence_parallelism: int
    dcn_pipeline_parallelism: int
    dcn_expert_parallelism: int
    dcn_autoregressive_parallelism: int
    ici_data_parallelism: int
    ici_fsdp_parallelism: int
    ici_fsdp_transpose_parallelism: int
    ici_sequence_parallelism: int
    ici_context_parallelism: int
    ici_context_autoregressive_parallelism: int
    ici_tensor_parallelism: int
    ici_tensor_transpose_parallelism: int
    ici_tensor_sequence_parallelism: int
    ici_autoregressive_parallelism: int
    ici_pipeline_parallelism: int
    ici_expert_parallelism: int
    vocab_size: PositiveInt
    tokenizer_type: TokenizerTypeEnum
    use_chat_template: bool
    tokenize_train_data: bool
    tokenize_eval_data: bool
    add_bos: bool
    add_eos: bool
    per_device_batch_size: float
    expansion_factor_real_data: int
    eval_per_device_batch_size: NonNegativeFloat
    max_corpus_chars: Optional[PositiveInt]
    train_data_columns: List[str]
    eval_data_columns: List[str]
    packing: bool
    num_epoch: PositiveInt
    dataset_type: DatasetType
    dataset_path: Optional[str]
    dataset_name: str
    eval_dataset_name: str
    train_split: str
    eval_split: str
    hf_path: Optional[str]
    hf_data_dir: Optional[str]
    hf_train_files: Optional[str]
    hf_eval_split: Optional[str]
    hf_eval_files: Optional[str]
    hf_access_token: Optional[str]
    grain_train_files: Optional[str]
    grain_eval_files: Optional[str]
    grain_file_type: GrainFileType
    grain_worker_count: NonNegativeInt
    grain_worker_count_eval: NonNegativeInt
    colocated_python_data_input: bool
    use_dpo: bool
    dpo_label_smoothing: NonNegativeFloat
    dpo_beta: NonNegativeFloat
    use_sft: bool
    sft_train_on_completion_only: bool
    max_target_length: PositiveInt
    max_prefill_predict_length: PositiveInt
    enable_dropout: bool
    enable_data_shuffling: bool
    data_shuffle_seed: NonNegativeInt
    init_weights_seed: NonNegativeInt
    gradient_clipping_threshold: NonNegativeFloat
    gradient_accumulation_steps: PositiveInt
    scan_layers: bool
    learning_rate: NonNegativeFloat
    cosine_learning_rate_final_fraction: NonNegativeFloat
    warmup_steps_fraction: NonNegativeFloat
    learning_rate_schedule_steps: int
    opt_type: OptimizerType
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_eps_root: NonNegativeFloat
    adam_weight_decay: NonNegativeFloat
    mu_dtype: Optional[str]
    prompt: str
    load_from_prefill_dir: bool
    autoregressive_decode_assert: Optional[str]
    rope_type: RoPEType
    rope_use_scale: bool
    rope_min_timescale: PositiveInt
    rope_max_timescale: PositiveInt
    local_rope_max_timescale: int
    max_position_embeddings: Optional[PositiveInt]
    original_max_position_embeddings: Optional[PositiveInt]
    rope_factor: Optional[PositiveInt]
    beta_fast: Optional[PositiveInt]
    beta_slow: Optional[PositiveInt]
    mscale: Optional[NonNegativeFloat]
    yarn_rope_config: Optional[
        Any
    ]  # Handles legacy/test keys but isn't used in final nested config logic
    record_internal_nn_metrics: NonNegativeInt = Field(0)
    optimizer_memory_host_offload: bool = Field(False)
    parameter_memory_host_offload: bool = Field(False)
    # Global batch sizes
    global_batch_size_to_eval_on: int = Field(1)
    global_batch_size_to_load: int = Field(1)
    global_batch_size_to_load_eval: int = Field(1)
    global_batch_size_to_train_on: int = Field(1)
    micro_batch_size_to_eval_on: int = Field(1)
    micro_batch_size_to_train_on: int = Field(1)

    # Include remaining keys if any, use ConfigDict(extra='ignore') to be safe
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


# -----------------------------------------------------------------------------
# User-Facing Nested Model and Builder Function
# -----------------------------------------------------------------------------


class MaxTextConfig(BaseModel):
    """A nested, user-facing Pydantic model for MaxText configuration."""

    path: PathConfig
    run: GeneralRunSetting
    checkpoint: CheckpointSetting
    model_architecture: ModelArchitecture
    attention: AttentionSetting
    mla: Optional[MlaSetting]
    parallelism: HardwareAndParallelismSetting
    training: TrainingSetting

    @computed_field()
    @property
    def ici_parallelism(self) -> List[int]:  # Name matches "Expect This"
        # This logic assumes the parallelism fields are found on the top-level final config object.
        # The build_config function below will need to ensure this is true.
        p = self.parallelism_dims_ici
        return [
            p.ici_data_parallelism,
            p.ici_pipeline_parallelism,
            p.ici_fsdp_parallelism,
            p.ici_fsdp_transpose_parallelism,
            p.ici_sequence_parallelism,
            p.ici_context_parallelism,
            p.ici_context_autoregressive_parallelism,
            p.ici_tensor_parallelism,
            p.ici_tensor_transpose_parallelism,
            p.ici_tensor_sequence_parallelism,
            p.ici_expert_parallelism,
            p.ici_autoregressive_parallelism,
        ]

    @computed_field()
    @property
    def dcn_parallelism(self) -> List[int]:  # Name matches "Expect This"
        p = self.parallelism_dims_dcn
        return [
            p.dcn_data_parallelism,
            p.dcn_pipeline_parallelism,
            p.dcn_fsdp_parallelism,
            p.dcn_fsdp_transpose_parallelism,
            p.dcn_sequence_parallelism,
            p.dcn_context_parallelism,
            p.dcn_context_autoregressive_parallelism,
            p.dcn_tensor_parallelism,
            p.dcn_tensor_transpose_parallelism,
            p.dcn_tensor_sequence_parallelism,
            p.dcn_expert_parallelism,
            p.dcn_autoregressive_parallelism,
        ]

    # Placeholder for parallelism dims, populated by build_config
    parallelism_dims_ici: IciParallelismConfig = Field(exclude=True)
    parallelism_dims_dcn: DcnParallelismConfig = Field(exclude=True)


def build_config(flat_cfg: _FlatConfig) -> MaxTextConfig:
    """Builds the nested MaxTextConfig from the validated flat config."""

    path_cfg = PathConfig(
        base_output_directory=flat_cfg.base_output_directory, run_name=flat_cfg.run_name
    )

    run_cfg = GeneralRunSetting(
        log_period=flat_cfg.log_period,
        steps=flat_cfg.steps,
        log_config=flat_cfg.log_config,
        enable_tensorboard=flat_cfg.enable_tensorboard,
        metrics_file=flat_cfg.metrics_file,
        gcs_metrics=flat_cfg.gcs_metrics,
        save_config_to_gcs=flat_cfg.save_config_to_gcs,
        max_checkify=flat_cfg.max_checkify,
        rep=flat_cfg.rep,
    )

    checkpoint_cfg = CheckpointSetting(
        load_parameters_path=flat_cfg.load_parameters_path,
        lora_input_adapters_path=flat_cfg.lora_input_adapters_path,
        load_full_state_path=flat_cfg.load_full_state_path,
        checkpoint_is_quantized=flat_cfg.checkpoint_is_quantized,
        enable_checkpointing=flat_cfg.enable_checkpointing,
        async_checkpointing=flat_cfg.async_checkpointing,
        checkpoint_period=flat_cfg.checkpoint_period,
        save_quantized_params_path=flat_cfg.save_quantized_params_path,
        force_unroll=flat_cfg.force_unroll,
    )

    model_arch_cfg = ModelArchitecture(
        model_name=flat_cfg.model_name,
        decoder_block=flat_cfg.decoder_block,
        emb_dim=flat_cfg.emb_dim,
        mlp_dim=flat_cfg.mlp_dim,
        num_decoder_layers=flat_cfg.num_decoder_layers,
        num_query_heads=flat_cfg.num_query_heads,
        num_kv_heads=flat_cfg.num_kv_heads,
        head_dim=flat_cfg.head_dim,
        global_parameter_scale=flat_cfg.global_parameter_scale,
        base_moe_mlp_dim=flat_cfg.base_moe_mlp_dim,
    )

    attention_cfg = AttentionSetting(
        attention=flat_cfg.attention,
        attention_type=flat_cfg.attention_type,
        sliding_window_size=flat_cfg.sliding_window_size,
        chunk_attn_window_size=flat_cfg.chunk_attn_window_size,
        fused_qkv=flat_cfg.fused_qkv,
        fused_mlp=flat_cfg.fused_mlp,
        attn_logits_soft_cap=flat_cfg.attn_logits_soft_cap,
        final_logits_soft_cap=flat_cfg.final_logits_soft_cap,
        use_post_attn_norm=flat_cfg.use_post_attn_norm,
        use_post_ffw_norm=flat_cfg.use_post_ffw_norm,
    )

    mla_cfg = (
        MlaSetting(
            q_lora_rank=flat_cfg.q_lora_rank,
            kv_lora_rank=flat_cfg.kv_lora_rank,
            qk_nope_head_dim=flat_cfg.qk_nope_head_dim,
            qk_rope_head_dim=flat_cfg.qk_rope_head_dim,
            v_head_dim=flat_cfg.v_head_dim,
            mla_naive_kvcache=flat_cfg.mla_naive_kvcache,
        )
        if flat_cfg.attention_type == AttentionType.MLA
        else None
    )

    parallelism_cfg = HardwareAndParallelismSetting(
        hardware=flat_cfg.hardware,
        num_slices=flat_cfg.num_slices,
        ici_fsdp_parallelism=flat_cfg.ici_fsdp_parallelism,
        dcn_data_parallelism=flat_cfg.dcn_data_parallelism,
        mesh_axes=flat_cfg.mesh_axes,
        logical_axis_rules=flat_cfg.logical_axis_rules,
    )

    training_cfg = TrainingSetting(
        per_device_batch_size=flat_cfg.per_device_batch_size,
        eval_per_device_batch_size=flat_cfg.eval_per_device_batch_size,
        max_target_length=flat_cfg.max_target_length,
        max_prefill_predict_length=flat_cfg.max_prefill_predict_length,
        learning_rate=flat_cfg.learning_rate,
        learning_rate_schedule_steps=flat_cfg.learning_rate_schedule_steps,
        warmup_steps_fraction=flat_cfg.warmup_steps_fraction,
        opt_type=flat_cfg.opt_type,
        adam_b1=flat_cfg.adam_b1,
        adam_b2=flat_cfg.adam_b2,
        adam_weight_decay=flat_cfg.adam_weight_decay,
        mu_dtype=flat_cfg.mu_dtype,
    )

    ici_dims_cfg = IciParallelismConfig(
        **{k: getattr(flat_cfg, k) for k in IciParallelismConfig.model_fields.keys()}
    )
    dcn_dims_cfg = DcnParallelismConfig(
        **{k: getattr(flat_cfg, k) for k in DcnParallelismConfig.model_fields.keys()}
    )

    # Assemble the final nested config, including the full dim details for computed fields
    nested_config = MaxTextConfig(
        path=path_cfg,
        run=run_cfg,
        checkpoint=checkpoint_cfg,
        model_architecture=model_arch_cfg,
        attention=attention_cfg,
        mla=mla_cfg,
        parallelism=parallelism_cfg,
        training=training_cfg,
        parallelism_dims_ici=ici_dims_cfg,
        parallelism_dims_dcn=dcn_dims_cfg,
    )

    # Transfer all other top-level fields from flat to nested
    for field in _FlatConfig.model_fields:
        if not hasattr(nested_config, field) and field not in [
            "base_output_directory",
            "run_name",
        ]:  # Avoid overwriting sub-model fields
            setattr(nested_config, field, getattr(flat_cfg, field))

    return nested_config
