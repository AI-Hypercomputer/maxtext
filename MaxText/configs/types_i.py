#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Pydantic models and loader for MaxText configuration."""
import os
from typing import Any, List, Optional, Tuple, Union

import jax
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class GeneralConfig(BaseModel):
    """General configuration for the run."""

    run_name: str = Field(
        "", description="Name of the run for logging and checkpointing."
    )
    hardware: str = Field(
        "tpu", description="Hardware platform: 'tpu', 'gpu', or 'cpu'."
    )
    model_name: str = Field(
        "default", description="Name of the model configuration to load."
    )
    override_model_config: bool = Field(
        False, description="Allow CLI to override model-specific configs."
    )
    log_config: bool = Field(
        True, description="Print the final configuration at startup."
    )
    enable_single_controller: bool = Field(
        False, description="Enable single-controller mode."
    )
    model_call_mode: str = Field(
        "", description="Mode for model execution, e.g., 'inference'."
    )
    base_output_directory: Optional[str] = Field(
        None, description="Base GCS directory for output artifacts."
    )
    reuse_example_batch: int = Field(
        0, description="Repeatedly uses the same batch for performance testing."
    )


class CheckpointingConfig(BaseModel):
    """Configuration for checkpointing."""

    enable_checkpointing: bool = Field(True, description="Enables checkpointing.")
    async_checkpointing: bool = Field(
        True, description="Use asynchronous checkpointing."
    )
    checkpoint_period: int = Field(10_000, description="Steps between checkpoints.")
    load_parameters_path: str = Field(
        "", description="Path to load model parameters from."
    )
    load_full_state_path: str = Field(
        "", description="Path to load full training state from."
    )
    lora_input_adapters_path: str = Field(
        "", description="Path to a directory with LoRA adapters."
    )
    checkpoint_storage_target_data_file_size_bytes: int = Field(
        2147483648, description="Target file size for Orbax checkpoint chunks."
    )
    checkpoint_storage_use_zarr3: bool = Field(
        True, description="Use Zarr3 for checkpointing."
    )
    save_quantized_params_path: str = Field(
        "", description="Path to save on-the-fly quantized parameters."
    )
    checkpoint_is_quantized: bool = Field(
        False, description="Indicates if the checkpoint being loaded is quantized."
    )
    force_unroll: bool = Field(
        False,
        description="Force unroll of the loop during a parameter-only checkpoint generation.",
    )


class DCNParallelismConfig(BaseModel):
    """Configuration for Data Center Network (DCN) parallelism."""

    dcn_data_parallelism: int = Field(-1, description="DCN data parallelism degree.")
    dcn_fsdp_parallelism: int = Field(1, description="DCN FSDP degree.")
    dcn_fsdp_transpose_parallelism: int = Field(
        1, description="DCN FSDP transpose degree."
    )
    dcn_sequence_parallelism: int = Field(
        1, description="DCN sequence parallelism degree."
    )
    dcn_context_parallelism: int = Field(
        1, description="DCN context parallelism degree."
    )
    dcn_context_autoregressive_parallelism: int = Field(
        1, description="DCN context autoregressive degree."
    )
    dcn_tensor_parallelism: int = Field(1, description="DCN tensor parallelism degree.")
    dcn_tensor_transpose_parallelism: int = Field(
        1, description="DCN tensor transpose degree."
    )
    dcn_tensor_sequence_parallelism: int = Field(
        1, description="DCN tensor sequence degree."
    )
    dcn_pipeline_parallelism: int = Field(
        1, description="DCN pipeline parallelism degree."
    )
    dcn_expert_parallelism: int = Field(1, description="DCN expert parallelism degree.")
    dcn_autoregressive_parallelism: int = Field(
        1, description="DCN autoregressive degree."
    )


class ICIParallelismConfig(BaseModel):
    """Configuration for Inter-Core Interconnect (ICI) parallelism."""

    ici_data_parallelism: int = Field(1, description="ICI data parallelism degree.")
    ici_fsdp_parallelism: int = Field(-1, description="ICI FSDP degree, -1 for auto.")
    ici_fsdp_transpose_parallelism: int = Field(
        1, description="ICI FSDP transpose degree."
    )
    ici_sequence_parallelism: int = Field(
        1, description="ICI sequence parallelism degree."
    )
    ici_context_parallelism: int = Field(
        1, description="ICI context parallelism degree."
    )
    ici_context_autoregressive_parallelism: int = Field(
        1, description="ICI context autoregressive degree."
    )
    ici_tensor_parallelism: int = Field(1, description="ICI tensor parallelism degree.")
    ici_tensor_transpose_parallelism: int = Field(
        1, description="ICI tensor transpose degree."
    )
    ici_tensor_sequence_parallelism: int = Field(
        1, description="ICI tensor sequence degree."
    )
    ici_pipeline_parallelism: int = Field(
        1, description="ICI pipeline parallelism degree."
    )
    ici_expert_parallelism: int = Field(1, description="ICI expert parallelism degree.")
    ici_autoregressive_parallelism: int = Field(
        1, description="ICI autoregressive degree."
    )


class ParallelismConfig(DCNParallelismConfig, ICIParallelismConfig):
    """Aggregated parallelism configuration."""

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
        ],
        description="Names of the mesh axes.",
    )
    logical_axis_rules: List[Tuple[str, Union[str, List[str], None]]]
    num_slices: int = Field(
        -1, description="Number of TPU slices (do not set manually)."
    )
    context_parallel_load_balance: bool = Field(
        True, description="Enable load balancing for context parallelism."
    )


class ModelArchitectureConfig(BaseModel):
    """Core model architecture parameters."""

    global_parameter_scale: int = Field(
        1, description="Global parameter scaling factor."
    )
    base_emb_dim: int = Field(2048, description="Base embedding dimension.")
    base_num_query_heads: int = Field(16, description="Base number of query heads.")
    base_num_kv_heads: int = Field(16, description="Base number of key/value heads.")
    base_mlp_dim: int = Field(7168, description="Base MLP dimension.")
    base_num_decoder_layers: int = Field(
        16, description="Base number of decoder layers."
    )
    head_dim: int = Field(128, description="Dimension of each attention head.")
    mlp_activations: List[str]
    decoder_block: str = Field("llama2", description="Type of decoder block to use.")
    normalization_layer_epsilon: float = Field(
        1e-5, description="Epsilon for normalization layers."
    )
    logits_via_embedding: bool = Field(
        False, description="Share weights between embedding and logits layers."
    )
    use_iota_embed: bool = Field(
        False, description="Use iota operator in embedding for performance."
    )


class AttentionConfig(BaseModel):
    """Configuration for attention mechanisms."""

    attention: str = Field(
        "autoselected",
        description="Attention algorithm (e.g., 'dot_product', 'flash').",
    )
    attention_type: str = Field(
        "global", description="Attention variant (e.g., 'global', 'local_sliding')."
    )
    fused_qkv: bool = Field(False, description="Use a fused QKV projection layer.")
    fused_mlp: bool = Field(False, description="Use a fused MLP layer.")
    dropout_rate: float = Field(0.0, description="Dropout rate.")
    sliding_window_size: int = Field(
        0, description="Size of the sliding window for local attention."
    )


class RoPEConfig(BaseModel):
    """Configuration for Rotary Positional Embeddings."""

    rope_type: str = Field(
        "default", description="Type of RoPE ('default', 'yarn', 'llama3.1')."
    )
    rope_max_timescale: int = Field(10_000, description="Maximum timescale for RoPE.")
    rope_use_scale: bool = Field(True, description="Apply RoPE scaling for Llama 3.1.")
    local_rope_max_timescale: int = Field(
        -1, description="RoPE max timescale for local attention."
    )
    rope_factor: int = Field(40, description="YaRN RoPE scaling factor.")
    beta_fast: int = Field(32, description="YaRN RoPE beta_fast parameter.")
    beta_slow: int = Field(1, description="YaRN RoPE beta_slow parameter.")
    use_untrainable_positional_embedding: bool = Field(
        False, description="Use non-trainable positional embeddings."
    )
    trainable_position_size: int = Field(
        -1, description="Size of trainable positional embeddings (for gpt3)."
    )


class MoeConfig(BaseModel):
    """Configuration for Mixture of Experts (MoE) layers."""

    num_experts: int = Field(1, description="Total number of experts.")
    num_experts_per_tok: int = Field(
        1, description="Number of experts to route each token to."
    )
    megablox: bool = Field(True, description="Use Megablox for MoE.")
    sparse_matmul: bool = Field(True, description="Use sparse matmul for MoE.")
    capacity_factor: float = Field(
        -1.0, description="Expert capacity factor; -1 implies no token dropping."
    )
    base_moe_mlp_dim: int = Field(
        7168, description="Intermediate dimension for MoE MLPs (DeepSeek)."
    )
    routed_scaling_factor: float = Field(
        1.0, description="Scaling factor for routing scores (DeepSeek)."
    )
    routed_score_func: str = Field(
        "", description="Scoring function for routing (DeepSeek)."
    )
    routed_bias: bool = Field(False, description="Add bias for routing (DeepSeek).")
    n_routing_groups: int = Field(
        -1, description="Number of routing groups (DeepSeek)."
    )
    topk_routing_group: int = Field(
        -1, description="Number of top groups to route to (DeepSeek)."
    )


class DatasetConfig(BaseModel):
    """Configuration for the input data pipeline."""

    dataset_type: str = Field(
        "tfds", description="Type of dataset ('tfds', 'hf', 'grain')."
    )
    per_device_batch_size: float = Field(12.0, description="Batch size per device.")
    eval_per_device_batch_size: float = Field(
        0.0, description="Eval batch size per device."
    )
    packing: bool = Field(
        True, description="Enable packing of multiple sequences into one example."
    )
    dataset_path: str = Field("", description="Path for TFDS or Grain dataset.")
    hf_path: str = Field("", description="Path or name of a Hugging Face dataset.")
    hf_train_files: str = Field("", description="Glob pattern for HF training files.")
    hf_eval_files: str = Field("", description="Glob pattern for HF evaluation files.")
    train_data_columns: List[str] = Field(
        default_factory=lambda: ["text"], description="Columns to use for training."
    )
    eval_data_columns: List[str] = Field(
        default_factory=lambda: ["text"], description="Columns to use for evaluation."
    )


class TokenizerConfig(BaseModel):
    """Configuration for the tokenizer."""

    tokenizer_path: str = Field(
        "assets/tokenizer.llama2", description="Path to the tokenizer model."
    )
    tokenizer_type: str = Field(
        "sentencepiece",
        description="Type of tokenizer ('sentencepiece', 'tiktoken', 'huggingface').",
    )
    vocab_size: int = Field(32_000, description="Vocabulary size.")
    add_bos: bool = Field(True, description="Add BOS token to sequences.")
    add_eos: bool = Field(True, description="Add EOS token to sequences.")
    use_chat_template: bool = Field(
        False, description="Apply a chat template to the data."
    )


class TrainingConfig(BaseModel):
    """Configuration for the training loop and optimizer."""

    steps: int = Field(150_001, description="Total number of training steps.")
    learning_rate: float = Field(3.0e-5, description="Peak learning rate.")
    warmup_steps_fraction: float = Field(
        0.1, description="Fraction of steps for learning rate warmup."
    )
    learning_rate_schedule_steps: int = Field(
        -1, description="Length of the learning rate schedule."
    )
    gradient_clipping_threshold: float = Field(
        1.0, description="Threshold for gradient clipping."
    )
    gradient_accumulation_steps: int = Field(
        1, description="Number of steps to accumulate gradients."
    )
    opt_type: str = Field(
        "adamw", description="Optimizer type ('adamw', 'adam_pax', 'sgd')."
    )
    adam_b1: float = Field(0.9, description="Adam optimizer beta1 parameter.")
    adam_b2: float = Field(0.95, description="Adam optimizer beta2 parameter.")
    adam_weight_decay: float = Field(0.1, description="AdamW weight decay.")
    remat_policy: str = Field("full", description="Rematerialization policy.")
    scan_layers: bool = Field(
        True, description="Use jax.lax.scan to iterate over decoder layers."
    )


class QuantizationConfig(BaseModel):
    """Configuration for quantization."""

    quantization: str = Field(
        "", description="Quantization scheme (e.g., 'int8', 'fp8')."
    )
    quantize_kvcache: bool = Field(
        False, description="Enable quantization of the K/V cache."
    )
    kv_quant_axis: str = Field(
        "heads_and_dkv", description="Quantization axis for K/V cache."
    )
    kv_quant_dtype: str = Field(
        "int8", description="Data type for K/V cache quantization."
    )
    quant_cfg_path: str = Field(
        "", description="Path to a custom quantization configuration file."
    )
    replicate_quant_scale: bool = Field(
        False, description="Replicate quantization scale to avoid inefficient fusion."
    )


class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning methods like DPO and SFT."""

    use_dpo: bool = Field(
        False, description="Enable Direct Preference Optimization (DPO)."
    )
    dpo_beta: float = Field(0.1, description="Beta parameter for DPO loss.")
    dpo_label_smoothing: float = Field(0.0, description="Label smoothing for DPO.")
    use_sft: bool = Field(False, description="Enable Supervised Fine-Tuning (SFT).")
    sft_train_on_completion_only: bool = Field(
        False, description="For SFT, train only on completion tokens."
    )


class InferenceConfig(BaseModel):
    """Configuration for inference and decoding."""

    max_prefill_predict_length: int = Field(
        64, description="Maximum prefill length for autoregression."
    )
    max_target_length: int = Field(2048, description="Maximum sequence length.")
    prompt: str = Field("I love to", description="Default prompt for decoding.")
    autoregressive_decode_assert: str = Field(
        "", description="Assertion for autoregressive decoding tests."
    )
    decode_sampling_strategy: str = Field(
        "greedy", description="Sampling strategy for decoding."
    )
    decode_sampling_nucleus_p: float = Field(
        -1, description="Nucleus (top-p) sampling parameter."
    )
    decode_sampling_top_k: int = Field(0, description="Top-k sampling parameter.")
    decode_sampling_temperature: float = Field(1.0, description="Sampling temperature.")
    inference_server: str = Field(
        "MaxtextInterleavedServer", description="Inference server to start."
    )
    return_log_prob: bool = Field(
        False, description="Whether to return log probabilities during inference."
    )


class SpecializedModelConfig(BaseModel):
    """Configuration for specialized model architectures like Gemma and Llama4."""

    attn_logits_soft_cap: float = Field(
        0.0, description="Value for soft-capping attention logits (Gemma 2)."
    )
    final_logits_soft_cap: float = Field(
        0.0, description="Value for soft-capping final logits (Gemma 2)."
    )
    use_post_attn_norm: bool = Field(
        False, description="Use post-attention normalization (Gemma 2/3)."
    )
    use_post_ffw_norm: bool = Field(
        False, description="Use post-feedforward normalization (Gemma 2/3)."
    )
    use_qk_norm: bool = Field(
        False, description="Apply L2 normalization to Q/K after RoPE (Llama 4)."
    )
    nope_layer_interval: int = Field(
        -1, description="Interval for layers without RoPE (Llama 4)."
    )
    interleave_moe_layer_step: int = Field(
        1, description="Interval for MoE layers (Llama 4)."
    )
    temperature_tuning: bool = Field(
        False, description="Dynamically scale attention temperature (Llama 4)."
    )


class LayoutConfig(BaseModel):
    """Advanced configuration for tensor layouts."""

    reshape_q: bool = Field(False, description="Reshape Q projection for performance.")
    prefill_cache_axis_order: Tuple[int, ...] = Field(
        (1, 2, 0, 3), description="Layout of K/V cache for prefill."
    )
    ar_cache_axis_order: Tuple[int, ...] = Field(
        (1, 2, 0, 3), description="Layout of K/V cache for autoregression."
    )
    compute_axis_order: Tuple[int, ...] = Field(
        (0, 1, 2, 3), description="Layout for attention computation."
    )


class MlaConfig(BaseModel):
    """Configuration for Multi-Headed Latent Attention (MLA)."""

    q_lora_rank: int = Field(0, description="LoRA rank for query projection in MLA.")
    kv_lora_rank: int = Field(
        512, description="LoRA rank for key/value projection in MLA."
    )
    qk_nope_head_dim: int = Field(
        128, description="Head dimension for queries/keys without RoPE in MLA."
    )
    qk_rope_head_dim: int = Field(
        64, description="Head dimension for queries/keys with RoPE in MLA."
    )
    v_head_dim: int = Field(128, description="Head dimension for values in MLA.")


class SplashAttentionConfig(BaseModel):
    """Configuration for Splash Attention on TPUs."""

    sa_block_q: int = Field(512, description="Block size for Q.")
    sa_block_kv: int = Field(512, description="Block size for K/V.")
    sa_block_kv_compute: int = Field(512, description="Block size for K/V compute.")
    sa_block_q_dkv: int = Field(512, description="Block size for Q in dKV computation.")
    sa_block_kv_dkv: int = Field(
        512, description="Block size for K/V in dKV computation."
    )
    sa_block_kv_dkv_compute: int = Field(
        512, description="Block size for K/V compute in dKV computation."
    )
    sa_block_q_dq: int = Field(512, description="Block size for Q in dQ computation.")
    sa_block_kv_dq: int = Field(
        512, description="Block size for K/V in dQ computation."
    )


class MaxTextConfig(
    GeneralConfig,
    CheckpointingConfig,
    ParallelismConfig,
    ModelArchitectureConfig,
    AttentionConfig,
    RoPEConfig,
    MoeConfig,
    DatasetConfig,
    TokenizerConfig,
    TrainingConfig,
    QuantizationConfig,
    FineTuningConfig,
    InferenceConfig,
    SpecializedModelConfig,
    LayoutConfig,
    MlaConfig,
    SplashAttentionConfig,
    extra="allow",
):
    """The root Pydantic configuration for MaxText."""

    # Top-level fields
    dtype: str = Field("bfloat16", description="Data type for activations.")

    # Derived Fields (populated by model_validator)
    global_batch_size_to_train_on: Optional[int] = None
    num_query_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    emb_dim: Optional[int] = None
    mlp_dim: Optional[int] = None

    @field_validator(
        "prefill_cache_axis_order",
        "ar_cache_axis_order",
        "compute_axis_order",
        "mlp_activations",
        "train_data_columns",
        "eval_data_columns",
        mode="before",
    )
    @classmethod
    def _parse_str_to_tuple(cls, v: Any) -> Any:
        """Handles parsing of string-encoded tuples/lists from YAML."""
        if isinstance(v, str):
            if "," in v:
                return tuple(map(int, v.split(","))) if v[0].isdigit() else v.split(",")
            return [v]
        return v

    @field_validator("logical_axis_rules", mode="before")
    @classmethod
    def _parse_logical_axis_rules(
        cls, v: Any
    ) -> List[Tuple[str, Union[str, List[str], None]]]:
        """Converts a list of lists into a list of tuples for logical axis rules."""
        if isinstance(v, list) and all(isinstance(i, list) for i in v):
            return [tuple(item) for item in v]
        return v

    @model_validator(mode="after")
    def set_derived_fields(self) -> "MaxTextConfig":
        """Computes derived configuration fields after initial validation."""
        scale = self.global_parameter_scale
        self.num_query_heads = self.base_num_query_heads * scale
        self.num_kv_heads = self.base_num_kv_heads * scale
        self.emb_dim = self.base_emb_dim * scale
        self.mlp_dim = self.base_mlp_dim * scale
        try:
            device_count = jax.device_count()
        except (RuntimeError, ValueError):
            device_count = 1
            print(
                f"Warning: Could not determine JAX device count. Defaulting to {device_count}"
            )
        self.global_batch_size_to_train_on = int(
            self.per_device_batch_size * device_count
        )
        return self


def _load_and_merge_configs(config_files, **kwargs):
    """Loads base and override configs from YAML files and merges them with kwargs."""
    merged_config = {}
    if not isinstance(config_files, list):
        config_files = [config_files]

    for config_file in config_files:
        if (
            not isinstance(config_file, str)
            or not os.path.exists(config_file)
            or config_file[config_file.rfind(os.path.extsep) :] != ".yml"
        ):
            continue
        if not os.path.exists(config_file):
            print(f"Warning: Config file not found: {config_file}")
            continue

        with open(config_file, "rt", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        if "base_config" in yaml_config:
            base_path = os.path.join(
                os.path.dirname(config_file), yaml_config["base_config"]
            )
            base_config_data = _load_and_merge_configs([base_path])
            base_config_data.update(yaml_config)
            yaml_config = base_config_data

        merged_config.update(yaml_config)

    merged_config.update(kwargs)
    return merged_config


def initialize(config_files: list[str], **kwargs) -> MaxTextConfig:
    """
    Loads YAML configs, merges them, applies kwarg overrides, and returns a Pydantic Config object.
    This function is a Pydantic-based replacement for `MaxText.pyconfig.initialize`.
    """
    raw_config = _load_and_merge_configs(config_files, **kwargs)
    raw_config.pop("base_config", None)
    return MaxTextConfig(**raw_config)
