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

from typing import Optional, Union

from pydantic import BaseModel, Field


class CoreConfig(BaseModel):
  """Core general configuration fields, common to all config files.

  Attributes:
      run_name (str): User-defined name for run.
      model_name (str): Model identifier name.
      override_model_config (bool): Whether to allow CLI model param override.
      base_config (Optional[str]): Relative path to base config if inherited.
  """

  run_name: str = ""
  model_name: str = "default"
  override_model_config: bool = False
  base_config: Optional[str] = None


class ModelConfig(BaseModel):
  """Model architecture-specific configuration.

  Attributes:
      decoder_block (str): Style of decoder block used.
      base_emb_dim (int): Base embedding dimension.
      base_num_query_heads (int): Number of query attention heads.
      base_num_kv_heads (int): Number of key-value attention heads.
      base_mlp_dim (int): Dimension of MLP intermediate layers.
      base_num_decoder_layers (int): Number of decoder layers.
      head_dim (int): Head dimension size.
      mlp_activations (list[str]): list of activations used in MLPs.
      vocab_size (int): Vocabulary size.
      normalization_layer_epsilon (float): Epsilon for normalization layers.
      logits_via_embedding (bool): Whether to calculate logits via embedding.
      enable_dropout (bool): Whether dropout is enabled.
      shared_experts (Optional[int]): Number of shared experts in MoE.
      num_experts (Optional[int]): Number of experts in MoE.
      num_experts_per_tok (Optional[int]): Experts used per token.
      base_moe_mlp_dim (Optional[int]): MoE mlp intermediate dim.
      first_num_dense_layers (Optional[int]): Initial dense layers count.
      routed_scaling_factor (Optional[float]): Routing score scaling factor.
      routed_score_func (Optional[str]): Routing scoring function.
      routed_bias (Optional[bool]): Whether routing adds a bias term.
      n_routing_groups (Optional[int]): Number of routing groups.
      topk_routing_group (Optional[int]): Top K routing groups count.
      use_qk_norm (Optional[bool]): Use normalization on Q/K projections.
      sliding_window_size (Optional[int]): Sliding window size for attention.
      attn_logits_soft_cap (Optional[float]): Attention logits soft cap.
      final_logits_soft_cap (Optional[float]): Final logits soft cap.
      use_post_attn_norm (Optional[bool]): Whether to use norm after attention.
      use_post_ffw_norm (Optional[bool]): Whether to use norm after FFN.
      rope_type (Optional[str]): RoPE embedding type.
      rope_max_timescale (Optional[int]): Maximum timescale for RoPE.
      rope_use_scale (Optional[bool]): Whether to apply RoPE scaling.
  """

  decoder_block: str = "llama2"
  base_emb_dim: int = 2048
  base_num_query_heads: int = 16
  base_num_kv_heads: int = 16
  base_mlp_dim: int = 7168
  base_num_decoder_layers: int = 16
  head_dim: int = 128
  mlp_activations: list[str] = Field(default_factory=lambda: ["silu", "linear"])
  vocab_size: int = 32000
  normalization_layer_epsilon: float = 1e-5
  logits_via_embedding: bool = False
  enable_dropout: bool = True

  shared_experts: Optional[int] = None
  num_experts: Optional[int] = None
  num_experts_per_tok: Optional[int] = None
  base_moe_mlp_dim: Optional[int] = None
  first_num_dense_layers: Optional[int] = None
  routed_scaling_factor: Optional[float] = None
  routed_score_func: Optional[str] = None
  routed_bias: Optional[bool] = None
  n_routing_groups: Optional[int] = None
  topk_routing_group: Optional[int] = None

  use_qk_norm: Optional[bool] = None

  sliding_window_size: Optional[int] = None
  attn_logits_soft_cap: Optional[float] = None
  final_logits_soft_cap: Optional[float] = None
  use_post_attn_norm: Optional[bool] = None
  use_post_ffw_norm: Optional[bool] = None

  rope_type: Optional[str] = None
  rope_max_timescale: Optional[int] = None
  rope_use_scale: Optional[bool] = None


class CheckpointConfig(BaseModel):
  """Checkpointing related parameters controlling saving/loading model state.

  Attributes:
      enable_checkpointing (bool): Enable checkpointing.
      async_checkpointing (bool): Use asynchronous checkpointing.
      checkpoint_period (int): Steps between checkpointing.
      load_parameters_path (str): Path for parameter-only checkpoint load.
      load_full_state_path (str): Path for full state checkpoint load.
      lora_input_adapters_path (str): GCS path for LoRA adapter input.
      enable_single_replica_ckpt_restoring (bool): Use single replica checkpoint read.
      checkpoint_storage_target_data_file_size_bytes (int): Target file size for checkpoint chunks.
      checkpoint_storage_use_ocdbt (bool): Use OCDBT kvstore for checkpointing.
      checkpoint_storage_use_zarr3 (bool): Use Zarr3 storage format.
      checkpoint_storage_concurrent_gb (int): Concurrent GB for IO operations in checkpoint.
  """

  enable_checkpointing: bool = True
  async_checkpointing: bool = True
  checkpoint_period: int = 10_000
  load_parameters_path: str = ""
  load_full_state_path: str = ""
  lora_input_adapters_path: str = ""
  enable_single_replica_ckpt_restoring: bool = False
  checkpoint_storage_target_data_file_size_bytes: int = 2147483648
  checkpoint_storage_use_ocdbt: bool = True
  checkpoint_storage_use_zarr3: bool = True
  checkpoint_storage_concurrent_gb: int = 96


class OptimizerConfig(BaseModel):
  """Optimizer hyperparameters for the training run.

  Attributes:
      opt_type (str): Optimizer type ("adamw","adam_pax","sgd").
      adam_b1 (float): Beta1 decay rate for Adam optimizer.
      adam_b2 (float): Beta2 decay rate for Adam optimizer.
      adam_eps (float): Epsilon value to prevent division by zero.
      adam_eps_root (float): Additional epsilon for root variance.
      adam_weight_decay (float): Weight decay coefficient for AdamW.
      mu_dtype (str): Data type for first moment storage (optional).
  """

  opt_type: str = "adamw"
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1e-8
  adam_eps_root: float = 0.0
  adam_weight_decay: float = 0.1
  mu_dtype: str = ""


class DatasetConfig(BaseModel):
  """Dataset loading and processing-related configuration.

  Attributes:
      dataset_type (str): Dataset pipeline type (e.g., "tfds", "hf", "grain", "synthetic").
      dataset_path (str): Path or URI for dataset location.
      dataset_name (str): Name/version for the training dataset.
      eval_dataset_name (str): Name/version for evaluation dataset.
      train_split (str): Train split name.
      eval_split (str): Eval split name.
      train_data_columns (list[str]): list of columns used for training data.
      eval_data_columns (list[str]): list of columns used for eval data.
      per_device_batch_size (float): Per device batch size for training.
      eval_per_device_batch_size (float): Per device eval batch size.
      num_epoch (int): Number of epochs to train.
      packing (bool): If True, enable packing data batches.
      expansion_factor_real_data (int): Host expansion factor for real data.
      hf_path (str): Huggingface dataset path.
      hf_data_dir (str): Huggingface data directory.
      hf_train_files (str): Huggingface train files pattern.
      hf_eval_split (str): Huggingface eval split name.
      hf_eval_files (str): Huggingface eval files pattern.
      hf_access_token (str): Huggingface access token.
      grain_train_files (str): Grain pipeline train files.
      grain_eval_files (str): Grain pipeline eval files.
      grain_file_type (str): Grain file format ("arrayrecord" or "parquet").
      grain_worker_count (int): Number of grain workers for training.
      grain_worker_count_eval (int): Number of grain workers for evaluation.
      colocated_python_data_input (bool): Use colocated python data input.
  """

  dataset_type: str = "tfds"
  dataset_path: str = ""
  dataset_name: str = "c4/en:3.0.1"
  eval_dataset_name: str = "c4/en:3.0.1"
  train_split: str = "train"
  eval_split: str = "validation"
  train_data_columns: list[str] = Field(default_factory=lambda: ["text"])
  eval_data_columns: list[str] = Field(default_factory=lambda: ["text"])
  per_device_batch_size: float = 12.0
  eval_per_device_batch_size: float = 0.0
  num_epoch: int = 1
  packing: bool = True
  expansion_factor_real_data: int = -1

  hf_path: str = ""
  hf_data_dir: str = ""
  hf_train_files: str = ""
  hf_eval_split: str = ""
  hf_eval_files: str = ""
  hf_access_token: str = ""

  grain_train_files: str = ""
  grain_eval_files: str = ""
  grain_file_type: str = "arrayrecord"
  grain_worker_count: int = 1
  grain_worker_count_eval: int = 1

  colocated_python_data_input: bool = False


class TokenizerConfig(BaseModel):
  """Tokenizer related configuration parameters.

  Attributes:
      tokenizer_path (str): Path to tokenizer assets.
      tokenizer_type (str): Tokenizer type.
      use_chat_template (bool): Use chat template tokenization.
      tokenize_train_data (bool): Whether to tokenize train data.
      tokenize_eval_data (bool): Whether to tokenize eval data.
      add_bos (bool): Add beginning-of-sentence token.
      add_eos (bool): Add end-of-sentence token.
  """

  tokenizer_path: str = "assets/tokenizer.llama2"
  tokenizer_type: str = "sentencepiece"
  use_chat_template: bool = False
  tokenize_train_data: bool = True
  tokenize_eval_data: bool = True
  add_bos: bool = True
  add_eos: bool = True


class ParallelismConfig(BaseModel):
  """Configuration related to model parallelism and mesh axes.

  Attributes:
      mesh_axes (list[str]): Names of axes in the device mesh.
      logical_axis_rules (list[list[Union[str, list[str]]]]): Logical axis rules for sharding.
      data_sharding (list[list[str]]): Lists specifying data sharding axes.
      input_data_sharding_logical_axes (list[str]): Logical axes for input data sharding.
      sharding_tolerance (float): Allowed percentage of non-sharded parameters.
  """

  mesh_axes: list[str] = Field(
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

  logical_axis_rules: list[list[Union[str, list[str]]]] = Field(
      default_factory=lambda: [
          ["activation_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_batch_no_exp", ["data", "fsdp", "fsdp_transpose"]],
          ["activation_embed_and_logits_batch", ["data", "stage", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_heads", ["tensor", "tensor_transpose", "sequence", "tensor_sequence", "autoregressive"]],
          ["activation_kv_heads", ["tensor", "tensor_transpose", "sequence", "tensor_sequence"]],
          ["activation_length", ["sequence", "context"]],
          ["activation_length", ["context"]],
          ["activation_norm_length", ["tensor_sequence", "context", "sequence"]],
          ["activation_q_length", ["context"]],
          ["activation_kv_length", []],
          ["activation_embed", ["tensor", "tensor_transpose"]],
          ["activation_mlp", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["activation_kv", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["activation_prefill_kv_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_kv_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["activation_kv_head_dim", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["activation_vocab", ["tensor", "tensor_transpose", "sequence", "tensor_sequence"]],
          ["activation_vocab", ["tensor", "tensor_transpose"]],
          ["activation_vocab", "tensor_sequence"],
          ["activation_vocab", ["sequence", "context"]],
          ["activation_stage", "stage"],
          ["activation_exp", ["expert"]],
          ["decode_batch", ["data", "fsdp", "fsdp_transpose", "expert"]],
          ["decode_length", ["sequence"]],
          ["mlp", ["fsdp_transpose", "tensor", "tensor_sequence", "autoregressive"]],
          ["vocab", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["heads", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["q_heads", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["kv_heads", ["tensor", "tensor_transpose", "tensor_sequence", "autoregressive"]],
          ["embed", ["fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context", "expert"]],
          ["embed", ["fsdp", "sequence", "tensor_transpose", "context", "expert"]],
          ["embed", ["fsdp", "fsdp_transpose", "sequence", "context", "expert"]],
          ["embed", ["fsdp", "sequence", "context", "expert"]],
          ["embed_no_exp", ["fsdp", "fsdp_transpose", "sequence", "tensor_transpose", "context"]],
          ["embed_no_exp", ["fsdp", "sequence", "tensor_transpose", "context"]],
          ["embed_no_exp", ["fsdp", "fsdp_transpose", "sequence", "context"]],
          ["embed_no_exp", ["fsdp", "sequence", "context"]],
          ["q_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "tensor_transpose", "expert"]],
          ["q_lora", ["fsdp", "sequence", "context", "tensor_transpose", "expert"]],
          ["q_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "expert"]],
          ["q_lora", ["fsdp", "sequence", "context", "expert"]],
          ["kv_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "tensor_transpose", "expert"]],
          ["kv_lora", ["fsdp", "sequence", "context", "tensor_transpose", "expert"]],
          ["kv_lora", ["fsdp", "fsdp_transpose", "sequence", "context", "expert"]],
          ["kv_lora", ["fsdp", "sequence", "context", "expert"]],
          ["norm", ["tensor", "tensor_transpose", "tensor_sequence"]],
          ["layers", "stage"],
          ["kv", []],
          ["kv_head_dim", []],
          ["cache_batch_prefill", []],
          ["cache_batch", []],
          ["cache_heads", ["autoregressive", "tensor", "tensor_transpose", "tensor_sequence"]],
          ["cache_heads", ["autoregressive", "tensor", "tensor_sequence"]],
          ["cache_kv", []],
          ["cache_sequence", []],
          ["exp", "expert"],
          ["paged_kv_heads", ["tensor"]],
          ["num_pages", []],
          ["tokens_per_page", []],
          ["paged_kv_head_dim_size", []],
      ]
  )

  data_sharding: list[list[str]] = Field(
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

  input_data_sharding_logical_axes: list[str] = Field(
      default_factory=lambda: ["activation_embed_and_logits_batch", "activation_norm_length"]
  )

  sharding_tolerance: float = 0.02


class InferenceConfig(BaseModel):
  """Inference-specific configuration parameters.

  Attributes:
      inference_server (str): Server to launch for inference.
      inference_microbenchmark_prefill_lengths (str): Prefill lengths for benchmarking.
      inference_microbenchmark_stages (str): Benchmarking stages.
      inference_microbenchmark_loop_iters (int): Number iterations for microbenchmark loop.
      inference_microbenchmark_log_file_path (str): File path for microbenchmark logs.
      inference_microbenchmark_num_samples (list[int]): Number of samples for microbenchmarking.
      inference_metadata_file (str): Path to metadata JSON.
      prefill_slice (str): Slice to use for prefill in disaggregation.
      generate_slice (str): Slice to use for generation in disaggregation.
      inference_benchmark_test (bool): Flag to enable benchmark test.
      enable_model_warmup (bool): Enable warmup before inference.
      enable_llm_inference_pool (bool): Use LLM inference pool.
      multi_sampling (bool): Multi-sample decoding.
      return_log_prob (bool): Return log probabilities.
      enable_prefix_caching (bool): Enable prefix caching optimizations.
  """

  inference_server: str = "MaxtextInterleavedServer"
  inference_microbenchmark_prefill_lengths: str = "64,128,256,512,1024"
  inference_microbenchmark_stages: str = "prefill,generate"
  inference_microbenchmark_loop_iters: int = 10
  inference_microbenchmark_log_file_path: str = ""
  inference_microbenchmark_num_samples: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
  inference_metadata_file: str = ""

  prefill_slice: str = "v5e-16"
  generate_slice: str = "v5e-16"
  inference_benchmark_test: bool = False
  enable_model_warmup: bool = False
  enable_llm_inference_pool: bool = False
  multi_sampling: bool = False
  return_log_prob: bool = False
  enable_prefix_caching: bool = False


class MaxTextConfig(CoreConfig):
  """Top-level MaxText configuration aggregating all sub-configurations.

  Attributes:
      model (ModelConfig): Model architecture configuration.
      checkpoint (CheckpointConfig): Checkpointing config.
      optimizer (OptimizerConfig): Optimizer parameters.
      dataset (DatasetConfig): Dataset input parameters.
      tokenizer (TokenizerConfig): Tokenizer settings.
      parallelism (ParallelismConfig): Parallelism and sharding.
      inference (InferenceConfig): Inference-specific options.
      hardware (str): Hardware type (e.g., "tpu").
      steps (int): Number of training steps.
      learning_rate (float): Learning rate for training.
      dropout_rate (float): Dropout rate used.
      gradient_clipping_threshold (float): Threshold for gradient clipping.
      gradient_accumulation_steps (int): Accumulation steps for gradient.
      log_period (int): Interval steps for logging.
      use_dpo (bool): Use Direct Preference Optimization.
      dpo_label_smoothing (float): Label smoothing for DPO.
      dpo_beta (float): Beta parameter for DPO loss.
      use_sft (bool): Use Supervised Fine Tuning.
      sft_train_on_completion_only (bool): Train only on completion tokens.
  """

  model: ModelConfig = ModelConfig()
  checkpoint: CheckpointConfig = CheckpointConfig()
  optimizer: OptimizerConfig = OptimizerConfig()
  dataset: DatasetConfig = DatasetConfig()
  tokenizer: TokenizerConfig = TokenizerConfig()
  parallelism: ParallelismConfig = ParallelismConfig()
  inference: InferenceConfig = InferenceConfig()

  hardware: str = "tpu"
  steps: int = 150_001
  learning_rate: float = 3e-5
  dropout_rate: float = 0.0
  gradient_clipping_threshold: float = 1.0
  gradient_accumulation_steps: int = 1
  log_period: int = 100

  use_dpo: bool = False
  dpo_label_smoothing: float = 0.0
  dpo_beta: float = 0.1

  use_sft: bool = False
  sft_train_on_completion_only: bool = False


class DPOConfig(MaxTextConfig):
  """Configuration class customized for Direct Preference Optimization (DPO).

  Attributes:
      base_config (str): Path to base config.
      use_dpo (bool): Enables DPO.
      train_data_columns (list[str]): Columns specific for DPO training.
      eval_data_columns (list[str]): Columns specific for DPO evaluation.
      base_output_directory (str): Output directory path.
      per_device_batch_size (float): Batch size per device.
      steps (int): Number of steps to run.
      max_target_length (int): Max sequence target length.
      eval_interval (int): Interval between evaluation runs.
      eval_steps (int): Number of evaluation steps.
      dataset_type (str): Dataset pipeline type.
      dataset_path (str): Dataset path.
      dataset_name (str): Name/version of train dataset.
      eval_dataset_name (str): Name/version of eval dataset.
      eval_split (str): Evaluation split.
      hf_eval_split (str): Huggingface eval split (if applicable).
      gradient_clipping_threshold (float): Gradient clipping threshold.
      learning_rate (float): Learning rate.
      dpo_label_smoothing (float): Label smoothing (DPO).
      dpo_beta (float): Beta parameter (DPO).
      enable_goodput_recording (bool): Enable goodput recordings.
      monitor_goodput (bool): Monitor goodput during training.
      enable_checkpointing (bool): Enable checkpointing.
  """

  base_config: str = "base.yml"
  use_dpo: bool = True
  train_data_columns: list[str] = Field(default_factory=lambda: ["chosen", "rejected"])
  eval_data_columns: list[str] = Field(default_factory=lambda: ["chosen", "rejected"])
  base_output_directory: str = "gs://maxtext-external/logs"
  per_device_batch_size: float = 2.0
  steps: int = 10
  max_target_length: int = 512
  eval_interval: int = 5
  eval_steps: int = 2
  dataset_type: str = "tfds"
  dataset_path: str = "gs://maxtext-dataset/dpo/anthropic_rlhf"
  dataset_name: str = "tfds:1.0.0"
  eval_dataset_name: str = "tfds:1.0.0"
  eval_split: str = "test"
  hf_eval_split: str = "test"
  gradient_clipping_threshold: float = 10.0
  learning_rate: float = 5e-7
  dpo_label_smoothing: float = 0.0
  dpo_beta: float = 0.1
  enable_goodput_recording: bool = False
  monitor_goodput: bool = False
  enable_checkpointing: bool = True


class GPUConfig(MaxTextConfig):
  """GPU specific configuration overrides for smoke test or training on GPU.

  Attributes:
      base_config (str): Path to base config file.
      hardware (str): Hardware is 'gpu'.
      attention (str): Type of attention mechanism.
      base_emb_dim (int): Model embedding dim.
      base_num_query_heads (int): Number of query heads.
      base_num_kv_heads (int): Number of key-value heads.
      base_mlp_dim (int): Dimensionality of mlp/intermediate.
      base_num_decoder_layers (int): Num decoder layers.
      head_dim (int): Head dimension.
      per_device_batch_size (float): Batch size per device.
      max_target_length (int): Max sequence length for training.
      dataset_type (str): Dataset platform.
      steps (int): Number of training steps.
  """

  base_config: str = "base.yml"
  hardware: str = "gpu"
  attention: str = "dot_product"
  base_emb_dim: int = 8
  base_num_query_heads: int = 4
  base_num_kv_heads: int = 4
  base_mlp_dim: int = 32
  base_num_decoder_layers: int = 8
  head_dim: int = 16
  per_device_batch_size: float = 2
  max_target_length: int = 1024
  dataset_type: str = "synthetic"
  steps: int = 10


class SFTConfig(MaxTextConfig):
  """Supervised Fine-Tuning (SFT) configuration.

  Attributes:
      base_config (str): Path to base config.
      use_sft (bool): Enable SFT.
      sft_train_on_completion_only (bool): Train only on completion tokens.
      packing (bool): Enable packing.
      learning_rate (float): Learning rate for fine-tuning.
      dataset_type (str): Dataset type (hf pipeline).
      hf_path (str): Huggingface dataset path.
      train_split (str): Train split.
      hf_eval_split (str): Huggingface evaluation split.
      train_data_columns (list[str]): Training data columns.
      eval_data_columns (list[str]): Eval data columns.
  """

  base_config: str = "base.yml"
  use_sft: bool = True
  sft_train_on_completion_only: bool = True
  packing: bool = True
  learning_rate: float = 2e-5
  dataset_type: str = "hf"
  hf_path: str = "HuggingFaceH4/ultrachat_200k"
  train_split: str = "train_sft"
  hf_eval_split: str = "test_sft"
  train_data_columns: list[str] = Field(default_factory=lambda: ["messages"])
  eval_data_columns: list[str] = Field(default_factory=lambda: ["messages"])


__all__ = [
    "CheckpointConfig",
    "CoreConfig",
    "DPOConfig",
    "DatasetConfig",
    "GPUConfig",
    "InferenceConfig",
    "MaxTextConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParallelismConfig",
    "SFTConfig",
    "TokenizerConfig",
]
