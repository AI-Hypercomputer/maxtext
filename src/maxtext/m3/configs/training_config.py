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

"""Validated Pydantic v2 `TrainingConfig` for m3 trainers."""

from collections.abc import Mapping
from typing import Any, Literal

from maxtext.configs.types import (
    AOT,
    DType,
    DatasetType,
    LearningRateScheduleType,
    MatmulPrecision,
    OrbaxStorage,
    Profiling,
    QuantizationType,
    RematPolicy,
)
from maxtext.m3.models.model_config import ModelConfig
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

ALLOWED_MESH_AXES: frozenset[str] = frozenset({"data", "fsdp", "tensor", "expert"})


def _validate_mesh_axis_dict(axis_map: dict[str, int], label: str) -> None:
  """Validates a single ICI or DCN mesh sizing dictionary over the 4 v1 m3 mesh axes."""
  unknown = sorted(set(axis_map) - ALLOWED_MESH_AXES)
  if unknown:
    raise ValueError(
        f"`{label}` contains unsupported mesh axis name(s) {unknown}; " f"supported axes are {sorted(ALLOWED_MESH_AXES)}."
    )
  auto_axes: list[str] = []
  for axis_name, size in axis_map.items():
    if isinstance(size, bool) or (size != -1 and size < 1):
      raise ValueError(f"`{label}[{axis_name!r}]` must be -1 or a positive integer; got {size!r}.")
    if size == -1:
      auto_axes.append(axis_name)
  if len(auto_axes) > 1:
    raise ValueError(f"`{label}` may specify at most one -1 auto axis; got {sorted(auto_axes)}.")


class MeshConfig(BaseModel):
  """Hardware-neutral 4-axis ICI and DCN mesh sizing over `('data', 'fsdp', 'tensor', 'expert')`."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  ici: dict[str, int] = Field(
      default_factory=lambda: {"fsdp": -1},
      description="Per-slice ICI mesh axis sizes (at most one -1 auto axis).",
  )
  dcn: dict[str, int] = Field(
      default_factory=lambda: {"data": -1},
      description="Cross-slice DCN mesh axis sizes (at most one -1 auto axis).",
  )
  shard_mode: Literal["explicit", "auto"] = Field(
      "auto",
      description="Mesh axis sharding mode ('auto' or 'explicit'; 'explicit' enables ZeRO-1 and deferred all-reduce).",
  )
  shard_optimizer_over_data: bool = Field(
      False,
      description="Shard optimizer state across the data-parallel axis (ZeRO-1).",
  )

  @field_validator("ici", "dcn", mode="before")
  @classmethod
  def _reject_bool_axis_sizes(cls, value: Any) -> Any:
    if isinstance(value, Mapping):
      for axis_name, size in value.items():
        if isinstance(size, bool):
          raise ValueError(f"Mesh axis {axis_name!r} must be -1 or a positive integer; got bool {size!r}.")
    return value

  @model_validator(mode="after")
  def _validate_axes(self) -> "MeshConfig":
    _validate_mesh_axis_dict(self.ici, "ici")
    _validate_mesh_axis_dict(self.dcn, "dcn")
    return self


class TransformConfig(BaseModel):
  """Post-construction infrastructure transform configuration (`remat`, `scan`, `quantization`, `offload`)."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  remat_policy: RematPolicy | Literal["none"] = Field(
      RematPolicy.FULL,
      description="Activation checkpointing policy applied per decoder layer ('none' disables remat).",
  )
  remat_save_names: tuple[str, ...] = Field(
      (),
      description="Checkpoint tag names kept on device when `remat_policy='custom'`.",
  )
  scan_layers: bool = Field(True, description="Stack homogeneous decoder layers into `ScannedLayers`.")
  param_scan_axis: int = Field(
      1,
      ge=0,
      description="Axis at which `ScannedLayers` stacks `nnx.Param` tensors (default 1 matches scanned checkpoints).",
  )
  named_scopes: bool = Field(True, description="Wrap transformed layers in JAX named scopes for profiling.")
  quantization: QuantizationType = Field(QuantizationType.NONE, description="Post-construction quantization recipe.")
  optimizer_memory_host_offload: bool = Field(False, description="Offload optimizer state to host RAM.")


class OptimizerConfig(BaseModel):
  """Optimizer algorithm, learning-rate schedule, clipping, and parameter masking configuration."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  optimizer_type: Literal["adamw", "adam_pax", "sgd"] = Field(
      "adamw",
      description="Supported v1 optimizer algorithm ('adamw', 'adam_pax', or 'sgd').",
  )
  learning_rate: float = Field(3e-5, ge=0.0, description="Peak learning rate.")
  lr_schedule_type: LearningRateScheduleType | Literal["constant", "linear"] = Field(
      LearningRateScheduleType.COSINE,
      description="Learning rate decay schedule ('cosine', 'wsd', 'constant', 'linear').",
  )
  learning_rate_schedule_steps: int = Field(
      -1,
      ge=-1,
      description="Total steps for the LR decay schedule (-1 defaults to `steps`).",
  )
  warmup_steps_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of steps spent in linear warmup.")
  cosine_learning_rate_final_fraction: float = Field(
      0.1,
      ge=0.0,
      le=1.0,
      description="Final learning rate as a fraction of peak `learning_rate`.",
  )
  adam_b1: float = Field(0.9, ge=0.0, lt=1.0, description="Adam beta1.")
  adam_b2: float = Field(0.95, ge=0.0, lt=1.0, description="Adam beta2.")
  adam_eps: float = Field(1e-8, gt=0.0, description="Adam epsilon.")
  adam_eps_root: float = Field(0.0, ge=0.0, description="Adam epsilon inside the square root.")
  adam_weight_decay: float = Field(0.1, ge=0.0, description="Decoupled weight decay.")
  adamw_mask: tuple[str, ...] = Field(
      (),
      description="Regex patterns selecting parameter paths excluded from AdamW weight decay.",
  )
  mu_dtype: DType = Field(DType.FLOAT32, description="Storage dtype for first-moment optimizer state.")
  gradient_clipping_threshold: float = Field(1.0, ge=0.0, description="Global gradient L2 norm clip (0 disables).")
  trainable_parameters_mask: tuple[str, ...] = Field(
      (),
      description="Regex patterns restricting optimizer updates to matching parameter paths.",
  )
  freeze_vision_encoder: bool = Field(True, description="Mask vision tower parameters to zero update in Optax.")

  @field_validator("optimizer_type", mode="before")
  @classmethod
  def _coerce_optimizer_type(cls, value: Any) -> Any:
    return getattr(value, "value", value)


class CheckpointConfig(BaseModel):
  """Orbax checkpointing configuration."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  enable_checkpointing: bool = Field(True, description="Enable Orbax checkpoint saving and auto-resume.")
  checkpoint_period: int = Field(10000, gt=0, description="Step interval between Orbax checkpoints.")
  max_num_checkpoints_to_keep: int | None = Field(
      None,
      gt=0,
      description="Maximum number of checkpoints retained (`None` keeps all).",
  )
  async_checkpointing: bool = Field(True, description="Save Orbax checkpoints asynchronously.")
  async_checkpointing_timeout_secs: int | None = Field(
      None,
      gt=0,
      description="Optional timeout in seconds for waiting on asynchronous Orbax saves.",
  )
  save_checkpoint_on_start: bool = Field(
      True,
      description="Save an initial checkpoint via `InitialSavePolicy` when starting a run.",
  )
  checkpoint_format: Literal["train_state", "nnx"] = Field(
      "train_state",
      description="On-disk save layout: `'train_state'` (pre-training + Grain `iter`) or `'nnx'` (post-training).",
  )
  load_full_state_path: str = Field(
      "",
      description="Explicit full-state Orbax checkpoint directory (weights, optimizer state, step).",
  )
  load_parameters_path: str = Field(
      "",
      description="Explicit weights-only checkpoint directory (Orbax or `safetensors_dynamic`).",
  )
  source_checkpoint_layout: Literal["orbax", "safetensors_dynamic"] = Field(
      "orbax",
      description="Storage layout for `load_parameters_path` (`'orbax'` or `'safetensors_dynamic'`).",
  )
  enable_single_replica_ckpt_restoring: bool = Field(
      False,
      description="Restore from a single replica and broadcast shards across data-parallel replicas.",
  )
  todelete_full_path: str | None = Field(
      None,
      description="Optional full directory path for Orbax v1 garbage-collected checkpoints.",
  )
  enable_pathways_persistence: bool = Field(
      False,
      description="Require `CloudPathwaysArrayHandler` persistence verification at initialization.",
  )

  @model_validator(mode="after")
  def _validate_restore_paths(self) -> "CheckpointConfig":
    if self.load_full_state_path and self.load_parameters_path:
      raise ValueError("At most one of `load_parameters_path` or `load_full_state_path` should be set.")
    if self.load_full_state_path and self.source_checkpoint_layout == "safetensors_dynamic":
      raise ValueError(
          "`source_checkpoint_layout='safetensors_dynamic'` only supports weights-only `load_parameters_path`."
      )
    return self


class MetricsConfig(BaseModel):
  """Training metric logging, TensorBoard output, and pipelined NaN/Inf abort controls."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  log_period: int = Field(100, gt=0, description="Step interval for host metric logging.")
  enable_tensorboard: bool = Field(True, description="Write scalar summaries to TensorBoard.")
  metrics_file: str | None = Field("", description="Optional JSONL file path for local metric dumps.")
  gcs_metrics: bool = Field(False, description="Flush scalar metrics to the GCS output directory.")
  abort_on_nan_loss: bool = Field(True, description="Halt training when the loss becomes NaN.")
  abort_on_inf_loss: bool = Field(True, description="Halt training when the loss becomes infinite.")


class DataPipelineConfig(BaseModel):
  """Input pipeline configuration."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  dataset_type: DatasetType = Field(DatasetType.SYNTHETIC, description="Input pipeline backend.")
  dataset_path: str = Field("", description="Dataset root path or GCS URI.")
  dataset_name: str = Field("c4/en:3.1.0", description="Dataset name or split identifier.")
  grain_train_files: str = Field("", description="Grain ArrayRecord / Parquet glob pattern for training.")
  grain_eval_files: str = Field("", description="Grain ArrayRecord / Parquet glob pattern for evaluation.")
  grain_worker_count: int = Field(1, ge=0, description="Number of Grain worker processes.")
  tokenizer_path: str = Field("", description="Tokenizer identifier or path.")
  hf_access_token: SecretStr | None = Field(None, description="Redacted HuggingFace access token.")
  packing: bool = Field(True, description="Enable sequence packing in the underlying MaxText data loader.")
  grain_packing_type: Literal["first_fit", "best_fit"] = Field(
      "first_fit",
      description="Grain bin-packing heuristic.",
  )
  pad_id: int = Field(0, ge=0, description="Padding token ID.")
  data_shuffle_seed: int = Field(0, description="Deterministic epoch shuffle seed.")


class TrainingConfig(BaseModel):
  """Runtime configuration for m3 trainer."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  # Run identity & output paths
  run_name: str = Field("", description="Logical run name for checkpoints and TensorBoard.")
  base_output_directory: str = Field("", description="Base output directory.")
  seed: int = Field(0, description="PRNG seed for parameter initialization and dropout.")

  # Batching, sequence length, and training horizon
  steps: int = Field(150001, gt=0, description="Total optimizer update steps.")
  max_target_length: int = Field(2048, gt=0, description="Sequence length per training example.")
  per_device_batch_size: float = Field(12.0, gt=0.0, description="Per-device microbatch size.")
  gradient_accumulation_steps: int = Field(1, gt=0, description="Number of microbatches scanned per step.")
  eval_interval: int = Field(-1, ge=-1, description="Step interval for evaluation (`-1` disables periodic eval).")
  eval_steps: int = Field(-1, ge=-1, description="Number of batches per evaluation run (`-1` runs entire eval split).")

  # Precision, loss, & step-skipping stability
  param_dtype: DType = Field(DType.FLOAT32, description="Master parameter dtype.")
  compute_dtype: DType = Field(DType.BFLOAT16, description="Forward/backward compute dtype.")
  grad_dtype: DType = Field(DType.FLOAT32, description="Gradient accumulation dtype.")
  matmul_precision: MatmulPrecision = Field(MatmulPrecision.DEFAULT, description="XLA default matmul precision.")
  cast_grads_after_all_reduce: bool = Field(
      False,
      description="All-reduce microbatch gradients before casting to `grad_dtype`.",
  )
  num_vocab_tiling: int = Field(
      1,
      ge=1,
      description="Number of batch-sequence tiles for `custom_vjp` cross-entropy (`>1` skips dense logits).",
  )
  z_loss_multiplier: float = Field(0.0, ge=0.0, description="Log-Z auxiliary stabilization loss coefficient.")
  aux_loss_weight: float = Field(1.0, ge=0.0, description="Multiplier applied to `CausalLMOutput.aux_loss`.")
  router_bias_update_rate: float = Field(
      0.0,
      ge=0.0,
      description="Step size for post-optimizer auxiliary-loss-free `RouterBias` updates.",
  )
  skip_step_on_nan: bool = Field(False, description="Skip optimizer update when gradients contain NaN/Inf.")
  skip_step_on_spikes: bool = Field(False, description="Skip optimizer update on loss spikes.")
  skip_step_interval: int = Field(128, gt=0, description="Rolling window length for spike detection.")
  skip_step_scaling_factor: float = Field(6.0, gt=0.0, description="Spike threshold multiplier.")

  mesh: MeshConfig = Field(default_factory=MeshConfig)
  transforms: TransformConfig = Field(default_factory=TransformConfig)
  optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
  checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
  metrics: MetricsConfig = Field(default_factory=MetricsConfig)
  data: DataPipelineConfig = Field(default_factory=DataPipelineConfig)
  aot: AOT = Field(default_factory=AOT)
  orbax_storage: OrbaxStorage = Field(default_factory=OrbaxStorage)
  profiling: Profiling = Field(default_factory=Profiling)

  def to_json(self, *, indent: int = 2) -> str:
    """Serializes the validated config to JSON with `SecretStr` fields redacted."""
    return self.model_dump_json(indent=indent)


def validate_training_config(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    mesh_shape: Mapping[str, int] | None = None,
) -> None:
  """Validates cross-config and mesh divisibility invariants.

  Args:
    model_cfg: Frozen `ModelConfig` subclass for the target model family.
    train_cfg: Frozen `TrainingConfig` for the training run.
    mesh_shape: Optional `{axis_name: size}` mapping (e.g. `mesh.shape`) for divisibility checks.
  """
  if train_cfg.num_vocab_tiling > 1:
    tokens_per_device = train_cfg.per_device_batch_size * train_cfg.max_target_length
    if not tokens_per_device.is_integer() or int(tokens_per_device) % train_cfg.num_vocab_tiling != 0:
      raise ValueError(
          f"per_device_batch_size * max_target_length ({tokens_per_device:g}) must be evenly divisible "
          f"by num_vocab_tiling ({train_cfg.num_vocab_tiling})."
      )

  if mesh_shape is None:
    return

  tp_size = int(mesh_shape.get("tensor", 1))
  if tp_size > 1:
    for attr in ("num_attention_heads", "num_key_value_heads"):
      val = getattr(model_cfg, attr, None)
      if val is not None and val % tp_size != 0:
        raise ValueError(f"{attr} ({val}) must be divisible by tensor parallelism size ({tp_size}).")

  ep_size = int(mesh_shape.get("expert", 1))
  if ep_size > 1:
    for attr in ("n_routed_experts", "num_local_experts"):
      n_experts = getattr(model_cfg, attr, None)
      if n_experts is not None:
        if n_experts % ep_size != 0:
          raise ValueError(f"{attr} ({n_experts}) must be divisible by expert parallelism size ({ep_size}).")
        break
