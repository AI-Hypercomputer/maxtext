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

"""Scoped, strictly validated Pydantic v2 `TrainingConfig` for m3 trainers.

Composes shared leaf models and enums from `maxtext.configs.types` (`DType`, `DatasetType`,
`LearningRateScheduleType`, `MatmulPrecision`, `OptimizerType`, `QuantizationType`, `RematPolicy`,
`AOT`, `Goodput`, `ManagedMLDiagnostics`, `OrbaxStorage`, `Profiling`, `Tensorboard`) without
calling `jax.devices()` or mutating global JAX configuration at instantiation time.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from maxtext.configs.types import (
    AOT,
    DType,
    DatasetType,
    Goodput,
    LearningRateScheduleType,
    ManagedMLDiagnostics,
    MatmulPrecision,
    OptimizerType,
    OrbaxStorage,
    Profiling,
    QuantizationType,
    RematPolicy,
    Tensorboard,
)


class MeshConfig(BaseModel):
  """Hardware-neutral 5-axis ICI and DCN mesh sizing over `('data', 'fsdp', 'context', 'tensor', 'expert')`."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  ici: dict[str, int] = Field(
      default_factory=lambda: {"fsdp": -1},
      description="Per-slice ICI mesh axis sizes (at most one -1 auto axis).",
  )
  dcn: dict[str, int] = Field(
      default_factory=lambda: {"data": -1},
      description="Cross-slice DCN mesh axis sizes (at most one -1 auto axis).",
  )
  shard_mode: Literal["auto", "explicit"] = Field("auto", description="Mesh axis sharding mode.")


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
  quantization: QuantizationType = Field(QuantizationType.NONE, description="Post-construction quantization recipe.")
  optimizer_memory_host_offload: bool = Field(False, description="Offload optimizer state to host RAM.")
  parameter_memory_host_offload: bool = Field(False, description="Offload master parameters to host RAM.")


class DataPipelineConfig(BaseModel):
  """Deterministic data loading, sequence length, and pre-shift document packing configuration."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  dataset_type: DatasetType = Field(DatasetType.SYNTHETIC, description="Input pipeline backend.")
  dataset_path: str = Field("", description="Dataset root path or GCS URI.")
  dataset_name: str = Field("", description="Dataset name or split identifier.")
  grain_train_files: str = Field("", description="Grain ArrayRecord / Parquet glob pattern for training.")
  grain_eval_files: str = Field("", description="Grain ArrayRecord / Parquet glob pattern for evaluation.")
  grain_worker_count: int = Field(1, ge=0, description="Number of Grain worker processes.")
  tokenizer_path: str = Field("", description="Tokenizer identifier or path.")
  hf_access_token: SecretStr | None = Field(None, description="Redacted HuggingFace access token.")
  packing: bool = Field(True, description="Enable pre-shift intra-document causal bin-packing.")
  grain_packing_type: Literal["first_fit", "best_fit"] = Field(
      "first_fit",
      description="Deterministic document bin-packing heuristic.",
  )
  pad_id: int = Field(0, ge=0, description="Padding token ID placed in unfilled bin suffixes.")
  data_shuffle_seed: int = Field(42, description="Deterministic epoch shuffle seed.")


# Mapping of leaf sub-config attribute names to their Pydantic model classes for flat override routing.
_NESTED_SUBCONFIG_FIELDS: dict[str, type[BaseModel]] = {
    "mesh": MeshConfig,
    "transforms": TransformConfig,
    "data": DataPipelineConfig,
    "aot": AOT,
    "goodput": Goodput,
    "diagnostics": ManagedMLDiagnostics,
    "orbax_storage": OrbaxStorage,
    "profiling": Profiling,
    "tensorboard": Tensorboard,
}


class TrainingConfig(BaseModel):
  """Pure-Python, frozen, strictly validated runtime configuration for `M3Trainer`.

  Contains zero model-architecture fields (those live on `ModelConfig`) and triggers zero
  `jax.devices()` or global `jax.config` side effects during instantiation.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)

  # Run identity & output paths
  run_name: str = Field("m3_run", description="Logical run name for checkpoints and TensorBoard.")
  base_output_directory: str = Field("", description="Root output directory or GCS bucket.")
  seed: int = Field(0, description="PRNG seed for parameter initialization and dropout.")

  # Batching, sequence length, and training horizon
  steps: int = Field(100, gt=0, description="Total optimizer update steps.")
  max_target_length: int = Field(2048, gt=0, description="Sequence length S per packed training example.")
  per_device_batch_size: float = Field(1.0, gt=0.0, description="Per-device microbatch size.")
  gradient_accumulation_steps: int = Field(1, gt=0, description="Number of microbatches scanned per step.")

  # Precision & loss
  param_dtype: DType = Field(DType.FLOAT32, description="Master parameter dtype.")
  compute_dtype: DType = Field(DType.BFLOAT16, description="Forward/backward compute dtype.")
  grad_dtype: DType = Field(DType.FLOAT32, description="Gradient accumulation dtype.")
  matmul_precision: MatmulPrecision = Field(MatmulPrecision.DEFAULT, description="XLA default matmul precision.")
  num_vocab_tiles: int = Field(
      1,
      ge=1,
      description="Number of vocabulary tiles for `jax.checkpoint`-scanned cross-entropy (`>1` skips dense logits).",
  )
  z_loss_multiplier: float = Field(0.0, ge=0.0, description="Log-Z auxiliary stabilization loss coefficient.")
  aux_loss_weight: float = Field(1.0, ge=0.0, description="Multiplier applied to `CausalLMOutput.aux_loss`.")
  router_bias_update_rate: float = Field(
      0.0,
      ge=0.0,
      description="Step size for post-optimizer auxiliary-loss-free `RouterBias` updates.",
  )

  # Optimizer & LR schedule
  optimizer_type: OptimizerType = Field(OptimizerType.ADAMW, description="Optimizer algorithm.")
  learning_rate: float = Field(3e-4, ge=0.0, description="Peak learning rate.")
  lr_schedule_type: LearningRateScheduleType | Literal["constant", "linear"] = Field(
      "constant",
      description="Learning rate decay schedule ('constant', 'cosine', 'wsd', 'linear').",
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
  adam_weight_decay: float = Field(0.1, ge=0.0, description="Decoupled weight decay.")
  gradient_clipping_threshold: float = Field(1.0, ge=0.0, description="Global gradient L2 norm clip (0 disables).")
  freeze_vision_encoder: bool = Field(False, description="Mask vision tower parameters to zero update in Optax.")

  # Logging, evaluation, & checkpointing cadence
  log_period: int = Field(10, gt=0, description="Step interval for host metric logging.")
  eval_interval: int = Field(0, ge=0, description="Step interval for evaluation (`0` disables periodic eval).")
  eval_steps: int = Field(1, gt=0, description="Number of batches per evaluation run.")
  enable_checkpointing: bool = Field(False, description="Enable Orbax checkpoint saving and auto-resume.")
  checkpoint_period: int = Field(100, gt=0, description="Step interval between Orbax checkpoints.")
  load_checkpoint_path: str = Field("", description="Explicit Orbax checkpoint directory to restore at startup.")

  # Scoped nested configurations (including shared `maxtext.configs.types` leaf models)
  mesh: MeshConfig = Field(default_factory=MeshConfig)
  transforms: TransformConfig = Field(default_factory=TransformConfig)
  data: DataPipelineConfig = Field(default_factory=DataPipelineConfig)
  aot: AOT = Field(default_factory=AOT)
  goodput: Goodput = Field(default_factory=Goodput)
  diagnostics: ManagedMLDiagnostics = Field(default_factory=ManagedMLDiagnostics)
  orbax_storage: OrbaxStorage = Field(default_factory=OrbaxStorage)
  profiling: Profiling = Field(default_factory=Profiling)
  tensorboard: Tensorboard = Field(default_factory=Tensorboard)

  @model_validator(mode="before")
  @classmethod
  def _route_flat_leaf_overrides(cls, raw: Any) -> Any:
    """Routes flat kwargs belonging to a single nested leaf sub-config into that sub-config dict.

    Keeps Python ergonomics flat when unambiguous (e.g. `TrainingConfig(profiler='xplane')` or
    `TrainingConfig(scan_layers=False)`) while rejecting unknown keys via `extra='forbid'`.
    """
    if not isinstance(raw, dict):
      return raw
    top_fields = set(cls.model_fields.keys())
    routed = dict(raw)
    nested_updates: dict[str, dict[str, Any]] = {}

    for key in list(routed.keys()):
      if key in top_fields:
        continue
      owners = [sub_name for sub_name, sub_cls in _NESTED_SUBCONFIG_FIELDS.items() if key in sub_cls.model_fields]
      if len(owners) == 1:
        owner = owners[0]
        nested_updates.setdefault(owner, {})[key] = routed.pop(key)

    for owner, updates in nested_updates.items():
      existing = routed.get(owner)
      if isinstance(existing, BaseModel):
        merged = existing.model_dump()
        merged.update(updates)
        routed[owner] = merged
      elif isinstance(existing, dict):
        merged = dict(updates)
        merged.update(existing)
        routed[owner] = merged
      else:
        routed[owner] = updates
    return routed

  def to_json(self, *, indent: int = 2) -> str:
    """Serializes the validated config to JSON with `SecretStr` fields redacted."""
    return self.model_dump_json(indent=indent)
