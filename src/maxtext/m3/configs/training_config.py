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

from maxtext.configs.components import (
    AOT,
    AdamW,
    DType,
    DatasetType,
    LearningRateScheduleType,
    MatmulPrecision,
    OrbaxStorage,
    Profiling,
    QuantizationType,
    RematPolicy,
    WsdDecayStyle,
)
from maxtext.m3.models.model_config import ModelConfig
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SecretStr, StrictInt, field_validator, model_validator


class _FrozenConfig(BaseModel):
  """Base for m3 runtime configs: immutable, hashable, and rejects unknown keys (`extra="forbid"`)."""

  model_config = ConfigDict(extra="forbid", frozen=True)


# Mesh axes `m3/core/sharding.py` builds and shards over (`DEFAULT_SHARDING_AXES`, in its mesh order). `stage` is held
# at 1 until a pipeline transform exists, so it is not configurable here. Not imported from `m3.core.sharding` to keep
# this module free of `jax`; `training_config_test.py` asserts the two stay in sync.
MESH_AXES: tuple[str, ...] = ("data", "fsdp", "fsdp_transpose", "context", "tensor", "tensor_sequence", "expert")


class MeshAxes(_FrozenConfig):
  """Immutable, hashable sizes for the m3 mesh axes in `MESH_AXES` (`-1` = auto-fill, at most one per mesh).

  Accepts dict input (`ici={"fsdp": -1, "tensor": 2}`). Axes m3 does not shard over (e.g. `stage`, `sequence`,
  `pipeline`) are rejected by `extra="forbid"`, and `StrictInt` rejects `bool` sizes. Model families reject axes they
  do not implement yet via `ModelConfig.supported_mesh_axes()` (see `validate_training_config`).
  """

  data: StrictInt = 1
  fsdp: StrictInt = 1
  fsdp_transpose: StrictInt = 1
  context: StrictInt = 1
  tensor: StrictInt = 1
  tensor_sequence: StrictInt = 1
  expert: StrictInt = 1

  @field_validator(*MESH_AXES)
  @classmethod
  def _validate_size(cls, size: int) -> int:
    if size != -1 and size < 1:
      raise ValueError(f"mesh axis size must be -1 or a positive integer; got {size}.")
    return size

  @model_validator(mode="after")
  def _at_most_one_auto_axis(self) -> "MeshAxes":
    auto_axes = [name for name, size in self.model_dump().items() if size == -1]
    if len(auto_axes) > 1:
      raise ValueError(f"at most one -1 auto axis is allowed; got {auto_axes}.")
    return self


class IciMeshAxes(MeshAxes):
  """Per-slice ICI axes; `fsdp` auto-fills by default (`base.yml`: `ici_fsdp_parallelism: -1`)."""

  fsdp: StrictInt = -1


class DcnMeshAxes(MeshAxes):
  """Cross-slice DCN axes; `data` auto-fills by default (`base.yml`: `dcn_data_parallelism: -1`)."""

  data: StrictInt = -1


class MeshConfig(_FrozenConfig):
  """ICI and DCN mesh sizing over `MESH_AXES`, plus the physical device-layout options `m3/core/sharding.py` supports.

  Partial dicts merge onto the per-mesh defaults, so `ici={"tensor": 4}` keeps `fsdp=-1` like `base.yml`.
  """

  ici: IciMeshAxes = Field(
      default_factory=IciMeshAxes,
      description="Per-slice ICI mesh axis sizes (at most one -1 auto axis).",
  )
  dcn: DcnMeshAxes = Field(
      default_factory=DcnMeshAxes,
      description="Cross-slice DCN mesh axis sizes (at most one -1 auto axis).",
  )
  shard_mode: Literal["explicit", "auto"] = Field(
      "auto",
      description="JAX mesh axis type used for sharding ('auto' or 'explicit').",
  )
  shard_optimizer_over_data: bool = Field(
      False,
      description="Shard optimizer state across the data-parallel axis (ZeRO-1).",
  )
  custom_mesh: Literal["", "hybrid_ring_64x4", "hybrid_ring_32x8"] = Field(
      "",
      description="Hand-tuned physical device ordering ('' for the default topology-aware layout).",
  )
  allow_split_physical_axes: bool = Field(False, description="Allow splitting physical axes for device mesh creation.")
  optimize_mesh_for_tpu_v6e: bool = Field(False, description="Reorder devices to align the mesh with TPU v6e topology.")


# `components` leaf models are shared with the mutable `MaxTextConfig`, so they stay mutable + `extra="ignore"`.
# Mix in `_FrozenConfig` here so typos fail fast and `TrainingConfig` stays deeply immutable.


class AOTConfig(_FrozenConfig, AOT):
  """Frozen, strictly validated view of `maxtext.configs.components.AOT`."""


class OrbaxStorageConfig(_FrozenConfig, OrbaxStorage):
  """Frozen, strictly validated view of `maxtext.configs.components.OrbaxStorage`."""


class ProfilingConfig(_FrozenConfig, Profiling):
  """Frozen, strictly validated view of `maxtext.configs.components.Profiling`."""


class TransformConfig(_FrozenConfig):
  """Post-construction infrastructure transform configuration (`remat`, `scan`, `quantization`, `offload`)."""

  remat_policy: RematPolicy = Field(
      RematPolicy.FULL,
      description="Activation checkpointing policy applied per decoder layer ('none' disables remat).",
  )
  remat_save_names: tuple[str, ...] = Field(
      (),
      description="Checkpoint tag names kept on device when `remat_policy='custom'`.",
  )
  scan_layers: bool = Field(True, description="Stack homogeneous decoder layers into `ScannedLayers`.")
  named_scopes: bool = Field(True, description="Wrap transformed layers in JAX named scopes for profiling.")
  quantization: QuantizationType = Field(QuantizationType.NONE, description="Post-construction quantization recipe.")
  optimizer_memory_host_offload: bool = Field(False, description="Offload optimizer state to host RAM.")


class OptimizerConfig(_FrozenConfig, AdamW):
  """Optimizer algorithm, learning-rate schedule, clipping, and parameter masking configuration.

  AdamW hyperparameters (`adam_b1`, `adam_b2`, `adam_eps`, `adam_eps_root`, `adam_weight_decay`, `adamw_mask`,
  `mu_dtype`) are inherited from `maxtext.configs.components.AdamW` so defaults and descriptions have one source of truth.
  """

  opt_type: Literal["adamw", "adam_pax", "sgd"] = Field(
      "adamw",
      description="Supported v1 optimizer algorithm ('adamw', 'adam_pax', or 'sgd').",
  )
  learning_rate: float = Field(3e-5, ge=0.0, description="Peak learning rate.")
  lr_schedule_type: LearningRateScheduleType = Field(
      LearningRateScheduleType.COSINE,
      description="Learning rate decay schedule ('cosine' or 'wsd').",
  )
  learning_rate_schedule_steps: PositiveInt | None = Field(
      None,
      description="Length of the learning-rate schedule in steps. If None, matches `steps`.",
  )
  warmup_steps_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of steps spent in linear warmup.")
  learning_rate_final_fraction: float = Field(
      0.1,
      ge=0.0,
      le=1.0,
      description="Final learning rate as a fraction of peak `learning_rate` (applies to cosine and WSD schedules).",
  )
  wsd_decay_steps_fraction: float = Field(
      0.1,
      ge=0.0,
      le=1.0,
      description="Fraction of total steps for the decay phase of the WSD schedule.",
  )
  wsd_decay_style: WsdDecayStyle = Field(
      WsdDecayStyle.LINEAR,
      description="Decay style for the WSD schedule ('linear' or 'cosine').",
  )
  # Redeclared from `AdamW` as a tuple (not `list`) so the frozen config stays hashable.
  adamw_mask: tuple[str, ...] = Field(
      (),
      description="Regex patterns selecting parameter paths excluded from AdamW weight decay.",
  )
  gradient_clipping_threshold: float = Field(1.0, ge=0.0, description="Global gradient L2 norm clip (0 disables).")
  trainable_parameters_mask: tuple[str, ...] = Field(
      (),
      description="Regex patterns restricting optimizer updates to matching parameter paths.",
  )
  freeze_vision_encoder_params: bool = Field(True, description="Mask vision tower parameters to zero update in Optax.")

  @field_validator("mu_dtype", mode="before")
  @classmethod
  def _validate_mu_dtype(cls, value: Any) -> Any:
    """Accepts `""` (inherit `ModelConfig.param_dtype`, matching `base.yml`) or a supported `DType` value."""
    value = getattr(value, "value", value)
    if value and value not in {d.value for d in DType}:
      raise ValueError(f"`mu_dtype` must be '' or one of {sorted(d.value for d in DType)}; got {value!r}.")
    return value

  @model_validator(mode="after")
  def _validate_hyperparameters(self) -> "OptimizerConfig":
    """Validates that AdamW hyperparameters are in valid ranges and that WSD fractions do not exceed 1.0."""
    if not (0.0 <= self.adam_b1 < 1.0 and 0.0 <= self.adam_b2 < 1.0):
      raise ValueError(f"`adam_b1` ({self.adam_b1}) and `adam_b2` ({self.adam_b2}) must be in [0, 1).")
    if self.adam_eps <= 0.0 or self.adam_eps_root < 0.0 or self.adam_weight_decay < 0.0:
      raise ValueError("`adam_eps` must be > 0; `adam_eps_root` and `adam_weight_decay` must be >= 0.")
    if (
        self.lr_schedule_type == LearningRateScheduleType.WSD
        and self.warmup_steps_fraction + self.wsd_decay_steps_fraction > 1.0
    ):
      raise ValueError(
          f"warmup_steps_fraction ({self.warmup_steps_fraction}) + wsd_decay_steps_fraction"
          f" ({self.wsd_decay_steps_fraction}) must not exceed 1.0."
      )
    return self


class CheckpointConfig(_FrozenConfig):
  """Orbax checkpointing configuration."""

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
  checkpoint_todelete_full_path: str | None = Field(
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


class MetricsConfig(_FrozenConfig):
  """Training metric logging, TensorBoard output, and pipelined NaN/Inf abort controls."""

  log_period: int = Field(100, gt=0, description="Step interval for host metric logging.")
  enable_tensorboard: bool = Field(True, description="Write scalar summaries to TensorBoard.")
  metrics_file: str = Field("", description="Optional JSONL file path for local metric dumps.")
  gcs_metrics: bool = Field(False, description="Flush scalar metrics to the GCS output directory.")
  abort_on_nan_loss: bool = Field(True, description="Halt training when the loss becomes NaN.")
  abort_on_inf_loss: bool = Field(True, description="Halt training when the loss becomes infinite.")


class DataPipelineConfig(_FrozenConfig):
  """Input pipeline configuration."""

  dataset_type: DatasetType = Field(DatasetType.SYNTHETIC, description="Input pipeline backend.")
  dataset_path: str = Field("", description="Dataset root path or GCS URI.")
  dataset_name: str = Field("c4/en:3.1.0", description="Dataset name or split identifier.")
  grain_train_files: str = Field("", description="Grain ArrayRecord / Parquet glob pattern for training.")
  grain_eval_files: str = Field("", description="Grain ArrayRecord / Parquet glob pattern for evaluation.")
  grain_worker_count: int = Field(1, ge=-1, description="Number of Grain worker processes (-1 enables auto-tuning).")
  tokenizer_path: str = Field("", description="Tokenizer identifier or path.")
  hf_access_token: SecretStr | None = Field(None, description="Redacted HuggingFace access token.")
  packing: bool = Field(True, description="Enable sequence packing in the underlying MaxText data loader.")
  grain_packing_type: Literal["first_fit", "best_fit", "concat_then_split"] = Field(
      "first_fit",
      description="Grain bin-packing heuristic.",
  )
  pad_id: int = Field(0, ge=0, description="Padding token ID.")
  data_shuffle_seed: int = Field(0, description="Deterministic epoch shuffle seed.")


class TrainingConfig(_FrozenConfig):
  """Runtime configuration for m3 trainer."""

  # Run identity & output paths
  run_name: str = Field("", description="Logical run name for checkpoints and TensorBoard.")
  base_output_directory: str = Field("", description="Base output directory.")
  init_weights_seed: int = Field(0, description="Seed for model weight initialization.")

  # Batching, sequence length, and training horizon
  steps: PositiveInt | None = Field(
      150001,
      description="Number of optimizer steps to train for. If None, uses `optimizer.learning_rate_schedule_steps`.",
  )
  max_target_length: int = Field(2048, gt=0, description="Sequence length per training example.")
  per_device_batch_size: float = Field(12.0, gt=0.0, description="Per-device microbatch size.")
  gradient_accumulation_steps: int = Field(1, gt=0, description="Number of microbatches scanned per step.")
  eval_interval: PositiveInt | None = Field(
      None,
      description="Run evaluation every N training steps. If None, periodic evaluation is disabled.",
  )
  eval_steps: PositiveInt | None = Field(
      None,
      description="Number of eval batches per evaluation. If None, evaluates the full eval split.",
  )

  # Precision, loss, & step-skipping stability (`param_dtype` / `compute_dtype` live on `ModelConfig`)
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
  routed_bias_update_rate: float = Field(
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
  aot: AOTConfig = Field(default_factory=AOTConfig)
  orbax_storage: OrbaxStorageConfig = Field(default_factory=OrbaxStorageConfig)
  profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)

  @model_validator(mode="after")
  def _validate_steps(self) -> "TrainingConfig":
    if self.steps is None and self.optimizer.learning_rate_schedule_steps is None:
      raise ValueError("`steps` and `optimizer.learning_rate_schedule_steps` cannot both be None.")
    return self


def validate_training_config(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    mesh_shape: Mapping[str, int] | None = None,
) -> None:
  """Validates cross-config invariants: vocab tiling, per-family mesh-axis support, and mesh divisibility.

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

  supported = model_cfg.supported_mesh_axes()
  unsupported = {
      f"{network}.{axis}": size
      for network, axes in (("ici", train_cfg.mesh.ici), ("dcn", train_cfg.mesh.dcn))
      for axis, size in axes.model_dump().items()
      if size != 1 and axis not in supported  # `-1` auto-fill counts as active.
  }
  if unsupported:
    raise ValueError(
        f"{type(model_cfg).__name__} does not support mesh axes {unsupported}; supported axes: {sorted(supported)}."
    )

  if mesh_shape is None:
    return

  for axis_name, dims in model_cfg.mesh_divisibility_constraints().items():
    axis_size = int(mesh_shape.get(axis_name, 1))
    if axis_size <= 1:
      continue
    for field_name, value in dims.items():
      if value % axis_size != 0:
        raise ValueError(f"{field_name} ({value}) must be divisible by the {axis_name!r} mesh axis size ({axis_size}).")
