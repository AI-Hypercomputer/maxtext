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

"""Reusable config components: enums and leaf Pydantic sections shared across MaxText config schemas.

Used by `maxtext.configs.types.MaxTextConfig` and by m3's `TrainingConfig`. `types.py` imports the names it uses from
here, so existing imports such as `from maxtext.configs.types import DType` keep working; `ProfilingCore` and
`XProfTPUPowerTraceMode` are only used by `Profiling` and are available from this module only. Keep this module
dependency-free: only the standard library and `pydantic`, never `jax` or `maxtext.utils`, so lightweight configs can
import it without pulling in the training runtime.
"""

import enum
from enum import Enum

from pydantic.fields import Field
from pydantic.main import BaseModel

PathStr = str


class XProfTPUPowerTraceMode(enum.IntEnum):
  """Enum for XProfTPUPowerTraceMode."""

  POWER_TRACE_NONE = 0
  POWER_TRACE_NORMAL = 1
  POWER_TRACE_SPI = 2


class DType(str, Enum):
  """Supported data types for weights and activations."""

  BFLOAT16 = "bfloat16"
  FLOAT32 = "float32"
  FLOAT16 = "float16"
  FLOAT8_E4M3FN = "float8_e4m3fn"
  FLOAT8_E5M2 = "float8_e5m2"


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
  FP4 = "fp4"
  INT4 = "int4"
  INT8 = "int8"
  INTMP = "intmp"
  FP8_E5M2 = "fp8_e5m2"
  FP8_E4M3 = "fp8_e4m3"
  FP8 = "fp8"
  NANOO_FP8 = "nanoo_fp8"
  FP8_NANO_V2 = "fp8_nanoo"
  FP8_GPU = "fp8_gpu"
  FP8_FULL = "fp8_full"
  TE_NO_QUANT = "te_no_quant"
  TE_FP8_DS = "te_fp8_delayedscaling"
  TE_FP8_CS = "te_fp8_currentscaling"
  TE_MXFP8 = "te_mxfp8"
  JAX_MXFP8 = "jax_mxfp8"
  JAX_MXFP4 = "jax_mxfp4"
  TE_NVFP4 = "te_nvfp4"
  TE_NVFP4_NO_RHT = "te_nvfp4_no_rht"
  SERVE_FP8_WEIGHT = "serve_fp8_weight"
  MULTIPASS = "multipass"


class RematPolicy(str, Enum):
  """Available rematerialization (gradient checkpointing) policies."""

  NONE = "none"
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


class LearningRateScheduleType(str, Enum):
  """Supported learning rate schedule types."""

  COSINE = "cosine"
  WSD = "wsd"


class WsdDecayStyle(str, Enum):
  """Supported decay styles for WSD schedule."""

  LINEAR = "linear"
  COSINE = "cosine"


class DatasetType(str, Enum):
  """Supported data loading pipelines."""

  SYNTHETIC = "synthetic"
  HF = "hf"
  GRAIN = "grain"
  TFDS = "tfds"
  C4MLPERF = "c4_mlperf"
  OLMO_GRAIN = "olmo_grain"


class ProfilerType(str, Enum):
  """Supported performance profilers."""

  NONE = ""
  XPLANE = "xplane"
  NSYS = "nsys"


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
  # Concurrent GB limit for device->host staging during checkpoint saves.
  # When set, bounds in-flight bytes staged from accelerator to host RAM per handler.
  # None restores the Orbax upstream default (unbounded).
  checkpoint_storage_device_host_concurrent_gb: int | None = Field(
      8, description="Concurrent GB for device->host staging during checkpoint save. None = unbounded."
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
  adamw_mask: list[str] = Field(
      default_factory=list,
      description=(
          "List of parameter names/patterns to exclude from weight decay in AdamW," " like ['bias', '.*norm', '.*ln.*']"
      ),
  )
  mu_dtype: str = Field(
      "",
      description="Data type for 'mu' (first moment) in AdamW. Inherits from weight_dtype if empty.",
  )


class AOT(BaseModel):
  """Ahead of Time (AOT) Compilation settings."""

  compiled_trainstep_file: PathStr = Field("", description="Name of saved serialized compiled train_step.")
  compile_topology: str = Field("", description="Target hardware version, e.g. 'v5e-256'.")
  compile_topology_num_slices: int = Field(-1, description="Number of target slices.")
  write_estimator_result: bool = Field(False, description="Write estimator.py results in a separate file.")


class ProfilingCore(BaseModel):
  """Profiler backend selection shared by every trainer."""

  profiler: ProfilerType = Field(ProfilerType.NONE, description="Profiler to use ('xplane', 'nsys').")


class Profiling(ProfilingCore):
  """Configuration for step-window performance profiling (`common/profiler.py`)."""

  upload_all_profiler_results: bool = Field(False, description="Upload profiler results from all hosts.")
  skip_first_n_steps_for_profiler: int = Field(1, description="Number of initial steps to skip for profiling.")
  profiler_steps: int = Field(5, description="Number of steps to profile.")
  profile_cleanly: bool = Field(True, description="Add block_until_ready to align profile for each step.")
  profile_periodically_period: int = Field(-1, description="If positive, profile every N steps.")
  hide_profiler_step_metric: bool = Field(False, description="Whether to enable profiler step metric.")
  enable_continuous_profiling: bool = Field(False, description="If true, it will support saving profile > 2GB.")
  enable_jax_profiler: bool = Field(False, description="Enable the JAX live profiler.")
  jax_profiler_port: int = Field(9999, description="Port for the JAX profiler.")
  enable_tpu_profiling_options: bool = Field(False, description="Enable TPU advanced profiling options.")
  tpu_num_chips_to_profile_per_task: int = Field(1, description="Specifies the number of TPU chips to profile per task.")
  tpu_num_sparse_cores_to_trace: int = Field(2, description="Specifies the number of TPU chips to profile per task.")
  tpu_num_sparse_core_tiles_to_trace: int = Field(
      1,
      description="Specifies the number of tiles within each sparse core to trace on the TPU.",
  )
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
  profile_power_events: bool = Field(
      False,
      description="Enable TPU-specific power/thermal profiling events."
      " Defaults to False to avoid breaking GPU xplane tracing.",
  )
