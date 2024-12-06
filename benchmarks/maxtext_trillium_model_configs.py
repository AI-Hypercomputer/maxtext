"""
 Copyright 2024 Google LLC

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
"""Shared Benchmark config for v6e orchestrations."""

import dataclasses
import typing
import xla_flags_library


@dataclasses.dataclass
class MaxTextModel:
  model_name: str
  model_type: str
  tuning_params: dict[str, typing.Any]
  xla_flags: str


default_basic = MaxTextModel(
    model_name="default-basic",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 1,
        "remat_policy": "full",
        "global_parameter_scale": 1,
        "attention": "flash",
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags="",
)

# Test model for FSDP + TP.
default_1 = MaxTextModel(
    model_name="default-1",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": 16,
        "ici_tensor_parallelism": 16,
        "dcn_pipeline_parallelism": 2,
        "dcn_fsdp_parallelism": 1,
        "remat_policy": "full",
        "global_parameter_scale": 1,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)


default_32 = MaxTextModel(
    model_name="default-32",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 13,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "global_parameter_scale": 32,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 1024,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

default_64 = MaxTextModel(
    model_name="default-64",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 6,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "global_parameter_scale": 64,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

default_128 = MaxTextModel(
    model_name="default-128",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "global_parameter_scale": 128,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

default_256 = MaxTextModel(
    model_name="default-256",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": -1,
        # "dcn_fsdp_parallelism": 2,
        "dcn_fsdp_transpose_parallelism": -1,
        "remat_policy": "full",
        "global_parameter_scale": 256,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

# TODO(b/368441022) DEFAULT_512 which requires host offloading for optimizer
# state.
# Currently OOM
default_512 = MaxTextModel(
    model_name="default-512",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": -1,
        # "dcn_fsdp_parallelism": 2,
        "dcn_fsdp_parallelism": -1,
        "remat_policy": "full",
        "global_parameter_scale": 512,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)


gpt_3_175b = MaxTextModel(
    model_name="gpt-3-175b",
    model_type="gpt3-175b",
    tuning_params={
        "per_device_batch_size": 3,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "attention": "flash",
        "quantization": "int8",
        "gcs_metrics": True,
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.DISABLE_BUNDLE_AWARE_COST_MODEL
    ),
)

llama2_7b_4096 = MaxTextModel(
    model_name="llama2-7b-4096",
    model_type="llama2-7b",
    tuning_params={
        "per_device_batch_size": 12,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)


llama2_70b_4096_real_data = MaxTextModel(
    model_name="llama2-70b-4096-rd",
    model_type="llama2-70b",
    tuning_params={
        "per_device_batch_size": 7,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "tfds",
        "tokenizer_path": "assets/tokenizer.llama2",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

# ici_fsdp_transpose_parallelism gives one TFLOP better performance.
llama2_70b_4096 = MaxTextModel(
    model_name="llama2-70b-4096",
    model_type="llama2-70b",
    tuning_params={
        "per_device_batch_size": 4,
        "ici_fsdp_parallelism": 1,
        "ici_fsdp_transpose_parallelism": -1,
        "ici_tensor_parallelism": 1,
        "remat_policy": "full",
        "max_target_length": 4096,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

llama3_8b_8192 = MaxTextModel(
    model_name="llama3-8b-8192",
    model_type="llama3-8b",
    tuning_params={
        "per_device_batch_size": 8,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
)

llama3_70b_8192 = MaxTextModel(
    model_name="llama3-70b-8192",
    model_type="llama3-70b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "full",
        "optimizer_memory_host_offload": True,
        "gradient_clipping_threshold": 0,
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + " --xla_tpu_scheduler_percent_shared_memory_limit=90"
    ),
)

llama3_1_405b_8192_fsdp_dcn = MaxTextModel(
    model_name="llama3-1-405b-8192-fsdp-dcn",
    model_type="llama3.1-405b",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": 64,
        "ici_tensor_parallelism": 4,
        "dcn_fsdp_parallelism": 2,
        "allow_split_physical_axes": True,
        "custom_mesh": "hybrid_ring_64x4",
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "out_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 1024,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
    ),
)

llama3_1_70b_129024 = MaxTextModel(
    model_name="llama3_1-70b-129024",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 0.125,
        "ici_fsdp_parallelism": -1,
        "ici_sequence_parallelism": 8,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "out_proj": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 129024,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "enable_checkpointing": False,
        "sa_block_q": 2048,
        "sa_block_kv": 2048,
        "sa_block_kv_compute": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048,
        "sa_block_q_dq": 2048,
        "sa_block_kv_dq": 2048,
        "sa_use_fused_bwd_kernel": True,
        "profiler": "xplane",
        "skip_first_n_steps_for_profiler": 10,
        "profiler_steps": 5,
        "allow_split_physical_axes": True,
        "custom_mesh": "hybrid_ring_32x8",
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_1D_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
    ),
)

mixtral_8x7b_dropless = MaxTextModel(
    model_name="mixtral-8x7b",
    model_type="mixtral-8x7b",
    tuning_params={
        "per_device_batch_size": 12,
        "ici_fsdp_parallelism": -1,
        "max_target_length": 4096,
        "remat_policy": "full",
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "megablox": True,
    },
    xla_flags=(
        xla_flags_library.MOE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

mixtral_8x7b_dropped = MaxTextModel(
    model_name="mixtral-8x7b",
    model_type="mixtral-8x7b",
    tuning_params={
        "per_device_batch_size": 8,
        "ici_fsdp_parallelism": -1,
        "max_target_length": 4096,
        "remat_policy": "full",
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "megablox": False,
        "capacity_factor": 1.25,
        "tokenizer_path": "assets/tokenizer.mistral-v1",
    },
    xla_flags=(
        xla_flags_library.MOE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

mixtral_8x7b_dropped_int8 = MaxTextModel(
    model_name="mixtral-8x7b",
    model_type="mixtral-8x7b",
    tuning_params={
        "per_device_batch_size": 8,
        "ici_fsdp_parallelism": -1,
        "max_target_length": 4096,
        "remat_policy": "full",
        "attention": "flash",
        "gcs_metrics": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
        "megablox": False,
        "capacity_factor": 1.25,
        "quantization": "int8",
        "tokenizer_path": "assets/tokenizer.mistral-v1",
    },
    xla_flags=(
        xla_flags_library.MOE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
    ),
)

gemma2_9b_8192 = MaxTextModel(
    model_name="gemma2-9b-8192",
    model_type="gemma2-9b",
    tuning_params={
        "per_device_batch_size": 3,
        "ici_fsdp_transpose_parallelism": 256,
        "remat_policy": "full",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "tokenizer_path": "assets/tokenizer.llama2",
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.CUSTOM_VMEM_LIMIT_FLAG(114688)
        + xla_flags_library.REDUCE_SCATTER_FUSION
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
    ),
)

gemma2_27b_8192 = MaxTextModel(
    model_name="gemma2-27b-8192",
    model_type="gemma2-27b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_transpose_parallelism": 256,
        "remat_policy": "full",
        "max_target_length": 8192,
        "attention": "flash",
        "gcs_metrics": True,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
        "tokenizer_path": "assets/tokenizer.llama2",
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.CUSTOM_VMEM_LIMIT_FLAG(122880)
        + xla_flags_library.REDUCE_SCATTER_FUSION
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
    ),
)

# TODO(b/368441022) LLAMA3.1 8B, 70B, 405B
# TODO(b/368441022) MaxDiffusion BEST
# TODO(b/368441022) Determine largest batch per slice for non-optimized models
# List of all models
maxstar_models = [
    default_basic,
    default_32,
    default_64,  # Not Optimizied yet
    default_128,  # Not Optimizied yet
    # default_256,  # OOM, Not Optimizied yet
    # default_512,  # OOM, Not Optimizied yet
    gpt_3_175b,
    llama2_7b_4096,
    llama2_70b_4096,
    llama2_70b_4096_real_data,
    llama3_8b_8192,  # Not Optimizied yet
    llama3_70b_8192,  # Not Optimizied yet
    llama3_1_405b_8192_fsdp_dcn,
    llama3_1_70b_129024,
    mixtral_8x7b_dropped,
    mixtral_8x7b_dropped_int8,
    mixtral_8x7b_dropless,
    gemma2_9b_8192,
    gemma2_27b_8192,
]
