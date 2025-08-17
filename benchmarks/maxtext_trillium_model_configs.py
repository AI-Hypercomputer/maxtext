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
import os.path
import typing
from tempfile import gettempdir
from benchmarks.benchmark_utils import MaxTextModel, _add_to_model_dictionary
from benchmarks import xla_flags_library

# TODO(vbarr@) Abstract software features like checkpointing,
# real data / synthetic data out of this config
# TODO(vbarr@) Make slice dependent configurations to allow for a model's tuning
# to adjust at scales.

# The minimum set of tuning params required for pathways. Does not include
# checkpointing params.
BASE_PATHWAYS_TUNING_PARAMS = {
    "checkpoint_storage_use_ocdbt": False,
    "checkpoint_storage_use_zarr3": False,
    "enable_pathways_goodput": True,
    "enable_goodput_recording": True,
    "enable_single_controller": True,
    "metrics_file": "metrics.txt",
    "goodput_upload_interval_seconds": 30,
}

# The set of tuning params required for long-running pathways jobs.
PATHWAYS_LONG_RUN_CHECKPOINTING_TUNING_PARAMS = {
    "enable_checkpointing": True,
    "async_checkpointing": True,
    "checkpoint_period": 100,
    "enable_checkpoint_cloud_logger": True,
    "enable_goodput_recording": True,
}

# The set of tuning params required for short-running pathways jobs.
PATHWAYS_SHORT_RUN_CHECKPOINTING_TUNING_PARAMS = {
    "enable_checkpointing": True,
    "async_checkpointing": True,
    "checkpoint_period": 20,
    "enable_checkpoint_cloud_logger": True,
    "enable_goodput_recording": True,
}



trillium_model_dict = {}



default_basic_1 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="default-basic-1",
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
    ),
)


default_32 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)


default_64 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)


default_128 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)


# OOM, Not Optimized yet
default_256 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="default-256",
        model_type="default",
        tuning_params={
            "per_device_batch_size": 1,
            "ici_fsdp_parallelism": -1,
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
    ),
)


# OOM, Not Optimized yet
default_512 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)


gpt_3_175b = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)

gpt_3_175b_bf16 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="gpt-3-175b-bf16",
        model_type="gpt3-175b",
        tuning_params={
            "per_device_batch_size": 3,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "full",
            "attention": "flash",
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
    ),
)


llama2_7b_4096 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)


llama2_70b_4096 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
    ),
)


llama2_70b_4096_optimized = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama2_70b_4096_synthetic",
        model_type="llama2-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1,
            "remat_policy": "qkv_proj_offloaded",
            "max_target_length": 4096,
            "attention": "flash",
            "gcs_metrics": True,
            "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic",
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
    ),
)


# Enable SparseCore Offloading of AR in an optimized model.
llama2_70b_4096_sc = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama2-70b-4096-sc",
        model_type="llama2-70b",
        tuning_params={
            "per_device_batch_size": 3,
            "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1,
            "remat_policy": "qkv_proj_offloaded",
            "max_target_length": 4096,
            "attention": "flash",
            "gcs_metrics": True,
            "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic",
            "enable_checkpointing": False,
            "profiler": "xplane",
            "sa_block_q": 1024,
            "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048,
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)


llama2_70b_4096_sc_real_data_tfds = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama2-70b-4096-sc-real-data-tfds",
        model_type="llama2-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1,
            "remat_policy": "qkv_proj_offloaded",
            "max_target_length": 4096,
            "attention": "flash",
            "gcs_metrics": True,
            "use_iota_embed": True,
            "dataset_path": "gs://trillium-storage-datasets-sr",
            "enable_checkpointing": False,
            "profiler": "xplane",
            "sa_block_q": 1024,
            "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048,
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)

temp_dir = gettempdir()
llama2_70b_4096_sc_real_data_grain = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama2-70b-4096-sc-real-data-grain",
        model_type="llama2-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1,
            "remat_policy": "qkv_proj_offloaded",
            "max_target_length": 4096,
            "attention": "flash",
            "gcs_metrics": True,
            "use_iota_embed": True,
            "dataset_path": "gs://trillium-storage-datasets-sr",
            "base_output_directory": (
                "gs://trillium-storage-tests-nov24-sr/long-run-dec11"
            ),
            "enable_checkpointing": False,
            "dataset_type": "grain",
            "grain_train_files": (
                os.path.join(temp_dir, "dataset", "array-record", "c4", "en", "3.0.1", "c4-train.array_record*")
            ),
            "grain_worker_count": 24,
            "profiler": "xplane",
            "sa_block_q": 1024,
            "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048,
            "profile_cleanly": False,
        },
        pathways_tuning_params=PATHWAYS_LONG_RUN_CHECKPOINTING_TUNING_PARAMS,
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)


llama2_70b_4096_sc_real_data_grain_checkpoint = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama2-70b-4096-sc-real-data-grain-checkpoint",
        model_type="llama2-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1,
            "remat_policy": "qkv_proj_offloaded",
            "max_target_length": 4096,
            "attention": "flash",
            "gcs_metrics": True,
            "use_iota_embed": True,
            "dataset_path": "gs://trillium-storage-datasets-sr",
            "base_output_directory": (
                "gs://trillium-storage-tests-nov24-sr/long-run-dec11"
            ),
            "checkpoint_period": 100,
            "enable_checkpointing": True,
            "async_checkpointing": True,
            "dataset_type": "grain",
            "grain_train_files": (
                os.path.join(temp_dir, "dataset", "array-record", "c4", "en", "3.0.1", "c4-train.array_record*")
            ),
            "grain_worker_count": 24,
            "profiler": "xplane",
            "sa_block_q": 1024,
            "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048,
        },
        pathways_tuning_params=PATHWAYS_LONG_RUN_CHECKPOINTING_TUNING_PARAMS,
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)

llama2_70b_4096_real_data_long_run = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama2-70b-4096-rd-lr",
        model_type="llama2-70b",
        tuning_params={
            "per_device_batch_size": 4,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "full",
            "max_target_length": 4096,
            "attention": "flash",
            "gcs_metrics": True,
            "use_iota_embed": True,
            "reuse_example_batch": 0,
            "profiler": "xplane",
            "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "tfds",
            "tokenizer_path": os.path.join("assets", "tokenizer.llama2"),
            "sa_block_q": 1024,
            "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048,
        },
        pathways_tuning_params=PATHWAYS_LONG_RUN_CHECKPOINTING_TUNING_PARAMS,
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
        ),
        pathways_xla_flag_options={
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)


llama3_8b_8192 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
        pathways_xla_flag_options={
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)


llama3_70b_8192 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)

# Config only runs on v6e-256
llama3_1_405b_8192_fsdp_dcn = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)

# Config only runs on v6e-256
llama3_1_405b_8192_pure_fsdp_ici = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="llama3-1-405b-8192-pure-fsdp-ici",
    model_type="llama3.1-405b",
    tuning_params={
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": 256,
        "dcn_fsdp_parallelism": 2,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
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
)

llama3_1_8b_8192 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-8b-8192",
        model_type="llama3.1-8b",
        tuning_params={
            "per_device_batch_size": 4,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "out_proj": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)

# Config for v6e-64
llama3_1_8b_8192_bs5 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-8b-8192-bs5",
        model_type="llama3.1-8b",
        tuning_params={
            "per_device_batch_size": 5,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "out_proj": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    ),
)


llama3_1_8b_8192_no_collective_matmul = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="llama3_1-8b-8192-no-collective-matmul",
    model_type="llama3.1-8b",
    tuning_params={
        "per_device_batch_size": 3,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "out_proj": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
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
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.DISABLE_COLLECTIVE_MATMUL
    ),
  )
)


llama3_1_70b_8192 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 5,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    ),
)

# Config for v6e-64
llama3_1_70b_8192_bs2 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192-bs2",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    ),
)

# Config for v6e-32
llama3_1_70b_8192_bs2_bfloat16_no_collective_matmul = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192-bs2-bfloat16-no-collective-matmul",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
            "weight_dtype": "bfloat16",
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
            + xla_flags_library.DISABLE_COLLECTIVE_MATMUL
        ),
    ),
)

# Config for v6e-128
llama3_1_70b_8192_bs4 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192-bs4",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 4,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    ),
)

llama3_1_70b_8192_iter_synthetic = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="llama3_1_70b_8192_synthetic",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "use_iota_embed": True,
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
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        +  " --xla_tpu_iova_dma_chunk_size_bytes=104857"
    ),
  )
)

llama3_1_70b_8192_iter_real_data_grain = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="llama3_1_70b_8192_rd_grain",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "/tmp/dataset",
        "dataset_type": "grain",
        "grain_train_files": "/tmp/dataset/array-record/c4/en/3.0.1/c4-train.array_record*",
        "grain_worker_count": 24,
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
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        +  " --xla_tpu_iova_dma_chunk_size_bytes=104857"
    ),
  )
)

llama3_1_70b_8192_iter_synthetic_ckpt = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="llama3_1_70b_8192_synthetic_ckpt",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_type": "synthetic",
        "enable_checkpointing": True,
        "async_checkpointing": True,
        "checkpoint_period": 20,
        "sa_block_q": 2048,
        "sa_block_kv": 2048,
        "sa_block_kv_compute": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048,
        "sa_block_q_dq": 2048,
        "sa_block_kv_dq": 2048,
        "sa_use_fused_bwd_kernel": True,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        +  " --xla_tpu_iova_dma_chunk_size_bytes=104857"
    ),
  )
)

llama3_1_70b_8192_iter_real_data_and_checkpointing = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="llama3_1_70b_8192_rd_ckpt_grain",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 2,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "/tmp/dataset",
        "dataset_type": "grain",
        "grain_train_files": "/tmp/dataset/array-record/c4/en/3.0.1/c4-train.array_record*",
        "grain_worker_count": 24,
        "enable_checkpointing": True,
        "async_checkpointing": True,
        "checkpoint_period": 20,
        "sa_block_q": 2048,
        "sa_block_kv": 2048,
        "sa_block_kv_compute": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048,
        "sa_block_q_dq": 2048,
        "sa_block_kv_dq": 2048,
        "sa_use_fused_bwd_kernel": True,
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        +  " --xla_tpu_iova_dma_chunk_size_bytes=104857"
    ),
  )
)

llama3_1_70b_8192_lr_real_data = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192-pw-lr-rd",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 4,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
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
        },
        pathways_tuning_params=PATHWAYS_LONG_RUN_CHECKPOINTING_TUNING_PARAMS,
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)

llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192-iter-real-data-and-checkpointing-tfds",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
            "attention": "flash",
            "use_iota_embed": True,
            "dataset_path": "gs://trillium-scale-datasets-q1-25-west",
            "dataset_type": "tfds",
            "enable_checkpointing": True,
            "async_checkpointing": True,
            "checkpoint_period": 20,
            "enable_checkpoint_cloud_logger": True,
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True,
            "gcs_metrics": True,
            "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10,
            "profiler_steps": 5,
            "tokenizer_type": "tiktoken",
            "tokenizer_path": "assets/tokenizer_llama3.tiktoken",
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
            + " --xla_tpu_iova_dma_chunk_size_bytes=104857"
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        }
    ),
)

llama3_1_70b_8192_iter_synth_data_and_checkpointing = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="llama3_1-70b-8192-synth",
        model_type="llama3.1-70b",
        tuning_params={
            "per_device_batch_size": 2,
            "ici_fsdp_parallelism": -1,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "max_target_length": 8192,
            "attention": "flash",
            "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic",
            "enable_checkpointing": True,
            "async_checkpointing": True,
            "checkpoint_period": 20,
            "enable_checkpoint_cloud_logger": True,
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True,
            "gcs_metrics": True,
            "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10,
            "profiler_steps": 5,
            "tokenizer_type": "tiktoken",
            "tokenizer_path": "assets/tokenizer_llama3.tiktoken",
        },
        xla_flags=(
            xla_flags_library.DENSE_VMEM_LIMIT_FLAG
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
            + " --xla_tpu_iova_dma_chunk_size_bytes=104857"
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        }
    ),
)

llama3_1_70b_129024 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
            + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
        pathways_xla_flag_options={
            xla_flags_library.REMOVE: [
                "--2a886c8_chip_config_name=megachip_tccontrol"
            ],
            xla_flags_library.ADD_SERVER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_PROXY: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
            xla_flags_library.ADD_WORKER: (
                xla_flags_library.ENHANCED_LAUNCH_BARRIER
            ),
        },
    ),
)


mistral_7b = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="mistral-7b",
    model_type="mistral-7b",
    tuning_params={
        "per_device_batch_size": 6,
        "ici_fsdp_parallelism": -1,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "out_proj": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 8192,
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
    },
    xla_flags=(
        xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        + xla_flags_library.DISABLE_COLLECTIVE_MATMUL
    ),
  )
)


mixtral_8x7b_dropless = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="mixtral_8x7b_dropless",
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
            "sparse_matmul": True,
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
        ),
    ),
)


mixtral_8x7b_dropped = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="mixtral_8x7b_dropped",
        model_type="mixtral-8x7b",
        tuning_params={
            "per_device_batch_size": 12,
            "ici_fsdp_parallelism": -1,
            "max_target_length": 4096,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "out_proj": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
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
            "sparse_matmul": False,
            "capacity_factor": 1.25,
            "tokenizer_path": "assets/tokenizer.mistral-v1",
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
        ),
    ),
)


mixtral_8x7b_dropped_int8 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="mixtral_8x7b_dropped_int8",
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
            "sparse_matmul": False,
            "capacity_factor": 1.25,
            "quantization": "int8",
            "tokenizer_path": "assets/tokenizer.mistral-v1",
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
        ),
    ),
)

# Config only runs on v6e-256
mixtral_8x22b_dropped = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="mixtral_8x22b_dropped",
        model_type="mixtral-8x22b",
        tuning_params={
            "per_device_batch_size": 8,
            "max_target_length": 4096,
            "ici_fsdp_parallelism": 64,
            "ici_expert_parallelism": 4,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "out_proj": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
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
            "sparse_matmul": False,
            "capacity_factor": 1.25,
            "tokenizer_path": "assets/tokenizer.mistral-v3",
            "dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "allow_split_physical_axes": True,
            "custom_mesh": "hybrid_ring_64x4",
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
        ),
    ),
)

# Config only runs on v6e-256
deepseek_v3_ep16 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="deepseek_v3_ep16",
        model_type="deepseek3-671b",
        tuning_params={
            "per_device_batch_size": 1,
            "max_target_length": 8192,
            "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16,
            "dcn_fsdp_parallelism": 2,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
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
            "sparse_matmul": False,
            "capacity_factor": 1.0,
            "tokenizer_path": "assets/tokenizer.mistral-v3",
            "dtype": "bfloat16",
            "opt_type": "sgd",
            "weight_dtype": "bfloat16",
            "attention": "flash",
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
        ),
    ),
)

# Config only runs on v6e-256
gemma2_9b_8192 = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
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
            "tokenizer_path": os.path.join("assets", "tokenizer.llama2"),
            "sa_block_q": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048,
        },
        xla_flags=(
            xla_flags_library.CUSTOM_VMEM_LIMIT_FLAG(vmem_limit=114688)
            + xla_flags_library.REDUCE_SCATTER_FUSION
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        ),
    ),
)

# Config only runs on v6e-256
gemma2_27b_8192 = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
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
        "tokenizer_path": os.path.join("assets", "tokenizer.llama2"),
        "sa_block_q": 2048,
        "sa_block_q_dkv": 2048,
        "sa_block_q_dq": 2048,
    },
    xla_flags=(
        xla_flags_library.CUSTOM_VMEM_LIMIT_FLAG(vmem_limit=122880)
        + xla_flags_library.REDUCE_SCATTER_FUSION
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
    ),
    )
)

# Config for Llama3.1 70B model with 131072 max target length aka context length
llama3_1_70b_131072 = _add_to_model_dictionary(
  trillium_model_dict,
    MaxTextModel(
    model_name="llama3_1_70b_131072",
    model_type="llama3.1-70b",
    tuning_params={
        "per_device_batch_size": 0.125,
        "ici_fsdp_parallelism": -1,
        "ici_context_parallelism": 16,
        "remat_policy": "custom",
        "decoder_layer_input": "offload",
        "out_proj": "offload",
        "query_proj": "offload",
        "key_proj": "offload",
        "value_proj": "offload",
        "max_target_length": 131072,
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
        "tokenizer_type": "tiktoken",
        "tokenizer_path": "assets/tokenizer_llama3.tiktoken", 
        "packing": False,
    },
    xla_flags=(xla_flags_library.DENSE_VMEM_LIMIT_FLAG
        + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR
        + xla_flags_library.HOST_OFFLOAD_FLAGS
        
    ),
    pathways_xla_flag_options={
        xla_flags_library.REMOVE: [
            "--2a886c8_chip_config_name=megachip_tccontrol"
        ],
        xla_flags_library.ADD_SERVER: (
            xla_flags_library.ENHANCED_LAUNCH_BARRIER
        ),
        xla_flags_library.ADD_PROXY: (
            xla_flags_library.ENHANCED_LAUNCH_BARRIER
        ),
        xla_flags_library.ADD_WORKER: (
            xla_flags_library.ENHANCED_LAUNCH_BARRIER
        ),
    },
  )
)

# Customized MoE model - 700B, and config only runs on v6e-256
custom_moe_700b = _add_to_model_dictionary(
    trillium_model_dict,
    MaxTextModel(
        model_name="custom_moe_700b",
        model_type="default",
        tuning_params={
            "per_device_batch_size": 1,
            "max_target_length": 8192,
            "decoder_block": "mixtral",
            "base_emb_dim": 8192,
            "base_mlp_dim": 32768,
            "base_num_decoder_layers": 56,
            "head_dim": 256,
            "base_num_kv_heads": 8,
            "base_num_query_heads": 128,
            "vocab_size": 32000,
            "enable_dropout": False,
            "logits_via_embedding": False,
            "normalization_layer_epsilon": 1.0e-5,
            "rope_max_timescale": 1_000_000,
            "num_experts": 16,
            "num_experts_per_tok": 2,
            "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "out_proj": "offload",
            "query_proj": "offload",
            "key_proj": "offload",
            "value_proj": "offload",
            "dataset_type": "synthetic",
            "reuse_example_batch": 1,
            "enable_checkpointing": False,
            "profiler": "xplane",            
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True,
            "sparse_matmul": False,
            "capacity_factor": 1.5,
            "tokenizer_path": "assets/tokenizer.mistral-v1",
            "dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "opt_type": "sgd",
            "attention": "flash",
        },
        xla_flags=(
            xla_flags_library.MOE_VMEM_LIMIT_FLAG
            + xla_flags_library.CF_FOR_ALL_GATHER
            + xla_flags_library.DATA_PARALLEL_OVERLAP
            + xla_flags_library.LAYOUT_FOR_ALL_REDUCE_SCATTER
            + xla_flags_library.HOST_OFFLOAD_FLAGS
        ),
    ),
)

# docker_image_flag = '--docker-image="gcr.io/tpu-prod-env-multipod/mattdavidow-successive-a1"'
# commit 91a6aedc9a6dd8babbdc48360ba0346cf9fef4e8 (HEAD -> mattdavidow-ds-gogo, origin/mattdavidow-ds-gogo)
deepseek_manual_matt_a1 = _add_to_model_dictionary(
  trillium_model_dict,
  MaxTextModel(
    model_name="deepseek_manual_matt_a1",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 2,
        "max_target_length": 4096,
        "ici_fsdp_parallelism": 16,
        "ici_pipeline_parallelism": 16,
        "pipeline_fsdp_ag_once": True,
        "num_successive_pipelines": 4,
        "pipeline_parallel_layers": 64,
        "num_pipeline_microbatches": 32,
        "remat_policy": "full",
        # "decoder_layer_input": "offload",
        # "out_proj": "offload",
        # "query_proj": "offload",
        # "key_proj": "offload",
        # "value_proj": "offload",
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
        "sparse_matmul": True,
        "tokenizer_path": "assets/tokenizer.mistral-v3",
        "dtype": "bfloat16",
        "opt_type": "sgd",
        "weight_dtype": "bfloat16",

        # DS def
        "base_emb_dim": 7168,
        "base_num_query_heads": 128,
        "base_num_kv_heads": 128,
        "base_mlp_dim": 18432,
        "base_moe_mlp_dim": 2048,
        "base_num_decoder_layers": 68,
        "first_num_dense_layers": 3,
        #"mlp_activations": ["silu","linear"],
        "vocab_size": 129280,
        "enable_dropout": False,
        "logits_via_embedding": False,
        "normalization_layer_epsilon": 1.0e-6,
        "num_experts": 32,
        "num_experts_per_tok": 8,
        "shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "routed_score_func": "sigmoid",
        "routed_bias": True,
        #MLA
        "attention_type": "mla",
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "rope_type": "yarn",
        "mscale": 1.0,
        "decoder_block": "deepseek"
    },
    xla_flags=(
        xla_flags_library.MOE_VMEM_LIMIT_FLAG
        + xla_flags_library.CF_FOR_ALL_GATHER
        + xla_flags_library.DATA_PARALLEL_OVERLAP
        #+ xla_flags_library.ASYNC_ALL_TO_ALL
    ),
  )
)
