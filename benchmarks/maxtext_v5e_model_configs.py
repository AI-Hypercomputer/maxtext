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
"""Shared Benchmark config for v5e orchestrations."""

import xla_flags_library
from maxtext_trillium_model_configs import MaxTextModel, _add_to_model_dictionary


v5e_model_dict = {}

default_16b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="default-16b-v5e-256",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 6,
        "remat_policy": "full",
        "global_parameter_scale": 16,
        "max_target_length": 2048,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)


default_32b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="default-32b-v5e-256",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 4,
        "remat_policy": "full",
        "global_parameter_scale": 32,
        "max_target_length": 2048,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)


default_64b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="default-64b-v5e-256",
    model_type="default",
    tuning_params={
        "per_device_batch_size": 2,
        "remat_policy": "full",
        "global_parameter_scale": 64,
        "max_target_length": 2048,
        "attention": "flash",
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)

default_128b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="default-128b-v5e-256",
    model_type="default",
    tuning_params={
        "ici_fsdp_parallelism": -1,
        "ici_tensor_parallelism": 16,
        "per_device_batch_size": 1,
        "remat_policy": "qkv_proj_offloaded",
        "global_parameter_scale": 128,
        "max_target_length": 2048,
        "attention": "flash",
        "use_iota_embed": True,
        "fused_qkv": True,
        "fused_mlp": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)

gpt_3_175b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="gpt-3-175b-v5e-256",
    model_type="gpt3-175b",
    tuning_params={
        "ici_fsdp_parallelism": -1,
        "ici_tensor_parallelism": 16,
        "per_device_batch_size": 0.5,
        "remat_policy": "full",
        "max_target_length": 2048,
        "attention": "flash",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)

llama2_7b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="llama2-7b-v5e-256",
    model_type="llama2-7b",
    tuning_params={
        "ici_fsdp_parallelism": -1,
        "per_device_batch_size": 4,
        "remat_policy": "save_qkv_proj",
        "max_target_length": 2048,
        "use_iota_embed": True,
        "tokenizer_path": "assets/tokenizer.llama2",
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)

llama2_13b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="llama2-13b-v5e-256",
    model_type="llama2-13b",
    tuning_params={
        "ici_fsdp_parallelism": -1,
        "per_device_batch_size": 8,
        "remat_policy": "qkv_proj_offloaded",
        "max_target_length": 2048,
        "use_iota_embed": True,
        "tokenizer_path": "assets/tokenizer.llama2",
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)

llama2_70b_v5e_256 = _add_to_model_dictionary(
  v5e_model_dict,
  MaxTextModel(
    model_name="llama2-70b-v5e-256",
    model_type="llama2-70b",
    tuning_params={
        "ici_fsdp_parallelism": -1,
        "per_device_batch_size": 2,
        "remat_policy": "qkv_proj_offloaded",
        "max_target_length": 2048,
        "use_iota_embed": True,
        "dataset_path": "gs://max-datasets-rogue",
        "dataset_type": "synthetic",
        "reuse_example_batch": 1,
        "enable_checkpointing": False,
        "profiler": "xplane",
    },
    xla_flags=(
        xla_flags_library.DATA_PARALLEL_OVERLAP
        + xla_flags_library.CF_FOR_ALL_GATHER
    ),
	)
)
