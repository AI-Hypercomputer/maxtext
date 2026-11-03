# Copyright 2023–2026 Google LLC
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

"""Shared Benchmark config for TPU v7x (Ironwood) orchestrations."""

from benchmarks import xla_flags_library
from benchmarks.benchmark_utils import MaxTextModel, _add_to_model_dictionary

_QWEN3_235B_V7X_BF16_COMMON_PARAMS = {
    "per_device_batch_size": 16.0,
    "max_target_length": 4096,
    "dataset_type": "synthetic",
    "use_random_routing": False,
    "reuse_example_batch": 1,
    "enable_checkpointing": False,
    "profiler": "xplane",
    "gcs_metrics": True,
    "attention": "flash",
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "ici_fsdp_parallelism": -1,
}

v7x_model_dict: dict[str, MaxTextModel] = {}

qwen3_235b_a22b_v7x_bf16_baseline = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_bf16_baseline",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **_QWEN3_235B_V7X_BF16_COMMON_PARAMS,
            "ici_expert_parallelism": 1,
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("baseline"),
    ),
)

# Track A: FSDP-only with FSDP weight prefetching
qwen3_235b_a22b_v7x_bf16_fsdp_prefetch = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_bf16_fsdp_prefetch",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **_QWEN3_235B_V7X_BF16_COMMON_PARAMS,
            "ici_expert_parallelism": 1,
            "moe_flat_fsdp_weights": True,
            "prefetch_fsdp_weights": True,
            "prefetch_fsdp_defer_grad_reduce_scatter": True,
            "use_tokamax_gmm": True,
            "use_gmm_v2": True,
            "shard_embed_moe_on_fsdp": False,
            "shard_mode": "explicit",
            "moe_gmm_v2_dlhs_transpose_rhs": True,
            "moe_use_direct_token_gather": True,
            "remat_policy": "custom",
            "decoder_layer_input": "device",
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("fsdp_prefetch"),
    ),
)

# Track B: FSDP + Ring of Experts
qwen3_235b_a22b_v7x_bf16_roe = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_bf16_roe",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **_QWEN3_235B_V7X_BF16_COMMON_PARAMS,
            "ici_expert_parallelism": 16,
            "use_random_routing": True,
            "use_ring_of_experts": True,
            "use_ragged_sort": True,
            "moe_tc_ragged_sort": True,
            "moe_tc_ragged_3d_dispatch": True,
            "moe_tc_ragged_3d_gmm": True,
            "moe_tc_ragged_weights_on_activation": True,
            "moe_tc_window_routing": True,
            "ragged_buffer_factor": 2.5,
            "use_tokamax_gmm": True,
            "use_gmm_v2": True,
            "custom_mesh_and_rule": "fsdp-as-dp-for-attn-rf",
            "remat_policy": "custom",
            "decoder_layer_input": "device",
            "moe_routing_maps": "device",
            "moe_tc_routing_checkpoint": "device",
            "wi_tile_fwd_batch_seq": 512,
            "wi_tile_dlhs_batch_seq": 256,
            "wi_tile_dlhs_embed_dim": 1536,
            "wi_tile_drhs_batch_seq": 1024,
            "wi_tile_drhs_embed_dim": 1024,
            "wo_tile_dlhs_batch_seq": 256,
            "wo_tile_dlhs_embed_dim": 4096,
            "wo_tile_drhs_batch_seq": 1024,
            "wo_tile_drhs_embed_dim": 1024,
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("roe"),
    ),
)

_QWEN3_235B_V7X_FP8_COMMON_PARAMS = {
    **_QWEN3_235B_V7X_BF16_COMMON_PARAMS,
    "quantization": "fp8_full",
    "use_qwix_quantization": True,
    "quantize_router_proj": False,
}

# Track Baseline: FP8 unquantized router baseline
qwen3_235b_a22b_v7x_fp8_baseline = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_fp8_baseline",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **_QWEN3_235B_V7X_FP8_COMMON_PARAMS,
            "ici_expert_parallelism": 1,
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("baseline"),
    ),
)

# Track A: FSDP-only with FSDP weight prefetching (FP8)
qwen3_235b_a22b_v7x_fp8_fsdp_prefetch = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_fp8_fsdp_prefetch",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **qwen3_235b_a22b_v7x_bf16_fsdp_prefetch.tuning_params,
            **_QWEN3_235B_V7X_FP8_COMMON_PARAMS,
            "shard_embed_moe_on_fsdp": True,
            "weight_quantization_calibration_method": "fixed,-224,224",
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("fsdp_prefetch"),
    ),
)

# Track B: FSDP + Ring of Experts (FP8 fixed-scale)
qwen3_235b_a22b_v7x_fp8_roe = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_fp8_roe",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **qwen3_235b_a22b_v7x_bf16_roe.tuning_params,
            **_QWEN3_235B_V7X_FP8_COMMON_PARAMS,
            "use_ring_of_experts": True,
            "moe_bwd_prequant_before_unsort": True,
            "moe_3d_dlhs_partial_sum": True,
            "moe_accumulate_wi_dlhs": True,
            "moe_tc_ragged_3d_gmm": True,
            "moe_combine_bwd_direct_qarray": True,
            "weight_quantization_calibration_method": "fixed,-224,224",
            "act_quantization_calibration_method": "fixed,-224,224",
            "bwd_quantization_calibration_method": "fixed,-224,224",
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("roe"),
    ),
)

# Track B: FSDP + Ring of Experts (FP8 dynamic-scale)
qwen3_235b_a22b_v7x_fp8_roe_dynamic = _add_to_model_dictionary(
    v7x_model_dict,
    MaxTextModel(
        model_name="qwen3_235b_a22b_v7x_fp8_roe_dynamic",
        model_type="qwen3-235b-a22b",
        tuning_params={
            **qwen3_235b_a22b_v7x_bf16_roe.tuning_params,
            **_QWEN3_235B_V7X_FP8_COMMON_PARAMS,
            "use_ring_of_experts": True,
            "moe_3d_dlhs_partial_sum": True,
            "moe_accumulate_wi_dlhs": True,
            "moe_tc_ragged_3d_gmm": True,
            "weight_quantization_calibration_method": "dynamic",
            "act_quantization_calibration_method": "dynamic",
            "bwd_quantization_calibration_method": "dynamic",
        },
        xla_flags=xla_flags_library.get_v7x_qwen3_235b_flags("roe"),
    ),
)
