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
"""
Select XLA flag preconfigurations
"""

# Flags which are manually set.
xla_flags_manually_selected = {
    "xla_jf_auto_cross_replica_sharding": "False",
    "xla_tpu_enable_windowed_einsum_for_reduce_scatter": "False",
    "xla_tpu_enable_windowed_einsum_for_all_gather": "False",
    "xla_tpu_prefer_latch_optimized_rhs_layouts": "True",
}

# Flags that are autotuned.
xla_flags_autotuned = {
    "xla_jf_auto_cross_replica_sharding": "false",
    "xla_tpu_enable_windowed_einsum_for_reduce_scatter": "false",
    "xla_tpu_enable_windowed_einsum_for_all_gather": "false",
    "xla_tpu_prefer_latch_optimized_rhs_layouts": "true",
    "xla_tpu_enable_experimental_fusion_cost_model": "false",
    "xla_tpu_dot_dot_fusion_duplicated": "false",
    "xla_tpu_dot_dot_fusion": "true",
    "xla_jf_conv_input_fusion": "true",
    "xla_jf_conv_output_fusion": "true",
    "xla_tpu_rwb_fusion": "false",
    "xla_tpu_copy_fusion_pad_unpad_ratio": "0",
    "xla_tpu_licm_size_inflation_ratio": "1",
    "xla_tpu_copy_elision_analysis_allowance": "150000",
    "xla_tpu_copy_insertion_use_region_analysis_limit": "10000",
    "xla_tpu_order_dot_after_layout": "true",
    "xla_jf_rematerialization_percent_shared_memory_limit": "100",
    "xla_tpu_use_repeated_instance_for_preferred_prefetch_time": "true",
    "xla_tpu_enforce_prefetch_fifo_order": "false",
    "xla_tpu_prefetch_interval_picker_size_override": "6000000",
    "xla_tpu_async_copy_bandwidth_scaling_factor": "1",
    "xla_tpu_nd_short_transfer_max_chunks": "-1",
    "xla_tpu_enable_aggressive_broadcast_priority_update": "true",
    "xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers": "SQRT",
    "xla_tpu_memory_bound_loop_optimizer_options": "enabled:true",
    "xla_tpu_enable_copy_fusion": "true",
    "xla_tpu_enable_cross_program_prefetch_freeing": "false",
    "xla_tpu_enable_dot_strength_reduction": "true",
    "xla_tpu_layout_use_dot_grouping": "false",
    "xla_tpu_msa_inefficient_use_to_copy_ratio": "0.5",
    "xla_tpu_reduce_loop_fusion_dup_with_unfusable_user": "false",
    "xla_tpu_vector_load_fusion_window": "1024",
    "xla_tpu_vector_store_fusion_window": "256",
    "xla_jf_conv_reshape_fusion": "false",
    "xla_tpu_input_conv_multi_users": "false",
    "xla_tpu_enable_multi_level_input_dot_dot_fusion": "false",
    "xla_tpu_enable_multi_level_output_dot_dot_fusion": "false",
    "xla_tpu_dot_dot_fusion_separable_convs_only": "false",
    "xla_tpu_enable_multi_level_nested_loop_fusion": "true",
    "xla_tpu_nested_dot_fusion": "true",
    "xla_tpu_enable_multi_level_nested_dot_fusion": "false",
    "xla_jf_enable_multi_output_fusion": "true",
    "xla_tpu_use_lp_llo_scheduler_for_dot_dot_fusions": "false",
    "xla_tpu_enable_flash_attention": "true",
}


def _build_xla_flag_str(flags_set):
  flags_str = [f"--{k}={v}" for k, v in flags_set.items()]
  return " ".join(flags_str)


def main():
  print(_build_xla_flag_str(xla_flags_autotuned))


if __name__ == "__main__":
  main()
