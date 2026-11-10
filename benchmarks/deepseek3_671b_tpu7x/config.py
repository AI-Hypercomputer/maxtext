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

"""DeepSeek-V3 671B FP8 (FSDP + EP) training recipes for TPU7x (Ironwood).

Each recipe is a `MaxTextModel` (MaxText overrides + XLA flags) plus the TPU7x
slice it was tuned for, and can be launched with `maxtext_ctk_runner` (Cluster
Toolkit / `gcluster`). See `benchmark.py` in this directory.

Recipes:
  * `8k`: 8,192 chips (16,384 TensorCores), one tpu7x 8x32x32 slice.
    FSDP 1024 x EP 16, per_device_batch_size 1 (global batch 16,384 x 4,096).
"""

# TODO: Add the 2k and 4k recipes (new `MaxTextModel` + `RECIPES` entry each).

import dataclasses

from benchmarks.benchmark_utils import MaxTextModel, _add_to_model_dictionary
from benchmarks.deepseek3_671b_tpu7x import xla_flags

tpu7x_deepseek3_model_dict = {}

# Placeholders: point these at your own copies of the C4 dataset (tfds,
# resharded for 16,384 hosts' worth of shards) and of the DeepSeek-V3 671B
# checkpoint, converted to MaxText format and pre-sharded for the recipe's mesh
# (flat MoE layout, FSDP x EP as below). `benchmark.py` overrides both.
DATASET_PATH_PLACEHOLDER = "gs://<your-bucket>/tfds-reshard-16384/"
CHECKPOINT_PATH_PLACEHOLDER = "gs://<your-bucket>/<deepseek3-671b-checkpoint>/0/items"


@dataclasses.dataclass(frozen=True)
class Tpu7xRecipe:
  """A tuned recipe and the TPU7x slice it targets.

  Attributes:
    model: MaxText overrides and XLA flags.
    device_type: `ClusterConfig` / `WorkloadConfig` device type,
      `tpu7x-<TensorCores>` (two TensorCores per chip).
    topology: `gcluster --topology` (chips, e.g. 8x32x32).
    compute_type: `gcluster --compute-type`.
    num_slices: Number of slices.
  """

  model: MaxTextModel
  device_type: str
  topology: str
  compute_type: str = "tpu7x"
  num_slices: int = 1


deepseek3_671b_fp8_tpu7x_8k = _add_to_model_dictionary(
    tpu7x_deepseek3_model_dict,
    MaxTextModel(
        model_name="deepseek3_671b_fp8_tpu7x_8k",
        model_type="deepseek3-671b",
        tuning_params={
            # --- Batch & sequence ---
            "per_device_batch_size": 1.0,
            "max_target_length": 4096,
            "dtype": "bfloat16",
            "mu_dtype": "bfloat16",
            "grad_dtype": "bfloat16",
            # --- Optimizer & LR schedule ---
            "opt_type": "adamw",
            "learning_rate": 2.4e-05,
            "lr_schedule_type": "cosine",
            "learning_rate_schedule_steps": 12000,
            "warmup_steps_fraction": 0.00034,
            "learning_rate_final_fraction": 0.000417,
            "adam_b1": 0.9,
            "adam_b2": 0.95,
            "adam_eps": 1e-08,
            "adam_weight_decay": 0.1,
            "gradient_clipping_threshold": 1.0,
            # --- Evaluation (C4 eval set, ep-as-cp eval mesh) ---
            "target_eval_loss": 3.6,
            "eval_interval": 1,
            "eval_steps": 1,
            "eval_start_step": 42,
            "eval_per_device_batch_size": 0.0625,
            "eval_sa_block_q": 256,
            "eval_sa_block_kv": 256,
            "eval_sa_block_kv_compute": 256,
            "context_parallel_load_balance": False,
            "custom_mesh_and_rule_for_eval": "ep-as-cp",
            # --- Dataset & tokenizer ---
            "dataset_type": "c4_mlperf",
            "dataset_path": DATASET_PATH_PLACEHOLDER,
            "dataset_name": "c4/en:3.0.5",
            "eval_dataset_name": "c4/en:3.0.9",
            "train_data_columns": "'[ids]'",
            "eval_data_columns": "'[ids]'",
            "tokenizer_path": "src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken",
            "tokenizer_type": "tiktoken",
            "tokenize_train_data": False,
            "tokenize_eval_data": False,
            "data_shuffle_seed": 2767272,
            "train_shard_in_read": True,
            "train_interleave_cycle_length": 4,
            # --- Pre-RUN_START warmups ---
            "block_state_before_run_start": True,
            "warm_input_reshard_before_run_start": True,
            "warm_eval_input_reshard_before_run_start": True,
            "eval_cache_prefill_in_background": True,
            "warmup_programs_in_init": True,
            # --- Parallelism & sharding (8k: FSDP 1024 x EP 16) ---
            "ici_fsdp_parallelism": 1024,
            "ici_expert_parallelism": 16,
            "shard_exp_on_fsdp": False,
            "custom_mesh_and_rule": "fsdp-as-dp-for-attn-rf",
            "allow_split_physical_axes": False,
            "moe_flat_fsdp_weights": True,
            # --- MoE backward / routing / RoPE optimizations ---
            "moe_bwd_prequant_before_unsort": True,
            "moe_3d_dlhs_partial_sum": True,
            "moe_combine_bwd_direct_qarray": True,
            "moe_tc_window_routing": True,
            "moe_tc_routing_checkpoint": "device",
            "rope_direct_position_freqs": True,
            "rope_dot_pairwise": True,
            # --- MoE dispatch, ragged buffers & kernels ---
            "use_ring_of_experts": True,
            "use_ragged_sort": True,
            "moe_tc_ragged_sort": True,
            "moe_tc_ragged_flatten_block_size": 256,
            "moe_tc_ragged_weights_on_activation": True,
            "moe_tc_ragged_3d_gmm": True,
            "moe_tc_ragged_3d_dispatch": True,
            "ragged_buffer_factor": 2.5,
            "first_phase_ragged_buffer_factor": 3.5,
            "retry_dropless_first_steps": 10,
            "eval_ragged_buffer_factor": 4.0,
            "log_required_ragged_buffer_factor": True,
            "moe_dropless_fallback": "step",
            "num_moe_token_chunks": 2,
            "use_custom_sort_vjp": True,
            "moe_pin_sparse_core_all_gathers": True,
            "moe_pin_sparse_core_ep_all_gathers": True,
            "moe_ep_all_gather_sparse_core_id": 1,
            "megablox": True,
            "sparse_matmul": True,
            "use_gmm_v2": True,
            "merge_gating_gmm": False,
            "moe_accumulate_chunk_wgrad": True,
            "moe_accumulate_wi_dlhs": True,
            "moe_gmm_v2_dlhs_transpose_rhs": True,
            # --- Router ---
            "router_topk_matmul_vjp": True,
            "moe_topk_before_ep_all_gather": True,
            "use_random_routing": False,
            "float32_gate_logits": True,
            "n_routing_groups": 8,
            "topk_routing_group": 4,
            "routed_bias": True,
            "routed_bias_update_rate": 0.001,
            "load_balance_loss_weight": 0.01,
            "moe_use_megatron_seq_aux_loss": True,
            "defer_small_all_reduces": True,
            # --- Multi-token prediction ---
            "mtp_num_layers": 1,
            "mtp_loss_scaling_factor": 0.1,
            "mtp_reuse_input_embedding": True,
            # --- FP8 quantization (qwix) ---
            "use_qwix_quantization": True,
            "quantization": "fp8_full",
            "weight_quantization_calibration_method": "fixed,-224,224",
            "act_quantization_calibration_method": "fixed,-224,224",
            "bwd_quantization_calibration_method": "fixed,0.00125",
            "drhs_grad_quantization_calibration_method": "fixed,0.00125",
            "logits_proj_bwd_quant_calibration_method": "absmax",
            "quantize_mtp": True,
            "moe_quantize_token_all_gather": True,
            "shard_embed_moe_on_fsdp": True,
            "quantize_router_proj": False,
            "quantize_logits_proj": False,
            # --- Attention (tokamax splash) ---
            "attention": "flash",
            "use_tokamax_splash": True,
            "sa_use_fused_bwd_kernel": True,
            "sa_fuse_reciprocal": True,
            "use_splash_scheduler": False,
            "sa_use_base2_exp": True,
            "dq_reduction_steps": 3,
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048,
            "sa_qk_diag_skip": True,
            "sa_sv_diag_skip": False,
            "sa_qk_diag_grid": 4,
            "sa_block_q_dkv": 1024,
            "sa_block_kv_dkv": 1024,
            "sa_block_kv_dkv_compute": 1024,
            "sa_block_q_dq": 512,
            "sa_block_kv_dq": 512,
            "sa_q_layout": "SEQ_MINOR",
            "sa_k_layout": "SEQ_MINOR",
            "use_max_logit_estimate": -1,
            "cost_estimate_flops_fwd": 5000000000000,
            "cost_estimate_flops_bwd": 5000000000000,
            "float32_weight_sum": False,
            "use_sliced_mla_proj": False,
            "rope_pairwise": True,
            # --- GMM v2 tiles (train + eval) ---
            "use_tokamax_gmm": True,
            "wi_tile_fwd_batch_seq": 256,
            "wi_tile_fwd_embed_dim": 7168,
            "wi_tile_fwd_mlp_dim": 2048,
            "wi_tile_dlhs_batch_seq": 256,
            "wi_tile_dlhs_embed_dim": 3584,
            "wi_tile_dlhs_mlp_dim": 2048,
            "wi_tile_drhs_batch_seq": 512,
            "wi_tile_drhs_embed_dim": 1024,
            "wi_tile_drhs_mlp_dim": 2048,
            "wo_tile_fwd_batch_seq": 256,
            "wo_tile_fwd_embed_dim": 7168,
            "wo_tile_fwd_mlp_dim": 2048,
            "wo_tile_dlhs_batch_seq": 256,
            "wo_tile_dlhs_embed_dim": 7168,
            "wo_tile_dlhs_mlp_dim": 2048,
            "wo_tile_drhs_batch_seq": 512,
            "wo_tile_drhs_embed_dim": 7168,
            "wo_tile_drhs_mlp_dim": 512,
            "eval_wi_tile_fwd_batch_seq": 128,
            "eval_wi_tile_fwd_embed_dim": 3584,
            "eval_wi_tile_fwd_mlp_dim": 2048,
            "eval_wo_tile_fwd_batch_seq": 128,
            "eval_wo_tile_fwd_embed_dim": 7168,
            "eval_wo_tile_fwd_mlp_dim": 2048,
            # --- Remat (custom: keep these activations on device) ---
            "remat_policy": "custom",
            "context": "device",
            "out_proj": "device",
            "decoder_layer_input": "device",
            "query_proj": "device",
            "key_proj": "device",
            "value_proj": "device",
            "query_wa_proj": "device",
            "kv_wa_proj": "device",
            "mla_q": "device",
            "mla_kv": "device",
            "moe_routing_maps": "device",
            "use_iota_embed": True,
            # --- Checkpoint loading (parameters only; no saves) ---
            "enable_checkpointing": True,
            "async_checkpointing": True,
            "checkpoint_storage_concurrent_gb": 1024,
            "load_parameters_path": CHECKPOINT_PATH_PLACEHOLDER,
            "save_checkpoint_on_start": False,
            "save_checkpoint_on_completion": False,
            # --- Compile-only XLA flags (single-quoted so a space-separated flag list stays one shell argument) ---
            "compile_xla_flags": "'--xla_while_loop_all_reduce_dus_code_motion_max_size_bytes=268435456'",
            # --- Logging & diagnostics ---
            "abort_on_nan_loss": True,
            "abort_on_inf_loss": True,
            "log_config": False,
            "monitor_step_time_deviation": False,
            "enable_gcp_goodput_metrics": False,
            "enable_gcp_step_deviation_metrics": False,
            "enable_tensorboard": True,
            "enable_mllog": True,
        },
        xla_flags=(xla_flags.BASE_FLAGS + xla_flags.SPLIT_SPARSE_CORE_ALL_REDUCE + xla_flags.ALL_REDUCE_COMBINER_40MIB),
    ),
)


RECIPES = {
    "8k": Tpu7xRecipe(
        model=deepseek3_671b_fp8_tpu7x_8k,
        device_type="tpu7x-16384",
        topology="8x32x32",
    ),
}
