#!/bin/bash
set -e

# --- Configuration ---
TIMESTAMP=$(date +%m%d%H%M%S)
export PROJECT_ID="cloud-tpu-multipod-dev"
export CLUSTER_NAME="mlperf-v5p"
export ZONE="europe-west4-b"
export WORKLOAD_IMAGE="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-08-05"
export WORKLOAD_NAME="mohit-qn80b-v5p64-${TIMESTAMP}"
export DEVICE_TYPE="v5p-64"
export NUM_SLICES=1
export NUM_STEPS=15
export MODEL_NAME="qwen3-next-80b-a3b"
export BASE_OUTPUT_DIR="gs://runner-maxtext-logs/offload_profile/run-${TIMESTAMP}"

# --- XLA Flags ---
XLA_FLAGS_ARRAY=(
  "--xla_tpu_scheduler_percent_shared_memory_limit=35"
  "--xla_msa_enable_sync_slice_replacement=false"
  "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
  "--xla_msa_enable_sync_copy_replacement=false"
  "--xla_tpu_scoped_vmem_limit_kib=81000"
  "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
  "--xla_tpu_offload_gather_to_sparsecore=true"
  "--xla_tpu_dvfs_p_state=7"
  "--xla_tpu_disable_sparse_core_collective_offload_remover=true"
  "--xla_tpu_enable_async_collective_fusion=true"
  "--xla_tpu_overlap_compute_collective_tc=true"
  "--xla_tpu_enable_async_collective_fusion_multiple_steps=true"
  "--xla_tpu_enable_latency_hiding_scheduler=true"
  "--xla_latency_hiding_scheduler_rerun=10"
  "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative"
  "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative"
  "--xla_latency_hiding_scheduler_enable_selective_resources=true"
  "--xla_tpu_enable_ilp_latency_hiding_scheduler=true"
  "--xla_tpu_enable_all_experimental_scheduler_features=true"
  "--xla_tpu_enable_scheduler_memory_pressure_tracking=true"
  "--xla_tpu_host_transfer_overlap_limit=24"
  "--xla_tpu_aggressive_opt_barrier_removal=ENABLED"
  "--xla_lhs_prioritize_async_depth_over_stall=DISABLED"
  "--xla_tpu_enable_ag_backward_pipelining=true"
  "--xla_should_allow_loop_variant_parameter_in_chain=ENABLED"
  "--xla_should_add_loop_invariant_op_in_chain=ENABLED"
  "--xla_max_concurrent_host_send_recv=100"
)
export XLA_FLAGS="${XLA_FLAGS_ARRAY[*]}"

# --- MaxText Workload Overrides ---
MAXTEXT_ARGS_ARRAY=(
  "model_name=${MODEL_NAME}"
  "base_output_directory=${BASE_OUTPUT_DIR}"
  "run_name=param-3"
  "dataset_type=synthetic"
  "dataset_name=synthetic"
  "dtype=bfloat16"
  "allow_split_physical_axes=True"
  "ici_expert_parallelism=4"
  "use_ring_of_experts=True"
  "use_ragged_sort=True"
  "use_random_routing=True"
  "per_device_batch_size=3"
  "opt_type=muon"
  "muon_consistent_rms=0.2"
  "muon_weight_decay=0.1"
  "max_target_length=2048"
  "ragged_buffer_factor=1.5"
  "remat_policy=custom"
  "reuse_example_batch=1"
  "decoder_layer_input=offload"
  "context=device"
  "ici_fsdp_parallelism=-1"
  "steps=15"
  "sa_q_layout=SEQ_MINOR"
  "sa_k_layout=HEAD_DIM_MINOR"
  "sa_v_layout=HEAD_DIM_MINOR"
  "sa_block_q=2048"
  "sa_block_kv=2048"
  "sa_block_kv_compute=1024"
  "sa_block_q_dkv=2048"
  "sa_block_kv_dkv=2048"
  "sa_block_kv_dkv_compute=1024"
  "hardware=tpu"
  "skip_jax_distributed_system=False"
  "attention=flash"
  "use_tokamax_splash=True"
  "sa_use_fused_bwd_kernel=True"
  "sparse_matmul=True"
  "megablox=True"
  "wi_tile_fwd_batch_seq=128"
  "wi_tile_dlhs_batch_seq=128"
  "wi_tile_drhs_batch_seq=128"
  "wo_tile_fwd_batch_seq=128"
  "wo_tile_dlhs_batch_seq=128"
  "wo_tile_drhs_batch_seq=128"
  "wi_tile_fwd_embed_dim=3072"
  "wi_tile_fwd_mlp_dim=1536"
  "wi_tile_dlhs_embed_dim=3072"
  "wi_tile_dlhs_mlp_dim=1536"
  "wi_tile_drhs_embed_dim=3072"
  "wi_tile_drhs_mlp_dim=1536"
  "wo_tile_fwd_embed_dim=3072"
  "wo_tile_fwd_mlp_dim=1536"
  "wo_tile_dlhs_embed_dim=3072"
  "wo_tile_dlhs_mlp_dim=1536"
  "wo_tile_drhs_embed_dim=3072"
  "wo_tile_drhs_mlp_dim=1536"
  "use_tokamax_gmm=True"
  "use_gmm_v2=True"
  "optimizer_memory_host_offload=True"
  "parameter_memory_host_offload=True"
  "enable_checkpointing=False"
  "async_checkpointing=False"
  "tokenizer_type=tiktoken"
  "tokenizer_path=tokenizer_74B/"
  "override_model_config=true"
  "mhc_expansion_rate=4"
  "use_gdn_kernel=False"
  "use_hybrid_gdn=False"
  "profiler=xplane"
  "profiler_steps=5"
  "skip_first_n_steps_for_profiler=2"
  "enable_tpu_profiling_options=True"
  "upload_all_profiler_results=true"
)
MAXTEXT_ARGS="${MAXTEXT_ARGS_ARRAY[*]}"

# Command to run inside the container
RUN_COMMAND="set -e && \
export LIBTPU_INIT_ARGS=\"${XLA_FLAGS}\" && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
export JAX_DISTRIBUTED_INITIALIZE_TIMEOUT=1800 && \
export PYTHONPATH=/app/src:/app:\$PYTHONPATH && \
python3 -u -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml ${MAXTEXT_ARGS}"

# --- Launch Workload ---
echo "Creating XPK workload: ${WORKLOAD_NAME} on cluster: ${CLUSTER_NAME}"
/usr/local/google/home/mohitkhatwani/max_venv/bin/xpk workload create \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --device-type="${DEVICE_TYPE}" \
  --num-slices="${NUM_SLICES}" \
  --base-docker-image="${WORKLOAD_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  --command="${RUN_COMMAND}"
