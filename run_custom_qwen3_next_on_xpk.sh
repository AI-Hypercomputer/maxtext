#!/bin/bash
set -e

# Activate Python virtual environment
source ${HOME}/.local/bin/venv/bin/activate

# --- Environment Variables ---
export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_NAME="bodaborg-v6e-256-lcscld-c"
export ZONE="southamerica-west1-a"

# --- Configuration & Automated Image Build ---
TIMESTAMP=$(date +%m%d%H%M%S)
export WORKLOAD_IMAGE="gcr.io/tpu-prod-env-one-vm/param3_21jul:dipakg_${TIMESTAMP}"
export WORKLOAD_NAME="dipakg-qn80b-${TIMESTAMP}"
export DEVICE_TYPE="v6e-256"
export NUM_SLICES=1
export PRIORITY="very-high"
export MAX_RESTARTS=0
export NUM_STEPS=15
export MODEL_NAME="qwen3-next-80b-a3b"
export BASE_OUTPUT_DIR="/tmp/"

echo "========================================================================"
echo "Building and uploading Docker runner image from /usr/local/google/home/dipakg/workspace/qwen3_5_maxtext"
echo "Target Image: ${WORKLOAD_IMAGE}"
echo "========================================================================"

(
  cd /usr/local/google/home/dipakg/workspace/qwen3_5_maxtext && \
  CLOUD_IMAGE_NAME="${WORKLOAD_IMAGE}" \
       BASE_IMAGE="gcr.io/tpu-prod-env-one-vm/param3_21jul:latest" \
       bash src/dependencies/scripts/docker_upload_runner.sh
)

echo "Docker image upload complete: ${WORKLOAD_IMAGE}"

# --- XLA Flags ---
XLA_FLAGS_ARRAY=(
  "--xla_msa_enable_sync_slice_replacement=false"
  "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
  "--xla_msa_enable_sync_copy_replacement=false"
  "--xla_tpu_scoped_vmem_limit_kib=81000"
  "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
  "--xla_tpu_enable_concurrent_sparse_core_offloading=true"
  "--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true"
  "--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
  "--xla_tpu_use_single_sparse_core_for_all_gather_offload=true"
  "--xla_tpu_sparse_core_all_gather_latency_multiplier=1"
  "--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3"
  "--xla_tpu_offload_gather_to_sparsecore=true"
  "--xla_tpu_dvfs_p_state=7"
  "--xla_tpu_disable_sparse_core_collective_offload_remover=true"
  "--xla_tpu_use_tc_device_shape_on_sc=true"
  "--xla_sc_enable_instruction_fusion=false"
  "--xla_sc_disable_megacore_partitioning=true"
  "--xla_tpu_enable_async_collective_fusion=true"
  "--xla_tpu_overlap_compute_collective_tc=true"
  "--xla_tpu_enable_async_collective_fusion_multiple_steps=true"
  "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
  "--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false"
  "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
  "--xla_tpu_enable_latency_hiding_scheduler=true"
  "--xla_latency_hiding_scheduler_rerun=10"
  "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative"
  "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative"
  "--xla_latency_hiding_scheduler_enable_selective_resources=true"
  "--xla_tpu_enable_ilp_latency_hiding_scheduler=true"
  "--xla_tpu_enable_all_experimental_scheduler_features=true"
  "--xla_tpu_enable_scheduler_memory_pressure_tracking=true"
  "--xla_tpu_host_transfer_overlap_limit=4"
  "--xla_tpu_aggressive_opt_barrier_removal=ENABLED"
  "--xla_lhs_prioritize_async_depth_over_stall=ENABLED"
  "--xla_tpu_enable_ag_backward_pipelining=true"
  "--xla_should_allow_loop_variant_parameter_in_chain=ENABLED"
  "--xla_should_add_loop_invariant_op_in_chain=ENABLED"
  "--xla_max_concurrent_host_send_recv=100"
  "--xla_tpu_scheduler_percent_shared_memory_limit=150"
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
  "custom_mesh=hybrid_ring_64x4"
  "use_ragged_sort=True"
  "use_random_routing=True"
  "num_moe_token_chunks=2"
  "per_device_batch_size=8"
  "opt_type=adamw"
  "max_target_length=2048"
  "ragged_buffer_factor=1.5"
  "remat_policy=full"
  "reuse_example_batch=1"
  "decoder_layer_input=device"
  "ici_fsdp_parallelism=-1"
  "steps=15"
  "sa_q_layout=SEQ_MINOR"
  "sa_k_layout=HEAD_DIM_MINOR"
  "sa_v_layout=HEAD_DIM_MINOR"
  "sa_block_q=1024"
  "sa_block_kv=1024"
  "sa_block_kv_compute=512"
  "sa_block_q_dkv=1024"
  "sa_block_kv_dkv=1024"
  "sa_block_kv_dkv_compute=1024"
  "sa_fuse_reciprocal=false"
  "use_splash_scheduler=true"
  "sa_use_base2_exp=true"
  "dq_reduction_steps=3"
  "hardware=tpu"
  "skip_jax_distributed_system=False"
  "attention=flash"
  "use_tokamax_splash=True"
  "sa_use_fused_bwd_kernel=True"
  "sparse_matmul=True"
  "megablox=True"
  "wi_tile_fwd_batch_seq=512"
  "wi_tile_dlhs_batch_seq=512"
  "wi_tile_drhs_batch_seq=512"
  "wo_tile_fwd_batch_seq=512"
  "wo_tile_dlhs_batch_seq=512"
  "wo_tile_drhs_batch_seq=512"
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
  "optimizer_memory_host_offload=False"
  "parameter_memory_host_offload=False"
  "enable_checkpointing=False"
  "async_checkpointing=False"
  "tokenizer_type=tiktoken"
  "tokenizer_path=tokenizer_74B/"
  "override_model_config=true"
  "mhc_expansion_rate=4"
  "use_gdn_kernel=True"
  "use_hybrid_gdn=True"
  "profiler=xplane"
  "profiler_steps=4"
  "skip_first_n_steps_for_profiler=2"
  "enable_tpu_profiling_options=True"
  "upload_all_profiler_results=False"
)
MAXTEXT_ARGS="${MAXTEXT_ARGS_ARRAY[*]}"

# The command to run inside the container
RUN_COMMAND="set -e && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
export JAX_DISTRIBUTED_INITIALIZE_TIMEOUT=1800 && \
export PYTHONPATH='/deps/src:\$PYTHONPATH' && \
python3 -m maxtext.trainers.pre_train.train /deps/src/maxtext/configs/base.yml ${MAXTEXT_ARGS}"

# --- XPK Workload Creation ---
echo "Creating XPK workload: ${WORKLOAD_NAME} on cluster: ${CLUSTER_NAME}"

PYTHONPATH=/usr/local/google/home/dipakg/workspace/maxtext_temp/xpk/src python3 -m xpk.main workload create \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --priority="${PRIORITY}" \
  --max-restarts="${MAX_RESTARTS}" \
  --device-type="${DEVICE_TYPE}" \
  --num-slices="${NUM_SLICES}" \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="${RUN_COMMAND}"

LOGS_URL="https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Aresource.labels.project_id%3D%22${PROJECT_ID}%22%0Aresource.labels.location%3D%22southamerica-west1%22%0Aresource.labels.cluster_name%3D%22${CLUSTER_NAME}%22%0Aresource.labels.namespace_name%3D%22default%22%0Aresource.labels.pod_name%3A%22${WORKLOAD_NAME}-slice-job-0-0-%22%0Aseverity%3E%3DDEFAULT;storageScope=project;duration=P1D?project=${PROJECT_ID}"
GKE_URL="https://console.cloud.google.com/kubernetes/service/southamerica-west1/${CLUSTER_NAME}/default/${WORKLOAD_NAME}/details?project=${PROJECT_ID}"

echo "========================================================================"
echo "📋 Pantheon Cloud Logging (Worker 0 Logs):"
echo "${LOGS_URL}"
echo ""
echo "☸️ GKE Workload Details:"
echo "${GKE_URL}"
echo ""
echo "========================================================================"
