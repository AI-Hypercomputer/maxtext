#!/bin/bash
set -e

# Baseline variant of run_qwen3_80b_xpk.sh.
#
# Diffs vs the parent script (everything else is byte-identical):
#   1. context=offload -> context=remat
#      The activation-offload baseline: decoder_layer_input still goes to host
#      memory, but the attention context is recomputed instead of round-tripping
#      268 MB/layer over PCIe. Profile of run-0812030226 showed the context read
#      costing 8.4 ms/layer at 31.9 GB/s vs ~3.3 ms/layer to just re-run splash
#      attention forward.
#   2. profiler_steps 2 -> 1 and tpu_num_sparse_core_tiles_to_trace 1 -> 0.
#      Profiler-only change, no effect on model math. The previous capture was
#      TRUNCATED at 69% of the step because SparseCore TEC tile tracing emitted
#      4.82M of the 5.0M trace events and filled the buffer. Dropping tile
#      tracing keeps the SparseCore OFFLOAD_* breakdown (that lives on the
#      "SparseCore Offload Type" thread, ~1.4k events) while losing only the
#      per-tile TEC noise.
#   3. Reuses the already-built image from run-0812030226 instead of rebuilding,
#      so the code under test is bit-identical to the profiled run.

source /mnt/data/workspace/max_venv/bin/activate

# --- Environment Variables ---
export PROJECT_ID="tpu-prod-env-one-vm"
export CLUSTER_NAME="bodaborg-v6e-256-lcscld-c"
export ZONE="southamerica-west1-a"

TIMESTAMP=$(date +%m%d%H%M%S)
# Reuse the image that produced the run-0812030226 profile (no code changes here).
export WORKLOAD_IMAGE="gcr.io/tpu-prod-env-one-vm/param3_21jul:mohit_0812030226"
export WORKLOAD_NAME="mohit-qn80b-base-${TIMESTAMP}"
export DEVICE_TYPE="v6e-256"
export NUM_SLICES=1
export PRIORITY="very-high"
export MAX_RESTARTS=0
export MODEL_NAME="qwen3-next-80b-a3b"
export BASE_OUTPUT_DIR="gs://runner-maxtext-logs/qwen3-next-80b-profiles/baseline-ctxremat-${TIMESTAMP}"

echo "========================================================================"
echo "BASELINE RUN (decoder_layer_input=offload, context=remat)"
echo "  workload : ${WORKLOAD_NAME}"
echo "  image    : ${WORKLOAD_IMAGE}  (reused, no rebuild)"
echo "  output   : ${BASE_OUTPUT_DIR}"
echo "========================================================================"

# --- XLA Flags (identical to parent script) ---
XLA_FLAGS_ARRAY=(
  "--xla_msa_enable_sync_slice_replacement=false"
  "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true"
  "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
  "--xla_msa_enable_sync_copy_replacement=false"
  "--xla_tpu_scoped_vmem_limit_kib=78500"
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
  "--xla_tpu_scheduler_percent_shared_memory_limit=100"
  "--xla_tpu_rematerialization_use_host_memory_offload=true"
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
  "remat_policy=custom"
  "reuse_example_batch=1"
  "decoder_layer_input=offload"
  "context=remat"                     # <-- THE CHANGE (was: offload)
  "ici_fsdp_parallelism=-1"
  "steps=20"
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
  "profiler_steps=1"                        # <-- was 2, halves trace volume
  "skip_first_n_steps_for_profiler=2"
  "enable_tpu_profiling_options=True"
  "tpu_num_sparse_core_tiles_to_trace=0"    # <-- was 1; kills the TEC event flood
  "upload_all_profiler_results=False"
)
MAXTEXT_ARGS="${MAXTEXT_ARGS_ARRAY[*]}"

RUN_COMMAND="set -e && \
export LIBTPU_INIT_ARGS=\"${XLA_FLAGS}\" && \
export JAX_PLATFORMS='tpu,cpu' && \
export ENABLE_PJRT_COMPATIBILITY='true' && \
export JAX_DISTRIBUTED_INITIALIZE_TIMEOUT=1800 && \
export PYTHONPATH=/deps:/deps/src:/deps/src/maxtext/src && \
python3 src/maxtext/trainers/pre_train/train.py src/maxtext/configs/base.yml ${MAXTEXT_ARGS}"

echo "Creating XPK workload: ${WORKLOAD_NAME} on cluster: ${CLUSTER_NAME}"

PYTHONPATH=/mnt/data/workspace/xpk/src python3 -m xpk.main workload create \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --priority="${PRIORITY}" \
  --max-restarts="${MAX_RESTARTS}" \
  --device-type="${DEVICE_TYPE}" \
  --num-slices="${NUM_SLICES}" \
  --docker-image="${WORKLOAD_IMAGE}" \
  --workload="${WORKLOAD_NAME}" \
  --command="${RUN_COMMAND}"

echo "========================================================================"
echo "WORKLOAD_NAME=${WORKLOAD_NAME}"
echo "PROFILE_DIR=${BASE_OUTPUT_DIR}/param-3/tensorboard/plugins/profile/"
echo "  kubectl logs -f ${WORKLOAD_NAME}-slice-job-0-0-<pod> "
echo "========================================================================"
