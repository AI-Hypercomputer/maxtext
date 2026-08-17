#!/bin/bash

# Reproduces the tuned gemma4-12b training-throughput configuration on a single
# Ironwood slice (tpu7x-4x4x4 = 64 chips / 128 JAX devices).
#
# Measured: 4.703 s/step, 521.3 TFLOP/s/device, 6968 tokens/s/device
#           -> 45.3% MFU against the 1150 bf16 TFLOP/s/device peak.
#
# Run this inside a container that already has the slice attached.
#
# Defaults below are a BENCHMARK harness, not a training recipe: synthetic data,
# 30 steps and the profiler are there to measure step time. For real training set
# DATASET/STEPS and drop the profiler flags.

set -ex

MODEL_NAME='gemma4-12b'
RUN_NAME=${RUN_NAME:-"g12b-$(date +%Y-%m-%d-%H-%M)"}
BASE_OUTPUT_DIRECTORY=${BASE_OUTPUT_DIRECTORY:?set BASE_OUTPUT_DIRECTORY, e.g. gs://your-bucket/gemma4}
STEPS=${STEPS:-30}

# Ironwood collective bundle. Note we deliberately do NOT set --xla_tpu_dvfs_p_state:
# the default DVFS state is left alone.
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_offloading_sort_to_sparsecore=true \
--xla_tpu_enable_offloading_gather_to_sparsecore=true \
--xla_tpu_enable_offloading_scatter_to_sparsecore=true \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_use_tc_device_shape_on_sc=true \
--xla_sc_disable_megacore_partitioning=true \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_all_gather_offload_tracing=true"

# The mesh is the whole story: dp4 x fsdp32 beats pure FSDP-128 by +4.4 pp
# (5.211 -> 4.703 s/step) by cutting per-device weight all-gather volume. It is a
# narrow peak -- dp2 is +0.5 pp and dp8 is -0.3 pp from here.
#
# num_vocab_tiling=8 is a requirement, not a knob: the untiled 262k-vocab logits are
# ~34 GB at pbs 8 and abort.
#
# Measured-negative, so do not "optimise" these: remat relaxation
# (save_dot_except_mlp -1.1 pp, save_qkv_proj -1.4 pp, minimal OOMs at 163 G vs a
# 94.75 G budget), context=remat (-1.3 pp), seq 8192 (-1.4 pp).
python3 -m maxtext.trainers.pre_train.train \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    per_device_batch_size=8 \
    max_target_length=4096 \
    dtype=bfloat16 \
    num_vocab_tiling=8 \
    remat_policy=custom \
    context=device \
    use_iota_embed=True \
    allow_split_physical_axes=True \
    attention=flash \
    use_tokamax_splash=True \
    sa_use_fused_bwd_kernel=True \
    sa_block_q=1024 sa_block_kv=1024 sa_block_kv_compute=512 \
    sa_block_q_dkv=1024 sa_block_kv_dkv=1024 sa_block_kv_dkv_compute=256 \
    ici_data_parallelism=4 \
    ici_fsdp_parallelism=32 \
    ici_fsdp_transpose_parallelism=1 \
    opt_type=adamw \
    dataset_type=synthetic \
    steps=${STEPS} \
    enable_checkpointing=False \
    async_checkpointing=False \
    skip_jax_distributed_system=True \
    profiler=xplane \
    skip_first_n_steps_for_profiler=10 \
    profiler_steps=2 \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    run_name=${RUN_NAME}
