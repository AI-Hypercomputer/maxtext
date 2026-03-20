p_state=3
date_str=$(date +%s|tail -c5)
run_name=$USER-perf-$date_str
docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:2026-03-17"
#docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:qinwen-2026-03-10" to old
#docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:qinwen-2026-03-17" #libtpu_nightly_20260315_a_RC00, split=2 error
#docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:qinwen-2026-03-18-1" # gai patch 
#docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:qinwen-2d-2026-03-18" #03-18 head 
docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:qinwen-2026-03-19" # 0319 head, 2k pdbs=4 failed
cluster=bodaborg-super-xpk-v54
#scale=tpu7x-512
DCN_TFSDP=1

# scale=tpu7x-1024
# ICI_DATA=1
# EP=8
# FSDP=128
# ICI_TFSDP=1

scale=tpu7x-4x16x16
ICI_DATA=1
EP=4
FSDP=256
ICI_TFSDP=2

# scale=tpu7x-8x16x16
# ICI_DATA=1
# EP=8
# FSDP=256
# ICI_TFSDP=2

# scale=tpu7x-4x16x32
# ICI_DATA=1
# EP=4
# FSDP=64
# ICI_TFSDP=16

# scale=tpu7x-4x32x32
# ICI_DATA=1
# EP=4
# FSDP=64
# ICI_TFSDP=32

# scale=tpu7x-4x16x64
# ICI_DATA=1
# EP=4
# FSDP=128
# ICI_TFSDP=16

# not working
# scale=tpu7x-4x8x128
# ICI_DATA=1
# EP=8
# FSDP=256
# ICI_TFSDP=4

# best config
# scale=tpu7x-8x8x64
# ICI_DATA=1
# EP=8
# FSDP=128
# ICI_TFSDP=8

# scale=tpu7x-4x8x128
# ICI_DATA=1
# EP=4
# FSDP=256
# ICI_TFSDP=8

# scale=tpu7x-4x16x16
# ICI_DATA=1
# EP=4
# FSDP=256
# ICI_TFSDP=2
# DCN_TFSDP=4

# ICI_DATA=1
# EP=4
# FSDP=128
# ICI_TFSDP=1
# DCN_TFSDP=8

eval_interval=20
STEPS=10
seeds=0
augment=${1:-1}

if [[ $augment == 1 ]]; then
  PDBS=1
  SPLIT=1
  aggregate=false
  AUG_MAXTEXT_ARGS="mlpwi_0=device \
  mlpwi_1=device \
  context=device"
elif [[ $augment == 2 ]]; then
  PDBS=2
  SPLIT=2
  aggregate=true
  AUG_MAXTEXT_ARGS="mlpwi_0=device \
  context=device"
elif [[ $augment == 4 ]]; then
  PDBS=4
  SPLIT=1
  aggregate=true
  AUG_MAXTEXT_ARGS=""
fi
# mtp_num_layers=1 mtp_loss_scaling_factor=0.1 \
#load_parameters_path=gs://maxtext-experiments-tpem/qinwen/qinwen-dsv3-conv-seed-4536-1767145923/checkpoints/900/items 
# checkpoint_storage_concurrent_gb=900 enable_checkpointing=true load_parameters_path='gs://maxtext-model-checkpoints/deepseek3-671b/0/items' \
#gs://maxtext-experiments-tpem/qinwen/qinwen-dsv3-conv-seed-11882-1767806183/checkpoints/900
checkpoint_period=10000
final_seed=$((seeds + 100))

BASE_OUTPUT_DIR='gs://maxtext-experiments-tpem/qinwen/perf/pdbs${PDBS}'
XLA_FLAGS="--xla_tpu_dvfs_p_state=7 \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_num_sparse_cores_for_gather_offloading=1 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=false \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_enable_all_reduce_offload_tracing=false \
  --xla_max_concurrent_async_all_gathers=2 \
  --xla_max_concurrent_async_reduce_scatters=2 \
  --xla_tpu_enable_ici_ag_pipelining=true \
  --xla_tpu_enable_ici_rs_pipelining=true \
  --xla_tpu_aggregate_data_dependent_sc_ops=${aggregate} \
  --xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=true "

MAXTEXT_ARGS="\
merge_gating_gmm=true \
skip_jax_distributed_system=false \
per_device_batch_size=$PDBS \
max_target_length=4096 \
use_2d_fsdp_sharding=false \
shard_exp_on_fsdp=false \
use_batch_split_schedule=true \
dcn_data_parallelism=$DCN_TFSDP \
ici_data_parallelism=$ICI_DATA \
ici_fsdp_transpose_parallelism=$ICI_TFSDP \
ici_fsdp_parallelism=$FSDP \
ici_expert_parallelism=$EP \
batch_split_factor=$SPLIT \
remat_policy=custom \
decoder_layer_input=offload \
${AUG_MAXTEXT_ARGS} \
mla_q="offload" \
mla_kv="offload" \
gcs_metrics=True \
use_iota_embed=True \
use_custom_sort_vjp=True \
dataset_path=gs://max-datasets-rogue \
dataset_type=synthetic \
dataset_name='c4/en:3.0.7' \
eval_dataset_name='c4/en:3.0.9' \
train_split=train2 \
reuse_example_batch=0 \
tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken \
enable_checkpointing=false \
checkpoint_period=$checkpoint_period \
checkpoint_storage_concurrent_gb=900 \
learning_rate=1e-4 warmup_steps_fraction=0.00267 learning_rate_schedule_steps=12000 \
eval_per_device_batch_size=1 eval_interval=$eval_interval eval_steps=1 \
data_shuffle_seed=$final_seed \
skip_first_n_steps_for_profiler=5 \
profiler_steps=1 \
profiler=xplane \
steps=$STEPS \
attention=flash \
megablox=True \
sparse_matmul=True \
use_tokamax_gmm=true \
use_tokamax_splash=true \
use_max_logit_estimate=-1 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000  \
sa_use_fused_bwd_kernel=True \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=512 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_kv_dq=2048 \
sa_block_q_dq=2048 \
wi_tile_fwd_batch_seq=512 \
wi_tile_fwd_embed_dim=7168 \
wi_tile_fwd_mlp_dim=512 \
wi_tile_dlhs_batch_seq=512 \
wi_tile_dlhs_embed_dim=512 \
wi_tile_dlhs_mlp_dim=7168 \
wi_tile_drhs_batch_seq=512 \
wi_tile_drhs_embed_dim=7168 \
wi_tile_drhs_mlp_dim=512 \
wo_tile_fwd_batch_seq=512 \
wo_tile_fwd_embed_dim=512 \
wo_tile_fwd_mlp_dim=7168 \
wo_tile_dlhs_batch_seq=512 \
wo_tile_dlhs_embed_dim=7168 \
wo_tile_dlhs_mlp_dim=512 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_embed_dim=512 \
wo_tile_drhs_mlp_dim=7168 \
opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 dtype=bfloat16 \
load_balance_loss_weight=1e-4 routed_bias_update_rate=0.001 \
target_eval_loss=2.7 \
model_name=deepseek3-671b-2dfsdp \
base_output_directory=${BASE_OUTPUT_DIR} \
use_vertex_tensorboard=false \
run_name=$run_name " 
#old=9234a525c3237e3ee36b93a48e31eaa9378ee057
#02118e34d42deb5ba00cbae7f847d7bc61254a00f062
#a04710eb79691080705afa97c60a543d117c41a5
#1a2a13022f2d55f04147b2340253c59e3517f7e8
#  --env="TPU_ACCELERATOR_TYPE=$scale" \
#  --env="TPU_CHIPS_PER_HOST_BOUNDS=2,2,1" \
xpk workload create --cluster=$cluster --project=cloud-tpu-multipod-dev --zone=us-central1-ai1a  --device-type=$scale \
--num-slices=$DCN_TFSDP \
  --command="export LIBTPU_INIT_ARGS='$XLA_FLAGS' \
        && export ENABLE_PATHWAYS_PERSISTENCE=1 && export JAX_PLATFORMS=tpu,cpu && export ENABLE_PJRT_COMPATIBILITY=true && export MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets MAXTEXT_PKG_DIR=/app/src/maxtext && \
        cd /app && pip install --no-deps -e . && \
        python3 -m src.maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS}; " \
        --base-docker-image=$docker_image --enable-debug-logs --workload=$run_name --priority=high --max-restarts=0