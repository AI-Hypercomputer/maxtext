#!/bin/bash
export PATH=/mnt/disks/persist/vllm_conda/bin:$PATH
export TPUINF_WT=/home/wenxindong_google_com/work/tpu-inference/.claude/worktrees/qwen35-maxtext-mapping
export VLLM_WT=/home/wenxindong_google_com/work/vllm/.claude/worktrees/qwen35-mapping-d626108b
export PYDEPS=/home/wenxindong_google_com/work/pydeps_rl_mapping
export PYTHONPATH=$PYDEPS:$TPUINF_WT:$VLLM_WT
export HF_HOME=/mnt/disks/persist HF_HUB_OFFLINE=1
# production serving env (from the user's vllm serve command)
export MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=4 \
  ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 SLICE_ROPE_CACHE=1 DP_SCHED_BATCH_PREFILL=false \
  NEW_MODEL_DESIGN=1 NUM_PRECOMPILE_WORKERS=8 SKIP_JAX_PRECOMPILE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ENGINE_READY_TIMEOUT_S=7200
export LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'
cd /home/wenxindong_google_com/.claude/jobs/7ea88918/tmp
exec python /home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/vllm_generate.py "$@"
