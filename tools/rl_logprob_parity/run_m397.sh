#!/bin/bash
# env: shared conda + private --no-deps dir + tpu-inference/vllm/maxtext(pr4925) worktrees
export PATH=/mnt/disks/persist/vllm_conda/bin:$PATH
export TPUINF_WT=/home/wenxindong_google_com/work/tpu-inference/.claude/worktrees/qwen35-maxtext-mapping
export VLLM_WT=/home/wenxindong_google_com/work/vllm/.claude/worktrees/qwen35-mapping-d626108b
export MAXTEXT_WT=/home/wenxindong_google_com/work/maxtext/.claude/worktrees/pr4925
export PYDEPS=/home/wenxindong_google_com/work/pydeps_rl_mapping
export PYTHONPATH=$PYDEPS:$TPUINF_WT:$VLLM_WT:$MAXTEXT_WT/src
export NEW_MODEL_DESIGN=1
export VLLM_TARGET_DEVICE=tpu
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OUT_DIR=/mnt/disks/persist/pr4925_repro
mkdir -p $OUT_DIR
cd $MAXTEXT_WT
exec python /home/wenxindong_google_com/.claude/jobs/7ea88918/tmp/maxtext_prefix_397b.py "$@"
