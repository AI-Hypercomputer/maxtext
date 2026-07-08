#!/bin/bash
set -ex

export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_id=$(date +%Y-%m-%d-%H-%M-%S)

export PYTHONPATH=/home/yixuannwang_google_com/projects/tunix:${PYTHONPATH}

bash test_qwen3_to_mt.sh $run_id
bash test_qwen3_rl.sh $run_id
