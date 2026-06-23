#!/bin/bash
WORKLOAD_NAME="mohitkhatwani-vb-$RANDOM-xpk"

/usr/local/google/home/mohitkhatwani/max_venv/bin/python3 /usr/local/google/home/mohitkhatwani/xpk/xpk.py workload create \
  --cluster=bodaborg-super-xpk-x8p \
  --project=cloud-tpu-multipod-dev \
  --zone=us-central1-ai1a \
  --workload="${WORKLOAD_NAME}" \
  --tpu-type=tpu7x-128 \
  --num-slices=1 \
  --priority=very-high \
  --no-use-parallel-containers \
  --base-docker-image=gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-runner:post-training-2026-06-23 \
  --script-dir=/usr/local/google/home/mohitkhatwani/maxtext \
  --env PHASED_PROFILING_DIR="gs://runner-maxtext-logs/${WORKLOAD_NAME}/vllm-profile/" \
  --command="bash vllm_bench_runner.sh"
