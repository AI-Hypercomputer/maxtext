#!/bin/bash

# Exit on error
set -e

echo "Running end-to-end test for MMLU evaluating qwen3-0.6b without pre-trained checkpoints (randomly initialized weights)"
echo "Since we are running with steps=5, this just verifies that no OOM issues exist, data loaders work, and the pipeline crashes are prevented."

MODEL_NAME="qwen3-0.6b"
CONFIG_FILE="${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}/base.yml"

PYTHONPATH=src:. python3 -m benchmarks.mmlu.mmlu_eval_perplexity ${CONFIG_FILE} \
    model_name=${MODEL_NAME} \
    per_device_batch_size=1 \
    max_target_length=128 \
    steps=5

if [ $? -eq 0 ]
then
    echo "Successfully ran mmlu_eval_perplexity for $MODEL_NAME."
else
    echo "Failed evaluating mmlu_eval_perplexity for $MODEL_NAME."
    exit 1
fi