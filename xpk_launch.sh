#!/bin/bash

set -ex

# -- Model configuration --
# The MaxText model name. See `src/maxtext/configs/types.py` for `ModelName` for a
# full list of supported models.
MODEL="qwen3.5-35b-a3b" # e.g., 'llama3.1-8b-Instruct'

# -- MaxText configuration --
# Use a GCS bucket you own to store logs and checkpoints. Ideally in the same
# region as your TPUs to minimize latency and costs.
# You can list your buckets and their locations in the
# [Cloud Console](https://console.cloud.google.com/storage/browser).
BASE_OUTPUT_DIRECTORY="gs://snehalv-data/qwen3-5/unscanned" # e.g., gs://my-bucket/maxtext-runs

# An arbitrary string to identify this specific run.
# We recommend to include the model, user, and timestamp.
# Note: Kubernetes requires workload names to be valid DNS labels (lowercase, no underscores or periods).
# RUN_NAME="qwen35-sft-testing2"

STEPS=5 # e.g., 1000
PER_DEVICE_BATCH_SIZE=1 # e.g., 1

# -- Dataset configuration --
DATASET_NAME="HuggingFaceH4/ultrachat_200k" # e.g., HuggingFaceH4/ultrachat_200k
TRAIN_SPLIT="train_sft" # e.g., train_sft
TRAIN_DATA_COLUMNS='["messages"]'
MAXTEXT_CKPT_PATH="gs://rbierneni-deepseekv3-1/Qwen3.5/scanned/qwen35-35b-a3b/0/items"
# MAXTEXT_CKPT_PATH="gs://rbierneni-deepseekv3-1/Qwen3.5/unscanned/qwen35-35b-a3b/0/items"
RUN_NAME="qwen3-5-unscanned-sft"

CMD="unset PYTHONPATH && pip install --no-deps -e . && python3 -m maxtext.trainers.post_train.sft.train_sft \
    maxtext/configs/base.yml \
    load_parameters_path=${MAXTEXT_CKPT_PATH} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL?} \
    max_num_checkpoints_to_keep=1 \
    per_device_batch_size=1 \
    steps=10 \
    dataset_type=hf \
    hf_path=HuggingFaceH4/ultrachat_200k \
    train_split=train_sft \
    tokenizer_path=Qwen/Qwen3.5-35B-A3B \
    attention=flash \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    scan_layers=True \
    use_sft=True \
    train_data_columns=['messages'] \
    use_tokamax_splash=True \
    sparse_matmul=True \
    megablox=True \
    opt_type=sgd \
    max_target_length=256"

# # We add a Python one-liner to patch 'tpu_inference/utils.py' on the fly 
# # so it only queries memory_stats for local devices attached to the current process.
RUN_NAME="qwen3-5-decode1"
CMD="pip install --no-deps -e . && pip install --no-deps https://github.com/AI-Hypercomputer/JetStream/archive/29329e8e73820993f77cfc8efe34eb2a73f5de98.zip && python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
    base_output_directory=gs://snehalv-data \
    load_parameters_path=gs://snehalv-data/qwen3-5/unscanned/qwen3.5-35b-a3b_2026-06-09-01-41/checkpoints/10/unscanned_checkpoint/0/items \
    run_name=q3-decode \
    per_device_batch_size=1 \
    model_name=qwen3.5-35b-a3b \
    max_prefill_predict_length=64 \
    max_target_length=128 \
    tokenizer_type=huggingface \
    tokenizer_path=Qwen/Qwen3.5-35B-A3B \
    attention=dot_product \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    megablox=True \
    sparse_matmul=True \
    ici_tensor_parallelism=2 \
    ici_fsdp_parallelism=-1 \
    ici_expert_parallelism=1 \
    prompt=\"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is attention in modern day LLMs?<|im_end|>\n<|im_start|>assistant\n\" \
    scan_layers=False"

GKE_CLUSTER="mlperf-v5p"
BASE_OUTPUT_DIRECTORY="gs://snehalv-data/${RUN_NAME?}/"
NUM_SLICES="1"
DOCKER_IMAGE="gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:latest"
# DOCKER_IMAGE="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest"
# DOCKER_IMAGE="maxtext_base_image"

./xpk_venv/bin/xpk workload create \
  --cluster ${GKE_CLUSTER} \
  --zone europe-west4-b \
  --project cloud-tpu-multipod-dev \
  --workload ${RUN_NAME} \
  --docker-image "${DOCKER_IMAGE}" \
  --tpu-type v5p-64 \
  --num-slices ${NUM_SLICES?} \
  --command "cd .. && rm -rf /deps/* && git clone -b snehalv-qwen35-sft-fix https://github.com/AI-Hypercomputer/maxtext.git && cp -rf ./maxtext/ /deps/  && cd .. && export MAXTEXT_REPO_ROOT=/deps/maxtext && pip install --no-deps -e ./maxtext && cd maxtext && ${CMD}"

# xpk workload list --cluster mlperf-v5p-256-2 --project cloud-tpu-multipod-dev  --zone europe-west4-b

# xpk workload delete --workload qwen3-5-unscanned-sft --cluster mlperf-v5p --zone europe-west4-b
# xpk workload delete --workload qwen3-5-decode1 --cluster mlperf-v5p --zone europe-west4-b