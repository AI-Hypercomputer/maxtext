#!/bin/bash

set -xe

RUN_NAME=dpo_$(date +%Y-%m-%d-%H-%M-%S)

# get latest converted Gemma2 2B checkpoint from internal GCS bucket
export GEMMA_2B_CKPT_PATH=$(gcloud storage ls gs://maxtext-gemma/gemma2/2b | sort -r | head -1)
LOGS="gs://maxtext-external/logs"

# tfds pipeline
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/dpo.yml tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/assets}"/tokenizer.gemma \
    run_name="$RUN_NAME-tfds" model_name=gemma2-2b base_output_directory=${LOGS} \
    load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items \
    per_device_batch_size=0.5 allow_split_physical_axes=True \
    ici_data_parallelism=2 ici_tensor_parallelism=2 ici_fsdp_parallelism=1

# grain pipeline
mkdir -p /tmp/anthropic_rlhf || true
gcloud storage cp -r gs://maxtext-dataset/dpo/anthropic_rlhf/array_record /tmp/anthropic_rlhf
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/dpo.yml tokenizer_path="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/assets}"/tokenizer.gemma \
    run_name="$RUN_NAME-grain" model_name=gemma2-2b base_output_directory=${LOGS} \
    load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items \
    dataset_type=grain grain_worker_count=16 \
    grain_train_files='/tmp/anthropic_rlhf/array_record/anthropic_rlhf_tfds-train.array_record*' \
    grain_eval_files='/tmp/anthropic_rlhf/array_record/anthropic_rlhf_tfds-test.array_record*' \
    per_device_batch_size=0.5 allow_split_physical_axes=True \
    ici_data_parallelism=2 ici_tensor_parallelism=2 ici_fsdp_parallelism=1

# hf pipeline
python3 -m MaxText.train "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/configs/dpo.yml tokenizer_path='google/gemma-2-2b-it' \
    run_name="$RUN_NAME-grain" model_name=gemma2-2b base_output_directory=${LOGS} \
    load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items \
    dataset_type=hf hf_access_token=$HF_TOKEN hf_path='Anthropic/hh-rlhf' \
    per_device_batch_size=0.5 allow_split_physical_axes=True ici_tensor_parallelism=2 \
    ici_data_parallelism=2 ici_tensor_parallelism=2 ici_fsdp_parallelism=1
