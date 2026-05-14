#!/bin/bash

set -xe

RUN_NAME=dpo_$(date +%Y-%m-%d-%H-%M-%S)

# get latest converted Gemma2 2B checkpoint from internal GCS bucket
export GEMMA_2B_CKPT_PATH=$(gcloud storage ls gs://maxtext-gemma/gemma2/2b | sort -r | head -1)
LOGS="gs://maxtext-external/logs"

# hf pipeline
python3 -m maxtext.trainers.post_train.dpo.train_dpo "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/post_train/dpo.yml tokenizer_path='google/gemma-2-2b-it' \
    run_name="$RUN_NAME-hf" model_name=gemma2-2b base_output_directory=${LOGS} \
    load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items \
    dataset_type=hf hf_access_token=$HF_TOKEN hf_path='Anthropic/hh-rlhf' \
    train_data_columns="['chosen', 'rejected']" \
    per_device_batch_size=0.5 allow_split_physical_axes=True eval_interval=0 \
    ici_data_parallelism=1 ici_tensor_parallelism=1 ici_fsdp_parallelism=1 \
    use_grpo=False steps=2

