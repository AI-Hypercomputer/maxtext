#!/bin/bash

set -xe

RUN_NAME=dpo_$(date +%Y-%m-%d-%H-%M-%S)

# get latest converted Gemma2 2B checkpoint from internal GCS bucket
export GEMMA_2B_CKPT_PATH=$(gsutil ls gs://maxtext-gemma/gemma2/2b | sort -r | head -1)

# tfds pipeline
python MaxText/train.py MaxText/configs/dpo.yml tokenizer_path=assets/tokenizer.gemma \
    per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) \
    max_prefill_predict_length=512 max_target_length=512 \
    steps=10 model_name=gemma2-2b base_output_directory=gs://rdyro/logs \
    load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items enable_checkpointing=true

# grain pipeline
mkdir -p /tmp/anthropic_rlhf
gcloud storage cp -r gs://maxtext-dataset/dpo/anthropic_rlhf/array_record /tmp/anthropic_rlhf
python MaxText/train.py MaxText/configs/dpo.yml tokenizer_path=assets/tokenizer.gemma \
    per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) \
    max_prefill_predict_length=512 max_target_length=512 \
    steps=10 model_name=gemma2-2b base_output_directory=gs://rdyro/logs \
    load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items enable_checkpointing=true \
    dataset_type=grain grain_worker_count=16 \
    grain_train_files='/tmp/anthropic_rlhf/array_record/anthropic_rlhf_tfds-train.array_record*' \
    grain_eval_files='/tmp/anthropic_rlhf/array_record/anthropic_rlhf_tfds-test.array_record*'
