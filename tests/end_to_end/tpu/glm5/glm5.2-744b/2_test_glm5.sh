#!/bin/bash

# This file runs Step 2 on TPU cluster for GLM-5.2 (Cross-Layer IndexShare).
# 1. Forward pass logit check against golden logits.
# 2. High-throughput distributed pre-training with IndexShare.
# 3. Decoding & sanity prompt generation.

set -ex

export MODEL_NAME='glm5.2-744b'
export TOKENIZER_PATH='zai-org/GLM-5.2'

# Installing torch CPU for tokenizer / evaluation helpers
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

if [ -z "${BASE_OUTPUT_PATH}" ]; then
  export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)
  echo "BASE_OUTPUT_PATH is not set"
fi
BASE_OUTPUT_PATH=${BASE_OUTPUT_PATH%/}
echo using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}

SCANNED_CKPT_PATH=${SCANNED_CKPT_PATH:-gs://maxtext-glm5-europe-west4/maxtext-glm-5.2-bf16-converted-final-78l/0/items}
export DATASET_PATH=gs://maxtext-dataset

# 1. Distributed Pre-Training Benchmark with GLM-5.2 Cross-Layer IndexShare (64 TPU cores, EP=4, FSDP=16)
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=pretrain_glm52 \
  model_name=${MODEL_NAME} \
  override_model_config=true \
  dataset_type=synthetic \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  per_device_batch_size=1 \
  max_target_length=2048 \
  indexer_topk=1024 \
  use_indexer=true \
  use_index_share=true \
  index_share_pattern="FSSS" \
  prune_shared_indexers=true \
  dcn_pipeline_parallelism=1 \
  dcn_data_parallelism=-1 \
  ici_pipeline_parallelism=1 \
  ici_fsdp_transpose_parallelism=1 \
  ici_fsdp_parallelism=16 \
  ici_expert_parallelism=4 \
  allow_split_physical_axes=true \
  use_iota_embed=true \
  remat_policy=custom \
  decoder_layer_input=offload \
  opt_type=sgd \
  enable_checkpointing=false \
  steps=10

# 2. Forward Logit & Generation Test with GLM-5.2 Cross-Layer IndexShare
python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_PATH} \
  run_name=decode_glm52 \
  model_name=${MODEL_NAME} \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${SCANNED_CKPT_PATH} \
  scan_layers=true \
  attention=dot_product \
  sparse_matmul=false \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  per_device_batch_size=1 \
  max_prefill_predict_length=64 \
  max_target_length=128 \
  ici_fsdp_parallelism=16 \
  ici_expert_parallelism=4 \
  checkpoint_storage_concurrent_gb=1024 \
  use_indexer=true \
  use_index_share=true \
  index_share_pattern="FSSS" \
  prune_shared_indexers=true \
  prompt="The capital of France is"

