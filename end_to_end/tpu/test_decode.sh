#!/bin/bash
set -ex

# Decode without checkpoint
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic async_checkpointing=false model_name=gemma-2b attention=dot_product

# Get latest converted Gemma 2B checkpoint from internal GCS bucket
export GEMMA_2B_CKPT_PATH=$(gsutil ls gs://maxtext-gemma/2b | sort -r | head -1)
# Decode with different sampling strategies.
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product decode_sampling_strategy=weighted decode_sampling_temperature=.00001 prompt="I love to" autoregressive_decode_assert=" travel and I love to write. I"
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product decode_sampling_strategy=nucleus decode_sampling_nucleus_p=0 prompt="I love to" autoregressive_decode_assert=" travel and I love to write. I"
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product decode_sampling_strategy=topk decode_sampling_top_k=1 prompt="I love to" autoregressive_decode_assert=" travel and I love to write. I"
