#!/bin/bash
set -ex

NUM_TOKEN_THRESHOLD=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}
# Run name is optional 4th input - our daily XLML tests will use one.


if [ -z ${4} ]
then
    RUN_NAME=${USER}_$(date +%Y-%m-%d-%H-%M-%S)
else
    RUN_NAME=${4}_$(date +%Y-%m-%d-%H)
fi

if [ -z ${5} ]
then
    ICI_TENSOR_PARALLELISM=4
else
    ICI_TENSOR_PARALLELISM=${5}
fi

# Decode without checkpoint
python3 MaxText/decode.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=50 enable_checkpointing=False metrics_file=/tmp/${RUN_NAME}_metrics.txt \
    base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH \
    attention=dot_product ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM}


# Get latest converted Gemma 2B checkpoint from internal GCS bucket
export GEMMA_2B_CKPT_PATH=$(gsutil ls gs://maxtext-gemma/2b | sort -r | head -1)
# Decode with different sampling strategies. 
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product decode_sampling_strategy=weighted decode_sampling_temperature=.00001 prompt="I love to" autoregressive_decode_assert=" cook and bake. I love to eat"
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product decode_sampling_strategy=nucleus decode_sampling_nucleus_p=0 prompt="I love to" autoregressive_decode_assert=" cook and bake. I love to eat"
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${GEMMA_2B_CKPT_PATH}/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product decode_sampling_strategy=topk decode_sampling_top_k=1 prompt="I love to" autoregressive_decode_assert=" cook and bake. I love to eat"
