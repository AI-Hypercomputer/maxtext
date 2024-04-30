#!/bin/bash

# This script is designed for internal use within Google. External users can adapt it by:
#  - Updating GCS paths (gs://) to your accessible locations.
#  - Using the checkpoint generated from train.py or available one in open source (i.e. https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar).

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

# Download checkpoint, convert it to MaxText, and run inference
pip3 install torch
gsutil -m cp -r gs://maxtext-external/mistral-7B-v0.1 /tmp
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/mistral-7B-v0.1 --model-size mistral-7b --maxtext-model-path gs://maxtext-mistral/test/${idx}/decode-ckpt-maxtext/
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=gs://maxtext-mistral/test/${idx}/decode-ckpt-maxtext/0/items run_name=runner_direct_${idx} per_device_batch_size=1 model_name='mistral-7b' tokenizer_path=gs://maxtext-external/mistral-7B-v0.1/tokenizer.mistral ici_tensor_parallelism=4 max_prefill_predict_length=4 max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to read about the Bible. I love" attention=dot_product

# Training
python3 MaxText/train.py MaxText/configs/base.yml load_parameters_path=gs://maxtext-mistral/test/${idx}/decode-ckpt-maxtext/0/items run_name=runner_${idx} per_device_batch_size=1 model_name='mistral-7b' ici_tensor_parallelism=4 steps=10 max_target_length=1024 tokenizer_path=gs://maxtext-external/mistral-7B-v0.1/tokenizer.mistral
