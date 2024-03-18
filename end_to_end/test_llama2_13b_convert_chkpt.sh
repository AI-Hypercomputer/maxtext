#!/bin/bash

# This script is designed for internal use within Google. External users can adapt it by:
#  - Updating GCS paths (gs://) to your accessible locations.
#  - Using the checkpoint generated from train.py or available one in open source (https://llama.meta.com/llama-downloads/).

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

# Step 1: Download Llama2's pytorch checkpoint
pip install torch --index-url https://download.pytorch.org/whl/cpu
gcloud storage cp -r gs://maxtext-llama/llama2-13b/meta-ckpt /tmp/

# Step 2: Convert Llama2's pytorch checkpoint into MaxText parameter checkpoint
export converted_checkpoint_path=gs://maxtext-llama/test/${idx}/decode-ckpt-maxtext
export converted_checkpoint=${converted_checkpoint_path}/0/items
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/meta-ckpt --model-size llama2-13b --maxtext-model-path ${converted_checkpoint_path}
echo "converted_checkpoint: ${converted_checkpoint}"

# Step 3: Load converted parameter checkpoint into decode.py and verify result
unset M_LOAD_PARAMETERS_PATH
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${converted_checkpoint} run_name=runner_direct_${idx} per_device_batch_size=1 model_name='llama2-13b' ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to write. I love to teach." attention=dot_product weight_dtype=bfloat16
