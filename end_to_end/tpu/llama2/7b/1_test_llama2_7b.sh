#!/bin/bash

# This file, combined with step 2 in the same directory, demonstrates converting a Llama2-7B checkpoint from Meta and running various MaxText operations on it.
# This step is tested nightly on an ordinary CPU VM.

# The flow of this file is as follows:
# 1. Pull the checkpoint from a GCS bucket and uploads the new MaxText compatible checkpoint to destination GCS bucket.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.

# Example Usage: bash end_to_end/tpu/llama2/70b/1_test_llama2_7b.sh
# Use the RUN_ID from this script to use the checkpoint in end_to_end/tpu/llama2/70b/2_test_llama2_7b.sh.
# Please note that the script 2_test_llama2_7b.sh assumes you have run this script and created MaxText compatible checkpoints.


set -ex
RUN_ID=$(date +%Y-%m-%d-%H-%M)

export MODEL='llama2-7b'
export ASYNC_CHECKPOINTING=false
export CKPT_BUCKET=gs://maxtext-model-checkpoints
# `SCANNED_CHECKPOINT` is the path to the GCS bucket where we want to save our converted (Orbax) checkpoint. Non-Googlers please remember to point `SCANNED_CHECKPOINT` to a GCS bucket that you own
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

pip install torch --index-url https://download.pytorch.org/whl/cpu

# We define a var for the path to the Meta checkpoint. Non-Googlers please remember to update the source `META_CHECKPOINT_PATH` to the GCS bucket where you have your Meta checkpoint
export META_CHECKPOINT_PATH=gs://maxtext-llama/llama2-7b/meta-ckpt

# In the following command, we are copying Meta's checkpoint into a local directory `tmp`. 
# You can use a different local directory than /tmp/, if you do so, please use the same local path for `base-model-path` when running `python3 MaxText/llama_or_mistral_ckpt.py`
gcloud storage cp -r ${META_CHECKPOINT_PATH} /tmp/

# Next, run the conversion script `MaxText/llama_or_mistral_ckpt.py` to convert Meta's PyTorch checkpoint in `base-model-path` and save the new converted (Orbax) checkpoint in the `maxtext-model-path`
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/meta-ckpt --model-size llama2-7b --maxtext-model-path ${SCANNED_CHECKPOINT}

# We define `SCANNED_CHECKPOINT` to refer to the checkpoint subdirectory exactly inside `SCANNED_CHECKPOINT`. This way it is easier to use this path in future commands
export SCANNED_CHECKPOINT=${SCANNED_CHECKPOINT}/0/items

# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `CONVERTED_CHECKPOINT` with `force_unroll=true`. 
JAX_PLATFORMS=cpu python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${UNSCANNED_CHECKPOINT} load_parameters_path=${SCANNED_CHECKPOINT} async_checkpointing=${ASYNC_CHECKPOINTING} run_name=unscanned model_name=${MODEL} force_unroll=true

# Converting MaxText orbax checkpoint to HF
JAX_PLATFORMS=cpu python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs load_parameters_path=${SCANNED_CHECKPOINT} async_checkpointing=${ASYNC_CHECKPOINTING} run_name=convert_to_hf model_name=${MODEL} hf_model_path=/tmp/hf_llama2

gcloud storage cp -r /tmp/hf_llama2 ${HF_CHECKPOINT}

echo "All Checkpoints saved with RUN_ID=${RUN_ID}"