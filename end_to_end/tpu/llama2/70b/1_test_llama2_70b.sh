#!/bin/bash

# This file, combined with step 2 in the same directory, demonstrates converting a Llama2-70B checkpoint from Meta and running various MaxText operations on it.
# This step is tested nightly on an ordinary CPU VM.

# The flow of this file is as follows:
# 1. Pull the checkpoint from a GCS bucket and uploads the new MaxText compatible checkpoint to destination GCS bucket.
# 2. Convert the scanned checkpoint from step 1 into unscanned checkpoint format and run more efficient decoding.

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/llama2/70b/1_test_llama2_70b.sh
# Use the same BASE_OUTPUT_PATH as end_to_end/tpu/llama2/70b/2_test_llama2_70b.sh.
# Please note that in these two scripts (1_test_llama2_70b.sh and 2_test_llama2_70b.sh) BASE_OUTPUT_PATH is assumed to be already a unique path across multiple runs and 
# the subfolders names aka RUN_NAMEs are static. Please remember to change BASE_OUTPUT_PATH across different runs.

set -ex
RUN_ID=$(date +%Y-%m-%d-%H-%M)

export MODEL='llama2-70b'
export ASYNC_CHECKPOINTING=false
export CKPT_BUCKET=gs://maxtext-model-checkpoints
# `SCANNED_CHECKPOINT` is the path to the GCS bucket where we want to save our converted (Orbax) checkpoint. Non-Googlers please remember to point `SCANNED_CHECKPOINT` to a GCS bucket that you own
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

# We install torch CPU because the checkpoint conversion script MaxText/llama_or_mistral_ckpt.py does not need a TPU/GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# We define a var for the path to the Meta checkpoint. Non-Googlers please remember to update the source `META_CHECKPOINT_PATH` to the GCS bucket where you have your Meta checkpoint
export META_CHECKPOINT_PATH=gs://maxtext-llama/llama2-70b/meta-ckpt

# In the following command, we are copying Meta's checkpoint into a local directory `tmp`. 
# You can use a different local directory than /tmp/, if you do so, please use the same local path for `base-model-path` when running `python3 MaxText/llama_or_mistral_ckpt.py`
gcloud storage cp -r ${META_CHECKPOINT_PATH} /tmp/

# Next, run the conversion script `MaxText/llama_or_mistral_ckpt.py` to convert Meta's PyTorch checkpoint in `base-model-path` and save the new converted (Orbax) checkpoint in the `maxtext-model-path`
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/meta-ckpt --maxtext-model-path ${SCANNED_CHECKPOINT} --model-size ${MODEL}

# We define `SCANNED_CHECKPOINT` to refer to the checkpoint subdirectory exactly inside `SCANNED_CHECKPOINT`. This way it is easier to use this path in future commands
export SCANNED_CHECKPOINT=${SCANNED_CHECKPOINT}/0/items

# Note that the `CONVERTED_CHECKPOINT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
# We can do this by running `MaxText/generate_param_only_checkpoint.py` on `CONVERTED_CHECKPOINT` with `force_unroll=true`. 
JAX_PLATFORMS=cpu python3 MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml base_output_directory=${UNSCANNED_CHECKPOINT} load_parameters_path=${SCANNED_CHECKPOINT} async_checkpointing=${ASYNC_CHECKPOINTING} run_name=unscanned model_name=${MODEL} force_unroll=true

# Converting MaxText orbax checkpoint to HF
# TODO(b/391634569): OOMs
# JAX_PLATFORMS=cpu python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs load_parameters_path=${SCANNED_CHECKPOINT} async_checkpointing=${ASYNC_CHECKPOINTING} run_name=convert_to_hf model_name=${MODEL} hf_model_path=/tmp/hf_llama2

# gcloud storage cp -r /tmp/hf_llama2 ${HF_CHECKPOINT}

echo "All Checkpoints saved with RUN_ID=${RUN_ID}"
