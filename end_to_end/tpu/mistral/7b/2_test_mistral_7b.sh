
set -ex

if [ -z "${RUN_ID}" ]; then
    echo "Please set the RUN_ID used to create checkpoint from 1st script in this folder"
fi

export MODEL='mistral-7b'
export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export CKPT_BUCKET=gs://maxtext-model-checkpoints
export DATASET_PATH=gs://maxtext-dataset
export ASYNC_CHECKPOINTING=false
export UNSCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/unscanned/checkpoints/0/items
export SCANNED_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/scanned/0/items
export HF_CHECKPOINT=${CKPT_BUCKET}/${MODEL}/${RUN_ID}/huggingface

# Installing torch for deps in forward_pass_logit_chekcker.py
pip install torch --index-url https://download.pytorch.org/whl/cpu


# Run decoding with converted ckpt - matmul implementation
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=${SCANNED_CHECKPOINT} run_name=scanned_decoding per_device_batch_size=1 model_name=mistral-7b async_checkpointing=false tokenizer_path=assets/tokenizer.mistral-v1 max_prefill_predict_length=11 max_target_length=16 prompt="[INST] I love to [/INST]" attention=dot_product megablox=False sparse_matmul=False

# Test whether the forward pass logits match the golden logits - matmul implementation
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} load_parameters_path=${SCANNED_CHECKPOINT} run_name=matmul_forward_pass_test per_device_batch_size=1 model_name=mistral-7b tokenizer_path=assets/tokenizer.mistral-v1 max_prefill_predict_length=11 max_target_length=11 dataset_type=synthetic dtype=float32 megablox=False sparse_matmul=False --atol=3 --rtol=1 --token_size=4
