set -e
idx=$(date +%Y-%m-%d-%H-%M)

export M_ENABLE_CHECKPOINTING=true
export M_BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export M_DATASET_PATH=gs://maxtext-dataset
export M_ASYNC_CHECKPOINTING=false

# Download checkpoint, convert it to MaxText, and run inference
pip3 install torch
gsutil -m cp -r gs://maxtext-external/mistral-7B-v0.1 /tmp
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path /tmp/mistral-7B-v0.1 --model-size mistral-7b --maxtext-model-path gs://maxtext-mistral/test/${idx}/decode-ckpt-maxtext/
python3 MaxText/decode.py MaxText/configs/base.yml load_parameters_path=gs://maxtext-mistral/test/${idx}/decode-ckpt-maxtext/0/default run_name=runner_direct_${idx} per_device_batch_size=1 model_name='mistral-7b' assets_path=gs://maxtext-external/mistral-7B-v0.1 ici_tensor_parallelism=4 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to read about the Bible. I love" attention=dot_product

# TODO(ranran): add training and fine-tuning tests
