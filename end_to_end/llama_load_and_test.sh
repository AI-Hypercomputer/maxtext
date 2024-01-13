set -e
idx=$(date +%Y-%m-%d-%H-%M)

export M_LOAD_PARAMETERS_PATH=gs://maxtext-llama/test/2024-01-12-17-46/decode-ckpt-maxtext/0/default
export M_ENABLE_CHECKPOINTING=true
export M_ASYNC_CHECKPOINTING=true

#TODO(internal bug -- migrate to XLML)
#pip install torch
#gsutil cp -r gs://maxtext-llama/llama2-7b/meta-ckpt /tmp/
#python3 MaxText/convert_llama_ckpt.py --base-model-path /tmp/meta-ckpt --model-size 7b --maxtext-model-path gs://maxtext-llama/test/${idx}/decode-ckpt-maxtext/

#TODO(Training with Llama is not complete)
python3 MaxText/train.py MaxText/configs/base.yml run_name=runner_${idx} base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset per_device_batch_size=1 model_name='llama2-7b' assets_path=gs://maxtext-llama/llama2-7b ici_tensor_parallelism=4 steps=10 max_target_length=1024 per_device_batch_size=1
python3 MaxText/decode.py MaxText/configs/base.yml run_name=runner_${idx} base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset per_device_batch_size=1 model_name='llama2-7b' assets_path=gs://maxtext-llama/llama2-7b ici_tensor_parallelism=4 steps=1 max_prefill_predict_length=4  max_target_length=16 prompt="I love to" autoregressive_decode_assert="read. I love to write. I love to share." attention=dot_product