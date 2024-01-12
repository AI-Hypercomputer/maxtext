set -e
idx=$(date +%Y-%m-%d-%H-%M)
pip install torch
gsutil cp -r gs://maxtext-llama/llama2-7b/meta-ckpt /tmp/
python3 MaxText/convert_llama_ckpt.py --base-model-path /tmp/meta-ckpt --model-size 7b --maxtext-model-path gs://maxtext-llama/test/${idx}/decode-ckpt-maxtext/
python3 MaxText/decode.py MaxText/configs/base.yml run_name=runner_${idx} base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset load_from_other_directory=gs://maxtext-llama/test/${idx}/decode-ckpt-maxtext/ load_from_other_directory_step=0 per_device_batch_size=1 model_name='llama2-7b' assets_path=gs://maxtext-llama/llama2-7b ici_tensor_parallelism=4 steps=1 max_prefill_predict_length=4  max_target_length=16 async_checkpointing=false prompt="I love to" autoregressive_decode_assert="read. I love to write. I love to share." attention=dot_product
