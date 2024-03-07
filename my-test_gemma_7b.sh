set -e
set -x
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=/home/rwitten/gemma_7b/0/default per_device_batch_size=1 run_name=runner_2024-03-07-17-57 max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-7b attention=dot_product 'prompt=I love to' 'autoregressive_decode_assert= use this product in my hair. It'
