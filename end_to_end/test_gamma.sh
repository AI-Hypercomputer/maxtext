#!/bin/bash
set -e
idx=$(date +%Y-%m-%d-%H-%M)
# convert 2.5B checkpoint
export base_model_path=gs://maxtext-gamma/gamma/g_mini~2b_pt_final_orbax~0/
export maxtext_model_path=gs://maxtext-gamma/2b/${idx}
python MaxText/convert_gamma_chkpt.py --base_model_path ${base_model_path} --maxtext_model_path ${maxtext_model_path} --model_size 2b
# Test Gamma 2.5B decode
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=gs://maxtext-gamma/gamma/tokenizer.gamma load_parameters_path=${maxtext_model_path}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-2b attention=dot_product prompt="I love to" autoregressive_decode_assert=" travel and I love to write about it"
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=gs://maxtext-gamma/gamma/tokenizer.gamma load_parameters_path=${maxtext_model_path}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-2b attention=dot_product prompt="I love to" autoregressive_decode_assert=" travel and I love to write about it" decode_sampling_strategy=weighted decode_sampling_temperature=.00001
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=gs://maxtext-gamma/gamma/tokenizer.gamma load_parameters_path=${maxtext_model_path}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-2b attention=dot_product prompt="I love to" autoregressive_decode_assert=" travel and I love to write about it" decode_sampling_strategy=nucleus decode_sampling_nucleus_p=0
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=gs://maxtext-gamma/gamma/tokenizer.gamma load_parameters_path=${maxtext_model_path}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-2b attention=dot_product prompt="I love to" autoregressive_decode_assert=" travel and I love to write about it" decode_sampling_strategy=topk decode_sampling_top_k=1

# convert 7B checkpoint
export base_model_path=gs://maxtext-gamma/gamma/g_mini~7b_pt_final_orbax~0/
export maxtext_model_path=gs://maxtext-gamma/7b/${idx}
python MaxText/convert_gamma_chkpt.py --base_model_path ${base_model_path} --maxtext_model_path ${maxtext_model_path}  --model_size 7b
# Test Gamma 7B decode
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=gs://maxtext-gamma/gamma/tokenizer.gamma load_parameters_path=${maxtext_model_path}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=15 dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false scan_layers=false model_name=gamma-7b attention=dot_product prompt="I love to" autoregressive_decode_assert=" travel and I love to eat."