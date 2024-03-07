#!/bin/bash
set -e
set -x
idx=$(date +%Y-%m-%d-%H-%M)
# convert 2.5B checkpoint
export base_model_path=gs://maxtext-gemma/flax/2b
export maxtext_model_path=gs://maxtext-gemma/2b/${idx}
python MaxText/convert_gemma_chkpt.py --base_model_path ${base_model_path} --maxtext_model_path ${maxtext_model_path} --model_size 2b
# Test Gemma 2.5B decode
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.gemma load_parameters_path=${maxtext_model_path}/0/default per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=gemma-2b attention=dot_product prompt="I love to" autoregressive_decode_assert=" travel and I love to write. I"
