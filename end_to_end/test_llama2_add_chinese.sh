#!/bin/bash
set -xe

# export JAX_TRACEBACK_FILTERING=off

idx=$(date +%Y-%m-%d-%H-%M)

base_ckpt_path=/home/genggui001/gdrive/genggui001/pretrain_weights/nlp/llama2-7b-maxtext-add-chinese-other/0/default
dataset_path=/home/genggui001/gdrive/gg-nlp-lm-new

#TODO(Training with Llama is not complete)
python3 -u MaxText/train.py MaxText/configs/base.yml \
 load_parameters_path=${base_ckpt_path} \
 run_name=runner_${idx} \
 model_name='llama2-7b-add-chinese' \
 ici_tensor_parallelism=4 \
 steps=10000 \
 warmup_steps_fraction=0.0001 \
 eval_interval=512 \
 checkpoint_period=512 \
 max_target_length=2048 \
 per_device_batch_size=0.125 \
 gradient_accumulation_steps=8 \
 base_output_directory=/home/genggui001/code/maxtext/tmp/llama2-7b-add-chinese  \
 dataset_path=${dataset_path} \
 attention=dot_product

