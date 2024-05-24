#!/bin/bash

# This file, combined with step 1 in the same directory, runs on daily basis and demonstrates:
# 1. Converts the Mistral PyTorch checkpoint to MaxText(orbax) format using a CPU VM.
# 2. Takes the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM.

# The flow of this file is to take the MaxText(orbax) checkpoint to run inference, fine-tuning, and pre-training on a TPU VM. 
# Please make sure you have run end_to_end/tpu/mixtral/8x7b/1_test_mixtral.sh before running commands from this file. 

# Example Usage: export BASE_OUTPUT_PATH=/path/to/GCS/bucket; bash end_to_end/tpu/mixtral/8x7b/2_test_mixtral.sh
# Use the same BASE_OUTPUT_PATH for both 1_test_mixtral.sh & 2_test_mixtral.sh.

set -x
MODEL_VARIATION='8x7b'

if [ -z "${BASE_OUTPUT_PATH}" ]; then
    # Non-Googlers please remember to point BASE_OUTPUT_PATH to GCS buckets that you own, this script uses internal buckets for testing.
    export BASE_OUTPUT_PATH=gs://runner-maxtext-logs/$(date +%Y-%m-%d-%H-%M)
    echo "BASE_OUTPUT_PATH is not set, using BASE_OUTPUT_PATH = ${BASE_OUTPUT_PATH}"
fi

export DATASET_PATH=gs://maxtext-dataset

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py` 
export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}${MODEL_VARIATION}/scanned_ckpt/0/items
export MATMUL_SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}${MODEL_VARIATION}/matmul_scanned_ckpt/0/items

mkdir /tmp/HLO_dumps

# export XLA_FLAGS="--xla_dump_to=$/tmp/HLO_dumps/ --xla_dump_hlo_pass_re=.*
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
 --xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0
 --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=1073741824 --xla_gpu_all_gather_combine_threshold_bytes=134217728
 --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
 --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false
 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_disable_hlo_passes=rematerialization"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export TF_FORCE_GPU_ALLOW_GROWTH=true

python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} \
    run_name=moe-train-forloop per_device_batch_size=0.5 model_name=mixtral-8x7b ici_fsdp_parallelism=-1 \
    ici_tensor_parallelism=1 skip_first_n_steps_for_profiler=5 steps=10 dtype=bfloat16 \
    weight_dtype=bfloat16 max_target_length=128 moe_matmul=False megablox=False \
    profiler=xplane attention=dot_product dataset_type=synthetic tokenizer_path=assets/tokenizer.mistral hardware=gpu

# python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} \
#     run_name=moe-train-matmul per_device_batch_size=1 model_name=mixtral-8x7b ici_fsdp_parallelism=-1 \
#     ici_tensor_parallelism=1 skip_first_n_steps_for_profiler=5 steps=10 dtype=bfloat16 \
#     weight_dtype=bfloat16 max_target_length=1024 moe_matmul=True megablox=False \
#     profiler=xplane attention=dot_product dataset_type=synthetic tokenizer_path=assets/tokenizer.mistral hardware=gpu

# python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${MATMUL_SCANNED_CHECKPOINT} run_name=megablox_forward_pass_test per_device_batch_size=0.5 model_name=mixtral-8x7b tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral ici_tensor_parallelism=1 ici_fsdp_parallelism=-1 max_prefill_predict_length=11 max_target_length=11 dataset_type=synthetic dtype=float32 moe_matmul=True megablox=True hardware=gpu profiler=xplane --atol=3 --rtol=1 --token_size=4

# gsutil cp -r /tmp/HLO_dumps/ $BASE_OUTPUT_PATH/

# python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} \
#     run_name=megablox_forward_pass_test per_device_batch_size=1 model_name=mixtral-8x7b \
#     tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral \
#     ici_tensor_parallelism=4 ici_fsdp_parallelism=16 \
#     max_prefill_predict_length=11 max_target_length=11 \
#     dataset_type=synthetic dtype=float32 moe_matmul=True megablox=True

# python3 MaxText/decode.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} \
#     run_name=megablox_forward_pass_test per_device_batch_size=1 model_name=mixtral-8x7b \
#     tokenizer_path=gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral \
#     ici_tensor_parallelism=4 ici_fsdp_parallelism=16 \
#     max_prefill_predict_length=11 max_target_length=11 \
#     dataset_type=synthetic dtype=float32 moe_matmul=True megablox=True

