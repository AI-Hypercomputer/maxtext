#!/bin/bash
echo "Running llama2-7b model"

# Stop execution if any command exits with error
set -e

# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NVTE_FUSED_ATTN=1

RUN_NAME=$1


# everything true
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false
                --xla_gpu_simplify_all_fp_conversions --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_reduce_combine_threshold_bytes=8589934592 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
                --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true
                --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_disable_hlo_passes=rematerialization"


echo $XLA_FLAGS

# this is for llama-2-7b
#RUN_SETTINGS="flash_maxtext/MaxText/train.py flash_maxtext/MaxText/configs/base.yml dataset_path=gs://maxtext-dataset\
#     load_parameters_path=gs://maxtext-llama/test/yooh-2024-0123-1440/decode-ckpt-maxtext/0/default\
#     run_name=runner_$(date +%Y-%m-%d-%H-%M) model_name=llama2-7b tokenizer_path=gs://maxtext-llama/llama2-7b/tokenizer.llama2\
#     attention=cudnn_flash_te async_checkpointing=False base_output_directory=gs://ninacai-sandbox enable_profiler=True steps=30"

RUN_SETTINGS="flash_maxtext/MaxText/decode.py flash_maxtext/MaxText/configs/base.yml load_parameters_path=gs://maxtext-llama/test/2024-01-15-06-49/decode-ckpt-maxtext/0/default run_name=runner_2024-02-09-20-02  per_device_batch_size=1 model_name=llama2-7b tokenizer_path=gs://maxtext-llama/llama2-7b/tokenizer.llama2 max_prefill_predict_length=64  max_target_length=128  attention=dot_product dataset_path=gs://maxtext-dataset steps=10 async_checkpointing=false ici_autoregressive_parallelism=1"
#RUN_SETTINGS="flash_maxtext/MaxText/decode.py flash_maxtext/MaxText/configs/base.yml run_name=runner_$(date +%Y-%m-%d-%H-%M) base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset steps=2 ici_tensor_parallelism=4 attention=dot_product enable_checkpointing=false max_target_length=128 per_device_batch_size=1"

echo "Command: nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 $RUN_SETTINGS"
#nsys profile -s none -o nsys_profile.out --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python3 $RUN_SETTINGS
python3 $RUN_SETTINGS

set +e

