#!/bin/bash


export PAX_DATE=$1
export exp_name=$2
export BUCKET_NAME=$3

job_dir="gs://${BUCKET_NAME}/GPT3/runs/${exp_name}/"
# install praxis and pax
max_iteration=5
for i in $(seq 1 $max_iteration)
do
  pip install /tmp/mlperf_test_script/praxis-nightly+"${PAX_DATE}"-py3-none-any.whl --no-cache-dir
  result=$?
  if [[ $result -eq 0 ]]
  then
    echo "praxis install successful"
    break
  else
    echo "praxis unsuccessful, retry"
    sleep 1
  fi
done

for i in $(seq 1 $max_iteration)
do
  pip install /tmp/mlperf_test_script/paxml-nightly+"${PAX_DATE}"-py3-none-any.whl --no-cache-dir
  result=$?
  if [[ $result -eq 0 ]]
  then
    echo "paxml install successful"
    break
  else
    echo "paxml unsuccessful, retry"
    sleep 1
  fi
done

# copy libtpu
# change to your own libptu path if you need, please don't overwirte this one.

# [example] How to build libtpu?
# blaze build -c opt --config=gce --copt=-mno-avx --copt=-mno-avx2 --copt=-mfma --copt=-DPLATFORM_CLOUD_TPU --copt='-DSTRIP_FLAG_HELP=1' --define=with_tpu_support=true --define=tensorflow_mkldnn_contraction_kernel=0 --//learning/brain/tfrc/executor:max_gce_tpu_version=viperlite //learning/brain/tfrc/executor:_libtpu.so
# [example] How to upload to gcs
# GCS_PATH=wenqicao-prod-multipods
# gsutil cp blaze-bin/learning/brain/tfrc/executor/_libtpu.so gs://${GCS_PATH}/

# gsutil cp gs://wenqicao-prod-multipods/_libtpu.so /tmp/libtpu.so

# # gke download exp config
gsutil cp gs://"${BUCKET_NAME}"/mlperf_test_script/src/*.py /usr/local/lib/python3.10/site-packages/paxml/tasks/lm/params/

# gce download exp config
# gsutil cp gs://"${BUCKET_NAME}"/mlperf_test_script/src/*.py $HOME/.local/lib/python3.10/site-packages/paxml/tasks/lm/params/

#gsutil cp gs://"${BUCKET_NAME}"/mlperf_test_script/patch_src/opt/*.py ./venv/lib/python3.10/site-packages/paxml/
#gsutil cp gs://${BUCKET_NAME}/mlperf_test_script/patch_src/profile/*.py ./venv/lib/python3.10/site-packages/paxml/


# gke experiment command
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto" TF_CPP_MIN_LOG_LEVEL=0 \
LIBTPU_INIT_ARGS="--xla_tpu_enable_megascale_barrier=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_enable_async_collective_permute=true --xla_jf_rematerialization_percent_shared_memory_limit=97 --xla_tpu_decompose_all_gather_einsum=false --xla_tpu_spmd_threshold_for_allgather_cse=10 --xla_tpu_prefuse_self_attention=false --xla_tpu_rwb_fusion=false --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_dcn_max_overlap_estimation=32.0 --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio=10 --megascale_enable_async_host_commands=true --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_enable_flash_attention=true --xla_tpu_scavenge_vmem_for_fusions=true" TPU_PREMAPPED_BUFFER_SIZE=4294967296 TPU_NAME=local TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 JAX_USE_PJRT_C_API_ON_TPU=1 \
    python3 /usr/local/lib/python3.10/site-packages/paxml/main.py \
    --exp=tasks.lm.params.c4_mlperf_test."${exp_name}" \
    --job_log_dir="${job_dir}" \
    --jax_profiler_port=9999 \
    --enable_checkpoint_saving=false \
    --mode=train --eval_on_test 2>&1 | tee /tmp/large_scale_multislice_test_log

# # gce experiment command
# XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto" TF_CPP_MIN_LOG_LEVEL=0 LIBTPU_INIT_ARGS="--xla_tpu_enable_megascale_barrier=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_enable_async_collective_permute=true --xla_jf_rematerialization_percent_shared_memory_limit=97 --xla_tpu_decompose_all_gather_einsum=false --xla_tpu_spmd_threshold_for_allgather_cse=10 --xla_tpu_prefuse_self_attention=false --xla_tpu_rwb_fusion=false --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_dcn_max_overlap_estimation=32.0 --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio=10 --megascale_enable_async_host_commands=true --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_enable_flash_attention=true --xla_tpu_scavenge_vmem_for_fusions=true" TPU_PREMAPPED_BUFFER_SIZE=4294967296 TPU_NAME=local TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 JAX_USE_PJRT_C_API_ON_TPU=1 \
# multiprocess_gpu=True num_hosts=2 \
# python3 $HOME/.local/lib/python3.10/site-packages/paxml/main.py \
#     --exp=tasks.lm.params.c4_mlperf_test."${exp_name}" \
#     --job_log_dir="${job_dir}" \
#     --jax_profiler_port=9999 \
#     --enable_checkpoint_saving=false \
#     --mode=train --eval_on_test 2>&1 | sudo tee /tmp/large_scale_multislice_test_log

# source /tmp/mlperf_test_script/parser_metrics.sh


# /tmp/2023-08-12-05-29-59/
