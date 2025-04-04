#!/bin/bash
# GPT-3 175B Model.
# This script abstracts away all the shared config between Google's various MLPerf 4.0 submissions, which 
# achieved high performance on the Training benchmark. See the various different TPU type configs in this 
# directory to actually run the training.

# Example to invoke this script for compilation
# ./MaxText/configs/v5p/gpt3_175b/gpt3_175b_base.sh 4 "full" 1 64 8 "some_run" "gs://some_bucket" "train_compile.py" "v5p-1024" 1

# Example to invoke this script for training
# ./MaxText/configs/v5p/gpt3_175b/gpt3_175b_base.sh 4 "full" 1 64 8 "some_run" "gs://some_bucket"

set -euox pipefail

bash preflight.sh PLATFORM=gke

# flags set as default

# hlo dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file"

# debug
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0

export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_enable_experimental_fusion_cost_model=false --xla_tpu_dot_dot_fusion_duplicated=false --xla_tpu_dot_dot_fusion=false --xla_jf_conv_input_fusion=true --xla_jf_conv_output_fusion=false --xla_tpu_rwb_fusion=false  --xla_tpu_copy_fusion_pad_unpad_ratio=300 --xla_tpu_enable_aggressive_loop_fusion_layout_opt=false --xla_tpu_enable_copy_fusion=false --xla_tpu_reduce_loop_fusion_dup_with_unfusable_user=false --xla_tpu_scavenge_vmem_for_fusions=false --xla_tpu_vector_load_fusion_window=256 --xla_tpu_vector_store_fusion_window=64 --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_megacore_fusion=true --xla_enable_async_all_gather=true --xla_enable_async_collective_permute=true --xla_always_enable_all_gather_2d_asymmetric=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_dcn_max_overlap_estimation=32"}

# Read arguments or use defaults from environment variables
PER_DEVICE_BATCH_SIZE=${1:-${PER_DEVICE_BATCH_SIZE:-4}}
REMAT_POLICY=${2:-${REMAT_POLICY:-full}}
ICI_DATA_PARALLELISM=${3:-${ICI_DATA_PARALLELISM:-1}}
ICI_FSDP_PARALLELISM=${4:-${ICI_FSDP_PARALLELISM:-64}}
ICI_TENSOR_PARALLELISM=${5:-${ICI_TENSOR_PARALLELISM:-8}}
RUNNAME=${6:-${RUNNAME:-convergence_test_0}}
BASE_OUTPUT_DIRECTORY=${7:-${BASE_OUTPUT_DIRECTORY:-gs://some-bucket}}
EXECUTABLE=${8:-train} # Or train_compile

if [[ "$EXECUTABLE" == "train_compile" ]]; then
  COMPILE_TOPOLOGY=${9}
  COMPILE_TOPOLOGY_NUM_SLICES=${10}
  
  python3 -m MaxText."$EXECUTABLE" MaxText/configs/base.yml run_name="${RUNNAME}" model_name=gpt3-175b\
    base_output_directory="${BASE_OUTPUT_DIRECTORY}"\
    enable_checkpointing=false async_checkpointing=false\
    steps=20\
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE}"\
    ici_data_parallelism="${ICI_DATA_PARALLELISM}" ici_fsdp_parallelism="${ICI_FSDP_PARALLELISM}" ici_tensor_parallelism="${ICI_TENSOR_PARALLELISM}"\
    remat_policy="${REMAT_POLICY}"\
    attention="flash" \
    quantization=int8\
    dataset_type=synthetic\
    compile_topology="${COMPILE_TOPOLOGY}"\
    compile_topology_num_slices="${COMPILE_TOPOLOGY_NUM_SLICES}"
else
  python3 -m MaxText."$EXECUTABLE" MaxText/configs/base.yml run_name="${RUNNAME}" model_name=gpt3-175b\
    base_output_directory="${BASE_OUTPUT_DIRECTORY}"\
    enable_checkpointing=false async_checkpointing=false\
    steps=20\
    per_device_batch_size="${PER_DEVICE_BATCH_SIZE}"\
    ici_data_parallelism="${ICI_DATA_PARALLELISM}" ici_fsdp_parallelism="${ICI_FSDP_PARALLELISM}" ici_tensor_parallelism="${ICI_TENSOR_PARALLELISM}"\
    remat_policy="${REMAT_POLICY}"\
    attention="flash" \
    quantization=int8\
    dataset_type=synthetic
fi
