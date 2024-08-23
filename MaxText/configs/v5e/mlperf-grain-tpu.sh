echo "Running mlperf-grain-tpu.sh"
# 16B parameter model.
# This config will work out of the box for any number of v5e-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5e/16b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"
#
# Example to AOT compile:
# bash MaxText/configs/v5e/16b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v5e-256 M_COMPILE_TOPOLOGY_NUM_SLICES=2


# Stop execution if any command exits with error
set -e -x

export PLATFORM="gce"
export EXECUTABLE="train.py" # or train_compile.py

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# The setup accommodates two cases:
# 1) Passing the 'RUN_NAME' variable at runtime
# 2) Propagating the 'M_RUN_NAME' variable within an Airflow sweeping workflow
if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

if [[ -z "${PER_DEVICE_BATCH_SIZE}" ]]; then
    export PER_DEVICE_BATCH_SIZE=1
    echo "PER_DEVICE_BATCH_SIZE is not set, setting to 1"
else
    echo "PER_DEVICE_BATCH_SIZE is set to ${PER_DEVICE_BATCH_SIZE}"
fi

if [[ -z "${GRAIN_WORKER_COUNT}" ]]; then
    export GRAIN_WORKER_COUNT=1
    echo "GRAIN_WORKER_COUNT is not set, setting to 1"
else
    echo "GRAIN_WORKER_COUNT is set to ${GRAIN_WORKER_COUNT}"
fi

if [[ -z "${STEPS}" ]]; then
    export STEPS=10
    echo "STEPS is not set, setting to 10"
else
    echo "STEPS is set to ${STEPS}"
fi

if [[ -z "${ICI_DATA}" ]]; then
    export ICI_DATA=1
    echo "ICI_DATA is not set, setting to 1"
else
    echo "ICI_DATA is set to ${ICI_DATA}"
fi

if [[ -z "${ICI_FSDP}" ]]; then
    export ICI_FSDP=-1
    echo "ICI_FSDP is not set, setting to -1"
else
    echo "ICI_FSDP is set to ${ICI_FSDP}"
fi

if [[ -z "${ICI_TENSOR}" ]]; then
    export ICI_TENSOR=1
    echo "ICI_TENSOR is not set, setting to 1"
else
    echo "ICI_TENSOR is set to ${ICI_TENSOR}"
fi

if [[ -z "${DCN_DATA}" ]]; then
    export DCN_DATA=-1
    echo "DCN_DATA is not set, setting to -1"
else
    echo "DCN_DATA is set to ${DCN_DATA}"
fi

if [[ -z "${DCN_FSDP}" ]]; then
    export DCN_FSDP=1
    echo "DCN_FSDP is not set, setting to 1"
else
    echo "DCN_FSDP is set to ${DCN_FSDP}"
fi

if [[ -z "${DCN_TENSOR}" ]]; then
    export DCN_TENSOR=1
    echo "DCN_TENSOR is not set, setting to 1"
else
    echo "DCN_TENSOR is set to ${DCN_TENSOR}"
fi

if [[ -z "${NUM_SLICES}" ]]; then
    export NUM_SLICES=-1
    echo "NUM_SLICES is not set, setting to -1"
else
    echo "NUM_SLICES is set to ${NUM_SLICES}"
fi

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# using gcsfuse
echo "Mounting bucket to /tmp/gcsfuse/"
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=mlperf-exp-us-east1-cp0 MOUNT_PATH=/tmp/gcsfuse
DATASET_PATH=/tmp/gcsfuse

# download tokenizer
gsutil cp gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model assets/

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

JAX_PLATFORMS=tpu python3 MaxText/$EXECUTABLE MaxText/configs/base.yml \
    steps=$STEPS \
    model_name=gpt3-175b \
    per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
    enable_checkpointing=False \
    base_output_directory=gs://aireenmei-multipod/cpu \
    grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.4/c4-train2*.array_record \
    grain_eval_files=/tmp/gcsfuse/array-record/c4/en/3.0.5/c4-validation*.array_record \
    add_bos=False \
    add_eos=False \
    tokenize_eval_data=False \
    eval_data_column='ids' \
    eval_interval=5 \
    dataset_type=grain \
    ici_data_parallelism=$ICI_DATA \
    ici_fsdp_parallelism=$ICI_FSDP \
    ici_tensor_parallelism=$ICI_TENSOR \
    dcn_data_parallelism=$DCN_DATA \
    dcn_fsdp_parallelism=$DCN_FSDP \
    dcn_tensor_parallelism=$DCN_TENSOR \
    num_slices=$NUM_SLICES \
    tokenizer_path=assets/c4_en_301_5Mexp2_spm.model \
    remat_policy=full \
    attention=flash \
    data_shuffle_seed=8745 \
    quantization=int8 \
    grain_worker_count=$GRAIN_WORKER_COUNT
    #run_name=$(date +%m%d-%H%M)
    #model_name=gpt3-175b \
    
    # grain_eval_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation* \
    # grain_eval_files=/tmp/gcsfuse/maxtext-dataset/array-record/c4/en/3.0.5/*.array_record
    # eval_per_device_batch_size=4 eval_interval=10 eval_steps=5 \
    # tokenize_eval_data=False eval_data_column='ids'\

#gsutil cp $HOME/gcsfuse.json ${OUTPUT_PATH}/${M_RUN_NAME}/
