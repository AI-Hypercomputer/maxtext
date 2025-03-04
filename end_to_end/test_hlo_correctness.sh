#!/bin/bash

set -ex
pip3 install --upgrade protobuf
pip3 show protobuf

MODEL_NAME="llama3.1-8b"
export M_COMPILE_TOPOLOGY=v5e-256 
export M_COMPILE_TOPOLOGY_NUM_SLICES=1
PER_DEVICE_BATCH_SIZE="2"
DATASET_PATH=gs://maxtext-dataset
LOCAL_DIR="/tmp/xla_dump"

echo "topology: ${M_COMPILE_TOPOLOGY}"
echo "topology_num_slices: ${M_COMPILE_TOPOLOGY_NUM_SLICES}"
echo "model_name: ${MODEL_NAME}"

# Dump HLO for train_compile.py.
XLA_COMPILE=${LOCAL_DIR}/compile
mkdir -p ${XLA_COMPILE}
export XLA_FLAGS="--xla_dump_to=${XLA_COMPILE} --xla_dump_large_constants=true"
echo "XLA_FLAGS for compile: ${XLA_FLAGS}"

TEST_DIR="gs://hengtaoguo-maxtext-logs/1-correctness"

# Run train_compile to dump HLO to LOCAL_DIR
python MaxText/train_compile.py MaxText/configs/base.yml model_name=$MODEL_NAME \
    base_output_directory=$TEST_DIR run_name=compile dataset_path=$DATASET_PATH \
    tokenizer_path=assets/tokenizer_llama3.tiktoken \
    enable_checkpointing=false per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
    save_config_to_gcs=true \
    dump_hlo=true dump_hlo_gcs_dir=$TEST_DIR/compile/xla_dump

# Copy to GCS remote dir
gcloud storage cp -r ${XLA_COMPILE} ${TEST_DIR}/compile/xla_dump

# Dump HLO for train.py.
# XLA_TRAIN=${LOCAL_DIR}/train
# mkdir -p ${XLA_TRAIN}
# export XLA_FLAGS="--xla_dump_to=${XLA_TRAIN} --xla_dump_large_constants=true"
# echo "XLA_FLAGS for train: ${XLA_FLAGS}"

# # Train a workload with same config
# python MaxText/train.py MaxText/configs/base.yml model_name=${MODEL_NAME} \
#     base_output_directory=$TEST_DIR run_name=train dataset_path=$DATASET_PATH \
#     tokenizer_path=assets/tokenizer_llama3.tiktoken \
#     enable_checkpointing=false per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
#     save_config_to_gcs=true steps=2 \
#     dump_hlo=true dump_hlo_gcs_dir=$TEST_DIR/train/xla_dump

rm -r ${LOCAL_DIR}