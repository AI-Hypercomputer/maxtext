# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Running 52b.sh"
# 52B parameter model.
# This config will work out of the box for any number of v4-384 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v4/52b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"
#
# Example to AOT compile:
# bash MaxText/configs/v4/52b.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v4-384 M_COMPILE_TOPOLOGY_NUM_SLICES=2


# Stop execution if any command exits with error
set -e

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

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# Train
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
python3 MaxText/$EXECUTABLE MaxText/configs/base.yml\
    enable_profiler=true enable_checkpointing=false steps=10\
    ici_fsdp_parallelism=192 ici_tensor_parallelism=1 per_device_batch_size=7 remat_policy=full\
    base_num_decoder_layers=32 base_emb_dim=12288 base_mlp_dim=49152 base_num_query_heads=32 base_num_kv_heads=32 learning_rate=1e-8\
    base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH
