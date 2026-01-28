#!/bin/bash

# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

AR_THRESHOLD=134217728
AG_THRESHOLD=134217728
RS_THRESHOLD=67108864

MODEL="llama2-7b"
RUN_NAME=$MODEL-$(date +%Y-%m-%d-%H-%M)
BASE_OUTPUT_DIR=/tmp/local_train

export XLA_FLAGS="--xla_dump_hlo_as_text
    --xla_dump_to=$BASE_OUTPUT_DIR/$RUN_NAME/HLO_dumps/
    --xla_gpu_enable_latency_hiding_scheduler=true
    --xla_gpu_enable_triton_gemm=false
    --xla_gpu_enable_command_buffer=''
    --xla_gpu_all_reduce_combine_threshold_bytes=${AR_THRESHOLD}
    --xla_gpu_all_gather_combine_threshold_bytes=${AG_THRESHOLD}
    --xla_gpu_reduce_scatter_combine_threshold_bytes=${RS_THRESHOLD}
    --xla_gpu_enable_pipelined_all_gather=false
    --xla_gpu_enable_pipelined_reduce_scatter=false
    --xla_gpu_enable_pipelined_all_reduce=false
    --xla_gpu_enable_while_loop_double_buffering=false
    --xla_gpu_enable_all_gather_combine_by_dim=false
    --xla_gpu_enable_reduce_scatter_combine_by_dim=false
    --xla_disable_hlo_passes=rematerialization"

python3 -m MaxText.train \
    "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml \
    model_name=${MODEL} \
    quantization=fp8 \
    per_device_batch_size=0.125 \
    steps=1 \
    scan_layers=true \
    monitor_goodput=false \
    enable_goodput_recording=false \
    remat_policy=minimal_with_context \
    attention=cudnn_flash_te \
    max_target_length=4096 \
    use_iota_embed=true \
    logits_dot_in_fp32=false\
    enable_checkpointing=false \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=8 \
    base_output_directory=$BASE_OUTPUT_DIR \
    dataset_path=local \
    dataset_type=synthetic \
    hardware=gpu \
    run_name=$RUN_NAME

FILE_PATTERN="module_[0-9]+\.jit_train_step\.sm_[0-9]+\.[0-9]+_gpu_after_optimizations\.txt"

search_file() {
    local dir="$1"
    local pattern="$2"
    find "$dir" -type f | grep -E ".*/${pattern}"
}

HLO_FILE=$(search_file $BASE_OUTPUT_DIR/$RUN_NAME/HLO_dumps/ "$FILE_PATTERN")

if [ ! -f "$HLO_FILE" ]; then
    echo "Error: $HLO_FILE file does not exist."
    exit 1
fi

EXPECTED_FP8_GEMM=$((21))

python3 -m tests.end_to_end.gpu.test_feature fp8_gemm $HLO_FILE $((EXPECTED_FP8_GEMM))
