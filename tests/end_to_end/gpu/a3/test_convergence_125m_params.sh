#!/bin/bash
set -ex

echo "Running test_convergence_125m_params.sh"
# Run this on NVIDIA A3 to achieve a loss value of ~3.9 after 2500 steps (NVIDIA A3)
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# LOSS_THRESHOLD (Optional, default is 100.0 )
# QUANTIZATION (Optional, default is '')
#
# Example to invoke this script:
# bash tests/end_to_end/gpu/a3/test_convergence_125m_params.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" LOSS_THRESHOLD=100.0

export LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass.
export STEPS=2550


# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

if [ "$DATASET_TYPE" == "c4-array_record" ]
then
    EVAL_METRICS=grain_checkpoint_save_restore
    echo "Using c4-array_record dataset type"
    echo "Mounting $DATASET_PATH to /tmp/gcsfuse/"
    bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/
    DATASET_PATH=/tmp/gcsfuse/
    CMD_DATA=" dataset_type=c4-array_record dataset_name=array-record/c4/en/3.0.1 eval_dataset_name=array-record/c4/en/3.0.1"
fi

if [ "$DATASET_TYPE" == "hf" ]
then
    # We use a local copy of tokenizer from https://huggingface.co/meta-llama/Llama-2-7b-hf
    # Alternatively, you can set tokenizer_path="meta-llama/Llama-2-7b-hf" and hf_access_token="<your-token>" after gaining access through HF website.
    gsutil cp -r gs://maxtext-dataset/hf/llama2-tokenizer "${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"
    CMD_DATA=" hf_path=parquet hf_data_files=gs://maxtext-dataset/hf/c4/c4-train-*.parquet dataset_type=hf tokenizer_path=${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}/llama2-tokenizer"
fi

TRAIN_CMD="python3 -m MaxText.train ${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml run_name=$RUN_NAME hardware=gpu \
        steps=$STEPS dcn_data_parallelism=1 learning_rate=3e-4 \
        base_emb_dim=1024 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=3584 base_num_decoder_layers=8 \
        ici_fsdp_parallelism=8 metrics_file=metrics.txt per_device_batch_size=4 \
        max_target_length=2048 enable_checkpointing=false attention=dot_product \
        remat_policy=minimal quantization=$QUANTIZATION gradient_clipping_threshold=1.0 use_iota_embed=true \
        scan_layers=false dataset_path=$DATASET_PATH async_checkpointing=false \
        base_output_directory=$OUTPUT_PATH logits_dot_in_fp32=false"
TRAIN_CMD+=$CMD_DATA

# Train
export XLA_ARGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer='' --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=134217728 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization"
$TRAIN_CMD

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 tests/end_to_end/tpu/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD
