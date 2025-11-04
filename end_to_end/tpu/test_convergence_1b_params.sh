#!/bin/bash
set -ex

echo "Running test_convergence_1b_params.sh"
# Run this on 64 chips to achieve a loss value of ~2.5 after 20400 steps, or ~2.7 after 10200 steps (v4-128)
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# LOSS_THRESHOLD (Optional, default is 100.0 )
#
# Example to invoke this script:
# bash end_to_end/tpu/test_convergence_1b_params.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" LOSS_THRESHOLD=100.0

# default values, can be override from command line
export LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass.
export STEPS=20400 # Run for 20B tokens for a 1B sized mode for "chinchilla" scaling https://arxiv.org/abs/2203.15556
export EVAL_STEPS=160
export EVAL_INTERVAL=100
export DATASET_TYPE=tfds
export MTP_NUM_LAYERS=0 # Disable MTP by default
export PER_DEVICE_BATCH_SIZE=8.0 # With the default learning rate (3e-4) this should have global batch of 512, with 2k sequence length (1M global batch in tokens)

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [ -n "$RUN_NAME" ];
then
    export M_RUN_NAME=$RUN_NAME
fi

if [ "$DATASET_TYPE" == "grain" ]
then
    EVAL_METRICS=grain_checkpoint_save_restore
    echo "dataset_type is grain"
    echo "Mounting $DATASET_PATH to /tmp/gcsfuse/"
    bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/
    DATASET_PATH=/tmp/gcsfuse/
    CMD_DATA=" grain_worker_count=2 \
            grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record* \
            grain_eval_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-validation.array_record* "
fi

if [ "$DATASET_TYPE" == "hf" ]
then
    # We use a local copy of tokenizer from https://huggingface.co/meta-llama/Llama-2-7b-hf
    # Alternatively, you can set tokenizer_path="meta-llama/Llama-2-7b-hf" and hf_access_token="<your-token>" after gaining access through HF website.
    gsutil cp -r gs://maxtext-dataset/hf/llama2-tokenizer "${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}"
    CMD_DATA=" hf_path=parquet tokenizer_path=${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText/assets}}/llama2-tokenizer"\
        "hf_train_files=$DATASET_PATH/hf/c4/c4-train-*.parquet" \
        "hf_eval_files=$DATASET_PATH/hf/c4/c4-validation-*.parquet "
fi

TRAIN_CMD="python3 -m MaxText.train ${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/configs/base.yml \
        steps=$STEPS eval_steps=$EVAL_STEPS eval_interval=$EVAL_INTERVAL \
        per_device_batch_size=$PER_DEVICE_BATCH_SIZE learning_rate=3e-4 enable_checkpointing=false \
        max_target_length=2048 global_parameter_scale=1 \
        metrics_file=metrics.txt base_output_directory=$OUTPUT_PATH \
        dataset_path=$DATASET_PATH dataset_type=$DATASET_TYPE log_period=150 \
        remat_policy=minimal enable_data_shuffling=false mtp_num_layers=$MTP_NUM_LAYERS"
TRAIN_CMD+=$CMD_DATA

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
$TRAIN_CMD

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/tpu/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD
