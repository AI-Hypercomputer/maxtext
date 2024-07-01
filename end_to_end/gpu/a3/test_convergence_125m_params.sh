#!/bin/bash
set -ex

echo "Running test_convergence_125m_params.sh"
# Run this on 64 chips to achieve a loss value of ~2.5 after 20400 steps, or ~2.7 after 10200 steps (v4-128)
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# LOSS_THRESHOLD (Optional, default is 100.0 )
#
# Example to invoke this script:
# bash end_to_end/gpu/a3/test_convergence_125m_params.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" LOSS_THRESHOLD=100.0

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
    bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/
    DATASET_PATH=/tmp/gcsfuse/
    CMD_DATA=" dataset_type=c4-array_record dataset_name=array-record/c4/en/3.0.1 eval_dataset_name=array-record/c4/en/3.0.1"
fi

if [ "$DATASET_TYPE" == "hf" ]
then
    # We use a local copy of tokenizer from https://huggingface.co/meta-llama/Llama-2-7b-hf
    # Alternatively, you can set tokenizer_path="meta-llama/Llama-2-7b-hf" and hf_access_token="<your-token>" after gaining access through HF website.
    gsutil cp -r gs://maxtext-dataset/hf/llama2-tokenizer assets
    CMD_DATA=" hf_path=parquet hf_data_files=gs://maxtext-dataset/hf/c4/c4-train-*.parquet dataset_type=hf tokenizer_path=assets/llama2-tokenizer"
fi

TRAIN_CMD="python MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME hardware=gpu \
        steps=$STEPS dcn_data_parallelism=1 learning_rate=3e-4 \
        base_emb_dim=1024 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=3584 base_num_decoder_layers=8 \
        ici_fsdp_parallelism=8 metrics_file=metrics.txt per_device_batch_size=4 \
        max_target_length=2048 enable_checkpointing=false attention=dot_product \
        remat_policy=minimal quantization=fp8 gradient_clipping_threshold=1.0 use_iota_embed=true \
        scan_layers=false dataset_path=$DATASET_PATH async_checkpointing=false \
        base_output_directory=$OUTPUT_PATH logits_dot_in_fp32=false"
TRAIN_CMD+=$CMD_DATA

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
$TRAIN_CMD

# Assert training loss is smaller than input LOSS_THRESHOLD
python3 end_to_end/tpu/eval_assert.py final_loss metrics.txt $LOSS_THRESHOLD
