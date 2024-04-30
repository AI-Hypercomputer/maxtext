#!/bin/bash
set -ex

if [ -f "run_*_metrics.txt" ]; then
    rm run_*_metrics.txt
    echo "removed existing run_*_metrics.txt"
fi

RUN_NAME=${1}-$(date +%Y-%m-%d-%H-%M)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
model_params=" base_emb_dim=384 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=192 base_num_decoder_layers=8 head_dim=128"

echo "Mounting $DATASET_PATH to /tmp/gcsfuse/"
bash setup_gcsfuse.sh DATASET_GCS_BUCKET=$DATASET_PATH MOUNT_PATH=/tmp/gcsfuse/

echo "Run_1: Starting the first run using the grain input pipeline"

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=3 ${model_params}\
    max_target_length=128 per_device_batch_size=1\
    metrics_file=run_1_metrics.txt checkpoint_period=2 async_checkpointing=false\
    dataset_path=/tmp/gcsfuse base_output_directory=$OUTPUT_PATH\
    dataset_type=c4-array_record grain_worker_count=0\
    dataset_name=array-record/c4/en/3.0.1 eval_dataset_name=array-record/c4/en/3.0.1

echo
echo "Finished Run_1 at step 2"
echo "Run_2: Resuming using the tfds input pipeline"
echo

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=5 ${model_params}\
    max_target_length=128 per_device_batch_size=1\
    metrics_file=run_2_metrics.txt checkpoint_period=2 async_checkpointing=false\
    dataset_path=/tmp/gcsfuse base_output_directory=$OUTPUT_PATH\

echo
echo "Finished Run_2 at step 4"
echo "Run_3: Resuming using the grain input pipeline"
echo

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=7 ${model_params}\
    max_target_length=128 per_device_batch_size=1\
    metrics_file=run_3_metrics.txt checkpoint_period=2 async_checkpointing=false\
    dataset_path=/tmp/gcsfuse base_output_directory=$OUTPUT_PATH\
    dataset_type=c4-array_record grain_worker_count=0\
    dataset_name=array-record/c4/en/3.0.1 eval_dataset_name=array-record/c4/en/3.0.1

python3 end_to_end/tpu/eval_assert.py test_start_step run_2_metrics.txt 3.0
python3 end_to_end/tpu/eval_assert.py test_start_step run_3_metrics.txt 5.0
