set -euox pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M)
export OUTPUT_PATH=gs://maxtext-gpt3/tests
export RUN_NAME=test_${TIMESTAMP}

# Run gpt3-52k with the converted ckpt
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} model_name=gpt3-6b\
    steps=10 per_device_batch_size=1 enable_checkpointing=false async_checkpointing=false\
    enable_profiler=false remat_policy=full\
    attention=dot_product\
    base_output_directory=${OUTPUT_PATH}\
    dataset_type=synthetic
