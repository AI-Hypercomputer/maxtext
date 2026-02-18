set -euox pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M)
export PAXML_CKPT_PATH=gs://maxtext-gpt3/ckpt_test/paxml/checkpoints/checkpoint_00000000/state
export OUTPUT_PATH=gs://maxtext-gpt3/tests
export RUN_NAME=test_${TIMESTAMP}

# convert gpt3-52k model
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_gpt3_ckpt_from_paxml --paxml-ckpt-path=${PAXML_CKPT_PATH} --maxtext-model-name=gpt3-52k --run-name=${RUN_NAME} --base-output-directory=${OUTPUT_PATH}

# Run gpt3-52k with the converted ckpt
python3 -m MaxText.train "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml run_name=${RUN_NAME} model_name=gpt3-52k\
    steps=10 per_device_batch_size=6 enable_checkpointing=true async_checkpointing=false\
    remat_policy=full max_target_length=2048 base_output_directory=${OUTPUT_PATH}\
    dataset_type=synthetic
