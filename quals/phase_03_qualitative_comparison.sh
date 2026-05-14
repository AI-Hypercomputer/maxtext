#!/bin/bash
# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
  source ~/maxtext_venv/bin/activate
fi

export PYTHONPATH=src

# Set base output directory
BASE_OUTPUT_DIR="gs://igorts_europe/ttl=30d/dpo_quals"
DPO_CHECKPOINT="${BASE_OUTPUT_DIR}/maxtext_run/dpo-verification-qwen-v3/inference_ckpt/0/items"
SFT_BASELINE="gs://igorts_europe/converted_checkpoints/qwen2.5-1.5b-instruct_mcjax_v2/0/items"

# Common config
COMMON_ARGS="run_name=qualitative_comparison base_output_directory=${BASE_OUTPUT_DIR} dataset_type=synthetic steps=0 model_name=qwen2.5-1.5b attention=dot_product scan_layers=False per_device_batch_size=1 max_prefill_predict_length=512 max_target_length=1024"

PROMPTS=(
    "Explain the concept of Direct Preference Optimization in simple terms."
    "Write a short story about a robot learning to cook."
    "What are the pros and cons of using JAX for machine learning?"
    "How do I optimize a MaxText training run on TPU v4-8?"
    "Give me a recipe for a healthy vegetarian dinner."
)

generate_responses() {
    local checkpoint=$1
    local output_file=$2
    local label=$3
    echo "=== Generating responses for ${label} from ${checkpoint} ==="
    > "$output_file"
    for i in "${!PROMPTS[@]}"; do
        local prompt="${PROMPTS[$i]}"
        echo "Prompt: $prompt" >> "$output_file"
        python3 -m maxtext.inference.decode \
            ${COMMON_ARGS} \
            run_name="qual-eval-${label}-${i}" \
            load_parameters_path="${checkpoint}" \
            prompt="$prompt" >> "$output_file" 2>&1
    done
}

# Create local logs directory if it doesn't exist
mkdir -p quals/logs

echo "=== Generating responses from SFT Baseline ==="
generate_responses "${SFT_BASELINE}" "quals/logs/sft_responses.log" "sft"

echo "=== Generating responses from DPO Checkpoint ==="
generate_responses "${DPO_CHECKPOINT}" "quals/logs/dpo_responses.log" "dpo"

echo "Decoding complete. Compare quals/logs/sft_responses.log and quals/logs/dpo_responses.log."
