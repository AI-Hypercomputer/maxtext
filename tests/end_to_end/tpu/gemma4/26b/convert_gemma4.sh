#!/bin/bash

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

MODEL_NAME='gemma4-26b'
export MODEL_VARIATION='26b'
TOKENIZER_PATH='google/gemma-4-26b-a4b-it'
# To convert the multimodal model, make sure the use_multimodal is set to be true
USE_MULTIMODAL=false
USE_SCAN_LAYERS=false


# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# After downloading checkpoints, copy them to GCS bucket at $MODEL_BUCKET
export MODEL_BUCKET='gs://maxtext-gemma/gemma4'
export HF_MODEL='path/to/your/hf/gemma-4-26b-a4b-it'

# To get converted ckpt:
python3 -m maxtext.checkpoint_conversion.to_maxtext "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    --hf_model_path=${HF_MODEL} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_VARIATION}/converted/${idx} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=${USE_SCAN_LAYERS}


export MAXTEXT_CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/converted/${idx}/0/items


if [ ${USE_MULTIMODAL} == true ]; then
    # Set the shared multimodal prompt and image
    TEST_PROMPT='Describe image <|image|>'
    TEST_IMAGE='tests/assets/test_image.jpg'
    export GOLDEN_LOGITS_PATH=/tmp/golden_gemma4_26b_vision.pickle

    python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
        --model-id=${TOKENIZER_PATH} \
        --output-path=${GOLDEN_LOGITS_PATH} \
        --prompts="${TEST_PROMPT}" \
        --image-paths=${TEST_IMAGE} \
        --hf-model-path=${HF_MODEL} \
        --output-format=pickle

    echo "=== Running MaxText Forward Pass Logit Checker ==="
    python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
        tokenizer_path=${TOKENIZER_PATH} \
        load_parameters_path=${MAXTEXT_CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=${USE_SCAN_LAYERS} \
        dtype=float32 \
        matmul_precision=highest \
        per_device_batch_size=1 \
        vision_output_length=280 \
        attention=dot_product \
        prompt="${TEST_PROMPT}" \
        image_path=${TEST_IMAGE} \
        --max_kl_div=0.03 \
        --golden_logits_path=${GOLDEN_LOGITS_PATH}
else
    python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
        tokenizer_path=${TOKENIZER_PATH}  \
        load_parameters_path=${MAXTEXT_CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=${USE_SCAN_LAYERS} \
        per_device_batch_size=1 \
        dtype=float32 \
        --max_kl_div=0.03 \
        --run_hf_model=true \
        --hf_model_path=${HF_MODEL}
fi