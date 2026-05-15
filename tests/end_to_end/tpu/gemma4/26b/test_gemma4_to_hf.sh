#!/bin/bash

# This script is both an end-to-end test and documentation for converting a
# Gemma4-26B MaxText checkpoint to Hugging Face format. Can be run on a v5p-8.

# The flow of this script is as follows:
# 1. Convert a MaxText checkpoint to a Hugging Face model checkpoint.
# 2. Run a forward pass check to compare the logits and KL divergence between
#    the converted checkpoint and the original HF model.

# Pre-requisites:
# 1. Set HF_TOKEN environment variable to your Hugging Face access token.
#    export HF_TOKEN=<Hugging Face access token>
# 2. Provide a MaxText-format Gemma4-26B checkpoint via CKPT_PATH.
#    One can be produced with tests/end_to_end/tpu/gemma4/26b/convert_gemma4.sh.

set -ex
idx=$(date +%Y-%m-%d-%H-%M)
MODEL_NAME='gemma4-26b'
export MODEL_VARIATION='26b-it'
# To convert the multimodal model, set USE_MULTIMODAL=true
USE_MULTIMODAL=false
# Set USE_SCAN_LAYERS=true if the checkpoint was trained with scanned layers
USE_SCAN_LAYERS=true

# Installing torch for deps in forward_pass_logit_checker.py
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Non-Googlers: point MODEL_BUCKET to a GCS bucket you own.
export MODEL_BUCKET=gs://maxtext-gemma/gemma4
# Path to a pre-existing MaxText checkpoint for gemma4-26b. Must match USE_SCAN_LAYERS.
# Run tests/end_to_end/tpu/gemma4/26b/convert_gemma4.sh to produce one.
export CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/converted/unscanned/0/items

# Path to the original HF model weights for logit comparison.
export HF_MODEL=google/gemma-4-26b-a4b-it

export LOCAL_PATH=./tmp/hf/${MODEL_NAME}/${idx}

python3 -m maxtext.checkpoint_conversion.to_huggingface \
    "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${CKPT_PATH} \
    base_output_directory=${LOCAL_PATH} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=${USE_SCAN_LAYERS}

# Run forward pass logit checker to validate the converted checkpoint.
# The *_tile_fwd_*_dim flags are for reducing vmem usage to fit into v5p chips,
# not for performance purpose.
if [ "${USE_MULTIMODAL}" == true ]; then
    TEST_PROMPT='Describe image <|image|>'
    TEST_IMAGE='tests/assets/test_image.jpg'
    export GOLDEN_LOGITS_PATH=/tmp/golden_gemma4_26b_vision.pickle

    python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
        --model-id=${HF_MODEL} \
        --output-path=${GOLDEN_LOGITS_PATH} \
        --prompts="${TEST_PROMPT}" \
        --image-paths=${TEST_IMAGE} \
        --hf-model-path=${LOCAL_PATH} \
        --output-format=pickle

    echo "=== Running MaxText Forward Pass Logit Checker ==="
    python3 -m tests.utils.forward_pass_logit_checker \
        "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
        tokenizer_path=${HF_MODEL} \
        load_parameters_path=${CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=${USE_SCAN_LAYERS} \
        dtype=float32 \
        wi_tile_fwd_embed_dim=512 \
        wi_tile_fwd_mlp_dim=512 \
        wo_tile_fwd_embed_dim=512 \
        wo_tile_fwd_mlp_dim=512 \
        matmul_precision=highest \
        per_device_batch_size=1 \
        attention=dot_product \
        prompt="${TEST_PROMPT}" \
        image_path=${TEST_IMAGE} \
        --max_kl_div=0.1 \
        --golden_logits_path=${GOLDEN_LOGITS_PATH}
else
    echo "=== Running MaxText Forward Pass Logit Checker ==="
    python3 -m tests.utils.forward_pass_logit_checker \
        "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
        tokenizer_path=${HF_MODEL} \
        load_parameters_path=${CKPT_PATH} \
        model_name=${MODEL_NAME} \
        use_multimodal=${USE_MULTIMODAL} \
        scan_layers=${USE_SCAN_LAYERS} \
        per_device_batch_size=1 \
        dtype=float32 \
        wi_tile_fwd_embed_dim=512 \
        wi_tile_fwd_mlp_dim=512 \
        wo_tile_fwd_embed_dim=512 \
        wo_tile_fwd_mlp_dim=512 \
        --max_kl_div=0.1 \
        --run_hf_model=true \
        --hf_model_path=${LOCAL_PATH}
fi
