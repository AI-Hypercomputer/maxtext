#!/bin/bash

# End-to-end image (multimodal) parity check for Gemma-4 E2B.
#
# Unlike gemma4/e2b/convert_gemma4.sh (text-only), this exercises the image path enabled by
# PR #4790: vision clipped-linears + padded-patch masking + PLE image-row pad substitution +
# causal image spans. It converts the HF checkpoint (with the vision tower), pre-generates the
# HF golden logits for an image+prompt, and runs forward_pass_logit_checker on the NNX path
# (enable_nnx defaults True; per-layer KV sharing is incompatible with nn.scan, so scan_layers=false).
#
# See tests/end_to_end/tpu/gemma4/Run_Gemma4.md for an overview.
#
# Usage:
#   export HF_TOKEN=<your Hugging Face access token>
#   export HF_MODEL=path/to/your/hf/gemma-4-E2B-it
#   bash tests/end_to_end/tpu/gemma4/e2b/test_gemma4_multimodal.sh

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

MODEL_NAME='gemma4-e2b'
export MODEL_VARIATION='e2b-it'
MODEL_ID='google/gemma-4-E2B-it'
TOKENIZER_PATH='google/gemma-4-E2B-it'
USE_SCAN_LAYERS=false  # Per-layer KV sharing is incompatible with nn.scan.
IMAGE_PATH='tests/assets/test_image.jpg'
PROMPT='Describe this image'

# Installing torch for deps in the HF golden-logits generator / logit checker.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# HF checkpoint (must contain the vision tower for multimodal).
export MODEL_BUCKET='gs://maxtext-gemma/gemma4'
export HF_MODEL=${HF_MODEL:-'path/to/your/hf/gemma-4-E2B-it'}

# Step 1: HF -> MaxText conversion WITH the vision tower (use_multimodal=true).
python3 -m maxtext.checkpoint_conversion.to_maxtext "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    --hf_model_path=${HF_MODEL} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_VARIATION}/converted_mm/${idx} \
    use_multimodal=true \
    scan_layers=${USE_SCAN_LAYERS}

export MAXTEXT_CKPT_PATH=${MODEL_BUCKET}/${MODEL_VARIATION}/converted_mm/${idx}/0/items

# Step 2: Pre-generate the HF golden logits for the image+prompt (required for multimodal).
GOLDEN_LOGITS_PATH="golden_${MODEL_VARIATION}_vision_${idx}.jsonl"
python3 -m tests.assets.logits_generation.generate_hf_golden_logits \
    --model-id=${MODEL_ID} --hf-model-path=${HF_MODEL} \
    --prompts="${PROMPT}" --image-paths=${IMAGE_PATH} \
    --output-path=${GOLDEN_LOGITS_PATH} \
    --apply-chat-template --output-format=json

# Step 3: MaxText vs HF golden-logit parity on the image path.
# Gate mirrors the PR's post_image parity (max_KL << 0.03 observed at 4.29e-05); we use the
# conservative repo default multimodal gate here.
python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"/base.yml \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${MAXTEXT_CKPT_PATH} \
    model_name=${MODEL_NAME} \
    use_multimodal=true \
    use_clipped_linears_for_vit=true \
    scan_layers=${USE_SCAN_LAYERS} \
    per_device_batch_size=1 \
    dtype=float32 \
    attention=dot_product \
    prompt="${PROMPT}" \
    image_path=${IMAGE_PATH} \
    --max_kl_div=0.03 \
    --golden_logits_path=${GOLDEN_LOGITS_PATH}
