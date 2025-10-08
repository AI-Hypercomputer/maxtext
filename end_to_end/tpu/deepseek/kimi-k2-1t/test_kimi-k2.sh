#!/usr/bin/env bash
set -euo pipefail
source ./pipeline.env

# This file tests the implementation of Kimi-K2.

# The flow of this file is as follows:
# 1. Download the checkpoint from HuggingFace (fp8 weights).
# 2. Convert the checkpoint from FP8 to BF16 in HuggingFace format and upload the BF16 checkpoint to your GCS bucket.
# 3a. Convert the BF16 HuggingFace checkpoint to a MaxText scanned Orbax checkpoint.
# 3b. Convert the scanned checkpoint to an unscanned checkpoint for efficient decoding.
# 4. Run logits check test between HuggingFace and MaxText using the unscanned checkpoint.

# Skip download/convert if BF16 exists in bucket
if gsutil -q stat "${CHKPT_BUCKET}/model.safetensors.index.json" || gsutil ls "${CHKPT_BUCKET}/*.safetensors" >/dev/null 2>&1; then
  echo "[skip] BF16 already present in ${CHKPT_BUCKET}. Skipping 01 & 02."
else
  ./01_download_fp8.sh
  ./02_convert_fp8_to_bf16_sharded.sh
fi

./03_convert_to_orbax.sh
./04_logit_check.sh
