#!/usr/bin/env bash
set -euo pipefail
source ./pipeline.env

# Helpers
mount_gcs() {
  local bucket_uri="$1" mount_path="$2"
  local name="${bucket_uri#gs://}"

  if mountpoint -q "$mount_path"; then
    echo "[gcsfuse] unmounting existing $mount_path"
    gcsfuse -unmount "$mount_path" 2>/dev/null || fusermount -u "$mount_path" 2>/dev/null || true
  fi

  mkdir -p "$mount_path"
  echo "[gcsfuse] mounting $name at $mount_path (rw)"
  gcsfuse --implicit-dirs "$name" "$mount_path"
  echo "[gcsfuse] mounted $bucket_uri -> $mount_path"
}

unmount_gcs() {
  local mount_path="$1"
  if mountpoint -q "$mount_path"; then
    echo "[gcsfuse] unmounting $mount_path"
    gcsfuse -unmount "$mount_path" 2>/dev/null || fusermount -u "$mount_path" 2>/dev/null || true
  fi
}

# Mount and prep bucket
mount_gcs "${BUCKET_URI}" "${MOUNT_PATH}"
mkdir -p "${CHKPT_DIR}"

# Upload metadata/tokenizer once (skip .safetensors)
echo "[sync] syncing non-safetensors -> ${CHKPT_BUCKET}"
gsutil -m rsync -r -x '.*\.safetensors$' "${FP8_DIR}" "${CHKPT_BUCKET}"

# Converter deps
python3 -m pip install -q --disable-pip-version-check \
  torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -q --disable-pip-version-check safetensors==0.4.5

# Convert FP8 (local) -> BF16
echo "[convert] FP8 -> BF16 writing to ${CHKPT_DIR}"
python3 -m MaxText.deepseek_fp8_to_bf16 \
  --input-fp8-hf-path  "${FP8_DIR}" \
  --output-bf16-hf-path "${CHKPT_DIR}" \
  --cache-file-num 1

echo "[INFO] BF16 written to ${CHKPT_BUCKET}"
