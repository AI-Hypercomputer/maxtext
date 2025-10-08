#!/usr/bin/env bash
set -euo pipefail
source ./pipeline.env

# Helpers
mount_gcs() {
  local bucket_uri="$1" mount_path="$2"
  local name="${bucket_uri#gs://}"

  if mountpoint -q "$mount_path"; then
    echo "[gcsfuse] unmounting existing $mount_path"
    gcsfuse -u "$mount_path" 2>/dev/null || true
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
    gcsfuse -u "$mount_path" 2>/dev/null || true
  fi
}

mount_gcs "${BUCKET_URI}" "${MOUNT_PATH}"

echo "[orbax] HF BF16 -> scanned @ ${MODEL_BUCKET}/${IDX}"
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_ckpt \
  --base_model_path    "${CHKPT_DIR}" \
  --maxtext_model_path "${MODEL_BUCKET}/${IDX}" \
  --model_size         "${MODEL_NAME}"

echo "[orbax] scanned -> unscanned @ ${MODEL_BUCKET}/${IDX}/unscanned"
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_family_unscanned_ckpt \
  --base_model_path    "${CHKPT_DIR}" \
  --maxtext_model_path "${MODEL_BUCKET}/${IDX}/unscanned" \
  --model_size         "${MODEL_NAME}"

unmount_gcs "${BUCKET_URI}" "${MOUNT_PATH}"
