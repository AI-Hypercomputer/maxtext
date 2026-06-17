#!/bin/bash
# Wrapper script to run RL training on GKE.
# Patches tpu_inference at runtime to apply the cache-clearing fix.

set -e

echo "=== GKE Wrapper Script Start ==="
export PATH="/usr/local/bin:$PATH"
env | grep -E "JAX|TPU|VLLM|PATH" || true

# Environment checks

# Run integration patches unconditionally
echo "Applying MaxText/Tunix RL integration patches..."
python3 scratch/patch_vllm.py

# Patch tpu_inference JAX cache if APPLY_PATCH is 1
if [ "${APPLY_PATCH}" = "1" ]; then
  echo "Applying JAX cache patch..."
  TPU_INF_DIR=$(python3 -c "import tpu_inference; import os; print(os.path.dirname(tpu_inference.__file__))" | tail -n 1)
  echo "Found tpu_inference at: ${TPU_INF_DIR}"
  echo "Applying JAX cache patch to weight_utils.py..."
  sed -i 's/jax.clear_caches()/# jax.clear_caches()/g' "${TPU_INF_DIR}/models/jax/utils/weight_utils.py"
  echo "JAX cache patch applied successfully."
else
  echo "Skipping JAX cache patch (APPLY_PATCH != 1)"
fi

# Uninstall triton to prevent segfault on import
echo "Uninstalling triton..."
pip uninstall -y triton

# Set required environment variables
export NEW_MODEL_DESIGN=1
export SKIP_JAX_PRECOMPILE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_PLATFORMS=proxy,cpu
if [ "${DISABLE_PERSISTENT_CACHE}" = "1" ]; then
  echo "Disabling JAX persistent compilation cache (GCS)..."
  export JAX_ENABLE_COMPILATION_CACHE=false
fi
export PYTHONFAULTHANDLER=1
export JAX_DEBUG_SIGNALS=1

echo "Starting train_rl.py with args: $@"
python3 src/maxtext/trainers/post_train/rl/train_rl.py "$@"

