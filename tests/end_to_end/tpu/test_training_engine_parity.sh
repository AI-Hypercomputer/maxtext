#!/bin/bash
# Complete 6-Step Training Engine Parity Verification Suite
#
# Usage on TPU VM:
#   1. Activate your MaxText Python virtual environment (if applicable):
#      source /path/to/your/maxtext_venv/bin/activate
#   2. Navigate to your MaxText repository root and run this script:
#      cd ~/workspace/maxtext
#      bash tests/end_to_end/tpu/test_training_engine_parity.sh
#
# Why run serially in separate Python processes?
#   Running production LLMs (e.g., Llama 3.1 8B) sequentially within a single Python invocation
#   accumulates compiled XLA binaries and device HBM buffers across test functions, reaching 339GB+
#   host memory usage and triggering OOM stalls. This script runs each JIT suite as an isolated
#   process invocation to guarantee a clean JAX memory footprint for every test step.

set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

# Shared CLI overrides for Llama 3.1 8B evaluation across 4 sharded TPU chips
LLAMA_OVERRIDES=(
  "src/maxtext/configs/post_train/rl.yml"
  "model_name=llama3.1-8b"
  "use_pathways=false"
  "chips_per_vm=4"
  "ici_fsdp_parallelism=-1"
  "batch_size=8"
  "max_target_length=32"
  "convert_checkpoint_if_possible=false"
)

echo "=========================================================================="
echo "=== [Steps 1-3] Eager Mode Evaluation on default (Tiny MLP)            ==="
echo "=========================================================================="
python3 tests/end_to_end/tpu/compare_training_engine.py model_name=default test_suite=eager_all

echo ""
echo "=========================================================================="
echo "=== [Step 4] JIT Evaluation: Standalone train_step vs Engine (Llama 8B) ==="
echo "=========================================================================="
python3 tests/end_to_end/tpu/compare_training_engine.py "${LLAMA_OVERRIDES[@]}" test_suite=jit_train_step

echo ""
echo "=========================================================================="
echo "=== [Step 5] JIT Evaluation: Auxiliary Metrics & Telemetry (Llama 8B)  ==="
echo "=========================================================================="
python3 tests/end_to_end/tpu/compare_training_engine.py "${LLAMA_OVERRIDES[@]}" test_suite=jit_auxiliary_metrics

echo ""
echo "=========================================================================="
echo "=== [Step 6] JIT Evaluation: Multi-Microbatch Gradient Accumulation    ==="
echo "=========================================================================="
python3 tests/end_to_end/tpu/compare_training_engine.py "${LLAMA_OVERRIDES[@]}" test_suite=jit_gradient_accumulation

echo ""
echo "=========================================================================="
echo "✓ ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY IN ISOLATED PROCESSES!"
echo "=========================================================================="
