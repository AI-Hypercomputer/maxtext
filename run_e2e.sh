#!/bin/bash
set -e

echo "=========================================================================="
echo "Starting MaxText End-to-End TPU Validation Matrix"
echo "=========================================================================="
echo "This script will orchestrate 224 training jobs across 8 models and 2 scan modes."
echo "Models and Checkpoints will be saved to GCS: gs://mesa-maxtext/validation_runs/post_train_layout_v1/"
echo "Execution Logs will be saved locally to: ./local_logs/"
echo ""

# Ensure we are in the correct MaxText directory
if [ ! -f "src/maxtext/trainers/pre_train/train.py" ]; then
    echo "[ERROR] Please run this script from the root of the maxtext repository."
    exit 1
fi

echo "Cleaning up any lingering MaxText python processes to free the TPU..."
pkill -f "python.*maxtext" || true

# Run the python orchestrator
python3 run_e2e_matrix.py

echo "=========================================================================="
echo "Validation Matrix execution completed."
echo "Results written to validation_summary.csv"
echo "To view summary:"
echo "cat validation_summary.csv | column -t -s,"
echo "=========================================================================="
