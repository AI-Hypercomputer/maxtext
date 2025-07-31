#!/bin/bash
# This script starts the MaxText server from the project root,
# ensuring that Python can find the necessary modules regardless of
# where the script is invoked from.
set -e

# Check if a model config file was provided.
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_model_config.yml>"
  exit 1
fi

MODEL_CONFIG_PATH="$1"

# Get the absolute path of the directory where the script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The project root is two levels up from the script's directory.
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# Change to the project root directory.
cd "$PROJECT_ROOT"

echo "Starting MaxText server on http://0.0.0.0:8000"
echo "Executing from project root: $(pwd)"
echo "Using model config: $MODEL_CONFIG_PATH"

# Export the environment variable for the server to use.
export MAXTEXT_MODEL_CONFIG="$MODEL_CONFIG_PATH"

# Now that we are in the project root, the module path is correct.
python -m uvicorn benchmarks.lm_eval.maxtext_server:app --host 0.0.0.0 --port 8000
