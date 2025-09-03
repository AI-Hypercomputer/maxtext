#!/bin/bash
# This script starts the MaxText server from the project root,
# ensuring that Python can find the necessary modules regardless of
# where the script is invoked from.
set -e

# Check if arguments were provided.
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_base_config.yml> [arg1=value1 arg2=value2 ...]"
  echo "Or: $0 <path_to_full_model_config.yml>"
  exit 1
fi

# Get the absolute path of the directory where the script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The project root is two levels up from the script's directory.
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# Change to the project root directory.
cd "$PROJECT_ROOT"

echo "Starting MaxText server on http://0.0.0.0:8000"
echo "Executing from project root: $(pwd)"
echo "Using arguments: $@"

# Pass all script arguments directly to the python module.
python -u -m benchmarks.api_server.maxtext_server "$@"
