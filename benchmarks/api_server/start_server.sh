# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash
# This script starts the MaxText server from the project root,
# ensuring that Python can find the necessary modules regardless of
# where the script is invoked from.
#
# Example:
# bash benchmarks/api_server/start_server.sh \
#     MaxText/configs/base.yml \
#     model_name="qwen3-30b-a3b" \
#     tokenizer_path="Qwen/Qwen3-30B-A3B-Thinking-2507" \
#     load_parameters_path="<path_to_your_checkpoint>" \
#     per_device_batch_size=4 \
#     ici_tensor_parallelism=4 \
#     max_prefill_predict_length=1024 \
#     max_target_length=2048 \
#     async_checkpointing=false \
#     scan_layers=false \
#     attention="dot_product" \
#     tokenizer_type="huggingface" \
#     return_log_prob=True
set -e


# Check if arguments were provided.
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_base_config.yml> [arg1=value1 arg2=value2 ...]" >&2
  echo "Or: $0 <path_to_full_model_config.yml>" >&2
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
