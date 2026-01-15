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

"""
This file provides utilities to extract basic metadata about Hugging Face models
without downloading their full weights. It's useful for quickly getting
information like the main architecture class name, the expected file path within
the `transformers` library, and the model type.

Example Invocations:

1. Get info for a specific model:
   python get_model_info.py --model-id "meta-llama/Llama-3-8B"

2. Get info for another model:
   python get_model_info.py --model-id "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
"""
import argparse
from transformers import AutoConfig


def get_model_info(model_id: str):
  """Return basic model metadata from Hugging Face without loading weights.

  Loads only the `AutoConfig` for the given `model_id` to infer the primary
  architecture class name, the expected modeling file path within the
  `transformers` package, and the model type token.

  Args:
      model_id (str): A Hugging Face model identifier (e.g., "meta-llama/Llama-3-8B").

  Returns:
      dict: A dictionary with keys:
          - "class_name" (str | None): The architecture class name, if available.
          - "file_path" (str): The canonical `transformers` path to the modeling file.
          - "model_type" (str): The model type string from the config.
  """
  # Load config only (very lightweight, no weights)
  config = AutoConfig.from_pretrained(model_id)

  # Step 1: Class name comes from architectures
  class_name = config.architectures[0] if getattr(config, "architectures", None) else None

  # Step 2: Model type gives us the directory + file
  model_type = config.model_type
  file_path = f"transformers/models/{model_type}/modeling_{model_type}.py"

  return {"class_name": class_name, "file_path": file_path, "model_type": model_type}


def parse_args():
  """
  Parses command-line arguments for file or folder processing.

  Returns:
      argparse.Namespace: The parsed command-line arguments.
  """
  parser = argparse.ArgumentParser(description="Get model info from HuggingFace model id")
  parser.add_argument(
      "--model-id", type=str, default="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", help="HuggingFace model id"
  )
  return parser.parse_args()


# Example Run
if __name__ == "__main__":
  info = get_model_info(parse_args().model_id)

  print("Class Name:", info["class_name"])
  print("File Path:", info["file_path"])
  print("Model Type:", info["model_type"])
