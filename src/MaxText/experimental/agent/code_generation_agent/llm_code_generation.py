# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file leverages the ``GeminiAgent`` to automate the conversion of Python
code, specifically focusing on transforming PyTorch, NumPy, or similar framework
code into functionally equivalent JAX code.

It can process either a single Python file or all Python files within a
specified folder.

Please add your API key and chosen model name in ``.env`` file.
Please add your output directory in the ``JAX_OUTPUT_DIR`` variable.

Here's a breakdown of its functionality:

1. **Initialization**:

   * It loads environment variables (presumably for API keys and model names)
     using ``python-dotenv``.
   * It sets up logging for better traceability.
   * It defines an output directory (``JAX_OUTPUT_DIR``) for the converted JAX
     files.
   * It initializes a ``GeminiAgent`` instance, providing it with a
     ``SystemPrompt`` from ``prompt_code_generation.py``. This prompt guides the
     Gemini model on its role as an expert machine learning engineer for code
     conversion.

2. **``get_chat_dict`` Function**:

   * A helper function to format user messages into the dictionary structure
     expected by the Gemini API.

3. **``convert_code_from_torch_to_jax`` Function**:

   * This is the core conversion logic.
   * It takes a ``codeComponent`` (a string containing the Python code to be
     converted) and an optional ``memory_list`` (for conversational context).
   * It constructs a user message using the ``CODE`` template from
     ``prompt_code_generation.py``, embedding the ``codeComponent`` into it.
   * It appends this user message to the ``memory_list``.
   * It then calls the ``llm_agent`` (which wraps the Gemini model) with the
     ``memory_list`` to get a response.
   * Upon receiving a response, it extracts the converted code using
     ``parse_python_code`` (from ``orchestration_agent.Utils``) and updates the
     ``memory_list`` with the model's response.
   * It handles potential errors if the LLM agent fails to return a valid
     response.

4. **``parse_args`` Function**:

   * Parses command-line arguments, allowing the user to specify either a single
     file (``--file``) or an entire folder (``--folder``) for processing. This
     ensures mutual exclusivity, meaning only one of these options can be
     provided.

5. **``process_single_file`` Function**:

   * Reads the content of a specified Python file.
   * Calls ``convert_code_from_torch_to_jax`` to perform the conversion.
   * Writes the converted code to the output directory.

6. **``process_files`` Function**:

   * Determines whether to process a single file or all Python files in a
     specified folder based on the parsed command-line arguments.
   * It validates the existence of the specified file or folder and logs errors
     if they do not exist.
   * It iterates over all ``.py`` files in the folder if a folder is specified.

7. **Main Execution Block**:

   * Parses command-line arguments and initiates the file processing.

Example Invocations:

1. Convert a single PyTorch file to JAX::

     python llm_code_generation.py --file dataset/PyTorch/masking_utils__or_masks.py
  
   (Ensure the output directory exists or modify the script to create it if
   necessary.)

2. Convert all PyTorch files in a folder to JAX::

     python llm_code_generation.py --folder dataset/PyTorch/

   (Ensure the output directory exists or modify the script to create it if
   necessary.)
"""

from pathlib import Path
from dotenv import load_dotenv  # If this is not available, try ``pip install python-dotenv``

load_dotenv()
import os
import logging
import argparse

from MaxText.experimental.agent.code_generation_agent.llm_agent import GeminiAgent
from MaxText.experimental.agent.orchestration_agent.utils import parse_python_code
from MaxText.experimental.agent.code_generation_agent.prompt_code_generation import CodeGeneration

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("__name__")

JAX_OUTPUT_DIR = "./dataset/jax_converted"  # Please set your output directory here, eg. "./dataset/jax_converted"
os.makedirs(JAX_OUTPUT_DIR, exist_ok=True)

llm_agent = GeminiAgent(CodeGeneration["SystemPrompt"])


def get_chat_dict(input_message=""):
  """
  Creates a chat dictionary for a user message.

  Args:
    input_message (str, optional): The user's message. Defaults to "".

  Returns:
    dict: A dictionary formatted for the Gemini API.
  """
  return {"role": "user", "parts": input_message}


def convert_code_from_torch_to_jax(codeComponent, memory_list):
  """
  Converts a single code component from PyTorch to JAX using the LLM agent.

  Args:
    code_component (str): The Python code to be converted.
    memory_list (list, optional): A list of previous chat messages to provide
      context. Defaults to None.

  Returns:
    tuple
      A tuple containing

      str:
        The converted JAX code.
      list:
        The updated memory list.
  """
  if memory_list is None:
    memory_list = []
  # This appends the user's input message to the memory list for context.
  memory_list.append(get_chat_dict(input_message=CodeGeneration["CODE"].replace("{TORCHCODE}", codeComponent)))
  response = llm_agent(memory_list)
  if response:
    converted_code = parse_python_code(response.text)
    memory_list.append({"role": "model", "parts": response.text})
    return converted_code, memory_list
  else:
    logger.error("Failed to get a valid response from the LLM agent.")
    return "", memory_list


def parse_args():
  """
  Parses command-line arguments for file or folder processing.

  Returns:
    argparse.Namespace: The parsed command-line arguments.
  """
  parser = argparse.ArgumentParser(description="Code Conversion and Test Case Generation Agent")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--file", type=str, help="Path to the Python file for conversion or test case generation.")
  group.add_argument("--folder", type=str, help="Path to the folder containing Python files for batch processing.")
  return parser.parse_args()


def process_single_file(input_path):
  """
  Reads a single file, converts its code, and writes the output.

  Args:
    input_path (Path): The path to the input Python file.
  """
  try:
    with open(input_path, "rt", encoding="utf-8") as f:
      code = f.read()

    converted_code, _ = convert_code_from_torch_to_jax(code, [])

    output_dir = Path(JAX_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / input_path.name

    with open(output_file, "wt", encoding="utf-8") as f:
      f.write(converted_code)

    logger.info("Converted code written to %s", output_file)
  except FileNotFoundError:
    logger.error("File not found at %s", input_path)
  except IOError as e:
    logger.error("An error occurred while processing %s: %s", input_path, e)


def process_files(args):
  """
  Handles file processing based on command-line arguments (single file or folder).

  Args:
    args (argparse.Namespace): The command-line arguments.
  """
  if args.file:
    input_path = Path(args.file)
    if not input_path.exists():
      logger.error("File %s does not exist.", input_path)
      return
    process_single_file(input_path)
  elif args.folder:
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
      logger.error("Folder %s does not exist or is not a directory.", folder_path)
      return
    for file_path in folder_path.glob("*.py"):
      logger.info("Processing File: %s", file_path)
      process_single_file(file_path)


if __name__ == "__main__":
  process_files(parse_args())
