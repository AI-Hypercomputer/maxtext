"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
This file implements an agent that evaluates the correctness of JAX code
generated from PyTorch code by running pytest test cases. It uses a language
model to generate the test cases and captures the results of the tests.

The agent performs the following steps:
1. Reads pairs of PyTorch and JAX files from specified directories.
2. For each pair, it generates a pytest-compatible test case using a language
   model.
3. It runs the generated test case and captures the output, including the number
   of passed and failed tests.
4. It logs the results and calculates overall accuracy metrics.

Example Invocation:
python code_evaluation_agent.py

Ensure the paths to the PyTorch and JAX code directories are correctly set in
the script. The script will create a directory for test cases if it doesn't
exist and will overwrite existing test cases based on the `overwrite_existing_files`
flag.

Overall Accuracy Metrics:
- Test Case Accuracy: The percentage of individual test cases that passed across
  all generated tests.
- File Accuracy: The percentage of files for which all generated test cases passed.

Relevant Files:
- `prompt_code_evaluation.py`: Contains the prompts used by the language model
  for generating test cases.
- `utils.py`: Provides utility functions such as `get_last_defined_module`
  (to extract the main module from a code string) and `run_pytest_capture_output`
  (to execute pytest and capture its results).
- `code_generation_agent/llm_agent.py`: Contains the `GeminiAgent` class used
  to interact with the language model.
- `orchestration_agent/Utils.py`: Contains `parse_python_code` for extracting
  code from LLM responses.
"""
import argparse
import os, logging, sys
from prompt_code_evaluation import CodeEvaluation
from utils import get_last_defined_module, run_pytest_capture_output

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_generation_agent.llm_agent import GeminiAgent
from orchestration_agent.Utils import parse_python_code

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# logging.raiseExceptions = False


parser = argparse.ArgumentParser(description="Code Evaluation Agent")
parser.add_argument("--error_penalty", type=int, default=10, help="Penalty for errors in test case generation or execution.")
parser.add_argument("--pytorch_path", type=str, default="../code_generation_agent/dataset/PyTorch/", help="Path to the directory containing PyTorch files.")
parser.add_argument("--jax_path", type=str, default="../code_generation_agent/dataset/jax_converted/", help="Path to the directory containing JAX files.")
parser.add_argument("--testcase_path", type=str, default="../code_generation_agent/dataset/test_cases/", help="Path to the directory for generated test cases.")
parser.add_argument("--overwrite_existing_files", action="store_true", help="Overwrite existing test case files.")
args = parser.parse_args()

overwrite_existing_files = args.overwrite_existing_files
error_penalty = args.error_penalty
pytorch_path = args.pytorch_path
jax_path = args.jax_path
testcase_path = args.testcase_path
os.makedirs(testcase_path, exist_ok=True)

llm_agent = GeminiAgent(CodeEvaluation["SystemPrompt"])


def get_file_pairs(pytorch_path, jax_path):
  """Generates lists of file paths for PyTorch and JAX files that have a common name.

  This function finds files with the same name in the specified PyTorch and JAX
  directories, filtering out any files in the JAX directory that start with "__".

  Args:
      pytorch_path: The path to the directory containing PyTorch files.
      jax_path: The path to the directory containing JAX files.

  Returns:
      A tuple containing two lists of strings:
          - The first list contains the full paths to the common PyTorch files.
          - The second list contains the full paths to the common JAX files.
  """
  pytorch_files = os.listdir(pytorch_path)
  jax_files = list(filter(lambda x: not x.startswith("__"), os.listdir(jax_path)))
  common_files = list(set(pytorch_files).intersection(jax_files))
  return list(map(lambda x: pytorch_path + x, common_files)), list(map(lambda x: jax_path + x, common_files))


def make_test_case_and_run(python_file, jax_file):
  """Generates a test case and runs it for a given PyTorch and JAX file pair.

  This function uses a language model to generate a pytest-compatible test case
  for a PyTorch and JAX code file pair. It then runs the test and captures the output.
  If the files have inconsistent entry points or the test case cannot be generated,
  a penalty is applied.

  Args:
      python_file: The path to the PyTorch code file.
      jax_file: The path to the JAX code file.

  Returns:
      A tuple containing the number of passed and failed test cases.
  """
  try:
    logger.info(f"Processing {python_file}")
    out_file_path = os.path.join(testcase_path, python_file.split("/")[-1])
    if overwrite_existing_files or not os.path.exists(out_file_path):
      with open(python_file) as f:
        python_code = f.read()
      with open(jax_file) as f:
        jax_code = f.read()
      entry_module = get_last_defined_module(python_code)
      if get_last_defined_module(jax_code) != entry_module:
        logger.error(
            f"It seems inconsistency in {python_file} code PyTorch have {entry_module} and JAX have {get_last_defined_module(jax_code)} as entry Module"
        )
        # Penalty in case of Entry point not exist or different from torch
        return 0, error_penalty
      prompt = CodeEvaluation["TESTCASE"]
      python_code = (
          "from " + ".".join(python_file.split("/")[1:]).replace(".py", " import " + entry_module) + "\n\n" + python_code
      )
      jax_code = "from " + ".".join(jax_file.split("/")[1:]).replace(".py", " import " + entry_module) + "\n\n" + jax_code
      prompt = prompt.replace("<module.path.to.pytorch_code>", python_code)
      prompt = prompt.replace("<module.path.to.jax_code>", jax_code)
      prompt = prompt.replace("<function_or_class_to_call>", entry_module)
      response = llm_agent(prompt)
      generated_code = parse_python_code(response.text)
      with open(out_file_path, "w") as f:
        f.write("import os,sys\nsys.path.append(os.path.abspath('..'))\n")
        f.write(generated_code)
      logger.info("Written at %s", out_file_path)
      if "<UNABLETOGENERATE>" in response:
        return 0, error_penalty
    else:
      logger.info("File Exists using same")
    file = python_file.split("/")[-1]
    output, exit_code, is_dependency_error, passed, failed = run_pytest_capture_output(file, code_folder=testcase_path)
    return passed, failed
  except Exception as e:
    logger.error("Exception in code generation %s", e)
    logger.error("The code file is %s", python_file.split("/")[-1])
    logger.error("The generated Code is %s", response)
    # Penalty in case of Exception
    return 0, error_penalty


def run_code_evaluation():
  """Runs the full code evaluation process.

  This function orchestrates the evaluation of PyTorch and JAX code file pairs.
  It iterates through the common files, generates and runs a test case for each,
  and then logs the results. It also calculates and prints the overall
  test case and file accuracy.
  """
  total_passed, total_failed = 0, 0
  all_passed, all_failed, total_files = 0, 0, 0
  for python_file, jax_file in zip(*get_file_pairs(pytorch_path, jax_path)):
    num_passed, num_failed = make_test_case_and_run(python_file, jax_file)
    if num_passed == num_failed == 0: # when the code cannot be executed
      # Penalty in case of issue in test case and not executed
      num_failed = error_penalty
    logger.info(f"{python_file.split('/')[-1]} have {num_passed} cases passed and {num_failed} cases failed")
    total_passed += num_passed
    total_failed += num_failed
    if num_passed == 0:
      all_failed += 1
    if num_failed == 0:
      all_passed += 1
    total_files += 1

  logger.info("****** Results ******")
  logger.info(f"{all_passed} files have all module passed {all_failed} files have all module failed")
  logger.info(
      f"Test case Accuracy {total_passed*100/(total_passed+total_failed):.2f}%",
  )
  logger.info(
      f"File Accuracy {all_passed * 100 / total_files:.2f}%",
  )


if __name__ == "__main__":
  run_code_evaluation()
