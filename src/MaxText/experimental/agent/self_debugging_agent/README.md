# Self-Debugging Agent

## Overview

The Self-Debugging Agent is an experimental tool designed to automate analyze the bugs that arise during the code conversion or testing process. It leverages Gemini to facilitate code generation (calling code generation agent), test case creation (calling code evaluation agent), and debugging.

## Workflow

The agent processes each PyTorch file through a multi-step pipeline. It is designed to be stateful: if a JAX file or test case already exists, it will be used; otherwise, it will be generated.

1.  **Code Generation**: The agent takes a PyTorch file as input and uses an LLM to generate the corresponding JAX implementation. This step includes a retry mechanism to handle and fix basic syntax errors in the LLM's output.

2.  **Test Case Generation**: An LLM generates a `pytest`-compatible test case to verify the functional equivalence of the original PyTorch code and the newly generated JAX code. The test typically compares the outputs of both implementations given the same inputs.

3.  **Test Execution**: The generated test case is executed using `pytest`. The agent captures the output, including which tests passed and which failed, along with any stack traces from failures.

4.  **Self-Debugging**: If any tests fail, the agent enters a debugging loop. It feeds the original code, the failing JAX code, the test case, and the captured test output (stack trace) back to the LLM with a specific debugging prompt. The LLM then attempts to fix either the JAX code or the test case. This process is repeated for a configurable number of attempts or until all tests pass.

5.  **Result Logging**: The final results of the process are logged, including the number of passed and failed tests for each file. The agent also calculates overall accuracy metrics:
    *   **Test Case Accuracy**: The percentage of individual test cases that passed across all files.
    *   **File Accuracy**: The percentage of files for which all generated test cases passed.

## File Descriptions

-   **`self_debugging_agent.py`**: The main executable script that orchestrates the entire workflow, from code generation to self-debugging and result logging.
-   **`prompt_debugging.py`**: Contains the system and user prompt templates that instruct the LLM on how to debug the JAX code and test cases based on `pytest` output.
-   **`utils.py`**: Provides helper functions for saving code to files (`save_in_file_and_check_code_syntax`) and parsing JSON from LLM responses (`parse_json_response`).
-   **`code_evaluation_agent/prompt_code_evaluation.py`**: Contains prompts for generating the initial `pytest` test cases.
-   **`code_evaluation_agent/utils.py`**: Provides utilities for executing `pytest` and capturing its results (`run_pytest_capture_output`) and for identifying the primary component in a code file (`get_last_defined_module`).
-   **`code_generation_agent/llm_agent.py`**: Contains the `GeminiAgent` class for interacting with the Google Gemini model.
-   **`orchestration_agent/Utils.py`**: Contains the `parse_python_code` utility for extracting code blocks from LLM responses.

## Usage

### Setup

1.  **Install Dependencies**: Ensure all required packages from the parent directories are available in your Python environment.
2.  **Configure Environment**: Make sure your `.env` file is configured with the necessary `GOOGLE_API_KEY` and `Model` for the `GeminiAgent`.
3.  **Prepare Data**:
    *   Place your source PyTorch files in the directory specified by the `--pytorch_path` argument (default: `../code_generation_agent/dataset/PyTorch/`).
    *   The agent will create output directories for JAX code and test cases if they don't exist.

### Running the Agent

To start the self-debugging process, run the main script from the `self_debugging_agent` directory.

To process all files in the source directory:
```bash
python self_debugging_agent.py
```
To process a single file:
```bash
python self_debugging_agent.py --file_name <file_name>.py

```

## Configuration

You can customize the agent's behavior using command-line arguments:

-   `--pytorch_path`: Path to the directory containing source PyTorch files. (Default: `../code_generation_agent/dataset/PyTorch/`)
-   `--jax_path`: Path to the directory where converted JAX files will be saved. (Default: `./dataset/jax_converted/`)
-   `--testcase_path`: Path to the directory where new debugged and generated test cases will be saved.  (Default: `./dataset/test_cases/`)
-   `--base_jax_path`: Base path for JAX files. (Default: `../code_generation_agent/dataset/jax_converted/`)
-   `--base_testcase_path`: Base path for old test cases. (Default: `../code_generation_agent/dataset/test_cases/`)

-   `--code_syntax_error_tries`: Number of retries for fixing syntax errors during initial code generation. (Default: 5)
-   `--code_debug_error_tries`: Number of retries for the self-debugging loop when tests fail. (Default: 5)
-   `--code_generation_tries`: Number of times to restart the entire process (generation + debugging) for a file if it fails. (Default: 2)
-   `--error_penalty`: A penalty value assigned when test cases cannot be generated or executed, affecting the failure count. (Default: 10)
