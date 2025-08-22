# Code Evaluation Agent

This agent automates the evaluation of JAX code that has been converted from PyTorch. It works by generating and executing `pytest` test cases to compare the functional equivalence of the original PyTorch code and the converted JAX code. The agent leverages a large language model (Gemini) to create these test cases dynamically.

## Workflow

1.  **File Pairing**: The agent identifies pairs of corresponding PyTorch and JAX files from specified input directories.
2.  **Test Case Generation**: For each file pair, it prompts the Gemini model to generate a comprehensive `pytest` test case. The generated test compares the outputs of the PyTorch and JAX modules using randomized inputs to ensure they are numerically close (`numpy.allclose`).
3.  **Test Execution**: The generated test case is saved as a Python file and executed using `pytest`.
4.  **Result Aggregation**: The agent captures the results (pass/fail counts) from each test run.
5.  **Reporting**: Finally, it calculates and logs two key metrics:
    *   **Test Case Accuracy**: The percentage of individual test cases that passed across all files.
    *   **File Accuracy**: The percentage of files for which all generated test cases passed.

## File Descriptions

-   **`code_evaluation_agent.py`**: The main executable script that orchestrates the entire evaluation process.
-   **`prompt_code_evaluation.py`**: Contains the system and user prompt templates that instruct the Gemini model on how to generate the `pytest` test cases.
-   **`utils.py`**: Provides helper functions, including `run_pytest_capture_output` to execute `pytest` and capture its results, and `get_last_defined_module` to identify the primary component in a code file.

## Setup

1.  **Install Dependencies**:
    Make sure you have the required Python packages installed.
    ```bash
    pip install pytest google-generativeai backoff python-dotenv
    ```

2.  **Configure Environment Variables**:
    This agent uses the `GeminiAgent` from the `code_generation_agent`, which requires a `.env` file in the `code_generation_agent` directory.

    ```.env
    # in src/MaxText/experimental/agent/code_generation_agent/.env
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    Model="gemini-2.5-pro"
    ```

3.  **Configure Paths**:
    In `code_evaluation_agent.py`, set the following path variables to point to your datasets. The script will create the test case directory if it doesn't exist. You can modify the paths as needed.

    ```python
    # in code_evaluation_agent.py
    pytorch_path="../code_generation_agent/dataset/PyTorch/"
    jax_path="../code_generation_agent/dataset/jax_converted/"
    testcase_path="../code_generation_agent/dataset/test_cases/"
    ```

## Usage

Before running the agent, ensure you have:

1.  Your original PyTorch files in the directory specified by `pytorch_path`.
2.  The corresponding converted JAX files in the directory specified by `jax_path`. The filenames must match between the two directories.

To start the evaluation process, run the following command from within the `code_evaluation_agent` directory:

```bash
python code_evaluation_agent.py
```

The agent will process each file pair, generate tests, run them, and print the progress and final accuracy metrics to the console.

## Output

The agent provides real-time logging for each file being processed. At the end of the run, it prints a summary of the results, including:

- The number of files that passed all tests.
- The number of files that had at least one failing test.
- The overall **Test Case Accuracy**.
- The overall **File Accuracy**.
